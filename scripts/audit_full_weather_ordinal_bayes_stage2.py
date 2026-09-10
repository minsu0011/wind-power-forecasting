"""Read-only local audit of the rejected ordinal-Bayes G3 Stage2."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_ficr_bayes_decision_strict as strict  # noqa: E402
from scripts import run_full_weather_ordinal_bayes_stage2_final as producer  # noqa: E402
from src.full_weather_ordinal_bayes import BIN_COUNT, FullWeatherOrdinalBayes  # noqa: E402
from src.manifest import describe_file, sha256_file, utc_now  # noqa: E402
from src.metric import CAPACITY_KWH, TARGET_COLS  # noqa: E402


CANONICAL = ROOT / "artifacts/postgate/full_weather_ordinal_bayes_strict_v1"
REPORT = ROOT / "artifacts/audits/full_weather_ordinal_bayes_g3_stage2_v1_local.json"
APPEND_MANIFEST_SHA256 = "a443c2ab714053d0a2f23b6f7de5d21e1498ae6161c61341df93835b7c2325e1"


def _verify(record: Mapping[str, Any]) -> Path:
    path = Path(record["path"])
    if path.stat().st_size != int(record["size_bytes"]) or sha256_file(path) != record["sha256"]:
        raise AssertionError(f"record changed: {path}")
    return path


def _bits_equal(left: pd.Series, right: pd.Series) -> bool:
    return left.to_numpy().tobytes() == right.to_numpy().tobytes()


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(REPORT)
    stage1_manifest_path = CANONICAL / "manifest.json"
    append_manifest_path = CANONICAL / "manifest_stage2_final_g3.json"
    if sha256_file(stage1_manifest_path) != producer.STAGE1_MANIFEST_SHA256:
        raise AssertionError("Stage1 manifest changed")
    if sha256_file(append_manifest_path) != APPEND_MANIFEST_SHA256:
        raise AssertionError("Stage2 manifest changed")
    append = json.loads(append_manifest_path.read_text(encoding="utf-8"))
    if append["status"] != "stage2_reject_no_final_csv":
        raise AssertionError("append status changed")
    for record in append["source_and_config_snapshot"].values():
        _verify(record)
    for stage_records in append["input_snapshot"].values():
        for record in stage_records.values():
            _verify(record)
    for record in append["outputs"]:
        _verify(record)
    if append["input_snapshot"]["final_after_stage2_promotion"] != {}:
        raise AssertionError("future inputs were snapshotted despite rejection")

    primary_base = pd.read_parquet(CANONICAL / "stage2/primary_baseline_2024.parquet")
    recent_base = pd.read_parquet(CANONICAL / "stage2/recent_baseline_2024.parquet")
    primary_candidate = pd.read_parquet(CANONICAL / "stage2/primary_candidate_2024.parquet")
    recent_candidate = pd.read_parquet(CANONICAL / "stage2/recent_same_delta_candidate_2024.parquet")
    delta = pd.read_parquet(CANONICAL / "stage2/primary_delta_cf_2024.parquet").iloc[:, 0]
    action = pd.read_parquet(CANONICAL / "stage2/raw_action_cf_g3.parquet").iloc[:, 0]
    expected_primary, expected_recent, expected_delta = producer._same_delta_candidate(
        primary_base, recent_base, action
    )
    if not np.array_equal(primary_candidate.to_numpy(), expected_primary.to_numpy()):
        raise AssertionError("primary candidate formula differs")
    if not np.array_equal(recent_candidate.to_numpy(), expected_recent.to_numpy()):
        raise AssertionError("recent same-delta formula differs")
    if not np.array_equal(delta.to_numpy(), expected_delta.to_numpy()):
        raise AssertionError("stored primary delta differs")
    identities: dict[str, bool] = {}
    for baseline_name, baseline, candidate in (
        ("primary", primary_base, primary_candidate),
        ("recent", recent_base, recent_candidate),
    ):
        for group in TARGET_COLS[:2]:
            identities[f"{baseline_name}/{group}"] = _bits_equal(
                baseline[group], candidate[group]
            )
    if not all(identities.values()):
        raise AssertionError("G1/G2 identity differs")

    probability = pd.read_parquet(CANONICAL / "stage2/probability42_g3.parquet")
    if probability.shape != (8784, BIN_COUNT) or tuple(probability.columns) != tuple(
        f"class_{value:02d}" for value in range(BIN_COUNT)
    ):
        raise AssertionError("fixed probability shape/schema differs")
    row_sum_error = float(np.max(np.abs(probability.sum(axis=1).to_numpy() - 1.0)))
    if row_sum_error > 1e-12 or not np.isfinite(probability.to_numpy()).all():
        raise AssertionError("probability validity differs")
    model: FullWeatherOrdinalBayes = joblib.load(CANONICAL / "stage2/model_g3.joblib")
    unseen = [value for value in range(BIN_COUNT) if value not in model.observed_classes_]
    if unseen and not np.all(
        probability[[f"class_{value:02d}" for value in unseen]].to_numpy() == 0.0
    ):
        raise AssertionError("unseen probability columns differ")

    result_path = CANONICAL / "stage2_results_g3.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    lock = json.loads((CANONICAL / "stage2_promotion_lock_g3.json").read_text(encoding="utf-8"))
    if lock["stage2_results"]["sha256"] != sha256_file(result_path):
        raise AssertionError("Stage2 result lock differs")
    labels = strict._read_full_labels(
        Path(r"data/local/open/train/train_labels.csv")
    )
    passed, recomputed = producer._dual_gate(
        labels,
        {"primary_v3": primary_base, "recent_v4": recent_base},
        {"primary_v3": primary_candidate, "recent_v4": recent_candidate},
        2024,
    )
    if passed or result["candidate_promoted"] or lock["candidate_promoted"]:
        raise AssertionError("rejected candidate promotion changed")
    if strict._canonical_sha256(recomputed) != result["comparisons_sha256"]:
        raise AssertionError("independent metric recomputation differs")

    if (CANONICAL / "final").exists() or list(CANONICAL.rglob("*.csv")):
        raise AssertionError("rejected candidate created final/CSV")
    if producer.stage1.HEAVY_GUARD_PATH.exists():
        raise AssertionError("heavy guard remains")
    primary_gate = recomputed["primary_v3"]["gate"]
    recent_gate = recomputed["recent_v4"]["gate"]
    report = {
        "schema_version": 1,
        "audit_id": "full_weather_ordinal_bayes_g3_stage2_v1_local",
        "created_utc": utc_now(),
        "status": "PASS_REPRODUCED_STAGE2_REJECT",
        "canonical": str(CANONICAL.resolve()),
        "frozen_stage1_manifest": describe_file(stage1_manifest_path),
        "append_stage2_manifest": describe_file(append_manifest_path),
        "preregister_sha256": producer.PREREGISTER_SHA256,
        "checks": {
            "source_and_input_current": True,
            "append_outputs_current": True,
            "stage1_manifest_unchanged": True,
            "primary_formula_exact": True,
            "recent_exact_primary_delta": True,
            "g1_g2_bit_identity": identities,
            "probability_shape": list(probability.shape),
            "probability_row_sum_max_abs_error": row_sum_error,
            "unseen_probability_columns_exact_zero": True,
            "metric_recompute_exact": True,
            "primary_gate_pass": primary_gate["passed"],
            "recent_gate_pass": recent_gate["passed"],
            "stage2_reject_exact": True,
            "final_2025_sample_csv_read_or_created": False,
            "heavy_guard_absent": True,
        },
        "failure": {
            "recent_negative_group_slices": {
                name: record["delta"]
                for name, record in recomputed["recent_v4"]["group_g3"].items()
                if float(record["delta"]) <= 0.0
            },
            "recent_negative_mixed_slices": {
                name: record["delta"]
                for name, record in recomputed["recent_v4"]["mixed"].items()
                if float(record["delta"]) <= 0.0
            },
            "recent_group_full_delta_ficr": recent_gate["group_full_delta_ficr"],
            "recent_mixed_full_delta_ficr": recent_gate["mixed_full_delta_ficr"],
        },
        "no_retune_rescue_or_csv": True,
    }
    strict._write_json(REPORT, report)
    if sha256_file(stage1_manifest_path) != producer.STAGE1_MANIFEST_SHA256 or sha256_file(append_manifest_path) != APPEND_MANIFEST_SHA256:
        raise AssertionError("canonical manifests changed during audit")
    print(f"local audit PASS {sha256_file(REPORT)}", flush=True)


if __name__ == "__main__":
    main()

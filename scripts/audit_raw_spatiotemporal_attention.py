"""Independent reconstruction audit for raw spatiotemporal attention Stage1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts import run_raw_grid_wind_lgb as raw_protocol
from scripts import run_raw_spatiotemporal_attention as runner
from src.manifest import describe_file, sha256_file, utc_now, write_json_atomic
from src.metric import CAPACITY_KWH, TARGET_COLS, group_metrics
from src.raw_spatiotemporal_attention import (
    BLEND_WEIGHTS,
    CANDIDATE_KEYS,
    RawSpatiotemporalAttentionRegressor,
    candidate_frame,
    select_group_candidate,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir", type=Path, default=Path(r"data/local/open")
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/postgate/raw_spatiotemporal_attention_strict_v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/audits/raw_spatiotemporal_attention_strict_v1_cross_audit.json"),
    )
    return parser.parse_args(argv)


def _tree_snapshot(root: Path) -> dict[str, dict[str, Any]]:
    return {
        path.relative_to(root).as_posix(): {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _summary(actual: pd.Series, prediction: pd.Series, group: str) -> dict[str, Any]:
    details = group_metrics(actual, prediction, CAPACITY_KWH[group], group_name=group)
    result = details.as_dict()
    result["score"] = 0.5 * (details.one_minus_nmae + details.ficr)
    return result


def _comparison(
    actual: pd.Series,
    baseline: pd.Series,
    candidate: pd.Series,
    group: str,
    segments: Mapping[str, pd.DatetimeIndex],
) -> dict[str, Any]:
    return {
        name: {
            "baseline": _summary(actual.loc[index], baseline.loc[index], group),
            "candidate": _summary(actual.loc[index], candidate.loc[index], group),
            "delta": _summary(actual.loc[index], candidate.loc[index], group)["score"]
            - _summary(actual.loc[index], baseline.loc[index], group)["score"],
        }
        for name, index in segments.items()
    }


def _json_equal(left: Any, right: Any) -> bool:
    return json.dumps(left, sort_keys=True, default=str) == json.dumps(
        right, sort_keys=True, default=str
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    raw_dir = args.raw_dir.resolve()
    artifact_dir = args.artifact_dir.resolve()
    output = args.output.resolve()
    before = _tree_snapshot(artifact_dir)
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    result = json.loads((artifact_dir / "stage1_results.json").read_text(encoding="utf-8"))
    lock = json.loads((artifact_dir / "stage1_promotion_lock.json").read_text(encoding="utf-8"))

    output_checks: list[dict[str, Any]] = []
    for record in manifest["outputs"]:
        path = Path(record["path"])
        exact = path.stat().st_size == int(record["size_bytes"]) and sha256_file(path) == record["sha256"]
        if not exact:
            raise AssertionError(f"manifest output changed: {path}")
        output_checks.append({"path": str(path), "exact": True})
    source_checks: list[dict[str, Any]] = []
    for name, record in manifest["source_provenance"].items():
        path = Path(record["path"])
        exact = path.stat().st_size == int(record["size_bytes"]) and sha256_file(path) == record["sha256"]
        if not exact:
            raise AssertionError(f"source provenance changed: {name}")
        source_checks.append({"name": name, "exact": True})

    preregister_path = PROJECT_DIR / "configs/raw_spatiotemporal_attention_preregister_v1.json"
    _, raw_contract, dependency = runner._verify_preregister(preregister_path)
    if manifest["preregister_sha256"] != runner.PREREGISTER_SHA256:
        raise AssertionError("manifest preregister binding changed")
    if lock["stage1_results"]["sha256"] != sha256_file(artifact_dir / "stage1_results.json"):
        raise AssertionError("promotion lock result binding changed")
    if lock["passed_groups"] != result["passed_groups"] or result["passed_groups"]:
        raise AssertionError("Stage1 identity decision changed")

    prefix = raw_contract["physical_stage1_inputs"]
    prefix_checks = {
        "labels_g12_fit": raw_protocol._prefix_snapshot(
            raw_dir / "train/train_labels.csv", prefix["labels_g12_fit_prefix"]
        ),
        "labels_g3_fit": raw_protocol._prefix_snapshot(
            raw_dir / "train/train_labels.csv", prefix["labels_g3_fit_prefix"]
        ),
        "labels_score": raw_protocol._prefix_snapshot(
            raw_dir / "train/train_labels.csv", prefix["labels_stage1_score_prefix_after_lock"]
        ),
        "ldaps": raw_protocol._prefix_snapshot(
            raw_dir / "train/ldaps_train.csv", prefix["ldaps_weather_prefix"]
        ),
        "gfs": raw_protocol._prefix_snapshot(
            raw_dir / "train/gfs_train.csv", prefix["gfs_weather_prefix"]
        ),
    }
    stored_prefixes = json.loads(
        (artifact_dir / "stage1_global_prescore_record.json").read_text(encoding="utf-8")
    )["input_prefixes_before"]
    if not _json_equal(prefix_checks, stored_prefixes):
        raise AssertionError("physical prefix identity changed")

    features, weather_evidence = raw_protocol._read_weather_features(
        raw_dir, raw_contract, period="stage1"
    )
    replay: dict[str, Any] = {}
    applications = {
        "g12_2022": runner.YEAR_2023,
        "g3_through_2023h1": runner.G3_H2,
    }
    predictions: dict[str, pd.DataFrame] = {}
    for fit_id, index in applications.items():
        model: RawSpatiotemporalAttentionRegressor = joblib.load(
            artifact_dir / "models" / f"stage1__{fit_id}.joblib"
        )
        repeated = model.predict(features.loc[index])
        stored = pd.read_parquet(
            artifact_dir / "predictions" / f"stage1__{fit_id}__raw_cf.parquet"
        )
        if not repeated.index.equals(stored.index) or not np.array_equal(
            repeated.to_numpy(), stored.to_numpy()
        ):
            raise AssertionError(f"model replay changed: {fit_id}")
        predictions[fit_id] = stored
        replay[fit_id] = {
            "bit_exact": True,
            "rows": len(stored),
            "metadata": model.metadata(),
        }

    labels, label_evidence = raw_protocol._read_label_prefix(
        raw_dir, prefix["labels_stage1_score_prefix_after_lock"], TARGET_COLS
    )
    comparisons_recomputed = 0
    formulas_recomputed = 0
    selected_recomputed: dict[str, str | None] = {}
    for group in TARGET_COLS:
        fit_id = "g12_2022" if group in TARGET_COLS[:2] else "g3_through_2023h1"
        index = runner.YEAR_2023 if group in TARGET_COLS[:2] else runner.G3_H2
        raw = predictions[fit_id][group]
        baseline = pd.read_parquet(
            artifact_dir / "predictions" / f"stage1__{fit_id}__{group}__baseline.parquet"
        )[group]
        calculated = candidate_frame(baseline, raw, capacity_kwh=CAPACITY_KWH[group])
        stored_candidates = pd.read_parquet(
            artifact_dir / "predictions" / f"stage1__{fit_id}__{group}__candidates.parquet"
        )
        if not np.array_equal(calculated.to_numpy(), stored_candidates.to_numpy()):
            raise AssertionError(f"candidate formula changed: {group}")
        formulas_recomputed += len(BLEND_WEIGHTS)
        comparisons: dict[str, Any] = {}
        for key in CANDIDATE_KEYS:
            recomputed = _comparison(
                labels.loc[index, group], baseline, calculated[key], group, runner._segments(group)
            )
            if not _json_equal(recomputed, result["group_results"][group]["comparisons"][key]):
                raise AssertionError(f"metric reconstruction changed: {group}/{key}")
            comparisons[key] = recomputed
            comparisons_recomputed += len(recomputed)
        selected, audit = select_group_candidate(comparisons, runner.STAGE1_REQUIRED[group])
        if selected is not None or not _json_equal(
            audit, result["group_results"][group]["selection"]
        ):
            raise AssertionError(f"selection reconstruction changed: {group}")
        selected_recomputed[group] = selected

    if any(artifact_dir.rglob("*.csv")):
        raise AssertionError("canonical rejected artifact contains CSV")
    if manifest["read_flags"] != {
        "2024_weather_read": False,
        "2024_label_read": False,
        "2025_weather_read": False,
        "sample_submission_read": False,
        "csv_created": False,
    }:
        raise AssertionError("future read flags changed")
    after = _tree_snapshot(artifact_dir)
    if before != after:
        raise AssertionError("audit mutated canonical artifact")

    payload = {
        "schema_version": 1,
        "audit_type": "raw_spatiotemporal_attention_cross_reconstruction_v1",
        "created_utc": utc_now(),
        "status": "pass",
        "canonical_manifest": describe_file(artifact_dir / "manifest.json"),
        "preregister_sha256": runner.PREREGISTER_SHA256,
        "manifest_outputs_exact": len(output_checks),
        "source_provenance_exact": len(source_checks),
        "physical_prefixes_exact": True,
        "physical_weather_rebuild": weather_evidence,
        "model_replay": replay,
        "candidate_formulas_recomputed": formulas_recomputed,
        "slice_metrics_recomputed": comparisons_recomputed,
        "selection_recomputed": selected_recomputed,
        "future_read_flags_all_false": True,
        "csv_count": 0,
        "canonical_nonmutation": True,
        "score_label_evidence": label_evidence,
        "dependency": describe_file(dependency),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(json.dumps(payload, default=str))
    write_json_atomic(output, payload)
    print(json.dumps({"status": "pass", "output": str(output), "sha256": sha256_file(output)}))


if __name__ == "__main__":
    main()

"""Score one explicitly selection-unsafe G1/G2-only rescue after a hash lock."""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metric import CAPACITY_KWH, TARGET_COLS, group_metrics, score_details


EXPERIMENT_ID = "multi_nwp_joint_g12_posthoc_rescue_v1"
CONFIG_PATH = ROOT / "configs/multi_nwp_joint_g12_posthoc_rescue_preregister_v1.json"
SIDECAR_PATH = CONFIG_PATH.with_suffix(".sha256")
OUTPUT_DIR = ROOT / f"artifacts/postgate/{EXPERIMENT_ID}"
INCREMENT_PATH = ROOT / "artifacts/postgate/multi_nwp_joint_disagreement_2024_forward_v1/predictions/joint_paired_increment_cf.parquet"
ORIGINAL_LOCK_PATH = ROOT / "artifacts/postgate/multi_nwp_joint_disagreement_2024_forward_v1/candidate_lock_before_h2_label_access.json"
ORIGINAL_RESULTS_PATH = ROOT / "artifacts/postgate/multi_nwp_joint_disagreement_2024_forward_v1/stage_results.json"
BASELINE_PATH = ROOT / "artifacts/oof/gate2024_recent_v4_cf_fix_calibration_fit.parquet"
LABEL_PATH = Path(r"data/local/open/train/train_labels.csv")
FULL_LABEL_ROWS = 26_304
SCORE_INDEX = pd.date_range("2024-07-01 00:00", "2025-01-01 00:00", freq="h", name="forecast_kst_dtm")
SLICES = {
    "H2": (pd.Timestamp("2024-07-01 00:00"), pd.Timestamp("2025-01-01 00:00")),
    "Q3": (pd.Timestamp("2024-07-01 00:00"), pd.Timestamp("2024-09-30 23:00")),
    "Q4": (pd.Timestamp("2024-10-01 00:00"), pd.Timestamp("2025-01-01 00:00")),
    "Jul": (pd.Timestamp("2024-07-01 00:00"), pd.Timestamp("2024-07-31 23:00")),
    "Aug": (pd.Timestamp("2024-08-01 00:00"), pd.Timestamp("2024-08-31 23:00")),
    "Sep": (pd.Timestamp("2024-09-01 00:00"), pd.Timestamp("2024-09-30 23:00")),
    "Oct": (pd.Timestamp("2024-10-01 00:00"), pd.Timestamp("2024-10-31 23:00")),
    "Nov": (pd.Timestamp("2024-11-01 00:00"), pd.Timestamp("2024-11-30 23:00")),
    "Dec": (pd.Timestamp("2024-12-01 00:00"), pd.Timestamp("2025-01-01 00:00")),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    frame.to_parquet(temporary, engine="pyarrow", compression="zstd", index=True)
    os.replace(temporary, path)


def load_config() -> dict[str, Any]:
    expected = SIDECAR_PATH.read_text(encoding="ascii").split()[0]
    if sha256(CONFIG_PATH) != expected:
        raise RuntimeError("rescue preregistration sidecar mismatch")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["status"] != "frozen_before_rescue_candidate_construction_or_additional_metric":
        raise RuntimeError("rescue preregistration is not frozen")
    for relative, wanted in config["bound_files"].items():
        path = ROOT / relative
        if sha256(path) != wanted["sha256"] or path.stat().st_size != wanted["bytes"]:
            raise RuntimeError(f"bound rescue input changed: {relative}")
    return config


def bounded_labels() -> tuple[pd.DataFrame, dict[str, Any]]:
    chunks: list[bytes] = []
    digest = hashlib.sha256()
    with LABEL_PATH.open("rb", buffering=0) as stream:
        for line_number in range(FULL_LABEL_ROWS + 1):
            line = stream.readline()
            if not line:
                raise ValueError(f"label file ended at line {line_number}")
            chunks.append(line)
            digest.update(line)
    payload = b"".join(chunks)
    frame = pd.read_csv(io.BytesIO(payload), encoding="utf-8-sig")
    if tuple(frame.columns) != ("kst_dtm", *TARGET_COLS) or len(frame) != FULL_LABEL_ROWS:
        raise ValueError("label prefix changed")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("kst_dtm"), errors="raise"), name="forecast_kst_dtm")
    return frame.astype(np.float64).loc[SCORE_INDEX], {
        "path": str(LABEL_PATH), "data_rows": FULL_LABEL_ROWS, "bytes": len(payload), "sha256": digest.hexdigest()
    }


def group_payload(actual: pd.Series, predicted: pd.Series, group: str) -> dict[str, Any]:
    payload = group_metrics(actual, predicted, CAPACITY_KWH[group], group_name=group).as_dict()
    payload["total_score"] = 0.5 * (payload["one_minus_nmae"] + payload["ficr"])
    return payload


def comparisons(labels: pd.DataFrame, baseline: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {"groups": {}, "mixed": {}}
    for group in TARGET_COLS:
        output["groups"][group] = {}
        for slice_name, (start, end) in SLICES.items():
            index = labels.index[(labels.index >= start) & (labels.index <= end)]
            base = group_payload(labels.loc[index, group], baseline.loc[index, group], group)
            cand = group_payload(labels.loc[index, group], candidate.loc[index, group], group)
            output["groups"][group][slice_name] = {
                "baseline": base, "candidate": cand,
                "delta": {key: cand[key] - base[key] for key in ("total_score", "one_minus_nmae", "ficr")},
            }
    for slice_name, (start, end) in SLICES.items():
        index = labels.index[(labels.index >= start) & (labels.index <= end)]
        base = score_details(labels.loc[index], baseline.loc[index]).as_dict()
        cand = score_details(labels.loc[index], candidate.loc[index]).as_dict()
        output["mixed"][slice_name] = {
            "baseline": base, "candidate": cand,
            "delta": {key: cand[key] - base[key] for key in ("total_score", "one_minus_nmae", "ficr")},
        }
    return output


def frozen_gate(comparison: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    rules = config["promotion_gate"]
    h2 = comparison["mixed"]["H2"]["delta"]
    group_h2 = {group: comparison["groups"][group]["H2"]["delta"]["total_score"] for group in ("kpx_group_1", "kpx_group_2")}
    quarters = {name: comparison["mixed"][name]["delta"]["total_score"] for name in ("Q3", "Q4")}
    mixed = {name: comparison["mixed"][name]["delta"]["total_score"] for name in SLICES}
    checks = {
        "mixed_h2_total_positive": h2["total_score"] > rules["mixed_h2_min_total_score_delta"],
        "mixed_h2_ficr_positive": h2["ficr"] > rules["mixed_h2_min_ficr_delta"],
        "g1_g2_h2_total_positive": min(group_h2.values()) > rules["minimum_g1_g2_h2_total_score_delta"],
        "q3_q4_mixed_floor": min(quarters.values()) > rules["minimum_q3_q4_mixed_total_score_delta"],
        "mixed_slice_breadth": sum(value > 0.0 for value in mixed.values()) >= rules["minimum_positive_mixed_slices"],
    }
    return {
        "rules": dict(rules), "checks": checks, "pass": bool(all(checks.values())),
        "h2_mixed_delta": h2, "g1_g2_h2_total_score_deltas": group_h2,
        "q3_q4_mixed_total_score_deltas": quarters,
        "mixed_slice_total_score_deltas": mixed,
        "positive_mixed_slices": int(sum(value > 0.0 for value in mixed.values())),
    }


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)
    config = load_config()
    increments = pd.read_parquet(INCREMENT_PATH)
    increments.index = pd.DatetimeIndex(increments.index, name="forecast_kst_dtm")
    increments = increments.loc[SCORE_INDEX, list(TARGET_COLS)].astype(np.float64)
    baseline = pd.read_parquet(BASELINE_PATH)
    baseline.index = pd.DatetimeIndex(baseline.index, name="forecast_kst_dtm")
    baseline = baseline.loc[SCORE_INDEX, list(TARGET_COLS)].astype(np.float64)
    for group in TARGET_COLS:
        baseline[group] = np.clip(0.97 * baseline[group], 0.0, 1.02 * CAPACITY_KWH[group])
    candidate = baseline.copy()
    for group in ("kpx_group_1", "kpx_group_2"):
        candidate[group] = np.clip(
            baseline[group] + 0.25 * CAPACITY_KWH[group] * increments[group],
            0.0, 1.02 * CAPACITY_KWH[group],
        )
    if not np.array_equal(candidate["kpx_group_3"].to_numpy(), baseline["kpx_group_3"].to_numpy()):
        raise AssertionError("G3 rescue must be exact baseline identity")
    candidate_path = OUTPUT_DIR / "predictions/joint_g12_candidate_scale097.parquet"
    atomic_parquet(candidate, candidate_path)
    lock_path = OUTPUT_DIR / "rescue_candidate_lock_before_additional_label_read_or_metric.json"
    atomic_json({
        "schema_version": 1,
        "status": "posthoc_selection_unsafe_single_rescue_locked_before_metric",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "preregister": record(CONFIG_PATH),
        "original_increment": record(INCREMENT_PATH),
        "original_candidate_lock": record(ORIGINAL_LOCK_PATH),
        "candidate": record(candidate_path),
        "formula": "G1/G2=clip(scale097_recent_v4+0.25*C*original_joint_increment,0,1.02C); G3=scale097_recent_v4 exact identity",
        "feedback_files_read": 0,
        "2025_external_requests_response_bytes_or_values": 0,
    }, lock_path)

    labels, label_record = bounded_labels()
    label_access_path = OUTPUT_DIR / "score_label_access_after_rescue_candidate_lock.json"
    atomic_json({"status": "labels_opened_after_rescue_candidate_hash_lock", "candidate_lock": record(lock_path), "bounded_labels": label_record}, label_access_path)
    comparison = comparisons(labels, baseline, candidate)
    gate = frozen_gate(comparison, config)
    status = "promoted_for_separate_through2024_fit_and_2025_inference" if gate["pass"] else "rejected_no_further_rescue"
    results_path = OUTPUT_DIR / "stage_results.json"
    atomic_json({
        "schema_version": 1, "experiment_id": EXPERIMENT_ID, "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status, "selection_status": "posthoc_selection_unsafe",
        "comparison": comparison, "gate": gate,
        "physical_access": {"maximum_external_year": 2024, "year_2025_external_value_cells": 0, "submission_csv_created": False},
    }, results_path)
    decision_path = OUTPUT_DIR / ("promotion_lock.json" if gate["pass"] else "rejection.json")
    atomic_json({
        "status": status, "selection_status": "posthoc_selection_unsafe", "preregister_sha256": sha256(CONFIG_PATH),
        "candidate_lock_sha256": sha256(lock_path), "stage_results_sha256": sha256(results_path),
        "separate_2025_stage_authorized": bool(gate["pass"]), "no_retune_or_further_rescue": True,
    }, decision_path)
    outputs = [candidate_path, lock_path, label_access_path, results_path, decision_path]
    manifest_path = OUTPUT_DIR / "manifest.json"
    atomic_json({
        "schema_version": 1, "experiment_id": EXPERIMENT_ID, "selection_status": "posthoc_selection_unsafe",
        "created_utc": datetime.now(timezone.utc).isoformat(), "preregister": record(CONFIG_PATH),
        "outputs_excluding_manifest_and_sidecar": [record(path) for path in outputs],
        "unlisted_files_before_manifest": sorted(str(path.relative_to(OUTPUT_DIR)) for path in OUTPUT_DIR.rglob("*") if path.is_file() and path not in outputs),
        "physical_access": {"maximum_external_year": 2024, "year_2025_external_value_cells": 0, "submission_csv_created": False},
    }, manifest_path)
    if json.loads(manifest_path.read_text(encoding="utf-8"))["unlisted_files_before_manifest"]:
        raise RuntimeError("unlisted rescue outputs")
    (OUTPUT_DIR / "manifest.sha256").write_text(f"{sha256(manifest_path)}  manifest.json\n", encoding="ascii")
    print(f"status={status} gate={gate}", flush=True)
    print(f"manifest={manifest_path} sha256={sha256(manifest_path)}", flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metric import CAPACITY_KWH, TARGET_COLS, group_metrics, score_details


CONFIG_PATH = ROOT / "configs/multi_nwp_consensus_g23_rescue_preregister_v1.json"
SIDECAR_PATH = CONFIG_PATH.with_suffix(".sha256")
LABEL_PATH = Path(r"data/local/open/train/train_labels.csv")
OUTPUT_DIR = ROOT / "artifacts/postgate/multi_nwp_consensus_g23_rescue_v1"
LABEL_PREFIX_DATA_ROWS = 26_304
SOURCE_COLUMN = "B_day1_hours01_13_else_day2_w025__{group}"
TRANSFER_WEIGHT = 0.15
BASE_SCALE = 0.97

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


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    frame.to_parquet(temporary, engine="pyarrow", compression="zstd", index=True)
    os.replace(temporary, path)


def load_frozen_config() -> tuple[dict[str, Any], str]:
    expected = SIDECAR_PATH.read_text(encoding="utf-8").split()[0].lower()
    observed = sha256(CONFIG_PATH)
    if observed != expected:
        raise RuntimeError(f"preregistration hash mismatch: {observed} != {expected}")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["status"] != "frozen_before_candidate_materialization_or_label_score":
        raise RuntimeError("preregistration is not frozen")
    candidate = config["candidate"]
    if (
        candidate["count"] != 1
        or float(candidate["transfer_weight"]) != TRANSFER_WEIGHT
        or candidate["source_column_template"] != SOURCE_COLUMN
        or not candidate["g1_identity_relative_to_deployment_baseline"]
    ):
        raise RuntimeError("frozen candidate differs from runner constants")
    return config, observed


def checked_frame(spec: Mapping[str, Any]) -> pd.DataFrame:
    relative_path = spec.get("path", spec.get("increment_path"))
    if relative_path is None:
        raise KeyError("input specification has neither path nor increment_path")
    path = ROOT / str(relative_path)
    record = file_record(path)
    if record["bytes"] != int(spec["bytes"]) or record["sha256"] != str(spec["sha256"]):
        raise RuntimeError(f"input lock mismatch: {path}")
    frame = pd.read_parquet(path)
    if not isinstance(frame.index, pd.DatetimeIndex) or not frame.index.is_unique:
        raise ValueError(f"invalid index: {path}")
    return frame


def build_candidate_before_labels(config: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    recent = checked_frame(config["deployment_baseline"]["validation_input"])
    sources = [checked_frame(spec) for spec in config["fixed_sources"]]
    expected_index = pd.date_range(
        "2024-07-01 00:00", "2025-01-01 00:00", freq="h", name="forecast_kst_dtm"
    )
    for source in sources:
        if not source.index.equals(expected_index):
            raise ValueError("source increment index differs from frozen H2 index")
    recent_h2 = recent.reindex(expected_index)
    if tuple(recent_h2.columns) != TARGET_COLS or not np.isfinite(recent_h2.to_numpy()).all():
        raise ValueError("recent-v4 H2 baseline differs or is incomplete")

    baseline = pd.DataFrame(index=expected_index, columns=TARGET_COLS, dtype=np.float64)
    candidate = pd.DataFrame(index=expected_index, columns=TARGET_COLS, dtype=np.float64)
    consensus = pd.DataFrame(index=expected_index, columns=TARGET_COLS, dtype=np.float64)
    for group in TARGET_COLS:
        capacity = CAPACITY_KWH[group]
        baseline[group] = np.clip(
            BASE_SCALE * recent_h2[group].to_numpy(dtype=np.float64),
            0.0,
            1.02 * capacity,
        )
        column = SOURCE_COLUMN.format(group=group)
        values = np.column_stack([source[column].to_numpy(dtype=np.float64) for source in sources])
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite source increments for {group}")
        consensus[group] = values.mean(axis=1)
        if group == "kpx_group_1":
            candidate[group] = baseline[group].to_numpy(dtype=np.float64)
        else:
            candidate[group] = np.clip(
                baseline[group].to_numpy(dtype=np.float64)
                + TRANSFER_WEIGHT * capacity * consensus[group].to_numpy(dtype=np.float64),
                0.0,
                1.02 * capacity,
            )
    if not np.array_equal(
        candidate["kpx_group_1"].to_numpy(), baseline["kpx_group_1"].to_numpy()
    ):
        raise AssertionError("G1 is not exact identity")
    return baseline, candidate, consensus


def bounded_label_prefix() -> tuple[bytes, dict[str, Any]]:
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    with LABEL_PATH.open("rb", buffering=0) as stream:
        for line_number in range(LABEL_PREFIX_DATA_ROWS + 1):
            line = stream.readline()
            if not line:
                raise ValueError(f"label file ended at physical line {line_number}")
            chunks.append(line)
            digest.update(line)
    payload = b"".join(chunks)
    return payload, {
        "path": str(LABEL_PATH),
        "physical_lines_read": LABEL_PREFIX_DATA_ROWS + 1,
        "data_rows": LABEL_PREFIX_DATA_ROWS,
        "physical_prefix_bytes": len(payload),
        "physical_prefix_sha256": digest.hexdigest(),
        "later_bytes_read_or_parsed": False,
    }


def read_labels(payload: bytes) -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(payload), encoding="utf-8-sig")
    if tuple(frame.columns) != ("kst_dtm", *TARGET_COLS) or len(frame) != LABEL_PREFIX_DATA_ROWS:
        raise ValueError("bounded label prefix differs")
    frame.index = pd.DatetimeIndex(
        pd.to_datetime(frame.pop("kst_dtm"), errors="raise"), name="forecast_kst_dtm"
    )
    frame = frame.astype(np.float64)
    if frame.index[0] != pd.Timestamp("2022-01-01 01:00") or frame.index[-1] != pd.Timestamp(
        "2025-01-01 00:00"
    ):
        raise ValueError("bounded labels do not end at the operating-2024 terminal row")
    return frame


def one_group_metric(actual: pd.Series, prediction: pd.Series, group: str) -> dict[str, Any]:
    result = group_metrics(actual, prediction, CAPACITY_KWH[group], group_name=group).as_dict()
    result["total_score"] = 0.5 * result["one_minus_nmae"] + 0.5 * result["ficr"]
    return result


def delta(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, float]:
    return {
        key: float(candidate[key]) - float(baseline[key])
        for key in ("total_score", "one_minus_nmae", "ficr")
    }


def score_slices(
    labels: pd.DataFrame, baseline: pd.DataFrame, candidate: pd.DataFrame
) -> dict[str, Any]:
    payload: dict[str, Any] = {"mixed": {}, "by_group": {group: {} for group in TARGET_COLS}}
    for slice_name, (start, end) in SLICES.items():
        index = baseline.index[(baseline.index >= start) & (baseline.index <= end)]
        actual = labels.reindex(index)
        base_mixed = score_details(actual, baseline.loc[index]).as_dict()
        cand_mixed = score_details(actual, candidate.loc[index]).as_dict()
        payload["mixed"][slice_name] = {
            "baseline": base_mixed,
            "candidate": cand_mixed,
            "delta": delta(cand_mixed, base_mixed),
        }
        for group in TARGET_COLS:
            base_group = one_group_metric(actual[group], baseline.loc[index, group], group)
            cand_group = one_group_metric(actual[group], candidate.loc[index, group], group)
            payload["by_group"][group][slice_name] = {
                "baseline": base_group,
                "candidate": cand_group,
                "delta": delta(cand_group, base_group),
            }
    return payload


def run_validation() -> dict[str, Any]:
    config, config_hash = load_frozen_config()
    baseline, candidate, consensus = build_candidate_before_labels(config)
    candidate_path = OUTPUT_DIR / "validation/candidate_2024_h2.parquet"
    baseline_path = OUTPUT_DIR / "validation/baseline_recent097_2024_h2.parquet"
    consensus_path = OUTPUT_DIR / "validation/consensus_increment_cf_2024_h2.parquet"
    atomic_parquet(baseline, baseline_path)
    atomic_parquet(candidate, candidate_path)
    atomic_parquet(consensus, consensus_path)
    lock = {
        "experiment_id": config["experiment_id"],
        "config_sha256": config_hash,
        "candidate_materialized_before_label_read": True,
        "g1_exact_identity": bool(np.array_equal(candidate.iloc[:, 0], baseline.iloc[:, 0])),
        "files": {
            "baseline": file_record(baseline_path),
            "candidate": file_record(candidate_path),
            "consensus_increment_cf": file_record(consensus_path),
        },
    }
    lock_path = OUTPUT_DIR / "candidate_lock_before_2024_labels.json"
    atomic_json(lock, lock_path)

    label_bytes, label_record = bounded_label_prefix()
    labels = read_labels(label_bytes)
    comparisons = score_slices(labels, baseline, candidate)
    h2_delta = comparisons["mixed"]["H2"]["delta"]
    promoted = bool(h2_delta["total_score"] > 0.0 and h2_delta["ficr"] > 0.0)
    result = {
        "experiment_id": config["experiment_id"],
        "selection_safety": config["selection_safety"],
        "config_sha256": config_hash,
        "candidate_lock": file_record(lock_path),
        "label_prefix": label_record,
        "comparisons": comparisons,
        "promotion_gate": {
            "expression": "mixed_H2_delta_total_score > 0 AND mixed_H2_delta_FICR > 0",
            "mixed_H2_delta": h2_delta,
            "promoted": promoted,
            "final_stage_authorized": promoted,
        },
    }
    result_path = OUTPUT_DIR / "validation_results.json"
    atomic_json(result, result_path)
    manifest = {
        "experiment_id": config["experiment_id"],
        "status": "validation_promoted" if promoted else "validation_rejected",
        "files": [
            file_record(CONFIG_PATH),
            file_record(SIDECAR_PATH),
            file_record(Path(__file__)),
            file_record(lock_path),
            file_record(baseline_path),
            file_record(candidate_path),
            file_record(consensus_path),
            file_record(result_path),
        ],
    }
    atomic_json(manifest, OUTPUT_DIR / "manifest.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("validation",), default="validation")
    args = parser.parse_args()
    if args.stage == "validation":
        result = run_validation()
        print(json.dumps(result["promotion_gate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Validate and, only after promotion, finalize the frozen 50/50 parent hedge."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metric import CAPACITY_KWH, TARGET_COLS, group_metrics, score_details


EXPERIMENT_ID = "multi_nwp_parent_5050_hedge_v1"
CONFIG_PATH = ROOT / "configs/multi_nwp_parent_5050_hedge_preregister_v1.json"
SIDECAR_PATH = CONFIG_PATH.with_suffix(".sha256")
OUTPUT_DIR = ROOT / f"artifacts/postgate/{EXPERIMENT_ID}"
LABEL_PATH = Path(r"data/local/open/train/train_labels.csv")
SAMPLE_PATH = Path(r"data/local/open/sample_submission.csv")
LABEL_ROWS = 26_304
PARENT_JOINT = ROOT / "artifacts/postgate/multi_nwp_joint_g12_posthoc_rescue_v1/predictions/joint_g12_candidate_scale097.parquet"
PARENT_CONSENSUS = ROOT / "artifacts/postgate/multi_nwp_consensus_g23_rescue_v1/validation/candidate_2024_h2.parquet"
PARENT_BASE = ROOT / "artifacts/postgate/multi_nwp_consensus_g23_rescue_v1/validation/baseline_recent097_2024_h2.parquet"
PARENT_BASE_SHA256 = "496687b38986f0a8933bd9d0d2404d1711de802b0e4e14d31e2ec4c2d00c7a55"
VALIDATION_INDEX = pd.date_range("2024-07-01 00:00", "2025-01-01 00:00", freq="h", name="forecast_kst_dtm")
SLICES = {
    "H2": (pd.Timestamp("2024-07-01 01:00"), pd.Timestamp("2025-01-01 00:00")),
    "Q3": (pd.Timestamp("2024-07-01 01:00"), pd.Timestamp("2024-10-01 00:00")),
    "Q4": (pd.Timestamp("2024-10-01 01:00"), pd.Timestamp("2025-01-01 00:00")),
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
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    frame.to_parquet(tmp, engine="pyarrow", compression="zstd", index=True)
    os.replace(tmp, path)


def load_config() -> tuple[dict[str, Any], str]:
    observed = sha256(CONFIG_PATH)
    expected = SIDECAR_PATH.read_text(encoding="ascii").split()[0].lower()
    if observed != expected:
        raise RuntimeError(f"preregister sidecar mismatch: {observed} != {expected}")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["experiment_id"] != EXPERIMENT_ID or config["selection_status"] != "posthoc_selection_unsafe":
        raise RuntimeError("unexpected frozen registration")
    if config["fixed_formula"]["joint_weight"] != 0.5 or config["fixed_formula"]["consensus_weight"] != 0.5:
        raise RuntimeError("only the registered equal-weight hedge is allowed")
    for key, path in (("joint_g12", PARENT_JOINT), ("consensus_g23", PARENT_CONSENSUS)):
        if sha256(path) != config["parents"][key]["validation_candidate_sha256"]:
            raise RuntimeError(f"changed validation parent: {key}")
    if sha256(PARENT_BASE) != PARENT_BASE_SHA256:
        raise RuntimeError("changed common validation baseline")
    return config, observed


def checked_frame(path: Path, expected_index: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    frame = pd.read_parquet(path).loc[:, list(TARGET_COLS)].astype(np.float64)
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if not frame.index.is_unique or tuple(frame.columns) != TARGET_COLS:
        raise ValueError(f"invalid parent schema: {path}")
    if expected_index is not None and not frame.index.equals(expected_index):
        raise ValueError(f"unexpected parent index: {path}")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"non-finite parent: {path}")
    return frame


def clipped_hedge(joint: pd.DataFrame, consensus: pd.DataFrame, base: pd.DataFrame | None = None) -> pd.DataFrame:
    if not joint.index.equals(consensus.index):
        raise ValueError("parent indices differ")
    if base is not None and (not joint.index.equals(base.index) or tuple(base.columns) != TARGET_COLS):
        raise ValueError("common base differs")
    result = pd.DataFrame(index=joint.index, columns=list(TARGET_COLS), dtype=np.float64)
    for group in TARGET_COLS:
        capacity = CAPACITY_KWH[group]
        if base is None:
            values = 0.5 * joint[group].to_numpy() + 0.5 * consensus[group].to_numpy()
        else:
            base_values = base[group].to_numpy()
            values = base_values + 0.5 * (joint[group].to_numpy() - base_values) + 0.5 * (consensus[group].to_numpy() - base_values)
            direct = 0.5 * joint[group].to_numpy() + 0.5 * consensus[group].to_numpy()
            # The two algebraically identical evaluation orders can differ by
            # one float64 ULP at this kWh scale; this is only an invariant check.
            if not np.allclose(values, direct, rtol=0.0, atol=5e-12):
                raise AssertionError("registered expression differs from arithmetic mean")
        result[group] = np.clip(values, 0.0, 1.02 * capacity)
    return result


def bounded_labels() -> tuple[pd.DataFrame, dict[str, Any]]:
    chunks: list[bytes] = []
    digest = hashlib.sha256()
    with LABEL_PATH.open("rb", buffering=0) as stream:
        for line_number in range(LABEL_ROWS + 1):
            line = stream.readline()
            if not line:
                raise ValueError(f"label file ended at physical line {line_number}")
            chunks.append(line)
            digest.update(line)
    payload = b"".join(chunks)
    frame = pd.read_csv(io.BytesIO(payload), encoding="utf-8-sig")
    if tuple(frame.columns) != ("kst_dtm", *TARGET_COLS) or len(frame) != LABEL_ROWS:
        raise ValueError("bounded label prefix changed")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("kst_dtm"), errors="raise"), name="forecast_kst_dtm")
    return frame.astype(np.float64), {
        "path": str(LABEL_PATH), "data_rows": LABEL_ROWS, "physical_prefix_bytes": len(payload),
        "physical_prefix_sha256": digest.hexdigest(), "later_bytes_read_or_parsed": False,
    }


def metric_payload(actual: pd.DataFrame, prediction: pd.DataFrame) -> dict[str, Any]:
    return score_details(actual, prediction).as_dict()


def group_payload(actual: pd.Series, prediction: pd.Series, group: str) -> dict[str, Any]:
    payload = group_metrics(actual, prediction, CAPACITY_KWH[group], group_name=group).as_dict()
    payload["total_score"] = 0.5 * (payload["one_minus_nmae"] + payload["ficr"])
    return payload


def diff(after: dict[str, Any], before: dict[str, Any]) -> dict[str, float]:
    return {key: float(after[key]) - float(before[key]) for key in ("total_score", "one_minus_nmae", "ficr")}


def score_registered_slices(labels: pd.DataFrame, base: pd.DataFrame, hedge: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {"mixed": {}, "groups": {group: {} for group in TARGET_COLS}}
    for name, (start, end) in SLICES.items():
        index = hedge.index[(hedge.index >= start) & (hedge.index <= end)]
        actual = labels.reindex(index)
        b = metric_payload(actual, base.loc[index])
        h = metric_payload(actual, hedge.loc[index])
        output["mixed"][name] = {"baseline": b, "candidate": h, "delta": diff(h, b), "rows": len(index)}
        for group in TARGET_COLS:
            bg = group_payload(actual[group], base.loc[index, group], group)
            hg = group_payload(actual[group], hedge.loc[index, group], group)
            output["groups"][group][name] = {"baseline": bg, "candidate": hg, "delta": diff(hg, bg), "rows": len(index)}
    return output


def gate(comparison: dict[str, Any]) -> dict[str, Any]:
    h2 = comparison["mixed"]["H2"]["delta"]
    q3 = comparison["mixed"]["Q3"]["delta"]["total_score"]
    q4 = comparison["mixed"]["Q4"]["delta"]["total_score"]
    group_slices = {
        f"{group}__{name}": comparison["groups"][group][name]["delta"]["total_score"]
        for group in TARGET_COLS for name in ("H2", "Q3", "Q4")
    }
    positive = sum(value > 0.0 for value in group_slices.values())
    checks = {
        "mixed_H2_total_score_delta_gt_0": h2["total_score"] > 0.0,
        "mixed_H2_FICR_delta_gt_0": h2["ficr"] > 0.0,
        "mixed_Q3_total_score_delta_ge_minus_0.0005": q3 >= -0.0005,
        "mixed_Q4_total_score_delta_ge_minus_0.0005": q4 >= -0.0005,
        "positive_group_slices_ge_6_of_9": positive >= 6,
    }
    return {
        "pass": bool(all(checks.values())), "checks": checks, "mixed_H2_delta": h2,
        "mixed_Q3_total_score_delta": q3, "mixed_Q4_total_score_delta": q4,
        "group_slice_total_score_deltas": group_slices, "positive_group_slices": positive,
    }


def run_validation() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True)
    config, config_hash = load_config()
    joint = checked_frame(PARENT_JOINT, VALIDATION_INDEX)
    consensus = checked_frame(PARENT_CONSENSUS, VALIDATION_INDEX)
    base = checked_frame(PARENT_BASE, VALIDATION_INDEX)
    hedge = clipped_hedge(joint, consensus, base)
    prediction_path = OUTPUT_DIR / "validation/parent_5050_hedge_2024_h2.parquet"
    atomic_parquet(hedge, prediction_path)
    lock_path = OUTPUT_DIR / "candidate_lock_before_label_access.json"
    atomic_json({
        "created_utc": datetime.now(timezone.utc).isoformat(), "experiment_id": EXPERIMENT_ID,
        "status": "single_fixed_5050_candidate_locked_before_label_access",
        "preregister": record(CONFIG_PATH), "candidate": record(prediction_path),
        "parents": [record(PARENT_JOINT), record(PARENT_CONSENSUS)], "common_base": record(PARENT_BASE),
        "formula": config["fixed_formula"]["expression"], "candidate_count": 1,
    }, lock_path)
    labels, label_record = bounded_labels()
    comparison = score_registered_slices(labels, base, hedge)
    decision = gate(comparison)
    result_path = OUTPUT_DIR / "validation_results.json"
    atomic_json({
        "created_utc": datetime.now(timezone.utc).isoformat(), "experiment_id": EXPERIMENT_ID,
        "selection_status": "posthoc_selection_unsafe", "config_sha256": config_hash,
        "candidate_lock": record(lock_path), "label_prefix": label_record,
        "comparison": comparison, "promotion_gate": decision,
        "status": "validation_promoted" if decision["pass"] else "validation_rejected_no_rescue",
    }, result_path)
    promotion_path = OUTPUT_DIR / ("promotion_lock.json" if decision["pass"] else "rejection_lock.json")
    atomic_json({
        "status": "promoted_for_fixed_finalization" if decision["pass"] else "rejected_no_finalization",
        "validation_results": record(result_path), "final_stage_authorized": decision["pass"],
        "no_alternate_weight_group_subset_or_rescue": True,
    }, promotion_path)
    outputs = [prediction_path, lock_path, result_path, promotion_path]
    manifest_path = OUTPUT_DIR / "manifest.json"
    atomic_json({
        "created_utc": datetime.now(timezone.utc).isoformat(), "experiment_id": EXPERIMENT_ID,
        "selection_status": "posthoc_selection_unsafe", "status": "validation_promoted" if decision["pass"] else "validation_rejected",
        "preregister": record(CONFIG_PATH), "outputs": [record(path) for path in outputs],
    }, manifest_path)
    (OUTPUT_DIR / "manifest.sha256").write_text(f"{sha256(manifest_path)}  manifest.json\n", encoding="ascii")
    print(json.dumps(decision, ensure_ascii=False, indent=2))


def read_final_parent(path: Path) -> pd.DataFrame:
    frame = checked_frame(path)
    if len(frame) != 8760:
        raise ValueError(f"final parent must have 8760 rows: {path}")
    return frame


def run_final(joint_path: Path, consensus_path: Path) -> None:
    promotion_path = OUTPUT_DIR / "promotion_lock.json"
    if not promotion_path.exists() or not json.loads(promotion_path.read_text(encoding="utf-8"))["final_stage_authorized"]:
        raise RuntimeError("validation did not authorize finalization")
    final_dir = OUTPUT_DIR / "final"
    if final_dir.exists():
        raise FileExistsError(f"refusing to overwrite {final_dir}")
    joint = read_final_parent(joint_path)
    consensus = read_final_parent(consensus_path)
    if not joint.index.equals(consensus.index):
        raise ValueError("final parent indices differ")
    hedge = clipped_hedge(joint, consensus)
    final_dir.mkdir(parents=True)
    parquet_path = final_dir / "multi_nwp_parent_5050_hedge_v1_2025.parquet"
    atomic_parquet(hedge, parquet_path)

    sample = pd.read_csv(SAMPLE_PATH, encoding="utf-8-sig")
    if len(sample) != 8760 or tuple(sample.columns) != ("ID", *TARGET_COLS):
        raise ValueError("unexpected sample submission schema")
    output = sample.copy()
    for group in TARGET_COLS:
        output[group] = hedge[group].to_numpy(dtype=np.float64)
    csv_path = final_dir / "multi_nwp_parent_5050_hedge_v1_2025.csv"
    output.to_csv(csv_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    raw = csv_path.read_bytes()
    if not raw.startswith(b"\xef\xbb\xbf"):
        raise AssertionError("CSV lacks UTF-8 BOM")
    reopened = pd.read_csv(csv_path, encoding="utf-8-sig")
    if not reopened["ID"].equals(sample["ID"]) or tuple(reopened.columns) != tuple(sample.columns) or len(reopened) != 8760:
        raise AssertionError("CSV order/schema changed")
    values = reopened.loc[:, list(TARGET_COLS)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise AssertionError("CSV has non-finite values")
    for group in TARGET_COLS:
        if reopened[group].min() < 0.0 or reopened[group].max() > 1.02 * CAPACITY_KWH[group] + 5e-7:
            raise AssertionError(f"CSV bounds failed: {group}")
    manifest_path = final_dir / "manifest.json"
    atomic_json({
        "created_utc": datetime.now(timezone.utc).isoformat(), "experiment_id": EXPERIMENT_ID,
        "status": "final_csv_created_after_frozen_gate_pass", "selection_status": "posthoc_selection_unsafe",
        "promotion_lock": record(promotion_path), "parent_final_parquets": [record(joint_path), record(consensus_path)],
        "formula": "clip(0.5*joint_final_prediction + 0.5*consensus_final_prediction, 0, 1.02*C)",
        "outputs": [record(parquet_path), record(csv_path)],
        "csv_validation": {"encoding_bom": "utf-8-sig", "rows": 8760, "columns": list(sample.columns), "decimal_places": 6, "finite": True, "bounds_pass": True, "sample_id_order_exact": True},
        "leaderboard_score_claim": False,
    }, manifest_path)
    (final_dir / "manifest.sha256").write_text(f"{sha256(manifest_path)}  manifest.json\n", encoding="ascii")
    print(json.dumps({"csv": record(csv_path), "parquet": record(parquet_path), "manifest": record(manifest_path)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("validation", "final"), required=True)
    parser.add_argument("--joint-final", type=Path)
    parser.add_argument("--consensus-final", type=Path)
    args = parser.parse_args()
    if args.stage == "validation":
        run_validation()
    else:
        if args.joint_final is None or args.consensus_final is None:
            parser.error("--joint-final and --consensus-final are required for --stage final")
        run_final(args.joint_final.resolve(), args.consensus_final.resolve())


if __name__ == "__main__":
    main()

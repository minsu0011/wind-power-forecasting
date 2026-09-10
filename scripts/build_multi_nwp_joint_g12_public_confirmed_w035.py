"""Validate and build the single frozen public-confirmed G12 w=0.35 candidate.

The candidate is constructed and hash-locked before the 2024 H2 labels are
opened.  A 2025 CSV is emitted only if every preregistered confirmation gate
passes.  There is deliberately no weight search or fallback in this runner.
"""

from __future__ import annotations

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

from src.metric import CAPACITY_KWH, TARGET_COLS, score_details


EXPERIMENT_ID = "multi_nwp_joint_g12_public_confirmed_w035_v1"
CONFIG = ROOT / "configs/multi_nwp_joint_g12_public_confirmed_w035_preregister_v2.json"
SIDECAR = CONFIG.with_suffix(".sha256")
FEEDBACK = ROOT / "artifacts/audits/public_feedback_multi_nwp_joint_g12_20260809_v1.json"
PARENT_PREREGISTER = ROOT / "configs/multi_nwp_joint_disagreement_paired_increment_preregister_v1.json"
RAW_INCREMENT_2024 = ROOT / "artifacts/postgate/multi_nwp_joint_disagreement_2024_forward_v1/predictions/joint_paired_increment_cf.parquet"
BASE_2024 = ROOT / "artifacts/postgate/multi_nwp_consensus_g23_rescue_v1/validation/baseline_recent097_2024_h2.parquet"
LABELS = Path(r"data/local/open/train/train_labels.csv")
PARENT_FINAL_MANIFEST = ROOT / "artifacts/postgate/multi_nwp_joint_g12_posthoc_rescue_final_2025_v1/manifest.json"
RAW_INCREMENT_2025 = ROOT / "artifacts/postgate/multi_nwp_joint_g12_posthoc_rescue_final_2025_v1/final_joint_increment_cf_2025.parquet"
BASE_2025 = ROOT / "artifacts/postgate/multi_nwp_joint_g12_posthoc_rescue_final_2025_v1/base_scale097_2025.parquet"
SAMPLE = Path(r"data/local/open/sample_submission.csv")
OUTPUT = ROOT / f"artifacts/postgate/{EXPERIMENT_ID}"

EXPECTED = {
    CONFIG: "e1948869a9acce950cffaad805b4f77bd223ac47c66381e2b6dd3bec15caeca9",
    FEEDBACK: "75caab7deaee9fd0370e8fc5ead687056593c5dd9bec8924ba37f27d61e1eead",
    PARENT_PREREGISTER: "bf52d742cb9b0160f31d27e949f6f1b4575c40575472520e9307f5af998957af",
    RAW_INCREMENT_2024: "14419ac0a939af154f334b2dc543fef0085e29be863a96432e5fdfd3378fb191",
    BASE_2024: "496687b38986f0a8933bd9d0d2404d1711de802b0e4e14d31e2ec4c2d00c7a55",
    LABELS: "47bb64252195cf4734e67394d6e50485f27a608def3b5a8791fcc7674bbceb03",
    PARENT_FINAL_MANIFEST: "a8e2cda26ae133b1263784e92572b9e797a1d5a2c8e25e8ca4dcd0ff93c92f0f",
    RAW_INCREMENT_2025: "204d01db3fdf8919440126423963d89a32afb8c2b46bd451f5027da012c62a30",
    BASE_2025: "a09ab6906dc026ad82b0ea7d8a0e9f6e88e8b4f839ba5d4d2426f37f0b7405cb",
    SAMPLE: "c925d2066a834f937f8091ed55acfe50ff86c8be4745b52c3adc95b056c5aaaa",
}
SCORE_INDEX = pd.date_range("2024-07-01 00:00", "2025-01-01 00:00", freq="h", name="forecast_kst_dtm")
TEST_INDEX = pd.date_range("2025-01-01 01:00", "2026-01-01 00:00", freq="h", name="forecast_kst_dtm")
WINDOWS = {
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


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    frame.to_parquet(temporary, engine="pyarrow", compression="zstd", index=True)
    os.replace(temporary, path)


def verify(path: Path) -> None:
    expected = EXPECTED[path]
    if not path.is_file() or sha256(path) != expected:
        raise RuntimeError(f"frozen input changed: {path}")


def load_frame(path: Path, index: pd.DatetimeIndex) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if not frame.index.equals(index) or tuple(frame.columns) != tuple(TARGET_COLS):
        raise ValueError(f"frame schema/index differs: {path}")
    values = frame.loc[:, list(TARGET_COLS)].astype(np.float64)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"non-finite values: {path}")
    return values


def build_candidate(base: pd.DataFrame, increment_cf: pd.DataFrame) -> pd.DataFrame:
    if not base.index.equals(increment_cf.index):
        raise ValueError("base/increment index mismatch")
    candidate = base.copy()
    for group in ("kpx_group_1", "kpx_group_2"):
        candidate[group] = np.clip(
            base[group].to_numpy(dtype=np.float64)
            + 0.35 * CAPACITY_KWH[group] * increment_cf[group].to_numpy(dtype=np.float64),
            0.0,
            1.02 * CAPACITY_KWH[group],
        )
    if not np.array_equal(candidate["kpx_group_3"].to_numpy(), base["kpx_group_3"].to_numpy()):
        raise AssertionError("G3 must be exact identity")
    if np.count_nonzero(increment_cf.iloc[-1].to_numpy(dtype=np.float64)):
        raise AssertionError("terminal increment must be exact zero")
    candidate.iloc[-1] = base.iloc[-1]
    if not np.array_equal(candidate.iloc[-1].to_numpy(), base.iloc[-1].to_numpy()):
        raise AssertionError("terminal row must be exact identity")
    return candidate


def read_bounded_labels() -> tuple[pd.DataFrame, dict[str, Any]]:
    # The training label file contains exactly the registered 2022-2024 rows.
    # Hash verification happens only after the candidate lock is durable.
    verify(LABELS)
    raw = LABELS.read_bytes()
    frame = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
    if tuple(frame.columns) != ("kst_dtm", *TARGET_COLS):
        raise ValueError("label schema changed")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("kst_dtm"), errors="raise"), name="forecast_kst_dtm")
    labels = frame.loc[SCORE_INDEX, list(TARGET_COLS)].astype(np.float64)
    return labels, {**record(LABELS), "selected_rows": len(labels), "selected_start": str(labels.index[0]), "selected_end": str(labels.index[-1])}


def metric_delta(actual: pd.DataFrame, base: pd.DataFrame, candidate: pd.DataFrame, columns: tuple[str, ...]) -> dict[str, float]:
    before = score_details(actual.loc[:, list(columns)], base.loc[:, list(columns)], target_cols=columns).as_dict()
    after = score_details(actual.loc[:, list(columns)], candidate.loc[:, list(columns)], target_cols=columns).as_dict()
    return {
        "total_score": float(after["total_score"] - before["total_score"]),
        "one_minus_nmae": float(after["one_minus_nmae"] - before["one_minus_nmae"]),
        "ficr": float(after["ficr"] - before["ficr"]),
    }


def evaluate(labels: pd.DataFrame, base: pd.DataFrame, candidate: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    mixed: dict[str, dict[str, float]] = {}
    groups: dict[str, dict[str, dict[str, float]]] = {group: {} for group in TARGET_COLS}
    for name, (start, end) in WINDOWS.items():
        idx = labels.index[(labels.index >= start) & (labels.index <= end)]
        mixed[name] = metric_delta(labels.loc[idx], base.loc[idx], candidate.loc[idx], tuple(TARGET_COLS))
        for group in TARGET_COLS:
            groups[group][name] = metric_delta(labels.loc[idx], base.loc[idx], candidate.loc[idx], (group,))
    positive = sum(mixed[name]["total_score"] > 0.0 for name in WINDOWS)
    checks = {
        "mixed_h2_total_ge_0p0015": mixed["H2"]["total_score"] >= 0.0015,
        "mixed_h2_ficr_ge_0p002": mixed["H2"]["ficr"] >= 0.002,
        "g1_g2_h2_total_positive": min(groups[group]["H2"]["total_score"] for group in ("kpx_group_1", "kpx_group_2")) > 0.0,
        "mixed_q3_q4_floor": min(mixed[name]["total_score"] for name in ("Q3", "Q4")) >= -0.001,
        "positive_mixed_segments_ge_6": positive >= 6,
    }
    gate = {"checks": checks, "pass": bool(all(checks.values())), "positive_mixed_segments": positive}
    return {"mixed": mixed, "groups": groups}, gate


def write_submission(prediction: pd.DataFrame, path: Path) -> dict[str, Any]:
    verify(SAMPLE)
    sample = pd.read_csv(SAMPLE, encoding="utf-8-sig", dtype="string", keep_default_na=False)
    expected_columns = ("forecast_id", "forecast_kst_dtm", *TARGET_COLS)
    if tuple(sample.columns) != expected_columns or len(sample) != len(TEST_INDEX):
        raise ValueError("sample schema differs")
    sample_index = pd.DatetimeIndex(pd.to_datetime(sample["forecast_kst_dtm"], errors="raise"), name="forecast_kst_dtm")
    if not sample_index.equals(TEST_INDEX) or not prediction.index.equals(TEST_INDEX):
        raise ValueError("sample/prediction index differs")
    output = sample.loc[:, ["forecast_id", "forecast_kst_dtm"]].copy()
    for group in TARGET_COLS:
        output[group] = prediction[group].to_numpy(dtype=np.float64)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    output.to_csv(temporary, index=False, encoding="utf-8-sig", float_format="%.6f", lineterminator="\n")
    os.replace(temporary, path)
    raw = path.read_bytes()
    if not raw.startswith(b"\xef\xbb\xbf") or b"\r\n" in raw:
        raise AssertionError("CSV encoding/line endings differ")
    rendered = pd.read_csv(path, encoding="utf-8-sig", dtype="string", keep_default_na=False)
    if not rendered.loc[:, ["forecast_id", "forecast_kst_dtm"]].equals(sample.loc[:, ["forecast_id", "forecast_kst_dtm"]]):
        raise AssertionError("CSV identifier order differs")
    for group in TARGET_COLS:
        expected = pd.Series([f"{value:.6f}" for value in prediction[group].to_numpy(dtype=np.float64)], dtype="string")
        if not rendered[group].reset_index(drop=True).equals(expected):
            raise AssertionError(f"CSV text replay differs: {group}")
    return {**record(path), "rows": len(rendered), "columns": list(rendered.columns), "utf8_bom": True, "six_decimals": True}


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"fresh output required: {OUTPUT}")
    if SIDECAR.read_text(encoding="ascii").split()[0] != EXPECTED[CONFIG]:
        raise RuntimeError("preregister sidecar mismatch")
    for path in (CONFIG, FEEDBACK, PARENT_PREREGISTER, RAW_INCREMENT_2024, BASE_2024):
        verify(path)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config["status"] != "frozen_before_w035_candidate_materialization_or_2024_metric_or_2025_values":
        raise RuntimeError("preregister is not frozen")
    OUTPUT.mkdir(parents=True)

    base_2024 = load_frame(BASE_2024, SCORE_INDEX)
    increment_2024 = load_frame(RAW_INCREMENT_2024, SCORE_INDEX)
    candidate_2024 = build_candidate(base_2024, increment_2024)
    base_path = OUTPUT / "validation/base_scale097_2024_h2.parquet"
    candidate_path = OUTPUT / "validation/candidate_w035_2024_h2.parquet"
    atomic_parquet(base_path, base_2024)
    atomic_parquet(candidate_path, candidate_2024)
    candidate_lock = OUTPUT / "candidate_before_2024_label_access.json"
    atomic_json(candidate_lock, {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "candidate_hash_locked_before_2024_label_access",
        "preregister": record(CONFIG),
        "public_feedback_aggregate_only": record(FEEDBACK),
        "base": record(base_path),
        "raw_increment": record(RAW_INCREMENT_2024),
        "candidate": record(candidate_path),
        "formula": "G1/G2=clip(base_scale097+0.35*C*raw_joint_increment_cf,0,1.02C); G3 and terminal exact identity",
        "candidate_count": 1,
        "labels_or_metric_read_before_lock": 0,
    })

    labels, label_record = read_bounded_labels()
    atomic_json(OUTPUT / "label_access_after_candidate_lock.json", {"candidate_lock": record(candidate_lock), "labels": label_record})
    comparisons, gate = evaluate(labels, base_2024, candidate_2024)
    results_path = OUTPUT / "validation_results.json"
    atomic_json(results_path, {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": EXPERIMENT_ID,
        "selection_status": "public_adaptive_posthoc_selection_unsafe",
        "comparisons": comparisons,
        "gate": gate,
        "year_2025_values_read_before_gate": 0,
        "no_weight_grid_or_rescue": True,
    })

    outputs = [base_path, candidate_path, candidate_lock, OUTPUT / "label_access_after_candidate_lock.json", results_path]
    csv_record: dict[str, Any] | None = None
    if gate["pass"]:
        for path in (PARENT_FINAL_MANIFEST, RAW_INCREMENT_2025, BASE_2025, SAMPLE):
            verify(path)
        base_2025 = load_frame(BASE_2025, TEST_INDEX)
        increment_2025 = load_frame(RAW_INCREMENT_2025, TEST_INDEX)
        prediction_2025 = build_candidate(base_2025, increment_2025)
        prediction_path = OUTPUT / "prediction_w035_2025.parquet"
        atomic_parquet(prediction_path, prediction_2025)
        csv_path = OUTPUT / "multi_nwp_joint_g12_public_confirmed_w035_2025.csv"
        csv_record = write_submission(prediction_2025, csv_path)
        final_lock = OUTPUT / "final_candidate_lock.json"
        atomic_json(final_lock, {
            "status": "confirmation_passed_final_csv_ready",
            "validation_results": record(results_path),
            "parent_final_manifest": record(PARENT_FINAL_MANIFEST),
            "base_2025": record(BASE_2025),
            "raw_increment_2025": record(RAW_INCREMENT_2025),
            "prediction_2025": record(prediction_path),
            "csv": csv_record,
            "g3_identity": bool(np.array_equal(prediction_2025["kpx_group_3"].to_numpy(), base_2025["kpx_group_3"].to_numpy())),
            "terminal_identity": bool(np.array_equal(prediction_2025.iloc[-1].to_numpy(), base_2025.iloc[-1].to_numpy())),
        })
        outputs.extend([prediction_path, csv_path, final_lock])
    else:
        rejection = OUTPUT / "REJECTED_NO_2025_READ_NO_CSV.json"
        atomic_json(rejection, {"status": "confirmation_failed", "validation_results": record(results_path), "year_2025_values_read": 0, "csv_created": False, "no_retry": True})
        outputs.append(rejection)

    manifest = OUTPUT / "manifest.json"
    atomic_json(manifest, {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": EXPERIMENT_ID,
        "selection_status": "public_adaptive_posthoc_selection_unsafe",
        "preregister": record(CONFIG),
        "gate_pass": gate["pass"],
        "csv_created": csv_record is not None,
        "outputs_excluding_manifest_and_sidecar": [record(path) for path in outputs],
    })
    (OUTPUT / "manifest.sha256").write_text(f"{sha256(manifest)}  manifest.json\n", encoding="ascii")
    print(json.dumps({"gate": gate, "h2": comparisons["mixed"]["H2"], "q3": comparisons["mixed"]["Q3"], "q4": comparisons["mixed"]["Q4"], "csv": csv_record, "manifest": record(manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

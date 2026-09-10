"""Build the one frozen G1-joint/G2-consensus/G3-identity hedge."""

from __future__ import annotations

import hashlib
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


EXPERIMENT = "multi_nwp_disjoint_g1joint_g2consensus_v1"
CONFIG = ROOT / "configs/multi_nwp_disjoint_g1joint_g2consensus_preregister_v1.json"
SIDECAR = CONFIG.with_suffix(".sha256")
FEEDBACK = ROOT / "artifacts/audits/public_feedback_multi_nwp_joint_g12_20260809_v1.json"
JOINT_MANIFEST_2024 = ROOT / "artifacts/postgate/multi_nwp_joint_g12_posthoc_rescue_v1/manifest.json"
JOINT_2024 = ROOT / "artifacts/postgate/multi_nwp_joint_g12_posthoc_rescue_v1/predictions/joint_g12_candidate_scale097.parquet"
CONSENSUS_MANIFEST_2024 = ROOT / "artifacts/postgate/multi_nwp_consensus_g23_rescue_v1/manifest.json"
CONSENSUS_2024 = ROOT / "artifacts/postgate/multi_nwp_consensus_g23_rescue_v1/validation/candidate_2024_h2.parquet"
BASE_2024 = ROOT / "artifacts/postgate/multi_nwp_consensus_g23_rescue_v1/validation/baseline_recent097_2024_h2.parquet"
LABELS = Path(r"data/local/open/train/train_labels.csv")
JOINT_MANIFEST_2025 = ROOT / "artifacts/postgate/multi_nwp_joint_g12_posthoc_rescue_final_2025_v1/manifest.json"
JOINT_2025 = ROOT / "artifacts/postgate/multi_nwp_joint_g12_posthoc_rescue_final_2025_v1/final_prediction_kwh_2025.parquet"
CONSENSUS_MANIFEST_2025 = ROOT / "artifacts/final_multi_nwp_consensus_g23_rescue_v1/manifest.json"
CONSENSUS_2025 = ROOT / "artifacts/final_multi_nwp_consensus_g23_rescue_v1/predictions/multi_nwp_consensus_g23_rescue_v1_2025.parquet"
BASE_2025 = ROOT / "artifacts/postgate/multi_nwp_joint_g12_posthoc_rescue_final_2025_v1/base_scale097_2025.parquet"
SAMPLE = Path(r"data/local/open/sample_submission.csv")
OUTPUT = ROOT / f"artifacts/postgate/{EXPERIMENT}"

EXPECTED = {
    CONFIG: "cf18aa56dd1d5cb22597cdaa66e9e7f266d3c026e7bd2fc47fd32c785dae9483",
    FEEDBACK: "75caab7deaee9fd0370e8fc5ead687056593c5dd9bec8924ba37f27d61e1eead",
    JOINT_MANIFEST_2024: "0071c7c516c7ac3c1fefa507c0adcd22838ad1cbd40d23c2c3864979bcabed51",
    JOINT_2024: "09c9f5ae28bb18aaa6d586e56abcbb8de281fad865ff7457b3e3466f1cbe6599",
    CONSENSUS_MANIFEST_2024: "1a09724433d99cb37b3d129ef8eb57df3885fdf66b56493974eea22c90dd934f",
    CONSENSUS_2024: "fe1951bc7747cafbba2e3a7ae2c15a05ec98f01fa254cb9dbc87ab3c3e43dd0e",
    BASE_2024: "496687b38986f0a8933bd9d0d2404d1711de802b0e4e14d31e2ec4c2d00c7a55",
    LABELS: "47bb64252195cf4734e67394d6e50485f27a608def3b5a8791fcc7674bbceb03",
    JOINT_MANIFEST_2025: "a8e2cda26ae133b1263784e92572b9e797a1d5a2c8e25e8ca4dcd0ff93c92f0f",
    JOINT_2025: "5cfbdf04d2d48e67efbec219f4cc24b61e7ffdff6e87fbe366f1468226c2fb83",
    CONSENSUS_MANIFEST_2025: "acb90e9a85fcb1277ad0d1476d7f86afc3c72aed4bd387107e48fcfaeded290d",
    CONSENSUS_2025: "e3e0fc6a4311680b637fca919e2b7dd305058144b8882224cb70aabe1564f2da",
    BASE_2025: "a09ab6906dc026ad82b0ea7d8a0e9f6e88e8b4f839ba5d4d2426f37f0b7405cb",
    SAMPLE: "c925d2066a834f937f8091ed55acfe50ff86c8be4745b52c3adc95b056c5aaaa",
}

H2_INDEX = pd.date_range("2024-07-01 00:00", "2025-01-01 00:00", freq="h", name="forecast_kst_dtm")
TEST_INDEX = pd.date_range("2025-01-01 01:00", "2026-01-01 00:00", freq="h", name="forecast_kst_dtm")
WINDOWS = {
    "H2": ("2024-07-01 00:00", "2025-01-01 00:00"),
    "Q3": ("2024-07-01 00:00", "2024-09-30 23:00"),
    "Q4": ("2024-10-01 00:00", "2025-01-01 00:00"),
    "Jul": ("2024-07-01 00:00", "2024-07-31 23:00"),
    "Aug": ("2024-08-01 00:00", "2024-08-31 23:00"),
    "Sep": ("2024-09-01 00:00", "2024-09-30 23:00"),
    "Oct": ("2024-10-01 00:00", "2024-10-31 23:00"),
    "Nov": ("2024-11-01 00:00", "2024-11-30 23:00"),
    "Dec": ("2024-12-01 00:00", "2025-01-01 00:00"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def verify(path: Path) -> None:
    if not path.is_file() or sha256(path) != EXPECTED[path]:
        raise RuntimeError(f"frozen input changed: {path}")


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


def read_frame(path: Path, expected_index: pd.DatetimeIndex) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if not frame.index.equals(expected_index) or tuple(frame.columns) != tuple(TARGET_COLS):
        raise ValueError(f"schema/index mismatch: {path}")
    frame = frame.loc[:, list(TARGET_COLS)].astype(np.float64)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"nonfinite frame: {path}")
    return frame


def compose(base: pd.DataFrame, joint: pd.DataFrame, consensus: pd.DataFrame) -> pd.DataFrame:
    if not base.index.equals(joint.index) or not base.index.equals(consensus.index):
        raise ValueError("parent index mismatch")
    candidate = base.copy()
    candidate["kpx_group_1"] = joint["kpx_group_1"].to_numpy(dtype=np.float64)
    candidate["kpx_group_2"] = consensus["kpx_group_2"].to_numpy(dtype=np.float64)
    candidate["kpx_group_3"] = base["kpx_group_3"].to_numpy(dtype=np.float64)
    candidate.iloc[-1] = base.iloc[-1]
    for group in TARGET_COLS:
        values = candidate[group].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or values.min() < 0.0 or values.max() > 1.02 * CAPACITY_KWH[group]:
            raise AssertionError(f"candidate bounds fail: {group}")
    if not np.array_equal(candidate["kpx_group_3"].to_numpy(), base["kpx_group_3"].to_numpy()):
        raise AssertionError("G3 identity failed")
    if not np.array_equal(candidate.iloc[-1].to_numpy(), base.iloc[-1].to_numpy()):
        raise AssertionError("terminal identity failed")
    return candidate


def delta(actual: pd.DataFrame, base: pd.DataFrame, candidate: pd.DataFrame, cols: tuple[str, ...]) -> dict[str, float]:
    old = score_details(actual.loc[:, list(cols)], base.loc[:, list(cols)], target_cols=cols).as_dict()
    new = score_details(actual.loc[:, list(cols)], candidate.loc[:, list(cols)], target_cols=cols).as_dict()
    return {key: float(new[key] - old[key]) for key in ("total_score", "one_minus_nmae", "ficr")}


def evaluate(labels: pd.DataFrame, base: pd.DataFrame, candidate: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    mixed: dict[str, dict[str, float]] = {}
    groups: dict[str, dict[str, dict[str, float]]] = {group: {} for group in TARGET_COLS}
    for name, (start_text, end_text) in WINDOWS.items():
        idx = labels.index[(labels.index >= pd.Timestamp(start_text)) & (labels.index <= pd.Timestamp(end_text))]
        mixed[name] = delta(labels.loc[idx], base.loc[idx], candidate.loc[idx], tuple(TARGET_COLS))
        for group in TARGET_COLS:
            groups[group][name] = delta(labels.loc[idx], base.loc[idx], candidate.loc[idx], (group,))
    positive = sum(mixed[name]["total_score"] > 0.0 for name in WINDOWS)
    monthly_min = min(mixed[name]["total_score"] for name in ("Jul", "Aug", "Sep", "Oct", "Nov", "Dec"))
    checks = {
        "mixed_h2_score": mixed["H2"]["total_score"] >= 0.00075,
        "mixed_h2_ficr": mixed["H2"]["ficr"] >= 0.001,
        "mixed_h2_nmae": mixed["H2"]["one_minus_nmae"] >= 0.0,
        "g1_g2_h2_positive": min(groups[g]["H2"]["total_score"] for g in ("kpx_group_1", "kpx_group_2")) > 0.0,
        "q3_q4_nonnegative": min(mixed[q]["total_score"] for q in ("Q3", "Q4")) >= 0.0,
        "monthly_floor": monthly_min >= -0.0015,
        "positive_segments": positive >= 7,
    }
    return {"mixed": mixed, "groups": groups}, {"checks": checks, "pass": bool(all(checks.values())), "positive_segments": positive, "monthly_min": monthly_min}


def write_csv(prediction: pd.DataFrame, path: Path) -> dict[str, Any]:
    verify(SAMPLE)
    sample = pd.read_csv(SAMPLE, encoding="utf-8-sig", dtype="string", keep_default_na=False)
    if tuple(sample.columns) != ("forecast_id", "forecast_kst_dtm", *TARGET_COLS) or len(sample) != len(TEST_INDEX):
        raise ValueError("sample schema differs")
    idx = pd.DatetimeIndex(pd.to_datetime(sample["forecast_kst_dtm"], errors="raise"), name="forecast_kst_dtm")
    if not idx.equals(TEST_INDEX) or not prediction.index.equals(TEST_INDEX):
        raise ValueError("sample index differs")
    output = sample.loc[:, ["forecast_id", "forecast_kst_dtm"]].copy()
    for group in TARGET_COLS:
        output[group] = prediction[group].to_numpy(dtype=np.float64)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    output.to_csv(temporary, index=False, encoding="utf-8-sig", float_format="%.6f", lineterminator="\n")
    os.replace(temporary, path)
    raw = path.read_bytes()
    if not raw.startswith(b"\xef\xbb\xbf") or b"\r\n" in raw:
        raise AssertionError("CSV encoding differs")
    rendered = pd.read_csv(path, encoding="utf-8-sig", dtype="string", keep_default_na=False)
    if not rendered.loc[:, ["forecast_id", "forecast_kst_dtm"]].equals(sample.loc[:, ["forecast_id", "forecast_kst_dtm"]]):
        raise AssertionError("CSV identifiers differ")
    for group in TARGET_COLS:
        expected = pd.Series([f"{value:.6f}" for value in prediction[group].to_numpy(dtype=np.float64)], dtype="string")
        if not rendered[group].reset_index(drop=True).equals(expected):
            raise AssertionError(f"CSV numeric rendering differs: {group}")
    return {**record(path), "rows": len(rendered), "utf8_bom": True, "six_decimals": True, "sample_identity": True}


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"fresh output required: {OUTPUT}")
    if SIDECAR.read_text(encoding="ascii").split()[0] != EXPECTED[CONFIG]:
        raise RuntimeError("config sidecar mismatch")
    for path in (CONFIG, FEEDBACK, JOINT_MANIFEST_2024, JOINT_2024, CONSENSUS_MANIFEST_2024, CONSENSUS_2024, BASE_2024):
        verify(path)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config["status"] != "frozen_before_composite_candidate_metric_or_2025_parent_array_access":
        raise RuntimeError("config is not frozen")
    OUTPUT.mkdir(parents=True)

    base24 = read_frame(BASE_2024, H2_INDEX)
    joint24 = read_frame(JOINT_2024, H2_INDEX)
    consensus24 = read_frame(CONSENSUS_2024, H2_INDEX)
    candidate24 = compose(base24, joint24, consensus24)
    base_path = OUTPUT / "validation/base_2024_h2.parquet"
    candidate_path = OUTPUT / "validation/candidate_2024_h2.parquet"
    atomic_parquet(base_path, base24)
    atomic_parquet(candidate_path, candidate24)
    lock = OUTPUT / "candidate_before_label_access.json"
    atomic_json(lock, {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "single_posthoc_disjoint_candidate_locked_before_labels",
        "preregister": record(CONFIG),
        "feedback_aggregate_only": record(FEEDBACK),
        "joint_parent": record(JOINT_2024),
        "consensus_parent": record(CONSENSUS_2024),
        "base": record(base_path),
        "candidate": record(candidate_path),
        "formula": "G1=joint exact; G2=consensus exact; G3=base exact; terminal=base exact",
        "candidate_count": 1,
        "label_values_read_before_lock": 0,
        "year_2025_parent_arrays_read_before_gate": 0,
    })

    verify(LABELS)
    labels = pd.read_csv(LABELS, encoding="utf-8-sig", parse_dates=["kst_dtm"]).set_index("kst_dtm")
    labels.index = pd.DatetimeIndex(labels.index, name="forecast_kst_dtm")
    labels = labels.loc[H2_INDEX, list(TARGET_COLS)].astype(np.float64)
    label_access = OUTPUT / "label_access_after_candidate_lock.json"
    atomic_json(label_access, {"candidate_lock": record(lock), "labels": record(LABELS), "selected_rows": len(labels)})
    comparisons, gate = evaluate(labels, base24, candidate24)
    results = OUTPUT / "validation_results.json"
    atomic_json(results, {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_status": "posthoc_selection_unsafe",
        "comparisons": comparisons,
        "gate": gate,
        "year_2025_parent_arrays_read_before_gate": 0,
        "no_alternative_ownership_or_rescue": True,
    })
    outputs = [base_path, candidate_path, lock, label_access, results]
    csv_record: dict[str, Any] | None = None
    if gate["pass"]:
        for path in (JOINT_MANIFEST_2025, JOINT_2025, CONSENSUS_MANIFEST_2025, CONSENSUS_2025, BASE_2025, SAMPLE):
            verify(path)
        base25 = read_frame(BASE_2025, TEST_INDEX)
        joint25 = read_frame(JOINT_2025, TEST_INDEX)
        consensus25 = read_frame(CONSENSUS_2025, TEST_INDEX)
        if not np.array_equal(joint25["kpx_group_3"].to_numpy(), base25["kpx_group_3"].to_numpy()):
            raise AssertionError("joint parent G3 is not base identity")
        if not np.array_equal(consensus25["kpx_group_1"].to_numpy(), base25["kpx_group_1"].to_numpy()):
            raise AssertionError("consensus parent G1 is not base identity")
        prediction = compose(base25, joint25, consensus25)
        prediction_path = OUTPUT / "prediction_2025.parquet"
        atomic_parquet(prediction_path, prediction)
        csv_path = OUTPUT / "multi_nwp_disjoint_g1joint_g2consensus_2025.csv"
        csv_record = write_csv(prediction, csv_path)
        final_lock = OUTPUT / "final_lock.json"
        atomic_json(final_lock, {
            "status": "ready_posthoc_selection_unsafe",
            "validation_results": record(results),
            "joint_final_manifest": record(JOINT_MANIFEST_2025),
            "consensus_final_manifest": record(CONSENSUS_MANIFEST_2025),
            "base": record(BASE_2025),
            "prediction": record(prediction_path),
            "csv": csv_record,
            "g3_identity": bool(np.array_equal(prediction["kpx_group_3"].to_numpy(), base25["kpx_group_3"].to_numpy())),
            "terminal_identity": bool(np.array_equal(prediction.iloc[-1].to_numpy(), base25.iloc[-1].to_numpy())),
        })
        outputs.extend([prediction_path, csv_path, final_lock])
    else:
        rejection = OUTPUT / "REJECTED_NO_2025_PARENT_ARRAY_READ_NO_CSV.json"
        atomic_json(rejection, {"status": "rejected", "results": record(results), "year_2025_parent_array_reads": 0, "csv_created": False})
        outputs.append(rejection)

    manifest = OUTPUT / "manifest.json"
    atomic_json(manifest, {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": EXPERIMENT,
        "selection_status": "posthoc_selection_unsafe",
        "preregister": record(CONFIG),
        "gate_pass": gate["pass"],
        "csv_created": csv_record is not None,
        "outputs_excluding_manifest_and_sidecar": [record(path) for path in outputs],
    })
    (OUTPUT / "manifest.sha256").write_text(f"{sha256(manifest)}  manifest.json\n", encoding="ascii")
    print(json.dumps({"gate": gate, "mixed": comparisons["mixed"], "csv": csv_record, "manifest": record(manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

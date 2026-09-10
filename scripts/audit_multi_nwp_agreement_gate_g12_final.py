"""Independent read-only replay audit for the final agreement-gated CSV."""

from __future__ import annotations

import argparse
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

from src.metric import CAPACITY_KWH, TARGET_COLS
from src.multi_nwp_agreement_gate import SOURCE_ORDER, agreement_gated_increment


VALIDATION = ROOT / "artifacts/postgate/multi_nwp_agreement_gate_g12_v1"
FINAL = VALIDATION / "final"
CONFIG = ROOT / "configs/multi_nwp_agreement_gate_g12_preregister_v1.json"
CONFIG_SIDECAR = CONFIG.with_suffix(".sha256")
SAMPLE = Path(r"data/local/open/sample_submission.csv")
PARENT_CSV = ROOT / "artifacts/postgate/multi_nwp_joint_g12_posthoc_rescue_final_2025_v1/multi_nwp_joint_g12_posthoc_rescue_recent097_2025.csv"
BASELINE = ROOT / "artifacts/postgate/multi_nwp_joint_g12_posthoc_rescue_final_2025_v1/base_scale097_2025.parquet"
JOINT = ROOT / "artifacts/postgate/multi_nwp_joint_g12_posthoc_rescue_final_2025_v1/final_joint_increment_cf_2025.parquet"
SOURCE_INCREMENT_PATHS = {
    "ecmwf": FINAL / "predictions/ecmwf_increment_cf_2025.parquet",
    "icon": FINAL / "predictions/icon_increment_cf_2025.parquet",
    "gfs": FINAL / "predictions/gfs_increment_cf_2025.parquet",
}
GATED = FINAL / "predictions/agreement_gated_joint_increment_cf_2025.parquet"
CONFIDENCE = FINAL / "diagnostics/agreement_confidence_2025.parquet"
COUNT = FINAL / "diagnostics/source_sign_agreement_count_2025.parquet"
FINAL_PARQUET = FINAL / "multi_nwp_agreement_gate_g12_recent097_2025.parquet"
FINAL_CSV = FINAL / "multi_nwp_agreement_gate_g12_recent097_2025.csv"
FINAL_AUDIT = FINAL / "final_audit.json"
MANIFEST = FINAL / "manifest.json"
MANIFEST_SIDECAR = FINAL / "manifest.sha256"
TEST_INDEX = pd.date_range(
    "2025-01-01 01:00", "2026-01-01 00:00", freq="h", name="forecast_kst_dtm"
)
TERMINAL = pd.Timestamp("2026-01-01 00:00:00")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def exact_float_frame(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    return (
        left.index.equals(right.index)
        and left.columns.equals(right.columns)
        and np.array_equal(
            np.ascontiguousarray(left.to_numpy(dtype=np.float64)).view(np.uint64),
            np.ascontiguousarray(right.to_numpy(dtype=np.float64)).view(np.uint64),
        )
    )


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/audits/multi_nwp_agreement_gate_g12_final_independent_v1.json",
    )
    args = parser.parse_args()

    checks: dict[str, bool] = {}
    config_hash = sha256(CONFIG)
    checks["config_sidecar_exact"] = CONFIG_SIDECAR.read_text(encoding="ascii").split()[0] == config_hash
    promotion = json.loads((VALIDATION / "promotion_lock.json").read_text(encoding="utf-8"))
    stage = json.loads((VALIDATION / "stage_results.json").read_text(encoding="utf-8"))
    checks["promotion_chain_exact"] = (
        promotion["separate_final_stage_authorized"]
        and promotion["preregister_sha256"] == config_hash
        and promotion["stage_results_sha256"] == sha256(VALIDATION / "stage_results.json")
        and stage["gate"]["pass"]
        and all(stage["gate"]["checks"].values())
    )

    manifest_hash = sha256(MANIFEST)
    checks["manifest_sidecar_exact"] = MANIFEST_SIDECAR.read_text(encoding="ascii").split()[0] == manifest_hash
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    checks["manifest_no_unlisted_files"] = manifest["unlisted_files_before_manifest"] == []
    checks["all_manifest_output_hashes_exact"] = all(
        Path(item["path"]).stat().st_size == item["bytes"]
        and sha256(Path(item["path"])) == item["sha256"]
        for item in manifest["outputs_excluding_manifest_and_sidecar"]
    )

    baseline = pd.read_parquet(BASELINE).loc[TEST_INDEX, list(TARGET_COLS)].astype(np.float64)
    joint = pd.read_parquet(JOINT).loc[TEST_INDEX, list(TARGET_COLS)].astype(np.float64)
    sources = {
        name: pd.read_parquet(path).loc[TEST_INDEX, list(TARGET_COLS)].astype(np.float64)
        for name, path in SOURCE_INCREMENT_PATHS.items()
    }
    gated_expected, confidence_expected, count_expected = agreement_gated_increment(joint, sources)
    gated_expected["kpx_group_3"] = 0.0
    confidence_expected["kpx_group_3"] = 0.0
    count_expected["kpx_group_3"] = 0
    gated = pd.read_parquet(GATED).loc[TEST_INDEX, list(TARGET_COLS)]
    confidence = pd.read_parquet(CONFIDENCE).loc[TEST_INDEX, list(TARGET_COLS)]
    count = pd.read_parquet(COUNT).loc[TEST_INDEX, list(TARGET_COLS)]
    checks["gated_increment_float64_bit_replay"] = exact_float_frame(gated, gated_expected)
    checks["confidence_float64_bit_replay"] = exact_float_frame(confidence, confidence_expected)
    checks["agreement_count_exact_replay"] = np.array_equal(
        count.to_numpy(dtype=np.int8), count_expected.to_numpy(dtype=np.int8)
    )
    checks["gate_never_enlarges_joint"] = bool(
        (np.abs(gated.to_numpy()) <= np.abs(joint.to_numpy()) + 1e-15).all()
    )

    final_expected = baseline.copy()
    for group in ("kpx_group_1", "kpx_group_2"):
        final_expected[group] = np.clip(
            baseline[group].to_numpy(dtype=np.float64)
            + 0.25 * CAPACITY_KWH[group] * gated[group].to_numpy(dtype=np.float64),
            0.0,
            1.02 * CAPACITY_KWH[group],
        )
    final = pd.read_parquet(FINAL_PARQUET).loc[TEST_INDEX, list(TARGET_COLS)].astype(np.float64)
    checks["final_parquet_float64_bit_formula_replay"] = exact_float_frame(final, final_expected)
    checks["g3_baseline_float64_bit_identity"] = np.array_equal(
        final["kpx_group_3"].to_numpy().view(np.uint64),
        baseline["kpx_group_3"].to_numpy().view(np.uint64),
    )
    checks["terminal_baseline_float64_bit_identity"] = np.array_equal(
        final.loc[[TERMINAL]].to_numpy().view(np.uint64),
        baseline.loc[[TERMINAL]].to_numpy().view(np.uint64),
    )
    checks["bounds_and_finite"] = bool(
        np.isfinite(final.to_numpy()).all()
        and all(
            (final[group] >= 0.0).all()
            and (final[group] <= 1.02 * CAPACITY_KWH[group]).all()
            for group in TARGET_COLS
        )
    )

    sample = pd.read_csv(SAMPLE, encoding="utf-8-sig", dtype="string", keep_default_na=False)
    csv = pd.read_csv(FINAL_CSV, encoding="utf-8-sig", dtype="string", keep_default_na=False)
    checks["csv_bom"] = FINAL_CSV.read_bytes().startswith(b"\xef\xbb\xbf")
    checks["csv_schema_rows_identifiers_exact"] = (
        tuple(csv.columns) == ("forecast_id", "forecast_kst_dtm", *TARGET_COLS)
        and len(csv) == len(TEST_INDEX)
        and csv.loc[:, ["forecast_id", "forecast_kst_dtm"]].equals(
            sample.loc[:, ["forecast_id", "forecast_kst_dtm"]]
        )
    )
    checks["csv_six_decimal_replay"] = all(
        csv[group].reset_index(drop=True).equals(
            pd.Series([f"{value:.6f}" for value in final[group].to_numpy()], dtype="string")
        )
        for group in TARGET_COLS
    )

    parent = pd.read_csv(PARENT_CSV, encoding="utf-8-sig")
    differing_from_parent = {
        group: int(
            np.count_nonzero(
                csv[group].astype(np.float64).to_numpy()
                != parent[group].astype(np.float64).to_numpy()
            )
        )
        for group in TARGET_COLS
    }
    checks["distinct_from_submitted_parent_on_active_groups"] = all(
        differing_from_parent[group] >= 1_000 for group in ("kpx_group_1", "kpx_group_2")
    )
    checks["parent_and_candidate_g3_text_identity"] = csv["kpx_group_3"].equals(
        parent["kpx_group_3"].map(lambda value: f"{float(value):.6f}").astype("string")
    )

    final_audit = json.loads(FINAL_AUDIT.read_text(encoding="utf-8"))
    checks["all_twelve_model_reload_claim"] = final_audit["all_twelve_models_two_reload_prediction_exact"]
    checks["final_no_retune_claim"] = final_audit["no_final_retune_or_fallback"]
    passed = bool(all(checks.values()))
    report = {
        "schema_version": 1,
        "audit_id": "multi_nwp_agreement_gate_g12_final_independent_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "all_checks_passed": passed,
        "validation_gate": stage["gate"],
        "final_csv": record(FINAL_CSV),
        "final_manifest": record(MANIFEST),
        "differing_cells_from_submitted_parent": differing_from_parent,
        "final_confidence_summary": final_audit["diagnostics"],
        "auditor": record(Path(__file__).resolve()),
    }
    if not passed:
        raise AssertionError(f"independent audit failed: {checks}")
    atomic_json(report, args.output.resolve())
    print(f"AUDIT=PASS path={args.output.resolve()} sha256={sha256(args.output.resolve())}")


if __name__ == "__main__":
    main()

"""Bind two independently audited CSVs into an ordered submission-readiness record."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("kpx_group_1", "kpx_group_2", "kpx_group_3")
G12_CSV = (
    ROOT
    / "artifacts/postgate/multi_nwp_joint_g12_posthoc_rescue_final_2025_v1/"
    "multi_nwp_joint_g12_posthoc_rescue_recent097_2025.csv"
)
G23_CSV = (
    ROOT
    / "artifacts/final_multi_nwp_consensus_g23_rescue_v1/predictions/"
    "multi_nwp_consensus_g23_rescue_v1_2025.csv"
)
SAMPLE = Path(r"data/local/open/sample_submission.csv")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main(g12_audit_path: Path, g23_audit_path: Path, output_path: Path) -> None:
    paths = [G12_CSV.resolve(), G23_CSV.resolve(), SAMPLE.resolve()]
    before = {
        str(path): (path.stat().st_size, path.stat().st_mtime_ns, sha256(path))
        for path in paths
    }
    g12_audit_path = g12_audit_path.resolve()
    g23_audit_path = g23_audit_path.resolve()
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    g12_audit = read_json(g12_audit_path)
    g23_audit = read_json(g23_audit_path)
    if not str(g12_audit["verdict"]).startswith("GO_") or not str(
        g23_audit["verdict"]
    ).startswith("GO_"):
        raise AssertionError("both independent audits must have GO verdicts")
    g12_record = g12_audit["checks"]["submission_csv"]["file"]
    g23_record = g23_audit["checks"]["submission_csv"]["file"]
    if file_record(G12_CSV)["sha256"] != g12_record["sha256"]:
        raise AssertionError("G12 CSV differs from its independent audit")
    if file_record(G23_CSV)["sha256"] != g23_record["sha256"]:
        raise AssertionError("G23 CSV differs from its independent audit")

    sample = pd.read_csv(SAMPLE, encoding="utf-8-sig", dtype="string", keep_default_na=False)
    g12 = pd.read_csv(G12_CSV, encoding="utf-8-sig", dtype="string", keep_default_na=False)
    g23 = pd.read_csv(G23_CSV, encoding="utf-8-sig", dtype="string", keep_default_na=False)
    expected_columns = ("forecast_id", "forecast_kst_dtm", *TARGETS)
    if tuple(sample.columns) != expected_columns or tuple(g12.columns) != expected_columns or tuple(
        g23.columns
    ) != expected_columns:
        raise AssertionError("schema differs among sample/G12/G23")
    if len(sample) != 8760 or len(g12) != 8760 or len(g23) != 8760:
        raise AssertionError("row count differs among sample/G12/G23")
    identifiers = ["forecast_id", "forecast_kst_dtm"]
    if not g12.loc[:, identifiers].equals(sample.loc[:, identifiers]) or not g23.loc[
        :, identifiers
    ].equals(sample.loc[:, identifiers]):
        raise AssertionError("submission identifiers differ from sample")

    by_group: dict[str, Any] = {}
    changed_masks: list[np.ndarray] = []
    for group in TARGETS:
        first_text = g12[group].to_numpy(dtype=str)
        second_text = g23[group].to_numpy(dtype=str)
        changed = first_text != second_text
        changed_masks.append(changed)
        first = g12[group].to_numpy(dtype=np.float64)
        second = g23[group].to_numpy(dtype=np.float64)
        delta = second - first
        by_group[group] = {
            "changed_cells_at_submission_precision": int(changed.sum()),
            "unchanged_cells_at_submission_precision": int((~changed).sum()),
            "changed_fraction": float(changed.mean()),
            "mean_second_minus_first_kwh": float(delta.mean()),
            "mean_abs_difference_kwh": float(np.abs(delta).mean()),
            "max_abs_difference_kwh": float(np.abs(delta).max()),
            "first_min_kwh": float(first.min()),
            "first_max_kwh": float(first.max()),
            "second_min_kwh": float(second.min()),
            "second_max_kwh": float(second.max()),
        }
    changed_matrix = np.column_stack(changed_masks)
    changed_cells = int(changed_matrix.sum())
    changed_rows = int(changed_matrix.any(axis=1).sum())
    if sha256(G12_CSV) == sha256(G23_CSV) or changed_cells == 0:
        raise AssertionError("the two submission files are duplicates")

    after = {
        str(path): (path.stat().st_size, path.stat().st_mtime_ns, sha256(path))
        for path in paths
    }
    if before != after:
        raise AssertionError("CSV/sample input changed during readiness audit")

    readiness = {
        "schema_version": 1,
        "readiness_id": "two_submission_g12_first_g23_second_readiness_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": "READY_TWO_DISTINCT_AUDITED_CSVS_G12_FIRST_G23_SECOND",
        "scope_note": (
            "Both candidates are posthoc-selection-unsafe; readiness means file integrity and "
            "ordered diversity, not a guaranteed leaderboard improvement."
        ),
        "independent_audits": {
            "g12": {**file_record(g12_audit_path), "verdict": g12_audit["verdict"]},
            "g23": {**file_record(g23_audit_path), "verdict": g23_audit["verdict"]},
        },
        "common_contract": {
            "rows": 8760,
            "columns": list(expected_columns),
            "identifier_and_timestamp_exact_sample_for_both": True,
            "sample": file_record(SAMPLE),
            "both_single_utf8_bom_lf_six_decimals_and_bounds_audited": True,
            "inputs_nonmutated_during_readiness_check": True,
        },
        "nonduplicate_and_changed_cells": {
            "passed": True,
            "byte_sha256_distinct": True,
            "target_matrix_distinct_at_six_decimals": True,
            "target_cells_total": int(8760 * len(TARGETS)),
            "changed_target_cells_at_submission_precision": changed_cells,
            "unchanged_target_cells_at_submission_precision": int(
                8760 * len(TARGETS) - changed_cells
            ),
            "changed_target_cell_fraction": float(changed_cells / (8760 * len(TARGETS))),
            "rows_with_at_least_one_changed_target": changed_rows,
            "rows_with_no_changed_target": int(8760 - changed_rows),
            "by_group": by_group,
        },
        "submission_order": [
            {
                "position": 1,
                "candidate_id": "multi_nwp_joint_g12_posthoc_rescue_final_2025_v1",
                "role": "G1/G2 joint-NWP paired-increment hedge; G3 baseline identity",
                "csv": file_record(G12_CSV),
                "action": "submit_first",
            },
            {
                "position": 2,
                "candidate_id": "multi_nwp_consensus_g23_rescue_v1",
                "role": "G2/G3 three-source consensus paired-increment hedge; G1 baseline identity",
                "csv": file_record(G23_CSV),
                "action": "submit_second",
            },
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)
    print(
        json.dumps(
            {
                "verdict": readiness["verdict"],
                "readiness_path": str(output_path),
                "readiness_bytes": output_path.stat().st_size,
                "readiness_sha256": sha256(output_path),
                "changed_cells": changed_cells,
                "changed_rows": changed_rows,
                "by_group_changed": {
                    group: by_group[group]["changed_cells_at_submission_precision"]
                    for group in TARGETS
                },
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("g12_audit", type=Path)
    parser.add_argument("g23_audit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.g12_audit, args.g23_audit, args.output)

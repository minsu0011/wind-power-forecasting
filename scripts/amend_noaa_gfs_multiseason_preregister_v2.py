#!/usr/bin/env python
"""Append-only coordinate, missingness and target-free-estimator prereg amendment."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
ROOT_DEFAULT = REPO / "artifacts" / "baram2026_ncei_scada_longrun_20260810_v2"
INFO_XLSX = Path(r"data/local/open/info.xlsx")
INFO_SHA256 = "89e83a52e0eb2ce367a3573a96d6795ed4b4d4ac624965cb3530beec0cbd2bd6"
TURBINE_CONFIG = REPO / "configs" / "copernicus_dem_directional_exposure_paired_increment_preregister_v1.json"
PARENT_MANIFEST_SHA256 = "ac9ab919b6eba7ba4c6f36170b7035c7e0f21e1bf16dac8e19b4aa6a993b0bf2"
DMS = re.compile(
    r"^(\d+)°(\d+)'([0-9.]+)\"([NS])\s+(\d+)°(\d+)'([0-9.]+)\"([EW])$"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path, base: Path | None = None) -> dict[str, Any]:
    return {
        "path": path.relative_to(base).as_posix() if base else str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def dms_to_decimal(value: str) -> tuple[float, float]:
    match = DMS.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"malformed coordinate: {value!r}")
    latitude = int(match[1]) + int(match[2]) / 60 + float(match[3]) / 3600
    longitude = int(match[5]) + int(match[6]) / 60 + float(match[7]) / 3600
    if match[4] == "S":
        latitude *= -1
    if match[8] == "W":
        longitude *= -1
    return latitude, longitude


def authoritative_sites() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if sha256_file(INFO_XLSX) != INFO_SHA256:
        raise RuntimeError("authoritative info.xlsx identity mismatch")
    import pandas as pd

    frame = pd.read_excel(INFO_XLSX, sheet_name="info", header=3)
    if len(frame) != 17:
        raise RuntimeError(f"expected 17 turbine rows, observed {len(frame)}")
    groups = frame["KPX그룹"].ffill().astype(int)
    config = json.loads(TURBINE_CONFIG.read_text(encoding="utf-8"))["turbines"]["rows"]
    if len(config) != 17:
        raise RuntimeError("frozen turbine config does not have 17 rows")
    rows = []
    maximum_coordinate_abs_diff = 0.0
    for index, (row, group) in enumerate(zip(frame.to_dict("records"), groups), start=1):
        latitude, longitude = dms_to_decimal(str(row["좌표(Google)"]))
        configured = config[index - 1]
        expected_group = f"kpx_group_{int(group)}"
        coordinate_diff = max(
            abs(latitude - float(configured["latitude"])),
            abs(longitude - float(configured["longitude"])),
        )
        maximum_coordinate_abs_diff = max(maximum_coordinate_abs_diff, coordinate_diff)
        exact = (
            int(configured["id"]) == index
            and configured["group"] == expected_group
            and abs(float(configured["capacity_mw"]) - float(row["설비용량(MW)"])) <= 1e-12
            and coordinate_diff <= 1e-12
        )
        if not exact:
            raise RuntimeError(
                f"info.xlsx/config mismatch at turbine {index}: info={(expected_group, latitude, longitude, row['설비용량(MW)'])}, config={configured}"
            )
        rows.append(
            {
                "site_id": index,
                "stage": int(row["단계"]),
                "site_name": str(row["명칭"]),
                "manufacturer": str(row["제작사"]),
                "model": str(row["모델명"]),
                "unit_within_stage": int(row["호기"]),
                "group": expected_group,
                "latitude": latitude,
                "longitude": longitude,
                "capacity_mw": float(row["설비용량(MW)"]),
                "hub_height_m": float(row["Hub Height(m)"]),
                "raw_coordinate": str(row["좌표(Google)"]),
            }
        )
    group_capacity = {
        group: sum(row["capacity_mw"] for row in rows if row["group"] == group)
        for group in sorted({row["group"] for row in rows})
    }
    return rows, {
        "row_count": len(rows),
        "all_config_rows_match": True,
        "coordinate_tolerance_degrees": 1e-12,
        "maximum_coordinate_abs_diff_degrees": maximum_coordinate_abs_diff,
        "group_capacity_mw": group_capacity,
    }


def run(root: Path) -> dict[str, Any]:
    root = root.resolve()
    parent_manifest = root / "manifest_preregister_v1.json"
    parent_plan = root / "prereg" / "target_free_multiseason_sampling_plan_v1.json"
    amendment_path = root / "prereg" / "target_free_multiseason_amendment_v2.json"
    manifest_path = root / "manifest_preregister_v2.json"
    coordinate_path = root / "prereg" / "authoritative_turbine_coordinate_lock_v1.json"
    conflicts = [str(path) for path in (amendment_path, manifest_path, coordinate_path) if path.exists()]
    if conflicts:
        raise FileExistsError(f"append-only prereg amendment preflight failed: {conflicts}")
    if sha256_file(parent_manifest) != PARENT_MANIFEST_SHA256:
        raise RuntimeError("parent prereg manifest identity changed")
    sites, coordinate_audit = authoritative_sites()
    coordinate_lock = {
        "artifact_type": "AUTHORITATIVE_TURBINE_COORDINATE_LOCK",
        "authoritative_source": identity(INFO_XLSX),
        "comparison_config": identity(TURBINE_CONFIG),
        "audit": coordinate_audit,
        "sites": sites,
        "frozen_before_index_census_or_decode": True,
        "coordinate_change_after_values_forbidden": True,
        "labels_read": False,
    }
    write_json_exclusive(coordinate_path, coordinate_lock)

    amendment = {
        "artifact_type": "TARGET_FREE_MULTISEASON_PREREGISTRATION_APPEND_ONLY_AMENDMENT",
        "schema_version": 2,
        "parent_plan": identity(parent_plan, root),
        "parent_manifest": identity(parent_manifest, root),
        "coordinate_lock": identity(coordinate_path, root),
        "coordinate_authority": {
            "info_xlsx_sha256": INFO_SHA256,
            "config_must_match_all_17_rows": True,
            "on_mismatch": "STOP_BEFORE_RAW_LAUNCH",
            "spatial_method_unchanged": "bilinear site interpolation followed by capacity-weighted group mean",
        },
        "family_missingness_and_salvage": {
            "PBL_HEIGHT": {
                "required_finite_fraction": 1.0,
                "physical_range_inclusive": [0.0, 10_000.0],
                "on_failure": "REJECT_PBL_HEIGHT_ONLY",
            },
            "LOW_LEVEL_ISOBARIC_WIND_PROFILE": {
                "required_finite_fraction": 1.0,
                "component_absolute_max_mps": 150.0,
                "below_ground_missing_is_failure": True,
                "all_eight_locked_components_required": True,
                "posthoc_level_dropping_forbidden": True,
                "on_failure": "REJECT_LOW_LEVEL_ISOBARIC_WIND_PROFILE_ONLY",
            },
            "independence_rule": "A vertical-profile failure never automatically rejects PBL_HEIGHT; a PBL_HEIGHT failure never salvages or rejects the vertical family. Each locked family is gated independently.",
            "rule_frozen_before_values": True,
        },
        "model_fit_scope_clarification": {
            "competition_generation_target_models": "FORBIDDEN",
            "competition_labels": "FORBIDDEN",
            "target_free_duplicate_estimator": "ALLOWED_ONLY_AFTER_DECODED_MATRIX_SHA_LOCK",
            "estimator_response": "one decoded external weather feature; never generation or competition target",
            "allowed_estimators": [
                {
                    "name": "RIDGE_AFFINE",
                    "pipeline": "StandardScaler + Ridge(alpha=1.0)",
                },
                {
                    "name": "EXTRA_TREES_NONLINEAR",
                    "params": {
                        "n_estimators": 128,
                        "max_depth": 16,
                        "min_samples_leaf": 8,
                        "max_features": 0.8,
                        "random_state": 260810,
                        "n_jobs": 7,
                    },
                },
            ],
            "predictors": "Provided DACON GFS weather columns only, same valid time; no generation/SCADA/labels.",
            "folds": [
                "2022-01-01..2022-06-30",
                "2022-07-01..2022-12-31",
                "2023-01-01..2023-06-30",
                "2023-07-01..2023-12-31",
            ],
            "cross_fit_rule": "Each block is predicted by an estimator trained on the other three blocks.",
        },
        "novelty_thresholds": {
            "feature_redundant_if": "max cross-fitted R2 across the two locked estimators >= 0.98 AND corresponding NRMSE <= 0.10",
            "feature_exact_or_affine_duplicate_if": "absolute Pearson >= 0.9999 AND absolute Spearman >= 0.9999 AND affine NRMSE <= 0.01",
            "family_pass_if": "At least half of the family's locked scalar components are not redundant and every family physical/coverage gate passes.",
            "family_fail_if": "Fewer than half of locked scalar components are nonredundant, or its independent physical/coverage gate fails.",
            "no_threshold_change_after_decode": True,
        },
        "stability_thresholds": {
            "expected_hours_per_selected_month_year": 48,
            "expected_site_rows_per_hour": 17,
            "expected_group_rows_per_hour": 3,
            "key_coverage_required": 1.0,
            "family_finite_fraction": "as independently frozen above",
            "year_distribution_psi_max": 0.50,
            "monthly_iqr_ratio_allowed": [0.10, 10.0],
            "constant_feature_unique_values_min": 3,
        },
        "full_expansion_gate": {
            "requires_all": [
                "100% selected-object publication and raw-SHA provenance VERIFIED",
                "100% expected date/hour/site/group key coverage",
                "at least one independently gated family passes physical, stability and target-free novelty thresholds",
                "decoded matrices and target-free audit outputs are SHA locked",
                "independent postrun audit PASS",
            ],
            "does_not_authorize": [
                "competition target fit",
                "2024 or 2025 access",
                "submission CSV",
                "full 2022-2025 acquisition without a new preregistration",
            ],
        },
        "index_files_read_at_freeze": 0,
        "external_values_decoded_at_freeze": 0,
        "labels_read": False,
    }
    write_json_exclusive(amendment_path, amendment)
    source = Path(__file__).resolve()
    test = REPO / "tests" / "test_noaa_gfs_multiseason_preregister_amendment_v2.py"
    manifest = {
        "artifact_type": "NOAA_GFS_MULTISEASON_PREREGISTRATION_MANIFEST",
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "parent_manifest": identity(parent_manifest, root),
        "amendment": identity(amendment_path, root),
        "coordinate_lock": identity(coordinate_path, root),
        "reproduction_code": identity(source),
        "test_code": identity(test),
        "existing_files_modified": [],
        "index_files_read": 0,
        "external_values_decoded": 0,
        "network_requests": 0,
        "labels_read": False,
        "models_fit": 0,
        "submission_csv_created": False,
    }
    write_json_exclusive(manifest_path, manifest)
    return {
        "amendment": identity(amendment_path, root),
        "coordinate_lock": identity(coordinate_path, root),
        "manifest": identity(manifest_path, root),
        "coordinate_audit": coordinate_audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(run(args.root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

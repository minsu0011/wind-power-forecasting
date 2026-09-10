#!/usr/bin/env python
"""Freeze the target-free NOAA GFS multiseason sample before index census."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
ROOT_DEFAULT = REPO / "artifacts" / "baram2026_ncei_scada_longrun_20260810_v2"
PREVIOUS_ROOT = (
    REPO
    / "artifacts"
    / "baram2026_ncei_scada_research_20260810_210756"
    / "track_a"
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


def write_text_exclusive(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def operating_days() -> list[str]:
    return [
        date(year, month, day).isoformat()
        for year in (2022, 2023)
        for month in range(1, 13)
        for day in (5, 20)
    ]


def run(root: Path) -> dict[str, Any]:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    targets = {
        "plan": root / "prereg" / "target_free_multiseason_sampling_plan_v1.json",
        "previous_json": root / "audit" / "PREVIOUS_PILOT_APPEND_ONLY_AUDIT.json",
        "previous_md": root / "audit" / "PREVIOUS_PILOT_APPEND_ONLY_AUDIT.md",
        "manifest": root / "manifest_preregister_v1.json",
    }
    conflicts = [str(path) for path in targets.values() if path.exists()]
    if conflicts:
        raise FileExistsError(f"no-overwrite prereg preflight failed: {conflicts}")

    previous_files = [
        PREVIOUS_ROOT / "manifest.json",
        PREVIOUS_ROOT / "manifest_v2.json",
        PREVIOUS_ROOT / "manifest_v3.json",
        PREVIOUS_ROOT / "provenance" / "GFS_PROVENANCE_LEDGER.parquet",
        PREVIOUS_ROOT / "provenance" / "RAW_MANIFEST_SHA256.csv",
        PREVIOUS_ROOT / "audit" / "TRACK_A_DECISION.json",
        PREVIOUS_ROOT / "audit" / "HPBL_CAUSAL_FEASIBILITY.json",
        PREVIOUS_ROOT / "audit" / "HPBL_ECCODES_DECODER_AUDIT.json",
        REPO / "src" / "noaa_gfs_provenance.py",
        REPO / "scripts" / "run_noaa_gfs_provenance_pilot.py",
        REPO / "scripts" / "finalize_noaa_gfs_track_a_phase_c_placeholders.py",
        REPO / "scripts" / "amend_noaa_gfs_track_a_decoder_v3.py",
    ]
    if any(not path.is_file() for path in previous_files):
        raise FileNotFoundError("a previous Track-A artifact/code identity is missing")
    previous_identities = [identity(path) for path in previous_files]
    prior_audit = {
        "artifact_type": "PREVIOUS_TRACK_A_APPEND_ONLY_READ_ONLY_AUDIT",
        "previous_root": str(PREVIOUS_ROOT),
        "verified_identities": previous_identities,
        "lineage": {
            "manifest_v1_sha256": sha256_file(PREVIOUS_ROOT / "manifest.json"),
            "manifest_v2_sha256": sha256_file(PREVIOUS_ROOT / "manifest_v2.json"),
            "manifest_v3_sha256": sha256_file(PREVIOUS_ROOT / "manifest_v3.json"),
            "v2_parent_is_v1": json.loads(
                (PREVIOUS_ROOT / "manifest_v2.json").read_text(encoding="utf-8")
            )["parent_manifest"]["sha256"]
            == sha256_file(PREVIOUS_ROOT / "manifest.json"),
            "v3_parent_is_v2": json.loads(
                (PREVIOUS_ROOT / "manifest_v3.json").read_text(encoding="utf-8")
            )["parent_manifest"]["sha256"]
            == sha256_file(PREVIOUS_ROOT / "manifest_v2.json"),
        },
        "issues": [
            {
                "id": "PILOT_COVERAGE_TOO_SMALL",
                "severity": "BLOCKS_GENERALIZATION",
                "evidence": "Only f028/f039/f051 for one 2023 operating day were raw-SHA and publication verified.",
                "v2_control": "Freeze 48 operating days across every month of 2022 and 2023, then require 100% object/field census before payload launch.",
            },
            {
                "id": "ARCHIVE_PROBES_NOT_CENSUS",
                "severity": "BLOCKS_FULL_EXPANSION",
                "evidence": "The old 2022-2025 result was one July f039 HEAD probe per year.",
                "v2_control": "List exact metadata and parse the exact index for every selected hourly object.",
            },
            {
                "id": "INITIAL_MANIFEST_DID_NOT_BIND_EXECUTED_CODE",
                "severity": "REPRODUCIBILITY_GAP",
                "evidence": "manifest.json inventories artifacts but does not bind the runner source; cleanup code was patched after its run.",
                "v2_control": "Every v2 lock binds executable source and prereg identity before payload launch.",
            },
            {
                "id": "INITIAL_LEDGER_SCHEMA_USED_SEMANTIC_NOT_PROMPT_EXACT_FIELDS",
                "severity": "SCHEMA_GAP",
                "evidence": "Several required concepts were present under different names; exact retrieved_at/source_archive/archive_product fields were absent.",
                "v2_control": "Use exact prompt fields plus object key/ETag/size/range identities for every record.",
            },
            {
                "id": "V2_DECODER_FALSE_NEGATIVE",
                "severity": "CORRECTED_APPEND_ONLY",
                "evidence": "v2 marked decoder unavailable despite direct Python ecCodes support; manifest_v3 corrects and locally decodes the SHA-bound message.",
                "v2_control": "Require a direct ecCodes decode smoke test before raw launch.",
            },
            {
                "id": "NO_INCREMENTAL_INFORMATION_ESTABLISHED",
                "severity": "BLOCKS_MODEL_USE",
                "evidence": "All prior duplicate/incremental/spatial/vertical audits are NOT_RUN/UNDETERMINED.",
                "v2_control": "Build target-free decoded parquet only; no label, fit, score, model or CSV is authorized.",
            },
        ],
        "previous_files_modified": [],
        "labels_read": False,
        "models_fit": 0,
    }
    write_json_exclusive(targets["previous_json"], prior_audit)
    write_text_exclusive(
        targets["previous_md"],
        """# Previous Track-A append-only audit

The original pilot remains immutable and its v1 -> v2 -> v3 hash lineage is valid. It proved three exact HPBL messages, not multiseason coverage or incremental model value.

Material gaps carried into this preregistration are: one-day coverage, non-census archive probes, no runner binding in the first manifest, non-exact ledger field names, the append-only decoder correction, and no completed incremental-information test. Track A v2 therefore remains target-free and cannot fit, score, select, read 2024/2025 arrays, or create a submission CSV.
""",
    )

    days = operating_days()
    candidate_records = [
        {
            "family": "PBL_HEIGHT",
            "selectors": [{"variable": "HPBL", "level": "surface"}],
            "schema_novelty": "DACON GFS has PBL U/V/VRATE but no HPBL height.",
        },
        {
            "family": "LOW_LEVEL_ISOBARIC_WIND_PROFILE",
            "selectors": [
                {"variable": variable, "level": f"{level} mb"}
                for level in (925, 950, 975, 1000)
                for variable in ("UGRD", "VGRD")
            ],
            "schema_novelty": "DACON GFS wind levels are 850/700/500 mb; 925/950/975/1000 mb are absent.",
        },
    ]
    plan = {
        "artifact_type": "TARGET_FREE_MULTISEASON_GFS_PREREGISTRATION",
        "schema_version": 1,
        "experiment_id": "noaa_gfs_dminus2_12z_multiseason_target_free_v2",
        "created_before_index_census": True,
        "operating_days_kst": days,
        "date_rule": "For each month of 2022 and 2023 select calendar days 05 and 20, ordered year/month/day.",
        "operating_day_count": len(days),
        "hour_rule": "D 01:00 through D+1 00:00 KST inclusive",
        "hourly_object_count": len(days) * 24,
        "run_policy": "D-2 12Z",
        "forecast_hours": list(range(28, 52)),
        "cutoff_policy": "D-1 14:00 Asia/Seoul = D-1 05:00 UTC",
        "source": {
            "archive": "NOAA Global Forecast System public NODD S3 bucket",
            "bucket": "noaa-gfs-bdp-pds",
            "product": "gfs.t12z.pgrb2.0p25.fFFF",
            "allowed_domains": [
                "noaa-gfs-bdp-pds.s3.amazonaws.com",
                "www.ncei.noaa.gov",
                "www.nco.ncep.noaa.gov",
                "docs.aws.amazon.com",
                "registry.opendata.aws",
            ],
            "open_meteo_forbidden": True,
        },
        "candidate_families": candidate_records,
        "excluded_before_census": [
            {
                "fields": "20/30/40/50 m UGRD/VGRD",
                "reason": "Dense near-surface interpolation-like levels are excluded until a truly incremental family is independently justified.",
            },
            {
                "fields": "PBL UGRD/VGRD/VRATE and all existing DACON schema fields",
                "reason": "Already represented in provided DACON GFS schema; no duplicate payload acquisition.",
            },
            {
                "fields": "TMP/RH/HGT at new pressure levels",
                "reason": "Keep the first family minimal; only low-level U/V is admitted in v1.",
            },
        ],
        "field_census_gate": {
            "required": "Every selector occurs exactly once in every one of 1,152 .idx files.",
            "publication": "Exact full-object ListObjectsV2 Key/LastModified/ETag/Size must exist and LastModified <= cutoff for every object.",
            "raw_launch_requires_100_percent_pass": True,
            "any_missing_duplicate_or_after_cutoff": "STOP_NO_RAW_LAUNCH",
        },
        "download_contract": {
            "index_first": True,
            "raw_mode": "HTTP Range only, one complete indexed GRIB message per selector",
            "complete_global_object_download_forbidden": True,
            "resume": "Existing files are accepted only after exact size and SHA match against durable manifest; .part is exclusive and atomically renamed.",
            "no_overwrite": True,
            "concurrency": 8,
            "retry_attempts": 4,
            "retry_backoff_seconds": [1, 2, 4],
            "max_index_census_transfer_bytes": 100_000_000,
            "max_raw_payload_bytes": 20_000_000_000,
            "max_total_http_requests": 20_000,
            "minimum_free_disk_after_reserved_raw_bytes": 200_000_000_000,
        },
        "decode_contract": {
            "decoder": "python eccodes direct",
            "smoke_test_before_launch": True,
            "coordinates": "17 frozen Korean turbine sites from copernicus_dem_directional_exposure_paired_increment_preregister_v1.json",
            "spatial_method": "deterministic bilinear interpolation on regular_ll grid, then capacity-weighted group mean",
            "outputs": ["site_values_target_free.parquet", "group_values_target_free.parquet"],
            "target_columns_forbidden": True,
        },
        "provenance_required_fields": [
            "source_archive",
            "archive_product",
            "run_init_utc",
            "forecast_hour",
            "valid_time_utc",
            "retrieval_url_or_request_id",
            "retrieved_at",
            "raw_filename",
            "raw_sha256",
            "official_metadata",
            "publication_evidence_type",
            "publication_evidence_reference",
            "cutoff_utc",
            "cutoff_margin",
            "status",
        ],
        "forbidden": {
            "competition_labels_read": True,
            "2024_arrays_read": True,
            "2025_arrays_read": True,
            "model_fit": True,
            "score_computation": True,
            "feature_selection_from_target": True,
            "submission_csv": True,
            "gpu": True,
        },
        "postdecode_target_free_gates": [
            "100% rows provenance VERIFIED",
            "100% expected operating-day/hour/group/site keys",
            "all values finite and physically audited",
            "monthly availability and distribution stability reported without targets",
            "no full-expansion/model authorization without a separate preregistration",
        ],
    }
    write_json_exclusive(targets["plan"], plan)

    source = Path(__file__).resolve()
    test = REPO / "tests" / "test_noaa_gfs_multiseason_preregister_v2.py"
    manifest = {
        "artifact_type": "NOAA_GFS_MULTISEASON_PREREGISTRATION_MANIFEST",
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "sampling_plan": identity(targets["plan"], root),
        "previous_pilot_audit": identity(targets["previous_json"], root),
        "previous_pilot_audit_markdown": identity(targets["previous_md"], root),
        "reproduction_code": identity(source),
        "test_code": identity(test),
        "index_files_read": 0,
        "network_requests": 0,
        "downloaded_bytes": 0,
        "labels_read": False,
        "models_fit": 0,
        "submission_csv_created": False,
    }
    write_json_exclusive(targets["manifest"], manifest)
    return {
        "root": str(root),
        "sampling_plan": identity(targets["plan"], root),
        "manifest": identity(targets["manifest"], root),
        "operating_days": len(days),
        "hourly_objects": len(days) * 24,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(run(args.root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


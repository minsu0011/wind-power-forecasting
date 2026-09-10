from __future__ import annotations

import json
import hashlib
import shutil
import sys
import threading
import time
import types
from pathlib import Path

import pandas as pd
import pytest

from scripts import audit_noaa_gfs_multiseason_raw_postrun_v1 as subject


def synthetic_decoder(info: object, sites: object) -> dict[str, object]:
    row = info["row"]
    feature = f"{row['variable']}_{str(row['level']).replace(' ', '')}"
    feature_index = subject.FEATURE_COLUMNS.index(feature)
    base = 100.0 if feature_index == 0 else float(feature_index)
    return {
        "feature": feature,
        "site_values": [base + index for index, _site in enumerate(sites)],
        "metadata_verified": True,
        "single_message_verified": True,
    }


def audit_fixture(
    root: Path, runner: Path, contract: subject.AuditContract
) -> dict[str, object]:
    return subject.audit(
        root,
        contract=contract,
        runner_path=runner,
        _test_message_decoder=synthetic_decoder,
    )


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def artifact(path: Path, root: Path | None = None) -> dict[str, object]:
    return subject.identity(path, root)


def snapshot(root: Path) -> dict[str, tuple[int, str]]:
    return {
        path.relative_to(root).as_posix(): (path.stat().st_size, subject.sha256_file(path))
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def rebind_canonical_transaction(root: Path) -> None:
    """Refresh only the producer-style hash envelopes after a semantic tamper."""

    outputs = {
        name: root / relative for name, relative in subject.OUTPUT_RELATIVE_PATHS.items()
    }
    lock = json.loads(outputs["lock"].read_text(encoding="utf-8"))
    for field, output_name in {
        "raw_manifest_parquet": "raw_parquet",
        "raw_manifest_csv": "raw_csv",
        "site_matrix": "site",
        "group_matrix": "group",
        "physical_and_coverage_audit": "audit",
        "access_ledger": "access",
    }.items():
        lock[field] = artifact(outputs[output_name], root)
    write_json(outputs["lock"], lock)

    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    manifest["decoded_matrix_lock"] = artifact(outputs["lock"], root)
    if "request_event_inventory" in manifest:
        manifest["request_event_inventory"] = [
            artifact(path, root)
            for path in sorted((root / "raw" / "request_events").glob("*.json"))
        ]
    if "progress_checkpoints" in manifest:
        manifest["progress_checkpoints"] = [
            artifact(path, root)
            for path in sorted((root / "raw" / "progress").glob("*.json"))
        ]
    write_json(outputs["manifest"], manifest)

    plan_path = next((root / "raw" / "output_transactions").glob("*__plan.json"))
    commit_path = next(
        (root / "raw" / "output_transactions").glob("*__committed.json")
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    for item in plan["items"]:
        current = outputs[item["name"]]
        item["size_bytes"] = current.stat().st_size
        item["sha256"] = subject.sha256_file(current)
    write_json(plan_path, plan)
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    commit["plan"] = artifact(plan_path, root)
    commit["outputs"] = {name: artifact(path, root) for name, path in outputs.items()}
    write_json(commit_path, commit)


def mutate_recovery_provenance_and_rebind(
    root: Path, mutation: object
) -> None:
    outputs = {
        name: root / relative for name, relative in subject.OUTPUT_RELATIVE_PATHS.items()
    }
    for name in ("audit", "access"):
        payload = json.loads(outputs[name].read_text(encoding="utf-8"))
        mutation(payload["recovery_provenance"])
        write_json(outputs[name], payload)
    lock = json.loads(outputs["lock"].read_text(encoding="utf-8"))
    mutation(lock["recovery_provenance"])
    lock["physical_and_coverage_audit"] = artifact(outputs["audit"], root)
    lock["access_ledger"] = artifact(outputs["access"], root)
    write_json(outputs["lock"], lock)
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    mutation(manifest["recovery_provenance"])
    manifest["decoded_matrix_lock"] = artifact(outputs["lock"], root)
    write_json(outputs["manifest"], manifest)
    rebind_canonical_transaction(root)


def build_complete_fixture(
    tmp_path: Path,
    *,
    attempt_id: str | None = None,
    exact_no_retry: bool = False,
) -> tuple[Path, Path, subject.AuditContract]:
    root = tmp_path / "producer"
    runner = tmp_path / "frozen_runner.py"
    runner.write_text("# frozen synthetic runner\n", encoding="utf-8")
    runner_test = tmp_path / "frozen_runner_test.py"
    runner_test.write_text("# frozen synthetic runner test\n", encoding="utf-8")
    runtime_executable = tmp_path / "python.exe"
    runtime_executable.write_bytes(b"synthetic-python")
    runtime_module = tmp_path / "numpy_init.py"
    runtime_module.write_bytes(b"synthetic-module")
    contract = subject.AuditContract(
        range_rows=9,
        range_bytes=81,
        object_rows=1,
        site_count=2,
        groups=(
            subject.GroupContract("group_a", 1, 3.0),
            subject.GroupContract("group_b", 1, 4.0),
        ),
        raw_attempt_cap=20,
        census_worst_case_attempts=11,
        combined_attempt_cap=31,
        census_successful_requests=7,
        operating_years=(2022,),
        operating_days_of_month=(5,),
        forecast_hours=(28,),
    )
    attempt_id = attempt_id or "20220811T000000000000Z__pid123__synthetic001"
    object_key = "gfs.20220103/12/atmos/gfs.t12z.pgrb2.0p25.f028"
    url = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/" + object_key
    retrieved_at = "2026-08-10T15:00:00Z"
    last_modified = "2022-01-03T15:00:00.000Z"
    object_etag = "0123456789abcdef0123456789abcdef"

    census_rows: list[dict[str, object]] = []
    for index, (variable, level) in enumerate(subject.FEATURE_PAIRS):
        start = index * 9
        family = "PBL_HEIGHT" if variable == "HPBL" else "LOW_LEVEL_ISOBARIC_WIND_PROFILE"
        census_rows.append(
            {
                "target_operating_day_kst": "2022-01-05",
                "run_init_utc": "2022-01-03T12:00:00Z",
                "forecast_hour": 28,
                "valid_time_utc": "2022-01-04T16:00:00Z",
                "cutoff_utc": "2022-01-04T05:00:00Z",
                "object_key": object_key,
                "object_size_bytes": 81,
                "object_etag": object_etag,
                "publication_last_modified_utc": last_modified,
                "cutoff_margin_seconds": 50_400,
                "idx_key": object_key + ".idx",
                "idx_size_bytes": 10,
                "idx_etag": "idx-etag",
                "idx_last_modified_utc": last_modified,
                "status": "CENSUS_VERIFIED",
                "source_archive": "NOAA_NODD_S3",
                "archive_product": "gfs.t12z.pgrb2.0p25.fFFF",
                "retrieval_url_or_request_id": url,
                "official_metadata": "ListObjectsV2 exact Key/LastModified/ETag/Size",
                "publication_evidence_type": "S3_LISTOBJECTSV2_EXACT_KEY_LASTMODIFIED_ETAG_SIZE",
                "publication_evidence_reference": "census/list.xml",
                "family": family,
                "variable": variable,
                "level": level,
                "grib_record_number": index + 1,
                "forecast_descriptor": "28 hour fcst",
                "range_start": start,
                "range_end": start + 8,
                "range_bytes": 9,
                "idx_relative_path": "census/index/f028.idx",
                "idx_sha256": "a" * 64,
            }
        )
    census_path = root / "census" / "field_range_census.parquet"
    census_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(census_rows).to_parquet(census_path, index=False)

    coordinate_path = root / "prereg" / "authoritative_turbine_coordinate_lock_v1.json"
    coordinates = {
        "artifact_type": "AUTHORITATIVE_TURBINE_COORDINATE_LOCK",
        "labels_read": False,
        "sites": [
            {"site_id": 1, "group": "group_a", "latitude": 37.1, "longitude": 128.1, "capacity_mw": 3.0},
            {"site_id": 2, "group": "group_b", "latitude": 37.2, "longitude": 128.2, "capacity_mw": 4.0},
        ],
    }
    write_json(coordinate_path, coordinates)
    cumulative_path = root / "audit" / "CENSUS_CUMULATIVE_ACCESS_AMENDMENT.json"
    write_json(
        cumulative_path,
        {
            "artifact_type": "CENSUS_CUMULATIVE_ACCESS_APPEND_ONLY_AMENDMENT",
            "labels_read": False,
            "2024_arrays_read": False,
            "2025_arrays_read": False,
            "raw_payload_read": False,
            "initial_population_invocation": {"successful_http_requests": 7},
            "phase_total": {
                "successful_http_requests": 7,
                "raw_range_requests": 0,
                "raw_range_bytes": 0,
            },
        },
    )
    census_manifest_path = root / "manifest_census_v1.json"
    write_json(
        census_manifest_path,
        {
            "artifact_type": "NOAA_GFS_MULTISEASON_CENSUS_MANIFEST",
            "schema_version": 1,
            "field_range_census": artifact(census_path, root),
            "labels_read": False,
            "models_fit": 0,
            "submission_csv_created": False,
        },
    )

    amendment_path = root / "prelaunch" / "RAW_RUNNER_STATIC_PREFLIGHT_AMENDMENT_V5.json"
    write_json(
        amendment_path,
        {
            "artifact_type": "RAW_RUNNER_STATIC_PREFLIGHT_APPEND_ONLY_AMENDMENT",
            "schema_version": 5,
            "runner": artifact(runner),
            "runner_test": artifact(runner_test),
            "py_compile_result": "PASS",
            "independent_windows_test_result": "25 passed",
            "labels_read": False,
            "2024_arrays_read": False,
            "2025_arrays_read": False,
            "models_fit": 0,
            "submission_csv_created": False,
            "raw_network_requests": 0,
            "raw_network_bytes": 0,
        },
    )
    preflight_path = root / "manifest_raw_preflight_v5.json"
    write_json(
        preflight_path,
        {
            "artifact_type": "RAW_RUNNER_STATIC_PREFLIGHT_MANIFEST",
            "schema_version": 5,
            "runner": artifact(runner),
            "runner_test": artifact(runner_test),
            "py_compile_result": "PASS",
            "independent_windows_test_result": "25 passed",
            "amendment": artifact(amendment_path, root),
        },
    )
    independent_path = root / "independent_redteam" / "TRACK_A_RAW_RUNNER_INDEPENDENT_PREFLIGHT_AUDIT_V1.json"
    write_json(
        independent_path,
        {
            "artifact_type": "TRACK_A_RAW_RUNNER_INDEPENDENT_PREFLIGHT_AUDIT",
            "schema_version": 1,
            "status": "PASS_TO_CREATE_FINAL_AUTH_ONLY",
            "code_and_test_identity": {
                "runner": artifact(runner),
                "runner_test": artifact(runner_test),
            },
            "code_review": {
                "actual_http_attempt_reserved_before_urlopen": True,
                "attempt_15200_valid_and_attempt_15201_forbidden": True,
                "bounded_parallel_fail_fast_wave_max_workers": 8,
                "canonical_outputs_transactional_after_decode_and_physical_audit": True,
                "content_range_and_archive_header_identity_fail_closed": True,
                "crash_consistent_prefix_suffix_and_complete_part_recovery": True,
                "direct_sequential_exact_event_paths_no_per_range_directory_scan": True,
                "event_attempt_path_payload_and_global_inventory_fail_closed": True,
                "exact_single_grib_grid_run_valid_variable_and_level_checks": True,
                "final_and_projected_200gb_free_space_gates": True,
                "global_event_inventory_scanned_once": True,
                "remaining_concrete_static_or_windows_runtime_blockers": [],
                "windows_handle_types_and_writable_fsync_paths": True,
            },
            "independent_execution_evidence": {
                "py_compile_result": "PASS",
                "pytest_result": "31 passed",
                "network_requests": 0,
            },
            "effective_preflight_chain": {"manifest_v5": artifact(preflight_path, root)},
        },
    )
    runtime = {
        "python_version": "synthetic",
        "python_executable": str(runtime_executable.resolve()),
        "python_executable_size_bytes": runtime_executable.stat().st_size,
        "python_executable_sha256": subject.sha256_file(runtime_executable),
        "platform": "synthetic",
        "packages": {
            "numpy": {
                "version": "synthetic",
                "module_file": str(runtime_module.resolve()),
                "module_file_size_bytes": runtime_module.stat().st_size,
                "module_file_sha256": subject.sha256_file(runtime_module),
            }
        },
    }
    authorization_path = root / "prereg" / "raw_launch_authorization_v1.json"
    write_json(
        authorization_path,
        {
            "artifact_type": "NOAA_GFS_MULTISEASON_RAW_LAUNCH_AUTHORIZATION",
            "schema_version": 2,
            "status": "AUTHORIZED_RAW_RANGE_LAUNCH_ONLY_AFTER_INDEPENDENT_GO",
            "labels_read": False,
            "2024_arrays_read": False,
            "2025_arrays_read": False,
            "models_fit": 0,
            "submission_csv_created": False,
            "expected_range_rows": contract.range_rows,
            "expected_range_bytes": contract.range_bytes,
            "raw_actual_http_attempt_budget": contract.raw_attempt_cap,
            "census_worst_case_http_attempts": contract.census_worst_case_attempts,
            "census_plus_raw_max_http_attempts": contract.combined_attempt_cap,
            "runner": artifact(runner),
            "runner_test": artifact(runner_test),
            "py_compile_result": "PASS",
            "independent_windows_test_result": "25 passed",
            "field_range_census": artifact(census_path, root),
            "census_manifest": artifact(census_manifest_path, root),
            "coordinate_lock": artifact(coordinate_path, root),
            "census_cumulative_access": {
                **artifact(cumulative_path, root),
                "successful_http_requests": contract.census_successful_requests,
                "worst_case_http_attempts": contract.census_worst_case_attempts,
                "attempts_per_logical_request_hard_max": 4,
            },
            "effective_preflight_manifest": artifact(preflight_path, root),
            "effective_preflight_amendment": artifact(amendment_path, root),
            "independent_prelaunch_audit": artifact(independent_path, root),
            "runtime_identity": runtime,
            "runtime_identity_sha256": subject.canonical_payload_sha256(runtime),
        },
    )
    go_path = root / "independent_redteam" / "TRACK_A_RAW_LAUNCH_GO.json"
    write_json(
        go_path,
        {
            "artifact_type": "TRACK_A_INDEPENDENT_RAW_RANGE_LAUNCH_GO",
            "status": "GO_RAW_RANGE_LAUNCH",
            "labels_read": False,
            "2024_arrays_read": False,
            "2025_arrays_read": False,
            "models_fit": 0,
            "submission_csv_created": False,
            "bound_authorization_sha256": subject.sha256_file(authorization_path),
            "bound_authorization_size_bytes": authorization_path.stat().st_size,
            "bound_runner_sha256": subject.sha256_file(runner),
            "bound_census_manifest_sha256": subject.sha256_file(census_manifest_path),
            "bound_coordinate_lock_sha256": subject.sha256_file(coordinate_path),
            "bound_effective_preflight_manifest_sha256": subject.sha256_file(preflight_path),
            "bound_independent_prelaunch_audit_sha256": subject.sha256_file(independent_path),
            "bound_runtime_identity_sha256": subject.canonical_payload_sha256(runtime),
            "expected_range_rows": contract.range_rows,
            "expected_range_bytes": contract.range_bytes,
            "max_actual_http_attempts": contract.raw_attempt_cap,
            "census_worst_case_http_attempts": contract.census_worst_case_attempts,
            "census_plus_raw_max_http_attempts": contract.combined_attempt_cap,
        },
    )

    raw_metas: list[dict[str, object]] = []
    event_paths: list[Path] = []
    for index, census in enumerate(census_rows):
        payload = b"GRIB" + bytes([65 + index]) + b"7777"
        relative = subject.raw_relative_path(census)
        raw_path = root / relative
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(payload)
        event_id = subject.event_id_for(census, int(census["range_start"]), int(census["range_end"]))
        final_attempt = 1 if exact_no_retry else (2 if index in (0, 1) else 1)
        final_global = index + 1 if exact_no_retry else (
            3 if index == 0 else 4 if index == 1 else index + 3
        )
        selected_start_path: Path | None = None
        selected_complete_path: Path | None = None
        semantic_completion_count = 1
        if not exact_no_retry and index == 0:
            interrupted_start = (
                root
                / "raw"
                / "request_events"
                / f"{event_id}__attempt_001_start.json"
            )
            write_json(
                interrupted_start,
                {
                    "event": "RANGE_REQUEST_START",
                    "event_id": event_id,
                    "attempt": 1,
                    "global_raw_attempt_number": 1,
                    "url": url,
                    "range_start": census["range_start"],
                    "range_end": census["range_end"],
                    "created_utc": retrieved_at,
                },
            )
            event_paths.append(interrupted_start)
        elif not exact_no_retry and index == 1:
            earlier_retrieved_at = "2026-08-10T14:59:00Z"
            earlier_start = (
                root
                / "raw"
                / "request_events"
                / f"{event_id}__attempt_001_start.json"
            )
            earlier_complete = (
                root
                / "raw"
                / "request_events"
                / f"{event_id}__attempt_001_complete.json"
            )
            write_json(
                earlier_start,
                {
                    "event": "RANGE_REQUEST_START",
                    "event_id": event_id,
                    "attempt": 1,
                    "global_raw_attempt_number": 2,
                    "url": url,
                    "range_start": census["range_start"],
                    "range_end": census["range_end"],
                    "created_utc": earlier_retrieved_at,
                },
            )
            write_json(
                earlier_complete,
                {
                    "event": "RANGE_REQUEST_COMPLETE",
                    "event_id": event_id,
                    "attempt": 1,
                    "global_raw_attempt_number": 2,
                    "http_status": 206,
                    "range_start": census["range_start"],
                    "range_end": census["range_end"],
                    "content_range": f"bytes {census['range_start']}-{census['range_end']}/81",
                    "etag": object_etag,
                    "last_modified_utc": last_modified,
                    "bytes": len(payload),
                    "payload_sha256": subject.sha256_file(raw_path),
                    "retrieved_at": earlier_retrieved_at,
                },
            )
            event_paths.extend((earlier_complete, earlier_start))
            selected_start_path = earlier_start
            selected_complete_path = earlier_complete
            semantic_completion_count = 2
        start_path = (
            root
            / "raw"
            / "request_events"
            / f"{event_id}__attempt_{final_attempt:03d}_start.json"
        )
        complete_path = (
            root
            / "raw"
            / "request_events"
            / f"{event_id}__attempt_{final_attempt:03d}_complete.json"
        )
        write_json(
            start_path,
            {
                "event": "RANGE_REQUEST_START",
                "event_id": event_id,
                "attempt": final_attempt,
                "global_raw_attempt_number": final_global,
                "url": url,
                "range_start": census["range_start"],
                "range_end": census["range_end"],
                "created_utc": retrieved_at,
            },
        )
        write_json(
            complete_path,
            {
                "event": "RANGE_REQUEST_COMPLETE",
                "event_id": event_id,
                "attempt": final_attempt,
                "global_raw_attempt_number": final_global,
                "http_status": 206,
                "range_start": census["range_start"],
                "range_end": census["range_end"],
                "content_range": f"bytes {census['range_start']}-{census['range_end']}/81",
                "etag": object_etag,
                "last_modified_utc": last_modified,
                "bytes": len(payload),
                "payload_sha256": subject.sha256_file(raw_path),
                "retrieved_at": retrieved_at,
            },
        )
        event_paths.extend((complete_path, start_path))
        if selected_start_path is None:
            selected_start_path = start_path
            selected_complete_path = complete_path
        assert selected_complete_path is not None
        meta = {
            "source_archive": census["source_archive"],
            "archive_product": census["archive_product"],
            "target_operating_day_kst": census["target_operating_day_kst"],
            "run_init_utc": census["run_init_utc"],
            "forecast_hour": census["forecast_hour"],
            "valid_time_utc": census["valid_time_utc"],
            "retrieval_url_or_request_id": census["retrieval_url_or_request_id"],
            "retrieved_at": retrieved_at,
            "raw_filename": relative,
            "raw_sha256": subject.sha256_file(raw_path),
            "raw_size_bytes": len(payload),
            "official_metadata": census["official_metadata"],
            "publication_evidence_type": census["publication_evidence_type"],
            "publication_evidence_reference": census["publication_evidence_reference"],
            "cutoff_utc": census["cutoff_utc"],
            "cutoff_margin": census["cutoff_margin_seconds"],
            "object_key": census["object_key"],
            "object_etag": census["object_etag"],
            "object_size_bytes": census["object_size_bytes"],
            "publication_last_modified_utc": census["publication_last_modified_utc"],
            "range_start": census["range_start"],
            "range_end": census["range_end"],
            "range_bytes": census["range_bytes"],
            "family": census["family"],
            "variable": census["variable"],
            "level": census["level"],
            "idx_sha256": census["idx_sha256"],
            "http_status": 206,
            "content_range": f"bytes {census['range_start']}-{census['range_end']}/81",
            "request_range_start": census["range_start"],
            "resumed_from_bytes": 0,
            "status": "VERIFIED",
            "request_completion_evidence": {
                "semantically_identical_completion_event_count": semantic_completion_count,
                "selected_start_event": artifact(selected_start_path, root),
                "selected_complete_event": artifact(selected_complete_path, root),
            },
        }
        write_json(raw_path.with_suffix(".grib2.meta.json"), meta)
        raw_metas.append(meta)

    raw_parquet = root / subject.OUTPUT_RELATIVE_PATHS["raw_parquet"]
    raw_csv = root / subject.OUTPUT_RELATIVE_PATHS["raw_csv"]
    raw_parquet.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(raw_metas).to_parquet(raw_parquet, index=False)
    pd.DataFrame(raw_metas).to_csv(raw_csv, index=False)

    sites: list[dict[str, object]] = []
    for site_id, group_name, latitude, longitude, capacity, adjustment in (
        (1, "group_a", 37.1, 128.1, 3.0, 0.0),
        (2, "group_b", 37.2, 128.2, 4.0, 1.0),
    ):
        row: dict[str, object] = {
            "valid_time_utc": "2022-01-04T16:00:00Z",
            "target_operating_day_kst": "2022-01-05",
            "run_init_utc": "2022-01-03T12:00:00Z",
            "forecast_hour": 28,
            "site_id": site_id,
            "group": group_name,
            "latitude": latitude,
            "longitude": longitude,
            "capacity_mw": capacity,
            "HPBL_surface": 100.0 + adjustment,
        }
        for feature_index, feature in enumerate(subject.FEATURE_COLUMNS[1:], start=1):
            row[feature] = float(feature_index) + adjustment
        sites.append(row)
    groups: list[dict[str, object]] = []
    for site in sites:
        groups.append(
            {
                "valid_time_utc": site["valid_time_utc"],
                "target_operating_day_kst": site["target_operating_day_kst"],
                "run_init_utc": site["run_init_utc"],
                "forecast_hour": site["forecast_hour"],
                "group": site["group"],
                "site_count": 1,
                "capacity_mw": site["capacity_mw"],
                **{feature: site[feature] for feature in subject.FEATURE_COLUMNS},
            }
        )
    site_path = root / subject.OUTPUT_RELATIVE_PATHS["site"]
    group_path = root / subject.OUTPUT_RELATIVE_PATHS["group"]
    site_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sites).to_parquet(site_path, index=False)
    pd.DataFrame(groups).to_parquet(group_path, index=False)

    physical_path = root / subject.OUTPUT_RELATIVE_PATHS["audit"]
    write_json(
        physical_path,
        {
            "expected_site_rows": 2,
            "observed_site_rows": 2,
            "expected_group_rows": 2,
            "observed_group_rows": 2,
            "duplicate_site_keys": 0,
            "duplicate_group_keys": 0,
            "families": {
                "PBL_HEIGHT": {
                    "finite_fraction": 1.0,
                    "minimum": 100.0,
                    "maximum": 101.0,
                    "physical_gate_pass": True,
                    "status": "PASS",
                },
                "LOW_LEVEL_ISOBARIC_WIND_PROFILE": {
                    "finite_fraction": 1.0,
                    "maximum_absolute_component_mps": 9.0,
                    "physical_gate_pass": True,
                    "status": "PASS",
                },
            },
            "independent_family_salvage_applied_exactly": True,
            "labels_read": False,
            "models_fit": 0,
            "final_free_disk_before_transaction_bytes": 300_000_000_000,
            "final_200gb_reserve_pass": True,
        },
    )
    access_path = root / subject.OUTPUT_RELATIVE_PATHS["access"]
    write_json(
        access_path,
        {
            "expected_ranges": 9,
            "expected_bytes": 81,
            "network_requests_this_invocation": 9,
            "network_bytes_this_invocation": 81,
            "actual_http_attempts_before_invocation": 0 if exact_no_retry else 2,
            "actual_http_attempts_after_invocation": 9 if exact_no_retry else 11,
            "actual_http_attempts_this_invocation": 9,
            "raw_actual_http_attempt_budget": 20,
            "census_worst_case_http_attempts": 11,
            "census_plus_raw_max_http_attempts": 31,
            "census_worst_case_plus_raw_budget_le_20000": True,
            "census_successful_http_requests": 7,
            "census_plus_raw_actual_attempts_or_successes": 16 if exact_no_retry else 18,
            "cached_reuses": 0,
            "durable_transport_attempt_starts": 9 if exact_no_retry else 11,
            "durable_transport_attempt_completions": 9 if exact_no_retry else 10,
            "durable_transport_attempt_errors": 0,
            "free_disk_before_bytes": 400_000_000_000,
            "free_disk_after_bytes": 300_000_000_000,
            "final_free_disk_before_transaction_bytes": 300_000_000_000,
            "final_200gb_reserve_pass": True,
            "concurrency": 8,
            "labels_read": False,
            "2024_arrays_read": False,
            "2025_arrays_read": False,
            "models_fit": 0,
            "submission_csv_created": False,
        },
    )
    lock_path = root / subject.OUTPUT_RELATIVE_PATHS["lock"]
    write_json(
        lock_path,
        {
            "artifact_type": "TARGET_FREE_DECODED_MATRIX_LOCK",
            "authorization": artifact(authorization_path, root),
            "independent_go": artifact(go_path, root),
            "coordinate_lock": artifact(coordinate_path, root),
            "runtime_identity": runtime,
            "runtime_identity_sha256": subject.canonical_payload_sha256(runtime),
            "raw_manifest_parquet": artifact(raw_parquet, root),
            "raw_manifest_csv": artifact(raw_csv, root),
            "site_matrix": artifact(site_path, root),
            "group_matrix": artifact(group_path, root),
            "physical_and_coverage_audit": artifact(physical_path, root),
            "access_ledger": artifact(access_path, root),
            "target_free_duplicate_estimator_may_start": True,
            "labels_read": False,
        },
    )
    progress_root = root / "raw" / "progress"
    raw_checkpoint = progress_root / f"raw_ranges__{attempt_id}__000009.json"
    decode_checkpoint = progress_root / f"decoded_messages__{attempt_id}__000009.json"
    completed_keys = [
        "|".join(
            (
                str(row["object_key"]),
                str(int(row["forecast_hour"])),
                str(row["variable"]),
                str(row["level"]),
            )
        )
        for row in census_rows
    ]
    completed_key_digest = hashlib.sha256(
        ("\n".join(sorted(completed_keys)) + "\n").encode("utf-8")
    ).hexdigest()
    write_json(
        raw_checkpoint,
        {
            "phase": "RAW_RANGE_DOWNLOAD",
            "attempt_id": attempt_id,
            "completed_ranges": 9,
            "completed_key_set_sha256": completed_key_digest,
            "network_requests_this_invocation_so_far": 9,
            "network_bytes_this_invocation_so_far": 81,
            "actual_http_attempts_cumulative": 9 if exact_no_retry else 11,
        },
    )
    write_json(
        decode_checkpoint,
        {
            "phase": "ECCODES_BILINEAR_DECODE",
            "attempt_id": attempt_id,
            "completed_messages": 9,
            "network_requests": 0,
        },
    )
    manifest_path = root / subject.OUTPUT_RELATIVE_PATHS["manifest"]
    write_json(
        manifest_path,
        {
            "artifact_type": "NOAA_GFS_MULTISEASON_RAW_AND_DECODE_MANIFEST",
            "schema_version": 1,
            "created_utc": retrieved_at,
            "runner": artifact(runner),
            "launch_attempt_id": attempt_id,
            "authorization": artifact(authorization_path, root),
            "independent_go": artifact(go_path, root),
            "census_manifest": artifact(census_manifest_path, root),
            "coordinate_lock": artifact(coordinate_path, root),
            "runtime_identity": runtime,
            "runtime_identity_sha256": subject.canonical_payload_sha256(runtime),
            "decoded_matrix_lock": artifact(lock_path, root),
            "progress_checkpoints": [artifact(path, root) for path in sorted((raw_checkpoint, decode_checkpoint))],
            "request_event_inventory": [artifact(path, root) for path in sorted(event_paths)],
            "exact_ranges": 9,
            "exact_bytes": 81,
            "labels_read": False,
            "2024_arrays_read": False,
            "2025_arrays_read": False,
            "models_fit": 0,
            "submission_csv_created": False,
        },
    )

    outputs = {name: root / relative for name, relative in subject.OUTPUT_RELATIVE_PATHS.items()}
    transaction_root = root / "raw" / "output_transactions"
    plan_path = transaction_root / f"{attempt_id}__plan.json"
    items = []
    for name, destination in sorted(outputs.items()):
        relative = destination.relative_to(root).as_posix()
        actual = artifact(destination, root)
        items.append(
            {
                "name": name,
                "staged_path": f"raw/output_transactions/{attempt_id}/staged/{relative}",
                "destination_path": relative,
                "size_bytes": actual["size_bytes"],
                "sha256": actual["sha256"],
            }
        )
    write_json(
        plan_path,
        {
            "artifact_type": "RAW_OUTPUT_TRANSACTION_PLAN",
            "schema_version": 1,
            "attempt_id": attempt_id,
            "created_utc": retrieved_at,
            "canonical_outputs_absent_before_plan": True,
            "decode_and_physical_audit_completed_before_plan": True,
            "items": items,
        },
    )
    commit_path = transaction_root / f"{attempt_id}__committed.json"
    write_json(
        commit_path,
        {
            "artifact_type": "RAW_OUTPUT_TRANSACTION_COMMIT",
            "schema_version": 1,
            "attempt_id": attempt_id,
            "plan": artifact(plan_path, root),
            "outputs": {name: artifact(path, root) for name, path in outputs.items()},
            "all_outputs_reopened_and_rehashed": True,
        },
    )
    complete_lock = root / "raw" / "launch_history" / f"{attempt_id}__complete.lock"
    write_json(complete_lock, {"attempt_id": attempt_id, "pid": 123, "created_utc": retrieved_at})
    return root, runner, contract


def fixture_inventory(paths: object, root: Path) -> dict[str, object]:
    rows = [artifact(path, root) for path in sorted(paths)]
    return {
        "file_count": len(rows),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in rows),
        "identity_rows_sha256": subject.canonical_payload_sha256(rows),
    }


def build_recovered_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, subject.AuditContract]:
    failed_attempt = subject.RECOVERY_FAILED_ATTEMPT
    recovery_attempt = "decode_recovery_v1__synthetic0001"
    root, runner, contract = build_complete_fixture(
        tmp_path,
        attempt_id=failed_attempt,
        exact_no_retry=True,
    )
    outputs = {
        name: root / relative for name, relative in subject.OUTPUT_RELATIVE_PATHS.items()
    }
    original_authorization_path = root / "prereg" / "raw_launch_authorization_v1.json"
    original_authorization = json.loads(
        original_authorization_path.read_text(encoding="utf-8")
    )
    original_go_path = root / "independent_redteam" / "TRACK_A_RAW_LAUNCH_GO.json"

    original_plan = next((root / "raw" / "output_transactions").glob("*__plan.json"))
    original_commit = next(
        (root / "raw" / "output_transactions").glob("*__committed.json")
    )
    original_plan.unlink()
    original_commit.unlink()
    old_staged_root = (
        root
        / "raw"
        / "output_transactions"
        / failed_attempt
        / "staged"
        / "raw"
    )
    old_staged_root.mkdir(parents=True)
    old_csv = old_staged_root / "RAW_RANGE_MANIFEST.csv"
    old_parquet = old_staged_root / "RAW_RANGE_MANIFEST.parquet"
    shutil.copyfile(outputs["raw_csv"], old_csv)
    shutil.copyfile(outputs["raw_parquet"], old_parquet)
    remnants = [artifact(path, root) for path in (old_csv, old_parquet)]

    decode_checkpoint = (
        root
        / "raw"
        / "progress"
        / f"decoded_messages__{failed_attempt}__{contract.range_rows:06d}.json"
    )
    decode_checkpoint.unlink()
    complete_lock = (
        root / "raw" / "launch_history" / f"{failed_attempt}__complete.lock"
    )
    complete_lock.unlink()
    failed_lock = (
        root / "raw" / "launch_history" / f"{failed_attempt}__failed.lock"
    )
    write_json(
        failed_lock,
        {"attempt_id": failed_attempt, "pid": 3536, "created_utc": "2026-08-10T14:00:00Z"},
    )
    stderr_path = root / "logs" / "TRACK_A_RAW_RUN.stderr.log"
    stdout_path = root / "logs" / "TRACK_A_RAW_RUN.stdout.log"
    stderr_path.parent.mkdir(parents=True)
    stderr_path.write_text("synthetic ecCodes MEMFS failure\n", encoding="utf-8")
    stdout_path.write_bytes(b"")

    final_progress = (
        root
        / "raw"
        / "progress"
        / f"raw_ranges__{failed_attempt}__{contract.range_rows:06d}.json"
    )
    final_progress_payload = json.loads(final_progress.read_text(encoding="utf-8"))
    final_progress_payload.update(
        {
            "checkpoint_name": final_progress.stem,
            "created_utc": "2026-08-10T16:59:59Z",
        }
    )
    write_json(final_progress, final_progress_payload)
    ordered_raw_rows = pd.read_parquet(outputs["raw_parquet"]).to_dict("records")
    for count in subject.recovery_progress_counts(contract):
        if count == contract.range_rows:
            continue
        prefix = ordered_raw_rows[:count]
        prefix_keys = [
            "|".join(
                (
                    str(row["object_key"]),
                    str(int(row["forecast_hour"])),
                    str(row["variable"]),
                    str(row["level"]),
                )
            )
            for row in prefix
        ]
        prefix_digest = hashlib.sha256(
            ("\n".join(sorted(prefix_keys)) + "\n").encode("utf-8")
        ).hexdigest()
        checkpoint = (
            root
            / "raw"
            / "progress"
            / f"raw_ranges__{failed_attempt}__{count:06d}.json"
        )
        write_json(
            checkpoint,
            {
                "phase": "RAW_RANGE_DOWNLOAD",
                "attempt_id": failed_attempt,
                "completed_ranges": count,
                "completed_key_set_sha256": prefix_digest,
                "network_requests_this_invocation_so_far": count,
                "network_bytes_this_invocation_so_far": sum(
                    int(row["raw_size_bytes"]) for row in prefix
                ),
                "actual_http_attempts_cumulative": contract.range_rows,
                "checkpoint_name": checkpoint.stem,
                "created_utc": "2026-08-10T16:59:58Z",
            },
        )
    evidence = [artifact(path, root) for path in (failed_lock, final_progress, stderr_path, stdout_path)]
    incident_path = root / subject.RECOVERY_INCIDENT_RELATIVE
    incident = {
        "schema_version": 1,
        "artifact_type": "TRACK_A_RAW_DECODE_MEMFS_THREAD_INIT_FAILURE_INCIDENT",
        "created_kst": "2026-08-11T01:33:12+09:00",
        "status": "RAW_DOWNLOAD_COMPLETE_DECODE_FAILED_RECOVERY_NOT_AUTHORIZED",
        "failed_attempt_id": failed_attempt,
        "failure_boundary": {
            "raw_ranges_completed": contract.range_rows,
            "raw_bytes_completed": contract.range_bytes,
            "successful_network_requests": contract.range_rows,
            "actual_http_attempts": contract.range_rows,
            "decoded_messages_completed": 0,
            "canonical_outputs_committed": 0,
            "active_lock_present_after_failure": False,
            "complete_lock_present_after_failure": False,
        },
        "cause": {"classification": "SYNTHETIC_MEMFS_THREAD_RACE"},
        "immutable_failure_evidence": evidence,
        "documented_preplan_staging_remnants": remnants,
        "preservation_policy": {"preplan_remnants_immutable": True},
        "recovery_gate": {"currently_authorized": False},
    }
    write_json(incident_path, incident)
    incident_identity = artifact(incident_path, root)

    raw_files = sorted(
        path
        for path in (root / "raw" / "ranges").rglob("*")
        if path.is_file()
    )
    event_files = sorted((root / "raw" / "request_events").glob("*.json"))
    progress_files = sorted((root / "raw" / "progress").glob("*.json"))
    progress_payloads = [
        json.loads(path.read_text(encoding="utf-8")) for path in progress_files
    ]
    progress_counts = subject.recovery_progress_counts(contract)
    failed_payload = json.loads(failed_lock.read_text(encoding="utf-8"))
    progress_semantics = {
        "progress_checkpoint_count": len(progress_payloads),
        "checkpoint_completed_sequence_sha256": subject.canonical_payload_sha256(
            progress_counts
        ),
        "checkpoint_filename_sequence_sha256": subject.canonical_payload_sha256(
            [path.name for path in progress_files]
        ),
        "progress_payload_sequence_sha256": subject.canonical_payload_sha256(
            progress_payloads
        ),
        "first_checkpoint_created_utc": progress_payloads[0]["created_utc"],
        "final_checkpoint_created_utc": progress_payloads[-1]["created_utc"],
        "final_completed_ranges": contract.range_rows,
        "final_network_requests": contract.range_rows,
        "final_network_bytes": contract.range_bytes,
        "final_actual_http_attempts": contract.range_rows,
        "final_completed_key_set_sha256": progress_payloads[-1][
            "completed_key_set_sha256"
        ],
        "start_event_count": contract.range_rows,
        "start_global_attempts_exact_1_through_10368": True,
        "attempt_count_reconstructed_from_start_event_timestamps": True,
        "failed_lock_payload_sha256": subject.canonical_payload_sha256(
            failed_payload
        ),
        "failed_lock_attempt_id": failed_attempt,
        "failed_lock_pid": 3536,
    }
    inventories = {
        "raw_ranges_and_sidecars": fixture_inventory(raw_files, root),
        "request_events": fixture_inventory(event_files, root),
        "original_progress": fixture_inventory(progress_files, root),
        "original_launch_history": fixture_inventory([failed_lock], root),
        "original_progress_and_failed_lock_semantics": progress_semantics,
        "validated_raw_payload_bytes": contract.range_bytes,
        "raw_file_count": contract.range_rows,
        "sidecar_file_count": contract.range_rows,
        "request_event_file_count": contract.range_rows * 2,
    }
    prerecovery_topology = {
        "raw_top_level_entries": sorted(
            {"launch_history", "output_transactions", "progress", "ranges", "request_events"}
        ),
        "decoded_tree_absent": True,
        "documented_preplan_transactions": fixture_inventory(
            [old_csv, old_parquet], root
        ),
    }
    raw_cache_lock_path = root / subject.RECOVERY_RAW_CACHE_LOCK_RELATIVE
    write_json(
        raw_cache_lock_path,
        {
            "schema_version": 1,
            "artifact_type": "TRACK_A_DECODE_RECOVERY_RAW_CACHE_LOCK",
            "status": "PASS_IMMUTABLE_RAW_CACHE_READY_FOR_DECODE_ONLY",
            "created_utc": "2026-08-10T17:10:00Z",
            "incident": incident_identity,
            "inventories": inventories,
            "prerecovery_topology": prerecovery_topology,
            "documented_preplan_remnants": remnants,
            "canonical_outputs_absent": True,
            "network_requests_during_preflight": 0,
            "raw_event_progress_mutations": 0,
            "labels_read": False,
            "2024_arrays_read": False,
            "2025_arrays_read": False,
        },
    )

    recovery_identities = {
        "original_runner": artifact(runner),
        "recovery_runner": artifact(subject.RECOVERY_RUNNER),
        "recovery_module": artifact(subject.RECOVERY_MODULE),
        "recovery_bootstrap": artifact(subject.RECOVERY_BOOTSTRAP),
        "recovery_test": artifact(subject.RECOVERY_TEST),
        "recovery_bootstrap_test": artifact(subject.RECOVERY_BOOTSTRAP_TEST),
        "recovery_runner_test": artifact(subject.RECOVERY_RUNNER_TEST),
        "postrun_auditor": artifact(Path(subject.__file__).resolve()),
        "postrun_auditor_test": artifact(subject.AUDITOR_TEST),
    }
    runtime_paths = subject._recovery_runtime_file_paths()
    original_runtime = original_authorization["runtime_identity"]
    runtime_lock = {
        "api_version": "2.47.0",
        "definitions_path": "/MEMFS/definitions",
        **{key: artifact(path) for key, path in runtime_paths.items()},
        "frozen_runner": recovery_identities["original_runner"],
        "recovery_module": recovery_identities["recovery_module"],
        "original_runtime_identity": original_runtime,
    }
    runtime_report = {
        "pid": 999,
        "api_version": "2.47.0",
        "definitions_path": "/MEMFS/definitions",
        "library_path": str(runtime_paths["library"].resolve()),
        "memfs_library_path": str(runtime_paths["memfs_library"].resolve()),
        "python_binding_path": str(runtime_paths["python_binding"].resolve()),
        "gribapi_binding_path": str(runtime_paths["gribapi_binding"].resolve()),
        "eccodes_python_api_path": str(runtime_paths["eccodes_python_api"].resolve()),
        "gribapi_python_api_path": str(runtime_paths["gribapi_python_api"].resolve()),
        "gribapi_errors_path": str(runtime_paths["gribapi_errors"].resolve()),
        "frozen_runner_path": str(runner.resolve()),
        "frozen_runner_sha256": subject.sha256_file(runner),
        "recovery_module_path": str(subject.RECOVERY_MODULE.resolve()),
        "recovery_module_sha256": subject.sha256_file(subject.RECOVERY_MODULE),
        "original_runtime_identity": subject._expected_original_runtime_report(
            original_runtime
        ),
    }
    pilot_identity = artifact(subject.RECOVERY_REAL_PILOT)
    pilot_report = {
        "status": "PASS_REAL_GRIB_STDLIB_BOOTSTRAP_PROCESS_ISOLATION",
        "workers_requested": 7,
        "worker_pids_observed": list(range(701, 708)),
        "tasks": 70,
        "first_wave_synchronized": True,
        "reference": {
            "metadata": {"gridType": "regular_ll"},
            "value_count": 1440 * 721,
            "value_min": 1.0,
            "value_max": 2.0,
        },
        "network_requests": 0,
    }
    production_report = {
        "tasks": 70,
        "worker_pids_observed": list(range(711, 718)),
        "worker_count": 7,
        "first_wave_synchronized": True,
        "feature": "HPBL_surface",
        "site_count": 17,
        "site_values_binary64_sha256": "edeca661e457c32431218ccb50c5dbf0c428905a36e03c28cfe4d76d32ddcc0b",
        "all_rows_array_equal": True,
        "network_requests": 0,
    }
    recovery_sealers = {
        "recovery_sealer": artifact(subject.RECOVERY_SEALER),
        "recovery_sealer_test": artifact(subject.RECOVERY_SEALER_TEST),
    }
    bootstrap_import_contract = (
        subject._expected_recovery_bootstrap_import_contract()
    )
    entrypoint_incident_path = (
        root / subject.RECOVERY_SEALER_ENTRYPOINT_INCIDENT_RELATIVE
    )
    write_json(
        entrypoint_incident_path,
        {
            "artifact_type": (
                "TRACK_A_DECODE_RECOVERY_SEAL_DIRECT_FILE_PICKLE_FAILURE_INCIDENT"
            ),
            "schema_version": 1,
            "status": "FAILED_BEFORE_PUBLICATION_RETRY_FORBIDDEN_PENDING_REFREEZE",
            "created_utc": "2026-08-10T17:05:00Z",
            "failed_execution": {
                "observed_start_utc": "2026-08-10T17:00:00Z",
                "command": [
                    str(Path(sys.executable).resolve()),
                    "-B",
                    "scripts/seal_noaa_gfs_multiseason_decode_recovery_v1.py",
                    "--root",
                    str(root),
                ],
                "invocation_mode": "DIRECT_FILE",
                "execution_wrapper_observed_exit_code": 1,
                "sealer_failure_handler_declared_exit_code": 2,
                "observed_stdout_json": {
                    "status": "FAIL_DECODE_RECOVERY_SEAL",
                    "error_type": "PicklingError",
                    "error": (
                        "Can't pickle worker: import of module "
                        "src.noaa_gfs_decode_recovery_bootstrap failed"
                    ),
                    "network_requests": 0,
                },
                "observed_stderr_exact_normalized": (
                    "multiprocessing spawn child failure"
                ),
                "primary_exception_exact": (
                    "PicklingError: Can't pickle initialize_spawned_worker because "
                    "import of module src.noaa_gfs_decode_recovery_bootstrap failed"
                ),
                "secondary_child_exception_exact": "OSError: synthetic parent abort",
                "secondary_exception_is_parent_abort_consequence": True,
            },
            "failure_boundary": {
                "raw_cache_validation_completed_read_only": True,
                "spawned_pilot_submission_started": True,
                "spawned_pilot_tasks_completed": 0,
                "immutable_test_subprocesses_started": 0,
                "seal_publications_created": 0,
                "independent_review_created": 0,
                "independent_go_created": 0,
                "recovery_launch_started": False,
                "network_requests": 0,
                "network_bytes": 0,
                "labels_read": False,
                "arrays_2024_read": False,
                "arrays_2025_read": False,
                "models_fit": 0,
                "submission_csv_created": False,
            },
            "failed_head_identities": {
                "recovery_core": recovery_identities["recovery_module"],
                "recovery_bootstrap": recovery_identities["recovery_bootstrap"],
                "recovery_runner": recovery_identities["recovery_runner"],
                "recovery_sealer": recovery_sealers["recovery_sealer"],
                "recovery_sealer_test": recovery_sealers[
                    "recovery_sealer_test"
                ],
                "planned_postrun_auditor": recovery_identities[
                    "postrun_auditor"
                ],
                "planned_postrun_auditor_test": recovery_identities[
                    "postrun_auditor_test"
                ],
            },
            "immutable_raw_prestate": {
                "prior_decode_failure_incident": incident_identity,
                "final_original_progress": artifact(final_progress, root),
                "original_failed_lock": artifact(failed_lock, root),
                "documented_preplan_remnants": remnants,
                "raw_grib_files": contract.range_rows,
                "raw_sidecar_files": contract.range_rows,
                "request_event_files": contract.range_rows * 2,
                "progress_files": len(subject.recovery_progress_counts(contract)),
                "launch_history_files": 1,
                "validated_raw_payload_bytes": contract.range_bytes,
            },
            "verified_zero_state_after_failure": {
                "absent_control_paths": [
                    subject.RECOVERY_RAW_CACHE_LOCK_RELATIVE,
                    subject.RECOVERY_REAL_PREFLIGHT_RELATIVE,
                    subject.RECOVERY_AUTH_RELATIVE,
                    subject.RECOVERY_REVIEW_RELATIVE,
                    subject.RECOVERY_GO_RELATIVE,
                ],
                "canonical_recovery_outputs_absent": True,
                "recovery_active_locks_absent": True,
                "seal_temporary_files_absent": True,
                "matching_bootstrap_pyc_files": 0,
                "only_original_failed_output_transaction_present": True,
            },
            "root_cause": {
                "category": "WINDOWS_SPAWN_PICKLE_IMPORTABILITY",
                "direct_file_sys_path_root_missing": True,
                "exact_loaded_bootstrap_module_name": (
                    "src.noaa_gfs_decode_recovery_bootstrap"
                ),
                "parent_package_importability_not_established": True,
                "raw_data_or_decoder_value_failure": False,
            },
            "mandatory_remediation": {
                "canonical_write_invocation": (
                    "python -B -m scripts.seal_noaa_gfs_multiseason_decode_recovery_v1"
                ),
                "direct_file_write_invocation_forbidden": True,
                "bootstrap_top_level_module_name": (
                    "noaa_gfs_decode_recovery_bootstrap"
                ),
                "bootstrap_import_root": str((subject.REPO / "src").resolve()),
                "src_package_initializer_must_not_execute": True,
                "bootstrap_source_origin_hash_ast_and_pyc_gates_remain_required": True,
                "real_module_entrypoint_seven_spawn_regression_required": True,
                "revised_code_test_auditor_hash_chain_and_independent_pass_required": True,
                "retry_authorized": False,
            },
        },
    )
    entrypoint_incident_identity = artifact(entrypoint_incident_path, root)
    test_guard_source = """
import socket
import sys

def _deny_network(*_args, **_kwargs):
    raise RuntimeError("network forbidden in immutable recovery test subprocess")

def _deny_network_audit(event, _args):
    if event in {
        "socket.__new__", "socket.bind", "socket.connect", "socket.connect_ex",
        "socket.getaddrinfo", "socket.gethostbyaddr", "socket.gethostbyname",
        "socket.getnameinfo", "socket.sendmsg", "socket.sendto",
    }:
        raise RuntimeError("network audit event forbidden in immutable recovery test subprocess")

sys.addaudithook(_deny_network_audit)
socket.create_connection = _deny_network
socket.getaddrinfo = _deny_network
socket.gethostbyaddr = _deny_network
socket.gethostbyname = _deny_network
socket.getnameinfo = _deny_network
import pytest
raise SystemExit(pytest.main(sys.argv[1:]))
""".strip()
    test_guard_sha = hashlib.sha256(test_guard_source.encode("utf-8")).hexdigest()
    pytest_runs = []
    for relative in subject.RECOVERY_TEST_RELATIVES:
        summary = "1 passed in 0.01s"
        pytest_runs.append(
            {
                "test_file": relative,
                "command": [
                    str(Path(sys.executable).resolve()),
                    "-B",
                    "-c",
                    test_guard_source,
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    relative,
                ],
                "exit_code": 0,
                "summary": summary,
                "stdout_sha256": hashlib.sha256(
                    f"{relative}:{summary}".encode("utf-8")
                ).hexdigest(),
                "stderr_sha256": hashlib.sha256(b"").hexdigest(),
            }
        )
    pytest_summary = " | ".join(
        f"{run['test_file']}: {run['summary']}" for run in pytest_runs
    )
    test_evidence = {
        "schema_version": 1,
        "artifact_type": "DECODE_RECOVERY_IMMUTABLE_CODE_TEST_EVIDENCE",
        "status": "PASS_BOUND_CODE_COMPILE_AND_TEST_SUITE",
        "created_utc": "2026-08-10T17:10:00Z",
        "bound_code_and_test_identities": {
            **recovery_identities,
            **recovery_sealers,
        },
        "source_compile_result": "PASS",
        "compiled_source_identities": [
            artifact(subject.REPO / relative)
            for relative in subject.RECOVERY_COMPILED_RELATIVES
        ],
        "pytest_isolation": "ONE_TEST_FILE_PER_CLEAN_SUBPROCESS",
        "pytest_runs": pytest_runs,
        "pytest_summary": pytest_summary,
        "noncontract_monolithic_diagnostic": (
            subject.RECOVERY_NONCONTRACT_MONOLITHIC_DIAGNOSTIC
        ),
        "isolated_subprocess_count": len(pytest_runs),
        "all_pytest_exit_codes_zero": True,
        "pytest_child_network_guard_sha256": test_guard_sha,
        "pytest_child_network_guard_installed": True,
        "pytest_cacheprovider_disabled": True,
        "required_test_names": subject.RECOVERY_REQUIRED_TEST_NAMES,
        "required_test_names_present": True,
        "python_executable": artifact(Path(sys.executable).resolve()),
        "python_dont_write_bytecode_env": "1",
        "python_dont_write_bytecode_flag": True,
        "python_pycache_prefix_absent": True,
        "bootstrap_matching_pyc_absent_before_tests": True,
        "bootstrap_matching_pyc_absent_after_tests": True,
        "network_requests": 0,
        "labels_read": False,
        "arrays_2024_read": False,
        "arrays_2025_read": False,
        "models_fit": 0,
        "submission_csv_created": False,
    }
    preflight_path = root / subject.RECOVERY_REAL_PREFLIGHT_RELATIVE
    preflight = {
        "schema_version": 1,
        "artifact_type": "TRACK_A_DECODE_RECOVERY_REAL_PROCESS_PREFLIGHT",
        "status": "PASS_REAL_PILOT_AND_PRODUCTION_PATH_EXACT_7_SPAWN_STDLIB_BOOTSTRAP",
        "created_utc": "2026-08-10T17:10:00Z",
        "incident": incident_identity,
        **{key: recovery_identities[key] for key in (
            "recovery_runner", "recovery_module", "recovery_bootstrap",
            "recovery_test", "recovery_bootstrap_test", "recovery_runner_test",
            "postrun_auditor", "postrun_auditor_test",
        )},
        **recovery_sealers,
        "bootstrap_trust_policy": subject.RECOVERY_BOOTSTRAP_TRUST_POLICY,
        "runtime_lock": runtime_lock,
        "bootstrap_import_contract": bootstrap_import_contract,
        "test_evidence": test_evidence,
        "process_scheduling_flake_incident": (
            subject.RECOVERY_PROCESS_SCHEDULING_FLAKE_INCIDENT
        ),
        "sealer_entrypoint_incident": entrypoint_incident_identity,
        "real_grib_pilot": pilot_identity,
        "max_spawn_processes": 7,
        "thread_decoders": 0,
        "network_requests": 0,
        "production_decode_task_worker_real_test": True,
        "stdlib_bootstrap_worker_real_test": True,
        "poisoned_pyc_source_execution_test": True,
        "bootstrap_ast_stdlib_policy_pass": True,
        "python_dont_write_bytecode_env": "1",
        "python_dont_write_bytecode_flag": True,
        "python_pycache_prefix_absent": True,
        "bootstrap_matching_pyc_absent_before_tests": True,
        "bootstrap_matching_pyc_absent_after_tests": True,
        "all_seven_worker_pids_observed": True,
        "pilot_report": pilot_report,
        "production_worker_report": production_report,
        "focused_test_result": pytest_summary,
        "source_compile_result": test_evidence["source_compile_result"],
        "labels_read": False,
        "arrays_2024_read": False,
        "arrays_2025_read": False,
        "models_fit": 0,
        "submission_csv_created": False,
    }
    write_json(preflight_path, preflight)

    authorization_path = root / subject.RECOVERY_AUTH_RELATIVE
    authorization = {
        "schema_version": 1,
        "artifact_type": "NOAA_GFS_NETWORK_ZERO_DECODE_RECOVERY_AUTHORIZATION",
        "status": "AUTHORIZED_PENDING_INDEPENDENT_GO",
        "created_utc": "2026-08-10T17:10:00Z",
        "experiment_id": "noaa_gfs_dminus2_12z_multiseason_target_free_decode_recovery_v1",
        "recovery_attempt_id": recovery_attempt,
        "incident": incident_identity,
        "sealer_entrypoint_incident": entrypoint_incident_identity,
        **recovery_identities,
        "original_authorization": artifact(original_authorization_path, root),
        "original_independent_go": artifact(original_go_path, root),
        "original_failed_lock": artifact(failed_lock, root),
        "original_final_progress": artifact(final_progress, root),
        "original_stderr": artifact(stderr_path, root),
        "original_stdout": artifact(stdout_path, root),
        "field_range_census": original_authorization["field_range_census"],
        "census_manifest": original_authorization["census_manifest"],
        "coordinate_lock": original_authorization["coordinate_lock"],
        "raw_cache_lock": artifact(raw_cache_lock_path, root),
        "real_spawn_preflight": artifact(preflight_path, root),
        "documented_preplan_remnants": remnants,
        "runtime_lock": runtime_lock,
        "bootstrap_trust_policy": subject.RECOVERY_BOOTSTRAP_TRUST_POLICY,
        "bootstrap_import_contract": bootstrap_import_contract,
        "real_grib_pilot": pilot_identity,
        "original_runtime_identity": original_runtime,
        "expected_range_rows": contract.range_rows,
        "expected_range_bytes": contract.range_bytes,
        "max_decode_processes": 7,
        "network_requests_allowed": 0,
        "raw_event_progress_mutation_allowed": False,
        "labels_read": False,
        "2024_arrays_read": False,
        "2025_arrays_read": False,
        "models_fit": 0,
        "submission_csv_created": False,
    }
    write_json(authorization_path, authorization)

    checks = {
        key: True
        for key in {
            "incident_authorization_and_go_chain_exact",
            "recovery_code_test_sealer_identities_exact",
            "runtime_memfs_and_bootstrap_trust_policy_exact",
            "raw_range_event_progress_and_failed_lock_closure_exact",
            "prerecovery_topology_remnants_and_canonical_absence_exact",
            "network_zero_target_free_scope_exact",
            "spawn_worker_core_source_binding_exact",
            "generic_resume_path_absent",
            "transaction_create_if_absent_no_overwrite_exact",
            "real_process_and_test_evidence_exact",
            "postrun_recovery_branch_afterstate_coverage_exact",
            "bootstrap_trust_anchor_limitation_recorded",
        }
    }
    review_path = root / subject.RECOVERY_REVIEW_RELATIVE
    review = {
        "schema_version": 1,
        "artifact_type": "TRACK_A_DECODE_RECOVERY_INDEPENDENT_REVIEW",
        "status": "PASS_NETWORK_ZERO_PROCESS_ISOLATED_RECOVERY_AUTHORIZED",
        "created_utc": "2026-08-10T17:11:00Z",
        "authorization": artifact(authorization_path, root),
        "incident": incident_identity,
        **{key: recovery_identities[key] for key in (
            "recovery_runner", "recovery_module", "recovery_bootstrap",
            "recovery_test", "recovery_bootstrap_test", "recovery_runner_test",
            "postrun_auditor", "postrun_auditor_test",
        )},
        **recovery_sealers,
        "bootstrap_trust_policy": subject.RECOVERY_BOOTSTRAP_TRUST_POLICY,
        "raw_cache_lock": artifact(raw_cache_lock_path, root),
        "real_spawn_preflight": artifact(preflight_path, root),
        "max_decode_processes": 7,
        "network_requests_allowed": 0,
        "verdict": "GO",
        "independent_checks": checks,
    }
    write_json(review_path, review)
    go_path = root / subject.RECOVERY_GO_RELATIVE
    go = {
        "schema_version": 1,
        "artifact_type": "TRACK_A_DECODE_RECOVERY_INDEPENDENT_GO",
        "status": "GO_NETWORK_ZERO_PROCESS_ISOLATED_DECODE_RECOVERY",
        "created_utc": "2026-08-10T17:12:00Z",
        "authorization": artifact(authorization_path, root),
        "incident": incident_identity,
        **{key: recovery_identities[key] for key in (
            "recovery_runner", "recovery_module", "recovery_bootstrap",
            "recovery_test", "recovery_bootstrap_test", "recovery_runner_test",
            "postrun_auditor", "postrun_auditor_test",
        )},
        "bootstrap_trust_policy": subject.RECOVERY_BOOTSTRAP_TRUST_POLICY,
        "raw_cache_lock": artifact(raw_cache_lock_path, root),
        "real_spawn_preflight": artifact(preflight_path, root),
        "independent_review": artifact(review_path, root),
        "network_requests_allowed": 0,
        "max_decode_processes": 7,
    }
    write_json(go_path, go)

    external_paths = {
        "original_runner": runner,
        "recovery_runner": subject.RECOVERY_RUNNER,
        "recovery_module": subject.RECOVERY_MODULE,
        "recovery_bootstrap": subject.RECOVERY_BOOTSTRAP,
        "recovery_test": subject.RECOVERY_TEST,
        "recovery_bootstrap_test": subject.RECOVERY_BOOTSTRAP_TEST,
        "recovery_runner_test": subject.RECOVERY_RUNNER_TEST,
        "postrun_auditor": Path(subject.__file__).resolve(),
        "postrun_auditor_test": subject.AUDITOR_TEST,
        "real_grib_pilot": subject.RECOVERY_REAL_PILOT,
    }
    fixed_snapshot = subject._reconstruct_recovery_fixed_input_snapshot(
        root,
        authorization,
        go,
        failed_attempt,
        contract,
        external_paths,
    )

    history_root = root / "decoded" / "recovery_history"
    progress_root = root / "decoded" / "recovery_progress"
    mutex_payload = {
        "artifact_type": "DECODE_RECOVERY_ACTIVE_LOCK",
        "attempt_id": recovery_attempt,
        "pid": 999,
        "created_utc": "2026-08-10T17:13:00Z",
        "network_requests": 0,
    }
    raw_history = history_root / f"{recovery_attempt}__raw_mutex__complete.lock"
    decoded_history = history_root / f"{recovery_attempt}__decode_mutex__complete.lock"
    write_json(raw_history, mutex_payload)
    write_json(decoded_history, mutex_payload)
    progress_paths = []
    for count in subject.recovery_progress_counts(contract):
        progress_path = (
            progress_root
            / f"decoded_messages__{recovery_attempt}__{count:06d}.json"
        )
        write_json(
            progress_path,
            {
                "artifact_type": "DECODE_RECOVERY_PROGRESS",
                "attempt_id": recovery_attempt,
                "completed_messages": count,
                "network_requests": 0,
                "process_model": "spawn",
                "max_processes": 7,
            },
        )
        progress_paths.append(progress_path)
    locked_topology = {
        "raw_top_level_directories": sorted(
            {"launch_history", "output_transactions", "progress", "ranges", "request_events"}
        ),
        "documented_preplan_transactions": prerecovery_topology[
            "documented_preplan_transactions"
        ],
        "raw_mutex": {**artifact(raw_history, root), "path": "raw/RAW_LAUNCH_ACTIVE.lock"},
        "decoded_mutex": {
            **artifact(decoded_history, root),
            "path": "decoded/DECODE_RECOVERY_ACTIVE.lock",
        },
        "mutex_payload": mutex_payload,
        "canonical_outputs_absent": True,
    }
    recovery_provenance = {
        "incident": incident_identity,
        "recovery_authorization": artifact(authorization_path, root),
        "independent_recovery_go": artifact(go_path, root),
        "recovery_runner": recovery_identities["recovery_runner"],
        "recovery_module": recovery_identities["recovery_module"],
        "recovery_bootstrap": recovery_identities["recovery_bootstrap"],
        "bootstrap_trust_policy": subject.RECOVERY_BOOTSTRAP_TRUST_POLICY,
        "bootstrap_import_contract": bootstrap_import_contract,
        "recovery_test": recovery_identities["recovery_test"],
        "recovery_bootstrap_test": recovery_identities["recovery_bootstrap_test"],
        "recovery_runner_test": recovery_identities["recovery_runner_test"],
        "postrun_auditor": recovery_identities["postrun_auditor"],
        "postrun_auditor_test": recovery_identities["postrun_auditor_test"],
        "original_frozen_runner": recovery_identities["original_runner"],
        "raw_cache_lock": artifact(raw_cache_lock_path, root),
        "real_spawn_preflight": artifact(preflight_path, root),
        "pilot_runtime_report": pilot_report,
        "fixed_inputs_initial": fixed_snapshot,
        "locked_prestate_topology": locked_topology,
        "locked_fixed_inputs": fixed_snapshot,
    }

    physical = json.loads(outputs["audit"].read_text(encoding="utf-8"))
    physical.update(
        {
            "artifact_type": "TRACK_A_PHYSICAL_AND_COVERAGE_AUDIT_RECOVERED",
            "recovery_provenance": recovery_provenance,
            "decode_process_model": "WINDOWS_SPAWN_PROCESS_ISOLATION_SERIAL_ECCODES_PER_PROCESS",
            "decode_worker_pids": list(range(801, 808)),
            "decode_process_count": 7,
        }
    )
    write_json(outputs["audit"], physical)
    access = {
        "artifact_type": "TRACK_A_RAW_ACCESS_LEDGER_RECOVERED_NETWORK_ZERO",
        "expected_ranges": contract.range_rows,
        "expected_bytes": contract.range_bytes,
        "network_requests_this_invocation": 0,
        "network_bytes_this_invocation": 0,
        "raw_cache_reuses": contract.range_rows,
        "decode_processes": 7,
        "decode_threads": 0,
        "before_inventories": inventories,
        "after_inventories": inventories,
        "inventories_equal": True,
        "raw_event_progress_mutations": 0,
        "documented_preplan_remnants_preserved": remnants,
        "labels_read": False,
        "2024_arrays_read": False,
        "2025_arrays_read": False,
        "models_fit": 0,
        "submission_csv_created": False,
        "recovery_provenance": recovery_provenance,
    }
    write_json(outputs["access"], access)
    lock = {
        "artifact_type": "TARGET_FREE_DECODED_MATRIX_LOCK_RECOVERED_NETWORK_ZERO",
        "schema_version": 1,
        "recovery_provenance": recovery_provenance,
        "runtime_identity": runtime_report,
        "raw_manifest_parquet": artifact(outputs["raw_parquet"], root),
        "raw_manifest_csv": artifact(outputs["raw_csv"], root),
        "site_matrix": artifact(outputs["site"], root),
        "group_matrix": artifact(outputs["group"], root),
        "physical_and_coverage_audit": artifact(outputs["audit"], root),
        "access_ledger": artifact(outputs["access"], root),
        "target_free_duplicate_estimator_may_start": True,
        "network_requests": 0,
        "labels_read": False,
    }
    write_json(outputs["lock"], lock)
    manifest = {
        "artifact_type": "NOAA_GFS_MULTISEASON_RAW_AND_DECODE_MANIFEST_RECOVERED",
        "schema_version": 1,
        "created_utc": "2026-08-10T17:14:00Z",
        "launch_attempt_id": recovery_attempt,
        "recovery_provenance": recovery_provenance,
        "runtime_identity": runtime_report,
        "decoded_matrix_lock": artifact(outputs["lock"], root),
        "exact_ranges": contract.range_rows,
        "exact_bytes": contract.range_bytes,
        "decode_processes": 7,
        "decode_threads": 0,
        "network_requests": 0,
        "labels_read": False,
        "2024_arrays_read": False,
        "2025_arrays_read": False,
        "models_fit": 0,
        "submission_csv_created": False,
    }
    write_json(outputs["manifest"], manifest)

    transaction_root = root / "raw" / "output_transactions"
    plan_path = transaction_root / f"{recovery_attempt}__plan.json"
    items = []
    for name, destination in sorted(outputs.items()):
        relative = destination.relative_to(root).as_posix()
        staged_relative = (
            f"raw/output_transactions/{recovery_attempt}/staged/{relative}"
        )
        (root / staged_relative).parent.mkdir(parents=True, exist_ok=True)
        actual = artifact(destination, root)
        items.append(
            {
                "name": name,
                "staged_path": staged_relative,
                "destination_path": relative,
                "size_bytes": actual["size_bytes"],
                "sha256": actual["sha256"],
            }
        )
    write_json(
        plan_path,
        {
            "artifact_type": "DECODE_RECOVERY_OUTPUT_TRANSACTION_PLAN",
            "schema_version": 1,
            "attempt_id": recovery_attempt,
            "created_utc": "2026-08-10T17:15:00Z",
            "recovery_authorization": artifact(authorization_path, root),
            "independent_recovery_go": artifact(go_path, root),
            "incident": incident_identity,
            "items": items,
            "canonical_outputs_absent_before_plan": True,
            "all_10368_messages_decoded_before_plan": True,
            "physical_audit_completed_before_plan": True,
            "network_requests": 0,
        },
    )
    commit_path = transaction_root / f"{recovery_attempt}__committed.json"
    write_json(
        commit_path,
        {
            "artifact_type": "DECODE_RECOVERY_OUTPUT_TRANSACTION_COMMIT",
            "schema_version": 1,
            "attempt_id": recovery_attempt,
            "plan": artifact(plan_path, root),
            "outputs": {name: artifact(path, root) for name, path in outputs.items()},
            "atomic_method": "os.link_create_if_absent_then_unlink_staging",
            "all_outputs_reopened_and_rehashed": True,
            "network_requests": 0,
        },
    )
    postcommit_path = (
        history_root / f"{recovery_attempt}__postcommit_input_audit.json"
    )
    write_json(
        postcommit_path,
        {
            "artifact_type": "DECODE_RECOVERY_POSTCOMMIT_INPUT_AUDIT",
            "attempt_id": recovery_attempt,
            "initial": fixed_snapshot,
            "preplan": fixed_snapshot,
            "postcommit": fixed_snapshot,
            "all_equal": True,
            "network_requests": 0,
        },
    )
    return root, runner, contract


def test_active_lock_conditionally_skips_before_any_producer_closure_read(tmp_path: Path) -> None:
    root = tmp_path / "active"
    lock = root / "raw" / "RAW_LAUNCH_ACTIVE.lock"
    write_json(lock, {"attempt_id": "active", "pid": 9, "created_utc": "now"})
    before = snapshot(root)
    with pytest.raises(subject.AuditNotReady) as exc_info:
        subject.audit(root, runner_path=tmp_path / "absent.py")
    assert exc_info.value.active_lock["attempt_id"] == "active"
    assert snapshot(root) == before


def test_complete_synthetic_producer_passes_and_audit_is_read_only(tmp_path: Path) -> None:
    root, runner, contract = build_complete_fixture(tmp_path)
    before = snapshot(tmp_path)
    report = audit_fixture(root, runner, contract)
    assert report["status"] == "PASS_PRODUCER_POSTRUN_READONLY_AUDIT"
    assert report["raw_closure"] == {
        "objects": 1,
        "ranges": 9,
        "bytes": 81,
        "sidecars": 9,
        "request_attempt_starts": 11,
        "request_attempt_completions": 10,
        "request_attempt_errors": 0,
        "request_attempts_without_terminal_event": 1,
        "max_global_attempt_number": 11,
    }
    assert report["decoded_closure"]["downstream_target_free_estimator_may_start"] is True
    assert report["launch_and_transaction"]["transaction_inventory_recursively_closed"] is True
    assert report["launch_and_transaction"]["planned_staged_files_present"] == 0
    assert report["audit_network_requests"] == report["audit_files_written"] == 0
    assert snapshot(tmp_path) == before


def test_complete_synthetic_recovery_passes_exact_afterstate_and_is_read_only(
    tmp_path: Path,
) -> None:
    root, runner, contract = build_recovered_fixture(tmp_path)
    before = snapshot(tmp_path)
    report = audit_fixture(root, runner, contract)
    assert report["status"] == "PASS_PRODUCER_POSTRUN_READONLY_AUDIT"
    assert report["launch_and_transaction"]["transaction_inventory_recursively_closed"] is True
    assert report["launch_and_transaction"]["planned_staged_files_present"] == 0
    access = report["access_and_zero_target_facts"]
    assert access["recovery_mode"] is True
    assert access["network_ranges_this_final_invocation"] == 0
    assert access["cached_ranges_this_final_invocation"] == 9
    assert access["decode_processes"] == 7
    assert access["decode_threads"] == 0
    assert access["recovery_progress_checkpoint_count"] == 2
    assert access["original_raw_progress_checkpoint_count"] == 2
    assert access["recovery_history_file_count"] == 3
    assert access["incident_bound_preplan_remnant_count"] == 2
    assert access["recovery_current_bootstrap_pyc_gate"] == {
        "status": "PASS_CURRENT_MATCHING_BOOTSTRAP_PYC_EXACT_ZERO",
        "source_root": str((subject.REPO / "src").resolve()),
        "legacy_relative_path": "noaa_gfs_decode_recovery_bootstrap.pyc",
        "cache_tag_glob": (
            "__pycache__/noaa_gfs_decode_recovery_bootstrap.*.pyc"
        ),
        "matching_pyc_files": 0,
    }
    assert access["recovery_current_bootstrap_import_contract"] == (
        subject._expected_recovery_bootstrap_import_contract()
    )
    temporary_gate = report["launch_and_transaction"][
        "recovery_control_temporary_file_gate"
    ]
    assert temporary_gate["start"] == temporary_gate["end"]
    assert temporary_gate["start"]["matching_temporary_files"] == 0
    assert temporary_gate["start"]["checked_destination_count"] == 5
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("relative", subject.RECOVERY_CONTROL_RELATIVES)
def test_each_recovery_control_publisher_temporary_prefix_fails_closed(
    tmp_path: Path, relative: str
) -> None:
    root = tmp_path / "producer"
    destination = root / relative
    candidate = destination.with_name(
        f"{destination.name}.tmp.1234.{'a' * 20}"
    )
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_bytes(b"abandoned no-overwrite publication")
    with pytest.raises(
        subject.AuditFailure,
        match="recovery control publisher temporary file remains",
    ):
        subject._audit_recovery_control_temporary_files_absent(root)


def test_recovery_control_publisher_temporary_sibling_blocks_full_audit(
    tmp_path: Path,
) -> None:
    root, runner, contract = build_recovered_fixture(tmp_path)
    destination = root / subject.RECOVERY_AUTH_RELATIVE
    candidate = destination.with_name(
        f"{destination.name}.tmp.7777.{'b' * 20}"
    )
    candidate.write_bytes(b"abandoned authorization publication")
    with pytest.raises(
        subject.AuditFailure,
        match="recovery control publisher temporary file remains",
    ):
        audit_fixture(root, runner, contract)


@pytest.mark.parametrize(
    "cache_namespace", ("legacy", "cache_tag", "mixed_case_cache_tag")
)
def test_recovery_current_matching_bootstrap_pyc_fails_closed(
    tmp_path: Path, cache_namespace: str
) -> None:
    source_root = tmp_path / "src"
    source_root.mkdir()
    if cache_namespace == "legacy":
        candidate = source_root / "noaa_gfs_decode_recovery_bootstrap.pyc"
    elif cache_namespace == "cache_tag":
        candidate = (
            source_root
            / "__pycache__"
            / "noaa_gfs_decode_recovery_bootstrap.cp313.pyc"
        )
        candidate.parent.mkdir()
    else:
        candidate = (
            source_root
            / "__pycache__"
            / "NOAA_GFS_DECODE_RECOVERY_BOOTSTRAP.CP313.PYC"
        )
        candidate.parent.mkdir()
    candidate.write_bytes(b"forbidden bytecode cache")
    with pytest.raises(
        subject.AuditFailure,
        match="matching recovery bootstrap pyc exists at postrun audit",
    ):
        subject._audit_current_recovery_bootstrap_pyc_absence(source_root)


def test_recovery_bootstrap_import_contract_tamper_fails_closed() -> None:
    contract = subject._expected_recovery_bootstrap_import_contract()
    contract["module_name"] = "src.noaa_gfs_decode_recovery_bootstrap"
    with pytest.raises(subject.AuditFailure, match="bootstrap import contract"):
        subject._validate_recovery_bootstrap_import_contract(contract)


@pytest.mark.parametrize(
    ("candidate_name", "candidate_is_directory"),
    (
        ("noaa_gfs_decode_recovery_bootstrap", True),
        ("noaa_gfs_decode_recovery_bootstrap.pyd", False),
        ("noaa_gfs_decode_recovery_bootstrap.py.bak", False),
        ("NOAA_GFS_DECODE_RECOVERY_BOOTSTRAP.CP313-WIN_AMD64.PYD", False),
    ),
)
def test_recovery_bootstrap_same_alias_competitor_fails_postrun_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    candidate_name: str,
    candidate_is_directory: bool,
) -> None:
    workspace = tmp_path / "workspace"
    source_root = workspace / "src"
    source_root.mkdir(parents=True)
    source = source_root / "noaa_gfs_decode_recovery_bootstrap.py"
    source.write_bytes(subject.RECOVERY_BOOTSTRAP.read_bytes())
    monkeypatch.setattr(subject, "REPO", workspace)
    monkeypatch.setattr(subject, "RECOVERY_BOOTSTRAP", source)
    assert subject._expected_recovery_bootstrap_import_contract()[
        "top_level_module_competing_candidates"
    ] == []
    candidate = source_root / candidate_name
    if candidate_is_directory:
        candidate.mkdir()
    else:
        candidate.write_bytes(b"forbidden same-alias import candidate")
    with pytest.raises(
        subject.AuditFailure,
        match="top-level bootstrap competitors",
    ):
        subject._expected_recovery_bootstrap_import_contract()


def test_recovery_sealer_entrypoint_incident_remediation_tamper_fails(
    tmp_path: Path,
) -> None:
    root, _runner, contract = build_recovered_fixture(tmp_path)
    incident_path = root / subject.RECOVERY_SEALER_ENTRYPOINT_INCIDENT_RELATIVE
    incident = json.loads(incident_path.read_text(encoding="utf-8"))
    incident["mandatory_remediation"]["retry_authorized"] = True
    write_json(incident_path, incident)
    authorization = json.loads(
        (root / subject.RECOVERY_AUTH_RELATIVE).read_text(encoding="utf-8")
    )
    with pytest.raises(subject.AuditFailure, match="mandatory remediation"):
        subject._validate_recovery_sealer_entrypoint_incident(
            incident,
            artifact(incident_path, root),
            root,
            authorization,
            contract,
        )


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("transaction_junk", "unplanned recovery transaction file"),
        ("new_staged_file", "recovery staged output remains"),
        ("transaction_empty_dir", "unplanned recovery transaction directory"),
        ("history_extra", "recovered decoded file inventory mismatch"),
        ("progress_extra", "recovered decoded file inventory mismatch"),
        ("raw_history_extra", "preserve only the original failed lock"),
    ),
)
def test_recovery_recursive_inventory_rejects_every_unplanned_afterstate(
    tmp_path: Path, case: str, message: str
) -> None:
    root, runner, contract = build_recovered_fixture(tmp_path)
    transaction_root = root / "raw" / "output_transactions"
    recovery_attempt = "decode_recovery_v1__synthetic0001"
    if case == "transaction_junk":
        write_json(transaction_root / "junk.json", {"unexpected": True})
    elif case == "new_staged_file":
        staged = (
            transaction_root
            / recovery_attempt
            / "staged"
            / "raw"
            / "RAW_RANGE_MANIFEST.csv"
        )
        staged.write_bytes(b"forbidden recovery staging")
    elif case == "transaction_empty_dir":
        (transaction_root / "abandoned_empty").mkdir()
    elif case == "history_extra":
        write_json(
            root / "decoded" / "recovery_history" / "unexpected__failed.lock",
            {"unexpected": True},
        )
    elif case == "progress_extra":
        write_json(
            root
            / "decoded"
            / "recovery_progress"
            / f"decoded_messages__{recovery_attempt}__000001.json",
            {
                "artifact_type": "DECODE_RECOVERY_PROGRESS",
                "attempt_id": recovery_attempt,
                "completed_messages": 1,
                "network_requests": 0,
                "process_model": "spawn",
                "max_processes": 7,
            },
        )
    else:
        write_json(
            root / "raw" / "launch_history" / "unexpected__complete.lock",
            {"attempt_id": "unexpected", "pid": 1, "created_utc": "now"},
        )
    with pytest.raises(subject.AuditFailure, match=message):
        audit_fixture(root, runner, contract)


def test_recovery_incident_bound_remnant_content_tamper_fails(tmp_path: Path) -> None:
    root, runner, contract = build_recovered_fixture(tmp_path)
    remnant = (
        root
        / "raw"
        / "output_transactions"
        / subject.RECOVERY_FAILED_ATTEMPT
        / "staged"
        / "raw"
        / "RAW_RANGE_MANIFEST.csv"
    )
    remnant.write_bytes(remnant.read_bytes() + b"tamper")
    with pytest.raises(subject.AuditFailure, match="preplan remnant"):
        audit_fixture(root, runner, contract)


def test_recovery_mutex_history_must_be_exactly_byte_identical(tmp_path: Path) -> None:
    root, runner, contract = build_recovered_fixture(tmp_path)
    decode_lock = next(
        (root / "decoded" / "recovery_history").glob("*__decode_mutex__complete.lock")
    )
    payload = json.loads(decode_lock.read_text(encoding="utf-8"))
    payload["pid"] += 1
    write_json(decode_lock, payload)
    with pytest.raises(subject.AuditFailure, match="byte-identical"):
        audit_fixture(root, runner, contract)


def test_recovery_postcommit_snapshot_tamper_fails(tmp_path: Path) -> None:
    root, runner, contract = build_recovered_fixture(tmp_path)
    postcommit = next(
        (root / "decoded" / "recovery_history").glob("*__postcommit_input_audit.json")
    )
    payload = json.loads(postcommit.read_text(encoding="utf-8"))
    payload["postcommit"]["combined_sha256"] = "0" * 64
    write_json(postcommit, payload)
    with pytest.raises(subject.AuditFailure, match="fixed-input snapshots differ"):
        audit_fixture(root, runner, contract)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("network_requests_this_invocation", 1),
        ("network_bytes_this_invocation", 1),
        ("raw_cache_reuses", 8),
        ("inventories_equal", False),
    ),
)
def test_recovered_access_ledger_tamper_fails_after_all_hash_envelopes_rebound(
    tmp_path: Path, field: str, value: object
) -> None:
    root, runner, contract = build_recovered_fixture(tmp_path)
    access_path = root / subject.OUTPUT_RELATIVE_PATHS["access"]
    access = json.loads(access_path.read_text(encoding="utf-8"))
    access[field] = value
    write_json(access_path, access)
    rebind_canonical_transaction(root)
    with pytest.raises(subject.AuditFailure, match="recovered raw access ledger accounting"):
        audit_fixture(root, runner, contract)


def test_recovered_locked_prestate_tamper_fails_after_all_envelopes_rebound(
    tmp_path: Path,
) -> None:
    root, runner, contract = build_recovered_fixture(tmp_path)

    def mutate(provenance: dict[str, object]) -> None:
        provenance["locked_prestate_topology"]["canonical_outputs_absent"] = False

    mutate_recovery_provenance_and_rebind(root, mutate)
    with pytest.raises(subject.AuditFailure, match="locked recovery prestate topology"):
        audit_fixture(root, runner, contract)


def test_recovery_independent_review_false_check_is_hash_bound_and_fails(
    tmp_path: Path,
) -> None:
    root, runner, contract = build_recovered_fixture(tmp_path)
    review_path = root / subject.RECOVERY_REVIEW_RELATIVE
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["independent_checks"]["generic_resume_path_absent"] = False
    write_json(review_path, review)
    with pytest.raises(subject.AuditFailure, match="independent recovery review"):
        audit_fixture(root, runner, contract)


def test_recovery_test_evidence_rejects_self_consistent_nonapproved_network_guard(
    tmp_path: Path,
) -> None:
    root, _runner, _contract = build_recovered_fixture(tmp_path)
    preflight = json.loads(
        (root / subject.RECOVERY_REAL_PREFLIGHT_RELATIVE).read_text(encoding="utf-8")
    )
    evidence = json.loads(json.dumps(preflight["test_evidence"]))
    malicious_guard = "import pytest; raise SystemExit(pytest.main())"
    evidence["pytest_child_network_guard_sha256"] = hashlib.sha256(
        malicious_guard.encode("utf-8")
    ).hexdigest()
    for run in evidence["pytest_runs"]:
        run["command"][3] = malicious_guard
    bound = evidence["bound_code_and_test_identities"]
    external_actuals = {
        key: bound[key]
        for key in (
            "original_runner",
            "recovery_runner",
            "recovery_module",
            "recovery_bootstrap",
            "recovery_test",
            "recovery_bootstrap_test",
            "recovery_runner_test",
            "postrun_auditor",
            "postrun_auditor_test",
        )
    }
    sealer_actuals = {
        key: bound[key]
        for key in ("recovery_sealer", "recovery_sealer_test")
    }
    with pytest.raises(subject.AuditFailure, match="approved network-deny"):
        subject._validate_recovery_test_evidence(
            evidence, preflight, external_actuals, sealer_actuals
        )


def test_recovery_test_evidence_rejects_selected_noncontract_diagnostic(
    tmp_path: Path,
) -> None:
    root, _runner, _contract = build_recovered_fixture(tmp_path)
    preflight = json.loads(
        (root / subject.RECOVERY_REAL_PREFLIGHT_RELATIVE).read_text(encoding="utf-8")
    )
    evidence = json.loads(json.dumps(preflight["test_evidence"]))
    evidence["noncontract_monolithic_diagnostic"][
        "used_as_contract_evidence"
    ] = True
    bound = evidence["bound_code_and_test_identities"]
    external_actuals = {
        key: bound[key]
        for key in (
            "original_runner",
            "recovery_runner",
            "recovery_module",
            "recovery_bootstrap",
            "recovery_test",
            "recovery_bootstrap_test",
            "recovery_runner_test",
            "postrun_auditor",
            "postrun_auditor_test",
        )
    }
    sealer_actuals = {
        key: bound[key]
        for key in ("recovery_sealer", "recovery_sealer_test")
    }
    with pytest.raises(subject.AuditFailure, match="test-evidence gates"):
        subject._validate_recovery_test_evidence(
            evidence, preflight, external_actuals, sealer_actuals
        )


def test_all_nine_messages_drive_decode_offline_grib_with_fake_eccodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import numpy as np

    root, runner, contract = build_complete_fixture(tmp_path)
    longitude = np.arange(1440, dtype=float) * 0.25
    latitude = 90.0 - np.arange(721, dtype=float) * 0.25
    plane = (
        5.0 * (latitude[:, None] - 37.1)
        + 5.0 * (longitude[None, :] - 128.1)
    )
    arrays = {
        feature: np.asarray(
            plane + (100.0 if index == 0 else float(index)), dtype=float
        ).reshape(-1)
        for index, feature in enumerate(subject.FEATURE_COLUMNS)
    }
    tracker = {"open_calls": 0, "array_calls": 0, "active": 0, "max_active": 0}
    tracker_lock = threading.Lock()

    def feature_for(handle: Path) -> str:
        return handle.name.removesuffix(".grib2")

    def codes_grib_new_from_file(stream: object) -> Path | None:
        with tracker_lock:
            tracker["open_calls"] += 1
        if stream.tell() != 0:
            return None
        stream.seek(0, 2)
        return Path(stream.name)

    def codes_get(handle: Path, key: str) -> object:
        feature = feature_for(handle)
        variable, level_token = feature.split("_", 1)
        parameter = {
            "HPBL": ("unknown", 0, 3, 196),
            "UGRD": ("u", 0, 2, 2),
            "VGRD": ("v", 0, 2, 3),
        }[variable]
        level = "surface" if level_token == "surface" else level_token.removesuffix("mb")
        values = {
            "edition": 2,
            "discipline": parameter[1],
            "parameterCategory": parameter[2],
            "parameterNumber": parameter[3],
            "shortName": parameter[0],
            "typeOfLevel": "surface" if level == "surface" else "isobaricInhPa",
            "level": 0 if level == "surface" else int(level),
            "dataDate": 20220103,
            "dataTime": 1200,
            "forecastTime": 28,
            "stepUnits": 1,
            "indicatorOfUnitOfTimeRange": 1,
            "validityDate": 20220104,
            "validityTime": 1600,
            "gridType": "regular_ll",
            "gridDefinitionTemplateNumber": 0,
            "Ni": 1440,
            "Nj": 721,
            "numberOfDataPoints": 1_038_240,
            "latitudeOfFirstGridPointInDegrees": 90.0,
            "longitudeOfFirstGridPointInDegrees": 0.0,
            "latitudeOfLastGridPointInDegrees": -90.0,
            "longitudeOfLastGridPointInDegrees": 359.75,
            "iDirectionIncrementInDegrees": 0.25,
            "jDirectionIncrementInDegrees": 0.25,
            "iScansNegatively": 0,
            "jScansPositively": 0,
            "jPointsAreConsecutive": 0,
            "alternativeRowScanning": 0,
            "missingValue": 9_999.0,
        }
        return values[key]

    def codes_get_array(handle: Path, key: str) -> object:
        assert key == "values"
        with tracker_lock:
            tracker["array_calls"] += 1
            tracker["active"] += 1
            tracker["max_active"] = max(tracker["max_active"], tracker["active"])
        try:
            time.sleep(0.01)
            return arrays[feature_for(handle)]
        finally:
            with tracker_lock:
                tracker["active"] -= 1

    fake_eccodes = types.SimpleNamespace(
        codes_grib_new_from_file=codes_grib_new_from_file,
        codes_get=codes_get,
        codes_get_array=codes_get_array,
        codes_release=lambda _handle: None,
    )
    monkeypatch.setitem(sys.modules, "eccodes", fake_eccodes)
    report = subject.audit(
        root,
        contract=contract,
        runner_path=runner,
        _test_message_decoder=subject.decode_offline_grib_message,
    )
    replay = report["full_offline_raw_redecode"]
    assert replay["messages"] == 9
    assert replay["site_value_comparisons"] == 18
    assert replay["group_value_comparisons"] == 18
    assert replay["metadata_and_single_message_gates"] == 9
    assert tracker["open_calls"] == 18
    assert tracker["array_calls"] == 9
    assert tracker["max_active"] == 1
    assert replay["execution_backend"] == "serial_synthetic_test_override"
    assert replay["max_streaming_workers"] == 1


def test_canonical_value_tamper_rebound_fails_at_offline_replay(tmp_path: Path) -> None:
    root, runner, contract = build_complete_fixture(tmp_path)
    site_path = root / subject.OUTPUT_RELATIVE_PATHS["site"]
    group_path = root / subject.OUTPUT_RELATIVE_PATHS["group"]
    audit_path = root / subject.OUTPUT_RELATIVE_PATHS["audit"]
    site = pd.read_parquet(site_path)
    group = pd.read_parquet(group_path)
    site.loc[site["site_id"] == 1, "HPBL_surface"] = 100.5
    group.loc[group["group"] == "group_a", "HPBL_surface"] = 100.5
    site.to_parquet(site_path, index=False)
    group.to_parquet(group_path, index=False)
    physical = json.loads(audit_path.read_text(encoding="utf-8"))
    physical["families"]["PBL_HEIGHT"]["minimum"] = 100.5
    write_json(audit_path, physical)
    rebind_canonical_transaction(root)
    with pytest.raises(subject.AuditFailure, match="offline replay"):
        audit_fixture(root, runner, contract)


def test_real_local_pilot_grib_decodes_with_frozen_digest() -> None:
    pilot = (
        subject.REPO
        / "artifacts"
        / "baram2026_ncei_scada_research_20260810_210756"
        / "track_a"
        / "provenance"
        / "raw"
        / "f039"
        / "gfs.t12z.pgrb2.0p25.f039.HPBL_surface.grib2"
    )
    coordinate_path = (
        subject.REPO
        / "artifacts"
        / "baram2026_ncei_scada_longrun_20260810_v2"
        / "prereg"
        / "authoritative_turbine_coordinate_lock_v1.json"
    )
    assert pilot.stat().st_size == 1_455_839
    assert subject.sha256_file(pilot) == "575fe7876ab93e5470c43dc09c78895385925f5662cca75bd0d8c7f91e677407"
    coordinates = json.loads(coordinate_path.read_text(encoding="utf-8"))["sites"]
    decoded = subject.decode_offline_grib_message(
        {
            "path": pilot,
            "row": {
                "run_init_utc": "2023-07-01T12:00:00Z",
                "valid_time_utc": "2023-07-03T03:00:00Z",
                "forecast_hour": 39,
                "variable": "HPBL",
                "level": "surface",
            },
        },
        coordinates,
    )
    assert decoded["metadata_verified"] is True
    assert decoded["single_message_verified"] is True
    assert decoded["feature"] == "HPBL_surface"
    assert len(decoded["site_values"]) == 17
    assert subject.canonical_payload_sha256(decoded["site_values"]) == "25ca80932d8da2accd4bc9de86b064a9f2f032a8a573d74edee714d889a4dab4"


def test_full_replay_uses_spawned_seven_process_pool_on_real_local_pilot(
    tmp_path: Path,
) -> None:
    pilot = (
        subject.REPO
        / "artifacts"
        / "baram2026_ncei_scada_research_20260810_210756"
        / "track_a"
        / "provenance"
        / "raw"
        / "f039"
        / "gfs.t12z.pgrb2.0p25.f039.HPBL_surface.grib2"
    )
    coordinate_path = (
        subject.REPO
        / "artifacts"
        / "baram2026_ncei_scada_longrun_20260810_v2"
        / "prereg"
        / "authoritative_turbine_coordinate_lock_v1.json"
    )
    coordinate = json.loads(coordinate_path.read_text(encoding="utf-8"))
    runtime_identity = json.loads(
        (
            coordinate_path.parents[1]
            / "prereg"
            / "raw_launch_authorization_v1.json"
        ).read_text(encoding="utf-8")
    )["runtime_identity"]
    sites = coordinate["sites"]
    info = {
        "path": pilot,
        "row": {
            "run_init_utc": "2023-07-01T12:00:00Z",
            "valid_time_utc": "2023-07-03T03:00:00Z",
            "forecast_hour": 39,
            "variable": "HPBL",
            "level": "surface",
        },
    }
    expected = subject.decode_offline_grib_message(info, sites)["site_values"]
    site_rows = []
    for site, value in zip(sites, expected):
        site_rows.append(
            {
                "valid_time_utc": "2023-07-03T03:00:00Z",
                "target_operating_day_kst": "2023-07-03",
                "run_init_utc": "2023-07-01T12:00:00Z",
                "forecast_hour": 39,
                "site_id": int(site["site_id"]),
                "group": site["group"],
                "latitude": float(site["latitude"]),
                "longitude": float(site["longitude"]),
                "capacity_mw": float(site["capacity_mw"]),
                **{
                    feature: (float(value) if feature == "HPBL_surface" else 0.0)
                    for feature in subject.FEATURE_COLUMNS
                },
            }
        )
    group_rows = []
    for group in subject.PRODUCTION_CONTRACT.groups:
        members = [row for row in site_rows if row["group"] == group.name]
        total = sum(float(row["capacity_mw"]) for row in members)
        hpbl = sum(
            float(row["HPBL_surface"]) * float(row["capacity_mw"])
            for row in members
        ) / total
        group_rows.append(
            {
                "valid_time_utc": "2023-07-03T03:00:00Z",
                "target_operating_day_kst": "2023-07-03",
                "run_init_utc": "2023-07-01T12:00:00Z",
                "forecast_hour": 39,
                "group": group.name,
                "site_count": group.site_count,
                "capacity_mw": group.capacity_mw,
                **{
                    feature: (hpbl if feature == "HPBL_surface" else 0.0)
                    for feature in subject.FEATURE_COLUMNS
                },
            }
        )
    site_path = tmp_path / "pilot_site.parquet"
    group_path = tmp_path / "pilot_group.parquet"
    pd.DataFrame(site_rows).to_parquet(site_path, index=False)
    pd.DataFrame(group_rows).to_parquet(group_path, index=False)
    contract = subject.AuditContract(
        range_rows=7,
        range_bytes=7,
        object_rows=1,
        site_count=17,
        groups=subject.PRODUCTION_CONTRACT.groups,
        operating_years=None,
        operating_days_of_month=None,
        forecast_hours=None,
    )
    raw_info = {
        ("pilot", str(index), "HPBL_surface"): dict(info)
        for index in range(subject.DECODE_MAX_WORKERS)
    }
    replay = subject._audit_full_offline_redecode(
        raw_info,
        coordinate,
        {"site": site_path, "group": group_path},
        contract,
        decoder=subject.decode_offline_grib_message,
        synthetic_decoder_override=False,
        expected_runtime_identity_sha256=subject.canonical_payload_sha256(
            runtime_identity
        ),
    )
    assert replay["messages"] == 7
    assert replay["site_value_comparisons"] == 119
    assert replay["group_value_comparisons"] == 21
    assert replay["execution_backend"] == "spawn_process_pool_eccodes_serial_per_process"
    assert replay["max_streaming_workers"] == subject.DECODE_MAX_WORKERS == 7
    assert replay["decoder_processes_observed"] == 7


def test_abandoned_preplan_staging_file_fails_transaction_inventory_closure(
    tmp_path: Path,
) -> None:
    root, runner, contract = build_complete_fixture(tmp_path)
    abandoned = (
        root
        / "raw"
        / "output_transactions"
        / "abandoned_attempt"
        / "staged"
        / "decoded"
        / "abandoned.parquet"
    )
    abandoned.parent.mkdir(parents=True, exist_ok=True)
    abandoned.write_bytes(b"unplanned pre-plan staging payload")
    with pytest.raises(subject.AuditFailure, match="unplanned output transaction file"):
        audit_fixture(root, runner, contract)


@pytest.mark.parametrize(
    "relative",
    (
        "raw/junk.bin",
        "raw/ranges/junk.bin",
        "raw/request_events/junk.bin",
        "raw/progress/junk.bin",
        "decoded/junk.bin",
    ),
)
def test_unrecognized_files_fail_recursive_filesystem_closure(
    tmp_path: Path, relative: str
) -> None:
    root, runner, contract = build_complete_fixture(tmp_path)
    junk = root / relative
    junk.parent.mkdir(parents=True, exist_ok=True)
    junk.write_bytes(b"unrecognized")
    with pytest.raises(subject.AuditFailure):
        audit_fixture(root, runner, contract)


def test_unrecognized_launch_history_file_fails(tmp_path: Path) -> None:
    root, runner, contract = build_complete_fixture(tmp_path)
    write_json(
        root / "raw" / "launch_history" / "junk.lock",
        {"attempt_id": "junk", "pid": 1, "created_utc": "now"},
    )
    with pytest.raises(subject.AuditFailure, match="unexpected launch history filename"):
        audit_fixture(root, runner, contract)


def test_duplicate_launch_status_for_one_attempt_fails(tmp_path: Path) -> None:
    root, runner, contract = build_complete_fixture(tmp_path)
    complete = next((root / "raw" / "launch_history").glob("*__complete.lock"))
    attempt_id = complete.name.removesuffix("__complete.lock")
    write_json(
        complete.with_name(f"{attempt_id}__failed.lock"),
        {"attempt_id": attempt_id, "pid": 123, "created_utc": "now"},
    )
    with pytest.raises(subject.AuditFailure, match="duplicate launch-history status"):
        audit_fixture(root, runner, contract)


def test_symlink_in_request_inventory_fails_without_following(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, runner, contract = build_complete_fixture(tmp_path)
    target = tmp_path / "external-target.json"
    write_json(target, {"forbidden": True})
    link = root / "raw" / "request_events" / "linked.json"
    try:
        link.symlink_to(target)
    except OSError:
        write_json(link, {"synthetic_symlink_placeholder": True})
        original_is_symlink = Path.is_symlink
        monkeypatch.setattr(
            Path,
            "is_symlink",
            lambda path: str(path) == str(link) or original_is_symlink(path),
        )
    with pytest.raises(subject.AuditFailure, match="symlink forbidden"):
        audit_fixture(root, runner, contract)


def test_canonical_parquet_symlink_fails_before_schema_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, runner, contract = build_complete_fixture(tmp_path)
    canonical = root / subject.OUTPUT_RELATIVE_PATHS["site"]
    target = tmp_path / "external-label-array.parquet"
    shutil.copy2(canonical, target)
    canonical.unlink()
    try:
        canonical.symlink_to(target)
    except OSError:
        shutil.copy2(target, canonical)
        original_is_symlink = Path.is_symlink
        monkeypatch.setattr(
            Path,
            "is_symlink",
            lambda path: str(path) == str(canonical) or original_is_symlink(path),
        )
    with pytest.raises(subject.AuditFailure, match="symlink forbidden"):
        audit_fixture(root, runner, contract)


def test_active_lock_symlink_conditionally_skips_without_target_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "active-symlink"
    active = root / "raw" / "RAW_LAUNCH_ACTIVE.lock"
    active.parent.mkdir(parents=True)
    target = tmp_path / "forbidden-active-target.json"
    write_json(target, {"secret": "must-not-read"})
    try:
        active.symlink_to(target)
    except OSError:
        write_json(active, {"synthetic_symlink_placeholder": True})
        original_is_symlink = Path.is_symlink
        monkeypatch.setattr(
            Path,
            "is_symlink",
            lambda path: str(path) == str(active) or original_is_symlink(path),
        )
    original_open = Path.open

    def guarded_open(path: Path, *args: object, **kwargs: object) -> object:
        if str(path) == str(target):
            raise AssertionError("active-lock symlink target was opened")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(subject.AuditNotReady) as exc_info:
        subject.audit(root, runner_path=tmp_path / "absent.py")
    assert exc_info.value.active_lock == {
        "path": "raw/RAW_LAUNCH_ACTIVE.lock",
        "symlink": True,
    }


def test_extra_label_column_fails_before_any_parquet_value_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, runner, contract = build_complete_fixture(tmp_path)
    site_path = root / subject.OUTPUT_RELATIVE_PATHS["site"]
    site = pd.read_parquet(site_path)
    site["label_2025"] = [1.0, 2.0]
    site.to_parquet(site_path, index=False)
    value_reads = 0

    def forbidden_value_read(*_args: object, **_kwargs: object) -> object:
        nonlocal value_reads
        value_reads += 1
        raise AssertionError("Parquet values materialized before schema rejection")

    monkeypatch.setattr(pd, "read_parquet", forbidden_value_read)
    with pytest.raises(subject.AuditFailure, match="zero-target schema boundary"):
        audit_fixture(root, runner, contract)
    assert value_reads == 0


def test_unknown_identity_cannot_dereference_absolute_label_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, runner, contract = build_complete_fixture(tmp_path)
    label_path = tmp_path / "forbidden-2025-labels.bin"
    label_path.write_bytes(b"labels-must-never-be-opened")
    authorization_path = root / "prereg" / "raw_launch_authorization_v1.json"
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    authorization["malicious_extra_identity"] = artifact(label_path)
    write_json(authorization_path, authorization)
    original_open = Path.open
    original_stat = Path.stat
    touches = 0

    def guarded_open(path: Path, *args: object, **kwargs: object) -> object:
        nonlocal touches
        if str(path) == str(label_path):
            touches += 1
            raise AssertionError("untrusted identity target opened")
        return original_open(path, *args, **kwargs)

    def guarded_stat(path: Path, *args: object, **kwargs: object) -> object:
        nonlocal touches
        if str(path) == str(label_path):
            touches += 1
            raise AssertionError("untrusted identity target statted")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "stat", guarded_stat)
    with pytest.raises(subject.AuditFailure, match="unrecognized fields"):
        audit_fixture(root, runner, contract)
    assert touches == 0


def test_wrong_known_identity_path_is_rejected_before_stat_or_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = tmp_path / "expected-source.py"
    expected.write_text("EXPECTED = True\n", encoding="utf-8")
    sentinel = tmp_path / "forbidden-label-2025.bin"
    sentinel.write_bytes(b"target values must remain unread")
    record = artifact(expected)
    record["path"] = str(sentinel.absolute())
    touches = 0
    original_stat = Path.stat
    original_open = Path.open

    def guarded_stat(path: Path, *args: object, **kwargs: object) -> object:
        nonlocal touches
        if path.absolute() == sentinel.absolute():
            touches += 1
            raise AssertionError("wrong identity path was stat'ed")
        return original_stat(path, *args, **kwargs)

    def guarded_open(path: Path, *args: object, **kwargs: object) -> object:
        nonlocal touches
        if path.absolute() == sentinel.absolute():
            touches += 1
            raise AssertionError("wrong identity path was opened")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", guarded_stat)
    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(subject.AuditFailure, match="path identity mismatch"):
        subject.verify_identity(record, expected, root=None, label="known role")
    assert touches == 0


def test_production_contract_forbids_decoder_override(tmp_path: Path) -> None:
    root = tmp_path / "irrelevant"
    root.mkdir()
    with pytest.raises(subject.AuditFailure, match="forbids decoder override"):
        subject.audit(
            root,
            contract=subject.PRODUCTION_CONTRACT,
            _test_message_decoder=synthetic_decoder,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("network_requests_this_invocation", 8, "cache/network range reconstruction"),
        ("network_bytes_this_invocation", 80, "payload-byte reconstruction"),
        ("actual_http_attempts_before_invocation", 0, "START-only attempt"),
    ),
)
def test_rebound_access_ledger_accounting_tamper_fails(
    tmp_path: Path, field: str, value: int, message: str
) -> None:
    root, runner, contract = build_complete_fixture(tmp_path)
    access_path = root / subject.OUTPUT_RELATIVE_PATHS["access"]
    access = json.loads(access_path.read_text(encoding="utf-8"))
    access[field] = value
    if field == "actual_http_attempts_before_invocation":
        access["actual_http_attempts_this_invocation"] = 11
    write_json(access_path, access)
    rebind_canonical_transaction(root)
    with pytest.raises(subject.AuditFailure, match=message):
        audit_fixture(root, runner, contract)


def test_rebound_final_progress_accounting_tamper_fails(tmp_path: Path) -> None:
    root, runner, contract = build_complete_fixture(tmp_path)
    progress = next((root / "raw" / "progress").glob("raw_ranges__*.json"))
    payload = json.loads(progress.read_text(encoding="utf-8"))
    payload["network_requests_this_invocation_so_far"] = 8
    write_json(progress, payload)
    rebind_canonical_transaction(root)
    with pytest.raises(subject.AuditFailure, match="progress/access/event accounting"):
        audit_fixture(root, runner, contract)


@pytest.mark.parametrize("corruption", ("json", "parquet", "wrong_type"))
def test_cli_normalizes_malformed_artifacts_to_structured_fail(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    corruption: str,
) -> None:
    root, _runner, _contract = build_complete_fixture(tmp_path)
    authorization = root / "prereg" / "raw_launch_authorization_v1.json"
    if corruption == "json":
        authorization.write_text("{", encoding="utf-8")
    elif corruption == "parquet":
        (root / "census" / "field_range_census.parquet").write_bytes(
            b"not-a-parquet-file"
        )
    else:
        write_json(authorization, ["wrong", "root", "type"])
    assert subject.main(["--root", str(root)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "FAIL_PRODUCER_POSTRUN_AUDIT"
    assert report["audit_network_requests"] == 0
    assert report["audit_files_written"] == 0


@pytest.mark.parametrize("error", (TypeError("decode"), OverflowError("decode")))
def test_cli_normalizes_decoder_exceptions_to_structured_fail(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    def failing_audit(*_args: object, **_kwargs: object) -> object:
        raise error

    monkeypatch.setattr(subject, "audit", failing_audit)
    assert subject.main(["--root", str(tmp_path)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "FAIL_PRODUCER_POSTRUN_AUDIT"
    assert report["reason"] == "decode"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("raw", "raw bytes differ from sidecar"),
        ("complete_lock", "matching producer complete lock is absent"),
        ("transaction", "canonical output differs from transaction plan"),
        ("label_fact", "violates zero-access fact labels_read"),
        ("physical", "PBL maximum"),
        ("event_global", "duplicate global raw HTTP-attempt number"),
    ],
)
def test_tampering_fails_closed(tmp_path: Path, mutation: str, message: str) -> None:
    root, runner, contract = build_complete_fixture(tmp_path)
    if mutation == "raw":
        path = sorted((root / "raw" / "ranges").rglob("*.grib2"))[0]
        payload = bytearray(path.read_bytes())
        payload[4] ^= 1
        path.write_bytes(payload)
    elif mutation == "complete_lock":
        next((root / "raw" / "launch_history").glob("*__complete.lock")).unlink()
    elif mutation == "transaction":
        path = root / subject.OUTPUT_RELATIVE_PATHS["raw_csv"]
        path.write_text(path.read_text(encoding="utf-8") + "tamper\n", encoding="utf-8")
    elif mutation == "label_fact":
        path = root / subject.OUTPUT_RELATIVE_PATHS["access"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["labels_read"] = True
        write_json(path, payload)
        # Rebind every enclosing identity so the semantic zero-fact check, not
        # an outer hash, is the first failing boundary.
        lock_path = root / subject.OUTPUT_RELATIVE_PATHS["lock"]
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        lock["access_ledger"] = artifact(path, root)
        write_json(lock_path, lock)
        manifest_path = root / subject.OUTPUT_RELATIVE_PATHS["manifest"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["decoded_matrix_lock"] = artifact(lock_path, root)
        write_json(manifest_path, manifest)
        plan_path = next((root / "raw" / "output_transactions").glob("*__plan.json"))
        commit_path = next((root / "raw" / "output_transactions").glob("*__committed.json"))
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        for item in plan["items"]:
            changed = {
                "access": path,
                "lock": lock_path,
                "manifest": manifest_path,
            }.get(item["name"])
            if changed is not None:
                item["size_bytes"] = changed.stat().st_size
                item["sha256"] = subject.sha256_file(changed)
        write_json(plan_path, plan)
        commit = json.loads(commit_path.read_text(encoding="utf-8"))
        commit["plan"] = artifact(plan_path, root)
        commit["outputs"]["access"] = artifact(path, root)
        commit["outputs"]["lock"] = artifact(lock_path, root)
        commit["outputs"]["manifest"] = artifact(manifest_path, root)
        write_json(commit_path, commit)
    elif mutation == "physical":
        path = root / subject.OUTPUT_RELATIVE_PATHS["audit"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["families"]["PBL_HEIGHT"]["maximum"] = 999.0
        write_json(path, payload)
        plan_path = next((root / "raw" / "output_transactions").glob("*__plan.json"))
        commit_path = next((root / "raw" / "output_transactions").glob("*__committed.json"))
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        for item in plan["items"]:
            if item["name"] == "audit":
                item["size_bytes"] = path.stat().st_size
                item["sha256"] = subject.sha256_file(path)
        write_json(plan_path, plan)
        commit = json.loads(commit_path.read_text(encoding="utf-8"))
        commit["plan"] = artifact(plan_path, root)
        commit["outputs"]["audit"] = artifact(path, root)
        write_json(commit_path, commit)
    elif mutation == "event_global":
        paths = [
            path
            for path in sorted((root / "raw" / "request_events").glob("*_start.json"))
            if path.with_name(path.name.replace("_start.json", "_complete.json")).is_file()
        ][:2]
        first = json.loads(paths[0].read_text(encoding="utf-8"))
        second = json.loads(paths[1].read_text(encoding="utf-8"))
        second["global_raw_attempt_number"] = first["global_raw_attempt_number"]
        write_json(paths[1], second)
        outcome = paths[1].with_name(paths[1].name.replace("_start.json", "_complete.json"))
        outcome_payload = json.loads(outcome.read_text(encoding="utf-8"))
        outcome_payload["global_raw_attempt_number"] = first["global_raw_attempt_number"]
        write_json(outcome, outcome_payload)
        # Event inventory is checked later; make the producer manifest's recorded
        # inventory accurately bind the tampered events.
        manifest_path = root / subject.OUTPUT_RELATIVE_PATHS["manifest"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["request_event_inventory"] = [
            artifact(path, root)
            for path in sorted((root / "raw" / "request_events").glob("*.json"))
        ]
        write_json(manifest_path, manifest)
        plan_path = next((root / "raw" / "output_transactions").glob("*__plan.json"))
        commit_path = next((root / "raw" / "output_transactions").glob("*__committed.json"))
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        for item in plan["items"]:
            if item["name"] == "manifest":
                item["size_bytes"] = manifest_path.stat().st_size
                item["sha256"] = subject.sha256_file(manifest_path)
        write_json(plan_path, plan)
        commit = json.loads(commit_path.read_text(encoding="utf-8"))
        commit["plan"] = artifact(plan_path, root)
        commit["outputs"]["manifest"] = artifact(manifest_path, root)
        write_json(commit_path, commit)
    with pytest.raises(subject.AuditFailure, match=message):
        audit_fixture(root, runner, contract)


def test_auditor_source_has_no_network_or_filesystem_mutation_primitives() -> None:
    source = Path(subject.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "import urllib",
        "import requests",
        "import socket",
        "urlopen(",
        ".write_text(",
        ".write_bytes(",
        ".mkdir(",
        "os.replace(",
        ".unlink(",
    ):
        assert forbidden not in source
    assert "ThreadPoolExecutor" not in source
    assert "ProcessPoolExecutor" in source
    assert 'get_context("spawn")' in source

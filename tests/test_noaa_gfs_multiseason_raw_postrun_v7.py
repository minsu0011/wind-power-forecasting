from __future__ import annotations

import ast
import concurrent.futures
from dataclasses import replace
import json
import hashlib
import multiprocessing
import os
import shutil
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pandas as pd
import pytest

from scripts import audit_noaa_gfs_multiseason_raw_postrun_v7 as subject
from scripts import audit_noaa_gfs_multiseason_raw_postrun_v6 as frozen_v6_subject
from scripts import audit_noaa_gfs_multiseason_raw_postrun_v5 as frozen_v5_subject
from scripts import seal_noaa_gfs_multiseason_postrun_audit_v7 as v7_sealer
from scripts import seal_noaa_gfs_multiseason_postrun_audit_v6 as frozen_v6_sealer


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


def mutate_independent_prelaunch_record_and_rebind_go(
    root: Path, mutation: object
) -> None:
    authorization_path = root / "prereg" / "raw_launch_authorization_v1.json"
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    record = authorization["independent_prelaunch_audit"]
    mutation(record)
    write_json(authorization_path, authorization)
    go_path = root / "independent_redteam" / "TRACK_A_RAW_LAUNCH_GO.json"
    go = json.loads(go_path.read_text(encoding="utf-8"))
    go["bound_authorization_sha256"] = subject.sha256_file(authorization_path)
    go["bound_authorization_size_bytes"] = authorization_path.stat().st_size
    write_json(go_path, go)


def copy_v4_predata_controls(root: Path) -> None:
    relatives = [
        subject.V2_FALSE_REJECT_INCIDENT_RELATIVE,
        subject.V4_PATH_MISPUBLISH_INCIDENT_RELATIVE,
        subject.FROZEN_V3_AUTH_RELATIVE,
        subject.FROZEN_V3_REVIEW_RELATIVE,
        subject.FROZEN_V3_MISPLACED_GO_RELATIVE,
        *subject.V4_V2_CONTROL_RELATIVES.values(),
        subject.V4_ORIGINAL_V1_PROVENANCE["original_raw_authorization_v1"]["path"],
        subject.V4_ORIGINAL_V1_PROVENANCE[
            "original_independent_prelaunch_audit_v1"
        ]["path"],
    ]
    for relative in relatives:
        source = subject.ROOT_DEFAULT / str(relative)
        destination = root / str(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def synthetic_v4_test_evidence(
    created_utc: str,
    *,
    v4_auditor: dict[str, object],
    v4_auditor_test: dict[str, object],
    v4_sealer: dict[str, object],
    v4_sealer_test: dict[str, object],
) -> dict[str, object]:
    bound = {
        "v2_auditor": subject.V4_SUPERSEDED_V2_AUDITOR_IDENTITY,
        "v2_auditor_test": subject.V4_SUPERSEDED_V2_AUDITOR_TEST_IDENTITY,
        **subject.FROZEN_V3_SOURCE_IDENTITIES,
        "v4_auditor": v4_auditor,
        "v4_auditor_test": v4_auditor_test,
        "v4_sealer": v4_sealer,
        "v4_sealer_test": v4_sealer_test,
    }
    runs: dict[str, object] = {}
    for role, relative in {
        "v4_auditor_tests": "tests/test_noaa_gfs_multiseason_raw_postrun_v4.py",
        "v4_sealer_tests": "tests/test_seal_noaa_gfs_multiseason_postrun_audit_v4.py",
        "frozen_v3_auditor_tests": "tests/test_noaa_gfs_multiseason_raw_postrun_v3.py",
        "frozen_v3_sealer_tests": "tests/test_seal_noaa_gfs_multiseason_postrun_audit_v3.py",
        "frozen_v2_auditor_tests": "tests/test_noaa_gfs_multiseason_raw_postrun_v2.py",
    }.items():
        runs[role] = {
            "command": subject._v4_test_command(relative),
            "exit_code": 0,
            "summary": "1 passed",
            "stdout_sha256": "0" * 64,
            "stderr_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "network_guard_installed": True,
            "cacheprovider_disabled": True,
        }
    return {
        "schema_version": 1,
        "artifact_type": "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_V4_TEST_EVIDENCE",
        "status": "PASS_FROZEN_V4_AUDITOR_AND_SEALER_TESTS",
        "created_utc": created_utc,
        "bound_identities": bound,
        "source_compile": {
            "method": "compile_exact_source_no_pyc",
            "result": "PASS",
            "files": [v4_auditor, v4_auditor_test, v4_sealer, v4_sealer_test],
        },
        "pytest_isolation": "ONE_TEST_FILE_PER_CLEAN_NETWORK_DENIED_SUBPROCESS",
        "test_runs": runs,
        "required_test_names": list(subject.V4_REQUIRED_TEST_NAMES),
        "required_test_names_present": True,
        "all_exit_codes_zero": True,
        "production_shape_regression": {
            "four_key_identity_passed": True,
            "transaction_reached": True,
            "raw_reached": True,
            "decoded_reached": True,
            "offline_replay_reached": True,
            "synthetic_fixture": True,
        },
        "real_seven_spawn_regression": {
            "start_method": "spawn",
            "entrypoint_module": "scripts.audit_noaa_gfs_multiseason_raw_postrun_v4",
            "worker_callable_module": "scripts.audit_noaa_gfs_multiseason_raw_postrun_v4",
            "max_workers": 7,
            "distinct_worker_pids": 7,
            "first_wave_synchronized": True,
            "network_requests": 0,
            "audit_files_written": 0,
            "labels_read": False,
            "arrays_2024_read": False,
            "arrays_2025_read": False,
        },
        "stdout_only_regression": True,
        "network_guard": {
            "parent_test_process_guarded": True,
            "sys_audit_hook_in_parent": True,
            "parent_socket_api_denied": True,
            "spawned_worker_network_route_static_absent": True,
            "spawned_worker_runtime_guard_installed": False,
            "external_packet_capture": False,
        },
        "network_requests": 0,
        "audit_files_written": 0,
        "labels_read": False,
        "arrays_2024_read": False,
        "arrays_2025_read": False,
        "models_fit": 0,
        "submission_csv_created": False,
    }


def build_valid_v4_authorization_with_unavailable_review(
    root: Path,
    *,
    sealer: Path,
    sealer_test: Path,
) -> tuple[Path, Path]:
    copy_v4_predata_controls(root)
    created_utc = "2026-08-10T22:00:00.000000Z"
    audit_attempt_id = "postrun_audit_v4__20260810T220000000000Z"
    progress_inventory = [
        artifact(path, subject.ROOT_DEFAULT)
        for path in sorted(
            (subject.ROOT_DEFAULT / "decoded" / "recovery_progress").iterdir(),
            key=lambda path: path.name,
        )
    ]
    progress = {
        "file_count": 104,
        "inventory": progress_inventory,
        "inventory_canonical_sha256": subject.V4_PROGRESS_INVENTORY_CANONICAL_SHA256,
        "final_checkpoint": subject.V4_FINAL_PROGRESS_IDENTITY,
    }
    v4_auditor = artifact(Path(subject.__file__).resolve())
    v4_auditor_test = artifact(subject.AUDITOR_TEST)
    v4_sealer = artifact(sealer)
    v4_sealer_test = artifact(sealer_test)
    evidence = synthetic_v4_test_evidence(
        created_utc,
        v4_auditor=v4_auditor,
        v4_auditor_test=v4_auditor_test,
        v4_sealer=v4_sealer,
        v4_sealer_test=v4_sealer_test,
    )
    authorization_path = root / subject.V4_AUTH_RELATIVE
    go_path = root / subject.V4_GO_RELATIVE
    frozen_v3 = subject._v4_validate_frozen_v3_chain(root)
    authorization = {
        "schema_version": 4,
        "artifact_type": "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDIT_AUTHORIZATION_V4",
        "status": "AUTHORIZED_PENDING_INDEPENDENT_REVIEW_AND_GO_V4",
        "created_utc": created_utc,
        "audit_attempt_id": audit_attempt_id,
        "incident": artifact(root / subject.V4_PATH_MISPUBLISH_INCIDENT_RELATIVE, root),
        "v2_false_reject_incident": artifact(
            root / subject.V2_FALSE_REJECT_INCIDENT_RELATIVE, root
        ),
        "v3_path_mispublish_state": frozen_v3["state"],
        "superseded_v2_auditor": subject.V4_SUPERSEDED_V2_AUDITOR_IDENTITY,
        "superseded_v2_auditor_test": subject.V4_SUPERSEDED_V2_AUDITOR_TEST_IDENTITY,
        **subject.FROZEN_V3_SOURCE_IDENTITIES,
        "v4_auditor": v4_auditor,
        "v4_auditor_test": v4_auditor_test,
        "v4_sealer": v4_sealer,
        "v4_sealer_test": v4_sealer_test,
        "v2_recovery_controls": {
            role: artifact(root / relative, root)
            for role, relative in subject.V4_V2_CONTROL_RELATIVES.items()
        },
        "v2_recovery_support": {
            role: artifact(path)
            for role, path in subject.V4_V2_SUPPORT_PATHS.items()
        },
        "recovery_attempt_id": subject.V4_RECOVERY_ATTEMPT_ID,
        "recovery_transaction": subject.V4_TRANSACTION_IDENTITIES,
        "recovery_history": {
            "file_count": 3,
            "completion_locks": subject.V4_COMPLETION_LOCK_IDENTITIES,
            "postcommit_input_audit": subject.V4_POSTCOMMIT_INPUT_AUDIT_IDENTITY,
        },
        "recovery_progress": progress,
        "canonical_outputs": subject.V4_CANONICAL_OUTPUT_IDENTITIES,
        "original_v1_provenance": subject.V4_ORIGINAL_V1_PROVENANCE,
        "preaudit_zero_mutation_snapshot": {
            "incident_postfailure_state_canonical_sha256": (
                subject.V4_INCIDENT_POSTFAILURE_STATE_CANONICAL_SHA256
            ),
            "recovered_afterstate_canonical_sha256": (
                subject.V4_RECOVERED_AFTERSTATE_CANONICAL_SHA256
            ),
            "active_locks": {
                "raw": {"path": "raw/RAW_LAUNCH_ACTIVE.lock", "present": False},
                "decoded": {"path": "decoded/DECODE_RECOVERY_ACTIVE.lock", "present": False},
            },
            "v4_control_temporary_files": {
                "checked_destination_relative_paths": [
                    subject.FROZEN_V3_AUTH_RELATIVE,
                    subject.FROZEN_V3_REVIEW_RELATIVE,
                    subject.FROZEN_V3_CANONICAL_GO_RELATIVE,
                    subject.FROZEN_V3_MISPLACED_GO_RELATIVE,
                    subject.V4_AUTH_RELATIVE,
                    subject.V4_REVIEW_RELATIVE,
                    subject.V4_GO_RELATIVE,
                    subject.V2_FALSE_REJECT_INCIDENT_RELATIVE,
                    subject.V4_PATH_MISPUBLISH_INCIDENT_RELATIVE,
                ],
                "matching_temporary_files": 0,
            },
            "postrun_audit_output_files_present": 0,
            "network_requests": 0,
            "audit_files_written": 0,
            "labels_read": False,
            "arrays_2024_read": False,
            "arrays_2025_read": False,
            "models_fit": 0,
            "submission_csv_created": False,
        },
        "runtime_identity_sha256": subject.V4_RUNTIME_IDENTITY_SHA256,
        "test_evidence": evidence,
        "required_command": subject._v4_required_command(root, authorization_path, go_path),
        "max_spawn_processes": 7,
        "network_requests_allowed": 0,
        "audit_files_written_allowed": 0,
        "labels_read_allowed": False,
        "arrays_2024_read_allowed": False,
        "arrays_2025_read_allowed": False,
        "models_fit_allowed": 0,
        "submission_csv_allowed": False,
        "full_offline_redecode_required": True,
        "stdout_only": True,
        "independent_review_required": True,
        "independent_go_required": True,
    }
    write_json(authorization_path, authorization)
    afterstate = subject._v4_recovered_afterstate_from_authorization(authorization)
    missing_review = root / subject.V4_REVIEW_RELATIVE
    go = {
        "schema_version": 4,
        "artifact_type": "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_GO_V4",
        "status": "GO_V4_READ_ONLY_NETWORK_ZERO_FULL_OFFLINE_REPLAY",
        "created_utc": "2026-08-10T22:00:02.000000Z",
        "audit_attempt_id": audit_attempt_id,
        "authorization": artifact(authorization_path, root),
        "independent_review": {
            "path": missing_review.relative_to(root).as_posix(),
            "size_bytes": 1,
            "sha256": "0" * 64,
        },
        "incident": authorization["incident"],
        "v2_false_reject_incident": authorization["v2_false_reject_incident"],
        "v3_path_mispublish_state": authorization["v3_path_mispublish_state"],
        "superseded_v2_auditor": authorization["superseded_v2_auditor"],
        "superseded_v2_auditor_test": authorization["superseded_v2_auditor_test"],
        **{
            role: authorization[role]
            for role in subject.FROZEN_V3_SOURCE_IDENTITIES
        },
        "v4_auditor": v4_auditor,
        "v4_auditor_test": v4_auditor_test,
        "v4_sealer": v4_sealer,
        "v4_sealer_test": v4_sealer_test,
        "recovery_attempt_id": subject.V4_RECOVERY_ATTEMPT_ID,
        "recovered_afterstate": afterstate,
        "required_command": authorization["required_command"],
        "max_spawn_processes": 7,
        "network_requests_allowed": 0,
        "audit_files_written_allowed": 0,
        "labels_read_allowed": False,
        "arrays_2024_read_allowed": False,
        "arrays_2025_read_allowed": False,
        "models_fit_allowed": 0,
        "submission_csv_allowed": False,
        "full_offline_redecode_required": True,
        "stdout_only": True,
        "recovery_rerun_authorized": False,
    }
    write_json(go_path, go)
    return authorization_path, go_path


def complete_v4_review_and_go(
    root: Path, authorization_path: Path, go_path: Path
) -> Path:
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    afterstate = subject._v4_recovered_afterstate_from_authorization(authorization)
    review_path = root / subject.V4_REVIEW_RELATIVE
    recheck = {
        "authorization_test_evidence_canonical_sha256": (
            subject.canonical_payload_sha256(authorization["test_evidence"])
        ),
        "bound_identities_rehashed": True,
        "source_compile_exact": True,
        "required_test_names_exact": True,
        "all_exit_codes_zero": True,
        "production_shape_regression_exact": True,
        "real_seven_spawn_regression_exact": True,
        "stdout_only_zero_write_exact": True,
        "network_zero_target_free_exact": True,
    }
    review = {
        "schema_version": 4,
        "artifact_type": "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_REVIEW_V4",
        "status": "PASS_PENDING_INDEPENDENT_GO_V4",
        "verdict": "GO_RECOMMENDED",
        "created_utc": "2026-08-10T22:00:01.000000Z",
        "audit_attempt_id": authorization["audit_attempt_id"],
        "authorization": artifact(authorization_path, root),
        "incident": authorization["incident"],
        "v2_false_reject_incident": authorization["v2_false_reject_incident"],
        "v3_path_mispublish_state": authorization["v3_path_mispublish_state"],
        "superseded_v2_auditor": authorization["superseded_v2_auditor"],
        "superseded_v2_auditor_test": authorization["superseded_v2_auditor_test"],
        **{
            role: authorization[role]
            for role in subject.FROZEN_V3_SOURCE_IDENTITIES
        },
        "v4_auditor": authorization["v4_auditor"],
        "v4_auditor_test": authorization["v4_auditor_test"],
        "v4_sealer": authorization["v4_sealer"],
        "v4_sealer_test": authorization["v4_sealer_test"],
        "v2_recovery_controls": authorization["v2_recovery_controls"],
        "recovered_afterstate": afterstate,
        "test_evidence_recheck": recheck,
        "independent_checks": {key: True for key in subject.V4_REVIEW_CHECKS},
        "network_requests_allowed": 0,
        "audit_files_written_allowed": 0,
        "max_spawn_processes": 7,
        "labels_read_allowed": False,
        "arrays_2024_read_allowed": False,
        "arrays_2025_read_allowed": False,
        "models_fit_allowed": 0,
        "submission_csv_allowed": False,
        "full_offline_redecode_required": True,
        "stdout_only": True,
        "auditor_execution_started": False,
        "independent_go_required": True,
    }
    write_json(review_path, review)
    go = json.loads(go_path.read_text(encoding="utf-8"))
    go["independent_review"] = artifact(review_path, root)
    write_json(go_path, go)
    return review_path


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
            "independent_prelaunch_audit": {
                **artifact(independent_path, root),
                "status": "PASS_TO_CREATE_FINAL_AUTH_ONLY",
            },
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
    recovery_attempt = "decode_recovery_v2__synthetic0001"
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

    flawed_launch_path = root / subject.RECOVERY_FLAWED_LAUNCH_INCIDENT_RELATIVE
    required_v2 = {
        "new_attempt_id_required": True,
        "new_postrun_auditor_binding_required": True,
        "new_raw_lock_preflight_authorization_review_and_go_required": True,
        "new_runner_and_dedicated_run_prefix_regression_required": True,
        "new_sealer_and_process_evidence_required": True,
        "old_v1_chain_must_not_be_overwritten": True,
        "regression_must_execute_run_through_postpilot_check": True,
        "regression_must_stop_before_recovery_claim_or_target_write": True,
        "retry_performed_by_this_incident_seal": False,
        "this_incident_must_be_transitively_bound": True,
    }
    flawed_launch = {
        "artifact_type": (
            "TRACK_A_DECODE_RECOVERY_LAUNCH_BOOTSTRAP_IMPORT_CONTRACT_NAMEERROR_INCIDENT"
        ),
        "captured_result": {"error_type": "NameError", "network_requests": 0},
        "created_utc": "2026-08-10T19:50:00Z",
        "failed_recovery_attempt_id": "decode_recovery_v1__20260810T194716214040Z",
        "failed_v1_chain": subject.RECOVERY_FLAWED_V1_CHAIN_IDENTITIES,
        "first_monitor_observation": {"status": "FAILED"},
        "invocation": {"mode": "CANONICAL_MODULE"},
        "network_evidence": {"network_requests": 0},
        "observed_time_bounds": {"start_utc": "2026-08-10T19:47:16Z"},
        "postfailure_state_closure": {
            "canonical_outputs_present": 0,
            "new_recovery_transaction_entries_present": 0,
            "decoded_directory_present": False,
            "raw_active_lock_present": False,
            "decoded_active_lock_present": False,
            "recovery_target_writes": 0,
        },
        "required_v2_supersession": required_v2,
        "root_cause": {
            "classification": "RUN_PATH_LOCAL_BINDING_OMISSION",
            "frozen_runner_sha256": subject.RECOVERY_V1_CHAIN_IDENTITIES[
                "recovery_runner"
            ]["sha256"],
            "recovery_claim_acquired": False,
        },
        "schema_version": 1,
        "status": "CLOSED_FAIL_SAFE_REQUIRES_APPEND_ONLY_V2_SUPERSESSION",
        "target_free_scope": {"labels_read": False},
        "v1_disposition": {
            "v1_retry_allowed": False,
            "v1_runner_remains_byte_frozen": True,
        },
    }
    write_json(flawed_launch_path, flawed_launch)
    flawed_launch_identity = artifact(flawed_launch_path, root)

    preserved = {
        "captured_result": flawed_launch["captured_result"],
        "postfailure_state_closure": flawed_launch["postfailure_state_closure"],
    }
    correction_path = root / subject.RECOVERY_LAUNCH_IDENTITY_CORRECTION_RELATIVE
    correction = {
        "artifact_type": (
            "TRACK_A_DECODE_RECOVERY_LAUNCH_NAMEERROR_INCIDENT_IDENTITY_CORRECTION"
        ),
        "authoritative_failed_v1_chain": subject.RECOVERY_V1_CHAIN_IDENTITIES,
        "authoritative_v1_support": subject.RECOVERY_V1_SUPPORT_IDENTITIES,
        "authority_basis": {
            "v1_authorization_review_and_go_bind_runner_sha256": (
                subject.RECOVERY_V1_CHAIN_IDENTITIES["recovery_runner"]["sha256"]
            ),
            "restored_v1_runner_rehash_matches_authorized_identity": True,
        },
        "correction_reason": "synthetic relative-vs-absolute authority fixture",
        "created_utc": "2026-08-10T20:00:00Z",
        "postcorrection_rehash": {
            "all_exact": True,
            "v1_chain": subject.RECOVERY_V1_CHAIN_IDENTITIES,
            "v1_support": subject.RECOVERY_V1_SUPPORT_IDENTITIES,
        },
        "preserved_failure_evidence": preserved,
        "preserved_failure_evidence_canonical_sha256": {
            key: subject.canonical_payload_sha256(value)
            for key, value in preserved.items()
        },
        "prior_incidents": [flawed_launch_identity],
        "rejected_transient_identity": {
            "identity": subject.RECOVERY_REJECTED_TRANSIENT_V1_RUNNER_IDENTITY,
            "disposition": (
                "REJECTED_UNLAUNCHED_UNAUTHORIZED_MID_EDIT_OBSERVATION"
            ),
            "historical_authority": False,
            "executed_in_failed_launch": False,
            "authorized_by_v1_controls": False,
        },
        "schema_version": 1,
        "status": "PASS_APPEND_ONLY_CORRECTION_SUPERSEDES_FLAWED_IDENTITY_AUTHORITY",
        "superseded_flawed_incident": flawed_launch_identity,
        "supersession_policy": {
            "correction_is_authoritative_for_failed_v1_chain": True,
            "flawed_incident_authority_superseded": True,
            "flawed_incident_preserved_append_only": True,
            "rejected_transient_identity_must_never_be_selected": True,
            "v1_runner_controls_review_and_go_remain_immutable": True,
            "v2_chain_must_bind_both_flawed_incident_and_this_correction": True,
            "v2_chain_must_use_only_authoritative_failed_v1_chain": True,
            "v2_execution_or_retry_authorized_by_this_document": False,
        },
        "target_free_scope": {"labels_read": False},
        "unchanged_fail_safe_state": {
            "canonical_outputs_present": 0,
            "decoded_directory_present": False,
            "new_recovery_transaction_entries_present": 0,
            "recovery_claim_acquired": False,
            "recovery_target_writes": 0,
            "runner_guard_and_failure_report_network_requests": 0,
        },
    }
    write_json(correction_path, correction)
    correction_identity = artifact(correction_path, root)
    superseded_v1_support = subject._authoritative_v1_support_external()
    failed_v1_zero_state = {
        "failed_v1_attempt_id": "decode_recovery_v1__20260810T194716214040Z",
        "recovery_claim_acquired": False,
        "decoded_tree_absent": True,
        "canonical_outputs_absent": True,
        "v1_transaction_entries": [],
        "v1_progress_entries": [],
        "v1_history_entries": [],
    }

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
            "schema_version": 2,
            "artifact_type": "TRACK_A_DECODE_RECOVERY_RAW_CACHE_LOCK_V2",
            "status": "PASS_V2_IMMUTABLE_RAW_CACHE_AND_V1_ZERO_STATE",
            "created_utc": "2026-08-10T17:10:00Z",
            "incident": incident_identity,
            "flawed_launch_incident": flawed_launch_identity,
            "launch_identity_correction": correction_identity,
            "superseded_v1_chain": subject.RECOVERY_V1_CHAIN_IDENTITIES,
            "superseded_v1_support": superseded_v1_support,
            "inventories": inventories,
            "prerecovery_topology": prerecovery_topology,
            "failed_v1_recovery_zero_state": failed_v1_zero_state,
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
        "postrun_auditor": artifact(subject.SUPERSEDED_V2_AUDITOR),
        "postrun_auditor_test": artifact(subject.SUPERSEDED_V2_AUDITOR_TEST),
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
        "schema_version": 2,
        "artifact_type": "TRACK_A_DECODE_RECOVERY_REAL_PROCESS_PREFLIGHT_V2",
        "status": "PASS_V2_REAL_PILOT_PRODUCTION_AND_RUN_PREFIX_EXACT_7_SPAWN",
        "created_utc": "2026-08-10T17:10:00Z",
        "incident": incident_identity,
        "flawed_launch_incident": flawed_launch_identity,
        "launch_identity_correction": correction_identity,
        "superseded_v1_chain": subject.RECOVERY_V1_CHAIN_IDENTITIES,
        "superseded_v1_support": superseded_v1_support,
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
        "run_prefix_regression": {
            "status": "PASS_V2_RUN_REACHED_POSTPILOT_ALIAS_CHECK_BEFORE_CLAIM",
            "run_function_called": True,
            "shared_preclaim_function_called": True,
            "postpilot_bootstrap_alias_check_reached": True,
            "recovery_claim_acquired": False,
            "target_writes": 0,
            "network_requests": 0,
            "dedicated_test_name": (
                "test_v2_run_prefix_reaches_postpilot_alias_check_before_claim_and_writes_nothing"
            ),
        },
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
        "schema_version": 2,
        "artifact_type": "NOAA_GFS_NETWORK_ZERO_DECODE_RECOVERY_AUTHORIZATION_V2",
        "status": "AUTHORIZED_PENDING_INDEPENDENT_GO_V2",
        "created_utc": "2026-08-10T17:10:00Z",
        "experiment_id": "noaa_gfs_dminus2_12z_multiseason_target_free_decode_recovery_v2",
        "recovery_attempt_id": recovery_attempt,
        "incident": incident_identity,
        "flawed_launch_incident": flawed_launch_identity,
        "launch_identity_correction": correction_identity,
        "superseded_v1_chain": subject.RECOVERY_V1_CHAIN_IDENTITIES,
        "superseded_v1_support": superseded_v1_support,
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
            "launch_nameerror_incident_exact",
            "v1_chain_immutable_and_not_selected",
            "v2_run_prefix_regression_reached_postpilot_before_claim",
        }
    }
    review_path = root / subject.RECOVERY_REVIEW_RELATIVE
    review = {
        "schema_version": 2,
        "artifact_type": "TRACK_A_DECODE_RECOVERY_INDEPENDENT_REVIEW_V2",
        "status": "PASS_V2_NETWORK_ZERO_PROCESS_ISOLATED_RECOVERY_AUTHORIZED",
        "created_utc": "2026-08-10T17:11:00Z",
        "authorization": artifact(authorization_path, root),
        "incident": incident_identity,
        "flawed_launch_incident": flawed_launch_identity,
        "launch_identity_correction": correction_identity,
        "superseded_v1_chain": subject.RECOVERY_V1_CHAIN_IDENTITIES,
        "superseded_v1_support": superseded_v1_support,
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
        "schema_version": 2,
        "artifact_type": "TRACK_A_DECODE_RECOVERY_INDEPENDENT_GO_V2",
        "status": "GO_V2_NETWORK_ZERO_PROCESS_ISOLATED_DECODE_RECOVERY",
        "created_utc": "2026-08-10T17:12:00Z",
        "authorization": artifact(authorization_path, root),
        "incident": incident_identity,
        "flawed_launch_incident": flawed_launch_identity,
        "launch_identity_correction": correction_identity,
        "superseded_v1_chain": subject.RECOVERY_V1_CHAIN_IDENTITIES,
        "superseded_v1_support": superseded_v1_support,
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
        "postrun_auditor": subject.SUPERSEDED_V2_AUDITOR,
        "postrun_auditor_test": subject.SUPERSEDED_V2_AUDITOR_TEST,
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
        "artifact_type": "DECODE_RECOVERY_ACTIVE_LOCK_V2",
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
                "artifact_type": "DECODE_RECOVERY_PROGRESS_V2",
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
        "decode_failure_incident": incident_identity,
        "flawed_launch_incident": flawed_launch_identity,
        "launch_identity_correction": correction_identity,
        "superseded_v1_chain": subject.RECOVERY_V1_CHAIN_IDENTITIES,
        "superseded_v1_support": superseded_v1_support,
        "recovery_authorization_v2": artifact(authorization_path, root),
        "independent_recovery_go_v2": artifact(go_path, root),
        "independent_recovery_review_v2": artifact(review_path, root),
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
        "raw_cache_lock_v2": artifact(raw_cache_lock_path, root),
        "real_spawn_preflight_v2": artifact(preflight_path, root),
        "pilot_runtime_report": pilot_report,
        "failed_v1_recovery_zero_state": failed_v1_zero_state,
        "fixed_inputs_initial": fixed_snapshot,
        "locked_prestate_topology": locked_topology,
        "locked_fixed_inputs": fixed_snapshot,
    }

    physical = json.loads(outputs["audit"].read_text(encoding="utf-8"))
    physical.update(
        {
            "artifact_type": "TRACK_A_PHYSICAL_AND_COVERAGE_AUDIT_RECOVERED_V2",
            "recovery_provenance": recovery_provenance,
            "decode_process_model": "WINDOWS_SPAWN_PROCESS_ISOLATION_SERIAL_ECCODES_PER_PROCESS",
            "decode_worker_pids": list(range(801, 808)),
            "decode_process_count": 7,
        }
    )
    write_json(outputs["audit"], physical)
    access = {
        "artifact_type": "TRACK_A_RAW_ACCESS_LEDGER_RECOVERED_NETWORK_ZERO_V2",
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
        "artifact_type": "TARGET_FREE_DECODED_MATRIX_LOCK_RECOVERED_NETWORK_ZERO_V2",
        "schema_version": 2,
        "recovery_provenance": recovery_provenance,
        "runtime_identity": runtime_report,
        "raw_manifest_parquet": artifact(outputs["raw_parquet"], root),
        "raw_manifest_csv": artifact(outputs["raw_csv"], root),
        "site_matrix": artifact(outputs["site"], root),
        "group_matrix": artifact(outputs["group"], root),
        "physical_and_coverage_audit": artifact(outputs["audit"], root),
        "access_ledger": artifact(outputs["access"], root),
        "physical_family_ready_for_independent_postrun": True,
        "independent_postrun_audit_required_before_target_free_estimator": True,
        "target_free_duplicate_estimator_may_start": False,
        "network_requests": 0,
        "labels_read": False,
    }
    write_json(outputs["lock"], lock)
    manifest = {
        "artifact_type": "NOAA_GFS_MULTISEASON_RAW_AND_DECODE_MANIFEST_RECOVERED_V2",
        "schema_version": 2,
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
            "artifact_type": "DECODE_RECOVERY_OUTPUT_TRANSACTION_PLAN_V2",
            "schema_version": 2,
            "attempt_id": recovery_attempt,
            "created_utc": "2026-08-10T17:15:00Z",
            "recovery_authorization": artifact(authorization_path, root),
            "independent_recovery_go": artifact(go_path, root),
            "decode_failure_incident": incident_identity,
            "flawed_launch_incident": flawed_launch_identity,
            "launch_identity_correction": correction_identity,
            "superseded_v1_chain": subject.RECOVERY_V1_CHAIN_IDENTITIES,
            "superseded_v1_support": superseded_v1_support,
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
            "artifact_type": "DECODE_RECOVERY_OUTPUT_TRANSACTION_COMMIT_V2",
            "schema_version": 2,
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
            "artifact_type": "DECODE_RECOVERY_POSTCOMMIT_INPUT_AUDIT_V2",
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
        subject.audit(
            root,
            runner_path=tmp_path / "absent.py",
            contract=replace(subject.PRODUCTION_CONTRACT, range_rows=1),
        )
    assert exc_info.value.active_lock["attempt_id"] == "active"
    assert snapshot(root) == before


def test_complete_synthetic_producer_passes_and_audit_is_read_only(tmp_path: Path) -> None:
    root, runner, contract = build_complete_fixture(tmp_path)
    before = snapshot(tmp_path)
    report = audit_fixture(root, runner, contract)
    assert report["status"] == "PASS_PRODUCER_POSTRUN_READONLY_AUDIT_V7"
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
    assert report["status"] == "PASS_PRODUCER_POSTRUN_READONLY_AUDIT_V7"
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
    assert temporary_gate["start"]["checked_destination_count"] == 10
    assert snapshot(tmp_path) == before


def test_v2_support_uses_absolute_auth_and_relative_correction_forms(
    tmp_path: Path,
) -> None:
    root, _runner, _contract = build_recovered_fixture(tmp_path)
    authorization = json.loads(
        (root / subject.RECOVERY_AUTH_RELATIVE).read_text(encoding="utf-8")
    )
    correction = json.loads(
        (root / subject.RECOVERY_LAUNCH_IDENTITY_CORRECTION_RELATIVE).read_text(
            encoding="utf-8"
        )
    )
    observed = subject._validate_v2_support_path_forms(
        authorization["superseded_v1_support"],
        correction["authoritative_v1_support"],
    )
    assert observed == subject._authoritative_v1_support_external()
    assert all(Path(record["path"]).is_absolute() for record in observed.values())
    assert all(
        not Path(record["path"]).is_absolute()
        for record in correction["authoritative_v1_support"].values()
    )


@pytest.mark.parametrize("swap", ["auth_relative", "correction_absolute"])
def test_v2_support_path_form_swap_fails_closed(swap: str) -> None:
    absolute = subject._authoritative_v1_support_external()
    relative = subject.RECOVERY_V1_SUPPORT_IDENTITIES
    auth_support = relative if swap == "auth_relative" else absolute
    correction_support = absolute if swap == "correction_absolute" else relative
    with pytest.raises(subject.AuditFailure, match="support must"):
        subject._validate_v2_support_path_forms(auth_support, correction_support)


@pytest.mark.parametrize("mutation", ["missing", "extra", "disposition"])
def test_v2_correction_rejected_wrapper_exact_schema_fails_closed(
    tmp_path: Path, mutation: str
) -> None:
    root, _runner, _contract = build_recovered_fixture(tmp_path)
    flawed_identity = artifact(
        root / subject.RECOVERY_FLAWED_LAUNCH_INCIDENT_RELATIVE, root
    )
    correction = json.loads(
        (root / subject.RECOVERY_LAUNCH_IDENTITY_CORRECTION_RELATIVE).read_text(
            encoding="utf-8"
        )
    )
    rejected = correction["rejected_transient_identity"]
    if mutation == "missing":
        rejected.pop("disposition")
    elif mutation == "extra":
        rejected["unexpected"] = False
    else:
        rejected["disposition"] = "SELECTED"
    with pytest.raises(subject.AuditFailure, match="correction semantic mismatch"):
        subject._validate_v2_launch_identity_correction(
            correction,
            expected_flawed_identity=flawed_identity,
        )


def test_recovery_required_test_names_exactly_match_frozen_sealer_ast() -> None:
    tree = ast.parse(subject.RECOVERY_SEALER.read_text(encoding="utf-8"))
    observed: dict[str, list[str]] | None = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "required_names"
            for target in node.targets
        ):
            continue
        literal = ast.literal_eval(node.value)
        observed = {key: list(value) for key, value in literal.items()}
        break
    assert observed == subject.RECOVERY_REQUIRED_TEST_NAMES
    assert [len(value) for value in observed.values()] == [8, 17, 13]


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
    recovery_attempt = "decode_recovery_v2__synthetic0001"
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
                "artifact_type": "DECODE_RECOVERY_PROGRESS_V2",
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


@pytest.mark.parametrize("mutation", ["missing", "extra", "renamed"])
def test_recovery_required_test_name_mapping_tamper_fails_closed(
    tmp_path: Path, mutation: str
) -> None:
    root, _runner, _contract = build_recovered_fixture(tmp_path)
    preflight = json.loads(
        (root / subject.RECOVERY_REAL_PREFLIGHT_RELATIVE).read_text(encoding="utf-8")
    )
    evidence = json.loads(json.dumps(preflight["test_evidence"]))
    relative = "tests/test_seal_noaa_gfs_multiseason_decode_recovery_v2.py"
    names = evidence["required_test_names"][relative]
    if mutation == "missing":
        names.pop()
    elif mutation == "extra":
        names.append("test_unsealed_extra_name")
    else:
        names[0] = "test_static_audit_writes_anything"
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


def test_v4_worker_is_importable_in_exactly_seven_spawn_processes(
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
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(subject.DECODE_MAX_WORKERS)
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=subject.DECODE_MAX_WORKERS,
        mp_context=context,
        initializer=subject.v4_spawn_probe_barrier_initializer,
        initargs=(barrier,),
    ) as pool:
        futures = [
            pool.submit(subject.decode_offline_grib_process_worker, info, sites)
            for _index in range(subject.DECODE_MAX_WORKERS)
        ]
        decoded = [future.result(timeout=180) for future in futures]
    assert len({int(report["decoder_process_id"]) for report in decoded}) == 7
    assert {
        report["decoder_runtime_identity_sha256"] for report in decoded
    } == {subject.canonical_payload_sha256(runtime_identity)}
    for report in decoded:
        assert report["metadata_verified"] is True
        assert report["single_message_verified"] is True
        assert report["feature"] == "HPBL_surface"
        assert report["site_values"] == pytest.approx(expected, abs=1e-12, rel=0.0)


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
        subject.audit(
            root,
            runner_path=tmp_path / "absent.py",
            contract=replace(subject.PRODUCTION_CONTRACT, range_rows=1),
        )
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
    assert report["status"] == "FAIL_PRODUCER_POSTRUN_AUDIT_V7"
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
    monkeypatch.setattr(
        subject,
        "_v7_require_canonical_runtime_entrypoint",
        lambda *_args, **_kwargs: None,
    )
    assert subject.main(
        [
            "--root", str(tmp_path),
            "--authorization", str(tmp_path / "authorization.json"),
            "--independent-go", str(tmp_path / "go.json"),
        ]
    ) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "FAIL_PRODUCER_POSTRUN_AUDIT_V7"
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
    tree = ast.parse(source)
    imported_roots = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in (
            node.names
            if isinstance(node, ast.Import)
            else [ast.alias(name=node.module or "")]
        )
    }
    assert not ({"socket", "urllib", "requests"} & imported_roots)
    for forbidden in (
        "import urllib",
        "import requests",
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


def test_real_immutable_four_key_independent_prelaunch_identity_passes(
) -> None:
    root = subject.ROOT_DEFAULT
    authorization = json.loads(
        (root / "prereg" / "raw_launch_authorization_v1.json").read_text(
            encoding="utf-8"
        )
    )
    record = authorization["independent_prelaunch_audit"]
    assert record == subject.V4_ORIGINAL_V1_PROVENANCE[
        "original_independent_prelaunch_audit_v1"
    ]
    prelaunch = root / str(record["path"])
    assert prelaunch.stat().st_size == 11_439
    assert subject.sha256_file(prelaunch) == (
        "4279935206cc245c0443d156bad114b96e6c7d7624dee670df9305a44b1009b5"
    )
    actual = subject.verify_identity(
        record,
        prelaunch,
        root=root,
        label="independent prelaunch audit",
        allowed_extra_fields=("status",),
    )
    assert set(actual) == {"path", "size_bytes", "sha256"}
    assert record["status"] == "PASS_TO_CREATE_FINAL_AUTH_ONLY"


def test_missing_independent_prelaunch_status_fails(tmp_path: Path) -> None:
    root, runner, contract = build_complete_fixture(tmp_path)
    mutate_independent_prelaunch_record_and_rebind_go(
        root, lambda record: record.pop("status")
    )
    with pytest.raises(subject.AuditFailure, match="schema mismatch"):
        audit_fixture(root, runner, contract)


def test_wrong_independent_prelaunch_status_fails(tmp_path: Path) -> None:
    root, runner, contract = build_complete_fixture(tmp_path)
    mutate_independent_prelaunch_record_and_rebind_go(
        root, lambda record: record.__setitem__("status", "PASS")
    )
    with pytest.raises(subject.AuditFailure, match="status mismatch"):
        audit_fixture(root, runner, contract)


@pytest.mark.parametrize("invalid", (None, 1))
def test_null_or_non_string_independent_prelaunch_status_fails(
    tmp_path: Path, invalid: object
) -> None:
    root, runner, contract = build_complete_fixture(tmp_path)
    mutate_independent_prelaunch_record_and_rebind_go(
        root, lambda record: record.__setitem__("status", invalid)
    )
    with pytest.raises(subject.AuditFailure, match="status mismatch"):
        audit_fixture(root, runner, contract)


def test_unknown_fifth_independent_prelaunch_identity_key_fails(
    tmp_path: Path,
) -> None:
    root, runner, contract = build_complete_fixture(tmp_path)
    mutate_independent_prelaunch_record_and_rebind_go(
        root, lambda record: record.__setitem__("unknown", True)
    )
    with pytest.raises(subject.AuditFailure, match="schema mismatch"):
        audit_fixture(root, runner, contract)


@pytest.mark.parametrize("field", ("path", "size_bytes", "sha256"))
def test_independent_prelaunch_path_size_or_sha_mismatch_fails(
    tmp_path: Path, field: str
) -> None:
    root, runner, contract = build_complete_fixture(tmp_path)

    def mutation(record: dict[str, object]) -> None:
        record[field] = (
            "independent_redteam/wrong.json"
            if field == "path"
            else int(record[field]) + 1
            if field == "size_bytes"
            else "0" * 64
        )

    mutate_independent_prelaunch_record_and_rebind_go(root, mutation)
    with pytest.raises(subject.AuditFailure, match="path identity|size mismatch|SHA-256 mismatch"):
        audit_fixture(root, runner, contract)


def test_full_recovered_production_shape_reaches_transaction_raw_decoded_and_offline_replay(
    tmp_path: Path,
) -> None:
    root, runner, contract = build_recovered_fixture(tmp_path)
    report = audit_fixture(root, runner, contract)
    assert report["status"] == "PASS_PRODUCER_POSTRUN_READONLY_AUDIT_V7"
    assert report["launch_and_transaction"]["transaction_inventory_recursively_closed"] is True
    assert report["raw_closure"]["ranges"] == contract.range_rows
    assert report["decoded_closure"]["site_rows"] == contract.site_count
    assert report["full_offline_raw_redecode"]["messages"] == contract.range_rows
    assert report["full_offline_raw_redecode"]["site_value_comparisons"] == (
        contract.range_rows * contract.site_count
    )
    assert report["full_offline_raw_redecode"]["group_value_comparisons"] == (
        contract.range_rows * len(contract.groups)
    )
    assert report["provenance"]["independent_prelaunch_audit"] == {
        **report["provenance"]["independent_prelaunch_audit_base_identity"],
        "status": "PASS_TO_CREATE_FINAL_AUTH_ONLY",
    }
    assert set(report["provenance"]["independent_prelaunch_audit"]) == {
        "path",
        "size_bytes",
        "sha256",
        "status",
    }


def test_frozen_v2_false_reject_incident_and_source_identities_are_exact() -> None:
    root = subject.ROOT_DEFAULT
    incident_path = root / subject.V2_FALSE_REJECT_INCIDENT_RELATIVE
    assert incident_path.stat().st_size == subject.V2_FALSE_REJECT_INCIDENT_SIZE_BYTES
    assert subject.sha256_file(incident_path) == subject.V2_FALSE_REJECT_INCIDENT_SHA256
    incident = json.loads(incident_path.read_text(encoding="utf-8"))
    assert incident["status"] == "SEALED_FALSE_REJECT_NO_POSTRUN_PASS_V3_SUPERSESSION_REQUIRED"
    assert "required_v3_supersession" in incident
    assert incident["root_cause"]["immutable_authorization_record"] == (
        subject.V4_ORIGINAL_V1_PROVENANCE[
            "original_independent_prelaunch_audit_v1"
        ]
    )
    assert artifact(subject.SUPERSEDED_V2_AUDITOR) == subject.V4_SUPERSEDED_V2_AUDITOR_IDENTITY
    assert artifact(subject.SUPERSEDED_V2_AUDITOR_TEST) == subject.V4_SUPERSEDED_V2_AUDITOR_TEST_IDENTITY


def test_v4_auditor_is_stdout_only_and_writes_zero_files(tmp_path: Path) -> None:
    root, runner, contract = build_complete_fixture(tmp_path)
    before = snapshot(tmp_path)
    report = audit_fixture(root, runner, contract)
    assert report["audit_files_written"] == 0
    assert report["audit_network_requests"] == 0
    assert snapshot(tmp_path) == before


def test_network_target_label_model_and_submission_routes_remain_absent() -> None:
    source = Path(subject.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        (node.module or "").split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    } | {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not ({"requests", "socket", "urllib", "http", "ftplib"} & imports)
    assert "submission_csv_created" in source
    assert "audit_labels_read" in source
    assert '"audit_models_fit": 0' in source
    assert '"audit_network_requests": 0' in source


@pytest.mark.parametrize(
    "authority_failure",
    ("missing_review", "missing_go", "malformed_go"),
)
def test_v4_authority_and_go_precede_any_parquet_or_data_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    authority_failure: str,
) -> None:
    root = tmp_path / "authority-first"
    sealer = tmp_path / "seal_v4.py"
    sealer_test = tmp_path / "test_seal_v4.py"
    sealer.write_text("# synthetic frozen V4 sealer\n", encoding="utf-8")
    sealer_test.write_text("# synthetic frozen V4 sealer test\n", encoding="utf-8")
    monkeypatch.setattr(subject, "V4_SEALER", sealer)
    monkeypatch.setattr(subject, "V4_SEALER_TEST", sealer_test)
    authorization_path, go_path = build_valid_v4_authorization_with_unavailable_review(
        root, sealer=sealer, sealer_test=sealer_test
    )
    expected_reason = "independent review is absent"
    if authority_failure == "missing_go":
        complete_v4_review_and_go(root, authorization_path, go_path)
        go_path.unlink()
        expected_reason = "independent GO is absent"
    elif authority_failure == "malformed_go":
        complete_v4_review_and_go(root, authorization_path, go_path)
        go = json.loads(go_path.read_text(encoding="utf-8"))
        go.pop("recovery_rerun_authorized")
        write_json(go_path, go)
        expected_reason = "independent GO schema mismatch"

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("data/parquet/decoder route was reached before V4 authority")

    monkeypatch.setattr(subject, "preflight_zero_target_parquet_schemas", forbidden)
    monkeypatch.setattr(pd, "read_parquet", forbidden)
    monkeypatch.setattr(subject, "decode_offline_grib_message", forbidden)
    original_open = Path.open

    def guarded_open(path: Path, *args: object, **kwargs: object) -> object:
        lowered = path.name.casefold()
        if lowered.endswith((".parquet", ".grib2", ".csv")):
            raise AssertionError("data file opened before V4 authority")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(subject.AuditFailure, match=expected_reason):
        subject._v4_validate_predata_authority(root, authorization_path, go_path)


@pytest.mark.parametrize(
    "mutation, expected_reason",
    (
        ("missing_v4_sealer", "authorization schema mismatch"),
        ("extra_v4_sealer_alias", "authorization schema mismatch"),
        ("support_alias", "V2-support binding schema mismatch"),
    ),
)
def test_v4_authorization_rejects_sealer_schema_and_support_aliases_before_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_reason: str,
) -> None:
    root = tmp_path / "authority-schema"
    sealer = tmp_path / "seal_v4.py"
    sealer_test = tmp_path / "test_seal_v4.py"
    sealer.write_text("# synthetic frozen V4 sealer\n", encoding="utf-8")
    sealer_test.write_text("# synthetic frozen V4 sealer test\n", encoding="utf-8")
    monkeypatch.setattr(subject, "V4_SEALER", sealer)
    monkeypatch.setattr(subject, "V4_SEALER_TEST", sealer_test)
    authorization_path, go_path = build_valid_v4_authorization_with_unavailable_review(
        root, sealer=sealer, sealer_test=sealer_test
    )
    complete_v4_review_and_go(root, authorization_path, go_path)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    if mutation == "missing_v4_sealer":
        authorization.pop("v4_sealer")
    elif mutation == "extra_v4_sealer_alias":
        authorization["sealer"] = authorization["v4_sealer"]
    else:
        support = authorization["v2_recovery_support"]
        support["module"] = support.pop("recovery_module")
    write_json(authorization_path, authorization)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("data route was reached before V4 authority rejection")

    monkeypatch.setattr(subject, "preflight_zero_target_parquet_schemas", forbidden)
    monkeypatch.setattr(pd, "read_parquet", forbidden)
    monkeypatch.setattr(subject, "decode_offline_grib_message", forbidden)
    with pytest.raises(subject.AuditFailure, match=expected_reason):
        subject._v4_validate_predata_authority(root, authorization_path, go_path)


def test_v4_predata_authority_accepts_complete_production_shaped_control_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "authority-complete"
    sealer = tmp_path / "seal_v4.py"
    sealer_test = tmp_path / "test_seal_v4.py"
    sealer.write_text("# synthetic frozen V4 sealer\n", encoding="utf-8")
    sealer_test.write_text("# synthetic frozen V4 sealer test\n", encoding="utf-8")
    monkeypatch.setattr(subject, "V4_SEALER", sealer)
    monkeypatch.setattr(subject, "V4_SEALER_TEST", sealer_test)
    authorization_path, go_path = build_valid_v4_authorization_with_unavailable_review(
        root, sealer=sealer, sealer_test=sealer_test
    )
    review_path = complete_v4_review_and_go(root, authorization_path, go_path)
    authority = subject._v4_validate_predata_authority(
        root, authorization_path, go_path
    )
    assert authority["authorization_identity"] == artifact(authorization_path, root)
    assert authority["review_identity"] == artifact(review_path, root)
    assert authority["go_identity"] == artifact(go_path, root)
    assert subject.canonical_payload_sha256(authority["recovered_afterstate"]) == (
        subject.V4_RECOVERED_AFTERSTATE_CANONICAL_SHA256
    )


@pytest.mark.parametrize(
    "control, mutation, expected_reason",
    (
        ("review", "missing", "independent review schema mismatch"),
        ("review", "extra", "independent review schema mismatch"),
        ("review", "false_check", "check set/verdict mismatch"),
        ("go", "extra", "independent GO schema mismatch"),
        ("go", "wrong_status", "independent GO header/binding mismatch"),
    ),
)
def test_v4_review_and_go_schema_or_verdict_tamper_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    control: str,
    mutation: str,
    expected_reason: str,
) -> None:
    root = tmp_path / f"authority-tamper-{control}-{mutation}"
    sealer = tmp_path / "seal_v4.py"
    sealer_test = tmp_path / "test_seal_v4.py"
    sealer.write_text("# synthetic frozen V4 sealer\n", encoding="utf-8")
    sealer_test.write_text("# synthetic frozen V4 sealer test\n", encoding="utf-8")
    monkeypatch.setattr(subject, "V4_SEALER", sealer)
    monkeypatch.setattr(subject, "V4_SEALER_TEST", sealer_test)
    authorization_path, go_path = build_valid_v4_authorization_with_unavailable_review(
        root, sealer=sealer, sealer_test=sealer_test
    )
    review_path = complete_v4_review_and_go(root, authorization_path, go_path)
    path = review_path if control == "review" else go_path
    payload = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "missing":
        payload.pop("verdict")
    elif mutation == "extra":
        payload["unexpected"] = True
    elif mutation == "false_check":
        payload["independent_checks"]["false_reject_incident_exact"] = False
    else:
        payload["status"] = "GO"
    write_json(path, payload)
    if control == "review":
        go = json.loads(go_path.read_text(encoding="utf-8"))
        go["independent_review"] = artifact(review_path, root)
        write_json(go_path, go)
    with pytest.raises(subject.AuditFailure, match=expected_reason):
        subject._v4_validate_predata_authority(root, authorization_path, go_path)


def build_v4_postauthority_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "postauthority"
    for relative in (subject.V4_AUTH_RELATIVE, subject.V4_REVIEW_RELATIVE, subject.V4_GO_RELATIVE):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)

    production_progress = sorted(
        (subject.ROOT_DEFAULT / "decoded" / "recovery_progress").iterdir(),
        key=lambda path: path.name,
    )
    progress_inventory = [artifact(path, subject.ROOT_DEFAULT) for path in production_progress]
    authorization: dict[str, object] = {
        "recovery_transaction": subject.V4_TRANSACTION_IDENTITIES,
        "recovery_history": {
            "file_count": 3,
            "completion_locks": subject.V4_COMPLETION_LOCK_IDENTITIES,
            "postcommit_input_audit": subject.V4_POSTCOMMIT_INPUT_AUDIT_IDENTITY,
        },
        "recovery_progress": {
            "file_count": 104,
            "inventory": progress_inventory,
            "inventory_canonical_sha256": subject.V4_PROGRESS_INVENTORY_CANONICAL_SHA256,
            "final_checkpoint": subject.V4_FINAL_PROGRESS_IDENTITY,
        },
        "canonical_outputs": subject.V4_CANONICAL_OUTPUT_IDENTITIES,
    }
    afterstate = subject._v4_recovered_afterstate_from_authorization(authorization)
    synthetic_state = {
        "incident": {}, "authorization": {}, "independent_review": {},
        "misplaced_go": {}, "canonical_go": {
            "path": subject.FROZEN_V3_CANONICAL_GO_RELATIVE,
            "present": False, "must_remain_absent": True,
        },
        "review_predates_misplaced_go": True,
        "misplaced_go_payload_canonical_sha256": (
            subject.FROZEN_V3_MISPLACED_GO_PAYLOAD_CANONICAL_SHA256
        ),
        "misplaced_go_payload_exact_v3_schema_and_crossbindings": True,
        "misplaced_go_authoritative": False,
        "v3_auditor_execution_authorized": False,
    }
    authorization.update(
        {
            "incident": {},
            "v2_false_reject_incident": {},
            **{role: dict(identity) for role, identity in subject.FROZEN_V3_SOURCE_IDENTITIES.items()},
            "v4_auditor": artifact(Path(subject.__file__).resolve()),
            "v4_auditor_test": artifact(subject.AUDITOR_TEST),
            "v4_sealer": artifact(subject.V4_SEALER),
            "v4_sealer_test": artifact(subject.V4_SEALER_TEST),
        }
    )
    authority = {
        "authorization": authorization,
        "authorization_identity": {}, "review_identity": {}, "go_identity": {},
        "recovered_afterstate": afterstate,
        "v3_path_mispublish_state": synthetic_state,
        "control_namespace": {},
    }

    for record in subject.V4_CANONICAL_OUTPUT_IDENTITIES.values():
        path = root / str(record["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    for record in subject.V4_TRANSACTION_IDENTITIES.values():
        path = root / str(record["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    for record in subject.V4_COMPLETION_LOCK_IDENTITIES.values():
        path = root / str(record["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    postcommit = root / str(subject.V4_POSTCOMMIT_INPUT_AUDIT_IDENTITY["path"])
    postcommit.parent.mkdir(parents=True, exist_ok=True)
    postcommit.touch()

    source_v2_authorization = json.loads(
        (
            subject.ROOT_DEFAULT
            / subject.V4_V2_CONTROL_RELATIVES["recovery_authorization_v2"]
        ).read_text(encoding="utf-8")
    )
    remnants = source_v2_authorization["documented_preplan_remnants"]
    v2_authorization_path = (
        root / subject.V4_V2_CONTROL_RELATIVES["recovery_authorization_v2"]
    )
    write_json(v2_authorization_path, {"documented_preplan_remnants": remnants})
    for record in remnants:
        path = root / str(record["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    for relative in (
        f"raw/output_transactions/{subject.V4_RECOVERY_ATTEMPT_ID}/staged/decoded",
        f"raw/output_transactions/{subject.V4_RECOVERY_ATTEMPT_ID}/staged/raw",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)

    progress_by_path = {str(record["path"]): record for record in progress_inventory}
    for relative in progress_by_path:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    def fake_verify(record: object, *_args: object, **_kwargs: object) -> dict[str, object]:
        return dict(record)

    def fake_identity(path: Path, identity_root: Path | None = None) -> dict[str, object]:
        relative = Path(path).relative_to(root).as_posix()
        if relative not in progress_by_path:
            raise AssertionError(f"unexpected identity read: {relative}")
        return dict(progress_by_path[relative])

    monkeypatch.setattr(subject, "verify_identity", fake_verify)
    monkeypatch.setattr(subject, "identity", fake_identity)
    monkeypatch.setattr(subject, "_v4_validate_path_mispublish_incident", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(subject, "_v4_validate_false_reject_incident", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        subject,
        "_v4_validate_frozen_v3_chain",
        lambda *_args, **_kwargs: {"state": synthetic_state},
    )
    monkeypatch.setattr(subject, "_v4_validate_control_namespace", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(subject, "_v4_control_temporary_files", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        subject, "_v4_verify_external_identity",
        lambda record, *_args, **_kwargs: dict(record),
    )
    return root, authority


@pytest.mark.parametrize("mutation", ("extra_staged", "extra_history"))
def test_v4_postauthority_recursive_closure_rejects_junk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    root, authority = build_v4_postauthority_fixture(tmp_path, monkeypatch)
    if mutation == "extra_staged":
        junk = (
            root
            / "raw"
            / "output_transactions"
            / subject.V4_RECOVERY_ATTEMPT_ID
            / "staged"
            / "decoded"
            / "junk.bin"
        )
        junk.write_bytes(b"junk")
        expected = "transaction recursive exact filesystem closure"
    else:
        junk = root / "decoded" / "recovery_history" / "junk.json"
        junk.write_text("{}\n", encoding="utf-8")
        expected = "recovery-history exact filesystem closure"
    with pytest.raises(subject.AuditFailure, match=expected):
        subject._v4_validate_postauthority_afterstate(root, authority)


def test_v4_postauthority_exact_closure_reports_bound_counts_and_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, authority = build_v4_postauthority_fixture(tmp_path, monkeypatch)
    summary = subject._v4_validate_postauthority_afterstate(root, authority)
    assert summary == {
        "recovered_afterstate": authority["recovered_afterstate"],
        "recovered_afterstate_canonical_sha256": (
            subject.V4_RECOVERED_AFTERSTATE_CANONICAL_SHA256
        ),
        "canonical_output_count": 8,
        "transaction_recursive_file_count": 4,
        "transaction_recursive_directory_count": 7,
        "recovery_history_file_count": 3,
        "recovery_progress_file_count": 104,
        "active_locks_present": 0,
        "v4_control_temporary_files_present": 0,
        "canonical_v3_go_present": 0,
        "misplaced_v3_go_rejected_and_rehashed": True,
        "control_namespace_exact": True,
    }


def test_v4_postauthority_recursive_closure_rejects_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, authority = build_v4_postauthority_fixture(tmp_path, monkeypatch)
    target = root / str(subject.V4_TRANSACTION_IDENTITIES["plan"]["path"])
    link = target.parent / "unexpected_link"
    try:
        os.symlink(target, link)
    except OSError as exc:
        pytest.skip(f"symlink privilege unavailable: {exc}")
    with pytest.raises(subject.AuditFailure, match="link/junction"):
        subject._v4_validate_postauthority_afterstate(root, authority)


def _run_v5_cli(
    tmp_path: Path,
    *,
    module: bool = True,
    include_controls: bool = True,
    include_dash_b: bool = True,
    env_override: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    root = tmp_path / "cli-root"
    (root / "prereg").mkdir(parents=True, exist_ok=True)
    (root / "independent_redteam").mkdir(parents=True, exist_ok=True)
    command = [str(Path(sys.executable).resolve())]
    if include_dash_b:
        command.append("-B")
    if module:
        command += ["-m", "scripts.audit_noaa_gfs_multiseason_raw_postrun_v5"]
    else:
        command += [str(Path(subject.__file__).resolve())]
    command += ["--root", str(root.resolve())]
    if include_controls:
        command += [
            "--authorization", str((root / subject.V5_AUTH_RELATIVE).resolve()),
            "--independent-go", str((root / subject.V5_GO_RELATIVE).resolve()),
        ]
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        }
    )
    environment.pop("PYTHONPYCACHEPREFIX", None)
    if env_override:
        environment.update(env_override)
    return subprocess.run(
        command,
        cwd=subject.REPO,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )


def test_canonical_module_entrypoint_reaches_authority_gate_read_only(
    tmp_path: Path,
) -> None:
    result = _run_v5_cli(tmp_path)
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report["status"] == "FAIL_PRODUCER_POSTRUN_AUDIT_V5"
    assert "canonical -m" not in report["reason"]
    assert "authorization" in report["reason"].casefold()
    assert report["audit_network_requests"] == report["audit_files_written"] == 0


def test_direct_file_entrypoint_is_rejected_before_authority_or_data(
    tmp_path: Path,
) -> None:
    result = _run_v5_cli(tmp_path, module=False)
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert "canonical -m module entrypoint" in report["reason"]


def test_missing_dash_b_is_rejected_before_authority_or_data(
    tmp_path: Path,
) -> None:
    result = _run_v5_cli(tmp_path, include_dash_b=False)
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert "interpreter/module argv is not exact authorized command" in report["reason"]


def test_omitted_explicit_control_args_are_rejected_before_data(tmp_path: Path) -> None:
    result = _run_v5_cli(tmp_path, include_controls=False)
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert "requires explicit --authorization and --independent-go" in report["reason"]


@pytest.mark.parametrize(
    "env_override",
    (
        {"PYTHONDONTWRITEBYTECODE": "0"},
        {"PYTHONPYCACHEPREFIX": "C:\\v5-forbidden-pycache"},
    ),
)
def test_bytecode_or_pycache_environment_is_rejected_before_data(
    tmp_path: Path, env_override: dict[str, str]
) -> None:
    result = _run_v5_cli(tmp_path, env_override=env_override)
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert "bytecode/cache environment mismatch" in report["reason"]


def test_recovered_afterstate_completion_lock_key_names_are_digest_bound() -> None:
    root = subject.ROOT_DEFAULT
    inventory = [
        artifact(path, root)
        for path in sorted(
            (root / "decoded" / "recovery_progress").iterdir(),
            key=lambda path: path.name,
        )
    ]
    authorization = {
        "recovery_transaction": subject.V4_TRANSACTION_IDENTITIES,
        "recovery_history": {
            "file_count": 3,
            "completion_locks": subject.V4_COMPLETION_LOCK_IDENTITIES,
            "postcommit_input_audit": subject.V4_POSTCOMMIT_INPUT_AUDIT_IDENTITY,
        },
        "recovery_progress": {
            "file_count": 104,
            "inventory": inventory,
            "inventory_canonical_sha256": (
                subject.V4_PROGRESS_INVENTORY_CANONICAL_SHA256
            ),
            "final_checkpoint": subject.V4_FINAL_PROGRESS_IDENTITY,
        },
        "canonical_outputs": subject.V4_CANONICAL_OUTPUT_IDENTITIES,
    }
    afterstate = subject._v4_recovered_afterstate_from_authorization(authorization)
    assert len(subject._v4_canonical_bytes(afterstate)) == 24_234
    assert subject.canonical_payload_sha256(afterstate) == (
        subject.V4_RECOVERED_AFTERSTATE_CANONICAL_SHA256
    )
    tampered = json.loads(json.dumps(authorization))
    locks = tampered["recovery_history"]["completion_locks"]
    locks["decode"] = locks.pop("decode_mutex_complete")
    locks["raw"] = locks.pop("raw_mutex_complete")
    with pytest.raises(subject.AuditFailure, match="recovery-history declaration"):
        subject._v4_recovered_afterstate_from_authorization(tampered)


def _complete_v4_authority_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> tuple[Path, Path, Path]:
    root = tmp_path / name
    sealer = tmp_path / f"{name}_sealer.py"
    sealer_test = tmp_path / f"{name}_sealer_test.py"
    sealer.write_text("# synthetic frozen V4 sealer\n", encoding="utf-8")
    sealer_test.write_text("# synthetic frozen V4 sealer test\n", encoding="utf-8")
    monkeypatch.setattr(subject, "V4_SEALER", sealer)
    monkeypatch.setattr(subject, "V4_SEALER_TEST", sealer_test)
    authorization_path, go_path = build_valid_v4_authorization_with_unavailable_review(
        root, sealer=sealer, sealer_test=sealer_test
    )
    complete_v4_review_and_go(root, authorization_path, go_path)
    return root, authorization_path, go_path


def _assert_v4_predata_failure_without_data(
    root: Path,
    authorization_path: Path,
    go_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    match: str,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("data/parquet/raw/decoder route reached before V4 authority")

    for name in (
        "preflight_zero_target_parquet_schemas", "_audit_provenance",
        "_audit_transaction_and_launch", "_audit_census", "_audit_raw_ranges",
        "_audit_request_events", "_audit_decoded", "_audit_full_offline_redecode",
        "decode_offline_grib_message", "decode_offline_grib_process_worker",
    ):
        monkeypatch.setattr(subject, name, forbidden)
    with pytest.raises(subject.AuditFailure, match=match):
        subject._v4_validate_predata_authority(root, authorization_path, go_path)


def test_v3_path_mispublish_incident_and_frozen_chain_are_exact() -> None:
    root = subject.ROOT_DEFAULT
    record = {
        "path": subject.V4_PATH_MISPUBLISH_INCIDENT_RELATIVE,
        "size_bytes": subject.V4_PATH_MISPUBLISH_INCIDENT_SIZE_BYTES,
        "sha256": subject.V4_PATH_MISPUBLISH_INCIDENT_SHA256,
    }
    incident = subject._v4_validate_path_mispublish_incident(root, record)
    frozen = subject._v4_validate_frozen_v3_chain(root)
    assert incident["status"] == (
        "SEALED_V3_GO_PATH_MISPUBLISH_NO_LIVE_AUDIT_V4_SUPERSESSION_REQUIRED"
    )
    assert frozen["state"]["misplaced_go"] == subject.FROZEN_V3_MISPLACED_GO_IDENTITY
    assert frozen["state"]["canonical_go"] == {
        "path": subject.FROZEN_V3_CANONICAL_GO_RELATIVE,
        "present": False,
        "must_remain_absent": True,
    }
    assert not os.path.lexists(str(root / subject.FROZEN_V3_CANONICAL_GO_RELATIVE))


def test_real_v3_authority_and_full_synthetic_core_proofs_are_compositional(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority_root, authorization_path, go_path = _complete_v4_authority_fixture(
        tmp_path, monkeypatch, "v3-chain-positive"
    )
    authority = subject._v4_validate_predata_authority(
        authority_root, authorization_path, go_path
    )
    assert authority["v3_path_mispublish_state"]["misplaced_go_authoritative"] is False
    assert authority["control_namespace"] == subject._v4_validate_control_namespace(
        authority_root
    )

    fixture_root = tmp_path / "full-shape"
    fixture_root.mkdir()
    recovered_root, runner, contract = build_recovered_fixture(fixture_root)
    report = audit_fixture(recovered_root, runner, contract)
    assert report["raw_closure"]["ranges"] == contract.range_rows
    assert report["decoded_closure"]["site_rows"] == contract.site_count
    assert report["full_offline_raw_redecode"]["messages"] == contract.range_rows


def test_canonical_v3_go_presence_fails_before_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, authorization_path, go_path = _complete_v4_authority_fixture(
        tmp_path, monkeypatch, "canonical-v3-go"
    )
    path = root / subject.FROZEN_V3_CANONICAL_GO_RELATIVE
    path.write_text("{}\n", encoding="utf-8")
    _assert_v4_predata_failure_without_data(
        root, authorization_path, go_path, monkeypatch,
        "namespace inventory|must remain absent",
    )


@pytest.mark.parametrize("mutation", ("missing", "bytes", "link"))
def test_misplaced_v3_go_missing_identity_or_link_tamper_fails_before_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    root, authorization_path, go_path = _complete_v4_authority_fixture(
        tmp_path, monkeypatch, f"misplaced-{mutation}"
    )
    path = root / subject.FROZEN_V3_MISPLACED_GO_RELATIVE
    if mutation == "missing":
        path.unlink()
    elif mutation == "bytes":
        path.write_text("{}\n", encoding="utf-8")
    else:
        target = path.with_name("unbound-v3-go-target.json")
        target.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        path.unlink()
        try:
            os.symlink(target, path)
        except OSError as exc:
            pytest.skip(f"symlink privilege unavailable: {exc}")
    _assert_v4_predata_failure_without_data(
        root, authorization_path, go_path, monkeypatch,
        "namespace inventory|non-regular|identity|size mismatch|absent",
    )


@pytest.mark.parametrize("mutation", ("chronology", "crosslink"))
def test_v3_chain_chronology_or_crosslink_tamper_fails_before_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    root = tmp_path / f"v3-chain-{mutation}"
    copy_v4_predata_controls(root)
    misplaced_path = root / subject.FROZEN_V3_MISPLACED_GO_RELATIVE
    misplaced = json.loads(misplaced_path.read_text(encoding="utf-8"))
    if mutation == "chronology":
        misplaced["created_utc"] = "2026-08-10T22:00:00.000000Z"
    else:
        misplaced["independent_review"] = {
            "path": subject.FROZEN_V3_REVIEW_RELATIVE,
            "size_bytes": 1,
            "sha256": "0" * 64,
        }
    write_json(misplaced_path, misplaced)
    monkeypatch.setattr(
        subject,
        "FROZEN_V3_MISPLACED_GO_IDENTITY",
        artifact(misplaced_path, root),
    )
    monkeypatch.setattr(
        subject,
        "FROZEN_V3_MISPLACED_GO_PAYLOAD_CANONICAL_SHA256",
        subject.canonical_payload_sha256(misplaced),
    )

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("data route reached while checking frozen V3 control chain")

    for name in (
        "preflight_zero_target_parquet_schemas", "_audit_raw_ranges",
        "_audit_decoded", "_audit_full_offline_redecode",
    ):
        monkeypatch.setattr(subject, name, forbidden)
    with pytest.raises(subject.AuditFailure, match="chronology|cross-binding"):
        subject._v4_validate_frozen_v3_chain(root)


@pytest.mark.parametrize(
    "relative",
    (
        "prereg/decode_recovery_postrun_audit_EVIL.json",
        "incidents/.TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_EVIL.json",
    ),
)
def test_v4_unbound_matching_control_namespace_entry_fails_before_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative: str,
) -> None:
    root, authorization_path, go_path = _complete_v4_authority_fixture(
        tmp_path, monkeypatch, "unbound-namespace"
    )
    extra = root / relative
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_text("{}\n", encoding="utf-8")
    _assert_v4_predata_failure_without_data(
        root,
        authorization_path,
        go_path,
        monkeypatch,
        "namespace inventory",
    )


@pytest.mark.parametrize(
    "relative",
    (
        "TRACK_A_RAW_PRODUCER_POSTRUN_READONLY_AUDIT_V3.json",
        "raw/TRACK_A_RAW_PRODUCER_POSTRUN_READONLY_AUDIT_V2.json",
        "decoded/TRACK_A_RAW_PRODUCER_POSTRUN_READONLY_AUDIT_V4.json",
    ),
)
def test_v2_v3_v4_report_candidate_fails_before_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative: str,
) -> None:
    root, authorization_path, go_path = _complete_v4_authority_fixture(
        tmp_path, monkeypatch, "report-before-data"
    )
    report = root / relative
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("{}\n", encoding="utf-8")
    _assert_v4_predata_failure_without_data(
        root,
        authorization_path,
        go_path,
        monkeypatch,
        "postrun report candidate",
    )


@pytest.mark.parametrize(
    "mutation",
    ("authorization_extra", "go_status", "go_crosslink", "required_command", "path"),
)
def test_v4_authority_schema_path_status_crosslink_or_command_tamper_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    root, authorization_path, go_path = _complete_v4_authority_fixture(
        tmp_path, monkeypatch, f"v4-authority-{mutation}"
    )
    called_authorization_path = authorization_path
    if mutation == "authorization_extra":
        authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
        authorization["unknown"] = True
        write_json(authorization_path, authorization)
    elif mutation == "go_status":
        go = json.loads(go_path.read_text(encoding="utf-8"))
        go["status"] = "GO_V4_WRONG"
        write_json(go_path, go)
    elif mutation == "go_crosslink":
        go = json.loads(go_path.read_text(encoding="utf-8"))
        go["authorization"] = {
            "path": subject.V4_AUTH_RELATIVE,
            "size_bytes": 1,
            "sha256": "0" * 64,
        }
        write_json(go_path, go)
    elif mutation == "required_command":
        authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
        authorization["required_command"] = ["python", "-m", "wrong.module"]
        write_json(authorization_path, authorization)
    else:
        alternate = authorization_path.with_name("authorization-alias.json")
        shutil.copy2(authorization_path, alternate)
        called_authorization_path = alternate
    _assert_v4_predata_failure_without_data(
        root,
        called_authorization_path,
        go_path,
        monkeypatch,
        "schema|header|binding|cross-binding|command|canonical",
    )


@pytest.mark.parametrize(
    "mutation", ("namespace", "authorization", "source", "lock", "report")
)
def test_v4_postaudit_rechecks_mispublication_namespace_and_afterstate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    root, authorization_path, go_path = _complete_v4_authority_fixture(
        tmp_path, monkeypatch, f"postaudit-{mutation}"
    )
    authority = subject._v4_validate_predata_authority(root, authorization_path, go_path)
    if mutation == "namespace":
        extra = root / "incidents" / ".TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_EVIL.json"
        extra.write_text("{}\n", encoding="utf-8")
    elif mutation == "authorization":
        with authorization_path.open("ab") as handle:
            handle.write(b" ")
    elif mutation == "source":
        source = Path(subject.V4_SEALER)
        monkeypatch.setattr(
            subject,
            "_v4_verify_external_identity",
            lambda record, path, **kwargs: (
                (_ for _ in ()).throw(subject.AuditFailure("forced V4 source drift"))
                if Path(path) == source
                else dict(record)
            ),
        )
    elif mutation == "lock":
        lock = root / "raw" / "RAW_LAUNCH_ACTIVE.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text("{}\n", encoding="utf-8")
    else:
        report = root / "raw" / "TRACK_A_RAW_PRODUCER_POSTRUN_READONLY_AUDIT_V3.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        subject.AuditFailure,
        match="namespace|identity|source drift|active lock|postaudit|afterstate/temp",
    ):
        subject._v4_validate_postauthority_afterstate(root, authority)


def copy_v5_predata_controls(root: Path) -> None:
    copy_v4_predata_controls(root)
    for relative in (
        subject.V4_AUTH_RELATIVE,
        subject.V4_REVIEW_RELATIVE,
        subject.V5_TRANSPORT_INCIDENT_RELATIVE,
        subject.V5_TRANSPORT_CORRECTION_RELATIVE,
    ):
        source = subject.ROOT_DEFAULT / relative
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def frozen_v4_authorization_base() -> dict[str, object]:
    return {
        "identity": dict(subject.FROZEN_V4_AUTH_IDENTITY),
        "schema_version": 4,
        "status": "AUTHORIZED_PENDING_INDEPENDENT_REVIEW_AND_GO_V4",
        "audit_attempt_id": subject.FROZEN_V4_AUDIT_ATTEMPT_ID,
        "recovery_attempt_id": subject.V4_RECOVERY_ATTEMPT_ID,
        "payload_canonical_sha256": subject.FROZEN_V4_AUTH_PAYLOAD_CANONICAL_SHA256,
        "test_evidence_canonical_sha256": (
            subject.FROZEN_V4_TEST_EVIDENCE_CANONICAL_SHA256
        ),
        "recovered_afterstate_canonical_sha256": (
            subject.V4_RECOVERED_AFTERSTATE_CANONICAL_SHA256
        ),
    }


def synthetic_v5_test_evidence(
    created_utc: str, bound: dict[str, dict[str, object]]
) -> dict[str, object]:
    runs: dict[str, object] = {}
    for role, relative in {
        "v5_auditor_tests": "tests/test_noaa_gfs_multiseason_raw_postrun_v5.py",
        "v5_sealer_tests": "tests/test_seal_noaa_gfs_multiseason_postrun_audit_v5.py",
        "frozen_v4_auditor_tests": "tests/test_noaa_gfs_multiseason_raw_postrun_v4.py",
        "frozen_v4_sealer_tests": "tests/test_seal_noaa_gfs_multiseason_postrun_audit_v4.py",
    }.items():
        runs[role] = {
            "command": subject._v5_test_command(relative),
            "exit_code": 0,
            "summary": "1 passed",
            "stdout_sha256": "0" * 64,
            "stderr_sha256": (
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            ),
            "network_guard_installed": True,
            "cacheprovider_disabled": True,
        }
    return {
        "schema_version": 1,
        "artifact_type": "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_V5_TEST_EVIDENCE",
        "status": "PASS_FROZEN_V5_AUDITOR_AND_SEALER_TESTS",
        "created_utc": created_utc,
        "bound_identities": bound,
        "source_compile": {
            "method": "compile_exact_source_no_pyc",
            "result": "PASS",
            "files": [
                bound["v5_auditor"],
                bound["v5_auditor_test"],
                bound["v5_sealer"],
                bound["v5_sealer_test"],
            ],
        },
        "pytest_isolation": "ONE_TEST_FILE_PER_CLEAN_NETWORK_DENIED_SUBPROCESS",
        "test_runs": runs,
        "required_test_names": list(subject.V5_REQUIRED_TEST_NAMES),
        "required_test_names_present": True,
        "all_exit_codes_zero": True,
        "production_shape_regression": {
            "v4_authorization_base_validated": True,
            "v4_transport_truncation_incident_validated": True,
            "compact_afterstate_commitment_validated": True,
            "full_afterstate_reconstructed_in_memory": True,
            "original_v1_four_key_identity_passed": True,
            "transaction_reached": True,
            "raw_reached": True,
            "decoded_reached": True,
            "offline_replay_reached": True,
            "synthetic_fixture": True,
        },
        "real_seven_spawn_regression": {
            "start_method": "spawn",
            "entrypoint_module": "scripts.audit_noaa_gfs_multiseason_raw_postrun_v5",
            "worker_callable_module": "scripts.audit_noaa_gfs_multiseason_raw_postrun_v5",
            "max_workers": 7,
            "distinct_worker_pids": 7,
            "first_wave_synchronized": True,
            "network_requests": 0,
            "audit_files_written": 0,
            "labels_read": False,
            "arrays_2024_read": False,
            "arrays_2025_read": False,
        },
        "stdout_only_regression": True,
        "network_guard": {
            "parent_test_process_guarded": True,
            "sys_audit_hook_in_parent": True,
            "parent_socket_api_denied": True,
            "spawned_worker_network_route_static_absent": True,
            "spawned_worker_runtime_guard_installed": False,
            "external_packet_capture": False,
        },
        "network_requests": 0,
        "audit_files_written": 0,
        "labels_read": False,
        "arrays_2024_read": False,
        "arrays_2025_read": False,
        "models_fit": 0,
        "submission_csv_created": False,
    }


def build_valid_v5_authority(root: Path) -> tuple[Path, Path, Path, dict[str, object]]:
    copy_v5_predata_controls(root)
    base_record = frozen_v4_authorization_base()
    v4_base = subject._v5_load_and_validate_v4_authorization_base(root, base_record)
    rejected = subject._v5_load_rejected_v4_review(root, v4_base)
    incident_record = artifact(root / subject.V5_TRANSPORT_INCIDENT_RELATIVE, root)
    correction_record = artifact(root / subject.V5_TRANSPORT_CORRECTION_RELATIVE, root)
    subject._v5_validate_incident_chain(root, incident_record, correction_record)
    source_identities = {
        **subject.FROZEN_V4_SOURCE_IDENTITIES,
        "v5_auditor": artifact(Path(subject.__file__).resolve()),
        "v5_auditor_test": artifact(subject.AUDITOR_TEST),
        "v5_sealer": artifact(subject.V5_SEALER),
        "v5_sealer_test": artifact(subject.V5_SEALER_TEST),
    }
    bound = {
        "v4_authorization": dict(subject.FROZEN_V4_AUTH_IDENTITY),
        "v4_review_transport_truncation_incident": incident_record,
        "v4_review_transport_truncation_incident_correction": correction_record,
        **source_identities,
    }
    created_utc = "2026-08-11T01:30:00.000000Z"
    attempt_id = "postrun_audit_v5__20260811T013000000000Z"
    authorization_path = root / subject.V5_AUTH_RELATIVE
    review_path = root / subject.V5_REVIEW_RELATIVE
    go_path = root / subject.V5_GO_RELATIVE
    policy = {
        "max_spawn_processes": 7,
        "network_requests_allowed": 0,
        "audit_files_written_allowed": 0,
        "labels_read_allowed": False,
        "arrays_2024_read_allowed": False,
        "arrays_2025_read_allowed": False,
        "models_fit_allowed": 0,
        "submission_csv_allowed": False,
        "full_offline_redecode_required": True,
        "stdout_only": True,
    }
    authorization = {
        "schema_version": 5,
        "artifact_type": "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDIT_AUTHORIZATION_V5",
        "status": "AUTHORIZED_PENDING_INDEPENDENT_REVIEW_AND_GO_V5",
        "created_utc": created_utc,
        "audit_attempt_id": attempt_id,
        "incident": incident_record,
        "incident_correction": correction_record,
        "v4_authorization_base": base_record,
        "v4_review_transport_truncation_state": rejected["state"],
        **source_identities,
        "recovery_attempt_id": subject.V4_RECOVERY_ATTEMPT_ID,
        "recovered_afterstate_commitment": dict(
            subject.V5_RECOVERED_AFTERSTATE_COMMITMENT
        ),
        "preaudit_zero_mutation_snapshot": {
            "incident_postfailure_state_canonical_sha256": (
                subject.V5_INCIDENT_POSTFAILURE_STATE_CANONICAL_SHA256
            ),
            "recovered_afterstate_commitment_canonical_sha256": (
                subject.V5_RECOVERED_AFTERSTATE_COMMITMENT_CANONICAL_SHA256
            ),
            "active_locks": {
                "raw": {"path": "raw/RAW_LAUNCH_ACTIVE.lock", "present": False},
                "decoded": {
                    "path": "decoded/DECODE_RECOVERY_ACTIVE.lock",
                    "present": False,
                },
            },
            "v5_control_temporary_files": {
                "checked_destination_relative_paths": list(
                    subject._v5_control_destination_relatives()
                ),
                "matching_temporary_files": [],
            },
            "postrun_audit_output_files_present": 0,
            "network_requests": 0,
            "audit_files_written": 0,
            "labels_read": False,
            "arrays_2024_read": False,
            "arrays_2025_read": False,
            "models_fit": 0,
            "submission_csv_created": False,
        },
        "runtime_identity_sha256": subject.V4_RUNTIME_IDENTITY_SHA256,
        "test_evidence": synthetic_v5_test_evidence(created_utc, bound),
        "required_command": subject._v5_required_command(
            root, authorization_path, go_path
        ),
        **policy,
        "independent_review_required": True,
        "independent_go_required": True,
    }
    assert set(authorization) == subject.V5_AUTH_KEYS
    write_json(authorization_path, authorization)
    common = {
        field: authorization[field]
        for field in (
            "incident", "incident_correction", "v4_authorization_base",
            "v4_review_transport_truncation_state",
            *subject.FROZEN_V4_SOURCE_IDENTITIES.keys(),
            "v5_auditor", "v5_auditor_test", "v5_sealer", "v5_sealer_test",
            "recovered_afterstate_commitment",
        )
    }
    review = {
        "schema_version": 5,
        "artifact_type": "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_REVIEW_V5",
        "status": "PASS_PENDING_INDEPENDENT_GO_V5",
        "verdict": "GO_RECOMMENDED",
        "created_utc": "2026-08-11T01:31:00.000000Z",
        "audit_attempt_id": attempt_id,
        "authorization": artifact(authorization_path, root),
        **common,
        "test_evidence_recheck": {
            "authorization_test_evidence_canonical_sha256": (
                subject.canonical_payload_sha256(authorization["test_evidence"])
            ),
            "bound_identities_rehashed": True,
            "source_compile_exact": True,
            "required_test_names_exact": True,
            "all_exit_codes_zero": True,
            "production_shape_regression_exact": True,
            "real_seven_spawn_regression_exact": True,
            "stdout_only_zero_write_exact": True,
            "network_zero_target_free_exact": True,
        },
        "independent_checks": {key: True for key in subject.V5_REVIEW_CHECKS},
        **policy,
        "auditor_execution_started": False,
        "independent_go_required": True,
    }
    assert set(review) == subject.V5_REVIEW_KEYS
    write_json(review_path, review)
    go = {
        "schema_version": 5,
        "artifact_type": "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_GO_V5",
        "status": "GO_V5_READ_ONLY_NETWORK_ZERO_FULL_OFFLINE_REPLAY",
        "created_utc": "2026-08-11T01:32:00.000000Z",
        "audit_attempt_id": attempt_id,
        "authorization": artifact(authorization_path, root),
        "independent_review": artifact(review_path, root),
        **common,
        "recovery_attempt_id": subject.V4_RECOVERY_ATTEMPT_ID,
        "required_command": authorization["required_command"],
        **policy,
        "recovery_rerun_authorized": False,
    }
    assert set(go) == subject.V5_GO_KEYS
    write_json(go_path, go)
    return authorization_path, review_path, go_path, authorization


def build_v5_postauthority_metadata_closure(
    root: Path,
    authority: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Materialize the exact metadata namespace without copying recovered payloads."""

    v4_authorization = authority["v4_base"]["authorization"]
    expected_records: dict[str, dict[str, object]] = {}
    for record in v4_authorization["canonical_outputs"].values():
        expected_records[str(record["path"])] = dict(record)
    for record in v4_authorization["recovery_transaction"].values():
        expected_records[str(record["path"])] = dict(record)
    history = v4_authorization["recovery_history"]
    for record in history["completion_locks"].values():
        expected_records[str(record["path"])] = dict(record)
    postcommit = history["postcommit_input_audit"]
    expected_records[str(postcommit["path"])] = dict(postcommit)
    v2_authorization = json.loads(
        (root / subject.V4_V2_CONTROL_RELATIVES["recovery_authorization_v2"])
        .read_text(encoding="utf-8")
    )
    for record in v2_authorization["documented_preplan_remnants"]:
        expected_records[str(record["path"])] = dict(record)
    for record in v4_authorization["recovery_progress"]["inventory"]:
        expected_records[str(record["path"])] = dict(record)

    transaction_directories = {
        "raw/output_transactions/20260810T151057590738Z__pid3536__5f3e0fb20132",
        "raw/output_transactions/20260810T151057590738Z__pid3536__5f3e0fb20132/staged",
        "raw/output_transactions/20260810T151057590738Z__pid3536__5f3e0fb20132/staged/raw",
        f"raw/output_transactions/{subject.V4_RECOVERY_ATTEMPT_ID}",
        f"raw/output_transactions/{subject.V4_RECOVERY_ATTEMPT_ID}/staged",
        f"raw/output_transactions/{subject.V4_RECOVERY_ATTEMPT_ID}/staged/decoded",
        f"raw/output_transactions/{subject.V4_RECOVERY_ATTEMPT_ID}/staged/raw",
    }
    for relative in transaction_directories:
        (root / relative).mkdir(parents=True, exist_ok=True)
    (root / "decoded" / "recovery_history").mkdir(parents=True, exist_ok=True)
    (root / "decoded" / "recovery_progress").mkdir(parents=True, exist_ok=True)
    for relative in expected_records:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)

    original_identity = subject.identity

    def metadata_identity(
        path: Path, identity_root: Path | None = None
    ) -> dict[str, object]:
        resolved = Path(path).resolve()
        try:
            relative = resolved.relative_to(root.resolve()).as_posix()
        except ValueError:
            return original_identity(path, identity_root)
        record = expected_records.get(relative)
        if record is None:
            return original_identity(path, identity_root)
        assert resolved.is_file() and not resolved.is_symlink()
        return dict(record)

    monkeypatch.setattr(subject, "identity", metadata_identity)


def test_real_immutable_v4_authorization_base_passes(tmp_path: Path) -> None:
    root = tmp_path / "v5-positive"
    authorization_path, review_path, go_path, _payload = build_valid_v5_authority(root)
    authority = subject._v5_validate_predata_authority(
        root, authorization_path, go_path
    )
    assert authority["authorization_identity"] == artifact(authorization_path, root)
    assert authority["review_identity"] == artifact(review_path, root)
    assert authority["recovered_afterstate_commitment"] == (
        subject.V5_RECOVERED_AFTERSTATE_COMMITMENT
    )
    assert len(authority["recovered_afterstate"]["recovery_progress"]["inventory"]) == 104


@pytest.mark.parametrize(
    "field",
    (
        "identity_path", "identity_size", "identity_sha", "schema_version", "status",
        "payload_digest", "test_digest", "afterstate_digest",
    ),
)
def test_v4_authorization_base_path_size_sha_schema_status_or_digest_tamper_fails(
    tmp_path: Path, field: str
) -> None:
    root = tmp_path / f"base-tamper-{field}"
    copy_v5_predata_controls(root)
    base = frozen_v4_authorization_base()
    if field.startswith("identity_"):
        part = field.removeprefix("identity_")
        key = {"path": "path", "size": "size_bytes", "sha": "sha256"}[part]
        base["identity"][key] = (
            "wrong.json" if key == "path" else (1 if key == "size_bytes" else "0" * 64)
        )
    elif field == "payload_digest":
        base["payload_canonical_sha256"] = "0" * 64
    elif field == "test_digest":
        base["test_evidence_canonical_sha256"] = "0" * 64
    elif field == "afterstate_digest":
        base["recovered_afterstate_canonical_sha256"] = "0" * 64
    else:
        base[field] = 99 if field == "schema_version" else "WRONG"
    with pytest.raises(subject.AuditFailure):
        subject._v5_load_and_validate_v4_authorization_base(root, base)


def test_v4_review_transport_truncation_incident_and_frozen_chain_are_exact() -> None:
    root = subject.ROOT_DEFAULT
    incident = artifact(root / subject.V5_TRANSPORT_INCIDENT_RELATIVE, root)
    correction = artifact(root / subject.V5_TRANSPORT_CORRECTION_RELATIVE, root)
    chain = subject._v5_validate_incident_chain(root, incident, correction)
    assert chain["exact_one_override_applied"] is True
    assert chain["corrected_incident"]["mispublication"][
        "actual_review_progress_inventory_canonical_sha256"
    ] == subject.FROZEN_V4_ACTUAL_PROGRESS_INVENTORY_CANONICAL_SHA256
    v4_tree = ast.parse(subject.FROZEN_V4_AUDITOR.read_text(encoding="utf-8"))
    v5_tree = ast.parse(subject.FROZEN_V5_AUDITOR.read_text(encoding="utf-8"))
    v4_functions = {
        node.name: node
        for node in v4_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    v5_functions = {
        node.name: node
        for node in v5_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    ordered_v4_names = [
        node.name
        for node in v4_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    first = ordered_v4_names.index("recovery_progress_counts")
    last = ordered_v4_names.index("_audit_canonical_metadata")
    production_core_names = ordered_v4_names[first : last + 1]
    assert len(production_core_names) == 61
    assert "v5_spawn_probe_barrier_initializer" not in production_core_names
    for name in production_core_names:
        assert ast.dump(v5_functions[name], include_attributes=False) == ast.dump(
            v4_functions[name], include_attributes=False
        ), name


def test_v4_review_transport_truncation_correction_tamper_fails(
    tmp_path: Path,
) -> None:
    root = tmp_path / "correction-tamper"
    copy_v5_predata_controls(root)
    correction_path = root / subject.V5_TRANSPORT_CORRECTION_RELATIVE
    correction = json.loads(correction_path.read_text(encoding="utf-8"))
    correction["correction"]["authoritative_sha256"] = "0" * 64
    write_json(correction_path, correction)
    with pytest.raises(subject.AuditFailure, match="correction"):
        subject._v5_validate_incident_chain(
            root,
            artifact(root / subject.V5_TRANSPORT_INCIDENT_RELATIVE, root),
            artifact(correction_path, root),
        )


def test_primary_transport_incident_without_correction_fails(tmp_path: Path) -> None:
    root = tmp_path / "primary-alone"
    copy_v5_predata_controls(root)
    correction = root / subject.V5_TRANSPORT_CORRECTION_RELATIVE
    correction.unlink()
    with pytest.raises(subject.AuditFailure, match="correction|absent|identity"):
        subject._v5_validate_incident_chain(
            root,
            artifact(root / subject.V5_TRANSPORT_INCIDENT_RELATIVE, root),
            {
                "path": subject.V5_TRANSPORT_CORRECTION_RELATIVE,
                "size_bytes": subject.V5_TRANSPORT_CORRECTION_SIZE_BYTES,
                "sha256": subject.V5_TRANSPORT_CORRECTION_SHA256,
            },
        )


def test_v4_review_raw_duplicate_and_transport_marker_are_detected_exactly(
    tmp_path: Path,
) -> None:
    root = tmp_path / "raw-duplicate"
    copy_v5_predata_controls(root)
    base = subject._v5_load_and_validate_v4_authorization_base(
        root, frozen_v4_authorization_base()
    )
    rejected = subject._v5_load_rejected_v4_review(root, base)
    assert rejected["duplicate_count"] == 1
    assert rejected["transport_marker_count"] == 1


def test_v4_review_missing_progress_rows_and_wrong_effective_sha_are_exact(
    tmp_path: Path,
) -> None:
    root = tmp_path / "missing-progress"
    copy_v5_predata_controls(root)
    base = subject._v5_load_and_validate_v4_authorization_base(
        root, frozen_v4_authorization_base()
    )
    rejected = subject._v5_load_rejected_v4_review(root, base)
    inventory = rejected["review"]["recovered_afterstate"]["recovery_progress"]["inventory"]
    assert len(inventory) == 101
    assert subject.canonical_payload_sha256(inventory) == (
        subject.FROZEN_V4_ACTUAL_PROGRESS_INVENTORY_CANONICAL_SHA256
    )


def test_intended_v4_review_reconstruction_matches_frozen_38918_d0e1(
    tmp_path: Path,
) -> None:
    root = tmp_path / "intended-review"
    copy_v5_predata_controls(root)
    base = subject._v5_load_and_validate_v4_authorization_base(
        root, frozen_v4_authorization_base()
    )
    intended = subject._v5_load_rejected_v4_review(root, base)["intended_review"]
    encoded = (json.dumps(intended, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    assert len(encoded) == 38_918
    assert hashlib.sha256(encoded).hexdigest() == (
        subject.FROZEN_V4_INTENDED_REVIEW_PRETTY_IDENTITY["sha256"]
    )


def assert_v5_predata_failure_without_data(
    root: Path,
    authorization_path: Path,
    go_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    match: str,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("data/parquet/raw/decoder route reached before V5 authority")

    for name in (
        "preflight_zero_target_parquet_schemas", "parquet_schema_columns",
        "_audit_provenance", "_audit_transaction_and_launch", "_audit_census",
        "_audit_raw_ranges", "_audit_request_events", "_audit_decoded",
        "_audit_full_offline_redecode", "decode_offline_grib_message",
        "decode_offline_grib_process_worker",
    ):
        monkeypatch.setattr(subject, name, forbidden)
    original_open = Path.open

    def guarded_open(path: Path, *args: object, **kwargs: object) -> object:
        if path.suffix.casefold() in {".parquet", ".grib2", ".csv"}:
            raise AssertionError("output/data file opened before V5 authority")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(subject.AuditFailure, match=match):
        subject._v5_validate_predata_authority(root, authorization_path, go_path)


def test_canonical_v4_go_presence_fails_before_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "canonical-v4-go"
    authorization_path, _review_path, go_path, _ = build_valid_v5_authority(root)
    forbidden = root / subject.V4_GO_RELATIVE
    forbidden.write_text("{}\n", encoding="utf-8")
    assert_v5_predata_failure_without_data(
        root, authorization_path, go_path, monkeypatch,
        "namespace inventory|canonical V4 GO",
    )


@pytest.mark.parametrize("mutation", ("missing", "bytes", "link"))
def test_malformed_v4_review_identity_link_or_payload_tamper_fails_before_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    root = tmp_path / f"malformed-review-{mutation}"
    authorization_path, _review_path, go_path, _ = build_valid_v5_authority(root)
    malformed = root / subject.V4_REVIEW_RELATIVE
    if mutation == "missing":
        malformed.unlink()
    elif mutation == "bytes":
        malformed.write_text("{}\n", encoding="utf-8")
    else:
        target = malformed.with_name("unbound-review-target.json")
        shutil.copy2(malformed, target)
        malformed.unlink()
        try:
            os.symlink(target, malformed)
        except OSError as exc:
            pytest.skip(f"symlink privilege unavailable: {exc}")
    assert_v5_predata_failure_without_data(
        root, authorization_path, go_path, monkeypatch,
        "namespace inventory|identity|absent|link|size mismatch",
    )


@pytest.mark.parametrize("mutation", ("chronology", "crosslink"))
def test_v4_chain_chronology_or_crosslink_tamper_fails_before_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    root = tmp_path / f"chain-{mutation}"
    authorization_path, review_path, go_path, _ = build_valid_v5_authority(root)
    if mutation == "chronology":
        authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
        authorization["created_utc"] = "2026-08-11T00:58:00.000000Z"
        authorization["audit_attempt_id"] = "postrun_audit_v5__20260811T005800000000Z"
        write_json(authorization_path, authorization)
    else:
        review = json.loads(review_path.read_text(encoding="utf-8"))
        review["v4_authorization_base"]["payload_canonical_sha256"] = "0" * 64
        write_json(review_path, review)
        go = json.loads(go_path.read_text(encoding="utf-8"))
        go["independent_review"] = artifact(review_path, root)
        write_json(go_path, go)
    assert_v5_predata_failure_without_data(
        root, authorization_path, go_path, monkeypatch, "chronology|cross-binding"
    )


def test_compact_recovered_afterstate_commitment_matches_v4_authorization(
    tmp_path: Path,
) -> None:
    root = tmp_path / "commitment-positive"
    copy_v5_predata_controls(root)
    base = subject._v5_load_and_validate_v4_authorization_base(
        root, frozen_v4_authorization_base()
    )
    assert subject._v5_recovered_afterstate_commitment(base["afterstate"]) == (
        subject.V5_RECOVERED_AFTERSTATE_COMMITMENT
    )


@pytest.mark.parametrize("field", tuple(subject.V5_RECOVERED_AFTERSTATE_COMMITMENT))
def test_legacy_v5_compact_commitment_field_or_inventory_digest_tamper_fails(
    tmp_path: Path, field: str
) -> None:
    root = tmp_path / "commitment"
    authorization_path, review_path, go_path, _ = build_valid_v5_authority(root)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    go = json.loads(go_path.read_text(encoding="utf-8"))
    value = authorization["recovered_afterstate_commitment"][field]
    replacement = (not value) if isinstance(value, bool) else (value + 1 if isinstance(value, int) else "0" * 64)
    for payload in (authorization, review, go):
        payload["recovered_afterstate_commitment"][field] = replacement
    write_json(authorization_path, authorization)
    review["authorization"] = artifact(authorization_path, root)
    write_json(review_path, review)
    go["authorization"] = artifact(authorization_path, root)
    go["independent_review"] = artifact(review_path, root)
    write_json(go_path, go)
    with pytest.raises(subject.AuditFailure, match="commitment"):
        subject._v5_validate_predata_authority(root, authorization_path, go_path)


def test_full_recovered_afterstate_is_not_embedded_in_v5_authority_controls(
    tmp_path: Path,
) -> None:
    root = tmp_path / "compact-only"
    authorization_path, review_path, go_path, _ = build_valid_v5_authority(root)
    for path in (authorization_path, review_path, go_path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert "recovered_afterstate" not in payload
        assert "recovery_progress" not in payload
        assert "inventory" not in payload
        subject._v5_assert_compact_control(payload, label=path.name)
    assert review_path.stat().st_size <= 12_000
    assert go_path.stat().st_size <= 12_000


def test_real_v5_authority_and_full_synthetic_core_proofs_are_compositional(
    tmp_path: Path,
) -> None:
    authority_root = tmp_path / "authority"
    authorization_path, _review_path, go_path, _ = build_valid_v5_authority(
        authority_root
    )
    authority = subject._v5_validate_predata_authority(
        authority_root, authorization_path, go_path
    )
    assert authority["recovered_afterstate_commitment"] == (
        subject.V5_RECOVERED_AFTERSTATE_COMMITMENT
    )
    fixture_parent = tmp_path / "synthetic-core"
    fixture_parent.mkdir()
    recovered_root, runner, contract = build_recovered_fixture(fixture_parent)
    report = audit_fixture(recovered_root, runner, contract)
    assert report["raw_closure"]["ranges"] == contract.range_rows
    assert report["decoded_closure"]["site_rows"] == contract.site_count
    assert report["full_offline_raw_redecode"]["messages"] == contract.range_rows


def test_legacy_v5_unbound_matching_control_namespace_entry_fails_before_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "v5-namespace"
    authorization_path, _review_path, go_path, _ = build_valid_v5_authority(root)
    extra = root / "incidents" / ".TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_EVIL.json"
    extra.write_text("{}\n", encoding="utf-8")
    assert_v5_predata_failure_without_data(
        root, authorization_path, go_path, monkeypatch, "namespace inventory"
    )


@pytest.mark.parametrize(
    "relative",
    (
        "TRACK_A_RAW_PRODUCER_POSTRUN_READONLY_AUDIT_V2.json",
        "raw/TRACK_A_RAW_PRODUCER_POSTRUN_READONLY_AUDIT_V3.json",
        "decoded/TRACK_A_RAW_PRODUCER_POSTRUN_READONLY_AUDIT_V4.json",
        "TRACK_A_RAW_PRODUCER_POSTRUN_READONLY_AUDIT_V5.json",
    ),
)
def test_v2_v3_v4_v5_report_candidate_fails_before_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative: str
) -> None:
    root = tmp_path / "v5-report"
    authorization_path, _review_path, go_path, _ = build_valid_v5_authority(root)
    report = root / relative
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("{}\n", encoding="utf-8")
    assert_v5_predata_failure_without_data(
        root, authorization_path, go_path, monkeypatch, "postrun report candidate"
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "authorization_extra",
        "authorization_status",
        "review_crosslink",
        "go_status",
        "required_command",
        "authorization_path",
    ),
)
def test_v5_authority_schema_path_status_crosslink_or_command_tamper_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    root = tmp_path / f"authority-{mutation}"
    authorization_path, review_path, go_path, _ = build_valid_v5_authority(root)
    called_authorization = authorization_path
    if mutation.startswith("authorization_") and mutation != "authorization_path":
        authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
        if mutation == "authorization_extra":
            authorization["unknown"] = True
        else:
            authorization["status"] = "AUTHORIZED_WRONG"
        write_json(authorization_path, authorization)
    elif mutation == "review_crosslink":
        review = json.loads(review_path.read_text(encoding="utf-8"))
        review["incident_correction"] = dict(review["incident"])
        write_json(review_path, review)
        go = json.loads(go_path.read_text(encoding="utf-8"))
        go["independent_review"] = artifact(review_path, root)
        write_json(go_path, go)
    elif mutation == "go_status":
        go = json.loads(go_path.read_text(encoding="utf-8"))
        go["status"] = "GO_V5_WRONG"
        write_json(go_path, go)
    elif mutation == "required_command":
        authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
        authorization["required_command"] = ["python", "-m", "wrong.module"]
        write_json(authorization_path, authorization)
    else:
        called_authorization = authorization_path.with_name("authorization-alias.json")
        shutil.copy2(authorization_path, called_authorization)
    assert_v5_predata_failure_without_data(
        root,
        called_authorization,
        go_path,
        monkeypatch,
        "schema|header|cross-binding|required command|canonical",
    )


@pytest.mark.parametrize(
    "mutation", ("truncated_review", "incident_correction", "namespace", "commitment")
)
def test_v5_postaudit_rechecks_v4_truncation_namespace_and_afterstate_commitment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    root = tmp_path / f"poststate-{mutation}"
    authorization_path, _review_path, go_path, _ = build_valid_v5_authority(root)
    authority = subject._v5_validate_predata_authority(root, authorization_path, go_path)
    build_v5_postauthority_metadata_closure(root, authority, monkeypatch)
    if mutation == "truncated_review":
        with (root / subject.V4_REVIEW_RELATIVE).open("ab") as handle:
            handle.write(b" ")
    elif mutation == "incident_correction":
        with (root / subject.V5_TRANSPORT_CORRECTION_RELATIVE).open("ab") as handle:
            handle.write(b" ")
    elif mutation == "namespace":
        extra = root / "incidents" / ".TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_EVIL.json"
        extra.write_text("{}\n", encoding="utf-8")
    else:
        authority["recovered_afterstate_commitment"] = {
            **authority["recovered_afterstate_commitment"],
            "canonical_output_count": 7,
        }
    expected = {
        "truncated_review": "rejected malformed V4 independent review size mismatch",
        "incident_correction": "inventory-digest correction size mismatch",
        "namespace": "namespace inventory mismatch",
        "commitment": "afterstate/commitment/temp/report recheck mismatch",
    }[mutation]
    with pytest.raises(subject.AuditFailure, match=expected):
        subject._v5_validate_postauthority_afterstate(root, authority)


def test_inherited_core_auditor_is_stdout_only_and_writes_zero_files(tmp_path: Path) -> None:
    root, runner, contract = build_complete_fixture(tmp_path)
    before = snapshot(tmp_path)
    report = audit_fixture(root, runner, contract)
    assert report["artifact_type"] == "TRACK_A_RAW_PRODUCER_POSTRUN_READONLY_AUDIT_V7"
    assert report["audit_files_written"] == 0
    assert report["audit_network_requests"] == 0
    assert snapshot(tmp_path) == before


def test_v5_worker_is_importable_in_exactly_seven_spawn_processes(
    tmp_path: Path,
) -> None:
    del tmp_path
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
    assert subject.sha256_file(pilot) == (
        "575fe7876ab93e5470c43dc09c78895385925f5662cca75bd0d8c7f91e677407"
    )
    coordinate = json.loads(coordinate_path.read_text(encoding="utf-8"))
    runtime_identity = json.loads(
        (coordinate_path.parents[1] / "prereg" / "raw_launch_authorization_v1.json")
        .read_text(encoding="utf-8")
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
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(subject.DECODE_MAX_WORKERS)
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=subject.DECODE_MAX_WORKERS,
        mp_context=context,
        initializer=subject.v5_spawn_probe_barrier_initializer,
        initargs=(barrier,),
    ) as pool:
        futures = [
            pool.submit(subject.decode_offline_grib_process_worker, info, sites)
            for _index in range(subject.DECODE_MAX_WORKERS)
        ]
        decoded = [future.result(timeout=180) for future in futures]
    assert len({int(report["decoder_process_id"]) for report in decoded}) == 7
    assert {report["decoder_runtime_identity_sha256"] for report in decoded} == {
        subject.canonical_payload_sha256(runtime_identity)
    }
    for report in decoded:
        assert report["metadata_verified"] is True
        assert report["single_message_verified"] is True
        assert report["feature"] == "HPBL_surface"
        assert report["site_values"] == pytest.approx(expected, abs=1e-12, rel=0.0)


@pytest.mark.parametrize("failure", ("missing_review", "missing_go", "duplicate_go"))
def test_v5_authority_and_go_precede_any_parquet_or_data_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    root = tmp_path / f"authority-order-{failure}"
    authorization_path, review_path, go_path, _ = build_valid_v5_authority(root)
    if failure == "missing_review":
        review_path.unlink()
    elif failure == "missing_go":
        go_path.unlink()
    else:
        raw = go_path.read_text(encoding="utf-8")
        go_path.write_text('{"schema_version":5,' + raw[1:], encoding="utf-8")

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("data/parquet/raw/decoder route reached before V5 authority")

    for name in (
        "preflight_zero_target_parquet_schemas", "parquet_schema_columns",
        "_audit_provenance", "_audit_transaction_and_launch", "_audit_census",
        "_audit_raw_ranges", "_audit_request_events", "_audit_decoded",
        "_audit_full_offline_redecode", "decode_offline_grib_message",
        "decode_offline_grib_process_worker",
    ):
        monkeypatch.setattr(subject, name, forbidden)
    monkeypatch.setattr(pd, "read_parquet", forbidden)
    original_open = Path.open

    def guarded_open(path: Path, *args: object, **kwargs: object) -> object:
        if path.suffix.casefold() in {".parquet", ".grib2", ".csv"}:
            raise AssertionError("data file opened before V5 authority")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(subject.AuditFailure, match="absent|duplicate JSON key"):
        subject._v5_validate_predata_authority(root, authorization_path, go_path)


def test_v5_review_and_go_never_embed_full_progress_inventory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "compact-review-go"
    _authorization_path, review_path, go_path, _ = build_valid_v5_authority(root)
    for path in (review_path, go_path):
        payload = subject._v5_load_strict_json(path, label=path.name)
        subject._v5_assert_compact_control(payload, label=path.name)
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        assert '"recovered_afterstate"' not in serialized
        assert '"recovery_progress"' not in serialized
        assert '"inventory"' not in serialized
        assert path.stat().st_size <= 12_000


def test_legacy_v5_duplicate_json_key_rejection_is_not_bypassed_by_standard_last_wins_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "duplicate-key"
    authorization_path, _review_path, go_path, _ = build_valid_v5_authority(root)
    raw = authorization_path.read_text(encoding="utf-8")
    authorization_path.write_text('{"schema_version":5,' + raw[1:], encoding="utf-8")
    assert json.loads(authorization_path.read_text(encoding="utf-8"))["schema_version"] == 5
    assert_v5_predata_failure_without_data(
        root, authorization_path, go_path, monkeypatch, "duplicate JSON key"
    )


@pytest.mark.parametrize("role", ("primary_incident", "malformed_v4_review"))
def test_incident_or_v4_review_cannot_authorize_v4_or_v5_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: str
) -> None:
    root = tmp_path / f"nonauthority-{role}"
    authorization_path, _review_path, go_path, authorization = build_valid_v5_authority(root)
    incident_chain = subject._v5_validate_incident_chain(
        root, authorization["incident"], authorization["incident_correction"]
    )
    rejected = subject._v5_load_rejected_v4_review(
        root,
        subject._v5_load_and_validate_v4_authorization_base(
            root, authorization["v4_authorization_base"]
        ),
    )
    assert incident_chain["corrected_incident"]["authority_scope"][
        "live_postrun_auditor_authorized"
    ] is False
    assert incident_chain["corrected_incident"]["prohibitions"][
        "do_not_execute_v4_live_auditor"
    ] is True
    assert rejected["state"]["v4_auditor_execution_authorized"] is False
    nonauthority = root / (
        subject.V5_TRANSPORT_INCIDENT_RELATIVE
        if role == "primary_incident"
        else subject.V4_REVIEW_RELATIVE
    )
    assert_v5_predata_failure_without_data(
        root, nonauthority, go_path, monkeypatch, "canonical"
    )


def copy_v6_predata_controls(root: Path) -> None:
    """Copy only frozen JSON/control metadata needed by the V6 authority gate."""

    copy_v5_predata_controls(root)
    for relative in (
        subject.V5_AUTH_RELATIVE,
        subject.V5_REVIEW_RELATIVE,
        subject.V5_GO_RELATIVE,
        subject.V6_HISTORICAL_FALSE_REJECT_INCIDENT_RELATIVE,
        subject.RECOVERY_SEALER_ENTRYPOINT_INCIDENT_RELATIVE,
    ):
        source = subject.ROOT_DEFAULT / relative
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def frozen_v5_authority_base() -> dict[str, object]:
    return {
        "authorization": dict(subject.FROZEN_V5_CONTROL_IDENTITIES["authorization"]),
        "independent_review": dict(
            subject.FROZEN_V5_CONTROL_IDENTITIES["independent_review"]
        ),
        "independent_go": dict(subject.FROZEN_V5_CONTROL_IDENTITIES["independent_go"]),
        "audit_attempt_id": subject.FROZEN_V5_AUDIT_ATTEMPT_ID,
        "recovery_attempt_id": subject.FROZEN_V5_RECOVERY_ATTEMPT_ID,
        "authorization_status": "AUTHORIZED_PENDING_INDEPENDENT_REVIEW_AND_GO_V5",
        "review_status": "PASS_PENDING_INDEPENDENT_GO_V5",
        "go_status": "GO_V5_READ_ONLY_NETWORK_ZERO_FULL_OFFLINE_REPLAY",
        "recovered_afterstate_canonical_sha256": (
            subject.V4_RECOVERED_AFTERSTATE_CANONICAL_SHA256
        ),
    }


def synthetic_v6_test_evidence(
    created_utc: str, bound: dict[str, dict[str, object]]
) -> dict[str, object]:
    runs: dict[str, object] = {}
    for role, relative in {
        "v6_auditor_tests": "tests/test_noaa_gfs_multiseason_raw_postrun_v6.py",
        "v6_sealer_tests": "tests/test_seal_noaa_gfs_multiseason_postrun_audit_v6.py",
        "frozen_v5_auditor_tests": "tests/test_noaa_gfs_multiseason_raw_postrun_v5.py",
        "frozen_v5_sealer_tests": "tests/test_seal_noaa_gfs_multiseason_postrun_audit_v5.py",
    }.items():
        runs[role] = {
            "command": subject._v6_test_command(relative),
            "exit_code": 0,
            "summary": "1 passed",
            "stdout_sha256": "0" * 64,
            "stderr_sha256": (
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            ),
            "network_guard_installed": True,
            "cacheprovider_disabled": True,
        }
    return {
        "schema_version": 1,
        "artifact_type": "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_V6_TEST_EVIDENCE",
        "status": "PASS_FROZEN_V6_AUDITOR_AND_SEALER_TESTS",
        "created_utc": created_utc,
        "bound_identities": bound,
        "source_compile": {
            "method": "compile_exact_source_no_pyc",
            "result": "PASS",
            "files": [
                bound["v6_auditor"],
                bound["v6_auditor_test"],
                bound["v6_sealer"],
                bound["v6_sealer_test"],
            ],
        },
        "pytest_isolation": "ONE_TEST_FILE_PER_CLEAN_NETWORK_DENIED_SUBPROCESS",
        "test_runs": runs,
        "required_test_names": list(subject.V6_REQUIRED_TEST_NAMES),
        "required_test_names_present": True,
        "all_exit_codes_zero": True,
        "production_shape_regression": {
            "actual_root_historical_predata_validated": True,
            "compact_afterstate_commitment_validated": True,
            "decoded_reached": True,
            "direct_file_failure_incident_9e74_validated_predata": True,
            "full_afterstate_reconstructed_in_memory": True,
            "historical_current_role_distinction_validated": True,
            "offline_replay_reached": True,
            "original_v1_four_key_identity_passed": True,
            "raw_reached": True,
            "synthetic_fixture": True,
            "transaction_reached": True,
            "v5_authority_base_validated": True,
            "v5_false_reject_incident_validated": True,
            "v5_production_data_and_replay_core_ast_identical": True,
        },
        "real_seven_spawn_regression": {
            "start_method": "spawn",
            "entrypoint_module": "scripts.audit_noaa_gfs_multiseason_raw_postrun_v6",
            "worker_callable_module": "scripts.audit_noaa_gfs_multiseason_raw_postrun_v6",
            "max_workers": 7,
            "distinct_worker_pids": 7,
            "first_wave_synchronized": True,
            "network_requests": 0,
            "audit_files_written": 0,
            "labels_read": False,
            "arrays_2024_read": False,
            "arrays_2025_read": False,
        },
        "stdout_only_regression": True,
        "network_guard": {
            "parent_test_process_guarded": True,
            "sys_audit_hook_in_parent": True,
            "parent_socket_api_denied": True,
            "spawned_worker_network_route_static_absent": True,
            "spawned_worker_runtime_guard_installed": False,
            "external_packet_capture": False,
        },
        "network_requests": 0,
        "audit_files_written": 0,
        "labels_read": False,
        "arrays_2024_read": False,
        "arrays_2025_read": False,
        "models_fit": 0,
        "submission_csv_created": False,
    }


def build_valid_v6_authority(
    root: Path,
    *,
    frozen_sources: bool = False,
) -> tuple[Path, Path, Path, dict[str, object]]:
    copy_v6_predata_controls(root)
    incident_record = artifact(
        root / subject.V6_HISTORICAL_FALSE_REJECT_INCIDENT_RELATIVE, root
    )
    direct_record = artifact(
        root / subject.RECOVERY_SEALER_ENTRYPOINT_INCIDENT_RELATIVE, root
    )
    direct_payload = subject._v5_load_strict_json(
        root / subject.RECOVERY_SEALER_ENTRYPOINT_INCIDENT_RELATIVE,
        label="test immutable 9e74 incident",
    )
    historical_contract = subject._v6_validate_historical_9e74_payload(
        direct_payload, direct_record
    )
    executing_module = frozen_v6_subject if frozen_sources else subject
    source_identities = {
        **subject.FROZEN_V5_SOURCE_IDENTITIES,
        "v6_auditor": artifact(Path(executing_module.__file__).resolve()),
        "v6_auditor_test": artifact(executing_module.AUDITOR_TEST),
        "v6_sealer": artifact(executing_module.V6_SEALER),
        "v6_sealer_test": artifact(executing_module.V6_SEALER_TEST),
    }
    bound = {
        "v5_authorization": dict(subject.FROZEN_V5_CONTROL_IDENTITIES["authorization"]),
        "v5_independent_review": dict(
            subject.FROZEN_V5_CONTROL_IDENTITIES["independent_review"]
        ),
        "v5_independent_go": dict(subject.FROZEN_V5_CONTROL_IDENTITIES["independent_go"]),
        "v5_historical_identity_false_reject_incident": incident_record,
        "direct_file_failure_incident": direct_record,
        **source_identities,
    }
    created_utc = "2026-08-11T03:00:00.000000Z"
    attempt_id = "postrun_audit_v6__20260811T030000000000Z"
    authorization_path = root / subject.V6_AUTH_RELATIVE
    review_path = root / subject.V6_REVIEW_RELATIVE
    go_path = root / subject.V6_GO_RELATIVE
    policy = {
        "max_spawn_processes": 7,
        "network_requests_allowed": 0,
        "audit_files_written_allowed": 0,
        "labels_read_allowed": False,
        "arrays_2024_read_allowed": False,
        "arrays_2025_read_allowed": False,
        "models_fit_allowed": 0,
        "submission_csv_allowed": False,
        "full_offline_redecode_required": True,
        "stdout_only": True,
    }
    authorization = {
        "schema_version": 6,
        "artifact_type": "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDIT_AUTHORIZATION_V6",
        "status": "AUTHORIZED_PENDING_INDEPENDENT_REVIEW_AND_GO_V6",
        "created_utc": created_utc,
        "audit_attempt_id": attempt_id,
        "incident": incident_record,
        "v5_authority_base": frozen_v5_authority_base(),
        "historical_failed_head_identity_contract": historical_contract,
        **source_identities,
        "recovery_attempt_id": subject.FROZEN_V5_RECOVERY_ATTEMPT_ID,
        "recovered_afterstate_commitment": dict(
            subject.V5_RECOVERED_AFTERSTATE_COMMITMENT
        ),
        "preaudit_zero_mutation_snapshot": {
            "incident_postfailure_state_canonical_sha256": (
                "866d6564abb4d00cb1cb2961677cf767b9934ba44829cee3fec0ff46e9961690"
            ),
            "recovered_afterstate_commitment_canonical_sha256": (
                subject.V5_RECOVERED_AFTERSTATE_COMMITMENT_CANONICAL_SHA256
            ),
            "active_locks": {
                "raw": {"path": "raw/RAW_LAUNCH_ACTIVE.lock", "present": False},
                "decoded": {
                    "path": "decoded/DECODE_RECOVERY_ACTIVE.lock",
                    "present": False,
                },
            },
            "v6_control_temporary_files": {
                "checked_destination_relative_paths": list(
                    subject._v6_control_destination_relatives()
                ),
                "matching_temporary_files": [],
            },
            "postrun_audit_output_files_present": 0,
            "network_requests": 0,
            "audit_files_written": 0,
            "labels_read": False,
            "arrays_2024_read": False,
            "arrays_2025_read": False,
            "models_fit": 0,
            "submission_csv_created": False,
        },
        "runtime_identity_sha256": subject.V4_RUNTIME_IDENTITY_SHA256,
        "test_evidence": synthetic_v6_test_evidence(created_utc, bound),
        "required_command": subject._v6_required_command(
            root, authorization_path, go_path
        ),
        **policy,
        "independent_review_required": True,
        "independent_go_required": True,
    }
    assert set(authorization) == subject.V6_AUTH_KEYS
    write_json(authorization_path, authorization)
    common = {
        field: authorization[field]
        for field in (
            "incident", "v5_authority_base",
            "historical_failed_head_identity_contract",
            *subject.FROZEN_V5_SOURCE_IDENTITIES.keys(),
            "v6_auditor", "v6_auditor_test", "v6_sealer", "v6_sealer_test",
            "recovered_afterstate_commitment",
        )
    }
    review = {
        "schema_version": 6,
        "artifact_type": "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_REVIEW_V6",
        "status": "PASS_PENDING_INDEPENDENT_GO_V6",
        "verdict": "GO_RECOMMENDED",
        "created_utc": "2026-08-11T03:01:00.000000Z",
        "audit_attempt_id": attempt_id,
        "authorization": artifact(authorization_path, root),
        **common,
        "test_evidence_recheck": {
            "authorization_test_evidence_canonical_sha256": (
                subject.canonical_payload_sha256(authorization["test_evidence"])
            ),
            "bound_identities_rehashed": True,
            "source_compile_exact": True,
            "required_test_names_exact": True,
            "all_exit_codes_zero": True,
            "production_shape_regression_exact": True,
            "real_seven_spawn_regression_exact": True,
            "stdout_only_zero_write_exact": True,
            "network_zero_target_free_exact": True,
        },
        "independent_checks": {key: True for key in subject.V6_REVIEW_CHECKS},
        **policy,
        "auditor_execution_started": False,
        "independent_go_required": True,
    }
    assert set(review) == subject.V6_REVIEW_KEYS
    write_json(review_path, review)
    go = {
        "schema_version": 6,
        "artifact_type": "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_GO_V6",
        "status": "GO_V6_READ_ONLY_NETWORK_ZERO_FULL_OFFLINE_REPLAY",
        "created_utc": "2026-08-11T03:02:00.000000Z",
        "audit_attempt_id": attempt_id,
        "authorization": artifact(authorization_path, root),
        "independent_review": artifact(review_path, root),
        **common,
        "recovery_attempt_id": subject.FROZEN_V5_RECOVERY_ATTEMPT_ID,
        "required_command": authorization["required_command"],
        **policy,
        "recovery_rerun_authorized": False,
    }
    assert set(go) == subject.V6_GO_KEYS
    write_json(go_path, go)
    return authorization_path, review_path, go_path, authorization


def copy_v7_predata_controls(root: Path) -> None:
    """Copy only frozen JSON/control metadata needed by the V7 gate."""

    copy_v6_predata_controls(root)
    relative = subject.V7_TIMESTAMP_FALSE_REJECT_INCIDENT_RELATIVE
    source = subject.ROOT_DEFAULT / relative
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def synthetic_v7_test_evidence(
    created_utc: str, bound: dict[str, dict[str, object]]
) -> dict[str, object]:
    runs: dict[str, object] = {}
    for role, relative in {
        "v7_auditor_tests": "tests/test_noaa_gfs_multiseason_raw_postrun_v7.py",
        "v7_sealer_tests": "tests/test_seal_noaa_gfs_multiseason_postrun_audit_v7.py",
        "frozen_v6_auditor_tests": "tests/test_noaa_gfs_multiseason_raw_postrun_v6.py",
        "frozen_v6_sealer_tests": "tests/test_seal_noaa_gfs_multiseason_postrun_audit_v6.py",
    }.items():
        runs[role] = {
            "command": subject._v7_test_command(relative),
            "exit_code": 0,
            "summary": "1 passed",
            "stdout_sha256": "0" * 64,
            "stderr_sha256": (
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            ),
            "network_guard_installed": True,
            "cacheprovider_disabled": True,
        }
    return {
        "schema_version": 1,
        "artifact_type": "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_V7_TEST_EVIDENCE",
        "status": "PASS_FROZEN_V7_AUDITOR_AND_SEALER_TESTS",
        "created_utc": created_utc,
        "bound_identities": bound,
        "source_compile": {
            "method": "compile_exact_source_no_pyc",
            "result": "PASS",
            "files": [
                bound["v7_auditor"], bound["v7_auditor_test"],
                bound["v7_sealer"], bound["v7_sealer_test"],
            ],
        },
        "pytest_isolation": "ONE_TEST_FILE_PER_CLEAN_NETWORK_DENIED_SUBPROCESS",
        "test_runs": runs,
        "required_test_names": list(subject.V7_REQUIRED_TEST_NAMES),
        "required_test_names_present": True,
        "all_exit_codes_zero": True,
        "production_shape_regression": {
            "actual_root_historical_predata_validated": True,
            "actual_root_collect_build_self_validator_validated": True,
            "actual_timestamp_incident_validated_predata": True,
            "compact_afterstate_commitment_validated": True,
            "decoded_reached": True,
            "direct_file_failure_incident_9e74_validated_predata": True,
            "frozen_v6_timestamp_false_reject_reproduced": True,
            "full_afterstate_reconstructed_in_memory": True,
            "historical_current_role_distinction_validated": True,
            "offline_replay_reached": True,
            "original_v1_four_key_identity_passed": True,
            "raw_reached": True,
            "rfc3339_100ns_parser_zero_through_seven_exact": True,
            "synthetic_fixture": True,
            "transaction_reached": True,
            "v5_authority_base_validated": True,
            "v5_false_reject_incident_validated": True,
            "v6_auth_sealer_timestamp_false_reject_incident_validated": True,
            "v6_and_v5_production_data_and_replay_core_ast_identical": True,
        },
        "real_seven_spawn_regression": {
            "start_method": "spawn",
            "entrypoint_module": "scripts.audit_noaa_gfs_multiseason_raw_postrun_v7",
            "worker_callable_module": "scripts.audit_noaa_gfs_multiseason_raw_postrun_v7",
            "max_workers": 7,
            "distinct_worker_pids": 7,
            "first_wave_synchronized": True,
            "network_requests": 0,
            "audit_files_written": 0,
            "labels_read": False,
            "arrays_2024_read": False,
            "arrays_2025_read": False,
        },
        "stdout_only_regression": True,
        "network_guard": {
            "parent_test_process_guarded": True,
            "sys_audit_hook_in_parent": True,
            "parent_socket_api_denied": True,
            "spawned_worker_network_route_static_absent": True,
            "spawned_worker_runtime_guard_installed": False,
            "external_packet_capture": False,
        },
        "network_requests": 0,
        "audit_files_written": 0,
        "labels_read": False,
        "arrays_2024_read": False,
        "arrays_2025_read": False,
        "models_fit": 0,
        "submission_csv_created": False,
    }


def build_valid_v7_authority(
    root: Path,
) -> tuple[Path, Path, Path, dict[str, object]]:
    copy_v7_predata_controls(root)
    timestamp_record = artifact(
        root / subject.V7_TIMESTAMP_FALSE_REJECT_INCIDENT_RELATIVE, root
    )
    false_reject_record = artifact(
        root / subject.V6_HISTORICAL_FALSE_REJECT_INCIDENT_RELATIVE, root
    )
    direct_record = artifact(
        root / subject.RECOVERY_SEALER_ENTRYPOINT_INCIDENT_RELATIVE, root
    )
    direct_payload = subject._v5_load_strict_json(
        root / subject.RECOVERY_SEALER_ENTRYPOINT_INCIDENT_RELATIVE,
        label="test immutable 9e74 incident",
    )
    historical_contract = subject._v6_validate_historical_9e74_payload(
        direct_payload, direct_record
    )
    source_identities = {
        **subject.FROZEN_V6_SOURCE_IDENTITIES,
        "v7_auditor": artifact(Path(subject.__file__).resolve()),
        "v7_auditor_test": artifact(subject.AUDITOR_TEST),
        "v7_sealer": artifact(subject.V7_SEALER),
        "v7_sealer_test": artifact(subject.V7_SEALER_TEST),
    }
    bound = {
        "v5_authorization": dict(subject.FROZEN_V5_CONTROL_IDENTITIES["authorization"]),
        "v5_independent_review": dict(
            subject.FROZEN_V5_CONTROL_IDENTITIES["independent_review"]
        ),
        "v5_independent_go": dict(subject.FROZEN_V5_CONTROL_IDENTITIES["independent_go"]),
        "v5_historical_identity_false_reject_incident": false_reject_record,
        "direct_file_failure_incident": direct_record,
        "v6_auth_sealer_timestamp_precision_false_reject_incident": timestamp_record,
        **source_identities,
    }
    created_utc = "2026-08-11T05:00:00.000000Z"
    attempt_id = "postrun_audit_v7__20260811T050000000000Z"
    authorization_path = root / subject.V7_AUTH_RELATIVE
    review_path = root / subject.V7_REVIEW_RELATIVE
    go_path = root / subject.V7_GO_RELATIVE
    policy = {
        "max_spawn_processes": 7,
        "network_requests_allowed": 0,
        "audit_files_written_allowed": 0,
        "labels_read_allowed": False,
        "arrays_2024_read_allowed": False,
        "arrays_2025_read_allowed": False,
        "models_fit_allowed": 0,
        "submission_csv_allowed": False,
        "full_offline_redecode_required": True,
        "stdout_only": True,
    }
    authorization = {
        "schema_version": 7,
        "artifact_type": "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDIT_AUTHORIZATION_V7",
        "status": "AUTHORIZED_PENDING_INDEPENDENT_REVIEW_AND_GO_V7",
        "created_utc": created_utc,
        "audit_attempt_id": attempt_id,
        "incident": timestamp_record,
        "v5_authority_base": frozen_v5_authority_base(),
        "historical_failed_head_identity_contract": historical_contract,
        **source_identities,
        "recovery_attempt_id": subject.FROZEN_V5_RECOVERY_ATTEMPT_ID,
        "recovered_afterstate_commitment": dict(
            subject.V5_RECOVERED_AFTERSTATE_COMMITMENT
        ),
        "preaudit_zero_mutation_snapshot": {
            "incident_postfailure_state_canonical_sha256": (
                subject.V7_TIMESTAMP_INCIDENT_POSTFAILURE_CANONICAL_SHA256
            ),
            "recovered_afterstate_commitment_canonical_sha256": (
                subject.V5_RECOVERED_AFTERSTATE_COMMITMENT_CANONICAL_SHA256
            ),
            "active_locks": {
                "raw": {"path": "raw/RAW_LAUNCH_ACTIVE.lock", "present": False},
                "decoded": {
                    "path": "decoded/DECODE_RECOVERY_ACTIVE.lock",
                    "present": False,
                },
            },
            "v7_control_temporary_files": {
                "checked_destination_relative_paths": list(
                    subject._v7_control_destination_relatives()
                ),
                "matching_temporary_files": [],
            },
            "postrun_audit_output_files_present": 0,
            "network_requests": 0,
            "audit_files_written": 0,
            "labels_read": False,
            "arrays_2024_read": False,
            "arrays_2025_read": False,
            "models_fit": 0,
            "submission_csv_created": False,
        },
        "runtime_identity_sha256": subject.V4_RUNTIME_IDENTITY_SHA256,
        "test_evidence": synthetic_v7_test_evidence(created_utc, bound),
        "required_command": subject._v7_required_command(
            root, authorization_path, go_path
        ),
        **policy,
        "independent_review_required": True,
        "independent_go_required": True,
    }
    assert set(authorization) == subject.V7_AUTH_KEYS
    write_json(authorization_path, authorization)
    common = {
        field: authorization[field]
        for field in (
            "incident", "v5_authority_base",
            "historical_failed_head_identity_contract",
            *subject.FROZEN_V6_SOURCE_IDENTITIES.keys(),
            "v7_auditor", "v7_auditor_test", "v7_sealer", "v7_sealer_test",
            "recovered_afterstate_commitment",
        )
    }
    review = {
        "schema_version": 7,
        "artifact_type": "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_REVIEW_V7",
        "status": "PASS_PENDING_INDEPENDENT_GO_V7",
        "verdict": "GO_RECOMMENDED",
        "created_utc": "2026-08-11T05:01:00.000000Z",
        "audit_attempt_id": attempt_id,
        "authorization": artifact(authorization_path, root),
        **common,
        "test_evidence_recheck": {
            "authorization_test_evidence_canonical_sha256": (
                subject.canonical_payload_sha256(authorization["test_evidence"])
            ),
            "bound_identities_rehashed": True,
            "source_compile_exact": True,
            "required_test_names_exact": True,
            "all_exit_codes_zero": True,
            "production_shape_regression_exact": True,
            "real_seven_spawn_regression_exact": True,
            "stdout_only_zero_write_exact": True,
            "network_zero_target_free_exact": True,
        },
        "independent_checks": {key: True for key in subject.V7_REVIEW_CHECKS},
        **policy,
        "auditor_execution_started": False,
        "independent_go_required": True,
    }
    assert set(review) == subject.V7_REVIEW_KEYS
    write_json(review_path, review)
    go = {
        "schema_version": 7,
        "artifact_type": "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_GO_V7",
        "status": "GO_V7_READ_ONLY_NETWORK_ZERO_FULL_OFFLINE_REPLAY",
        "created_utc": "2026-08-11T05:02:00.000000Z",
        "audit_attempt_id": attempt_id,
        "authorization": artifact(authorization_path, root),
        "independent_review": artifact(review_path, root),
        **common,
        "recovery_attempt_id": subject.FROZEN_V5_RECOVERY_ATTEMPT_ID,
        "required_command": authorization["required_command"],
        **policy,
        "recovery_rerun_authorized": False,
    }
    assert set(go) == subject.V7_GO_KEYS
    write_json(go_path, go)
    return authorization_path, review_path, go_path, authorization


def test_real_immutable_v5_authority_base_passes(tmp_path: Path) -> None:
    root = tmp_path / "v6-base-positive"
    copy_v6_predata_controls(root)
    result = subject._v6_load_and_validate_v5_authority_base(
        root, frozen_v5_authority_base()
    )
    assert result["base"] == frozen_v5_authority_base()
    assert result["authorization_created"] < result["review_created"] < result["go_created"]


@pytest.mark.parametrize(
    "mutation",
    ("review_equal_auth", "review_backdated", "go_equal_review", "go_backdated", "auth_equal_incident"),
)
def test_v6_strict_authority_chronology_fails(
    tmp_path: Path, mutation: str
) -> None:
    root = tmp_path / f"v6-chronology-{mutation}"
    authorization_path, review_path, go_path, _ = build_valid_v6_authority(root)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    go = json.loads(go_path.read_text(encoding="utf-8"))
    if mutation == "review_equal_auth":
        review["created_utc"] = authorization["created_utc"]
    elif mutation == "review_backdated":
        review["created_utc"] = "2026-08-11T02:59:59.999999Z"
    elif mutation == "go_equal_review":
        go["created_utc"] = review["created_utc"]
    elif mutation == "go_backdated":
        go["created_utc"] = "2026-08-11T03:00:30.000000Z"
    else:
        incident = subject._v6_validate_v5_false_reject_incident(
            root, authorization["incident"]
        )
        authorization["created_utc"] = incident["payload"]["created_utc"]
        authorization["audit_attempt_id"] = (
            "postrun_audit_v6__"
            + authorization["created_utc"].translate(str.maketrans("", "", "-:."))
        )
        review["audit_attempt_id"] = authorization["audit_attempt_id"]
        go["audit_attempt_id"] = authorization["audit_attempt_id"]
    write_json(authorization_path, authorization)
    review["authorization"] = artifact(authorization_path, root)
    write_json(review_path, review)
    go["authorization"] = artifact(authorization_path, root)
    go["independent_review"] = artifact(review_path, root)
    write_json(go_path, go)
    with pytest.raises(subject.AuditFailure, match="chronology|header"):
        subject._v6_validate_predata_authority(root, authorization_path, go_path)


def test_v6_real_9e74_historical_validator_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = subject.ROOT_DEFAULT
    original_open = Path.open

    def metadata_only_open(path: Path, *args: object, **kwargs: object) -> object:
        if path.suffix.casefold() in {".parquet", ".grib2", ".csv"}:
            raise AssertionError("actual-root data opened by historical predata validator")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", metadata_only_open)
    false_reject_record = artifact(
        root / subject.V6_HISTORICAL_FALSE_REJECT_INCIDENT_RELATIVE, root
    )
    false_reject = subject._v6_validate_v5_false_reject_incident(
        root, false_reject_record
    )
    direct_path = root / subject.RECOVERY_SEALER_ENTRYPOINT_INCIDENT_RELATIVE
    direct_record = artifact(direct_path, root)
    direct_payload = subject._v5_load_strict_json(
        direct_path, label="actual-root immutable 9e74"
    )
    historical = subject._v6_validate_historical_9e74_payload(
        direct_payload, direct_record
    )
    base = subject._v6_load_and_validate_v5_authority_base(
        root, frozen_v5_authority_base()
    )
    assert false_reject["identity"] == false_reject_record
    assert base["base"] == frozen_v5_authority_base()
    assert historical["historical_incident_contract"] == {
        "failed_head_identities": {
            "canonical_size_bytes": 1672,
            "canonical_sha256": subject.V6_HISTORICAL_FAILED_HEADS_CANONICAL_SHA256,
        },
        "verified_zero_state_after_failure": {
            "canonical_size_bytes": 513,
            "canonical_sha256": subject.V6_HISTORICAL_ZERO_STATE_CANONICAL_SHA256,
        },
        "absent_control_paths": {
            "canonical_size_bytes": 288,
            "canonical_sha256": subject.V6_HISTORICAL_ABSENT_CONTROLS_CANONICAL_SHA256,
        },
    }


def test_v6_synthetic_authority_chain_passes(tmp_path: Path) -> None:
    root = tmp_path / "v6-authority-positive"
    authorization_path, review_path, go_path, _ = build_valid_v6_authority(root)
    before = snapshot(root)
    authority = subject._v6_validate_predata_authority(root, authorization_path, go_path)
    assert authority["authorization_identity"] == artifact(authorization_path, root)
    assert authority["review_identity"] == artifact(review_path, root)
    assert authority["historical_contract"] == (
        authority["authorization"]["historical_failed_head_identity_contract"]
    )
    assert snapshot(root) == before


def immutable_9e74_payload() -> tuple[dict[str, object], dict[str, object]]:
    path = subject.ROOT_DEFAULT / subject.RECOVERY_SEALER_ENTRYPOINT_INCIDENT_RELATIVE
    return (
        subject._v5_load_strict_json(path, label="test immutable 9e74 incident"),
        artifact(path, subject.ROOT_DEFAULT),
    )


def test_v5_historical_identity_false_reject_incident_is_exact() -> None:
    root = subject.ROOT_DEFAULT
    record = artifact(
        root / subject.V6_HISTORICAL_FALSE_REJECT_INCIDENT_RELATIVE, root
    )
    validated = subject._v6_validate_v5_false_reject_incident(root, record)
    assert validated["identity"] == {
        "path": subject.V6_HISTORICAL_FALSE_REJECT_INCIDENT_RELATIVE,
        "size_bytes": 10523,
        "sha256": "09b92677c719631c512a4b227653a5e5b2def9dd4b64eca787bd7c323ba1d61d",
    }
    assert validated["payload"]["root_cause"]["first_reject"] == (
        "failed_head_identities.recovery_runner"
    )
    assert validated["payload"]["root_cause"]["latent_second_reject"] == (
        "verified_zero_state_after_failure.absent_control_paths"
    )


def test_direct_file_failure_incident_9e74_historical_state_is_exact() -> None:
    payload, record = immutable_9e74_payload()
    result = subject._v6_validate_historical_9e74_payload(payload, record)
    assert payload["failed_head_identities"] == subject.V6_HISTORICAL_FAILED_HEAD_IDENTITIES
    assert payload["verified_zero_state_after_failure"] == subject.V6_HISTORICAL_ZERO_STATE
    assert result == subject._v6_expected_historical_contract(record)
    assert payload["failed_head_identities"]["recovery_runner"]["sha256"] not in {
        subject.RECOVERY_REJECTED_TRANSIENT_V1_RUNNER_IDENTITY["sha256"],
        subject.RECOVERY_V1_CHAIN_IDENTITIES["recovery_runner"]["sha256"],
    }


def test_frozen_v5_reproduces_exact_recovery_runner_false_reject() -> None:
    payload, record = immutable_9e74_payload()
    with pytest.raises(
        frozen_v5_subject.AuditFailure,
        match="sealer-entrypoint historical identity invalid: recovery_runner",
    ):
        frozen_v5_subject._validate_recovery_sealer_entrypoint_incident(
            payload,
            record,
            frozen_v5_subject.ROOT_DEFAULT,
            {},
            frozen_v5_subject.PRODUCTION_CONTRACT,
        )


@pytest.mark.parametrize("role", tuple(subject.V6_HISTORICAL_FAILED_HEAD_IDENTITIES))
@pytest.mark.parametrize("field", ("path", "size_bytes", "sha256"))
def test_historical_failed_head_path_size_or_sha_tamper_fails(
    role: str, field: str
) -> None:
    payload, record = immutable_9e74_payload()
    payload = json.loads(json.dumps(payload))
    payload["failed_head_identities"][role][field] = {
        "path": "scripts/current-role-alias.py",
        "size_bytes": 1,
        "sha256": "0" * 64,
    }[field]
    with pytest.raises(subject.AuditFailure, match="historical failed-head"):
        subject._v6_validate_historical_9e74_payload(payload, record)


@pytest.mark.parametrize("mutation", ("missing", "extra", "order", "v2_substitution"))
def test_historical_v1_absent_control_path_tamper_fails(mutation: str) -> None:
    payload, record = immutable_9e74_payload()
    payload = json.loads(json.dumps(payload))
    paths = payload["verified_zero_state_after_failure"]["absent_control_paths"]
    if mutation == "missing":
        paths.pop()
    elif mutation == "extra":
        paths.append("prereg/unbound.json")
    elif mutation == "order":
        paths[0], paths[1] = paths[1], paths[0]
    else:
        payload["verified_zero_state_after_failure"]["absent_control_paths"] = list(
            subject.RECOVERY_V2_CONTROL_RELATIVES
        )
    with pytest.raises(subject.AuditFailure, match="zero-state|absent-control"):
        subject._v6_validate_historical_9e74_payload(payload, record)


@pytest.mark.parametrize(
    "kind",
    ("runner_v2", "runner_transient_v1", "runner_final_v1", "controls_v2"),
)
def test_current_v2_role_or_control_substitution_fails(kind: str) -> None:
    payload, record = immutable_9e74_payload()
    payload = json.loads(json.dumps(payload))
    if kind == "controls_v2":
        payload["verified_zero_state_after_failure"]["absent_control_paths"] = list(
            subject.RECOVERY_V2_CONTROL_RELATIVES
        )
    else:
        replacement = {
            "runner_v2": {
                "path": str(subject.RECOVERY_RUNNER.resolve()),
                "size_bytes": 1,
                "sha256": "0" * 64,
            },
            "runner_transient_v1": subject.RECOVERY_REJECTED_TRANSIENT_V1_RUNNER_IDENTITY,
            "runner_final_v1": subject.RECOVERY_V1_CHAIN_IDENTITIES["recovery_runner"],
        }[kind]
        payload["failed_head_identities"]["recovery_runner"] = dict(replacement)
    with pytest.raises(subject.AuditFailure, match="historical|zero-state|absent-control"):
        subject._v6_validate_historical_9e74_payload(payload, record)


def test_historical_predata_gate_precedes_schema_parquet_data_and_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "v6-historical-before-data"
    authorization_path, _review_path, go_path, _ = build_valid_v6_authority(
        root, frozen_sources=True
    )
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    authorization["historical_failed_head_identity_contract"]["validation_mode"] = "WRONG"
    write_json(authorization_path, authorization)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("schema/parquet/data/replay route reached before V6 history")

    for name in (
        "preflight_zero_target_parquet_schemas", "parquet_schema_columns",
        "_audit_provenance", "_audit_transaction_and_launch", "_audit_census",
        "_audit_raw_ranges", "_audit_request_events", "_audit_decoded",
        "_audit_full_offline_redecode", "decode_offline_grib_message",
        "decode_offline_grib_process_worker",
    ):
        monkeypatch.setattr(frozen_v6_subject, name, forbidden)
    monkeypatch.setattr(pd, "read_parquet", forbidden)
    monkeypatch.setattr(frozen_v6_subject, "_V6_CANONICAL_MAIN_ACTIVE", True)
    runner = tmp_path / "runner.py"
    runner.write_text("# metadata-only ordering sentinel\n", encoding="utf-8")
    with pytest.raises(
        frozen_v6_subject.AuditFailure,
        match="historical failed-head identity contract",
    ):
        frozen_v6_subject.audit(
            root,
            contract=frozen_v6_subject.PRODUCTION_CONTRACT,
            runner_path=runner,
            authorization_path=authorization_path,
            independent_go_path=go_path,
        )


def test_late_reuse_revalidates_historical_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "v6-late-reuse"
    authorization_path, _review_path, go_path, _ = build_valid_v6_authority(
        root, frozen_sources=True
    )
    authority = frozen_v6_subject._v6_validate_predata_authority(
        root, authorization_path, go_path
    )
    build_v5_postauthority_metadata_closure(
        root, {"v4_base": authority["v5_base"]["v4_base"]}, monkeypatch
    )
    monkeypatch.setattr(frozen_v6_subject, "identity", subject.identity)
    runner = tmp_path / "runner.py"
    runner.write_text("# V6 postauthority call-count sentinel\n", encoding="utf-8")
    before = snapshot(root)
    original_poststate = frozen_v6_subject._v6_validate_postauthority_afterstate
    poststate_calls: list[int] = []

    def counted_poststate(
        call_root: Path, call_authority: dict[str, object]
    ) -> dict[str, object]:
        poststate_calls.append(len(poststate_calls) + 1)
        return original_poststate(call_root, call_authority)

    monkeypatch.setattr(
        frozen_v6_subject, "_v6_validate_postauthority_afterstate", counted_poststate
    )
    monkeypatch.setattr(
        frozen_v6_subject, "_v6_validate_predata_authority", lambda *_args: authority
    )
    monkeypatch.setattr(frozen_v6_subject, "_V6_CANONICAL_MAIN_ACTIVE", True)
    monkeypatch.setattr(
        frozen_v6_subject,
        "require_no_symlink_chain",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        frozen_v6_subject, "_audit_active_lock", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        frozen_v6_subject,
        "_audit_recovery_control_temporary_files_absent",
        lambda *_args: {"matching_temporary_files": 0},
    )
    monkeypatch.setattr(
        frozen_v6_subject,
        "preflight_zero_target_parquet_schemas",
        lambda *_args: {},
    )
    monkeypatch.setattr(
        frozen_v6_subject,
        "_audit_provenance",
        lambda *_args: {
            "field_census_path": tmp_path / "unused-census.json",
            "coordinate": {},
            "runtime_identity_sha256": frozen_v6_subject.V4_RUNTIME_IDENTITY_SHA256,
            "authorization_identity": {}, "go_identity": {}, "preflight_identity": {},
            "preflight_amendment_identity": {}, "independent_prelaunch_identity": {},
            "independent_prelaunch_identity_base": {}, "runner_identity": {},
            "runner_test_identity": {}, "embedded_identity_rehashes": {},
            "external_identity_unique_file_count": 0,
            "external_identity_unique_file_bytes": 0,
        },
    )
    monkeypatch.setattr(
        frozen_v6_subject,
        "_audit_transaction_and_launch",
        lambda *_args: {
            "outputs": {"raw_parquet": None, "raw_csv": None},
            "attempt_id": "synthetic", "complete_lock_identity": {},
            "launch_history_lock_count": 1, "plan_identity": {}, "commit_identity": {},
            "transaction_inventory_recursively_closed": True,
            "planned_staged_files_present": 0,
        },
    )
    monkeypatch.setattr(
        frozen_v6_subject, "_audit_census", lambda *_args: (None, [])
    )
    monkeypatch.setattr(
        frozen_v6_subject, "_audit_raw_ranges",
        lambda *_args: {"raw_info": [], "raw_rows": 0, "raw_bytes": 0},
    )
    monkeypatch.setattr(
        frozen_v6_subject, "_audit_request_events",
        lambda *_args: {
            "starts": 0, "completions": 0, "errors": 0,
            "indeterminate_starts": 0, "max_global_attempt_number": 0,
        },
    )
    monkeypatch.setattr(frozen_v6_subject, "_audit_decoded", lambda *_args: {})
    monkeypatch.setattr(
        frozen_v6_subject,
        "_audit_full_offline_redecode",
        lambda *_args, **_kwargs: {},
    )

    def mutate_in_memory_between_poststate_calls(*_args: object) -> dict[str, object]:
        authority["authorization"]["historical_failed_head_identity_contract"][
            "validation_mode"
        ] = "DRIFTED_BETWEEN_START_AND_END"
        return {
            "external_identity_unique_file_count": 0,
            "external_identity_unique_file_bytes": 0,
        }

    monkeypatch.setattr(
        frozen_v6_subject,
        "_audit_canonical_metadata",
        mutate_in_memory_between_poststate_calls,
    )
    with pytest.raises(
        frozen_v6_subject.AuditFailure,
        match="historical failed-head identity contract mismatch",
    ):
        frozen_v6_subject.audit(
            root,
            contract=frozen_v6_subject.PRODUCTION_CONTRACT,
            runner_path=runner,
            authorization_path=authorization_path,
            independent_go_path=go_path,
        )
    assert poststate_calls == [1, 2]
    assert snapshot(root) == before
    audit_node = next(
        node
        for node in ast.parse(
            Path(frozen_v6_subject.__file__).read_text(encoding="utf-8")
        ).body
        if isinstance(node, ast.FunctionDef) and node.name == "audit"
    )
    called_names = [
        node.func.id for node in ast.walk(audit_node)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert called_names.count("_v6_validate_predata_authority") == 1
    assert called_names.count("_v6_validate_postauthority_afterstate") == 2


def test_v5_production_data_and_replay_core_ast_is_identical() -> None:
    v5_tree = ast.parse(subject.FROZEN_V5_AUDITOR.read_text(encoding="utf-8"))
    v6_tree = ast.parse(Path(subject.__file__).read_text(encoding="utf-8"))
    v5_functions = {
        node.name: node for node in v5_tree.body if isinstance(node, ast.FunctionDef)
    }
    v6_functions = {
        node.name: node for node in v6_tree.body if isinstance(node, ast.FunctionDef)
    }
    ordered = [
        node.name for node in v5_tree.body if isinstance(node, ast.FunctionDef)
    ]
    first = ordered.index("recovery_progress_counts")
    last = ordered.index("_audit_full_offline_redecode")
    core_names = ordered[first : last + 1]
    assert len(core_names) == 39
    assert all(name in v6_functions for name in core_names)
    assert {
        name: ast.dump(v5_functions[name], include_attributes=False)
        for name in core_names
    } == {
        name: ast.dump(v6_functions[name], include_attributes=False)
        for name in core_names
    }


@pytest.mark.parametrize(
    "mutation",
    (
        "authorization_path", "authorization_size", "authorization_sha",
        "authorization_status", "review_status", "go_status", "afterstate_digest",
        "schema_extra",
    ),
)
def test_v5_authority_base_path_size_sha_schema_status_or_digest_tamper_fails(
    tmp_path: Path, mutation: str
) -> None:
    root = tmp_path / f"v5-base-tamper-{mutation}"
    copy_v6_predata_controls(root)
    base = frozen_v5_authority_base()
    if mutation.startswith("authorization_") and mutation != "authorization_status":
        field = mutation.removeprefix("authorization_")
        key = {"path": "path", "size": "size_bytes", "sha": "sha256"}[field]
        base["authorization"][key] = {
            "path": "prereg/wrong.json",
            "size_bytes": 1,
            "sha256": "0" * 64,
        }[key]
    elif mutation == "schema_extra":
        base["unknown"] = True
    elif mutation == "afterstate_digest":
        base["recovered_afterstate_canonical_sha256"] = "0" * 64
    else:
        base[mutation] = "WRONG"
    with pytest.raises(subject.AuditFailure, match="authority-base|frozen V5"):
        subject._v6_load_and_validate_v5_authority_base(root, base)


def write_rebound_v6_chain(
    root: Path,
    authorization_path: Path,
    review_path: Path,
    go_path: Path,
    authorization: dict[str, object],
    review: dict[str, object],
    go: dict[str, object],
) -> None:
    write_json(authorization_path, authorization)
    review["authorization"] = artifact(authorization_path, root)
    write_json(review_path, review)
    go["authorization"] = artifact(authorization_path, root)
    go["independent_review"] = artifact(review_path, root)
    write_json(go_path, go)


def test_compact_recovered_afterstate_commitment_matches_v5_authority(
    tmp_path: Path,
) -> None:
    root = tmp_path / "v6-commitment-positive"
    authorization_path, _review_path, go_path, _ = build_valid_v6_authority(root)
    authority = subject._v6_validate_predata_authority(root, authorization_path, go_path)
    reconstructed = subject._v5_recovered_afterstate_commitment(
        authority["recovered_afterstate"]
    )
    assert reconstructed == subject.V5_RECOVERED_AFTERSTATE_COMMITMENT
    assert subject.canonical_payload_sha256(reconstructed) == (
        subject.V5_RECOVERED_AFTERSTATE_COMMITMENT_CANONICAL_SHA256
    )


@pytest.mark.parametrize("field", tuple(subject.V5_RECOVERED_AFTERSTATE_COMMITMENT))
def test_compact_commitment_field_or_inventory_digest_tamper_fails(
    tmp_path: Path, field: str
) -> None:
    root = tmp_path / "r"
    authorization_path, review_path, go_path, _ = build_valid_v6_authority(root)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    go = json.loads(go_path.read_text(encoding="utf-8"))
    value = authorization["recovered_afterstate_commitment"][field]
    replacement = (
        not value if isinstance(value, bool)
        else value + 1 if isinstance(value, int)
        else "0" * 64
    )
    for payload in (authorization, review, go):
        payload["recovered_afterstate_commitment"][field] = replacement
    write_rebound_v6_chain(
        root, authorization_path, review_path, go_path, authorization, review, go
    )
    with pytest.raises(subject.AuditFailure, match="commitment"):
        subject._v6_validate_predata_authority(root, authorization_path, go_path)


def test_full_recovered_afterstate_is_not_embedded_in_v6_authority_controls(
    tmp_path: Path,
) -> None:
    root = tmp_path / "v6-compact-controls"
    authorization_path, review_path, go_path, _ = build_valid_v6_authority(root)
    for path in (authorization_path, review_path, go_path):
        payload = subject._v5_load_strict_json(path, label=path.name)
        subject._v6_assert_compact_control(payload, label=path.name)
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        assert '"recovered_afterstate"' not in encoded
        assert '"recovery_progress"' not in encoded
        assert '"inventory"' not in encoded
        assert not any(isinstance(value, list) and len(value) == 104 for value in payload.values())
    assert review_path.stat().st_size <= 12_000
    assert go_path.stat().st_size <= 12_000


def test_real_v6_authority_and_full_synthetic_core_proofs_are_compositional(
    tmp_path: Path,
) -> None:
    authority_root = tmp_path / "v6-authority"
    authorization_path, _review_path, go_path, _ = build_valid_v6_authority(
        authority_root
    )
    authority = subject._v6_validate_predata_authority(
        authority_root, authorization_path, go_path
    )
    assert authority["historical_contract"] == (
        authority["authorization"]["historical_failed_head_identity_contract"]
    )
    fixture_parent = tmp_path / "v6-synthetic-core"
    fixture_parent.mkdir()
    recovered_root, runner, contract = build_recovered_fixture(fixture_parent)
    report = audit_fixture(recovered_root, runner, contract)
    assert report["status"] == "PASS_PRODUCER_POSTRUN_READONLY_AUDIT_V7"
    assert report["raw_closure"]["ranges"] == contract.range_rows
    assert report["decoded_closure"]["site_rows"] == contract.site_count
    assert report["full_offline_raw_redecode"]["messages"] == contract.range_rows


def test_unbound_matching_control_namespace_entry_fails_before_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "v6-unbound-namespace"
    authorization_path, _review_path, go_path, _ = build_valid_v6_authority(root)
    extra = root / "incidents" / ".TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_EVIL.json"
    extra.write_text("{}\n", encoding="utf-8")
    called = False

    def forbidden(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("data reached before namespace rejection")

    monkeypatch.setattr(subject, "preflight_zero_target_parquet_schemas", forbidden)
    with pytest.raises(subject.AuditFailure, match="namespace inventory"):
        subject._v6_validate_predata_authority(root, authorization_path, go_path)
    assert called is False


@pytest.mark.parametrize(
    "relative",
    (
        "TRACK_A_RAW_PRODUCER_POSTRUN_READONLY_AUDIT_V2.json",
        "raw/TRACK_A_RAW_PRODUCER_POSTRUN_READONLY_AUDIT_V3.json",
        "decoded/TRACK_A_RAW_PRODUCER_POSTRUN_READONLY_AUDIT_V4.json",
        "TRACK_A_RAW_PRODUCER_POSTRUN_READONLY_AUDIT_V5.json",
        "raw/TRACK_A_RAW_PRODUCER_POSTRUN_READONLY_AUDIT_V6.json",
    ),
)
def test_v2_v3_v4_v5_v6_report_candidate_fails_before_data(
    tmp_path: Path, relative: str
) -> None:
    root = tmp_path / "v6-report-candidate"
    authorization_path, _review_path, go_path, _ = build_valid_v6_authority(root)
    report = root / relative
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("{}\n", encoding="utf-8")
    with pytest.raises(subject.AuditFailure, match="postrun report candidate"):
        subject._v6_validate_predata_authority(root, authorization_path, go_path)


@pytest.mark.parametrize(
    "mutation",
    (
        "authorization_extra", "authorization_status", "review_status",
        "review_crosslink", "go_status", "go_crosslink", "required_command",
        "authorization_alias", "review_equal_auth", "go_equal_review",
    ),
)
def test_v6_authority_schema_path_status_crosslink_or_command_tamper_fails(
    tmp_path: Path, mutation: str
) -> None:
    root = tmp_path / f"v6-authority-tamper-{mutation}"
    authorization_path, review_path, go_path, _ = build_valid_v6_authority(root)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    go = json.loads(go_path.read_text(encoding="utf-8"))
    called_authorization = authorization_path
    if mutation == "authorization_extra":
        authorization["unknown"] = True
    elif mutation == "authorization_status":
        authorization["status"] = "WRONG"
    elif mutation == "review_status":
        review["status"] = "WRONG"
    elif mutation == "review_crosslink":
        review["historical_failed_head_identity_contract"]["validation_mode"] = "WRONG"
    elif mutation == "go_status":
        go["status"] = "WRONG"
    elif mutation == "go_crosslink":
        go["incident"] = dict(go["v5_authority_base"]["authorization"])
    elif mutation == "required_command":
        authorization["required_command"] = ["python", "-m", "wrong.module"]
    elif mutation == "authorization_alias":
        called_authorization = authorization_path.with_name("authorization-alias.json")
        shutil.copy2(authorization_path, called_authorization)
    elif mutation == "review_equal_auth":
        review["created_utc"] = authorization["created_utc"]
    else:
        go["created_utc"] = review["created_utc"]
    if mutation != "authorization_alias":
        write_rebound_v6_chain(
            root, authorization_path, review_path, go_path, authorization, review, go
        )
    with pytest.raises(subject.AuditFailure, match="schema|header|cross-binding|command|canonical"):
        subject._v6_validate_predata_authority(root, called_authorization, go_path)


@pytest.mark.parametrize(
    "mutation", ("historical_incident", "false_reject_incident", "namespace", "commitment", "report")
)
def test_v6_postaudit_rechecks_historical_chain_namespace_and_afterstate_commitment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    root = tmp_path / f"v6-postaudit-{mutation}"
    authorization_path, _review_path, go_path, _ = build_valid_v6_authority(root)
    authority = subject._v6_validate_predata_authority(root, authorization_path, go_path)
    build_v5_postauthority_metadata_closure(
        root, {"v4_base": authority["v5_base"]["v4_base"]}, monkeypatch
    )
    if mutation == "historical_incident":
        with (root / subject.RECOVERY_SEALER_ENTRYPOINT_INCIDENT_RELATIVE).open("ab") as handle:
            handle.write(b" ")
        expected = "historical direct-file incident identity mismatch"
    elif mutation == "false_reject_incident":
        with (root / subject.V6_HISTORICAL_FALSE_REJECT_INCIDENT_RELATIVE).open("ab") as handle:
            handle.write(b" ")
        expected = "false-reject incident.*mismatch"
    elif mutation == "namespace":
        extra = root / "incidents" / ".TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_EVIL.json"
        extra.write_text("{}\n", encoding="utf-8")
        expected = "namespace inventory mismatch"
    elif mutation == "commitment":
        authority["recovered_afterstate_commitment"] = {
            **authority["recovered_afterstate_commitment"],
            "canonical_output_count": 7,
        }
        expected = "afterstate/commitment/temp/report recheck mismatch"
    else:
        report = root / "decoded" / "TRACK_A_RAW_PRODUCER_POSTRUN_READONLY_AUDIT_V6.json"
        report.write_text("{}\n", encoding="utf-8")
        expected = "afterstate/commitment/temp/report recheck mismatch"
    with pytest.raises(subject.AuditFailure, match=expected):
        subject._v6_validate_postauthority_afterstate(root, authority)


def test_v6_auditor_is_stdout_only_and_writes_zero_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "v6-main-stdout-only"
    authorization_path, review_path, go_path, _ = build_valid_v6_authority(
        root, frozen_sources=True
    )
    review_path.unlink()
    before = snapshot(root)
    original_open = Path.open

    def deny_write_open(path: Path, mode: str = "r", *args: object, **kwargs: object) -> object:
        assert not any(flag in mode for flag in ("w", "a", "x", "+")), (
            f"V6 main attempted a filesystem write: {path} {mode}"
        )
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", deny_write_open)
    monkeypatch.setattr(
        frozen_v6_subject,
        "_v6_require_canonical_runtime_entrypoint",
        lambda *_args: None,
    )
    rc = frozen_v6_subject.main(
        [
            "--root", str(root), "--authorization", str(authorization_path),
            "--independent-go", str(go_path),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 1
    assert captured.err == ""
    assert payload["artifact_type"] == "TRACK_A_RAW_PRODUCER_POSTRUN_READONLY_AUDIT_V6"
    assert payload["status"] == "FAIL_PRODUCER_POSTRUN_AUDIT_V6"
    assert payload["audit_network_requests"] == 0
    assert payload["audit_files_written"] == 0
    assert snapshot(root) == before
    assert frozen_v6_subject._v6_postrun_report_candidates(root) == []
    source_tree = ast.parse(
        Path(frozen_v6_subject.__file__).read_text(encoding="utf-8")
    )
    forbidden_writers = {
        "write_text", "write_bytes", "unlink", "rename", "mkdir",
        "touch", "symlink_to", "hardlink_to",
    }
    assert not {
        node.func.attr for node in ast.walk(source_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr in forbidden_writers
    }


def test_v6_worker_is_importable_in_exactly_seven_spawn_processes(
    tmp_path: Path,
) -> None:
    del tmp_path
    pilot = (
        subject.REPO
        / "artifacts"
        / "baram2026_ncei_scada_research_20260810_210756"
        / "track_a" / "provenance" / "raw" / "f039"
        / "gfs.t12z.pgrb2.0p25.f039.HPBL_surface.grib2"
    )
    coordinate_path = (
        subject.ROOT_DEFAULT / "prereg" / "authoritative_turbine_coordinate_lock_v1.json"
    )
    assert pilot.stat().st_size == 1_455_839
    assert subject.sha256_file(pilot) == (
        "575fe7876ab93e5470c43dc09c78895385925f5662cca75bd0d8c7f91e677407"
    )
    coordinate = json.loads(coordinate_path.read_text(encoding="utf-8"))
    runtime_identity = json.loads(
        (subject.ROOT_DEFAULT / "prereg" / "raw_launch_authorization_v1.json")
        .read_text(encoding="utf-8")
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
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(subject.DECODE_MAX_WORKERS)
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=subject.DECODE_MAX_WORKERS,
        mp_context=context,
        initializer=subject.v6_spawn_probe_barrier_initializer,
        initargs=(barrier,),
    ) as pool:
        futures = [
            pool.submit(subject.decode_offline_grib_process_worker, info, sites)
            for _index in range(subject.DECODE_MAX_WORKERS)
        ]
        decoded = [future.result(timeout=180) for future in futures]
    assert len({int(report["decoder_process_id"]) for report in decoded}) == 7
    assert {report["decoder_runtime_identity_sha256"] for report in decoded} == {
        subject.canonical_payload_sha256(runtime_identity)
    }
    assert all(report["site_values"] == pytest.approx(expected, abs=1e-12, rel=0.0) for report in decoded)


@pytest.mark.parametrize(
    "failure", ("missing_review", "missing_go", "duplicate_authorization", "historical_contract")
)
def test_v6_authority_and_historical_gate_precede_any_parquet_or_data_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    root = tmp_path / f"v6-order-{failure}"
    authorization_path, review_path, go_path, _ = build_valid_v6_authority(
        root, frozen_sources=True
    )
    if failure == "missing_review":
        review_path.unlink()
    elif failure == "missing_go":
        go_path.unlink()
    elif failure == "duplicate_authorization":
        raw = authorization_path.read_text(encoding="utf-8")
        authorization_path.write_text('{"schema_version":6,' + raw[1:], encoding="utf-8")
    else:
        authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
        authorization["historical_failed_head_identity_contract"][
            "current_role_path_dereference_forbidden"
        ] = False
        write_json(authorization_path, authorization)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("schema/parquet/data/replay reached before V6 authority")

    for name in (
        "preflight_zero_target_parquet_schemas", "parquet_schema_columns",
        "_audit_provenance", "_audit_transaction_and_launch", "_audit_census",
        "_audit_raw_ranges", "_audit_request_events", "_audit_decoded",
        "_audit_full_offline_redecode", "decode_offline_grib_message",
        "decode_offline_grib_process_worker",
    ):
        monkeypatch.setattr(frozen_v6_subject, name, forbidden)
    monkeypatch.setattr(pd, "read_parquet", forbidden)
    monkeypatch.setattr(frozen_v6_subject, "_V6_CANONICAL_MAIN_ACTIVE", True)
    runner = tmp_path / "runner.py"
    runner.write_text("# ordering sentinel\n", encoding="utf-8")
    with pytest.raises(
        frozen_v6_subject.AuditFailure,
        match="absent|duplicate JSON key|historical",
    ):
        frozen_v6_subject.audit(
            root,
            contract=frozen_v6_subject.PRODUCTION_CONTRACT,
            runner_path=runner,
            authorization_path=authorization_path,
            independent_go_path=go_path,
        )


def test_v6_review_and_go_never_embed_full_progress_inventory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "v6-compact-review-go"
    _authorization_path, review_path, go_path, _ = build_valid_v6_authority(root)
    for path in (review_path, go_path):
        payload = subject._v5_load_strict_json(path, label=path.name)
        subject._v6_assert_compact_control(payload, label=path.name)
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        assert '"recovered_afterstate"' not in encoded
        assert '"recovery_progress"' not in encoded
        assert '"inventory"' not in encoded
        assert path.stat().st_size <= 12_000


def test_duplicate_json_key_rejection_is_not_bypassed_by_standard_last_wins_parse(
    tmp_path: Path,
) -> None:
    root = tmp_path / "v6-duplicate-key"
    authorization_path, _review_path, go_path, _ = build_valid_v6_authority(root)
    raw = authorization_path.read_text(encoding="utf-8")
    authorization_path.write_text('{"schema_version":6,' + raw[1:], encoding="utf-8")
    assert json.loads(authorization_path.read_text(encoding="utf-8"))["schema_version"] == 6
    with pytest.raises(subject.AuditFailure, match="duplicate JSON key"):
        subject._v6_validate_predata_authority(root, authorization_path, go_path)


@pytest.mark.parametrize("role", ("false_reject_incident", "direct_file_incident"))
def test_incident_cannot_authorize_v5_or_v6_execution(
    tmp_path: Path, role: str
) -> None:
    root = tmp_path / f"v6-nonauthority-{role}"
    _authorization_path, _review_path, go_path, authorization = build_valid_v6_authority(root)
    false_reject = subject._v6_validate_v5_false_reject_incident(
        root, authorization["incident"]
    )
    assert false_reject["payload"]["authority_scope"] == {
        "documentary_only": True,
        "v5_retry_authorized": False,
        "v6_authorized": False,
    }
    direct_payload = subject._v5_load_strict_json(
        root / subject.RECOVERY_SEALER_ENTRYPOINT_INCIDENT_RELATIVE,
        label="non-authoritative direct-file incident",
    )
    assert direct_payload["status"] == (
        "FAILED_BEFORE_PUBLICATION_RETRY_FORBIDDEN_PENDING_REFREEZE"
    )
    assert direct_payload["mandatory_remediation"]["retry_authorized"] is False
    nonauthority = root / (
        subject.V6_HISTORICAL_FALSE_REJECT_INCIDENT_RELATIVE
        if role == "false_reject_incident"
        else subject.RECOVERY_SEALER_ENTRYPOINT_INCIDENT_RELATIVE
    )
    with pytest.raises(subject.AuditFailure, match="canonical"):
        subject._v6_validate_predata_authority(root, nonauthority, go_path)
    frozen_nonauthority = subject.ROOT_DEFAULT / (
        subject.V6_HISTORICAL_FALSE_REJECT_INCIDENT_RELATIVE
        if role == "false_reject_incident"
        else subject.RECOVERY_SEALER_ENTRYPOINT_INCIDENT_RELATIVE
    )
    with pytest.raises(frozen_v5_subject.AuditFailure, match="canonical"):
        frozen_v5_subject._v5_validate_predata_authority(
            frozen_v5_subject.ROOT_DEFAULT,
            frozen_nonauthority,
            frozen_v5_subject.ROOT_DEFAULT / frozen_v5_subject.V5_GO_RELATIVE,
        )


def test_v6_auth_sealer_timestamp_precision_false_reject_incident_is_exact() -> None:
    record = artifact(
        subject.ROOT_DEFAULT / subject.V7_TIMESTAMP_FALSE_REJECT_INCIDENT_RELATIVE,
        subject.ROOT_DEFAULT,
    )
    result = subject._v7_validate_timestamp_false_reject_incident(
        subject.ROOT_DEFAULT, record
    )
    assert result["identity"] == {
        "path": subject.V7_TIMESTAMP_FALSE_REJECT_INCIDENT_RELATIVE,
        "size_bytes": 11_799,
        "sha256": "722f1540cf73ed78aeff25ea3a2ad9300e40aa13fcc068f998898251e68ffd19",
    }
    assert result["payload"]["created_utc"] == "2026-08-11T04:22:34.325428Z"


def test_v7_rfc3339_100ns_parser_accepts_zero_through_seven_fractional_digits() -> None:
    for digits in range(8):
        fraction = "" if digits == 0 else "." + ("1" * digits)
        raw = f"2026-08-10T18:45:00{fraction}Z"
        parsed = subject._v7_parse_rfc3339_utc_100ns(raw, label="test")
        assert parsed["raw"] == raw
        assert parsed["fractional_digits"] == digits
        assert isinstance(parsed["ticks_100ns"], int)


def test_v7_rfc3339_100ns_parser_preserves_exact_seven_digit_tick_ordering() -> None:
    lower = subject._v7_parse_rfc3339_utc_100ns(
        "2026-08-10T18:45:00.5807527Z", label="lower"
    )
    exact = subject._v7_parse_rfc3339_utc_100ns(
        "2026-08-10T18:45:00.5807528Z", label="exact"
    )
    assert exact["ticks_100ns"] - lower["ticks_100ns"] == 1
    short = subject._v7_parse_rfc3339_utc_100ns(
        "2026-08-10T18:45:00.1Z", label="short"
    )
    long = subject._v7_parse_rfc3339_utc_100ns(
        "2026-08-10T18:45:00.1000000Z", label="long"
    )
    assert short["ticks_100ns"] == long["ticks_100ns"]
    assert short["raw"] != long["raw"]


@pytest.mark.parametrize(
    "raw",
    [
        "2026-08-10T18:45:00.58075280Z",
        "2026-08-10T18:45:00+00:00",
        "2026-08-10T18:45:00z",
        "2026-08-10T18:45:00.Z",
        "2026-08-10T18:45:00,5Z",
        "2026-08-10T18:45:00.5Z ",
        "２０２６-08-10T18:45:00Z",
        "2026-02-29T18:45:00Z",
        "2026-08-10T24:00:00Z",
        "not-a-timestamp",
    ],
)
def test_v7_rfc3339_100ns_parser_rejects_eight_digits_offsets_non_utc_and_malformed(
    raw: str,
) -> None:
    with pytest.raises(subject.AuditFailure, match="timestamp is invalid"):
        subject._v7_parse_rfc3339_utc_100ns(raw, label="test")


def test_v7_actual_722f_incident_and_raw_timestamp_bindings_pass(
    tmp_path: Path,
) -> None:
    result = subject._v7_validate_timestamp_false_reject_incident(
        subject.ROOT_DEFAULT,
        artifact(
            subject.ROOT_DEFAULT
            / subject.V7_TIMESTAMP_FALSE_REJECT_INCIDENT_RELATIVE,
            subject.ROOT_DEFAULT,
        ),
    )
    assert result["historical_created"]["raw"] == "2026-08-10T18:45:00.5807528Z"
    assert result["historical_created"]["fractional_digits"] == 7
    assert (
        result["historical_created"]["ticks_100ns"]
        < result["v5_authorization_created"]["ticks_100ns"]
        < result["created"]["ticks_100ns"]
    )
    root = tmp_path / "v7-authority-positive"
    authorization_path, _review_path, go_path, _authorization = (
        build_valid_v7_authority(root)
    )
    authority = subject._v7_validate_predata_authority(
        root, authorization_path, go_path
    )
    assert authority["timestamp_chain"]["raw"]["historical_9e74"] == (
        "2026-08-10T18:45:00.5807528Z"
    )
    assert authority["timestamp_incident"]["identity"] == result["identity"]


def test_timestamp_raw_value_or_chronology_substitution_fails(
    tmp_path: Path,
) -> None:
    root = tmp_path / "v7-raw-substitution"
    authorization_path, _review_path, go_path, _ = build_valid_v7_authority(root)
    authority = subject._v7_validate_predata_authority(
        root, authorization_path, go_path
    )
    timestamp_incident = json.loads(
        json.dumps(authority["timestamp_incident"], ensure_ascii=False)
    )
    direct_payload = subject._v5_load_strict_json(
        root / subject.RECOVERY_SEALER_ENTRYPOINT_INCIDENT_RELATIVE,
        label="raw substitution direct incident",
    )
    timestamp_incident["payload"]["historical_timestamp"]["value"] = (
        "2026-08-10T18:45:00.1Z"
    )
    direct_payload["created_utc"] = "2026-08-10T18:45:00.1000000Z"
    assert (
        subject._v7_parse_rfc3339_utc_100ns(
            timestamp_incident["payload"]["historical_timestamp"]["value"],
            label="short",
        )["ticks_100ns"]
        == subject._v7_parse_rfc3339_utc_100ns(
            direct_payload["created_utc"], label="long"
        )["ticks_100ns"]
    )
    with pytest.raises(subject.AuditFailure, match="raw timestamp binding"):
        subject._v7_validate_timestamp_chain(
            timestamp_incident=timestamp_incident,
            direct_payload=direct_payload,
            false_reject=authority["false_reject_incident"],
            v5_base=authority["v5_base"],
            authorization=authority["authorization"],
            review=authority["review"],
            go=authority["go"],
        )
    review = dict(authority["review"])
    review["created_utc"] = authority["authorization"]["created_utc"]
    with pytest.raises(subject.AuditFailure, match="chronology"):
        subject._v7_validate_timestamp_chain(
            timestamp_incident=authority["timestamp_incident"],
            direct_payload=subject._v5_load_strict_json(
                root / subject.RECOVERY_SEALER_ENTRYPOINT_INCIDENT_RELATIVE,
                label="chronology direct incident",
            ),
            false_reject=authority["false_reject_incident"],
            v5_base=authority["v5_base"],
            authorization=authority["authorization"],
            review=review,
            go=authority["go"],
        )


def test_v6_and_v5_production_data_and_replay_core_ast_is_identical() -> None:
    v5_tree = ast.parse(subject.FROZEN_V5_AUDITOR.read_text(encoding="utf-8"))
    v6_tree = ast.parse(subject.FROZEN_V6_AUDITOR.read_text(encoding="utf-8"))
    v7_tree = ast.parse(Path(subject.__file__).read_text(encoding="utf-8"))
    trees = [v5_tree, v6_tree, v7_tree]
    function_maps = [
        {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        for tree in trees
    ]
    ordered = [
        node.name for node in v5_tree.body if isinstance(node, ast.FunctionDef)
    ]
    first = ordered.index("recovery_progress_counts")
    last = ordered.index("_audit_full_offline_redecode")
    core_names = ordered[first : last + 1]
    assert len(core_names) == 39
    dumps = [
        {
            name: ast.dump(functions[name], include_attributes=False)
            for name in core_names
        }
        for functions in function_maps
    ]
    assert dumps[0] == dumps[1] == dumps[2]


def test_full_recovered_afterstate_is_not_embedded_in_v7_authority_controls(
    tmp_path: Path,
) -> None:
    root = tmp_path / "v7-compact-controls"
    authorization_path, review_path, go_path, _ = build_valid_v7_authority(root)
    for path in (authorization_path, review_path, go_path):
        payload = subject._v5_load_strict_json(path, label=path.name)
        subject._v7_assert_compact_control(payload, label=path.name)
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        assert '"recovered_afterstate"' not in encoded
        assert '"recovery_progress"' not in encoded
        assert '"inventory"' not in encoded
        assert not any(
            isinstance(value, list) and len(value) == 104
            for value in payload.values()
        )
    assert review_path.stat().st_size <= 12_000
    assert go_path.stat().st_size <= 12_000


def test_real_v7_authority_and_full_synthetic_core_proofs_are_compositional(
    tmp_path: Path,
) -> None:
    authority_root = tmp_path / "v7-authority"
    authorization_path, _review_path, go_path, _ = build_valid_v7_authority(
        authority_root
    )
    authority = subject._v7_validate_predata_authority(
        authority_root, authorization_path, go_path
    )
    assert authority["historical_contract"] == (
        authority["authorization"]["historical_failed_head_identity_contract"]
    )
    assert authority["timestamp_incident"]["identity"]["sha256"] == (
        subject.V7_TIMESTAMP_FALSE_REJECT_INCIDENT_SHA256
    )
    fixture_parent = tmp_path / "v7-synthetic-core"
    fixture_parent.mkdir()
    recovered_root, runner, contract = build_recovered_fixture(fixture_parent)
    report = audit_fixture(recovered_root, runner, contract)
    assert report["status"] == "PASS_PRODUCER_POSTRUN_READONLY_AUDIT_V7"
    assert report["raw_closure"]["ranges"] == contract.range_rows
    assert report["decoded_closure"]["site_rows"] == contract.site_count
    assert report["full_offline_raw_redecode"]["messages"] == contract.range_rows


@pytest.mark.parametrize(
    "relative", subject.V7_POSTRUN_REPORT_CANDIDATES,
)
def test_v2_v3_v4_v5_v6_v7_report_candidate_fails_before_data(
    tmp_path: Path, relative: str
) -> None:
    root = tmp_path / "v7-report-candidate"
    authorization_path, _review_path, go_path, _ = build_valid_v7_authority(root)
    report = root / relative
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("{}\n", encoding="utf-8")
    with pytest.raises(subject.AuditFailure, match="postrun report candidate"):
        subject._v7_validate_predata_authority(root, authorization_path, go_path)


@pytest.mark.parametrize(
    "mutation",
    (
        "authorization_extra", "authorization_status", "review_status",
        "review_crosslink", "go_status", "go_crosslink", "required_command",
        "authorization_alias", "review_equal_auth", "go_equal_review",
    ),
)
def test_v7_authority_schema_path_status_crosslink_or_command_tamper_fails(
    tmp_path: Path, mutation: str
) -> None:
    root = tmp_path / f"v7-authority-tamper-{mutation}"
    authorization_path, review_path, go_path, _ = build_valid_v7_authority(root)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    go = json.loads(go_path.read_text(encoding="utf-8"))
    called_authorization = authorization_path
    if mutation == "authorization_extra":
        authorization["unknown"] = True
    elif mutation == "authorization_status":
        authorization["status"] = "WRONG"
    elif mutation == "review_status":
        review["status"] = "WRONG"
    elif mutation == "review_crosslink":
        review["historical_failed_head_identity_contract"]["validation_mode"] = "WRONG"
    elif mutation == "go_status":
        go["status"] = "WRONG"
    elif mutation == "go_crosslink":
        go["incident"] = dict(go["v5_authority_base"]["authorization"])
    elif mutation == "required_command":
        authorization["required_command"] = ["python", "-m", "wrong.module"]
    elif mutation == "authorization_alias":
        called_authorization = authorization_path.with_name("authorization-alias.json")
        shutil.copy2(authorization_path, called_authorization)
    elif mutation == "review_equal_auth":
        review["created_utc"] = authorization["created_utc"]
    else:
        go["created_utc"] = review["created_utc"]
    if mutation != "authorization_alias":
        write_rebound_v6_chain(
            root, authorization_path, review_path, go_path,
            authorization, review, go,
        )
    with pytest.raises(
        subject.AuditFailure,
        match="schema|header|cross-binding|command|canonical|chronology",
    ):
        subject._v7_validate_predata_authority(root, called_authorization, go_path)


def test_v7_auditor_is_stdout_only_and_writes_zero_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "v7-main-stdout-only"
    authorization_path, review_path, go_path, _ = build_valid_v7_authority(root)
    review_path.unlink()
    before = snapshot(root)
    original_open = Path.open

    def deny_write_open(
        path: Path, mode: str = "r", *args: object, **kwargs: object
    ) -> object:
        assert not any(flag in mode for flag in ("w", "a", "x", "+")), (
            f"V7 main attempted a filesystem write: {path} {mode}"
        )
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", deny_write_open)
    monkeypatch.setattr(
        subject, "_v7_require_canonical_runtime_entrypoint", lambda *_args: None
    )
    rc = subject.main(
        [
            "--root", str(root), "--authorization", str(authorization_path),
            "--independent-go", str(go_path),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 1
    assert captured.err == ""
    assert payload["artifact_type"] == "TRACK_A_RAW_PRODUCER_POSTRUN_READONLY_AUDIT_V7"
    assert payload["status"] == "FAIL_PRODUCER_POSTRUN_AUDIT_V7"
    assert payload["audit_network_requests"] == 0
    assert payload["audit_files_written"] == 0
    assert snapshot(root) == before
    assert subject._v7_postrun_report_candidates(root) == []
    source_tree = ast.parse(Path(subject.__file__).read_text(encoding="utf-8"))
    forbidden_writers = {
        "write_text", "write_bytes", "unlink", "rename", "mkdir",
        "touch", "symlink_to", "hardlink_to",
    }
    assert not {
        node.func.attr for node in ast.walk(source_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr in forbidden_writers
    }


def test_v7_worker_is_importable_in_exactly_seven_spawn_processes(
    tmp_path: Path,
) -> None:
    del tmp_path
    pilot = (
        subject.REPO
        / "artifacts" / "baram2026_ncei_scada_research_20260810_210756"
        / "track_a" / "provenance" / "raw" / "f039"
        / "gfs.t12z.pgrb2.0p25.f039.HPBL_surface.grib2"
    )
    coordinate_path = (
        subject.ROOT_DEFAULT / "prereg" / "authoritative_turbine_coordinate_lock_v1.json"
    )
    assert pilot.stat().st_size == 1_455_839
    assert subject.sha256_file(pilot) == (
        "575fe7876ab93e5470c43dc09c78895385925f5662cca75bd0d8c7f91e677407"
    )
    coordinate = json.loads(coordinate_path.read_text(encoding="utf-8"))
    runtime_identity = json.loads(
        (subject.ROOT_DEFAULT / "prereg" / "raw_launch_authorization_v1.json")
        .read_text(encoding="utf-8")
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
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(subject.DECODE_MAX_WORKERS)
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=subject.DECODE_MAX_WORKERS,
        mp_context=context,
        initializer=subject.v7_spawn_probe_barrier_initializer,
        initargs=(barrier,),
    ) as pool:
        futures = [
            pool.submit(subject.decode_offline_grib_process_worker, info, sites)
            for _index in range(subject.DECODE_MAX_WORKERS)
        ]
        decoded = [future.result(timeout=180) for future in futures]
    assert len({int(report["decoder_process_id"]) for report in decoded}) == 7
    assert {report["decoder_runtime_identity_sha256"] for report in decoded} == {
        subject.canonical_payload_sha256(runtime_identity)
    }
    assert all(
        report["site_values"] == pytest.approx(expected, abs=1e-12, rel=0.0)
        for report in decoded
    )


def test_v7_review_and_go_never_embed_full_progress_inventory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "v7-compact-review-go"
    _authorization_path, review_path, go_path, _ = build_valid_v7_authority(root)
    for path in (review_path, go_path):
        payload = subject._v5_load_strict_json(path, label=path.name)
        subject._v7_assert_compact_control(payload, label=path.name)
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        assert '"recovered_afterstate"' not in encoded
        assert '"recovery_progress"' not in encoded
        assert '"inventory"' not in encoded
        assert path.stat().st_size <= 12_000


@pytest.mark.parametrize("role", ("timestamp_incident", "direct_file_incident"))
def test_incident_cannot_authorize_v6_or_v7_execution(
    tmp_path: Path, role: str
) -> None:
    root = tmp_path / f"v7-nonauthority-{role}"
    _authorization_path, _review_path, go_path, authorization = (
        build_valid_v7_authority(root)
    )
    timestamp = subject._v7_validate_timestamp_false_reject_incident(
        root, authorization["incident"]
    )
    assert timestamp["payload"]["authority_scope"] == {
        "documentary_only": True,
        "v6_retry_authorized": False,
        "v6_review_go_or_live_audit_authorized": False,
        "v7_implementation_or_execution_authorized": False,
    }
    direct_payload = subject._v5_load_strict_json(
        root / subject.RECOVERY_SEALER_ENTRYPOINT_INCIDENT_RELATIVE,
        label="non-authoritative direct-file incident",
    )
    assert direct_payload["mandatory_remediation"]["retry_authorized"] is False
    nonauthority = root / (
        subject.V7_TIMESTAMP_FALSE_REJECT_INCIDENT_RELATIVE
        if role == "timestamp_incident"
        else subject.RECOVERY_SEALER_ENTRYPOINT_INCIDENT_RELATIVE
    )
    with pytest.raises(subject.AuditFailure, match="canonical"):
        subject._v7_validate_predata_authority(root, nonauthority, go_path)
    with pytest.raises(frozen_v6_subject.AuditFailure, match="canonical"):
        frozen_v6_subject._v6_validate_predata_authority(
            frozen_v6_subject.ROOT_DEFAULT,
            subject.ROOT_DEFAULT / subject.V7_TIMESTAMP_FALSE_REJECT_INCIDENT_RELATIVE,
            frozen_v6_subject.ROOT_DEFAULT / frozen_v6_subject.V6_GO_RELATIVE,
        )


@pytest.mark.parametrize(
    "failure", ("missing_review", "missing_go", "duplicate_authorization", "timestamp_incident")
)
def test_timestamp_and_historical_predata_gate_precedes_schema_parquet_data_and_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    root = tmp_path / f"v7-order-{failure}"
    authorization_path, review_path, go_path, _ = build_valid_v7_authority(root)
    if failure == "missing_review":
        review_path.unlink()
    elif failure == "missing_go":
        go_path.unlink()
    elif failure == "duplicate_authorization":
        raw = authorization_path.read_text(encoding="utf-8")
        authorization_path.write_text('{"schema_version":7,' + raw[1:], encoding="utf-8")
    else:
        incident_path = root / subject.V7_TIMESTAMP_FALSE_REJECT_INCIDENT_RELATIVE
        with incident_path.open("ab") as handle:
            handle.write(b" ")

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("schema/parquet/data/replay reached before V7 authority")

    for name in (
        "preflight_zero_target_parquet_schemas", "parquet_schema_columns",
        "_audit_provenance", "_audit_transaction_and_launch", "_audit_census",
        "_audit_raw_ranges", "_audit_request_events", "_audit_decoded",
        "_audit_full_offline_redecode", "decode_offline_grib_message",
        "decode_offline_grib_process_worker",
    ):
        monkeypatch.setattr(subject, name, forbidden)
    monkeypatch.setattr(pd, "read_parquet", forbidden)
    monkeypatch.setattr(subject, "_V7_CANONICAL_MAIN_ACTIVE", True)
    runner = tmp_path / "runner.py"
    runner.write_text("# V7 ordering sentinel\n", encoding="utf-8")
    with pytest.raises(subject.AuditFailure, match="absent|duplicate JSON key|timestamp incident"):
        subject.audit(
            root,
            contract=subject.PRODUCTION_CONTRACT,
            runner_path=runner,
            authorization_path=authorization_path,
            independent_go_path=go_path,
        )


def test_frozen_v6_reproduces_exact_sealer_timestamp_precision_false_reject() -> None:
    """Carry the real seven-digit 9e74 value into the frozen V6 builder."""

    root = subject.ROOT_DEFAULT.resolve()
    v7_code = v7_sealer.collect_code_state(v7_sealer.REPO)
    v7_inputs = v7_sealer.collect_root_inputs(root, v7_code)
    assert v7_inputs["historical_incident_created_utc"] == (
        "2026-08-10T18:45:00.5807528Z"
    )

    frozen_code = frozen_v6_sealer.collect_code_state(frozen_v6_sealer.REPO)
    frozen_inputs = dict(v7_inputs)
    frozen_inputs["incident"] = dict(
        frozen_v6_sealer.V6_HISTORICAL_FALSE_REJECT_INCIDENT_IDENTITY
    )
    frozen_inputs["incident_created_utc"] = v7_inputs[
        "v5_false_reject_incident_created_utc"
    ]
    # These deliberately non-authoritative later fields prove the exact timestamp
    # parser rejects before evidence/namespace semantics are consulted.
    frozen_inputs["preaudit_zero_mutation_snapshot"] = {
        "sentinel": "must not reach V6 zero-snapshot validation"
    }
    with pytest.raises(
        frozen_v6_sealer.PostrunAuditSealError,
        match="^historical_incident_created_utc timestamp is invalid$",
    ):
        frozen_v6_sealer.build_authorization(
            root,
            frozen_code,
            frozen_inputs,
            {"sentinel": "must not reach V6 test-evidence validation"},
            created_utc="2026-08-11T05:00:01.000000Z",
        )


def test_v7_actual_9e74_collect_build_and_self_validator_passes() -> None:
    """Exercise the real metadata-only V7 sealer path without publication."""

    root = subject.ROOT_DEFAULT.resolve()
    control_relatives = (
        v7_sealer.AUTH_RELATIVE,
        v7_sealer.REVIEW_RELATIVE,
        v7_sealer.GO_RELATIVE,
    )
    before = {
        relative: os.path.lexists(root / relative)
        for relative in control_relatives
    }
    assert before == {relative: False for relative in control_relatives}

    code_state = v7_sealer.collect_code_state(v7_sealer.REPO)
    inputs = v7_sealer.collect_root_inputs(root, code_state)
    assert inputs["historical_incident_created_utc"] == (
        "2026-08-10T18:45:00.5807528Z"
    )
    bound = v7_sealer._evidence_bound_identities(
        root, code_state["bound_identities"]
    )
    evidence = synthetic_v7_test_evidence(
        "2026-08-11T05:00:00.000000Z", bound
    )
    payload = v7_sealer.build_authorization(
        root,
        code_state,
        inputs,
        evidence,
        created_utc="2026-08-11T05:00:01.000000Z",
    )
    v7_sealer.validate_authorization_payload(
        root, payload, code_state, inputs
    )
    assert set(payload) == subject.V7_AUTH_KEYS
    assert payload["incident"] == (
        v7_sealer.V7_TIMESTAMP_PRECISION_FALSE_REJECT_INCIDENT_IDENTITY
    )
    assert {
        relative: os.path.lexists(root / relative)
        for relative in control_relatives
    } == before
    assert v7_sealer.control_temporary_files(root) == []


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        ("timestamp_incident", "timestamp incident changed during audit"),
        (
            "historical_contract",
            "historical failed-head identity contract mismatch",
        ),
    ),
)
def test_late_reuse_revalidates_timestamp_incident_and_historical_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected: str,
) -> None:
    root = tmp_path / f"v7-late-reuse-{mutation}"
    authorization_path, _review_path, go_path, _ = build_valid_v7_authority(root)
    authority = subject._v7_validate_predata_authority(
        root, authorization_path, go_path
    )
    build_v5_postauthority_metadata_closure(
        root, {"v4_base": authority["v5_base"]["v4_base"]}, monkeypatch
    )
    runner = tmp_path / f"runner-{mutation}.py"
    runner.write_text("# V7 postauthority start/end sentinel\n", encoding="utf-8")
    before = snapshot(root)
    original_poststate = subject._v7_validate_postauthority_afterstate
    poststate_calls: list[int] = []

    def counted_poststate(
        call_root: Path, call_authority: dict[str, object]
    ) -> dict[str, object]:
        poststate_calls.append(len(poststate_calls) + 1)
        return original_poststate(call_root, call_authority)

    monkeypatch.setattr(
        subject, "_v7_validate_postauthority_afterstate", counted_poststate
    )
    monkeypatch.setattr(
        subject, "_v7_validate_predata_authority", lambda *_args: authority
    )
    monkeypatch.setattr(subject, "_V7_CANONICAL_MAIN_ACTIVE", True)
    monkeypatch.setattr(
        subject, "require_no_symlink_chain", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(subject, "_audit_active_lock", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        subject,
        "_audit_recovery_control_temporary_files_absent",
        lambda *_args: {"matching_temporary_files": 0},
    )
    monkeypatch.setattr(
        subject, "preflight_zero_target_parquet_schemas", lambda *_args: {}
    )
    monkeypatch.setattr(
        subject,
        "_audit_provenance",
        lambda *_args: {
            "field_census_path": tmp_path / "unused-census.json",
            "coordinate": {},
            "runtime_identity_sha256": subject.V4_RUNTIME_IDENTITY_SHA256,
            "authorization_identity": {}, "go_identity": {},
            "preflight_identity": {}, "preflight_amendment_identity": {},
            "independent_prelaunch_identity": {},
            "independent_prelaunch_identity_base": {}, "runner_identity": {},
            "runner_test_identity": {}, "embedded_identity_rehashes": {},
            "external_identity_unique_file_count": 0,
            "external_identity_unique_file_bytes": 0,
        },
    )
    monkeypatch.setattr(
        subject,
        "_audit_transaction_and_launch",
        lambda *_args: {
            "outputs": {"raw_parquet": None, "raw_csv": None},
            "attempt_id": "synthetic", "complete_lock_identity": {},
            "launch_history_lock_count": 1, "plan_identity": {},
            "commit_identity": {},
            "transaction_inventory_recursively_closed": True,
            "planned_staged_files_present": 0,
        },
    )
    monkeypatch.setattr(subject, "_audit_census", lambda *_args: (None, []))
    monkeypatch.setattr(
        subject,
        "_audit_raw_ranges",
        lambda *_args: {"raw_info": [], "raw_rows": 0, "raw_bytes": 0},
    )
    monkeypatch.setattr(
        subject,
        "_audit_request_events",
        lambda *_args: {
            "starts": 0, "completions": 0, "errors": 0,
            "indeterminate_starts": 0, "max_global_attempt_number": 0,
        },
    )
    monkeypatch.setattr(subject, "_audit_decoded", lambda *_args: {})
    monkeypatch.setattr(
        subject, "_audit_full_offline_redecode", lambda *_args, **_kwargs: {}
    )

    def drift_between_start_and_end(*_args: object) -> dict[str, object]:
        if mutation == "timestamp_incident":
            authority["timestamp_incident"]["identity"]["sha256"] = "0" * 64
        else:
            authority["authorization"][
                "historical_failed_head_identity_contract"
            ]["validation_mode"] = "DRIFTED_BETWEEN_START_AND_END"
        return {
            "external_identity_unique_file_count": 0,
            "external_identity_unique_file_bytes": 0,
        }

    monkeypatch.setattr(
        subject, "_audit_canonical_metadata", drift_between_start_and_end
    )
    with pytest.raises(subject.AuditFailure, match=expected):
        subject.audit(
            root,
            contract=subject.PRODUCTION_CONTRACT,
            runner_path=runner,
            authorization_path=authorization_path,
            independent_go_path=go_path,
        )
    assert poststate_calls == [1, 2]
    assert snapshot(root) == before
    audit_node = next(
        node
        for node in ast.parse(
            Path(subject.__file__).read_text(encoding="utf-8")
        ).body
        if isinstance(node, ast.FunctionDef) and node.name == "audit"
    )
    called_names = [
        node.func.id
        for node in ast.walk(audit_node)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert called_names.count("_v7_validate_predata_authority") == 1
    assert called_names.count("_v7_validate_postauthority_afterstate") == 2


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        ("timestamp_incident", "timestamp incident.*(physical|size|identity)"),
        (
            "historical_incident",
            "historical direct-file incident identity mismatch",
        ),
        ("false_reject_incident", "false-reject incident.*mismatch"),
        ("namespace", "namespace inventory mismatch"),
        ("commitment", "afterstate/commitment/temp/report recheck mismatch"),
        ("report", "afterstate/commitment/temp/report recheck mismatch"),
    ),
)
def test_v7_postaudit_rechecks_timestamp_incident_historical_chain_namespace_and_afterstate_commitment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected: str,
) -> None:
    root = tmp_path / f"v7-postaudit-{mutation}"
    authorization_path, _review_path, go_path, _ = build_valid_v7_authority(root)
    authority = subject._v7_validate_predata_authority(
        root, authorization_path, go_path
    )
    build_v5_postauthority_metadata_closure(
        root, {"v4_base": authority["v5_base"]["v4_base"]}, monkeypatch
    )
    if mutation == "timestamp_incident":
        path = root / subject.V7_TIMESTAMP_FALSE_REJECT_INCIDENT_RELATIVE
        with path.open("ab") as handle:
            handle.write(b" ")
    elif mutation == "historical_incident":
        path = root / subject.RECOVERY_SEALER_ENTRYPOINT_INCIDENT_RELATIVE
        with path.open("ab") as handle:
            handle.write(b" ")
    elif mutation == "false_reject_incident":
        path = root / subject.V6_HISTORICAL_FALSE_REJECT_INCIDENT_RELATIVE
        with path.open("ab") as handle:
            handle.write(b" ")
    elif mutation == "namespace":
        extra = (
            root / "incidents"
            / ".TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_EVIL.json"
        )
        extra.write_text("{}\n", encoding="utf-8")
    elif mutation == "commitment":
        authority["recovered_afterstate_commitment"] = {
            **authority["recovered_afterstate_commitment"],
            "canonical_output_count": 7,
        }
    else:
        report = (
            root / "decoded"
            / "TRACK_A_RAW_PRODUCER_POSTRUN_READONLY_AUDIT_V7.json"
        )
        report.write_text("{}\n", encoding="utf-8")
    with pytest.raises(subject.AuditFailure, match=expected):
        subject._v7_validate_postauthority_afterstate(root, authority)

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import audit_noaa_gfs_multiseason_raw_postrun_v7 as auditor
from scripts import seal_noaa_gfs_multiseason_postrun_audit_v7 as subject


@pytest.mark.parametrize(
    ("fraction", "expected_fractional_ticks"),
    (
        ("", 0),
        (".1", 1_000_000),
        (".12", 1_200_000),
        (".123", 1_230_000),
        (".1234", 1_234_000),
        (".12345", 1_234_500),
        (".123456", 1_234_560),
        (".1234567", 1_234_567),
    ),
)
def test_exact_utc_z_parser_accepts_zero_through_seven_fractional_digits(
    fraction: str, expected_fractional_ticks: int
) -> None:
    base = subject._exact_utc_z_100ns_ticks(
        "2026-08-10T18:45:00Z", label="base"
    )
    observed = subject._exact_utc_z_100ns_ticks(
        f"2026-08-10T18:45:00{fraction}Z", label="candidate"
    )
    assert observed - base == expected_fractional_ticks


def test_actual_immutable_9e74_seven_digit_timestamp_is_exact() -> None:
    earlier = subject._exact_utc_z_100ns_ticks(
        "2026-08-10T18:45:00.5807527Z", label="earlier"
    )
    actual = subject._exact_utc_z_100ns_ticks(
        "2026-08-10T18:45:00.5807528Z", label="actual 9e74"
    )
    assert actual - earlier == 1


def test_actual_722f_incident_binds_raw_9e74_timestamp_and_integer_ticks() -> None:
    state = subject._load_and_validate_v7_timestamp_precision_incident(
        subject.ROOT_DEFAULT.resolve(),
        subject.V7_TIMESTAMP_PRECISION_FALSE_REJECT_INCIDENT_IDENTITY,
    )
    assert state["identity"] == subject.V7_TIMESTAMP_PRECISION_FALSE_REJECT_INCIDENT_IDENTITY
    assert state["historical_created_utc"] == "2026-08-10T18:45:00.5807528Z"
    assert state["historical_ticks_100ns"] == subject._exact_utc_z_100ns_ticks(
        state["historical_created_utc"], label="actual historical timestamp"
    )
    assert state["historical_ticks_100ns"] < state["created_ticks_100ns"]


@pytest.mark.parametrize("field", ("path", "size_bytes", "sha256"))
def test_722f_incident_declared_identity_tamper_fails(field: str) -> None:
    tampered = dict(subject.V7_TIMESTAMP_PRECISION_FALSE_REJECT_INCIDENT_IDENTITY)
    if field == "path":
        tampered[field] = subject.V6_HISTORICAL_FALSE_REJECT_INCIDENT_IDENTITY["path"]
    elif field == "size_bytes":
        tampered[field] = int(tampered[field]) + 1
    else:
        tampered[field] = "0" * 64
    with pytest.raises(subject.PostrunAuditSealError, match="declaration mismatch"):
        subject._load_and_validate_v7_timestamp_precision_incident(
            subject.ROOT_DEFAULT.resolve(), tampered
        )


@pytest.mark.parametrize(
    "value",
    (
        "2026-08-10T18:45:00.12345678Z",
        "2026-08-10T18:45:00.Z",
        "2026-08-10T18:45:00+00:00",
        "2026-08-10T18:45:00.1+00:00",
        "2026-08-10T18:45:00z",
        "2026-08-10T18:45:00UTC",
        " 2026-08-10T18:45:00Z",
        "2026-02-29T18:45:00Z",
        "2026-08-10T18:45:60Z",
        None,
        1,
    ),
)
def test_exact_utc_z_parser_rejects_overprecision_offset_non_utc_or_malformed(
    value: object,
) -> None:
    with pytest.raises(subject.PostrunAuditSealError, match="timestamp"):
        subject._exact_utc_z_100ns_ticks(value, label="invalid")


def test_distinct_raw_timestamp_spellings_can_share_ticks_but_not_identity() -> None:
    short = "2026-08-10T18:45:00.1Z"
    padded = "2026-08-10T18:45:00.1000000Z"
    assert short != padded
    assert subject._exact_utc_z_100ns_ticks(short, label="short") == (
        subject._exact_utc_z_100ns_ticks(padded, label="padded")
    )


def test_100ns_parser_uses_only_integer_calendar_arithmetic() -> None:
    source = Path(subject.__file__).resolve().read_text(encoding="utf-8")
    tree = ast.parse(source)
    parser = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_exact_utc_z_100ns_ticks"
    )
    parser_source = ast.get_source_segment(source, parser) or ""
    forbidden_calls = {
        "float", "fromisoformat", "strptime", "timestamp", "total_seconds"
    }
    observed_calls = {
        node.func.id
        if isinstance(node.func, ast.Name)
        else node.func.attr
        for node in ast.walk(parser)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert forbidden_calls.isdisjoint(observed_calls)
    assert "%f" not in parser_source
    assert '.ljust(7, "0")' in parser_source
    assert "10_000_000" in parser_source


def write_bytes(path: Path, data: bytes = b"{}") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def identity(path: Path, root: Path | None = None) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix() if root else str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def control_root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    for relative in (
        "prereg", "independent_redteam", "incidents", "raw", "decoded"
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    return root


def populate_historical_namespace(root: Path) -> None:
    for relative in (
        auditor.FROZEN_V3_AUTH_RELATIVE,
        auditor.FROZEN_V3_REVIEW_RELATIVE,
        auditor.FROZEN_V3_MISPLACED_GO_RELATIVE,
        auditor.V4_AUTH_RELATIVE,
        auditor.V4_REVIEW_RELATIVE,
        auditor.V5_AUTH_RELATIVE,
        auditor.V5_REVIEW_RELATIVE,
        auditor.V5_GO_RELATIVE,
        auditor.V2_FALSE_REJECT_INCIDENT_RELATIVE,
        auditor.V4_PATH_MISPUBLISH_INCIDENT_RELATIVE,
        auditor.V5_TRANSPORT_INCIDENT_RELATIVE,
        auditor.V5_TRANSPORT_CORRECTION_RELATIVE,
        auditor.V6_HISTORICAL_FALSE_REJECT_INCIDENT_RELATIVE,
        auditor.V7_TIMESTAMP_FALSE_REJECT_INCIDENT_RELATIVE,
    ):
        write_bytes(root / relative)


def fake_code_state(tmp_path: Path) -> dict[str, object]:
    paths: dict[str, Path] = {}
    for role in sorted(subject.BOUND_SOURCE_ROLES):
        if role in subject.FROZEN_V6_SOURCE_IDENTITIES:
            path = subject.BOUND_SOURCE_PATHS[role]
        elif role == "v7_auditor":
            path = subject.AUDITOR_SOURCE_PATH
        elif role == "v7_auditor_test":
            path = auditor.AUDITOR_TEST
        else:
            path = tmp_path / f"{role}.py"
            write_bytes(path, b"VALUE = 1\n")
        paths[role] = path
    return {"bound_identities": {role: identity(path) for role, path in paths.items()}}


def fake_inputs() -> dict[str, object]:
    snapshot = {
        "incident_postfailure_state_canonical_sha256": (
            subject.V7_TIMESTAMP_PRECISION_POSTFAILURE_CANONICAL_SHA256
        ),
        "recovered_afterstate_commitment_canonical_sha256": (
            auditor.V5_RECOVERED_AFTERSTATE_COMMITMENT_CANONICAL_SHA256
        ),
        "active_locks": {
            "raw": {"path": "raw/RAW_LAUNCH_ACTIVE.lock", "present": False},
            "decoded": {
                "path": "decoded/DECODE_RECOVERY_ACTIVE.lock",
                "present": False,
            },
        },
        "v7_control_temporary_files": {
            "checked_destination_relative_paths": list(subject.CONTROL_TEMP_RELATIVES),
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
    }
    return {
        "incident": dict(subject.V7_TIMESTAMP_PRECISION_FALSE_REJECT_INCIDENT_IDENTITY),
        "historical_incident_created_utc": "2026-08-10T18:45:00.5807528Z",
        "v5_authorization_created_utc": "2026-08-11T02:16:15.245290Z",
        "v5_review_created_utc": "2026-08-11T02:25:11.379223Z",
        "v5_go_created_utc": "2026-08-11T02:31:34.302619Z",
        "v5_false_reject_incident_created_utc": "2026-08-11T02:51:47.476539Z",
        "incident_created_utc": "2026-08-11T04:22:34.325428Z",
        "v5_authority_base": json.loads(json.dumps(subject.V5_AUTHORITY_BASE)),
        "historical_failed_head_identity_contract": (
            subject.expected_historical_failed_head_identity_contract()
        ),
        "recovered_afterstate_commitment": dict(
            auditor.V5_RECOVERED_AFTERSTATE_COMMITMENT
        ),
        "metadata_only_recovered_closure": {
            "mode": "PATH_TYPE_SIZE_METADATA_ONLY_NO_RECOVERED_BYTES_READ"
        },
        "preaudit_zero_mutation_snapshot": snapshot,
    }


def fake_evidence(code: dict[str, object]) -> dict[str, object]:
    evidence_bound = {
        "v5_authorization": dict(subject.V5_AUTHORITY_BASE["authorization"]),
        "v5_independent_review": dict(subject.V5_AUTHORITY_BASE["independent_review"]),
        "v5_independent_go": dict(subject.V5_AUTHORITY_BASE["independent_go"]),
        "v5_historical_identity_false_reject_incident": dict(
            subject.V6_HISTORICAL_FALSE_REJECT_INCIDENT_IDENTITY
        ),
        "direct_file_failure_incident": dict(
            subject.HISTORICAL_DIRECT_FILE_FAILURE_INCIDENT_IDENTITY
        ),
        "v6_auth_sealer_timestamp_precision_false_reject_incident": dict(
            subject.V7_TIMESTAMP_PRECISION_FALSE_REJECT_INCIDENT_IDENTITY
        ),
        **code["bound_identities"],
    }
    runs = {
        role: {
            "command": auditor._v7_test_command(relative),
            "exit_code": 0,
            "summary": "1 passed in 0.01s",
            "stdout_sha256": "0" * 64,
            "stderr_sha256": "1" * 64,
            "network_guard_installed": True,
            "cacheprovider_disabled": True,
        }
        for role, relative in subject.TEST_RELATIVES.items()
    }
    return {
        "schema_version": 1,
        "artifact_type": "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_V7_TEST_EVIDENCE",
        "status": "PASS_FROZEN_V7_AUDITOR_AND_SEALER_TESTS",
        "created_utc": "2026-08-11T04:30:00.000000Z",
        "bound_identities": evidence_bound,
        "source_compile": {
            "method": "compile_exact_source_no_pyc",
            "result": "PASS",
            "files": [
                code["bound_identities"][role]
                for role in (
                    "v7_auditor", "v7_auditor_test", "v7_sealer", "v7_sealer_test"
                )
            ],
        },
        "pytest_isolation": "ONE_TEST_FILE_PER_CLEAN_NETWORK_DENIED_SUBPROCESS",
        "test_runs": runs,
        "required_test_names": list(auditor.V7_REQUIRED_TEST_NAMES),
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


def patch_authorization_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    inputs: dict[str, object],
) -> None:
    history = inputs["historical_failed_head_identity_contract"]
    monkeypatch.setattr(
        subject,
        "_load_and_validate_v7_timestamp_precision_incident",
        lambda *_args: {
            "identity": inputs["incident"],
            "created_utc": inputs["incident_created_utc"],
            "historical_created_utc": inputs["historical_incident_created_utc"],
        },
    )
    monkeypatch.setattr(
        subject,
        "_load_and_validate_v6_false_reject_incident",
        lambda *_args: {
            "incident": dict(subject.V6_HISTORICAL_FALSE_REJECT_INCIDENT_IDENTITY),
            "incident_payload": {
                "created_utc": inputs["v5_false_reject_incident_created_utc"]
            },
            "historical_contract": history,
            "historical_incident_payload": {
                "created_utc": inputs["historical_incident_created_utc"]
            },
            "failed_head_count": 7,
            "absent_control_path_count": 5,
        },
    )
    monkeypatch.setattr(
        auditor,
        "_v7_validate_timestamp_false_reject_incident",
        lambda *_args: {
            "identity": inputs["incident"],
            "created": {"raw": inputs["incident_created_utc"]},
            "historical_created": {
                "raw": inputs["historical_incident_created_utc"]
            },
        },
    )
    monkeypatch.setattr(
        auditor,
        "_v6_validate_v5_false_reject_incident",
        lambda *_args: {
            "identity": dict(subject.V6_HISTORICAL_FALSE_REJECT_INCIDENT_IDENTITY),
            "payload": {
                "created_utc": inputs["v5_false_reject_incident_created_utc"]
            },
        },
    )
    monkeypatch.setattr(
        auditor,
        "_v6_load_and_validate_v5_authority_base",
        lambda *_args: {
            "base": inputs["v5_authority_base"],
            "commitment": inputs["recovered_afterstate_commitment"],
            "authorization": {
                "created_utc": inputs["v5_authorization_created_utc"]
            },
            "review": {"created_utc": inputs["v5_review_created_utc"]},
            "go": {"created_utc": inputs["v5_go_created_utc"]},
        },
    )
    monkeypatch.setattr(
        auditor,
        "_v6_validate_historical_9e74_payload",
        lambda *_args: history,
    )
    monkeypatch.setattr(
        auditor, "_v6_validate_historical_contract", lambda *_args: None
    )
    monkeypatch.setattr(auditor, "_v7_validate_zero_snapshot", lambda *_args: None)


def metadata_closure_fixture(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, object], dict[str, object], dict[str, Path]]:
    progress_records: list[dict[str, object]] = []
    for index in range(104):
        path = root / f"decoded/recovery_progress/progress_{index:03d}.json"
        write_bytes(path, f"p{index:03d}".encode())
        progress_records.append({**identity(path, root)})

    history: dict[str, dict[str, object]] = {}
    for role in ("decode_mutex_complete", "raw_mutex_complete"):
        path = root / f"decoded/recovery_history/{role}.lock"
        write_bytes(path, role.encode())
        history[role] = identity(path, root)
    postcommit = root / "decoded/recovery_history/postcommit.json"
    write_bytes(postcommit, b"postcommit")

    outputs: dict[str, dict[str, object]] = {}
    for index in range(8):
        path = root / f"decoded/output_{index}.parquet"
        write_bytes(path, f"output-{index}".encode())
        outputs[f"output_{index}"] = identity(path, root)

    failed = "20260810T151057590738Z__pid3536__5f3e0fb20132"
    attempt = auditor.V4_RECOVERY_ATTEMPT_ID
    directories = {
        "failed": root / f"raw/output_transactions/{failed}",
        "failed_staged": root / f"raw/output_transactions/{failed}/staged",
        "failed_raw": root / f"raw/output_transactions/{failed}/staged/raw",
        "attempt": root / f"raw/output_transactions/{attempt}",
        "attempt_staged": root / f"raw/output_transactions/{attempt}/staged",
        "attempt_decoded": root
        / f"raw/output_transactions/{attempt}/staged/decoded",
        "attempt_raw": root / f"raw/output_transactions/{attempt}/staged/raw",
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    files = {
        "plan": directories["attempt"] / "plan.json",
        "commit": directories["attempt"] / "commit.json",
        "remnant_0": directories["failed"] / "plan.json",
        "remnant_1": directories["failed_raw"] / "orphan.tmp",
    }
    for role, path in files.items():
        write_bytes(path, role.encode())
    remnants = [identity(files["remnant_0"], root), identity(files["remnant_1"], root)]
    recovery_payload = {"documented_preplan_remnants": remnants}
    recovery_path = root / "prereg/decode_recovery_authorization_v2.json"
    write_bytes(recovery_path, json.dumps(recovery_payload).encode())
    recovery_record = identity(recovery_path, root)
    monkeypatch.setattr(
        auditor,
        "V4_V2_CONTROL_IDENTITIES",
        {"recovery_authorization_v2": recovery_record},
    )
    authorization = {
        "v2_recovery_controls": {"recovery_authorization_v2": recovery_record}
    }
    afterstate = {
        "attempt_id": attempt,
        "canonical_output_count": 8,
        "canonical_outputs": outputs,
        "transaction": {
            "plan": identity(files["plan"], root),
            "commit": identity(files["commit"], root),
            "total_recursive_files": 4,
        },
        "recovery_history_file_count": 3,
        "completion_locks": history,
        "postcommit_input_audit": identity(postcommit, root),
        "recovery_progress": {
            "file_count": 104,
            "inventory": progress_records,
            "inventory_canonical_sha256": hashlib.sha256(
                subject.canonical_bytes(progress_records)
            ).hexdigest(),
        },
        "raw_active_lock_present": False,
        "decoded_active_lock_present": False,
    }
    return afterstate, authorization, {**files, **directories, "recovery": recovery_path}


def test_sealer_source_is_static_network_zero_compact_and_single_publisher() -> None:
    source = Path(subject.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert imported.isdisjoint(
        {"requests", "urllib", "httpx", "aiohttp", "pandas", "pyarrow", "eccodes"}
    )
    seal_node = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "seal"
    )
    calls = [
        node
        for node in ast.walk(seal_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "publish_json_no_overwrite"
    ]
    assert len(calls) == 1
    build = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_authorization"
    )
    build_source = ast.get_source_segment(source, build) or ""
    for forbidden in (
        '"recovery_progress"', '"canonical_outputs"', '"recovery_transaction"',
        '"recovery_history"', '"v2_recovery_controls"', '"original_v1_provenance"',
    ):
        assert forbidden not in build_source


def test_contract_is_exact_eight_sources_four_runs_twenty_two_temps_eighteen_reports() -> None:
    assert len(subject.BOUND_SOURCE_ROLES) == 8
    assert len(subject.BOUND_IDENTITY_ROLES) == 14
    assert set(subject.BOUND_SOURCE_PATHS) == subject.BOUND_SOURCE_ROLES
    assert set(subject.TEST_RELATIVES) == auditor.V7_TEST_RUN_KEYS
    assert len(subject.TEST_RELATIVES) == 4
    assert tuple(subject.CONTROL_TEMP_RELATIVES) == auditor._v7_control_destination_relatives()
    assert len(subject.CONTROL_TEMP_RELATIVES) == 22
    assert subject.POSTRUN_REPORT_CANDIDATES == auditor.V7_POSTRUN_REPORT_CANDIDATES
    assert len(subject.POSTRUN_REPORT_CANDIDATES) == 18


def test_code_state_binds_frozen_v6_and_executing_v7_roles() -> None:
    state = subject.collect_code_state(subject.REPO)
    assert set(state) == {"bound_identities"}
    assert set(state["bound_identities"]) == subject.BOUND_SOURCE_ROLES
    for role, expected in subject.FROZEN_V6_SOURCE_IDENTITIES.items():
        assert state["bound_identities"][role] == expected
    assert state["bound_identities"]["v7_auditor"] == (
        subject.EXPECTED_V7_AUDITOR_IDENTITY
    )
    assert state["bound_identities"]["v7_auditor_test"] == (
        subject.EXPECTED_V7_AUDITOR_TEST_IDENTITY
    )
    for role in ("v7_auditor", "v7_auditor_test", "v7_sealer", "v7_sealer_test"):
        assert state["bound_identities"][role] == subject.strict_external_identity(
            subject.BOUND_SOURCE_PATHS[role], label=role
        )


def test_v7_review_checks_bind_frozen_v6_and_v5_ast_identity_key() -> None:
    assert "v6_and_v5_production_data_and_replay_core_ast_identical" in auditor.V7_REVIEW_CHECKS
    checks = {key: True for key in auditor.V7_REVIEW_CHECKS}
    assert set(checks) == auditor.V7_REVIEW_CHECKS
    assert all(checks.values())


@pytest.mark.parametrize(
    "role", ("superseded_v6_auditor", "superseded_v6_auditor_test")
)
@pytest.mark.parametrize("field", ("size_bytes", "sha256"))
def test_frozen_v6_auditor_or_test_size_sha_tamper_fails_self_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: str, field: str
) -> None:
    root = control_root(tmp_path)
    code = fake_code_state(tmp_path)
    inputs = fake_inputs()
    evidence = fake_evidence(code)
    patch_authorization_dependencies(monkeypatch, inputs)
    tampered = dict(code["bound_identities"][role])
    tampered[field] = (
        int(tampered[field]) + 1 if field == "size_bytes" else "f" * 64
    )
    code["bound_identities"][role] = tampered
    evidence["bound_identities"][role] = tampered
    with pytest.raises(subject.PostrunAuditSealError, match="frozen V6 source"):
        subject.build_authorization(
            root,
            code,
            inputs,
            evidence,
            created_utc="2026-08-11T05:00:01.000000Z",
        )


@pytest.mark.parametrize("role", ("v7_auditor", "v7_auditor_test"))
@pytest.mark.parametrize("field", ("size_bytes", "sha256"))
def test_executing_v7_auditor_or_test_payload_tamper_fails_self_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    field: str,
) -> None:
    root = control_root(tmp_path)
    code = fake_code_state(tmp_path)
    inputs = fake_inputs()
    evidence = fake_evidence(code)
    patch_authorization_dependencies(monkeypatch, inputs)
    payload = subject.build_authorization(
        root,
        code,
        inputs,
        evidence,
        created_utc="2026-08-11T05:00:01.000000Z",
    )
    tampered = dict(payload[role])
    tampered[field] = (
        int(tampered[field]) + 1 if field == "size_bytes" else "f" * 64
    )
    payload[role] = tampered
    with pytest.raises(subject.PostrunAuditSealError, match="source identity binding"):
        subject.validate_authorization_payload(root, payload, code, inputs)


@pytest.mark.parametrize("role", ("v7_auditor", "v7_auditor_test"))
@pytest.mark.parametrize("field", ("size_bytes", "sha256"))
def test_collect_code_state_rejects_settled_v7_auditor_or_test_tamper(
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    field: str,
) -> None:
    original = subject.strict_external_identity

    def tampered_identity(path: Path, *, label: str) -> dict[str, object]:
        observed = original(path, label=label)
        if label == role:
            observed[field] = (
                int(observed[field]) + 1 if field == "size_bytes" else "f" * 64
            )
        return observed

    monkeypatch.setattr(subject, "strict_external_identity", tampered_identity)
    with pytest.raises(subject.PostrunAuditSealError, match="frozen V7 auditor/test"):
        subject.collect_code_state(subject.REPO)


@pytest.mark.parametrize("role", ("v7_auditor", "v7_auditor_test"))
@pytest.mark.parametrize("field", ("size_bytes", "sha256"))
def test_self_validator_rejects_settled_v7_auditor_or_test_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    field: str,
) -> None:
    root = control_root(tmp_path)
    code = fake_code_state(tmp_path)
    inputs = fake_inputs()
    evidence = fake_evidence(code)
    patch_authorization_dependencies(monkeypatch, inputs)
    tampered = dict(code["bound_identities"][role])
    tampered[field] = (
        int(tampered[field]) + 1 if field == "size_bytes" else "f" * 64
    )
    code["bound_identities"][role] = tampered
    evidence["bound_identities"][role] = tampered
    compile_index = (
        "v7_auditor", "v7_auditor_test", "v7_sealer", "v7_sealer_test"
    ).index(role)
    evidence["source_compile"]["files"][compile_index] = tampered
    with pytest.raises(subject.PostrunAuditSealError, match="frozen V7 auditor/test"):
        subject.build_authorization(
            root,
            code,
            inputs,
            evidence,
            created_utc="2026-08-11T05:00:01.000000Z",
        )


def test_external_identity_rejects_relative_and_link(tmp_path: Path) -> None:
    with pytest.raises(subject.PostrunAuditSealError, match="absolute"):
        subject.strict_external_identity(Path("relative.py"), label="relative")
    target = tmp_path / "target.py"
    write_bytes(target, b"VALUE=1\n")
    link = tmp_path / "link.py"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink unavailable")
    with pytest.raises(subject.PostrunAuditSealError, match="symlink|junction"):
        subject.strict_external_identity(link.absolute(), label="link")


def test_duplicate_json_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    write_bytes(path, b'{"a":1,"a":2}')
    with pytest.raises(subject.PostrunAuditSealError, match="duplicate JSON key"):
        subject.load_json_control(path, label="duplicate")


def test_preimport_guard_rejects_top_level_io(tmp_path: Path) -> None:
    source = tmp_path / "auditor.py"
    write_bytes(source, b"from pathlib import Path\nVALUE=Path('x').read_bytes()\n")
    with pytest.raises(RuntimeError, match="unapproved import-time call"):
        subject._preimport_auditor_source_guard(source)


@pytest.mark.parametrize("role", sorted(subject.BOUND_SOURCE_ROLES))
def test_bytecode_competitor_is_rejected_for_every_bound_source(
    tmp_path: Path, role: str
) -> None:
    code = fake_code_state(tmp_path)
    replacement = tmp_path / f"replacement_{role}.py"
    write_bytes(replacement, b"VALUE = 1\n")
    code["bound_identities"][role] = identity(replacement)
    record = code["bound_identities"][role]
    source = Path(str(record["path"]))
    write_bytes(source.with_suffix(".pyc"), b"timestamp-valid-looking-bytecode")
    with pytest.raises(subject.PostrunAuditSealError, match="bytecode/cache competitor"):
        subject._require_bound_bytecode_absent(code["bound_identities"])


@pytest.mark.parametrize("competitor", ("package", "mixed_pyd"))
def test_auditor_same_stem_origin_competitor_fails(
    tmp_path: Path, competitor: str
) -> None:
    source = tmp_path / "audit_v6.py"
    write_bytes(source, b"VALUE=1\n")
    if competitor == "package":
        (tmp_path / source.stem).mkdir()
    else:
        write_bytes(tmp_path / f"{source.stem}.PYD", b"binary")
    with pytest.raises(RuntimeError, match="same-stem source-origin competitor"):
        subject._require_auditor_source_competitors_absent(source)


def test_namespace_pre_and_postpublication_are_exact(tmp_path: Path) -> None:
    root = control_root(tmp_path)
    populate_historical_namespace(root)
    pre = subject.require_prepublication_control_state(root)
    assert pre["authorization_present"] is False
    write_bytes(root / subject.AUTH_RELATIVE)
    post = subject.require_postpublication_control_state(root)
    assert post["authorization_present"] is True
    assert not os.path.lexists(root / subject.REVIEW_RELATIVE)
    assert not os.path.lexists(root / subject.GO_RELATIVE)


@pytest.mark.parametrize(
    "relative",
    (
        "prereg/decode_recovery_postrun_audit_shadow.json",
        "independent_redteam/.TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_EVIL.json",
        "incidents/track_a_decode_recovery_postrun_auditor_alias.json",
    ),
)
def test_unbound_namespace_alias_fails(tmp_path: Path, relative: str) -> None:
    root = control_root(tmp_path)
    populate_historical_namespace(root)
    write_bytes(root / relative)
    with pytest.raises(subject.PostrunAuditSealError, match="namespace inventory"):
        subject.require_prepublication_control_state(root)


@pytest.mark.parametrize(
    "relative",
    (auditor.FROZEN_V3_CANONICAL_GO_RELATIVE, auditor.FROZEN_V4_CANONICAL_GO_RELATIVE),
)
def test_canonical_v3_or_v4_go_presence_fails(tmp_path: Path, relative: str) -> None:
    root = control_root(tmp_path)
    populate_historical_namespace(root)
    write_bytes(root / relative)
    with pytest.raises(subject.PostrunAuditSealError, match="canonical V[34] GO"):
        subject.require_prepublication_control_state(root)


@pytest.mark.parametrize("relative", subject.CONTROL_TEMP_RELATIVES)
def test_all_twenty_two_control_destinations_scan_temporary_siblings(
    tmp_path: Path, relative: str
) -> None:
    root = control_root(tmp_path)
    destination = root / relative
    temporary = destination.with_name(f".{destination.name}.tmp.7.deadbeef")
    write_bytes(temporary)
    assert temporary.relative_to(root).as_posix() in subject.control_temporary_files(root)
    with pytest.raises(subject.PostrunAuditSealError, match="temporary"):
        subject.build_zero_snapshot(root)


@pytest.mark.parametrize("relative", subject.POSTRUN_REPORT_CANDIDATES)
def test_every_v2_to_v7_report_candidate_fails_zero_snapshot(
    tmp_path: Path, relative: str
) -> None:
    root = control_root(tmp_path)
    write_bytes(root / relative)
    with pytest.raises(subject.PostrunAuditSealError, match="output file exists"):
        subject.build_zero_snapshot(root)


def test_metadata_closure_accepts_exact_tree_without_reading_recovered_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = control_root(tmp_path)
    afterstate, authorization, paths = metadata_closure_fixture(root, monkeypatch)
    original = subject.sha256_file
    hashed: list[Path] = []

    def guarded(path: Path) -> str:
        hashed.append(path)
        assert path == paths["recovery"]
        return original(path)

    monkeypatch.setattr(subject, "sha256_file", guarded)
    closure = subject.validate_recovered_filesystem_closure(
        root, afterstate, authorization
    )
    assert closure == {
        "mode": "PATH_TYPE_SIZE_METADATA_ONLY_NO_RECOVERED_BYTES_READ",
        "recovery_progress_file_count": 104,
        "recovery_history_file_count": 3,
        "transaction_recursive_file_count": 4,
        "transaction_recursive_directory_count": 7,
        "canonical_output_count": 8,
        "recovered_artifact_files_opened": 0,
    }
    assert hashed == [paths["recovery"]]


@pytest.mark.parametrize("substitution", ("file", "directory"))
def test_same_count_transaction_path_substitution_fails_metadata_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, substitution: str
) -> None:
    root = control_root(tmp_path)
    afterstate, authorization, paths = metadata_closure_fixture(root, monkeypatch)
    if substitution == "file":
        source = paths["remnant_1"]
        replacement = source.with_name("same_size_substitute.tmp")
    else:
        source = paths["attempt_decoded"]
        replacement = source.with_name("renamed_decoded")
    source.rename(replacement)
    with pytest.raises(subject.PostrunAuditSealError, match="transaction recursive"):
        subject.validate_recovered_filesystem_closure(root, afterstate, authorization)


def test_remnant_same_path_size_tamper_fails_metadata_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = control_root(tmp_path)
    afterstate, authorization, paths = metadata_closure_fixture(root, monkeypatch)
    paths["remnant_0"].write_bytes(b"different-size")
    with pytest.raises(subject.PostrunAuditSealError, match="metadata size"):
        subject.validate_recovered_filesystem_closure(root, afterstate, authorization)


def test_compact_commitment_is_exact_ten_fields_and_470_bytes() -> None:
    commitment = auditor.V5_RECOVERED_AFTERSTATE_COMMITMENT
    assert len(commitment) == 10
    encoded = subject.canonical_bytes(commitment)
    assert len(encoded) == auditor.V5_RECOVERED_AFTERSTATE_COMMITMENT_CANONICAL_BYTES
    assert hashlib.sha256(encoded).hexdigest() == (
        auditor.V5_RECOVERED_AFTERSTATE_COMMITMENT_CANONICAL_SHA256
    )
    assert "inventory" not in commitment


def test_build_authorization_is_exact_auth34_and_never_embeds_full_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = control_root(tmp_path)
    code = fake_code_state(tmp_path)
    inputs = fake_inputs()
    evidence = fake_evidence(code)
    patch_authorization_dependencies(monkeypatch, inputs)
    payload = subject.build_authorization(
        root,
        code,
        inputs,
        evidence,
        created_utc="2026-08-11T05:00:01.000000Z",
    )
    assert set(payload) == auditor.V7_AUTH_KEYS
    assert len(payload) == 34
    assert payload["schema_version"] == 7
    assert payload["historical_failed_head_identity_contract"] == inputs[
        "historical_failed_head_identity_contract"
    ]
    assert payload["recovered_afterstate_commitment"] == (
        auditor.V5_RECOVERED_AFTERSTATE_COMMITMENT
    )
    forbidden = {
        "recovery_progress", "canonical_outputs", "recovery_transaction",
        "recovery_history", "v2_recovery_controls", "original_v1_provenance",
    }
    assert forbidden.isdisjoint(payload)
    assert '"inventory"' not in json.dumps(payload, sort_keys=True)
    assert set(payload["test_evidence"]["bound_identities"]) == (
        subject.BOUND_IDENTITY_ROLES
    )
    assert len(payload["test_evidence"]["bound_identities"]) == 14


def test_actual_root_collect_build_and_self_validator_passes_read_only() -> None:
    root = subject.ROOT_DEFAULT.resolve()
    assert root.is_dir()
    before = {
        relative: os.path.lexists(root / relative)
        for relative in (subject.AUTH_RELATIVE, subject.REVIEW_RELATIVE, subject.GO_RELATIVE)
    }
    assert before == {
        subject.AUTH_RELATIVE: False,
        subject.REVIEW_RELATIVE: False,
        subject.GO_RELATIVE: False,
    }
    code = subject.collect_code_state(subject.REPO)
    inputs = subject.collect_root_inputs(root, code)
    payload = subject.build_authorization(
        root,
        code,
        inputs,
        fake_evidence(code),
        created_utc="2026-08-11T05:00:01.000000Z",
    )
    subject.validate_authorization_payload(root, payload, code, inputs)
    assert payload["incident"] == subject.V7_TIMESTAMP_PRECISION_FALSE_REJECT_INCIDENT_IDENTITY
    assert inputs["historical_incident_created_utc"] == "2026-08-10T18:45:00.5807528Z"
    assert {
        relative: os.path.lexists(root / relative)
        for relative in (subject.AUTH_RELATIVE, subject.REVIEW_RELATIVE, subject.GO_RELATIVE)
    } == before


def test_v7_timestamp_incident_must_strictly_predate_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = control_root(tmp_path)
    code = fake_code_state(tmp_path)
    inputs = fake_inputs()
    patch_authorization_dependencies(monkeypatch, inputs)
    with pytest.raises(subject.PostrunAuditSealError, match="chronology"):
        subject.build_authorization(
            root,
            code,
            inputs,
            fake_evidence(code),
            created_utc=str(inputs["incident_created_utc"]),
        )


@pytest.mark.parametrize(
    "mutation",
    ("incident", "base", "history_digest", "history_mode", "source", "evidence"),
)
def test_v7_authorization_compact_binding_tamper_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    root = control_root(tmp_path)
    code = fake_code_state(tmp_path)
    inputs = fake_inputs()
    patch_authorization_dependencies(monkeypatch, inputs)
    payload = subject.build_authorization(
        root,
        code,
        inputs,
        fake_evidence(code),
        created_utc="2026-08-11T05:00:01.000000Z",
    )
    if mutation == "incident":
        payload["incident"] = {"tampered": True}
    elif mutation == "base":
        payload["v5_authority_base"] = {"tampered": True}
    elif mutation == "history_digest":
        payload["historical_failed_head_identity_contract"] = json.loads(
            json.dumps(payload["historical_failed_head_identity_contract"])
        )
        payload["historical_failed_head_identity_contract"][
            "historical_incident_contract"
        ]["failed_head_identities"]["canonical_sha256"] = "0" * 64
    elif mutation == "history_mode":
        payload["historical_failed_head_identity_contract"] = dict(
            payload["historical_failed_head_identity_contract"]
        )
        payload["historical_failed_head_identity_contract"]["validation_mode"] = (
            "CURRENT_ROLE_DEREFERENCE"
        )
    elif mutation == "source":
        payload["v7_sealer"] = {"tampered": True}
    else:
        payload["test_evidence"] = {"tampered": True}
    with pytest.raises(subject.PostrunAuditSealError):
        subject.validate_authorization_payload(root, payload, code, inputs)


@pytest.mark.parametrize(
    "chronology",
    (
        "historical_incident_not_before_v5_authorization",
        "v5_authorization_not_before_review",
        "v5_review_not_before_go",
        "v5_go_not_before_v5_false_reject_incident",
        "v5_false_reject_incident_not_before_timestamp_incident",
        "authorization_not_after_incident",
        "evidence_after_authorization",
    ),
)
def test_authorization_and_evidence_chronology_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, chronology: str
) -> None:
    root = control_root(tmp_path)
    code = fake_code_state(tmp_path)
    inputs = fake_inputs()
    evidence = fake_evidence(code)
    patch_authorization_dependencies(monkeypatch, inputs)
    created_utc = "2026-08-11T05:00:01.000000Z"
    if chronology == "historical_incident_not_before_v5_authorization":
        inputs["historical_incident_created_utc"] = inputs[
            "v5_authorization_created_utc"
        ]
    elif chronology == "v5_authorization_not_before_review":
        inputs["v5_authorization_created_utc"] = inputs["v5_review_created_utc"]
    elif chronology == "v5_review_not_before_go":
        inputs["v5_review_created_utc"] = inputs["v5_go_created_utc"]
    elif chronology == "v5_go_not_before_v5_false_reject_incident":
        inputs["v5_go_created_utc"] = inputs["v5_false_reject_incident_created_utc"]
    elif chronology == "v5_false_reject_incident_not_before_timestamp_incident":
        inputs["v5_false_reject_incident_created_utc"] = inputs[
            "incident_created_utc"
        ]
    elif chronology == "authorization_not_after_incident":
        created_utc = str(inputs["incident_created_utc"])
    else:
        evidence["created_utc"] = "2026-08-11T05:00:02.000000Z"
    with pytest.raises(
        subject.PostrunAuditSealError,
        match="chronology|predates|postdates|evidence",
    ):
        subject.build_authorization(
            root,
            code,
            inputs,
            evidence,
            created_utc=created_utc,
        )


@pytest.mark.parametrize(
    ("field", "tampered_value"),
    (
        ("historical_incident_created_utc", "2026-08-10T18:05:00Z"),
        ("v5_authorization_created_utc", "2026-08-11T02:17:15.245290Z"),
        ("v5_review_created_utc", "2026-08-11T02:26:11.379223Z"),
        ("v5_go_created_utc", "2026-08-11T02:32:34.302619Z"),
        ("v5_false_reject_incident_created_utc", "2026-08-11T02:52:47.476539Z"),
        ("incident_created_utc", "2026-08-11T04:23:34.325428Z"),
    ),
)
def test_helper_backed_chronology_timestamp_tamper_fails_even_when_order_is_valid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    tampered_value: str,
) -> None:
    root = control_root(tmp_path)
    code = fake_code_state(tmp_path)
    helper_inputs = fake_inputs()
    patch_authorization_dependencies(monkeypatch, helper_inputs)
    payload = subject.build_authorization(
        root,
        code,
        helper_inputs,
        fake_evidence(code),
        created_utc="2026-08-11T05:00:01.000000Z",
    )
    declared_inputs = dict(helper_inputs)
    declared_inputs[field] = tampered_value
    with pytest.raises(subject.PostrunAuditSealError, match="raw timestamp binding"):
        subject.validate_authorization_payload(root, payload, code, declared_inputs)


def test_equal_tick_raw_timestamp_substitution_fails_helper_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = control_root(tmp_path)
    code = fake_code_state(tmp_path)
    helper_inputs = fake_inputs()
    patch_authorization_dependencies(monkeypatch, helper_inputs)
    payload = subject.build_authorization(
        root,
        code,
        helper_inputs,
        fake_evidence(code),
        created_utc="2026-08-11T05:00:01.000000Z",
    )
    substituted = dict(helper_inputs)
    original = str(helper_inputs["v5_authorization_created_utc"])
    substituted["v5_authorization_created_utc"] = "2026-08-11T02:16:15.24529Z"
    assert original != substituted["v5_authorization_created_utc"]
    assert subject._exact_utc_z_100ns_ticks(original, label="original") == (
        subject._exact_utc_z_100ns_ticks(
            substituted["v5_authorization_created_utc"], label="substituted"
        )
    )
    with pytest.raises(subject.PostrunAuditSealError, match="raw timestamp binding"):
        subject.validate_authorization_payload(root, payload, code, substituted)


@pytest.mark.parametrize(
    "mutation",
    (
        "missing", "extra", "schema", "status", "attempt", "incident",
        "base", "history", "source", "recovery",
        "commitment", "snapshot", "runtime", "command", "review", "evidence",
    ),
)
def test_authorization_schema_and_binding_tamper_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    root = control_root(tmp_path)
    code = fake_code_state(tmp_path)
    inputs = fake_inputs()
    patch_authorization_dependencies(monkeypatch, inputs)
    payload = subject.build_authorization(
        root,
        code,
        inputs,
        fake_evidence(code),
        created_utc="2026-08-11T05:00:01.000000Z",
    )
    if mutation == "missing":
        payload.pop("historical_failed_head_identity_contract")
    elif mutation == "extra":
        payload["unknown"] = True
    elif mutation == "schema":
        payload["schema_version"] = 6
    elif mutation == "status":
        payload["status"] = "PASS"
    elif mutation == "attempt":
        payload["audit_attempt_id"] = "postrun_audit_v7__wildcard"
    elif mutation == "incident":
        payload["incident"] = {"tampered": True}
    elif mutation == "base":
        payload["v5_authority_base"] = {"tampered": True}
    elif mutation == "history":
        payload["historical_failed_head_identity_contract"] = {"tampered": True}
    elif mutation == "source":
        payload["v7_auditor"] = {"tampered": True}
    elif mutation == "recovery":
        payload["recovery_attempt_id"] = "wrong"
    elif mutation == "commitment":
        payload["recovered_afterstate_commitment"] = {"tampered": True}
    elif mutation == "snapshot":
        payload["preaudit_zero_mutation_snapshot"] = {"tampered": True}
    elif mutation == "runtime":
        payload["runtime_identity_sha256"] = "f" * 64
    elif mutation == "command":
        payload["required_command"] = ["python", "audit.py"]
    elif mutation == "review":
        payload["independent_review_required"] = False
    else:
        payload["test_evidence"] = {"tampered": True}
    with pytest.raises(subject.PostrunAuditSealError):
        subject.validate_authorization_payload(root, payload, code, inputs)


def test_required_command_matches_auditor_exactly(tmp_path: Path) -> None:
    root = control_root(tmp_path)
    assert subject.required_audit_command(root) == auditor._v7_required_command(
        root, root / subject.AUTH_RELATIVE, root / subject.GO_RELATIVE
    )


def test_v7_timestamp_incident_base_and_historical_contract_are_required_during_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = control_root(tmp_path)
    populate_historical_namespace(root)
    code = fake_code_state(tmp_path)
    inputs = fake_inputs()
    calls: list[str] = []
    afterstate = {"stub": True}
    local_timestamp_state = {
        "identity": inputs["incident"],
        "created_utc": inputs["incident_created_utc"],
        "historical_created_utc": inputs["historical_incident_created_utc"],
    }
    local_history_state = {
        "incident": dict(subject.V6_HISTORICAL_FALSE_REJECT_INCIDENT_IDENTITY),
        "incident_payload": {
            "created_utc": inputs["v5_false_reject_incident_created_utc"]
        },
        "historical_contract": inputs["historical_failed_head_identity_contract"],
        "historical_incident_payload": {
            "created_utc": inputs["historical_incident_created_utc"]
        },
        "failed_head_count": 7,
        "absent_control_path_count": 5,
    }

    def local_timestamp(*_args: object) -> dict[str, object]:
        calls.append("local_timestamp")
        return local_timestamp_state

    def local_history(*_args: object) -> dict[str, object]:
        calls.append("local_history")
        return local_history_state

    def auditor_timestamp(*_args: object) -> dict[str, object]:
        calls.append("auditor_timestamp")
        return {
            "identity": inputs["incident"],
            "created": {"raw": inputs["incident_created_utc"]},
            "historical_created": {
                "raw": inputs["historical_incident_created_utc"]
            },
        }

    def auditor_history(*_args: object) -> dict[str, object]:
        calls.append("auditor_history")
        return {
            "identity": dict(subject.V6_HISTORICAL_FALSE_REJECT_INCIDENT_IDENTITY)
        }

    base_state = {
        "base": inputs["v5_authority_base"],
        "authorization": {"created_utc": inputs["v5_authorization_created_utc"]},
        "review": {"created_utc": inputs["v5_review_created_utc"]},
        "go": {"created_utc": inputs["v5_go_created_utc"]},
        "recovered_afterstate": afterstate,
        "v4_base": {"authorization": {}},
    }
    monkeypatch.setattr(subject, "validate_control_namespace", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        subject, "_load_and_validate_v7_timestamp_precision_incident", local_timestamp
    )
    monkeypatch.setattr(
        subject, "_load_and_validate_v6_false_reject_incident", local_history
    )
    monkeypatch.setattr(
        auditor, "_v7_validate_timestamp_false_reject_incident", auditor_timestamp
    )
    monkeypatch.setattr(
        auditor, "_v6_validate_v5_false_reject_incident", auditor_history
    )
    monkeypatch.setattr(auditor, "_v6_load_and_validate_v5_authority_base", lambda *_args: base_state)
    monkeypatch.setattr(
        auditor,
        "_v6_validate_historical_9e74_payload",
        lambda *_args: inputs["historical_failed_head_identity_contract"],
    )
    monkeypatch.setattr(auditor, "_v6_validate_historical_contract", lambda *_args: None)
    monkeypatch.setattr(
        subject,
        "_recovered_afterstate_commitment",
        lambda _state: dict(auditor.V5_RECOVERED_AFTERSTATE_COMMITMENT),
    )
    monkeypatch.setattr(
        subject,
        "validate_recovered_filesystem_closure",
        lambda *_args: {"mode": "PATH_TYPE_SIZE_METADATA_ONLY_NO_RECOVERED_BYTES_READ"},
    )
    monkeypatch.setattr(
        subject, "build_zero_snapshot", lambda _root: fake_inputs()["preaudit_zero_mutation_snapshot"]
    )
    result = subject.collect_root_inputs(root, code)
    assert calls == [
        "local_timestamp", "local_history", "auditor_timestamp", "auditor_history"
    ]
    assert result["incident"] == inputs["incident"]
    assert result["v5_authority_base"] == inputs["v5_authority_base"]
    assert result["historical_failed_head_identity_contract"] == inputs[
        "historical_failed_head_identity_contract"
    ]


def test_isolated_evidence_uses_four_exact_guarded_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code = fake_code_state(tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(subject, "require_canonical_module_entrypoint", lambda _root: None)
    monkeypatch.setattr(subject, "collect_code_state", lambda _root: code)
    monkeypatch.setattr(
        subject, "_source_function_names", lambda _path: set(auditor.V7_REQUIRED_TEST_NAMES)
    )
    monkeypatch.setattr(
        subject,
        "_evidence_bound_identities",
        lambda *_args: fake_evidence(code)["bound_identities"],
    )

    def completed(command: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="1 passed in 0.01s\n", stderr="")

    monkeypatch.setattr(subject.subprocess, "run", completed)
    evidence = subject.run_immutable_test_evidence(tmp_path, subject.REPO, code)
    assert calls == [auditor._v7_test_command(path) for path in subject.TEST_RELATIVES.values()]
    assert set(evidence["test_runs"]) == auditor.V7_TEST_RUN_KEYS
    assert len(evidence["bound_identities"]) == 14
    assert evidence["source_compile"]["files"] == [
        code["bound_identities"][role]
        for role in ("v7_auditor", "v7_auditor_test", "v7_sealer", "v7_sealer_test")
    ]
    assert evidence["all_exit_codes_zero"] is True


def test_isolated_evidence_failure_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code = fake_code_state(tmp_path)
    monkeypatch.setattr(subject, "require_canonical_module_entrypoint", lambda _root: None)
    monkeypatch.setattr(subject, "collect_code_state", lambda _root: code)
    monkeypatch.setattr(
        subject, "_source_function_names", lambda _path: set(auditor.V7_REQUIRED_TEST_NAMES)
    )
    monkeypatch.setattr(
        subject,
        "_evidence_bound_identities",
        lambda *_args: fake_evidence(code)["bound_identities"],
    )
    monkeypatch.setattr(
        subject.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1, stdout="1 failed in 0.01s\n", stderr="boom"
        ),
    )
    with pytest.raises(subject.PostrunAuditSealError, match="isolated test failed"):
        subject.run_immutable_test_evidence(tmp_path, subject.REPO, code)


@pytest.mark.parametrize("summary", ("1 passed, 1 failed", "1 passed, 1 error"))
def test_local_evidence_validator_rejects_mixed_failure_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    summary: str,
) -> None:
    root = control_root(tmp_path)
    code = fake_code_state(tmp_path)
    inputs = fake_inputs()
    evidence = fake_evidence(code)
    evidence["test_runs"]["v7_sealer_tests"]["summary"] = summary
    patch_authorization_dependencies(monkeypatch, inputs)
    monkeypatch.setattr(auditor, "_v7_validate_test_evidence", lambda *_args, **_kwargs: None)
    with pytest.raises(subject.PostrunAuditSealError, match="run invalid"):
        subject.build_authorization(
            root,
            code,
            inputs,
            evidence,
            created_utc="2026-08-11T05:00:01.000000Z",
        )


def test_auditor_evidence_validator_is_independently_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = control_root(tmp_path)
    code = fake_code_state(tmp_path)
    inputs = fake_inputs()
    patch_authorization_dependencies(monkeypatch, inputs)

    def reject(*_args: object, **_kwargs: object) -> None:
        raise auditor.AuditFailure("independent rejection")

    monkeypatch.setattr(auditor, "_v7_validate_test_evidence", reject)
    with pytest.raises(
        subject.PostrunAuditSealError, match="independent auditor test-evidence rejection"
    ):
        subject.build_authorization(
            root,
            code,
            inputs,
            fake_evidence(code),
            created_utc="2026-08-11T05:00:01.000000Z",
        )


def test_publish_no_overwrite_preserves_existing(tmp_path: Path) -> None:
    destination = tmp_path / "authorization.json"
    write_bytes(destination, b"existing")
    with pytest.raises(subject.PostrunAuditSealError, match="already exists"):
        subject.publish_json_no_overwrite(destination, {"new": True})
    assert destination.read_bytes() == b"existing"


def test_publish_race_never_overwrites_attacker_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "authorization.json"

    def racing_link(_source: Path, target: Path, **_kwargs: object) -> None:
        Path(target).write_bytes(b"attacker")
        raise FileExistsError("race")

    monkeypatch.setattr(subject.os, "link", racing_link)
    with pytest.raises(subject.PostrunAuditSealError, match="create-if-absent"):
        subject.publish_json_no_overwrite(destination, {"authorized": True})
    assert destination.read_bytes() == b"attacker"
    assert not list(tmp_path.glob(".*.tmp.*"))


def test_seal_publishes_authorization_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = control_root(tmp_path)
    populate_historical_namespace(root)
    code = fake_code_state(tmp_path)
    inputs = fake_inputs()
    payload = {"schema_version": 7, "artifact_type": "TEST_AUTH"}
    monkeypatch.setattr(subject, "require_canonical_module_entrypoint", lambda _root: None)
    monkeypatch.setattr(subject, "install_entry_guard", lambda: None)
    monkeypatch.setattr(subject, "collect_code_state", lambda _root: code)
    monkeypatch.setattr(subject, "collect_root_inputs", lambda *_args: inputs)
    monkeypatch.setattr(subject, "run_immutable_test_evidence", lambda *_args: {"pass": True})
    monkeypatch.setattr(subject, "build_authorization", lambda *_args, **_kwargs: payload)
    calls: list[Path] = []
    original = subject.publish_json_no_overwrite

    def spy(path: Path, value: dict[str, object]) -> None:
        calls.append(path)
        original(path, value)

    monkeypatch.setattr(subject, "publish_json_no_overwrite", spy)
    report = subject.seal(root, tmp_path)
    assert report["authorization_files_published"] == 1
    assert calls == [root / subject.AUTH_RELATIVE]
    assert not os.path.lexists(root / subject.REVIEW_RELATIVE)
    assert not os.path.lexists(root / subject.GO_RELATIVE)


def test_postpublication_failure_reports_one_without_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = control_root(tmp_path)
    populate_historical_namespace(root)
    code = fake_code_state(tmp_path)
    inputs = fake_inputs()
    payload = {"schema_version": 7, "artifact_type": "TEST_AUTH"}
    monkeypatch.setattr(subject, "require_canonical_module_entrypoint", lambda _root: None)
    monkeypatch.setattr(subject, "install_entry_guard", lambda: None)
    monkeypatch.setattr(subject, "collect_code_state", lambda _root: code)
    calls = 0

    def root_inputs(*_args: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if os.path.lexists(root / subject.AUTH_RELATIVE):
            raise subject.PostrunAuditSealError("forced postpublication failure")
        return inputs

    monkeypatch.setattr(subject, "collect_root_inputs", root_inputs)
    monkeypatch.setattr(subject, "run_immutable_test_evidence", lambda *_args: {"pass": True})
    monkeypatch.setattr(subject, "build_authorization", lambda *_args, **_kwargs: payload)
    rc = subject.main(["--root", str(root)])
    report = json.loads(capsys.readouterr().out)
    assert rc == 3
    assert report["authorization_files_published"] == 1
    assert report["authorization_rollback_attempted"] is False
    assert (root / subject.AUTH_RELATIVE).is_file()
    assert calls >= 3


def test_existing_authorization_fails_before_code_or_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = control_root(tmp_path)
    destination = root / subject.AUTH_RELATIVE
    write_bytes(destination, b"immutable")
    monkeypatch.setattr(subject, "require_canonical_module_entrypoint", lambda _root: None)
    monkeypatch.setattr(subject, "install_entry_guard", lambda: None)
    monkeypatch.setattr(
        subject,
        "collect_code_state",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must not run")),
    )
    with pytest.raises(subject.PostrunAuditSealError, match="already exists"):
        subject.seal(root, tmp_path)
    assert destination.read_bytes() == b"immutable"


def test_direct_file_entrypoint_is_rejected(tmp_path: Path) -> None:
    root = control_root(tmp_path)
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.pop("PYTHONPYCACHEPREFIX", None)
    completed = subprocess.run(
        [sys.executable, "-B", str(Path(subject.__file__).resolve()), "--root", str(root)],
        cwd=subject.REPO,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    report = json.loads(completed.stdout)
    assert completed.returncode == 2
    assert "canonical" in report["error"]
    assert not os.path.lexists(root / subject.AUTH_RELATIVE)


def test_network_route_is_unconditionally_denied() -> None:
    with pytest.raises(subject._NetworkDenied, match="network access is forbidden"):
        subject._network_forbidden(("127.0.0.1", 1))


def _historical_zero_state() -> dict[str, object]:
    return {
        "absent_control_paths": list(subject.HISTORICAL_V1_ABSENT_CONTROL_PATHS),
        "canonical_recovery_outputs_absent": True,
        "recovery_active_locks_absent": True,
        "seal_temporary_files_absent": True,
        "matching_bootstrap_pyc_files": 0,
        "only_original_failed_output_transaction_present": True,
    }


def _historical_drift_records() -> list[dict[str, object]]:
    roles = (
        "recovery_runner",
        "recovery_sealer",
        "recovery_sealer_test",
        "planned_postrun_auditor",
        "planned_postrun_auditor_test",
    )
    current_paths = (
        "scripts/run_noaa_gfs_multiseason_decode_recovery_v2.py",
        "scripts/seal_noaa_gfs_multiseason_decode_recovery_v2.py",
        "tests/test_seal_noaa_gfs_multiseason_decode_recovery_v2.py",
        "scripts/audit_noaa_gfs_multiseason_raw_postrun_v2.py",
        "tests/test_noaa_gfs_multiseason_raw_postrun_v2.py",
    )
    records: list[dict[str, object]] = []
    for role, current_path in zip(roles, current_paths):
        historical = dict(subject.HISTORICAL_FAILED_HEAD_IDENTITIES[role])
        historical["path"] = Path(str(historical["path"])).relative_to(
            subject.REPO
        ).as_posix()
        records.append(
            {
                "historical": historical,
                "incorrect_current_path": current_path,
                "role": role,
            }
        )
    return records


def _synthetic_false_reject_incidents(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, dict[str, object]]:
    historical_path = root / str(
        subject.HISTORICAL_DIRECT_FILE_FAILURE_INCIDENT_IDENTITY["path"]
    )
    historical_payload: dict[str, object] = {
        "artifact_type": (
            "TRACK_A_DECODE_RECOVERY_SEAL_DIRECT_FILE_PICKLE_FAILURE_INCIDENT"
        ),
        "schema_version": 1,
        "status": "FAILED_BEFORE_PUBLICATION_RETRY_FORBIDDEN_PENDING_REFREEZE",
        "failed_head_identities": json.loads(
            json.dumps(subject.HISTORICAL_FAILED_HEAD_IDENTITIES)
        ),
        "verified_zero_state_after_failure": _historical_zero_state(),
    }
    write_json(historical_path, historical_payload)
    historical_identity = identity(historical_path, root)
    monkeypatch.setattr(
        subject,
        "HISTORICAL_DIRECT_FILE_FAILURE_INCIDENT_IDENTITY",
        historical_identity,
    )

    false_reject_path = root / str(
        subject.V6_HISTORICAL_FALSE_REJECT_INCIDENT_IDENTITY["path"]
    )
    digest = subject.HISTORICAL_CONTRACT_DIGESTS
    false_reject_payload: dict[str, object] = {
        "artifact_type": (
            "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_V5_"
            "HISTORICAL_IDENTITY_FALSE_REJECT_INCIDENT"
        ),
        "authority_scope": {
            "documentary_only": True,
            "v5_retry_authorized": False,
            "v6_authorized": False,
        },
        "created_utc": "2026-08-11T02:51:47.476539Z",
        "exact_mismatch_set": {
            "failed_head_role_path_drift": {
                "matching_roles": ["recovery_core", "recovery_bootstrap"],
                "mismatches": _historical_drift_records(),
            },
            "zero_state_control_version_drift": {
                "historical_v1_absent_paths": list(
                    subject.HISTORICAL_V1_ABSENT_CONTROL_PATHS
                ),
                "incorrect_current_v2_expected_paths": list(
                    subject.HISTORICAL_V2_INCORRECT_CONTROL_PATHS
                ),
            },
        },
        "execution_boundary": {},
        "failed_execution": {},
        "historical_incident": {
            "identity": historical_identity,
            "failed_heads": {
                "size_bytes": digest["failed_head_identities"][
                    "canonical_size_bytes"
                ],
                "sha256": digest["failed_head_identities"]["canonical_sha256"],
            },
            "zero_state": {
                "size_bytes": digest["verified_zero_state_after_failure"][
                    "canonical_size_bytes"
                ],
                "sha256": digest["verified_zero_state_after_failure"][
                    "canonical_sha256"
                ],
            },
            "absent_control_paths": {
                "size_bytes": digest["absent_control_paths"][
                    "canonical_size_bytes"
                ],
                "sha256": digest["absent_control_paths"]["canonical_sha256"],
            },
        },
        "immutable_v5_chain": {},
        "postfailure_state": {},
        "prohibitions": {},
        "required_v6_supersession": {
            "code": [
                "scripts/audit_noaa_gfs_multiseason_raw_postrun_v6.py",
                "tests/test_noaa_gfs_multiseason_raw_postrun_v6.py",
                "scripts/seal_noaa_gfs_multiseason_postrun_audit_v6.py",
                "tests/test_seal_noaa_gfs_multiseason_postrun_audit_v6.py",
            ],
            "controls": [
                "prereg/decode_recovery_postrun_audit_authorization_v6.json",
                (
                    "independent_redteam/"
                    "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_REVIEW_V6.json"
                ),
                (
                    "independent_redteam/"
                    "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_GO_V6.json"
                ),
            ],
            "fix": [
                "freeze exact historical heads and V1 zero-state",
                "forbid current-role/control substitution",
                "validate 9e74 before parquet/schema/data/replay",
                "add production-9e74 positive and head/zero-state tamper tests",
                (
                    "bind this incident in V6 AUTH/REVIEW/GO and keep V5 "
                    "production data/replay core AST identical while retaining "
                    "stdout-only network-zero policy"
                ),
            ],
            "incident_relative_path": str(
                subject.V6_HISTORICAL_FALSE_REJECT_INCIDENT_IDENTITY["path"]
            ),
            "must_bind_this_incident_identity": True,
            "report_file": None,
            "status": "REQUIRED_NOT_YET_AUTHORIZED",
        },
        "root_cause": {},
        "schema_version": 1,
        "status": (
            "SEALED_V5_POSTRUN_AUDITOR_HISTORICAL_IDENTITY_FALSE_REJECT_"
            "AFTER_FULL_REPLAY_NO_REPORT_V6_SUPERSESSION_REQUIRED"
        ),
    }
    write_json(false_reject_path, false_reject_payload)
    false_reject_identity = identity(false_reject_path, root)
    monkeypatch.setattr(
        subject,
        "V6_HISTORICAL_FALSE_REJECT_INCIDENT_IDENTITY",
        false_reject_identity,
    )
    return historical_path, false_reject_path, false_reject_payload


def _rebind_synthetic_incidents(
    root: Path,
    historical_path: Path,
    false_reject_path: Path,
    false_reject_payload: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], dict[str, object]]:
    historical_identity = identity(historical_path, root)
    monkeypatch.setattr(
        subject,
        "HISTORICAL_DIRECT_FILE_FAILURE_INCIDENT_IDENTITY",
        historical_identity,
    )
    false_reject_payload["historical_incident"]["identity"] = historical_identity  # type: ignore[index]
    write_json(false_reject_path, false_reject_payload)
    false_reject_identity = identity(false_reject_path, root)
    monkeypatch.setattr(
        subject,
        "V6_HISTORICAL_FALSE_REJECT_INCIDENT_IDENTITY",
        false_reject_identity,
    )
    return false_reject_identity, (
        subject.expected_historical_failed_head_identity_contract()
    )


def test_historical_exact_seven_and_v1_absent_five_pass_without_source_rehash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = control_root(tmp_path)
    _, _, _ = _synthetic_false_reject_incidents(root, monkeypatch)
    monkeypatch.setattr(
        subject,
        "strict_external_identity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("historical current-role source must not be rehashed")
        ),
    )
    declared_history = subject.expected_historical_failed_head_identity_contract()
    result = subject._load_and_validate_v6_false_reject_incident(
        root,
        subject.V6_HISTORICAL_FALSE_REJECT_INCIDENT_IDENTITY,
        declared_history,
    )
    assert result["failed_head_count"] == 7
    assert result["absent_control_path_count"] == 5


@pytest.mark.parametrize("role", tuple(subject.HISTORICAL_FAILED_HEAD_IDENTITIES))
def test_each_historical_failed_head_tamper_fails_without_current_role_rehash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: str
) -> None:
    root = control_root(tmp_path)
    historical_path, false_reject_path, false_reject = (
        _synthetic_false_reject_incidents(root, monkeypatch)
    )
    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    historical["failed_head_identities"][role]["sha256"] = "0" * 64
    write_json(historical_path, historical)
    incident_identity, declared_history = _rebind_synthetic_incidents(
        root,
        historical_path,
        false_reject_path,
        false_reject,
        monkeypatch,
    )
    monkeypatch.setattr(
        subject,
        "strict_external_identity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("historical source rehash forbidden")
        ),
    )
    with pytest.raises(subject.PostrunAuditSealError, match="exact-seven"):
        subject._load_and_validate_v6_false_reject_incident(
            root, incident_identity, declared_history
        )


@pytest.mark.parametrize(
    "mutation",
    ("v2_substitution", "missing", "extra", "reordered", "zero_state_flag"),
)
def test_historical_v1_zero_state_and_absent_five_tamper_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    root = control_root(tmp_path)
    historical_path, false_reject_path, false_reject = (
        _synthetic_false_reject_incidents(root, monkeypatch)
    )
    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    zero = historical["verified_zero_state_after_failure"]
    if mutation == "v2_substitution":
        zero["absent_control_paths"] = list(
            subject.HISTORICAL_V2_INCORRECT_CONTROL_PATHS
        )
    elif mutation == "missing":
        zero["absent_control_paths"].pop()
    elif mutation == "extra":
        zero["absent_control_paths"].append("prereg/unbound.json")
    elif mutation == "reordered":
        zero["absent_control_paths"].reverse()
    else:
        zero["canonical_recovery_outputs_absent"] = False
    write_json(historical_path, historical)
    incident_identity, declared_history = _rebind_synthetic_incidents(
        root,
        historical_path,
        false_reject_path,
        false_reject,
        monkeypatch,
    )
    with pytest.raises(subject.PostrunAuditSealError, match="zero-state"):
        subject._load_and_validate_v6_false_reject_incident(
            root, incident_identity, declared_history
        )


@pytest.mark.parametrize(
    "field",
    (
        "failed_head_identities",
        "verified_zero_state_after_failure",
        "absent_control_paths",
    ),
)
@pytest.mark.parametrize("attribute", ("canonical_size_bytes", "canonical_sha256"))
def test_each_historical_canonical_digest_or_size_tamper_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    attribute: str,
) -> None:
    root = control_root(tmp_path)
    _synthetic_false_reject_incidents(root, monkeypatch)
    declared = subject.expected_historical_failed_head_identity_contract()
    record = declared["historical_incident_contract"][field]
    if attribute == "canonical_size_bytes":
        record[attribute] += 1
    else:
        record[attribute] = "0" * 64
    with pytest.raises(subject.PostrunAuditSealError, match="contract mismatch"):
        subject._load_and_validate_v6_false_reject_incident(
            root, subject.V6_HISTORICAL_FALSE_REJECT_INCIDENT_IDENTITY, declared
        )


def test_historical_runner_observation_is_distinct_from_transient_and_final_v1(
) -> None:
    historical = subject.HISTORICAL_FAILED_HEAD_IDENTITIES["recovery_runner"][
        "sha256"
    ]
    transient = auditor.RECOVERY_REJECTED_TRANSIENT_V1_RUNNER_IDENTITY["sha256"]
    final = auditor.RECOVERY_V1_CHAIN_IDENTITIES["recovery_runner"]["sha256"]
    assert historical == (
        "9a1ea2a118d497dbe93fd9230dd82fbc832985f599a5c8ba13236a8d58919520"
    )
    assert len({historical, transient, final}) == 3

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import audit_noaa_gfs_multiseason_raw_postrun_v8 as auditor
from scripts import seal_noaa_gfs_multiseason_postrun_audit_v8 as subject


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


def namespace_root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    for relative in ("prereg", "independent_redteam", "incidents", "raw", "decoded"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    expected = subject._expected_control_namespace(authorization_present=False)
    for parent, names in expected.items():
        for name in names:
            write_bytes(root / parent / name)
    return root


def actual_code_state() -> dict[str, object]:
    return subject.collect_code_state(subject.REPO)


_FROZEN_V7_PRESEAL_EVIDENCE: dict[str, object] | None = None


def frozen_v7_preseal_evidence() -> dict[str, object]:
    global _FROZEN_V7_PRESEAL_EVIDENCE
    if _FROZEN_V7_PRESEAL_EVIDENCE is None:
        base = subject._load_and_validate_v7_authority_base(
            subject.ROOT_DEFAULT.resolve(), subject.V7_AUTHORITY_BASE
        )
        _FROZEN_V7_PRESEAL_EVIDENCE = (
            subject._frozen_v7_preseal_test_evidence_from_base(base)
        )
    return json.loads(json.dumps(_FROZEN_V7_PRESEAL_EVIDENCE))


def synthetic_evidence(code_state: dict[str, object]) -> dict[str, object]:
    bound_sources = code_state["bound_identities"]
    bound = {
        "v7_authorization": dict(subject.V7_AUTHORITY_BASE["authorization"]),
        "v7_independent_review": dict(
            subject.V7_AUTHORITY_BASE["independent_review"]
        ),
        "v7_independent_go": dict(subject.V7_AUTHORITY_BASE["independent_go"]),
        "primary_duration_suffix_incident": dict(
            subject.V8_PRIMARY_INCIDENT_IDENTITY
        ),
        "afterstate_commitment_correction": dict(
            subject.V8_CORRECTION_RECORD_IDENTITY
        ),
        "immutable_v2_preflight": dict(subject.V2_PREFLIGHT_IDENTITY),
        **bound_sources,
    }
    summaries = {
        "v8_auditor_tests": "1 passed in 0.01s",
        "v8_sealer_tests": "1 passed in 0.01s",
        "frozen_v7_auditor_tests": (
            "357 passed, 3 skipped, 2 deselected in 1.00s"
        ),
        "frozen_v7_sealer_tests": (
            "181 passed, 1 skipped, 1 deselected in 1.00s"
        ),
    }
    runs = {}
    for role, relative in subject.TEST_RELATIVES.items():
        nodeids = list(subject.V8_FROZEN_V7_TEST_DESELECTS.get(relative, ()))
        runs[role] = {
            "command": subject._v8_test_command(relative),
            "exit_code": 0,
            "summary": summaries[role],
            "stdout_sha256": "0" * 64,
            "stderr_sha256": "1" * 64,
            "network_guard_installed": True,
            "cacheprovider_disabled": True,
            "selection_mode": (
                "FULL_FILE_MINUS_EXACT_OBSOLETE_PRESEAL_STATE_TESTS"
                if nodeids else "FULL_FILE"
            ),
            "deselected_nodeids": nodeids,
            "deselected_count": len(nodeids),
        }
    return {
        "schema_version": 1,
        "artifact_type": "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_V8_TEST_EVIDENCE",
        "status": "PASS_BOUND_V8_CODE_COMPILE_AND_TEST_SUITE",
        "created_utc": "2026-08-11T07:00:00.000000Z",
        "bound_identities": bound,
        "source_compile": {
            "method": "compile_exact_source_no_pyc",
            "result": "PASS",
            "files": [
                bound_sources[role]
                for role in (
                    "v8_auditor", "v8_auditor_test", "v8_sealer", "v8_sealer_test"
                )
            ],
        },
        "pytest_isolation": "ONE_TEST_FILE_PER_CLEAN_NETWORK_DENIED_SUBPROCESS",
        "test_runs": runs,
        "required_test_names": list(auditor.V8_REQUIRED_TEST_NAMES),
        "required_test_names_present": True,
        "all_exit_codes_zero": True,
        "production_shape_regression": {
            key: True for key in auditor.V8_PRODUCTION_SHAPE_KEYS
        },
        "real_seven_spawn_regression": {
            "start_method": "spawn",
            "entrypoint_module": "scripts.audit_noaa_gfs_multiseason_raw_postrun_v8",
            "worker_callable_module": "scripts.audit_noaa_gfs_multiseason_raw_postrun_v8",
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
        "frozen_v7_preseal_test_evidence": frozen_v7_preseal_evidence(),
    }


@pytest.mark.parametrize(
    "summary",
    (
        "11 passed, 1 skipped in 5.34s",
        "14 passed in 3.47s",
        "60 passed in 2.25s",
        "27 passed in 2.67s",
        "25 passed in 1.31s",
        "85 passed in 65.33s (0:01:05)",
        "1 passed in 123s",
        "1 passed in 123.456s",
        "1 passed in 60.00s",
        "1 passed in 60.00s (0:01:00)",
        "1 passed in 61.00s (0:01:00)",
    ),
)
def test_local_and_auditor_pytest_summary_parsers_accept_contract(
    summary: str,
) -> None:
    assert subject._parse_pytest_success_summary_local(
        summary, label="valid"
    ) == auditor._v8_parse_pytest_success_summary(summary, label="valid")


@pytest.mark.parametrize(
    "summary",
    (
        "0 passed in 0.01s",
        "1 failed in 0.01s",
        "1 passed in 65.3s (0:01:05)",
        "1 passed in 65.333s (0:01:05)",
        "1 passed in 59.99s (0:00:59)",
        "1 passed in 65.33s (0:01:04)",
        "1 passed in 65.33s (00:01:05)",
        "1 passed in 65.33s (0:1:05)",
        "1 passed in 65.33s (0:01:5)",
        "1 passed in 65.33s (0:60:00)",
        "1 passed in 65.33s (0:01:60)",
        " 1 passed in 65.33s (0:01:05)",
    ),
)
def test_pytest_summary_parser_rejects_malformed_or_inconsistent_suffix(
    summary: str,
) -> None:
    with pytest.raises(subject.PostrunAuditSealError):
        subject._parse_pytest_success_summary_local(summary, label="invalid")


def test_parser_uses_integer_arithmetic_without_float() -> None:
    source = Path(subject.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_parse_pytest_success_summary_local"
    )
    calls = {
        ast.unparse(node.func)
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
    }
    assert "float" not in calls
    assert "timedelta" not in calls
    assert "total_seconds" not in calls


def test_actual_v7_authority_base_passes_read_only() -> None:
    state = subject._load_and_validate_v7_authority_base(
        subject.ROOT_DEFAULT.resolve(), subject.V7_AUTHORITY_BASE
    )
    assert state["base"] == subject.V7_AUTHORITY_BASE
    assert state["recovered_afterstate_commitment"] == (
        auditor.V5_RECOVERED_AFTERSTATE_COMMITMENT
    )


def test_frozen_v7_original_preseal_evidence_is_extracted_from_actual_auth() -> None:
    local_base = subject._load_and_validate_v7_authority_base(
        subject.ROOT_DEFAULT.resolve(), subject.V7_AUTHORITY_BASE
    )
    independent_base = auditor._v8_load_and_validate_v7_authority_base(
        subject.ROOT_DEFAULT.resolve(), subject.V7_AUTHORITY_BASE
    )
    local = subject._frozen_v7_preseal_test_evidence_from_base(local_base)
    independent = auditor._v8_frozen_v7_preseal_test_evidence(independent_base)
    assert local == independent == frozen_v7_preseal_evidence()
    assert set(local) == {
        "authorization", "canonical_size_bytes", "canonical_sha256",
        "artifact_type", "status", "created_utc", "original_full_runs",
        "state_obsolete_current_rerun_nodeids",
    }
    assert set(local["original_full_runs"]) == {
        "v7_auditor_tests", "v7_sealer_tests"
    }
    assert local["state_obsolete_current_rerun_nodeids"] == [
        *subject.V8_FROZEN_V7_TEST_DESELECTS[
            "tests/test_noaa_gfs_multiseason_raw_postrun_v7.py"
        ],
        *subject.V8_FROZEN_V7_TEST_DESELECTS[
            "tests/test_seal_noaa_gfs_multiseason_postrun_audit_v7.py"
        ],
    ]
    assert len(local["state_obsolete_current_rerun_nodeids"]) == 3


def test_frozen_v7_original_preseal_evidence_tamper_fails() -> None:
    base = subject._load_and_validate_v7_authority_base(
        subject.ROOT_DEFAULT.resolve(), subject.V7_AUTHORITY_BASE
    )
    tampered = {
        "authorization": json.loads(json.dumps(base["authorization"]))
    }
    tampered["authorization"]["test_evidence"]["test_runs"][
        "v7_auditor_tests"
    ]["summary"] = "358 passed, 3 skipped in 1.00s"
    with pytest.raises(subject.PostrunAuditSealError, match="identity"):
        subject._frozen_v7_preseal_test_evidence_from_base(tampered)


@pytest.mark.parametrize(
    "field",
    (
        "authorization",
        "authorization_status",
        "authorization_payload_canonical_sha256",
        "audit_attempt_id",
    ),
)
def test_v7_authority_base_declaration_tamper_fails(field: str) -> None:
    tampered = json.loads(json.dumps(subject.V7_AUTHORITY_BASE))
    if isinstance(tampered[field], dict):
        tampered[field]["sha256"] = "0" * 64
    else:
        tampered[field] = "tampered"
    with pytest.raises(subject.PostrunAuditSealError, match="declaration"):
        subject._load_and_validate_v7_authority_base(
            subject.ROOT_DEFAULT.resolve(), tampered
        )


def test_actual_primary_and_correction_apply_before_semantic_use() -> None:
    state = subject._load_and_validate_v8_live_failure_state(
        subject.ROOT_DEFAULT.resolve()
    )
    assert state["state"]["primary_incident_alone_must_fail"] is True
    assert state["state"]["correction_must_be_applied_before_semantic_use"] is True
    assert state["state"]["corrected_logical_primary_incident"] == (
        subject.V8_CORRECTED_LOGICAL_PRIMARY_IDENTITY
    )
    assert state["corrected_primary"]["postfailure_state"][
        "recovered_afterstate_commitment"
    ]["recovery_progress_file_count"] == 104


def test_primary_without_correction_fails(tmp_path: Path) -> None:
    root = tmp_path / "root"
    primary = subject.ROOT_DEFAULT / subject.V8_PRIMARY_INCIDENT_IDENTITY["path"]
    destination = root / subject.V8_PRIMARY_INCIDENT_IDENTITY["path"]
    write_bytes(destination, primary.read_bytes())
    with pytest.raises(subject.PostrunAuditSealError, match="correction"):
        subject._load_and_validate_v8_live_failure_state(root)


@pytest.mark.parametrize(
    "mutation",
    ("pointer", "value", "primary_sha", "logical_sha"),
)
def test_correction_semantic_tamper_fails(mutation: str) -> None:
    path = subject.ROOT_DEFAULT / subject.V8_CORRECTION_RECORD_IDENTITY["path"]
    correction = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "pointer":
        correction["correction"]["corrected_json_pointer"] += "/wrong"
    elif mutation == "value":
        correction["correction"]["corrected_value"] = 103
    elif mutation == "primary_sha":
        correction["erroneous_incident"]["sha256"] = "0" * 64
    else:
        correction["correction"]["corrected_logical_primary_incident"][
            "canonical_sha256"
        ] = "0" * 64
    with pytest.raises(subject.PostrunAuditSealError):
        subject._validate_correction_record(correction)


def test_actual_v2_preflight_local_and_auditor_validators_match() -> None:
    root = subject.ROOT_DEFAULT.resolve()
    local = subject._load_and_validate_immutable_v2_preflight_evidence(root)
    independent = auditor._v8_validate_actual_v2_preflight_evidence(root)
    assert local["commitment"] == independent["commitment"]
    assert local["preflight"] == independent["preflight"]
    assert local["evidence"] == independent["test_evidence"]
    assert local["commitment"] == subject.V8_IMMUTABLE_V2_PREFLIGHT_EVIDENCE


def test_actual_six_summary_commitments_are_exact() -> None:
    state = subject._load_and_validate_immutable_v2_preflight_evidence(
        subject.ROOT_DEFAULT.resolve()
    )
    summaries = [run["summary"] for run in state["evidence"]["pytest_runs"]]
    assert len(summaries) == 6
    assert summaries[5] == "85 passed in 65.33s (0:01:05)"
    assert len(subject.canonical_bytes(summaries)) == 149
    assert hashlib.sha256(subject.canonical_bytes(summaries)).hexdigest() == (
        "4ff61d69b17fc71bb64e59d6bde5c281828424c92055ffdfb2410c247a127f28"
    )


def test_preflight_physical_tamper_fails(tmp_path: Path) -> None:
    root = tmp_path / "root"
    destination = root / subject.V2_PREFLIGHT_IDENTITY["path"]
    source = subject.ROOT_DEFAULT / subject.V2_PREFLIGHT_IDENTITY["path"]
    write_bytes(destination, source.read_bytes() + b"\n")
    with pytest.raises(subject.PostrunAuditSealError, match="physical"):
        subject._load_and_validate_immutable_v2_preflight_evidence(root)


def test_contract_census_is_exact() -> None:
    assert len(subject.BOUND_SOURCE_ROLES) == 8
    assert len(subject.BOUND_IDENTITY_ROLES) == 14
    assert len(subject.CONTROL_TEMP_RELATIVES) == 27
    assert len(set(subject.CONTROL_TEMP_RELATIVES)) == 27
    assert len(subject.POSTRUN_REPORT_CANDIDATES) == 21
    assert set(subject.TEST_RELATIVES) == auditor.V8_TEST_RUN_KEYS
    assert subject.V8_FROZEN_V7_TEST_DESELECTS == auditor.V8_FROZEN_V7_TEST_DESELECTS
    assert len(auditor.V8_TEST_EVIDENCE_KEYS) == 23
    assert len(auditor.V8_TEST_RUN_RECORD_KEYS) == 10
    assert len(auditor.V8_REQUIRED_TEST_NAMES) == 42
    assert auditor.V8_REQUIRED_TEST_NAMES[-2:] == (
        "test_frozen_v7_original_preseal_full_suite_evidence_is_exact",
        "test_frozen_v7_postseal_rerun_deselects_only_three_obsolete_preseal_state_dependent_tests",
    )
    assert set(auditor.V8_AUTH_KEYS) == {
        "schema_version", "artifact_type", "status", "created_utc",
        "audit_attempt_id", "v7_authority_base", "v7_live_failure_state",
        "immutable_v2_preflight_evidence", "superseded_v7_auditor",
        "superseded_v7_auditor_test", "superseded_v7_sealer",
        "superseded_v7_sealer_test", "v8_auditor", "v8_auditor_test",
        "v8_sealer", "v8_sealer_test", "recovery_attempt_id",
        "recovered_afterstate_commitment", "preaudit_zero_mutation_snapshot",
        "runtime_identity_sha256", "test_evidence", "required_command",
        "max_spawn_processes", "network_requests_allowed",
        "audit_files_written_allowed", "labels_read_allowed",
        "arrays_2024_read_allowed", "arrays_2025_read_allowed",
        "models_fit_allowed", "submission_csv_allowed",
        "full_offline_redecode_required", "stdout_only",
        "independent_review_required", "independent_go_required",
    }


def test_code_state_binds_frozen_v7_and_executing_v8() -> None:
    state = actual_code_state()
    bound = state["bound_identities"]
    assert set(bound) == subject.BOUND_SOURCE_ROLES
    for role, expected in subject.FROZEN_V7_SOURCE_IDENTITIES.items():
        assert bound[role] == expected
    assert bound["v8_auditor"] == subject.EXPECTED_V8_AUDITOR_IDENTITY
    assert bound["v8_auditor_test"] == subject.EXPECTED_V8_AUDITOR_TEST_IDENTITY


@pytest.mark.parametrize("role", sorted(subject.BOUND_SOURCE_ROLES))
def test_bound_source_pyc_competitor_is_rejected(
    tmp_path: Path, role: str
) -> None:
    path = tmp_path / f"{role}.py"
    write_bytes(path, b"VALUE = 1\n")
    bound = {
        item: identity(path)
        for item in subject.BOUND_SOURCE_ROLES
    }
    pyc = path.with_suffix(".pyc")
    write_bytes(pyc, b"pyc")
    with pytest.raises(subject.PostrunAuditSealError, match="bytecode"):
        subject._require_bound_bytecode_absent(bound)


def test_preimport_guard_rejects_top_level_io(tmp_path: Path) -> None:
    path = tmp_path / "auditor.py"
    path.write_text("from pathlib import Path\nVALUE = Path('x').read_bytes()\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unapproved import-time call"):
        subject._preimport_auditor_source_guard(path)


def test_duplicate_json_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"a": 1, "a": 2}\n', encoding="utf-8")
    with pytest.raises(subject.PostrunAuditSealError, match="duplicate"):
        subject.load_json_control(path, label="duplicate")


def test_actual_zero_snapshot_is_compact_exact13() -> None:
    snapshot = subject.build_zero_snapshot(subject.ROOT_DEFAULT.resolve())
    assert set(snapshot) == auditor.V8_ZERO_SNAPSHOT_KEYS
    assert snapshot["preauthorization_control_namespace"] == {
        "counts": {"prereg": 5, "independent_redteam": 6, "incidents": 8},
        "canonical_size_bytes": 1388,
        "canonical_sha256": (
            "c16a239873847da42d138b5cab941c3eb81e5141aaebdabb2695140a3f90347c"
        ),
    }
    assert "TRACK_A_DECODE" not in json.dumps(
        snapshot["preauthorization_control_namespace"], sort_keys=True
    )


def test_full_namespace_map_cannot_substitute_compact_snapshot() -> None:
    snapshot = subject.build_zero_snapshot(subject.ROOT_DEFAULT.resolve())
    snapshot["preauthorization_control_namespace"] = (
        subject.validate_control_namespace(
            subject.ROOT_DEFAULT.resolve(), authorization_present=False
        )
    )
    assert set(snapshot["preauthorization_control_namespace"]) != {
        "counts", "canonical_size_bytes", "canonical_sha256"
    }


def test_namespace_exact_and_unbound_alias_fails(tmp_path: Path) -> None:
    root = namespace_root(tmp_path)
    observed = subject.validate_control_namespace(root, authorization_present=False)
    assert {key: len(value) for key, value in observed.items()} == {
        "prereg": 5,
        "independent_redteam": 6,
        "incidents": 8,
    }
    write_bytes(root / "incidents/TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_ALIAS.json")
    with pytest.raises(subject.PostrunAuditSealError, match="inventory"):
        subject.validate_control_namespace(root, authorization_present=False)


def test_zero_snapshot_normalizes_only_published_v8_authorization(
    tmp_path: Path,
) -> None:
    root = namespace_root(tmp_path)
    before = subject.build_zero_snapshot(root)
    write_bytes(root / subject.AUTH_RELATIVE)
    after = subject.build_zero_snapshot(root)
    assert after == before
    assert after["preauthorization_control_namespace"]["counts"] == {
        "prereg": 5,
        "independent_redteam": 6,
        "incidents": 8,
    }


@pytest.mark.parametrize(
    "unexpected_relative",
    [
        subject.REVIEW_RELATIVE,
        subject.GO_RELATIVE,
        "prereg/decode_recovery_postrun_audit_unbound_v8.json",
    ],
)
def test_zero_snapshot_with_published_authorization_rejects_other_controls(
    tmp_path: Path, unexpected_relative: str,
) -> None:
    root = namespace_root(tmp_path)
    write_bytes(root / subject.AUTH_RELATIVE)
    write_bytes(root / unexpected_relative)
    with pytest.raises(subject.PostrunAuditSealError, match="inventory"):
        subject.build_zero_snapshot(root)


def test_collect_root_inputs_is_equal_after_only_authorization_is_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = namespace_root(tmp_path)
    code = {"bound_identities": {role: {} for role in subject.BOUND_SOURCE_ROLES}}
    created = {
        "authorization": {"created_utc": "2026-08-11T05:38:54.057056Z"},
        "review": {"created_utc": "2026-08-11T05:46:24.609078Z"},
        "go": {"created_utc": "2026-08-11T05:51:30.322606Z"},
    }
    base = {
        "base": {"stable": True},
        **created,
        "recovered_afterstate_commitment": {"commitment": True},
    }
    failure = {
        "state": {"failure": True},
        "primary_created_utc": "2026-08-11T06:14:43.374633Z",
        "correction_created_utc": "2026-08-11T06:43:59.084809Z",
    }
    preflight = {
        "commitment": {"preflight": True},
        "preflight": {"authorization": True},
        "evidence": {"tests": True},
    }
    monkeypatch.setattr(subject, "_load_and_validate_v7_authority_base", lambda *_a: base)
    monkeypatch.setattr(subject, "_load_and_validate_v8_live_failure_state", lambda *_a: failure)
    monkeypatch.setattr(
        subject, "_load_and_validate_immutable_v2_preflight_evidence", lambda *_a: preflight
    )
    monkeypatch.setattr(
        auditor,
        "_v8_validate_actual_v2_preflight_evidence",
        lambda *_a: {
            "commitment": preflight["commitment"],
            "preflight_identity": subject.V2_PREFLIGHT_IDENTITY,
            "preflight": preflight["preflight"],
            "test_evidence": preflight["evidence"],
        },
    )
    monkeypatch.setattr(
        auditor,
        "_v8_load_and_validate_v7_authority_base",
        lambda *_a: {"summary": base["base"], **created},
    )
    monkeypatch.setattr(
        auditor,
        "_v8_validate_duration_suffix_incident_chain",
        lambda *_a: {
            "state": failure["state"],
            "primary_created": {"raw": failure["primary_created_utc"]},
            "correction_created": {"raw": failure["correction_created_utc"]},
        },
    )
    afterstate = {"actual": "metadata"}
    monkeypatch.setattr(
        auditor,
        "_v6_load_and_validate_v5_authority_base",
        lambda *_a: {
            "recovered_afterstate": afterstate,
            "v4_base": {"authorization": {"metadata": True}},
        },
    )
    monkeypatch.setattr(
        subject, "_recovered_afterstate_commitment",
        lambda actual: base["recovered_afterstate_commitment"]
        if actual == afterstate else None,
    )
    monkeypatch.setattr(
        subject, "validate_recovered_filesystem_closure", lambda *_a: {"closed": True}
    )

    before = subject.collect_root_inputs(root, code)
    write_bytes(root / subject.AUTH_RELATIVE)
    after = subject.collect_root_inputs(root, code)
    assert after == before


def test_required_command_is_exact_v8(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    command = subject.required_audit_command(root)
    assert command[1:4] == [
        "-B", "-m", "scripts.audit_noaa_gfs_multiseason_raw_postrun_v8"
    ]
    assert command[-4:] == [
        "--authorization", str((root / subject.AUTH_RELATIVE).resolve()),
        "--independent-go", str((root / subject.GO_RELATIVE).resolve()),
    ]


def test_test_commands_select_full_v8_and_exact_frozen_v7_subsets() -> None:
    for role, relative in subject.TEST_RELATIVES.items():
        command = subject._v8_test_command(relative)
        expected = list(subject.V8_FROZEN_V7_TEST_DESELECTS.get(relative, ()))
        deselects = [item for item in command if item.startswith("--deselect=")]
        if role.startswith("frozen_v7_"):
            assert deselects == [f"--deselect={nodeid}" for nodeid in expected]
            assert command[-len(deselects):] == deselects
        else:
            assert expected == []
            assert deselects == []
        assert "-k" not in command
        assert not any(item.startswith("--ignore") for item in command)
        assert command[3] == auditor.V8_PYTEST_NETWORK_GUARD_SOURCE
        assert command == auditor._v8_test_command(relative)


def test_compact_v8_pytest_guard_retains_audit_hook_and_socket_denials() -> None:
    guard = auditor.V8_PYTEST_NETWORK_GUARD_SOURCE
    assert len(guard.encode("utf-8")) < len(
        auditor.V6_PYTEST_NETWORK_GUARD_SOURCE.encode("utf-8")
    )
    assert "sys.addaudithook(_audit)" in guard
    for event in (
        "socket.__new__", "socket.bind", "socket.connect",
        "socket.connect_ex", "socket.getaddrinfo", "socket.gethostbyaddr",
        "socket.gethostbyname", "socket.getnameinfo", "socket.sendmsg",
        "socket.sendto",
    ):
        assert f'"{event}"' in guard
    for api in (
        "create_connection", "getaddrinfo", "gethostbyaddr",
        "gethostbyname", "getnameinfo",
    ):
        assert f'"{api}"' in guard


@pytest.mark.parametrize(
    ("role", "summary"),
    (
        ("v8_auditor_tests", "1 passed, 1 deselected in 0.01s"),
        ("frozen_v7_auditor_tests", "358 passed, 3 skipped, 2 deselected in 1.00s"),
        ("frozen_v7_auditor_tests", "357 passed, 3 skipped, 1 deselected in 1.00s"),
        ("frozen_v7_auditor_tests", "357 passed, 3 skipped in 1.00s"),
        ("frozen_v7_sealer_tests", "181 passed, 2 skipped, 1 deselected in 1.00s"),
    ),
)
def test_bound_test_summary_wrong_counts_or_deselection_fail(
    role: str, summary: str,
) -> None:
    with pytest.raises(subject.PostrunAuditSealError):
        subject._validate_bound_test_run_summary_local(role, summary)


def test_evidence_rejects_wrong_multiple_k_or_ignore_selection() -> None:
    code = actual_code_state()
    evidence = synthetic_evidence(code)
    bound = evidence["bound_identities"]
    v7_base = subject._load_and_validate_v7_authority_base(
        subject.ROOT_DEFAULT.resolve(), subject.V7_AUTHORITY_BASE
    )
    command_mutations = (
        ["--deselect=tests/wrong.py::test_wrong"],
        [
            *[
                f"--deselect={nodeid}"
                for nodeid in subject.V8_FROZEN_V7_TEST_DESELECTS[
                    subject.TEST_RELATIVES["frozen_v7_auditor_tests"]
                ]
            ],
            "--deselect=tests/wrong.py::test_second",
        ],
        ["-k", "not obsolete"],
        ["--ignore=tests/test_seal_noaa_gfs_multiseason_postrun_audit_v7.py"],
    )
    for suffix in command_mutations:
        tampered = json.loads(json.dumps(evidence))
        relative = subject.TEST_RELATIVES["frozen_v7_auditor_tests"]
        deselect_count = len(subject.V8_FROZEN_V7_TEST_DESELECTS[relative])
        base_command = subject._v8_test_command(relative)[:-deselect_count]
        tampered["test_runs"]["frozen_v7_auditor_tests"]["command"] = [
            *base_command, *suffix
        ]
        with pytest.raises(subject.PostrunAuditSealError, match="run invalid"):
            subject._validate_v8_test_evidence_local(
                tampered,
                authorization_created_ticks=subject._exact_utc_z_100ns_ticks(
                    "2026-08-11T07:00:01.000000Z", label="test AUTH"
                ),
                bound_identities=bound,
                v7_base=v7_base,
            )


def test_actual_root_collect_build_and_self_validator_passes() -> None:
    code = actual_code_state()
    root = subject.ROOT_DEFAULT.resolve()
    inputs = subject.collect_root_inputs(root, code)
    evidence = synthetic_evidence(code)
    payload = subject.build_authorization(
        root,
        code,
        inputs,
        evidence,
        created_utc="2026-08-11T07:00:01.000000Z",
    )
    subject.validate_authorization_payload(root, payload, code, inputs)
    assert set(payload) == auditor.V8_AUTH_KEYS
    assert payload["v7_live_failure_state"]["primary_incident"] == (
        subject.V8_PRIMARY_INCIDENT_IDENTITY
    )
    assert payload["v7_live_failure_state"]["correction_record"] == (
        subject.V8_CORRECTION_RECORD_IDENTITY
    )
    assert len(subject.canonical_bytes(payload)) <= 32_768
    physical = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode(
            "utf-8"
        )
        + b"\n"
    )
    assert len(physical) <= 32_768


@pytest.mark.parametrize(
    "field",
    (
        "artifact_type",
        "status",
        "v7_authority_base",
        "v7_live_failure_state",
        "immutable_v2_preflight_evidence",
        "required_command",
        "recovered_afterstate_commitment",
        "preaudit_zero_mutation_snapshot",
    ),
)
def test_authorization_schema_or_compact_binding_tamper_fails(
    field: str,
) -> None:
    code = actual_code_state()
    root = subject.ROOT_DEFAULT.resolve()
    inputs = subject.collect_root_inputs(root, code)
    payload = subject.build_authorization(
        root,
        code,
        inputs,
        synthetic_evidence(code),
        created_utc="2026-08-11T07:00:01.000000Z",
    )
    if isinstance(payload[field], dict):
        payload[field] = {**payload[field], "tamper": True}
    elif isinstance(payload[field], list):
        payload[field] = [*payload[field], "tamper"]
    else:
        payload[field] = "tamper"
    with pytest.raises(subject.PostrunAuditSealError):
        subject.validate_authorization_payload(root, payload, code, inputs)


@pytest.mark.parametrize(
    ("evidence_created", "authorization_created"),
    (
        ("2026-08-11T06:43:59.084808Z", "2026-08-11T07:00:01.000000Z"),
        ("2026-08-11T07:00:02.000000Z", "2026-08-11T07:00:01.000000Z"),
    ),
)
def test_evidence_or_authorization_chronology_fails(
    evidence_created: str,
    authorization_created: str,
) -> None:
    code = actual_code_state()
    root = subject.ROOT_DEFAULT.resolve()
    inputs = subject.collect_root_inputs(root, code)
    evidence = synthetic_evidence(code)
    evidence["created_utc"] = evidence_created
    with pytest.raises(subject.PostrunAuditSealError, match="chronology"):
        subject.build_authorization(
            root, code, inputs, evidence, created_utc=authorization_created
        )


def test_all_five_order_preserving_raw_timestamp_substitutions_fail() -> None:
    code = actual_code_state()
    root = subject.ROOT_DEFAULT.resolve()
    inputs = subject.collect_root_inputs(root, code)
    payload = subject.build_authorization(
        root,
        code,
        inputs,
        synthetic_evidence(code),
        created_utc="2026-08-11T07:00:01.000000Z",
    )
    substitutions = {
        "v7_authorization_created_utc": "2026-08-11T05:38:54.057057Z",
        "v7_review_created_utc": "2026-08-11T05:46:24.609079Z",
        "v7_go_created_utc": "2026-08-11T05:51:30.322607Z",
        "primary_incident_created_utc": "2026-08-11T06:14:43.374634Z",
        "correction_created_utc": "2026-08-11T06:43:59.084810Z",
    }
    for field, substitute in substitutions.items():
        tampered = dict(inputs)
        tampered[field] = substitute
        with pytest.raises(subject.PostrunAuditSealError, match="raw authority"):
            subject.validate_authorization_payload(root, payload, code, tampered)


def test_nested_zero_snapshot_tampers_fail_independent_validator() -> None:
    code = actual_code_state()
    root = subject.ROOT_DEFAULT.resolve()
    inputs = subject.collect_root_inputs(root, code)
    payload = subject.build_authorization(
        root,
        code,
        inputs,
        synthetic_evidence(code),
        created_utc="2026-08-11T07:00:01.000000Z",
    )
    mutations = (
        lambda snap: snap["preauthorization_control_namespace"]["counts"].__setitem__(
            "prereg", 6
        ),
        lambda snap: snap["preauthorization_control_namespace"].__setitem__(
            "canonical_sha256", "0" * 64
        ),
        lambda snap: snap["v8_control_temporary_files"][
            "checked_destination_relative_paths"
        ].pop(),
        lambda snap: snap["active_locks"]["raw"].__setitem__("present", True),
        lambda snap: snap.__setitem__("postrun_audit_output_files_present", 1),
        lambda snap: snap.__setitem__("network_requests", 1),
    )
    for mutate in mutations:
        tampered_payload = json.loads(json.dumps(payload))
        tampered_inputs = json.loads(json.dumps(inputs))
        mutate(tampered_payload["preaudit_zero_mutation_snapshot"])
        mutate(tampered_inputs["preaudit_zero_mutation_snapshot"])
        with pytest.raises(subject.PostrunAuditSealError):
            subject.validate_authorization_payload(
                root, tampered_payload, code, tampered_inputs
            )


def test_compact_guard_rejects_inventory_list104_and_report() -> None:
    for payload in (
        {"inventory": []},
        {"nested": {"records": list(range(104))}},
        {"nested": {"report": {}}},
    ):
        with pytest.raises(subject.PostrunAuditSealError):
            subject._assert_v8_authorization_compact(payload)


def test_collect_root_inputs_gates_preflight_before_recovered_closure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code = {"bound_identities": {role: {} for role in subject.BOUND_SOURCE_ROLES}}
    calls = {"closure": 0, "afterstate": 0}
    monkeypatch.setattr(subject, "validate_control_namespace", lambda *_a, **_k: {})
    monkeypatch.setattr(subject, "_load_and_validate_v7_authority_base", lambda *_a: {})
    monkeypatch.setattr(subject, "_load_and_validate_v8_live_failure_state", lambda *_a: {})

    def fail_preflight(*_args: object) -> object:
        raise subject.PostrunAuditSealError("preflight sentinel")

    monkeypatch.setattr(
        subject, "_load_and_validate_immutable_v2_preflight_evidence", fail_preflight
    )
    monkeypatch.setattr(
        auditor,
        "_v6_load_and_validate_v5_authority_base",
        lambda *_a: calls.__setitem__("afterstate", calls["afterstate"] + 1),
    )
    monkeypatch.setattr(
        subject,
        "validate_recovered_filesystem_closure",
        lambda *_a: calls.__setitem__("closure", calls["closure"] + 1),
    )
    with pytest.raises(subject.PostrunAuditSealError, match="preflight sentinel"):
        subject.collect_root_inputs(subject.ROOT_DEFAULT.resolve(), code)
    assert calls == {"closure": 0, "afterstate": 0}


def test_isolated_evidence_uses_four_exact_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code = actual_code_state()
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(command)
        relative = next(
            item for item in command if item in subject.TEST_RELATIVES.values()
        )
        summaries = {
            "tests/test_noaa_gfs_multiseason_raw_postrun_v7.py": (
                "357 passed, 3 skipped, 2 deselected in 1.00s"
            ),
            "tests/test_seal_noaa_gfs_multiseason_postrun_audit_v7.py": (
                "181 passed, 1 skipped, 1 deselected in 1.00s"
            ),
        }
        return SimpleNamespace(
            returncode=0,
            stdout=summaries.get(relative, "1 passed in 0.01s") + "\n",
            stderr="",
        )

    monkeypatch.setattr(subject.subprocess, "run", fake_run)
    monkeypatch.setattr(
        subject, "_source_function_names", lambda _path: set(auditor.V8_REQUIRED_TEST_NAMES)
    )
    monkeypatch.setattr(subject, "require_canonical_module_entrypoint", lambda _root: None)
    evidence = subject.run_immutable_test_evidence(
        subject.ROOT_DEFAULT.resolve(), subject.REPO, code
    )
    assert calls == [
        subject._v8_test_command(relative)
        for relative in subject.TEST_RELATIVES.values()
    ]
    assert set(evidence) == auditor.V8_TEST_EVIDENCE_KEYS
    assert evidence["bound_identities"].keys() == subject.BOUND_IDENTITY_ROLES


def test_publish_no_overwrite_preserves_existing(tmp_path: Path) -> None:
    path = tmp_path / "authorization.json"
    write_bytes(path, b"existing")
    with pytest.raises(subject.PostrunAuditSealError, match="already exists"):
        subject.publish_json_no_overwrite(path, {"new": True})
    assert path.read_bytes() == b"existing"


def test_publish_json_creates_exact_single_file(tmp_path: Path) -> None:
    path = tmp_path / "authorization.json"
    payload = {"schema_version": 8, "status": "test"}
    subject.publish_json_no_overwrite(path, payload)
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert not list(tmp_path.glob(".*.tmp.*"))


def test_seal_publishes_authorization_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = namespace_root(tmp_path)
    code = {"bound_identities": {role: {} for role in subject.BOUND_SOURCE_ROLES}}
    inputs = {"stable": True}
    evidence = {"evidence": True}
    authorization = {"schema_version": 8, "status": "synthetic"}
    monkeypatch.setattr(subject, "require_canonical_module_entrypoint", lambda _root: None)
    monkeypatch.setattr(subject, "install_entry_guard", lambda: None)
    monkeypatch.setattr(subject, "collect_code_state", lambda _workspace: code)
    monkeypatch.setattr(subject, "collect_root_inputs", lambda _root, _code: inputs)
    monkeypatch.setattr(
        subject, "run_immutable_test_evidence", lambda *_args: evidence
    )
    monkeypatch.setattr(subject, "build_authorization", lambda *_a, **_k: authorization)
    monkeypatch.setattr(auditor, "_v5_load_strict_json", lambda *_a, **_k: authorization)
    report = subject.seal(root, subject.REPO)
    assert report["authorization_files_published"] == 1
    assert (root / subject.AUTH_RELATIVE).is_file()
    assert not (root / subject.REVIEW_RELATIVE).exists()
    assert not (root / subject.GO_RELATIVE).exists()


def test_existing_authorization_fails_before_code_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = namespace_root(tmp_path)
    write_bytes(root / subject.AUTH_RELATIVE)
    monkeypatch.setattr(subject, "require_canonical_module_entrypoint", lambda _root: None)
    monkeypatch.setattr(subject, "install_entry_guard", lambda: None)
    monkeypatch.setattr(
        subject,
        "collect_code_state",
        lambda _workspace: (_ for _ in ()).throw(AssertionError("must not collect")),
    )
    with pytest.raises(subject.PostrunAuditSealError, match="already exists"):
        subject.seal(root, subject.REPO)


def test_sealer_source_is_static_network_zero_and_single_publisher() -> None:
    source = Path(subject.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name.split(".", 1)[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".", 1)[0]
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
    }
    assert imported.isdisjoint(
        {"requests", "urllib", "http", "pandas", "pyarrow", "eccodes"}
    )
    publisher_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "publish_json_no_overwrite"
    ]
    assert len(publisher_calls) == 1
    assert "scripts.audit_noaa_gfs_multiseason_raw_postrun_v8" in source
    assert "PASS_BOUND_V8_CODE_COMPILE_AND_TEST_SUITE" in source


def test_source_and_test_compile_without_writing_pyc() -> None:
    for path in (Path(subject.__file__).resolve(), Path(__file__).resolve()):
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        assert not path.with_suffix(".pyc").exists()

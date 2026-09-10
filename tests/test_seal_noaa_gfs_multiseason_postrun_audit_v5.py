from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import audit_noaa_gfs_multiseason_raw_postrun_v5 as auditor
from scripts import seal_noaa_gfs_multiseason_postrun_audit_v5 as subject


def write_bytes(path: Path, data: bytes = b"{}") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


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
        auditor.V2_FALSE_REJECT_INCIDENT_RELATIVE,
        auditor.V4_PATH_MISPUBLISH_INCIDENT_RELATIVE,
        auditor.V5_TRANSPORT_INCIDENT_RELATIVE,
        auditor.V5_TRANSPORT_CORRECTION_RELATIVE,
    ):
        write_bytes(root / relative)


def fake_code_state(tmp_path: Path) -> dict[str, object]:
    paths: dict[str, Path] = {}
    for role in sorted(subject.BOUND_SOURCE_ROLES):
        if role == "v5_auditor":
            path = Path(subject.EXPECTED_V5_AUDITOR_IDENTITY["path"])
        elif role == "v5_auditor_test":
            path = Path(subject.EXPECTED_V5_AUDITOR_TEST_IDENTITY["path"])
        else:
            path = tmp_path / f"{role}.py"
            write_bytes(path, b"VALUE = 1\n")
        paths[role] = path
    return {"bound_identities": {role: identity(path) for role, path in paths.items()}}


def fake_inputs() -> dict[str, object]:
    incident, correction = subject._expected_incident_identities()
    snapshot = {
        "incident_postfailure_state_canonical_sha256": (
            auditor.V5_INCIDENT_POSTFAILURE_STATE_CANONICAL_SHA256
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
        "v5_control_temporary_files": {
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
    base = {
        "identity": dict(auditor.FROZEN_V4_AUTH_IDENTITY),
        "schema_version": 4,
        "status": "AUTHORIZED_PENDING_INDEPENDENT_REVIEW_AND_GO_V4",
        "audit_attempt_id": auditor.FROZEN_V4_AUDIT_ATTEMPT_ID,
        "recovery_attempt_id": auditor.V4_RECOVERY_ATTEMPT_ID,
        "payload_canonical_sha256": auditor.FROZEN_V4_AUTH_PAYLOAD_CANONICAL_SHA256,
        "test_evidence_canonical_sha256": (
            auditor.FROZEN_V4_TEST_EVIDENCE_CANONICAL_SHA256
        ),
        "recovered_afterstate_canonical_sha256": (
            auditor.V4_RECOVERED_AFTERSTATE_CANONICAL_SHA256
        ),
    }
    truncation = {
        "authorization": dict(auditor.FROZEN_V4_AUTH_IDENTITY),
        "independent_review": dict(auditor.FROZEN_V4_REVIEW_IDENTITY),
        "canonical_go": {
            "path": auditor.FROZEN_V4_CANONICAL_GO_RELATIVE,
            "present": False,
        },
        "actual_review_payload_canonical_sha256": (
            auditor.FROZEN_V4_ACTUAL_REVIEW_PAYLOAD_CANONICAL_SHA256
        ),
        "intended_review_payload_canonical_sha256": (
            auditor.FROZEN_V4_INTENDED_REVIEW_PAYLOAD_CANONICAL_SHA256
        ),
        "actual_recovered_afterstate_canonical_sha256": (
            auditor.FROZEN_V4_ACTUAL_AFTERSTATE_CANONICAL_SHA256
        ),
        "expected_recovered_afterstate_canonical_sha256": (
            auditor.V4_RECOVERED_AFTERSTATE_CANONICAL_SHA256
        ),
        "actual_progress_inventory_count": 101,
        "expected_progress_inventory_count": 104,
        "intended_review_pretty_serialization": dict(
            auditor.FROZEN_V4_INTENDED_REVIEW_PRETTY_IDENTITY
        ),
        "v4_auditor_execution_started": False,
        "v4_auditor_execution_authorized": False,
    }
    return {
        "incident": incident,
        "incident_correction": correction,
        "v4_authorization_created_utc": "2026-08-10T20:00:00.000000Z",
        "v4_review_created_utc": "2026-08-10T21:00:00.000000Z",
        "incident_created_utc": "2026-08-10T22:00:00.000000Z",
        "incident_correction_created_utc": "2026-08-10T23:59:59.000000Z",
        "v4_authorization_base": base,
        "v4_review_transport_truncation_state": truncation,
        "recovered_afterstate_commitment": dict(
            auditor.V5_RECOVERED_AFTERSTATE_COMMITMENT
        ),
        "metadata_only_recovered_closure": {
            "mode": "PATH_TYPE_SIZE_METADATA_ONLY_NO_RECOVERED_BYTES_READ"
        },
        "preaudit_zero_mutation_snapshot": snapshot,
    }


def fake_evidence(code: dict[str, object]) -> dict[str, object]:
    incident, correction = subject._expected_incident_identities()
    evidence_bound = {
        "v4_authorization": dict(auditor.FROZEN_V4_AUTH_IDENTITY),
        "v4_review_transport_truncation_incident": incident,
        "v4_review_transport_truncation_incident_correction": correction,
        **code["bound_identities"],
    }
    runs = {
        role: {
            "command": auditor._v5_test_command(relative),
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
        "artifact_type": "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_V5_TEST_EVIDENCE",
        "status": "PASS_FROZEN_V5_AUDITOR_AND_SEALER_TESTS",
        "created_utc": "2026-08-11T00:00:00.000000Z",
        "bound_identities": evidence_bound,
        "source_compile": {
            "method": "compile_exact_source_no_pyc",
            "result": "PASS",
            "files": [
                code["bound_identities"][role]
                for role in (
                    "v5_auditor", "v5_auditor_test", "v5_sealer", "v5_sealer_test"
                )
            ],
        },
        "pytest_isolation": "ONE_TEST_FILE_PER_CLEAN_NETWORK_DENIED_SUBPROCESS",
        "test_runs": runs,
        "required_test_names": list(auditor.V5_REQUIRED_TEST_NAMES),
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


def test_contract_is_exact_eight_sources_four_runs_fourteen_temps_twelve_reports() -> None:
    assert len(subject.BOUND_SOURCE_ROLES) == 8
    assert len(subject.BOUND_IDENTITY_ROLES) == 11
    assert set(subject.BOUND_SOURCE_PATHS) == subject.BOUND_SOURCE_ROLES
    assert set(subject.TEST_RELATIVES) == auditor.V5_TEST_RUN_KEYS
    assert len(subject.TEST_RELATIVES) == 4
    assert tuple(subject.CONTROL_TEMP_RELATIVES) == auditor._v5_control_destination_relatives()
    assert len(subject.CONTROL_TEMP_RELATIVES) == 14
    assert subject.POSTRUN_REPORT_CANDIDATES == auditor.V5_POSTRUN_REPORT_CANDIDATES
    assert len(subject.POSTRUN_REPORT_CANDIDATES) == 12


def test_code_state_binds_frozen_v4_and_executing_v5_roles() -> None:
    state = subject.collect_code_state(subject.REPO)
    assert set(state) == {"bound_identities"}
    assert set(state["bound_identities"]) == subject.BOUND_SOURCE_ROLES
    for role, expected in auditor.FROZEN_V4_SOURCE_IDENTITIES.items():
        assert state["bound_identities"][role] == expected
    assert state["bound_identities"]["v5_auditor"] == (
        subject.EXPECTED_V5_AUDITOR_IDENTITY
    )
    assert state["bound_identities"]["v5_auditor_test"] == (
        subject.EXPECTED_V5_AUDITOR_TEST_IDENTITY
    )


def test_v5_review_checks_accept_new_v4_ast_identity_key() -> None:
    assert "v4_data_and_replay_logic_ast_identical" in auditor.V5_REVIEW_CHECKS
    checks = {key: True for key in auditor.V5_REVIEW_CHECKS}
    assert set(checks) == auditor.V5_REVIEW_CHECKS
    assert all(checks.values())


@pytest.mark.parametrize("role", ("v5_auditor", "v5_auditor_test"))
@pytest.mark.parametrize("field", ("size_bytes", "sha256"))
def test_frozen_v5_auditor_or_test_size_sha_tamper_fails_self_validator(
    tmp_path: Path, role: str, field: str
) -> None:
    root = control_root(tmp_path)
    code = fake_code_state(tmp_path)
    inputs = fake_inputs()
    evidence = fake_evidence(code)
    tampered = dict(code["bound_identities"][role])
    tampered[field] = (
        int(tampered[field]) + 1 if field == "size_bytes" else "f" * 64
    )
    code["bound_identities"][role] = tampered
    evidence["bound_identities"][role] = tampered
    source_index = (
        "v5_auditor", "v5_auditor_test", "v5_sealer", "v5_sealer_test"
    ).index(role)
    evidence["source_compile"]["files"][source_index] = tampered
    with pytest.raises(subject.PostrunAuditSealError, match="frozen auditor/test"):
        subject.build_authorization(
            root,
            code,
            inputs,
            evidence,
            created_utc="2026-08-11T00:00:01.000000Z",
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
    if role in {"v5_auditor", "v5_auditor_test"}:
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
    source = tmp_path / "audit_v5.py"
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
def test_all_fourteen_control_destinations_scan_temporary_siblings(
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
def test_every_v2_to_v5_report_candidate_fails_zero_snapshot(
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


def test_build_authorization_is_exact_auth35_and_never_embeds_full_inventory(
    tmp_path: Path
) -> None:
    root = control_root(tmp_path)
    code = fake_code_state(tmp_path)
    inputs = fake_inputs()
    evidence = fake_evidence(code)
    payload = subject.build_authorization(
        root,
        code,
        inputs,
        evidence,
        created_utc="2026-08-11T00:00:01.000000Z",
    )
    assert set(payload) == auditor.V5_AUTH_KEYS
    assert len(payload) == 35
    assert payload["schema_version"] == 5
    assert payload["incident_correction"] == inputs["incident_correction"]
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
    assert len(payload["test_evidence"]["bound_identities"]) == 11


@pytest.mark.parametrize(
    "chronology",
    (
        "review_not_before_incident",
        "incident_not_before_correction",
        "authorization_before_correction",
        "evidence_after_authorization",
    ),
)
def test_authorization_and_evidence_chronology_fails_closed(
    tmp_path: Path, chronology: str
) -> None:
    root = control_root(tmp_path)
    code = fake_code_state(tmp_path)
    inputs = fake_inputs()
    evidence = fake_evidence(code)
    if chronology == "review_not_before_incident":
        inputs["v4_review_created_utc"] = inputs["incident_created_utc"]
    elif chronology == "incident_not_before_correction":
        inputs["incident_created_utc"] = inputs[
            "incident_correction_created_utc"
        ]
    elif chronology == "authorization_before_correction":
        inputs["incident_correction_created_utc"] = "2026-08-11T00:00:02.000000Z"
    else:
        evidence["created_utc"] = "2026-08-11T00:00:02.000000Z"
    with pytest.raises(subject.PostrunAuditSealError, match="chronology|predates|postdates|evidence"):
        subject.build_authorization(
            root,
            code,
            inputs,
            evidence,
            created_utc="2026-08-11T00:00:01.000000Z",
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "missing", "extra", "schema", "status", "attempt", "incident",
        "correction", "base", "truncation", "source", "recovery",
        "commitment", "snapshot", "runtime", "command", "review", "evidence",
    ),
)
def test_authorization_schema_and_binding_tamper_fails(
    tmp_path: Path, mutation: str
) -> None:
    root = control_root(tmp_path)
    code = fake_code_state(tmp_path)
    inputs = fake_inputs()
    payload = subject.build_authorization(
        root,
        code,
        inputs,
        fake_evidence(code),
        created_utc="2026-08-11T00:00:01.000000Z",
    )
    if mutation == "missing":
        payload.pop("incident_correction")
    elif mutation == "extra":
        payload["unknown"] = True
    elif mutation == "schema":
        payload["schema_version"] = 4
    elif mutation == "status":
        payload["status"] = "PASS"
    elif mutation == "attempt":
        payload["audit_attempt_id"] = "postrun_audit_v5__wildcard"
    elif mutation == "incident":
        payload["incident"] = {"tampered": True}
    elif mutation == "correction":
        payload["incident_correction"] = {"tampered": True}
    elif mutation == "base":
        payload["v4_authorization_base"] = {"tampered": True}
    elif mutation == "truncation":
        payload["v4_review_transport_truncation_state"] = {"tampered": True}
    elif mutation == "source":
        payload["v5_auditor"] = {"tampered": True}
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
    assert subject.required_audit_command(root) == auditor._v5_required_command(
        root, root / subject.AUTH_RELATIVE, root / subject.GO_RELATIVE
    )


def test_incident_chain_exact_one_override_is_required_during_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = control_root(tmp_path)
    populate_historical_namespace(root)
    code = fake_code_state(tmp_path)
    incident, correction = subject._expected_incident_identities()
    calls: list[tuple[object, object]] = []

    def incident_chain(_root: Path, first: object, second: object) -> dict[str, object]:
        calls.append((first, second))
        return {
            "incident": {"created_utc": "2026-08-10T22:00:00.000000Z"},
            "correction": {"created_utc": "2026-08-10T23:59:59.000000Z"},
            "incident_identity": incident,
            "correction_identity": correction,
            "exact_one_override_applied": True,
        }

    afterstate = {"stub": True}
    monkeypatch.setattr(auditor, "_v5_validate_incident_chain", incident_chain)
    monkeypatch.setattr(
        subject,
        "_load_v4_authorization_base",
        lambda _root: (
            fake_inputs()["v4_authorization_base"],
            {"created_utc": fake_inputs()["v4_authorization_created_utc"]},
            afterstate,
        ),
    )
    monkeypatch.setattr(
        subject,
        "_load_v4_transport_truncation_state",
        lambda *_args: (
            fake_inputs()["v4_review_transport_truncation_state"],
            fake_inputs()["v4_review_created_utc"],
        ),
    )
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
    assert calls == [(incident, correction)]
    assert result["incident_correction"] == correction


def test_isolated_evidence_uses_four_exact_guarded_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code = fake_code_state(tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(subject, "require_canonical_module_entrypoint", lambda _root: None)
    monkeypatch.setattr(subject, "collect_code_state", lambda _root: code)
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
    assert calls == [auditor._v5_test_command(path) for path in subject.TEST_RELATIVES.values()]
    assert set(evidence["test_runs"]) == auditor.V5_TEST_RUN_KEYS
    assert evidence["all_exit_codes_zero"] is True


def test_isolated_evidence_failure_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code = fake_code_state(tmp_path)
    monkeypatch.setattr(subject, "require_canonical_module_entrypoint", lambda _root: None)
    monkeypatch.setattr(subject, "collect_code_state", lambda _root: code)
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
    payload = {"schema_version": 5, "artifact_type": "TEST_AUTH"}
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
    payload = {"schema_version": 5, "artifact_type": "TEST_AUTH"}
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

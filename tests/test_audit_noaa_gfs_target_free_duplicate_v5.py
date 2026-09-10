from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import audit_noaa_gfs_target_free_duplicate_v5 as subject


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "artifacts" / "baram2026_ncei_scada_longrun_20260810_v2"


@pytest.fixture(scope="module")
def amendment() -> dict:
    return subject._validate_amendment(ROOT)[0]


@pytest.fixture(scope="module")
def amendment_v4(amendment: dict) -> dict:
    _, identity = subject._validate_amendment(ROOT)
    return subject._validate_amendment_v4(ROOT, amendment, identity)[0]


@pytest.fixture(scope="module")
def amendment_v5(amendment: dict, amendment_v4: dict) -> dict:
    _, identity = subject._validate_amendment(ROOT)
    _, identity_v4 = subject._validate_amendment_v4(ROOT, amendment, identity)
    _, incident_identity = subject._validate_failure_incident(
        ROOT, REPO, identity, identity_v4
    )
    return subject._validate_amendment_v5(
        ROOT,
        amendment,
        identity,
        amendment_v4,
        identity_v4,
        incident_identity,
    )[0]


def test_module_import_does_not_import_numpy_or_pyarrow() -> None:
    code = (
        "import sys; "
        "import scripts.audit_noaa_gfs_target_free_duplicate_v5; "
        "assert 'numpy' not in sys.modules; "
        "assert 'pyarrow' not in sys.modules"
    )
    completed = subprocess.run(
        [str(REPO / ".venv" / "Scripts" / "python.exe"), "-B", "-c", code],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_final_amendment_physical_and_canonical_identity() -> None:
    payload, identity = subject._validate_amendment(ROOT)
    assert len(payload) == 26
    assert identity == {
        "path": subject.AMENDMENT_REL,
        "size_bytes": 102361,
        "sha256": "f9d072fc24ab29608efc6483098ada5a8d5e8129e1d1bd478e1d84438a2e9c14",
        "canonical_size_bytes": 84601,
        "canonical_sha256": "ae5c5d05e63169112799c48a8ede52e8ce29319bc5d9e96927175ccbfd9b91d1",
    }


def test_final_v4_physical_canonical_and_exact2_overlay(amendment: dict) -> None:
    _, v3_identity = subject._validate_amendment(ROOT)
    payload, identity = subject._validate_amendment_v4(ROOT, amendment, v3_identity)
    assert len(payload) == 18
    assert identity == {
        "path": subject.AMENDMENT_V4_REL,
        "size_bytes": 14385,
        "sha256": "9f8abbe878bd10e7b91f78a03f44389a4973c8c1cb7da3c94752487957be7980",
        "canonical_size_bytes": 12501,
        "canonical_sha256": "3ad1f38508b373aac70ee19b4253e91b9104f9527cb77952121757226f35abed",
    }
    assert payload["correction_scope"]["superseded_pointers_exact_order"] == list(subject.V4_POINTERS)
    assert payload["correction_scope"]["correction_count_exact"] == 2
    assert payload["corrected_parquet_serialization"]["explicit_write_table_kwargs_exact"] == subject.PARQUET_WRITE_KWARGS_EXACT
    assert payload["corrected_csv_boundary_serialization"]["corrected_v3_literal"] == subject.V4_CSV_BOUNDARY_LITERAL


def test_final_v5_and_incident_identity_chain(
    amendment: dict, amendment_v4: dict
) -> None:
    _, v3_identity = subject._validate_amendment(ROOT)
    _, v4_identity = subject._validate_amendment_v4(ROOT, amendment, v3_identity)
    incident, incident_identity = subject._validate_failure_incident(
        ROOT, REPO, v3_identity, v4_identity
    )
    payload, identity = subject._validate_amendment_v5(
        ROOT,
        amendment,
        v3_identity,
        amendment_v4,
        v4_identity,
        incident_identity,
    )
    assert len(incident) == 20
    assert incident_identity["sha256"] == subject.INCIDENT_SHA256
    assert len(payload) == 24
    assert identity == {
        "path": subject.AMENDMENT_V5_REL,
        "size_bytes": 59798,
        "sha256": subject.AMENDMENT_V5_SHA256,
        "canonical_size_bytes": 48167,
        "canonical_sha256": subject.AMENDMENT_V5_CANONICAL_SHA256,
    }


def test_amendment_bound_input_count_and_release_paths(
    amendment: dict, amendment_v5: dict
) -> None:
    assert len(amendment["bound_inputs"]) == 31
    assert amendment_v5["code_and_test_paths_exact"] == subject.CODE_ROLE_PATHS
    assert amendment_v5["control_contract"]["control_paths_exact"] == {
        "authorization": subject.AUTH_REL,
        "code_seal": subject.CODE_SEAL_REL,
        "independent_go": subject.GO_REL,
        "independent_postrun": subject.POSTRUN_REL,
        "independent_review": subject.REVIEW_REL,
    }


def test_all_31_bound_input_physical_identities_rehash(amendment: dict) -> None:
    subject._validate_bound_inputs(ROOT, amendment)


def test_bound_runtime_package_executable_and_native_identities(amendment: dict) -> None:
    subject._validate_runtime_identity(amendment)


def test_amendment_output_counts_are_exact(amendment: dict) -> None:
    schemas = amendment["output_contract"]["output_record_schemas_exact"]
    assert schemas["TARGET_FREE_DUPLICATE_METRICS_csv"]["row_slots_exact"] == 333
    assert schemas["TARGET_FREE_STABILITY_BY_YEAR_SEASON_FH_SITE_csv"]["row_slots_exact"] == 348
    assert schemas["TARGET_FREE_DUPLICATE_OOF_V3_parquet"]["row_slots_exact"] == 352512
    assert schemas["TARGET_FREE_FIT_LEDGER_V3_json"]["unit_slots_exact"] == 1332
    assert schemas["TARGET_FREE_DISTRIBUTION_STABILITY_V3_json"]["record_slots_exact"] == 624
    assert schemas["TARGET_FREE_DISTRIBUTION_STABILITY_V3_json"]["endpoint_pooled_metric_records_exact"] == 3
    assert schemas["TARGET_FREE_DERIVED_WIND_DIAGNOSTICS_V3_parquet"]["row_slots_exact"] == 414720
    assert subject.OUTPUT_COUNTS == {
        "aliases": 2,
        "runner_staged_outputs": 7,
        "metrics_rows": 333,
        "stability_rows": 348,
        "oof_rows": 352512,
        "fit_ledger_rows": 1332,
        "distribution_records": 624,
        "endpoint_pooled_metric_records": 3,
        "derived_rows": 414720,
    }


def test_exact11_positive_and_negative_sets_match(amendment_v5: dict) -> None:
    output = amendment_v5["output_contract"]
    assert output["completed_positive_paths_exact"] == output["completed_negative_or_ambiguity_paths_exact"]
    assert len(output["completed_positive_paths_exact"]) == 11
    assert output["completed_positive_paths_exact"][-1].endswith("TRACK_A_TARGET_FREE_RUN_MANIFEST.json")


def test_future_lock_and_manifest_exact_top_contracts_are_frozen() -> None:
    assert len(subject.LOCK_TOP_KEYS) == 16
    assert len(subject.MANIFEST_TOP_KEYS) == 14
    assert subject.LOCK_ARTIFACT_TYPE == "TARGET_FREE_FAMILY_LOCK_V5"
    assert subject.MANIFEST_ARTIFACT_TYPE == "TRACK_A_TARGET_FREE_RUN_MANIFEST_V5"
    assert subject.MANIFEST_STATUS == "COMMITTED_V5_TARGET_FREE_DECISION"
    assert subject.SITE_GROUP_SCOPE_KEYS == (
        "external_site_rows",
        "external_group_rows",
        "valid_timestamps",
        "canonical_site_order",
        "site_join_key_exact",
        "group_join_key_exact",
        "metadata_parity_required",
    )
    assert subject.SPATIAL_TRANSFORM_KEYS == ("spatial_method", "group_construction")
    assert subject.CODE_CONFIG_RUNTIME_KEYS == {
        "amendment",
        "amendment_v4",
        "amendment_v5",
        "code_identities",
        "runtime_identity",
    }
    assert subject.OOF_SLOT_COUNT_KEYS == {
        "planned_oof_slots",
        "completed_oof_slots",
        "skipped_oof_slots",
        "independent_selected_oof_slots",
    }
    assert subject.FINAL_OUTPUT_ORDER == subject.RUNNER_OUTPUTS + (
        "TARGET_FREE_FAMILY_LOCK.json",
    )
    assert "TRACK_A_TARGET_FREE_RUN_MANIFEST.json" not in subject.FINAL_OUTPUT_ORDER


def test_strict_json_rejects_duplicate_keys() -> None:
    with pytest.raises(subject.AuditError, match="duplicate JSON key"):
        subject.strict_json_bytes(b'{"a":1,"a":2}', label="duplicate")


@pytest.mark.parametrize("literal", [b"NaN", b"Infinity", b"-Infinity"])
def test_strict_json_rejects_nonfinite_constants(literal: bytes) -> None:
    with pytest.raises(subject.AuditError, match="non-finite JSON constant"):
        subject.strict_json_bytes(b'{"a":' + literal + b"}", label="nonfinite")


def test_strict_json_rejects_bom_and_non_object() -> None:
    with pytest.raises(subject.AuditError, match="BOM"):
        subject.strict_json_bytes(b"\xef\xbb\xbf{}", label="bom")
    with pytest.raises(subject.AuditError, match="top-level object"):
        subject.strict_json_bytes(b"[]", label="array")


def test_python_pinned_json_serializers() -> None:
    value = {"z": 1e-9, "a": "한글"}
    compact = subject.canonical_json_bytes(value)
    assert compact == '{"a":"한글","z":1e-09}'.encode()
    assert subject.pretty_json_bytes(value, ensure_ascii=False).endswith(b"\n")
    assert b"\\u" in subject.pretty_json_bytes(value, ensure_ascii=True)


@pytest.mark.parametrize(
    "value",
    [
        "2026-08-11T00:00:00Z",
        "2026-08-11T00:00:00.1Z",
        "2026-08-11T00:00:00.0000000Z",
        "2026-08-11T00:00:60.000000Z",
        "not-a-time",
    ],
)
def test_timestamp_parser_rejects_noncanonical_values(value: str) -> None:
    with pytest.raises(subject.AuditError):
        subject._parse_utc_microseconds(value, "time")


def test_timestamp_parser_is_exact_integer_microseconds() -> None:
    left = subject._parse_utc_microseconds("2026-08-11T00:00:00.000001Z", "left")
    right = subject._parse_utc_microseconds("2026-08-11T00:00:00.000002Z", "right")
    assert right - left == 1


def test_attempt_id_requires_compact_calendar_valid_microsecond_utc() -> None:
    valid = "target_free_duplicate_v5__20260811T123456123456Z"
    assert subject._validate_attempt_id(valid) == valid
    for invalid in (
        "target_free_duplicate_v5__20260230T123456123456Z",
        "target_free_duplicate_v5__20260811T246000123456Z",
        "target_free_duplicate_v5__20260811T12345612345Z",
        "other__20260811T123456123456Z",
    ):
        with pytest.raises(subject.AuditError):
            subject._validate_attempt_id(invalid)


def test_artifact_path_rejects_absolute_and_traversal(tmp_path: Path) -> None:
    with pytest.raises(subject.AuditError):
        subject._artifact_path(tmp_path, "../escape")
    with pytest.raises(subject.AuditError):
        subject._artifact_path(tmp_path, str((tmp_path / "absolute").resolve()))


def test_file_identity_rehashes_exact_bytes(tmp_path: Path) -> None:
    path = tmp_path / "x.bin"
    path.write_bytes(b"abc")
    record = {"path": "x.bin", "size_bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}
    assert subject.verify_file_identity(record, path, label="x", expected_path="x.bin") == record
    record["sha256"] = "0" * 64
    with pytest.raises(subject.AuditError):
        subject.verify_file_identity(record, path, label="x", expected_path="x.bin")


def test_control_top_key_sets_and_scopes_are_frozen() -> None:
    assert len(subject.CODE_SEAL_TOP_KEYS) == 13
    assert len(subject.AUTH_TOP_KEYS) == 17
    assert len(subject.REVIEW_TOP_KEYS) == 16
    assert len(subject.GO_TOP_KEYS) == 17
    assert len(subject.POSTRUN_TOP_KEYS) == 18
    assert len(subject.AUDITOR_STDOUT_TOP_KEYS) == 18
    assert {"amendment_v4", "amendment_v5"} <= subject.AUDITOR_STDOUT_TOP_KEYS
    assert len(subject.DOCUMENTARY_SCOPE) == 8
    assert len(subject.AUTHORITY_SCOPE) == 14
    assert len(subject.GO_SCOPE) == 9
    assert len(subject.REVIEW_CHECKS) == 12 and all(subject.REVIEW_CHECKS.values())
    assert subject.EXECUTION_BUDGET == {
        "attempt_count": 1,
        "alias_count": 2,
        "primary_model_fit_units": 72,
        "analytic_affine_fit_units": 1260,
        "total_decision_fit_units": 1332,
        "fit_ledger_rows": 1332,
        "oof_rows": 352512,
        "metrics_csv_rows": 333,
        "stability_csv_rows": 348,
        "distribution_records": 624,
        "endpoint_pooled_records": 3,
        "derived_rows": 414720,
        "runner_staged_outputs": 7,
        "final_complete_outputs": 11,
    }


def test_common_control_constants_match_runner_and_sealer() -> None:
    runner = pytest.importorskip("scripts.run_noaa_gfs_target_free_duplicate_v5")
    sealer = pytest.importorskip("scripts.seal_noaa_gfs_target_free_duplicate_v5")

    assert dict(runner.CONTROL_OUTPUT_NAMESPACE) == subject.OUTPUT_NAMESPACE == sealer.OUTPUT_NAMESPACE
    assert dict(runner.DOCUMENTARY_AUTHORITY_SCOPE) == subject.DOCUMENTARY_SCOPE == sealer.DOCUMENTARY_SCOPE
    assert dict(runner.EXECUTION_AUTHORITY_SCOPE) == subject.AUTHORITY_SCOPE == sealer.AUTHORIZATION_SCOPE
    assert dict(runner.GO_AUTHORITY_SCOPE) == subject.GO_SCOPE == sealer.GO_SCOPE
    assert dict(runner.EXECUTION_BUDGET) == subject.EXECUTION_BUDGET == sealer.EXECUTION_BUDGET
    assert dict(runner.REQUIRED_NEXT_CONTROLS) == subject.REQUIRED_NEXT_CONTROLS == sealer.REQUIRED_NEXT_CONTROLS
    assert dict(runner.SINGLE_ATTEMPT) == subject.SINGLE_ATTEMPT == sealer.SINGLE_ATTEMPT
    assert set(runner.INDEPENDENT_CHECK_KEYS) == set(subject.REVIEW_CHECKS) == set(sealer.REVIEW_CHECK_KEYS)
    assert dict(runner.FORBIDDEN_ACCESS_ATTESTATION) == sealer.DECISION_FORBIDDEN_ACCESS_ATTESTATION
    assert runner.ATTEMPT_ID_PATTERN.pattern == subject.ATTEMPT_ID_RE.pattern == sealer.ATTEMPT_ID.pattern
    assert tuple(runner.RUNNER_ARTIFACT_BASENAMES) == subject.RUNNER_OUTPUTS == tuple(sealer.RUNNER_PUBLICATION_ORDER)
    assert sealer.AUDITOR_OUTPUT_COUNTS == subject.OUTPUT_COUNTS
    assert sealer.AUDITOR_POSTSTATE == subject.POSTSTATE
    assert tuple(runner.RAW_COMPONENT_RESULT_KEYS) == subject.RAW_COMPONENT_RESULT_KEYS
    assert tuple(runner.ENDPOINT_VETO_RESULT_KEYS) == subject.ENDPOINT_VETO_RESULT_KEYS
    assert tuple(runner.THRESHOLD_WITNESS_KEYS) == subject.THRESHOLD_WITNESS_KEYS
    assert tuple(runner.TIED_WITNESS_KEYS) == subject.TIED_WITNESS_KEYS
    assert tuple(runner.RAW_CLASSIFICATIONS) == subject.RAW_CLASSIFICATIONS
    assert set(runner.ENDPOINT_VERDICTS) == subject.ENDPOINT_VERDICTS == set(sealer.ENDPOINT_VERDICTS)
    assert dict(runner.CUTOFF_VALUES) == subject.CUTOFF_VALUES == sealer.CUTOFF_VALUES
    assert set(runner.TIE_RESOLUTIONS) == subject.TIE_RESOLUTIONS == set(sealer.TIE_RESOLUTIONS)
    assert [len(sealer.CONTROL_SPECS[key]["keys"]) for key in (
        "code_seal",
        "authorization",
        "review",
        "go",
        "postrun",
    )] == [13, 17, 16, 17, 18]
    assert len(sealer.AUDITOR_STDOUT_KEYS) == 18
    assert len(sealer.RUNNER_STDOUT_KEYS) == len(runner.RUNNER_STDOUT_KEYS) == 11
    assert {"amendment_v4", "amendment_v5"} <= set(sealer.AUDITOR_STDOUT_KEYS)
    assert "amendment_v4" not in sealer.RUNNER_STDOUT_KEYS
    assert "amendment_v4" not in runner.RUNNER_STDOUT_KEYS
    assert "amendment_v5" not in sealer.RUNNER_STDOUT_KEYS
    assert "amendment_v5" not in runner.RUNNER_STDOUT_KEYS
    assert (
        dict(runner.PARQUET_WRITE_TABLE_KWARGS_EXACT)
        == subject.PARQUET_WRITE_KWARGS_EXACT
        == sealer.V4_EXPLICIT_WRITE_TABLE_KWARGS
    )
    assert (
        dict(runner.PYARROW_WRITE_TABLE_UNLISTED_DEFAULTS_EXACT)
        == subject.PYARROW_WRITE_TABLE_UNLISTED_DEFAULTS_EXACT
        == sealer.V4_UNLISTED_WRITE_TABLE_DEFAULTS
    )


def test_required_control_paths_and_single_attempt_are_frozen() -> None:
    assert subject.REQUIRED_NEXT_CONTROLS == {
        "authorization": subject.AUTH_REL,
        "independent_review": subject.REVIEW_REL,
        "independent_go": subject.GO_REL,
        "postrun_pass": subject.POSTRUN_REL,
    }
    assert subject.SINGLE_ATTEMPT["attempt_number"] == 1
    assert subject.SINGLE_ATTEMPT["retry_allowed"] is False
    assert subject.SINGLE_ATTEMPT["prior_v5_failure_incident"] is None


def test_required_command_shapes(amendment: dict) -> None:
    commands = subject._expected_commands(ROOT, amendment)
    assert list(commands) == ["runner", "auditor", "sealer"]
    assert commands["runner"][1:4] == ["-B", "-m", "scripts.run_noaa_gfs_target_free_duplicate_v5"]
    assert commands["auditor"][1:4] == ["-B", "-m", "scripts.audit_noaa_gfs_target_free_duplicate_v5"]
    assert commands["sealer"][1:4] == ["-B", "-m", "scripts.seal_noaa_gfs_target_free_duplicate_v5"]
    assert commands["sealer"][-2:] == ["--postrun-pass", str(ROOT / subject.POSTRUN_REL)]


def test_required_test_command_shapes(amendment: dict) -> None:
    commands = subject._expected_test_commands(REPO, amendment)
    assert list(commands) == ["runner", "auditor", "sealer"]
    for role, command in commands.items():
        assert command[1:8] == ["-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", str(REPO / subject.CODE_ROLE_PATHS[role + "_test"])]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(subject.pretty_json_bytes(payload, ensure_ascii=True))


def _build_control_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    root = tmp_path / "artifact"
    repo = tmp_path / "repo"
    amendment_path = root / subject.AMENDMENT_REL
    amendment_path.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / subject.AMENDMENT_REL, amendment_path)
    amendment, amendment_identity = subject._validate_amendment(root)
    amendment_v4_path = root / subject.AMENDMENT_V4_REL
    amendment_v4_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / subject.AMENDMENT_V4_REL, amendment_v4_path)
    amendment_v4, amendment_v4_identity = subject._validate_amendment_v4(
        root, amendment, amendment_identity
    )
    incident_path = root / subject.INCIDENT_REL
    incident_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / subject.INCIDENT_REL, incident_path)
    _, incident_identity = subject._validate_failure_incident(
        root,
        repo,
        amendment_identity,
        amendment_v4_identity,
        validate_historical_identities=False,
    )
    amendment_v5_path = root / subject.AMENDMENT_V5_REL
    amendment_v5_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / subject.AMENDMENT_V5_REL, amendment_v5_path)
    _, amendment_v5_identity = subject._validate_amendment_v5(
        root,
        amendment,
        amendment_identity,
        amendment_v4,
        amendment_v4_identity,
        incident_identity,
    )

    code_identities = {}
    for role, relative in subject.CODE_ROLE_PATHS.items():
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, destination)
        code_identities[role] = subject.file_identity(destination, destination.resolve().as_posix())

    commands = subject._expected_test_commands(repo, amendment)
    results = {
        role: {
            "test_file": subject.CODE_ROLE_PATHS[role + "_test"],
            "exit_code": 0,
            "passed": 1,
            "skipped": 0,
            "deselected": 0,
            "summary": "1 passed in 0.01s",
        }
        for role in ("runner", "auditor", "sealer")
    }
    seal = {
        "schema_version": 5,
        "artifact_type": "TARGET_FREE_DUPLICATE_CODE_SEAL_V5",
        "status": "SEALED_V5_CODE_AND_TESTS_NO_EXECUTION_AUTHORITY",
        "created_utc": "2026-08-11T18:05:00.000000Z",
        "amendment": amendment_identity,
        "amendment_v4": amendment_v4_identity,
        "amendment_v5": amendment_v5_identity,
        "code_identities": code_identities,
        "test_evidence": {
            "commands": commands,
            "results": results,
            "all_exit_codes_zero": True,
            "all_expected_test_files_covered": True,
            "network_denied": True,
            "pycache_disabled": True,
            "code_identities_revalidated": True,
            "completed_utc": "2026-08-11T18:04:00.000000Z",
        },
        "output_namespace": subject.OUTPUT_NAMESPACE,
        "authority_scope": subject.DOCUMENTARY_SCOPE,
        "required_next_controls": subject.REQUIRED_NEXT_CONTROLS,
        "publisher": "V5_SEALER_CODE_SEAL_MODE_CREATE_IF_ABSENT_ONLY",
    }
    _write_json(root / subject.CODE_SEAL_REL, seal)
    seal_identity = subject._control_identity(root, subject.CODE_SEAL_REL)
    attempt = "target_free_duplicate_v5__20260811T180600000000Z"
    auth = {
        "schema_version": 5,
        "artifact_type": "TARGET_FREE_DUPLICATE_EXECUTION_AUTHORIZATION_V5",
        "status": "AUTHORIZED_V5_SINGLE_ATTEMPT_PENDING_INDEPENDENT_REVIEW_AND_GO",
        "created_utc": "2026-08-11T18:06:00.000000Z",
        "attempt_id": attempt,
        "amendment": amendment_identity,
        "amendment_v4": amendment_v4_identity,
        "amendment_v5": amendment_v5_identity,
        "code_seal": seal_identity,
        "code_identities": code_identities,
        "output_namespace": subject.OUTPUT_NAMESPACE,
        "execution_budget": subject.EXECUTION_BUDGET,
        "required_commands": subject._expected_commands(root, amendment),
        "required_review": {"path": subject.REVIEW_REL, "status_required": "PASS_V5_EXECUTION_REVIEW_PENDING_INDEPENDENT_GO"},
        "required_go": {"path": subject.GO_REL, "status_required": "GO_V5_SINGLE_TARGET_FREE_ATTEMPT"},
        "authority_scope": subject.AUTHORITY_SCOPE,
        "publisher": "ROOT_AUTHORITY_APPEND_ONLY_ONLY",
    }
    _write_json(root / subject.AUTH_REL, auth)
    auth_identity = subject._control_identity(root, subject.AUTH_REL)
    review = {
        "schema_version": 5,
        "artifact_type": "TRACK_A_TARGET_FREE_DUPLICATE_EXECUTION_REVIEW_V5",
        "status": "PASS_V5_EXECUTION_REVIEW_PENDING_INDEPENDENT_GO",
        "created_utc": "2026-08-11T18:07:00.000000Z",
        "attempt_id": attempt,
        "amendment": amendment_identity,
        "amendment_v4": amendment_v4_identity,
        "amendment_v5": amendment_v5_identity,
        "code_seal": seal_identity,
        "authorization": auth_identity,
        "code_identities": code_identities,
        "output_namespace": subject.OUTPUT_NAMESPACE,
        "independent_checks": subject.REVIEW_CHECKS,
        "authority_scope": subject.DOCUMENTARY_SCOPE,
        "verdict": "PASS",
        "publisher": "INDEPENDENT_REVIEWER_APPEND_ONLY_ONLY",
    }
    _write_json(root / subject.REVIEW_REL, review)
    review_identity = subject._control_identity(root, subject.REVIEW_REL)
    go = {
        "schema_version": 5,
        "artifact_type": "TRACK_A_TARGET_FREE_DUPLICATE_EXECUTION_GO_V5",
        "status": "GO_V5_SINGLE_TARGET_FREE_ATTEMPT",
        "created_utc": "2026-08-11T18:08:00.000000Z",
        "attempt_id": attempt,
        "amendment": amendment_identity,
        "amendment_v4": amendment_v4_identity,
        "amendment_v5": amendment_v5_identity,
        "code_seal": seal_identity,
        "authorization": auth_identity,
        "independent_review": review_identity,
        "code_identities": code_identities,
        "output_namespace": subject.OUTPUT_NAMESPACE,
        "required_commands": subject._expected_commands(root, amendment),
        "authority_scope": subject.GO_SCOPE,
        "single_attempt": subject.SINGLE_ATTEMPT,
        "publisher": "INDEPENDENT_GO_REVIEWER_APPEND_ONLY_ONLY",
    }
    _write_json(root / subject.GO_REL, go)
    return root, repo, root / subject.AUTH_REL, root / subject.GO_REL


def test_full_synthetic_control_chain_passes_before_arrow(tmp_path: Path) -> None:
    root, repo, auth, go = _build_control_fixture(tmp_path)
    assert "pyarrow" not in subject.validate_predata_authority.__globals__.get("sys").modules
    context = subject.validate_predata_authority(
        root,
        auth,
        go,
        repo=repo,
        validate_bound_inputs=False,
        validate_runtime=False,
        validate_historical_state=False,
    )
    assert context.attempt_id == "target_free_duplicate_v5__20260811T180600000000Z"
    assert context.amendment_v4_identity["sha256"] == subject.AMENDMENT_V4_SHA256
    assert context.amendment_v5_identity["sha256"] == subject.AMENDMENT_V5_SHA256
    seal = subject.strict_json_file(root / subject.CODE_SEAL_REL)
    assert (
        subject._parse_utc_microseconds(context.amendment["created_utc"], "V3")
        < subject._parse_utc_microseconds(context.amendment_v4["created_utc"], "V4")
        < subject._parse_utc_microseconds(context.failure_incident["created_utc"], "incident")
        < subject._parse_utc_microseconds(context.amendment_v5["created_utc"], "V5")
        < subject._parse_utc_microseconds(seal["test_evidence"]["completed_utc"], "completed")
        <= subject._parse_utc_microseconds(seal["created_utc"], "seal")
    )


def _authorized_auditor_command_context(
    tmp_path: Path,
) -> tuple[subject.PredataContext, list[str]]:
    root, repo, auth, go = _build_control_fixture(tmp_path)
    context = subject.validate_predata_authority(
        root,
        auth,
        go,
        repo=repo,
        validate_bound_inputs=False,
        validate_runtime=False,
        validate_historical_state=False,
    )
    expected = context.authorization["required_commands"]["auditor"]
    assert type(expected) is list
    return context, expected


def test_actual_auditor_command_accepts_only_authorized_exact_argv(
    tmp_path: Path,
) -> None:
    context, expected = _authorized_auditor_command_context(tmp_path)
    subject._validate_actual_auditor_command(context, list(expected))
    assert all(Path(expected[index]).is_absolute() for index in (0, 5, 7, 9))


@pytest.mark.parametrize(
    "variant",
    (
        "extra",
        "reordered_flags",
        "missing_dash_b",
        "wrong_module",
        "wrong_absolute_interpreter",
        "relative_interpreter",
        "relative_root",
        "relative_authorization",
        "relative_go",
    ),
)
def test_actual_auditor_command_rejects_unauthorized_argv(
    tmp_path: Path, variant: str
) -> None:
    context, expected = _authorized_auditor_command_context(tmp_path)
    actual = list(expected)
    if variant == "extra":
        actual.append("--unexpected")
    elif variant == "reordered_flags":
        actual[4:8] = actual[6:8] + actual[4:6]
    elif variant == "missing_dash_b":
        del actual[1]
    elif variant == "wrong_module":
        actual[3] = "scripts.run_noaa_gfs_target_free_duplicate_v5"
    elif variant == "wrong_absolute_interpreter":
        actual[0] = str(Path(actual[0]).with_name("python-other.exe"))
    elif variant == "relative_interpreter":
        actual[0] = Path(actual[0]).name
    elif variant == "relative_root":
        actual[5] = Path(actual[5]).name
    elif variant == "relative_authorization":
        actual[7] = Path(actual[7]).name
    else:
        assert variant == "relative_go"
        actual[9] = Path(actual[9]).name
    with pytest.raises(subject.AuditError, match="actual auditor command"):
        subject._validate_actual_auditor_command(context, actual)


def test_actual_auditor_command_rejects_non_list_orig_argv(tmp_path: Path) -> None:
    context, expected = _authorized_auditor_command_context(tmp_path)
    with pytest.raises(subject.AuditError, match="sys.orig_argv must be a list"):
        subject._validate_actual_auditor_command(context, tuple(expected))


def test_audit_command_check_is_after_controls_and_before_staged_reads() -> None:
    tree = ast.parse((REPO / subject.CODE_ROLE_PATHS["auditor"]).read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "audit"
    )
    calls = {
        node.func.id: node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id
        in {
            "validate_predata_authority",
            "_validate_actual_auditor_command",
            "validate_staged_outputs",
        }
    }
    assert (
        calls["validate_predata_authority"]
        < calls["_validate_actual_auditor_command"]
        < calls["validate_staged_outputs"]
    )
    injection = inspect.signature(subject.main).parameters["actual_orig_argv_for_test"]
    assert injection.kind is inspect.Parameter.KEYWORD_ONLY
    assert injection.default is None


def test_test_evidence_summary_counts_and_long_duration_suffix_are_bound(tmp_path: Path) -> None:
    root, repo, auth, go = _build_control_fixture(tmp_path)
    seal_path = root / subject.CODE_SEAL_REL
    seal = subject.strict_json_file(seal_path)
    seal["test_evidence"]["results"]["auditor"].update(
        {
            "passed": 89,
            "skipped": 3,
            "deselected": 1,
            "summary": "89 passed, 3 skipped, 1 deselected in 152.06s (0:02:32)",
        }
    )
    _write_json(seal_path, seal)
    with pytest.raises(subject.AuditError, match="code-seal mismatch"):
        subject.validate_predata_authority(
            root,
            auth,
            go,
            repo=repo,
            validate_bound_inputs=False,
            validate_runtime=False,
        validate_historical_state=False,
        )
    subject._validate_test_evidence(seal["test_evidence"], repo, subject._validate_amendment(root)[0])
    seal["test_evidence"]["results"]["auditor"]["passed"] = 88
    with pytest.raises(subject.AuditError, match="summary/count"):
        subject._validate_test_evidence(seal["test_evidence"], repo, subject._validate_amendment(root)[0])
    seal["test_evidence"]["results"]["auditor"]["passed"] = 89
    seal["test_evidence"]["results"]["auditor"]["summary"] = "89 passed, 3 skipped, 1 deselected in 152.06s"
    subject._validate_test_evidence(seal["test_evidence"], repo, subject._validate_amendment(root)[0])
    seal["test_evidence"]["results"]["auditor"]["summary"] = (
        "89 passed, 3 skipped, 1 deselected in 61.00s (0:01:00)"
    )
    subject._validate_test_evidence(seal["test_evidence"], repo, subject._validate_amendment(root)[0])
    for invalid in (
        "89 passed, 3 skipped, 1 deselected in 61.01s (0:01:00)",
        "89 passed, 3 skipped, 1 deselected in 61.00s (0:01:02)",
        "89 passed, 3 skipped, 1 deselected in 61.00s (00:01:00)",
        "89 passed, 3 skipped, 1 deselected in 60.00s (0:00:59)",
    ):
        seal["test_evidence"]["results"]["auditor"]["summary"] = invalid
        with pytest.raises(subject.AuditError, match="pytest duration|pytest summary"):
            subject._validate_test_evidence(seal["test_evidence"], repo, subject._validate_amendment(root)[0])


def test_test_evidence_rejects_boolean_exit_code(tmp_path: Path) -> None:
    root, repo, _, _ = _build_control_fixture(tmp_path)
    seal = subject.strict_json_file(root / subject.CODE_SEAL_REL)
    seal["test_evidence"]["results"]["runner"]["exit_code"] = False
    with pytest.raises(subject.AuditError, match="exit code must be integer zero"):
        subject._validate_test_evidence(
            seal["test_evidence"], repo, subject._validate_amendment(root)[0]
        )


def test_control_chain_rejects_authorization_extra_key(tmp_path: Path) -> None:
    root, repo, auth, go = _build_control_fixture(tmp_path)
    payload = subject.strict_json_file(auth)
    payload["extra"] = True
    _write_json(auth, payload)
    with pytest.raises(subject.AuditError, match="exact keys"):
        subject.validate_predata_authority(root, auth, go, repo=repo, validate_bound_inputs=False, validate_runtime=False, validate_historical_state=False)


def test_control_chain_rejects_missing_or_wrong_v4_binding(tmp_path: Path) -> None:
    root, repo, auth, go = _build_control_fixture(tmp_path)
    seal_path = root / subject.CODE_SEAL_REL
    payload = subject.strict_json_file(seal_path)
    del payload["amendment_v4"]
    _write_json(seal_path, payload)
    with pytest.raises(subject.AuditError, match="exact keys"):
        subject.validate_predata_authority(
            root,
            auth,
            go,
            repo=repo,
            validate_bound_inputs=False,
            validate_runtime=False,
        validate_historical_state=False,
        )

    root, repo, auth, go = _build_control_fixture(tmp_path / "wrong")
    payload = subject.strict_json_file(auth)
    payload["amendment_v4"]["sha256"] = "0" * 64
    _write_json(auth, payload)
    with pytest.raises(subject.AuditError, match="V4 amendment mismatch"):
        subject.validate_predata_authority(
            root,
            auth,
            go,
            repo=repo,
            validate_bound_inputs=False,
            validate_runtime=False,
        validate_historical_state=False,
        )


def test_control_chain_rejects_legacy_execution_budget_keys(tmp_path: Path) -> None:
    root, repo, auth, go = _build_control_fixture(tmp_path)
    payload = subject.strict_json_file(auth)
    payload["execution_budget"] = {
        **subject.EXECUTION_BUDGET,
        "attempt_number": payload["execution_budget"]["attempt_count"],
    }
    del payload["execution_budget"]["attempt_count"]
    _write_json(auth, payload)
    with pytest.raises(subject.AuditError, match="execution budget"):
        subject.validate_predata_authority(
            root,
            auth,
            go,
            repo=repo,
            validate_bound_inputs=False,
            validate_runtime=False,
        validate_historical_state=False,
        )


def test_control_chain_rejects_crosslink_identity_drift(tmp_path: Path) -> None:
    root, repo, auth, go = _build_control_fixture(tmp_path)
    payload = subject.strict_json_file(go)
    payload["authorization"]["sha256"] = "0" * 64
    _write_json(go, payload)
    with pytest.raises(subject.AuditError, match="authorization mismatch"):
        subject.validate_predata_authority(root, auth, go, repo=repo, validate_bound_inputs=False, validate_runtime=False, validate_historical_state=False)


def test_control_chain_rejects_non_strict_chronology(tmp_path: Path) -> None:
    root, repo, auth, go = _build_control_fixture(tmp_path)
    payload = subject.strict_json_file(go)
    payload["created_utc"] = "2026-08-11T16:03:00.000000Z"
    _write_json(go, payload)
    with pytest.raises(subject.AuditError, match="chronology"):
        subject.validate_predata_authority(root, auth, go, repo=repo, validate_bound_inputs=False, validate_runtime=False, validate_historical_state=False)


@pytest.mark.parametrize(
    ("control", "key_path", "replacement", "match"),
    (
        ("code_seal", ("schema_version",), 5.0, "code seal schema"),
        ("code_seal", ("amendment", "size_bytes"), 102361.0, "code seal amendment"),
        (
            "code_seal",
            ("output_namespace", "require_absent_at_go"),
            1,
            "code seal output namespace",
        ),
        (
            "code_seal",
            ("authority_scope", "execution_authorized"),
            0,
            "code seal authority scope",
        ),
        ("authorization", ("schema_version",), 5.0, "authorization schema"),
        (
            "authorization",
            ("execution_budget", "attempt_count"),
            1.0,
            "authorization execution budget",
        ),
        (
            "authorization",
            ("authority_scope", "single_attempt_authorized"),
            1,
            "authorization authority scope",
        ),
        ("review", ("schema_version",), 5.0, "review schema"),
        (
            "review",
            ("independent_checks", "chronology"),
            1,
            "review independent checks",
        ),
        ("go", ("schema_version",), 5.0, "GO schema"),
        (
            "go",
            ("single_attempt", "attempt_number"),
            1.0,
            "GO single attempt",
        ),
        (
            "go",
            ("authority_scope", "labels_read_allowed"),
            0,
            "GO authority scope",
        ),
    ),
)
def test_control_chain_rejects_python_numeric_boolean_aliases(
    tmp_path: Path,
    control: str,
    key_path: tuple[str, ...],
    replacement: object,
    match: str,
) -> None:
    root, repo, auth, go = _build_control_fixture(tmp_path)
    control_paths = {
        "code_seal": root / subject.CODE_SEAL_REL,
        "authorization": auth,
        "review": root / subject.REVIEW_REL,
        "go": go,
    }
    control_path = control_paths[control]
    payload = subject.strict_json_file(control_path)
    parent = payload
    for key in key_path[:-1]:
        parent = parent[key]
    parent[key_path[-1]] = replacement
    _write_json(control_path, payload)
    with pytest.raises(subject.AuditError, match=match):
        subject.validate_predata_authority(
            root,
            auth,
            go,
            repo=repo,
            validate_bound_inputs=False,
            validate_runtime=False,
        validate_historical_state=False,
        )


@pytest.mark.parametrize(
    ("control", "variant"),
    (
        ("code_seal", "compact"),
        ("authorization", "spacing"),
        ("review", "missing_lf"),
        ("go", "raw_non_ascii"),
    ),
)
def test_control_chain_rejects_noncanonical_physical_json(
    tmp_path: Path, control: str, variant: str
) -> None:
    root, repo, auth, go = _build_control_fixture(tmp_path)
    control_paths = {
        "code_seal": root / subject.CODE_SEAL_REL,
        "authorization": auth,
        "review": root / subject.REVIEW_REL,
        "go": go,
    }
    control_path = control_paths[control]
    payload = subject.strict_json_file(control_path)
    if variant == "compact":
        raw = (
            json.dumps(
                payload,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    elif variant == "spacing":
        raw = (
            json.dumps(
                payload,
                ensure_ascii=True,
                sort_keys=True,
                indent=4,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    elif variant == "missing_lf":
        raw = subject.pretty_json_bytes(payload, ensure_ascii=True)[:-1]
    else:
        assert variant == "raw_non_ascii"
        payload["publisher"] = "é"
        raw = (
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        assert b"\xc3\xa9" in raw
    control_path.write_bytes(raw)
    with pytest.raises(subject.AuditError, match="physical pretty JSON serialization mismatch"):
        subject.validate_predata_authority(
            root,
            auth,
            go,
            repo=repo,
            validate_bound_inputs=False,
            validate_runtime=False,
        validate_historical_state=False,
        )


@pytest.mark.parametrize(
    "completed_utc",
    (
        subject.AMENDMENT_V5_CREATED_UTC,
        "2026-08-11T18:03:24.804999Z",
    ),
    ids=("equal_to_v5", "earlier_than_v5"),
)
def test_control_chain_rejects_test_evidence_not_strictly_after_v5(
    tmp_path: Path, completed_utc: str
) -> None:
    root, repo, auth, go = _build_control_fixture(tmp_path)
    seal_path = root / subject.CODE_SEAL_REL
    payload = subject.strict_json_file(seal_path)
    payload["test_evidence"]["completed_utc"] = completed_utc
    _write_json(seal_path, payload)
    with pytest.raises(subject.AuditError, match="test evidence chronology"):
        subject.validate_predata_authority(
            root,
            auth,
            go,
            repo=repo,
            validate_bound_inputs=False,
            validate_runtime=False,
        validate_historical_state=False,
        )


def test_code_seal_allows_test_completion_at_seal_time(tmp_path: Path) -> None:
    root, repo, _, _ = _build_control_fixture(tmp_path)
    seal_path = root / subject.CODE_SEAL_REL
    payload = subject.strict_json_file(seal_path)
    payload["test_evidence"]["completed_utc"] = payload["created_utc"]
    _write_json(seal_path, payload)
    amendment, amendment_identity = subject._validate_amendment(root)
    amendment_v4, amendment_v4_identity = subject._validate_amendment_v4(
        root, amendment, amendment_identity
    )
    _, incident_identity = subject._validate_failure_incident(
        root,
        repo,
        amendment_identity,
        amendment_v4_identity,
        validate_historical_identities=False,
    )
    _, amendment_v5_identity = subject._validate_amendment_v5(
        root,
        amendment,
        amendment_identity,
        amendment_v4,
        amendment_v4_identity,
        incident_identity,
    )
    seal, _ = subject._validate_code_seal(
        root,
        repo,
        amendment,
        amendment_identity,
        amendment_v4_identity,
        amendment_v5_identity,
    )
    assert seal["test_evidence"]["completed_utc"] == seal["created_utc"]


def test_control_chain_rejects_code_identity_drift(tmp_path: Path) -> None:
    root, repo, auth, go = _build_control_fixture(tmp_path)
    (repo / subject.CODE_ROLE_PATHS["runner"]).write_bytes(b"drift")
    with pytest.raises(subject.AuditError, match="physical identity"):
        subject.validate_predata_authority(root, auth, go, repo=repo, validate_bound_inputs=False, validate_runtime=False, validate_historical_state=False)


def test_control_chain_rejects_attempt_calendar_drift(tmp_path: Path) -> None:
    root, repo, auth, go = _build_control_fixture(tmp_path)
    payload = subject.strict_json_file(auth)
    payload["attempt_id"] = "target_free_duplicate_v5__20260230T120200000000Z"
    _write_json(auth, payload)
    with pytest.raises(subject.AuditError, match="calendar"):
        subject.validate_predata_authority(root, auth, go, repo=repo, validate_bound_inputs=False, validate_runtime=False, validate_historical_state=False)


def _context_for_namespace(tmp_path: Path, amendment: dict) -> subject.PredataContext:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "raw").mkdir()
    (tmp_path / subject.OUTPUT_ROOT_REL / ".transaction_v5").mkdir(parents=True)
    manifest_bytes = (ROOT / "manifest_census_v1.json").read_bytes()
    provenance_bytes = (ROOT / "raw" / "RAW_RANGE_MANIFEST.parquet").read_bytes()
    (tmp_path / "manifest_census_v1.json").write_bytes(manifest_bytes)
    (tmp_path / "raw" / "RAW_RANGE_MANIFEST.parquet").write_bytes(provenance_bytes)
    output = tmp_path / subject.OUTPUT_ROOT_REL
    (output / "FIELD_CENSUS_LOCK.json").write_bytes(manifest_bytes)
    (output / "PROVENANCE_LEDGER.parquet").write_bytes(provenance_bytes)
    for name in subject.RUNNER_OUTPUTS:
        (output / ".transaction_v5" / name).write_bytes(b"placeholder\n")
    return subject.PredataContext(
        root=tmp_path,
        repo=REPO,
        amendment=amendment,
        amendment_identity={},
        amendment_v4={},
        amendment_v4_identity={},
        amendment_v5={},
        amendment_v5_identity={},
        failure_incident={},
        failure_incident_identity={},
        code_seal={},
        code_seal_identity={},
        authorization={},
        authorization_identity={},
        review={},
        review_identity={},
        independent_go={},
        independent_go_identity={},
        attempt_id="target_free_duplicate_v5__20260811T000000000000Z",
    )


def test_namespace_accepts_exact_alias2_and_staged7(tmp_path: Path, amendment: dict) -> None:
    context = _context_for_namespace(tmp_path, amendment)
    aliases, staged = subject._validate_namespace_structure(context)
    assert list(aliases) == ["FIELD_CENSUS_LOCK.json", "PROVENANCE_LEDGER.parquet"]
    assert list(staged) == list(subject.RUNNER_OUTPUTS)


def test_namespace_rejects_extra_entry(tmp_path: Path, amendment: dict) -> None:
    context = _context_for_namespace(tmp_path, amendment)
    (tmp_path / subject.OUTPUT_ROOT_REL / "extra.tmp").write_bytes(b"x")
    with pytest.raises(subject.AuditError, match="exact3"):
        subject._validate_namespace_structure(context)


def test_namespace_rejects_premature_final(tmp_path: Path, amendment: dict) -> None:
    context = _context_for_namespace(tmp_path, amendment)
    transaction_file = tmp_path / subject.TRANSACTION_REL / subject.RUNNER_OUTPUTS[0]
    transaction_file.unlink()
    (tmp_path / subject.OUTPUT_ROOT_REL / subject.RUNNER_OUTPUTS[0]).write_bytes(b"x")
    with pytest.raises(subject.AuditError):
        subject._validate_namespace_structure(context)


@pytest.mark.skipif(os.name != "nt", reason="Windows inode/link semantics are bound")
def test_namespace_rejects_persistent_source_hardlink(tmp_path: Path, amendment: dict) -> None:
    context = _context_for_namespace(tmp_path, amendment)
    alias = tmp_path / subject.OUTPUT_ROOT_REL / "FIELD_CENSUS_LOCK.json"
    alias.unlink()
    os.link(tmp_path / "manifest_census_v1.json", alias)
    with pytest.raises(subject.AuditError, match="inode|link count"):
        subject._validate_namespace_structure(context)


def test_csv_reader_accepts_utf8_lf_exact_schema(tmp_path: Path) -> None:
    path = tmp_path / "x.csv"
    path.write_bytes(b"a,b\n1,TRUE\n")
    rows, raw = subject._read_strict_csv(path, ["a", "b"], 1, label="x")
    assert rows == [{"a": "1", "b": "TRUE"}]
    assert raw.endswith(b"\n")


@pytest.mark.parametrize(
    "raw,reason",
    [
        (b"\xef\xbb\xbfa\n1\n", "BOM"),
        (b"a\r\n1\r\n", "CR"),
        (b"a\nnan\n", "non-finite"),
        (b"a\n1", "terminal LF"),
    ],
)
def test_csv_reader_rejects_noncanonical_bytes(tmp_path: Path, raw: bytes, reason: str) -> None:
    path = tmp_path / "x.csv"
    path.write_bytes(raw)
    with pytest.raises(subject.AuditError, match=reason):
        subject._read_strict_csv(path, ["a"], 1, label="x")


def test_output_json_requires_exact_keys_and_python_pretty_bytes(tmp_path: Path) -> None:
    path = tmp_path / "x.json"
    payload = {"b": 2, "a": 1}
    path.write_bytes(subject.pretty_json_bytes(payload, ensure_ascii=True))
    observed, raw = subject._read_strict_output_json(path, ["a", "b"], label="x")
    assert observed == payload and raw.startswith(b'{\n  "a"')
    path.write_text('{"a":1,"b":2}\n', encoding="utf-8", newline="\n")
    with pytest.raises(subject.AuditError, match="serialization"):
        subject._read_strict_output_json(path, ["a", "b"], label="x")


def test_csv_float_pair_requires_bit_exact_hex_and_positive_zero() -> None:
    assert subject._csv_float_pair({"x": "1.5", "x_float_hex": "0x1.8000000000000p+0"}, "x", label="x") == 1.5
    assert subject._csv_float_pair({"x": "0", "x_float_hex": "0x0.0p+0"}, "x", label="x") == 0.0
    with pytest.raises(subject.AuditError):
        subject._csv_float_pair({"x": "1.5", "x_float_hex": "0x1.0000000000000p+0"}, "x", label="x")
    with pytest.raises(subject.AuditError):
        subject._csv_float_pair({"x": "-0.0", "x_float_hex": "-0x0.0p+0"}, "x", label="x")


def test_v4_csv_boolean_and_active_boundary_name_grammar() -> None:
    assert subject._csv_bool("TRUE", "bool") is True
    assert subject._csv_bool("FALSE", "bool") is False
    assert subject._csv_boundary_flags("", "flags") == []
    selected = [subject.CUTOFF_NAMES[0], subject.CUTOFF_NAMES[3], subject.CUTOFF_NAMES[-1]]
    assert subject._csv_boundary_flags(";".join(selected), "flags") == selected
    for invalid in (
        "true",
        "FALSE;TRUE",
        f"{subject.CUTOFF_NAMES[1]};{subject.CUTOFF_NAMES[0]}",
        f"{subject.CUTOFF_NAMES[0]};{subject.CUTOFF_NAMES[0]}",
        "NOT_A_FROZEN_CUTOFF",
    ):
        if invalid == "true":
            with pytest.raises(subject.AuditError, match="uppercase boolean"):
                subject._csv_bool(invalid, "bool")
        else:
            with pytest.raises(subject.AuditError):
                subject._csv_boundary_flags(invalid, "flags")


def test_json_float_pair_requires_bit_exact_hex() -> None:
    record = {"x": 2.5, "x_float_hex": float(2.5).hex()}
    assert subject._json_float_pair(record, "x", label="x") == 2.5
    with pytest.raises(subject.AuditError):
        subject._json_float_pair({"x": 2.5, "x_float_hex": float(2.0).hex()}, "x", label="x")


def _oof_table(*, bad_residual: bool = False):
    import pyarrow as pa

    residual = [0.5, 0.5]
    if bad_residual:
        residual[1] = 0.25
    return pa.table(
        {
            "planned_oof_slot_id": ["a", "b"],
            "response": ["HPBL_surface", "HPBL_surface"],
            "estimator": ["RIDGE_PIPELINE", "RIDGE_PIPELINE"],
            "fold": ["2022_H1", "2022_H1"],
            "valid_time_utc": ["2022-01-01T00:00:00Z", "2022-01-01T01:00:00Z"],
            "target_operating_day_kst": ["2022-01-01", "2022-01-01"],
            "forecast_hour": pa.array([28, 29], type=pa.int16()),
            "site_id": ["1", "1"],
            "group": ["kpx_group_1", "kpx_group_1"],
            "capacity_mw": [1.0, 1.0],
            "actual": [1.0, 2.0],
            "oof_prediction": [0.5, 1.5],
            "residual": residual,
            "slot_status": ["COMPLETED", "COMPLETED"],
            "invalid_reason": [None, None],
            "independent_selected": [True, True],
        }
    )


def test_oof_validator_accepts_exact_residual_arithmetic(amendment: dict) -> None:
    import numpy as np

    subject._validate_oof_table(np, _oof_table(), amendment, production_shape=False)


def test_oof_validator_rejects_residual_mismatch(amendment: dict) -> None:
    import numpy as np

    with pytest.raises(subject.AuditError, match="residual"):
        subject._validate_oof_table(np, _oof_table(bad_residual=True), amendment, production_shape=False)


@pytest.mark.parametrize(
    "operating_day,fold",
    [
        ("2022-01-01", "2022_H1"),
        ("2022-06-30", "2022_H1"),
        ("2022-07-01", "2022_H2"),
        ("2022-12-31", "2022_H2"),
        ("2023-01-01", "2023_H1"),
        ("2023-06-30", "2023_H1"),
        ("2023-07-01", "2023_H2"),
        ("2023-12-31", "2023_H2"),
    ],
)
def test_fold_name_is_recomputed_from_operating_day(operating_day: str, fold: str) -> None:
    assert subject._fold_name_for_operating_day(operating_day) == fold


@pytest.mark.parametrize("operating_day", ["2021-12-31", "2024-01-01", "2022-02-30", "2022-1-01"])
def test_fold_name_rejects_outside_or_noncanonical_operating_day(operating_day: str) -> None:
    with pytest.raises(subject.AuditError):
        subject._fold_name_for_operating_day(operating_day)


def test_selected_oof_estimator_requires_all_four_folds_completed() -> None:
    import numpy as np

    folds = np.repeat(np.asarray(subject.FOLD_NAMES, dtype=object), 4_896)
    statuses = np.full(19_584, "COMPLETED", dtype=object)
    oof = {"fold": folds, "slot_status": statuses}
    subject._validate_selected_oof_block_completion(
        np,
        oof,
        0,
        19_584,
        response="HPBL_surface",
        estimator="RIDGE_PIPELINE",
    )
    statuses[4_896] = "SKIPPED_PREMODEL_CLEAR_VETO"
    with pytest.raises(subject.AuditError, match="wholly completed"):
        subject._validate_selected_oof_block_completion(
            np,
            oof,
            0,
            19_584,
            response="HPBL_surface",
            estimator="RIDGE_PIPELINE",
        )


def test_v5_exact4_footer_gate_is_synthetic_and_metadata_only(
    tmp_path: Path, amendment: dict, amendment_v5: dict
) -> None:
    _, pa, pq = subject._import_arrow_after_predata()
    synthetic_v5 = copy.deepcopy(amendment_v5)
    section = synthetic_v5["bound_parquet_footer_schemas_exact"]
    constructors = {
        "large_string": pa.large_string(),
        "int16": pa.int16(),
        "int64": pa.int64(),
        "float64": pa.float64(),
    }
    for role in section["gate_roles_exact_order"]:
        record = section[role]
        relative = f"synthetic/{role}.parquet"
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        schema = pa.schema(
            [
                pa.field(name, constructors[literal])
                for name, literal in zip(
                    record["column_names_exact"],
                    record["pyarrow_constructor_types_exact"],
                )
            ]
        )
        table = pa.Table.from_arrays(
            [pa.array([None], type=field.type) for field in schema], schema=schema
        )
        pq.write_table(table, path, **subject.PARQUET_WRITE_KWARGS_EXACT)
        record["source_identity"] = subject.file_identity(path, relative)
        record["row_count_exact"] = 1
        witness = {
            "columns": schema.names,
            "rows": 1,
            "types": [str(field.type) for field in schema],
        }
        raw = json.dumps(
            witness,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
        record["structural_schema_fingerprint"] = {
            "canonical_size_bytes": len(raw),
            "canonical_sha256": hashlib.sha256(raw).hexdigest(),
        }
    context = subject.PredataContext(
        root=tmp_path,
        repo=REPO,
        amendment=amendment,
        amendment_identity={},
        amendment_v4={},
        amendment_v4_identity={},
        amendment_v5=synthetic_v5,
        amendment_v5_identity={},
        failure_incident={},
        failure_incident_identity={},
        code_seal={},
        code_seal_identity={},
        authorization={},
        authorization_identity={},
        review={},
        review_identity={},
        independent_go={},
        independent_go_identity={},
        attempt_id="target_free_duplicate_v5__20260811T000000000000Z",
    )
    subject._validate_v5_bound_footer_metadata(context, pa, pq)

    drift = copy.deepcopy(synthetic_v5)
    drift["bound_parquet_footer_schemas_exact"]["decoded_site_matrix"][
        "pyarrow_constructor_types_exact"
    ][0] = "int64"
    with pytest.raises(subject.AuditError, match="structural type mismatch"):
        subject._validate_v5_bound_footer_metadata(
            subject.PredataContext(**{**context.__dict__, "amendment_v5": drift}),
            pa,
            pq,
        )


def test_arrow_schema_type_contract_and_logical_hash_are_stable() -> None:
    import pyarrow as pa

    schema = pa.schema(
        [
            pa.field("x", pa.float64(), nullable=False),
            pa.field("y", pa.int16(), nullable=False),
            pa.field("z", pa.bool_(), nullable=False),
        ]
    )
    table = pa.Table.from_arrays(
        [pa.array([1.0], type=pa.float64()), pa.array([2], type=pa.int16()), pa.array([True], type=pa.bool_())],
        schema=schema,
    )
    assert subject._arrow_type_matches(pa, schema.field("x"), "float64")
    assert subject._arrow_type_matches(pa, schema.field("y"), "int16")
    assert subject._arrow_type_matches(pa, schema.field("z"), "bool")
    first = subject._logical_parquet_sha256(pa, table)
    second = subject._logical_parquet_sha256(pa, table.combine_chunks())
    assert first == second and len(first) == 64


def _actual_v4_runtime_context(amendment: dict) -> subject.PredataContext:
    _, amendment_identity = subject._validate_amendment(ROOT)
    amendment_v4, amendment_v4_identity = subject._validate_amendment_v4(
        ROOT, amendment, amendment_identity
    )
    return subject.PredataContext(
        root=ROOT,
        repo=REPO,
        amendment=amendment,
        amendment_identity=amendment_identity,
        amendment_v4=amendment_v4,
        amendment_v4_identity=amendment_v4_identity,
        amendment_v5={},
        amendment_v5_identity={},
        failure_incident={},
        failure_incident_identity={},
        code_seal={},
        code_seal_identity={},
        authorization={},
        authorization_identity={},
        review={},
        review_identity={},
        independent_go={},
        independent_go_identity={},
        attempt_id="target_free_duplicate_v5__20260811T160200000000Z",
    )


def test_v4_pyarrow_exact12_native_ns_runtime_contract(amendment: dict) -> None:
    _, pa, pq = subject._import_arrow_after_predata()
    context = _actual_v4_runtime_context(amendment)
    subject._validate_imported_runtime(context, pa, pq)
    assert len(subject.PARQUET_WRITE_KWARGS_EXACT) == 12
    assert subject.PARQUET_WRITE_KWARGS_EXACT["coerce_timestamps"] is None

    drift = copy.deepcopy(context.amendment_v4)
    drift["corrected_parquet_serialization"]["explicit_write_table_kwargs_exact"][
        "write_page_checksum"
    ] = False
    drift_context = subject.PredataContext(
        **{**context.__dict__, "amendment_v4": drift}
    )
    with pytest.raises(subject.AuditError, match="exact12 kwargs crosslink"):
        subject._validate_imported_runtime(drift_context, pa, pq)


def test_v4_pyarrow_entire_signature_unlisted_defaults_and_variadic_boundary() -> None:
    import pyarrow.parquet as pq

    subject._validate_pyarrow_write_table_signature(pq.write_table)
    parameters = list(inspect.signature(pq.write_table).parameters.values())

    def synthetic(*args, **kwargs):
        del args, kwargs

    drifted_defaults = [
        parameter.replace(default=1)
        if parameter.name == "data_page_size"
        else parameter
        for parameter in parameters
    ]
    synthetic.__signature__ = inspect.Signature(drifted_defaults)
    with pytest.raises(subject.AuditError, match="unlisted write_table defaults"):
        subject._validate_pyarrow_write_table_signature(synthetic)

    drifted_variadic = [
        parameter.replace(kind=inspect.Parameter.KEYWORD_ONLY, default=None)
        if parameter.name == "kwargs"
        else parameter
        for parameter in parameters
    ]
    synthetic.__signature__ = inspect.Signature(drifted_variadic)
    with pytest.raises(subject.AuditError, match="variadic keyword boundary"):
        subject._validate_pyarrow_write_table_signature(synthetic)


def test_v4_affected_parquet_schema_requires_nonnullable_ns_utc_and_date32(
    amendment: dict,
) -> None:
    _, pa, _ = subject._import_arrow_after_predata()
    context = _actual_v4_runtime_context(amendment)
    basename = "TARGET_FREE_DUPLICATE_OOF_V5.parquet"
    valid = pa.schema(
        [
            pa.field("valid_time_utc", pa.timestamp("ns", tz="UTC"), nullable=False),
            pa.field("target_operating_day_kst", pa.date32(), nullable=False),
        ]
    )
    subject._validate_v4_affected_schema(pa, valid, basename, context.amendment_v4)
    invalid_schemas = (
        pa.schema(
            [
                pa.field("valid_time_utc", pa.timestamp("us", tz="UTC"), nullable=False),
                pa.field("target_operating_day_kst", pa.date32(), nullable=False),
            ]
        ),
        pa.schema(
            [
                pa.field("valid_time_utc", pa.timestamp("ns"), nullable=False),
                pa.field("target_operating_day_kst", pa.date32(), nullable=False),
            ]
        ),
        pa.schema(
            [
                pa.field("valid_time_utc", pa.timestamp("ns", tz="UTC"), nullable=True),
                pa.field("target_operating_day_kst", pa.date32(), nullable=False),
            ]
        ),
        pa.schema(
            [
                pa.field("valid_time_utc", pa.timestamp("ns", tz="UTC"), nullable=False),
                pa.field("target_operating_day_kst", pa.date64(), nullable=False),
            ]
        ),
    )
    for schema in invalid_schemas:
        with pytest.raises(subject.AuditError, match="V4 (timestamp|date32)"):
            subject._validate_v4_affected_schema(
                pa, schema, basename, context.amendment_v4
            )


def test_metric_recomputation_uses_float64_exact_formulas() -> None:
    import numpy as np

    result = subject._metric_values(np, np.array([1.0, 2.0, 4.0]), np.array([1.5, 2.5, 3.0]))
    residual = np.array([-0.5, -0.5, 1.0], dtype=np.float64)
    assert subject._float_bits_equal(result["sse"], float(np.sum(residual * residual, dtype=np.float64)))
    assert result["nrmse_denominator"] >= 1e-9


def _primary_slot(status: str = "COMPLETED") -> dict:
    skipped = status != "COMPLETED"
    return {
        "planned_unit_slot_id": "PMU__00__0__0",
        "decision_fit_ordinal": 1,
        "unit_kind": "PRIMARY_MODEL",
        "response": "HPBL_surface",
        "fold": "2022_H1",
        "estimator_unit": "RIDGE_PIPELINE",
        "training_rows": 10,
        "training_columns": 35,
        "heldout_rows": 2,
        "input_dtype": "float64",
        "response_dtype": "float64",
        "unit_status": status,
        "skip_reason": "SKIPPED_PREMODEL_CLEAR_NONNOVEL_CONSTANT_UNIQUE_COUNT_LT_3" if skipped else None,
        "pipeline_fit_calls": 0 if skipped else 1,
        "standard_scaler_fit_calls": 0 if skipped else 1,
        "ridge_fit_calls": 0 if skipped else 1,
        "extra_trees_fit_calls": 0,
        "predict_calls": 0 if skipped else 1,
        "random_state": None,
        "fit_completed": not skipped,
    }


def test_fit_ledger_validator_accepts_completed_synthetic_slot(amendment: dict) -> None:
    payload = {
        "planned_counts": {"total": 1},
        "executed_counts": {"total": 1},
        "internal_method_call_counts": {},
        "skipped_counts": {"total": 0},
        "zero_fit_counts": {"derived": 0, "group": 0, "hyperparameter_or_seed_search": 0},
        "unit_slots": [_primary_slot()],
    }
    subject._validate_fit_ledger(payload, amendment, expected_rows=1)


def test_fit_ledger_validator_accepts_skipped_synthetic_slot(amendment: dict) -> None:
    payload = {
        "planned_counts": {"total": 1},
        "executed_counts": {"total": 0},
        "internal_method_call_counts": {},
        "skipped_counts": {"total": 1},
        "zero_fit_counts": {"derived": 0, "group": 0, "hyperparameter_or_seed_search": 0},
        "unit_slots": [_primary_slot("SKIPPED_PREMODEL_CLEAR_VETO")],
    }
    subject._validate_fit_ledger(payload, amendment, expected_rows=1)
    payload["unit_slots"][0]["skip_reason"] = "STRUCTURALLY_NOT_APPLICABLE"
    with pytest.raises(subject.AuditError, match="slot reason invalid"):
        subject._validate_fit_ledger(payload, amendment, expected_rows=1)


def _all_skipped_fit_ledger(amendment: dict) -> dict:
    from scripts import run_noaa_gfs_target_free_duplicate_v5 as runner

    reason = "SKIPPED_PREMODEL_CLEAR_NONNOVEL_CONSTANT_UNIQUE_COUNT_LT_3"
    slots = list(runner.skipped_fit_ledger({response: reason for response in amendment["planned_fit_budget"]["response_order"]}))
    return {
        "schema_version": 5,
        "artifact_type": "TARGET_FREE_FIT_LEDGER_V5",
        "status": amendment["output_contract"]["decision_status_literals_exact"]["clear_no_family"],
        "planned_counts": {
            "primary_model_units": 72,
            "analytic_affine_units": 1260,
            "total_decision_units": 1332,
            "fit_ledger_rows": 1332,
        },
        "executed_counts": {
            "completed_primary_model_units": 0,
            "completed_analytic_affine_units": 0,
            "completed_total_decision_units": 0,
        },
        "internal_method_call_counts": {
            "sklearn.pipeline.Pipeline.fit": 0,
            "sklearn.preprocessing.StandardScaler.fit": 0,
            "sklearn.linear_model.Ridge.fit": 0,
            "sklearn.ensemble.ExtraTreesRegressor.fit": 0,
            "total_method_calls": 0,
        },
        "skipped_counts": {
            "skipped_primary_model_units": 72,
            "skipped_analytic_affine_units": 1260,
            "skipped_total_decision_units": 1332,
        },
        "zero_fit_counts": {"derived": 0, "group": 0, "hyperparameter_or_seed_search": 0},
        "unit_slots": slots,
    }


def test_full_skipped_fit_ledger_enforces_affine_null_and_zero_call_contract(amendment: dict) -> None:
    payload = _all_skipped_fit_ledger(amendment)
    subject._validate_fit_ledger(payload, amendment, expected_rows=1332)
    payload["unit_slots"][72]["sxx"] = 0.0
    with pytest.raises(subject.AuditError, match="skipped affine evidence"):
        subject._validate_fit_ledger(payload, amendment, expected_rows=1332)


def test_fit_ledger_oof_status_and_reason_crosslinks(amendment: dict) -> None:
    import pyarrow as pa

    payload = _all_skipped_fit_ledger(amendment)
    reason = "SKIPPED_PREMODEL_CLEAR_NONNOVEL_CONSTANT_UNIQUE_COUNT_LT_3"
    folds = [fold for fold in subject.FOLD_NAMES for _ in range(4_896)] * 18
    statuses = ["SKIPPED_PREMODEL_CLEAR_VETO"] * 352_512
    reasons = [reason] * 352_512
    table = pa.table({"fold": folds, "slot_status": statuses, "invalid_reason": reasons})
    import numpy as np

    subject._validate_fit_ledger_oof_crosslinks(np, payload, table, amendment)
    statuses[0] = "COMPLETED"
    bad = pa.table({"fold": folds, "slot_status": statuses, "invalid_reason": reasons})
    with pytest.raises(subject.AuditError, match="status mismatch"):
        subject._validate_fit_ledger_oof_crosslinks(np, payload, bad, amendment)


def test_manifest_oof_exact4_is_derived_from_ledger_rows_and_decision(
    amendment: dict,
) -> None:
    import numpy as np
    import pyarrow as pa

    decision = _decision(amendment, "positive")
    fit_ledger = {
        "planned_counts": {"primary_model_units": 72},
        "executed_counts": {"completed_primary_model_units": 72},
        "skipped_counts": {"skipped_primary_model_units": 0},
    }
    statuses = np.full(352_512, "COMPLETED", dtype=object)
    selected = np.zeros(352_512, dtype=np.bool_)
    for response_index in range(9):
        start = response_index * 2 * 19_584
        selected[start : start + 19_584] = True
    table = pa.table(
        {"slot_status": statuses, "independent_selected": selected}
    )
    assert subject._derive_manifest_oof_slot_counts(
        np, fit_ledger, table, decision
    ) == {
        "planned_oof_slots": 352_512,
        "completed_oof_slots": 352_512,
        "skipped_oof_slots": 0,
        "independent_selected_oof_slots": 176_256,
    }
    selected[0] = False
    bad = pa.table(
        {"slot_status": statuses, "independent_selected": selected}
    )
    with pytest.raises(subject.AuditError, match="selected count"):
        subject._derive_manifest_oof_slot_counts(
            np, fit_ledger, bad, decision
        )


def _csvize_record(record: dict) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in record.items():
        if value is None:
            result[key] = ""
        elif value is True:
            result[key] = "TRUE"
        elif value is False:
            result[key] = "FALSE"
        else:
            result[key] = str(value)
    return result


def test_stability_validator_recomputes_cell_verdict(amendment: dict) -> None:
    import numpy as np
    from scripts import run_noaa_gfs_target_free_duplicate_v5 as runner

    records = runner.empty_stability_records(
        "HPBL_surface",
        diagnostic_kind="RAW_RESPONSE",
        denominator=1.0,
        reason="SKIPPED_PREMODEL_CLEAR_NONNOVEL_CONSTANT_UNIQUE_COUNT_LT_3",
    )
    rows = [_csvize_record(dict(record)) for record in records]
    assert subject._validate_stability_csv(np, rows, amendment, production_shape=False) == {}
    rows[0]["invalid_reason"] = "AMBIGUOUS_INDEPENDENT_ESTIMATOR_TIE"
    with pytest.raises(subject.AuditError, match="outside frozen enum"):
        subject._validate_stability_csv(np, rows, amendment, production_shape=False)
    rows[0]["invalid_reason"] = "SKIPPED_PREMODEL_CLEAR_NONNOVEL_CONSTANT_UNIQUE_COUNT_LT_3"
    rows[0]["cell_verdict"] = "PASS_STRICT_INTERIOR"
    with pytest.raises(subject.AuditError, match="cell verdict"):
        subject._validate_stability_csv(np, rows, amendment, production_shape=False)


def test_distribution_validator_recomputes_structural_verdict(amendment: dict) -> None:
    from scripts import run_noaa_gfs_target_free_duplicate_v5 as runner

    records = [dict(record) for record in runner.empty_distribution_records(
        "HPBL_surface", reason="SKIPPED_PREMODEL_CLEAR_NONNOVEL_CONSTANT_UNIQUE_COUNT_LT_3"
    )]
    payload = {"records": records, "endpoint_pooled_metric_records": []}
    assert subject._validate_distribution(payload, amendment, expected_records=52, expected_endpoint=0) == {}
    records[0]["invalid_reason"] = "AMBIGUOUS_INDEPENDENT_ESTIMATOR_TIE"
    with pytest.raises(subject.AuditError, match="outside frozen enum"):
        subject._validate_distribution(payload, amendment, expected_records=52, expected_endpoint=0)
    records[0]["invalid_reason"] = "SKIPPED_PREMODEL_CLEAR_NONNOVEL_CONSTANT_UNIQUE_COUNT_LT_3"
    records[0]["verdict"] = "CLEAR_FAIL_UNUSABLE"
    with pytest.raises(subject.AuditError, match="verdict"):
        subject._validate_distribution(payload, amendment, expected_records=52, expected_endpoint=0)


def test_metrics_validator_rejects_invalid_reason_outside_published_enum(amendment: dict) -> None:
    import numpy as np

    columns = amendment["output_contract"]["output_record_schemas_exact"]["TARGET_FREE_DUPLICATE_METRICS_csv"]["columns_exact"]
    row = {column: "" for column in columns}
    row.update(
        {
            "record_kind": "POOLED_ESTIMATOR",
            "response": "HPBL_surface",
            "estimator": "RIDGE_PIPELINE",
            "rows": "1",
            "valid": "FALSE",
            "invalid_reason": "AMBIGUOUS_INDEPENDENT_ESTIMATOR_TIE",
            "independent_selected": "FALSE",
            "parent_max_r2_witness": "FALSE",
            "affine_qualifies": "FALSE",
        }
    )
    with pytest.raises(subject.AuditError, match="outside frozen enum"):
        subject._validate_metrics_csv(np, [row], None, amendment, production_shape=False)


def _minimal_invalid_metrics_row(amendment: dict) -> dict[str, str]:
    columns = amendment["output_contract"]["output_record_schemas_exact"][
        "TARGET_FREE_DUPLICATE_METRICS_csv"
    ]["columns_exact"]
    row = {column: "" for column in columns}
    row.update(
        {
            "record_kind": "POOLED_ESTIMATOR",
            "response": "HPBL_surface",
            "estimator": "RIDGE_PIPELINE",
            "rows": "1",
            "valid": "FALSE",
            "invalid_reason": "SKIPPED_PREMODEL_CLEAR_NONNOVEL_CONSTANT_UNIQUE_COUNT_LT_3",
            "independent_selected": "FALSE",
            "parent_max_r2_witness": "FALSE",
            "parent_redundancy_boolean": "",
            "affine_qualifies": "FALSE",
        }
    )
    return row


def test_metrics_boolean_columns_have_exact_v4_nullability(amendment: dict) -> None:
    import numpy as np

    row = _minimal_invalid_metrics_row(amendment)
    assert subject._validate_metrics_csv(
        np, [row], None, amendment, production_shape=False
    ) == {}
    required = (
        "independent_selected",
        "parent_max_r2_witness",
        "affine_qualifies",
    )
    for key in required:
        for invalid in ("", "true", "1"):
            drift = dict(row)
            drift[key] = invalid
            with pytest.raises(subject.AuditError, match="uppercase boolean"):
                subject._validate_metrics_csv(
                    np, [drift], None, amendment, production_shape=False
                )
    for invalid in ("true", "1"):
        drift = dict(row)
        drift["parent_redundancy_boolean"] = invalid
        with pytest.raises(subject.AuditError, match="uppercase boolean"):
            subject._validate_metrics_csv(
                np, [drift], None, amendment, production_shape=False
            )


def test_evidence_headers_bind_terminal_decision_status(amendment: dict) -> None:
    status = amendment["output_contract"]["decision_status_literals_exact"]["positive"]
    ledger = {"schema_version": 5, "artifact_type": "TARGET_FREE_FIT_LEDGER_V5", "status": status}
    distribution = {"schema_version": 5, "artifact_type": "TARGET_FREE_DISTRIBUTION_STABILITY_V5", "status": status}
    decision = {"status": status}
    subject._validate_evidence_document_headers(ledger, distribution, decision)
    distribution["status"] = amendment["output_contract"]["decision_status_literals_exact"]["clear_no_family"]
    with pytest.raises(subject.AuditError, match="status"):
        subject._validate_evidence_document_headers(ledger, distribution, decision)


def _decision(amendment: dict, kind: str) -> dict:
    statuses = amendment["output_contract"]["decision_status_literals_exact"]
    status = statuses[kind]
    responses = amendment["planned_fit_budget"]["response_order"]
    endpoints = amendment["derived_wind_contract"]["endpoint_veto_diagnostics_exact"]
    keys = amendment["output_contract"]["output_record_schemas_exact"]["TARGET_FREE_FAMILY_DECISION_json"]["top_keys_exact"]
    payload = {key: [] for key in keys}

    def raw(response: str, classification: str) -> dict:
        no_selector = classification in subject.PREMODEL_CLASSIFICATIONS or classification == "AMBIGUOUS_INDEPENDENT_ESTIMATOR_TIE"
        ambiguity = classification in subject.AMBIGUOUS_CLASSIFICATIONS
        return {
            "response": response,
            "classification": classification,
            "counts_as_nonredundant": classification == "PASS_NONREDUNDANT_MARGIN",
            "independent_estimator": None if no_selector else "RIDGE_PIPELINE",
            "parent_reference_estimators": [] if no_selector else ["RIDGE_PIPELINE"],
            "parent_redundancy_boolean": None if classification in subject.PREMODEL_CLASSIFICATIONS or ambiguity else False,
            "affine_qualifying_predictors": [],
            "cell_stability_pass": None if classification in subject.PREMODEL_CLASSIFICATIONS or ambiguity else True,
            "distribution_stability_pass": None if classification in subject.PREMODEL_CLASSIFICATIONS or ambiguity else True,
            "boundary_equality_flags": [],
            "ambiguity_reason": classification if ambiguity else None,
        }

    raw_results = []
    for response in responses:
        if kind == "clear_no_family":
            classification = "CLEAR_NONNOVEL_CONSTANT"
        elif kind == "global_ambiguity" and response == "HPBL_surface":
            classification = "AMBIGUOUS_INDEPENDENT_ESTIMATOR_TIE"
        else:
            classification = "PASS_NONREDUNDANT_MARGIN"
        raw_results.append(raw(response, classification))
    raw_by_response = {record["response"]: record for record in raw_results}

    endpoint_sources = {
        "ENDPOINT_DU_925_MINUS_1000": ["UGRD_925mb", "UGRD_1000mb"],
        "ENDPOINT_DV_925_MINUS_1000": ["VGRD_925mb", "VGRD_1000mb"],
        "ENDPOINT_VECTOR_SHEAR_MAG": ["UGRD_925mb", "VGRD_925mb", "UGRD_1000mb", "VGRD_1000mb"],
    }
    endpoint_results = []
    for diagnostic in endpoints:
        sources = endpoint_sources[diagnostic]
        binding = {source: raw_by_response[source]["independent_estimator"] for source in sources}
        source_null = any(value is None for value in binding.values())
        verdict = "CLEAR_ENDPOINT_VETO_FAILURE" if source_null else "PASS_ENDPOINT_NONREDUNDANT_MARGIN"
        endpoint_results.append(
            {
                "diagnostic": diagnostic,
                "verdict": verdict,
                "source_component_estimator_binding": binding,
                "pooled_stable_margin_pass": None if source_null else True,
                "cell_stability_pass": None if source_null else True,
                "distribution_stability_pass": None if source_null else True,
                "boundary_equality_flags": [],
                "clear_pass": verdict == "PASS_ENDPOINT_NONREDUNDANT_MARGIN",
                "ambiguity_reason": None,
            }
        )

    ambiguity = kind == "global_ambiguity"
    wind_clear = kind != "clear_no_family"
    pbl_clear = kind == "positive"
    selected = "LOW_LEVEL_ISOBARIC_WIND_PROFILE" if kind == "positive" else None
    tied = []
    reasons = []
    if ambiguity:
        tied = [
            {
                "tie_kind": "INDEPENDENT_RMSE",
                "diagnostic": "HPBL_surface",
                "estimators": ["RIDGE_PIPELINE", "EXTRA_TREES"],
                "metric_name": "RMSE",
                "metric_values_float_hex": [float(1.0).hex(), float(1.0).hex()],
                "resolution": "GLOBAL_AMBIGUITY",
                "global_ambiguity": True,
            }
        ]
        reasons = ["AMBIGUOUS_INDEPENDENT_ESTIMATOR_TIE:HPBL_surface"]
    payload.update(
        {
            "schema_version": 5,
            "artifact_type": "TARGET_FREE_FAMILY_DECISION_V5",
            "status": status,
            "raw_component_order": list(responses),
            "raw_component_results": raw_results,
            "endpoint_veto_results": endpoint_results,
            "global_ambiguity_reasons": reasons,
            "family_clear_pass_results": {
                "LOW_LEVEL_ISOBARIC_WIND_PROFILE": wind_clear,
                "PBL_HEIGHT": pbl_clear,
            },
            "fixed_priority": ["LOW_LEVEL_ISOBARIC_WIND_PROFILE", "PBL_HEIGHT"],
            "selected_family": selected,
            "selected_raw_columns": list(responses[1:]) if selected else [],
            "selected_derived_columns": amendment["derived_wind_contract"]["derived_order_exact"] if selected else [],
            "all_threshold_float_hex_witnesses": [],
            "all_tied_witnesses": tied,
            "global_ambiguity": ambiguity,
            "no_label_network_future_year_model_submission_attestation": {
                "labels_read": False,
                "arrays_2024_2025_read": False,
                "network_requests": 0,
                "generation_model_fit_units": 0,
                "submission_csv_files_written": 0,
            },
        }
    )
    return payload


def _fake_identity(path: str) -> dict:
    return {
        "path": path,
        "size_bytes": 1,
        "sha256": hashlib.sha256(path.encode("utf-8")).hexdigest(),
    }


def _document_identities(amendment: dict) -> tuple[dict, dict, dict]:
    amendment_payload, amendment_identity = subject._validate_amendment(ROOT)
    amendment_v4, amendment_v4_identity = subject._validate_amendment_v4(
        ROOT, amendment_payload, amendment_identity
    )
    _, incident_identity = subject._validate_failure_incident(
        ROOT, REPO, amendment_identity, amendment_v4_identity
    )
    _, amendment_v5_identity = subject._validate_amendment_v5(
        ROOT,
        amendment_payload,
        amendment_identity,
        amendment_v4,
        amendment_v4_identity,
        incident_identity,
    )
    return amendment_identity, amendment_v4_identity, amendment_v5_identity


def _fake_code_identities() -> dict:
    return {
        role: _fake_identity(path)
        for role, path in subject.CODE_ROLE_PATHS.items()
    }


def _lock_payload(amendment: dict, kind: str) -> tuple[dict, dict, dict, dict, dict]:
    decision = _decision(amendment, kind)
    amendment_identity, amendment_v4_identity, amendment_v5_identity = (
        _document_identities(amendment)
    )
    code_identities = _fake_code_identities()
    postrun_identity = _fake_identity(subject.POSTRUN_REL)
    site_group_scope, spatial_transforms = subject._lock_static_sections(ROOT, amendment)
    positive = kind == "positive"
    raw_columns = decision["selected_raw_columns"] if positive else None
    derived_columns = decision["selected_derived_columns"] if positive else None
    payload = {
        "schema_version": 5,
        "artifact_type": subject.LOCK_ARTIFACT_TYPE,
        "status": subject.LOCK_STATUS_BY_DECISION_STATUS[decision["status"]],
        "selected_family": decision["selected_family"] if positive else None,
        "ordered_raw_columns": raw_columns,
        "ordered_derived_columns": derived_columns,
        "units": subject._selected_units(raw_columns, derived_columns) if positive else None,
        "grib_selectors": (
            {
                name: subject.GRIB_SELECTOR_BY_RAW_COLUMN[name]
                for name in raw_columns
            }
            if positive
            else None
        ),
        "site_group_scope": site_group_scope,
        "spatial_transforms": spatial_transforms,
        "parent_and_independent_gate_results": decision,
        "bound_input_identities": amendment["bound_inputs"],
        "code_config_runtime_identities": {
            "amendment": amendment_identity,
            "amendment_v4": amendment_v4_identity,
            "amendment_v5": amendment_v5_identity,
            "code_identities": code_identities,
            "runtime_identity": amendment["runtime_identity"],
        },
        "postrun_pass_identity": postrun_identity,
        "no_label_attestation": decision[
            "no_label_network_future_year_model_submission_attestation"
        ],
        "downstream_authority": subject.LOCK_DOWNSTREAM_AUTHORITY,
    }
    return payload, decision, code_identities, postrun_identity, amendment_v5_identity


@pytest.mark.parametrize("kind", ["positive", "clear_no_family", "global_ambiguity"])
def test_future_family_lock_validator_accepts_all_terminal_states(
    amendment: dict, kind: str
) -> None:
    payload, decision, code_identities, postrun_identity, amendment_v5_identity = _lock_payload(
        amendment, kind
    )
    _, amendment_identity = subject._validate_amendment(ROOT)
    _, amendment_v4_identity = subject._validate_amendment_v4(
        ROOT, amendment, amendment_identity
    )
    result = subject.validate_family_lock_payload(
        payload,
        root=ROOT,
        amendment=amendment,
        amendment_identity=amendment_identity,
        amendment_v4_identity=amendment_v4_identity,
        amendment_v5_identity=amendment_v5_identity,
        decision_payload=decision,
        code_identities=code_identities,
        postrun_pass_identity=postrun_identity,
    )
    assert result["status"] == payload["status"]
    if kind == "positive":
        assert len(result["units"]) == 26
        assert len(result["grib_selectors"]) == 8
    else:
        assert all(
            result[key] is None
            for key in (
                "selected_family",
                "ordered_raw_columns",
                "ordered_derived_columns",
                "units",
                "grib_selectors",
            )
        )


def test_future_family_lock_sources_are_exact_and_direct(amendment: dict) -> None:
    site_group, spatial = subject._lock_static_sections(ROOT, amendment)
    assert site_group == {
        key: amendment["input_and_join_contract"][key]
        for key in subject.SITE_GROUP_SCOPE_KEYS
    }
    parent = subject._load_bound_parent_plan(ROOT, amendment)
    assert spatial == {
        "spatial_method": parent["decode_contract"]["spatial_method"],
        "group_construction": amendment["derived_wind_contract"][
            "group_construction"
        ],
    }
    assert subject._selected_units(["HPBL_surface"], []) == {
        "HPBL_surface": "m"
    }
    assert subject._selected_units(
        ["UGRD_925mb"],
        ["WS_925", "ENDPOINT_DIRECTION_COS"],
    ) == {
        "UGRD_925mb": "m/s",
        "WS_925": "m/s",
        "ENDPOINT_DIRECTION_COS": "dimensionless",
    }


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("units", {"UGRD_925mb": "knots"}),
        ("grib_selectors", {"UGRD_925mb": {"variable": "UGRD", "level": "850 mb"}}),
        ("parent_and_independent_gate_results", {}),
        ("no_label_attestation", {"labels_read": False}),
    ],
)
def test_future_family_lock_validator_rejects_nested_drift(
    amendment: dict, field: str, replacement: object
) -> None:
    payload, decision, code_identities, postrun_identity, amendment_v5_identity = _lock_payload(
        amendment, "positive"
    )
    payload[field] = replacement
    _, amendment_identity = subject._validate_amendment(ROOT)
    _, amendment_v4_identity = subject._validate_amendment_v4(
        ROOT, amendment, amendment_identity
    )
    with pytest.raises(subject.AuditError):
        subject.validate_family_lock_payload(
            payload,
            root=ROOT,
            amendment=amendment,
            amendment_identity=amendment_identity,
            amendment_v4_identity=amendment_v4_identity,
            amendment_v5_identity=amendment_v5_identity,
            decision_payload=decision,
            code_identities=code_identities,
            postrun_pass_identity=postrun_identity,
        )


def test_future_negative_family_lock_rejects_nonnull_locked_fields(
    amendment: dict,
) -> None:
    payload, decision, code_identities, postrun_identity, amendment_v5_identity = _lock_payload(
        amendment, "clear_no_family"
    )
    payload["ordered_raw_columns"] = []
    _, amendment_identity = subject._validate_amendment(ROOT)
    _, amendment_v4_identity = subject._validate_amendment_v4(
        ROOT, amendment, amendment_identity
    )
    with pytest.raises(subject.AuditError, match="null5"):
        subject.validate_family_lock_payload(
            payload,
            root=ROOT,
            amendment=amendment,
            amendment_identity=amendment_identity,
            amendment_v4_identity=amendment_v4_identity,
            amendment_v5_identity=amendment_v5_identity,
            decision_payload=decision,
            code_identities=code_identities,
            postrun_pass_identity=postrun_identity,
        )


def _fake_output_identity(basename: str) -> dict:
    digest = hashlib.sha256(basename.encode("utf-8")).hexdigest()
    logical = (
        digest
        if subject.FINAL_OUTPUT_FORMATS[basename] in {"CSV", "JSON"}
        else hashlib.sha256((basename + ":logical").encode("utf-8")).hexdigest()
    )
    return {
        "path": f"{subject.OUTPUT_ROOT_REL}/{basename}",
        "size_bytes": 1,
        "sha256": digest,
        "format": subject.FINAL_OUTPUT_FORMATS[basename],
        "row_count": subject.FINAL_OUTPUT_ROWS[basename],
        "logical_sha256": logical,
    }


def _manifest_payload(amendment: dict) -> tuple[dict, dict]:
    decision = _decision(amendment, "clear_no_family")
    fit_ledger = _all_skipped_fit_ledger(amendment)
    amendment_identity, amendment_v4_identity, amendment_v5_identity = (
        _document_identities(amendment)
    )
    code_identities = _fake_code_identities()
    control_identities = {
        role: _fake_identity(
            {
                "code_seal": subject.CODE_SEAL_REL,
                "authorization": subject.AUTH_REL,
                "independent_review": subject.REVIEW_REL,
                "independent_go": subject.GO_REL,
                "postrun_pass": subject.POSTRUN_REL,
            }[role]
        )
        for role in subject.CONTROL_IDENTITY_KEYS
    }
    aliases = {
        name: _fake_identity(f"{subject.OUTPUT_ROOT_REL}/{name}")
        for name in subject.ALIAS_SPECS
    }
    outputs = {
        name: _fake_output_identity(name) for name in subject.FINAL_OUTPUT_ORDER
    }
    oof_counts = {
        "planned_oof_slots": 352_512,
        "completed_oof_slots": 0,
        "skipped_oof_slots": 352_512,
        "independent_selected_oof_slots": 0,
    }
    threadpool = {
        "capture_phases": [
            "BEFORE_EXPLICIT_CONTEXT",
            "INSIDE_CONTEXT",
            "AFTER_CONTEXT",
        ],
        "capture_counts": {
            "BEFORE_EXPLICIT_CONTEXT": 1,
            "INSIDE_CONTEXT": 0,
            "AFTER_CONTEXT": 0,
        },
        "event_count": 1,
        "event_chain_sha256": hashlib.sha256(b"threadpool").hexdigest(),
        "first_observation_by_phase": {
            "BEFORE_EXPLICIT_CONTEXT": [],
            "INSIDE_CONTEXT": None,
            "AFTER_CONTEXT": None,
        },
        "last_observation_by_phase": {
            "BEFORE_EXPLICIT_CONTEXT": [],
            "INSIDE_CONTEXT": None,
            "AFTER_CONTEXT": None,
        },
    }
    payload = {
        "schema_version": 5,
        "artifact_type": subject.MANIFEST_ARTIFACT_TYPE,
        "status": subject.MANIFEST_STATUS,
        "terminal_state": decision["status"],
        "created_utc": "2026-08-11T18:20:00.000000Z",
        "code_config_runtime_identities": {
            "amendment": amendment_identity,
            "amendment_v4": amendment_v4_identity,
            "amendment_v5": amendment_v5_identity,
            "code_identities": code_identities,
            "runtime_identity": amendment["runtime_identity"],
        },
        "control_identities": control_identities,
        "alias_identities": aliases,
        "output_identities": outputs,
        "planned_completed_skipped_fit_counts": {
            key: fit_ledger[key] for key in subject.FIT_COUNT_TREE_KEYS
        },
        "oof_slot_counts": oof_counts,
        "threadpool_info_before_inside_after": threadpool,
        "no_forbidden_access_attestation": dict(
            subject.NO_FORBIDDEN_ACCESS_ATTESTATION
        ),
        "manifest_is_last_commit_marker": True,
    }
    dependencies = {
        "amendment_identity": copy.deepcopy(amendment_identity),
        "amendment_v4_identity": copy.deepcopy(amendment_v4_identity),
        "amendment_v5_identity": copy.deepcopy(amendment_v5_identity),
        "decision_payload": copy.deepcopy(decision),
        "code_identities": copy.deepcopy(code_identities),
        "control_identities": copy.deepcopy(control_identities),
        "alias_identities": copy.deepcopy(aliases),
        "output_identities": copy.deepcopy(outputs),
        "fit_ledger": fit_ledger,
        "expected_oof_slot_counts": copy.deepcopy(oof_counts),
        "threadpool_evidence": copy.deepcopy(threadpool),
        "postrun_created_utc": "2026-08-11T18:19:00.000000Z",
    }
    return payload, dependencies


def test_future_run_manifest_validator_accepts_non_circular_exact8(
    amendment: dict,
) -> None:
    payload, dependencies = _manifest_payload(amendment)
    result = subject.validate_run_manifest_payload(
        payload, amendment=amendment, **dependencies
    )
    assert result["manifest_is_last_commit_marker"] is True
    assert list(result["output_identities"]) == list(subject.FINAL_OUTPUT_ORDER)
    assert "TRACK_A_TARGET_FREE_RUN_MANIFEST.json" not in result["output_identities"]


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda payload: payload["oof_slot_counts"].__setitem__(
                "completed_oof_slots", 1
            ),
            "OOF slot counts",
        ),
        (
            lambda payload: payload["planned_completed_skipped_fit_counts"].__setitem__(
                "zero_fit_counts", {"derived": 1}
            ),
            "fit-ledger count tree",
        ),
        (
            lambda payload: payload.__setitem__(
                "terminal_state", "PASS_ONE_TARGET_FREE_FAMILY_SELECTED_PENDING_INDEPENDENT_POSTRUN_SEAL"
            ),
            "terminal_state",
        ),
        (
            lambda payload: payload.__setitem__(
                "manifest_is_last_commit_marker", False
            ),
            "last-commit",
        ),
    ],
)
def test_future_run_manifest_validator_rejects_crosslink_drift(
    amendment: dict, mutator, match: str
) -> None:
    payload, dependencies = _manifest_payload(amendment)
    mutator(payload)
    with pytest.raises(subject.AuditError, match=match):
        subject.validate_run_manifest_payload(
            payload, amendment=amendment, **dependencies
        )


def test_future_run_manifest_rejects_self_identity_and_stale_chronology(
    amendment: dict,
) -> None:
    payload, dependencies = _manifest_payload(amendment)
    payload["output_identities"]["TRACK_A_TARGET_FREE_RUN_MANIFEST.json"] = (
        _fake_output_identity("TARGET_FREE_FAMILY_LOCK.json")
    )
    with pytest.raises(subject.AuditError, match="output identities"):
        subject.validate_run_manifest_payload(
            payload, amendment=amendment, **dependencies
        )
    payload, dependencies = _manifest_payload(amendment)
    payload["created_utc"] = dependencies["postrun_created_utc"]
    with pytest.raises(subject.AuditError, match="chronology"):
        subject.validate_run_manifest_payload(
            payload, amendment=amendment, **dependencies
        )
    payload, dependencies = _manifest_payload(amendment)
    dependencies["postrun_created_utc"] = subject.AMENDMENT_V4_CREATED_UTC
    with pytest.raises(subject.AuditError, match="chronology"):
        subject.validate_run_manifest_payload(
            payload, amendment=amendment, **dependencies
        )


@pytest.mark.parametrize("kind", ["positive", "clear_no_family", "global_ambiguity"])
def test_decision_validator_accepts_all_frozen_terminal_states(amendment: dict, kind: str) -> None:
    result = subject._validate_decision(_decision(amendment, kind), amendment)
    assert result["status"] == amendment["output_contract"]["decision_status_literals_exact"][kind]


def test_decision_validator_rejects_selected_family_on_negative(amendment: dict) -> None:
    payload = _decision(amendment, "clear_no_family")
    payload["selected_family"] = "PBL_HEIGHT"
    with pytest.raises(subject.AuditError):
        subject._validate_decision(payload, amendment)


def _threshold_ambiguous_decision(amendment: dict) -> dict:
    payload = _decision(amendment, "global_ambiguity")
    raw = payload["raw_component_results"][0]
    raw.update(
        {
            "classification": "AMBIGUOUS_THRESHOLD_EQUALITY",
            "independent_estimator": "RIDGE_PIPELINE",
            "parent_reference_estimators": ["RIDGE_PIPELINE"],
            "boundary_equality_flags": ["INDEPENDENT_R2_EXACT_DUPLICATE_0.995"],
            "ambiguity_reason": "AMBIGUOUS_THRESHOLD_EQUALITY",
        }
    )
    payload["all_tied_witnesses"] = []
    cutoff = 0.995
    witness = {
        "cutoff_name": "INDEPENDENT_R2_EXACT_DUPLICATE_0.995",
        "scope": "RAW/HPBL_surface/INDEPENDENT/RIDGE_PIPELINE",
        "diagnostic": "HPBL_surface",
        "value": cutoff,
        "value_float_hex": cutoff.hex(),
        "cutoff_float_hex": cutoff.hex(),
        "boundary_equal": True,
    }
    payload["all_threshold_float_hex_witnesses"] = [witness]
    payload["global_ambiguity_reasons"] = [
        "AMBIGUOUS_THRESHOLD_EQUALITY:"
        f"{witness['cutoff_name']}:{witness['scope']}:{witness['diagnostic']}:{witness['value_float_hex']}"
    ]
    return payload


def test_decision_validator_accepts_threshold_equality_with_exact_witness(amendment: dict) -> None:
    payload = _threshold_ambiguous_decision(amendment)
    result = subject._validate_decision(payload, amendment)
    assert result["global_ambiguity"] is True


@pytest.mark.parametrize("field", ["value_float_hex", "cutoff_float_hex", "boundary_equal"])
def test_decision_validator_rejects_threshold_witness_drift(amendment: dict, field: str) -> None:
    payload = _threshold_ambiguous_decision(amendment)
    witness = payload["all_threshold_float_hex_witnesses"][0]
    witness[field] = False if field == "boundary_equal" else float(0.5).hex()
    with pytest.raises(subject.AuditError):
        subject._validate_decision(payload, amendment)


def test_decision_validator_rejects_endpoint_raw_selector_mismatch(amendment: dict) -> None:
    payload = _decision(amendment, "positive")
    payload["endpoint_veto_results"][0]["source_component_estimator_binding"]["UGRD_925mb"] = "EXTRA_TREES"
    with pytest.raises(subject.AuditError, match="binding"):
        subject._validate_decision(payload, amendment)


def test_all_derived_bindings_are_exactly_bound_to_raw_selectors(amendment: dict) -> None:
    decision = _decision(amendment, "positive")
    bindings = {
        diagnostic: {
            response: "RIDGE_PIPELINE"
            for response in subject._derived_source_responses(diagnostic)
        }
        for diagnostic in amendment["derived_wind_contract"]["derived_order_exact"]
    }
    subject._validate_derived_binding_crosslinks(bindings, decision, amendment)
    bindings["WS_925"]["UGRD_925mb"] = "EXTRA_TREES"
    with pytest.raises(subject.AuditError, match="source binding"):
        subject._validate_derived_binding_crosslinks(bindings, decision, amendment)


def test_invalid_derived_slot_preserves_finite_actual_but_nulls_prediction(amendment: dict) -> None:
    import numpy as np
    import pyarrow as pa

    binding = json.dumps({"UGRD_925mb": None}, sort_keys=True, separators=(",", ":"))
    table = pa.table(
        {
            "capacity_mw": [1.0],
            "valid": [False],
            "actual": [2.0],
            "oof_prediction": pa.array([None], type=pa.float64()),
            "residual": pa.array([None], type=pa.float64()),
            "invalid_reason": ["STRUCTURALLY_NOT_APPLICABLE"],
            "source_component_estimator_binding_json": [binding],
        }
    )
    assert subject._validate_derived_table(np, table, amendment, production_shape=False) == {}
    table = table.set_column(
        table.schema.get_field_index("oof_prediction"),
        "oof_prediction",
        pa.array([0.0], type=pa.float64()),
    )
    with pytest.raises(subject.AuditError, match="prediction must be null"):
        subject._validate_derived_table(np, table, amendment, production_shape=False)


def test_decision_validator_rejects_family_clear_and_fixed_priority_drift(amendment: dict) -> None:
    payload = _decision(amendment, "positive")
    payload["family_clear_pass_results"]["LOW_LEVEL_ISOBARIC_WIND_PROFILE"] = False
    with pytest.raises(subject.AuditError, match="family clear"):
        subject._validate_decision(payload, amendment)
    payload = _decision(amendment, "positive")
    payload["selected_family"] = "PBL_HEIGHT"
    payload["selected_raw_columns"] = ["HPBL_surface"]
    payload["selected_derived_columns"] = []
    with pytest.raises(subject.AuditError, match="fixed-priority"):
        subject._validate_decision(payload, amendment)


def test_decision_validator_rejects_global_reason_order_or_extra_reason(amendment: dict) -> None:
    payload = _decision(amendment, "global_ambiguity")
    payload["global_ambiguity_reasons"] = ["AMBIGUOUS_RAW_ESTIMATOR_TIE"]
    with pytest.raises(subject.AuditError, match="reason"):
        subject._validate_decision(payload, amendment)


def test_global_reason_derivation_is_detailed_and_source_ordered() -> None:
    raw = [
        {"response": "HPBL_surface", "classification": "AMBIGUOUS_INDEPENDENT_ESTIMATOR_TIE"},
        {"response": "UGRD_925mb", "classification": "AMBIGUOUS_PARENT_ESTIMATOR_TIE"},
        {"response": "VGRD_925mb", "classification": "AMBIGUOUS_GRAY_ZONE"},
    ]
    thresholds = [
        {
            "cutoff_name": "CELL_R2_0.995",
            "scope": "CELL/YEAR/2022",
            "diagnostic": "UGRD_925mb",
            "value_float_hex": float(0.995).hex(),
            "boundary_equal": True,
        }
    ]
    assert subject._expected_global_ambiguity_reasons(raw, thresholds) == [
        "AMBIGUOUS_INDEPENDENT_ESTIMATOR_TIE:HPBL_surface",
        f"AMBIGUOUS_THRESHOLD_EQUALITY:CELL_R2_0.995:CELL/YEAR/2022:UGRD_925mb:{float(0.995).hex()}",
        "AMBIGUOUS_PARENT_ESTIMATOR_TIE:UGRD_925mb",
        "AMBIGUOUS_GRAY_ZONE:VGRD_925mb",
    ]


def test_wind_coverage_requires_two_distinct_pressure_levels() -> None:
    assert not subject._wind_passes_cover_required_axes_and_levels(
        ["UGRD_925mb", "VGRD_925mb", "UGRD_925mb", "VGRD_925mb"]
    )
    assert subject._wind_passes_cover_required_axes_and_levels(
        ["UGRD_925mb", "VGRD_925mb", "UGRD_950mb", "VGRD_950mb"]
    )


def _masked_affine_parent_tie_decision(amendment: dict) -> dict:
    payload = _decision(amendment, "positive")
    raw = payload["raw_component_results"][0]
    raw.update(
        {
            "classification": "CLEAR_REDUNDANT_AFFINE",
            "counts_as_nonredundant": False,
            "parent_reference_estimators": ["RIDGE_PIPELINE", "EXTRA_TREES"],
            "parent_redundancy_boolean": None,
            "affine_qualifying_predictors": [amendment["input_and_join_contract"]["predictor_columns_exact"][0]],
        }
    )
    payload["family_clear_pass_results"]["PBL_HEIGHT"] = False
    payload["all_tied_witnesses"] = [
        {
            "tie_kind": "PARENT_MAX_R2",
            "diagnostic": "HPBL_surface",
            "estimators": ["RIDGE_PIPELINE", "EXTRA_TREES"],
            "metric_name": "R2",
            "metric_values_float_hex": [float(0.5).hex(), float(0.5).hex()],
            "resolution": "MASKED_BY_PRIOR_AFFINE_REDUNDANCY",
            "global_ambiguity": False,
        }
    ]
    return payload


def test_decision_validator_accepts_affine_masked_parent_tie(amendment: dict) -> None:
    payload = _masked_affine_parent_tie_decision(amendment)
    assert subject._validate_decision(payload, amendment)["global_ambiguity"] is False


def test_decision_validator_rejects_unmarked_affine_masked_parent_tie(amendment: dict) -> None:
    payload = _masked_affine_parent_tie_decision(amendment)
    payload["all_tied_witnesses"][0]["resolution"] = "SHARED_PARENT_REDUNDANCY_FALSE"
    with pytest.raises(subject.AuditError, match="masked"):
        subject._validate_decision(payload, amendment)


def test_staged_identity_record_has_exact6_fields() -> None:
    record = subject._staged_identity_record(
        {"path": "x.csv", "size_bytes": 1, "sha256": "0" * 64},
        "x.csv",
        1,
        "1" * 64,
    )
    assert set(record) == {"path", "size_bytes", "sha256", "format", "row_count", "logical_sha256"}
    assert record["format"] == "CSV"


def test_poststate_and_stdout_topology_are_frozen() -> None:
    assert subject.POSTSTATE == {
        "output_root_entries": 3,
        "transaction_entries": 7,
        "final_publication_files": 0,
        "manifest_present": False,
        "family_lock_present": False,
        "temporary_files": 0,
        "active_locks": 0,
    }
    source = ast.parse((REPO / subject.CODE_ROLE_PATHS["auditor"]).read_text(encoding="utf-8"))
    returns = [node for node in ast.walk(source) if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)]
    assert returns


def test_auditor_stdout_serializer_is_exact_pretty_ascii_lf() -> None:
    payload = {"z": "한글", "a": 1e-9}
    encoded = subject.pretty_json_bytes(payload, ensure_ascii=True)
    expected = (
        json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    assert encoded == expected
    assert encoded.endswith(b"\n")
    assert b"\r" not in encoded and not encoded.startswith(b"\xef\xbb\xbf") and b"\x00" not in encoded
    assert subject.strict_json_bytes(encoded, label="auditor stdout serializer") == payload


def test_alias_schema_closure_precedes_every_staged_logical_read() -> None:
    tree = ast.parse((REPO / subject.CODE_ROLE_PATHS["auditor"]).read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "validate_staged_outputs"
    )
    calls = [
        (node.lineno, node.func.id)
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    first = {name: min(line for line, observed in calls if observed == name) for name in {
        "_validate_namespace_structure",
        "_import_arrow_after_predata",
        "_validate_v5_bound_footer_metadata",
        "_validate_source_parquet_metadata",
        "_read_strict_csv",
        "_read_staged_parquet",
    }}
    assert first["_import_arrow_after_predata"] < first["_validate_v5_bound_footer_metadata"]
    assert first["_validate_v5_bound_footer_metadata"] < first["_validate_namespace_structure"]
    assert first["_validate_namespace_structure"] < first["_validate_source_parquet_metadata"]
    assert first["_validate_source_parquet_metadata"] < first["_read_strict_csv"]
    assert first["_validate_source_parquet_metadata"] < first["_read_staged_parquet"]


def test_artifact_path_rejects_symlink_component_before_read(tmp_path: Path) -> None:
    root = tmp_path / "root"
    target = tmp_path / "redirect_target"
    root.mkdir()
    target.mkdir()
    redirect = root / "redirect"
    try:
        redirect.symlink_to(target, target_is_directory=True)
    except OSError as exc:  # pragma: no cover - privilege/policy dependent
        pytest.skip(f"directory symlink unavailable: {exc}")
    with pytest.raises(subject.AuditError, match="symlink component"):
        subject._artifact_path(root, "redirect/payload.json")


def test_artifact_path_component_walk_rejects_junction_signal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    transaction = root / ".transaction_v5"
    transaction.mkdir(parents=True)
    original = getattr(Path, "is_junction", None)

    def fake_is_junction(path: Path) -> bool:
        if path == transaction:
            return True
        return bool(original(path)) if callable(original) else False

    monkeypatch.setattr(Path, "is_junction", fake_is_junction, raising=False)
    with pytest.raises(subject.AuditError, match="junction component"):
        subject._artifact_path(root, ".transaction_v5/payload.json")


@pytest.mark.skipif(os.name != "nt", reason="Windows junction test")
def test_artifact_path_rejects_physical_windows_junction(tmp_path: Path) -> None:
    root = tmp_path / "root"
    target = tmp_path / "junction_target"
    root.mkdir()
    target.mkdir()
    junction = root / "redirect"
    completed = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(junction), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:  # pragma: no cover - host policy dependent
        pytest.skip(f"junction creation unavailable: {completed.stderr or completed.stdout}")
    assert junction.is_junction()
    with pytest.raises(subject.AuditError, match="junction component"):
        subject._artifact_path(root, "redirect/payload.json")


def test_auditor_source_contains_no_fit_calls_or_write_helpers() -> None:
    tree = ast.parse((REPO / subject.CODE_ROLE_PATHS["auditor"]).read_text(encoding="utf-8"))
    fit_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "fit"
    ]
    assert fit_calls == []
    forbidden_names = {"write_text", "write_bytes", "unlink", "mkdir", "rename", "link", "symlink"}
    mutation_calls = [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in forbidden_names
    ]
    assert mutation_calls == []


def test_source_parquet_metadata_validator_never_calls_read_table() -> None:
    source = ast.parse((REPO / subject.CODE_ROLE_PATHS["auditor"]).read_text(encoding="utf-8"))
    function = next(
        node
        for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name == "_validate_source_parquet_metadata"
    )
    attrs = {
        node.func.attr
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "read_table" not in attrs
    assert "read" not in attrs


def test_readonly_guard_denies_network_and_writes_in_child_process(tmp_path: Path) -> None:
    code = (
        "import socket,tempfile; "
        "from scripts.audit_noaa_gfs_target_free_duplicate_v5 import install_readonly_runtime_guard; "
        "install_readonly_runtime_guard(); "
        "ok=0; "
        "\ntry: socket.socket()\nexcept PermissionError: ok+=1\n"
        "try: open(r'" + str(tmp_path / "blocked.txt").replace("\\", "\\\\") + "','wb')\nexcept PermissionError: ok+=1\n"
        "assert ok==2"
    )
    completed = subprocess.run(
        [str(REPO / ".venv" / "Scripts" / "python.exe"), "-B", "-c", code],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert not (tmp_path / "blocked.txt").exists()


def test_main_failure_is_stdout_only_and_zero_mutation(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    command = [
        str(REPO / ".venv" / "Scripts" / "python.exe"),
        "-B",
        "-m",
        "scripts.audit_noaa_gfs_target_free_duplicate_v5",
        "--root",
        str(root),
        "--authorization",
        str(root / subject.AUTH_REL),
        "--independent-go",
        str(root / subject.GO_REL),
    ]
    before = sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))
    completed = subprocess.run(command, cwd=REPO, capture_output=True, check=False)
    after = sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))
    assert completed.returncode == 1
    payload = subject.strict_json_bytes(completed.stdout, label="failure stdout")
    assert payload["status"] == subject.FAIL_STATUS
    assert payload["files_written"] == payload["network_requests"] == payload["models_fit"] == 0
    assert completed.stderr == b""
    assert before == after == []

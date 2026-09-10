"""Seal immutable prestate and a non-executable Track-A decode-recovery auth.

This sealer is network-zero and target-free.  It validates the already cached
10,368 NOAA byte ranges, runs the real GRIB bootstrap pilot and production
decode path, then publishes exactly three no-overwrite JSON artifacts: the raw
cache lock, process preflight, and authorization pending independent GO.  It
never creates the independent review or GO and never launches the recovery.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.machinery
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import types
from typing import Any, Mapping


ROOT_DEFAULT = Path("artifacts/baram2026_ncei_scada_longrun_20260810_v2")
CANONICAL_MODULE_ENTRYPOINT = (
    "scripts.seal_noaa_gfs_multiseason_decode_recovery_v1"
)
RAW_LOCK_RELATIVE = "prelaunch/DECODE_RECOVERY_RAW_CACHE_LOCK_V1.json"
PREFLIGHT_RELATIVE = "prelaunch/DECODE_RECOVERY_REAL_PROCESS_PREFLIGHT_V1.json"
AUTHORIZATION_RELATIVE = "prereg/decode_recovery_authorization_v1.json"
REVIEW_RELATIVE = (
    "independent_redteam/TRACK_A_DECODE_RECOVERY_INDEPENDENT_REVIEW_V1.json"
)
GO_RELATIVE = "independent_redteam/TRACK_A_DECODE_RECOVERY_GO_V1.json"
RECOVERY_RUNNER_SIZE_BYTES = 128_749
RECOVERY_RUNNER_SHA256 = (
    "af61a52044d68b4a798ba24395d6c17ae56b827ab291ae67bbf46fd3e2aa1dcd"
)
SPAWN_SCHEDULING_FLAKE_INCIDENT = {
    "artifact_type": "DECODE_RECOVERY_PRESEAL_PROCESS_SCHEDULING_FLAKE_INCIDENT",
    "status": "CLOSED_BY_SYNCHRONIZED_FIRST_WORKER_WAVE_GATE",
    "observed_during_unsealed_dry_run": True,
    "observed_test": "test_real_grib_cold_start_passes_in_exactly_seven_spawn_processes",
    "observed_failure": "LESS_THAN_SEVEN_DISTINCT_WORKER_PIDS_FROM_FAST_UNSYNCHRONIZED_POOL",
    "test_process_exit_code": 1,
    "seal_artifacts_published": 0,
    "network_requests": 0,
    "root_cause": "PROCESS_POOL_TASK_SCHEDULING_DID_NOT_PROVE_ALL_REQUESTED_WORKERS",
    "corrective_action": "FIRST_TASK_IN_EACH_OF_SEVEN_WORKERS_WAITS_ON_SHARED_BARRIER_BEFORE_DECODE",
    "retry_or_selected_result_used_as_final_evidence": False,
}
NONCONTRACT_MONOLITHIC_DIAGNOSTIC = {
    "artifact_type": "DECODE_RECOVERY_NONCONTRACT_MONOLITHIC_TEST_DIAGNOSTIC",
    "status": "EXPECTED_FAIL_PREIMPORT_RESIDUE_NOT_USED_AS_SEAL_EVIDENCE",
    "topology": "ONE_INTERPRETER_MULTI_FILE_NONCONTRACT",
    "command": [
        ".venv\\Scripts\\python.exe",
        "-B",
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_noaa_gfs_decode_recovery.py",
        "tests/test_noaa_gfs_decode_recovery_bootstrap.py",
        "tests/test_noaa_gfs_multiseason_decode_recovery_runner_v1.py",
        "tests/test_seal_noaa_gfs_multiseason_decode_recovery_v1.py",
        "tests/test_noaa_gfs_multiseason_raw_v2.py",
        "tests/test_noaa_gfs_multiseason_raw_postrun_v1.py",
    ],
    "environment": {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONPYCACHEPREFIX": "ABSENT",
    },
    "exit_code": 1,
    "pytest_duration_seconds": 62.83,
    "shell_duration_seconds": 63.3,
    "passed": 146,
    "skipped": 1,
    "failed": 13,
    "failure_partition": {
        "bootstrap_tests_rejected_preloaded_src": 11,
        "canonical_spawn_child_passed_then_parent_src_residue_failed": 1,
        "static_audit_rejected_preimported_runner": 1,
    },
    "canonical_spawn_subreport": {
        "worker_count": 7,
        "tasks": 70,
        "first_wave_synchronized": True,
        "child_src_package_initializer_executed": False,
        "controls_created": 0,
        "network_requests": 0,
        "writes": 0,
    },
    "outer_network_guard_installed": False,
    "outer_network_requests": "NOT_INSTRUMENTED_NO_ZERO_CLAIM",
    "outer_data_reads": "NOT_INSTRUMENTED_NO_ZERO_CLAIM",
    "outer_writes": "NOT_INSTRUMENTED_NO_ZERO_CLAIM",
    "seal_control_publications_observed_after_run": 0,
    "used_as_contract_evidence": False,
    "required_contract_topology": "ONE_TEST_FILE_PER_CLEAN_SUBPROCESS",
}
ISOLATED_PYTEST_NETWORK_GUARD_SOURCE = """
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
FIXED_ROOT_INPUTS = {
    "incident": "incidents/TRACK_A_RAW_DECODE_MEMFS_THREAD_INIT_FAILURE_V1.json",
    "sealer_entrypoint_incident": (
        "incidents/TRACK_A_DECODE_RECOVERY_SEAL_DIRECT_FILE_PICKLE_FAILURE_V1.json"
    ),
    "original_authorization": "prereg/raw_launch_authorization_v1.json",
    "original_independent_go": "independent_redteam/TRACK_A_RAW_LAUNCH_GO.json",
    "original_failed_lock": (
        "raw/launch_history/"
        "20260810T151057590738Z__pid3536__5f3e0fb20132__failed.lock"
    ),
    "original_final_progress": (
        "raw/progress/"
        "raw_ranges__20260810T151057590738Z__pid3536__5f3e0fb20132__010368.json"
    ),
    "original_stderr": "logs/TRACK_A_RAW_RUN.stderr.log",
    "original_stdout": "logs/TRACK_A_RAW_RUN.stdout.log",
    "field_range_census": "census/field_range_census.parquet",
    "census_manifest": "manifest_census_v1.json",
    "coordinate_lock": "prereg/authoritative_turbine_coordinate_lock_v1.json",
    "raw_cache_lock": RAW_LOCK_RELATIVE,
    "real_spawn_preflight": PREFLIGHT_RELATIVE,
}
recovery: Any = None
PILOT_PATH = (
    Path("artifacts")
    / "baram2026_ncei_scada_research_20260810_210756"
    / "track_a"
    / "provenance"
    / "raw"
    / "f039"
    / "gfs.t12z.pgrb2.0p25.f039.HPBL_surface.grib2"
)

PREFLIGHT_KEYS = {
    "schema_version",
    "artifact_type",
    "status",
    "created_utc",
    "incident",
    "recovery_runner",
    "recovery_module",
    "recovery_bootstrap",
    "recovery_test",
    "recovery_bootstrap_test",
    "recovery_runner_test",
    "postrun_auditor",
    "postrun_auditor_test",
    "recovery_sealer",
    "recovery_sealer_test",
    "bootstrap_trust_policy",
    "runtime_lock",
    "bootstrap_import_contract",
    "test_evidence",
    "process_scheduling_flake_incident",
    "sealer_entrypoint_incident",
    "real_grib_pilot",
    "max_spawn_processes",
    "thread_decoders",
    "network_requests",
    "production_decode_task_worker_real_test",
    "stdlib_bootstrap_worker_real_test",
    "poisoned_pyc_source_execution_test",
    "bootstrap_ast_stdlib_policy_pass",
    "python_dont_write_bytecode_env",
    "python_dont_write_bytecode_flag",
    "python_pycache_prefix_absent",
    "bootstrap_matching_pyc_absent_before_tests",
    "bootstrap_matching_pyc_absent_after_tests",
    "all_seven_worker_pids_observed",
    "pilot_report",
    "production_worker_report",
    "focused_test_result",
    "source_compile_result",
    "labels_read",
    "arrays_2024_read",
    "arrays_2025_read",
    "models_fit",
    "submission_csv_created",
}
RAW_LOCK_KEYS = {
    "schema_version",
    "artifact_type",
    "status",
    "created_utc",
    "incident",
    "inventories",
    "prerecovery_topology",
    "documented_preplan_remnants",
    "canonical_outputs_absent",
    "network_requests_during_preflight",
    "raw_event_progress_mutations",
    "labels_read",
    "2024_arrays_read",
    "2025_arrays_read",
}
AUTHORIZATION_KEYS = {
    "schema_version",
    "artifact_type",
    "status",
    "created_utc",
    "experiment_id",
    "recovery_attempt_id",
    "incident",
    "original_runner",
    "recovery_runner",
    "recovery_module",
    "recovery_bootstrap",
    "recovery_test",
    "recovery_bootstrap_test",
    "recovery_runner_test",
    "postrun_auditor",
    "postrun_auditor_test",
    *FIXED_ROOT_INPUTS.keys(),
    "documented_preplan_remnants",
    "runtime_lock",
    "bootstrap_trust_policy",
    "bootstrap_import_contract",
    "real_grib_pilot",
    "original_runtime_identity",
    "expected_range_rows",
    "expected_range_bytes",
    "max_decode_processes",
    "network_requests_allowed",
    "raw_event_progress_mutation_allowed",
    "labels_read",
    "2024_arrays_read",
    "2025_arrays_read",
    "models_fit",
    "submission_csv_created",
}
TEST_EVIDENCE_KEYS = {
    "schema_version",
    "artifact_type",
    "status",
    "created_utc",
    "bound_code_and_test_identities",
    "source_compile_result",
    "compiled_source_identities",
    "pytest_isolation",
    "pytest_runs",
    "pytest_summary",
    "noncontract_monolithic_diagnostic",
    "isolated_subprocess_count",
    "all_pytest_exit_codes_zero",
    "pytest_child_network_guard_sha256",
    "pytest_child_network_guard_installed",
    "pytest_cacheprovider_disabled",
    "required_test_names",
    "required_test_names_present",
    "python_executable",
    "python_dont_write_bytecode_env",
    "python_dont_write_bytecode_flag",
    "python_pycache_prefix_absent",
    "bootstrap_matching_pyc_absent_before_tests",
    "bootstrap_matching_pyc_absent_after_tests",
    "network_requests",
    "labels_read",
    "arrays_2024_read",
    "arrays_2025_read",
    "models_fit",
    "submission_csv_created",
}


class RecoverySealError(RuntimeError):
    """The immutable recovery prestate was not sealable."""


def require_canonical_module_entrypoint() -> None:
    """Require ``python -B -m ...`` for every process-spawning/write mode."""

    if (
        __name__ != "__main__"
        or __package__ != "scripts"
        or __spec__ is None
        or __spec__.name != CANONICAL_MODULE_ENTRYPOINT
    ):
        raise RecoverySealError(
            "write/process mode requires canonical `python -B -m "
            f"{CANONICAL_MODULE_ENTRYPOINT}` invocation"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _SealNetworkDenied(RuntimeError):
    pass


def _network_forbidden(*_args: Any, **_kwargs: Any) -> Any:
    raise _SealNetworkDenied("network access is forbidden during recovery sealing")


_NETWORK_GUARD_INSTALLED = False


def _network_audit_guard(event: str, _args: tuple[Any, ...]) -> None:
    if event in {
        "socket.__new__",
        "socket.bind",
        "socket.connect",
        "socket.connect_ex",
        "socket.getaddrinfo",
        "socket.gethostbyaddr",
        "socket.gethostbyname",
        "socket.getnameinfo",
        "socket.sendmsg",
        "socket.sendto",
    }:
        raise _SealNetworkDenied(f"network/socket event forbidden: {event}")


def install_entry_guard() -> None:
    global _NETWORK_GUARD_INSTALLED
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1" or not sys.dont_write_bytecode:
        raise RecoverySealError("PYTHONDONTWRITEBYTECODE=1/-B is required")
    if os.environ.get("PYTHONPYCACHEPREFIX") not in (None, "") or sys.pycache_prefix is not None:
        raise RecoverySealError("external Python bytecode cache prefix is forbidden")
    if not _NETWORK_GUARD_INSTALLED:
        sys.addaudithook(_network_audit_guard)
        _NETWORK_GUARD_INSTALLED = True
    socket.create_connection = _network_forbidden  # type: ignore[assignment]


def load_recovery_runner_exact(workspace_root: Path) -> Any:
    global recovery
    path = workspace_root / "scripts" / "run_noaa_gfs_multiseason_decode_recovery_v1.py"
    identity = external_identity(path)
    if (
        identity["size_bytes"] != RECOVERY_RUNNER_SIZE_BYTES
        or identity["sha256"] != RECOVERY_RUNNER_SHA256
    ):
        raise RecoverySealError("frozen recovery runner constant mismatch")
    module_name = "scripts.run_noaa_gfs_multiseason_decode_recovery_v1"
    if module_name in sys.modules:
        raise RecoverySealError("recovery runner imported before exact-source validation")
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != RECOVERY_RUNNER_SHA256:
        raise RecoverySealError("recovery runner changed before exact compile")
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = "scripts"
    module.__loader__ = None
    module.__cached__ = None
    module.__spec__ = importlib.machinery.ModuleSpec(
        module_name, loader=None, origin=str(path)
    )
    sys.modules[module_name] = module
    try:
        exec(compile(source, str(path), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    if sha256_file(path) != RECOVERY_RUNNER_SHA256:
        raise RecoverySealError("recovery runner changed during exact execution")
    if module.NONCONTRACT_MONOLITHIC_DIAGNOSTIC != NONCONTRACT_MONOLITHIC_DIAGNOSTIC:
        raise RecoverySealError("runner/sealer monolithic diagnostic mismatch")
    recovery = module
    return module


def external_identity(path: Path) -> dict[str, Any]:
    raw = Path(path)
    if not raw.is_absolute():
        raw = Path(os.path.abspath(str(raw)))
    lexical = Path(os.path.abspath(str(raw)))
    for candidate in (lexical, *lexical.parents):
        if candidate.is_symlink() or (
            hasattr(candidate, "is_junction") and candidate.is_junction()
        ):
            raise RecoverySealError(f"external identity symlink/junction: {candidate}")
    try:
        path = lexical.resolve(strict=True)
    except FileNotFoundError as exc:
        raise RecoverySealError(f"external identity absent: {lexical}") from exc
    if path != lexical or not path.is_file():
        raise RecoverySealError(f"external identity is not an exact file: {path}")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def publish_json_no_overwrite(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f"{path.name}.tmp.{os.getpid()}.{hashlib.sha256(encoded).hexdigest()[:20]}"
    )
    if path.exists() or temporary.exists():
        raise RecoverySealError(f"seal destination already exists: {path}")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise RecoverySealError(f"seal destination race: {path}") from exc
        if not os.path.samefile(temporary, path):
            raise RecoverySealError("seal hard-link identity mismatch")
        with path.open("r+b") as stream:
            os.fsync(stream.fileno())
    finally:
        if temporary.is_file() and not temporary.is_symlink():
            temporary.unlink()


def runtime_lock(workspace_root: Path, original_runtime: Mapping[str, Any]) -> dict[str, Any]:
    package = workspace_root / ".venv" / "Lib" / "site-packages"
    return {
        "api_version": "2.47.0",
        "definitions_path": "/MEMFS/definitions",
        "library": external_identity(package / "eccodes" / "eccodes.dll"),
        "memfs_library": external_identity(package / "eccodes" / "eccodes_memfs.dll"),
        "python_binding": external_identity(
            package / "eccodes" / "_eccodes.cp313-win_amd64.pyd"
        ),
        "gribapi_binding": external_identity(package / "gribapi" / "bindings.py"),
        "eccodes_python_api": external_identity(package / "eccodes" / "eccodes.py"),
        "gribapi_python_api": external_identity(package / "gribapi" / "gribapi.py"),
        "gribapi_errors": external_identity(package / "gribapi" / "errors.py"),
        "frozen_runner": external_identity(
            workspace_root / "scripts" / "run_noaa_gfs_multiseason_raw_v2.py"
        ),
        "recovery_module": external_identity(
            workspace_root / "src" / "noaa_gfs_decode_recovery.py"
        ),
        "original_runtime_identity": dict(original_runtime),
    }


def code_identities(workspace_root: Path) -> dict[str, dict[str, Any]]:
    paths = {
        "original_runner": workspace_root
        / "scripts"
        / "run_noaa_gfs_multiseason_raw_v2.py",
        "recovery_runner": workspace_root
        / "scripts"
        / "run_noaa_gfs_multiseason_decode_recovery_v1.py",
        "recovery_module": workspace_root / "src" / "noaa_gfs_decode_recovery.py",
        "recovery_bootstrap": workspace_root
        / "src"
        / "noaa_gfs_decode_recovery_bootstrap.py",
        "recovery_test": workspace_root / "tests" / "test_noaa_gfs_decode_recovery.py",
        "recovery_bootstrap_test": workspace_root
        / "tests"
        / "test_noaa_gfs_decode_recovery_bootstrap.py",
        "recovery_runner_test": workspace_root
        / "tests"
        / "test_noaa_gfs_multiseason_decode_recovery_runner_v1.py",
        "postrun_auditor": workspace_root
        / "scripts"
        / "audit_noaa_gfs_multiseason_raw_postrun_v1.py",
        "postrun_auditor_test": workspace_root
        / "tests"
        / "test_noaa_gfs_multiseason_raw_postrun_v1.py",
    }
    return {key: external_identity(path) for key, path in paths.items()}


def sealer_identities(workspace_root: Path) -> dict[str, dict[str, Any]]:
    return {
        "recovery_sealer": external_identity(
            workspace_root
            / "scripts"
            / "seal_noaa_gfs_multiseason_decode_recovery_v1.py"
        ),
        "recovery_sealer_test": external_identity(
            workspace_root
            / "tests"
            / "test_seal_noaa_gfs_multiseason_decode_recovery_v1.py"
        ),
    }


def isolated_pytest_commands(
    executable: Path, test_relatives: tuple[str, ...]
) -> list[list[str]]:
    return [
        [
            str(executable),
            "-B",
            "-c",
            ISOLATED_PYTEST_NETWORK_GUARD_SOURCE,
            "-q",
            "-p",
            "no:cacheprovider",
            relative,
        ]
        for relative in test_relatives
    ]


def run_immutable_test_evidence(
    workspace_root: Path,
    identities: Mapping[str, Mapping[str, Any]],
    sealers: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    test_relatives = (
        "tests/test_noaa_gfs_decode_recovery.py",
        "tests/test_noaa_gfs_decode_recovery_bootstrap.py",
        "tests/test_noaa_gfs_multiseason_decode_recovery_runner_v1.py",
        "tests/test_seal_noaa_gfs_multiseason_decode_recovery_v1.py",
        "tests/test_noaa_gfs_multiseason_raw_v2.py",
        "tests/test_noaa_gfs_multiseason_raw_postrun_v1.py",
    )
    source_relatives = (
        "src/noaa_gfs_decode_recovery.py",
        "src/noaa_gfs_decode_recovery_bootstrap.py",
        "scripts/run_noaa_gfs_multiseason_decode_recovery_v1.py",
        "scripts/seal_noaa_gfs_multiseason_decode_recovery_v1.py",
        "scripts/audit_noaa_gfs_multiseason_raw_postrun_v1.py",
        *test_relatives,
    )
    compiled: list[dict[str, Any]] = []
    for relative in source_relatives:
        path = workspace_root / relative
        source = path.read_bytes()
        compile(source, str(path), "exec", dont_inherit=True)
        compiled.append(external_identity(path))

    required_names = {
        "tests/test_noaa_gfs_decode_recovery_bootstrap.py": (
            "test_bootstrap_source_is_stdlib_only_and_has_no_decoder_import",
            "test_top_level_alias_is_picklable_and_clean_child_avoids_src_initializer",
            "test_child_import_contract_tamper_fails_closed",
            "test_child_rejects_same_alias_package_or_extension_candidate",
            "test_bootstrap_cache_and_bytecode_policy_are_fail_closed",
            "test_poisoned_timestamp_pyc_is_ignored_for_verified_source",
            "test_real_pilot_uses_exactly_seven_stdlib_bootstrap_processes",
            "test_production_worker_uses_stdlib_bootstrap_and_exact_source",
        ),
        "tests/test_noaa_gfs_multiseason_decode_recovery_runner_v1.py": (
            "test_bootstrap_top_level_import_contract_is_exact_and_src_free",
            "test_bootstrap_import_root_rejects_casefolded_stdlib_shadow",
            "test_bootstrap_import_root_rejects_same_alias_competitor",
            "test_noncontract_monolithic_diagnostic_exact_semantics_pass",
            "test_noncontract_monolithic_diagnostic_tamper_fails",
            "test_locked_prestate_recloses_transaction_race_after_initial_audit",
            "test_plan_publication_race_never_overwrites",
            "test_commit_record_publication_race_never_overwrites",
            "test_original_progress_payloads_close_every_checkpoint",
            "test_independent_review_exact_schema_and_twelve_checks_pass",
            "test_independent_review_missing_extra_or_false_check_fails",
        ),
        "tests/test_seal_noaa_gfs_multiseason_decode_recovery_v1.py": (
            "test_static_audit_writes_nothing",
            "test_json_publisher_fails_on_race_without_overwrite",
            "test_failed_direct_file_seal_incident_binds_exact_alias_remediation",
            "test_noncontract_monolithic_diagnostic_is_recorded_but_never_selected",
            "test_direct_file_write_mode_fails_before_data_or_publication",
            "test_imported_programmatic_seal_fails_before_data_or_publication",
            "test_imported_programmatic_spawn_audit_fails_before_data_access",
            "test_direct_file_static_audit_remains_read_only_and_allowed",
            "test_canonical_module_entrypoint_real_seven_spawn_is_read_only",
        ),
    }
    for relative, names in required_names.items():
        source = (workspace_root / relative).read_text(encoding="utf-8")
        if any(f"def {name}(" not in source for name in names):
            raise RecoverySealError(f"required bound regression test absent: {relative}")

    executable = Path(sys.executable).resolve()
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.pop("PYTHONPYCACHEPREFIX", None)
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    environment["PYTHONIOENCODING"] = "utf-8"
    before_pyc = recovery._matching_bootstrap_bytecode(
        workspace_root / "src" / "noaa_gfs_decode_recovery_bootstrap.py"
    )
    if before_pyc:
        raise RecoverySealError("bootstrap pyc exists before immutable test run")
    pytest_runs: list[dict[str, Any]] = []
    summaries: list[str] = []
    for relative, command in zip(
        test_relatives,
        isolated_pytest_commands(executable, test_relatives),
        strict=True,
    ):
        completed = subprocess.run(
            command,
            cwd=workspace_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
        )
        stdout = completed.stdout.replace("\r\n", "\n")
        stderr = completed.stderr.replace("\r\n", "\n")
        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        summary = next((line for line in reversed(lines) if " passed" in line), "")
        if completed.returncode != 0 or not summary:
            raise RecoverySealError(
                "isolated immutable recovery test failed: "
                f"{relative}, rc={completed.returncode}, summary={summary!r}"
            )
        summaries.append(f"{relative}: {summary}")
        pytest_runs.append(
            {
                "test_file": relative,
                "command": command,
                "exit_code": completed.returncode,
                "summary": summary,
                "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
                "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
            }
        )
    after_pyc = recovery._matching_bootstrap_bytecode(
        workspace_root / "src" / "noaa_gfs_decode_recovery_bootstrap.py"
    )
    if after_pyc:
        raise RecoverySealError("bootstrap pyc appeared during immutable test suite")
    observed_roles = {**dict(identities), **dict(sealers)}
    if code_identities(workspace_root) != dict(identities) or sealer_identities(
        workspace_root
    ) != dict(sealers):
        raise RecoverySealError("code/test identity changed across immutable test run")
    evidence = {
        "schema_version": 1,
        "artifact_type": "DECODE_RECOVERY_IMMUTABLE_CODE_TEST_EVIDENCE",
        "status": "PASS_BOUND_CODE_COMPILE_AND_TEST_SUITE",
        "created_utc": None,
        "bound_code_and_test_identities": observed_roles,
        "source_compile_result": "PASS",
        "compiled_source_identities": compiled,
        "pytest_isolation": "ONE_TEST_FILE_PER_CLEAN_SUBPROCESS",
        "pytest_runs": pytest_runs,
        "pytest_summary": " | ".join(summaries),
        "noncontract_monolithic_diagnostic": NONCONTRACT_MONOLITHIC_DIAGNOSTIC,
        "isolated_subprocess_count": len(pytest_runs),
        "all_pytest_exit_codes_zero": True,
        "pytest_child_network_guard_sha256": hashlib.sha256(
            ISOLATED_PYTEST_NETWORK_GUARD_SOURCE.encode("utf-8")
        ).hexdigest(),
        "pytest_child_network_guard_installed": True,
        "pytest_cacheprovider_disabled": True,
        "required_test_names": {
            key: list(value) for key, value in required_names.items()
        },
        "required_test_names_present": True,
        "python_executable": external_identity(executable),
        "python_dont_write_bytecode_env": "1",
        "python_dont_write_bytecode_flag": True,
        "python_pycache_prefix_absent": True,
        "bootstrap_matching_pyc_absent_before_tests": not before_pyc,
        "bootstrap_matching_pyc_absent_after_tests": not after_pyc,
        "network_requests": 0,
        "labels_read": False,
        "arrays_2024_read": False,
        "arrays_2025_read": False,
        "models_fit": 0,
        "submission_csv_created": False,
    }
    if set(evidence) != TEST_EVIDENCE_KEYS:
        raise RecoverySealError("internal immutable test-evidence schema mismatch")
    return evidence


def fixed_root_identities(root: Path) -> dict[str, dict[str, Any]]:
    return {
        key: recovery.root_identity(recovery.strict_under_root(root, relative), root)
        for key, relative in recovery.FIXED_ROOT_INPUTS.items()
        if key not in {"raw_cache_lock", "real_spawn_preflight"}
    }


def build_production_pilot_report(
    bootstrap: Any,
    pilot_identity: Mapping[str, Any],
    runtime: Mapping[str, Any],
    identities: Mapping[str, Mapping[str, Any]],
    sites: list[dict[str, Any]],
) -> dict[str, Any]:
    task = {
        "result": dict(pilot_identity),
        "row": {
            "run_init_utc": "2023-07-01T12:00:00Z",
            "valid_time_utc": "2023-07-03T03:00:00Z",
            "forecast_hour": 39,
            "variable": "HPBL",
            "level": "surface",
        },
        "sites": sites,
    }
    rows, audit = bootstrap.bounded_spawn_map(
        [task for _ in range(70)],
        worker_kind="decode",
        runtime_lock=runtime,
        bootstrap_identity=identities["recovery_bootstrap"],
        runner_identity=identities["recovery_runner"],
        core_identity=identities["recovery_module"],
        max_workers=recovery.MAX_DECODE_PROCESSES,
        synchronize_first_wave=True,
    )
    if (
        audit.completed_tasks != 70
        or len(audit.observed_worker_pids) != 7
        or audit.first_wave_synchronized is not True
    ):
        raise RecoverySealError("production bootstrap pilot process coverage mismatch")
    reference = rows[0]["site_values"]
    if any(row["site_values"] != reference for row in rows):
        raise RecoverySealError("production bootstrap pilot values differ")
    packed = b"".join(struct.pack("<d", float(value)) for value in reference)
    digest = hashlib.sha256(packed).hexdigest()
    expected = "edeca661e457c32431218ccb50c5dbf0c428905a36e03c28cfe4d76d32ddcc0b"
    if digest != expected or {row["feature"] for row in rows} != {"HPBL_surface"}:
        raise RecoverySealError("production bootstrap pilot exact-value identity mismatch")
    return {
        "tasks": 70,
        "worker_pids_observed": list(audit.observed_worker_pids),
        "worker_count": len(audit.observed_worker_pids),
        "first_wave_synchronized": audit.first_wave_synchronized,
        "feature": "HPBL_surface",
        "site_count": len(reference),
        "site_values_binary64_sha256": digest,
        "all_rows_array_equal": True,
        "network_requests": 0,
    }


def static_audit(root: Path, workspace_root: Path) -> dict[str, Any]:
    install_entry_guard()
    load_recovery_runner_exact(workspace_root)
    recovery.validate_parent_bytecode_policy()
    recovery.install_parent_network_deny_guard()
    destinations = {
        "raw_cache_lock": root / RAW_LOCK_RELATIVE,
        "real_spawn_preflight": root / PREFLIGHT_RELATIVE,
        "authorization": root / AUTHORIZATION_RELATIVE,
        "independent_review": root / REVIEW_RELATIVE,
        "independent_go": root / GO_RELATIVE,
    }
    if any(path.exists() for path in destinations.values()):
        raise RecoverySealError("static seal audit requires all future seal files absent")
    identities = code_identities(workspace_root)
    sealers = sealer_identities(workspace_root)
    recovery.validate_bootstrap_preexecution(
        identities["recovery_bootstrap"],
        workspace_root / "src" / "noaa_gfs_decode_recovery_bootstrap.py",
    )
    recovery.require_canonical_outputs_absent(root)
    return {
        "status": "PASS_STATIC_DECODE_RECOVERY_SEALER_READY",
        "code_identities": identities,
        "sealer_identities": sealers,
        "future_outputs_absent": True,
        "network_requests": 0,
        "writes": 0,
    }


def spawn_importability_audit(root: Path, workspace_root: Path) -> dict[str, Any]:
    """Exercise the real top-level bootstrap through seven Windows spawn workers.

    This is a read-only entrypoint regression.  It deliberately does not validate
    or publish the large raw-cache seal; the full sealer repeats that closure.
    """

    require_canonical_module_entrypoint()
    install_entry_guard()
    load_recovery_runner_exact(workspace_root)
    recovery.validate_parent_bytecode_policy()
    recovery.install_parent_network_deny_guard()
    root = root.resolve()
    workspace_root = workspace_root.resolve()
    destinations = (
        root / RAW_LOCK_RELATIVE,
        root / PREFLIGHT_RELATIVE,
        root / AUTHORIZATION_RELATIVE,
        root / REVIEW_RELATIVE,
        root / GO_RELATIVE,
    )
    if any(path.exists() for path in destinations):
        raise RecoverySealError("spawn audit requires all future controls absent")
    recovery.require_canonical_outputs_absent(root)
    identities = code_identities(workspace_root)
    fixed = fixed_root_identities(root)
    incident_record = fixed["sealer_entrypoint_incident"]
    if incident_record["sha256"] != recovery.SEALER_ENTRYPOINT_INCIDENT_SHA256:
        raise RecoverySealError("spawn audit incident identity mismatch")
    original_auth = recovery.read_json_object(
        root / recovery.FIXED_ROOT_INPUTS["original_authorization"],
        "original authorization",
    )
    runtime = runtime_lock(workspace_root, original_auth["runtime_identity"])
    import_contract = recovery.bootstrap_import_contract_report(workspace_root)
    if "src" in sys.modules:
        raise RecoverySealError("src package initializer executed during spawn audit")
    before_pyc = recovery._matching_bootstrap_bytecode(
        workspace_root / "src" / "noaa_gfs_decode_recovery_bootstrap.py"
    )
    if before_pyc:
        raise RecoverySealError("bootstrap pyc exists before spawn audit")
    bootstrap = recovery._load_recovery_bootstrap_preexecution(
        identities["recovery_bootstrap"], workspace_root
    )
    core = recovery._load_recovery_core_preexecution(
        identities["recovery_module"], workspace_root
    )
    core.validate_eccodes_runtime(runtime)
    pilot = external_identity(workspace_root / PILOT_PATH)
    report = bootstrap.run_real_spawn_pilot(
        pilot,
        runtime,
        identities["recovery_bootstrap"],
        identities["recovery_runner"],
        identities["recovery_module"],
        workers=7,
        repetitions=70,
    )
    if (
        report.get("first_wave_synchronized") is not True
        or len(report.get("worker_pids_observed", [])) != 7
        or report.get("tasks") != 70
    ):
        raise RecoverySealError("canonical module-entrypoint spawn coverage mismatch")
    if recovery.bootstrap_import_contract_report(workspace_root) != import_contract:
        raise RecoverySealError("bootstrap import contract changed during spawn audit")
    if "src" in sys.modules:
        raise RecoverySealError("src package initializer executed during spawn audit")
    if recovery._matching_bootstrap_bytecode(
        workspace_root / "src" / "noaa_gfs_decode_recovery_bootstrap.py"
    ):
        raise RecoverySealError("bootstrap pyc appeared during spawn audit")
    if any(path.exists() for path in destinations):
        raise RecoverySealError("control artifact appeared during spawn audit")
    recovery.require_canonical_outputs_absent(root)
    if code_identities(workspace_root) != identities:
        raise RecoverySealError("code/test identity changed during spawn audit")
    return {
        "status": "PASS_CANONICAL_MODULE_ENTRYPOINT_TOP_LEVEL_BOOTSTRAP_7SPAWN",
        "canonical_module_entrypoint": CANONICAL_MODULE_ENTRYPOINT,
        "bootstrap_import_contract": import_contract,
        "sealer_entrypoint_incident": incident_record,
        "worker_pids_observed": report["worker_pids_observed"],
        "worker_count": len(report["worker_pids_observed"]),
        "tasks": report["tasks"],
        "first_wave_synchronized": report["first_wave_synchronized"],
        "src_package_initializer_executed": False,
        "controls_created": 0,
        "network_requests": 0,
        "writes": 0,
    }


def seal(root: Path, workspace_root: Path) -> dict[str, Any]:
    require_canonical_module_entrypoint()
    install_entry_guard()
    load_recovery_runner_exact(workspace_root)
    recovery.validate_parent_bytecode_policy()
    recovery.install_parent_network_deny_guard()
    root = root.resolve()
    workspace_root = workspace_root.resolve()
    destinations = {
        "raw_cache_lock": root / RAW_LOCK_RELATIVE,
        "real_spawn_preflight": root / PREFLIGHT_RELATIVE,
        "authorization": root / AUTHORIZATION_RELATIVE,
        "independent_review": root / REVIEW_RELATIVE,
        "independent_go": root / GO_RELATIVE,
    }
    if any(path.exists() for path in destinations.values()):
        raise RecoverySealError("decode recovery seal/independent authority already exists")
    recovery.require_canonical_outputs_absent(root)

    identities = code_identities(workspace_root)
    sealers = sealer_identities(workspace_root)
    recovery.validate_bootstrap_preexecution(
        identities["recovery_bootstrap"],
        workspace_root / "src" / "noaa_gfs_decode_recovery_bootstrap.py",
    )
    fixed = fixed_root_identities(root)
    incident = fixed["incident"]
    if incident["sha256"] != recovery.INCIDENT_SHA256:
        raise RecoverySealError("incident constant mismatch")
    entrypoint_incident = fixed["sealer_entrypoint_incident"]
    if (
        entrypoint_incident["sha256"]
        != recovery.SEALER_ENTRYPOINT_INCIDENT_SHA256
    ):
        raise RecoverySealError("sealer-entrypoint incident constant mismatch")
    entrypoint_payload = recovery.read_json_object(
        root / recovery.FIXED_ROOT_INPUTS["sealer_entrypoint_incident"],
        "sealer-entrypoint incident",
    )
    if (
        entrypoint_payload.get("artifact_type")
        != "TRACK_A_DECODE_RECOVERY_SEAL_DIRECT_FILE_PICKLE_FAILURE_INCIDENT"
        or entrypoint_payload.get("schema_version") != 1
        or entrypoint_payload.get("status")
        != "FAILED_BEFORE_PUBLICATION_RETRY_FORBIDDEN_PENDING_REFREEZE"
        or entrypoint_payload.get("failure_boundary", {}).get(
            "seal_publications_created"
        )
        != 0
        or entrypoint_payload.get("failure_boundary", {}).get("network_requests")
        != 0
        or entrypoint_payload.get("failure_boundary", {}).get(
            "recovery_launch_started"
        )
        is not False
        or entrypoint_payload.get("mandatory_remediation")
        != recovery.expected_sealer_entrypoint_remediation(workspace_root)
    ):
        raise RecoverySealError("sealer-entrypoint incident semantics mismatch")
    original_auth_path = root / recovery.FIXED_ROOT_INPUTS["original_authorization"]
    original_auth = recovery.read_json_object(original_auth_path, "original authorization")
    original_runtime = original_auth["runtime_identity"]
    runtime = runtime_lock(workspace_root, original_runtime)
    bootstrap_import_contract = recovery.bootstrap_import_contract_report(
        workspace_root
    )
    if runtime["frozen_runner"] != identities["original_runner"]:
        raise RecoverySealError("runtime/original runner identity mismatch")
    if runtime["recovery_module"] != identities["recovery_module"]:
        raise RecoverySealError("runtime/recovery core identity mismatch")

    frozen = recovery._import_frozen_runner(identities["original_runner"], workspace_root)
    range_path = root / recovery.FIXED_ROOT_INPUTS["field_range_census"]
    rows = recovery.read_range_rows(range_path, fixed["field_range_census"])
    prerecovery_topology = recovery.validate_prerecovery_topology(root)
    raw_rows_before, inventories_before = recovery.validate_raw_cache(root, rows, frozen)
    incident_payload = recovery.read_json_object(root / recovery.FIXED_ROOT_INPUTS["incident"], "incident")
    remnants = incident_payload["documented_preplan_staging_remnants"]
    if not isinstance(remnants, list) or len(remnants) != 2:
        raise RecoverySealError("incident remnant schema mismatch")
    for record in remnants:
        recovery.require_root_identity(
            record, root, str(record["path"]), "documented preplan remnant"
        )

    raw_lock_payload = {
        "schema_version": 1,
        "artifact_type": "TRACK_A_DECODE_RECOVERY_RAW_CACHE_LOCK",
        "status": "PASS_IMMUTABLE_RAW_CACHE_READY_FOR_DECODE_ONLY",
        "created_utc": None,
        "incident": incident,
        "inventories": inventories_before,
        "prerecovery_topology": prerecovery_topology,
        "documented_preplan_remnants": remnants,
        "canonical_outputs_absent": True,
        "network_requests_during_preflight": 0,
        "raw_event_progress_mutations": 0,
        "labels_read": False,
        "2024_arrays_read": False,
        "2025_arrays_read": False,
    }
    if set(raw_lock_payload) != RAW_LOCK_KEYS:
        raise RecoverySealError("internal raw-cache lock schema mismatch")

    bootstrap = recovery._load_recovery_bootstrap_preexecution(
        identities["recovery_bootstrap"], workspace_root
    )
    core = recovery._load_recovery_core_preexecution(
        identities["recovery_module"], workspace_root
    )
    core.validate_eccodes_runtime(runtime)
    pilot = external_identity(workspace_root / PILOT_PATH)
    pyc_absent_before = not recovery._matching_bootstrap_bytecode(
        workspace_root / "src" / "noaa_gfs_decode_recovery_bootstrap.py"
    )
    pilot_report = bootstrap.run_real_spawn_pilot(
        pilot,
        runtime,
        identities["recovery_bootstrap"],
        identities["recovery_runner"],
        identities["recovery_module"],
        workers=7,
        repetitions=70,
    )
    coordinate = recovery.read_json_object(
        root / recovery.FIXED_ROOT_INPUTS["coordinate_lock"], "coordinate lock"
    )
    sites = coordinate.get("sites")
    if not isinstance(sites, list) or len(sites) != 17:
        raise RecoverySealError("coordinate site schema mismatch")
    production_report = build_production_pilot_report(
        bootstrap, pilot, runtime, identities, sites
    )
    if (
        recovery.bootstrap_import_contract_report(workspace_root)
        != bootstrap_import_contract
    ):
        raise RecoverySealError("bootstrap alias inventory changed after pilots")
    recovery.validate_bootstrap_preexecution(
        identities["recovery_bootstrap"],
        workspace_root / "src" / "noaa_gfs_decode_recovery_bootstrap.py",
    )
    if (
        recovery.bootstrap_import_contract_report(workspace_root)
        != bootstrap_import_contract
    ):
        raise RecoverySealError("bootstrap alias inventory changed before seal publication")
    pyc_absent_after = not recovery._matching_bootstrap_bytecode(
        workspace_root / "src" / "noaa_gfs_decode_recovery_bootstrap.py"
    )
    if not pyc_absent_before or not pyc_absent_after:
        raise RecoverySealError("bootstrap pyc appeared during process preflight")

    # The compile/test record is embedded in the process preflight itself.  Each
    # test file runs in a clean subprocess so pytest collection cannot preimport
    # the recovery runner ahead of the exact-source loader used by sealer tests.
    test_evidence = run_immutable_test_evidence(
        workspace_root, identities, sealers
    )

    # Full topology, fixed-input, remnant, raw-cache and code closure is repeated
    # after every process/test action and immediately before the first seal write.
    final_topology = recovery.validate_prerecovery_topology(root)
    if final_topology != prerecovery_topology:
        raise RecoverySealError("pre-recovery topology changed during preflight")
    recovery.require_canonical_outputs_absent(root)
    raw_rows_after, inventories_after = recovery.validate_raw_cache(root, rows, frozen)
    if inventories_after != inventories_before or raw_rows_after != raw_rows_before:
        raise RecoverySealError("raw cache changed during recovery preflight")
    fixed_after = fixed_root_identities(root)
    if fixed_after != fixed:
        raise RecoverySealError("fixed root inputs changed during recovery preflight")
    for record in remnants:
        if recovery.root_identity(
            recovery.require_root_identity(
                record, root, str(record["path"]), "final documented preplan remnant"
            ),
            root,
        ) != record:
            raise RecoverySealError("documented preplan remnant changed during preflight")
    if runtime_lock(workspace_root, original_runtime) != runtime:
        raise RecoverySealError("runtime identity changed during recovery preflight")
    if external_identity(workspace_root / PILOT_PATH) != pilot:
        raise RecoverySealError("real GRIB pilot changed during recovery preflight")
    if code_identities(workspace_root) != identities or sealer_identities(
        workspace_root
    ) != sealers:
        raise RecoverySealError("recovery code/test identities changed during preflight")
    recovery.validate_bootstrap_preexecution(
        identities["recovery_bootstrap"],
        workspace_root / "src" / "noaa_gfs_decode_recovery_bootstrap.py",
    )
    if any(path.exists() for path in destinations.values()):
        raise RecoverySealError("seal/authority file appeared during preflight")

    created = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    compact = created.replace("-", "").replace(":", "").replace(".", "")
    attempt_id = f"decode_recovery_v1__{compact}"
    raw_lock_payload["created_utc"] = created
    test_evidence["created_utc"] = created
    preflight_payload = {
        "schema_version": 1,
        "artifact_type": "TRACK_A_DECODE_RECOVERY_REAL_PROCESS_PREFLIGHT",
        "status": "PASS_REAL_PILOT_AND_PRODUCTION_PATH_EXACT_7_SPAWN_STDLIB_BOOTSTRAP",
        "created_utc": created,
        "incident": incident,
        "recovery_runner": identities["recovery_runner"],
        "recovery_module": identities["recovery_module"],
        "recovery_bootstrap": identities["recovery_bootstrap"],
        "recovery_test": identities["recovery_test"],
        "recovery_bootstrap_test": identities["recovery_bootstrap_test"],
        "recovery_runner_test": identities["recovery_runner_test"],
        "postrun_auditor": identities["postrun_auditor"],
        "postrun_auditor_test": identities["postrun_auditor_test"],
        "recovery_sealer": sealers["recovery_sealer"],
        "recovery_sealer_test": sealers["recovery_sealer_test"],
        "bootstrap_trust_policy": recovery.BOOTSTRAP_TRUST_POLICY,
        "runtime_lock": runtime,
        "bootstrap_import_contract": bootstrap_import_contract,
        "test_evidence": test_evidence,
        "process_scheduling_flake_incident": SPAWN_SCHEDULING_FLAKE_INCIDENT,
        "sealer_entrypoint_incident": entrypoint_incident,
        "real_grib_pilot": pilot,
        "max_spawn_processes": 7,
        "thread_decoders": 0,
        "network_requests": 0,
        "production_decode_task_worker_real_test": True,
        "stdlib_bootstrap_worker_real_test": True,
        "poisoned_pyc_source_execution_test": test_evidence[
            "required_test_names_present"
        ],
        "bootstrap_ast_stdlib_policy_pass": test_evidence[
            "required_test_names_present"
        ],
        "python_dont_write_bytecode_env": test_evidence[
            "python_dont_write_bytecode_env"
        ],
        "python_dont_write_bytecode_flag": test_evidence[
            "python_dont_write_bytecode_flag"
        ],
        "python_pycache_prefix_absent": test_evidence[
            "python_pycache_prefix_absent"
        ],
        "bootstrap_matching_pyc_absent_before_tests": test_evidence[
            "bootstrap_matching_pyc_absent_before_tests"
        ],
        "bootstrap_matching_pyc_absent_after_tests": test_evidence[
            "bootstrap_matching_pyc_absent_after_tests"
        ],
        "all_seven_worker_pids_observed": True,
        "pilot_report": pilot_report,
        "production_worker_report": production_report,
        "focused_test_result": test_evidence["pytest_summary"],
        "source_compile_result": test_evidence["source_compile_result"],
        "labels_read": False,
        "arrays_2024_read": False,
        "arrays_2025_read": False,
        "models_fit": 0,
        "submission_csv_created": False,
    }
    if set(preflight_payload) != PREFLIGHT_KEYS:
        raise RecoverySealError("internal real-process preflight schema mismatch")

    publish_json_no_overwrite(destinations["raw_cache_lock"], raw_lock_payload)
    raw_lock_identity = recovery.root_identity(destinations["raw_cache_lock"], root)
    publish_json_no_overwrite(destinations["real_spawn_preflight"], preflight_payload)
    preflight_identity = recovery.root_identity(destinations["real_spawn_preflight"], root)

    authorization = {
        "schema_version": 1,
        "artifact_type": "NOAA_GFS_NETWORK_ZERO_DECODE_RECOVERY_AUTHORIZATION",
        "status": "AUTHORIZED_PENDING_INDEPENDENT_GO",
        "created_utc": created,
        "experiment_id": "noaa_gfs_dminus2_12z_multiseason_target_free_decode_recovery_v1",
        "recovery_attempt_id": attempt_id,
        "incident": incident,
        **identities,
        **fixed,
        "raw_cache_lock": raw_lock_identity,
        "real_spawn_preflight": preflight_identity,
        "documented_preplan_remnants": remnants,
        "runtime_lock": runtime,
        "bootstrap_trust_policy": recovery.BOOTSTRAP_TRUST_POLICY,
        "bootstrap_import_contract": bootstrap_import_contract,
        "real_grib_pilot": pilot,
        "original_runtime_identity": original_runtime,
        "expected_range_rows": recovery.EXPECTED_ROWS,
        "expected_range_bytes": recovery.EXPECTED_BYTES,
        "max_decode_processes": recovery.MAX_DECODE_PROCESSES,
        "network_requests_allowed": 0,
        "raw_event_progress_mutation_allowed": False,
        "labels_read": False,
        "2024_arrays_read": False,
        "2025_arrays_read": False,
        "models_fit": 0,
        "submission_csv_created": False,
    }
    if set(authorization) != AUTHORIZATION_KEYS:
        raise RecoverySealError("internal recovery authorization schema mismatch")
    publish_json_no_overwrite(destinations["authorization"], authorization)
    return {
        "status": "PASS_AUTHORIZATION_SEALED_PENDING_INDEPENDENT_REVIEW_AND_GO",
        "raw_cache_lock": recovery.root_identity(destinations["raw_cache_lock"], root),
        "real_spawn_preflight": recovery.root_identity(
            destinations["real_spawn_preflight"], root
        ),
        "authorization": recovery.root_identity(destinations["authorization"], root),
        "recovery_attempt_id": attempt_id,
        "independent_review_created": False,
        "independent_go_created": False,
        "network_requests": 0,
        "labels_read": False,
        "2024_arrays_read": False,
        "2025_arrays_read": False,
        "models_fit": 0,
        "submission_csv_created": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--static-audit", action="store_true")
    mode.add_argument("--spawn-importability-audit", action="store_true")
    args = parser.parse_args()
    workspace_root = Path(__file__).resolve().parents[1]
    try:
        if args.static_audit:
            report = static_audit(args.root.resolve(), workspace_root)
        elif args.spawn_importability_audit:
            require_canonical_module_entrypoint()
            report = spawn_importability_audit(
                args.root.resolve(), workspace_root
            )
        else:
            require_canonical_module_entrypoint()
            report = seal(args.root.resolve(), workspace_root)
    except Exception as exc:
        report = {
            "status": "FAIL_DECODE_RECOVERY_SEAL",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "network_requests": 0,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(2) from None
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

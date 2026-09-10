"""Network-zero, process-isolated recovery of the completed Track-A raw cache.

This runner exists only because the frozen producer downloaded every one of
10,368 authorized byte ranges and then failed at the first threaded ecCodes
decode.  It never invokes the producer's download path.  All raw files,
sidecars, request events and original progress records are immutable inputs.

Execution is fail-closed and requires a separately sealed authorization and
independent GO.  ``--static-audit`` performs no writes and reads no GRIB value.
"""

from __future__ import annotations

import argparse
import ast
import atexit
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import importlib.machinery
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import socket
import sys
import types
from typing import Any


ROOT_DEFAULT = Path("artifacts/baram2026_ncei_scada_longrun_20260810_v2")
EXPECTED_ROWS = 10_368
EXPECTED_BYTES = 10_296_112_890
MAX_DECODE_PROCESSES = 7
ORIGINAL_PRODUCER_MAX_WORKERS = 8
BOOTSTRAP_TOP_LEVEL_MODULE_NAME = "noaa_gfs_decode_recovery_bootstrap"
BOOTSTRAP_TRUST_POLICY = {
    "sole_explicit_preinitializer_trust_anchor": True,
    "adversarial_code_swap_proof_claimed": False,
    "source_top_level_stdlib_allowlist_required": True,
    "matching_bootstrap_pyc_required_absent_before_and_after": True,
    "python_dont_write_bytecode_env_required": "1",
    "python_pycache_prefix_required_absent": True,
    "child_detects_drift_before_decoder_core_work": True,
    "top_level_module_name": BOOTSTRAP_TOP_LEVEL_MODULE_NAME,
    "workspace_src_search_path_required_first": True,
    "pythonpath_required_exact_workspace_src": True,
    "src_package_initializer_executed": False,
    "bootstrap_module_package": "",
    "stdlib_shadow_candidates_required_absent": True,
    "top_level_module_competing_candidates_required_absent": True,
}
BOOTSTRAP_STDLIB_IMPORTS = (
    "__future__",
    "ast",
    "collections.abc",
    "concurrent.futures",
    "hashlib",
    "importlib.machinery",
    "multiprocessing",
    "os",
    "pathlib",
    "socket",
    "sys",
    "types",
    "typing",
)
FAILED_ATTEMPT = "20260810T151057590738Z__pid3536__5f3e0fb20132"
FAILED_ATTEMPT_PID = 3536
ORIGINAL_PROGRESS_CHECKPOINTS = tuple(range(100, EXPECTED_ROWS, 100)) + (
    EXPECTED_ROWS,
)
ORIGINAL_PROGRESS_KEYS = frozenset(
    {
        "actual_http_attempts_cumulative",
        "attempt_id",
        "checkpoint_name",
        "completed_key_set_sha256",
        "completed_ranges",
        "created_utc",
        "network_bytes_this_invocation_so_far",
        "network_requests_this_invocation_so_far",
        "phase",
    }
)
ORIGINAL_FAILED_LOCK_KEYS = frozenset({"attempt_id", "created_utc", "pid"})
ORIGINAL_START_EVENT_KEYS = frozenset(
    {
        "attempt",
        "created_utc",
        "event",
        "event_id",
        "global_raw_attempt_number",
        "range_end",
        "range_start",
        "url",
    }
)
ORIGINAL_RUNNER_SHA256 = (
    "51e6c4b68d47916f56975b16b9ceaae2e345f9f68963dba197501448d655934c"
)
ORIGINAL_AUTHORIZATION_SHA256 = (
    "ff9a133e913a1b5a4a62028578b54eb4c5a43a067d6ff0da41357bfd2c889cb9"
)
ORIGINAL_GO_SHA256 = (
    "c194b7641b9ca9cca5b9416b1ff5a0dcae8777d33bd150fa1ee0f6ca19802beb"
)
INCIDENT_SHA256 = (
    "52bd92b00d3351e6b1061e47e3306ccd621311913113d6652033081be1cd836c"
)
SEALER_ENTRYPOINT_INCIDENT_SHA256 = (
    "9e74f57dcddbd0480464fda5953014e5be411b927de99beef58d7ab5bb10afbb"
)

FIELD_RANGE_COLUMNS = (
    "target_operating_day_kst",
    "run_init_utc",
    "forecast_hour",
    "valid_time_utc",
    "cutoff_utc",
    "object_key",
    "object_size_bytes",
    "object_etag",
    "publication_last_modified_utc",
    "cutoff_margin_seconds",
    "idx_key",
    "idx_size_bytes",
    "idx_etag",
    "idx_last_modified_utc",
    "status",
    "source_archive",
    "archive_product",
    "retrieval_url_or_request_id",
    "official_metadata",
    "publication_evidence_type",
    "publication_evidence_reference",
    "family",
    "variable",
    "level",
    "grib_record_number",
    "forecast_descriptor",
    "range_start",
    "range_end",
    "range_bytes",
    "idx_relative_path",
    "idx_sha256",
)

FEATURES = ("HPBL_surface",) + tuple(
    f"{variable}_{level}mb"
    for level in (925, 950, 975, 1000)
    for variable in ("UGRD", "VGRD")
)

CANONICAL_OUTPUTS = {
    "raw_parquet": "raw/RAW_RANGE_MANIFEST.parquet",
    "raw_csv": "raw/RAW_RANGE_MANIFEST.csv",
    "site": "decoded/site_values_target_free.parquet",
    "group": "decoded/group_values_target_free.parquet",
    "audit": "decoded/PHYSICAL_AND_COVERAGE_AUDIT.json",
    "lock": "decoded/DECODED_MATRIX_LOCK.json",
    "access": "raw/RAW_ACCESS_LEDGER.json",
    "manifest": "manifest_raw_v1.json",
}

FIXED_ROOT_INPUTS = {
    "incident": "incidents/TRACK_A_RAW_DECODE_MEMFS_THREAD_INIT_FAILURE_V1.json",
    "sealer_entrypoint_incident": (
        "incidents/TRACK_A_DECODE_RECOVERY_SEAL_DIRECT_FILE_PICKLE_FAILURE_V1.json"
    ),
    "original_authorization": "prereg/raw_launch_authorization_v1.json",
    "original_independent_go": "independent_redteam/TRACK_A_RAW_LAUNCH_GO.json",
    "original_failed_lock": (
        "raw/launch_history/"
        f"{FAILED_ATTEMPT}__failed.lock"
    ),
    "original_final_progress": (
        "raw/progress/"
        f"raw_ranges__{FAILED_ATTEMPT}__010368.json"
    ),
    "original_stderr": "logs/TRACK_A_RAW_RUN.stderr.log",
    "original_stdout": "logs/TRACK_A_RAW_RUN.stdout.log",
    "field_range_census": "census/field_range_census.parquet",
    "census_manifest": "manifest_census_v1.json",
    "coordinate_lock": "prereg/authoritative_turbine_coordinate_lock_v1.json",
    "raw_cache_lock": "prelaunch/DECODE_RECOVERY_RAW_CACHE_LOCK_V1.json",
    "real_spawn_preflight": "prelaunch/DECODE_RECOVERY_REAL_PROCESS_PREFLIGHT_V1.json",
}

PREPLAN_REMNANTS = (
    (
        "raw/output_transactions/"
        f"{FAILED_ATTEMPT}/staged/raw/RAW_RANGE_MANIFEST.csv"
    ),
    (
        "raw/output_transactions/"
        f"{FAILED_ATTEMPT}/staged/raw/RAW_RANGE_MANIFEST.parquet"
    ),
)
INDEPENDENT_REVIEW_RELATIVE = (
    "independent_redteam/TRACK_A_DECODE_RECOVERY_INDEPENDENT_REVIEW_V1.json"
)
INDEPENDENT_REVIEW_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "status",
        "created_utc",
        "authorization",
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
        "raw_cache_lock",
        "real_spawn_preflight",
        "max_decode_processes",
        "network_requests_allowed",
        "verdict",
        "independent_checks",
    }
)
INDEPENDENT_REVIEW_CHECK_KEYS = frozenset(
    {
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
REAL_SPAWN_PREFLIGHT_KEYS = frozenset(
    {
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
)
EMBEDDED_TEST_EVIDENCE_KEYS = frozenset(
    {
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
)


def validate_noncontract_monolithic_diagnostic(payload: Any) -> None:
    if payload != NONCONTRACT_MONOLITHIC_DIAGNOSTIC:
        raise RecoveryContractError(
            "noncontract monolithic diagnostic exact semantics mismatch"
        )


class RecoveryContractError(RuntimeError):
    """A recovery authorization, state or identity failed closed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_exact_file(record: Mapping[str, Any], label: str) -> Path:
    if set(record) != {"path", "size_bytes", "sha256"}:
        raise RecoveryContractError(f"{label} identity schema mismatch")
    raw_path = Path(str(record["path"]))
    if not raw_path.is_absolute():
        raise RecoveryContractError(f"{label} path is not absolute")
    lexical = Path(os.path.abspath(str(raw_path)))
    for candidate in (lexical, *lexical.parents):
        if candidate.is_symlink() or (
            hasattr(candidate, "is_junction") and candidate.is_junction()
        ):
            raise RecoveryContractError(f"{label} path is a symlink/junction")
    if not lexical.is_file():
        raise RecoveryContractError(f"{label} file is absent")
    if lexical.stat().st_size != int(record["size_bytes"]):
        raise RecoveryContractError(f"{label} size mismatch")
    if sha256_file(lexical) != str(record["sha256"]):
        raise RecoveryContractError(f"{label} hash mismatch")
    return lexical.resolve()


def _network_forbidden(*_args: Any, **_kwargs: Any) -> Any:
    raise RecoveryContractError("network access is forbidden during decode recovery")


_PARENT_NETWORK_GUARD_INSTALLED = False


def _parent_network_audit_guard(event: str, _args: tuple[Any, ...]) -> None:
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
        raise RecoveryContractError(
            f"network/socket audit event is forbidden during recovery: {event}"
        )


def install_parent_network_deny_guard() -> None:
    global _PARENT_NETWORK_GUARD_INSTALLED
    if not _PARENT_NETWORK_GUARD_INSTALLED:
        sys.addaudithook(_parent_network_audit_guard)
        _PARENT_NETWORK_GUARD_INSTALLED = True
    socket.create_connection = _network_forbidden  # type: ignore[assignment]


def _matching_bootstrap_bytecode(path: Path) -> list[Path]:
    matches: list[Path] = []
    legacy = path.with_suffix(".pyc")
    if legacy.exists():
        matches.append(legacy)
    cache = path.parent / "__pycache__"
    if cache.exists():
        matches.extend(cache.glob(f"{path.stem}.*.pyc"))
    return sorted(matches)


def validate_parent_bytecode_policy() -> None:
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1" or not sys.dont_write_bytecode:
        raise RecoveryContractError("PYTHONDONTWRITEBYTECODE=1/-B is required")
    if os.environ.get("PYTHONPYCACHEPREFIX") not in (None, "") or sys.pycache_prefix is not None:
        raise RecoveryContractError("external Python bytecode cache prefix is forbidden")


def validate_bootstrap_preexecution(
    record: Mapping[str, Any], expected: Path
) -> Path:
    validate_parent_bytecode_policy()
    path = require_external_identity(record, expected.resolve(), "recovery bootstrap")
    if _matching_bootstrap_bytecode(path):
        raise RecoveryContractError("recovery bootstrap bytecode cache must be absent")
    try:
        tree = ast.parse(path.read_bytes(), filename=str(path))
    except Exception as exc:
        raise RecoveryContractError("recovery bootstrap AST is invalid") from exc
    imports: list[str] = []
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            imports.extend(alias.name for alias in statement.names)
        elif isinstance(statement, ast.ImportFrom):
            imports.append(str(statement.module))
        elif isinstance(statement, ast.Expr):
            if not isinstance(statement.value, ast.Constant) or not isinstance(
                statement.value.value, str
            ):
                raise RecoveryContractError("bootstrap top-level expression is forbidden")
        elif isinstance(statement, (ast.Assign, ast.AnnAssign)):
            value = statement.value
            if value is not None and any(isinstance(node, ast.Call) for node in ast.walk(value)):
                raise RecoveryContractError("bootstrap top-level call is forbidden")
        elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if statement.decorator_list or any(
                isinstance(node, ast.Call)
                for node in (*statement.args.defaults, *statement.args.kw_defaults)
                if node is not None
            ):
                raise RecoveryContractError("bootstrap function definition does work")
        elif isinstance(statement, ast.ClassDef):
            if statement.decorator_list or any(
                isinstance(node, ast.Call)
                for base in (*statement.bases, *statement.keywords)
                for node in ast.walk(base)
            ):
                raise RecoveryContractError("bootstrap class definition does work")
            for member in statement.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if member.decorator_list:
                        raise RecoveryContractError("bootstrap class decorator forbidden")
                elif isinstance(member, ast.Expr) and isinstance(member.value, ast.Constant):
                    continue
                elif isinstance(member, (ast.Assign, ast.AnnAssign)):
                    value = member.value
                    if value is not None and any(
                        isinstance(node, ast.Call) for node in ast.walk(value)
                    ):
                        raise RecoveryContractError("bootstrap class-body call forbidden")
                else:
                    raise RecoveryContractError("bootstrap class-body work forbidden")
        else:
            raise RecoveryContractError("bootstrap top-level statement forbidden")
    if tuple(imports) != BOOTSTRAP_STDLIB_IMPORTS:
        raise RecoveryContractError("bootstrap stdlib import whitelist mismatch")
    if sha256_file(path) != str(record["sha256"]):
        raise RecoveryContractError("bootstrap changed during pre-execution audit")
    return path


def _load_recovery_core_preexecution(
    record: Mapping[str, Any], workspace_root: Path
) -> Any:
    """Hash and origin-lock the recovery core before executing its module body."""

    expected = (workspace_root / "src" / "noaa_gfs_decode_recovery.py").resolve()
    path = require_external_identity(record, expected, "recovery module")
    module_name = "src.noaa_gfs_decode_recovery"
    if module_name in sys.modules:
        raise RecoveryContractError("recovery module imported before identity validation")
    module = execute_exact_source_module(
        module_name, path, record, "recovery module"
    )
    if int(module.MAX_DECODE_PROCESSES) != MAX_DECODE_PROCESSES:
        raise RecoveryContractError("recovery process-count constant mismatch")
    return module


def install_bootstrap_top_level_import_contract(workspace_root: Path) -> dict[str, Any]:
    workspace_root = workspace_root.resolve()
    lexical = workspace_root / "src"
    for candidate in (lexical, *lexical.parents):
        if candidate.is_symlink() or (
            hasattr(candidate, "is_junction") and candidate.is_junction()
        ):
            raise RecoveryContractError("bootstrap import root is a symlink/junction")
    if not lexical.is_dir() or lexical.resolve() != lexical:
        raise RecoveryContractError("bootstrap import root is not an exact directory")
    if "src" in sys.modules:
        raise RecoveryContractError("src package initializer executed before bootstrap guard")
    stdlib_roots = tuple(
        sorted({name.split(".", 1)[0] for name in BOOTSTRAP_STDLIB_IMPORTS})
    )
    shadow_candidates = tuple(
        sorted(
            child.name
            for child in lexical.iterdir()
            if any(
                child.name.casefold() == root.casefold()
                or child.name.casefold().startswith(f"{root}.".casefold())
                for root in stdlib_roots
            )
        )
    )
    if shadow_candidates:
        raise RecoveryContractError(
            "workspace src shadows bootstrap stdlib import roots: "
            + ",".join(shadow_candidates)
        )
    expected_alias_source = f"{BOOTSTRAP_TOP_LEVEL_MODULE_NAME}.py"
    alias_candidates = tuple(
        sorted(
            (
                child
                for child in lexical.iterdir()
                if child.name.casefold()
                == BOOTSTRAP_TOP_LEVEL_MODULE_NAME.casefold()
                or child.name.casefold().startswith(
                    f"{BOOTSTRAP_TOP_LEVEL_MODULE_NAME}.".casefold()
                )
            ),
            key=lambda child: child.name.casefold(),
        )
    )
    if any(
        candidate.is_symlink()
        or (hasattr(candidate, "is_junction") and candidate.is_junction())
        for candidate in alias_candidates
    ):
        raise RecoveryContractError("bootstrap alias candidate is a symlink/junction")
    if tuple(candidate.name for candidate in alias_candidates) != (
        expected_alias_source,
    ) or not alias_candidates[0].is_file():
        raise RecoveryContractError(
            "bootstrap top-level alias source candidate set is not unique"
        )
    source_root = str(lexical)
    retained: list[str] = []
    for entry in sys.path:
        try:
            same = Path(entry or os.getcwd()).resolve() == lexical
        except (OSError, RuntimeError):
            same = False
        if not same:
            retained.append(entry)
    sys.path[:] = [source_root, *retained]
    os.environ["PYTHONPATH"] = source_root
    if (
        not sys.path
        or Path(sys.path[0]).resolve() != lexical
        or os.environ.get("PYTHONPATH") != source_root
        or "src" in sys.modules
    ):
        raise RecoveryContractError("bootstrap top-level import contract did not install")
    return {
        "module_name": BOOTSTRAP_TOP_LEVEL_MODULE_NAME,
        "source_root": source_root,
        "sys_path_index": 0,
        "pythonpath_exact": source_root,
        "src_package_initializer_executed": False,
        "bootstrap_module_package": "",
        "stdlib_import_roots": list(stdlib_roots),
        "stdlib_shadow_candidates": [],
        "top_level_module_competing_candidates": [],
        "bootstrap_source_path": str(
            lexical / "noaa_gfs_decode_recovery_bootstrap.py"
        ),
    }


def bootstrap_import_contract_report(workspace_root: Path) -> dict[str, Any]:
    report = install_bootstrap_top_level_import_contract(workspace_root)
    module = sys.modules.get(BOOTSTRAP_TOP_LEVEL_MODULE_NAME)
    if module is not None:
        module_spec = getattr(module, "__spec__", None)
        if (
            str(getattr(module, "__name__", ""))
            != BOOTSTRAP_TOP_LEVEL_MODULE_NAME
            or getattr(module, "__package__", None) not in ("", None)
            or Path(str(getattr(module, "__file__", ""))).resolve()
            != Path(report["bootstrap_source_path"]).resolve()
            or module_spec is None
            or Path(str(getattr(module_spec, "origin", ""))).resolve()
            != Path(report["bootstrap_source_path"]).resolve()
        ):
            raise RecoveryContractError("loaded top-level bootstrap identity mismatch")
    return report


def expected_sealer_entrypoint_remediation(workspace_root: Path) -> dict[str, Any]:
    return {
        "canonical_write_invocation": (
            "python -B -m scripts.seal_noaa_gfs_multiseason_decode_recovery_v1"
        ),
        "direct_file_write_invocation_forbidden": True,
        "bootstrap_top_level_module_name": BOOTSTRAP_TOP_LEVEL_MODULE_NAME,
        "bootstrap_import_root": str((workspace_root.resolve() / "src").resolve()),
        "src_package_initializer_must_not_execute": True,
        "bootstrap_source_origin_hash_ast_and_pyc_gates_remain_required": True,
        "real_module_entrypoint_seven_spawn_regression_required": True,
        "revised_code_test_auditor_hash_chain_and_independent_pass_required": True,
        "retry_authorized": False,
    }


def _load_recovery_bootstrap_preexecution(
    record: Mapping[str, Any], workspace_root: Path
) -> Any:
    bootstrap_import_contract_report(workspace_root)
    expected = (
        workspace_root / "src" / "noaa_gfs_decode_recovery_bootstrap.py"
    ).resolve()
    path = validate_bootstrap_preexecution(record, expected)
    module_name = BOOTSTRAP_TOP_LEVEL_MODULE_NAME
    if module_name in sys.modules:
        raise RecoveryContractError("recovery bootstrap imported before identity validation")
    module = execute_exact_source_module(
        module_name, path, record, "recovery bootstrap"
    )
    if (
        module.__name__ != BOOTSTRAP_TOP_LEVEL_MODULE_NAME
        or module.__package__ not in ("", None)
    ):
        raise RecoveryContractError("recovery bootstrap top-level identity mismatch")
    module.validate_bootstrap_trust_anchor(record, expected_path=path)
    if int(module.MAX_DECODE_PROCESSES) != MAX_DECODE_PROCESSES:
        raise RecoveryContractError("bootstrap process-count constant mismatch")
    return module


def execute_exact_source_module(
    module_name: str,
    path: Path,
    record: Mapping[str, Any],
    label: str,
) -> Any:
    """Compile the verified source bytes directly; never consult ``__pycache__``."""

    source = path.read_bytes()
    if len(source) != int(record["size_bytes"]) or hashlib.sha256(source).hexdigest() != str(
        record["sha256"]
    ):
        raise RecoveryContractError(f"{label} source changed before compile")
    code = compile(source, str(path), "exec", dont_inherit=True)
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = module_name.rpartition(".")[0]
    module.__loader__ = None
    module.__cached__ = None
    module.__spec__ = importlib.machinery.ModuleSpec(
        module_name, loader=None, origin=str(path)
    )
    sys.modules[module_name] = module
    try:
        exec(code, module.__dict__)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    if Path(str(module.__file__)).resolve() != path:
        raise RecoveryContractError(f"{label} executed path mismatch")
    if sha256_file(path) != str(record["sha256"]):
        raise RecoveryContractError(f"{label} source changed during execution")
    return module


def canonical_json_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def strict_under_root(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise RecoveryContractError(f"invalid canonical relative path: {relative!r}")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise RecoveryContractError(f"path escapes recovery root: {relative}")
    lexical = root / path
    for candidate in (lexical, *lexical.parents):
        if candidate == root.parent:
            break
        if candidate.is_symlink() or (
            hasattr(candidate, "is_junction") and candidate.is_junction()
        ):
            raise RecoveryContractError(f"symlink/junction path forbidden: {candidate}")
    resolved = lexical.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RecoveryContractError(f"path resolves outside recovery root: {relative}") from exc
    return lexical


def root_identity(path: Path, root: Path) -> dict[str, Any]:
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise RecoveryContractError(f"identity path absent/symlink: {path}")
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def require_root_identity(
    record: Mapping[str, Any], root: Path, expected_relative: str, label: str
) -> Path:
    if set(record) != {"path", "size_bytes", "sha256"}:
        raise RecoveryContractError(f"{label} identity schema mismatch")
    if record["path"] != expected_relative:
        raise RecoveryContractError(f"{label} canonical path mismatch")
    path = strict_under_root(root, expected_relative)
    if path.is_symlink() or not path.is_file():
        raise RecoveryContractError(f"{label} absent or symlink")
    if path.stat().st_size != int(record["size_bytes"]):
        raise RecoveryContractError(f"{label} size mismatch")
    if sha256_file(path) != str(record["sha256"]):
        raise RecoveryContractError(f"{label} hash mismatch")
    return path


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RecoveryContractError(f"invalid {label} JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise RecoveryContractError(f"{label} is not a JSON object")
    return payload


def require_external_identity(
    record: Mapping[str, Any], expected_path: Path, label: str
) -> Path:
    path = _require_exact_file(record, label)
    if path != expected_path.resolve():
        raise RecoveryContractError(f"{label} path differs from canonical path")
    return path


def file_inventory(paths: Sequence[Path], root: Path) -> dict[str, Any]:
    rows = [root_identity(path, root) for path in sorted(paths)]
    return {
        "file_count": len(rows),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in rows),
        "identity_rows_sha256": canonical_json_sha256(rows),
    }


def exact_recursive_files(directory: Path) -> list[Path]:
    if (
        directory.is_symlink()
        or (hasattr(directory, "is_junction") and directory.is_junction())
        or not directory.is_dir()
    ):
        raise RecoveryContractError(f"required directory absent/symlink: {directory}")
    files: list[Path] = []
    for item in directory.rglob("*"):
        if item.is_symlink() or (
            hasattr(item, "is_junction") and item.is_junction()
        ):
            raise RecoveryContractError(f"unexpected symlink/junction: {item}")
        if item.is_file():
            files.append(item)
        elif not item.is_dir():
            raise RecoveryContractError(f"unexpected filesystem object: {item}")
    return sorted(files)


def exact_recursive_dirs(directory: Path) -> list[Path]:
    if (
        directory.is_symlink()
        or (hasattr(directory, "is_junction") and directory.is_junction())
        or not directory.is_dir()
    ):
        raise RecoveryContractError(f"required directory absent/symlink: {directory}")
    directories: list[Path] = []
    for item in directory.rglob("*"):
        if item.is_symlink() or (
            hasattr(item, "is_junction") and item.is_junction()
        ):
            raise RecoveryContractError(f"unexpected symlink/junction: {item}")
        if item.is_dir():
            directories.append(item)
        elif not item.is_file():
            raise RecoveryContractError(f"unexpected filesystem object: {item}")
    return sorted(directories)


def validate_prerecovery_topology(root: Path) -> dict[str, Any]:
    raw_root = root / "raw"
    expected_raw_top = {
        "launch_history",
        "output_transactions",
        "progress",
        "ranges",
        "request_events",
    }
    if (
        not raw_root.is_dir()
        or raw_root.is_symlink()
        or (hasattr(raw_root, "is_junction") and raw_root.is_junction())
    ):
        raise RecoveryContractError("raw root absent or symlink")
    observed_raw_top = {path.name for path in raw_root.iterdir()}
    if observed_raw_top != expected_raw_top or any(
        not (raw_root / name).is_dir()
        or (raw_root / name).is_symlink()
        or (
            hasattr(raw_root / name, "is_junction")
            and (raw_root / name).is_junction()
        )
        for name in expected_raw_top
    ):
        raise RecoveryContractError("raw top-level pre-recovery inventory mismatch")
    if (root / "decoded").exists():
        raise RecoveryContractError("decoded tree must be absent before recovery")
    transaction_files = exact_recursive_files(raw_root / "output_transactions")
    expected_transaction_files = {strict_under_root(root, path) for path in PREPLAN_REMNANTS}
    if set(transaction_files) != expected_transaction_files:
        raise RecoveryContractError("documented preplan transaction inventory mismatch")
    transaction_root = raw_root / "output_transactions"
    expected_transaction_dirs: set[Path] = set()
    for path in expected_transaction_files:
        parent = path.parent
        while parent != transaction_root:
            expected_transaction_dirs.add(parent)
            parent = parent.parent
    if set(exact_recursive_dirs(transaction_root)) != expected_transaction_dirs:
        raise RecoveryContractError("documented preplan transaction directory mismatch")
    return prerecovery_topology_record(root)


def prerecovery_topology_record(root: Path) -> dict[str, Any]:
    transaction_files = [strict_under_root(root, path) for path in PREPLAN_REMNANTS]
    return {
        "raw_top_level_entries": sorted(
            {
                "launch_history",
                "output_transactions",
                "progress",
                "ranges",
                "request_events",
            }
        ),
        "decoded_tree_absent": True,
        "documented_preplan_transactions": file_inventory(transaction_files, root),
    }


def _import_frozen_runner(
    record: Mapping[str, Any], workspace_root: Path
) -> Any:
    path = require_external_identity(
        record,
        workspace_root / "scripts" / "run_noaa_gfs_multiseason_raw_v2.py",
        "frozen raw runner",
    )
    if sha256_file(path) != ORIGINAL_RUNNER_SHA256:
        raise RecoveryContractError("frozen raw runner constant mismatch")
    if "scripts.run_noaa_gfs_multiseason_raw_v2" in sys.modules:
        raise RecoveryContractError("frozen runner imported before parent rehash")
    return execute_exact_source_module(
        "scripts.run_noaa_gfs_multiseason_raw_v2",
        path,
        record,
        "frozen raw runner",
    )


def validate_authorization(
    root: Path, authorization_path: Path, workspace_root: Path
) -> tuple[dict[str, Any], dict[str, Any], Any, Any, Any]:
    authorization = read_json_object(authorization_path, "recovery authorization")
    expected_keys = {
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
    # incident is already present in FIXED_ROOT_INPUTS; set semantics collapse it.
    if set(authorization) != expected_keys:
        raise RecoveryContractError(
            f"recovery authorization schema mismatch: {sorted(set(authorization) ^ expected_keys)}"
        )
    if (
        authorization["schema_version"] != 1
        or authorization["artifact_type"]
        != "NOAA_GFS_NETWORK_ZERO_DECODE_RECOVERY_AUTHORIZATION"
        or authorization["status"] != "AUTHORIZED_PENDING_INDEPENDENT_GO"
        or authorization["experiment_id"]
        != "noaa_gfs_dminus2_12z_multiseason_target_free_decode_recovery_v1"
        or authorization["expected_range_rows"] != EXPECTED_ROWS
        or authorization["expected_range_bytes"] != EXPECTED_BYTES
        or authorization["max_decode_processes"] != MAX_DECODE_PROCESSES
        or authorization["network_requests_allowed"] != 0
        or authorization["raw_event_progress_mutation_allowed"] is not False
        or authorization["labels_read"] is not False
        or authorization["2024_arrays_read"] is not False
        or authorization["2025_arrays_read"] is not False
        or authorization["models_fit"] != 0
        or authorization["submission_csv_created"] is not False
        or authorization["bootstrap_trust_policy"] != BOOTSTRAP_TRUST_POLICY
        or authorization["bootstrap_import_contract"]
        != bootstrap_import_contract_report(workspace_root)
    ):
        raise RecoveryContractError("recovery authorization policy mismatch")
    attempt_id = str(authorization["recovery_attempt_id"])
    if not attempt_id.startswith("decode_recovery_v1__") or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for character in attempt_id
    ):
        raise RecoveryContractError("invalid frozen recovery attempt id")

    for key, relative in FIXED_ROOT_INPUTS.items():
        require_root_identity(authorization[key], root, relative, key)
    if str(authorization["incident"]["sha256"]) != INCIDENT_SHA256:
        raise RecoveryContractError("incident constant mismatch")
    if (
        str(authorization["sealer_entrypoint_incident"]["sha256"])
        != SEALER_ENTRYPOINT_INCIDENT_SHA256
    ):
        raise RecoveryContractError("sealer-entrypoint incident constant mismatch")
    if (
        str(authorization["original_authorization"]["sha256"])
        != ORIGINAL_AUTHORIZATION_SHA256
        or str(authorization["original_independent_go"]["sha256"])
        != ORIGINAL_GO_SHA256
    ):
        raise RecoveryContractError("original authorization/GO constant mismatch")

    incident_payload = read_json_object(
        strict_under_root(root, FIXED_ROOT_INPUTS["incident"]), "decode failure incident"
    )
    if (
        incident_payload.get("artifact_type")
        != "TRACK_A_RAW_DECODE_MEMFS_THREAD_INIT_FAILURE_INCIDENT"
        or incident_payload.get("status")
        != "RAW_DOWNLOAD_COMPLETE_DECODE_FAILED_RECOVERY_NOT_AUTHORIZED"
        or incident_payload.get("failed_attempt_id") != FAILED_ATTEMPT
        or incident_payload.get("failure_boundary")
        != {
            "raw_ranges_completed": EXPECTED_ROWS,
            "raw_bytes_completed": EXPECTED_BYTES,
            "successful_network_requests": EXPECTED_ROWS,
            "actual_http_attempts": EXPECTED_ROWS,
            "decoded_messages_completed": 0,
            "canonical_outputs_committed": 0,
            "active_lock_present_after_failure": False,
            "complete_lock_present_after_failure": False,
        }
        or incident_payload.get("recovery_gate", {}).get("currently_authorized") is not False
    ):
        raise RecoveryContractError("decode failure incident semantics mismatch")
    entrypoint_incident = read_json_object(
        strict_under_root(root, FIXED_ROOT_INPUTS["sealer_entrypoint_incident"]),
        "sealer entrypoint failure incident",
    )
    entrypoint_failure = entrypoint_incident.get("failure_boundary", {})
    entrypoint_remediation = entrypoint_incident.get("mandatory_remediation", {})
    if (
        entrypoint_incident.get("artifact_type")
        != "TRACK_A_DECODE_RECOVERY_SEAL_DIRECT_FILE_PICKLE_FAILURE_INCIDENT"
        or entrypoint_incident.get("schema_version") != 1
        or entrypoint_incident.get("status")
        != "FAILED_BEFORE_PUBLICATION_RETRY_FORBIDDEN_PENDING_REFREEZE"
        or entrypoint_failure.get("seal_publications_created") != 0
        or entrypoint_failure.get("network_requests") != 0
        or entrypoint_failure.get("recovery_launch_started") is not False
        or entrypoint_remediation
        != expected_sealer_entrypoint_remediation(workspace_root)
    ):
        raise RecoveryContractError("sealer entrypoint incident semantics mismatch")
    incident_evidence = {
        record["path"]: record
        for record in incident_payload.get("immutable_failure_evidence", [])
    }
    fixed_evidence_map = {
        FIXED_ROOT_INPUTS["original_failed_lock"]: authorization["original_failed_lock"],
        FIXED_ROOT_INPUTS["original_final_progress"]: authorization["original_final_progress"],
        FIXED_ROOT_INPUTS["original_stderr"]: authorization["original_stderr"],
        FIXED_ROOT_INPUTS["original_stdout"]: authorization["original_stdout"],
    }
    if incident_evidence != fixed_evidence_map:
        raise RecoveryContractError("authorization failure evidence differs from incident")
    failed_lock_payload = read_json_object(
        strict_under_root(root, FIXED_ROOT_INPUTS["original_failed_lock"]),
        "original failed lock",
    )
    final_progress_payload = read_json_object(
        strict_under_root(root, FIXED_ROOT_INPUTS["original_final_progress"]),
        "original final progress",
    )
    if (
        failed_lock_payload.get("attempt_id") != FAILED_ATTEMPT
        or failed_lock_payload.get("pid") != 3536
        or final_progress_payload.get("phase") != "RAW_RANGE_DOWNLOAD"
        or final_progress_payload.get("attempt_id") != FAILED_ATTEMPT
        or final_progress_payload.get("completed_ranges") != EXPECTED_ROWS
        or final_progress_payload.get("network_requests_this_invocation_so_far")
        != EXPECTED_ROWS
        or final_progress_payload.get("network_bytes_this_invocation_so_far")
        != EXPECTED_BYTES
        or final_progress_payload.get("actual_http_attempts_cumulative") != EXPECTED_ROWS
    ):
        raise RecoveryContractError("original failed lock/final progress semantics mismatch")

    original_runner_path = workspace_root / "scripts" / "run_noaa_gfs_multiseason_raw_v2.py"
    recovery_runner_path = Path(__file__).resolve()
    recovery_module_path = workspace_root / "src" / "noaa_gfs_decode_recovery.py"
    recovery_bootstrap_path = (
        workspace_root / "src" / "noaa_gfs_decode_recovery_bootstrap.py"
    )
    recovery_test_path = workspace_root / "tests" / "test_noaa_gfs_decode_recovery.py"
    recovery_bootstrap_test_path = (
        workspace_root / "tests" / "test_noaa_gfs_decode_recovery_bootstrap.py"
    )
    recovery_runner_test_path = (
        workspace_root
        / "tests"
        / "test_noaa_gfs_multiseason_decode_recovery_runner_v1.py"
    )
    postrun_auditor_path = (
        workspace_root / "scripts" / "audit_noaa_gfs_multiseason_raw_postrun_v1.py"
    )
    postrun_auditor_test_path = (
        workspace_root / "tests" / "test_noaa_gfs_multiseason_raw_postrun_v1.py"
    )
    require_external_identity(
        authorization["original_runner"], original_runner_path, "original runner"
    )
    require_external_identity(
        authorization["recovery_runner"], recovery_runner_path, "recovery runner"
    )
    require_external_identity(
        authorization["recovery_module"], recovery_module_path, "recovery module"
    )
    require_external_identity(
        authorization["recovery_bootstrap"],
        recovery_bootstrap_path,
        "recovery bootstrap",
    )
    require_external_identity(
        authorization["recovery_test"], recovery_test_path, "recovery test"
    )
    require_external_identity(
        authorization["recovery_bootstrap_test"],
        recovery_bootstrap_test_path,
        "recovery bootstrap test",
    )
    require_external_identity(
        authorization["recovery_runner_test"],
        recovery_runner_test_path,
        "recovery runner test",
    )
    require_external_identity(
        authorization["postrun_auditor"], postrun_auditor_path, "postrun auditor"
    )
    require_external_identity(
        authorization["postrun_auditor_test"],
        postrun_auditor_test_path,
        "postrun auditor test",
    )

    remnants = authorization["documented_preplan_remnants"]
    if not isinstance(remnants, list) or len(remnants) != 2:
        raise RecoveryContractError("preplan-remnant authorization mismatch")
    by_path = {str(record.get("path")): record for record in remnants}
    if set(by_path) != set(PREPLAN_REMNANTS):
        raise RecoveryContractError("preplan-remnant path set mismatch")
    for relative in PREPLAN_REMNANTS:
        require_root_identity(by_path[relative], root, relative, "preplan remnant")
    if incident_payload.get("documented_preplan_staging_remnants") != remnants:
        raise RecoveryContractError("authorization preplan remnants differ from incident")

    recovery_bootstrap = _load_recovery_bootstrap_preexecution(
        authorization["recovery_bootstrap"], workspace_root
    )
    recovery_core = _load_recovery_core_preexecution(
        authorization["recovery_module"], workspace_root
    )
    runtime_report = recovery_core.validate_eccodes_runtime(
        authorization["runtime_lock"]
    )
    if authorization["runtime_lock"]["frozen_runner"] != authorization["original_runner"]:
        raise RecoveryContractError("runtime frozen-runner binding mismatch")
    if authorization["runtime_lock"]["recovery_module"] != authorization["recovery_module"]:
        raise RecoveryContractError("runtime recovery-module binding mismatch")
    original_authorization_payload = read_json_object(
        strict_under_root(root, FIXED_ROOT_INPUTS["original_authorization"]),
        "original raw authorization",
    )
    if authorization["original_runtime_identity"] != original_authorization_payload.get(
        "runtime_identity"
    ):
        raise RecoveryContractError("original runtime identity differs from frozen authorization")
    if (
        authorization["runtime_lock"]["original_runtime_identity"]
        != authorization["original_runtime_identity"]
    ):
        raise RecoveryContractError("worker original-runtime binding mismatch")
    pilot_path = Path(str(authorization["real_grib_pilot"]["path"]))
    require_external_identity(
        authorization["real_grib_pilot"], pilot_path, "real GRIB pilot"
    )
    preflight_payload = read_json_object(
        strict_under_root(root, FIXED_ROOT_INPUTS["real_spawn_preflight"]),
        "real spawn preflight",
    )
    test_evidence = preflight_payload.get("test_evidence")
    pilot_report = preflight_payload.get("pilot_report")
    production_report = preflight_payload.get("production_worker_report")
    expected_test_identities = {
        key: authorization[key]
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
    expected_test_identities.update(
        {
            "recovery_sealer": preflight_payload.get("recovery_sealer"),
            "recovery_sealer_test": preflight_payload.get("recovery_sealer_test"),
        }
    )
    if (
        set(preflight_payload) != REAL_SPAWN_PREFLIGHT_KEYS
        or preflight_payload.get("schema_version") != 1
        or preflight_payload.get("artifact_type")
        != "TRACK_A_DECODE_RECOVERY_REAL_PROCESS_PREFLIGHT"
        or preflight_payload.get("status")
        != "PASS_REAL_PILOT_AND_PRODUCTION_PATH_EXACT_7_SPAWN_STDLIB_BOOTSTRAP"
        or preflight_payload.get("incident") != authorization["incident"]
        or preflight_payload.get("recovery_runner") != authorization["recovery_runner"]
        or preflight_payload.get("recovery_module") != authorization["recovery_module"]
        or preflight_payload.get("recovery_bootstrap")
        != authorization["recovery_bootstrap"]
        or preflight_payload.get("recovery_test") != authorization["recovery_test"]
        or preflight_payload.get("recovery_bootstrap_test")
        != authorization["recovery_bootstrap_test"]
        or preflight_payload.get("bootstrap_trust_policy")
        != authorization["bootstrap_trust_policy"]
        or preflight_payload.get("runtime_lock") != authorization["runtime_lock"]
        or preflight_payload.get("bootstrap_import_contract")
        != authorization["bootstrap_import_contract"]
        or preflight_payload.get("sealer_entrypoint_incident")
        != authorization["sealer_entrypoint_incident"]
        or preflight_payload.get("recovery_runner_test")
        != authorization["recovery_runner_test"]
        or preflight_payload.get("postrun_auditor")
        != authorization["postrun_auditor"]
        or preflight_payload.get("postrun_auditor_test")
        != authorization["postrun_auditor_test"]
        or preflight_payload.get("real_grib_pilot")
        != authorization["real_grib_pilot"]
        or preflight_payload.get("max_spawn_processes") != MAX_DECODE_PROCESSES
        or preflight_payload.get("thread_decoders") != 0
        or preflight_payload.get("network_requests") != 0
        or preflight_payload.get("production_decode_task_worker_real_test") is not True
        or preflight_payload.get("stdlib_bootstrap_worker_real_test") is not True
        or preflight_payload.get("poisoned_pyc_source_execution_test") is not True
        or preflight_payload.get("bootstrap_ast_stdlib_policy_pass") is not True
        or preflight_payload.get("python_dont_write_bytecode_env") != "1"
        or preflight_payload.get("python_dont_write_bytecode_flag") is not True
        or preflight_payload.get("python_pycache_prefix_absent") is not True
        or preflight_payload.get("bootstrap_matching_pyc_absent_before_tests") is not True
        or preflight_payload.get("bootstrap_matching_pyc_absent_after_tests") is not True
        or preflight_payload.get("all_seven_worker_pids_observed") is not True
        or preflight_payload.get("process_scheduling_flake_incident")
        != SPAWN_SCHEDULING_FLAKE_INCIDENT
        or not isinstance(pilot_report, Mapping)
        or pilot_report.get("first_wave_synchronized") is not True
        or not isinstance(pilot_report.get("worker_pids_observed"), list)
        or len(pilot_report["worker_pids_observed"])
        != MAX_DECODE_PROCESSES
        or not isinstance(production_report, Mapping)
        or production_report.get("first_wave_synchronized") is not True
        or production_report.get("worker_count") != MAX_DECODE_PROCESSES
        or not isinstance(test_evidence, Mapping)
        or set(test_evidence) != EMBEDDED_TEST_EVIDENCE_KEYS
        or test_evidence.get("schema_version") != 1
        or test_evidence.get("artifact_type")
        != "DECODE_RECOVERY_IMMUTABLE_CODE_TEST_EVIDENCE"
        or test_evidence.get("status")
        != "PASS_BOUND_CODE_COMPILE_AND_TEST_SUITE"
        or test_evidence.get("created_utc") != preflight_payload.get("created_utc")
        or test_evidence.get("bound_code_and_test_identities")
        != expected_test_identities
        or test_evidence.get("source_compile_result") != "PASS"
        or test_evidence.get("pytest_isolation")
        != "ONE_TEST_FILE_PER_CLEAN_SUBPROCESS"
        or test_evidence.get("isolated_subprocess_count") != 6
        or test_evidence.get("all_pytest_exit_codes_zero") is not True
        or test_evidence.get("pytest_child_network_guard_installed") is not True
        or test_evidence.get("pytest_cacheprovider_disabled") is not True
        or test_evidence.get("required_test_names_present") is not True
        or test_evidence.get("python_dont_write_bytecode_env") != "1"
        or test_evidence.get("python_dont_write_bytecode_flag") is not True
        or test_evidence.get("python_pycache_prefix_absent") is not True
        or test_evidence.get("bootstrap_matching_pyc_absent_before_tests") is not True
        or test_evidence.get("bootstrap_matching_pyc_absent_after_tests") is not True
        or test_evidence.get("network_requests") != 0
        or test_evidence.get("labels_read") is not False
        or test_evidence.get("arrays_2024_read") is not False
        or test_evidence.get("arrays_2025_read") is not False
        or test_evidence.get("models_fit") != 0
        or test_evidence.get("submission_csv_created") is not False
        or preflight_payload.get("focused_test_result")
        != test_evidence.get("pytest_summary")
        or preflight_payload.get("source_compile_result")
        != test_evidence.get("source_compile_result")
        or preflight_payload.get("labels_read") is not False
        or preflight_payload.get("arrays_2024_read") is not False
        or preflight_payload.get("arrays_2025_read") is not False
        or preflight_payload.get("models_fit") != 0
        or preflight_payload.get("submission_csv_created") is not False
    ):
        raise RecoveryContractError("real spawn preflight semantics mismatch")
    validate_noncontract_monolithic_diagnostic(
        test_evidence["noncontract_monolithic_diagnostic"]
    )
    _parse_exact_utc_z(
        preflight_payload.get("created_utc"), "real spawn preflight created_utc"
    )
    require_external_identity(
        preflight_payload["recovery_sealer"],
        workspace_root
        / "scripts"
        / "seal_noaa_gfs_multiseason_decode_recovery_v1.py",
        "recovery sealer",
    )
    require_external_identity(
        preflight_payload["recovery_sealer_test"],
        workspace_root
        / "tests"
        / "test_seal_noaa_gfs_multiseason_decode_recovery_v1.py",
        "recovery sealer test",
    )
    frozen = _import_frozen_runner(authorization["original_runner"], workspace_root)
    observed_original_runtime = frozen.runtime_identity()
    if observed_original_runtime != authorization["original_runtime_identity"]:
        raise RecoveryContractError("parent frozen-runner runtime identity mismatch")
    original_authorization_path = strict_under_root(
        root, FIXED_ROOT_INPUTS["original_authorization"]
    )
    original_go_path = strict_under_root(root, FIXED_ROOT_INPUTS["original_independent_go"])
    frozen.validate_independent_go(
        original_go_path,
        original_authorization_path,
        original_runner_path,
        strict_under_root(root, FIXED_ROOT_INPUTS["census_manifest"]),
        strict_under_root(root, FIXED_ROOT_INPUTS["coordinate_lock"]),
        observed_original_runtime,
    )
    if (
        authorization["census_manifest"]
        != original_authorization_payload.get("census_manifest")
        or authorization["coordinate_lock"]
        != original_authorization_payload.get("coordinate_lock")
        or authorization["field_range_census"]
        != original_authorization_payload.get("field_range_census")
    ):
        raise RecoveryContractError("recovery census/coordinate differs from original authorization")
    return authorization, runtime_report, frozen, recovery_core, recovery_bootstrap


def validate_independent_review_payload(
    review: Mapping[str, Any],
    go: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> None:
    if set(review) != INDEPENDENT_REVIEW_KEYS:
        raise RecoveryContractError("independent recovery review exact schema mismatch")
    checks = review.get("independent_checks")
    if (
        type(review.get("schema_version")) is not int
        or review["schema_version"] != 1
        or review.get("artifact_type")
        != "TRACK_A_DECODE_RECOVERY_INDEPENDENT_REVIEW"
        or review.get("status")
        != "PASS_NETWORK_ZERO_PROCESS_ISOLATED_RECOVERY_AUTHORIZED"
        or review.get("verdict") != "GO"
        or type(review.get("max_decode_processes")) is not int
        or review["max_decode_processes"] != MAX_DECODE_PROCESSES
        or type(review.get("network_requests_allowed")) is not int
        or review["network_requests_allowed"] != 0
        or not isinstance(checks, Mapping)
        or set(checks) != INDEPENDENT_REVIEW_CHECK_KEYS
        or any(checks.get(key) is not True for key in INDEPENDENT_REVIEW_CHECK_KEYS)
    ):
        raise RecoveryContractError("independent recovery review verdict/check mismatch")
    _parse_exact_utc_z(review.get("created_utc"), "independent review created_utc")
    for key in (
        "authorization",
        "incident",
        "recovery_runner",
        "recovery_module",
        "recovery_bootstrap",
        "recovery_test",
        "recovery_bootstrap_test",
        "recovery_runner_test",
        "postrun_auditor",
        "postrun_auditor_test",
        "bootstrap_trust_policy",
        "raw_cache_lock",
        "real_spawn_preflight",
    ):
        if review.get(key) != go.get(key):
            raise RecoveryContractError(f"independent recovery review {key} mismatch")
    for key in ("recovery_sealer", "recovery_sealer_test"):
        if review.get(key) != preflight.get(key):
            raise RecoveryContractError(f"independent recovery review {key} mismatch")


def validate_independent_go(
    root: Path,
    go_path: Path,
    authorization_path: Path,
    authorization: Mapping[str, Any],
    workspace_root: Path,
) -> dict[str, Any]:
    go = read_json_object(go_path, "independent recovery GO")
    expected_keys = {
        "schema_version",
        "artifact_type",
        "status",
        "created_utc",
        "authorization",
        "incident",
        "recovery_runner",
        "recovery_module",
        "recovery_bootstrap",
        "recovery_test",
        "recovery_bootstrap_test",
        "recovery_runner_test",
        "postrun_auditor",
        "postrun_auditor_test",
        "bootstrap_trust_policy",
        "raw_cache_lock",
        "real_spawn_preflight",
        "independent_review",
        "network_requests_allowed",
        "max_decode_processes",
    }
    if set(go) != expected_keys:
        raise RecoveryContractError("independent recovery GO schema mismatch")
    if (
        go["schema_version"] != 1
        or go["artifact_type"] != "TRACK_A_DECODE_RECOVERY_INDEPENDENT_GO"
        or go["status"] != "GO_NETWORK_ZERO_PROCESS_ISOLATED_DECODE_RECOVERY"
        or go["network_requests_allowed"] != 0
        or go["max_decode_processes"] != MAX_DECODE_PROCESSES
    ):
        raise RecoveryContractError("independent recovery GO policy mismatch")
    require_root_identity(
        go["authorization"],
        root,
        authorization_path.relative_to(root).as_posix(),
        "GO authorization",
    )
    if go["incident"] != authorization["incident"]:
        raise RecoveryContractError("GO incident mismatch")
    if go["bootstrap_trust_policy"] != authorization["bootstrap_trust_policy"]:
        raise RecoveryContractError("GO bootstrap trust policy mismatch")
    for key in (
        "recovery_runner",
        "recovery_module",
        "recovery_bootstrap",
        "recovery_test",
        "recovery_bootstrap_test",
        "recovery_runner_test",
        "postrun_auditor",
        "postrun_auditor_test",
    ):
        if go[key] != authorization[key]:
            raise RecoveryContractError(f"GO {key} mismatch")
    if go["raw_cache_lock"] != authorization["raw_cache_lock"]:
        raise RecoveryContractError("GO raw-cache lock mismatch")
    if go["real_spawn_preflight"] != authorization["real_spawn_preflight"]:
        raise RecoveryContractError("GO real-process preflight mismatch")
    review_path = require_root_identity(
        go["independent_review"],
        root,
        INDEPENDENT_REVIEW_RELATIVE,
        "independent recovery review",
    )
    review = read_json_object(review_path, "independent recovery review")
    preflight = read_json_object(
        strict_under_root(root, FIXED_ROOT_INPUTS["real_spawn_preflight"]),
        "independent-review-bound real spawn preflight",
    )
    validate_independent_review_payload(review, go, preflight)
    return go


def fixed_input_snapshot(
    root: Path,
    authorization_path: Path,
    go_path: Path,
    authorization: Mapping[str, Any],
    go: Mapping[str, Any],
) -> dict[str, Any]:
    validate_bootstrap_preexecution(
        authorization["recovery_bootstrap"],
        Path(__file__).resolve().parents[1]
        / "src"
        / "noaa_gfs_decode_recovery_bootstrap.py",
    )
    root_records = [
        root_identity(authorization_path, root),
        root_identity(go_path, root),
    ]
    for key, relative in sorted(FIXED_ROOT_INPUTS.items()):
        path = require_root_identity(authorization[key], root, relative, key)
        root_records.append(root_identity(path, root))
    for record in authorization["documented_preplan_remnants"]:
        path = require_root_identity(
            record, root, str(record["path"]), "fixed preplan remnant"
        )
        root_records.append(root_identity(path, root))
    review_path = require_root_identity(
        go["independent_review"], root, INDEPENDENT_REVIEW_RELATIVE, "fixed review"
    )
    root_records.append(root_identity(review_path, root))

    external_records: list[dict[str, Any]] = []
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
        "real_grib_pilot",
    ):
        path = _require_exact_file(authorization[key], f"snapshot {key}")
        external_records.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    runtime = authorization["runtime_lock"]
    for key in (
        "library",
        "memfs_library",
        "python_binding",
        "gribapi_binding",
        "eccodes_python_api",
        "gribapi_python_api",
        "gribapi_errors",
    ):
        path = _require_exact_file(runtime[key], f"snapshot runtime {key}")
        external_records.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    original_runtime = authorization["original_runtime_identity"]
    executable = _require_exact_file(
        {
            "path": original_runtime["python_executable"],
            "size_bytes": original_runtime["python_executable_size_bytes"],
            "sha256": original_runtime["python_executable_sha256"],
        },
        "snapshot Python executable",
    )
    external_records.append(
        {
            "path": str(executable),
            "size_bytes": executable.stat().st_size,
            "sha256": sha256_file(executable),
        }
    )
    for name, record in sorted(original_runtime["packages"].items()):
        path = _require_exact_file(
            {
                "path": record["module_file"],
                "size_bytes": record["module_file_size_bytes"],
                "sha256": record["module_file_sha256"],
            },
            f"snapshot original runtime package {name}",
        )
        external_records.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    root_records = sorted(root_records, key=lambda row: row["path"])
    external_records = sorted(external_records, key=lambda row: row["path"])
    return {
        "root_file_count": len(root_records),
        "root_files_sha256": canonical_json_sha256(root_records),
        "external_file_count": len(external_records),
        "external_files_sha256": canonical_json_sha256(external_records),
        "combined_sha256": canonical_json_sha256(
            {"root": root_records, "external": external_records}
        ),
    }


def read_range_rows(
    range_path: Path, expected_identity: Mapping[str, Any]
) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    if range_path.stat().st_size != int(expected_identity["size_bytes"]):
        raise RecoveryContractError("field census size changed before schema read")
    if sha256_file(range_path) != str(expected_identity["sha256"]):
        raise RecoveryContractError("field census hash changed before schema read")
    parquet = pq.ParquetFile(range_path)
    if tuple(parquet.schema_arrow.names) != FIELD_RANGE_COLUMNS:
        raise RecoveryContractError("field census exact schema mismatch")
    if parquet.metadata.num_rows != EXPECTED_ROWS:
        raise RecoveryContractError("field census metadata row count mismatch")
    table = parquet.read(columns=list(FIELD_RANGE_COLUMNS))
    rows = table.to_pylist()
    if (
        len(rows) != EXPECTED_ROWS
        or sum(int(row["range_bytes"]) for row in rows) != EXPECTED_BYTES
        or any(row["status"] != "CENSUS_VERIFIED" for row in rows)
    ):
        raise RecoveryContractError("field census value invariants failed")
    return rows


def _parse_exact_utc_z(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z") or value.strip() != value:
        raise RecoveryContractError(f"{label} is not an exact UTC-Z timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise RecoveryContractError(f"{label} is not a valid timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise RecoveryContractError(f"{label} is not UTC")
    return parsed


def _plain_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise RecoveryContractError(f"{label} must be a plain integer")
    return value


def _read_original_start_attempts(
    root: Path, expected_event_files: set[Path]
) -> list[dict[str, Any]]:
    start_files = sorted(
        (path for path in expected_event_files if path.name.endswith("_start.json")),
        key=lambda path: path.name,
    )
    if len(start_files) != EXPECTED_ROWS:
        raise RecoveryContractError("original START-event count mismatch")
    attempts: list[dict[str, Any]] = []
    for path in start_files:
        payload = read_json_object(path, "original START event")
        if set(payload) != ORIGINAL_START_EVENT_KEYS:
            raise RecoveryContractError("original START-event schema mismatch")
        event_id = payload.get("event_id")
        attempt = _plain_int(payload.get("attempt"), "START-event attempt")
        global_attempt = _plain_int(
            payload.get("global_raw_attempt_number"),
            "START-event global attempt",
        )
        if (
            payload.get("event") != "RANGE_REQUEST_START"
            or not isinstance(event_id, str)
            or len(event_id) != 32
            or any(character not in "0123456789abcdef" for character in event_id)
            or attempt != 1
            or path.name != f"{event_id}__attempt_{attempt:03d}_start.json"
            or global_attempt < 1
            or not isinstance(payload.get("url"), str)
            or not str(payload["url"]).startswith("https://noaa-gfs-bdp-pds.s3.amazonaws.com/")
            or _plain_int(payload.get("range_start"), "START-event range start") < 0
            or _plain_int(payload.get("range_end"), "START-event range end")
            < int(payload["range_start"])
        ):
            raise RecoveryContractError("original START-event semantics mismatch")
        created = _parse_exact_utc_z(
            payload.get("created_utc"), "START-event created_utc"
        )
        attempts.append(
            {
                "path": path.relative_to(root).as_posix(),
                "global_raw_attempt_number": global_attempt,
                "created_utc": payload["created_utc"],
                "created": created,
            }
        )
    if sorted(record["global_raw_attempt_number"] for record in attempts) != list(
        range(1, EXPECTED_ROWS + 1)
    ):
        raise RecoveryContractError("original START global attempts are not exact 1..10368")
    return attempts


def _validate_original_progress_payloads(
    rows: Sequence[Mapping[str, Any]],
    progress_records: Sequence[Mapping[str, Any]],
    failed_lock_payload: Mapping[str, Any],
    start_attempts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Independently reconstruct every original producer progress checkpoint."""

    if len(rows) != EXPECTED_ROWS:
        raise RecoveryContractError("progress validation field-row count mismatch")
    if len(progress_records) != len(ORIGINAL_PROGRESS_CHECKPOINTS):
        raise RecoveryContractError("original progress checkpoint count mismatch")
    if set(failed_lock_payload) != ORIGINAL_FAILED_LOCK_KEYS:
        raise RecoveryContractError("original failed-lock exact schema mismatch")
    if (
        failed_lock_payload.get("attempt_id") != FAILED_ATTEMPT
        or _plain_int(failed_lock_payload.get("pid"), "failed-lock pid")
        != FAILED_ATTEMPT_PID
    ):
        raise RecoveryContractError("original failed-lock semantics mismatch")
    failed_lock_created = _parse_exact_utc_z(
        failed_lock_payload.get("created_utc"), "failed-lock created_utc"
    )

    if len(start_attempts) != EXPECTED_ROWS:
        raise RecoveryContractError("progress START-attempt evidence count mismatch")
    parsed_starts: list[tuple[datetime, int]] = []
    for record in start_attempts:
        global_attempt = _plain_int(
            record.get("global_raw_attempt_number"), "START global attempt evidence"
        )
        created_value = record.get("created_utc")
        created = record.get("created")
        if not isinstance(created, datetime):
            created = _parse_exact_utc_z(created_value, "START created evidence")
        elif created_value is not None and created != _parse_exact_utc_z(
            created_value, "START created evidence"
        ):
            raise RecoveryContractError("START timestamp evidence disagrees")
        parsed_starts.append((created, global_attempt))
    if sorted(global_attempt for _created, global_attempt in parsed_starts) != list(
        range(1, EXPECTED_ROWS + 1)
    ):
        raise RecoveryContractError("progress START-attempt sequence mismatch")
    if min(created for created, _global in parsed_starts) < failed_lock_created:
        raise RecoveryContractError("START event predates original launch lock")

    row_keys: list[str] = []
    cumulative_bytes: list[int] = [0]
    for row in rows:
        key = "|".join(
            (
                str(row["object_key"]),
                str(_plain_int(row["forecast_hour"], "field forecast hour")),
                str(row["variable"]),
                str(row["level"]),
            )
        )
        row_keys.append(key)
        range_bytes = _plain_int(row["range_bytes"], "field range bytes")
        if range_bytes <= 0:
            raise RecoveryContractError("field range bytes must be positive")
        cumulative_bytes.append(cumulative_bytes[-1] + range_bytes)
    if len(set(row_keys)) != EXPECTED_ROWS or cumulative_bytes[-1] != EXPECTED_BYTES:
        raise RecoveryContractError("progress field key/byte closure mismatch")

    prior_created: datetime | None = None
    prior_attempts = 0
    payload_sequence: list[dict[str, Any]] = []
    filename_sequence: list[str] = []
    for expected_completed, record in zip(
        ORIGINAL_PROGRESS_CHECKPOINTS, progress_records, strict=True
    ):
        expected_checkpoint = (
            f"raw_ranges__{FAILED_ATTEMPT}__{expected_completed:06d}"
        )
        expected_filename = f"{expected_checkpoint}.json"
        if set(record) != {"filename", "payload"}:
            raise RecoveryContractError("progress record wrapper schema mismatch")
        if record.get("filename") != expected_filename:
            raise RecoveryContractError("original progress filename sequence mismatch")
        payload = record.get("payload")
        if not isinstance(payload, dict) or set(payload) != ORIGINAL_PROGRESS_KEYS:
            raise RecoveryContractError("original progress exact payload schema mismatch")
        completed = _plain_int(payload.get("completed_ranges"), "completed ranges")
        attempts = _plain_int(
            payload.get("actual_http_attempts_cumulative"),
            "cumulative HTTP attempts",
        )
        requests = _plain_int(
            payload.get("network_requests_this_invocation_so_far"),
            "cumulative network requests",
        )
        network_bytes = _plain_int(
            payload.get("network_bytes_this_invocation_so_far"),
            "cumulative network bytes",
        )
        created = _parse_exact_utc_z(
            payload.get("created_utc"), "progress created_utc"
        )
        reconstructed_attempts = sum(
            start_created <= created for start_created, _global in parsed_starts
        )
        expected_digest = hashlib.sha256(
            ("\n".join(sorted(row_keys[:expected_completed])) + "\n").encode(
                "utf-8"
            )
        ).hexdigest()
        if (
            payload.get("phase") != "RAW_RANGE_DOWNLOAD"
            or payload.get("attempt_id") != FAILED_ATTEMPT
            or payload.get("checkpoint_name") != expected_checkpoint
            or completed != expected_completed
            or requests != expected_completed
            or network_bytes != cumulative_bytes[expected_completed]
            or payload.get("completed_key_set_sha256") != expected_digest
            or attempts != reconstructed_attempts
            or attempts < completed
            or attempts > min(
                EXPECTED_ROWS, completed + ORIGINAL_PRODUCER_MAX_WORKERS - 1
            )
            or attempts < prior_attempts
            or created <= failed_lock_created
            or (prior_created is not None and created <= prior_created)
        ):
            raise RecoveryContractError(
                f"original progress semantics mismatch at {expected_completed}"
            )
        prior_created = created
        prior_attempts = attempts
        filename_sequence.append(expected_filename)
        payload_sequence.append(payload)

    final_payload = payload_sequence[-1]
    if (
        final_payload["actual_http_attempts_cumulative"] != EXPECTED_ROWS
        or final_payload["network_requests_this_invocation_so_far"] != EXPECTED_ROWS
        or final_payload["network_bytes_this_invocation_so_far"] != EXPECTED_BYTES
        or sum(
            start_created <= prior_created
            for start_created, _global in parsed_starts
        )
        != EXPECTED_ROWS
    ):
        raise RecoveryContractError("original final progress closure mismatch")
    return {
        "progress_checkpoint_count": len(payload_sequence),
        "checkpoint_completed_sequence_sha256": canonical_json_sha256(
            list(ORIGINAL_PROGRESS_CHECKPOINTS)
        ),
        "checkpoint_filename_sequence_sha256": canonical_json_sha256(
            filename_sequence
        ),
        "progress_payload_sequence_sha256": canonical_json_sha256(payload_sequence),
        "first_checkpoint_created_utc": payload_sequence[0]["created_utc"],
        "final_checkpoint_created_utc": final_payload["created_utc"],
        "final_completed_ranges": EXPECTED_ROWS,
        "final_network_requests": EXPECTED_ROWS,
        "final_network_bytes": EXPECTED_BYTES,
        "final_actual_http_attempts": EXPECTED_ROWS,
        "final_completed_key_set_sha256": final_payload[
            "completed_key_set_sha256"
        ],
        "start_event_count": EXPECTED_ROWS,
        "start_global_attempts_exact_1_through_10368": True,
        "attempt_count_reconstructed_from_start_event_timestamps": True,
        "failed_lock_payload_sha256": canonical_json_sha256(failed_lock_payload),
        "failed_lock_attempt_id": FAILED_ATTEMPT,
        "failed_lock_pid": FAILED_ATTEMPT_PID,
    }


def validate_original_progress_and_failed_lock(
    root: Path,
    rows: Sequence[Mapping[str, Any]],
    expected_event_files: set[Path],
    progress_files: Sequence[Path],
    launch_files: Sequence[Path],
) -> dict[str, Any]:
    expected_names = [
        f"raw_ranges__{FAILED_ATTEMPT}__{completed:06d}.json"
        for completed in ORIGINAL_PROGRESS_CHECKPOINTS
    ]
    ordered_progress = sorted(progress_files, key=lambda path: path.name)
    if [path.name for path in ordered_progress] != expected_names:
        raise RecoveryContractError("original progress exact file inventory mismatch")
    expected_failed_name = f"{FAILED_ATTEMPT}__failed.lock"
    if len(launch_files) != 1 or launch_files[0].name != expected_failed_name:
        raise RecoveryContractError("original launch-history inventory mismatch")
    progress_records = [
        {
            "filename": path.name,
            "payload": read_json_object(path, "original progress checkpoint"),
        }
        for path in ordered_progress
    ]
    failed_lock_payload = read_json_object(
        launch_files[0], "original failed launch lock"
    )
    start_attempts = _read_original_start_attempts(root, expected_event_files)
    return _validate_original_progress_payloads(
        rows, progress_records, failed_lock_payload, start_attempts
    )


def validate_raw_cache(
    root: Path,
    rows: Sequence[Mapping[str, Any]],
    frozen: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected_range_files: set[Path] = set()
    expected_event_files: set[Path] = set()
    raw_manifest_rows: list[dict[str, Any]] = []
    raw_bytes = 0
    for row in rows:
        path = frozen.raw_path_for(root, row)
        sidecar = path.with_suffix(".grib2.meta.json")
        if path.is_symlink() or sidecar.is_symlink():
            raise RecoveryContractError("raw or sidecar symlink is forbidden")
        if not path.is_file() or not sidecar.is_file():
            raise RecoveryContractError(f"raw/sidecar pair absent: {path}")
        meta = read_json_object(sidecar, "raw sidecar")
        frozen.validate_final_raw(path, meta, row)
        expected_range_files.update((path, sidecar))
        evidence = meta.get("request_completion_evidence")
        if not isinstance(evidence, dict) or set(evidence) != {
            "selected_complete_event",
            "selected_start_event",
            "semantically_identical_completion_event_count",
        }:
            raise RecoveryContractError("raw sidecar request evidence mismatch")
        if evidence["semantically_identical_completion_event_count"] != 1:
            raise RecoveryContractError("unexpected multiple completion evidence")
        for event_key in ("selected_start_event", "selected_complete_event"):
            event_record = evidence[event_key]
            event_path = require_root_identity(
                event_record,
                root,
                str(event_record.get("path")),
                f"sidecar {event_key}",
            )
            expected_event_files.add(event_path)
        if "resume_prefix_evidence" in meta or int(meta.get("resumed_from_bytes", -1)) != 0:
            raise RecoveryContractError("unexpected resumed raw in exact no-retry cache")
        raw_manifest_rows.append(meta)
        raw_bytes += int(meta["raw_size_bytes"])
    if raw_bytes != EXPECTED_BYTES:
        raise RecoveryContractError("validated raw byte total mismatch")

    actual_range_files = set(exact_recursive_files(root / "raw" / "ranges"))
    if actual_range_files != expected_range_files:
        raise RecoveryContractError("raw ranges filesystem inventory mismatch")
    ranges_root = root / "raw" / "ranges"
    expected_range_dirs: set[Path] = set()
    for path in expected_range_files:
        parent = path.parent
        while parent != ranges_root:
            expected_range_dirs.add(parent)
            parent = parent.parent
    if set(exact_recursive_dirs(ranges_root)) != expected_range_dirs:
        raise RecoveryContractError("raw ranges directory inventory mismatch")
    actual_event_files = set(exact_recursive_files(root / "raw" / "request_events"))
    if actual_event_files != expected_event_files:
        raise RecoveryContractError("request-event filesystem inventory mismatch")
    if exact_recursive_dirs(root / "raw" / "request_events"):
        raise RecoveryContractError("unexpected request-event subdirectory")

    progress_files = exact_recursive_files(root / "raw" / "progress")
    launch_files = exact_recursive_files(root / "raw" / "launch_history")
    if exact_recursive_dirs(root / "raw" / "progress") or exact_recursive_dirs(
        root / "raw" / "launch_history"
    ):
        raise RecoveryContractError("unexpected progress/launch-history subdirectory")
    progress_and_failed_lock = validate_original_progress_and_failed_lock(
        root,
        rows,
        expected_event_files,
        progress_files,
        launch_files,
    )
    inventories = {
        "raw_ranges_and_sidecars": file_inventory(list(expected_range_files), root),
        "request_events": file_inventory(list(expected_event_files), root),
        "original_progress": file_inventory(progress_files, root),
        "original_launch_history": file_inventory(launch_files, root),
        "original_progress_and_failed_lock_semantics": progress_and_failed_lock,
        "validated_raw_payload_bytes": raw_bytes,
        "raw_file_count": EXPECTED_ROWS,
        "sidecar_file_count": EXPECTED_ROWS,
        "request_event_file_count": EXPECTED_ROWS * 2,
    }
    return raw_manifest_rows, inventories


def validate_raw_cache_lock(
    root: Path,
    authorization: Mapping[str, Any],
    inventories: Mapping[str, Any],
    prerecovery_topology: Mapping[str, Any],
) -> dict[str, Any]:
    lock_path = strict_under_root(root, FIXED_ROOT_INPUTS["raw_cache_lock"])
    lock = read_json_object(lock_path, "raw-cache preflight lock")
    expected_keys = {
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
    if set(lock) != expected_keys:
        raise RecoveryContractError("raw-cache preflight lock schema mismatch")
    if (
        lock["schema_version"] != 1
        or lock["artifact_type"] != "TRACK_A_DECODE_RECOVERY_RAW_CACHE_LOCK"
        or lock["status"] != "PASS_IMMUTABLE_RAW_CACHE_READY_FOR_DECODE_ONLY"
        or lock["incident"] != authorization["incident"]
        or lock["inventories"] != inventories
        or lock["prerecovery_topology"] != prerecovery_topology
        or lock["documented_preplan_remnants"]
        != authorization["documented_preplan_remnants"]
        or lock["canonical_outputs_absent"] is not True
        or lock["network_requests_during_preflight"] != 0
        or lock["raw_event_progress_mutations"] != 0
        or lock["labels_read"] is not False
        or lock["2024_arrays_read"] is not False
        or lock["2025_arrays_read"] is not False
    ):
        raise RecoveryContractError("raw-cache preflight lock value mismatch")
    return lock


def require_canonical_outputs_absent(root: Path) -> dict[str, Path]:
    outputs = canonical_output_paths(root)
    conflicts = [str(path) for path in outputs.values() if path.exists()]
    if conflicts:
        raise RecoveryContractError(f"canonical outputs already exist: {conflicts}")
    return outputs


def canonical_output_paths(root: Path) -> dict[str, Path]:
    return {
        name: strict_under_root(root, relative)
        for name, relative in CANONICAL_OUTPUTS.items()
    }


class RecoveryClaim:
    def __init__(self, root: Path, attempt_id: str):
        self.root = root
        self.attempt_id = attempt_id
        # Use the producer's canonical raw mutex.  A separate decoded lock
        # would not prevent the original downloader from mutating raw/events.
        self.raw_active = root / "raw" / "RAW_LAUNCH_ACTIVE.lock"
        self.decoded_active = root / "decoded" / "DECODE_RECOVERY_ACTIVE.lock"
        self.closed = False
        self.payload: dict[str, Any] | None = None

    @classmethod
    def acquire(
        cls,
        root: Path,
        attempt_id: str,
        frozen: Any,
    ) -> "RecoveryClaim":
        claim = cls(root, attempt_id)
        if claim.raw_active.exists() or claim.decoded_active.exists():
            raise RecoveryContractError("decode recovery/shared raw active lock already exists")
        history = root / "decoded" / "recovery_history"
        if history.exists() and exact_recursive_files(history):
            raise RecoveryContractError("a prior decode recovery attempt already exists")
        claim.payload = {
            "artifact_type": "DECODE_RECOVERY_ACTIVE_LOCK",
            "attempt_id": attempt_id,
            "pid": os.getpid(),
            "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "network_requests": 0,
        }
        frozen.write_json_exclusive_fsync(claim.raw_active, claim.payload)
        try:
            frozen.write_json_exclusive_fsync(claim.decoded_active, claim.payload)
        except Exception:
            # Remove only the lock created by this failed two-lock acquisition.
            expected = (
                json.dumps(
                    claim.payload, ensure_ascii=False, indent=2, sort_keys=True
                )
                + "\n"
            ).encode("utf-8")
            if (
                claim.raw_active.is_symlink()
                or not claim.raw_active.is_file()
                or claim.raw_active.read_bytes() != expected
            ):
                raise RecoveryContractError(
                    "cannot safely clean up first recovery mutex after second-lock failure"
                )
            claim.raw_active.unlink()
            raise
        return claim

    def close(self, verdict: str) -> Path:
        if self.closed:
            raise RecoveryContractError("decode recovery claim already closed")
        if verdict not in {"complete", "failed"}:
            raise ValueError("invalid recovery verdict")
        history = (
            self.root
            / "decoded"
            / "recovery_history"
        )
        history.mkdir(parents=True, exist_ok=True)
        if self.payload is None:
            raise RecoveryContractError("recovery active payload is absent")
        destinations = {
            "raw_mutex": history
            / f"{self.attempt_id}__raw_mutex__{verdict}.lock",
            "decode_mutex": history
            / f"{self.attempt_id}__decode_mutex__{verdict}.lock",
        }
        sources = {
            "raw_mutex": self.raw_active,
            "decode_mutex": self.decoded_active,
        }
        expected_bytes = (
            json.dumps(self.payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        for name, source in sources.items():
            destination = destinations[name]
            if (
                source.is_symlink()
                or (
                    hasattr(source, "is_junction") and source.is_junction()
                )
                or
                not source.is_file()
                or source.read_bytes() != expected_bytes
                or destination.is_symlink()
                or (
                    hasattr(destination, "is_junction")
                    and destination.is_junction()
                )
                or destination.exists()
            ):
                raise RecoveryContractError("recovery lock close state/payload mismatch")
            try:
                os.link(source, destination, follow_symlinks=False)
            except FileExistsError as exc:
                raise RecoveryContractError("recovery history lock race detected") from exc
            if not os.path.samefile(source, destination):
                raise RecoveryContractError("recovery history lock link mismatch")
        for source in sources.values():
            source.unlink()
        self.closed = True
        # The raw-mutex history record is the canonical completion identity;
        # both linked records are byte-identical and independently audited.
        return destinations["raw_mutex"]


def validate_locked_prerecovery_topology(
    root: Path, claim: RecoveryClaim
) -> dict[str, Any]:
    """Close the full prestate only after both producer-visible mutexes exist."""

    raw_root = root / "raw"
    expected_raw_dirs = {
        "launch_history",
        "output_transactions",
        "progress",
        "ranges",
        "request_events",
    }
    expected_raw_entries = expected_raw_dirs | {"RAW_LAUNCH_ACTIVE.lock"}
    if (
        raw_root.is_symlink()
        or (hasattr(raw_root, "is_junction") and raw_root.is_junction())
        or not raw_root.is_dir()
        or {path.name for path in raw_root.iterdir()} != expected_raw_entries
    ):
        raise RecoveryContractError("locked raw top-level inventory mismatch")
    if any(
        not (raw_root / name).is_dir()
        or (raw_root / name).is_symlink()
        or (
            hasattr(raw_root / name, "is_junction")
            and (raw_root / name).is_junction()
        )
        for name in expected_raw_dirs
    ):
        raise RecoveryContractError("locked raw directory topology mismatch")

    if claim.payload is None:
        raise RecoveryContractError("locked prestate claim payload absent")
    expected_lock_bytes = (
        json.dumps(claim.payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    for path in (claim.raw_active, claim.decoded_active):
        if (
            path.is_symlink()
            or (hasattr(path, "is_junction") and path.is_junction())
            or not path.is_file()
            or path.read_bytes() != expected_lock_bytes
        ):
            raise RecoveryContractError("locked prestate mutex identity mismatch")

    decoded_root = root / "decoded"
    if (
        decoded_root.is_symlink()
        or (hasattr(decoded_root, "is_junction") and decoded_root.is_junction())
        or not decoded_root.is_dir()
        or {path.name for path in decoded_root.iterdir()}
        != {"DECODE_RECOVERY_ACTIVE.lock"}
    ):
        raise RecoveryContractError("locked decoded-tree inventory mismatch")

    transaction_root = raw_root / "output_transactions"
    expected_transaction_files = {
        strict_under_root(root, relative) for relative in PREPLAN_REMNANTS
    }
    if set(exact_recursive_files(transaction_root)) != expected_transaction_files:
        raise RecoveryContractError("locked transaction file closure mismatch")
    expected_transaction_dirs: set[Path] = set()
    for path in expected_transaction_files:
        parent = path.parent
        while parent != transaction_root:
            expected_transaction_dirs.add(parent)
            parent = parent.parent
    if set(exact_recursive_dirs(transaction_root)) != expected_transaction_dirs:
        raise RecoveryContractError("locked transaction directory closure mismatch")
    require_canonical_outputs_absent(root)
    return {
        "raw_top_level_directories": sorted(expected_raw_dirs),
        "documented_preplan_transactions": file_inventory(
            list(expected_transaction_files), root
        ),
        "raw_mutex": root_identity(claim.raw_active, root),
        "decoded_mutex": root_identity(claim.decoded_active, root),
        "mutex_payload": dict(claim.payload),
        "canonical_outputs_absent": True,
    }


def assemble_matrices(
    rows: Sequence[Mapping[str, Any]],
    decoded_rows: Sequence[Mapping[str, Any]],
    sites: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if len(rows) != EXPECTED_ROWS or len(decoded_rows) != EXPECTED_ROWS:
        raise RecoveryContractError("decoded message count mismatch")
    accumulator: dict[tuple[str, str, int], dict[str, Any]] = {}
    worker_pids: set[int] = set()
    for row, decoded in zip(rows, decoded_rows):
        runtime = decoded.get("worker_runtime")
        if not isinstance(runtime, dict):
            raise RecoveryContractError("decode worker runtime evidence absent")
        worker_pids.add(int(runtime["pid"]))
        values = decoded.get("site_values")
        if not isinstance(values, list) or len(values) != 17:
            raise RecoveryContractError("decoded site-value count mismatch")
        feature = str(decoded.get("feature"))
        if feature not in FEATURES:
            raise RecoveryContractError("decoded feature outside frozen set")
        for site, value in zip(sites, values):
            key = (
                str(row["valid_time_utc"]),
                str(row["target_operating_day_kst"]),
                int(site["site_id"]),
            )
            target = accumulator.setdefault(
                key,
                {
                    "valid_time_utc": row["valid_time_utc"],
                    "target_operating_day_kst": row["target_operating_day_kst"],
                    "run_init_utc": row["run_init_utc"],
                    "forecast_hour": int(row["forecast_hour"]),
                    "site_id": int(site["site_id"]),
                    "group": str(site["group"]),
                    "latitude": float(site["latitude"]),
                    "longitude": float(site["longitude"]),
                    "capacity_mw": float(site["capacity_mw"]),
                },
            )
            if feature in target:
                raise RecoveryContractError(f"duplicate decoded feature: {key}/{feature}")
            target[feature] = float(value)
    site_rows = sorted(accumulator.values(), key=lambda row: (row["valid_time_utc"], row["site_id"]))
    if len(site_rows) != 1_152 * 17 or any(
        any(feature not in row for feature in FEATURES) for row in site_rows
    ):
        raise RecoveryContractError("decoded site matrix coverage mismatch")

    by_time_group: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in site_rows:
        by_time_group.setdefault((str(row["valid_time_utc"]), str(row["group"])), []).append(row)
    group_rows: list[dict[str, Any]] = []
    for (valid, group), members in sorted(by_time_group.items()):
        total_capacity = sum(float(row["capacity_mw"]) for row in members)
        output = {
            "valid_time_utc": valid,
            "target_operating_day_kst": members[0]["target_operating_day_kst"],
            "run_init_utc": members[0]["run_init_utc"],
            "forecast_hour": members[0]["forecast_hour"],
            "group": group,
            "site_count": len(members),
            "capacity_mw": total_capacity,
        }
        for feature in FEATURES:
            output[feature] = sum(
                float(row[feature]) * float(row["capacity_mw"]) for row in members
            ) / total_capacity
        group_rows.append(output)
    if len(group_rows) != 1_152 * 3:
        raise RecoveryContractError("decoded group matrix coverage mismatch")
    counts = {"kpx_group_1": 6, "kpx_group_2": 6, "kpx_group_3": 5}
    capacities = {"kpx_group_1": 21.6, "kpx_group_2": 21.6, "kpx_group_3": 21.0}
    for row in group_rows:
        group = str(row["group"])
        if int(row["site_count"]) != counts[group] or abs(
            float(row["capacity_mw"]) - capacities[group]
        ) > 1e-12:
            raise RecoveryContractError("group membership/capacity mismatch")

    import numpy as np

    hpbl = np.asarray([row["HPBL_surface"] for row in site_rows], dtype=np.float64)
    vertical = np.asarray(
        [[row[feature] for feature in FEATURES[1:]] for row in site_rows],
        dtype=np.float64,
    )
    pbl_pass = bool(np.isfinite(hpbl).all() and np.all((hpbl >= 0.0) & (hpbl <= 10_000.0)))
    vertical_pass = bool(np.isfinite(vertical).all() and np.all(np.abs(vertical) <= 150.0))
    audit = {
        "expected_site_rows": 1_152 * 17,
        "observed_site_rows": len(site_rows),
        "expected_group_rows": 1_152 * 3,
        "observed_group_rows": len(group_rows),
        "duplicate_site_keys": len(site_rows)
        - len({(row["valid_time_utc"], row["site_id"]) for row in site_rows}),
        "duplicate_group_keys": len(group_rows)
        - len({(row["valid_time_utc"], row["group"]) for row in group_rows}),
        "decode_process_model": "WINDOWS_SPAWN_PROCESS_ISOLATION_SERIAL_ECCODES_PER_PROCESS",
        "decode_worker_pids": sorted(worker_pids),
        "decode_process_count": len(worker_pids),
        "families": {
            "PBL_HEIGHT": {
                "finite_fraction": float(np.isfinite(hpbl).mean()),
                "minimum": float(np.nanmin(hpbl)),
                "maximum": float(np.nanmax(hpbl)),
                "physical_gate_pass": pbl_pass,
                "status": "PASS" if pbl_pass else "REJECT_PBL_HEIGHT_ONLY",
            },
            "LOW_LEVEL_ISOBARIC_WIND_PROFILE": {
                "finite_fraction": float(np.isfinite(vertical).mean()),
                "maximum_absolute_component_mps": float(np.nanmax(np.abs(vertical))),
                "physical_gate_pass": vertical_pass,
                "status": "PASS" if vertical_pass else "REJECT_LOW_LEVEL_ISOBARIC_WIND_PROFILE_ONLY",
            },
        },
        "independent_family_salvage_applied_exactly": True,
        "labels_read": False,
        "models_fit": 0,
    }
    return site_rows, group_rows, audit


def write_progress(root: Path, attempt_id: str, completed: int, frozen: Any) -> Path:
    path = (
        root
        / "decoded"
        / "recovery_progress"
        / f"decoded_messages__{attempt_id}__{completed:06d}.json"
    )
    frozen.write_json_exclusive_fsync(
        path,
        {
            "artifact_type": "DECODE_RECOVERY_PROGRESS",
            "attempt_id": attempt_id,
            "completed_messages": completed,
            "network_requests": 0,
            "process_model": "spawn",
            "max_processes": MAX_DECODE_PROCESSES,
        },
    )
    return path


def publish_json_create_if_absent(
    path: Path, payload: Mapping[str, Any], frozen: Any
) -> None:
    """Durably publish JSON without any overwrite-capable filesystem call.

    The adjacent temporary file is fsynced first, then hard-linked into the
    canonical name.  ``os.link`` fails atomically if a concurrent writer has
    created that name.  Any interruption in this transaction is terminal for
    this authorization and requires a new incident plus exact-plan auth/GO;
    this runner intentionally has no generic transaction-resume mode.
    """

    encoded = (
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    token = hashlib.sha256(encoded).hexdigest()[:20]
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}.{token}")
    if path.exists() or temporary.exists():
        raise RecoveryContractError(f"exclusive JSON destination already exists: {path}")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise RecoveryContractError(
                f"exclusive JSON publication race detected: {path}"
            ) from exc
        if not os.path.samefile(temporary, path):
            raise RecoveryContractError("exclusive JSON hard-link identity mismatch")
        frozen.fsync_file(path)
    finally:
        if temporary.is_file() and not temporary.is_symlink():
            temporary.unlink()


def write_recovery_transaction_plan(
    root: Path,
    attempt_id: str,
    staged: Mapping[str, Path],
    outputs: Mapping[str, Path],
    authorization_record: Mapping[str, Any],
    go_record: Mapping[str, Any],
    incident_record: Mapping[str, Any],
    frozen: Any,
) -> Path:
    if set(staged) != set(outputs):
        raise RecoveryContractError("recovery transaction key mismatch")
    transaction_root = root / "raw" / "output_transactions"
    plan = transaction_root / f"{attempt_id}__plan.json"
    commit = transaction_root / f"{attempt_id}__committed.json"
    if plan.exists() or commit.exists():
        raise RecoveryContractError("recovery transaction identity already exists")
    items = []
    for name in sorted(outputs):
        source = staged[name]
        destination = outputs[name]
        if source.is_symlink() or not source.is_file() or destination.exists():
            raise RecoveryContractError(f"invalid recovery staging state: {name}")
        items.append(
            {
                "name": name,
                "staged_path": source.relative_to(root).as_posix(),
                "destination_path": destination.relative_to(root).as_posix(),
                "size_bytes": source.stat().st_size,
                "sha256": sha256_file(source),
            }
        )
    publish_json_create_if_absent(
        plan,
        {
            "artifact_type": "DECODE_RECOVERY_OUTPUT_TRANSACTION_PLAN",
            "schema_version": 1,
            "attempt_id": attempt_id,
            "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "recovery_authorization": dict(authorization_record),
            "independent_recovery_go": dict(go_record),
            "incident": dict(incident_record),
            "items": items,
            "canonical_outputs_absent_before_plan": True,
            "all_10368_messages_decoded_before_plan": True,
            "physical_audit_completed_before_plan": True,
            "network_requests": 0,
        },
        frozen,
    )
    return plan


def commit_recovery_transaction_fail_if_exists(
    root: Path,
    plan_path: Path,
    outputs: Mapping[str, Path],
    frozen: Any,
    *,
    expected_plan_identity: Mapping[str, Any],
    expected_authorization: Mapping[str, Any],
    expected_go: Mapping[str, Any],
    expected_incident: Mapping[str, Any],
) -> Path:
    """Atomically create destinations without overwrite semantics.

    ``os.replace`` is forbidden here because a destination injected after the
    prior absence check would be overwritten.  A same-filesystem hard link is
    an atomic create-if-absent operation on Windows and POSIX.  Any race raises
    ``FileExistsError`` and is terminal for this invocation.
    """

    plan = read_json_object(plan_path, "decode recovery transaction plan")
    expected_plan_keys = {
        "artifact_type",
        "schema_version",
        "attempt_id",
        "created_utc",
        "recovery_authorization",
        "independent_recovery_go",
        "incident",
        "items",
        "canonical_outputs_absent_before_plan",
        "all_10368_messages_decoded_before_plan",
        "physical_audit_completed_before_plan",
        "network_requests",
    }
    if (
        set(plan) != expected_plan_keys
        or
        plan.get("artifact_type") != "DECODE_RECOVERY_OUTPUT_TRANSACTION_PLAN"
        or plan.get("schema_version") != 1
        or plan_path.name != f"{plan.get('attempt_id')}__plan.json"
        or plan.get("network_requests") != 0
        or plan.get("canonical_outputs_absent_before_plan") is not True
        or plan.get("all_10368_messages_decoded_before_plan") is not True
        or plan.get("physical_audit_completed_before_plan") is not True
    ):
        raise RecoveryContractError("invalid decode recovery transaction plan")
    if root_identity(plan_path, root) != dict(expected_plan_identity):
        raise RecoveryContractError("decode recovery plan changed before commit")
    if (
        plan["recovery_authorization"] != dict(expected_authorization)
        or plan["independent_recovery_go"] != dict(expected_go)
        or plan["incident"] != dict(expected_incident)
    ):
        raise RecoveryContractError("decode recovery plan provenance mismatch")
    items = plan.get("items")
    if not isinstance(items, list) or len(items) != len(outputs):
        raise RecoveryContractError("decode recovery transaction item mismatch")
    by_name = {str(item.get("name")): item for item in items}
    if len(by_name) != len(items) or set(by_name) != set(outputs):
        raise RecoveryContractError("decode recovery transaction name mismatch")

    linked: list[tuple[Path, Path, Mapping[str, Any]]] = []
    for name in sorted(outputs):
        item = by_name[name]
        if set(item) != {
            "name",
            "staged_path",
            "destination_path",
            "size_bytes",
            "sha256",
        }:
            raise RecoveryContractError("decode recovery transaction item schema mismatch")
        source = strict_under_root(root, str(item.get("staged_path", "")))
        destination = strict_under_root(root, str(item.get("destination_path", "")))
        if destination != outputs[name]:
            raise RecoveryContractError("decode recovery destination mismatch")
        expected_source = (
            root
            / "raw"
            / "output_transactions"
            / str(plan["attempt_id"])
            / "staged"
            / destination.relative_to(root)
        )
        if source != expected_source:
            raise RecoveryContractError("decode recovery staged path mismatch")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise RecoveryContractError(
                f"decode recovery destination appeared before atomic create: {destination}"
            )
        if source.is_symlink() or not source.is_file():
            raise RecoveryContractError("decode recovery staged file absent/symlink")
        if (
            source.stat().st_size != int(item.get("size_bytes", -1))
            or sha256_file(source) != str(item.get("sha256", ""))
        ):
            raise RecoveryContractError("decode recovery staged identity mismatch")
        try:
            os.link(source, destination, follow_symlinks=False)
        except FileExistsError as exc:
            raise RecoveryContractError(
                f"decode recovery destination race detected: {destination}"
            ) from exc
        if not os.path.samefile(source, destination):
            raise RecoveryContractError("atomic recovery link does not share file identity")
        frozen.fsync_file(destination)
        linked.append((source, destination, item))

    for name in sorted(outputs):
        item = by_name[name]
        destination = outputs[name]
        if (
            destination.stat().st_size != int(item["size_bytes"])
            or sha256_file(destination) != str(item["sha256"])
        ):
            raise RecoveryContractError("linked recovery destination failed rehash")
    # Remove only the transaction-owned staging links after every canonical
    # destination is present and verified.  A crash before this point leaves
    # exact same-file links and no overwrite has occurred.
    for source, _destination, _item in linked:
        source.unlink()

    commit_path = plan_path.with_name(
        plan_path.name.replace("__plan.json", "__committed.json")
    )
    commit_payload = {
            "artifact_type": "DECODE_RECOVERY_OUTPUT_TRANSACTION_COMMIT",
            "schema_version": 1,
            "attempt_id": plan["attempt_id"],
            "plan": root_identity(plan_path, root),
            "outputs": {
                name: root_identity(outputs[name], root) for name in sorted(outputs)
            },
            "atomic_method": "os.link_create_if_absent_then_unlink_staging",
            "all_outputs_reopened_and_rehashed": True,
            "network_requests": 0,
        }
    publish_json_create_if_absent(commit_path, commit_payload, frozen)
    return commit_path


def static_audit(
    root: Path, authorization_path: Path, go_path: Path, workspace_root: Path
) -> dict[str, Any]:
    validate_parent_bytecode_policy()
    install_parent_network_deny_guard()
    (
        authorization,
        runtime_report,
        _frozen,
        _recovery_core,
        _recovery_bootstrap,
    ) = validate_authorization(
        root, authorization_path, workspace_root
    )
    bootstrap_import_contract = authorization["bootstrap_import_contract"]
    go = validate_independent_go(
        root, go_path, authorization_path, authorization, workspace_root
    )
    require_canonical_outputs_absent(root)
    return {
        "status": "PASS_STATIC_NETWORK_ZERO_DECODE_RECOVERY_READY",
        "authorization_sha256": sha256_file(authorization_path),
        "independent_go_sha256": sha256_file(go_path),
        "attempt_id": authorization["recovery_attempt_id"],
        "runtime": runtime_report,
        "network_requests": 0,
        "writes": 0,
    }


def run(
    root: Path, authorization_path: Path, go_path: Path, workspace_root: Path
) -> dict[str, Any]:
    validate_parent_bytecode_policy()
    install_parent_network_deny_guard()
    root = root.resolve()
    workspace_root = workspace_root.resolve()
    (
        authorization,
        runtime_report,
        frozen,
        recovery_core,
        recovery_bootstrap,
    ) = validate_authorization(
        root, authorization_path, workspace_root
    )
    go = validate_independent_go(
        root, go_path, authorization_path, authorization, workspace_root
    )
    fixed_inputs_initial = fixed_input_snapshot(
        root, authorization_path, go_path, authorization, go
    )
    outputs = require_canonical_outputs_absent(root)

    range_path = strict_under_root(root, FIXED_ROOT_INPUTS["field_range_census"])
    rows = read_range_rows(range_path, authorization["field_range_census"])
    prerecovery_topology = validate_prerecovery_topology(root)
    raw_manifest_rows, before_inventories = validate_raw_cache(root, rows, frozen)
    validate_raw_cache_lock(
        root, authorization, before_inventories, prerecovery_topology
    )

    # Re-run the exact real-GRIB process isolation preflight immediately before
    # claiming the decode attempt.  It cannot touch the Track-A raw cache.
    pilot_report = recovery_bootstrap.run_real_spawn_pilot(
        authorization["real_grib_pilot"],
        authorization["runtime_lock"],
        authorization["recovery_bootstrap"],
        authorization["recovery_runner"],
        authorization["recovery_module"],
        workers=MAX_DECODE_PROCESSES,
        repetitions=70,
    )
    if (
        pilot_report["status"]
        != "PASS_REAL_GRIB_STDLIB_BOOTSTRAP_PROCESS_ISOLATION"
    ):
        raise RecoveryContractError("real process-isolation pilot failed")
    validate_bootstrap_preexecution(
        authorization["recovery_bootstrap"],
        workspace_root / "src" / "noaa_gfs_decode_recovery_bootstrap.py",
    )
    if bootstrap_import_contract_report(workspace_root) != bootstrap_import_contract:
        raise RecoveryContractError("bootstrap alias inventory changed after pilot")

    claim = RecoveryClaim.acquire(
        root, str(authorization["recovery_attempt_id"]), frozen
    )

    def close_failed() -> None:
        if not claim.closed:
            claim.close("failed")

    atexit.register(close_failed)
    locked_prestate_topology = validate_locked_prerecovery_topology(root, claim)
    if (
        locked_prestate_topology["documented_preplan_transactions"]
        != prerecovery_topology["documented_preplan_transactions"]
    ):
        raise RecoveryContractError("prestate topology changed before mutex acquisition")
    locked_raw_manifest_rows, locked_inventories = validate_raw_cache(
        root, rows, frozen
    )
    if (
        locked_inventories != before_inventories
        or locked_raw_manifest_rows != raw_manifest_rows
    ):
        raise RecoveryContractError("raw cache changed before locked decode boundary")
    for relative in PREPLAN_REMNANTS:
        authorized_record = next(
            record
            for record in authorization["documented_preplan_remnants"]
            if record["path"] == relative
        )
        require_root_identity(
            authorized_record, root, relative, "locked prestate remnant"
        )
    locked_fixed_inputs = fixed_input_snapshot(
        root, authorization_path, go_path, authorization, go
    )
    if locked_fixed_inputs != fixed_inputs_initial:
        raise RecoveryContractError("fixed inputs changed before locked decode boundary")
    if bootstrap_import_contract_report(workspace_root) != bootstrap_import_contract:
        raise RecoveryContractError("bootstrap alias inventory changed before full decode")
    tasks = [
        {
            "result": {
                "path": str(frozen.raw_path_for(root, row).resolve()),
                "size_bytes": int(meta["raw_size_bytes"]),
                "sha256": str(meta["raw_sha256"]),
            },
            "row": dict(row),
            "sites": [],  # populated below after coordinate lock validation
        }
        for row, meta in zip(rows, raw_manifest_rows)
    ]
    coordinate_path = strict_under_root(root, FIXED_ROOT_INPUTS["coordinate_lock"])
    coordinate_lock = read_json_object(coordinate_path, "coordinate lock")
    sites = coordinate_lock.get("sites")
    if not isinstance(sites, list) or len(sites) != 17:
        raise RecoveryContractError("coordinate-lock site schema mismatch")
    for task in tasks:
        task["sites"] = sites

    decoded_rows, process_audit = recovery_bootstrap.bounded_spawn_map(
        tasks,
        worker_kind="decode",
        runtime_lock=authorization["runtime_lock"],
        bootstrap_identity=authorization["recovery_bootstrap"],
        runner_identity=authorization["recovery_runner"],
        core_identity=authorization["recovery_module"],
        max_workers=MAX_DECODE_PROCESSES,
    )
    if (
        process_audit.completed_tasks != EXPECTED_ROWS
        or len(process_audit.observed_worker_pids) != MAX_DECODE_PROCESSES
    ):
        raise RecoveryContractError("full decode did not use exact seven-process pool")
    validate_bootstrap_preexecution(
        authorization["recovery_bootstrap"],
        workspace_root / "src" / "noaa_gfs_decode_recovery_bootstrap.py",
    )
    if bootstrap_import_contract_report(workspace_root) != bootstrap_import_contract:
        raise RecoveryContractError("bootstrap alias inventory changed after full decode")
    # Progress is written only after all decode tasks succeed, so a decode
    # failure leaves no misleading partial-completion ledger.
    for completed in range(100, EXPECTED_ROWS, 100):
        write_progress(root, claim.attempt_id, completed, frozen)
    write_progress(root, claim.attempt_id, EXPECTED_ROWS, frozen)

    site_rows, group_rows, physical_audit = assemble_matrices(
        rows, decoded_rows, sites
    )
    if physical_audit["decode_worker_pids"] != list(process_audit.observed_worker_pids):
        raise RecoveryContractError("decode worker PID audit mismatch")

    # Full immutable cache closure is recomputed after decode and before any
    # staged output is written.
    post_raw_manifest_rows, after_inventories = validate_raw_cache(root, rows, frozen)
    if after_inventories != before_inventories:
        raise RecoveryContractError("raw/event/progress/launch inventory changed during decode")
    if post_raw_manifest_rows != raw_manifest_rows:
        raise RecoveryContractError("raw sidecar payload changed during decode")
    for relative in PREPLAN_REMNANTS:
        authorized_record = next(
            record
            for record in authorization["documented_preplan_remnants"]
            if record["path"] == relative
        )
        require_root_identity(authorized_record, root, relative, "preserved preplan remnant")
    require_canonical_outputs_absent(root)
    if shutil.disk_usage(root).free < 200_000_000_000:
        raise RecoveryContractError("final 200GB disk reserve failed")
    staged = frozen.staged_output_paths(root, claim.attempt_id, outputs)
    frozen.write_parquet_exclusive(staged["raw_parquet"], raw_manifest_rows)
    frozen.write_csv_exclusive(
        staged["raw_csv"], raw_manifest_rows, list(raw_manifest_rows[0])
    )
    frozen.write_parquet_exclusive(staged["site"], site_rows)
    frozen.write_parquet_exclusive(staged["group"], group_rows)

    recovery_provenance = {
        "incident": authorization["incident"],
        "recovery_authorization": root_identity(authorization_path, root),
        "independent_recovery_go": root_identity(go_path, root),
        "recovery_runner": authorization["recovery_runner"],
        "recovery_module": authorization["recovery_module"],
        "recovery_bootstrap": authorization["recovery_bootstrap"],
        "bootstrap_trust_policy": authorization["bootstrap_trust_policy"],
        "bootstrap_import_contract": bootstrap_import_contract,
        "recovery_test": authorization["recovery_test"],
        "recovery_bootstrap_test": authorization["recovery_bootstrap_test"],
        "recovery_runner_test": authorization["recovery_runner_test"],
        "postrun_auditor": authorization["postrun_auditor"],
        "postrun_auditor_test": authorization["postrun_auditor_test"],
        "original_frozen_runner": authorization["original_runner"],
        "raw_cache_lock": authorization["raw_cache_lock"],
        "real_spawn_preflight": authorization["real_spawn_preflight"],
        "pilot_runtime_report": pilot_report,
        "fixed_inputs_initial": fixed_inputs_initial,
        "locked_prestate_topology": locked_prestate_topology,
        "locked_fixed_inputs": locked_fixed_inputs,
    }
    physical_audit.update(
        {
            "artifact_type": "TRACK_A_PHYSICAL_AND_COVERAGE_AUDIT_RECOVERED",
            "recovery_provenance": recovery_provenance,
            "final_free_disk_before_transaction_bytes": shutil.disk_usage(root).free,
            "final_200gb_reserve_pass": True,
        }
    )
    frozen.write_json_exclusive(staged["audit"], physical_audit)
    access = {
        "artifact_type": "TRACK_A_RAW_ACCESS_LEDGER_RECOVERED_NETWORK_ZERO",
        "expected_ranges": EXPECTED_ROWS,
        "expected_bytes": EXPECTED_BYTES,
        "network_requests_this_invocation": 0,
        "network_bytes_this_invocation": 0,
        "raw_cache_reuses": EXPECTED_ROWS,
        "decode_processes": MAX_DECODE_PROCESSES,
        "decode_threads": 0,
        "before_inventories": before_inventories,
        "after_inventories": after_inventories,
        "inventories_equal": True,
        "raw_event_progress_mutations": 0,
        "documented_preplan_remnants_preserved": authorization[
            "documented_preplan_remnants"
        ],
        "labels_read": False,
        "2024_arrays_read": False,
        "2025_arrays_read": False,
        "models_fit": 0,
        "submission_csv_created": False,
        "recovery_provenance": recovery_provenance,
    }
    frozen.write_json_exclusive(staged["access"], access)
    pbl_pass = bool(
        physical_audit["families"]["PBL_HEIGHT"]["physical_gate_pass"]
    )
    wind_pass = bool(
        physical_audit["families"]["LOW_LEVEL_ISOBARIC_WIND_PROFILE"][
            "physical_gate_pass"
        ]
    )
    decoded_lock = {
        "artifact_type": "TARGET_FREE_DECODED_MATRIX_LOCK_RECOVERED_NETWORK_ZERO",
        "schema_version": 1,
        "recovery_provenance": recovery_provenance,
        "runtime_identity": runtime_report,
        "raw_manifest_parquet": frozen.identity_for_destination(
            staged["raw_parquet"], outputs["raw_parquet"], root
        ),
        "raw_manifest_csv": frozen.identity_for_destination(
            staged["raw_csv"], outputs["raw_csv"], root
        ),
        "site_matrix": frozen.identity_for_destination(
            staged["site"], outputs["site"], root
        ),
        "group_matrix": frozen.identity_for_destination(
            staged["group"], outputs["group"], root
        ),
        "physical_and_coverage_audit": frozen.identity_for_destination(
            staged["audit"], outputs["audit"], root
        ),
        "access_ledger": frozen.identity_for_destination(
            staged["access"], outputs["access"], root
        ),
        "target_free_duplicate_estimator_may_start": pbl_pass or wind_pass,
        "network_requests": 0,
        "labels_read": False,
    }
    frozen.write_json_exclusive(staged["lock"], decoded_lock)
    manifest = {
        "artifact_type": "NOAA_GFS_MULTISEASON_RAW_AND_DECODE_MANIFEST_RECOVERED",
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "launch_attempt_id": claim.attempt_id,
        "recovery_provenance": recovery_provenance,
        "runtime_identity": runtime_report,
        "decoded_matrix_lock": frozen.identity_for_destination(
            staged["lock"], outputs["lock"], root
        ),
        "exact_ranges": EXPECTED_ROWS,
        "exact_bytes": EXPECTED_BYTES,
        "decode_processes": MAX_DECODE_PROCESSES,
        "decode_threads": 0,
        "network_requests": 0,
        "labels_read": False,
        "2024_arrays_read": False,
        "2025_arrays_read": False,
        "models_fit": 0,
        "submission_csv_created": False,
    }
    frozen.write_json_exclusive(staged["manifest"], manifest)

    # Last control-plane snapshot is intentionally after all eight staged
    # outputs exist and immediately before the durable transaction plan.
    fixed_inputs_preplan = fixed_input_snapshot(
        root, authorization_path, go_path, authorization, go
    )
    if fixed_inputs_preplan != fixed_inputs_initial:
        raise RecoveryContractError("fixed inputs changed before transaction plan")
    if bootstrap_import_contract_report(workspace_root) != bootstrap_import_contract:
        raise RecoveryContractError("bootstrap alias inventory changed before plan")
    if shutil.disk_usage(root).free < 200_000_000_000:
        raise RecoveryContractError("disk reserve failed before transaction plan")
    plan = write_recovery_transaction_plan(
        root,
        claim.attempt_id,
        staged,
        outputs,
        root_identity(authorization_path, root),
        root_identity(go_path, root),
        authorization["incident"],
        frozen,
    )
    plan_identity = root_identity(plan, root)
    commit = commit_recovery_transaction_fail_if_exists(
        root,
        plan,
        outputs,
        frozen,
        expected_plan_identity=plan_identity,
        expected_authorization=root_identity(authorization_path, root),
        expected_go=root_identity(go_path, root),
        expected_incident=authorization["incident"],
    )
    postcommit_raw_manifest_rows, postcommit_inventories = validate_raw_cache(
        root, rows, frozen
    )
    if (
        postcommit_inventories != before_inventories
        or postcommit_raw_manifest_rows != raw_manifest_rows
    ):
        raise RecoveryContractError("raw/event/progress/failed-lock changed after commit")
    for relative in PREPLAN_REMNANTS:
        authorized_record = next(
            record
            for record in authorization["documented_preplan_remnants"]
            if record["path"] == relative
        )
        require_root_identity(
            authorized_record, root, relative, "postcommit preserved preplan remnant"
        )
    fixed_inputs_postcommit = fixed_input_snapshot(
        root, authorization_path, go_path, authorization, go
    )
    if fixed_inputs_postcommit != fixed_inputs_initial:
        raise RecoveryContractError("fixed inputs changed after canonical commit")
    if bootstrap_import_contract_report(workspace_root) != bootstrap_import_contract:
        raise RecoveryContractError("bootstrap alias inventory changed after commit")
    postcommit_audit = (
        root
        / "decoded"
        / "recovery_history"
        / f"{claim.attempt_id}__postcommit_input_audit.json"
    )
    frozen.write_json_exclusive_fsync(
        postcommit_audit,
        {
            "artifact_type": "DECODE_RECOVERY_POSTCOMMIT_INPUT_AUDIT",
            "attempt_id": claim.attempt_id,
            "initial": fixed_inputs_initial,
            "preplan": fixed_inputs_preplan,
            "postcommit": fixed_inputs_postcommit,
            "all_equal": True,
            "network_requests": 0,
        },
    )
    complete_lock = claim.close("complete")
    return {
        "status": "PASS_NETWORK_ZERO_PROCESS_ISOLATED_DECODE_RECOVERY",
        "manifest": root_identity(outputs["manifest"], root),
        "decoded_lock": root_identity(outputs["lock"], root),
        "transaction_plan": root_identity(plan, root),
        "transaction_commit": root_identity(commit, root),
        "recovery_complete_lock": root_identity(complete_lock, root),
        "postcommit_input_audit": root_identity(postcommit_audit, root),
        "raw_ranges": EXPECTED_ROWS,
        "raw_bytes": EXPECTED_BYTES,
        "network_requests": 0,
        "decode_processes": MAX_DECODE_PROCESSES,
        "family_status": {
            key: value["status"] for key, value in physical_audit["families"].items()
        },
    }


def main() -> None:
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument(
        "--authorization",
        type=Path,
        default=ROOT_DEFAULT / "prereg" / "decode_recovery_authorization_v1.json",
    )
    parser.add_argument(
        "--independent-go",
        type=Path,
        default=ROOT_DEFAULT
        / "independent_redteam"
        / "TRACK_A_DECODE_RECOVERY_GO_V1.json",
    )
    parser.add_argument("--static-audit", action="store_true")
    args = parser.parse_args()
    workspace_root = Path(__file__).resolve().parents[1]
    try:
        if args.static_audit:
            report = static_audit(
                args.root.resolve(),
                args.authorization.resolve(),
                args.independent_go.resolve(),
                workspace_root,
            )
        else:
            report = run(
                args.root.resolve(),
                args.authorization.resolve(),
                args.independent_go.resolve(),
                workspace_root,
            )
    except Exception as exc:
        report = {
            "status": "FAIL_NETWORK_ZERO_DECODE_RECOVERY",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "network_requests": 0,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(2) from None
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

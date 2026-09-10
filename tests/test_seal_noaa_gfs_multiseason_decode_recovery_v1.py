from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import seal_noaa_gfs_multiseason_decode_recovery_v1 as sealer


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _bytecode_policy(monkeypatch: pytest.MonkeyPatch):
    previous = sys.dont_write_bytecode
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.delenv("PYTHONPYCACHEPREFIX", raising=False)
    sys.dont_write_bytecode = True
    try:
        yield
    finally:
        sys.dont_write_bytecode = previous


def test_sealer_source_has_no_network_or_recovery_launch_path() -> None:
    source_path = ROOT / "scripts" / "seal_noaa_gfs_multiseason_decode_recovery_v1.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    } | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert imports.isdisjoint({"requests", "urllib", "httpx", "aiohttp", "boto3"})
    assert "fetch_exact_range" not in names
    assert "download_or_resume_range" not in names
    assert "run" not in {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "recovery"
    }
    assert "ThreadPoolExecutor" not in names


def test_frozen_supporting_schema_sets_are_exact() -> None:
    assert len(sealer.RAW_LOCK_KEYS) == 14
    assert len(sealer.PREFLIGHT_KEYS) == 44
    assert len(sealer.TEST_EVIDENCE_KEYS) == 30
    assert sealer.AUTHORIZATION_KEYS == {
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
        *sealer.FIXED_ROOT_INPUTS.keys(),
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


def test_immutable_pytest_commands_are_one_clean_file_each() -> None:
    relatives = ("tests/a.py", "tests/b.py", "tests/c.py")
    commands = sealer.isolated_pytest_commands(Path(sys.executable), relatives)
    assert len(commands) == len(relatives)
    assert [command[-1] for command in commands] == list(relatives)
    assert all(command[-3:-1] == ["-p", "no:cacheprovider"] for command in commands)
    assert all(command[2] == "-c" for command in commands)
    assert all(
        "network forbidden in immutable recovery test subprocess" in command[3]
        for command in commands
    )
    assert all(sum(value.startswith("tests/") for value in command) == 1 for command in commands)


def test_preflight_records_and_closes_initial_process_scheduling_flake() -> None:
    incident = sealer.SPAWN_SCHEDULING_FLAKE_INCIDENT
    assert incident["observed_during_unsealed_dry_run"] is True
    assert incident["seal_artifacts_published"] == 0
    assert incident["retry_or_selected_result_used_as_final_evidence"] is False
    assert incident["status"] == "CLOSED_BY_SYNCHRONIZED_FIRST_WORKER_WAVE_GATE"


def test_noncontract_monolithic_diagnostic_is_recorded_but_never_selected() -> None:
    diagnostic = sealer.NONCONTRACT_MONOLITHIC_DIAGNOSTIC
    assert diagnostic["topology"] == "ONE_INTERPRETER_MULTI_FILE_NONCONTRACT"
    assert (diagnostic["passed"], diagnostic["skipped"], diagnostic["failed"]) == (
        146,
        1,
        13,
    )
    assert diagnostic["used_as_contract_evidence"] is False
    assert diagnostic["required_contract_topology"] == (
        "ONE_TEST_FILE_PER_CLEAN_SUBPROCESS"
    )
    assert diagnostic["outer_network_guard_installed"] is False
    assert diagnostic["outer_network_requests"] == "NOT_INSTRUMENTED_NO_ZERO_CLAIM"
    assert diagnostic["outer_data_reads"] == "NOT_INSTRUMENTED_NO_ZERO_CLAIM"
    assert diagnostic["outer_writes"] == "NOT_INSTRUMENTED_NO_ZERO_CLAIM"
    assert diagnostic["canonical_spawn_subreport"]["network_requests"] == 0
    assert diagnostic["canonical_spawn_subreport"]["controls_created"] == 0


def test_failed_direct_file_seal_incident_binds_exact_alias_remediation() -> None:
    root = ROOT / "artifacts" / "baram2026_ncei_scada_longrun_20260810_v2"
    path = root / sealer.FIXED_ROOT_INPUTS["sealer_entrypoint_incident"]
    assert path.stat().st_size == 7_732
    assert sealer.sha256_file(path) == (
        "9e74f57dcddbd0480464fda5953014e5be411b927de99beef58d7ab5bb10afbb"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["failure_boundary"]["seal_publications_created"] == 0
    assert payload["failure_boundary"]["network_requests"] == 0
    assert payload["mandatory_remediation"] == {
        "canonical_write_invocation": (
            "python -B -m scripts.seal_noaa_gfs_multiseason_decode_recovery_v1"
        ),
        "direct_file_write_invocation_forbidden": True,
        "bootstrap_top_level_module_name": "noaa_gfs_decode_recovery_bootstrap",
        "bootstrap_import_root": str((ROOT / "src").resolve()),
        "src_package_initializer_must_not_execute": True,
        "bootstrap_source_origin_hash_ast_and_pyc_gates_remain_required": True,
        "real_module_entrypoint_seven_spawn_regression_required": True,
        "revised_code_test_auditor_hash_chain_and_independent_pass_required": True,
        "retry_authorized": False,
    }


def _clean_cli_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.pop("PYTHONPYCACHEPREFIX", None)
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return environment


def test_direct_file_write_mode_fails_before_data_or_publication(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-B",
        str(ROOT / "scripts" / "seal_noaa_gfs_multiseason_decode_recovery_v1.py"),
        "--root",
        str(tmp_path),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=_clean_cli_environment(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert completed.returncode == 2
    report = json.loads(completed.stdout)
    assert report["status"] == "FAIL_DECODE_RECOVERY_SEAL"
    assert report["error_type"] == "RecoverySealError"
    assert "canonical `python -B -m" in report["error"]
    assert list(tmp_path.rglob("*")) == []


def test_imported_programmatic_seal_fails_before_data_or_publication(
    tmp_path: Path,
) -> None:
    with pytest.raises(sealer.RecoverySealError, match="canonical `python -B -m"):
        sealer.seal(tmp_path, ROOT)
    assert list(tmp_path.rglob("*")) == []


def test_imported_programmatic_spawn_audit_fails_before_data_access(
    tmp_path: Path,
) -> None:
    with pytest.raises(sealer.RecoverySealError, match="canonical `python -B -m"):
        sealer.spawn_importability_audit(tmp_path, ROOT)
    assert list(tmp_path.rglob("*")) == []


def test_direct_file_static_audit_remains_read_only_and_allowed(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(ROOT / "scripts" / "seal_noaa_gfs_multiseason_decode_recovery_v1.py"),
            "--static-audit",
            "--root",
            str(tmp_path),
        ],
        cwd=ROOT,
        env=_clean_cli_environment(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    report = json.loads(completed.stdout)
    assert report["status"] == "PASS_STATIC_DECODE_RECOVERY_SEALER_READY"
    assert report["writes"] == 0
    assert report["network_requests"] == 0
    assert list(tmp_path.rglob("*")) == []


def test_canonical_module_entrypoint_real_seven_spawn_is_read_only() -> None:
    root = ROOT / "artifacts" / "baram2026_ncei_scada_longrun_20260810_v2"
    controls = (
        root / sealer.RAW_LOCK_RELATIVE,
        root / sealer.PREFLIGHT_RELATIVE,
        root / sealer.AUTHORIZATION_RELATIVE,
        root / sealer.REVIEW_RELATIVE,
        root / sealer.GO_RELATIVE,
    )
    assert not any(path.exists() for path in controls)
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            sealer.CANONICAL_MODULE_ENTRYPOINT,
            "--spawn-importability-audit",
            "--root",
            str(root),
        ],
        cwd=ROOT,
        env=_clean_cli_environment(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    report = json.loads(completed.stdout)
    assert report["status"] == (
        "PASS_CANONICAL_MODULE_ENTRYPOINT_TOP_LEVEL_BOOTSTRAP_7SPAWN"
    )
    assert report["worker_count"] == 7
    assert report["tasks"] == 70
    assert report["first_wave_synchronized"] is True
    assert report["src_package_initializer_executed"] is False
    assert report["controls_created"] == 0
    assert report["network_requests"] == 0
    assert report["writes"] == 0
    assert report["bootstrap_import_contract"] == {
        "module_name": "noaa_gfs_decode_recovery_bootstrap",
        "source_root": str((ROOT / "src").resolve()),
        "sys_path_index": 0,
        "pythonpath_exact": str((ROOT / "src").resolve()),
        "src_package_initializer_executed": False,
        "bootstrap_module_package": "",
        "stdlib_import_roots": [
            "__future__",
            "ast",
            "collections",
            "concurrent",
            "hashlib",
            "importlib",
            "multiprocessing",
            "os",
            "pathlib",
            "socket",
            "sys",
            "types",
            "typing",
        ],
        "stdlib_shadow_candidates": [],
        "top_level_module_competing_candidates": [],
        "bootstrap_source_path": str(
            (ROOT / "src" / "noaa_gfs_decode_recovery_bootstrap.py").resolve()
        ),
    }
    assert not any(path.exists() for path in controls)
    assert "src" not in sys.modules


def test_json_publisher_fails_on_race_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "seal.json"
    original_link = os.link

    def racing_link(source: object, target: object, **kwargs: object) -> None:
        Path(target).write_bytes(b"attacker")
        raise FileExistsError(target)

    monkeypatch.setattr(os, "link", racing_link)
    with pytest.raises(sealer.RecoverySealError, match="race"):
        sealer.publish_json_no_overwrite(destination, {"safe": True})
    assert destination.read_bytes() == b"attacker"
    monkeypatch.setattr(os, "link", original_link)


def test_static_audit_writes_nothing(tmp_path: Path) -> None:
    before = list(tmp_path.rglob("*"))
    report = sealer.static_audit(tmp_path, ROOT)
    after = list(tmp_path.rglob("*"))
    assert report["status"] == "PASS_STATIC_DECODE_RECOVERY_SEALER_READY"
    assert report["network_requests"] == 0
    assert report["writes"] == 0
    assert before == after == []


def test_seal_recloses_topology_after_tests_before_first_publication() -> None:
    source = (
        ROOT / "scripts" / "seal_noaa_gfs_multiseason_decode_recovery_v1.py"
    ).read_text(encoding="utf-8")
    tests_at = source.index("test_evidence = run_immutable_test_evidence")
    topology_at = source.index("final_topology = recovery.validate_prerecovery_topology")
    canonical_at = source.index("recovery.require_canonical_outputs_absent(root)", topology_at)
    publish_at = source.index(
        'publish_json_no_overwrite(destinations["raw_cache_lock"]', canonical_at
    )
    assert tests_at < topology_at < canonical_at < publish_at


def test_external_identity_rejects_lexical_symlink_before_resolve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "target.bin"
    target.write_bytes(b"safe")
    link = tmp_path / "link.bin"
    original_is_symlink = Path.is_symlink
    original_resolve = Path.resolve
    resolved = False

    def fake_is_symlink(path: Path) -> bool:
        return Path(os.path.abspath(str(path))) == Path(os.path.abspath(str(link))) or original_is_symlink(path)

    def tracked_resolve(path: Path, *args: object, **kwargs: object) -> Path:
        nonlocal resolved
        resolved = True
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "is_symlink", fake_is_symlink)
    monkeypatch.setattr(Path, "resolve", tracked_resolve)
    with pytest.raises(sealer.RecoverySealError, match="symlink/junction"):
        sealer.external_identity(link)
    assert resolved is False


def test_runtime_lock_binds_frozen_python_runtime_and_decoder() -> None:
    original = json.loads(
        (
            ROOT
            / "artifacts"
            / "baram2026_ncei_scada_longrun_20260810_v2"
            / "prereg"
            / "raw_launch_authorization_v1.json"
        ).read_text(encoding="utf-8")
    )
    payload = sealer.runtime_lock(ROOT, original["runtime_identity"])
    assert payload["api_version"] == "2.47.0"
    assert payload["definitions_path"] == "/MEMFS/definitions"
    assert payload["original_runtime_identity"] == original["runtime_identity"]
    assert payload["recovery_module"] == sealer.external_identity(
        ROOT / "src" / "noaa_gfs_decode_recovery.py"
    )

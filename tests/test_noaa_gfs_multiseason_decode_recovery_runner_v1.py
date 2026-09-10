from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import run_noaa_gfs_multiseason_decode_recovery_v1 as recovery


ROOT = Path(__file__).resolve().parents[1]


class FrozenStub:
    @staticmethod
    def _write(path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = (
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        with path.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())

    write_json_atomic_exclusive_fsync = _write
    write_json_exclusive_fsync = _write
    write_json_exclusive = _write

    @staticmethod
    def fsync_file(path: Path) -> None:
        with path.open("r+b") as stream:
            os.fsync(stream.fileno())


def _utc_z(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _synthetic_original_progress_evidence() -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
    list[dict[str, object]],
]:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows: list[dict[str, object]] = []
    for index in range(recovery.EXPECTED_ROWS):
        rows.append(
            {
                "object_key": f"gfs.synthetic/{index:05d}",
                "forecast_hour": index,
                "variable": "HPBL",
                "level": "surface",
                "range_bytes": (
                    1
                    if index < recovery.EXPECTED_ROWS - 1
                    else recovery.EXPECTED_BYTES - recovery.EXPECTED_ROWS + 1
                ),
            }
        )
    starts = [
        {
            "global_raw_attempt_number": global_attempt,
            "created_utc": _utc_z(
                base + timedelta(microseconds=global_attempt * 10)
            ),
        }
        for global_attempt in range(1, recovery.EXPECTED_ROWS + 1)
    ]
    keys = [
        "|".join(
            (
                str(row["object_key"]),
                str(row["forecast_hour"]),
                str(row["variable"]),
                str(row["level"]),
            )
        )
        for row in rows
    ]
    byte_prefix = [0]
    for row in rows:
        byte_prefix.append(byte_prefix[-1] + int(row["range_bytes"]))
    progress: list[dict[str, object]] = []
    for completed in recovery.ORIGINAL_PROGRESS_CHECKPOINTS:
        checkpoint = (
            f"raw_ranges__{recovery.FAILED_ATTEMPT}__{completed:06d}"
        )
        payload = {
            "actual_http_attempts_cumulative": completed,
            "attempt_id": recovery.FAILED_ATTEMPT,
            "checkpoint_name": checkpoint,
            "completed_key_set_sha256": hashlib.sha256(
                ("\n".join(sorted(keys[:completed])) + "\n").encode("utf-8")
            ).hexdigest(),
            "completed_ranges": completed,
            "created_utc": _utc_z(
                base + timedelta(microseconds=completed * 10 + 5)
            ),
            "network_bytes_this_invocation_so_far": byte_prefix[completed],
            "network_requests_this_invocation_so_far": completed,
            "phase": "RAW_RANGE_DOWNLOAD",
        }
        progress.append({"filename": f"{checkpoint}.json", "payload": payload})
    failed_lock = {
        "attempt_id": recovery.FAILED_ATTEMPT,
        "created_utc": _utc_z(base),
        "pid": recovery.FAILED_ATTEMPT_PID,
    }
    return rows, progress, failed_lock, starts


def _transaction_fixture(tmp_path: Path) -> tuple[
    Path,
    str,
    dict[str, Path],
    dict[str, Path],
    dict[str, str],
    dict[str, str],
    dict[str, str],
]:
    root = tmp_path.resolve()
    attempt = "decode_recovery_v1__synthetic"
    outputs = {
        "a": root / "raw" / "A.bin",
        "b": root / "decoded" / "B.bin",
    }
    staged = {
        name: root
        / "raw"
        / "output_transactions"
        / attempt
        / "staged"
        / output.relative_to(root)
        for name, output in outputs.items()
    }
    staged["a"].parent.mkdir(parents=True, exist_ok=True)
    staged["b"].parent.mkdir(parents=True, exist_ok=True)
    staged["a"].write_bytes(b"alpha")
    staged["b"].write_bytes(b"beta")
    auth = {"path": "prereg/auth.json", "size_bytes": 1, "sha256": "a" * 64}
    go = {"path": "independent/go.json", "size_bytes": 2, "sha256": "b" * 64}
    incident = {"path": "incidents/incident.json", "size_bytes": 3, "sha256": "c" * 64}
    return root, attempt, outputs, staged, auth, go, incident


def _write_plan(
    fixture: tuple[
        Path,
        str,
        dict[str, Path],
        dict[str, Path],
        dict[str, str],
        dict[str, str],
        dict[str, str],
    ]
) -> Path:
    root, attempt, outputs, staged, auth, go, incident = fixture
    return recovery.write_recovery_transaction_plan(
        root, attempt, staged, outputs, auth, go, incident, FrozenStub
    )


def _commit(
    fixture: tuple,
    plan: Path,
) -> Path:
    root, _attempt, outputs, _staged, auth, go, incident = fixture
    return recovery.commit_recovery_transaction_fail_if_exists(
        root,
        plan,
        outputs,
        FrozenStub,
        expected_plan_identity=recovery.root_identity(plan, root),
        expected_authorization=auth,
        expected_go=go,
        expected_incident=incident,
    )


def test_runner_source_has_no_fetch_client_or_thread_pool() -> None:
    source = (ROOT / "scripts" / "run_noaa_gfs_multiseason_decode_recovery_v1.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imported.isdisjoint({"urllib", "requests", "httpx", "aiohttp", "boto3"})
    names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    } | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert "ThreadPoolExecutor" not in names
    assert "urlopen" not in names
    assert "fetch_exact_range" not in names
    assert "download_or_resume_range" not in names
    assert not any(
        isinstance(node, ast.ImportFrom)
        and node.module == "src.noaa_gfs_decode_recovery"
        for node in ast.walk(tree)
    )
    assert "resume_existing_transaction" not in source
    assert "recovery_mode" not in source


def test_bootstrap_top_level_import_contract_is_exact_and_src_free() -> None:
    code = """
import json
import pathlib
import sys
from scripts import run_noaa_gfs_multiseason_decode_recovery_v1 as recovery
root = pathlib.Path.cwd()
report = recovery.bootstrap_import_contract_report(root)
print(json.dumps({'report': report, 'src_loaded': 'src' in sys.modules}))
"""
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.pop("PYTHONPYCACHEPREFIX", None)
    completed = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["src_loaded"] is False
    assert payload["report"] == {
        "module_name": recovery.BOOTSTRAP_TOP_LEVEL_MODULE_NAME,
        "source_root": str((ROOT / "src").resolve()),
        "sys_path_index": 0,
        "pythonpath_exact": str((ROOT / "src").resolve()),
        "src_package_initializer_executed": False,
        "bootstrap_module_package": "",
        "stdlib_import_roots": sorted(
            {name.split(".", 1)[0] for name in recovery.BOOTSTRAP_STDLIB_IMPORTS}
        ),
        "stdlib_shadow_candidates": [],
        "top_level_module_competing_candidates": [],
        "bootstrap_source_path": str(
            (ROOT / "src" / "noaa_gfs_decode_recovery_bootstrap.py").resolve()
        ),
    }


def test_bootstrap_import_root_rejects_casefolded_stdlib_shadow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "src"
    source_root.mkdir()
    (source_root / "noaa_gfs_decode_recovery_bootstrap.py").write_text(
        "# exact alias source\n", encoding="utf-8"
    )
    (source_root / "AsT.PY").write_text("raise RuntimeError('shadow')\n", encoding="utf-8")
    monkeypatch.delitem(sys.modules, "src", raising=False)
    with pytest.raises(recovery.RecoveryContractError, match="shadows bootstrap stdlib"):
        recovery.install_bootstrap_top_level_import_contract(tmp_path)


@pytest.mark.parametrize(
    "competitor",
    (
        "noaa_gfs_decode_recovery_bootstrap",
        "NoAa_GfS_DeCoDe_ReCoVeRy_BoOtStRaP.CP313-WIN_AMD64.PYD",
    ),
)
def test_bootstrap_import_root_rejects_same_alias_competitor(
    competitor: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "src"
    source_root.mkdir()
    (source_root / "noaa_gfs_decode_recovery_bootstrap.py").write_text(
        "# exact alias source\n", encoding="utf-8"
    )
    candidate = source_root / competitor
    if "." in competitor:
        candidate.write_bytes(b"extension competitor")
    else:
        candidate.mkdir()
        (candidate / "__init__.py").write_text(
            "raise RuntimeError('package competitor')\n", encoding="utf-8"
        )
    monkeypatch.delitem(sys.modules, "src", raising=False)
    with pytest.raises(recovery.RecoveryContractError, match="not unique"):
        recovery.install_bootstrap_top_level_import_contract(tmp_path)


def test_noncontract_monolithic_diagnostic_exact_semantics_pass() -> None:
    recovery.validate_noncontract_monolithic_diagnostic(
        json.loads(json.dumps(recovery.NONCONTRACT_MONOLITHIC_DIAGNOSTIC))
    )
    diagnostic = recovery.NONCONTRACT_MONOLITHIC_DIAGNOSTIC
    assert (diagnostic["passed"], diagnostic["skipped"], diagnostic["failed"]) == (
        146,
        1,
        13,
    )
    assert diagnostic["used_as_contract_evidence"] is False
    assert diagnostic["outer_network_requests"] == "NOT_INSTRUMENTED_NO_ZERO_CLAIM"


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("exit_code", 0),
        ("failed", 0),
        ("outer_network_requests", 0),
        ("used_as_contract_evidence", True),
    ),
)
def test_noncontract_monolithic_diagnostic_tamper_fails(
    field: str, replacement: object
) -> None:
    diagnostic = json.loads(json.dumps(recovery.NONCONTRACT_MONOLITHIC_DIAGNOSTIC))
    diagnostic[field] = replacement
    with pytest.raises(recovery.RecoveryContractError, match="exact semantics"):
        recovery.validate_noncontract_monolithic_diagnostic(diagnostic)


def test_recovery_core_bad_hash_fails_before_module_execution() -> None:
    module_name = "src.noaa_gfs_decode_recovery"
    prior = sys.modules.pop(module_name, None)
    path = ROOT / "src" / "noaa_gfs_decode_recovery.py"
    record = {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": "0" * 64,
    }
    try:
        with pytest.raises(recovery.RecoveryContractError, match="hash mismatch"):
            recovery._load_recovery_core_preexecution(record, ROOT)
        assert module_name not in sys.modules
    finally:
        if prior is not None:
            sys.modules[module_name] = prior


def test_plan_publication_race_never_overwrites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _transaction_fixture(tmp_path)
    root, attempt, outputs, staged, auth, go, incident = fixture
    plan = root / "raw" / "output_transactions" / f"{attempt}__plan.json"
    original_link = os.link

    def racing_link(source: object, destination: object, **kwargs: object) -> None:
        if Path(destination) == plan:
            Path(destination).write_bytes(b"attacker-plan")
            raise FileExistsError(destination)
        original_link(source, destination, **kwargs)

    monkeypatch.setattr(os, "link", racing_link)
    with pytest.raises(recovery.RecoveryContractError, match="publication race"):
        recovery.write_recovery_transaction_plan(
            root, attempt, staged, outputs, auth, go, incident, FrozenStub
        )
    assert plan.read_bytes() == b"attacker-plan"


def test_transaction_commit_uses_atomic_fail_if_exists_links(tmp_path: Path) -> None:
    fixture = _transaction_fixture(tmp_path)
    root, _attempt, outputs, staged, _auth, _go, _incident = fixture
    plan = _write_plan(fixture)
    commit = _commit(fixture, plan)
    assert commit.is_file()
    assert outputs["a"].read_bytes() == b"alpha"
    assert outputs["b"].read_bytes() == b"beta"
    assert not staged["a"].exists()
    assert not staged["b"].exists()
    payload = json.loads(commit.read_text(encoding="utf-8"))
    assert payload["atomic_method"] == "os.link_create_if_absent_then_unlink_staging"
    assert payload["all_outputs_reopened_and_rehashed"] is True
    assert payload["network_requests"] == 0


def test_destination_race_fails_without_overwriting_injected_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _transaction_fixture(tmp_path)
    _root, _attempt, outputs, _staged, _auth, _go, _incident = fixture
    plan = _write_plan(fixture)
    original_link = os.link
    first = True

    def racing_link(source: object, destination: object, **kwargs: object) -> None:
        nonlocal first
        if first:
            first = False
            Path(destination).write_bytes(b"attacker")
            raise FileExistsError(destination)
        original_link(source, destination, **kwargs)

    monkeypatch.setattr(os, "link", racing_link)
    with pytest.raises(recovery.RecoveryContractError, match="race detected"):
        _commit(fixture, plan)
    assert outputs["a"].read_bytes() == b"attacker"


def test_plan_staged_path_swap_fails_before_link(tmp_path: Path) -> None:
    fixture = _transaction_fixture(tmp_path)
    root, _attempt, outputs, _staged, auth, go, incident = fixture
    plan = _write_plan(fixture)
    payload = json.loads(plan.read_text(encoding="utf-8"))
    rogue = root / "rogue.bin"
    rogue.write_bytes(b"alpha")
    payload["items"][0]["staged_path"] = "rogue.bin"
    plan.unlink()
    FrozenStub._write(plan, payload)
    with pytest.raises(recovery.RecoveryContractError, match="staged path mismatch"):
        recovery.commit_recovery_transaction_fail_if_exists(
            root,
            plan,
            outputs,
            FrozenStub,
            expected_plan_identity=recovery.root_identity(plan, root),
            expected_authorization=auth,
            expected_go=go,
            expected_incident=incident,
        )
    assert all(not path.exists() for path in outputs.values())


def test_partial_hardlink_commit_is_terminal_under_initial_authorization(
    tmp_path: Path,
) -> None:
    fixture = _transaction_fixture(tmp_path)
    _root, _attempt, outputs, staged, _auth, _go, _incident = fixture
    plan = _write_plan(fixture)
    outputs["a"].parent.mkdir(parents=True, exist_ok=True)
    os.link(staged["a"], outputs["a"])
    with pytest.raises(recovery.RecoveryContractError, match="appeared"):
        _commit(fixture, plan)
    assert outputs["a"].read_bytes() == b"alpha"
    assert not outputs["b"].exists()


def test_commit_record_publication_race_never_overwrites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _transaction_fixture(tmp_path)
    root, attempt, _outputs, _staged, _auth, _go, _incident = fixture
    plan = _write_plan(fixture)
    commit_path = (
        root / "raw" / "output_transactions" / f"{attempt}__committed.json"
    )
    original_link = os.link

    def racing_link(source: object, destination: object, **kwargs: object) -> None:
        if Path(destination) == commit_path:
            Path(destination).write_bytes(b"attacker-commit")
            raise FileExistsError(destination)
        original_link(source, destination, **kwargs)

    monkeypatch.setattr(os, "link", racing_link)
    with pytest.raises(recovery.RecoveryContractError, match="publication race"):
        _commit(fixture, plan)
    assert commit_path.read_bytes() == b"attacker-commit"


def _minimal_prerecovery_tree(root: Path) -> None:
    for name in (
        "launch_history",
        "output_transactions",
        "progress",
        "ranges",
        "request_events",
    ):
        (root / "raw" / name).mkdir(parents=True, exist_ok=True)
    for relative in recovery.PREPLAN_REMNANTS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"documented")


def test_prerecovery_topology_rejects_unexpected_junk(tmp_path: Path) -> None:
    _minimal_prerecovery_tree(tmp_path)
    recovery.validate_prerecovery_topology(tmp_path)
    (tmp_path / "raw" / "junk.bin").write_bytes(b"junk")
    with pytest.raises(recovery.RecoveryContractError, match="top-level"):
        recovery.validate_prerecovery_topology(tmp_path)


def test_prerecovery_topology_rejects_unexpected_empty_nested_dir(
    tmp_path: Path,
) -> None:
    _minimal_prerecovery_tree(tmp_path)
    (tmp_path / "raw" / "output_transactions" / "unexpected_empty").mkdir()
    with pytest.raises(recovery.RecoveryContractError, match="directory mismatch"):
        recovery.validate_prerecovery_topology(tmp_path)


def test_locked_prestate_recloses_transaction_race_after_initial_audit(
    tmp_path: Path,
) -> None:
    _minimal_prerecovery_tree(tmp_path)
    initial = recovery.validate_prerecovery_topology(tmp_path)
    rogue = tmp_path / "raw" / "output_transactions" / "raced" / "junk.bin"
    rogue.parent.mkdir()
    rogue.write_bytes(b"raced-after-initial-audit")
    claim = recovery.RecoveryClaim.acquire(
        tmp_path, "decode_recovery_v1__synthetic", FrozenStub
    )
    with pytest.raises(recovery.RecoveryContractError, match="transaction file closure"):
        recovery.validate_locked_prerecovery_topology(tmp_path, claim)
    assert initial["decoded_tree_absent"] is True
    assert claim.raw_active.is_file()
    assert claim.decoded_active.is_file()


def test_locked_prestate_exact_topology_passes_under_both_mutexes(
    tmp_path: Path,
) -> None:
    _minimal_prerecovery_tree(tmp_path)
    initial = recovery.validate_prerecovery_topology(tmp_path)
    claim = recovery.RecoveryClaim.acquire(
        tmp_path, "decode_recovery_v1__synthetic", FrozenStub
    )
    locked = recovery.validate_locked_prerecovery_topology(tmp_path, claim)
    assert (
        locked["documented_preplan_transactions"]
        == initial["documented_preplan_transactions"]
    )
    assert locked["canonical_outputs_absent"] is True
    assert locked["mutex_payload"] == claim.payload


def test_dual_shared_lock_acquire_and_no_overwrite_close(tmp_path: Path) -> None:
    (tmp_path / "raw").mkdir()
    claim = recovery.RecoveryClaim.acquire(
        tmp_path, "decode_recovery_v1__synthetic", FrozenStub
    )
    assert claim.raw_active.is_file()
    assert claim.decoded_active.is_file()
    complete = claim.close("complete")
    assert complete.is_file()
    assert not claim.raw_active.exists()
    assert not claim.decoded_active.exists()
    history = tmp_path / "decoded" / "recovery_history"
    assert len(list(history.glob("*.lock"))) == 2


def test_dual_lock_close_history_race_never_overwrites(tmp_path: Path) -> None:
    (tmp_path / "raw").mkdir()
    claim = recovery.RecoveryClaim.acquire(
        tmp_path, "decode_recovery_v1__synthetic", FrozenStub
    )
    history = tmp_path / "decoded" / "recovery_history"
    history.mkdir(parents=True)
    raced = history / "decode_recovery_v1__synthetic__raw_mutex__complete.lock"
    raced.write_bytes(b"attacker")
    with pytest.raises(recovery.RecoveryContractError, match="state/payload"):
        claim.close("complete")
    assert raced.read_bytes() == b"attacker"
    assert claim.raw_active.is_file()
    assert claim.decoded_active.is_file()


def test_inventory_digest_detects_fixed_input_change(tmp_path: Path) -> None:
    first = tmp_path / "a.bin"
    second = tmp_path / "b.bin"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    before = recovery.file_inventory([first, second], tmp_path)
    second.write_bytes(b"changed")
    after = recovery.file_inventory([first, second], tmp_path)
    assert before != after


def test_original_progress_payloads_close_every_checkpoint() -> None:
    rows, progress, failed_lock, starts = _synthetic_original_progress_evidence()
    report = recovery._validate_original_progress_payloads(
        rows, progress, failed_lock, starts
    )
    assert report["progress_checkpoint_count"] == 104
    assert report["final_completed_ranges"] == recovery.EXPECTED_ROWS
    assert report["final_network_bytes"] == recovery.EXPECTED_BYTES
    assert report["start_global_attempts_exact_1_through_10368"] is True
    assert report["attempt_count_reconstructed_from_start_event_timestamps"] is True


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    (
        ("completed_key_set_sha256", "0" * 64, "progress semantics"),
        ("network_bytes_this_invocation_so_far", -1, "progress semantics"),
        ("actual_http_attempts_cumulative", 101, "progress semantics"),
        ("unexpected_label_column", 1, "exact payload schema"),
    ),
)
def test_original_progress_payload_tamper_fails(
    field: str, replacement: object, error: str
) -> None:
    rows, progress, failed_lock, starts = _synthetic_original_progress_evidence()
    first_payload = progress[0]["payload"]
    assert isinstance(first_payload, dict)
    first_payload[field] = replacement
    with pytest.raises(recovery.RecoveryContractError, match=error):
        recovery._validate_original_progress_payloads(
            rows, progress, failed_lock, starts
        )


def test_original_progress_filename_and_failed_lock_tamper_fail() -> None:
    rows, progress, failed_lock, starts = _synthetic_original_progress_evidence()
    progress[0]["filename"] = "raw_ranges__forged__000100.json"
    with pytest.raises(recovery.RecoveryContractError, match="filename sequence"):
        recovery._validate_original_progress_payloads(
            rows, progress, failed_lock, starts
        )

    rows, progress, failed_lock, starts = _synthetic_original_progress_evidence()
    failed_lock["pid"] = recovery.FAILED_ATTEMPT_PID + 1
    with pytest.raises(recovery.RecoveryContractError, match="failed-lock semantics"):
        recovery._validate_original_progress_payloads(
            rows, progress, failed_lock, starts
        )


def test_original_progress_attempt_count_is_reconstructed_from_start_times() -> None:
    rows, progress, failed_lock, starts = _synthetic_original_progress_evidence()
    starts[100]["created_utc"] = starts[99]["created_utc"]
    with pytest.raises(recovery.RecoveryContractError, match="progress semantics"):
        recovery._validate_original_progress_payloads(
            rows, progress, failed_lock, starts
        )


def test_live_original_progress_evidence_passes_read_only_when_present() -> None:
    root = ROOT / "artifacts" / "baram2026_ncei_scada_longrun_20260810_v2"
    census = root / "census" / "field_range_census.parquet"
    if not census.is_file():
        pytest.skip("immutable Track-A raw cache is not present")
    rows = recovery.read_range_rows(
        census,
        {"size_bytes": census.stat().st_size, "sha256": recovery.sha256_file(census)},
    )
    report = recovery.validate_original_progress_and_failed_lock(
        root.resolve(),
        rows,
        set((root / "raw" / "request_events").glob("*.json")),
        list((root / "raw" / "progress").glob("*")),
        list((root / "raw" / "launch_history").glob("*")),
    )
    assert report["progress_checkpoint_count"] == 104
    assert (
        report["final_completed_key_set_sha256"]
        == "cada48ecfbe02345b2fcb27b69cc3ad017f70191321dcf7edb9aad91cb9420b4"
    )


def _independent_review_fixture() -> tuple[
    dict[str, object], dict[str, object], dict[str, object]
]:
    common = {
        "authorization": {"path": "prereg/auth.json", "size_bytes": 1, "sha256": "a" * 64},
        "incident": {"path": "incidents/i.json", "size_bytes": 2, "sha256": "b" * 64},
        "recovery_runner": {"path": "runner.py", "size_bytes": 3, "sha256": "c" * 64},
        "recovery_module": {"path": "core.py", "size_bytes": 4, "sha256": "d" * 64},
        "recovery_bootstrap": {"path": "bootstrap.py", "size_bytes": 5, "sha256": "e" * 64},
        "recovery_test": {"path": "test_core.py", "size_bytes": 6, "sha256": "f" * 64},
        "recovery_bootstrap_test": {"path": "test_boot.py", "size_bytes": 7, "sha256": "0" * 64},
        "recovery_runner_test": {"path": "test_runner.py", "size_bytes": 8, "sha256": "1" * 64},
        "postrun_auditor": {"path": "audit.py", "size_bytes": 9, "sha256": "2" * 64},
        "postrun_auditor_test": {"path": "test_audit.py", "size_bytes": 10, "sha256": "3" * 64},
        "bootstrap_trust_policy": {"explicit_trust_anchor": True},
        "raw_cache_lock": {"path": "raw.lock", "size_bytes": 11, "sha256": "4" * 64},
        "real_spawn_preflight": {"path": "preflight.json", "size_bytes": 12, "sha256": "5" * 64},
    }
    go = dict(common)
    sealer_identity = {"path": "sealer.py", "size_bytes": 13, "sha256": "6" * 64}
    sealer_test_identity = {"path": "test_sealer.py", "size_bytes": 14, "sha256": "7" * 64}
    preflight = {
        "recovery_sealer": sealer_identity,
        "recovery_sealer_test": sealer_test_identity,
    }
    review: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "TRACK_A_DECODE_RECOVERY_INDEPENDENT_REVIEW",
        "status": "PASS_NETWORK_ZERO_PROCESS_ISOLATED_RECOVERY_AUTHORIZED",
        "created_utc": "2026-08-10T18:30:00Z",
        **common,
        "recovery_sealer": sealer_identity,
        "recovery_sealer_test": sealer_test_identity,
        "max_decode_processes": recovery.MAX_DECODE_PROCESSES,
        "network_requests_allowed": 0,
        "verdict": "GO",
        "independent_checks": {
            key: True for key in recovery.INDEPENDENT_REVIEW_CHECK_KEYS
        },
    }
    return review, go, preflight


def test_independent_review_exact_schema_and_twelve_checks_pass() -> None:
    review, go, preflight = _independent_review_fixture()
    recovery.validate_independent_review_payload(review, go, preflight)
    assert len(recovery.INDEPENDENT_REVIEW_CHECK_KEYS) == 12


def test_skeletal_independent_review_fails_closed() -> None:
    _review, go, preflight = _independent_review_fixture()
    with pytest.raises(recovery.RecoveryContractError, match="exact schema"):
        recovery.validate_independent_review_payload({}, go, preflight)


@pytest.mark.parametrize(("mutation", "message"), (("missing", "exact schema"), ("extra", "exact schema"), ("false", "verdict/check")))
def test_independent_review_missing_extra_or_false_check_fails(
    mutation: str, message: str
) -> None:
    review, go, preflight = _independent_review_fixture()
    if mutation == "missing":
        review.pop("verdict")
    elif mutation == "extra":
        review["unexpected"] = True
    else:
        checks = review["independent_checks"]
        assert isinstance(checks, dict)
        checks["generic_resume_path_absent"] = False
    with pytest.raises(recovery.RecoveryContractError, match=message):
        recovery.validate_independent_review_payload(review, go, preflight)


def test_independent_review_sealer_binding_tamper_fails() -> None:
    review, go, preflight = _independent_review_fixture()
    forged = dict(review["recovery_sealer"])
    forged["sha256"] = "8" * 64
    review["recovery_sealer"] = forged
    with pytest.raises(recovery.RecoveryContractError, match="recovery_sealer mismatch"):
        recovery.validate_independent_review_payload(review, go, preflight)

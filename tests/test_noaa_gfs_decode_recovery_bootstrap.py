from __future__ import annotations

import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import pickle
import py_compile
import subprocess
import struct
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]
PILOT = (
    ROOT
    / "artifacts"
    / "baram2026_ncei_scada_research_20260810_210756"
    / "track_a"
    / "provenance"
    / "raw"
    / "f039"
    / "gfs.t12z.pgrb2.0p25.f039.HPBL_surface.grib2"
)


@pytest.fixture(autouse=True)
def _bytecode_policy(monkeypatch: pytest.MonkeyPatch):
    previous = sys.dont_write_bytecode
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.delenv("PYTHONPYCACHEPREFIX", raising=False)
    source_root = str(ROOT / "src")
    monkeypatch.setenv("PYTHONPATH", source_root)
    monkeypatch.syspath_prepend(source_root)
    sys.dont_write_bytecode = True
    try:
        yield
    finally:
        sys.dont_write_bytecode = previous


def _bootstrap_module():
    assert "src" not in sys.modules
    module = importlib.import_module("noaa_gfs_decode_recovery_bootstrap")
    assert module.__name__ == "noaa_gfs_decode_recovery_bootstrap"
    assert module.__package__ in ("", None)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, object]:
    path = path.resolve()
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _runtime_lock() -> dict[str, object]:
    package = ROOT / ".venv" / "Lib" / "site-packages"
    original_authorization = json.loads(
        (
            ROOT
            / "artifacts"
            / "baram2026_ncei_scada_longrun_20260810_v2"
            / "prereg"
            / "raw_launch_authorization_v1.json"
        ).read_text(encoding="utf-8")
    )
    return {
        "api_version": "2.47.0",
        "definitions_path": "/MEMFS/definitions",
        "library": _identity(package / "eccodes" / "eccodes.dll"),
        "memfs_library": _identity(package / "eccodes" / "eccodes_memfs.dll"),
        "python_binding": _identity(
            package / "eccodes" / "_eccodes.cp313-win_amd64.pyd"
        ),
        "gribapi_binding": _identity(package / "gribapi" / "bindings.py"),
        "eccodes_python_api": _identity(package / "eccodes" / "eccodes.py"),
        "gribapi_python_api": _identity(package / "gribapi" / "gribapi.py"),
        "gribapi_errors": _identity(package / "gribapi" / "errors.py"),
        "frozen_runner": _identity(
            ROOT / "scripts" / "run_noaa_gfs_multiseason_raw_v2.py"
        ),
        "recovery_module": _identity(ROOT / "src" / "noaa_gfs_decode_recovery.py"),
        "original_runtime_identity": original_authorization["runtime_identity"],
    }


def _bootstrap_call_identities() -> tuple[dict[str, object], ...]:
    return (
        _identity(ROOT / "src" / "noaa_gfs_decode_recovery_bootstrap.py"),
        _identity(ROOT / "scripts" / "run_noaa_gfs_multiseason_decode_recovery_v1.py"),
        _identity(ROOT / "src" / "noaa_gfs_decode_recovery.py"),
    )


def test_bootstrap_source_is_stdlib_only_and_has_no_decoder_import() -> None:
    source = ROOT / "src" / "noaa_gfs_decode_recovery_bootstrap.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imports = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    ]
    expected_imports = {
        "annotations",
        "ast",
        "Iterable",
        "Iterator",
        "Mapping",
        "FIRST_COMPLETED",
        "Future",
        "ProcessPoolExecutor",
        "wait",
        "hashlib",
        "importlib.machinery",
        "multiprocessing",
        "os",
        "Path",
        "socket",
        "sys",
        "types",
        "Any",
    }
    assert set(imports) == expected_imports
    for statement in tree.body:
        assert isinstance(
            statement,
            (
                ast.Import,
                ast.ImportFrom,
                ast.Expr,
                ast.Assign,
                ast.AnnAssign,
                ast.FunctionDef,
                ast.ClassDef,
            ),
        )
        if isinstance(statement, ast.Expr):
            assert isinstance(statement.value, ast.Constant)
            assert isinstance(statement.value.value, str)
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            value = statement.value
            assert value is None or not any(
                isinstance(node, ast.Call) for node in ast.walk(value)
            )
        if isinstance(statement, (ast.FunctionDef, ast.ClassDef)):
            assert statement.decorator_list == []


def test_top_level_alias_is_picklable_and_clean_child_avoids_src_initializer() -> None:
    bootstrap = _bootstrap_module()
    payload = pickle.dumps(bootstrap.initialize_spawned_worker)
    assert payload
    assert b"noaa_gfs_decode_recovery_bootstrap" in payload
    assert b"src.noaa_gfs_decode_recovery_bootstrap" not in payload
    code = """
import json
import pathlib
import sys
import noaa_gfs_decode_recovery_bootstrap as module
print(json.dumps({
    'name': module.__name__,
    'package': module.__package__,
    'path': str(pathlib.Path(module.__file__).resolve()),
    'src_loaded': 'src' in sys.modules,
}))
"""
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.pop("PYTHONPYCACHEPREFIX", None)
    environment["PYTHONPATH"] = str(ROOT / "src")
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
    report = json.loads(completed.stdout)
    assert report == {
        "name": "noaa_gfs_decode_recovery_bootstrap",
        "package": "",
        "path": str(
            (ROOT / "src" / "noaa_gfs_decode_recovery_bootstrap.py").resolve()
        ),
        "src_loaded": False,
    }


def test_package_qualified_bootstrap_identity_fails_closed() -> None:
    bootstrap = _bootstrap_module()
    previous_package = bootstrap.__package__
    try:
        bootstrap.__package__ = "src"
        with pytest.raises(
            bootstrap.DecodeBootstrapError, match="exact top-level module"
        ):
            bootstrap.validate_bootstrap_trust_anchor(
                _identity(ROOT / "src" / "noaa_gfs_decode_recovery_bootstrap.py")
            )
    finally:
        bootstrap.__package__ = previous_package


@pytest.mark.parametrize("boundary", ("sys_path_zero", "pythonpath", "src_loaded"))
def test_child_import_contract_tamper_fails_closed(
    boundary: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bootstrap = _bootstrap_module()
    if boundary == "sys_path_zero":
        monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path[1:]])
    elif boundary == "pythonpath":
        monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    else:
        monkeypatch.setitem(sys.modules, "src", types.ModuleType("src"))
    with pytest.raises(bootstrap.DecodeBootstrapError, match="import contract"):
        bootstrap.validate_bootstrap_trust_anchor(
            _identity(ROOT / "src" / "noaa_gfs_decode_recovery_bootstrap.py")
        )


@pytest.mark.parametrize(
    "competitor",
    (
        "noaa_gfs_decode_recovery_bootstrap",
        "NoAa_GfS_DeCoDe_ReCoVeRy_BoOtStRaP.CP313-WIN_AMD64.PYD",
    ),
)
def test_child_rejects_same_alias_package_or_extension_candidate(
    competitor: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bootstrap = _bootstrap_module()
    source = tmp_path / "noaa_gfs_decode_recovery_bootstrap.py"
    source.write_bytes(
        (ROOT / "src" / "noaa_gfs_decode_recovery_bootstrap.py").read_bytes()
    )
    candidate = tmp_path / competitor
    if "." in competitor:
        candidate.write_bytes(b"extension competitor")
    else:
        candidate.mkdir()
        (candidate / "__init__.py").write_text(
            "raise RuntimeError('package competitor')\n", encoding="utf-8"
        )
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path[1:]])
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    with pytest.raises(bootstrap.DecodeBootstrapError, match="not unique"):
        bootstrap.validate_bootstrap_trust_anchor(_identity(source))


def test_synchronized_bootstrap_wave_requires_one_task_per_process() -> None:
    bootstrap = _bootstrap_module()
    with pytest.raises(ValueError, match="one task per worker"):
        bootstrap.bounded_spawn_map(
            [],
            worker_kind="pilot",
            runtime_lock={},
            bootstrap_identity={},
            runner_identity={},
            core_identity={},
            max_workers=7,
            synchronize_first_wave=True,
        )


def test_bootstrap_cache_and_bytecode_policy_are_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bootstrap = _bootstrap_module()
    source = tmp_path / "noaa_gfs_decode_recovery_bootstrap.py"
    source.write_bytes(
        (ROOT / "src" / "noaa_gfs_decode_recovery_bootstrap.py").read_bytes()
    )
    cache = source.parent / "__pycache__"
    cache.mkdir()
    poisoned = cache / "noaa_gfs_decode_recovery_bootstrap.cpython-313.pyc"
    poisoned.write_bytes(b"poison")
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path])
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    with pytest.raises(bootstrap.DecodeBootstrapError, match="cache must be absent"):
        bootstrap.validate_bootstrap_trust_anchor(_identity(source))
    poisoned.unlink()
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", str(tmp_path / "external-cache"))
    with pytest.raises(bootstrap.DecodeBootstrapError, match="cache prefix"):
        bootstrap.validate_bootstrap_trust_anchor(_identity(source))


def test_poisoned_timestamp_pyc_is_ignored_for_verified_source(
    tmp_path: Path,
) -> None:
    bootstrap = _bootstrap_module()
    source = tmp_path / "noaa_gfs_decode_recovery.py"
    malicious = "VALUE = 'evil'\n"
    expected = "VALUE = 'good'\n"
    assert len(malicious) == len(expected)
    source.write_text(malicious, encoding="utf-8", newline="\n")
    fixed_time = 1_700_000_000
    os.utime(source, (fixed_time, fixed_time))
    cache = Path(py_compile.compile(str(source), doraise=True))
    assert cache.is_file()
    source.write_text(expected, encoding="utf-8", newline="\n")
    os.utime(source, (fixed_time, fixed_time))
    module_name = "src.noaa_gfs_decode_recovery"
    prior = sys.modules.pop(module_name, None)
    try:
        module = bootstrap.load_decoder_core_preexecution(_identity(source))
        assert module.VALUE == "good"
    finally:
        sys.modules.pop(module_name, None)
        if prior is not None:
            sys.modules[module_name] = prior


def test_bad_core_hash_fails_before_source_side_effect(tmp_path: Path) -> None:
    bootstrap = _bootstrap_module()
    sentinel = tmp_path / "executed.txt"
    source = tmp_path / "noaa_gfs_decode_recovery.py"
    source.write_text(
        f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('bad')\n",
        encoding="utf-8",
        newline="\n",
    )
    record = _identity(source)
    record["sha256"] = "0" * 64
    module_name = "src.noaa_gfs_decode_recovery"
    prior = sys.modules.pop(module_name, None)
    try:
        with pytest.raises(bootstrap.DecodeBootstrapError, match="hash mismatch"):
            bootstrap.load_decoder_core_preexecution(record)
        assert not sentinel.exists()
    finally:
        sys.modules.pop(module_name, None)
        if prior is not None:
            sys.modules[module_name] = prior


@pytest.mark.skipif(not PILOT.is_file(), reason="immutable real GRIB pilot absent")
def test_real_pilot_uses_exactly_seven_stdlib_bootstrap_processes() -> None:
    bootstrap = _bootstrap_module()
    bootstrap_id, runner_id, core_id = _bootstrap_call_identities()
    report = bootstrap.run_real_spawn_pilot(
        _identity(PILOT),
        _runtime_lock(),
        bootstrap_id,
        runner_id,
        core_id,
        workers=7,
        repetitions=70,
    )
    assert report["status"] == "PASS_REAL_GRIB_STDLIB_BOOTSTRAP_PROCESS_ISOLATION"
    assert report["workers_requested"] == 7
    assert len(report["worker_pids_observed"]) == 7
    assert report["tasks"] == 70
    assert report["first_wave_synchronized"] is True
    assert report["network_requests"] == 0
    assert report["reference"]["metadata"]["forecastTime"] == 39


@pytest.mark.skipif(not PILOT.is_file(), reason="immutable real GRIB pilot absent")
def test_production_worker_uses_stdlib_bootstrap_and_exact_source() -> None:
    bootstrap = _bootstrap_module()
    coordinate_path = (
        ROOT
        / "artifacts"
        / "baram2026_ncei_scada_longrun_20260810_v2"
        / "prereg"
        / "authoritative_turbine_coordinate_lock_v1.json"
    )
    sites = json.loads(coordinate_path.read_text(encoding="utf-8"))["sites"]
    task = {
        "result": _identity(PILOT),
        "row": {
            "run_init_utc": "2023-07-01T12:00:00Z",
            "valid_time_utc": "2023-07-03T03:00:00Z",
            "forecast_hour": 39,
            "variable": "HPBL",
            "level": "surface",
        },
        "sites": sites,
    }
    bootstrap_id, runner_id, core_id = _bootstrap_call_identities()
    rows, audit = bootstrap.bounded_spawn_map(
        [task for _ in range(70)],
        worker_kind="decode",
        runtime_lock=_runtime_lock(),
        bootstrap_identity=bootstrap_id,
        runner_identity=runner_id,
        core_identity=core_id,
        max_workers=7,
        synchronize_first_wave=True,
    )
    assert audit.requested_workers == 7
    assert len(audit.observed_worker_pids) == 7
    assert audit.completed_tasks == 70
    assert audit.first_wave_synchronized is True
    assert {row["feature"] for row in rows} == {"HPBL_surface"}
    reference = rows[0]["site_values"]
    assert all(row["site_values"] == reference for row in rows)
    payload = b"".join(struct.pack("<d", value) for value in reference)
    assert hashlib.sha256(payload).hexdigest() == (
        "edeca661e457c32431218ccb50c5dbf0c428905a36e03c28cfe4d76d32ddcc0b"
    )

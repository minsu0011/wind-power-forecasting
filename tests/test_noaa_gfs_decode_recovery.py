from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import shutil
import struct

import pytest

from src.noaa_gfs_decode_recovery import (
    DecodeRecoveryError,
    EXPECTED_DEFINITION_PATH,
    _require_exact_file,
    bounded_spawn_map,
    decode_task_worker,
    pilot_decode_worker,
    run_real_spawn_pilot,
    validate_eccodes_runtime,
)


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
        "definitions_path": EXPECTED_DEFINITION_PATH,
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
        "recovery_module": _identity(
            ROOT / "src" / "noaa_gfs_decode_recovery.py"
        ),
        "original_runtime_identity": original_authorization["runtime_identity"],
    }


def test_recovery_source_has_no_thread_executor() -> None:
    source_path = ROOT / "src" / "noaa_gfs_decode_recovery.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
    }
    attributes = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
    }
    assert "ThreadPoolExecutor" not in names | attributes
    assert "threading" not in names | attributes
    assert "ProcessPoolExecutor" in names | attributes
    forbidden_imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    } & {"requests", "urllib", "httpx", "aiohttp", "boto3"}
    assert forbidden_imports == set()


def test_synchronized_first_wave_requires_one_task_per_requested_process() -> None:
    with pytest.raises(ValueError, match="one task per worker"):
        bounded_spawn_map(
            [],
            worker=pilot_decode_worker,
            runtime_lock={},
            max_workers=7,
            synchronize_first_wave=True,
        )


def test_runtime_lock_is_exact_memfs_binary_wheel() -> None:
    report = validate_eccodes_runtime(_runtime_lock())
    assert report["api_version"] == "2.47.0"
    assert report["definitions_path"] == "/MEMFS/definitions"
    assert report["library_path"].endswith("eccodes.dll")


def test_runtime_lock_rejects_memfs_tamper() -> None:
    runtime = _runtime_lock()
    runtime["memfs_library"] = dict(runtime["memfs_library"])
    runtime["memfs_library"]["sha256"] = "0" * 64
    with pytest.raises(DecodeRecoveryError, match="MEMFS library hash mismatch"):
        validate_eccodes_runtime(runtime)


def test_runtime_lock_rejects_recovery_module_tamper() -> None:
    runtime = _runtime_lock()
    runtime["recovery_module"] = dict(runtime["recovery_module"])
    runtime["recovery_module"]["sha256"] = "f" * 64
    with pytest.raises(DecodeRecoveryError, match="decode recovery module hash mismatch"):
        validate_eccodes_runtime(runtime)


def test_identity_record_rejects_relative_path() -> None:
    with pytest.raises(DecodeRecoveryError, match="not absolute"):
        _require_exact_file(
            {"path": "eccodes.dll", "size_bytes": 1, "sha256": "0" * 64},
            "relative",
        )


def test_runtime_lock_rejects_same_hash_library_copy(tmp_path: Path) -> None:
    runtime = _runtime_lock()
    original = Path(runtime["library"]["path"])
    copied = tmp_path / "eccodes.dll"
    shutil.copyfile(original, copied)
    runtime["library"] = _identity(copied)
    with pytest.raises(DecodeRecoveryError, match="loaded ecCodes library path mismatch"):
        validate_eccodes_runtime(runtime)


def test_runtime_lock_rejects_lexical_symlink(tmp_path: Path) -> None:
    if not hasattr(Path, "symlink_to"):
        pytest.skip("path symlink API unavailable")
    runtime = _runtime_lock()
    link = tmp_path / "eccodes-link.dll"
    try:
        link.symlink_to(Path(runtime["library"]["path"]))
    except OSError:
        pytest.skip("Windows symlink privilege unavailable")
    runtime["library"] = {
        "path": str(link),
        "size_bytes": Path(runtime["library"]["path"]).stat().st_size,
        "sha256": runtime["library"]["sha256"],
    }
    with pytest.raises(DecodeRecoveryError, match="symlink/junction"):
        validate_eccodes_runtime(runtime)


def test_lexical_symlink_check_precedes_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    record = _runtime_lock()["library"]
    target = Path(record["path"])
    original = Path.is_symlink

    def marked(path: Path) -> bool:
        return path == target or original(path)

    monkeypatch.setattr(Path, "is_symlink", marked)
    with pytest.raises(DecodeRecoveryError, match="symlink/junction"):
        _require_exact_file(record, "marked lexical path")


@pytest.mark.skipif(not PILOT.is_file(), reason="immutable real GRIB pilot absent")
def test_real_grib_cold_start_passes_in_exactly_seven_spawn_processes() -> None:
    report = run_real_spawn_pilot(
        _identity(PILOT),
        _runtime_lock(),
        workers=7,
        repetitions=70,
    )
    assert report["status"] == "PASS_REAL_GRIB_SPAWN_PROCESS_ISOLATION"
    assert report["workers_requested"] == 7
    assert len(report["worker_pids_observed"]) == 7
    assert report["tasks"] == 70
    assert report["first_wave_synchronized"] is True
    assert report["definitions_path"] == "/MEMFS/definitions"
    reference = report["reference"]
    assert reference["metadata"]["dataDate"] == 20230701
    assert reference["metadata"]["dataTime"] == 1200
    assert reference["metadata"]["forecastTime"] == 39
    assert reference["metadata"]["validityDate"] == 20230703
    assert reference["metadata"]["validityTime"] == 300
    assert reference["metadata"]["gridType"] == "regular_ll"
    assert reference["metadata"]["Ni"] == 1440
    assert reference["metadata"]["Nj"] == 721
    assert reference["value_count"] == 1_038_240
    assert reference["value_min"] == 7.358119506835938
    assert reference["value_max"] == 3918.078119506836


@pytest.mark.skipif(not PILOT.is_file(), reason="immutable real GRIB pilot absent")
def test_production_decode_worker_real_grib_passes_in_seven_spawn_processes() -> None:
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
    rows, audit = bounded_spawn_map(
        [task for _ in range(70)],
        worker=decode_task_worker,
        runtime_lock=_runtime_lock(),
        max_workers=7,
        synchronize_first_wave=True,
    )
    assert audit.requested_workers == 7
    assert len(audit.observed_worker_pids) == 7
    assert audit.completed_tasks == 70
    assert audit.first_wave_synchronized is True
    assert {row["feature"] for row in rows} == {"HPBL_surface"}
    reference = rows[0]["site_values"]
    assert len(reference) == 17
    assert all(row["site_values"] == reference for row in rows)
    payload = b"".join(struct.pack("<d", value) for value in reference)
    assert hashlib.sha256(payload).hexdigest() == (
        "edeca661e457c32431218ccb50c5dbf0c428905a36e03c28cfe4d76d32ddcc0b"
    )


@pytest.mark.skipif(not PILOT.is_file(), reason="immutable real GRIB pilot absent")
def test_production_worker_rejects_raw_identity_before_decode() -> None:
    coordinate_path = (
        ROOT
        / "artifacts"
        / "baram2026_ncei_scada_longrun_20260810_v2"
        / "prereg"
        / "authoritative_turbine_coordinate_lock_v1.json"
    )
    sites = json.loads(coordinate_path.read_text(encoding="utf-8"))["sites"]
    bad_identity = _identity(PILOT)
    bad_identity["sha256"] = "0" * 64
    task = {
        "result": bad_identity,
        "row": {
            "run_init_utc": "2023-07-01T12:00:00Z",
            "valid_time_utc": "2023-07-03T03:00:00Z",
            "forecast_hour": 39,
            "variable": "HPBL",
            "level": "surface",
        },
        "sites": sites,
    }
    with pytest.raises(DecodeRecoveryError, match="decode task raw GRIB hash mismatch"):
        bounded_spawn_map(
            [task],
            worker=decode_task_worker,
            runtime_lock=_runtime_lock(),
            max_workers=1,
        )

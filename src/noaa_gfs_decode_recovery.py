"""Process-isolated ecCodes decoding for the frozen NOAA GFS recovery.

The Windows binary wheel for ecCodes 2.47.0 exposes definitions through the
``/MEMFS/definitions`` virtual path.  Cold initialization of that parser is
not thread-safe: simultaneous first calls from Python threads can corrupt the
Flex scanner.  This module deliberately provides no threaded decoder.  Every
decode worker is a Windows ``spawn`` process and decodes messages serially.

The helpers are target-free and network-free.  They do not know about labels,
competition predictions, 2024/2025 arrays, or submission files.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
import hashlib
import importlib
import importlib.machinery
import multiprocessing
import os
from pathlib import Path
import platform
import socket
import sys
import types
from typing import Any


MAX_DECODE_PROCESSES = 7
EXPECTED_DEFINITION_PATH = "/MEMFS/definitions"


class DecodeRecoveryError(RuntimeError):
    """Fail-closed error raised by the decode-only recovery path."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _network_forbidden(*_args: Any, **_kwargs: Any) -> Any:
    raise DecodeRecoveryError("network access is forbidden during decode recovery")


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
        raise DecodeRecoveryError(
            f"network/socket audit event is forbidden during decode recovery: {event}"
        )


def install_network_deny_guard() -> None:
    """Make accidental network use fail before a socket can be created.

    Replacing ``socket.socket`` with a function breaks the standard library's
    ``ssl.SSLSocket`` class definition during an otherwise inert import.  A
    Python audit hook blocks actual socket construction/use without altering
    the class hierarchy, while the convenience connector is also replaced.
    """

    global _NETWORK_GUARD_INSTALLED
    if not _NETWORK_GUARD_INSTALLED:
        sys.addaudithook(_network_audit_guard)
        _NETWORK_GUARD_INSTALLED = True
    socket.create_connection = _network_forbidden  # type: ignore[assignment]


def _require_exact_file(record: Mapping[str, Any], label: str) -> Path:
    expected_keys = {"path", "size_bytes", "sha256"}
    if set(record) != expected_keys:
        raise DecodeRecoveryError(f"{label} identity schema mismatch")
    raw_path = Path(str(record["path"]))
    if not raw_path.is_absolute():
        raise DecodeRecoveryError(f"{label} path is not absolute")
    lexical = Path(os.path.abspath(str(raw_path)))
    # Reject lexical symlinks/junctions before resolving them.  Checking only
    # the resolved leaf would silently accept a symlinked input or parent.
    candidates = (lexical, *lexical.parents)
    for candidate in candidates:
        if candidate.is_symlink() or (
            hasattr(candidate, "is_junction") and candidate.is_junction()
        ):
            raise DecodeRecoveryError(f"{label} symlink/junction path is forbidden: {candidate}")
    try:
        path = lexical.resolve(strict=True)
    except FileNotFoundError as exc:
        raise DecodeRecoveryError(f"{label} is absent: {lexical}") from exc
    if path != lexical or not path.is_file():
        raise DecodeRecoveryError(f"{label} is absent or a symlink: {path}")
    if path.stat().st_size != int(record["size_bytes"]):
        raise DecodeRecoveryError(f"{label} size mismatch: {path}")
    if sha256_file(path) != str(record["sha256"]):
        raise DecodeRecoveryError(f"{label} hash mismatch: {path}")
    return path


def _runtime_package_identity(
    record: Mapping[str, Any], label: str
) -> tuple[Path, str]:
    expected = {
        "module_file",
        "module_file_size_bytes",
        "module_file_sha256",
        "version",
    }
    if set(record) != expected:
        raise DecodeRecoveryError(f"{label} original runtime schema mismatch")
    path = _require_exact_file(
        {
            "path": record["module_file"],
            "size_bytes": record["module_file_size_bytes"],
            "sha256": record["module_file_sha256"],
        },
        f"{label} original runtime module",
    )
    spec = importlib.machinery.PathFinder.find_spec(label, sys.path)
    if spec is None or spec.origin is None or Path(spec.origin).resolve() != path:
        raise DecodeRecoveryError(f"{label} import origin differs before execution")
    return path, str(record["version"])


def validate_original_runtime_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "packages",
        "platform",
        "python_executable",
        "python_executable_sha256",
        "python_executable_size_bytes",
        "python_version",
    }
    if set(payload) != expected:
        raise DecodeRecoveryError("original runtime identity schema mismatch")
    executable = _require_exact_file(
        {
            "path": payload["python_executable"],
            "size_bytes": payload["python_executable_size_bytes"],
            "sha256": payload["python_executable_sha256"],
        },
        "frozen Python executable",
    )
    if executable != Path(sys.executable).resolve():
        raise DecodeRecoveryError("active Python executable differs from frozen runtime")
    if str(payload["python_version"]) != platform.python_version():
        raise DecodeRecoveryError("active Python version differs from frozen runtime")
    if str(payload["platform"]) != platform.platform():
        raise DecodeRecoveryError("active platform differs from frozen runtime")
    packages = payload["packages"]
    if not isinstance(packages, dict) or set(packages) != {
        "eccodes",
        "numpy",
        "pandas",
        "pyarrow",
    }:
        raise DecodeRecoveryError("original runtime package set mismatch")

    # Verify every package file and import origin before executing any package.
    expected_modules = {
        name: _runtime_package_identity(packages[name], name)
        for name in ("eccodes", "numpy", "pandas", "pyarrow")
    }
    observed: dict[str, Any] = {}
    for name in ("eccodes", "numpy", "pandas", "pyarrow"):
        path, expected_version = expected_modules[name]
        module = importlib.import_module(name)
        if Path(str(module.__file__)).resolve() != path:
            raise DecodeRecoveryError(f"imported {name} path mismatch")
        if str(module.__version__) != expected_version:
            raise DecodeRecoveryError(f"imported {name} version mismatch")
        if sha256_file(path) != str(packages[name]["module_file_sha256"]):
            raise DecodeRecoveryError(f"{name} changed during import")
        observed[name] = {"path": str(path), "version": expected_version}
    return {
        "python_executable": str(executable),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": observed,
    }


def validate_eccodes_runtime(runtime_lock: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact ecCodes runtime inside one spawned process."""

    expected_keys = {
        "api_version",
        "definitions_path",
        "library",
        "memfs_library",
        "python_binding",
        "gribapi_binding",
        "eccodes_python_api",
        "gribapi_python_api",
        "gribapi_errors",
        "frozen_runner",
        "recovery_module",
        "original_runtime_identity",
    }
    if set(runtime_lock) != expected_keys:
        raise DecodeRecoveryError("ecCodes runtime-lock schema mismatch")
    for env_name in (
        "ECCODES_DEFINITION_PATH",
        "ECCODES_SAMPLES_PATH",
        "GRIB_DEFINITION_PATH",
        "GRIB_SAMPLES_PATH",
    ):
        if os.environ.get(env_name) not in (None, ""):
            raise DecodeRecoveryError(f"unexpected ecCodes environment override: {env_name}")

    # Resolve and hash every executable/imported identity before any ecCodes or
    # gribapi module code runs.  Only then are imports permitted.
    library_path = _require_exact_file(runtime_lock["library"], "ecCodes library")
    memfs_path = _require_exact_file(
        runtime_lock["memfs_library"], "ecCodes MEMFS library"
    )
    python_binding_path = _require_exact_file(
        runtime_lock["python_binding"], "ecCodes Python binding"
    )
    gribapi_binding_path = _require_exact_file(
        runtime_lock["gribapi_binding"], "gribapi binding"
    )
    eccodes_python_api_path = _require_exact_file(
        runtime_lock["eccodes_python_api"], "ecCodes Python API"
    )
    gribapi_python_api_path = _require_exact_file(
        runtime_lock["gribapi_python_api"], "gribapi Python API"
    )
    gribapi_errors_path = _require_exact_file(
        runtime_lock["gribapi_errors"], "gribapi error mapping"
    )
    frozen_runner_path = _require_exact_file(
        runtime_lock["frozen_runner"], "frozen raw runner"
    )
    recovery_module_path = _require_exact_file(
        runtime_lock["recovery_module"], "decode recovery module"
    )
    original_runtime_report = validate_original_runtime_identity(
        runtime_lock["original_runtime_identity"]
    )

    eccodes_spec = importlib.machinery.PathFinder.find_spec("eccodes", sys.path)
    gribapi_spec = importlib.machinery.PathFinder.find_spec("gribapi", sys.path)
    if (
        eccodes_spec is None
        or not eccodes_spec.submodule_search_locations
        or gribapi_spec is None
        or not gribapi_spec.submodule_search_locations
    ):
        raise DecodeRecoveryError("ecCodes/gribapi package import origins are absent")
    eccodes_dir = Path(next(iter(eccodes_spec.submodule_search_locations))).resolve()
    gribapi_dir = Path(next(iter(gribapi_spec.submodule_search_locations))).resolve()
    if eccodes_python_api_path.parent != eccodes_dir:
        raise DecodeRecoveryError("ecCodes Python API is outside the bound package")
    if gribapi_python_api_path.parent != gribapi_dir:
        raise DecodeRecoveryError("gribapi Python API is outside the bound package")

    import eccodes
    import eccodes._eccodes as eccodes_extension
    import eccodes.eccodes as eccodes_python_api
    import gribapi.bindings as gribapi_bindings
    import gribapi.errors as gribapi_errors
    import gribapi.gribapi as gribapi_python_api

    api_version = str(eccodes.codes_get_api_version())
    definitions_path = str(eccodes.codes_definition_path())
    observed_library = Path(str(eccodes.codes_get_library_path())).resolve()
    observed_memfs = observed_library.with_name("eccodes_memfs.dll")
    observed_python_binding = Path(str(eccodes_extension.__file__)).resolve()
    observed_gribapi_binding = Path(str(gribapi_bindings.__file__)).resolve()
    observed_eccodes_python_api = Path(str(eccodes_python_api.__file__)).resolve()
    observed_gribapi_python_api = Path(str(gribapi_python_api.__file__)).resolve()
    observed_gribapi_errors = Path(str(gribapi_errors.__file__)).resolve()
    if api_version != str(runtime_lock["api_version"]):
        raise DecodeRecoveryError("ecCodes API version mismatch")
    if definitions_path != str(runtime_lock["definitions_path"]):
        raise DecodeRecoveryError("ecCodes definitions path mismatch")
    if definitions_path != EXPECTED_DEFINITION_PATH:
        raise DecodeRecoveryError("ecCodes definitions are not the frozen MEMFS path")
    if observed_library != library_path:
        raise DecodeRecoveryError("loaded ecCodes library path mismatch")
    if observed_memfs != memfs_path:
        raise DecodeRecoveryError("loaded ecCodes MEMFS sibling path mismatch")
    if observed_python_binding != python_binding_path:
        raise DecodeRecoveryError("imported ecCodes Python binding path mismatch")
    if observed_gribapi_binding != gribapi_binding_path:
        raise DecodeRecoveryError("imported gribapi binding path mismatch")
    if observed_eccodes_python_api != eccodes_python_api_path:
        raise DecodeRecoveryError("imported ecCodes Python API path mismatch")
    if observed_gribapi_python_api != gribapi_python_api_path:
        raise DecodeRecoveryError("imported gribapi Python API path mismatch")
    if observed_gribapi_errors != gribapi_errors_path:
        raise DecodeRecoveryError("imported gribapi error mapping path mismatch")
    if Path(__file__).resolve() != recovery_module_path:
        raise DecodeRecoveryError("spawned decode recovery module path mismatch")
    if sha256_file(Path(__file__).resolve()) != str(
        runtime_lock["recovery_module"]["sha256"]
    ):
        raise DecodeRecoveryError("spawned decode recovery module hash mismatch")
    return {
        "pid": os.getpid(),
        "api_version": api_version,
        "definitions_path": definitions_path,
        "library_path": str(observed_library),
        "memfs_library_path": str(observed_memfs),
        "python_binding_path": str(observed_python_binding),
        "gribapi_binding_path": str(observed_gribapi_binding),
        "eccodes_python_api_path": str(observed_eccodes_python_api),
        "gribapi_python_api_path": str(observed_gribapi_python_api),
        "gribapi_errors_path": str(observed_gribapi_errors),
        "frozen_runner_path": str(frozen_runner_path),
        "frozen_runner_sha256": str(runtime_lock["frozen_runner"]["sha256"]),
        "recovery_module_path": str(recovery_module_path),
        "recovery_module_sha256": str(runtime_lock["recovery_module"]["sha256"]),
        "original_runtime_identity": original_runtime_report,
    }


_WORKER_RUNTIME: dict[str, Any] | None = None
_WORKER_RUNTIME_LOCK: dict[str, Any] | None = None
_FROZEN_RUNNER_MODULE: Any | None = None
_WORKER_FIRST_WAVE_BARRIER: Any | None = None
_WORKER_FIRST_TASK_SYNCHRONIZED = False


def initialize_decode_worker(
    runtime_lock: Mapping[str, Any], first_wave_barrier: Any | None = None
) -> None:
    """Spawn-process initializer; each worker owns one serial ecCodes context."""

    global _WORKER_RUNTIME, _WORKER_RUNTIME_LOCK, _FROZEN_RUNNER_MODULE
    global _WORKER_FIRST_WAVE_BARRIER, _WORKER_FIRST_TASK_SYNCHRONIZED
    install_network_deny_guard()
    _WORKER_RUNTIME = validate_eccodes_runtime(runtime_lock)
    _WORKER_RUNTIME_LOCK = dict(runtime_lock)
    _FROZEN_RUNNER_MODULE = None
    _WORKER_FIRST_WAVE_BARRIER = first_wave_barrier
    _WORKER_FIRST_TASK_SYNCHRONIZED = False


def _synchronize_first_worker_task() -> None:
    """Block each process's first task until every requested worker is live."""

    global _WORKER_FIRST_TASK_SYNCHRONIZED
    if _WORKER_FIRST_WAVE_BARRIER is not None and not _WORKER_FIRST_TASK_SYNCHRONIZED:
        try:
            _WORKER_FIRST_WAVE_BARRIER.wait(timeout=120)
        except BaseException as exc:
            raise DecodeRecoveryError(
                "synchronized first-worker-wave barrier failed"
            ) from exc
        _WORKER_FIRST_TASK_SYNCHRONIZED = True


def _import_frozen_runner_after_rehash() -> Any:
    """Rehash the frozen producer immediately before each worker imports it."""

    global _FROZEN_RUNNER_MODULE
    if _FROZEN_RUNNER_MODULE is not None:
        return _FROZEN_RUNNER_MODULE
    if _WORKER_RUNTIME_LOCK is None:
        raise DecodeRecoveryError("worker runtime lock is absent")
    expected_path = _require_exact_file(
        _WORKER_RUNTIME_LOCK["frozen_runner"], "frozen raw runner before import"
    )
    if "scripts.run_noaa_gfs_multiseason_raw_v2" in sys.modules:
        raise DecodeRecoveryError("frozen raw runner was imported before worker rehash")
    record = _WORKER_RUNTIME_LOCK["frozen_runner"]
    source = expected_path.read_bytes()
    if len(source) != int(record["size_bytes"]) or hashlib.sha256(source).hexdigest() != str(
        record["sha256"]
    ):
        raise DecodeRecoveryError("frozen raw runner changed before worker compile")
    module_name = "scripts.run_noaa_gfs_multiseason_raw_v2"
    code = compile(source, str(expected_path), "exec", dont_inherit=True)
    module = types.ModuleType(module_name)
    module.__file__ = str(expected_path)
    module.__package__ = module_name.rpartition(".")[0]
    module.__loader__ = None
    module.__cached__ = None
    module.__spec__ = importlib.machinery.ModuleSpec(
        module_name, loader=None, origin=str(expected_path)
    )
    sys.modules[module_name] = module
    try:
        exec(code, module.__dict__)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    imported_path = Path(str(module.__file__)).resolve()
    if imported_path != expected_path:
        raise DecodeRecoveryError("executed frozen raw runner path mismatch")
    if sha256_file(imported_path) != str(record["sha256"]):
        raise DecodeRecoveryError("frozen raw runner changed across worker execution")
    _FROZEN_RUNNER_MODULE = module
    return module


def decode_task_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    """Decode one message with the frozen producer's exact metadata semantics."""

    if _WORKER_RUNTIME is None:
        raise DecodeRecoveryError("decode worker was not initialized")
    _synchronize_first_worker_task()
    if set(task) != {"result", "row", "sites"}:
        raise DecodeRecoveryError("decode task schema mismatch")

    # Importing the frozen producer is safe here: only its pure decode function
    # is called.  The recovery authorization independently binds its file hash.
    frozen = _import_frozen_runner_after_rehash()

    result = dict(task["result"])
    raw_path = _require_exact_file(result, "decode task raw GRIB")
    result = {"path": raw_path}
    decoded = frozen.decode_message(result, dict(task["row"]), list(task["sites"]))
    # A second complete rehash closes in-process mutation during the decode.
    _require_exact_file(task["result"], "decoded task raw GRIB postcheck")
    return {
        "feature": str(decoded["feature"]),
        "site_values": [float(value) for value in decoded["site_values"]],
        "worker_runtime": dict(_WORKER_RUNTIME),
    }


def pilot_decode_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    """Decode a real immutable pilot in a spawned process for preflight."""

    if _WORKER_RUNTIME is None:
        raise DecodeRecoveryError("pilot worker was not initialized")
    _synchronize_first_worker_task()
    expected_keys = {"path", "sha256", "size_bytes"}
    if set(task) != expected_keys:
        raise DecodeRecoveryError("pilot task schema mismatch")
    path = _require_exact_file(task, "real GRIB pilot")

    import eccodes

    with path.open("rb") as stream:
        handle = eccodes.codes_grib_new_from_file(stream)
        if handle is None:
            raise DecodeRecoveryError("ecCodes returned no pilot handle")
        try:
            metadata = {
                key: eccodes.codes_get(handle, key)
                for key in (
                    "discipline",
                    "parameterCategory",
                    "parameterNumber",
                    "shortName",
                    "typeOfLevel",
                    "level",
                    "dataDate",
                    "dataTime",
                    "forecastTime",
                    "validityDate",
                    "validityTime",
                    "gridType",
                    "Ni",
                    "Nj",
                    "latitudeOfFirstGridPointInDegrees",
                    "longitudeOfFirstGridPointInDegrees",
                    "iDirectionIncrementInDegrees",
                    "jDirectionIncrementInDegrees",
                    "iScansNegatively",
                    "jScansPositively",
                    "jPointsAreConsecutive",
                    "alternativeRowScanning",
                )
            }
            values = eccodes.codes_get_array(handle, "values")
        finally:
            eccodes.codes_release(handle)
        second = eccodes.codes_grib_new_from_file(stream)
        if second is not None:
            try:
                raise DecodeRecoveryError("pilot contains more than one GRIB message")
            finally:
                eccodes.codes_release(second)

    if (
        metadata["gridType"] != "regular_ll"
        or int(metadata["Ni"]) != 1440
        or int(metadata["Nj"]) != 721
        or float(metadata["latitudeOfFirstGridPointInDegrees"]) != 90.0
        or float(metadata["longitudeOfFirstGridPointInDegrees"]) != 0.0
        or float(metadata["iDirectionIncrementInDegrees"]) != 0.25
        or float(metadata["jDirectionIncrementInDegrees"]) != 0.25
        or int(metadata["iScansNegatively"]) != 0
        or int(metadata["jScansPositively"]) != 0
        or int(metadata["jPointsAreConsecutive"]) != 0
        or int(metadata["alternativeRowScanning"]) != 0
    ):
        raise DecodeRecoveryError("real pilot grid metadata mismatch")
    return {
        "worker_runtime": dict(_WORKER_RUNTIME),
        "metadata": metadata,
        "value_count": int(len(values)),
        "value_min": float(values.min()),
        "value_max": float(values.max()),
    }


@dataclass(frozen=True)
class ProcessDecodeAudit:
    requested_workers: int
    observed_worker_pids: tuple[int, ...]
    completed_tasks: int
    first_wave_synchronized: bool


def bounded_spawn_map(
    tasks: Iterable[Mapping[str, Any]],
    *,
    worker: Any,
    runtime_lock: Mapping[str, Any],
    max_workers: int = MAX_DECODE_PROCESSES,
    synchronize_first_wave: bool = False,
) -> tuple[list[dict[str, Any]], ProcessDecodeAudit]:
    """Run at most ``max_workers`` process-isolated tasks, fail-fast.

    Only one wave is submitted at a time.  A failed completed batch is scanned
    before any replacement task is submitted, preventing post-failure work.
    """

    if not 1 <= int(max_workers) <= MAX_DECODE_PROCESSES:
        raise ValueError("decode worker count must be within 1..7")
    if synchronize_first_wave:
        synchronized_tasks = list(tasks)
        if len(synchronized_tasks) < int(max_workers):
            raise ValueError("synchronized decode wave needs at least one task per worker")
        iterator: Iterator[Mapping[str, Any]] = iter(synchronized_tasks)
    else:
        iterator = iter(tasks)
    context = multiprocessing.get_context("spawn")
    first_wave_barrier = (
        context.Barrier(int(max_workers)) if synchronize_first_wave else None
    )
    executor = ProcessPoolExecutor(
        max_workers=int(max_workers),
        mp_context=context,
        initializer=initialize_decode_worker,
        initargs=(dict(runtime_lock), first_wave_barrier),
    )
    pending: dict[Future[dict[str, Any]], int] = {}
    next_index = 0
    output: dict[int, dict[str, Any]] = {}
    failure: BaseException | None = None

    def submit_one() -> bool:
        nonlocal next_index
        try:
            task = next(iterator)
        except StopIteration:
            return False
        future = executor.submit(worker, dict(task))
        pending[future] = next_index
        next_index += 1
        return True

    try:
        for _ in range(int(max_workers)):
            if not submit_one():
                break
        while pending:
            done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            # Inspect every completed future before submitting replacement work.
            for future in done:
                exception = future.exception()
                if exception is not None and failure is None:
                    failure = exception
            if failure is not None:
                for future in pending:
                    if future not in done:
                        future.cancel()
                raise failure
            for future in sorted(done, key=lambda value: pending[value]):
                index = pending.pop(future)
                output[index] = future.result()
            for _ in range(len(done)):
                if not submit_one():
                    break
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    ordered = [output[index] for index in range(next_index)]
    pids = tuple(sorted({int(row["worker_runtime"]["pid"]) for row in ordered}))
    return ordered, ProcessDecodeAudit(
        requested_workers=int(max_workers),
        observed_worker_pids=pids,
        completed_tasks=len(ordered),
        first_wave_synchronized=bool(synchronize_first_wave),
    )


def run_real_spawn_pilot(
    pilot_identity: Mapping[str, Any],
    runtime_lock: Mapping[str, Any],
    *,
    workers: int = MAX_DECODE_PROCESSES,
    repetitions: int = 28,
) -> dict[str, Any]:
    """Require a real-GRIB, multi-process cold-start preflight."""

    if repetitions < workers * 2:
        raise ValueError("pilot must schedule at least two tasks per worker")
    rows, audit = bounded_spawn_map(
        [dict(pilot_identity) for _ in range(repetitions)],
        worker=pilot_decode_worker,
        runtime_lock=runtime_lock,
        max_workers=workers,
        synchronize_first_wave=True,
    )
    reference = {key: value for key, value in rows[0].items() if key != "worker_runtime"}
    if any(
        {key: value for key, value in row.items() if key != "worker_runtime"}
        != reference
        for row in rows[1:]
    ):
        raise DecodeRecoveryError("spawned real-pilot decodes are not identical")
    if len(audit.observed_worker_pids) != workers:
        raise DecodeRecoveryError(
            f"real pilot did not exercise all {workers} spawned processes"
        )
    return {
        "status": "PASS_REAL_GRIB_SPAWN_PROCESS_ISOLATION",
        "workers_requested": workers,
        "worker_pids_observed": list(audit.observed_worker_pids),
        "tasks": audit.completed_tasks,
        "first_wave_synchronized": audit.first_wave_synchronized,
        "definitions_path": EXPECTED_DEFINITION_PATH,
        "reference": reference,
    }

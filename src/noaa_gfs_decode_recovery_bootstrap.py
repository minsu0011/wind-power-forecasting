"""Stdlib-only spawned-process bootstrap for Track-A ecCodes recovery.

Windows ``spawn`` imports the module that owns a ProcessPool callable before
the pool initializer runs.  This module therefore deliberately imports only
the Python standard library and performs no work at import time.  Its
initializer installs the network guard, verifies its own file plus the bound
runner and decoder-core identities, and only then executes the decoder core.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Iterator, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
import hashlib
import importlib.machinery
import multiprocessing
import os
from pathlib import Path
import socket
import sys
import types
from typing import Any


MAX_DECODE_PROCESSES = 7
EXPECTED_TOP_LEVEL_MODULE_NAME = "noaa_gfs_decode_recovery_bootstrap"
TRUSTED_STDLIB_IMPORTS = (
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


class DecodeBootstrapError(RuntimeError):
    """A pre-execution identity or process-isolation check failed closed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _network_forbidden(*_args: Any, **_kwargs: Any) -> Any:
    raise DecodeBootstrapError("network access is forbidden during decode recovery")


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
        raise DecodeBootstrapError(
            f"network/socket audit event is forbidden during recovery: {event}"
        )


def install_network_deny_guard() -> None:
    global _NETWORK_GUARD_INSTALLED
    if not _NETWORK_GUARD_INSTALLED:
        sys.addaudithook(_network_audit_guard)
        _NETWORK_GUARD_INSTALLED = True
    socket.create_connection = _network_forbidden  # type: ignore[assignment]


def require_exact_file(record: Mapping[str, Any], label: str) -> Path:
    if set(record) != {"path", "size_bytes", "sha256"}:
        raise DecodeBootstrapError(f"{label} identity schema mismatch")
    raw_path = Path(str(record["path"]))
    if not raw_path.is_absolute():
        raise DecodeBootstrapError(f"{label} path is not absolute")
    lexical = Path(os.path.abspath(str(raw_path)))
    for candidate in (lexical, *lexical.parents):
        if candidate.is_symlink() or (
            hasattr(candidate, "is_junction") and candidate.is_junction()
        ):
            raise DecodeBootstrapError(f"{label} symlink/junction is forbidden")
    try:
        resolved = lexical.resolve(strict=True)
    except FileNotFoundError as exc:
        raise DecodeBootstrapError(f"{label} is absent") from exc
    if resolved != lexical or not resolved.is_file():
        raise DecodeBootstrapError(f"{label} is not an exact regular file")
    if resolved.stat().st_size != int(record["size_bytes"]):
        raise DecodeBootstrapError(f"{label} size mismatch")
    if sha256_file(resolved) != str(record["sha256"]):
        raise DecodeBootstrapError(f"{label} hash mismatch")
    return resolved


def _matching_bootstrap_bytecode(path: Path) -> list[Path]:
    candidates: list[Path] = []
    legacy = path.with_suffix(".pyc")
    if legacy.exists():
        candidates.append(legacy)
    cache = path.parent / "__pycache__"
    if cache.exists():
        candidates.extend(cache.glob(f"{path.stem}.*.pyc"))
    return sorted(candidates)


def _validate_top_level_ast(tree: ast.Module) -> None:
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
                raise DecodeBootstrapError("bootstrap top-level expression is forbidden")
        elif isinstance(statement, (ast.Assign, ast.AnnAssign)):
            value = statement.value
            if value is not None and any(isinstance(node, ast.Call) for node in ast.walk(value)):
                raise DecodeBootstrapError("bootstrap top-level call is forbidden")
        elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if statement.decorator_list or any(
                isinstance(node, ast.Call)
                for node in (*statement.args.defaults, *statement.args.kw_defaults)
                if node is not None
            ):
                raise DecodeBootstrapError("bootstrap function definition has top-level work")
        elif isinstance(statement, ast.ClassDef):
            if statement.decorator_list or any(
                isinstance(node, ast.Call)
                for base in (*statement.bases, *statement.keywords)
                for node in ast.walk(base)
            ):
                raise DecodeBootstrapError("bootstrap class definition has top-level work")
            for member in statement.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if member.decorator_list:
                        raise DecodeBootstrapError("bootstrap class decorator is forbidden")
                elif isinstance(member, ast.Expr) and isinstance(member.value, ast.Constant):
                    continue
                elif isinstance(member, (ast.Assign, ast.AnnAssign)):
                    value = member.value
                    if value is not None and any(
                        isinstance(node, ast.Call) for node in ast.walk(value)
                    ):
                        raise DecodeBootstrapError("bootstrap class-body call is forbidden")
                else:
                    raise DecodeBootstrapError("bootstrap class-body work is forbidden")
        else:
            raise DecodeBootstrapError("bootstrap top-level statement is forbidden")
    if tuple(imports) != TRUSTED_STDLIB_IMPORTS:
        raise DecodeBootstrapError("bootstrap stdlib import whitelist mismatch")


def _validate_top_level_import_contract(path: Path) -> None:
    expected_root = path.parent.resolve()
    if (
        __name__ != EXPECTED_TOP_LEVEL_MODULE_NAME
        or __package__ not in ("", None)
        or not sys.path
        or Path(sys.path[0]).resolve() != expected_root
        or os.environ.get("PYTHONPATH") != str(expected_root)
        or "src" in sys.modules
    ):
        raise DecodeBootstrapError("bootstrap top-level import contract mismatch")
    stdlib_roots = tuple(
        sorted({name.split(".", 1)[0] for name in TRUSTED_STDLIB_IMPORTS})
    )
    shadow_candidates = tuple(
        sorted(
            child.name
            for child in expected_root.iterdir()
            if any(
                child.name.casefold() == root.casefold()
                or child.name.casefold().startswith(f"{root}.".casefold())
                for root in stdlib_roots
            )
        )
    )
    if shadow_candidates:
        raise DecodeBootstrapError(
            "bootstrap import root shadows stdlib: " + ",".join(shadow_candidates)
        )
    expected_alias_source = f"{EXPECTED_TOP_LEVEL_MODULE_NAME}.py"
    alias_candidates = tuple(
        sorted(
            (
                child
                for child in expected_root.iterdir()
                if child.name.casefold() == EXPECTED_TOP_LEVEL_MODULE_NAME.casefold()
                or child.name.casefold().startswith(
                    f"{EXPECTED_TOP_LEVEL_MODULE_NAME}.".casefold()
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
        raise DecodeBootstrapError("bootstrap alias candidate is a symlink/junction")
    if tuple(candidate.name for candidate in alias_candidates) != (
        expected_alias_source,
    ) or not alias_candidates[0].is_file():
        raise DecodeBootstrapError(
            "bootstrap top-level alias source candidate set is not unique"
        )
    module_spec = globals().get("__spec__")
    if (
        Path(str(globals().get("__file__", ""))).resolve() != path
        or module_spec is None
        or Path(str(getattr(module_spec, "origin", ""))).resolve() != path
    ):
        raise DecodeBootstrapError("bootstrap imported source origin mismatch")


def validate_bootstrap_trust_anchor(
    identity: Mapping[str, Any], *, expected_path: Path | None = None
) -> Path:
    if __name__ != EXPECTED_TOP_LEVEL_MODULE_NAME or __package__ not in ("", None):
        raise DecodeBootstrapError(
            "bootstrap must execute as the exact top-level module without a package"
        )
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1" or not sys.dont_write_bytecode:
        raise DecodeBootstrapError("PYTHONDONTWRITEBYTECODE=1/-B is required")
    if os.environ.get("PYTHONPYCACHEPREFIX") not in (None, "") or sys.pycache_prefix is not None:
        raise DecodeBootstrapError("external Python bytecode cache prefix is forbidden")
    path = require_exact_file(identity, "bootstrap trust anchor")
    if expected_path is not None and path != expected_path.resolve():
        raise DecodeBootstrapError("bootstrap trust-anchor path mismatch")
    if _matching_bootstrap_bytecode(path):
        raise DecodeBootstrapError("bootstrap bytecode cache must be absent")
    _validate_top_level_import_contract(path)
    try:
        tree = ast.parse(path.read_bytes(), filename=str(path))
    except Exception as exc:
        raise DecodeBootstrapError("bootstrap source AST is invalid") from exc
    _validate_top_level_ast(tree)
    if sha256_file(path) != str(identity["sha256"]):
        raise DecodeBootstrapError("bootstrap source changed during trust check")
    return path


def load_decoder_core_preexecution(record: Mapping[str, Any]) -> Any:
    """Execute only the exact bound decoder-core file, after hash verification."""

    path = require_exact_file(record, "decoder core before child import")
    module_name = "src.noaa_gfs_decode_recovery"
    if module_name in sys.modules:
        raise DecodeBootstrapError("decoder core imported before bootstrap validation")
    source = path.read_bytes()
    if len(source) != int(record["size_bytes"]) or hashlib.sha256(source).hexdigest() != str(
        record["sha256"]
    ):
        raise DecodeBootstrapError("decoder core changed before compile")
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
        raise DecodeBootstrapError("executed decoder core path mismatch")
    if sha256_file(path) != str(record["sha256"]):
        raise DecodeBootstrapError("decoder core changed during execution")
    return module


_WORKER_CORE: Any | None = None
_WORKER_FIRST_WAVE_BARRIER: Any | None = None
_WORKER_FIRST_TASK_SYNCHRONIZED = False


def initialize_spawned_worker(
    runtime_lock: Mapping[str, Any],
    bootstrap_identity: Mapping[str, Any],
    runner_identity: Mapping[str, Any],
    core_identity: Mapping[str, Any],
    first_wave_barrier: Any | None = None,
) -> None:
    """Guard and verify every code boundary before importing decoder code."""

    global _WORKER_CORE, _WORKER_FIRST_WAVE_BARRIER
    global _WORKER_FIRST_TASK_SYNCHRONIZED
    install_network_deny_guard()
    bootstrap_path = validate_bootstrap_trust_anchor(
        bootstrap_identity, expected_path=Path(__file__)
    )
    require_exact_file(runner_identity, "recovery runner child identity")
    core = load_decoder_core_preexecution(core_identity)
    if int(core.MAX_DECODE_PROCESSES) != MAX_DECODE_PROCESSES:
        raise DecodeBootstrapError("decoder core process-count mismatch")
    core.initialize_decode_worker(dict(runtime_lock))
    _WORKER_CORE = core
    _WORKER_FIRST_WAVE_BARRIER = first_wave_barrier
    _WORKER_FIRST_TASK_SYNCHRONIZED = False


def _synchronize_first_worker_task() -> None:
    global _WORKER_FIRST_TASK_SYNCHRONIZED
    if _WORKER_FIRST_WAVE_BARRIER is not None and not _WORKER_FIRST_TASK_SYNCHRONIZED:
        try:
            _WORKER_FIRST_WAVE_BARRIER.wait(timeout=120)
        except BaseException as exc:
            raise DecodeBootstrapError(
                "synchronized bootstrap first-worker-wave barrier failed"
            ) from exc
        _WORKER_FIRST_TASK_SYNCHRONIZED = True


def spawned_decode_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    if _WORKER_CORE is None:
        raise DecodeBootstrapError("spawned decoder core was not initialized")
    _synchronize_first_worker_task()
    return dict(_WORKER_CORE.decode_task_worker(dict(task)))


def spawned_pilot_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    if _WORKER_CORE is None:
        raise DecodeBootstrapError("spawned pilot core was not initialized")
    _synchronize_first_worker_task()
    return dict(_WORKER_CORE.pilot_decode_worker(dict(task)))


class BootstrapProcessAudit:
    __slots__ = (
        "requested_workers",
        "observed_worker_pids",
        "completed_tasks",
        "first_wave_synchronized",
    )

    def __init__(
        self,
        requested_workers: int,
        observed_worker_pids: tuple[int, ...],
        completed_tasks: int,
        first_wave_synchronized: bool,
    ) -> None:
        self.requested_workers = requested_workers
        self.observed_worker_pids = observed_worker_pids
        self.completed_tasks = completed_tasks
        self.first_wave_synchronized = first_wave_synchronized


def bounded_spawn_map(
    tasks: Iterable[Mapping[str, Any]],
    *,
    worker_kind: str,
    runtime_lock: Mapping[str, Any],
    bootstrap_identity: Mapping[str, Any],
    runner_identity: Mapping[str, Any],
    core_identity: Mapping[str, Any],
    max_workers: int = MAX_DECODE_PROCESSES,
    synchronize_first_wave: bool = False,
) -> tuple[list[dict[str, Any]], BootstrapProcessAudit]:
    if worker_kind not in {"decode", "pilot"}:
        raise ValueError("bootstrap worker kind must be decode or pilot")
    if not 1 <= int(max_workers) <= MAX_DECODE_PROCESSES:
        raise ValueError("bootstrap worker count must be within 1..7")
    worker = spawned_decode_worker if worker_kind == "decode" else spawned_pilot_worker
    if synchronize_first_wave:
        synchronized_tasks = list(tasks)
        if len(synchronized_tasks) < int(max_workers):
            raise ValueError("synchronized bootstrap wave needs one task per worker")
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
        initializer=initialize_spawned_worker,
        initargs=(
            dict(runtime_lock),
            dict(bootstrap_identity),
            dict(runner_identity),
            dict(core_identity),
            first_wave_barrier,
        ),
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
    return ordered, BootstrapProcessAudit(
        requested_workers=int(max_workers),
        observed_worker_pids=pids,
        completed_tasks=len(ordered),
        first_wave_synchronized=bool(synchronize_first_wave),
    )


def run_real_spawn_pilot(
    pilot_identity: Mapping[str, Any],
    runtime_lock: Mapping[str, Any],
    bootstrap_identity: Mapping[str, Any],
    runner_identity: Mapping[str, Any],
    core_identity: Mapping[str, Any],
    *,
    workers: int = MAX_DECODE_PROCESSES,
    repetitions: int = 70,
) -> dict[str, Any]:
    if repetitions < workers * 2:
        raise ValueError("pilot must schedule at least two tasks per worker")
    rows, audit = bounded_spawn_map(
        [dict(pilot_identity) for _ in range(repetitions)],
        worker_kind="pilot",
        runtime_lock=runtime_lock,
        bootstrap_identity=bootstrap_identity,
        runner_identity=runner_identity,
        core_identity=core_identity,
        max_workers=workers,
        synchronize_first_wave=True,
    )
    reference = {key: value for key, value in rows[0].items() if key != "worker_runtime"}
    if any(
        {key: value for key, value in row.items() if key != "worker_runtime"}
        != reference
        for row in rows[1:]
    ):
        raise DecodeBootstrapError("spawned bootstrap pilot decodes differ")
    if len(audit.observed_worker_pids) != workers:
        raise DecodeBootstrapError("bootstrap pilot did not exercise all workers")
    return {
        "status": "PASS_REAL_GRIB_STDLIB_BOOTSTRAP_PROCESS_ISOLATION",
        "workers_requested": workers,
        "worker_pids_observed": list(audit.observed_worker_pids),
        "tasks": audit.completed_tasks,
        "first_wave_synchronized": audit.first_wave_synchronized,
        "reference": reference,
        "network_requests": 0,
    }

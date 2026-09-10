#!/usr/bin/env python
"""Download/decode the frozen NOAA GFS v2 ranges after independent GO.

This entry point is intentionally inert until a separate independent-red-team
GO file binds the durable raw-launch authorization and this source identity.
It never reads competition labels, 2024/2025 arrays, or fits any estimator.
"""

from __future__ import annotations

import argparse
import atexit
import concurrent.futures
import csv
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import sys
import threading
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO = Path(__file__).resolve().parents[1]
ROOT_DEFAULT = REPO / "artifacts" / "baram2026_ncei_scada_longrun_20260810_v2"
USER_AGENT = "baram2026-noaa-gfs-multiseason-range/2.0"
EXPECTED_RANGE_ROWS = 10_368
EXPECTED_RANGE_BYTES = 10_296_112_890
MAX_WORKERS = 8


class RequestBudget:
    def __init__(self, maximum_attempts: int, initial_attempts: int = 0) -> None:
        if initial_attempts < 0 or initial_attempts > maximum_attempts:
            raise ValueError("invalid initial request-attempt count")
        self.maximum_attempts = maximum_attempts
        self._attempts = initial_attempts
        self._lock = threading.Lock()

    def reserve(self) -> int:
        with self._lock:
            if self._attempts >= self.maximum_attempts:
                raise RuntimeError(
                    f"actual HTTP-attempt budget exhausted: {self._attempts}/{self.maximum_attempts}"
                )
            self._attempts += 1
            return self._attempts

    @property
    def attempts(self) -> int:
        with self._lock:
            return self._attempts


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            return False
        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return int(exit_code.value) == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class LaunchClaim:
    def __init__(self, root: Path, path: Path, attempt_id: str) -> None:
        self.root = root
        self.path = path
        self.attempt_id = attempt_id
        self.closed = False

    @classmethod
    def acquire(cls, root: Path) -> "LaunchClaim":
        active = root / "raw" / "RAW_LAUNCH_ACTIVE.lock"
        history = root / "raw" / "launch_history"
        active.parent.mkdir(parents=True, exist_ok=True)
        history.mkdir(parents=True, exist_ok=True)
        for _ in range(3):
            if active.exists():
                try:
                    old = json.loads(active.read_text(encoding="utf-8"))
                    old_id = str(old["attempt_id"])
                    old_pid = int(old["pid"])
                except Exception as exc:
                    raise RuntimeError("existing active launch lock is malformed") from exc
                if _pid_alive(old_pid):
                    raise RuntimeError(
                        f"another raw launch is active: attempt={old_id}, pid={old_pid}"
                    )
                stale = history / f"{old_id}__stale_dead_pid.lock"
                if stale.exists():
                    raise RuntimeError(f"stale launch history target already exists: {stale}")
                try:
                    os.replace(active, stale)
                except FileNotFoundError:
                    continue
            attempt_id = (
                datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                + f"__pid{os.getpid()}__{uuid.uuid4().hex[:12]}"
            )
            payload = (
                json.dumps(
                    {
                        "attempt_id": attempt_id,
                        "pid": os.getpid(),
                        "created_utc": datetime.now(timezone.utc)
                        .isoformat()
                        .replace("+00:00", "Z"),
                    },
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
            try:
                descriptor = os.open(active, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            except FileExistsError:
                continue
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            return cls(root, active, attempt_id)
        raise RuntimeError("could not acquire exclusive raw-launch claim")

    def close(self, outcome: str) -> None:
        if self.closed:
            return
        if outcome not in {"complete", "failed"}:
            raise ValueError(outcome)
        current = json.loads(self.path.read_text(encoding="utf-8"))
        if current.get("attempt_id") != self.attempt_id or int(current.get("pid", -1)) != os.getpid():
            raise RuntimeError("active launch claim identity changed")
        destination = (
            self.root
            / "raw"
            / "launch_history"
            / f"{self.attempt_id}__{outcome}.lock"
        )
        if destination.exists():
            raise FileExistsError(destination)
        os.replace(self.path, destination)
        self.closed = True


def bounded_parallel_map(
    items: Iterable[Any], worker: Any, *, max_workers: int
) -> Iterable[tuple[Any, Any]]:
    """Keep at most max_workers submitted and cancel pending on first failure."""

    iterator = iter(items)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    pending: dict[concurrent.futures.Future[Any], Any] = {}

    def submit_one() -> bool:
        try:
            item = next(iterator)
        except StopIteration:
            return False
        pending[executor.submit(worker, item)] = item
        return True

    for _ in range(max_workers):
        if not submit_one():
            break
    try:
        while pending:
            # Execute deterministic bounded waves. Waiting for the whole live
            # wave lets us observe every failure before submitting any new HTTP
            # or decode work, including simultaneous success/failure batches.
            done, _ = concurrent.futures.wait(
                pending, return_when=concurrent.futures.ALL_COMPLETED
            )
            failed = next(
                (
                    future
                    for future in pending
                    if not future.cancelled() and future.exception() is not None
                ),
                None,
            )
            if failed is not None:
                failure = failed.exception()
                for other in pending:
                    if other not in done:
                        other.cancel()
                executor.shutdown(wait=True, cancel_futures=True)
                pending.clear()
                assert failure is not None
                raise failure
            completed_wave = [
                (item, future.result()) for future, item in pending.items()
            ]
            pending.clear()
            for item, result in completed_wave:
                yield item, result
            for _ in range(max_workers):
                if not submit_one():
                    break
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path, root: Path | None = None) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix() if root else str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def canonical_payload_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def runtime_identity() -> dict[str, Any]:
    packages: dict[str, Any] = {}
    for distribution_name, module_name in (
        ("eccodes", "eccodes"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("pyarrow", "pyarrow"),
    ):
        module = importlib.import_module(module_name)
        module_path = Path(str(module.__file__)).resolve()
        packages[distribution_name] = {
            "version": importlib.metadata.version(distribution_name),
            "module_file": str(module_path),
            "module_file_size_bytes": module_path.stat().st_size,
            "module_file_sha256": sha256_file(module_path),
        }
    executable = Path(sys.executable).resolve()
    return {
        "python_version": platform.python_version(),
        "python_executable": str(executable),
        "python_executable_size_bytes": executable.stat().st_size,
        "python_executable_sha256": sha256_file(executable),
        "platform": platform.platform(),
        "packages": packages,
    }


def write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def write_json_exclusive_fsync(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def write_json_atomic_exclusive_fsync(path: Path, payload: Any) -> None:
    """Create JSON through a recoverable fsynced sibling, never overwrite."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    temporary = path.with_name(path.name + ".writepart")
    if path.exists():
        raise FileExistsError(path)
    if temporary.exists():
        if temporary.read_bytes() != encoded:
            raise RuntimeError(f"atomic JSON staging file differs: {temporary}")
    else:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    if path.exists():
        raise FileExistsError(path)
    os.replace(temporary, path)
    fsync_file(path)


def fsync_file(path: Path) -> None:
    # Windows requires a writable descriptor for FlushFileBuffers/os.fsync.
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())


def validate_final_raw(path: Path, meta: Mapping[str, Any], row: Mapping[str, Any]) -> None:
    expected = int(row["range_bytes"])
    if (
        path.stat().st_size != expected
        or meta["raw_sha256"] != sha256_file(path)
        or int(meta["raw_size_bytes"]) != expected
        or meta["object_key"] != row["object_key"]
        or int(meta["range_start"]) != int(row["range_start"])
        or int(meta["range_end"]) != int(row["range_end"])
        or meta["object_etag"] != row["object_etag"]
        or meta["publication_last_modified_utc"]
        != row["publication_last_modified_utc"]
        or int(meta["http_status"]) != 206
        or meta["content_range"]
        != f"bytes {int(meta['request_range_start'])}-{int(row['range_end'])}/{int(row['object_size_bytes'])}"
    ):
        raise RuntimeError(f"final raw/metadata mismatch: {path}")
    with path.open("rb") as stream:
        if stream.read(4) != b"GRIB":
            raise RuntimeError(f"raw range lacks GRIB magic: {path}")
        stream.seek(-4, os.SEEK_END)
        if stream.read(4) != b"7777":
            raise RuntimeError(f"raw range lacks GRIB terminator: {path}")


def build_raw_meta(
    root: Path,
    path: Path,
    row: Mapping[str, Any],
    *,
    raw_sha: str,
    headers: Mapping[str, str],
    status: int,
    retrieved_at: str,
    existing: int,
    prefix_evidence: Mapping[str, Any] | None = None,
    request_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "source_archive": row["source_archive"],
        "archive_product": row["archive_product"],
        "target_operating_day_kst": row["target_operating_day_kst"],
        "run_init_utc": row["run_init_utc"],
        "forecast_hour": int(row["forecast_hour"]),
        "valid_time_utc": row["valid_time_utc"],
        "retrieval_url_or_request_id": row["retrieval_url_or_request_id"],
        "retrieved_at": retrieved_at,
        "raw_filename": path.relative_to(root).as_posix(),
        "raw_sha256": raw_sha,
        "raw_size_bytes": int(row["range_bytes"]),
        "official_metadata": row["official_metadata"],
        "publication_evidence_type": row["publication_evidence_type"],
        "publication_evidence_reference": row["publication_evidence_reference"],
        "cutoff_utc": row["cutoff_utc"],
        "cutoff_margin": int(row["cutoff_margin_seconds"]),
        "object_key": row["object_key"],
        "object_etag": row["object_etag"],
        "object_size_bytes": int(row["object_size_bytes"]),
        "publication_last_modified_utc": row["publication_last_modified_utc"],
        "range_start": int(row["range_start"]),
        "range_end": int(row["range_end"]),
        "range_bytes": int(row["range_bytes"]),
        "family": row["family"],
        "variable": row["variable"],
        "level": row["level"],
        "idx_sha256": row["idx_sha256"],
        "http_status": status,
        "content_range": headers.get("content-range"),
        "request_range_start": int(row["range_start"]) + existing,
        "resumed_from_bytes": existing,
        "status": "VERIFIED",
    }
    if existing:
        if prefix_evidence is None:
            raise RuntimeError("resumed raw metadata requires durable prefix evidence")
        result["resume_prefix_evidence"] = dict(prefix_evidence)
    if request_evidence is not None:
        result["request_completion_evidence"] = dict(request_evidence)
    return result


def _sha256_segment(path: Path, start: int, length: int) -> str:
    digest = hashlib.sha256()
    remaining = length
    with path.open("rb") as stream:
        stream.seek(start)
        while remaining:
            chunk = stream.read(min(1 << 20, remaining))
            if not chunk:
                raise RuntimeError("file ended before requested segment")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def range_request_event_id(
    row: Mapping[str, Any], request_start: int, request_end: int
) -> str:
    return hashlib.sha256(
        (
            f"{row['object_key']}|{row['variable']}|{row['level']}|"
            f"{request_start}|{request_end}"
        ).encode("utf-8")
    ).hexdigest()[:32]


def _request_attempt_records(
    event_root: Path, event_id: str
) -> list[dict[str, Any]]:
    """Probe one request's contiguous attempt paths without directory scans."""

    records: list[dict[str, Any]] = []
    for attempt in range(1, 15_201):
        stem = f"{event_id}__attempt_{attempt:03d}"
        start = event_root / f"{stem}_start.json"
        complete = event_root / f"{stem}_complete.json"
        error = event_root / f"{stem}_error.json"
        start_exists = start.is_file()
        complete_exists = complete.is_file()
        error_exists = error.is_file()
        if not start_exists and not complete_exists and not error_exists:
            return records
        if not start_exists:
            raise RuntimeError(f"request attempt outcome lacks paired start: {stem}")
        if complete_exists and error_exists:
            raise RuntimeError(f"request attempt has both complete and error: {stem}")
        try:
            start_payload = json.loads(start.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"malformed request start event: {start}") from exc
        if (
            start_payload.get("event") != "RANGE_REQUEST_START"
            or start_payload.get("event_id") != event_id
            or int(start_payload.get("attempt", -1)) != attempt
            or not isinstance(start_payload.get("global_raw_attempt_number"), int)
            or int(start_payload["global_raw_attempt_number"]) <= 0
        ):
            raise RuntimeError(f"request start payload/path identity mismatch: {start}")
        outcome_path = complete if complete_exists else error if error_exists else None
        outcome_payload: dict[str, Any] | None = None
        if outcome_path is not None:
            try:
                outcome_payload = json.loads(outcome_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise RuntimeError(f"malformed request outcome event: {outcome_path}") from exc
            expected_event = (
                "RANGE_REQUEST_COMPLETE" if complete_exists else "RANGE_REQUEST_ERROR"
            )
            if (
                outcome_payload.get("event") != expected_event
                or outcome_payload.get("event_id") != event_id
                or int(outcome_payload.get("attempt", -1)) != attempt
                or outcome_payload.get("global_raw_attempt_number")
                != start_payload["global_raw_attempt_number"]
            ):
                raise RuntimeError(
                    f"request outcome payload/path identity mismatch: {outcome_path}"
                )
        records.append(
            {
                "attempt": attempt,
                "start_path": start,
                "complete_path": complete if complete_exists else None,
                "error_path": error if error_exists else None,
                "start": start_payload,
                "complete": outcome_payload if complete_exists else None,
                "error": outcome_payload if error_exists else None,
            }
        )
    # Attempt 15,200 is a valid terminal record. RequestBudget separately
    # forbids reserving or creating attempt 15,201.
    return records


def _matching_complete_events(
    root: Path,
    row: Mapping[str, Any],
    *,
    request_start: int,
    request_end: int,
    payload_bytes: int,
    payload_sha256: str,
) -> list[tuple[Path, Path, dict[str, Any], dict[str, Any]]]:
    expected_content_range = (
        f"bytes {request_start}-{request_end}/{int(row['object_size_bytes'])}"
    )
    matches = []
    event_id = range_request_event_id(row, request_start, request_end)
    for record in _request_attempt_records(
        root / "raw" / "request_events", event_id
    ):
        start_path = record["start_path"]
        complete_path = record["complete_path"]
        if complete_path is None:
            continue
        complete = record["complete"]
        if (
            int(complete.get("range_start", -1)) != request_start
            or int(complete.get("range_end", -1)) != request_end
            or int(complete.get("bytes", -1)) != payload_bytes
            or complete.get("payload_sha256") != payload_sha256
            or complete.get("content_range") != expected_content_range
            or complete.get("etag") != row["object_etag"]
            or complete.get("last_modified_utc")
            != row["publication_last_modified_utc"]
            or int(complete.get("http_status", -1)) != 206
        ):
            continue
        start_event = record["start"]
        if (
            start_event.get("event_id") != complete.get("event_id")
            or int(start_event.get("attempt", -1)) != int(complete.get("attempt", -2))
            or start_event.get("url") != row["retrieval_url_or_request_id"]
            or int(start_event.get("range_start", -1)) != request_start
            or int(start_event.get("range_end", -1)) != request_end
        ):
            continue
        matches.append((start_path, complete_path, start_event, complete))
    return matches


def ensure_resume_prefix(
    root: Path,
    part: Path,
    prefix_path: Path,
    row: Mapping[str, Any],
    prefix_size: int,
) -> dict[str, Any]:
    if not 0 < prefix_size < int(row["range_bytes"]):
        raise RuntimeError("resume prefix size is outside the proper partial range")
    prefix_sha = _sha256_segment(part, 0, prefix_size)
    if prefix_path.is_file():
        prefix = json.loads(prefix_path.read_text(encoding="utf-8"))
        if (
            int(prefix.get("prefix_size_bytes", -1)) != prefix_size
            or prefix.get("prefix_sha256") != prefix_sha
            or prefix.get("object_key") != row["object_key"]
            or int(prefix.get("range_start", -1)) != int(row["range_start"])
            or int(prefix.get("range_end", -1))
            != int(row["range_start"]) + prefix_size - 1
            or prefix.get("object_etag") != row["object_etag"]
            or prefix.get("publication_last_modified_utc")
            != row["publication_last_modified_utc"]
        ):
            raise RuntimeError("durable resume prefix record mismatch")
        matches = _matching_complete_events(
            root,
            row,
            request_start=int(row["range_start"]),
            request_end=int(row["range_start"]) + prefix_size - 1,
            payload_bytes=prefix_size,
            payload_sha256=prefix_sha,
        )
        if not matches:
            raise RuntimeError(
                f"durable resume prefix has {len(matches)} exact completion events"
            )
        bound_pair_present = any(
            prefix.get("start_event") == identity(start_path, root)
            and prefix.get("complete_event") == identity(complete_path, root)
            for start_path, complete_path, _start, _complete in matches
        )
        if not bound_pair_present:
            raise RuntimeError("durable resume prefix event identity mismatch")
        return prefix
    matches = _matching_complete_events(
        root,
        row,
        request_start=int(row["range_start"]),
        request_end=int(row["range_start"]) + prefix_size - 1,
        payload_bytes=prefix_size,
        payload_sha256=prefix_sha,
    )
    if not matches:
        raise RuntimeError(
            f"partial .part has {len(matches)} exact prefix completion events; cannot resume"
        )
    start_path, complete_path, _start, complete = sorted(
        matches, key=lambda match: (match[0].as_posix(), match[1].as_posix())
    )[0]
    prefix = {
        "object_key": row["object_key"],
        "object_etag": row["object_etag"],
        "publication_last_modified_utc": row["publication_last_modified_utc"],
        "range_start": int(row["range_start"]),
        "range_end": int(row["range_start"]) + prefix_size - 1,
        "prefix_size_bytes": prefix_size,
        "prefix_sha256": prefix_sha,
        "retrieved_at": complete["retrieved_at"],
        "start_event": identity(start_path, root),
        "complete_event": identity(complete_path, root),
    }
    write_json_atomic_exclusive_fsync(prefix_path, prefix)
    return prefix


def recover_complete_part_meta(
    root: Path,
    part: Path,
    final_path: Path,
    prefix_path: Path,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconstruct meta only from an exact durable full-range completion event."""

    expected_size = int(row["range_bytes"])
    if part.stat().st_size != expected_size:
        raise RuntimeError("complete-part recovery requires the exact final byte count")
    prefix_size = 0
    prefix: dict[str, Any] | None = None
    if prefix_path.is_file():
        prefix_payload = json.loads(prefix_path.read_text(encoding="utf-8"))
        prefix_size = int(prefix_payload.get("prefix_size_bytes", -1))
        prefix = ensure_resume_prefix(root, part, prefix_path, row, prefix_size)
    request_start = int(row["range_start"]) + prefix_size
    request_bytes = expected_size - prefix_size
    request_sha = _sha256_segment(part, prefix_size, request_bytes)
    candidates = _matching_complete_events(
        root,
        row,
        request_start=request_start,
        request_end=int(row["range_end"]),
        payload_bytes=request_bytes,
        payload_sha256=request_sha,
    )
    if not candidates:
        raise RuntimeError(
            f"completed .part has {len(candidates)} exact durable completion events; cannot recover"
        )
    start_event_path, complete_event_path, _, complete = sorted(
        candidates, key=lambda match: (match[0].as_posix(), match[1].as_posix())
    )[0]
    expected_content_range = (
        f"bytes {request_start}-{int(row['range_end'])}/{int(row['object_size_bytes'])}"
    )
    part_sha = sha256_file(part)
    meta = build_raw_meta(
        root,
        final_path,
        row,
        raw_sha=part_sha,
        headers={"content-range": expected_content_range},
        status=206,
        retrieved_at=str(complete["retrieved_at"]),
        existing=prefix_size,
        prefix_evidence=prefix,
        request_evidence={
            "semantically_identical_completion_event_count": len(candidates),
            "selected_start_event": identity(start_event_path, root),
            "selected_complete_event": identity(complete_event_path, root),
        },
    )
    validate_final_raw(part, meta, row)
    return meta


def _has_exact_completed_request(
    root: Path, row: Mapping[str, Any], request_start: int, request_end: int
) -> bool:
    expected_bytes = request_end - request_start + 1
    expected_content_range = (
        f"bytes {request_start}-{request_end}/{int(row['object_size_bytes'])}"
    )
    event_id = range_request_event_id(row, request_start, request_end)
    for record in _request_attempt_records(
        root / "raw" / "request_events", event_id
    ):
        start_path = record["start_path"]
        complete_path = record["complete_path"]
        if complete_path is None:
            continue
        complete = record["complete"]
        if (
            int(complete.get("range_start", -1)) != request_start
            or int(complete.get("range_end", -1)) != request_end
            or int(complete.get("bytes", -1)) != expected_bytes
            or complete.get("content_range") != expected_content_range
            or complete.get("etag") != row["object_etag"]
            or complete.get("last_modified_utc")
            != row["publication_last_modified_utc"]
            or int(complete.get("http_status", -1)) != 206
            or not isinstance(complete.get("payload_sha256"), str)
            or len(str(complete["payload_sha256"])) != 64
        ):
            continue
        start = record["start"]
        if (
            start.get("event_id") == complete.get("event_id")
            and int(start.get("attempt", -1)) == int(complete.get("attempt", -2))
            and start.get("url") == row["retrieval_url_or_request_id"]
            and int(start.get("range_start", -1)) == request_start
            and int(start.get("range_end", -1)) == request_end
        ):
            return True
    return False


def _record_resume_truncation(
    root: Path,
    row: Mapping[str, Any],
    *,
    old_size: int,
    retained_size: int,
    old_sha256: str,
) -> None:
    event_id = hashlib.sha256(
        (
            f"{row['object_key']}|{row['variable']}|{row['level']}|"
            f"{old_size}|{retained_size}|{old_sha256}"
        ).encode("utf-8")
    ).hexdigest()
    path = root / "raw" / "resume_recovery_events" / f"{event_id}.json"
    payload = {
        "event": "UNCOMMITTED_PART_SUFFIX_TRUNCATED",
        "object_key": row["object_key"],
        "variable": row["variable"],
        "level": row["level"],
        "range_start": int(row["range_start"]),
        "range_end": int(row["range_end"]),
        "old_part_size_bytes": old_size,
        "retained_durable_prefix_bytes": retained_size,
        "old_part_sha256": old_sha256,
        "network_requests_for_recovery": 0,
    }
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise RuntimeError("resume truncation event differs from prior record")
    else:
        write_json_atomic_exclusive_fsync(path, payload)


def normalize_partial_part(
    root: Path,
    part: Path,
    prefix_path: Path,
    row: Mapping[str, Any],
) -> int:
    """Discard only bytes beyond the last exact durable prefix after a crash."""

    existing = part.stat().st_size
    expected = int(row["range_bytes"])
    if not 0 < existing < expected:
        return existing
    retained = existing
    if prefix_path.is_file():
        prefix = json.loads(prefix_path.read_text(encoding="utf-8"))
        retained = int(prefix.get("prefix_size_bytes", -1))
        if not 0 < retained <= existing:
            raise RuntimeError("resume prefix cannot explain current partial file")
        ensure_resume_prefix(root, part, prefix_path, row, retained)
    else:
        try:
            ensure_resume_prefix(root, part, prefix_path, row, existing)
        except RuntimeError:
            if not _has_exact_completed_request(
                root, row, int(row["range_start"]), int(row["range_end"])
            ):
                raise
            # The complete response was durably recorded but the first append did
            # not finish. With no committed prefix, zero is the only safe boundary.
            retained = 0
    if existing != retained:
        old_sha = sha256_file(part)
        _record_resume_truncation(
            root,
            row,
            old_size=existing,
            retained_size=retained,
            old_sha256=old_sha,
        )
        with part.open("r+b") as stream:
            stream.truncate(retained)
            stream.flush()
            os.fsync(stream.fileno())
    return retained


def write_csv_exclusive(
    path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_parquet_exclusive(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(path)
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False, engine="pyarrow")


def _path_under_root(root: Path, relative: str) -> Path:
    root = root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"transaction path escapes root: {relative}") from exc
    return candidate


def staged_output_paths(
    root: Path, attempt_id: str, destinations: Mapping[str, Path]
) -> dict[str, Path]:
    root = root.resolve()
    base = root / "raw" / "output_transactions" / attempt_id / "staged"
    return {
        name: base / destination.relative_to(root)
        for name, destination in destinations.items()
    }


def identity_for_destination(
    staged_path: Path, destination: Path, root: Path
) -> dict[str, Any]:
    record = identity(staged_path)
    record["path"] = destination.relative_to(root).as_posix()
    return record


def write_output_transaction_plan(
    root: Path,
    attempt_id: str,
    staged: Mapping[str, Path],
    destinations: Mapping[str, Path],
) -> Path:
    root = root.resolve()
    if set(staged) != set(destinations):
        raise RuntimeError("transaction staged/destination key mismatch")
    transaction_root = root / "raw" / "output_transactions"
    plan_path = transaction_root / f"{attempt_id}__plan.json"
    committed_path = transaction_root / f"{attempt_id}__committed.json"
    if plan_path.exists() or committed_path.exists():
        raise FileExistsError(f"transaction identity already exists: {attempt_id}")
    items = []
    for name in sorted(destinations):
        source = staged[name]
        destination = destinations[name]
        if not source.is_file():
            raise RuntimeError(f"staged transaction artifact is absent: {source}")
        if destination.exists():
            raise FileExistsError(destination)
        items.append(
            {
                "name": name,
                "staged_path": source.relative_to(root).as_posix(),
                "destination_path": destination.relative_to(root).as_posix(),
                "size_bytes": source.stat().st_size,
                "sha256": sha256_file(source),
            }
        )
    write_json_atomic_exclusive_fsync(
        plan_path,
        {
            "artifact_type": "RAW_OUTPUT_TRANSACTION_PLAN",
            "schema_version": 1,
            "attempt_id": attempt_id,
            "created_utc": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "items": items,
            "canonical_outputs_absent_before_plan": True,
            "decode_and_physical_audit_completed_before_plan": True,
        },
    )
    return plan_path


def _validate_output_transaction_plan(
    root: Path, plan_path: Path, destinations: Mapping[str, Path]
) -> tuple[dict[str, Any], list[tuple[dict[str, Any], Path, Path]]]:
    root = root.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (
        plan.get("artifact_type") != "RAW_OUTPUT_TRANSACTION_PLAN"
        or plan.get("schema_version") != 1
        or not plan.get("canonical_outputs_absent_before_plan")
        or not plan.get("decode_and_physical_audit_completed_before_plan")
    ):
        raise RuntimeError(f"invalid output transaction plan header: {plan_path}")
    attempt_id = str(plan.get("attempt_id", ""))
    if plan_path.name != f"{attempt_id}__plan.json":
        raise RuntimeError("transaction plan filename/attempt mismatch")
    items = plan.get("items")
    if not isinstance(items, list) or len(items) != len(destinations):
        raise RuntimeError("transaction plan item count mismatch")
    by_name = {str(item.get("name")): item for item in items}
    if len(by_name) != len(items) or set(by_name) != set(destinations):
        raise RuntimeError("transaction plan output-name set mismatch")
    resolved = []
    for name in sorted(destinations):
        item = by_name[name]
        staged_path = _path_under_root(root, str(item.get("staged_path", "")))
        destination = _path_under_root(root, str(item.get("destination_path", "")))
        if destination != destinations[name].resolve():
            raise RuntimeError(f"transaction destination mismatch for {name}")
        expected_staged = (
            root
            / "raw"
            / "output_transactions"
            / attempt_id
            / "staged"
            / destination.relative_to(root)
        ).resolve()
        if staged_path != expected_staged:
            raise RuntimeError(f"transaction staged path mismatch for {name}")
        if not isinstance(item.get("size_bytes"), int) or int(item["size_bytes"]) < 0:
            raise RuntimeError(f"transaction size is invalid for {name}")
        if (
            not isinstance(item.get("sha256"), str)
            or len(str(item["sha256"])) != 64
        ):
            raise RuntimeError(f"transaction hash is invalid for {name}")
        resolved.append((item, staged_path, destination))
    return plan, resolved


def commit_output_transaction(
    root: Path, plan_path: Path, destinations: Mapping[str, Path]
) -> Path:
    """Idempotently finish a hash-locked, recoverable multi-file commit."""

    root = root.resolve()
    plan, resolved = _validate_output_transaction_plan(root, plan_path, destinations)
    for item, staged_path, destination in resolved:
        expected_size = int(item["size_bytes"])
        expected_sha = str(item["sha256"])
        if destination.is_file():
            if (
                destination.stat().st_size != expected_size
                or sha256_file(destination) != expected_sha
            ):
                raise RuntimeError(
                    f"partially committed destination differs from plan: {destination}"
                )
            if staged_path.exists() and (
                staged_path.stat().st_size != expected_size
                or sha256_file(staged_path) != expected_sha
            ):
                raise RuntimeError(
                    f"duplicate staged artifact differs from destination: {staged_path}"
                )
            continue
        if not staged_path.is_file():
            raise RuntimeError(
                f"neither staged nor committed transaction artifact exists: {staged_path}"
            )
        if (
            staged_path.stat().st_size != expected_size
            or sha256_file(staged_path) != expected_sha
        ):
            raise RuntimeError(f"staged artifact differs from plan: {staged_path}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged_path, destination)
        fsync_file(destination)
    for item, _staged_path, destination in resolved:
        if (
            destination.stat().st_size != int(item["size_bytes"])
            or sha256_file(destination) != item["sha256"]
        ):
            raise RuntimeError(f"committed output failed rehash: {destination}")
    committed_path = plan_path.with_name(
        plan_path.name.replace("__plan.json", "__committed.json")
    )
    committed = {
        "artifact_type": "RAW_OUTPUT_TRANSACTION_COMMIT",
        "schema_version": 1,
        "attempt_id": plan["attempt_id"],
        "plan": identity(plan_path, root),
        "outputs": {
            item["name"]: identity(destination, root)
            for item, _staged_path, destination in resolved
        },
        "all_outputs_reopened_and_rehashed": True,
    }
    if committed_path.is_file():
        if json.loads(committed_path.read_text(encoding="utf-8")) != committed:
            raise RuntimeError("existing transaction commit marker differs")
    else:
        write_json_atomic_exclusive_fsync(committed_path, committed)
    return committed_path


def recover_or_validate_output_transaction(
    root: Path, destinations: Mapping[str, Path]
) -> Path | None:
    root = root.resolve()
    transaction_root = root / "raw" / "output_transactions"
    plans = sorted(transaction_root.glob("*__plan.json"))
    if len(plans) > 1:
        raise RuntimeError("multiple canonical output transaction plans exist")
    if not plans:
        conflicts = [str(path) for path in destinations.values() if path.exists()]
        if conflicts:
            raise RuntimeError(
                f"canonical output exists without a transaction plan: {conflicts}"
            )
        return None
    return commit_output_transaction(root, plans[0], destinations)


def feature_name(variable: str, level: str) -> str:
    return f"{variable}_{level.replace(' ', '')}"


def raw_path_for(root: Path, row: Mapping[str, Any]) -> Path:
    run_date = str(row["run_init_utc"])[:10].replace("-", "")
    fh = int(row["forecast_hour"])
    name = feature_name(str(row["variable"]), str(row["level"]))
    return root / "raw" / "ranges" / f"gfs.{run_date}" / "12" / f"f{fh:03d}" / f"{name}.grib2"


def fetch_exact_range(
    url: str,
    start: int,
    end: int,
    *,
    attempts: int = 4,
    event_root: Path | None = None,
    event_id: str | None = None,
    request_budget: RequestBudget | None = None,
    expected_object_size: int | None = None,
    expected_etag: str | None = None,
    expected_last_modified_utc: str | None = None,
) -> tuple[bytes, dict[str, str], int, str]:
    if start < 0 or end < start:
        raise ValueError("invalid inclusive byte range")
    expected = end - start + 1
    headers = {
        "User-Agent": USER_AGENT,
        "Accept-Encoding": "identity",
        "Range": f"bytes={start}-{end}",
    }
    last_error: Exception | None = None
    existing_attempts = 0
    if event_root is not None:
        if not event_id:
            raise ValueError("event_id is required with event_root")
        event_root.mkdir(parents=True, exist_ok=True)
        existing_attempts = len(_request_attempt_records(event_root, event_id))
        if existing_attempts >= 15_200:
            raise RuntimeError("request-local attempt 15,201 is forbidden")
    for attempt in range(attempts):
        attempt_number = existing_attempts + attempt + 1
        global_attempt_number = request_budget.reserve() if request_budget is not None else None
        start_event = (
            event_root / f"{event_id}__attempt_{attempt_number:03d}_start.json"
            if event_root is not None
            else None
        )
        if start_event is not None:
            write_json_exclusive_fsync(
                start_event,
                {
                    "event": "RANGE_REQUEST_START",
                    "event_id": event_id,
                    "attempt": attempt_number,
                    "global_raw_attempt_number": global_attempt_number,
                    "url": url,
                    "range_start": start,
                    "range_end": end,
                    "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                },
            )
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=90) as response:
                response_headers = {
                    str(key).lower(): str(value).strip()
                    for key, value in response.headers.items()
                }
                if int(response.status) != 206:
                    raise RuntimeError(f"expected HTTP 206, observed {response.status}")
                if int(response_headers.get("content-length", "-1")) != expected:
                    raise RuntimeError("Content-Length does not equal requested range")
                if expected_object_size is None:
                    raise ValueError("expected_object_size is required")
                expected_content_range = f"bytes {start}-{end}/{expected_object_size}"
                if response_headers.get("content-range", "") != expected_content_range:
                    raise RuntimeError("Content-Range does not bind requested range")
                if expected_etag is None or response_headers.get("etag", "").strip('"') != expected_etag:
                    raise RuntimeError("range response ETag differs from census object ETag")
                if expected_last_modified_utc is None or "last-modified" not in response_headers:
                    raise RuntimeError("range response lacks frozen Last-Modified evidence")
                response_last_modified = parsedate_to_datetime(
                    response_headers["last-modified"]
                ).astimezone(timezone.utc)
                expected_last_modified = datetime.fromisoformat(
                    expected_last_modified_utc.replace("Z", "+00:00")
                ).astimezone(timezone.utc)
                if response_last_modified != expected_last_modified:
                    raise RuntimeError("range response Last-Modified differs from census")
                payload = response.read(expected + 1)
                if len(payload) != expected:
                    raise RuntimeError(
                        f"range payload size mismatch: {len(payload)} != {expected}"
                    )
                retrieved_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                if event_root is not None:
                    write_json_exclusive_fsync(
                        event_root
                        / f"{event_id}__attempt_{attempt_number:03d}_complete.json",
                        {
                            "event": "RANGE_REQUEST_COMPLETE",
                            "event_id": event_id,
                            "attempt": attempt_number,
                            "global_raw_attempt_number": global_attempt_number,
                            "http_status": 206,
                            "range_start": start,
                            "range_end": end,
                            "content_range": response_headers["content-range"],
                            "etag": expected_etag,
                            "last_modified_utc": expected_last_modified_utc,
                            "bytes": len(payload),
                            "payload_sha256": hashlib.sha256(payload).hexdigest(),
                            "retrieved_at": retrieved_at,
                        },
                    )
                return payload, response_headers, 206, retrieved_at
        except Exception as exc:
            last_error = exc
            if event_root is not None:
                write_json_exclusive_fsync(
                    event_root / f"{event_id}__attempt_{attempt_number:03d}_error.json",
                    {
                        "event": "RANGE_REQUEST_ERROR",
                        "event_id": event_id,
                        "attempt": attempt_number,
                        "global_raw_attempt_number": global_attempt_number,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "recorded_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    },
                )
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
    raise RuntimeError(f"failed range {url} {start}-{end}: {last_error}") from last_error


def download_or_resume_range(
    root: Path, row: Mapping[str, Any], request_budget: RequestBudget | None = None
) -> dict[str, Any]:
    path = raw_path_for(root, row)
    sidecar = path.with_suffix(".grib2.meta.json")
    meta_part = path.with_suffix(".grib2.meta.part.json")
    prefix_path = path.with_suffix(".grib2.prefix.json")
    expected = int(row["range_bytes"])
    if path.is_file() and sidecar.is_file():
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        if meta_part.exists():
            raise RuntimeError(f"unexpected stale meta.part beside final sidecar: {meta_part}")
        validate_final_raw(path, meta, row)
        return {"path": path, "meta": meta, "network": False, "network_bytes": 0}
    part = path.with_suffix(".grib2.part")
    # Deterministic crash recovery for either side of the two atomic renames.
    if meta_part.is_file() and not sidecar.exists():
        recovery_source = path if path.is_file() and not part.exists() else part
        if not recovery_source.is_file() or (path.exists() and part.exists()):
            raise RuntimeError(f"ambiguous meta.part recovery state: {path}")
        recovery_meta = json.loads(meta_part.read_text(encoding="utf-8"))
        validate_final_raw(recovery_source, recovery_meta, row)
        if recovery_source == part:
            os.replace(part, path)
        os.replace(meta_part, sidecar)
        validate_final_raw(path, recovery_meta, row)
        return {
            "path": path,
            "meta": recovery_meta,
            "network": False,
            "network_bytes": 0,
            "recovered_orphan_final": True,
        }
    if path.exists() or sidecar.exists() or meta_part.exists():
        raise RuntimeError(f"incomplete finalized raw state: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = part.stat().st_size if part.exists() else 0
    if existing > expected:
        raise RuntimeError(f"partial raw range exceeds expected size: {part}")
    if 0 < existing < expected:
        existing = normalize_partial_part(root, part, prefix_path, row)
    if existing == expected and existing > 0:
        recovered_meta = recover_complete_part_meta(root, part, path, prefix_path, row)
        write_json_atomic_exclusive_fsync(meta_part, recovered_meta)
        os.replace(part, path)
        os.replace(meta_part, sidecar)
        validate_final_raw(path, recovered_meta, row)
        return {
            "path": path,
            "meta": recovered_meta,
            "network": False,
            "network_bytes": 0,
            "recovered_completed_part_from_request_event": True,
        }
    if existing < expected:
        prefix = (
            ensure_resume_prefix(root, part, prefix_path, row, existing)
            if existing
            else None
        )
        absolute_start = int(row["range_start"]) + existing
        absolute_end = int(row["range_end"])
        request_identity = range_request_event_id(row, absolute_start, absolute_end)
        payload, headers, status, retrieved_at = fetch_exact_range(
            str(row["retrieval_url_or_request_id"]),
            absolute_start,
            absolute_end,
            event_root=root / "raw" / "request_events",
            event_id=request_identity,
            request_budget=request_budget,
            expected_object_size=int(row["object_size_bytes"]),
            expected_etag=str(row["object_etag"]),
            expected_last_modified_utc=str(row["publication_last_modified_utc"]),
        )
        with part.open("ab" if part.exists() else "xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        network_bytes = len(payload)
    if part.stat().st_size != expected:
        raise RuntimeError(f"completed raw range has wrong size: {part}")
    raw_sha = sha256_file(part)
    request_start = int(row["range_start"]) + existing
    request_bytes = expected - existing
    request_sha = _sha256_segment(part, existing, request_bytes)
    request_matches = _matching_complete_events(
        root,
        row,
        request_start=request_start,
        request_end=int(row["range_end"]),
        payload_bytes=request_bytes,
        payload_sha256=request_sha,
    )
    if not request_matches:
        raise RuntimeError("completed payload lacks exact durable request event")
    request_start_path, request_complete_path, _, _ = sorted(
        request_matches,
        key=lambda match: (match[0].as_posix(), match[1].as_posix()),
    )[0]
    meta = build_raw_meta(
        root,
        path,
        row,
        raw_sha=raw_sha,
        headers=headers,
        status=status,
        retrieved_at=retrieved_at,
        existing=existing,
        prefix_evidence=prefix,
        request_evidence={
            "semantically_identical_completion_event_count": len(request_matches),
            "selected_start_event": identity(request_start_path, root),
            "selected_complete_event": identity(request_complete_path, root),
        },
    )
    # Crash-consistent sequence: fsynced payload .part + fsynced meta.part,
    # followed by two atomic renames. Either intermediate state is recoverable.
    validate_final_raw(part, meta, row)
    write_json_atomic_exclusive_fsync(meta_part, meta)
    os.replace(part, path)
    os.replace(meta_part, sidecar)
    validate_final_raw(path, meta, row)
    return {"path": path, "meta": meta, "network": network_bytes > 0, "network_bytes": network_bytes}


def write_progress_checkpoint(root: Path, name: str, payload: dict[str, Any]) -> None:
    path = root / "raw" / "progress" / f"{name}.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("checkpoint_name") != name:
            raise RuntimeError(f"invalid existing checkpoint: {path}")
        return
    write_json_exclusive(
        path,
        {
            "checkpoint_name": name,
            "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            **payload,
        },
    )


def validate_independent_go(
    go_path: Path,
    authorization_path: Path,
    runner_path: Path,
    census_manifest_path: Path,
    coordinate_path: Path,
    observed_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    if not go_path.is_file():
        raise RuntimeError("independent raw-launch GO is absent; network launch is forbidden")
    go = json.loads(go_path.read_text(encoding="utf-8"))
    required = {
        "status": "GO_RAW_RANGE_LAUNCH",
        "bound_authorization_sha256": sha256_file(authorization_path),
        "bound_runner_sha256": sha256_file(runner_path),
        "bound_census_manifest_sha256": sha256_file(census_manifest_path),
        "bound_coordinate_lock_sha256": sha256_file(coordinate_path),
        "bound_runtime_identity_sha256": canonical_payload_sha256(observed_runtime),
        "expected_range_rows": EXPECTED_RANGE_ROWS,
        "expected_range_bytes": EXPECTED_RANGE_BYTES,
        "max_actual_http_attempts": 15_200,
        "census_worst_case_http_attempts": 4_800,
        "census_plus_raw_max_http_attempts": 20_000,
    }
    for key, value in required.items():
        if go.get(key) != value:
            raise RuntimeError(f"independent GO mismatch for {key}: {go.get(key)!r} != {value!r}")
    return go


def validate_census_access_binding(
    authorization: Mapping[str, Any], path: Path, root: Path
) -> dict[str, Any]:
    binding = authorization.get("census_cumulative_access")
    if not isinstance(binding, Mapping):
        raise RuntimeError("authorization lacks cumulative census access binding")
    actual_identity = identity(path, root)
    for key in ("path", "size_bytes", "sha256"):
        if binding.get(key) != actual_identity[key]:
            raise RuntimeError(f"cumulative census access identity mismatch: {key}")
    required_facts = {
        "successful_http_requests": 1_200,
        "attempts_per_logical_request_hard_max": 4,
        "worst_case_http_attempts": 4_800,
    }
    for key, value in required_facts.items():
        if binding.get(key) != value:
            raise RuntimeError(f"cumulative census access fact mismatch: {key}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("artifact_type")
        != "CENSUS_CUMULATIVE_ACCESS_APPEND_ONLY_AMENDMENT"
        or payload.get("initial_population_invocation", {}).get(
            "successful_http_requests"
        )
        != 1_200
        or payload.get("phase_total", {}).get("successful_http_requests") != 1_200
        or payload.get("phase_total", {}).get("raw_range_requests") != 0
        or payload.get("phase_total", {}).get("raw_range_bytes") != 0
        or payload.get("raw_payload_read") is not False
    ):
        raise RuntimeError("cumulative census access payload facts mismatch")
    return payload


def require_final_disk_reserve(root: Path, reserve_bytes: int = 200_000_000_000) -> int:
    free_bytes = int(shutil.disk_usage(root).free)
    if free_bytes < reserve_bytes:
        raise RuntimeError("final 200 GB free-disk reserve would be violated")
    return free_bytes


def bilinear_regular_ll(
    values: Any,
    latitude: float,
    longitude: float,
    *,
    ni: int,
    nj: int,
    first_latitude: float,
    first_longitude: float,
    di: float,
    dj: float,
    missing_value: float,
) -> float:
    import numpy as np

    grid = np.asarray(values).reshape(nj, ni)
    x = ((longitude - first_longitude) % 360.0) / di
    y = (first_latitude - latitude) / dj
    if y < 0 or y > nj - 1:
        raise ValueError("latitude outside regular_ll grid")
    i0 = int(math.floor(x)) % ni
    i1 = (i0 + 1) % ni
    j0 = min(int(math.floor(y)), nj - 1)
    j1 = min(j0 + 1, nj - 1)
    wx = x - math.floor(x)
    wy = y - math.floor(y)
    corners = np.asarray(
        [grid[j0, i0], grid[j0, i1], grid[j1, i0], grid[j1, i1]], dtype=float
    )
    if (
        not np.isfinite(corners).all()
        or np.any(corners == float(missing_value))
        or np.any(np.abs(corners) > 1e19)
    ):
        return float("nan")
    top = corners[0] * (1.0 - wx) + corners[1] * wx
    bottom = corners[2] * (1.0 - wx) + corners[3] * wx
    return float(top * (1.0 - wy) + bottom * wy)


def decode_message(
    result: dict[str, Any], row: Mapping[str, Any], sites: list[dict[str, Any]]
) -> dict[str, Any]:
    import eccodes

    path: Path = result["path"]
    with path.open("rb") as stream:
        handle = eccodes.codes_grib_new_from_file(stream)
        if handle is None:
            raise RuntimeError(f"ecCodes failed to open {path}")
        try:
            keys = {
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
                    "stepUnits",
                    "indicatorOfUnitOfTimeRange",
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
                    "missingValue",
                )
            }
            values = eccodes.codes_get_array(handle, "values")
        finally:
            eccodes.codes_release(handle)
        second_handle = eccodes.codes_grib_new_from_file(stream)
        if second_handle is not None:
            try:
                raise RuntimeError(f"raw range contains more than one GRIB message: {path}")
            finally:
                eccodes.codes_release(second_handle)
    if (
        keys["gridType"] != "regular_ll"
        or int(keys["Ni"]) != 1440
        or int(keys["Nj"]) != 721
        or int(keys["iScansNegatively"]) != 0
        or int(keys["jScansPositively"]) != 0
        or int(keys["jPointsAreConsecutive"]) != 0
        or int(keys["alternativeRowScanning"]) != 0
        or int(keys["forecastTime"]) != int(row["forecast_hour"])
        or int(keys["stepUnits"]) != 1
        or int(keys["indicatorOfUnitOfTimeRange"]) != 1
        or float(keys["latitudeOfFirstGridPointInDegrees"]) != 90.0
        or float(keys["longitudeOfFirstGridPointInDegrees"]) != 0.0
        or float(keys["iDirectionIncrementInDegrees"]) != 0.25
        or float(keys["jDirectionIncrementInDegrees"]) != 0.25
    ):
        raise RuntimeError(f"decoded GRIB metadata mismatch: {path}, {keys}")
    run = datetime.fromisoformat(str(row["run_init_utc"]).replace("Z", "+00:00")).astimezone(
        timezone.utc
    )
    valid = datetime.fromisoformat(str(row["valid_time_utc"]).replace("Z", "+00:00")).astimezone(
        timezone.utc
    )
    if (
        int(keys["dataDate"]) != int(run.strftime("%Y%m%d"))
        or int(keys["dataTime"]) != int(run.strftime("%H%M"))
        or int(keys["validityDate"]) != int(valid.strftime("%Y%m%d"))
        or int(keys["validityTime"]) != int(valid.strftime("%H%M"))
    ):
        raise RuntimeError(f"decoded GRIB run/valid timestamp mismatch: {path}, {keys}")
    variable = str(row["variable"])
    if variable == "HPBL":
        variable_match = (
            str(keys["shortName"]) == "unknown"
            and int(keys["discipline"]) == 0
            and int(keys["parameterCategory"]) == 3
            and int(keys["parameterNumber"]) == 196
        )
    else:
        expected_short_name = {"UGRD": "u", "VGRD": "v"}.get(variable)
        variable_match = str(keys["shortName"]) == expected_short_name
    if not variable_match:
        raise RuntimeError(f"decoded GRIB variable identity mismatch: {path}, {keys}")
    level = str(row["level"])
    expected_type = "surface" if level == "surface" else "isobaricInhPa"
    expected_level = 0 if level == "surface" else int(level.split()[0])
    if keys["typeOfLevel"] != expected_type or int(keys["level"]) != expected_level:
        raise RuntimeError(f"decoded GRIB level mismatch: {path}, {keys}")
    site_values = [
        bilinear_regular_ll(
            values,
            float(site["latitude"]),
            float(site["longitude"]),
            ni=int(keys["Ni"]),
            nj=int(keys["Nj"]),
            first_latitude=float(keys["latitudeOfFirstGridPointInDegrees"]),
            first_longitude=float(keys["longitudeOfFirstGridPointInDegrees"]),
            di=float(keys["iDirectionIncrementInDegrees"]),
            dj=float(keys["jDirectionIncrementInDegrees"]),
            missing_value=float(keys["missingValue"]),
        )
        for site in sites
    ]
    return {"feature": feature_name(str(row["variable"]), level), "site_values": site_values}


def run(root: Path) -> dict[str, Any]:
    root = root.resolve()
    runner_path = Path(__file__).resolve()
    authorization_path = root / "prereg" / "raw_launch_authorization_v1.json"
    census_manifest_path = root / "manifest_census_v1.json"
    range_path = root / "census" / "field_range_census.parquet"
    coordinate_path = root / "prereg" / "authoritative_turbine_coordinate_lock_v1.json"
    cumulative_access_path = root / "audit" / "CENSUS_CUMULATIVE_ACCESS_AMENDMENT.json"
    go_path = root / "independent_redteam" / "TRACK_A_RAW_LAUNCH_GO.json"
    if not authorization_path.is_file():
        raise RuntimeError("durable raw-launch authorization is absent")
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    if int(authorization.get("raw_actual_http_attempt_budget", -1)) != 15_200:
        raise RuntimeError("authorization must bind raw actual HTTP-attempt cap 15,200")
    if int(authorization.get("census_worst_case_http_attempts", -1)) != 4_800:
        raise RuntimeError("authorization must bind census worst-case attempts 4,800")
    if int(authorization.get("census_plus_raw_max_http_attempts", -1)) != 20_000:
        raise RuntimeError("authorization must bind combined actual-attempt cap 20,000")
    if (
        int(authorization["census_worst_case_http_attempts"])
        + int(authorization["raw_actual_http_attempt_budget"])
        > 20_000
    ):
        raise RuntimeError("census plus raw actual HTTP-attempt cap exceeds 20,000")
    if authorization["runner"]["sha256"] != sha256_file(runner_path):
        raise RuntimeError("runner source differs from raw-launch authorization")
    if authorization["field_range_census"]["sha256"] != sha256_file(range_path):
        raise RuntimeError("field range census differs from authorization")
    if authorization["census_manifest"]["sha256"] != sha256_file(census_manifest_path):
        raise RuntimeError("census manifest differs from authorization")
    validate_census_access_binding(
        authorization, cumulative_access_path, root
    )
    expected_coordinate_identity = identity(coordinate_path, root)
    if authorization.get("coordinate_lock") != expected_coordinate_identity:
        raise RuntimeError("coordinate lock differs from raw-launch authorization")
    observed_runtime = runtime_identity()
    if authorization.get("runtime_identity") != observed_runtime:
        raise RuntimeError("runtime differs from raw-launch authorization")
    go = validate_independent_go(
        go_path,
        authorization_path,
        runner_path,
        census_manifest_path,
        coordinate_path,
        observed_runtime,
    )
    claim = LaunchClaim.acquire(root)

    def _close_failed_at_exit() -> None:
        if not claim.closed:
            claim.close("failed")

    atexit.register(_close_failed_at_exit)

    outputs = {
        "raw_parquet": root / "raw" / "RAW_RANGE_MANIFEST.parquet",
        "raw_csv": root / "raw" / "RAW_RANGE_MANIFEST.csv",
        "site": root / "decoded" / "site_values_target_free.parquet",
        "group": root / "decoded" / "group_values_target_free.parquet",
        "audit": root / "decoded" / "PHYSICAL_AND_COVERAGE_AUDIT.json",
        "lock": root / "decoded" / "DECODED_MATRIX_LOCK.json",
        "access": root / "raw" / "RAW_ACCESS_LEDGER.json",
        "manifest": root / "manifest_raw_v1.json",
    }
    committed = recover_or_validate_output_transaction(root, outputs)
    if committed is not None:
        audit = json.loads(outputs["audit"].read_text(encoding="utf-8"))
        claim.close("complete")
        return {
            "manifest": identity(outputs["manifest"], root),
            "decoded_lock": identity(outputs["lock"], root),
            "transaction_commit": identity(committed, root),
            "raw_ranges": EXPECTED_RANGE_ROWS,
            "raw_bytes": EXPECTED_RANGE_BYTES,
            "family_status": {
                key: value["status"] for key, value in audit["families"].items()
            },
            "recovered_or_validated_transaction_without_network": True,
        }
    staged = staged_output_paths(root, claim.attempt_id, outputs)
    import pandas as pd

    range_frame = pd.read_parquet(range_path)
    if len(range_frame) != EXPECTED_RANGE_ROWS:
        raise RuntimeError("range row count differs from frozen authorization")
    if int(range_frame["range_bytes"].sum()) != EXPECTED_RANGE_BYTES:
        raise RuntimeError("range byte total differs from frozen authorization")
    if not (range_frame["status"] == "CENSUS_VERIFIED").all():
        raise RuntimeError("a field census row is not verified")
    free_before = shutil.disk_usage(root).free
    if free_before - EXPECTED_RANGE_BYTES < 200_000_000_000:
        raise RuntimeError("200 GB post-download reserve would be violated")
    rows = range_frame.to_dict("records")
    event_root = root / "raw" / "request_events"
    previous_attempt_starts = len(list(event_root.glob("*_start.json")))
    request_budget = RequestBudget(
        maximum_attempts=int(authorization["raw_actual_http_attempt_budget"]),
        initial_attempts=previous_attempt_starts,
    )

    results: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    completed_keys: list[str] = []
    current_network_requests = 0
    current_network_bytes = 0
    for row, result in bounded_parallel_map(
        rows,
        lambda item: download_or_resume_range(root, item, request_budget),
        max_workers=MAX_WORKERS,
    ):
        key = (
            str(row["object_key"]),
            int(row["forecast_hour"]),
            str(row["variable"]),
            str(row["level"]),
        )
        results[key] = result
        completed_keys.append("|".join(map(str, key)))
        current_network_requests += int(result["network"])
        current_network_bytes += int(result["network_bytes"])
        completed = len(completed_keys)
        if completed % 100 == 0 or completed == EXPECTED_RANGE_ROWS:
            digest = hashlib.sha256(
                ("\n".join(sorted(completed_keys)) + "\n").encode("utf-8")
            ).hexdigest()
            write_progress_checkpoint(
                root,
                f"raw_ranges__{claim.attempt_id}__{completed:06d}",
                {
                    "phase": "RAW_RANGE_DOWNLOAD",
                    "attempt_id": claim.attempt_id,
                    "completed_ranges": completed,
                    "completed_key_set_sha256": digest,
                    "network_requests_this_invocation_so_far": current_network_requests,
                    "network_bytes_this_invocation_so_far": current_network_bytes,
                    "actual_http_attempts_cumulative": request_budget.attempts,
                },
            )
    if len(results) != EXPECTED_RANGE_ROWS:
        raise RuntimeError("raw result count mismatch")

    raw_manifest_rows = []
    for row in rows:
        key = (str(row["object_key"]), int(row["forecast_hour"]), str(row["variable"]), str(row["level"]))
        raw_manifest_rows.append(results[key]["meta"])
    if sum(int(row["raw_size_bytes"]) for row in raw_manifest_rows) != EXPECTED_RANGE_BYTES:
        raise RuntimeError("raw manifest byte total mismatch")
    write_parquet_exclusive(staged["raw_parquet"], raw_manifest_rows)
    write_csv_exclusive(staged["raw_csv"], raw_manifest_rows, list(raw_manifest_rows[0]))

    coordinate_lock = json.loads(coordinate_path.read_text(encoding="utf-8"))
    sites = coordinate_lock["sites"]
    site_accumulator: dict[tuple[str, str, int], dict[str, Any]] = {}
    decoded_count = 0
    decode_items = []
    for row in rows:
        key = (
            str(row["object_key"]),
            int(row["forecast_hour"]),
            str(row["variable"]),
            str(row["level"]),
        )
        decode_items.append((row, results[key]))
    for item, decoded in bounded_parallel_map(
        decode_items,
        lambda value: decode_message(value[1], value[0], sites),
        max_workers=7,
    ):
        row = item[0]
        for site, value in zip(sites, decoded["site_values"]):
            key = (
                str(row["valid_time_utc"]),
                str(row["target_operating_day_kst"]),
                int(site["site_id"]),
            )
            target = site_accumulator.setdefault(
                key,
                {
                    "valid_time_utc": row["valid_time_utc"],
                    "target_operating_day_kst": row["target_operating_day_kst"],
                    "run_init_utc": row["run_init_utc"],
                    "forecast_hour": int(row["forecast_hour"]),
                    "site_id": int(site["site_id"]),
                    "group": site["group"],
                    "latitude": float(site["latitude"]),
                    "longitude": float(site["longitude"]),
                    "capacity_mw": float(site["capacity_mw"]),
                },
            )
            if decoded["feature"] in target:
                raise RuntimeError(
                    f"duplicate decoded feature key: {key}/{decoded['feature']}"
                )
            target[decoded["feature"]] = value
        decoded_count += 1
        if decoded_count % 100 == 0 or decoded_count == EXPECTED_RANGE_ROWS:
            write_progress_checkpoint(
                root,
                f"decoded_messages__{claim.attempt_id}__{decoded_count:06d}",
                {
                    "phase": "ECCODES_BILINEAR_DECODE",
                    "attempt_id": claim.attempt_id,
                    "completed_messages": decoded_count,
                    "network_requests": 0,
                },
            )
    features = ["HPBL_surface"] + [
        f"{variable}_{level}mb"
        for level in (925, 950, 975, 1000)
        for variable in ("UGRD", "VGRD")
    ]
    site_rows = sorted(
        site_accumulator.values(),
        key=lambda row: (row["valid_time_utc"], row["site_id"]),
    )
    if len(site_rows) != 1_152 * 17 or any(
        any(feature not in row for feature in features) for row in site_rows
    ):
        raise RuntimeError("decoded site key/feature coverage mismatch")
    group_rows = []
    by_time_group: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in site_rows:
        by_time_group.setdefault((row["valid_time_utc"], row["group"]), []).append(row)
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
        for feature in features:
            values = [float(row[feature]) for row in members]
            output[feature] = (
                sum(value * float(row["capacity_mw"]) for value, row in zip(values, members))
                / total_capacity
                if all(math.isfinite(value) for value in values)
                else float("nan")
            )
        group_rows.append(output)
    if len(group_rows) != 1_152 * 3:
        raise RuntimeError("decoded group key coverage mismatch")
    expected_site_counts = {"kpx_group_1": 6, "kpx_group_2": 6, "kpx_group_3": 5}
    expected_capacities = {"kpx_group_1": 21.6, "kpx_group_2": 21.6, "kpx_group_3": 21.0}
    for row in group_rows:
        group = str(row["group"])
        if int(row["site_count"]) != expected_site_counts[group]:
            raise RuntimeError(f"group site-count mismatch: {row}")
        if abs(float(row["capacity_mw"]) - expected_capacities[group]) > 1e-12:
            raise RuntimeError(f"group capacity mismatch: {row}")
    write_parquet_exclusive(staged["site"], site_rows)
    write_parquet_exclusive(staged["group"], group_rows)

    import numpy as np

    hpbl = np.asarray([row["HPBL_surface"] for row in site_rows], dtype=float)
    vertical = np.asarray(
        [[row[feature] for feature in features[1:]] for row in site_rows], dtype=float
    )
    pbl_pass = bool(np.isfinite(hpbl).all() and np.all((hpbl >= 0.0) & (hpbl <= 10_000.0)))
    vertical_pass = bool(np.isfinite(vertical).all() and np.all(np.abs(vertical) <= 150.0))
    final_free_disk_before_transaction = require_final_disk_reserve(root)
    audit = {
        "expected_site_rows": 1_152 * 17,
        "observed_site_rows": len(site_rows),
        "expected_group_rows": 1_152 * 3,
        "observed_group_rows": len(group_rows),
        "duplicate_site_keys": len(site_rows)
        - len({(row["valid_time_utc"], row["site_id"]) for row in site_rows}),
        "duplicate_group_keys": len(group_rows)
        - len({(row["valid_time_utc"], row["group"]) for row in group_rows}),
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
                "status": "PASS"
                if vertical_pass
                else "REJECT_LOW_LEVEL_ISOBARIC_WIND_PROFILE_ONLY",
            },
        },
        "independent_family_salvage_applied_exactly": True,
        "labels_read": False,
        "models_fit": 0,
        "final_free_disk_before_transaction_bytes": final_free_disk_before_transaction,
        "final_200gb_reserve_pass": True,
    }
    write_json_exclusive(staged["audit"], audit)
    access = {
        "expected_ranges": EXPECTED_RANGE_ROWS,
        "expected_bytes": EXPECTED_RANGE_BYTES,
        "network_requests_this_invocation": current_network_requests,
        "network_bytes_this_invocation": current_network_bytes,
        "actual_http_attempts_before_invocation": previous_attempt_starts,
        "actual_http_attempts_after_invocation": request_budget.attempts,
        "actual_http_attempts_this_invocation": request_budget.attempts
        - previous_attempt_starts,
        "raw_actual_http_attempt_budget": request_budget.maximum_attempts,
        "census_worst_case_http_attempts": 4_800,
        "census_plus_raw_max_http_attempts": 20_000,
        "census_worst_case_plus_raw_budget_le_20000": (
            4_800 + request_budget.maximum_attempts <= 20_000
        ),
        "census_successful_http_requests": int(
            authorization["census_cumulative_access"]["successful_http_requests"]
        ),
        "census_plus_raw_actual_attempts_or_successes": int(
            authorization["census_cumulative_access"]["successful_http_requests"]
        )
        + request_budget.attempts,
        "cached_reuses": EXPECTED_RANGE_ROWS - current_network_requests,
        "durable_transport_attempt_starts": len(
            list((root / "raw" / "request_events").glob("*_start.json"))
        ),
        "durable_transport_attempt_completions": len(
            list((root / "raw" / "request_events").glob("*_complete.json"))
        ),
        "durable_transport_attempt_errors": len(
            list((root / "raw" / "request_events").glob("*_error.json"))
        ),
        "free_disk_before_bytes": free_before,
        "free_disk_after_bytes": final_free_disk_before_transaction,
        "final_free_disk_before_transaction_bytes": final_free_disk_before_transaction,
        "final_200gb_reserve_pass": True,
        "concurrency": MAX_WORKERS,
        "labels_read": False,
        "2024_arrays_read": False,
        "2025_arrays_read": False,
        "models_fit": 0,
        "submission_csv_created": False,
    }
    write_json_exclusive(staged["access"], access)
    decoded_lock = {
        "artifact_type": "TARGET_FREE_DECODED_MATRIX_LOCK",
        "authorization": identity(authorization_path, root),
        "independent_go": identity(go_path, root),
        "coordinate_lock": expected_coordinate_identity,
        "runtime_identity": observed_runtime,
        "runtime_identity_sha256": canonical_payload_sha256(observed_runtime),
        "raw_manifest_parquet": identity_for_destination(
            staged["raw_parquet"], outputs["raw_parquet"], root
        ),
        "raw_manifest_csv": identity_for_destination(
            staged["raw_csv"], outputs["raw_csv"], root
        ),
        "site_matrix": identity_for_destination(
            staged["site"], outputs["site"], root
        ),
        "group_matrix": identity_for_destination(
            staged["group"], outputs["group"], root
        ),
        "physical_and_coverage_audit": identity_for_destination(
            staged["audit"], outputs["audit"], root
        ),
        "access_ledger": identity_for_destination(
            staged["access"], outputs["access"], root
        ),
        "target_free_duplicate_estimator_may_start": pbl_pass or vertical_pass,
        "labels_read": False,
    }
    write_json_exclusive(staged["lock"], decoded_lock)
    manifest = {
        "artifact_type": "NOAA_GFS_MULTISEASON_RAW_AND_DECODE_MANIFEST",
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "runner": identity(runner_path),
        "launch_attempt_id": claim.attempt_id,
        "authorization": identity(authorization_path, root),
        "independent_go": identity(go_path, root),
        "census_manifest": identity(census_manifest_path, root),
        "coordinate_lock": expected_coordinate_identity,
        "runtime_identity": observed_runtime,
        "runtime_identity_sha256": canonical_payload_sha256(observed_runtime),
        "decoded_matrix_lock": identity_for_destination(
            staged["lock"], outputs["lock"], root
        ),
        "progress_checkpoints": [
            identity(path, root)
            for path in sorted((root / "raw" / "progress").glob("*.json"))
        ],
        "request_event_inventory": [
            identity(path, root)
            for path in sorted((root / "raw" / "request_events").glob("*.json"))
        ],
        "exact_ranges": EXPECTED_RANGE_ROWS,
        "exact_bytes": EXPECTED_RANGE_BYTES,
        "labels_read": False,
        "2024_arrays_read": False,
        "2025_arrays_read": False,
        "models_fit": 0,
        "submission_csv_created": False,
    }
    write_json_exclusive(staged["manifest"], manifest)
    require_final_disk_reserve(root)
    transaction_plan = write_output_transaction_plan(
        root, claim.attempt_id, staged, outputs
    )
    transaction_commit = commit_output_transaction(root, transaction_plan, outputs)
    claim.close("complete")
    return {
        "manifest": identity(outputs["manifest"], root),
        "decoded_lock": identity(outputs["lock"], root),
        "transaction_commit": identity(transaction_commit, root),
        "raw_ranges": EXPECTED_RANGE_ROWS,
        "raw_bytes": EXPECTED_RANGE_BYTES,
        "family_status": {key: value["status"] for key, value in audit["families"].items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(run(args.root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

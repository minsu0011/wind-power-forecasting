"""Small reproducibility helpers for artifacts and experiment JSONL logs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence


MANIFEST_SCHEMA_VERSION = 1
EXPERIMENT_SCHEMA_VERSION = 1
DEFAULT_PACKAGES: tuple[str, ...] = (
    "numpy",
    "pandas",
    "scikit-learn",
    "lightgbm",
    "xgboost",
    "catboost",
    "pyarrow",
)


def utc_now() -> str:
    """Return a stable, timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def describe_file(path: str | Path, *, include_hash: bool = True) -> dict[str, Any]:
    """Return identity metadata suitable for an input/output manifest."""

    file_path = Path(path).resolve()
    stat = file_path.stat()
    result: dict[str, Any] = {
        "path": str(file_path),
        "size_bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(
            stat.st_mtime, timezone.utc
        ).isoformat(timespec="seconds"),
    }
    if include_hash:
        result["sha256"] = sha256_file(file_path)
    return result


def package_versions(packages: Sequence[str] = DEFAULT_PACKAGES) -> dict[str, str | None]:
    """Return installed versions; unavailable optional packages map to null."""

    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def git_state(cwd: str | Path | None = None) -> dict[str, Any]:
    """Read Git identity without failing outside a working tree."""

    working_dir = Path(cwd or Path.cwd()).resolve()

    def run(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=working_dir,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip()

    commit = run("rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "dirty": None}
    status = run("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status), "status": status}


def make_manifest(
    *,
    artifact_type: str,
    parameters: Mapping[str, Any],
    input_files: Sequence[str | Path] = (),
    output_files: Sequence[str | Path] = (),
    results: Mapping[str, Any] | None = None,
    project_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Create a versioned artifact manifest with content-addressed files."""

    if not artifact_type.strip():
        raise ValueError("artifact_type must be non-empty")
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_type": artifact_type,
        "created_utc": utc_now(),
        "command": [sys.executable, *sys.argv],
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": package_versions(),
        },
        "git": git_state(project_dir),
        "parameters": dict(parameters),
        "inputs": [describe_file(path) for path in input_files],
        "outputs": [describe_file(path) for path in output_files],
        "results": dict(results or {}),
    }


def write_json_atomic(
    path: str | Path, payload: Mapping[str, Any], *, overwrite: bool = False
) -> Path:
    """Atomically write formatted UTF-8 JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def make_experiment_record(
    *,
    experiment_id: str,
    status: str,
    validation: Mapping[str, Any],
    model: Mapping[str, Any],
    dev_metrics: Mapping[str, Any] | None = None,
    leaderboard_metrics: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, str] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Create one experiment record while keeping dev and leaderboard separate."""

    allowed_status = {"planned", "running", "completed", "failed"}
    if status not in allowed_status:
        raise ValueError(f"status must be one of {sorted(allowed_status)}")
    if not experiment_id.strip():
        raise ValueError("experiment_id must be non-empty")
    return {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "created_utc": utc_now(),
        "status": status,
        "validation": dict(validation),
        "model": dict(model),
        "dev_metrics": dict(dev_metrics or {}),
        "leaderboard_metrics": dict(leaderboard_metrics or {}),
        "artifacts": dict(artifacts or {}),
        "notes": notes,
    }


def append_experiment_jsonl(path: str | Path, record: Mapping[str, Any]) -> Path:
    """Append one validated JSON object to an experiment log."""

    required = {
        "schema_version",
        "experiment_id",
        "created_utc",
        "status",
        "validation",
        "model",
        "dev_metrics",
        "leaderboard_metrics",
        "artifacts",
        "notes",
    }
    missing = required.difference(record)
    if missing:
        raise ValueError(f"experiment record missing fields: {sorted(missing)}")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True))
        stream.write("\n")
    return destination


__all__ = (
    "DEFAULT_PACKAGES",
    "EXPERIMENT_SCHEMA_VERSION",
    "MANIFEST_SCHEMA_VERSION",
    "append_experiment_jsonl",
    "describe_file",
    "git_state",
    "make_experiment_record",
    "make_manifest",
    "package_versions",
    "sha256_file",
    "utc_now",
    "write_json_atomic",
)

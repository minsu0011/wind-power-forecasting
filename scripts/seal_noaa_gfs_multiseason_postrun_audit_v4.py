#!/usr/bin/env python
"""Seal the append-only V4 postrun-audit authorization.

This publisher is deliberately narrower than the auditor it authorizes.  It
hashes already-frozen controls and recovered outputs, compiles frozen sources,
runs five test files in isolated network-denied subprocesses, and publishes
exactly one create-if-absent JSON file.  It never publishes the independent
review or GO, executes the production postrun audit, materializes a production
output/target value, or opens a network client.  The bound auditor test process
owns its synthetic and frozen-pilot regressions.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
from typing import Any, Mapping, Sequence


# Make direct-file execution reach the fail-closed entrypoint check instead of
# failing during package import.  Publication still requires canonical ``-m``.
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


AUDITOR_SOURCE_PATH = REPO / "scripts" / "audit_noaa_gfs_multiseason_raw_postrun_v4.py"


def _require_preimport_cache_runtime() -> None:
    if not (
        os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
        and sys.dont_write_bytecode is True
        and "PYTHONPYCACHEPREFIX" not in os.environ
        and sys.pycache_prefix is None
    ):
        raise RuntimeError(
            "V4 sealer import requires -B, PYTHONDONTWRITEBYTECODE=1, "
            "and no pycache prefix"
        )


def _matching_python_bytecode(path: Path) -> list[Path]:
    """Return every legacy/cache-tag bytecode competitor without following it."""

    matches: list[Path] = []
    legacy = path.with_suffix(".pyc")
    if os.path.lexists(str(legacy)):
        matches.append(legacy)
    cache = path.parent / "__pycache__"
    if os.path.lexists(str(cache)):
        if cache.is_symlink() or (
            hasattr(cache, "is_junction") and cache.is_junction()
        ) or not cache.is_dir():
            matches.append(cache)
        else:
            prefix = f"{path.stem}.".casefold()
            for candidate in cache.iterdir():
                folded = candidate.name.casefold()
                if folded.startswith(prefix) and folded.endswith(".pyc"):
                    matches.append(candidate)
    return sorted(matches, key=lambda candidate: str(candidate).casefold())


def _require_auditor_bytecode_absent(path: Path) -> None:
    matches = _matching_python_bytecode(path)
    if matches:
        raise RuntimeError(
            "V4 auditor matching bytecode/cache competitor exists: "
            + ", ".join(str(candidate) for candidate in matches)
        )


def _require_bound_bytecode_absent(
    bound: Mapping[str, Mapping[str, Any]],
) -> None:
    required = {
        "v2_auditor", "v2_auditor_test",
        "superseded_v3_auditor", "superseded_v3_auditor_test",
        "superseded_v3_sealer", "superseded_v3_sealer_test",
        "v4_auditor", "v4_auditor_test", "v4_sealer", "v4_sealer_test",
    }
    if set(bound) != required:
        raise PostrunAuditSealError("bound executable-source role set mismatch")
    matches: dict[str, list[str]] = {}
    for role in sorted(required):
        path = Path(str(bound[role].get("path", "")))
        if not path.is_absolute():
            raise PostrunAuditSealError(f"bound executable source is relative: {role}")
        found = _matching_python_bytecode(path)
        if found:
            matches[role] = [str(candidate) for candidate in found]
    if matches:
        raise PostrunAuditSealError(
            "bound executable-source bytecode/cache competitor exists: "
            + json.dumps(matches, sort_keys=True)
        )


def _auditor_source_competitors(path: Path) -> list[Path]:
    parent = path.parent
    if parent.is_symlink() or (
        hasattr(parent, "is_junction") and parent.is_junction()
    ) or not parent.is_dir():
        return [parent]
    stem = path.stem.casefold()
    competitors: list[Path] = []
    for candidate in parent.iterdir():
        folded = candidate.name.casefold()
        is_alias = folded == stem or folded.startswith(f"{stem}.")
        if not is_alias:
            continue
        if (
            candidate == path
            and candidate.name == path.name
            and candidate.is_file()
            and not candidate.is_symlink()
            and not (hasattr(candidate, "is_junction") and candidate.is_junction())
        ):
            continue
        competitors.append(candidate)
    return sorted(competitors, key=lambda candidate: candidate.name.casefold())


def _require_auditor_source_competitors_absent(path: Path) -> None:
    competitors = _auditor_source_competitors(path)
    if competitors:
        raise RuntimeError(
            "V4 auditor same-stem source-origin competitor exists: "
            + ", ".join(str(candidate) for candidate in competitors)
        )


def _preimport_auditor_source_guard(path: Path) -> None:
    """Reject top-level I/O/network routes before importing the moving V4 source."""

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".", 1)[0])
    forbidden_imports = {
        "socket", "requests", "urllib", "http", "ftplib", "httpx", "aiohttp",
        "pandas", "pyarrow", "eccodes",
    }
    if imported & forbidden_imports:
        raise RuntimeError("V4 auditor has a forbidden import-time route")
    allowed_top_level = {
        ast.Expr, ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign,
        ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.If,
    }
    if any(type(node) not in allowed_top_level for node in tree.body):
        raise RuntimeError("V4 auditor has an unexpected top-level statement")
    for node in tree.body:
        if isinstance(node, ast.Expr) and not isinstance(node.value, ast.Constant):
            raise RuntimeError("V4 auditor has a top-level expression side effect")
        if isinstance(node, ast.If):
            if (
                not isinstance(node.test, ast.Compare)
                or ast.unparse(node.test) != "__name__ == '__main__'"
                or node.orelse
            ):
                raise RuntimeError("V4 auditor has an unexpected top-level branch")
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            for call in (item for item in ast.walk(node) if isinstance(item, ast.Call)):
                function = ast.unparse(call.func)
                allowed = (
                    function in {"Path", "tuple", "str", "AuditContract", "GroupContract", "re.compile"}
                    or function.endswith((".resolve", ".replace", ".strip"))
                )
                if not allowed:
                    raise RuntimeError(
                        f"V4 auditor has an unapproved import-time call: {function}"
                    )


_require_preimport_cache_runtime()
_require_auditor_bytecode_absent(Path(__file__).resolve())
_require_auditor_bytecode_absent(AUDITOR_SOURCE_PATH)
_require_auditor_source_competitors_absent(AUDITOR_SOURCE_PATH)
_preimport_auditor_source_guard(AUDITOR_SOURCE_PATH)

from scripts import audit_noaa_gfs_multiseason_raw_postrun_v4 as auditor

if not (
    auditor.__name__ == "scripts.audit_noaa_gfs_multiseason_raw_postrun_v4"
    and auditor.__package__ == "scripts"
    and auditor.__spec__ is not None
    and auditor.__spec__.name == auditor.__name__
    and Path(auditor.__file__).resolve() == AUDITOR_SOURCE_PATH.resolve()
):
    raise RuntimeError("V4 auditor import source origin mismatch")


ROOT_DEFAULT = auditor.ROOT_DEFAULT
CANONICAL_MODULE_ENTRYPOINT = (
    "scripts.seal_noaa_gfs_multiseason_postrun_audit_v4"
)
AUTH_RELATIVE = auditor.V4_AUTH_RELATIVE
REVIEW_RELATIVE = auditor.V4_REVIEW_RELATIVE
GO_RELATIVE = auditor.V4_GO_RELATIVE
SEALER_TEST = REPO / "tests" / "test_seal_noaa_gfs_multiseason_postrun_audit_v4.py"

TEST_RELATIVES = {
    "v4_auditor_tests": "tests/test_noaa_gfs_multiseason_raw_postrun_v4.py",
    "v4_sealer_tests": "tests/test_seal_noaa_gfs_multiseason_postrun_audit_v4.py",
    "frozen_v3_auditor_tests": "tests/test_noaa_gfs_multiseason_raw_postrun_v3.py",
    "frozen_v3_sealer_tests": "tests/test_seal_noaa_gfs_multiseason_postrun_audit_v3.py",
    "frozen_v2_auditor_tests": "tests/test_noaa_gfs_multiseason_raw_postrun_v2.py",
}
BOUND_SOURCE_PATHS = {
    "superseded_v3_auditor": auditor.FROZEN_V3_AUDITOR.resolve(),
    "superseded_v3_auditor_test": auditor.FROZEN_V3_AUDITOR_TEST.resolve(),
    "superseded_v3_sealer": auditor.FROZEN_V3_SEALER.resolve(),
    "superseded_v3_sealer_test": auditor.FROZEN_V3_SEALER_TEST.resolve(),
    "v4_auditor": Path(auditor.__file__).resolve(),
    "v4_auditor_test": auditor.AUDITOR_TEST.resolve(),
    "v4_sealer": Path(__file__).resolve(),
    "v4_sealer_test": SEALER_TEST.resolve(),
}
EXPECTED_V2_SUPPORT_PATHS = {
    "recovery_runner": REPO / "scripts" / "run_noaa_gfs_multiseason_decode_recovery_v2.py",
    "recovery_runner_test": REPO / "tests" / "test_noaa_gfs_multiseason_decode_recovery_runner_v2.py",
    "recovery_sealer": REPO / "scripts" / "seal_noaa_gfs_multiseason_decode_recovery_v2.py",
    "recovery_sealer_test": REPO / "tests" / "test_seal_noaa_gfs_multiseason_decode_recovery_v2.py",
    "recovery_module": REPO / "src" / "noaa_gfs_decode_recovery.py",
    "recovery_bootstrap": REPO / "src" / "noaa_gfs_decode_recovery_bootstrap.py",
    "recovery_test": REPO / "tests" / "test_noaa_gfs_decode_recovery.py",
    "recovery_bootstrap_test": REPO / "tests" / "test_noaa_gfs_decode_recovery_bootstrap.py",
}
POSTRUN_REPORT_CANDIDATES = auditor.V4_POSTRUN_REPORT_CANDIDATES
CONTROL_TEMP_RELATIVES = (
    auditor.FROZEN_V3_AUTH_RELATIVE,
    auditor.FROZEN_V3_REVIEW_RELATIVE,
    auditor.FROZEN_V3_CANONICAL_GO_RELATIVE,
    auditor.FROZEN_V3_MISPLACED_GO_RELATIVE,
    AUTH_RELATIVE,
    REVIEW_RELATIVE,
    GO_RELATIVE,
    auditor.V2_FALSE_REJECT_INCIDENT_RELATIVE,
    auditor.V4_PATH_MISPUBLISH_INCIDENT_RELATIVE,
)
BOUND_SOURCE_ROLES = {
    "v2_auditor", "v2_auditor_test",
    "superseded_v3_auditor", "superseded_v3_auditor_test",
    "superseded_v3_sealer", "superseded_v3_sealer_test",
    "v4_auditor", "v4_auditor_test", "v4_sealer", "v4_sealer_test",
}
SOURCE_ORIGIN_TRUST_MODEL = {
    "model": "NON_ADVERSARIAL_NO_CONCURRENT_FILESYSTEM_SWAP",
    "preimport_source_ast_and_competitor_scan": True,
    "pretest_posttest_and_prepublish_ten_source_rehash_and_pyc_scan": True,
    "adversarial_concurrent_swap_resistance_claimed": False,
}


class PostrunAuditSealError(RuntimeError):
    """The immutable V4 authorization could not be sealed safely."""


class PostrunAuditPostPublicationError(PostrunAuditSealError):
    """AUTH was created by this invocation before a later fail-closed check."""

    def __init__(self, message: str, authorization_identity: Mapping[str, Any]):
        super().__init__(message)
        self.authorization_identity = dict(authorization_identity)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PostrunAuditSealError(message)


def _lexists(path: Path) -> bool:
    return os.path.lexists(str(path))


def _linklike(path: Path) -> bool:
    return path.is_symlink() or (
        hasattr(path, "is_junction") and path.is_junction()
    )


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def utc_now_exact() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_no_link_chain(path: Path, boundary: Path, *, label: str) -> Path:
    lexical = Path(os.path.abspath(path))
    root = Path(os.path.abspath(boundary))
    _require(
        lexical == root or root in lexical.parents,
        f"{label} escapes the authorized boundary",
    )
    current = lexical
    while True:
        _require(not _linklike(current), f"{label} has a symlink/junction: {current}")
        if current == root:
            break
        current = current.parent
    return lexical


def root_path(root: Path, relative: str, *, label: str) -> Path:
    _require(
        isinstance(relative, str)
        and relative
        and not Path(relative).is_absolute()
        and ".." not in Path(relative).parts,
        f"{label} relative path is invalid",
    )
    return _require_no_link_chain(root / Path(relative), root, label=label)


def strict_root_identity(root: Path, relative: str, *, label: str) -> dict[str, Any]:
    path = root_path(root, relative, label=label)
    _require(path.is_file() and not _linklike(path), f"{label} is not an exact file")
    return {
        "path": relative.replace("\\", "/"),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def strict_external_identity(path: Path, *, label: str) -> dict[str, Any]:
    raw = Path(path)
    _require(raw.is_absolute(), f"{label} is not absolute")
    lexical = _require_no_link_chain(raw, Path(raw.anchor), label=label)
    _require(lexical.is_file() and not _linklike(lexical), f"{label} is not an exact file")
    resolved = lexical.resolve(strict=True)
    _require(resolved == lexical, f"{label} canonical path mismatch")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def load_json_control(path: Path, *, label: str) -> dict[str, Any]:
    _require(path.is_file() and not _linklike(path), f"{label} is not an exact file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PostrunAuditSealError(f"{label} JSON is invalid") from exc
    _require(isinstance(value, dict), f"{label} is not an object")
    return value


def require_canonical_module_entrypoint(root: Path) -> None:
    _require(
        __name__ == "__main__"
        and __package__ == "scripts"
        and __spec__ is not None
        and __spec__.name == CANONICAL_MODULE_ENTRYPOINT,
        "publication requires canonical `python -B -m scripts."
        "seal_noaa_gfs_multiseason_postrun_audit_v4` entrypoint",
    )
    _require(
        os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
        and sys.dont_write_bytecode is True
        and "PYTHONPYCACHEPREFIX" not in os.environ
        and sys.pycache_prefix is None,
        "publication requires -B, PYTHONDONTWRITEBYTECODE=1, and no pycache prefix",
    )
    expected = [str(Path(__file__).resolve()), "--root", str(root.resolve())]
    actual = [str(Path(sys.argv[0]).resolve()), *sys.argv[1:]]
    _require(actual == expected, "publication argv is not exact canonical order")
    expected_orig = [
        str(Path(sys.executable).resolve()),
        "-B",
        "-m",
        CANONICAL_MODULE_ENTRYPOINT,
        "--root",
        str(root.resolve()),
    ]
    actual_orig = [
        str(Path(sys.orig_argv[0]).resolve()),
        *sys.orig_argv[1:],
    ]
    _require(actual_orig == expected_orig, "publication orig_argv is not exact -B module command")


class _NetworkDenied(RuntimeError):
    pass


def _network_forbidden(*_args: Any, **_kwargs: Any) -> Any:
    raise _NetworkDenied("network access is forbidden during V4 authorization sealing")


_NETWORK_GUARD_INSTALLED = False


def _network_audit_guard(event: str, _args: tuple[Any, ...]) -> None:
    if event in {
        "socket.__new__", "socket.bind", "socket.connect", "socket.connect_ex",
        "socket.getaddrinfo", "socket.gethostbyaddr", "socket.gethostbyname",
        "socket.getnameinfo", "socket.sendmsg", "socket.sendto",
    }:
        raise _NetworkDenied(f"network event forbidden: {event}")


def install_entry_guard() -> None:
    global _NETWORK_GUARD_INSTALLED
    if not _NETWORK_GUARD_INSTALLED:
        sys.addaudithook(_network_audit_guard)
        _NETWORK_GUARD_INSTALLED = True
    socket.create_connection = _network_forbidden  # type: ignore[assignment]
    socket.getaddrinfo = _network_forbidden  # type: ignore[assignment]
    socket.gethostbyaddr = _network_forbidden  # type: ignore[assignment]
    socket.gethostbyname = _network_forbidden  # type: ignore[assignment]
    socket.getnameinfo = _network_forbidden  # type: ignore[assignment]


def control_paths(root: Path) -> dict[str, Path]:
    return {
        "authorization": root_path(root, AUTH_RELATIVE, label="V4 authorization"),
        "independent_review": root_path(root, REVIEW_RELATIVE, label="V4 review"),
        "independent_go": root_path(root, GO_RELATIVE, label="V4 GO"),
    }


def control_temporary_files(root: Path) -> list[str]:
    matches: list[str] = []
    for relative in CONTROL_TEMP_RELATIVES:
        path = root_path(root, relative, label=f"V4 control temp destination {relative}")
        parent = path.parent
        _require(parent.is_dir() and not _linklike(parent), "V4 control parent invalid")
        folded = path.name.casefold()
        for sibling in parent.iterdir():
            name = sibling.name.casefold()
            if sibling.name == path.name:
                continue
            if (
                name.startswith(f".{folded}.")
                or name.startswith(f"{folded}.")
                or name == f"{folded}.tmp"
            ):
                matches.append(sibling.relative_to(root).as_posix())
    return sorted(set(matches))


def _expected_control_namespace(*, authorization_present: bool) -> dict[str, set[str]]:
    prereg = {
        Path(auditor.FROZEN_V3_AUTH_RELATIVE).name,
        Path(auditor.FROZEN_V3_MISPLACED_GO_RELATIVE).name,
    }
    if authorization_present:
        prereg.add(Path(AUTH_RELATIVE).name)
    return {
        "prereg": prereg,
        "independent_redteam": {Path(auditor.FROZEN_V3_REVIEW_RELATIVE).name},
        "incidents": {
            Path(auditor.V2_FALSE_REJECT_INCIDENT_RELATIVE).name,
            Path(auditor.V4_PATH_MISPUBLISH_INCIDENT_RELATIVE).name,
        },
    }


def validate_control_namespace(
    root: Path, *, authorization_present: bool
) -> dict[str, list[str]]:
    """Close every matching auditor-control namespace before or after AUTH."""

    prefixes = {
        "prereg": "decode_recovery_postrun_audit_",
        "independent_redteam": "track_a_decode_recovery_postrun_auditor_",
        "incidents": "track_a_decode_recovery_postrun_auditor_",
    }
    expected = _expected_control_namespace(
        authorization_present=authorization_present
    )
    canonical_v3_go = root_path(
        root,
        auditor.FROZEN_V3_CANONICAL_GO_RELATIVE,
        label="canonical V3 GO",
    )
    _require(
        not _lexists(canonical_v3_go),
        "canonical V3 GO must remain absent permanently",
    )
    result: dict[str, list[str]] = {}
    for relative, prefix in prefixes.items():
        parent = root_path(root, relative, label=f"V4 namespace {relative}")
        _require(
            _lexists(parent) and parent.is_dir() and not _linklike(parent),
            f"V4 control namespace parent invalid: {relative}",
        )
        matched: list[str] = []
        for entry in parent.iterdir():
            if not entry.name.casefold().lstrip(".").startswith(prefix):
                continue
            _require(
                _lexists(entry) and entry.is_file() and not _linklike(entry),
                f"V4 control namespace contains non-regular/link entry: {entry}",
            )
            matched.append(entry.name)
        _require(
            set(matched) == expected[relative]
            and len({name.casefold() for name in matched}) == len(matched),
            f"V4 control namespace inventory mismatch: {relative}",
        )
        result[relative] = sorted(matched, key=str.casefold)
    return result


def require_prepublication_control_state(root: Path) -> dict[str, Any]:
    paths = control_paths(root)
    _require(
        all(not _lexists(path) for path in paths.values()),
        "V4 authorization/review/GO already exists",
    )
    _require(control_temporary_files(root) == [], "V4 control temporary sibling exists")
    namespace = validate_control_namespace(root, authorization_present=False)
    return {
        "authorization_present": False,
        "independent_review_present": False,
        "independent_go_present": False,
        "temporary_siblings": [],
        "control_namespace": namespace,
    }


def require_postpublication_control_state(root: Path) -> dict[str, Any]:
    paths = control_paths(root)
    _require(
        _lexists(paths["authorization"])
        and paths["authorization"].is_file()
        and not _linklike(paths["authorization"]),
        "published V4 authorization is absent, non-regular or link-like",
    )
    _require(
        not _lexists(paths["independent_review"])
        and not _lexists(paths["independent_go"]),
        "V4 sealer observed an independent review/GO",
    )
    _require(control_temporary_files(root) == [], "V4 control temporary sibling exists")
    namespace = validate_control_namespace(root, authorization_present=True)
    return {
        "authorization_present": True,
        "independent_review_present": False,
        "independent_go_present": False,
        "temporary_siblings": [],
        "control_namespace": namespace,
    }


def collect_code_state(workspace_root: Path) -> dict[str, Any]:
    workspace = Path(os.path.abspath(workspace_root))
    _require(workspace == REPO, "workspace root mismatch")
    try:
        _require_auditor_bytecode_absent(AUDITOR_SOURCE_PATH)
        _require_auditor_source_competitors_absent(AUDITOR_SOURCE_PATH)
    except RuntimeError as exc:
        raise PostrunAuditSealError(str(exc)) from exc
    _require(Path(auditor.V4_SEALER).resolve() == Path(__file__).resolve(), "auditor V4 sealer path mismatch")
    _require(Path(auditor.V4_SEALER_TEST).resolve() == SEALER_TEST.resolve(), "auditor V4 sealer-test path mismatch")

    bound = {
        "v2_auditor": strict_external_identity(auditor.SUPERSEDED_V2_AUDITOR, label="V2 auditor"),
        "v2_auditor_test": strict_external_identity(auditor.SUPERSEDED_V2_AUDITOR_TEST, label="V2 auditor test"),
        **{
            role: strict_external_identity(path, label=role)
            for role, path in BOUND_SOURCE_PATHS.items()
        },
    }
    _require(set(bound) == BOUND_SOURCE_ROLES, "bound source role schema mismatch")
    _require(bound["v2_auditor"] == auditor.V4_SUPERSEDED_V2_AUDITOR_IDENTITY, "V2 auditor identity mismatch")
    _require(bound["v2_auditor_test"] == auditor.V4_SUPERSEDED_V2_AUDITOR_TEST_IDENTITY, "V2 auditor-test identity mismatch")
    for role, expected in auditor.FROZEN_V3_SOURCE_IDENTITIES.items():
        _require(bound[role] == expected, f"frozen V3 source identity mismatch: {role}")

    support: dict[str, dict[str, Any]] = {}
    _require(
        set(auditor.V4_V2_SUPPORT_PATHS)
        == set(auditor.V4_V2_SUPPORT_SIZE_SHA256)
        == set(EXPECTED_V2_SUPPORT_PATHS),
        "V2 support exact-eight role contract mismatch",
    )
    for role, path in auditor.V4_V2_SUPPORT_PATHS.items():
        _require(
            Path(path).resolve() == EXPECTED_V2_SUPPORT_PATHS[role].resolve(),
            f"V2 support canonical path mismatch: {role}",
        )
        record = strict_external_identity(path, label=f"V2 support {role}")
        size, digest = auditor.V4_V2_SUPPORT_SIZE_SHA256[role]
        _require(record["size_bytes"] == size and record["sha256"] == digest, f"V2 support identity mismatch: {role}")
        support[role] = record
    _require_bound_bytecode_absent(bound)
    return {"bound_identities": bound, "v2_recovery_support": support}


def _verify_root_map(root: Path, expected: Mapping[str, Mapping[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    actual: dict[str, dict[str, Any]] = {}
    for role, record in expected.items():
        _require(isinstance(record, Mapping) and set(record) == {"path", "size_bytes", "sha256"}, f"{label} contract schema mismatch: {role}")
        observed = strict_root_identity(root, str(record["path"]), label=f"{label} {role}")
        _require(observed == dict(record), f"{label} identity mismatch: {role}")
        actual[role] = observed
    return actual


def collect_progress(root: Path) -> dict[str, Any]:
    directory = root_path(root, "decoded/recovery_progress", label="recovery progress")
    _require(directory.is_dir() and not _linklike(directory), "recovery progress directory invalid")
    entries = sorted(directory.iterdir(), key=lambda path: path.name)
    _require(
        len(entries) == 104
        and all(path.is_file() and not _linklike(path) for path in entries),
        "recovery progress exact inventory mismatch",
    )
    inventory = [
        strict_root_identity(
            root, path.relative_to(root).as_posix(), label=f"recovery progress {path.name}"
        )
        for path in entries
    ]
    declaration = {
        "file_count": 104,
        "inventory": inventory,
        "inventory_canonical_sha256": hashlib.sha256(canonical_bytes(inventory)).hexdigest(),
        "final_checkpoint": dict(auditor.V4_FINAL_PROGRESS_IDENTITY),
    }
    auditor._v4_validate_progress_declaration(declaration)
    return declaration


def validate_recovered_filesystem_closure(
    root: Path,
    declaration: Mapping[str, Any],
) -> dict[str, Any]:
    """Reclose the exact recovered history and transaction trees prepublish."""

    history = declaration["recovery_history"]
    history_dir = root_path(root, "decoded/recovery_history", label="recovery history")
    _require(
        history_dir.is_dir() and not _linklike(history_dir),
        "recovery-history directory invalid",
    )
    expected_history_paths = {
        str(record["path"])
        for record in history["completion_locks"].values()
    } | {str(history["postcommit_input_audit"]["path"])}
    history_entries = list(history_dir.iterdir())
    _require(
        len(history_entries) == 3
        and all(path.is_file() and not _linklike(path) for path in history_entries)
        and {path.relative_to(root).as_posix() for path in history_entries}
        == expected_history_paths,
        "recovery-history exact filesystem closure mismatch",
    )

    transaction_root = root_path(
        root, "raw/output_transactions", label="output transaction root"
    )
    _require(
        transaction_root.is_dir() and not _linklike(transaction_root),
        "output-transaction root invalid",
    )
    recovery_authorization_relative = auditor.V4_V2_CONTROL_RELATIVES[
        "recovery_authorization_v2"
    ]
    recovery_authorization = load_json_control(
        root_path(
            root,
            recovery_authorization_relative,
            label="V2 recovery authorization",
        ),
        label="V2 recovery authorization",
    )
    remnants = recovery_authorization.get("documented_preplan_remnants")
    _require(
        isinstance(remnants, list) and len(remnants) == 2,
        "incident-bound preplan-remnant declaration mismatch",
    )
    expected_transaction_files = {
        str(declaration["recovery_transaction"]["plan"]["path"]),
        str(declaration["recovery_transaction"]["commit"]["path"]),
    }
    for index, record in enumerate(remnants):
        _require(
            isinstance(record, Mapping)
            and set(record) == {"path", "size_bytes", "sha256"},
            f"preplan remnant identity invalid: {index}",
        )
        observed = strict_root_identity(
            root,
            str(record["path"]),
            label=f"incident-bound preplan remnant {index}",
        )
        _require(
            observed == dict(record),
            f"incident-bound preplan remnant identity mismatch: {index}",
        )
        expected_transaction_files.add(str(record["path"]))
    expected_transaction_dirs = {
        "raw/output_transactions/20260810T151057590738Z__pid3536__5f3e0fb20132",
        "raw/output_transactions/20260810T151057590738Z__pid3536__5f3e0fb20132/staged",
        "raw/output_transactions/20260810T151057590738Z__pid3536__5f3e0fb20132/staged/raw",
        f"raw/output_transactions/{auditor.V4_RECOVERY_ATTEMPT_ID}",
        f"raw/output_transactions/{auditor.V4_RECOVERY_ATTEMPT_ID}/staged",
        f"raw/output_transactions/{auditor.V4_RECOVERY_ATTEMPT_ID}/staged/decoded",
        f"raw/output_transactions/{auditor.V4_RECOVERY_ATTEMPT_ID}/staged/raw",
    }
    transaction_files: set[str] = set()
    transaction_dirs: set[str] = set()
    for path in transaction_root.rglob("*"):
        _require(
            not _linklike(path),
            "output-transaction tree contains a link/junction",
        )
        relative = path.relative_to(root).as_posix()
        if path.is_file():
            transaction_files.add(relative)
        elif path.is_dir():
            transaction_dirs.add(relative)
        else:
            raise PostrunAuditSealError(
                "output-transaction tree contains a special object"
            )
    _require(
        transaction_files == expected_transaction_files
        and transaction_dirs == expected_transaction_dirs
        and len(transaction_files) == 4,
        "output-transaction recursive exact filesystem closure mismatch",
    )
    return {
        "recovery_history_file_count": len(history_entries),
        "transaction_recursive_file_count": len(transaction_files),
        "transaction_recursive_directory_count": len(transaction_dirs),
        "original_preplan_remnant_count": len(remnants),
        "v2_plan_file_count": 1,
        "v2_commit_file_count": 1,
        "v2_staged_files_remaining": 0,
    }


def build_zero_snapshot(root: Path) -> dict[str, Any]:
    active = {
        "raw": {"path": "raw/RAW_LAUNCH_ACTIVE.lock", "present": _lexists(root / "raw/RAW_LAUNCH_ACTIVE.lock")},
        "decoded": {"path": "decoded/DECODE_RECOVERY_ACTIVE.lock", "present": _lexists(root / "decoded/DECODE_RECOVERY_ACTIVE.lock")},
    }
    temporary = control_temporary_files(root)
    reports = [relative for relative in POSTRUN_REPORT_CANDIDATES if _lexists(root / relative)]
    _require(not any(item["present"] for item in active.values()), "active producer/recovery lock exists")
    _require(temporary == [], "V4 control temporary sibling exists")
    _require(reports == [], "postrun audit output file exists despite stdout-only contract")
    snapshot = {
        "incident_postfailure_state_canonical_sha256": auditor.V4_INCIDENT_POSTFAILURE_STATE_CANONICAL_SHA256,
        "recovered_afterstate_canonical_sha256": auditor.V4_RECOVERED_AFTERSTATE_CANONICAL_SHA256,
        "active_locks": active,
        "v4_control_temporary_files": {
            "checked_destination_relative_paths": list(CONTROL_TEMP_RELATIVES),
            "matching_temporary_files": 0,
        },
        "postrun_audit_output_files_present": 0,
        "network_requests": 0,
        "audit_files_written": 0,
        "labels_read": False,
        "arrays_2024_read": False,
        "arrays_2025_read": False,
        "models_fit": 0,
        "submission_csv_created": False,
    }
    auditor._v4_validate_zero_snapshot(
        snapshot,
        incident_postfailure_sha256=auditor.V4_INCIDENT_POSTFAILURE_STATE_CANONICAL_SHA256,
    )
    return snapshot


def collect_root_inputs(root: Path, code_state: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(os.path.abspath(root))
    _require(root.is_dir() and not _linklike(root), "artifact root invalid")
    incident = {
        "path": auditor.V4_PATH_MISPUBLISH_INCIDENT_RELATIVE,
        "size_bytes": auditor.V4_PATH_MISPUBLISH_INCIDENT_SIZE_BYTES,
        "sha256": auditor.V4_PATH_MISPUBLISH_INCIDENT_SHA256,
    }
    v2_false_reject_incident = {
        "path": auditor.V2_FALSE_REJECT_INCIDENT_RELATIVE,
        "size_bytes": auditor.V2_FALSE_REJECT_INCIDENT_SIZE_BYTES,
        "sha256": auditor.V2_FALSE_REJECT_INCIDENT_SHA256,
    }
    auditor._v4_validate_path_mispublish_incident(root, incident)
    auditor._v4_validate_false_reject_incident(root, v2_false_reject_incident)
    frozen_v3 = auditor._v4_validate_frozen_v3_chain(root)
    _require(
        frozen_v3.get("source_identities")
        == {
            role: code_state["bound_identities"][role]
            for role in auditor.FROZEN_V3_SOURCE_IDENTITIES
        },
        "frozen V3 source identities changed during root collection",
    )
    validate_control_namespace(
        root, authorization_present=_lexists(root / AUTH_RELATIVE)
    )

    controls = _verify_root_map(root, auditor.V4_V2_CONTROL_IDENTITIES, label="V2 control")
    _require(code_state.get("v2_recovery_support") == collect_code_state(REPO)["v2_recovery_support"], "V2 support changed during root collection")

    original_raw = dict(auditor.V4_ORIGINAL_V1_PROVENANCE["original_raw_authorization_v1"])
    raw_actual = strict_root_identity(root, str(original_raw["path"]), label="original V1 raw authorization")
    _require(raw_actual == original_raw, "original V1 raw authorization identity mismatch")
    prelaunch = dict(auditor.V4_ORIGINAL_V1_PROVENANCE["original_independent_prelaunch_audit_v1"])
    prelaunch_three = {key: prelaunch[key] for key in ("path", "size_bytes", "sha256")}
    prelaunch_actual = strict_root_identity(root, str(prelaunch["path"]), label="original independent prelaunch audit")
    _require(prelaunch_actual == prelaunch_three and prelaunch.get("status") == "PASS_TO_CREATE_FINAL_AUTH_ONLY", "original four-key prelaunch identity mismatch")
    raw_payload = load_json_control(root_path(root, str(original_raw["path"]), label="original raw authorization"), label="original raw authorization")
    _require(raw_payload.get("independent_prelaunch_audit") == prelaunch, "original authorization does not bind four-key prelaunch record")

    transaction = _verify_root_map(root, auditor.V4_TRANSACTION_IDENTITIES, label="recovery transaction")
    locks = _verify_root_map(root, auditor.V4_COMPLETION_LOCK_IDENTITIES, label="recovery completion lock")
    postcommit = strict_root_identity(root, str(auditor.V4_POSTCOMMIT_INPUT_AUDIT_IDENTITY["path"]), label="postcommit input audit")
    _require(postcommit == auditor.V4_POSTCOMMIT_INPUT_AUDIT_IDENTITY, "postcommit input-audit identity mismatch")
    outputs = _verify_root_map(root, auditor.V4_CANONICAL_OUTPUT_IDENTITIES, label="canonical output")
    progress = collect_progress(root)
    declaration = {
        "recovery_transaction": transaction,
        "recovery_history": {
            "file_count": 3,
            "completion_locks": locks,
            "postcommit_input_audit": postcommit,
        },
        "recovery_progress": progress,
        "canonical_outputs": outputs,
    }
    afterstate = auditor._v4_recovered_afterstate_from_authorization(declaration)
    validate_recovered_filesystem_closure(root, declaration)
    snapshot = build_zero_snapshot(root)
    return {
        "incident": incident,
        "v2_false_reject_incident": v2_false_reject_incident,
        "v3_path_mispublish_state": frozen_v3["state"],
        "v2_recovery_controls": controls,
        "original_v1_provenance": {
            "original_raw_authorization_v1": raw_actual,
            "original_independent_prelaunch_audit_v1": prelaunch,
        },
        **declaration,
        "recovered_afterstate": afterstate,
        "preaudit_zero_mutation_snapshot": snapshot,
    }


def _source_function_names(path: Path) -> set[str]:
    source = path.read_text(encoding="utf-8")
    return {
        node.name
        for node in ast.walk(ast.parse(source, filename=str(path)))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _compile_bound_sources(bound: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered_roles = ("v4_auditor", "v4_auditor_test", "v4_sealer", "v4_sealer_test")
    compiled: list[dict[str, Any]] = []
    for role in ordered_roles:
        path = Path(str(bound[role]["path"]))
        source = path.read_bytes()
        compile(source, str(path), "exec", dont_inherit=True)
        observed = strict_external_identity(path, label=f"compile source {role}")
        _require(observed == dict(bound[role]), f"compile source changed: {role}")
        compiled.append(observed)
    return compiled


def _pytest_summary(stdout: str) -> str:
    lines = [line.strip() for line in stdout.replace("\r\n", "\n").splitlines() if line.strip()]
    return next((line for line in reversed(lines) if " passed" in line), "")


def run_immutable_test_evidence(
    root: Path, workspace_root: Path, code_state: Mapping[str, Any]
) -> dict[str, Any]:
    require_canonical_module_entrypoint(root)
    bound = code_state["bound_identities"]
    compiled = _compile_bound_sources(bound)
    auditor_test_names = _source_function_names(Path(str(bound["v4_auditor_test"]["path"])))
    _require(set(auditor.V4_REQUIRED_TEST_NAMES).issubset(auditor_test_names), "required V4 regression test is absent")

    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.pop("PYTHONPYCACHEPREFIX", None)
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    environment["PYTHONIOENCODING"] = "utf-8"
    runs: dict[str, dict[str, Any]] = {}
    for role, relative in TEST_RELATIVES.items():
        _require_bound_bytecode_absent(bound)
        command = auditor._v4_test_command(relative)
        completed = subprocess.run(
            command,
            cwd=workspace_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=1800,
        )
        stdout = completed.stdout.replace("\r\n", "\n")
        stderr = completed.stderr.replace("\r\n", "\n")
        summary = _pytest_summary(stdout)
        _require(completed.returncode == 0 and summary and "failed" not in summary.casefold() and "error" not in summary.casefold(), f"isolated test failed: {relative}")
        runs[role] = {
            "command": command,
            "exit_code": completed.returncode,
            "summary": summary,
            "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
            "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
            "network_guard_installed": True,
            "cacheprovider_disabled": True,
        }
        _require_bound_bytecode_absent(bound)
    _require(set(runs) == auditor.V4_TEST_RUN_KEYS, "test-run role mismatch")
    _require(collect_code_state(workspace_root) == code_state, "code/test identity changed during isolated tests")
    evidence = {
        "schema_version": 1,
        "artifact_type": "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_V4_TEST_EVIDENCE",
        "status": "PASS_FROZEN_V4_AUDITOR_AND_SEALER_TESTS",
        "created_utc": utc_now_exact(),
        "bound_identities": bound,
        "source_compile": {"method": "compile_exact_source_no_pyc", "result": "PASS", "files": compiled},
        "pytest_isolation": "ONE_TEST_FILE_PER_CLEAN_NETWORK_DENIED_SUBPROCESS",
        "test_runs": runs,
        "required_test_names": list(auditor.V4_REQUIRED_TEST_NAMES),
        "required_test_names_present": True,
        "all_exit_codes_zero": True,
        "production_shape_regression": {
            "four_key_identity_passed": True,
            "transaction_reached": True,
            "raw_reached": True,
            "decoded_reached": True,
            "offline_replay_reached": True,
            "synthetic_fixture": True,
        },
        "real_seven_spawn_regression": {
            "start_method": "spawn",
            "entrypoint_module": "scripts.audit_noaa_gfs_multiseason_raw_postrun_v4",
            "worker_callable_module": "scripts.audit_noaa_gfs_multiseason_raw_postrun_v4",
            "max_workers": 7,
            "distinct_worker_pids": 7,
            "first_wave_synchronized": True,
            "network_requests": 0,
            "audit_files_written": 0,
            "labels_read": False,
            "arrays_2024_read": False,
            "arrays_2025_read": False,
        },
        "stdout_only_regression": True,
        "network_guard": {
            "parent_test_process_guarded": True,
            "sys_audit_hook_in_parent": True,
            "parent_socket_api_denied": True,
            "spawned_worker_network_route_static_absent": True,
            "spawned_worker_runtime_guard_installed": False,
            "external_packet_capture": False,
        },
        "network_requests": 0,
        "audit_files_written": 0,
        "labels_read": False,
        "arrays_2024_read": False,
        "arrays_2025_read": False,
        "models_fit": 0,
        "submission_csv_created": False,
    }
    _require(set(evidence) == auditor.V4_TEST_EVIDENCE_KEYS, "internal V4 test-evidence schema mismatch")
    return evidence


def build_authorization(
    root: Path,
    code_state: Mapping[str, Any],
    inputs: Mapping[str, Any],
    test_evidence: Mapping[str, Any],
    *,
    created_utc: str,
) -> dict[str, Any]:
    compact = created_utc.replace("-", "").replace(":", "").replace(".", "")
    audit_attempt_id = f"postrun_audit_v4__{compact}"
    bound = code_state["bound_identities"]
    payload = {
        "schema_version": 4,
        "artifact_type": "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDIT_AUTHORIZATION_V4",
        "status": "AUTHORIZED_PENDING_INDEPENDENT_REVIEW_AND_GO_V4",
        "created_utc": created_utc,
        "audit_attempt_id": audit_attempt_id,
        "incident": inputs["incident"],
        "v2_false_reject_incident": inputs["v2_false_reject_incident"],
        "v3_path_mispublish_state": inputs["v3_path_mispublish_state"],
        "superseded_v2_auditor": bound["v2_auditor"],
        "superseded_v2_auditor_test": bound["v2_auditor_test"],
        "superseded_v3_auditor": bound["superseded_v3_auditor"],
        "superseded_v3_auditor_test": bound["superseded_v3_auditor_test"],
        "superseded_v3_sealer": bound["superseded_v3_sealer"],
        "superseded_v3_sealer_test": bound["superseded_v3_sealer_test"],
        "v4_auditor": bound["v4_auditor"],
        "v4_auditor_test": bound["v4_auditor_test"],
        "v4_sealer": bound["v4_sealer"],
        "v4_sealer_test": bound["v4_sealer_test"],
        "v2_recovery_controls": inputs["v2_recovery_controls"],
        "v2_recovery_support": code_state["v2_recovery_support"],
        "recovery_attempt_id": auditor.V4_RECOVERY_ATTEMPT_ID,
        "recovery_transaction": inputs["recovery_transaction"],
        "recovery_history": inputs["recovery_history"],
        "recovery_progress": inputs["recovery_progress"],
        "canonical_outputs": inputs["canonical_outputs"],
        "original_v1_provenance": inputs["original_v1_provenance"],
        "preaudit_zero_mutation_snapshot": inputs["preaudit_zero_mutation_snapshot"],
        "runtime_identity_sha256": auditor.V4_RUNTIME_IDENTITY_SHA256,
        "test_evidence": dict(test_evidence),
        "required_command": auditor._v4_required_command(
            root, root / AUTH_RELATIVE, root / GO_RELATIVE
        ),
        "max_spawn_processes": 7,
        "network_requests_allowed": 0,
        "audit_files_written_allowed": 0,
        "labels_read_allowed": False,
        "arrays_2024_read_allowed": False,
        "arrays_2025_read_allowed": False,
        "models_fit_allowed": 0,
        "submission_csv_allowed": False,
        "full_offline_redecode_required": True,
        "stdout_only": True,
        "independent_review_required": True,
        "independent_go_required": True,
    }
    validate_authorization_payload(root, payload, code_state, inputs)
    return payload


def validate_authorization_payload(
    root: Path,
    payload: Mapping[str, Any],
    code_state: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> None:
    _require(set(payload) == auditor.V4_AUTH_KEYS, "V4 authorization schema mismatch")
    created = auditor._v4_created_utc(
        payload.get("created_utc"), label="V4 authorization sealer"
    )
    expected_attempt_id = "postrun_audit_v4__" + str(
        payload.get("created_utc")
    ).replace("-", "").replace(":", "").replace(".", "")
    _require(
        payload.get("schema_version") == 4
        and payload.get("artifact_type") == "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDIT_AUTHORIZATION_V4"
        and payload.get("status") == "AUTHORIZED_PENDING_INDEPENDENT_REVIEW_AND_GO_V4"
        and isinstance(payload.get("audit_attempt_id"), str)
        and re.fullmatch(r"postrun_audit_v4__[0-9]{8}T[0-9]{12}Z", str(payload["audit_attempt_id"])) is not None,
        "V4 authorization header mismatch",
    )
    _require(
        payload.get("audit_attempt_id") == expected_attempt_id,
        "V4 authorization created-time/attempt binding mismatch",
    )
    auditor._v4_validate_policy(payload, label="V4 authorization sealer")
    bound = code_state.get("bound_identities")
    _require(
        isinstance(bound, Mapping)
        and set(bound) == BOUND_SOURCE_ROLES,
        "V4 authorization bound-source role schema mismatch",
    )
    expected_source_bindings = {
        "superseded_v2_auditor": bound["v2_auditor"],
        "superseded_v2_auditor_test": bound["v2_auditor_test"],
        **{
            role: bound[role]
            for role in auditor.FROZEN_V3_SOURCE_IDENTITIES
        },
        "v4_auditor": bound["v4_auditor"],
        "v4_auditor_test": bound["v4_auditor_test"],
        "v4_sealer": bound["v4_sealer"],
        "v4_sealer_test": bound["v4_sealer_test"],
    }
    _require(
        all(payload.get(role) == expected for role, expected in expected_source_bindings.items()),
        "V4 authorization top-level source identity binding mismatch",
    )
    _require(
        payload.get("recovery_attempt_id") == auditor.V4_RECOVERY_ATTEMPT_ID,
        "V4 authorization recovery-attempt binding mismatch",
    )
    _require(
        payload.get("runtime_identity_sha256") == auditor.V4_RUNTIME_IDENTITY_SHA256,
        "V4 authorization runtime-identity binding mismatch",
    )
    _require(
        payload.get("required_command")
        == auditor._v4_required_command(
            root, root / AUTH_RELATIVE, root / GO_RELATIVE
        ),
        "V4 authorization required-command binding mismatch",
    )
    _require(
        payload.get("independent_review_required") is True
        and payload.get("independent_go_required") is True,
        "V4 authorization independent review/GO requirement mismatch",
    )
    _require(payload.get("incident") == inputs["incident"], "incident binding mismatch")
    _require(
        payload.get("v2_false_reject_incident")
        == inputs["v2_false_reject_incident"],
        "V2 false-reject incident binding mismatch",
    )
    _require(
        payload.get("v3_path_mispublish_state")
        == inputs["v3_path_mispublish_state"],
        "V3 path-mispublication state binding mismatch",
    )
    _require(payload.get("v2_recovery_controls") == inputs["v2_recovery_controls"], "control binding mismatch")
    _require(payload.get("v2_recovery_support") == code_state["v2_recovery_support"], "support binding mismatch")
    _require(payload.get("original_v1_provenance") == inputs["original_v1_provenance"], "V1 provenance binding mismatch")
    afterstate = auditor._v4_recovered_afterstate_from_authorization(payload)
    _require(afterstate == inputs["recovered_afterstate"], "recovered afterstate mismatch")
    auditor._v4_validate_zero_snapshot(
        payload.get("preaudit_zero_mutation_snapshot"),
        incident_postfailure_sha256=auditor.V4_INCIDENT_POSTFAILURE_STATE_CANONICAL_SHA256,
    )
    auditor._v4_validate_test_evidence(
        payload.get("test_evidence"),
        authorization_created=created,
        bound_identities=code_state["bound_identities"],
    )


def publish_json_no_overwrite(path: Path, payload: Mapping[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    digest = hashlib.sha256(data).hexdigest()
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{digest[:16]}")
    _require(path.parent.is_dir() and not _linklike(path.parent), "authorization parent invalid")
    _require(not _lexists(path) and not _lexists(temporary), "authorization destination/temp already exists")
    linked = False
    failure: Exception | None = None
    cleanup_failure: Exception | None = None
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
        linked = True
        _require(os.path.samefile(temporary, path), "authorization publication is not the staged hard link")
        _require(path.stat().st_size == len(data) and sha256_file(path) == digest, "published authorization bytes mismatch")
    except Exception as exc:
        failure = exc
    try:
        if temporary.is_file() and not temporary.is_symlink():
            temporary.unlink()
    except Exception as exc:
        cleanup_failure = exc
    expected_identity = {
        "path": str(path),
        "size_bytes": len(data),
        "sha256": digest,
    }
    if linked and (failure is not None or cleanup_failure is not None):
        cause = failure if failure is not None else cleanup_failure
        raise PostrunAuditPostPublicationError(
            "authorization was published but post-link verification/cleanup failed",
            expected_identity,
        ) from cause
    if failure is not None:
        if isinstance(failure, PostrunAuditSealError):
            raise failure
        raise PostrunAuditSealError(
            "authorization create-if-absent publication failed"
        ) from failure
    if cleanup_failure is not None:
        raise PostrunAuditSealError("authorization staging cleanup failed") from cleanup_failure


def static_audit(root: Path, workspace_root: Path = REPO) -> dict[str, Any]:
    root = Path(os.path.abspath(root))
    require_prepublication_control_state(root)
    code_state = collect_code_state(workspace_root)
    inputs = collect_root_inputs(root, code_state)
    return {
        "status": "PASS_V4_AUTHORIZATION_PRESEAL_STATIC_AUDIT",
        "artifact_type": "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDIT_V4_AUTH_SEALER_STATIC_AUDIT",
        "bound_identities": code_state["bound_identities"],
        "source_origin_trust_model": SOURCE_ORIGIN_TRUST_MODEL,
        "v2_recovery_support_role_count": len(code_state["v2_recovery_support"]),
        "recovered_afterstate_canonical_sha256": hashlib.sha256(
            canonical_bytes(inputs["recovered_afterstate"])
        ).hexdigest(),
        "authorization_files_published": 0,
        "independent_review_files_published": 0,
        "independent_go_files_published": 0,
        "network_requests": 0,
        "production_output_values_materialized": 0,
        "target_or_label_values_read": 0,
        "labels_read": False,
        "models_fit": 0,
        "submission_csv_created": False,
    }


def seal(root: Path, workspace_root: Path = REPO) -> dict[str, Any]:
    root = Path(os.path.abspath(root))
    require_canonical_module_entrypoint(root)
    install_entry_guard()
    require_prepublication_control_state(root)
    code_before = collect_code_state(workspace_root)
    inputs_before = collect_root_inputs(root, code_before)
    evidence = run_immutable_test_evidence(root, workspace_root, code_before)
    code_after = collect_code_state(workspace_root)
    inputs_after = collect_root_inputs(root, code_after)
    _require(code_after == code_before, "code/test identities changed during sealing")
    _require(inputs_after == inputs_before, "recovered afterstate changed during sealing")
    require_prepublication_control_state(root)

    created = utc_now_exact()
    authorization = build_authorization(
        root, code_before, inputs_before, evidence, created_utc=created
    )
    # Last fail-closed recheck before the sole publisher call.
    _require(collect_root_inputs(root, code_before) == inputs_before, "afterstate changed before publication")
    require_prepublication_control_state(root)
    # This is the final read before the sole mutating call.  It rehashes the
    # imported auditor plus all ten bound code/test files after every subprocess and
    # afterstate check.
    _require(collect_code_state(workspace_root) == code_before, "code changed before publication")
    destination = control_paths(root)["authorization"]
    try:
        publish_json_no_overwrite(destination, authorization)
    except PostrunAuditPostPublicationError as exc:
        relative_identity = dict(exc.authorization_identity)
        relative_identity["path"] = AUTH_RELATIVE
        raise PostrunAuditPostPublicationError(
            str(exc), relative_identity
        ) from exc
    serialized = (
        json.dumps(authorization, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )
    expected_published_identity = {
        "path": AUTH_RELATIVE,
        "size_bytes": len(serialized),
        "sha256": hashlib.sha256(serialized).hexdigest(),
    }
    published_identity = expected_published_identity
    try:
        observed_identity = strict_root_identity(
            root, AUTH_RELATIVE, label="published V4 authorization"
        )
        _require(
            observed_identity == expected_published_identity,
            "published authorization identity changed after hard-link verification",
        )
        published_identity = observed_identity
        require_postpublication_control_state(root)
        _require(load_json_control(destination, label="published V4 authorization") == authorization, "published authorization payload mismatch")
        _require(
            collect_code_state(workspace_root) == code_before,
            "code changed after publication",
        )
        _require(
            collect_root_inputs(root, code_before) == inputs_before,
            "recovered afterstate changed after publication",
        )
    except Exception as exc:
        raise PostrunAuditPostPublicationError(
            "authorization remains published after a postpublication check failed: "
            f"{exc}",
            published_identity,
        ) from exc
    return {
        "status": "PASS_V4_AUTHORIZATION_SEALED_PENDING_INDEPENDENT_REVIEW_AND_GO",
        "artifact_type": "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDIT_V4_AUTH_SEAL_REPORT",
        "authorization": published_identity,
        "source_origin_trust_model": SOURCE_ORIGIN_TRUST_MODEL,
        "authorization_files_published": 1,
        "independent_review_files_published": 0,
        "independent_go_files_published": 0,
        "postrun_audit_executed": False,
        "network_requests": 0,
        "production_output_values_materialized": 0,
        "target_or_label_values_read": 0,
        "labels_read": False,
        "arrays_2024_read": False,
        "arrays_2025_read": False,
        "models_fit": 0,
        "submission_csv_created": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--static-audit", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = (
            static_audit(args.root.resolve(), REPO)
            if args.static_audit
            else seal(args.root.resolve(), REPO)
        )
    except PostrunAuditPostPublicationError as exc:
        report = {
            "status": "FAIL_V4_POSTRUN_AUDIT_AUTHORIZATION_POSTPUBLICATION",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "authorization": exc.authorization_identity,
            "authorization_files_published": 1,
            "authorization_rollback_attempted": False,
            "independent_review_files_published": 0,
            "independent_go_files_published": 0,
            "postrun_audit_executed": False,
            "source_origin_trust_model": SOURCE_ORIGIN_TRUST_MODEL,
            "network_requests": 0,
            "production_output_values_materialized": 0,
            "target_or_label_values_read": 0,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 3
    except Exception as exc:
        report = {
            "status": "FAIL_V4_POSTRUN_AUDIT_AUTHORIZATION_SEAL",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "authorization_files_published": 0,
            "independent_review_files_published": 0,
            "independent_go_files_published": 0,
            "postrun_audit_executed": False,
            "source_origin_trust_model": SOURCE_ORIGIN_TRUST_MODEL,
            "network_requests": 0,
            "production_output_values_materialized": 0,
            "target_or_label_values_read": 0,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

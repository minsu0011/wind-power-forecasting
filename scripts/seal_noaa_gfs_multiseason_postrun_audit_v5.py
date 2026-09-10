#!/usr/bin/env python
"""Seal the append-only V5 postrun-audit authorization.

This publisher is deliberately narrower than the auditor it authorizes.  It
hashes immutable controls and eight bound code identities, validates recovered
files by metadata only, compiles the four executing V5 sources, runs four test
files in isolated network-denied subprocesses, and publishes exactly one
create-if-absent JSON file.  It never
embeds the full recovered inventory, publishes the independent review or GO,
executes the production postrun audit, materializes a production output/target
value, or opens a network client.
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


AUDITOR_SOURCE_PATH = REPO / "scripts" / "audit_noaa_gfs_multiseason_raw_postrun_v5.py"


def _require_preimport_cache_runtime() -> None:
    if not (
        os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
        and sys.dont_write_bytecode is True
        and "PYTHONPYCACHEPREFIX" not in os.environ
        and sys.pycache_prefix is None
    ):
        raise RuntimeError(
            "V5 sealer import requires -B, PYTHONDONTWRITEBYTECODE=1, "
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
            "V5 auditor matching bytecode/cache competitor exists: "
            + ", ".join(str(candidate) for candidate in matches)
        )


def _require_bound_bytecode_absent(
    bound: Mapping[str, Mapping[str, Any]],
) -> None:
    required = {
        "superseded_v4_auditor", "superseded_v4_auditor_test",
        "superseded_v4_sealer", "superseded_v4_sealer_test",
        "v5_auditor", "v5_auditor_test", "v5_sealer", "v5_sealer_test",
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
            "V5 auditor same-stem source-origin competitor exists: "
            + ", ".join(str(candidate) for candidate in competitors)
        )


def _preimport_auditor_source_guard(path: Path) -> None:
    """Reject top-level I/O/network routes before importing the moving V5 source."""

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
        raise RuntimeError("V5 auditor has a forbidden import-time route")
    allowed_top_level = {
        ast.Expr, ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign,
        ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.If,
    }
    if any(type(node) not in allowed_top_level for node in tree.body):
        raise RuntimeError("V5 auditor has an unexpected top-level statement")
    for node in tree.body:
        if isinstance(node, ast.Expr) and not isinstance(node.value, ast.Constant):
            raise RuntimeError("V5 auditor has a top-level expression side effect")
        if isinstance(node, ast.If):
            if (
                not isinstance(node.test, ast.Compare)
                or ast.unparse(node.test) != "__name__ == '__main__'"
                or node.orelse
            ):
                raise RuntimeError("V5 auditor has an unexpected top-level branch")
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            for call in (item for item in ast.walk(node) if isinstance(item, ast.Call)):
                function = ast.unparse(call.func)
                allowed = (
                    function in {"Path", "tuple", "str", "AuditContract", "GroupContract", "re.compile"}
                    or function.endswith((".resolve", ".replace", ".strip"))
                )
                if not allowed:
                    raise RuntimeError(
                        f"V5 auditor has an unapproved import-time call: {function}"
                    )


_require_preimport_cache_runtime()
_require_auditor_bytecode_absent(Path(__file__).resolve())
_require_auditor_bytecode_absent(AUDITOR_SOURCE_PATH)
_require_auditor_source_competitors_absent(AUDITOR_SOURCE_PATH)
_preimport_auditor_source_guard(AUDITOR_SOURCE_PATH)

from scripts import audit_noaa_gfs_multiseason_raw_postrun_v5 as auditor

if not (
    auditor.__name__ == "scripts.audit_noaa_gfs_multiseason_raw_postrun_v5"
    and auditor.__package__ == "scripts"
    and auditor.__spec__ is not None
    and auditor.__spec__.name == auditor.__name__
    and Path(auditor.__file__).resolve() == AUDITOR_SOURCE_PATH.resolve()
):
    raise RuntimeError("V5 auditor import source origin mismatch")


ROOT_DEFAULT = auditor.ROOT_DEFAULT
CANONICAL_MODULE_ENTRYPOINT = (
    "scripts.seal_noaa_gfs_multiseason_postrun_audit_v5"
)
AUTH_RELATIVE = auditor.V5_AUTH_RELATIVE
REVIEW_RELATIVE = auditor.V5_REVIEW_RELATIVE
GO_RELATIVE = auditor.V5_GO_RELATIVE
SEALER_TEST = REPO / "tests" / "test_seal_noaa_gfs_multiseason_postrun_audit_v5.py"
EXPECTED_V5_AUDITOR_IDENTITY = {
    "path": str(AUDITOR_SOURCE_PATH.resolve()),
    "size_bytes": 481_173,
    "sha256": "81e18256bf53bcdc3ba27a37787e1b2240bf0ea286d7f8c758d9f7d5e96ce6b1",
}
EXPECTED_V5_AUDITOR_TEST_IDENTITY = {
    "path": str(auditor.AUDITOR_TEST.resolve()),
    "size_bytes": 228_289,
    "sha256": "45ce935aeb945a06d4eaafb0885320348d27e12d8343387f53a8cac108182fc1",
}

TEST_RELATIVES = {
    "v5_auditor_tests": "tests/test_noaa_gfs_multiseason_raw_postrun_v5.py",
    "v5_sealer_tests": "tests/test_seal_noaa_gfs_multiseason_postrun_audit_v5.py",
    "frozen_v4_auditor_tests": "tests/test_noaa_gfs_multiseason_raw_postrun_v4.py",
    "frozen_v4_sealer_tests": "tests/test_seal_noaa_gfs_multiseason_postrun_audit_v4.py",
}
BOUND_SOURCE_PATHS = {
    "superseded_v4_auditor": auditor.FROZEN_V4_AUDITOR.resolve(),
    "superseded_v4_auditor_test": auditor.FROZEN_V4_AUDITOR_TEST.resolve(),
    "superseded_v4_sealer": auditor.FROZEN_V4_SEALER.resolve(),
    "superseded_v4_sealer_test": auditor.FROZEN_V4_SEALER_TEST.resolve(),
    "v5_auditor": Path(auditor.__file__).resolve(),
    "v5_auditor_test": auditor.AUDITOR_TEST.resolve(),
    "v5_sealer": Path(__file__).resolve(),
    "v5_sealer_test": SEALER_TEST.resolve(),
}
POSTRUN_REPORT_CANDIDATES = auditor.V5_POSTRUN_REPORT_CANDIDATES
CONTROL_TEMP_RELATIVES = (
    auditor.FROZEN_V3_AUTH_RELATIVE,
    auditor.FROZEN_V3_REVIEW_RELATIVE,
    auditor.FROZEN_V3_CANONICAL_GO_RELATIVE,
    auditor.FROZEN_V3_MISPLACED_GO_RELATIVE,
    auditor.V4_AUTH_RELATIVE,
    auditor.V4_REVIEW_RELATIVE,
    auditor.V4_GO_RELATIVE,
    AUTH_RELATIVE, REVIEW_RELATIVE, GO_RELATIVE,
    auditor.V2_FALSE_REJECT_INCIDENT_RELATIVE,
    auditor.V4_PATH_MISPUBLISH_INCIDENT_RELATIVE,
    auditor.V5_TRANSPORT_INCIDENT_RELATIVE,
    auditor.V5_TRANSPORT_CORRECTION_RELATIVE,
)
BOUND_SOURCE_ROLES = {
    "superseded_v4_auditor", "superseded_v4_auditor_test",
    "superseded_v4_sealer", "superseded_v4_sealer_test",
    "v5_auditor", "v5_auditor_test", "v5_sealer", "v5_sealer_test",
}
BOUND_IDENTITY_ROLES = {
    "v4_authorization",
    "v4_review_transport_truncation_incident",
    "v4_review_transport_truncation_incident_correction",
    *BOUND_SOURCE_ROLES,
}
SOURCE_ORIGIN_TRUST_MODEL = {
    "model": "NON_ADVERSARIAL_NO_CONCURRENT_FILESYSTEM_SWAP",
    "preimport_source_ast_and_competitor_scan": True,
    "pretest_posttest_and_prepublish_eight_source_rehash_and_pyc_scan": True,
    "test_evidence_exact_eleven_bound_identities": True,
    "source_compile_exact_executing_v5_four": True,
    "adversarial_concurrent_swap_resistance_claimed": False,
}


class PostrunAuditSealError(RuntimeError):
    """The immutable V5 authorization could not be sealed safely."""


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

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise PostrunAuditSealError(f"{label} has a duplicate JSON key")
            value[key] = item
        return value

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except PostrunAuditSealError:
        raise
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
        "seal_noaa_gfs_multiseason_postrun_audit_v5` entrypoint",
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
    raise _NetworkDenied("network access is forbidden during V5 authorization sealing")


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
        "authorization": root_path(root, AUTH_RELATIVE, label="V5 authorization"),
        "independent_review": root_path(root, REVIEW_RELATIVE, label="V5 review"),
        "independent_go": root_path(root, GO_RELATIVE, label="V5 GO"),
    }


def required_audit_command(root: Path) -> list[str]:
    return [
        str(Path(sys.executable).resolve()),
        "-B",
        "-m",
        "scripts.audit_noaa_gfs_multiseason_raw_postrun_v5",
        "--root",
        str(root.resolve()),
        "--authorization",
        str((root / AUTH_RELATIVE).resolve()),
        "--independent-go",
        str((root / GO_RELATIVE).resolve()),
    ]


def control_temporary_files(root: Path) -> list[str]:
    matches: list[str] = []
    for relative in CONTROL_TEMP_RELATIVES:
        path = root_path(root, relative, label=f"V5 control temp destination {relative}")
        parent = path.parent
        _require(parent.is_dir() and not _linklike(parent), "V5 control parent invalid")
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
        Path(auditor.V4_AUTH_RELATIVE).name,
    }
    if authorization_present:
        prereg.add(Path(AUTH_RELATIVE).name)
    return {
        "prereg": prereg,
        "independent_redteam": {
            Path(auditor.FROZEN_V3_REVIEW_RELATIVE).name,
            Path(auditor.V4_REVIEW_RELATIVE).name,
        },
        "incidents": {
            Path(auditor.V2_FALSE_REJECT_INCIDENT_RELATIVE).name,
            Path(auditor.V4_PATH_MISPUBLISH_INCIDENT_RELATIVE).name,
            Path(auditor.V5_TRANSPORT_INCIDENT_RELATIVE).name,
            Path(auditor.V5_TRANSPORT_CORRECTION_RELATIVE).name,
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
    for relative, label in (
        (auditor.FROZEN_V3_CANONICAL_GO_RELATIVE, "canonical V3 GO"),
        (auditor.FROZEN_V4_CANONICAL_GO_RELATIVE, "canonical V4 GO"),
    ):
        _require(
            not _lexists(root_path(root, relative, label=label)),
            f"{label} must remain absent permanently",
        )
    result: dict[str, list[str]] = {}
    for relative, prefix in prefixes.items():
        parent = root_path(root, relative, label=f"V5 namespace {relative}")
        _require(
            _lexists(parent) and parent.is_dir() and not _linklike(parent),
            f"V5 control namespace parent invalid: {relative}",
        )
        matched: list[str] = []
        for entry in parent.iterdir():
            if not entry.name.casefold().lstrip(".").startswith(prefix):
                continue
            _require(
                _lexists(entry) and entry.is_file() and not _linklike(entry),
                f"V5 control namespace contains non-regular/link entry: {entry}",
            )
            matched.append(entry.name)
        _require(
            set(matched) == expected[relative]
            and len({name.casefold() for name in matched}) == len(matched),
            f"V5 control namespace inventory mismatch: {relative}",
        )
        result[relative] = sorted(matched, key=str.casefold)
    return result


def require_prepublication_control_state(root: Path) -> dict[str, Any]:
    paths = control_paths(root)
    _require(
        all(not _lexists(path) for path in paths.values()),
        "V5 authorization/review/GO already exists",
    )
    _require(control_temporary_files(root) == [], "V5 control temporary sibling exists")
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
        "published V5 authorization is absent, non-regular or link-like",
    )
    _require(
        not _lexists(paths["independent_review"])
        and not _lexists(paths["independent_go"]),
        "V5 sealer observed an independent review/GO",
    )
    _require(control_temporary_files(root) == [], "V5 control temporary sibling exists")
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
    _require(Path(auditor.V5_SEALER).resolve() == Path(__file__).resolve(), "auditor V5 sealer path mismatch")
    _require(Path(auditor.V5_SEALER_TEST).resolve() == SEALER_TEST.resolve(), "auditor V5 sealer-test path mismatch")

    bound = {
        role: strict_external_identity(path, label=role)
        for role, path in BOUND_SOURCE_PATHS.items()
    }
    _require(set(bound) == BOUND_SOURCE_ROLES, "bound source role schema mismatch")
    for role, expected in auditor.FROZEN_V4_SOURCE_IDENTITIES.items():
        _require(bound[role] == expected, f"frozen V4 source identity mismatch: {role}")
    _require(
        bound["v5_auditor"] == EXPECTED_V5_AUDITOR_IDENTITY,
        "frozen executing V5 auditor identity mismatch",
    )
    _require(
        bound["v5_auditor_test"] == EXPECTED_V5_AUDITOR_TEST_IDENTITY,
        "frozen executing V5 auditor-test identity mismatch",
    )
    _require_bound_bytecode_absent(bound)
    return {"bound_identities": bound}


def _verify_metadata_record(
    root: Path, record: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    """Verify path/type/size only; recovered artifact bytes are never opened."""

    _require(
        isinstance(record, Mapping)
        and set(record) == {"path", "size_bytes", "sha256"},
        f"{label} contract schema mismatch",
    )
    path = root_path(root, str(record["path"]), label=label)
    _require(path.is_file() and not _linklike(path), f"{label} is not an exact file")
    _require(
        path.stat().st_size == record["size_bytes"],
        f"{label} metadata size mismatch",
    )
    return {"path": str(record["path"]), "size_bytes": path.stat().st_size}


def _canonical_length_sha256(value: Any) -> tuple[int, str]:
    encoded = canonical_bytes(value)
    return len(encoded), hashlib.sha256(encoded).hexdigest()


def validate_recovered_filesystem_closure(
    root: Path,
    afterstate: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    """Reclose recovered path/type/size metadata without reading artifact bytes."""

    _require(
        afterstate.get("canonical_output_count") == 8
        and afterstate.get("recovery_history_file_count") == 3
        and afterstate.get("raw_active_lock_present") is False
        and afterstate.get("decoded_active_lock_present") is False,
        "recovered-afterstate closure header mismatch",
    )

    progress = afterstate.get("recovery_progress")
    _require(
        isinstance(progress, Mapping)
        and progress.get("file_count") == 104
        and isinstance(progress.get("inventory"), list)
        and len(progress["inventory"]) == 104,
        "recovery-progress commitment source mismatch",
    )
    progress_dir = root_path(
        root, "decoded/recovery_progress", label="recovery progress"
    )
    _require(
        progress_dir.is_dir() and not _linklike(progress_dir),
        "recovery-progress directory invalid",
    )
    expected_progress = {
        str(record["path"]): record for record in progress["inventory"]
    }
    progress_entries = list(progress_dir.iterdir())
    _require(
        len(expected_progress) == 104
        and len(progress_entries) == 104
        and {path.relative_to(root).as_posix() for path in progress_entries}
        == set(expected_progress),
        "recovery-progress metadata inventory mismatch",
    )
    for relative, record in expected_progress.items():
        _verify_metadata_record(root, record, label=f"recovery progress {relative}")

    history_dir = root_path(root, "decoded/recovery_history", label="recovery history")
    _require(
        history_dir.is_dir() and not _linklike(history_dir),
        "recovery-history directory invalid",
    )
    expected_history_paths = {
        str(record["path"])
        for record in afterstate["completion_locks"].values()
    } | {str(afterstate["postcommit_input_audit"]["path"])}
    history_entries = list(history_dir.iterdir())
    _require(
        len(history_entries) == 3
        and all(path.is_file() and not _linklike(path) for path in history_entries)
        and {path.relative_to(root).as_posix() for path in history_entries}
        == expected_history_paths,
        "recovery-history exact filesystem closure mismatch",
    )
    for role, record in afterstate["completion_locks"].items():
        _verify_metadata_record(root, record, label=f"recovery history {role}")
    _verify_metadata_record(
        root,
        afterstate["postcommit_input_audit"],
        label="recovery postcommit input audit",
    )

    outputs = afterstate.get("canonical_outputs")
    _require(
        isinstance(outputs, Mapping) and len(outputs) == 8,
        "canonical-output commitment source mismatch",
    )
    for role, record in outputs.items():
        _verify_metadata_record(root, record, label=f"canonical output {role}")

    transaction_root = root_path(
        root, "raw/output_transactions", label="output transaction root"
    )
    _require(
        transaction_root.is_dir() and not _linklike(transaction_root),
        "output-transaction root invalid",
    )
    recovery_authorization_record = authorization["v2_recovery_controls"][
        "recovery_authorization_v2"
    ]
    _require(
        recovery_authorization_record
        == auditor.V4_V2_CONTROL_IDENTITIES["recovery_authorization_v2"]
        and strict_root_identity(
            root,
            str(recovery_authorization_record["path"]),
            label="V2 recovery authorization",
        )
        == recovery_authorization_record,
        "V2 recovery authorization identity mismatch",
    )
    recovery_authorization = load_json_control(
        root / str(recovery_authorization_record["path"]),
        label="V2 recovery authorization",
    )
    remnants = recovery_authorization.get("documented_preplan_remnants")
    _require(
        isinstance(remnants, list)
        and len(remnants) == 2
        and all(
            isinstance(record, Mapping)
            and set(record) == {"path", "size_bytes", "sha256"}
            for record in remnants
        ),
        "incident-bound preplan-remnant declaration mismatch",
    )
    expected_transaction_files = {
        str(afterstate["transaction"]["plan"]["path"]),
        str(afterstate["transaction"]["commit"]["path"]),
        *(str(record["path"]) for record in remnants),
    }
    failed_attempt = "20260810T151057590738Z__pid3536__5f3e0fb20132"
    expected_transaction_dirs = {
        f"raw/output_transactions/{failed_attempt}",
        f"raw/output_transactions/{failed_attempt}/staged",
        f"raw/output_transactions/{failed_attempt}/staged/raw",
        f"raw/output_transactions/{auditor.V4_RECOVERY_ATTEMPT_ID}",
        f"raw/output_transactions/{auditor.V4_RECOVERY_ATTEMPT_ID}/staged",
        f"raw/output_transactions/{auditor.V4_RECOVERY_ATTEMPT_ID}/staged/decoded",
        f"raw/output_transactions/{auditor.V4_RECOVERY_ATTEMPT_ID}/staged/raw",
    }
    transaction_files: list[Path] = []
    transaction_dirs: list[Path] = []
    for path in transaction_root.rglob("*"):
        _require(
            not _linklike(path),
            "output-transaction tree contains a link/junction",
        )
        if path.is_file():
            transaction_files.append(path)
        elif path.is_dir():
            transaction_dirs.append(path)
        else:
            raise PostrunAuditSealError(
                "output-transaction tree contains a special object"
            )
    _require(
        len(transaction_files) == afterstate["transaction"]["total_recursive_files"]
        == 4
        and {path.relative_to(root).as_posix() for path in transaction_files}
        == expected_transaction_files
        and {path.relative_to(root).as_posix() for path in transaction_dirs}
        == expected_transaction_dirs,
        "output-transaction recursive metadata closure mismatch",
    )
    for role in ("plan", "commit"):
        _verify_metadata_record(
            root,
            afterstate["transaction"][role],
            label=f"recovery transaction {role}",
        )
    for index, record in enumerate(remnants):
        _verify_metadata_record(
            root, record, label=f"incident-bound preplan remnant {index}"
        )

    _require(
        not _lexists(root / "raw/RAW_LAUNCH_ACTIVE.lock")
        and not _lexists(root / "decoded/DECODE_RECOVERY_ACTIVE.lock"),
        "active producer/recovery lock exists",
    )
    return {
        "mode": "PATH_TYPE_SIZE_METADATA_ONLY_NO_RECOVERED_BYTES_READ",
        "recovery_progress_file_count": len(progress_entries),
        "recovery_history_file_count": len(history_entries),
        "transaction_recursive_file_count": len(transaction_files),
        "transaction_recursive_directory_count": len(transaction_dirs),
        "canonical_output_count": len(outputs),
        "recovered_artifact_files_opened": 0,
    }


def build_zero_snapshot(root: Path) -> dict[str, Any]:
    active = {
        "raw": {"path": "raw/RAW_LAUNCH_ACTIVE.lock", "present": _lexists(root / "raw/RAW_LAUNCH_ACTIVE.lock")},
        "decoded": {"path": "decoded/DECODE_RECOVERY_ACTIVE.lock", "present": _lexists(root / "decoded/DECODE_RECOVERY_ACTIVE.lock")},
    }
    temporary = control_temporary_files(root)
    reports = [relative for relative in POSTRUN_REPORT_CANDIDATES if _lexists(root / relative)]
    _require(not any(item["present"] for item in active.values()), "active producer/recovery lock exists")
    _require(temporary == [], "V5 control temporary sibling exists")
    _require(reports == [], "postrun audit output file exists despite stdout-only contract")
    snapshot = {
        "incident_postfailure_state_canonical_sha256": auditor.V5_INCIDENT_POSTFAILURE_STATE_CANONICAL_SHA256,
        "recovered_afterstate_commitment_canonical_sha256": (
            auditor.V5_RECOVERED_AFTERSTATE_COMMITMENT_CANONICAL_SHA256
        ),
        "active_locks": active,
        "v5_control_temporary_files": {
            "checked_destination_relative_paths": list(CONTROL_TEMP_RELATIVES),
            "matching_temporary_files": list(temporary),
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
    _require(
        set(snapshot) == auditor.V5_ZERO_SNAPSHOT_KEYS,
        "V5 zero-snapshot schema mismatch",
    )
    _require(
        snapshot["v5_control_temporary_files"]
        == {
            "checked_destination_relative_paths": list(CONTROL_TEMP_RELATIVES),
            "matching_temporary_files": [],
        },
        "V5 zero-snapshot temporary-destination declaration mismatch",
    )
    try:
        auditor._v5_validate_zero_snapshot(root, snapshot)
    except Exception as exc:
        raise PostrunAuditSealError(f"V5 zero-snapshot invalid: {exc}") from exc
    return snapshot


def _expected_incident_identities() -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        {
            "path": auditor.V5_TRANSPORT_INCIDENT_RELATIVE,
            "size_bytes": auditor.V5_TRANSPORT_INCIDENT_SIZE_BYTES,
            "sha256": auditor.V5_TRANSPORT_INCIDENT_SHA256,
        },
        {
            "path": auditor.V5_TRANSPORT_CORRECTION_RELATIVE,
            "size_bytes": auditor.V5_TRANSPORT_CORRECTION_SIZE_BYTES,
            "sha256": auditor.V5_TRANSPORT_CORRECTION_SHA256,
        },
    )


def _load_v4_authorization_base(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    declared_base = {
        "identity": dict(auditor.FROZEN_V4_AUTH_IDENTITY),
        "schema_version": 4,
        "status": "AUTHORIZED_PENDING_INDEPENDENT_REVIEW_AND_GO_V4",
        "audit_attempt_id": auditor.FROZEN_V4_AUDIT_ATTEMPT_ID,
        "recovery_attempt_id": auditor.V4_RECOVERY_ATTEMPT_ID,
        "payload_canonical_sha256": auditor.FROZEN_V4_AUTH_PAYLOAD_CANONICAL_SHA256,
        "test_evidence_canonical_sha256": (
            auditor.FROZEN_V4_TEST_EVIDENCE_CANONICAL_SHA256
        ),
        "recovered_afterstate_canonical_sha256": (
            auditor.V4_RECOVERED_AFTERSTATE_CANONICAL_SHA256
        ),
    }
    _require(
        set(declared_base) == auditor.V5_AUTHORIZATION_BASE_KEYS,
        "V5 authorization-base schema mismatch",
    )
    try:
        validated = auditor._v5_load_and_validate_v4_authorization_base(
            root, declared_base
        )
    except Exception as exc:
        raise PostrunAuditSealError(
            f"frozen V4 authorization base invalid: {exc}"
        ) from exc
    _require(
        validated.get("base") == declared_base
        and validated.get("identity") == declared_base["identity"]
        and validated.get("commitment")
        == auditor.V5_RECOVERED_AFTERSTATE_COMMITMENT
        and isinstance(validated.get("authorization"), Mapping)
        and isinstance(validated.get("afterstate"), Mapping),
        "frozen V4 authorization-base validator result mismatch",
    )
    return (
        declared_base,
        dict(validated["authorization"]),
        dict(validated["afterstate"]),
    )


def _legacy_local_v4_transport_truncation_state(
    root: Path,
    authorization_identity: Mapping[str, Any],
    expected_afterstate: Mapping[str, Any],
) -> dict[str, Any]:
    review_identity = strict_root_identity(
        root, auditor.V4_REVIEW_RELATIVE, label="rejected V4 independent review"
    )
    _require(
        review_identity == auditor.FROZEN_V4_REVIEW_IDENTITY,
        "rejected V4 review identity mismatch",
    )
    _require(
        not _lexists(root / auditor.FROZEN_V4_CANONICAL_GO_RELATIVE),
        "canonical V4 GO must remain absent permanently",
    )
    review = load_json_control(
        root / auditor.V4_REVIEW_RELATIVE,
        label="rejected V4 independent review",
    )
    review_length, review_sha = _canonical_length_sha256(review)
    actual_afterstate = review.get("recovered_afterstate")
    actual_afterstate_length, actual_afterstate_sha = _canonical_length_sha256(
        actual_afterstate
    )
    actual_progress = (
        actual_afterstate.get("recovery_progress")
        if isinstance(actual_afterstate, Mapping)
        else None
    )
    actual_inventory = (
        actual_progress.get("inventory")
        if isinstance(actual_progress, Mapping)
        else None
    )
    expected_inventory = expected_afterstate["recovery_progress"]["inventory"]
    _require(
        set(review) == auditor.V4_REVIEW_KEYS
        and review.get("schema_version") == 4
        and review.get("artifact_type")
        == "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_REVIEW_V4"
        and review.get("status") == "PASS_PENDING_INDEPENDENT_GO_V4"
        and review.get("verdict") == "GO_RECOMMENDED"
        and review.get("audit_attempt_id") == auditor.FROZEN_V4_AUDIT_ATTEMPT_ID
        and review.get("authorization") == dict(authorization_identity)
        and review_length == auditor.FROZEN_V4_ACTUAL_REVIEW_PAYLOAD_CANONICAL_BYTES
        and review_sha == auditor.FROZEN_V4_ACTUAL_REVIEW_PAYLOAD_CANONICAL_SHA256
        and actual_afterstate_length
        == auditor.FROZEN_V4_ACTUAL_AFTERSTATE_CANONICAL_BYTES
        and actual_afterstate_sha
        == auditor.FROZEN_V4_ACTUAL_AFTERSTATE_CANONICAL_SHA256
        and isinstance(actual_inventory, list)
        and len(actual_inventory) == 101
        and len(expected_inventory) == 104,
        "rejected V4 review transport-truncation evidence mismatch",
    )
    actual_inventory_length, actual_inventory_sha = _canonical_length_sha256(
        actual_inventory
    )
    _require(
        actual_inventory_length
        == auditor.FROZEN_V4_ACTUAL_PROGRESS_INVENTORY_CANONICAL_BYTES
        and actual_inventory_sha
        == auditor.FROZEN_V4_ACTUAL_PROGRESS_INVENTORY_CANONICAL_SHA256,
        "rejected V4 review progress-inventory digest mismatch",
    )
    reconstructed = dict(review)
    reconstructed["recovered_afterstate"] = dict(expected_afterstate)
    intended_length, intended_sha = _canonical_length_sha256(reconstructed)
    pretty = (
        json.dumps(reconstructed, ensure_ascii=False, indent=2, sort_keys=True)
        .encode("utf-8")
        + b"\n"
    )
    _require(
        intended_length == auditor.FROZEN_V4_INTENDED_REVIEW_PAYLOAD_CANONICAL_BYTES
        and intended_sha
        == auditor.FROZEN_V4_INTENDED_REVIEW_PAYLOAD_CANONICAL_SHA256
        and len(pretty)
        == auditor.FROZEN_V4_INTENDED_REVIEW_PRETTY_IDENTITY["size_bytes"]
        and hashlib.sha256(pretty).hexdigest()
        == auditor.FROZEN_V4_INTENDED_REVIEW_PRETTY_IDENTITY["sha256"],
        "intended V4 review reconstruction mismatch",
    )
    state = {
        "authorization": dict(authorization_identity),
        "independent_review": review_identity,
        "canonical_go": {
            "path": auditor.FROZEN_V4_CANONICAL_GO_RELATIVE,
            "present": False,
            "must_remain_absent": True,
        },
        "actual_review_payload_canonical_sha256": review_sha,
        "intended_review_payload_canonical_sha256": intended_sha,
        "actual_recovered_afterstate_canonical_sha256": actual_afterstate_sha,
        "expected_recovered_afterstate_canonical_sha256": (
            auditor.V4_RECOVERED_AFTERSTATE_CANONICAL_SHA256
        ),
        "actual_progress_inventory_count": len(actual_inventory),
        "expected_progress_inventory_count": len(expected_inventory),
        "intended_review_pretty_serialization": dict(
            auditor.FROZEN_V4_INTENDED_REVIEW_PRETTY_IDENTITY
        ),
        "v4_auditor_execution_started": False,
        "v4_auditor_execution_authorized": False,
    }
    _require(
        set(state) == auditor.V5_TRUNCATION_STATE_KEYS,
        "V5 transport-truncation state schema mismatch",
    )
    return state


def _load_v4_transport_truncation_state(
    root: Path,
    base: Mapping[str, Any],
    authorization: Mapping[str, Any],
    afterstate: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    try:
        authorization_created = auditor._v4_created_utc(
            authorization.get("created_utc"), label="frozen V4 authorization"
        )
        validated = auditor._v5_load_rejected_v4_review(
            root,
            {
                "authorization": authorization,
                "authorization_created": authorization_created,
                "identity": base["identity"],
                "base": base,
                "afterstate": afterstate,
            },
        )
    except Exception as exc:
        raise PostrunAuditSealError(
            f"rejected V4 transport-truncated review invalid: {exc}"
        ) from exc
    state = validated.get("state")
    _require(
        isinstance(state, Mapping)
        and set(state) == auditor.V5_TRUNCATION_STATE_KEYS
        and state.get("authorization") == base["identity"]
        and state.get("independent_review") == auditor.FROZEN_V4_REVIEW_IDENTITY
        and state.get("canonical_go")
        == {"path": auditor.FROZEN_V4_CANONICAL_GO_RELATIVE, "present": False}
        and state.get("v4_auditor_execution_started") is False
        and state.get("v4_auditor_execution_authorized") is False
        and validated.get("duplicate_count") == 1
        and validated.get("transport_marker_count") == 1,
        "rejected V4 review validation result mismatch",
    )
    try:
        auditor._v5_validate_transport_truncation_state(state, validated["state"])
    except Exception as exc:
        raise PostrunAuditSealError(
            f"V5 transport-truncation state invalid: {exc}"
        ) from exc
    review = validated.get("review")
    _require(
        isinstance(review, Mapping) and isinstance(review.get("created_utc"), str),
        "rejected V4 review chronology source mismatch",
    )
    return dict(state), str(review["created_utc"])


def _recovered_afterstate_commitment(
    afterstate: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        auditor_commitment = auditor._v5_recovered_afterstate_commitment(afterstate)
    except Exception as exc:
        raise PostrunAuditSealError(
            f"V5 recovered-afterstate commitment invalid: {exc}"
        ) from exc
    afterstate_length, afterstate_sha = _canonical_length_sha256(afterstate)
    progress = afterstate["recovery_progress"]
    commitment = {
        "canonical_sha256": afterstate_sha,
        "canonical_size_bytes": afterstate_length,
        "recovery_progress_file_count": progress["file_count"],
        "recovery_progress_inventory_count": len(progress["inventory"]),
        "recovery_progress_inventory_canonical_sha256": progress[
            "inventory_canonical_sha256"
        ],
        "canonical_output_count": afterstate["canonical_output_count"],
        "transaction_total_recursive_files": afterstate["transaction"][
            "total_recursive_files"
        ],
        "recovery_history_file_count": afterstate["recovery_history_file_count"],
        "raw_active_lock_present": afterstate["raw_active_lock_present"],
        "decoded_active_lock_present": afterstate["decoded_active_lock_present"],
    }
    length, digest = _canonical_length_sha256(commitment)
    _require(
        commitment == auditor_commitment
        == auditor.V5_RECOVERED_AFTERSTATE_COMMITMENT
        and length == auditor.V5_RECOVERED_AFTERSTATE_COMMITMENT_CANONICAL_BYTES
        and digest
        == auditor.V5_RECOVERED_AFTERSTATE_COMMITMENT_CANONICAL_SHA256,
        "V5 compact recovered-afterstate commitment mismatch",
    )
    return commitment


def collect_root_inputs(root: Path, code_state: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(os.path.abspath(root))
    _require(root.is_dir() and not _linklike(root), "artifact root invalid")
    _require(
        isinstance(code_state.get("bound_identities"), Mapping)
        and set(code_state["bound_identities"]) == BOUND_SOURCE_ROLES,
        "bound source state mismatch during root collection",
    )
    validate_control_namespace(
        root, authorization_present=_lexists(root / AUTH_RELATIVE)
    )

    incident, correction = _expected_incident_identities()
    try:
        incident_chain = auditor._v5_validate_incident_chain(
            root, incident, correction
        )
    except Exception as exc:
        raise PostrunAuditSealError(
            f"V5 incident/correction chain invalid: {exc}"
        ) from exc
    _require(
        incident_chain.get("incident_identity") == incident
        and incident_chain.get("correction_identity") == correction
        and incident_chain.get("exact_one_override_applied") is True,
        "V5 incident/correction exact-one override mismatch",
    )
    correction_payload = incident_chain.get("correction")
    _require(
        isinstance(correction_payload, Mapping)
        and isinstance(correction_payload.get("created_utc"), str),
        "V5 incident correction chronology source mismatch",
    )

    base, _authorization, afterstate = _load_v4_authorization_base(root)
    truncation, review_created_utc = _load_v4_transport_truncation_state(
        root, base, _authorization, afterstate
    )
    commitment = _recovered_afterstate_commitment(afterstate)
    closure = validate_recovered_filesystem_closure(
        root, afterstate, _authorization
    )
    snapshot = build_zero_snapshot(root)
    return {
        "incident": incident,
        "incident_correction": correction,
        "v4_authorization_created_utc": _authorization["created_utc"],
        "v4_review_created_utc": review_created_utc,
        "incident_created_utc": incident_chain["incident"]["created_utc"],
        "incident_correction_created_utc": correction_payload["created_utc"],
        "v4_authorization_base": base,
        "v4_review_transport_truncation_state": truncation,
        "recovered_afterstate_commitment": commitment,
        "metadata_only_recovered_closure": closure,
        "preaudit_zero_mutation_snapshot": snapshot,
    }


def _source_function_names(path: Path) -> set[str]:
    source = path.read_text(encoding="utf-8")
    return {
        node.name
        for node in ast.walk(ast.parse(source, filename=str(path)))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _evidence_bound_identities(
    root: Path, source_bound: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    incident, correction = _expected_incident_identities()
    root_bound = {
        "v4_authorization": dict(auditor.FROZEN_V4_AUTH_IDENTITY),
        "v4_review_transport_truncation_incident": incident,
        "v4_review_transport_truncation_incident_correction": correction,
    }
    for role, record in root_bound.items():
        _require(
            strict_root_identity(root, str(record["path"]), label=role) == record,
            f"V5 evidence root identity mismatch: {role}",
        )
    combined = {**root_bound, **{role: dict(value) for role, value in source_bound.items()}}
    _require(
        set(combined) == BOUND_IDENTITY_ROLES and len(combined) == 11,
        "V5 evidence exact-eleven bound-identity schema mismatch",
    )
    return combined


def _compile_bound_sources(bound: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered_roles = ("v5_auditor", "v5_auditor_test", "v5_sealer", "v5_sealer_test")
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
    source_bound = code_state["bound_identities"]
    evidence_bound = _evidence_bound_identities(root, source_bound)
    compiled = _compile_bound_sources(source_bound)
    auditor_test_names = _source_function_names(Path(str(source_bound["v5_auditor_test"]["path"])))
    _require(set(auditor.V5_REQUIRED_TEST_NAMES).issubset(auditor_test_names), "required V5 regression test is absent")

    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.pop("PYTHONPYCACHEPREFIX", None)
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    environment["PYTHONIOENCODING"] = "utf-8"
    runs: dict[str, dict[str, Any]] = {}
    for role, relative in TEST_RELATIVES.items():
        _require_bound_bytecode_absent(source_bound)
        command = auditor._v5_test_command(relative)
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
        _require_bound_bytecode_absent(source_bound)
    _require(set(runs) == auditor.V5_TEST_RUN_KEYS, "test-run role mismatch")
    _require(collect_code_state(workspace_root) == code_state, "code/test identity changed during isolated tests")
    evidence = {
        "schema_version": 1,
        "artifact_type": "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_V5_TEST_EVIDENCE",
        "status": "PASS_FROZEN_V5_AUDITOR_AND_SEALER_TESTS",
        "created_utc": utc_now_exact(),
        "bound_identities": evidence_bound,
        "source_compile": {"method": "compile_exact_source_no_pyc", "result": "PASS", "files": compiled},
        "pytest_isolation": "ONE_TEST_FILE_PER_CLEAN_NETWORK_DENIED_SUBPROCESS",
        "test_runs": runs,
        "required_test_names": list(auditor.V5_REQUIRED_TEST_NAMES),
        "required_test_names_present": True,
        "all_exit_codes_zero": True,
        "production_shape_regression": {
            "v4_authorization_base_validated": True,
            "v4_transport_truncation_incident_validated": True,
            "compact_afterstate_commitment_validated": True,
            "full_afterstate_reconstructed_in_memory": True,
            "original_v1_four_key_identity_passed": True,
            "transaction_reached": True,
            "raw_reached": True,
            "decoded_reached": True,
            "offline_replay_reached": True,
            "synthetic_fixture": True,
        },
        "real_seven_spawn_regression": {
            "start_method": "spawn",
            "entrypoint_module": "scripts.audit_noaa_gfs_multiseason_raw_postrun_v5",
            "worker_callable_module": "scripts.audit_noaa_gfs_multiseason_raw_postrun_v5",
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
    _require(set(evidence) == auditor.V5_TEST_EVIDENCE_KEYS, "internal V5 test-evidence schema mismatch")
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
    audit_attempt_id = f"postrun_audit_v5__{compact}"
    bound = code_state["bound_identities"]
    payload = {
        "schema_version": 5,
        "artifact_type": "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDIT_AUTHORIZATION_V5",
        "status": "AUTHORIZED_PENDING_INDEPENDENT_REVIEW_AND_GO_V5",
        "created_utc": created_utc,
        "audit_attempt_id": audit_attempt_id,
        "incident": inputs["incident"],
        "incident_correction": inputs["incident_correction"],
        "v4_authorization_base": inputs["v4_authorization_base"],
        "v4_review_transport_truncation_state": inputs[
            "v4_review_transport_truncation_state"
        ],
        "superseded_v4_auditor": bound["superseded_v4_auditor"],
        "superseded_v4_auditor_test": bound["superseded_v4_auditor_test"],
        "superseded_v4_sealer": bound["superseded_v4_sealer"],
        "superseded_v4_sealer_test": bound["superseded_v4_sealer_test"],
        "v5_auditor": bound["v5_auditor"],
        "v5_auditor_test": bound["v5_auditor_test"],
        "v5_sealer": bound["v5_sealer"],
        "v5_sealer_test": bound["v5_sealer_test"],
        "recovery_attempt_id": auditor.V4_RECOVERY_ATTEMPT_ID,
        "recovered_afterstate_commitment": inputs[
            "recovered_afterstate_commitment"
        ],
        "preaudit_zero_mutation_snapshot": inputs["preaudit_zero_mutation_snapshot"],
        "runtime_identity_sha256": auditor.V4_RUNTIME_IDENTITY_SHA256,
        "test_evidence": dict(test_evidence),
        "required_command": required_audit_command(root),
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
    _require(set(payload) == auditor.V5_AUTH_KEYS, "V5 authorization schema mismatch")
    created_value = payload.get("created_utc")
    _require(
        isinstance(created_value, str)
        and re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}Z",
            created_value,
        )
        is not None,
        "V5 authorization created_utc is not exact UTC RFC3339 microseconds",
    )
    try:
        created = datetime.strptime(
            created_value, "%Y-%m-%dT%H:%M:%S.%fZ"
        ).replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise PostrunAuditSealError(
            "V5 authorization created_utc is invalid"
        ) from exc
    chronology_fields = (
        "v4_authorization_created_utc",
        "v4_review_created_utc",
        "incident_created_utc",
        "incident_correction_created_utc",
    )
    historical_created: dict[str, datetime] = {}
    for field in chronology_fields:
        value = inputs.get(field)
        _require(isinstance(value, str), f"V5 chronology binding is absent: {field}")
        try:
            historical_created[field] = datetime.strptime(
                value, "%Y-%m-%dT%H:%M:%S.%fZ"
            ).replace(tzinfo=timezone.utc)
        except ValueError as exc:
            raise PostrunAuditSealError(
                f"V5 chronology timestamp is invalid: {field}"
            ) from exc
    _require(
        historical_created["v4_authorization_created_utc"]
        <= historical_created["v4_review_created_utc"]
        < historical_created["incident_created_utc"]
        < historical_created["incident_correction_created_utc"]
        <= created,
        "V4/V5 incident and authority chronology mismatch",
    )
    expected_attempt_id = "postrun_audit_v5__" + str(
        payload.get("created_utc")
    ).replace("-", "").replace(":", "").replace(".", "")
    _require(
        payload.get("schema_version") == 5
        and payload.get("artifact_type") == "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDIT_AUTHORIZATION_V5"
        and payload.get("status") == "AUTHORIZED_PENDING_INDEPENDENT_REVIEW_AND_GO_V5"
        and isinstance(payload.get("audit_attempt_id"), str)
        and re.fullmatch(r"postrun_audit_v5__[0-9]{8}T[0-9]{12}Z", str(payload["audit_attempt_id"])) is not None,
        "V5 authorization header mismatch",
    )
    _require(
        payload.get("audit_attempt_id") == expected_attempt_id,
        "V5 authorization created-time/attempt binding mismatch",
    )
    _require(
        payload.get("max_spawn_processes") == 7
        and payload.get("network_requests_allowed") == 0
        and payload.get("audit_files_written_allowed") == 0
        and payload.get("labels_read_allowed") is False
        and payload.get("arrays_2024_read_allowed") is False
        and payload.get("arrays_2025_read_allowed") is False
        and payload.get("models_fit_allowed") == 0
        and payload.get("submission_csv_allowed") is False
        and payload.get("full_offline_redecode_required") is True
        and payload.get("stdout_only") is True,
        "V5 authorization execution policy mismatch",
    )
    try:
        auditor._v5_validate_policy(payload, label="V5 authorization sealer")
    except Exception as exc:
        raise PostrunAuditSealError(f"V5 authorization policy invalid: {exc}") from exc
    bound = code_state.get("bound_identities")
    _require(
        isinstance(bound, Mapping)
        and set(bound) == BOUND_SOURCE_ROLES,
        "V5 authorization bound-source role schema mismatch",
    )
    _require(
        bound["v5_auditor"] == EXPECTED_V5_AUDITOR_IDENTITY
        and bound["v5_auditor_test"] == EXPECTED_V5_AUDITOR_TEST_IDENTITY,
        "V5 authorization frozen auditor/test identity mismatch",
    )
    expected_source_bindings = {
        "superseded_v4_auditor": bound["superseded_v4_auditor"],
        "superseded_v4_auditor_test": bound["superseded_v4_auditor_test"],
        "superseded_v4_sealer": bound["superseded_v4_sealer"],
        "superseded_v4_sealer_test": bound["superseded_v4_sealer_test"],
        "v5_auditor": bound["v5_auditor"],
        "v5_auditor_test": bound["v5_auditor_test"],
        "v5_sealer": bound["v5_sealer"],
        "v5_sealer_test": bound["v5_sealer_test"],
    }
    _require(
        all(payload.get(role) == expected for role, expected in expected_source_bindings.items()),
        "V5 authorization top-level source identity binding mismatch",
    )
    _require(
        payload.get("recovery_attempt_id") == auditor.V4_RECOVERY_ATTEMPT_ID,
        "V5 authorization recovery-attempt binding mismatch",
    )
    _require(
        payload.get("runtime_identity_sha256") == auditor.V4_RUNTIME_IDENTITY_SHA256,
        "V5 authorization runtime-identity binding mismatch",
    )
    _require(
        payload.get("required_command") == required_audit_command(root),
        "V5 authorization required-command binding mismatch",
    )
    _require(
        payload.get("independent_review_required") is True
        and payload.get("independent_go_required") is True,
        "V5 authorization independent review/GO requirement mismatch",
    )
    _require(payload.get("incident") == inputs["incident"], "incident binding mismatch")
    _require(
        payload.get("incident_correction") == inputs["incident_correction"],
        "incident-correction binding mismatch",
    )
    _require(
        payload.get("v4_authorization_base")
        == inputs["v4_authorization_base"],
        "V4 authorization-base binding mismatch",
    )
    _require(
        payload.get("v4_review_transport_truncation_state")
        == inputs["v4_review_transport_truncation_state"],
        "V4 review transport-truncation-state binding mismatch",
    )
    try:
        auditor._v5_validate_transport_truncation_state(
            payload.get("v4_review_transport_truncation_state"),
            inputs["v4_review_transport_truncation_state"],
        )
    except Exception as exc:
        raise PostrunAuditSealError(
            f"V4 review transport-truncation state invalid: {exc}"
        ) from exc
    _require(
        payload.get("recovered_afterstate_commitment")
        == inputs["recovered_afterstate_commitment"]
        == auditor.V5_RECOVERED_AFTERSTATE_COMMITMENT,
        "compact recovered-afterstate commitment binding mismatch",
    )
    _require(
        payload.get("preaudit_zero_mutation_snapshot")
        == inputs["preaudit_zero_mutation_snapshot"]
        and set(payload["preaudit_zero_mutation_snapshot"])
        == auditor.V5_ZERO_SNAPSHOT_KEYS,
        "V5 preaudit zero-snapshot binding mismatch",
    )
    evidence = payload.get("test_evidence")
    expected_evidence_bound = {
        "v4_authorization": inputs["v4_authorization_base"]["identity"],
        "v4_review_transport_truncation_incident": inputs["incident"],
        "v4_review_transport_truncation_incident_correction": inputs[
            "incident_correction"
        ],
        **{role: bound[role] for role in BOUND_SOURCE_ROLES},
    }
    _require(
        set(expected_evidence_bound) == BOUND_IDENTITY_ROLES
        and len(expected_evidence_bound) == 11,
        "V5 authorization exact-eleven evidence binding mismatch",
    )
    _require(
        isinstance(evidence, Mapping)
        and set(evidence) == auditor.V5_TEST_EVIDENCE_KEYS
        and evidence.get("artifact_type")
        == "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_V5_TEST_EVIDENCE"
        and evidence.get("status")
        == "PASS_FROZEN_V5_AUDITOR_AND_SEALER_TESTS"
        and evidence.get("bound_identities") == expected_evidence_bound
        and set(evidence.get("test_runs", {})) == auditor.V5_TEST_RUN_KEYS
        and evidence.get("required_test_names")
        == list(auditor.V5_REQUIRED_TEST_NAMES)
        and evidence.get("required_test_names_present") is True
        and evidence.get("all_exit_codes_zero") is True
        and evidence.get("network_requests") == 0
        and evidence.get("audit_files_written") == 0
        and evidence.get("labels_read") is False
        and evidence.get("arrays_2024_read") is False
        and evidence.get("arrays_2025_read") is False
        and evidence.get("models_fit") == 0
        and evidence.get("submission_csv_created") is False,
        "V5 immutable test-evidence binding mismatch",
    )
    evidence_created = datetime.strptime(
        str(evidence["created_utc"]), "%Y-%m-%dT%H:%M:%S.%fZ"
    ).replace(tzinfo=timezone.utc)
    _require(evidence_created <= created, "V5 test evidence postdates authorization")
    try:
        auditor._v5_validate_zero_snapshot(
            root, payload.get("preaudit_zero_mutation_snapshot")
        )
        auditor._v5_validate_test_evidence(
            evidence,
            authorization_created=created,
            bound_identities=expected_evidence_bound,
        )
        auditor._v5_assert_compact_control(payload, label="V5 authorization sealer")
    except Exception as exc:
        raise PostrunAuditSealError(f"V5 authorization evidence invalid: {exc}") from exc


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
        "status": "PASS_V5_AUTHORIZATION_PRESEAL_STATIC_AUDIT",
        "artifact_type": "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDIT_V5_AUTH_SEALER_STATIC_AUDIT",
        "bound_identities": code_state["bound_identities"],
        "source_origin_trust_model": SOURCE_ORIGIN_TRUST_MODEL,
        "bound_executable_source_count": len(code_state["bound_identities"]),
        "recovered_afterstate_commitment_canonical_sha256": (
            auditor.V5_RECOVERED_AFTERSTATE_COMMITMENT_CANONICAL_SHA256
        ),
        "recovered_closure_mode": inputs["metadata_only_recovered_closure"][
            "mode"
        ],
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
    # imported auditor plus all eight bound code/test files after every subprocess and
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
            root, AUTH_RELATIVE, label="published V5 authorization"
        )
        _require(
            observed_identity == expected_published_identity,
            "published authorization identity changed after hard-link verification",
        )
        published_identity = observed_identity
        require_postpublication_control_state(root)
        _require(
            load_json_control(destination, label="published V5 authorization")
            == authorization,
            "published authorization payload mismatch",
        )
        try:
            strict_published = auditor._v5_load_strict_json(
                destination, label="published V5 authorization"
            )
        except Exception as exc:
            raise PostrunAuditSealError(
                f"published V5 authorization strict JSON rejection: {exc}"
            ) from exc
        _require(
            strict_published == authorization,
            "published authorization strict payload mismatch",
        )
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
        "status": "PASS_V5_AUTHORIZATION_SEALED_PENDING_INDEPENDENT_REVIEW_AND_GO",
        "artifact_type": "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDIT_V5_AUTH_SEAL_REPORT",
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
            "status": "FAIL_V5_POSTRUN_AUDIT_AUTHORIZATION_POSTPUBLICATION",
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
            "status": "FAIL_V5_POSTRUN_AUDIT_AUTHORIZATION_SEAL",
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

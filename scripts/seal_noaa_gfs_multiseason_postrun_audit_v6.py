#!/usr/bin/env python
"""Seal the append-only V6 postrun-audit authorization.

This publisher is deliberately narrower than the auditor it authorizes.  It
hashes immutable controls and eight bound code identities, validates recovered
files by metadata only, compiles the four executing V6 sources, runs four test
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


AUDITOR_SOURCE_PATH = REPO / "scripts" / "audit_noaa_gfs_multiseason_raw_postrun_v6.py"


def _require_preimport_cache_runtime() -> None:
    if not (
        os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
        and sys.dont_write_bytecode is True
        and "PYTHONPYCACHEPREFIX" not in os.environ
        and sys.pycache_prefix is None
    ):
        raise RuntimeError(
            "V6 sealer import requires -B, PYTHONDONTWRITEBYTECODE=1, "
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
            "V6 auditor matching bytecode/cache competitor exists: "
            + ", ".join(str(candidate) for candidate in matches)
        )


def _require_bound_bytecode_absent(
    bound: Mapping[str, Mapping[str, Any]],
) -> None:
    required = {
        "superseded_v5_auditor", "superseded_v5_auditor_test",
        "superseded_v5_sealer", "superseded_v5_sealer_test",
        "v6_auditor", "v6_auditor_test", "v6_sealer", "v6_sealer_test",
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
            "V6 auditor same-stem source-origin competitor exists: "
            + ", ".join(str(candidate) for candidate in competitors)
        )


def _preimport_auditor_source_guard(path: Path) -> None:
    """Reject top-level I/O/network routes before importing the moving V6 source."""

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
        raise RuntimeError("V6 auditor has a forbidden import-time route")
    allowed_top_level = {
        ast.Expr, ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign,
        ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.If,
    }
    if any(type(node) not in allowed_top_level for node in tree.body):
        raise RuntimeError("V6 auditor has an unexpected top-level statement")
    for node in tree.body:
        if isinstance(node, ast.Expr) and not isinstance(node.value, ast.Constant):
            raise RuntimeError("V6 auditor has a top-level expression side effect")
        if isinstance(node, ast.If):
            if (
                not isinstance(node.test, ast.Compare)
                or ast.unparse(node.test) != "__name__ == '__main__'"
                or node.orelse
            ):
                raise RuntimeError("V6 auditor has an unexpected top-level branch")
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            for call in (item for item in ast.walk(node) if isinstance(item, ast.Call)):
                function = ast.unparse(call.func)
                allowed = (
                    function in {"Path", "tuple", "str", "AuditContract", "GroupContract", "re.compile"}
                    or function.endswith((".resolve", ".replace", ".strip"))
                )
                if not allowed:
                    raise RuntimeError(
                        f"V6 auditor has an unapproved import-time call: {function}"
                    )


_require_preimport_cache_runtime()
_require_auditor_bytecode_absent(Path(__file__).resolve())
_require_auditor_bytecode_absent(AUDITOR_SOURCE_PATH)
_require_auditor_source_competitors_absent(AUDITOR_SOURCE_PATH)
_preimport_auditor_source_guard(AUDITOR_SOURCE_PATH)

from scripts import audit_noaa_gfs_multiseason_raw_postrun_v6 as auditor

if not (
    auditor.__name__ == "scripts.audit_noaa_gfs_multiseason_raw_postrun_v6"
    and auditor.__package__ == "scripts"
    and auditor.__spec__ is not None
    and auditor.__spec__.name == auditor.__name__
    and Path(auditor.__file__).resolve() == AUDITOR_SOURCE_PATH.resolve()
):
    raise RuntimeError("V6 auditor import source origin mismatch")


ROOT_DEFAULT = auditor.ROOT_DEFAULT
CANONICAL_MODULE_ENTRYPOINT = (
    "scripts.seal_noaa_gfs_multiseason_postrun_audit_v6"
)
AUTH_RELATIVE = auditor.V6_AUTH_RELATIVE
REVIEW_RELATIVE = auditor.V6_REVIEW_RELATIVE
GO_RELATIVE = auditor.V6_GO_RELATIVE
SEALER_TEST = REPO / "tests" / "test_seal_noaa_gfs_multiseason_postrun_audit_v6.py"
EXPECTED_V6_AUDITOR_IDENTITY = {
    "path": str(AUDITOR_SOURCE_PATH.resolve()),
    "size_bytes": 556_811,
    "sha256": "97519582cb57c426bd456a30f1cc34a27d966ae7fd1cc2df450d69795bfba415",
}
EXPECTED_V6_AUDITOR_TEST_IDENTITY = {
    "path": str(auditor.AUDITOR_TEST.resolve()),
    "size_bytes": 278_146,
    "sha256": "eda9db77675d69fc161c9838ae54d3e824859a44025235630b50ddb02875d97a",
}
FROZEN_V5_SOURCE_IDENTITIES = {
    "superseded_v5_auditor": {
    "path": str(auditor.FROZEN_V5_AUDITOR.resolve()),
    "size_bytes": 481_173,
    "sha256": "81e18256bf53bcdc3ba27a37787e1b2240bf0ea286d7f8c758d9f7d5e96ce6b1",
    },
    "superseded_v5_auditor_test": {
    "path": str(auditor.FROZEN_V5_AUDITOR_TEST.resolve()),
    "size_bytes": 228_289,
    "sha256": "45ce935aeb945a06d4eaafb0885320348d27e12d8343387f53a8cac108182fc1",
    },
    "superseded_v5_sealer": {
        "path": str(auditor.FROZEN_V5_SEALER.resolve()),
        "size_bytes": 76_117,
        "sha256": "bab2302dad801f554ea2c62546abd66edbca6e94e1841dcd05f4f5dbe7e215a4",
    },
    "superseded_v5_sealer_test": {
        "path": str(auditor.FROZEN_V5_SEALER_TEST.resolve()),
        "size_bytes": 39_039,
        "sha256": "215aa7295aeed09cfc799d7950999f19d9f243b62388d7b88a5e39db0a8dc963",
    },
}
V6_HISTORICAL_FALSE_REJECT_INCIDENT_IDENTITY = {
    "path": (
        "incidents/"
        "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_V5_"
        "HISTORICAL_IDENTITY_FALSE_REJECT_V1.json"
    ),
    "size_bytes": 10_523,
    "sha256": "09b92677c719631c512a4b227653a5e5b2def9dd4b64eca787bd7c323ba1d61d",
}
HISTORICAL_DIRECT_FILE_FAILURE_INCIDENT_IDENTITY = {
    "path": (
        "incidents/"
        "TRACK_A_DECODE_RECOVERY_SEAL_DIRECT_FILE_PICKLE_FAILURE_V1.json"
    ),
    "size_bytes": 7_732,
    "sha256": "9e74f57dcddbd0480464fda5953014e5be411b927de99beef58d7ab5bb10afbb",
}
HISTORICAL_FAILED_HEAD_IDENTITIES = {
    "recovery_core": {
        "path": str((REPO / "src" / "noaa_gfs_decode_recovery.py").resolve()),
        "size_bytes": 26_415,
        "sha256": "6d1f05e36a1d60cdb73a6d8cbd6e3cc9bdfeac36871d75a6fc27a76952a8825c",
    },
    "recovery_bootstrap": {
        "path": str(
            (REPO / "src" / "noaa_gfs_decode_recovery_bootstrap.py").resolve()
        ),
        "size_bytes": 16_925,
        "sha256": "e7637ea8f96a41d9cd9cb45467ede90b0dc08aaa143a97f28f438eacc79dc78e",
    },
    "recovery_runner": {
        "path": str(
            (
                REPO
                / "scripts"
                / "run_noaa_gfs_multiseason_decode_recovery_v1.py"
            ).resolve()
        ),
        "size_bytes": 117_445,
        "sha256": "9a1ea2a118d497dbe93fd9230dd82fbc832985f599a5c8ba13236a8d58919520",
    },
    "recovery_sealer": {
        "path": str(
            (
                REPO
                / "scripts"
                / "seal_noaa_gfs_multiseason_decode_recovery_v1.py"
            ).resolve()
        ),
        "size_bytes": 40_285,
        "sha256": "da4b3d5eef14ab0b156faed158af668a5627bb5dc2a50f7d8c5e7a4c8589370d",
    },
    "recovery_sealer_test": {
        "path": str(
            (
                REPO
                / "tests"
                / "test_seal_noaa_gfs_multiseason_decode_recovery_v1.py"
            ).resolve()
        ),
        "size_bytes": 7_261,
        "sha256": "3ea32a3a9ac17d1a2b36efacbde066bb89e7ca772173d2317425afcb8380b446",
    },
    "planned_postrun_auditor": {
        "path": str(
            (
                REPO
                / "scripts"
                / "audit_noaa_gfs_multiseason_raw_postrun_v1.py"
            ).resolve()
        ),
        "size_bytes": 247_925,
        "sha256": "84ba333a794430d95cb83a7c98d6d0477345d8b6c494b51505a83c49f1a2bc2b",
    },
    "planned_postrun_auditor_test": {
        "path": str(
            (
                REPO
                / "tests"
                / "test_noaa_gfs_multiseason_raw_postrun_v1.py"
            ).resolve()
        ),
        "size_bytes": 106_360,
        "sha256": "770bdfc2c657565237de7f90b488fb8aa47a6c43a8e50e1492eabebc4eb9f46e",
    },
}
HISTORICAL_V1_ABSENT_CONTROL_PATHS = (
    "prelaunch/DECODE_RECOVERY_RAW_CACHE_LOCK_V1.json",
    "prelaunch/DECODE_RECOVERY_REAL_PROCESS_PREFLIGHT_V1.json",
    "prereg/decode_recovery_authorization_v1.json",
    "independent_redteam/TRACK_A_DECODE_RECOVERY_INDEPENDENT_REVIEW_V1.json",
    "independent_redteam/TRACK_A_DECODE_RECOVERY_GO_V1.json",
)
HISTORICAL_V2_INCORRECT_CONTROL_PATHS = (
    "prelaunch/DECODE_RECOVERY_RAW_CACHE_LOCK_V2.json",
    "prelaunch/DECODE_RECOVERY_REAL_PROCESS_PREFLIGHT_V2.json",
    "prereg/decode_recovery_authorization_v2.json",
    "independent_redteam/TRACK_A_DECODE_RECOVERY_INDEPENDENT_REVIEW_V2.json",
    "independent_redteam/TRACK_A_DECODE_RECOVERY_GO_V2.json",
)
HISTORICAL_CONTRACT_DIGESTS = {
    "failed_head_identities": {
        "canonical_size_bytes": 1_672,
        "canonical_sha256": (
            "e3c36f0e55d12d4ff62c04feabdeadd4c5cbca9edad712fd50bf6dbde7f7cd1c"
        ),
    },
    "verified_zero_state_after_failure": {
        "canonical_size_bytes": 513,
        "canonical_sha256": (
            "c053b483bdeff428ada753609bc698677ac3f01169a953758412206cebb32f63"
        ),
    },
    "absent_control_paths": {
        "canonical_size_bytes": 288,
        "canonical_sha256": (
            "1a0505b4cfd9a93cf1838a9689058f7447645077008ac39272939f547ff9f524"
        ),
    },
}
HISTORICAL_VALIDATION_MODE = (
    "IMMUTABLE_9E74_HISTORICAL_CONTRACT_PREDATA_NO_CURRENT_ROLE_DEREFERENCE"
)
V5_AUTHORITY_BASE = {
    "authorization": {
        "path": "prereg/decode_recovery_postrun_audit_authorization_v5.json",
        "size_bytes": 23_792,
        "sha256": "7b23fcf77b35e529398bd27ee523c6c21773a152f791c599fc5af1f7b906cb3c",
    },
    "independent_review": {
        "path": (
            "independent_redteam/"
            "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_REVIEW_V5.json"
        ),
        "size_bytes": 8_455,
        "sha256": "5808d9b370ab769971a1502a659cf3849609cad4702c3cbb411b9db9f40c2ee1",
    },
    "independent_go": {
        "path": (
            "independent_redteam/"
            "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_GO_V5.json"
        ),
        "size_bytes": 7_346,
        "sha256": "ebb147b212d2c767e1fad1aaa631f2cc01f9292a769941f064e1fef01a0b022f",
    },
    "audit_attempt_id": "postrun_audit_v5__20260811T021615245290Z",
    "recovery_attempt_id": "decode_recovery_v2__20260810T211017500046Z",
    "authorization_status": "AUTHORIZED_PENDING_INDEPENDENT_REVIEW_AND_GO_V5",
    "review_status": "PASS_PENDING_INDEPENDENT_GO_V5",
    "go_status": "GO_V5_READ_ONLY_NETWORK_ZERO_FULL_OFFLINE_REPLAY",
    "recovered_afterstate_canonical_sha256": (
        "8e434429e6768e14dbfde2c2592609bcd8167e991fa5eaea65b5ba40e7d991cb"
    ),
}

TEST_RELATIVES = {
    "v6_auditor_tests": "tests/test_noaa_gfs_multiseason_raw_postrun_v6.py",
    "v6_sealer_tests": "tests/test_seal_noaa_gfs_multiseason_postrun_audit_v6.py",
    "frozen_v5_auditor_tests": "tests/test_noaa_gfs_multiseason_raw_postrun_v5.py",
    "frozen_v5_sealer_tests": "tests/test_seal_noaa_gfs_multiseason_postrun_audit_v5.py",
}
BOUND_SOURCE_PATHS = {
    "superseded_v5_auditor": auditor.FROZEN_V5_AUDITOR.resolve(),
    "superseded_v5_auditor_test": auditor.FROZEN_V5_AUDITOR_TEST.resolve(),
    "superseded_v5_sealer": auditor.FROZEN_V5_SEALER.resolve(),
    "superseded_v5_sealer_test": auditor.FROZEN_V5_SEALER_TEST.resolve(),
    "v6_auditor": Path(auditor.__file__).resolve(),
    "v6_auditor_test": auditor.AUDITOR_TEST.resolve(),
    "v6_sealer": Path(__file__).resolve(),
    "v6_sealer_test": SEALER_TEST.resolve(),
}
POSTRUN_REPORT_CANDIDATES = auditor.V6_POSTRUN_REPORT_CANDIDATES
CONTROL_TEMP_RELATIVES = (
    auditor.FROZEN_V3_AUTH_RELATIVE,
    auditor.FROZEN_V3_REVIEW_RELATIVE,
    auditor.FROZEN_V3_CANONICAL_GO_RELATIVE,
    auditor.FROZEN_V3_MISPLACED_GO_RELATIVE,
    auditor.V4_AUTH_RELATIVE,
    auditor.V4_REVIEW_RELATIVE,
    auditor.V4_GO_RELATIVE,
    auditor.V5_AUTH_RELATIVE,
    auditor.V5_REVIEW_RELATIVE,
    auditor.V5_GO_RELATIVE,
    AUTH_RELATIVE, REVIEW_RELATIVE, GO_RELATIVE,
    auditor.V2_FALSE_REJECT_INCIDENT_RELATIVE,
    auditor.V4_PATH_MISPUBLISH_INCIDENT_RELATIVE,
    auditor.V5_TRANSPORT_INCIDENT_RELATIVE,
    auditor.V5_TRANSPORT_CORRECTION_RELATIVE,
    auditor.V6_HISTORICAL_FALSE_REJECT_INCIDENT_RELATIVE,
)
BOUND_SOURCE_ROLES = {
    "superseded_v5_auditor", "superseded_v5_auditor_test",
    "superseded_v5_sealer", "superseded_v5_sealer_test",
    "v6_auditor", "v6_auditor_test", "v6_sealer", "v6_sealer_test",
}
BOUND_IDENTITY_ROLES = {
    "v5_authorization",
    "v5_independent_review",
    "v5_independent_go",
    "v5_historical_identity_false_reject_incident",
    "direct_file_failure_incident",
    *BOUND_SOURCE_ROLES,
}
SOURCE_ORIGIN_TRUST_MODEL = {
    "model": "NON_ADVERSARIAL_NO_CONCURRENT_FILESYSTEM_SWAP",
    "preimport_source_ast_and_competitor_scan": True,
    "pretest_posttest_and_prepublish_eight_source_rehash_and_pyc_scan": True,
    "test_evidence_exact_thirteen_bound_identities": True,
    "source_compile_exact_executing_v6_four": True,
    "adversarial_concurrent_swap_resistance_claimed": False,
}


class PostrunAuditSealError(RuntimeError):
    """The immutable V6 authorization could not be sealed safely."""


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
        "seal_noaa_gfs_multiseason_postrun_audit_v6` entrypoint",
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
    raise _NetworkDenied("network access is forbidden during V6 authorization sealing")


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
        "authorization": root_path(root, AUTH_RELATIVE, label="V6 authorization"),
        "independent_review": root_path(root, REVIEW_RELATIVE, label="V6 review"),
        "independent_go": root_path(root, GO_RELATIVE, label="V6 GO"),
    }


def required_audit_command(root: Path) -> list[str]:
    return [
        str(Path(sys.executable).resolve()),
        "-B",
        "-m",
        "scripts.audit_noaa_gfs_multiseason_raw_postrun_v6",
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
        path = root_path(root, relative, label=f"V6 control temp destination {relative}")
        parent = path.parent
        _require(parent.is_dir() and not _linklike(parent), "V6 control parent invalid")
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
        Path(auditor.V5_AUTH_RELATIVE).name,
    }
    if authorization_present:
        prereg.add(Path(AUTH_RELATIVE).name)
    return {
        "prereg": prereg,
        "independent_redteam": {
            Path(auditor.FROZEN_V3_REVIEW_RELATIVE).name,
            Path(auditor.V4_REVIEW_RELATIVE).name,
            Path(auditor.V5_REVIEW_RELATIVE).name,
            Path(auditor.V5_GO_RELATIVE).name,
        },
        "incidents": {
            Path(auditor.V2_FALSE_REJECT_INCIDENT_RELATIVE).name,
            Path(auditor.V4_PATH_MISPUBLISH_INCIDENT_RELATIVE).name,
            Path(auditor.V5_TRANSPORT_INCIDENT_RELATIVE).name,
            Path(auditor.V5_TRANSPORT_CORRECTION_RELATIVE).name,
            Path(auditor.V6_HISTORICAL_FALSE_REJECT_INCIDENT_RELATIVE).name,
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
        parent = root_path(root, relative, label=f"V6 namespace {relative}")
        _require(
            _lexists(parent) and parent.is_dir() and not _linklike(parent),
            f"V6 control namespace parent invalid: {relative}",
        )
        matched: list[str] = []
        for entry in parent.iterdir():
            if not entry.name.casefold().lstrip(".").startswith(prefix):
                continue
            _require(
                _lexists(entry) and entry.is_file() and not _linklike(entry),
                f"V6 control namespace contains non-regular/link entry: {entry}",
            )
            matched.append(entry.name)
        _require(
            set(matched) == expected[relative]
            and len({name.casefold() for name in matched}) == len(matched),
            f"V6 control namespace inventory mismatch: {relative}",
        )
        result[relative] = sorted(matched, key=str.casefold)
    return result


def require_prepublication_control_state(root: Path) -> dict[str, Any]:
    paths = control_paths(root)
    _require(
        all(not _lexists(path) for path in paths.values()),
        "V6 authorization/review/GO already exists",
    )
    _require(control_temporary_files(root) == [], "V6 control temporary sibling exists")
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
        "published V6 authorization is absent, non-regular or link-like",
    )
    _require(
        not _lexists(paths["independent_review"])
        and not _lexists(paths["independent_go"]),
        "V6 sealer observed an independent review/GO",
    )
    _require(control_temporary_files(root) == [], "V6 control temporary sibling exists")
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
    _require(Path(auditor.V6_SEALER).resolve() == Path(__file__).resolve(), "auditor V6 sealer path mismatch")
    _require(Path(auditor.V6_SEALER_TEST).resolve() == SEALER_TEST.resolve(), "auditor V6 sealer-test path mismatch")

    bound = {
        role: strict_external_identity(path, label=role)
        for role, path in BOUND_SOURCE_PATHS.items()
    }
    _require(set(bound) == BOUND_SOURCE_ROLES, "bound source role schema mismatch")
    for role, expected in FROZEN_V5_SOURCE_IDENTITIES.items():
        _require(bound[role] == expected, f"frozen V5 source identity mismatch: {role}")
    _require(
        bound["v6_auditor"] == EXPECTED_V6_AUDITOR_IDENTITY,
        "frozen executing V6 auditor identity mismatch",
    )
    _require(
        bound["v6_auditor_test"] == EXPECTED_V6_AUDITOR_TEST_IDENTITY,
        "frozen executing V6 auditor-test identity mismatch",
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


def expected_historical_failed_head_identity_contract() -> dict[str, Any]:
    return {
        "incident": dict(HISTORICAL_DIRECT_FILE_FAILURE_INCIDENT_IDENTITY),
        "historical_incident_contract": {
            key: dict(value) for key, value in HISTORICAL_CONTRACT_DIGESTS.items()
        },
        "validation_mode": HISTORICAL_VALIDATION_MODE,
        "validation_moved_before_parquet_and_data_reads": True,
        "current_role_path_dereference_forbidden": True,
    }


def _validate_historical_failed_head_identity_contract(
    root: Path,
    declared: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate immutable 9e74 facts without opening any current-role source."""

    expected = expected_historical_failed_head_identity_contract()
    _require(
        isinstance(declared, Mapping)
        and set(declared)
        == {
            "incident",
            "historical_incident_contract",
            "validation_mode",
            "validation_moved_before_parquet_and_data_reads",
            "current_role_path_dereference_forbidden",
        }
        and dict(declared) == expected,
        "V6 historical failed-head identity contract mismatch",
    )
    incident_identity = strict_root_identity(
        root,
        HISTORICAL_DIRECT_FILE_FAILURE_INCIDENT_IDENTITY["path"],
        label="historical direct-file failure incident",
    )
    _require(
        incident_identity == HISTORICAL_DIRECT_FILE_FAILURE_INCIDENT_IDENTITY,
        "historical direct-file failure incident identity mismatch",
    )
    payload = load_json_control(
        root_path(
            root,
            HISTORICAL_DIRECT_FILE_FAILURE_INCIDENT_IDENTITY["path"],
            label="historical direct-file failure incident",
        ),
        label="historical direct-file failure incident",
    )
    _require(
        payload.get("schema_version") == 1
        and payload.get("artifact_type")
        == "TRACK_A_DECODE_RECOVERY_SEAL_DIRECT_FILE_PICKLE_FAILURE_INCIDENT"
        and payload.get("status")
        == "FAILED_BEFORE_PUBLICATION_RETRY_FORBIDDEN_PENDING_REFREEZE",
        "historical direct-file failure incident header mismatch",
    )

    failed_heads = payload.get("failed_head_identities")
    _require(
        isinstance(failed_heads, Mapping)
        and len(failed_heads) == 7
        and set(failed_heads) == set(HISTORICAL_FAILED_HEAD_IDENTITIES)
        and dict(failed_heads) == HISTORICAL_FAILED_HEAD_IDENTITIES,
        "historical exact-seven failed-head identities mismatch",
    )
    zero_state = payload.get("verified_zero_state_after_failure")
    _require(
        isinstance(zero_state, Mapping)
        and set(zero_state)
        == {
            "absent_control_paths",
            "canonical_recovery_outputs_absent",
            "recovery_active_locks_absent",
            "seal_temporary_files_absent",
            "matching_bootstrap_pyc_files",
            "only_original_failed_output_transaction_present",
        },
        "historical verified-zero-state schema mismatch",
    )
    absent_paths = zero_state.get("absent_control_paths")
    _require(
        isinstance(absent_paths, list)
        and len(absent_paths) == 5
        and absent_paths == list(HISTORICAL_V1_ABSENT_CONTROL_PATHS)
        and not any(path in HISTORICAL_V2_INCORRECT_CONTROL_PATHS for path in absent_paths)
        and zero_state.get("canonical_recovery_outputs_absent") is True
        and zero_state.get("recovery_active_locks_absent") is True
        and zero_state.get("seal_temporary_files_absent") is True
        and zero_state.get("matching_bootstrap_pyc_files") == 0
        and zero_state.get("only_original_failed_output_transaction_present") is True,
        "historical V1 exact-five zero-state mismatch",
    )
    observed_values = {
        "failed_head_identities": failed_heads,
        "verified_zero_state_after_failure": zero_state,
        "absent_control_paths": absent_paths,
    }
    for key, value in observed_values.items():
        length, digest = _canonical_length_sha256(value)
        _require(
            length == HISTORICAL_CONTRACT_DIGESTS[key]["canonical_size_bytes"]
            and digest == HISTORICAL_CONTRACT_DIGESTS[key]["canonical_sha256"],
            f"historical {key} canonical commitment mismatch",
        )
    return {
        "contract": expected,
        "incident_payload": payload,
        "failed_head_count": len(failed_heads),
        "absent_control_path_count": len(absent_paths),
    }


def _load_and_validate_v6_false_reject_incident(
    root: Path,
    declared_incident: Mapping[str, Any],
    declared_history: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        isinstance(declared_incident, Mapping)
        and dict(declared_incident) == V6_HISTORICAL_FALSE_REJECT_INCIDENT_IDENTITY,
        "V6 false-reject incident declaration mismatch",
    )
    observed_incident = strict_root_identity(
        root,
        V6_HISTORICAL_FALSE_REJECT_INCIDENT_IDENTITY["path"],
        label="V6 historical false-reject incident",
    )
    _require(
        observed_incident == V6_HISTORICAL_FALSE_REJECT_INCIDENT_IDENTITY,
        "V6 historical false-reject incident identity mismatch",
    )
    incident = load_json_control(
        root_path(
            root,
            V6_HISTORICAL_FALSE_REJECT_INCIDENT_IDENTITY["path"],
            label="V6 historical false-reject incident",
        ),
        label="V6 historical false-reject incident",
    )
    _require(
        set(incident)
        == {
            "artifact_type",
            "authority_scope",
            "created_utc",
            "exact_mismatch_set",
            "execution_boundary",
            "failed_execution",
            "historical_incident",
            "immutable_v5_chain",
            "postfailure_state",
            "prohibitions",
            "required_v6_supersession",
            "root_cause",
            "schema_version",
            "status",
        }
        and incident.get("schema_version") == 1
        and incident.get("artifact_type")
        == (
            "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_V5_"
            "HISTORICAL_IDENTITY_FALSE_REJECT_INCIDENT"
        )
        and incident.get("status")
        == (
            "SEALED_V5_POSTRUN_AUDITOR_HISTORICAL_IDENTITY_FALSE_REJECT_"
            "AFTER_FULL_REPLAY_NO_REPORT_V6_SUPERSESSION_REQUIRED"
        )
        and incident.get("authority_scope")
        == {
            "documentary_only": True,
            "v5_retry_authorized": False,
            "v6_authorized": False,
        },
        "V6 historical false-reject incident header mismatch",
    )
    required = incident.get("required_v6_supersession")
    _require(
        required
        == {
            "code": [
                "scripts/audit_noaa_gfs_multiseason_raw_postrun_v6.py",
                "tests/test_noaa_gfs_multiseason_raw_postrun_v6.py",
                "scripts/seal_noaa_gfs_multiseason_postrun_audit_v6.py",
                "tests/test_seal_noaa_gfs_multiseason_postrun_audit_v6.py",
            ],
            "controls": [
                "prereg/decode_recovery_postrun_audit_authorization_v6.json",
                (
                    "independent_redteam/"
                    "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_REVIEW_V6.json"
                ),
                (
                    "independent_redteam/"
                    "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_GO_V6.json"
                ),
            ],
            "fix": [
                "freeze exact historical heads and V1 zero-state",
                "forbid current-role/control substitution",
                "validate 9e74 before parquet/schema/data/replay",
                "add production-9e74 positive and head/zero-state tamper tests",
                (
                    "bind this incident in V6 AUTH/REVIEW/GO and keep V5 "
                    "production data/replay core AST identical while retaining "
                    "stdout-only network-zero policy"
                ),
            ],
            "incident_relative_path": (
                V6_HISTORICAL_FALSE_REJECT_INCIDENT_IDENTITY["path"]
            ),
            "must_bind_this_incident_identity": True,
            "report_file": None,
            "status": "REQUIRED_NOT_YET_AUTHORIZED",
        },
        "V6 false-reject incident required-supersession contract mismatch",
    )
    historical = _validate_historical_failed_head_identity_contract(
        root, declared_history
    )
    declared_historical = incident.get("historical_incident")
    _require(
        isinstance(declared_historical, Mapping)
        and declared_historical
        == {
            "identity": HISTORICAL_DIRECT_FILE_FAILURE_INCIDENT_IDENTITY,
            "failed_heads": {
                "size_bytes": HISTORICAL_CONTRACT_DIGESTS[
                    "failed_head_identities"
                ]["canonical_size_bytes"],
                "sha256": HISTORICAL_CONTRACT_DIGESTS[
                    "failed_head_identities"
                ]["canonical_sha256"],
            },
            "zero_state": {
                "size_bytes": HISTORICAL_CONTRACT_DIGESTS[
                    "verified_zero_state_after_failure"
                ]["canonical_size_bytes"],
                "sha256": HISTORICAL_CONTRACT_DIGESTS[
                    "verified_zero_state_after_failure"
                ]["canonical_sha256"],
            },
            "absent_control_paths": {
                "size_bytes": HISTORICAL_CONTRACT_DIGESTS[
                    "absent_control_paths"
                ]["canonical_size_bytes"],
                "sha256": HISTORICAL_CONTRACT_DIGESTS[
                    "absent_control_paths"
                ]["canonical_sha256"],
            },
        },
        "V6 incident historical digest declarations mismatch",
    )
    mismatch = incident.get("exact_mismatch_set")
    _require(
        isinstance(mismatch, Mapping)
        and set(mismatch)
        == {"failed_head_role_path_drift", "zero_state_control_version_drift"},
        "V6 incident exact-mismatch schema mismatch",
    )
    role_drift = mismatch["failed_head_role_path_drift"]
    _require(
        isinstance(role_drift, Mapping)
        and role_drift.get("matching_roles")
        == ["recovery_core", "recovery_bootstrap"]
        and isinstance(role_drift.get("mismatches"), list)
        and len(role_drift["mismatches"]) == 5,
        "V6 incident failed-head mismatch inventory invalid",
    )
    expected_drift_roles = (
        "recovery_runner",
        "recovery_sealer",
        "recovery_sealer_test",
        "planned_postrun_auditor",
        "planned_postrun_auditor_test",
    )
    expected_v2_paths = (
        "scripts/run_noaa_gfs_multiseason_decode_recovery_v2.py",
        "scripts/seal_noaa_gfs_multiseason_decode_recovery_v2.py",
        "tests/test_seal_noaa_gfs_multiseason_decode_recovery_v2.py",
        "scripts/audit_noaa_gfs_multiseason_raw_postrun_v2.py",
        "tests/test_noaa_gfs_multiseason_raw_postrun_v2.py",
    )
    for record, role, current_path in zip(
        role_drift["mismatches"], expected_drift_roles, expected_v2_paths
    ):
        expected_historical = dict(HISTORICAL_FAILED_HEAD_IDENTITIES[role])
        expected_historical["path"] = Path(expected_historical["path"]).relative_to(
            REPO
        ).as_posix()
        _require(
            isinstance(record, Mapping)
            and set(record) == {"historical", "incorrect_current_path", "role"}
            and record.get("role") == role
            and record.get("incorrect_current_path") == current_path
            and record.get("historical") == expected_historical,
            f"V6 incident mismatch record invalid: {role}",
        )
    zero_drift = mismatch["zero_state_control_version_drift"]
    _require(
        zero_drift
        == {
            "historical_v1_absent_paths": list(HISTORICAL_V1_ABSENT_CONTROL_PATHS),
            "incorrect_current_v2_expected_paths": list(
                HISTORICAL_V2_INCORRECT_CONTROL_PATHS
            ),
        },
        "V6 incident zero-state control-version drift mismatch",
    )
    return {
        "incident": observed_incident,
        "incident_payload": incident,
        "historical_contract": historical["contract"],
        "historical_incident_payload": historical["incident_payload"],
        "failed_head_count": historical["failed_head_count"],
        "absent_control_path_count": historical["absent_control_path_count"],
    }


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
    _require(temporary == [], "V6 control temporary sibling exists")
    _require(reports == [], "postrun audit output file exists despite stdout-only contract")
    snapshot = {
        "incident_postfailure_state_canonical_sha256": (
            "866d6564abb4d00cb1cb2961677cf767b9934ba44829cee3fec0ff46e9961690"
        ),
        "recovered_afterstate_commitment_canonical_sha256": (
            auditor.V5_RECOVERED_AFTERSTATE_COMMITMENT_CANONICAL_SHA256
        ),
        "active_locks": active,
        "v6_control_temporary_files": {
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
        set(snapshot) == auditor.V6_ZERO_SNAPSHOT_KEYS,
        "V6 zero-snapshot schema mismatch",
    )
    _require(
        snapshot["v6_control_temporary_files"]
        == {
            "checked_destination_relative_paths": list(CONTROL_TEMP_RELATIVES),
            "matching_temporary_files": [],
        },
        "V6 zero-snapshot temporary-destination declaration mismatch",
    )
    try:
        auditor._v6_validate_zero_snapshot(root, snapshot)
    except Exception as exc:
        raise PostrunAuditSealError(f"V6 zero-snapshot invalid: {exc}") from exc
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

    incident = dict(V6_HISTORICAL_FALSE_REJECT_INCIDENT_IDENTITY)
    historical_contract = expected_historical_failed_head_identity_contract()
    local_incident = _load_and_validate_v6_false_reject_incident(
        root, incident, historical_contract
    )
    try:
        auditor_incident = auditor._v6_validate_v5_false_reject_incident(
            root, incident
        )
        base_state = auditor._v6_load_and_validate_v5_authority_base(
            root, V5_AUTHORITY_BASE
        )
    except Exception as exc:
        raise PostrunAuditSealError(f"V6 frozen authority input invalid: {exc}") from exc
    _require(
        auditor_incident.get("identity") == incident
        and local_incident["incident"] == incident
        and base_state.get("base") == V5_AUTHORITY_BASE,
        "V6 incident/base independent validator result mismatch",
    )
    direct_payload = local_incident["historical_incident_payload"]
    try:
        auditor_historical = auditor._v6_validate_historical_9e74_payload(
            direct_payload, HISTORICAL_DIRECT_FILE_FAILURE_INCIDENT_IDENTITY
        )
        auditor._v6_validate_historical_contract(
            historical_contract, auditor_historical
        )
    except Exception as exc:
        raise PostrunAuditSealError(
            f"V6 historical 9e74 contract invalid: {exc}"
        ) from exc
    _require(
        auditor_historical == historical_contract
        and local_incident["failed_head_count"] == 7
        and local_incident["absent_control_path_count"] == 5,
        "V6 local/auditor historical contract mismatch",
    )
    afterstate = base_state["recovered_afterstate"]
    commitment = _recovered_afterstate_commitment(afterstate)
    closure = validate_recovered_filesystem_closure(
        root, afterstate, base_state["v4_base"]["authorization"]
    )
    snapshot = build_zero_snapshot(root)
    return {
        "incident": incident,
        "incident_created_utc": local_incident["incident_payload"]["created_utc"],
        "historical_incident_created_utc": direct_payload["created_utc"],
        "v5_authorization_created_utc": base_state["authorization"]["created_utc"],
        "v5_review_created_utc": base_state["review"]["created_utc"],
        "v5_go_created_utc": base_state["go"]["created_utc"],
        "v5_authority_base": dict(V5_AUTHORITY_BASE),
        "historical_failed_head_identity_contract": historical_contract,
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
    root_bound = {
        "v5_authorization": dict(V5_AUTHORITY_BASE["authorization"]),
        "v5_independent_review": dict(V5_AUTHORITY_BASE["independent_review"]),
        "v5_independent_go": dict(V5_AUTHORITY_BASE["independent_go"]),
        "v5_historical_identity_false_reject_incident": dict(
            V6_HISTORICAL_FALSE_REJECT_INCIDENT_IDENTITY
        ),
        "direct_file_failure_incident": dict(
            HISTORICAL_DIRECT_FILE_FAILURE_INCIDENT_IDENTITY
        ),
    }
    for role, record in root_bound.items():
        _require(
            strict_root_identity(root, str(record["path"]), label=role) == record,
            f"V6 evidence root identity mismatch: {role}",
        )
    combined = {**root_bound, **{role: dict(value) for role, value in source_bound.items()}}
    _require(
        set(combined) == BOUND_IDENTITY_ROLES and len(combined) == 13,
        "V6 evidence exact-thirteen bound-identity schema mismatch",
    )
    return combined


def _compile_bound_sources(bound: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered_roles = ("v6_auditor", "v6_auditor_test", "v6_sealer", "v6_sealer_test")
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
    auditor_test_names = _source_function_names(Path(str(source_bound["v6_auditor_test"]["path"])))
    _require(set(auditor.V6_REQUIRED_TEST_NAMES).issubset(auditor_test_names), "required V6 regression test is absent")

    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.pop("PYTHONPYCACHEPREFIX", None)
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    environment["PYTHONIOENCODING"] = "utf-8"
    runs: dict[str, dict[str, Any]] = {}
    for role, relative in TEST_RELATIVES.items():
        _require_bound_bytecode_absent(source_bound)
        command = auditor._v6_test_command(relative)
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
    _require(set(runs) == auditor.V6_TEST_RUN_KEYS, "test-run role mismatch")
    _require(collect_code_state(workspace_root) == code_state, "code/test identity changed during isolated tests")
    evidence = {
        "schema_version": 1,
        "artifact_type": "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_V6_TEST_EVIDENCE",
        "status": "PASS_FROZEN_V6_AUDITOR_AND_SEALER_TESTS",
        "created_utc": utc_now_exact(),
        "bound_identities": evidence_bound,
        "source_compile": {"method": "compile_exact_source_no_pyc", "result": "PASS", "files": compiled},
        "pytest_isolation": "ONE_TEST_FILE_PER_CLEAN_NETWORK_DENIED_SUBPROCESS",
        "test_runs": runs,
        "required_test_names": list(auditor.V6_REQUIRED_TEST_NAMES),
        "required_test_names_present": True,
        "all_exit_codes_zero": True,
        "production_shape_regression": {
            "actual_root_historical_predata_validated": True,
            "compact_afterstate_commitment_validated": True,
            "decoded_reached": True,
            "direct_file_failure_incident_9e74_validated_predata": True,
            "full_afterstate_reconstructed_in_memory": True,
            "historical_current_role_distinction_validated": True,
            "offline_replay_reached": True,
            "original_v1_four_key_identity_passed": True,
            "raw_reached": True,
            "synthetic_fixture": True,
            "transaction_reached": True,
            "v5_authority_base_validated": True,
            "v5_false_reject_incident_validated": True,
            "v5_production_data_and_replay_core_ast_identical": True,
        },
        "real_seven_spawn_regression": {
            "start_method": "spawn",
            "entrypoint_module": "scripts.audit_noaa_gfs_multiseason_raw_postrun_v6",
            "worker_callable_module": "scripts.audit_noaa_gfs_multiseason_raw_postrun_v6",
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
    _require(set(evidence) == auditor.V6_TEST_EVIDENCE_KEYS, "internal V6 test-evidence schema mismatch")
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
    audit_attempt_id = f"postrun_audit_v6__{compact}"
    bound = code_state["bound_identities"]
    payload = {
        "schema_version": 6,
        "artifact_type": "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDIT_AUTHORIZATION_V6",
        "status": "AUTHORIZED_PENDING_INDEPENDENT_REVIEW_AND_GO_V6",
        "created_utc": created_utc,
        "audit_attempt_id": audit_attempt_id,
        "incident": inputs["incident"],
        "v5_authority_base": inputs["v5_authority_base"],
        "historical_failed_head_identity_contract": inputs[
            "historical_failed_head_identity_contract"
        ],
        "superseded_v5_auditor": bound["superseded_v5_auditor"],
        "superseded_v5_auditor_test": bound["superseded_v5_auditor_test"],
        "superseded_v5_sealer": bound["superseded_v5_sealer"],
        "superseded_v5_sealer_test": bound["superseded_v5_sealer_test"],
        "v6_auditor": bound["v6_auditor"],
        "v6_auditor_test": bound["v6_auditor_test"],
        "v6_sealer": bound["v6_sealer"],
        "v6_sealer_test": bound["v6_sealer_test"],
        "recovery_attempt_id": V5_AUTHORITY_BASE["recovery_attempt_id"],
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
    _require(set(payload) == auditor.V6_AUTH_KEYS, "V6 authorization schema mismatch")

    def parse_created(value: Any, *, label: str) -> datetime:
        _require(isinstance(value, str), f"{label} timestamp is absent")
        for pattern in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
            try:
                return datetime.strptime(value, pattern).replace(tzinfo=timezone.utc)
            except ValueError:
                pass
        raise PostrunAuditSealError(f"{label} timestamp is invalid")

    created_value = payload.get("created_utc")
    _require(
        isinstance(created_value, str)
        and re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}Z",
            created_value,
        )
        is not None,
        "V6 authorization created_utc is not exact UTC RFC3339 microseconds",
    )
    created = parse_created(created_value, label="V6 authorization")
    chronology = {
        field: parse_created(inputs.get(field), label=field)
        for field in (
            "historical_incident_created_utc",
            "v5_authorization_created_utc",
            "v5_review_created_utc",
            "v5_go_created_utc",
            "incident_created_utc",
        )
    }
    _require(
        chronology["historical_incident_created_utc"]
        < chronology["v5_authorization_created_utc"]
        < chronology["v5_review_created_utc"]
        < chronology["v5_go_created_utc"]
        < chronology["incident_created_utc"]
        < created,
        "V5 AUTH/REVIEW/GO/false-reject/V6 authorization chronology mismatch",
    )
    expected_attempt_id = "postrun_audit_v6__" + created_value.translate(
        str.maketrans("", "", "-:.")
    )
    _require(
        payload.get("schema_version") == 6
        and payload.get("artifact_type")
        == "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDIT_AUTHORIZATION_V6"
        and payload.get("status")
        == "AUTHORIZED_PENDING_INDEPENDENT_REVIEW_AND_GO_V6"
        and payload.get("audit_attempt_id") == expected_attempt_id
        and re.fullmatch(
            r"postrun_audit_v6__[0-9]{8}T[0-9]{12}Z",
            str(payload.get("audit_attempt_id")),
        )
        is not None,
        "V6 authorization header/attempt mismatch",
    )
    try:
        auditor._v6_validate_policy(payload, label="V6 authorization sealer")
    except Exception as exc:
        raise PostrunAuditSealError(f"V6 authorization policy invalid: {exc}") from exc

    bound = code_state.get("bound_identities")
    _require(
        isinstance(bound, Mapping) and set(bound) == BOUND_SOURCE_ROLES,
        "V6 authorization bound-source role schema mismatch",
    )
    for role, expected in FROZEN_V5_SOURCE_IDENTITIES.items():
        _require(bound[role] == expected, f"frozen V5 source changed: {role}")
    _require(
        bound["v6_auditor"] == EXPECTED_V6_AUDITOR_IDENTITY
        and bound["v6_auditor_test"] == EXPECTED_V6_AUDITOR_TEST_IDENTITY,
        "V6 authorization frozen auditor/test identity mismatch",
    )
    _require(
        all(payload.get(role) == bound[role] for role in BOUND_SOURCE_ROLES),
        "V6 authorization source identity binding mismatch",
    )
    _require(
        payload.get("recovery_attempt_id") == V5_AUTHORITY_BASE["recovery_attempt_id"]
        and payload.get("runtime_identity_sha256")
        == auditor.V4_RUNTIME_IDENTITY_SHA256
        and payload.get("required_command") == required_audit_command(root)
        and payload.get("independent_review_required") is True
        and payload.get("independent_go_required") is True,
        "V6 authorization immutable execution binding mismatch",
    )
    _require(
        payload.get("incident") == inputs["incident"]
        == V6_HISTORICAL_FALSE_REJECT_INCIDENT_IDENTITY
        and payload.get("v5_authority_base") == inputs["v5_authority_base"]
        == V5_AUTHORITY_BASE
        and payload.get("historical_failed_head_identity_contract")
        == inputs["historical_failed_head_identity_contract"]
        == expected_historical_failed_head_identity_contract(),
        "V6 incident/base/historical contract binding mismatch",
    )
    local_incident = _load_and_validate_v6_false_reject_incident(
        root,
        payload["incident"],
        payload["historical_failed_head_identity_contract"],
    )
    try:
        auditor_incident = auditor._v6_validate_v5_false_reject_incident(
            root, payload["incident"]
        )
        base_state = auditor._v6_load_and_validate_v5_authority_base(
            root, payload["v5_authority_base"]
        )
        auditor_historical = auditor._v6_validate_historical_9e74_payload(
            local_incident["historical_incident_payload"],
            HISTORICAL_DIRECT_FILE_FAILURE_INCIDENT_IDENTITY,
        )
        auditor._v6_validate_historical_contract(
            payload["historical_failed_head_identity_contract"],
            auditor_historical,
        )
    except Exception as exc:
        raise PostrunAuditSealError(
            f"V6 authority incident/base/history revalidation failed: {exc}"
        ) from exc
    _require(
        auditor_incident.get("identity") == payload["incident"]
        and base_state.get("base") == payload["v5_authority_base"]
        and auditor_historical
        == payload["historical_failed_head_identity_contract"],
        "V6 independent local/auditor authority input mismatch",
    )
    _require(
        parse_created(
            local_incident["historical_incident_payload"].get("created_utc"),
            label="historical 9e74 incident helper",
        )
        == chronology["historical_incident_created_utc"]
        and base_state.get("authorization_created")
        == chronology["v5_authorization_created_utc"]
        and base_state.get("review_created")
        == chronology["v5_review_created_utc"]
        and base_state.get("go_created") == chronology["v5_go_created_utc"]
        and parse_created(
            local_incident["incident_payload"].get("created_utc"),
            label="V5 historical-identity false-reject incident helper",
        )
        == auditor_incident.get("created")
        == chronology["incident_created_utc"],
        "V6 helper-backed authority chronology mismatch",
    )
    _require(
        payload.get("recovered_afterstate_commitment")
        == inputs["recovered_afterstate_commitment"]
        == base_state["commitment"]
        == auditor.V5_RECOVERED_AFTERSTATE_COMMITMENT,
        "V6 compact recovered-afterstate commitment binding mismatch",
    )
    _require(
        payload.get("preaudit_zero_mutation_snapshot")
        == inputs["preaudit_zero_mutation_snapshot"]
        and set(payload["preaudit_zero_mutation_snapshot"])
        == auditor.V6_ZERO_SNAPSHOT_KEYS,
        "V6 preaudit zero-snapshot binding mismatch",
    )
    evidence = payload.get("test_evidence")
    expected_evidence_bound = {
        "v5_authorization": V5_AUTHORITY_BASE["authorization"],
        "v5_independent_review": V5_AUTHORITY_BASE["independent_review"],
        "v5_independent_go": V5_AUTHORITY_BASE["independent_go"],
        "v5_historical_identity_false_reject_incident": inputs["incident"],
        "direct_file_failure_incident": (
            HISTORICAL_DIRECT_FILE_FAILURE_INCIDENT_IDENTITY
        ),
        **{role: bound[role] for role in BOUND_SOURCE_ROLES},
    }
    _require(
        set(expected_evidence_bound) == BOUND_IDENTITY_ROLES
        and len(expected_evidence_bound) == 13
        and isinstance(evidence, Mapping)
        and evidence.get("bound_identities") == expected_evidence_bound,
        "V6 authorization exact-thirteen evidence binding mismatch",
    )
    try:
        auditor._v6_validate_zero_snapshot(
            root, payload["preaudit_zero_mutation_snapshot"]
        )
        auditor._v6_validate_test_evidence(
            evidence,
            authorization_created=created,
            bound_identities=expected_evidence_bound,
        )
        auditor._v6_assert_compact_control(payload, label="V6 authorization sealer")
    except Exception as exc:
        raise PostrunAuditSealError(f"V6 authorization evidence invalid: {exc}") from exc


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
        "status": "PASS_V6_AUTHORIZATION_PRESEAL_STATIC_AUDIT",
        "artifact_type": "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDIT_V6_AUTH_SEALER_STATIC_AUDIT",
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
            root, AUTH_RELATIVE, label="published V6 authorization"
        )
        _require(
            observed_identity == expected_published_identity,
            "published authorization identity changed after hard-link verification",
        )
        published_identity = observed_identity
        require_postpublication_control_state(root)
        _require(
            load_json_control(destination, label="published V6 authorization")
            == authorization,
            "published authorization payload mismatch",
        )
        try:
            strict_published = auditor._v5_load_strict_json(
                destination, label="published V6 authorization"
            )
        except Exception as exc:
            raise PostrunAuditSealError(
                f"published V6 authorization strict JSON rejection: {exc}"
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
        "status": "PASS_V6_AUTHORIZATION_SEALED_PENDING_INDEPENDENT_REVIEW_AND_GO",
        "artifact_type": "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDIT_V6_AUTH_SEAL_REPORT",
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
            "status": "FAIL_V6_POSTRUN_AUDIT_AUTHORIZATION_POSTPUBLICATION",
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
            "status": "FAIL_V6_POSTRUN_AUDIT_AUTHORIZATION_SEAL",
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

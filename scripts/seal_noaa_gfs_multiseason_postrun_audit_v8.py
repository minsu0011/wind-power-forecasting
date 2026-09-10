#!/usr/bin/env python
"""Seal the append-only V8 postrun-audit authorization.

This publisher is deliberately narrower than the auditor it authorizes.  It
hashes immutable controls and eight bound code identities, validates recovered
files by metadata only, compiles the four executing V8 sources, binds the
original full V7 preseal-suite evidence, runs the four current applicable test
selections in isolated network-denied subprocesses, and publishes exactly one
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


AUDITOR_SOURCE_PATH = REPO / "scripts" / "audit_noaa_gfs_multiseason_raw_postrun_v8.py"


def _require_preimport_cache_runtime() -> None:
    if not (
        os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
        and sys.dont_write_bytecode is True
        and "PYTHONPYCACHEPREFIX" not in os.environ
        and sys.pycache_prefix is None
    ):
        raise RuntimeError(
            "V8 sealer import requires -B, PYTHONDONTWRITEBYTECODE=1, "
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
            "V8 auditor matching bytecode/cache competitor exists: "
            + ", ".join(str(candidate) for candidate in matches)
        )


def _require_bound_bytecode_absent(
    bound: Mapping[str, Mapping[str, Any]],
) -> None:
    required = {
        "superseded_v7_auditor", "superseded_v7_auditor_test",
        "superseded_v7_sealer", "superseded_v7_sealer_test",
        "v8_auditor", "v8_auditor_test", "v8_sealer", "v8_sealer_test",
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
            "V8 auditor same-stem source-origin competitor exists: "
            + ", ".join(str(candidate) for candidate in competitors)
        )


def _preimport_auditor_source_guard(path: Path) -> None:
    """Reject top-level I/O/network routes before importing the moving V8 source."""

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
        raise RuntimeError("V8 auditor has a forbidden import-time route")
    allowed_top_level = {
        ast.Expr, ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign,
        ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.If,
    }
    if any(type(node) not in allowed_top_level for node in tree.body):
        raise RuntimeError("V8 auditor has an unexpected top-level statement")
    for node in tree.body:
        if isinstance(node, ast.Expr) and not isinstance(node.value, ast.Constant):
            raise RuntimeError("V8 auditor has a top-level expression side effect")
        if isinstance(node, ast.If):
            if (
                not isinstance(node.test, ast.Compare)
                or ast.unparse(node.test) != "__name__ == '__main__'"
                or node.orelse
            ):
                raise RuntimeError("V8 auditor has an unexpected top-level branch")
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            for call in (item for item in ast.walk(node) if isinstance(item, ast.Call)):
                function = ast.unparse(call.func)
                allowed = (
                    function in {"Path", "tuple", "str", "AuditContract", "GroupContract", "re.compile"}
                    or function.endswith((".resolve", ".replace", ".strip"))
                )
                if not allowed:
                    raise RuntimeError(
                        f"V8 auditor has an unapproved import-time call: {function}"
                    )


_require_preimport_cache_runtime()
_require_auditor_bytecode_absent(Path(__file__).resolve())
_require_auditor_bytecode_absent(AUDITOR_SOURCE_PATH)
_require_auditor_source_competitors_absent(AUDITOR_SOURCE_PATH)
_preimport_auditor_source_guard(AUDITOR_SOURCE_PATH)

from scripts import audit_noaa_gfs_multiseason_raw_postrun_v8 as auditor

if not (
    auditor.__name__ == "scripts.audit_noaa_gfs_multiseason_raw_postrun_v8"
    and auditor.__package__ == "scripts"
    and auditor.__spec__ is not None
    and auditor.__spec__.name == auditor.__name__
    and Path(auditor.__file__).resolve() == AUDITOR_SOURCE_PATH.resolve()
):
    raise RuntimeError("V8 auditor import source origin mismatch")


ROOT_DEFAULT = auditor.ROOT_DEFAULT
CANONICAL_MODULE_ENTRYPOINT = (
    "scripts.seal_noaa_gfs_multiseason_postrun_audit_v8"
)
AUTH_RELATIVE = auditor.V8_AUTH_RELATIVE
REVIEW_RELATIVE = auditor.V8_REVIEW_RELATIVE
GO_RELATIVE = auditor.V8_GO_RELATIVE
SEALER_TEST = REPO / "tests" / "test_seal_noaa_gfs_multiseason_postrun_audit_v8.py"
FROZEN_V7_SOURCE_IDENTITIES = {
    "superseded_v7_auditor": {
        "path": str(
            (REPO / "scripts" / "audit_noaa_gfs_multiseason_raw_postrun_v7.py").resolve()
        ),
        "size_bytes": 614_349,
        "sha256": "02aa1da05105380c8b47d4e820873d2ec5734b49ea99d3124da37e22d73a6f20",
    },
    "superseded_v7_auditor_test": {
        "path": str(
            (REPO / "tests" / "test_noaa_gfs_multiseason_raw_postrun_v7.py").resolve()
        ),
        "size_bytes": 321_992,
        "sha256": "92992d7aadc5c07f74832c059d4689f94240bc2b0934408c4c9d131e78b4a405",
    },
    "superseded_v7_sealer": {
        "path": str(
            (REPO / "scripts" / "seal_noaa_gfs_multiseason_postrun_audit_v7.py").resolve()
        ),
        "size_bytes": 111_724,
        "sha256": "a77a159b48ffb589fbcb77aa21089560195c17d130e714774ba0ef3062142bf0",
    },
    "superseded_v7_sealer_test": {
        "path": str(
            (REPO / "tests" / "test_seal_noaa_gfs_multiseason_postrun_audit_v7.py").resolve()
        ),
        "size_bytes": 72_796,
        "sha256": "d1b677c500f1e477d85d4b81b5312cdea1f75120c72ae2292968ff662dd799ab",
    },
}
EXPECTED_V8_AUDITOR_IDENTITY = {
    "path": str(AUDITOR_SOURCE_PATH.resolve()),
    "size_bytes": 716_167,
    "sha256": "635f45d8024216470da590ce47d690b63fdadd8a8e6f805db0fc84fd35e906e1",
}
EXPECTED_V8_AUDITOR_TEST_IDENTITY = {
    "path": str(
        (REPO / "tests" / "test_noaa_gfs_multiseason_raw_postrun_v8.py").resolve()
    ),
    "size_bytes": 380_982,
    "sha256": "77b6ab8381159fcae52be1e79984e325cb33272f399f0393012d06af8082a5b6",
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
V7_TIMESTAMP_PRECISION_FALSE_REJECT_INCIDENT_IDENTITY = {
    "path": (
        "incidents/"
        "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_V6_"
        "AUTH_SEALER_TIMESTAMP_PRECISION_FALSE_REJECT_V1.json"
    ),
    "size_bytes": 11_799,
    "sha256": "722f1540cf73ed78aeff25ea3a2ad9300e40aa13fcc068f998898251e68ffd19",
}
V7_TIMESTAMP_PRECISION_INCIDENT_CANONICAL_BYTES = 9_607
V7_TIMESTAMP_PRECISION_INCIDENT_CANONICAL_SHA256 = (
    "ba20996218170b773cde52e99acafaa7e4b426d679c87c6fbeaafb31a6813113"
)
V7_TIMESTAMP_PRECISION_POSTFAILURE_CANONICAL_BYTES = 1_825
V7_TIMESTAMP_PRECISION_POSTFAILURE_CANONICAL_SHA256 = (
    "08b5d2c4bf653243deae63ed18067f9e098edb16b972a66ce121af88ee337837"
)
V8_PRIMARY_INCIDENT_IDENTITY = {
    "path": (
        "incidents/"
        "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_V7_IMMUTABLE_RECOVERY_"
        "PYTEST_DURATION_SUFFIX_FALSE_REJECT_V1.json"
    ),
    "size_bytes": 10_834,
    "sha256": "5ce616c66ff1eaed2d18e97296de1b45a7ed9ce8280f6253bc1acc05beed91db",
}
V8_PRIMARY_INCIDENT_CANONICAL_BYTES = 9_252
V8_PRIMARY_INCIDENT_CANONICAL_SHA256 = (
    "7919232ce35a73aec120ffe5f8dc99052e265f2a8f7409f855e61eec0ed5b297"
)
V8_CORRECTION_RECORD_IDENTITY = {
    "path": (
        "incidents/"
        "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_V7_PYTEST_DURATION_SUFFIX_"
        "AFTERSTATE_COMMITMENT_CORRECTION_V1.json"
    ),
    "size_bytes": 4_221,
    "sha256": "bd6a1e07274c99908e0c9bf27e8061f575371a6dd7f892b5a85b40468b3b0a89",
}
V8_CORRECTION_RECORD_CANONICAL_BYTES = 3_715
V8_CORRECTION_RECORD_CANONICAL_SHA256 = (
    "e1c7a417516bbb1469e86bb69d0d3772af76c4615b52e0dc24de435bfec4f9c2"
)
V8_CORRECTED_LOGICAL_PRIMARY_IDENTITY = {
    "canonical_size_bytes": 9_287,
    "canonical_sha256": (
        "3e9b374cdcf2f7baad366b69d34256ae694f97a299a89e3cafc285db737ecef8"
    ),
}
V8_LIVE_FAILURE_STATE_CANONICAL_BYTES = 1_192
V8_LIVE_FAILURE_STATE_CANONICAL_SHA256 = (
    "125477d4f619d8092a5a992d58b32a397fc187bb10f8d0d971a2ef71c8917119"
)
V7_AUTHORITY_BASE = {
    "authorization": {
        "path": "prereg/decode_recovery_postrun_audit_authorization_v7.json",
        "size_bytes": 26_074,
        "sha256": "136ac1c9878cc907ad652833d72dbc1967f8a1606cf7490c7727cf2e62c8458e",
    },
    "independent_review": {
        "path": (
            "independent_redteam/"
            "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_REVIEW_V7.json"
        ),
        "size_bytes": 8_687,
        "sha256": "e699abc43b9551844f5013cd81fbb698884a855db27583c0832847b64d10b4cc",
    },
    "independent_go": {
        "path": (
            "independent_redteam/"
            "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_GO_V7.json"
        ),
        "size_bytes": 7_079,
        "sha256": "c7bb3118df4237ecbd872a9b59766d32403f5c4160a55bae557e8305d021a1e3",
    },
    "audit_attempt_id": "postrun_audit_v7__20260811T053854057056Z",
    "recovery_attempt_id": "decode_recovery_v2__20260810T211017500046Z",
    "authorization_status": "AUTHORIZED_PENDING_INDEPENDENT_REVIEW_AND_GO_V7",
    "review_status": "PASS_PENDING_INDEPENDENT_GO_V7",
    "go_status": "GO_V7_READ_ONLY_NETWORK_ZERO_FULL_OFFLINE_REPLAY",
    "authorization_payload_canonical_sha256": (
        "00ef914127c84519dcb673d562bc6ff01315ded136328eeb388344b2e25ba27f"
    ),
    "review_payload_canonical_sha256": (
        "3684d84beae8ac7b6fb49b617678c0af5acf4ba2af22a73bb505e4a51dc040ad"
    ),
    "go_payload_canonical_sha256": (
        "8352320a53ec4a72df2d022a9079b30ff97186409d48143ac97b47e33a6989a2"
    ),
    "authorization_test_evidence_canonical_sha256": (
        "d86248d24b3af7f7b286d1d85ad56a4ccc0a449a0a493495c21193d6b16bf647"
    ),
    "recovered_afterstate_commitment_canonical_sha256": (
        "34be9415ded25118af85185f9bc6b15ca418720320d1e5c257d387e247858698"
    ),
}
V2_PREFLIGHT_IDENTITY = {
    "path": "prelaunch/DECODE_RECOVERY_REAL_PROCESS_PREFLIGHT_V2.json",
    "size_bytes": 38_972,
    "sha256": "5c9b79845c03015f85647833b68603cbed187d3dc09f63c7d214a379b5940481",
}
V8_IMMUTABLE_V2_PREFLIGHT_EVIDENCE = {
    "preflight": dict(V2_PREFLIGHT_IDENTITY),
    "preflight_payload_canonical_size_bytes": 33_445,
    "preflight_payload_canonical_sha256": (
        "5b6fb0976cd1682d74b35319d7224895e5d37acd99d3a624b7cb0c6211a28e0a"
    ),
    "test_evidence_canonical_size_bytes": 18_759,
    "test_evidence_canonical_sha256": (
        "b2d8cfdf1d2e1caa146c56ecefe331cab7670c969bd34370e9c39a060243fecb"
    ),
    "pytest_run_count": 6,
    "test_file_summary_pairs_canonical_size_bytes": 605,
    "test_file_summary_pairs_canonical_sha256": (
        "e20a8920edefbe38aebf66801f998af1fc6d44f591f8b92a8046ff78aa2c2803"
    ),
    "ordered_summaries_canonical_size_bytes": 149,
    "ordered_summaries_canonical_sha256": (
        "4ff61d69b17fc71bb64e59d6bde5c281828424c92055ffdfb2410c247a127f28"
    ),
    "aggregate_pytest_summary_canonical_size_bytes": 453,
    "aggregate_pytest_summary_canonical_sha256": (
        "a335063641a4385bac815f66d5d8338de1fc3acf22d278b1c2edaee4d4b84eda"
    ),
    "long_duration_run_index": 5,
    "long_duration_summary": "85 passed in 65.33s (0:01:05)",
    "validation_before_schema_parquet_data_and_replay": True,
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
    "v8_auditor_tests": "tests/test_noaa_gfs_multiseason_raw_postrun_v8.py",
    "v8_sealer_tests": "tests/test_seal_noaa_gfs_multiseason_postrun_audit_v8.py",
    "frozen_v7_auditor_tests": "tests/test_noaa_gfs_multiseason_raw_postrun_v7.py",
    "frozen_v7_sealer_tests": "tests/test_seal_noaa_gfs_multiseason_postrun_audit_v7.py",
}
V8_FROZEN_V7_TEST_DESELECTS = {
    "tests/test_noaa_gfs_multiseason_raw_postrun_v7.py": (
        "tests/test_noaa_gfs_multiseason_raw_postrun_v7.py::"
        "test_frozen_v6_reproduces_exact_sealer_timestamp_precision_false_reject",
        "tests/test_noaa_gfs_multiseason_raw_postrun_v7.py::"
        "test_v7_actual_9e74_collect_build_and_self_validator_passes",
    ),
    "tests/test_seal_noaa_gfs_multiseason_postrun_audit_v7.py": (
        "tests/test_seal_noaa_gfs_multiseason_postrun_audit_v7.py::"
        "test_actual_root_collect_build_and_self_validator_passes_read_only",
    ),
}
BOUND_SOURCE_PATHS = {
    "superseded_v7_auditor": Path(FROZEN_V7_SOURCE_IDENTITIES["superseded_v7_auditor"]["path"]),
    "superseded_v7_auditor_test": Path(FROZEN_V7_SOURCE_IDENTITIES["superseded_v7_auditor_test"]["path"]),
    "superseded_v7_sealer": Path(FROZEN_V7_SOURCE_IDENTITIES["superseded_v7_sealer"]["path"]),
    "superseded_v7_sealer_test": Path(FROZEN_V7_SOURCE_IDENTITIES["superseded_v7_sealer_test"]["path"]),
    "v8_auditor": Path(auditor.__file__).resolve(),
    "v8_auditor_test": auditor.AUDITOR_TEST.resolve(),
    "v8_sealer": Path(__file__).resolve(),
    "v8_sealer_test": SEALER_TEST.resolve(),
}
POSTRUN_REPORT_CANDIDATES = auditor.V8_POSTRUN_REPORT_CANDIDATES
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
    auditor.V6_AUTH_RELATIVE,
    auditor.V6_REVIEW_RELATIVE,
    auditor.V6_GO_RELATIVE,
    auditor.V7_AUTH_RELATIVE,
    auditor.V7_REVIEW_RELATIVE,
    auditor.V7_GO_RELATIVE,
    AUTH_RELATIVE, REVIEW_RELATIVE, GO_RELATIVE,
    auditor.V2_FALSE_REJECT_INCIDENT_RELATIVE,
    auditor.V4_PATH_MISPUBLISH_INCIDENT_RELATIVE,
    auditor.V5_TRANSPORT_INCIDENT_RELATIVE,
    auditor.V5_TRANSPORT_CORRECTION_RELATIVE,
    auditor.V6_HISTORICAL_FALSE_REJECT_INCIDENT_RELATIVE,
    auditor.V7_TIMESTAMP_FALSE_REJECT_INCIDENT_RELATIVE,
    auditor.V8_DURATION_SUFFIX_INCIDENT_RELATIVE,
    auditor.V8_DURATION_SUFFIX_CORRECTION_RELATIVE,
)
BOUND_SOURCE_ROLES = {
    "superseded_v7_auditor", "superseded_v7_auditor_test",
    "superseded_v7_sealer", "superseded_v7_sealer_test",
    "v8_auditor", "v8_auditor_test", "v8_sealer", "v8_sealer_test",
}
BOUND_IDENTITY_ROLES = {
    "v7_authorization",
    "v7_independent_review",
    "v7_independent_go",
    "primary_duration_suffix_incident",
    "afterstate_commitment_correction",
    "immutable_v2_preflight",
    *BOUND_SOURCE_ROLES,
}
SOURCE_ORIGIN_TRUST_MODEL = {
    "model": "NON_ADVERSARIAL_NO_CONCURRENT_FILESYSTEM_SWAP",
    "preimport_source_ast_and_competitor_scan": True,
    "pretest_posttest_and_prepublish_eight_source_rehash_and_pyc_scan": True,
    "test_evidence_exact_fourteen_bound_identities": True,
    "source_compile_exact_executing_v8_four": True,
    "adversarial_concurrent_swap_resistance_claimed": False,
}


class PostrunAuditSealError(RuntimeError):
    """The immutable V8 authorization could not be sealed safely."""


class PostrunAuditPostPublicationError(PostrunAuditSealError):
    """AUTH was created by this invocation before a later fail-closed check."""

    def __init__(self, message: str, authorization_identity: Mapping[str, Any]):
        super().__init__(message)
        self.authorization_identity = dict(authorization_identity)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PostrunAuditSealError(message)


def _exact_utc_z_100ns_ticks(value: Any, *, label: str) -> int:
    """Parse exact UTC-Z text with 0..7 fractional digits into integer ticks."""

    _require(isinstance(value, str), f"{label} timestamp is absent")
    match = re.fullmatch(
        r"([0-9]{4})-([0-9]{2})-([0-9]{2})T"
        r"([0-9]{2}):([0-9]{2}):([0-9]{2})(?:\.([0-9]{1,7}))?Z",
        value,
    )
    _require(match is not None, f"{label} timestamp is invalid")
    parts = tuple(int(component) for component in match.groups()[:6])
    try:
        whole = datetime(*parts, tzinfo=timezone.utc)
    except ValueError as exc:
        raise PostrunAuditSealError(f"{label} timestamp is invalid") from exc
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = whole - epoch
    whole_seconds = delta.days * 86_400 + delta.seconds
    fraction = match.group(7) or ""
    fractional_ticks = int(fraction.ljust(7, "0") or "0")
    return whole_seconds * 10_000_000 + fractional_ticks


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
        "seal_noaa_gfs_multiseason_postrun_audit_v8` entrypoint",
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
    raise _NetworkDenied("network access is forbidden during V8 authorization sealing")


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
        "authorization": root_path(root, AUTH_RELATIVE, label="V8 authorization"),
        "independent_review": root_path(root, REVIEW_RELATIVE, label="V8 review"),
        "independent_go": root_path(root, GO_RELATIVE, label="V8 GO"),
    }


def required_audit_command(root: Path) -> list[str]:
    return [
        str(Path(sys.executable).resolve()),
        "-B",
        "-m",
        "scripts.audit_noaa_gfs_multiseason_raw_postrun_v8",
        "--root",
        str(root.resolve()),
        "--authorization",
        str((root / AUTH_RELATIVE).resolve()),
        "--independent-go",
        str((root / GO_RELATIVE).resolve()),
    ]


def _v8_test_command(relative: str) -> list[str]:
    command = [
        str(Path(sys.executable).resolve()), "-B", "-c",
        auditor.V8_PYTEST_NETWORK_GUARD_SOURCE,
        "-q", "-p", "no:cacheprovider", relative,
    ]
    for deselect in V8_FROZEN_V7_TEST_DESELECTS.get(relative, ()):
        command.append(f"--deselect={deselect}")
    return command


def control_temporary_files(root: Path) -> list[str]:
    matches: list[str] = []
    for relative in CONTROL_TEMP_RELATIVES:
        path = root_path(root, relative, label=f"V8 control temp destination {relative}")
        parent = path.parent
        _require(parent.is_dir() and not _linklike(parent), "V8 control parent invalid")
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
        Path(auditor.V7_AUTH_RELATIVE).name,
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
            Path(auditor.V7_REVIEW_RELATIVE).name,
            Path(auditor.V7_GO_RELATIVE).name,
        },
        "incidents": {
            Path(auditor.V2_FALSE_REJECT_INCIDENT_RELATIVE).name,
            Path(auditor.V4_PATH_MISPUBLISH_INCIDENT_RELATIVE).name,
            Path(auditor.V5_TRANSPORT_INCIDENT_RELATIVE).name,
            Path(auditor.V5_TRANSPORT_CORRECTION_RELATIVE).name,
            Path(auditor.V6_HISTORICAL_FALSE_REJECT_INCIDENT_RELATIVE).name,
            Path(auditor.V7_TIMESTAMP_FALSE_REJECT_INCIDENT_RELATIVE).name,
            Path(V8_PRIMARY_INCIDENT_IDENTITY["path"]).name,
            Path(V8_CORRECTION_RECORD_IDENTITY["path"]).name,
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
        parent = root_path(root, relative, label=f"V8 namespace {relative}")
        _require(
            _lexists(parent) and parent.is_dir() and not _linklike(parent),
            f"V8 control namespace parent invalid: {relative}",
        )
        matched: list[str] = []
        for entry in parent.iterdir():
            if not entry.name.casefold().lstrip(".").startswith(prefix):
                continue
            _require(
                _lexists(entry) and entry.is_file() and not _linklike(entry),
                f"V8 control namespace contains non-regular/link entry: {entry}",
            )
            matched.append(entry.name)
        _require(
            set(matched) == expected[relative]
            and len({name.casefold() for name in matched}) == len(matched),
            f"V8 control namespace inventory mismatch: {relative}",
        )
        result[relative] = sorted(matched, key=str.casefold)
    return result


def require_prepublication_control_state(root: Path) -> dict[str, Any]:
    paths = control_paths(root)
    _require(
        all(not _lexists(path) for path in paths.values()),
        "V8 authorization/review/GO already exists",
    )
    _require(control_temporary_files(root) == [], "V8 control temporary sibling exists")
    authorization_present = _lexists(root / AUTH_RELATIVE)
    namespace = validate_control_namespace(
        root, authorization_present=authorization_present
    )
    if authorization_present:
        namespace = {key: list(value) for key, value in namespace.items()}
        namespace["prereg"].remove(Path(AUTH_RELATIVE).name)
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
        "published V8 authorization is absent, non-regular or link-like",
    )
    _require(
        not _lexists(paths["independent_review"])
        and not _lexists(paths["independent_go"]),
        "V8 sealer observed an independent review/GO",
    )
    _require(control_temporary_files(root) == [], "V8 control temporary sibling exists")
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
    _require(Path(auditor.V8_SEALER).resolve() == Path(__file__).resolve(), "auditor V8 sealer path mismatch")
    _require(Path(auditor.V8_SEALER_TEST).resolve() == SEALER_TEST.resolve(), "auditor V8 sealer-test path mismatch")

    bound = {
        role: strict_external_identity(path, label=role)
        for role, path in BOUND_SOURCE_PATHS.items()
    }
    _require(set(bound) == BOUND_SOURCE_ROLES, "bound source role schema mismatch")
    for role, expected in FROZEN_V7_SOURCE_IDENTITIES.items():
        _require(bound[role] == expected, f"frozen V7 source identity mismatch: {role}")
    _require(
        bound["v8_auditor"] == EXPECTED_V8_AUDITOR_IDENTITY
        and bound["v8_auditor_test"] == EXPECTED_V8_AUDITOR_TEST_IDENTITY,
        "settled V8 auditor/test identity mismatch",
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


def _load_and_validate_v7_timestamp_precision_incident(
    root: Path,
    declared_incident: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently bind 722f and its immutable V6 failure boundary."""

    _require(
        isinstance(declared_incident, Mapping)
        and dict(declared_incident)
        == V7_TIMESTAMP_PRECISION_FALSE_REJECT_INCIDENT_IDENTITY,
        "V7 timestamp-precision incident declaration mismatch",
    )
    actual = strict_root_identity(
        root,
        V7_TIMESTAMP_PRECISION_FALSE_REJECT_INCIDENT_IDENTITY["path"],
        label="V7 timestamp-precision false-reject incident",
    )
    _require(
        actual == V7_TIMESTAMP_PRECISION_FALSE_REJECT_INCIDENT_IDENTITY,
        "V7 timestamp-precision incident identity mismatch",
    )
    payload = load_json_control(
        root_path(
            root,
            V7_TIMESTAMP_PRECISION_FALSE_REJECT_INCIDENT_IDENTITY["path"],
            label="V7 timestamp-precision false-reject incident",
        ),
        label="V7 timestamp-precision false-reject incident",
    )
    encoded = canonical_bytes(payload)
    _require(
        len(encoded) == V7_TIMESTAMP_PRECISION_INCIDENT_CANONICAL_BYTES
        and hashlib.sha256(encoded).hexdigest()
        == V7_TIMESTAMP_PRECISION_INCIDENT_CANONICAL_SHA256,
        "V7 timestamp-precision incident canonical identity mismatch",
    )
    _require(
        set(payload)
        == {
            "artifact_type", "authority_scope", "created_utc",
            "execution_boundary", "failed_execution", "historical_timestamp",
            "immutable_v6_candidate", "postfailure_state", "prohibitions",
            "required_v7_supersession", "root_cause", "schema_version", "status",
        }
        and payload.get("schema_version") == 1
        and payload.get("artifact_type")
        == "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_V6_AUTH_SEALER_TIMESTAMP_PRECISION_FALSE_REJECT_INCIDENT"
        and payload.get("status")
        == "SEALED_V6_AUTH_SEALER_TIMESTAMP_PRECISION_FALSE_REJECT_BEFORE_AUTH_PUBLICATION_V7_SUPERSESSION_REQUIRED",
        "V7 timestamp-precision incident schema/header mismatch",
    )
    postfailure = payload.get("postfailure_state")
    _require(isinstance(postfailure, Mapping), "V7 incident postfailure state invalid")
    postfailure_bytes = canonical_bytes(postfailure)
    _require(
        len(postfailure_bytes) == V7_TIMESTAMP_PRECISION_POSTFAILURE_CANONICAL_BYTES
        and hashlib.sha256(postfailure_bytes).hexdigest()
        == V7_TIMESTAMP_PRECISION_POSTFAILURE_CANONICAL_SHA256,
        "V7 incident postfailure-state canonical identity mismatch",
    )
    immutable_inputs = postfailure.get("immutable_input_identities")
    _require(
        immutable_inputs
        == {
            "direct_file_failure_incident": HISTORICAL_DIRECT_FILE_FAILURE_INCIDENT_IDENTITY,
            "v5_authorization": V5_AUTHORITY_BASE["authorization"],
            "v5_historical_identity_false_reject_incident": (
                V6_HISTORICAL_FALSE_REJECT_INCIDENT_IDENTITY
            ),
            "v5_independent_go": V5_AUTHORITY_BASE["independent_go"],
            "v5_independent_review": V5_AUTHORITY_BASE["independent_review"],
        }
        and postfailure.get("v6_control_file_count") == 0
        and postfailure.get("v6_temporary_files") == [],
        "V7 incident immutable-input/zero-control state mismatch",
    )
    historical = payload.get("historical_timestamp")
    _require(
        isinstance(historical, Mapping)
        and historical.get("incident")
        == HISTORICAL_DIRECT_FILE_FAILURE_INCIDENT_IDENTITY
        and historical.get("value") == "2026-08-10T18:45:00.5807528Z"
        and historical.get("fractional_second_digits") == 7
        and historical.get("precision_unit") == "100_NANOSECONDS"
        and historical.get("rfc3339_utc") is True
        and historical.get("immutable") is True
        and historical.get("chronology_valid") is True,
        "V7 incident historical raw timestamp mismatch",
    )
    created_raw = payload.get("created_utc")
    historical_raw = historical.get("value")
    created_ticks = _exact_utc_z_100ns_ticks(
        created_raw, label="V7 timestamp incident created_utc"
    )
    historical_ticks = _exact_utc_z_100ns_ticks(
        historical_raw, label="historical 9e74 created_utc"
    )
    _require(
        historical_ticks < created_ticks,
        "V7 timestamp incident internal chronology mismatch",
    )
    return {
        "identity": actual,
        "payload": payload,
        "created_utc": created_raw,
        "created_ticks_100ns": created_ticks,
        "historical_created_utc": historical_raw,
        "historical_ticks_100ns": historical_ticks,
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
    _require(temporary == [], "V8 control temporary sibling exists")
    _require(reports == [], "postrun audit output file exists despite stdout-only contract")
    authorization_present = _lexists(root / AUTH_RELATIVE)
    namespace = validate_control_namespace(
        root, authorization_present=authorization_present
    )
    if authorization_present:
        namespace = {key: list(value) for key, value in namespace.items()}
        namespace["prereg"].remove(Path(AUTH_RELATIVE).name)
    namespace_bytes = canonical_bytes(namespace)
    _require(
        len(namespace_bytes) == 1_388
        and hashlib.sha256(namespace_bytes).hexdigest()
        == "c16a239873847da42d138b5cab941c3eb81e5141aaebdabb2695140a3f90347c",
        "V8 preauthorization namespace canonical commitment mismatch",
    )
    snapshot = {
        "v7_live_failure_state_canonical_sha256": (
            V8_LIVE_FAILURE_STATE_CANONICAL_SHA256
        ),
        "recovered_afterstate_commitment_canonical_sha256": (
            auditor.V5_RECOVERED_AFTERSTATE_COMMITMENT_CANONICAL_SHA256
        ),
        "preauthorization_control_namespace": {
            "counts": {
                "prereg": 5,
                "independent_redteam": 6,
                "incidents": 8,
            },
            "canonical_size_bytes": 1_388,
            "canonical_sha256": (
                "c16a239873847da42d138b5cab941c3eb81e5141aaebdabb2695140a3f90347c"
            ),
        },
        "active_locks": active,
        "v8_control_temporary_files": {
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
        set(snapshot) == auditor.V8_ZERO_SNAPSHOT_KEYS,
        "V8 zero-snapshot schema mismatch",
    )
    _require(
        snapshot["v8_control_temporary_files"]
        == {
            "checked_destination_relative_paths": list(CONTROL_TEMP_RELATIVES),
            "matching_temporary_files": [],
        },
        "V8 zero-snapshot temporary-destination declaration mismatch",
    )
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


def _load_and_validate_v7_authority_base(
    root: Path, declared: Mapping[str, Any]
) -> dict[str, Any]:
    """Strictly bind the complete frozen V7 AUTH/REVIEW/GO authority chain."""

    _require(dict(declared) == V7_AUTHORITY_BASE, "V7 authority-base declaration mismatch")
    loaded: dict[str, dict[str, Any]] = {}
    for role in ("authorization", "independent_review", "independent_go"):
        record = V7_AUTHORITY_BASE[role]
        _require(
            strict_root_identity(root, str(record["path"]), label=f"V7 {role}")
            == record,
            f"V7 authority-base physical identity mismatch: {role}",
        )
        loaded[role] = load_json_control(
            root_path(root, str(record["path"]), label=f"V7 {role}"),
            label=f"V7 {role}",
        )
    authorization = loaded["authorization"]
    review = loaded["independent_review"]
    go = loaded["independent_go"]
    _require(set(authorization) == auditor.V7_AUTH_KEYS, "frozen V7 AUTH schema mismatch")
    _require(set(review) == auditor.V7_REVIEW_KEYS, "frozen V7 REVIEW schema mismatch")
    _require(set(go) == auditor.V7_GO_KEYS, "frozen V7 GO schema mismatch")
    for role, payload, field in (
        ("authorization", authorization, "authorization_payload_canonical_sha256"),
        ("review", review, "review_payload_canonical_sha256"),
        ("go", go, "go_payload_canonical_sha256"),
    ):
        _require(
            hashlib.sha256(canonical_bytes(payload)).hexdigest()
            == V7_AUTHORITY_BASE[field],
            f"frozen V7 {role} canonical payload mismatch",
        )
    created = {
        "authorization": _exact_utc_z_100ns_ticks(
            authorization.get("created_utc"), label="V7 authorization"
        ),
        "review": _exact_utc_z_100ns_ticks(review.get("created_utc"), label="V7 review"),
        "go": _exact_utc_z_100ns_ticks(go.get("created_utc"), label="V7 GO"),
    }
    attempt = V7_AUTHORITY_BASE["audit_attempt_id"]
    _require(
        authorization.get("schema_version") == 7
        and authorization.get("artifact_type")
        == "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDIT_AUTHORIZATION_V7"
        and authorization.get("status") == V7_AUTHORITY_BASE["authorization_status"]
        and authorization.get("audit_attempt_id") == attempt
        and authorization.get("recovery_attempt_id")
        == V7_AUTHORITY_BASE["recovery_attempt_id"]
        and review.get("schema_version") == 7
        and review.get("artifact_type")
        == "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_REVIEW_V7"
        and review.get("status") == V7_AUTHORITY_BASE["review_status"]
        and review.get("verdict") == "GO_RECOMMENDED"
        and review.get("audit_attempt_id") == attempt
        and go.get("schema_version") == 7
        and go.get("artifact_type")
        == "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_GO_V7"
        and go.get("status") == V7_AUTHORITY_BASE["go_status"]
        and go.get("audit_attempt_id") == attempt
        and go.get("recovery_attempt_id") == V7_AUTHORITY_BASE["recovery_attempt_id"],
        "frozen V7 authority header/status binding mismatch",
    )
    _require(
        created["authorization"] < created["review"] < created["go"]
        and review.get("authorization") == V7_AUTHORITY_BASE["authorization"]
        and go.get("authorization") == V7_AUTHORITY_BASE["authorization"]
        and go.get("independent_review") == V7_AUTHORITY_BASE["independent_review"],
        "frozen V7 authority chronology/cross-link mismatch",
    )
    evidence = authorization.get("test_evidence")
    commitment = authorization.get("recovered_afterstate_commitment")
    _require(
        isinstance(evidence, Mapping)
        and hashlib.sha256(canonical_bytes(evidence)).hexdigest()
        == V7_AUTHORITY_BASE["authorization_test_evidence_canonical_sha256"]
        and isinstance(commitment, Mapping)
        and len(canonical_bytes(commitment)) == 470
        and hashlib.sha256(canonical_bytes(commitment)).hexdigest()
        == V7_AUTHORITY_BASE["recovered_afterstate_commitment_canonical_sha256"]
        and commitment == auditor.V5_RECOVERED_AFTERSTATE_COMMITMENT,
        "frozen V7 evidence/afterstate commitment mismatch",
    )
    for role, expected in FROZEN_V7_SOURCE_IDENTITIES.items():
        payload_role = role.removeprefix("superseded_")
        _require(
            authorization.get(payload_role) == expected
            and review.get(payload_role) == expected
            and go.get(payload_role) == expected,
            f"frozen V7 source binding mismatch: {payload_role}",
        )
    return {
        "base": dict(V7_AUTHORITY_BASE),
        "authorization": authorization,
        "review": review,
        "go": go,
        "created_ticks": created,
        "recovered_afterstate_commitment": dict(commitment),
    }


def _validate_correction_record(correction: Mapping[str, Any]) -> None:
    """Validate the append-only correction before touching primary semantics."""

    _require(
        set(correction)
        == {
            "artifact_type", "authority_scope", "correction", "created_utc",
            "erroneous_incident", "independent_validation", "prohibitions",
            "required_v8_binding", "schema_version", "status",
        },
        "V8 correction top-level schema mismatch",
    )
    expected_erroneous = {
        "artifact_type": (
            "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_V7_IMMUTABLE_RECOVERY_"
            "PYTEST_DURATION_SUFFIX_FALSE_REJECT_INCIDENT"
        ),
        "canonical_payload": {
            "canonical_size_bytes": V8_PRIMARY_INCIDENT_CANONICAL_BYTES,
            "canonical_sha256": V8_PRIMARY_INCIDENT_CANONICAL_SHA256,
        },
        **V8_PRIMARY_INCIDENT_IDENTITY,
        "status": (
            "SEALED_V7_LIVE_POSTRUN_AUDITOR_PYTEST_DURATION_SUFFIX_FALSE_REJECT_"
            "AFTER_FULL_REPLAY_NO_REPORT_V8_SUPERSESSION_REQUIRED"
        ),
    }
    expected_correction = {
        "all_other_primary_incident_facts_unchanged": True,
        "authoritative_commitment": {
            "canonical_sha256": V7_AUTHORITY_BASE[
                "recovered_afterstate_commitment_canonical_sha256"
            ],
            "canonical_size_bytes": 470,
        },
        "authoritative_source": {
            "json_pointer": (
                "/recovered_afterstate_commitment/recovery_progress_file_count"
            ),
            **V7_AUTHORITY_BASE["authorization"],
            "value": 104,
        },
        "corrected_json_pointer": (
            "/postfailure_state/recovered_afterstate_commitment/"
            "recovery_progress_file_count"
        ),
        "corrected_logical_primary_incident": V8_CORRECTED_LOGICAL_PRIMARY_IDENTITY,
        "corrected_value": 104,
        "erroneous_commitment": {
            "canonical_sha256": (
                "c85da4f7311c03ff6adbb1efc631a370bd0dc0d5252bae21d82f9f952f968125"
            ),
            "canonical_size_bytes": 435,
        },
        "erroneous_json_pointer": (
            "/postfailure_state/recovered_afterstate_commitment/"
            "recovery_progress_file_count"
        ),
        "operation": "ADD_MISSING_FIELD",
        "this_is_the_only_authoritative_override": True,
    }
    _require(
        correction.get("schema_version") == 1
        and correction.get("artifact_type")
        == (
            "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_V7_IMMUTABLE_RECOVERY_"
            "PYTEST_DURATION_SUFFIX_FALSE_REJECT_AFTERSTATE_COMMITMENT_"
            "CORRECTION_V1"
        )
        and correction.get("status")
        == (
            "SEALED_V7_DURATION_SUFFIX_FALSE_REJECT_AFTERSTATE_COMMITMENT_"
            "CORRECTION_NO_V8_AUTHORITY_DUAL_BINDING_REQUIRED"
        )
        and correction.get("created_utc") == "2026-08-11T06:43:59.084809Z"
        and correction.get("erroneous_incident") == expected_erroneous
        and correction.get("correction") == expected_correction,
        "V8 correction identity/override semantics mismatch",
    )
    _require(
        correction.get("independent_validation")
        == {
            "authoritative_commitment_key_count": 10,
            "erroneous_commitment_key_count": 9,
            "exact_recursive_difference": {
                "after": 104,
                "before_present": False,
                "json_pointer": expected_correction["corrected_json_pointer"],
            },
            "exact_recursive_difference_count": 1,
            "extra_key_set": [],
            "missing_key_set": ["recovery_progress_file_count"],
            "root_recomputed_exact": True,
            "secondary_reviewer_recomputed_exact": True,
            "shared_value_mismatch_count": 0,
        }
        and correction.get("required_v8_binding")
        == {
            "authorization_review_and_go_must_bind_both": True,
            "correction_must_be_applied_before_semantic_use": True,
            "correction_relative_path": V8_CORRECTION_RECORD_IDENTITY["path"],
            "must_bind_correction_record_identity": True,
            "must_bind_primary_incident_identity": True,
            "primary_incident_alone_must_fail": True,
            "primary_incident_relative_path": V8_PRIMARY_INCIDENT_IDENTITY["path"],
            "status": "REQUIRED_NOT_YET_AUTHORIZED",
        },
        "V8 correction independent-validation/binding mismatch",
    )


def _load_and_validate_v8_live_failure_state(root: Path) -> dict[str, Any]:
    """Bind both records and apply the correction before semantic use."""

    for label, expected in (
        ("V8 primary incident", V8_PRIMARY_INCIDENT_IDENTITY),
        ("V8 correction record", V8_CORRECTION_RECORD_IDENTITY),
    ):
        _require(
            strict_root_identity(root, str(expected["path"]), label=label) == expected,
            f"{label} physical identity mismatch",
        )
    correction = load_json_control(
        root_path(root, str(V8_CORRECTION_RECORD_IDENTITY["path"]), label="V8 correction"),
        label="V8 correction",
    )
    _require(
        len(canonical_bytes(correction)) == V8_CORRECTION_RECORD_CANONICAL_BYTES
        and hashlib.sha256(canonical_bytes(correction)).hexdigest()
        == V8_CORRECTION_RECORD_CANONICAL_SHA256,
        "V8 correction canonical identity mismatch",
    )
    _validate_correction_record(correction)

    # Loading and hashing is not semantic use.  The validated correction is
    # applied before any primary payload field is interpreted.
    primary = load_json_control(
        root_path(root, str(V8_PRIMARY_INCIDENT_IDENTITY["path"]), label="V8 primary"),
        label="V8 primary",
    )
    _require(
        len(canonical_bytes(primary)) == V8_PRIMARY_INCIDENT_CANONICAL_BYTES
        and hashlib.sha256(canonical_bytes(primary)).hexdigest()
        == V8_PRIMARY_INCIDENT_CANONICAL_SHA256,
        "V8 primary incident canonical identity mismatch",
    )
    corrected = json.loads(json.dumps(primary, ensure_ascii=False))
    postfailure = corrected.get("postfailure_state")
    _require(isinstance(postfailure, dict), "corrected primary postfailure state absent")
    commitment = postfailure.get("recovered_afterstate_commitment")
    _require(
        isinstance(commitment, dict)
        and "recovery_progress_file_count" not in commitment,
        "primary incident alone did not reproduce the missing commitment key",
    )
    commitment["recovery_progress_file_count"] = 104
    corrected_bytes = canonical_bytes(corrected)
    _require(
        len(corrected_bytes) == V8_CORRECTED_LOGICAL_PRIMARY_IDENTITY[
            "canonical_size_bytes"
        ]
        and hashlib.sha256(corrected_bytes).hexdigest()
        == V8_CORRECTED_LOGICAL_PRIMARY_IDENTITY["canonical_sha256"],
        "corrected logical primary incident identity mismatch",
    )
    _require(
        set(corrected)
        == {
            "artifact_type", "authority_scope", "created_utc", "exact_mismatch_set",
            "failed_execution", "immutable_v7_chain", "postfailure_state",
            "prohibitions", "required_v8_supersession", "root_cause",
            "schema_version", "status", "test_gap",
        }
        and corrected.get("schema_version") == 1
        and corrected.get("artifact_type")
        == (
            "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_V7_IMMUTABLE_RECOVERY_"
            "PYTEST_DURATION_SUFFIX_FALSE_REJECT_INCIDENT"
        )
        and corrected.get("status")
        == (
            "SEALED_V7_LIVE_POSTRUN_AUDITOR_PYTEST_DURATION_SUFFIX_FALSE_REJECT_"
            "AFTER_FULL_REPLAY_NO_REPORT_V8_SUPERSESSION_REQUIRED"
        )
        and corrected.get("created_utc") == "2026-08-11T06:14:43.374633Z",
        "corrected primary header/schema mismatch",
    )
    expected_failure = {
        "primary_incident": dict(V8_PRIMARY_INCIDENT_IDENTITY),
        "correction_record": dict(V8_CORRECTION_RECORD_IDENTITY),
        "primary_incident_alone_must_fail": True,
        "correction_must_be_applied_before_semantic_use": True,
        "corrected_logical_primary_incident": dict(
            V8_CORRECTED_LOGICAL_PRIMARY_IDENTITY
        ),
        "failure_reason": (
            "immutable recovery pytest result invalid: "
            "tests/test_noaa_gfs_multiseason_raw_postrun_v2.py"
        ),
        "exit_code": 1,
        "wall_seconds": 266.4,
        "stdout_pretty_lf": {
            "size_bytes": 307,
            "sha256": (
                "1dde5ef1aef95e29fa04a1efb32f179bf8dfacd4ff99a1d7408b53e38ae844ac"
            ),
        },
        "full_offline_replay_returned_before_failure": True,
        "success_report_constructed": False,
        "persisted_report_files": 0,
        "audit_files_written": 0,
        "audit_network_requests_reported": 0,
        "v7_retry_authorized": False,
    }
    failed = corrected.get("failed_execution")
    _require(
        isinstance(failed, Mapping)
        and failed.get("stdout_payload", {}).get("reason")
        == expected_failure["failure_reason"]
        and failed.get("exit_code") == expected_failure["exit_code"]
        and failed.get("wall_seconds") == expected_failure["wall_seconds"]
        and failed.get("stdout_pretty_lf") == expected_failure["stdout_pretty_lf"]
        and failed.get("full_offline_replay_returned_before_failure") is True
        and failed.get("success_report_constructed") is False
        and failed.get("persisted_report_files") == 0
        and failed.get("audit_files_written") == 0
        and failed.get("audit_network_requests_reported") == 0
        and corrected.get("authority_scope", {}).get("v7_retry_authorized") is False,
        "corrected primary failure semantics mismatch",
    )
    corrected_commitment = postfailure["recovered_afterstate_commitment"]
    _require(
        corrected_commitment == auditor.V5_RECOVERED_AFTERSTATE_COMMITMENT
        and len(canonical_bytes(corrected_commitment)) == 470
        and hashlib.sha256(canonical_bytes(corrected_commitment)).hexdigest()
        == V7_AUTHORITY_BASE["recovered_afterstate_commitment_canonical_sha256"],
        "corrected primary afterstate commitment mismatch",
    )
    primary_ticks = _exact_utc_z_100ns_ticks(
        corrected["created_utc"], label="V8 primary incident"
    )
    correction_ticks = _exact_utc_z_100ns_ticks(
        correction["created_utc"], label="V8 correction"
    )
    _require(primary_ticks < correction_ticks, "V8 primary/correction chronology mismatch")
    _require(
        len(canonical_bytes(expected_failure)) == V8_LIVE_FAILURE_STATE_CANONICAL_BYTES
        and hashlib.sha256(canonical_bytes(expected_failure)).hexdigest()
        == V8_LIVE_FAILURE_STATE_CANONICAL_SHA256,
        "V8 compact live-failure state canonical mismatch",
    )
    return {
        "state": expected_failure,
        "primary_created_utc": corrected["created_utc"],
        "correction_created_utc": correction["created_utc"],
        "corrected_primary": corrected,
    }


def _parse_pytest_success_summary_local(value: Any, *, label: str) -> dict[str, Any]:
    """Independently parse pytest hundredths plus optional canonical HMS."""

    _require(isinstance(value, str), label)
    match = re.fullmatch(
        r"([0-9]+) passed(?:, ([0-9]+) skipped)? in "
        r"([0-9]+)(?:\.([0-9]+))?s"
        r"(?: \(([0-9]+):([0-5][0-9]):([0-5][0-9])\))?",
        value,
    )
    _require(match is not None, label)
    assert match is not None
    passed = int(match.group(1))
    skipped = int(match.group(2)) if match.group(2) is not None else 0
    whole = int(match.group(3))
    fraction_text = match.group(4)
    suffix_present = match.group(5) is not None
    duration_hundredths: int | None = None
    suffix_total: int | None = None
    _require(passed > 0, label)
    if suffix_present:
        _require(
            isinstance(fraction_text, str) and len(fraction_text) == 2,
            label,
        )
        duration_hundredths = whole * 100 + int(fraction_text)
        hours_text = str(match.group(5))
        _require(hours_text == str(int(hours_text)), label)
        suffix_total = (
            int(hours_text) * 3_600
            + int(str(match.group(6))) * 60
            + int(str(match.group(7)))
        )
        candidates = {duration_hundredths // 100}
        if duration_hundredths > 6_000 and duration_hundredths % 100 == 0:
            candidates.add(duration_hundredths // 100 - 1)
        _require(
            duration_hundredths >= 6_000
            and suffix_total >= 60
            and suffix_total in candidates
            and value.endswith(
                f" ({suffix_total // 3_600}:"
                f"{(suffix_total % 3_600) // 60:02d}:"
                f"{suffix_total % 60:02d})"
            ),
            label,
        )
        if duration_hundredths == 6_000:
            _require(suffix_total == 60, label)
    else:
        if fraction_text is None:
            duration_hundredths = whole * 100
        elif len(fraction_text) <= 2:
            duration_hundredths = whole * 100 + int(fraction_text.ljust(2, "0"))
    result = {
        "passed": passed,
        "skipped": skipped,
        "duration_hundredths": duration_hundredths,
        "duration_fraction_digits": len(fraction_text or ""),
        "suffix_present": suffix_present,
        "suffix_total_seconds": suffix_total,
    }
    try:
        independent = auditor._v8_parse_pytest_success_summary(value, label=label)
    except Exception as exc:
        raise PostrunAuditSealError(
            f"V8 auditor pytest-summary parser rejected locally valid input: {label}: {exc}"
        ) from exc
    _require(independent == result, f"V8 local/auditor pytest-summary parse mismatch: {label}")
    return result


def _validate_bound_test_run_summary_local(
    role: str, value: Any,
) -> dict[str, int]:
    """Validate V8 current-suite selection without widening the V2 parser."""

    _require(isinstance(value, str), f"V8 immutable pytest summary absent: {role}")
    match = re.fullmatch(
        r"([0-9]+) passed(?:, ([0-9]+) skipped)?"
        r"(?:, ([0-9]+) deselected)? in (.+)",
        value,
    )
    _require(match is not None, f"V8 immutable pytest summary invalid: {role}")
    assert match is not None
    passed = int(match.group(1))
    skipped = int(match.group(2)) if match.group(2) is not None else 0
    deselected = int(match.group(3)) if match.group(3) is not None else 0
    frozen_counts = {
        "frozen_v7_auditor_tests": (357, 3, 2),
        "frozen_v7_sealer_tests": (181, 1, 1),
    }
    if role in frozen_counts:
        _require(
            (passed, skipped, deselected) == frozen_counts[role]
            and match.group(2) is not None
            and match.group(3) is not None
            and value.count("deselected") == 1,
            f"V8 frozen pytest deselection summary mismatch: {role}",
        )
        normalized = f"{passed} passed, {skipped} skipped in {match.group(4)}"
        parsed = _parse_pytest_success_summary_local(
            normalized, label=f"V8 frozen pytest duration invalid: {role}"
        )
        _require(
            parsed["passed"] == passed and parsed["skipped"] == skipped,
            f"V8 frozen pytest normalized summary mismatch: {role}",
        )
    else:
        _require(
            role in {"v8_auditor_tests", "v8_sealer_tests"}
            and match.group(3) is None
            and deselected == 0
            and "deselected" not in value,
            f"V8 executing pytest suite unexpectedly deselected tests: {role}",
        )
        parsed = _parse_pytest_success_summary_local(
            value, label=f"V8 executing pytest summary invalid: {role}"
        )
        _require(
            parsed["passed"] == passed and parsed["skipped"] == skipped,
            f"V8 executing pytest summary count mismatch: {role}",
        )
    try:
        auditor._v8_validate_bound_test_run_summary(role, value)
    except Exception as exc:
        raise PostrunAuditSealError(
            f"V8 auditor bound-test summary rejection: {role}: {exc}"
        ) from exc
    return {"passed": passed, "skipped": skipped, "deselected": deselected}


def _frozen_v7_preseal_test_evidence_from_base(
    v7_base: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract the original full-suite proof from the strict-loaded V7 AUTH."""

    authorization = v7_base.get("authorization")
    _require(
        isinstance(authorization, Mapping),
        "V8 frozen V7 authorization unavailable for evidence inheritance",
    )
    evidence = authorization.get("test_evidence")
    _require(
        isinstance(evidence, Mapping),
        "V8 frozen V7 test evidence unavailable",
    )
    encoded = canonical_bytes(evidence)
    _require(
        len(encoded) == 14_174
        and hashlib.sha256(encoded).hexdigest()
        == V7_AUTHORITY_BASE["authorization_test_evidence_canonical_sha256"]
        and evidence.get("artifact_type")
        == "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_V7_TEST_EVIDENCE"
        and evidence.get("status")
        == "PASS_FROZEN_V7_AUDITOR_AND_SEALER_TESTS"
        and evidence.get("created_utc") == "2026-08-11T05:38:53.748612Z",
        "V8 frozen V7 original test-evidence identity mismatch",
    )
    original_full_runs = {
        "v7_auditor_tests": {
            "exit_code": 0,
            "summary": "359 passed, 3 skipped in 153.94s (0:02:33)",
            "stdout_sha256": (
                "2a7e8256ca1fb5122722b140036567934f7e00f11ebe73eec145216d7d8b239e"
            ),
            "stderr_sha256": (
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            ),
        },
        "v7_sealer_tests": {
            "exit_code": 0,
            "summary": "182 passed, 1 skipped in 5.58s",
            "stdout_sha256": (
                "142248e1d981b60609c547f7cf39200a3875d1ff46b4c42efac127a5e21d1573"
            ),
            "stderr_sha256": (
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            ),
        },
    }
    actual_runs = evidence.get("test_runs")
    _require(
        isinstance(actual_runs, Mapping),
        "V8 frozen V7 original test-run map absent",
    )
    extracted_runs: dict[str, dict[str, Any]] = {}
    for role in original_full_runs:
        record = actual_runs.get(role)
        _require(
            isinstance(record, Mapping),
            f"V8 frozen V7 original test run absent: {role}",
        )
        extracted_runs[role] = {
            key: record.get(key)
            for key in ("exit_code", "summary", "stdout_sha256", "stderr_sha256")
        }
    _require(
        extracted_runs == original_full_runs,
        "V8 frozen V7 original full-suite evidence mismatch",
    )
    inherited = {
        "authorization": dict(V7_AUTHORITY_BASE["authorization"]),
        "canonical_size_bytes": 14_174,
        "canonical_sha256": V7_AUTHORITY_BASE[
            "authorization_test_evidence_canonical_sha256"
        ],
        "artifact_type": "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_V7_TEST_EVIDENCE",
        "status": "PASS_FROZEN_V7_AUDITOR_AND_SEALER_TESTS",
        "created_utc": "2026-08-11T05:38:53.748612Z",
        "original_full_runs": original_full_runs,
        "state_obsolete_current_rerun_nodeids": [
            *V8_FROZEN_V7_TEST_DESELECTS[
                "tests/test_noaa_gfs_multiseason_raw_postrun_v7.py"
            ],
            *V8_FROZEN_V7_TEST_DESELECTS[
                "tests/test_seal_noaa_gfs_multiseason_postrun_audit_v7.py"
            ],
        ],
    }
    _require(len(inherited) == 8, "V8 frozen V7 inherited evidence schema mismatch")
    return inherited


def _load_and_validate_immutable_v2_preflight_evidence(
    root: Path,
) -> dict[str, Any]:
    """Validate the actual immutable V2 preflight and its six summaries."""

    _require(
        strict_root_identity(
            root, str(V2_PREFLIGHT_IDENTITY["path"]), label="immutable V2 preflight"
        )
        == V2_PREFLIGHT_IDENTITY,
        "immutable V2 preflight physical identity mismatch",
    )
    preflight = load_json_control(
        root_path(
            root, str(V2_PREFLIGHT_IDENTITY["path"]), label="immutable V2 preflight"
        ),
        label="immutable V2 preflight",
    )
    _require(
        set(preflight)
        == {
            "all_seven_worker_pids_observed", "arrays_2024_read",
            "arrays_2025_read", "artifact_type", "bootstrap_ast_stdlib_policy_pass",
            "bootstrap_import_contract", "bootstrap_matching_pyc_absent_after_tests",
            "bootstrap_matching_pyc_absent_before_tests", "bootstrap_trust_policy",
            "created_utc", "flawed_launch_incident", "focused_test_result", "incident",
            "labels_read", "launch_identity_correction", "max_spawn_processes",
            "models_fit", "network_requests", "pilot_report",
            "poisoned_pyc_source_execution_test", "postrun_auditor",
            "postrun_auditor_test", "process_scheduling_flake_incident",
            "production_decode_task_worker_real_test", "production_worker_report",
            "python_dont_write_bytecode_env", "python_dont_write_bytecode_flag",
            "python_pycache_prefix_absent", "real_grib_pilot", "recovery_bootstrap",
            "recovery_bootstrap_test", "recovery_module", "recovery_runner",
            "recovery_runner_test", "recovery_sealer", "recovery_sealer_test",
            "recovery_test", "run_prefix_regression", "runtime_lock", "schema_version",
            "sealer_entrypoint_incident", "source_compile_result", "status",
            "stdlib_bootstrap_worker_real_test", "submission_csv_created",
            "superseded_v1_chain", "superseded_v1_support", "test_evidence",
            "thread_decoders",
        }
        and preflight.get("schema_version") == 2
        and preflight.get("artifact_type")
        == "TRACK_A_DECODE_RECOVERY_REAL_PROCESS_PREFLIGHT_V2"
        and preflight.get("status")
        == "PASS_V2_REAL_PILOT_PRODUCTION_AND_RUN_PREFIX_EXACT_7_SPAWN",
        "immutable V2 preflight schema/header mismatch",
    )
    preflight_bytes = canonical_bytes(preflight)
    _require(
        len(preflight_bytes)
        == V8_IMMUTABLE_V2_PREFLIGHT_EVIDENCE[
            "preflight_payload_canonical_size_bytes"
        ]
        and hashlib.sha256(preflight_bytes).hexdigest()
        == V8_IMMUTABLE_V2_PREFLIGHT_EVIDENCE[
            "preflight_payload_canonical_sha256"
        ],
        "immutable V2 preflight canonical identity mismatch",
    )
    evidence = preflight.get("test_evidence")
    _require(
        isinstance(evidence, Mapping)
        and set(evidence)
        == {
            "all_pytest_exit_codes_zero", "arrays_2024_read", "arrays_2025_read",
            "artifact_type", "bootstrap_matching_pyc_absent_after_tests",
            "bootstrap_matching_pyc_absent_before_tests",
            "bound_code_and_test_identities", "compiled_source_identities",
            "created_utc", "isolated_subprocess_count", "labels_read", "models_fit",
            "network_requests", "noncontract_monolithic_diagnostic",
            "pytest_cacheprovider_disabled", "pytest_child_network_guard_installed",
            "pytest_child_network_guard_sha256", "pytest_isolation", "pytest_runs",
            "pytest_summary", "python_dont_write_bytecode_env",
            "python_dont_write_bytecode_flag", "python_executable",
            "python_pycache_prefix_absent", "required_test_names",
            "required_test_names_present", "schema_version", "source_compile_result",
            "status", "submission_csv_created",
        }
        and evidence.get("schema_version") == 1
        and evidence.get("artifact_type") == "DECODE_RECOVERY_IMMUTABLE_CODE_TEST_EVIDENCE"
        and evidence.get("status") == "PASS_BOUND_CODE_COMPILE_AND_TEST_SUITE"
        and evidence.get("all_pytest_exit_codes_zero") is True,
        "immutable V2 test-evidence schema/header mismatch",
    )
    evidence_bytes = canonical_bytes(evidence)
    _require(
        len(evidence_bytes)
        == V8_IMMUTABLE_V2_PREFLIGHT_EVIDENCE["test_evidence_canonical_size_bytes"]
        and hashlib.sha256(evidence_bytes).hexdigest()
        == V8_IMMUTABLE_V2_PREFLIGHT_EVIDENCE["test_evidence_canonical_sha256"],
        "immutable V2 test-evidence canonical identity mismatch",
    )
    runs = evidence.get("pytest_runs")
    _require(
        isinstance(runs, list)
        and len(runs) == V8_IMMUTABLE_V2_PREFLIGHT_EVIDENCE["pytest_run_count"],
        "immutable V2 pytest-run inventory mismatch",
    )
    pairs: list[dict[str, str]] = []
    summaries: list[str] = []
    aggregate_parts: list[str] = []
    for index, run in enumerate(runs):
        _require(
            isinstance(run, Mapping)
            and set(run)
            == {
                "test_file", "command", "exit_code", "summary",
                "stdout_sha256", "stderr_sha256",
            }
            and run.get("exit_code") == 0
            and isinstance(run.get("test_file"), str)
            and isinstance(run.get("summary"), str),
            f"immutable V2 pytest-run invalid: {index}",
        )
        summary = str(run["summary"])
        _parse_pytest_success_summary_local(
            summary, label=f"immutable V2 pytest summary invalid: {index}"
        )
        test_file = str(run["test_file"])
        pairs.append({"test_file": test_file, "summary": summary})
        summaries.append(summary)
        aggregate_parts.append(f"{test_file}: {summary}")
    aggregate = " | ".join(aggregate_parts)
    _require(
        evidence.get("pytest_summary") == aggregate
        and runs[5].get("summary")
        == V8_IMMUTABLE_V2_PREFLIGHT_EVIDENCE["long_duration_summary"],
        "immutable V2 ordered/aggregate pytest summary mismatch",
    )
    for value, size_field, sha_field, label in (
        (
            pairs, "test_file_summary_pairs_canonical_size_bytes",
            "test_file_summary_pairs_canonical_sha256", "test-file/summary pairs",
        ),
        (
            summaries, "ordered_summaries_canonical_size_bytes",
            "ordered_summaries_canonical_sha256", "ordered summaries",
        ),
        (
            aggregate, "aggregate_pytest_summary_canonical_size_bytes",
            "aggregate_pytest_summary_canonical_sha256", "aggregate summary",
        ),
    ):
        encoded = canonical_bytes(value)
        _require(
            len(encoded) == V8_IMMUTABLE_V2_PREFLIGHT_EVIDENCE[size_field]
            and hashlib.sha256(encoded).hexdigest()
            == V8_IMMUTABLE_V2_PREFLIGHT_EVIDENCE[sha_field],
            f"immutable V2 {label} canonical commitment mismatch",
        )
    return {
        "commitment": json.loads(
            json.dumps(V8_IMMUTABLE_V2_PREFLIGHT_EVIDENCE, ensure_ascii=False)
        ),
        "preflight": preflight,
        "evidence": evidence,
    }


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

    # These three documentary gates precede recovered metadata reconstruction.
    v7_base = _load_and_validate_v7_authority_base(root, V7_AUTHORITY_BASE)
    live_failure = _load_and_validate_v8_live_failure_state(root)
    immutable_preflight = _load_and_validate_immutable_v2_preflight_evidence(root)
    try:
        auditor_preflight = auditor._v8_validate_actual_v2_preflight_evidence(root)
    except Exception as exc:
        raise PostrunAuditSealError(
            f"independent auditor V2 preflight evidence rejection: {exc}"
        ) from exc
    _require(
        auditor_preflight.get("commitment") == immutable_preflight["commitment"]
        and auditor_preflight.get("preflight_identity") == V2_PREFLIGHT_IDENTITY
        and auditor_preflight.get("preflight") == immutable_preflight["preflight"]
        and auditor_preflight.get("test_evidence") == immutable_preflight["evidence"],
        "local/auditor immutable V2 preflight evidence mismatch",
    )
    try:
        auditor_base = auditor._v8_load_and_validate_v7_authority_base(
            root, V7_AUTHORITY_BASE
        )
        auditor_failure = auditor._v8_validate_duration_suffix_incident_chain(
            root, V8_PRIMARY_INCIDENT_IDENTITY, V8_CORRECTION_RECORD_IDENTITY
        )
    except Exception as exc:
        raise PostrunAuditSealError(
            f"independent auditor V7 authority/failure-chain rejection: {exc}"
        ) from exc
    _require(
        auditor_base.get("summary") == v7_base["base"]
        and auditor_base.get("authorization") == v7_base["authorization"]
        and auditor_base.get("review") == v7_base["review"]
        and auditor_base.get("go") == v7_base["go"]
        and auditor_failure.get("state") == live_failure["state"]
        and auditor_failure.get("primary_created", {}).get("raw")
        == live_failure["primary_created_utc"]
        and auditor_failure.get("correction_created", {}).get("raw")
        == live_failure["correction_created_utc"],
        "local/auditor V7 authority/failure-chain mismatch",
    )
    try:
        base_state = auditor._v6_load_and_validate_v5_authority_base(
            root, V5_AUTHORITY_BASE
        )
    except Exception as exc:
        raise PostrunAuditSealError(f"V8 transitive V5 authority invalid: {exc}") from exc
    afterstate = base_state["recovered_afterstate"]
    commitment = _recovered_afterstate_commitment(afterstate)
    _require(
        commitment == v7_base["recovered_afterstate_commitment"],
        "V8 transitive/V7 compact afterstate commitment mismatch",
    )
    closure = validate_recovered_filesystem_closure(
        root, afterstate, base_state["v4_base"]["authorization"]
    )
    snapshot = build_zero_snapshot(root)
    return {
        "v7_authorization_created_utc": v7_base["authorization"]["created_utc"],
        "v7_review_created_utc": v7_base["review"]["created_utc"],
        "v7_go_created_utc": v7_base["go"]["created_utc"],
        "primary_incident_created_utc": live_failure["primary_created_utc"],
        "correction_created_utc": live_failure["correction_created_utc"],
        "v7_authority_base": dict(V7_AUTHORITY_BASE),
        "v7_live_failure_state": live_failure["state"],
        "immutable_v2_preflight_evidence": immutable_preflight["commitment"],
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
        "v7_authorization": dict(V7_AUTHORITY_BASE["authorization"]),
        "v7_independent_review": dict(V7_AUTHORITY_BASE["independent_review"]),
        "v7_independent_go": dict(V7_AUTHORITY_BASE["independent_go"]),
        "primary_duration_suffix_incident": dict(V8_PRIMARY_INCIDENT_IDENTITY),
        "afterstate_commitment_correction": dict(V8_CORRECTION_RECORD_IDENTITY),
        "immutable_v2_preflight": dict(V2_PREFLIGHT_IDENTITY),
    }
    for role, record in root_bound.items():
        _require(
            strict_root_identity(root, str(record["path"]), label=role) == record,
            f"V8 evidence root identity mismatch: {role}",
        )
    combined = {**root_bound, **{role: dict(value) for role, value in source_bound.items()}}
    _require(
        set(combined) == BOUND_IDENTITY_ROLES and len(combined) == 14,
        "V8 evidence exact-fourteen bound-identity schema mismatch",
    )
    return combined


def _compile_bound_sources(bound: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered_roles = ("v8_auditor", "v8_auditor_test", "v8_sealer", "v8_sealer_test")
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
    local_v7_base = _load_and_validate_v7_authority_base(root, V7_AUTHORITY_BASE)
    inherited_v7_evidence = _frozen_v7_preseal_test_evidence_from_base(
        local_v7_base
    )
    try:
        auditor_v7_base = auditor._v8_load_and_validate_v7_authority_base(
            root, V7_AUTHORITY_BASE
        )
        auditor_inherited_v7_evidence = auditor._v8_frozen_v7_preseal_test_evidence(
            auditor_v7_base
        )
    except Exception as exc:
        raise PostrunAuditSealError(
            f"V8 auditor frozen V7 evidence inheritance rejection: {exc}"
        ) from exc
    _require(
        auditor_inherited_v7_evidence == inherited_v7_evidence,
        "V8 local/auditor frozen V7 evidence inheritance mismatch",
    )
    auditor_test_names = _source_function_names(Path(str(source_bound["v8_auditor_test"]["path"])))
    _require(set(auditor.V8_REQUIRED_TEST_NAMES).issubset(auditor_test_names), "required V8 regression test is absent")

    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.pop("PYTHONPYCACHEPREFIX", None)
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    environment["PYTHONIOENCODING"] = "utf-8"
    runs: dict[str, dict[str, Any]] = {}
    for role, relative in TEST_RELATIVES.items():
        _require_bound_bytecode_absent(source_bound)
        command = _v8_test_command(relative)
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
        _validate_bound_test_run_summary_local(role, summary)
        deselected_nodeids = list(V8_FROZEN_V7_TEST_DESELECTS.get(relative, ()))
        runs[role] = {
            "command": command,
            "exit_code": completed.returncode,
            "summary": summary,
            "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
            "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
            "network_guard_installed": True,
            "cacheprovider_disabled": True,
            "selection_mode": (
                "FULL_FILE_MINUS_EXACT_OBSOLETE_PRESEAL_STATE_TESTS"
                if deselected_nodeids else "FULL_FILE"
            ),
            "deselected_nodeids": deselected_nodeids,
            "deselected_count": len(deselected_nodeids),
        }
        _require_bound_bytecode_absent(source_bound)
    _require(set(runs) == auditor.V8_TEST_RUN_KEYS, "test-run role mismatch")
    _require(collect_code_state(workspace_root) == code_state, "code/test identity changed during isolated tests")
    evidence = {
        "schema_version": 1,
        "artifact_type": "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_V8_TEST_EVIDENCE",
        "status": "PASS_BOUND_V8_CODE_COMPILE_AND_TEST_SUITE",
        "created_utc": utc_now_exact(),
        "bound_identities": evidence_bound,
        "source_compile": {"method": "compile_exact_source_no_pyc", "result": "PASS", "files": compiled},
        "pytest_isolation": "ONE_TEST_FILE_PER_CLEAN_NETWORK_DENIED_SUBPROCESS",
        "test_runs": runs,
        "required_test_names": list(auditor.V8_REQUIRED_TEST_NAMES),
        "required_test_names_present": True,
        "all_exit_codes_zero": True,
        "production_shape_regression": {
            key: True for key in auditor.V8_PRODUCTION_SHAPE_KEYS
        },
        "real_seven_spawn_regression": {
            "start_method": "spawn",
            "entrypoint_module": "scripts.audit_noaa_gfs_multiseason_raw_postrun_v8",
            "worker_callable_module": "scripts.audit_noaa_gfs_multiseason_raw_postrun_v8",
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
        "frozen_v7_preseal_test_evidence": inherited_v7_evidence,
    }
    _require(set(evidence) == auditor.V8_TEST_EVIDENCE_KEYS, "internal V8 test-evidence schema mismatch")
    return evidence


def _validate_v8_test_evidence_local(
    evidence: Any,
    *,
    authorization_created_ticks: int,
    bound_identities: Mapping[str, Mapping[str, Any]],
    v7_base: Mapping[str, Any],
) -> None:
    _require(
        isinstance(evidence, Mapping)
        and set(evidence) == auditor.V8_TEST_EVIDENCE_KEYS,
        "V8 test-evidence schema mismatch",
    )
    evidence_created_ticks = _exact_utc_z_100ns_ticks(
        evidence.get("created_utc"), label="V8 test evidence created_utc"
    )
    expected_compile = [
        bound_identities[role]
        for role in ("v8_auditor", "v8_auditor_test", "v8_sealer", "v8_sealer_test")
    ]
    _require(
        evidence.get("schema_version") == 1
        and evidence.get("artifact_type")
        == "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDITOR_V8_TEST_EVIDENCE"
        and evidence.get("status") == "PASS_BOUND_V8_CODE_COMPILE_AND_TEST_SUITE"
        and evidence_created_ticks <= authorization_created_ticks
        and evidence.get("bound_identities") == dict(bound_identities)
        and evidence.get("source_compile")
        == {
            "method": "compile_exact_source_no_pyc",
            "result": "PASS",
            "files": expected_compile,
        }
        and evidence.get("pytest_isolation")
        == "ONE_TEST_FILE_PER_CLEAN_NETWORK_DENIED_SUBPROCESS"
        and evidence.get("required_test_names")
        == list(auditor.V8_REQUIRED_TEST_NAMES)
        and evidence.get("required_test_names_present") is True
        and evidence.get("all_exit_codes_zero") is True,
        "V8 test-evidence header/source binding mismatch",
    )
    _require(
        evidence.get("frozen_v7_preseal_test_evidence")
        == _frozen_v7_preseal_test_evidence_from_base(v7_base),
        "V8 frozen V7 preseal test-evidence inheritance mismatch",
    )
    runs = evidence.get("test_runs")
    _require(
        isinstance(runs, Mapping) and set(runs) == auditor.V8_TEST_RUN_KEYS,
        "V8 test-evidence run schema mismatch",
    )
    for role, relative in TEST_RELATIVES.items():
        record = runs.get(role)
        _require(
            isinstance(record, Mapping)
            and set(record) == auditor.V8_TEST_RUN_RECORD_KEYS
            and record.get("command") == _v8_test_command(relative)
            and record.get("exit_code") == 0
            and isinstance(record.get("summary"), str)
            and "passed" in record["summary"]
            and "failed" not in record["summary"].casefold()
            and "error" not in record["summary"].casefold()
            and re.fullmatch(r"[0-9a-f]{64}", str(record.get("stdout_sha256")))
            and re.fullmatch(r"[0-9a-f]{64}", str(record.get("stderr_sha256")))
            and record.get("network_guard_installed") is True
            and record.get("cacheprovider_disabled") is True,
            f"V8 test-evidence run invalid: {role}",
        )
        expected_nodeids = list(V8_FROZEN_V7_TEST_DESELECTS.get(relative, ()))
        _require(
            record.get("selection_mode")
            == (
                "FULL_FILE_MINUS_EXACT_OBSOLETE_PRESEAL_STATE_TESTS"
                if expected_nodeids else "FULL_FILE"
            )
            and record.get("deselected_nodeids") == expected_nodeids
            and type(record.get("deselected_count")) is int
            and record.get("deselected_count") == len(expected_nodeids),
            f"V8 test-evidence selection mismatch: {role}",
        )
        _validate_bound_test_run_summary_local(role, record["summary"])
    expected_shape_keys = auditor.V8_PRODUCTION_SHAPE_KEYS
    shape = evidence.get("production_shape_regression")
    _require(
        isinstance(shape, Mapping)
        and set(shape) == expected_shape_keys
        and all(shape[key] is True for key in expected_shape_keys),
        "V8 test-evidence production-shape regression mismatch",
    )
    real_seven = evidence.get("real_seven_spawn_regression")
    _require(
        real_seven
        == {
            "start_method": "spawn",
            "entrypoint_module": "scripts.audit_noaa_gfs_multiseason_raw_postrun_v8",
            "worker_callable_module": "scripts.audit_noaa_gfs_multiseason_raw_postrun_v8",
            "max_workers": 7,
            "distinct_worker_pids": 7,
            "first_wave_synchronized": True,
            "network_requests": 0,
            "audit_files_written": 0,
            "labels_read": False,
            "arrays_2024_read": False,
            "arrays_2025_read": False,
        },
        "V8 test-evidence real-seven regression mismatch",
    )
    _require(
        evidence.get("stdout_only_regression") is True
        and evidence.get("network_guard")
        == {
            "parent_test_process_guarded": True,
            "sys_audit_hook_in_parent": True,
            "parent_socket_api_denied": True,
            "spawned_worker_network_route_static_absent": True,
            "spawned_worker_runtime_guard_installed": False,
            "external_packet_capture": False,
        }
        and evidence.get("network_requests") == 0
        and evidence.get("audit_files_written") == 0
        and evidence.get("labels_read") is False
        and evidence.get("arrays_2024_read") is False
        and evidence.get("arrays_2025_read") is False
        and evidence.get("models_fit") == 0
        and evidence.get("submission_csv_created") is False,
        "V8 test-evidence safety mismatch",
    )
def build_authorization(
    root: Path,
    code_state: Mapping[str, Any],
    inputs: Mapping[str, Any],
    test_evidence: Mapping[str, Any],
    *,
    created_utc: str,
) -> dict[str, Any]:
    compact = created_utc.replace("-", "").replace(":", "").replace(".", "")
    audit_attempt_id = f"postrun_audit_v8__{compact}"
    bound = code_state["bound_identities"]
    payload = {
        "schema_version": 8,
        "artifact_type": "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDIT_AUTHORIZATION_V8",
        "status": "AUTHORIZED_PENDING_INDEPENDENT_REVIEW_AND_GO_V8",
        "created_utc": created_utc,
        "audit_attempt_id": audit_attempt_id,
        "v7_authority_base": inputs["v7_authority_base"],
        "v7_live_failure_state": inputs["v7_live_failure_state"],
        "immutable_v2_preflight_evidence": inputs[
            "immutable_v2_preflight_evidence"
        ],
        "superseded_v7_auditor": bound["superseded_v7_auditor"],
        "superseded_v7_auditor_test": bound["superseded_v7_auditor_test"],
        "superseded_v7_sealer": bound["superseded_v7_sealer"],
        "superseded_v7_sealer_test": bound["superseded_v7_sealer_test"],
        "v8_auditor": bound["v8_auditor"],
        "v8_auditor_test": bound["v8_auditor_test"],
        "v8_sealer": bound["v8_sealer"],
        "v8_sealer_test": bound["v8_sealer_test"],
        "recovery_attempt_id": V7_AUTHORITY_BASE["recovery_attempt_id"],
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


def _assert_v8_authorization_compact(payload: Mapping[str, Any]) -> None:
    encoded = canonical_bytes(payload)
    physical = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )
    _require(
        len(encoded) <= 32_768 and len(physical) <= 32_768,
        "V8 authorization exceeds the 32KB cap",
    )
    forbidden_keys = {
        "inventory", "recovered_afterstate", "recovery_progress",
        "canonical_outputs", "recovery_transaction", "recovery_history",
        "report", "postrun_report",
    }

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            _require(
                forbidden_keys.isdisjoint(str(key) for key in value),
                "V8 authorization embeds a forbidden full inventory/report field",
            )
            for nested in value.values():
                walk(nested)
        elif isinstance(value, list):
            _require(
                len(value) != 104,
                "V8 authorization embeds a forbidden 104-item inventory",
            )
            for nested in value:
                walk(nested)

    walk(payload)


def validate_authorization_payload(
    root: Path,
    payload: Mapping[str, Any],
    code_state: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> None:
    _require(set(payload) == auditor.V8_AUTH_KEYS, "V8 authorization schema mismatch")
    created_raw = payload.get("created_utc")
    _require(
        isinstance(created_raw, str)
        and re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:"
            r"[0-9]{2}:[0-9]{2}\.[0-9]{6}Z",
            created_raw,
        )
        is not None,
        "V8 authorization created_utc is not exact UTC RFC3339 microseconds",
    )
    evidence = payload.get("test_evidence")
    _require(isinstance(evidence, Mapping), "V8 authorization test evidence absent")
    raw_chronology = {
        "v7_authorization_created_utc": inputs.get("v7_authorization_created_utc"),
        "v7_review_created_utc": inputs.get("v7_review_created_utc"),
        "v7_go_created_utc": inputs.get("v7_go_created_utc"),
        "primary_incident_created_utc": inputs.get("primary_incident_created_utc"),
        "correction_created_utc": inputs.get("correction_created_utc"),
        "evidence_created_utc": evidence.get("created_utc"),
        "authorization_created_utc": created_raw,
    }
    chronology = {
        role: _exact_utc_z_100ns_ticks(value, label=role)
        for role, value in raw_chronology.items()
    }
    for role, raw in raw_chronology.items():
        try:
            parsed = auditor._v7_parse_rfc3339_utc_100ns(
                raw, label=f"V8 independent {role}"
            )
        except Exception as exc:
            raise PostrunAuditSealError(
                f"V8 independent timestamp rejection: {role}: {exc}"
            ) from exc
        _require(
            parsed.get("raw") == raw
            and parsed.get("ticks_100ns") == chronology[role],
            f"V8 local/auditor timestamp mismatch: {role}",
        )
    strict_roles = tuple(raw_chronology)[:5]
    _require(
        all(
            chronology[left] < chronology[right]
            for left, right in zip(strict_roles, strict_roles[1:])
        )
        and chronology["correction_created_utc"]
        <= chronology["evidence_created_utc"]
        <= chronology["authorization_created_utc"],
        "V7 AUTH/REVIEW/GO/primary/correction/V8 evidence/AUTH chronology mismatch",
    )
    expected_attempt = "postrun_audit_v8__" + created_raw.translate(
        str.maketrans("", "", "-:.")
    )
    _require(
        payload.get("schema_version") == 8
        and payload.get("artifact_type")
        == "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDIT_AUTHORIZATION_V8"
        and payload.get("status")
        == "AUTHORIZED_PENDING_INDEPENDENT_REVIEW_AND_GO_V8"
        and payload.get("audit_attempt_id") == expected_attempt
        and re.fullmatch(
            r"postrun_audit_v8__[0-9]{8}T[0-9]{12}Z", expected_attempt
        )
        is not None,
        "V8 authorization header/attempt mismatch",
    )
    expected_policy = {
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
    }
    _require(
        all(payload.get(key) == value for key, value in expected_policy.items()),
        "V8 authorization policy mismatch",
    )
    _assert_v8_authorization_compact(payload)

    bound = code_state.get("bound_identities")
    _require(
        isinstance(bound, Mapping) and set(bound) == BOUND_SOURCE_ROLES,
        "V8 authorization bound-source schema mismatch",
    )
    for role, expected in FROZEN_V7_SOURCE_IDENTITIES.items():
        _require(bound[role] == expected, f"frozen V7 source changed: {role}")
    _require(
        bound["v8_auditor"] == EXPECTED_V8_AUDITOR_IDENTITY
        and bound["v8_auditor_test"] == EXPECTED_V8_AUDITOR_TEST_IDENTITY,
        "settled V8 auditor/test identity mismatch",
    )
    _require(
        all(payload.get(role) == bound[role] for role in BOUND_SOURCE_ROLES),
        "V8 authorization source identity binding mismatch",
    )
    _require(
        payload.get("v7_authority_base") == inputs.get("v7_authority_base")
        == V7_AUTHORITY_BASE
        and payload.get("v7_live_failure_state")
        == inputs.get("v7_live_failure_state")
        and payload.get("immutable_v2_preflight_evidence")
        == inputs.get("immutable_v2_preflight_evidence")
        == V8_IMMUTABLE_V2_PREFLIGHT_EVIDENCE,
        "V8 authority/failure/preflight compact binding mismatch",
    )
    local_base = _load_and_validate_v7_authority_base(
        root, payload["v7_authority_base"]
    )
    local_failure = _load_and_validate_v8_live_failure_state(root)
    local_preflight = _load_and_validate_immutable_v2_preflight_evidence(root)
    try:
        auditor_preflight = auditor._v8_validate_actual_v2_preflight_evidence(root)
        auditor_base = auditor._v8_load_and_validate_v7_authority_base(
            root, payload["v7_authority_base"]
        )
        auditor_failure = auditor._v8_validate_duration_suffix_incident_chain(
            root, V8_PRIMARY_INCIDENT_IDENTITY, V8_CORRECTION_RECORD_IDENTITY
        )
    except Exception as exc:
        raise PostrunAuditSealError(
            f"V8 auditor preflight self-validator rejection: {exc}"
        ) from exc
    _require(
        local_failure["state"] == payload["v7_live_failure_state"]
        and local_preflight["commitment"]
        == payload["immutable_v2_preflight_evidence"]
        == auditor_preflight.get("commitment")
        and auditor_preflight.get("preflight_identity") == V2_PREFLIGHT_IDENTITY
        and auditor_preflight.get("preflight") == local_preflight["preflight"]
        and auditor_preflight.get("test_evidence") == local_preflight["evidence"]
        and local_base["base"] == payload["v7_authority_base"]
        == auditor_base.get("summary")
        and local_base["authorization"] == auditor_base.get("authorization")
        and local_base["review"] == auditor_base.get("review")
        and local_base["go"] == auditor_base.get("go")
        and local_failure["state"] == auditor_failure.get("state"),
        "V8 independent authority input revalidation mismatch",
    )
    _require(
        raw_chronology["v7_authorization_created_utc"]
        == local_base["authorization"].get("created_utc")
        and raw_chronology["v7_review_created_utc"]
        == local_base["review"].get("created_utc")
        and raw_chronology["v7_go_created_utc"]
        == local_base["go"].get("created_utc")
        and raw_chronology["primary_incident_created_utc"]
        == local_failure["primary_created_utc"]
        and raw_chronology["correction_created_utc"]
        == local_failure["correction_created_utc"],
        "V8 helper-returned raw authority timestamp binding mismatch",
    )
    _require(
        raw_chronology["v7_authorization_created_utc"]
        == auditor_base.get("authorization_created", {}).get("raw")
        and raw_chronology["v7_review_created_utc"]
        == auditor_base.get("review_created", {}).get("raw")
        and raw_chronology["v7_go_created_utc"]
        == auditor_base.get("go_created", {}).get("raw")
        and raw_chronology["primary_incident_created_utc"]
        == auditor_failure.get("primary_created", {}).get("raw")
        and raw_chronology["correction_created_utc"]
        == auditor_failure.get("correction_created", {}).get("raw"),
        "V8 independent helper raw authority timestamp binding mismatch",
    )
    _require(
        payload.get("recovery_attempt_id")
        == V7_AUTHORITY_BASE["recovery_attempt_id"]
        and payload.get("recovered_afterstate_commitment")
        == inputs.get("recovered_afterstate_commitment")
        == local_base["recovered_afterstate_commitment"]
        == auditor.V5_RECOVERED_AFTERSTATE_COMMITMENT
        and payload.get("runtime_identity_sha256")
        == auditor.V4_RUNTIME_IDENTITY_SHA256
        and payload.get("required_command") == required_audit_command(root)
        and payload.get("independent_review_required") is True
        and payload.get("independent_go_required") is True,
        "V8 immutable execution/afterstate binding mismatch",
    )
    snapshot = payload.get("preaudit_zero_mutation_snapshot")
    _require(
        snapshot == inputs.get("preaudit_zero_mutation_snapshot")
        and isinstance(snapshot, Mapping)
        and set(snapshot) == auditor.V8_ZERO_SNAPSHOT_KEYS
        and snapshot.get("v7_live_failure_state_canonical_sha256")
        == V8_LIVE_FAILURE_STATE_CANONICAL_SHA256
        and snapshot.get("recovered_afterstate_commitment_canonical_sha256")
        == auditor.V5_RECOVERED_AFTERSTATE_COMMITMENT_CANONICAL_SHA256,
        "V8 preauthorization zero snapshot mismatch",
    )
    try:
        auditor._v8_validate_zero_snapshot(
            root, snapshot, failure_state=local_failure["state"]
        )
        auditor._v8_assert_compact_control(payload, label="V8 authorization sealer")
    except Exception as exc:
        raise PostrunAuditSealError(
            f"V8 independent zero-snapshot/compact-control rejection: {exc}"
        ) from exc
    expected_evidence_bound = {
        "v7_authorization": V7_AUTHORITY_BASE["authorization"],
        "v7_independent_review": V7_AUTHORITY_BASE["independent_review"],
        "v7_independent_go": V7_AUTHORITY_BASE["independent_go"],
        "primary_duration_suffix_incident": V8_PRIMARY_INCIDENT_IDENTITY,
        "afterstate_commitment_correction": V8_CORRECTION_RECORD_IDENTITY,
        "immutable_v2_preflight": V2_PREFLIGHT_IDENTITY,
        **{role: bound[role] for role in BOUND_SOURCE_ROLES},
    }
    _require(
        set(expected_evidence_bound) == BOUND_IDENTITY_ROLES
        and len(expected_evidence_bound) == 14
        and evidence.get("bound_identities") == expected_evidence_bound,
        "V8 authorization exact-fourteen evidence binding mismatch",
    )
    _validate_v8_test_evidence_local(
        evidence,
        authorization_created_ticks=chronology["authorization_created_utc"],
        bound_identities=expected_evidence_bound,
        v7_base=local_base,
    )
    try:
        auditor_evidence_created = auditor._v8_validate_test_evidence(
            evidence,
            authorization_created={
                "ticks_100ns": chronology["authorization_created_utc"]
            },
            bound_identities=expected_evidence_bound,
            v7_base=auditor_base,
        )
    except Exception as exc:
        raise PostrunAuditSealError(
            f"V8 independent auditor test-evidence rejection: {exc}"
        ) from exc
    _require(
        auditor_evidence_created.get("raw") == evidence.get("created_utc")
        and auditor_evidence_created.get("ticks_100ns")
        == chronology["evidence_created_utc"],
        "V8 local/auditor test-evidence timestamp mismatch",
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
        "status": "PASS_V8_AUTHORIZATION_PRESEAL_STATIC_AUDIT",
        "artifact_type": "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDIT_V8_AUTH_SEALER_STATIC_AUDIT",
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
            root, AUTH_RELATIVE, label="published V8 authorization"
        )
        _require(
            observed_identity == expected_published_identity,
            "published authorization identity changed after hard-link verification",
        )
        published_identity = observed_identity
        require_postpublication_control_state(root)
        _require(
            load_json_control(destination, label="published V8 authorization")
            == authorization,
            "published authorization payload mismatch",
        )
        try:
            strict_published = auditor._v5_load_strict_json(
                destination, label="published V8 authorization"
            )
        except Exception as exc:
            raise PostrunAuditSealError(
                f"published V8 authorization strict JSON rejection: {exc}"
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
        "status": "PASS_V8_AUTHORIZATION_SEALED_PENDING_INDEPENDENT_REVIEW_AND_GO",
        "artifact_type": "TRACK_A_DECODE_RECOVERY_POSTRUN_AUDIT_V8_AUTH_SEAL_REPORT",
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
            "status": "FAIL_V8_POSTRUN_AUDIT_AUTHORIZATION_POSTPUBLICATION",
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
            "status": "FAIL_V8_POSTRUN_AUDIT_AUTHORIZATION_SEAL",
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

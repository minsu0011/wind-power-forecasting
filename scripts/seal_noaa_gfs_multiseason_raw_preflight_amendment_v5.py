#!/usr/bin/env python
"""Append-only test-result and final-authorization requirements for raw v4."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
ROOT_DEFAULT = REPO / "artifacts" / "baram2026_ncei_scada_longrun_20260810_v2"
PARENT_SHA = "f7ce0d39b0b69a6ae400e2bf95df4706b6f2c8188a76742e1b573d0f1ea64b27"
RUNNER_SHA = "51e6c4b68d47916f56975b16b9ceaae2e345f9f68963dba197501448d655934c"
TEST_SHA = "068ea40e2afab6b12421741229c14e73d71b7dda3e81df3ee9d0874f8bba2468"


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


def write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def run(root: Path) -> dict[str, Any]:
    root = root.resolve()
    parent = root / "manifest_raw_preflight_v4.json"
    parent_draft = root / "prelaunch" / "RAW_LAUNCH_AUTHORIZATION_DRAFT_V1.json"
    runner = REPO / "scripts" / "run_noaa_gfs_multiseason_raw_v2.py"
    runner_test = REPO / "tests" / "test_noaa_gfs_multiseason_raw_v2.py"
    amendment_path = root / "prelaunch" / "RAW_RUNNER_STATIC_PREFLIGHT_AMENDMENT_V5.json"
    draft_amendment_path = (
        root / "prelaunch" / "RAW_LAUNCH_AUTHORIZATION_DRAFT_AMENDMENT_V2.json"
    )
    manifest_path = root / "manifest_raw_preflight_v5.json"
    seal_code = Path(__file__).resolve()
    seal_test = REPO / "tests" / "test_noaa_gfs_multiseason_raw_preflight_amendment_v5.py"
    for path in (amendment_path, draft_amendment_path, manifest_path):
        if path.exists():
            raise FileExistsError(path)
    if sha256_file(parent) != PARENT_SHA:
        raise RuntimeError("v4 preflight parent changed")
    if sha256_file(runner) != RUNNER_SHA or sha256_file(runner_test) != TEST_SHA:
        raise RuntimeError("reviewed runner/test changed")
    actual_auth = root / "prereg" / "raw_launch_authorization_v1.json"
    go = root / "independent_redteam" / "TRACK_A_RAW_LAUNCH_GO.json"
    raw = root / "raw"
    decoded = root / "decoded"
    if any(path.exists() for path in (actual_auth, go, raw, decoded)):
        raise RuntimeError("zero-state changed before v5 result seal")

    amendment = {
        "artifact_type": "RAW_RUNNER_STATIC_PREFLIGHT_APPEND_ONLY_AMENDMENT",
        "schema_version": 5,
        "status": "INDEPENDENT_WINDOWS_TEST_PASS_AWAITING_PRELAUNCH_AUDIT_NOT_AUTHORIZED",
        "parent_manifest": identity(parent, root),
        "runner": identity(runner),
        "runner_test": identity(runner_test),
        "independent_windows_test": {
            "command": ".venv\\Scripts\\python.exe -m pytest -q tests/test_noaa_gfs_multiseason_raw_v2.py",
            "result": "25 passed",
            "platform": "Windows-11-10.0.26200-SP0",
            "network_requests": 0,
        },
        "independent_windows_test_result": "25 passed",
        "py_compile": {
            "command": ".venv\\Scripts\\python.exe -m py_compile scripts/run_noaa_gfs_multiseason_raw_v2.py tests/test_noaa_gfs_multiseason_raw_v2.py",
            "result": "PASS",
        },
        "py_compile_result": "PASS",
        "v4_artifacts_immutable": True,
        "authorization_created": False,
        "independent_go_present": False,
        "raw_network_requests": 0,
        "raw_network_bytes": 0,
        "labels_read": False,
        "2024_arrays_read": False,
        "2025_arrays_read": False,
        "models_fit": 0,
        "submission_csv_created": False,
    }
    write_json_exclusive(amendment_path, amendment)

    draft_amendment = {
        "artifact_type": "RAW_LAUNCH_AUTHORIZATION_DRAFT_APPEND_ONLY_AMENDMENT",
        "schema_version": 2,
        "status": "NON_EXECUTABLE_AWAITING_INDEPENDENT_PRELAUNCH_AUDIT",
        "parent_authorization_draft": identity(parent_draft, root),
        "effective_test_result_amendment": identity(amendment_path, root),
        "final_executable_authorization_requirements": {
            "must_not_copy_v4_proposed_payload_verbatim": True,
            "must_bind_effective_preflight_manifest_v5": True,
            "must_bind_forthcoming_independent_prelaunch_audit_identity": True,
            "independent_prelaunch_audit_status_must_be_pass": True,
            "final_authorization_sha_must_be_bound_by_independent_go": True,
            "runner_sha256": RUNNER_SHA,
            "runner_test_sha256": TEST_SHA,
            "independent_windows_test_result": "25 passed",
            "py_compile_result": "PASS",
            "raw_actual_http_attempt_budget": 15_200,
            "census_worst_case_http_attempts": 4_800,
            "census_plus_raw_max_http_attempts": 20_000,
        },
        "actual_authorization_created": False,
        "independent_go_created": False,
        "raw_network_requests": 0,
    }
    write_json_exclusive(draft_amendment_path, draft_amendment)

    manifest = {
        "artifact_type": "RAW_RUNNER_STATIC_PREFLIGHT_MANIFEST",
        "schema_version": 5,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "parent_manifest": identity(parent, root),
        "amendment": identity(amendment_path, root),
        "authorization_draft_amendment": identity(draft_amendment_path, root),
        "runner": identity(runner),
        "runner_test": identity(runner_test),
        "seal_code": identity(seal_code),
        "seal_test": identity(seal_test),
        "independent_windows_test_result": "25 passed",
        "py_compile_result": "PASS",
        "authorization_created": False,
        "independent_go_present": False,
        "raw_network_requests": 0,
    }
    write_json_exclusive(manifest_path, manifest)
    return {
        "amendment": identity(amendment_path, root),
        "authorization_draft_amendment": identity(draft_amendment_path, root),
        "manifest": identity(manifest_path, root),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(run(args.root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

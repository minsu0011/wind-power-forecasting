#!/usr/bin/env python
"""Append-only global HTTP-attempt budget correction for raw preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
ROOT_DEFAULT = REPO / "artifacts" / "baram2026_ncei_scada_longrun_20260810_v2"
PARENT_SHA = "47ddc9df6e4346df20ed259de093e96a88dc2202424ff3f36ccceaff025d59a3"
RUNNER_SHA = "52e7bbe4fcc4556a6c3001b679ad91685296b0a59957fc06de8010cd852334cd"
TEST_SHA = "df72d17c62f53cdf304785faad379dbf373e503cf7e87fbb007954c34bde6c45"


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
    parent = root / "manifest_raw_preflight_v2.json"
    runner = REPO / "scripts" / "run_noaa_gfs_multiseason_raw_v2.py"
    test = REPO / "tests" / "test_noaa_gfs_multiseason_raw_v2.py"
    census_runner = REPO / "scripts" / "census_noaa_gfs_multiseason_v2.py"
    if sha256_file(parent) != PARENT_SHA:
        raise RuntimeError("parent raw preflight v2 identity changed")
    if sha256_file(runner) != RUNNER_SHA or sha256_file(test) != TEST_SHA:
        raise RuntimeError("corrected raw runner/test identity changed")
    amendment_path = root / "prelaunch" / "RAW_RUNNER_STATIC_PREFLIGHT_AMENDMENT_V3.json"
    manifest_path = root / "manifest_raw_preflight_v3.json"
    if amendment_path.exists() or manifest_path.exists():
        raise FileExistsError("append-only raw preflight v3 already exists")
    amendment = {
        "artifact_type": "RAW_RUNNER_STATIC_PREFLIGHT_APPEND_ONLY_AMENDMENT",
        "schema_version": 3,
        "status": "READY_FOR_INDEPENDENT_RUNNER_REVIEW_NOT_AUTHORIZED",
        "parent_manifest": identity(parent, root),
        "runner": identity(runner),
        "runner_test": identity(test),
        "global_actual_http_attempt_cap": 20_000,
        "census": {
            "logical_successful_requests": 1_200,
            "attempts_per_logical_request_hard_max": 4,
            "actual_attempt_upper_bound": 4_800,
            "failed_attempt_exact_count": "UNDETERMINED_PRE_INSTRUMENTATION",
            "upper_bound_source": identity(census_runner),
        },
        "raw": {
            "required_successful_ranges": 10_368,
            "actual_attempt_budget": 15_200,
            "retry_headroom_after_one_success_per_range": 4_832,
            "thread_safe_reservation_before_every_attempt": True,
            "durable_attempt_events": True,
        },
        "proof": "census actual attempts <= 1,200*4 = 4,800 and raw reservation rejects attempt 15,201; therefore global actual attempts <= 4,800+15,200 = 20,000.",
        "supersedes_budget_only": {
            "provisional_raw_actual_http_attempt_budget": 18_800,
            "effective_raw_actual_http_attempt_budget": 15_200,
        },
        "all_other_v2_runner_hardening_and_bindings_unchanged": True,
        "authorization_created": False,
        "independent_go_present": False,
        "raw_network_requests": 0,
        "labels_read": False,
    }
    write_json_exclusive(amendment_path, amendment)
    source = Path(__file__).resolve()
    seal_test = REPO / "tests" / "test_noaa_gfs_multiseason_raw_preflight_amendment_v3.py"
    manifest = {
        "artifact_type": "RAW_RUNNER_STATIC_PREFLIGHT_MANIFEST",
        "schema_version": 3,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "parent_manifest": identity(parent, root),
        "amendment": identity(amendment_path, root),
        "runner": identity(runner),
        "runner_test": identity(test),
        "seal_code": identity(source),
        "seal_test": identity(seal_test),
        "effective_raw_actual_http_attempt_budget": 15_200,
        "authorization_created": False,
        "raw_network_requests": 0,
        "labels_read": False,
    }
    write_json_exclusive(manifest_path, manifest)
    return {
        "amendment": identity(amendment_path, root),
        "manifest": identity(manifest_path, root),
        "effective_raw_actual_http_attempt_budget": 15_200,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(run(args.root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


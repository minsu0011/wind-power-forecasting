#!/usr/bin/env python
"""Append-only seal for the completed-part crash-recovery raw runner head."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
ROOT_DEFAULT = REPO / "artifacts" / "baram2026_ncei_scada_longrun_20260810_v2"
EXPECTED = {
    "parent_manifest": "b2fc180579f239730e0ca01582fc6d3dfbb62113447a0a24eefca476326f38ad",
    "parent_preflight": "9e66a1fa68d5d89dbaf5fc4edd6caa7421106ca9a8fb1e3de4fd301d5b09d689",
    "runner": "52e7bbe4fcc4556a6c3001b679ad91685296b0a59957fc06de8010cd852334cd",
    "test": "df72d17c62f53cdf304785faad379dbf373e503cf7e87fbb007954c34bde6c45",
}


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
    paths = {
        "parent_manifest": root / "manifest_raw_preflight_v1.json",
        "parent_preflight": root / "prelaunch" / "RAW_RUNNER_STATIC_PREFLIGHT.json",
        "runner": REPO / "scripts" / "run_noaa_gfs_multiseason_raw_v2.py",
        "test": REPO / "tests" / "test_noaa_gfs_multiseason_raw_v2.py",
    }
    for key, expected in EXPECTED.items():
        if sha256_file(paths[key]) != expected:
            raise RuntimeError(f"raw preflight v2 input identity mismatch: {key}")
    amendment_path = root / "prelaunch" / "RAW_RUNNER_STATIC_PREFLIGHT_AMENDMENT_V2.json"
    manifest_path = root / "manifest_raw_preflight_v2.json"
    if amendment_path.exists() or manifest_path.exists():
        raise FileExistsError("append-only raw preflight v2 already exists")
    amendment = {
        "artifact_type": "RAW_RUNNER_STATIC_PREFLIGHT_APPEND_ONLY_AMENDMENT",
        "schema_version": 2,
        "status": "READY_FOR_INDEPENDENT_RUNNER_REVIEW_NOT_AUTHORIZED",
        "parent_manifest": identity(paths["parent_manifest"], root),
        "parent_preflight": identity(paths["parent_preflight"], root),
        "superseded_runner": json.loads(
            paths["parent_preflight"].read_text(encoding="utf-8")
        )["runner"],
        "corrected_runner": identity(paths["runner"]),
        "corrected_runner_test": identity(paths["test"]),
        "correction": {
            "crash_window": "payload .part was fully fsynced but process died before meta.part creation",
            "recovery": "Require exactly one matching durable RANGE_REQUEST_START/COMPLETE pair; validate event_id, attempt, URL, exact full range, bytes, payload SHA256, exact Content-Range total, ETag, Last-Modified and HTTP 206; reconstruct fsynced meta.part and finish the two atomic renames with network zero.",
            "ambiguous_missing_or_tampered_event": "STOP_FAIL_CLOSED",
            "synthetic_exact_crash_recovery_test": "PASS",
            "synthetic_tampered_payload_test": "PASS_FAIL_CLOSED",
        },
        "inherited_bindings_unchanged": {
            "census_manifest_sha256": "0269f6717b1110fafb6bba052c78824c616c648dbbdfe9f630947012c3f571a1",
            "field_range_census_sha256": "0623325476998ce173c9e6a809cdd67fa5413c27c7d9db09bf468d6b55936df3",
            "independent_prereg_sha256": "3605f88682060fe73faedad57852ee357bc6243489eabc00022533b4b81e5630",
            "independent_census_audit_sha256": "7fc717406e6e7587530cc922a16bf785b4471f99f0488c7483b2abc72890af68",
            "exact_ranges": 10_368,
            "exact_range_bytes": 10_296_112_890,
            "raw_actual_http_attempt_budget": 18_800,
        },
        "authorization_created": False,
        "independent_go_present": False,
        "raw_network_requests": 0,
        "labels_read": False,
    }
    write_json_exclusive(amendment_path, amendment)
    source = Path(__file__).resolve()
    test = REPO / "tests" / "test_noaa_gfs_multiseason_raw_preflight_amendment_v2.py"
    manifest = {
        "artifact_type": "RAW_RUNNER_STATIC_PREFLIGHT_MANIFEST",
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "parent_manifest": identity(paths["parent_manifest"], root),
        "amendment": identity(amendment_path, root),
        "corrected_runner": identity(paths["runner"]),
        "corrected_runner_test": identity(paths["test"]),
        "seal_code": identity(source),
        "seal_test": identity(test),
        "authorization_created": False,
        "raw_network_requests": 0,
        "labels_read": False,
    }
    write_json_exclusive(manifest_path, manifest)
    return {
        "amendment": identity(amendment_path, root),
        "manifest": identity(manifest_path, root),
        "runner": identity(paths["runner"]),
        "test": identity(paths["test"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(run(args.root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


#!/usr/bin/env python
"""Seal cumulative census access and static raw-runner closure before review."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
ROOT_DEFAULT = REPO / "artifacts" / "baram2026_ncei_scada_longrun_20260810_v2"
EXPECTED = {
    "census_manifest": "0269f6717b1110fafb6bba052c78824c616c648dbbdfe9f630947012c3f571a1",
    "field_ranges": "0623325476998ce173c9e6a809cdd67fa5413c27c7d9db09bf468d6b55936df3",
    "download_plan": "0ce241e4c9006b1f3710228fb2c50edb2ba02841cb8ca529f4795421d76c1423",
    "independent_prereg": "3605f88682060fe73faedad57852ee357bc6243489eabc00022533b4b81e5630",
    "independent_census_audit": "7fc717406e6e7587530cc922a16bf785b4471f99f0488c7483b2abc72890af68",
    "runner": "8618ea58dc9bc205fa7fc793bfa9e22522f368f7f15f70edf740341ee9d841a5",
    "runner_test": "75322357d06314ee4756143a130221a9d0467ae611791b0b7a52f3af138325bd",
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


def package_versions() -> dict[str, str]:
    result = {}
    for distribution in ("numpy", "pandas", "pyarrow", "eccodes", "pytest"):
        try:
            result[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            result[distribution] = "NOT_INSTALLED"
    return result


def imported_roots(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return sorted(roots)


def run(root: Path) -> dict[str, Any]:
    root = root.resolve()
    paths = {
        "census_manifest": root / "manifest_census_v1.json",
        "field_ranges": root / "census" / "field_range_census.parquet",
        "download_plan": root / "census" / "RAW_DOWNLOAD_PLAN_EXACT.json",
        "independent_prereg": root
        / "independent_redteam"
        / "TRACK_A_TARGET_FREE_AND_EXPANSION_PREREG.json",
        "independent_census_audit": root
        / "independent_redteam"
        / "TRACK_A_CENSUS_INDEPENDENT_AUDIT_V1.json",
        "runner": REPO / "scripts" / "run_noaa_gfs_multiseason_raw_v2.py",
        "runner_test": REPO / "tests" / "test_noaa_gfs_multiseason_raw_v2.py",
    }
    for name, expected in EXPECTED.items():
        if not paths[name].is_file() or sha256_file(paths[name]) != expected:
            raise RuntimeError(f"preflight input identity mismatch: {name}")
    outputs = {
        "access": root / "audit" / "CENSUS_CUMULATIVE_ACCESS_AMENDMENT.json",
        "incident": root
        / "audit"
        / "incidents"
        / "CENSUS_INITIALIZATION_PARSE_INCIDENT.json",
        "preflight": root / "prelaunch" / "RAW_RUNNER_STATIC_PREFLIGHT.json",
        "manifest": root / "manifest_raw_preflight_v1.json",
    }
    conflicts = [str(path) for path in outputs.values() if path.exists()]
    if conflicts:
        raise FileExistsError(f"no-overwrite preflight seal failed: {conflicts}")

    list_xml = sorted((root / "census" / "list_xml").glob("*.xml"))
    indexes = sorted((root / "census" / "index").rglob("*.idx"))
    checkpoints = sorted((root / "census" / "progress").glob("*.json"))
    if len(list_xml) != 48 or len(indexes) != 1_152:
        raise RuntimeError("census cached source count mismatch")
    final_checkpoint = json.loads(
        (root / "census" / "progress" / "idx_files_001152.json").read_text(
            encoding="utf-8"
        )
    )
    list_checkpoint = json.loads(
        (root / "census" / "progress" / "list_runs_000048.json").read_text(
            encoding="utf-8"
        )
    )
    list_bytes = sum(path.stat().st_size for path in list_xml)
    index_bytes = sum(path.stat().st_size for path in indexes)
    if (
        list_checkpoint["network_requests_this_invocation"] != 48
        or final_checkpoint["network_requests_this_invocation_so_far"] != 1_152
        or final_checkpoint["network_bytes_this_invocation_so_far"] != index_bytes
    ):
        raise RuntimeError("durable initial census checkpoints do not close access")
    access = {
        "artifact_type": "CENSUS_CUMULATIVE_ACCESS_APPEND_ONLY_AMENDMENT",
        "why_needed": "The final CENSUS_ACCESS_LEDGER describes the successful cache-reuse invocation (zero network), not the earlier invocation that populated the cache.",
        "initial_population_invocation": {
            "successful_listobjectsv2_requests": 48,
            "successful_listobjectsv2_payload_bytes": list_bytes,
            "successful_index_requests": 1_152,
            "successful_index_payload_bytes": index_bytes,
            "successful_http_requests": 1_200,
            "successful_payload_bytes": list_bytes + index_bytes,
            "durable_checkpoint_identities": [identity(path, root) for path in checkpoints],
        },
        "successful_cache_reuse_invocation": {
            "list_network_requests": 0,
            "index_network_requests": 0,
            "network_bytes": 0,
        },
        "phase_total": {
            "successful_http_requests": 1_200,
            "successful_payload_bytes": list_bytes + index_bytes,
            "raw_range_requests": 0,
            "raw_range_bytes": 0,
            "failed_transport_retry_attempts": "UNDETERMINED_NOT_INSTRUMENTED_BY_CENSUS_RUNNER",
            "transport_attempt_total_exact": False,
            "content_and_successful_request_total_exact": True,
        },
        "effect_on_science": "The retry-attempt logging gap does not change the exact cached object/index bytes, SHA/ETag publication census, field ranges or caps. The raw runner now records and globally caps every attempt before network access.",
        "labels_read": False,
        "2024_arrays_read": False,
        "2025_arrays_read": False,
        "raw_payload_read": False,
    }
    write_json_exclusive(outputs["access"], access)
    incident = {
        "artifact_type": "CENSUS_POSTDOWNLOAD_PARSE_INCIDENT",
        "occurred_after_all_metadata_and_indexes_were_durably_cached": True,
        "failure": "Expected GRIB initialization was incorrectly formed as 20220103T1 because the ISO T separator was not removed.",
        "effect": "The first invocation stopped before final census tables; it made zero raw-range requests.",
        "fix": "Parse the timezone-aware ISO instant and format with %Y%m%d%H; regression test expects 2022010312.",
        "resume": "The second invocation verified/reused all 48 ListObjectsV2 responses and 1,152 indexes with zero network requests, then sealed the census.",
        "current_census_runner": identity(
            REPO / "scripts" / "census_noaa_gfs_multiseason_v2.py"
        ),
        "census_manifest": identity(paths["census_manifest"], root),
        "raw_requests_before_and_after_incident": 0,
        "labels_read": False,
    }
    write_json_exclusive(outputs["incident"], incident)

    expected_outputs = [
        "raw/RAW_RANGE_MANIFEST.parquet",
        "raw/RAW_RANGE_MANIFEST.csv",
        "raw/RAW_ACCESS_LEDGER.json",
        "decoded/site_values_target_free.parquet",
        "decoded/group_values_target_free.parquet",
        "decoded/PHYSICAL_AND_COVERAGE_AUDIT.json",
        "decoded/DECODED_MATRIX_LOCK.json",
        "manifest_raw_v1.json",
    ]
    if any((root / relative).exists() for relative in expected_outputs):
        raise RuntimeError("a final raw/decode output exists before authorization")
    source_imports = imported_roots(paths["runner"])
    preflight = {
        "artifact_type": "RAW_RUNNER_STATIC_PREFLIGHT",
        "status": "READY_FOR_INDEPENDENT_RUNNER_REVIEW_NOT_AUTHORIZED",
        "runner": identity(paths["runner"]),
        "runner_test": identity(paths["runner_test"]),
        "ast_import_roots": source_imports,
        "local_import_closure": [],
        "runtime": {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "packages": package_versions(),
        },
        "bound_inputs": {
            key: identity(path, root if str(path).startswith(str(root)) else None)
            for key, path in paths.items()
            if key not in {"runner", "runner_test"}
        },
        "census_cumulative_access": identity(outputs["access"], root),
        "incident": identity(outputs["incident"], root),
        "expected_exact_ranges": 10_368,
        "expected_exact_range_bytes": 10_296_112_890,
        "census_successful_requests_already_spent_from_20k_cap": 1_200,
        "raw_actual_http_attempt_budget": 18_800,
        "concurrency": 8,
        "minimum_postdownload_free_disk_bytes": 200_000_000_000,
        "hardening": {
            "attempt_budget_reserved_before_each_request": True,
            "durable_start_complete_error_request_events": True,
            "exclusive_pid_attempt_launch_claim": True,
            "attempt_token_progress_paths": True,
            "fsynced_payload_and_meta_part_atomic_orphan_recovery": True,
            "exact_content_range_total_etag_last_modified": True,
            "exact_run_valid_variable_level_scanning_decode": True,
            "group_site_count_and_capacity_assertions": True,
        },
        "independent_go_required_path": "independent_redteam/TRACK_A_RAW_LAUNCH_GO.json",
        "independent_go_required_contract": {
            "status": "GO_RAW_RANGE_LAUNCH",
            "bound_authorization_sha256": "filled_after_independent_runner_PASS",
            "bound_runner_sha256": sha256_file(paths["runner"]),
            "bound_census_manifest_sha256": sha256_file(paths["census_manifest"]),
            "expected_range_rows": 10_368,
            "expected_range_bytes": 10_296_112_890,
        },
        "physically_bounded_predictor_future_blocker": {
            "required_before_target_free_duplicate_estimator": True,
            "full_single_row_group_weather_cache_read_forbidden": True,
            "required_proof": "physically bounded <=2023 official GFS source/prefix with exact bytes/SHA/rows/max timestamp and 2024 physical reads=0",
            "does_not_block_target_free_raw_decode": True,
        },
        "expected_outputs": expected_outputs,
        "all_expected_outputs_absent": True,
        "network_requests_by_this_seal": 0,
        "labels_read": False,
        "models_fit": 0,
        "submission_csv_created": False,
    }
    write_json_exclusive(outputs["preflight"], preflight)
    source = Path(__file__).resolve()
    test = REPO / "tests" / "test_noaa_gfs_multiseason_raw_preflight_v2.py"
    manifest = {
        "artifact_type": "RAW_RUNNER_STATIC_PREFLIGHT_MANIFEST",
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "census_manifest": identity(paths["census_manifest"], root),
        "cumulative_access_amendment": identity(outputs["access"], root),
        "incident": identity(outputs["incident"], root),
        "static_preflight": identity(outputs["preflight"], root),
        "runner": identity(paths["runner"]),
        "runner_test": identity(paths["runner_test"]),
        "seal_code": identity(source),
        "seal_test": identity(test),
        "raw_authorization_created": False,
        "raw_network_requests": 0,
        "labels_read": False,
    }
    write_json_exclusive(outputs["manifest"], manifest)
    return {
        "cumulative_access": identity(outputs["access"], root),
        "preflight": identity(outputs["preflight"], root),
        "manifest": identity(outputs["manifest"], root),
        "runner": identity(paths["runner"]),
        "status": preflight["status"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(run(args.root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


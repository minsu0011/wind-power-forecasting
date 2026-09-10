#!/usr/bin/env python
"""Seal the reviewed raw runner and a non-executable launch-auth draft."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
ROOT_DEFAULT = REPO / "artifacts" / "baram2026_ncei_scada_longrun_20260810_v2"
RUNNER_SHA = "51e6c4b68d47916f56975b16b9ceaae2e345f9f68963dba197501448d655934c"
TEST_SHA = "068ea40e2afab6b12421741229c14e73d71b7dda3e81df3ee9d0874f8bba2468"
PARENT_MANIFEST_SHA = "47ddc9df6e4346df20ed259de093e96a88dc2202424ff3f36ccceaff025d59a3"
EXPECTED_RANGE_ROWS = 10_368
EXPECTED_RANGE_BYTES = 10_296_112_890


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


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def load_runner_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("frozen_raw_runner_v4", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import reviewed raw runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(root: Path) -> dict[str, Any]:
    root = root.resolve()
    runner = REPO / "scripts" / "run_noaa_gfs_multiseason_raw_v2.py"
    runner_test = REPO / "tests" / "test_noaa_gfs_multiseason_raw_v2.py"
    seal_code = Path(__file__).resolve()
    seal_test = REPO / "tests" / "test_noaa_gfs_multiseason_raw_preflight_amendment_v4.py"
    parent_manifest = root / "manifest_raw_preflight_v2.json"
    amendment_path = root / "prelaunch" / "RAW_RUNNER_STATIC_PREFLIGHT_AMENDMENT_V4.json"
    draft_path = root / "prelaunch" / "RAW_LAUNCH_AUTHORIZATION_DRAFT_V1.json"
    manifest_path = root / "manifest_raw_preflight_v4.json"
    for path in (amendment_path, draft_path, manifest_path):
        if path.exists():
            raise FileExistsError(path)
    if sha256_file(runner) != RUNNER_SHA or sha256_file(runner_test) != TEST_SHA:
        raise RuntimeError("independently reviewed runner/test identity changed")
    if sha256_file(parent_manifest) != PARENT_MANIFEST_SHA:
        raise RuntimeError("parent raw-preflight v2 identity changed")

    paths = {
        "sampling_plan": root / "prereg" / "target_free_multiseason_sampling_plan_v1.json",
        "sampling_amendment": root / "prereg" / "target_free_multiseason_amendment_v2.json",
        "coordinate_lock": root / "prereg" / "authoritative_turbine_coordinate_lock_v1.json",
        "preregister_manifest_v1": root / "manifest_preregister_v1.json",
        "preregister_manifest_v2": root / "manifest_preregister_v2.json",
        "census_manifest": root / "manifest_census_v1.json",
        "field_range_census": root / "census" / "field_range_census.parquet",
        "raw_download_plan": root / "census" / "RAW_DOWNLOAD_PLAN_EXACT.json",
        "census_access": root / "audit" / "CENSUS_CUMULATIVE_ACCESS_AMENDMENT.json",
        "census_incident": root / "audit" / "incidents" / "CENSUS_INITIALIZATION_PARSE_INCIDENT.json",
        "previous_pilot_audit": root / "audit" / "PREVIOUS_PILOT_APPEND_ONLY_AUDIT.json",
        "redteam_preregister": root / "independent_redteam" / "TRACK_A_TARGET_FREE_AND_EXPANSION_PREREG.json",
        "redteam_census_audit": root / "independent_redteam" / "TRACK_A_CENSUS_INDEPENDENT_AUDIT_V1.json",
        "bounded_predictor_manifest": root / "independent_bounded_predictor" / "MANIFEST_V1.json",
        "raw_preflight_v1": root / "prelaunch" / "RAW_RUNNER_STATIC_PREFLIGHT.json",
        "raw_preflight_manifest_v1": root / "manifest_raw_preflight_v1.json",
        "raw_preflight_v2": root / "prelaunch" / "RAW_RUNNER_STATIC_PREFLIGHT_AMENDMENT_V2.json",
        "raw_preflight_manifest_v2": parent_manifest,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"required immutable provenance is absent: {missing}")

    raw_plan = json.loads(paths["raw_download_plan"].read_text(encoding="utf-8"))
    if (
        int(raw_plan.get("field_range_rows", -1)) != EXPECTED_RANGE_ROWS
        or int(raw_plan.get("exact_raw_range_bytes", -1)) != EXPECTED_RANGE_BYTES
    ):
        raise RuntimeError("raw download plan totals changed")
    census_access_payload = json.loads(paths["census_access"].read_text(encoding="utf-8"))
    if (
        census_access_payload.get("initial_population_invocation", {}).get(
            "successful_http_requests"
        )
        != 1_200
        or census_access_payload.get("phase_total", {}).get(
            "successful_http_requests"
        )
        != 1_200
        or census_access_payload.get("phase_total", {}).get("raw_range_requests") != 0
    ):
        raise RuntimeError("cumulative census-access facts changed")

    expected_absent = [
        "prereg/raw_launch_authorization_v1.json",
        "independent_redteam/TRACK_A_RAW_LAUNCH_GO.json",
        "raw",
        "decoded",
        "manifest_raw_v1.json",
        "raw/RAW_RANGE_MANIFEST.parquet",
        "raw/RAW_RANGE_MANIFEST.csv",
        "raw/RAW_ACCESS_LEDGER.json",
        "decoded/site_values_target_free.parquet",
        "decoded/group_values_target_free.parquet",
        "decoded/PHYSICAL_AND_COVERAGE_AUDIT.json",
        "decoded/DECODED_MATRIX_LOCK.json",
    ]
    present = [relative for relative in expected_absent if (root / relative).exists()]
    if present:
        raise RuntimeError(f"raw zero-output preflight failed: {present}")

    runner_module = load_runner_module(runner)
    runtime = runner_module.runtime_identity()
    source_chain = {name: identity(path, root) for name, path in paths.items()}
    amendment = {
        "artifact_type": "RAW_RUNNER_STATIC_PREFLIGHT_APPEND_ONLY_AMENDMENT",
        "schema_version": 4,
        "status": "INDEPENDENT_STATIC_REVIEW_PASS_AWAITING_GO_NOT_AUTHORIZED",
        "parent_manifest": identity(parent_manifest, root),
        "version_note": (
            "The provisional v3 sealer was code-only and never created v3 artifacts. "
            "This fresh v4 seal supersedes all stale runner/test constants without "
            "altering any v1/v2 artifact."
        ),
        "runner": identity(runner),
        "runner_test": identity(runner_test),
        "runtime_identity": runtime,
        "runtime_identity_sha256": canonical_sha256(runtime),
        "source_and_provenance_chain": source_chain,
        "attempt_caps": {
            "census_successful_logical_requests": 1_200,
            "census_attempts_per_logical_request_hard_max": 4,
            "census_worst_case_http_attempts": 4_800,
            "raw_required_successful_ranges": EXPECTED_RANGE_ROWS,
            "raw_actual_http_attempt_budget": 15_200,
            "raw_retry_headroom": 4_832,
            "census_plus_raw_max_http_attempts": 20_000,
            "proof": "1200*4 + 15200 = 20000",
        },
        "reviewed_closures": {
            "thread_safe_request_budget_before_urlopen": True,
            "at_most_8_in_flight_fail_fast_waves": True,
            "direct_sequential_event_probes_no_per_range_scandir": True,
            "attempt_15200_valid_and_attempt_15201_forbidden": True,
            "event_path_payload_attempt_global_identity_fail_closed": True,
            "prefix_suffix_mid_append_crash_recovery": True,
            "content_range_etag_last_modified_exact": True,
            "single_grib_exact_grid_run_valid_variable_level": True,
            "coordinate_and_runtime_authorization_binding": True,
            "cumulative_census_access_rehashed_before_launch_claim": True,
            "decode_and_physical_audit_before_canonical_transaction": True,
            "partial_transaction_recovery_no_overwrite": True,
            "start_projection_and_final_200gb_reserve": True,
            "windows_handle_and_fsync_regressions": True,
        },
        "expected_zero_output_inventory": {
            "checked_paths": expected_absent,
            "all_absent": True,
        },
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

    cumulative_binding = {
        **identity(paths["census_access"], root),
        "successful_http_requests": 1_200,
        "attempts_per_logical_request_hard_max": 4,
        "worst_case_http_attempts": 4_800,
    }
    proposed_authorization = {
        "artifact_type": "NOAA_GFS_MULTISEASON_RAW_LAUNCH_AUTHORIZATION",
        "schema_version": 1,
        "status": "AUTHORIZED_RAW_RANGE_LAUNCH_ONLY_AFTER_INDEPENDENT_GO",
        "runner": identity(runner),
        "runner_test": identity(runner_test),
        "field_range_census": identity(paths["field_range_census"], root),
        "census_manifest": identity(paths["census_manifest"], root),
        "coordinate_lock": identity(paths["coordinate_lock"], root),
        "census_cumulative_access": cumulative_binding,
        "runtime_identity": runtime,
        "runtime_identity_sha256": canonical_sha256(runtime),
        "effective_preflight_amendment": identity(amendment_path, root),
        "independent_target_free_preregister": identity(paths["redteam_preregister"], root),
        "independent_census_audit": identity(paths["redteam_census_audit"], root),
        "bounded_predictor_manifest": identity(paths["bounded_predictor_manifest"], root),
        "raw_actual_http_attempt_budget": 15_200,
        "census_worst_case_http_attempts": 4_800,
        "census_plus_raw_max_http_attempts": 20_000,
        "expected_range_rows": EXPECTED_RANGE_ROWS,
        "expected_range_bytes": EXPECTED_RANGE_BYTES,
        "post_download_free_disk_reserve_bytes": 200_000_000_000,
        "expected_zero_output_inventory_at_draft": expected_absent,
        "labels_read": False,
        "2024_arrays_read": False,
        "2025_arrays_read": False,
        "models_fit": 0,
        "submission_csv_created": False,
    }
    draft = {
        "artifact_type": "RAW_LAUNCH_AUTHORIZATION_DRAFT",
        "schema_version": 1,
        "status": "DRAFT_ONLY_NON_EXECUTABLE_AWAITING_INDEPENDENT_GO",
        "proposed_authorization_target": "prereg/raw_launch_authorization_v1.json",
        "proposed_authorization_payload": proposed_authorization,
        "independent_go_must_bind_final_authorization_sha": True,
        "actual_authorization_created": False,
        "independent_go_created": False,
        "raw_network_requests": 0,
    }
    write_json_exclusive(draft_path, draft)

    manifest = {
        "artifact_type": "RAW_RUNNER_STATIC_PREFLIGHT_MANIFEST",
        "schema_version": 4,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "parent_manifest": identity(parent_manifest, root),
        "amendment": identity(amendment_path, root),
        "authorization_draft": identity(draft_path, root),
        "runner": identity(runner),
        "runner_test": identity(runner_test),
        "seal_code": identity(seal_code),
        "seal_test": identity(seal_test),
        "effective_raw_actual_http_attempt_budget": 15_200,
        "census_worst_case_http_attempts": 4_800,
        "census_plus_raw_max_http_attempts": 20_000,
        "expected_zero_output_inventory_all_absent": True,
        "authorization_created": False,
        "independent_go_present": False,
        "raw_network_requests": 0,
        "raw_network_bytes": 0,
        "labels_read": False,
    }
    write_json_exclusive(manifest_path, manifest)
    return {
        "amendment": identity(amendment_path, root),
        "authorization_draft": identity(draft_path, root),
        "manifest": identity(manifest_path, root),
        "status": manifest["artifact_type"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(run(args.root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Append-only closure for Track A Phase-C artifacts and 3-hour feasibility.

The first Track-A run was intentionally provenance-only.  This script makes
the resulting NOT_RUN boundary explicit without changing any existing file.
It may perform at most one in-memory GRIB range replay for a network benchmark.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO = Path(__file__).resolve().parents[1]
ROOT_DEFAULT = (
    REPO
    / "artifacts"
    / "baram2026_ncei_scada_research_20260810_210756"
    / "track_a"
)
ORIGINAL_MANIFEST_SHA256 = "8b09d9493449bbaf9926794f014026286b9d3915cfa29c5d800bbbfe0a92bc47"
F039_URL = (
    "https://noaa-gfs-bdp-pds.s3.amazonaws.com/"
    "gfs.20230701/12/atmos/gfs.t12z.pgrb2.0p25.f039"
)
F039_RANGE = (519_964_299, 521_420_137)
F039_EXPECTED_SHA256 = "575fe7876ab93e5470c43dc09c78895385925f5662cca75bd0d8c7f91e677407"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path, root: Path | None = None) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix() if root is not None else str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_csv_exclusive(
    path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str]
) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json_exclusive(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def write_text_exclusive(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def benchmark_one_range() -> dict[str, Any]:
    request = urllib.request.Request(
        F039_URL,
        headers={
            "User-Agent": "baram2026-cutoff-provenance-audit/1.0 (one-range benchmark)",
            "Accept-Encoding": "identity",
            "Range": f"bytes={F039_RANGE[0]}-{F039_RANGE[1]}",
        },
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=45) as response:
        status = int(response.status)
        payload = response.read(F039_RANGE[1] - F039_RANGE[0] + 2)
        headers = {str(k).lower(): str(v).strip() for k, v in response.headers.items()}
    elapsed = time.perf_counter() - started
    expected_bytes = F039_RANGE[1] - F039_RANGE[0] + 1
    observed_sha = hashlib.sha256(payload).hexdigest()
    if status != 206 or len(payload) != expected_bytes or observed_sha != F039_EXPECTED_SHA256:
        raise RuntimeError(
            f"bounded range replay mismatch: status={status}, bytes={len(payload)}, sha={observed_sha}"
        )
    return {
        "performed": True,
        "requests": 1,
        "persisted_to_disk": False,
        "url": F039_URL,
        "range_start": F039_RANGE[0],
        "range_end": F039_RANGE[1],
        "bytes": len(payload),
        "sha256": observed_sha,
        "matches_original_pilot_raw": True,
        "http_status": status,
        "content_range": headers.get("content-range"),
        "elapsed_seconds": elapsed,
        "effective_bytes_per_second": len(payload) / elapsed,
    }


def decoder_audit() -> dict[str, Any]:
    modules = {
        name: importlib.util.find_spec(name) is not None
        for name in ("cfgrib", "eccodes", "pygrib", "xarray")
    }
    executables = {
        name: shutil.which(name)
        for name in ("wgrib2", "grib_get", "grib_ls")
    }
    operational_decoder_available = bool(
        modules["cfgrib"] and modules["eccodes"]
        or modules["pygrib"]
        or executables["wgrib2"]
    )
    return {
        "python_modules": modules,
        "executables": executables,
        "operational_grib_decoder_available": operational_decoder_available,
    }


def run(root: Path, perform_one_range_benchmark: bool) -> dict[str, Any]:
    root = root.resolve()
    audit = root / "audit"
    original_manifest = root / "manifest.json"
    original_decision = audit / "TRACK_A_DECISION.json"
    original_ledger = root / "provenance" / "GFS_PROVENANCE_LEDGER.csv"
    targets = {
        "duplicate": audit / "DUPLICATE_FEATURE_AUDIT.csv",
        "incremental": audit / "INCREMENTAL_INFORMATION.csv",
        "spatial": audit / "SPATIAL_FEATURE_AUDIT.csv",
        "vertical": audit / "VERTICAL_FEATURE_AUDIT.csv",
        "feasibility_json": audit / "HPBL_CAUSAL_FEASIBILITY.json",
        "feasibility_md": audit / "HPBL_CAUSAL_FEASIBILITY.md",
        "manifest_v2": root / "manifest_v2.json",
    }
    conflicts = [str(path) for path in targets.values() if path.exists()]
    if conflicts:
        raise FileExistsError(f"append-only preflight failed: {conflicts}")
    if sha256_file(original_manifest) != ORIGINAL_MANIFEST_SHA256:
        raise RuntimeError("original manifest identity changed; refusing amendment")
    decision = json.loads(original_decision.read_text(encoding="utf-8"))
    if any(
        (
            decision["competition_labels_read"],
            decision["model_fit_authorized"],
            decision["2024_selection_authorized"],
            decision["2025_test_arrays_read"],
            decision["submission_csv_authorized"],
        )
    ):
        raise RuntimeError("original provenance-only decision boundary changed")
    with original_ledger.open("r", encoding="utf-8", newline="") as stream:
        ledger = list(csv.DictReader(stream))
    if len(ledger) != 3 or any(row["status"] != "VERIFIED" for row in ledger):
        raise RuntimeError("original three-row provenance pilot is not VERIFIED")

    blank_metrics = {
        "aligned_row_count": 0,
        "pearson": "",
        "spearman": "",
        "mae": "",
        "rmse": "",
        "max_abs_diff": "",
        "unique_ratio": "",
        "linear_r2": "",
        "rank_equality": "",
    }
    duplicate_rows = [
        {
            "audit_scope": "DACON_35_WEATHER_FIELDS_VS_DMINUS2_SAME_SCHEMA_FIELDS",
            "dacon_feature_count": 35,
            "external_decoded_feature_count": 0,
            **blank_metrics,
            "status": "NOT_RUN",
            "determination": "UNDETERMINED",
            "reason_code": "PROVENANCE_ONLY_NO_VALID_TIME_ALIGNED_EXTERNAL_MATRIX",
            "model_use_allowed": False,
            "labels_read": False,
        }
    ]
    write_csv_exclusive(targets["duplicate"], duplicate_rows, list(duplicate_rows[0]))

    incremental_families = [
        ("BOUNDARY_LAYER_HEIGHT", "HPBL surface", True),
        ("NEAR_SURFACE_VERTICAL_WIND", "20/30/40/50 m UGRD/VGRD", False),
        ("HUB_HEIGHT_DENSITY", "80 m TMP/PRES and 100 m TMP", False),
        ("LOW_LEVEL_PRESSURE_PROFILE", "925/950/975/1000 mb HGT/TMP/RH/UGRD/VGRD", False),
    ]
    incremental_rows = [
        {
            "feature_family": family,
            "candidate_fields": fields,
            "present_in_2023_f039_index": True,
            "pilot_message_range_sha_bound": raw_bound,
            "decoded_valid_time_aligned_rows": 0,
            "cross_fitted_r2_external_from_dacon": "",
            "conditional_information_proxy": "",
            "residual_target_correlation": "",
            "status": "NOT_RUN",
            "determination": "UNDETERMINED",
            "reason_code": "PROVENANCE_ONLY_NO_DECODED_CAUSAL_FEATURE_MATRIX",
            "model_use_allowed": False,
            "labels_read": False,
        }
        for family, fields, raw_bound in incremental_families
    ]
    write_csv_exclusive(targets["incremental"], incremental_rows, list(incremental_rows[0]))

    spatial_families = [
        "CENTER_AND_8_NEIGHBORS",
        "WIND_SPEED_DIRECTION_GRADIENT",
        "UPWIND_DOWNWIND_SPEED",
        "HORIZONTAL_DIVERGENCE_PROXY",
        "PRESSURE_TEMPERATURE_GRADIENT",
    ]
    spatial_rows = [
        {
            "feature_family": family,
            "required_spatial_points": "center,N,S,E,W,NE,NW,SE,SW",
            "external_grid_messages_decoded": 0,
            "site_interpolation_verified": False,
            "causal_alignment_verified": False,
            "status": "NOT_RUN",
            "determination": "UNDETERMINED",
            "reason_code": "PROVENANCE_ONLY_HPBL_RANGE_NOT_DECODED_OR_SPATIALLY_SUBSET",
            "model_use_allowed": False,
            "labels_read": False,
        }
        for family in spatial_families
    ]
    write_csv_exclusive(targets["spatial"], spatial_rows, list(spatial_rows[0]))

    vertical_families = [
        ("NEAR_SURFACE_SHEAR", "10/20/30/40/50/80/100 m wind"),
        ("LOW_LEVEL_WIND_PROFILE", "1000/975/950/925/850 mb wind"),
        ("DIRECTIONAL_VEER", "multi-height and multi-pressure U/V"),
        ("VERTICAL_STABILITY", "80/100 m TMP/PRES and 1000/975/950/925 mb TMP/HGT"),
    ]
    vertical_rows = [
        {
            "feature_family": family,
            "required_fields": fields,
            "required_fields_index_presence_sampled": True,
            "required_fields_raw_downloaded": False,
            "decoded_valid_time_aligned_rows": 0,
            "status": "NOT_RUN",
            "determination": "UNDETERMINED",
            "reason_code": "PROVENANCE_ONLY_NO_VERTICAL_PAYLOAD_OR_DECODED_MATRIX",
            "model_use_allowed": False,
            "labels_read": False,
        }
        for family, fields in vertical_families
    ]
    write_csv_exclusive(targets["vertical"], vertical_rows, list(vertical_rows[0]))

    benchmark = (
        benchmark_one_range()
        if perform_one_range_benchmark
        else {"performed": False, "requests": 0, "persisted_to_disk": False}
    )
    decoder = decoder_audit()
    mean_hpbl = sum(int(row["message_bytes"]) for row in ledger) / len(ledger)
    preflight = json.loads(
        (root / "provenance" / "PREFLIGHT_DOWNLOAD_BUDGET.json").read_text(encoding="utf-8")
    )
    mean_idx = int(preflight["expected_index_bytes"]) / len(ledger)
    split_hours = {"2022_fit": 8_760, "2023_validation": 8_760, "2025_inference": 8_760}
    split_estimates = {
        split: {
            "hourly_objects": hours,
            "hpbl_range_bytes_estimate": int(round(hours * mean_hpbl)),
            "idx_bytes_estimate": int(round(hours * mean_idx)),
            "minimum_requests_head_or_list_plus_idx_plus_range": hours * 3,
            "pilot_style_requests_head_idxhead_idx_list_range": hours * 5,
        }
        for split, hours in split_hours.items()
    }
    total_hours = sum(split_hours.values())
    total_hpbl = int(round(total_hours * mean_hpbl))
    total_idx = int(round(total_hours * mean_idx))
    timing: dict[str, Any] = {"empirical_range_benchmark": benchmark}
    if benchmark["performed"]:
        throughput = float(benchmark["effective_bytes_per_second"])
        timing.update(
            {
                "payload_transfer_only_seconds_at_observed_single_range_throughput": total_hpbl
                / throughput,
                "sequential_hpbl_range_request_seconds_extrapolated": total_hours
                * float(benchmark["elapsed_seconds"]),
                "idealized_seven_lane_range_seconds_dividing_sequential_by_7": total_hours
                * float(benchmark["elapsed_seconds"])
                / 7,
                "timing_is_not_service_level_guarantee": True,
            }
        )
    blockers = [
        "Only three 2023 HPBL messages are provenance-verified; no daily 2022/2023/2025 coverage census exists.",
        "No decoded, site-subset, valid-time-aligned external feature matrix exists.",
        "Incremental-information and duplicate audits are NOT_RUN; score value is unproven.",
        "At least 26,280 hourly objects and 78,840 minimum HTTP requests are required for HPBL alone.",
        "No model fit or competition-label use is authorized by this provenance task.",
    ]
    if not decoder["operational_grib_decoder_available"]:
        blockers.insert(0, "No operational GRIB decoder is installed in the project environment.")
    feasibility = {
        "artifact_type": "HPBL_CAUSAL_FEASIBILITY_READ_ONLY_ESTIMATE",
        "policy": "D-2 12Z, 24 hourly valid times D01 through D+1 00 KST",
        "scope": split_estimates,
        "pilot_basis": {
            "verified_messages": len(ledger),
            "mean_hpbl_message_bytes": mean_hpbl,
            "mean_index_bytes": mean_idx,
            "estimate_warning": "Three-message byte-size extrapolation, not archive census.",
        },
        "totals": {
            "hourly_objects": total_hours,
            "hpbl_range_bytes_estimate": total_hpbl,
            "idx_bytes_estimate": total_idx,
            "hpbl_plus_idx_bytes_estimate": total_hpbl + total_idx,
            "minimum_http_requests": total_hours * 3,
            "pilot_style_http_requests": total_hours * 5,
        },
        "decoder": decoder,
        "timing": timing,
        "honest_model_possible_within_3_hours": False,
        "three_hour_verdict": "STOP_3H_HONEST_MODEL",
        "blockers": blockers,
        "network_range_requests_in_this_amendment": int(benchmark["requests"]),
        "network_range_bytes_in_this_amendment": int(benchmark.get("bytes", 0)),
        "extra_range_persisted": False,
        "bulk_download_launched": False,
        "labels_read": False,
        "model_fit": False,
        "submission_csv_created": False,
    }
    write_json_exclusive(targets["feasibility_json"], feasibility)
    feasibility_md = f"""# HPBL-only causal feasibility

**3-hour verdict: `STOP_3H_HONEST_MODEL`.** This does not reverse the bounded provenance GO; it stops bulk acquisition/modeling under the current three-hour constraint.

For 2022 fit + 2023 validation + 2025 inference, the D-2 12Z policy needs {total_hours:,} hourly objects. Based on the three verified pilot messages, HPBL ranges are estimated at {total_hpbl:,} bytes and indexes at {total_idx:,} bytes ({total_hpbl + total_idx:,} total). A cutoff-audited workflow needs at least {total_hours * 3:,} HTTP requests (exact object metadata, index, range), or {total_hours * 5:,} using the pilot's stronger HEAD/index-HEAD/index/List/range proof pattern.

Operational decoder available: **{decoder['operational_grib_decoder_available']}**. The optional benchmark performed exactly {benchmark['requests']} additional in-memory range request, persisted zero bytes, and matched the original raw SHA: {benchmark.get('matches_original_pilot_raw', 'not run')}.

The decisive blockers are not merely bandwidth: archive-wide cutoff completeness is unproven, no decoded/aligned feature matrix exists, and incremental information is `UNDETERMINED`. Launching a model now would not be an honest causal experiment.
"""
    write_text_exclusive(targets["feasibility_md"], feasibility_md)

    added = [
        targets[name]
        for name in (
            "duplicate",
            "incremental",
            "spatial",
            "vertical",
            "feasibility_json",
            "feasibility_md",
        )
    ]
    code_path = Path(__file__).resolve()
    test_path = REPO / "tests" / "test_noaa_gfs_track_a_phase_c_placeholders.py"
    manifest_v2 = {
        "artifact_type": "TRACK_A_APPEND_ONLY_AMENDMENT",
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "parent_manifest": identity(original_manifest, root),
        "parent_manifest_expected_sha256": ORIGINAL_MANIFEST_SHA256,
        "parent_manifest_unchanged": True,
        "reason": "Add prompt-mandatory Phase-C audit files as explicit NOT_RUN/UNDETERMINED placeholders and a read-only HPBL-only 3-hour feasibility estimate.",
        "added_artifacts": [identity(path, root) for path in added],
        "reproduction_code": identity(code_path),
        "test_code": identity(test_path),
        "network_range_requests_added": int(benchmark["requests"]),
        "network_range_bytes_added": int(benchmark.get("bytes", 0)),
        "existing_files_modified": [],
        "labels_read": False,
        "models_fit": 0,
        "bulk_download_launched": False,
        "submission_csv_created": False,
    }
    write_json_exclusive(targets["manifest_v2"], manifest_v2)
    return {
        "manifest_v2": identity(targets["manifest_v2"], root),
        "added_artifacts": manifest_v2["added_artifacts"],
        "three_hour_verdict": feasibility["three_hour_verdict"],
        "decoder_available": decoder["operational_grib_decoder_available"],
        "extra_range_requests": benchmark["requests"],
        "extra_range_bytes": benchmark.get("bytes", 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--one-range-benchmark", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.root, perform_one_range_benchmark=args.one_range_benchmark),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


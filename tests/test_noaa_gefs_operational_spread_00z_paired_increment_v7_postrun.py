from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "artifacts/external/noaa_gefs_operational_spread_00z_original_v7/preflight_one_range"
SOURCE = ROOT / "artifacts/external/noaa_gefs_operational_spread_00z_original_v7/stage1_source_through_operating_2023"
STAGE2_SOURCE = ROOT / "artifacts/external/noaa_gefs_operational_spread_00z_original_v7/stage2_source_operating_2024"
MODEL_OUTPUT = ROOT / "artifacts/postgate/noaa_gefs_operational_spread_00z_paired_increment_strict_v7"
SOURCE_LAUNCH_LOCK = ROOT / "artifacts/audits/noaa_gefs_operational_spread_00z_original_v7_stage1_launch_lock.json"
POST_PREFLIGHT_REVIEW = ROOT / "artifacts/audits/noaa_gefs_operational_spread_00z_original_v7_preflight_postrun_independent_review_v1.json"
OLD_SEAL = ROOT / "artifacts/incidents/noaa_gefs_operational_spread_00z_original_v2_stage1_protocol_failure_seal_v1.json"
OLD_SEAL_SHA = "89d114289de2a7d3c6793c243955330cdb6285ea028ded9275ab258086592a88"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_old_terminal_incident_is_immutable() -> None:
    assert sha256(OLD_SEAL) == OLD_SEAL_SHA


def test_preflight_canonical_if_present() -> None:
    if not PREFLIGHT.exists():
        pytest.skip("v7 live preflight has not run")
    manifest_path = PREFLIGHT / "manifest.json"
    result_path = PREFLIGHT / "preflight_result.json"
    response_path = PREFLIGHT / "response.grib2"
    assert manifest_path.is_file() and result_path.is_file() and response_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert manifest["status"] == result["status"] == "PASS_ONE_RANGE_NEW_DECODER_EQUALS_REAL_FILE_REFERENCE"
    assert result["request_accounting"] == {"requests": 1, "response_bytes": 464970, "retries": 0}
    assert result["exact_metadata_and_value_equality"] is True
    assert result["full_extraction_payload_reuse_allowed"] is False
    assert response_path.stat().st_size == 464970
    assert sha256(response_path) == result["response"]["sha256"]
    assert manifest["payload_reuse_by_full_extraction"] == 0


def test_stage1_source_manifest_and_artifact_closure_if_present() -> None:
    if not SOURCE.exists():
        pytest.skip("v7 full Stage1 source has not run")
    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((SOURCE / "extraction_summary.json").read_text(encoding="utf-8"))
    assert manifest["status"] == summary["status"] == "PASS_V7_STAGE1_SOURCE_EXTRACTION"
    assert manifest["nonmutation"] == {"stage2_namespace_created": False, "label_model_metric_csv_writes": 0, "old_v2_partial_reused": 0}
    assert summary["scope"]["days"] == 730
    assert summary["scope"]["ranges"] == 26280
    assert summary["scope"]["raw_bytes"] == 11524284739
    assert summary["request_accounting"]["preflight_payload_reused"] == 0
    assert summary["request_accounting"]["operating_2024_grib_range_gets_or_values"] == 0
    assert summary["request_accounting"]["2025_requests_or_values"] == 0
    for record in manifest["artifacts"]:
        path = ROOT / record["path"]
        assert path.is_file() and path.stat().st_size == int(record["bytes"])
        assert sha256(path) == record["sha256"]


def test_full_launch_lock_requires_distinct_post_preflight_review_if_present() -> None:
    if not SOURCE_LAUNCH_LOCK.exists():
        pytest.skip("v7 full launch lock has not been frozen")
    assert POST_PREFLIGHT_REVIEW.is_file()
    lock = json.loads(SOURCE_LAUNCH_LOCK.read_text(encoding="utf-8"))
    review = json.loads(POST_PREFLIGHT_REVIEW.read_text(encoding="utf-8"))
    assert review["verdict"] == "PASS"
    assert review["preflight_network_requests"] == 1
    assert review["full_stage1_network_value_label_fit_score_access"] == 0
    assert lock["post_preflight_independent_review"]["sha256"] == sha256(POST_PREFLIGHT_REVIEW)


def test_stage1_response_manifest_exact_counts_if_present() -> None:
    path = SOURCE / "range_response_manifest.csv.gz"
    if not path.is_file():
        pytest.skip("v7 full Stage1 source has not run")
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        frame = pd.read_csv(handle, usecols=["operating_date", "source_date", "byte_count", "attempts", "data_key"])
    assert len(frame) == 26280
    assert int(frame["byte_count"].sum()) == 11524284739
    assert frame["operating_date"].min() == "2022-01-01"
    assert frame["operating_date"].max() == "2023-12-31"
    assert frame["source_date"].min() == "2021-12-30"
    assert frame["source_date"].max() == "2023-12-29"
    assert not frame["data_key"].str.contains("gefs.2024|gefs.2025", regex=True).any()
    assert (frame["attempts"] >= 1).all()


def test_hourly_cache_physical_boundary_if_present() -> None:
    path = SOURCE / "hourly_components.parquet"
    if not path.is_file():
        pytest.skip("v7 full Stage1 source has not run")
    frame = pd.read_parquet(path)
    assert len(frame) == 17520
    assert frame["forecast_kst_dtm"].min() == "2022-01-01 01:00:00"
    assert frame["forecast_kst_dtm"].max() == "2024-01-01 00:00:00"
    assert frame["operating_date"].min() == "2022-01-01"
    assert frame["operating_date"].max() == "2023-12-31"
    assert set(frame["missing_gefs_00z"]) == {0}


def test_no_conditional_source_or_model_before_source_gate() -> None:
    assert not STAGE2_SOURCE.exists()
    if not SOURCE.exists():
        assert not MODEL_OUTPUT.exists()

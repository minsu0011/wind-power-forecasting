from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts import census_noaa_gfs_multiseason_v2 as census


ROOT = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "baram2026_ncei_scada_longrun_20260810_v2"
)


def test_compact_initialization_matches_grib_index_format() -> None:
    assert census.compact_initialization("2022-01-03T12:00:00Z") == "2022010312"


def test_exact_download_plan_is_bounded_and_target_free() -> None:
    path = ROOT / "census" / "RAW_DOWNLOAD_PLAN_EXACT.json"
    if not path.is_file():
        pytest.skip("post-census artifacts not present")
    plan = json.loads(path.read_text(encoding="utf-8"))
    assert plan["object_rows"] == 1_152
    assert plan["field_range_rows"] == 10_368
    assert plan["exact_index_bytes"] <= 100_000_000
    assert plan["exact_raw_range_bytes"] <= 20_000_000_000
    assert plan["total_requests_census_plus_raw"] <= 20_000
    assert all(plan["caps"].values())
    assert plan["all_1_152_objects_publication_verified"] is True
    assert plan["all_10_368_field_ranges_census_verified"] is True
    assert plan["raw_launch_executed"] is False
    assert plan["labels_read"] is False


def test_each_locked_field_occurs_once_in_every_index() -> None:
    path = ROOT / "census" / "FIELD_CENSUS_SUMMARY.csv"
    if not path.is_file():
        pytest.skip("post-census artifacts not present")
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 9
    assert all(row["expected_occurrences"] == "1152" for row in rows)
    assert all(row["observed_unique_occurrences"] == "1152" for row in rows)
    assert all(float(row["presence_fraction"]) == 1.0 for row in rows)
    assert all(row["status"] == "PASS" for row in rows)


def test_each_frozen_day_has_24_objects_216_ranges_and_positive_margin() -> None:
    path = ROOT / "census" / "DAY_CENSUS_SUMMARY.csv"
    if not path.is_file():
        pytest.skip("post-census artifacts not present")
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 48
    assert all(row["object_count"] == "24" for row in rows)
    assert all(row["field_range_count"] == "216" for row in rows)
    assert all(int(row["minimum_cutoff_margin_seconds"]) > 0 for row in rows)
    assert all(row["status"] == "PASS" for row in rows)


def test_census_manifest_hash_closes_outputs_and_raw_access_is_zero() -> None:
    path = ROOT / "manifest_census_v1.json"
    if not path.is_file():
        pytest.skip("post-census artifacts not present")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["raw_network_requests"] == 0
    assert manifest["raw_downloaded_bytes"] == 0
    assert manifest["labels_read"] is False
    for key in (
        "parent_preregister_manifest",
        "object_census",
        "field_range_census",
        "field_summary",
        "day_summary",
        "access_ledger",
        "raw_download_plan",
        "summary",
    ):
        record = manifest[key]
        payload = (ROOT / record["path"]).read_bytes()
        assert len(payload) == record["size_bytes"]
        assert hashlib.sha256(payload).hexdigest() == record["sha256"]


def test_index_partial_download_resumes_and_final_reuse_is_network_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"0123456789abcdef"
    key = "gfs.20220103/12/atmos/gfs.t12z.pgrb2.0p25.f028.idx"
    record = {
        "idx_key": key,
        "run_init_utc": "2022-01-03T12:00:00Z",
        "forecast_hour": 28,
        "idx_size_bytes": len(payload),
        "idx_etag": hashlib.md5(payload, usedforsecurity=False).hexdigest(),
    }
    part = (
        tmp_path
        / "census"
        / "index"
        / "gfs.20220103"
        / "12"
        / "f028.idx.part"
    )
    part.parent.mkdir(parents=True)
    part.write_bytes(payload[:5])
    calls = []

    def fake_fetch(url: str, *, range_start: int | None, max_bytes: int, attempts: int = 4):
        calls.append((url, range_start, max_bytes))
        assert range_start == 5
        assert max_bytes == len(payload) - 5
        return payload[5:], {"content-range": f"bytes 5-{len(payload)-1}/{len(payload)}"}, 206, "2026-08-10T00:00:00Z"

    monkeypatch.setattr(census, "fetch", fake_fetch)
    first = census.get_or_resume_idx(tmp_path, record)
    assert first["network"] is True
    assert first["network_bytes"] == len(payload) - 5
    assert first["path"].read_bytes() == payload
    second = census.get_or_resume_idx(tmp_path, record)
    assert second["network"] is False
    assert second["network_bytes"] == 0
    assert len(calls) == 1

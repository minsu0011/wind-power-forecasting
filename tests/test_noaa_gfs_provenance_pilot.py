from __future__ import annotations

import csv
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from src.noaa_gfs_provenance import (
    conservative_run_init_utc,
    cutoff_utc,
    find_unique_record,
    forecast_hour,
    object_key,
    operating_valid_times_utc,
    parse_grib_index,
    publication_status,
    record_byte_range,
)


ARTIFACT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "baram2026_ncei_scada_research_20260810_210756"
    / "track_a"
)


def test_frozen_cutoff_and_dminus2_run_are_exact_utc_instants() -> None:
    target = date(2023, 7, 3)
    assert cutoff_utc(target) == datetime(2023, 7, 2, 5, tzinfo=timezone.utc)
    assert conservative_run_init_utc(target) == datetime(
        2023, 7, 1, 12, tzinfo=timezone.utc
    )


def test_operating_day_maps_to_exact_forecast_hours_28_through_51() -> None:
    target = date(2023, 7, 3)
    run = conservative_run_init_utc(target)
    valid = operating_valid_times_utc(target)
    assert len(valid) == 24
    assert [forecast_hour(run, value) for value in valid] == list(range(28, 52))
    assert valid[0] == datetime(2023, 7, 2, 16, tzinfo=timezone.utc)
    assert valid[-1] == datetime(2023, 7, 3, 15, tzinfo=timezone.utc)


def test_object_identity_is_exact_operational_gfs_key() -> None:
    run = datetime(2023, 7, 1, 12, tzinfo=timezone.utc)
    assert object_key(run, 39) == (
        "gfs.20230701/12/atmos/gfs.t12z.pgrb2.0p25.f039"
    )
    with pytest.raises(ValueError):
        object_key(datetime(2023, 7, 1, 13, tzinfo=timezone.utc), 39)


def test_idx_parser_and_byte_range_are_fail_closed() -> None:
    text = (
        "1:0:d=2023070112:TMP:2 m above ground:39 hour fcst:\n"
        "2:100:d=2023070112:HPBL:surface:39 hour fcst:\n"
        "3:250:d=2023070112:UGRD:20 m above ground:39 hour fcst:\n"
    )
    records = parse_grib_index(text)
    hpbl = find_unique_record(records, "HPBL", "surface")
    assert record_byte_range(records, hpbl, object_size=400).start == 100
    assert record_byte_range(records, hpbl, object_size=400).end == 249
    assert record_byte_range(records, hpbl, object_size=400).length == 150
    with pytest.raises(ValueError):
        parse_grib_index("2:0:d=2023070112:HPBL:surface:39 hour fcst:\n")


def test_publication_gate_requires_run_time_and_bound_raw() -> None:
    cutoff = datetime(2023, 7, 2, 5, tzinfo=timezone.utc)
    before = datetime(2023, 7, 1, 15, 40, tzinfo=timezone.utc)
    after = datetime(2023, 7, 2, 5, 1, tzinfo=timezone.utc)
    good_sha = "a" * 64
    common = dict(
        exact_object_key_observed=True,
        cutoff=cutoff,
        raw_range_sha256=good_sha,
        raw_range_bytes=100,
        range_matches_index=True,
    )
    assert publication_status(last_modified=before, **common) == "VERIFIED"
    assert publication_status(last_modified=after, **common) == "AFTER_CUTOFF"
    assert (
        publication_status(last_modified=None, **common)
        == "MISSING_PUBLICATION_EVIDENCE"
    )
    assert (
        publication_status(last_modified=before, **{**common, "raw_range_sha256": None})
        == "MISSING_RAW"
    )
    assert (
        publication_status(
            last_modified=before,
            **{**common, "exact_object_key_observed": False},
        )
        == "MISSING_RUN_ID"
    )


def test_duplicate_hpbl_index_is_rejected() -> None:
    records = parse_grib_index(
        "1:0:d=2023070112:HPBL:surface:39 hour fcst:\n"
        "2:100:d=2023070112:HPBL:surface:39 hour fcst:\n"
    )
    with pytest.raises(ValueError, match="observed 2"):
        find_unique_record(records, "HPBL", "surface")


def test_persisted_pilot_has_no_after_cutoff_row() -> None:
    ledger_path = ARTIFACT_ROOT / "provenance" / "GFS_PROVENANCE_LEDGER.csv"
    if not ledger_path.is_file():
        pytest.skip("postrun Track A artifact is not present")
    with ledger_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    assert {row["forecast_hour"] for row in rows} == {"28", "39", "51"}
    assert all(row["status"] == "VERIFIED" for row in rows)
    assert all(row["metadata_head_list_agree"] == "True" for row in rows)
    assert all(int(row["cutoff_margin_seconds"]) > 0 for row in rows)
    assert all(
        datetime.fromisoformat(row["publication_last_modified_utc"].replace("Z", "+00:00"))
        <= datetime.fromisoformat(row["cutoff_utc"].replace("Z", "+00:00"))
        for row in rows
    )


def test_persisted_raw_sha_manifest_replays_exactly() -> None:
    manifest_path = ARTIFACT_ROOT / "provenance" / "RAW_MANIFEST_SHA256.csv"
    if not manifest_path.is_file():
        pytest.skip("postrun Track A artifact is not present")
    with manifest_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows
    for row in rows:
        path = ARTIFACT_ROOT / row["relative_path"]
        assert path.is_file(), row["relative_path"]
        payload = path.read_bytes()
        assert len(payload) == int(row["size_bytes"])
        assert hashlib.sha256(payload).hexdigest() == row["sha256"]


def test_persisted_pilot_stays_under_raw_cap_and_contains_only_message_ranges() -> None:
    preflight_path = ARTIFACT_ROOT / "provenance" / "PREFLIGHT_DOWNLOAD_BUDGET.json"
    if not preflight_path.is_file():
        pytest.skip("postrun Track A artifact is not present")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    assert preflight["checked_before_grib_payload_download"] is True
    assert preflight["cap_pass"] is True
    assert preflight["expected_bounded_raw_total_bytes"] <= 8_000_000
    assert preflight["full_global_objects_downloaded"] == 0
    raw_files = sorted((ARTIFACT_ROOT / "provenance" / "raw").rglob("*.grib2"))
    assert len(raw_files) == 3
    assert all("HPBL_surface" in path.name for path in raw_files)
    assert sum(path.stat().st_size for path in raw_files) < 8_000_000


def test_track_a_decision_does_not_authorize_fit_bulk_download_or_csv() -> None:
    decision_path = ARTIFACT_ROOT / "audit" / "TRACK_A_DECISION.json"
    if not decision_path.is_file():
        pytest.skip("postrun Track A artifact is not present")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["decision"] == "PROCEED_NCEI_GFS"
    assert decision["go_stop"] == "GO_BOUNDED_PILOT_ONLY"
    assert decision["full_acquisition_authorized"] is False
    assert decision["model_fit_authorized"] is False
    assert decision["competition_labels_read"] is False
    assert decision["2024_selection_authorized"] is False
    assert decision["submission_csv_authorized"] is False

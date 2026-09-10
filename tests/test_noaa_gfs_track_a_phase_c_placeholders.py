from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest


ROOT = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "baram2026_ncei_scada_research_20260810_210756"
    / "track_a"
)


def _rows(name: str) -> list[dict[str, str]]:
    path = ROOT / "audit" / name
    if not path.is_file():
        pytest.skip("append-only Track A amendment not present")
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


@pytest.mark.parametrize(
    "name",
    [
        "DUPLICATE_FEATURE_AUDIT.csv",
        "INCREMENTAL_INFORMATION.csv",
        "SPATIAL_FEATURE_AUDIT.csv",
        "VERTICAL_FEATURE_AUDIT.csv",
    ],
)
def test_phase_c_placeholder_is_fail_closed(name: str) -> None:
    rows = _rows(name)
    assert rows
    assert all(row["status"] == "NOT_RUN" for row in rows)
    assert all(row["determination"] == "UNDETERMINED" for row in rows)
    assert all(row["model_use_allowed"] == "False" for row in rows)
    assert all(row["labels_read"] == "False" for row in rows)


def test_hpbl_feasibility_stops_three_hour_model_and_bulk_download() -> None:
    path = ROOT / "audit" / "HPBL_CAUSAL_FEASIBILITY.json"
    if not path.is_file():
        pytest.skip("append-only Track A amendment not present")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["three_hour_verdict"] == "STOP_3H_HONEST_MODEL"
    assert report["honest_model_possible_within_3_hours"] is False
    assert report["totals"]["hourly_objects"] == 26_280
    assert report["totals"]["minimum_http_requests"] == 78_840
    assert report["network_range_requests_in_this_amendment"] <= 1
    assert report["extra_range_persisted"] is False
    assert report["bulk_download_launched"] is False
    assert report["labels_read"] is False
    assert report["model_fit"] is False
    assert report["submission_csv_created"] is False


def test_manifest_v2_is_append_only_and_hash_closes_added_artifacts() -> None:
    path = ROOT / "manifest_v2.json"
    if not path.is_file():
        pytest.skip("append-only Track A amendment not present")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["parent_manifest"]["sha256"] == (
        "8b09d9493449bbaf9926794f014026286b9d3915cfa29c5d800bbbfe0a92bc47"
    )
    assert manifest["existing_files_modified"] == []
    assert manifest["network_range_requests_added"] <= 1
    for record in manifest["added_artifacts"]:
        artifact = ROOT / record["path"]
        payload = artifact.read_bytes()
        assert len(payload) == record["size_bytes"]
        assert hashlib.sha256(payload).hexdigest() == record["sha256"]


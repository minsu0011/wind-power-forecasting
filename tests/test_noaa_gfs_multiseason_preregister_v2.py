from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pytest


ROOT = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "baram2026_ncei_scada_longrun_20260810_v2"
)


def _plan() -> dict:
    path = ROOT / "prereg" / "target_free_multiseason_sampling_plan_v1.json"
    if not path.is_file():
        pytest.skip("multiseason preregistration not generated")
    return json.loads(path.read_text(encoding="utf-8"))


def test_dates_are_frozen_two_per_month_2022_2023() -> None:
    plan = _plan()
    expected = [
        date(year, month, day).isoformat()
        for year in (2022, 2023)
        for month in range(1, 13)
        for day in (5, 20)
    ]
    assert plan["operating_days_kst"] == expected
    assert len(expected) == 48
    assert plan["hourly_object_count"] == 1_152
    assert plan["forecast_hours"] == list(range(28, 52))


def test_minimal_candidate_allowlist_excludes_interpolation_duplicates() -> None:
    plan = _plan()
    selectors = [
        (row["variable"], row["level"])
        for family in plan["candidate_families"]
        for row in family["selectors"]
    ]
    assert selectors == [
        ("HPBL", "surface"),
        ("UGRD", "925 mb"),
        ("VGRD", "925 mb"),
        ("UGRD", "950 mb"),
        ("VGRD", "950 mb"),
        ("UGRD", "975 mb"),
        ("VGRD", "975 mb"),
        ("UGRD", "1000 mb"),
        ("VGRD", "1000 mb"),
    ]
    assert not any(level.startswith(("20 ", "30 ", "40 ", "50 ")) for _, level in selectors)


def test_prereg_is_target_free_and_hard_bounded() -> None:
    plan = _plan()
    assert all(plan["forbidden"].values())
    contract = plan["download_contract"]
    assert contract["raw_mode"].startswith("HTTP Range only")
    assert contract["complete_global_object_download_forbidden"] is True
    assert contract["no_overwrite"] is True
    assert 6 <= contract["concurrency"] <= 8
    assert contract["max_raw_payload_bytes"] == 20_000_000_000
    assert contract["max_total_http_requests"] == 20_000


def test_preregister_manifest_hash_closes_plan_and_previous_audit() -> None:
    path = ROOT / "manifest_preregister_v1.json"
    if not path.is_file():
        pytest.skip("multiseason preregistration not generated")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for key in ("sampling_plan", "previous_pilot_audit", "previous_pilot_audit_markdown"):
        record = manifest[key]
        payload = (ROOT / record["path"]).read_bytes()
        assert len(payload) == record["size_bytes"]
        assert hashlib.sha256(payload).hexdigest() == record["sha256"]
    assert manifest["index_files_read"] == 0
    assert manifest["network_requests"] == 0
    assert manifest["labels_read"] is False


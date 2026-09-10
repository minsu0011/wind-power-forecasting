from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.download_noaa_isd_synop_lags import cutoff_summary, fetch, normalize_synop


def row(date: str, *, report_type: str = "FM-12", wnd: str = "090,1,N,0010,1") -> dict[str, object]:
    return {
        "STATION": "47100099999",
        "DATE": date,
        "SOURCE": 4,
        "LATITUDE": 37.683,
        "LONGITUDE": 128.717,
        "ELEVATION": 844.0,
        "NAME": "DAEGWALLYEONG, KS",
        "REPORT_TYPE": report_type,
        "QUALITY_CONTROL": "V020",
        "WND": wnd,
        "TMP": "+0010,1",
        "SLP": "10123,1",
        "MA1": "99999,9,09369,1",
    }


def test_normalize_retains_only_fixed_land_synop_and_converts_units() -> None:
    raw = pd.DataFrame([
        row("2022-01-01T00:00:00"),
        row("2022-01-01T01:00:00", report_type="FM-15"),
    ])
    normalized, diagnostics = normalize_synop(raw, "47100099999")
    assert len(normalized) == 1
    assert diagnostics["report_type_counts"] == {"FM-12": 1, "FM-15": 1}
    assert normalized.loc[0, "wind_speed_ms"] == 1.0
    assert normalized.loc[0, "temperature_c"] == 1.0
    assert normalized.loc[0, "sea_level_pressure_hpa"] == 1012.3
    assert normalized.loc[0, "station_pressure_hpa"] == 936.9
    assert normalized.loc[0, "time_kst"] == pd.Timestamp("2022-01-01 09:00")


def test_strict_qc_excludes_suspect_or_ncei_origin_codes() -> None:
    raw = pd.DataFrame([
        row("2022-01-01T00:00:00", wnd="090,2,N,0010,5"),
    ])
    normalized, _ = normalize_synop(raw, "47100099999")
    assert np.isnan(normalized.loc[0, "wind_speed_ms"])
    assert np.isnan(normalized.loc[0, "wind_direction_deg"])


def test_cutoff_window_is_left_closed_and_strictly_right_open() -> None:
    raw = pd.DataFrame([
        row("2022-01-01T05:00:00"),  # 2022-01-01 14:00 KST, included for D=Jan 3
        row("2022-01-02T04:00:00"),  # 2022-01-02 13:00 KST, included
        row("2022-01-02T05:00:00"),  # exact 14:00 cutoff, excluded
    ])
    normalized, _ = normalize_synop(raw, "47100099999")
    summary = cutoff_summary(normalized, "daegwallyeong")
    jan3 = summary.loc[summary["decision_date"].eq(pd.Timestamp("2022-01-03"))].iloc[0]
    assert jan3["daegwallyeong__report_count_24h"] == 2
    assert jan3["daegwallyeong__latest_age_h"] == 1.0


def test_network_cap_rejects_2025_observation_path_before_request() -> None:
    with pytest.raises(AssertionError, match="calendar 2025"):
        fetch("https://www.ncei.noaa.gov/data/global-hourly/access/2025/47100099999.csv")

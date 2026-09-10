from __future__ import annotations

import numpy as np
import pandas as pd

from src.multi_nwp_joint import JOINT_COLUMNS, SOURCE_ORDER, build_joint_features, paired_transfer


def _source(index: pd.DatetimeIndex, heights: tuple[int, ...], offset: float) -> pd.DataFrame:
    frame = pd.DataFrame(index=index)
    for height in heights:
        for day in (1, 2):
            frame[f"wind_speed_{height}m_previous_day{day}"] = offset + height / 100 + day
            frame[f"wind_direction_{height}m_previous_day{day}"] = 90.0 * day
    return frame


def test_joint_schema_cutoff_and_finiteness() -> None:
    index = pd.DatetimeIndex(["2024-06-01 00:00", "2024-06-01 01:00", "2024-06-01 13:00", "2024-06-01 14:00"], name="forecast_kst_dtm")
    sources = {
        "ecmwf": _source(index, (100,), 1.0),
        "icon": _source(index, (80, 120), 2.0),
        "gfs": _source(index, (80, 100), 3.0),
    }
    result = build_joint_features(sources, pd.Series(5.0, index=index))
    assert tuple(sources) == SOURCE_ORDER
    assert tuple(result.columns) == JOINT_COLUMNS
    assert result.shape == (4, 30)
    assert np.isfinite(result.to_numpy()).all()
    # 00 and 14 use the 48-hour source; 01--13 use the 24-hour source.
    assert result.iloc[0]["mnwp__ecmwf_ws100"] == result.iloc[3]["mnwp__ecmwf_ws100"]
    assert result.iloc[1]["mnwp__ecmwf_ws100"] == result.iloc[2]["mnwp__ecmwf_ws100"]
    assert result.iloc[0]["mnwp__ecmwf_ws100"] != result.iloc[1]["mnwp__ecmwf_ws100"]


def test_paired_transfer_is_bounded_and_zero_increment_is_identity() -> None:
    index = pd.date_range("2024-07-01", periods=3, freq="h")
    baseline = pd.Series([100.0, 200.0, 300.0], index=index, name="g")
    control = pd.Series([0.2, 0.2, 0.2], index=index)
    identity = paired_transfer(baseline, control, control.copy(), capacity_kwh=1000.0, transfer_weight=0.25)
    np.testing.assert_array_equal(identity.to_numpy(), baseline.to_numpy())
    extended = pd.Series([1.2, -1.0, 0.4], index=index)
    changed = paired_transfer(baseline, control, extended, capacity_kwh=1000.0, transfer_weight=0.25)
    assert ((changed >= 0.0) & (changed <= 1020.0)).all()

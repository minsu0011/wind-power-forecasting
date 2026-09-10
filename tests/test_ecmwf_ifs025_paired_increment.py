from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.run_ecmwf_ifs025_paired_increment import (
    CANDIDATES,
    acquire_heavy_guard,
    add_disagreement,
    apply_paired_increment,
    pid_is_alive,
    release_heavy_guard,
    select_external_features,
)


def external_frame() -> pd.DataFrame:
    index = pd.DatetimeIndex(
        ["2024-07-01 00:00", "2024-07-01 01:00", "2024-07-01 13:00", "2024-07-01 14:00"],
        name="forecast_kst_dtm",
    )
    return pd.DataFrame(
        {
            "wind_speed_100m_previous_day1": [1.0, 2.0, 3.0, 4.0],
            "wind_direction_100m_previous_day1": [0.0, 90.0, 180.0, 270.0],
            "wind_speed_100m_previous_day2": [10.0, 20.0, 30.0, 40.0],
            "wind_direction_100m_previous_day2": [0.0, 90.0, 180.0, 270.0],
        },
        index=index,
    )


def test_candidate_masks_are_exact_and_hour14_day1_is_forbidden() -> None:
    frame = external_frame()
    day2 = select_external_features(frame, CANDIDATES[0])
    composite = select_external_features(frame, CANDIDATES[1])
    assert day2["ecmwf__ws100_ms"].tolist() == [10.0, 20.0, 30.0, 40.0]
    assert composite["ecmwf__ws100_ms"].tolist() == [10.0, 2.0, 3.0, 40.0]


def test_meteorological_direction_conversion_and_disagreement() -> None:
    frame = external_frame()
    selected = select_external_features(frame, CANDIDATES[0])
    np.testing.assert_allclose(selected["ecmwf__u100_ms"], [0.0, -20.0, 0.0, 40.0], atol=1e-5)
    np.testing.assert_allclose(selected["ecmwf__v100_ms"], [-10.0, 0.0, 30.0, 0.0], atol=1e-5)
    control = pd.DataFrame({"cross__hub_ws_mean": [1.0, 2.0, 3.0, 4.0]}, index=frame.index)
    extended = add_disagreement(selected, control)
    assert extended["ecmwf__ws100_minus_cross_hub_ws_mean"].tolist() == [9.0, 18.0, 27.0, 36.0]


def test_zero_terminal_increment_preserves_baseline_bit_identity() -> None:
    index = pd.DatetimeIndex(["2024-12-31 23:00", "2025-01-01 00:00"])
    baseline = pd.Series([1234.567890123, 9876.543210987], index=index)
    increment = pd.Series([0.01, 0.0], index=index)
    candidate = apply_paired_increment(baseline, increment, capacity_kwh=21_600.0)
    assert candidate.iloc[-1] == baseline.iloc[-1]
    assert np.float64(candidate.iloc[-1]).tobytes() == np.float64(baseline.iloc[-1]).tobytes()


def test_heavy_guard_exclusive_and_owner_only_release(tmp_path: Path) -> None:
    path = tmp_path / "heavy_cpu_fit.pid.json"
    owner = acquire_heavy_guard(path)
    assert owner["pid"] == os.getpid()
    assert pid_is_alive(owner["pid"])
    with pytest.raises(RuntimeError, match="held by live PID"):
        acquire_heavy_guard(path)
    fake_owner = {"pid": owner["pid"] + 1}
    release_heavy_guard(path, fake_owner)
    assert path.exists()
    release_heavy_guard(path, owner)
    assert not path.exists()


def test_heavy_guard_reclaims_confirmed_stale_pid(tmp_path: Path) -> None:
    path = tmp_path / "heavy_cpu_fit.pid.json"
    path.write_text(json.dumps({"pid": 2_147_483_647}), encoding="utf-8")
    owner = acquire_heavy_guard(path)
    assert owner["pid"] == os.getpid()
    release_heavy_guard(path, owner)

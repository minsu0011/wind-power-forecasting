"""Causal availability assertions shared by breakthrough data builders."""

from __future__ import annotations

import pandas as pd


def assert_available_before_cutoff(
    frame: pd.DataFrame,
    *,
    forecast_column: str = "forecast_kst_dtm",
    available_column: str = "data_available_kst_dtm",
    cutoff_hour_kst: int = 14,
) -> None:
    forecast = pd.to_datetime(frame[forecast_column], errors="raise")
    available = pd.to_datetime(frame[available_column], errors="raise")
    operating_day = (forecast - pd.to_timedelta(1, unit="h")).dt.floor("D")
    cutoff = operating_day - pd.to_timedelta(1, unit="D") + pd.to_timedelta(
        cutoff_hour_kst, unit="h"
    )
    invalid = available > cutoff
    if invalid.any():
        examples = frame.loc[invalid, [forecast_column, available_column]].head().to_dict("records")
        raise AssertionError(f"Post-cutoff information detected: {examples}")

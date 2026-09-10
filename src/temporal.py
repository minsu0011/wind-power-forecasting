"""Leakage-safe smoothing inside one already-issued 24-hour forecast run."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def operating_run_key(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Map operating hours 01:00..next-day 00:00 to their common run date."""

    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("index must be a DatetimeIndex")
    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("index must be unique and increasing")
    return (index - pd.Timedelta(hours=1)).normalize()


def smooth_within_runs(
    predictions: pd.DataFrame,
    strengths: Mapping[str, float],
    *,
    expected_run_hours: int = 24,
) -> pd.DataFrame:
    """Apply symmetric neighbour smoothing without crossing forecast runs.

    For strength ``s``, each interior prediction becomes
    ``(1-s)*current + s/2*(previous+next)``.  Run-edge values are replicated.
    Every horizon in a run is available at the same issuance time, so this
    operation does not consume observations or a later forecast issuance.
    """

    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("predictions must be a DataFrame")
    if not isinstance(predictions.index, pd.DatetimeIndex):
        raise TypeError("predictions must use a DatetimeIndex")
    if not predictions.columns.is_unique:
        raise ValueError("prediction columns must be unique")
    if set(strengths) != set(map(str, predictions.columns)):
        raise ValueError("strengths must contain exactly the prediction columns")
    values = predictions.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("predictions contain non-finite values")

    run_key = operating_run_key(predictions.index)
    groups = pd.Series(np.arange(len(predictions)), index=predictions.index).groupby(
        run_key, sort=True
    )
    output = predictions.astype(float).copy()
    for _, positions_series in groups:
        positions = positions_series.to_numpy(dtype=int)
        if len(positions) != expected_run_hours:
            raise ValueError(
                f"forecast run has {len(positions)} rows; expected {expected_run_hours}"
            )
        run_index = predictions.index[positions]
        expected = pd.date_range(run_index[0], periods=expected_run_hours, freq="h")
        if not run_index.equals(expected):
            raise ValueError("forecast run is not contiguous hourly data")
        for column in predictions.columns:
            strength = float(strengths[column])
            if not np.isfinite(strength) or not 0.0 <= strength <= 1.0:
                raise ValueError(f"invalid smoothing strength for {column}: {strength}")
            if strength == 0.0:
                continue
            current = predictions[column].iloc[positions].to_numpy(dtype=float)
            previous = np.concatenate(([current[0]], current[:-1]))
            following = np.concatenate((current[1:], [current[-1]]))
            output.loc[run_index, column] = (
                (1.0 - strength) * current
                + 0.5 * strength * (previous + following)
            )
    return output


__all__ = ["operating_run_key", "smooth_within_runs"]

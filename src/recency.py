"""Small deterministic helpers for leakage-safe time-decay experiments."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def exponential_time_weights(
    index: pd.DatetimeIndex,
    *,
    training_end: pd.Timestamp,
    half_life_days: float,
) -> np.ndarray:
    """Return mean-one past-only exponential weights.

    ``training_end`` must be at or after every training timestamp.  Rejecting a
    negative age prevents a validation/future row from silently entering the
    training weight calculation.
    """

    if not isinstance(index, pd.DatetimeIndex) or len(index) == 0:
        raise TypeError("index must be a non-empty DatetimeIndex")
    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("index must be unique and monotonically increasing")
    half_life = float(half_life_days)
    if not np.isfinite(half_life) or half_life <= 0:
        raise ValueError("half_life_days must be positive and finite")
    end = pd.Timestamp(training_end)
    ages = (end - index).total_seconds().to_numpy(dtype=float) / 86_400.0
    if np.any(ages < -1e-12):
        raise ValueError("training index contains rows after training_end")
    weights = np.exp2(-np.maximum(ages, 0.0) / half_life)
    weights /= np.mean(weights)
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise AssertionError("time weights must be positive and finite")
    return weights


def select_stable_half_life(
    deltas: Mapping[str, Mapping[str, float]],
    *,
    half_lives: Sequence[int],
    required_segments: Sequence[str],
) -> int | None:
    """Select only a half-life with a strict positive delta in every segment."""

    segments = tuple(required_segments)
    if not segments:
        raise ValueError("required_segments must not be empty")
    stable: list[tuple[float, float, int]] = []
    for half_life in half_lives:
        key = str(int(half_life))
        if key not in deltas:
            raise ValueError(f"missing deltas for half-life {key}")
        values = deltas[key]
        missing = [segment for segment in segments if segment not in values]
        if missing:
            raise ValueError(f"half-life {key} missing segments {missing!r}")
        segment_values = [float(values[segment]) for segment in segments]
        if all(value > 0.0 for value in segment_values):
            stable.append((min(segment_values), float(np.mean(segment_values)), int(half_life)))
    if not stable:
        return None
    return max(stable, key=lambda item: (item[0], item[1], item[2]))[2]


__all__ = ("exponential_time_weights", "select_stable_half_life")

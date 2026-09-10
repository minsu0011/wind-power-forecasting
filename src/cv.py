"""Time-ordered validation utilities for BARAM 2026.

The competition asks for an entire unseen calendar year.  A random split would
allow almost identical weather regimes from the future to influence model and
post-processing choices.  These folds therefore use complete future years and
never expose validation labels to fitting or calibration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeFold:
    name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp

    def masks(self, index: pd.Index) -> tuple[np.ndarray, np.ndarray]:
        dt = pd.DatetimeIndex(index)
        train = (dt >= self.train_start) & (dt < self.train_end)
        valid = (dt >= self.valid_start) & (dt < self.valid_end)
        if not train.any() or not valid.any():
            raise ValueError(f"{self.name}: empty train or validation interval")
        if dt[train].max() >= dt[valid].min():
            raise AssertionError(f"{self.name}: future leakage in fold boundaries")
        if np.any(train & valid):
            raise AssertionError(f"{self.name}: overlapping train/validation rows")
        return np.asarray(train), np.asarray(valid)


FOLD_2023 = TimeFold(
    name="train_2022_valid_2023",
    train_start=pd.Timestamp("2022-01-01 01:00:00"),
    train_end=pd.Timestamp("2023-01-01 00:00:01"),
    valid_start=pd.Timestamp("2023-01-01 00:00:01"),
    valid_end=pd.Timestamp("2024-01-01 00:00:01"),
)

FOLD_2024 = TimeFold(
    name="train_2022_2023_valid_2024",
    train_start=pd.Timestamp("2022-01-01 01:00:00"),
    train_end=pd.Timestamp("2024-01-01 00:00:01"),
    valid_start=pd.Timestamp("2024-01-01 00:00:01"),
    valid_end=pd.Timestamp("2025-01-01 00:00:01"),
)


def available_folds(target: str) -> tuple[TimeFold, ...]:
    """Return expanding-year folds supported by each target's label history."""

    if target in {"kpx_group_1", "kpx_group_2"}:
        return (FOLD_2023, FOLD_2024)
    if target == "kpx_group_3":
        # Group 3 has no 2022 labels, so only a strict 2023 -> 2024 year fold is
        # available. The mask is later intersected with non-null labels.
        return (FOLD_2024,)
    raise KeyError(f"unknown target: {target}")


def assert_aligned(*frames: pd.DataFrame | pd.Series) -> None:
    """Fail loudly when feature/label timestamps are not identically aligned."""

    if not frames:
        return
    reference = frames[0].index
    if not isinstance(reference, pd.DatetimeIndex):
        raise TypeError("aligned frames must use a DatetimeIndex")
    if not reference.is_unique or not reference.is_monotonic_increasing:
        raise ValueError("reference index must be unique and sorted")
    for position, frame in enumerate(frames[1:], start=1):
        if not reference.equals(frame.index):
            raise ValueError(f"frame {position} is not timestamp-aligned")


__all__ = [
    "FOLD_2023",
    "FOLD_2024",
    "TimeFold",
    "assert_aligned",
    "available_folds",
]


"""Target-free agreement gate for a joint multi-NWP paired increment.

The gate uses only predictions that are available at application time.  It
does not use labels, residuals, calendar slices, or public-leaderboard
membership.  Three independently fitted source-specific paired increments
vote on the sign of the joint increment.  A two-source majority is required;
the surviving correction is additionally capped by the median absolute
source-specific increment.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


SOURCE_ORDER = ("ecmwf", "icon", "gfs")


def agreement_gated_increment(
    joint_increment_cf: pd.DataFrame,
    source_increments_cf: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return gated increment, confidence, and integer agreement count.

    The deterministic formula is, cell by cell::

        n = number of source increments having the joint increment's sign
        vote = n / 3 if n >= 2 else 0
        cap = min(1, median(abs(source increments)) / abs(joint increment))
        gated = joint increment * vote * cap

    For an exactly zero joint increment, both ``cap`` and ``gated`` are zero.
    A source increment of exactly zero is not a sign match.  There are no
    fitted thresholds or label-derived parameters.
    """

    if tuple(source_increments_cf) != SOURCE_ORDER:
        raise ValueError(f"sources must be ordered exactly as {SOURCE_ORDER}")
    if not joint_increment_cf.index.is_unique or not joint_increment_cf.columns.is_unique:
        raise ValueError("joint increment keys must be unique")
    if not all(
        frame.index.equals(joint_increment_cf.index)
        and frame.columns.equals(joint_increment_cf.columns)
        for frame in source_increments_cf.values()
    ):
        raise ValueError("source and joint increment schemas differ")

    joint = joint_increment_cf.to_numpy(dtype=np.float64)
    sources = np.stack(
        [source_increments_cf[name].to_numpy(dtype=np.float64) for name in SOURCE_ORDER],
        axis=0,
    )
    if not np.isfinite(joint).all() or not np.isfinite(sources).all():
        raise ValueError("increment arrays contain non-finite values")

    joint_sign = np.sign(joint)[None, :, :]
    source_sign = np.sign(sources)
    agreement_count_array = np.sum(
        (joint_sign != 0.0) & (source_sign != 0.0) & (source_sign == joint_sign),
        axis=0,
        dtype=np.int8,
    )
    vote = np.where(agreement_count_array >= 2, agreement_count_array / 3.0, 0.0)
    median_absolute = np.median(np.abs(sources), axis=0)
    joint_absolute = np.abs(joint)
    magnitude_cap = np.divide(
        median_absolute,
        joint_absolute,
        out=np.zeros_like(joint_absolute),
        where=joint_absolute > 0.0,
    )
    magnitude_cap = np.minimum(1.0, magnitude_cap)
    confidence_array = vote * magnitude_cap
    gated_array = joint * confidence_array

    kwargs = {"index": joint_increment_cf.index, "columns": joint_increment_cf.columns}
    gated = pd.DataFrame(gated_array, dtype=np.float64, **kwargs)
    confidence = pd.DataFrame(confidence_array, dtype=np.float64, **kwargs)
    agreement_count = pd.DataFrame(agreement_count_array, dtype=np.int8, **kwargs)

    if not ((confidence.to_numpy() >= 0.0) & (confidence.to_numpy() <= 1.0)).all():
        raise AssertionError("confidence escaped [0, 1]")
    if (np.abs(gated.to_numpy()) > np.abs(joint) + 1e-15).any():
        raise AssertionError("gate enlarged an increment")
    return gated, confidence, agreement_count


__all__ = ["SOURCE_ORDER", "agreement_gated_increment"]

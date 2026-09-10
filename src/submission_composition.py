"""Small, auditable helpers for composing submission predictions.

The functions in this module deliberately do not fit a model.  They only
transfer a prediction delta that was learned by an upstream, separately
audited procedure.  Keeping this operation separate makes it difficult to
silently re-fit or re-scale a correction while assembling a submission.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


TARGET_COLUMNS: tuple[str, ...] = (
    "kpx_group_1",
    "kpx_group_2",
    "kpx_group_3",
)

CAPACITY_KWH: Mapping[str, float] = {
    "kpx_group_1": 21_600.0,
    "kpx_group_2": 21_600.0,
    "kpx_group_3": 21_000.0,
}


def _validate_prediction_frame(frame: pd.DataFrame, *, name: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if tuple(frame.columns) != TARGET_COLUMNS:
        raise ValueError(
            f"{name} columns must be exactly {TARGET_COLUMNS}; got {tuple(frame.columns)}"
        )
    if not frame.index.is_unique:
        raise ValueError(f"{name} index must be unique")
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinite values")


def transfer_prediction_delta(
    target: pd.DataFrame,
    reference: pd.DataFrame,
    adjusted_reference: pd.DataFrame,
    *,
    groups: Sequence[str],
    upper_capacity_fraction: float = 1.02,
) -> pd.DataFrame:
    """Add an upstream prediction delta to selected target groups.

    ``adjusted_reference - reference`` is transferred without any fitted
    scale, offset, or data-dependent tuning.  Non-selected groups remain
    value-bit identical to ``target``.  All frames must have identical index
    and schema.
    """

    _validate_prediction_frame(target, name="target")
    _validate_prediction_frame(reference, name="reference")
    _validate_prediction_frame(adjusted_reference, name="adjusted_reference")
    if not target.index.equals(reference.index) or not target.index.equals(
        adjusted_reference.index
    ):
        raise ValueError("target/reference/adjusted_reference indices differ")
    selected = tuple(groups)
    if not selected:
        raise ValueError("groups must contain at least one target group")
    if len(set(selected)) != len(selected):
        raise ValueError("groups contains duplicates")
    unknown = sorted(set(selected).difference(TARGET_COLUMNS))
    if unknown:
        raise ValueError(f"unknown target groups: {unknown}")
    upper = float(upper_capacity_fraction)
    if not np.isfinite(upper) or upper <= 0:
        raise ValueError("upper_capacity_fraction must be finite and positive")

    result = target.copy(deep=True)
    delta = adjusted_reference - reference
    for group in selected:
        values = target[group].to_numpy(dtype=float) + delta[group].to_numpy(
            dtype=float
        )
        result[group] = np.clip(values, 0.0, upper * CAPACITY_KWH[group])

    _validate_prediction_frame(result, name="result")
    for group in set(TARGET_COLUMNS).difference(selected):
        if not np.array_equal(
            result[group].to_numpy(), target[group].to_numpy(), equal_nan=True
        ):
            raise AssertionError(f"non-selected {group} changed during composition")
    return result


__all__ = [
    "CAPACITY_KWH",
    "TARGET_COLUMNS",
    "transfer_prediction_delta",
]

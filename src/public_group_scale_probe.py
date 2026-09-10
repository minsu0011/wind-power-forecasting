"""Auditable helpers for one-group-at-a-time Public-adaptive scale probes."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .metric import CAPACITY_KWH, TARGET_COLS
from .public_scale_probe import validate_prediction_frame


def scale_one_group(
    prediction: pd.DataFrame,
    group: str,
    *,
    factor: float = 0.92,
    upper_capacity_fraction: float = 1.02,
) -> pd.DataFrame:
    """Scale exactly one group while retaining other float values bit-for-bit."""

    validate_prediction_frame(prediction, name="prediction")
    if group not in TARGET_COLS:
        raise ValueError(f"unknown target group: {group}")
    multiplier = float(factor)
    upper = float(upper_capacity_fraction)
    if not np.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("factor must be finite and positive")
    if not np.isfinite(upper) or upper <= 0.0:
        raise ValueError("upper_capacity_fraction must be finite and positive")

    output = prediction.astype(np.float64).copy(deep=True)
    output[group] = np.clip(
        prediction[group].to_numpy(dtype=np.float64) * multiplier,
        0.0,
        upper * CAPACITY_KWH[group],
    )
    for identity_group in TARGET_COLS:
        if identity_group == group:
            continue
        if not np.array_equal(
            output[identity_group].to_numpy(dtype=np.float64),
            prediction[identity_group].to_numpy(dtype=np.float64),
        ):
            raise AssertionError(f"{identity_group} identity values changed")
    validate_prediction_frame(output, name="group-only scaled prediction")
    return output


def macro_separability_residuals(
    *,
    base: dict[str, float],
    global_scaled: dict[str, float],
    group_only: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Check additive macro deltas for total, 1-NMAE and FICR."""

    if tuple(group_only) != TARGET_COLS:
        raise ValueError("group_only mappings must use canonical group order")
    components = ("score", "one_minus_nmae", "ficr")
    result: dict[str, Any] = {}
    for component in components:
        base_value = float(base[component])
        global_delta = float(global_scaled[component]) - base_value
        group_deltas = {
            group: float(group_only[group][component]) - base_value
            for group in TARGET_COLS
        }
        summed = float(sum(group_deltas.values()))
        result[component] = {
            "global_delta": global_delta,
            "group_only_deltas": group_deltas,
            "sum_group_only_deltas": summed,
            "residual": float(global_delta - summed),
        }
    return result


__all__ = ["macro_separability_residuals", "scale_one_group"]

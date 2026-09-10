"""Official-metric facade with an explicit implementation-parity assertion."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.metric import CAPACITY_KWH, score_details


TARGETS = ("kpx_group_1", "kpx_group_2", "kpx_group_3")


def independent_metric(
    actual: pd.DataFrame,
    prediction: pd.DataFrame,
    targets: Sequence[str] = TARGETS,
) -> dict[str, Any]:
    one_minus_values: list[float] = []
    ficr_values: list[float] = []
    by_group: dict[str, dict[str, float | int]] = {}
    for target in targets:
        capacity = float(CAPACITY_KWH[target])
        y = actual[target].to_numpy(dtype=np.float64)
        p = prediction[target].to_numpy(dtype=np.float64)
        valid = np.isfinite(y) & np.isfinite(p) & (y >= 0.10 * capacity)
        error = np.abs(p[valid] - y[valid]) / capacity
        unit_price = np.where(error <= 0.06, 4.0, np.where(error <= 0.08, 3.0, 0.0))
        one_minus = 1.0 - float(np.mean(error))
        ficr = float(np.sum(y[valid] * unit_price) / (4.0 * np.sum(y[valid])))
        one_minus_values.append(one_minus)
        ficr_values.append(ficr)
        by_group[target] = {
            "n_evaluated": int(valid.sum()),
            "one_minus_nmae": one_minus,
            "ficr": ficr,
        }
    one_minus_nmae = float(np.mean(one_minus_values))
    ficr = float(np.mean(ficr_values))
    return {
        "total_score": 0.5 * (one_minus_nmae + ficr),
        "one_minus_nmae": one_minus_nmae,
        "ficr": ficr,
        "by_group": by_group,
    }


def parity_report(actual: pd.DataFrame, prediction: pd.DataFrame) -> dict[str, Any]:
    official = asdict(
        score_details(actual, prediction, target_cols=TARGETS, capacities=CAPACITY_KWH)
    )
    independent = independent_metric(actual, prediction)
    differences = {
        name: float(independent[name] - official[name])
        for name in ("total_score", "one_minus_nmae", "ficr")
    }
    if max(abs(value) for value in differences.values()) > 2e-15:
        raise AssertionError(f"Official metric parity failed: {differences}")
    return {"official": official, "independent": independent, "differences": differences}

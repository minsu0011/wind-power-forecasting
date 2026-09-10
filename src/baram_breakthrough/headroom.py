"""Exact design-period oracle utilities for the breakthrough study.

The functions in this module do not learn a policy.  They quantify an upper
bound obtained by choosing among a preregistered finite action/model family
with the actual outcome.  The choice is therefore explicitly oracle-only and
must never be used to construct a submission.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from src.metric import group_metrics


@dataclass(frozen=True)
class OracleResult:
    prediction: np.ndarray
    selected_position: np.ndarray
    eligible: np.ndarray
    baseline_metrics: dict[str, float]
    oracle_metrics: dict[str, float]
    delta: dict[str, float]
    selected_counts: dict[str, int]


def payment(error_cf: np.ndarray) -> np.ndarray:
    """Return the official 4/3/0 payment for capacity-normalized error."""

    error = np.asarray(error_cf, dtype=np.float64)
    return np.select(
        [error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0
    ).astype(np.float64)


def _metric_dict(actual: np.ndarray, pred: np.ndarray, capacity: float) -> dict[str, float]:
    result = group_metrics(actual, pred, capacity)
    return {
        "total_score": float(0.5 * result.one_minus_nmae + 0.5 * result.ficr),
        "one_minus_nmae": float(result.one_minus_nmae),
        "ficr": float(result.ficr),
    }


def rowwise_oracle(
    actual_kwh: np.ndarray,
    candidates_kwh: Mapping[str, np.ndarray],
    *,
    baseline_name: str,
    capacity_kwh: float,
) -> OracleResult:
    """Choose the exact official-score-maximizing candidate independently per row.

    Official group score is separable after its eligible-row count and actual
    energy denominator are fixed.  We therefore maximize each row's exact
    contribution.  This is a label oracle, not a deployable policy.
    """

    names = tuple(candidates_kwh)
    if not names or baseline_name not in candidates_kwh:
        raise ValueError("candidate pool must contain the named baseline")
    actual = np.asarray(actual_kwh, dtype=np.float64)
    if actual.ndim != 1 or actual.size == 0 or np.isinf(actual).any():
        raise ValueError("actual_kwh must be a nonempty 1-D array without infinity")
    capacity = float(capacity_kwh)
    if not np.isfinite(capacity) or capacity <= 0:
        raise ValueError("capacity_kwh must be positive and finite")

    matrix = np.column_stack(
        [np.asarray(candidates_kwh[name], dtype=np.float64) for name in names]
    )
    if matrix.shape[0] != actual.size or not np.isfinite(matrix).all():
        raise ValueError("all candidates must be finite and aligned to actual")
    eligible = np.isfinite(actual) & (actual >= 0.10 * capacity)
    if not eligible.any():
        raise ValueError("no eligible rows")

    y = actual[eligible]
    p = matrix[eligible]
    error_cf = np.abs(p - y[:, None]) / capacity
    # Constants shared across candidates are omitted.  The remaining terms are
    # the exact row contributions to 0.5*(1-NMAE)+0.5*FICR.
    contribution = (
        -0.5 * error_cf / float(y.size)
        + 0.5 * (y[:, None] * payment(error_cf)) / (4.0 * float(y.sum()))
    )
    best = np.argmax(contribution, axis=1)
    baseline_position = names.index(baseline_name)
    selected = np.full(actual.size, baseline_position, dtype=np.int32)
    selected[eligible] = best.astype(np.int32)
    prediction = matrix[np.arange(actual.size), selected]

    baseline = matrix[:, baseline_position]
    baseline_metrics = _metric_dict(actual, baseline, capacity)
    oracle_metrics = _metric_dict(actual, prediction, capacity)
    delta = {
        key: float(oracle_metrics[key] - baseline_metrics[key])
        for key in baseline_metrics
    }
    counts = {
        name: int(np.sum(best == position))
        for position, name in enumerate(names)
    }
    return OracleResult(
        prediction=prediction,
        selected_position=selected,
        eligible=eligible,
        baseline_metrics=baseline_metrics,
        oracle_metrics=oracle_metrics,
        delta=delta,
        selected_counts=counts,
    )


def boundary_energy_census(
    actual_kwh: np.ndarray,
    prediction_kwh: np.ndarray,
    *,
    capacity_kwh: float,
    group: str,
) -> pd.DataFrame:
    """Return count and actual-energy share in the official error bands."""

    actual = np.asarray(actual_kwh, dtype=np.float64)
    pred = np.asarray(prediction_kwh, dtype=np.float64)
    if actual.shape != pred.shape or actual.ndim != 1:
        raise ValueError("actual and prediction must be aligned 1-D arrays")
    eligible = np.isfinite(actual) & (actual >= 0.10 * float(capacity_kwh))
    if not eligible.any() or not np.isfinite(pred[eligible]).all():
        raise ValueError("invalid eligible rows")
    y = actual[eligible]
    error = np.abs(pred[eligible] - y) / float(capacity_kwh)
    definitions = (
        ("within_6", error <= 0.06),
        ("between_6_and_8", (error > 0.06) & (error <= 0.08)),
        ("over_8", error > 0.08),
    )
    rows: list[dict[str, object]] = []
    for band, mask in definitions:
        rows.append(
            {
                "group": group,
                "band": band,
                "count": int(mask.sum()),
                "actual_energy_kwh": float(y[mask].sum()),
                "actual_energy_share": float(y[mask].sum() / y.sum()),
            }
        )
    return pd.DataFrame(rows)


__all__ = ["OracleResult", "boundary_energy_census", "payment", "rowwise_oracle"]


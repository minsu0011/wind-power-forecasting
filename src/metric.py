"""Official BARAM 2026 leaderboard metric.

The implementation mirrors DACON's published ``metric.ipynb``:

* only rows with actual generation >= 10% of group capacity are scored;
* group NMAE and FICR are averaged with equal group weights;
* FICR pays 4 / 3 / 0 for errors <= 6%, <= 8%, and > 8%; and
* settlement is weighted by actual generation, not by row count.

Predictions are deliberately not clipped.  Clipping inside the metric would make
local validation disagree with the leaderboard.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np


TARGET_COLS: tuple[str, ...] = (
    "kpx_group_1",
    "kpx_group_2",
    "kpx_group_3",
)

CAPACITY_KWH: dict[str, float] = {
    "kpx_group_1": 21_600.0,
    "kpx_group_2": 21_600.0,
    "kpx_group_3": 21_000.0,
}

MIN_ACTUAL_FRACTION = 0.10
FULL_PAYMENT_ERROR = 0.06
PARTIAL_PAYMENT_ERROR = 0.08


@dataclass(frozen=True)
class GroupMetrics:
    """Metric values and diagnostics for one wind-farm group."""

    group: str
    capacity_kwh: float
    n_evaluated: int
    actual_energy_kwh: float
    nmae: float
    one_minus_nmae: float
    ficr: float
    earned_settlement: float
    max_settlement: float
    within_6_count: int
    between_6_and_8_count: int
    over_8_count: int
    within_6_energy_share: float
    between_6_and_8_energy_share: float
    over_8_energy_share: float

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON/table-friendly representation."""

        return asdict(self)


@dataclass(frozen=True)
class CompetitionMetrics:
    """Official aggregate values plus equal-weight group diagnostics."""

    total_score: float
    one_minus_nmae: float
    ficr: float
    by_group: dict[str, GroupMetrics]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON/table-friendly nested representation."""

        return asdict(self)


def _as_1d_float(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}")
    return array


def group_metrics(
    actual: Any,
    forecast: Any,
    capacity_kwh: float,
    *,
    group_name: str = "group",
) -> GroupMetrics:
    """Calculate the official metrics and diagnostics for a single group.

    Non-finite actual values are treated as unavailable labels and therefore are
    excluded.  A non-finite forecast on an evaluated row raises an error so an
    incomplete OOF prediction cannot silently receive a misleading score.
    """

    capacity = float(capacity_kwh)
    if not np.isfinite(capacity) or capacity <= 0:
        raise ValueError("capacity_kwh must be a positive finite number")

    actual_array = _as_1d_float(actual, name="actual")
    forecast_array = _as_1d_float(forecast, name="forecast")
    if actual_array.shape != forecast_array.shape:
        raise ValueError(
            "actual and forecast must have the same shape; "
            f"got {actual_array.shape} and {forecast_array.shape}"
        )

    valid = np.isfinite(actual_array) & (
        actual_array >= capacity * MIN_ACTUAL_FRACTION
    )
    if not np.any(valid):
        raise ValueError(
            f"{group_name!r} has no actual values at or above 10% of capacity"
        )

    evaluated_actual = actual_array[valid]
    evaluated_forecast = forecast_array[valid]
    if not np.all(np.isfinite(evaluated_forecast)):
        raise ValueError(f"{group_name!r} has non-finite forecasts on evaluated rows")

    error_rate = np.abs(evaluated_forecast - evaluated_actual) / capacity
    unit_price = np.select(
        [
            error_rate <= FULL_PAYMENT_ERROR,
            error_rate <= PARTIAL_PAYMENT_ERROR,
        ],
        [4.0, 3.0],
        default=0.0,
    )

    within_6 = error_rate <= FULL_PAYMENT_ERROR
    between_6_and_8 = (~within_6) & (error_rate <= PARTIAL_PAYMENT_ERROR)
    over_8 = error_rate > PARTIAL_PAYMENT_ERROR

    actual_energy = float(np.sum(evaluated_actual))
    earned_settlement = float(np.sum(evaluated_actual * unit_price))
    max_settlement = float(np.sum(evaluated_actual * 4.0))
    nmae = float(np.mean(error_rate))

    def energy_share(mask: np.ndarray) -> float:
        return float(np.sum(evaluated_actual[mask]) / actual_energy)

    return GroupMetrics(
        group=group_name,
        capacity_kwh=capacity,
        n_evaluated=int(valid.sum()),
        actual_energy_kwh=actual_energy,
        nmae=nmae,
        one_minus_nmae=1.0 - nmae,
        ficr=earned_settlement / max_settlement,
        earned_settlement=earned_settlement,
        max_settlement=max_settlement,
        within_6_count=int(within_6.sum()),
        between_6_and_8_count=int(between_6_and_8.sum()),
        over_8_count=int(over_8.sum()),
        within_6_energy_share=energy_share(within_6),
        between_6_and_8_energy_share=energy_share(between_6_and_8),
        over_8_energy_share=energy_share(over_8),
    )


def _column(frame: Any, column: str, *, frame_name: str) -> Any:
    try:
        return frame[column]
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError(f"{frame_name} is missing target column {column!r}") from exc


def score_details(
    answer: Any,
    prediction: Any,
    *,
    target_cols: Sequence[str] = TARGET_COLS,
    capacities: Mapping[str, float] = CAPACITY_KWH,
) -> CompetitionMetrics:
    """Calculate official aggregate score and per-group diagnostics.

    ``answer`` and ``prediction`` may be pandas DataFrames or any objects that
    support ``obj[column_name]``.  Groups are always averaged equally, even when
    their evaluated row counts or generated energy differ.
    """

    columns = tuple(target_cols)
    if not columns:
        raise ValueError("target_cols must contain at least one group")
    if len(set(columns)) != len(columns):
        raise ValueError("target_cols must not contain duplicates")

    by_group: dict[str, GroupMetrics] = {}
    for column in columns:
        if column not in capacities:
            raise ValueError(f"capacities is missing target column {column!r}")
        by_group[column] = group_metrics(
            _column(answer, column, frame_name="answer"),
            _column(prediction, column, frame_name="prediction"),
            capacities[column],
            group_name=column,
        )

    mean_nmae = float(np.mean([values.nmae for values in by_group.values()]))
    one_minus_nmae = 1.0 - mean_nmae
    ficr = float(np.mean([values.ficr for values in by_group.values()]))
    total_score = 0.5 * one_minus_nmae + 0.5 * ficr

    return CompetitionMetrics(
        total_score=total_score,
        one_minus_nmae=one_minus_nmae,
        ficr=ficr,
        by_group=by_group,
    )


def metric(answer_df: Any, pred_df: Any) -> tuple[float, float, float]:
    """Return ``(Score, 1-NMAE, FICR)`` exactly like DACON's official API."""

    result = score_details(answer_df, pred_df)
    return result.total_score, result.one_minus_nmae, result.ficr


__all__ = [
    "CAPACITY_KWH",
    "TARGET_COLS",
    "CompetitionMetrics",
    "GroupMetrics",
    "group_metrics",
    "metric",
    "score_details",
]

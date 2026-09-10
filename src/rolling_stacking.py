"""Strict-forward helpers for low-degree-of-freedom OOF stacking.

The fitted object records the last timestamp used for parameter estimation and
refuses to predict an overlapping/earlier period by default.  The optimization
is deliberately small: six non-negative simplex weights and one bounded
capacity-factor intercept.  Eligibility filtering mirrors the official metric.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .metric import group_metrics


DEFAULT_COMPONENTS: tuple[str, ...] = (
    "lgb_l1",
    "lgb_q07",
    "shared_l1",
    "shared_q07",
    "top200_q07",
    "energy_q06",
)


@dataclass(frozen=True)
class SimplexHuberParameter:
    """A content-serializable fitted stacking parameter."""

    component_names: tuple[str, ...]
    weights: tuple[float, ...]
    intercept_cf: float
    regularization: float
    fit_start: str
    fit_end: str
    eligible_rows: int
    optimizer_objective: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _matrix(values: object, *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype="float64")
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return result


def _vector(values: object, *, name: str, expected_rows: int) -> np.ndarray:
    result = np.asarray(values, dtype="float64")
    if result.ndim != 1 or len(result) != expected_rows:
        raise ValueError(
            f"{name} must be one-dimensional with {expected_rows} rows"
        )
    if np.isinf(result).any():
        raise ValueError(f"{name} contains infinite values")
    return result


def _time_index(index: object, *, expected_rows: int, name: str) -> pd.DatetimeIndex:
    result = pd.DatetimeIndex(index)
    if len(result) != expected_rows:
        raise ValueError(f"{name} must contain {expected_rows} timestamps")
    if not result.is_unique or not result.is_monotonic_increasing:
        raise ValueError(f"{name} must be unique and monotonically increasing")
    return result


def pseudo_huber_loss(residual_cf: object, *, delta_cf: float = 0.04) -> np.ndarray:
    """Return the element-wise pseudo-Huber loss in capacity-factor units."""

    delta = float(delta_cf)
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("delta_cf must be positive and finite")
    residual = np.asarray(residual_cf, dtype="float64")
    if not np.isfinite(residual).all():
        raise ValueError("residual_cf contains NaN or infinite values")
    return delta * delta * (np.sqrt(1.0 + np.square(residual / delta)) - 1.0)


def fit_simplex_huber(
    matrix_cf: object,
    actual_cf: object,
    index: object,
    *,
    component_names: Sequence[str] = DEFAULT_COMPONENTS,
    regularization: float,
    sample_weight: object | None = None,
    delta_cf: float = 0.04,
    intercept_bounds: tuple[float, float] = (-0.025, 0.025),
    intercept_penalty: float = 0.10,
    eligible_floor_cf: float = 0.10,
    maxiter: int = 2_000,
    ftol: float = 1e-12,
) -> SimplexHuberParameter:
    """Fit the registered non-negative simplex pseudo-Huber stack.

    Rows below the official ten-percent eligibility threshold, or with missing
    actual values, are excluded.  Prediction values must always be finite.
    """

    matrix = _matrix(matrix_cf, name="matrix_cf")
    actual = _vector(actual_cf, name="actual_cf", expected_rows=len(matrix))
    timestamps = _time_index(index, expected_rows=len(matrix), name="fit index")
    names = tuple(str(value) for value in component_names)
    if len(names) != matrix.shape[1] or len(set(names)) != len(names):
        raise ValueError("component_names must uniquely match matrix columns")
    ridge = float(regularization)
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("regularization must be finite and non-negative")
    low, high = map(float, intercept_bounds)
    if not np.isfinite([low, high]).all() or not low <= 0.0 <= high:
        raise ValueError("intercept_bounds must be finite and contain zero")
    penalty = float(intercept_penalty)
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError("intercept_penalty must be finite and non-negative")
    floor = float(eligible_floor_cf)
    if not np.isfinite(floor) or floor < 0:
        raise ValueError("eligible_floor_cf must be finite and non-negative")

    eligible = np.isfinite(actual) & (actual >= floor)
    if not eligible.any():
        raise ValueError("no eligible finite actual rows")
    fit_matrix = matrix[eligible]
    fit_actual = actual[eligible]
    if sample_weight is None:
        row_weight = np.ones(len(matrix), dtype="float64")
    else:
        row_weight = _vector(
            sample_weight, name="sample_weight", expected_rows=len(matrix)
        )
        if np.isnan(row_weight).any() or np.any(row_weight < 0):
            raise ValueError("sample_weight must be finite and non-negative")
    fit_weight = row_weight[eligible]
    weight_sum = float(fit_weight.sum())
    if weight_sum <= 0:
        raise ValueError("eligible sample weights must have positive total")

    n_components = matrix.shape[1]
    prior = np.full(n_components, 1.0 / n_components, dtype="float64")
    delta = float(delta_cf)
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("delta_cf must be positive and finite")

    def objective(parameter: np.ndarray) -> float:
        residual = fit_matrix @ parameter[:-1] + parameter[-1] - fit_actual
        robust = pseudo_huber_loss(residual, delta_cf=delta)
        return float(
            np.dot(fit_weight, robust) / weight_sum
            + ridge * np.sum(np.square(parameter[:-1] - prior))
            + penalty * parameter[-1] * parameter[-1]
        )

    def jacobian(parameter: np.ndarray) -> np.ndarray:
        residual = fit_matrix @ parameter[:-1] + parameter[-1] - fit_actual
        derivative = residual / np.sqrt(1.0 + np.square(residual / delta))
        weighted = fit_weight * derivative / weight_sum
        gradient = np.empty(n_components + 1, dtype="float64")
        gradient[:-1] = (
            fit_matrix.T @ weighted + 2.0 * ridge * (parameter[:-1] - prior)
        )
        gradient[-1] = float(weighted.sum() + 2.0 * penalty * parameter[-1])
        return gradient

    initial = np.concatenate([prior, np.array([0.0])])
    result = minimize(
        objective,
        initial,
        jac=jacobian,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_components + [(low, high)],
        constraints={"type": "eq", "fun": lambda value: value[:-1].sum() - 1.0},
        options={"maxiter": int(maxiter), "ftol": float(ftol)},
    )
    if not result.success:
        raise RuntimeError(f"SLSQP failed: {result.message}")
    fitted = np.asarray(result.x, dtype="float64")
    if not np.isfinite(fitted).all():
        raise RuntimeError("SLSQP returned non-finite parameters")
    if np.any(fitted[:-1] < -1e-10):
        raise RuntimeError("SLSQP violated non-negative weight bounds")
    if abs(float(fitted[:-1].sum()) - 1.0) > 1e-8:
        raise RuntimeError("SLSQP violated the simplex constraint")
    weights = np.clip(fitted[:-1], 0.0, 1.0)
    weights /= weights.sum()
    intercept = float(fitted[-1])
    return SimplexHuberParameter(
        component_names=names,
        weights=tuple(float(value) for value in weights),
        intercept_cf=intercept,
        regularization=ridge,
        fit_start=timestamps.min().isoformat(),
        fit_end=timestamps.max().isoformat(),
        eligible_rows=int(eligible.sum()),
        optimizer_objective=objective(np.concatenate([weights, [intercept]])),
    )


def apply_simplex_huber(
    parameter: SimplexHuberParameter,
    matrix_cf: object,
    index: object,
    *,
    upper_cf: float = 1.02,
    require_strictly_future: bool = True,
) -> np.ndarray:
    """Apply locked parameters, rejecting fit/application overlap by default."""

    matrix = _matrix(matrix_cf, name="matrix_cf")
    timestamps = _time_index(index, expected_rows=len(matrix), name="application index")
    if matrix.shape[1] != len(parameter.component_names):
        raise ValueError("application component count differs from fitted parameter")
    if require_strictly_future and timestamps.min() <= pd.Timestamp(parameter.fit_end):
        raise ValueError("application timestamps are not strictly after fit timestamps")
    weights = np.asarray(parameter.weights, dtype="float64")
    if (
        len(weights) != matrix.shape[1]
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1e-8)
    ):
        raise ValueError("stored weights are not a valid simplex")
    upper = float(upper_cf)
    if not np.isfinite(upper) or upper <= 0:
        raise ValueError("upper_cf must be positive and finite")
    prediction = matrix @ weights + float(parameter.intercept_cf)
    return np.clip(prediction, 0.0, upper)


def residual_blend(
    baseline_cf: object,
    direct_stack_cf: object,
    blend_weight: float,
    *,
    upper_cf: float = 1.02,
) -> np.ndarray:
    """Take a registered bounded step from baseline toward the direct stack."""

    baseline = np.asarray(baseline_cf, dtype="float64")
    direct = np.asarray(direct_stack_cf, dtype="float64")
    if baseline.ndim != 1 or baseline.shape != direct.shape or len(baseline) == 0:
        raise ValueError("baseline_cf and direct_stack_cf must be aligned vectors")
    if not np.isfinite(baseline).all() or not np.isfinite(direct).all():
        raise ValueError("residual blend inputs must be finite")
    weight = float(blend_weight)
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("blend_weight must be in [0, 1]")
    return np.clip(baseline + weight * (direct - baseline), 0.0, float(upper_cf))


def group_score_triplet(
    actual_kwh: object,
    prediction_kwh: object,
    capacity_kwh: float,
    *,
    group_name: str,
) -> dict[str, float]:
    """Return official group score, 1-NMAE and FICR."""

    metrics = group_metrics(
        actual_kwh,
        prediction_kwh,
        capacity_kwh,
        group_name=group_name,
    )
    return {
        "score": 0.5 * (metrics.one_minus_nmae + metrics.ficr),
        "one_minus_nmae": metrics.one_minus_nmae,
        "ficr": metrics.ficr,
    }


def compare_segments(
    index: object,
    actual_kwh: object,
    baseline_kwh: object,
    candidate_kwh: object,
    capacity_kwh: float,
    segments: Mapping[str, tuple[str, str]],
    *,
    group_name: str,
) -> dict[str, dict[str, object]]:
    """Compare fixed candidate/baseline values on named closed intervals."""

    actual = np.asarray(actual_kwh, dtype="float64")
    baseline = np.asarray(baseline_kwh, dtype="float64")
    candidate = np.asarray(candidate_kwh, dtype="float64")
    if actual.ndim != 1 or baseline.shape != actual.shape or candidate.shape != actual.shape:
        raise ValueError("actual, baseline and candidate must be aligned vectors")
    timestamps = _time_index(index, expected_rows=len(actual), name="score index")
    output: dict[str, dict[str, object]] = {}
    for name, bounds in segments.items():
        start, end = map(pd.Timestamp, bounds)
        selected = np.asarray((timestamps >= start) & (timestamps <= end))
        if not selected.any():
            raise ValueError(f"segment {name!r} is empty")
        base_metrics = group_score_triplet(
            actual[selected], baseline[selected], capacity_kwh, group_name=group_name
        )
        candidate_metrics = group_score_triplet(
            actual[selected], candidate[selected], capacity_kwh, group_name=group_name
        )
        output[name] = {
            "rows": int(selected.sum()),
            "baseline": base_metrics,
            "candidate": candidate_metrics,
            "delta": {
                key: float(candidate_metrics[key] - base_metrics[key])
                for key in ("score", "one_minus_nmae", "ficr")
            },
        }
    return output


def passes_registered_gate(
    comparisons: Mapping[str, Mapping[str, object]],
    *,
    full_segment_name: str,
) -> bool:
    """Apply the v2 gate using an explicitly named full validation interval."""

    if not comparisons:
        raise ValueError("comparisons must not be empty")
    for values in comparisons.values():
        delta = values.get("delta")
        if not isinstance(delta, Mapping) or float(delta["score"]) <= 0.0:
            return False
    if full_segment_name not in comparisons:
        raise ValueError(
            f"full segment {full_segment_name!r} is absent from comparisons"
        )
    full_delta = comparisons[full_segment_name]["delta"]
    nmae_delta = float(full_delta["one_minus_nmae"])
    ficr_delta = float(full_delta["ficr"])
    return nmae_delta >= 0.0 and ficr_delta >= 0.0 and (
        nmae_delta > 0.0 or ficr_delta > 0.0
    )


__all__ = [
    "DEFAULT_COMPONENTS",
    "SimplexHuberParameter",
    "apply_simplex_huber",
    "compare_segments",
    "fit_simplex_huber",
    "group_score_triplet",
    "passes_registered_gate",
    "pseudo_huber_loss",
    "residual_blend",
]

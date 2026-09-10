"""ExtraTrees leaf-neighborhood empirical residual Bayes action.

The fitted forest is used only as a deterministic conditional partition.  For
each application row, every tree assigns equal probability to the eligible fit
residuals in the matching leaf and the 400 tree distributions are averaged.
The official-utility action is evaluated exactly from that continuous weighted
empirical support; no residual bins, quantile interpolation, or top-k pruning
are used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer

from src.full_weather_residual_pmf import (
    STATE_FEATURE_COLUMNS,
    TOTAL_FEATURE_COUNT,
    build_full_weather_state_features,
    transfer_residual_action_kwh,
)
from src.residual_histogram_bayes import ACTION_DELTAS_CF


FOREST_PARAMETERS: dict[str, Any] = {
    "n_estimators": 400,
    "criterion": "squared_error",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 64,
    "max_features": 0.5,
    "bootstrap": False,
    "oob_score": False,
    "max_samples": None,
    "random_state": 42,
    "n_jobs": 7,
}
TRANSFER_WEIGHT = 0.10


def _frame(value: pd.DataFrame, *, name: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{name} must be a DataFrame")
    if value.shape[1] != TOTAL_FEATURE_COUNT:
        raise ValueError(f"{name} must contain {TOTAL_FEATURE_COUNT} columns")
    if not isinstance(value.index, pd.DatetimeIndex):
        raise TypeError(f"{name} index must be DatetimeIndex")
    if not value.index.is_unique or not value.index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be unique and increasing")
    values = value.to_numpy(dtype=np.float64, copy=False)
    if np.isinf(values).any():
        raise ValueError(f"{name} contains infinite values")
    return value


def _series(
    value: pd.Series, *, name: str, allow_nan: bool = False
) -> pd.Series:
    if not isinstance(value, pd.Series):
        raise TypeError(f"{name} must be a Series")
    output = value.astype(np.float64)
    if not isinstance(output.index, pd.DatetimeIndex):
        raise TypeError(f"{name} index must be DatetimeIndex")
    if not output.index.is_unique or not output.index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be unique and increasing")
    values = output.to_numpy()
    if np.isinf(values).any() or (not allow_nan and np.isnan(values).any()):
        suffix = "may contain NaN but not infinity" if allow_nan else "must be finite"
        raise ValueError(f"{name} {suffix}")
    return output


def qrf_weight_matrix_from_leaf_ids(
    fit_leaf_ids: Any,
    application_leaf_ids: Any,
    *,
    dtype: np.dtype[Any] = np.dtype(np.float64),
) -> np.ndarray:
    """Return exact standard same-sample QRF weights from leaf-id matrices."""

    fit = np.asarray(fit_leaf_ids)
    application = np.asarray(application_leaf_ids)
    if fit.ndim != 2 or application.ndim != 2:
        raise ValueError("leaf-id arrays must be two-dimensional")
    if fit.shape[1] == 0 or fit.shape[1] != application.shape[1]:
        raise ValueError("fit/application tree counts differ or are empty")
    if fit.shape[0] == 0 or application.shape[0] == 0:
        raise ValueError("leaf-id arrays must have rows")
    if not np.issubdtype(fit.dtype, np.integer) or not np.issubdtype(
        application.dtype, np.integer
    ):
        raise TypeError("leaf ids must be integers")
    tree_count = fit.shape[1]
    weights = np.zeros((application.shape[0], fit.shape[0]), dtype=dtype)
    tree_weight = 1.0 / float(tree_count)
    for tree_index in range(tree_count):
        fit_ids = fit[:, tree_index]
        query_ids = application[:, tree_index]
        fit_order = np.argsort(fit_ids, kind="stable")
        sorted_fit = fit_ids[fit_order]
        unique_query = np.unique(query_ids)
        for leaf_id in unique_query:
            left = int(np.searchsorted(sorted_fit, leaf_id, side="left"))
            right = int(np.searchsorted(sorted_fit, leaf_id, side="right"))
            if left == right:
                raise AssertionError("query reached a leaf without fitted rows")
            fit_positions = fit_order[left:right]
            query_positions = np.flatnonzero(query_ids == leaf_id)
            weights[np.ix_(query_positions, fit_positions)] += tree_weight / float(
                len(fit_positions)
            )
    row_sum = weights.sum(axis=1, dtype=np.float64)
    if np.any(weights < 0.0) or not np.allclose(
        row_sum, 1.0, atol=1e-12, rtol=0.0
    ):
        raise AssertionError("QRF weights are not nonnegative unit masses")
    return weights


def _interval_weighted_sum(
    sorted_values: np.ndarray,
    cumulative: np.ndarray,
    weighted_values: np.ndarray,
    actions: np.ndarray,
    radius: float,
) -> np.ndarray:
    lower = actions - float(radius)
    upper = actions + float(radius)
    left = np.searchsorted(sorted_values, lower, side="left")
    right = np.searchsorted(sorted_values, upper, side="right")
    padded = np.concatenate(([0.0], cumulative))
    result = padded[right] - padded[left]
    # ``abs(action-y) <= radius`` is the metric contract.  Algebraically
    # equivalent action +/- radius bounds can round one ulp differently at an
    # exact 6%/8% boundary.  Correct only the adjacent insertion candidates
    # with the literal subtraction predicate so this fast CDF calculation is
    # bit-semantically identical to the dense definition.
    for row, action in enumerate(actions):
        candidates: set[int] = set()
        for centre in (int(left[row]), int(right[row])):
            for position in range(max(0, centre - 2), min(len(sorted_values), centre + 3)):
                candidates.add(position)
        for position in candidates:
            included_by_cdf = bool(left[row] <= position < right[row])
            included_exact = bool(abs(float(action) - float(sorted_values[position])) <= radius)
            if included_by_cdf and not included_exact:
                result[row] -= weighted_values[position]
            elif included_exact and not included_by_cdf:
                result[row] += weighted_values[position]
    return result


def empirical_expected_utility_actions(
    weights: Any,
    fit_residual_cf: Any,
    base_cf: Any,
    *,
    mean_actual_cf: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Select the frozen action exactly under weighted continuous supports."""

    weight = np.asarray(weights, dtype=np.float64)
    residual = np.asarray(fit_residual_cf, dtype=np.float64)
    base = np.asarray(base_cf, dtype=np.float64)
    if weight.ndim != 2 or residual.shape != (weight.shape[1],):
        raise ValueError("weight/support shapes differ")
    if base.shape != (weight.shape[0],):
        raise ValueError("base rows differ from weight rows")
    if not np.isfinite(weight).all() or not np.isfinite(residual).all() or not np.isfinite(base).all():
        raise ValueError("empirical action inputs must be finite")
    if np.any(weight < 0.0) or not np.allclose(
        weight.sum(axis=1), 1.0, atol=1e-12, rtol=0.0
    ):
        raise ValueError("weights must be nonnegative and sum to one")
    mean = float(mean_actual_cf)
    if not np.isfinite(mean) or mean < 0.10:
        raise ValueError("mean_actual_cf must be finite and at least 0.10")

    selected = np.empty(len(base), dtype=np.float64)
    selected_utility = np.empty(len(base), dtype=np.float64)
    effective_n = np.empty(len(base), dtype=np.float64)
    nonzero_n = np.empty(len(base), dtype=np.int32)
    for row in range(len(base)):
        local_weight = weight[row]
        support_mask = local_weight > 0.0
        local_weight = local_weight[support_mask]
        outcomes = np.clip(base[row] + residual[support_mask], 0.10, 1.20)
        order = np.argsort(outcomes, kind="stable")
        outcomes = outcomes[order]
        local_weight = local_weight[order]
        cumulative_w = np.cumsum(local_weight, dtype=np.float64)
        cumulative_yw = np.cumsum(local_weight * outcomes, dtype=np.float64)
        total_yw = float(cumulative_yw[-1])

        actions = np.clip(base[row] + ACTION_DELTAS_CF, 0.0, 1.02)
        split = np.searchsorted(outcomes, actions, side="right")
        padded_w = np.concatenate(([0.0], cumulative_w))
        padded_yw = np.concatenate(([0.0], cumulative_yw))
        left_w = padded_w[split]
        left_yw = padded_yw[split]
        expected_abs = (
            actions * left_w
            - left_yw
            + (total_yw - left_yw)
            - actions * (1.0 - left_w)
        )
        within_6_yw = _interval_weighted_sum(
            outcomes, cumulative_yw, local_weight * outcomes, actions, 0.06
        )
        within_8_yw = _interval_weighted_sum(
            outcomes, cumulative_yw, local_weight * outcomes, actions, 0.08
        )
        settlement_yw = 4.0 * within_6_yw + 3.0 * (
            within_8_yw - within_6_yw
        )
        utility = -expected_abs + settlement_yw / (4.0 * mean)
        maximum = float(np.max(utility))
        tied = np.flatnonzero(utility >= maximum - 1e-12)
        tied_delta = ACTION_DELTAS_CF[tied]
        minimum_abs = float(np.min(np.abs(tied_delta)))
        tied = tied[np.abs(tied_delta) <= minimum_abs + 1e-15]
        chosen = int(tied[0])
        selected[row] = ACTION_DELTAS_CF[chosen]
        selected_utility[row] = utility[chosen]
        effective_n[row] = 1.0 / float(np.sum(local_weight**2))
        nonzero_n[row] = int(len(local_weight))
    diagnostics = {
        "effective_support_size": effective_n,
        "nonzero_support_count": nonzero_n,
    }
    return selected, selected_utility, diagnostics


@dataclass(frozen=True)
class LeafEmpiricalSettings:
    n_estimators: int = 400
    min_samples_leaf: int = 64
    max_features: float = 0.5
    random_state: int = 42
    n_jobs: int = 7


class ExtraTreesLeafEmpiricalBayes:
    """One fixed ExtraTrees partition and continuous residual QRF action."""

    def __init__(self) -> None:
        self.settings = LeafEmpiricalSettings()

    def fit(
        self,
        features: pd.DataFrame,
        actual_kwh: pd.Series,
        baseline_kwh: pd.Series,
        *,
        capacity_kwh: float,
    ) -> "ExtraTreesLeafEmpiricalBayes":
        frame = _frame(features, name="fit_features")
        actual = _series(actual_kwh, name="fit_actual_kwh", allow_nan=True)
        baseline = _series(baseline_kwh, name="fit_baseline_kwh")
        if not frame.index.equals(actual.index) or not frame.index.equals(baseline.index):
            raise ValueError("fit inputs are not row-aligned")
        capacity = float(capacity_kwh)
        if not np.isfinite(capacity) or capacity <= 0.0:
            raise ValueError("capacity_kwh must be positive and finite")
        actual_cf = actual.to_numpy(dtype=np.float64) / capacity
        baseline_cf = baseline.to_numpy(dtype=np.float64) / capacity
        eligible = np.isfinite(actual_cf) & (actual_cf >= 0.10)
        if int(eligible.sum()) < 2 * FOREST_PARAMETERS["min_samples_leaf"]:
            raise ValueError("too few eligible rows for frozen forest leaves")
        fit_frame = frame.loc[eligible]
        self.feature_columns_ = tuple(str(value) for value in fit_frame.columns)
        self.imputer_ = SimpleImputer(strategy="median").fit(fit_frame)
        transformed = self.imputer_.transform(fit_frame).astype(np.float32, copy=False)
        if not np.isfinite(transformed).all():
            raise AssertionError("imputed fit matrix contains nonfinite values")
        residual = actual_cf[eligible] - baseline_cf[eligible]
        estimator = ExtraTreesRegressor(**FOREST_PARAMETERS)
        estimator.fit(transformed, residual)
        original_jobs = estimator.n_jobs
        try:
            estimator.n_jobs = 1
            fit_leaf_ids = np.asarray(estimator.apply(transformed), dtype=np.int64)
        finally:
            estimator.n_jobs = original_jobs
        if fit_leaf_ids.shape != (len(transformed), FOREST_PARAMETERS["n_estimators"]):
            raise AssertionError("fit leaf-id matrix shape changed")
        leaf_sizes: list[int] = []
        for tree in range(fit_leaf_ids.shape[1]):
            _, counts = np.unique(fit_leaf_ids[:, tree], return_counts=True)
            leaf_sizes.extend(counts.astype(int).tolist())
        if min(leaf_sizes) < FOREST_PARAMETERS["min_samples_leaf"]:
            raise AssertionError("fitted leaf violates minimum support")
        self.estimator_ = estimator
        self.fit_leaf_ids_ = fit_leaf_ids
        self.fit_residual_cf_ = residual.astype(np.float64, copy=True)
        self.fit_index_ = fit_frame.index.copy()
        self.capacity_kwh_ = capacity
        self.mean_actual_cf_ = float(np.mean(actual_cf[eligible]))
        self.fit_rows_total_ = int(len(frame))
        self.fit_rows_eligible_ = int(eligible.sum())
        self.leaf_size_min_ = int(min(leaf_sizes))
        self.leaf_size_max_ = int(max(leaf_sizes))
        return self

    def predict_action(
        self,
        features: pd.DataFrame,
        baseline_kwh: pd.Series,
    ) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
        if not hasattr(self, "estimator_"):
            raise RuntimeError("fit must be called before predict_action")
        frame = _frame(features, name="application_features")
        if tuple(str(value) for value in frame.columns) != self.feature_columns_:
            raise ValueError("application feature schema/order differs from fit")
        baseline = _series(baseline_kwh, name="application_baseline_kwh")
        if not frame.index.equals(baseline.index):
            raise ValueError("application inputs are not row-aligned")
        if len(frame.index.intersection(self.fit_index_)):
            raise ValueError("fit/application row overlap")
        transformed = self.imputer_.transform(frame).astype(np.float32, copy=False)
        if not np.isfinite(transformed).all():
            raise AssertionError("imputed application matrix contains nonfinite values")
        original_jobs = self.estimator_.n_jobs
        try:
            self.estimator_.n_jobs = 1
            application_leaf_ids = np.asarray(
                self.estimator_.apply(transformed), dtype=np.int64
            )
        finally:
            self.estimator_.n_jobs = original_jobs
        weights = qrf_weight_matrix_from_leaf_ids(
            self.fit_leaf_ids_, application_leaf_ids
        )
        base_cf = baseline.to_numpy(dtype=np.float64) / self.capacity_kwh_
        action, utility, support = empirical_expected_utility_actions(
            weights,
            self.fit_residual_cf_,
            base_cf,
            mean_actual_cf=self.mean_actual_cf_,
        )
        action_series = pd.Series(
            action, index=frame.index, name="raw_action_delta_cf"
        )
        leaf_frame = pd.DataFrame(
            application_leaf_ids,
            index=frame.index,
            columns=[f"tree_{value:03d}" for value in range(application_leaf_ids.shape[1])],
        )
        diagnostics = pd.DataFrame(
            {
                "raw_action_delta_cf": action,
                "raw_action_cf": np.clip(base_cf + action, 0.0, 1.02),
                "expected_utility": utility,
                "qrf_weight_sum": weights.sum(axis=1),
                "effective_support_size": support["effective_support_size"],
                "nonzero_support_count": support["nonzero_support_count"],
            },
            index=frame.index,
        )
        return action_series, leaf_frame, diagnostics

    def metadata(self) -> dict[str, Any]:
        if not hasattr(self, "estimator_"):
            raise RuntimeError("fit must be called before metadata")
        return {
            "model_class": type(self.estimator_).__name__,
            "settings": asdict(self.settings),
            "resolved_parameters": self.estimator_.get_params(deep=False),
            "fit_rows_total": self.fit_rows_total_,
            "fit_rows_eligible": self.fit_rows_eligible_,
            "fit_start": self.fit_index_.min().isoformat(),
            "fit_end": self.fit_index_.max().isoformat(),
            "fit_mean_actual_cf": self.mean_actual_cf_,
            "fit_residual_min_cf": float(np.min(self.fit_residual_cf_)),
            "fit_residual_max_cf": float(np.max(self.fit_residual_cf_)),
            "leaf_size_min": self.leaf_size_min_,
            "leaf_size_max": self.leaf_size_max_,
            "tree_count": int(self.fit_leaf_ids_.shape[1]),
            "feature_count": TOTAL_FEATURE_COUNT,
            "state_feature_columns": list(STATE_FEATURE_COLUMNS),
            "distribution_kind": "continuous_standard_same_sample_qrf",
            "transfer_weight": TRANSFER_WEIGHT,
        }


__all__ = [
    "FOREST_PARAMETERS",
    "TRANSFER_WEIGHT",
    "ExtraTreesLeafEmpiricalBayes",
    "LeafEmpiricalSettings",
    "build_full_weather_state_features",
    "empirical_expected_utility_actions",
    "qrf_weight_matrix_from_leaf_ids",
    "transfer_residual_action_kwh",
]

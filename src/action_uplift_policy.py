"""Baseline-conditioned direct policy for two fixed multiplicative actions.

The two supervised targets are deterministic, label-derived counterfactual
utilities for applying 0.98 or 0.95 to a fixed corrected-v3 forecast.  They are
used only on an earlier fit block.  Application features contain baseline,
causal component and exogenous weather state, never actual generation or a
same-row residual/utility.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from lightgbm import LGBMClassifier, LGBMRegressor
import numpy as np
import pandas as pd

from src.metric import CAPACITY_KWH, group_metrics
from src.residual_histogram_bayes import META_FEATURE_COLUMNS


ACTION_FACTORS: tuple[float, ...] = (0.98, 0.95)
PROBABILITY_THRESHOLD = 0.55
ACTION_FEATURE_COLUMNS: tuple[str, ...] = (
    "action_factor",
    "action_delta_factor",
    "proposed_cf",
    "proposed_minus_base_cf",
    "proposed_minus_component_mean_cf",
)
MODEL_FEATURE_COLUMNS: tuple[str, ...] = META_FEATURE_COLUMNS + ACTION_FEATURE_COLUMNS

CLASSIFIER_PARAMETERS: dict[str, Any] = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "n_estimators": 160,
    "learning_rate": 0.025,
    "num_leaves": 7,
    "max_depth": 3,
    "min_child_samples": 240,
    "min_split_gain": 0.0,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.75,
    "reg_alpha": 2.0,
    "reg_lambda": 24.0,
    "max_bin": 63,
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
    "n_jobs": 7,
    "random_state": 5801,
}
REGRESSOR_PARAMETERS: dict[str, Any] = {
    "boosting_type": "gbdt",
    "objective": "regression_l1",
    "n_estimators": 160,
    "learning_rate": 0.025,
    "num_leaves": 7,
    "max_depth": 3,
    "min_child_samples": 240,
    "min_split_gain": 0.0,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.75,
    "reg_alpha": 2.0,
    "reg_lambda": 24.0,
    "max_bin": 63,
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
    "n_jobs": 7,
    "random_state": 5802,
}


def _finite_series(value: pd.Series, *, name: str) -> pd.Series:
    if not isinstance(value, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    if value.empty or not value.index.is_unique or not value.index.is_monotonic_increasing:
        raise ValueError(f"{name} must have a nonempty, unique, increasing index")
    output = value.astype(np.float64)
    if not np.isfinite(output.to_numpy()).all():
        raise ValueError(f"{name} contains non-finite values")
    return output


def _validate_base_features(features: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")
    if tuple(features.columns) != META_FEATURE_COLUMNS:
        raise ValueError("base feature schema/order differs from preregistration")
    if features.empty or not features.index.is_unique or not features.index.is_monotonic_increasing:
        raise ValueError("features must have a nonempty, unique, increasing index")
    output = features.astype(np.float64)
    if not np.isfinite(output.to_numpy()).all():
        raise ValueError("features contain non-finite values")
    return output


def expanded_action_design(features: pd.DataFrame) -> np.ndarray:
    """Return row-major [timestamp, 0.98 then 0.95] fixed-action design."""

    frame = _validate_base_features(features)
    base = frame.to_numpy(dtype=np.float64, copy=False)
    action_count = len(ACTION_FACTORS)
    rows = len(frame)
    output = np.empty((rows * action_count, len(MODEL_FEATURE_COLUMNS)), dtype=np.float64)
    output[:, : len(META_FEATURE_COLUMNS)] = np.repeat(base, action_count, axis=0)
    factor = np.tile(np.asarray(ACTION_FACTORS, dtype=np.float64), rows)
    repeated_base = np.repeat(frame["base_cf"].to_numpy(dtype=np.float64), action_count)
    repeated_mean = np.repeat(
        frame["component_mean_cf"].to_numpy(dtype=np.float64), action_count
    )
    proposed = np.clip(factor * repeated_base, 0.0, 1.02)
    action_values = np.column_stack(
        (
            factor,
            factor - 1.0,
            proposed,
            proposed - repeated_base,
            proposed - repeated_mean,
        )
    )
    output[:, len(META_FEATURE_COLUMNS) :] = action_values
    if not np.isfinite(output).all():
        raise AssertionError("expanded action design contains non-finite values")
    return output


def _payment(error_cf: np.ndarray) -> np.ndarray:
    return np.select(
        [error_cf <= 0.06, error_cf <= 0.08],
        [4.0, 3.0],
        default=0.0,
    ).astype(np.float64)


def exact_row_utility_deltas(
    baseline_kwh: pd.Series,
    actual_kwh: pd.Series,
    *,
    capacity_kwh: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return eligible mask and an exact fit-block utility delta per action.

    For each action, the arithmetic mean of the returned eligible-row deltas
    equals ``group_metrics(action).score - group_metrics(baseline).score``.
    The equality follows because FICR can be written as the mean of
    ``actual_cf / mean(actual_cf) * payment / 4`` on eligible rows.
    """

    base = _finite_series(baseline_kwh, name="baseline_kwh")
    if not isinstance(actual_kwh, pd.Series) or not actual_kwh.index.equals(base.index):
        raise ValueError("actual and baseline indices differ")
    actual = actual_kwh.astype(np.float64)
    if np.isinf(actual.to_numpy()).any():
        raise ValueError("actual contains infinity")
    capacity = float(capacity_kwh)
    if not np.isfinite(capacity) or capacity <= 0.0:
        raise ValueError("capacity_kwh must be positive and finite")
    actual_cf = actual.to_numpy(dtype=np.float64, copy=False) / capacity
    base_cf = np.clip(base.to_numpy(dtype=np.float64, copy=False) / capacity, 0.0, 1.02)
    eligible = np.isfinite(actual_cf) & (actual_cf >= 0.10)
    if int(eligible.sum()) < 1:
        raise ValueError("fit block has no eligible actual rows")
    y = actual_cf[eligible]
    b = base_cf[eligible]
    mean_actual = float(np.mean(y))
    if not np.isfinite(mean_actual) or mean_actual <= 0.0:
        raise AssertionError("eligible fit mean actual CF is invalid")
    base_error = np.abs(b - y)
    base_payment = _payment(base_error)
    deltas = np.empty((len(y), len(ACTION_FACTORS)), dtype=np.float64)
    exact_score_deltas: dict[str, float] = {}
    row_mean_deltas: dict[str, float] = {}
    for position, factor in enumerate(ACTION_FACTORS):
        action = np.clip(float(factor) * b, 0.0, 1.02)
        action_error = np.abs(action - y)
        action_payment = _payment(action_error)
        nmae_delta = -(action_error - base_error)
        ficr_delta = (y / mean_actual) * (action_payment - base_payment) / 4.0
        delta = 0.5 * nmae_delta + 0.5 * ficr_delta
        deltas[:, position] = delta
        base_metrics = group_metrics(y * capacity, b * capacity, capacity)
        action_metrics = group_metrics(y * capacity, action * capacity, capacity)
        exact = float(
            0.5 * (action_metrics.one_minus_nmae - base_metrics.one_minus_nmae)
            + 0.5 * (action_metrics.ficr - base_metrics.ficr)
        )
        mean_delta = float(np.mean(delta))
        if not np.isclose(mean_delta, exact, rtol=0.0, atol=5e-15):
            raise AssertionError("row utility mean no longer equals exact official score delta")
        key = f"{factor:.2f}"
        exact_score_deltas[key] = exact
        row_mean_deltas[key] = mean_delta
    if not np.isfinite(deltas).all():
        raise AssertionError("utility targets contain non-finite values")
    return eligible, deltas, {
        "rows_total": int(len(base)),
        "rows_eligible": int(eligible.sum()),
        "fit_mean_actual_cf": mean_actual,
        "row_mean_deltas": row_mean_deltas,
        "exact_score_deltas": exact_score_deltas,
        "maximum_exactness_error": float(
            max(
                abs(row_mean_deltas[key] - exact_score_deltas[key])
                for key in row_mean_deltas
            )
        ),
    }


def assert_strict_forward(
    fit_index: pd.DatetimeIndex, application_index: pd.DatetimeIndex
) -> None:
    if len(fit_index) == 0 or len(application_index) == 0:
        raise ValueError("fit and application indices must be nonempty")
    if fit_index.intersection(application_index).size:
        raise ValueError("fit and application indices overlap")
    if not fit_index.max() < application_index.min():
        raise ValueError("fit must end before application begins")


class ActionUpliftPolicy:
    """Two fixed low-complexity heads trained on pooled action rows."""

    def __init__(self, *, minimum_eligible_fit_rows: int = 500) -> None:
        self.minimum_eligible_fit_rows = int(minimum_eligible_fit_rows)
        self.classifier_: LGBMClassifier | None = None
        self.regressor_: LGBMRegressor | None = None
        self.fit_index_: pd.DatetimeIndex | None = None
        self.fit_metadata_: dict[str, Any] | None = None

    def fit(
        self,
        features: pd.DataFrame,
        baseline_kwh: pd.Series,
        actual_kwh: pd.Series,
        *,
        capacity_kwh: float,
    ) -> "ActionUpliftPolicy":
        frame = _validate_base_features(features)
        base = _finite_series(baseline_kwh, name="baseline_kwh")
        if not frame.index.equals(base.index):
            raise ValueError("fit feature and baseline indices differ")
        eligible, deltas, target_metadata = exact_row_utility_deltas(
            base, actual_kwh, capacity_kwh=capacity_kwh
        )
        if int(eligible.sum()) < self.minimum_eligible_fit_rows:
            raise ValueError(
                f"too few eligible fit rows: {int(eligible.sum())} "
                f"< {self.minimum_eligible_fit_rows}"
            )
        design = expanded_action_design(frame.loc[eligible])
        regression_target = deltas.reshape(-1)
        classifier_target = (regression_target > 0.0).astype(np.uint8)
        if not np.isfinite(regression_target).all():
            raise AssertionError("regression target contains non-finite values")
        classes, counts = np.unique(classifier_target, return_counts=True)
        if not np.array_equal(classes, np.asarray([0, 1], dtype=np.uint8)):
            raise ValueError("classifier target must retain both classes")
        if int(counts.min()) < 25:
            raise ValueError("classifier target minority class is too small")
        classifier = LGBMClassifier(**CLASSIFIER_PARAMETERS)
        regressor = LGBMRegressor(**REGRESSOR_PARAMETERS)
        classifier.fit(design, classifier_target)
        regressor.fit(design, regression_target)
        self.classifier_ = classifier
        self.regressor_ = regressor
        self.fit_index_ = frame.index.copy()
        quantiles = (0.0, 0.01, 0.10, 0.50, 0.90, 0.99, 1.0)
        self.fit_metadata_ = {
            **target_metadata,
            "expanded_rows": int(len(design)),
            "expanded_feature_count": int(design.shape[1]),
            "classifier_negative_rows": int(counts[0]),
            "classifier_positive_rows": int(counts[1]),
            "classifier_positive_fraction": float(np.mean(classifier_target)),
            "regression_target_quantiles": {
                str(q): float(np.quantile(regression_target, q)) for q in quantiles
            },
            "classifier_parameters": dict(CLASSIFIER_PARAMETERS),
            "regressor_parameters": dict(REGRESSOR_PARAMETERS),
            "fit_start": frame.index.min().isoformat(),
            "fit_end": frame.index.max().isoformat(),
            "same_row_fit_score_computed": False,
            "application_labels_used": False,
        }
        return self

    def predict(
        self,
        features: pd.DataFrame,
        baseline_kwh: pd.Series,
        *,
        capacity_kwh: float,
    ) -> tuple[pd.Series, pd.DataFrame]:
        if self.classifier_ is None or self.regressor_ is None or self.fit_index_ is None:
            raise RuntimeError("fit must be called before predict")
        frame = _validate_base_features(features)
        base = _finite_series(baseline_kwh, name="baseline_kwh")
        if not frame.index.equals(base.index):
            raise ValueError("application feature and baseline indices differ")
        assert_strict_forward(self.fit_index_, frame.index)
        design = expanded_action_design(frame)
        shape = (len(frame), len(ACTION_FACTORS))
        probability = np.asarray(
            self.classifier_.predict_proba(design)[:, 1], dtype=np.float64
        ).reshape(shape)
        magnitude = np.asarray(
            self.regressor_.predict(design), dtype=np.float64
        ).reshape(shape)
        if not np.isfinite(probability).all() or not np.isfinite(magnitude).all():
            raise ValueError("policy head produced non-finite values")
        qualified = (probability >= PROBABILITY_THRESHOLD) & (magnitude > 0.0)
        objective = np.where(qualified, magnitude, -np.inf)
        selected_position = np.argmax(objective, axis=1)
        any_qualified = qualified.any(axis=1)
        factors = np.ones(len(frame), dtype=np.float64)
        action_array = np.asarray(ACTION_FACTORS, dtype=np.float64)
        factors[any_qualified] = action_array[selected_position[any_qualified]]
        base_values = base.to_numpy(dtype=np.float64, copy=False)
        candidate_values = base_values.copy()
        changed = factors != 1.0
        candidate_values[changed] = np.clip(
            factors[changed] * base_values[changed],
            0.0,
            1.02 * float(capacity_kwh),
        )
        if not np.array_equal(candidate_values[~changed], base_values[~changed]):
            raise AssertionError("identity policy rows are not bit exact")
        row = np.arange(len(frame))
        selected_probability = np.full(len(frame), np.nan, dtype=np.float64)
        selected_magnitude = np.zeros(len(frame), dtype=np.float64)
        selected_probability[any_qualified] = probability[
            row[any_qualified], selected_position[any_qualified]
        ]
        selected_magnitude[any_qualified] = magnitude[
            row[any_qualified], selected_position[any_qualified]
        ]
        diagnostics = pd.DataFrame(
            {
                "p_positive_098": probability[:, 0],
                "p_positive_095": probability[:, 1],
                "predicted_delta_098": magnitude[:, 0],
                "predicted_delta_095": magnitude[:, 1],
                "qualified_098": qualified[:, 0].astype(np.uint8),
                "qualified_095": qualified[:, 1].astype(np.uint8),
                "selected_factor": factors,
                "selected_probability": selected_probability,
                "selected_predicted_delta": selected_magnitude,
                "candidate_minus_baseline_kwh": candidate_values - base_values,
            },
            index=frame.index,
        )
        diagnostics.index.name = "forecast_kst_dtm"
        candidate = pd.Series(candidate_values, index=frame.index, name=base.name)
        return candidate, diagnostics

    def metadata(self) -> dict[str, Any]:
        if self.fit_metadata_ is None:
            raise RuntimeError("fit must be called before metadata")
        return dict(self.fit_metadata_)


def compare_group_blocks(
    actual_kwh: pd.Series,
    baseline_kwh: pd.Series,
    candidate_kwh: pd.Series,
    *,
    group: str,
    blocks: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    if group not in CAPACITY_KWH:
        raise KeyError(group)
    if not actual_kwh.index.equals(baseline_kwh.index) or not actual_kwh.index.equals(
        candidate_kwh.index
    ):
        raise ValueError("score inputs are not aligned")
    output: dict[str, Any] = {}
    for name, bounds in blocks.items():
        if len(bounds) != 2:
            raise ValueError(f"invalid block {name}")
        start, end = map(pd.Timestamp, bounds)
        selected = actual_kwh.index[
            (actual_kwh.index >= start) & (actual_kwh.index <= end)
        ]
        if len(selected) == 0:
            raise ValueError(f"empty block {name}")
        base = group_metrics(
            actual_kwh.loc[selected], baseline_kwh.loc[selected], CAPACITY_KWH[group], group_name=group
        )
        candidate = group_metrics(
            actual_kwh.loc[selected], candidate_kwh.loc[selected], CAPACITY_KWH[group], group_name=group
        )
        base_triplet = {
            "score": float(0.5 * base.one_minus_nmae + 0.5 * base.ficr),
            "one_minus_nmae": float(base.one_minus_nmae),
            "ficr": float(base.ficr),
        }
        candidate_triplet = {
            "score": float(0.5 * candidate.one_minus_nmae + 0.5 * candidate.ficr),
            "one_minus_nmae": float(candidate.one_minus_nmae),
            "ficr": float(candidate.ficr),
        }
        output[name] = {
            "rows": int(len(selected)),
            "baseline": base_triplet,
            "candidate": candidate_triplet,
            "delta": {
                metric: float(candidate_triplet[metric] - base_triplet[metric])
                for metric in base_triplet
            },
        }
    return output


def passes_all_blocks(comparisons: Mapping[str, Any], *, full_block: str) -> bool:
    if full_block not in comparisons:
        raise KeyError(full_block)
    if any(float(record["delta"]["score"]) <= 0.0 for record in comparisons.values()):
        return False
    full = comparisons[full_block]["delta"]
    nmae = float(full["one_minus_nmae"])
    ficr = float(full["ficr"])
    return nmae >= 0.0 and ficr >= 0.0 and (nmae > 0.0 or ficr > 0.0)


def transfer_delta(
    recent_baseline_kwh: pd.Series,
    v3_baseline_kwh: pd.Series,
    candidate_v3_kwh: pd.Series,
    *,
    capacity_kwh: float,
) -> pd.Series:
    if not (
        recent_baseline_kwh.index.equals(v3_baseline_kwh.index)
        and recent_baseline_kwh.index.equals(candidate_v3_kwh.index)
    ):
        raise ValueError("delta-transfer indices differ")
    recent = _finite_series(recent_baseline_kwh, name="recent_baseline_kwh")
    v3 = _finite_series(v3_baseline_kwh, name="v3_baseline_kwh")
    candidate = _finite_series(candidate_v3_kwh, name="candidate_v3_kwh")
    values = np.clip(
        recent.to_numpy(dtype=np.float64)
        + candidate.to_numpy(dtype=np.float64)
        - v3.to_numpy(dtype=np.float64),
        0.0,
        1.02 * float(capacity_kwh),
    )
    return pd.Series(values, index=recent.index, name=recent.name)


__all__ = [
    "ACTION_FACTORS",
    "ACTION_FEATURE_COLUMNS",
    "CLASSIFIER_PARAMETERS",
    "MODEL_FEATURE_COLUMNS",
    "PROBABILITY_THRESHOLD",
    "REGRESSOR_PARAMETERS",
    "ActionUpliftPolicy",
    "assert_strict_forward",
    "compare_group_blocks",
    "exact_row_utility_deltas",
    "expanded_action_design",
    "passes_all_blocks",
    "transfer_delta",
]

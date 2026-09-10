"""Fixed-bin weather ordinal PMF and exact official-utility action.

The fitted model uses only the locked weather feature frame.  Target capacity
factor is represented by 42 fixed 0.025-wide nominal bins.  Unsupported fit
classes map to exact zero columns in the fixed probability vector, and the
decision integrates that vector over one preregistered baseline-local action
grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.weather_quantile import TrainOnlyMedianTransform


BIN_COUNT = 42
BIN_WIDTH_CF = 0.025
BIN_START_CF = 0.10
BIN_CENTRES_CF = np.round(
    BIN_START_CF
    + BIN_WIDTH_CF * (np.arange(BIN_COUNT, dtype=np.float64) + 0.5),
    12,
)
ACTION_DELTAS_CF = np.round(
    np.arange(-0.150, 0.150 + 0.0025, 0.005, dtype=np.float64), 12
)
TRANSFER_WEIGHT = 0.10
MODEL_PARAMETERS: dict[str, Any] = {
    "boosting_type": "gbdt",
    "objective": "multiclass",
    "n_estimators": 120,
    "learning_rate": 0.035,
    "num_leaves": 7,
    "max_depth": 3,
    "min_child_samples": 100,
    "min_split_gain": 0.0,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.35,
    "reg_alpha": 0.5,
    "reg_lambda": 10.0,
    "max_bin": 63,
    "random_state": 42,
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
    "n_jobs": 7,
}
STAGE1_REQUIRED: dict[str, tuple[str, ...]] = {
    "kpx_group_1": ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4"),
    "kpx_group_2": ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4"),
    "kpx_group_3": ("full", "Q3", "Q4"),
}


@dataclass(frozen=True)
class OrdinalActionConfig:
    bin_count: int = BIN_COUNT
    bin_start_cf: float = BIN_START_CF
    bin_width_cf: float = BIN_WIDTH_CF
    action_lower_cf: float = 0.0
    action_upper_cf: float = 1.02
    transfer_weight: float = TRANSFER_WEIGHT
    tie_tolerance: float = 1e-12


def target_classes(actual_cf: np.ndarray) -> np.ndarray:
    """Map eligible target CF values to the exact fixed 42-class vocabulary."""

    values = np.asarray(actual_cf, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("actual_cf must be a finite one-dimensional array")
    if np.any(values < BIN_START_CF):
        raise ValueError("target_classes accepts eligible actual_cf>=0.10 only")
    # Explicit generated edges preserve the nominal half-open decimal bins at
    # representable boundary values (plain division can turn 0.125 into
    # 0.9999999999999998 before floor).
    upper_edges = np.round(
        BIN_START_CF
        + BIN_WIDTH_CF * np.arange(1, BIN_COUNT + 1, dtype=np.float64),
        12,
    )
    classes = np.searchsorted(upper_edges, values, side="right").astype(np.int64)
    return np.clip(classes, 0, BIN_COUNT - 1)


def fixed_probability_matrix(
    predict_proba: np.ndarray,
    model_classes: np.ndarray,
) -> np.ndarray:
    """Place observed model probabilities in a fixed 42-column matrix."""

    raw = np.asarray(predict_proba, dtype=np.float64)
    classes_raw = np.asarray(model_classes)
    if raw.ndim != 2 or classes_raw.ndim != 1:
        raise ValueError("probabilities must be 2-D and classes must be 1-D")
    if raw.shape[1] != len(classes_raw) or raw.shape[0] == 0:
        raise ValueError("probability/class shape mismatch")
    if not np.isfinite(raw).all() or np.any(raw < 0.0):
        raise ValueError("probabilities must be finite and nonnegative")
    classes_float = classes_raw.astype(np.float64)
    classes = classes_float.astype(np.int64)
    if not np.array_equal(classes_float, classes.astype(np.float64)):
        raise ValueError("model classes must be exact integers")
    if len(np.unique(classes)) != len(classes) or np.any(classes < 0) or np.any(
        classes >= BIN_COUNT
    ):
        raise ValueError("model classes are duplicate or outside fixed vocabulary")
    output = np.zeros((len(raw), BIN_COUNT), dtype=np.float64)
    output[:, classes] = raw
    row_sums = output.sum(axis=1)
    if not np.all(np.abs(row_sums - 1.0) <= 1e-12):
        raise ValueError("mapped probability rows do not sum to one within 1e-12")
    return output


def expected_official_utility(
    action_cf: np.ndarray,
    probabilities: np.ndarray,
    *,
    mean_train_actual_cf: float,
) -> np.ndarray:
    """Return exact registered expected action-dependent official score."""

    actions = np.asarray(action_cf, dtype=np.float64)
    probability = np.asarray(probabilities, dtype=np.float64)
    if actions.ndim != 2 or probability.ndim != 2:
        raise ValueError("actions and probabilities must be two-dimensional")
    if actions.shape[0] != probability.shape[0] or probability.shape[1] != BIN_COUNT:
        raise ValueError("action/probability shape mismatch")
    if not np.isfinite(actions).all() or not np.isfinite(probability).all():
        raise ValueError("action inputs must be finite")
    mean_actual = float(mean_train_actual_cf)
    if not np.isfinite(mean_actual) or mean_actual <= 0.0:
        raise ValueError("mean_train_actual_cf must be positive and finite")
    error = np.abs(actions[:, :, None] - BIN_CENTRES_CF[None, None, :])
    settlement = np.select(
        [error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0
    )
    class_utility = (
        -0.5 * error
        + BIN_CENTRES_CF[None, None, :]
        * settlement
        / (8.0 * mean_actual)
    )
    return np.sum(class_utility * probability[:, None, :], axis=2)


def ordinal_bayes_action_cf(
    probabilities: np.ndarray,
    baseline_cf: np.ndarray,
    *,
    mean_train_actual_cf: float,
    config: OrdinalActionConfig | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Choose one tie-broken action on the fixed baseline-local grid."""

    settings = config or OrdinalActionConfig()
    if settings != OrdinalActionConfig():
        raise ValueError("only the preregistered OrdinalActionConfig is allowed")
    probability = np.asarray(probabilities, dtype=np.float64)
    baseline = np.asarray(baseline_cf, dtype=np.float64)
    if probability.ndim != 2 or probability.shape[1] != BIN_COUNT:
        raise ValueError("probabilities must have exactly 42 columns")
    if baseline.shape != (len(probability),):
        raise ValueError("baseline shape differs from probability rows")
    if not np.isfinite(probability).all() or not np.isfinite(baseline).all():
        raise ValueError("Bayes action inputs must be finite")
    if np.any(probability < 0.0) or not np.all(
        np.abs(probability.sum(axis=1) - 1.0) <= 1e-12
    ):
        raise ValueError("invalid fixed probability matrix")
    actions = np.clip(
        baseline[:, None] + ACTION_DELTAS_CF[None, :],
        settings.action_lower_cf,
        settings.action_upper_cf,
    )
    utilities = expected_official_utility(
        actions, probability, mean_train_actual_cf=mean_train_actual_cf
    )
    selected = np.empty(len(baseline), dtype=np.float64)
    selected_delta = np.empty(len(baseline), dtype=np.float64)
    selected_utility = np.empty(len(baseline), dtype=np.float64)
    tied_count = np.empty(len(baseline), dtype=np.int64)
    for row in range(len(baseline)):
        row_actions = actions[row]
        row_utilities = utilities[row]
        maximum = float(np.max(row_utilities))
        tied = np.flatnonzero(row_utilities >= maximum - settings.tie_tolerance)
        # Clipping can duplicate actions.  The action value, not original delta,
        # governs the preregistered distance and lower-action tie break.
        order = np.lexsort(
            (row_actions[tied], np.abs(row_actions[tied] - baseline[row]))
        )
        chosen = int(tied[int(order[0])])
        selected[row] = row_actions[chosen]
        selected_delta[row] = row_actions[chosen] - baseline[row]
        selected_utility[row] = row_utilities[chosen]
        tied_count[row] = len(tied)
    diagnostics = pd.DataFrame(
        {
            "baseline_cf": baseline,
            "raw_action_cf": selected,
            "raw_action_delta_cf": selected_delta,
            "expected_utility": selected_utility,
            "tie_count_within_1e12": tied_count,
        }
    )
    return selected, diagnostics


def transfer_action_kwh(
    baseline_kwh: pd.Series,
    raw_action_cf: pd.Series,
    *,
    capacity_kwh: float,
) -> pd.Series:
    """Apply the one fixed 0.10 transfer; zero/alternate weights are forbidden."""

    if not baseline_kwh.index.equals(raw_action_cf.index):
        raise ValueError("baseline/action indexes differ")
    capacity = float(capacity_kwh)
    if not np.isfinite(capacity) or capacity <= 0.0:
        raise ValueError("capacity_kwh must be positive and finite")
    baseline_cf = baseline_kwh.to_numpy(dtype=np.float64) / capacity
    action_cf = raw_action_cf.to_numpy(dtype=np.float64)
    if not np.isfinite(baseline_cf).all() or not np.isfinite(action_cf).all():
        raise ValueError("baseline/action inputs must be finite")
    candidate_cf = np.clip(
        (1.0 - TRANSFER_WEIGHT) * baseline_cf + TRANSFER_WEIGHT * action_cf,
        0.0,
        1.02,
    )
    return pd.Series(
        candidate_cf * capacity,
        index=baseline_kwh.index,
        name=baseline_kwh.name,
    )


class FullWeatherOrdinalBayes:
    """One direct target-CF LightGBM multiclass PMF per group."""

    def __init__(self) -> None:
        self.model_parameters = dict(MODEL_PARAMETERS)
        self.minimum_actual_cf = BIN_START_CF
        self.action_config = OrdinalActionConfig()

    def fit(
        self,
        features: pd.DataFrame,
        actual_kwh: pd.Series,
        *,
        capacity_kwh: float,
    ) -> "FullWeatherOrdinalBayes":
        from lightgbm import LGBMClassifier

        if not isinstance(features, pd.DataFrame) or features.empty:
            raise TypeError("features must be a non-empty DataFrame")
        if not features.index.is_unique or not features.index.is_monotonic_increasing:
            raise ValueError("feature index must be unique and increasing")
        if not isinstance(actual_kwh, pd.Series) or not features.index.equals(
            actual_kwh.index
        ):
            raise ValueError("fit feature/actual indexes differ")
        capacity = float(capacity_kwh)
        if not np.isfinite(capacity) or capacity <= 0.0:
            raise ValueError("capacity_kwh must be positive and finite")
        actual_cf = actual_kwh.to_numpy(dtype=np.float64) / capacity
        eligible = np.isfinite(actual_cf) & (actual_cf >= self.minimum_actual_cf)
        if int(eligible.sum()) < 300:
            raise ValueError(f"too few eligible fit rows: {int(eligible.sum())}")
        eligible_features = features.loc[eligible]
        eligible_target = actual_cf[eligible]
        classes = target_classes(eligible_target)
        observed = np.unique(classes)
        if len(observed) < 2 or not np.array_equal(
            observed, np.arange(observed.min(), observed.max() + 1)
        ):
            raise ValueError("observed target classes must be contiguous")
        self.transform_ = TrainOnlyMedianTransform().fit(eligible_features)
        transformed = self.transform_.transform(eligible_features)
        self.model_ = LGBMClassifier(**self.model_parameters)
        self.model_.fit(transformed, classes)
        self.capacity_kwh_ = capacity
        self.fit_index_ = features.index.copy()
        self.feature_columns_ = tuple(features.columns)
        self.fit_rows_ = len(features)
        self.eligible_fit_rows_ = int(eligible.sum())
        self.mean_train_actual_cf_ = float(np.mean(eligible_target))
        self.observed_classes_ = tuple(int(value) for value in observed)
        self.class_counts_ = tuple(
            int(value)
            for value in np.bincount(classes, minlength=BIN_COUNT).tolist()
        )
        return self

    def predict_probability(self, features: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "model_"):
            raise RuntimeError("fit must be called before predict")
        if not isinstance(features, pd.DataFrame) or features.empty:
            raise TypeError("features must be a non-empty DataFrame")
        if tuple(features.columns) != self.feature_columns_:
            raise ValueError("application feature columns/order differ from fit")
        if not features.index.is_unique or not features.index.is_monotonic_increasing:
            raise ValueError("application index must be unique and increasing")
        if len(features.index.intersection(self.fit_index_)):
            raise ValueError("same-row fit/application prediction is forbidden")
        transformed = self.transform_.transform(features)
        raw = self.model_.predict_proba(transformed)
        fixed = fixed_probability_matrix(raw, self.model_.classes_)
        return pd.DataFrame(
            fixed,
            index=features.index,
            columns=[f"class_{value:02d}" for value in range(BIN_COUNT)],
        )

    def predict_action(
        self,
        features: pd.DataFrame,
        baseline_kwh: pd.Series,
    ) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
        if not features.index.equals(baseline_kwh.index):
            raise ValueError("application feature/baseline indexes differ")
        probability = self.predict_probability(features)
        action_cf, diagnostics = ordinal_bayes_action_cf(
            probability.to_numpy(dtype=np.float64),
            baseline_kwh.to_numpy(dtype=np.float64) / self.capacity_kwh_,
            mean_train_actual_cf=self.mean_train_actual_cf_,
            config=self.action_config,
        )
        diagnostics.index = features.index
        action = pd.Series(
            action_cf,
            index=features.index,
            name=baseline_kwh.name,
        )
        return action, probability, diagnostics

    def metadata(self) -> dict[str, Any]:
        if not hasattr(self, "model_"):
            raise RuntimeError("fit must be called before metadata")
        positive = np.asarray(self.class_counts_, dtype=np.int64)
        positive = positive[positive > 0]
        return {
            "model_parameters": dict(self.model_parameters),
            "fit_rows": self.fit_rows_,
            "eligible_fit_rows": self.eligible_fit_rows_,
            "feature_count": len(self.feature_columns_),
            "mean_train_actual_cf": self.mean_train_actual_cf_,
            "observed_classes": list(self.observed_classes_),
            "observed_class_count": len(self.observed_classes_),
            "class_counts_0_through_41": list(self.class_counts_),
            "minimum_positive_class_count": int(positive.min()),
            "median_positive_class_count": float(np.median(positive)),
            "action_delta_count": len(ACTION_DELTAS_CF),
            "transfer_weight": TRANSFER_WEIGHT,
            "fit_score_calculated": False,
        }


def group_stage1_pass(
    comparisons: Mapping[str, Mapping[str, Any]],
    group: str,
) -> tuple[bool, dict[str, Any]]:
    required = STAGE1_REQUIRED[group]
    if set(comparisons) != set(required):
        raise ValueError(f"{group} comparison segment set changed")
    deltas = {name: float(comparisons[name]["delta"]) for name in required}
    full = comparisons["full"]
    nmae = float(full["candidate"]["one_minus_nmae"]) - float(
        full["baseline"]["one_minus_nmae"]
    )
    ficr = float(full["candidate"]["ficr"]) - float(full["baseline"]["ficr"])
    passed = all(value > 0.0 for value in deltas.values()) and nmae >= 0.0 and ficr >= 0.0
    return passed, {
        "segment_total_score_deltas": deltas,
        "all_segment_total_score_deltas_strictly_positive": all(
            value > 0.0 for value in deltas.values()
        ),
        "full_delta_one_minus_nmae": nmae,
        "full_delta_ficr": ficr,
        "full_components_nonnegative": nmae >= 0.0 and ficr >= 0.0,
        "passed": passed,
    }


__all__ = (
    "ACTION_DELTAS_CF",
    "BIN_CENTRES_CF",
    "BIN_COUNT",
    "BIN_START_CF",
    "BIN_WIDTH_CF",
    "FullWeatherOrdinalBayes",
    "MODEL_PARAMETERS",
    "OrdinalActionConfig",
    "STAGE1_REQUIRED",
    "TRANSFER_WEIGHT",
    "expected_official_utility",
    "fixed_probability_matrix",
    "group_stage1_pass",
    "ordinal_bayes_action_cf",
    "target_classes",
    "transfer_action_kwh",
)

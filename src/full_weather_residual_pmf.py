"""Full-weather baseline-conditioned residual PMF with one fixed Bayes action."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from src.residual_histogram_bayes import (
    ACTION_DELTAS_CF,
    COMPONENT_NAMES,
    RESIDUAL_CENTRES_CF,
    expected_utility_actions,
    residual_class_index,
)
from src.weather_quantile import TrainOnlyMedianTransform


WEATHER_FEATURE_COUNT = 612
STATE_FEATURE_COLUMNS: tuple[str, ...] = (
    "state__primary_base_cf",
    "state__component_mean_cf",
    "state__component_std_cf",
)
TOTAL_FEATURE_COUNT = WEATHER_FEATURE_COUNT + len(STATE_FEATURE_COLUMNS)
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


def _frame(value: pd.DataFrame, *, name: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame) or value.empty:
        raise TypeError(f"{name} must be a non-empty DataFrame")
    if not value.index.is_unique or not value.index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be unique and increasing")
    if not value.columns.is_unique:
        raise ValueError(f"{name} columns must be unique")
    return value


def _series(value: pd.Series, *, name: str) -> pd.Series:
    if not isinstance(value, pd.Series) or value.empty:
        raise TypeError(f"{name} must be a non-empty Series")
    if not value.index.is_unique or not value.index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be unique and increasing")
    output = value.astype(np.float64)
    if not np.isfinite(output.to_numpy()).all():
        raise ValueError(f"{name} contains non-finite values")
    return output


def build_full_weather_state_features(
    weather: pd.DataFrame,
    primary_baseline_kwh: pd.Series,
    component_predictions_kwh: Mapping[str, pd.Series],
    *,
    capacity_kwh: float,
) -> pd.DataFrame:
    """Append only the frozen three baseline/component-state columns."""

    weather_frame = _frame(weather, name="weather")
    if weather_frame.shape[1] != WEATHER_FEATURE_COUNT:
        raise ValueError("weather must contain exactly 612 ordered features")
    if any(str(column) in STATE_FEATURE_COLUMNS for column in weather_frame.columns):
        raise ValueError("weather collides with frozen state feature names")
    forbidden = [
        str(column)
        for column in weather_frame.columns
        if any(
            token in str(column).lower()
            for token in ("actual", "target", "label", "scada", "power_kw")
        )
    ]
    if forbidden:
        raise ValueError(f"forbidden target-like weather features: {forbidden[:5]}")
    baseline = _series(primary_baseline_kwh, name="primary_baseline_kwh")
    if not weather_frame.index.equals(baseline.index):
        raise ValueError("weather and baseline indexes differ")
    if tuple(component_predictions_kwh) != COMPONENT_NAMES:
        raise ValueError("component order differs from frozen contract")
    capacity = float(capacity_kwh)
    if not np.isfinite(capacity) or capacity <= 0.0:
        raise ValueError("capacity_kwh must be positive and finite")
    components: list[np.ndarray] = []
    for name in COMPONENT_NAMES:
        component = _series(
            component_predictions_kwh[name], name=f"component_{name}"
        )
        if not component.index.equals(weather_frame.index):
            raise ValueError(f"component {name} index differs")
        components.append(component.to_numpy(dtype=np.float64) / capacity)
    matrix = np.column_stack(components)
    state = pd.DataFrame(
        {
            STATE_FEATURE_COLUMNS[0]: baseline.to_numpy(dtype=np.float64) / capacity,
            STATE_FEATURE_COLUMNS[1]: matrix.mean(axis=1),
            STATE_FEATURE_COLUMNS[2]: matrix.std(axis=1, ddof=0),
        },
        index=weather_frame.index,
        dtype=np.float32,
    )
    result = pd.concat((weather_frame.astype(np.float32), state), axis=1)
    if result.shape[1] != TOTAL_FEATURE_COUNT:
        raise AssertionError("full-weather state feature count changed")
    if tuple(result.columns[-3:]) != STATE_FEATURE_COLUMNS:
        raise AssertionError("state feature order changed")
    result.index.name = "forecast_kst_dtm"
    return result


def fixed_residual_probability_matrix(
    predict_proba: Any,
    model_classes: Any,
) -> np.ndarray:
    """Map observed LightGBM classes into the fixed 43-column vocabulary."""

    raw = np.asarray(predict_proba, dtype=np.float64)
    classes_raw = np.asarray(model_classes)
    if raw.ndim != 2 or classes_raw.ndim != 1:
        raise ValueError("probability/classes must be 2-D/1-D")
    if raw.shape[0] == 0 or raw.shape[1] != len(classes_raw):
        raise ValueError("probability/class shape mismatch")
    if not np.isfinite(raw).all() or np.any(raw < 0.0):
        raise ValueError("probabilities must be finite and nonnegative")
    classes_float = classes_raw.astype(np.float64)
    classes = classes_float.astype(np.int64)
    if not np.array_equal(classes_float, classes.astype(np.float64)):
        raise ValueError("model classes must be exact integers")
    if len(np.unique(classes)) != len(classes) or np.any(classes < 0) or np.any(
        classes >= len(RESIDUAL_CENTRES_CF)
    ):
        raise ValueError("model classes outside fixed residual vocabulary")
    output = np.zeros((len(raw), len(RESIDUAL_CENTRES_CF)), dtype=np.float64)
    output[:, classes] = raw
    if not np.all(np.abs(output.sum(axis=1) - 1.0) <= 1e-12):
        raise ValueError("fixed residual probabilities do not sum to one")
    return output


def transfer_residual_action_kwh(
    baseline_kwh: pd.Series,
    raw_action_delta_cf: pd.Series,
    *,
    capacity_kwh: float,
) -> pd.Series:
    """Apply only the frozen 0.10 residual-action transfer."""

    baseline = _series(baseline_kwh, name="baseline_kwh")
    action = _series(raw_action_delta_cf, name="raw_action_delta_cf")
    if not baseline.index.equals(action.index):
        raise ValueError("baseline/action indexes differ")
    capacity = float(capacity_kwh)
    candidate_cf = np.clip(
        baseline.to_numpy(dtype=np.float64) / capacity
        + TRANSFER_WEIGHT * action.to_numpy(dtype=np.float64),
        0.0,
        1.02,
    )
    return pd.Series(
        candidate_cf * capacity,
        index=baseline.index,
        name=baseline.name,
    )


class FullWeatherResidualPMF:
    """One frozen LightGBM residual PMF and exact expected-utility action."""

    def __init__(self) -> None:
        self.model_parameters = dict(MODEL_PARAMETERS)
        self.minimum_fit_rows = 300

    def fit(
        self,
        features: pd.DataFrame,
        actual_kwh: pd.Series,
        baseline_kwh: pd.Series,
        *,
        capacity_kwh: float,
    ) -> "FullWeatherResidualPMF":
        frame = _frame(features, name="fit_features")
        if frame.shape[1] != TOTAL_FEATURE_COUNT or tuple(
            frame.columns[-3:]
        ) != STATE_FEATURE_COLUMNS:
            raise ValueError("fit feature schema differs from frozen contract")
        if not isinstance(actual_kwh, pd.Series) or not actual_kwh.index.equals(
            frame.index
        ):
            raise ValueError("fit actual index differs")
        actual = actual_kwh.astype(np.float64)
        if np.isinf(actual.to_numpy()).any():
            raise ValueError("fit actual contains infinity")
        baseline = _series(baseline_kwh, name="fit_baseline_kwh")
        if not baseline.index.equals(frame.index):
            raise ValueError("fit baseline index differs")
        capacity = float(capacity_kwh)
        if not np.isfinite(capacity) or capacity <= 0.0:
            raise ValueError("capacity_kwh must be positive and finite")
        actual_cf = actual.to_numpy(dtype=np.float64) / capacity
        baseline_cf = baseline.to_numpy(dtype=np.float64) / capacity
        eligible = np.isfinite(actual_cf) & (actual_cf >= 0.10)
        if int(eligible.sum()) < self.minimum_fit_rows:
            raise ValueError("too few eligible residual-PMF fit rows")
        eligible_features = frame.loc[eligible]
        residual = actual_cf[eligible] - baseline_cf[eligible]
        classes = residual_class_index(residual)
        if np.unique(classes).size < 3:
            raise ValueError("residual PMF needs at least three observed classes")
        self.transform_ = TrainOnlyMedianTransform().fit(eligible_features)
        transformed = self.transform_.transform(eligible_features)
        self.model_ = LGBMClassifier(**self.model_parameters)
        self.model_.fit(transformed, classes)
        self.capacity_kwh_ = capacity
        self.fit_index_ = frame.index.copy()
        self.feature_columns_ = tuple(frame.columns)
        self.fit_rows_ = len(frame)
        self.eligible_fit_rows_ = int(eligible.sum())
        self.mean_train_actual_cf_ = float(np.mean(actual_cf[eligible]))
        self.observed_classes_ = tuple(int(value) for value in self.model_.classes_)
        self.class_counts_ = tuple(
            int(value)
            for value in np.bincount(classes, minlength=43).tolist()
        )
        self.residual_min_cf_ = float(np.min(residual))
        self.residual_max_cf_ = float(np.max(residual))
        return self

    def predict_probability(self, features: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "model_"):
            raise RuntimeError("fit must be called before predict")
        frame = _frame(features, name="application_features")
        if tuple(frame.columns) != self.feature_columns_:
            raise ValueError("application feature schema/order differs")
        if len(frame.index.intersection(self.fit_index_)):
            raise ValueError("fit/application overlap is forbidden")
        transformed = self.transform_.transform(frame)
        fixed = fixed_residual_probability_matrix(
            self.model_.predict_proba(transformed), self.model_.classes_
        )
        return pd.DataFrame(
            fixed,
            index=frame.index,
            columns=[f"residual_class_{value:02d}" for value in range(43)],
        )

    def predict_action(
        self,
        features: pd.DataFrame,
        baseline_kwh: pd.Series,
    ) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
        if not features.index.equals(baseline_kwh.index):
            raise ValueError("application feature/baseline indexes differ")
        probability = self.predict_probability(features)
        base_cf = baseline_kwh.to_numpy(dtype=np.float64) / self.capacity_kwh_
        action, utility = expected_utility_actions(
            probability.to_numpy(dtype=np.float64),
            base_cf,
            mean_actual_cf=self.mean_train_actual_cf_,
        )
        # residual_histogram_bayes uses an algebraically equivalent utility
        # multiplied by two.  Keep its action/tie implementation, but persist
        # the exact preregistered -0.5*AE + settlement/(8*mean) scale.
        utility = 0.5 * utility
        action_series = pd.Series(
            action, index=features.index, name="raw_action_delta_cf"
        )
        diagnostics = pd.DataFrame(
            {
                "primary_baseline_cf": base_cf,
                "raw_action_delta_cf": action,
                "raw_action_cf": np.clip(base_cf + action, 0.0, 1.02),
                "expected_utility": utility,
            },
            index=features.index,
        )
        return action_series, probability, diagnostics

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
            "state_feature_columns": list(STATE_FEATURE_COLUMNS),
            "mean_train_actual_cf": self.mean_train_actual_cf_,
            "observed_classes": list(self.observed_classes_),
            "observed_class_count": len(self.observed_classes_),
            "class_counts_0_through_42": list(self.class_counts_),
            "minimum_positive_class_count": int(positive.min()),
            "median_positive_class_count": float(np.median(positive)),
            "residual_min_cf": self.residual_min_cf_,
            "residual_max_cf": self.residual_max_cf_,
            "action_delta_count": len(ACTION_DELTAS_CF),
            "transfer_weight": TRANSFER_WEIGHT,
            "fit_score_calculated": False,
        }


__all__ = (
    "MODEL_PARAMETERS",
    "STATE_FEATURE_COLUMNS",
    "TOTAL_FEATURE_COUNT",
    "TRANSFER_WEIGHT",
    "FullWeatherResidualPMF",
    "build_full_weather_state_features",
    "fixed_residual_probability_matrix",
    "transfer_residual_action_kwh",
)

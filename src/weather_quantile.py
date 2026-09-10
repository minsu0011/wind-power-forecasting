"""Direct weather-to-quantile surfaces and official-metric Bayes actions.

The model predicts target capacity-factor quantiles directly from weather
features.  Every fitted transform is learned on eligible training rows only.
Application indexes must be disjoint from the fit index, and predicted
quantiles are monotonized before the point action is selected.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BayesActionConfig:
    quantile_levels: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)
    interpolation_count: int = 33
    outcome_lower_cf: float = 0.10
    outcome_upper_cf: float = 1.20
    candidate_lower_cf: float = 0.0
    candidate_upper_cf: float = 1.02
    candidate_step_cf: float = 0.01


class TrainOnlyMedianTransform:
    """Column/order lock plus training-only median imputation."""

    def fit(self, features: pd.DataFrame) -> "TrainOnlyMedianTransform":
        frame = _feature_frame(features, name="transform_fit_features")
        self.columns_ = tuple(frame.columns)
        self.fit_index_ = frame.index.copy()
        numeric = frame.replace([np.inf, -np.inf], np.nan)
        medians = numeric.median(axis=0, skipna=True)
        if medians.isna().any():
            missing = medians.index[medians.isna()].tolist()
            raise ValueError(f"all-missing transform columns: {missing[:5]}")
        self.medians_ = medians.astype(float)
        self.fit_rows_ = len(frame)
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "columns_"):
            raise RuntimeError("fit must be called before transform")
        frame = _feature_frame(features, name="transform_application_features")
        if tuple(frame.columns) != self.columns_:
            raise ValueError("feature columns/order differ from fitted transform")
        result = frame.replace([np.inf, -np.inf], np.nan).fillna(self.medians_)
        values = result.to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("transformed features remain non-finite")
        return pd.DataFrame(
            values,
            index=frame.index,
            columns=frame.columns,
            dtype=np.float32,
        )


def _feature_frame(values: Any, *, name: str) -> pd.DataFrame:
    if not isinstance(values, pd.DataFrame) or values.empty:
        raise TypeError(f"{name} must be a non-empty pandas DataFrame")
    if not values.index.is_unique or not values.index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be unique and monotonically increasing")
    if not values.columns.is_unique:
        raise ValueError(f"{name} columns must be unique")
    return values.astype(np.float32)


def repair_quantile_crossing(values: np.ndarray) -> tuple[np.ndarray, int]:
    raw = np.asarray(values, dtype=float)
    if raw.ndim != 2 or raw.shape[1] < 2 or not np.isfinite(raw).all():
        raise ValueError("quantile predictions must be a finite 2-D matrix")
    crossing_rows = int(np.any(np.diff(raw, axis=1) < 0.0, axis=1).sum())
    repaired = np.sort(raw, axis=1, kind="stable")
    if np.any(np.diff(repaired, axis=1) < 0.0):
        raise AssertionError("quantile crossing repair failed")
    return repaired, crossing_rows


def conditional_bayes_action_cf(
    repaired_quantiles_cf: np.ndarray,
    baseline_cf: np.ndarray,
    *,
    mean_train_actual_cf: float,
    config: BayesActionConfig | None = None,
) -> np.ndarray:
    """Choose the candidate with maximum expected official row utility."""

    settings = config or BayesActionConfig()
    quantiles = np.asarray(repaired_quantiles_cf, dtype=float)
    baseline = np.asarray(baseline_cf, dtype=float)
    levels = np.asarray(settings.quantile_levels, dtype=float)
    if quantiles.ndim != 2 or quantiles.shape[1] != len(levels):
        raise ValueError("quantile matrix width differs from quantile_levels")
    if baseline.shape != (len(quantiles),):
        raise ValueError("baseline_cf shape differs from quantile rows")
    if not np.isfinite(quantiles).all() or not np.isfinite(baseline).all():
        raise ValueError("Bayes action inputs must be finite")
    if np.any(np.diff(quantiles, axis=1) < 0.0):
        raise ValueError("quantiles must be crossing-repaired before action")
    mean_actual = float(mean_train_actual_cf)
    if not np.isfinite(mean_actual) or mean_actual <= 0.0:
        raise ValueError("mean_train_actual_cf must be positive and finite")
    probabilities = np.linspace(
        levels[0], levels[-1], settings.interpolation_count, endpoint=True
    )
    absolute_grid = np.arange(
        settings.candidate_lower_cf,
        settings.candidate_upper_cf + settings.candidate_step_cf / 2.0,
        settings.candidate_step_cf,
        dtype=float,
    )
    actions = np.empty(len(quantiles), dtype=float)
    for row in range(len(quantiles)):
        outcomes = np.interp(probabilities, levels, quantiles[row])
        outcomes = np.clip(
            outcomes, settings.outcome_lower_cf, settings.outcome_upper_cf
        )
        candidates = np.unique(
            np.clip(
                np.concatenate(
                    [absolute_grid, [baseline[row]], quantiles[row]]
                ),
                settings.candidate_lower_cf,
                settings.candidate_upper_cf,
            )
        )
        error = np.abs(candidates[:, None] - outcomes[None, :])
        settlement = np.select(
            [error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0
        )
        utility = (
            -0.5 * error
            + outcomes[None, :] * settlement / (8.0 * mean_actual)
        ).mean(axis=1)
        maximum = float(np.max(utility))
        tied = np.flatnonzero(utility >= maximum - 1e-12)
        order = np.lexsort(
            (candidates[tied], np.abs(candidates[tied] - baseline[row]))
        )
        actions[row] = candidates[tied[int(order[0])]]
    return np.clip(
        actions, settings.candidate_lower_cf, settings.candidate_upper_cf
    )


def blend_with_baseline_kwh(
    baseline_kwh: pd.Series,
    action_kwh: pd.Series,
    *,
    weight: float,
    capacity_kwh: float,
) -> pd.Series:
    if not baseline_kwh.index.equals(action_kwh.index):
        raise ValueError("baseline/action indexes differ")
    blend_weight = float(weight)
    if not 0.0 <= blend_weight <= 1.0:
        raise ValueError("weight must lie in [0,1]")
    baseline = baseline_kwh.to_numpy(dtype=float)
    action = action_kwh.to_numpy(dtype=float)
    if not np.isfinite(baseline).all() or not np.isfinite(action).all():
        raise ValueError("baseline/action predictions must be finite")
    if blend_weight == 0.0:
        values = baseline.copy()
    else:
        values = (1.0 - blend_weight) * baseline + blend_weight * action
    values = np.clip(values, 0.0, 1.02 * float(capacity_kwh))
    return pd.Series(values, index=baseline_kwh.index, name=baseline_kwh.name)


class WeatherQuantileSurface:
    """Five LightGBM target-CF quantiles fit directly on weather features."""

    def __init__(
        self,
        *,
        quantile_levels: Sequence[float],
        model_parameters: Mapping[str, Any],
        random_state: int,
        n_jobs: int,
        minimum_actual_cf: float = 0.10,
        action_config: BayesActionConfig | None = None,
    ) -> None:
        self.quantile_levels = tuple(float(value) for value in quantile_levels)
        self.model_parameters = dict(model_parameters)
        self.random_state = int(random_state)
        self.n_jobs = int(n_jobs)
        self.minimum_actual_cf = float(minimum_actual_cf)
        self.action_config = action_config or BayesActionConfig(
            quantile_levels=self.quantile_levels
        )
        if len(self.quantile_levels) != 5 or np.any(
            np.diff(self.quantile_levels) <= 0.0
        ):
            raise ValueError("exactly five increasing quantile levels are required")

    def fit(
        self,
        features: pd.DataFrame,
        actual_kwh: pd.Series,
        *,
        capacity_kwh: float,
    ) -> "WeatherQuantileSurface":
        from lightgbm import LGBMRegressor

        frame = _feature_frame(features, name="fit_features")
        if not isinstance(actual_kwh, pd.Series) or not frame.index.equals(
            actual_kwh.index
        ):
            raise ValueError("fit actual index differs from features")
        capacity = float(capacity_kwh)
        if not np.isfinite(capacity) or capacity <= 0.0:
            raise ValueError("capacity_kwh must be positive and finite")
        actual_cf = actual_kwh.to_numpy(dtype=float) / capacity
        eligible = np.isfinite(actual_cf) & (actual_cf >= self.minimum_actual_cf)
        if int(eligible.sum()) < 300:
            raise ValueError(f"too few eligible training rows: {int(eligible.sum())}")
        eligible_features = frame.loc[eligible]
        target = actual_cf[eligible]
        self.transform_ = TrainOnlyMedianTransform().fit(eligible_features)
        transformed = self.transform_.transform(eligible_features)
        self.models_: dict[float, Any] = {}
        for level in self.quantile_levels:
            parameters = {
                **self.model_parameters,
                "objective": "quantile",
                "alpha": float(level),
                "random_state": self.random_state,
                "n_jobs": self.n_jobs,
                "device_type": "cpu",
            }
            model = LGBMRegressor(**parameters)
            model.fit(transformed, target)
            self.models_[level] = model
        self.capacity_kwh_ = capacity
        self.fit_index_ = frame.index.copy()
        self.fit_rows_ = len(frame)
        self.eligible_fit_rows_ = int(eligible.sum())
        self.mean_train_actual_cf_ = float(np.mean(target))
        self.feature_columns_ = tuple(frame.columns)
        return self

    def predict_quantiles(self, features: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "models_"):
            raise RuntimeError("fit must be called before predict")
        frame = _feature_frame(features, name="application_features")
        if tuple(frame.columns) != self.feature_columns_:
            raise ValueError("application feature columns/order differ from fit")
        overlap = frame.index.intersection(self.fit_index_)
        if len(overlap):
            raise ValueError(
                "application rows overlap fit rows; same-row prediction is forbidden"
            )
        transformed = self.transform_.transform(frame)
        raw = np.column_stack(
            [self.models_[level].predict(transformed) for level in self.quantile_levels]
        )
        repaired, crossings = repair_quantile_crossing(raw)
        self.last_raw_crossing_rows_ = crossings
        self.last_application_rows_ = len(frame)
        columns = [f"q{int(round(level * 100)):02d}_cf" for level in self.quantile_levels]
        return pd.DataFrame(repaired, index=frame.index, columns=columns)

    def predict_action(
        self,
        features: pd.DataFrame,
        baseline_kwh: pd.Series,
    ) -> tuple[pd.Series, pd.DataFrame]:
        if not features.index.equals(baseline_kwh.index):
            raise ValueError("application baseline index differs from features")
        quantiles = self.predict_quantiles(features)
        action_cf = conditional_bayes_action_cf(
            quantiles.to_numpy(dtype=float),
            baseline_kwh.to_numpy(dtype=float) / self.capacity_kwh_,
            mean_train_actual_cf=self.mean_train_actual_cf_,
            config=self.action_config,
        )
        action = pd.Series(
            action_cf * self.capacity_kwh_,
            index=features.index,
            name=baseline_kwh.name,
        )
        return action, quantiles

    def metadata(self) -> dict[str, Any]:
        if not hasattr(self, "models_"):
            raise RuntimeError("fit must be called before metadata")
        return {
            "quantile_levels": list(self.quantile_levels),
            "model_parameters": self.model_parameters,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "minimum_actual_cf": self.minimum_actual_cf,
            "action_config": asdict(self.action_config),
            "fit_rows": self.fit_rows_,
            "eligible_fit_rows": self.eligible_fit_rows_,
            "features": len(self.feature_columns_),
            "mean_train_actual_cf": self.mean_train_actual_cf_,
            "transform_fit_rows": self.transform_.fit_rows_,
            "transform_fit_on_eligible_rows_only": True,
            "fit_score_calculated": False,
            "last_raw_crossing_rows": getattr(
                self, "last_raw_crossing_rows_", None
            ),
            "last_application_rows": getattr(self, "last_application_rows_", None),
        }


__all__ = (
    "BayesActionConfig",
    "TrainOnlyMedianTransform",
    "WeatherQuantileSurface",
    "blend_with_baseline_kwh",
    "conditional_bayes_action_cf",
    "repair_quantile_crossing",
)

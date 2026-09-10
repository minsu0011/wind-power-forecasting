"""CatBoost MultiQuantile surface with the locked official-utility action.

This module is intentionally estimator-specific: one CatBoost model predicts
all five conditional target-capacity-factor quantiles jointly.  Feature
transforms are fitted on eligible training rows only, and applications must be
strictly later than and disjoint from the fit index.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.weather_quantile import (
    TrainOnlyMedianTransform,
    blend_with_baseline_kwh,
    repair_quantile_crossing,
)


@dataclass(frozen=True)
class CatBoostBayesActionConfig:
    quantile_levels: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)
    interpolation_count: int = 33
    outcome_lower_cf: float = 0.10
    outcome_upper_cf: float = 1.20
    absolute_grid_lower_cf: float = 0.01
    absolute_grid_upper_cf: float = 1.02
    absolute_grid_step_cf: float = 0.01
    candidate_lower_cf: float = 0.0
    candidate_upper_cf: float = 1.02
    tie_tolerance: float = 1e-12


def exact_official_utility_action_cf(
    repaired_quantiles_cf: np.ndarray,
    baseline_cf: np.ndarray,
    *,
    mean_train_actual_cf: float,
    config: CatBoostBayesActionConfig | None = None,
) -> np.ndarray:
    """Maximize the registered conditional expectation of official utility."""

    settings = config or CatBoostBayesActionConfig()
    quantiles = np.asarray(repaired_quantiles_cf, dtype=np.float64)
    baseline = np.asarray(baseline_cf, dtype=np.float64)
    levels = np.asarray(settings.quantile_levels, dtype=np.float64)
    if quantiles.ndim != 2 or quantiles.shape[1] != len(levels):
        raise ValueError("quantile matrix width differs from quantile levels")
    if baseline.shape != (len(quantiles),):
        raise ValueError("baseline shape differs from quantile rows")
    if not np.isfinite(quantiles).all() or not np.isfinite(baseline).all():
        raise ValueError("action inputs must be finite")
    if np.any(np.diff(quantiles, axis=1) < 0.0):
        raise ValueError("quantiles must be crossing-repaired before action")
    if len(levels) != 5 or np.any(np.diff(levels) <= 0.0):
        raise ValueError("exactly five increasing quantile levels are required")
    mean_actual = float(mean_train_actual_cf)
    if not np.isfinite(mean_actual) or mean_actual <= 0.0:
        raise ValueError("mean_train_actual_cf must be positive and finite")
    if not (
        0.0 <= settings.candidate_lower_cf
        <= settings.absolute_grid_lower_cf
        <= settings.absolute_grid_upper_cf
        <= settings.candidate_upper_cf
    ):
        raise ValueError("registered action bounds are inconsistent")

    probabilities = np.linspace(
        levels[0], levels[-1], settings.interpolation_count, endpoint=True
    )
    absolute_grid = np.arange(
        settings.absolute_grid_lower_cf,
        settings.absolute_grid_upper_cf + settings.absolute_grid_step_cf / 2.0,
        settings.absolute_grid_step_cf,
        dtype=np.float64,
    )
    if absolute_grid[0] != 0.01 or absolute_grid[-1] != 1.02:
        raise AssertionError("registered 0.01..1.02 action grid changed")

    actions = np.empty(len(quantiles), dtype=np.float64)
    for row in range(len(quantiles)):
        outcomes = np.interp(probabilities, levels, quantiles[row])
        outcomes = np.clip(
            outcomes, settings.outcome_lower_cf, settings.outcome_upper_cf
        )
        candidates = np.unique(
            np.clip(
                np.concatenate((absolute_grid, [baseline[row]], quantiles[row])),
                settings.candidate_lower_cf,
                settings.candidate_upper_cf,
            )
        )
        error = np.abs(candidates[:, None] - outcomes[None, :])
        settlement = np.select(
            (error <= 0.06, error <= 0.08), (4.0, 3.0), default=0.0
        )
        utility = (
            -error
            + outcomes[None, :] * settlement / (4.0 * mean_actual)
        ).mean(axis=1)
        maximum = float(np.max(utility))
        tied = np.flatnonzero(utility >= maximum - settings.tie_tolerance)
        order = np.lexsort(
            (candidates[tied], np.abs(candidates[tied] - baseline[row]))
        )
        actions[row] = candidates[tied[int(order[0])]]
    return np.clip(
        actions, settings.candidate_lower_cf, settings.candidate_upper_cf
    )


def exact_official_residual_utility_action_cf(
    repaired_residual_quantiles_cf: np.ndarray,
    baseline_cf: np.ndarray,
    *,
    mean_train_actual_cf: float,
    config: CatBoostBayesActionConfig | None = None,
) -> np.ndarray:
    """Exact registered action for a conditional residual distribution.

    The five model outputs are quantiles of ``actual_cf - baseline_cf``.
    Interpolation therefore happens in residual space and the row's immutable
    baseline is added before the official utility is evaluated.
    """

    settings = config or CatBoostBayesActionConfig()
    residual_quantiles = np.asarray(
        repaired_residual_quantiles_cf, dtype=np.float64
    )
    baseline = np.asarray(baseline_cf, dtype=np.float64)
    levels = np.asarray(settings.quantile_levels, dtype=np.float64)
    if residual_quantiles.ndim != 2 or residual_quantiles.shape[1] != len(levels):
        raise ValueError("residual quantile matrix width differs from quantile levels")
    if baseline.shape != (len(residual_quantiles),):
        raise ValueError("baseline shape differs from residual quantile rows")
    if not np.isfinite(residual_quantiles).all() or not np.isfinite(baseline).all():
        raise ValueError("action inputs must be finite")
    if np.any(np.diff(residual_quantiles, axis=1) < 0.0):
        raise ValueError("residual quantiles must be crossing-repaired before action")
    mean_actual = float(mean_train_actual_cf)
    if not np.isfinite(mean_actual) or mean_actual <= 0.0:
        raise ValueError("mean_train_actual_cf must be positive and finite")

    probabilities = np.linspace(
        levels[0], levels[-1], settings.interpolation_count, endpoint=True
    )
    absolute_grid = np.arange(
        settings.absolute_grid_lower_cf,
        settings.absolute_grid_upper_cf + settings.absolute_grid_step_cf / 2.0,
        settings.absolute_grid_step_cf,
        dtype=np.float64,
    )
    if absolute_grid[0] != 0.01 or absolute_grid[-1] != 1.02:
        raise AssertionError("registered 0.01..1.02 action grid changed")

    actions = np.empty(len(residual_quantiles), dtype=np.float64)
    for row in range(len(residual_quantiles)):
        residual_outcomes = np.interp(
            probabilities, levels, residual_quantiles[row]
        )
        outcomes = np.clip(
            baseline[row] + residual_outcomes,
            settings.outcome_lower_cf,
            settings.outcome_upper_cf,
        )
        candidates = np.unique(
            np.clip(
                np.concatenate(
                    (
                        absolute_grid,
                        [baseline[row]],
                        baseline[row] + residual_quantiles[row],
                    )
                ),
                settings.candidate_lower_cf,
                settings.candidate_upper_cf,
            )
        )
        error = np.abs(candidates[:, None] - outcomes[None, :])
        settlement = np.select(
            (error <= 0.06, error <= 0.08), (4.0, 3.0), default=0.0
        )
        utility = (
            -error + outcomes[None, :] * settlement / (4.0 * mean_actual)
        ).mean(axis=1)
        maximum = float(np.max(utility))
        tied = np.flatnonzero(utility >= maximum - settings.tie_tolerance)
        order = np.lexsort(
            (candidates[tied], np.abs(candidates[tied] - baseline[row]))
        )
        actions[row] = candidates[tied[int(order[0])]]
    return np.clip(
        actions, settings.candidate_lower_cf, settings.candidate_upper_cf
    )


class CatBoostMultiQuantileSurface:
    """One joint CatBoost MultiQuantile target-CF model."""

    def __init__(
        self,
        *,
        quantile_levels: Sequence[float],
        loss_function: str,
        model_parameters: Mapping[str, Any],
        minimum_actual_cf: float = 0.10,
        action_config: CatBoostBayesActionConfig | None = None,
    ) -> None:
        self.quantile_levels = tuple(float(value) for value in quantile_levels)
        self.loss_function = str(loss_function)
        self.model_parameters = dict(model_parameters)
        self.minimum_actual_cf = float(minimum_actual_cf)
        self.action_config = action_config or CatBoostBayesActionConfig(
            quantile_levels=self.quantile_levels
        )
        if self.quantile_levels != (0.10, 0.25, 0.50, 0.75, 0.90):
            raise ValueError("registered five quantile levels changed")
        expected_loss = "MultiQuantile:alpha=0.10,0.25,0.50,0.75,0.90"
        if self.loss_function != expected_loss:
            raise ValueError("registered MultiQuantile loss changed")

    def fit(
        self,
        features: pd.DataFrame,
        actual_kwh: pd.Series,
        *,
        capacity_kwh: float,
    ) -> "CatBoostMultiQuantileSurface":
        from catboost import CatBoostRegressor, __version__ as catboost_version

        frame = self._feature_frame(features, name="fit_features")
        if not isinstance(actual_kwh, pd.Series) or not frame.index.equals(
            actual_kwh.index
        ):
            raise ValueError("fit actual index differs from features")
        capacity = float(capacity_kwh)
        if not np.isfinite(capacity) or capacity <= 0.0:
            raise ValueError("capacity must be positive and finite")
        actual_cf = actual_kwh.to_numpy(dtype=np.float64) / capacity
        eligible = np.isfinite(actual_cf) & (actual_cf >= self.minimum_actual_cf)
        if int(eligible.sum()) < 300:
            raise ValueError(f"too few eligible training rows: {int(eligible.sum())}")
        eligible_features = frame.loc[eligible]
        target = actual_cf[eligible]
        self.transform_ = TrainOnlyMedianTransform().fit(eligible_features)
        transformed = self.transform_.transform(eligible_features)
        self.model_ = CatBoostRegressor(
            loss_function=self.loss_function, **self.model_parameters
        )
        self.model_.fit(transformed, target)
        expected_iterations = int(self.model_parameters["iterations"])
        if int(self.model_.tree_count_) != expected_iterations:
            raise AssertionError("CatBoost tree count differs from preregistration")
        self.catboost_version_ = str(catboost_version)
        self.capacity_kwh_ = capacity
        self.fit_index_ = frame.index.copy()
        self.fit_rows_ = len(frame)
        self.eligible_fit_rows_ = int(eligible.sum())
        self.mean_train_actual_cf_ = float(np.mean(target))
        self.feature_columns_ = tuple(frame.columns)
        return self

    def predict_quantiles(self, features: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "model_"):
            raise RuntimeError("fit must be called before predict")
        frame = self._feature_frame(features, name="application_features")
        if tuple(frame.columns) != self.feature_columns_:
            raise ValueError("application feature columns/order differ from fit")
        overlap = frame.index.intersection(self.fit_index_)
        if len(overlap):
            raise ValueError("application rows overlap fit rows")
        if not self.fit_index_.max() < frame.index.min():
            raise ValueError("fit end must be strictly before application start")
        transformed = self.transform_.transform(frame)
        raw = np.asarray(self.model_.predict(transformed), dtype=np.float64)
        if raw.shape != (len(frame), len(self.quantile_levels)):
            raise AssertionError(
                f"CatBoost quantile output shape changed: {raw.shape}"
            )
        repaired, crossings = repair_quantile_crossing(raw)
        self.last_raw_quantiles_ = raw
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
        action_cf = exact_official_utility_action_cf(
            quantiles.to_numpy(dtype=np.float64),
            baseline_kwh.to_numpy(dtype=np.float64) / self.capacity_kwh_,
            mean_train_actual_cf=self.mean_train_actual_cf_,
            config=self.action_config,
        )
        return (
            pd.Series(
                action_cf * self.capacity_kwh_,
                index=features.index,
                name=baseline_kwh.name,
            ),
            quantiles,
        )

    def metadata(self) -> dict[str, Any]:
        if not hasattr(self, "model_"):
            raise RuntimeError("fit must be called before metadata")
        return {
            "estimator": "catboost.CatBoostRegressor",
            "catboost_version": self.catboost_version_,
            "model_count": 1,
            "loss_function": self.loss_function,
            "quantile_levels": list(self.quantile_levels),
            "prediction_column_order": [
                f"q{int(round(level * 100)):02d}_cf"
                for level in self.quantile_levels
            ],
            "model_parameters": self.model_parameters,
            "minimum_actual_cf": self.minimum_actual_cf,
            "action_config": asdict(self.action_config),
            "fit_rows": self.fit_rows_,
            "eligible_fit_rows": self.eligible_fit_rows_,
            "features": len(self.feature_columns_),
            "mean_train_actual_cf": self.mean_train_actual_cf_,
            "transform_fit_rows": self.transform_.fit_rows_,
            "transform_fit_on_eligible_rows_only": True,
            "fit_score_calculated": False,
            "fit_end_before_application_start": True,
            "last_raw_crossing_rows": getattr(
                self, "last_raw_crossing_rows_", None
            ),
            "last_application_rows": getattr(self, "last_application_rows_", None),
        }

    @staticmethod
    def _feature_frame(values: Any, *, name: str) -> pd.DataFrame:
        if not isinstance(values, pd.DataFrame) or values.empty:
            raise TypeError(f"{name} must be a non-empty pandas DataFrame")
        if not values.index.is_unique or not values.index.is_monotonic_increasing:
            raise ValueError(f"{name} index must be unique and increasing")
        if not values.columns.is_unique:
            raise ValueError(f"{name} columns must be unique")
        return values.astype(np.float32)


class CatBoostResidualMultiQuantileSurface(CatBoostMultiQuantileSurface):
    """One joint CatBoost model for corrected-v3 residual quantiles."""

    def fit(
        self,
        features: pd.DataFrame,
        actual_kwh: pd.Series,
        baseline_kwh: pd.Series,
        *,
        capacity_kwh: float,
    ) -> "CatBoostResidualMultiQuantileSurface":
        from catboost import CatBoostRegressor, __version__ as catboost_version

        frame = self._feature_frame(features, name="fit_features")
        if not isinstance(actual_kwh, pd.Series) or not frame.index.equals(
            actual_kwh.index
        ):
            raise ValueError("fit actual index differs from features")
        if not isinstance(baseline_kwh, pd.Series) or not frame.index.equals(
            baseline_kwh.index
        ):
            raise ValueError("fit baseline index differs from features")
        capacity = float(capacity_kwh)
        if not np.isfinite(capacity) or capacity <= 0.0:
            raise ValueError("capacity must be positive and finite")
        actual_cf = actual_kwh.to_numpy(dtype=np.float64) / capacity
        baseline_cf = baseline_kwh.to_numpy(dtype=np.float64) / capacity
        eligible = (
            np.isfinite(actual_cf)
            & np.isfinite(baseline_cf)
            & (actual_cf >= self.minimum_actual_cf)
        )
        minimum_rows = int(getattr(self, "minimum_fit_rows", 800))
        if int(eligible.sum()) < minimum_rows:
            raise ValueError(f"too few eligible training rows: {int(eligible.sum())}")
        eligible_features = frame.loc[eligible]
        target = actual_cf[eligible] - baseline_cf[eligible]
        self.transform_ = TrainOnlyMedianTransform().fit(eligible_features)
        transformed = self.transform_.transform(eligible_features)
        self.model_ = CatBoostRegressor(
            loss_function=self.loss_function, **self.model_parameters
        )
        self.model_.fit(transformed, target)
        expected_iterations = int(self.model_parameters["iterations"])
        if int(self.model_.tree_count_) != expected_iterations:
            raise AssertionError("CatBoost tree count differs from preregistration")
        self.catboost_version_ = str(catboost_version)
        self.capacity_kwh_ = capacity
        self.fit_index_ = frame.index.copy()
        self.fit_rows_ = len(frame)
        self.eligible_fit_rows_ = int(eligible.sum())
        self.mean_train_actual_cf_ = float(np.mean(actual_cf[eligible]))
        self.feature_columns_ = tuple(frame.columns)
        self.target_kind_ = "actual_cf_minus_corrected_v3_cf"
        return self

    def predict_action(
        self,
        features: pd.DataFrame,
        baseline_kwh: pd.Series,
    ) -> tuple[pd.Series, pd.DataFrame]:
        if not features.index.equals(baseline_kwh.index):
            raise ValueError("application baseline index differs from features")
        residual_quantiles = self.predict_quantiles(features)
        action_cf = exact_official_residual_utility_action_cf(
            residual_quantiles.to_numpy(dtype=np.float64),
            baseline_kwh.to_numpy(dtype=np.float64) / self.capacity_kwh_,
            mean_train_actual_cf=self.mean_train_actual_cf_,
            config=self.action_config,
        )
        return (
            pd.Series(
                action_cf * self.capacity_kwh_,
                index=features.index,
                name=baseline_kwh.name,
            ),
            residual_quantiles,
        )

    def metadata(self) -> dict[str, Any]:
        result = super().metadata()
        result.update(
            {
                "target_kind": self.target_kind_,
                "minimum_eligible_fit_rows": int(
                    getattr(self, "minimum_fit_rows", 800)
                ),
                "baseline_conditioned": True,
            }
        )
        return result


__all__ = (
    "CatBoostBayesActionConfig",
    "CatBoostMultiQuantileSurface",
    "CatBoostResidualMultiQuantileSurface",
    "TrainOnlyMedianTransform",
    "blend_with_baseline_kwh",
    "exact_official_utility_action_cf",
    "exact_official_residual_utility_action_cf",
    "repair_quantile_crossing",
)

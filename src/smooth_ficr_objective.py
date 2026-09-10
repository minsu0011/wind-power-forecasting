"""Smooth approximation to the official BARAM FICR plus 1-NMAE utility.

The registered loss uses the exact official relative coefficients and replaces
the two settlement indicators by fixed-temperature logistic transitions.  Its
analytic gradient is exact.  Because the smooth threshold terms make the true
second derivative non-convex, tree boosters receive a declared, strictly
positive curvature-magnitude surrogate instead of a falsely labelled Hessian.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.weather_quantile import TrainOnlyMedianTransform


@dataclass(frozen=True)
class SmoothFICRLossConfig:
    thresholds_cf: tuple[float, float] = (0.06, 0.08)
    settlement_weights: tuple[float, float] = (1.0, 3.0)
    mae_coefficient: float = 0.5
    temperature_cf: float = 0.005
    smooth_absolute_epsilon_cf: float = 0.0025
    hessian_floor: float = 0.001
    hessian_ceiling: float = 200.0
    sigmoid_clip: float = 40.0

    def validate(self) -> None:
        if self.thresholds_cf != (0.06, 0.08):
            raise ValueError("official thresholds must remain (0.06, 0.08)")
        if self.settlement_weights != (1.0, 3.0):
            raise ValueError("settlement decomposition must remain (1, 3)")
        values = (
            self.mae_coefficient,
            self.temperature_cf,
            self.smooth_absolute_epsilon_cf,
            self.hessian_floor,
            self.hessian_ceiling,
            self.sigmoid_clip,
        )
        if not np.isfinite(values).all() or any(value <= 0.0 for value in values):
            raise ValueError("loss constants must be positive and finite")
        if self.hessian_floor > self.hessian_ceiling:
            raise ValueError("hessian floor exceeds ceiling")


def _one_dimensional(values: Any, *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite one-dimensional array")
    return result


def _sigmoid(values: np.ndarray, clip: float) -> np.ndarray:
    bounded = np.clip(values, -clip, clip)
    return 1.0 / (1.0 + np.exp(-bounded))


def smooth_ficr_loss_grad_hess(
    actual_cf: Any,
    prediction_cf: Any,
    *,
    mean_train_actual_cf: float,
    config: SmoothFICRLossConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-row registered loss, exact gradient and positive surrogate."""

    settings = config or SmoothFICRLossConfig()
    settings.validate()
    actual = _one_dimensional(actual_cf, name="actual_cf")
    prediction = _one_dimensional(prediction_cf, name="prediction_cf")
    if actual.shape != prediction.shape:
        raise ValueError("actual and prediction shapes differ")
    if np.any(actual < 0.10):
        raise ValueError("custom loss accepts only officially eligible actual rows")
    mean_actual = float(mean_train_actual_cf)
    if not np.isfinite(mean_actual) or mean_actual <= 0.0:
        raise ValueError("mean_train_actual_cf must be positive and finite")

    error = prediction - actual
    epsilon = settings.smooth_absolute_epsilon_cf
    temperature = settings.temperature_cf
    radius = np.sqrt(error * error + epsilon * epsilon)
    actual_scale = actual / (8.0 * mean_actual)

    settlement = np.zeros_like(radius)
    radial_slope = np.full_like(radius, settings.mae_coefficient)
    radial_slope_derivative = np.zeros_like(radius)
    for threshold, weight in zip(
        settings.thresholds_cf, settings.settlement_weights
    ):
        probability = _sigmoid(
            (threshold - radius) / temperature, settings.sigmoid_clip
        )
        transition = probability * (1.0 - probability)
        settlement += weight * probability
        radial_slope += actual_scale * weight * transition / temperature
        radial_slope_derivative -= (
            actual_scale
            * weight
            * transition
            * (1.0 - 2.0 * probability)
            / (temperature * temperature)
        )

    loss = settings.mae_coefficient * radius - actual_scale * settlement
    direction = error / radius
    gradient = radial_slope * direction
    exact_second_derivative = (
        radial_slope * epsilon * epsilon / (radius * radius * radius)
        + radial_slope_derivative * direction * direction
    )
    positive_surrogate = np.clip(
        np.abs(exact_second_derivative),
        settings.hessian_floor,
        settings.hessian_ceiling,
    )
    if not (
        np.isfinite(loss).all()
        and np.isfinite(gradient).all()
        and np.isfinite(positive_surrogate).all()
        and np.all(positive_surrogate >= settings.hessian_floor)
    ):
        raise FloatingPointError("smooth FICR objective produced invalid values")
    return loss, gradient, positive_surrogate


@dataclass(frozen=True)
class SmoothFICRObjective:
    """Pickle-safe callable used by LightGBM and XGBoost sklearn APIs."""

    mean_train_actual_cf: float
    prediction_offset_cf: float
    config: SmoothFICRLossConfig = SmoothFICRLossConfig()

    def __call__(
        self, actual_cf: np.ndarray, raw_prediction: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        _, gradient, surrogate = smooth_ficr_loss_grad_hess(
            actual_cf,
            np.asarray(raw_prediction, dtype=float) + self.prediction_offset_cf,
            mean_train_actual_cf=self.mean_train_actual_cf,
            config=self.config,
        )
        return gradient, surrogate


def finite_difference_gradient_audit(
    *,
    config: SmoothFICRLossConfig | None = None,
    step: float = 1e-6,
) -> dict[str, Any]:
    settings = config or SmoothFICRLossConfig()
    errors = np.asarray(
        (-0.081, -0.080, -0.079, -0.061, -0.060, -0.059, 0.0,
         0.059, 0.060, 0.061, 0.079, 0.080, 0.081),
        dtype=float,
    )
    actual = np.resize(np.asarray((0.12, 0.30, 0.55, 0.90), dtype=float), len(errors))
    prediction = actual + errors
    mean_actual = float(np.mean(actual))
    _, analytic, surrogate = smooth_ficr_loss_grad_hess(
        actual,
        prediction,
        mean_train_actual_cf=mean_actual,
        config=settings,
    )
    plus = smooth_ficr_loss_grad_hess(
        actual,
        prediction + float(step),
        mean_train_actual_cf=mean_actual,
        config=settings,
    )[0]
    minus = smooth_ficr_loss_grad_hess(
        actual,
        prediction - float(step),
        mean_train_actual_cf=mean_actual,
        config=settings,
    )[0]
    numeric = (plus - minus) / (2.0 * float(step))
    difference = np.abs(analytic - numeric)
    return {
        "points": len(errors),
        "step": float(step),
        "maximum_absolute_error": float(np.max(difference)),
        "mean_absolute_error": float(np.mean(difference)),
        "gradient_finite": bool(np.isfinite(analytic).all()),
        "surrogate_hessian_finite": bool(np.isfinite(surrogate).all()),
        "surrogate_hessian_minimum": float(np.min(surrogate)),
        "surrogate_hessian_maximum": float(np.max(surrogate)),
        "surrogate_hessian_strictly_positive": bool(np.all(surrogate > 0.0)),
        "tested_errors_cf": errors.tolist(),
    }


class SmoothFICRRegressor:
    """Strict-fit direct weather regressor with one registered backend."""

    BACKENDS = ("lgb_smooth_ficr", "xgb_smooth_ficr")

    def __init__(
        self,
        *,
        backend: str,
        common_parameters: Mapping[str, Any],
        backend_parameters: Mapping[str, Any],
        random_state: int,
        n_jobs: int,
        loss_config: SmoothFICRLossConfig | None = None,
    ) -> None:
        if backend not in self.BACKENDS:
            raise ValueError(f"unsupported backend: {backend}")
        self.backend = backend
        self.common_parameters = dict(common_parameters)
        self.backend_parameters = dict(backend_parameters)
        self.random_state = int(random_state)
        self.n_jobs = int(n_jobs)
        self.loss_config = loss_config or SmoothFICRLossConfig()
        self.loss_config.validate()

    def _make_model(self, objective: SmoothFICRObjective) -> Any:
        common = dict(self.common_parameters)
        if self.backend == "lgb_smooth_ficr":
            from lightgbm import LGBMRegressor

            return LGBMRegressor(
                **common,
                **self.backend_parameters,
                objective=objective,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                device_type="cpu",
            )
        from xgboost import XGBRegressor

        return XGBRegressor(
            **common,
            **self.backend_parameters,
            objective=objective,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            device="cpu",
        )

    def fit(
        self,
        features: pd.DataFrame,
        actual_kwh: pd.Series,
        *,
        capacity_kwh: float,
    ) -> "SmoothFICRRegressor":
        if not isinstance(features, pd.DataFrame) or features.empty:
            raise TypeError("features must be a non-empty DataFrame")
        if not features.index.is_unique or not features.index.is_monotonic_increasing:
            raise ValueError("fit feature index must be unique and increasing")
        if not isinstance(actual_kwh, pd.Series) or not features.index.equals(
            actual_kwh.index
        ):
            raise ValueError("actual index differs from features")
        capacity = float(capacity_kwh)
        if not np.isfinite(capacity) or capacity <= 0.0:
            raise ValueError("capacity_kwh must be positive and finite")
        actual_cf = actual_kwh.to_numpy(dtype=float) / capacity
        eligible = np.isfinite(actual_cf) & (actual_cf >= 0.10)
        if int(eligible.sum()) < 300:
            raise ValueError(f"too few eligible training rows: {int(eligible.sum())}")
        eligible_features = features.loc[eligible]
        target = actual_cf[eligible]
        self.transform_ = TrainOnlyMedianTransform().fit(eligible_features)
        transformed = self.transform_.transform(eligible_features)
        self.offset_cf_ = float(np.median(target))
        self.mean_train_actual_cf_ = float(np.mean(target))
        objective = SmoothFICRObjective(
            mean_train_actual_cf=self.mean_train_actual_cf_,
            prediction_offset_cf=self.offset_cf_,
            config=self.loss_config,
        )
        self.model_ = self._make_model(objective)
        self.model_.fit(transformed, target)
        self.capacity_kwh_ = capacity
        self.fit_index_ = features.index.copy()
        self.feature_columns_ = tuple(features.columns)
        self.fit_rows_ = len(features)
        self.eligible_fit_rows_ = int(eligible.sum())
        return self

    def predict_cf(self, features: pd.DataFrame) -> pd.Series:
        if not hasattr(self, "model_"):
            raise RuntimeError("fit must be called before predict")
        if not isinstance(features, pd.DataFrame) or tuple(
            features.columns
        ) != self.feature_columns_:
            raise ValueError("application feature columns/order differ from fit")
        if not features.index.is_unique or not features.index.is_monotonic_increasing:
            raise ValueError("application index must be unique and increasing")
        if len(features.index.intersection(self.fit_index_)):
            raise ValueError("application rows overlap fit rows")
        transformed = self.transform_.transform(features)
        raw = np.asarray(self.model_.predict(transformed), dtype=float)
        prediction = np.clip(raw + self.offset_cf_, 0.0, 1.02)
        if prediction.shape != (len(features),) or not np.isfinite(prediction).all():
            raise FloatingPointError("model prediction is invalid")
        return pd.Series(prediction, index=features.index, name="prediction_cf")

    def metadata(self) -> dict[str, Any]:
        if not hasattr(self, "model_"):
            raise RuntimeError("fit must be called before metadata")
        return {
            "backend": self.backend,
            "common_parameters": self.common_parameters,
            "backend_parameters": self.backend_parameters,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "loss_config": asdict(self.loss_config),
            "fit_rows": self.fit_rows_,
            "eligible_fit_rows": self.eligible_fit_rows_,
            "features": len(self.feature_columns_),
            "offset_cf": self.offset_cf_,
            "mean_train_actual_cf": self.mean_train_actual_cf_,
            "transform_fit_rows": self.transform_.fit_rows_,
            "transform_fit_on_eligible_rows_only": True,
            "fit_score_calculated": False,
            "true_hessian_used": False,
            "positive_surrogate_hessian_used": True,
        }


__all__ = (
    "SmoothFICRLossConfig",
    "SmoothFICRObjective",
    "SmoothFICRRegressor",
    "finite_difference_gradient_audit",
    "smooth_ficr_loss_grad_hess",
)

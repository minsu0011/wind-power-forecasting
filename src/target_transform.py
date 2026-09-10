"""Leakage-safe nonlinear capacity-factor target transforms.

The weather matrix is unchanged.  Only eligible fold-training targets are
transformed before fitting the locked LightGBM L1 regressor, and predictions
are inverted and clipped back to capacity factor before scoring or blending.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping

from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd

from src.run_sequence_residual import FORBIDDEN_FEATURE_TOKENS


TRANSFORM_IDS = ("identity", "logit_eps02", "arcsin_sqrt", "cube_root")


def _as_finite_array(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def forward_target_transform(
    capacity_factor: Any, transform_id: str, *, epsilon: float = 0.02
) -> np.ndarray:
    """Transform finite eligible capacity-factor targets."""

    values = _as_finite_array(capacity_factor, name="capacity factor")
    if transform_id == "identity":
        output = values.copy()
    elif transform_id == "logit_eps02":
        if epsilon != 0.02:
            raise ValueError("logit_eps02 requires the preregistered epsilon=0.02")
        probability = np.clip(values, epsilon, 1.0 - epsilon)
        output = np.log(probability / (1.0 - probability))
    elif transform_id == "arcsin_sqrt":
        output = np.arcsin(np.sqrt(np.clip(values, 0.0, 1.0)))
    elif transform_id == "cube_root":
        output = np.cbrt(np.clip(values, 0.0, 1.02))
    else:
        raise ValueError(f"unknown target transform: {transform_id}")
    if not np.isfinite(output).all():
        raise ValueError(f"{transform_id} produced non-finite training targets")
    return output


def _stable_sigmoid(values: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float64)
    positive = values >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def inverse_target_transform(
    transformed: Any, transform_id: str, *, epsilon: float = 0.02
) -> np.ndarray:
    """Invert transformed predictions and return finite CF in ``[0, 1.02]``."""

    values = _as_finite_array(transformed, name="transformed prediction")
    if transform_id == "identity":
        output = values
    elif transform_id == "logit_eps02":
        if epsilon != 0.02:
            raise ValueError("logit_eps02 requires the preregistered epsilon=0.02")
        output = _stable_sigmoid(values)
    elif transform_id == "arcsin_sqrt":
        angle = np.clip(values, 0.0, np.pi / 2.0)
        output = np.square(np.sin(angle))
    elif transform_id == "cube_root":
        output = np.power(np.maximum(values, 0.0), 3.0)
    else:
        raise ValueError(f"unknown target transform: {transform_id}")
    output = np.clip(output, 0.0, 1.02)
    if not np.isfinite(output).all():
        raise ValueError(f"{transform_id} inverse produced non-finite capacity factor")
    return output


def assert_strict_fit_apply_order(
    fit_index: pd.DatetimeIndex, apply_index: pd.DatetimeIndex
) -> None:
    if not isinstance(fit_index, pd.DatetimeIndex) or not isinstance(
        apply_index, pd.DatetimeIndex
    ):
        raise TypeError("fit/apply indices must be DatetimeIndex")
    if len(fit_index) == 0 or len(apply_index) == 0:
        raise ValueError("fit/apply indices must be non-empty")
    if fit_index.intersection(apply_index).size:
        raise ValueError("fit and apply indices overlap")
    if fit_index.max() >= apply_index.min():
        raise ValueError("fit.max must be strictly earlier than apply.min")


def _validate_features(features: pd.DataFrame) -> None:
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a DataFrame")
    if not isinstance(features.index, pd.DatetimeIndex):
        raise TypeError("feature index must be a DatetimeIndex")
    if not features.index.is_unique or not features.index.is_monotonic_increasing:
        raise ValueError("feature index must be unique and increasing")
    if not features.columns.is_unique:
        raise ValueError("feature columns must be unique")
    forbidden = [
        str(column)
        for column in features.columns
        if any(token in str(column).lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if forbidden:
        raise ValueError(f"forbidden target/SCADA/group features: {forbidden[:5]}")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes):
        raise TypeError("all target-transform features must be numeric")
    if not np.isfinite(features.to_numpy(dtype=np.float32, copy=False)).all():
        raise ValueError("target-transform weather contains non-finite values")


def _array_sha256(values: Any) -> str:
    array = np.asarray(values, dtype="<f8", order="C")
    return hashlib.sha256(array.tobytes()).hexdigest()


@dataclass
class TargetTransformRegressor:
    """Locked LightGBM L1 regressor with one preregistered target transform."""

    transform_id: str
    model_params: Mapping[str, Any]
    epsilon: float = 0.02

    model_: LGBMRegressor | None = None
    feature_names_: tuple[str, ...] | None = None
    fit_audit_: dict[str, Any] | None = None

    def fit(
        self, features: pd.DataFrame, target_cf: pd.Series
    ) -> "TargetTransformRegressor":
        _validate_features(features)
        if self.transform_id not in TRANSFORM_IDS:
            raise ValueError(f"unsupported transform {self.transform_id}")
        if not isinstance(target_cf, pd.Series) or not features.index.equals(
            target_cf.index
        ):
            raise ValueError("features and target capacity factor indices differ")
        target = target_cf.to_numpy(dtype=np.float64, copy=False)
        eligible = np.isfinite(target) & (target >= 0.1)
        if not eligible.any():
            raise ValueError("no eligible capacity-factor training targets")
        transformed = forward_target_transform(
            target[eligible], self.transform_id, epsilon=self.epsilon
        )
        params = dict(self.model_params)
        required = {
            "objective": "regression_l1",
            "random_state": 42,
            "n_jobs": 7,
        }
        for name, expected in required.items():
            if params.get(name) != expected:
                raise ValueError(f"locked model parameter {name} changed")
        model = LGBMRegressor(**params)
        model.fit(features.iloc[np.flatnonzero(eligible)], transformed)
        self.model_ = model
        self.feature_names_ = tuple(map(str, features.columns))
        self.fit_audit_ = {
            "transform_id": self.transform_id,
            "fit_rows_total": int(len(features)),
            "fit_rows_eligible": int(eligible.sum()),
            "fit_start": features.index.min().isoformat(),
            "fit_end": features.index.max().isoformat(),
            "feature_count": int(features.shape[1]),
            "actual_or_scada_feature_count": 0,
            "target_cf_min": float(np.min(target[eligible])),
            "target_cf_max": float(np.max(target[eligible])),
            "transformed_min": float(np.min(transformed)),
            "transformed_max": float(np.max(transformed)),
            "eligible_target_sha256": _array_sha256(target[eligible]),
            "transformed_target_sha256": _array_sha256(transformed),
        }
        return self

    def predict_cf(self, features: pd.DataFrame) -> tuple[pd.Series, dict[str, Any]]:
        if self.model_ is None or self.feature_names_ is None:
            raise RuntimeError("target-transform regressor is not fitted")
        _validate_features(features)
        if tuple(map(str, features.columns)) != self.feature_names_:
            raise ValueError("prediction feature schema differs from fit schema")
        transformed = np.asarray(self.model_.predict(features), dtype=np.float64)
        if transformed.shape != (len(features),) or not np.isfinite(transformed).all():
            raise ValueError("LightGBM produced invalid transformed predictions")
        capacity_factor = inverse_target_transform(
            transformed, self.transform_id, epsilon=self.epsilon
        )
        output = pd.Series(capacity_factor, index=features.index, name="prediction_cf")
        return output, {
            "apply_rows": int(len(features)),
            "apply_start": features.index.min().isoformat(),
            "apply_end": features.index.max().isoformat(),
            "transformed_prediction_min": float(transformed.min()),
            "transformed_prediction_max": float(transformed.max()),
            "inverse_prediction_min_cf": float(capacity_factor.min()),
            "inverse_prediction_max_cf": float(capacity_factor.max()),
            "inverse_finite": True,
            "inverse_clipped_0_1p02": True,
            "prediction_cf_sha256": _array_sha256(capacity_factor),
            "query_actual_values_accessed": False,
        }


def blend_with_corrected_v3(
    transformed_prediction_cf: pd.Series,
    corrected_v3_kwh: pd.Series,
    *,
    capacity_kwh: float,
    transformed_weight: float,
) -> pd.Series:
    if not transformed_prediction_cf.index.equals(corrected_v3_kwh.index):
        raise ValueError("transformed prediction and corrected-v3 indices differ")
    weight = float(transformed_weight)
    if weight not in {0.05, 0.10, 0.20}:
        raise ValueError("blend weight is not preregistered")
    transformed = transformed_prediction_cf.to_numpy(dtype=np.float64, copy=False)
    baseline = corrected_v3_kwh.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(transformed).all() or not np.isfinite(baseline).all():
        raise ValueError("blend inputs contain non-finite values")
    prediction = np.clip(
        (1.0 - weight) * baseline + weight * transformed * capacity_kwh,
        0.0,
        1.02 * capacity_kwh,
    )
    return pd.Series(
        prediction, index=corrected_v3_kwh.index, name=corrected_v3_kwh.name
    )


__all__ = [
    "TRANSFORM_IDS",
    "TargetTransformRegressor",
    "assert_strict_fit_apply_order",
    "blend_with_corrected_v3",
    "forward_target_transform",
    "inverse_target_transform",
]

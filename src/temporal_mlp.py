"""Leakage-safe joint 24-hour MLP for one already-issued forecast run.

The model consumes six small, preregistered run trajectories, projects each
trajectory onto a fixed low-frequency DCT basis, and predicts all 24 target
capacity factors jointly.  It intentionally differs from the scalar
run-sequence residual model: there is one training sample per issued run and
the target is a 24-dimensional direct CF vector.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

from src.run_sequence_residual import (
    FORBIDDEN_FEATURE_TOKENS,
    validate_complete_runs,
)
from src.temporal import operating_run_key


def orthonormal_dct2_basis(hours: int = 24, coefficients: int = 8) -> np.ndarray:
    """Return the first ``coefficients`` rows of an orthonormal DCT-II basis."""

    if not isinstance(hours, int) or hours <= 1:
        raise ValueError("hours must be an integer greater than one")
    if not isinstance(coefficients, int) or not 1 <= coefficients <= hours:
        raise ValueError("coefficients must be in [1, hours]")
    position = np.arange(hours, dtype=np.float64) + 0.5
    frequency = np.arange(coefficients, dtype=np.float64)[:, None]
    basis = np.cos(np.pi * frequency * position[None, :] / float(hours))
    basis[0] *= np.sqrt(1.0 / float(hours))
    basis[1:] *= np.sqrt(2.0 / float(hours))
    gram = basis @ basis.T
    if not np.allclose(gram, np.eye(coefficients), atol=1e-12, rtol=0.0):
        raise AssertionError("DCT-II basis is not orthonormal")
    return basis


def _validate_inputs(
    weather: pd.DataFrame,
    baseline_kwh: pd.Series,
    *,
    capacity_kwh: float,
    weather_columns: Sequence[str],
    expected_hours: int,
) -> None:
    if not isinstance(weather, pd.DataFrame):
        raise TypeError("weather must be a DataFrame")
    if not isinstance(baseline_kwh, pd.Series):
        raise TypeError("baseline_kwh must be a Series")
    if not weather.index.equals(baseline_kwh.index):
        raise ValueError("weather and baseline indices differ")
    if not np.isfinite(capacity_kwh) or capacity_kwh <= 0:
        raise ValueError("capacity_kwh must be positive and finite")
    validate_complete_runs(weather.index, expected_hours=expected_hours)
    registered = tuple(map(str, weather_columns))
    if len(registered) != len(set(registered)):
        raise ValueError("registered weather columns must be unique")
    missing = set(registered).difference(map(str, weather.columns))
    if missing:
        raise KeyError(f"missing registered weather columns: {sorted(missing)}")
    forbidden = [
        str(column)
        for column in weather.columns
        if any(token in str(column).lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if forbidden:
        raise ValueError(f"forbidden actual/target/SCADA feature columns: {forbidden[:5]}")
    arrays = [
        weather.loc[:, list(registered)].to_numpy(dtype=np.float64, copy=False),
        baseline_kwh.to_numpy(dtype=np.float64, copy=False),
    ]
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("run inputs contain non-finite values")


def build_run_channel_tensor(
    weather: pd.DataFrame,
    baseline_kwh: pd.Series,
    *,
    capacity_kwh: float,
    weather_columns: Sequence[str],
    expected_hours: int = 24,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Create ``run x channel x hour`` tensor in the preregistered order."""

    _validate_inputs(
        weather,
        baseline_kwh,
        capacity_kwh=capacity_kwh,
        weather_columns=weather_columns,
        expected_hours=expected_hours,
    )
    run_count = len(weather) // expected_hours
    channels = [
        (baseline_kwh.to_numpy(dtype=np.float64, copy=False) / capacity_kwh).reshape(
            run_count, expected_hours
        )
    ]
    values = weather.loc[:, list(weather_columns)].to_numpy(
        dtype=np.float64, copy=False
    )
    for position in range(values.shape[1]):
        channels.append(values[:, position].reshape(run_count, expected_hours))
    tensor = np.stack(channels, axis=1)
    if tensor.shape != (run_count, len(weather_columns) + 1, expected_hours):
        raise AssertionError("unexpected run tensor shape")
    keys = pd.DatetimeIndex(
        operating_run_key(weather.index)[::expected_hours], name="forecast_run_kst_date"
    )
    if not keys.is_unique or not keys.is_monotonic_increasing:
        raise AssertionError("run keys must be unique and increasing")
    return tensor, keys


@dataclass
class RunDCTTransformer:
    """Fold-fit channel scaler followed by a fixed low-frequency DCT-II."""

    weather_columns: tuple[str, ...]
    coefficients: int = 8
    expected_hours: int = 24
    scale_floor: float = 1e-6
    channel_mean_: np.ndarray | None = None
    channel_scale_: np.ndarray | None = None

    def fit(
        self,
        weather: pd.DataFrame,
        baseline_kwh: pd.Series,
        *,
        capacity_kwh: float,
    ) -> "RunDCTTransformer":
        tensor, _ = build_run_channel_tensor(
            weather,
            baseline_kwh,
            capacity_kwh=capacity_kwh,
            weather_columns=self.weather_columns,
            expected_hours=self.expected_hours,
        )
        mean = tensor.mean(axis=(0, 2))
        scale = tensor.std(axis=(0, 2), ddof=0)
        if not np.isfinite(mean).all() or not np.isfinite(scale).all():
            raise ValueError("non-finite channel scaler")
        self.channel_mean_ = mean
        self.channel_scale_ = np.maximum(scale, float(self.scale_floor))
        return self

    def transform(
        self,
        weather: pd.DataFrame,
        baseline_kwh: pd.Series,
        *,
        capacity_kwh: float,
    ) -> tuple[np.ndarray, pd.DatetimeIndex]:
        if self.channel_mean_ is None or self.channel_scale_ is None:
            raise RuntimeError("RunDCTTransformer must be fitted first")
        tensor, keys = build_run_channel_tensor(
            weather,
            baseline_kwh,
            capacity_kwh=capacity_kwh,
            weather_columns=self.weather_columns,
            expected_hours=self.expected_hours,
        )
        if tensor.shape[1] != len(self.channel_mean_):
            raise AssertionError("channel count differs from fitted scaler")
        standardized = (
            tensor - self.channel_mean_[None, :, None]
        ) / self.channel_scale_[None, :, None]
        basis = orthonormal_dct2_basis(self.expected_hours, self.coefficients)
        coefficients = np.einsum("rch,kh->rck", standardized, basis, optimize=True)
        matrix = coefficients.reshape(len(tensor), -1)
        expected = (len(self.weather_columns) + 1) * self.coefficients
        if matrix.shape[1] != expected:
            raise AssertionError("unexpected DCT input dimension")
        if not np.isfinite(matrix).all():
            raise AssertionError("DCT input contains non-finite values")
        return matrix, keys

    def fit_transform(
        self,
        weather: pd.DataFrame,
        baseline_kwh: pd.Series,
        *,
        capacity_kwh: float,
    ) -> tuple[np.ndarray, pd.DatetimeIndex]:
        return self.fit(
            weather, baseline_kwh, capacity_kwh=capacity_kwh
        ).transform(weather, baseline_kwh, capacity_kwh=capacity_kwh)


def build_joint_cf_target(
    actual_kwh: pd.Series,
    *,
    capacity_kwh: float,
    expected_hours: int = 24,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Create a finite ``run x 24`` direct capacity-factor target matrix."""

    if not isinstance(actual_kwh, pd.Series):
        raise TypeError("actual_kwh must be a Series")
    if not np.isfinite(capacity_kwh) or capacity_kwh <= 0:
        raise ValueError("capacity_kwh must be positive and finite")
    validate_complete_runs(actual_kwh.index, expected_hours=expected_hours)
    values = actual_kwh.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("joint target requires complete finite 24-hour runs")
    target = (values / capacity_kwh).reshape(-1, expected_hours)
    keys = pd.DatetimeIndex(
        operating_run_key(actual_kwh.index)[::expected_hours],
        name="forecast_run_kst_date",
    )
    return target, keys


def fit_seed_ensemble(
    specification: Mapping[str, Any], x: np.ndarray, y: np.ndarray
) -> list[MLPRegressor]:
    """Fit the preregistered deterministic seed ensemble."""

    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if x_array.ndim != 2 or y_array.ndim != 2 or len(x_array) != len(y_array):
        raise ValueError("MLP inputs must be aligned two-dimensional arrays")
    if y_array.shape[1] != 24:
        raise ValueError("joint MLP target must contain 24 horizons")
    if not np.isfinite(x_array).all() or not np.isfinite(y_array).all():
        raise ValueError("MLP fit arrays contain non-finite values")
    hidden = tuple(int(value) for value in specification["hidden_layer_sizes"])
    models: list[MLPRegressor] = []
    for seed in specification["seeds"]:
        model = MLPRegressor(
            hidden_layer_sizes=hidden,
            activation=str(specification["activation"]),
            solver=str(specification["solver"]),
            alpha=float(specification["alpha"]),
            random_state=int(seed),
            max_iter=int(specification["max_iter"]),
            max_fun=int(specification["max_fun"]),
            tol=float(specification["tol"]),
            shuffle=bool(specification["shuffle"]),
            early_stopping=bool(specification["early_stopping"]),
        )
        model.fit(x_array, y_array)
        models.append(model)
    return models


def predict_seed_ensemble(models: Sequence[MLPRegressor], x: np.ndarray) -> np.ndarray:
    """Average seed predictions and clip only the direct CF network output."""

    if not models:
        raise ValueError("seed ensemble must not be empty")
    x_array = np.asarray(x, dtype=np.float64)
    predictions = [np.asarray(model.predict(x_array), dtype=np.float64) for model in models]
    if any(prediction.shape != (len(x_array), 24) for prediction in predictions):
        raise AssertionError("MLP prediction is not run x 24")
    output = np.mean(predictions, axis=0)
    if not np.isfinite(output).all():
        raise ValueError("MLP prediction contains non-finite values")
    return np.clip(output, 0.0, 1.02)


def blend_joint_prediction(
    baseline_kwh: pd.Series,
    network_cf: np.ndarray,
    *,
    capacity_kwh: float,
    blend_weight: float,
    expected_hours: int = 24,
) -> pd.Series:
    """Blend a joint neural CF trajectory with the fixed baseline."""

    if not 0.0 < float(blend_weight) <= 1.0:
        raise ValueError("blend_weight must be in (0, 1]")
    expected_shape = (len(baseline_kwh) // expected_hours, expected_hours)
    neural = np.asarray(network_cf, dtype=np.float64)
    if neural.shape != expected_shape or not np.isfinite(neural).all():
        raise ValueError("network CF has unexpected shape or non-finite values")
    baseline = baseline_kwh.to_numpy(dtype=np.float64, copy=False) / capacity_kwh
    if not np.isfinite(baseline).all():
        raise ValueError("baseline contains non-finite values")
    candidate_cf = np.clip(
        (1.0 - float(blend_weight)) * baseline.reshape(expected_shape)
        + float(blend_weight) * neural,
        0.0,
        1.02,
    )
    return pd.Series(
        candidate_cf.reshape(-1) * capacity_kwh,
        index=baseline_kwh.index,
        name=baseline_kwh.name,
    )


def choose_group_candidates(
    comparisons: Mapping[str, Mapping[str, Mapping[str, Mapping[str, float]]]],
    groups: Sequence[str],
) -> dict[str, str | None]:
    """Choose at most one all-slices-positive recipe independently per group."""

    selected: dict[str, str | None] = {}
    for group in groups:
        eligible: list[tuple[float, float, str]] = []
        for candidate_id, by_group in comparisons.items():
            slices = by_group.get(str(group), {})
            if not slices:
                raise ValueError(f"candidate {candidate_id} has no slices for {group}")
            deltas = [float(result["delta"]) for result in slices.values()]
            if not np.isfinite(deltas).all():
                raise ValueError("candidate comparison contains non-finite delta")
            if all(delta > 0.0 for delta in deltas):
                eligible.append((min(deltas), float(np.mean(deltas)), str(candidate_id)))
        eligible.sort(key=lambda item: (-item[0], -item[1], item[2]))
        selected[str(group)] = eligible[0][2] if eligible else None
    return selected


__all__ = [
    "RunDCTTransformer",
    "blend_joint_prediction",
    "build_joint_cf_target",
    "build_run_channel_tensor",
    "choose_group_candidates",
    "fit_seed_ensemble",
    "orthonormal_dct2_basis",
    "predict_seed_ensemble",
]

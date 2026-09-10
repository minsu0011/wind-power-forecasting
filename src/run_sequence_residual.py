"""Compact, leakage-safe features and models for one issued 24-hour run.

Every operating hour from 01:00 through the following 00:00 is issued
together.  This module may summarize or shift values inside that run, but it
never joins adjacent issuance runs and never accepts actual/target/SCADA
columns as predictors.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.temporal import operating_run_key


FORBIDDEN_FEATURE_TOKENS = (
    "actual",
    "target",
    "label",
    "scada",
    "power_kw",
    "kpx_group_",
)


def validate_complete_runs(
    index: pd.DatetimeIndex, *, expected_hours: int = 24
) -> pd.DatetimeIndex:
    """Validate contiguous, complete operating runs and return their keys."""

    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("index must be a DatetimeIndex")
    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("index must be unique and increasing")
    if len(index) == 0:
        raise ValueError("index must not be empty")
    key = operating_run_key(index)
    positions = pd.Series(np.arange(len(index)), index=index).groupby(key, sort=True)
    for _, group in positions:
        loc = group.to_numpy(dtype=int)
        if len(loc) != expected_hours:
            raise ValueError(
                f"forecast run has {len(loc)} rows; expected {expected_hours}"
            )
        run_index = index[loc]
        expected = pd.date_range(run_index[0], periods=expected_hours, freq="h")
        if not run_index.equals(expected):
            raise ValueError("forecast run is not contiguous hourly data")
        if run_index[0].hour != 1 or run_index[-1].hour != 0:
            raise ValueError("forecast run must cover operating hours 01:00..00:00")
    return key


def _matrix_feature(
    output: dict[str, np.ndarray], name: str, values: np.ndarray
) -> None:
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"non-finite run feature {name}")
    output[name] = array.reshape(-1)


def _sequence_features(
    name: str, values: np.ndarray, *, expected_hours: int
) -> dict[str, np.ndarray]:
    """Return registered per-horizon summaries for an n_run x 24 matrix."""

    if values.ndim != 2 or values.shape[1] != expected_hours:
        raise ValueError("sequence matrix must be n_run x expected_hours")
    if not np.isfinite(values).all():
        raise ValueError(f"sequence {name} contains non-finite values")
    run_count = values.shape[0]
    mean = values.mean(axis=1)
    std = values.std(axis=1, ddof=0)
    minimum = values.min(axis=1)
    maximum = values.max(axis=1)
    quantiles = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90], axis=1)
    centered = values - mean[:, None]
    safe_std = np.maximum(std, 1e-6)
    previous = np.concatenate([values[:, :1], values[:, :-1]], axis=1)
    following = np.concatenate([values[:, 1:], values[:, -1:]], axis=1)
    order = np.argsort(values, axis=1, kind="stable")
    rank = np.empty_like(order, dtype=np.float64)
    row = np.arange(run_count)[:, None]
    rank[row, order] = np.arange(expected_hours, dtype=np.float64)[None, :]
    rank /= float(expected_hours - 1)
    horizon = np.arange(expected_hours, dtype=np.float64)
    horizon_centered = horizon - horizon.mean()
    slope = (values @ horizon_centered) / np.square(horizon_centered).sum()
    transform = np.fft.rfft(values, axis=1) / float(expected_hours)

    repeated = lambda x: np.repeat(np.asarray(x)[:, None], expected_hours, axis=1)
    features: dict[str, np.ndarray] = {
        f"{name}__current": values,
        f"{name}__run_mean": repeated(mean),
        f"{name}__run_std": repeated(std),
        f"{name}__run_min": repeated(minimum),
        f"{name}__run_max": repeated(maximum),
        f"{name}__run_range": repeated(maximum - minimum),
        f"{name}__run_q10": repeated(quantiles[0]),
        f"{name}__run_q25": repeated(quantiles[1]),
        f"{name}__run_q50": repeated(quantiles[2]),
        f"{name}__run_q75": repeated(quantiles[3]),
        f"{name}__run_q90": repeated(quantiles[4]),
        f"{name}__current_minus_mean": centered,
        f"{name}__current_zscore": centered / safe_std[:, None],
        f"{name}__within_run_rank": rank,
        f"{name}__same_run_previous": previous,
        f"{name}__same_run_next": following,
        f"{name}__same_run_gradient": 0.5 * (following - previous),
        f"{name}__same_run_curvature": following - 2.0 * values + previous,
        f"{name}__run_peak_position": repeated(
            np.argmax(values, axis=1) / float(expected_hours - 1)
        ),
        f"{name}__run_trough_position": repeated(
            np.argmin(values, axis=1) / float(expected_hours - 1)
        ),
        f"{name}__run_linear_slope": repeated(slope),
        f"{name}__run_fft1_real": repeated(transform[:, 1].real),
        f"{name}__run_fft1_imag": repeated(transform[:, 1].imag),
        f"{name}__run_fft2_real": repeated(transform[:, 2].real),
        f"{name}__run_fft2_imag": repeated(transform[:, 2].imag),
    }
    return features


def build_run_sequence_features(
    weather: pd.DataFrame,
    baseline_kwh: pd.Series,
    *,
    capacity_kwh: float,
    weather_columns: Sequence[str],
    expected_hours: int = 24,
    expected_max_features: int = 300,
) -> pd.DataFrame:
    """Build compact summaries using only values in each issued run."""

    if not isinstance(weather, pd.DataFrame):
        raise TypeError("weather must be a DataFrame")
    if not isinstance(baseline_kwh, pd.Series):
        raise TypeError("baseline_kwh must be a Series")
    if not weather.index.equals(baseline_kwh.index):
        raise ValueError("weather and baseline indices differ")
    if not np.isfinite(capacity_kwh) or capacity_kwh <= 0:
        raise ValueError("capacity_kwh must be positive and finite")
    validate_complete_runs(weather.index, expected_hours=expected_hours)
    forbidden = [
        str(column)
        for column in weather.columns
        if any(token in str(column).lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if forbidden:
        raise ValueError(f"forbidden actual/target/SCADA feature columns: {forbidden[:5]}")
    missing = [column for column in weather_columns if column not in weather.columns]
    if missing:
        raise KeyError(f"missing registered weather columns: {missing}")
    if len(set(weather_columns)) != len(tuple(weather_columns)):
        raise ValueError("registered weather columns must be unique")
    baseline = baseline_kwh.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(baseline).all():
        raise ValueError("baseline contains non-finite values")
    selected = weather.loc[:, list(weather_columns)].to_numpy(
        dtype=np.float64, copy=False
    )
    if not np.isfinite(selected).all():
        raise ValueError("registered weather values contain non-finite values")
    if len(weather) % expected_hours:
        raise ValueError("row count is not a whole number of runs")
    run_count = len(weather) // expected_hours
    series_values: dict[str, np.ndarray] = {
        "baseline_cf": (baseline / capacity_kwh).reshape(run_count, expected_hours)
    }
    for position, column in enumerate(weather_columns):
        series_values[str(column)] = selected[:, position].reshape(
            run_count, expected_hours
        )

    flat: dict[str, np.ndarray] = {}
    registered: dict[str, dict[str, np.ndarray]] = {}
    for name, matrix in series_values.items():
        features = _sequence_features(name, matrix, expected_hours=expected_hours)
        registered[name] = features
        for feature_name, value in features.items():
            _matrix_feature(flat, feature_name, value)

    horizon = np.tile(np.arange(expected_hours, dtype=np.float64), run_count)
    angle = 2.0 * np.pi * horizon / float(expected_hours)
    flat["horizon__position_fraction"] = horizon / float(expected_hours - 1)
    flat["horizon__sin_1"] = np.sin(angle)
    flat["horizon__cos_1"] = np.cos(angle)
    flat["horizon__sin_2"] = np.sin(2.0 * angle)
    flat["horizon__cos_2"] = np.cos(2.0 * angle)

    baseline_center = registered["baseline_cf"][
        "baseline_cf__current_minus_mean"
    ].reshape(-1)
    baseline_current = registered["baseline_cf"]["baseline_cf__current"].reshape(-1)
    baseline_rank = registered["baseline_cf"][
        "baseline_cf__within_run_rank"
    ].reshape(-1)
    wind_name = "cross__hub_ws_mean"
    if wind_name not in registered:
        raise KeyError("registered weather columns must include cross__hub_ws_mean")
    wind_current = registered[wind_name][f"{wind_name}__current"].reshape(-1)
    wind_center = registered[wind_name][
        f"{wind_name}__current_minus_mean"
    ].reshape(-1)
    wind_rank = registered[wind_name][f"{wind_name}__within_run_rank"].reshape(-1)
    flat["interaction__baseline_center_x_horizon_sin1"] = (
        baseline_center * flat["horizon__sin_1"]
    )
    flat["interaction__baseline_center_x_horizon_cos1"] = (
        baseline_center * flat["horizon__cos_1"]
    )
    flat["interaction__baseline_cf_x_cross_hub_ws"] = (
        baseline_current * wind_current
    )
    flat["interaction__baseline_center_x_cross_hub_ws_center"] = (
        baseline_center * wind_center
    )
    flat["interaction__baseline_rank_x_cross_hub_ws_rank"] = (
        baseline_rank * wind_rank
    )

    output = pd.DataFrame(flat, index=weather.index, dtype=np.float32)
    if not output.columns.is_unique:
        raise AssertionError("run feature names are not unique")
    if output.shape[1] > expected_max_features:
        raise AssertionError(
            f"run feature count {output.shape[1]} exceeds {expected_max_features}"
        )
    if not np.isfinite(output.to_numpy(dtype=np.float32, copy=False)).all():
        raise AssertionError("run features contain non-finite values")
    output.index.name = "forecast_kst_dtm"
    return output


def assert_strict_fit_predict_order(
    fit_index: pd.DatetimeIndex, predict_index: pd.DatetimeIndex
) -> None:
    """Reject any residual fit that overlaps or follows its prediction period."""

    if not isinstance(fit_index, pd.DatetimeIndex) or not isinstance(
        predict_index, pd.DatetimeIndex
    ):
        raise TypeError("fit and predict indices must be DatetimeIndex")
    if len(fit_index) == 0 or len(predict_index) == 0:
        raise ValueError("fit and predict indices must not be empty")
    if fit_index.intersection(predict_index).size:
        raise ValueError("residual fit and prediction intervals overlap")
    if fit_index.max() >= predict_index.min():
        raise ValueError("residual fit must be strictly earlier than prediction")


def fit_residual_model(
    specification: Mapping[str, Any],
    features: pd.DataFrame,
    residual_cf: pd.Series,
) -> Any:
    """Fit one preregistered compact residual model."""

    if not features.index.equals(residual_cf.index):
        raise ValueError("residual feature and target indices differ")
    x = features.to_numpy(dtype=np.float64, copy=False)
    y = residual_cf.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("residual fit data contain non-finite values")
    kind = str(specification["kind"])
    params = dict(specification["params"])
    if kind == "ridge":
        estimator: Any = Ridge(**params)
    elif kind == "huber":
        estimator = HuberRegressor(**params)
    elif kind == "lightgbm_l1":
        estimator = LGBMRegressor(**params)
    else:
        raise ValueError(f"unknown residual model kind: {kind}")
    if bool(specification["standardize"]):
        estimator = Pipeline(
            [("scale", StandardScaler()), ("model", estimator)]
        )
    estimator.fit(features, y)
    return estimator


def apply_residual_model(
    model: Any,
    specification: Mapping[str, Any],
    features: pd.DataFrame,
    baseline_kwh: pd.Series,
    *,
    capacity_kwh: float,
) -> tuple[pd.Series, pd.Series]:
    """Apply the locked residual correction and return prediction/correction."""

    if not features.index.equals(baseline_kwh.index):
        raise ValueError("prediction feature and baseline indices differ")
    raw = np.asarray(model.predict(features), dtype=np.float64)
    if raw.shape != (len(features),) or not np.isfinite(raw).all():
        raise ValueError("residual model produced invalid predictions")
    maximum = float(specification["max_abs_residual_cf"])
    shrinkage = float(specification["correction_shrinkage"])
    if maximum <= 0 or not 0 < shrinkage <= 1:
        raise ValueError("invalid correction bound or shrinkage")
    correction = shrinkage * np.clip(raw, -maximum, maximum)
    baseline_cf = baseline_kwh.to_numpy(dtype=np.float64, copy=False) / capacity_kwh
    candidate = np.clip(baseline_cf + correction, 0.0, 1.02) * capacity_kwh
    return (
        pd.Series(candidate, index=features.index, name=baseline_kwh.name),
        pd.Series(correction, index=features.index, name="residual_correction_cf"),
    )


def choose_single_candidate(
    comparisons: Mapping[str, Mapping[str, Mapping[str, Mapping[str, float]]]]
) -> str | None:
    """Choose one candidate only when every group/slice delta is positive."""

    eligible: list[tuple[float, float, str]] = []
    for candidate_id, by_group in comparisons.items():
        deltas: list[float] = []
        if not by_group:
            raise ValueError(f"candidate {candidate_id} has no group comparisons")
        for slices in by_group.values():
            if not slices:
                raise ValueError(f"candidate {candidate_id} has empty slices")
            for result in slices.values():
                delta = float(result["delta"])
                if not np.isfinite(delta):
                    raise ValueError("candidate comparison contains non-finite delta")
                deltas.append(delta)
        if deltas and all(delta > 0.0 for delta in deltas):
            eligible.append((min(deltas), float(np.mean(deltas)), str(candidate_id)))
    if not eligible:
        return None
    eligible.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return eligible[0][2]


__all__ = [
    "FORBIDDEN_FEATURE_TOKENS",
    "apply_residual_model",
    "assert_strict_fit_predict_order",
    "build_run_sequence_features",
    "choose_single_candidate",
    "fit_residual_model",
    "validate_complete_runs",
]

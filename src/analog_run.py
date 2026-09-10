"""Strict-forward analog retrieval over complete 24-hour forecast runs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from src.run_sequence_residual import FORBIDDEN_FEATURE_TOKENS, validate_complete_runs
from src.temporal import operating_run_key


SUMMARY_STATISTICS = (
    "mean",
    "std",
    "min",
    "max",
    "q10",
    "q50",
    "q90",
    "linear_slope",
    "fft1_real",
    "fft1_imag",
    "peak_position",
)


def _validate_weather(
    weather: pd.DataFrame, channels: Sequence[str], *, expected_hours: int = 24
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    if not isinstance(weather, pd.DataFrame):
        raise TypeError("weather must be a DataFrame")
    validate_complete_runs(weather.index, expected_hours=expected_hours)
    forbidden = [
        str(column)
        for column in weather.columns
        if any(token in str(column).lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if forbidden:
        raise ValueError(f"forbidden actual/target/SCADA weather columns: {forbidden[:5]}")
    if len(set(channels)) != len(tuple(channels)):
        raise ValueError("weather channels must be unique")
    missing = set(channels).difference(weather.columns)
    if missing:
        raise KeyError(f"missing analog weather channels: {sorted(missing)}")
    selected = weather.loc[:, list(channels)].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(selected).all():
        raise ValueError("analog weather contains non-finite values")
    run_count = len(weather) // expected_hours
    matrix = selected.reshape(run_count, expected_hours, len(channels))
    key = operating_run_key(weather.index)
    run_keys = pd.DatetimeIndex(key[::expected_hours], name="run_key")
    if not run_keys.is_unique or not run_keys.is_monotonic_increasing:
        raise AssertionError("run keys must be unique and increasing")
    return matrix, run_keys


def _calendar_features(run_keys: pd.DatetimeIndex) -> np.ndarray:
    day = run_keys.dayofyear.to_numpy(dtype=np.float64)
    angle = 2.0 * np.pi * (day - 1.0) / 365.2425
    return np.column_stack(
        [np.sin(angle), np.cos(angle), np.sin(2.0 * angle), np.cos(2.0 * angle)]
    )


def _compact_summary(matrix: np.ndarray, run_keys: pd.DatetimeIndex) -> np.ndarray:
    run_count, hours, channels = matrix.shape
    horizon = np.arange(hours, dtype=np.float64)
    centered_horizon = horizon - horizon.mean()
    denominator = np.square(centered_horizon).sum()
    blocks: list[np.ndarray] = []
    for channel in range(channels):
        values = matrix[:, :, channel]
        quantiles = np.quantile(values, [0.10, 0.50, 0.90], axis=1).T
        transform = np.fft.rfft(values, axis=1) / float(hours)
        blocks.append(
            np.column_stack(
                [
                    values.mean(axis=1),
                    values.std(axis=1, ddof=0),
                    values.min(axis=1),
                    values.max(axis=1),
                    quantiles,
                    (values @ centered_horizon) / denominator,
                    transform[:, 1].real,
                    transform[:, 1].imag,
                    np.argmax(values, axis=1) / float(hours - 1),
                ]
            )
        )
    output = np.column_stack([*blocks, _calendar_features(run_keys)])
    expected = channels * len(SUMMARY_STATISTICS) + 4
    if output.shape != (run_count, expected):
        raise AssertionError("compact summary shape changed")
    return output


def _pca_input(matrix: np.ndarray, run_keys: pd.DatetimeIndex) -> np.ndarray:
    # Channel-major flattening of only the eight preregistered core channels.
    flattened = matrix.transpose(0, 2, 1).reshape(matrix.shape[0], -1)
    return np.column_stack([flattened, _calendar_features(run_keys)])


def _raw_embedding(
    matrix: np.ndarray, run_keys: pd.DatetimeIndex, kind: str
) -> np.ndarray:
    if kind == "compact_summary":
        return _compact_summary(matrix, run_keys)
    if kind == "pca12_core_trajectory":
        return _pca_input(matrix, run_keys)
    raise ValueError(f"unknown analog embedding: {kind}")


def _array_sha256(array: np.ndarray) -> str:
    values = np.asarray(array, dtype="<f8", order="C")
    return hashlib.sha256(values.tobytes()).hexdigest()


@dataclass
class AnalogRunRetriever:
    channels: tuple[str, ...]
    embedding_kind: str
    k: int
    pca_params: Mapping[str, Any] | None = None
    expected_hours: int = 24

    scaler_: StandardScaler | None = None
    pca_: PCA | None = None
    train_embedding_: np.ndarray | None = None
    train_run_keys_: pd.DatetimeIndex | None = None
    train_trajectories_cf_: np.ndarray | None = None
    fit_audit_: dict[str, Any] | None = None

    def fit(self, weather: pd.DataFrame, actual_cf: pd.Series) -> "AnalogRunRetriever":
        if not isinstance(actual_cf, pd.Series) or not weather.index.equals(actual_cf.index):
            raise ValueError("fit weather and actual trajectory indices differ")
        matrix, run_keys = _validate_weather(
            weather, self.channels, expected_hours=self.expected_hours
        )
        raw = _raw_embedding(matrix, run_keys, self.embedding_kind)
        self.scaler_ = StandardScaler().fit(raw)
        scaled = self.scaler_.transform(raw)
        if self.embedding_kind == "pca12_core_trajectory":
            params = dict(self.pca_params or {})
            self.pca_ = PCA(**params).fit(scaled)
            embedding = self.pca_.transform(scaled)
        else:
            if self.pca_params:
                raise ValueError("PCA parameters supplied to compact summary")
            self.pca_ = None
            embedding = scaled
        trajectories = actual_cf.to_numpy(dtype=np.float64).reshape(
            len(run_keys), self.expected_hours
        )
        complete = np.isfinite(trajectories).all(axis=1)
        if int(complete.sum()) < self.k:
            raise ValueError(
                f"only {int(complete.sum())} complete actual runs for k={self.k}"
            )
        self.train_embedding_ = np.asarray(embedding[complete], dtype=np.float64)
        self.train_run_keys_ = run_keys[complete]
        self.train_trajectories_cf_ = np.clip(trajectories[complete], 0.0, 1.02)
        self.fit_audit_ = {
            "weather_runs": int(len(run_keys)),
            "complete_actual_neighbor_runs": int(complete.sum()),
            "excluded_incomplete_actual_runs": int((~complete).sum()),
            "fit_run_start": run_keys.min().isoformat(),
            "fit_run_end": run_keys.max().isoformat(),
            "neighbor_run_start": self.train_run_keys_.min().isoformat(),
            "neighbor_run_end": self.train_run_keys_.max().isoformat(),
            "raw_embedding_columns": int(raw.shape[1]),
            "final_embedding_columns": int(self.train_embedding_.shape[1]),
            "scaler_fit_runs": int(len(run_keys)),
            "pca_fit_runs": int(len(run_keys)) if self.pca_ is not None else 0,
            "actual_as_embedding_feature": False,
            "embedding_sha256": _array_sha256(self.train_embedding_),
            "trajectory_sha256": _array_sha256(self.train_trajectories_cf_),
        }
        return self

    def _transform(self, weather: pd.DataFrame) -> tuple[np.ndarray, pd.DatetimeIndex]:
        if self.scaler_ is None or self.train_embedding_ is None:
            raise RuntimeError("retriever is not fitted")
        matrix, run_keys = _validate_weather(
            weather, self.channels, expected_hours=self.expected_hours
        )
        raw = _raw_embedding(matrix, run_keys, self.embedding_kind)
        scaled = self.scaler_.transform(raw)
        embedding = self.pca_.transform(scaled) if self.pca_ is not None else scaled
        if not np.isfinite(embedding).all():
            raise ValueError("apply embedding contains non-finite values")
        return np.asarray(embedding, dtype=np.float64), run_keys

    def predict(self, weather: pd.DataFrame) -> tuple[pd.Series, dict[str, Any]]:
        if (
            self.train_embedding_ is None
            or self.train_run_keys_ is None
            or self.train_trajectories_cf_ is None
        ):
            raise RuntimeError("retriever is not fitted")
        apply_embedding, apply_keys = self._transform(weather)
        if self.train_run_keys_.max() >= apply_keys.min():
            raise ValueError("neighbor library is not strictly earlier than apply runs")
        distance = pairwise_distances(
            apply_embedding, self.train_embedding_, metric="euclidean", n_jobs=1
        )
        prediction = np.empty((len(apply_keys), self.expected_hours), dtype=np.float64)
        chosen_positions = np.empty((len(apply_keys), self.k), dtype=np.int32)
        chosen_distances = np.empty((len(apply_keys), self.k), dtype=np.float64)
        minimum_gap_hours = np.inf
        for row, apply_key in enumerate(apply_keys):
            eligible = np.flatnonzero(self.train_run_keys_ < apply_key)
            if len(eligible) < self.k:
                raise ValueError(f"fewer than k past neighbors for apply run {apply_key}")
            order = np.argsort(distance[row, eligible], kind="stable")[: self.k]
            chosen = eligible[order]
            values = distance[row, chosen]
            zero = values <= 1e-12
            if zero.any():
                weights = zero.astype(np.float64) / float(zero.sum())
            else:
                inverse = 1.0 / np.maximum(values, 1e-6)
                weights = inverse / inverse.sum()
            prediction[row] = weights @ self.train_trajectories_cf_[chosen]
            chosen_positions[row] = chosen
            chosen_distances[row] = values
            latest_neighbor = self.train_run_keys_[chosen].max()
            gap = (apply_key - latest_neighbor) / pd.Timedelta(hours=1)
            minimum_gap_hours = min(minimum_gap_hours, float(gap))
        prediction = np.clip(prediction, 0.0, 1.02)
        output = pd.Series(
            prediction.reshape(-1), index=weather.index, name="analog_cf"
        )
        audit = {
            "apply_runs": int(len(apply_keys)),
            "apply_run_start": apply_keys.min().isoformat(),
            "apply_run_end": apply_keys.max().isoformat(),
            "neighbor_count_per_run": int(self.k),
            "neighbor_timestamp_overlap_count": 0,
            "neighbor_strictly_before_apply": True,
            "minimum_neighbor_gap_hours": float(minimum_gap_hours),
            "chosen_position_sha256": hashlib.sha256(
                np.asarray(chosen_positions, dtype="<i4").tobytes()
            ).hexdigest(),
            "chosen_distance_sha256": _array_sha256(chosen_distances),
            "prediction_sha256": _array_sha256(prediction),
            "query_actual_values_accessed": False,
        }
        return output, audit


def blend_analog_with_baseline(
    analog_cf: pd.Series,
    baseline_kwh: pd.Series,
    *,
    capacity_kwh: float,
    analog_weight: float,
) -> pd.Series:
    if not analog_cf.index.equals(baseline_kwh.index):
        raise ValueError("analog and baseline indices differ")
    weight = float(analog_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("analog weight must be in [0,1]")
    analog = analog_cf.to_numpy(dtype=np.float64, copy=False)
    baseline = baseline_kwh.to_numpy(dtype=np.float64, copy=False) / capacity_kwh
    if not np.isfinite(analog).all() or not np.isfinite(baseline).all():
        raise ValueError("analog or baseline contains non-finite values")
    output = np.clip((1.0 - weight) * baseline + weight * analog, 0.0, 1.02)
    return pd.Series(output * capacity_kwh, index=analog_cf.index, name=baseline_kwh.name)


__all__ = [
    "AnalogRunRetriever",
    "SUMMARY_STATISTICS",
    "blend_analog_with_baseline",
]

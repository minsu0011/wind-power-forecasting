"""Run-level linear models over all raw LDAPS/GFS wind-node trajectories.

One sample is one complete 24-hour operating run.  Each of the 200 registered
raw wind-node channels is independently standardized with fit-period values,
projected onto the first four coefficients of a fixed orthonormal DCT-II, and
kept as a separate channel.  The estimators jointly emit all 24 capacity
factors.  No baseline, target, SCADA, calendar, issuance, Public, or scale
value enters the model features.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.raw_grid_wind import TIME_FEATURES, canonical_sha256
from src.run_sequence_residual import FORBIDDEN_FEATURE_TOKENS, validate_complete_runs
from src.temporal import operating_run_key


EXPECTED_HOURS = 24
EXPECTED_RAW_CHANNELS = 200
DCT_COEFFICIENTS = 4
EXPECTED_INPUT_DIMENSION = EXPECTED_RAW_CHANNELS * DCT_COEFFICIENTS
ESTIMATOR_IDS: tuple[str, ...] = ("ridge_a100", "pls12")
BLEND_WEIGHTS: tuple[float, ...] = (0.05, 0.10)
CANDIDATE_KEYS: tuple[str, ...] = tuple(
    f"{estimator_id}_w{int(round(weight * 100)):02d}"
    for estimator_id in ESTIMATOR_IDS
    for weight in BLEND_WEIGHTS
)


def orthonormal_dct2_basis(
    hours: int = EXPECTED_HOURS, coefficients: int = DCT_COEFFICIENTS
) -> np.ndarray:
    """Return the first rows of a deterministic orthonormal DCT-II basis."""

    if not isinstance(hours, int) or hours <= 1:
        raise ValueError("hours must be an integer greater than one")
    if not isinstance(coefficients, int) or not 1 <= coefficients <= hours:
        raise ValueError("coefficients must be in [1, hours]")
    position = np.arange(hours, dtype=np.float64) + 0.5
    frequency = np.arange(coefficients, dtype=np.float64)[:, None]
    basis = np.cos(np.pi * frequency * position[None, :] / float(hours))
    basis[0] *= np.sqrt(1.0 / float(hours))
    basis[1:] *= np.sqrt(2.0 / float(hours))
    if not np.allclose(
        basis @ basis.T, np.eye(coefficients), rtol=0.0, atol=1e-12
    ):
        raise AssertionError("DCT-II basis lost orthonormality")
    return basis


def registered_raw_node_columns(features: pd.DataFrame) -> tuple[str, ...]:
    """Extract and validate the frozen 200 raw-node columns from the 212 frame."""

    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a DataFrame")
    columns = tuple(map(str, features.columns))
    if len(columns) != EXPECTED_RAW_CHANNELS + len(TIME_FEATURES):
        raise ValueError("raw-grid hourly feature count changed")
    if columns[-len(TIME_FEATURES) :] != tuple(TIME_FEATURES):
        raise ValueError("clock/issuance suffix changed")
    raw = columns[:EXPECTED_RAW_CHANNELS]
    if len(set(raw)) != EXPECTED_RAW_CHANNELS:
        raise ValueError("raw-node feature names are not unique")
    if any(column in TIME_FEATURES for column in raw):
        raise ValueError("clock/issuance feature entered raw-node channels")
    expected_ldaps = 16 * 8
    if sum(column.startswith("ldaps__grid_") for column in raw) != expected_ldaps:
        raise ValueError("LDAPS raw-node channel count changed")
    if sum(column.startswith("gfs__grid_") for column in raw) != 9 * 8:
        raise ValueError("GFS raw-node channel count changed")
    forbidden = [
        column
        for column in raw
        if any(token in column.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if forbidden:
        raise ValueError(f"forbidden target/SCADA channel: {forbidden[:3]}")
    return raw


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _run_keys(index: pd.DatetimeIndex, expected_hours: int) -> pd.DatetimeIndex:
    validate_complete_runs(index, expected_hours=expected_hours)
    keys = pd.DatetimeIndex(
        operating_run_key(index)[::expected_hours], name="forecast_run_kst_date"
    )
    if not keys.is_unique or not keys.is_monotonic_increasing:
        raise AssertionError("run keys must be unique and increasing")
    return keys


@dataclass
class RawNodeDCTTransformer:
    """Fit-fold-only raw-channel imputation/scaling followed by fixed DCT."""

    raw_columns: tuple[str, ...]
    coefficients: int = DCT_COEFFICIENTS
    expected_hours: int = EXPECTED_HOURS
    scale_floor: float = 1e-6
    medians_: np.ndarray | None = None
    means_: np.ndarray | None = None
    scales_: np.ndarray | None = None

    def _validate(self, features: pd.DataFrame) -> np.ndarray:
        if not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a DataFrame")
        observed = registered_raw_node_columns(features)
        if observed != self.raw_columns:
            raise ValueError("raw-node feature schema/order changed")
        _run_keys(features.index, self.expected_hours)
        values = features.loc[:, list(self.raw_columns)].to_numpy(
            dtype=np.float64, copy=True
        )
        if np.isinf(values).any():
            raise ValueError("raw-node features contain infinity")
        return values

    def fit(self, features: pd.DataFrame) -> "RawNodeDCTTransformer":
        values = self._validate(features)
        with np.errstate(all="ignore"):
            medians = np.nanmedian(values, axis=0)
        if not np.isfinite(medians).all():
            raise ValueError("a fit raw-node channel is entirely missing")
        missing = np.isnan(values)
        if missing.any():
            rows, columns = np.where(missing)
            values[rows, columns] = medians[columns]
        means = values.mean(axis=0)
        scales = values.std(axis=0, ddof=0)
        if not np.isfinite(means).all() or not np.isfinite(scales).all():
            raise ValueError("non-finite fit-fold raw-node scaler")
        self.medians_ = medians
        self.means_ = means
        self.scales_ = np.maximum(scales, float(self.scale_floor))
        self.fit_missing_cells_ = int(missing.sum())
        self.fit_hourly_rows_ = int(len(features))
        self.fit_run_count_ = int(len(features) // self.expected_hours)
        return self

    def transform(
        self, features: pd.DataFrame
    ) -> tuple[np.ndarray, pd.DatetimeIndex]:
        if self.medians_ is None or self.means_ is None or self.scales_ is None:
            raise RuntimeError("RawNodeDCTTransformer must be fitted first")
        values = self._validate(features)
        missing = np.isnan(values)
        if missing.any():
            rows, columns = np.where(missing)
            values[rows, columns] = self.medians_[columns]
        if not np.isfinite(values).all():
            raise ValueError("raw-node imputation left non-finite values")
        standardized = (values - self.means_[None, :]) / self.scales_[None, :]
        run_count = len(features) // self.expected_hours
        tensor = standardized.reshape(
            run_count, self.expected_hours, EXPECTED_RAW_CHANNELS
        ).transpose(0, 2, 1)
        basis = orthonormal_dct2_basis(self.expected_hours, self.coefficients)
        projected = np.einsum("rch,kh->rck", tensor, basis, optimize=True)
        matrix = projected.reshape(run_count, -1)
        if matrix.shape != (run_count, EXPECTED_INPUT_DIMENSION):
            raise AssertionError("raw temporal DCT dimension changed")
        if not np.isfinite(matrix).all():
            raise ValueError("raw temporal DCT matrix contains non-finite values")
        return matrix, _run_keys(features.index, self.expected_hours)

    def fit_transform(
        self, features: pd.DataFrame
    ) -> tuple[np.ndarray, pd.DatetimeIndex]:
        return self.fit(features).transform(features)

    def metadata(self) -> dict[str, Any]:
        if self.medians_ is None or self.means_ is None or self.scales_ is None:
            raise RuntimeError("transformer is not fitted")
        return {
            "raw_channel_count": len(self.raw_columns),
            "raw_columns_sha256": canonical_sha256(self.raw_columns),
            "dct_coefficients_per_channel": self.coefficients,
            "input_dimension": len(self.raw_columns) * self.coefficients,
            "median_sha256": _array_sha256(self.medians_),
            "mean_sha256": _array_sha256(self.means_),
            "scale_sha256": _array_sha256(self.scales_),
            "fit_missing_cells": self.fit_missing_cells_,
            "fit_hourly_rows": self.fit_hourly_rows_,
            "fit_run_count": self.fit_run_count_,
            "clock_issuance_features_used": False,
            "baseline_target_scada_public_scale_features_used": False,
        }


class RawTemporalMultiOutputRegressor:
    """One of two fixed joint 24-output estimators over raw-node DCT inputs."""

    def __init__(self, estimator_id: str) -> None:
        if estimator_id not in ESTIMATOR_IDS:
            raise ValueError(f"estimator_id must be one of {ESTIMATOR_IDS}")
        self.estimator_id = estimator_id

    def _make_estimator(self) -> Any:
        if self.estimator_id == "ridge_a100":
            return Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        Ridge(alpha=100.0, fit_intercept=True, solver="svd"),
                    ),
                ]
            )
        return PLSRegression(
            n_components=12, scale=True, max_iter=500, tol=1e-6
        )

    def fit(
        self, features: pd.DataFrame, target_cf: pd.Series
    ) -> "RawTemporalMultiOutputRegressor":
        if not isinstance(target_cf, pd.Series):
            raise TypeError("target_cf must be a Series")
        if not features.index.equals(target_cf.index):
            raise ValueError("feature and target indexes differ")
        requested_fit_keys = _run_keys(features.index, EXPECTED_HOURS)
        values = target_cf.to_numpy(dtype=np.float64, copy=False)
        target_by_run = values.reshape(-1, EXPECTED_HOURS)
        eligible_runs = np.isfinite(target_by_run).all(axis=1)
        if not eligible_runs.any():
            raise ValueError("no complete run has 24 finite target values")
        eligible_hourly = np.repeat(eligible_runs, EXPECTED_HOURS)
        eligible_features = features.iloc[eligible_hourly]
        target = target_by_run[eligible_runs]
        fit_keys = _run_keys(eligible_features.index, EXPECTED_HOURS)
        raw_columns = registered_raw_node_columns(features)
        transformer = RawNodeDCTTransformer(raw_columns=raw_columns)
        matrix, transformed_keys = transformer.fit_transform(eligible_features)
        if not transformed_keys.equals(fit_keys):
            raise AssertionError("fit run-key transformation changed")
        estimator = self._make_estimator()
        estimator.fit(matrix, target)
        self.transformer_ = transformer
        self.estimator_ = estimator
        self.requested_fit_index_ = features.index.copy()
        self.fit_index_ = eligible_features.index.copy()
        self.requested_fit_run_keys_ = requested_fit_keys
        self.fit_run_keys_ = fit_keys
        self.dropped_run_keys_ = requested_fit_keys[~eligible_runs]
        self.eligible_run_mask_sha256_ = _array_sha256(eligible_runs.astype(np.uint8))
        self.fit_target_sha256_ = _array_sha256(target)
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not hasattr(self, "estimator_"):
            raise RuntimeError("regressor must be fitted first")
        matrix, _ = self.transformer_.transform(features)
        prediction = np.asarray(self.estimator_.predict(matrix), dtype=np.float64)
        expected = (len(features) // EXPECTED_HOURS, EXPECTED_HOURS)
        if prediction.shape != expected:
            raise AssertionError(f"joint prediction shape changed: {prediction.shape}")
        if not np.isfinite(prediction).all():
            raise ValueError("joint prediction contains non-finite values")
        return pd.Series(
            np.clip(prediction.reshape(-1), 0.0, 1.02),
            index=features.index,
            name=f"{self.estimator_id}_cf",
        )

    def metadata(self) -> dict[str, Any]:
        if not hasattr(self, "estimator_"):
            raise RuntimeError("regressor is not fitted")
        parameters: dict[str, Any]
        if self.estimator_id == "ridge_a100":
            parameters = {"alpha": 100.0, "fit_intercept": True, "solver": "svd"}
        else:
            parameters = {
                "n_components": 12,
                "scale": True,
                "max_iter": 500,
                "tol": 1e-6,
            }
        return {
            "estimator_id": self.estimator_id,
            "parameters": parameters,
            "target": "direct_joint_capacity_factor_24",
            "fit_start": self.fit_index_.min(),
            "fit_end": self.fit_index_.max(),
            "fit_hourly_rows": len(self.fit_index_),
            "fit_runs": len(self.fit_run_keys_),
            "requested_fit_hourly_rows": len(self.requested_fit_index_),
            "requested_fit_runs": len(self.requested_fit_run_keys_),
            "dropped_incomplete_target_runs": len(self.dropped_run_keys_),
            "dropped_incomplete_target_run_keys": [
                value.isoformat() for value in self.dropped_run_keys_
            ],
            "dropped_incomplete_target_run_keys_sha256": canonical_sha256(
                [value.isoformat() for value in self.dropped_run_keys_]
            ),
            "eligible_run_mask_sha256": self.eligible_run_mask_sha256_,
            "all_retained_runs_have_24_finite_targets": True,
            "fit_target_sha256": self.fit_target_sha256_,
            "transformer": self.transformer_.metadata(),
        }


def assert_fit_before_apply(
    fit_index: pd.DatetimeIndex, application_index: pd.DatetimeIndex
) -> dict[str, Any]:
    """Enforce complete-run, disjoint, strict chronological application."""

    _run_keys(fit_index, EXPECTED_HOURS)
    _run_keys(application_index, EXPECTED_HOURS)
    overlap = fit_index.intersection(application_index)
    if len(overlap) or fit_index.max() >= application_index.min():
        raise ValueError("fit period must be disjoint and strictly before application")
    return {
        "fit_start": fit_index.min(),
        "fit_end": fit_index.max(),
        "application_start": application_index.min(),
        "application_end": application_index.max(),
        "fit_end_before_application_start": True,
        "overlap_count": 0,
        "fit_runs_complete": True,
        "application_runs_complete": True,
    }


def blend_direct_prediction(
    baseline_kwh: pd.Series,
    model_cf: pd.Series,
    *,
    capacity_kwh: float,
    weight: float,
) -> pd.Series:
    """Blend a direct joint-CF prediction with the fixed baseline."""

    if weight not in (0.0, *BLEND_WEIGHTS):
        raise ValueError("blend weight is not preregistered")
    if not baseline_kwh.index.equals(model_cf.index):
        raise ValueError("baseline and model prediction indexes differ")
    if not np.isfinite(capacity_kwh) or capacity_kwh <= 0:
        raise ValueError("capacity_kwh must be positive and finite")
    if weight == 0.0:
        return baseline_kwh.copy()
    baseline = baseline_kwh.to_numpy(dtype=np.float64, copy=False)
    model = model_cf.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(baseline).all() or not np.isfinite(model).all():
        raise ValueError("blend inputs contain non-finite values")
    values = (1.0 - weight) * baseline + weight * (
        np.clip(model, 0.0, 1.02) * capacity_kwh
    )
    return pd.Series(
        np.clip(values, 0.0, 1.02 * capacity_kwh),
        index=baseline_kwh.index,
        name=baseline_kwh.name,
    )


def candidate_frame(
    baseline_kwh: pd.Series,
    predictions: Mapping[str, pd.Series],
    *,
    capacity_kwh: float,
) -> pd.DataFrame:
    """Materialize exactly the four preregistered candidates in frozen order."""

    if tuple(predictions) != ESTIMATOR_IDS:
        raise ValueError("estimator prediction order changed")
    result = pd.DataFrame(index=baseline_kwh.index)
    for estimator_id in ESTIMATOR_IDS:
        for weight in BLEND_WEIGHTS:
            key = f"{estimator_id}_w{int(round(weight * 100)):02d}"
            result[key] = blend_direct_prediction(
                baseline_kwh,
                predictions[estimator_id],
                capacity_kwh=capacity_kwh,
                weight=weight,
            )
    if tuple(result.columns) != CANDIDATE_KEYS:
        raise AssertionError("candidate order changed")
    identity = blend_direct_prediction(
        baseline_kwh,
        predictions[ESTIMATOR_IDS[0]],
        capacity_kwh=capacity_kwh,
        weight=0.0,
    )
    if identity.to_numpy().tobytes() != baseline_kwh.to_numpy().tobytes():
        raise AssertionError("weight-zero identity is not value-bit exact")
    return result


def parse_candidate_key(key: str) -> tuple[str, float]:
    for estimator_id in ESTIMATOR_IDS:
        for weight in BLEND_WEIGHTS:
            expected = f"{estimator_id}_w{int(round(weight * 100)):02d}"
            if key == expected:
                return estimator_id, weight
    raise ValueError(f"candidate is not preregistered: {key}")


def select_group_candidate(
    comparisons: Mapping[str, Mapping[str, Mapping[str, Any]]],
    required_slices: Sequence[str],
) -> tuple[str | None, dict[str, Any]]:
    """Select only candidates strictly positive on every registered slice."""

    if set(comparisons) != set(CANDIDATE_KEYS) or len(comparisons) != len(CANDIDATE_KEYS):
        raise ValueError("candidate key set changed")
    required = tuple(required_slices)
    audit: dict[str, Any] = {}
    eligible: list[str] = []
    for key in CANDIDATE_KEYS:
        records = comparisons[key]
        if set(records) != set(required) or len(records) != len(required):
            raise ValueError(f"slice key set changed for {key}")
        deltas: dict[str, float] = {}
        for name in required:
            record = records[name]
            exact = float(record["candidate"]["score"]) - float(
                record["baseline"]["score"]
            )
            if float(record["delta"]) != exact:
                raise ValueError("candidate delta arithmetic changed")
            deltas[name] = exact
        estimator_id, weight = parse_candidate_key(key)
        row = {
            "estimator_id": estimator_id,
            "weight": weight,
            "deltas": deltas,
            "minimum": min(deltas.values()),
            "mean": float(np.mean(list(deltas.values()))),
            "all_slices_strictly_positive": all(value > 0.0 for value in deltas.values()),
        }
        audit[key] = row
        if row["all_slices_strictly_positive"]:
            eligible.append(key)
    if not eligible:
        return None, {"candidates": audit, "selected": "identity"}

    def order(key: str) -> tuple[float, float, int, float]:
        row = audit[key]
        return (
            float(row["minimum"]),
            float(row["mean"]),
            -ESTIMATOR_IDS.index(str(row["estimator_id"])),
            -float(row["weight"]),
        )

    selected = max(eligible, key=order)
    return selected, {"candidates": audit, "selected": selected}


__all__ = [
    "BLEND_WEIGHTS",
    "CANDIDATE_KEYS",
    "DCT_COEFFICIENTS",
    "ESTIMATOR_IDS",
    "EXPECTED_HOURS",
    "EXPECTED_INPUT_DIMENSION",
    "EXPECTED_RAW_CHANNELS",
    "RawNodeDCTTransformer",
    "RawTemporalMultiOutputRegressor",
    "assert_fit_before_apply",
    "blend_direct_prediction",
    "candidate_frame",
    "orthonormal_dct2_basis",
    "parse_candidate_key",
    "registered_raw_node_columns",
    "select_group_candidate",
]

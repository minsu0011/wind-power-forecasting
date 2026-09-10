"""Nonlinear joint run model over the complete 200-node by 24-hour tensor."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler

from src.raw_grid_wind import TIME_FEATURES, canonical_sha256
from src.run_sequence_residual import FORBIDDEN_FEATURE_TOKENS, validate_complete_runs
from src.temporal import operating_run_key


EXPECTED_HOURS = 24
EXPECTED_RAW_CHANNELS = 200
EXPECTED_FLAT_DIMENSION = EXPECTED_HOURS * EXPECTED_RAW_CHANNELS
PCA_COMPONENTS = 32
ESTIMATOR_ID = "pca32_rbf_a10_g03125"
BLEND_WEIGHTS: tuple[float, ...] = (0.05, 0.10)
CANDIDATE_KEYS: tuple[str, ...] = tuple(
    f"{ESTIMATOR_ID}_w{int(round(weight * 100)):02d}" for weight in BLEND_WEIGHTS
)


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(values)).tobytes()).hexdigest()


def registered_raw_node_columns(features: pd.DataFrame) -> tuple[str, ...]:
    """Return the exact 200 raw-node columns and reject feature contamination."""

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
    if sum(value.startswith("ldaps__grid_") for value in raw) != 16 * 8:
        raise ValueError("LDAPS raw-node channel count changed")
    if sum(value.startswith("gfs__grid_") for value in raw) != 9 * 8:
        raise ValueError("GFS raw-node channel count changed")
    forbidden = [
        value
        for value in raw
        if value in TIME_FEATURES
        or any(token in value.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if forbidden:
        raise ValueError(f"forbidden raw-node feature: {forbidden[:3]}")
    return raw


def _run_keys(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    validate_complete_runs(index, expected_hours=EXPECTED_HOURS)
    keys = pd.DatetimeIndex(
        operating_run_key(index)[::EXPECTED_HOURS], name="forecast_run_kst_date"
    )
    if not keys.is_unique or not keys.is_monotonic_increasing:
        raise AssertionError("run keys must be unique and increasing")
    return keys


@dataclass
class RawRunPCAKernelTransformer:
    """Eligible-fit-only imputation, horizon scaling, and full-SVD PCA32."""

    raw_columns: tuple[str, ...]
    scale_floor: float = 1e-12
    medians_: np.ndarray | None = None
    scaler_: StandardScaler | None = None
    pca_: PCA | None = None

    def _hourly_values(self, features: pd.DataFrame) -> np.ndarray:
        observed = registered_raw_node_columns(features)
        if observed != self.raw_columns:
            raise ValueError("raw-node feature schema/order changed")
        _run_keys(features.index)
        values = features.loc[:, list(self.raw_columns)].to_numpy(
            dtype=np.float64, copy=True
        )
        if np.isinf(values).any():
            raise ValueError("raw-node values contain infinity")
        return values

    @staticmethod
    def _flatten(values: np.ndarray) -> np.ndarray:
        if values.ndim != 2 or values.shape[1] != EXPECTED_RAW_CHANNELS:
            raise ValueError("hourly raw-node matrix shape changed")
        if len(values) % EXPECTED_HOURS:
            raise ValueError("hourly matrix does not contain complete runs")
        runs = len(values) // EXPECTED_HOURS
        matrix = values.reshape(runs, EXPECTED_HOURS, EXPECTED_RAW_CHANNELS)
        matrix = matrix.transpose(0, 2, 1).reshape(runs, EXPECTED_FLAT_DIMENSION)
        if matrix.shape != (runs, EXPECTED_FLAT_DIMENSION):
            raise AssertionError("run flatten dimension changed")
        return matrix

    def fit(self, features: pd.DataFrame) -> "RawRunPCAKernelTransformer":
        values = self._hourly_values(features)
        with np.errstate(all="ignore"):
            medians = np.nanmedian(values, axis=0)
        if not np.isfinite(medians).all():
            raise ValueError("a fit raw-node channel is entirely missing")
        missing = np.isnan(values)
        if missing.any():
            rows, columns = np.where(missing)
            values[rows, columns] = medians[columns]
        flat = self._flatten(values)
        if len(flat) < PCA_COMPONENTS:
            raise ValueError("eligible fit runs are fewer than PCA components")
        scaler = StandardScaler(with_mean=True, with_std=True).fit(flat)
        if np.any(scaler.scale_ < self.scale_floor):
            raise ValueError("a horizon-specific raw position has zero fit variance")
        standardized = scaler.transform(flat)
        pca = PCA(
            n_components=PCA_COMPONENTS,
            svd_solver="full",
            whiten=False,
        ).fit(standardized)
        projected = pca.transform(standardized)
        if projected.shape != (len(flat), PCA_COMPONENTS):
            raise AssertionError("PCA fit output dimension changed")
        if not np.isfinite(projected).all():
            raise ValueError("PCA fit output contains non-finite values")
        self.medians_ = medians
        self.scaler_ = scaler
        self.pca_ = pca
        self.fit_missing_cells_ = int(missing.sum())
        self.fit_hourly_rows_ = len(features)
        self.fit_runs_ = len(flat)
        return self

    def transform(
        self, features: pd.DataFrame
    ) -> tuple[np.ndarray, pd.DatetimeIndex]:
        if self.medians_ is None or self.scaler_ is None or self.pca_ is None:
            raise RuntimeError("RawRunPCAKernelTransformer must be fitted first")
        values = self._hourly_values(features)
        missing = np.isnan(values)
        if missing.any():
            rows, columns = np.where(missing)
            values[rows, columns] = self.medians_[columns]
        if not np.isfinite(values).all():
            raise ValueError("raw-node imputation left non-finite values")
        flat = self._flatten(values)
        projected = self.pca_.transform(self.scaler_.transform(flat))
        if projected.shape != (len(flat), PCA_COMPONENTS):
            raise AssertionError("PCA application output dimension changed")
        if not np.isfinite(projected).all():
            raise ValueError("PCA application output contains non-finite values")
        return projected, _run_keys(features.index)

    def fit_transform(
        self, features: pd.DataFrame
    ) -> tuple[np.ndarray, pd.DatetimeIndex]:
        self.fit(features)
        return self.transform(features)

    def metadata(self) -> dict[str, Any]:
        if self.medians_ is None or self.scaler_ is None or self.pca_ is None:
            raise RuntimeError("transformer is not fitted")
        return {
            "raw_channel_count": len(self.raw_columns),
            "raw_columns_sha256": canonical_sha256(self.raw_columns),
            "flat_dimension": EXPECTED_FLAT_DIMENSION,
            "pca_components": PCA_COMPONENTS,
            "pca_svd_solver": "full",
            "pca_whiten": False,
            "median_sha256": _array_sha256(self.medians_),
            "scaler_mean_sha256": _array_sha256(self.scaler_.mean_),
            "scaler_scale_sha256": _array_sha256(self.scaler_.scale_),
            "pca_components_sha256": _array_sha256(self.pca_.components_),
            "pca_mean_sha256": _array_sha256(self.pca_.mean_),
            "explained_variance_ratio_sum": float(
                self.pca_.explained_variance_ratio_.sum()
            ),
            "fit_missing_cells": self.fit_missing_cells_,
            "fit_hourly_rows": self.fit_hourly_rows_,
            "fit_runs": self.fit_runs_,
            "baseline_clock_issuance_target_scada_public_scale_features_used": False,
        }


class RawTemporalKernelRegressor:
    """Fixed PCA32 plus multi-output RBF KernelRidge model."""

    def fit(
        self, features: pd.DataFrame, target_cf: pd.Series
    ) -> "RawTemporalKernelRegressor":
        if not isinstance(target_cf, pd.Series):
            raise TypeError("target_cf must be a Series")
        if not features.index.equals(target_cf.index):
            raise ValueError("feature and target indexes differ")
        requested_keys = _run_keys(features.index)
        target_by_run = target_cf.to_numpy(dtype=np.float64, copy=False).reshape(
            -1, EXPECTED_HOURS
        )
        eligible = np.isfinite(target_by_run).all(axis=1)
        if not eligible.any():
            raise ValueError("no complete run has 24 finite target values")
        eligible_features = features.iloc[np.repeat(eligible, EXPECTED_HOURS)]
        target = target_by_run[eligible]
        eligible_keys = _run_keys(eligible_features.index)
        transformer = RawRunPCAKernelTransformer(
            raw_columns=registered_raw_node_columns(features)
        )
        matrix, transformed_keys = transformer.fit_transform(eligible_features)
        if not transformed_keys.equals(eligible_keys):
            raise AssertionError("eligible run keys changed during preprocessing")
        estimator = KernelRidge(
            alpha=10.0,
            kernel="rbf",
            gamma=1.0 / PCA_COMPONENTS,
        ).fit(matrix, target)
        self.transformer_ = transformer
        self.estimator_ = estimator
        self.requested_fit_index_ = features.index.copy()
        self.fit_index_ = eligible_features.index.copy()
        self.requested_fit_run_keys_ = requested_keys
        self.fit_run_keys_ = eligible_keys
        self.dropped_run_keys_ = requested_keys[~eligible]
        self.eligible_run_mask_sha256_ = _array_sha256(eligible.astype(np.uint8))
        self.fit_target_sha256_ = _array_sha256(target)
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not hasattr(self, "estimator_"):
            raise RuntimeError("regressor must be fitted first")
        matrix, _ = self.transformer_.transform(features)
        prediction = np.asarray(self.estimator_.predict(matrix), dtype=np.float64)
        expected = (len(features) // EXPECTED_HOURS, EXPECTED_HOURS)
        if prediction.shape != expected:
            raise AssertionError(f"joint kernel prediction shape changed: {prediction.shape}")
        if not np.isfinite(prediction).all():
            raise ValueError("joint kernel prediction contains non-finite values")
        return pd.Series(
            np.clip(prediction.reshape(-1), 0.0, 1.02),
            index=features.index,
            name=f"{ESTIMATOR_ID}_cf",
        )

    def metadata(self) -> dict[str, Any]:
        if not hasattr(self, "estimator_"):
            raise RuntimeError("regressor is not fitted")
        dropped = [value.isoformat() for value in self.dropped_run_keys_]
        return {
            "estimator_id": ESTIMATOR_ID,
            "parameters": {"alpha": 10.0, "kernel": "rbf", "gamma": 0.03125},
            "target": "direct_joint_capacity_factor_24",
            "fit_start": self.fit_index_.min(),
            "fit_end": self.fit_index_.max(),
            "fit_hourly_rows": len(self.fit_index_),
            "fit_runs": len(self.fit_run_keys_),
            "requested_fit_hourly_rows": len(self.requested_fit_index_),
            "requested_fit_runs": len(self.requested_fit_run_keys_),
            "dropped_incomplete_target_runs": len(dropped),
            "dropped_incomplete_target_run_keys": dropped,
            "dropped_incomplete_target_run_keys_sha256": canonical_sha256(dropped),
            "eligible_run_mask_sha256": self.eligible_run_mask_sha256_,
            "all_retained_runs_have_24_finite_targets": True,
            "fit_target_sha256": self.fit_target_sha256_,
            "dual_coef_sha256": _array_sha256(self.estimator_.dual_coef_),
            "transformer": self.transformer_.metadata(),
        }


def assert_fit_before_apply(
    fit_index: pd.DatetimeIndex, application_index: pd.DatetimeIndex
) -> dict[str, Any]:
    _run_keys(fit_index)
    _run_keys(application_index)
    if len(fit_index.intersection(application_index)) or fit_index.max() >= application_index.min():
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


def blend_kernel_prediction(
    baseline_kwh: pd.Series,
    model_cf: pd.Series,
    *,
    capacity_kwh: float,
    weight: float,
) -> pd.Series:
    if weight not in (0.0, *BLEND_WEIGHTS):
        raise ValueError("blend weight is not preregistered")
    if not baseline_kwh.index.equals(model_cf.index):
        raise ValueError("baseline and model prediction indexes differ")
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
    baseline_kwh: pd.Series, model_cf: pd.Series, *, capacity_kwh: float
) -> pd.DataFrame:
    result = pd.DataFrame(index=baseline_kwh.index)
    for weight in BLEND_WEIGHTS:
        key = f"{ESTIMATOR_ID}_w{int(round(weight * 100)):02d}"
        result[key] = blend_kernel_prediction(
            baseline_kwh, model_cf, capacity_kwh=capacity_kwh, weight=weight
        )
    if tuple(result.columns) != CANDIDATE_KEYS:
        raise AssertionError("kernel candidate order changed")
    identity = blend_kernel_prediction(
        baseline_kwh, model_cf, capacity_kwh=capacity_kwh, weight=0.0
    )
    if identity.to_numpy().tobytes() != baseline_kwh.to_numpy().tobytes():
        raise AssertionError("weight-zero identity is not value-bit exact")
    return result


def parse_candidate_key(key: str) -> float:
    for weight in BLEND_WEIGHTS:
        if key == f"{ESTIMATOR_ID}_w{int(round(weight * 100)):02d}":
            return weight
    raise ValueError(f"candidate is not preregistered: {key}")


def select_group_candidate(
    comparisons: Mapping[str, Mapping[str, Mapping[str, Any]]],
    required_slices: Sequence[str],
) -> tuple[str | None, dict[str, Any]]:
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
        weight = parse_candidate_key(key)
        row = {
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
    selected = max(
        eligible,
        key=lambda key: (
            float(audit[key]["minimum"]),
            float(audit[key]["mean"]),
            -float(audit[key]["weight"]),
        ),
    )
    return selected, {"candidates": audit, "selected": selected}


__all__ = [
    "BLEND_WEIGHTS",
    "CANDIDATE_KEYS",
    "ESTIMATOR_ID",
    "EXPECTED_FLAT_DIMENSION",
    "EXPECTED_HOURS",
    "EXPECTED_RAW_CHANNELS",
    "PCA_COMPONENTS",
    "RawRunPCAKernelTransformer",
    "RawTemporalKernelRegressor",
    "assert_fit_before_apply",
    "blend_kernel_prediction",
    "candidate_frame",
    "parse_candidate_key",
    "registered_raw_node_columns",
    "select_group_candidate",
]

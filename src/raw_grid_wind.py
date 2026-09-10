"""All-node raw NWP wind-field features and fixed LightGBM regressors.

The ordinary weather feature builder deliberately summarizes the weather grid.
This module instead binds every registered grid id to its own ordered feature
columns.  It contains no label, SCADA, residual, prediction, or leaderboard
feature path; labels enter only the estimator's explicit ``fit`` call.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd


TIME_COL = "forecast_kst_dtm"
AVAILABLE_COL = "data_available_kst_dtm"
GRID_COL = "grid_id"
COORD_COLS = ("latitude", "longitude")

LDAPS_CHANNELS: tuple[str, ...] = (
    "heightAboveGround_10_10u",
    "heightAboveGround_10_10v",
    "heightAboveGround_50_50MUmax",
    "heightAboveGround_50_50MUmin",
    "heightAboveGround_50_50MVmax",
    "heightAboveGround_50_50MVmin",
    "heightAboveGround_5_XBLWS",
    "heightAboveGround_5_YBLWS",
)
GFS_CHANNELS: tuple[str, ...] = (
    "heightAboveGround_10_10u",
    "heightAboveGround_10_10v",
    "heightAboveGround_80_u",
    "heightAboveGround_80_v",
    "heightAboveGround_100_100u",
    "heightAboveGround_100_100v",
    "planetaryBoundaryLayer_0_u",
    "planetaryBoundaryLayer_0_v",
)
SOURCE_CHANNELS: Mapping[str, tuple[str, ...]] = {
    "ldaps": LDAPS_CHANNELS,
    "gfs": GFS_CHANNELS,
}
SOURCE_GRID_COUNTS = {"ldaps": 16, "gfs": 9}

TIME_FEATURES: tuple[str, ...] = (
    "forecast_hour_sin",
    "forecast_hour_cos",
    "forecast_dayofyear_sin",
    "forecast_dayofyear_cos",
    "forecast_month_sin",
    "forecast_month_cos",
    "ldaps_lead_hours",
    "gfs_lead_hours",
    "ldaps_run_hour_sin",
    "ldaps_run_hour_cos",
    "gfs_run_hour_sin",
    "gfs_run_hour_cos",
)
EXPECTED_FEATURE_COUNT = 212
OBJECTIVES: tuple[str, ...] = ("q07", "l1")
BLEND_WEIGHTS: tuple[float, ...] = (0.05, 0.10, 0.20)
CANDIDATE_KEYS: tuple[str, ...] = tuple(
    f"{objective}_w{int(round(weight * 100)):02d}"
    for objective in OBJECTIVES
    for weight in BLEND_WEIGHTS
)

COMMON_MODEL_PARAMETERS: Mapping[str, Any] = {
    "n_estimators": 900,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 80,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 3.0,
    "max_bin": 127,
    "random_state": 42,
    "n_jobs": 7,
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
}


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _catalog_records(frame: pd.DataFrame, source: str) -> list[dict[str, Any]]:
    coordinate_counts = frame.groupby(GRID_COL, sort=True)[list(COORD_COLS)].nunique()
    if (coordinate_counts > 1).to_numpy().any():
        raise ValueError(f"{source} grid id maps to multiple coordinates")
    catalog = (
        frame[[GRID_COL, *COORD_COLS]]
        .drop_duplicates(GRID_COL)
        .sort_values(GRID_COL)
        .reset_index(drop=True)
    )
    expected_ids = list(range(1, SOURCE_GRID_COUNTS[source] + 1))
    observed_ids = catalog[GRID_COL].astype(int).tolist()
    if observed_ids != expected_ids:
        raise ValueError(f"{source} grid ids changed: {observed_ids}")
    return [
        {
            "grid_id": int(row[GRID_COL]),
            "latitude": float(row[COORD_COLS[0]]),
            "longitude": float(row[COORD_COLS[1]]),
        }
        for _, row in catalog.iterrows()
    ]


def _assert_registered_catalog(
    observed: Sequence[Mapping[str, Any]], registered: Sequence[Mapping[str, Any]], source: str
) -> None:
    if len(observed) != len(registered):
        raise ValueError(f"{source} catalog size changed")
    for left, right in zip(observed, registered):
        if int(left["grid_id"]) != int(right["grid_id"]):
            raise ValueError(f"{source} catalog grid identity changed")
        for coordinate in COORD_COLS:
            if not np.isclose(
                float(left[coordinate]), float(right[coordinate]), rtol=0.0, atol=1e-9
            ):
                raise ValueError(f"{source} grid coordinate changed")


def _source_matrix(
    raw: pd.DataFrame,
    source: str,
    registered_catalog: Sequence[Mapping[str, Any]],
) -> tuple[pd.DataFrame, pd.Series, list[dict[str, Any]], dict[str, Any]]:
    if source not in SOURCE_CHANNELS:
        raise ValueError(f"unknown weather source: {source}")
    required = (TIME_COL, AVAILABLE_COL, GRID_COL, *COORD_COLS, *SOURCE_CHANNELS[source])
    if tuple(raw.columns) != required:
        raise ValueError(
            f"{source} must contain only the registered ordered raw columns; "
            f"observed={tuple(raw.columns)}"
        )
    frame = raw.copy()
    frame[TIME_COL] = pd.to_datetime(frame[TIME_COL], errors="raise")
    frame[AVAILABLE_COL] = pd.to_datetime(frame[AVAILABLE_COL], errors="raise")
    if frame[[TIME_COL, GRID_COL]].duplicated().any():
        raise ValueError(f"{source} contains duplicate timestamp/grid rows")
    counts = frame.groupby(TIME_COL, sort=True)[GRID_COL].size()
    if not (counts == SOURCE_GRID_COUNTS[source]).all():
        raise ValueError(f"{source} does not have every node at every timestamp")
    issuance_counts = frame.groupby(TIME_COL, sort=True)[AVAILABLE_COL].nunique()
    if not (issuance_counts == 1).all():
        raise ValueError(f"{source} timestamp maps to multiple issuance times")

    catalog = _catalog_records(frame, source)
    _assert_registered_catalog(catalog, registered_catalog, source)
    grid_ids = [record["grid_id"] for record in catalog]
    times = pd.DatetimeIndex(sorted(frame[TIME_COL].unique()), name=TIME_COL)
    blocks: list[np.ndarray] = []
    columns: list[str] = []
    for channel in SOURCE_CHANNELS[source]:
        wide = frame.pivot(index=TIME_COL, columns=GRID_COL, values=channel).reindex(
            index=times, columns=grid_ids
        )
        blocks.append(wide.to_numpy(dtype=np.float32, copy=False))
        columns.extend(f"{source}__grid_{grid_id:02d}__{channel}" for grid_id in grid_ids)
    matrix = np.concatenate(blocks, axis=1).astype(np.float32, copy=False)
    features = pd.DataFrame(matrix, index=times, columns=columns, copy=False)
    issuance = (
        frame.groupby(TIME_COL, sort=True)[AVAILABLE_COL].first().reindex(times)
    )
    metadata = {
        "source": source,
        "raw_rows": len(frame),
        "timestamp_rows": len(times),
        "rows_per_timestamp": SOURCE_GRID_COUNTS[source],
        "channels": list(SOURCE_CHANNELS[source]),
        "catalog": catalog,
        "catalog_sha256": canonical_sha256(catalog),
        "feature_columns": columns,
        "feature_columns_sha256": canonical_sha256(columns),
        "raw_missing_cells": int(np.isnan(matrix).sum()),
    }
    return features, issuance, catalog, metadata


def build_raw_grid_wind_features(
    ldaps: pd.DataFrame,
    gfs: pd.DataFrame,
    registered_catalog: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the fixed 212-column all-node wind-field matrix."""

    ldaps_features, ldaps_issuance, _, ldaps_meta = _source_matrix(
        ldaps, "ldaps", registered_catalog["ldaps"]
    )
    gfs_features, gfs_issuance, _, gfs_meta = _source_matrix(
        gfs, "gfs", registered_catalog["gfs"]
    )
    if not ldaps_features.index.equals(gfs_features.index):
        raise ValueError("LDAPS and GFS forecast timestamps are not aligned")
    times = ldaps_features.index
    hour_angle = 2.0 * np.pi * times.hour.to_numpy(dtype=float) / 24.0
    day_angle = 2.0 * np.pi * (times.dayofyear.to_numpy(dtype=float) - 1.0) / 365.25
    month_angle = 2.0 * np.pi * (times.month.to_numpy(dtype=float) - 1.0) / 12.0
    ldaps_run_angle = 2.0 * np.pi * ldaps_issuance.dt.hour.to_numpy(dtype=float) / 24.0
    gfs_run_angle = 2.0 * np.pi * gfs_issuance.dt.hour.to_numpy(dtype=float) / 24.0
    time_values = np.column_stack(
        (
            np.sin(hour_angle),
            np.cos(hour_angle),
            np.sin(day_angle),
            np.cos(day_angle),
            np.sin(month_angle),
            np.cos(month_angle),
            (times.to_series(index=times) - ldaps_issuance).dt.total_seconds().to_numpy()
            / 3600.0,
            (times.to_series(index=times) - gfs_issuance).dt.total_seconds().to_numpy()
            / 3600.0,
            np.sin(ldaps_run_angle),
            np.cos(ldaps_run_angle),
            np.sin(gfs_run_angle),
            np.cos(gfs_run_angle),
        )
    ).astype(np.float32, copy=False)
    time_frame = pd.DataFrame(time_values, index=times, columns=TIME_FEATURES)
    result = pd.concat((ldaps_features, gfs_features, time_frame), axis=1)
    if result.shape[1] != EXPECTED_FEATURE_COUNT:
        raise AssertionError(
            f"raw-grid feature count changed: {result.shape[1]} != {EXPECTED_FEATURE_COUNT}"
        )
    if result.columns.duplicated().any():
        raise AssertionError("raw-grid feature names are not unique")
    metadata = {
        "feature_count": result.shape[1],
        "ordered_columns": result.columns.tolist(),
        "ordered_columns_sha256": canonical_sha256(result.columns.tolist()),
        "index_start": result.index.min(),
        "index_end": result.index.max(),
        "index_rows": len(result),
        "sources": {"ldaps": ldaps_meta, "gfs": gfs_meta},
        "ldaps_gfs_forecast_index_equal": True,
        "only_registered_raw_wind_and_time_fields": True,
    }
    return result.astype(np.float32, copy=False), metadata


def assert_fit_before_apply(
    fit_index: pd.DatetimeIndex, application_index: pd.DatetimeIndex
) -> dict[str, Any]:
    overlap = fit_index.intersection(application_index)
    if len(fit_index) == 0 or len(application_index) == 0:
        raise ValueError("fit and application indexes must be non-empty")
    if len(overlap) or not fit_index.max() < application_index.min():
        raise ValueError("fit/application split is not strict-forward")
    return {
        "fit_start": fit_index.min(),
        "fit_end": fit_index.max(),
        "application_start": application_index.min(),
        "application_end": application_index.max(),
        "fit_end_before_application_start": True,
        "overlap_count": 0,
    }


class RawGridWindRegressor:
    """Training-only median imputation followed by one registered LGB objective."""

    def __init__(self, objective: str) -> None:
        if objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}")
        self.objective = objective

    def _make_estimator(self) -> LGBMRegressor:
        parameters = dict(COMMON_MODEL_PARAMETERS)
        if self.objective == "q07":
            parameters.update(objective="quantile", alpha=0.7)
        else:
            parameters.update(objective="regression_l1")
        return LGBMRegressor(**parameters)

    def _transform(self, features: pd.DataFrame) -> np.ndarray:
        if tuple(features.columns) != self.feature_columns_:
            raise ValueError("raw-grid feature schema/order changed")
        values = features.to_numpy(dtype=np.float32, copy=True)
        if np.isinf(values).any():
            raise ValueError("raw-grid features contain infinity")
        missing = np.isnan(values)
        if missing.any():
            rows, columns = np.where(missing)
            values[rows, columns] = self.medians_[columns]
        if not np.isfinite(values).all():
            raise ValueError("raw-grid imputation left non-finite values")
        return values

    def fit(
        self,
        features: pd.DataFrame,
        target_cf: pd.Series,
        *,
        minimum_target_cf: float = 0.10,
    ) -> "RawGridWindRegressor":
        if not features.index.equals(target_cf.index):
            raise ValueError("feature and target indexes differ")
        target = target_cf.to_numpy(dtype=float)
        eligible = np.isfinite(target) & (target >= minimum_target_cf)
        if not eligible.any():
            raise ValueError("no eligible target rows")
        self.feature_columns_ = tuple(features.columns)
        train_values = features.to_numpy(dtype=np.float32, copy=True)[eligible]
        if np.isinf(train_values).any():
            raise ValueError("fit features contain infinity")
        with np.errstate(all="ignore"):
            medians = np.nanmedian(train_values.astype(np.float64), axis=0)
        if not np.isfinite(medians).all():
            raise ValueError("a fit feature is entirely missing")
        self.medians_ = medians.astype(np.float32)
        missing = np.isnan(train_values)
        if missing.any():
            rows, columns = np.where(missing)
            train_values[rows, columns] = self.medians_[columns]
        self.estimator_ = self._make_estimator()
        self.estimator_.fit(train_values, target[eligible])
        self.fit_rows_ = int(eligible.sum())
        self.dropped_target_rows_ = int((~eligible).sum())
        self.fit_missing_cells_ = int(missing.sum())
        self.minimum_target_cf_ = float(minimum_target_cf)
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        values = self._transform(features)
        prediction = np.asarray(self.estimator_.predict(values), dtype=float)
        prediction = np.clip(prediction, 0.0, 1.02)
        if not np.isfinite(prediction).all():
            raise ValueError("raw-grid model returned non-finite predictions")
        return pd.Series(prediction, index=features.index, name=f"{self.objective}_cf")

    def metadata(self) -> dict[str, Any]:
        return {
            "objective_key": self.objective,
            "objective": "quantile" if self.objective == "q07" else "regression_l1",
            "alpha": 0.7 if self.objective == "q07" else None,
            "common_parameters": dict(COMMON_MODEL_PARAMETERS),
            "feature_count": len(self.feature_columns_),
            "feature_columns_sha256": canonical_sha256(self.feature_columns_),
            "median_sha256": hashlib.sha256(
                np.ascontiguousarray(self.medians_).tobytes()
            ).hexdigest(),
            "fit_rows": self.fit_rows_,
            "dropped_target_rows": self.dropped_target_rows_,
            "fit_missing_cells": self.fit_missing_cells_,
            "minimum_target_cf": self.minimum_target_cf_,
        }


def blend_raw_prediction(
    baseline_kwh: pd.Series,
    raw_prediction_cf: pd.Series,
    *,
    weight: float,
    capacity_kwh: float,
) -> pd.Series:
    if weight not in (0.0, *BLEND_WEIGHTS):
        raise ValueError("blend weight was not registered")
    if not baseline_kwh.index.equals(raw_prediction_cf.index):
        raise ValueError("baseline and raw prediction indexes differ")
    if weight == 0.0:
        return baseline_kwh.copy()
    values = (1.0 - weight) * baseline_kwh.to_numpy(dtype=float) + weight * (
        np.clip(raw_prediction_cf.to_numpy(dtype=float), 0.0, 1.02) * capacity_kwh
    )
    return pd.Series(
        np.clip(values, 0.0, 1.02 * capacity_kwh),
        index=baseline_kwh.index,
        name=baseline_kwh.name,
    )


def select_group_candidate(
    comparisons: Mapping[str, Mapping[str, Mapping[str, Any]]],
    required_slices: Sequence[str],
) -> tuple[str | None, dict[str, Any]]:
    """Apply the preregistered per-group all-slice candidate gate."""

    if set(comparisons) != set(CANDIDATE_KEYS) or len(comparisons) != len(CANDIDATE_KEYS):
        raise ValueError("candidate key set changed")
    expected_slices = tuple(required_slices)
    eligible: list[str] = []
    audit: dict[str, Any] = {}
    for key in CANDIDATE_KEYS:
        records = comparisons[key]
        if set(records) != set(expected_slices) or len(records) != len(expected_slices):
            raise ValueError(f"registered slice set changed for {key}")
        deltas: dict[str, float] = {}
        for name in expected_slices:
            record = records[name]
            expected_delta = float(record["candidate"]["score"]) - float(
                record["baseline"]["score"]
            )
            if float(record["delta"]) != expected_delta:
                raise ValueError("candidate delta arithmetic changed")
            deltas[name] = expected_delta
        objective, encoded_weight = key.split("_w")
        weight = int(encoded_weight) / 100.0
        row = {
            "objective": objective,
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

    def selection_key(key: str) -> tuple[float, float, int, float]:
        row = audit[key]
        return (
            float(row["minimum"]),
            float(row["mean"]),
            1 if row["objective"] == "q07" else 0,
            -float(row["weight"]),
        )

    selected = max(eligible, key=selection_key)
    return selected, {"candidates": audit, "selected": selected}


__all__ = [
    "AVAILABLE_COL",
    "BLEND_WEIGHTS",
    "CANDIDATE_KEYS",
    "COORD_COLS",
    "COMMON_MODEL_PARAMETERS",
    "EXPECTED_FEATURE_COUNT",
    "GFS_CHANNELS",
    "GRID_COL",
    "LDAPS_CHANNELS",
    "OBJECTIVES",
    "RawGridWindRegressor",
    "SOURCE_CHANNELS",
    "SOURCE_GRID_COUNTS",
    "TIME_COL",
    "TIME_FEATURES",
    "assert_fit_before_apply",
    "blend_raw_prediction",
    "build_raw_grid_wind_features",
    "canonical_sha256",
    "select_group_candidate",
]

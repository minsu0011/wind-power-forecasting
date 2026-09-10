"""Leakage-safe reproduction of DACON's official BARAM RF baseline.

The raw weather aggregation, calendar formulas, median imputer, and explicit
RandomForest settings match official code-share 14031.  Only the temporal
training protocol and target eligibility are adapted: every imputer/model sees
an earlier fit interval only, and the forest learns capacity factor from rows
eligible under the current competition metric.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

from src.features import AVAILABLE_COL, COORD_COLS, GRID_COL, SOURCE_VARIABLES, TIME_COL


CALENDAR_COLUMNS: tuple[str, ...] = (
    "month",
    "day",
    "hour",
    "dayofweek",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
)

LDAPS_MEAN_COLUMNS: tuple[str, ...] = tuple(
    f"ldaps_{column}_mean" for column in SOURCE_VARIABLES["ldaps"]
)
GFS_MEAN_COLUMNS: tuple[str, ...] = tuple(
    f"gfs_{column}_mean" for column in SOURCE_VARIABLES["gfs"]
)
FEATURE_COLUMNS: tuple[str, ...] = (
    *CALENDAR_COLUMNS,
    *LDAPS_MEAN_COLUMNS,
    *GFS_MEAN_COLUMNS,
)

STAGE1_REQUIRED: dict[str, tuple[str, ...]] = {
    "kpx_group_1": ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4"),
    "kpx_group_2": ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4"),
    "kpx_group_3": ("full", "Q3", "Q4"),
}
STAGE2_REQUIRED: tuple[str, ...] = (
    "full",
    "H1",
    "H2",
    "Q1",
    "Q2",
    "Q3",
    "Q4",
)

EXPLICIT_RF_PARAMETERS: dict[str, Any] = {
    "n_estimators": 120,
    "max_depth": 14,
    "min_samples_leaf": 8,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1,
}
RELEVANT_DEFAULT_RF_PARAMETERS: dict[str, Any] = {
    "criterion": "squared_error",
    "min_samples_split": 2,
    "bootstrap": True,
    "oob_score": False,
    "max_samples": None,
    "max_leaf_nodes": None,
    "min_impurity_decrease": 0.0,
    "ccp_alpha": 0.0,
    "warm_start": False,
}


def official_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Return the official notebook's nine calendar columns exactly."""

    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("index must be DatetimeIndex")
    if len(index) == 0 or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("index must be nonempty, unique, and increasing")
    normalized = pd.DatetimeIndex(index, name="forecast_kst_dtm")
    month = normalized.month.to_numpy(dtype=np.float64)
    day = normalized.day.to_numpy(dtype=np.float64)
    hour = normalized.hour.to_numpy(dtype=np.float64)
    dayofweek = normalized.dayofweek.to_numpy(dtype=np.float64)
    output = pd.DataFrame(index=normalized)
    output["month"] = month
    output["day"] = day
    output["hour"] = hour
    output["dayofweek"] = dayofweek
    output["is_weekend"] = np.isin(dayofweek, (5.0, 6.0)).astype(np.float64)
    output["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    output["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    output["month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
    output["month_cos"] = np.cos(2.0 * np.pi * month / 12.0)
    if tuple(output.columns) != CALENDAR_COLUMNS:
        raise AssertionError("official calendar schema changed")
    return output


def _aggregate_weather(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    if source not in SOURCE_VARIABLES:
        raise KeyError(f"unknown source: {source}")
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("weather must be a DataFrame")
    required = {
        TIME_COL,
        AVAILABLE_COL,
        GRID_COL,
        *COORD_COLS,
        *SOURCE_VARIABLES[source],
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"{source} weather missing columns: {missing}")
    local = frame.loc[:, [TIME_COL, *SOURCE_VARIABLES[source]]].copy()
    local[TIME_COL] = pd.to_datetime(local[TIME_COL], errors="raise")
    values = local.loc[:, list(SOURCE_VARIABLES[source])].to_numpy(
        dtype=np.float64, copy=False
    )
    if np.isinf(values).any():
        raise ValueError(f"{source} raw weather contains infinite values")
    # This is intentionally the literal notebook operation: arithmetic mean
    # across all spatial grids for each forecast timestamp, in raw column order.
    aggregated = local.groupby(TIME_COL)[list(SOURCE_VARIABLES[source])].mean()
    aggregated.index = pd.DatetimeIndex(aggregated.index, name="forecast_kst_dtm")
    aggregated.columns = [f"{source}_{column}_mean" for column in aggregated.columns]
    expected = LDAPS_MEAN_COLUMNS if source == "ldaps" else GFS_MEAN_COLUMNS
    if tuple(aggregated.columns) != expected:
        raise AssertionError(f"{source} mean feature schema changed")
    if not aggregated.index.is_unique or not aggregated.index.is_monotonic_increasing:
        raise AssertionError(f"{source} aggregate index changed")
    return aggregated.astype(np.float64)


def build_official_rf_features(
    ldaps: pd.DataFrame, gfs: pd.DataFrame
) -> pd.DataFrame:
    """Build the exact 74-column official mean-weather/calendar design."""

    ldaps_mean = _aggregate_weather(ldaps, "ldaps")
    gfs_mean = _aggregate_weather(gfs, "gfs")
    weather = ldaps_mean.join(gfs_mean, how="inner")
    if len(weather) != len(ldaps_mean) or len(weather) != len(gfs_mean):
        raise ValueError("LDAPS/GFS forecast timestamp intersection changed")
    calendar = official_calendar_features(pd.DatetimeIndex(weather.index))
    output = pd.concat((calendar, weather), axis=1)
    output.index = pd.DatetimeIndex(output.index, name="forecast_kst_dtm")
    if tuple(output.columns) != FEATURE_COLUMNS or output.shape[1] != 74:
        raise AssertionError("official RF feature schema changed")
    values = output.to_numpy(dtype=np.float64, copy=False)
    if np.isinf(values).any():
        raise ValueError("official RF features contain infinite values")
    forbidden = ("actual", "target", "scada", "baseline", "residual", "public", "scale")
    if any(token in column.lower() for column in output.columns for token in forbidden):
        raise AssertionError("forbidden feature channel entered official RF design")
    return output


@dataclass
class OfficialRandomForestBaseline:
    """Exact official RF/imputer with fixed metric-aligned target handling."""

    n_estimators: int = 120
    max_depth: int = 14
    min_samples_leaf: int = 8
    max_features: str = "sqrt"
    random_state: int = 42
    n_jobs: int = -1
    minimum_actual_cf: float = 0.10

    def __post_init__(self) -> None:
        observed = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "minimum_actual_cf": self.minimum_actual_cf,
        }
        expected = {**EXPLICIT_RF_PARAMETERS, "minimum_actual_cf": 0.10}
        if observed != expected:
            raise ValueError("official RF parameters differ from preregistration")
        self.imputer_: SimpleImputer | None = None
        self.estimator_: RandomForestRegressor | None = None
        self.fit_metadata_: dict[str, Any] | None = None

    def fit(
        self,
        features: pd.DataFrame,
        actual_kwh: pd.Series,
        *,
        capacity_kwh: float,
    ) -> "OfficialRandomForestBaseline":
        self._validate_features(features)
        if not isinstance(actual_kwh, pd.Series) or not features.index.equals(actual_kwh.index):
            raise ValueError("feature and target indices differ")
        capacity = float(capacity_kwh)
        if not np.isfinite(capacity) or capacity <= 0.0:
            raise ValueError("capacity must be positive and finite")
        capacity_factor = actual_kwh.to_numpy(dtype=np.float64, copy=False) / capacity
        eligible = np.isfinite(capacity_factor) & (capacity_factor >= self.minimum_actual_cf)
        if int(eligible.sum()) < 2:
            raise ValueError("too few eligible direct-CF training rows")

        # Match notebook ordering: fit the unsupervised imputer on the complete
        # chronological training feature interval, then apply the target mask.
        imputer = SimpleImputer(strategy="median")
        transformed = np.asarray(imputer.fit_transform(features), dtype=np.float64)
        if transformed.shape != (len(features), len(FEATURE_COLUMNS)):
            raise ValueError("median imputer dropped an all-missing feature")
        if not np.isfinite(transformed).all() or not np.isfinite(imputer.statistics_).all():
            raise ValueError("median imputer produced non-finite values")

        estimator = RandomForestRegressor(**EXPLICIT_RF_PARAMETERS)
        estimator.fit(transformed[eligible], capacity_factor[eligible])
        resolved = estimator.get_params(deep=False)
        if any(resolved[key] != value for key, value in EXPLICIT_RF_PARAMETERS.items()):
            raise AssertionError("explicit RF parameters changed")
        if any(
            resolved[key] != value
            for key, value in RELEVANT_DEFAULT_RF_PARAMETERS.items()
        ):
            raise AssertionError("relevant RF defaults changed")
        statistics = np.ascontiguousarray(imputer.statistics_, dtype=np.float64)
        self.imputer_ = imputer
        self.estimator_ = estimator
        self.fit_metadata_ = {
            "rows_total": int(len(features)),
            "rows_eligible": int(eligible.sum()),
            "feature_count": len(FEATURE_COLUMNS),
            "feature_columns": list(FEATURE_COLUMNS),
            "target_kind": "direct_capacity_factor",
            "minimum_actual_cf": self.minimum_actual_cf,
            "imputer_class": type(imputer).__name__,
            "imputer_strategy": imputer.strategy,
            "imputer_fit_rows": int(len(features)),
            "imputer_statistics_sha256": hashlib.sha256(statistics.tobytes()).hexdigest(),
            "model_class": type(estimator).__name__,
            "explicit_parameters": {
                key: resolved[key] for key in EXPLICIT_RF_PARAMETERS
            },
            "relevant_default_parameters": {
                key: resolved[key] for key in RELEVANT_DEFAULT_RF_PARAMETERS
            },
        }
        return self

    def predict_cf(self, features: pd.DataFrame) -> pd.Series:
        self._validate_features(features)
        if self.imputer_ is None or self.estimator_ is None:
            raise RuntimeError("model is not fitted")
        transformed = np.asarray(self.imputer_.transform(features), dtype=np.float64)
        if transformed.shape != (len(features), len(FEATURE_COLUMNS)):
            raise ValueError("median imputer output schema changed")
        if not np.isfinite(transformed).all():
            raise ValueError("median imputer produced non-finite application values")
        # Parallel forest prediction can change the final ulp with thread
        # completion order.  Preserve n_jobs=-1 in the fitted model but serialize
        # inference so model reload/prediction locks are truly bit exact.
        original_n_jobs = self.estimator_.n_jobs
        try:
            self.estimator_.n_jobs = 1
            predicted = np.asarray(self.estimator_.predict(transformed), dtype=np.float64)
        finally:
            self.estimator_.n_jobs = original_n_jobs
        if predicted.shape != (len(features),) or not np.isfinite(predicted).all():
            raise ValueError("official RF produced invalid predictions")
        return pd.Series(
            np.clip(predicted, 0.0, 1.0),
            index=features.index,
            name="direct_cf",
        )

    @staticmethod
    def _validate_features(features: pd.DataFrame) -> None:
        if not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a DataFrame")
        if tuple(features.columns) != FEATURE_COLUMNS:
            raise ValueError("feature schema/order differs from preregistration")
        if not isinstance(features.index, pd.DatetimeIndex):
            raise TypeError("feature index must be DatetimeIndex")
        if not features.index.is_unique or not features.index.is_monotonic_increasing:
            raise ValueError("feature index must be unique and increasing")
        values = features.to_numpy(dtype=np.float64, copy=False)
        if np.isinf(values).any():
            raise ValueError("features contain infinite values")


def assert_strict_forward(
    fit_index: pd.DatetimeIndex, application_index: pd.DatetimeIndex
) -> None:
    if len(fit_index) == 0 or len(application_index) == 0:
        raise ValueError("fit and application indices must be nonempty")
    if not fit_index.is_unique or not application_index.is_unique:
        raise ValueError("fit and application indices must be unique")
    if not fit_index.is_monotonic_increasing or not application_index.is_monotonic_increasing:
        raise ValueError("fit and application indices must be increasing")
    if fit_index.max() >= application_index.min():
        raise ValueError("fit interval must end before application begins")
    if len(fit_index.intersection(application_index)):
        raise ValueError("fit and application indices overlap")


def blend_direct_kwh(
    baseline_kwh: pd.Series,
    direct_cf: pd.Series,
    *,
    weight: float,
    capacity_kwh: float,
) -> pd.Series:
    if not baseline_kwh.index.equals(direct_cf.index):
        raise ValueError("baseline/direct indices differ")
    fixed_weight = float(weight)
    if fixed_weight not in (0.0, 0.05, 0.10, 1.0):
        raise ValueError("blend weight differs from preregistration")
    baseline = baseline_kwh.to_numpy(dtype=np.float64, copy=False)
    direct = direct_cf.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(baseline).all() or not np.isfinite(direct).all():
        raise ValueError("blend inputs contain non-finite values")
    capacity = float(capacity_kwh)
    if fixed_weight == 0.0:
        values = baseline.copy()
    else:
        values = np.clip(
            (1.0 - fixed_weight) * baseline + fixed_weight * direct * capacity,
            0.0,
            capacity,
        )
    return pd.Series(values, index=baseline_kwh.index, name=baseline_kwh.name)


def select_stage1_weight(
    comparisons: Mapping[str, Any],
    *,
    weights: Mapping[str, float],
) -> tuple[float | None, dict[str, Any]]:
    if tuple(comparisons) != tuple(weights):
        raise ValueError("candidate key/order differs from preregistration")
    audit: dict[str, Any] = {}
    eligible: list[str] = []
    for key, weight in weights.items():
        if float(weight) not in (0.05, 0.10):
            raise ValueError("candidate weight differs from preregistration")
        deltas: dict[str, float] = {}
        for group, required in STAGE1_REQUIRED.items():
            if set(comparisons[key][group]) != set(required):
                raise ValueError(f"{key}/{group} registered segments changed")
            for segment in required:
                record = comparisons[key][group][segment]
                delta = float(record["candidate"]["score"]) - float(
                    record["baseline"]["score"]
                )
                if float(record["delta"]) != delta:
                    raise ValueError("delta arithmetic changed")
                deltas[f"{group}/{segment}"] = delta
        if len(deltas) != 17:
            raise AssertionError("Stage1 slice count changed")
        audit[key] = {
            "weight": float(weight),
            "deltas": deltas,
            "minimum": min(deltas.values()),
            "mean": float(np.mean(list(deltas.values()))),
            "all_17_strictly_positive": all(delta > 0.0 for delta in deltas.values()),
        }
        if audit[key]["all_17_strictly_positive"]:
            eligible.append(key)
    if not eligible:
        return None, {"candidates": audit, "selected": "identity"}
    selected = max(
        eligible,
        key=lambda item: (
            audit[item]["minimum"],
            audit[item]["mean"],
            -audit[item]["weight"],
        ),
    )
    return float(weights[selected]), {"candidates": audit, "selected": selected}


def stage2_promoted(
    group_comparisons: Mapping[str, Mapping[str, Mapping[str, Any]]],
    mixed_comparisons: Mapping[str, Mapping[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    group_deltas: dict[str, float] = {}
    for group in ("kpx_group_1", "kpx_group_2", "kpx_group_3"):
        if set(group_comparisons[group]) != set(STAGE2_REQUIRED):
            raise ValueError(f"{group} Stage2 segments changed")
        for segment in STAGE2_REQUIRED:
            record = group_comparisons[group][segment]
            delta = float(record["candidate"]["score"]) - float(
                record["baseline"]["score"]
            )
            if float(record["delta"]) != delta:
                raise ValueError("Stage2 group delta arithmetic changed")
            group_deltas[f"{group}/{segment}"] = delta
    if set(mixed_comparisons) != set(STAGE2_REQUIRED):
        raise ValueError("Stage2 mixed segments changed")
    mixed_deltas: dict[str, float] = {}
    for segment in STAGE2_REQUIRED:
        record = mixed_comparisons[segment]
        delta = float(record["candidate"]["total_score"]) - float(
            record["baseline"]["total_score"]
        )
        if float(record["delta_total_score"]) != delta:
            raise ValueError("Stage2 mixed delta arithmetic changed")
        mixed_deltas[segment] = delta
    promoted = all(delta > 0.0 for delta in group_deltas.values()) and all(
        delta > 0.0 for delta in mixed_deltas.values()
    )
    return promoted, {
        "group_slice_count": len(group_deltas),
        "mixed_slice_count": len(mixed_deltas),
        "group_deltas": group_deltas,
        "mixed_total_deltas": mixed_deltas,
        "minimum_group_delta": min(group_deltas.values()),
        "minimum_mixed_total_delta": min(mixed_deltas.values()),
        "all_21_group_strictly_positive": all(delta > 0.0 for delta in group_deltas.values()),
        "all_7_mixed_strictly_positive": all(delta > 0.0 for delta in mixed_deltas.values()),
    }


__all__ = [
    "CALENDAR_COLUMNS",
    "LDAPS_MEAN_COLUMNS",
    "GFS_MEAN_COLUMNS",
    "FEATURE_COLUMNS",
    "STAGE1_REQUIRED",
    "STAGE2_REQUIRED",
    "EXPLICIT_RF_PARAMETERS",
    "RELEVANT_DEFAULT_RF_PARAMETERS",
    "OfficialRandomForestBaseline",
    "assert_strict_forward",
    "blend_direct_kwh",
    "build_official_rf_features",
    "official_calendar_features",
    "select_stage1_weight",
    "stage2_promoted",
]

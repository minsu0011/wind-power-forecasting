"""Strict compact-physics squared-error ExtraTrees candidate.

The estimator consumes only the 36 weather/time columns frozen in the
preregistration.  It learns direct capacity factor from an earlier eligible
label prefix; baseline predictions are used only after inference for the two
fixed, low-weight blends.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor


FEATURE_COLUMNS: tuple[str, ...] = (
    "ldaps__idw__ws10",
    "ldaps__idw__ws10__flow_sin",
    "ldaps__idw__ws10__flow_cos",
    "ldaps__idw__ws50",
    "ldaps__idw__ws50__flow_sin",
    "ldaps__idw__ws50__flow_cos",
    "ldaps__idw__wind_range50",
    "ldaps__idw__shear_alpha_10_50",
    "ldaps__idw__hub_ws",
    "ldaps__idw__air_density",
    "ldaps__idw__wind_power_density",
    "gfs__idw__ws10",
    "gfs__idw__ws10__flow_sin",
    "gfs__idw__ws10__flow_cos",
    "gfs__idw__ws80",
    "gfs__idw__ws100",
    "gfs__idw__ws100__flow_sin",
    "gfs__idw__ws100__flow_cos",
    "gfs__idw__shear_alpha_10_100",
    "gfs__idw__hub_ws",
    "gfs__idw__air_density",
    "gfs__idw__wind_power_density",
    "gfs__idw__gust_excess",
    "cross__hub_ws_mean",
    "cross__hub_ws_difference",
    "cross__ws10_difference",
    "cross__u10_difference",
    "cross__v10_difference",
    "cross__hub_ws_product",
    "time__hour_sin",
    "time__hour_cos",
    "time__doy_sin",
    "time__doy_cos",
    "time__month_sin",
    "time__month_cos",
    "time__lead_hours",
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


def build_compact_physics_features(weather: pd.DataFrame) -> pd.DataFrame:
    """Select the exact immutable 36-column design without fitted transforms."""

    if not isinstance(weather, pd.DataFrame):
        raise TypeError("weather must be a DataFrame")
    if not isinstance(weather.index, pd.DatetimeIndex):
        raise TypeError("weather index must be DatetimeIndex")
    if len(weather) == 0 or not weather.index.is_unique:
        raise ValueError("weather index must be nonempty and unique")
    if not weather.index.is_monotonic_increasing:
        raise ValueError("weather index must be increasing")
    missing = [column for column in FEATURE_COLUMNS if column not in weather.columns]
    if missing:
        raise KeyError(f"weather is missing preregistered columns: {missing}")
    output = weather.loc[:, FEATURE_COLUMNS].astype(np.float64).copy()
    output.index = pd.DatetimeIndex(output.index, name="forecast_kst_dtm")
    if tuple(output.columns) != FEATURE_COLUMNS or output.shape[1] != 36:
        raise AssertionError("compact physical feature schema changed")
    if not np.isfinite(output.to_numpy(dtype=np.float64, copy=False)).all():
        raise ValueError("compact physical features contain non-finite values")
    forbidden_tokens = ("actual", "target", "scada", "baseline", "residual", "public", "scale")
    lowered = tuple(column.lower() for column in output.columns)
    if any(token in column for token in forbidden_tokens for column in lowered):
        raise AssertionError("forbidden feature channel entered compact design")
    return output


@dataclass
class CompactPhysicsExtraTrees:
    """The exact fixed legacy squared-error ExtraTrees estimator."""

    n_estimators: int = 400
    criterion: str = "squared_error"
    max_depth: int = 30
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: float = 1.0
    bootstrap: bool = False
    oob_score: bool = False
    max_samples: int | float | None = None
    random_state: int = 42
    n_jobs: int = 7
    minimum_actual_cf: float = 0.10

    def __post_init__(self) -> None:
        observed = {
            "n_estimators": self.n_estimators,
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "bootstrap": self.bootstrap,
            "oob_score": self.oob_score,
            "max_samples": self.max_samples,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "minimum_actual_cf": self.minimum_actual_cf,
        }
        expected = {
            "n_estimators": 400,
            "criterion": "squared_error",
            "max_depth": 30,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": 1.0,
            "bootstrap": False,
            "oob_score": False,
            "max_samples": None,
            "random_state": 42,
            "n_jobs": 7,
            "minimum_actual_cf": 0.10,
        }
        if observed != expected:
            raise ValueError("ExtraTrees parameters differ from preregistration")
        self.estimator_: ExtraTreesRegressor | None = None
        self.fit_metadata_: dict[str, Any] | None = None

    def fit(
        self,
        features: pd.DataFrame,
        actual_kwh: pd.Series,
        *,
        capacity_kwh: float,
    ) -> "CompactPhysicsExtraTrees":
        self._validate_features(features)
        if not isinstance(actual_kwh, pd.Series) or not features.index.equals(actual_kwh.index):
            raise ValueError("feature and target indices differ")
        capacity = float(capacity_kwh)
        if not np.isfinite(capacity) or capacity <= 0.0:
            raise ValueError("capacity must be positive and finite")
        actual = actual_kwh.to_numpy(dtype=np.float64, copy=False)
        capacity_factor = actual / capacity
        eligible = np.isfinite(capacity_factor) & (capacity_factor >= self.minimum_actual_cf)
        if int(eligible.sum()) < len(FEATURE_COLUMNS) + 1:
            raise ValueError("too few eligible direct-CF training rows")
        estimator = ExtraTreesRegressor(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        estimator.fit(features.loc[eligible, FEATURE_COLUMNS], capacity_factor[eligible])
        self.estimator_ = estimator
        self.fit_metadata_ = {
            "rows_total": int(len(features)),
            "rows_eligible": int(eligible.sum()),
            "feature_count": len(FEATURE_COLUMNS),
            "feature_columns": list(FEATURE_COLUMNS),
            "target_kind": "direct_capacity_factor",
            "minimum_actual_cf": self.minimum_actual_cf,
            "model_class": type(estimator).__name__,
            "resolved_parameters": {
                key: estimator.get_params(deep=False)[key]
                for key in (
                    "n_estimators",
                    "criterion",
                    "max_depth",
                    "min_samples_split",
                    "min_samples_leaf",
                    "max_features",
                    "bootstrap",
                    "oob_score",
                    "max_samples",
                    "random_state",
                    "n_jobs",
                )
            },
        }
        return self

    def predict_cf(self, features: pd.DataFrame) -> pd.Series:
        self._validate_features(features)
        if self.estimator_ is None:
            raise RuntimeError("model is not fitted")
        # sklearn's parallel forest prediction accumulates tree outputs under a
        # lock; thread completion order can change the final ulp after a model
        # reload.  Keep the preregistered fitted estimator parameter at n_jobs=7,
        # but serialize inference so the pre-score prediction/reload lock is
        # genuinely value-bit exact.
        original_n_jobs = self.estimator_.n_jobs
        try:
            self.estimator_.n_jobs = 1
            predicted = np.asarray(self.estimator_.predict(features), dtype=np.float64)
        finally:
            self.estimator_.n_jobs = original_n_jobs
        if predicted.shape != (len(features),) or not np.isfinite(predicted).all():
            raise ValueError("ExtraTrees produced invalid predictions")
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
        if not np.isfinite(features.to_numpy(dtype=np.float64, copy=False)).all():
            raise ValueError("features contain non-finite values")


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
    "FEATURE_COLUMNS",
    "STAGE1_REQUIRED",
    "STAGE2_REQUIRED",
    "CompactPhysicsExtraTrees",
    "assert_strict_forward",
    "blend_direct_kwh",
    "build_compact_physics_features",
    "select_stage1_weight",
    "stage2_promoted",
]

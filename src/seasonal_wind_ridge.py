"""Low-degree deterministic seasonal/wind Ridge model.

The design deliberately accepts one weather channel and forecast timestamps.
It cannot consume target, baseline, residual, SCADA, Public, or application-label
features.  All nonlinearities are fixed before fitting; only a standardized
Ridge response surface is learned from an earlier label prefix.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS: tuple[str, ...] = (
    "wind_z",
    "wind_z2",
    "wind_hinge2_03",
    "wind_hinge2_06",
    "wind_hinge2_09",
    "wind_hinge2_12",
    "wind_hinge2_15",
    "doy_sin1",
    "doy_cos1",
    "doy_sin2",
    "doy_cos2",
    "hour_sin1",
    "hour_cos1",
    "hour_sin2",
    "hour_cos2",
    "wind_x_doy_sin1",
    "wind_x_doy_cos1",
    "wind_x_doy_sin2",
    "wind_x_doy_cos2",
    "wind_x_hour_sin1",
    "wind_x_hour_cos1",
    "wind_x_hour_sin2",
    "wind_x_hour_cos2",
)

WIND_KNOTS_MS: tuple[float, ...] = (3.0, 6.0, 9.0, 12.0, 15.0)
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


def build_seasonal_wind_features(
    index: pd.DatetimeIndex,
    consensus_hub_wind_ms: object,
    *,
    clip_ms: tuple[float, float] = (0.0, 30.0),
    scale_ms: float = 15.0,
    knots_ms: Sequence[float] = WIND_KNOTS_MS,
) -> pd.DataFrame:
    """Build the exact 23-column fixed low-degree design matrix."""

    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("index must be a DatetimeIndex")
    if len(index) == 0 or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("index must be nonempty, unique, and increasing")
    wind = np.asarray(consensus_hub_wind_ms, dtype=np.float64)
    if wind.shape != (len(index),) or not np.isfinite(wind).all():
        raise ValueError("consensus wind must be a finite one-dimensional index match")
    lower, upper = map(float, clip_ms)
    if not np.isfinite([lower, upper, scale_ms]).all() or not lower < upper:
        raise ValueError("invalid wind clip or scale")
    if scale_ms <= 0:
        raise ValueError("scale_ms must be positive")
    knots = tuple(map(float, knots_ms))
    if knots != WIND_KNOTS_MS:
        raise ValueError("quadratic hinge knots differ from preregistration")

    wind_z = np.clip(wind, lower, upper) / float(scale_ms)
    hour = index.hour.to_numpy(dtype=np.float64)
    fractional_day = index.dayofyear.to_numpy(dtype=np.float64) - 1.0 + hour / 24.0
    day_angle = 2.0 * np.pi * fractional_day / 365.2425
    hour_angle = 2.0 * np.pi * hour / 24.0
    cycles = {
        "doy_sin1": np.sin(day_angle),
        "doy_cos1": np.cos(day_angle),
        "doy_sin2": np.sin(2.0 * day_angle),
        "doy_cos2": np.cos(2.0 * day_angle),
        "hour_sin1": np.sin(hour_angle),
        "hour_cos1": np.cos(hour_angle),
        "hour_sin2": np.sin(2.0 * hour_angle),
        "hour_cos2": np.cos(2.0 * hour_angle),
    }
    data: dict[str, np.ndarray] = {
        "wind_z": wind_z,
        "wind_z2": np.square(wind_z),
    }
    for knot in knots:
        data[f"wind_hinge2_{int(knot):02d}"] = np.square(
            np.maximum(wind_z - knot / float(scale_ms), 0.0)
        )
    data.update(cycles)
    for name, values in cycles.items():
        data[f"wind_x_{name}"] = wind_z * values
    output = pd.DataFrame(data, index=index, columns=FEATURE_COLUMNS, dtype=np.float64)
    output.index.name = "forecast_kst_dtm"
    if tuple(output.columns) != FEATURE_COLUMNS or output.shape[1] != 23:
        raise AssertionError("seasonal wind feature schema changed")
    if not np.isfinite(output.to_numpy(dtype=np.float64, copy=False)).all():
        raise AssertionError("seasonal wind features contain non-finite values")
    return output


@dataclass
class SeasonalWindRidge:
    """One fixed StandardScaler/Ridge direct capacity-factor estimator."""

    alpha: float = 250.0
    solver: str = "cholesky"
    tol: float = 1e-8
    minimum_actual_cf: float = 0.10

    def __post_init__(self) -> None:
        if self.alpha != 250.0 or self.solver != "cholesky" or self.tol != 1e-8:
            raise ValueError("model parameters differ from preregistration")
        if self.minimum_actual_cf != 0.10:
            raise ValueError("eligibility threshold differs from preregistration")
        self.pipeline_: Pipeline | None = None
        self.fit_metadata_: dict[str, Any] | None = None

    def fit(
        self,
        features: pd.DataFrame,
        actual_kwh: pd.Series,
        *,
        capacity_kwh: float,
    ) -> "SeasonalWindRidge":
        self._validate_features(features)
        if not features.index.equals(actual_kwh.index):
            raise ValueError("feature and target indices differ")
        capacity = float(capacity_kwh)
        if not np.isfinite(capacity) or capacity <= 0:
            raise ValueError("capacity must be positive and finite")
        actual = actual_kwh.to_numpy(dtype=np.float64, copy=False)
        cf = actual / capacity
        eligible = np.isfinite(cf) & (cf >= self.minimum_actual_cf)
        if int(eligible.sum()) < len(FEATURE_COLUMNS) + 1:
            raise ValueError("too few eligible direct-CF training rows")
        model = Pipeline(
            [
                ("scale", StandardScaler(with_mean=True, with_std=True)),
                (
                    "ridge",
                    Ridge(
                        alpha=self.alpha,
                        fit_intercept=True,
                        solver=self.solver,
                        tol=self.tol,
                    ),
                ),
            ]
        )
        model.fit(features.loc[eligible, FEATURE_COLUMNS], cf[eligible])
        self.pipeline_ = model
        self.fit_metadata_ = {
            "rows_total": int(len(features)),
            "rows_eligible": int(eligible.sum()),
            "feature_count": len(FEATURE_COLUMNS),
            "feature_columns": list(FEATURE_COLUMNS),
            "target_kind": "direct_capacity_factor",
            "minimum_actual_cf": self.minimum_actual_cf,
            "alpha": self.alpha,
            "solver": self.solver,
            "tol": self.tol,
        }
        return self

    def predict_cf(self, features: pd.DataFrame) -> pd.Series:
        self._validate_features(features)
        if self.pipeline_ is None:
            raise RuntimeError("model is not fitted")
        predicted = np.asarray(
            self.pipeline_.predict(features.loc[:, FEATURE_COLUMNS]), dtype=np.float64
        )
        if predicted.shape != (len(features),) or not np.isfinite(predicted).all():
            raise ValueError("Ridge produced invalid predictions")
        return pd.Series(
            np.clip(predicted, 0.0, 1.02),
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
        raise ValueError("fit/application indices must not be empty")
    if fit_index.intersection(application_index).size:
        raise ValueError("fit/application overlap")
    if not fit_index.max() < application_index.min():
        raise ValueError("fit must end strictly before application")


def blend_direct_kwh(
    baseline_kwh: pd.Series,
    direct_cf: pd.Series,
    *,
    weight: float,
    capacity_kwh: float,
) -> pd.Series:
    if not baseline_kwh.index.equals(direct_cf.index):
        raise ValueError("baseline/direct indices differ")
    weight_float = float(weight)
    if weight_float not in (0.0, 0.025, 0.05):
        raise ValueError("blend weight differs from preregistration")
    baseline = baseline_kwh.to_numpy(dtype=np.float64, copy=False)
    direct = direct_cf.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(baseline).all() or not np.isfinite(direct).all():
        raise ValueError("blend inputs contain non-finite values")
    if weight_float == 0.0:
        values = baseline.copy()
    else:
        values = np.clip(
            (1.0 - weight_float) * baseline
            + weight_float * direct * float(capacity_kwh),
            0.0,
            1.02 * float(capacity_kwh),
        )
    return pd.Series(values, index=baseline_kwh.index, name=baseline_kwh.name)


def select_stage1_weight(
    comparisons: Mapping[str, Any],
    *,
    weights: Mapping[str, float],
) -> tuple[float | None, dict[str, Any]]:
    """Apply the exact one-global-weight/17-strict-positive preregistered gate."""

    if tuple(comparisons) != tuple(weights):
        raise ValueError("candidate key/order differs from preregistration")
    audit: dict[str, Any] = {}
    eligible: list[str] = []
    for key, weight in weights.items():
        if float(weight) not in (0.025, 0.05):
            raise ValueError("candidate weight differs from preregistration")
        deltas: dict[str, float] = {}
        for group, required in STAGE1_REQUIRED.items():
            if set(comparisons[key][group]) != set(required):
                raise ValueError(f"{key}/{group} registered segments changed")
            for segment in required:
                record = comparisons[key][group][segment]
                expected = float(record["candidate"]["score"]) - float(
                    record["baseline"]["score"]
                )
                if float(record["delta"]) != expected:
                    raise ValueError("delta arithmetic changed")
                deltas[f"{group}/{segment}"] = expected
        if len(deltas) != 17:
            raise AssertionError("Stage1 slice count changed")
        audit[key] = {
            "weight": float(weight),
            "deltas": deltas,
            "minimum": min(deltas.values()),
            "mean": float(np.mean(list(deltas.values()))),
            "all_17_strictly_positive": all(value > 0.0 for value in deltas.values()),
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
    """Require 21 group and seven mixed official-score deltas all positive."""

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
    mixed_deltas: dict[str, float] = {}
    if set(mixed_comparisons) != set(STAGE2_REQUIRED):
        raise ValueError("Stage2 mixed segments changed")
    for segment in STAGE2_REQUIRED:
        record = mixed_comparisons[segment]
        delta = float(record["candidate"]["total_score"]) - float(
            record["baseline"]["total_score"]
        )
        if float(record["delta_total_score"]) != delta:
            raise ValueError("Stage2 mixed delta arithmetic changed")
        mixed_deltas[segment] = delta
    promoted = all(value > 0.0 for value in group_deltas.values()) and all(
        value > 0.0 for value in mixed_deltas.values()
    )
    return promoted, {
        "group_slice_count": len(group_deltas),
        "mixed_slice_count": len(mixed_deltas),
        "group_deltas": group_deltas,
        "mixed_total_deltas": mixed_deltas,
        "minimum_group_delta": min(group_deltas.values()),
        "minimum_mixed_total_delta": min(mixed_deltas.values()),
        "all_21_group_strictly_positive": all(
            value > 0.0 for value in group_deltas.values()
        ),
        "all_7_mixed_strictly_positive": all(
            value > 0.0 for value in mixed_deltas.values()
        ),
    }


__all__ = [
    "FEATURE_COLUMNS",
    "STAGE1_REQUIRED",
    "STAGE2_REQUIRED",
    "WIND_KNOTS_MS",
    "SeasonalWindRidge",
    "assert_strict_forward",
    "blend_direct_kwh",
    "build_seasonal_wind_features",
    "select_stage1_weight",
    "stage2_promoted",
]

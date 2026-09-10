"""Strict-forward residual histogram classifier and exact Bayes action.

The classifier estimates a 43-class conditional distribution of the
capacity-factor residual above a fixed OOF baseline.  A point correction is
then selected by maximizing the class-probability expectation of the exact
action-dependent part of the official group score.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd


COMPONENT_NAMES: tuple[str, ...] = (
    "lgb_l1",
    "lgb_q07",
    "shared_l1",
    "shared_q07",
    "top200_q07",
    "energy_q06",
)
WEATHER_COLUMNS: tuple[str, ...] = (
    "cross__hub_ws_mean",
    "ldaps__idw__hub_ws",
    "gfs__idw__hub_ws",
    "ldaps__idw__wind_power_density",
    "gfs__idw__wind_power_density",
    "gfs__idw__surface_0_gust",
)
META_FEATURE_COLUMNS: tuple[str, ...] = (
    "base_cf",
    "lgb_l1_cf",
    "lgb_q07_cf",
    "shared_l1_cf",
    "shared_q07_cf",
    "top200_q07_cf",
    "energy_q06_cf",
    "component_mean_cf",
    "component_std_cf",
    "component_min_cf",
    "component_max_cf",
    "component_range_cf",
    "lgb_q07_minus_lgb_l1_cf",
    "shared_q07_minus_shared_l1_cf",
    "cross_hub_ws_mean",
    "ldaps_hub_ws",
    "gfs_hub_ws",
    "ldaps_wind_power_density",
    "gfs_wind_power_density",
    "gfs_surface_gust",
    "lead_fraction",
    "lead_sin",
    "lead_cos",
    "day_of_year_sin",
    "day_of_year_cos",
)
RESIDUAL_CENTRES_CF = np.asarray(
    [-0.525 + 0.025 * index for index in range(43)], dtype=np.float64
)
ACTION_DELTAS_CF = np.asarray(
    [-0.150 + 0.005 * index for index in range(61)], dtype=np.float64
)


def _finite_series(value: pd.Series, *, name: str) -> pd.Series:
    if not isinstance(value, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    if value.empty or not value.index.is_unique or not value.index.is_monotonic_increasing:
        raise ValueError(f"{name} must use a nonempty, unique, increasing index")
    result = value.astype(float)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def _finite_frame(value: pd.DataFrame, *, name: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if value.empty or not value.index.is_unique or not value.index.is_monotonic_increasing:
        raise ValueError(f"{name} must use a nonempty, unique, increasing index")
    if not value.columns.is_unique:
        raise ValueError(f"{name} columns must be unique")
    result = value.astype(float)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def residual_class_index(residual_cf: Any) -> np.ndarray:
    """Assign residuals to nearest fixed centre; endpoint classes absorb tails."""

    residual = np.asarray(residual_cf, dtype=np.float64)
    if not np.isfinite(residual).all():
        raise ValueError("residual_cf must be finite")
    coordinate = (residual - float(RESIDUAL_CENTRES_CF[0])) / 0.025
    # ceil(x - 1/2) implements nearest-centre assignment with an exact
    # midpoint going to the lower centre.  The tiny tolerance neutralises
    # binary floating-point noise at decimal half-step boundaries.
    selected = np.ceil(coordinate - 0.5 - 1e-12)
    return np.clip(selected, 0, len(RESIDUAL_CENTRES_CF) - 1).astype(np.int16)


def build_meta_features(
    base_prediction_kwh: pd.Series,
    component_predictions_kwh: Mapping[str, pd.Series],
    weather: pd.DataFrame,
    *,
    capacity_kwh: float,
) -> pd.DataFrame:
    """Build the exact preregistered 25 safe meta features."""

    base = _finite_series(base_prediction_kwh, name="base_prediction_kwh")
    weather_frame = _finite_frame(weather, name="weather")
    if not base.index.equals(weather_frame.index):
        raise ValueError("weather index differs from baseline index")
    if tuple(component_predictions_kwh) != COMPONENT_NAMES:
        raise ValueError("component prediction order differs from preregistration")
    if not np.isfinite(capacity_kwh) or capacity_kwh <= 0:
        raise ValueError("capacity_kwh must be positive and finite")
    missing_weather = set(WEATHER_COLUMNS).difference(weather_frame.columns)
    if missing_weather:
        raise KeyError(f"missing preregistered weather columns: {sorted(missing_weather)}")
    forbidden = [
        str(column)
        for column in weather_frame.columns
        if any(token in str(column).lower() for token in ("actual", "target", "label", "scada", "power_kw"))
    ]
    if forbidden:
        raise ValueError(f"forbidden target/SCADA-like weather columns: {forbidden[:5]}")

    components: list[np.ndarray] = []
    for name in COMPONENT_NAMES:
        component = _finite_series(
            component_predictions_kwh[name], name=f"component_{name}"
        )
        if not component.index.equals(base.index):
            raise ValueError(f"component {name} index differs from baseline")
        components.append(component.to_numpy(dtype=np.float64) / capacity_kwh)
    matrix = np.column_stack(components)
    base_cf = base.to_numpy(dtype=np.float64) / capacity_kwh
    index = pd.DatetimeIndex(base.index)
    lead_position = np.mod(index.hour.to_numpy(dtype=np.int16) - 1, 24).astype(float)
    lead_angle = 2.0 * np.pi * lead_position / 24.0
    day_angle = (
        2.0
        * np.pi
        * (index.dayofyear.to_numpy(dtype=np.int16).astype(float) - 1.0)
        / 365.2425
    )
    selected_weather = weather_frame.loc[:, list(WEATHER_COLUMNS)].to_numpy(
        dtype=np.float64, copy=False
    )
    output = pd.DataFrame(
        {
            "base_cf": base_cf,
            **{
                f"{name}_cf": matrix[:, position]
                for position, name in enumerate(COMPONENT_NAMES)
            },
            "component_mean_cf": matrix.mean(axis=1),
            "component_std_cf": matrix.std(axis=1, ddof=0),
            "component_min_cf": matrix.min(axis=1),
            "component_max_cf": matrix.max(axis=1),
            "component_range_cf": matrix.max(axis=1) - matrix.min(axis=1),
            "lgb_q07_minus_lgb_l1_cf": matrix[:, 1] - matrix[:, 0],
            "shared_q07_minus_shared_l1_cf": matrix[:, 3] - matrix[:, 2],
            "cross_hub_ws_mean": selected_weather[:, 0],
            "ldaps_hub_ws": selected_weather[:, 1],
            "gfs_hub_ws": selected_weather[:, 2],
            "ldaps_wind_power_density": selected_weather[:, 3],
            "gfs_wind_power_density": selected_weather[:, 4],
            "gfs_surface_gust": selected_weather[:, 5],
            "lead_fraction": lead_position / 23.0,
            "lead_sin": np.sin(lead_angle),
            "lead_cos": np.cos(lead_angle),
            "day_of_year_sin": np.sin(day_angle),
            "day_of_year_cos": np.cos(day_angle),
        },
        index=base.index,
        dtype=np.float64,
    )
    if tuple(output.columns) != META_FEATURE_COLUMNS:
        raise AssertionError("meta feature order differs from preregistration")
    if not np.isfinite(output.to_numpy()).all():
        raise AssertionError("meta features contain non-finite values")
    output.index.name = "forecast_kst_dtm"
    return output


def expected_utility_actions(
    probabilities: Any,
    base_cf: Any,
    *,
    mean_actual_cf: float,
    chunk_rows: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Maximize the exact boundary-aware expected official action utility."""

    probability = np.asarray(probabilities, dtype=np.float64)
    base = np.asarray(base_cf, dtype=np.float64)
    if probability.ndim != 2 or probability.shape[1] != len(RESIDUAL_CENTRES_CF):
        raise ValueError("probabilities must have 43 fixed residual classes")
    if base.shape != (len(probability),):
        raise ValueError("base_cf and probabilities are not row-aligned")
    if not np.isfinite(probability).all() or not np.isfinite(base).all():
        raise ValueError("Bayes action inputs must be finite")
    if np.any(probability < -1e-12):
        raise ValueError("class probabilities must be nonnegative")
    row_sum = probability.sum(axis=1)
    if not np.allclose(row_sum, 1.0, atol=1e-9, rtol=0.0):
        raise ValueError("class probabilities must sum to one")
    mean = float(mean_actual_cf)
    if not np.isfinite(mean) or mean < 0.10:
        raise ValueError("mean_actual_cf must be finite and at least 0.10")
    if not isinstance(chunk_rows, int) or chunk_rows <= 0:
        raise ValueError("chunk_rows must be a positive integer")

    selected_actions = np.empty(len(base), dtype=np.float64)
    selected_utility = np.empty(len(base), dtype=np.float64)
    for start in range(0, len(base), chunk_rows):
        stop = min(start + chunk_rows, len(base))
        local_base = base[start:stop]
        candidates = np.clip(
            local_base[:, None] + ACTION_DELTAS_CF[None, :], 0.0, 1.02
        )
        outcomes = np.clip(
            local_base[:, None] + RESIDUAL_CENTRES_CF[None, :], 0.10, 1.20
        )
        error = np.abs(candidates[:, :, None] - outcomes[:, None, :])
        settlement = np.select(
            [error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0
        )
        utility = -error + outcomes[:, None, :] * settlement / (4.0 * mean)
        expectation = np.einsum(
            "rac,rc->ra", utility, probability[start:stop], optimize=True
        )
        for row in range(len(local_base)):
            maximum = float(expectation[row].max())
            tied = np.flatnonzero(expectation[row] >= maximum - 1e-12)
            tied_actions = ACTION_DELTAS_CF[tied]
            minimum_abs = float(np.abs(tied_actions).min())
            nearest = tied[np.abs(tied_actions) <= minimum_abs + 1e-15]
            chosen = int(nearest[0])
            selected_actions[start + row] = ACTION_DELTAS_CF[chosen]
            selected_utility[start + row] = expectation[row, chosen]
    return selected_actions, selected_utility


class ResidualHistogramBayesClassifier:
    """Fixed LightGBM residual histogram followed by official-utility action."""

    def __init__(
        self,
        *,
        model_parameters: Mapping[str, Any],
        minimum_eligible_fit_rows: int = 300,
    ) -> None:
        self.model_parameters = dict(model_parameters)
        self.minimum_eligible_fit_rows = int(minimum_eligible_fit_rows)

    def fit(
        self,
        features: pd.DataFrame,
        actual_kwh: pd.Series,
        base_prediction_kwh: pd.Series,
        *,
        capacity_kwh: float,
    ) -> "ResidualHistogramBayesClassifier":
        feature_frame = _finite_frame(features, name="fit_features")
        if tuple(feature_frame.columns) != META_FEATURE_COLUMNS:
            raise ValueError("fit feature schema differs from preregistration")
        actual = actual_kwh.astype(float)
        if not isinstance(actual, pd.Series) or not actual.index.equals(feature_frame.index):
            raise ValueError("fit actual index differs from features")
        if np.isinf(actual.to_numpy()).any():
            raise ValueError("fit actual contains infinite values")
        base = _finite_series(base_prediction_kwh, name="fit_base_prediction_kwh")
        if not base.index.equals(feature_frame.index):
            raise ValueError("fit baseline index differs from features")
        capacity = float(capacity_kwh)
        if not np.isfinite(capacity) or capacity <= 0:
            raise ValueError("capacity_kwh must be positive and finite")
        actual_cf = actual.to_numpy(dtype=np.float64) / capacity
        base_cf = base.to_numpy(dtype=np.float64) / capacity
        eligible = np.isfinite(actual_cf) & (actual_cf >= 0.10)
        if int(eligible.sum()) < self.minimum_eligible_fit_rows:
            raise ValueError(
                f"too few eligible rows: {int(eligible.sum())} < {self.minimum_eligible_fit_rows}"
            )
        residual = actual_cf[eligible] - base_cf[eligible]
        classes = residual_class_index(residual)
        if np.unique(classes).size < 3:
            raise ValueError("multiclass residual fit requires at least three classes")
        model = LGBMClassifier(**self.model_parameters)
        model.fit(feature_frame.to_numpy(dtype=np.float64)[eligible], classes)
        observed_classes = np.asarray(model.classes_, dtype=int)
        if (
            observed_classes.ndim != 1
            or np.any(observed_classes < 0)
            or np.any(observed_classes >= len(RESIDUAL_CENTRES_CF))
        ):
            raise AssertionError("LightGBM returned invalid residual classes")
        self.model_ = model
        self.capacity_kwh_ = capacity
        self.fit_index_ = feature_frame.index.copy()
        self.fit_rows_ = int(len(feature_frame))
        self.eligible_fit_rows_ = int(eligible.sum())
        self.mean_actual_cf_ = float(actual_cf[eligible].mean())
        self.observed_classes_ = observed_classes
        self.class_counts_ = np.bincount(classes, minlength=43).astype(int)
        self.residual_min_cf_ = float(residual.min())
        self.residual_max_cf_ = float(residual.max())
        return self

    def predict_raw_action(
        self,
        features: pd.DataFrame,
        base_prediction_kwh: pd.Series,
    ) -> tuple[pd.Series, pd.Series, np.ndarray]:
        if not hasattr(self, "model_"):
            raise RuntimeError("fit must be called before predict")
        feature_frame = _finite_frame(features, name="application_features")
        if tuple(feature_frame.columns) != META_FEATURE_COLUMNS:
            raise ValueError("application feature schema differs from preregistration")
        base = _finite_series(
            base_prediction_kwh, name="application_base_prediction_kwh"
        )
        if not base.index.equals(feature_frame.index):
            raise ValueError("application baseline index differs from features")
        overlap = feature_frame.index.intersection(self.fit_index_)
        if len(overlap):
            raise ValueError("application rows overlap fitted rows")
        compact_probability = np.asarray(
            self.model_.predict_proba(feature_frame.to_numpy(dtype=np.float64)),
            dtype=np.float64,
        )
        if compact_probability.shape != (
            len(feature_frame),
            len(self.observed_classes_),
        ):
            raise AssertionError("LightGBM probability shape differs from classes_")
        probability = np.zeros((len(feature_frame), 43), dtype=np.float64)
        probability[:, self.observed_classes_] = compact_probability
        row_sum = probability.sum(axis=1)
        if not np.allclose(row_sum, 1.0, atol=1e-9, rtol=0.0):
            raise AssertionError("mapped class probabilities do not sum to one")
        base_cf = base.to_numpy(dtype=np.float64) / self.capacity_kwh_
        action, utility = expected_utility_actions(
            probability, base_cf, mean_actual_cf=self.mean_actual_cf_
        )
        return (
            pd.Series(action, index=feature_frame.index, name="raw_action_delta_cf"),
            pd.Series(utility, index=feature_frame.index, name="expected_utility"),
            probability,
        )

    def metadata(self) -> dict[str, Any]:
        if not hasattr(self, "model_"):
            raise RuntimeError("fit must be called before metadata")
        return {
            "fit_rows": self.fit_rows_,
            "eligible_fit_rows": self.eligible_fit_rows_,
            "fit_start": pd.Timestamp(self.fit_index_.min()).isoformat(),
            "fit_end": pd.Timestamp(self.fit_index_.max()).isoformat(),
            "mean_actual_cf": self.mean_actual_cf_,
            "observed_classes": self.observed_classes_.tolist(),
            "class_counts": self.class_counts_.tolist(),
            "residual_min_cf": self.residual_min_cf_,
            "residual_max_cf": self.residual_max_cf_,
            "feature_columns": list(META_FEATURE_COLUMNS),
            "model_parameters": dict(self.model_parameters),
            "same_row_fit_score_computed": False,
        }


def apply_action_shrink(
    base_prediction_kwh: pd.Series,
    raw_action_delta_cf: pd.Series,
    *,
    capacity_kwh: float,
    shrink: float,
) -> pd.Series:
    """Apply a preregistered action shrink and physical output bound."""

    base = _finite_series(base_prediction_kwh, name="base_prediction_kwh")
    action = _finite_series(raw_action_delta_cf, name="raw_action_delta_cf")
    if not base.index.equals(action.index):
        raise ValueError("base and action indexes differ")
    value = float(shrink)
    if value not in (0.25, 0.5, 1.0):
        raise ValueError("shrink is not preregistered")
    base_cf = base.to_numpy(dtype=np.float64) / capacity_kwh
    candidate = np.clip(
        base_cf + value * action.to_numpy(dtype=np.float64), 0.0, 1.02
    )
    return pd.Series(
        candidate * capacity_kwh, index=base.index, name=base.name
    )


def choose_group_shrinks(
    comparisons: Mapping[str, Mapping[str, Mapping[str, Mapping[str, float]]]],
    groups: Sequence[str],
) -> dict[str, float | None]:
    """Select an all-segments-positive shrink independently for each group."""

    selected: dict[str, float | None] = {}
    for group in groups:
        eligible: list[tuple[float, float, float]] = []
        for shrink_name, by_group in comparisons.items():
            slices = by_group.get(str(group), {})
            if not slices:
                raise ValueError(f"{shrink_name} has no comparisons for {group}")
            deltas = np.asarray(
                [float(item["delta"]) for item in slices.values()], dtype=float
            )
            if not np.isfinite(deltas).all():
                raise ValueError("comparison deltas must be finite")
            shrink = float(shrink_name)
            if np.all(deltas > 0.0):
                eligible.append((float(deltas.min()), float(deltas.mean()), shrink))
        eligible.sort(key=lambda item: (-item[0], -item[1], item[2]))
        selected[str(group)] = eligible[0][2] if eligible else None
    return selected


__all__ = [
    "ACTION_DELTAS_CF",
    "COMPONENT_NAMES",
    "META_FEATURE_COLUMNS",
    "RESIDUAL_CENTRES_CF",
    "ResidualHistogramBayesClassifier",
    "WEATHER_COLUMNS",
    "apply_action_shrink",
    "build_meta_features",
    "choose_group_shrinks",
    "expected_utility_actions",
    "residual_class_index",
]

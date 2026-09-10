"""Cutoff-safe JMA GSM paired incremental-delta model primitives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd


CANDIDATES: tuple[str, ...] = (
    "A_day2_w025",
    "B_day1_hours01_13_else_day2_w025",
)
JMA_COLUMNS: tuple[str, ...] = (
    "jma__ws10_ms",
    "jma__u10_ms",
    "jma__v10_ms",
    "jma__ws10_minus_cross_hub_ws_mean",
)
CONTROL_DISAGREEMENT_COLUMN = "cross__hub_ws_mean"
TRANSFER_WEIGHT = 0.25
MODEL_PARAMETERS: dict[str, Any] = {
    "objective": "regression_l1",
    "n_estimators": 1500,
    "learning_rate": 0.025,
    "num_leaves": 31,
    "min_child_samples": 30,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.75,
    "reg_alpha": 0.05,
    "reg_lambda": 2.0,
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
    "random_state": 42,
    "n_jobs": 7,
}


def _validate_index(index: pd.DatetimeIndex) -> None:
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("index must be a DatetimeIndex")
    if index.empty or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("index must be nonempty, unique, and increasing")


def build_jma_features(
    source: pd.DataFrame,
    *,
    group: str,
    index: pd.DatetimeIndex,
    candidate: str,
    cross_hub_ws_mean: pd.Series,
) -> pd.DataFrame:
    """Build the four immutable JMA features for one group/candidate."""

    _validate_index(index)
    if candidate not in CANDIDATES:
        raise ValueError(f"unknown candidate: {candidate}")
    required = {
        "group",
        "time",
        "wind_speed_10m_previous_day1",
        "wind_direction_10m_previous_day1",
        "wind_speed_10m_previous_day2",
        "wind_direction_10m_previous_day2",
    }
    if set(source.columns) != required:
        raise ValueError("JMA source schema differs")
    selected = source.loc[source["group"] == group].copy()
    selected.index = pd.DatetimeIndex(selected.pop("time"), name="forecast_kst_dtm")
    if not selected.index.is_unique:
        raise ValueError(f"duplicate JMA timestamps: {group}")
    selected = selected.reindex(index)
    if selected.isna().any().any():
        missing = selected.index[selected.isna().any(axis=1)]
        raise ValueError(f"JMA coverage differs for {group}: {missing[:3].tolist()}")
    if not cross_hub_ws_mean.index.equals(index):
        raise ValueError("cross-wind series index differs")

    if candidate == CANDIDATES[0]:
        speed = selected["wind_speed_10m_previous_day2"].to_numpy(np.float64)
        direction = selected["wind_direction_10m_previous_day2"].to_numpy(np.float64)
    else:
        use_day1 = (index.hour >= 1) & (index.hour <= 13)
        speed = np.where(
            use_day1,
            selected["wind_speed_10m_previous_day1"].to_numpy(np.float64),
            selected["wind_speed_10m_previous_day2"].to_numpy(np.float64),
        )
        direction = np.where(
            use_day1,
            selected["wind_direction_10m_previous_day1"].to_numpy(np.float64),
            selected["wind_direction_10m_previous_day2"].to_numpy(np.float64),
        )
    radians = np.deg2rad(direction)
    u10 = -speed * np.sin(radians)
    v10 = -speed * np.cos(radians)
    cross = cross_hub_ws_mean.to_numpy(np.float64)
    result = pd.DataFrame(
        {
            JMA_COLUMNS[0]: speed,
            JMA_COLUMNS[1]: u10,
            JMA_COLUMNS[2]: v10,
            JMA_COLUMNS[3]: speed - cross,
        },
        index=index,
    ).astype(np.float32)
    if tuple(result.columns) != JMA_COLUMNS or not np.isfinite(result.to_numpy()).all():
        raise AssertionError("JMA feature contract differs")
    return result


def extended_features(control: pd.DataFrame, jma: pd.DataFrame) -> pd.DataFrame:
    if not control.index.equals(jma.index):
        raise ValueError("control/JMA indices differ")
    if CONTROL_DISAGREEMENT_COLUMN not in control.columns:
        raise KeyError(CONTROL_DISAGREEMENT_COLUMN)
    if tuple(jma.columns) != JMA_COLUMNS:
        raise ValueError("JMA schema/order differs")
    if set(control.columns).intersection(JMA_COLUMNS):
        raise ValueError("JMA columns collide with control features")
    result = pd.concat([control, jma], axis=1).astype(np.float32)
    if tuple(result.columns[-len(JMA_COLUMNS) :]) != JMA_COLUMNS:
        raise AssertionError("extended feature order differs")
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("extended features contain non-finite values")
    return result


def make_model() -> LGBMRegressor:
    return LGBMRegressor(**MODEL_PARAMETERS)


def eligible_rows(actual_kwh: pd.Series, capacity_kwh: float) -> np.ndarray:
    values = actual_kwh.to_numpy(np.float64)
    return np.isfinite(values) & (values >= 0.10 * float(capacity_kwh))


def fit_direct_model(
    features: pd.DataFrame,
    actual_kwh: pd.Series,
    *,
    capacity_kwh: float,
    eligible: np.ndarray | None = None,
) -> tuple[LGBMRegressor, dict[str, Any]]:
    if not features.index.equals(actual_kwh.index):
        raise ValueError("features/labels indices differ")
    mask = eligible_rows(actual_kwh, capacity_kwh) if eligible is None else np.asarray(eligible, bool)
    if mask.shape != (len(features),) or not mask.any():
        raise ValueError("eligible mask differs or is empty")
    target = actual_kwh.to_numpy(np.float64)[mask] / float(capacity_kwh)
    model = make_model()
    model.fit(features.iloc[np.flatnonzero(mask)], target)
    return model, {
        "rows_total": int(len(features)),
        "rows_eligible": int(mask.sum()),
        "feature_count": int(features.shape[1]),
        "target_cf_mean": float(np.mean(target)),
        "parameters": dict(MODEL_PARAMETERS),
    }


def predict_cf(model: LGBMRegressor, features: pd.DataFrame) -> np.ndarray:
    prediction = np.asarray(model.predict(features), dtype=np.float64)
    if prediction.shape != (len(features),) or not np.isfinite(prediction).all():
        raise ValueError("direct model prediction differs")
    return np.clip(prediction, 0.0, 1.02)


def paired_increment(control_cf: Any, extended_cf: Any) -> np.ndarray:
    control = np.clip(np.asarray(control_cf, dtype=np.float64), 0.0, 1.02)
    extended = np.clip(np.asarray(extended_cf, dtype=np.float64), 0.0, 1.02)
    if control.shape != extended.shape or control.ndim != 1:
        raise ValueError("paired predictions must be aligned vectors")
    if not np.isfinite(control).all() or not np.isfinite(extended).all():
        raise ValueError("paired predictions contain non-finite values")
    return extended - control


def transfer_increment(
    baseline_kwh: pd.Series,
    increment_cf: Any,
    *,
    capacity_kwh: float,
) -> pd.Series:
    increment = np.asarray(increment_cf, dtype=np.float64)
    if increment.shape != (len(baseline_kwh),) or not np.isfinite(increment).all():
        raise ValueError("increment shape/values differ")
    values = np.clip(
        baseline_kwh.to_numpy(np.float64)
        + TRANSFER_WEIGHT * float(capacity_kwh) * increment,
        0.0,
        1.02 * float(capacity_kwh),
    )
    return pd.Series(values, index=baseline_kwh.index, name=baseline_kwh.name)


def select_stage1_candidate(
    delta_vectors: Mapping[str, Sequence[float]],
) -> tuple[str | None, dict[str, Any]]:
    if tuple(delta_vectors) != CANDIDATES:
        raise ValueError("candidate order differs")
    records: dict[str, Any] = {}
    passing: list[tuple[float, float, int, str]] = []
    for order, candidate in enumerate(CANDIDATES):
        values = np.asarray(delta_vectors[candidate], dtype=np.float64)
        if values.shape != (17,) or not np.isfinite(values).all():
            raise ValueError(f"{candidate}: expected 17 finite deltas")
        minimum = float(values.min())
        mean = float(values.mean())
        passed = bool(np.all(values > 0.0))
        records[candidate] = {"minimum_delta": minimum, "mean_delta": mean, "passed": passed}
        if passed:
            passing.append((minimum, mean, -order, candidate))
    winner = max(passing)[3] if passing else None
    return winner, {"candidates": records, "winner": winner, "tie_break": "minimum_then_mean_then_A"}


__all__: Sequence[str] = (
    "CANDIDATES",
    "CONTROL_DISAGREEMENT_COLUMN",
    "JMA_COLUMNS",
    "MODEL_PARAMETERS",
    "TRANSFER_WEIGHT",
    "build_jma_features",
    "eligible_rows",
    "extended_features",
    "fit_direct_model",
    "make_model",
    "paired_increment",
    "predict_cf",
    "select_stage1_candidate",
    "transfer_increment",
)

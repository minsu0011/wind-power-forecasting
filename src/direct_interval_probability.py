"""Action-conditional interval probabilities for the exact BARAM thresholds.

The model expands every eligible weather row across a frozen absolute
capacity-factor action grid.  Two binary heads estimate the energy-weighted
probability of landing inside the official 6% and 8% error intervals, while a
third head estimates row-weighted absolute error.  The point action is chosen
from those three quantities; no validation or Public result enters the action.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from lightgbm import LGBMClassifier, LGBMRegressor
import numpy as np
import pandas as pd


CONTEXT_COLUMNS: tuple[str, ...] = (
    "ldaps__idw__hub_ws",
    "ldaps__idw__wind_power_density",
    "ldaps__idw__air_density",
    "ldaps__idw__shear_alpha_10_50",
    "ldaps__idw__ws50__flow_sin",
    "ldaps__idw__ws50__flow_cos",
    "ldaps__near4_std__heightAboveGround_50_50MUmax",
    "ldaps__near4_std__heightAboveGround_50_50MVmax",
    "gfs__idw__hub_ws",
    "gfs__idw__wind_power_density",
    "gfs__idw__air_density",
    "gfs__idw__shear_alpha_10_100",
    "gfs__idw__surface_0_gust",
    "gfs__idw__gust_excess",
    "gfs__idw__ws100__flow_sin",
    "gfs__idw__ws100__flow_cos",
    "gfs__near4_std__heightAboveGround_100_100u",
    "gfs__near4_std__heightAboveGround_100_100v",
    "cross__hub_ws_mean",
    "cross__hub_ws_difference",
    "cross__hub_ws_product",
    "time__hour_sin",
    "time__hour_cos",
    "time__doy_sin",
    "time__doy_cos",
    "time__run_position",
)
MODEL_COLUMNS: tuple[str, ...] = CONTEXT_COLUMNS + ("action_cf",)
ACTION_GRID_CF = np.arange(103, dtype=np.float64) / 100.0
BLEND_WEIGHTS: dict[str, float] = {"w025": 0.025, "w05": 0.05, "w10": 0.10}
STAGE1_REQUIRED: dict[str, tuple[str, ...]] = {
    "kpx_group_1": ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4"),
    "kpx_group_2": ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4"),
    "kpx_group_3": ("full", "Q3", "Q4"),
}

COMMON_PARAMETERS: dict[str, Any] = {
    "boosting_type": "gbdt",
    "n_estimators": 140,
    "learning_rate": 0.04,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 400,
    "min_split_gain": 0.0,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "reg_alpha": 1.0,
    "reg_lambda": 12.0,
    "max_bin": 63,
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
    "n_jobs": 7,
}


def compact_context(weather: pd.DataFrame) -> pd.DataFrame:
    """Select the exact immutable compact weather schema without fitting."""

    if not isinstance(weather, pd.DataFrame):
        raise TypeError("weather must be a DataFrame")
    if not isinstance(weather.index, pd.DatetimeIndex):
        raise TypeError("weather index must be DatetimeIndex")
    if weather.empty or not weather.index.is_unique or not weather.index.is_monotonic_increasing:
        raise ValueError("weather index must be nonempty, unique, and increasing")
    missing = set(CONTEXT_COLUMNS).difference(weather.columns)
    if missing:
        raise KeyError(f"missing compact context columns: {sorted(missing)}")
    forbidden = [
        str(column)
        for column in weather.columns
        if any(
            token in str(column).lower()
            for token in ("actual", "target", "label", "scada", "power_kw", "public")
        )
    ]
    if forbidden:
        raise ValueError(f"forbidden target/SCADA/Public-like columns: {forbidden[:5]}")
    result = weather.loc[:, list(CONTEXT_COLUMNS)].astype(np.float32)
    if tuple(result.columns) != CONTEXT_COLUMNS:
        raise AssertionError("compact context order changed")
    if not np.isfinite(result.to_numpy(dtype=np.float64, copy=False)).all():
        raise ValueError("compact context contains non-finite values")
    result.index.name = "forecast_kst_dtm"
    return result


def _validate_context(context: pd.DataFrame) -> None:
    if not isinstance(context, pd.DataFrame):
        raise TypeError("context must be a DataFrame")
    if tuple(context.columns) != CONTEXT_COLUMNS:
        raise ValueError("context schema/order differs from preregistration")
    if not isinstance(context.index, pd.DatetimeIndex):
        raise TypeError("context index must be DatetimeIndex")
    if context.empty or not context.index.is_unique or not context.index.is_monotonic_increasing:
        raise ValueError("context index must be nonempty, unique, and increasing")
    if not np.isfinite(context.to_numpy(dtype=np.float64, copy=False)).all():
        raise ValueError("context contains non-finite values")


def expanded_action_design(context: pd.DataFrame) -> np.ndarray:
    """Return row-major [timestamp, increasing action] float32 design."""

    _validate_context(context)
    base = context.to_numpy(dtype=np.float32, copy=False)
    rows = len(context)
    actions = len(ACTION_GRID_CF)
    expanded = np.empty((rows * actions, len(MODEL_COLUMNS)), dtype=np.float32)
    expanded[:, :-1] = np.repeat(base, actions, axis=0)
    expanded[:, -1] = np.tile(ACTION_GRID_CF.astype(np.float32), rows)
    if not np.isfinite(expanded).all():
        raise AssertionError("expanded action design contains non-finite values")
    return expanded


def repair_interval_probabilities(
    p6_raw: Any, p8_raw: Any
) -> tuple[np.ndarray, np.ndarray, int]:
    """Project each violating (p6,p8) pair onto p6<=p8 in Euclidean norm."""

    p6 = np.asarray(p6_raw, dtype=np.float64)
    p8 = np.asarray(p8_raw, dtype=np.float64)
    if p6.shape != p8.shape or p6.ndim != 2:
        raise ValueError("p6 and p8 must be aligned two-dimensional arrays")
    if not np.isfinite(p6).all() or not np.isfinite(p8).all():
        raise ValueError("interval probabilities must be finite")
    p6 = np.clip(p6, 0.0, 1.0)
    p8 = np.clip(p8, 0.0, 1.0)
    violation = p6 > p8
    midpoint = 0.5 * (p6 + p8)
    repaired6 = np.where(violation, midpoint, p6)
    repaired8 = np.where(violation, midpoint, p8)
    if np.any(repaired6 > repaired8 + 1e-15):
        raise AssertionError("probability monotonic repair failed")
    return repaired6, repaired8, int(violation.sum())


def official_action_utility(p6: Any, p8: Any, eae: Any) -> np.ndarray:
    """Expected official utility up to the action-independent +0.5 constant."""

    p6_array = np.asarray(p6, dtype=np.float64)
    p8_array = np.asarray(p8, dtype=np.float64)
    error = np.asarray(eae, dtype=np.float64)
    if p6_array.shape != p8_array.shape or p6_array.shape != error.shape:
        raise ValueError("utility surfaces must be aligned")
    if np.any(p6_array > p8_array + 1e-15):
        raise ValueError("utility requires p6<=p8")
    if not np.isfinite(error).all():
        raise ValueError("EAE must be finite")
    return -0.5 * np.clip(error, 0.0, 1.20) + 0.5 * (
        0.25 * p6_array + 0.75 * p8_array
    )


def select_actions(
    utility: Any, baseline_cf: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the exact utility/nearest-baseline/lower-action tie break."""

    values = np.asarray(utility, dtype=np.float64)
    baseline = np.asarray(baseline_cf, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(ACTION_GRID_CF):
        raise ValueError("utility must have one column per fixed action")
    if baseline.shape != (values.shape[0],):
        raise ValueError("baseline and utility row counts differ")
    if not np.isfinite(values).all() or not np.isfinite(baseline).all():
        raise ValueError("action selection inputs must be finite")
    maximum = values.max(axis=1)
    tied = values >= maximum[:, None] - 1e-12
    distance = np.abs(ACTION_GRID_CF[None, :] - baseline[:, None])
    nearest_distance = np.where(tied, distance, np.inf).min(axis=1)
    nearest = tied & (distance <= nearest_distance[:, None] + 1e-15)
    index = np.argmax(nearest, axis=1).astype(np.int16)
    row = np.arange(len(index))
    return ACTION_GRID_CF[index], values[row, index], index


class DirectIntervalProbabilityModel:
    """Three fixed LightGBM heads on an action-expanded compact design."""

    def __init__(self) -> None:
        self.p6_model_: LGBMClassifier | None = None
        self.p8_model_: LGBMClassifier | None = None
        self.eae_model_: LGBMRegressor | None = None
        self.fit_metadata_: dict[str, Any] | None = None

    @staticmethod
    def _classifier(seed: int) -> LGBMClassifier:
        return LGBMClassifier(objective="binary", random_state=seed, **COMMON_PARAMETERS)

    @staticmethod
    def _regressor(seed: int) -> LGBMRegressor:
        return LGBMRegressor(
            objective="regression_l2", random_state=seed, **COMMON_PARAMETERS
        )

    def fit(
        self,
        context: pd.DataFrame,
        actual_kwh: pd.Series,
        *,
        capacity_kwh: float,
    ) -> "DirectIntervalProbabilityModel":
        _validate_context(context)
        if not isinstance(actual_kwh, pd.Series) or not context.index.equals(actual_kwh.index):
            raise ValueError("context and target indices differ")
        capacity = float(capacity_kwh)
        if not np.isfinite(capacity) or capacity <= 0:
            raise ValueError("capacity_kwh must be positive and finite")
        actual_cf = actual_kwh.to_numpy(dtype=np.float64, copy=False) / capacity
        eligible = np.isfinite(actual_cf) & (actual_cf >= 0.10)
        if int(eligible.sum()) < 100:
            raise ValueError("too few eligible rows")
        eligible_context = context.loc[eligible]
        eligible_actual = actual_cf[eligible]
        design = expanded_action_design(eligible_context)
        repeated_actual = np.repeat(eligible_actual, len(ACTION_GRID_CF))
        tiled_action = np.tile(ACTION_GRID_CF, len(eligible_actual))
        absolute_error = np.abs(tiled_action - repeated_actual)
        target6 = (absolute_error <= 0.06).astype(np.uint8)
        target8 = (absolute_error <= 0.08).astype(np.uint8)
        energy_weight = repeated_actual.astype(np.float64, copy=False)
        if np.unique(target6).size != 2 or np.unique(target8).size != 2:
            raise ValueError("binary interval target lost a class")

        p6 = self._classifier(4206)
        p8 = self._classifier(4208)
        eae = self._regressor(4210)
        p6.fit(design, target6, sample_weight=energy_weight)
        p8.fit(design, target8, sample_weight=energy_weight)
        eae.fit(design, absolute_error)
        self.p6_model_ = p6
        self.p8_model_ = p8
        self.eae_model_ = eae
        self.fit_metadata_ = {
            "rows_total": int(len(context)),
            "rows_eligible": int(eligible.sum()),
            "expanded_rows": int(len(design)),
            "context_feature_count": len(CONTEXT_COLUMNS),
            "expanded_feature_count": len(MODEL_COLUMNS),
            "action_count": len(ACTION_GRID_CF),
            "eligible_actual_cf_mean": float(np.mean(eligible_actual)),
            "p6_positive_fraction_unweighted": float(np.mean(target6)),
            "p8_positive_fraction_unweighted": float(np.mean(target8)),
            "p6_seed": 4206,
            "p8_seed": 4208,
            "eae_seed": 4210,
            "parameters": dict(COMMON_PARAMETERS),
            "fit_score_calculated": False,
            "application_labels_used": False,
        }
        return self

    def predict_surfaces(self, context: pd.DataFrame) -> dict[str, np.ndarray | int]:
        _validate_context(context)
        if self.p6_model_ is None or self.p8_model_ is None or self.eae_model_ is None:
            raise RuntimeError("model is not fitted")
        design = expanded_action_design(context)
        shape = (len(context), len(ACTION_GRID_CF))
        raw6 = np.asarray(self.p6_model_.predict_proba(design)[:, 1], dtype=np.float64).reshape(shape)
        raw8 = np.asarray(self.p8_model_.predict_proba(design)[:, 1], dtype=np.float64).reshape(shape)
        eae = np.asarray(self.eae_model_.predict(design), dtype=np.float64).reshape(shape)
        if not np.isfinite(raw6).all() or not np.isfinite(raw8).all() or not np.isfinite(eae).all():
            raise ValueError("model produced non-finite surface")
        repaired6, repaired8, violations = repair_interval_probabilities(raw6, raw8)
        return {
            "p6_raw": raw6,
            "p8_raw": raw8,
            "p6": repaired6,
            "p8": repaired8,
            "eae": np.clip(eae, 0.0, 1.20),
            "repair_count": violations,
        }

    def predict_action(
        self,
        context: pd.DataFrame,
        baseline_kwh: pd.Series,
        *,
        capacity_kwh: float,
    ) -> tuple[pd.Series, pd.DataFrame, dict[str, np.ndarray | int]]:
        _validate_context(context)
        if not isinstance(baseline_kwh, pd.Series) or not context.index.equals(baseline_kwh.index):
            raise ValueError("context and baseline indices differ")
        baseline_cf = baseline_kwh.to_numpy(dtype=np.float64, copy=False) / float(capacity_kwh)
        if not np.isfinite(baseline_cf).all():
            raise ValueError("baseline contains non-finite values")
        surface = self.predict_surfaces(context)
        utility = official_action_utility(surface["p6"], surface["p8"], surface["eae"])
        selected, selected_utility, selected_index = select_actions(utility, baseline_cf)
        row = np.arange(len(context))
        diagnostics = pd.DataFrame(
            {
                "baseline_cf": baseline_cf,
                "selected_action_cf": selected,
                "selected_action_index": selected_index,
                "selected_p6_raw": np.asarray(surface["p6_raw"])[row, selected_index],
                "selected_p8_raw": np.asarray(surface["p8_raw"])[row, selected_index],
                "selected_p6": np.asarray(surface["p6"])[row, selected_index],
                "selected_p8": np.asarray(surface["p8"])[row, selected_index],
                "selected_eae_cf": np.asarray(surface["eae"])[row, selected_index],
                "selected_utility": selected_utility,
            },
            index=context.index,
        )
        diagnostics.index.name = "forecast_kst_dtm"
        action = pd.Series(selected, index=context.index, name="action_cf")
        return action, diagnostics, surface

    def metadata(self) -> dict[str, Any]:
        if self.fit_metadata_ is None:
            raise RuntimeError("model is not fitted")
        return dict(self.fit_metadata_)


def assert_strict_forward(
    fit_index: pd.DatetimeIndex, application_index: pd.DatetimeIndex
) -> None:
    if len(fit_index) == 0 or len(application_index) == 0:
        raise ValueError("fit/application indices must be nonempty")
    if fit_index.intersection(application_index).size:
        raise ValueError("fit/application indices overlap")
    if not fit_index.max() < application_index.min():
        raise ValueError("fit must end before application begins")


def blend_action_kwh(
    baseline_kwh: pd.Series,
    action_cf: pd.Series,
    *,
    weight: float,
    capacity_kwh: float,
) -> pd.Series:
    if not baseline_kwh.index.equals(action_cf.index):
        raise ValueError("baseline/action indices differ")
    chosen = float(weight)
    if chosen not in (0.0, 0.025, 0.05, 0.10):
        raise ValueError("blend weight differs from preregistration")
    baseline = baseline_kwh.to_numpy(dtype=np.float64, copy=False)
    action = action_cf.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(baseline).all() or not np.isfinite(action).all():
        raise ValueError("blend inputs contain non-finite values")
    if chosen == 0.0:
        values = baseline.copy()
    else:
        values = np.clip(
            (1.0 - chosen) * baseline + chosen * action * float(capacity_kwh),
            0.0,
            1.02 * float(capacity_kwh),
        )
    return pd.Series(values, index=baseline_kwh.index, name=baseline_kwh.name)


def select_stage1_weight(
    comparisons: Mapping[str, Any],
) -> tuple[float | None, dict[str, Any]]:
    """Require all 17 exact official-score deltas to be strictly positive."""

    if tuple(comparisons) != tuple(BLEND_WEIGHTS):
        raise ValueError("candidate key/order differs from preregistration")
    audit: dict[str, Any] = {}
    eligible: list[str] = []
    for key, weight in BLEND_WEIGHTS.items():
        deltas: dict[str, float] = {}
        for group, required in STAGE1_REQUIRED.items():
            if set(comparisons[key][group]) != set(required):
                raise ValueError(f"{key}/{group} registered segments changed")
            for segment in required:
                record = comparisons[key][group][segment]
                delta = float(record["candidate"]["score"]) - float(record["baseline"]["score"])
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
    return float(BLEND_WEIGHTS[selected]), {"candidates": audit, "selected": selected}


__all__ = [
    "ACTION_GRID_CF",
    "BLEND_WEIGHTS",
    "COMMON_PARAMETERS",
    "CONTEXT_COLUMNS",
    "MODEL_COLUMNS",
    "STAGE1_REQUIRED",
    "DirectIntervalProbabilityModel",
    "assert_strict_forward",
    "blend_action_kwh",
    "compact_context",
    "expanded_action_design",
    "official_action_utility",
    "repair_interval_probabilities",
    "select_actions",
    "select_stage1_weight",
]

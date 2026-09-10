"""Causal FICR-boundary router for one fixed paired multi-NWP increment.

The router never predicts the target directly.  It receives a baseline and one
already-frozen paired information increment, expands each row over the fixed
0/50/100 percent actions, and estimates the two official interval
probabilities plus expected absolute error.  The selected action maximizes the
exact official utility represented by those three conditional quantities.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor

from src.direct_interval_probability import CONTEXT_COLUMNS
from src.multi_nwp_joint import JOINT_COLUMNS


ACTION_ALPHAS = np.asarray((1.0, 0.5, 0.0), dtype=np.float64)
PAIR_TRANSFER_WEIGHT = 0.25
STATE_COLUMNS = (
    "boundary__base_cf",
    "boundary__paired_increment_cf",
    "boundary__full_action_cf",
    "boundary__abs_paired_increment_cf",
)
BASE_FEATURE_COLUMNS = CONTEXT_COLUMNS + JOINT_COLUMNS + STATE_COLUMNS
ACTION_FEATURE_COLUMNS = (
    "route__alpha",
    "route__action_cf",
    "route__action_minus_base_cf",
    "route__distance_from_full_action_cf",
)
MODEL_FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + ACTION_FEATURE_COLUMNS

COMMON_PARAMETERS: dict[str, Any] = {
    "boosting_type": "gbdt",
    "n_estimators": 240,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 80,
    "min_split_gain": 0.0,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.75,
    "reg_alpha": 1.0,
    "reg_lambda": 16.0,
    "max_bin": 63,
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
    "n_jobs": 4,
}


def build_boundary_context(
    canonical_weather: pd.DataFrame,
    joint_features: pd.DataFrame,
    baseline_kwh: pd.Series,
    paired_increment_cf: pd.Series,
    *,
    capacity_kwh: float,
) -> pd.DataFrame:
    """Build the frozen compact context without reading an outcome."""

    if not isinstance(canonical_weather.index, pd.DatetimeIndex):
        raise TypeError("canonical_weather index must be DatetimeIndex")
    index = canonical_weather.index
    if not (
        joint_features.index.equals(index)
        and baseline_kwh.index.equals(index)
        and paired_increment_cf.index.equals(index)
    ):
        raise ValueError("boundary context inputs are not row-aligned")
    missing = set(CONTEXT_COLUMNS).difference(canonical_weather.columns)
    if missing:
        raise KeyError(f"canonical context columns missing: {sorted(missing)}")
    if tuple(joint_features.columns) != JOINT_COLUMNS:
        raise ValueError("joint feature schema/order differs")
    capacity = float(capacity_kwh)
    if not np.isfinite(capacity) or capacity <= 0.0:
        raise ValueError("capacity_kwh must be positive and finite")
    base_cf = baseline_kwh.to_numpy(dtype=np.float64) / capacity
    increment = paired_increment_cf.to_numpy(dtype=np.float64)
    state = pd.DataFrame(
        {
            STATE_COLUMNS[0]: base_cf,
            STATE_COLUMNS[1]: increment,
            STATE_COLUMNS[2]: np.clip(
                base_cf + PAIR_TRANSFER_WEIGHT * increment, 0.0, 1.02
            ),
            STATE_COLUMNS[3]: np.abs(increment),
        },
        index=index,
        dtype=np.float32,
    )
    result = pd.concat(
        (
            canonical_weather.loc[:, list(CONTEXT_COLUMNS)].astype(np.float32),
            joint_features.astype(np.float32),
            state,
        ),
        axis=1,
    )
    if tuple(result.columns) != BASE_FEATURE_COLUMNS:
        raise AssertionError("boundary context schema changed")
    if not np.isfinite(result.to_numpy(dtype=np.float64, copy=False)).all():
        raise ValueError("boundary context contains non-finite values")
    result.index.name = "forecast_kst_dtm"
    return result


def expanded_action_design(context: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return row-major context/action design and its action-CF matrix."""

    if tuple(context.columns) != BASE_FEATURE_COLUMNS:
        raise ValueError("boundary context schema/order differs")
    if context.empty or not context.index.is_unique or not context.index.is_monotonic_increasing:
        raise ValueError("boundary context index must be nonempty, unique and increasing")
    base = context.to_numpy(dtype=np.float32, copy=False)
    if not np.isfinite(base).all():
        raise ValueError("boundary context contains non-finite values")
    rows = len(context)
    actions = len(ACTION_ALPHAS)
    base_cf = context[STATE_COLUMNS[0]].to_numpy(dtype=np.float64)
    increment = context[STATE_COLUMNS[1]].to_numpy(dtype=np.float64)
    action_cf = np.clip(
        base_cf[:, None]
        + PAIR_TRANSFER_WEIGHT * increment[:, None] * ACTION_ALPHAS[None, :],
        0.0,
        1.02,
    )
    full_action = action_cf[:, 0]
    design = np.empty((rows * actions, len(MODEL_FEATURE_COLUMNS)), dtype=np.float32)
    design[:, : len(BASE_FEATURE_COLUMNS)] = np.repeat(base, actions, axis=0)
    alpha = np.tile(ACTION_ALPHAS.astype(np.float32), rows)
    flat_action = action_cf.reshape(-1).astype(np.float32)
    repeated_base = np.repeat(base_cf, actions).astype(np.float32)
    repeated_full = np.repeat(full_action, actions).astype(np.float32)
    design[:, len(BASE_FEATURE_COLUMNS) :] = np.column_stack(
        (
            alpha,
            flat_action,
            flat_action - repeated_base,
            np.abs(flat_action - repeated_full),
        )
    )
    if not np.isfinite(design).all():
        raise AssertionError("expanded boundary design contains non-finite values")
    return design, action_cf


def repair_probabilities(
    p6_raw: np.ndarray, p8_raw: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    p6 = np.clip(np.asarray(p6_raw, dtype=np.float64), 0.0, 1.0)
    p8 = np.clip(np.asarray(p8_raw, dtype=np.float64), 0.0, 1.0)
    if p6.shape != p8.shape or p6.ndim != 2:
        raise ValueError("p6/p8 arrays must be aligned and two-dimensional")
    violation = p6 > p8
    midpoint = 0.5 * (p6 + p8)
    return np.where(violation, midpoint, p6), np.where(violation, midpoint, p8), int(violation.sum())


def exact_conditional_utility(
    p6: np.ndarray, p8: np.ndarray, expected_absolute_error_cf: np.ndarray
) -> np.ndarray:
    """Return expected official utility up to an action-independent constant."""

    p6_array = np.asarray(p6, dtype=np.float64)
    p8_array = np.asarray(p8, dtype=np.float64)
    eae = np.asarray(expected_absolute_error_cf, dtype=np.float64)
    if p6_array.shape != p8_array.shape or p6_array.shape != eae.shape:
        raise ValueError("utility surfaces are not aligned")
    if np.any(p6_array > p8_array + 1e-15) or not np.isfinite(eae).all():
        raise ValueError("invalid utility surface")
    return -0.5 * np.clip(eae, 0.0, 1.20) + 0.5 * (
        0.25 * p6_array + 0.75 * p8_array
    )


class MultiNWPBoundaryRouter:
    """Three shallow heads routing a fixed paired increment at 0/50/100%."""

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
        return LGBMRegressor(objective="regression_l1", random_state=seed, **COMMON_PARAMETERS)

    def fit(
        self,
        context: pd.DataFrame,
        actual_kwh: pd.Series,
        *,
        capacity_kwh: float,
    ) -> "MultiNWPBoundaryRouter":
        if tuple(context.columns) != BASE_FEATURE_COLUMNS:
            raise ValueError("fit context schema differs")
        if not isinstance(actual_kwh, pd.Series) or not actual_kwh.index.equals(context.index):
            raise ValueError("fit context/actual indexes differ")
        capacity = float(capacity_kwh)
        actual_cf = actual_kwh.to_numpy(dtype=np.float64) / capacity
        eligible = np.isfinite(actual_cf) & (actual_cf >= 0.10)
        if int(eligible.sum()) < 500:
            raise ValueError("too few eligible router fit rows")
        fit_context = context.loc[eligible]
        y = actual_cf[eligible]
        design, action_cf = expanded_action_design(fit_context)
        repeated_y = np.repeat(y, len(ACTION_ALPHAS))
        error = np.abs(action_cf.reshape(-1) - repeated_y)
        target6 = (error <= 0.06).astype(np.uint8)
        target8 = (error <= 0.08).astype(np.uint8)
        energy_weight = repeated_y
        if np.unique(target6).size != 2 or np.unique(target8).size != 2:
            raise ValueError("interval target lost a class")
        self.p6_model_ = self._classifier(7606)
        self.p8_model_ = self._classifier(7608)
        self.eae_model_ = self._regressor(7610)
        self.p6_model_.fit(design, target6, sample_weight=energy_weight)
        self.p8_model_.fit(design, target8, sample_weight=energy_weight)
        self.eae_model_.fit(design, error)
        self.fit_index_ = context.index.copy()
        self.fit_metadata_ = {
            "rows_total": int(len(context)),
            "rows_eligible": int(eligible.sum()),
            "expanded_rows": int(len(design)),
            "feature_count": int(design.shape[1]),
            "mean_actual_cf": float(np.mean(y)),
            "p6_positive_fraction": float(np.mean(target6)),
            "p8_positive_fraction": float(np.mean(target8)),
            "parameters": dict(COMMON_PARAMETERS),
            "action_alphas": ACTION_ALPHAS.tolist(),
            "pair_transfer_weight": PAIR_TRANSFER_WEIGHT,
        }
        return self

    def predict(
        self,
        context: pd.DataFrame,
        baseline_kwh: pd.Series,
        *,
        capacity_kwh: float,
    ) -> tuple[pd.Series, pd.DataFrame]:
        if self.p6_model_ is None or self.p8_model_ is None or self.eae_model_ is None:
            raise RuntimeError("fit must be called before predict")
        if not context.index.equals(baseline_kwh.index):
            raise ValueError("application context/baseline indexes differ")
        if len(context.index.intersection(self.fit_index_)) or not self.fit_index_.max() < context.index.min():
            raise ValueError("router fit/application must be strict forward")
        design, action_cf = expanded_action_design(context)
        shape = action_cf.shape
        raw6 = self.p6_model_.predict_proba(design)[:, 1].reshape(shape)
        raw8 = self.p8_model_.predict_proba(design)[:, 1].reshape(shape)
        eae = self.eae_model_.predict(design).reshape(shape)
        p6, p8, repairs = repair_probabilities(raw6, raw8)
        utility = exact_conditional_utility(p6, p8, eae)
        # ACTION_ALPHAS is ordered 1.0, 0.5, 0.0: exact utility ties retain
        # the previously confirmed full joint action rather than manufacture
        # an arbitrary identity preference.
        selected_position = np.argmax(utility, axis=1)
        row = np.arange(len(context))
        selected_alpha = ACTION_ALPHAS[selected_position]
        selected_cf = action_cf[row, selected_position]
        capacity = float(capacity_kwh)
        prediction = pd.Series(
            selected_cf * capacity, index=context.index, name=baseline_kwh.name
        )
        diagnostics = pd.DataFrame(
            {
                "selected_alpha": selected_alpha,
                "selected_action_cf": selected_cf,
                "selected_p6": p6[row, selected_position],
                "selected_p8": p8[row, selected_position],
                "selected_eae_cf": eae[row, selected_position],
                "selected_utility": utility[row, selected_position],
                "utility_alpha100": utility[:, 0],
                "utility_alpha050": utility[:, 1],
                "utility_alpha000": utility[:, 2],
                "probability_repair_count_global": np.full(len(context), repairs),
                "candidate_minus_baseline_kwh": selected_cf * capacity
                - baseline_kwh.to_numpy(dtype=np.float64),
            },
            index=context.index,
        )
        diagnostics.index.name = "forecast_kst_dtm"
        return prediction, diagnostics

    def metadata(self) -> dict[str, Any]:
        if self.fit_metadata_ is None:
            raise RuntimeError("fit must be called before metadata")
        return dict(self.fit_metadata_)


__all__ = [
    "ACTION_ALPHAS",
    "ACTION_FEATURE_COLUMNS",
    "BASE_FEATURE_COLUMNS",
    "COMMON_PARAMETERS",
    "MODEL_FEATURE_COLUMNS",
    "PAIR_TRANSFER_WEIGHT",
    "STATE_COLUMNS",
    "MultiNWPBoundaryRouter",
    "build_boundary_context",
    "exact_conditional_utility",
    "expanded_action_design",
    "repair_probabilities",
]

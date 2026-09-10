"""Deterministic nested chronological HPO primitives for BARAM 2026.

The module is intentionally unaware of Public scores and submission files.  It
selects one group-specific direct-capacity-factor LightGBM recipe using only
causal inner folds supplied by the caller.  The caller is responsible for
locking the selected recipe and outer predictions before opening outer labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from lightgbm import LGBMRegressor
import numpy as np
import optuna
import pandas as pd

from src.metric import group_metrics


SAMPLER_SEED = 20_260_808
N_TRIALS = 16
N_STARTUP_TRIALS = 8
FIXED_PARAMETERS: dict[str, Any] = {
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
    "subsample_freq": 1,
    "random_state": SAMPLER_SEED,
    "n_jobs": 7,
}
OBJECTIVES: tuple[str, ...] = ("regression_l1", "quantile")
ALPHAS: tuple[float, ...] = (0.55, 0.60, 0.65, 0.70, 0.75)
N_ESTIMATORS: tuple[int, ...] = (500, 800, 1200)
NUM_LEAVES: tuple[int, ...] = (15, 31, 63)
MAX_DEPTHS: tuple[int, ...] = (-1, 6, 10)
MIN_CHILD_SAMPLES: tuple[int, ...] = (20, 40, 80, 120)
SUBSAMPLES: tuple[float, ...] = (0.70, 0.85, 1.0)
COLSAMPLES: tuple[float, ...] = (0.55, 0.70, 0.85, 1.0)
MIN_SPLIT_GAINS: tuple[float, ...] = (0.0, 0.01, 0.05)
MAX_BINS: tuple[int, ...] = (127, 255)


@dataclass(frozen=True)
class InnerFold:
    fit_start: pd.Timestamp
    fit_end: pd.Timestamp
    apply_start: pd.Timestamp
    apply_end: pd.Timestamp

    def __post_init__(self) -> None:
        if not (
            self.fit_start <= self.fit_end < self.apply_start <= self.apply_end
        ):
            raise ValueError("inner fold must be strictly causal and nonempty")


def suggest_parameters(trial: optuna.Trial) -> dict[str, Any]:
    """Sample exactly the preregistered fixed-dimensional search space."""

    objective = trial.suggest_categorical("objective", list(OBJECTIVES))
    # Alpha is deliberately sampled for every trial.  It is ignored for L1,
    # which keeps Optuna's parameter space fixed rather than conditional.
    alpha = float(trial.suggest_categorical("alpha", list(ALPHAS)))
    sampled: dict[str, Any] = {
        "objective": objective,
        "alpha": alpha,
        "n_estimators": int(
            trial.suggest_categorical("n_estimators", list(N_ESTIMATORS))
        ),
        "learning_rate": float(
            trial.suggest_float("learning_rate", 0.015, 0.06, log=True)
        ),
        "num_leaves": int(trial.suggest_categorical("num_leaves", list(NUM_LEAVES))),
        "max_depth": int(trial.suggest_categorical("max_depth", list(MAX_DEPTHS))),
        "min_child_samples": int(
            trial.suggest_categorical("min_child_samples", list(MIN_CHILD_SAMPLES))
        ),
        "subsample": float(trial.suggest_categorical("subsample", list(SUBSAMPLES))),
        "colsample_bytree": float(
            trial.suggest_categorical("colsample_bytree", list(COLSAMPLES))
        ),
        "reg_alpha": float(trial.suggest_float("reg_alpha", 0.001, 2.0, log=True)),
        "reg_lambda": float(
            trial.suggest_float("reg_lambda", 0.1, 20.0, log=True)
        ),
        "min_split_gain": float(
            trial.suggest_categorical("min_split_gain", list(MIN_SPLIT_GAINS))
        ),
        "max_bin": int(trial.suggest_categorical("max_bin", list(MAX_BINS))),
    }
    return sampled


def resolved_estimator_parameters(sampled: Mapping[str, Any]) -> dict[str, Any]:
    """Convert sampled values into the exact LightGBM constructor mapping."""

    objective = str(sampled["objective"])
    if objective not in OBJECTIVES:
        raise ValueError(f"unregistered objective: {objective}")
    output = {
        key: sampled[key]
        for key in (
            "objective",
            "n_estimators",
            "learning_rate",
            "num_leaves",
            "max_depth",
            "min_child_samples",
            "subsample",
            "colsample_bytree",
            "reg_alpha",
            "reg_lambda",
            "min_split_gain",
            "max_bin",
        )
    }
    if objective == "quantile":
        alpha = float(sampled["alpha"])
        if alpha not in ALPHAS:
            raise ValueError(f"unregistered alpha: {alpha}")
        output["alpha"] = alpha
    output.update(FIXED_PARAMETERS)
    return output


def fit_direct_cf(
    features: pd.DataFrame,
    target_cf: pd.Series,
    sampled: Mapping[str, Any],
) -> tuple[LGBMRegressor, dict[str, Any]]:
    """Fit on the metric-eligible rows only, with exact schema assertions."""

    if not isinstance(features, pd.DataFrame) or not isinstance(target_cf, pd.Series):
        raise TypeError("features and target must be pandas objects")
    if not features.index.equals(target_cf.index):
        raise ValueError("feature and target indices differ")
    matrix = features.to_numpy(dtype=np.float32, copy=False)
    target = target_cf.to_numpy(dtype=np.float64, copy=False)
    if matrix.ndim != 2 or matrix.shape[1] != 612:
        raise ValueError(f"expected 612 features, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("features contain non-finite values")
    eligible = np.isfinite(target) & (target >= 0.10)
    if int(eligible.sum()) < 100:
        raise ValueError("too few metric-eligible target rows")
    params = resolved_estimator_parameters(sampled)
    model = LGBMRegressor(**params)
    model.fit(features.loc[eligible], target[eligible])
    metadata = {
        "rows_total": int(len(features)),
        "rows_eligible": int(eligible.sum()),
        "feature_count": int(features.shape[1]),
        "sampled_parameters": dict(sampled),
        "resolved_parameters": params,
        "target_kind": "direct_capacity_factor",
        "minimum_actual_cf": 0.10,
        "sample_weight": "none",
    }
    return model, metadata


def predict_direct_cf(model: LGBMRegressor, features: pd.DataFrame) -> pd.Series:
    if features.shape[1] != 612:
        raise ValueError("application feature schema changed")
    values = np.asarray(model.predict(features), dtype=np.float64)
    if values.shape != (len(features),) or not np.isfinite(values).all():
        raise ValueError("model produced invalid direct-CF predictions")
    return pd.Series(
        np.clip(values, 0.0, 1.02),
        index=features.index,
        name="direct_cf",
    )


def exact_group_score(
    actual_kwh: pd.Series,
    predicted_cf: pd.Series,
    *,
    capacity_kwh: float,
    group: str,
) -> tuple[float, dict[str, Any]]:
    if not actual_kwh.index.equals(predicted_cf.index):
        raise ValueError("actual/prediction indices differ")
    predicted_kwh = predicted_cf.to_numpy(dtype=np.float64) * float(capacity_kwh)
    details = group_metrics(
        actual_kwh.to_numpy(dtype=np.float64),
        predicted_kwh,
        float(capacity_kwh),
        group_name=group,
    )
    score = 0.5 * (details.one_minus_nmae + details.ficr)
    output = details.as_dict()
    output["score"] = float(score)
    return float(score), output


def run_nested_search(
    *,
    features: pd.DataFrame,
    actual_kwh: pd.Series,
    capacity_kwh: float,
    group: str,
    folds: Sequence[InnerFold],
) -> tuple[dict[str, Any], list[dict[str, Any]], optuna.Study]:
    """Run the frozen TPE search and return a deterministic best recipe."""

    if not folds:
        raise ValueError("at least one inner fold is required")
    if not features.index.equals(actual_kwh.index):
        raise ValueError("features and labels must share the full fit-period index")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(
        seed=SAMPLER_SEED,
        n_startup_trials=N_STARTUP_TRIALS,
        multivariate=False,
        group=False,
        constant_liar=False,
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=optuna.pruners.NopPruner(),
    )

    def objective(trial: optuna.Trial) -> float:
        sampled = suggest_parameters(trial)
        fold_records: list[dict[str, Any]] = []
        fold_scores: list[float] = []
        for fold_number, fold in enumerate(folds, start=1):
            fit_index = features.loc[fold.fit_start : fold.fit_end].index
            apply_index = features.loc[fold.apply_start : fold.apply_end].index
            if (
                len(fit_index) == 0
                or len(apply_index) == 0
                or fit_index.max() >= apply_index.min()
                or len(fit_index.intersection(apply_index))
            ):
                raise AssertionError("inner fold causality changed")
            model, training = fit_direct_cf(
                features.loc[fit_index],
                actual_kwh.loc[fit_index] / float(capacity_kwh),
                sampled,
            )
            prediction = predict_direct_cf(model, features.loc[apply_index])
            score, details = exact_group_score(
                actual_kwh.loc[apply_index],
                prediction,
                capacity_kwh=capacity_kwh,
                group=group,
            )
            fold_scores.append(score)
            fold_records.append(
                {
                    "fold": fold_number,
                    "fit_start": fit_index.min().isoformat(),
                    "fit_end": fit_index.max().isoformat(),
                    "apply_start": apply_index.min().isoformat(),
                    "apply_end": apply_index.max().isoformat(),
                    "fit_before_apply": bool(fit_index.max() < apply_index.min()),
                    "training": training,
                    "metrics": details,
                }
            )
        aggregate = float(np.mean(fold_scores))
        trial.set_user_attr("fold_records", fold_records)
        trial.set_user_attr("fold_scores", fold_scores)
        trial.set_user_attr("aggregate_formula", "unweighted_arithmetic_mean")
        return aggregate

    study.optimize(objective, n_trials=N_TRIALS, n_jobs=1, show_progress_bar=False)
    completed = [
        trial
        for trial in study.trials
        if trial.state is optuna.trial.TrialState.COMPLETE
        and trial.value is not None
        and np.isfinite(float(trial.value))
    ]
    if len(completed) != N_TRIALS:
        raise AssertionError(f"expected {N_TRIALS} completed trials, got {len(completed)}")
    selected = max(completed, key=lambda trial: (float(trial.value), -trial.number))
    if selected.number != study.best_trial.number:
        raise AssertionError("explicit best-trial tie break differs from Optuna")
    best = {
        "trial_number": int(selected.number),
        "objective_value": float(selected.value),
        "sampled_parameters": dict(selected.params),
        "resolved_estimator_parameters": resolved_estimator_parameters(selected.params),
        "fold_records": selected.user_attrs["fold_records"],
        "objective_aggregation": "unweighted_arithmetic_mean",
        "tie_break": "highest objective then smallest trial number",
    }
    history: list[dict[str, Any]] = []
    for trial in study.trials:
        history.append(
            {
                "trial_number": int(trial.number),
                "state": trial.state.name,
                "value": None if trial.value is None else float(trial.value),
                "params": dict(trial.params),
                "fold_scores": trial.user_attrs.get("fold_scores"),
                "fold_records": trial.user_attrs.get("fold_records"),
            }
        )
    return best, history, study


def blend_with_baseline(
    baseline_kwh: pd.Series,
    direct_cf: pd.Series,
    *,
    weight: float,
    capacity_kwh: float,
) -> pd.Series:
    if not baseline_kwh.index.equals(direct_cf.index):
        raise ValueError("baseline/direct model indices differ")
    if weight not in (0.025, 0.05, 0.10, 0.20, 1.0):
        raise ValueError("unregistered blend weight")
    values = (
        (1.0 - float(weight)) * baseline_kwh.to_numpy(dtype=np.float64)
        + float(weight)
        * np.clip(direct_cf.to_numpy(dtype=np.float64), 0.0, 1.02)
        * float(capacity_kwh)
    )
    values = np.clip(values, 0.0, 1.02 * float(capacity_kwh))
    return pd.Series(values, index=baseline_kwh.index, name=baseline_kwh.name)


__all__ = [
    "ALPHAS",
    "COLSAMPLES",
    "FIXED_PARAMETERS",
    "InnerFold",
    "MAX_BINS",
    "MAX_DEPTHS",
    "MIN_CHILD_SAMPLES",
    "MIN_SPLIT_GAINS",
    "N_ESTIMATORS",
    "N_STARTUP_TRIALS",
    "N_TRIALS",
    "NUM_LEAVES",
    "OBJECTIVES",
    "SAMPLER_SEED",
    "SUBSAMPLES",
    "blend_with_baseline",
    "exact_group_score",
    "fit_direct_cf",
    "predict_direct_cf",
    "resolved_estimator_parameters",
    "run_nested_search",
    "suggest_parameters",
]

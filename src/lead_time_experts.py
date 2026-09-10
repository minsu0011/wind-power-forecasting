"""Leakage-safe utilities for forecast lead-bin LightGBM experts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.run_sequence_residual import assert_strict_fit_predict_order, validate_complete_runs
from src.temporal import operating_run_key


LEAD_COLUMN = "time__lead_hours"
FORBIDDEN_FEATURE_TOKENS = (
    "actual",
    "target",
    "label",
    "scada",
    "power_kw",
    "kpx_group_",
)


def lead_bin_masks(
    lead: pd.Series, bins: Sequence[Mapping[str, Any]]
) -> dict[str, np.ndarray]:
    """Return an exact, disjoint partition of integer forecast leads."""

    values = lead.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("lead values must be finite")
    if not np.array_equal(values, np.rint(values)):
        raise ValueError("lead values must be integer hours")
    membership = np.zeros(len(values), dtype=np.int8)
    output: dict[str, np.ndarray] = {}
    for specification in bins:
        name = str(specification["id"])
        lower = float(specification["lower_inclusive"])
        upper = float(specification["upper_exclusive"])
        if name in output or not lower < upper:
            raise ValueError("lead-bin ids must be unique and bounds increasing")
        mask = (values >= lower) & (values < upper)
        output[name] = mask
        membership += mask.astype(np.int8)
    if not np.all(membership == 1):
        raise ValueError("lead bins must assign every row exactly once")
    return output


def validate_lead_partition(
    features: pd.DataFrame,
    bins: Sequence[Mapping[str, Any]],
    *,
    lead_column: str = LEAD_COLUMN,
    expected_leads: Sequence[int] = tuple(range(12, 36)),
    expected_hours_per_run: int = 24,
    expected_rows_per_bin_per_run: int = 6,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Validate run completeness and the registered four-bin lead partition."""

    if lead_column not in features:
        raise KeyError(f"missing lead column {lead_column}")
    forbidden = [
        str(column)
        for column in features.columns
        if any(token in str(column).lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if forbidden:
        raise ValueError(f"forbidden feature columns: {forbidden[:5]}")
    validate_complete_runs(features.index, expected_hours=expected_hours_per_run)
    lead = features[lead_column].astype(float)
    masks = lead_bin_masks(lead, bins)
    observed = tuple(sorted(map(int, pd.unique(lead))))
    expected = tuple(map(int, expected_leads))
    if observed != expected:
        raise ValueError(f"lead support changed: {observed} != {expected}")
    run_key = operating_run_key(features.index)
    expected_sequence = np.asarray(expected, dtype=np.float64)
    for _, positions in pd.Series(np.arange(len(features)), index=features.index).groupby(
        run_key, sort=True
    ):
        run_values = lead.to_numpy(dtype=np.float64, copy=False)[
            positions.to_numpy(dtype=int)
        ]
        if not np.array_equal(run_values, expected_sequence):
            raise ValueError("each run must contain ordered leads 12,13,...,35")
    per_bin: dict[str, Any] = {}
    for name, mask in masks.items():
        counts = pd.Series(mask.astype(np.int8), index=features.index).groupby(
            run_key, sort=True
        ).sum()
        if not counts.eq(expected_rows_per_bin_per_run).all():
            raise ValueError(f"{name} does not contain six rows in every run")
        per_bin[name] = {
            "rows": int(mask.sum()),
            "runs": int(len(counts)),
            "rows_per_run_min": int(counts.min()),
            "rows_per_run_max": int(counts.max()),
            "lead_min": int(lead.to_numpy()[mask].min()),
            "lead_max": int(lead.to_numpy()[mask].max()),
        }
    return masks, {
        "rows": int(len(features)),
        "runs": int(len(features) // expected_hours_per_run),
        "lead_column": lead_column,
        "lead_values": list(observed),
        "ordered_lead_sequence_exact_every_run": True,
        "membership_min": 1,
        "membership_max": 1,
        "rows_per_run": expected_hours_per_run,
        "bins": per_bin,
    }


def model_parameters(
    recipe: Mapping[str, Any], model_contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the exact locked LightGBM parameter mapping."""

    params = dict(model_contract["common_params"])
    params.update(
        {
            "objective": str(recipe["objective"]),
            "random_state": int(model_contract["random_state"]),
            "n_jobs": int(model_contract["n_jobs"]),
        }
    )
    if recipe.get("alpha") is not None:
        params["alpha"] = float(recipe["alpha"])
    return params


def fit_global_and_lead_experts(
    *,
    features: pd.DataFrame,
    actual_cf: pd.Series,
    fit_index: pd.DatetimeIndex,
    apply_index: pd.DatetimeIndex,
    bins: Sequence[Mapping[str, Any]],
    recipe: Mapping[str, Any],
    model_contract: Mapping[str, Any],
) -> tuple[LGBMRegressor, dict[str, LGBMRegressor], pd.Series, pd.Series, dict[str, Any]]:
    """Fit one global control and four disjoint lead experts."""

    if not features.index.equals(actual_cf.index):
        raise ValueError("feature and target indices differ")
    if not features.index.is_unique or not features.index.is_monotonic_increasing:
        raise ValueError("feature index must be unique and increasing")
    if not fit_index.isin(features.index).all() or not apply_index.isin(features.index).all():
        raise ValueError("fit/apply indices must be contained in features")
    assert_strict_fit_predict_order(fit_index, apply_index)
    x = features.to_numpy(dtype=np.float32, copy=False)
    if not np.isfinite(x).all():
        raise ValueError("weather features contain non-finite values")
    values = actual_cf.to_numpy(dtype=np.float64, copy=False)
    eligible = np.isfinite(values) & (values >= 0.10)
    fit_mask = features.index.isin(fit_index)
    apply_mask = features.index.isin(apply_index)
    train_global = fit_mask & eligible
    if int(train_global.sum()) < 100:
        raise ValueError("too few eligible global training rows")
    params = model_parameters(recipe, model_contract)
    global_model = LGBMRegressor(**params)
    global_model.fit(features.loc[train_global], actual_cf.loc[train_global])
    global_values = np.asarray(
        global_model.predict(features.loc[apply_index]), dtype=np.float64
    )
    if global_values.shape != (len(apply_index),) or not np.isfinite(global_values).all():
        raise ValueError("global model prediction is invalid")
    masks = lead_bin_masks(features[LEAD_COLUMN].astype(float), bins)
    stitched = pd.Series(np.nan, index=apply_index, dtype=float, name="expert_cf")
    expert_models: dict[str, LGBMRegressor] = {}
    bin_audit: dict[str, Any] = {}
    for specification in bins:
        name = str(specification["id"])
        train_mask = fit_mask & masks[name] & eligible
        predict_mask = apply_mask & masks[name]
        train_index = features.index[train_mask]
        prediction_index = features.index[predict_mask]
        if int(train_mask.sum()) < 100 or len(prediction_index) == 0:
            raise ValueError(f"{name} has too few fit rows or no apply rows")
        assert_strict_fit_predict_order(train_index, prediction_index)
        model = LGBMRegressor(**params)
        model.fit(features.loc[train_index], actual_cf.loc[train_index])
        prediction = np.asarray(
            model.predict(features.loc[prediction_index]), dtype=np.float64
        )
        if prediction.shape != (len(prediction_index),) or not np.isfinite(prediction).all():
            raise ValueError(f"{name} prediction is invalid")
        if stitched.loc[prediction_index].notna().any():
            raise AssertionError("expert predictions overlap")
        stitched.loc[prediction_index] = prediction
        expert_models[name] = model
        bin_audit[name] = {
            "fit_rows_total": int((fit_mask & masks[name]).sum()),
            "fit_rows_eligible": int(train_mask.sum()),
            "apply_rows": int(len(prediction_index)),
            "fit_start": train_index.min().isoformat(),
            "fit_end": train_index.max().isoformat(),
            "apply_start": prediction_index.min().isoformat(),
            "apply_end": prediction_index.max().isoformat(),
            "fit_max_strictly_before_apply_min": True,
        }
    if stitched.isna().any() or not np.isfinite(stitched.to_numpy()).all():
        raise AssertionError("stitched expert predictions contain gaps")
    global_prediction = pd.Series(
        global_values, index=apply_index, dtype=float, name="global_cf"
    )
    return global_model, expert_models, global_prediction, stitched, {
        "recipe_id": str(recipe["id"]),
        "feature_columns": int(features.shape[1]),
        "fit_rows_total": int(len(fit_index)),
        "fit_rows_eligible": int(train_global.sum()),
        "apply_rows": int(len(apply_index)),
        "fit_start": fit_index.min().isoformat(),
        "fit_end": fit_index.max().isoformat(),
        "apply_start": apply_index.min().isoformat(),
        "apply_end": apply_index.max().isoformat(),
        "fit_apply_overlap_rows": int(fit_index.intersection(apply_index).size),
        "fit_max_strictly_before_apply_min": bool(fit_index.max() < apply_index.min()),
        "target_or_scada_feature_columns": 0,
        "bins": bin_audit,
    }


def specialization_candidate(
    baseline_kwh: pd.Series,
    expert_kwh: pd.Series,
    global_kwh: pd.Series,
    *,
    weight: float,
    capacity_kwh: float,
) -> pd.Series:
    """Replace only a fixed fraction of a matching global component."""

    if not baseline_kwh.index.equals(expert_kwh.index) or not baseline_kwh.index.equals(
        global_kwh.index
    ):
        raise ValueError("baseline, expert, and global indices differ")
    if not 0.0 < float(weight) <= 1.0:
        raise ValueError("weight must be in (0, 1]")
    values = (
        baseline_kwh.to_numpy(dtype=np.float64, copy=False)
        + float(weight)
        * (
            expert_kwh.to_numpy(dtype=np.float64, copy=False)
            - global_kwh.to_numpy(dtype=np.float64, copy=False)
        )
    )
    if not np.isfinite(values).all():
        raise ValueError("candidate contains non-finite values")
    return pd.Series(
        np.clip(values, 0.0, 1.02 * float(capacity_kwh)),
        index=baseline_kwh.index,
        name=baseline_kwh.name,
    )


def choose_registered_candidate(
    comparisons: Mapping[str, Mapping[str, Mapping[str, Mapping[str, float]]]],
    specifications: Mapping[str, Mapping[str, Any]],
) -> str | None:
    """Select one global recipe only if every registered delta is positive."""

    eligible: list[tuple[float, float, float, str, str]] = []
    for candidate_id, by_group in comparisons.items():
        if candidate_id not in specifications:
            raise KeyError(f"missing specification for {candidate_id}")
        deltas = [
            float(result["delta"])
            for slices in by_group.values()
            for result in slices.values()
        ]
        if not deltas or not np.isfinite(deltas).all():
            raise ValueError(f"invalid comparison deltas for {candidate_id}")
        if all(delta > 0.0 for delta in deltas):
            specification = specifications[candidate_id]
            eligible.append(
                (
                    -min(deltas),
                    -float(np.mean(deltas)),
                    float(specification["weight"]),
                    str(specification["recipe_id"]),
                    str(candidate_id),
                )
            )
    if not eligible:
        return None
    eligible.sort()
    return eligible[0][-1]


__all__ = [
    "FORBIDDEN_FEATURE_TOKENS",
    "LEAD_COLUMN",
    "choose_registered_candidate",
    "fit_global_and_lead_experts",
    "lead_bin_masks",
    "model_parameters",
    "specialization_candidate",
    "validate_lead_partition",
]

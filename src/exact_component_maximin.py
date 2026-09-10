"""Exact-official-score maximin search over a low-DOF component simplex.

This differs from the repository's pseudo-Huber stacking experiment: Optuna
directly evaluates baseline-relative official score on registered temporal
blocks.  The simplex can only transfer mass from one locked-v3 donor to one
recipient, and affine perturbations are deliberately small.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import optuna
import pandas as pd

from src.metric import CAPACITY_KWH, group_metrics


COMPONENTS: tuple[str, ...] = (
    "lgb_l1",
    "lgb_q07",
    "shared_l1",
    "shared_q07",
    "top200_q07",
    "energy_q06",
)
DONORS: dict[str, tuple[str, ...]] = {
    "kpx_group_1": ("lgb_l1", "lgb_q07", "shared_l1", "shared_q07", "energy_q06"),
    "kpx_group_2": ("lgb_l1", "lgb_q07", "top200_q07"),
    "kpx_group_3": ("lgb_q07", "shared_q07", "energy_q06"),
}
RECIPIENT_OFFSETS: tuple[int, ...] = (1, 2, 3, 4, 5)
SAMPLER_SEED = 20_260_808
N_TRIALS = 32
N_STARTUP_TRIALS = 12
MASS_LIMIT = 0.08
SCALE_LIMIT = 0.015
BIAS_LIMIT_CF = 0.0075
REGULARIZATION_COEFFICIENT = 0.0001


def suggest_parameters(trial: optuna.Trial, group: str) -> dict[str, Any]:
    if group not in DONORS:
        raise KeyError(group)
    return {
        "donor": str(trial.suggest_categorical("donor", list(DONORS[group]))),
        "recipient_offset": int(
            trial.suggest_categorical("recipient_offset", list(RECIPIENT_OFFSETS))
        ),
        "mass_shift": float(trial.suggest_float("mass_shift", 0.0, MASS_LIMIT)),
        "scale_delta": float(trial.suggest_float("scale_delta", -SCALE_LIMIT, SCALE_LIMIT)),
        "bias_delta_cf": float(
            trial.suggest_float("bias_delta_cf", -BIAS_LIMIT_CF, BIAS_LIMIT_CF)
        ),
    }


def is_identity_parameter(parameter: Mapping[str, Any]) -> bool:
    return (
        float(parameter["mass_shift"]) == 0.0
        and float(parameter["scale_delta"]) == 0.0
        and float(parameter["bias_delta_cf"]) == 0.0
    )


def perturbed_weights(
    locked_weights: Mapping[str, float], parameter: Mapping[str, Any]
) -> dict[str, float]:
    if tuple(locked_weights) != COMPONENTS:
        raise ValueError("locked component order changed")
    weights = np.asarray([locked_weights[name] for name in COMPONENTS], dtype=np.float64)
    if (
        not np.isfinite(weights).all()
        or np.any(weights < 0.0)
        or not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("locked weights are not a simplex")
    donor = str(parameter["donor"])
    if donor not in COMPONENTS:
        raise ValueError("unknown donor")
    offset = int(parameter["recipient_offset"])
    if offset not in RECIPIENT_OFFSETS:
        raise ValueError("unregistered recipient offset")
    shift = float(parameter["mass_shift"])
    if not np.isfinite(shift) or not 0.0 <= shift <= MASS_LIMIT:
        raise ValueError("mass shift outside preregistration")
    donor_index = COMPONENTS.index(donor)
    recipient_index = (donor_index + offset) % len(COMPONENTS)
    if donor_index == recipient_index:
        raise AssertionError("recipient must differ from donor")
    if weights[donor_index] + 1e-15 < shift:
        raise ValueError("mass shift exceeds donor weight")
    weights[donor_index] -= shift
    weights[recipient_index] += shift
    if np.any(weights < -1e-14) or not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise AssertionError("perturbed weights left simplex")
    weights = np.maximum(weights, 0.0)
    # Only round-off correction is allowed; no projection/search occurs.
    weights[donor_index] += 1.0 - float(weights.sum())
    return {name: float(weights[position]) for position, name in enumerate(COMPONENTS)}


def assemble_candidate(
    components_kwh: Mapping[str, pd.Series],
    *,
    group: str,
    recipe: Mapping[str, Any],
    parameter: Mapping[str, Any],
    identity_baseline_kwh: pd.Series,
) -> tuple[pd.Series, dict[str, Any]]:
    if tuple(components_kwh) != COMPONENTS:
        raise ValueError("component order changed")
    index = identity_baseline_kwh.index
    if any(not series.index.equals(index) for series in components_kwh.values()):
        raise ValueError("component indices differ")
    if is_identity_parameter(parameter):
        return identity_baseline_kwh.copy(), {
            "identity_short_circuit": True,
            "weights": dict(recipe["ensemble"]["weights"][group]),
            "scale_delta": 0.0,
            "bias_delta_cf": 0.0,
            "power_bin_changed": False,
        }
    capacity = float(CAPACITY_KWH[group])
    locked = recipe["ensemble"]
    weights = perturbed_weights(locked["weights"][group], parameter)
    matrix = np.column_stack(
        [components_kwh[name].to_numpy(dtype=np.float64) for name in COMPONENTS]
    )
    if matrix.shape != (len(index), len(COMPONENTS)) or not np.isfinite(matrix).all():
        raise ValueError("component matrix is invalid")
    weighted = matrix @ np.asarray([weights[name] for name in COMPONENTS], dtype=np.float64)
    scale_delta = float(parameter["scale_delta"])
    bias_delta_cf = float(parameter["bias_delta_cf"])
    if not -SCALE_LIMIT <= scale_delta <= SCALE_LIMIT:
        raise ValueError("scale perturbation outside preregistration")
    if not -BIAS_LIMIT_CF <= bias_delta_cf <= BIAS_LIMIT_CF:
        raise ValueError("bias perturbation outside preregistration")
    affine = locked["affine"][group]
    scale = float(affine["scale"]) * (1.0 + scale_delta)
    bias_kwh = float(affine["bias_kwh"]) + bias_delta_cf * capacity
    lower = float(locked["clip"][group]["lower_capacity_fraction"]) * capacity
    upper = float(locked["clip"][group]["upper_capacity_fraction"]) * capacity
    output = np.clip(scale * weighted + bias_kwh, lower, upper)
    bin_specification = locked["power_bins"][group]
    if bin_specification is not None:
        edges = np.asarray(bin_specification["edges_cf"], dtype=np.float64)
        deltas = np.asarray(bin_specification["delta_kwh"], dtype=np.float64)
        positions = np.searchsorted(edges[1:-1], output / capacity, side="right")
        if np.any(positions < 0) or np.any(positions >= len(deltas)):
            raise ValueError("fixed power-bin edges do not cover candidate")
        output = np.clip(output + deltas[positions], lower, upper)
    if not np.isfinite(output).all():
        raise ValueError("candidate assembly produced non-finite values")
    metadata = {
        "identity_short_circuit": False,
        "weights": weights,
        "donor": str(parameter["donor"]),
        "recipient": COMPONENTS[
            (COMPONENTS.index(str(parameter["donor"])) + int(parameter["recipient_offset"]))
            % len(COMPONENTS)
        ],
        "mass_shift": float(parameter["mass_shift"]),
        "scale": scale,
        "bias_kwh": bias_kwh,
        "scale_delta": scale_delta,
        "bias_delta_cf": bias_delta_cf,
        "power_bin_changed": False,
        "power_bin_applied": bin_specification is not None,
    }
    return pd.Series(output, index=index, name=group), metadata


def group_triplet(actual: pd.Series, prediction: pd.Series, group: str) -> dict[str, float]:
    if not actual.index.equals(prediction.index):
        raise ValueError("actual/prediction indices differ")
    details = group_metrics(actual, prediction, CAPACITY_KWH[group], group_name=group)
    return {
        "score": float(0.5 * (details.one_minus_nmae + details.ficr)),
        "one_minus_nmae": float(details.one_minus_nmae),
        "ficr": float(details.ficr),
    }


def compare_blocks(
    actual: pd.Series,
    baseline: pd.Series,
    candidate: pd.Series,
    *,
    group: str,
    blocks: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    if not actual.index.equals(baseline.index) or not actual.index.equals(candidate.index):
        raise ValueError("score frames are not aligned")
    output: dict[str, Any] = {}
    for name, bounds in blocks.items():
        start, end = map(pd.Timestamp, bounds)
        selected = actual.index[(actual.index >= start) & (actual.index <= end)]
        if len(selected) == 0:
            raise ValueError(f"empty block: {name}")
        base = group_triplet(actual.loc[selected], baseline.loc[selected], group)
        cand = group_triplet(actual.loc[selected], candidate.loc[selected], group)
        output[name] = {
            "rows": len(selected),
            "baseline": base,
            "candidate": cand,
            "delta": {metric: float(cand[metric] - base[metric]) for metric in base},
        }
    return output


def parameter_penalty(parameter: Mapping[str, Any]) -> float:
    return float(
        REGULARIZATION_COEFFICIENT
        * (
            float(parameter["mass_shift"]) / MASS_LIMIT
            + abs(float(parameter["scale_delta"])) / SCALE_LIMIT
            + abs(float(parameter["bias_delta_cf"])) / BIAS_LIMIT_CF
        )
    )


def robust_objective(comparisons: Mapping[str, Any], parameter: Mapping[str, Any]) -> float:
    deltas = np.asarray(
        [float(values["delta"]["score"]) for values in comparisons.values()],
        dtype=np.float64,
    )
    if len(deltas) == 0 or not np.isfinite(deltas).all():
        raise ValueError("invalid block score deltas")
    return float(np.min(deltas) + 0.25 * np.mean(deltas) - parameter_penalty(parameter))


def passes_block_gate(comparisons: Mapping[str, Any], *, full_block: str) -> bool:
    if full_block not in comparisons:
        raise KeyError(full_block)
    if any(float(values["delta"]["score"]) <= 0.0 for values in comparisons.values()):
        return False
    full = comparisons[full_block]["delta"]
    nmae = float(full["one_minus_nmae"])
    ficr = float(full["ficr"])
    return nmae >= 0.0 and ficr >= 0.0 and (nmae > 0.0 or ficr > 0.0)


def run_exact_search(
    *,
    group: str,
    components_kwh: Mapping[str, pd.Series],
    actual_kwh: pd.Series,
    baseline_kwh: pd.Series,
    recipe: Mapping[str, Any],
    blocks: Mapping[str, Sequence[str]],
    full_block: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not actual_kwh.index.equals(baseline_kwh.index):
        raise ValueError("search labels/baseline differ")
    if any(not series.index.equals(actual_kwh.index) for series in components_kwh.values()):
        raise ValueError("search component index differs")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(
        seed=SAMPLER_SEED,
        n_startup_trials=N_STARTUP_TRIALS,
        multivariate=False,
        group=False,
        constant_liar=False,
    )
    study = optuna.create_study(
        direction="maximize", sampler=sampler, pruner=optuna.pruners.NopPruner()
    )
    study.enqueue_trial(
        {
            "donor": DONORS[group][0],
            "recipient_offset": 1,
            "mass_shift": 0.0,
            "scale_delta": 0.0,
            "bias_delta_cf": 0.0,
        },
        skip_if_exists=False,
    )

    def objective(trial: optuna.Trial) -> float:
        parameter = suggest_parameters(trial, group)
        candidate, assembly = assemble_candidate(
            components_kwh,
            group=group,
            recipe=recipe,
            parameter=parameter,
            identity_baseline_kwh=baseline_kwh,
        )
        comparisons = compare_blocks(
            actual_kwh,
            baseline_kwh,
            candidate,
            group=group,
            blocks=blocks,
        )
        value = robust_objective(comparisons, parameter)
        trial.set_user_attr("comparisons", comparisons)
        trial.set_user_attr("assembly", assembly)
        trial.set_user_attr("penalty", parameter_penalty(parameter))
        trial.set_user_attr("all_blocks_pass", passes_block_gate(comparisons, full_block=full_block))
        return value

    study.optimize(objective, n_trials=N_TRIALS, n_jobs=1, show_progress_bar=False)
    completed = [
        trial
        for trial in study.trials
        if trial.state is optuna.trial.TrialState.COMPLETE
        and trial.value is not None
        and np.isfinite(float(trial.value))
    ]
    if len(completed) != N_TRIALS:
        raise AssertionError("not every registered Optuna trial completed")
    selected = max(completed, key=lambda trial: (float(trial.value), -trial.number))
    if selected.number != study.best_trial.number:
        raise AssertionError("explicit Optuna tie break changed")
    candidate, assembly = assemble_candidate(
        components_kwh,
        group=group,
        recipe=recipe,
        parameter=selected.params,
        identity_baseline_kwh=baseline_kwh,
    )
    comparisons = compare_blocks(
        actual_kwh, baseline_kwh, candidate, group=group, blocks=blocks
    )
    inner_passed = (
        not is_identity_parameter(selected.params)
        and passes_block_gate(comparisons, full_block=full_block)
    )
    best = {
        "trial_number": int(selected.number),
        "objective_value": float(selected.value),
        "parameter": dict(selected.params),
        "assembly": assembly,
        "comparisons": comparisons,
        "penalty": parameter_penalty(selected.params),
        "inner_passed": bool(inner_passed),
        "locked_parameter": dict(selected.params) if inner_passed else None,
        "full_block": full_block,
    }
    history = [
        {
            "trial_number": int(trial.number),
            "state": trial.state.name,
            "objective_value": None if trial.value is None else float(trial.value),
            "parameter": dict(trial.params),
            "comparisons": trial.user_attrs.get("comparisons"),
            "assembly": trial.user_attrs.get("assembly"),
            "penalty": trial.user_attrs.get("penalty"),
            "all_blocks_pass": trial.user_attrs.get("all_blocks_pass"),
        }
        for trial in study.trials
    ]
    return best, history


def transfer_delta(
    recent_baseline_kwh: pd.Series,
    v3_baseline_kwh: pd.Series,
    candidate_v3_kwh: pd.Series,
    *,
    capacity_kwh: float,
) -> pd.Series:
    if not (
        recent_baseline_kwh.index.equals(v3_baseline_kwh.index)
        and recent_baseline_kwh.index.equals(candidate_v3_kwh.index)
    ):
        raise ValueError("delta-transfer indices differ")
    output = np.clip(
        recent_baseline_kwh.to_numpy(dtype=np.float64)
        + candidate_v3_kwh.to_numpy(dtype=np.float64)
        - v3_baseline_kwh.to_numpy(dtype=np.float64),
        0.0,
        1.02 * float(capacity_kwh),
    )
    return pd.Series(output, index=recent_baseline_kwh.index, name=recent_baseline_kwh.name)


__all__ = [
    "BIAS_LIMIT_CF",
    "COMPONENTS",
    "DONORS",
    "MASS_LIMIT",
    "N_STARTUP_TRIALS",
    "N_TRIALS",
    "RECIPIENT_OFFSETS",
    "REGULARIZATION_COEFFICIENT",
    "SAMPLER_SEED",
    "SCALE_LIMIT",
    "assemble_candidate",
    "compare_blocks",
    "group_triplet",
    "is_identity_parameter",
    "parameter_penalty",
    "passes_block_gate",
    "perturbed_weights",
    "robust_objective",
    "run_exact_search",
    "suggest_parameters",
    "transfer_delta",
]

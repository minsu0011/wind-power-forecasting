"""Frozen selective 0.98 scale transfer over direct interval utility surfaces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


ACTION_GRID_CF = np.arange(103, dtype=np.float64) / 100.0
SCALE_FACTOR = 0.98
UTILITY_MARGIN = 0.01
PROMOTED_GROUPS: tuple[str, ...] = ("kpx_group_1", "kpx_group_2")
IDENTITY_GROUPS: tuple[str, ...] = ("kpx_group_3",)
SEGMENTS: tuple[str, ...] = ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4")


def interpolate_row_utility(utility: Any, query_cf: Any) -> np.ndarray:
    """Linearly interpolate each row on the immutable 0.01 absolute-CF grid."""

    surface = np.asarray(utility, dtype=np.float64)
    query = np.asarray(query_cf, dtype=np.float64)
    if surface.ndim != 2 or surface.shape[1] != len(ACTION_GRID_CF):
        raise ValueError("utility must have 103 fixed action columns")
    if query.shape != (surface.shape[0],):
        raise ValueError("query_cf must have one value per utility row")
    if not np.isfinite(surface).all() or not np.isfinite(query).all():
        raise ValueError("utility interpolation inputs must be finite")
    coordinate = 100.0 * np.clip(query, 0.0, 1.02)
    lower = np.floor(coordinate).astype(np.int16)
    upper = np.minimum(lower + 1, 102).astype(np.int16)
    fraction = coordinate - lower
    row = np.arange(len(query))
    return (
        (1.0 - fraction) * surface[row, lower]
        + fraction * surface[row, upper]
    )


def selective_scale_candidate(
    baseline_kwh: pd.Series,
    utility: Any,
    *,
    capacity_kwh: float,
) -> tuple[pd.Series, pd.DataFrame]:
    """Apply exactly one fixed 0.98 action when utility advantage exceeds 0.01."""

    if not isinstance(baseline_kwh, pd.Series):
        raise TypeError("baseline_kwh must be a Series")
    if baseline_kwh.empty or not baseline_kwh.index.is_unique or not baseline_kwh.index.is_monotonic_increasing:
        raise ValueError("baseline index must be nonempty, unique, and increasing")
    capacity = float(capacity_kwh)
    if not np.isfinite(capacity) or capacity <= 0:
        raise ValueError("capacity_kwh must be positive and finite")
    baseline = baseline_kwh.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(baseline).all():
        raise ValueError("baseline contains non-finite values")
    baseline_cf = baseline / capacity
    scaled_cf = SCALE_FACTOR * baseline_cf
    base_utility = interpolate_row_utility(utility, baseline_cf)
    scaled_utility = interpolate_row_utility(utility, scaled_cf)
    advantage = scaled_utility - base_utility
    gate = advantage > UTILITY_MARGIN
    candidate_cf = np.where(gate, scaled_cf, baseline_cf)
    candidate_kwh = np.clip(candidate_cf * capacity, 0.0, 1.02 * capacity)
    candidate = pd.Series(candidate_kwh, index=baseline_kwh.index, name=baseline_kwh.name)
    diagnostics = pd.DataFrame(
        {
            "baseline_cf": baseline_cf,
            "scaled_cf": scaled_cf,
            "base_utility": base_utility,
            "scaled_utility": scaled_utility,
            "utility_advantage": advantage,
            "gate": gate,
            "candidate_cf": candidate_kwh / capacity,
        },
        index=baseline_kwh.index,
    )
    diagnostics.index.name = baseline_kwh.index.name
    return candidate, diagnostics


def stage1_group_promotions(
    comparisons: Mapping[str, Mapping[str, Any]],
    required: Mapping[str, Sequence[str]],
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, Any]]:
    """Promote each group only when all of its registered deltas are positive."""

    if set(comparisons) != set(required):
        raise ValueError("Stage1 group set differs")
    passed: list[str] = []
    failed: list[str] = []
    audit: dict[str, Any] = {}
    for group, names in required.items():
        if set(comparisons[group]) != set(names):
            raise ValueError(f"{group} Stage1 segment set differs")
        deltas: dict[str, float] = {}
        for name in names:
            record = comparisons[group][name]
            delta = float(record["candidate"]["score"]) - float(record["baseline"]["score"])
            if float(record["delta"]) != delta:
                raise ValueError("Stage1 delta arithmetic differs")
            deltas[name] = delta
        group_passed = all(value > 0.0 for value in deltas.values())
        audit[group] = {
            "deltas": deltas,
            "minimum": min(deltas.values()),
            "mean": float(np.mean(list(deltas.values()))),
            "all_strictly_positive": group_passed,
        }
        (passed if group_passed else failed).append(group)
    return tuple(passed), tuple(failed), audit


def stage2_promoted(
    group_comparisons: Mapping[str, Mapping[str, Any]],
    mixed_comparisons: Mapping[str, Mapping[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    """Apply the frozen G1/G2 group, mixed total, and full-component gate."""

    if tuple(group_comparisons) != PROMOTED_GROUPS:
        raise ValueError("Stage2 group set/order differs")
    group_deltas: dict[str, float] = {}
    for group in PROMOTED_GROUPS:
        if set(group_comparisons[group]) != set(SEGMENTS):
            raise ValueError(f"{group} Stage2 segments differ")
        for name in SEGMENTS:
            record = group_comparisons[group][name]
            delta = float(record["candidate"]["score"]) - float(record["baseline"]["score"])
            if float(record["delta"]) != delta:
                raise ValueError("Stage2 group delta arithmetic differs")
            group_deltas[f"{group}/{name}"] = delta
    if set(mixed_comparisons) != set(SEGMENTS):
        raise ValueError("mixed Stage2 segments differ")
    mixed_total_deltas: dict[str, float] = {}
    for name in SEGMENTS:
        record = mixed_comparisons[name]
        delta = float(record["candidate"]["total_score"]) - float(
            record["baseline"]["total_score"]
        )
        if float(record["delta_total_score"]) != delta:
            raise ValueError("mixed total delta arithmetic differs")
        mixed_total_deltas[name] = delta
    full = mixed_comparisons["full"]
    nmae_component = float(full["candidate"]["one_minus_nmae"]) - float(
        full["baseline"]["one_minus_nmae"]
    )
    ficr_component = float(full["candidate"]["ficr"]) - float(
        full["baseline"]["ficr"]
    )
    if float(full["delta_one_minus_nmae"]) != nmae_component:
        raise ValueError("mixed full NMAE component arithmetic differs")
    if float(full["delta_ficr"]) != ficr_component:
        raise ValueError("mixed full FICR component arithmetic differs")
    group_pass = all(value > 0.0 for value in group_deltas.values())
    mixed_pass = all(value > 0.0 for value in mixed_total_deltas.values())
    components_pass = nmae_component >= 0.0 and ficr_component >= 0.0
    promoted = group_pass and mixed_pass and components_pass
    return promoted, {
        "group_slice_count": len(group_deltas),
        "mixed_slice_count": len(mixed_total_deltas),
        "group_deltas": group_deltas,
        "mixed_total_deltas": mixed_total_deltas,
        "full_delta_one_minus_nmae": nmae_component,
        "full_delta_ficr": ficr_component,
        "all_14_group_deltas_strictly_positive": group_pass,
        "all_7_mixed_total_deltas_strictly_positive": mixed_pass,
        "full_components_nonnegative": components_pass,
        "promoted": promoted,
    }


__all__ = [
    "ACTION_GRID_CF",
    "IDENTITY_GROUPS",
    "PROMOTED_GROUPS",
    "SCALE_FACTOR",
    "SEGMENTS",
    "UTILITY_MARGIN",
    "interpolate_row_utility",
    "selective_scale_candidate",
    "stage1_group_promotions",
    "stage2_promoted",
]

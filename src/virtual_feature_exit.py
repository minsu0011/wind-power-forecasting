"""Score-aware feature taxonomy and robust virtual-score helpers.

The helpers in this module are deliberately model agnostic.  They support the
causal feature-exit study without turning a validation label into a feature:

* every cached weather feature is assigned to one physical family;
* four predeclared exit sets are derived from names only;
* row utility mirrors the official NMAE/FICR decomposition; and
* the virtual score rewards full-block improvement while penalising unstable
  group/season slices.

The virtual score is a validation proxy, never a leaderboard-score estimate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import math
import re
from typing import Any

import numpy as np


PHYSICAL_FAMILIES: tuple[str, ...] = (
    "calendar",
    "lead",
    "static_site",
    "cross_source_wind",
    "quality_geometry",
    "wind",
    "thermodynamic",
    "cloud",
    "precip_snow",
    "radiation",
    "pressure_geopotential",
    "boundary_vertical",
    "surface_static_field",
)

EXIT_FAMILIES: tuple[str, ...] = (
    "run_dynamics",
    "spatial_redundancy",
    "nonwind_atmospheric",
    "cross_source_disagreement",
)

_RUN_TOKEN = re.compile(r"__run_(?:lag|lead|gradient|curvature)_(?:1h|3h)$")
_SPATIAL_OPERATOR = re.compile(
    r"__(?:global_std|global_min|global_max|near4_std|rank[1-4])__"
)
_RANK_DISTANCE = re.compile(r"__rank[1-4]_distance_km$")


def _contains_any(value: str, tokens: Sequence[str]) -> bool:
    lowered = value.lower()
    return any(token.lower() in lowered for token in tokens)


def physical_family(feature_name: str) -> str:
    """Assign one cached feature name to one disjoint physical family."""

    name = str(feature_name)
    lower = name.lower()
    if lower.startswith("time__"):
        if lower in {"time__lead_hours", "time__run_position"}:
            return "lead"
        return "calendar"
    if lower.startswith("site__"):
        return "static_site"
    if lower.startswith("cross__"):
        return "cross_source_wind"
    if _contains_any(lower, ("raw_missing", "distance_km")):
        return "quality_geometry"
    core = _RUN_TOKEN.sub("", name)
    leaf = core.split("__")[-1]
    leaf_lower = leaf.lower()

    radiation_leaves = {
        "heightaboveground_2_swdif",
        "heightaboveground_2_swdir",
        "surface_0_ndnlw",
        "surface_0_ndnsw",
        "surface_0_dlwrf",
        "surface_0_dswrf",
    }
    cloud_leaves = {
        "atmosphere_0_tcc",
        "etc_0_vlcdc",
        "etc_0_hcc",
        "etc_0_lcc",
        "etc_0_mcc",
        "highcloudlayer_0_hcc",
        "lowcloudlayer_0_lcc",
        "middlecloudlayer_0_mcc",
    }
    precip_leaves = {
        "surface_0_snom",
        "surface_0_avg_lsprate",
        "surface_0_lssrate",
        "surface_0_ncpcp",
        "surface_0_prate",
        "surface_0_snol",
        "surface_0_tp",
    }
    if leaf_lower in radiation_leaves:
        return "radiation"
    if leaf_lower in cloud_leaves:
        return "cloud"
    if leaf_lower in precip_leaves:
        return "precip_snow"

    wind_exact = {
        "cube",
        "flow_cos",
        "flow_sin",
        "gust_excess",
        "gust_ratio",
        "hub_ws",
        "shear_alpha_10_100",
        "shear_alpha_10_50",
        "u50_mid",
        "v50_mid",
        "wind_power_density",
        "wind_range50",
        "ws10",
        "ws100",
        "ws50",
        "ws80",
        "ws_pbl",
    }
    wind_height_prefixes = (
        "heightaboveground_100_100u",
        "heightaboveground_100_100v",
        "heightaboveground_10_10u",
        "heightaboveground_10_10v",
        "heightaboveground_50_50mu",
        "heightaboveground_50_50mv",
        "heightaboveground_5_xblws",
        "heightaboveground_5_yblws",
        "heightaboveground_80_u",
        "heightaboveground_80_v",
    )
    wind_level_exact = {
        "isobaricinhpa_500_u",
        "isobaricinhpa_500_v",
        "isobaricinhpa_700_u",
        "isobaricinhpa_700_v",
        "isobaricinhpa_850_u",
        "isobaricinhpa_850_v",
        "planetaryboundarylayer_0_u",
        "planetaryboundarylayer_0_v",
        "surface_0_gust",
    }
    if (
        leaf_lower in wind_exact
        or leaf_lower in wind_level_exact
        or leaf_lower.startswith(wind_height_prefixes)
    ):
        return "wind"

    thermo_leaves = {
        "air_density",
        "dewpoint_depression",
        "heightaboveground_2_2d",
        "heightaboveground_2_2r",
        "heightaboveground_2_2sh",
        "heightaboveground_2_2t",
        "heightaboveground_2_dpt",
        "heightaboveground_2_q",
        "heightaboveground_2_r",
        "heightaboveground_2_t",
        "isobaricinhpa_500_t",
        "isobaricinhpa_700_t",
        "isobaricinhpa_850_r",
        "isobaricinhpa_850_t",
    }
    if leaf_lower in thermo_leaves:
        return "thermodynamic"
    if leaf_lower in {"isobaricinhpa_500_gh", "meansea_0_prmsl", "surface_0_sp"}:
        return "pressure_geopotential"
    if leaf_lower in {"etc_0_blh", "planetaryboundarylayer_0_vrate"}:
        return "boundary_vertical"
    if leaf_lower in {"surface_0_h", "surface_0_lsm"}:
        return "surface_static_field"
    raise ValueError(f"unclassified cached feature: {feature_name!r}")


def exit_membership(feature_name: str) -> dict[str, bool]:
    """Return membership in the four predeclared feature-exit sets."""

    name = str(feature_name)
    family = physical_family(name)
    run_dynamics = bool(_RUN_TOKEN.search(name)) or name in {
        "time__lead_hours",
        "time__run_position",
    }
    spatial_redundancy = bool(
        _SPATIAL_OPERATOR.search(name) or _RANK_DISTANCE.search(name)
    )
    nonwind_atmospheric = family in {
        "thermodynamic",
        "cloud",
        "precip_snow",
        "radiation",
        "pressure_geopotential",
        "boundary_vertical",
        "surface_static_field",
    }
    return {
        "run_dynamics": run_dynamics,
        "spatial_redundancy": spatial_redundancy,
        "nonwind_atmospheric": nonwind_atmospheric,
        "cross_source_disagreement": name.startswith("cross__"),
    }


def build_feature_taxonomy(feature_names: Sequence[str]) -> list[dict[str, Any]]:
    """Build deterministic taxonomy records in original feature order."""

    records: list[dict[str, Any]] = []
    for position, feature in enumerate(feature_names):
        membership = exit_membership(feature)
        records.append(
            {
                "position": position,
                "feature": str(feature),
                "physical_family": physical_family(feature),
                **membership,
            }
        )
    return records


def kept_features(
    feature_names: Sequence[str], exit_family: str | None
) -> tuple[str, ...]:
    """Return ordered columns after one declared family exit."""

    names = tuple(str(value) for value in feature_names)
    if exit_family is None or exit_family == "control_all":
        return names
    if exit_family not in EXIT_FAMILIES:
        raise ValueError(f"unknown exit family: {exit_family!r}")
    kept = tuple(
        name for name in names if not exit_membership(name)[exit_family]
    )
    if not kept or len(kept) == len(names):
        raise ValueError(
            f"exit family {exit_family!r} removed zero or all features"
        )
    return kept


def payment_units(error_fraction: np.ndarray) -> np.ndarray:
    """Return official 4/3/0 settlement units for absolute errors."""

    error = np.asarray(error_fraction, dtype=np.float64)
    return np.select([error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0)


def exact_group_row_utility(
    actual_kwh: np.ndarray,
    prediction_kwh: np.ndarray,
    capacity_kwh: float,
) -> np.ndarray:
    """Return row utilities whose eligible-row mean equals group score.

    Ineligible rows are NaN.  For eligible rows the mean of this vector is
    exactly ``0.5 * (1-NMAE) + 0.5 * FICR`` up to floating-point summation.
    """

    actual = np.asarray(actual_kwh, dtype=np.float64)
    prediction = np.asarray(prediction_kwh, dtype=np.float64)
    if actual.shape != prediction.shape or actual.ndim != 1:
        raise ValueError("actual and prediction must be aligned 1-D arrays")
    capacity = float(capacity_kwh)
    valid = np.isfinite(actual) & (actual >= 0.10 * capacity)
    if not np.any(valid):
        raise ValueError("no eligible rows")
    if not np.all(np.isfinite(prediction[valid])):
        raise ValueError("non-finite prediction on eligible row")
    output = np.full(actual.shape, np.nan, dtype=np.float64)
    error = np.abs(prediction[valid] - actual[valid]) / capacity
    units = payment_units(error)
    mean_actual = float(np.mean(actual[valid]))
    output[valid] = (
        0.5 * (1.0 - error)
        + 0.5 * (actual[valid] / mean_actual) * (units / 4.0)
    )
    return output


def correlation_targets(
    actual_kwh: np.ndarray,
    baseline_kwh: np.ndarray,
    capacity_kwh: float,
    *,
    action_fraction: float = 0.01,
) -> dict[str, np.ndarray]:
    """Build score-aware targets around a frozen baseline prediction."""

    actual = np.asarray(actual_kwh, dtype=np.float64)
    baseline = np.asarray(baseline_kwh, dtype=np.float64)
    capacity = float(capacity_kwh)
    valid = np.isfinite(actual) & (actual >= 0.10 * capacity)
    residual = (actual - baseline) / capacity
    error = np.abs(residual)
    plus = baseline + float(action_fraction) * capacity
    minus = baseline - float(action_fraction) * capacity
    base_u = exact_group_row_utility(actual, baseline, capacity)
    plus_u = exact_group_row_utility(actual, plus, capacity)
    minus_u = exact_group_row_utility(actual, minus, capacity)
    direction = plus_u - minus_u
    best_gain = np.maximum(plus_u, minus_u) - base_u
    result = {
        "eligible": valid.astype(np.float64),
        "actual_cf": actual / capacity,
        "signed_residual_cf": residual,
        "absolute_error_cf": error,
        "within_6": (valid & (error <= 0.06)).astype(np.float64),
        "within_8": (valid & (error <= 0.08)).astype(np.float64),
        "margin_6": 0.06 - error,
        "margin_8": 0.08 - error,
        "plus_vs_minus_utility": direction,
        "best_one_percent_utility_gain": best_gain,
    }
    for key, values in result.items():
        values = np.asarray(values, dtype=np.float64)
        if key != "eligible":
            values = values.copy()
            values[~valid] = np.nan
        result[key] = values
    return result


@dataclass(frozen=True)
class VirtualScore:
    """Selection-adjusted causal proxy on the official score scale."""

    baseline_score: float
    raw_delta: float
    simultaneous_lcb: float
    minimum_cell_delta: float
    delta_ficr: float
    delta_one_minus_nmae: float
    instability_penalty: float
    risk_adjusted_increment: float
    local_virtual_score: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def risk_adjusted_virtual_score(
    *,
    baseline_score: float,
    raw_delta: float,
    simultaneous_lcb: float,
    cell_deltas: Sequence[float],
    delta_ficr: float,
    delta_one_minus_nmae: float,
) -> VirtualScore:
    """Compute the frozen max-T and stability-adjusted virtual score.

    ``simultaneous_lcb`` must be computed across every candidate ever tried in
    the preregistered family.  Public feedback is intentionally absent from
    this selection primitive; a report-only anchor may be added by the caller
    only after selection has terminated.
    """

    deltas = np.asarray(tuple(cell_deltas), dtype=np.float64)
    if deltas.ndim != 1 or not len(deltas) or not np.all(np.isfinite(deltas)):
        raise ValueError("cell_deltas must be finite and non-empty")
    numeric = (
        float(baseline_score),
        float(raw_delta),
        float(simultaneous_lcb),
        float(delta_ficr),
        float(delta_one_minus_nmae),
    )
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("virtual-score inputs must be finite")
    minimum_delta = float(np.min(deltas))
    penalty = (
        max(0.0, -minimum_delta)
        + 0.5 * max(0.0, -float(delta_ficr))
        + 0.5 * max(0.0, -(float(delta_one_minus_nmae) + 0.0002))
    )
    increment = float(simultaneous_lcb) - penalty
    return VirtualScore(
        baseline_score=float(baseline_score),
        raw_delta=float(raw_delta),
        simultaneous_lcb=float(simultaneous_lcb),
        minimum_cell_delta=minimum_delta,
        delta_ficr=float(delta_ficr),
        delta_one_minus_nmae=float(delta_one_minus_nmae),
        instability_penalty=penalty,
        risk_adjusted_increment=increment,
        local_virtual_score=float(baseline_score) + increment,
    )


def fixed_paired_increment(
    baseline_kwh: np.ndarray,
    exited_cf: np.ndarray,
    control_cf: np.ndarray,
    capacity_kwh: float,
    *,
    weight: float = 0.25,
) -> np.ndarray:
    """Apply a pure feature-exit increment to an immutable baseline."""

    baseline = np.asarray(baseline_kwh, dtype=np.float64)
    exited = np.asarray(exited_cf, dtype=np.float64)
    control = np.asarray(control_cf, dtype=np.float64)
    if baseline.shape != exited.shape or baseline.shape != control.shape:
        raise ValueError("baseline, exited_cf and control_cf must align")
    candidate = baseline + float(weight) * float(capacity_kwh) * (exited - control)
    return np.clip(candidate, 0.0, 1.02 * float(capacity_kwh))


__all__ = [
    "EXIT_FAMILIES",
    "PHYSICAL_FAMILIES",
    "VirtualScore",
    "build_feature_taxonomy",
    "correlation_targets",
    "exact_group_row_utility",
    "exit_membership",
    "fixed_paired_increment",
    "kept_features",
    "payment_units",
    "physical_family",
    "risk_adjusted_virtual_score",
]

"""Target-free dynamic features from the frozen 2019 Global Wind Atlas v3."""

from __future__ import annotations

import numpy as np
import pandas as pd


RESOURCE_COLUMNS = (
    "gwa3_ws100_ms",
    "gwa3_weibull_k100",
    "gwa3_weibull_a100_ms",
    "gwa3_power_density100_wm2",
    "gwa3_rix",
    "gwa3_cf_iec1",
    "gwa3_cf_iec2",
    "gwa3_cf_iec3",
)
RESOURCE_NAMES = tuple(name.removeprefix("gwa3_") for name in RESOURCE_COLUMNS)
SOURCE_NAMES = ("ldaps", "gfs")
ALIGNMENT_COLUMNS = tuple(
    f"gwa3__{source}__alongflow_alignment__{name}"
    for source in SOURCE_NAMES
    for name in RESOURCE_NAMES
)
PHYSICS_COLUMNS = tuple(
    f"gwa3__{source}__resource_curve__{weight}"
    for source in SOURCE_NAMES
    for weight in ("capacity", "iec1", "iec2", "iec3")
)
EXTENDED_COLUMNS = ALIGNMENT_COLUMNS + PHYSICS_COLUMNS

LDAPS_VECTOR = (
    "ldaps__idw__heightAboveGround_50_50MUmax",
    "ldaps__idw__heightAboveGround_50_50MUmin",
    "ldaps__idw__heightAboveGround_50_50MVmax",
    "ldaps__idw__heightAboveGround_50_50MVmin",
)
GFS_VECTOR = (
    "gfs__idw__heightAboveGround_100_100u",
    "gfs__idw__heightAboveGround_100_100v",
)
HUB_WS = {"ldaps": "ldaps__idw__hub_ws", "gfs": "gfs__idw__hub_ws"}


def _weighted_standardize(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    centered = values - float(values @ weights)
    scale = float(np.sqrt(np.maximum((centered * centered) @ weights, 0.0)))
    return centered / max(scale, 1e-12)


def _positions_km(table: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
    lat0 = float(table["latitude"].to_numpy(float) @ weights)
    lon0 = float(table["longitude"].to_numpy(float) @ weights)
    x = (table["longitude"].to_numpy(float) - lon0) * 111.320 * np.cos(np.radians(lat0))
    y = (table["latitude"].to_numpy(float) - lat0) * 110.574
    return np.column_stack((x, y))


def _vectors(control: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    required = (*LDAPS_VECTOR, *GFS_VECTOR, *HUB_WS.values())
    missing = sorted(set(required).difference(control.columns))
    if missing:
        raise KeyError(f"missing control columns: {missing}")
    ld = control.loc[:, list(LDAPS_VECTOR)].to_numpy(np.float64)
    gf = control.loc[:, list(GFS_VECTOR)].to_numpy(np.float64)
    return {
        "ldaps": (0.5 * (ld[:, 0] + ld[:, 1]), 0.5 * (ld[:, 2] + ld[:, 3])),
        "gfs": (gf[:, 0], gf[:, 1]),
    }


def build_gwa3_features(
    control: pd.DataFrame, static_table: pd.DataFrame, *, group: str
) -> pd.DataFrame:
    """Build the exactly 24 preregistered causal GWA resource features."""

    required_static = {
        "turbine_id", "group", "capacity_mw", "latitude", "longitude", *RESOURCE_COLUMNS
    }
    missing = sorted(required_static.difference(static_table.columns))
    if missing:
        raise KeyError(f"missing static columns: {missing}")
    table = static_table.loc[static_table["group"] == group].sort_values("turbine_id")
    if table.empty or table.isna().any().any():
        raise ValueError(f"invalid GWA table for {group}")
    capacity = table["capacity_mw"].to_numpy(np.float64)
    weights = capacity / capacity.sum()
    positions = _positions_km(table, weights)
    vectors = _vectors(control)
    output: dict[str, np.ndarray] = {}

    for source in SOURCE_NAMES:
        u, v = vectors[source]
        speed = np.hypot(u, v)
        ux = np.divide(u, speed, out=np.zeros_like(u), where=speed > 1e-8)
        uy = np.divide(v, speed, out=np.zeros_like(v), where=speed > 1e-8)
        along = ux[:, None] * positions[None, :, 0] + uy[:, None] * positions[None, :, 1]
        along -= along @ weights[:, None]
        scale = np.sqrt(np.maximum((along * along) @ weights, 0.0))
        standardized_along = np.divide(
            along, scale[:, None], out=np.zeros_like(along), where=scale[:, None] > 1e-12
        )
        for column, name in zip(RESOURCE_COLUMNS, RESOURCE_NAMES):
            resource = _weighted_standardize(table[column].to_numpy(np.float64), weights)
            output[f"gwa3__{source}__alongflow_alignment__{name}"] = (
                standardized_along * resource[None, :]
            ) @ weights

        base_ws = control[HUB_WS[source]].to_numpy(np.float64)
        atlas_ws = table["gwa3_ws100_ms"].to_numpy(np.float64)
        microsite_ratio = atlas_ws / float(atlas_ws @ weights)
        turbine_ws = np.maximum(base_ws[:, None] * microsite_ratio[None, :], 0.0)
        curve = np.clip((turbine_ws**3 - 27.0) / (1728.0 - 27.0), 0.0, 1.0)
        aggregations = {"capacity": weights}
        for iec in ("iec1", "iec2", "iec3"):
            weighted = weights * table[f"gwa3_cf_{iec}"].to_numpy(np.float64)
            aggregations[iec] = weighted / weighted.sum()
        for name, aggregation_weights in aggregations.items():
            output[f"gwa3__{source}__resource_curve__{name}"] = curve @ aggregation_weights

    result = pd.DataFrame(output, index=control.index).loc[:, list(EXTENDED_COLUMNS)]
    if result.shape[1] != 24 or not np.isfinite(result.to_numpy(np.float64)).all():
        raise AssertionError("GWA feature contract differs")
    return result.astype(np.float32)


__all__ = ["EXTENDED_COLUMNS", "RESOURCE_COLUMNS", "build_gwa3_features"]

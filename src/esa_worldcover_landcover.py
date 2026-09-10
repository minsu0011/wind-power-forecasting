"""Target-free directional ESA WorldCover features.

The static table is built only from ESA WorldCover 2020 v100 and the 17
provided turbine coordinates.  At application time the already-available
LDAPS wind vector selects one frozen upwind sector; no labels, predictions or
post-cutoff observations are accepted by this module.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from src.copernicus_dem_exposure import Turbine


SECTOR_HALF_WIDTH_DEG = 11.25
U_MAX = "ldaps__idw__heightAboveGround_50_50MUmax"
U_MIN = "ldaps__idw__heightAboveGround_50_50MUmin"
V_MAX = "ldaps__idw__heightAboveGround_50_50MVmax"
V_MIN = "ldaps__idw__heightAboveGround_50_50MVmin"
WIND_COLUMNS = (U_MAX, U_MIN, V_MAX, V_MIN)

RING_NAMES = ("r0100_0500m", "r0500_2000m", "r2000_5000m")
CLASS_NAMES = ("tree", "grass", "crop", "built", "bare", "water")
LANDCOVER_VALUE_COLUMNS = tuple(
    f"{name}_fraction__{ring}" for ring in RING_NAMES for name in CLASS_NAMES
)
EXTENDED_COLUMNS = tuple(
    f"worldcover2020__{name}_cw_{stat}"
    for name in LANDCOVER_VALUE_COLUMNS
    for stat in ("mean", "std")
)


def validate_static_table(table: pd.DataFrame, turbines: Sequence[Turbine]) -> None:
    """Validate the frozen 17 turbine x 16 sector lookup identity."""

    required = {
        "turbine_id",
        "group",
        "capacity_mw",
        "sector_index",
        *LANDCOVER_VALUE_COLUMNS,
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise KeyError(f"missing WorldCover columns: {missing}")
    if len(table) != 272 or table["turbine_id"].nunique() != 17:
        raise ValueError("WorldCover lookup must be exactly 17x16")
    expected_ids = {t.turbine_id for t in turbines}
    if set(table["turbine_id"].astype(int)) != expected_ids:
        raise ValueError("WorldCover turbine identities differ")
    for turbine_id, frame in table.groupby("turbine_id", sort=False):
        if tuple(sorted(frame["sector_index"].astype(int))) != tuple(range(16)):
            raise ValueError(f"sector identity differs for turbine {turbine_id}")
    values = table.loc[:, list(LANDCOVER_VALUE_COLUMNS)].to_numpy(np.float64)
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("invalid WorldCover fractions")
    for ring in RING_NAMES:
        columns = [f"{name}_fraction__{ring}" for name in CLASS_NAMES]
        if not np.allclose(table[columns].sum(axis=1), 1.0, atol=1e-7):
            raise ValueError(f"class fractions do not sum to one for {ring}")


def build_dynamic_landcover_features(
    control: pd.DataFrame,
    static_table: pd.DataFrame,
    turbines: Sequence[Turbine],
    *,
    group: str,
) -> pd.DataFrame:
    """Select the causal upwind sector and capacity-weight across turbines."""

    missing = [column for column in WIND_COLUMNS if column not in control.columns]
    if missing:
        raise KeyError(f"missing wind columns: {missing}")
    validate_static_table(static_table, turbines)
    wind = control.loc[:, list(WIND_COLUMNS)].to_numpy(np.float64)
    if not np.isfinite(wind).all():
        raise ValueError("nonfinite direction input")
    u = 0.5 * (wind[:, 0] + wind[:, 1])
    v = 0.5 * (wind[:, 2] + wind[:, 3])
    # u/v encode flow-to direction; the lookup bearing is the upwind source.
    source_bearing = np.mod(np.degrees(np.arctan2(-u, -v)), 360.0)
    sectors = np.floor((source_bearing + SECTOR_HALF_WIDTH_DEG) / 22.5).astype(np.int64) % 16

    group_turbines = [t for t in turbines if t.group == group]
    if not group_turbines:
        raise ValueError(f"no turbines for {group}")
    weights = np.asarray([t.capacity_mw for t in group_turbines], dtype=np.float64)
    weights /= weights.sum()
    lookup = static_table.set_index(["turbine_id", "sector_index"])
    arrays: dict[str, np.ndarray] = {}
    for value_column in LANDCOVER_VALUE_COLUMNS:
        selected = np.column_stack(
            [
                lookup.loc[(t.turbine_id, slice(None)), value_column]
                .sort_index()
                .to_numpy(np.float64)[sectors]
                for t in group_turbines
            ]
        )
        mean = selected @ weights
        variance = ((selected - mean[:, None]) ** 2) @ weights
        arrays[f"worldcover2020__{value_column}_cw_mean"] = mean
        arrays[f"worldcover2020__{value_column}_cw_std"] = np.sqrt(
            np.maximum(variance, 0.0)
        )
    result = pd.DataFrame(arrays, index=control.index).loc[:, list(EXTENDED_COLUMNS)]
    if not np.isfinite(result.to_numpy(np.float64)).all():
        raise AssertionError("nonfinite dynamic WorldCover features")
    return result.astype(np.float32)


__all__ = [
    "CLASS_NAMES",
    "EXTENDED_COLUMNS",
    "LANDCOVER_VALUE_COLUMNS",
    "RING_NAMES",
    "WIND_COLUMNS",
    "build_dynamic_landcover_features",
    "validate_static_table",
]

from __future__ import annotations

import numpy as np
import pandas as pd

from src.copernicus_dem_exposure import Turbine
from src.esa_worldcover_landcover import (
    EXTENDED_COLUMNS,
    LANDCOVER_VALUE_COLUMNS,
    WIND_COLUMNS,
    build_dynamic_landcover_features,
)


def _table() -> pd.DataFrame:
    rows = []
    for turbine_id in range(1, 18):
        for sector in range(16):
            row = {
                "turbine_id": turbine_id,
                "group": "kpx_group_1",
                "capacity_mw": 3.6,
                "sector_index": sector,
            }
            for column in LANDCOVER_VALUE_COLUMNS:
                row[column] = 1.0 if column.startswith("tree_") else 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def test_dynamic_shape_order_and_finite() -> None:
    index = pd.date_range("2023-01-01", periods=3, freq="h")
    control = pd.DataFrame(0.0, index=index, columns=WIND_COLUMNS)
    control.loc[:, WIND_COLUMNS[0]] = 1.0
    control.loc[:, WIND_COLUMNS[1]] = 1.0
    turbines = tuple(
        Turbine(i, "kpx_group_1", 3.6, 37.28, 128.95) for i in range(1, 18)
    )
    result = build_dynamic_landcover_features(
        control, _table(), turbines, group="kpx_group_1"
    )
    assert tuple(result.columns) == EXTENDED_COLUMNS
    assert result.shape == (3, 36)
    assert np.isfinite(result.to_numpy()).all()
    tree = [column for column in result if "tree_fraction" in column and column.endswith("_mean")]
    assert np.allclose(result[tree], 1.0)

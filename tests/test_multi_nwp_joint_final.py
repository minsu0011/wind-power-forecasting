from __future__ import annotations

import pandas as pd

from scripts.run_multi_nwp_joint_g12_final import (
    ACTIVE_GROUPS,
    IDENTITY_GROUP,
    FIT_INDEX,
    MODEL_APPLY_INDEX,
    SOURCE_2025_PATHS,
    TEST_INDEX,
)


def test_final_calendar_and_group_contract() -> None:
    assert ACTIVE_GROUPS == ("kpx_group_1", "kpx_group_2")
    assert IDENTITY_GROUP == "kpx_group_3"
    assert FIT_INDEX[0] == pd.Timestamp("2024-03-09 00:00")
    assert FIT_INDEX[-1] == pd.Timestamp("2024-12-31 23:00")
    assert len(TEST_INDEX) == 8760
    assert len(MODEL_APPLY_INDEX) == 8759
    assert TEST_INDEX[-1] == pd.Timestamp("2026-01-01 00:00")


def test_compatible_source_archive_names_are_frozen() -> None:
    assert SOURCE_2025_PATHS["ecmwf"].name == "ecmwf_ifs025_group_centroids_2025.parquet"
    assert SOURCE_2025_PATHS["icon"].name == "icon_global_group_centroids_2025.parquet"
    assert SOURCE_2025_PATHS["gfs"].name == "gfs_global_group_centroids_2025.parquet"

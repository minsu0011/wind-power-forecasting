from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from scripts.run_full_feature_year_qm import (
    CANDIDATES,
    DELTA_WEIGHT,
    MAPPED_COLUMN_COUNT,
    PREREGISTER_SHA256,
    STAGE1_SLICES,
    STAGE2_SLICES,
    _candidate_values,
    _verify_contract,
)
from src.manifest import sha256_file


ROOT = Path(__file__).resolve().parents[1]


class FullFeatureYearQuantileMapTests(unittest.TestCase):
    def test_preregister_and_exact_mapped_column_list_are_pinned(self) -> None:
        preregister = ROOT / "configs/full_feature_year_qm_preregister.json"
        columns = ROOT / "configs/full_qm_mapped_columns.txt"
        config, mapped = _verify_contract(preregister, columns)
        self.assertEqual(sha256_file(preregister), PREREGISTER_SHA256)
        self.assertEqual(len(mapped), MAPPED_COLUMN_COUNT)
        self.assertEqual(config["fixed_predictions_and_candidates"]["candidate_count"], 6)

    def test_candidate_formulas_are_exact(self) -> None:
        index = pd.date_range("2023-01-01 01:00", periods=3, freq="h")
        frame = pd.DataFrame(
            {
                "raw_l1": [100.0, 200.0, 300.0],
                "mapped_l1": [140.0, 160.0, 320.0],
                "raw_q07": [120.0, 220.0, 320.0],
                "mapped_q07": [160.0, 180.0, 360.0],
            },
            index=index,
        )
        frame["raw_mean"] = 0.5 * (frame["raw_l1"] + frame["raw_q07"])
        frame["mapped_mean"] = 0.5 * (
            frame["mapped_l1"] + frame["mapped_q07"]
        )
        baseline = pd.Series([1000.0, 1000.0, 1000.0], index=index)
        baseline.attrs["capacity_kwh"] = 21600.0
        expected = baseline + DELTA_WEIGHT * (
            frame["mapped_mean"] - frame["raw_mean"]
        )
        observed = _candidate_values("v3_delta_mean_w025", frame, baseline)
        np.testing.assert_array_equal(observed.to_numpy(), expected.to_numpy())
        np.testing.assert_array_equal(
            _candidate_values("mapped_l1_absolute", frame, baseline).to_numpy(),
            frame["mapped_l1"].to_numpy(),
        )
        self.assertEqual(len(CANDIDATES), 6)

    def test_frozen_slices_partition_application_windows(self) -> None:
        for group in ("kpx_group_1", "kpx_group_2"):
            full = STAGE1_SLICES[group]["full"]
            h1 = STAGE1_SLICES[group]["H1"]
            h2 = STAGE1_SLICES[group]["H2"]
            self.assertEqual(h1[0], full[0])
            self.assertEqual(h2[1], full[1])
            self.assertEqual(h1[1] + pd.Timedelta(hours=1), h2[0])
        self.assertEqual(
            STAGE2_SLICES["H1"][1] + pd.Timedelta(hours=1),
            STAGE2_SLICES["H2"][0],
        )
        self.assertEqual(
            sum(
                len(pd.date_range(*STAGE2_SLICES[name], freq="h"))
                for name in ("Q1", "Q2", "Q3", "Q4")
            ),
            len(pd.date_range(*STAGE2_SLICES["full"], freq="h")),
        )


if __name__ == "__main__":
    unittest.main()

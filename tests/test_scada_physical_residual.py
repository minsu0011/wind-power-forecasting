import hashlib
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_scada_physical_residual import (
    PREREGISTER_SHA256,
    _all_slices,
    residual_blend,
    select_stage1_candidate,
    shift_hedged_application_features,
    stage1_candidate_passes,
    stage2_candidate_passes,
)


class ScadaPhysicalResidualTests(unittest.TestCase):
    def test_preregister_is_immutable(self):
        path = Path("configs/scada_physical_residual_preregister.json")
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(observed, PREREGISTER_SHA256)

    def test_shift_hedge_is_half_strength_and_clipped(self):
        index = pd.date_range("2022-01-01", periods=4, freq="h")
        train = pd.DataFrame(
            {
                "ldaps__idw__hub_ws": [8.0, 10.0, 10.0, 12.0],
                "gfs__idw__hub_ws": [3.0, 4.0, 4.0, 5.0],
            },
            index=index,
        )
        application = train * 1.21
        adjusted, audit = shift_hedged_application_features(train, application)
        # Raw 1.21 is clipped to 1.10; the fixed divisor is sqrt(1.10).
        np.testing.assert_allclose(adjusted.to_numpy(), application.to_numpy() / np.sqrt(1.10))
        self.assertEqual(audit["ldaps__idw__hub_ws"]["clipped_ratio"], 1.10)
        self.assertEqual(audit["gfs__idw__hub_ws"]["clipped_ratio"], 1.10)

    def test_residual_is_identity_in_mid_wind(self):
        index = pd.date_range("2023-01-01", periods=5, freq="h")
        baseline = pd.Series([1000.0] * 5, index=index)
        physical = pd.Series([2000.0] * 5, index=index)
        wind = pd.Series([5.0, 6.0, 6.1, 11.9, 12.0], index=index)
        result = residual_blend(
            baseline, physical, wind, "kpx_group_1", weight=0.20
        )
        np.testing.assert_allclose(result.to_numpy(), [1200.0, 1200.0, 1000.0, 1000.0, 1200.0])

    @staticmethod
    def _comparison(delta=0.001, nmae_delta=0.0):
        return {
            name: {
                "delta": delta,
                "one_minus_nmae_delta": nmae_delta,
                "ficr_delta": 2 * delta - nmae_delta,
            }
            for name in ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4")
        }

    def test_stage1_quarter_floor_and_count_are_enforced(self):
        passing = self._comparison()
        self.assertTrue(stage1_candidate_passes("kpx_group_1", passing))
        passing["Q1"]["delta"] = -0.0009
        self.assertTrue(stage1_candidate_passes("kpx_group_1", passing))
        passing["Q2"]["delta"] = -0.0009
        self.assertFalse(stage1_candidate_passes("kpx_group_1", passing))
        passing["Q2"]["delta"] = 0.001
        passing["Q1"]["delta"] = -0.0011
        self.assertFalse(stage1_candidate_passes("kpx_group_1", passing))

    def test_stage1_ranking_prefers_minimum_core_then_full_then_weight(self):
        weak = self._comparison(delta=0.001)
        strong = self._comparison(delta=0.002)
        candidates = {"extreme_w10": weak, "extreme_w20": strong}
        passing, selected = select_stage1_candidate("kpx_group_1", candidates)
        self.assertEqual(passing, ["extreme_w10", "extreme_w20"])
        self.assertEqual(selected, "extreme_w20")

    def test_stage2_rejects_half_reversal_or_too_many_quarter_reversals(self):
        passing = self._comparison()
        self.assertTrue(stage2_candidate_passes(passing))
        passing["H2"]["delta"] = -1e-6
        self.assertFalse(stage2_candidate_passes(passing))
        passing = self._comparison()
        passing["Q1"]["delta"] = -0.0005
        passing["Q2"]["delta"] = -0.0005
        self.assertFalse(stage2_candidate_passes(passing))

    def test_group3_half_year_slice_set_omits_empty_h1(self):
        index = pd.date_range("2023-07-01 01:00:00", "2024-01-01 00:00:00", freq="h")
        slices = _all_slices(index)
        self.assertNotIn("H1", slices)
        self.assertIn("H2", slices)
        self.assertEqual(set(slices), {"full", "H2", "Q3", "Q4"})


if __name__ == "__main__":
    unittest.main()

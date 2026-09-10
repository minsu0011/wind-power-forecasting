from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.run_model_zoo_diversity import _blend, _select_stage1


class ModelZooDiversityProtocolTests(unittest.TestCase):
    @staticmethod
    def _record(*deltas: float):
        return {
            "segments": {
                f"segment_{position}": {"delta_score": delta}
                for position, delta in enumerate(deltas)
            }
        }

    def test_selection_requires_every_segment_strictly_positive(self):
        records = {
            group: {
                "fails_zero": self._record(0.01, 0.0, 0.02),
                "fails_negative": self._record(0.03, -0.001, 0.04),
            }
            for group in ("kpx_group_1", "kpx_group_2", "kpx_group_3")
        }
        selection = _select_stage1(records)
        self.assertTrue(
            all(value["selected_candidate"] == "identity" for value in selection.values())
        )

    def test_selection_maximises_minimum_delta_then_lexical_tie(self):
        candidates = {
            "z_model": self._record(0.02, 0.01, 0.03),
            "b_model": self._record(0.01, 0.03, 0.02),
            "a_model": self._record(0.01, 0.02, 0.04),
            "weak_model": self._record(0.005, 0.05, 0.05),
        }
        records = {
            group: candidates
            for group in ("kpx_group_1", "kpx_group_2", "kpx_group_3")
        }
        selection = _select_stage1(records)
        self.assertTrue(
            all(value["selected_candidate"] == "a_model" for value in selection.values())
        )

    def test_blend_uses_raw_candidate_then_one_post_blend_clip(self):
        index = pd.date_range("2023-01-01", periods=2, freq="h")
        locked = pd.Series([1000.0, 100.0], index=index)
        config = {
            "blend": {"locked_v3_weight": 0.9, "candidate_weight": 0.1}
        }
        raw, blended = _blend(
            locked,
            np.array([2.0, -1.0]),
            capacity=1000.0,
            config=config,
        )
        np.testing.assert_allclose(raw.to_numpy(), [2000.0, -1000.0])
        # 0.9*1000 + 0.1*2000 = 1100, clipped once to 1.02C.
        # 0.9*100 + 0.1*(-1000) = -10, clipped once to zero.
        np.testing.assert_allclose(blended.to_numpy(), [1020.0, 0.0])


if __name__ == "__main__":
    unittest.main()

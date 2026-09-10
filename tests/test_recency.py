from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.recency import exponential_time_weights, select_stable_half_life


class RecencyTests(unittest.TestCase):
    def test_weights_are_past_only_monotone_and_mean_one(self):
        index = pd.date_range("2023-01-01", periods=20, freq="D")
        weights = exponential_time_weights(
            index, training_end=index[-1], half_life_days=10
        )
        self.assertTrue(np.all(np.diff(weights) > 0))
        self.assertAlmostEqual(float(np.mean(weights)), 1.0)
        with self.assertRaisesRegex(ValueError, "after training_end"):
            exponential_time_weights(
                index, training_end=index[-2], half_life_days=10
            )

    def test_selection_requires_every_segment_to_improve(self):
        deltas = {
            "90": {"a": 0.1, "b": -0.01},
            "180": {"a": 0.02, "b": 0.03},
        }
        self.assertEqual(
            select_stable_half_life(
                deltas, half_lives=(90, 180), required_segments=("a", "b")
            ),
            180,
        )
        self.assertIsNone(
            select_stable_half_life(
                {"90": deltas["90"]},
                half_lives=(90,),
                required_segments=("a", "b"),
            )
        )


if __name__ == "__main__":
    unittest.main()

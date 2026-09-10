import unittest

import numpy as np
import pandas as pd

from src.temporal import smooth_within_runs


class WithinRunSmoothingTests(unittest.TestCase):
    def test_smoothing_does_not_cross_operating_run_boundary(self):
        index = pd.date_range("2025-01-01 01:00", periods=48, freq="h")
        values = np.r_[np.arange(24, dtype=float), 100 + np.arange(24, dtype=float)]
        frame = pd.DataFrame({"group": values}, index=index)
        result = smooth_within_runs(frame, {"group": 0.2})

        self.assertAlmostEqual(result.iloc[0, 0], 0.1)
        self.assertAlmostEqual(result.iloc[23, 0], 22.9)
        self.assertAlmostEqual(result.iloc[24, 0], 100.1)
        self.assertAlmostEqual(result.iloc[47, 0], 122.9)

    def test_incomplete_run_is_rejected(self):
        index = pd.date_range("2025-01-01 01:00", periods=23, freq="h")
        frame = pd.DataFrame({"group": np.arange(23, dtype=float)}, index=index)
        with self.assertRaisesRegex(ValueError, "expected 24"):
            smooth_within_runs(frame, {"group": 0.1})


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.submission_composition import TARGET_COLUMNS, transfer_prediction_delta


class SubmissionCompositionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = pd.date_range("2025-01-01 01:00", periods=3, freq="h")
        self.target = pd.DataFrame(
            {
                "kpx_group_1": [100.0, 21_900.0, 400.0],
                "kpx_group_2": [200.0, 300.0, 400.0],
                "kpx_group_3": [300.0, 400.0, 500.0],
            },
            index=self.index,
        )
        self.reference = self.target.copy()
        self.adjusted = self.reference.copy()
        self.adjusted["kpx_group_1"] += np.array([50.0, 1_000.0, -900.0])

    def test_transfers_only_selected_group_and_clips(self) -> None:
        result = transfer_prediction_delta(
            self.target, self.reference, self.adjusted, groups=["kpx_group_1"]
        )
        np.testing.assert_allclose(result["kpx_group_1"], [150.0, 22_032.0, 0.0])
        for group in TARGET_COLUMNS[1:]:
            np.testing.assert_array_equal(result[group], self.target[group])

    def test_does_not_mutate_inputs(self) -> None:
        before = self.target.copy(deep=True)
        transfer_prediction_delta(
            self.target, self.reference, self.adjusted, groups=["kpx_group_1"]
        )
        pd.testing.assert_frame_equal(self.target, before)

    def test_rejects_index_mismatch(self) -> None:
        shifted = self.adjusted.copy()
        shifted.index = shifted.index + pd.Timedelta(hours=1)
        with self.assertRaisesRegex(ValueError, "indices differ"):
            transfer_prediction_delta(
                self.target, self.reference, shifted, groups=["kpx_group_1"]
            )

    def test_rejects_unknown_group_and_nonfinite(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown target groups"):
            transfer_prediction_delta(
                self.target, self.reference, self.adjusted, groups=["missing"]
            )
        bad = self.adjusted.copy()
        bad.iloc[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            transfer_prediction_delta(
                self.target, self.reference, bad, groups=["kpx_group_1"]
            )


if __name__ == "__main__":
    unittest.main()

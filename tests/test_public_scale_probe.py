from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.metric import score_details
from src.public_scale_probe import (
    fit_oof_scale_to_public,
    scale_predictions,
    validate_public_triplet,
)


class PublicScaleProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = pd.date_range(
            "2024-01-01 01:00", periods=6, freq="h", name="forecast_kst_dtm"
        )
        self.prediction = pd.DataFrame(
            {
                "kpx_group_1": [2_200.0, 4_000.0, 8_000.0, 12_000.0, 20_000.0, 23_000.0],
                "kpx_group_2": [2_400.0, 4_200.0, 8_200.0, 12_200.0, 20_200.0, 23_000.0],
                "kpx_group_3": [2_300.0, 4_100.0, 8_100.0, 12_100.0, 20_100.0, 23_000.0],
            },
            index=self.index,
        )

    def test_scale_is_fixed_nonmutating_and_clipped(self) -> None:
        before = self.prediction.copy(deep=True)
        result = scale_predictions(self.prediction, 0.95)
        pd.testing.assert_frame_equal(self.prediction, before)
        np.testing.assert_allclose(
            result.iloc[:-1].to_numpy(), before.iloc[:-1].to_numpy() * 0.95
        )
        self.assertLessEqual(result["kpx_group_1"].max(), 21_600.0 * 1.02)
        self.assertLessEqual(result["kpx_group_3"].max(), 21_000.0 * 1.02)

    def test_scale_rejects_invalid_values(self) -> None:
        for value in (0.0, -1.0, np.nan, np.inf):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "scale"):
                    scale_predictions(self.prediction, value)

    def test_public_triplet_identity(self) -> None:
        validate_public_triplet(score=0.6, one_minus_nmae=0.8, ficr=0.4)
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            validate_public_triplet(score=0.61, one_minus_nmae=0.8, ficr=0.4)

    def test_grid_fit_recovers_synthetic_scale(self) -> None:
        labels = self.prediction * 1.03
        public = score_details(labels, self.prediction * 1.06)
        result = fit_oof_scale_to_public(
            labels,
            self.prediction,
            {
                "score": public.total_score,
                "one_minus_nmae": public.one_minus_nmae,
                "ficr": public.ficr,
            },
            scale_min=1.04,
            scale_max=1.08,
            scale_step=0.01,
        )
        self.assertAlmostEqual(result["joint_fit"]["scale"], 1.06)
        self.assertAlmostEqual(result["joint_fit"]["component_l2"], 0.0)

    def test_grid_fit_rejects_index_mismatch(self) -> None:
        labels = self.prediction.copy()
        labels.index = labels.index + pd.Timedelta(hours=1)
        with self.assertRaisesRegex(ValueError, "indices differ"):
            fit_oof_scale_to_public(
                labels,
                self.prediction,
                {"score": 1.0, "one_minus_nmae": 1.0, "ficr": 1.0},
                scale_min=0.9,
                scale_max=1.1,
                scale_step=0.1,
            )


if __name__ == "__main__":
    unittest.main()

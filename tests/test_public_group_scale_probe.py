from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.build_public_group_scale_probe import preflight
from src.metric import CAPACITY_KWH, TARGET_COLS, score_details
from src.public_group_scale_probe import (
    macro_separability_residuals,
    scale_one_group,
)
from src.public_scale_probe import scale_predictions


class PublicGroupScaleProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        index = pd.date_range(
            "2024-01-01 01:00", periods=8, freq="h", name="forecast_kst_dtm"
        )
        self.prediction = pd.DataFrame(
            {
                "kpx_group_1": [3_000.0, 4_000.0, 8_000.0, 12_000.0, 18_000.0, 21_000.0, 22_000.0, 23_000.0],
                "kpx_group_2": [3_200.0, 4_200.0, 8_200.0, 12_200.0, 18_200.0, 21_200.0, 22_200.0, 23_200.0],
                "kpx_group_3": [3_100.0, 4_100.0, 8_100.0, 12_100.0, 18_100.0, 20_500.0, 21_500.0, 22_500.0],
            },
            index=index,
            dtype=np.float64,
        )

    def test_scales_only_selected_group_and_does_not_mutate(self) -> None:
        before = self.prediction.copy(deep=True)
        result = scale_one_group(self.prediction, "kpx_group_2", factor=0.92)
        pd.testing.assert_frame_equal(self.prediction, before)
        for group in ("kpx_group_1", "kpx_group_3"):
            self.assertTrue(
                np.array_equal(
                    result[group].to_numpy(dtype=np.float64),
                    before[group].to_numpy(dtype=np.float64),
                )
            )
        np.testing.assert_array_equal(
            result["kpx_group_2"].to_numpy(),
            np.clip(
                before["kpx_group_2"].to_numpy() * 0.92,
                0.0,
                1.02 * CAPACITY_KWH["kpx_group_2"],
            ),
        )

    def test_rejects_invalid_group_and_factors(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown target group"):
            scale_one_group(self.prediction, "unknown")
        for factor in (0.0, -1.0, np.nan, np.inf):
            with self.subTest(factor=factor):
                with self.assertRaisesRegex(ValueError, "factor"):
                    scale_one_group(self.prediction, "kpx_group_1", factor=factor)

    def test_macro_metric_deltas_are_additively_separable(self) -> None:
        labels = self.prediction * pd.Series(
            {"kpx_group_1": 0.94, "kpx_group_2": 1.03, "kpx_group_3": 0.98}
        )

        def compact(frame: pd.DataFrame) -> dict[str, float]:
            result = score_details(labels, frame)
            return {
                "score": result.total_score,
                "one_minus_nmae": result.one_minus_nmae,
                "ficr": result.ficr,
            }

        residuals = macro_separability_residuals(
            base=compact(self.prediction),
            global_scaled=compact(scale_predictions(self.prediction, 0.92)),
            group_only={
                group: compact(scale_one_group(self.prediction, group, factor=0.92))
                for group in TARGET_COLS
            },
        )
        for component in ("score", "one_minus_nmae", "ficr"):
            self.assertAlmostEqual(residuals[component]["residual"], 0.0, places=15)

    def test_nonempty_output_directory_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            (out_dir / "sentinel.txt").write_text("keep", encoding="utf-8")
            names = {group: f"probe_{group}" for group in TARGET_COLS}
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                preflight(out_dir, names)
            self.assertEqual((out_dir / "sentinel.txt").read_text(encoding="utf-8"), "keep")

    def test_protocol_has_no_recommendation_before_global_result(self) -> None:
        config_path = Path("configs/public_group_scale_probe_20260808.json")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        protocol = config["sequential_protocol"]
        self.assertTrue(config["public_adaptive"])
        self.assertTrue(config["selection_unsafe"])
        self.assertFalse(config["private_champion"])
        self.assertFalse(protocol["global_092_result_available_at_build_time"])
        self.assertIsNone(protocol["automatic_next_file_when_global_result_missing"])
        self.assertEqual(
            protocol["if_gate_passes_group_only_submission_order"],
            ["kpx_group_3", "kpx_group_1"],
        )
        self.assertEqual(protocol["inferred_unsubmitted_group"], "kpx_group_2")
        self.assertFalse(config["prepublic_group_order_evidence"]["public_results_used"])


if __name__ == "__main__":
    unittest.main()

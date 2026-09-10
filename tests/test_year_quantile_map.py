from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from scripts.run_year_quantile_map_audit import _quantile_map
from scripts.run_year_quantile_map_transfer import _gate, _load_label_window_bounded


class YearQuantileMapTests(unittest.TestCase):
    def test_location_shift_maps_to_reference_distribution(self) -> None:
        index = pd.date_range("2025-01-01", periods=5, freq="h")
        reference = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0], index=index)
        application = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0], index=index)

        mapped, audit = _quantile_map(reference, application, grid_size=1001)

        np.testing.assert_allclose(mapped.to_numpy(), reference.to_numpy(), atol=1e-12)
        self.assertTrue(mapped.index.equals(index))
        self.assertAlmostEqual(audit["mapped_mean"], audit["reference_mean"])
        self.assertAlmostEqual(audit["mapped_median"], audit["reference_median"])

    def test_constant_application_is_finite_and_bounded(self) -> None:
        reference = pd.Series([-2.0, -1.0, 0.0, 2.0, 5.0])
        application = pd.Series([3.0, 3.0, 3.0, 3.0])

        mapped, _ = _quantile_map(reference, application, grid_size=101)

        self.assertTrue(np.isfinite(mapped.to_numpy()).all())
        self.assertTrue(mapped.between(reference.min(), reference.max()).all())
        self.assertEqual(mapped.nunique(), 1)

    def test_mapping_is_monotone(self) -> None:
        reference = pd.Series([0.0, 0.5, 2.0, 9.0, 10.0])
        application = pd.Series([100.0, 4.0, 8.0, 2.0, 16.0, 12.0])

        mapped, _ = _quantile_map(reference, application, grid_size=101)
        ordered = pd.DataFrame({"x": application, "mapped": mapped}).sort_values("x")

        self.assertTrue(np.diff(ordered["mapped"].to_numpy()).min() >= -1e-12)

    def test_gate_rejects_one_negative_period(self) -> None:
        periods = {}
        for period in ("full", "H1", "H2"):
            periods[period] = {
                "delta": {
                    "total_score": 0.001,
                    "one_minus_nmae": 0.0,
                    "by_group_total_score": {"kpx_group_1": 0.0},
                }
            }
        periods["H2"]["delta"]["total_score"] = -1e-4

        result = _gate(periods)

        self.assertFalse(result["passed"])
        self.assertFalse(result["all_period_total_deltas_nonnegative"])

    def test_bounded_loader_materializes_no_later_label(self) -> None:
        text = (
            "kst_dtm,kpx_group_1,kpx_group_2,kpx_group_3\n"
            "2022-12-31 23:00:00,1,2,3\n"
            "2023-01-01 00:00:00,4,5,6\n"
            "2023-01-01 01:00:00,7,8,9\n"
            "2024-01-01 00:00:00,10,11,12\n"
            "2024-01-01 01:00:00,SECRET,SECRET,SECRET\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.csv"
            path.write_text(text, encoding="utf-8-sig")
            expected = pd.DatetimeIndex(
                [pd.Timestamp("2023-01-01 00:00:00"), pd.Timestamp("2023-01-01 01:00:00"), pd.Timestamp("2024-01-01 00:00:00")],
                name="kst_dtm",
            )

            frame, audit = _load_label_window_bounded(
                path,
                expected_index=expected,
                groups=["kpx_group_1", "kpx_group_2"],
            )

        self.assertEqual(frame["kpx_group_1"].tolist(), [4.0, 7.0, 10.0])
        self.assertEqual(audit["later_label_values_materialized"], 0)
        self.assertTrue(audit["stopped_at_requested_end"])
        self.assertEqual(audit["max_materialized_timestamp"], "2024-01-01T00:00:00")
        self.assertGreater(audit["source_prefix_bytes_read"], 0)
        self.assertEqual(len(audit["source_prefix_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()

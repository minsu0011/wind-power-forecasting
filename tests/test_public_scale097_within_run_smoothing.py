from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts import run_public_scale097_within_run_smoothing as runner


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/public_scale097_within_run_smoothing_preregister_v1.json"


class PublicScale097WithinRunSmoothingTests(unittest.TestCase):
    def test_preregister_hash_and_literals(self) -> None:
        digest = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
        self.assertEqual(digest, runner.CONFIG_SHA)
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(config["immutable_candidate"]["strengths"], runner.STRENGTHS)
        self.assertEqual(config["immutable_candidate"]["factor"], 0.97)
        self.assertTrue(config["risk_classification"]["selection_unsafe"])
        self.assertTrue(config["risk_classification"]["multiple_testing_across_today_candidates"])

    def test_exact_scale_then_smooth_formula_and_g3_identity(self) -> None:
        index = pd.date_range("2024-01-01 01:00", periods=24, freq="h", name="forecast_kst_dtm")
        base = pd.DataFrame(
            {
                "kpx_group_1": np.arange(1.0, 25.0),
                "kpx_group_2": np.arange(101.0, 125.0),
                "kpx_group_3": np.arange(201.0, 225.0),
            },
            index=index,
        )
        scaled, candidate = runner._candidate(base)
        expected_scaled = 0.97 * base["kpx_group_1"].to_numpy()
        previous = np.concatenate(([expected_scaled[0]], expected_scaled[:-1]))
        following = np.concatenate((expected_scaled[1:], [expected_scaled[-1]]))
        expected = 0.85 * expected_scaled + 0.075 * (previous + following)
        np.testing.assert_array_equal(candidate["kpx_group_1"].to_numpy(), expected)
        np.testing.assert_array_equal(runner._bits(candidate["kpx_group_3"]), runner._bits(scaled["kpx_group_3"]))

    def test_empty_label_cell_parses_as_nan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.csv"
            path.write_text(
                "kst_dtm,kpx_group_1,kpx_group_2,kpx_group_3\n"
                "2024-01-01 01:00:00,100,,300\n"
                "2024-01-01 02:00:00,101,201,301\n",
                encoding="utf-8",
            )
            frame = runner._read_labels(path, 2)
            self.assertTrue(np.isnan(frame.iloc[0]["kpx_group_2"]))
            self.assertEqual(frame.iloc[1]["kpx_group_2"], 201.0)

    def test_candidate_and_mask_lock_precedes_label_parse(self) -> None:
        source = (ROOT / "scripts/run_public_scale097_within_run_smoothing.py").read_text(encoding="utf-8")
        main_source = source[source.index("def main(") :]
        self.assertLess(main_source.index("_write_json(lock_path, lock)"), main_source.index("labels = _read_labels"))
        self.assertIn('"2024_label_value_cells_read": 0', main_source)

    def test_2025_materialization_is_pass_gated(self) -> None:
        source = (ROOT / "scripts/run_public_scale097_within_run_smoothing.py").read_text(encoding="utf-8")
        main_source = source[source.index("def main(") :]
        self.assertIn("final = _final(config, args.out_dir) if passed else None", main_source)
        self.assertIn('"2025_prediction_value_cells_read": 8760 * 3 if passed else 0', main_source)


if __name__ == "__main__":
    unittest.main()

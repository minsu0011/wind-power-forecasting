from __future__ import annotations

import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import run_raw_spatial_grid_temporal_cnn as runner


class RawSpatialGridTemporalCNNRunnerTests(unittest.TestCase):
    def test_preregister_and_dependency_are_exact(self) -> None:
        config, contract = runner.verify_config(runner.CONFIG_PATH)
        self.assertEqual(config["single_candidate"]["count"], 1)
        self.assertEqual(config["single_candidate"]["blend_weight"], 0.025)
        self.assertEqual(config["training"]["learning_rate"], 0.002)
        self.assertIn("grid_catalog", contract)

    def test_source_closure_covers_module_tests_launcher_and_runner(self) -> None:
        closure = runner.source_closure(runner.DEFAULT_RAW_DIR)
        paths = set(closure["resolved_relative_paths"])
        self.assertIn("scripts/run_raw_spatial_grid_temporal_cnn.py", paths)
        self.assertIn("src/raw_spatial_grid_temporal_cnn.py", paths)
        explicit = {Path(row["path"]).name for row in closure["explicit_sources"]}
        self.assertIn("test_raw_spatial_grid_temporal_cnn.py", explicit)
        self.assertIn("test_raw_spatial_grid_temporal_cnn_runner.py", explicit)
        self.assertIn("launch_raw_spatial_grid_temporal_cnn_v1.ps1", explicit)

    def test_g3_reader_is_fail_armed_before_g12_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            absent = Path(directory) / "absent.json"
            with mock.patch.object(runner.raw_protocol, "_read_label_prefix") as reader:
                with self.assertRaises(RuntimeError):
                    runner._read_g3_fit_after_g12_lock(Path(directory), {}, absent)
                reader.assert_not_called()

    def test_score_reader_is_fail_armed_before_global_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            absent = Path(directory) / "absent.json"
            with mock.patch.object(runner.raw_protocol, "_read_label_prefix") as reader:
                with self.assertRaises(RuntimeError):
                    runner._read_score_after_global_lock(Path(directory), {}, absent)
                reader.assert_not_called()

    def test_reader_order_locks_both_fits_before_score_labels(self) -> None:
        source = inspect.getsource(runner.run_stage1)
        self.assertLess(source.index("stage1_g12_candidate_lock_before_application_labels.json"), source.index("_read_g3_fit_after_g12_lock"))
        self.assertLess(source.index("stage1_global_candidate_lock_before_application_labels.json"), source.index("_read_score_after_global_lock"))
        self.assertLess(source.index("_read_score_after_global_lock"), source.index("bayes._comparison("))

    def test_stage1_runner_has_no_conditional_2024_or_2025_reader(self) -> None:
        source = inspect.getsource(runner.run_stage1)
        self.assertNotIn("period=\"train_all\"", source)
        self.assertNotIn("period=\"test\"", source)
        self.assertNotIn("_read_full_labels", source)
        config = json.loads(runner.CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(config["sequence_alignment"]["selected"], "availability_run_01_through_next_day_00")


if __name__ == "__main__":
    unittest.main()

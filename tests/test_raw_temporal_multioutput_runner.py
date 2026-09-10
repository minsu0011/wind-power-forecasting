from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from scripts import run_raw_temporal_multioutput as runner
from src.manifest import sha256_file
from src.metric import TARGET_COLS


class RawTemporalMultiOutputRunnerTests(unittest.TestCase):
    def test_preregister_feasibility_and_dependency_hashes_are_exact(self) -> None:
        path = Path("configs/raw_temporal_multioutput_preregister_v1.json")
        self.assertEqual(sha256_file(path), runner.PREREGISTER_SHA256)
        payload, raw_contract, dependency = runner._verify_preregister(path)
        self.assertEqual(payload["run_input"]["input_dimension"], 800)
        self.assertEqual([item["id"] for item in payload["estimators"]], ["ridge_a100", "pls12"])
        self.assertEqual(sha256_file(dependency), runner.RAW_DEPENDENCY_SHA256)
        self.assertIn("physical_stage1_inputs", raw_contract)
        sources = runner._source_paths(path, dependency)
        self.assertTrue(
            {
                "weather_protocol",
                "run_sequence_guard",
                "temporal_run_key",
                "feature_dependency",
                "weather_quantile_dependency",
                "probabilistic_dependency",
            }.issubset(sources)
        )

    def test_stage1_registered_intervals_are_exact_and_complete_runs(self) -> None:
        self.assertEqual(len(runner.YEAR_2022), 365 * 24)
        self.assertEqual(len(runner.YEAR_2023), 365 * 24)
        self.assertEqual(len(runner.G3_H1), 181 * 24)
        self.assertEqual(len(runner.G3_H2), 184 * 24)
        self.assertEqual(
            tuple(runner._stage1_segments("kpx_group_1")), runner.STAGE1_REQUIRED["kpx_group_1"]
        )
        self.assertEqual(
            tuple(runner._stage1_segments("kpx_group_3")), runner.STAGE1_REQUIRED["kpx_group_3"]
        )

    def test_aggregate_comparison_uses_official_macro_score(self) -> None:
        index = pd.date_range("2024-01-01 01:00:00", periods=24, freq="h", name=runner.TIME_COL)
        actual = pd.DataFrame(
            {
                "kpx_group_1": 9000.0,
                "kpx_group_2": 8500.0,
                "kpx_group_3": 8000.0,
            },
            index=index,
        )
        baseline = actual + 3000.0
        candidate = actual + 1000.0
        result = runner._aggregate_comparison(
            actual, baseline, candidate, {"full": index}
        )["full"]
        exact = (
            float(result["candidate"]["total_score"])
            - float(result["baseline"]["total_score"])
        )
        self.assertEqual(result["delta"], exact)
        self.assertGreater(result["delta"], 0.0)

    def test_stage1_source_orders_each_fit_label_and_global_score_lock(self) -> None:
        source = Path(runner.__file__).read_text(encoding="utf-8")
        block = source[source.index("def _stage1(") : source.index("def _load_stage1_lock(")]
        g12_read = block.index("labels_g12, labels_g12_evidence")
        g12_lock = block.index('g12_lock_path = out_dir / "stage1_g12_prescore_lock.json"')
        g3_read = block.index("labels_g3, labels_g3_evidence")
        g3_lock = block.index('g3_lock_path = out_dir / "stage1_g3_prescore_lock.json"')
        global_lock = block.index('global_lock_path = out_dir / "stage1_global_prescore_lock.json"')
        score_read = block.index("score_labels, score_label_evidence")
        self.assertLess(g12_read, g12_lock)
        self.assertLess(g12_lock, g3_read)
        self.assertLess(g3_read, g3_lock)
        self.assertLess(g3_lock, global_lock)
        self.assertLess(global_lock, score_read)
        self.assertIn("TARGET_COLS[:2]", block[g12_read:g12_lock])
        self.assertIn("(TARGET_COLS[2],)", block[g3_read:g3_lock])

    def test_stage2_source_locks_before_full_2024_labels(self) -> None:
        source = Path(runner.__file__).read_text(encoding="utf-8")
        block = source[source.index("def _stage2(") : source.index("def _load_stage2_lock(")]
        lock = block.index('lock_path = out_dir / "stage2_global_prescore_lock.json"')
        score = block.index("score_labels, score_label_evidence")
        self.assertLess(lock, score)
        self.assertIn("passed_groups", block)
        self.assertIn("mixed_aggregate", block)

    def test_stage2_identity_opens_no_2024_reader(self) -> None:
        stage1_lock = {
            "passed_groups": [],
            "locked_candidates": {group: None for group in TARGET_COLS},
        }
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            (out_dir / "stage1_promotion_lock.json").write_text("{}\n", encoding="utf-8")
            forbidden = AssertionError("a 2024 reader was called")
            with (
                mock.patch.object(runner, "_load_stage1_lock", return_value=(stage1_lock, {})),
                mock.patch.object(runner.raw_protocol, "_read_weather_features", side_effect=forbidden),
                mock.patch.object(runner.raw_protocol, "_read_label_prefix", side_effect=forbidden),
                mock.patch.object(runner.raw_protocol, "_read_full_labels", side_effect=forbidden),
            ):
                lock = runner._stage2(
                    raw_dir=Path("unused"),
                    artifact_root=Path("unused"),
                    out_dir=out_dir,
                    raw_contract={},
                )
            self.assertFalse(lock["2024_read"])
            self.assertFalse(lock["2025_read"])
            self.assertFalse(lock["executed"])
            self.assertEqual(lock["group_gate_passed"], {})
            self.assertFalse(lock["mixed_aggregate_gate_passed"])
            result = json.loads((out_dir / "stage2_results.json").read_text(encoding="utf-8"))
            self.assertFalse(result["2024_weather_read"])
            self.assertFalse(result["2024_label_read"])

    def test_final_nonpromotion_opens_no_2025_or_sample_reader(self) -> None:
        stage2_lock = {"promoted": False, "passed_groups": []}
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            (out_dir / "stage2_promotion_lock.json").write_text("{}\n", encoding="utf-8")
            forbidden = AssertionError("a 2025 or sample reader was called")
            with (
                mock.patch.object(runner, "_load_stage2_lock", return_value=(stage2_lock, {})),
                mock.patch.object(runner.raw_protocol, "_read_weather_features", side_effect=forbidden),
                mock.patch.object(runner.raw_protocol, "_read_full_labels", side_effect=forbidden),
                mock.patch.object(runner.pd, "read_csv", side_effect=forbidden),
            ):
                result = runner._final(
                    raw_dir=Path("unused"),
                    artifact_root=Path("unused"),
                    out_dir=out_dir,
                    raw_contract={},
                )
            self.assertFalse(result["2025_weather_read"])
            self.assertFalse(result["sample_submission_read"])
            self.assertFalse(result["submission_created"])


if __name__ == "__main__":
    unittest.main()

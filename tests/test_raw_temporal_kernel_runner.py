from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import pandas as pd

from scripts import run_raw_temporal_kernel as runner
from src.manifest import sha256_file
from src.metric import TARGET_COLS


class RawTemporalKernelRunnerTests(unittest.TestCase):
    def test_preregister_feasibility_dependency_and_parameters_are_exact(self) -> None:
        path = Path("configs/raw_temporal_kernel_preregister_v1.json")
        self.assertEqual(sha256_file(path), runner.PREREGISTER_SHA256)
        payload, raw_contract, dependency = runner._verify_preregister(path)
        self.assertEqual(payload["run_input"]["run_flatten_dimension"], 4800)
        self.assertEqual(payload["estimator"]["model"]["parameters"]["alpha"], 10.0)
        self.assertEqual(payload["estimator"]["model"]["parameters"]["gamma"], 0.03125)
        self.assertEqual(sha256_file(dependency), runner.RAW_DEPENDENCY_SHA256)
        self.assertIn("physical_stage1_inputs", raw_contract)

    def test_single_thread_environment_is_pinned(self) -> None:
        for name in runner.THREAD_ENV:
            self.assertEqual(os.environ[name], "1")

    def test_transitive_source_closure_contains_direct_and_helper_imports(self) -> None:
        path = Path("configs/raw_temporal_kernel_preregister_v1.json")
        _, _, dependency = runner._verify_preregister(path)
        sources = runner._source_paths(path, dependency)
        required = {
            "runner",
            "kernel_module",
            "raw_grid_module",
            "raw_grid_reader",
            "weather_protocol",
            "bounded_reader",
            "score_lock_helper",
            "run_sequence_guard",
            "temporal_run_key",
            "feature_dependency",
            "weather_quantile_dependency",
            "probabilistic_dependency",
            "metric",
            "manifest",
            "focused_model_test",
            "focused_runner_test",
        }
        self.assertTrue(required.issubset(sources))
        self.assertTrue(all(value.exists() for value in sources.values()))

    def test_registered_intervals_and_segments_are_exact(self) -> None:
        self.assertEqual(len(runner.YEAR_2022), 365 * 24)
        self.assertEqual(len(runner.YEAR_2023), 365 * 24)
        self.assertEqual(len(runner.G3_H1), 181 * 24)
        self.assertEqual(len(runner.G3_H2), 184 * 24)
        self.assertEqual(
            tuple(runner._stage1_segments(TARGET_COLS[0])),
            runner.STAGE1_REQUIRED[TARGET_COLS[0]],
        )
        self.assertEqual(
            tuple(runner._stage1_segments(TARGET_COLS[2])),
            runner.STAGE1_REQUIRED[TARGET_COLS[2]],
        )

    def test_aggregate_comparison_uses_official_macro_score(self) -> None:
        index = pd.date_range("2024-01-01 01:00:00", periods=24, freq="h", name=runner.TIME_COL)
        actual = pd.DataFrame(
            {TARGET_COLS[0]: 9000.0, TARGET_COLS[1]: 8500.0, TARGET_COLS[2]: 8000.0},
            index=index,
        )
        baseline = actual + 3000.0
        candidate = actual + 1000.0
        row = runner._aggregate_comparison(actual, baseline, candidate, {"full": index})["full"]
        self.assertEqual(
            row["delta"],
            float(row["candidate"]["total_score"])
            - float(row["baseline"]["total_score"]),
        )
        self.assertGreater(row["delta"], 0.0)

    def test_stage1_and_stage2_source_order_prescore_before_validation_labels(self) -> None:
        source = Path(runner.__file__).read_text(encoding="utf-8")
        stage1 = source[source.index("def _stage1(") : source.index("def _load_stage1_lock(")]
        self.assertLess(stage1.index("labels_g12, labels_g12_evidence"), stage1.index("g12_lock ="))
        self.assertLess(stage1.index("g12_lock ="), stage1.index("labels_g3, labels_g3_evidence"))
        self.assertLess(stage1.index("labels_g3, labels_g3_evidence"), stage1.index("g3_lock ="))
        self.assertLess(stage1.index("g3_lock ="), stage1.index("global_lock ="))
        self.assertLess(stage1.index("global_lock ="), stage1.index("score_labels, score_label_evidence"))
        self.assertIn("TARGET_COLS[:2]", stage1)
        self.assertIn("(TARGET_COLS[2],)", stage1)
        stage2 = source[source.index("def _stage2(") : source.index("def _load_stage2_lock(")]
        self.assertLess(stage2.index("prescore_lock ="), stage2.index("score_labels, score_label_evidence"))

    def test_stage2_identity_opens_no_2024_reader_and_has_complete_skip_schema(self) -> None:
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
            self.assertFalse(lock["executed"])
            self.assertEqual(lock["group_gate_passed"], {})
            self.assertFalse(lock["mixed_aggregate_gate_passed"])
            self.assertFalse(lock["2024_read"])
            self.assertFalse(lock["2025_read"])

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

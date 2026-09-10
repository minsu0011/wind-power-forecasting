from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from scripts import run_raw_attention_sficr_fixed_average as runner
from src.metric import TARGET_COLS


ROOT = Path(__file__).resolve().parents[1]


def _records(names: tuple[str, ...], delta: float, nmae: float, ficr: float) -> dict[str, object]:
    result = {}
    for name in names:
        baseline = {"score": 0.6, "one_minus_nmae": 0.85, "ficr": 0.35}
        candidate = {"score": 0.6 + delta, "one_minus_nmae": 0.85, "ficr": 0.35}
        if name == "full":
            candidate["one_minus_nmae"] += nmae
            candidate["ficr"] += ficr
        result[name] = {
            "baseline": baseline,
            "candidate": candidate,
            "delta": candidate["score"] - baseline["score"],
        }
    return result


class RawAttentionSFICRFixedAverageTests(unittest.TestCase):
    def test_01_preregister_hash_and_single_candidate(self) -> None:
        self.assertEqual(hashlib.sha256(runner.CONFIG_PATH.read_bytes()).hexdigest(), runner.CONFIG_SHA256)
        config, _ = runner.verify_config(runner.CONFIG_PATH)
        self.assertEqual(config["candidate"]["count"], 1)
        self.assertEqual(config["candidate"]["id"], runner.CANDIDATE_ID)
        self.assertTrue(config["risk_classification"]["selection_unsafe"])

    def test_02_fixed_formula_is_exact_and_ordered(self) -> None:
        index = pd.date_range("2023-01-01 01:00", periods=4, freq="h")
        base = pd.Series([1000.0, 3000.0, 6000.0, 9000.0], index=index, name=TARGET_COLS[0])
        raw = pd.Series([0.2, 0.3, 0.4, 0.5], index=index)
        smooth = pd.Series([0.25, 0.35, 0.45, 0.55], index=index)
        observed = runner.fixed_candidate(base, raw, smooth, capacity_kwh=21600.0)
        b = base.to_numpy(np.float64)
        expected = np.clip(
            b + 0.0125 * (raw.to_numpy() * 21600.0 - b)
            + 0.0125 * (smooth.to_numpy() * 21600.0 - b),
            0.0,
            1.02 * 21600.0,
        )
        np.testing.assert_array_equal(observed.to_numpy(), expected)

    def test_03_gate_passes_only_all_groups(self) -> None:
        comparisons = {
            group: _records(runner.REQUIRED[group], 0.001, 0.001, 0.001)
            for group in TARGET_COLS
        }
        result = runner.gate(comparisons)
        self.assertTrue(result["passed_all_groups"])
        self.assertEqual(result["registered_slice_count"], 17)
        self.assertEqual(result["decision"], runner.CANDIDATE_ID)

    def test_04_gate_rejects_one_bad_slice_without_group_rescue(self) -> None:
        comparisons = {
            group: _records(runner.REQUIRED[group], 0.001, 0.001, 0.001)
            for group in TARGET_COLS
        }
        bad = comparisons[TARGET_COLS[1]]["Q4"]
        bad["candidate"]["score"] = bad["baseline"]["score"] - 0.0001
        bad["delta"] = bad["candidate"]["score"] - bad["baseline"]["score"]
        result = runner.gate(comparisons)
        self.assertFalse(result["passed_all_groups"])
        self.assertEqual(result["decision"], "identity")

    def test_05_source_closure_is_target_free_and_pre2024(self) -> None:
        config, _ = runner.verify_config(runner.CONFIG_PATH)
        closure = runner.source_closure(config, runner.DEFAULT_RAW_DIR)
        self.assertEqual(closure["2024_or_2025_value_inputs"], [])
        self.assertEqual(closure["label_metadata_only"]["content_values_materialized"], 0)
        paths = set(closure["resolved_relative_paths"])
        self.assertIn("scripts/run_raw_attention_sficr_fixed_average.py", paths)

    def test_06_score_reader_is_fail_armed_by_candidate_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.json"
            with mock.patch.object(runner.raw_protocol, "_read_label_prefix") as reader:
                with self.assertRaisesRegex(RuntimeError, "lock is absent"):
                    runner._read_score_after_lock(Path(directory), {}, missing)
                reader.assert_not_called()

    def test_07_runner_orders_lock_before_label_and_metric(self) -> None:
        source = inspect.getsource(runner.run_stage1)
        lock_position = source.index('candidate_lock = out_dir / "candidate_lock_before_application_labels.json"')
        read_position = source.index("_read_score_after_lock")
        comparison_position = source.index("bayes._comparison")
        self.assertLess(lock_position, read_position)
        self.assertLess(read_position, comparison_position)

    def test_08_no_group_specific_candidate_or_alternative_weight(self) -> None:
        config, _ = runner.verify_config(runner.CONFIG_PATH)
        candidate = config["candidate"]
        self.assertEqual(candidate["groups"], list(TARGET_COLS))
        self.assertEqual(candidate["constituent_model_average_weights"], [0.5, 0.5])
        self.assertTrue(candidate["no_weight_grid_alternative_average_scale_group_selection_rescue_or_fallback"])


if __name__ == "__main__":
    unittest.main()

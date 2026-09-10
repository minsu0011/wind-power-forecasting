from __future__ import annotations

import inspect
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import run_covariate_shift_feature_pruning as runner
from src.covariate_shift_pruning import (
    QUANTILE_PROBABILITIES,
    blend_candidate_kwh,
    eligible_feature_names,
    select_stable_features,
    standardized_quantile_wasserstein,
)
from src.manifest import sha256_file


PROJECT_DIR = Path(__file__).resolve().parents[1]


class CovariateShiftFeaturePruningTests(unittest.TestCase):
    def test_preregister_hash_and_fixed_candidate_grid_are_exact(self) -> None:
        path = PROJECT_DIR / "configs/covariate_shift_feature_pruning_preregister_v1.json"
        self.assertEqual(sha256_file(path), runner.PREREGISTER_SHA256)
        payload = runner._verify_preregister(path)
        self.assertEqual(tuple(payload["feature_contract"]["fixed_keep_fractions_in_order"]), runner.KEEP_FRACTIONS)
        self.assertEqual(tuple(payload["model_family"]["objectives_in_fixed_order"]), runner.OBJECTIVES)
        self.assertEqual(tuple(payload["candidate_family"]["fixed_blend_weights_in_order"]), runner.BLEND_WEIGHTS)
        self.assertEqual(len(runner.MODEL_KEYS), 4)
        self.assertEqual(len(runner.CANDIDATE_KEYS), 8)

    def test_forensic_shift_table_has_exact_name_only_exclusion_count(self) -> None:
        # The failed-v1 forensic table is safe to inspect; this test deliberately
        # does not open the single-row-group weather cache before a v2 lock.
        path = PROJECT_DIR / (
            "artifacts/postgate/"
            "covariate_shift_feature_pruning_strict_v1_failed_physical_future_weather_access/"
            "shift/stage1__kpx_group_1.parquet"
        )
        names = tuple(pd.read_parquet(path, engine="pyarrow")["feature"].astype(str))
        self.assertEqual(len(names), 597)
        self.assertEqual(eligible_feature_names(names), names)
        self.assertFalse(any(name.startswith(("time__", "site__")) for name in names))

    def test_forbidden_feature_names_are_removed_without_value_inspection(self) -> None:
        columns = (
            "wind_speed",
            "time__hour_sin",
            "site__capacity_kwh",
            "actual_power",
            "scada__rpm",
            "model_prediction",
            "pred__old",
            "public_score",
            "forecast_id",
            "calendar_year",
            "temperature",
        )
        self.assertEqual(eligible_feature_names(columns), ("wind_speed", "temperature"))

    def test_standardized_wasserstein_matches_independent_quantile_formula(self) -> None:
        source_index = pd.date_range("2022-01-01", periods=8, freq="h")
        apply_index = pd.date_range("2023-01-01", periods=8, freq="h")
        source = pd.DataFrame({"x": np.arange(8.0), "constant": 4.0}, index=source_index)
        application = pd.DataFrame({"x": np.arange(8.0) + 2.0, "constant": 4.0}, index=apply_index)
        first = standardized_quantile_wasserstein(source, application)
        second = standardized_quantile_wasserstein(source, application)
        pd.testing.assert_frame_equal(first, second, check_exact=True)
        pooled = np.r_[source["x"].to_numpy(), application["x"].to_numpy()]
        numerator = np.trapezoid(
            np.abs(
                np.quantile(source["x"], QUANTILE_PROBABILITIES)
                - np.quantile(application["x"], QUANTILE_PROBABILITIES)
            ),
            QUANTILE_PROBABILITIES,
        )
        observed = first.set_index("feature")
        self.assertEqual(observed.loc["x", "shift_score"], numerator / np.std(pooled, ddof=0))
        self.assertFalse(bool(observed.loc["constant", "eligible_for_ranking"]))
        self.assertTrue(np.isnan(observed.loc["constant", "shift_score"]))

    def test_selector_uses_floor_count_shift_then_original_position_and_returns_original_order(self) -> None:
        table = pd.DataFrame(
            {
                "feature": ["a", "b", "c", "d", "constant"],
                "original_position": [0, 1, 2, 3, 4],
                "shift_score": [0.4, 0.1, 0.1, 0.2, np.nan],
                "eligible_for_ranking": [True, True, True, True, False],
            }
        )
        self.assertEqual(select_stable_features(table, 0.50), ("b", "c"))
        self.assertEqual(select_stable_features(table, 0.75), ("b", "c", "d"))

    def test_candidate_formula_keeps_weight_zero_bit_exact_and_changes_no_inputs(self) -> None:
        baseline = np.array([0.0, 100.0, 101.9], dtype=np.float64)
        raw_cf = np.array([-2.0, 0.75, 3.0], dtype=np.float64)
        baseline_before = baseline.copy()
        raw_before = raw_cf.copy()
        identity = blend_candidate_kwh(baseline, raw_cf, weight=0.0, capacity_kwh=100.0)
        np.testing.assert_array_equal(identity, baseline)
        candidate = blend_candidate_kwh(baseline, raw_cf, weight=0.10, capacity_kwh=100.0)
        np.testing.assert_allclose(candidate, [0.0, 97.5, 101.91])
        np.testing.assert_array_equal(baseline, baseline_before)
        np.testing.assert_array_equal(raw_cf, raw_before)

    @staticmethod
    def _comparisons(delta_by_key: dict[str, float]) -> dict[str, object]:
        return {
            key: {
                segment: {
                    "baseline": {"score": 0.5},
                    "candidate": {"score": 0.5 + delta_by_key[key]},
                    "delta": delta_by_key[key],
                }
                for segment in runner.STAGE1_REQUIRED["kpx_group_1"]
            }
            for key in runner.CANDIDATE_KEYS
        }

    def test_selection_requires_every_slice_and_is_json_order_invariant(self) -> None:
        deltas = {key: 0.001 for key in runner.CANDIDATE_KEYS}
        deltas[runner.CANDIDATE_KEYS[2]] = 0.002
        comparisons = self._comparisons(deltas)
        selected, _ = runner._select_group_candidate(
            comparisons, runner.STAGE1_REQUIRED["kpx_group_1"]
        )
        self.assertEqual(selected, runner.CANDIDATE_KEYS[2])
        roundtripped = json.loads(json.dumps(comparisons, sort_keys=True))
        self.assertEqual(
            runner._select_group_candidate(
                roundtripped, runner.STAGE1_REQUIRED["kpx_group_1"]
            )[0],
            selected,
        )
        roundtripped[selected]["Q4"]["delta"] = 0.0
        selected_after, audit = runner._select_group_candidate(
            roundtripped, runner.STAGE1_REQUIRED["kpx_group_1"]
        )
        self.assertNotEqual(selected_after, selected)
        self.assertFalse(audit["candidates"][selected]["all_required_slices_strictly_positive"])

    def test_runner_source_has_locked_stage2_mixed_gate_and_promoted_only_final(self) -> None:
        stage2 = inspect.getsource(runner._stage2)
        final = inspect.getsource(runner._final)
        self.assertIn("individually_passed", stage2)
        self.assertIn("aggregate_passed", stage2)
        self.assertIn("for group in promoted_groups", final)
        self.assertNotIn("public_scale", stage2.lower())
        self.assertNotIn("corrected_recent_v4", final.lower())


if __name__ == "__main__":
    unittest.main()

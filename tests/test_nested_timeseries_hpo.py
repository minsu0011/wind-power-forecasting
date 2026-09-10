from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np
import optuna
import pandas as pd

from scripts import run_nested_timeseries_hpo as runner
from src.manifest import sha256_file
from src.nested_timeseries_hpo import (
    InnerFold,
    N_TRIALS,
    SAMPLER_SEED,
    blend_with_baseline,
    fit_direct_cf,
    predict_direct_cf,
    resolved_estimator_parameters,
    suggest_parameters,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]


class NestedTimeseriesHPOTests(unittest.TestCase):
    def test_preregister_is_frozen_and_complete(self) -> None:
        path = PROJECT_DIR / "configs/nested_timeseries_hpo_preregister_v1.json"
        self.assertEqual(sha256_file(path), runner.PREREGISTER_SHA256)
        sidecar = path.with_suffix(".sha256").read_text(encoding="utf-8")
        self.assertEqual(
            sidecar,
            f"{runner.PREREGISTER_SHA256}  {path.name}\n",
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["inner_hpo"]["n_trials_per_group"], N_TRIALS)
        self.assertEqual(payload["inner_hpo"]["sampler"]["seed"], SAMPLER_SEED)
        self.assertEqual(
            tuple(payload["stage1_outer"]["candidate_weights_in_fixed_order"]),
            runner.WEIGHTS,
        )
        self.assertFalse(payload["pre_score_contract"]["public_or_scale_artifacts_may_be_read"])

    def test_all_inner_folds_are_strictly_causal_and_match_preregister(self) -> None:
        self.assertEqual(len(runner._inner_folds("kpx_group_1")), 3)
        self.assertEqual(len(runner._inner_folds("kpx_group_2")), 3)
        self.assertEqual(len(runner._inner_folds("kpx_group_3")), 4)
        for group in runner.TARGET_COLS:
            for fold in runner._inner_folds(group):
                self.assertIsInstance(fold, InnerFold)
                self.assertLess(fold.fit_end, fold.apply_start)

    def test_fixed_trial_resolves_exact_estimator_contract(self) -> None:
        values = {
            "objective": "quantile",
            "alpha": 0.65,
            "n_estimators": 500,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 40,
            "subsample": 0.85,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.01,
            "reg_lambda": 1.0,
            "min_split_gain": 0.01,
            "max_bin": 127,
        }
        sampled = suggest_parameters(optuna.trial.FixedTrial(values))
        resolved = resolved_estimator_parameters(sampled)
        self.assertEqual(resolved["objective"], "quantile")
        self.assertEqual(resolved["alpha"], 0.65)
        self.assertTrue(resolved["deterministic"])
        self.assertTrue(resolved["force_col_wise"])
        self.assertEqual(resolved["random_state"], SAMPLER_SEED)
        l1 = dict(sampled)
        l1["objective"] = "regression_l1"
        self.assertNotIn("alpha", resolved_estimator_parameters(l1))

    def test_blend_formula_and_registered_weight_guard(self) -> None:
        index = pd.date_range("2023-01-01", periods=3, freq="h")
        baseline = pd.Series([1000.0, 2000.0, 3000.0], index=index, name="g")
        direct = pd.Series([0.2, 0.3, 1.5], index=index)
        observed = blend_with_baseline(
            baseline, direct, weight=0.10, capacity_kwh=10_000.0
        )
        expected = np.clip(
            0.9 * baseline.to_numpy() + 0.1 * np.clip(direct.to_numpy(), 0.0, 1.02) * 10_000.0,
            0.0,
            10_200.0,
        )
        np.testing.assert_array_equal(observed.to_numpy(), expected)
        with self.assertRaises(ValueError):
            blend_with_baseline(baseline, direct, weight=0.15, capacity_kwh=10_000.0)

    def test_all_slice_selection_has_no_aggregate_rescue(self) -> None:
        required = ("full", "H1")
        comparisons = {}
        for weight in runner.WEIGHTS:
            key = runner.WEIGHT_KEYS[weight]
            comparisons[key] = {
                "full": {"delta": 0.1},
                "H1": {"delta": -0.0001},
            }
        selected, diagnostics = runner._select_weight(comparisons, required)
        self.assertEqual(selected, "identity")
        self.assertFalse(any(v["all_strictly_positive"] for v in diagnostics["statistics"].values()))

    def test_small_model_fit_predict_is_finite_and_clipped(self) -> None:
        rng = np.random.default_rng(8)
        index = pd.date_range("2022-01-01", periods=180, freq="h")
        features = pd.DataFrame(
            rng.normal(size=(180, 612)).astype(np.float32),
            index=index,
            columns=[f"f{i}" for i in range(612)],
        )
        target = pd.Series(np.linspace(0.11, 0.9, len(index)), index=index)
        sampled = {
            "objective": "regression_l1",
            "alpha": 0.55,
            "n_estimators": 500,
            "learning_rate": 0.015,
            "num_leaves": 15,
            "max_depth": 6,
            "min_child_samples": 120,
            "subsample": 0.7,
            "colsample_bytree": 0.55,
            "reg_alpha": 2.0,
            "reg_lambda": 20.0,
            "min_split_gain": 0.05,
            "max_bin": 127,
        }
        model, metadata = fit_direct_cf(features.iloc[:150], target.iloc[:150], sampled)
        prediction = predict_direct_cf(model, features.iloc[150:])
        self.assertEqual(metadata["rows_total"], 150)
        self.assertTrue(np.isfinite(prediction.to_numpy()).all())
        self.assertTrue(prediction.between(0.0, 1.02).all())

    def test_provenance_paths_exclude_forbidden_artifact_names(self) -> None:
        preregister = PROJECT_DIR / "configs/nested_timeseries_hpo_preregister_v1.json"
        paths = runner._provenance_paths(preregister)
        self.assertGreaterEqual(len(paths), 8)
        for path in paths.values():
            rendered = path.as_posix().lower()
            self.assertNotIn("public", rendered)
            self.assertNotIn("scale", rendered)


if __name__ == "__main__":
    unittest.main()

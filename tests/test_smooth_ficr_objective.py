from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from scripts import run_smooth_ficr_objective as runner
from src.manifest import sha256_file
from src.smooth_ficr_objective import (
    SmoothFICRLossConfig,
    SmoothFICRObjective,
    SmoothFICRRegressor,
    finite_difference_gradient_audit,
    smooth_ficr_loss_grad_hess,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]


class SmoothFICRObjectiveTests(unittest.TestCase):
    def test_preregister_hash_loss_and_candidate_family_are_exact(self) -> None:
        path = PROJECT_DIR / "configs" / "smooth_ficr_objective_preregister.json"
        self.assertEqual(sha256_file(path), runner.PREREGISTER_SHA256)
        payload, specs, config = runner._verify_preregister(path)
        self.assertEqual(tuple(specs), runner.BACKENDS)
        self.assertEqual(len(runner.RECIPE_KEYS), 6)
        self.assertEqual(config.thresholds_cf, (0.06, 0.08))
        self.assertEqual(config.settlement_weights, (1.0, 3.0))
        self.assertEqual(config.temperature_cf, 0.005)
        self.assertEqual(config.smooth_absolute_epsilon_cf, 0.0025)
        self.assertTrue(payload["interpretation"]["2024_historically_consumed"])
        self.assertTrue(payload["smooth_official_loss"]["no_loss_temperature_threshold_weight_or_hessian_search"])

    def test_analytic_gradient_matches_central_finite_difference(self) -> None:
        audit = finite_difference_gradient_audit(step=1e-6)
        self.assertEqual(audit["points"], 13)
        self.assertLessEqual(audit["maximum_absolute_error"], 2e-5)
        self.assertTrue(audit["gradient_finite"])
        self.assertTrue(audit["surrogate_hessian_finite"])
        self.assertTrue(audit["surrogate_hessian_strictly_positive"])

        rng = np.random.default_rng(42)
        actual = rng.uniform(0.10, 1.0, size=64)
        prediction = actual + rng.uniform(-0.12, 0.12, size=64)
        mean_actual = float(actual.mean())
        _, analytic, surrogate = smooth_ficr_loss_grad_hess(
            actual, prediction, mean_train_actual_cf=mean_actual
        )
        step = 1e-6
        plus = smooth_ficr_loss_grad_hess(
            actual, prediction + step, mean_train_actual_cf=mean_actual
        )[0]
        minus = smooth_ficr_loss_grad_hess(
            actual, prediction - step, mean_train_actual_cf=mean_actual
        )[0]
        numeric = (plus - minus) / (2 * step)
        self.assertLess(float(np.max(np.abs(analytic - numeric))), 2e-5)
        self.assertTrue(np.all(surrogate >= 0.001))
        self.assertTrue(np.all(surrogate <= 200.0))

    def test_objective_offset_matches_registered_prediction_scale(self) -> None:
        actual = np.array([0.2, 0.5, 0.9])
        raw = np.array([-0.05, 0.02, 0.09])
        offset = 0.45
        mean_actual = float(actual.mean())
        objective = SmoothFICRObjective(mean_actual, offset)
        gradient, hessian = objective(actual, raw)
        _, expected_gradient, expected_hessian = smooth_ficr_loss_grad_hess(
            actual,
            raw + offset,
            mean_train_actual_cf=mean_actual,
        )
        np.testing.assert_array_equal(gradient, expected_gradient)
        np.testing.assert_array_equal(hessian, expected_hessian)

    def test_loss_rejects_ineligible_target_and_changed_official_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "officially eligible"):
            smooth_ficr_loss_grad_hess(
                np.array([0.09]), np.array([0.10]), mean_train_actual_cf=0.5
            )
        config = SmoothFICRLossConfig(thresholds_cf=(0.05, 0.08))
        with self.assertRaisesRegex(ValueError, "official thresholds"):
            config.validate()

    def test_both_backends_fit_deterministically_and_reject_overlap(self) -> None:
        rng = np.random.default_rng(7)
        index = pd.date_range("2022-01-01", periods=320, freq="h")
        features = pd.DataFrame(
            rng.normal(size=(len(index), 5)), index=index, columns=list("abcde")
        )
        actual = pd.Series(6500.0 + 400.0 * features["a"], index=index)
        application = pd.DataFrame(
            rng.normal(size=(24, 5)),
            index=pd.date_range("2023-01-01", periods=24, freq="h"),
            columns=features.columns,
        )
        common = {
            "n_estimators": 3,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.75,
            "reg_alpha": 0.05,
            "reg_lambda": 2.0,
        }
        backend_parameters = {
            "lgb_smooth_ficr": {
                "num_leaves": 7,
                "min_child_samples": 10,
                "subsample_freq": 1,
                "verbosity": -1,
                "deterministic": True,
                "force_col_wise": True,
            },
            "xgb_smooth_ficr": {
                "max_depth": 3,
                "min_child_weight": 2.0,
                "tree_method": "hist",
                "max_bin": 31,
                "verbosity": 0,
                "base_score": 0.0,
            },
        }
        for backend in runner.BACKENDS:
            predictions = []
            for _ in range(2):
                model = SmoothFICRRegressor(
                    backend=backend,
                    common_parameters=common,
                    backend_parameters=backend_parameters[backend],
                    random_state=42,
                    n_jobs=1,
                ).fit(features, actual, capacity_kwh=21600.0)
                predictions.append(model.predict_cf(application).to_numpy())
                self.assertTrue(model.metadata()["positive_surrogate_hessian_used"])
                self.assertFalse(model.metadata()["true_hessian_used"])
                with self.assertRaisesRegex(ValueError, "overlap fit rows"):
                    model.predict_cf(features.iloc[:2])
            np.testing.assert_array_equal(predictions[0], predictions[1])
            self.assertTrue(np.isfinite(predictions[0]).all())
            self.assertTrue(np.all((predictions[0] >= 0.0) & (predictions[0] <= 1.02)))

    @staticmethod
    def _comparisons(deltas: dict[str, float]) -> dict[str, object]:
        result: dict[str, object] = {}
        for recipe in runner.RECIPE_KEYS:
            result[recipe] = {}
            for group in runner.TARGET_COLS:
                result[recipe][group] = {}
                for segment in runner.STAGE1_REQUIRED[group]:
                    baseline = 0.5
                    candidate = baseline + deltas[recipe]
                    result[recipe][group][segment] = {
                        "baseline": {"score": baseline},
                        "candidate": {"score": candidate},
                        "delta": candidate - baseline,
                    }
        return result

    def test_selection_is_global_all_segments_and_registered_tie_break(self) -> None:
        deltas = {recipe: 0.001 for recipe in runner.RECIPE_KEYS}
        deltas["lgb_smooth_ficr__w10"] = 0.002
        deltas["xgb_smooth_ficr__w10"] = 0.002
        selected, diagnostics = runner._select_recipe(self._comparisons(deltas))
        self.assertEqual(selected, "lgb_smooth_ficr__w10")
        self.assertEqual(diagnostics["selected"], selected)

        comparisons = self._comparisons({recipe: 0.001 for recipe in runner.RECIPE_KEYS})
        for position, recipe in enumerate(runner.RECIPE_KEYS):
            group = runner.TARGET_COLS[position % len(runner.TARGET_COLS)]
            segment = runner.STAGE1_REQUIRED[group][-1]
            record = comparisons[recipe][group][segment]
            record["candidate"]["score"] = record["baseline"]["score"]
            record["delta"] = 0.0
        selected, diagnostics = runner._select_recipe(comparisons)
        self.assertIsNone(selected)
        self.assertEqual(diagnostics["selected"], "identity")

    def test_zero_blend_is_value_bit_exact(self) -> None:
        index = pd.date_range("2023-01-01", periods=4, freq="h")
        baseline = pd.DataFrame(
            {group: [0.0, 123.125, 1000.5, 20000.0] for group in runner.TARGET_COLS},
            index=index,
        )
        raw = pd.DataFrame(
            {group: [500.0, 200.0, 900.0, 0.0] for group in runner.TARGET_COLS},
            index=index,
        )
        application = {group: index for group in runner.TARGET_COLS}
        observed = runner._blend_frame(baseline, raw, application, 0.0)
        self.assertEqual(
            np.ascontiguousarray(observed.to_numpy()).tobytes(),
            np.ascontiguousarray(baseline.to_numpy()).tobytes(),
        )

    def test_identity_postlock_and_stage2_skip_every_2024_reader(self) -> None:
        lock = {"locked_recipe": None}
        forbidden = AssertionError("2024 reader called")
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            (out_dir / "stage1_recipe_lock.json").write_text("{}", encoding="utf-8")
            with (
                mock.patch.object(runner, "_load_stage1_lock", return_value=(lock, {})),
                mock.patch.object(runner.bayes, "_read_bounded_labels", side_effect=forbidden) as bounded,
                mock.patch.object(runner.shared_strict, "_read_stage1_raw_features", side_effect=forbidden) as raw,
                mock.patch.object(pd, "read_parquet", side_effect=forbidden) as cache,
            ):
                audit = runner._postlock_cache_audit(
                    raw_dir=out_dir / "raw", cache_dir=out_dir / "cache", out_dir=out_dir
                )
            self.assertFalse(audit["performed"])
            self.assertFalse(audit["2024_cache_values_read"])
            bounded.assert_not_called()
            raw.assert_not_called()
            cache.assert_not_called()

            with (
                mock.patch.object(runner, "_load_stage1_lock", return_value=(lock, {})),
                mock.patch.object(runner.weather, "_stage2_input_snapshot", side_effect=forbidden) as snapshot,
                mock.patch.object(runner.bayes, "_read_full_labels", side_effect=forbidden) as labels,
                mock.patch.object(runner.shared_strict, "_read_features", side_effect=forbidden) as features,
                mock.patch.object(runner.bayes, "_read_prediction", side_effect=forbidden) as baseline,
            ):
                stage2 = runner._stage2(
                    raw_dir=out_dir / "raw",
                    artifact_root=out_dir / "artifacts",
                    cache_dir=out_dir / "cache",
                    out_dir=out_dir,
                    specs={},
                    loss_config=SmoothFICRLossConfig(),
                )
            self.assertFalse(stage2["candidate_promoted"])
            snapshot.assert_not_called()
            labels.assert_not_called()
            features.assert_not_called()
            baseline.assert_not_called()
            result = json.loads((out_dir / "stage2_results.json").read_text())
            self.assertFalse(result["2024_read"])

    def test_stage2_promotion_requires_every_group_and_quarter(self) -> None:
        comparisons = {}
        for group in runner.TARGET_COLS:
            comparisons[group] = {}
            for segment in runner.STAGE2_REQUIRED:
                baseline = 0.5
                candidate = 0.501
                comparisons[group][segment] = {
                    "baseline": {"score": baseline},
                    "candidate": {"score": candidate},
                    "delta": candidate - baseline,
                }
        promoted, _ = runner._stage2_promotion(comparisons)
        self.assertTrue(promoted)
        record = comparisons["kpx_group_3"]["Q4"]
        record["candidate"]["score"] = record["baseline"]["score"]
        record["delta"] = 0.0
        promoted, _ = runner._stage2_promotion(comparisons)
        self.assertFalse(promoted)


if __name__ == "__main__":
    unittest.main()

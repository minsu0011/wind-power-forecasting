from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from scripts import run_rolling_oof_stacking as runner
from src.manifest import sha256_file
from src.rolling_stacking import (
    apply_simplex_huber,
    fit_simplex_huber,
    passes_registered_gate,
    pseudo_huber_loss,
    residual_blend,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]


class RollingStackingTests(unittest.TestCase):
    def test_preregistrations_are_content_addressed(self) -> None:
        self.assertEqual(
            sha256_file(PROJECT_DIR / "configs/rolling_oof_stacking_preregister.json"),
            runner.V1_PREREGISTER_SHA256,
        )
        self.assertEqual(
            sha256_file(
                PROJECT_DIR / "configs/rolling_oof_stacking_preregister_v2.json"
            ),
            runner.V2_PREREGISTER_SHA256,
        )

    def test_v2_uses_only_corrected_capacity_factor_references(self) -> None:
        self.assertEqual(
            runner.STAGE2_BASELINE, "oof/gate2024_locked_v3_cf_fix.parquet"
        )
        self.assertEqual(
            runner.FINAL_BASELINE_PREDICTION,
            "final_cf_fix/predictions/corrected_v3_test.parquet",
        )
        payload = json.loads(
            (PROJECT_DIR / "configs/rolling_oof_stacking_preregister_v2.json").read_text(
                encoding="utf-8"
            )
        )
        override = payload["reference_baseline_override"]
        self.assertTrue(override["original_gate_v3_and_original_final_v3_forbidden"])
        self.assertIn("locked_v3_cf_fix", override["stage2"])

    def test_pseudo_huber_is_even_and_zero_at_origin(self) -> None:
        residual = np.array([-0.08, -0.02, 0.0, 0.02, 0.08])
        loss = pseudo_huber_loss(residual, delta_cf=0.04)
        self.assertEqual(loss[2], 0.0)
        np.testing.assert_allclose(loss, loss[::-1], rtol=0.0, atol=0.0)
        self.assertTrue(np.all(np.diff(loss[2:]) > 0))

    def test_simplex_fit_is_bounded_and_improves_synthetic_future(self) -> None:
        fit_index = pd.date_range("2023-01-01 01:00", periods=240, freq="h")
        actual = np.linspace(0.12, 0.88, len(fit_index))
        phase = np.linspace(0.0, 6.0 * np.pi, len(fit_index))
        matrix = np.column_stack(
            [
                actual + 0.002 * np.sin(phase),
                actual + 0.08,
                actual - 0.08,
                0.85 * actual,
                1.15 * actual,
                actual + 0.06 * np.cos(phase),
            ]
        )
        parameter = fit_simplex_huber(
            matrix,
            actual,
            fit_index,
            regularization=0.0001,
        )
        weights = np.asarray(parameter.weights)
        self.assertTrue(np.all(weights >= 0.0))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=9)
        self.assertGreaterEqual(parameter.intercept_cf, -0.025)
        self.assertLessEqual(parameter.intercept_cf, 0.025)
        valid_index = pd.date_range("2023-02-01 01:00", periods=240, freq="h")
        predicted = apply_simplex_huber(parameter, matrix, valid_index)
        uniform = np.mean(matrix, axis=1)
        self.assertLess(
            float(np.mean(np.abs(predicted - actual))),
            float(np.mean(np.abs(uniform - actual))),
        )

    def test_application_rejects_fit_or_past_timestamps(self) -> None:
        index = pd.date_range("2023-01-01 01:00", periods=48, freq="h")
        actual = np.linspace(0.2, 0.8, len(index))
        matrix = np.tile(actual[:, None], (1, len(runner.COMPONENTS)))
        parameter = fit_simplex_huber(
            matrix, actual, index, regularization=0.001
        )
        with self.assertRaisesRegex(ValueError, "not strictly after"):
            apply_simplex_huber(parameter, matrix, index)

    def test_zero_residual_blend_is_bit_exact_identity(self) -> None:
        baseline = np.array([0.0, 0.2, 0.9, 1.02], dtype="float64")
        direct = np.array([0.5, 0.4, 0.1, 0.2], dtype="float64")
        observed = residual_blend(baseline, direct, 0.0)
        self.assertTrue(np.array_equal(observed, baseline))

    def test_registered_gate_requires_all_segments_and_full_components(self) -> None:
        passing = {
            name: {
                "delta": {
                    "score": 0.001,
                    "one_minus_nmae": 0.0005,
                    "ficr": 0.0015,
                }
            }
            for name in ("H2", "Q3", "Q4")
        }
        self.assertTrue(passes_registered_gate(passing, full_segment_name="H2"))
        failing_segment = json.loads(json.dumps(passing))
        failing_segment["Q4"]["delta"]["score"] = 0.0
        self.assertFalse(
            passes_registered_gate(failing_segment, full_segment_name="H2")
        )
        failing_ficr = json.loads(json.dumps(passing))
        failing_ficr["H2"]["delta"]["ficr"] = -1e-12
        self.assertFalse(passes_registered_gate(failing_ficr, full_segment_name="H2"))

    def test_stage2_gate_uses_explicit_full_metrics_not_h2_metrics(self) -> None:
        comparisons = {
            name: {
                "delta": {
                    "score": 0.001,
                    "one_minus_nmae": 0.001,
                    "ficr": 0.001,
                }
            }
            for name in ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4")
        }
        comparisons["full"]["delta"]["ficr"] = -0.0001
        # A positive H2 must not conceal a failing full-period component metric.
        self.assertFalse(
            passes_registered_gate(comparisons, full_segment_name="full")
        )

    def test_gate_rejects_missing_explicit_full_segment(self) -> None:
        comparisons = {
            "H2": {
                "delta": {
                    "score": 0.001,
                    "one_minus_nmae": 0.001,
                    "ficr": 0.001,
                }
            }
        }
        with self.assertRaisesRegex(ValueError, "full segment 'full' is absent"):
            passes_registered_gate(comparisons, full_segment_name="full")

    @staticmethod
    def _write_unlocked_stage1(directory: Path) -> None:
        results = {
            "groups": {
                group: {
                    "locked": False,
                    "selected_blend_weight": None,
                    "outer_refit_parameter": {},
                }
                for group in runner.GROUPS
            }
        }
        results_path = directory / "stage1_results.json"
        results_path.write_text(json.dumps(results), encoding="utf-8")
        lock = {
            "v2_preregister_sha256": runner.V2_PREREGISTER_SHA256,
            "stage1_results_sha256": sha256_file(results_path),
            "locked_groups": [],
            "locked_parameters": {},
        }
        (directory / "stage1_lock.json").write_text(
            json.dumps(lock), encoding="utf-8"
        )

    def test_no_stage1_lock_skips_every_2024_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            self._write_unlocked_stage1(out_dir)
            forbidden = AssertionError("2024 read attempted")
            with (
                mock.patch.object(runner, "_read_labels", side_effect=forbidden) as labels,
                mock.patch.object(
                    runner, "_read_component_set", side_effect=forbidden
                ) as components,
                mock.patch.object(
                    runner, "_read_prediction", side_effect=forbidden
                ) as prediction,
            ):
                result = runner._stage2(
                    raw_dir=out_dir / "raw",
                    artifact_root=out_dir / "artifacts",
                    out_dir=out_dir,
                )
            self.assertFalse(result["results"]["2024_read"])
            self.assertEqual(result["lock"]["promoted_groups"], [])
            labels.assert_not_called()
            components.assert_not_called()
            prediction.assert_not_called()

    def test_no_promotion_skips_every_2025_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            self._write_unlocked_stage1(out_dir)
            stage2_results = {
                "promoted_groups": [],
                "candidate_promoted": False,
            }
            stage2_path = out_dir / "stage2_results.json"
            stage2_path.write_text(json.dumps(stage2_results), encoding="utf-8")
            promotion = {
                "stage1_lock_sha256": sha256_file(out_dir / "stage1_lock.json"),
                "stage2_results_sha256": sha256_file(stage2_path),
                "promoted_groups": [],
                "candidate_promoted": False,
            }
            (out_dir / "stage2_promotion_lock.json").write_text(
                json.dumps(promotion), encoding="utf-8"
            )
            forbidden = AssertionError("2025 read attempted")
            with (
                mock.patch.object(runner, "_read_labels", side_effect=forbidden) as labels,
                mock.patch.object(
                    runner, "_read_component_set", side_effect=forbidden
                ) as components,
                mock.patch.object(
                    runner, "_read_prediction", side_effect=forbidden
                ) as prediction,
                mock.patch.object(
                    runner.pd, "read_csv", side_effect=forbidden
                ) as csv_reader,
            ):
                result = runner._final(
                    raw_dir=out_dir / "raw",
                    artifact_root=out_dir / "artifacts",
                    out_dir=out_dir,
                )
            self.assertFalse(result["results"]["candidate_created"])
            self.assertFalse(result["results"]["2025_predictions_read"])
            labels.assert_not_called()
            components.assert_not_called()
            prediction.assert_not_called()
            csv_reader.assert_not_called()

    def test_tampered_stage1_result_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            self._write_unlocked_stage1(out_dir)
            (out_dir / "stage1_results.json").write_text(
                json.dumps({"tampered": True}), encoding="utf-8"
            )
            with self.assertRaisesRegex(AssertionError, "modified after locking"):
                runner._load_stage1_lock(out_dir)

    def test_period_weighting_equalizes_eligible_period_mass(self) -> None:
        actual = np.array([0.05, 0.2, 0.3, 0.4, 0.5, 0.6])
        periods = np.array([0, 0, 0, 1, 1, 1])
        weights = runner._equal_period_weights(actual, periods)
        eligible = actual >= 0.10
        first = weights[(periods == 0) & eligible].sum()
        second = weights[(periods == 1) & eligible].sum()
        self.assertAlmostEqual(first, second, places=15)


if __name__ == "__main__":
    unittest.main()

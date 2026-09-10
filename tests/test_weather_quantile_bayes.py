from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from scripts import run_weather_quantile_bayes as runner
from src.manifest import sha256_file
from src.weather_quantile import (
    BayesActionConfig,
    TrainOnlyMedianTransform,
    WeatherQuantileSurface,
    blend_with_baseline_kwh,
    conditional_bayes_action_cf,
    repair_quantile_crossing,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]


class WeatherQuantileBayesTests(unittest.TestCase):
    def test_preregister_hash_and_candidate_family_are_exact(self) -> None:
        path = PROJECT_DIR / "configs" / "weather_quantile_bayes_preregister.json"
        self.assertEqual(sha256_file(path), runner.PREREGISTER_SHA256)
        payload, model_spec, action_config = runner._verify_preregister(path)
        self.assertEqual(tuple(payload["quantile_surface"]["quantile_levels"]), runner.QUANTILE_LEVELS)
        self.assertEqual(tuple(payload["candidate_family"]["fixed_blend_weights"]), runner.BLEND_WEIGHTS)
        self.assertEqual(model_spec["random_state"], 42)
        self.assertEqual(action_config.interpolation_count, 33)
        self.assertTrue(payload["interpretation"]["2024_historically_consumed"])

    def test_year_segments_are_exact_partition(self) -> None:
        for year, expected_rows in ((2023, 8760), (2024, 8784)):
            segments = runner._year_segments(year)
            self.assertEqual(len(segments["full"]), expected_rows)
            self.assertTrue(
                segments["Q1"]
                .append([segments["Q2"], segments["Q3"], segments["Q4"]])
                .equals(segments["full"])
            )
            self.assertTrue(
                segments["H1"].append(segments["H2"]).equals(segments["full"])
            )
            for left, right in (("Q1", "Q2"), ("Q2", "Q3"), ("Q3", "Q4")):
                self.assertEqual(len(segments[left].intersection(segments[right])), 0)

    def test_transform_medians_are_fit_only_on_training_rows(self) -> None:
        train = pd.DataFrame(
            {"a": [1.0, np.nan, 5.0], "b": [2.0, 4.0, 6.0]},
            index=pd.date_range("2022-01-01", periods=3, freq="h"),
        )
        application = pd.DataFrame(
            {"a": [np.nan, 1000.0], "b": [np.nan, 1000.0]},
            index=pd.date_range("2023-01-01", periods=2, freq="h"),
        )
        transform = TrainOnlyMedianTransform().fit(train)
        observed = transform.transform(application)
        self.assertEqual(observed.iloc[0, 0], 3.0)
        self.assertEqual(observed.iloc[0, 1], 4.0)
        self.assertTrue(transform.fit_index_.equals(train.index))
        with self.assertRaisesRegex(ValueError, "columns/order"):
            transform.transform(application[["b", "a"]])

    def test_crossing_repair_is_stable_and_monotone(self) -> None:
        raw = np.array([[0.7, 0.2, 0.2, 0.8, 0.6], [0.1, 0.2, 0.3, 0.4, 0.5]])
        repaired, crossing_rows = repair_quantile_crossing(raw)
        self.assertEqual(crossing_rows, 1)
        self.assertTrue(np.all(np.diff(repaired, axis=1) >= 0.0))
        np.testing.assert_array_equal(repaired[0], [0.2, 0.2, 0.6, 0.7, 0.8])

    def test_bayes_action_is_deterministic_bounded_and_base_tie_aware(self) -> None:
        quantiles = np.full((3, 5), 0.50)
        baseline = np.array([0.50, 0.40, 0.60])
        first = conditional_bayes_action_cf(
            quantiles,
            baseline,
            mean_train_actual_cf=0.50,
            config=BayesActionConfig(),
        )
        second = conditional_bayes_action_cf(
            quantiles,
            baseline,
            mean_train_actual_cf=0.50,
            config=BayesActionConfig(),
        )
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all((first >= 0.0) & (first <= 1.02)))
        np.testing.assert_allclose(first, 0.50)
        with self.assertRaisesRegex(ValueError, "crossing-repaired"):
            conditional_bayes_action_cf(
                np.array([[0.2, 0.4, 0.3, 0.5, 0.6]]),
                np.array([0.4]),
                mean_train_actual_cf=0.5,
            )

    def test_zero_weight_baseline_is_value_bit_exact(self) -> None:
        index = pd.date_range("2023-01-01", periods=4, freq="h")
        baseline = pd.Series([0.0, 123.125, 1000.5, 22000.0], index=index)
        action = pd.Series([500.0, 200.0, 900.0, 0.0], index=index)
        observed = blend_with_baseline_kwh(
            baseline, action, weight=0.0, capacity_kwh=21600.0
        )
        self.assertEqual(
            np.ascontiguousarray(observed.to_numpy()).tobytes(),
            np.ascontiguousarray(baseline.to_numpy()).tobytes(),
        )
        blended = blend_with_baseline_kwh(
            baseline, action, weight=0.10, capacity_kwh=21600.0
        )
        np.testing.assert_allclose(
            blended.to_numpy(),
            np.clip(0.9 * baseline.to_numpy() + 0.1 * action.to_numpy(), 0, 22032),
        )

    @staticmethod
    def _comparisons(deltas: dict[str, float]) -> dict[str, object]:
        comparisons: dict[str, object] = {}
        for key in runner.WEIGHT_KEYS:
            comparisons[key] = {}
            for group in runner.TARGET_COLS:
                comparisons[key][group] = {}
                for segment in runner.STAGE1_REQUIRED[group]:
                    baseline = 0.50
                    candidate = baseline + deltas[key]
                    comparisons[key][group][segment] = {
                        "baseline": {"score": baseline},
                        "candidate": {"score": candidate},
                        "delta": candidate - baseline,
                    }
        return comparisons

    def test_global_selection_requires_every_group_segment_and_tie_breaks(self) -> None:
        comparisons = self._comparisons({"w05": 0.001, "w10": 0.002, "w20": 0.002})
        selected, diagnostics = runner._select_weight(comparisons)
        self.assertEqual(selected, 0.10)
        self.assertEqual(diagnostics["selected"], "w10")

        for key, group, segment in (
            ("w05", "kpx_group_1", "Q1"),
            ("w10", "kpx_group_2", "H2"),
            ("w20", "kpx_group_3", "Q4"),
        ):
            record = comparisons[key][group][segment]
            record["candidate"]["score"] = record["baseline"]["score"]
            record["delta"] = 0.0
        selected, diagnostics = runner._select_weight(comparisons)
        self.assertIsNone(selected)
        self.assertEqual(diagnostics["selected"], "identity")

    def test_surface_overlap_guard_and_training_only_metadata(self) -> None:
        rng = np.random.default_rng(42)
        index = pd.date_range("2022-01-01", periods=320, freq="h")
        features = pd.DataFrame(
            rng.normal(size=(len(index), 3)), index=index, columns=["a", "b", "c"]
        )
        actual = pd.Series(6000.0 + 500.0 * features["a"], index=index)
        surface = WeatherQuantileSurface(
            quantile_levels=runner.QUANTILE_LEVELS,
            model_parameters={
                "n_estimators": 3,
                "learning_rate": 0.1,
                "num_leaves": 7,
                "min_child_samples": 10,
                "verbosity": -1,
                "deterministic": True,
                "force_col_wise": True,
            },
            random_state=42,
            n_jobs=1,
        ).fit(features, actual, capacity_kwh=21600.0)
        metadata = surface.metadata()
        self.assertTrue(metadata["transform_fit_on_eligible_rows_only"])
        self.assertFalse(metadata["fit_score_calculated"])
        with self.assertRaisesRegex(ValueError, "overlap fit rows"):
            surface.predict_quantiles(features.iloc[-2:])

    def test_identity_stage2_skips_every_2024_reader(self) -> None:
        lock = {"locked_weight": None}
        forbidden = AssertionError("2024 reader called")
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            (out_dir / "stage1_candidate_lock.json").write_text(
                "{}", encoding="utf-8"
            )
            (out_dir / "postlock_cache_audit.json").write_text(
                "{}", encoding="utf-8"
            )
            with (
                mock.patch.object(runner, "_load_stage1_lock", return_value=(lock, {})),
                mock.patch.object(runner, "_stage2_input_snapshot", side_effect=forbidden) as snapshots,
                mock.patch.object(runner.bayes, "_read_full_labels", side_effect=forbidden) as labels,
                mock.patch.object(runner.shared_strict, "_read_features", side_effect=forbidden) as features,
                mock.patch.object(runner.bayes, "_read_prediction", side_effect=forbidden) as baseline,
            ):
                observed = runner._stage2(
                    raw_dir=out_dir / "raw",
                    artifact_root=out_dir / "artifacts",
                    cache_dir=out_dir / "cache",
                    out_dir=out_dir,
                    model_spec={},
                    action_config=BayesActionConfig(),
                )
            self.assertFalse(observed["candidate_promoted"])
            snapshots.assert_not_called()
            labels.assert_not_called()
            features.assert_not_called()
            baseline.assert_not_called()
            result = json.loads(
                (out_dir / "stage2_results.json").read_text(encoding="utf-8")
            )
            self.assertFalse(result["2024_read"])

    def test_identity_postlock_audit_does_not_open_cache_or_raw_values(self) -> None:
        lock = {"locked_weight": None}
        forbidden = AssertionError("post-lock value reader called")
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            (out_dir / "stage1_candidate_lock.json").write_text(
                "{}", encoding="utf-8"
            )
            with (
                mock.patch.object(runner, "_load_stage1_lock", return_value=(lock, {})),
                mock.patch.object(runner.bayes, "_read_bounded_labels", side_effect=forbidden) as labels,
                mock.patch.object(runner.shared_strict, "_read_stage1_raw_features", side_effect=forbidden) as raw,
                mock.patch.object(pd, "read_parquet", side_effect=forbidden) as cache,
            ):
                observed = runner._postlock_cache_audit(
                    raw_dir=out_dir / "raw",
                    cache_dir=out_dir / "cache",
                    out_dir=out_dir,
                )
            self.assertFalse(observed["performed"])
            self.assertFalse(observed["2024_cache_values_read"])
            labels.assert_not_called()
            raw.assert_not_called()
            cache.assert_not_called()


if __name__ == "__main__":
    unittest.main()

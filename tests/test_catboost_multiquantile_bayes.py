from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import joblib
import numpy as np
import pandas as pd

from scripts import run_catboost_multiquantile_bayes as runner
from src.catboost_multiquantile import (
    CatBoostBayesActionConfig,
    CatBoostMultiQuantileSurface,
    blend_with_baseline_kwh,
    exact_official_utility_action_cf,
    repair_quantile_crossing,
)
from src.manifest import sha256_file


PROJECT_DIR = Path(__file__).resolve().parents[1]


class CatBoostMultiQuantileBayesTests(unittest.TestCase):
    def test_preregister_hash_and_exact_contract(self) -> None:
        path = PROJECT_DIR / "configs/catboost_multiquantile_bayes_preregister_v1.json"
        self.assertEqual(sha256_file(path), runner.PREREGISTER_SHA256)
        payload, spec, action = runner._verify_preregister(path)
        self.assertEqual(spec["loss_function"], runner.LOSS_FUNCTION)
        self.assertEqual(tuple(spec["quantile_levels"]), runner.QUANTILE_LEVELS)
        self.assertEqual(spec["model_parameters"]["iterations"], 900)
        self.assertEqual(spec["model_parameters"]["random_seed"], 42)
        self.assertEqual(spec["model_parameters"]["thread_count"], 7)
        self.assertEqual(action.interpolation_count, 33)
        self.assertEqual(action.absolute_grid_lower_cf, 0.01)
        self.assertEqual(action.absolute_grid_upper_cf, 1.02)
        self.assertEqual(
            tuple(payload["candidate_family"]["fixed_global_blend_weights"]),
            runner.BLEND_WEIGHTS,
        )

    def test_action_matches_independent_registered_utility(self) -> None:
        repaired = np.array(
            [[0.18, 0.30, 0.44, 0.61, 0.83], [0.10, 0.21, 0.38, 0.55, 0.74]],
            dtype=float,
        )
        baseline = np.array([0.47, 0.35])
        mean_actual = 0.46
        config = CatBoostBayesActionConfig()
        observed = exact_official_utility_action_cf(
            repaired, baseline, mean_train_actual_cf=mean_actual, config=config
        )
        levels = np.array(config.quantile_levels)
        probs = np.linspace(0.10, 0.90, 33)
        grid = np.arange(0.01, 1.02 + 0.005, 0.01)
        expected: list[float] = []
        for row in range(len(repaired)):
            outcomes = np.clip(np.interp(probs, levels, repaired[row]), 0.10, 1.20)
            candidates = np.unique(
                np.clip(np.r_[grid, baseline[row], repaired[row]], 0.0, 1.02)
            )
            error = np.abs(candidates[:, None] - outcomes)
            unit = np.where(error <= 0.06, 4.0, np.where(error <= 0.08, 3.0, 0.0))
            utility = (-error + outcomes * unit / (4.0 * mean_actual)).mean(axis=1)
            tied = np.flatnonzero(utility >= utility.max() - 1e-12)
            order = np.lexsort((candidates[tied], np.abs(candidates[tied] - baseline[row])))
            expected.append(candidates[tied[order[0]]])
        np.testing.assert_array_equal(observed, expected)

    def test_joint_surface_alpha_order_crossing_and_joblib_roundtrip(self) -> None:
        rng = np.random.default_rng(42)
        train_index = pd.date_range("2022-01-01", periods=340, freq="h")
        apply_index = pd.date_range("2023-01-01", periods=32, freq="h")
        columns = [f"f{i}" for i in range(6)]
        train = pd.DataFrame(rng.normal(size=(340, 6)), index=train_index, columns=columns)
        apply = pd.DataFrame(rng.normal(size=(32, 6)), index=apply_index, columns=columns)
        actual_cf = np.clip(0.42 + 0.08 * train["f0"] + rng.normal(0, 0.04, 340), 0.02, 0.95)
        actual = actual_cf * 21600.0
        model = CatBoostMultiQuantileSurface(
            quantile_levels=runner.QUANTILE_LEVELS,
            loss_function=runner.LOSS_FUNCTION,
            model_parameters={
                "iterations": 8,
                "depth": 3,
                "learning_rate": 0.05,
                "l2_leaf_reg": 5.0,
                "random_strength": 0.25,
                "bootstrap_type": "Bayesian",
                "bagging_temperature": 0.5,
                "rsm": 0.8,
                "border_count": 32,
                "random_seed": 42,
                "thread_count": 1,
                "task_type": "CPU",
                "allow_writing_files": False,
                "verbose": False,
                "use_best_model": False,
            },
        ).fit(train, actual, capacity_kwh=21600.0)
        baseline = pd.Series(9000.0, index=apply_index, name="kpx_group_1")
        action, repaired = model.predict_action(apply, baseline)
        self.assertEqual(tuple(repaired.columns), runner.QUANTILE_COLUMNS)
        self.assertEqual(model.last_raw_quantiles_.shape, (32, 5))
        np.testing.assert_array_equal(
            repaired.to_numpy(),
            np.sort(model.last_raw_quantiles_, axis=1, kind="stable"),
        )
        self.assertEqual(model.eligible_fit_rows_, int((actual_cf >= 0.10).sum()))
        self.assertAlmostEqual(
            model.mean_train_actual_cf_, float(actual_cf[actual_cf >= 0.10].mean())
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "surface.joblib"
            joblib.dump(model, path)
            loaded = joblib.load(path)
            loaded_action, loaded_repaired = loaded.predict_action(apply, baseline)
        np.testing.assert_array_equal(loaded_repaired, repaired)
        np.testing.assert_array_equal(loaded_action, action)
        with self.assertRaisesRegex(ValueError, "overlap fit rows"):
            model.predict_quantiles(train.iloc[-2:])

    def test_crossing_repair_and_weight_zero_are_value_exact(self) -> None:
        raw = np.array([[0.6, 0.2, 0.4, 0.4, 0.8]])
        repaired, count = repair_quantile_crossing(raw)
        self.assertEqual(count, 1)
        np.testing.assert_array_equal(repaired, [[0.2, 0.4, 0.4, 0.6, 0.8]])
        index = pd.date_range("2023-01-01", periods=4, freq="h")
        baseline = pd.Series([0.0, 10.125, 5000.0, 22032.0], index=index)
        action = pd.Series([100.0, 20.0, 6000.0, 0.0], index=index)
        identity = blend_with_baseline_kwh(
            baseline, action, weight=0.0, capacity_kwh=21600.0
        )
        self.assertEqual(identity.to_numpy().tobytes(), baseline.to_numpy().tobytes())

    @staticmethod
    def _comparisons(deltas: dict[str, float]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key in runner.WEIGHT_KEYS:
            result[key] = {}
            for group in runner.TARGET_COLS:
                result[key][group] = {}
                for segment in runner.STAGE1_REQUIRED[group]:
                    candidate = 0.5 + deltas[key]
                    result[key][group][segment] = {
                        "baseline": {"score": 0.5},
                        "candidate": {"score": candidate},
                        "delta": candidate - 0.5,
                    }
        return result

    def test_global_selection_requires_all_17_slices(self) -> None:
        comparisons = self._comparisons({"w05": 0.001, "w10": 0.002, "w20": 0.002})
        selected, audit = runner._select_weight(comparisons)
        self.assertEqual(selected, 0.10)
        self.assertEqual(audit["selected"], "w10")
        comparisons["w10"]["kpx_group_3"]["Q4"]["candidate"]["score"] = 0.5
        comparisons["w10"]["kpx_group_3"]["Q4"]["delta"] = 0.0
        comparisons["w05"]["kpx_group_1"]["Q1"]["candidate"]["score"] = 0.5
        comparisons["w05"]["kpx_group_1"]["Q1"]["delta"] = 0.0
        comparisons["w20"]["kpx_group_2"]["H2"]["candidate"]["score"] = 0.5
        comparisons["w20"]["kpx_group_2"]["H2"]["delta"] = 0.0
        selected, audit = runner._select_weight(comparisons)
        self.assertIsNone(selected)
        self.assertEqual(audit["selected"], "identity")

    def test_bounded_label_reader_materializes_only_registered_usecols(self) -> None:
        text = "kst_dtm,kpx_group_1,kpx_group_2,kpx_group_3\n2022-01-01 01:00:00,1,2,3\n2022-01-01 02:00:00,4,5,6\n"
        payload = text.encode("utf-8")
        first_row_limit = len(text.splitlines(keepends=True)[0].encode()) + len(
            text.splitlines(keepends=True)[1].encode()
        )
        prefix = payload[:first_row_limit]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.csv"
            path.write_bytes(payload)
            spec = {
                "data_rows": 1,
                "bytes": first_row_limit,
                "sha256": hashlib.sha256(prefix).hexdigest(),
                "end": "2022-01-01T01:00:00",
                "usecols": ["kst_dtm", "kpx_group_3"],
                "next_row_first_field_only": "2022-01-01 02:00:00",
            }
            frame, evidence = runner._bounded_label_prefix(path, spec)
        self.assertEqual(tuple(frame.columns), ("kpx_group_3",))
        self.assertEqual(frame.iloc[0, 0], 3.0)
        self.assertEqual(evidence["omitted_target_value_cells_materialized"], 0)
        self.assertEqual(evidence["next_row_target_value_cells_materialized"], 0)

    def test_identity_stage2_reads_no_2024_input_and_binds_cache_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            cache_audit = out / "postlock_cache_audit.json"
            cache_audit.write_text(json.dumps({"performed": False}), encoding="utf-8")
            (out / "stage1_promotion_lock.json").write_text("{}", encoding="utf-8")
            forbidden = AssertionError("future reader called")
            with (
                mock.patch.object(runner, "_load_stage1_lock", return_value=({"locked_weight": None}, {})),
                mock.patch.object(runner, "_bounded_label_prefix", side_effect=forbidden) as labels,
                mock.patch.object(runner.shared, "_read_features", side_effect=forbidden) as weather,
                mock.patch.object(runner.bayes, "_read_prediction", side_effect=forbidden) as baseline,
            ):
                lock = runner._stage2(
                    raw_dir=out / "raw",
                    artifact_root=out / "artifacts",
                    cache_dir=out / "cache",
                    out_dir=out,
                    preregister={},
                    model_spec={},
                    action_config=CatBoostBayesActionConfig(),
                )
            self.assertFalse(lock["2024_read"])
            self.assertFalse(lock["csv_allowed"])
            self.assertEqual(
                lock["postlock_cache_audit"]["sha256"], sha256_file(cache_audit)
            )
            labels.assert_not_called()
            weather.assert_not_called()
            baseline.assert_not_called()


if __name__ == "__main__":
    unittest.main()

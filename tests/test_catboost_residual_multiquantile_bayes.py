from __future__ import annotations

import tempfile
from pathlib import Path
import inspect
import json
import unittest
from unittest import mock

import joblib
import numpy as np
import pandas as pd

from scripts import run_catboost_residual_multiquantile_bayes as runner
from src.catboost_multiquantile import (
    CatBoostBayesActionConfig,
    CatBoostResidualMultiQuantileSurface,
    exact_official_residual_utility_action_cf,
)
from src.manifest import sha256_file


PROJECT_DIR = Path(__file__).resolve().parents[1]


class ResidualMultiQuantileTests(unittest.TestCase):
    def test_preregister_and_feasibility_hashes_are_exact(self) -> None:
        prereg = PROJECT_DIR / "configs/catboost_residual_multiquantile_bayes_preregister_v3.json"
        feasibility = PROJECT_DIR / "artifacts/audits/catboost_residual_multiquantile_feasibility_v2.json"
        causal = PROJECT_DIR / "artifacts/audits/catboost_residual_causal_source_provenance_v1.json"
        self.assertEqual(sha256_file(prereg), runner.PREREGISTER_SHA256)
        self.assertEqual(sha256_file(feasibility), runner.FEASIBILITY_SHA256)
        self.assertEqual(sha256_file(causal), runner.CAUSAL_PROVENANCE_SHA256)
        payload, action = runner._verify_preregister(prereg, feasibility, causal)
        ledger = runner._verify_causal_source_provenance(causal, payload)
        self.assertEqual(payload["target_contract"]["minimum_eligible_fit_rows"], 800)
        self.assertEqual(tuple(payload["candidate_family"]["fixed_blend_weights"]), runner.BLEND_WEIGHTS)
        self.assertEqual(action.interpolation_count, 33)
        self.assertEqual(len(ledger["fitted_source_records"]), 19)
        self.assertEqual(
            payload["novelty_and_separation"]["output_directory"],
            "artifacts/postgate/catboost_residual_multiquantile_bayes_strict_v3",
        )

    def test_residual_action_matches_independent_formula(self) -> None:
        residual_q = np.array([[-0.20, -0.08, 0.01, 0.09, 0.23], [-0.10, -0.03, 0.02, 0.08, 0.15]])
        baseline = np.array([0.55, 0.32])
        mean_actual = 0.48
        config = CatBoostBayesActionConfig()
        observed = exact_official_residual_utility_action_cf(residual_q, baseline, mean_train_actual_cf=mean_actual, config=config)
        levels = np.array(config.quantile_levels); probs = np.linspace(0.10, 0.90, 33); grid = np.arange(0.01, 1.02 + 0.005, 0.01)
        expected = []
        for row in range(len(residual_q)):
            outcomes = np.clip(baseline[row] + np.interp(probs, levels, residual_q[row]), 0.10, 1.20)
            candidates = np.unique(np.clip(np.r_[grid, baseline[row], baseline[row] + residual_q[row]], 0.0, 1.02))
            error = np.abs(candidates[:, None] - outcomes)
            unit = np.where(error <= 0.06, 4.0, np.where(error <= 0.08, 3.0, 0.0))
            utility = (-error + outcomes * unit / (4.0 * mean_actual)).mean(axis=1)
            tied = np.flatnonzero(utility >= utility.max() - 1e-12)
            order = np.lexsort((candidates[tied], np.abs(candidates[tied] - baseline[row])))
            expected.append(candidates[tied[order[0]]])
        np.testing.assert_array_equal(observed, expected)

    def test_residual_surface_is_strict_and_joblib_exact(self) -> None:
        rng = np.random.default_rng(42)
        train_index = pd.date_range("2023-01-01", periods=820, freq="h")
        apply_index = pd.date_range("2023-05-01", periods=24, freq="h")
        columns = [f"f{i}" for i in range(8)]
        train = pd.DataFrame(rng.normal(size=(820, 8)), index=train_index, columns=columns)
        apply = pd.DataFrame(rng.normal(size=(24, 8)), index=apply_index, columns=columns)
        base_cf = np.clip(0.45 + 0.06 * train["f0"], 0.12, 0.90)
        actual_cf = np.clip(base_cf + 0.04 * train["f1"] + rng.normal(0, 0.03, 820), 0.10, 1.0)
        model = CatBoostResidualMultiQuantileSurface(
            quantile_levels=runner.QUANTILE_LEVELS,
            loss_function=runner.LOSS_FUNCTION,
            model_parameters={**runner.MODEL_PARAMETERS, "iterations": 6, "depth": 3, "thread_count": 1, "border_count": 32},
            minimum_actual_cf=0.10,
        )
        model.minimum_fit_rows = 800
        model.fit(train, actual_cf * 21600.0, base_cf * 21600.0, capacity_kwh=21600.0)
        apply_base = pd.Series(9000.0, index=apply_index, name="kpx_group_1")
        action, repaired = model.predict_action(apply, apply_base)
        np.testing.assert_array_equal(repaired.to_numpy(), np.sort(model.last_raw_quantiles_, axis=1, kind="stable"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "residual.joblib"; joblib.dump(model, path); loaded = joblib.load(path)
            action2, repaired2 = loaded.predict_action(apply, apply_base)
        np.testing.assert_array_equal(action, action2); np.testing.assert_array_equal(repaired, repaired2)
        with self.assertRaisesRegex(ValueError, "overlap fit rows"):
            model.predict_quantiles(train.iloc[-2:])

    def test_meta_feature_order_and_component_units(self) -> None:
        index = pd.date_range("2023-01-01 01:00", periods=24, freq="h", name="forecast_kst_dtm")
        baseline = pd.Series(np.linspace(4000, 12000, 24), index=index)
        components = {name: pd.Series(np.linspace(3000 + i*100, 13000 + i*100, 24), index=index) for i, name in enumerate(runner.COMPONENT_ORDER)}
        frame = runner._meta_features(index, "kpx_group_1", baseline, components)
        self.assertEqual(tuple(frame.columns), runner.META_COLUMNS)
        self.assertEqual(frame.shape, (24, 16))
        self.assertEqual(frame.iloc[0]["meta__lead_fraction"], 0.0)
        self.assertEqual(frame.iloc[-1]["meta__lead_fraction"], 1.0)
        self.assertAlmostEqual(frame.iloc[0]["meta__base_cf"], 4000 / 21600)

    @staticmethod
    def _comparisons(deltas: dict[str, float]) -> dict[str, object]:
        result: dict[str, object] = {key: {} for key in runner.WEIGHT_KEYS}
        for key in runner.WEIGHT_KEYS:
            for group in runner.TARGET_COLS:
                result[key][group] = {}
                for segment in runner.STAGE1_REQUIRED[group]:
                    result[key][group][segment] = {"baseline": {"score": 0.5}, "candidate": {"score": 0.5 + deltas[key]}, "delta": deltas[key]}
        return result

    def test_group_selection_requires_every_registered_slice(self) -> None:
        comparisons = self._comparisons({"w05": 0.001, "w10": 0.002, "w20": 0.0015})
        selected, _ = runner._select_groups(comparisons)
        self.assertEqual(set(selected.values()), {0.10})
        for key, group, segment in (("w05", "kpx_group_1", "Q2"), ("w10", "kpx_group_1", "Q4"), ("w20", "kpx_group_1", "H2")):
            comparisons[key][group][segment]["delta"] = 0.0
        selected, audit = runner._select_groups(comparisons)
        self.assertIsNone(selected["kpx_group_1"])
        self.assertEqual(audit["kpx_group_1"]["selected"], "identity")

    def test_group_selection_json_roundtrip_uses_exact_keys_and_frozen_order(self) -> None:
        comparisons = self._comparisons({"w05": 0.001, "w10": 0.002, "w20": 0.0015})
        expected = runner._select_groups(comparisons)
        roundtripped = json.loads(json.dumps(comparisons, sort_keys=True))
        self.assertNotEqual(
            tuple(roundtripped["w05"]["kpx_group_1"]),
            runner.STAGE1_REQUIRED["kpx_group_1"],
        )
        self.assertEqual(runner._select_groups(roundtripped), expected)

        missing_slice = json.loads(json.dumps(roundtripped))
        del missing_slice["w05"]["kpx_group_1"]["Q2"]
        with self.assertRaisesRegex(AssertionError, "segment keys changed"):
            runner._select_groups(missing_slice)

        extra_slice = json.loads(json.dumps(roundtripped))
        extra_slice["w05"]["kpx_group_1"]["unexpected"] = {"delta": 1.0}
        with self.assertRaisesRegex(AssertionError, "segment keys changed"):
            runner._select_groups(extra_slice)

        missing_weight = json.loads(json.dumps(roundtripped))
        del missing_weight["w20"]
        with self.assertRaisesRegex(AssertionError, "weight keys changed"):
            runner._select_groups(missing_weight)

        extra_group = json.loads(json.dumps(roundtripped))
        extra_group["w05"]["unexpected"] = {}
        with self.assertRaisesRegex(AssertionError, "group keys changed"):
            runner._select_groups(extra_group)

    def test_stage2_one_pass_one_fail_uses_partial_candidate(self) -> None:
        index = pd.date_range("2024-01-01 01:00", periods=8, freq="h", name="forecast_kst_dtm")
        baseline = pd.DataFrame(
            {
                "kpx_group_1": np.linspace(1000.0, 2000.0, len(index)),
                "kpx_group_2": np.linspace(2000.0, 3000.0, len(index)),
                "kpx_group_3": np.linspace(3000.0, 4000.0, len(index)),
            },
            index=index,
        )
        full = baseline.copy(deep=True)
        full["kpx_group_1"] += 125.0
        full["kpx_group_2"] -= 75.0
        comparisons: dict[str, object] = {}
        for group, deltas in (
            ("kpx_group_1", [0.01] * 7),
            ("kpx_group_2", [0.01, 0.01, 0.0, 0.01, 0.01, 0.01, 0.01]),
        ):
            comparisons[group] = {
                segment: {"delta": delta}
                for segment, delta in zip(runner.STAGE2_REQUIRED, deltas)
            }
        partial, passed_by_group, passed, identity = runner._stage2_partial_candidate(
            baseline,
            full,
            ["kpx_group_1", "kpx_group_2"],
            comparisons,
        )
        self.assertEqual(passed_by_group, {"kpx_group_1": True, "kpx_group_2": False})
        self.assertEqual(passed, ["kpx_group_1"])
        np.testing.assert_array_equal(partial["kpx_group_1"], full["kpx_group_1"])
        self.assertTrue(runner._series_value_bits_equal(partial["kpx_group_2"], baseline["kpx_group_2"]))
        self.assertTrue(runner._series_value_bits_equal(partial["kpx_group_3"], baseline["kpx_group_3"]))
        self.assertEqual(identity, {"kpx_group_2": True, "kpx_group_3": True})
        positive_aggregate = {
            segment: {"delta": 0.001} for segment in runner.STAGE2_REQUIRED
        }
        decision = runner._stage2_promotion_decision(passed, positive_aggregate)
        self.assertTrue(decision["promoted"])
        self.assertEqual(decision["promoted_groups"], ["kpx_group_1"])
        positive_aggregate["Q4"]["delta"] = 0.0
        rejected = runner._stage2_promotion_decision(passed, positive_aggregate)
        self.assertFalse(rejected["promoted"])
        self.assertEqual(rejected["promoted_groups"], [])

    def test_block_lock_binds_three_blends_and_causal_ledger(self) -> None:
        index = pd.date_range("2023-04-01 01:00", periods=4, freq="h", name="forecast_kst_dtm")
        groups = ("kpx_group_1", "kpx_group_2")
        baseline = pd.DataFrame(5000.0, index=index, columns=groups)
        actions = pd.DataFrame(0.30, index=index, columns=groups)
        causal = runner._causal_descriptor(
            {"causal_source_provenance": {"path": "artifacts/audits/catboost_residual_causal_source_provenance_v1.json"}}
        )
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            model_output = out_dir / "model.bin"
            model_output.write_bytes(b"fixed-model-output")
            lock_path, candidates = runner._write_block_prescore_lock(
                out_dir=out_dir,
                block_id="g12_Q1_to_Q2",
                groups=groups,
                fit_prefix_evidence={"physical_prefix_sha256": "f" * 64},
                application_index=index,
                baseline=baseline,
                actions=actions,
                model_output_paths=[model_output],
                feature_hashes={group: "a" * 64 for group in groups},
                causal_provenance_record=causal,
            )
            payload = runner._require_block_lock(lock_path, block_id="g12_Q1_to_Q2")
            self.assertEqual(len(payload["all_three_blend_outputs"]), 3)
            self.assertEqual(payload["causal_source_provenance"]["sha256"], runner.CAUSAL_PROVENANCE_SHA256)
            candidates[0].write_bytes(b"tampered")
            with self.assertRaisesRegex(AssertionError, "locked artifact changed"):
                runner._require_block_lock(lock_path, block_id="g12_Q1_to_Q2")

    def test_final_trains_only_stage2_promoted_subset_and_manifest_is_v3(self) -> None:
        source = inspect.getsource(runner._finalize)
        self.assertIn('for group in lock["promoted_groups"]', source)
        self.assertNotIn('for group in lock["locked_groups"]', source)
        self.assertIn(
            '"artifact_type": "catboost_residual_multiquantile_bayes_strict_forward_v3"',
            source,
        )

    def test_stage1_provenance_verifier_reads_only_safe_source_label_prefixes(self) -> None:
        prereg_path = PROJECT_DIR / "configs/catboost_residual_multiquantile_bayes_preregister_v3.json"
        causal_path = PROJECT_DIR / "artifacts/audits/catboost_residual_causal_source_provenance_v1.json"
        prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
        ledger = json.loads(causal_path.read_text(encoding="utf-8"))
        opened: list[Path] = []
        prefix_rows: list[int] = []

        def capture(path: Path, spec: object) -> None:
            del spec
            opened.append(Path(path))

        def safe_prefix(path: Path, *, data_rows: int) -> tuple[str, int]:
            self.assertEqual(path, Path(ledger["official_label_file"]["path"]))
            prefix_rows.append(data_rows)
            record = next(
                item
                for item in ledger["bounded_label_prefixes"].values()
                if int(item["data_rows"]) == data_rows
            )
            return record["physical_prefix_sha256"], int(record["physical_byte_limit"])

        def safe_next(
            path: Path, *, after_data_rows: int, prefix_bytes: int
        ) -> str:
            self.assertEqual(path, Path(ledger["official_label_file"]["path"]))
            record = next(
                item
                for item in ledger["bounded_label_prefixes"].values()
                if int(item["data_rows"]) == after_data_rows
            )
            self.assertEqual(prefix_bytes, int(record["physical_byte_limit"]))
            return str(record["next_row_first_field_only"])

        with mock.patch.object(runner, "_assert_file_spec", side_effect=capture), mock.patch.object(
            runner.shared,
            "_csv_prefix_identity",
            side_effect=safe_prefix,
        ), mock.patch.object(
            runner.shared,
            "_next_csv_first_field",
            side_effect=safe_next,
        ):
            runner._verify_causal_source_provenance(
                causal_path,
                prereg,
                stages=("stage1",),
                verify_label_prefix_bytes=True,
            )
        rendered = {path.as_posix() for path in opened}
        self.assertTrue(any("dev2023_lgb_top200_q07" in path for path in rendered))
        self.assertTrue(any("g3dev2023h2_candidates" in path for path in rendered))
        self.assertFalse(any("gate/v3" in path for path in rendered))
        self.assertFalse(any("final_v3" in path or "final_cf_fix" in path for path in rendered))
        self.assertFalse(any(path.endswith("train_labels.csv") for path in rendered))
        self.assertEqual(set(prefix_rows), {8760, 13104})
        self.assertEqual(len(prefix_rows), 2)


if __name__ == "__main__":
    unittest.main()

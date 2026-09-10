from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import optuna
import pandas as pd

from scripts import run_exact_component_maximin_hpo as runner
from src.exact_component_maximin import (
    COMPONENTS,
    N_TRIALS,
    assemble_candidate,
    is_identity_parameter,
    parameter_penalty,
    passes_block_gate,
    perturbed_weights,
    robust_objective,
    suggest_parameters,
    transfer_delta,
)
from src.manifest import sha256_file


PROJECT_DIR = Path(__file__).resolve().parents[1]
PREREGISTER_SHA256 = "dee8d03fb0d150e0732a8a92eb1e7e027c6f1fdf1ea16e4f534a0493762eb635"


def _recipe() -> dict:
    return {
        "ensemble": {
            "weights": {
                "kpx_group_1": dict(zip(COMPONENTS, (0.3, 0.1, 0.13, 0.3, 0.0, 0.17)))
            },
            "affine": {"kpx_group_1": {"scale": 1.15, "bias_kwh": -370.0}},
            "clip": {
                "kpx_group_1": {
                    "lower_capacity_fraction": 0.0,
                    "upper_capacity_fraction": 1.02,
                }
            },
            "power_bins": {"kpx_group_1": None},
        }
    }


class ExactComponentMaximinTests(unittest.TestCase):
    def test_preregister_hash_search_and_interaction_contract(self) -> None:
        path = PROJECT_DIR / "configs/exact_component_maximin_hpo_preregister_v1.json"
        self.assertEqual(sha256_file(path), PREREGISTER_SHA256)
        self.assertEqual(
            path.with_suffix(".sha256").read_text(encoding="utf-8"),
            f"{PREREGISTER_SHA256}  {path.name}\n",
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["census_and_distinctness"]["go_decision"], "GO_distinct_objective_and_parameterization")
        self.assertEqual(payload["optuna"]["n_trials_per_group"], N_TRIALS)
        self.assertIn("corrected_recent_v4", payload["stage2_fixed_2024"]["interaction_formula"])
        self.assertTrue(payload["locked_v3_assembly"]["no_power_bin_search"])

    def test_fixed_trial_search_space_is_exact(self) -> None:
        values = {
            "donor": "lgb_l1",
            "recipient_offset": 2,
            "mass_shift": 0.04,
            "scale_delta": 0.01,
            "bias_delta_cf": -0.005,
        }
        self.assertEqual(suggest_parameters(optuna.trial.FixedTrial(values), "kpx_group_1"), values)

    def test_mass_transfer_stays_on_simplex_and_moves_only_two_weights(self) -> None:
        locked = _recipe()["ensemble"]["weights"]["kpx_group_1"]
        parameter = {
            "donor": "lgb_l1",
            "recipient_offset": 2,
            "mass_shift": 0.04,
            "scale_delta": 0.0,
            "bias_delta_cf": 0.0,
        }
        observed = perturbed_weights(locked, parameter)
        self.assertAlmostEqual(sum(observed.values()), 1.0, places=15)
        self.assertAlmostEqual(observed["lgb_l1"], 0.26)
        self.assertAlmostEqual(observed["shared_l1"], 0.17)
        for component in set(COMPONENTS) - {"lgb_l1", "shared_l1"}:
            self.assertEqual(observed[component], locked[component])

    def test_identity_assembly_is_value_bit_exact(self) -> None:
        index = pd.date_range("2023-01-01", periods=4, freq="h")
        components = {
            name: pd.Series(np.arange(4, dtype=float) + position, index=index)
            for position, name in enumerate(COMPONENTS)
        }
        baseline = pd.Series([100.0, 200.0, 300.0, 400.0], index=index, name="kpx_group_1")
        parameter = {
            "donor": "lgb_l1",
            "recipient_offset": 1,
            "mass_shift": 0.0,
            "scale_delta": 0.0,
            "bias_delta_cf": 0.0,
        }
        self.assertTrue(is_identity_parameter(parameter))
        output, metadata = assemble_candidate(
            components,
            group="kpx_group_1",
            recipe=_recipe(),
            parameter=parameter,
            identity_baseline_kwh=baseline,
        )
        np.testing.assert_array_equal(output.to_numpy(), baseline.to_numpy())
        self.assertTrue(metadata["identity_short_circuit"])

    def test_robust_objective_and_gate_require_every_block(self) -> None:
        comparisons = {
            "full": {"delta": {"score": 0.002, "one_minus_nmae": 0.001, "ficr": 0.003}},
            "late": {"delta": {"score": 0.001, "one_minus_nmae": 0.0, "ficr": 0.002}},
        }
        identity = {
            "donor": "lgb_l1",
            "recipient_offset": 1,
            "mass_shift": 0.0,
            "scale_delta": 0.0,
            "bias_delta_cf": 0.0,
        }
        self.assertAlmostEqual(robust_objective(comparisons, identity), 0.001375)
        self.assertTrue(passes_block_gate(comparisons, full_block="full"))
        comparisons["late"]["delta"]["score"] = 0.0
        self.assertFalse(passes_block_gate(comparisons, full_block="full"))

    def test_penalty_is_normalized_and_transfer_is_clipped(self) -> None:
        parameter = {
            "donor": "lgb_l1",
            "recipient_offset": 1,
            "mass_shift": 0.08,
            "scale_delta": -0.015,
            "bias_delta_cf": 0.0075,
        }
        self.assertAlmostEqual(parameter_penalty(parameter), 0.0003)
        index = pd.date_range("2024-01-01", periods=2, freq="h")
        recent = pd.Series([100.0, 10_000.0], index=index)
        v3 = pd.Series([200.0, 9_000.0], index=index)
        candidate = pd.Series([0.0, 20_000.0], index=index)
        output = transfer_delta(recent, v3, candidate, capacity_kwh=10_000.0)
        np.testing.assert_array_equal(output.to_numpy(), np.array([0.0, 10_200.0]))

    def test_static_source_closure_is_exact_and_forbidden_paths_are_absent(self) -> None:
        closure = tuple(
            path.relative_to(PROJECT_DIR).as_posix() for path in runner._source_closure()
        )
        self.assertEqual(closure, runner.EXPECTED_SOURCE_CLOSURE)
        paths = runner._provenance_paths(
            PROJECT_DIR / "configs/exact_component_maximin_hpo_preregister_v1.json"
        )
        self.assertGreaterEqual(len(paths), len(closure) + 4)
        for path in paths.values():
            self.assertNotIn("public", path.as_posix().lower())
            self.assertNotIn("scale", path.as_posix().lower())

    def test_no_stage1_pass_skips_every_conditional_2024_reader(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = {
                "group_results": {
                    group: {"outer_passed": False} for group in runner.TARGET_COLS
                },
                "passed_groups": [],
            }
            runner._write_json(root / "stage1_results.json", result)
            lock = {
                "preregister_sha256": runner.PREREGISTER_SHA256,
                "stage1_results": runner.describe_file(root / "stage1_results.json"),
                "passed_groups": [],
                "locked_parameters": {},
            }
            runner._write_json(root / "stage1_promotion_lock.json", lock)
            observed = runner._stage2(
                raw_dir=root / "missing_raw",
                artifact_root=root / "missing_artifacts",
                out_dir=root,
                preregister={},
            )
            self.assertEqual(observed["promoted_groups"], [])
            stage2 = json.loads((root / "stage2_results.json").read_text(encoding="utf-8"))
            self.assertFalse(stage2["2024_component_read"])
            self.assertFalse(stage2["2024_label_read"])
            self.assertFalse(stage2["recent_v4_read"])

    def test_runner_source_orders_outer_label_read_after_prescore_lock(self) -> None:
        source = (PROJECT_DIR / "scripts/run_exact_component_maximin_hpo.py").read_text(
            encoding="utf-8"
        )
        stage1 = source[source.index("def _stage1(") : source.index("def _load_stage1_lock")]
        lock_position = stage1.index('prescore_path = out_dir / "stage1_prescore_lock.json"')
        outer_label_position = stage1.index('prefixes["stage1_outer_after_lock"]')
        self.assertLess(lock_position, outer_label_position)
        stage2 = source[source.index("def _stage2(") : source.index("def _load_stage2_lock")]
        prediction_lock = stage2.index('prescore_path = out_dir / "stage2_prescore_lock.json"')
        label_read = stage2.index("labels, label_identity = _read_full_labels(raw_dir)")
        self.assertLess(prediction_lock, label_read)


if __name__ == "__main__":
    unittest.main()

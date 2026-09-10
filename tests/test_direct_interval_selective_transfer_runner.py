from __future__ import annotations

import inspect
from pathlib import Path
import unittest

from scripts import run_direct_interval_selective_transfer as runner


class DirectIntervalSelectiveTransferRunnerTests(unittest.TestCase):
    def test_all_three_frozen_hashes_and_candidate_specs(self) -> None:
        args = runner.parse_args(["--stage", "prescore"])
        configs = runner._verify_configs(args)
        self.assertEqual(configs["v2"]["experiment_id"], "direct_interval_selective_scale_transfer_v2")
        self.assertEqual(configs["v3"]["experiment_id"], "direct_interval_public_adaptive_transfer_v3")
        self.assertEqual(
            configs["addendum"]["addendum_id"],
            "direct_interval_v2_v3_dual_baseline_interaction_gate_20260808",
        )
        self.assertEqual(runner.VARIANT_SPECS["v2"]["kpx_group_2"], (0.98, 0.01))
        self.assertEqual(runner.VARIANT_SPECS["v3"]["kpx_group_2"], (0.95, 0.0025))

    def test_execution_is_split_at_joint_prescore_lock(self) -> None:
        prescore = inspect.getsource(runner.run_prescore)
        score = inspect.getsource(runner.run_score_final)
        self.assertNotIn("_load_full_labels_after_locks", prescore)
        self.assertIn("_load_full_labels_after_locks", score)
        loader = inspect.getsource(runner._load_full_labels_after_locks)
        self.assertLess(loader.index("joint_before_2024_label_lock.json"), loader.index("sha256_file(path)"))
        self.assertLess(loader.index("sha256_file(path)"), loader.index("_read_full_labels(path)"))

    def test_same_surface_is_used_for_both_baselines_and_variants(self) -> None:
        source = inspect.getsource(runner._prepare_stage2)
        self.assertEqual(source.count("DirectIntervalProbabilityModel().fit("), 1)
        self.assertIn("shared_arrays[group][\"utility\"]", source)
        self.assertIn('for baseline_name, baseline in baselines.items()', source)
        self.assertIn("_copy_exclusive(v2_model, v3_model)", source)
        self.assertIn("_copy_exclusive(v2_surface, v3_surface)", source)

    def test_final_requires_dual_baseline_promotion(self) -> None:
        score = inspect.getsource(runner._score_variant_stage2)
        self.assertIn("both_pass = both_pass and promoted", score)
        self.assertIn('baseline_results["primary_v3"]["passed"]', score)
        self.assertIn('baseline_results["interaction_v4"]["passed"]', score)
        final = inspect.getsource(runner.run_score_final)
        self.assertIn('if result["dual_baseline_promoted"]', final)
        self.assertIn("_finalize_passing", final)

    def test_final_csv_names_are_distinct_and_g3_identity_is_checked(self) -> None:
        source = inspect.getsource(runner._finalize_passing)
        self.assertIn("direct_interval_selective_scale_v4_2025.csv", source)
        self.assertIn("direct_interval_public_adaptive_v4_2025.csv", source)
        self.assertIn('candidate["kpx_group_3"]', source)
        self.assertIn("final G3 identity differs", source)


if __name__ == "__main__":
    unittest.main()

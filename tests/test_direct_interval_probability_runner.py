from __future__ import annotations

import inspect
import json
from pathlib import Path
import unittest

from scripts import run_direct_interval_probability as runner
from src.direct_interval_probability import BLEND_WEIGHTS, CONTEXT_COLUMNS


PROJECT_DIR = Path(__file__).resolve().parents[1]


class DirectIntervalProbabilityRunnerTests(unittest.TestCase):
    def test_preregister_verifier_binds_all_candidate_axes(self) -> None:
        path = PROJECT_DIR / "configs/direct_interval_probability_preregister_v1.json"
        payload = runner._verify_preregister(path)
        self.assertEqual(payload["stage1"]["registered_slice_count"], 17)
        self.assertEqual(
            tuple(payload["feature_contract"]["context_columns_in_exact_order"]),
            CONTEXT_COLUMNS,
        )
        self.assertEqual(
            tuple(payload["candidate_family"]["fixed_global_blend_weights"]),
            tuple(BLEND_WEIGHTS.values()),
        )
        self.assertFalse(payload["public_contract"]["public_metric_triplets_used"])
        self.assertFalse(payload["public_contract"]["public_scale_artifacts_used"])

    def test_static_import_closure_contains_runner_model_metric_and_physical_helper(self) -> None:
        closure = runner._static_repo_import_closure((Path(runner.__file__),))
        relative = {path.relative_to(PROJECT_DIR).as_posix() for path in closure}
        for required in (
            "scripts/run_direct_interval_probability.py",
            "scripts/run_catboost_multiquantile_bayes.py",
            "scripts/run_shared_q07_multiseed.py",
            "src/direct_interval_probability.py",
            "src/features.py",
            "src/metric.py",
            "src/manifest.py",
        ):
            self.assertIn(required, relative)

    def test_application_score_label_call_is_after_candidate_hash_lock(self) -> None:
        source = inspect.getsource(runner.run_stage1)
        lock_position = source.index('global_lock_path = out_dir / "stage1_global_prescore_lock.json"')
        lock_write_position = source.index("strict._write_json(\n        global_lock_path", lock_position)
        score_position = source.index(
            'score_labels, score_label_evidence = bounded._bounded_label_prefix',
            lock_write_position,
        )
        self.assertLess(lock_write_position, score_position)
        self.assertIn('specs["labels_stage1_score_prefix_after_prescore_lock"]', source[score_position:])

    def test_runner_has_no_stage2_final_or_csv_execution_mode(self) -> None:
        args = runner.parse_args([])
        self.assertEqual(args.stage, "stage1")
        self.assertEqual(
            args.out_dir.as_posix(),
            "artifacts/postgate/direct_interval_probability_strict_v1",
        )
        source = inspect.getsource(runner.run_stage1)
        self.assertNotIn("read_csv(args.raw_dir / \"sample_submission.csv\")", source)
        self.assertNotIn("_atomic_csv(", source)
        self.assertIn('"year_2025_value_bytes_read": 0', source)
        self.assertIn('"public_or_scale_artifact_bytes_read": 0', source)

    def test_census_paths_exist_and_exclude_public_probe_implementations(self) -> None:
        payload = json.loads(
            (PROJECT_DIR / "configs/direct_interval_probability_preregister_v1.json").read_text(
                encoding="utf-8"
            )
        )
        paths = [
            path
            for record in payload["census_and_novelty_contract"]["censused_implementations"]
            for path in record["paths"]
        ]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertTrue(all((PROJECT_DIR / path).is_file() for path in paths))
        self.assertFalse(any("public_scale" in path for path in paths))


if __name__ == "__main__":
    unittest.main()

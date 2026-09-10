from __future__ import annotations

from pathlib import Path
import unittest

from scripts import run_raw_spatiotemporal_attention as runner
from src.manifest import sha256_file
from src.metric import TARGET_COLS


class RawSpatiotemporalAttentionRunnerTests(unittest.TestCase):
    def test_preregister_feasibility_dependency_and_training_are_frozen(self) -> None:
        path = Path("configs/raw_spatiotemporal_attention_preregister_v1.json")
        self.assertEqual(sha256_file(path), runner.PREREGISTER_SHA256)
        payload, raw_contract, dependency = runner._verify_preregister(path)
        self.assertEqual(payload["architecture"]["id"], runner.MODEL_ID)
        self.assertEqual(payload["training"]["epochs"], runner.EPOCHS)
        self.assertEqual(tuple(payload["candidate_family"]["candidate_order"]), runner.CANDIDATE_KEYS)
        self.assertEqual(sha256_file(dependency), runner.RAW_DEPENDENCY_SHA256)
        self.assertIn("physical_stage1_inputs", raw_contract)

    def test_source_closure_includes_new_and_inherited_protocol(self) -> None:
        path = Path("configs/raw_spatiotemporal_attention_preregister_v1.json")
        _, _, dependency = runner._verify_preregister(path)
        sources = runner._source_paths(path, dependency)
        required = {
            "runner",
            "attention_module",
            "focused_test",
            "focused_runner_test",
            "preregister",
            "feasibility",
            "kernel_dependency__raw_grid_reader",
            "kernel_dependency__bounded_reader",
            "kernel_dependency__score_lock_helper",
            "kernel_dependency__metric",
        }
        self.assertTrue(required.issubset(sources))
        self.assertTrue(all(value.exists() for value in sources.values()))

    def test_causal_stage1_intervals_and_slices_are_exact(self) -> None:
        self.assertEqual(len(runner.YEAR_2022), 365 * 24)
        self.assertEqual(len(runner.YEAR_2023), 365 * 24)
        self.assertEqual(len(runner.G3_H1), 181 * 24)
        self.assertEqual(len(runner.G3_H2), 184 * 24)
        self.assertLess(runner.YEAR_2022.max(), runner.YEAR_2023.min())
        self.assertLess(runner.G3_H1.max(), runner.G3_H2.min())
        self.assertEqual(tuple(runner._segments(TARGET_COLS[0])), runner.STAGE1_REQUIRED[TARGET_COLS[0]])
        self.assertEqual(tuple(runner._segments(TARGET_COLS[2])), runner.STAGE1_REQUIRED[TARGET_COLS[2]])

    def test_source_orders_both_prescore_locks_before_validation_labels(self) -> None:
        source = Path(runner.__file__).read_text(encoding="utf-8")
        stage1 = source[source.index("def _stage1(") : source.index("def _manifest(")]
        self.assertLess(stage1.index("labels_g12, labels_g12_evidence"), stage1.index("lock_g12 ="))
        self.assertLess(stage1.index("lock_g12 ="), stage1.index("labels_g3, labels_g3_evidence"))
        self.assertLess(stage1.index("labels_g3, labels_g3_evidence"), stage1.index("lock_g3 ="))
        self.assertLess(stage1.index("lock_g3 ="), stage1.index("global_lock ="))
        self.assertLess(stage1.index("global_lock ="), stage1.index("score_labels, score_label_evidence"))
        self.assertIn("TARGET_COLS[:2]", stage1)
        self.assertIn("TARGET_COLS\n", stage1)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

from scripts import run_covariate_shift_feature_pruning_v2 as v2
from scripts import run_covariate_shift_feature_pruning_v3 as runner
from src.manifest import sha256_file


PROJECT_DIR = Path(__file__).resolve().parents[1]


class CovariateShiftFeaturePruningV3Tests(unittest.TestCase):
    def test_v3_preregister_hash_and_no_retune_supersession_are_exact(self) -> None:
        path = PROJECT_DIR / "configs/covariate_shift_feature_pruning_preregister_v3.json"
        self.assertEqual(sha256_file(path), runner.PREREGISTER_SHA256)
        payload, execution, _ = runner._verify_preregister(path)
        self.assertTrue(payload["supersession"]["v2_is_ineligible_for_selection"])
        self.assertTrue(payload["supersession"]["no_retune_reselection_or_recipe_change"])
        self.assertEqual(payload["fixed_candidate_contract"], execution["fixed_candidate_contract"])

    def test_ast_source_closure_is_exact_and_contains_all_previously_missing_files(self) -> None:
        preregister = json.loads(
            (PROJECT_DIR / "configs/covariate_shift_feature_pruning_preregister_v3.json").read_text(encoding="utf-8")
        )
        discovered = runner.discover_local_source_closure(
            PROJECT_DIR / "scripts/run_covariate_shift_feature_pruning_v3.py"
        )
        names = runner._relative_paths(discovered)
        contract = preregister["source_closure_contract"]
        self.assertEqual(names, contract["ordered_paths"])
        self.assertEqual(len(names), contract["expected_count"])
        self.assertTrue(set(contract["independent_audit_missing_eight_all_included"]).issubset(names))
        self.assertIn("src/__init__.py", names)

    def test_prescore_source_lock_binds_every_source_and_config(self) -> None:
        preregister_path = PROJECT_DIR / "configs/covariate_shift_feature_pruning_preregister_v3.json"
        preregister = json.loads(preregister_path.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "fresh"
            lock_path = runner._write_source_closure_lock(out, preregister, preregister_path)
            locked = runner._require_source_closure_unchanged(lock_path, preregister, preregister_path)
        self.assertEqual(locked["source_count"], 16)
        self.assertEqual(set(locked["source_files"]), set(preregister["source_closure_contract"]["ordered_paths"]))
        self.assertEqual(len(locked["configuration_files"]), 6)
        for record in [*locked["source_files"].values(), *locked["configuration_files"].values()]:
            self.assertEqual(len(record["sha256"]), 64)
            self.assertGreater(record["size_bytes"], 0)

    def test_source_lock_tamper_is_rejected(self) -> None:
        preregister_path = PROJECT_DIR / "configs/covariate_shift_feature_pruning_preregister_v3.json"
        preregister = json.loads(preregister_path.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "fresh"
            lock_path = runner._write_source_closure_lock(out, preregister, preregister_path)
            locked = json.loads(lock_path.read_text(encoding="utf-8"))
            first = locked["ordered_paths"][0]
            locked["source_files"][first]["sha256"] = "0" * 64
            lock_path.write_text(json.dumps(locked), encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "source closure/configuration changed"):
                runner._require_source_closure_unchanged(lock_path, preregister, preregister_path)

    def test_v2_execution_hook_binds_source_lock_into_prescore_and_identity(self) -> None:
        source = inspect.getsource(v2._run_stage1)
        self.assertIn("source_closure_lock_path", source)
        self.assertIn('"source_closure_prescore_lock": source_closure_snapshot', source)
        self.assertIn("source_closure_locked_before_feature_build_and_fit", source)
        self.assertIn("reproduction_record_key", source)
        check_position = source.index("source_closure_snapshot = describe_file")
        build_position = source.index("_build_stage1_features")
        self.assertLess(check_position, build_position)

    def test_reproduction_reference_refuses_to_open_before_v3_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing-v3-results.json"
            with self.assertRaisesRegex(AssertionError, "v3 metrics must exist"):
                runner._v2_v3_reproduction_audit(Path(directory), missing)

    def test_physical_suffix_and_no_read_contract_is_unchanged(self) -> None:
        preregister = json.loads(
            (PROJECT_DIR / "configs/covariate_shift_feature_pruning_preregister_v3.json").read_text(encoding="utf-8")
        )
        contract = preregister["physical_and_label_contract"]
        self.assertEqual(contract["suffix_bytes_returned_to_weather_parser"], 0)
        self.assertEqual(contract["future_weather_value_cells_materialized"], 0)
        self.assertIn("2024 labels", contract["forbidden"])
        self.assertIn("Public", contract["forbidden"])

    def test_v2_is_quarantined_and_old_canonical_is_absent(self) -> None:
        failed = PROJECT_DIR / "artifacts/postgate/covariate_shift_feature_pruning_strict_v2_failed_provenance_coverage"
        canonical = PROJECT_DIR / "artifacts/postgate/covariate_shift_feature_pruning_strict_v2"
        self.assertTrue(failed.is_dir())
        self.assertFalse(canonical.exists())
        incident = json.loads((failed / "INCIDENT_LEDGER.json").read_text(encoding="utf-8"))
        self.assertFalse(incident["affected_scope"]["source_provenance_audit_passed"])
        self.assertEqual(len(incident["missing_source_files"]), 8)


if __name__ == "__main__":
    unittest.main()

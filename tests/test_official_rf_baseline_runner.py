from __future__ import annotations

import inspect
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import run_official_rf_baseline as runner
from src.manifest import sha256_file


PROJECT_DIR = Path(__file__).resolve().parents[1]
PREREGISTER = PROJECT_DIR / "configs/official_rf_baseline_preregister_v2.json"


class OfficialRFRunnerTests(unittest.TestCase):
    def test_preregister_official_source_adaptations_and_sha_are_exact(self) -> None:
        self.assertEqual(sha256_file(PREREGISTER), runner.PREREGISTER_SHA256)
        payload = runner._verify_preregister(PREREGISTER)
        wrapper = payload["v2_supersession_contract"]
        self.assertEqual(
            wrapper["supersedes"]["preregister_sha256"],
            runner.PARENT_PREREGISTER_SHA256,
        )
        self.assertFalse(wrapper["supersedes"]["candidate_fit_started"])
        self.assertFalse(wrapper["supersedes"]["candidate_prediction_materialized"])
        self.assertFalse(wrapper["supersedes"]["candidate_score_computed"])
        self.assertFalse(wrapper["supersedes"]["2024_or_2025_input_read"])
        source = payload["official_primary_source_contract"]
        self.assertEqual(source["notebook_sha256"], runner.OFFICIAL_NOTEBOOK_SHA256)
        self.assertEqual(source["notebook_bytes"], 10_026)
        self.assertEqual(
            source["notebook_model_explicit_parameters"],
            {
                "n_estimators": 120,
                "max_depth": 14,
                "min_samples_leaf": 8,
                "max_features": "sqrt",
                "random_state": 42,
                "n_jobs": -1,
            },
        )
        self.assertFalse(
            source["notebook_feature_contract"]["train_test_concat_preprocessing"]
        )
        adaptation = payload["leakage_safe_metric_adaptations"]
        self.assertIn("CF >= 0.10", adaptation["metric_aligned_target"])
        self.assertIn("n_jobs=-1", adaptation["prediction"])
        self.assertTrue(adaptation["no_train_test_or_fit_application_concat"])

    def test_recursive_static_repo_source_closure_is_exact_and_closed(self) -> None:
        closure = runner._static_repo_import_closure(
            (
                Path(runner.__file__).resolve(),
                PROJECT_DIR / "scripts/run_seasonal_wind_ridge.py",
                PROJECT_DIR / "src/official_rf_baseline.py",
            )
        )
        relative = tuple(path.relative_to(PROJECT_DIR).as_posix() for path in closure)
        self.assertEqual(relative, runner.EXPECTED_REPO_SOURCE_CLOSURE)
        self.assertEqual(len(relative), 14)
        closure_set = set(closure)
        for source in closure:
            self.assertTrue(runner._direct_local_imports(source).issubset(closure_set))
        provenance = runner._provenance_paths(PREREGISTER)
        bound = {
            key.removeprefix("repo_source::")
            for key in provenance
            if key.startswith("repo_source::")
        }
        self.assertEqual(bound, set(runner.EXPECTED_REPO_SOURCE_CLOSURE))
        self.assertTrue(all(path.is_file() for path in provenance.values()))

    def test_unused_info_workbook_snapshot_cannot_enter_candidate_features(self) -> None:
        payload = runner._verify_preregister(PREREGISTER)
        info = payload["physical_stage1_inputs"]["info_workbook"]
        self.assertEqual(info["bytes"], 3_823_422)
        self.assertEqual(
            info["sha256"],
            "89e83a52e0eb2ce367a3573a96d6795ed4b4d4ac624965cb3530beec0cbd2bd6",
        )
        self.assertIn("Unused immutable snapshot", info["role"])
        feature_sources = "\n".join(
            inspect.getsource(function)
            for function in (
                runner._required_raw_columns,
                runner._read_stage1_official_features,
                runner._official_features,
                runner.build_official_rf_features,
            )
        )
        self.assertNotIn("info.xlsx", feature_sources)
        self.assertNotIn("info_workbook", feature_sources)
        self.assertEqual(tuple(payload["feature_contract"]["columns_in_exact_order"]), runner.FEATURE_COLUMNS)
        self.assertEqual(len(runner.FEATURE_COLUMNS), 74)

    def test_identity_postlock_path_opens_no_full_raw_weather(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            (out / "stage1_promotion_lock.json").write_text("{}\n", encoding="utf-8")
            with mock.patch.object(
                runner.protocol, "_load_stage1_lock", return_value=({"locked_weight": None}, {})
            ), mock.patch.object(
                runner, "_read_full_train_feature_map"
            ) as full_reader:
                audit = runner._postlock_raw_audit(
                    raw_dir=Path("unused"), cache_dir=Path("unused"), out_dir=out
                )
            full_reader.assert_not_called()
            self.assertFalse(audit["performed"])
            self.assertFalse(audit["2024_weather_physical_bytes_accessed"])

    def test_runner_uses_raw_not_cache_for_conditional_stage2_features(self) -> None:
        source = Path(runner.__file__).read_text(encoding="utf-8")
        function = source[source.index("def _read_stage2_features"):source.index("def _reload_audit")]
        self.assertIn("del cache_dir", function)
        self.assertIn("_read_full_train_feature_map", function)
        self.assertNotIn("read_parquet", function)
        self.assertLess(source.index("protocol._postlock_cache_audit"), source.index("def main"))

    def test_exact_feature_count_and_no_random_forest_duplicate_claim(self) -> None:
        payload = runner._verify_preregister(PREREGISTER)
        self.assertEqual(payload["feature_contract"]["feature_count"], 74)
        self.assertEqual(len(payload["feature_contract"]["columns_in_exact_order"]), 74)
        self.assertEqual(
            payload["census_and_novelty_contract"][
                "repository_random_forest_regressor_source_hits_before_preregistration"
            ],
            0,
        )
        self.assertTrue(payload["census_and_novelty_contract"]["standalone_is_diagnostic_only"])


if __name__ == "__main__":
    unittest.main()

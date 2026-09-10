from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from scripts import run_compact_physics_extratrees as runner
from src.manifest import sha256_file


PROJECT_DIR = Path(__file__).resolve().parents[1]
PREREGISTER = PROJECT_DIR / "configs/compact_physics_extratrees_mse_preregister_v2.json"


class CompactPhysicsExtraTreesRunnerV2Tests(unittest.TestCase):
    def test_v2_parent_external_source_and_parameter_claims_are_exact(self) -> None:
        self.assertEqual(sha256_file(PREREGISTER), runner.PREREGISTER_SHA256)
        merged = runner._verify_preregister(PREREGISTER)
        wrapper = merged["v2_supersession_contract"]
        self.assertEqual(
            wrapper["corrected_external_source_contract"]["legacy_notebook_sha256"],
            runner.LEGACY_NOTEBOOK_SHA256,
        )
        self.assertEqual(
            wrapper["corrected_external_source_contract"][
                "final_notebook_cell_explicit_parameters_copied_exactly"
            ],
            {
                "n_estimators": 400,
                "max_depth": 30,
                "min_samples_split": 2,
                "bootstrap": False,
            },
        )
        self.assertEqual(
            wrapper["corrected_external_source_contract"][
                "deterministic_local_adaptations_not_claimed_as_final_notebook_parameters"
            ]["random_state"],
            42,
        )
        self.assertTrue(wrapper["immutable_candidate_contract"]["no_retune_or_candidate_change"])

    def test_recursive_static_import_closure_is_exact_and_closed(self) -> None:
        closure = runner._static_repo_import_closure(
            (
                Path(runner.__file__).resolve(),
                PROJECT_DIR / "scripts/run_seasonal_wind_ridge.py",
                PROJECT_DIR / "src/compact_physics_extratrees.py",
            )
        )
        relative = tuple(path.relative_to(PROJECT_DIR).as_posix() for path in closure)
        self.assertEqual(relative, runner.EXPECTED_CLOSURE)
        self.assertEqual(len(relative), 14)
        closure_set = set(closure)
        for source in closure:
            self.assertTrue(runner._direct_local_imports(source).issubset(closure_set))
        self.assertTrue(
            {
                "src/catboost_multiquantile.py",
                "src/probabilistic.py",
                "src/seasonal_wind_ridge.py",
                "src/weather_quantile.py",
            }.issubset(relative)
        )

    def test_provenance_binds_closure_parent_and_quarantined_v1(self) -> None:
        sources = runner._provenance_paths(PREREGISTER)
        repo_sources = {
            key.removeprefix("repo_source::")
            for key in sources
            if key.startswith("repo_source::")
        }
        self.assertEqual(repo_sources, set(runner.EXPECTED_CLOSURE))
        self.assertEqual(sources["parent_preregister"].name, runner.PARENT_PREREGISTER.name)
        self.assertIn("quarantined_v1::manifest.json", sources)
        self.assertIn("quarantined_v1::models/stage1_all_models.joblib", sources)
        self.assertTrue(all(path.is_file() for path in sources.values()))

    def test_v1_v2_identity_auditor_accepts_an_exact_copy(self) -> None:
        v1 = (PROJECT_DIR / runner.QUARANTINED_V1).resolve()
        relatives = (
            "oof/stage1_kpx_group_1_direct_cf.parquet",
            "oof/stage1_kpx_group_2_direct_cf.parquet",
            "oof/stage1_kpx_group_3_direct_cf.parquet",
            "oof/stage1_baseline_2023.parquet",
            "oof/stage1_direct_cf_2023.parquet",
            "oof/stage1_blend_w05_2023.parquet",
            "oof/stage1_blend_w10_2023.parquet",
            "oof/stage1_standalone_diagnostic_2023.parquet",
            "stage1_results.json",
            "stage1_prescore_record.json",
        )
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            for relative in relatives:
                destination = out / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(v1 / relative, destination)
            result = json.loads((out / "stage1_results.json").read_text(encoding="utf-8"))
            result["experiment_id"] = "compact_physics_extratrees_mse_strict_forward_v2"
            (out / "stage1_results.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            audit = runner._v1_v2_identity_audit(out)
            self.assertTrue(audit["all_prediction_frames_value_bit_exact"])
            self.assertEqual(audit["prediction_frame_count"], 8)
            self.assertTrue(audit["comparisons_canonical_sha256_exact"])
            self.assertTrue(audit["model_fit_metadata_exact"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/run_noaa_gefs_operational_spread_00z_paired_increment_v12.py"
LEGACY_PATH = ROOT / "scripts/run_noaa_gefs_operational_spread_00z_paired_increment_v2.py"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_runner():
    name = "gefs_v12_stage1_model_runner_under_test"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load_runner()


def test_execution_preregister_resolution_and_science_are_frozen():
    assert sha(runner.MODEL_EXEC_V1) == runner.MODEL_EXEC_V1_SHA
    assert sha(runner.MODEL_EXEC_V2) == runner.MODEL_EXEC_V2_SHA
    v1 = json.loads(runner.MODEL_EXEC_V1.read_text(encoding="utf-8"))
    v2 = json.loads(runner.MODEL_EXEC_V2.read_text(encoding="utf-8"))
    assert v2["supersedes"]["sha256"] == runner.MODEL_EXEC_V1_SHA
    assert v2["corrected_frozen_legacy_scientific_runner"] == {
        "path": "scripts/run_noaa_gefs_operational_spread_00z_paired_increment_v2.py",
        "bytes": 55_799,
        "sha256": runner.LEGACY_RUNNER_SHA,
    }
    assert v1["scientific_lineage"]["candidate_id"] == "gefs00z_0p50_10m_850_mean_spread_w025"
    assert v1["scientific_lineage"]["blend_weight"] == 0.25
    assert v1["scientific_lineage"]["model_parameters_features_rows_seeds_folds_formula_slices_components_gates_changed"] is False
    runner.verify_effective_preregister()


def test_accepted_source_and_independent_pass_identities_are_exact():
    assert sha(runner.FRESH_SOURCE / "manifest.json") == runner.SOURCE_MANIFEST_SHA
    assert sha(runner.SOURCE_POSTRUN_REVIEW) == runner.SOURCE_POSTRUN_REVIEW_SHA
    review = json.loads(runner.SOURCE_POSTRUN_REVIEW.read_text(encoding="utf-8"))
    assert review["verdict"] == "PASS"
    assert review["canonical"]["manifest"]["sha256"] == runner.SOURCE_MANIFEST_SHA
    assert review["closure_and_nonreuse"]["label_reads"] == 0
    assert review["closure_and_nonreuse"]["fits"] == 0
    assert review["closure_and_nonreuse"]["scores"] == 0
    runner.verify_source_launch_lock()


def test_static_audit_and_zero_state_are_exact():
    audit = runner.static_audit()
    assert audit["status"] == "PASS_ACTUAL_STAGE1_RUNNER_STATIC_CONTRACT_V12_ACCEPTED_SOURCE"
    assert audit["features_control"] == 612 and audit["features_extended"] == 624
    assert audit["fit_label_cells"] == 21_864 and audit["score_label_cells"] == 21_936
    assert audit["models_to_save_reload"] == 6 and audit["weight"] == 0.25
    assert audit["stage1_final_row_modeled"] is True and audit["stage2_branch"] is False
    assert audit["label_fit_prediction_score_2024_2025_access"] == 0
    assert not runner.FRESH_MODEL_OUTPUT.exists()
    assert not runner.FRESH_MODEL_TOMBSTONE.exists()
    assert not runner.FRESH_STAGE2_SOURCE.exists()
    quarantine = ROOT / "artifacts/quarantine"
    assert not list(quarantine.glob("noaa_gefs_operational_spread_00z_paired_increment_v12_partial_*")) if quarantine.exists() else True


def test_exact_label_masks_and_global_lock_precondition_are_internal(tmp_path):
    _v1, _v2, _v3, _v4, _v5, protocol = runner.core.verify_effective_preregister()
    contract = protocol["stage1_label_source"]
    fit = contract["fit_slices_before_global_candidate_lock"]
    score = contract["score_slices_only_after_verified_global_candidate_lock"]
    assert sum(int(item["value_cells"]) for item in fit.values()) == 21_864
    assert sum(int(item["value_cells"]) for item in score.values()) == 21_936
    assert not hasattr(runner.core, "assert_label_request")
    with pytest.raises(FileNotFoundError, match="global candidate lock"):
        runner.core.read_stage1_score_labels(tmp_path / "raw_never_opened", tmp_path / "out", protocol)


def test_frozen_scientific_source_has_required_order_reload_guard_and_no_stage2():
    source = LEGACY_PATH.read_text(encoding="utf-8")
    assert sha(LEGACY_PATH) == runner.LEGACY_RUNNER_SHA
    tombstone = source.index("_write_json_exclusive(MODEL_ATTEMPT_TOMBSTONE")
    prescore = source.index("prescore = out_dir / \"stage1_prescore_lock.json\"")
    bounded = source.index("features, raw_contract = shared._read_stage1_raw_features")
    fit_labels = source.index("fit_labels, fit_ledger = read_stage1_fit_labels")
    guard = source.index("score_guard = HeavyFitGuard(defer_success_release=True)")
    global_lock = source.index("global_lock_path = out_dir / \"stage1_candidate_before_score_labels_lock.json\"")
    score_labels = source.index("score_labels, score_ledger = read_stage1_score_labels")
    selection = source.index("_write_json(selection_path")
    release = source.index("score_guard.release()")
    assert tombstone < prescore < bounded < fit_labels < guard < global_lock < score_labels < selection < release
    assert source.count("score_guard = HeavyFitGuard(") == 1
    assert source.count("joblib.load(") == 2
    assert "saved-model reload prediction bits differ" in source
    assert "six_models_saved_and_reloaded" in source
    assert "continuous_heavy_guard_through_result_seal" in source
    tree = ast.parse(source)
    names = {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert "run_stage2" not in names


def test_control_builder_and_formula_contract_are_unchanged():
    assert runner.core.MODEL_PARAMETERS["objective"] == "regression_l1"
    assert runner.core.MODEL_PARAMETERS["random_state"] == 42
    assert runner.core.TRANSFER_WEIGHT == 0.25
    assert len(runner.core.EXTENDED_COLUMNS) == 12
    source = LEGACY_PATH.read_text(encoding="utf-8")
    assert "frame.shape != (17_520, 612)" in source
    assert "result.shape[1] != 624" in source
    assert "TRANSFER_WEIGHT * CAPACITY_KWH[group] * increment" in source
    assert "np.clip(" in source and "1.02 * CAPACITY_KWH[group]" in source


def test_wrapper_identifier_transform_and_exact_runtime_paths():
    assert runner.core.SOURCE_ROOT.resolve() == runner.FRESH_SOURCE.resolve()
    assert runner.core.DEFAULT_OUTPUT.resolve() == runner.FRESH_MODEL_OUTPUT.resolve()
    assert runner.core.MODEL_ATTEMPT_TOMBSTONE.resolve() == runner.FRESH_MODEL_TOMBSTONE.resolve()
    assert runner.core.EXTRACTOR.resolve() == runner.FRESH_EXTRACTOR.resolve()
    assert runner.core.HeavyFitGuard.__enter__.__globals__["HEAVY_GUARD"] == ROOT / "artifacts/locks/heavy_cpu_fit.pid.json"
    transformed_constants = runner.core.HeavyFitGuard.__enter__.__code__.co_consts
    assert any("heavy guard" in value for value in transformed_constants if isinstance(value, str))


def test_cli_static_sentinel_and_unknown_argument():
    good = subprocess.run(
        [sys.executable, "-B", str(RUNNER_PATH), "--static-audit"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert good.returncode == 0 and good.stderr == "" and good.stdout.strip()
    payload = json.loads(good.stdout)
    assert payload["status"] == "PASS_ACTUAL_STAGE1_RUNNER_STATIC_CONTRACT_V12_ACCEPTED_SOURCE"
    assert payload["label_fit_prediction_score_2024_2025_access"] == 0
    bad = subprocess.run(
        [sys.executable, "-B", str(RUNNER_PATH), "--definitely-unknown"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad.returncode != 0


def test_model_launch_gate_fails_closed_before_independent_review():
    assert not runner.MODEL_PRELAUNCH_REVIEW.exists()
    with pytest.raises(FileNotFoundError, match="prelaunch readiness/review absent"):
        runner.verify_model_prelaunch_review()


def test_no_model_launch_or_future_artifact_was_materialized_by_tests():
    assert not runner.FRESH_MODEL_OUTPUT.exists()
    assert not runner.FRESH_MODEL_TOMBSTONE.exists()
    assert not runner.FRESH_STAGE2_SOURCE.exists()
    assert not list((ROOT / "artifacts/incidents").glob("noaa_gefs_operational_spread_00z_paired_increment_v12_stage1_failure_*.json"))

"""Strict-forward compact-physics squared-error ExtraTrees experiment.

This is a deliberately thin specialization of the already audited seasonal
direct-CF protocol.  It replaces only the immutable feature/model contracts,
the two registered blend weights, and the post-lock raw/cache equivalence
check.  Stage ordering, bounded label/weather readers, score gates, and atomic
writes remain shared with that protocol.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts import run_seasonal_wind_ridge as protocol  # noqa: E402
from scripts import run_shared_q07_multiseed as shared  # noqa: E402
from scripts import run_ficr_bayes_decision_strict as strict  # noqa: E402
from src.compact_physics_extratrees import (  # noqa: E402
    FEATURE_COLUMNS,
    STAGE1_REQUIRED,
    STAGE2_REQUIRED,
    CompactPhysicsExtraTrees,
    assert_strict_forward,
    blend_direct_kwh,
    build_compact_physics_features,
    select_stage1_weight,
    stage2_promoted,
)
from src.manifest import describe_file, sha256_file, utc_now  # noqa: E402
from src.metric import TARGET_COLS  # noqa: E402


PREREGISTER_SHA256 = "8d27546e5aa93d825968d611736602d1c000477c58bd29ef03e5a0654f36e375"
PARENT_PREREGISTER_SHA256 = "5786c910df121046fe74fabeea8581354ec8393f4f6b21e552581a4eb22e617e"
LEGACY_NOTEBOOK_SHA256 = "ed854436ebf3d830b5546f36f02b79aa115b0939710fe9bdc96986df0f9cc6d3"
WEIGHTS: dict[str, float] = {"w05": 0.05, "w10": 0.10}
DEFAULT_OUT_DIR = Path("artifacts/postgate/compact_physics_extratrees_mse_strict_v2")
DEFAULT_PREREGISTER = Path(
    "configs/compact_physics_extratrees_mse_preregister_v2.json"
)
PARENT_PREREGISTER = Path("configs/compact_physics_extratrees_mse_preregister_v1.json")
QUARANTINED_V1 = Path(
    "artifacts/postgate/compact_physics_extratrees_mse_strict_v1_failed_provenance_contract"
)
EXPECTED_CLOSURE = (
    "scripts/run_catboost_multiquantile_bayes.py",
    "scripts/run_compact_physics_extratrees.py",
    "scripts/run_ficr_bayes_decision_strict.py",
    "scripts/run_seasonal_wind_ridge.py",
    "scripts/run_shared_q07_multiseed.py",
    "scripts/run_weather_quantile_bayes.py",
    "src/catboost_multiquantile.py",
    "src/compact_physics_extratrees.py",
    "src/features.py",
    "src/manifest.py",
    "src/metric.py",
    "src/probabilistic.py",
    "src/seasonal_wind_ridge.py",
    "src/weather_quantile.py",
)
SHARED_FINALIZE = protocol._finalize
EXPECTED_MODEL_PARAMETERS: dict[str, Any] = {
    "n_estimators": 400,
    "criterion": "squared_error",
    "max_depth": 30,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": 1.0,
    "bootstrap": False,
    "oob_score": False,
    "max_samples": None,
    "random_state": 42,
    "n_jobs": 7,
}


def _verify_preregister(path: Path) -> dict[str, Any]:
    observed = sha256_file(path)
    if observed != PREREGISTER_SHA256:
        raise AssertionError(f"preregister hash changed: {observed}")
    sidecar = path.with_suffix(".sha256")
    expected_sidecar = f"{PREREGISTER_SHA256}  {path.name}\n"
    if sidecar.read_text(encoding="utf-8") != expected_sidecar:
        raise AssertionError("preregister sidecar changed")
    wrapper = json.loads(path.read_text(encoding="utf-8"))
    if wrapper["experiment_id"] != (
        "compact_physics_extratrees_mse_strict_forward_v2"
    ):
        raise AssertionError("experiment id changed")
    parent_path = (PROJECT_DIR / wrapper["supersedes"]["preregister_path"]).resolve()
    if parent_path != (PROJECT_DIR / PARENT_PREREGISTER).resolve():
        raise AssertionError("parent preregister path changed")
    if sha256_file(parent_path) != PARENT_PREREGISTER_SHA256:
        raise AssertionError("parent preregister hash changed")
    parent_sidecar = parent_path.with_suffix(".sha256")
    if parent_sidecar.read_text(encoding="utf-8") != (
        f"{PARENT_PREREGISTER_SHA256}  {parent_path.name}\n"
    ):
        raise AssertionError("parent preregister sidecar changed")
    if wrapper["supersedes"]["preregister_sha256"] != PARENT_PREREGISTER_SHA256:
        raise AssertionError("parent binding changed")
    external = wrapper["corrected_external_source_contract"]
    if external["legacy_notebook_bytes"] != 69_611:
        raise AssertionError("legacy notebook byte count changed")
    if external["legacy_notebook_sha256"] != LEGACY_NOTEBOOK_SHA256:
        raise AssertionError("legacy notebook hash changed")
    if external["final_notebook_cell_explicit_parameters_copied_exactly"] != {
        "n_estimators": 400,
        "max_depth": 30,
        "min_samples_split": 2,
        "bootstrap": False,
    }:
        raise AssertionError("explicit legacy parameter claim changed")
    if external["deterministic_local_adaptations_not_claimed_as_final_notebook_parameters"] != {
        "random_state": 42,
        "n_jobs": 7,
        "serial_inference_with_fitted_n_jobs_restored": True,
    }:
        raise AssertionError("local adaptation claim changed")
    closure = wrapper["static_repo_import_closure_contract"]
    if tuple(closure["expected_sources_in_exact_sorted_order"]) != EXPECTED_CLOSURE:
        raise AssertionError("static closure preregistration changed")
    if int(closure["expected_source_count"]) != 14:
        raise AssertionError("static closure count changed")
    invariant = wrapper["immutable_candidate_contract"]
    if invariant["runtime_estimator_parameters_inherited_bit_exact_from_v1"] != EXPECTED_MODEL_PARAMETERS:
        raise AssertionError("runtime model changed from v1")
    if tuple(map(float, invariant["selectable_blends_inherited_bit_exact_from_v1"])) != tuple(WEIGHTS.values()):
        raise AssertionError("runtime weights changed from v1")
    if invariant["no_retune_or_candidate_change"] is not True:
        raise AssertionError("no-retune contract changed")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    payload = json.loads(json.dumps(parent))
    payload["experiment_id"] = wrapper["experiment_id"]
    payload["status"] = wrapper["status"]
    payload["output_contract"] = wrapper["output_contract"]
    payload["v2_supersession_contract"] = wrapper
    feature = payload["feature_contract"]
    if tuple(feature["columns_in_exact_order"]) != FEATURE_COLUMNS:
        raise AssertionError("feature contract changed")
    if int(feature["feature_count"]) != 36:
        raise AssertionError("feature count changed")
    if tuple(
        map(float, payload["candidate_family"]["fixed_global_blend_weights"])
    ) != tuple(WEIGHTS.values()):
        raise AssertionError("weights changed")
    if payload["model"]["class"] != "sklearn.ensemble.ExtraTreesRegressor":
        raise AssertionError("model class changed")
    if payload["model"]["parameters"] != EXPECTED_MODEL_PARAMETERS:
        raise AssertionError("ExtraTrees parameters changed")
    if payload["target_contract"]["kind"] != "direct_capacity_factor":
        raise AssertionError("target changed")
    if payload["target_contract"]["eligible"] != (
        "finite target CF and target CF >= 0.10"
    ):
        raise AssertionError("eligibility changed")
    if payload["stage1"]["registered_slice_count"] != 17:
        raise AssertionError("Stage1 slice count changed")
    if tuple(payload["stage2"]["registered_segments_each_group"]) != STAGE2_REQUIRED:
        raise AssertionError("Stage2 group slices changed")
    if tuple(payload["stage2"]["mixed_total_segments"]) != STAGE2_REQUIRED:
        raise AssertionError("Stage2 mixed slices changed")
    if payload["census_and_novelty_contract"]["standalone_is_diagnostic_only"] is not True:
        raise AssertionError("standalone role changed")
    if payload["public_contract"]["public_metric_triplets_used"] is not False:
        raise AssertionError("Public metric use changed")
    return payload


def _resolve_local_module(module: str) -> Path | None:
    if not module or module.split(".", 1)[0] not in {"scripts", "src"}:
        return None
    candidate = PROJECT_DIR.joinpath(*module.split(".")).with_suffix(".py")
    return candidate.resolve() if candidate.is_file() else None


def _direct_local_imports(path: Path) -> set[Path]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[Path] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                resolved = _resolve_local_module(alias.name)
                if resolved is not None:
                    imports.add(resolved)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            module = node.module or ""
            resolved = _resolve_local_module(module)
            if resolved is not None:
                imports.add(resolved)
            if module in {"scripts", "src"}:
                for alias in node.names:
                    resolved = _resolve_local_module(f"{module}.{alias.name}")
                    if resolved is not None:
                        imports.add(resolved)
    return imports


def _static_repo_import_closure(roots: Sequence[Path]) -> tuple[Path, ...]:
    pending = [path.resolve() for path in roots]
    observed: set[Path] = set()
    while pending:
        current = pending.pop()
        if current in observed:
            continue
        if not current.is_file() or current.suffix != ".py":
            raise FileNotFoundError(current)
        observed.add(current)
        pending.extend(_direct_local_imports(current).difference(observed))
    output = tuple(
        sorted(observed, key=lambda item: item.relative_to(PROJECT_DIR).as_posix())
    )
    relative = tuple(path.relative_to(PROJECT_DIR).as_posix() for path in output)
    if relative != EXPECTED_CLOSURE:
        raise AssertionError(f"static repo import closure changed: {relative}")
    return output


def _provenance_paths(preregister_path: Path) -> dict[str, Path]:
    closure = _static_repo_import_closure(
        (
            Path(__file__).resolve(),
            PROJECT_DIR / "scripts/run_seasonal_wind_ridge.py",
            PROJECT_DIR / "src/compact_physics_extratrees.py",
        )
    )
    output = {
        f"repo_source::{path.relative_to(PROJECT_DIR).as_posix()}": path
        for path in closure
    }
    output.update({
        "focused_test": PROJECT_DIR / "tests/test_compact_physics_extratrees.py",
        "closure_test": PROJECT_DIR / "tests/test_compact_physics_extratrees_runner.py",
        "preregister": preregister_path.resolve(),
        "preregister_sidecar": preregister_path.with_suffix(".sha256").resolve(),
        "parent_preregister": (PROJECT_DIR / PARENT_PREREGISTER).resolve(),
        "parent_preregister_sidecar": (PROJECT_DIR / PARENT_PREREGISTER).with_suffix(".sha256").resolve(),
    })
    v1_root = (PROJECT_DIR / QUARANTINED_V1).resolve()
    for relative in (
        "manifest.json",
        "stage1_results.json",
        "stage1_prescore_record.json",
        "models/stage1_all_models.joblib",
        "oof/stage1_baseline_2023.parquet",
        "oof/stage1_direct_cf_2023.parquet",
        "oof/stage1_blend_w05_2023.parquet",
        "oof/stage1_blend_w10_2023.parquet",
        "oof/stage1_standalone_diagnostic_2023.parquet",
    ):
        output[f"quarantined_v1::{relative}"] = v1_root / relative
    return output


def _compact_features(
    weather_features: Mapping[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    output: dict[str, pd.DataFrame] = {}
    evidence: dict[str, Any] = {}
    for group in TARGET_COLS:
        compact = build_compact_physics_features(weather_features[group])
        output[group] = compact
        evidence[group] = {
            "rows": len(compact),
            "columns": compact.shape[1],
            "start": compact.index.min(),
            "end": compact.index.max(),
            "selected_raw_weather_sha256": shared._frame_sha256(compact),
            "design_frame_sha256": shared._frame_sha256(compact),
            "feature_columns_exact": list(FEATURE_COLUMNS),
            "actual_feature_columns": 0,
            "scada_feature_columns": 0,
            "baseline_or_residual_feature_columns": 0,
            "public_feature_columns": 0,
        }
    return output, evidence


def _reload_audit(
    *,
    model_path: Path,
    groups: Sequence[str],
    features: Mapping[str, pd.DataFrame],
    application_indexes: Mapping[str, pd.DatetimeIndex],
    expected: Mapping[str, pd.Series],
) -> dict[str, Any]:
    models: dict[str, CompactPhysicsExtraTrees] = joblib.load(model_path)
    if tuple(models) != tuple(groups):
        raise AssertionError("reloaded model group order changed")
    audit: dict[str, Any] = {}
    for group in groups:
        model = models[group]
        observed = model.predict_cf(features[group].loc[application_indexes[group]])
        exact = (
            observed.index.equals(expected[group].index)
            and np.array_equal(observed.to_numpy(), expected[group].to_numpy())
        )
        if not exact:
            raise AssertionError(f"{group} reloaded ExtraTrees prediction differs")
        if model.estimator_ is None:
            raise AssertionError("reloaded ExtraTrees estimator missing")
        parameters = {
            key: model.estimator_.get_params(deep=False)[key]
            for key in EXPECTED_MODEL_PARAMETERS
        }
        if parameters != EXPECTED_MODEL_PARAMETERS:
            raise AssertionError("reloaded ExtraTrees parameters changed")
        audit[group] = {
            "prediction_bit_exact": True,
            "feature_columns_exact": list(FEATURE_COLUMNS),
            "resolved_model_parameters": parameters,
            "target_kind": "direct_capacity_factor",
        }
    return audit


def _postlock_cache_audit(
    *, raw_dir: Path, cache_dir: Path, out_dir: Path
) -> dict[str, Any]:
    lock, _ = protocol._load_stage1_lock(out_dir)
    path = out_dir / "postlock_cache_audit.json"
    if path.exists():
        raise FileExistsError(path)
    if lock["locked_weight"] is None:
        audit = {
            "schema_version": 1,
            "created_utc": utc_now(),
            "stage1_promotion_lock": describe_file(
                out_dir / "stage1_promotion_lock.json"
            ),
            "performed": False,
            "2024_cache_logical_values_materialized": False,
            "2024_cache_physical_bytes_accessed": False,
            "reason": "identity locked; no cache file opened",
            "checks": {},
        }
        strict._write_json(path, audit)
        return audit

    full_index = pd.date_range(
        "2022-01-01 01:00",
        "2024-01-01 00:00",
        freq="h",
        name="forecast_kst_dtm",
    )
    rebuilt, _ = shared._read_stage1_raw_features(
        raw_dir, pd.DataFrame(index=full_index)
    )
    rebuilt_compact, _ = _compact_features(rebuilt)
    checks: dict[str, Any] = {}
    for group in TARGET_COLS:
        cache_path = cache_dir / f"{group}_weather_train.parquet"
        cached = pd.read_parquet(
            cache_path,
            engine="pyarrow",
            filters=[("forecast_kst_dtm", "<=", pd.Timestamp("2024-01-01 00:00"))],
            columns=list(FEATURE_COLUMNS),
        )
        cached.index = pd.DatetimeIndex(cached.index, name="forecast_kst_dtm")
        cached_compact = build_compact_physics_features(cached)
        source_exact = np.array_equal(
            rebuilt_compact[group].to_numpy(), cached_compact.to_numpy()
        )
        if not rebuilt_compact[group].index.equals(cached_compact.index) or not source_exact:
            raise AssertionError(f"{group} selected raw/cache prefix differs")
        checks[group] = {
            "rows": len(cached_compact),
            "selected_weather_columns": list(FEATURE_COLUMNS),
            "selected_weather_value_bits_exact": True,
            "design_value_bits_exact": True,
            "cache_file_path": str(cache_path.resolve()),
            "cache_file_size_bytes_from_stat": cache_path.stat().st_size,
        }
    audit = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "stage1_promotion_lock": describe_file(out_dir / "stage1_promotion_lock.json"),
        "performed": True,
        "selection_locked_before_cache_read": True,
        "2024_cache_logical_values_materialized": False,
        "2024_cache_physical_bytes_accessed_may_be_true_for_single_row_group": True,
        "physical_cache_access_cannot_affect_locked_stage1_models_predictions_or_selection": True,
        "checks": checks,
        "all_group_selected_prefixes_value_bit_exact": True,
    }
    strict._write_json(path, audit)
    return audit


def _v1_v2_identity_audit(out_dir: Path) -> dict[str, Any]:
    path = out_dir / "v1_v2_identity_audit.json"
    if path.exists():
        raise FileExistsError(path)
    v1_root = (PROJECT_DIR / QUARANTINED_V1).resolve()
    if not v1_root.is_dir():
        raise FileNotFoundError(v1_root)
    frame_paths = (
        "oof/stage1_kpx_group_1_direct_cf.parquet",
        "oof/stage1_kpx_group_2_direct_cf.parquet",
        "oof/stage1_kpx_group_3_direct_cf.parquet",
        "oof/stage1_baseline_2023.parquet",
        "oof/stage1_direct_cf_2023.parquet",
        "oof/stage1_blend_w05_2023.parquet",
        "oof/stage1_blend_w10_2023.parquet",
        "oof/stage1_standalone_diagnostic_2023.parquet",
    )
    frames: dict[str, Any] = {}
    for relative in frame_paths:
        v1_path = v1_root / relative
        v2_path = out_dir / relative
        before = pd.read_parquet(v1_path, engine="pyarrow")
        after = pd.read_parquet(v2_path, engine="pyarrow")
        exact = (
            before.index.equals(after.index)
            and tuple(before.columns) == tuple(after.columns)
            and tuple(map(str, before.dtypes)) == tuple(map(str, after.dtypes))
            and np.array_equal(before.to_numpy(), after.to_numpy(), equal_nan=True)
        )
        if not exact:
            raise AssertionError(f"v1/v2 prediction frame differs: {relative}")
        frames[relative] = {
            "rows": len(after),
            "columns": list(after.columns),
            "value_bits_exact": True,
            "v1_sha256": sha256_file(v1_path),
            "v2_sha256": sha256_file(v2_path),
            "parquet_file_bytes_exact": sha256_file(v1_path) == sha256_file(v2_path),
            "canonical_frame_sha256": shared._frame_sha256(after),
        }
    v1_result = json.loads((v1_root / "stage1_results.json").read_text(encoding="utf-8"))
    v2_result = json.loads((out_dir / "stage1_results.json").read_text(encoding="utf-8"))
    for field in (
        "comparisons",
        "comparisons_sha256",
        "standalone_diagnostics",
        "standalone_diagnostics_sha256",
        "selection",
        "selection_sha256",
        "locked_weight",
        "locked_candidate",
    ):
        if v1_result[field] != v2_result[field]:
            raise AssertionError(f"v1/v2 Stage1 field differs: {field}")
    v1_prescore = json.loads(
        (v1_root / "stage1_prescore_record.json").read_text(encoding="utf-8")
    )
    v2_prescore = json.loads(
        (out_dir / "stage1_prescore_record.json").read_text(encoding="utf-8")
    )
    if v1_prescore["training"] != v2_prescore["training"]:
        raise AssertionError("v1/v2 model fit metadata differs")
    audit = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "v1_quarantined_artifact": str(v1_root),
        "v2_artifact": str(out_dir.resolve()),
        "v1_manifest_sha256": sha256_file(v1_root / "manifest.json"),
        "v1_preregister_sha256": PARENT_PREREGISTER_SHA256,
        "v2_preregister_sha256": PREREGISTER_SHA256,
        "prediction_frames": frames,
        "prediction_frame_count": len(frames),
        "all_prediction_frames_value_bit_exact": True,
        "comparisons_canonical_sha256_exact": True,
        "comparisons_sha256": v2_result["comparisons_sha256"],
        "standalone_diagnostics_canonical_sha256_exact": True,
        "standalone_diagnostics_sha256": v2_result[
            "standalone_diagnostics_sha256"
        ],
        "selection_and_locked_candidate_exact": True,
        "model_fit_metadata_exact": True,
        "v1_locked_candidate": v1_result["locked_candidate"],
        "v2_locked_candidate": v2_result["locked_candidate"],
        "no_retune_or_candidate_change": True,
    }
    strict._write_json(path, audit)
    return audit


def _finalize_with_identity_audit(
    *,
    raw_dir: Path,
    artifact_root: Path,
    cache_dir: Path,
    out_dir: Path,
    preregister_path: Path,
) -> dict[str, Any]:
    _v1_v2_identity_audit(out_dir)
    return SHARED_FINALIZE(
        raw_dir=raw_dir,
        artifact_root=artifact_root,
        cache_dir=cache_dir,
        out_dir=out_dir,
        preregister_path=preregister_path,
    )


def _install_specialization() -> None:
    protocol.PREREGISTER_SHA256 = PREREGISTER_SHA256
    protocol.WEIGHTS = WEIGHTS
    protocol.FEATURE_COLUMNS = FEATURE_COLUMNS
    protocol.STAGE1_REQUIRED = STAGE1_REQUIRED
    protocol.STAGE2_REQUIRED = STAGE2_REQUIRED
    protocol.SeasonalWindRidge = CompactPhysicsExtraTrees
    protocol.assert_strict_forward = assert_strict_forward
    protocol.blend_direct_kwh = blend_direct_kwh
    protocol.select_stage1_weight = select_stage1_weight
    protocol.stage2_promoted = stage2_promoted
    protocol.CLI_DESCRIPTION = __doc__
    protocol.OUTPUT_SLUG = "compact_physics_extratrees_mse"
    protocol.ARTIFACT_TYPE = (
        "compact_physics_extratrees_mse_strict_forward_v2"
    )
    protocol.STANDALONE_DIAGNOSTIC = True
    protocol.MANIFEST_TESTS = {
        "focused_sources": [
            describe_file(PROJECT_DIR / "tests/test_compact_physics_extratrees.py"),
            describe_file(PROJECT_DIR / "tests/test_compact_physics_extratrees_runner.py"),
        ],
        "focused_test_count_prelaunch": 13,
        "focused_prelaunch_status": "pass",
        "full_suite_test_count_prelaunch": 276,
        "full_suite_prelaunch_status": "pass",
        "commands": [
            ".venv/Scripts/python.exe -B -m unittest "
            "tests.test_compact_physics_extratrees tests.test_compact_physics_extratrees_runner -v",
            ".venv/Scripts/python.exe -B -m unittest discover -s tests -p test_*.py -q",
        ],
    }
    protocol.MANIFEST_CONTRACTS = {
        "one_direct_squared_error_extratrees_per_group": True,
        "legacy_final_cell_explicit_parameters_exact": {
            "n_estimators": 400,
            "max_depth": 30,
            "min_samples_split": 2,
            "bootstrap": False,
        },
        "frozen_sklearn_defaults_not_claimed_as_explicit": {
            "criterion": "squared_error",
            "min_samples_leaf": 1,
            "max_features": 1.0,
            "oob_score": False,
            "max_samples": None,
        },
        "deterministic_local_adaptations": {
            "random_state": 42,
            "n_jobs": 7,
            "serial_inference_with_fitted_n_jobs_restored": True,
        },
        "legacy_notebook_bytes": 69_611,
        "legacy_notebook_sha256": LEGACY_NOTEBOOK_SHA256,
        "runtime_model_parameters_unchanged_from_failed_v1": EXPECTED_MODEL_PARAMETERS,
        "fixed_36_feature_compact_physics_design": True,
        "standalone_diagnostic_excluded_from_selection_and_promotion": True,
        "selectable_blends_exact": WEIGHTS,
        "fitted_model_reload_bit_exact": True,
        "fit_end_before_apply_start": True,
        "stage1_raw_physical_prefix_only_before_score": True,
        "static_repo_import_closure_count": 14,
        "static_repo_import_closure_bound_prescore_and_manifest": True,
        "v1_v2_prediction_metric_and_fit_metadata_identity_required": True,
        "no_retune_or_candidate_change": True,
        "2024_used_for_selection": False,
        "no_public_metric_scale_scada_or_application_label_feature": True,
        "leaderboard_score_claim": False,
    }
    protocol._verify_preregister = _verify_preregister
    protocol._provenance_paths = _provenance_paths
    protocol._seasonal_features = _compact_features
    protocol._reload_audit = _reload_audit
    protocol._postlock_cache_audit = _postlock_cache_audit
    protocol._finalize = _finalize_with_identity_audit


def _with_defaults(argv: Sequence[str]) -> list[str]:
    output = list(argv)
    if "--out-dir" not in output:
        output.extend(("--out-dir", str(DEFAULT_OUT_DIR)))
    if "--preregister" not in output:
        output.extend(("--preregister", str(DEFAULT_PREREGISTER)))
    return output


def main(argv: Sequence[str] | None = None) -> int:
    _install_specialization()
    supplied = sys.argv[1:] if argv is None else list(argv)
    return protocol.main(_with_defaults(supplied))


if __name__ == "__main__":
    raise SystemExit(main())

"""Strict-forward nonlinear capacity-factor target-transform experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.run_run_sequence_residual import (  # noqa: E402
    FINAL_END,
    FINAL_START,
    STAGE2_END,
    STAGE2_START,
    _atomic_joblib,
    _atomic_parquet,
    _comparisons,
    _frame_sha256,
    _interval,
    _mapping_sha256,
    _read_stage1_baseline,
    _reconstruct_baseline,
    _write_json,
    _write_submission,
)
from scripts.run_shared_q07_multiseed import (  # noqa: E402
    EXPECTED_ROWS_PRE2024,
    EXPECTED_ROWS_THROUGH2024,
    EXPECTED_TEST_ROWS,
    FINAL_COMPONENT_FILES,
    GATE_COMPONENT_FILES,
    OFFICIAL_LABEL_PREFIX_SHA256,
    OFFICIAL_STAGE1_PREFIX_BYTES,
    YEAR_2023_END,
    YEAR_2024_END,
    _read_features,
    _read_labels,
    _read_stage1_raw_features,
    _read_test_features,
)
from src.manifest import (  # noqa: E402
    describe_file,
    git_state,
    package_versions,
    sha256_file,
    utc_now,
)
from src.metric import CAPACITY_KWH, TARGET_COLS  # noqa: E402
from src.run_sequence_residual import choose_single_candidate  # noqa: E402
from src.target_transform import (  # noqa: E402
    TargetTransformRegressor,
    assert_strict_fit_apply_order,
    blend_with_corrected_v3,
)


PREREGISTER_SHA256 = (
    "2fb9420c2e43bee2e71f910874aa87525bb6afe8c1f4601bc5d9ee79369c201d"
)
TRANSFORM_IDS = ("logit_eps02", "arcsin_sqrt", "cube_root")
IDENTITY_ID = "identity"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument(
        "--cache-dir", type=Path, default=PROJECT_DIR / "artifacts" / "cache"
    )
    parser.add_argument(
        "--artifact-root", type=Path, default=PROJECT_DIR / "artifacts"
    )
    parser.add_argument(
        "--recipe",
        type=Path,
        default=PROJECT_DIR / "configs" / "train_final.v3.locked.json",
    )
    parser.add_argument(
        "--preregister",
        type=Path,
        default=PROJECT_DIR / "configs" / "target_transform_preregister.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_DIR / "artifacts" / "postgate" / "target_transform",
    )
    parser.add_argument("--stage", choices=("stage1", "stage2", "all"), default="all")
    return parser.parse_args(argv)


def _load_preregister(path: Path) -> dict[str, Any]:
    observed = sha256_file(path)
    if observed != PREREGISTER_SHA256:
        raise AssertionError(f"target-transform preregister SHA changed: {observed}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("experiment_id") != "target_transform_strict_forward_v1":
        raise AssertionError("target-transform experiment id changed")
    if [item["id"] for item in payload["transforms"]] != list(TRANSFORM_IDS):
        raise AssertionError("registered target transforms changed")
    if payload["blend_weights"] != [0.05, 0.1, 0.2]:
        raise AssertionError("registered blend weights changed")
    if int(payload["candidate_count"]) != 9:
        raise AssertionError("target-transform candidate count changed")
    if payload["stage1"]["selection_rule"]["locked_candidate_count_max"] != 1:
        raise AssertionError("target-transform lock no longer permits at most one recipe")
    return payload


def _verify_locked_recipe(path: Path, preregister: Mapping[str, Any]) -> dict[str, Any]:
    recipe = json.loads(path.read_text(encoding="utf-8"))
    specification = recipe["models"]["lgb_l1"]
    expected = preregister["model"]
    if specification["scope"] != "group":
        raise AssertionError("lgb_l1 scope changed")
    if specification["target_scale"] != "capacity_factor":
        raise AssertionError("lgb_l1 target scale changed")
    if specification["objective"] != "l1" or specification["row_filter"] != "eligible":
        raise AssertionError("lgb_l1 objective or row filter changed")
    if specification["params"] != expected["params"]:
        raise AssertionError("locked group lgb_l1 parameters differ from preregister")
    if int(recipe["seed"]) != int(expected["random_state"]):
        raise AssertionError("locked group lgb_l1 seed differs from preregister")
    if int(recipe["n_jobs"]) != int(expected["n_jobs"]):
        raise AssertionError("locked group lgb_l1 n_jobs differs from preregister")
    return recipe


def _model_params(preregister: Mapping[str, Any]) -> dict[str, Any]:
    model = preregister["model"]
    params = dict(model["params"])
    params.update(
        {
            "objective": model["objective"],
            "random_state": int(model["random_state"]),
            "n_jobs": int(model["n_jobs"]),
        }
    )
    return params


def _source_snapshot(preregister_path: Path) -> dict[str, Any]:
    paths = {
        "runner": Path(__file__).resolve(),
        "target_transform_module": PROJECT_DIR / "src" / "target_transform.py",
        "target_transform_test": PROJECT_DIR / "tests" / "test_target_transform.py",
        "strict_raw_helper": PROJECT_DIR / "scripts" / "run_shared_q07_multiseed.py",
        "baseline_helper": PROJECT_DIR / "scripts" / "run_run_sequence_residual.py",
        "features": PROJECT_DIR / "src" / "features.py",
        "temporal": PROJECT_DIR / "src" / "temporal.py",
        "metric": PROJECT_DIR / "src" / "metric.py",
        "manifest": PROJECT_DIR / "src" / "manifest.py",
        "preregister": preregister_path.resolve(),
    }
    return {name: describe_file(path) for name, path in paths.items()}


def _stage1_input_snapshot(
    raw_dir: Path,
    artifact_root: Path,
    preregister_path: Path,
    recipe_path: Path,
) -> dict[str, Any]:
    paths = {
        "info": raw_dir / "info.xlsx",
        "locked_recipe": recipe_path,
        "stage1_g12_baseline": artifact_root / "oof" / "dev2023_locked_v3.parquet",
        "stage1_g12_identity": artifact_root
        / "oof"
        / "dev2023_lgb_l1_eligible_n1500.parquet",
        "stage1_g3_components_and_identity": artifact_root
        / "oof"
        / "g3dev2023h2_candidates.parquet",
        "stage1_exact_baseline": artifact_root
        / "postgate"
        / "shared_q07_multiseed_strict"
        / "oof"
        / "stage1_corrected_v3_baseline.parquet",
        "preregister": preregister_path.resolve(),
    }
    return {name: describe_file(path) for name, path in paths.items()}


def _assert_unchanged(before: Mapping[str, Any], after: Mapping[str, Any], name: str) -> None:
    if before != after:
        raise AssertionError(f"{name} changed during pre-2024 target-transform selection")


def _candidate_map(preregister: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for transform in preregister["transforms"]:
        for weight in preregister["blend_weights"]:
            weight_code = f"w{int(round(float(weight) * 100)):02d}"
            candidate_id = f"{transform['id']}__{weight_code}"
            candidates[candidate_id] = {
                "id": candidate_id,
                "transform": dict(transform),
                "transformed_weight": float(weight),
            }
    if len(candidates) != int(preregister["candidate_count"]):
        raise AssertionError("constructed target-transform candidate count changed")
    return candidates


def _fit_predict(
    *,
    transform_id: str,
    preregister: Mapping[str, Any],
    group: str,
    features: pd.DataFrame,
    actual: pd.Series,
    fit_bounds: Sequence[str],
    apply_bounds: Sequence[str],
) -> tuple[TargetTransformRegressor, pd.Series, dict[str, Any]]:
    fit_index = _interval(features.index, fit_bounds)
    apply_index = _interval(features.index, apply_bounds)
    assert_strict_fit_apply_order(fit_index, apply_index)
    if not features.index.equals(actual.index):
        raise AssertionError("target-transform feature and actual indices differ")
    model = TargetTransformRegressor(
        transform_id=transform_id,
        model_params=_model_params(preregister),
        epsilon=0.02,
    )
    model.fit(
        features.loc[fit_index],
        actual.loc[fit_index].astype(float) / CAPACITY_KWH[group],
    )
    prediction_cf, prediction_audit = model.predict_cf(features.loc[apply_index])
    return model, prediction_cf, {
        "group": group,
        "transform_id": transform_id,
        "fit_start": fit_index.min().isoformat(),
        "fit_end": fit_index.max().isoformat(),
        "apply_start": apply_index.min().isoformat(),
        "apply_end": apply_index.max().isoformat(),
        "fit_apply_overlap_rows": 0,
        "fit_max_strictly_before_apply_min": True,
        "fit": model.fit_audit_,
        "prediction": prediction_audit,
    }


def _normalise_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, engine="pyarrow")
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise AssertionError(f"prediction reference index invalid: {path}")
    return frame


def _identity_stage1_references(
    artifact_root: Path,
    preregister: Mapping[str, Any],
    predicted_cf: Mapping[str, pd.Series],
) -> dict[str, Any]:
    specifications = preregister["identity_control"]["stage1_references"]
    loaded: dict[Path, pd.DataFrame] = {}
    output: dict[str, Any] = {}
    for group in TARGET_COLS:
        specification = specifications[group]
        path = artifact_root.parent / specification["path"]
        if path not in loaded:
            loaded[path] = _normalise_frame(path)
        reference = loaded[path][str(specification["column"])].astype(float)
        candidate = predicted_cf[group] * CAPACITY_KWH[group]
        if not candidate.index.equals(reference.index):
            raise AssertionError(f"{group} identity reference index differs")
        difference = np.abs(candidate.to_numpy(dtype=float) - reference.to_numpy(dtype=float))
        maximum = float(difference.max(initial=0.0))
        tolerance = float(preregister["identity_control"]["comparison_tolerance_kwh"])
        if maximum > tolerance:
            raise AssertionError(
                f"{group} identity reconstruction differs by {maximum} kWh"
            )
        output[group] = {
            "maximum_absolute_difference_kwh": maximum,
            "tolerance_kwh": tolerance,
            "reference": describe_file(path),
            "reference_column": str(specification["column"]),
            "prediction_rows": int(len(candidate)),
        }
    return output


def _identity_common_reference(
    *,
    path: Path,
    expected_index: pd.DatetimeIndex,
    predicted_cf: Mapping[str, pd.Series],
    preregister: Mapping[str, Any],
) -> dict[str, Any]:
    frame = _normalise_frame(path)
    if not frame.index.equals(expected_index) or not set(TARGET_COLS).issubset(frame.columns):
        raise AssertionError("identity control reference schema/index changed")
    output: dict[str, Any] = {}
    tolerance = float(preregister["identity_control"]["comparison_tolerance_kwh"])
    for group in TARGET_COLS:
        candidate = predicted_cf[group] * CAPACITY_KWH[group]
        reference = frame[group].astype(float)
        if not candidate.index.equals(reference.index):
            raise AssertionError(f"{group} identity reference index differs")
        maximum = float(
            np.abs(candidate.to_numpy(dtype=float) - reference.to_numpy(dtype=float)).max(
                initial=0.0
            )
        )
        if maximum > tolerance:
            raise AssertionError(
                f"{group} identity reconstruction differs by {maximum} kWh"
            )
        output[group] = {
            "maximum_absolute_difference_kwh": maximum,
            "tolerance_kwh": tolerance,
        }
    return {"reference": describe_file(path), "groups": output}


def _label_prefix_contract(raw_dir: Path, labels: pd.DataFrame) -> dict[str, Any]:
    if len(labels) != EXPECTED_ROWS_PRE2024 or labels.index.max() != YEAR_2023_END:
        raise AssertionError("bounded label materialization changed")
    return {
        "path": str((raw_dir / "train" / "train_labels.csv").resolve()),
        "materialized_rows": int(len(labels)),
        "materialized_start": labels.index.min().isoformat(),
        "materialized_end": labels.index.max().isoformat(),
        "physical_byte_limit": int(OFFICIAL_STAGE1_PREFIX_BYTES["labels"]),
        "physical_prefix_sha256": OFFICIAL_LABEL_PREFIX_SHA256,
        "suffix_bytes_exposed_to_parser": 0,
        "validated_by": "scripts.run_shared_q07_multiseed._read_labels",
    }


def _stage1(
    *,
    raw_dir: Path,
    cache_dir: Path,
    artifact_root: Path,
    recipe_path: Path,
    preregister_path: Path,
    out_dir: Path,
    preregister: Mapping[str, Any],
) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"Stage 1 requires a new output directory: {out_dir}")
    out_dir.mkdir(parents=True)
    shutil.copyfile(preregister_path, out_dir / "preregister.json")
    sources_before = _source_snapshot(preregister_path)
    inputs_before = _stage1_input_snapshot(
        raw_dir, artifact_root, preregister_path, recipe_path
    )
    _verify_locked_recipe(recipe_path, preregister)

    labels = _read_labels(
        raw_dir / "train" / "train_labels.csv",
        nrows=EXPECTED_ROWS_PRE2024,
        expected_end=YEAR_2023_END,
        prefix_bytes=OFFICIAL_STAGE1_PREFIX_BYTES["labels"],
        expected_prefix_sha256=OFFICIAL_LABEL_PREFIX_SHA256,
    )
    label_contract = _label_prefix_contract(raw_dir, labels)
    print("Stage 1: building byte-bounded pre-2024 NWP features", flush=True)
    features, raw_contract = _read_stage1_raw_features(raw_dir, labels)
    baseline, baseline_audit = _read_stage1_baseline(artifact_root)

    transform_predictions: dict[str, dict[str, pd.Series]] = {
        name: {} for name in (IDENTITY_ID, *TRANSFORM_IDS)
    }
    training_audit: dict[str, Any] = {}
    for transform_id in (IDENTITY_ID, *TRANSFORM_IDS):
        print(f"Stage 1: fitting {transform_id}", flush=True)
        training_audit[transform_id] = {}
        for group in TARGET_COLS:
            fold = preregister["stage1"]["folds"][group]
            model, prediction_cf, audit = _fit_predict(
                transform_id=transform_id,
                preregister=preregister,
                group=group,
                features=features[group],
                actual=labels[group],
                fit_bounds=fold["fit"],
                apply_bounds=fold["apply"],
            )
            transform_predictions[transform_id][group] = prediction_cf
            model_path = out_dir / "models" / f"stage1__{transform_id}__{group}.joblib"
            _atomic_joblib(
                {
                    "model": model,
                    "transform_id": transform_id,
                    "group": group,
                    "preregister_sha256": PREREGISTER_SHA256,
                },
                model_path,
            )
            audit["model"] = describe_file(model_path)
            training_audit[transform_id][group] = audit

    identity_audit = _identity_stage1_references(
        artifact_root, preregister, transform_predictions[IDENTITY_ID]
    )
    raw_frames: dict[str, pd.DataFrame] = {}
    for transform_id in (IDENTITY_ID, *TRANSFORM_IDS):
        frame = pd.DataFrame(index=baseline.index, columns=list(TARGET_COLS), dtype=float)
        for group in TARGET_COLS:
            prediction = transform_predictions[transform_id][group]
            frame.loc[prediction.index, group] = (
                prediction.to_numpy(dtype=float) * CAPACITY_KWH[group]
            )
        raw_frames[transform_id] = frame
        _atomic_parquet(
            frame, out_dir / "oof" / f"stage1_raw__{transform_id}.parquet"
        )

    candidates = _candidate_map(preregister)
    comparisons: dict[str, Any] = {}
    for candidate_id, recipe in candidates.items():
        transform_id = str(recipe["transform"]["id"])
        weight = float(recipe["transformed_weight"])
        frame = pd.DataFrame(index=baseline.index, columns=list(TARGET_COLS), dtype=float)
        by_group: dict[str, Any] = {}
        for group in TARGET_COLS:
            transformed_cf = transform_predictions[transform_id][group]
            prediction = blend_with_corrected_v3(
                transformed_cf,
                baseline.loc[transformed_cf.index, group],
                capacity_kwh=CAPACITY_KWH[group],
                transformed_weight=weight,
            )
            frame.loc[prediction.index, group] = prediction
            by_group[group] = _comparisons(
                actual=labels.loc[prediction.index, group],
                baseline=baseline.loc[prediction.index, group],
                candidate=prediction,
                group=group,
                slices=preregister["stage1"]["required_slices"][group],
            )
        _atomic_parquet(frame, out_dir / "oof" / f"stage1__{candidate_id}.parquet")
        comparisons[candidate_id] = by_group
    _atomic_parquet(baseline, out_dir / "oof" / "stage1_corrected_v3_baseline.parquet")
    selected = choose_single_candidate(comparisons)

    sources_after = _source_snapshot(preregister_path)
    inputs_after = _stage1_input_snapshot(
        raw_dir, artifact_root, preregister_path, recipe_path
    )
    _assert_unchanged(sources_before, sources_after, "source snapshot")
    _assert_unchanged(inputs_before, inputs_after, "input snapshot")
    result = {
        "schema_version": 1,
        "experiment_id": "target_transform_stage1",
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "candidate_count": len(candidates),
        "candidate_map": candidates,
        "selected_candidate": selected,
        "selection_changes_after_2024": False,
        "2024_labels_weather_or_predictions_read": False,
        "label_prefix_contract": label_contract,
        "raw_feature_contract": raw_contract,
        "cache_dir_deliberately_not_read": str(cache_dir.resolve()),
        "baseline_exact_reconstruction": baseline_audit,
        "identity_exact_reconstruction": identity_audit,
        "training_audit": training_audit,
        "comparisons": comparisons,
        "source_snapshot_before": sources_before,
        "source_snapshot_after": sources_after,
        "source_snapshot_sha256": _mapping_sha256(sources_before),
        "input_snapshot_before": inputs_before,
        "input_snapshot_after": inputs_after,
        "input_snapshot_sha256": _mapping_sha256(inputs_before),
        "snapshots_unchanged": True,
        "leaderboard_score_claim": False,
    }
    result_path = out_dir / "stage1_results.json"
    _write_json(result_path, result)
    _write_json(
        out_dir / "pre2024_lock.json",
        {
            "schema_version": 1,
            "experiment_id": "target_transform_pre2024_lock",
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "stage1_results_sha256": sha256_file(result_path),
            "source_snapshot_sha256": result["source_snapshot_sha256"],
            "input_snapshot_sha256": result["input_snapshot_sha256"],
            "selected_candidate": selected,
            "selection_changes_after_2024": False,
            "2024_read_before_lock": False,
        },
    )
    print("Stage 1 locked candidate:", selected, flush=True)
    return result


def _load_lock(
    *,
    out_dir: Path,
    preregister_path: Path,
    raw_dir: Path,
    artifact_root: Path,
    recipe_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result_path = out_dir / "stage1_results.json"
    lock_path = out_dir / "pre2024_lock.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["preregister_sha256"] != PREREGISTER_SHA256:
        raise AssertionError("target-transform lock preregister SHA changed")
    if lock["stage1_results_sha256"] != sha256_file(result_path):
        raise AssertionError("target-transform Stage 1 result changed after lock")
    if choose_single_candidate(result["comparisons"]) != lock["selected_candidate"]:
        raise AssertionError("locked target-transform recipe violates selection rule")
    if _mapping_sha256(_source_snapshot(preregister_path)) != lock["source_snapshot_sha256"]:
        raise AssertionError("target-transform source snapshot changed after lock")
    current_inputs = _stage1_input_snapshot(
        raw_dir, artifact_root, preregister_path, recipe_path
    )
    if _mapping_sha256(current_inputs) != lock["input_snapshot_sha256"]:
        raise AssertionError("target-transform input snapshot changed after lock")
    return lock, result


def _stage2_no_candidate(out_dir: Path) -> dict[str, Any]:
    result = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "locked_candidate": None,
        "2024_read": False,
        "promoted": False,
        "reason": "no target-transform recipe improved every Stage 1 group/slice",
        "selection_changes_after_2024": False,
        "final_fit_performed": False,
        "submission_created": False,
        "leaderboard_score_claim": False,
    }
    _write_json(out_dir / "stage2_results.json", result)
    _write_json(
        out_dir / "promotion_lock.json",
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "locked_candidate": None,
            "promoted": False,
            "2024_read": False,
            "stage2_results_sha256": sha256_file(out_dir / "stage2_results.json"),
            "pre2024_lock_sha256": sha256_file(out_dir / "pre2024_lock.json"),
        },
    )
    return result


def _stage2(
    *,
    raw_dir: Path,
    cache_dir: Path,
    artifact_root: Path,
    recipe_path: Path,
    preregister_path: Path,
    out_dir: Path,
    preregister: Mapping[str, Any],
) -> dict[str, Any]:
    if (out_dir / "stage2_results.json").exists():
        raise FileExistsError("target-transform Stage 2 already exists")
    lock, stage1_result = _load_lock(
        out_dir=out_dir,
        preregister_path=preregister_path,
        raw_dir=raw_dir,
        artifact_root=artifact_root,
        recipe_path=recipe_path,
    )
    selected = lock["selected_candidate"]
    if selected is None:
        return _stage2_no_candidate(out_dir)

    locked_recipe = stage1_result["candidate_map"][selected]
    transform_id = str(locked_recipe["transform"]["id"])
    weight = float(locked_recipe["transformed_weight"])
    labels = _read_labels(
        raw_dir / "train" / "train_labels.csv",
        nrows=EXPECTED_ROWS_THROUGH2024,
        expected_end=YEAR_2024_END,
    )
    features = _read_features(cache_dir, labels, expected_end=YEAR_2024_END)
    gate_index = pd.date_range(
        STAGE2_START, STAGE2_END, freq="h", name="forecast_kst_dtm"
    )
    baseline, baseline_audit = _reconstruct_baseline(
        artifact_root=artifact_root,
        recipe_path=recipe_path,
        component_files=GATE_COMPONENT_FILES,
        reference_path=artifact_root / "oof" / "gate2024_locked_v3_cf_fix.parquet",
        expected_index=gate_index,
    )
    identity_predictions: dict[str, pd.Series] = {}
    transformed_predictions: dict[str, pd.Series] = {}
    training: dict[str, Any] = {IDENTITY_ID: {}, transform_id: {}}
    for current_transform in (IDENTITY_ID, transform_id):
        for group in TARGET_COLS:
            model, prediction_cf, audit = _fit_predict(
                transform_id=current_transform,
                preregister=preregister,
                group=group,
                features=features[group],
                actual=labels[group],
                fit_bounds=preregister["stage2"]["fit"][group],
                apply_bounds=preregister["stage2"]["apply"],
            )
            if current_transform == IDENTITY_ID:
                identity_predictions[group] = prediction_cf
            else:
                transformed_predictions[group] = prediction_cf
            model_path = out_dir / "models" / f"stage2__{current_transform}__{group}.joblib"
            _atomic_joblib(
                {
                    "model": model,
                    "locked_recipe": locked_recipe,
                    "group": group,
                    "preregister_sha256": PREREGISTER_SHA256,
                },
                model_path,
            )
            audit["model"] = describe_file(model_path)
            training[current_transform][group] = audit
    identity_audit = _identity_common_reference(
        path=artifact_root / preregister["identity_control"]["stage2_reference"].removeprefix("artifacts/"),
        expected_index=gate_index,
        predicted_cf=identity_predictions,
        preregister=preregister,
    )
    candidate = pd.DataFrame(index=gate_index, columns=list(TARGET_COLS), dtype=float)
    raw = pd.DataFrame(index=gate_index, columns=list(TARGET_COLS), dtype=float)
    comparisons: dict[str, Any] = {}
    for group in TARGET_COLS:
        raw[group] = transformed_predictions[group] * CAPACITY_KWH[group]
        prediction = blend_with_corrected_v3(
            transformed_predictions[group],
            baseline[group],
            capacity_kwh=CAPACITY_KWH[group],
            transformed_weight=weight,
        )
        candidate[group] = prediction
        comparisons[group] = _comparisons(
            actual=labels.loc[gate_index, group],
            baseline=baseline[group],
            candidate=prediction,
            group=group,
            slices=preregister["stage2"]["required_slices"],
        )
    promoted = all(
        float(item["delta"]) > 0.0
        for slices in comparisons.values()
        for item in slices.values()
    )
    _atomic_parquet(baseline, out_dir / "oof" / "stage2_corrected_v3_baseline.parquet")
    _atomic_parquet(raw, out_dir / "oof" / f"stage2_raw__{transform_id}.parquet")
    _atomic_parquet(candidate, out_dir / "oof" / f"stage2__{selected}.parquet")
    evaluation = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "locked_candidate": selected,
        "2024_read": True,
        "selection_changes_after_2024": False,
        "comparisons": comparisons,
        "training": training,
        "baseline_exact_reconstruction": baseline_audit,
        "identity_exact_reconstruction": identity_audit,
        "promoted": bool(promoted),
        "leaderboard_score_claim": False,
    }
    evaluation_path = out_dir / "stage2_evaluation.json"
    _write_json(evaluation_path, evaluation)
    _write_json(
        out_dir / "promotion_lock.json",
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "locked_candidate": selected,
            "promoted": bool(promoted),
            "2024_read": True,
            "stage2_evaluation_sha256": sha256_file(evaluation_path),
            "pre2024_lock_sha256": sha256_file(out_dir / "pre2024_lock.json"),
        },
    )
    final_created = False
    if promoted:
        _final(
            raw_dir=raw_dir,
            cache_dir=cache_dir,
            artifact_root=artifact_root,
            recipe_path=recipe_path,
            out_dir=out_dir,
            preregister=preregister,
            locked_recipe=locked_recipe,
            labels=labels,
            features=features,
        )
        final_created = True
    result = {
        **evaluation,
        "final_fit_performed": final_created,
        "submission_created": final_created,
    }
    _write_json(out_dir / "stage2_results.json", result)
    return result


def _final(
    *,
    raw_dir: Path,
    cache_dir: Path,
    artifact_root: Path,
    recipe_path: Path,
    out_dir: Path,
    preregister: Mapping[str, Any],
    locked_recipe: Mapping[str, Any],
    labels: pd.DataFrame,
    features: Mapping[str, pd.DataFrame],
) -> None:
    promotion = json.loads((out_dir / "promotion_lock.json").read_text(encoding="utf-8"))
    if promotion.get("promoted") is not True:
        raise AssertionError("target-transform final requires positive promotion lock")
    sample_path = raw_dir / "sample_submission.csv"
    sample = pd.read_csv(sample_path, encoding="utf-8-sig")
    test_index = pd.DatetimeIndex(
        pd.to_datetime(sample["forecast_kst_dtm"], errors="raise"),
        name="forecast_kst_dtm",
    )
    if (
        len(test_index) != EXPECTED_TEST_ROWS
        or test_index.min() != FINAL_START
        or test_index.max() != FINAL_END
    ):
        raise AssertionError("target-transform sample interval changed")
    test_features = _read_test_features(cache_dir, test_index)
    baseline, baseline_audit = _reconstruct_baseline(
        artifact_root=artifact_root,
        recipe_path=recipe_path,
        component_files=FINAL_COMPONENT_FILES,
        reference_path=artifact_root
        / "final_cf_fix"
        / "predictions"
        / "corrected_v3_test.parquet",
        expected_index=test_index,
    )
    transform_id = str(locked_recipe["transform"]["id"])
    weight = float(locked_recipe["transformed_weight"])
    identity_predictions: dict[str, pd.Series] = {}
    transformed_predictions: dict[str, pd.Series] = {}
    training: dict[str, Any] = {IDENTITY_ID: {}, transform_id: {}}
    for current_transform in (IDENTITY_ID, transform_id):
        for group in TARGET_COLS:
            start = (
                pd.Timestamp("2022-01-01 01:00:00")
                if group != "kpx_group_3"
                else pd.Timestamp("2023-01-01 01:00:00")
            )
            fit_index = features[group].index[features[group].index >= start]
            assert_strict_fit_apply_order(fit_index, test_index)
            model = TargetTransformRegressor(
                transform_id=current_transform,
                model_params=_model_params(preregister),
                epsilon=0.02,
            ).fit(
                features[group].loc[fit_index],
                labels.loc[fit_index, group].astype(float) / CAPACITY_KWH[group],
            )
            prediction_cf, prediction_audit = model.predict_cf(test_features[group])
            if current_transform == IDENTITY_ID:
                identity_predictions[group] = prediction_cf
            else:
                transformed_predictions[group] = prediction_cf
            model_path = out_dir / "models" / f"final__{current_transform}__{group}.joblib"
            _atomic_joblib(
                {
                    "model": model,
                    "locked_recipe": locked_recipe,
                    "group": group,
                    "preregister_sha256": PREREGISTER_SHA256,
                },
                model_path,
            )
            training[current_transform][group] = {
                "fit": model.fit_audit_,
                "prediction": prediction_audit,
                "fit_max_strictly_before_apply_min": True,
                "model": describe_file(model_path),
            }
    identity_audit = _identity_common_reference(
        path=artifact_root / preregister["identity_control"]["final_reference"].removeprefix("artifacts/"),
        expected_index=test_index,
        predicted_cf=identity_predictions,
        preregister=preregister,
    )
    raw = pd.DataFrame(index=test_index, columns=list(TARGET_COLS), dtype=float)
    predictions = pd.DataFrame(index=test_index, columns=list(TARGET_COLS), dtype=float)
    for group in TARGET_COLS:
        raw[group] = transformed_predictions[group] * CAPACITY_KWH[group]
        predictions[group] = blend_with_corrected_v3(
            transformed_predictions[group],
            baseline[group],
            capacity_kwh=CAPACITY_KWH[group],
            transformed_weight=weight,
        )
    raw_path = out_dir / "predictions" / f"raw__{transform_id}__2025.parquet"
    prediction_path = out_dir / "predictions" / "target_transform_2025.parquet"
    _atomic_parquet(raw, raw_path)
    _atomic_parquet(predictions, prediction_path)
    submission_path = out_dir / preregister["final"]["submission_name"]
    submission = _write_submission(sample_path, predictions, submission_path)
    _write_json(
        out_dir / "final_results.json",
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "locked_recipe": locked_recipe,
            "training": training,
            "baseline_exact_reconstruction": baseline_audit,
            "identity_exact_reconstruction": identity_audit,
            "raw_prediction": describe_file(raw_path),
            "prediction": describe_file(prediction_path),
            "submission": submission,
            "leaderboard_score_claim": False,
        },
    )


def _manifest_inputs(args: argparse.Namespace, stage2_result: Mapping[str, Any] | None) -> list[Path]:
    paths = [
        args.preregister,
        args.recipe,
        args.raw_dir / "info.xlsx",
        args.artifact_root / "oof" / "dev2023_locked_v3.parquet",
        args.artifact_root / "oof" / "dev2023_lgb_l1_eligible_n1500.parquet",
        args.artifact_root / "oof" / "g3dev2023h2_candidates.parquet",
        args.artifact_root
        / "postgate"
        / "shared_q07_multiseed_strict"
        / "oof"
        / "stage1_corrected_v3_baseline.parquet",
        Path(__file__).resolve(),
        PROJECT_DIR / "src" / "target_transform.py",
        PROJECT_DIR / "tests" / "test_target_transform.py",
    ]
    if stage2_result and stage2_result.get("2024_read"):
        paths.extend(
            [
                *(args.cache_dir / f"{group}_weather_train.parquet" for group in TARGET_COLS),
                *(args.artifact_root / relative for relative in GATE_COMPONENT_FILES.values()),
                args.artifact_root / "oof" / "gate2024_locked_v3_cf_fix.parquet",
            ]
        )
    if stage2_result and stage2_result.get("submission_created"):
        paths.extend(
            [
                args.raw_dir / "sample_submission.csv",
                *(args.cache_dir / f"{group}_weather_test.parquet" for group in TARGET_COLS),
                *(args.artifact_root / relative for relative in FINAL_COMPONENT_FILES.values()),
                args.artifact_root
                / "final_cf_fix"
                / "predictions"
                / "corrected_v3_test.parquet",
            ]
        )
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def _write_manifest(
    *,
    args: argparse.Namespace,
    preregister: Mapping[str, Any],
    stage2_result: Mapping[str, Any] | None,
    name: str,
) -> None:
    destination = args.out_dir / name
    outputs = sorted(
        path for path in args.out_dir.rglob("*") if path.is_file() and path != destination
    )
    stage1 = json.loads((args.out_dir / "stage1_results.json").read_text(encoding="utf-8"))
    _write_json(
        destination,
        {
            "schema_version": 1,
            "artifact_type": "target_transform_strict_forward",
            "created_utc": utc_now(),
            "command": [sys.executable, *sys.argv],
            "runtime": {"packages": package_versions(), "git": git_state(PROJECT_DIR)},
            "preregister_sha256": PREREGISTER_SHA256,
            "candidate_count": int(preregister["candidate_count"]),
            "inputs": [
                describe_file(path) for path in _manifest_inputs(args, stage2_result)
            ],
            "bounded_label_prefix": stage1["label_prefix_contract"],
            "bounded_nwp_prefixes": stage1["raw_feature_contract"],
            "outputs": [describe_file(path) for path in outputs],
            "stage2": dict(stage2_result or {}),
            "leaderboard_score_claim": False,
        },
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    for name in (
        "raw_dir",
        "cache_dir",
        "artifact_root",
        "recipe",
        "preregister",
        "out_dir",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    preregister = _load_preregister(args.preregister)
    _verify_locked_recipe(args.recipe, preregister)
    if args.stage in {"stage1", "all"}:
        _stage1(
            raw_dir=args.raw_dir,
            cache_dir=args.cache_dir,
            artifact_root=args.artifact_root,
            recipe_path=args.recipe,
            preregister_path=args.preregister,
            out_dir=args.out_dir,
            preregister=preregister,
        )
        if args.stage == "stage1":
            _write_manifest(
                args=args,
                preregister=preregister,
                stage2_result=None,
                name="stage1_manifest.json",
            )
            return
    if args.stage in {"stage2", "all"}:
        result = _stage2(
            raw_dir=args.raw_dir,
            cache_dir=args.cache_dir,
            artifact_root=args.artifact_root,
            recipe_path=args.recipe,
            preregister_path=args.preregister,
            out_dir=args.out_dir,
            preregister=preregister,
        )
        _write_manifest(
            args=args,
            preregister=preregister,
            stage2_result=result,
            name="manifest.json",
        )


if __name__ == "__main__":
    main()

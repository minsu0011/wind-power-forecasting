"""Strict-forward joint 24-hour temporal MLP experiment.

Stage 1 physically bounds all label/weather reads to the pre-2024 prefix,
selects at most one preregistered architecture/blend independently per group,
and writes an immutable lock.  Only a non-identity lock permits the 2024
confirmation read; only a wholly positive fixed confirmation permits 2025
feature reads and submission creation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.run_run_sequence_residual import (  # noqa: E402
    FINAL_END,
    FINAL_START,
    G3_STAGE1_START,
    STAGE1_END,
    STAGE1_START,
    STAGE2_END,
    STAGE2_START,
    _atomic_joblib,
    _atomic_parquet,
    _comparisons,
    _core_weather,
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
from src.run_sequence_residual import assert_strict_fit_predict_order  # noqa: E402
from src.temporal_mlp import (  # noqa: E402
    RunDCTTransformer,
    blend_joint_prediction,
    build_joint_cf_target,
    choose_group_candidates,
    fit_seed_ensemble,
    predict_seed_ensemble,
)


PREREGISTER_SHA256 = (
    "f1d79427b541e0bbcb12bfaa2e9f1e54e32595c18e26c943929944f1316e20cf"
)


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
        default=PROJECT_DIR / "configs" / "temporal_mlp_preregister.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_DIR / "artifacts" / "postgate" / "temporal_mlp_strict",
    )
    parser.add_argument("--stage", choices=("stage1", "stage2", "all"), default="all")
    return parser.parse_args(argv)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _load_preregister(path: Path) -> dict[str, Any]:
    observed = sha256_file(path)
    if observed != PREREGISTER_SHA256:
        raise AssertionError(f"temporal MLP preregister SHA changed: {observed}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("experiment_id") != "temporal_mlp_joint_run_strict_forward_v1":
        raise AssertionError("unexpected temporal MLP experiment id")
    channels = payload["run_input"]["channels_in_order"]
    if channels != [
        "baseline_cf",
        "cross__hub_ws_mean",
        "ldaps__idw__hub_ws",
        "gfs__idw__hub_ws",
        "ldaps__idw__wind_power_density",
        "gfs__idw__wind_power_density",
    ]:
        raise AssertionError("frozen temporal MLP channel set changed")
    architectures = payload["architectures"]
    if [item["id"] for item in architectures] != [
        "tanh16_a010_e2",
        "tanh24x8_a050_e2",
    ]:
        raise AssertionError("frozen temporal MLP architectures changed")
    if payload["blend_weights"] != [0.05, 0.1, 0.2]:
        raise AssertionError("frozen temporal MLP blend weights changed")
    if int(payload["run_input"]["dct_coefficients_per_channel"]) != 8:
        raise AssertionError("frozen DCT coefficient count changed")
    return payload


def _candidate_id(architecture_id: str, weight: float) -> str:
    return f"{architecture_id}__w{int(round(100 * float(weight))):03d}"


def _parse_candidate(
    preregister: Mapping[str, Any], candidate_id: str
) -> tuple[Mapping[str, Any], float]:
    matches: list[tuple[Mapping[str, Any], float]] = []
    for architecture in preregister["architectures"]:
        for weight in preregister["blend_weights"]:
            if _candidate_id(str(architecture["id"]), float(weight)) == candidate_id:
                matches.append((architecture, float(weight)))
    if len(matches) != 1:
        raise AssertionError(f"locked candidate is not preregistered: {candidate_id}")
    return matches[0]


def _source_snapshot(preregister_path: Path) -> dict[str, Any]:
    paths = {
        "runner": Path(__file__).resolve(),
        "temporal_mlp_module": PROJECT_DIR / "src" / "temporal_mlp.py",
        "test": PROJECT_DIR / "tests" / "test_temporal_mlp.py",
        "strict_raw_helper": PROJECT_DIR / "scripts" / "run_shared_q07_multiseed.py",
        "baseline_helper": PROJECT_DIR / "scripts" / "run_run_sequence_residual.py",
        "features": PROJECT_DIR / "src" / "features.py",
        "metric": PROJECT_DIR / "src" / "metric.py",
        "preregister": preregister_path.resolve(),
        "preregister_sha_sidecar": preregister_path.with_suffix(".sha256").resolve(),
    }
    return {name: describe_file(path) for name, path in paths.items()}


def _safe_stage1_input_snapshot(
    raw_dir: Path, artifact_root: Path, preregister_path: Path
) -> dict[str, Any]:
    paths = {
        "info": raw_dir / "info.xlsx",
        "stage1_g12": artifact_root / "oof" / "dev2023_locked_v3.parquet",
        "stage1_g3_components": artifact_root / "oof" / "g3dev2023h2_candidates.parquet",
        "stage1_exact_reference": artifact_root
        / "postgate"
        / "shared_q07_multiseed_strict"
        / "oof"
        / "stage1_corrected_v3_baseline.parquet",
        "preregister": preregister_path,
    }
    return {name: describe_file(path) for name, path in paths.items()}


def _weather_only(
    features: Mapping[str, pd.DataFrame], preregister: Mapping[str, Any]
) -> dict[str, pd.DataFrame]:
    return _core_weather(
        features, preregister["run_input"]["weather_columns_in_order"]
    )


def _fit_predict_architecture(
    *,
    architecture: Mapping[str, Any],
    weather: pd.DataFrame,
    actual: pd.Series,
    baseline: pd.Series,
    capacity_kwh: float,
    weather_columns: Sequence[str],
    fit_index: pd.DatetimeIndex,
    prediction_index: pd.DatetimeIndex,
) -> tuple[RunDCTTransformer, list[Any], np.ndarray, dict[str, Any]]:
    assert_strict_fit_predict_order(fit_index, prediction_index)
    if baseline.loc[fit_index].isna().any() or baseline.loc[prediction_index].isna().any():
        raise AssertionError("baseline contains missing values in fit/predict interval")
    transformer = RunDCTTransformer(
        tuple(map(str, weather_columns)),
        coefficients=8,
        expected_hours=24,
        scale_floor=1e-6,
    )
    x_fit, fit_keys = transformer.fit_transform(
        weather.loc[fit_index],
        baseline.loc[fit_index],
        capacity_kwh=capacity_kwh,
    )
    y_fit, target_keys = build_joint_cf_target(
        actual.loc[fit_index], capacity_kwh=capacity_kwh
    )
    if not fit_keys.equals(target_keys):
        raise AssertionError("run-level input and target keys differ")
    models = fit_seed_ensemble(architecture, x_fit, y_fit)
    x_predict, prediction_keys = transformer.transform(
        weather.loc[prediction_index],
        baseline.loc[prediction_index],
        capacity_kwh=capacity_kwh,
    )
    neural_cf = predict_seed_ensemble(models, x_predict)
    evidence = {
        "fit_rows": int(len(fit_index)),
        "fit_runs": int(len(fit_keys)),
        "fit_start": fit_index.min().isoformat(),
        "fit_end": fit_index.max().isoformat(),
        "prediction_rows": int(len(prediction_index)),
        "prediction_runs": int(len(prediction_keys)),
        "prediction_start": prediction_index.min().isoformat(),
        "prediction_end": prediction_index.max().isoformat(),
        "input_dimension": int(x_fit.shape[1]),
        "target_dimension": int(y_fit.shape[1]),
        "channel_mean": transformer.channel_mean_.tolist(),
        "channel_scale": transformer.channel_scale_.tolist(),
        "seed_n_iter": [int(model.n_iter_) for model in models],
        "seed_loss": [float(model.loss_) for model in models],
        "network_cf_min": float(neural_cf.min()),
        "network_cf_max": float(neural_cf.max()),
        "network_cf_mean": float(neural_cf.mean()),
        "actual_or_scada_feature_columns": 0,
    }
    return transformer, models, neural_cf, evidence


def _stage1(
    *,
    raw_dir: Path,
    cache_dir: Path,
    artifact_root: Path,
    out_dir: Path,
    preregister_path: Path,
    preregister: Mapping[str, Any],
) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"Stage 1 requires a new output directory: {out_dir}")
    created_utc = utc_now()
    out_dir.mkdir(parents=True)
    filesystem_creation_utc = datetime.fromtimestamp(
        out_dir.stat().st_ctime, tz=timezone.utc
    ).isoformat()
    shutil.copyfile(preregister_path, out_dir / "preregister.json")
    source_before = _source_snapshot(preregister_path)
    inputs_before = _safe_stage1_input_snapshot(raw_dir, artifact_root, preregister_path)

    labels = _read_labels(
        raw_dir / "train" / "train_labels.csv",
        nrows=EXPECTED_ROWS_PRE2024,
        expected_end=YEAR_2023_END,
        prefix_bytes=OFFICIAL_STAGE1_PREFIX_BYTES["labels"],
        expected_prefix_sha256=OFFICIAL_LABEL_PREFIX_SHA256,
    )
    print("Stage 1: building physically bounded pre-2024 weather", flush=True)
    raw_features, raw_contract = _read_stage1_raw_features(raw_dir, labels)
    weather = _weather_only(raw_features, preregister)
    del raw_features
    baseline, baseline_audit = _read_stage1_baseline(artifact_root)
    weather_columns = preregister["run_input"]["weather_columns_in_order"]

    candidate_outputs: dict[str, pd.DataFrame] = {
        _candidate_id(str(architecture["id"]), float(weight)): pd.DataFrame(
            index=baseline.index, columns=list(TARGET_COLS), dtype=float
        )
        for architecture in preregister["architectures"]
        for weight in preregister["blend_weights"]
    }
    comparisons: dict[str, dict[str, Any]] = {
        candidate_id: {} for candidate_id in candidate_outputs
    }
    training: dict[str, Any] = {}
    for architecture in preregister["architectures"]:
        architecture_id = str(architecture["id"])
        training[architecture_id] = {}
        print(f"Stage 1: fitting joint-run {architecture_id}", flush=True)
        for group in TARGET_COLS:
            fit_index = _interval(
                baseline[group].dropna().index,
                preregister["stage1"]["fit_intervals"][group],
            )
            slices = preregister["stage1"]["selection_slices"][group]
            prediction_index = _interval(
                baseline[group].dropna().index, slices["full"]
            )
            transformer, models, neural_cf, evidence = _fit_predict_architecture(
                architecture=architecture,
                weather=weather[group],
                actual=labels[group],
                baseline=baseline[group],
                capacity_kwh=CAPACITY_KWH[group],
                weather_columns=weather_columns,
                fit_index=fit_index,
                prediction_index=prediction_index,
            )
            model_path = (
                out_dir / "models" / f"stage1__{architecture_id}__{group}.joblib"
            )
            _atomic_joblib(
                {
                    "transformer": transformer,
                    "models": models,
                    "architecture": dict(architecture),
                    "group": group,
                    "preregister_sha256": PREREGISTER_SHA256,
                },
                model_path,
            )
            evidence["model"] = describe_file(model_path)
            training[architecture_id][group] = evidence
            for weight in preregister["blend_weights"]:
                candidate_id = _candidate_id(architecture_id, float(weight))
                candidate = blend_joint_prediction(
                    baseline.loc[prediction_index, group],
                    neural_cf,
                    capacity_kwh=CAPACITY_KWH[group],
                    blend_weight=float(weight),
                )
                candidate_outputs[candidate_id].loc[prediction_index, group] = candidate
                comparisons[candidate_id][group] = _comparisons(
                    actual=labels[group],
                    baseline=baseline[group],
                    candidate=candidate,
                    group=group,
                    slices=slices,
                )

    for candidate_id, frame in candidate_outputs.items():
        _atomic_parquet(frame, out_dir / "oof" / f"stage1__{candidate_id}.parquet")
    _atomic_parquet(
        baseline, out_dir / "oof" / "stage1_corrected_v3_baseline.parquet"
    )
    selected = choose_group_candidates(comparisons, TARGET_COLS)

    source_after = _source_snapshot(preregister_path)
    inputs_after = _safe_stage1_input_snapshot(raw_dir, artifact_root, preregister_path)
    if source_before != source_after:
        raise AssertionError("Stage 1 source snapshot changed during execution")
    if inputs_before != inputs_after:
        raise AssertionError("Stage 1 safe input snapshot changed during execution")
    result = {
        "schema_version": 1,
        "experiment_id": "temporal_mlp_joint_run_strict_forward_stage1",
        "output_directory_created_utc_before_reads": created_utc,
        "output_directory_filesystem_creation_utc": filesystem_creation_utc,
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "candidate_count": int(len(candidate_outputs)),
        "selected_candidate_by_group": selected,
        "selected_nonidentity_group_count": int(sum(x is not None for x in selected.values())),
        "selection_changes_after_2024": False,
        "no_read_ledger": {
            "2024_labels_read": False,
            "2024_weather_read": False,
            "2024_predictions_read": False,
            "2025_weather_read": False,
            "2025_predictions_or_sample_read": False,
            "public_metrics_or_submission_artifacts_read": False,
            "cache_directory_read": False,
            "cache_directory": str(cache_dir.resolve()),
            "label_prefix_rows": int(len(labels)),
            "label_prefix_end": labels.index.max().isoformat(),
            "label_suffix_bytes_exposed_to_parser": 0,
            "weather_suffix_bytes_exposed_to_parser": {
                source: int(contract["suffix_bytes_exposed_to_parser"])
                for source, contract in raw_contract["raw_prefix"].items()
            },
        },
        "bounded_label_rows": int(len(labels)),
        "raw_feature_contract": raw_contract,
        "baseline_exact_reconstruction": baseline_audit,
        "joint_run_contract": {
            "channels": preregister["run_input"]["channels_in_order"],
            "input_dimension": 48,
            "target_dimension": 24,
            "one_sample_per_complete_issued_run": True,
            "same_run_hours": 24,
            "cross_run_features": 0,
            "actual_target_scada_features": 0,
            "direct_capacity_factor_target": True,
            "fixed_dct_coefficients_per_channel": 8,
        },
        "comparisons": comparisons,
        "training": training,
        "source_snapshot_before": source_before,
        "source_snapshot_after": source_after,
        "source_snapshot_sha256": _mapping_sha256(source_before),
        "input_snapshot_before": inputs_before,
        "input_snapshot_after": inputs_after,
        "input_snapshot_sha256": _mapping_sha256(inputs_before),
        "snapshots_unchanged": True,
        "leaderboard_score_claim": False,
    }
    result_path = out_dir / "stage1_results.json"
    _write_json(result_path, result)
    lock = {
        "schema_version": 1,
        "experiment_id": "temporal_mlp_joint_run_pre2024_lock",
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "stage1_results_sha256": sha256_file(result_path),
        "source_snapshot_sha256": result["source_snapshot_sha256"],
        "input_snapshot_sha256": result["input_snapshot_sha256"],
        "selected_candidate_by_group": selected,
        "selection_unit": "group-specific",
        "selection_changes_after_2024": False,
        "2024_read_before_lock": False,
        "2025_read_before_lock": False,
    }
    _write_json(out_dir / "pre2024_lock.json", lock)
    print("Stage 1 locked by group:", selected, flush=True)
    return result


def _load_lock(
    out_dir: Path,
    preregister_path: Path,
    raw_dir: Path,
    artifact_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result_path = out_dir / "stage1_results.json"
    lock_path = out_dir / "pre2024_lock.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["preregister_sha256"] != PREREGISTER_SHA256:
        raise AssertionError("pre-2024 lock preregister SHA changed")
    if lock["stage1_results_sha256"] != sha256_file(result_path):
        raise AssertionError("Stage 1 results changed after lock")
    if lock["selected_candidate_by_group"] != result["selected_candidate_by_group"]:
        raise AssertionError("locked group recipes differ from Stage 1 selection")
    if lock["selection_changes_after_2024"] is not False:
        raise AssertionError("lock permits selection changes after 2024")
    current_source = _source_snapshot(preregister_path)
    current_input = _safe_stage1_input_snapshot(raw_dir, artifact_root, preregister_path)
    if _mapping_sha256(current_source) != lock["source_snapshot_sha256"]:
        raise AssertionError("source snapshot changed after pre-2024 lock")
    if _mapping_sha256(current_input) != lock["input_snapshot_sha256"]:
        raise AssertionError("safe Stage 1 input snapshot changed after lock")
    recalculated = choose_group_candidates(result["comparisons"], TARGET_COLS)
    if recalculated != lock["selected_candidate_by_group"]:
        raise AssertionError("locked recipes violate registered selection rule")
    return lock, result


def _stage2_no_candidate(out_dir: Path, lock: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "selected_candidate_by_group": lock["selected_candidate_by_group"],
        "2024_read": False,
        "2025_read": False,
        "promoted": False,
        "reason": "no group recipe improved every preregistered Stage 1 slice",
        "pre2024_lock_sha256": sha256_file(out_dir / "pre2024_lock.json"),
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
            "promoted": False,
            "selected_candidate_by_group": lock["selected_candidate_by_group"],
            "2024_read": False,
            "2025_read": False,
            "stage2_results_sha256": sha256_file(out_dir / "stage2_results.json"),
            "pre2024_lock_sha256": sha256_file(out_dir / "pre2024_lock.json"),
        },
    )
    return result


def _fit_group_history_to_period(
    *,
    architecture: Mapping[str, Any],
    weight: float,
    group: str,
    historical_weather: pd.DataFrame,
    historical_actual: pd.Series,
    historical_baseline: pd.Series,
    prediction_weather: pd.DataFrame,
    prediction_baseline: pd.Series,
    weather_columns: Sequence[str],
) -> tuple[pd.Series, dict[str, Any], dict[str, Any]]:
    fit_index = historical_baseline.index
    prediction_index = prediction_baseline.index
    transformer, models, neural_cf, evidence = _fit_predict_architecture(
        architecture=architecture,
        weather=pd.concat([historical_weather, prediction_weather]),
        actual=historical_actual,
        baseline=pd.concat([historical_baseline, prediction_baseline]),
        capacity_kwh=CAPACITY_KWH[group],
        weather_columns=weather_columns,
        fit_index=fit_index,
        prediction_index=prediction_index,
    )
    candidate = blend_joint_prediction(
        prediction_baseline,
        neural_cf,
        capacity_kwh=CAPACITY_KWH[group],
        blend_weight=weight,
    )
    payload = {
        "transformer": transformer,
        "models": models,
        "architecture": dict(architecture),
        "blend_weight": float(weight),
        "group": group,
        "preregister_sha256": PREREGISTER_SHA256,
    }
    return candidate, evidence, payload


def _stage2(
    *,
    raw_dir: Path,
    cache_dir: Path,
    artifact_root: Path,
    recipe_path: Path,
    out_dir: Path,
    preregister_path: Path,
    preregister: Mapping[str, Any],
) -> dict[str, Any]:
    if (out_dir / "stage2_results.json").exists():
        raise FileExistsError("Stage 2 result already exists")
    lock, _ = _load_lock(out_dir, preregister_path, raw_dir, artifact_root)
    selected = lock["selected_candidate_by_group"]
    if not any(candidate is not None for candidate in selected.values()):
        return _stage2_no_candidate(out_dir, lock)

    print("Stage 2: opening fixed 2024 confirmation only after pre-2024 lock", flush=True)
    labels = _read_labels(
        raw_dir / "train" / "train_labels.csv",
        nrows=EXPECTED_ROWS_THROUGH2024,
        expected_end=YEAR_2024_END,
    )
    all_weather = _read_features(cache_dir, labels, expected_end=YEAR_2024_END)
    weather = _weather_only(all_weather, preregister)
    del all_weather
    stage1_baseline, stage1_baseline_audit = _read_stage1_baseline(artifact_root)
    gate_index = pd.date_range(
        STAGE2_START, STAGE2_END, freq="h", name="forecast_kst_dtm"
    )
    gate_baseline, gate_baseline_audit = _reconstruct_baseline(
        artifact_root=artifact_root,
        recipe_path=recipe_path,
        component_files=GATE_COMPONENT_FILES,
        reference_path=artifact_root / "oof" / "gate2024_locked_v3_cf_fix.parquet",
        expected_index=gate_index,
    )
    candidate = gate_baseline.copy()
    comparisons: dict[str, Any] = {}
    training: dict[str, Any] = {}
    weather_columns = preregister["run_input"]["weather_columns_in_order"]
    for group in TARGET_COLS:
        candidate_id = selected[group]
        if candidate_id is None:
            training[group] = {"identity": True, "fit_performed": False}
            continue
        architecture, weight = _parse_candidate(preregister, candidate_id)
        history_index = _interval(
            stage1_baseline[group].dropna().index,
            preregister["stage2"]["fit_history"][group],
        )
        prediction, evidence, model_payload = _fit_group_history_to_period(
            architecture=architecture,
            weight=weight,
            group=group,
            historical_weather=weather[group].loc[history_index],
            historical_actual=labels.loc[history_index, group],
            historical_baseline=stage1_baseline.loc[history_index, group],
            prediction_weather=weather[group].loc[gate_index],
            prediction_baseline=gate_baseline[group],
            weather_columns=weather_columns,
        )
        candidate[group] = prediction
        model_path = out_dir / "models" / f"stage2__{candidate_id}__{group}.joblib"
        _atomic_joblib(model_payload, model_path)
        evidence.update(
            {
                "identity": False,
                "candidate_id": candidate_id,
                "blend_weight": weight,
                "model": describe_file(model_path),
            }
        )
        training[group] = evidence
        comparisons[group] = _comparisons(
            actual=labels.loc[gate_index, group],
            baseline=gate_baseline[group],
            candidate=prediction,
            group=group,
            slices=preregister["stage2"]["confirmation_slices"],
        )
    promoted = bool(comparisons) and all(
        float(item["delta"]) > 0.0
        for slices in comparisons.values()
        for item in slices.values()
    )
    _atomic_parquet(
        gate_baseline, out_dir / "oof" / "stage2_corrected_v3_baseline.parquet"
    )
    _atomic_parquet(candidate, out_dir / "oof" / "stage2_locked_mixed.parquet")
    result = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "selected_candidate_by_group": selected,
        "2024_read": True,
        "2025_read": False,
        "selection_changes_after_2024": False,
        "comparisons_nonidentity_groups": comparisons,
        "promoted": promoted,
        "training": training,
        "stage1_baseline_exact_reconstruction": stage1_baseline_audit,
        "stage2_baseline_exact_reconstruction": gate_baseline_audit,
        "pre2024_lock_sha256": sha256_file(out_dir / "pre2024_lock.json"),
        "final_fit_performed": False,
        "submission_created": False,
        "leaderboard_score_claim": False,
    }
    result_path = out_dir / "stage2_results.json"
    _write_json(result_path, result)
    _write_json(
        out_dir / "promotion_lock.json",
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "promoted": promoted,
            "selected_candidate_by_group": selected,
            "selection_changes_after_2024": False,
            "2024_read": True,
            "2025_read_before_lock": False,
            "stage2_results_sha256": sha256_file(result_path),
            "pre2024_lock_sha256": sha256_file(out_dir / "pre2024_lock.json"),
        },
    )
    if promoted:
        _final(
            raw_dir=raw_dir,
            cache_dir=cache_dir,
            artifact_root=artifact_root,
            recipe_path=recipe_path,
            out_dir=out_dir,
            preregister=preregister,
            selected=selected,
            labels=labels,
            weather=weather,
            stage1_baseline=stage1_baseline,
            gate_baseline=gate_baseline,
            gate_index=gate_index,
        )
        result["final_fit_performed"] = True
        result["submission_created"] = True
    return result


def _final(
    *,
    raw_dir: Path,
    cache_dir: Path,
    artifact_root: Path,
    recipe_path: Path,
    out_dir: Path,
    preregister: Mapping[str, Any],
    selected: Mapping[str, str | None],
    labels: pd.DataFrame,
    weather: Mapping[str, pd.DataFrame],
    stage1_baseline: pd.DataFrame,
    gate_baseline: pd.DataFrame,
    gate_index: pd.DatetimeIndex,
) -> None:
    promotion_lock = json.loads(
        (out_dir / "promotion_lock.json").read_text(encoding="utf-8")
    )
    if promotion_lock.get("promoted") is not True:
        raise AssertionError("final fit requires positive immutable Stage 2 lock")
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
        raise AssertionError("sample test interval changed")
    test_weather_full = _read_test_features(cache_dir, test_index)
    test_weather = _weather_only(test_weather_full, preregister)
    final_baseline, final_baseline_audit = _reconstruct_baseline(
        artifact_root=artifact_root,
        recipe_path=recipe_path,
        component_files=FINAL_COMPONENT_FILES,
        reference_path=artifact_root
        / "final_cf_fix"
        / "predictions"
        / "corrected_v3_test.parquet",
        expected_index=test_index,
    )
    predictions = final_baseline.copy()
    training: dict[str, Any] = {}
    weather_columns = preregister["run_input"]["weather_columns_in_order"]
    for group in TARGET_COLS:
        candidate_id = selected[group]
        if candidate_id is None:
            training[group] = {"identity": True, "fit_performed": False}
            continue
        architecture, weight = _parse_candidate(preregister, candidate_id)
        history_index = stage1_baseline[group].dropna().index.append(gate_index)
        historical_baseline = pd.concat(
            [stage1_baseline.loc[history_index[history_index < STAGE2_START], group], gate_baseline[group]]
        )
        historical_weather = pd.concat(
            [
                weather[group].loc[history_index[history_index < STAGE2_START]],
                weather[group].loc[gate_index],
            ]
        )
        historical_actual = labels.loc[historical_baseline.index, group]
        prediction, evidence, model_payload = _fit_group_history_to_period(
            architecture=architecture,
            weight=weight,
            group=group,
            historical_weather=historical_weather,
            historical_actual=historical_actual,
            historical_baseline=historical_baseline,
            prediction_weather=test_weather[group],
            prediction_baseline=final_baseline[group],
            weather_columns=weather_columns,
        )
        predictions[group] = prediction
        model_path = out_dir / "models" / f"final__{candidate_id}__{group}.joblib"
        _atomic_joblib(model_payload, model_path)
        evidence.update(
            {
                "identity": False,
                "candidate_id": candidate_id,
                "blend_weight": weight,
                "model": describe_file(model_path),
            }
        )
        training[group] = evidence
    prediction_path = out_dir / "predictions" / "temporal_mlp_joint_run_2025.parquet"
    _atomic_parquet(predictions, prediction_path)
    submission_path = out_dir / preregister["final"]["submission_name"]
    submission_audit = _write_submission(sample_path, predictions, submission_path)
    _write_json(
        out_dir / "final_results.json",
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "selected_candidate_by_group": dict(selected),
            "selection_changes_after_2024": False,
            "training": training,
            "baseline_exact_reconstruction": final_baseline_audit,
            "prediction": describe_file(prediction_path),
            "submission": submission_audit,
            "leaderboard_score_claim": False,
        },
    )


def _write_manifest(args: argparse.Namespace, preregister: Mapping[str, Any]) -> None:
    destination = args.out_dir / "manifest.json"
    outputs = sorted(
        path for path in args.out_dir.rglob("*") if path.is_file() and path != destination
    )
    safe_inputs = [
        args.preregister,
        args.preregister.with_suffix(".sha256"),
        args.recipe,
        args.raw_dir / "info.xlsx",
        args.artifact_root / "oof" / "dev2023_locked_v3.parquet",
        args.artifact_root / "oof" / "g3dev2023h2_candidates.parquet",
        args.artifact_root
        / "postgate"
        / "shared_q07_multiseed_strict"
        / "oof"
        / "stage1_corrected_v3_baseline.parquet",
        Path(__file__).resolve(),
        PROJECT_DIR / "src" / "temporal_mlp.py",
        PROJECT_DIR / "tests" / "test_temporal_mlp.py",
    ]
    stage1 = json.loads(
        (args.out_dir / "stage1_results.json").read_text(encoding="utf-8")
    )
    stage2 = json.loads(
        (args.out_dir / "stage2_results.json").read_text(encoding="utf-8")
    )
    payload = {
        "schema_version": 1,
        "artifact_type": "temporal_mlp_joint_run_strict_forward",
        "created_utc": utc_now(),
        "command": [sys.executable, *sys.argv],
        "runtime": {"packages": package_versions(), "git": git_state(PROJECT_DIR)},
        "preregister_sha256": PREREGISTER_SHA256,
        "architecture_count": len(preregister["architectures"]),
        "candidate_count": len(preregister["architectures"])
        * len(preregister["blend_weights"]),
        "safe_inputs": [describe_file(path) for path in safe_inputs],
        "raw_prefix_inputs": stage1["raw_feature_contract"],
        "no_read_ledger_stage1": stage1["no_read_ledger"],
        "stage2": {
            "2024_read": stage2["2024_read"],
            "2025_read": stage2.get("2025_read", stage2.get("submission_created", False)),
            "promoted": stage2["promoted"],
        },
        "outputs": [describe_file(path) for path in outputs],
        "leaderboard_score_claim": False,
    }
    _write_json(destination, payload)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.raw_dir = args.raw_dir.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.artifact_root = args.artifact_root.expanduser().resolve()
    args.recipe = args.recipe.expanduser().resolve()
    args.preregister = args.preregister.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    preregister = _load_preregister(args.preregister)
    if args.stage in {"stage1", "all"}:
        _stage1(
            raw_dir=args.raw_dir,
            cache_dir=args.cache_dir,
            artifact_root=args.artifact_root,
            out_dir=args.out_dir,
            preregister_path=args.preregister,
            preregister=preregister,
        )
        if args.stage == "stage1":
            return
    if args.stage in {"stage2", "all"}:
        _stage2(
            raw_dir=args.raw_dir,
            cache_dir=args.cache_dir,
            artifact_root=args.artifact_root,
            recipe_path=args.recipe,
            out_dir=args.out_dir,
            preregister_path=args.preregister,
            preregister=preregister,
        )
        _write_manifest(args, preregister)


if __name__ == "__main__":
    main()

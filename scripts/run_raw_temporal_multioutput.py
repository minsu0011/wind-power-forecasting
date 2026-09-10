"""Strict-forward raw-node DCT multi-output Ridge/PLS experiment."""

from __future__ import annotations

import argparse
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

from scripts import run_ficr_bayes_decision_strict as bayes  # noqa: E402
from scripts import run_raw_grid_wind_lgb as raw_protocol  # noqa: E402
from scripts import run_shared_q07_multiseed as shared  # noqa: E402
from scripts import run_weather_quantile_bayes as weather_protocol  # noqa: E402
from src.manifest import (  # noqa: E402
    describe_file,
    git_state,
    package_versions,
    sha256_file,
    utc_now,
)
from src.metric import CAPACITY_KWH, TARGET_COLS, score_details  # noqa: E402
from src.raw_temporal_multioutput import (  # noqa: E402
    BLEND_WEIGHTS,
    CANDIDATE_KEYS,
    DCT_COEFFICIENTS,
    ESTIMATOR_IDS,
    EXPECTED_INPUT_DIMENSION,
    EXPECTED_RAW_CHANNELS,
    RawTemporalMultiOutputRegressor,
    assert_fit_before_apply,
    blend_direct_prediction,
    candidate_frame,
    parse_candidate_key,
    select_group_candidate,
)


PREREGISTER_SHA256 = "dce763f52e80ab0c90abad97b6811efec9ceab92c256331a424a0814fb425e69"
FEASIBILITY_SHA256 = "c03db059c92780a23c05aacaaf665177962019bd1199d79c1597fb690b5f50c4"
RAW_DEPENDENCY_SHA256 = "06b56426347f64d91f1a69e1c56083014f0dd003ee8186fa98ffe97ad3b61d01"
STAGE1_REQUIRED: Mapping[str, tuple[str, ...]] = {
    "kpx_group_1": ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4"),
    "kpx_group_2": ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4"),
    "kpx_group_3": ("full", "Q3", "Q4"),
}
STAGE2_REQUIRED = ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4")
TIME_COL = "forecast_kst_dtm"
YEAR_2022 = raw_protocol.YEAR_2022
YEAR_2023 = raw_protocol.YEAR_2023
YEAR_2024 = raw_protocol.YEAR_2024
YEAR_2025 = raw_protocol.YEAR_2025
PRE2024 = raw_protocol.PRE2024
TRAIN_ALL = raw_protocol.TRAIN_ALL
G3_H1 = weather_protocol._year_segments(2023)["H1"]
G3_H2 = weather_protocol._year_segments(2023)["H2"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("stage1", "stage2", "final", "manifest", "all"),
        required=True,
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=Path(r"data/local/open")
    )
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/postgate/raw_temporal_multioutput_strict_v1"),
    )
    parser.add_argument(
        "--preregister",
        type=Path,
        default=Path("configs/raw_temporal_multioutput_preregister_v1.json"),
    )
    return parser.parse_args(argv)


def _verify_file_and_sidecar(path: Path, expected: str) -> None:
    if sha256_file(path) != expected:
        raise AssertionError(f"content hash changed: {path}")
    sidecar = path.with_suffix(".sha256")
    if sidecar.read_text(encoding="utf-8") != f"{expected}  {path.name}\n":
        raise AssertionError(f"SHA sidecar changed: {sidecar}")


def _verify_preregister(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    _verify_file_and_sidecar(path, PREREGISTER_SHA256)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["experiment_id"] != "raw_temporal_multioutput_strict_forward_v1":
        raise AssertionError("experiment id changed")
    if int(payload["run_input"]["raw_channel_count"]) != EXPECTED_RAW_CHANNELS:
        raise AssertionError("raw channel count changed")
    if int(payload["run_input"]["dct_coefficients_per_channel"]) != DCT_COEFFICIENTS:
        raise AssertionError("DCT coefficient count changed")
    if int(payload["run_input"]["input_dimension"]) != EXPECTED_INPUT_DIMENSION:
        raise AssertionError("DCT input dimension changed")
    if tuple(item["id"] for item in payload["estimators"]) != ESTIMATOR_IDS:
        raise AssertionError("estimator order changed")
    family = payload["candidate_family"]
    if tuple(map(float, family["fixed_blend_weights"])) != BLEND_WEIGHTS:
        raise AssertionError("blend weights changed")
    if tuple(family["candidate_order"]) != CANDIDATE_KEYS:
        raise AssertionError("candidate order changed")

    feasibility = PROJECT_DIR / "configs/raw_temporal_multioutput_feasibility_v1.json"
    _verify_file_and_sidecar(feasibility, FEASIBILITY_SHA256)
    feasibility_payload = json.loads(feasibility.read_text(encoding="utf-8"))
    if feasibility_payload["preregister_sha256"] != PREREGISTER_SHA256:
        raise AssertionError("feasibility/preregister binding changed")

    dependency = PROJECT_DIR / payload["dependency_contract"]["raw_grid_preregister"]["path"]
    if sha256_file(dependency) != RAW_DEPENDENCY_SHA256:
        raise AssertionError("raw-grid dependency changed")
    if payload["dependency_contract"]["raw_grid_preregister"]["sha256"] != RAW_DEPENDENCY_SHA256:
        raise AssertionError("raw-grid dependency binding changed")
    raw_payload = json.loads(dependency.read_text(encoding="utf-8"))
    return payload, raw_payload, dependency


def _source_paths(
    preregister_path: Path, raw_dependency_path: Path
) -> dict[str, Path]:
    return {
        "runner": Path(__file__).resolve(),
        "raw_temporal_module": PROJECT_DIR / "src/raw_temporal_multioutput.py",
        "raw_grid_module": PROJECT_DIR / "src/raw_grid_wind.py",
        "raw_grid_reader": PROJECT_DIR / "scripts/run_raw_grid_wind_lgb.py",
        "weather_protocol": PROJECT_DIR / "scripts/run_weather_quantile_bayes.py",
        "bounded_reader": PROJECT_DIR / "scripts/run_shared_q07_multiseed.py",
        "score_lock_helper": PROJECT_DIR / "scripts/run_ficr_bayes_decision_strict.py",
        "run_sequence_guard": PROJECT_DIR / "src/run_sequence_residual.py",
        "temporal_run_key": PROJECT_DIR / "src/temporal.py",
        "feature_dependency": PROJECT_DIR / "src/features.py",
        "weather_quantile_dependency": PROJECT_DIR / "src/weather_quantile.py",
        "probabilistic_dependency": PROJECT_DIR / "src/probabilistic.py",
        "metric": PROJECT_DIR / "src/metric.py",
        "manifest": PROJECT_DIR / "src/manifest.py",
        "focused_test": PROJECT_DIR / "tests/test_raw_temporal_multioutput.py",
        "preregister": preregister_path.resolve(),
        "preregister_sidecar": preregister_path.with_suffix(".sha256").resolve(),
        "feasibility": PROJECT_DIR / "configs/raw_temporal_multioutput_feasibility_v1.json",
        "feasibility_sidecar": PROJECT_DIR / "configs/raw_temporal_multioutput_feasibility_v1.sha256",
        "raw_grid_preregister_dependency": raw_dependency_path.resolve(),
        "raw_grid_preregister_sidecar": raw_dependency_path.with_suffix(".sha256").resolve(),
    }


def _snapshot_named(paths: Mapping[str, Path]) -> dict[str, Any]:
    return {name: shared._snapshot_file(path) for name, path in paths.items()}


def _assert_snapshot_equal(
    before: Mapping[str, Any], after: Mapping[str, Any], *, name: str
) -> None:
    if json.dumps(before, sort_keys=True, default=str) != json.dumps(
        after, sort_keys=True, default=str
    ):
        raise AssertionError(f"{name} changed during execution")


def _stage1_segments(group: str) -> dict[str, pd.DatetimeIndex]:
    segments = weather_protocol._year_segments(2023)
    if group == "kpx_group_3":
        return {"full": segments["H2"], "Q3": segments["Q3"], "Q4": segments["Q4"]}
    return {name: segments[name] for name in STAGE1_REQUIRED[group]}


def _aggregate_summary(actual: pd.DataFrame, prediction: pd.DataFrame) -> dict[str, Any]:
    return score_details(actual, prediction).as_dict()


def _aggregate_comparison(
    actual: pd.DataFrame,
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    segments: Mapping[str, pd.DatetimeIndex],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, index in segments.items():
        base = _aggregate_summary(actual.loc[index], baseline.loc[index])
        proposed = _aggregate_summary(actual.loc[index], candidate.loc[index])
        result[name] = {
            "baseline": base,
            "candidate": proposed,
            "delta": float(proposed["total_score"]) - float(base["total_score"]),
        }
    return result


def _fit_group(
    *,
    features: pd.DataFrame,
    actual_kwh: pd.Series,
    baseline: pd.Series,
    group: str,
    fit_index: pd.DatetimeIndex,
    application_index: pd.DatetimeIndex,
    estimator_ids: Sequence[str] = ESTIMATOR_IDS,
) -> tuple[
    dict[str, RawTemporalMultiOutputRegressor],
    dict[str, pd.Series],
    pd.DataFrame,
    dict[str, Any],
]:
    split = assert_fit_before_apply(fit_index, application_index)
    models: dict[str, RawTemporalMultiOutputRegressor] = {}
    raw_predictions: dict[str, pd.Series] = {}
    metadata: dict[str, Any] = {"split": split, "models": {}}
    for estimator_id in estimator_ids:
        print(
            f"fit raw-temporal {group}/{estimator_id}: "
            f"{fit_index.min()} -> {application_index.min()}",
            flush=True,
        )
        model = RawTemporalMultiOutputRegressor(estimator_id).fit(
            features.loc[fit_index],
            actual_kwh.loc[fit_index] / CAPACITY_KWH[group],
        )
        prediction = model.predict(features.loc[application_index])
        models[estimator_id] = model
        raw_predictions[estimator_id] = prediction
        metadata["models"][estimator_id] = model.metadata()
    if tuple(estimator_ids) == ESTIMATOR_IDS:
        candidates = candidate_frame(
            baseline.loc[application_index],
            raw_predictions,
            capacity_kwh=CAPACITY_KWH[group],
        )
    elif len(estimator_ids) == 1:
        candidates = pd.DataFrame(index=application_index)
    else:
        raise ValueError("estimator subset must be all Stage1 models or one locked model")
    return models, raw_predictions, candidates, metadata


def _save_and_reload_group(
    *,
    out_dir: Path,
    stage: str,
    group: str,
    models: Mapping[str, RawTemporalMultiOutputRegressor],
    raw_predictions: Mapping[str, pd.Series],
    candidates: pd.DataFrame,
    baseline: pd.Series,
    application_features: pd.DataFrame,
) -> tuple[list[Path], dict[str, Any]]:
    model_path = out_dir / "models" / f"{stage}__{group}.joblib"
    raw_path = out_dir / "predictions" / f"{stage}__{group}__raw_cf.parquet"
    candidate_path = out_dir / "predictions" / f"{stage}__{group}__candidates.parquet"
    baseline_path = out_dir / "predictions" / f"{stage}__{group}__baseline.parquet"
    bayes._atomic_joblib(dict(models), model_path)
    raw_frame = pd.DataFrame(raw_predictions)
    bayes._atomic_parquet(raw_frame, raw_path)
    bayes._atomic_parquet(candidates, candidate_path)
    bayes._atomic_parquet(baseline.rename(group).to_frame(), baseline_path)

    loaded: dict[str, RawTemporalMultiOutputRegressor] = joblib.load(model_path)
    if tuple(loaded) != tuple(models):
        raise AssertionError("reloaded estimator order changed")
    checks: dict[str, Any] = {}
    for estimator_id, model in models.items():
        repeated = loaded[estimator_id].predict(application_features)
        exact = np.array_equal(
            repeated.to_numpy(dtype=np.float64),
            raw_predictions[estimator_id].to_numpy(dtype=np.float64),
        )
        transformer_exact = all(
            np.array_equal(
                getattr(loaded[estimator_id].transformer_, field),
                getattr(model.transformer_, field),
            )
            for field in ("medians_", "means_", "scales_")
        )
        if not exact or not transformer_exact:
            raise AssertionError(f"{stage}/{group}/{estimator_id} reload changed output")
        checks[estimator_id] = {
            "prediction_bit_exact": True,
            "transformer_arrays_bit_exact": True,
            "metadata": loaded[estimator_id].metadata(),
        }
    stored_raw = pd.read_parquet(raw_path)
    stored_candidates = pd.read_parquet(candidate_path)
    if not stored_raw.index.equals(raw_frame.index) or not np.array_equal(
        stored_raw.to_numpy(), raw_frame.to_numpy()
    ):
        raise AssertionError("raw prediction parquet roundtrip changed")
    if not stored_candidates.index.equals(candidates.index) or not np.array_equal(
        stored_candidates.to_numpy(), candidates.to_numpy()
    ):
        raise AssertionError("candidate parquet roundtrip changed")
    paths = [model_path, raw_path, candidate_path, baseline_path]
    return paths, {
        "estimators": checks,
        "raw_prediction_parquet_bit_exact": True,
        "candidate_parquet_bit_exact": True,
        "candidate_columns": list(candidates.columns),
    }


def _fixed_candidate(
    baseline: pd.Series,
    prediction: pd.Series,
    *,
    group: str,
    candidate_key: str,
) -> pd.Series:
    _, weight = parse_candidate_key(candidate_key)
    return blend_direct_prediction(
        baseline,
        prediction,
        capacity_kwh=CAPACITY_KWH[group],
        weight=weight,
    )


def _stage1(
    *,
    raw_dir: Path,
    artifact_root: Path,
    out_dir: Path,
    preregister_path: Path,
    preregister: Mapping[str, Any],
    raw_contract: Mapping[str, Any],
    raw_dependency_path: Path,
) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"Stage1 requires a new output directory: {out_dir}")
    out_dir.mkdir(parents=True)
    for source in (
        preregister_path,
        preregister_path.with_suffix(".sha256"),
        PROJECT_DIR / "configs/raw_temporal_multioutput_feasibility_v1.json",
        PROJECT_DIR / "configs/raw_temporal_multioutput_feasibility_v1.sha256",
    ):
        bayes._copy_exclusive(source, out_dir / source.name)

    source_paths = _source_paths(preregister_path, raw_dependency_path)
    sources_before = _snapshot_named(source_paths)
    prefix_specs = raw_contract["physical_stage1_inputs"]
    input_prefixes_before = {
        "labels_g12_fit": raw_protocol._prefix_snapshot(
            raw_dir / "train/train_labels.csv", prefix_specs["labels_g12_fit_prefix"]
        ),
        "labels_g3_fit": raw_protocol._prefix_snapshot(
            raw_dir / "train/train_labels.csv", prefix_specs["labels_g3_fit_prefix"]
        ),
        "labels_stage1_score": raw_protocol._prefix_snapshot(
            raw_dir / "train/train_labels.csv",
            prefix_specs["labels_stage1_score_prefix_after_lock"],
        ),
        "ldaps_weather": raw_protocol._prefix_snapshot(
            raw_dir / "train/ldaps_train.csv", prefix_specs["ldaps_weather_prefix"]
        ),
        "gfs_weather": raw_protocol._prefix_snapshot(
            raw_dir / "train/gfs_train.csv", prefix_specs["gfs_weather_prefix"]
        ),
    }
    features, weather_evidence = raw_protocol._read_weather_features(
        raw_dir, raw_contract, period="stage1"
    )
    baseline = raw_protocol._stage1_baseline(artifact_root, raw_contract)
    if not baseline.loc[YEAR_2023, list(TARGET_COLS[:2])].notna().all().all():
        raise AssertionError("G1/G2 Stage1 baseline is incomplete")
    if not baseline.loc[G3_H2, "kpx_group_3"].notna().all():
        raise AssertionError("G3 Stage1 baseline is incomplete")

    all_models: dict[str, dict[str, RawTemporalMultiOutputRegressor]] = {}
    all_raw: dict[str, dict[str, pd.Series]] = {}
    all_candidates: dict[str, pd.DataFrame] = {}
    training: dict[str, Any] = {}
    reload_checks: dict[str, Any] = {}
    output_paths: list[Path] = []

    labels_g12, labels_g12_evidence = raw_protocol._read_label_prefix(
        raw_dir,
        prefix_specs["labels_g12_fit_prefix"],
        TARGET_COLS[:2],
    )
    if not labels_g12.index.equals(YEAR_2022):
        raise AssertionError("G1/G2 fit-label prefix changed")
    for group in TARGET_COLS[:2]:
        models, raw_predictions, candidates, metadata = _fit_group(
            features=features,
            actual_kwh=labels_g12[group],
            baseline=baseline[group],
            group=group,
            fit_index=YEAR_2022,
            application_index=YEAR_2023,
        )
        paths, checks = _save_and_reload_group(
            out_dir=out_dir,
            stage="stage1",
            group=group,
            models=models,
            raw_predictions=raw_predictions,
            candidates=candidates,
            baseline=baseline.loc[YEAR_2023, group],
            application_features=features.loc[YEAR_2023],
        )
        all_models[group] = models
        all_raw[group] = raw_predictions
        all_candidates[group] = candidates
        training[group] = metadata
        reload_checks[group] = checks
        output_paths.extend(paths)
    g12_record_path = out_dir / "stage1_g12_prescore_record.json"
    bayes._write_json(
        g12_record_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "fit_label_evidence": labels_g12_evidence,
            "materialized_target_columns": list(TARGET_COLS[:2]),
            "application_target_values_materialized": False,
            "training": {group: training[group] for group in TARGET_COLS[:2]},
            "reload_checks": {group: reload_checks[group] for group in TARGET_COLS[:2]},
            "outputs": [describe_file(path) for path in output_paths],
            "created_before_any_2023_application_target_value": True,
        },
    )
    g12_lock_path = out_dir / "stage1_g12_prescore_lock.json"
    bayes._write_json(
        g12_lock_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "record": describe_file(g12_record_path),
            "candidate_artifacts": [describe_file(path) for path in output_paths],
            "models_preprocessors_predictions_candidates_frozen": True,
            "created_before_any_2023_application_target_value": True,
        },
    )

    labels_g3, labels_g3_evidence = raw_protocol._read_label_prefix(
        raw_dir,
        prefix_specs["labels_g3_fit_prefix"],
        (TARGET_COLS[2],),
    )
    if not labels_g3.index.equals(YEAR_2022.append(G3_H1)):
        raise AssertionError("G3 fit-label prefix changed")
    group = TARGET_COLS[2]
    models, raw_predictions, candidates, metadata = _fit_group(
        features=features,
        actual_kwh=labels_g3[group],
        baseline=baseline[group],
        group=group,
        fit_index=G3_H1,
        application_index=G3_H2,
    )
    g3_paths, checks = _save_and_reload_group(
        out_dir=out_dir,
        stage="stage1",
        group=group,
        models=models,
        raw_predictions=raw_predictions,
        candidates=candidates,
        baseline=baseline.loc[G3_H2, group],
        application_features=features.loc[G3_H2],
    )
    all_models[group] = models
    all_raw[group] = raw_predictions
    all_candidates[group] = candidates
    training[group] = metadata
    reload_checks[group] = checks
    output_paths.extend(g3_paths)
    g3_record_path = out_dir / "stage1_g3_prescore_record.json"
    bayes._write_json(
        g3_record_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "g12_prescore_lock": describe_file(g12_lock_path),
            "fit_label_evidence": labels_g3_evidence,
            "materialized_target_columns": [group],
            "other_group_application_target_values_materialized": False,
            "training": training[group],
            "reload_checks": reload_checks[group],
            "outputs": [describe_file(path) for path in g3_paths],
            "created_before_any_g3_H2_application_target_value": True,
        },
    )
    g3_lock_path = out_dir / "stage1_g3_prescore_lock.json"
    bayes._write_json(
        g3_lock_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "g12_prescore_lock": describe_file(g12_lock_path),
            "record": describe_file(g3_record_path),
            "candidate_artifacts": [describe_file(path) for path in g3_paths],
            "models_preprocessors_predictions_candidates_frozen": True,
            "created_before_any_g3_H2_application_target_value": True,
        },
    )

    global_record_path = out_dir / "stage1_global_prescore_record.json"
    bayes._write_json(
        global_record_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "g12_prescore_lock": describe_file(g12_lock_path),
            "g3_prescore_lock": describe_file(g3_lock_path),
            "weather_evidence": weather_evidence,
            "baseline_sources": {
                "g12": describe_file(artifact_root / "oof/dev2023_locked_v3.parquet"),
                "g3": describe_file(artifact_root / "oof/g3dev2023h2_candidates.parquet"),
            },
            "sources_before": sources_before,
            "input_prefixes_before": input_prefixes_before,
            "training": training,
            "reload_checks": reload_checks,
            "all_outputs": [describe_file(path) for path in output_paths],
            "validation_target_columns_materialized": [],
            "created_before_any_stage1_application_target_value": True,
            "2024_weather_read": False,
            "2024_label_read": False,
            "2025_read": False,
        },
    )
    global_lock_path = out_dir / "stage1_global_prescore_lock.json"
    bayes._write_json(
        global_lock_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "record": describe_file(global_record_path),
            "g12_prescore_lock": describe_file(g12_lock_path),
            "g3_prescore_lock": describe_file(g3_lock_path),
            "candidate_artifacts": [describe_file(path) for path in output_paths],
            "all_models_preprocessors_predictions_candidates_frozen": True,
            "created_before_any_stage1_application_target_value": True,
        },
    )

    score_labels, score_label_evidence = raw_protocol._read_label_prefix(
        raw_dir,
        prefix_specs["labels_stage1_score_prefix_after_lock"],
        TARGET_COLS,
    )
    if not score_labels.index.equals(PRE2024):
        raise AssertionError("Stage1 score-label prefix changed")
    group_results: dict[str, Any] = {}
    locked_candidates: dict[str, str | None] = {}
    passed_groups: list[str] = []
    for group in TARGET_COLS:
        application_index = YEAR_2023 if group in TARGET_COLS[:2] else G3_H2
        segments = _stage1_segments(group)
        comparisons = {
            key: bayes._comparison(
                score_labels.loc[application_index, group],
                baseline.loc[application_index, group],
                all_candidates[group][key],
                group,
                segments,
            )
            for key in CANDIDATE_KEYS
        }
        selected, selection_audit = select_group_candidate(
            comparisons, STAGE1_REQUIRED[group]
        )
        locked_candidates[group] = selected
        if selected is not None:
            passed_groups.append(group)
        group_results[group] = {
            "fit_interval": training[group]["split"],
            "application_interval": {
                "start": application_index.min(),
                "end": application_index.max(),
            },
            "required_slices": list(STAGE1_REQUIRED[group]),
            "comparisons": comparisons,
            "selection": selection_audit,
            "locked_candidate": selected if selected is not None else "identity",
            "passed": selected is not None,
        }
    sources_after = _snapshot_named(source_paths)
    _assert_snapshot_equal(sources_before, sources_after, name="transitive sources")
    result = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "global_prescore_lock": describe_file(global_lock_path),
        "score_label_evidence_after_global_prescore_lock": score_label_evidence,
        "group_results": group_results,
        "locked_candidates": locked_candidates,
        "passed_groups": passed_groups,
        "passed_groups_sha256": raw_protocol.canonical_sha256(passed_groups),
        "stage2_unlocked": bool(passed_groups),
        "2024_weather_read": False,
        "2024_label_read": False,
        "2025_read": False,
        "leaderboard_score_claim": False,
    }
    result_path = out_dir / "stage1_results.json"
    bayes._write_json(result_path, result)
    lock = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "global_prescore_lock": describe_file(global_lock_path),
        "stage1_results": describe_file(result_path),
        "locked_candidates": locked_candidates,
        "passed_groups": passed_groups,
        "passed_groups_sha256": result["passed_groups_sha256"],
        "stage2_unlocked": bool(passed_groups),
        "no_2024_or_2025_read": True,
    }
    bayes._write_json(out_dir / "stage1_promotion_lock.json", lock)
    print(f"Stage1 passed groups: {passed_groups}", flush=True)
    return lock


def _load_stage1_lock(out_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    lock_path = out_dir / "stage1_promotion_lock.json"
    result_path = out_dir / "stage1_results.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if lock["preregister_sha256"] != PREREGISTER_SHA256:
        raise AssertionError("Stage1 preregister binding changed")
    if lock["stage1_results"]["sha256"] != sha256_file(result_path):
        raise AssertionError("Stage1 result changed")
    if lock["passed_groups"] != result["passed_groups"]:
        raise AssertionError("Stage1 passed-group lock changed")
    return lock, result


def _stage2_skip(out_dir: Path, stage1_lock: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "stage1_promotion_lock": describe_file(out_dir / "stage1_promotion_lock.json"),
        "performed": False,
        "reason": "no Stage1 group passed every registered slice",
        "stage1_passed_groups": [],
        "group_gate_passed": False,
        "mixed_aggregate_gate_passed": False,
        "promoted": False,
        "2024_weather_read": False,
        "2024_label_read": False,
        "2025_read": False,
    }
    path = out_dir / "stage2_results.json"
    bayes._write_json(path, result)
    lock = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "stage1_promotion_lock": describe_file(out_dir / "stage1_promotion_lock.json"),
        "stage2_results": describe_file(path),
        "locked_candidates": dict(stage1_lock["locked_candidates"]),
        "passed_groups": [],
        "executed": False,
        "group_gate_passed": {},
        "mixed_aggregate_gate_passed": False,
        "promoted": False,
        "2024_read": False,
        "2025_read": False,
    }
    bayes._write_json(out_dir / "stage2_promotion_lock.json", lock)
    return lock


def _stage2(
    *,
    raw_dir: Path,
    artifact_root: Path,
    out_dir: Path,
    raw_contract: Mapping[str, Any],
) -> dict[str, Any]:
    if (out_dir / "stage2_results.json").exists():
        raise FileExistsError(out_dir / "stage2_results.json")
    stage1_lock, _ = _load_stage1_lock(out_dir)
    passed_groups = list(stage1_lock["passed_groups"])
    if not passed_groups:
        return _stage2_skip(out_dir, stage1_lock)

    features, weather_evidence = raw_protocol._read_weather_features(
        raw_dir, raw_contract, period="train_all"
    )
    fit_labels, fit_label_evidence = raw_protocol._read_label_prefix(
        raw_dir,
        raw_contract["physical_stage1_inputs"]["labels_stage1_score_prefix_after_lock"],
        passed_groups,
    )
    baseline_spec = raw_contract["baseline_contract"]["stage2_after_group_promotion_only"]
    baseline_path = artifact_root / "oof/gate2024_locked_v3_cf_fix.parquet"
    raw_protocol._assert_file_identity(baseline_path, baseline_spec, "Stage2 baseline")
    baseline = bayes._read_prediction(
        baseline_path, YEAR_2024, required_columns=TARGET_COLS
    )

    fixed_candidates = pd.DataFrame(index=YEAR_2024, columns=passed_groups, dtype=float)
    training: dict[str, Any] = {}
    reload_checks: dict[str, Any] = {}
    output_paths: list[Path] = []
    for group in passed_groups:
        candidate_key = str(stage1_lock["locked_candidates"][group])
        estimator_id, _ = parse_candidate_key(candidate_key)
        fit_index = PRE2024 if group in TARGET_COLS[:2] else YEAR_2023
        models, raw_predictions, _, metadata = _fit_group(
            features=features,
            actual_kwh=fit_labels[group],
            baseline=baseline[group],
            group=group,
            fit_index=fit_index,
            application_index=YEAR_2024,
            estimator_ids=(estimator_id,),
        )
        fixed = _fixed_candidate(
            baseline[group],
            raw_predictions[estimator_id],
            group=group,
            candidate_key=candidate_key,
        )
        candidate_frame_fixed = fixed.rename(candidate_key).to_frame()
        paths, checks = _save_and_reload_group(
            out_dir=out_dir,
            stage="stage2",
            group=group,
            models=models,
            raw_predictions=raw_predictions,
            candidates=candidate_frame_fixed,
            baseline=baseline[group],
            application_features=features.loc[YEAR_2024],
        )
        fixed_candidates[group] = fixed
        training[group] = {
            **metadata,
            "fixed_candidate": candidate_key,
            "selection_changes": False,
        }
        reload_checks[group] = checks
        output_paths.extend(paths)

    mixed = baseline.copy()
    for group in passed_groups:
        mixed[group] = fixed_candidates[group]
    mixed_path = out_dir / "predictions/stage2_mixed_candidate.parquet"
    bayes._atomic_parquet(mixed, mixed_path)
    output_paths.append(mixed_path)
    record_path = out_dir / "stage2_prescore_record.json"
    bayes._write_json(
        record_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "stage1_promotion_lock": describe_file(out_dir / "stage1_promotion_lock.json"),
            "passed_groups_only": passed_groups,
            "fixed_candidates": {
                group: stage1_lock["locked_candidates"][group] for group in passed_groups
            },
            "weather_evidence": weather_evidence,
            "fit_label_evidence": fit_label_evidence,
            "training": training,
            "reload_checks": reload_checks,
            "outputs": [describe_file(path) for path in output_paths],
            "application_target_columns_materialized": [],
            "created_before_any_2024_application_target_value": True,
            "2024_weather_read": True,
            "2024_label_read": False,
            "2025_read": False,
        },
    )
    lock_path = out_dir / "stage2_global_prescore_lock.json"
    bayes._write_json(
        lock_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "record": describe_file(record_path),
            "candidate_artifacts": [describe_file(path) for path in output_paths],
            "fixed_models_preprocessors_predictions_mixed_candidate_frozen": True,
            "created_before_any_2024_application_target_value": True,
        },
    )

    score_labels, score_label_evidence = raw_protocol._read_full_labels(
        raw_dir, raw_contract, TARGET_COLS
    )
    segments = {
        name: weather_protocol._year_segments(2024)[name] for name in STAGE2_REQUIRED
    }
    group_results: dict[str, Any] = {}
    all_groups_positive = True
    for group in passed_groups:
        comparison = bayes._comparison(
            score_labels.loc[YEAR_2024, group],
            baseline[group],
            fixed_candidates[group],
            group,
            segments,
        )
        deltas = [float(comparison[name]["delta"]) for name in STAGE2_REQUIRED]
        positive = all(delta > 0.0 for delta in deltas)
        all_groups_positive = all_groups_positive and positive
        group_results[group] = {
            "locked_candidate": stage1_lock["locked_candidates"][group],
            "comparison": comparison,
            "minimum_delta": min(deltas),
            "mean_delta": float(np.mean(deltas)),
            "all_seven_strictly_positive": positive,
        }
    aggregate = _aggregate_comparison(
        score_labels.loc[YEAR_2024, list(TARGET_COLS)], baseline, mixed, segments
    )
    aggregate_deltas = [float(aggregate[name]["delta"]) for name in STAGE2_REQUIRED]
    aggregate_positive = all(delta > 0.0 for delta in aggregate_deltas)
    promoted = bool(all_groups_positive and aggregate_positive)
    result = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "stage1_promotion_lock": describe_file(out_dir / "stage1_promotion_lock.json"),
        "stage2_prescore_lock": describe_file(lock_path),
        "score_label_evidence_after_prescore_lock": score_label_evidence,
        "passed_groups": passed_groups,
        "group_results": group_results,
        "mixed_aggregate_comparison": aggregate,
        "mixed_minimum_delta": min(aggregate_deltas),
        "mixed_mean_delta": float(np.mean(aggregate_deltas)),
        "group_gate_passed": all_groups_positive,
        "mixed_aggregate_gate_passed": aggregate_positive,
        "promoted": promoted,
        "2024_weather_read": True,
        "2024_label_read": True,
        "2025_read": False,
        "leaderboard_score_claim": False,
    }
    result_path = out_dir / "stage2_results.json"
    bayes._write_json(result_path, result)
    lock = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "stage1_promotion_lock": describe_file(out_dir / "stage1_promotion_lock.json"),
        "stage2_prescore_lock": describe_file(lock_path),
        "stage2_results": describe_file(result_path),
        "locked_candidates": dict(stage1_lock["locked_candidates"]),
        "passed_groups": passed_groups,
        "group_gate_passed": all_groups_positive,
        "mixed_aggregate_gate_passed": aggregate_positive,
        "promoted": promoted,
        "no_reselection_or_retuning": True,
        "2024_read": True,
        "2025_read": False,
    }
    bayes._write_json(out_dir / "stage2_promotion_lock.json", lock)
    print(
        f"Stage2 group_gate={all_groups_positive}, mixed_gate={aggregate_positive}, "
        f"promoted={promoted}",
        flush=True,
    )
    return lock


def _load_stage2_lock(out_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    lock_path = out_dir / "stage2_promotion_lock.json"
    result_path = out_dir / "stage2_results.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if lock["preregister_sha256"] != PREREGISTER_SHA256:
        raise AssertionError("Stage2 preregister binding changed")
    if lock["stage2_results"]["sha256"] != sha256_file(result_path):
        raise AssertionError("Stage2 result changed")
    if bool(lock["promoted"]) != bool(result["promoted"]):
        raise AssertionError("Stage2 promotion changed")
    return lock, result


def _final_skip(out_dir: Path, stage2_lock: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "stage2_promotion_lock": describe_file(out_dir / "stage2_promotion_lock.json"),
        "performed": False,
        "reason": "strict Stage2 group and mixed-aggregate gates did not both pass",
        "passed_groups": list(stage2_lock["passed_groups"]),
        "2025_weather_read": False,
        "sample_submission_read": False,
        "submission_created": False,
        "leaderboard_score_claim": False,
    }
    bayes._write_json(out_dir / "final_results.json", result)
    return result


def _final(
    *,
    raw_dir: Path,
    artifact_root: Path,
    out_dir: Path,
    raw_contract: Mapping[str, Any],
) -> dict[str, Any]:
    if (out_dir / "final_results.json").exists():
        raise FileExistsError(out_dir / "final_results.json")
    stage2_lock, _ = _load_stage2_lock(out_dir)
    if not bool(stage2_lock["promoted"]):
        return _final_skip(out_dir, stage2_lock)

    passed_groups = list(stage2_lock["passed_groups"])
    train_features, train_weather_evidence = raw_protocol._read_weather_features(
        raw_dir, raw_contract, period="train_all"
    )
    labels, label_evidence = raw_protocol._read_full_labels(
        raw_dir, raw_contract, passed_groups
    )
    test_features, test_weather_evidence = raw_protocol._read_weather_features(
        raw_dir, raw_contract, period="test"
    )
    baseline_spec = raw_contract["baseline_contract"]["final_after_all_group_promotion_only"]
    baseline_path = artifact_root / "final_cf_fix/predictions/corrected_v3_test.parquet"
    raw_protocol._assert_file_identity(baseline_path, baseline_spec, "final baseline")
    baseline = bayes._read_prediction(
        baseline_path, YEAR_2025, required_columns=TARGET_COLS
    )

    prediction = baseline.copy()
    raw_prediction = pd.DataFrame(index=YEAR_2025)
    models: dict[str, RawTemporalMultiOutputRegressor] = {}
    training: dict[str, Any] = {}
    for group in passed_groups:
        candidate_key = str(stage2_lock["locked_candidates"][group])
        estimator_id, _ = parse_candidate_key(candidate_key)
        fit_index = TRAIN_ALL if group in TARGET_COLS[:2] else YEAR_2023.append(YEAR_2024)
        split = assert_fit_before_apply(fit_index, YEAR_2025)
        model = RawTemporalMultiOutputRegressor(estimator_id).fit(
            train_features.loc[fit_index], labels.loc[fit_index, group] / CAPACITY_KWH[group]
        )
        raw = model.predict(test_features)
        candidate = _fixed_candidate(
            baseline[group], raw, group=group, candidate_key=candidate_key
        )
        models[group] = model
        raw_prediction[group] = raw
        prediction[group] = candidate
        training[group] = {
            "locked_candidate": candidate_key,
            "split": split,
            "model": model.metadata(),
        }
    model_path = out_dir / "models/final_passed_groups.joblib"
    raw_path = out_dir / "predictions/final_raw_cf_2025.parquet"
    prediction_path = out_dir / "predictions/raw_temporal_multioutput_2025.parquet"
    bayes._atomic_joblib(models, model_path)
    bayes._atomic_parquet(raw_prediction, raw_path)
    bayes._atomic_parquet(prediction, prediction_path)
    loaded: dict[str, RawTemporalMultiOutputRegressor] = joblib.load(model_path)
    reload_checks: dict[str, Any] = {}
    for group in passed_groups:
        repeated = loaded[group].predict(test_features)
        exact = np.array_equal(repeated.to_numpy(), raw_prediction[group].to_numpy())
        if not exact:
            raise AssertionError(f"final {group} reload changed prediction")
        reload_checks[group] = {"raw_prediction_bit_exact": True}

    sample_spec = raw_contract["full_input_identities_after_promotion_only"]["sample_submission"]
    sample_path = raw_dir / "sample_submission.csv"
    raw_protocol._assert_file_identity(sample_path, sample_spec, "sample submission")
    sample = pd.read_csv(
        sample_path,
        encoding="utf-8-sig",
        dtype={"forecast_id": "string", "forecast_kst_dtm": "string"},
    )
    if tuple(sample.columns) != ("forecast_id", "forecast_kst_dtm", *TARGET_COLS):
        raise AssertionError("sample submission schema changed")
    sample_index = pd.DatetimeIndex(
        pd.to_datetime(sample["forecast_kst_dtm"], errors="raise"), name=TIME_COL
    )
    if not sample_index.equals(YEAR_2025):
        raise AssertionError("sample submission timestamps changed")
    submission = sample.copy()
    for group in TARGET_COLS:
        submission[group] = prediction[group].to_numpy(dtype=float)
    csv_path = out_dir / "raw_temporal_multioutput_2025.csv"
    bayes._atomic_csv(submission, csv_path)
    verification = bayes._verify_submission(csv_path, sample, prediction)
    result = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "stage2_promotion_lock": describe_file(out_dir / "stage2_promotion_lock.json"),
        "performed": True,
        "passed_groups": passed_groups,
        "training": training,
        "train_weather_evidence": train_weather_evidence,
        "label_evidence": label_evidence,
        "test_weather_evidence": test_weather_evidence,
        "reload_checks": reload_checks,
        "outputs": [
            describe_file(model_path),
            describe_file(raw_path),
            describe_file(prediction_path),
            describe_file(csv_path),
        ],
        "submission_verification": verification,
        "2025_weather_read": True,
        "sample_submission_read": True,
        "submission_created": True,
        "leaderboard_score_claim": False,
    }
    bayes._write_json(out_dir / "final_results.json", result)
    return result


def _write_manifest(
    *,
    raw_dir: Path,
    out_dir: Path,
    preregister_path: Path,
    raw_dependency_path: Path,
) -> dict[str, Any]:
    path = out_dir / "manifest.json"
    if path.exists():
        raise FileExistsError(path)
    stage1_lock, stage1_result = _load_stage1_lock(out_dir)
    stage2_lock, stage2_result = _load_stage2_lock(out_dir)
    final_result = json.loads((out_dir / "final_results.json").read_text(encoding="utf-8"))
    outputs = sorted(
        item
        for item in out_dir.rglob("*")
        if item.is_file() and item != path and ".tmp-" not in item.name
    )
    payload = {
        "schema_version": 1,
        "artifact_type": "raw_temporal_multioutput_strict_forward_v1",
        "created_utc": utc_now(),
        "command": [sys.executable, *sys.argv],
        "runtime": {"packages": package_versions(), "git": git_state(PROJECT_DIR)},
        "preregister_sha256": PREREGISTER_SHA256,
        "feasibility_sha256": FEASIBILITY_SHA256,
        "contracts": {
            "raw_node_channels": EXPECTED_RAW_CHANNELS,
            "dct_coefficients_per_channel": DCT_COEFFICIENTS,
            "run_input_dimension": EXPECTED_INPUT_DIMENSION,
            "one_complete_run_to_24_outputs": True,
            "calendar_issuance_baseline_target_scada_public_scale_model_features": False,
            "fit_before_apply": True,
            "application_labels_read_after_model_prediction_hash_locks": True,
            "passed_groups_only_enter_2024": True,
            "mixed_aggregate_all_slices_positive_required_for_2025": True,
            "leaderboard_score_claim": False,
        },
        "selection": {
            "stage1_locked_candidates": stage1_lock["locked_candidates"],
            "stage1_passed_groups": stage1_lock["passed_groups"],
            "stage2_group_gate_passed": bool(stage2_lock["group_gate_passed"]),
            "stage2_mixed_aggregate_gate_passed": bool(
                stage2_lock["mixed_aggregate_gate_passed"]
            ),
            "promoted": bool(stage2_lock["promoted"]),
        },
        "read_flags": {
            "2024_weather_read": bool(stage2_result.get("2024_weather_read", False)),
            "2024_label_read": bool(stage2_result.get("2024_label_read", False)),
            "2025_weather_read": bool(final_result.get("2025_weather_read", False)),
            "sample_submission_read": bool(
                final_result.get("sample_submission_read", False)
            ),
            "csv_created": bool(final_result.get("submission_created", False)),
        },
        "source_provenance": _snapshot_named(
            _source_paths(preregister_path, raw_dependency_path)
        ),
        "inputs_read_conditionally": {
            "raw_root": str(raw_dir.resolve()),
            "stage1_uses_physical_prefix_only": True,
            "whole_2024_train_files_opened": bool(
                stage2_result.get("2024_weather_read", False)
            ),
            "test_files_opened": bool(final_result.get("2025_weather_read", False)),
        },
        "tests": {
            "focused_sources": [
                describe_file(PROJECT_DIR / "tests/test_raw_temporal_multioutput.py"),
                describe_file(
                    PROJECT_DIR / "tests/test_raw_temporal_multioutput_runner.py"
                ),
            ],
            "focused_test_count_prelaunch": 19,
            "focused_prelaunch_status": "pass",
            "full_suite_test_count_prelaunch": 236,
            "full_suite_prelaunch_status": "pass",
            "commands": [
                ".venv/Scripts/python.exe -m unittest tests.test_raw_temporal_multioutput -v",
                ".venv/Scripts/python.exe -m unittest discover -s tests -q",
            ],
        },
        "stage1_results_sha256": describe_file(out_dir / "stage1_results.json")["sha256"],
        "outputs": [describe_file(item) for item in outputs],
    }
    bayes._write_json(path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    preregister_path = args.preregister.resolve()
    preregister, raw_contract, raw_dependency_path = _verify_preregister(preregister_path)
    raw_dir = args.raw_dir.resolve()
    artifact_root = args.artifact_root.resolve()
    out_dir = args.out_dir.resolve()
    if args.stage in ("stage1", "all"):
        _stage1(
            raw_dir=raw_dir,
            artifact_root=artifact_root,
            out_dir=out_dir,
            preregister_path=preregister_path,
            preregister=preregister,
            raw_contract=raw_contract,
            raw_dependency_path=raw_dependency_path,
        )
    if args.stage in ("stage2", "all"):
        _stage2(
            raw_dir=raw_dir,
            artifact_root=artifact_root,
            out_dir=out_dir,
            raw_contract=raw_contract,
        )
    if args.stage in ("final", "all"):
        _final(
            raw_dir=raw_dir,
            artifact_root=artifact_root,
            out_dir=out_dir,
            raw_contract=raw_contract,
        )
    if args.stage in ("manifest", "all"):
        _write_manifest(
            raw_dir=raw_dir,
            out_dir=out_dir,
            preregister_path=preregister_path,
            raw_dependency_path=raw_dependency_path,
        )


if __name__ == "__main__":
    main()

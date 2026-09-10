"""Strict Stage1 for the raw-grid multitask geometry-attention Conv1d model."""

from __future__ import annotations

import argparse
import json
from importlib import metadata
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts import run_ficr_bayes_decision_strict as bayes
from scripts import run_raw_grid_wind_lgb as raw_protocol
from scripts import run_raw_temporal_kernel as kernel_protocol
from scripts import run_shared_q07_multiseed as shared
from scripts import run_weather_quantile_bayes as weather_protocol
from src.manifest import describe_file, git_state, package_versions, sha256_file, utc_now
from src.metric import CAPACITY_KWH, TARGET_COLS
from src.raw_spatiotemporal_attention import (
    BATCH_SIZE,
    BLEND_WEIGHTS,
    CANDIDATE_KEYS,
    EPOCHS,
    MODEL_ID,
    RawSpatiotemporalAttentionRegressor,
    assert_fit_before_apply,
    candidate_frame,
    select_group_candidate,
)


PREREGISTER_SHA256 = "93f6b3f0aa19407c3d5bdc14c5529fa29e7f32b03b23dc58f87050d50f5a142d"
FEASIBILITY_SHA256 = "e6fa78a3c93a9baf7ae610081809d5d2b76a539fa1999bb0b0589ba43f6d9cc3"
RAW_DEPENDENCY_SHA256 = "06b56426347f64d91f1a69e1c56083014f0dd003ee8186fa98ffe97ad3b61d01"
TIME_COL = "forecast_kst_dtm"
YEAR_2022 = raw_protocol.YEAR_2022
YEAR_2023 = raw_protocol.YEAR_2023
PRE2024 = raw_protocol.PRE2024
G3_H1 = weather_protocol._year_segments(2023)["H1"]
G3_H2 = weather_protocol._year_segments(2023)["H2"]
STAGE1_REQUIRED: Mapping[str, tuple[str, ...]] = {
    TARGET_COLS[0]: ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4"),
    TARGET_COLS[1]: ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4"),
    TARGET_COLS[2]: ("full", "Q3", "Q4"),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("stage1", "manifest", "all"), required=True)
    parser.add_argument(
        "--raw-dir", type=Path, default=Path(r"data/local/open")
    )
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/postgate/raw_spatiotemporal_attention_strict_v1"),
    )
    parser.add_argument(
        "--preregister",
        type=Path,
        default=Path("configs/raw_spatiotemporal_attention_preregister_v1.json"),
    )
    return parser.parse_args(argv)


def _verify_file_and_sidecar(path: Path, expected: str) -> None:
    if sha256_file(path) != expected:
        raise AssertionError(f"content hash changed: {path}")
    expected_text = f"{expected}  {path.name}\n"
    if path.with_suffix(".sha256").read_text(encoding="utf-8") != expected_text:
        raise AssertionError(f"SHA sidecar changed: {path}")


def _verify_preregister(path: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    _verify_file_and_sidecar(path, PREREGISTER_SHA256)
    preregister = json.loads(path.read_text(encoding="utf-8"))
    if preregister["experiment_id"] != "raw_spatiotemporal_attention_strict_forward_v1":
        raise AssertionError("experiment id changed")
    training = preregister["training"]
    if int(training["epochs"]) != EPOCHS or int(training["batch_size_runs"]) != BATCH_SIZE:
        raise AssertionError("training schedule changed")
    candidates = preregister["candidate_family"]
    if tuple(map(float, candidates["fixed_blend_weights"])) != BLEND_WEIGHTS:
        raise AssertionError("blend weights changed")
    if tuple(candidates["candidate_order"]) != CANDIDATE_KEYS:
        raise AssertionError("candidate order changed")
    feasibility = PROJECT_DIR / "configs/raw_spatiotemporal_attention_feasibility_v1.json"
    _verify_file_and_sidecar(feasibility, FEASIBILITY_SHA256)
    feasibility_payload = json.loads(feasibility.read_text(encoding="utf-8"))
    if feasibility_payload["preregister_sha256"] != PREREGISTER_SHA256:
        raise AssertionError("feasibility binding changed")
    dependency = PROJECT_DIR / preregister["dependency_contract"]["raw_grid_preregister"]["path"]
    if sha256_file(dependency) != RAW_DEPENDENCY_SHA256:
        raise AssertionError("raw dependency changed")
    raw_contract = json.loads(dependency.read_text(encoding="utf-8"))
    return preregister, raw_contract, dependency


def _source_paths(preregister: Path, dependency: Path) -> dict[str, Path]:
    inherited = kernel_protocol._source_paths(
        PROJECT_DIR / "configs/raw_temporal_kernel_preregister_v1.json", dependency
    )
    inherited = {f"kernel_dependency__{key}": value for key, value in inherited.items()}
    inherited.update(
        {
            "runner": Path(__file__).resolve(),
            "attention_module": PROJECT_DIR / "src/raw_spatiotemporal_attention.py",
            "focused_test": PROJECT_DIR / "tests/test_raw_spatiotemporal_attention.py",
            "focused_runner_test": PROJECT_DIR / "tests/test_raw_spatiotemporal_attention_runner.py",
            "preregister": preregister.resolve(),
            "preregister_sidecar": preregister.with_suffix(".sha256").resolve(),
            "feasibility": PROJECT_DIR / "configs/raw_spatiotemporal_attention_feasibility_v1.json",
            "feasibility_sidecar": PROJECT_DIR / "configs/raw_spatiotemporal_attention_feasibility_v1.sha256",
            "raw_grid_preregister_dependency": dependency.resolve(),
            "raw_grid_preregister_sidecar": dependency.with_suffix(".sha256").resolve(),
        }
    )
    return inherited


def _snapshot(paths: Mapping[str, Path]) -> dict[str, Any]:
    return {name: shared._snapshot_file(path) for name, path in paths.items()}


def _assert_snapshot_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    if json.dumps(left, sort_keys=True, default=str) != json.dumps(
        right, sort_keys=True, default=str
    ):
        raise AssertionError("transitive source snapshot changed during Stage1")


def _segments(group: str) -> dict[str, pd.DatetimeIndex]:
    year = weather_protocol._year_segments(2023)
    if group == TARGET_COLS[2]:
        return {"full": year["H2"], "Q3": year["Q3"], "Q4": year["Q4"]}
    return {name: year[name] for name in STAGE1_REQUIRED[group]}


def _capacity_factors(labels: pd.DataFrame) -> pd.DataFrame:
    result = labels.copy()
    for group in result.columns:
        result[group] = result[group] / CAPACITY_KWH[group]
    return result


def _save_reload_joint(
    *,
    out_dir: Path,
    fit_id: str,
    model: RawSpatiotemporalAttentionRegressor,
    prediction: pd.DataFrame,
    application_features: pd.DataFrame,
    candidates: Mapping[str, pd.DataFrame],
    baselines: Mapping[str, pd.Series],
) -> tuple[list[Path], dict[str, Any]]:
    model_path = out_dir / "models" / f"stage1__{fit_id}.joblib"
    prediction_path = out_dir / "predictions" / f"stage1__{fit_id}__raw_cf.parquet"
    bayes._atomic_joblib(model, model_path)
    bayes._atomic_parquet(prediction, prediction_path)
    paths = [model_path, prediction_path]
    for group, frame in candidates.items():
        candidate_path = out_dir / "predictions" / f"stage1__{fit_id}__{group}__candidates.parquet"
        baseline_path = out_dir / "predictions" / f"stage1__{fit_id}__{group}__baseline.parquet"
        bayes._atomic_parquet(frame, candidate_path)
        bayes._atomic_parquet(baselines[group].rename(group).to_frame(), baseline_path)
        paths.extend((candidate_path, baseline_path))
    loaded: RawSpatiotemporalAttentionRegressor = joblib.load(model_path)
    repeated = loaded.predict(application_features)
    if not np.array_equal(repeated.to_numpy(), prediction.to_numpy()):
        raise AssertionError(f"{fit_id} loaded neural prediction changed")
    if not np.array_equal(pd.read_parquet(prediction_path).to_numpy(), prediction.to_numpy()):
        raise AssertionError(f"{fit_id} raw prediction parquet changed")
    for group, frame in candidates.items():
        stored = pd.read_parquet(
            out_dir / "predictions" / f"stage1__{fit_id}__{group}__candidates.parquet"
        )
        if not np.array_equal(stored.to_numpy(), frame.to_numpy()):
            raise AssertionError(f"{fit_id}/{group} candidate parquet changed")
    return paths, {
        "prediction_reload_bit_exact": True,
        "prediction_and_candidate_parquet_roundtrip_bit_exact": True,
        "metadata": loaded.metadata(),
    }


def _fit_joint(
    *,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    catalogs: Mapping[str, Sequence[Mapping[str, Any]]],
    fit_index: pd.DatetimeIndex,
    application_index: pd.DatetimeIndex,
    application_groups: Sequence[str],
    baseline: pd.DataFrame,
) -> tuple[RawSpatiotemporalAttentionRegressor, pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    split = assert_fit_before_apply(fit_index, application_index)
    print(
        f"fit {MODEL_ID}: {fit_index.min()}..{fit_index.max()} -> {application_index.min()} "
        f"for {list(application_groups)}",
        flush=True,
    )
    model = RawSpatiotemporalAttentionRegressor(catalogs).fit(
        features.loc[fit_index], _capacity_factors(labels.loc[fit_index])
    )
    raw_prediction = model.predict(features.loc[application_index])
    candidates = {
        group: candidate_frame(
            baseline.loc[application_index, group],
            raw_prediction[group],
            capacity_kwh=CAPACITY_KWH[group],
        )
        for group in application_groups
    }
    return model, raw_prediction, candidates, {"split": split, "model": model.metadata()}


def _stage1(
    *,
    raw_dir: Path,
    artifact_root: Path,
    out_dir: Path,
    preregister_path: Path,
    raw_contract: Mapping[str, Any],
    dependency_path: Path,
) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"Stage1 requires a new output directory: {out_dir}")
    out_dir.mkdir(parents=True)
    for source in (
        preregister_path,
        preregister_path.with_suffix(".sha256"),
        PROJECT_DIR / "configs/raw_spatiotemporal_attention_feasibility_v1.json",
        PROJECT_DIR / "configs/raw_spatiotemporal_attention_feasibility_v1.sha256",
    ):
        bayes._copy_exclusive(source, out_dir / source.name)
    source_paths = _source_paths(preregister_path, dependency_path)
    sources_before = _snapshot(source_paths)
    prefix = raw_contract["physical_stage1_inputs"]
    prefixes_before = {
        "labels_g12_fit": raw_protocol._prefix_snapshot(
            raw_dir / "train/train_labels.csv", prefix["labels_g12_fit_prefix"]
        ),
        "labels_g3_fit": raw_protocol._prefix_snapshot(
            raw_dir / "train/train_labels.csv", prefix["labels_g3_fit_prefix"]
        ),
        "labels_score": raw_protocol._prefix_snapshot(
            raw_dir / "train/train_labels.csv", prefix["labels_stage1_score_prefix_after_lock"]
        ),
        "ldaps": raw_protocol._prefix_snapshot(
            raw_dir / "train/ldaps_train.csv", prefix["ldaps_weather_prefix"]
        ),
        "gfs": raw_protocol._prefix_snapshot(
            raw_dir / "train/gfs_train.csv", prefix["gfs_weather_prefix"]
        ),
    }
    features, weather_evidence = raw_protocol._read_weather_features(
        raw_dir, raw_contract, period="stage1"
    )
    baseline = raw_protocol._stage1_baseline(artifact_root, raw_contract)
    catalogs = raw_contract["grid_catalog"]

    labels_g12, labels_g12_evidence = raw_protocol._read_label_prefix(
        raw_dir, prefix["labels_g12_fit_prefix"], TARGET_COLS[:2]
    )
    if not labels_g12.index.equals(YEAR_2022):
        raise AssertionError("G1/G2 physical fit prefix changed")
    model_g12, prediction_g12, candidates_g12, training_g12 = _fit_joint(
        features=features,
        labels=labels_g12,
        catalogs=catalogs,
        fit_index=YEAR_2022,
        application_index=YEAR_2023,
        application_groups=TARGET_COLS[:2],
        baseline=baseline,
    )
    paths_g12, replay_g12 = _save_reload_joint(
        out_dir=out_dir,
        fit_id="g12_2022",
        model=model_g12,
        prediction=prediction_g12,
        application_features=features.loc[YEAR_2023],
        candidates=candidates_g12,
        baselines={group: baseline.loc[YEAR_2023, group] for group in TARGET_COLS[:2]},
    )
    record_g12 = out_dir / "stage1_g12_prescore_record.json"
    bayes._write_json(
        record_g12,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "fit_label_evidence": labels_g12_evidence,
            "training": training_g12,
            "reload": replay_g12,
            "outputs": [describe_file(path) for path in paths_g12],
            "created_before_any_2023_application_target_value": True,
        },
    )
    lock_g12 = out_dir / "stage1_g12_prescore_lock.json"
    bayes._write_json(
        lock_g12,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "record": describe_file(record_g12),
            "candidate_artifacts": [describe_file(path) for path in paths_g12],
            "model_preprocessing_predictions_candidates_frozen": True,
        },
    )

    labels_g3, labels_g3_evidence = raw_protocol._read_label_prefix(
        raw_dir, prefix["labels_g3_fit_prefix"], TARGET_COLS
    )
    expected_g3_prefix = YEAR_2022.append(G3_H1)
    if not labels_g3.index.equals(expected_g3_prefix):
        raise AssertionError("G3 physical fit prefix changed")
    model_g3, prediction_g3, candidates_g3, training_g3 = _fit_joint(
        features=features,
        labels=labels_g3,
        catalogs=catalogs,
        fit_index=expected_g3_prefix,
        application_index=G3_H2,
        application_groups=(TARGET_COLS[2],),
        baseline=baseline,
    )
    paths_g3, replay_g3 = _save_reload_joint(
        out_dir=out_dir,
        fit_id="g3_through_2023h1",
        model=model_g3,
        prediction=prediction_g3,
        application_features=features.loc[G3_H2],
        candidates=candidates_g3,
        baselines={TARGET_COLS[2]: baseline.loc[G3_H2, TARGET_COLS[2]]},
    )
    record_g3 = out_dir / "stage1_g3_prescore_record.json"
    bayes._write_json(
        record_g3,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "g12_lock": describe_file(lock_g12),
            "fit_label_evidence": labels_g3_evidence,
            "training": training_g3,
            "reload": replay_g3,
            "outputs": [describe_file(path) for path in paths_g3],
            "created_before_any_g3_h2_application_target_value": True,
        },
    )
    lock_g3 = out_dir / "stage1_g3_prescore_lock.json"
    bayes._write_json(
        lock_g3,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "record": describe_file(record_g3),
            "g12_lock": describe_file(lock_g12),
            "candidate_artifacts": [describe_file(path) for path in paths_g3],
            "model_preprocessing_predictions_candidates_frozen": True,
        },
    )

    all_paths = [*paths_g12, *paths_g3]
    global_record = out_dir / "stage1_global_prescore_record.json"
    bayes._write_json(
        global_record,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "g12_lock": describe_file(lock_g12),
            "g3_lock": describe_file(lock_g3),
            "weather_evidence": weather_evidence,
            "sources_before": sources_before,
            "input_prefixes_before": prefixes_before,
            "outputs": [describe_file(path) for path in all_paths],
            "validation_target_columns_materialized": [],
            "created_before_any_stage1_application_target_value": True,
            "torch_runtime": {
                "version": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            },
            "2024_weather_read": False,
            "2024_label_read": False,
            "2025_read": False,
        },
    )
    global_lock = out_dir / "stage1_global_prescore_lock.json"
    bayes._write_json(
        global_lock,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "record": describe_file(global_record),
            "candidate_artifacts": [describe_file(path) for path in all_paths],
            "all_models_preprocessing_predictions_candidates_frozen": True,
            "created_before_any_stage1_application_target_value": True,
        },
    )

    score_labels, score_label_evidence = raw_protocol._read_label_prefix(
        raw_dir, prefix["labels_stage1_score_prefix_after_lock"], TARGET_COLS
    )
    if not score_labels.index.equals(PRE2024):
        raise AssertionError("Stage1 score prefix changed")
    group_results: dict[str, Any] = {}
    locked: dict[str, str | None] = {}
    passed: list[str] = []
    for group in TARGET_COLS:
        application = YEAR_2023 if group in TARGET_COLS[:2] else G3_H2
        group_candidates = candidates_g12[group] if group in TARGET_COLS[:2] else candidates_g3[group]
        comparisons = {
            key: bayes._comparison(
                score_labels.loc[application, group],
                baseline.loc[application, group],
                group_candidates[key],
                group,
                _segments(group),
            )
            for key in CANDIDATE_KEYS
        }
        selected, audit = select_group_candidate(comparisons, STAGE1_REQUIRED[group])
        locked[group] = selected
        if selected is not None:
            passed.append(group)
        group_results[group] = {
            "comparisons": comparisons,
            "selection": audit,
            "locked_candidate": selected if selected is not None else "identity",
            "passed": selected is not None,
        }
    _assert_snapshot_equal(sources_before, _snapshot(source_paths))
    result = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "global_prescore_lock": describe_file(global_lock),
        "score_label_evidence_after_global_prescore_lock": score_label_evidence,
        "training": {"g12": training_g12, "g3": training_g3},
        "group_results": group_results,
        "locked_candidates": locked,
        "passed_groups": passed,
        "stage2_unlocked": bool(passed),
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
        "global_prescore_lock": describe_file(global_lock),
        "stage1_results": describe_file(result_path),
        "locked_candidates": locked,
        "passed_groups": passed,
        "stage2_unlocked": bool(passed),
        "no_2024_or_2025_read": True,
    }
    bayes._write_json(out_dir / "stage1_promotion_lock.json", lock)
    print(f"Stage1 passed groups: {passed}", flush=True)
    return lock


def _manifest(
    *, out_dir: Path, preregister_path: Path, dependency_path: Path, raw_dir: Path
) -> dict[str, Any]:
    path = out_dir / "manifest.json"
    if path.exists():
        raise FileExistsError(path)
    result = json.loads((out_dir / "stage1_results.json").read_text(encoding="utf-8"))
    outputs = sorted(
        item for item in out_dir.rglob("*") if item.is_file() and item != path and ".tmp-" not in item.name
    )
    payload = {
        "schema_version": 1,
        "artifact_type": "raw_spatiotemporal_attention_strict_forward_v1_stage1",
        "created_utc": utc_now(),
        "command": [sys.executable, *sys.argv],
        "runtime": {
            "packages": {**package_versions(), "torch": metadata.version("torch")},
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "git": git_state(PROJECT_DIR),
        },
        "preregister_sha256": PREREGISTER_SHA256,
        "feasibility_sha256": FEASIBILITY_SHA256,
        "contracts": {
            "fold_causal_multitask": True,
            "shared_node_encoder": True,
            "source_specific_geometry_attention": True,
            "residual_temporal_conv1d": True,
            "group_specific_direct_cf_heads": True,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "blend_weights": list(BLEND_WEIGHTS),
            "all_registered_slices_positive_required": True,
            "2024_or_2025_input_used": False,
            "leaderboard_score_claim": False,
        },
        "selection": {
            "locked_candidates": result["locked_candidates"],
            "passed_groups": result["passed_groups"],
            "stage2_unlocked": result["stage2_unlocked"],
        },
        "read_flags": {
            "2024_weather_read": False,
            "2024_label_read": False,
            "2025_weather_read": False,
            "sample_submission_read": False,
            "csv_created": False,
        },
        "source_provenance": _snapshot(_source_paths(preregister_path, dependency_path)),
        "inputs": {"raw_root": str(raw_dir.resolve()), "physical_prefix_only": True},
        "tests": {
            "focused_command": ".venv/Scripts/python.exe -B -m unittest tests.test_raw_spatiotemporal_attention -v",
            "focused_count": 8,
            "focused_status": "pass",
        },
        "outputs": [describe_file(item) for item in outputs],
    }
    bayes._write_json(path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    preregister_path = args.preregister.resolve()
    _, raw_contract, dependency_path = _verify_preregister(preregister_path)
    raw_dir = args.raw_dir.resolve()
    artifact_root = args.artifact_root.resolve()
    out_dir = args.out_dir.resolve()
    if args.stage in ("stage1", "all"):
        _stage1(
            raw_dir=raw_dir,
            artifact_root=artifact_root,
            out_dir=out_dir,
            preregister_path=preregister_path,
            raw_contract=raw_contract,
            dependency_path=dependency_path,
        )
    if args.stage in ("manifest", "all"):
        _manifest(
            out_dir=out_dir,
            preregister_path=preregister_path,
            dependency_path=dependency_path,
            raw_dir=raw_dir,
        )


if __name__ == "__main__":
    main()

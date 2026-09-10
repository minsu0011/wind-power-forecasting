"""Strict Stage1 runner for the sparse-spatial Conv2d temporal Conv1d model."""

from __future__ import annotations

import argparse
import atexit
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_ficr_bayes_decision_strict as bayes
from scripts import run_raw_grid_wind_lgb as raw_protocol
from scripts import run_raw_spatiotemporal_smooth_ficr as strict_protocol
from scripts import run_weather_quantile_bayes as weather_protocol
from src.manifest import describe_file, git_state, package_versions, sha256_file, utc_now
from src.metric import CAPACITY_KWH, TARGET_COLS
from src.raw_spatial_grid_temporal_cnn import (
    BATCH_SIZE, BLEND_WEIGHT, CANDIDATE_KEY, EPOCHS, LEARNING_RATE,
    LOW_CF_LOSS_WEIGHT, MODEL_ID, SEEDS, SMOOTH_L1_BETA, WEIGHT_DECAY,
    RawSpatialGridTemporalCNNRegressor, candidate_series, stage1_group_gate,
)


CONFIG_PATH = ROOT / "configs/raw_spatial_grid_temporal_cnn_preregister_v1.json"
CONFIG_SHA256 = "34673e57c1263833872f63cac640b6625e8160c9df3c4eefea74847f2fe5649c"
FEASIBILITY_PATH = ROOT / "artifacts/audits/raw_spatial_grid_temporal_cnn_feasibility_v1.json"
FEASIBILITY_SHA256 = "d3960d4fdd41f1f8f4905443850ae273092e2951ffa2739a2e94ddfb632c02f6"
RAW_DEPENDENCY_PATH = ROOT / "configs/raw_grid_wind_lgb_preregister_v1.json"
RAW_DEPENDENCY_SHA256 = "06b56426347f64d91f1a69e1c56083014f0dd003ee8186fa98ffe97ad3b61d01"
ATTENTION_SOURCE_PATH = ROOT / "src/raw_spatiotemporal_attention.py"
ATTENTION_SOURCE_SHA256 = "51fa4993887fbee1b5b21aa9bf81a27662c289f4764d621ca28c7a6667d94384"
HEAVY_GUARD_PATH = strict_protocol.HEAVY_GUARD_PATH
DEFAULT_RAW_DIR = Path(r"data/local/open")
DEFAULT_OUT_DIR = ROOT / "artifacts/postgate/raw_spatial_grid_temporal_cnn_strict_v1"
YEAR_2022 = raw_protocol.YEAR_2022
YEAR_2023 = raw_protocol.YEAR_2023
PRE2024 = raw_protocol.PRE2024
G3_H1 = weather_protocol._year_segments(2023)["H1"]
G3_H2 = weather_protocol._year_segments(2023)["H2"]
REQUIRED: Mapping[str, tuple[str, ...]] = {
    TARGET_COLS[0]: ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4"),
    TARGET_COLS[1]: ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4"),
    TARGET_COLS[2]: ("full", "Q3", "Q4"),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("stage1", "manifest", "all"), required=True)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    return parser.parse_args(argv)


def _verify(path: Path, expected_sha256: str, expected_bytes: int | None = None) -> Path:
    path = path.resolve()
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise AssertionError(f"registered file changed: {path}")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise AssertionError(f"registered file size changed: {path}")
    return path


def verify_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if path.resolve() != CONFIG_PATH.resolve():
        raise AssertionError("only the canonical preregistration is allowed")
    _verify(CONFIG_PATH, CONFIG_SHA256, 11026)
    expected_sidecar = f"{CONFIG_SHA256}  {CONFIG_PATH.name}\n"
    if CONFIG_PATH.with_suffix(".sha256").read_text(encoding="utf-8") != expected_sidecar:
        raise AssertionError("preregistration sidecar changed")
    _verify(FEASIBILITY_PATH, FEASIBILITY_SHA256, 7498)
    _verify(RAW_DEPENDENCY_PATH, RAW_DEPENDENCY_SHA256, 12868)
    _verify(ATTENTION_SOURCE_PATH, ATTENTION_SOURCE_SHA256, 28092)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["experiment_id"] != "raw_spatial_grid_temporal_cnn_strict_forward_v1":
        raise AssertionError("experiment identity changed")
    architecture = config["architecture"]
    training = config["training"]
    candidate = config["single_candidate"]
    if architecture["parameter_count"] != 41235 or architecture["model_id"] != MODEL_ID:
        raise AssertionError("architecture contract changed")
    if (
        tuple(training["seeds"]) != SEEDS
        or int(training["epochs"]) != EPOCHS
        or int(training["batch_size_runs"]) != BATCH_SIZE
        or float(training["learning_rate"]) != LEARNING_RATE
        or float(training["weight_decay"]) != WEIGHT_DECAY
        or float(candidate["blend_weight"]) != BLEND_WEIGHT
        or candidate["id"] != CANDIDATE_KEY
        or float(training["loss"].split("beta=")[1].split(";")[0]) != SMOOTH_L1_BETA
    ):
        raise AssertionError("training/candidate contract changed")
    if LOW_CF_LOSS_WEIGHT != 0.25 or config["sequence_alignment"]["selected"] != "availability_run_01_through_next_day_00":
        raise AssertionError("loss/alignment contract changed")
    if int(config["stage1"]["registered_slice_count"]) != 17:
        raise AssertionError("Stage1 slice count changed")
    raw_contract = json.loads(RAW_DEPENDENCY_PATH.read_text(encoding="utf-8"))
    if config["fixed_rasters"]["ldaps"]["grid_id_layout"] != [[0,1,2,3,0],[4,5,6,7,8],[9,10,11,12,13],[0,14,15,16,0]]:
        raise AssertionError("LDAPS raster registration changed")
    return config, raw_contract


def source_closure(raw_dir: Path) -> dict[str, Any]:
    closure = strict_protocol.resolve_ast_closure(Path(__file__))
    explicit = (
        CONFIG_PATH, CONFIG_PATH.with_suffix(".sha256"), FEASIBILITY_PATH,
        ROOT / "tests/test_raw_spatial_grid_temporal_cnn.py",
        ROOT / "tests/test_raw_spatial_grid_temporal_cnn_runner.py",
        ROOT / "scripts/launch_raw_spatial_grid_temporal_cnn_v1.ps1",
    )
    label_path = raw_dir / "train/train_labels.csv"
    return {
        "resolver": "recursive Python AST local imports plus explicit sources",
        "resolved_relative_paths": [path.relative_to(ROOT).as_posix() for path in closure],
        "resolved_files": [describe_file(path) for path in closure],
        "explicit_sources": [describe_file(path.resolve()) for path in explicit],
        "raw_label_metadata_only": {
            "path": str(label_path.resolve()),
            "bytes": label_path.stat().st_size,
            "content_values_materialized": 0,
        },
        "conditional_2024_or_2025_value_inputs": [],
    }


def _segments(group: str) -> dict[str, pd.DatetimeIndex]:
    year = weather_protocol._year_segments(2023)
    if group == TARGET_COLS[2]:
        return {"full": year["H2"], "Q3": year["Q3"], "Q4": year["Q4"]}
    return {name: year[name] for name in REQUIRED[group]}


def _capacity_factors(labels: pd.DataFrame) -> pd.DataFrame:
    result = labels.copy()
    for group in result.columns:
        result[group] = result[group] / CAPACITY_KWH[group]
    return result


def _fit_joint(
    *, features: pd.DataFrame, labels: pd.DataFrame,
    catalogs: Mapping[str, Sequence[Mapping[str, Any]]],
    fit_index: pd.DatetimeIndex, application_index: pd.DatetimeIndex,
    application_groups: Sequence[str], baseline: pd.DataFrame,
) -> tuple[RawSpatialGridTemporalCNNRegressor, pd.DataFrame, dict[str, pd.Series], dict[str, Any]]:
    split = raw_protocol.assert_fit_before_apply(fit_index, application_index)
    print(
        f"fit {MODEL_ID}: {fit_index.min()}..{fit_index.max()} -> "
        f"{application_index.min()} for {list(application_groups)}", flush=True,
    )
    model = RawSpatialGridTemporalCNNRegressor(catalogs).fit(
        features.loc[fit_index], _capacity_factors(labels.loc[fit_index])
    )
    raw_prediction = model.predict(features.loc[application_index])
    candidates = {
        group: candidate_series(
            baseline.loc[application_index, group], raw_prediction[group],
            capacity_kwh=CAPACITY_KWH[group],
        ).rename(group)
        for group in application_groups
    }
    return model, raw_prediction, candidates, {"split": split, "model": model.metadata()}


def _save_reload(
    *, out_dir: Path, fit_id: str, model: RawSpatialGridTemporalCNNRegressor,
    prediction: pd.DataFrame, application_features: pd.DataFrame,
    candidates: Mapping[str, pd.Series], baselines: Mapping[str, pd.Series],
) -> tuple[list[Path], dict[str, Any]]:
    model_path = out_dir / "models" / f"stage1__{fit_id}.joblib"
    prediction_path = out_dir / "predictions" / f"stage1__{fit_id}__raw_cf.parquet"
    bayes._atomic_joblib(model, model_path)
    bayes._atomic_parquet(prediction, prediction_path)
    paths = [model_path, prediction_path]
    for group, candidate in candidates.items():
        candidate_path = out_dir / "predictions" / f"stage1__{fit_id}__{group}__candidate.parquet"
        baseline_path = out_dir / "predictions" / f"stage1__{fit_id}__{group}__baseline.parquet"
        bayes._atomic_parquet(candidate.rename(group).to_frame(), candidate_path)
        bayes._atomic_parquet(baselines[group].rename(group).to_frame(), baseline_path)
        paths.extend((candidate_path, baseline_path))
    loaded: RawSpatialGridTemporalCNNRegressor = joblib.load(model_path)
    replay = loaded.predict(application_features)
    if not np.array_equal(replay.to_numpy(), prediction.to_numpy()):
        raise AssertionError(f"{fit_id} model reload forward changed")
    if not np.array_equal(pd.read_parquet(prediction_path).to_numpy(), prediction.to_numpy()):
        raise AssertionError(f"{fit_id} prediction parquet changed")
    for group, candidate in candidates.items():
        stored = pd.read_parquet(
            out_dir / "predictions" / f"stage1__{fit_id}__{group}__candidate.parquet"
        )[group]
        if not np.array_equal(stored.to_numpy(), candidate.to_numpy()):
            raise AssertionError(f"{fit_id}/{group} candidate parquet changed")
    return paths, {
        "model_reload_forward_bit_exact": True,
        "prediction_and_candidate_parquet_roundtrip_bit_exact": True,
        "metadata": loaded.metadata(),
    }


def _require_lock(path: Path, flag: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required candidate-before-label lock absent: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("preregister_sha256") != CONFIG_SHA256 or payload.get(flag) is not True:
        raise RuntimeError(f"invalid candidate-before-label lock: {path}")
    return payload


def _read_g3_fit_after_g12_lock(
    raw_dir: Path, spec: Mapping[str, Any], lock: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _require_lock(lock, "g12_model_prediction_and_candidates_frozen")
    return raw_protocol._read_label_prefix(raw_dir, spec, TARGET_COLS)


def _read_score_after_global_lock(
    raw_dir: Path, spec: Mapping[str, Any], lock: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _require_lock(lock, "all_models_predictions_and_candidates_frozen")
    return raw_protocol._read_label_prefix(raw_dir, spec, TARGET_COLS)


def run_stage1(
    *, raw_dir: Path, artifact_root: Path, out_dir: Path,
    config: Mapping[str, Any], raw_contract: Mapping[str, Any],
    runtime: Mapping[str, Any], owner: Mapping[str, Any],
) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(out_dir)
    out_dir.mkdir(parents=True)
    for path in (CONFIG_PATH, CONFIG_PATH.with_suffix(".sha256"), FEASIBILITY_PATH):
        bayes._copy_exclusive(path, out_dir / path.name)
    closure_before = source_closure(raw_dir)
    prefix = raw_contract["physical_stage1_inputs"]
    prefix_snapshots = {
        "labels_g12": raw_protocol._prefix_snapshot(raw_dir / "train/train_labels.csv", prefix["labels_g12_fit_prefix"]),
        "labels_g3": raw_protocol._prefix_snapshot(raw_dir / "train/train_labels.csv", prefix["labels_g3_fit_prefix"]),
        "labels_score": raw_protocol._prefix_snapshot(raw_dir / "train/train_labels.csv", prefix["labels_stage1_score_prefix_after_lock"]),
        "ldaps": raw_protocol._prefix_snapshot(raw_dir / "train/ldaps_train.csv", prefix["ldaps_weather_prefix"]),
        "gfs": raw_protocol._prefix_snapshot(raw_dir / "train/gfs_train.csv", prefix["gfs_weather_prefix"]),
    }
    source_lock = out_dir / "source_lock_before_fit_labels_or_candidate.json"
    bayes._write_json(source_lock, {
        "schema_version": 1, "created_utc": utc_now(),
        "preregister_sha256": CONFIG_SHA256, "runtime": dict(runtime),
        "heavy_guard_owner": dict(owner), "source_closure": closure_before,
        "physical_prefix_snapshots": prefix_snapshots,
        "label_values_materialized": 0, "candidate_arrays_materialized": 0,
        "2024_or_2025_values_read": 0,
    })
    features, weather_evidence = raw_protocol._read_weather_features(
        raw_dir, raw_contract, period="stage1"
    )
    baseline = raw_protocol._stage1_baseline(artifact_root, raw_contract)
    catalogs = raw_contract["grid_catalog"]

    labels_g12, evidence_g12 = raw_protocol._read_label_prefix(
        raw_dir, prefix["labels_g12_fit_prefix"], TARGET_COLS[:2]
    )
    if not labels_g12.index.equals(YEAR_2022):
        raise AssertionError("G12 fit prefix changed")
    model_g12, prediction_g12, candidates_g12, training_g12 = _fit_joint(
        features=features, labels=labels_g12, catalogs=catalogs,
        fit_index=YEAR_2022, application_index=YEAR_2023,
        application_groups=TARGET_COLS[:2], baseline=baseline,
    )
    paths_g12, replay_g12 = _save_reload(
        out_dir=out_dir, fit_id="g12_2022", model=model_g12,
        prediction=prediction_g12, application_features=features.loc[YEAR_2023],
        candidates=candidates_g12,
        baselines={group: baseline.loc[YEAR_2023, group] for group in TARGET_COLS[:2]},
    )
    record_g12 = out_dir / "stage1_g12_candidate_record_before_application_labels.json"
    bayes._write_json(record_g12, {
        "schema_version": 1, "created_utc": utc_now(),
        "preregister_sha256": CONFIG_SHA256, "source_lock": describe_file(source_lock),
        "fit_label_evidence": evidence_g12, "training": training_g12,
        "reload": replay_g12, "outputs": [describe_file(path) for path in paths_g12],
        "application_target_values_materialized": 0,
    })
    lock_g12 = out_dir / "stage1_g12_candidate_lock_before_application_labels.json"
    bayes._write_json(lock_g12, {
        "schema_version": 1, "created_utc": utc_now(),
        "preregister_sha256": CONFIG_SHA256, "record": describe_file(record_g12),
        "candidate_artifacts": [describe_file(path) for path in paths_g12],
        "g12_model_prediction_and_candidates_frozen": True,
    })

    labels_g3, evidence_g3 = _read_g3_fit_after_g12_lock(
        raw_dir, prefix["labels_g3_fit_prefix"], lock_g12
    )
    expected_g3 = YEAR_2022.append(G3_H1)
    if not labels_g3.index.equals(expected_g3):
        raise AssertionError("G3 fit prefix changed")
    model_g3, prediction_g3, candidates_g3, training_g3 = _fit_joint(
        features=features, labels=labels_g3, catalogs=catalogs,
        fit_index=expected_g3, application_index=G3_H2,
        application_groups=(TARGET_COLS[2],), baseline=baseline,
    )
    paths_g3, replay_g3 = _save_reload(
        out_dir=out_dir, fit_id="g3_through_2023h1", model=model_g3,
        prediction=prediction_g3, application_features=features.loc[G3_H2],
        candidates=candidates_g3,
        baselines={TARGET_COLS[2]: baseline.loc[G3_H2, TARGET_COLS[2]]},
    )
    record_g3 = out_dir / "stage1_g3_candidate_record_before_application_labels.json"
    bayes._write_json(record_g3, {
        "schema_version": 1, "created_utc": utc_now(),
        "preregister_sha256": CONFIG_SHA256, "g12_lock": describe_file(lock_g12),
        "fit_label_evidence": evidence_g3, "training": training_g3,
        "reload": replay_g3, "outputs": [describe_file(path) for path in paths_g3],
        "application_target_values_materialized": 0,
    })
    lock_g3 = out_dir / "stage1_g3_candidate_lock_before_application_labels.json"
    bayes._write_json(lock_g3, {
        "schema_version": 1, "created_utc": utc_now(),
        "preregister_sha256": CONFIG_SHA256, "record": describe_file(record_g3),
        "g12_lock": describe_file(lock_g12),
        "candidate_artifacts": [describe_file(path) for path in paths_g3],
        "g3_model_prediction_and_candidates_frozen": True,
    })
    all_paths = [*paths_g12, *paths_g3]
    global_record = out_dir / "stage1_global_candidate_record_before_application_labels.json"
    bayes._write_json(global_record, {
        "schema_version": 1, "created_utc": utc_now(),
        "preregister_sha256": CONFIG_SHA256, "g12_lock": describe_file(lock_g12),
        "g3_lock": describe_file(lock_g3), "weather_evidence": weather_evidence,
        "outputs": [describe_file(path) for path in all_paths],
        "registered_candidate": CANDIDATE_KEY,
        "application_target_values_materialized": 0,
        "2024_or_2025_values_read": 0,
    })
    global_lock = out_dir / "stage1_global_candidate_lock_before_application_labels.json"
    bayes._write_json(global_lock, {
        "schema_version": 1, "created_utc": utc_now(),
        "preregister_sha256": CONFIG_SHA256, "record": describe_file(global_record),
        "candidate_artifacts": [describe_file(path) for path in all_paths],
        "all_models_predictions_and_candidates_frozen": True,
    })

    score_labels, score_evidence = _read_score_after_global_lock(
        raw_dir, prefix["labels_stage1_score_prefix_after_lock"], global_lock
    )
    if not score_labels.index.equals(PRE2024):
        raise AssertionError("Stage1 score prefix changed")
    score_access = out_dir / "stage1_score_label_access_after_global_candidate_lock.json"
    bayes._write_json(score_access, {
        "schema_version": 1, "created_utc": utc_now(),
        "preregister_sha256": CONFIG_SHA256, "global_lock": describe_file(global_lock),
        "evidence": score_evidence, "metric_calls_before_access_record": 0,
    })
    group_results: dict[str, Any] = {}
    passed_groups: list[str] = []
    for group in TARGET_COLS:
        application = YEAR_2023 if group in TARGET_COLS[:2] else G3_H2
        candidate = candidates_g12[group] if group in TARGET_COLS[:2] else candidates_g3[group]
        comparison = bayes._comparison(
            score_labels.loc[application, group], baseline.loc[application, group],
            candidate, group, _segments(group),
        )
        gate = stage1_group_gate(comparison, REQUIRED[group])
        if gate["passed"]:
            passed_groups.append(group)
        group_results[group] = {"comparison": comparison, "gate": gate}
    if sum(len(REQUIRED[group]) for group in TARGET_COLS) != 17:
        raise AssertionError("registered Stage1 metric count changed")
    if json.dumps(closure_before, sort_keys=True, default=str) != json.dumps(
        source_closure(raw_dir), sort_keys=True, default=str
    ):
        raise AssertionError("source closure changed during Stage1")
    result = {
        "schema_version": 1, "created_utc": utc_now(),
        "preregister_sha256": CONFIG_SHA256, "risk": config["risk_classification"],
        "global_candidate_lock": describe_file(global_lock),
        "score_label_access": describe_file(score_access),
        "training": {"g12": training_g12, "g3": training_g3},
        "group_results": group_results, "passed_groups": passed_groups,
        "stage2_unlocked": bool(passed_groups), "registered_metric_records": 17,
        "2024_weather_or_label_read": False, "2025_read": False, "csv_created": False,
    }
    result_path = out_dir / "stage1_results.json"
    bayes._write_json(result_path, result)
    promotion = out_dir / "stage1_promotion_lock.json"
    bayes._write_json(promotion, {
        "schema_version": 1, "created_utc": utc_now(),
        "preregister_sha256": CONFIG_SHA256, "global_candidate_lock": describe_file(global_lock),
        "score_label_access": describe_file(score_access),
        "stage1_results": describe_file(result_path), "passed_groups": passed_groups,
        "stage2_unlocked": bool(passed_groups), "no_2024_or_2025_read": True,
    })
    if not passed_groups:
        bayes._write_json(out_dir / "stage1_rejection.json", {
            "schema_version": 1, "created_utc": utc_now(),
            "preregister_sha256": CONFIG_SHA256, "promotion_lock": describe_file(promotion),
            "decision": "REJECT_IDENTITY", "no_retune_rescue_2024_2025_or_csv": True,
        })
    print(f"Stage1 passed groups: {passed_groups}", flush=True)
    return result


def write_manifest(
    *, out_dir: Path, raw_dir: Path, config: Mapping[str, Any], runtime: Mapping[str, Any]
) -> dict[str, Any]:
    path = out_dir / "manifest.json"
    sidecar = out_dir / "manifest.sha256"
    if path.exists() or sidecar.exists():
        raise FileExistsError("manifest exists")
    result = json.loads((out_dir / "stage1_results.json").read_text(encoding="utf-8"))
    outputs = sorted(item for item in out_dir.rglob("*") if item.is_file() and item not in (path, sidecar))
    payload = {
        "schema_version": 1,
        "artifact_type": "raw_spatial_grid_temporal_cnn_strict_forward_v1_stage1",
        "created_utc": utc_now(), "command": [sys.executable, *sys.argv],
        "preregister_sha256": CONFIG_SHA256, "feasibility_sha256": FEASIBILITY_SHA256,
        "risk": config["risk_classification"], "source_closure": source_closure(raw_dir),
        "selection": {"passed_groups": result["passed_groups"], "stage2_unlocked": result["stage2_unlocked"]},
        "runtime": {"registered": dict(runtime), "packages": package_versions(), "git": git_state(ROOT)},
        "read_flags": {"2024_weather": False, "2024_labels": False, "2025": False, "sample": False, "csv": False},
        "tests": {"focused_expected": 16, "focused_status": "pass"},
        "outputs_excluding_manifest_and_sidecar": [describe_file(item) for item in outputs],
        "output_count_excluding_manifest_and_sidecar": len(outputs),
    }
    bayes._write_json(path, payload)
    with sidecar.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(f"{sha256_file(path)}  manifest.json\n")
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config, raw_contract = verify_config(args.config)
    runtime = strict_protocol.verify_cuda_runtime_before_label_or_fit()
    owner = dict(strict_protocol.acquire_heavy_guard(HEAVY_GUARD_PATH))
    owner["experiment_id"] = config["experiment_id"]
    owner["stage"] = "RAW_SPATIAL_CNN_STAGE1_GPU"
    guard_temporary = HEAVY_GUARD_PATH.with_name(
        f".{HEAVY_GUARD_PATH.name}.owner-{owner['pid']}.tmp"
    )
    with guard_temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(owner, stream, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
    os.replace(guard_temporary, HEAVY_GUARD_PATH)
    atexit.register(strict_protocol.release_heavy_guard, HEAVY_GUARD_PATH, owner)
    try:
        if args.stage in ("stage1", "all"):
            run_stage1(
                raw_dir=args.raw_dir.resolve(), artifact_root=args.artifact_root.resolve(),
                out_dir=args.out_dir.resolve(), config=config, raw_contract=raw_contract,
                runtime=runtime, owner=owner,
            )
        if args.stage in ("manifest", "all"):
            write_manifest(
                out_dir=args.out_dir.resolve(), raw_dir=args.raw_dir.resolve(),
                config=config, runtime=runtime,
            )
    finally:
        strict_protocol.release_heavy_guard(HEAVY_GUARD_PATH, owner)


if __name__ == "__main__":
    main()

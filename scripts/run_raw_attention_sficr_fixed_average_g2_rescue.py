"""Strict posthoc G2-only 2024 transfer for the fixed raw/smooth-FICR average."""

from __future__ import annotations

import argparse
import atexit
import hashlib
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
from scripts import run_raw_spatiotemporal_smooth_ficr as smooth_protocol
from scripts import run_weather_quantile_bayes as weather_protocol
from src.manifest import describe_file, git_state, package_versions, sha256_file, utc_now
from src.metric import CAPACITY_KWH, TARGET_COLS, score_details
from src.raw_spatiotemporal_attention import (
    BATCH_SIZE as RAW_BATCH_SIZE,
    EPOCHS as RAW_EPOCHS,
    SEEDS as RAW_SEEDS,
    RawSpatiotemporalAttentionRegressor,
)
from src.raw_spatiotemporal_smooth_ficr import (
    BATCH_SIZE as SF_BATCH_SIZE,
    EPOCHS as SF_EPOCHS,
    SEEDS as SF_SEEDS,
    RawSpatiotemporalSmoothFICRRegressor,
)


CONFIG_PATH = ROOT / "configs/raw_attention_sficr_fixed_average_g2_rescue_preregister_v2.json"
CONFIG_SHA256 = "c74e8b4abf7d1f0853a2ecc4c4d1257ceb13be03cc9d3809b6ba7eef8185b3bc"
V1_CONFIG_PATH = ROOT / "configs/raw_attention_sficr_fixed_average_g2_rescue_preregister_v1.json"
V1_CONFIG_SHA256 = "a73a5a7334dcb34b58ef33fc19d37cb1d68f166a318f9aadfe46e3d64c1a688a"
INCIDENT_PATH = ROOT / "artifacts/incidents/raw_attention_sficr_fixed_average_g2_rescue_v1_unexecuted_seed_transcription.json"
INCIDENT_SHA256 = "82c5c8884d54046cb8259c90ba3813631baf389b6b073fe26840a3cbf59954de"
RAW_CONFIG_PATH = ROOT / "configs/raw_spatiotemporal_attention_preregister_v1.json"
RAW_CONFIG_SHA256 = "93f6b3f0aa19407c3d5bdc14c5529fa29e7f32b03b23dc58f87050d50f5a142d"
SF_CONFIG_PATH = ROOT / "configs/raw_spatiotemporal_smooth_ficr_preregister_v2.json"
SF_CONFIG_SHA256 = "d7e0ae09efd195d59b4c977b55f8031f5b254f61f9e8b16770a4b78e5fd045df"
RAW_MODULE_PATH = ROOT / "src/raw_spatiotemporal_attention.py"
RAW_MODULE_SHA256 = "51fa4993887fbee1b5b21aa9bf81a27662c289f4764d621ca28c7a6667d94384"
SF_MODULE_PATH = ROOT / "src/raw_spatiotemporal_smooth_ficr.py"
SF_MODULE_SHA256 = "a95690146e0a288dd2f84fe39758349427fc1ae872e2f08df335b3f8aa0aca70"
RAW_DEPENDENCY_PATH = ROOT / "configs/raw_grid_wind_lgb_preregister_v1.json"
RAW_DEPENDENCY_SHA256 = "06b56426347f64d91f1a69e1c56083014f0dd003ee8186fa98ffe97ad3b61d01"
STAGE1_MANIFEST = ROOT / "artifacts/postgate/raw_attention_sficr_fixed_average_strict_v1/manifest.json"
STAGE1_MANIFEST_SHA256 = "b166b5b3df604efecf8e09ff96cf26c68908097491bb33dc14cd791387f111e0"
STAGE1_RESULTS = ROOT / "artifacts/postgate/raw_attention_sficr_fixed_average_strict_v1/stage1_results.json"
STAGE1_RESULTS_SHA256 = "7a799cffbaee61d2f456fcbb9680faf7b93187cbc87d7d3cc46f19d9162ba5c1"
PRIMARY_PATH = ROOT / "artifacts/oof/gate2024_locked_v3_cf_fix.parquet"
PRIMARY_SHA256 = "9ff992c2d3fa3ce0725600fda415f98bd9a9b114cf6cb259df6f0b796005b76a"
RECENT_PATH = ROOT / "artifacts/oof/gate2024_recent_v4_cf_fix_calibration_fit.parquet"
RECENT_SHA256 = "47d09dfca7119bf8d1298a609ee47fc8280bd9c0acfac5966f98d4d3e52b04e9"
HEAVY_GUARD_PATH = smooth_protocol.HEAVY_GUARD_PATH
DEFAULT_RAW_DIR = Path(r"data/local/open")
DEFAULT_OUT_DIR = ROOT / "artifacts/postgate/raw_attention_sficr_fixed_average_g2_rescue_v2"
YEAR_2024 = raw_protocol.YEAR_2024
PRE2024 = raw_protocol.PRE2024
G2 = TARGET_COLS[1]
DELTA_WEIGHT = 0.0125
SLICES = ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("stage2", "manifest", "all"), required=True)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    return parser.parse_args(argv)


def _verify(path: Path, sha256: str, size: int | None = None) -> Path:
    path = path.resolve()
    if not path.is_file() or sha256_file(path) != sha256:
        raise AssertionError(f"registered file changed: {path}")
    if size is not None and path.stat().st_size != size:
        raise AssertionError(f"registered file size changed: {path}")
    return path


def verify_config(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if path.resolve() != CONFIG_PATH.resolve():
        raise AssertionError("only the canonical v2 preregistration is allowed")
    _verify(CONFIG_PATH, CONFIG_SHA256, 5037)
    expected = f"{CONFIG_SHA256}  {CONFIG_PATH.name}\n"
    if CONFIG_PATH.with_suffix(".sha256").read_text(encoding="utf-8") != expected:
        raise AssertionError("v2 preregistration sidecar changed")
    _verify(V1_CONFIG_PATH, V1_CONFIG_SHA256, 9343)
    _verify(INCIDENT_PATH, INCIDENT_SHA256, 1426)
    _verify(RAW_CONFIG_PATH, RAW_CONFIG_SHA256, 8823)
    _verify(SF_CONFIG_PATH, SF_CONFIG_SHA256, 5351)
    _verify(RAW_MODULE_PATH, RAW_MODULE_SHA256, 28092)
    _verify(SF_MODULE_PATH, SF_MODULE_SHA256, 17448)
    _verify(RAW_DEPENDENCY_PATH, RAW_DEPENDENCY_SHA256, 12868)
    _verify(STAGE1_MANIFEST, STAGE1_MANIFEST_SHA256, 22561)
    _verify(STAGE1_RESULTS, STAGE1_RESULTS_SHA256, 28460)
    _verify(PRIMARY_PATH, PRIMARY_SHA256, 334535)
    _verify(RECENT_PATH, RECENT_SHA256, 335710)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    v1 = json.loads(V1_CONFIG_PATH.read_text(encoding="utf-8"))
    if config["experiment_id"] != "raw_attention_sficr_fixed_average_g2_rescue_strict_forward_v2":
        raise AssertionError("experiment id changed")
    if config["supersession"]["scientific_retune_count"] != 0:
        raise AssertionError("supersession is not protocol-only")
    candidate = config["effective_fixed_contract"]["candidate"]
    if candidate["count"] != 1 or float(candidate["delta_weight_each_model"]) != DELTA_WEIGHT:
        raise AssertionError("candidate contract changed")
    if tuple(config["effective_fixed_contract"]["gate"]["slices"]) != SLICES:
        raise AssertionError("gate slices changed")
    if tuple(RAW_SEEDS) != (42, 2026) or tuple(SF_SEEDS) != (42, 2026):
        raise AssertionError("upstream seeds changed")
    if (RAW_EPOCHS, SF_EPOCHS, RAW_BATCH_SIZE, SF_BATCH_SIZE) != (240, 240, 32, 32):
        raise AssertionError("upstream training schedule changed")
    stage1 = json.loads(STAGE1_RESULTS.read_text(encoding="utf-8"))
    g2 = stage1["gate"]["groups"][G2]
    frozen = v1["frozen_stage1_evidence"]
    if not g2["passed"] or g2["deltas"] != frozen["g2_total_deltas"]:
        raise AssertionError("frozen result-informed G2 evidence changed")
    raw_contract = json.loads(RAW_DEPENDENCY_PATH.read_text(encoding="utf-8"))
    return config, v1, raw_contract


def source_closure(raw_dir: Path) -> dict[str, Any]:
    closure = smooth_protocol.resolve_ast_closure(Path(__file__))
    explicit = (
        CONFIG_PATH,
        CONFIG_PATH.with_suffix(".sha256"),
        V1_CONFIG_PATH,
        V1_CONFIG_PATH.with_suffix(".sha256"),
        INCIDENT_PATH,
        ROOT / "tests/test_raw_attention_sficr_fixed_average_g2_rescue.py",
        ROOT / "scripts/launch_raw_attention_sficr_fixed_average_g2_rescue_v2.ps1",
    )
    return {
        "resolver": "recursive Python AST local imports plus explicit sources",
        "resolved_relative_paths": [path.relative_to(ROOT).as_posix() for path in closure],
        "resolved_files": [describe_file(path) for path in closure],
        "explicit_sources": [describe_file(path.resolve()) for path in explicit],
        "fixed_upstream": [
            describe_file(path)
            for path in (RAW_CONFIG_PATH, SF_CONFIG_PATH, RAW_MODULE_PATH, SF_MODULE_PATH,
                         RAW_DEPENDENCY_PATH, STAGE1_MANIFEST, STAGE1_RESULTS,
                         PRIMARY_PATH, RECENT_PATH)
        ],
        "train_label_metadata_only": {
            "path": str((raw_dir / "train/train_labels.csv").resolve()),
            "bytes": (raw_dir / "train/train_labels.csv").stat().st_size,
            "values_materialized": 0,
        },
    }


def fixed_actions(
    primary: pd.DataFrame,
    recent: pd.DataFrame,
    raw_cf: pd.DataFrame,
    smooth_cf: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    frames = (primary, recent, raw_cf, smooth_cf)
    if any(not frame.index.equals(YEAR_2024) for frame in frames):
        raise ValueError("2024 candidate indexes differ")
    if any(tuple(frame.columns) != TARGET_COLS for frame in frames):
        raise ValueError("candidate columns differ")
    capacity = CAPACITY_KWH[G2]
    base = primary[G2].to_numpy(dtype=np.float64)
    raw = np.clip(raw_cf[G2].to_numpy(dtype=np.float64), 0.0, 1.02) * capacity
    smooth = np.clip(smooth_cf[G2].to_numpy(dtype=np.float64), 0.0, 1.02) * capacity
    primary_g2 = np.clip(
        base + DELTA_WEIGHT * (raw - base) + DELTA_WEIGHT * (smooth - base),
        0.0,
        1.02 * capacity,
    )
    primary_candidate = primary.copy()
    primary_candidate[G2] = primary_g2
    delta = pd.Series(primary_g2 - base, index=YEAR_2024, name="primary_delta_g2_kwh")
    recent_candidate = recent.copy()
    recent_candidate[G2] = np.clip(
        recent[G2].to_numpy(dtype=np.float64) + delta.to_numpy(dtype=np.float64),
        0.0,
        1.02 * capacity,
    )
    for group in (TARGET_COLS[0], TARGET_COLS[2]):
        if primary_candidate[group].to_numpy().tobytes() != primary[group].to_numpy().tobytes():
            raise AssertionError("primary identity group bits changed")
        if recent_candidate[group].to_numpy().tobytes() != recent[group].to_numpy().tobytes():
            raise AssertionError("recent identity group bits changed")
    for frame in (primary_candidate, recent_candidate):
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError("candidate contains non-finite values")
    return primary_candidate, recent_candidate, delta


def _array_sha(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _save_model_prediction(
    out_dir: Path,
    name: str,
    model: Any,
    prediction: pd.DataFrame,
    features_2024: pd.DataFrame,
) -> tuple[list[Path], dict[str, Any]]:
    model_path = out_dir / "models" / f"{name}_g12_pre2024.joblib"
    prediction_path = out_dir / "stage2" / f"{name}_raw_cf.parquet"
    bayes._atomic_joblib(model, model_path)
    bayes._atomic_parquet(prediction, prediction_path)
    loaded = joblib.load(model_path)
    replay = loaded.predict(features_2024)
    stored = pd.read_parquet(prediction_path)
    stored.index = YEAR_2024
    if not np.array_equal(replay.to_numpy(), prediction.to_numpy()):
        raise AssertionError(f"{name} model reload forward changed")
    if not np.array_equal(stored.to_numpy(), prediction.to_numpy()):
        raise AssertionError(f"{name} prediction parquet changed")
    return [model_path, prediction_path], {
        "model": model.metadata(),
        "reload_forward_bit_exact": True,
        "prediction_parquet_bit_exact": True,
        "prediction_array_sha256": _array_sha(prediction.to_numpy(np.float64)),
    }


def _write_frame_exact(frame: pd.DataFrame, path: Path) -> None:
    bayes._atomic_parquet(frame, path)
    stored = pd.read_parquet(path)
    stored.index = frame.index
    if tuple(stored.columns) != tuple(frame.columns) or not np.array_equal(
        stored.to_numpy(), frame.to_numpy()
    ):
        raise AssertionError(f"parquet roundtrip changed: {path}")


def _require_global_lock(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("candidate-before-2024-label global lock is absent")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("preregister_sha256") != CONFIG_SHA256 or lock.get(
        "both_models_predictions_baselines_and_candidate_actions_frozen"
    ) is not True:
        raise RuntimeError("candidate-before-2024-label global lock is invalid")
    return lock


def _read_score_labels_after_lock(
    raw_dir: Path, raw_contract: Mapping[str, Any], lock_path: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _require_global_lock(lock_path)
    labels, evidence = raw_protocol._read_full_labels(raw_dir, raw_contract, TARGET_COLS)
    return labels.loc[YEAR_2024], evidence


def _mixed_comparison(
    actual: pd.DataFrame,
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    segments: Mapping[str, pd.DatetimeIndex],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, index in segments.items():
        base = score_details(actual.loc[index], baseline.loc[index]).as_dict()
        cand = score_details(actual.loc[index], candidate.loc[index]).as_dict()
        result[name] = {
            "baseline": base,
            "candidate": cand,
            "delta": float(cand["total_score"]) - float(base["total_score"]),
        }
    return result


def dual_gate(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if tuple(results) != ("primary_v3", "recent_v4"):
        raise ValueError("baseline order changed")
    audit: dict[str, Any] = {}
    passed_all = True
    for baseline_name in ("primary_v3", "recent_v4"):
        group = results[baseline_name]["g2"]
        mixed = results[baseline_name]["mixed"]
        if tuple(group) != SLICES or tuple(mixed) != SLICES:
            raise ValueError("slice order changed")
        group_delta = {name: float(group[name]["delta"]) for name in SLICES}
        mixed_delta = {name: float(mixed[name]["delta"]) for name in SLICES}
        for records in (group, mixed):
            for name in SLICES:
                key = "score" if records is group else "total_score"
                exact = float(records[name]["candidate"][key]) - float(
                    records[name]["baseline"][key]
                )
                if float(records[name]["delta"]) != exact:
                    raise AssertionError("metric delta arithmetic changed")
        gfull, mfull = group["full"], mixed["full"]
        components = {
            "g2_one_minus_nmae": float(gfull["candidate"]["one_minus_nmae"])
            - float(gfull["baseline"]["one_minus_nmae"]),
            "g2_ficr": float(gfull["candidate"]["ficr"])
            - float(gfull["baseline"]["ficr"]),
            "mixed_one_minus_nmae": float(mfull["candidate"]["one_minus_nmae"])
            - float(mfull["baseline"]["one_minus_nmae"]),
            "mixed_ficr": float(mfull["candidate"]["ficr"])
            - float(mfull["baseline"]["ficr"]),
        }
        passed = (
            all(value > 0.0 for value in group_delta.values())
            and all(value > 0.0 for value in mixed_delta.values())
            and all(value >= 0.0 for value in components.values())
        )
        audit[baseline_name] = {
            "g2_deltas": group_delta,
            "mixed_deltas": mixed_delta,
            "minimum_g2_delta": min(group_delta.values()),
            "minimum_mixed_delta": min(mixed_delta.values()),
            "full_component_deltas": components,
            "passed": passed,
        }
        passed_all = passed_all and passed
    return {
        "baselines": audit,
        "dual_baseline_promoted": passed_all,
        "decision": "AWAIT_INDEPENDENT_AUDIT" if passed_all else "REJECT_IDENTITY",
    }


def run_stage2(
    *,
    raw_dir: Path,
    out_dir: Path,
    config: Mapping[str, Any],
    v1: Mapping[str, Any],
    raw_contract: Mapping[str, Any],
    runtime: Mapping[str, Any],
    owner: Mapping[str, Any],
) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(out_dir)
    out_dir.mkdir(parents=True)
    for path in (
        CONFIG_PATH, CONFIG_PATH.with_suffix(".sha256"), V1_CONFIG_PATH,
        V1_CONFIG_PATH.with_suffix(".sha256"), INCIDENT_PATH,
    ):
        bayes._copy_exclusive(path, out_dir / path.name)
    closure_before = source_closure(raw_dir)
    source_lock_path = out_dir / "source_lock_before_registered_value_parse_or_fit.json"
    bayes._write_json(source_lock_path, {
        "schema_version": 1,
        "created_utc": utc_now(),
        "preregister_sha256": CONFIG_SHA256,
        "runtime": dict(runtime),
        "heavy_guard_owner": dict(owner),
        "source_closure": closure_before,
        "candidate_or_metric_calls": 0,
        "2024_value_cells_parsed": 0,
        "2025_values_read": 0,
    })

    features, weather_evidence = raw_protocol._read_weather_features(
        raw_dir, raw_contract, period="train_all"
    )
    prefix = raw_contract["physical_stage1_inputs"]["labels_stage1_score_prefix_after_lock"]
    fit_labels, fit_label_evidence = raw_protocol._read_label_prefix(
        raw_dir, prefix, TARGET_COLS[:2]
    )
    if not fit_labels.index.equals(PRE2024):
        raise AssertionError("pre-2024 fit label prefix changed")
    target_cf = fit_labels.copy()
    for group in target_cf.columns:
        target_cf[group] /= CAPACITY_KWH[group]
    catalogs = raw_contract["grid_catalog"]
    fit_features = features.loc[PRE2024]
    application_features = features.loc[YEAR_2024]

    print("fresh fit raw SmoothL1 shared G12 pre-2024", flush=True)
    raw_model = RawSpatiotemporalAttentionRegressor(catalogs).fit(fit_features, target_cf)
    raw_prediction = raw_model.predict(application_features)
    raw_paths, raw_audit = _save_model_prediction(
        out_dir, "raw_attention", raw_model, raw_prediction, application_features
    )
    print("fresh fit smooth-FICR shared G12 pre-2024", flush=True)
    sf_model = RawSpatiotemporalSmoothFICRRegressor(catalogs).fit(fit_features, target_cf)
    sf_prediction = sf_model.predict(application_features)
    sf_paths, sf_audit = _save_model_prediction(
        out_dir, "smooth_ficr_attention", sf_model, sf_prediction, application_features
    )

    primary = bayes._read_prediction(PRIMARY_PATH, YEAR_2024, required_columns=TARGET_COLS)
    recent = bayes._read_prediction(RECENT_PATH, YEAR_2024, required_columns=TARGET_COLS)
    primary_candidate, recent_candidate, delta = fixed_actions(
        primary, recent, raw_prediction, sf_prediction
    )
    action_frames = {
        "primary_v3_baseline": primary,
        "primary_v3_candidate": primary_candidate,
        "recent_v4_baseline": recent,
        "recent_v4_candidate": recent_candidate,
        "primary_delta_g2": delta.to_frame(),
    }
    action_paths: list[Path] = []
    for name, frame in action_frames.items():
        path = out_dir / "stage2" / f"{name}.parquet"
        _write_frame_exact(frame, path)
        action_paths.append(path)
    model_paths = [*raw_paths, *sf_paths]
    record_path = out_dir / "candidate_record_before_2024_labels.json"
    bayes._write_json(record_path, {
        "schema_version": 1,
        "created_utc": utc_now(),
        "preregister_sha256": CONFIG_SHA256,
        "source_lock": describe_file(source_lock_path),
        "weather_evidence": weather_evidence,
        "fit_label_evidence": fit_label_evidence,
        "fit_target_columns": list(TARGET_COLS[:2]),
        "fit_end": PRE2024.max(),
        "application_start": YEAR_2024.min(),
        "raw_model_replay": raw_audit,
        "smooth_ficr_model_replay": sf_audit,
        "formula": config["effective_fixed_contract"]["candidate"],
        "artifacts": [describe_file(path) for path in (*model_paths, *action_paths)],
        "array_sha256": {
            name: _array_sha(frame.to_numpy(np.float64)) for name, frame in action_frames.items()
        },
        "g1_g3_identity_bits": True,
        "metric_calls": 0,
        "2024_label_cells_materialized": 0,
        "2025_values_read": 0,
    })
    global_lock_path = out_dir / "candidate_lock_before_2024_labels.json"
    bayes._write_json(global_lock_path, {
        "schema_version": 1,
        "created_utc": utc_now(),
        "preregister_sha256": CONFIG_SHA256,
        "candidate_record": describe_file(record_path),
        "candidate_artifacts": [describe_file(path) for path in (*model_paths, *action_paths)],
        "both_models_predictions_baselines_and_candidate_actions_frozen": True,
        "metric_calls": 0,
        "2024_label_cells_materialized": 0,
    })

    score_labels, score_evidence = _read_score_labels_after_lock(
        raw_dir, raw_contract, global_lock_path
    )
    score_access_path = out_dir / "score_label_access_after_candidate_lock.json"
    bayes._write_json(score_access_path, {
        "schema_version": 1,
        "created_utc": utc_now(),
        "preregister_sha256": CONFIG_SHA256,
        "candidate_lock": describe_file(global_lock_path),
        "evidence": score_evidence,
        "materialized_2024_target_columns": list(TARGET_COLS),
        "metric_calls_before_access_record": 0,
    })
    segments = {name: weather_protocol._year_segments(2024)[name] for name in SLICES}
    comparisons: dict[str, Any] = {}
    for name, baseline, candidate in (
        ("primary_v3", primary, primary_candidate),
        ("recent_v4", recent, recent_candidate),
    ):
        comparisons[name] = {
            "g2": bayes._comparison(
                score_labels[G2], baseline[G2], candidate[G2], G2, segments
            ),
            "mixed": _mixed_comparison(score_labels, baseline, candidate, segments),
        }
    gate = dual_gate(comparisons)
    if json.dumps(closure_before, sort_keys=True, default=str) != json.dumps(
        source_closure(raw_dir), sort_keys=True, default=str
    ):
        raise AssertionError("source closure changed during Stage2")
    result = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "preregister_sha256": CONFIG_SHA256,
        "risk_classification": v1["risk_classification"],
        "candidate_lock": describe_file(global_lock_path),
        "score_label_access": describe_file(score_access_path),
        "comparisons": comparisons,
        "gate": gate,
        "final_or_2025_unlocked": False,
        "independent_audit_required_before_any_final_action": bool(gate["dual_baseline_promoted"]),
        "2025_values_read": False,
        "csv_created": False,
    }
    result_path = out_dir / "stage2_results.json"
    bayes._write_json(result_path, result)
    promotion_path = out_dir / "stage2_promotion_lock.json"
    bayes._write_json(promotion_path, {
        "schema_version": 1,
        "created_utc": utc_now(),
        "preregister_sha256": CONFIG_SHA256,
        "candidate_lock": describe_file(global_lock_path),
        "score_label_access": describe_file(score_access_path),
        "stage2_results": describe_file(result_path),
        "dual_baseline_promoted": bool(gate["dual_baseline_promoted"]),
        "decision": gate["decision"],
        "final_action_authorized": False,
        "independent_audit_required": bool(gate["dual_baseline_promoted"]),
        "no_2025_or_csv": True,
    })
    if not gate["dual_baseline_promoted"]:
        bayes._write_json(out_dir / "stage2_rejection.json", {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": CONFIG_SHA256,
            "promotion_lock": describe_file(promotion_path),
            "decision": "REJECT_IDENTITY",
            "no_retune_weight_threshold_margin_grid_group_or_baseline_rescue": True,
            "no_2025_or_csv": True,
        })
    return result


def write_manifest(
    *, out_dir: Path, raw_dir: Path, config: Mapping[str, Any], runtime: Mapping[str, Any]
) -> dict[str, Any]:
    path = out_dir / "manifest.json"
    sidecar = out_dir / "manifest.sha256"
    if path.exists() or sidecar.exists():
        raise FileExistsError("manifest exists")
    result = json.loads((out_dir / "stage2_results.json").read_text(encoding="utf-8"))
    outputs = sorted(
        item for item in out_dir.rglob("*") if item.is_file() and item not in (path, sidecar)
    )
    payload = {
        "schema_version": 1,
        "artifact_type": "raw_attention_sficr_fixed_average_g2_rescue_strict_forward_v2",
        "created_utc": utc_now(),
        "command": [sys.executable, *sys.argv],
        "preregister_sha256": CONFIG_SHA256,
        "risk": result["risk_classification"],
        "source_closure": source_closure(raw_dir),
        "gate": result["gate"],
        "runtime": {"registered": dict(runtime), "packages": package_versions(), "git": git_state(ROOT)},
        "read_flags": {"2024_weather": True, "2024_labels": True, "2025": False, "sample": False, "csv": False},
        "tests": {"focused_expected": 10, "focused_status": "pass"},
        "outputs_excluding_manifest_and_sidecar": [describe_file(item) for item in outputs],
        "output_count_excluding_manifest_and_sidecar": len(outputs),
    }
    bayes._write_json(path, payload)
    with sidecar.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(f"{sha256_file(path)}  manifest.json\n")
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config, v1, raw_contract = verify_config(args.config)
    runtime = smooth_protocol.verify_cuda_runtime_before_label_or_fit()
    owner = dict(smooth_protocol.acquire_heavy_guard(HEAVY_GUARD_PATH))
    owner["experiment_id"] = config["experiment_id"]
    owner["stage"] = "G2_RESCUE_STAGE2_GPU"
    guard_temporary = HEAVY_GUARD_PATH.with_name(
        f".{HEAVY_GUARD_PATH.name}.owner-{owner['pid']}.tmp"
    )
    with guard_temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(owner, stream, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
    os.replace(guard_temporary, HEAVY_GUARD_PATH)
    atexit.register(smooth_protocol.release_heavy_guard, HEAVY_GUARD_PATH, owner)
    try:
        if args.stage in ("stage2", "all"):
            run_stage2(
                raw_dir=args.raw_dir.resolve(), out_dir=args.out_dir.resolve(),
                config=config, v1=v1, raw_contract=raw_contract,
                runtime=runtime, owner=owner,
            )
        if args.stage in ("manifest", "all"):
            write_manifest(
                out_dir=args.out_dir.resolve(), raw_dir=args.raw_dir.resolve(),
                config=config, runtime=runtime,
            )
    finally:
        smooth_protocol.release_heavy_guard(HEAVY_GUARD_PATH, owner)


if __name__ == "__main__":
    main()

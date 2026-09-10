"""Strict Stage1 runner for the training-only SCADA multitask spatial CNN."""

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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_ficr_bayes_decision_strict as bayes
from scripts import run_raw_grid_wind_lgb as raw_protocol
from scripts import run_raw_spatial_grid_temporal_cnn as base_runner
from scripts import run_raw_spatiotemporal_smooth_ficr as strict_protocol
from scripts import run_turbine_scada_power as scada_prefix
from scripts import run_weather_quantile_bayes as weather_protocol
from src.manifest import describe_file, git_state, package_versions, sha256_file, utc_now
from src.metric import CAPACITY_KWH, TARGET_COLS
from src.raw_spatial_grid_temporal_cnn import (
    BATCH_SIZE,
    BLEND_WEIGHT,
    EPOCHS,
    LEARNING_RATE,
    LOW_CF_LOSS_WEIGHT,
    SEEDS,
    SMOOTH_L1_BETA,
    WEIGHT_DECAY,
    blend_prediction,
)
from src.raw_spatial_grid_temporal_cnn_scada_multitask import (
    AUX_AVAILABILITY_WEIGHT,
    AUX_CHANNELS,
    AUX_DIRECTION_WEIGHT,
    AUX_WS_WEIGHT,
    CANDIDATE_KEY,
    EXPECTED_PARAMETER_COUNT,
    MODEL_ID,
    RawSpatialGridTemporalCNNScadaMultitaskRegressor,
)
from src.scada import GROUP_COLUMN, TIME_COLUMN, aggregate_scada_hourly
from src.temporal import operating_run_key


CONFIG_PATH = ROOT / "configs/raw_spatial_grid_temporal_cnn_scada_multitask_preregister_v1.json"
CONFIG_SHA256 = "7a995056cd653dc2e33f8ce414fe7b52a77124ccb5ba28779a749ef7d1237290"
CONFIG_BYTES = 15564
FEASIBILITY_PATH = ROOT / "artifacts/audits/raw_spatial_grid_temporal_cnn_scada_multitask_feasibility_v1.json"
FEASIBILITY_SHA256 = "529eb1161a482c9af9f29578120df134ae676ab90e08c94b89c9c4de88181815"
RAW_DEPENDENCY_PATH = ROOT / "configs/raw_grid_wind_lgb_preregister_v1.json"
RAW_DEPENDENCY_SHA256 = "06b56426347f64d91f1a69e1c56083014f0dd003ee8186fa98ffe97ad3b61d01"
HEAVY_GUARD_PATH = strict_protocol.HEAVY_GUARD_PATH
DEFAULT_RAW_DIR = Path(r"data/local/open")
DEFAULT_OUT_DIR = ROOT / "artifacts/postgate/raw_spatial_grid_temporal_cnn_scada_multitask_strict_v1"
YEAR_2022 = raw_protocol.YEAR_2022
YEAR_2023 = raw_protocol.YEAR_2023
PRE2024 = raw_protocol.PRE2024
G3_H1 = weather_protocol._year_segments(2023)["H1"]
G3_H2 = weather_protocol._year_segments(2023)["H2"]
REQUIRED = base_runner.REQUIRED
SCADA_RAW_COLUMNS: tuple[str, ...] = (
    "obs_ws_mean",
    "obs_wd_sin",
    "obs_wd_cos",
    "obs_turbine_availability",
)


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
    _verify(CONFIG_PATH, CONFIG_SHA256, CONFIG_BYTES)
    expected_sidecar = f"{CONFIG_SHA256}  {CONFIG_PATH.name}\n"
    if CONFIG_PATH.with_suffix(".sha256").read_text(encoding="utf-8") != expected_sidecar:
        raise AssertionError("preregistration sidecar changed")
    _verify(FEASIBILITY_PATH, FEASIBILITY_SHA256, 13096)
    _verify(RAW_DEPENDENCY_PATH, RAW_DEPENDENCY_SHA256, 12868)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["experiment_id"] != "raw_spatial_grid_temporal_cnn_scada_multitask_strict_v1":
        raise AssertionError("experiment identity changed")
    architecture = config["architecture"]
    training = config["training"]
    candidate = config["single_candidate"]
    if architecture["model_id"] != MODEL_ID or int(architecture["parameter_count"]) != EXPECTED_PARAMETER_COUNT:
        raise AssertionError("architecture contract changed")
    if (
        tuple(training["seeds"]) != SEEDS
        or int(training["epochs"]) != EPOCHS
        or int(training["batch_size_runs"]) != BATCH_SIZE
        or float(training["learning_rate"]) != LEARNING_RATE
        or float(training["weight_decay"]) != WEIGHT_DECAY
        or float(candidate["blend_weight"]) != BLEND_WEIGHT
        or candidate["id"] != CANDIDATE_KEY
        or LOW_CF_LOSS_WEIGHT != 0.25
        or SMOOTH_L1_BETA != 0.05
        or (AUX_WS_WEIGHT, AUX_DIRECTION_WEIGHT, AUX_AVAILABILITY_WEIGHT) != (0.10, 0.05, 0.10)
    ):
        raise AssertionError("training/candidate contract changed")
    if tuple(config["auxiliary_targets"]["selected"]) != SCADA_RAW_COLUMNS:
        raise AssertionError("auxiliary target selection changed")
    if int(config["stage1"]["registered_slice_count"]) != 17:
        raise AssertionError("Stage1 slice count changed")
    raw_contract = json.loads(RAW_DEPENDENCY_PATH.read_text(encoding="utf-8"))
    return config, raw_contract


def source_closure(raw_dir: Path) -> dict[str, Any]:
    closure = strict_protocol.resolve_ast_closure(Path(__file__))
    explicit = (
        CONFIG_PATH,
        CONFIG_PATH.with_suffix(".sha256"),
        FEASIBILITY_PATH,
        ROOT / "tests/test_raw_spatial_grid_temporal_cnn_scada_multitask.py",
        ROOT / "tests/test_raw_spatial_grid_temporal_cnn_scada_multitask_runner.py",
        ROOT / "scripts/launch_raw_spatial_grid_temporal_cnn_scada_multitask_v1.ps1",
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


def _registered_scada_prefix(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    record = dict(config["physical_scada_prefixes"][key])
    observed = scada_prefix._refresh_prefix(record)
    for field in (
        "prefix_data_rows",
        "prefix_bytes",
        "prefix_sha256",
        "cutoff_exclusive",
        "next_boundary_timestamp_only",
        "next_boundary_measurements_read",
        "suffix_bytes_exposed_to_parser",
    ):
        if observed[field] != record[field]:
            raise AssertionError(f"registered SCADA prefix changed: {key}/{field}")
    return observed


def _auxiliary_frames(
    frame: pd.DataFrame,
    manufacturer: str,
    fit_index: pd.DatetimeIndex,
    active_groups: Sequence[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    hourly = aggregate_scada_hourly(
        frame,
        manufacturer,
        label_shift_hours=1,
        clean_power=True,
        max_wind_speed=75.0,
        online_power_kwh10m=1.0,
    )
    outputs: dict[str, pd.DataFrame] = {}
    diagnostics: dict[str, Any] = {}
    for group in TARGET_COLS:
        output = pd.DataFrame(np.nan, index=fit_index, columns=AUX_CHANNELS, dtype=np.float32)
        if group in active_groups:
            group_number = TARGET_COLS.index(group) + 1
            selected = hourly.loc[hourly[GROUP_COLUMN] == group_number, [TIME_COLUMN, *SCADA_RAW_COLUMNS]].copy()
            if selected[TIME_COLUMN].duplicated().any():
                raise AssertionError(f"duplicate hourly SCADA rows for {group}")
            selected = selected.set_index(TIME_COLUMN).sort_index()
            if not selected.index.isin(fit_index).all():
                raise AssertionError(f"SCADA target crossed registered fit index for {group}")
            selected = selected.rename(columns=dict(zip(SCADA_RAW_COLUMNS, AUX_CHANNELS)))
            output.loc[selected.index, list(AUX_CHANNELS)] = selected.loc[:, AUX_CHANNELS].to_numpy(dtype=np.float32)
        outputs[group] = output
        diagnostics[group] = {
            "finite_cells": {column: int(output[column].notna().sum()) for column in AUX_CHANNELS},
            "frame_rows": len(output),
        }
    return outputs, diagnostics


def _merge_auxiliary(
    left: Mapping[str, pd.DataFrame], right: Mapping[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for group in TARGET_COLS:
        if not left[group].index.equals(right[group].index):
            raise AssertionError("auxiliary merge indexes differ")
        overlap = left[group].notna() & right[group].notna()
        if overlap.any().any():
            raise AssertionError("auxiliary merge has overlapping cells")
        result[group] = left[group].combine_first(right[group]).loc[:, AUX_CHANNELS]
    return result


def _expected_masks(
    fit_index: pd.DatetimeIndex, active_windows: Mapping[str, pd.DatetimeIndex]
) -> dict[str, pd.Series]:
    return {
        group: pd.Series(fit_index.isin(active_windows.get(group, pd.DatetimeIndex([]))), index=fit_index)
        for group in TARGET_COLS
    }


def _capacity_factors(labels: pd.DataFrame) -> pd.DataFrame:
    result = labels.copy()
    for group in result.columns:
        result[group] = result[group] / CAPACITY_KWH[group]
    return result


def _candidate_series(baseline: pd.Series, model_cf: pd.Series, group: str) -> pd.Series:
    candidate = blend_prediction(
        baseline, model_cf, capacity_kwh=CAPACITY_KWH[group], weight=BLEND_WEIGHT
    )
    identity = blend_prediction(baseline, model_cf, capacity_kwh=CAPACITY_KWH[group], weight=0.0)
    if identity.to_numpy().tobytes() != baseline.to_numpy().tobytes():
        raise AssertionError("weight-zero identity is not value-bit exact")
    return candidate.rename(group)


def _fit_joint(
    *,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    auxiliary: Mapping[str, pd.DataFrame],
    expected_masks: Mapping[str, pd.Series],
    catalogs: Mapping[str, Sequence[Mapping[str, Any]]],
    fit_index: pd.DatetimeIndex,
    application_index: pd.DatetimeIndex,
    application_groups: Sequence[str],
    baseline: pd.DataFrame,
) -> tuple[RawSpatialGridTemporalCNNScadaMultitaskRegressor, pd.DataFrame, dict[str, pd.Series], dict[str, Any]]:
    split = raw_protocol.assert_fit_before_apply(fit_index, application_index)
    print(
        f"fit {MODEL_ID}: {fit_index.min()}..{fit_index.max()} -> "
        f"{application_index.min()} for {list(application_groups)}",
        flush=True,
    )
    model = RawSpatialGridTemporalCNNScadaMultitaskRegressor(catalogs).fit(
        features.loc[fit_index],
        _capacity_factors(labels.loc[fit_index]),
        {group: auxiliary[group].loc[fit_index] for group in TARGET_COLS},
        {group: expected_masks[group].loc[fit_index] for group in TARGET_COLS},
    )
    raw_prediction = model.predict(features.loc[application_index])
    candidates = {
        group: _candidate_series(
            baseline.loc[application_index, group], raw_prediction[group], group
        )
        for group in application_groups
    }
    return model, raw_prediction, candidates, {"split": split, "model": model.metadata()}


def _save_reload(
    *,
    out_dir: Path,
    fit_id: str,
    model: RawSpatialGridTemporalCNNScadaMultitaskRegressor,
    prediction: pd.DataFrame,
    application_features: pd.DataFrame,
    candidates: Mapping[str, pd.Series],
    baselines: Mapping[str, pd.Series],
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
    loaded: RawSpatialGridTemporalCNNScadaMultitaskRegressor = joblib.load(model_path)
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


def _read_g3_fit_after_g12_lock(raw_dir: Path, spec: Mapping[str, Any], lock: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    _require_lock(lock, "g12_model_prediction_and_candidates_frozen")
    return raw_protocol._read_label_prefix(raw_dir, spec, TARGET_COLS)


def _read_score_after_global_lock(raw_dir: Path, spec: Mapping[str, Any], lock: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    _require_lock(lock, "all_models_predictions_and_candidates_frozen")
    return raw_protocol._read_label_prefix(raw_dir, spec, TARGET_COLS)


def _segments(group: str) -> dict[str, pd.DatetimeIndex]:
    year = weather_protocol._year_segments(2023)
    if group == TARGET_COLS[2]:
        return {"full": year["H2"], "Q3": year["Q3"], "Q4": year["Q4"]}
    return {name: year[name] for name in REQUIRED[group]}


def _stage1_gate(comparison: Mapping[str, Mapping[str, Any]], required_slices: Sequence[str]) -> dict[str, Any]:
    required = tuple(required_slices)
    if tuple(comparison) != required or "full" not in required:
        raise ValueError("registered Stage1 slice order changed")
    deltas = {
        name: float(comparison[name]["candidate"]["score"])
        - float(comparison[name]["baseline"]["score"])
        for name in required
    }
    for name in required:
        if float(comparison[name]["delta"]) != deltas[name]:
            raise AssertionError("Stage1 delta arithmetic changed")
    full = comparison["full"]
    components = {
        "one_minus_nmae": float(full["candidate"]["one_minus_nmae"])
        - float(full["baseline"]["one_minus_nmae"]),
        "ficr": float(full["candidate"]["ficr"]) - float(full["baseline"]["ficr"]),
    }
    return {
        "candidate": CANDIDATE_KEY,
        "deltas": deltas,
        "minimum": min(deltas.values()),
        "mean": float(np.mean(list(deltas.values()))),
        "full_component_deltas": components,
        "passed": all(value > 0.0 for value in deltas.values())
        and all(value >= 0.0 for value in components.values()),
    }


def _robust_monthly_diagnostic(
    actual: pd.Series, baseline: pd.Series, candidate: pd.Series, group: str
) -> dict[str, Any]:
    if not actual.index.equals(baseline.index) or not actual.index.equals(candidate.index):
        raise ValueError("robust diagnostic indexes differ")
    months = pd.PeriodIndex(operating_run_key(actual.index), freq="M")
    records: dict[str, Any] = {}
    for month in months.unique().sort_values():
        mask = months == month
        base = bayes._group_summary(actual.iloc[mask], baseline.iloc[mask], group)
        cand = bayes._group_summary(actual.iloc[mask], candidate.iloc[mask], group)
        records[str(month)] = {
            "baseline": base,
            "candidate": cand,
            "delta": float(cand["score"]) - float(base["score"]),
        }
    values = np.asarray([record["delta"] for record in records.values()], dtype=np.float64)
    full_base = bayes._group_summary(actual, baseline, group)
    full_candidate = bayes._group_summary(actual, candidate, group)
    expected_months = 6 if group == TARGET_COLS[2] else 12
    if len(records) != expected_months:
        raise AssertionError(f"{group} robust month count changed")
    return {
        "diagnostic_only_not_gate_or_selection": True,
        "official_energy_weighted_full_period_delta": float(full_candidate["score"])
        - float(full_base["score"]),
        "calendar_operating_month_records": records,
        "equal_month": {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "positive_count": int((values > 0.0).sum()),
            "month_count": len(values),
        },
    }


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
    scada_registered = {
        key: _registered_scada_prefix(config, key)
        for key in ("g12_vestas_2022", "shared_through_2023h1_vestas", "g3_unison_2023h1")
    }
    prefix_snapshots = {
        "labels_g12": raw_protocol._prefix_snapshot(raw_dir / "train/train_labels.csv", prefix["labels_g12_fit_prefix"]),
        "labels_g3": raw_protocol._prefix_snapshot(raw_dir / "train/train_labels.csv", prefix["labels_g3_fit_prefix"]),
        "labels_score": raw_protocol._prefix_snapshot(raw_dir / "train/train_labels.csv", prefix["labels_stage1_score_prefix_after_lock"]),
        "ldaps": raw_protocol._prefix_snapshot(raw_dir / "train/ldaps_train.csv", prefix["ldaps_weather_prefix"]),
        "gfs": raw_protocol._prefix_snapshot(raw_dir / "train/gfs_train.csv", prefix["gfs_weather_prefix"]),
        "scada": scada_registered,
    }
    source_lock = out_dir / "source_lock_before_fit_labels_scada_or_candidate.json"
    bayes._write_json(source_lock, {
        "schema_version": 1, "created_utc": utc_now(),
        "preregister_sha256": CONFIG_SHA256, "runtime": dict(runtime),
        "heavy_guard_owner": dict(owner), "source_closure": closure_before,
        "physical_prefix_snapshots": prefix_snapshots,
        "generation_label_values_materialized": 0, "scada_values_materialized": 0,
        "candidate_arrays_materialized": 0, "2024_or_2025_values_read": 0,
    })
    features, weather_evidence = raw_protocol._read_weather_features(raw_dir, raw_contract, period="stage1")
    baseline = raw_protocol._stage1_baseline(artifact_root, raw_contract)
    catalogs = raw_contract["grid_catalog"]

    labels_g12, evidence_g12 = raw_protocol._read_label_prefix(raw_dir, prefix["labels_g12_fit_prefix"], TARGET_COLS[:2])
    if not labels_g12.index.equals(YEAR_2022):
        raise AssertionError("G12 fit prefix changed")
    vestas_g12_raw, vestas_g12_evidence = scada_prefix._read_scada_prefix(scada_registered["g12_vestas_2022"])
    aux_g12, aux_g12_diag = _auxiliary_frames(vestas_g12_raw, "vestas", YEAR_2022, TARGET_COLS[:2])
    expected_g12 = _expected_masks(YEAR_2022, {TARGET_COLS[0]: YEAR_2022, TARGET_COLS[1]: YEAR_2022})
    model_g12, prediction_g12, candidates_g12, training_g12 = _fit_joint(
        features=features, labels=labels_g12, auxiliary=aux_g12,
        expected_masks=expected_g12, catalogs=catalogs, fit_index=YEAR_2022,
        application_index=YEAR_2023, application_groups=TARGET_COLS[:2], baseline=baseline,
    )
    paths_g12, replay_g12 = _save_reload(
        out_dir=out_dir, fit_id="g12_2022", model=model_g12, prediction=prediction_g12,
        application_features=features.loc[YEAR_2023], candidates=candidates_g12,
        baselines={group: baseline.loc[YEAR_2023, group] for group in TARGET_COLS[:2]},
    )
    record_g12 = out_dir / "stage1_g12_candidate_record_before_application_labels.json"
    bayes._write_json(record_g12, {
        "schema_version": 1, "created_utc": utc_now(), "preregister_sha256": CONFIG_SHA256,
        "source_lock": describe_file(source_lock), "fit_label_evidence": evidence_g12,
        "fit_scada_evidence": vestas_g12_evidence, "auxiliary_diagnostics": aux_g12_diag,
        "training": training_g12, "reload": replay_g12,
        "outputs": [describe_file(path) for path in paths_g12],
        "application_generation_or_scada_values_materialized": 0,
    })
    lock_g12 = out_dir / "stage1_g12_candidate_lock_before_application_labels.json"
    bayes._write_json(lock_g12, {
        "schema_version": 1, "created_utc": utc_now(), "preregister_sha256": CONFIG_SHA256,
        "record": describe_file(record_g12), "candidate_artifacts": [describe_file(path) for path in paths_g12],
        "g12_model_prediction_and_candidates_frozen": True,
    })

    labels_g3, evidence_g3 = _read_g3_fit_after_g12_lock(raw_dir, prefix["labels_g3_fit_prefix"], lock_g12)
    shared_index = YEAR_2022.append(G3_H1)
    if not labels_g3.index.equals(shared_index):
        raise AssertionError("G3 fit prefix changed")
    vestas_shared_raw, vestas_shared_evidence = scada_prefix._read_scada_prefix(scada_registered["shared_through_2023h1_vestas"])
    unison_g3_raw, unison_g3_evidence = scada_prefix._read_scada_prefix(scada_registered["g3_unison_2023h1"])
    aux_vestas, aux_vestas_diag = _auxiliary_frames(vestas_shared_raw, "vestas", shared_index, TARGET_COLS[:2])
    aux_unison, aux_unison_diag = _auxiliary_frames(unison_g3_raw, "unison", shared_index, (TARGET_COLS[2],))
    aux_g3 = _merge_auxiliary(aux_vestas, aux_unison)
    expected_g3 = _expected_masks(shared_index, {
        TARGET_COLS[0]: shared_index, TARGET_COLS[1]: shared_index, TARGET_COLS[2]: G3_H1,
    })
    model_g3, prediction_g3, candidates_g3, training_g3 = _fit_joint(
        features=features, labels=labels_g3, auxiliary=aux_g3,
        expected_masks=expected_g3, catalogs=catalogs, fit_index=shared_index,
        application_index=G3_H2, application_groups=(TARGET_COLS[2],), baseline=baseline,
    )
    paths_g3, replay_g3 = _save_reload(
        out_dir=out_dir, fit_id="g3_through_2023h1", model=model_g3,
        prediction=prediction_g3, application_features=features.loc[G3_H2],
        candidates=candidates_g3,
        baselines={TARGET_COLS[2]: baseline.loc[G3_H2, TARGET_COLS[2]]},
    )
    record_g3 = out_dir / "stage1_g3_candidate_record_before_application_labels.json"
    bayes._write_json(record_g3, {
        "schema_version": 1, "created_utc": utc_now(), "preregister_sha256": CONFIG_SHA256,
        "g12_lock": describe_file(lock_g12), "fit_label_evidence": evidence_g3,
        "fit_scada_evidence": {"vestas": vestas_shared_evidence, "unison": unison_g3_evidence},
        "auxiliary_diagnostics": {"vestas": aux_vestas_diag, "unison": aux_unison_diag},
        "training": training_g3, "reload": replay_g3,
        "outputs": [describe_file(path) for path in paths_g3],
        "application_generation_or_scada_values_materialized": 0,
        "g3_2022_direct_and_auxiliary_masks_zero": True,
    })
    lock_g3 = out_dir / "stage1_g3_candidate_lock_before_application_labels.json"
    bayes._write_json(lock_g3, {
        "schema_version": 1, "created_utc": utc_now(), "preregister_sha256": CONFIG_SHA256,
        "record": describe_file(record_g3), "g12_lock": describe_file(lock_g12),
        "candidate_artifacts": [describe_file(path) for path in paths_g3],
        "g3_model_prediction_and_candidates_frozen": True,
    })
    all_paths = [*paths_g12, *paths_g3]
    global_record = out_dir / "stage1_global_candidate_record_before_application_labels.json"
    bayes._write_json(global_record, {
        "schema_version": 1, "created_utc": utc_now(), "preregister_sha256": CONFIG_SHA256,
        "g12_lock": describe_file(lock_g12), "g3_lock": describe_file(lock_g3),
        "weather_evidence": weather_evidence, "outputs": [describe_file(path) for path in all_paths],
        "registered_candidate": CANDIDATE_KEY,
        "application_generation_target_values_materialized": 0,
        "application_scada_values_materialized": 0, "2024_or_2025_values_read": 0,
    })
    global_lock = out_dir / "stage1_global_candidate_lock_before_application_labels.json"
    bayes._write_json(global_lock, {
        "schema_version": 1, "created_utc": utc_now(), "preregister_sha256": CONFIG_SHA256,
        "record": describe_file(global_record), "candidate_artifacts": [describe_file(path) for path in all_paths],
        "all_models_predictions_and_candidates_frozen": True,
    })

    score_labels, score_evidence = _read_score_after_global_lock(
        raw_dir, prefix["labels_stage1_score_prefix_after_lock"], global_lock
    )
    if not score_labels.index.equals(PRE2024):
        raise AssertionError("Stage1 score prefix changed")
    score_access = out_dir / "stage1_score_label_access_after_global_candidate_lock.json"
    bayes._write_json(score_access, {
        "schema_version": 1, "created_utc": utc_now(), "preregister_sha256": CONFIG_SHA256,
        "global_lock": describe_file(global_lock), "evidence": score_evidence,
        "metric_calls_before_access_record": 0, "application_scada_values_read": 0,
    })
    group_results: dict[str, Any] = {}
    passed_groups: list[str] = []
    for group in TARGET_COLS:
        application = YEAR_2023 if group in TARGET_COLS[:2] else G3_H2
        candidate = candidates_g12[group] if group in TARGET_COLS[:2] else candidates_g3[group]
        comparison = bayes._comparison(
            score_labels.loc[application, group], baseline.loc[application, group], candidate,
            group, _segments(group),
        )
        gate = _stage1_gate(comparison, REQUIRED[group])
        if gate["passed"]:
            passed_groups.append(group)
        robust = _robust_monthly_diagnostic(
            score_labels.loc[application, group], baseline.loc[application, group], candidate, group
        )
        group_results[group] = {"comparison": comparison, "gate": gate, "robust_full_period_diagnostic": robust}
    if sum(len(REQUIRED[group]) for group in TARGET_COLS) != 17:
        raise AssertionError("registered Stage1 metric count changed")
    if json.dumps(closure_before, sort_keys=True, default=str) != json.dumps(
        source_closure(raw_dir), sort_keys=True, default=str
    ):
        raise AssertionError("source closure changed during Stage1")
    result = {
        "schema_version": 1, "created_utc": utc_now(), "preregister_sha256": CONFIG_SHA256,
        "risk": config["risk_classification"], "global_candidate_lock": describe_file(global_lock),
        "score_label_access": describe_file(score_access),
        "training": {"g12": training_g12, "g3": training_g3},
        "group_results": group_results, "passed_groups": passed_groups,
        "stage2_unlocked": bool(passed_groups), "registered_metric_records": 17,
        "robust_monthly_records": 30, "robust_diagnostics_used_for_selection": False,
        "validation_or_test_scada_read": False, "current_observation_inference": False,
        "2024_weather_or_label_read": False, "2025_read": False, "csv_created": False,
    }
    result_path = out_dir / "stage1_results.json"
    bayes._write_json(result_path, result)
    promotion = out_dir / "stage1_promotion_lock.json"
    bayes._write_json(promotion, {
        "schema_version": 1, "created_utc": utc_now(), "preregister_sha256": CONFIG_SHA256,
        "global_candidate_lock": describe_file(global_lock), "score_label_access": describe_file(score_access),
        "stage1_results": describe_file(result_path), "passed_groups": passed_groups,
        "stage2_unlocked": bool(passed_groups), "no_2024_or_2025_read": True,
    })
    if not passed_groups:
        bayes._write_json(out_dir / "stage1_rejection.json", {
            "schema_version": 1, "created_utc": utc_now(), "preregister_sha256": CONFIG_SHA256,
            "promotion_lock": describe_file(promotion), "decision": "REJECT_IDENTITY",
            "no_retune_rescue_2024_2025_or_csv": True,
        })
    print(f"Stage1 passed groups: {passed_groups}", flush=True)
    return result


def write_manifest(*, out_dir: Path, raw_dir: Path, config: Mapping[str, Any], runtime: Mapping[str, Any]) -> dict[str, Any]:
    path = out_dir / "manifest.json"
    sidecar = out_dir / "manifest.sha256"
    if path.exists() or sidecar.exists():
        raise FileExistsError("manifest exists")
    result = json.loads((out_dir / "stage1_results.json").read_text(encoding="utf-8"))
    outputs = sorted(item for item in out_dir.rglob("*") if item.is_file() and item not in (path, sidecar))
    payload = {
        "schema_version": 1,
        "artifact_type": "raw_spatial_grid_temporal_cnn_scada_multitask_strict_v1_stage1",
        "created_utc": utc_now(), "command": [sys.executable, *sys.argv],
        "preregister_sha256": CONFIG_SHA256, "feasibility_sha256": FEASIBILITY_SHA256,
        "risk": config["risk_classification"], "source_closure": source_closure(raw_dir),
        "selection": {"passed_groups": result["passed_groups"], "stage2_unlocked": result["stage2_unlocked"]},
        "runtime": {"registered": dict(runtime), "packages": package_versions(), "git": git_state(ROOT)},
        "read_flags": {"application_scada": False, "2024_weather": False, "2024_labels": False, "2025": False, "sample": False, "csv": False},
        "tests": {"focused_expected": 14, "focused_status": "pass"},
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
    owner["stage"] = "RAW_SPATIAL_CNN_SCADA_MULTITASK_STAGE1_GPU"
    temporary = HEAVY_GUARD_PATH.with_name(f".{HEAVY_GUARD_PATH.name}.owner-{owner['pid']}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(owner, stream, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, HEAVY_GUARD_PATH)
    atexit.register(strict_protocol.release_heavy_guard, HEAVY_GUARD_PATH, owner)
    try:
        if args.stage in ("stage1", "all"):
            run_stage1(
                raw_dir=args.raw_dir.resolve(), artifact_root=args.artifact_root.resolve(),
                out_dir=args.out_dir.resolve(), config=config, raw_contract=raw_contract,
                runtime=runtime, owner=owner,
            )
        if args.stage in ("manifest", "all"):
            write_manifest(out_dir=args.out_dir.resolve(), raw_dir=args.raw_dir.resolve(), config=config, runtime=runtime)
    finally:
        strict_protocol.release_heavy_guard(HEAVY_GUARD_PATH, owner)


if __name__ == "__main__":
    main()

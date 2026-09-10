"""Strict-forward all-node LDAPS/GFS LightGBM experiment."""

from __future__ import annotations

import argparse
import io
import json
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

from scripts import run_ficr_bayes_decision_strict as bayes  # noqa: E402
from scripts import run_shared_q07_multiseed as shared  # noqa: E402
from scripts import run_weather_quantile_bayes as weather_protocol  # noqa: E402
from src.manifest import (  # noqa: E402
    describe_file,
    git_state,
    package_versions,
    sha256_file,
    utc_now,
)
from src.metric import CAPACITY_KWH, TARGET_COLS  # noqa: E402
from src.raw_grid_wind import (  # noqa: E402
    AVAILABLE_COL,
    BLEND_WEIGHTS,
    CANDIDATE_KEYS,
    COORD_COLS,
    COMMON_MODEL_PARAMETERS,
    EXPECTED_FEATURE_COUNT,
    GRID_COL,
    OBJECTIVES,
    RawGridWindRegressor,
    SOURCE_CHANNELS,
    TIME_COL,
    assert_fit_before_apply,
    blend_raw_prediction,
    build_raw_grid_wind_features,
    canonical_sha256,
    select_group_candidate,
)


PREREGISTER_SHA256 = "06b56426347f64d91f1a69e1c56083014f0dd003ee8186fa98ffe97ad3b61d01"
FEASIBILITY_SHA256 = "2d290cb4c6dd5ea6bedb2b1fb1f8c89d8a8b99ebed34ab387cc035b63d975a60"
STAGE1_REQUIRED: Mapping[str, tuple[str, ...]] = {
    "kpx_group_1": ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4"),
    "kpx_group_2": ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4"),
    "kpx_group_3": ("full", "Q3", "Q4"),
}
STAGE2_REQUIRED = ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4")
YEAR_2022 = pd.date_range(
    "2022-01-01 01:00:00", "2023-01-01 00:00:00", freq="h", name=TIME_COL
)
YEAR_2023 = pd.date_range(
    "2023-01-01 01:00:00", "2024-01-01 00:00:00", freq="h", name=TIME_COL
)
YEAR_2024 = pd.date_range(
    "2024-01-01 01:00:00", "2025-01-01 00:00:00", freq="h", name=TIME_COL
)
YEAR_2025 = pd.date_range(
    "2025-01-01 01:00:00", "2026-01-01 00:00:00", freq="h", name=TIME_COL
)
PRE2024 = YEAR_2022.append(YEAR_2023)
TRAIN_ALL = PRE2024.append(YEAR_2024)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("stage1", "stage2", "final", "all"), required=True)
    parser.add_argument("--raw-dir", type=Path, default=Path(r"data/local/open"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/postgate/raw_grid_wind_lgb_strict_v1"),
    )
    parser.add_argument(
        "--preregister",
        type=Path,
        default=Path("configs/raw_grid_wind_lgb_preregister_v1.json"),
    )
    return parser.parse_args(argv)


def _verify_preregister(path: Path) -> dict[str, Any]:
    observed = sha256_file(path)
    if observed != PREREGISTER_SHA256:
        raise AssertionError(f"preregister hash changed: {observed}")
    sidecar = path.with_suffix(".sha256")
    expected_sidecar = f"{PREREGISTER_SHA256}  {path.name}\n"
    if sidecar.read_text(encoding="utf-8") != expected_sidecar:
        raise AssertionError("preregister SHA sidecar changed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["experiment_id"] != "raw_grid_wind_lgb_strict_forward_v1":
        raise AssertionError("experiment id changed")
    feature = payload["feature_contract"]
    if int(feature["ordered_feature_count"]) != EXPECTED_FEATURE_COUNT:
        raise AssertionError("feature count changed")
    if tuple(feature["ldaps_channels"]) != SOURCE_CHANNELS["ldaps"]:
        raise AssertionError("LDAPS channels changed")
    if tuple(feature["gfs_channels"]) != SOURCE_CHANNELS["gfs"]:
        raise AssertionError("GFS channels changed")
    family = payload["model_family"]
    if tuple(family["objectives_in_fixed_order"]) != OBJECTIVES:
        raise AssertionError("objectives changed")
    if family["common_parameters"] != dict(COMMON_MODEL_PARAMETERS):
        raise AssertionError("LightGBM parameters changed")
    candidates = payload["candidate_family"]
    if tuple(map(float, candidates["fixed_blend_weights"])) != BLEND_WEIGHTS:
        raise AssertionError("blend weights changed")
    if int(candidates["candidate_count_per_group"]) != len(CANDIDATE_KEYS):
        raise AssertionError("candidate count changed")
    feasibility = PROJECT_DIR / "configs/raw_grid_wind_lgb_feasibility_v1.json"
    feasibility_sidecar = feasibility.with_suffix(".sha256")
    if sha256_file(feasibility) != FEASIBILITY_SHA256:
        raise AssertionError("feasibility record changed")
    if feasibility_sidecar.read_text(encoding="utf-8") != (
        f"{FEASIBILITY_SHA256}  {feasibility.name}\n"
    ):
        raise AssertionError("feasibility sidecar changed")
    feasibility_payload = json.loads(feasibility.read_text(encoding="utf-8"))
    if feasibility_payload["preregister_sha256"] != PREREGISTER_SHA256:
        raise AssertionError("feasibility/preregister binding changed")
    return payload


def _provenance_paths(preregister_path: Path) -> dict[str, Path]:
    return {
        "runner": Path(__file__).resolve(),
        "raw_grid_module": PROJECT_DIR / "src/raw_grid_wind.py",
        "metric": PROJECT_DIR / "src/metric.py",
        "manifest": PROJECT_DIR / "src/manifest.py",
        "bounded_reader": PROJECT_DIR / "scripts/run_shared_q07_multiseed.py",
        "score_helper": PROJECT_DIR / "scripts/run_ficr_bayes_decision_strict.py",
        "baseline_helper": PROJECT_DIR / "scripts/run_weather_quantile_bayes.py",
        "focused_test": PROJECT_DIR / "tests/test_raw_grid_wind.py",
        "preregister": preregister_path.resolve(),
        "preregister_sidecar": preregister_path.with_suffix(".sha256").resolve(),
        "feasibility": PROJECT_DIR / "configs/raw_grid_wind_lgb_feasibility_v1.json",
        "feasibility_sidecar": PROJECT_DIR
        / "configs/raw_grid_wind_lgb_feasibility_v1.sha256",
        "prior_failed_run_incident_ledger": PROJECT_DIR
        / "artifacts/postgate/raw_grid_wind_lgb_strict_v1_failed_json_order_audit/INCIDENT_LEDGER.json",
    }


def _snapshot_named(paths: Mapping[str, Path]) -> dict[str, Any]:
    return {name: shared._snapshot_file(path) for name, path in paths.items()}


def _stage1_baseline_causal_paths(artifact_root: Path) -> dict[str, Path]:
    """Best available content-addressed fit/recipe provenance for locked-v3 OOF."""

    models = artifact_root / "models"
    return {
        "locked_v3_recipe": PROJECT_DIR / "configs/train_final.v3.locked.json",
        "dev_training_runner": PROJECT_DIR / "scripts/train_dev.py",
        "dev_locked_v3_metric_ledger": artifact_root / "logs/dev2023_locked_v3.json",
        "dev_lgb_l1_fit": models / "dev2023_lgb_l1_eligible_1500.joblib",
        "dev_lgb_q07_fit": models / "dev2023_lgb_q07_eligible.joblib",
        "dev_lgb_q06_fit": models / "dev2023_lgb_q06_eligible.joblib",
        "dev_xgb_l1_fit": models / "dev2023_xgb_l1_eligible.joblib",
        "dev_cat_mae_fit": models / "dev2023_cat_mae_eligible.joblib",
        "dev_extra_fit": models / "dev2023_extra_eligible.joblib",
        "dev_shared_l1_fit": models / "dev2023_shared_l1_eligible.joblib",
        "dev_shared_q07_fit": models / "dev2023_shared_q07_eligible.joblib",
        "dev_energy_q06_fit": models / "dev2023_lgb_q06_energywt_eligible.joblib",
        "dev_top200_q07_fit": models / "dev2023_lgb_top200_q07_eligible.joblib",
        "g3_h1_fit_h2_apply_models": models / "g3dev2023h2_candidates.joblib",
    }


def _assert_file_identity(path: Path, spec: Mapping[str, Any], name: str) -> dict[str, Any]:
    observed = describe_file(path)
    if observed["size_bytes"] != int(spec["bytes"]) or observed["sha256"] != spec["sha256"]:
        raise AssertionError(f"{name} identity changed")
    return observed


def _prefix_snapshot(path: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    digest, size = shared._csv_prefix_identity(
        path, data_rows=int(spec["data_rows"]), byte_limit=int(spec["bytes"])
    )
    if digest != spec["sha256"] or size != int(spec["bytes"]):
        raise AssertionError(f"physical prefix changed: {path}")
    return {
        "snapshot_kind": "physical_csv_prefix",
        "path": str(path.resolve()),
        "data_rows": int(spec["data_rows"]),
        "size_bytes": size,
        "sha256": digest,
    }


def _read_bounded_csv(
    path: Path,
    spec: Mapping[str, Any],
    *,
    usecols: Sequence[str],
    whole_file: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = int(spec["rows"] if whole_file else spec["data_rows"])
    byte_limit = int(spec["bytes"])
    if whole_file:
        if path.stat().st_size != byte_limit or sha256_file(path) != spec["sha256"]:
            raise AssertionError(f"whole-file identity changed: {path}")
    else:
        _prefix_snapshot(path, spec)
    bounded = shared._BoundedRawReader(path, byte_limit=byte_limit)
    try:
        with io.BufferedReader(bounded, buffer_size=1024 * 1024) as stream:
            frame = pd.read_csv(
                stream,
                encoding="utf-8-sig",
                usecols=list(usecols),
                memory_map=False,
            )
            bytes_returned = bounded.bytes_returned
            underlying_position = bounded.underlying_position
    finally:
        bounded.close()
    if bytes_returned != byte_limit or underlying_position != byte_limit:
        raise AssertionError("parser crossed or failed to consume the physical byte cap")
    if len(frame) != rows or tuple(frame.columns) != tuple(usecols):
        raise AssertionError(f"bounded schema/row count changed for {path}")
    next_field: str | None = None
    if not whole_file:
        next_field = shared._next_csv_first_field(
            path, after_data_rows=rows, prefix_bytes=byte_limit
        )
        if next_field != spec["next_row_first_field_only"]:
            raise AssertionError("next-row timestamp changed")
    evidence = {
        "path": str(path.resolve()),
        "whole_file": whole_file,
        "data_rows": rows,
        "physical_byte_limit": byte_limit,
        "physical_sha256": spec["sha256"],
        "bytes_returned_to_parser": bytes_returned,
        "underlying_position_after_read": underlying_position,
        "materialized_columns": list(frame.columns),
        "next_row_first_field_only": next_field,
    }
    return frame, evidence


def _weather_usecols(source: str) -> tuple[str, ...]:
    return (TIME_COL, AVAILABLE_COL, GRID_COL, *COORD_COLS, *SOURCE_CHANNELS[source])


def _read_weather_features(
    raw_dir: Path,
    preregister: Mapping[str, Any],
    *,
    period: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if period == "stage1":
        container = preregister["physical_stage1_inputs"]
        source_specs = {
            "ldaps": container["ldaps_weather_prefix"],
            "gfs": container["gfs_weather_prefix"],
        }
        paths = {
            source: raw_dir / "train" / f"{source}_train.csv" for source in source_specs
        }
        whole = False
        expected_index = PRE2024
    elif period == "train_all":
        container = preregister["full_input_identities_after_promotion_only"]
        source_specs = {
            "ldaps": container["ldaps_train"],
            "gfs": container["gfs_train"],
        }
        paths = {
            source: raw_dir / "train" / f"{source}_train.csv" for source in source_specs
        }
        whole = True
        expected_index = TRAIN_ALL
    elif period == "test":
        container = preregister["full_input_identities_after_promotion_only"]
        source_specs = {
            "ldaps": container["ldaps_test"],
            "gfs": container["gfs_test"],
        }
        paths = {
            source: raw_dir / "test" / f"{source}_test.csv" for source in source_specs
        }
        whole = True
        expected_index = YEAR_2025
    else:
        raise ValueError(period)
    raw: dict[str, pd.DataFrame] = {}
    ledgers: dict[str, Any] = {}
    for source in ("ldaps", "gfs"):
        raw[source], ledgers[source] = _read_bounded_csv(
            paths[source], source_specs[source], usecols=_weather_usecols(source), whole_file=whole
        )
    features, feature_metadata = build_raw_grid_wind_features(
        raw["ldaps"], raw["gfs"], preregister["grid_catalog"]
    )
    if not features.index.equals(expected_index):
        raise AssertionError(f"{period} raw weather timestamp sequence changed")
    if tuple(features.columns) != tuple(feature_metadata["ordered_columns"]):
        raise AssertionError("feature metadata columns changed")
    return features, {"physical_reads": ledgers, "feature_metadata": feature_metadata}


def _read_label_prefix(
    raw_dir: Path,
    spec: Mapping[str, Any],
    groups: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = raw_dir / "train/train_labels.csv"
    usecols = ("kst_dtm", *groups)
    frame, evidence = _read_bounded_csv(path, spec, usecols=usecols, whole_file=False)
    frame.index = pd.DatetimeIndex(
        pd.to_datetime(frame.pop("kst_dtm"), errors="raise"), name=TIME_COL
    )
    if frame.index.min() != YEAR_2022.min() or frame.index.max() != pd.Timestamp(spec["end"]):
        raise AssertionError("bounded label timestamp sequence changed")
    evidence.update(
        {
            "target_columns_materialized": list(groups),
            "omitted_target_columns": [group for group in TARGET_COLS if group not in groups],
        }
    )
    return frame.astype(float), evidence


def _read_full_labels(
    raw_dir: Path,
    preregister: Mapping[str, Any],
    groups: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = preregister["full_input_identities_after_promotion_only"]["train_labels"]
    path = raw_dir / "train/train_labels.csv"
    usecols = ("kst_dtm", *groups)
    frame, evidence = _read_bounded_csv(path, spec, usecols=usecols, whole_file=True)
    frame.index = pd.DatetimeIndex(
        pd.to_datetime(frame.pop("kst_dtm"), errors="raise"), name=TIME_COL
    )
    if not frame.index.equals(TRAIN_ALL):
        raise AssertionError("full label timestamp sequence changed")
    evidence["target_columns_materialized"] = list(groups)
    return frame.astype(float), evidence


def _stage1_baseline(artifact_root: Path, preregister: Mapping[str, Any]) -> pd.DataFrame:
    baselines = preregister["baseline_contract"]
    _assert_file_identity(
        artifact_root / "oof/dev2023_locked_v3.parquet",
        baselines["stage1_group_1_and_2"],
        "Stage1 G1/G2 baseline",
    )
    _assert_file_identity(
        artifact_root / "oof/g3dev2023h2_candidates.parquet",
        baselines["stage1_group_3"],
        "Stage1 G3 baseline",
    )
    baseline, _ = weather_protocol._load_stage1_baseline(artifact_root)
    return baseline


def _candidate_frame(
    baseline: pd.Series,
    raw_predictions: Mapping[str, pd.Series],
    group: str,
) -> pd.DataFrame:
    result = pd.DataFrame(index=baseline.index)
    for objective in OBJECTIVES:
        for weight in BLEND_WEIGHTS:
            key = f"{objective}_w{int(round(weight * 100)):02d}"
            result[key] = blend_raw_prediction(
                baseline,
                raw_predictions[objective],
                weight=weight,
                capacity_kwh=CAPACITY_KWH[group],
            )
    if tuple(result.columns) != CANDIDATE_KEYS:
        raise AssertionError("candidate order changed")
    identity = blend_raw_prediction(
        baseline,
        raw_predictions[OBJECTIVES[0]],
        weight=0.0,
        capacity_kwh=CAPACITY_KWH[group],
    )
    if np.ascontiguousarray(identity.to_numpy()).tobytes() != np.ascontiguousarray(
        baseline.to_numpy()
    ).tobytes():
        raise AssertionError("weight-zero baseline is not value-bit exact")
    return result


def _fit_group(
    *,
    features: pd.DataFrame,
    actual_kwh: pd.Series,
    baseline: pd.Series,
    group: str,
    fit_index: pd.DatetimeIndex,
    application_index: pd.DatetimeIndex,
    objectives: Sequence[str],
) -> tuple[
    dict[str, RawGridWindRegressor],
    dict[str, pd.Series],
    pd.DataFrame,
    dict[str, Any],
]:
    split = assert_fit_before_apply(fit_index, application_index)
    models: dict[str, RawGridWindRegressor] = {}
    raw_predictions: dict[str, pd.Series] = {}
    metadata: dict[str, Any] = {"split": split, "models": {}}
    for objective in objectives:
        print(
            f"fit raw-grid {group}/{objective}: {fit_index.min()} -> {application_index.min()}",
            flush=True,
        )
        model = RawGridWindRegressor(objective).fit(
            features.loc[fit_index],
            actual_kwh.loc[fit_index] / CAPACITY_KWH[group],
        )
        prediction = model.predict(features.loc[application_index])
        models[objective] = model
        raw_predictions[objective] = prediction
        metadata["models"][objective] = model.metadata()
    candidates = _candidate_frame(baseline.loc[application_index], raw_predictions, group)
    return models, raw_predictions, candidates, metadata


def _save_and_reload_group(
    *,
    out_dir: Path,
    stage: str,
    group: str,
    models: Mapping[str, RawGridWindRegressor],
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
    loaded: dict[str, RawGridWindRegressor] = joblib.load(model_path)
    if tuple(loaded) != tuple(models):
        raise AssertionError("reloaded objective order changed")
    objective_checks: dict[str, Any] = {}
    for objective in models:
        repeated = loaded[objective].predict(application_features)
        exact = np.array_equal(
            repeated.to_numpy(dtype=float), raw_predictions[objective].to_numpy(dtype=float)
        )
        median_exact = np.array_equal(loaded[objective].medians_, models[objective].medians_)
        if not exact or not median_exact:
            raise AssertionError(f"{stage}/{group}/{objective} reload changed predictions")
        objective_checks[objective] = {
            "prediction_bit_exact": exact,
            "training_median_bit_exact": median_exact,
            "metadata": loaded[objective].metadata(),
        }
    return [model_path, raw_path, candidate_path, baseline_path], {
        "objectives": objective_checks,
        "candidate_columns": list(candidates.columns),
        "weight_zero_identity_value_bits_exact": True,
    }


def _stage1_segments(group: str) -> dict[str, pd.DatetimeIndex]:
    segments = weather_protocol._year_segments(2023)
    if group == "kpx_group_3":
        return {"full": segments["H2"], "Q3": segments["Q3"], "Q4": segments["Q4"]}
    return {name: segments[name] for name in STAGE1_REQUIRED[group]}


def _score_group_candidates(
    actual: pd.Series,
    baseline: pd.Series,
    candidates: pd.DataFrame,
    group: str,
    segments: Mapping[str, pd.DatetimeIndex],
) -> dict[str, Any]:
    return {
        key: bayes._comparison(actual, baseline, candidates[key], group, segments)
        for key in CANDIDATE_KEYS
    }


def _stage1(
    *,
    raw_dir: Path,
    artifact_root: Path,
    out_dir: Path,
    preregister_path: Path,
    preregister: Mapping[str, Any],
) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"Stage1 requires a new output directory: {out_dir}")
    out_dir.mkdir(parents=True)
    bayes._copy_exclusive(preregister_path, out_dir / "preregister.json")
    bayes._copy_exclusive(preregister_path.with_suffix(".sha256"), out_dir / "preregister.sha256")
    provenance_before = _snapshot_named(_provenance_paths(preregister_path))
    prefix_specs = preregister["physical_stage1_inputs"]
    stage1_inputs_before = {
        "labels_g12_fit": _prefix_snapshot(
            raw_dir / "train/train_labels.csv", prefix_specs["labels_g12_fit_prefix"]
        ),
        "labels_g3_fit": _prefix_snapshot(
            raw_dir / "train/train_labels.csv", prefix_specs["labels_g3_fit_prefix"]
        ),
        "labels_score": _prefix_snapshot(
            raw_dir / "train/train_labels.csv",
            prefix_specs["labels_stage1_score_prefix_after_lock"],
        ),
        "ldaps_weather": _prefix_snapshot(
            raw_dir / "train/ldaps_train.csv", prefix_specs["ldaps_weather_prefix"]
        ),
        "gfs_weather": _prefix_snapshot(
            raw_dir / "train/gfs_train.csv", prefix_specs["gfs_weather_prefix"]
        ),
    }
    baseline_specs = preregister["baseline_contract"]
    stage1_inputs_before["baseline_g12"] = _assert_file_identity(
        artifact_root / "oof/dev2023_locked_v3.parquet",
        baseline_specs["stage1_group_1_and_2"],
        "Stage1 G1/G2 baseline",
    )
    stage1_inputs_before["baseline_g3"] = _assert_file_identity(
        artifact_root / "oof/g3dev2023h2_candidates.parquet",
        baseline_specs["stage1_group_3"],
        "Stage1 G3 baseline",
    )
    stage1_inputs_before["baseline_causal_provenance"] = _snapshot_named(
        _stage1_baseline_causal_paths(artifact_root)
    )

    features, weather_evidence = _read_weather_features(
        raw_dir, preregister, period="stage1"
    )
    baseline = _stage1_baseline(artifact_root, preregister)
    fit_indexes = {
        "kpx_group_1": YEAR_2022,
        "kpx_group_2": YEAR_2022,
        "kpx_group_3": weather_protocol._year_segments(2023)["H1"],
    }
    application_indexes = {
        "kpx_group_1": YEAR_2023,
        "kpx_group_2": YEAR_2023,
        "kpx_group_3": weather_protocol._year_segments(2023)["H2"],
    }

    labels_g12, labels_g12_evidence = _read_label_prefix(
        raw_dir,
        prefix_specs["labels_g12_fit_prefix"],
        TARGET_COLS[:2],
    )
    group_models: dict[str, dict[str, RawGridWindRegressor]] = {}
    group_raw: dict[str, dict[str, pd.Series]] = {}
    group_candidates: dict[str, pd.DataFrame] = {}
    training: dict[str, Any] = {}
    all_candidate_paths: list[Path] = []
    reload_audits: dict[str, Any] = {}
    g12_paths: list[Path] = []
    for group in TARGET_COLS[:2]:
        models, raw_predictions, candidates, metadata = _fit_group(
            features=features,
            actual_kwh=labels_g12[group],
            baseline=baseline[group],
            group=group,
            fit_index=fit_indexes[group],
            application_index=application_indexes[group],
            objectives=OBJECTIVES,
        )
        paths, reload = _save_and_reload_group(
            out_dir=out_dir,
            stage="stage1",
            group=group,
            models=models,
            raw_predictions=raw_predictions,
            candidates=candidates,
            baseline=baseline.loc[application_indexes[group], group],
            application_features=features.loc[application_indexes[group]],
        )
        group_models[group] = models
        group_raw[group] = raw_predictions
        group_candidates[group] = candidates
        training[group] = metadata
        reload_audits[group] = reload
        g12_paths.extend(paths)
        all_candidate_paths.extend(paths)
    g12_lock_path = out_dir / "stage1_g12_prescore_lock.json"
    bayes._write_json(
        g12_lock_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "fit_label_evidence": labels_g12_evidence,
            "fit_groups": list(TARGET_COLS[:2]),
            "application_target_value_cells_materialized": 0,
            "outputs": [describe_file(path) for path in g12_paths],
            "reload_audit": {group: reload_audits[group] for group in TARGET_COLS[:2]},
            "created_before_g3_fit_label_prefix_and_all_application_labels": True,
        },
    )

    labels_g3, labels_g3_evidence = _read_label_prefix(
        raw_dir,
        prefix_specs["labels_g3_fit_prefix"],
        ("kpx_group_3",),
    )
    group = "kpx_group_3"
    models, raw_predictions, candidates, metadata = _fit_group(
        features=features,
        actual_kwh=labels_g3[group],
        baseline=baseline[group],
        group=group,
        fit_index=fit_indexes[group],
        application_index=application_indexes[group],
        objectives=OBJECTIVES,
    )
    g3_paths, reload = _save_and_reload_group(
        out_dir=out_dir,
        stage="stage1",
        group=group,
        models=models,
        raw_predictions=raw_predictions,
        candidates=candidates,
        baseline=baseline.loc[application_indexes[group], group],
        application_features=features.loc[application_indexes[group]],
    )
    group_models[group] = models
    group_raw[group] = raw_predictions
    group_candidates[group] = candidates
    training[group] = metadata
    reload_audits[group] = reload
    all_candidate_paths.extend(g3_paths)
    g3_lock_path = out_dir / "stage1_g3_prescore_lock.json"
    bayes._write_json(
        g3_lock_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "fit_label_evidence": labels_g3_evidence,
            "fit_group": group,
            "g12_target_value_cells_materialized_by_g3_read": 0,
            "application_target_value_cells_materialized": 0,
            "outputs": [describe_file(path) for path in g3_paths],
            "reload_audit": reload,
            "created_before_all_stage1_application_labels": True,
        },
    )

    provenance_after = _snapshot_named(_provenance_paths(preregister_path))
    stage1_inputs_after = {
        "labels_g12_fit": _prefix_snapshot(
            raw_dir / "train/train_labels.csv", prefix_specs["labels_g12_fit_prefix"]
        ),
        "labels_g3_fit": _prefix_snapshot(
            raw_dir / "train/train_labels.csv", prefix_specs["labels_g3_fit_prefix"]
        ),
        "labels_score": _prefix_snapshot(
            raw_dir / "train/train_labels.csv",
            prefix_specs["labels_stage1_score_prefix_after_lock"],
        ),
        "ldaps_weather": _prefix_snapshot(
            raw_dir / "train/ldaps_train.csv", prefix_specs["ldaps_weather_prefix"]
        ),
        "gfs_weather": _prefix_snapshot(
            raw_dir / "train/gfs_train.csv", prefix_specs["gfs_weather_prefix"]
        ),
        "baseline_g12": _assert_file_identity(
            artifact_root / "oof/dev2023_locked_v3.parquet",
            baseline_specs["stage1_group_1_and_2"],
            "Stage1 G1/G2 baseline",
        ),
        "baseline_g3": _assert_file_identity(
            artifact_root / "oof/g3dev2023h2_candidates.parquet",
            baseline_specs["stage1_group_3"],
            "Stage1 G3 baseline",
        ),
        "baseline_causal_provenance": _snapshot_named(
            _stage1_baseline_causal_paths(artifact_root)
        ),
    }
    shared._assert_snapshot_equal(provenance_before, provenance_after, name="provenance")
    shared._assert_snapshot_equal(stage1_inputs_before, stage1_inputs_after, name="Stage1 inputs")
    prescore_record_path = out_dir / "stage1_prescore_record.json"
    bayes._write_json(
        prescore_record_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "weather_evidence": weather_evidence,
            "training": training,
            "reload_audits": reload_audits,
            "candidate_keys": list(CANDIDATE_KEYS),
            "candidate_artifacts": [describe_file(path) for path in all_candidate_paths],
            "g12_lock": describe_file(g12_lock_path),
            "g3_lock": describe_file(g3_lock_path),
            "stage1_application_target_value_cells_materialized": 0,
            "multi_year_feature_cache_opened": False,
            "2024_read": False,
            "2025_read": False,
            "provenance_before": provenance_before,
            "provenance_after": provenance_after,
            "stage1_inputs_before": stage1_inputs_before,
            "stage1_inputs_after": stage1_inputs_after,
        },
    )
    global_lock_path = out_dir / "stage1_global_prescore_lock.json"
    bayes._write_json(
        global_lock_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "prescore_record": describe_file(prescore_record_path),
            "g12_lock": describe_file(g12_lock_path),
            "g3_lock": describe_file(g3_lock_path),
            "candidate_artifacts": [describe_file(path) for path in all_candidate_paths],
            "model_reload_and_all_six_candidates_frozen": True,
            "created_before_any_stage1_application_target_value": True,
            "2024_read": False,
            "2025_read": False,
        },
    )

    score_labels, score_label_evidence = _read_label_prefix(
        raw_dir,
        prefix_specs["labels_stage1_score_prefix_after_lock"],
        TARGET_COLS,
    )
    group_results: dict[str, Any] = {}
    selected: dict[str, str | None] = {}
    for group in TARGET_COLS:
        app_index = application_indexes[group]
        segments = _stage1_segments(group)
        comparisons = _score_group_candidates(
            score_labels.loc[app_index, group],
            baseline.loc[app_index, group],
            group_candidates[group],
            group,
            segments,
        )
        selected[group], selection = select_group_candidate(
            comparisons, STAGE1_REQUIRED[group]
        )
        group_results[group] = {
            "comparisons": comparisons,
            "comparisons_sha256": canonical_sha256(comparisons),
            "selection": selection,
            "selection_sha256": canonical_sha256(selection),
            "locked_candidate": selected[group] or "identity",
        }
    passed_groups = [group for group in TARGET_COLS if selected[group] is not None]
    results = {
        "schema_version": 1,
        "experiment_id": preregister["experiment_id"],
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "global_prescore_lock": describe_file(global_lock_path),
        "score_label_evidence_after_prescore_lock": score_label_evidence,
        "group_results": group_results,
        "passed_groups": passed_groups,
        "passed_groups_sha256": canonical_sha256(passed_groups),
        "2024_read": False,
        "2025_read": False,
        "leaderboard_score_claim": False,
    }
    result_path = out_dir / "stage1_results.json"
    bayes._write_json(result_path, results)
    lock = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "global_prescore_lock": describe_file(global_lock_path),
        "stage1_results": describe_file(result_path),
        "locked_candidates": {
            group: selected[group] or "identity" for group in TARGET_COLS
        },
        "passed_groups": passed_groups,
        "passed_groups_sha256": results["passed_groups_sha256"],
        "only_passing_groups_may_open_2024": True,
        "no_2024_reselection_or_retuning": True,
        "2024_read": False,
        "2025_read": False,
    }
    bayes._write_json(out_dir / "stage1_promotion_lock.json", lock)
    print(f"Stage1 passed groups: {passed_groups}", flush=True)
    return lock


def _load_stage1_lock(out_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    lock_path = out_dir / "stage1_promotion_lock.json"
    result_path = out_dir / "stage1_results.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    results = json.loads(result_path.read_text(encoding="utf-8"))
    if lock["preregister_sha256"] != PREREGISTER_SHA256:
        raise AssertionError("Stage1 preregister changed")
    if lock["stage1_results"]["sha256"] != sha256_file(result_path):
        raise AssertionError("Stage1 result changed")
    recomputed: dict[str, str] = {}
    passed: list[str] = []
    for group in TARGET_COLS:
        group_result = results["group_results"][group]
        selected, selection = select_group_candidate(
            group_result["comparisons"], STAGE1_REQUIRED[group]
        )
        recomputed[group] = selected or "identity"
        if selected is not None:
            passed.append(group)
        if selection != group_result["selection"]:
            raise AssertionError(f"Stage1 selection changed for {group}")
    if recomputed != lock["locked_candidates"] or passed != lock["passed_groups"]:
        raise AssertionError("Stage1 lock cannot be independently reproduced")
    if canonical_sha256(passed) != lock["passed_groups_sha256"]:
        raise AssertionError("Stage1 passed-group hash changed")
    return lock, results


def _candidate_recipe(key: str) -> tuple[str, float]:
    if key not in CANDIDATE_KEYS:
        raise ValueError(f"unregistered candidate: {key}")
    objective, encoded = key.split("_w")
    return objective, int(encoded) / 100.0


def _fit_fixed_group(
    *,
    features: pd.DataFrame,
    actual_kwh: pd.Series,
    baseline: pd.Series,
    group: str,
    fit_index: pd.DatetimeIndex,
    application_index: pd.DatetimeIndex,
    candidate_key: str,
) -> tuple[
    dict[str, RawGridWindRegressor],
    dict[str, pd.Series],
    pd.DataFrame,
    dict[str, Any],
]:
    objective, weight = _candidate_recipe(candidate_key)
    split = assert_fit_before_apply(fit_index, application_index)
    print(
        f"fit fixed raw-grid {group}/{candidate_key}: {fit_index.min()} -> {application_index.min()}",
        flush=True,
    )
    model = RawGridWindRegressor(objective).fit(
        features.loc[fit_index], actual_kwh.loc[fit_index] / CAPACITY_KWH[group]
    )
    raw = model.predict(features.loc[application_index])
    candidate = blend_raw_prediction(
        baseline.loc[application_index],
        raw,
        weight=weight,
        capacity_kwh=CAPACITY_KWH[group],
    )
    return (
        {objective: model},
        {objective: raw},
        candidate.rename(candidate_key).to_frame(),
        {
            "split": split,
            "candidate_key": candidate_key,
            "objective": objective,
            "weight": weight,
            "model": model.metadata(),
        },
    )


def _stage2(
    *,
    raw_dir: Path,
    artifact_root: Path,
    out_dir: Path,
    preregister: Mapping[str, Any],
) -> dict[str, Any]:
    stage1_lock, _ = _load_stage1_lock(out_dir)
    passed_groups = list(stage1_lock["passed_groups"])
    if (out_dir / "stage2_results.json").exists():
        raise FileExistsError(out_dir / "stage2_results.json")
    if not passed_groups:
        result = {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "stage1_promotion_lock": describe_file(out_dir / "stage1_promotion_lock.json"),
            "performed": False,
            "reason": "no Stage1 group passed every registered slice",
            "stage1_passed_groups": [],
            "stage2_passed_groups": [],
            "all_three_groups_promoted": False,
            "2024_weather_read": False,
            "2024_label_read": False,
            "2025_read": False,
        }
        result_path = out_dir / "stage2_results.json"
        bayes._write_json(result_path, result)
        lock = {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "stage1_promotion_lock": describe_file(out_dir / "stage1_promotion_lock.json"),
            "stage2_results": describe_file(result_path),
            "stage2_passed_groups": [],
            "all_three_groups_promoted": False,
            "2024_read": False,
            "2025_read": False,
        }
        bayes._write_json(out_dir / "stage2_promotion_lock.json", lock)
        return lock

    features, weather_evidence = _read_weather_features(
        raw_dir, preregister, period="train_all"
    )
    fit_labels, fit_label_evidence = _read_label_prefix(
        raw_dir,
        preregister["physical_stage1_inputs"]["labels_stage1_score_prefix_after_lock"],
        passed_groups,
    )
    baseline_spec = preregister["baseline_contract"]["stage2_after_group_promotion_only"]
    baseline_path = artifact_root / "oof/gate2024_locked_v3_cf_fix.parquet"
    _assert_file_identity(baseline_path, baseline_spec, "Stage2 baseline")
    baseline = bayes._read_prediction(
        baseline_path, YEAR_2024, required_columns=TARGET_COLS
    )
    fixed_candidates: dict[str, pd.DataFrame] = {}
    training: dict[str, Any] = {}
    reload_audits: dict[str, Any] = {}
    output_paths: list[Path] = []
    for group in passed_groups:
        key = stage1_lock["locked_candidates"][group]
        fit_index = PRE2024 if group in TARGET_COLS[:2] else YEAR_2023
        models, raw_predictions, candidates, metadata = _fit_fixed_group(
            features=features,
            actual_kwh=fit_labels[group],
            baseline=baseline[group],
            group=group,
            fit_index=fit_index,
            application_index=YEAR_2024,
            candidate_key=key,
        )
        paths, reload = _save_and_reload_group(
            out_dir=out_dir,
            stage="stage2",
            group=group,
            models=models,
            raw_predictions=raw_predictions,
            candidates=candidates,
            baseline=baseline[group],
            application_features=features.loc[YEAR_2024],
        )
        fixed_candidates[group] = candidates
        training[group] = metadata
        reload_audits[group] = reload
        output_paths.extend(paths)
    prescore_record_path = out_dir / "stage2_prescore_record.json"
    bayes._write_json(
        prescore_record_path,
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
            "reload_audits": reload_audits,
            "outputs": [describe_file(path) for path in output_paths],
            "2024_application_target_columns_materialized": [],
            "created_before_2024_application_target_values": True,
            "2024_weather_read": True,
            "2024_label_read": False,
            "2025_read": False,
        },
    )
    prescore_lock_path = out_dir / "stage2_global_prescore_lock.json"
    bayes._write_json(
        prescore_lock_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "stage1_promotion_lock": describe_file(out_dir / "stage1_promotion_lock.json"),
            "prescore_record": describe_file(prescore_record_path),
            "candidate_artifacts": [describe_file(path) for path in output_paths],
            "fixed_models_medians_predictions_candidates_reload_frozen": True,
            "created_before_2024_application_target_values": True,
            "2025_read": False,
        },
    )

    score_labels, score_label_evidence = _read_full_labels(
        raw_dir, preregister, passed_groups
    )
    segments = {
        name: weather_protocol._year_segments(2024)[name] for name in STAGE2_REQUIRED
    }
    group_results: dict[str, Any] = {}
    stage2_passed: list[str] = []
    for group in passed_groups:
        key = stage1_lock["locked_candidates"][group]
        comparison = bayes._comparison(
            score_labels.loc[YEAR_2024, group],
            baseline[group],
            fixed_candidates[group][key],
            group,
            segments,
        )
        deltas: dict[str, float] = {}
        for name in STAGE2_REQUIRED:
            record = comparison[name]
            expected = float(record["candidate"]["score"]) - float(
                record["baseline"]["score"]
            )
            if float(record["delta"]) != expected:
                raise AssertionError("Stage2 delta arithmetic changed")
            deltas[name] = expected
        promoted = all(value > 0.0 for value in deltas.values())
        if promoted:
            stage2_passed.append(group)
        group_results[group] = {
            "locked_candidate": key,
            "comparison": comparison,
            "comparison_sha256": canonical_sha256(comparison),
            "minimum_delta": min(deltas.values()),
            "mean_delta": float(np.mean(list(deltas.values()))),
            "all_seven_strictly_positive": promoted,
        }
    all_three = stage2_passed == list(TARGET_COLS) and passed_groups == list(TARGET_COLS)
    result = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "stage1_promotion_lock": describe_file(out_dir / "stage1_promotion_lock.json"),
        "stage2_prescore_lock": describe_file(prescore_lock_path),
        "score_label_evidence_after_prescore_lock": score_label_evidence,
        "stage1_passed_groups": passed_groups,
        "group_results": group_results,
        "stage2_passed_groups": stage2_passed,
        "stage2_passed_groups_sha256": canonical_sha256(stage2_passed),
        "all_three_groups_promoted": all_three,
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
        "stage2_prescore_lock": describe_file(prescore_lock_path),
        "stage2_results": describe_file(result_path),
        "locked_candidates": {
            group: stage1_lock["locked_candidates"][group] for group in passed_groups
        },
        "stage2_passed_groups": stage2_passed,
        "stage2_passed_groups_sha256": result["stage2_passed_groups_sha256"],
        "all_three_groups_promoted": all_three,
        "no_2025_reselection_or_retuning": True,
        "2024_read": True,
        "2025_read": False,
    }
    bayes._write_json(out_dir / "stage2_promotion_lock.json", lock)
    print(f"Stage2 passed groups: {stage2_passed}; all-three={all_three}", flush=True)
    return lock


def _load_stage2_lock(out_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    lock_path = out_dir / "stage2_promotion_lock.json"
    result_path = out_dir / "stage2_results.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if lock["preregister_sha256"] != PREREGISTER_SHA256:
        raise AssertionError("Stage2 preregister changed")
    if lock["stage2_results"]["sha256"] != sha256_file(result_path):
        raise AssertionError("Stage2 result changed")
    if bool(lock["all_three_groups_promoted"]) != bool(result["all_three_groups_promoted"]):
        raise AssertionError("Stage2 promotion changed")
    if lock["stage2_passed_groups"] != result["stage2_passed_groups"]:
        raise AssertionError("Stage2 group promotion changed")
    return lock, result


def _final(
    *,
    raw_dir: Path,
    artifact_root: Path,
    out_dir: Path,
    preregister: Mapping[str, Any],
) -> dict[str, Any]:
    stage2_lock, _ = _load_stage2_lock(out_dir)
    if (out_dir / "final_results.json").exists():
        raise FileExistsError(out_dir / "final_results.json")
    if not bool(stage2_lock["all_three_groups_promoted"]):
        result = {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "stage2_promotion_lock": describe_file(out_dir / "stage2_promotion_lock.json"),
            "performed": False,
            "reason": "all three groups did not pass both strict forward gates",
            "2025_weather_read": False,
            "sample_submission_read": False,
            "submission_created": False,
            "leaderboard_score_claim": False,
        }
        bayes._write_json(out_dir / "final_results.json", result)
        return result

    train_features, train_weather_evidence = _read_weather_features(
        raw_dir, preregister, period="train_all"
    )
    labels, label_evidence = _read_full_labels(raw_dir, preregister, TARGET_COLS)
    test_features, test_weather_evidence = _read_weather_features(
        raw_dir, preregister, period="test"
    )
    baseline_spec = preregister["baseline_contract"][
        "final_after_all_group_promotion_only"
    ]
    baseline_path = artifact_root / "final_cf_fix/predictions/corrected_v3_test.parquet"
    _assert_file_identity(baseline_path, baseline_spec, "final corrected-v3 baseline")
    baseline = bayes._read_prediction(
        baseline_path, YEAR_2025, required_columns=TARGET_COLS
    )
    sample_spec = preregister["full_input_identities_after_promotion_only"][
        "sample_submission"
    ]
    sample_path = raw_dir / "sample_submission.csv"
    _assert_file_identity(sample_path, sample_spec, "sample submission")
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

    models: dict[str, RawGridWindRegressor] = {}
    raw_predictions = pd.DataFrame(index=YEAR_2025)
    prediction = pd.DataFrame(index=YEAR_2025, columns=TARGET_COLS, dtype=float)
    training: dict[str, Any] = {}
    for group in TARGET_COLS:
        candidate_key = stage2_lock["locked_candidates"][group]
        objective, weight = _candidate_recipe(candidate_key)
        model = RawGridWindRegressor(objective).fit(
            train_features,
            labels[group] / CAPACITY_KWH[group],
        )
        raw = model.predict(test_features)
        candidate = blend_raw_prediction(
            baseline[group], raw, weight=weight, capacity_kwh=CAPACITY_KWH[group]
        )
        models[group] = model
        raw_predictions[group] = raw
        prediction[group] = candidate
        training[group] = {
            "locked_candidate": candidate_key,
            "objective": objective,
            "weight": weight,
            "model": model.metadata(),
            "fit_start": TRAIN_ALL.min(),
            "fit_end": TRAIN_ALL.max(),
            "application_start": YEAR_2025.min(),
            "application_end": YEAR_2025.max(),
            "fit_end_before_application_start": True,
            "overlap_count": 0,
        }
    model_path = out_dir / "models/final_all_groups.joblib"
    raw_path = out_dir / "predictions/final_raw_cf_2025.parquet"
    prediction_path = out_dir / "predictions/raw_grid_wind_lgb_2025.parquet"
    csv_path = out_dir / "raw_grid_wind_lgb_2025.csv"
    bayes._atomic_joblib(models, model_path)
    bayes._atomic_parquet(raw_predictions, raw_path)
    bayes._atomic_parquet(prediction, prediction_path)
    loaded: dict[str, RawGridWindRegressor] = joblib.load(model_path)
    reload_checks: dict[str, Any] = {}
    for group in TARGET_COLS:
        repeated = loaded[group].predict(test_features)
        exact = np.array_equal(
            repeated.to_numpy(dtype=float), raw_predictions[group].to_numpy(dtype=float)
        )
        if not exact:
            raise AssertionError(f"final {group} model reload changed raw prediction")
        reload_checks[group] = {"raw_prediction_bit_exact": True}
    submission = sample.copy()
    for group in TARGET_COLS:
        submission[group] = prediction[group].to_numpy(dtype=float)
    bayes._atomic_csv(submission, csv_path)
    verification = bayes._verify_submission(csv_path, sample, prediction)
    result = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "stage2_promotion_lock": describe_file(out_dir / "stage2_promotion_lock.json"),
        "performed": True,
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
    artifact_root: Path,
    out_dir: Path,
    preregister_path: Path,
) -> dict[str, Any]:
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    stage1_lock, _ = _load_stage1_lock(out_dir)
    stage2_lock, stage2_result = _load_stage2_lock(out_dir)
    final_result = json.loads((out_dir / "final_results.json").read_text(encoding="utf-8"))
    outputs = sorted(
        path
        for path in out_dir.rglob("*")
        if path.is_file() and path != manifest_path and ".tmp-" not in path.name
    )
    payload = {
        "schema_version": 1,
        "artifact_type": "raw_grid_wind_lgb_strict_forward_v1",
        "created_utc": utc_now(),
        "command": [sys.executable, *sys.argv],
        "runtime": {"packages": package_versions(), "git": git_state(PROJECT_DIR)},
        "preregister_sha256": PREREGISTER_SHA256,
        "contracts": {
            "all_16_ldaps_and_9_gfs_node_identities_direct": True,
            "feature_count": EXPECTED_FEATURE_COUNT,
            "target_scada_residual_public_scale_features": False,
            "fit_before_apply": True,
            "application_labels_read_only_after_model_prediction_hash_locks": True,
            "passed_groups_only_enter_2024": True,
            "all_three_groups_required_for_2025": True,
            "leaderboard_score_claim": False,
        },
        "selection": {
            "stage1_locked_candidates": stage1_lock["locked_candidates"],
            "stage1_passed_groups": stage1_lock["passed_groups"],
            "stage2_passed_groups": stage2_lock["stage2_passed_groups"],
            "all_three_groups_promoted": bool(stage2_lock["all_three_groups_promoted"]),
        },
        "read_flags": {
            "2024_weather_read": bool(stage2_result.get("2024_weather_read", False)),
            "2024_label_read": bool(stage2_result.get("2024_label_read", False)),
            "2025_weather_read": bool(final_result.get("2025_weather_read", False)),
            "sample_submission_read": bool(final_result.get("sample_submission_read", False)),
            "csv_created": bool(final_result.get("submission_created", False)),
        },
        "provenance": _snapshot_named(_provenance_paths(preregister_path)),
        "stage1_baseline_causal_provenance": _snapshot_named(
            _stage1_baseline_causal_paths(artifact_root)
        ),
        "inputs_read_conditionally": {
            "raw_root": str(raw_dir.resolve()),
            "stage1_uses_physical_prefix_only": True,
            "whole_2024_train_files_opened": bool(stage2_result.get("2024_weather_read", False)),
            "test_files_opened": bool(final_result.get("2025_weather_read", False)),
        },
        "tests": {
            "focused_source": describe_file(PROJECT_DIR / "tests/test_raw_grid_wind.py"),
            "focused_test_count": 12,
            "focused_prelaunch_status": "pass",
            "full_suite_test_count_prelaunch": 197,
            "full_suite_prelaunch_status": "pass",
            "commands": [
                ".venv/Scripts/python.exe -m unittest tests.test_raw_grid_wind -v",
                ".venv/Scripts/python.exe -m unittest discover -s tests -q"
            ],
        },
        "outputs": [describe_file(path) for path in outputs],
    }
    bayes._write_json(manifest_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    preregister_path = args.preregister.resolve()
    preregister = _verify_preregister(preregister_path)
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
        )
    if args.stage in ("stage2", "all"):
        _stage2(
            raw_dir=raw_dir,
            artifact_root=artifact_root,
            out_dir=out_dir,
            preregister=preregister,
        )
    if args.stage in ("final", "all"):
        _final(
            raw_dir=raw_dir,
            artifact_root=artifact_root,
            out_dir=out_dir,
            preregister=preregister,
        )
    if args.stage == "all":
        _write_manifest(
            raw_dir=raw_dir,
            artifact_root=artifact_root,
            out_dir=out_dir,
            preregister_path=preregister_path,
        )


if __name__ == "__main__":
    main()

"""Strict-forward CatBoost MultiQuantile Bayes experiment for BARAM."""

from __future__ import annotations

import argparse
from dataclasses import asdict
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
from src.catboost_multiquantile import (  # noqa: E402
    CatBoostBayesActionConfig,
    CatBoostMultiQuantileSurface,
    blend_with_baseline_kwh,
    exact_official_utility_action_cf,
)
from src.manifest import (  # noqa: E402
    describe_file,
    git_state,
    package_versions,
    sha256_file,
    utc_now,
)
from src.metric import CAPACITY_KWH, TARGET_COLS, score_details  # noqa: E402


PREREGISTER_SHA256 = "48e5ac657f5a72e453c20eddae238759c020558a26b10c5c477bed965280e23d"
QUANTILE_LEVELS = (0.10, 0.25, 0.50, 0.75, 0.90)
QUANTILE_COLUMNS = ("q10_cf", "q25_cf", "q50_cf", "q75_cf", "q90_cf")
LOSS_FUNCTION = "MultiQuantile:alpha=0.10,0.25,0.50,0.75,0.90"
BLEND_WEIGHTS = (0.05, 0.10, 0.20)
WEIGHT_KEYS = ("w05", "w10", "w20")
WEIGHT_BY_KEY = dict(zip(WEIGHT_KEYS, BLEND_WEIGHTS))
STAGE1_REQUIRED: dict[str, tuple[str, ...]] = {
    "kpx_group_1": ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4"),
    "kpx_group_2": ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4"),
    "kpx_group_3": ("full", "Q3", "Q4"),
}
STAGE2_REQUIRED = ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("stage1", "stage2", "final", "all"), required=True)
    parser.add_argument("--raw-dir", type=Path, default=Path(r"data/local/open"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/cache"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/postgate/catboost_multiquantile_bayes_strict_v1"),
    )
    parser.add_argument(
        "--preregister",
        type=Path,
        default=Path("configs/catboost_multiquantile_bayes_preregister_v1.json"),
    )
    parser.add_argument(
        "--verification-record",
        type=Path,
        default=Path("configs/catboost_multiquantile_bayes_verification_v1.json"),
    )
    return parser.parse_args(argv)


def _verify_preregister(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], CatBoostBayesActionConfig]:
    observed = sha256_file(path)
    if observed != PREREGISTER_SHA256:
        raise AssertionError(f"preregister hash changed: {observed}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    distribution = payload["conditional_distribution"]
    if tuple(map(float, distribution["quantile_levels"])) != QUANTILE_LEVELS:
        raise AssertionError("quantile levels changed")
    if tuple(distribution["prediction_column_order"]) != QUANTILE_COLUMNS:
        raise AssertionError("CatBoost alpha output order changed")
    if distribution["loss_function"] != LOSS_FUNCTION:
        raise AssertionError("MultiQuantile loss changed")
    if tuple(map(float, payload["candidate_family"]["fixed_global_blend_weights"])) != BLEND_WEIGHTS:
        raise AssertionError("blend weights changed")
    expected_parameters = {
        "iterations": 900,
        "depth": 6,
        "learning_rate": 0.03,
        "l2_leaf_reg": 5.0,
        "random_strength": 0.25,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 0.5,
        "rsm": 0.8,
        "border_count": 64,
        "random_seed": 42,
        "thread_count": 7,
        "task_type": "CPU",
        "allow_writing_files": False,
        "verbose": False,
        "use_best_model": False,
    }
    if distribution["parameters"] != expected_parameters:
        raise AssertionError("CatBoost parameters changed")
    action = payload["official_utility_action"]
    grid = action["absolute_action_grid"]
    config = CatBoostBayesActionConfig(
        quantile_levels=QUANTILE_LEVELS,
        interpolation_count=33,
        outcome_lower_cf=float(action["interpolated_outcome_clip_cf"][0]),
        outcome_upper_cf=float(action["interpolated_outcome_clip_cf"][1]),
        absolute_grid_lower_cf=float(grid["start_cf"]),
        absolute_grid_upper_cf=float(grid["end_cf"]),
        absolute_grid_step_cf=float(grid["step_cf"]),
        candidate_lower_cf=float(action["candidate_clip_cf"][0]),
        candidate_upper_cf=float(action["candidate_clip_cf"][1]),
        tie_tolerance=float(action["tie_tolerance"]),
    )
    model_spec = {
        "quantile_levels": QUANTILE_LEVELS,
        "loss_function": LOSS_FUNCTION,
        "model_parameters": expected_parameters,
        "minimum_actual_cf": 0.10,
    }
    return payload, model_spec, config


def _surface(
    model_spec: Mapping[str, Any], action_config: CatBoostBayesActionConfig
) -> CatBoostMultiQuantileSurface:
    return CatBoostMultiQuantileSurface(
        quantile_levels=model_spec["quantile_levels"],
        loss_function=model_spec["loss_function"],
        model_parameters=model_spec["model_parameters"],
        minimum_actual_cf=model_spec["minimum_actual_cf"],
        action_config=action_config,
    )


def _year_segments(year: int) -> dict[str, pd.DatetimeIndex]:
    return weather_protocol._year_segments(year)


def _provenance_paths(preregister_path: Path, verification_path: Path) -> dict[str, Path]:
    return {
        "runner": Path(__file__).resolve(),
        "catboost_surface": PROJECT_DIR / "src/catboost_multiquantile.py",
        "decision_helpers": PROJECT_DIR / "src/weather_quantile.py",
        "features": PROJECT_DIR / "src/features.py",
        "metric": PROJECT_DIR / "src/metric.py",
        "manifest": PROJECT_DIR / "src/manifest.py",
        "bounded_raw_helper": PROJECT_DIR / "scripts/run_shared_q07_multiseed.py",
        "strict_artifact_helper": PROJECT_DIR / "scripts/run_ficr_bayes_decision_strict.py",
        "weather_protocol_helper": PROJECT_DIR / "scripts/run_weather_quantile_bayes.py",
        "test": PROJECT_DIR / "tests/test_catboost_multiquantile_bayes.py",
        "preregister": preregister_path.resolve(),
        "preregister_sidecar": preregister_path.with_suffix(".sha256").resolve(),
        "verification_record": verification_path.resolve(),
    }


def _snapshot_named(paths: Mapping[str, Path]) -> dict[str, Any]:
    return {name: shared._snapshot_file(path) for name, path in paths.items()}


def _bounded_label_prefix(
    path: Path,
    spec: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = int(spec["data_rows"])
    byte_limit = int(spec["bytes"])
    observed_sha, observed_bytes = shared._csv_prefix_identity(
        path, data_rows=rows, byte_limit=byte_limit
    )
    if observed_sha != spec["sha256"] or observed_bytes != byte_limit:
        raise AssertionError("bounded label prefix identity changed")
    bounded = shared._BoundedRawReader(path, byte_limit=byte_limit)
    try:
        with io.BufferedReader(bounded, buffer_size=1024 * 1024) as stream:
            frame = pd.read_csv(
                stream,
                encoding="utf-8-sig",
                usecols=list(spec["usecols"]),
                memory_map=False,
            )
            bytes_returned = bounded.bytes_returned
            position = bounded.underlying_position
    finally:
        bounded.close()
    if bytes_returned != byte_limit or position != byte_limit:
        raise AssertionError("label parser crossed physical byte cap")
    if len(frame) != rows or tuple(frame.columns) != tuple(spec["usecols"]):
        raise AssertionError("bounded label schema or rows changed")
    times = pd.to_datetime(frame.pop("kst_dtm"), errors="raise")
    frame.index = pd.DatetimeIndex(times, name="forecast_kst_dtm")
    frame = frame.astype(float)
    if frame.index.min() != pd.Timestamp("2022-01-01 01:00:00"):
        raise AssertionError("bounded label start changed")
    if frame.index.max() != pd.Timestamp(spec["end"]):
        raise AssertionError("bounded label end changed")
    next_field = shared._next_csv_first_field(
        path, after_data_rows=rows, prefix_bytes=byte_limit
    )
    if next_field != spec["next_row_first_field_only"]:
        raise AssertionError("bounded label next timestamp changed")
    omitted = [column for column in TARGET_COLS if column not in frame.columns]
    return frame, {
        "path": str(path.resolve()),
        "data_rows": rows,
        "physical_byte_limit": byte_limit,
        "physical_prefix_sha256": observed_sha,
        "bytes_returned_to_parser": bytes_returned,
        "underlying_file_position_after_read": position,
        "materialized_columns": list(frame.columns),
        "omitted_target_columns": omitted,
        "omitted_target_value_cells_materialized": 0,
        "materialized_start": frame.index.min(),
        "materialized_end": frame.index.max(),
        "next_row_first_field_only": next_field,
        "next_row_target_value_cells_materialized": 0,
    }


def _prefix_snapshot(path: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    observed_sha, observed_bytes = shared._csv_prefix_identity(
        path, data_rows=int(spec["data_rows"]), byte_limit=int(spec["bytes"])
    )
    if observed_sha != spec["sha256"] or observed_bytes != int(spec["bytes"]):
        raise AssertionError("input prefix snapshot changed")
    return {
        "snapshot_kind": "physical_csv_prefix",
        "path": str(path.resolve()),
        "data_rows": int(spec["data_rows"]),
        "size_bytes": observed_bytes,
        "sha256": observed_sha,
    }


def _stage1_input_snapshot(
    raw_dir: Path, artifact_root: Path, preregister: Mapping[str, Any]
) -> dict[str, Any]:
    inputs = preregister["physical_stage1_inputs"]
    label_path = raw_dir / "train/train_labels.csv"
    return {
        "labels_g12_fit_prefix": _prefix_snapshot(label_path, inputs["labels_g12_fit_prefix"]),
        "labels_g3_fit_prefix": _prefix_snapshot(label_path, inputs["labels_g3_fit_prefix"]),
        "ldaps_prefix": _prefix_snapshot(raw_dir / "train/ldaps_train.csv", inputs["ldaps_weather_prefix"]),
        "gfs_prefix": _prefix_snapshot(raw_dir / "train/gfs_train.csv", inputs["gfs_weather_prefix"]),
        "info_workbook": shared._snapshot_file(raw_dir / "info.xlsx"),
        "baseline_2023_g12": shared._snapshot_file(artifact_root / "oof/dev2023_locked_v3.parquet"),
        "baseline_2023_g3": shared._snapshot_file(artifact_root / "oof/g3dev2023h2_candidates.parquet"),
    }


def _assert_preregistered_stage1_files(
    snapshot: Mapping[str, Any], preregister: Mapping[str, Any]
) -> None:
    inputs = preregister["physical_stage1_inputs"]
    for key in ("info_workbook",):
        if snapshot[key]["sha256"] != inputs[key]["sha256"] or snapshot[key]["size_bytes"] != inputs[key]["bytes"]:
            raise AssertionError(f"{key} differs from preregistration")
    baselines = preregister["baseline_contract"]
    for key, spec_key in (("baseline_2023_g12", "stage1_group_1_and_2"), ("baseline_2023_g3", "stage1_group_3")):
        spec = baselines[spec_key]
        if snapshot[key]["sha256"] != spec["sha256"] or snapshot[key]["size_bytes"] != spec["bytes"]:
            raise AssertionError(f"{key} differs from preregistration")


def _load_stage1_baseline(artifact_root: Path) -> pd.DataFrame:
    baseline, _ = weather_protocol._load_stage1_baseline(artifact_root)
    return baseline


def _fit_predict_group(
    *,
    group: str,
    all_features: pd.DataFrame,
    actual: pd.Series,
    baseline: pd.Series,
    fit_index: pd.DatetimeIndex,
    application_index: pd.DatetimeIndex,
    model_spec: Mapping[str, Any],
    action_config: CatBoostBayesActionConfig,
) -> tuple[CatBoostMultiQuantileSurface, pd.Series, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if len(fit_index.intersection(application_index)) or not fit_index.max() < application_index.min():
        raise AssertionError(f"{group} fit/application is not strict-forward")
    print(f"fit CatBoost MultiQuantile {group}: {fit_index.min()} -> {application_index.min()}", flush=True)
    model = _surface(model_spec, action_config).fit(
        all_features.loc[fit_index],
        actual.loc[fit_index],
        capacity_kwh=CAPACITY_KWH[group],
    )
    action, repaired = model.predict_action(
        all_features.loc[application_index], baseline.loc[application_index]
    )
    raw = pd.DataFrame(
        np.asarray(model.last_raw_quantiles_, dtype=np.float64),
        index=application_index,
        columns=QUANTILE_COLUMNS,
    )
    metadata = model.metadata()
    metadata.update(
        {
            "train_start": fit_index.min(),
            "train_end": fit_index.max(),
            "application_start": application_index.min(),
            "application_end": application_index.max(),
            "fit_end_before_application_start": True,
            "fit_application_overlap_count": 0,
            "raw_output_column_order": list(raw.columns),
            "repaired_output_column_order": list(repaired.columns),
        }
    )
    return model, action, raw, repaired, metadata


def _blend_frame(
    baseline: pd.DataFrame,
    actions: pd.DataFrame,
    application_indexes: Mapping[str, pd.DatetimeIndex],
    weight: float,
) -> pd.DataFrame:
    result = baseline.copy()
    for group in TARGET_COLS:
        index = application_indexes[group]
        result.loc[index, group] = blend_with_baseline_kwh(
            baseline.loc[index, group],
            actions.loc[index, group],
            weight=weight,
            capacity_kwh=CAPACITY_KWH[group],
        )
    return result


def _select_weight(comparisons: Mapping[str, Any]) -> tuple[float | None, dict[str, Any]]:
    if tuple(comparisons) != WEIGHT_KEYS:
        raise AssertionError("Stage1 candidate keys/order changed")
    audit: dict[str, Any] = {}
    eligible: list[str] = []
    for key in WEIGHT_KEYS:
        deltas: dict[str, float] = {}
        for group in TARGET_COLS:
            if set(comparisons[key][group]) != set(STAGE1_REQUIRED[group]):
                raise AssertionError(f"{key}/{group} registered segments changed")
            for segment in STAGE1_REQUIRED[group]:
                record = comparisons[key][group][segment]
                expected = float(record["candidate"]["score"]) - float(record["baseline"]["score"])
                if float(record["delta"]) != expected:
                    raise AssertionError("Stage1 delta arithmetic changed")
                deltas[f"{group}/{segment}"] = expected
        if len(deltas) != 17:
            raise AssertionError("Stage1 registered slice count changed")
        audit[key] = {
            "weight": WEIGHT_BY_KEY[key],
            "deltas": deltas,
            "minimum": min(deltas.values()),
            "mean": float(np.mean(list(deltas.values()))),
            "all_17_strictly_positive": all(value > 0.0 for value in deltas.values()),
        }
        if audit[key]["all_17_strictly_positive"]:
            eligible.append(key)
    if not eligible:
        return None, {"candidates": audit, "selected": "identity"}
    selected = max(
        eligible,
        key=lambda key: (audit[key]["minimum"], audit[key]["mean"], -audit[key]["weight"]),
    )
    return WEIGHT_BY_KEY[selected], {"candidates": audit, "selected": selected}


def _save_group_outputs(
    out_dir: Path,
    stage: str,
    group: str,
    action: pd.Series,
    raw: pd.DataFrame,
    repaired: pd.DataFrame,
) -> list[Path]:
    paths = [
        out_dir / "oof" / f"{stage}_{group}_action.parquet",
        out_dir / "oof" / f"{stage}_{group}_raw_quantiles.parquet",
        out_dir / "oof" / f"{stage}_{group}_repaired_quantiles.parquet",
    ]
    bayes._atomic_parquet(action.to_frame(), paths[0])
    bayes._atomic_parquet(raw, paths[1])
    bayes._atomic_parquet(repaired, paths[2])
    return paths


def _reload_audit(
    *,
    model_path: Path,
    groups: Sequence[str],
    features: Mapping[str, pd.DataFrame],
    baselines: pd.DataFrame,
    application_indexes: Mapping[str, pd.DatetimeIndex],
    stored_actions: Mapping[str, pd.Series],
    stored_raw: Mapping[str, pd.DataFrame],
    stored_repaired: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    loaded: dict[str, CatBoostMultiQuantileSurface] = joblib.load(model_path)
    if tuple(loaded) != tuple(groups):
        raise AssertionError("reloaded model group order changed")
    audit: dict[str, Any] = {}
    for group in groups:
        index = application_indexes[group]
        action, repaired = loaded[group].predict_action(
            features[group].loc[index], baselines.loc[index, group]
        )
        raw = np.asarray(loaded[group].last_raw_quantiles_, dtype=np.float64)
        raw_equal = np.array_equal(raw, stored_raw[group].to_numpy(dtype=np.float64))
        repaired_equal = np.array_equal(
            repaired.to_numpy(dtype=np.float64),
            stored_repaired[group].to_numpy(dtype=np.float64),
        )
        action_equal = np.array_equal(
            action.to_numpy(dtype=np.float64),
            stored_actions[group].to_numpy(dtype=np.float64),
        )
        crossing_sort_equal = np.array_equal(
            np.sort(raw, axis=1, kind="stable"), repaired.to_numpy(dtype=np.float64)
        )
        if not all((raw_equal, repaired_equal, action_equal, crossing_sort_equal)):
            raise AssertionError(f"{group} reload/action/crossing audit failed")
        independent_action = exact_official_utility_action_cf(
            repaired.to_numpy(dtype=np.float64),
            baselines.loc[index, group].to_numpy(dtype=np.float64) / CAPACITY_KWH[group],
            mean_train_actual_cf=loaded[group].mean_train_actual_cf_,
            config=loaded[group].action_config,
        )
        utility_action_equal = np.array_equal(
            independent_action * CAPACITY_KWH[group], action.to_numpy(dtype=np.float64)
        )
        if not utility_action_equal:
            raise AssertionError(f"{group} independent utility action differs")
        audit[group] = {
            "model_reload_raw_quantiles_bit_exact": raw_equal,
            "model_reload_repaired_quantiles_bit_exact": repaired_equal,
            "model_reload_action_bit_exact": action_equal,
            "stable_crossing_sort_bit_exact": crossing_sort_equal,
            "independent_official_utility_action_bit_exact": utility_action_equal,
            "alpha_output_column_order": list(QUANTILE_COLUMNS),
            "eligible_train_mean_actual_cf": loaded[group].mean_train_actual_cf_,
        }
    return audit


def _stage1(
    *,
    raw_dir: Path,
    artifact_root: Path,
    cache_dir: Path,
    out_dir: Path,
    preregister_path: Path,
    verification_path: Path,
    preregister: Mapping[str, Any],
    model_spec: Mapping[str, Any],
    action_config: CatBoostBayesActionConfig,
) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"Stage1 requires a new output directory: {out_dir}")
    out_dir.mkdir(parents=True)
    bayes._copy_exclusive(preregister_path, out_dir / "preregister.json")
    bayes._copy_exclusive(preregister_path.with_suffix(".sha256"), out_dir / "preregister.sha256")
    bayes._copy_exclusive(verification_path, out_dir / "verification_preflight.json")
    provenance_before = _snapshot_named(_provenance_paths(preregister_path, verification_path))
    input_before = _stage1_input_snapshot(raw_dir, artifact_root, preregister)
    _assert_preregistered_stage1_files(input_before, preregister)

    full_index = pd.date_range(
        "2022-01-01 01:00:00", "2024-01-01 00:00:00", freq="h", name="forecast_kst_dtm"
    )
    feature_index_holder = pd.DataFrame(index=full_index)
    features, raw_feature_contract = shared._read_stage1_raw_features(raw_dir, feature_index_holder)
    if any(frame.isna().to_numpy().any() for frame in features.values()):
        raise AssertionError("raw-built weather unexpectedly contains missing values")
    baseline = _load_stage1_baseline(artifact_root)
    segments = _year_segments(2023)
    fit_indexes = {
        "kpx_group_1": bayes._year_index(2022),
        "kpx_group_2": bayes._year_index(2022),
        "kpx_group_3": segments["H1"],
    }
    application_indexes = {
        "kpx_group_1": segments["full"],
        "kpx_group_2": segments["full"],
        "kpx_group_3": segments["H2"],
    }
    label_path = raw_dir / "train/train_labels.csv"
    specs = preregister["physical_stage1_inputs"]
    labels_g12, g12_label_evidence = _bounded_label_prefix(label_path, specs["labels_g12_fit_prefix"])
    models: dict[str, CatBoostMultiQuantileSurface] = {}
    actions_by_group: dict[str, pd.Series] = {}
    raw_by_group: dict[str, pd.DataFrame] = {}
    repaired_by_group: dict[str, pd.DataFrame] = {}
    training: dict[str, Any] = {}
    g12_paths: list[Path] = []
    for group in TARGET_COLS[:2]:
        model, action, raw, repaired, metadata = _fit_predict_group(
            group=group,
            all_features=features[group],
            actual=labels_g12[group],
            baseline=baseline[group],
            fit_index=fit_indexes[group],
            application_index=application_indexes[group],
            model_spec=model_spec,
            action_config=action_config,
        )
        models[group] = model
        actions_by_group[group] = action
        raw_by_group[group] = raw
        repaired_by_group[group] = repaired
        training[group] = metadata
        g12_paths.extend(_save_group_outputs(out_dir, "stage1", group, action, raw, repaired))
    g12_model_path = out_dir / "models/stage1_g12_models.joblib"
    bayes._atomic_joblib({group: models[group] for group in TARGET_COLS[:2]}, g12_model_path)
    g12_paths.append(g12_model_path)
    g12_reload = _reload_audit(
        model_path=g12_model_path,
        groups=TARGET_COLS[:2],
        features=features,
        baselines=baseline,
        application_indexes=application_indexes,
        stored_actions=actions_by_group,
        stored_raw=raw_by_group,
        stored_repaired=repaired_by_group,
    )
    g12_lock = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "groups": list(TARGET_COLS[:2]),
        "fit_label_evidence": g12_label_evidence,
        "application_target_value_cells_materialized": 0,
        "outputs": [describe_file(path) for path in g12_paths],
        "reload_audit": g12_reload,
        "created_before_g3_fit_prefix_and_before_all_stage1_score_labels": True,
    }
    g12_lock_path = out_dir / "stage1_g12_prescore_lock.json"
    bayes._write_json(g12_lock_path, g12_lock)

    labels_g3, g3_label_evidence = _bounded_label_prefix(label_path, specs["labels_g3_fit_prefix"])
    group = "kpx_group_3"
    model, action, raw, repaired, metadata = _fit_predict_group(
        group=group,
        all_features=features[group],
        actual=labels_g3[group],
        baseline=baseline[group],
        fit_index=fit_indexes[group],
        application_index=application_indexes[group],
        model_spec=model_spec,
        action_config=action_config,
    )
    models[group] = model
    actions_by_group[group] = action
    raw_by_group[group] = raw
    repaired_by_group[group] = repaired
    training[group] = metadata
    g3_paths = _save_group_outputs(out_dir, "stage1", group, action, raw, repaired)
    g3_model_path = out_dir / "models/stage1_g3_model.joblib"
    bayes._atomic_joblib({group: model}, g3_model_path)
    g3_paths.append(g3_model_path)
    g3_reload = _reload_audit(
        model_path=g3_model_path,
        groups=(group,),
        features=features,
        baselines=baseline,
        application_indexes=application_indexes,
        stored_actions=actions_by_group,
        stored_raw=raw_by_group,
        stored_repaired=repaired_by_group,
    )
    g3_lock_path = out_dir / "stage1_g3_prescore_lock.json"
    bayes._write_json(
        g3_lock_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "group": group,
            "fit_label_evidence": g3_label_evidence,
            "g12_omitted_target_value_cells_materialized": 0,
            "g3_application_target_value_cells_materialized": 0,
            "outputs": [describe_file(path) for path in g3_paths],
            "reload_audit": g3_reload,
            "created_before_all_stage1_score_labels": True,
        },
    )

    actions = pd.DataFrame(np.nan, index=baseline.index, columns=TARGET_COLS)
    for group in TARGET_COLS:
        actions.loc[application_indexes[group], group] = actions_by_group[group]
    zero_blend = _blend_frame(baseline, actions, application_indexes, 0.0)
    zero_exact = all(
        np.ascontiguousarray(zero_blend[group].to_numpy()).tobytes()
        == np.ascontiguousarray(baseline[group].to_numpy()).tobytes()
        for group in TARGET_COLS
    )
    if not zero_exact:
        raise AssertionError("weight-zero baseline identity is not bit exact")
    blends = {key: _blend_frame(baseline, actions, application_indexes, weight) for key, weight in WEIGHT_BY_KEY.items()}
    global_paths: list[Path] = []
    baseline_path = out_dir / "oof/stage1_baseline_2023.parquet"
    action_path = out_dir / "oof/stage1_action_2023.parquet"
    model_path = out_dir / "models/stage1_all_models.joblib"
    bayes._atomic_parquet(baseline, baseline_path)
    bayes._atomic_parquet(actions, action_path)
    bayes._atomic_joblib(models, model_path)
    global_paths.extend((baseline_path, action_path, model_path))
    for key, frame in blends.items():
        path = out_dir / "oof" / f"stage1_blend_{key}_2023.parquet"
        bayes._atomic_parquet(frame, path)
        global_paths.append(path)
    reload_audit = _reload_audit(
        model_path=model_path,
        groups=TARGET_COLS,
        features=features,
        baselines=baseline,
        application_indexes=application_indexes,
        stored_actions=actions_by_group,
        stored_raw=raw_by_group,
        stored_repaired=repaired_by_group,
    )
    provenance_after = _snapshot_named(_provenance_paths(preregister_path, verification_path))
    input_after = _stage1_input_snapshot(raw_dir, artifact_root, preregister)
    shared._assert_snapshot_equal(provenance_before, provenance_after, name="Stage1 provenance")
    shared._assert_snapshot_equal(input_before, input_after, name="Stage1 fit inputs")
    prescore_record = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "model_spec": model_spec,
        "action_config": asdict(action_config),
        "candidate_weights": BLEND_WEIGHTS,
        "candidate_formula": preregister["candidate_family"]["formula"],
        "training": training,
        "raw_feature_contract": raw_feature_contract,
        "g12_prescore_lock": describe_file(g12_lock_path),
        "g3_prescore_lock": describe_file(g3_lock_path),
        "global_outputs": [describe_file(path) for path in global_paths],
        "model_reload_action_audit": reload_audit,
        "weight_zero_identity_value_bits_exact": zero_exact,
        "stage1_application_label_value_cells_materialized": 0,
        "2024_read": False,
        "2025_read": False,
        "multi_year_cache_opened": False,
        "provenance_before": provenance_before,
        "provenance_after": provenance_after,
        "stage1_input_before": input_before,
        "stage1_input_after": input_after,
    }
    prescore_record_path = out_dir / "stage1_prescore_record.json"
    bayes._write_json(prescore_record_path, prescore_record)
    prescore_lock_path = out_dir / "stage1_global_prescore_lock.json"
    bayes._write_json(
        prescore_lock_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "prescore_record": describe_file(prescore_record_path),
            "candidate_artifacts": [describe_file(path) for path in global_paths],
            "g12_lock": describe_file(g12_lock_path),
            "g3_lock": describe_file(g3_lock_path),
            "candidate_model_reload_action_blend_frozen": True,
            "created_before_stage1_application_label_values": True,
            "2024_read": False,
            "2025_read": False,
        },
    )

    score_labels, score_label_evidence = _bounded_label_prefix(
        label_path, specs["labels_stage1_score_prefix_after_prescore_lock"]
    )
    comparisons: dict[str, Any] = {}
    for key in WEIGHT_KEYS:
        comparisons[key] = {}
        for group in TARGET_COLS:
            app_index = application_indexes[group]
            segment_indexes = (
                {name: segments[name] for name in STAGE1_REQUIRED[group]}
                if group in TARGET_COLS[:2]
                else {"full": segments["H2"], "Q3": segments["Q3"], "Q4": segments["Q4"]}
            )
            comparisons[key][group] = bayes._comparison(
                score_labels.loc[app_index, group],
                baseline.loc[app_index, group],
                blends[key].loc[app_index, group],
                group,
                segment_indexes,
            )
    locked_weight, selection = _select_weight(comparisons)
    result = {
        "schema_version": 1,
        "experiment_id": preregister["experiment_id"],
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "global_prescore_lock": describe_file(prescore_lock_path),
        "score_label_evidence_after_prescore_lock": score_label_evidence,
        "registered_slice_count": 17,
        "comparisons": comparisons,
        "comparisons_sha256": bayes._canonical_sha256(comparisons),
        "selection": selection,
        "selection_sha256": bayes._canonical_sha256(selection),
        "locked_weight": locked_weight,
        "locked_candidate": "identity" if locked_weight is None else f"blend_{locked_weight:.2f}",
        "2024_read": False,
        "2025_read": False,
        "leaderboard_score_claim": False,
    }
    result_path = out_dir / "stage1_results.json"
    bayes._write_json(result_path, result)
    promotion_lock = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "global_prescore_lock": describe_file(prescore_lock_path),
        "stage1_results": describe_file(result_path),
        "comparisons_sha256": result["comparisons_sha256"],
        "selection_sha256": result["selection_sha256"],
        "locked_weight": locked_weight,
        "locked_candidate": result["locked_candidate"],
        "no_2024_reselection_or_retuning": True,
        "2024_read": False,
        "2025_read": False,
    }
    lock_path = out_dir / "stage1_promotion_lock.json"
    bayes._write_json(lock_path, promotion_lock)
    print(f"Stage1 locked candidate: {result['locked_candidate']}", flush=True)
    return promotion_lock


def _load_stage1_lock(out_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    lock_path = out_dir / "stage1_promotion_lock.json"
    result_path = out_dir / "stage1_results.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if lock["preregister_sha256"] != PREREGISTER_SHA256:
        raise AssertionError("Stage1 preregister hash changed")
    if lock["stage1_results"]["sha256"] != sha256_file(result_path):
        raise AssertionError("Stage1 result changed")
    if lock["global_prescore_lock"]["sha256"] != sha256_file(out_dir / "stage1_global_prescore_lock.json"):
        raise AssertionError("Stage1 prescore lock changed")
    selected, selection = _select_weight(result["comparisons"])
    if selected != result["locked_weight"] or selected != lock["locked_weight"]:
        raise AssertionError("Stage1 selection changed")
    if selection != result["selection"] or bayes._canonical_sha256(selection) != lock["selection_sha256"]:
        raise AssertionError("Stage1 selection audit changed")
    return lock, result


def _postlock_cache_audit(
    *, raw_dir: Path, cache_dir: Path, out_dir: Path
) -> dict[str, Any]:
    lock, stage1 = _load_stage1_lock(out_dir)
    path = out_dir / "postlock_cache_audit.json"
    if path.exists():
        raise FileExistsError(path)
    if lock["locked_weight"] is None:
        audit = {
            "schema_version": 1,
            "created_utc": utc_now(),
            "stage1_promotion_lock": describe_file(out_dir / "stage1_promotion_lock.json"),
            "performed": False,
            "2024_cache_values_read": False,
            "reason": "identity locked; no cache file was opened",
            "checks": {},
        }
        bayes._write_json(path, audit)
        return audit
    # This audit is post-selection only. It compares the exact 2022-2023 prefix.
    full_index = pd.date_range("2022-01-01 01:00", "2024-01-01 00:00", freq="h", name="forecast_kst_dtm")
    rebuilt, _ = shared._read_stage1_raw_features(raw_dir, pd.DataFrame(index=full_index))
    checks: dict[str, Any] = {}
    for group in TARGET_COLS:
        cache_path = cache_dir / f"{group}_weather_train.parquet"
        cached = pd.read_parquet(
            cache_path,
            engine="pyarrow",
            filters=[("forecast_kst_dtm", "<=", pd.Timestamp("2024-01-01 00:00"))],
        )
        cached.index = pd.DatetimeIndex(cached.index, name="forecast_kst_dtm")
        observed = rebuilt[group]
        exact = (
            observed.index.equals(cached.index)
            and tuple(observed.columns) == tuple(cached.columns)
            and np.ascontiguousarray(observed.to_numpy()).tobytes()
            == np.ascontiguousarray(cached.to_numpy()).tobytes()
        )
        if not exact:
            raise AssertionError(f"{group} raw/cache prefix differs")
        checks[group] = {
            "rows": len(observed),
            "columns": observed.shape[1],
            "value_bits_exact": True,
            "frame_sha256": shared._frame_sha256(observed),
            "cache_file": describe_file(cache_path),
        }
    audit = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "stage1_promotion_lock": describe_file(out_dir / "stage1_promotion_lock.json"),
        "performed": True,
        "selection_was_locked_before_cache_read": True,
        "2024_cache_values_read": False,
        "checks": checks,
        "all_group_2022_2023_prefixes_value_bit_exact": True,
    }
    bayes._write_json(path, audit)
    return audit


def _stage2_promotion(comparisons: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    return weather_protocol._stage2_promotion(comparisons)


def _stage2(
    *,
    raw_dir: Path,
    artifact_root: Path,
    cache_dir: Path,
    out_dir: Path,
    preregister: Mapping[str, Any],
    model_spec: Mapping[str, Any],
    action_config: CatBoostBayesActionConfig,
) -> dict[str, Any]:
    stage1_lock, _ = _load_stage1_lock(out_dir)
    cache_audit_path = out_dir / "postlock_cache_audit.json"
    if not cache_audit_path.is_file():
        raise AssertionError("postlock cache audit missing")
    cache_audit = json.loads(cache_audit_path.read_text(encoding="utf-8"))
    weight = stage1_lock["locked_weight"]
    if weight is None:
        result = {
            "schema_version": 1,
            "created_utc": utc_now(),
            "executed": False,
            "locked_weight": None,
            "2024_read": False,
            "candidate_promoted": False,
            "comparisons": {},
            "promotion_audit": {},
            "reason": "No global weight passed every Stage1 slice",
        }
        result_path = out_dir / "stage2_results.json"
        bayes._write_json(result_path, result)
        lock = {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "stage1_promotion_lock": describe_file(out_dir / "stage1_promotion_lock.json"),
            "postlock_cache_audit": describe_file(cache_audit_path),
            "stage2_results": describe_file(result_path),
            "locked_weight": None,
            "candidate_promoted": False,
            "2024_read": False,
            "2025_read": False,
            "csv_allowed": False,
        }
        bayes._write_json(out_dir / "stage2_promotion_lock.json", lock)
        return lock

    if not cache_audit.get("performed") or not cache_audit.get("all_group_2022_2023_prefixes_value_bit_exact"):
        raise AssertionError("postlock cache audit did not pass")
    label_path = raw_dir / "train/train_labels.csv"
    source_labels, source_evidence = _bounded_label_prefix(
        label_path,
        preregister["physical_stage1_inputs"]["labels_stage1_score_prefix_after_prescore_lock"],
    )
    full_index = pd.date_range("2022-01-01 01:00", "2025-01-01 00:00", freq="h", name="forecast_kst_dtm")
    features = shared._read_features(cache_dir, pd.DataFrame(index=full_index), expected_end=bayes.YEAR_2024_END)
    index_2024 = bayes._year_index(2024)
    baseline_path = artifact_root / "oof/gate2024_locked_v3_cf_fix.parquet"
    baseline = bayes._read_prediction(baseline_path, index_2024, required_columns=TARGET_COLS)
    fit_indexes = {
        "kpx_group_1": source_labels.index,
        "kpx_group_2": source_labels.index,
        "kpx_group_3": bayes._year_index(2023),
    }
    application_indexes = {group: index_2024 for group in TARGET_COLS}
    models: dict[str, CatBoostMultiQuantileSurface] = {}
    actions = pd.DataFrame(index=index_2024, columns=TARGET_COLS, dtype=float)
    raws: dict[str, pd.DataFrame] = {}
    repaireds: dict[str, pd.DataFrame] = {}
    actions_by_group: dict[str, pd.Series] = {}
    training: dict[str, Any] = {}
    output_paths: list[Path] = []
    for group in TARGET_COLS:
        model, action, raw, repaired, metadata = _fit_predict_group(
            group=group,
            all_features=features[group],
            actual=source_labels[group],
            baseline=baseline[group],
            fit_index=fit_indexes[group],
            application_index=index_2024,
            model_spec=model_spec,
            action_config=action_config,
        )
        models[group] = model
        actions[group] = action
        actions_by_group[group] = action
        raws[group] = raw
        repaireds[group] = repaired
        training[group] = metadata
        output_paths.extend(_save_group_outputs(out_dir, "stage2", group, action, raw, repaired))
    candidate = _blend_frame(baseline, actions, application_indexes, float(weight))
    for name, frame in (("stage2_baseline_2024", baseline), ("stage2_action_2024", actions), ("stage2_fixed_candidate_2024", candidate)):
        path = out_dir / "oof" / f"{name}.parquet"
        bayes._atomic_parquet(frame, path)
        output_paths.append(path)
    model_path = out_dir / "models/stage2_models.joblib"
    bayes._atomic_joblib(models, model_path)
    output_paths.append(model_path)
    reload_audit = _reload_audit(
        model_path=model_path,
        groups=TARGET_COLS,
        features=features,
        baselines=baseline,
        application_indexes=application_indexes,
        stored_actions=actions_by_group,
        stored_raw=raws,
        stored_repaired=repaireds,
    )
    prescore_record_path = out_dir / "stage2_prescore_record.json"
    bayes._write_json(
        prescore_record_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "stage1_promotion_lock": describe_file(out_dir / "stage1_promotion_lock.json"),
            "postlock_cache_audit": describe_file(cache_audit_path),
            "locked_weight": weight,
            "source_label_evidence": source_evidence,
            "training": training,
            "reload_audit": reload_audit,
            "outputs": [describe_file(path) for path in output_paths],
            "2024_application_label_value_cells_materialized": 0,
        },
    )
    prescore_lock_path = out_dir / "stage2_2024_prescore_lock.json"
    bayes._write_json(
        prescore_lock_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "prescore_record": describe_file(prescore_record_path),
            "postlock_cache_audit": describe_file(cache_audit_path),
            "candidate_model_reload_action_weight_frozen": True,
            "created_before_2024_application_label_values": True,
        },
    )
    full_labels = bayes._read_full_labels(label_path)
    segments = _year_segments(2024)
    comparisons = {
        group: bayes._comparison(
            full_labels.loc[index_2024, group],
            baseline[group],
            candidate[group],
            group,
            {name: segments[name] for name in STAGE2_REQUIRED},
        )
        for group in TARGET_COLS
    }
    promoted, promotion_audit = _stage2_promotion(comparisons)
    base_metric = score_details(full_labels.loc[index_2024], baseline).as_dict()
    cand_metric = score_details(full_labels.loc[index_2024], candidate).as_dict()
    result = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "executed": True,
        "stage2_prescore_lock": describe_file(prescore_lock_path),
        "locked_weight": weight,
        "2024_read": True,
        "comparisons": comparisons,
        "comparisons_sha256": bayes._canonical_sha256(comparisons),
        "promotion_audit": promotion_audit,
        "candidate_promoted": promoted,
        "official_full_metrics": {
            "baseline": base_metric,
            "candidate": cand_metric,
            "delta_total_score": cand_metric["total_score"] - base_metric["total_score"],
        },
        "no_2024_reselection_or_retuning": True,
    }
    result_path = out_dir / "stage2_results.json"
    bayes._write_json(result_path, result)
    lock = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "stage1_promotion_lock": describe_file(out_dir / "stage1_promotion_lock.json"),
        "postlock_cache_audit": describe_file(cache_audit_path),
        "stage2_prescore_lock": describe_file(prescore_lock_path),
        "stage2_results": describe_file(result_path),
        "locked_weight": weight,
        "candidate_promoted": promoted,
        "2024_read": True,
        "2025_read": False,
        "no_2024_reselection_or_retuning": True,
        "csv_allowed": promoted,
    }
    bayes._write_json(out_dir / "stage2_promotion_lock.json", lock)
    return lock


def _load_stage2_lock(out_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    stage1_lock, _ = _load_stage1_lock(out_dir)
    lock_path = out_dir / "stage2_promotion_lock.json"
    result_path = out_dir / "stage2_results.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if lock["preregister_sha256"] != PREREGISTER_SHA256:
        raise AssertionError("Stage2 preregister changed")
    if lock["stage2_results"]["sha256"] != sha256_file(result_path):
        raise AssertionError("Stage2 results changed")
    if lock["postlock_cache_audit"]["sha256"] != sha256_file(out_dir / "postlock_cache_audit.json"):
        raise AssertionError("postlock cache audit changed")
    if lock["locked_weight"] != stage1_lock["locked_weight"]:
        raise AssertionError("Stage2 weight changed")
    if result.get("executed"):
        promoted, audit = _stage2_promotion(result["comparisons"])
        if promoted != result["candidate_promoted"] or audit != result["promotion_audit"]:
            raise AssertionError("Stage2 promotion changed")
    else:
        promoted = False
    if bool(lock["candidate_promoted"]) != promoted or bool(lock["csv_allowed"]) != promoted:
        raise AssertionError("Stage2 lock promotion changed")
    return lock, result


def _finalize(
    *,
    raw_dir: Path,
    artifact_root: Path,
    cache_dir: Path,
    out_dir: Path,
    preregister_path: Path,
    verification_path: Path,
    model_spec: Mapping[str, Any],
    action_config: CatBoostBayesActionConfig,
) -> dict[str, Any]:
    lock, stage2_result = _load_stage2_lock(out_dir)
    final_path = out_dir / "final_results.json"
    manifest_path = out_dir / "manifest.json"
    if final_path.exists() or manifest_path.exists():
        raise FileExistsError("final outputs already exist")
    final_inputs: dict[str, Any] = {}
    if not lock["candidate_promoted"]:
        final_result = {
            "schema_version": 1,
            "created_utc": utc_now(),
            "executed": False,
            "locked_weight": lock["locked_weight"],
            "candidate_promoted": False,
            "2025_read": False,
            "submission_created": False,
            "reason": "global candidate failed a preregistered gate",
            "leaderboard_score_claim": False,
        }
        bayes._write_json(final_path, final_result)
    else:
        final_inputs = {
            "labels_full": shared._snapshot_file(raw_dir / "train/train_labels.csv"),
            "sample": shared._snapshot_file(raw_dir / "sample_submission.csv"),
            "baseline": shared._snapshot_file(artifact_root / "final_cf_fix/predictions/corrected_v3_test.parquet"),
            **{f"train_cache_{g}": shared._snapshot_file(cache_dir / f"{g}_weather_train.parquet") for g in TARGET_COLS},
            **{f"test_cache_{g}": shared._snapshot_file(cache_dir / f"{g}_weather_test.parquet") for g in TARGET_COLS},
        }
        labels = bayes._read_full_labels(raw_dir / "train/train_labels.csv")
        train_features = shared._read_features(cache_dir, labels, expected_end=bayes.YEAR_2024_END)
        sample = pd.read_csv(raw_dir / "sample_submission.csv", encoding="utf-8-sig", dtype={"forecast_id": "string", "forecast_kst_dtm": "string"})
        if tuple(sample.columns) != ("forecast_id", "forecast_kst_dtm", *TARGET_COLS):
            raise AssertionError("sample schema changed")
        test_index = pd.DatetimeIndex(pd.to_datetime(sample["forecast_kst_dtm"]), name="forecast_kst_dtm")
        if not test_index.equals(bayes._year_index(2025)):
            raise AssertionError("sample index changed")
        test_features = shared._read_test_features(cache_dir, test_index)
        baseline = bayes._read_prediction(
            artifact_root / "final_cf_fix/predictions/corrected_v3_test.parquet",
            test_index,
            required_columns=TARGET_COLS,
        )
        models: dict[str, CatBoostMultiQuantileSurface] = {}
        actions = pd.DataFrame(index=test_index, columns=TARGET_COLS, dtype=float)
        output_paths: list[Path] = []
        training: dict[str, Any] = {}
        for group in TARGET_COLS:
            joined = pd.concat((train_features[group], test_features[group]), axis=0)
            model, action, raw, repaired, metadata = _fit_predict_group(
                group=group,
                all_features=joined,
                actual=labels[group],
                baseline=baseline[group],
                fit_index=labels.index,
                application_index=test_index,
                model_spec=model_spec,
                action_config=action_config,
            )
            models[group] = model
            actions[group] = action
            training[group] = metadata
            output_paths.extend(_save_group_outputs(out_dir, "final", group, action, raw, repaired))
        candidate = _blend_frame(baseline, actions, {g: test_index for g in TARGET_COLS}, float(lock["locked_weight"]))
        prediction_path = out_dir / "predictions/catboost_multiquantile_bayes_2025.parquet"
        action_path = out_dir / "predictions/catboost_multiquantile_action_2025.parquet"
        model_path = out_dir / "models/final_models.joblib"
        csv_path = out_dir / "catboost_multiquantile_bayes_2025.csv"
        bayes._atomic_parquet(candidate, prediction_path)
        bayes._atomic_parquet(actions, action_path)
        bayes._atomic_joblib(models, model_path)
        submission = sample.copy()
        for group in TARGET_COLS:
            submission[group] = candidate[group].to_numpy(dtype=float)
        bayes._atomic_csv(submission, csv_path)
        verification = bayes._verify_submission(csv_path, sample, candidate)
        output_paths.extend((prediction_path, action_path, model_path, csv_path))
        final_result = {
            "schema_version": 1,
            "created_utc": utc_now(),
            "executed": True,
            "locked_weight": lock["locked_weight"],
            "candidate_promoted": True,
            "2025_read": True,
            "submission_created": True,
            "training": training,
            "outputs": [describe_file(path) for path in output_paths],
            "submission_verification": verification,
            "leaderboard_score_claim": False,
        }
        bayes._write_json(final_path, final_result)

    outputs = sorted(path for path in out_dir.rglob("*") if path.is_file() and path != manifest_path)
    stage1_result = json.loads((out_dir / "stage1_results.json").read_text(encoding="utf-8"))
    manifest = {
        "schema_version": 1,
        "artifact_type": "catboost_multiquantile_bayes_strict_forward_v1",
        "created_utc": utc_now(),
        "command": [sys.executable, *sys.argv],
        "runtime": {"packages": package_versions(), "git": git_state(PROJECT_DIR)},
        "preregister_sha256": PREREGISTER_SHA256,
        "candidate_weights": BLEND_WEIGHTS,
        "locked_weight": lock["locked_weight"],
        "candidate_promoted": bool(lock["candidate_promoted"]),
        "locks": {
            "stage1_g12_prescore": describe_file(out_dir / "stage1_g12_prescore_lock.json"),
            "stage1_g3_prescore": describe_file(out_dir / "stage1_g3_prescore_lock.json"),
            "stage1_global_prescore": describe_file(out_dir / "stage1_global_prescore_lock.json"),
            "stage1_promotion": describe_file(out_dir / "stage1_promotion_lock.json"),
            "postlock_cache_audit": describe_file(out_dir / "postlock_cache_audit.json"),
            "stage2_promotion": describe_file(out_dir / "stage2_promotion_lock.json"),
        },
        "provenance": stage1_result.get("provenance", _snapshot_named(_provenance_paths(preregister_path, verification_path))),
        "final_inputs": list(final_inputs.values()),
        "outputs": [describe_file(path) for path in outputs],
        "read_flags": {
            "stage1_multi_year_cache_values_read_prelock": False,
            "2024_read": bool(stage2_result.get("2024_read")),
            "2025_read": bool(final_result["2025_read"]),
            "csv_created": bool(final_result["submission_created"]),
        },
        "contracts": {
            "catboost_joint_multiquantile_only": True,
            "alpha_output_order_locked": True,
            "crossing_sort_reload_verified": True,
            "official_utility_action_reload_verified": True,
            "fit_end_before_apply_start": True,
            "2024_used_for_selection": False,
            "no_public_metric_or_scale_input": True,
            "leaderboard_score_claim": False,
        },
    }
    bayes._write_json(manifest_path, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    raw_dir = args.raw_dir.expanduser().resolve()
    artifact_root = args.artifact_root.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    preregister_path = args.preregister.expanduser().resolve()
    verification_path = args.verification_record.expanduser().resolve()
    preregister, model_spec, action_config = _verify_preregister(preregister_path)
    if args.stage in ("stage1", "all"):
        _stage1(
            raw_dir=raw_dir,
            artifact_root=artifact_root,
            cache_dir=cache_dir,
            out_dir=out_dir,
            preregister_path=preregister_path,
            verification_path=verification_path,
            preregister=preregister,
            model_spec=model_spec,
            action_config=action_config,
        )
        _postlock_cache_audit(raw_dir=raw_dir, cache_dir=cache_dir, out_dir=out_dir)
        if args.stage == "stage1":
            return 0
    if args.stage in ("stage2", "all"):
        stage2_lock = _stage2(
            raw_dir=raw_dir,
            artifact_root=artifact_root,
            cache_dir=cache_dir,
            out_dir=out_dir,
            preregister=preregister,
            model_spec=model_spec,
            action_config=action_config,
        )
        if args.stage == "stage2":
            return 0
    else:
        stage2_lock = None
    if args.stage in ("final", "all"):
        _finalize(
            raw_dir=raw_dir,
            artifact_root=artifact_root,
            cache_dir=cache_dir,
            out_dir=out_dir,
            preregister_path=preregister_path,
            verification_path=verification_path,
            model_spec=model_spec,
            action_config=action_config,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

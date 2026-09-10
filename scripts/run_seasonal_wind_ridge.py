"""Strict-forward low-degree seasonal/wind Ridge experiment for BARAM."""

from __future__ import annotations

import argparse
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

from scripts import run_catboost_multiquantile_bayes as cb_protocol  # noqa: E402
from scripts import run_ficr_bayes_decision_strict as strict  # noqa: E402
from scripts import run_shared_q07_multiseed as shared  # noqa: E402
from src.manifest import (  # noqa: E402
    describe_file,
    git_state,
    package_versions,
    sha256_file,
    utc_now,
)
from src.metric import CAPACITY_KWH, TARGET_COLS, score_details  # noqa: E402
from src.seasonal_wind_ridge import (  # noqa: E402
    FEATURE_COLUMNS,
    STAGE1_REQUIRED,
    STAGE2_REQUIRED,
    SeasonalWindRidge,
    assert_strict_forward,
    blend_direct_kwh,
    build_seasonal_wind_features,
    select_stage1_weight,
    stage2_promoted,
)


PREREGISTER_SHA256 = "de98bdab79a4f1669f2ca97c2c7d37e088baca971ea8927708093ef0fa53b7ad"
WEIGHTS: dict[str, float] = {"w025": 0.025, "w05": 0.05}
WIND_COLUMN = "cross__hub_ws_mean"
CLI_DESCRIPTION = __doc__
OUTPUT_SLUG = "seasonal_wind_ridge"
ARTIFACT_TYPE = "seasonal_wind_ridge_strict_forward_v1"
STANDALONE_DIAGNOSTIC = False
MANIFEST_TESTS: dict[str, Any] = {}
MANIFEST_CONTRACTS: dict[str, Any] = {
    "one_direct_ridge_only": True,
    "fixed_23_feature_design": True,
    "fitted_model_reload_bit_exact": True,
    "fit_end_before_apply_start": True,
    "2024_used_for_selection": False,
    "no_public_metric_scale_scada_or_application_label_feature": True,
    "leaderboard_score_claim": False,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=CLI_DESCRIPTION)
    parser.add_argument(
        "--stage", choices=("stage1", "stage2", "final", "all"), required=True
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=Path(r"data/local/open")
    )
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/cache"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/postgate/seasonal_wind_ridge_strict_v1"),
    )
    parser.add_argument(
        "--preregister",
        type=Path,
        default=Path("configs/seasonal_wind_ridge_preregister_v1.json"),
    )
    return parser.parse_args(argv)


def _verify_preregister(path: Path) -> dict[str, Any]:
    observed = sha256_file(path)
    if observed != PREREGISTER_SHA256:
        raise AssertionError(f"preregister hash changed: {observed}")
    sidecar = path.with_suffix(".sha256")
    expected_sidecar = f"{PREREGISTER_SHA256}  {path.name}\n"
    if sidecar.read_text(encoding="utf-8") != expected_sidecar:
        raise AssertionError("preregister sidecar changed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["experiment_id"] != "seasonal_wind_ridge_strict_forward_v1":
        raise AssertionError("experiment id changed")
    feature = payload["feature_contract"]
    if tuple(feature["columns_in_exact_order"]) != FEATURE_COLUMNS:
        raise AssertionError("feature contract changed")
    if feature["sole_weather_source_column"] != WIND_COLUMN:
        raise AssertionError("weather source changed")
    if tuple(map(float, payload["candidate_family"]["fixed_global_blend_weights"])) != tuple(
        WEIGHTS.values()
    ):
        raise AssertionError("weights changed")
    model = payload["model"]["pipeline"][1]
    if model != {
        "class": "sklearn.linear_model.Ridge",
        "alpha": 250.0,
        "fit_intercept": True,
        "solver": "cholesky",
        "tol": 1e-08,
    }:
        raise AssertionError("Ridge contract changed")
    if payload["stage1"]["registered_slice_count"] != 17:
        raise AssertionError("Stage1 slice count changed")
    return payload


def _provenance_paths(preregister_path: Path) -> dict[str, Path]:
    return {
        "runner": Path(__file__).resolve(),
        "model": PROJECT_DIR / "src/seasonal_wind_ridge.py",
        "test": PROJECT_DIR / "tests/test_seasonal_wind_ridge.py",
        "features": PROJECT_DIR / "src/features.py",
        "metric": PROJECT_DIR / "src/metric.py",
        "manifest": PROJECT_DIR / "src/manifest.py",
        "bounded_helper": PROJECT_DIR / "scripts/run_shared_q07_multiseed.py",
        "strict_helper": PROJECT_DIR / "scripts/run_ficr_bayes_decision_strict.py",
        "baseline_helper": PROJECT_DIR / "scripts/run_weather_quantile_bayes.py",
        "bounded_protocol": PROJECT_DIR / "scripts/run_catboost_multiquantile_bayes.py",
        "preregister": preregister_path.resolve(),
        "preregister_sidecar": preregister_path.with_suffix(".sha256").resolve(),
    }


def _snapshot_named(paths: Mapping[str, Path]) -> dict[str, Any]:
    return {name: shared._snapshot_file(path) for name, path in paths.items()}


def _copy_exclusive(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    with source.open("rb") as input_stream, destination.open("xb") as output_stream:
        shutil.copyfileobj(input_stream, output_stream)


def _seasonal_features(
    weather_features: Mapping[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    output: dict[str, pd.DataFrame] = {}
    evidence: dict[str, Any] = {}
    for group in TARGET_COLS:
        weather = weather_features[group]
        if WIND_COLUMN not in weather.columns:
            raise KeyError(f"{group} missing {WIND_COLUMN}")
        low = build_seasonal_wind_features(weather.index, weather[WIND_COLUMN])
        output[group] = low
        evidence[group] = {
            "rows": len(low),
            "columns": low.shape[1],
            "start": low.index.min(),
            "end": low.index.max(),
            "raw_consensus_wind_sha256": shared._frame_sha256(
                weather.loc[:, [WIND_COLUMN]]
            ),
            "design_frame_sha256": shared._frame_sha256(low),
            "actual_feature_columns": 0,
            "scada_feature_columns": 0,
            "baseline_or_residual_feature_columns": 0,
            "public_feature_columns": 0,
        }
    return output, evidence


def _fit_predict(
    *,
    group: str,
    features: pd.DataFrame,
    actual: pd.Series,
    fit_index: pd.DatetimeIndex,
    application_index: pd.DatetimeIndex,
) -> tuple[SeasonalWindRidge, pd.Series, dict[str, Any]]:
    assert_strict_forward(fit_index, application_index)
    fit_features = features.loc[fit_index]
    fit_actual = actual.loc[fit_index]
    model = SeasonalWindRidge().fit(
        fit_features, fit_actual, capacity_kwh=CAPACITY_KWH[group]
    )
    predicted = model.predict_cf(features.loc[application_index])
    metadata = dict(model.fit_metadata_ or {})
    metadata.update(
        {
            "group": group,
            "fit_start": fit_index.min(),
            "fit_end": fit_index.max(),
            "application_start": application_index.min(),
            "application_end": application_index.max(),
            "fit_end_before_application_start": True,
            "fit_application_overlap_count": 0,
            "prediction_min_cf": float(predicted.min()),
            "prediction_max_cf": float(predicted.max()),
        }
    )
    return model, predicted, metadata


def _reload_audit(
    *,
    model_path: Path,
    groups: Sequence[str],
    features: Mapping[str, pd.DataFrame],
    application_indexes: Mapping[str, pd.DatetimeIndex],
    expected: Mapping[str, pd.Series],
) -> dict[str, Any]:
    models: dict[str, SeasonalWindRidge] = joblib.load(model_path)
    if tuple(models) != tuple(groups):
        raise AssertionError("reloaded model group order changed")
    audit: dict[str, Any] = {}
    for group in groups:
        observed = models[group].predict_cf(features[group].loc[application_indexes[group]])
        exact = (
            observed.index.equals(expected[group].index)
            and np.array_equal(observed.to_numpy(), expected[group].to_numpy())
        )
        if not exact:
            raise AssertionError(f"{group} reloaded Ridge prediction differs")
        ridge = models[group].pipeline_.named_steps["ridge"]
        audit[group] = {
            "prediction_bit_exact": True,
            "feature_columns_exact": list(FEATURE_COLUMNS),
            "ridge_alpha": float(ridge.alpha),
            "ridge_solver": str(ridge.solver),
            "target_kind": "direct_capacity_factor",
        }
    return audit


def _blend_frame(
    baseline: pd.DataFrame,
    direct: pd.DataFrame,
    application_indexes: Mapping[str, pd.DatetimeIndex],
    weight: float,
) -> pd.DataFrame:
    result = baseline.copy()
    for group in TARGET_COLS:
        index = application_indexes[group]
        result.loc[index, group] = blend_direct_kwh(
            baseline.loc[index, group],
            direct.loc[index, group],
            weight=weight,
            capacity_kwh=CAPACITY_KWH[group],
        )
    return result


def _load_stage1_lock(out_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    lock_path = out_dir / "stage1_promotion_lock.json"
    result_path = out_dir / "stage1_results.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if lock["preregister_sha256"] != PREREGISTER_SHA256:
        raise AssertionError("Stage1 preregister changed")
    if lock["stage1_results"]["sha256"] != sha256_file(result_path):
        raise AssertionError("Stage1 results changed")
    selected, audit = select_stage1_weight(result["comparisons"], weights=WEIGHTS)
    if selected != result["locked_weight"] or selected != lock["locked_weight"]:
        raise AssertionError("Stage1 selection changed")
    if audit != result["selection"]:
        raise AssertionError("Stage1 selection audit changed")
    return lock, result


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
    _copy_exclusive(preregister_path, out_dir / "preregister.json")
    _copy_exclusive(preregister_path.with_suffix(".sha256"), out_dir / "preregister.sha256")
    provenance_before = _snapshot_named(_provenance_paths(preregister_path))
    input_before = cb_protocol._stage1_input_snapshot(raw_dir, artifact_root, preregister)
    cb_protocol._assert_preregistered_stage1_files(input_before, preregister)

    full_index = pd.date_range(
        "2022-01-01 01:00:00",
        "2024-01-01 00:00:00",
        freq="h",
        name="forecast_kst_dtm",
    )
    raw_features, raw_contract = shared._read_stage1_raw_features(
        raw_dir, pd.DataFrame(index=full_index)
    )
    features, feature_contract = _seasonal_features(raw_features)
    baseline = cb_protocol._load_stage1_baseline(artifact_root)
    segments = cb_protocol._year_segments(2023)
    fit_indexes = {
        "kpx_group_1": strict._year_index(2022),
        "kpx_group_2": strict._year_index(2022),
        "kpx_group_3": segments["H1"],
    }
    application_indexes = {
        "kpx_group_1": segments["full"],
        "kpx_group_2": segments["full"],
        "kpx_group_3": segments["H2"],
    }
    label_path = raw_dir / "train/train_labels.csv"
    specs = preregister["physical_stage1_inputs"]
    labels_g12, g12_label_evidence = cb_protocol._bounded_label_prefix(
        label_path, specs["labels_g12_fit_prefix"]
    )
    models: dict[str, SeasonalWindRidge] = {}
    predictions: dict[str, pd.Series] = {}
    training: dict[str, Any] = {}
    g12_paths: list[Path] = []
    for group in TARGET_COLS[:2]:
        model, prediction, metadata = _fit_predict(
            group=group,
            features=features[group],
            actual=labels_g12[group],
            fit_index=fit_indexes[group],
            application_index=application_indexes[group],
        )
        models[group] = model
        predictions[group] = prediction
        training[group] = metadata
        path = out_dir / "oof" / f"stage1_{group}_direct_cf.parquet"
        strict._atomic_parquet(prediction.to_frame(), path)
        g12_paths.append(path)
    g12_model_path = out_dir / "models/stage1_g12_models.joblib"
    strict._atomic_joblib({group: models[group] for group in TARGET_COLS[:2]}, g12_model_path)
    g12_paths.append(g12_model_path)
    g12_reload = _reload_audit(
        model_path=g12_model_path,
        groups=TARGET_COLS[:2],
        features=features,
        application_indexes=application_indexes,
        expected=predictions,
    )
    g12_lock_path = out_dir / "stage1_g12_prescore_lock.json"
    strict._write_json(
        g12_lock_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "groups": list(TARGET_COLS[:2]),
            "fit_label_evidence": g12_label_evidence,
            "application_target_value_cells_materialized": 0,
            "outputs": [describe_file(path) for path in g12_paths],
            "reload_audit": g12_reload,
            "created_before_g3_fit_prefix_and_all_stage1_score_labels": True,
        },
    )

    labels_g3, g3_label_evidence = cb_protocol._bounded_label_prefix(
        label_path, specs["labels_g3_fit_prefix"]
    )
    group = "kpx_group_3"
    model, prediction, metadata = _fit_predict(
        group=group,
        features=features[group],
        actual=labels_g3[group],
        fit_index=fit_indexes[group],
        application_index=application_indexes[group],
    )
    models[group] = model
    predictions[group] = prediction
    training[group] = metadata
    g3_prediction_path = out_dir / "oof/stage1_kpx_group_3_direct_cf.parquet"
    g3_model_path = out_dir / "models/stage1_g3_model.joblib"
    strict._atomic_parquet(prediction.to_frame(), g3_prediction_path)
    strict._atomic_joblib({group: model}, g3_model_path)
    g3_reload = _reload_audit(
        model_path=g3_model_path,
        groups=(group,),
        features=features,
        application_indexes=application_indexes,
        expected=predictions,
    )
    g3_lock_path = out_dir / "stage1_g3_prescore_lock.json"
    strict._write_json(
        g3_lock_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "group": group,
            "fit_label_evidence": g3_label_evidence,
            "g12_omitted_target_value_cells_materialized": 0,
            "application_target_value_cells_materialized": 0,
            "outputs": [describe_file(g3_prediction_path), describe_file(g3_model_path)],
            "reload_audit": g3_reload,
            "created_before_all_stage1_score_labels": True,
        },
    )

    direct = pd.DataFrame(np.nan, index=baseline.index, columns=TARGET_COLS)
    for group in TARGET_COLS:
        direct.loc[application_indexes[group], group] = predictions[group]
    zero = _blend_frame(baseline, direct, application_indexes, 0.0)
    zero_exact = all(
        np.ascontiguousarray(zero[group].to_numpy()).tobytes()
        == np.ascontiguousarray(baseline[group].to_numpy()).tobytes()
        for group in TARGET_COLS
    )
    if not zero_exact:
        raise AssertionError("weight-zero identity is not bit exact")
    blends = {
        key: _blend_frame(baseline, direct, application_indexes, weight)
        for key, weight in WEIGHTS.items()
    }
    standalone = (
        _blend_frame(baseline, direct, application_indexes, 1.0)
        if STANDALONE_DIAGNOSTIC
        else None
    )
    global_paths: list[Path] = []
    for name, frame in (("stage1_baseline_2023", baseline), ("stage1_direct_cf_2023", direct)):
        path = out_dir / "oof" / f"{name}.parquet"
        strict._atomic_parquet(frame, path)
        global_paths.append(path)
    for key, frame in blends.items():
        path = out_dir / "oof" / f"stage1_blend_{key}_2023.parquet"
        strict._atomic_parquet(frame, path)
        global_paths.append(path)
    if standalone is not None:
        path = out_dir / "oof/stage1_standalone_diagnostic_2023.parquet"
        strict._atomic_parquet(standalone, path)
        global_paths.append(path)
    all_model_path = out_dir / "models/stage1_all_models.joblib"
    strict._atomic_joblib(models, all_model_path)
    global_paths.append(all_model_path)
    reload_audit = _reload_audit(
        model_path=all_model_path,
        groups=TARGET_COLS,
        features=features,
        application_indexes=application_indexes,
        expected=predictions,
    )
    provenance_after = _snapshot_named(_provenance_paths(preregister_path))
    input_after = cb_protocol._stage1_input_snapshot(raw_dir, artifact_root, preregister)
    shared._assert_snapshot_equal(provenance_before, provenance_after, name="Stage1 provenance")
    shared._assert_snapshot_equal(input_before, input_after, name="Stage1 inputs")
    prescore_record_path = out_dir / "stage1_prescore_record.json"
    strict._write_json(
        prescore_record_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "candidate_weights": WEIGHTS,
            "training": training,
            "raw_feature_contract": raw_contract,
            "low_degree_feature_contract": feature_contract,
            "g12_lock": describe_file(g12_lock_path),
            "g3_lock": describe_file(g3_lock_path),
            "global_outputs": [describe_file(path) for path in global_paths],
            "reload_audit": reload_audit,
            "weight_zero_identity_value_bits_exact": True,
            "standalone_diagnostic_frozen_before_score": standalone is not None,
            "standalone_diagnostic_selectable": False,
            "stage1_application_label_value_cells_materialized": 0,
            "2024_read": False,
            "2025_read": False,
            "multi_year_cache_opened": False,
            "provenance_before": provenance_before,
            "provenance_after": provenance_after,
            "input_before": input_before,
            "input_after": input_after,
        },
    )
    global_lock_path = out_dir / "stage1_global_prescore_lock.json"
    strict._write_json(
        global_lock_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "prescore_record": describe_file(prescore_record_path),
            "candidate_artifacts": [describe_file(path) for path in global_paths],
            "g12_lock": describe_file(g12_lock_path),
            "g3_lock": describe_file(g3_lock_path),
            "direct_model_reload_and_two_blends_frozen": True,
            "standalone_diagnostic_frozen_before_score": standalone is not None,
            "standalone_diagnostic_selectable": False,
            "created_before_stage1_application_label_values": True,
            "2024_read": False,
            "2025_read": False,
        },
    )

    score_labels, score_label_evidence = cb_protocol._bounded_label_prefix(
        label_path, specs["labels_stage1_score_prefix_after_prescore_lock"]
    )
    comparisons: dict[str, Any] = {}
    for key in WEIGHTS:
        comparisons[key] = {}
        for group in TARGET_COLS:
            app_index = application_indexes[group]
            if group in TARGET_COLS[:2]:
                slice_indexes = {
                    name: segments[name] for name in STAGE1_REQUIRED[group]
                }
            else:
                slice_indexes = {
                    "full": segments["H2"],
                    "Q3": segments["Q3"],
                    "Q4": segments["Q4"],
                }
            comparisons[key][group] = strict._comparison(
                score_labels.loc[app_index, group],
                baseline.loc[app_index, group],
                blends[key].loc[app_index, group],
                group,
                slice_indexes,
            )
    standalone_diagnostics: dict[str, Any] = {}
    if standalone is not None:
        for group in TARGET_COLS:
            app_index = application_indexes[group]
            if group in TARGET_COLS[:2]:
                slice_indexes = {
                    name: segments[name] for name in STAGE1_REQUIRED[group]
                }
            else:
                slice_indexes = {
                    "full": segments["H2"],
                    "Q3": segments["Q3"],
                    "Q4": segments["Q4"],
                }
            standalone_diagnostics[group] = strict._comparison(
                score_labels.loc[app_index, group],
                baseline.loc[app_index, group],
                standalone.loc[app_index, group],
                group,
                slice_indexes,
            )
    locked_weight, selection = select_stage1_weight(comparisons, weights=WEIGHTS)
    result = {
        "schema_version": 1,
        "experiment_id": preregister["experiment_id"],
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "global_prescore_lock": describe_file(global_lock_path),
        "score_label_evidence_after_prescore_lock": score_label_evidence,
        "registered_slice_count": 17,
        "comparisons": comparisons,
        "comparisons_sha256": strict._canonical_sha256(comparisons),
        "standalone_diagnostics": standalone_diagnostics,
        "standalone_diagnostics_sha256": strict._canonical_sha256(
            standalone_diagnostics
        ),
        "standalone_excluded_from_selection": True,
        "selection": selection,
        "selection_sha256": strict._canonical_sha256(selection),
        "locked_weight": locked_weight,
        "locked_candidate": "identity" if locked_weight is None else f"blend_{locked_weight}",
        "provenance": provenance_before,
        "2024_read": False,
        "2025_read": False,
        "leaderboard_score_claim": False,
    }
    result_path = out_dir / "stage1_results.json"
    strict._write_json(result_path, result)
    lock = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "global_prescore_lock": describe_file(global_lock_path),
        "stage1_results": describe_file(result_path),
        "comparisons_sha256": result["comparisons_sha256"],
        "selection_sha256": result["selection_sha256"],
        "locked_weight": locked_weight,
        "locked_candidate": result["locked_candidate"],
        "no_2024_reselection_or_retuning": True,
        "2024_read": False,
        "2025_read": False,
    }
    strict._write_json(out_dir / "stage1_promotion_lock.json", lock)
    print(f"Stage1 locked candidate: {result['locked_candidate']}", flush=True)
    return lock


def _postlock_cache_audit(
    *, raw_dir: Path, cache_dir: Path, out_dir: Path
) -> dict[str, Any]:
    lock, _ = _load_stage1_lock(out_dir)
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
            "reason": "identity locked; no cache file opened",
            "checks": {},
        }
        strict._write_json(path, audit)
        return audit
    full_index = pd.date_range(
        "2022-01-01 01:00", "2024-01-01 00:00", freq="h", name="forecast_kst_dtm"
    )
    rebuilt, _ = shared._read_stage1_raw_features(raw_dir, pd.DataFrame(index=full_index))
    rebuilt_low, _ = _seasonal_features(rebuilt)
    checks: dict[str, Any] = {}
    for group in TARGET_COLS:
        cache_path = cache_dir / f"{group}_weather_train.parquet"
        cached = pd.read_parquet(
            cache_path,
            engine="pyarrow",
            filters=[("forecast_kst_dtm", "<=", pd.Timestamp("2024-01-01 00:00"))],
            columns=[WIND_COLUMN],
        )
        cached.index = pd.DatetimeIndex(cached.index, name="forecast_kst_dtm")
        cached_low = build_seasonal_wind_features(cached.index, cached[WIND_COLUMN])
        wind_exact = np.array_equal(
            rebuilt[group][WIND_COLUMN].to_numpy(), cached[WIND_COLUMN].to_numpy()
        )
        design_exact = np.array_equal(
            rebuilt_low[group].to_numpy(), cached_low.to_numpy()
        )
        if not rebuilt[group].index.equals(cached.index) or not wind_exact or not design_exact:
            raise AssertionError(f"{group} selected raw/cache prefix differs")
        checks[group] = {
            "rows": len(cached),
            "selected_weather_columns": [WIND_COLUMN],
            "selected_weather_value_bits_exact": True,
            "design_value_bits_exact": True,
            "cache_file": describe_file(cache_path),
        }
    audit = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "stage1_promotion_lock": describe_file(out_dir / "stage1_promotion_lock.json"),
        "performed": True,
        "selection_locked_before_cache_read": True,
        "2024_cache_values_read": False,
        "checks": checks,
        "all_group_selected_prefixes_value_bit_exact": True,
    }
    strict._write_json(path, audit)
    return audit


def _read_low_cache(
    cache_dir: Path, expected_index: pd.DatetimeIndex
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    weather = shared._read_features(
        cache_dir,
        pd.DataFrame(index=expected_index),
        expected_end=expected_index.max(),
    )
    return _seasonal_features(weather)


def _mixed_comparisons(
    labels: pd.DataFrame,
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    segments: Mapping[str, pd.DatetimeIndex],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name in STAGE2_REQUIRED:
        index = segments[name]
        base = score_details(labels.loc[index], baseline.loc[index]).as_dict()
        cand = score_details(labels.loc[index], candidate.loc[index]).as_dict()
        output[name] = {
            "baseline": base,
            "candidate": cand,
            "delta_total_score": cand["total_score"] - base["total_score"],
            "delta_one_minus_nmae": cand["one_minus_nmae"] - base["one_minus_nmae"],
            "delta_ficr": cand["ficr"] - base["ficr"],
        }
    return output


def _stage2(
    *,
    raw_dir: Path,
    artifact_root: Path,
    cache_dir: Path,
    out_dir: Path,
    preregister: Mapping[str, Any],
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
            "group_comparisons": {},
            "mixed_comparisons": {},
            "promotion_audit": {},
            "reason": "No global weight passed every Stage1 slice",
        }
        result_path = out_dir / "stage2_results.json"
        strict._write_json(result_path, result)
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
        strict._write_json(out_dir / "stage2_promotion_lock.json", lock)
        return lock
    if not cache_audit.get("all_group_selected_prefixes_value_bit_exact"):
        raise AssertionError("postlock cache audit did not pass")

    label_path = raw_dir / "train/train_labels.csv"
    source_labels, source_evidence = cb_protocol._bounded_label_prefix(
        label_path,
        preregister["physical_stage1_inputs"]["labels_stage1_score_prefix_after_prescore_lock"],
    )
    expected_index = pd.date_range(
        "2022-01-01 01:00", "2025-01-01 00:00", freq="h", name="forecast_kst_dtm"
    )
    features, feature_evidence = _read_low_cache(cache_dir, expected_index)
    index_2024 = strict._year_index(2024)
    baseline = strict._read_prediction(
        artifact_root / "oof/gate2024_locked_v3_cf_fix.parquet",
        index_2024,
        required_columns=TARGET_COLS,
    )
    fit_indexes = {
        "kpx_group_1": source_labels.index,
        "kpx_group_2": source_labels.index,
        "kpx_group_3": strict._year_index(2023),
    }
    app_indexes = {group: index_2024 for group in TARGET_COLS}
    models: dict[str, SeasonalWindRidge] = {}
    predictions: dict[str, pd.Series] = {}
    training: dict[str, Any] = {}
    direct = pd.DataFrame(index=index_2024, columns=TARGET_COLS, dtype=float)
    output_paths: list[Path] = []
    for group in TARGET_COLS:
        model, prediction, metadata = _fit_predict(
            group=group,
            features=features[group],
            actual=source_labels[group],
            fit_index=fit_indexes[group],
            application_index=index_2024,
        )
        models[group] = model
        predictions[group] = prediction
        training[group] = metadata
        direct[group] = prediction
        path = out_dir / "oof" / f"stage2_{group}_direct_cf.parquet"
        strict._atomic_parquet(prediction.to_frame(), path)
        output_paths.append(path)
    candidate = _blend_frame(baseline, direct, app_indexes, float(weight))
    standalone = (
        _blend_frame(baseline, direct, app_indexes, 1.0)
        if STANDALONE_DIAGNOSTIC
        else None
    )
    for name, frame in (
        ("stage2_baseline_2024", baseline),
        ("stage2_direct_cf_2024", direct),
        ("stage2_fixed_candidate_2024", candidate),
    ):
        path = out_dir / "oof" / f"{name}.parquet"
        strict._atomic_parquet(frame, path)
        output_paths.append(path)
    if standalone is not None:
        path = out_dir / "oof/stage2_standalone_diagnostic_2024.parquet"
        strict._atomic_parquet(standalone, path)
        output_paths.append(path)
    model_path = out_dir / "models/stage2_models.joblib"
    strict._atomic_joblib(models, model_path)
    output_paths.append(model_path)
    reload_audit = _reload_audit(
        model_path=model_path,
        groups=TARGET_COLS,
        features=features,
        application_indexes=app_indexes,
        expected=predictions,
    )
    prescore_record_path = out_dir / "stage2_prescore_record.json"
    strict._write_json(
        prescore_record_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "stage1_promotion_lock": describe_file(out_dir / "stage1_promotion_lock.json"),
            "postlock_cache_audit": describe_file(cache_audit_path),
            "locked_weight": weight,
            "source_label_evidence": source_evidence,
            "feature_evidence": feature_evidence,
            "training": training,
            "reload_audit": reload_audit,
            "outputs": [describe_file(path) for path in output_paths],
            "standalone_diagnostic_frozen_before_score": standalone is not None,
            "standalone_diagnostic_selectable": False,
            "2024_application_label_value_cells_materialized": 0,
        },
    )
    prescore_lock_path = out_dir / "stage2_2024_prescore_lock.json"
    strict._write_json(
        prescore_lock_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "prescore_record": describe_file(prescore_record_path),
            "candidate_model_reload_weight_frozen": True,
            "created_before_2024_application_label_values": True,
        },
    )
    full_labels = strict._read_full_labels(label_path)
    segments = cb_protocol._year_segments(2024)
    group_comparisons = {
        group: strict._comparison(
            full_labels.loc[index_2024, group],
            baseline[group],
            candidate[group],
            group,
            {name: segments[name] for name in STAGE2_REQUIRED},
        )
        for group in TARGET_COLS
    }
    mixed = _mixed_comparisons(
        full_labels.loc[index_2024], baseline, candidate, segments
    )
    standalone_group_diagnostics: dict[str, Any] = {}
    standalone_mixed_diagnostics: dict[str, Any] = {}
    if standalone is not None:
        standalone_group_diagnostics = {
            group: strict._comparison(
                full_labels.loc[index_2024, group],
                baseline[group],
                standalone[group],
                group,
                {name: segments[name] for name in STAGE2_REQUIRED},
            )
            for group in TARGET_COLS
        }
        standalone_mixed_diagnostics = _mixed_comparisons(
            full_labels.loc[index_2024], baseline, standalone, segments
        )
    promoted, promotion_audit = stage2_promoted(group_comparisons, mixed)
    result = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "executed": True,
        "stage2_prescore_lock": describe_file(prescore_lock_path),
        "locked_weight": weight,
        "2024_read": True,
        "group_comparisons": group_comparisons,
        "mixed_comparisons": mixed,
        "standalone_group_diagnostics": standalone_group_diagnostics,
        "standalone_mixed_diagnostics": standalone_mixed_diagnostics,
        "standalone_excluded_from_promotion": True,
        "comparisons_sha256": strict._canonical_sha256(
            {"groups": group_comparisons, "mixed": mixed}
        ),
        "promotion_audit": promotion_audit,
        "candidate_promoted": promoted,
        "no_2024_reselection_or_retuning": True,
    }
    result_path = out_dir / "stage2_results.json"
    strict._write_json(result_path, result)
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
    strict._write_json(out_dir / "stage2_promotion_lock.json", lock)
    return lock


def _load_stage2_lock(out_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    stage1, _ = _load_stage1_lock(out_dir)
    lock_path = out_dir / "stage2_promotion_lock.json"
    result_path = out_dir / "stage2_results.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if lock["stage2_results"]["sha256"] != sha256_file(result_path):
        raise AssertionError("Stage2 results changed")
    if lock["locked_weight"] != stage1["locked_weight"]:
        raise AssertionError("Stage2 weight changed")
    if result["executed"]:
        promoted, audit = stage2_promoted(
            result["group_comparisons"], result["mixed_comparisons"]
        )
        if promoted != result["candidate_promoted"] or audit != result["promotion_audit"]:
            raise AssertionError("Stage2 promotion changed")
    else:
        promoted = False
    if promoted != bool(lock["candidate_promoted"]):
        raise AssertionError("Stage2 lock promotion changed")
    return lock, result


def _finalize(
    *,
    raw_dir: Path,
    artifact_root: Path,
    cache_dir: Path,
    out_dir: Path,
    preregister_path: Path,
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
        strict._write_json(final_path, final_result)
    else:
        final_inputs = {
            "labels": shared._snapshot_file(raw_dir / "train/train_labels.csv"),
            "sample": shared._snapshot_file(raw_dir / "sample_submission.csv"),
            "baseline": shared._snapshot_file(
                artifact_root / "final_cf_fix/predictions/corrected_v3_test.parquet"
            ),
            **{
                f"train_cache_{group}": shared._snapshot_file(
                    cache_dir / f"{group}_weather_train.parquet"
                )
                for group in TARGET_COLS
            },
            **{
                f"test_cache_{group}": shared._snapshot_file(
                    cache_dir / f"{group}_weather_test.parquet"
                )
                for group in TARGET_COLS
            },
        }
        labels = strict._read_full_labels(raw_dir / "train/train_labels.csv")
        train_weather = shared._read_features(
            cache_dir, labels, expected_end=strict.YEAR_2024_END
        )
        train_features, train_feature_evidence = _seasonal_features(train_weather)
        sample = pd.read_csv(
            raw_dir / "sample_submission.csv",
            encoding="utf-8-sig",
            dtype={"forecast_id": "string", "forecast_kst_dtm": "string"},
        )
        if tuple(sample.columns) != ("forecast_id", "forecast_kst_dtm", *TARGET_COLS):
            raise AssertionError("sample schema changed")
        test_index = pd.DatetimeIndex(
            pd.to_datetime(sample["forecast_kst_dtm"]), name="forecast_kst_dtm"
        )
        if not test_index.equals(strict._year_index(2025)):
            raise AssertionError("sample index changed")
        test_weather = shared._read_test_features(cache_dir, test_index)
        test_features, test_feature_evidence = _seasonal_features(test_weather)
        baseline = strict._read_prediction(
            artifact_root / "final_cf_fix/predictions/corrected_v3_test.parquet",
            test_index,
            required_columns=TARGET_COLS,
        )
        models: dict[str, SeasonalWindRidge] = {}
        predictions: dict[str, pd.Series] = {}
        direct = pd.DataFrame(index=test_index, columns=TARGET_COLS, dtype=float)
        training: dict[str, Any] = {}
        for group in TARGET_COLS:
            joined = pd.concat((train_features[group], test_features[group]), axis=0)
            model, prediction, metadata = _fit_predict(
                group=group,
                features=joined,
                actual=labels[group],
                fit_index=labels.index,
                application_index=test_index,
            )
            models[group] = model
            predictions[group] = prediction
            direct[group] = prediction
            training[group] = metadata
        candidate = _blend_frame(
            baseline,
            direct,
            {group: test_index for group in TARGET_COLS},
            float(lock["locked_weight"]),
        )
        standalone = (
            _blend_frame(
                baseline,
                direct,
                {group: test_index for group in TARGET_COLS},
                1.0,
            )
            if STANDALONE_DIAGNOSTIC
            else None
        )
        model_path = out_dir / "models/final_models.joblib"
        direct_path = out_dir / f"predictions/{OUTPUT_SLUG}_direct_cf_2025.parquet"
        prediction_path = out_dir / f"predictions/{OUTPUT_SLUG}_2025.parquet"
        csv_path = out_dir / f"{OUTPUT_SLUG}_2025.csv"
        strict._atomic_joblib(models, model_path)
        strict._atomic_parquet(direct, direct_path)
        strict._atomic_parquet(candidate, prediction_path)
        standalone_path: Path | None = None
        if standalone is not None:
            standalone_path = (
                out_dir / f"predictions/{OUTPUT_SLUG}_standalone_diagnostic_2025.parquet"
            )
            strict._atomic_parquet(standalone, standalone_path)
        reload_audit = _reload_audit(
            model_path=model_path,
            groups=TARGET_COLS,
            features={
                group: pd.concat((train_features[group], test_features[group]))
                for group in TARGET_COLS
            },
            application_indexes={group: test_index for group in TARGET_COLS},
            expected=predictions,
        )
        submission = sample.copy()
        for group in TARGET_COLS:
            submission[group] = candidate[group].to_numpy(dtype=float)
        strict._atomic_csv(submission, csv_path)
        verification = strict._verify_submission(csv_path, sample, candidate)
        final_result = {
            "schema_version": 1,
            "created_utc": utc_now(),
            "executed": True,
            "locked_weight": lock["locked_weight"],
            "candidate_promoted": True,
            "2025_read": True,
            "submission_created": True,
            "training": training,
            "train_feature_evidence": train_feature_evidence,
            "test_feature_evidence": test_feature_evidence,
            "reload_audit": reload_audit,
            "outputs": [
                describe_file(model_path),
                describe_file(direct_path),
                describe_file(prediction_path),
                describe_file(csv_path),
            ]
            + ([] if standalone_path is None else [describe_file(standalone_path)]),
            "standalone_excluded_from_submission": True,
            "submission_verification": verification,
            "leaderboard_score_claim": False,
        }
        strict._write_json(final_path, final_result)

    outputs = sorted(
        path for path in out_dir.rglob("*") if path.is_file() and path != manifest_path
    )
    stage1_result = json.loads((out_dir / "stage1_results.json").read_text(encoding="utf-8"))
    manifest = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "created_utc": utc_now(),
        "command": [sys.executable, *sys.argv],
        "runtime": {"packages": package_versions(), "git": git_state(PROJECT_DIR)},
        "preregister_sha256": PREREGISTER_SHA256,
        "candidate_weights": WEIGHTS,
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
        "provenance": stage1_result["provenance"],
        "final_inputs": list(final_inputs.values()),
        "outputs": [describe_file(path) for path in outputs],
        "read_flags": {
            "stage1_multi_year_cache_values_read_prelock": False,
            "2024_read": bool(stage2_result.get("2024_read")),
            "2025_read": bool(final_result["2025_read"]),
            "csv_created": bool(final_result["submission_created"]),
        },
        "tests": MANIFEST_TESTS,
        "contracts": MANIFEST_CONTRACTS,
    }
    strict._write_json(manifest_path, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    raw_dir = args.raw_dir.expanduser().resolve()
    artifact_root = args.artifact_root.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    preregister_path = args.preregister.expanduser().resolve()
    preregister = _verify_preregister(preregister_path)
    if args.stage in ("stage1", "all"):
        _stage1(
            raw_dir=raw_dir,
            artifact_root=artifact_root,
            out_dir=out_dir,
            preregister_path=preregister_path,
            preregister=preregister,
        )
        _postlock_cache_audit(raw_dir=raw_dir, cache_dir=cache_dir, out_dir=out_dir)
        if args.stage == "stage1":
            return 0
    if args.stage in ("stage2", "all"):
        _stage2(
            raw_dir=raw_dir,
            artifact_root=artifact_root,
            cache_dir=cache_dir,
            out_dir=out_dir,
            preregister=preregister,
        )
        if args.stage == "stage2":
            return 0
    if args.stage in ("final", "all"):
        _finalize(
            raw_dir=raw_dir,
            artifact_root=artifact_root,
            cache_dir=cache_dir,
            out_dir=out_dir,
            preregister_path=preregister_path,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

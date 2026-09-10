"""Strict-forward direct smooth-FICR custom-objective experiment."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts import run_ficr_bayes_decision_strict as bayes  # noqa: E402
from scripts import run_shared_q07_multiseed as shared_strict  # noqa: E402
from scripts import run_weather_quantile_bayes as weather  # noqa: E402
from src.manifest import (  # noqa: E402
    describe_file,
    git_state,
    package_versions,
    sha256_file,
    utc_now,
)
from src.metric import CAPACITY_KWH, TARGET_COLS, score_details  # noqa: E402
from src.smooth_ficr_objective import (  # noqa: E402
    SmoothFICRLossConfig,
    SmoothFICRRegressor,
    finite_difference_gradient_audit,
)
from src.weather_quantile import blend_with_baseline_kwh  # noqa: E402


PREREGISTER_SHA256 = "f6ce4f8be8ca6d16c42615f3e84a851c502aa08b0651e22b1d8ff281589d018c"
BACKENDS = ("lgb_smooth_ficr", "xgb_smooth_ficr")
BLEND_WEIGHTS = (0.05, 0.10, 0.20)
WEIGHT_KEYS = ("w05", "w10", "w20")
WEIGHT_BY_KEY = dict(zip(WEIGHT_KEYS, BLEND_WEIGHTS))
RECIPE_KEYS = tuple(
    f"{backend}__{weight_key}"
    for backend in BACKENDS
    for weight_key in WEIGHT_KEYS
)
STAGE1_REQUIRED: dict[str, tuple[str, ...]] = {
    "kpx_group_1": ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4"),
    "kpx_group_2": ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4"),
    "kpx_group_3": ("full", "Q3", "Q4"),
}
STAGE2_REQUIRED = ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("stage1", "stage2", "final"), required=True)
    parser.add_argument(
        "--raw-dir", type=Path, default=Path(r"data/local/open")
    )
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/cache"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/postgate/smooth_ficr_objective"),
    )
    parser.add_argument(
        "--preregister",
        type=Path,
        default=Path("configs/smooth_ficr_objective_preregister.json"),
    )
    return parser.parse_args(argv)


def _verify_preregister(
    path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], SmoothFICRLossConfig]:
    observed = sha256_file(path)
    if observed != PREREGISTER_SHA256:
        raise AssertionError(f"preregister hash changed: {observed}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    loss = payload["smooth_official_loss"]
    config = SmoothFICRLossConfig(
        thresholds_cf=tuple(map(float, loss["thresholds_cf"])),
        settlement_weights=tuple(map(float, loss["settlement_decomposition_weights"])),
        mae_coefficient=float(loss["mae_coefficient"]),
        temperature_cf=float(loss["temperature_cf"]),
        smooth_absolute_epsilon_cf=float(loss["smooth_absolute_epsilon_cf"]),
        hessian_floor=float(loss["hessian"]["lower_bound"]),
        hessian_ceiling=float(loss["hessian"]["upper_bound"]),
        sigmoid_clip=float(loss["numerical_contract"]["sigmoid_argument_clip"][1]),
    )
    config.validate()
    candidates = payload["candidate_models"]
    if int(candidates["model_count"]) != 2:
        raise AssertionError("candidate model count changed")
    if tuple(model["name"] for model in candidates["models"]) != BACKENDS:
        raise AssertionError("candidate backend identity/order changed")
    recipe = payload["candidate_recipe_family"]
    if tuple(map(float, recipe["fixed_blend_weights"])) != BLEND_WEIGHTS or int(
        recipe["recipe_count"]
    ) != len(RECIPE_KEYS):
        raise AssertionError("fixed recipe family changed")
    common_payload = candidates["common"]
    common = {
        name: common_payload[name]
        for name in (
            "n_estimators",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "reg_alpha",
            "reg_lambda",
        )
    }
    specs = {
        model["name"]: {
            "backend": model["name"],
            "common_parameters": dict(common),
            "backend_parameters": dict(model["parameters"]),
            "random_state": int(common_payload["seed"]),
            "n_jobs": int(common_payload["n_jobs"]),
        }
        for model in candidates["models"]
    }
    return payload, specs, config


def _provenance_paths(preregister: Path) -> dict[str, Path]:
    return {
        "runner": Path(__file__).resolve(),
        "smooth_objective": PROJECT_DIR / "src" / "smooth_ficr_objective.py",
        "transform": PROJECT_DIR / "src" / "weather_quantile.py",
        "weather_protocol_helper": PROJECT_DIR / "scripts" / "run_weather_quantile_bayes.py",
        "bounded_raw_helper": PROJECT_DIR / "scripts" / "run_shared_q07_multiseed.py",
        "artifact_helper": PROJECT_DIR / "scripts" / "run_ficr_bayes_decision_strict.py",
        "features": PROJECT_DIR / "src" / "features.py",
        "metric": PROJECT_DIR / "src" / "metric.py",
        "manifest": PROJECT_DIR / "src" / "manifest.py",
        "test": PROJECT_DIR / "tests" / "test_smooth_ficr_objective.py",
        "preregister": preregister.resolve(),
    }


def _snapshot_named(paths: Mapping[str, Path]) -> dict[str, Any]:
    return {name: shared_strict._snapshot_file(path) for name, path in paths.items()}


def _fit_backend(
    *,
    backend: str,
    spec: Mapping[str, Any],
    loss_config: SmoothFICRLossConfig,
    features: Mapping[str, pd.DataFrame],
    labels: pd.DataFrame,
    baseline: pd.DataFrame,
    train_indexes: Mapping[str, pd.DatetimeIndex],
    application_indexes: Mapping[str, pd.DatetimeIndex],
) -> tuple[dict[str, SmoothFICRRegressor], pd.DataFrame, dict[str, Any]]:
    models: dict[str, SmoothFICRRegressor] = {}
    predictions = pd.DataFrame(np.nan, index=baseline.index, columns=TARGET_COLS)
    metadata: dict[str, Any] = {}
    for group in TARGET_COLS:
        train_index = train_indexes[group]
        application_index = application_indexes[group]
        if len(train_index.intersection(application_index)) or not (
            train_index.max() < application_index.min()
        ):
            raise AssertionError(f"{backend}/{group} is not strict-forward")
        print(
            f"fit {backend} {group}: {train_index.min()} -> {application_index.min()}",
            flush=True,
        )
        model = SmoothFICRRegressor(
            backend=backend,
            common_parameters=spec["common_parameters"],
            backend_parameters=spec["backend_parameters"],
            random_state=spec["random_state"],
            n_jobs=spec["n_jobs"],
            loss_config=loss_config,
        ).fit(
            features[group].loc[train_index],
            labels.loc[train_index, group],
            capacity_kwh=CAPACITY_KWH[group],
        )
        prediction_cf = model.predict_cf(features[group].loc[application_index])
        predictions.loc[application_index, group] = (
            prediction_cf.to_numpy(dtype=float) * CAPACITY_KWH[group]
        )
        models[group] = model
        metadata[group] = {
            **model.metadata(),
            "train_start": train_index.min(),
            "train_end": train_index.max(),
            "application_start": application_index.min(),
            "application_end": application_index.max(),
            "fit_end_before_application_start": True,
            "fit_application_overlap_count": 0,
        }
    return models, predictions, metadata


def _blend_frame(
    baseline: pd.DataFrame,
    model_prediction: pd.DataFrame,
    application_indexes: Mapping[str, pd.DatetimeIndex],
    weight: float,
) -> pd.DataFrame:
    result = baseline.copy()
    for group in TARGET_COLS:
        index = application_indexes[group]
        result.loc[index, group] = blend_with_baseline_kwh(
            baseline.loc[index, group],
            model_prediction.loc[index, group],
            weight=weight,
            capacity_kwh=CAPACITY_KWH[group],
        )
    return result


def _validate_comparisons(comparisons: Mapping[str, Any]) -> None:
    if tuple(comparisons) != RECIPE_KEYS:
        raise AssertionError("recipe keys/order changed")
    for recipe in RECIPE_KEYS:
        if set(comparisons[recipe]) != set(TARGET_COLS):
            raise AssertionError(f"{recipe} group set changed")
        for group in TARGET_COLS:
            records = comparisons[recipe][group]
            if set(records) != set(STAGE1_REQUIRED[group]):
                raise AssertionError(f"{recipe}/{group} segment set changed")
            for segment in STAGE1_REQUIRED[group]:
                record = records[segment]
                expected = float(record["candidate"]["score"]) - float(
                    record["baseline"]["score"]
                )
                if float(record["delta"]) != expected:
                    raise AssertionError(f"{recipe}/{group}/{segment} delta changed")


def _recipe_parts(recipe: str) -> tuple[str, str, float]:
    backend, weight_key = recipe.rsplit("__", 1)
    if backend not in BACKENDS or weight_key not in WEIGHT_BY_KEY:
        raise ValueError(f"invalid registered recipe: {recipe}")
    return backend, weight_key, WEIGHT_BY_KEY[weight_key]


def _select_recipe(
    comparisons: Mapping[str, Any]
) -> tuple[str | None, dict[str, Any]]:
    _validate_comparisons(comparisons)
    diagnostics: dict[str, Any] = {}
    eligible: list[str] = []
    for recipe in RECIPE_KEYS:
        backend, weight_key, weight = _recipe_parts(recipe)
        deltas = {
            f"{group}/{segment}": float(comparisons[recipe][group][segment]["delta"])
            for group in TARGET_COLS
            for segment in STAGE1_REQUIRED[group]
        }
        diagnostics[recipe] = {
            "backend": backend,
            "weight_key": weight_key,
            "weight": weight,
            "deltas": deltas,
            "minimum": min(deltas.values()),
            "mean": float(np.mean(list(deltas.values()))),
            "all_strictly_positive": all(value > 0.0 for value in deltas.values()),
        }
        if diagnostics[recipe]["all_strictly_positive"]:
            eligible.append(recipe)
    if not eligible:
        return None, {"candidates": diagnostics, "selected": "identity"}
    selected = max(
        eligible,
        key=lambda recipe: (
            diagnostics[recipe]["minimum"],
            diagnostics[recipe]["mean"],
            -diagnostics[recipe]["weight"],
            -BACKENDS.index(diagnostics[recipe]["backend"]),
        ),
    )
    return selected, {"candidates": diagnostics, "selected": selected}


def _stage1(
    *,
    raw_dir: Path,
    artifact_root: Path,
    cache_dir: Path,
    out_dir: Path,
    preregister_path: Path,
    preregister: Mapping[str, Any],
    specs: Mapping[str, Mapping[str, Any]],
    loss_config: SmoothFICRLossConfig,
) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"Stage1 requires a new output directory: {out_dir}")
    out_dir.mkdir(parents=True)
    bayes._copy_exclusive(preregister_path, out_dir / "preregister.json")
    provenance_before = _snapshot_named(_provenance_paths(preregister_path))
    input_before = weather._stage1_input_snapshot(raw_dir, artifact_root)
    gradient_audit = finite_difference_gradient_audit(config=loss_config, step=1e-6)
    if (
        gradient_audit["maximum_absolute_error"] > 2e-5
        or not gradient_audit["surrogate_hessian_strictly_positive"]
    ):
        raise AssertionError("registered objective numerical audit failed")

    labels, label_evidence = bayes._read_bounded_labels(
        raw_dir / "train" / "train_labels.csv"
    )
    features, raw_feature_contract = shared_strict._read_stage1_raw_features(
        raw_dir, labels
    )
    baseline, _ = weather._load_stage1_baseline(artifact_root)
    segments = weather._year_segments(2023)
    train_indexes = {
        "kpx_group_1": bayes._year_index(2022),
        "kpx_group_2": bayes._year_index(2022),
        "kpx_group_3": segments["H1"],
    }
    application_indexes = {
        "kpx_group_1": segments["full"],
        "kpx_group_2": segments["full"],
        "kpx_group_3": segments["H2"],
    }
    all_models: dict[str, dict[str, SmoothFICRRegressor]] = {}
    predictions: dict[str, pd.DataFrame] = {}
    training: dict[str, Any] = {}
    for backend in BACKENDS:
        models, frame, metadata = _fit_backend(
            backend=backend,
            spec=specs[backend],
            loss_config=loss_config,
            features=features,
            labels=labels,
            baseline=baseline,
            train_indexes=train_indexes,
            application_indexes=application_indexes,
        )
        all_models[backend] = models
        predictions[backend] = frame
        training[backend] = metadata

    zero_blend = _blend_frame(
        baseline, predictions[BACKENDS[0]], application_indexes, 0.0
    )
    zero_exact = all(
        np.ascontiguousarray(zero_blend[group].to_numpy()).tobytes()
        == np.ascontiguousarray(baseline[group].to_numpy()).tobytes()
        for group in TARGET_COLS
    )
    if not zero_exact:
        raise AssertionError("zero-weight baseline is not bit exact")
    blends: dict[str, pd.DataFrame] = {}
    for recipe in RECIPE_KEYS:
        backend, _, weight = _recipe_parts(recipe)
        blends[recipe] = _blend_frame(
            baseline, predictions[backend], application_indexes, weight
        )
    comparisons: dict[str, Any] = {}
    for recipe in RECIPE_KEYS:
        comparisons[recipe] = {}
        for group in TARGET_COLS:
            app_index = application_indexes[group]
            segment_indexes = {
                name: segments[name] for name in STAGE1_REQUIRED[group]
            }
            if group == "kpx_group_3":
                segment_indexes["full"] = app_index
            comparisons[recipe][group] = bayes._comparison(
                labels.loc[app_index, group],
                baseline.loc[app_index, group],
                blends[recipe].loc[app_index, group],
                group,
                segment_indexes,
            )
    locked_recipe, selection = _select_recipe(comparisons)

    output_paths: list[Path] = []
    baseline_path = out_dir / "oof" / "stage1_baseline_2023.parquet"
    bayes._atomic_parquet(baseline, baseline_path)
    output_paths.append(baseline_path)
    for backend in BACKENDS:
        prediction_path = out_dir / "oof" / f"stage1_{backend}_2023.parquet"
        model_path = out_dir / "models" / f"stage1_{backend}.joblib"
        bayes._atomic_parquet(predictions[backend], prediction_path)
        bayes._atomic_joblib(all_models[backend], model_path)
        output_paths.extend([prediction_path, model_path])
    for recipe, frame in blends.items():
        path = out_dir / "oof" / f"stage1_{recipe}_2023.parquet"
        bayes._atomic_parquet(frame, path)
        output_paths.append(path)

    provenance_after = _snapshot_named(_provenance_paths(preregister_path))
    input_after = weather._stage1_input_snapshot(raw_dir, artifact_root)
    shared_strict._assert_snapshot_equal(
        provenance_before, provenance_after, name="Stage1 provenance"
    )
    shared_strict._assert_snapshot_equal(
        input_before, input_after, name="Stage1 bounded inputs"
    )
    result = {
        "schema_version": 1,
        "experiment_id": preregister["experiment_id"],
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "backend_count": len(BACKENDS),
        "recipe_count": len(RECIPE_KEYS),
        "recipe_keys": RECIPE_KEYS,
        "model_specs": specs,
        "loss_config": asdict(loss_config),
        "gradient_audit": gradient_audit,
        "2024_read": False,
        "multi_year_cache_values_read_prelock": False,
        "cache_dir_deliberately_excluded_prelock": str(cache_dir.resolve()),
        "label_prefix_evidence": label_evidence,
        "raw_feature_contract": raw_feature_contract,
        "training": training,
        "fit_application_overlap_count": 0,
        "same_row_fit_score_reported": False,
        "zero_weight_baseline_value_bits_exact": zero_exact,
        "comparisons": comparisons,
        "comparisons_sha256": bayes._canonical_sha256(comparisons),
        "selection": selection,
        "locked_recipe": locked_recipe,
        "locked_candidate": "identity" if locked_recipe is None else locked_recipe,
        "provenance_before": provenance_before,
        "provenance_after": provenance_after,
        "provenance_snapshot_sha256": bayes._canonical_sha256(provenance_before),
        "stage1_input_snapshot_before": input_before,
        "stage1_input_snapshot_after": input_after,
        "stage1_input_snapshot_sha256": bayes._canonical_sha256(input_before),
        "snapshots_unchanged": True,
        "outputs_before_result": [describe_file(path) for path in output_paths],
        "leaderboard_score_claim": False,
        "untouched_gate_claim": False,
    }
    result_path = out_dir / "stage1_results.json"
    bayes._write_json(result_path, result)
    lock = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "stage1_results_sha256": sha256_file(result_path),
        "comparisons_sha256": result["comparisons_sha256"],
        "locked_recipe": locked_recipe,
        "locked_candidate": result["locked_candidate"],
        "selection_sha256": bayes._canonical_sha256(selection),
        "provenance_snapshot_sha256": result["provenance_snapshot_sha256"],
        "stage1_input_snapshot_sha256": result["stage1_input_snapshot_sha256"],
        "objective_backend_blend_frozen": True,
        "no_2024_reselection_or_retuning": True,
    }
    bayes._write_json(out_dir / "stage1_recipe_lock.json", lock)
    print(f"Stage1 locked candidate: {result['locked_candidate']}", flush=True)
    return lock


def _load_stage1_lock(out_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    result_path = out_dir / "stage1_results.json"
    lock_path = out_dir / "stage1_recipe_lock.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["preregister_sha256"] != PREREGISTER_SHA256:
        raise AssertionError("Stage1 preregister changed")
    if lock["stage1_results_sha256"] != sha256_file(result_path):
        raise AssertionError("Stage1 result changed after lock")
    if tuple(result["recipe_keys"]) != RECIPE_KEYS or int(
        result["recipe_count"]
    ) != len(RECIPE_KEYS):
        raise AssertionError("Stage1 recipe family changed")
    comparisons = result["comparisons"]
    if bayes._canonical_sha256(comparisons) != result["comparisons_sha256"] or result[
        "comparisons_sha256"
    ] != lock["comparisons_sha256"]:
        raise AssertionError("Stage1 comparisons changed")
    selected, selection = _select_recipe(comparisons)
    if selected != result["locked_recipe"] or selected != lock["locked_recipe"]:
        raise AssertionError("Stage1 locked recipe differs from recomputation")
    if selection != result["selection"] or bayes._canonical_sha256(selection) != lock[
        "selection_sha256"
    ]:
        raise AssertionError("Stage1 selection changed")
    for key, digest_key in (
        ("provenance_before", "provenance_snapshot_sha256"),
        ("stage1_input_snapshot_before", "stage1_input_snapshot_sha256"),
    ):
        snapshot = result[key]
        if bayes._canonical_sha256(snapshot) != result[digest_key] or result[
            digest_key
        ] != lock[digest_key]:
            raise AssertionError(f"Stage1 {key} digest changed")
        shared_strict._assert_snapshot_equal(
            snapshot, shared_strict._refresh_snapshot(snapshot), name=f"current {key}"
        )
    return lock, result


def _postlock_cache_audit(
    *, raw_dir: Path, cache_dir: Path, out_dir: Path
) -> dict[str, Any]:
    lock, result = _load_stage1_lock(out_dir)
    path = out_dir / "postlock_cache_audit.json"
    if path.exists():
        raise FileExistsError(path)
    if lock["locked_recipe"] is None:
        audit = {
            "schema_version": 1,
            "created_utc": utc_now(),
            "stage1_recipe_lock_sha256": sha256_file(
                out_dir / "stage1_recipe_lock.json"
            ),
            "locked_recipe": None,
            "performed": False,
            "2024_cache_values_read": False,
            "reason": "identity locked; preregistered no-2024-read branch",
            "checks": {},
        }
        bayes._write_json(path, audit)
        return audit
    labels, _ = bayes._read_bounded_labels(raw_dir / "train" / "train_labels.csv")
    rebuilt, _ = shared_strict._read_stage1_raw_features(raw_dir, labels)
    checks: dict[str, Any] = {}
    for group in TARGET_COLS:
        cache_path = cache_dir / f"{group}_weather_train.parquet"
        cached = pd.read_parquet(cache_path, engine="pyarrow").iloc[
            : shared_strict.EXPECTED_ROWS_PRE2024
        ]
        cached.index = pd.DatetimeIndex(cached.index, name="forecast_kst_dtm")
        observed = rebuilt[group]
        conditions = {
            "index_exact": observed.index.equals(cached.index),
            "column_order_exact": tuple(observed.columns) == tuple(cached.columns),
            "dtype_exact": tuple(map(str, observed.dtypes))
            == tuple(map(str, cached.dtypes)),
            "value_bits_exact": np.ascontiguousarray(
                observed.to_numpy(copy=False)
            ).tobytes()
            == np.ascontiguousarray(cached.to_numpy(copy=False)).tobytes(),
        }
        if not all(conditions.values()):
            raise AssertionError(f"{group} bounded raw/cache prefix differs")
        checks[group] = {
            **conditions,
            "rows": len(observed),
            "columns": observed.shape[1],
            "frame_sha256": shared_strict._frame_sha256(observed),
            "cache_file": describe_file(cache_path),
        }
    audit = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "stage1_recipe_lock_sha256": sha256_file(
            out_dir / "stage1_recipe_lock.json"
        ),
        "locked_recipe": lock["locked_recipe"],
        "performed": True,
        "2024_cache_values_read": True,
        "selection_was_immutable_before_cache_read": True,
        "checks": checks,
        "raw_feature_hashes": result["raw_feature_contract"]["built_features"],
    }
    bayes._write_json(path, audit)
    return audit


def _stage2_promotion(
    comparisons: Mapping[str, Any]
) -> tuple[bool, dict[str, Any]]:
    if set(comparisons) != set(TARGET_COLS):
        raise AssertionError("Stage2 group set changed")
    audit: dict[str, Any] = {}
    for group in TARGET_COLS:
        records = comparisons[group]
        if set(records) != set(STAGE2_REQUIRED):
            raise AssertionError(f"Stage2 {group} segment set changed")
        deltas: dict[str, float] = {}
        for segment in STAGE2_REQUIRED:
            record = records[segment]
            expected = float(record["candidate"]["score"]) - float(
                record["baseline"]["score"]
            )
            if float(record["delta"]) != expected:
                raise AssertionError(f"Stage2 {group}/{segment} delta changed")
            deltas[segment] = expected
        audit[group] = {
            "deltas": deltas,
            "all_strictly_positive": all(value > 0.0 for value in deltas.values()),
        }
    return all(value["all_strictly_positive"] for value in audit.values()), audit


def _write_stage2_lock(out_dir: Path, result: Mapping[str, Any]) -> dict[str, Any]:
    result_path = out_dir / "stage2_results.json"
    bayes._write_json(result_path, result)
    lock = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "stage1_recipe_lock_sha256": sha256_file(
            out_dir / "stage1_recipe_lock.json"
        ),
        "stage2_results_sha256": sha256_file(result_path),
        "locked_recipe": result["locked_recipe"],
        "candidate_promoted": bool(result["candidate_promoted"]),
        "promotion_audit_sha256": bayes._canonical_sha256(
            result["promotion_audit"]
        ),
        "no_2024_reselection_or_retuning": True,
        "csv_allowed": bool(result["candidate_promoted"]),
    }
    bayes._write_json(out_dir / "stage2_promotion_lock.json", lock)
    return lock


def _stage2(
    *,
    raw_dir: Path,
    artifact_root: Path,
    cache_dir: Path,
    out_dir: Path,
    specs: Mapping[str, Mapping[str, Any]],
    loss_config: SmoothFICRLossConfig,
) -> dict[str, Any]:
    lock, _ = _load_stage1_lock(out_dir)
    if not (out_dir / "postlock_cache_audit.json").is_file():
        raise AssertionError("post-lock cache audit is required")
    if (out_dir / "stage2_results.json").exists() or (
        out_dir / "stage2_promotion_lock.json"
    ).exists():
        raise FileExistsError("Stage2 outputs already exist")
    recipe = lock["locked_recipe"]
    if recipe is None:
        return _write_stage2_lock(
            out_dir,
            {
                "schema_version": 1,
                "created_utc": utc_now(),
                "stage1_recipe_lock_sha256": sha256_file(
                    out_dir / "stage1_recipe_lock.json"
                ),
                "locked_recipe": None,
                "2024_read": False,
                "comparisons": {},
                "promotion_audit": {},
                "candidate_promoted": False,
                "reason": "No global recipe passed every Stage1 group/segment",
                "no_2024_reselection_or_retuning": True,
            },
        )
    backend, _, weight = _recipe_parts(recipe)
    input_before = weather._stage2_input_snapshot(raw_dir, cache_dir, artifact_root)
    labels = bayes._read_full_labels(raw_dir / "train" / "train_labels.csv")
    features = shared_strict._read_features(
        cache_dir, labels, expected_end=bayes.YEAR_2024_END
    )
    index_2024 = bayes._year_index(2024)
    baseline = bayes._read_prediction(
        artifact_root / "oof" / "gate2024_locked_v3_cf_fix.parquet",
        index_2024,
        required_columns=TARGET_COLS,
    )
    train_indexes = {
        "kpx_group_1": pd.date_range(
            bayes.YEAR_2022_START,
            bayes.YEAR_2023_END,
            freq="h",
            name="forecast_kst_dtm",
        ),
        "kpx_group_2": pd.date_range(
            bayes.YEAR_2022_START,
            bayes.YEAR_2023_END,
            freq="h",
            name="forecast_kst_dtm",
        ),
        "kpx_group_3": bayes._year_index(2023),
    }
    application_indexes = {group: index_2024 for group in TARGET_COLS}
    models, raw_prediction, training = _fit_backend(
        backend=backend,
        spec=specs[backend],
        loss_config=loss_config,
        features=features,
        labels=labels,
        baseline=baseline,
        train_indexes=train_indexes,
        application_indexes=application_indexes,
    )
    candidate = _blend_frame(
        baseline, raw_prediction, application_indexes, weight
    )
    segments = weather._year_segments(2024)
    comparisons = {
        group: bayes._comparison(
            labels.loc[index_2024, group],
            baseline[group],
            candidate[group],
            group,
            {name: segments[name] for name in STAGE2_REQUIRED},
        )
        for group in TARGET_COLS
    }
    promoted, promotion_audit = _stage2_promotion(comparisons)
    baseline_metrics = score_details(labels.loc[index_2024], baseline).as_dict()
    candidate_metrics = score_details(labels.loc[index_2024], candidate).as_dict()
    output_paths: list[Path] = []
    for name, frame in (
        ("stage2_baseline_2024", baseline),
        ("stage2_raw_model_2024", raw_prediction),
        ("stage2_fixed_candidate_2024", candidate),
    ):
        path = out_dir / "oof" / f"{name}.parquet"
        bayes._atomic_parquet(frame, path)
        output_paths.append(path)
    model_path = out_dir / "models" / "stage2_models.joblib"
    bayes._atomic_joblib(models, model_path)
    output_paths.append(model_path)
    input_after = weather._stage2_input_snapshot(raw_dir, cache_dir, artifact_root)
    shared_strict._assert_snapshot_equal(input_before, input_after, name="Stage2 inputs")
    result = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "stage1_recipe_lock_sha256": sha256_file(
            out_dir / "stage1_recipe_lock.json"
        ),
        "locked_recipe": recipe,
        "backend": backend,
        "blend_weight": weight,
        "2024_read": True,
        "training": training,
        "fit_application_overlap_count": 0,
        "same_2024_fit_score_reported": False,
        "comparisons": comparisons,
        "comparisons_sha256": bayes._canonical_sha256(comparisons),
        "promotion_audit": promotion_audit,
        "candidate_promoted": promoted,
        "official_full_metrics": {
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "delta_total_score": candidate_metrics["total_score"]
            - baseline_metrics["total_score"],
        },
        "no_2024_reselection_or_retuning": True,
        "stage2_input_snapshot_before": input_before,
        "stage2_input_snapshot_after": input_after,
        "stage2_input_snapshot_sha256": bayes._canonical_sha256(input_before),
        "outputs_before_result": [describe_file(path) for path in output_paths],
        "leaderboard_score_claim": False,
        "untouched_gate_claim": False,
    }
    return _write_stage2_lock(out_dir, result)


def _load_stage2_lock(out_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    stage1_lock, _ = _load_stage1_lock(out_dir)
    result_path = out_dir / "stage2_results.json"
    lock_path = out_dir / "stage2_promotion_lock.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["stage2_results_sha256"] != sha256_file(result_path):
        raise AssertionError("Stage2 result changed after lock")
    if result["locked_recipe"] != stage1_lock["locked_recipe"] or lock[
        "locked_recipe"
    ] != stage1_lock["locked_recipe"]:
        raise AssertionError("Stage2 recipe differs from Stage1 lock")
    if result.get("2024_read"):
        promoted, audit = _stage2_promotion(result["comparisons"])
        before = result["stage2_input_snapshot_before"]
        shared_strict._assert_snapshot_equal(
            before, result["stage2_input_snapshot_after"], name="recorded Stage2 inputs"
        )
        shared_strict._assert_snapshot_equal(
            before, shared_strict._refresh_snapshot(before), name="current Stage2 inputs"
        )
    else:
        promoted, audit = False, {}
    if promoted != bool(result["candidate_promoted"]) or audit != result[
        "promotion_audit"
    ]:
        raise AssertionError("Stage2 promotion changed")
    if promoted != bool(lock["candidate_promoted"]) or promoted != bool(
        lock["csv_allowed"]
    ):
        raise AssertionError("Stage2 promotion lock changed")
    if bayes._canonical_sha256(audit) != lock["promotion_audit_sha256"]:
        raise AssertionError("Stage2 promotion audit changed")
    return lock, result


def _finalize(
    *,
    raw_dir: Path,
    artifact_root: Path,
    cache_dir: Path,
    out_dir: Path,
    specs: Mapping[str, Mapping[str, Any]],
    loss_config: SmoothFICRLossConfig,
) -> dict[str, Any]:
    lock, stage2_result = _load_stage2_lock(out_dir)
    final_result_path = out_dir / "final_results.json"
    manifest_path = out_dir / "manifest.json"
    if final_result_path.exists() or manifest_path.exists():
        raise FileExistsError("final outputs already exist")
    final_inputs: dict[str, Any] = {}
    output_details: dict[str, Any] = {}
    if not lock["candidate_promoted"]:
        final_result = {
            "schema_version": 1,
            "created_utc": utc_now(),
            "locked_recipe": lock["locked_recipe"],
            "candidate_promoted": False,
            "2025_read": False,
            "final_fit_performed": False,
            "submission_created": False,
            "reason": "The locked global recipe failed a preregistered gate",
            "leaderboard_score_claim": False,
        }
        bayes._write_json(final_result_path, final_result)
    else:
        recipe = str(lock["locked_recipe"])
        backend, _, weight = _recipe_parts(recipe)
        final_inputs = weather._final_input_snapshot(raw_dir, cache_dir, artifact_root)
        labels = bayes._read_full_labels(raw_dir / "train" / "train_labels.csv")
        train_features = shared_strict._read_features(
            cache_dir, labels, expected_end=bayes.YEAR_2024_END
        )
        sample = pd.read_csv(
            raw_dir / "sample_submission.csv",
            encoding="utf-8-sig",
            dtype={"forecast_id": "string", "forecast_kst_dtm": "string"},
        )
        if tuple(sample.columns) != ("forecast_id", "forecast_kst_dtm", *TARGET_COLS):
            raise AssertionError("sample schema changed")
        test_index = pd.DatetimeIndex(
            pd.to_datetime(sample["forecast_kst_dtm"], errors="raise"),
            name="forecast_kst_dtm",
        )
        if not test_index.equals(bayes._year_index(2025)):
            raise AssertionError("sample 2025 index changed")
        test_features = shared_strict._read_test_features(cache_dir, test_index)
        full_features = {
            group: pd.concat(
                [train_features[group], test_features[group]], axis=0, copy=False
            )
            for group in TARGET_COLS
        }
        baseline = bayes._read_prediction(
            artifact_root
            / "final_cf_fix"
            / "predictions"
            / "corrected_v3_test.parquet",
            test_index,
            required_columns=TARGET_COLS,
        )
        train_indexes = {group: labels.index for group in TARGET_COLS}
        application_indexes = {group: test_index for group in TARGET_COLS}
        models, raw_prediction, training = _fit_backend(
            backend=backend,
            spec=specs[backend],
            loss_config=loss_config,
            features=full_features,
            labels=labels,
            baseline=baseline,
            train_indexes=train_indexes,
            application_indexes=application_indexes,
        )
        candidate = _blend_frame(
            baseline, raw_prediction, application_indexes, weight
        )
        prediction_path = out_dir / "predictions" / "smooth_ficr_2025.parquet"
        raw_path = out_dir / "predictions" / "smooth_ficr_raw_2025.parquet"
        model_path = out_dir / "models" / "final_models.joblib"
        csv_path = out_dir / "smooth_ficr_objective_2025.csv"
        bayes._atomic_parquet(candidate, prediction_path)
        bayes._atomic_parquet(raw_prediction, raw_path)
        bayes._atomic_joblib(models, model_path)
        submission = sample.copy()
        for group in TARGET_COLS:
            submission[group] = candidate[group].to_numpy(dtype=float)
        bayes._atomic_csv(submission, csv_path)
        verification = bayes._verify_submission(csv_path, sample, candidate)
        output_details = {
            "prediction": describe_file(prediction_path),
            "raw_prediction": describe_file(raw_path),
            "models": describe_file(model_path),
            "submission": describe_file(csv_path),
            "submission_verification": verification,
        }
        final_result = {
            "schema_version": 1,
            "created_utc": utc_now(),
            "locked_recipe": recipe,
            "candidate_promoted": True,
            "2025_read": True,
            "final_fit_performed": True,
            "training": training,
            "fit_application_overlap_count": 0,
            "same_fit_score_reported": False,
            "submission_created": True,
            "outputs": output_details,
            "leaderboard_score_claim": False,
        }
        bayes._write_json(final_result_path, final_result)

    stage1_result = json.loads(
        (out_dir / "stage1_results.json").read_text(encoding="utf-8")
    )
    input_records = [
        *stage1_result["provenance_before"].values(),
        *stage1_result["stage1_input_snapshot_before"].values(),
        describe_file(out_dir / "postlock_cache_audit.json"),
    ]
    if stage2_result.get("2024_read"):
        input_records.extend(stage2_result["stage2_input_snapshot_before"].values())
    input_records.extend(final_inputs.values())
    outputs = sorted(
        path for path in out_dir.rglob("*") if path.is_file() and path != manifest_path
    )
    manifest = {
        "schema_version": 1,
        "artifact_type": "smooth_ficr_objective_strict_forward_v1",
        "created_utc": utc_now(),
        "command": [sys.executable, *sys.argv],
        "runtime": {"packages": package_versions(), "git": git_state(PROJECT_DIR)},
        "preregister_sha256": PREREGISTER_SHA256,
        "backends": BACKENDS,
        "candidate_weights": BLEND_WEIGHTS,
        "locked_recipe": lock["locked_recipe"],
        "candidate_promoted": bool(lock["candidate_promoted"]),
        "stage1_recipe_lock": describe_file(out_dir / "stage1_recipe_lock.json"),
        "stage2_promotion_lock": describe_file(
            out_dir / "stage2_promotion_lock.json"
        ),
        "inputs": bayes._deduplicate_records(input_records),
        "outputs": [describe_file(path) for path in outputs],
        "results": {
            "stage2_official_full_metrics": stage2_result.get(
                "official_full_metrics"
            ),
            "2024_read": bool(stage2_result.get("2024_read")),
            "2025_read": bool(final_result["2025_read"]),
            "submission_created": bool(final_result["submission_created"]),
            "output_details": output_details,
        },
        "contracts": {
            "stage1_multi_year_cache_values_read_prelock": False,
            "stage1_raw_nwp_physically_byte_capped": True,
            "exact_gradient_finite_difference_tested": True,
            "positive_surrogate_hessian": True,
            "true_hessian_claim": False,
            "fit_transforms_train_only": True,
            "2024_used_for_candidate_selection": False,
            "no_2024_reselection_or_retuning": True,
            "csv_requires_complete_promotion": True,
            "2024_historically_consumed": True,
            "untouched_gate_claim": False,
            "leaderboard_score_claim": False,
        },
    }
    bayes._write_json(manifest_path, manifest)
    print(
        f"Finalized promoted={lock['candidate_promoted']} manifest={manifest_path}",
        flush=True,
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    raw_dir = args.raw_dir.expanduser().resolve()
    artifact_root = args.artifact_root.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    preregister_path = args.preregister.expanduser().resolve()
    preregister, specs, loss_config = _verify_preregister(preregister_path)
    if out_dir == artifact_root or artifact_root not in out_dir.parents:
        raise ValueError("out-dir must be a distinct child of artifact-root")
    if args.stage == "stage1":
        _stage1(
            raw_dir=raw_dir,
            artifact_root=artifact_root,
            cache_dir=cache_dir,
            out_dir=out_dir,
            preregister_path=preregister_path,
            preregister=preregister,
            specs=specs,
            loss_config=loss_config,
        )
        _postlock_cache_audit(raw_dir=raw_dir, cache_dir=cache_dir, out_dir=out_dir)
    elif args.stage == "stage2":
        _stage2(
            raw_dir=raw_dir,
            artifact_root=artifact_root,
            cache_dir=cache_dir,
            out_dir=out_dir,
            specs=specs,
            loss_config=loss_config,
        )
    else:
        _finalize(
            raw_dir=raw_dir,
            artifact_root=artifact_root,
            cache_dir=cache_dir,
            out_dir=out_dir,
            specs=specs,
            loss_config=loss_config,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

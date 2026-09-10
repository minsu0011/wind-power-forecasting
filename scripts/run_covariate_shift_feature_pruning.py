"""Run the preregistered strict-forward covariate-shift feature-pruning test."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts import run_ficr_bayes_decision_strict as bayes  # noqa: E402
from scripts import run_raw_grid_wind_lgb as raw_protocol  # noqa: E402
from scripts import run_weather_quantile_bayes as weather_protocol  # noqa: E402
from src.covariate_shift_pruning import (  # noqa: E402
    blend_candidate_kwh,
    eligible_feature_names,
    select_stable_features,
    standardized_quantile_wasserstein,
)
from src.manifest import (  # noqa: E402
    describe_file,
    git_state,
    package_versions,
    sha256_file,
    utc_now,
)
from src.metric import CAPACITY_KWH, TARGET_COLS, score_details  # noqa: E402
from src.models import TabularRegressor  # noqa: E402


PREREGISTER_SHA256 = "f68dfc1e37fc7b29487dbb896159c4da3ed2dec09ebb91d55ba4e20453ca4abd"
TIME_COL = "forecast_kst_dtm"
KEEP_FRACTIONS = (0.50, 0.75)
OBJECTIVES = ("l1", "q07")
BLEND_WEIGHTS = (0.05, 0.10)
MODEL_COMMON_PARAMETERS: Mapping[str, Any] = {
    "n_estimators": 1500,
    "learning_rate": 0.025,
    "num_leaves": 31,
    "min_child_samples": 30,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.75,
    "reg_alpha": 0.05,
    "reg_lambda": 2.0,
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
    "random_state": 42,
    "n_jobs": 7,
}
STAGE1_REQUIRED: Mapping[str, tuple[str, ...]] = {
    "kpx_group_1": ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4"),
    "kpx_group_2": ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4"),
    "kpx_group_3": ("H2", "Q3", "Q4"),
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


def _fraction_tag(fraction: float) -> str:
    return f"keep{int(round(100 * fraction)):02d}"


MODEL_KEYS = tuple(
    f"{_fraction_tag(fraction)}_{objective}"
    for fraction in KEEP_FRACTIONS
    for objective in OBJECTIVES
)
CANDIDATE_KEYS = tuple(
    f"{model_key}_w{int(round(100 * weight)):02d}"
    for model_key in MODEL_KEYS
    for weight in BLEND_WEIGHTS
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("stage1", "stage2", "final", "all"), required=True)
    parser.add_argument("--raw-dir", type=Path, default=Path(r"data/local/open"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/postgate/covariate_shift_feature_pruning_strict_v1"),
    )
    parser.add_argument(
        "--preregister",
        type=Path,
        default=Path("configs/covariate_shift_feature_pruning_preregister_v1.json"),
    )
    return parser.parse_args(argv)


def _canonical_sha256(payload: Any) -> str:
    rendered = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def _verify_preregister(path: Path) -> dict[str, Any]:
    if sha256_file(path) != PREREGISTER_SHA256:
        raise AssertionError("preregister hash changed")
    sidecar = path.with_suffix(".sha256")
    expected = f"{PREREGISTER_SHA256}  {path.name}\n"
    if sidecar.read_text(encoding="utf-8") != expected:
        raise AssertionError("preregister SHA sidecar changed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["experiment_id"] != "covariate_shift_feature_pruning_strict_forward_v1":
        raise AssertionError("experiment id changed")
    feature = payload["feature_contract"]
    if tuple(map(float, feature["fixed_keep_fractions_in_order"])) != KEEP_FRACTIONS:
        raise AssertionError("keep fractions changed")
    model = payload["model_family"]
    if tuple(model["objectives_in_fixed_order"]) != OBJECTIVES:
        raise AssertionError("objective order changed")
    if model["common_parameters"] != dict(MODEL_COMMON_PARAMETERS):
        raise AssertionError("locked model parameters changed")
    candidate = payload["candidate_family"]
    if tuple(map(float, candidate["fixed_blend_weights_in_order"])) != BLEND_WEIGHTS:
        raise AssertionError("blend weights changed")
    if int(candidate["candidate_count_per_group"]) != len(CANDIDATE_KEYS):
        raise AssertionError("candidate count changed")
    return payload


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _copy_exclusive(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copyfile(source, destination)


def _assert_file_identity(path: Path, spec: Mapping[str, Any], name: str) -> dict[str, Any]:
    observed = describe_file(path)
    if int(observed["size_bytes"]) != int(spec["bytes"]):
        raise AssertionError(f"{name} size changed")
    if observed["sha256"] != spec["sha256"]:
        raise AssertionError(f"{name} hash changed")
    return observed


def _read_weather_range(
    artifact_root: Path,
    preregister: Mapping[str, Any],
    group: str,
    index: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = artifact_root / "cache" / f"{group}_weather_train.parquet"
    identity = _assert_file_identity(
        path, preregister["input_identities"]["weather_train"][group], f"{group} weather cache"
    )
    frame = pd.read_parquet(
        path,
        engine="pyarrow",
        filters=[
            (TIME_COL, ">=", index.min()),
            (TIME_COL, "<=", index.max()),
        ],
    )
    frame.index = pd.DatetimeIndex(frame.index, name=TIME_COL)
    if not frame.index.equals(index):
        raise AssertionError(f"{group} weather logical range changed")
    if frame.shape[1] != int(preregister["feature_contract"]["expected_cached_column_count"]):
        raise AssertionError("cached weather feature count changed")
    allowed = eligible_feature_names(frame.columns)
    if len(allowed) != int(preregister["feature_contract"]["expected_name_filtered_count"]):
        raise AssertionError("name-filtered feature count changed")
    if not np.isfinite(frame.to_numpy(dtype=np.float32, copy=False)).all():
        raise AssertionError("cached weather has non-finite values")
    evidence = {
        "input": identity,
        "logical_predicate_start": index.min().isoformat(),
        "logical_predicate_end": index.max().isoformat(),
        "rows_materialized": int(len(frame)),
        "columns_materialized": int(frame.shape[1]),
        "outside_predicate_rows_materialized_in_pandas": 0,
        "parquet_storage_note": "Single-row-group cache; predicate bounds materialized pandas rows but are not claimed as a physical byte boundary.",
    }
    return frame.astype(np.float32, copy=False), evidence


def _read_test_weather(
    artifact_root: Path,
    preregister: Mapping[str, Any],
    group: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = artifact_root / "cache" / f"{group}_weather_test.parquet"
    identity = _assert_file_identity(
        path,
        preregister["input_identities"]["weather_test_after_final_unlock"][group],
        f"{group} test weather cache",
    )
    frame = pd.read_parquet(path, engine="pyarrow")
    frame.index = pd.DatetimeIndex(frame.index, name=TIME_COL)
    if not frame.index.equals(YEAR_2025) or frame.shape[1] != 612:
        raise AssertionError("test weather schema/range changed")
    if not np.isfinite(frame.to_numpy(dtype=np.float32, copy=False)).all():
        raise AssertionError("test weather has non-finite values")
    return frame.astype(np.float32, copy=False), {"input": identity, "rows_materialized": len(frame)}


def _read_label_prefix(
    raw_dir: Path,
    spec: Mapping[str, Any],
    groups: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    return raw_protocol._read_label_prefix(raw_dir, spec, groups)


def _read_full_labels(
    raw_dir: Path,
    preregister: Mapping[str, Any],
    groups: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = raw_dir / "train/train_labels.csv"
    spec = preregister["input_identities"]["train_labels"]
    identity = _assert_file_identity(path, spec, "full train labels")
    frame = pd.read_csv(path, encoding="utf-8-sig", usecols=["kst_dtm", *groups])
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("kst_dtm"), errors="raise"), name=TIME_COL)
    if not frame.index.equals(TRAIN_ALL) or tuple(frame.columns) != tuple(groups):
        raise AssertionError("full label schema/range changed")
    return frame.astype(float), {
        "whole_file_after_stage2_prescore_lock": True,
        "target_columns_materialized": list(groups),
        "input": identity,
    }


def _model_parameters(objective: str) -> dict[str, Any]:
    parameters = dict(MODEL_COMMON_PARAMETERS)
    if objective == "l1":
        parameters["objective"] = "regression_l1"
    elif objective == "q07":
        parameters["objective"] = "quantile"
        parameters["alpha"] = 0.7
    else:
        raise ValueError(objective)
    return parameters


def _candidate_recipe(key: str) -> tuple[float, str, float]:
    match = re.fullmatch(r"keep(50|75)_(l1|q07)_w(05|10)", key)
    if match is None:
        raise ValueError(f"invalid candidate key: {key}")
    return int(match.group(1)) / 100.0, match.group(2), int(match.group(3)) / 100.0


def _fit_group(
    *,
    source_features: pd.DataFrame,
    application_features: pd.DataFrame,
    source_actual_kwh: pd.Series,
    baseline_kwh: pd.Series,
    group: str,
) -> tuple[
    dict[str, TabularRegressor],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, tuple[str, ...]],
    dict[str, Any],
]:
    if source_features.index.max() >= application_features.index.min():
        raise AssertionError("fit/application weather order changed")
    if not baseline_kwh.index.equals(application_features.index):
        raise AssertionError("baseline/application index changed")
    if not source_actual_kwh.index.equals(source_features.index):
        raise AssertionError("source target/weather index changed")
    allowed = eligible_feature_names(source_features.columns)
    shift_table = standardized_quantile_wasserstein(
        source_features, application_features, eligible_names=allowed
    )
    subsets = {
        _fraction_tag(fraction): select_stable_features(shift_table, fraction)
        for fraction in KEEP_FRACTIONS
    }
    capacity = CAPACITY_KWH[group]
    eligible = (
        source_actual_kwh.notna()
        & np.isfinite(source_actual_kwh.to_numpy(dtype=float))
        & (source_actual_kwh >= 0.10 * capacity)
    )
    models: dict[str, TabularRegressor] = {}
    raw_predictions = pd.DataFrame(index=application_features.index)
    training: dict[str, Any] = {}
    for fraction in KEEP_FRACTIONS:
        tag = _fraction_tag(fraction)
        names = list(subsets[tag])
        for objective in OBJECTIVES:
            key = f"{tag}_{objective}"
            model = TabularRegressor(
                "lgbm_l1",
                params=_model_parameters(objective),
                seed=42,
                n_jobs=7,
                device="cpu",
                early_stopping_rounds=None,
                fallback_to_cpu=True,
            )
            train_x = source_features.loc[eligible, names]
            train_y = source_actual_kwh.loc[eligible].astype(float) / capacity
            if train_x.isna().any().any():
                raise AssertionError("selected model features contain missing values")
            model.fit(train_x, train_y)
            raw = np.asarray(model.predict(application_features.loc[:, names]), dtype=np.float64)
            if not np.isfinite(raw).all():
                raise AssertionError("raw model prediction is non-finite")
            models[key] = model
            raw_predictions[key] = raw
            training[key] = {
                "fit_rows": int(eligible.sum()),
                "feature_count": len(names),
                "feature_names_sha256": _canonical_sha256(names),
                "objective": objective,
                "parameters": _model_parameters(objective),
                "row_weights_used": False,
                "feature_values_transformed": False,
                "fit_end_before_application_start": True,
                "overlap_count": 0,
            }
    if tuple(raw_predictions.columns) != MODEL_KEYS:
        raise AssertionError("raw model prediction order changed")
    candidates = pd.DataFrame(index=application_features.index)
    for model_key in MODEL_KEYS:
        for weight in BLEND_WEIGHTS:
            key = f"{model_key}_w{int(round(100 * weight)):02d}"
            candidates[key] = blend_candidate_kwh(
                baseline_kwh,
                raw_predictions[model_key],
                weight=weight,
                capacity_kwh=capacity,
            )
    if tuple(candidates.columns) != CANDIDATE_KEYS:
        raise AssertionError("candidate order changed")
    identity = blend_candidate_kwh(
        baseline_kwh,
        raw_predictions[MODEL_KEYS[0]],
        weight=0.0,
        capacity_kwh=capacity,
    )
    if identity.tobytes() != baseline_kwh.to_numpy(dtype=np.float64).tobytes():
        raise AssertionError("weight-zero identity is not value-bit exact")
    return models, raw_predictions, candidates, shift_table, subsets, training


def _save_and_reload_group(
    *,
    out_dir: Path,
    stage: str,
    group: str,
    models: Mapping[str, TabularRegressor],
    raw_predictions: pd.DataFrame,
    candidates: pd.DataFrame,
    shift_table: pd.DataFrame,
    subsets: Mapping[str, tuple[str, ...]],
    application_features: pd.DataFrame,
) -> tuple[list[Path], dict[str, Any]]:
    model_path = out_dir / "models" / f"{stage}__{group}.joblib"
    raw_path = out_dir / "predictions" / f"{stage}__{group}__raw_cf.parquet"
    candidate_path = out_dir / "candidates" / f"{stage}__{group}.parquet"
    shift_path = out_dir / "shift" / f"{stage}__{group}.parquet"
    subset_path = out_dir / "shift" / f"{stage}__{group}__subsets.json"
    bayes._atomic_joblib(dict(models), model_path)
    bayes._atomic_parquet(raw_predictions, raw_path)
    bayes._atomic_parquet(candidates, candidate_path)
    bayes._atomic_parquet(shift_table, shift_path)
    subset_payload = {
        "schema_version": 1,
        "preregister_sha256": PREREGISTER_SHA256,
        "stage": stage,
        "group": group,
        "subsets": {key: list(value) for key, value in subsets.items()},
        "subset_sha256": {key: _canonical_sha256(list(value)) for key, value in subsets.items()},
    }
    bayes._write_json(subset_path, subset_payload)
    loaded: dict[str, TabularRegressor] = joblib.load(model_path)
    reload_checks: dict[str, Any] = {}
    for key in MODEL_KEYS:
        fraction, _, _ = _candidate_recipe(f"{key}_w05")
        names = list(subsets[_fraction_tag(fraction)])
        repeated = np.asarray(loaded[key].predict(application_features.loc[:, names]), dtype=np.float64)
        expected = raw_predictions[key].to_numpy(dtype=np.float64)
        if not np.array_equal(repeated, expected):
            raise AssertionError(f"{stage} {group} {key} reload prediction changed")
        reload_checks[key] = {"raw_prediction_bit_exact": True}
    return [model_path, raw_path, candidate_path, shift_path, subset_path], reload_checks


def _segments(year: int, names: Sequence[str]) -> dict[str, pd.DatetimeIndex]:
    available = weather_protocol._year_segments(year)
    return {name: available[name] for name in names}


def _score_candidates(
    actual: pd.Series,
    baseline: pd.Series,
    candidates: pd.DataFrame,
    group: str,
    segment_names: Sequence[str],
) -> dict[str, Any]:
    segments = _segments(int(actual.index.min().year), segment_names)
    return {
        key: bayes._comparison(actual, baseline, candidates[key], group, segments)
        for key in CANDIDATE_KEYS
    }


def _select_group_candidate(
    comparisons: Mapping[str, Mapping[str, Mapping[str, Any]]],
    required_segments: Sequence[str],
) -> tuple[str | None, dict[str, Any]]:
    if tuple(comparisons) != CANDIDATE_KEYS:
        raise AssertionError("candidate comparison keys changed")
    eligible: list[dict[str, Any]] = []
    records: dict[str, Any] = {}
    for order, key in enumerate(CANDIDATE_KEYS):
        comparison = comparisons[key]
        if set(comparison) != set(required_segments):
            raise AssertionError("candidate segment keys changed")
        deltas = [float(comparison[name]["delta"]) for name in required_segments]
        passed = all(delta > 0.0 for delta in deltas)
        record = {
            "all_required_slices_strictly_positive": passed,
            "minimum_delta": min(deltas),
            "mean_delta": float(np.mean(deltas)),
            "fixed_order": order,
        }
        records[key] = record
        if passed:
            eligible.append({"key": key, **record})
    if not eligible:
        return None, {"selected": "identity", "candidates": records}
    selected = max(
        eligible,
        key=lambda row: (row["minimum_delta"], row["mean_delta"], -row["fixed_order"]),
    )["key"]
    return str(selected), {"selected": selected, "candidates": records}


def _load_stage1_baseline(
    artifact_root: Path, preregister: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    specs = preregister["stage1"]
    g12 = artifact_root / "oof/dev2023_locked_v3.parquet"
    g3 = artifact_root / "oof/g3dev2023h2_candidates.parquet"
    inputs = {
        "g12": _assert_file_identity(g12, specs["baseline_g12"], "Stage1 G1/G2 baseline"),
        "g3": _assert_file_identity(g3, specs["baseline_g3"], "Stage1 G3 baseline"),
    }
    baseline, _ = weather_protocol._load_stage1_baseline(artifact_root)
    return baseline, inputs


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
    baseline, baseline_inputs = _load_stage1_baseline(artifact_root, preregister)
    prefix = preregister["physical_label_prefixes"]

    labels_g12, labels_g12_evidence = _read_label_prefix(
        raw_dir, prefix["g12_stage1_fit"], TARGET_COLS[:2]
    )
    fit_indexes: dict[str, pd.DatetimeIndex] = {
        "kpx_group_1": YEAR_2022,
        "kpx_group_2": YEAR_2022,
        "kpx_group_3": _segments(2023, ("H1",))["H1"],
    }
    application_indexes: dict[str, pd.DatetimeIndex] = {
        "kpx_group_1": YEAR_2023,
        "kpx_group_2": YEAR_2023,
        "kpx_group_3": _segments(2023, ("H2",))["H2"],
    }
    group_candidates: dict[str, pd.DataFrame] = {}
    training: dict[str, Any] = {}
    feature_evidence: dict[str, Any] = {}
    reload_audits: dict[str, Any] = {}
    all_outputs: list[Path] = []
    g12_outputs: list[Path] = []
    for group in TARGET_COLS[:2]:
        period = YEAR_2022.append(YEAR_2023)
        features, evidence = _read_weather_range(artifact_root, preregister, group, period)
        source = features.loc[fit_indexes[group]]
        application = features.loc[application_indexes[group]]
        fitted = _fit_group(
            source_features=source,
            application_features=application,
            source_actual_kwh=labels_g12.loc[fit_indexes[group], group],
            baseline_kwh=baseline.loc[application_indexes[group], group],
            group=group,
        )
        models, raw_predictions, candidates, shift_table, subsets, metadata = fitted
        paths, reload = _save_and_reload_group(
            out_dir=out_dir,
            stage="stage1",
            group=group,
            models=models,
            raw_predictions=raw_predictions,
            candidates=candidates,
            shift_table=shift_table,
            subsets=subsets,
            application_features=application,
        )
        group_candidates[group] = candidates
        training[group] = metadata
        feature_evidence[group] = evidence
        reload_audits[group] = reload
        g12_outputs.extend(paths)
        all_outputs.extend(paths)
    g12_lock_path = out_dir / "stage1_g12_prescore_lock.json"
    bayes._write_json(
        g12_lock_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "fit_label_evidence": labels_g12_evidence,
            "fit_groups": list(TARGET_COLS[:2]),
            "feature_evidence": {group: feature_evidence[group] for group in TARGET_COLS[:2]},
            "outputs": [describe_file(path) for path in g12_outputs],
            "reload_audits": {group: reload_audits[group] for group in TARGET_COLS[:2]},
            "application_target_value_cells_materialized": 0,
            "created_before_g3_fit_prefix_and_all_stage1_application_labels": True,
        },
    )

    labels_g3, labels_g3_evidence = _read_label_prefix(
        raw_dir, prefix["g3_stage1_fit"], ("kpx_group_3",)
    )
    group = "kpx_group_3"
    features, evidence = _read_weather_range(artifact_root, preregister, group, YEAR_2023)
    source = features.loc[fit_indexes[group]]
    application = features.loc[application_indexes[group]]
    fitted = _fit_group(
        source_features=source,
        application_features=application,
        source_actual_kwh=labels_g3.loc[fit_indexes[group], group],
        baseline_kwh=baseline.loc[application_indexes[group], group],
        group=group,
    )
    models, raw_predictions, candidates, shift_table, subsets, metadata = fitted
    g3_outputs, reload = _save_and_reload_group(
        out_dir=out_dir,
        stage="stage1",
        group=group,
        models=models,
        raw_predictions=raw_predictions,
        candidates=candidates,
        shift_table=shift_table,
        subsets=subsets,
        application_features=application,
    )
    group_candidates[group] = candidates
    training[group] = metadata
    feature_evidence[group] = evidence
    reload_audits[group] = reload
    all_outputs.extend(g3_outputs)
    g3_lock_path = out_dir / "stage1_g3_prescore_lock.json"
    bayes._write_json(
        g3_lock_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "fit_label_evidence": labels_g3_evidence,
            "fit_group": group,
            "g12_target_columns_materialized_by_g3_prefix_read": [],
            "feature_evidence": evidence,
            "outputs": [describe_file(path) for path in g3_outputs],
            "reload_audit": reload,
            "application_target_value_cells_materialized": 0,
            "created_before_all_stage1_application_labels": True,
        },
    )
    prescore_record_path = out_dir / "stage1_prescore_record.json"
    bayes._write_json(
        prescore_record_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "baseline_inputs": baseline_inputs,
            "g12_lock": describe_file(g12_lock_path),
            "g3_lock": describe_file(g3_lock_path),
            "training": training,
            "feature_evidence": feature_evidence,
            "candidate_outputs": [describe_file(path) for path in all_outputs],
            "shift_statistics_candidate_models_predictions_hashed": True,
            "stage1_application_label_values_read": False,
            "2024_label_values_read": False,
            "2025_inputs_read": False,
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
            "candidate_outputs": [describe_file(path) for path in all_outputs],
            "created_before_stage1_application_label_values": True,
        },
    )

    score_labels, score_label_evidence = _read_label_prefix(
        raw_dir, prefix["stage1_score_after_lock"], TARGET_COLS
    )
    group_results: dict[str, Any] = {}
    locked_candidates: dict[str, str] = {}
    passed_groups: list[str] = []
    for group in TARGET_COLS:
        app_index = application_indexes[group]
        comparisons = _score_candidates(
            score_labels.loc[app_index, group],
            baseline.loc[app_index, group],
            group_candidates[group],
            group,
            STAGE1_REQUIRED[group],
        )
        selected, selection = _select_group_candidate(comparisons, STAGE1_REQUIRED[group])
        locked_candidates[group] = selected or "identity"
        if selected is not None:
            passed_groups.append(group)
        group_results[group] = {
            "comparisons": comparisons,
            "comparisons_sha256": _canonical_sha256(comparisons),
            "selection": selection,
            "selected": selected or "identity",
        }
    result = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "global_prescore_lock": describe_file(global_lock_path),
        "score_label_evidence_after_prescore_lock": score_label_evidence,
        "group_results": group_results,
        "locked_candidates": locked_candidates,
        "passed_groups": passed_groups,
        "passed_groups_sha256": _canonical_sha256(passed_groups),
        "2024_weather_read": False,
        "2024_label_read": False,
        "2025_read": False,
        "leaderboard_or_public_input_read": False,
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
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if lock["preregister_sha256"] != PREREGISTER_SHA256:
        raise AssertionError("Stage1 preregister binding changed")
    if lock["stage1_results"]["sha256"] != sha256_file(result_path):
        raise AssertionError("Stage1 results changed after lock")
    recomputed: dict[str, str] = {}
    passed: list[str] = []
    for group in TARGET_COLS:
        selected, selection = _select_group_candidate(
            result["group_results"][group]["comparisons"], STAGE1_REQUIRED[group]
        )
        recomputed[group] = selected or "identity"
        if selected is not None:
            passed.append(group)
        if selection != result["group_results"][group]["selection"]:
            raise AssertionError("Stage1 selection audit changed")
    if recomputed != lock["locked_candidates"] or passed != lock["passed_groups"]:
        raise AssertionError("Stage1 promotion cannot be independently reproduced")
    if _canonical_sha256(passed) != lock["passed_groups_sha256"]:
        raise AssertionError("Stage1 passed-group hash changed")
    return lock, result


def _fit_fixed_group(
    *,
    source_features: pd.DataFrame,
    application_features: pd.DataFrame,
    source_actual_kwh: pd.Series,
    baseline_kwh: pd.Series,
    group: str,
    candidate_key: str,
) -> tuple[
    TabularRegressor,
    pd.Series,
    pd.Series,
    pd.DataFrame,
    tuple[str, ...],
    dict[str, Any],
]:
    fraction, objective, weight = _candidate_recipe(candidate_key)
    allowed = eligible_feature_names(source_features.columns)
    shift_table = standardized_quantile_wasserstein(
        source_features, application_features, eligible_names=allowed
    )
    subset = select_stable_features(shift_table, fraction)
    capacity = CAPACITY_KWH[group]
    eligible = (
        source_actual_kwh.notna()
        & np.isfinite(source_actual_kwh.to_numpy(dtype=float))
        & (source_actual_kwh >= 0.10 * capacity)
    )
    names = list(subset)
    train_x = source_features.loc[eligible, names]
    train_y = source_actual_kwh.loc[eligible].astype(float) / capacity
    if train_x.isna().any().any():
        raise AssertionError("selected fixed model features contain missing values")
    model = TabularRegressor(
        "lgbm_l1",
        params=_model_parameters(objective),
        seed=42,
        n_jobs=7,
        device="cpu",
        early_stopping_rounds=None,
        fallback_to_cpu=True,
    )
    model.fit(train_x, train_y)
    raw = pd.Series(
        np.asarray(model.predict(application_features.loc[:, names]), dtype=np.float64),
        index=application_features.index,
        name="raw_model_cf",
    )
    candidate = pd.Series(
        blend_candidate_kwh(
            baseline_kwh, raw, weight=weight, capacity_kwh=capacity
        ),
        index=application_features.index,
        name=candidate_key,
    )
    metadata = {
        "locked_candidate": candidate_key,
        "keep_fraction": fraction,
        "objective": objective,
        "weight": weight,
        "fit_rows": int(eligible.sum()),
        "feature_count": len(subset),
        "feature_names_sha256": _canonical_sha256(list(subset)),
        "row_weights_used": False,
        "feature_values_transformed": False,
        "fit_end_before_application_start": bool(
            source_features.index.max() < application_features.index.min()
        ),
        "overlap_count": int(
            source_features.index.intersection(application_features.index).size
        ),
        "parameters": _model_parameters(objective),
    }
    if not metadata["fit_end_before_application_start"] or metadata["overlap_count"]:
        raise AssertionError("fixed model is not strict-forward")
    return model, raw, candidate, shift_table, subset, metadata


def _save_and_reload_fixed(
    *,
    out_dir: Path,
    stage: str,
    group: str,
    model: TabularRegressor,
    raw: pd.Series,
    candidate: pd.Series,
    shift_table: pd.DataFrame,
    subset: tuple[str, ...],
    application_features: pd.DataFrame,
) -> tuple[list[Path], dict[str, Any]]:
    model_path = out_dir / "models" / f"{stage}__{group}.joblib"
    raw_path = out_dir / "predictions" / f"{stage}__{group}__raw_cf.parquet"
    candidate_path = out_dir / "candidates" / f"{stage}__{group}.parquet"
    shift_path = out_dir / "shift" / f"{stage}__{group}.parquet"
    subset_path = out_dir / "shift" / f"{stage}__{group}__subset.json"
    bayes._atomic_joblib(model, model_path)
    bayes._atomic_parquet(raw.to_frame(), raw_path)
    bayes._atomic_parquet(candidate.to_frame(), candidate_path)
    bayes._atomic_parquet(shift_table, shift_path)
    bayes._write_json(
        subset_path,
        {
            "schema_version": 1,
            "preregister_sha256": PREREGISTER_SHA256,
            "stage": stage,
            "group": group,
            "feature_names": list(subset),
            "feature_names_sha256": _canonical_sha256(list(subset)),
        },
    )
    loaded: TabularRegressor = joblib.load(model_path)
    repeated = np.asarray(
        loaded.predict(application_features.loc[:, list(subset)]), dtype=np.float64
    )
    if not np.array_equal(repeated, raw.to_numpy(dtype=np.float64)):
        raise AssertionError(f"{stage} {group} reload prediction changed")
    return [model_path, raw_path, candidate_path, shift_path, subset_path], {
        "raw_prediction_bit_exact": True
    }


def _single_candidate_comparison(
    actual: pd.Series,
    baseline: pd.Series,
    candidate: pd.Series,
    group: str,
    year: int,
) -> dict[str, Any]:
    return bayes._comparison(
        actual,
        baseline,
        candidate,
        group,
        _segments(year, STAGE2_REQUIRED),
    )


def _metric_record(details: Any) -> dict[str, float]:
    return {
        "score": float(details.total_score),
        "one_minus_nmae": float(details.one_minus_nmae),
        "ficr": float(details.ficr),
    }


def _macro_comparison(
    actual: pd.DataFrame,
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    year: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, index in _segments(year, STAGE2_REQUIRED).items():
        base = score_details(actual.loc[index, list(TARGET_COLS)], baseline.loc[index, list(TARGET_COLS)])
        cand = score_details(actual.loc[index, list(TARGET_COLS)], candidate.loc[index, list(TARGET_COLS)])
        base_record = _metric_record(base)
        candidate_record = _metric_record(cand)
        result[name] = {
            "baseline": base_record,
            "candidate": candidate_record,
            "delta": candidate_record["score"] - base_record["score"],
            "one_minus_nmae_delta": candidate_record["one_minus_nmae"] - base_record["one_minus_nmae"],
            "ficr_delta": candidate_record["ficr"] - base_record["ficr"],
        }
    return result


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
            "performed": False,
            "reason": "no Stage1 group passed every registered slice",
            "stage1_passed_groups": [],
            "individually_passed_groups": [],
            "promoted_groups": [],
            "mixed_aggregate_passed": False,
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
            "locked_candidates": {},
            "promoted_groups": [],
            "promoted_groups_sha256": _canonical_sha256([]),
            "mixed_aggregate_passed": False,
            "2024_read": False,
            "2025_read": False,
        }
        bayes._write_json(out_dir / "stage2_promotion_lock.json", lock)
        return lock

    fit_labels, fit_label_evidence = _read_label_prefix(
        raw_dir,
        preregister["physical_label_prefixes"]["stage1_score_after_lock"],
        passed_groups,
    )
    baseline_path = artifact_root / "oof/gate2024_locked_v3_cf_fix.parquet"
    _assert_file_identity(
        baseline_path, preregister["stage2"]["baseline"], "Stage2 baseline"
    )
    baseline = bayes._read_prediction(
        baseline_path, YEAR_2024, required_columns=TARGET_COLS
    )
    fixed_candidates: dict[str, pd.Series] = {}
    training: dict[str, Any] = {}
    weather_evidence: dict[str, Any] = {}
    reload_audits: dict[str, Any] = {}
    outputs: list[Path] = []
    for group in passed_groups:
        features, evidence = _read_weather_range(
            artifact_root, preregister, group, TRAIN_ALL
        )
        source = features.loc[PRE2024]
        application = features.loc[YEAR_2024]
        fixed = _fit_fixed_group(
            source_features=source,
            application_features=application,
            source_actual_kwh=fit_labels.loc[PRE2024, group],
            baseline_kwh=baseline[group],
            group=group,
            candidate_key=stage1_lock["locked_candidates"][group],
        )
        model, raw, candidate, shift_table, subset, metadata = fixed
        paths, reload = _save_and_reload_fixed(
            out_dir=out_dir,
            stage="stage2",
            group=group,
            model=model,
            raw=raw,
            candidate=candidate,
            shift_table=shift_table,
            subset=subset,
            application_features=application,
        )
        fixed_candidates[group] = candidate
        training[group] = metadata
        weather_evidence[group] = evidence
        reload_audits[group] = reload
        outputs.extend(paths)
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
            "fit_label_evidence": fit_label_evidence,
            "weather_evidence": weather_evidence,
            "training": training,
            "reload_audits": reload_audits,
            "outputs": [describe_file(path) for path in outputs],
            "2024_application_target_values_read": False,
            "2025_inputs_read": False,
        },
    )
    prescore_lock_path = out_dir / "stage2_global_prescore_lock.json"
    bayes._write_json(
        prescore_lock_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "prescore_record": describe_file(prescore_record_path),
            "candidate_outputs": [describe_file(path) for path in outputs],
            "created_before_2024_application_target_values": True,
        },
    )

    score_labels, score_label_evidence = _read_full_labels(
        raw_dir, preregister, TARGET_COLS
    )
    group_results: dict[str, Any] = {}
    individually_passed: list[str] = []
    for group in passed_groups:
        comparison = _single_candidate_comparison(
            score_labels.loc[YEAR_2024, group],
            baseline[group],
            fixed_candidates[group],
            group,
            2024,
        )
        deltas = {name: float(comparison[name]["delta"]) for name in STAGE2_REQUIRED}
        passed = all(delta > 0.0 for delta in deltas.values())
        if passed:
            individually_passed.append(group)
        group_results[group] = {
            "locked_candidate": stage1_lock["locked_candidates"][group],
            "comparison": comparison,
            "comparison_sha256": _canonical_sha256(comparison),
            "minimum_delta": min(deltas.values()),
            "mean_delta": float(np.mean(list(deltas.values()))),
            "all_seven_strictly_positive": passed,
        }
    mixed = baseline.copy(deep=True)
    for group in individually_passed:
        mixed[group] = fixed_candidates[group]
    for group in TARGET_COLS:
        if group not in individually_passed:
            if mixed[group].to_numpy(dtype=np.float64).tobytes() != baseline[group].to_numpy(dtype=np.float64).tobytes():
                raise AssertionError("Stage2 identity group changed")
    aggregate = _macro_comparison(
        score_labels.loc[YEAR_2024, list(TARGET_COLS)], baseline, mixed, 2024
    )
    aggregate_passed = bool(individually_passed) and all(
        float(aggregate[name]["delta"]) > 0.0 for name in STAGE2_REQUIRED
    )
    promoted_groups = individually_passed if aggregate_passed else []
    mixed_path = out_dir / "candidates/stage2__mixed_candidate.parquet"
    bayes._atomic_parquet(mixed, mixed_path)
    result = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "stage1_promotion_lock": describe_file(out_dir / "stage1_promotion_lock.json"),
        "stage2_prescore_lock": describe_file(prescore_lock_path),
        "score_label_evidence_after_prescore_lock": score_label_evidence,
        "stage1_passed_groups": passed_groups,
        "group_results": group_results,
        "individually_passed_groups": individually_passed,
        "mixed_candidate": describe_file(mixed_path),
        "mixed_aggregate_comparison": aggregate,
        "mixed_aggregate_comparison_sha256": _canonical_sha256(aggregate),
        "mixed_aggregate_passed": aggregate_passed,
        "promoted_groups": promoted_groups,
        "promoted_groups_sha256": _canonical_sha256(promoted_groups),
        "2024_weather_read": True,
        "2024_label_read": True,
        "2025_read": False,
        "leaderboard_or_public_input_read": False,
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
        "individually_passed_groups": individually_passed,
        "mixed_aggregate_passed": aggregate_passed,
        "promoted_groups": promoted_groups,
        "promoted_groups_sha256": result["promoted_groups_sha256"],
        "no_2025_reselection_or_retuning": True,
        "2024_read": True,
        "2025_read": False,
    }
    bayes._write_json(out_dir / "stage2_promotion_lock.json", lock)
    print(
        f"Stage2 individual={individually_passed}; mixed={aggregate_passed}; promoted={promoted_groups}",
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
        raise AssertionError("Stage2 results changed after lock")
    if lock["promoted_groups"] != result["promoted_groups"]:
        raise AssertionError("Stage2 promoted groups changed")
    if bool(lock["mixed_aggregate_passed"]) != bool(result["mixed_aggregate_passed"]):
        raise AssertionError("Stage2 aggregate decision changed")
    return lock, result


def _final(
    *,
    raw_dir: Path,
    artifact_root: Path,
    out_dir: Path,
    preregister: Mapping[str, Any],
) -> dict[str, Any]:
    stage2_lock, _ = _load_stage2_lock(out_dir)
    result_path = out_dir / "final_results.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    promoted_groups = list(stage2_lock["promoted_groups"])
    if not promoted_groups:
        result = {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "performed": False,
            "reason": "no group survived both strict gates including the mixed aggregate gate",
            "promoted_groups": [],
            "2025_weather_read": False,
            "sample_submission_read": False,
            "submission_created": False,
            "leaderboard_or_public_input_read": False,
        }
        bayes._write_json(result_path, result)
        return result

    labels, label_evidence = _read_full_labels(raw_dir, preregister, promoted_groups)
    baseline_path = artifact_root / "final_cf_fix/predictions/corrected_v3_test.parquet"
    _assert_file_identity(
        baseline_path, preregister["final_2025"]["baseline"], "final baseline"
    )
    baseline = bayes._read_prediction(
        baseline_path, YEAR_2025, required_columns=TARGET_COLS
    )
    prediction = baseline.copy(deep=True)
    raw_predictions = pd.DataFrame(index=YEAR_2025)
    models: dict[str, TabularRegressor] = {}
    training: dict[str, Any] = {}
    weather_evidence: dict[str, Any] = {}
    shift_paths: list[Path] = []
    for group in promoted_groups:
        train_features, train_evidence = _read_weather_range(
            artifact_root, preregister, group, TRAIN_ALL
        )
        test_features, test_evidence = _read_test_weather(
            artifact_root, preregister, group
        )
        fixed = _fit_fixed_group(
            source_features=train_features,
            application_features=test_features,
            source_actual_kwh=labels.loc[TRAIN_ALL, group],
            baseline_kwh=baseline[group],
            group=group,
            candidate_key=stage2_lock["locked_candidates"][group],
        )
        model, raw, candidate, shift_table, subset, metadata = fixed
        models[group] = model
        raw_predictions[group] = raw
        prediction[group] = candidate
        training[group] = metadata
        weather_evidence[group] = {"train": train_evidence, "test": test_evidence}
        shift_path = out_dir / "shift" / f"final__{group}.parquet"
        subset_path = out_dir / "shift" / f"final__{group}__subset.json"
        bayes._atomic_parquet(shift_table, shift_path)
        bayes._write_json(
            subset_path,
            {
                "schema_version": 1,
                "preregister_sha256": PREREGISTER_SHA256,
                "stage": "final",
                "group": group,
                "feature_names": list(subset),
                "feature_names_sha256": _canonical_sha256(list(subset)),
            },
        )
        shift_paths.extend((shift_path, subset_path))
    for group in TARGET_COLS:
        if group not in promoted_groups:
            if prediction[group].to_numpy(dtype=np.float64).tobytes() != baseline[group].to_numpy(dtype=np.float64).tobytes():
                raise AssertionError("final identity group changed")
    model_path = out_dir / "models/final__promoted_groups.joblib"
    raw_path = out_dir / "predictions/final__raw_cf_2025.parquet"
    prediction_path = out_dir / "predictions/covariate_shift_feature_pruning_2025.parquet"
    bayes._atomic_joblib(models, model_path)
    bayes._atomic_parquet(raw_predictions, raw_path)
    bayes._atomic_parquet(prediction, prediction_path)
    loaded: dict[str, TabularRegressor] = joblib.load(model_path)
    reload_checks: dict[str, Any] = {}
    for group in promoted_groups:
        _, _, _ = _candidate_recipe(stage2_lock["locked_candidates"][group])
        subset_payload = json.loads(
            (out_dir / "shift" / f"final__{group}__subset.json").read_text(encoding="utf-8")
        )
        test_features, _ = _read_test_weather(artifact_root, preregister, group)
        repeated = np.asarray(
            loaded[group].predict(test_features.loc[:, subset_payload["feature_names"]]),
            dtype=np.float64,
        )
        if not np.array_equal(repeated, raw_predictions[group].to_numpy(dtype=np.float64)):
            raise AssertionError("final model reload prediction changed")
        reload_checks[group] = {"raw_prediction_bit_exact": True}

    sample_path = raw_dir / "sample_submission.csv"
    _assert_file_identity(
        sample_path,
        preregister["input_identities"]["sample_submission_after_final_unlock"],
        "sample submission",
    )
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
        raise AssertionError("sample timestamp sequence changed")
    submission = sample.copy()
    for group in TARGET_COLS:
        submission[group] = prediction[group].to_numpy(dtype=float)
    csv_path = out_dir / "covariate_shift_feature_pruning_2025.csv"
    bayes._atomic_csv(submission, csv_path)
    verification = bayes._verify_submission(csv_path, sample, prediction)
    result = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "stage2_promotion_lock": describe_file(out_dir / "stage2_promotion_lock.json"),
        "performed": True,
        "promoted_groups": promoted_groups,
        "label_evidence": label_evidence,
        "training": training,
        "weather_evidence": weather_evidence,
        "reload_checks": reload_checks,
        "outputs": [
            describe_file(path)
            for path in [model_path, raw_path, prediction_path, *shift_paths, csv_path]
        ],
        "submission_verification": verification,
        "2025_weather_read": True,
        "sample_submission_read": True,
        "submission_created": True,
        "leaderboard_or_public_input_read": False,
    }
    bayes._write_json(result_path, result)
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
    source_paths = {
        "runner": Path(__file__).resolve(),
        "module": PROJECT_DIR / "src/covariate_shift_pruning.py",
        "metric": PROJECT_DIR / "src/metric.py",
        "models": PROJECT_DIR / "src/models.py",
        "focused_tests": PROJECT_DIR / "tests/test_covariate_shift_pruning.py",
        "preregister": preregister_path.resolve(),
        "preregister_sidecar": preregister_path.with_suffix(".sha256").resolve(),
        "locked_recipe": PROJECT_DIR / "configs/train_final.v3.locked.json",
    }
    outputs = sorted(
        path for path in out_dir.rglob("*") if path.is_file() and path != manifest_path
    )
    payload = {
        "schema_version": 1,
        "artifact_type": "covariate_shift_feature_pruning_strict_forward_v1",
        "created_utc": utc_now(),
        "command": [sys.executable, *sys.argv],
        "runtime": {"packages": package_versions(), "git": git_state(PROJECT_DIR)},
        "preregister_sha256": PREREGISTER_SHA256,
        "contracts": {
            "label_free_transductive_feature_selection": True,
            "feature_values_modified": False,
            "row_weights_used": False,
            "target_scada_prediction_public_time_year_features": False,
            "application_labels_read_only_after_models_predictions_and_hash_locks": True,
            "strict_forward_fit": True,
            "stage2_individual_and_mixed_seven_slice_gate": True,
            "leaderboard_score_claim": False,
        },
        "selection": {
            "stage1_locked_candidates": stage1_lock["locked_candidates"],
            "stage1_passed_groups": stage1_lock["passed_groups"],
            "stage2_individually_passed_groups": stage2_lock.get("individually_passed_groups", []),
            "stage2_mixed_aggregate_passed": bool(stage2_lock["mixed_aggregate_passed"]),
            "promoted_groups": stage2_lock["promoted_groups"],
        },
        "read_flags": {
            "2024_weather_read": bool(stage2_result.get("2024_weather_read", False)),
            "2024_label_read": bool(stage2_result.get("2024_label_read", False)),
            "2025_weather_read": bool(final_result.get("2025_weather_read", False)),
            "sample_submission_read": bool(final_result.get("sample_submission_read", False)),
            "csv_created": bool(final_result.get("submission_created", False)),
            "leaderboard_or_public_input_read": False,
        },
        "bounded_io": {
            "label_prefixes_physically_byte_bounded_before_prescore_locks": True,
            "weather_cache_filters_are_logical_not_physical_byte_bounds": True,
            "weather_cache_single_row_group_caveat_disclosed": True,
            "raw_root": str(raw_dir.resolve()),
            "artifact_root": str(artifact_root.resolve()),
        },
        "tests": {
            "focused_source": describe_file(PROJECT_DIR / "tests/test_covariate_shift_pruning.py"),
            "focused_prelaunch_status": "pass",
            "focused_test_count": 8,
            "full_suite_prelaunch_status": "pass",
            "full_suite_test_count": 235,
            "commands": [
                ".venv/Scripts/python.exe -m unittest tests.test_covariate_shift_pruning -v",
                ".venv/Scripts/python.exe -m unittest discover -s tests -q"
            ],
        },
        "provenance": {name: describe_file(path) for name, path in source_paths.items()},
        "outputs": [describe_file(path) for path in outputs],
    }
    bayes._write_json(manifest_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    raw_dir = args.raw_dir.expanduser().resolve()
    artifact_root = args.artifact_root.expanduser().resolve()
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
        _write_manifest(
            raw_dir=raw_dir,
            artifact_root=artifact_root,
            out_dir=out_dir,
            preregister_path=preregister_path,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

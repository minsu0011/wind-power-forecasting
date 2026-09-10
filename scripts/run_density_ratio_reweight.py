"""Run the preregistered strict-forward density-ratio correction."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.run_shared_q07_multiseed import (  # noqa: E402
    _BoundedRawReader,
    _csv_prefix_identity,
    _next_csv_first_field,
)
from src.density_ratio import (  # noqa: E402
    assemble_locked_group,
    estimate_density_ratio_weights,
    half_delta_candidate,
)
from src.metric import group_metrics  # noqa: E402
from src.models import TabularRegressor  # noqa: E402


PREREG_SHA256 = "cec7ed311ad3b9358a7c5c97b66f788938de5fe17a77d27417e0a5b3f96c794e"
FEATURE_SHA256 = "990148af97c89aad5b21c12388ffbfeb214819424e14c8950ed0f37c94e60feb"
GROUPS = ("kpx_group_1", "kpx_group_2")
ALL_GROUPS = (*GROUPS, "kpx_group_3")
CAPACITY = {"kpx_group_1": 21600.0, "kpx_group_2": 21600.0, "kpx_group_3": 21000.0}
COMPONENTS = (
    "lgb_l1",
    "lgb_q07",
    "shared_l1",
    "shared_q07",
    "top200_q07",
    "energy_q06",
)
FORMULA = "clip(0.50*corrected_v3_baseline_kwh + 0.50*weighted_recipe_kwh, 0, 1.02*capacity_kwh)"

STAGE1_COMPONENTS = {
    "lgb_l1": "artifacts/oof/dev2023_lgb_l1_eligible_n1500.parquet",
    "lgb_q07": "artifacts/oof/dev2023_lgb_q07_eligible.parquet",
    "shared_l1": "artifacts/oof/dev2023_shared_l1_eligible.parquet",
    "shared_q07": "artifacts/oof/dev2023_shared_q07_eligible.parquet",
    "top200_q07": "artifacts/oof/dev2023_lgb_top200_q07_eligible.parquet",
    "energy_q06": "artifacts/oof/dev2023_lgb_q06_energywt_eligible.parquet",
}
STAGE2_COMPONENTS = {
    "lgb_l1": "artifacts/gate/v3/predictions/lgb_l1_gate.parquet",
    "lgb_q07": "artifacts/gate/v3/predictions/lgb_q07_gate.parquet",
    "shared_l1": "artifacts/oof/gate2024_shared_l1_cf.parquet",
    "shared_q07": "artifacts/oof/gate2024_shared_q07_cf.parquet",
    "top200_q07": "artifacts/gate/v3/predictions/top200_q07_gate.parquet",
    "energy_q06": "artifacts/gate/v3/predictions/energy_q06_gate.parquet",
}
FINAL_COMPONENTS = {
    "lgb_l1": "artifacts/final_v3/predictions/v3_locked_full_2025__lgb_l1_test.parquet",
    "lgb_q07": "artifacts/final_v3/predictions/v3_locked_full_2025__lgb_q07_test.parquet",
    "shared_l1": "artifacts/final_cf_fix/predictions/shared_l1_cf_seed42_test.parquet",
    "shared_q07": "artifacts/final_cf_fix/predictions/shared_q07_cf_seed42_test.parquet",
    "top200_q07": "artifacts/final_v3/predictions/v3_locked_full_2025__top200_q07_test.parquet",
    "energy_q06": "artifacts/final_v3/predictions/v3_locked_full_2025__energy_q06_test.parquet",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "sha256": sha256_file(path),
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, engine="pyarrow", compression="zstd", index=True)
    temporary.replace(path)


def read_bounded_labels(
    path: Path,
    *,
    rows: int,
    byte_limit: int,
    expected_sha256: str,
    expected_end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    observed_sha, observed_bytes = _csv_prefix_identity(
        path, data_rows=rows, byte_limit=byte_limit
    )
    if observed_sha != expected_sha256 or observed_bytes != byte_limit:
        raise AssertionError("bounded label prefix identity changed")
    bounded = _BoundedRawReader(path, byte_limit=byte_limit)
    try:
        import io

        with io.BufferedReader(bounded, buffer_size=1024 * 1024) as stream:
            frame = pd.read_csv(stream, encoding="utf-8-sig", memory_map=False)
            bytes_returned = bounded.bytes_returned
            position = bounded.underlying_position
    finally:
        bounded.close()
    if bytes_returned != byte_limit or position != byte_limit:
        raise AssertionError("label parser crossed its physical byte boundary")
    if len(frame) != rows or tuple(frame.columns) != ("kst_dtm", *ALL_GROUPS):
        raise AssertionError("bounded label schema/row count changed")
    times = pd.to_datetime(frame.pop("kst_dtm"), errors="raise")
    frame.index = pd.DatetimeIndex(times, name="forecast_kst_dtm")
    frame = frame.astype(float)
    if frame.index.min() != pd.Timestamp("2022-01-01 01:00:00"):
        raise AssertionError("label start changed")
    if frame.index.max() != expected_end:
        raise AssertionError("label end changed")
    next_timestamp = _next_csv_first_field(
        path, after_data_rows=rows, prefix_bytes=byte_limit
    )
    evidence = {
        "path": str(path.resolve()),
        "physical_byte_limit": byte_limit,
        "physical_prefix_sha256": observed_sha,
        "data_rows": rows,
        "bytes_returned_to_parser": bytes_returned,
        "underlying_file_position_after_read": position,
        "last_included_timestamp": frame.index.max().isoformat(),
        "next_row_first_field_only": next_timestamp,
        "future_value_cells_materialized": 0,
    }
    return frame, evidence


def read_full_labels(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if len(frame) != 26304 or tuple(frame.columns) != ("kst_dtm", *ALL_GROUPS):
        raise AssertionError("full label schema/rows changed")
    times = pd.to_datetime(frame.pop("kst_dtm"), errors="raise")
    frame.index = pd.DatetimeIndex(times, name="forecast_kst_dtm")
    frame = frame.astype(float)
    if frame.index.max() != pd.Timestamp("2025-01-01 00:00:00"):
        raise AssertionError("full label end changed")
    return frame, {"physical_scope": "whole_file_after_stage2_prescore_lock", **snapshot(path)}


def read_weather(cache_dir: Path, group: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, Path]:
    path = cache_dir / f"{group}_weather_train.parquet"
    frame = pd.read_parquet(
        path,
        engine="pyarrow",
        filters=[("forecast_kst_dtm", ">=", start), ("forecast_kst_dtm", "<=", end)],
    )
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    expected = pd.date_range(start, end, freq="h", name="forecast_kst_dtm")
    if not frame.index.equals(expected):
        raise AssertionError(f"{group} weather time range changed")
    values = frame.to_numpy(dtype=np.float32, copy=False)
    if not np.isfinite(values).all():
        raise AssertionError(f"{group} weather contains non-finite values")
    return frame.astype(np.float32, copy=False), path


def read_test_weather(cache_dir: Path, group: str) -> tuple[pd.DataFrame, Path]:
    path = cache_dir / f"{group}_weather_test.parquet"
    frame = pd.read_parquet(path, engine="pyarrow")
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    expected = pd.date_range("2025-01-01 01:00", "2026-01-01 00:00", freq="h", name="forecast_kst_dtm")
    if not frame.index.equals(expected):
        raise AssertionError(f"{group} test weather time range changed")
    if not np.isfinite(frame.to_numpy(dtype=np.float32, copy=False)).all():
        raise AssertionError(f"{group} test weather contains non-finite values")
    return frame.astype(np.float32, copy=False), path


def model_params(recipe: Mapping[str, Any], name: str) -> dict[str, Any]:
    spec = recipe["models"][name]
    params = dict(spec["params"])
    if name == "lgb_l1":
        params["objective"] = "regression_l1"
    elif name == "lgb_q07":
        params["objective"] = "quantile"
        params["alpha"] = 0.7
    else:
        raise ValueError(name)
    return params


def fit_weighted_components(
    source_weather: pd.DataFrame,
    application_weather: pd.DataFrame,
    source_actual: pd.Series,
    source_weights: pd.Series,
    *,
    group: str,
    recipe: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, TabularRegressor], dict[str, Any]]:
    capacity = CAPACITY[group]
    eligible = source_actual.notna() & np.isfinite(source_actual.to_numpy()) & (source_actual >= 0.10 * capacity)
    train_x = source_weather.loc[eligible]
    train_y = source_actual.loc[eligible].astype(float) / capacity
    train_weight = source_weights.loc[eligible].astype(float)
    predictions: dict[str, np.ndarray] = {}
    models: dict[str, TabularRegressor] = {}
    records: dict[str, Any] = {}
    for name in ("lgb_l1", "lgb_q07"):
        model = TabularRegressor(
            "lgbm_l1",
            params=model_params(recipe, name),
            seed=42,
            n_jobs=7,
            device="cpu",
            early_stopping_rounds=None,
            fallback_to_cpu=True,
        )
        model.fit(train_x, train_y, sample_weight=train_weight)
        prediction = model.predict(application_weather) * capacity
        if not np.isfinite(prediction).all():
            raise AssertionError("weighted component prediction is non-finite")
        predictions[name] = np.asarray(prediction, dtype=np.float64)
        models[name] = model
        records[name] = {
            "fit_rows": int(len(train_y)),
            "feature_count": int(train_x.shape[1]),
            "target_unit": "capacity_factor",
            "parameters": model_params(recipe, name),
        }
    return predictions, models, records


def load_components(project_dir: Path, paths: Mapping[str, str], index: pd.DatetimeIndex, group: str) -> tuple[dict[str, pd.Series], dict[str, Any]]:
    components: dict[str, pd.Series] = {}
    inputs: dict[str, Any] = {}
    for name in COMPONENTS:
        path = project_dir / paths[name]
        frame = pd.read_parquet(path, engine="pyarrow")
        frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
        if not frame.index.equals(index) or group not in frame.columns:
            raise AssertionError(f"component alignment changed: {path}")
        components[name] = frame[group].astype(float)
        inputs[name] = snapshot(path)
    return components, inputs


def compare_periods(actual: pd.Series, baseline: pd.Series, candidate: pd.Series, *, group: str) -> tuple[dict[str, Any], bool]:
    index = actual.index
    year = int(index.min().year)
    boundaries = [
        pd.Timestamp(year=year, month=1, day=1, hour=1),
        pd.Timestamp(year=year, month=4, day=1, hour=0),
        pd.Timestamp(year=year, month=7, day=1, hour=0),
        pd.Timestamp(year=year, month=10, day=1, hour=0),
        pd.Timestamp(year=year + 1, month=1, day=1, hour=0),
    ]
    masks = {
        "full": np.ones(len(index), dtype=bool),
        "H1": (index >= boundaries[0]) & (index <= boundaries[2]),
        "H2": (index > boundaries[2]) & (index <= boundaries[4]),
        "Q1": (index >= boundaries[0]) & (index <= boundaries[1]),
        "Q2": (index > boundaries[1]) & (index <= boundaries[2]),
        "Q3": (index > boundaries[2]) & (index <= boundaries[3]),
        "Q4": (index > boundaries[3]) & (index <= boundaries[4]),
    }
    records: dict[str, Any] = {}
    for segment, mask in masks.items():
        base_metric = group_metrics(actual[mask], baseline[mask], CAPACITY[group], group_name=group)
        cand_metric = group_metrics(actual[mask], candidate[mask], CAPACITY[group], group_name=group)
        base = base_metric.as_dict()
        cand = cand_metric.as_dict()
        records[segment] = {
            "baseline": base,
            "candidate": cand,
            "delta": {
                "total": float(0.5 * ((cand_metric.one_minus_nmae - base_metric.one_minus_nmae) + (cand_metric.ficr - base_metric.ficr))),
                "one_minus_nmae": float(cand_metric.one_minus_nmae - base_metric.one_minus_nmae),
                "ficr": float(cand_metric.ficr - base_metric.ficr),
            },
            "row_count": int(mask.sum()),
        }
    full = records["full"]["delta"]
    passed = (
        all(records[name]["delta"]["total"] > 0.0 for name in ("full", "H1", "H2"))
        and full["one_minus_nmae"] >= 0.0
        and full["ficr"] >= 0.0
        and (full["one_minus_nmae"] > 0.0 or full["ficr"] > 0.0)
        and all(records[name]["delta"]["total"] >= -0.001 for name in ("Q1", "Q2", "Q3", "Q4"))
    )
    return records, bool(passed)


def build_group_candidate(
    project_dir: Path,
    component_paths: Mapping[str, str],
    baseline_path: Path,
    weighted_predictions: Mapping[str, np.ndarray],
    application_index: pd.DatetimeIndex,
    *,
    group: str,
    recipe: Mapping[str, Any],
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    components, component_inputs = load_components(project_dir, component_paths, application_index, group)
    reconstructed = assemble_locked_group(
        components, group=group, capacity_kwh=CAPACITY[group], ensemble=recipe["ensemble"]
    )
    baseline_frame = pd.read_parquet(baseline_path, engine="pyarrow")
    baseline_frame.index = pd.DatetimeIndex(baseline_frame.index, name="forecast_kst_dtm")
    if not baseline_frame.index.equals(application_index):
        raise AssertionError("baseline application index changed")
    baseline = baseline_frame[group].astype(float)
    max_diff = float(np.max(np.abs(reconstructed - baseline.to_numpy(dtype=float))))
    if not np.array_equal(reconstructed, baseline.to_numpy(dtype=float)):
        raise AssertionError(f"baseline reconstruction is not bit exact: {max_diff}")
    weighted_components = dict(components)
    for name in ("lgb_l1", "lgb_q07"):
        weighted_components[name] = np.asarray(weighted_predictions[name], dtype=np.float64)
    weighted_recipe = assemble_locked_group(
        weighted_components,
        group=group,
        capacity_kwh=CAPACITY[group],
        ensemble=recipe["ensemble"],
    )
    candidate = half_delta_candidate(
        baseline.to_numpy(dtype=float), weighted_recipe, capacity_kwh=CAPACITY[group]
    )
    audit = {
        "formula": FORMULA,
        "baseline_reconstruction_bit_exact": True,
        "baseline_reconstruction_max_abs_kwh": max_diff,
        "baseline_input": snapshot(baseline_path),
        "component_inputs": component_inputs,
        "candidate_min_kwh": float(candidate.min()),
        "candidate_max_kwh": float(candidate.max()),
    }
    return (
        pd.Series(candidate, index=application_index, name=group),
        pd.Series(weighted_recipe, index=application_index, name=group),
        audit,
    )


def execute_stage(
    *,
    stage: str,
    project_dir: Path,
    cache_dir: Path,
    out_dir: Path,
    recipe: Mapping[str, Any],
    feature_names: list[str],
    source_labels: pd.DataFrame,
    source_start: pd.Timestamp,
    source_end: pd.Timestamp,
    application_start: pd.Timestamp,
    application_end: pd.Timestamp,
    groups: list[str],
    component_paths: Mapping[str, str],
    baseline_path: Path,
    label_evidence: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], Path]:
    application_index = pd.date_range(application_start, application_end, freq="h", name="forecast_kst_dtm")
    candidate_frame = pd.DataFrame(index=application_index, columns=groups, dtype=float)
    weighted_frame = pd.DataFrame(index=application_index, columns=groups, dtype=float)
    stage_records: dict[str, Any] = {}
    model_paths: dict[str, str] = {}
    output_paths: list[Path] = []
    for group in groups:
        source_weather, cache_path = read_weather(cache_dir, group, source_start, source_end)
        application_weather, _ = read_weather(cache_dir, group, application_start, application_end)
        actual = source_labels[group].reindex(source_weather.index)
        if actual.isna().all():
            raise AssertionError("source labels are unavailable")
        eligible = actual.notna() & np.isfinite(actual.to_numpy()) & (actual >= 0.10 * CAPACITY[group])
        density = estimate_density_ratio_weights(
            source_weather,
            application_weather,
            eligible_source=eligible,
            feature_names=feature_names,
        )
        weight_path = out_dir / "weights" / f"{stage}__{group}.parquet"
        weight_frame = pd.concat(
            [density.source_weight, density.source_probability], axis=1
        )
        write_parquet(weight_frame, weight_path)
        output_paths.append(weight_path)
        weighted_predictions, models, model_records = fit_weighted_components(
            source_weather,
            application_weather,
            actual,
            density.source_weight,
            group=group,
            recipe=recipe,
        )
        model_path = out_dir / "models" / f"{stage}__{group}.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(models, model_path, compress=3)
        output_paths.append(model_path)
        model_paths[group] = str(model_path)
        candidate, weighted_recipe, assembly_audit = build_group_candidate(
            project_dir,
            component_paths,
            baseline_path,
            weighted_predictions,
            application_index,
            group=group,
            recipe=recipe,
        )
        candidate_frame[group] = candidate
        weighted_frame[group] = weighted_recipe
        stage_records[group] = {
            "domain": density.audit,
            "outcome_models": model_records,
            "assembly": assembly_audit,
            "weather_cache": snapshot(cache_path),
            "weights_artifact": snapshot(weight_path),
            "models_artifact": snapshot(model_path),
        }
    candidate_path = out_dir / "oof" / f"{stage}__candidate.parquet"
    weighted_path = out_dir / "oof" / f"{stage}__weighted_recipe.parquet"
    write_parquet(candidate_frame, candidate_path)
    write_parquet(weighted_frame, weighted_path)
    output_paths.extend((candidate_path, weighted_path))
    lock_path = out_dir / f"{stage}_prescore_lock.json"
    lock = {
        "schema_version": 1,
        "stage": stage,
        "created_before_application_label_value_read": True,
        "groups": groups,
        "formula": FORMULA,
        "preregister_sha256": PREREG_SHA256,
        "feature_names_sha256": FEATURE_SHA256,
        "source_label_physical_evidence": dict(label_evidence),
        "candidate": snapshot(candidate_path),
        "weighted_recipe": snapshot(weighted_path),
        "models": {group: snapshot(Path(path)) for group, path in model_paths.items()},
        "records": stage_records,
        "source_hashes": {
            "runner": snapshot(Path(__file__)),
            "density_module": snapshot(project_dir / "src/density_ratio.py"),
            "test": snapshot(project_dir / "tests/test_density_ratio.py"),
        },
    }
    write_json(lock_path, lock)
    return candidate_frame, weighted_frame, stage_records, lock_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_DIR / "artifacts/cache")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "artifacts/postgate/density_ratio_reweight_strict_v1")
    args = parser.parse_args()
    raw_dir = args.raw_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    out_dir = args.out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"output directory must be new/empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    prereg_path = PROJECT_DIR / "configs/density_ratio_reweight_preregister_v1.json"
    feature_path = PROJECT_DIR / "configs/density_ratio_features_v1.txt"
    recipe_path = PROJECT_DIR / "configs/train_final.v3.locked.json"
    if sha256_file(prereg_path) != PREREG_SHA256 or sha256_file(feature_path) != FEATURE_SHA256:
        raise AssertionError("preregister or feature-list hash changed")
    shutil.copyfile(prereg_path, out_dir / "preregister.json")
    shutil.copyfile(feature_path, out_dir / "domain_features.txt")
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    feature_names = feature_path.read_text(encoding="utf-8").splitlines()
    if feature_names != prereg["domain_features"]["ordered_names"]:
        raise AssertionError("ordered feature names differ from preregistration")

    label_path = raw_dir / "train/train_labels.csv"
    fit_prefix = prereg["data_contract"]["label_stage1_fit_prefix"]
    labels_2022, fit_evidence = read_bounded_labels(
        label_path,
        rows=int(fit_prefix["data_rows"]),
        byte_limit=int(fit_prefix["bytes"]),
        expected_sha256=str(fit_prefix["sha256"]),
        expected_end=pd.Timestamp(fit_prefix["end"]),
    )
    print("Stage1: fitting density-weighted components without reading 2023 label values", flush=True)
    stage1_candidate, _, stage1_records, stage1_prescore = execute_stage(
        stage="stage1_2023",
        project_dir=PROJECT_DIR,
        cache_dir=cache_dir,
        out_dir=out_dir,
        recipe=recipe,
        feature_names=feature_names,
        source_labels=labels_2022,
        source_start=pd.Timestamp("2022-01-01 01:00:00"),
        source_end=pd.Timestamp("2023-01-01 00:00:00"),
        application_start=pd.Timestamp("2023-01-01 01:00:00"),
        application_end=pd.Timestamp("2024-01-01 00:00:00"),
        groups=list(GROUPS),
        component_paths=STAGE1_COMPONENTS,
        baseline_path=PROJECT_DIR / "artifacts/oof/dev2023_locked_v3.parquet",
        label_evidence=fit_evidence,
    )
    if not stage1_prescore.exists():
        raise AssertionError("Stage1 pre-score lock missing")

    validation_prefix = prereg["data_contract"]["label_stage1_validation_prefix"]
    labels_pre2024, validation_evidence = read_bounded_labels(
        label_path,
        rows=int(validation_prefix["data_rows"]),
        byte_limit=int(validation_prefix["bytes"]),
        expected_sha256=str(validation_prefix["sha256"]),
        expected_end=pd.Timestamp(validation_prefix["end"]),
    )
    index_2023 = stage1_candidate.index
    baseline_2023 = pd.read_parquet(PROJECT_DIR / "artifacts/oof/dev2023_locked_v3.parquet", engine="pyarrow")
    baseline_2023.index = pd.DatetimeIndex(baseline_2023.index, name="forecast_kst_dtm")
    stage1_results: dict[str, Any] = {}
    locked_groups: list[str] = []
    for group in GROUPS:
        periods, passed = compare_periods(
            labels_pre2024.loc[index_2023, group],
            baseline_2023.loc[index_2023, group],
            stage1_candidate[group],
            group=group,
        )
        stage1_results[group] = {"passed": passed, "segments": periods, "diagnostics": stage1_records[group]}
        if passed:
            locked_groups.append(group)
    stage1_payload = {
        "schema_version": 1,
        "stage": "stage1_2023",
        "prescore_lock": snapshot(stage1_prescore),
        "validation_label_physical_evidence": validation_evidence,
        "locked_groups": locked_groups,
        "forced_identity_groups": [group for group in ALL_GROUPS if group not in locked_groups],
        "results": stage1_results,
    }
    stage1_results_path = out_dir / "stage1_results.json"
    write_json(stage1_results_path, stage1_payload)
    stage1_lock_path = out_dir / "stage1_promotion_lock.json"
    write_json(
        stage1_lock_path,
        {
            "schema_version": 1,
            "locked_groups": locked_groups,
            "stage1_results": snapshot(stage1_results_path),
            "2024_label_values_read": False,
            "no_2024_reselection": True,
        },
    )
    print(f"Stage1 locked groups: {locked_groups}", flush=True)

    stage2_payload: dict[str, Any] = {
        "stage": "stage2_2024",
        "executed": False,
        "reason": "No Stage1 group passed; 2024 label values were not read.",
        "promoted_groups": [],
    }
    promoted_groups: list[str] = []
    full_labels: pd.DataFrame | None = None
    if locked_groups:
        stage2_candidate, _, stage2_records, stage2_prescore = execute_stage(
            stage="stage2_2024",
            project_dir=PROJECT_DIR,
            cache_dir=cache_dir,
            out_dir=out_dir,
            recipe=recipe,
            feature_names=feature_names,
            source_labels=labels_pre2024,
            source_start=pd.Timestamp("2022-01-01 01:00:00"),
            source_end=pd.Timestamp("2024-01-01 00:00:00"),
            application_start=pd.Timestamp("2024-01-01 01:00:00"),
            application_end=pd.Timestamp("2025-01-01 00:00:00"),
            groups=locked_groups,
            component_paths=STAGE2_COMPONENTS,
            baseline_path=PROJECT_DIR / "artifacts/oof/gate2024_locked_v3_cf_fix.parquet",
            label_evidence=validation_evidence,
        )
        if not stage2_prescore.exists():
            raise AssertionError("Stage2 pre-score lock missing")
        full_labels, full_label_evidence = read_full_labels(label_path)
        baseline_2024 = pd.read_parquet(PROJECT_DIR / "artifacts/oof/gate2024_locked_v3_cf_fix.parquet", engine="pyarrow")
        baseline_2024.index = pd.DatetimeIndex(baseline_2024.index, name="forecast_kst_dtm")
        group_results: dict[str, Any] = {}
        for group in locked_groups:
            periods, passed = compare_periods(
                full_labels.loc[stage2_candidate.index, group],
                baseline_2024.loc[stage2_candidate.index, group],
                stage2_candidate[group],
                group=group,
            )
            group_results[group] = {"passed": passed, "segments": periods, "diagnostics": stage2_records[group]}
            if passed:
                promoted_groups.append(group)
        stage2_payload = {
            "schema_version": 1,
            "stage": "stage2_2024",
            "executed": True,
            "prescore_lock": snapshot(stage2_prescore),
            "full_label_evidence": full_label_evidence,
            "promoted_groups": promoted_groups,
            "results": group_results,
            "no_reselection": True,
        }
    stage2_path = out_dir / "stage2_results.json"
    write_json(stage2_path, stage2_payload)
    write_json(
        out_dir / "stage2_promotion_lock.json",
        {
            "schema_version": 1,
            "promoted_groups": promoted_groups,
            "stage2_results": snapshot(stage2_path),
            "2025_label_values_read": False,
            "no_reselection": True,
        },
    )

    final_payload: dict[str, Any] = {
        "executed": False,
        "promoted_groups": promoted_groups,
        "submission_path": None,
        "reason": "No group passed both strict gates; 2025 weather/sample/baseline were not read and no CSV was created.",
    }
    if promoted_groups:
        if full_labels is None:
            full_labels, _ = read_full_labels(label_path)
        final_index = pd.date_range("2025-01-01 01:00", "2026-01-01 00:00", freq="h", name="forecast_kst_dtm")
        final_candidates: dict[str, pd.Series] = {}
        final_records: dict[str, Any] = {}
        for group in promoted_groups:
            source_weather, cache_path = read_weather(
                cache_dir, group, pd.Timestamp("2022-01-01 01:00"), pd.Timestamp("2025-01-01 00:00")
            )
            test_weather, test_cache_path = read_test_weather(cache_dir, group)
            actual = full_labels[group].reindex(source_weather.index)
            eligible = actual.notna() & np.isfinite(actual.to_numpy()) & (actual >= 0.10 * CAPACITY[group])
            density = estimate_density_ratio_weights(
                source_weather, test_weather, eligible_source=eligible, feature_names=feature_names
            )
            weighted_predictions, models, model_records = fit_weighted_components(
                source_weather, test_weather, actual, density.source_weight, group=group, recipe=recipe
            )
            model_path = out_dir / "models" / f"final_2025__{group}.joblib"
            joblib.dump(models, model_path, compress=3)
            candidate, _, assembly = build_group_candidate(
                PROJECT_DIR,
                FINAL_COMPONENTS,
                PROJECT_DIR / "artifacts/final_cf_fix/predictions/corrected_v3_test.parquet",
                weighted_predictions,
                final_index,
                group=group,
                recipe=recipe,
            )
            final_candidates[group] = candidate
            final_records[group] = {
                "domain": density.audit,
                "outcome_models": model_records,
                "assembly": assembly,
                "model": snapshot(model_path),
                "train_cache": snapshot(cache_path),
                "test_cache": snapshot(test_cache_path),
            }
        baseline_final_path = PROJECT_DIR / "artifacts/final_cf_fix/predictions/corrected_v3_test.parquet"
        predictions = pd.read_parquet(baseline_final_path, engine="pyarrow")
        predictions.index = pd.DatetimeIndex(predictions.index, name="forecast_kst_dtm")
        for group, values in final_candidates.items():
            predictions[group] = values
        prediction_path = out_dir / "predictions/density_ratio_selected_2025.parquet"
        write_parquet(predictions, prediction_path)
        sample_path = raw_dir / "sample_submission.csv"
        sample = pd.read_csv(sample_path, encoding="utf-8-sig")
        sample_times = pd.DatetimeIndex(pd.to_datetime(sample["forecast_kst_dtm"]), name="forecast_kst_dtm")
        if not sample_times.equals(predictions.index) or len(sample) != 8760:
            raise AssertionError("sample submission alignment changed")
        for group in ALL_GROUPS:
            sample[group] = predictions[group].to_numpy(dtype=float)
        submission_path = out_dir / "density_ratio_selected_2025.csv"
        sample.to_csv(submission_path, index=False, encoding="utf-8-sig", float_format="%.6f", lineterminator="\n")
        final_payload = {
            "executed": True,
            "promoted_groups": promoted_groups,
            "prediction": snapshot(prediction_path),
            "submission": snapshot(submission_path),
            "baseline": snapshot(baseline_final_path),
            "sample": snapshot(sample_path),
            "records": final_records,
            "score_claim": False,
        }
    final_path = out_dir / "final_results.json"
    write_json(final_path, final_payload)

    manifest = {
        "schema_version": 1,
        "experiment_id": prereg["experiment_id"],
        "preregister": snapshot(out_dir / "preregister.json"),
        "feature_names": snapshot(out_dir / "domain_features.txt"),
        "config": snapshot(recipe_path),
        "source": {
            "runner": snapshot(Path(__file__)),
            "module": snapshot(PROJECT_DIR / "src/density_ratio.py"),
            "test": snapshot(PROJECT_DIR / "tests/test_density_ratio.py"),
        },
        "physical_prefix_io_ledger": {
            "stage1_fit": fit_evidence,
            "stage1_validation_after_prescore_lock": validation_evidence,
            "2024_label_values_read": bool(locked_groups),
            "2025_label_values_read": False,
        },
        "locks": {
            "stage1_prescore": snapshot(stage1_prescore),
            "stage1_promotion": snapshot(stage1_lock_path),
            "stage2_promotion": snapshot(out_dir / "stage2_promotion_lock.json"),
        },
        "results": {
            "stage1": snapshot(stage1_results_path),
            "stage2": snapshot(stage2_path),
            "final": snapshot(final_path),
        },
        "locked_groups_stage1": locked_groups,
        "promoted_groups_stage2": promoted_groups,
        "group3_identity": True,
        "csv_created": bool(final_payload["executed"]),
        "leaderboard_score_claim": False,
    }
    write_json(out_dir / "manifest.json", manifest)
    print(json.dumps({"stage1_locked": locked_groups, "stage2_promoted": promoted_groups, "csv_created": bool(final_payload["executed"])}), flush=True)


if __name__ == "__main__":
    main()

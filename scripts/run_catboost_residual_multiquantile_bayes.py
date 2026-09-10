"""Strict corrected-v3-conditioned residual MultiQuantile Bayes experiment."""

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

from scripts import run_catboost_multiquantile_bayes as direct  # noqa: E402
from scripts import run_ficr_bayes_decision_strict as bayes  # noqa: E402
from scripts import run_shared_q07_multiseed as shared  # noqa: E402
from src.catboost_multiquantile import (  # noqa: E402
    CatBoostBayesActionConfig,
    CatBoostResidualMultiQuantileSurface,
    blend_with_baseline_kwh,
    exact_official_residual_utility_action_cf,
)
from src.manifest import describe_file, git_state, package_versions, sha256_file, utc_now  # noqa: E402
from src.metric import CAPACITY_KWH, TARGET_COLS, score_details  # noqa: E402


PREREGISTER_SHA256 = "60be50a27a6c4ffe46c4e4f3e1e5d61fead040b50aa59d867040031cd5209f7b"
FEASIBILITY_SHA256 = "9b93c12431afac043e389c00757a167724dd960e52c75c7cb1b6d06c2d116974"
CAUSAL_PROVENANCE_SHA256 = "0d252ff9518325be508327007235c2562d54d73fb524352bab30dafe2f9d0c86"
QUANTILE_LEVELS = (0.10, 0.25, 0.50, 0.75, 0.90)
QUANTILE_COLUMNS = ("q10_residual_cf", "q25_residual_cf", "q50_residual_cf", "q75_residual_cf", "q90_residual_cf")
LOSS_FUNCTION = "MultiQuantile:alpha=0.10,0.25,0.50,0.75,0.90"
BLEND_WEIGHTS = (0.05, 0.10, 0.20)
WEIGHT_KEYS = ("w05", "w10", "w20")
WEIGHT_BY_KEY = dict(zip(WEIGHT_KEYS, BLEND_WEIGHTS))
MINIMUM_FIT_ROWS = 800
STAGE1_REQUIRED = {
    "kpx_group_1": ("pooled_Q2_Q4", "Q2", "Q3", "Q4", "H2"),
    "kpx_group_2": ("pooled_Q2_Q4", "Q2", "Q3", "Q4", "H2"),
    "kpx_group_3": ("Q4", "OctNov", "Dec"),
}
STAGE2_REQUIRED = ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4")
MODEL_PARAMETERS = {
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
COMPONENT_ORDER = ("lgb_l1", "lgb_q07", "shared_l1", "shared_q07", "top200_q07", "energy_q06")
META_COLUMNS = (
    "meta__base_cf", "meta__lgb_l1_cf", "meta__lgb_q07_cf", "meta__shared_l1_cf",
    "meta__shared_q07_cf", "meta__top200_q07_cf", "meta__energy_q06_cf",
    "meta__component_mean_cf", "meta__component_std_cf", "meta__component_min_cf",
    "meta__component_max_cf", "meta__component_range_cf",
    "meta__lgb_q07_minus_lgb_l1_cf", "meta__shared_q07_minus_shared_l1_cf",
    "meta__base_minus_component_mean_cf", "meta__lead_fraction",
)
LABEL_PREFIX = {
    "Q1": (10920, "2023-04-01 00:00:00"),
    "Q2": (13104, "2023-07-01 00:00:00"),
    "Q3": (15312, "2023-10-01 00:00:00"),
    "pre2024": (17520, "2024-01-01 00:00:00"),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("stage1", "stage2", "final", "all"), default="all")
    parser.add_argument("--raw-dir", type=Path, default=Path(r"data/local/open"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/cache"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/postgate/catboost_residual_multiquantile_bayes_strict_v3"))
    parser.add_argument("--preregister", type=Path, default=Path("configs/catboost_residual_multiquantile_bayes_preregister_v3.json"))
    parser.add_argument("--feasibility", type=Path, default=Path("artifacts/audits/catboost_residual_multiquantile_feasibility_v2.json"))
    parser.add_argument("--causal-provenance", type=Path, default=Path("artifacts/audits/catboost_residual_causal_source_provenance_v1.json"))
    return parser.parse_args(argv)


def _verify_preregister(
    path: Path,
    feasibility_path: Path,
    causal_provenance_path: Path,
) -> tuple[dict[str, Any], CatBoostBayesActionConfig]:
    if sha256_file(path) != PREREGISTER_SHA256:
        raise AssertionError("residual preregister hash changed")
    if sha256_file(feasibility_path) != FEASIBILITY_SHA256:
        raise AssertionError("residual feasibility hash changed")
    if sha256_file(causal_provenance_path) != CAUSAL_PROVENANCE_SHA256:
        raise AssertionError("residual causal-source provenance hash changed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["feasibility_audit"]["sha256"] != FEASIBILITY_SHA256:
        raise AssertionError("canonical feasibility is not hash-bound by preregister")
    if payload["causal_source_provenance"]["sha256"] != CAUSAL_PROVENANCE_SHA256:
        raise AssertionError("causal provenance is not hash-bound by preregister")
    dist = payload["conditional_residual_distribution"]
    if dist["loss_function"] != LOSS_FUNCTION or tuple(dist["quantile_levels"]) != QUANTILE_LEVELS:
        raise AssertionError("residual quantile contract changed")
    if dist["parameters"] != MODEL_PARAMETERS:
        raise AssertionError("residual CatBoost parameters changed")
    if tuple(payload["candidate_family"]["fixed_blend_weights"]) != BLEND_WEIGHTS:
        raise AssertionError("residual blend weights changed")
    if tuple(payload["feature_contract"]["meta_features_in_fixed_order"]) != META_COLUMNS:
        raise AssertionError("residual meta feature order changed")
    if int(payload["target_contract"]["minimum_eligible_fit_rows"]) != MINIMUM_FIT_ROWS:
        raise AssertionError("minimum fit rows changed")
    action = CatBoostBayesActionConfig(
        quantile_levels=QUANTILE_LEVELS,
        interpolation_count=33,
        outcome_lower_cf=0.10,
        outcome_upper_cf=1.20,
        absolute_grid_lower_cf=0.01,
        absolute_grid_upper_cf=1.02,
        absolute_grid_step_cf=0.01,
        candidate_lower_cf=0.0,
        candidate_upper_cf=1.02,
        tie_tolerance=1e-12,
    )
    return payload, action


def _surface(action: CatBoostBayesActionConfig) -> CatBoostResidualMultiQuantileSurface:
    model = CatBoostResidualMultiQuantileSurface(
        quantile_levels=QUANTILE_LEVELS,
        loss_function=LOSS_FUNCTION,
        model_parameters=MODEL_PARAMETERS,
        minimum_actual_cf=0.10,
        action_config=action,
    )
    model.minimum_fit_rows = MINIMUM_FIT_ROWS
    return model


def _expected_index(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="h", name="forecast_kst_dtm")


def _label_prefix(path: Path, key: str, usecols: Sequence[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows, expected_end = LABEL_PREFIX[key]
    prefix_sha, byte_limit = shared._csv_prefix_identity(path, data_rows=rows)
    bounded = shared._BoundedRawReader(path, byte_limit=byte_limit)
    try:
        with io.BufferedReader(bounded, buffer_size=1024 * 1024) as stream:
            frame = pd.read_csv(stream, encoding="utf-8-sig", usecols=["kst_dtm", *usecols], memory_map=False)
            returned = bounded.bytes_returned
            position = bounded.underlying_position
    finally:
        bounded.close()
    if returned != byte_limit or position != byte_limit or len(frame) != rows:
        raise AssertionError("bounded residual label read crossed its cap")
    index = pd.DatetimeIndex(pd.to_datetime(frame.pop("kst_dtm"), errors="raise"), name="forecast_kst_dtm")
    frame.index = index
    frame = frame.astype(float)
    if index.min() != pd.Timestamp("2022-01-01 01:00:00") or index.max() != pd.Timestamp(expected_end):
        raise AssertionError("bounded residual label interval changed")
    next_field = shared._next_csv_first_field(path, after_data_rows=rows, prefix_bytes=byte_limit)
    return frame, {
        "path": str(path.resolve()), "key": key, "data_rows": rows,
        "physical_byte_limit": byte_limit, "physical_prefix_sha256": prefix_sha,
        "bytes_returned_to_parser": returned, "underlying_position": position,
        "materialized_columns": list(frame.columns), "materialized_end": index.max(),
        "next_row_first_field_only": next_field, "omitted_target_value_cells_materialized": 0,
    }


def _assert_file_spec(path: Path, spec: Sequence[Any]) -> None:
    _, expected_bytes, expected_sha = spec
    if path.stat().st_size != int(expected_bytes) or sha256_file(path) != expected_sha:
        raise AssertionError(f"registered input changed: {path}")


def _ledger_file(spec: Sequence[Any]) -> Path:
    path = Path(str(spec[0]))
    return path if path.is_absolute() else PROJECT_DIR / path


def _verify_causal_source_provenance(
    path: Path,
    preregister: Mapping[str, Any],
    *,
    stages: Sequence[str] = ("stage1", "stage2", "final"),
    verify_label_prefix_bytes: bool = True,
) -> dict[str, Any]:
    """Independently reject any source prediction lacking strict prior-label provenance."""

    if sha256_file(path) != CAUSAL_PROVENANCE_SHA256:
        raise AssertionError("causal-source provenance ledger changed")
    ledger = json.loads(path.read_text(encoding="utf-8"))
    selected_stages = set(stages)
    if not selected_stages or not selected_stages.issubset({"stage1", "stage2", "final"}):
        raise AssertionError("invalid causal-provenance verification stages")
    label_spec = ledger["official_label_file"]
    label_path = Path(label_spec["path"])
    evidence = ledger["immutable_evidence"]
    prefixes = ledger["bounded_label_prefixes"]

    fitted = ledger["fitted_source_records"]
    derived = ledger["derived_source_records"]
    all_ids = [record["source_id"] for record in (*fitted, *derived)]
    if len(all_ids) != len(set(all_ids)):
        raise AssertionError("causal-source IDs are not unique")
    known_ids = set(all_ids)
    persisted_paths_by_stage: dict[str, set[str]] = {
        "stage1": set(), "stage2": set(), "final": set()
    }
    needed_prefixes: set[str] = set()
    needed_evidence: set[str] = set()
    for record in fitted:
        if record["label_prefix"] not in prefixes:
            raise AssertionError(f"unknown label prefix for {record['source_id']}")
        for ref in record["evidence_refs"]:
            if ref not in evidence:
                raise AssertionError(f"unknown evidence for {record['source_id']}: {ref}")
        train_end = pd.Timestamp(record["train_end"])
        apply_start = pd.Timestamp(record["application_start"])
        prefix_end = pd.Timestamp(prefixes[record["label_prefix"]]["materialized_end"])
        if not train_end < apply_start or train_end != prefix_end:
            raise AssertionError(f"noncausal source interval: {record['source_id']}")
        if record["stage"] in selected_stages:
            prediction_path = _ledger_file(record["prediction"])
            model_path = _ledger_file(record["source_model"])
            _assert_file_spec(prediction_path, record["prediction"])
            _assert_file_spec(model_path, record["source_model"])
            persisted_paths_by_stage[record["stage"]].add(record["prediction"][0])
            needed_prefixes.add(record["label_prefix"])
            needed_evidence.update(record["evidence_refs"])

    for record in derived:
        unknown = set(record["parent_source_ids"]) - known_ids
        if unknown:
            raise AssertionError(f"unknown derived parents for {record['source_id']}: {unknown}")
        for ref in record["formula_evidence_refs"]:
            if ref not in evidence:
                raise AssertionError(f"unknown formula evidence for {record['source_id']}: {ref}")
        if record["stage"] in selected_stages:
            needed_evidence.update(record["formula_evidence_refs"])
        if record["stage"] in selected_stages and record["prediction"] is not None:
            prediction_path = _ledger_file(record["prediction"])
            _assert_file_spec(prediction_path, record["prediction"])
            persisted_paths_by_stage[record["stage"]].add(record["prediction"][0])

    for name in needed_evidence:
        spec = evidence[name]
        evidence_path = _ledger_file(spec)
        _assert_file_spec(evidence_path, spec)

    if verify_label_prefix_bytes:
        for prefix_id in needed_prefixes:
            record = prefixes[prefix_id]
            prefix_sha, byte_limit = shared._csv_prefix_identity(
                label_path, data_rows=int(record["data_rows"])
            )
            if byte_limit != int(record["physical_byte_limit"]):
                raise AssertionError(f"{prefix_id} label byte limit changed")
            if prefix_sha != record["physical_prefix_sha256"]:
                raise AssertionError(f"{prefix_id} label prefix SHA changed")
            if not record.get("is_complete_file", False):
                next_field = shared._next_csv_first_field(
                    label_path,
                    after_data_rows=int(record["data_rows"]),
                    prefix_bytes=byte_limit,
                )
                if next_field != record["next_row_first_field_only"]:
                    raise AssertionError(f"{prefix_id} next label timestamp changed")
        if "through_2024" in needed_prefixes:
            _assert_file_spec(
                label_path,
                (str(label_path), label_spec["bytes"], label_spec["sha256"]),
            )

    expected_paths_by_stage = {
        "stage1": {spec[0] for spec in preregister["stage1_expected_inputs"].values()},
        "stage2": {
            spec[0]
            for spec in preregister["stage2_expected_inputs_after_promotion_only"].values()
        },
        "final": {
            spec[0]
            for spec in preregister["final_expected_inputs_after_stage2_only"].values()
        },
    }
    for stage in selected_stages:
        if persisted_paths_by_stage[stage] != expected_paths_by_stage[stage]:
            raise AssertionError(
                f"causal-source ledger does not exactly cover {stage} predictions"
            )
    decision = ledger["causal_decision"]
    if not all(
        bool(decision[key])
        for key in (
            "all_fitted_sources_train_end_strictly_before_application_start",
            "all_derived_sources_have_only_registered_causal_parents",
            "all_stage1_top200_energy_and_six_g3_components_have_exact_model_sha",
            "fit_allowed_only_after_independent_verifier_pass",
        )
    ):
        raise AssertionError("causal-source ledger did not record a passing decision")
    return ledger


def _causal_descriptor(preregister: Mapping[str, Any]) -> dict[str, Any]:
    path = PROJECT_DIR / preregister["causal_source_provenance"]["path"]
    if sha256_file(path) != CAUSAL_PROVENANCE_SHA256:
        raise AssertionError("causal-source provenance changed before lock")
    return describe_file(path)


def _read_prediction(path: Path, index: pd.DatetimeIndex, columns: Sequence[str] = TARGET_COLS) -> pd.DataFrame:
    frame = pd.read_parquet(path, engine="pyarrow")
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if not frame.index.equals(index) or not set(columns).issubset(frame.columns):
        raise AssertionError(f"prediction index/schema changed: {path}")
    result = frame.loc[index, list(columns)].astype(float)
    if not np.isfinite(result.to_numpy()).all():
        raise AssertionError(f"prediction is nonfinite: {path}")
    return result


def _stage1_sources(artifact_root: Path, prereg: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    specs = prereg["stage1_expected_inputs"]
    for key, spec in specs.items():
        _assert_file_spec(PROJECT_DIR / spec[0], spec)
    index_g12 = _expected_index("2023-01-01 01:00", "2024-01-01 00:00")
    base12 = _read_prediction(PROJECT_DIR / specs["baseline_g12"][0], index_g12, TARGET_COLS[:2])
    components: dict[str, pd.DataFrame] = {}
    key_map = {
        "lgb_l1": "component_lgb_l1", "lgb_q07": "component_lgb_q07",
        "shared_l1": "component_shared_l1", "shared_q07": "component_shared_q07",
        "top200_q07": "component_top200_q07", "energy_q06": "component_energy_q06",
    }
    for name, key in key_map.items():
        components[name] = _read_prediction(PROJECT_DIR / specs[key][0], index_g12, TARGET_COLS[:2])
    g3_spec = specs["g3_components"]
    g3_path = PROJECT_DIR / g3_spec[0]
    index_g3 = _expected_index("2023-07-01 01:00", "2024-01-01 00:00")
    g3 = pd.read_parquet(g3_path, engine="pyarrow")
    g3.index = pd.DatetimeIndex(g3.index, name="forecast_kst_dtm")
    if not g3.index.equals(index_g3):
        raise AssertionError("g3 component index changed")
    mapping = {"lgb_l1": "l1", "lgb_q07": "q07", "shared_l1": "shared_l1", "shared_q07": "shared_q07", "top200_q07": "top200q07", "energy_q06": "ewq06"}
    for name, column in mapping.items():
        extension = pd.DataFrame(index=index_g12, columns=TARGET_COLS, dtype=float)
        extension.loc[index_g12, TARGET_COLS[:2]] = components[name].loc[index_g12, TARGET_COLS[:2]]
        extension.loc[index_g3, "kpx_group_3"] = g3[column].to_numpy(dtype=float)
        components[name] = extension
    baseline = pd.DataFrame(index=index_g12, columns=TARGET_COLS, dtype=float)
    baseline.loc[:, TARGET_COLS[:2]] = base12
    g3_base = np.clip(1.25 * (0.20*g3["q07"] + 0.075*g3["shared_l1"] + 0.425*g3["shared_q07"] + 0.025*g3["top200q07"] + 0.275*g3["ewq06"]) - 1200.0, 0.0, 1.02*CAPACITY_KWH["kpx_group_3"])
    baseline.loc[index_g3, "kpx_group_3"] = g3_base.to_numpy(dtype=float)
    return baseline, components, {key: describe_file(PROJECT_DIR / spec[0]) for key, spec in specs.items()}


def _period_sources(specs: Mapping[str, Sequence[Any]], index: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    for spec in specs.values():
        _assert_file_spec(PROJECT_DIR / spec[0], spec)
    baseline = _read_prediction(PROJECT_DIR / specs["baseline"][0], index)
    components = {name: _read_prediction(PROJECT_DIR / specs[name][0], index) for name in COMPONENT_ORDER}
    return baseline, components, {key: describe_file(PROJECT_DIR / spec[0]) for key, spec in specs.items()}


def _meta_features(index: pd.DatetimeIndex, group: str, baseline: pd.Series, components: Mapping[str, pd.Series]) -> pd.DataFrame:
    capacity = CAPACITY_KWH[group]
    values = np.column_stack([components[name].loc[index].to_numpy(dtype=float) / capacity for name in COMPONENT_ORDER])
    base = baseline.loc[index].to_numpy(dtype=float) / capacity
    mean = values.mean(axis=1)
    data = np.column_stack((
        base, values, mean, values.std(axis=1, ddof=0), values.min(axis=1), values.max(axis=1),
        values.max(axis=1)-values.min(axis=1), values[:,1]-values[:,0], values[:,3]-values[:,2],
        base-mean, ((index.hour.to_numpy()-1) % 24)/23.0,
    ))
    frame = pd.DataFrame(data.astype(np.float32), index=index, columns=META_COLUMNS)
    if frame.shape[1] != 16 or not np.isfinite(frame.to_numpy()).all():
        raise AssertionError("registered residual meta features changed")
    return frame


def _features(weather: pd.DataFrame, group: str, baseline: pd.Series, components: Mapping[str, pd.Series], index: pd.DatetimeIndex) -> pd.DataFrame:
    if weather.shape[1] != 612:
        raise AssertionError("weather feature count changed")
    result = pd.concat((weather.loc[index], _meta_features(index, group, baseline, components)), axis=1)
    if result.shape[1] != 628 or tuple(result.columns[-16:]) != META_COLUMNS:
        raise AssertionError("residual feature schema changed")
    return result


def _fit_predict(group: str, features: pd.DataFrame, actual: pd.Series, baseline: pd.Series, fit_index: pd.DatetimeIndex, apply_index: pd.DatetimeIndex, action_config: CatBoostBayesActionConfig) -> tuple[CatBoostResidualMultiQuantileSurface, pd.Series, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if len(fit_index.intersection(apply_index)) or not fit_index.max() < apply_index.min():
        raise AssertionError("residual fit/application is not strict-forward")
    print(f"fit residual MultiQuantile {group}: {fit_index.min()}..{fit_index.max()} -> {apply_index.min()}..{apply_index.max()}", flush=True)
    model = _surface(action_config).fit(features.loc[fit_index], actual.loc[fit_index], baseline.loc[fit_index], capacity_kwh=CAPACITY_KWH[group])
    action, repaired = model.predict_action(features.loc[apply_index], baseline.loc[apply_index])
    raw = pd.DataFrame(np.asarray(model.last_raw_quantiles_, dtype=float), index=apply_index, columns=QUANTILE_COLUMNS)
    repaired.columns = QUANTILE_COLUMNS
    metadata = model.metadata()
    metadata.update({"fit_start": fit_index.min(), "fit_end": fit_index.max(), "apply_start": apply_index.min(), "apply_end": apply_index.max(), "fit_application_overlap_count": 0})
    return model, action, raw, repaired, metadata


def _save_model_outputs(out_dir: Path, prefix: str, group: str, block: str, model: Any, action: pd.Series, raw: pd.DataFrame, repaired: pd.DataFrame) -> list[Path]:
    paths = [
        out_dir / "models" / f"{prefix}_{group}_{block}.joblib",
        out_dir / "oof" / f"{prefix}_{group}_{block}_action.parquet",
        out_dir / "oof" / f"{prefix}_{group}_{block}_raw_residual_quantiles.parquet",
        out_dir / "oof" / f"{prefix}_{group}_{block}_repaired_residual_quantiles.parquet",
    ]
    bayes._atomic_joblib(model, paths[0]); bayes._atomic_parquet(action.to_frame(), paths[1]); bayes._atomic_parquet(raw, paths[2]); bayes._atomic_parquet(repaired, paths[3])
    loaded = joblib.load(paths[0])
    return paths


def _write_block_prescore_lock(
    *,
    out_dir: Path,
    block_id: str,
    groups: Sequence[str],
    fit_prefix_evidence: Mapping[str, Any],
    application_index: pd.DatetimeIndex,
    baseline: pd.DataFrame,
    actions: pd.DataFrame,
    model_output_paths: Sequence[Path],
    feature_hashes: Mapping[str, str],
    causal_provenance_record: Mapping[str, Any],
) -> tuple[Path, list[Path]]:
    """Freeze one rolling block before its application labels can be read."""

    candidate_paths: list[Path] = []
    for key, weight in WEIGHT_BY_KEY.items():
        frame = baseline.loc[application_index, list(groups)].copy()
        for group in groups:
            frame[group] = blend_with_baseline_kwh(
                baseline.loc[application_index, group],
                actions.loc[application_index, group],
                weight=weight,
                capacity_kwh=CAPACITY_KWH[group],
            )
        path = out_dir / "oof" / f"stage1_{block_id}_candidate_{key}.parquet"
        bayes._atomic_parquet(frame, path)
        candidate_paths.append(path)
    lock_path = out_dir / "locks" / f"stage1_{block_id}_prescore_lock.json"
    bayes._write_json(
        lock_path,
        {
            "schema_version": 2,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "causal_source_provenance": dict(causal_provenance_record),
            "block_id": block_id,
            "groups": list(groups),
            "fit_prefix_evidence": fit_prefix_evidence,
            "application_start": application_index.min(),
            "application_end": application_index.max(),
            "application_label_value_cells_materialized": 0,
            "feature_frame_sha256": dict(feature_hashes),
            "model_quantile_action_outputs": [
                describe_file(path) for path in model_output_paths
            ],
            "all_three_blend_outputs": [
                describe_file(path) for path in candidate_paths
            ],
            "created_before_next_label_prefix_read": True,
            "public_information_used": False,
        },
    )
    return lock_path, candidate_paths


def _require_block_lock(path: Path, *, block_id: str) -> dict[str, Any]:
    if not path.is_file():
        raise AssertionError(f"{block_id} block lock missing before next label read")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("preregister_sha256") != PREREGISTER_SHA256:
        raise AssertionError(f"{block_id} block lock preregister changed")
    causal = payload.get("causal_source_provenance", {})
    if causal.get("sha256") != CAUSAL_PROVENANCE_SHA256:
        raise AssertionError(f"{block_id} block lock lost causal provenance")
    causal_path = Path(causal.get("path", ""))
    if not causal_path.is_file() or sha256_file(causal_path) != CAUSAL_PROVENANCE_SHA256:
        raise AssertionError(f"{block_id} causal provenance artifact changed")
    if payload.get("block_id") != block_id:
        raise AssertionError(f"{block_id} block lock id changed")
    if payload.get("application_label_value_cells_materialized") != 0:
        raise AssertionError(f"{block_id} application labels were materialized")
    for record in (
        payload.get("model_quantile_action_outputs", [])
        + payload.get("all_three_blend_outputs", [])
    ):
        artifact = Path(record["path"])
        if sha256_file(artifact) != record["sha256"]:
            raise AssertionError(f"{block_id} locked artifact changed: {artifact}")
    if len(payload.get("all_three_blend_outputs", [])) != 3:
        raise AssertionError(f"{block_id} does not lock all three blends")
    return payload


def _blend(baseline: pd.DataFrame, actions: pd.DataFrame, weight: float, active: Sequence[str]) -> pd.DataFrame:
    result = baseline.copy()
    for group in active:
        valid = actions[group].notna()
        result.loc[valid, group] = blend_with_baseline_kwh(baseline.loc[valid, group], actions.loc[valid, group], weight=weight, capacity_kwh=CAPACITY_KWH[group])
    return result


def _segments_stage1(group: str) -> dict[str, pd.DatetimeIndex]:
    if group != "kpx_group_3":
        return {
            "pooled_Q2_Q4": _expected_index("2023-04-01 01:00", "2024-01-01 00:00"),
            "Q2": _expected_index("2023-04-01 01:00", "2023-07-01 00:00"),
            "Q3": _expected_index("2023-07-01 01:00", "2023-10-01 00:00"),
            "Q4": _expected_index("2023-10-01 01:00", "2024-01-01 00:00"),
            "H2": _expected_index("2023-07-01 01:00", "2024-01-01 00:00"),
        }
    return {
        "Q4": _expected_index("2023-10-01 01:00", "2024-01-01 00:00"),
        "OctNov": _expected_index("2023-10-01 01:00", "2023-12-01 00:00"),
        "Dec": _expected_index("2023-12-01 01:00", "2024-01-01 00:00"),
    }


def _select_groups(comparisons: Mapping[str, Any]) -> tuple[dict[str, float | None], dict[str, Any]]:
    if len(comparisons) != len(WEIGHT_KEYS) or set(comparisons) != set(WEIGHT_KEYS):
        raise AssertionError("registered Stage1 weight keys changed")
    selected: dict[str, float | None] = {}
    audit: dict[str, Any] = {}
    for group in TARGET_COLS:
        candidates: dict[str, Any] = {}
        eligible: list[str] = []
        for key in WEIGHT_KEYS:
            if len(comparisons[key]) != len(TARGET_COLS) or set(comparisons[key]) != set(TARGET_COLS):
                raise AssertionError("registered Stage1 group keys changed")
            records = comparisons[key][group]
            required = STAGE1_REQUIRED[group]
            if len(records) != len(required) or set(records) != set(required):
                raise AssertionError("registered Stage1 segment keys changed")
            deltas = [float(records[s]["delta"]) for s in STAGE1_REQUIRED[group]]
            candidates[key] = {"weight": WEIGHT_BY_KEY[key], "minimum": min(deltas), "mean": float(np.mean(deltas)), "all_strictly_positive": all(x > 0 for x in deltas)}
            if candidates[key]["all_strictly_positive"]:
                eligible.append(key)
        if eligible:
            key = max(eligible, key=lambda k: (candidates[k]["minimum"], candidates[k]["mean"], -candidates[k]["weight"]))
            selected[group] = WEIGHT_BY_KEY[key]
            audit[group] = {"candidates": candidates, "selected": key}
        else:
            selected[group] = None
            audit[group] = {"candidates": candidates, "selected": "identity"}
    return selected, audit


def _stage1(
    raw_dir: Path,
    artifact_root: Path,
    out_dir: Path,
    prereg_path: Path,
    feasibility_path: Path,
    causal_path: Path,
    prereg: Mapping[str, Any],
    action_config: CatBoostBayesActionConfig,
) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"Stage1 requires new output directory: {out_dir}")
    out_dir.mkdir(parents=True)
    shutil.copyfile(prereg_path, out_dir / "preregister.json")
    shutil.copyfile(prereg_path.with_suffix(".sha256"), out_dir / "preregister.sha256")
    shutil.copyfile(feasibility_path, out_dir / "feasibility.json")
    shutil.copyfile(causal_path, out_dir / "causal_source_provenance.json")
    if sha256_file(out_dir / "causal_source_provenance.json") != CAUSAL_PROVENANCE_SHA256:
        raise AssertionError("copied causal-source ledger changed")
    causal_record = _causal_descriptor(prereg)
    baseline, components, source_audit = _stage1_sources(artifact_root, prereg)
    full_weather_index = _expected_index("2022-01-01 01:00", "2024-01-01 00:00")
    weather, raw_weather_audit = shared._read_stage1_raw_features(raw_dir, pd.DataFrame(index=full_weather_index))
    label_path = raw_dir / "train/train_labels.csv"
    q1_labels, q1_evidence = _label_prefix(label_path, "Q1", TARGET_COLS[:2])
    rolling = {
        "kpx_group_1": [("Q1_to_Q2", "Q1", _expected_index("2023-01-01 01:00", "2023-04-01 00:00"), _expected_index("2023-04-01 01:00", "2023-07-01 00:00")), ("H1_to_Q3", "Q2", _expected_index("2023-01-01 01:00", "2023-07-01 00:00"), _expected_index("2023-07-01 01:00", "2023-10-01 00:00")), ("Q1Q3_to_Q4", "Q3", _expected_index("2023-01-01 01:00", "2023-10-01 00:00"), _expected_index("2023-10-01 01:00", "2024-01-01 00:00"))],
        "kpx_group_2": [("Q1_to_Q2", "Q1", _expected_index("2023-01-01 01:00", "2023-04-01 00:00"), _expected_index("2023-04-01 01:00", "2023-07-01 00:00")), ("H1_to_Q3", "Q2", _expected_index("2023-01-01 01:00", "2023-07-01 00:00"), _expected_index("2023-07-01 01:00", "2023-10-01 00:00")), ("Q1Q3_to_Q4", "Q3", _expected_index("2023-01-01 01:00", "2023-10-01 00:00"), _expected_index("2023-10-01 01:00", "2024-01-01 00:00"))],
    }
    actions = pd.DataFrame(index=baseline.index, columns=TARGET_COLS, dtype=float)
    outputs: list[Path] = []
    training: dict[str, Any] = {}
    label_evidence: list[dict[str, Any]] = [q1_evidence]
    evidence_by_key: dict[str, dict[str, Any]] = {"Q1": q1_evidence}
    labels_by_key: dict[str, pd.DataFrame] = {"Q1": q1_labels}
    block_lock_paths: list[Path] = []
    for block_position in range(3):
        prefix_key = ("Q1", "Q2", "Q3")[block_position]
        if prefix_key not in labels_by_key:
            previous_id = ("g12_Q1_to_Q2", "g12_H1_to_Q3")[block_position - 1]
            _require_block_lock(block_lock_paths[-1], block_id=previous_id)
            labels_by_key[prefix_key], evidence = _label_prefix(label_path, prefix_key, TARGET_COLS[:2])
            label_evidence.append(evidence)
            evidence_by_key[prefix_key] = evidence
        labels = labels_by_key[prefix_key]
        block_outputs: list[Path] = []
        feature_hashes: dict[str, str] = {}
        block_id = ""
        for group in TARGET_COLS[:2]:
            block, expected_key, fit_index, apply_index = rolling[group][block_position]
            if expected_key != prefix_key:
                raise AssertionError("rolling prefix order changed")
            block_id = f"g12_{block}"
            cgroup = {name: frame[group] for name, frame in components.items()}
            all_index = fit_index.append(apply_index)
            frame = _features(weather[group], group, baseline[group], cgroup, all_index)
            feature_hashes[group] = shared._frame_sha256(frame)
            model, action, raw, repaired, meta = _fit_predict(group, frame, labels[group], baseline[group], fit_index, apply_index, action_config)
            actions.loc[apply_index, group] = action
            training[f"{group}/{block}"] = meta
            block_outputs.extend(_save_model_outputs(out_dir, "stage1", group, block, model, action, raw, repaired))
        block_lock, block_candidates = _write_block_prescore_lock(
            out_dir=out_dir,
            block_id=block_id,
            groups=TARGET_COLS[:2],
            fit_prefix_evidence=evidence_by_key[prefix_key],
            application_index=apply_index,
            baseline=baseline,
            actions=actions,
            model_output_paths=block_outputs,
            feature_hashes=feature_hashes,
            causal_provenance_record=causal_record,
        )
        _require_block_lock(block_lock, block_id=block_id)
        block_lock_paths.append(block_lock)
        outputs.extend((*block_outputs, *block_candidates, block_lock))
    _require_block_lock(block_lock_paths[-1], block_id="g12_Q1Q3_to_Q4")
    g3_labels, g3_evidence = _label_prefix(label_path, "Q3", ("kpx_group_3",))
    label_evidence.append(g3_evidence)
    group = "kpx_group_3"; fit_index = _expected_index("2023-07-01 01:00", "2023-10-01 00:00"); apply_index = _expected_index("2023-10-01 01:00", "2024-01-01 00:00")
    cgroup = {name: frame[group] for name, frame in components.items()}
    frame = _features(weather[group], group, baseline[group], cgroup, fit_index.append(apply_index))
    model, action, raw, repaired, meta = _fit_predict(group, frame, g3_labels[group], baseline[group], fit_index, apply_index, action_config)
    actions.loc[apply_index, group] = action; training[f"{group}/Q3_to_Q4"] = meta
    g3_outputs = _save_model_outputs(out_dir, "stage1", group, "Q3_to_Q4", model, action, raw, repaired)
    g3_lock, g3_candidates = _write_block_prescore_lock(
        out_dir=out_dir,
        block_id="g3_Q3_to_Q4",
        groups=(group,),
        fit_prefix_evidence=g3_evidence,
        application_index=apply_index,
        baseline=baseline,
        actions=actions,
        model_output_paths=g3_outputs,
        feature_hashes={group: shared._frame_sha256(frame)},
        causal_provenance_record=causal_record,
    )
    _require_block_lock(g3_lock, block_id="g3_Q3_to_Q4")
    block_lock_paths.append(g3_lock)
    outputs.extend((*g3_outputs, *g3_candidates, g3_lock))
    blends = {key: _blend(baseline, actions, weight, TARGET_COLS) for key, weight in WEIGHT_BY_KEY.items()}
    baseline_path = out_dir / "oof/stage1_baseline.parquet"; action_path = out_dir / "oof/stage1_actions.parquet"
    bayes._atomic_parquet(baseline, baseline_path); bayes._atomic_parquet(actions, action_path); outputs.extend((baseline_path, action_path))
    for key, candidate in blends.items():
        path = out_dir / "oof" / f"stage1_candidate_{key}.parquet"; bayes._atomic_parquet(candidate, path); outputs.append(path)
    zero = _blend(baseline, actions, 0.0, TARGET_COLS)
    zero_exact = all(zero[g].to_numpy().tobytes() == baseline[g].to_numpy().tobytes() for g in TARGET_COLS)
    if not zero_exact:
        raise AssertionError("weight-zero identity failed")
    prescore = {
        "schema_version": 1, "created_utc": utc_now(), "preregister_sha256": PREREGISTER_SHA256,
        "feasibility_sha256": FEASIBILITY_SHA256,
        "causal_source_provenance": causal_record,
        "causal_source_provenance_copy": describe_file(out_dir / "causal_source_provenance.json"),
        "source_predictions": source_audit,
        "raw_weather_audit": raw_weather_audit, "training": training, "bounded_fit_label_reads": label_evidence,
        "block_prescore_locks": [describe_file(path) for path in block_lock_paths],
        "candidate_outputs": [describe_file(path) for path in outputs], "weight_zero_identity_bit_exact": True,
        "stage1_application_label_values_materialized_before_prediction": 0,
        "2024_read": False, "2025_read": False, "public_information_used": False,
    }
    prescore_path = out_dir / "stage1_prescore_record.json"; bayes._write_json(prescore_path, prescore)
    prescore_lock_path = out_dir / "stage1_prescore_lock.json"; bayes._write_json(prescore_lock_path, {"schema_version": 3, "created_utc": utc_now(), "preregister_sha256": PREREGISTER_SHA256, "causal_source_provenance": causal_record, "prescore_record": describe_file(prescore_path), "block_prescore_locks": [describe_file(path) for path in block_lock_paths], "created_before_stage1_application_label_values": True, "2024_read": False, "2025_read": False})
    for expected_id, lock_path in zip(
        ("g12_Q1_to_Q2", "g12_H1_to_Q3", "g12_Q1Q3_to_Q4", "g3_Q3_to_Q4"),
        block_lock_paths,
    ):
        _require_block_lock(lock_path, block_id=expected_id)
    score_labels, score_evidence = _label_prefix(label_path, "pre2024", TARGET_COLS)
    comparisons: dict[str, Any] = {key: {} for key in WEIGHT_KEYS}
    for key in WEIGHT_KEYS:
        for group in TARGET_COLS:
            segments = _segments_stage1(group)
            comparisons[key][group] = bayes._comparison(score_labels[group], baseline[group], blends[key][group], group, segments)
    selected, selection = _select_groups(comparisons)
    result = {"schema_version": 3, "created_utc": utc_now(), "preregister_sha256": PREREGISTER_SHA256, "causal_source_provenance": causal_record, "prescore_lock": describe_file(prescore_lock_path), "score_label_evidence": score_evidence, "comparisons": comparisons, "comparisons_sha256": bayes._canonical_sha256(comparisons), "selected_weights": selected, "selection": selection, "locked_groups": [g for g,w in selected.items() if w is not None], "2024_read": False, "2025_read": False, "public_information_used": False}
    result_path = out_dir / "stage1_results.json"; bayes._write_json(result_path, result)
    lock = {"schema_version": 3, "created_utc": utc_now(), "preregister_sha256": PREREGISTER_SHA256, "causal_source_provenance": causal_record, "stage1_results": describe_file(result_path), "prescore_lock": describe_file(prescore_lock_path), "selected_weights": selected, "locked_groups": result["locked_groups"], "partial_group_stage2_intent": True, "2024_read": False, "2025_read": False}
    bayes._write_json(out_dir / "stage1_promotion_lock.json", lock)
    print(f"Stage1 locked groups: {lock['locked_groups']}", flush=True)
    return lock


def _load_stage1_lock(out_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    lock_path = out_dir / "stage1_promotion_lock.json"; result_path = out_dir / "stage1_results.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8")); result = json.loads(result_path.read_text(encoding="utf-8"))
    if lock["preregister_sha256"] != PREREGISTER_SHA256 or lock["stage1_results"]["sha256"] != sha256_file(result_path):
        raise AssertionError("Stage1 lock changed")
    for record in (lock.get("causal_source_provenance", {}), result.get("causal_source_provenance", {})):
        if record.get("sha256") != CAUSAL_PROVENANCE_SHA256 or sha256_file(Path(record["path"])) != CAUSAL_PROVENANCE_SHA256:
            raise AssertionError("Stage1 causal-source provenance changed")
    prescore_lock_path = Path(lock["prescore_lock"]["path"])
    if sha256_file(prescore_lock_path) != lock["prescore_lock"]["sha256"]:
        raise AssertionError("Stage1 global prescore lock changed")
    prescore_lock = json.loads(prescore_lock_path.read_text(encoding="utf-8"))
    if prescore_lock.get("preregister_sha256") != PREREGISTER_SHA256:
        raise AssertionError("Stage1 global prescore preregister changed")
    if prescore_lock.get("causal_source_provenance", {}).get("sha256") != CAUSAL_PROVENANCE_SHA256:
        raise AssertionError("Stage1 global prescore lost causal provenance")
    prescore_record = prescore_lock["prescore_record"]
    if sha256_file(Path(prescore_record["path"])) != prescore_record["sha256"]:
        raise AssertionError("Stage1 prescore record changed")
    expected_ids = ("g12_Q1_to_Q2", "g12_H1_to_Q3", "g12_Q1Q3_to_Q4", "g3_Q3_to_Q4")
    block_records = prescore_lock.get("block_prescore_locks", [])
    if len(block_records) != len(expected_ids):
        raise AssertionError("Stage1 global lock lost a rolling block")
    for block_id, block_record in zip(expected_ids, block_records):
        block_path = Path(block_record["path"])
        if sha256_file(block_path) != block_record["sha256"]:
            raise AssertionError(f"Stage1 rolling block lock changed: {block_id}")
        _require_block_lock(block_path, block_id=block_id)
    selected, selection = _select_groups(result["comparisons"])
    if selected != result["selected_weights"] or selection != result["selection"] or selected != lock["selected_weights"]:
        raise AssertionError("Stage1 selection recomputation changed")
    return lock, result


def _year_segments(year: int) -> dict[str, pd.DatetimeIndex]:
    return direct._year_segments(year)


def _total_comparison(actual: pd.DataFrame, baseline: pd.DataFrame, candidate: pd.DataFrame, segments: Mapping[str, pd.DatetimeIndex]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, index in segments.items():
        base = score_details(actual.loc[index], baseline.loc[index]).as_dict(); cand = score_details(actual.loc[index], candidate.loc[index]).as_dict()
        result[name] = {"baseline": base, "candidate": cand, "delta": float(cand["total_score"] - base["total_score"])}
    return result


def _series_value_bits_equal(left: pd.Series, right: pd.Series) -> bool:
    return (
        left.index.equals(right.index)
        and left.dtype == right.dtype
        and left.to_numpy(copy=False).tobytes() == right.to_numpy(copy=False).tobytes()
    )


def _stage2_partial_candidate(
    baseline: pd.DataFrame,
    full_locked_candidate: pd.DataFrame,
    locked_groups: Sequence[str],
    group_comparisons: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, bool], list[str], dict[str, bool]]:
    """Apply the preregistered per-group seven-slice identity fallback."""

    if not baseline.index.equals(full_locked_candidate.index):
        raise AssertionError("Stage2 candidate index changed")
    if tuple(baseline.columns) != tuple(full_locked_candidate.columns):
        raise AssertionError("Stage2 candidate schema changed")
    locked = list(locked_groups)
    if len(locked) != len(set(locked)) or any(group not in TARGET_COLS for group in locked):
        raise AssertionError("Stage2 locked groups changed")
    if set(group_comparisons) != set(locked):
        raise AssertionError("Stage2 group comparison coverage changed")
    passed_by_group: dict[str, bool] = {}
    for group in locked:
        records = group_comparisons[group]
        if tuple(records) != STAGE2_REQUIRED:
            raise AssertionError("registered Stage2 segment order changed")
        passed_by_group[group] = all(
            float(records[segment]["delta"]) > 0.0 for segment in STAGE2_REQUIRED
        )
    group_passed = [group for group in locked if passed_by_group[group]]
    partial = baseline.copy(deep=True)
    for group in group_passed:
        partial[group] = full_locked_candidate[group].to_numpy(copy=True)
    identity_by_group = {
        group: _series_value_bits_equal(partial[group], baseline[group])
        for group in TARGET_COLS
        if group not in group_passed
    }
    if not all(identity_by_group.values()):
        raise AssertionError("Stage2 rejected group is not bit-exact identity")
    return partial, passed_by_group, group_passed, identity_by_group


def _stage2_promotion_decision(
    group_passed: Sequence[str],
    aggregate_comparisons: Mapping[str, Any],
) -> dict[str, Any]:
    if tuple(aggregate_comparisons) != STAGE2_REQUIRED:
        raise AssertionError("registered Stage2 aggregate segment order changed")
    aggregate_all_positive = all(
        float(aggregate_comparisons[segment]["delta"]) > 0.0
        for segment in STAGE2_REQUIRED
    )
    promoted = bool(group_passed) and aggregate_all_positive
    return {
        "at_least_one_group_passed": bool(group_passed),
        "aggregate_all_seven_positive": aggregate_all_positive,
        "promoted": promoted,
        "promoted_groups": list(group_passed) if promoted else [],
    }


def _stage2(
    raw_dir: Path,
    artifact_root: Path,
    cache_dir: Path,
    out_dir: Path,
    prereg: Mapping[str, Any],
    action_config: CatBoostBayesActionConfig,
) -> dict[str, Any]:
    stage1_lock, _ = _load_stage1_lock(out_dir)
    locked = list(stage1_lock["locked_groups"])
    fixed_weights = dict(stage1_lock["selected_weights"])
    causal_record = _causal_descriptor(prereg)
    if not locked:
        result = {
            "schema_version": 3,
            "created_utc": utc_now(),
            "executed": False,
            "reason": "all Stage1 groups identity",
            "stage1_locked_groups": [],
            "group_passed_groups": [],
            "promoted_groups": [],
            "fixed_weights": fixed_weights,
            "partial_group_confirmation_intended": True,
            "2024_read": False,
            "2025_read": False,
            "promoted": False,
        }
        result_path = out_dir / "stage2_results.json"
        bayes._write_json(result_path, result)
        lock = {
            "schema_version": 3,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "causal_source_provenance": causal_record,
            "stage1_promotion_lock": describe_file(out_dir / "stage1_promotion_lock.json"),
            "stage2_results": describe_file(result_path),
            "stage1_locked_groups": [],
            "group_passed_groups": [],
            "promoted_groups": [],
            "final_identity_groups": list(TARGET_COLS),
            "fixed_weights": fixed_weights,
            "partial_group_confirmation_intended": True,
            "promoted": False,
            "2024_read": False,
            "2025_read": False,
            "csv_allowed": False,
        }
        bayes._write_json(out_dir / "stage2_promotion_lock.json", lock)
        return lock

    _verify_causal_source_provenance(
        PROJECT_DIR / prereg["causal_source_provenance"]["path"],
        prereg,
        stages=("stage2",),
        verify_label_prefix_bytes=True,
    )

    labels_2023, source_evidence = _label_prefix(
        raw_dir / "train/train_labels.csv", "pre2024", TARGET_COLS
    )
    stage1_base, stage1_components, _ = _stage1_sources(artifact_root, prereg)
    index_2024 = _expected_index("2024-01-01 01:00", "2025-01-01 00:00")
    gate_base, gate_components, gate_audit = _period_sources(
        prereg["stage2_expected_inputs_after_promotion_only"], index_2024
    )
    weather_index = _expected_index("2022-01-01 01:00", "2025-01-01 00:00")
    weather = shared._read_features(
        cache_dir, pd.DataFrame(index=weather_index), expected_end=bayes.YEAR_2024_END
    )
    actions = pd.DataFrame(np.nan, index=index_2024, columns=TARGET_COLS)
    outputs: list[Path] = []
    training: dict[str, Any] = {}
    for group in locked:
        fit_index = (
            _expected_index("2023-07-01 01:00", "2024-01-01 00:00")
            if group == "kpx_group_3"
            else _expected_index("2023-01-01 01:00", "2024-01-01 00:00")
        )
        joined_base = pd.concat((stage1_base.loc[fit_index, group], gate_base[group]))
        joined_components = {
            name: pd.concat(
                (stage1_components[name].loc[fit_index, group], gate_components[name][group])
            )
            for name in COMPONENT_ORDER
        }
        joined_weather = pd.concat(
            (weather[group].loc[fit_index], weather[group].loc[index_2024])
        )
        feature_frame = _features(
            joined_weather,
            group,
            joined_base,
            joined_components,
            fit_index.append(index_2024),
        )
        model, action, raw, repaired, meta = _fit_predict(
            group,
            feature_frame,
            labels_2023[group],
            joined_base,
            fit_index,
            index_2024,
            action_config,
        )
        actions[group] = action
        training[group] = meta
        outputs.extend(
            _save_model_outputs(out_dir, "stage2", group, "fixed", model, action, raw, repaired)
        )

    full_locked_candidate = gate_base.copy(deep=True)
    for group in locked:
        full_locked_candidate[group] = blend_with_baseline_kwh(
            gate_base[group],
            actions[group],
            weight=float(fixed_weights[group]),
            capacity_kwh=CAPACITY_KWH[group],
        )
    for name, frame in (
        ("stage2_baseline", gate_base),
        ("stage2_actions", actions),
        ("stage2_full_locked_candidate", full_locked_candidate),
    ):
        output_path = out_dir / "oof" / f"{name}.parquet"
        bayes._atomic_parquet(frame, output_path)
        outputs.append(output_path)
    prescore = {
        "schema_version": 3,
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "causal_source_provenance": causal_record,
        "stage1_lock": describe_file(out_dir / "stage1_promotion_lock.json"),
        "stage1_locked_groups": locked,
        "fixed_weights": fixed_weights,
        "source_label_evidence": source_evidence,
        "gate_input_audit": gate_audit,
        "training": training,
        "outputs": [describe_file(path) for path in outputs],
        "2024_application_label_values_materialized": 0,
        "partial_group_confirmation_intended": True,
    }
    prescore_path = out_dir / "stage2_prescore_record.json"
    bayes._write_json(prescore_path, prescore)
    prescore_lock_path = out_dir / "stage2_prescore_lock.json"
    bayes._write_json(
        prescore_lock_path,
        {
            "schema_version": 3,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "causal_source_provenance": causal_record,
            "prescore_record": describe_file(prescore_path),
            "created_before_2024_application_label_values": True,
        },
    )

    full_labels = bayes._read_full_labels(raw_dir / "train/train_labels.csv")
    segments = _year_segments(2024)
    registered_segments = {name: segments[name] for name in STAGE2_REQUIRED}
    group_comparisons = {
        group: bayes._comparison(
            full_labels[group],
            gate_base[group],
            full_locked_candidate[group],
            group,
            registered_segments,
        )
        for group in locked
    }
    partial_candidate, passed_by_group, group_passed, identity_by_group = (
        _stage2_partial_candidate(
            gate_base, full_locked_candidate, locked, group_comparisons
        )
    )
    aggregate = _total_comparison(
        full_labels.loc[index_2024], gate_base, partial_candidate, registered_segments
    )
    decision = _stage2_promotion_decision(group_passed, aggregate)
    partial_path = out_dir / "oof/stage2_partial_candidate.parquet"
    bayes._atomic_parquet(partial_candidate, partial_path)
    result = {
        "schema_version": 3,
        "created_utc": utc_now(),
        "executed": True,
        "preregister_sha256": PREREGISTER_SHA256,
        "causal_source_provenance": causal_record,
        "stage2_prescore_lock": describe_file(prescore_lock_path),
        "stage1_locked_groups": locked,
        "fixed_weights": fixed_weights,
        "group_comparisons": group_comparisons,
        "group_all_seven_positive_by_group": passed_by_group,
        "group_passed_groups": group_passed,
        "group_rejected_to_identity": [group for group in locked if group not in group_passed],
        "identity_bit_exact_by_nonpassed_group": identity_by_group,
        "partial_candidate": describe_file(partial_path),
        "aggregate_comparisons": aggregate,
        **decision,
        "partial_group_confirmation_intended": True,
        "no_weight_model_action_or_group_reselection": True,
        "2024_read": True,
        "2025_read": False,
    }
    result_path = out_dir / "stage2_results.json"
    bayes._write_json(result_path, result)
    promoted_groups = list(decision["promoted_groups"])
    lock = {
        "schema_version": 3,
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "causal_source_provenance": causal_record,
        "stage1_promotion_lock": describe_file(out_dir / "stage1_promotion_lock.json"),
        "stage2_results": describe_file(result_path),
        "stage2_prescore_lock": describe_file(prescore_lock_path),
        "stage1_locked_groups": locked,
        "group_passed_groups": group_passed,
        "promoted_groups": promoted_groups,
        "final_identity_groups": [group for group in TARGET_COLS if group not in promoted_groups],
        "fixed_weights": fixed_weights,
        "partial_group_confirmation_intended": True,
        "promoted": bool(decision["promoted"]),
        "2024_read": True,
        "2025_read": False,
        "csv_allowed": bool(decision["promoted"]),
    }
    bayes._write_json(out_dir / "stage2_promotion_lock.json", lock)
    return lock


def _load_stage2_lock(out_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    stage1, _ = _load_stage1_lock(out_dir)
    lock_path = out_dir / "stage2_promotion_lock.json"
    result_path = out_dir / "stage2_results.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if lock.get("preregister_sha256") != PREREGISTER_SHA256:
        raise AssertionError("Stage2 preregister lock changed")
    if lock["stage2_results"]["sha256"] != sha256_file(result_path):
        raise AssertionError("Stage2 result lock changed")
    if lock["stage1_locked_groups"] != stage1["locked_groups"]:
        raise AssertionError("Stage2 Stage1-group lock changed")
    if lock["fixed_weights"] != stage1["selected_weights"]:
        raise AssertionError("Stage2 fixed weights changed")
    causal = lock.get("causal_source_provenance", {})
    if causal.get("sha256") != CAUSAL_PROVENANCE_SHA256 or sha256_file(Path(causal["path"])) != CAUSAL_PROVENANCE_SHA256:
        raise AssertionError("Stage2 causal-source provenance changed")
    if not result.get("executed", False):
        required_identity = {
            "stage1_locked_groups": [],
            "group_passed_groups": [],
            "promoted_groups": [],
            "promoted": False,
            "csv_allowed": False,
        }
        for key, expected in required_identity.items():
            if lock.get(key) != expected:
                raise AssertionError(f"Stage2 identity lock changed: {key}")
        return lock, result

    prescore_lock_path = Path(lock["stage2_prescore_lock"]["path"])
    if sha256_file(prescore_lock_path) != lock["stage2_prescore_lock"]["sha256"]:
        raise AssertionError("Stage2 prescore lock changed")
    prescore_lock = json.loads(prescore_lock_path.read_text(encoding="utf-8"))
    if prescore_lock.get("preregister_sha256") != PREREGISTER_SHA256:
        raise AssertionError("Stage2 prescore preregister changed")
    if prescore_lock.get("causal_source_provenance", {}).get("sha256") != CAUSAL_PROVENANCE_SHA256:
        raise AssertionError("Stage2 prescore lost causal provenance")
    prescore_record = prescore_lock["prescore_record"]
    if sha256_file(Path(prescore_record["path"])) != prescore_record["sha256"]:
        raise AssertionError("Stage2 prescore record changed")
    prescore = json.loads(Path(prescore_record["path"]).read_text(encoding="utf-8"))
    for output in prescore["outputs"]:
        if sha256_file(Path(output["path"])) != output["sha256"]:
            raise AssertionError(f"Stage2 locked prescore output changed: {output['path']}")

    baseline = pd.read_parquet(out_dir / "oof/stage2_baseline.parquet", engine="pyarrow")
    full_candidate = pd.read_parquet(
        out_dir / "oof/stage2_full_locked_candidate.parquet", engine="pyarrow"
    )
    partial_path = Path(result["partial_candidate"]["path"])
    if sha256_file(partial_path) != result["partial_candidate"]["sha256"]:
        raise AssertionError("Stage2 partial candidate changed")
    partial = pd.read_parquet(partial_path, engine="pyarrow")
    for frame in (baseline, full_candidate, partial):
        frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    expected_partial, passed_by_group, group_passed, identity_by_group = (
        _stage2_partial_candidate(
            baseline,
            full_candidate,
            stage1["locked_groups"],
            result["group_comparisons"],
        )
    )
    if not all(
        _series_value_bits_equal(partial[group], expected_partial[group])
        for group in TARGET_COLS
    ):
        raise AssertionError("Stage2 partial candidate cannot be recomputed")
    decision = _stage2_promotion_decision(
        group_passed, result["aggregate_comparisons"]
    )
    if passed_by_group != result["group_all_seven_positive_by_group"]:
        raise AssertionError("Stage2 per-group seven-slice decision changed")
    if group_passed != result["group_passed_groups"]:
        raise AssertionError("Stage2 passed-group subset changed")
    if identity_by_group != result["identity_bit_exact_by_nonpassed_group"]:
        raise AssertionError("Stage2 identity fallback record changed")
    for key, expected in decision.items():
        if result.get(key) != expected:
            raise AssertionError(f"Stage2 promotion recomputation changed: {key}")
    if lock["group_passed_groups"] != group_passed:
        raise AssertionError("Stage2 group-passed lock changed")
    if lock["promoted_groups"] != decision["promoted_groups"]:
        raise AssertionError("Stage2 promoted subset lock changed")
    if bool(lock["promoted"]) != bool(decision["promoted"]):
        raise AssertionError("Stage2 aggregate promotion lock changed")
    if bool(lock["csv_allowed"]) != bool(decision["promoted"]):
        raise AssertionError("Stage2 CSV permission changed")
    return lock, result


def _finalize(raw_dir: Path, cache_dir: Path, out_dir: Path, prereg_path: Path, feasibility_path: Path, prereg: Mapping[str, Any], action_config: CatBoostBayesActionConfig) -> dict[str, Any]:
    lock, stage2 = _load_stage2_lock(out_dir); final_path = out_dir / "final_results.json"; manifest_path = out_dir / "manifest.json"
    if final_path.exists() or manifest_path.exists():
        raise FileExistsError("final artifacts already exist")
    if not lock["promoted"]:
        final = {"schema_version": 3, "created_utc": utc_now(), "executed": False, "promoted": False, "promoted_groups": [], "identity_groups": list(TARGET_COLS), "2025_read": False, "submission_created": False, "reason": "no partial subset passed both per-group and mixed-aggregate preregistered gates"}
        bayes._write_json(final_path, final)
    else:
        _verify_causal_source_provenance(
            PROJECT_DIR / prereg["causal_source_provenance"]["path"],
            prereg,
            stages=("final",),
            verify_label_prefix_bytes=True,
        )
        index_2024 = _expected_index("2024-01-01 01:00", "2025-01-01 00:00"); index_2025 = _expected_index("2025-01-01 01:00", "2026-01-01 00:00")
        labels = bayes._read_full_labels(raw_dir / "train/train_labels.csv")
        gate_base, gate_components, _ = _period_sources(prereg["stage2_expected_inputs_after_promotion_only"], index_2024)
        final_base, final_components, final_input_audit = _period_sources(prereg["final_expected_inputs_after_stage2_only"], index_2025)
        weather_index = _expected_index("2022-01-01 01:00", "2025-01-01 00:00"); train_weather = shared._read_features(cache_dir, pd.DataFrame(index=weather_index), expected_end=bayes.YEAR_2024_END); test_weather = shared._read_test_features(cache_dir, index_2025)
        candidate = final_base.copy(); actions = pd.DataFrame(np.nan, index=index_2025, columns=TARGET_COLS); outputs: list[Path] = []; training: dict[str, Any] = {}
        for group in lock["promoted_groups"]:
            joined_base = pd.concat((gate_base[group], final_base[group])); joined_components = {name: pd.concat((gate_components[name][group], final_components[name][group])) for name in COMPONENT_ORDER}; joined_weather = pd.concat((train_weather[group].loc[index_2024], test_weather[group]))
            frame = _features(joined_weather, group, joined_base, joined_components, index_2024.append(index_2025))
            model, action, raw, repaired, meta = _fit_predict(group, frame, labels[group], joined_base, index_2024, index_2025, action_config)
            actions[group] = action; training[group] = meta; outputs.extend(_save_model_outputs(out_dir, "final", group, "fixed", model, action, raw, repaired)); candidate[group] = blend_with_baseline_kwh(final_base[group], action, weight=float(lock["fixed_weights"][group]), capacity_kwh=CAPACITY_KWH[group])
        identity_groups = [group for group in TARGET_COLS if group not in lock["promoted_groups"]]
        identity_checks = {
            group: _series_value_bits_equal(candidate[group], final_base[group])
            for group in identity_groups
        }
        if not all(identity_checks.values()):
            raise AssertionError("final nonpromoted group is not bit-exact corrected-v3 identity")
        sample = pd.read_csv(raw_dir / "sample_submission.csv", encoding="utf-8-sig", dtype={"forecast_id": "string", "forecast_kst_dtm": "string"})
        if tuple(sample.columns) != ("forecast_id", "forecast_kst_dtm", *TARGET_COLS) or not pd.DatetimeIndex(pd.to_datetime(sample["forecast_kst_dtm"]), name="forecast_kst_dtm").equals(index_2025):
            raise AssertionError("sample schema/index changed")
        prediction_path = out_dir / "predictions/catboost_residual_multiquantile_bayes_2025.parquet"; action_path = out_dir / "predictions/catboost_residual_multiquantile_action_2025.parquet"; csv_path = out_dir / "catboost_residual_multiquantile_bayes_2025.csv"
        bayes._atomic_parquet(candidate, prediction_path); bayes._atomic_parquet(actions, action_path); submission = sample.copy()
        for group in TARGET_COLS: submission[group] = candidate[group].to_numpy(dtype=float)
        bayes._atomic_csv(submission, csv_path); verification = bayes._verify_submission(csv_path, sample, candidate); outputs.extend((prediction_path, action_path, csv_path))
        final = {"schema_version": 3, "created_utc": utc_now(), "executed": True, "promoted": True, "promoted_groups": list(lock["promoted_groups"]), "identity_groups": identity_groups, "identity_bit_exact_by_group": identity_checks, "partial_group_confirmation_intended": True, "2025_read": True, "submission_created": True, "training": training, "final_input_audit": final_input_audit, "outputs": [describe_file(path) for path in outputs], "submission_verification": verification}
        bayes._write_json(final_path, final)
    outputs = sorted(path for path in out_dir.rglob("*") if path.is_file() and path != manifest_path)
    manifest = {
        "schema_version": 3,
        "artifact_type": "catboost_residual_multiquantile_bayes_strict_forward_v3",
        "created_utc": utc_now(),
        "command": [sys.executable, *sys.argv],
        "runtime": {"packages": package_versions(), "git": git_state(PROJECT_DIR)},
        "preregister": describe_file(prereg_path),
        "preregister_sha256": PREREGISTER_SHA256,
        "feasibility": describe_file(feasibility_path),
        "feasibility_sha256": FEASIBILITY_SHA256,
        "causal_source_provenance": _causal_descriptor(prereg),
        "stage1_locked_groups": list(lock["stage1_locked_groups"]),
        "group_passed_groups": list(lock["group_passed_groups"]),
        "promoted_groups": list(lock["promoted_groups"]),
        "final_identity_groups": list(lock["final_identity_groups"]),
        "fixed_weights": lock["fixed_weights"],
        "promoted": bool(lock["promoted"]),
        "partial_group_confirmation_intended": True,
        "read_flags": {
            "2024_read": bool(stage2.get("2024_read")),
            "2025_read": bool(final["2025_read"]),
            "csv_created": bool(final["submission_created"]),
        },
        "contracts": {
            "continuous_residual_quantiles": True,
            "baseline_component_conditioned": True,
            "same_row_source_and_residual_fit_forbidden": True,
            "partial_group_stage2_then_mixed_aggregate": True,
            "nonpromoted_groups_bit_exact_identity": True,
            "public_information_used": False,
            "leaderboard_score_claim": False,
        },
        "outputs": [describe_file(path) for path in outputs],
    }
    bayes._write_json(manifest_path, manifest); return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv); raw_dir = args.raw_dir.resolve(); artifact_root = args.artifact_root.resolve(); cache_dir = args.cache_dir.resolve(); out_dir = args.out_dir.resolve(); prereg_path = args.preregister.resolve(); feasibility_path = args.feasibility.resolve(); causal_path = args.causal_provenance.resolve()
    prereg, action = _verify_preregister(prereg_path, feasibility_path, causal_path)
    _verify_causal_source_provenance(
        causal_path,
        prereg,
        stages=("stage1",),
        verify_label_prefix_bytes=True,
    )
    if args.stage in ("stage1", "all"):
        _stage1(raw_dir, artifact_root, out_dir, prereg_path, feasibility_path, causal_path, prereg, action)
        if args.stage == "stage1": return 0
    if args.stage in ("stage2", "all"):
        _stage2(raw_dir, artifact_root, cache_dir, out_dir, prereg, action)
        if args.stage == "stage2": return 0
    if args.stage in ("final", "all"):
        _finalize(raw_dir, cache_dir, out_dir, prereg_path, feasibility_path, prereg, action)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Strict-forward compact 24-hour forecast-run residual experiment.

Stage 1 builds weather features from a physically bounded pre-2024 raw prefix,
fits only the preregistered residual candidates, and creates an immutable lock.
Stage 2 reads 2024 only when exactly one recipe passed every registered Stage 1
slice.  A 2025 prediction and submission are created only after the fixed recipe
also improves every registered 2024 group/slice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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

from scripts.run_shared_q07_multiseed import (  # noqa: E402
    COMPONENTS,
    EXPECTED_ROWS_PRE2024,
    EXPECTED_ROWS_THROUGH2024,
    EXPECTED_TEST_ROWS,
    FINAL_COMPONENT_FILES,
    GATE_COMPONENT_FILES,
    OFFICIAL_LABEL_PREFIX_SHA256,
    OFFICIAL_STAGE1_PREFIX_BYTES,
    YEAR_2022_START,
    YEAR_2023_END,
    YEAR_2024_END,
    _assemble_group,
    _read_features,
    _read_labels,
    _read_prediction,
    _read_stage1_raw_features,
    _read_test_features,
)
from src.manifest import (  # noqa: E402
    describe_file,
    git_state,
    package_versions,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from src.metric import CAPACITY_KWH, TARGET_COLS, group_metrics  # noqa: E402
from src.run_sequence_residual import (  # noqa: E402
    apply_residual_model,
    assert_strict_fit_predict_order,
    build_run_sequence_features,
    choose_single_candidate,
    fit_residual_model,
)


PREREGISTER_SHA256 = (
    "cb0f87dfeed495b964fcaef8535e9b92bda83ad3313ad86064b4a7ca33f787d6"
)
STAGE1_START = pd.Timestamp("2023-01-01 01:00:00")
STAGE1_END = pd.Timestamp("2024-01-01 00:00:00")
G3_STAGE1_START = pd.Timestamp("2023-07-01 01:00:00")
STAGE2_START = pd.Timestamp("2024-01-01 01:00:00")
STAGE2_END = pd.Timestamp("2025-01-01 00:00:00")
FINAL_START = pd.Timestamp("2025-01-01 01:00:00")
FINAL_END = pd.Timestamp("2026-01-01 00:00:00")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument(
        "--cache-dir", type=Path, default=PROJECT_DIR / "artifacts" / "cache"
    )
    parser.add_argument(
        "--artifact-root", type=Path, default=PROJECT_DIR / "artifacts"
    )
    parser.add_argument(
        "--recipe",
        type=Path,
        default=PROJECT_DIR / "configs" / "train_final.v3.locked.json",
    )
    parser.add_argument(
        "--preregister",
        type=Path,
        default=PROJECT_DIR / "configs" / "run_sequence_residual_preregister.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_DIR
        / "artifacts"
        / "postgate"
        / "run_sequence_residual",
    )
    parser.add_argument(
        "--stage", choices=("stage1", "stage2", "all"), default="all"
    )
    return parser.parse_args(argv)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    write_json_atomic(path, _json_ready(payload), overwrite=False)


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        frame.to_parquet(temporary, engine="pyarrow", compression="zstd", index=True)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_joblib(payload: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        joblib.dump(payload, temporary, compress=3)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _frame_sha256(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update("\n".join(map(str, frame.columns)).encode("utf-8"))
    digest.update("\n".join(map(str, frame.dtypes)).encode("utf-8"))
    digest.update(np.asarray(frame.index.view("i8"), dtype="<i8").tobytes())
    values = np.asarray(frame.to_numpy(dtype=np.float64), dtype="<f8", order="C")
    digest.update(values.tobytes())
    return digest.hexdigest()


def _mapping_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _json_ready(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_preregister(path: Path) -> dict[str, Any]:
    observed = sha256_file(path)
    if observed != PREREGISTER_SHA256:
        raise AssertionError(
            f"preregister SHA changed before candidate reads: {observed}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("experiment_id") != "run_sequence_residual_strict_forward_v1":
        raise AssertionError("unexpected preregister experiment id")
    candidate_ids = [item["id"] for item in payload.get("candidates", [])]
    if candidate_ids != [
        "ridge_a10_s025",
        "huber_e135_a001_s025",
        "lgb_l1_leaf7_s025",
    ]:
        raise AssertionError("preregistered candidate set changed")
    if payload["stage1"]["selection_rule"]["locked_candidate_count_max"] != 1:
        raise AssertionError("preregister no longer limits the lock to one recipe")
    return payload


def _source_snapshot(preregister_path: Path) -> dict[str, Any]:
    paths = {
        "runner": Path(__file__).resolve(),
        "run_sequence_module": PROJECT_DIR / "src" / "run_sequence_residual.py",
        "strict_raw_helper": PROJECT_DIR / "scripts" / "run_shared_q07_multiseed.py",
        "features": PROJECT_DIR / "src" / "features.py",
        "temporal": PROJECT_DIR / "src" / "temporal.py",
        "metric": PROJECT_DIR / "src" / "metric.py",
        "manifest": PROJECT_DIR / "src" / "manifest.py",
        "preregister": preregister_path.resolve(),
    }
    return {name: describe_file(path) for name, path in paths.items()}


def _stage1_input_snapshot(
    raw_dir: Path, artifact_root: Path, preregister_path: Path
) -> dict[str, Any]:
    paths = {
        "info": raw_dir / "info.xlsx",
        "stage1_g12": artifact_root / "oof" / "dev2023_locked_v3.parquet",
        "stage1_g3_components": artifact_root
        / "oof"
        / "g3dev2023h2_candidates.parquet",
        "stage1_exact_reference": artifact_root
        / "postgate"
        / "shared_q07_multiseed_strict"
        / "oof"
        / "stage1_corrected_v3_baseline.parquet",
        "preregister": preregister_path,
    }
    return {name: describe_file(path) for name, path in paths.items()}


def _assert_snapshot_equal(
    before: Mapping[str, Any], after: Mapping[str, Any], *, name: str
) -> None:
    if before != after:
        raise AssertionError(f"{name} changed during Stage 1")


def _normalise_prediction_index(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output.index = pd.DatetimeIndex(output.index, name="forecast_kst_dtm")
    if not output.index.is_unique or not output.index.is_monotonic_increasing:
        raise AssertionError("prediction index must be unique and sorted")
    return output


def _read_stage1_baseline(
    artifact_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    g12_path = artifact_root / "oof" / "dev2023_locked_v3.parquet"
    g3_path = artifact_root / "oof" / "g3dev2023h2_candidates.parquet"
    reference_path = (
        artifact_root
        / "postgate"
        / "shared_q07_multiseed_strict"
        / "oof"
        / "stage1_corrected_v3_baseline.parquet"
    )
    g12 = _normalise_prediction_index(pd.read_parquet(g12_path, engine="pyarrow"))
    g3 = _normalise_prediction_index(pd.read_parquet(g3_path, engine="pyarrow"))
    reference = _normalise_prediction_index(
        pd.read_parquet(reference_path, engine="pyarrow")
    )
    expected_index = pd.date_range(
        STAGE1_START, STAGE1_END, freq="h", name="forecast_kst_dtm"
    )
    if not g12.index.equals(expected_index) or not reference.index.equals(expected_index):
        raise AssertionError("Stage 1 baseline index changed")
    if not g3.index.equals(expected_index[expected_index >= G3_STAGE1_START]):
        raise AssertionError("Stage 1 group-3 component index changed")
    baseline = pd.DataFrame(index=expected_index, columns=list(TARGET_COLS), dtype=float)
    baseline.loc[:, TARGET_COLS[:2]] = g12.loc[:, TARGET_COLS[:2]].to_numpy()
    raw = (
        0.2 * g3["q07"]
        + 0.075 * g3["shared_l1"]
        + 0.425 * g3["shared_q07"]
        + 0.025 * g3["top200q07"]
        + 0.275 * g3["ewq06"]
    )
    baseline.loc[g3.index, "kpx_group_3"] = np.clip(
        1.25 * raw.to_numpy(dtype=float) - 1200.0, 0.0, 21_420.0
    )
    finite = np.isfinite(reference.to_numpy(dtype=float))
    difference = np.abs(
        baseline.to_numpy(dtype=float)[finite] - reference.to_numpy(dtype=float)[finite]
    )
    maximum = float(difference.max(initial=0.0))
    if maximum != 0.0:
        raise AssertionError(f"Stage 1 baseline reconstruction differs by {maximum}")
    if not baseline.isna().equals(reference.isna()):
        raise AssertionError("Stage 1 baseline missingness differs from exact reference")
    return baseline, {
        "maximum_absolute_difference_kwh": maximum,
        "reference": describe_file(reference_path),
        "g12_source": describe_file(g12_path),
        "g3_source": describe_file(g3_path),
        "frame_sha256": _frame_sha256(baseline),
    }


def _component_paths(
    artifact_root: Path, mapping: Mapping[str, str]
) -> dict[str, Path]:
    return {name: artifact_root / relative for name, relative in mapping.items()}


def _reconstruct_baseline(
    *,
    artifact_root: Path,
    recipe_path: Path,
    component_files: Mapping[str, str],
    reference_path: Path,
    expected_index: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    components: dict[str, pd.DataFrame] = {}
    paths = _component_paths(artifact_root, component_files)
    for name in COMPONENTS:
        components[name] = _read_prediction(
            paths[name], expected_index=expected_index, required_groups=TARGET_COLS
        )
    reconstructed = pd.DataFrame(index=expected_index)
    for group in TARGET_COLS:
        reconstructed[group] = _assemble_group(
            recipe,
            group,
            {name: components[name][group] for name in COMPONENTS},
        )
    reference = _read_prediction(
        reference_path, expected_index=expected_index, required_groups=TARGET_COLS
    )
    maximum = float(
        np.max(
            np.abs(
                reconstructed.to_numpy(dtype=float)
                - reference.to_numpy(dtype=float)
            )
        )
    )
    if maximum != 0.0:
        raise AssertionError(f"corrected-v3 reconstruction differs by {maximum}")
    return reconstructed, {
        "maximum_absolute_difference_kwh": maximum,
        "reference": describe_file(reference_path),
        "components": {name: describe_file(path) for name, path in paths.items()},
        "recipe": describe_file(recipe_path),
        "frame_sha256": _frame_sha256(reconstructed),
    }


def _core_weather(
    features: Mapping[str, pd.DataFrame], weather_columns: Sequence[str]
) -> dict[str, pd.DataFrame]:
    output: dict[str, pd.DataFrame] = {}
    for group in TARGET_COLS:
        frame = features[group]
        missing = set(weather_columns).difference(frame.columns)
        if missing:
            raise AssertionError(f"{group} missing run weather columns {sorted(missing)}")
        selected = frame.loc[:, list(weather_columns)].copy()
        selected.index = pd.DatetimeIndex(selected.index, name="forecast_kst_dtm")
        if not np.isfinite(selected.to_numpy(dtype=float)).all():
            raise AssertionError(f"{group} core weather contains non-finite values")
        output[group] = selected
    return output


def _run_features(
    weather: pd.DataFrame,
    baseline: pd.Series,
    group: str,
    preregister: Mapping[str, Any],
) -> pd.DataFrame:
    available = baseline.dropna().index
    return build_run_sequence_features(
        weather.loc[available],
        baseline.loc[available],
        capacity_kwh=CAPACITY_KWH[group],
        weather_columns=preregister["sequence_features"]["weather_columns"],
        expected_hours=int(preregister["forecast_run_contract"]["hours_per_run"]),
        expected_max_features=int(
            preregister["sequence_features"]["expected_max_feature_count"]
        ),
    )


def _interval(
    index: pd.DatetimeIndex, bounds: Sequence[str]
) -> pd.DatetimeIndex:
    start, end = map(pd.Timestamp, bounds)
    selected = index[(index >= start) & (index <= end)]
    if len(selected) == 0:
        raise AssertionError(f"empty registered interval {start}..{end}")
    return selected


def _score_group(
    actual: pd.Series, prediction: pd.Series, group: str
) -> dict[str, Any]:
    if not actual.index.equals(prediction.index):
        raise AssertionError("score index mismatch")
    result = group_metrics(
        actual, prediction, CAPACITY_KWH[group], group_name=group
    )
    return {
        "score": float(0.5 * (result.one_minus_nmae + result.ficr)),
        "one_minus_nmae": float(result.one_minus_nmae),
        "ficr": float(result.ficr),
        "n_evaluated": int(result.n_evaluated),
    }


def _comparisons(
    *,
    actual: pd.Series,
    baseline: pd.Series,
    candidate: pd.Series,
    group: str,
    slices: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, bounds in slices.items():
        index = _interval(candidate.index, bounds)
        before = _score_group(actual.loc[index], baseline.loc[index], group)
        after = _score_group(actual.loc[index], candidate.loc[index], group)
        output[name] = {
            "baseline": before,
            "candidate": after,
            "delta": float(after["score"] - before["score"]),
        }
    return output


def _fit_one(
    *,
    specification: Mapping[str, Any],
    group: str,
    run_features: pd.DataFrame,
    actual: pd.Series,
    baseline: pd.Series,
    fit_bounds: Sequence[str],
    prediction_index: pd.DatetimeIndex,
) -> tuple[Any, pd.Series, pd.Series, dict[str, Any]]:
    fit_index = _interval(run_features.index, fit_bounds)
    assert_strict_fit_predict_order(fit_index, prediction_index)
    capacity = CAPACITY_KWH[group]
    target_cf = actual.loc[fit_index] / capacity
    eligible = target_cf.notna() & np.isfinite(target_cf) & (target_cf >= 0.1)
    eligible_index = fit_index[eligible.to_numpy()]
    residual = (
        actual.loc[eligible_index] / capacity
        - baseline.loc[eligible_index] / capacity
    )
    model = fit_residual_model(
        specification, run_features.loc[eligible_index], residual
    )
    prediction, correction = apply_residual_model(
        model,
        specification,
        run_features.loc[prediction_index],
        baseline.loc[prediction_index],
        capacity_kwh=capacity,
    )
    return model, prediction, correction, {
        "fit_rows_total": int(len(fit_index)),
        "fit_rows_eligible": int(len(eligible_index)),
        "fit_start": fit_index.min().isoformat(),
        "fit_end": fit_index.max().isoformat(),
        "prediction_start": prediction_index.min().isoformat(),
        "prediction_end": prediction_index.max().isoformat(),
        "actual_feature_columns": 0,
        "scada_feature_columns": 0,
    }


def _candidate_specification(
    preregister: Mapping[str, Any], candidate_id: str
) -> Mapping[str, Any]:
    matches = [item for item in preregister["candidates"] if item["id"] == candidate_id]
    if len(matches) != 1:
        raise AssertionError(f"locked candidate {candidate_id} missing or duplicated")
    return matches[0]


def _stage1(
    *,
    raw_dir: Path,
    cache_dir: Path,
    artifact_root: Path,
    out_dir: Path,
    preregister_path: Path,
    preregister: Mapping[str, Any],
) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"Stage 1 requires a new output directory: {out_dir}")
    out_dir.mkdir(parents=True)
    shutil.copyfile(preregister_path, out_dir / "preregister.json")
    source_before = _source_snapshot(preregister_path)
    inputs_before = _stage1_input_snapshot(raw_dir, artifact_root, preregister_path)

    labels = _read_labels(
        raw_dir / "train" / "train_labels.csv",
        nrows=EXPECTED_ROWS_PRE2024,
        expected_end=YEAR_2023_END,
        prefix_bytes=OFFICIAL_STAGE1_PREFIX_BYTES["labels"],
        expected_prefix_sha256=OFFICIAL_LABEL_PREFIX_SHA256,
    )
    print("Stage 1: building strict pre-2024 weather features", flush=True)
    raw_features, raw_contract = _read_stage1_raw_features(raw_dir, labels)
    weather = _core_weather(
        raw_features, preregister["sequence_features"]["weather_columns"]
    )
    del raw_features
    baseline, baseline_audit = _read_stage1_baseline(artifact_root)
    run_features = {
        group: _run_features(weather[group], baseline[group], group, preregister)
        for group in TARGET_COLS
    }
    feature_schema = tuple(run_features[TARGET_COLS[0]].columns)
    if any(tuple(run_features[group].columns) != feature_schema for group in TARGET_COLS):
        raise AssertionError("run feature schema differs by group")
    if len(feature_schema) > int(
        preregister["sequence_features"]["expected_max_feature_count"]
    ):
        raise AssertionError("compact run feature cap exceeded")

    all_comparisons: dict[str, Any] = {}
    training: dict[str, Any] = {}
    candidate_outputs: dict[str, pd.DataFrame] = {}
    for specification in preregister["candidates"]:
        candidate_id = str(specification["id"])
        print(f"Stage 1: fitting {candidate_id}", flush=True)
        frame = pd.DataFrame(index=baseline.index, columns=list(TARGET_COLS), dtype=float)
        by_group: dict[str, Any] = {}
        training[candidate_id] = {}
        for group in TARGET_COLS:
            slices = preregister["stage1"]["selection_slices"][group]
            prediction_bounds = slices["full"]
            prediction_index = _interval(run_features[group].index, prediction_bounds)
            model, prediction, correction, evidence = _fit_one(
                specification=specification,
                group=group,
                run_features=run_features[group],
                actual=labels[group],
                baseline=baseline[group],
                fit_bounds=preregister["stage1"]["fit_intervals"][group],
                prediction_index=prediction_index,
            )
            frame.loc[prediction.index, group] = prediction
            model_path = out_dir / "models" / f"stage1__{candidate_id}__{group}.joblib"
            _atomic_joblib(
                {
                    "model": model,
                    "feature_columns": list(feature_schema),
                    "candidate": specification,
                    "group": group,
                },
                model_path,
            )
            evidence.update(
                {
                    "model": describe_file(model_path),
                    "correction_cf_min": float(correction.min()),
                    "correction_cf_max": float(correction.max()),
                    "correction_cf_mean": float(correction.mean()),
                }
            )
            training[candidate_id][group] = evidence
            by_group[group] = _comparisons(
                actual=labels[group],
                baseline=baseline[group],
                candidate=prediction,
                group=group,
                slices=slices,
            )
        candidate_path = out_dir / "oof" / f"stage1__{candidate_id}.parquet"
        _atomic_parquet(frame, candidate_path)
        candidate_outputs[candidate_id] = frame
        all_comparisons[candidate_id] = by_group

    baseline_path = out_dir / "oof" / "stage1_corrected_v3_baseline.parquet"
    _atomic_parquet(baseline, baseline_path)
    selected = choose_single_candidate(all_comparisons)
    source_after = _source_snapshot(preregister_path)
    inputs_after = _stage1_input_snapshot(raw_dir, artifact_root, preregister_path)
    _assert_snapshot_equal(source_before, source_after, name="Stage 1 source snapshot")
    _assert_snapshot_equal(inputs_before, inputs_after, name="Stage 1 input snapshot")
    result = {
        "schema_version": 1,
        "experiment_id": "run_sequence_residual_strict_forward_stage1",
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "candidate_count": len(preregister["candidates"]),
        "selected_candidate": selected,
        "selection_changes_after_2024": False,
        "2024_labels_weather_or_predictions_read": False,
        "bounded_label_rows": int(len(labels)),
        "raw_feature_contract": raw_contract,
        "cache_dir_deliberately_not_read": str(cache_dir.resolve()),
        "baseline_exact_reconstruction": baseline_audit,
        "run_feature_contract": {
            "columns": int(len(feature_schema)),
            "column_names": list(feature_schema),
            "raw_24_by_612_flattened": False,
            "same_run_hours": 24,
            "cross_run_features": 0,
            "actual_target_scada_features": 0,
            "by_group": {
                group: {
                    "rows": int(len(run_features[group])),
                    "start": run_features[group].index.min().isoformat(),
                    "end": run_features[group].index.max().isoformat(),
                    "frame_sha256": _frame_sha256(run_features[group]),
                }
                for group in TARGET_COLS
            },
        },
        "comparisons": all_comparisons,
        "training": training,
        "source_snapshot_before": source_before,
        "source_snapshot_after": source_after,
        "source_snapshot_sha256": _mapping_sha256(source_before),
        "input_snapshot_before": inputs_before,
        "input_snapshot_after": inputs_after,
        "input_snapshot_sha256": _mapping_sha256(inputs_before),
        "snapshots_unchanged": True,
        "leaderboard_score_claim": False,
    }
    result_path = out_dir / "stage1_results.json"
    _write_json(result_path, result)
    lock = {
        "schema_version": 1,
        "experiment_id": "run_sequence_residual_pre2024_lock",
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "stage1_results_sha256": sha256_file(result_path),
        "source_snapshot_sha256": result["source_snapshot_sha256"],
        "input_snapshot_sha256": result["input_snapshot_sha256"],
        "selected_candidate": selected,
        "selection_changes_after_2024": False,
        "2024_read_before_lock": False,
    }
    _write_json(out_dir / "pre2024_lock.json", lock)
    print("Stage 1 locked candidate:", selected, flush=True)
    return result


def _load_lock(
    out_dir: Path,
    preregister_path: Path,
    raw_dir: Path,
    artifact_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result_path = out_dir / "stage1_results.json"
    lock_path = out_dir / "pre2024_lock.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["preregister_sha256"] != PREREGISTER_SHA256:
        raise AssertionError("pre-2024 lock preregister SHA changed")
    if lock["stage1_results_sha256"] != sha256_file(result_path):
        raise AssertionError("Stage 1 results changed after lock")
    if lock["selected_candidate"] != result["selected_candidate"]:
        raise AssertionError("locked candidate differs from Stage 1 selection")
    if lock["selection_changes_after_2024"] is not False:
        raise AssertionError("lock permits selection changes after 2024")
    current_source = _source_snapshot(preregister_path)
    current_input = _stage1_input_snapshot(raw_dir, artifact_root, preregister_path)
    if _mapping_sha256(current_source) != lock["source_snapshot_sha256"]:
        raise AssertionError("source snapshot changed after pre-2024 lock")
    if _mapping_sha256(current_input) != lock["input_snapshot_sha256"]:
        raise AssertionError("Stage 1 input snapshot changed after lock")
    recalculated = choose_single_candidate(result["comparisons"])
    if recalculated != lock["selected_candidate"]:
        raise AssertionError("locked candidate violates registered selection rule")
    return lock, result


def _stage2_no_candidate(out_dir: Path, lock: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "locked_candidate": None,
        "2024_read": False,
        "promoted": False,
        "reason": "no single recipe improved every Stage 1 group/slice",
        "pre2024_lock_sha256": sha256_file(out_dir / "pre2024_lock.json"),
        "selection_changes_after_2024": False,
        "final_fit_performed": False,
        "submission_created": False,
        "leaderboard_score_claim": False,
    }
    _write_json(out_dir / "stage2_results.json", result)
    _write_json(
        out_dir / "promotion_lock.json",
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "promoted": False,
            "locked_candidate": None,
            "2024_read": False,
            "stage2_results_sha256": sha256_file(out_dir / "stage2_results.json"),
            "pre2024_lock_sha256": sha256_file(out_dir / "pre2024_lock.json"),
        },
    )
    return result


def _stage2(
    *,
    raw_dir: Path,
    cache_dir: Path,
    artifact_root: Path,
    recipe_path: Path,
    out_dir: Path,
    preregister_path: Path,
    preregister: Mapping[str, Any],
) -> dict[str, Any]:
    if (out_dir / "stage2_results.json").exists():
        raise FileExistsError("Stage 2 result already exists")
    lock, stage1_result = _load_lock(
        out_dir, preregister_path, raw_dir, artifact_root
    )
    selected = lock["selected_candidate"]
    if selected is None:
        return _stage2_no_candidate(out_dir, lock)

    print(f"Stage 2: fixed 2024 confirmation for {selected}", flush=True)
    labels = _read_labels(
        raw_dir / "train" / "train_labels.csv",
        nrows=EXPECTED_ROWS_THROUGH2024,
        expected_end=YEAR_2024_END,
    )
    all_weather = _read_features(cache_dir, labels, expected_end=YEAR_2024_END)
    weather = _core_weather(
        all_weather, preregister["sequence_features"]["weather_columns"]
    )
    del all_weather
    stage1_baseline, stage1_baseline_audit = _read_stage1_baseline(artifact_root)
    gate_index = pd.date_range(
        STAGE2_START, STAGE2_END, freq="h", name="forecast_kst_dtm"
    )
    gate_baseline, gate_baseline_audit = _reconstruct_baseline(
        artifact_root=artifact_root,
        recipe_path=recipe_path,
        component_files=GATE_COMPONENT_FILES,
        reference_path=artifact_root / "oof" / "gate2024_locked_v3_cf_fix.parquet",
        expected_index=gate_index,
    )
    specification = _candidate_specification(preregister, selected)
    candidate = pd.DataFrame(index=gate_index, columns=list(TARGET_COLS), dtype=float)
    training: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    for group in TARGET_COLS:
        history_index = stage1_baseline[group].dropna().index
        history_features = _run_features(
            weather[group].loc[stage1_baseline.index],
            stage1_baseline[group],
            group,
            preregister,
        )
        gate_features = _run_features(
            weather[group].loc[gate_index], gate_baseline[group], group, preregister
        )
        assert_strict_fit_predict_order(history_index, gate_index)
        capacity = CAPACITY_KWH[group]
        target_cf = labels.loc[history_index, group] / capacity
        eligible = target_cf.notna() & np.isfinite(target_cf) & (target_cf >= 0.1)
        fit_index = history_index[eligible.to_numpy()]
        residual = (
            labels.loc[fit_index, group] / capacity
            - stage1_baseline.loc[fit_index, group] / capacity
        )
        model = fit_residual_model(
            specification, history_features.loc[fit_index], residual
        )
        prediction, correction = apply_residual_model(
            model,
            specification,
            gate_features,
            gate_baseline[group],
            capacity_kwh=capacity,
        )
        candidate[group] = prediction
        model_path = out_dir / "models" / f"stage2__{selected}__{group}.joblib"
        _atomic_joblib(
            {
                "model": model,
                "feature_columns": list(gate_features.columns),
                "candidate": specification,
                "group": group,
            },
            model_path,
        )
        training[group] = {
            "fit_rows": int(len(fit_index)),
            "fit_start": fit_index.min().isoformat(),
            "fit_end": fit_index.max().isoformat(),
            "prediction_start": gate_index.min().isoformat(),
            "prediction_end": gate_index.max().isoformat(),
            "correction_cf_min": float(correction.min()),
            "correction_cf_max": float(correction.max()),
            "correction_cf_mean": float(correction.mean()),
            "actual_target_scada_features": 0,
        }
        comparisons[group] = _comparisons(
            actual=labels.loc[gate_index, group],
            baseline=gate_baseline[group],
            candidate=prediction,
            group=group,
            slices=preregister["stage2"]["confirmation_slices"],
        )
    promoted = all(
        float(item["delta"]) > 0.0
        for slices in comparisons.values()
        for item in slices.values()
    )
    baseline_path = out_dir / "oof" / "stage2_corrected_v3_baseline.parquet"
    candidate_path = out_dir / "oof" / f"stage2__{selected}.parquet"
    _atomic_parquet(gate_baseline, baseline_path)
    _atomic_parquet(candidate, candidate_path)
    result = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "locked_candidate": selected,
        "2024_read": True,
        "selection_changes_after_2024": False,
        "comparisons": comparisons,
        "promoted": bool(promoted),
        "training": training,
        "stage1_baseline_exact_reconstruction": stage1_baseline_audit,
        "stage2_baseline_exact_reconstruction": gate_baseline_audit,
        "pre2024_lock_sha256": sha256_file(out_dir / "pre2024_lock.json"),
        "final_fit_performed": False,
        "submission_created": False,
        "leaderboard_score_claim": False,
    }
    result_path = out_dir / "stage2_results.json"
    _write_json(result_path, result)
    _write_json(
        out_dir / "promotion_lock.json",
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "promoted": bool(promoted),
            "locked_candidate": selected,
            "selection_changes_after_2024": False,
            "stage2_results_sha256": sha256_file(result_path),
            "pre2024_lock_sha256": sha256_file(out_dir / "pre2024_lock.json"),
        },
    )
    if promoted:
        _final(
            raw_dir=raw_dir,
            cache_dir=cache_dir,
            artifact_root=artifact_root,
            recipe_path=recipe_path,
            out_dir=out_dir,
            preregister=preregister,
            specification=specification,
            labels=labels,
            weather=weather,
            stage1_baseline=stage1_baseline,
            gate_baseline=gate_baseline,
            gate_index=gate_index,
        )
        result["final_fit_performed"] = True
        result["submission_created"] = True
    return result


def _write_submission(
    sample_path: Path, predictions: pd.DataFrame, destination: Path
) -> dict[str, Any]:
    sample = pd.read_csv(
        sample_path,
        encoding="utf-8-sig",
        dtype={"forecast_id": "string", "forecast_kst_dtm": "string"},
    )
    expected_index = pd.DatetimeIndex(
        pd.to_datetime(sample["forecast_kst_dtm"], errors="raise"),
        name="forecast_kst_dtm",
    )
    if not predictions.index.equals(expected_index):
        raise AssertionError("final prediction index differs from sample")
    output = sample.copy()
    for group in TARGET_COLS:
        output[group] = predictions[group].to_numpy(dtype=float)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        output.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    raw = destination.read_bytes()
    observed = pd.read_csv(
        destination,
        encoding="utf-8-sig",
        dtype={"forecast_id": "string", "forecast_kst_dtm": "string"},
    )
    if len(observed) != EXPECTED_TEST_ROWS or list(observed.columns) != list(sample.columns):
        raise AssertionError("submission schema or row count changed")
    for column in ("forecast_id", "forecast_kst_dtm"):
        if not observed[column].equals(sample[column]):
            raise AssertionError(f"submission {column} differs from sample")
    if not raw.startswith(b"\xef\xbb\xbf"):
        raise AssertionError("submission is missing UTF-8-SIG BOM")
    values = observed.loc[:, list(TARGET_COLS)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise AssertionError("submission contains non-finite values")
    for group in TARGET_COLS:
        if not observed[group].between(0.0, 1.02 * CAPACITY_KWH[group]).all():
            raise AssertionError(f"submission {group} exceeds clip")
    return {
        **describe_file(destination),
        "rows": int(len(observed)),
        "columns_exact": True,
        "id_time_exact": True,
        "utf8_sig": True,
        "finite": True,
        "capacity_clip": True,
    }


def _final(
    *,
    raw_dir: Path,
    cache_dir: Path,
    artifact_root: Path,
    recipe_path: Path,
    out_dir: Path,
    preregister: Mapping[str, Any],
    specification: Mapping[str, Any],
    labels: pd.DataFrame,
    weather: Mapping[str, pd.DataFrame],
    stage1_baseline: pd.DataFrame,
    gate_baseline: pd.DataFrame,
    gate_index: pd.DatetimeIndex,
) -> None:
    promotion_lock = json.loads(
        (out_dir / "promotion_lock.json").read_text(encoding="utf-8")
    )
    if promotion_lock.get("promoted") is not True:
        raise AssertionError("final fit requires a positive immutable promotion lock")
    sample_path = raw_dir / "sample_submission.csv"
    sample = pd.read_csv(sample_path, encoding="utf-8-sig")
    test_index = pd.DatetimeIndex(
        pd.to_datetime(sample["forecast_kst_dtm"], errors="raise"),
        name="forecast_kst_dtm",
    )
    if len(test_index) != EXPECTED_TEST_ROWS or test_index.min() != FINAL_START or test_index.max() != FINAL_END:
        raise AssertionError("sample test interval changed")
    test_weather_full = _read_test_features(cache_dir, test_index)
    test_weather = _core_weather(
        test_weather_full, preregister["sequence_features"]["weather_columns"]
    )
    final_baseline, final_baseline_audit = _reconstruct_baseline(
        artifact_root=artifact_root,
        recipe_path=recipe_path,
        component_files=FINAL_COMPONENT_FILES,
        reference_path=artifact_root
        / "final_cf_fix"
        / "predictions"
        / "corrected_v3_test.parquet",
        expected_index=test_index,
    )
    predictions = pd.DataFrame(index=test_index, columns=list(TARGET_COLS), dtype=float)
    training: dict[str, Any] = {}
    for group in TARGET_COLS:
        historical_baseline = pd.concat(
            [stage1_baseline[group].dropna(), gate_baseline[group]], axis=0
        )
        historical_weather = pd.concat(
            [
                weather[group].loc[stage1_baseline[group].dropna().index],
                weather[group].loc[gate_index],
            ],
            axis=0,
        )
        historical_features = build_run_sequence_features(
            historical_weather,
            historical_baseline,
            capacity_kwh=CAPACITY_KWH[group],
            weather_columns=preregister["sequence_features"]["weather_columns"],
            expected_max_features=int(
                preregister["sequence_features"]["expected_max_feature_count"]
            ),
        )
        test_features = _run_features(
            test_weather[group], final_baseline[group], group, preregister
        )
        assert_strict_fit_predict_order(historical_features.index, test_index)
        capacity = CAPACITY_KWH[group]
        target_cf = labels.loc[historical_features.index, group] / capacity
        eligible = target_cf.notna() & np.isfinite(target_cf) & (target_cf >= 0.1)
        fit_index = historical_features.index[eligible.to_numpy()]
        residual = (
            labels.loc[fit_index, group] / capacity
            - historical_baseline.loc[fit_index] / capacity
        )
        model = fit_residual_model(
            specification, historical_features.loc[fit_index], residual
        )
        prediction, correction = apply_residual_model(
            model,
            specification,
            test_features,
            final_baseline[group],
            capacity_kwh=capacity,
        )
        predictions[group] = prediction
        model_path = out_dir / "models" / f"final__{specification['id']}__{group}.joblib"
        _atomic_joblib(
            {
                "model": model,
                "feature_columns": list(test_features.columns),
                "candidate": specification,
                "group": group,
            },
            model_path,
        )
        training[group] = {
            "fit_rows": int(len(fit_index)),
            "fit_start": fit_index.min().isoformat(),
            "fit_end": fit_index.max().isoformat(),
            "prediction_start": test_index.min().isoformat(),
            "prediction_end": test_index.max().isoformat(),
            "correction_cf_min": float(correction.min()),
            "correction_cf_max": float(correction.max()),
            "correction_cf_mean": float(correction.mean()),
        }
    prediction_path = out_dir / "predictions" / "run_sequence_residual_2025.parquet"
    _atomic_parquet(predictions, prediction_path)
    submission_path = out_dir / preregister["final"]["submission_name"]
    submission_audit = _write_submission(sample_path, predictions, submission_path)
    _write_json(
        out_dir / "final_results.json",
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "locked_candidate": specification["id"],
            "training": training,
            "baseline_exact_reconstruction": final_baseline_audit,
            "prediction": describe_file(prediction_path),
            "submission": submission_audit,
            "leaderboard_score_claim": False,
        },
    )


def _write_manifest(
    *,
    args: argparse.Namespace,
    preregister: Mapping[str, Any],
    stage2_result: Mapping[str, Any] | None,
    destination_name: str,
) -> None:
    destination = args.out_dir / destination_name
    outputs = sorted(
        path
        for path in args.out_dir.rglob("*")
        if path.is_file() and path != destination
    )
    safe_inputs = [
        args.preregister,
        args.recipe,
        args.raw_dir / "info.xlsx",
        args.artifact_root / "oof" / "dev2023_locked_v3.parquet",
        args.artifact_root / "oof" / "g3dev2023h2_candidates.parquet",
        args.artifact_root
        / "postgate"
        / "shared_q07_multiseed_strict"
        / "oof"
        / "stage1_corrected_v3_baseline.parquet",
        Path(__file__).resolve(),
        PROJECT_DIR / "src" / "run_sequence_residual.py",
        PROJECT_DIR / "tests" / "test_run_sequence_residual.py",
    ]
    payload = {
        "schema_version": 1,
        "artifact_type": "run_sequence_residual_strict_forward",
        "created_utc": utc_now(),
        "command": [sys.executable, *sys.argv],
        "runtime": {
            "packages": package_versions(),
            "git": git_state(PROJECT_DIR),
        },
        "preregister_sha256": PREREGISTER_SHA256,
        "candidate_count": len(preregister["candidates"]),
        "inputs": [describe_file(path) for path in safe_inputs],
        "raw_prefix_inputs": json.loads(
            (args.out_dir / "stage1_results.json").read_text(encoding="utf-8")
        )["raw_feature_contract"],
        "outputs": [describe_file(path) for path in outputs],
        "stage2": dict(stage2_result or {}),
        "leaderboard_score_claim": False,
    }
    _write_json(destination, payload)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.raw_dir = args.raw_dir.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.artifact_root = args.artifact_root.expanduser().resolve()
    args.recipe = args.recipe.expanduser().resolve()
    args.preregister = args.preregister.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    preregister = _load_preregister(args.preregister)
    if args.stage in {"stage1", "all"}:
        _stage1(
            raw_dir=args.raw_dir,
            cache_dir=args.cache_dir,
            artifact_root=args.artifact_root,
            out_dir=args.out_dir,
            preregister_path=args.preregister,
            preregister=preregister,
        )
        if args.stage == "stage1":
            _write_manifest(
                args=args,
                preregister=preregister,
                stage2_result=None,
                destination_name="stage1_manifest.json",
            )
            return
    if args.stage in {"stage2", "all"}:
        stage2_result = _stage2(
            raw_dir=args.raw_dir,
            cache_dir=args.cache_dir,
            artifact_root=args.artifact_root,
            recipe_path=args.recipe,
            out_dir=args.out_dir,
            preregister_path=args.preregister,
            preregister=preregister,
        )
        _write_manifest(
            args=args,
            preregister=preregister,
            stage2_result=stage2_result,
            destination_name="manifest.json",
        )


if __name__ == "__main__":
    main()

"""Strict-forward residual-histogram multiclass Bayes experiment.

No candidate is scored until its predictions have been written and read back
from the canonical OOF artifact.  Stage 1 is physically bounded to pre-2024
label/weather bytes and pre-2024 OOF predictions.  A content-addressed lock is
verified before 2024 confirmation, and 2025 is inaccessible unless that fixed
confirmation promotes the whole mixed recipe.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.run_run_sequence_residual import (  # noqa: E402
    _atomic_joblib,
    _atomic_parquet,
    _core_weather,
    _frame_sha256,
    _interval,
    _mapping_sha256,
    _read_stage1_baseline,
    _reconstruct_baseline,
    _write_json,
)
from scripts.run_shared_q07_multiseed import (  # noqa: E402
    EXPECTED_ROWS_PRE2024,
    EXPECTED_ROWS_THROUGH2024,
    EXPECTED_TEST_ROWS,
    OFFICIAL_LABEL_PREFIX_SHA256,
    OFFICIAL_STAGE1_PREFIX_BYTES,
    YEAR_2023_END,
    YEAR_2024_END,
    _read_features,
    _read_labels,
    _read_stage1_raw_features,
    _read_test_features,
)
from src.manifest import (  # noqa: E402
    describe_file,
    git_state,
    package_versions,
    sha256_file,
    utc_now,
)
from src.metric import CAPACITY_KWH, TARGET_COLS, group_metrics, score_details  # noqa: E402
from src.residual_histogram_bayes import (  # noqa: E402
    ACTION_DELTAS_CF,
    COMPONENT_NAMES,
    META_FEATURE_COLUMNS,
    RESIDUAL_CENTRES_CF,
    ResidualHistogramBayesClassifier,
    WEATHER_COLUMNS,
    apply_action_shrink,
    build_meta_features,
    choose_group_shrinks,
)
from src.run_sequence_residual import assert_strict_fit_predict_order  # noqa: E402


PREREGISTER_SHA256 = (
    "b92eff04b33fac9f049afda6f8dd62cbd26439f3d2c9b67b4df84a4bd4144564"
)
YEAR_2023_START = pd.Timestamp("2023-01-01 01:00:00")
YEAR_2023_FINISH = pd.Timestamp("2024-01-01 00:00:00")
G3_2023_START = pd.Timestamp("2023-07-01 01:00:00")
YEAR_2024_START = pd.Timestamp("2024-01-01 01:00:00")
YEAR_2024_FINISH = pd.Timestamp("2025-01-01 00:00:00")
YEAR_2025_START = pd.Timestamp("2025-01-01 01:00:00")
YEAR_2025_FINISH = pd.Timestamp("2026-01-01 00:00:00")

DEV_COMPONENT_FILES = {
    "lgb_l1": "oof/dev2023_lgb_l1_eligible_n1500.parquet",
    "lgb_q07": "oof/dev2023_lgb_q07_eligible.parquet",
    "shared_l1": "oof/dev2023_shared_l1_eligible.parquet",
    "shared_q07": "oof/dev2023_shared_q07_eligible.parquet",
    "top200_q07": "oof/dev2023_lgb_top200_q07_eligible.parquet",
    "energy_q06": "oof/dev2023_lgb_q06_energywt_eligible.parquet",
}
G3_DEV_FILE = "oof/g3dev2023h2_candidates.parquet"
G3_COLUMN_MAP = {
    "lgb_l1": "l1",
    "lgb_q07": "q07",
    "shared_l1": "shared_l1",
    "shared_q07": "shared_q07",
    "top200_q07": "top200q07",
    "energy_q06": "ewq06",
}
GATE_COMPONENT_FILES = {
    "lgb_l1": "gate/v3/predictions/lgb_l1_gate.parquet",
    "lgb_q07": "gate/v3/predictions/lgb_q07_gate.parquet",
    "shared_l1": "oof/gate2024_shared_l1_cf.parquet",
    "shared_q07": "oof/gate2024_shared_q07_cf.parquet",
    "top200_q07": "gate/v3/predictions/top200_q07_gate.parquet",
    "energy_q06": "gate/v3/predictions/energy_q06_gate.parquet",
}
FINAL_COMPONENT_FILES = {
    "lgb_l1": "final_v3/predictions/v3_locked_full_2025__lgb_l1_test.parquet",
    "lgb_q07": "final_v3/predictions/v3_locked_full_2025__lgb_q07_test.parquet",
    "shared_l1": "final_cf_fix/predictions/shared_l1_cf_seed42_test.parquet",
    "shared_q07": "final_cf_fix/predictions/shared_q07_cf_seed42_test.parquet",
    "top200_q07": "final_v3/predictions/v3_locked_full_2025__top200_q07_test.parquet",
    "energy_q06": "final_v3/predictions/v3_locked_full_2025__energy_q06_test.parquet",
}


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
        default=PROJECT_DIR / "configs" / "residual_histogram_bayes_preregister.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_DIR
        / "artifacts"
        / "postgate"
        / "residual_histogram_bayes_strict",
    )
    parser.add_argument("--stage", choices=("stage1", "stage2", "all"), default="all")
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


def _load_preregister(path: Path) -> dict[str, Any]:
    observed = sha256_file(path)
    if observed != PREREGISTER_SHA256:
        raise AssertionError(f"residual histogram preregister SHA changed: {observed}")
    sidecar = path.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    if sidecar != PREREGISTER_SHA256:
        raise AssertionError("preregister SHA sidecar differs")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("experiment_id") != "residual_histogram_multiclass_bayes_strict_forward_v1":
        raise AssertionError("unexpected experiment id")
    if tuple(payload["meta_features"]["fixed_order"]) != META_FEATURE_COLUMNS:
        raise AssertionError("frozen meta feature order changed")
    if tuple(payload["component_predictions"]["fixed_order"]) != COMPONENT_NAMES:
        raise AssertionError("frozen component order changed")
    if tuple(payload["meta_features"]["weather_source_columns"]) != WEATHER_COLUMNS:
        raise AssertionError("frozen weather set changed")
    expected_centres = np.asarray(
        [-0.525 + 0.025 * index for index in range(43)], dtype=float
    )
    expected_actions = np.asarray(
        [-0.150 + 0.005 * index for index in range(61)], dtype=float
    )
    if not np.array_equal(RESIDUAL_CENTRES_CF, expected_centres):
        raise AssertionError("residual centre implementation changed")
    if not np.array_equal(ACTION_DELTAS_CF, expected_actions):
        raise AssertionError("action grid implementation changed")
    if payload["bayes_action"]["delta_shrink_candidates"] != [0.25, 0.5, 1.0]:
        raise AssertionError("frozen shrink grid changed")
    return payload


def _source_snapshot(preregister_path: Path) -> dict[str, Any]:
    paths = {
        "runner": Path(__file__).resolve(),
        "core": PROJECT_DIR / "src" / "residual_histogram_bayes.py",
        "test": PROJECT_DIR / "tests" / "test_residual_histogram_bayes.py",
        "metric": PROJECT_DIR / "src" / "metric.py",
        "features": PROJECT_DIR / "src" / "features.py",
        "raw_prefix_helper": PROJECT_DIR / "scripts" / "run_shared_q07_multiseed.py",
        "baseline_helper": PROJECT_DIR / "scripts" / "run_run_sequence_residual.py",
        "preregister": preregister_path,
        "preregister_sidecar": preregister_path.with_suffix(".sha256"),
    }
    return {name: describe_file(path) for name, path in paths.items()}


def _stage1_input_paths(
    raw_dir: Path, artifact_root: Path, recipe_path: Path, preregister_path: Path
) -> dict[str, Path]:
    paths = {
        "info": raw_dir / "info.xlsx",
        "locked_recipe": recipe_path,
        "baseline_group_1_and_2": artifact_root / "oof" / "dev2023_locked_v3.parquet",
        "exact_baseline_reference": artifact_root
        / "postgate"
        / "shared_q07_multiseed_strict"
        / "oof"
        / "stage1_corrected_v3_baseline.parquet",
        "group_3_components": artifact_root / G3_DEV_FILE,
        "preregister": preregister_path,
    }
    for name, relative in DEV_COMPONENT_FILES.items():
        paths[f"component_{name}"] = artifact_root / relative
    return paths


def _stage1_input_snapshot(
    raw_dir: Path, artifact_root: Path, recipe_path: Path, preregister_path: Path
) -> dict[str, Any]:
    return {
        name: describe_file(path)
        for name, path in _stage1_input_paths(
            raw_dir, artifact_root, recipe_path, preregister_path
        ).items()
    }


def _verify_stage1_expected_hashes(
    snapshot: Mapping[str, Mapping[str, Any]], preregister: Mapping[str, Any]
) -> None:
    expected = preregister["stage1_expected_sha256"]
    for name in (
        "locked_recipe",
        "baseline_group_1_and_2",
        "exact_baseline_reference",
        "group_3_components",
        "component_lgb_l1",
        "component_lgb_q07",
        "component_shared_l1",
        "component_shared_q07",
        "component_top200_q07",
        "component_energy_q06",
    ):
        if snapshot[name]["sha256"] != expected[name]:
            raise AssertionError(f"Stage 1 input hash changed for {name}")


def _read_prediction(
    path: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    rows: int,
    required_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    frame = pd.read_parquet(path, engine="pyarrow")
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if (
        len(frame) != rows
        or frame.index.min() != start
        or frame.index.max() != end
        or not frame.index.is_unique
        or not frame.index.is_monotonic_increasing
    ):
        raise ValueError(f"prediction coverage changed: {path}")
    if required_columns is not None and tuple(frame.columns) != tuple(required_columns):
        raise ValueError(f"prediction schema changed: {path}")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError(f"prediction contains non-finite values: {path}")
    return frame


def _assert_component_scale(
    components: Mapping[str, pd.Series], group: str
) -> None:
    if tuple(components) != COMPONENT_NAMES:
        raise AssertionError("component order differs")
    capacity = CAPACITY_KWH[group]
    reference: pd.Index | None = None
    for name, series in components.items():
        if reference is None:
            reference = series.index
        elif not reference.equals(series.index):
            raise ValueError("component indexes differ")
        values = series.to_numpy(dtype=float)
        if (
            not np.isfinite(values).all()
            or values.min() < 0.0
            or values.max() <= 2.0
            or values.max() > 1.02 * capacity + 1e-8
        ):
            raise ValueError(f"component target-scale/clip mismatch: {group}/{name}")


def _read_stage1_components(
    artifact_root: Path,
) -> dict[str, dict[str, pd.Series]]:
    g12_frames: dict[str, pd.DataFrame] = {}
    for name, relative in DEV_COMPONENT_FILES.items():
        g12_frames[name] = _read_prediction(
            artifact_root / relative,
            start=YEAR_2023_START,
            end=YEAR_2023_FINISH,
            rows=8760,
            required_columns=("kpx_group_1", "kpx_group_2"),
        )
    g3_frame = _read_prediction(
        artifact_root / G3_DEV_FILE,
        start=G3_2023_START,
        end=YEAR_2023_FINISH,
        rows=4416,
    )
    missing = set(G3_COLUMN_MAP.values()).difference(g3_frame.columns)
    if missing:
        raise ValueError(f"group-3 component columns missing: {sorted(missing)}")
    output: dict[str, dict[str, pd.Series]] = {
        group: {name: g12_frames[name][group] for name in COMPONENT_NAMES}
        for group in ("kpx_group_1", "kpx_group_2")
    }
    output["kpx_group_3"] = {
        name: g3_frame[column] for name, column in G3_COLUMN_MAP.items()
    }
    for group in TARGET_COLS:
        _assert_component_scale(output[group], group)
    return output


def _read_period_components(
    artifact_root: Path,
    mapping: Mapping[str, str],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    rows: int,
) -> dict[str, dict[str, pd.Series]]:
    frames = {
        name: _read_prediction(
            artifact_root / relative,
            start=start,
            end=end,
            rows=rows,
            required_columns=TARGET_COLS,
        )
        for name, relative in mapping.items()
    }
    output = {
        group: {name: frames[name][group] for name in COMPONENT_NAMES}
        for group in TARGET_COLS
    }
    for group in TARGET_COLS:
        _assert_component_scale(output[group], group)
    return output


def _metric_record(actual: pd.Series, prediction: pd.Series, group: str) -> dict[str, Any]:
    result = group_metrics(actual, prediction, CAPACITY_KWH[group], group_name=group)
    return {
        "score": float(0.5 * (result.one_minus_nmae + result.ficr)),
        "one_minus_nmae": float(result.one_minus_nmae),
        "ficr": float(result.ficr),
        "n_evaluated": int(result.n_evaluated),
    }


def _score_saved_group(
    *,
    labels: pd.DataFrame,
    baseline_path: Path,
    candidate_path: Path,
    group: str,
    segments: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    baseline = pd.read_parquet(baseline_path, engine="pyarrow")
    candidate = pd.read_parquet(candidate_path, engine="pyarrow")
    baseline.index = pd.DatetimeIndex(baseline.index, name="forecast_kst_dtm")
    candidate.index = pd.DatetimeIndex(candidate.index, name="forecast_kst_dtm")
    if not baseline.index.equals(candidate.index) or tuple(baseline.columns) != tuple(candidate.columns):
        raise AssertionError("stored OOF baseline/candidate schema differs")
    output: dict[str, Any] = {}
    for name, bounds in segments.items():
        index = _interval(candidate.index, bounds)
        before = _metric_record(labels.loc[index, group], baseline.loc[index, group], group)
        after = _metric_record(labels.loc[index, group], candidate.loc[index, group], group)
        output[name] = {
            "baseline": before,
            "candidate": after,
            "delta": float(after["score"] - before["score"]),
            "one_minus_nmae_delta": float(
                after["one_minus_nmae"] - before["one_minus_nmae"]
            ),
            "ficr_delta": float(after["ficr"] - before["ficr"]),
            "score_source": "reopened_stored_oof_official_group_metrics",
        }
    return output


def _aggregate_record(actual: pd.DataFrame, prediction: pd.DataFrame) -> dict[str, float]:
    result = score_details(actual, prediction)
    return {
        "total_score": float(result.total_score),
        "one_minus_nmae": float(result.one_minus_nmae),
        "ficr": float(result.ficr),
    }


def _score_saved_aggregate(
    *,
    labels: pd.DataFrame,
    baseline_path: Path,
    candidate_path: Path,
    segments: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    baseline = pd.read_parquet(baseline_path, engine="pyarrow")
    candidate = pd.read_parquet(candidate_path, engine="pyarrow")
    baseline.index = pd.DatetimeIndex(baseline.index, name="forecast_kst_dtm")
    candidate.index = pd.DatetimeIndex(candidate.index, name="forecast_kst_dtm")
    output: dict[str, Any] = {}
    for name, bounds in segments.items():
        index = _interval(candidate.index, bounds)
        before = _aggregate_record(labels.loc[index, list(TARGET_COLS)], baseline.loc[index])
        after = _aggregate_record(labels.loc[index, list(TARGET_COLS)], candidate.loc[index])
        output[name] = {
            "baseline": before,
            "candidate": after,
            "total_score_delta": float(after["total_score"] - before["total_score"]),
            "one_minus_nmae_delta": float(
                after["one_minus_nmae"] - before["one_minus_nmae"]
            ),
            "ficr_delta": float(after["ficr"] - before["ficr"]),
            "score_source": "reopened_stored_oof_official_score_details",
        }
    return output


def _stage1_segments(
    preregister: Mapping[str, Any], group: str
) -> Mapping[str, Sequence[str]]:
    if group in ("kpx_group_1", "kpx_group_2"):
        return preregister["stage1"]["group_1_and_2_required_segments"]
    return preregister["stage1"]["group_3_required_segments"]


def _stage1_blocks(
    preregister: Mapping[str, Any], group: str
) -> list[Mapping[str, Any]]:
    if group in ("kpx_group_1", "kpx_group_2"):
        return list(preregister["stage1"]["group_1_and_2_expanding_blocks"])
    return [preregister["stage1"]["group_3_block"]]


def _fit_apply_block(
    *,
    group: str,
    block: Mapping[str, Any],
    features: pd.DataFrame,
    actual: pd.Series,
    baseline: pd.Series,
    preregister: Mapping[str, Any],
) -> tuple[
    ResidualHistogramBayesClassifier,
    pd.Series,
    pd.Series,
    np.ndarray,
    dict[str, Any],
]:
    fit_index = _interval(features.index, block["fit"])
    application_index = _interval(features.index, block["apply"])
    assert_strict_fit_predict_order(fit_index, application_index)
    classifier = ResidualHistogramBayesClassifier(
        model_parameters=preregister["classifier"]["parameters"],
        minimum_eligible_fit_rows=int(
            preregister["stage1"]["minimum_eligible_fit_rows"]
        ),
    ).fit(
        features.loc[fit_index],
        actual.loc[fit_index],
        baseline.loc[fit_index],
        capacity_kwh=CAPACITY_KWH[group],
    )
    action, utility, probability = classifier.predict_raw_action(
        features.loc[application_index], baseline.loc[application_index]
    )
    evidence = {
        **classifier.metadata(),
        "block_id": str(block["id"]),
        "application_rows": int(len(application_index)),
        "application_start": application_index.min().isoformat(),
        "application_end": application_index.max().isoformat(),
        "fit_max_strictly_before_application_min": bool(
            fit_index.max() < application_index.min()
        ),
        "application_overlap_rows": int(fit_index.intersection(application_index).size),
        "raw_action_min_cf": float(action.min()),
        "raw_action_max_cf": float(action.max()),
        "raw_action_mean_cf": float(action.mean()),
        "expected_utility_mean": float(utility.mean()),
        "probability_row_sum_max_abs_error": float(
            np.max(np.abs(probability.sum(axis=1) - 1.0))
        ),
        "actual_target_scada_features": 0,
        "same_row_fit_score_computed": False,
    }
    return classifier, action, utility, probability, evidence


def _stage1(
    *,
    raw_dir: Path,
    cache_dir: Path,
    artifact_root: Path,
    recipe_path: Path,
    preregister_path: Path,
    out_dir: Path,
    preregister: Mapping[str, Any],
) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"Stage 1 requires a new output directory: {out_dir}")
    created_utc = utc_now()
    out_dir.mkdir(parents=True)
    filesystem_creation_utc = datetime.fromtimestamp(
        out_dir.stat().st_ctime, tz=timezone.utc
    ).isoformat()
    shutil.copyfile(preregister_path, out_dir / "preregister.json")
    shutil.copyfile(preregister_path.with_suffix(".sha256"), out_dir / "preregister.sha256")

    source_before = _source_snapshot(preregister_path)
    inputs_before = _stage1_input_snapshot(
        raw_dir, artifact_root, recipe_path, preregister_path
    )
    _verify_stage1_expected_hashes(inputs_before, preregister)
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    target_scales = {
        name: recipe["models"][name]["target_scale"] for name in COMPONENT_NAMES
    }
    if set(target_scales.values()) != {"capacity_factor"}:
        raise AssertionError("locked component target scales are not capacity_factor")

    labels = _read_labels(
        raw_dir / "train" / "train_labels.csv",
        nrows=EXPECTED_ROWS_PRE2024,
        expected_end=YEAR_2023_END,
        prefix_bytes=OFFICIAL_STAGE1_PREFIX_BYTES["labels"],
        expected_prefix_sha256=OFFICIAL_LABEL_PREFIX_SHA256,
    )
    print("Stage 1: materializing bounded pre-2024 compact weather", flush=True)
    raw_features, raw_contract = _read_stage1_raw_features(raw_dir, labels)
    weather = _core_weather(raw_features, WEATHER_COLUMNS)
    del raw_features
    expected_raw = preregister["stage1_expected_sha256"]
    for source in ("ldaps", "gfs"):
        contract = raw_contract["raw_prefix"][source]
        if contract["physical_byte_limit"] != expected_raw[f"{source}_prefix_bytes"]:
            raise AssertionError(f"{source} physical prefix byte limit changed")
        if contract["physical_prefix_sha256"] != expected_raw[f"{source}_prefix_sha256"]:
            raise AssertionError(f"{source} physical prefix hash changed")
        if contract["suffix_bytes_exposed_to_parser"] != 0:
            raise AssertionError(f"{source} suffix bytes reached Stage 1 parser")

    baseline, baseline_audit = _read_stage1_baseline(artifact_root)
    components = _read_stage1_components(artifact_root)
    meta: dict[str, pd.DataFrame] = {}
    for group in TARGET_COLS:
        index = baseline[group].dropna().index
        meta[group] = build_meta_features(
            baseline.loc[index, group],
            {name: components[group][name].loc[index] for name in COMPONENT_NAMES},
            weather[group].loc[index],
            capacity_kwh=CAPACITY_KWH[group],
        )
        _atomic_parquet(meta[group], out_dir / "features" / f"stage1__{group}.parquet")

    raw_action = pd.DataFrame(
        np.nan, index=baseline.index, columns=list(TARGET_COLS), dtype=float
    )
    expected_utility = raw_action.copy()
    maximum_probability = raw_action.copy()
    probability_entropy = raw_action.copy()
    training: dict[str, Any] = {}
    for group in TARGET_COLS:
        training[group] = {}
        for block in _stage1_blocks(preregister, group):
            block_id = str(block["id"])
            print(f"Stage 1: {group} {block_id}", flush=True)
            classifier, action, utility, probability, evidence = _fit_apply_block(
                group=group,
                block=block,
                features=meta[group],
                actual=labels[group],
                baseline=baseline[group],
                preregister=preregister,
            )
            if raw_action.loc[action.index, group].notna().any():
                raise AssertionError("Stage 1 application blocks overlap")
            raw_action.loc[action.index, group] = action
            expected_utility.loc[utility.index, group] = utility
            maximum_probability.loc[action.index, group] = probability.max(axis=1)
            safe_probability = np.where(probability > 0.0, probability, 1.0)
            probability_entropy.loc[action.index, group] = -np.sum(
                probability * np.log(safe_probability), axis=1
            )
            model_path = out_dir / "models" / f"stage1__{group}__{block_id}.joblib"
            _atomic_joblib(
                {
                    "classifier": classifier,
                    "group": group,
                    "block": dict(block),
                    "preregister_sha256": PREREGISTER_SHA256,
                },
                model_path,
            )
            evidence["model"] = describe_file(model_path)
            training[group][block_id] = evidence

    diagnostics = pd.concat(
        {
            "raw_action_delta_cf": raw_action,
            "expected_utility": expected_utility,
            "maximum_class_probability": maximum_probability,
            "class_probability_entropy": probability_entropy,
        },
        axis=1,
    )
    _atomic_parquet(diagnostics, out_dir / "oof" / "stage1_action_diagnostics.parquet")
    baseline_path = out_dir / "oof" / "stage1_corrected_v3_baseline.parquet"
    _atomic_parquet(baseline, baseline_path)

    candidate_paths: dict[str, Path] = {}
    for shrink in preregister["bayes_action"]["delta_shrink_candidates"]:
        key = str(float(shrink))
        candidate = baseline.copy()
        for group in TARGET_COLS:
            application_index = raw_action[group].dropna().index
            prediction = apply_action_shrink(
                baseline.loc[application_index, group],
                raw_action.loc[application_index, group],
                capacity_kwh=CAPACITY_KWH[group],
                shrink=float(shrink),
            )
            candidate.loc[application_index, group] = prediction
        path = out_dir / "oof" / f"stage1_shrink_{key.replace('.', 'p')}.parquet"
        _atomic_parquet(candidate, path)
        candidate_paths[key] = path

    # Selection metrics are computed only from re-opened canonical OOF files.
    comparisons: dict[str, dict[str, Any]] = {}
    for key, candidate_path in candidate_paths.items():
        comparisons[key] = {}
        for group in TARGET_COLS:
            comparisons[key][group] = _score_saved_group(
                labels=labels,
                baseline_path=baseline_path,
                candidate_path=candidate_path,
                group=group,
                segments=_stage1_segments(preregister, group),
            )
    selected = choose_group_shrinks(comparisons, TARGET_COLS)

    source_after = _source_snapshot(preregister_path)
    inputs_after = _stage1_input_snapshot(
        raw_dir, artifact_root, recipe_path, preregister_path
    )
    if source_before != source_after:
        raise AssertionError("Stage 1 source snapshot changed during execution")
    if inputs_before != inputs_after:
        raise AssertionError("Stage 1 safe input snapshot changed during execution")

    result = {
        "schema_version": 1,
        "experiment_id": "residual_histogram_multiclass_bayes_stage1",
        "output_directory_created_utc_before_reads": created_utc,
        "output_directory_filesystem_creation_utc": filesystem_creation_utc,
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "candidate_count": int(len(candidate_paths)),
        "model_family_count": 1,
        "selected_shrink_by_group": selected,
        "selected_nonidentity_group_count": int(
            sum(value is not None for value in selected.values())
        ),
        "selection_changes_after_2024": False,
        "comparisons_from_reopened_stored_oof": comparisons,
        "training": training,
        "target_scale_audit": {
            "locked_recipe_component_target_scales": target_scales,
            "component_artifact_units": "kWh",
            "capacity_divisions": 1,
            "baseline_exact_reconstruction": baseline_audit,
        },
        "meta_feature_audit": {
            "columns": list(META_FEATURE_COLUMNS),
            "column_count": len(META_FEATURE_COLUMNS),
            "actual_target_scada_features": 0,
            "by_group": {
                group: {
                    "rows": int(len(meta[group])),
                    "start": meta[group].index.min().isoformat(),
                    "end": meta[group].index.max().isoformat(),
                    "frame_sha256": _frame_sha256(meta[group]),
                }
                for group in TARGET_COLS
            },
        },
        "raw_feature_contract": raw_contract,
        "no_read_ledger": {
            "2024_labels_read": False,
            "2024_weather_read": False,
            "2024_component_or_baseline_predictions_read": False,
            "2025_weather_read": False,
            "2025_component_or_baseline_predictions_read": False,
            "sample_submission_read": False,
            "public_scores_or_scale_probe_read": False,
            "cache_directory_read": False,
            "cache_directory": str(cache_dir.resolve()),
            "label_prefix_rows": int(len(labels)),
            "label_prefix_end": labels.index.max().isoformat(),
            "label_suffix_bytes_exposed_to_parser": 0,
            "weather_suffix_bytes_exposed_to_parser": {
                source: int(raw_contract["raw_prefix"][source]["suffix_bytes_exposed_to_parser"])
                for source in ("ldaps", "gfs")
            },
        },
        "stored_oof": {
            "baseline": describe_file(baseline_path),
            "candidates": {
                key: describe_file(path) for key, path in candidate_paths.items()
            },
            "diagnostics": describe_file(
                out_dir / "oof" / "stage1_action_diagnostics.parquet"
            ),
        },
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
        "experiment_id": "residual_histogram_multiclass_bayes_pre2024_lock",
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "stage1_results_sha256": sha256_file(result_path),
        "source_snapshot_sha256": result["source_snapshot_sha256"],
        "input_snapshot_sha256": result["input_snapshot_sha256"],
        "selected_shrink_by_group": selected,
        "selection_changes_after_2024": False,
        "selection_score_source": "reopened_stored_oof_official_metric",
        "2024_read_before_lock": False,
        "2025_read_before_lock": False,
        "public_information_used": False,
    }
    _write_json(out_dir / "pre2024_lock.json", lock)
    print("Stage 1 locked shrink by group:", selected, flush=True)
    return result


def _stage1_candidate_path(out_dir: Path, shrink: float) -> Path:
    key = str(float(shrink)).replace(".", "p")
    return out_dir / "oof" / f"stage1_shrink_{key}.parquet"


def _load_and_verify_lock(
    *,
    raw_dir: Path,
    artifact_root: Path,
    recipe_path: Path,
    preregister_path: Path,
    out_dir: Path,
    preregister: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    result_path = out_dir / "stage1_results.json"
    lock_path = out_dir / "pre2024_lock.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["preregister_sha256"] != PREREGISTER_SHA256:
        raise AssertionError("pre-2024 lock preregister SHA changed")
    if lock["stage1_results_sha256"] != sha256_file(result_path):
        raise AssertionError("Stage 1 result changed after lock")
    if lock["selected_shrink_by_group"] != result["selected_shrink_by_group"]:
        raise AssertionError("lock selection differs from Stage 1 result")
    current_source = _source_snapshot(preregister_path)
    current_inputs = _stage1_input_snapshot(
        raw_dir, artifact_root, recipe_path, preregister_path
    )
    if _mapping_sha256(current_source) != lock["source_snapshot_sha256"]:
        raise AssertionError("source snapshot changed after Stage 1 lock")
    if _mapping_sha256(current_inputs) != lock["input_snapshot_sha256"]:
        raise AssertionError("Stage 1 input snapshot changed after lock")
    _verify_stage1_expected_hashes(current_inputs, preregister)

    # Independently re-read the bounded label prefix and canonical OOF outputs,
    # then recalculate every Stage 1 selection metric before 2024 is accessible.
    labels = _read_labels(
        raw_dir / "train" / "train_labels.csv",
        nrows=EXPECTED_ROWS_PRE2024,
        expected_end=YEAR_2023_END,
        prefix_bytes=OFFICIAL_STAGE1_PREFIX_BYTES["labels"],
        expected_prefix_sha256=OFFICIAL_LABEL_PREFIX_SHA256,
    )
    baseline_path = out_dir / "oof" / "stage1_corrected_v3_baseline.parquet"
    comparisons: dict[str, dict[str, Any]] = {}
    for shrink in preregister["bayes_action"]["delta_shrink_candidates"]:
        key = str(float(shrink))
        candidate_path = _stage1_candidate_path(out_dir, float(shrink))
        comparisons[key] = {
            group: _score_saved_group(
                labels=labels,
                baseline_path=baseline_path,
                candidate_path=candidate_path,
                group=group,
                segments=_stage1_segments(preregister, group),
            )
            for group in TARGET_COLS
        }
    if _mapping_sha256(comparisons) != _mapping_sha256(
        result["comparisons_from_reopened_stored_oof"]
    ):
        raise AssertionError("Stage 1 comparisons do not recompute exactly")
    recalculated = choose_group_shrinks(comparisons, TARGET_COLS)
    if recalculated != lock["selected_shrink_by_group"]:
        raise AssertionError("Stage 1 group selection does not recompute")
    return lock, result


def _stage2_input_snapshot(
    *,
    raw_dir: Path,
    cache_dir: Path,
    artifact_root: Path,
    recipe_path: Path,
) -> dict[str, Any]:
    paths: dict[str, Path] = {
        "labels_full_file": raw_dir / "train" / "train_labels.csv",
        "locked_recipe": recipe_path,
        "gate_baseline": artifact_root / "oof" / "gate2024_locked_v3_cf_fix.parquet",
    }
    for group in TARGET_COLS:
        paths[f"weather_train_cache_{group}"] = (
            cache_dir / f"{group}_weather_train.parquet"
        )
    for name, relative in GATE_COMPONENT_FILES.items():
        paths[f"gate_component_{name}"] = artifact_root / relative
    return {name: describe_file(path) for name, path in paths.items()}


def _stage2_no_candidate(out_dir: Path, lock: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "selected_shrink_by_group": lock["selected_shrink_by_group"],
        "2024_read": False,
        "2025_read": False,
        "public_information_used": False,
        "promoted": False,
        "reason": "no group shrink improved every registered Stage 1 segment",
        "selection_changes_after_2024": False,
        "pre2024_lock_sha256": sha256_file(out_dir / "pre2024_lock.json"),
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
            "promoted": False,
            "selected_shrink_by_group": lock["selected_shrink_by_group"],
            "2024_read": False,
            "2025_read": False,
            "public_information_used": False,
            "stage2_results_sha256": sha256_file(result_path),
            "pre2024_lock_sha256": sha256_file(out_dir / "pre2024_lock.json"),
        },
    )
    return result


def _fit_history_apply_period(
    *,
    group: str,
    fit_features: pd.DataFrame,
    fit_actual: pd.Series,
    fit_baseline: pd.Series,
    application_features: pd.DataFrame,
    application_baseline: pd.Series,
    preregister: Mapping[str, Any],
) -> tuple[
    ResidualHistogramBayesClassifier,
    pd.Series,
    pd.Series,
    np.ndarray,
    dict[str, Any],
]:
    assert_strict_fit_predict_order(fit_features.index, application_features.index)
    classifier = ResidualHistogramBayesClassifier(
        model_parameters=preregister["classifier"]["parameters"],
        minimum_eligible_fit_rows=int(
            preregister["stage1"]["minimum_eligible_fit_rows"]
        ),
    ).fit(
        fit_features,
        fit_actual,
        fit_baseline,
        capacity_kwh=CAPACITY_KWH[group],
    )
    action, utility, probability = classifier.predict_raw_action(
        application_features, application_baseline
    )
    evidence = {
        **classifier.metadata(),
        "application_rows": int(len(application_features)),
        "application_start": application_features.index.min().isoformat(),
        "application_end": application_features.index.max().isoformat(),
        "fit_max_strictly_before_application_min": bool(
            fit_features.index.max() < application_features.index.min()
        ),
        "raw_action_min_cf": float(action.min()),
        "raw_action_max_cf": float(action.max()),
        "raw_action_mean_cf": float(action.mean()),
        "expected_utility_mean": float(utility.mean()),
        "probability_row_sum_max_abs_error": float(
            np.max(np.abs(probability.sum(axis=1) - 1.0))
        ),
        "actual_target_scada_features": 0,
        "same_row_fit_score_computed": False,
    }
    return classifier, action, utility, probability, evidence


def _stage2(
    *,
    raw_dir: Path,
    cache_dir: Path,
    artifact_root: Path,
    recipe_path: Path,
    preregister_path: Path,
    out_dir: Path,
    preregister: Mapping[str, Any],
) -> dict[str, Any]:
    if (out_dir / "stage2_results.json").exists():
        raise FileExistsError("Stage 2 result already exists")
    lock, _ = _load_and_verify_lock(
        raw_dir=raw_dir,
        artifact_root=artifact_root,
        recipe_path=recipe_path,
        preregister_path=preregister_path,
        out_dir=out_dir,
        preregister=preregister,
    )
    selected = lock["selected_shrink_by_group"]
    if not any(value is not None for value in selected.values()):
        return _stage2_no_candidate(out_dir, lock)

    stage2_inputs_before = _stage2_input_snapshot(
        raw_dir=raw_dir,
        cache_dir=cache_dir,
        artifact_root=artifact_root,
        recipe_path=recipe_path,
    )
    print("Stage 2: immutable lock verified; opening fixed 2024 confirmation", flush=True)
    labels = _read_labels(
        raw_dir / "train" / "train_labels.csv",
        nrows=EXPECTED_ROWS_THROUGH2024,
        expected_end=YEAR_2024_END,
    )
    all_weather = _read_features(cache_dir, labels, expected_end=YEAR_2024_END)
    weather = _core_weather(all_weather, WEATHER_COLUMNS)
    del all_weather
    stage1_baseline, stage1_baseline_audit = _read_stage1_baseline(artifact_root)
    stage1_components = _read_stage1_components(artifact_root)
    gate_index = pd.date_range(
        YEAR_2024_START, YEAR_2024_FINISH, freq="h", name="forecast_kst_dtm"
    )
    gate_baseline, gate_baseline_audit = _reconstruct_baseline(
        artifact_root=artifact_root,
        recipe_path=recipe_path,
        component_files=GATE_COMPONENT_FILES,
        reference_path=artifact_root / "oof" / "gate2024_locked_v3_cf_fix.parquet",
        expected_index=gate_index,
    )
    gate_components = _read_period_components(
        artifact_root,
        GATE_COMPONENT_FILES,
        start=YEAR_2024_START,
        end=YEAR_2024_FINISH,
        rows=8760,
    )

    history_meta: dict[str, pd.DataFrame] = {}
    gate_meta: dict[str, pd.DataFrame] = {}
    for group in TARGET_COLS:
        history_index = _interval(
            stage1_baseline[group].dropna().index,
            preregister["stage2"]["fit_history"][group],
        )
        history_meta[group] = build_meta_features(
            stage1_baseline.loc[history_index, group],
            {
                name: stage1_components[group][name].loc[history_index]
                for name in COMPONENT_NAMES
            },
            weather[group].loc[history_index],
            capacity_kwh=CAPACITY_KWH[group],
        )
        gate_meta[group] = build_meta_features(
            gate_baseline[group],
            gate_components[group],
            weather[group].loc[gate_index],
            capacity_kwh=CAPACITY_KWH[group],
        )
        _atomic_parquet(gate_meta[group], out_dir / "features" / f"stage2__{group}.parquet")

    candidate = gate_baseline.copy()
    raw_action = pd.DataFrame(
        np.nan, index=gate_index, columns=list(TARGET_COLS), dtype=float
    )
    expected_utility = raw_action.copy()
    maximum_probability = raw_action.copy()
    probability_entropy = raw_action.copy()
    training: dict[str, Any] = {}
    for group in TARGET_COLS:
        shrink = selected[group]
        if shrink is None:
            training[group] = {"identity": True, "fit_performed": False}
            continue
        print(f"Stage 2: fitting fixed {group} shrink={shrink}", flush=True)
        fit_index = history_meta[group].index
        classifier, action, utility, probability, evidence = _fit_history_apply_period(
            group=group,
            fit_features=history_meta[group],
            fit_actual=labels.loc[fit_index, group],
            fit_baseline=stage1_baseline.loc[fit_index, group],
            application_features=gate_meta[group],
            application_baseline=gate_baseline[group],
            preregister=preregister,
        )
        raw_action[group] = action
        expected_utility[group] = utility
        maximum_probability[group] = probability.max(axis=1)
        safe_probability = np.where(probability > 0.0, probability, 1.0)
        probability_entropy[group] = -np.sum(
            probability * np.log(safe_probability), axis=1
        )
        candidate[group] = apply_action_shrink(
            gate_baseline[group],
            action,
            capacity_kwh=CAPACITY_KWH[group],
            shrink=float(shrink),
        )
        model_path = out_dir / "models" / f"stage2__{group}.joblib"
        _atomic_joblib(
            {
                "classifier": classifier,
                "group": group,
                "locked_shrink": float(shrink),
                "preregister_sha256": PREREGISTER_SHA256,
            },
            model_path,
        )
        evidence.update(
            {
                "identity": False,
                "locked_shrink": float(shrink),
                "model": describe_file(model_path),
            }
        )
        training[group] = evidence

    diagnostics = pd.concat(
        {
            "raw_action_delta_cf": raw_action,
            "expected_utility": expected_utility,
            "maximum_class_probability": maximum_probability,
            "class_probability_entropy": probability_entropy,
        },
        axis=1,
    )
    _atomic_parquet(diagnostics, out_dir / "oof" / "stage2_action_diagnostics.parquet")
    baseline_path = out_dir / "oof" / "stage2_corrected_v3_baseline.parquet"
    candidate_path = out_dir / "oof" / "stage2_locked_mixed_candidate.parquet"
    _atomic_parquet(gate_baseline, baseline_path)
    _atomic_parquet(candidate, candidate_path)

    comparisons = {
        group: _score_saved_group(
            labels=labels,
            baseline_path=baseline_path,
            candidate_path=candidate_path,
            group=group,
            segments=preregister["stage2"]["required_segments"],
        )
        for group in TARGET_COLS
        if selected[group] is not None
    }
    aggregate_segments = {
        name: preregister["stage2"]["required_segments"][name]
        for name in ("full", "H1", "H2")
    }
    aggregate = _score_saved_aggregate(
        labels=labels,
        baseline_path=baseline_path,
        candidate_path=candidate_path,
        segments=aggregate_segments,
    )
    group_gate = bool(comparisons) and all(
        float(item["delta"]) > 0.0
        for slices in comparisons.values()
        for item in slices.values()
    )
    aggregate_gate = all(
        float(item["total_score_delta"]) > 0.0 for item in aggregate.values()
    )
    promoted = bool(group_gate and aggregate_gate)
    stage2_inputs_after = _stage2_input_snapshot(
        raw_dir=raw_dir,
        cache_dir=cache_dir,
        artifact_root=artifact_root,
        recipe_path=recipe_path,
    )
    if stage2_inputs_before != stage2_inputs_after:
        raise AssertionError("Stage 2 inputs changed during confirmation")

    result = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "selected_shrink_by_group": selected,
        "selection_changes_after_2024": False,
        "2024_read": True,
        "2025_read": False,
        "public_information_used": False,
        "comparisons_nonidentity_groups_from_reopened_stored_oof": comparisons,
        "aggregate_mixed_candidate_from_reopened_stored_oof": aggregate,
        "group_confirmation_gate": bool(group_gate),
        "aggregate_confirmation_gate": bool(aggregate_gate),
        "promoted": promoted,
        "training": training,
        "stage1_baseline_exact_reconstruction": stage1_baseline_audit,
        "stage2_baseline_exact_reconstruction": gate_baseline_audit,
        "stage2_input_snapshot_before": stage2_inputs_before,
        "stage2_input_snapshot_after": stage2_inputs_after,
        "stage2_input_snapshot_sha256": _mapping_sha256(stage2_inputs_before),
        "stored_oof": {
            "baseline": describe_file(baseline_path),
            "candidate": describe_file(candidate_path),
            "diagnostics": describe_file(
                out_dir / "oof" / "stage2_action_diagnostics.parquet"
            ),
        },
        "pre2024_lock_sha256": sha256_file(out_dir / "pre2024_lock.json"),
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
            "promoted": promoted,
            "selected_shrink_by_group": selected,
            "selection_changes_after_2024": False,
            "2024_read": True,
            "2025_read_before_lock": False,
            "public_information_used": False,
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
            selected=selected,
            labels=labels,
            weather=weather,
            gate_baseline=gate_baseline,
            gate_components=gate_components,
            gate_meta=gate_meta,
        )
    return result


def _final_input_snapshot(
    *,
    raw_dir: Path,
    cache_dir: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    paths: dict[str, Path] = {
        "sample_submission": raw_dir / "sample_submission.csv",
        "final_baseline": artifact_root
        / "final_cf_fix"
        / "predictions"
        / "corrected_v3_test.parquet",
    }
    for group in TARGET_COLS:
        paths[f"weather_test_cache_{group}"] = (
            cache_dir / f"{group}_weather_test.parquet"
        )
    for name, relative in FINAL_COMPONENT_FILES.items():
        paths[f"final_component_{name}"] = artifact_root / relative
    return {name: describe_file(path) for name, path in paths.items()}


def _write_submission(
    sample_path: Path, prediction: pd.DataFrame, destination: Path
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
    if not prediction.index.equals(expected_index):
        raise AssertionError("final prediction index differs from sample")
    output = sample.copy()
    for group in TARGET_COLS:
        output[group] = prediction[group].to_numpy(dtype=float)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        output.to_csv(
            temporary,
            index=False,
            encoding="utf-8-sig",
            float_format="%.6f",
            lineterminator="\n",
        )
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
        raise AssertionError("submission contains non-finite predictions")
    for group in TARGET_COLS:
        if not observed[group].between(0.0, 1.02 * CAPACITY_KWH[group]).all():
            raise AssertionError(f"submission {group} exceeds capacity clip")
        expected = np.round(prediction[group].to_numpy(dtype=float), 6)
        if not np.array_equal(observed[group].to_numpy(dtype=float), expected):
            raise AssertionError(f"submission six-decimal readback differs for {group}")
    return {
        **describe_file(destination),
        "rows": int(len(observed)),
        "columns_exact": True,
        "id_time_exact": True,
        "utf8_sig": True,
        "six_decimal_readback_exact": True,
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
    selected: Mapping[str, float | None],
    labels: pd.DataFrame,
    weather: Mapping[str, pd.DataFrame],
    gate_baseline: pd.DataFrame,
    gate_components: Mapping[str, Mapping[str, pd.Series]],
    gate_meta: Mapping[str, pd.DataFrame],
) -> None:
    promotion_lock = json.loads(
        (out_dir / "promotion_lock.json").read_text(encoding="utf-8")
    )
    if promotion_lock.get("promoted") is not True:
        raise AssertionError("2025 access requires a positive immutable promotion lock")
    final_inputs_before = _final_input_snapshot(
        raw_dir=raw_dir, cache_dir=cache_dir, artifact_root=artifact_root
    )
    sample_path = raw_dir / "sample_submission.csv"
    sample = pd.read_csv(sample_path, encoding="utf-8-sig")
    test_index = pd.DatetimeIndex(
        pd.to_datetime(sample["forecast_kst_dtm"], errors="raise"),
        name="forecast_kst_dtm",
    )
    if (
        len(test_index) != EXPECTED_TEST_ROWS
        or test_index.min() != YEAR_2025_START
        or test_index.max() != YEAR_2025_FINISH
        or not test_index.is_unique
        or not test_index.is_monotonic_increasing
    ):
        raise AssertionError("sample 2025 interval changed")
    test_weather_full = _read_test_features(cache_dir, test_index)
    test_weather = _core_weather(test_weather_full, WEATHER_COLUMNS)
    del test_weather_full
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
    final_components = _read_period_components(
        artifact_root,
        FINAL_COMPONENT_FILES,
        start=YEAR_2025_START,
        end=YEAR_2025_FINISH,
        rows=8760,
    )
    final_meta = {
        group: build_meta_features(
            final_baseline[group],
            final_components[group],
            test_weather[group],
            capacity_kwh=CAPACITY_KWH[group],
        )
        for group in TARGET_COLS
    }
    predictions = final_baseline.copy()
    raw_action = pd.DataFrame(
        np.nan, index=test_index, columns=list(TARGET_COLS), dtype=float
    )
    expected_utility = raw_action.copy()
    training: dict[str, Any] = {}
    gate_index = gate_baseline.index
    for group in TARGET_COLS:
        _atomic_parquet(
            final_meta[group], out_dir / "features" / f"final__{group}.parquet"
        )
        shrink = selected[group]
        if shrink is None:
            if not np.array_equal(
                predictions[group].to_numpy(dtype=float),
                final_baseline[group].to_numpy(dtype=float),
            ):
                raise AssertionError("identity group changed from final baseline")
            training[group] = {"identity": True, "fit_performed": False}
            continue
        print(f"Final: refitting fixed {group} on 2024 only", flush=True)
        classifier, action, utility, probability, evidence = _fit_history_apply_period(
            group=group,
            fit_features=gate_meta[group],
            fit_actual=labels.loc[gate_index, group],
            fit_baseline=gate_baseline[group],
            application_features=final_meta[group],
            application_baseline=final_baseline[group],
            preregister=preregister,
        )
        raw_action[group] = action
        expected_utility[group] = utility
        predictions[group] = apply_action_shrink(
            final_baseline[group],
            action,
            capacity_kwh=CAPACITY_KWH[group],
            shrink=float(shrink),
        )
        model_path = out_dir / "models" / f"final__{group}.joblib"
        _atomic_joblib(
            {
                "classifier": classifier,
                "group": group,
                "locked_shrink": float(shrink),
                "fit_period": "2024_only",
                "preregister_sha256": PREREGISTER_SHA256,
            },
            model_path,
        )
        evidence.update(
            {
                "identity": False,
                "locked_shrink": float(shrink),
                "model": describe_file(model_path),
                "probability_row_sum_max_abs_error": float(
                    np.max(np.abs(probability.sum(axis=1) - 1.0))
                ),
            }
        )
        training[group] = evidence

    diagnostics = pd.concat(
        {
            "raw_action_delta_cf": raw_action,
            "expected_utility": expected_utility,
        },
        axis=1,
    )
    _atomic_parquet(diagnostics, out_dir / "predictions" / "final_action_diagnostics.parquet")
    prediction_path = out_dir / "predictions" / "residual_histogram_bayes_2025.parquet"
    _atomic_parquet(predictions, prediction_path)
    submission_path = out_dir / preregister["final"]["submission_name"]
    submission_audit = _write_submission(sample_path, predictions, submission_path)
    final_inputs_after = _final_input_snapshot(
        raw_dir=raw_dir, cache_dir=cache_dir, artifact_root=artifact_root
    )
    if final_inputs_before != final_inputs_after:
        raise AssertionError("final inputs changed during execution")
    _write_json(
        out_dir / "final_results.json",
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "selected_shrink_by_group": dict(selected),
            "selection_changes_after_2024": False,
            "fit_period": "2024_only",
            "application_period": "2025_unlabeled",
            "training": training,
            "baseline_exact_reconstruction": final_baseline_audit,
            "final_input_snapshot_before": final_inputs_before,
            "final_input_snapshot_after": final_inputs_after,
            "final_input_snapshot_sha256": _mapping_sha256(final_inputs_before),
            "prediction": describe_file(prediction_path),
            "submission": submission_audit,
            "leaderboard_score_claim": False,
            "public_information_used": False,
        },
    )


def _write_manifest(
    *, args: argparse.Namespace, preregister: Mapping[str, Any], name: str
) -> None:
    destination = args.out_dir / name
    outputs = sorted(
        path for path in args.out_dir.rglob("*") if path.is_file() and path != destination
    )
    stage1 = json.loads(
        (args.out_dir / "stage1_results.json").read_text(encoding="utf-8")
    )
    stage2_path = args.out_dir / "stage2_results.json"
    stage2 = (
        json.loads(stage2_path.read_text(encoding="utf-8"))
        if stage2_path.exists()
        else None
    )
    final_path = args.out_dir / "final_results.json"
    final = (
        json.loads(final_path.read_text(encoding="utf-8")) if final_path.exists() else None
    )
    payload = {
        "schema_version": 1,
        "artifact_type": "residual_histogram_multiclass_bayes_strict_forward",
        "created_utc": utc_now(),
        "command": [sys.executable, *sys.argv],
        "runtime": {"packages": package_versions(), "git": git_state(PROJECT_DIR)},
        "preregister_sha256": PREREGISTER_SHA256,
        "candidate_count": len(preregister["bayes_action"]["delta_shrink_candidates"]),
        "class_count": len(RESIDUAL_CENTRES_CF),
        "action_count": len(ACTION_DELTAS_CF),
        "stage1_source_snapshot": stage1["source_snapshot_before"],
        "stage1_input_snapshot": stage1["input_snapshot_before"],
        "stage1_raw_prefix_contract": stage1["raw_feature_contract"],
        "stage1_no_read_ledger": stage1["no_read_ledger"],
        "stage2": {
            "exists": stage2 is not None,
            "2024_read": bool(stage2 and stage2["2024_read"]),
            "2025_read": bool(final is not None),
            "promoted": bool(stage2 and stage2["promoted"]),
            "input_snapshot": (
                stage2.get("stage2_input_snapshot_before") if stage2 else None
            ),
        },
        "final_input_snapshot": (
            final.get("final_input_snapshot_before") if final else None
        ),
        "public_information_used": False,
        "outputs": [describe_file(path) for path in outputs],
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
            recipe_path=args.recipe,
            preregister_path=args.preregister,
            out_dir=args.out_dir,
            preregister=preregister,
        )
        if args.stage == "stage1":
            _write_manifest(args=args, preregister=preregister, name="stage1_manifest.json")
            return
    if args.stage in {"stage2", "all"}:
        _stage2(
            raw_dir=args.raw_dir,
            cache_dir=args.cache_dir,
            artifact_root=args.artifact_root,
            recipe_path=args.recipe,
            preregister_path=args.preregister,
            out_dir=args.out_dir,
            preregister=preregister,
        )
        _write_manifest(args=args, preregister=preregister, name="manifest.json")


if __name__ == "__main__":
    main()

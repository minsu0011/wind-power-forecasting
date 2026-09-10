"""Strict-forward full-feature annual quantile-map experiment.

Stage 1 physically reads only the byte-bounded raw 2022--2023 NWP/label
prefix.  Stage 2 cannot run until an immutable Stage-1 lock promotes at least
one exact group/candidate pair.  Test weather is inaccessible until the fixed
Stage-2 gate promotes at least one pair.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# The strict helpers below are the canonical physical-prefix readers already
# audited by the repository.  Importing the module performs no data reads.
from scripts.run_shared_q07_multiseed import (  # noqa: E402
    EXPECTED_ROWS_PRE2024,
    EXPECTED_ROWS_THROUGH2024,
    EXPECTED_TEST_ROWS,
    H2_2023_START,
    OFFICIAL_LABEL_PREFIX_SHA256,
    OFFICIAL_STAGE1_PREFIX_BYTES,
    Q4_2023_START,
    YEAR_2022_END,
    YEAR_2022_START,
    YEAR_2023_END,
    YEAR_2023_START,
    YEAR_2024_END,
    YEAR_2024_START,
    _atomic_csv,
    _atomic_joblib,
    _atomic_parquet,
    _frame_sha256,
    _read_features,
    _read_labels,
    _read_stage1_raw_features,
    _read_test_features,
)
from scripts.run_year_quantile_map_audit import _quantile_map  # noqa: E402
from src.manifest import (  # noqa: E402
    describe_file,
    git_state,
    package_versions,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from src.metric import CAPACITY_KWH, TARGET_COLS, group_metrics  # noqa: E402


# Pinned only after both the JSON contract and exact 565-column file existed.
PREREGISTER_SHA256 = (
    "5ea458d443c0306499fcf32e29e42cba8c06639dac8af6fd0cffab62b67b633c"
)
MAPPED_COLUMNS_SHA256 = (
    "2c80a7bbef581d19917bdb248ec68db52f8f73538c0f1383cc553bc10339bdee"
)
CANONICAL_COLUMNS_SHA256 = (
    "55835269be52c5ceb681435a17c8e1ff8c8f40f36393706691996cfc4c143276"
)
MAPPED_COLUMN_COUNT = 565
MODEL_COLUMN_COUNT = 612
DELTA_WEIGHT = 0.25

OBJECTIVES = ("l1", "q07")
CANDIDATES = (
    "mapped_l1_absolute",
    "mapped_q07_absolute",
    "mapped_mean_absolute",
    "v3_delta_l1_w025",
    "v3_delta_q07_w025",
    "v3_delta_mean_w025",
)
FINAL_PRIORITY = (
    "v3_delta_mean_w025",
    "v3_delta_l1_w025",
    "v3_delta_q07_w025",
    "mapped_mean_absolute",
    "mapped_l1_absolute",
    "mapped_q07_absolute",
)

STAGE1_SLICES = {
    "kpx_group_1": {
        "full": (YEAR_2023_START, YEAR_2023_END),
        "H1": (YEAR_2023_START, H2_2023_START - pd.Timedelta(hours=1)),
        "H2": (H2_2023_START, YEAR_2023_END),
    },
    "kpx_group_2": {
        "full": (YEAR_2023_START, YEAR_2023_END),
        "H1": (YEAR_2023_START, H2_2023_START - pd.Timedelta(hours=1)),
        "H2": (H2_2023_START, YEAR_2023_END),
    },
    "kpx_group_3": {
        "full": (H2_2023_START, YEAR_2023_END),
        "Q3": (H2_2023_START, Q4_2023_START - pd.Timedelta(hours=1)),
        "Q4": (Q4_2023_START, YEAR_2023_END),
    },
}

STAGE2_SLICES = {
    "full": (YEAR_2024_START, YEAR_2024_END),
    "H1": (YEAR_2024_START, pd.Timestamp("2024-07-01 00:00:00")),
    "H2": (pd.Timestamp("2024-07-01 01:00:00"), YEAR_2024_END),
    "Q1": (YEAR_2024_START, pd.Timestamp("2024-04-01 00:00:00")),
    "Q2": (
        pd.Timestamp("2024-04-01 01:00:00"),
        pd.Timestamp("2024-07-01 00:00:00"),
    ),
    "Q3": (
        pd.Timestamp("2024-07-01 01:00:00"),
        pd.Timestamp("2024-10-01 00:00:00"),
    ),
    "Q4": (pd.Timestamp("2024-10-01 01:00:00"), YEAR_2024_END),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir", type=Path, default=Path(r"data/local/open")
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/cache"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--preregister",
        type=Path,
        default=Path("configs/full_feature_year_qm_preregister.json"),
    )
    parser.add_argument(
        "--mapped-columns",
        type=Path,
        default=Path("configs/full_qm_mapped_columns.txt"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/postgate/full_feature_year_qm_strict"),
    )
    parser.add_argument(
        "--stage", choices=("stage1", "stage2", "final", "all"), default="all"
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


def _canonical_sha256_lines(values: Sequence[str]) -> str:
    import hashlib

    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def _verify_contract(
    preregister_path: Path, mapped_columns_path: Path
) -> tuple[dict[str, Any], list[str]]:
    if sha256_file(preregister_path) != PREREGISTER_SHA256:
        raise AssertionError("preregister hash changed; refusing all data/model reads")
    if sha256_file(mapped_columns_path) != MAPPED_COLUMNS_SHA256:
        raise AssertionError("mapped-column file hash changed")
    config = json.loads(preregister_path.read_text(encoding="utf-8"))
    columns = mapped_columns_path.read_text(encoding="utf-8").splitlines()
    if len(columns) != MAPPED_COLUMN_COUNT or len(set(columns)) != len(columns):
        raise AssertionError("mapped-column count/uniqueness changed")
    if _canonical_sha256_lines(columns) != MAPPED_COLUMNS_SHA256:
        raise AssertionError("mapped-column canonical digest changed")
    if float(config["fixed_predictions_and_candidates"]["delta_weight"]) != DELTA_WEIGHT:
        raise AssertionError("delta weight changed")
    if tuple(config["selection_gate"]["final_pair_priority_if_multiple_pass"]) != FINAL_PRIORITY:
        raise AssertionError("final priority changed")
    return config, columns


def _verify_feature_schema(
    features: Mapping[str, pd.DataFrame], mapped_columns: Sequence[str]
) -> list[str]:
    schemas = [list(features[group].columns) for group in TARGET_COLS]
    if any(schema != schemas[0] for schema in schemas[1:]):
        raise AssertionError("group feature schemas differ")
    canonical = schemas[0]
    if len(canonical) != MODEL_COLUMN_COUNT:
        raise AssertionError("locked 612-column schema changed")
    if _canonical_sha256_lines(canonical) != CANONICAL_COLUMNS_SHA256:
        raise AssertionError("canonical feature schema digest changed")
    if any(column not in canonical for column in mapped_columns):
        raise AssertionError("mapped-column file contains unknown columns")
    expected = [
        column
        for column in canonical
        if not (
            column.startswith("time__")
            or column.startswith("site__")
            or column.endswith("_distance_km")
            or "__raw_missing_" in column
            or column.endswith("__flow_sin")
            or column.endswith("__flow_cos")
        )
    ]
    if expected != list(mapped_columns):
        raise AssertionError("mapped-column list differs from frozen selector")
    return canonical


def _model_parameters(objective: str) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "objective": "l1" if objective == "l1" else "quantile",
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
    if objective == "q07":
        parameters["alpha"] = 0.7
    return parameters


def _map_application(
    fit: pd.DataFrame,
    application: pd.DataFrame,
    mapped_columns: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not fit.columns.equals(application.columns):
        raise AssertionError("fit/application feature schemas differ")
    adapted = application.astype(np.float64, copy=True)
    mapped_summaries: dict[str, Any] = {}
    for column in mapped_columns:
        mapped, audit = _quantile_map(
            fit[column], application[column], grid_size=1001
        )
        adapted.loc[:, column] = mapped.to_numpy(dtype=np.float64)
        mapped_summaries[column] = audit
    mapped_set = set(mapped_columns)
    nonmapped = [column for column in application.columns if column not in mapped_set]
    if not np.array_equal(
        adapted.loc[:, nonmapped].to_numpy(dtype=np.float64),
        application.loc[:, nonmapped].to_numpy(dtype=np.float64),
        equal_nan=True,
    ):
        raise AssertionError("nonmapped application features changed")
    values = adapted.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all():
        raise AssertionError("mapped application contains non-finite values")
    shifts = np.asarray(
        [
            abs(float(payload["mapped_mean"]) - float(payload["application_mean"]))
            for payload in mapped_summaries.values()
        ],
        dtype=float,
    )
    audit = {
        "mapped_columns": len(mapped_columns),
        "nonmapped_columns": len(nonmapped),
        "nonmapped_bit_exact": True,
        "mean_absolute_feature_mean_shift": float(shifts.mean()),
        "max_absolute_feature_mean_shift": float(shifts.max()),
        "adapted_frame_sha256": _frame_sha256(adapted),
        "per_feature": mapped_summaries,
    }
    return adapted, audit


def _fit_predict_group(
    *,
    group: str,
    labels: pd.DataFrame,
    features: pd.DataFrame,
    mapped_columns: Sequence[str],
    fit_start: pd.Timestamp,
    fit_end: pd.Timestamp,
    application_index: pd.DatetimeIndex,
    model_dir: Path,
    model_prefix: str,
) -> tuple[pd.DataFrame, dict[str, Any], list[Path]]:
    fit_mask = (features.index >= fit_start) & (features.index <= fit_end)
    fit = features.loc[fit_mask]
    application = features.loc[application_index]
    if not application.index.equals(application_index):
        raise AssertionError(f"{group} application index is incomplete")
    target = labels.loc[fit.index, group] / CAPACITY_KWH[group]
    eligible = target.notna() & np.isfinite(target) & target.ge(0.10)
    if not eligible.any():
        raise AssertionError(f"{group} has no eligible fit rows")
    adapted, mapping_audit = _map_application(fit, application, mapped_columns)

    predictions = pd.DataFrame(index=application_index)
    model_records: dict[str, Any] = {}
    paths: list[Path] = []
    for objective in OBJECTIVES:
        print(f"{model_prefix}: fitting {group} {objective}", flush=True)
        model = LGBMRegressor(**_model_parameters(objective))
        model.fit(fit.loc[eligible], target.loc[eligible])
        capacity = CAPACITY_KWH[group]
        predictions[f"raw_{objective}"] = np.clip(
            model.predict(application) * capacity, 0.0, 1.02 * capacity
        )
        predictions[f"mapped_{objective}"] = np.clip(
            model.predict(adapted) * capacity, 0.0, 1.02 * capacity
        )
        model_path = model_dir / f"{model_prefix}_{group}_{objective}.joblib"
        _atomic_joblib(model, model_path)
        paths.append(model_path)
        model_records[objective] = {
            "parameters": _model_parameters(objective),
            "eligible_rows": int(eligible.sum()),
            "fit_rows_total": int(len(fit)),
            "fit_start": fit.index.min(),
            "fit_end": fit.index.max(),
            "application_rows": int(len(application)),
            "application_start": application.index.min(),
            "application_end": application.index.max(),
            "fit_max_strictly_before_application_min": bool(
                fit.index.max() < application.index.min()
            ),
            "model_path": model_path,
            "model_sha256": sha256_file(model_path),
        }
    predictions["raw_mean"] = 0.5 * (
        predictions["raw_l1"] + predictions["raw_q07"]
    )
    predictions["mapped_mean"] = 0.5 * (
        predictions["mapped_l1"] + predictions["mapped_q07"]
    )
    return predictions, {
        "models": model_records,
        "mapping": mapping_audit,
    }, paths


def _candidate_values(
    name: str, predictions: pd.DataFrame, baseline: pd.Series
) -> pd.Series:
    capacity = float(baseline.attrs.get("capacity_kwh", np.inf))
    if name == "mapped_l1_absolute":
        values = predictions["mapped_l1"]
    elif name == "mapped_q07_absolute":
        values = predictions["mapped_q07"]
    elif name == "mapped_mean_absolute":
        values = predictions["mapped_mean"]
    elif name == "v3_delta_l1_w025":
        values = baseline + DELTA_WEIGHT * (
            predictions["mapped_l1"] - predictions["raw_l1"]
        )
    elif name == "v3_delta_q07_w025":
        values = baseline + DELTA_WEIGHT * (
            predictions["mapped_q07"] - predictions["raw_q07"]
        )
    elif name == "v3_delta_mean_w025":
        values = baseline + DELTA_WEIGHT * (
            predictions["mapped_mean"] - predictions["raw_mean"]
        )
    else:
        raise KeyError(name)
    if np.isfinite(capacity):
        values = values.clip(0.0, 1.02 * capacity)
    return pd.Series(values.to_numpy(dtype=float), index=predictions.index, name=name)


def _score(actual: pd.Series, prediction: pd.Series, group: str) -> dict[str, Any]:
    metric = group_metrics(
        actual.to_numpy(dtype=float),
        prediction.to_numpy(dtype=float),
        CAPACITY_KWH[group],
        group_name=group,
    )
    payload = metric.as_dict()
    payload["score"] = 0.5 * (metric.one_minus_nmae + metric.ficr)
    return payload


def _comparison(
    *, actual: pd.Series, baseline: pd.Series, candidate: pd.Series, group: str
) -> dict[str, Any]:
    baseline_score = _score(actual, baseline, group)
    candidate_score = _score(actual, candidate, group)
    return {
        "baseline": baseline_score,
        "candidate": candidate_score,
        "delta": {
            key: float(candidate_score[key] - baseline_score[key])
            for key in ("score", "one_minus_nmae", "ficr")
        },
    }


def _slice_frame(
    series: pd.Series, start: pd.Timestamp, end: pd.Timestamp
) -> pd.Series:
    result = series.loc[(series.index >= start) & (series.index <= end)]
    if result.empty:
        raise AssertionError(f"empty slice {start}..{end}")
    return result


def _read_prediction(
    path: Path, expected_index: pd.DatetimeIndex, groups: Sequence[str]
) -> pd.DataFrame:
    frame = pd.read_parquet(path, engine="pyarrow")
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if not frame.index.equals(expected_index):
        raise AssertionError(f"prediction index changed: {path}")
    missing = set(groups).difference(frame.columns)
    if missing:
        raise AssertionError(f"prediction missing columns {sorted(missing)}: {path}")
    values = frame.loc[:, list(groups)]
    for group in groups:
        finite = np.isfinite(values[group].to_numpy(dtype=float))
        if not finite.all():
            raise AssertionError(f"prediction has non-finite {group}: {path}")
    return values.astype(float)


def _stage1(
    *,
    raw_dir: Path,
    artifact_root: Path,
    out_dir: Path,
    preregister_path: Path,
    mapped_columns_path: Path,
    config: Mapping[str, Any],
    mapped_columns: Sequence[str],
) -> dict[str, Any]:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Stage1 requires empty directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    prereg_copy = out_dir / "preregister.json"
    columns_copy = out_dir / "mapped_columns.txt"
    shutil.copyfile(preregister_path, prereg_copy)
    shutil.copyfile(mapped_columns_path, columns_copy)
    if sha256_file(prereg_copy) != PREREGISTER_SHA256:
        raise AssertionError("copied preregister differs")
    if sha256_file(columns_copy) != MAPPED_COLUMNS_SHA256:
        raise AssertionError("copied mapped columns differ")

    label_path = raw_dir / "train" / "train_labels.csv"
    labels = _read_labels(
        label_path,
        nrows=EXPECTED_ROWS_PRE2024,
        expected_end=YEAR_2023_END,
        prefix_bytes=OFFICIAL_STAGE1_PREFIX_BYTES["labels"],
        expected_prefix_sha256=OFFICIAL_LABEL_PREFIX_SHA256,
    )
    features, raw_contract = _read_stage1_raw_features(raw_dir, labels)
    canonical = _verify_feature_schema(features, mapped_columns)

    baseline_path = (
        artifact_root
        / "postgate/shared_q07_multiseed_strict/oof/stage1_corrected_v3_baseline.parquet"
    )
    baseline_raw = pd.read_parquet(baseline_path, engine="pyarrow")
    baseline_raw.index = pd.DatetimeIndex(
        baseline_raw.index, name="forecast_kst_dtm"
    )
    expected_stage1_index = pd.date_range(YEAR_2023_START, YEAR_2023_END, freq="h")
    if not baseline_raw.index.equals(expected_stage1_index):
        raise AssertionError("Stage1 v3 baseline index changed")

    prediction_paths: list[Path] = []
    model_paths: list[Path] = []
    training: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    controls: dict[str, Any] = {}
    locked_pairs: list[dict[str, str]] = []

    for group in TARGET_COLS:
        if group == "kpx_group_3":
            fit_start, fit_end = YEAR_2023_START, H2_2023_START - pd.Timedelta(hours=1)
            app_index = labels.index[
                (labels.index >= H2_2023_START) & (labels.index <= YEAR_2023_END)
            ]
        else:
            fit_start, fit_end = YEAR_2022_START, YEAR_2022_END
            app_index = expected_stage1_index
        prediction, train_record, paths = _fit_predict_group(
            group=group,
            labels=labels,
            features=features[group],
            mapped_columns=mapped_columns,
            fit_start=fit_start,
            fit_end=fit_end,
            application_index=app_index,
            model_dir=out_dir / "models",
            model_prefix="stage1",
        )
        model_paths.extend(paths)
        baseline = baseline_raw.loc[app_index, group].astype(float)
        if not np.isfinite(baseline.to_numpy()).all():
            raise AssertionError(f"Stage1 baseline incomplete for {group}")
        baseline.attrs["capacity_kwh"] = CAPACITY_KWH[group]
        for name in CANDIDATES:
            prediction[name] = _candidate_values(name, prediction, baseline)
        prediction["baseline_v3"] = baseline
        prediction.index.name = "forecast_kst_dtm"
        prediction_path = out_dir / "oof" / f"stage1_{group}.parquet"
        _atomic_parquet(prediction, prediction_path)
        prediction_paths.append(prediction_path)
        training[group] = train_record

        group_comparisons: dict[str, Any] = {}
        group_controls: dict[str, Any] = {}
        slices = STAGE1_SLICES[group]
        for name in CANDIDATES:
            candidate_comparisons: dict[str, Any] = {}
            for slice_name, (start, end) in slices.items():
                actual_slice = _slice_frame(labels[group], start, end)
                baseline_slice = _slice_frame(baseline, start, end)
                candidate_slice = _slice_frame(prediction[name], start, end)
                candidate_comparisons[slice_name] = _comparison(
                    actual=actual_slice,
                    baseline=baseline_slice,
                    candidate=candidate_slice,
                    group=group,
                )
            group_comparisons[name] = candidate_comparisons
            if all(
                values["delta"]["score"] > 0.0
                for values in candidate_comparisons.values()
            ):
                locked_pairs.append({"group": group, "candidate": name})
        for name in ("raw_l1", "raw_q07", "raw_mean", "mapped_l1", "mapped_q07", "mapped_mean"):
            per_slice: dict[str, Any] = {}
            for slice_name, (start, end) in slices.items():
                actual_slice = _slice_frame(labels[group], start, end)
                baseline_slice = _slice_frame(baseline, start, end)
                values_slice = _slice_frame(prediction[name], start, end)
                per_slice[slice_name] = _comparison(
                    actual=actual_slice,
                    baseline=baseline_slice,
                    candidate=values_slice,
                    group=group,
                )
            group_controls[name] = per_slice
        comparisons[group] = group_comparisons
        controls[group] = group_controls

    result = {
        "schema_version": 1,
        "experiment_id": "full_feature_year_qm_strict_forward_stage1_v1",
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "mapped_columns_sha256": MAPPED_COLUMNS_SHA256,
        "candidate_count": len(CANDIDATES),
        "candidates": list(CANDIDATES),
        "cache_files_read": False,
        "forbidden_2024_labels_features_or_candidate_scores_read": False,
        "test_2025_read": False,
        "physical_prefix_contract": raw_contract,
        "label_prefix": {
            "rows": EXPECTED_ROWS_PRE2024,
            "bytes": OFFICIAL_STAGE1_PREFIX_BYTES["labels"],
            "sha256": OFFICIAL_LABEL_PREFIX_SHA256,
            "suffix_bytes_exposed_to_parser": 0,
        },
        "feature_schema": {
            "model_columns": len(canonical),
            "model_columns_sha256": _canonical_sha256_lines(canonical),
            "mapped_columns": len(mapped_columns),
            "mapped_columns_sha256": _canonical_sha256_lines(mapped_columns),
        },
        "training": training,
        "controls_vs_v3": controls,
        "comparisons_vs_v3": comparisons,
        "lock_rule": "each required slice official per-group score delta strictly > 0",
        "locked_pairs": locked_pairs,
        "stage2_allowed": bool(locked_pairs),
        "public_scores_read": False,
        "leaderboard_score_claim": False,
    }
    result_path = out_dir / "stage1_results.json"
    _write_json(result_path, result)
    lock = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "mapped_columns_sha256": MAPPED_COLUMNS_SHA256,
        "stage1_results_sha256": sha256_file(result_path),
        "locked_pairs": locked_pairs,
        "stage2_candidate_pairs_frozen": True,
        "no_2024_reselection": True,
    }
    lock_path = out_dir / "stage1_lock.json"
    _write_json(lock_path, lock)
    manifest = {
        "schema_version": 1,
        "artifact_type": "full_feature_year_qm_stage1_strict",
        "created_utc": utc_now(),
        "preregister": describe_file(preregister_path),
        "mapped_columns": describe_file(mapped_columns_path),
        "baseline": describe_file(baseline_path),
        "raw_prefix_contract": raw_contract,
        "outputs": [
            describe_file(path)
            for path in [
                prereg_copy,
                columns_copy,
                *model_paths,
                *prediction_paths,
                result_path,
                lock_path,
            ]
        ],
        "locked_pairs": locked_pairs,
        "stage2_allowed": bool(locked_pairs),
    }
    _write_json(out_dir / "stage1_manifest.json", manifest)
    return lock


def _load_stage1_lock(out_dir: Path) -> dict[str, Any]:
    result_path = out_dir / "stage1_results.json"
    lock_path = out_dir / "stage1_lock.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["preregister_sha256"] != PREREGISTER_SHA256:
        raise AssertionError("Stage1 lock preregister differs")
    if lock["mapped_columns_sha256"] != MAPPED_COLUMNS_SHA256:
        raise AssertionError("Stage1 lock mapped columns differ")
    if lock["stage1_results_sha256"] != sha256_file(result_path):
        raise AssertionError("Stage1 result changed after lock")
    if lock["locked_pairs"] != result["locked_pairs"]:
        raise AssertionError("Stage1 lock/result pairs differ")
    if not lock.get("stage2_candidate_pairs_frozen") or not lock.get("no_2024_reselection"):
        raise AssertionError("Stage1 lock does not freeze Stage2")
    return lock


def _stage2(
    *,
    raw_dir: Path,
    cache_dir: Path,
    artifact_root: Path,
    out_dir: Path,
    mapped_columns: Sequence[str],
) -> dict[str, Any]:
    lock = _load_stage1_lock(out_dir)
    locked_pairs = list(lock["locked_pairs"])
    if not locked_pairs:
        print("Stage1 promoted no pair; Stage2 and 2025 remain unread.", flush=True)
        return {"promoted_pairs": [], "stage2_performed": False}
    if (out_dir / "stage2_results.json").exists():
        raise FileExistsError(out_dir / "stage2_results.json")

    labels = _read_labels(
        raw_dir / "train/train_labels.csv",
        nrows=EXPECTED_ROWS_THROUGH2024,
        expected_end=YEAR_2024_END,
    )
    features = _read_features(cache_dir, labels, expected_end=YEAR_2024_END)
    _verify_feature_schema(features, mapped_columns)
    application_index = pd.date_range(YEAR_2024_START, YEAR_2024_END, freq="h")
    locked_groups = [group for group in TARGET_COLS if any(pair["group"] == group for pair in locked_pairs)]
    v3_path = artifact_root / "oof/gate2024_locked_v3_cf_fix.parquet"
    v4_path = artifact_root / "oof/gate2024_recent_v4_cf_fix_calibration_fit.parquet"
    v3 = _read_prediction(v3_path, application_index, locked_groups)
    v4 = _read_prediction(v4_path, application_index, locked_groups)

    comparisons: dict[str, Any] = {}
    training: dict[str, Any] = {}
    output_paths: list[Path] = []
    promoted_pairs: list[dict[str, str]] = []
    for group in locked_groups:
        fit_start = YEAR_2023_START if group == "kpx_group_3" else YEAR_2022_START
        prediction, train_record, model_paths = _fit_predict_group(
            group=group,
            labels=labels,
            features=features[group],
            mapped_columns=mapped_columns,
            fit_start=fit_start,
            fit_end=YEAR_2023_END,
            application_index=application_index,
            model_dir=out_dir / "models",
            model_prefix="stage2",
        )
        training[group] = train_record
        output_paths.extend(model_paths)
        base_v3 = v3[group].copy()
        base_v3.attrs["capacity_kwh"] = CAPACITY_KWH[group]
        base_v4 = v4[group].copy()
        base_v4.attrs["capacity_kwh"] = CAPACITY_KWH[group]
        pair_names = [pair["candidate"] for pair in locked_pairs if pair["group"] == group]
        group_results: dict[str, Any] = {}
        for name in pair_names:
            candidate_v3 = _candidate_values(name, prediction, base_v3)
            candidate_v4 = _candidate_values(name, prediction, base_v4)
            per_slice: dict[str, Any] = {}
            for slice_name, (start, end) in STAGE2_SLICES.items():
                actual = _slice_frame(labels[group], start, end)
                per_slice[slice_name] = {
                    "vs_v3": _comparison(
                        actual=actual,
                        baseline=_slice_frame(base_v3, start, end),
                        candidate=_slice_frame(candidate_v3, start, end),
                        group=group,
                    ),
                    "vs_v4_diagnostic_only": _comparison(
                        actual=actual,
                        baseline=_slice_frame(base_v4, start, end),
                        candidate=_slice_frame(candidate_v4, start, end),
                        group=group,
                    ),
                }
            full_delta = per_slice["full"]["vs_v3"]["delta"]
            passed = bool(
                all(per_slice[key]["vs_v3"]["delta"]["score"] > 0.0 for key in ("full", "H1", "H2"))
                and full_delta["one_minus_nmae"] >= -0.001
            )
            group_results[name] = {"slices": per_slice, "passed": passed}
            if passed:
                promoted_pairs.append({"group": group, "candidate": name})
            prediction[f"{name}__v3"] = candidate_v3
            prediction[f"{name}__v4"] = candidate_v4
        prediction["baseline_v3"] = base_v3
        prediction["baseline_v4"] = base_v4
        pred_path = out_dir / "oof" / f"stage2_{group}.parquet"
        _atomic_parquet(prediction, pred_path)
        output_paths.append(pred_path)
        comparisons[group] = group_results

    selected_by_group: dict[str, str] = {}
    promoted_set = {(pair["group"], pair["candidate"]) for pair in promoted_pairs}
    for group in TARGET_COLS:
        for name in FINAL_PRIORITY:
            if (group, name) in promoted_set:
                selected_by_group[group] = name
                break
    result = {
        "schema_version": 1,
        "experiment_id": "full_feature_year_qm_strict_forward_stage2_v1",
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "stage1_results_sha256": lock["stage1_results_sha256"],
        "stage1_locked_pairs": locked_pairs,
        "only_locked_pairs_scored": True,
        "training": training,
        "comparisons": comparisons,
        "v4_diagnostic_only": True,
        "promoted_pairs": promoted_pairs,
        "selected_by_frozen_priority": selected_by_group,
        "test_2025_read": False,
        "final_allowed": bool(selected_by_group),
        "public_scores_read": False,
    }
    result_path = out_dir / "stage2_results.json"
    _write_json(result_path, result)
    stage2_lock = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "stage1_results_sha256": lock["stage1_results_sha256"],
        "stage2_results_sha256": sha256_file(result_path),
        "promoted_pairs": promoted_pairs,
        "selected_by_frozen_priority": selected_by_group,
        "final_2025_read_allowed": bool(selected_by_group),
    }
    stage2_lock_path = out_dir / "stage2_lock.json"
    _write_json(stage2_lock_path, stage2_lock)
    manifest = {
        "schema_version": 1,
        "artifact_type": "full_feature_year_qm_stage2_confirmation",
        "created_utc": utc_now(),
        "stage1_lock": describe_file(out_dir / "stage1_lock.json"),
        "inputs": [
            describe_file(raw_dir / "train/train_labels.csv"),
            *[describe_file(cache_dir / f"{group}_weather_train.parquet") for group in TARGET_COLS],
            describe_file(v3_path),
            describe_file(v4_path),
        ],
        "outputs": [describe_file(path) for path in [*output_paths, result_path, stage2_lock_path]],
        "promoted_pairs": promoted_pairs,
        "selected_by_frozen_priority": selected_by_group,
        "final_allowed": bool(selected_by_group),
    }
    _write_json(out_dir / "stage2_manifest.json", manifest)
    return stage2_lock


def _load_stage2_lock(out_dir: Path) -> dict[str, Any]:
    lock = json.loads((out_dir / "stage2_lock.json").read_text(encoding="utf-8"))
    if lock["preregister_sha256"] != PREREGISTER_SHA256:
        raise AssertionError("Stage2 lock preregister differs")
    if lock["stage2_results_sha256"] != sha256_file(out_dir / "stage2_results.json"):
        raise AssertionError("Stage2 result changed after lock")
    return lock


def _verify_submission(path: Path, sample: pd.DataFrame, expected: pd.DataFrame) -> dict[str, Any]:
    with path.open("rb") as stream:
        if stream.read(3) != b"\xef\xbb\xbf":
            raise AssertionError("submission lacks UTF-8 BOM")
    observed = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"forecast_id": "string", "forecast_kst_dtm": "string"},
    )
    if len(observed) != EXPECTED_TEST_ROWS or tuple(observed.columns) != tuple(sample.columns):
        raise AssertionError("submission rows/schema differ from sample")
    for column in ("forecast_id", "forecast_kst_dtm"):
        if not np.array_equal(observed[column].astype(str), sample[column].astype(str)):
            raise AssertionError(f"submission {column} differs from sample")
    values = observed.loc[:, list(TARGET_COLS)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise AssertionError("submission contains non-finite predictions")
    difference = float(np.max(np.abs(values - expected.to_numpy(dtype=float))))
    if difference > 5.1e-7:
        raise AssertionError("six-decimal submission readback differs")
    for group in TARGET_COLS:
        if (observed[group] < -1e-9).any() or (observed[group] > 1.02 * CAPACITY_KWH[group] + 1e-9).any():
            raise AssertionError(f"submission capacity clip failed for {group}")
    return {"rows": len(observed), "schema_id_time_exact": True, "utf8_bom": True, "finite": True, "capacity_clip": True, "max_readback_abs_diff": difference, "sha256": sha256_file(path)}


def _final(
    *,
    raw_dir: Path,
    cache_dir: Path,
    artifact_root: Path,
    out_dir: Path,
    mapped_columns: Sequence[str],
) -> dict[str, Any]:
    lock = _load_stage2_lock(out_dir)
    selected = dict(lock["selected_by_frozen_priority"])
    if not selected or not lock.get("final_2025_read_allowed"):
        print("Stage2 promoted no pair; 2025 remains unread and no CSV is written.", flush=True)
        return {"submission_created": False}
    if (out_dir / "manifest.json").exists():
        raise FileExistsError(out_dir / "manifest.json")

    labels = _read_labels(
        raw_dir / "train/train_labels.csv",
        nrows=EXPECTED_ROWS_THROUGH2024,
        expected_end=YEAR_2024_END,
    )
    train_features = _read_features(cache_dir, labels, expected_end=YEAR_2024_END)
    _verify_feature_schema(train_features, mapped_columns)
    sample_path = raw_dir / "sample_submission.csv"
    sample = pd.read_csv(
        sample_path,
        encoding="utf-8-sig",
        dtype={"forecast_id": "string", "forecast_kst_dtm": "string"},
    )
    if len(sample) != EXPECTED_TEST_ROWS:
        raise AssertionError("sample row count changed")
    test_index = pd.DatetimeIndex(
        pd.to_datetime(sample["forecast_kst_dtm"], errors="raise"),
        name="forecast_kst_dtm",
    )
    test_features = _read_test_features(cache_dir, test_index)
    _verify_feature_schema(test_features, mapped_columns)
    v3_path = artifact_root / "final_cf_fix/predictions/corrected_v3_test.parquet"
    v4_path = artifact_root / "final_cf_fix/predictions/corrected_recent_v4_test.parquet"
    v3 = _read_prediction(v3_path, test_index, TARGET_COLS)
    v4 = _read_prediction(v4_path, test_index, TARGET_COLS)
    strict = v3.copy()
    diagnostic = v4.copy()
    output_paths: list[Path] = []
    training: dict[str, Any] = {}
    for group, name in selected.items():
        combined = pd.concat([train_features[group], test_features[group]], axis=0)
        fit_start = YEAR_2023_START if group == "kpx_group_3" else YEAR_2022_START
        prediction, record, model_paths = _fit_predict_group(
            group=group,
            labels=labels,
            features=combined,
            mapped_columns=mapped_columns,
            fit_start=fit_start,
            fit_end=YEAR_2024_END,
            application_index=test_index,
            model_dir=out_dir / "models",
            model_prefix="final",
        )
        output_paths.extend(model_paths)
        training[group] = record
        base_v3 = v3[group].copy()
        base_v3.attrs["capacity_kwh"] = CAPACITY_KWH[group]
        base_v4 = v4[group].copy()
        base_v4.attrs["capacity_kwh"] = CAPACITY_KWH[group]
        strict[group] = _candidate_values(name, prediction, base_v3)
        diagnostic[group] = _candidate_values(name, prediction, base_v4)
        pred_path = out_dir / "predictions" / f"final_{group}.parquet"
        prediction[name] = strict[group]
        prediction[f"{name}__v4"] = diagnostic[group]
        _atomic_parquet(prediction, pred_path)
        output_paths.append(pred_path)

    strict_path = out_dir / "predictions/full_feature_year_qm_strict_v3_2025.parquet"
    diagnostic_path = out_dir / "predictions/full_feature_year_qm_v4_transfer_2025.parquet"
    _atomic_parquet(strict, strict_path)
    _atomic_parquet(diagnostic, diagnostic_path)
    output_paths.extend([strict_path, diagnostic_path])
    csv_records: dict[str, Any] = {}
    for name, frame in (
        ("full_feature_year_qm_strict_v3_2025.csv", strict),
        ("full_feature_year_qm_v4_transfer_2025.csv", diagnostic),
    ):
        output = sample.copy()
        for group in TARGET_COLS:
            output[group] = frame[group].to_numpy(dtype=float)
        path = out_dir / name
        _atomic_csv(output, path)
        output_paths.append(path)
        csv_records[name] = _verify_submission(path, sample, frame)
    manifest = {
        "schema_version": 1,
        "artifact_type": "full_feature_year_qm_final",
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "selected_by_frozen_priority": selected,
        "training": training,
        "strict_v3_candidate": True,
        "v4_transfer_selection_unsafe": True,
        "csv_verification": csv_records,
        "inputs": [
            describe_file(out_dir / "stage2_lock.json"),
            describe_file(sample_path),
            describe_file(v3_path),
            describe_file(v4_path),
            *[describe_file(cache_dir / f"{group}_weather_test.parquet") for group in TARGET_COLS],
        ],
        "outputs": [describe_file(path) for path in output_paths],
        "leaderboard_score_claim": False,
    }
    _write_json(out_dir / "manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config, mapped_columns = _verify_contract(args.preregister, args.mapped_columns)
    if args.stage in ("stage1", "all"):
        lock = _stage1(
            raw_dir=args.raw_dir,
            artifact_root=args.artifact_root,
            out_dir=args.out_dir,
            preregister_path=args.preregister,
            mapped_columns_path=args.mapped_columns,
            config=config,
            mapped_columns=mapped_columns,
        )
        print(json.dumps({"stage1_locked_pairs": lock["locked_pairs"]}, indent=2))
        if args.stage == "all" and not lock["locked_pairs"]:
            return
    if args.stage in ("stage2", "all"):
        lock2 = _stage2(
            raw_dir=args.raw_dir,
            cache_dir=args.cache_dir,
            artifact_root=args.artifact_root,
            out_dir=args.out_dir,
            mapped_columns=mapped_columns,
        )
        print(json.dumps({"stage2_promoted_pairs": lock2.get("promoted_pairs", [])}, indent=2))
        if args.stage == "all" and not lock2.get("selected_by_frozen_priority"):
            return
    if args.stage in ("final", "all"):
        final = _final(
            raw_dir=args.raw_dir,
            cache_dir=args.cache_dir,
            artifact_root=args.artifact_root,
            out_dir=args.out_dir,
            mapped_columns=mapped_columns,
        )
        print(json.dumps({"submission_created": final.get("artifact_type") == "full_feature_year_qm_final"}, indent=2))


if __name__ == "__main__":
    main()

"""Run the preregistered strict-forward model-zoo diversity experiment.

Stage 1 is completed and atomically locked using only pre-2024 rows before this
script loads any 2024 label, feature, or candidate prediction.  Stage 2 fits
only the one Stage-1-selected model per group; 2024 can confirm or reject that
fixed choice but can never reselect it.  A final 2025 submission is written
only when at least one group passes all fixed 2024 confirmation segments.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.manifest import make_manifest, sha256_file, write_json_atomic  # noqa: E402
from src.metric import CAPACITY_KWH, TARGET_COLS, group_metrics  # noqa: E402
from src.models import TabularRegressor  # noqa: E402


EXPECTED_TRAIN_INDEX = pd.date_range(
    "2022-01-01 01:00:00",
    "2025-01-01 00:00:00",
    freq="h",
    name="forecast_kst_dtm",
)
PRE2024_INDEX = pd.date_range(
    "2022-01-01 01:00:00",
    "2024-01-01 00:00:00",
    freq="h",
    name="forecast_kst_dtm",
)
INDEX_2022 = pd.date_range(
    "2022-01-01 01:00:00",
    "2023-01-01 00:00:00",
    freq="h",
    name="forecast_kst_dtm",
)
INDEX_2023 = pd.date_range(
    "2023-01-01 01:00:00",
    "2024-01-01 00:00:00",
    freq="h",
    name="forecast_kst_dtm",
)
INDEX_2023_H1 = pd.date_range(
    "2023-01-01 01:00:00",
    "2023-07-01 00:00:00",
    freq="h",
    name="forecast_kst_dtm",
)
INDEX_2023_H2 = pd.date_range(
    "2023-07-01 01:00:00",
    "2024-01-01 00:00:00",
    freq="h",
    name="forecast_kst_dtm",
)
INDEX_2023_Q3 = pd.date_range(
    "2023-07-01 01:00:00",
    "2023-10-01 00:00:00",
    freq="h",
    name="forecast_kst_dtm",
)
INDEX_2023_Q4 = pd.date_range(
    "2023-10-01 01:00:00",
    "2024-01-01 00:00:00",
    freq="h",
    name="forecast_kst_dtm",
)
INDEX_2024 = pd.date_range(
    "2024-01-01 01:00:00",
    "2025-01-01 00:00:00",
    freq="h",
    name="forecast_kst_dtm",
)
INDEX_2024_H1 = pd.date_range(
    "2024-01-01 01:00:00",
    "2024-07-01 00:00:00",
    freq="h",
    name="forecast_kst_dtm",
)
INDEX_2024_H2 = pd.date_range(
    "2024-07-01 01:00:00",
    "2025-01-01 00:00:00",
    freq="h",
    name="forecast_kst_dtm",
)
INDEX_2025 = pd.date_range(
    "2025-01-01 01:00:00",
    "2026-01-01 00:00:00",
    freq="h",
    name="forecast_kst_dtm",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(r"data/local/open"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/cache"))
    parser.add_argument("--oof-dir", type=Path, default=Path("artifacts/oof"))
    parser.add_argument(
        "--corrected-dir", type=Path, default=Path("artifacts/final_cf_fix")
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model_zoo_diversity_preregister.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/postgate/model_zoo_diversity"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--stage1-only",
        action="store_true",
        help="stop after persisting the pre-2024 selection lock",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate config and pre-2024 schemas without fitting or reading 2024",
    )
    return parser.parse_args(argv)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    return value


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        frame.to_parquet(temporary, index=True, compression="snappy")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_joblib(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        joblib.dump(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_submission(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        frame.to_csv(
            temporary,
            index=False,
            encoding="utf-8-sig",
            float_format="%.6f",
            lineterminator="\n",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    required = {
        "selection_uses_2024",
        "seed",
        "cpu_only",
        "n_jobs",
        "wrapper",
        "candidates_in_fixed_order",
        "resolved_wrapper_defaults",
        "blend",
        "stage1_selection_rule",
        "stage2",
        "final_fit",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"preregister missing fields: {missing}")
    if config["selection_uses_2024"] is not False:
        raise ValueError("selection_uses_2024 must be false")
    if config["cpu_only"] is not True:
        raise ValueError("this experiment is preregistered CPU-only")
    if config["wrapper"]["runtime_reduction_from_wrapper_defaults"] is not False:
        raise ValueError("unexpected runtime reduction in preregister")
    candidates = tuple(config["candidates_in_fixed_order"])
    expected = (
        "lgbm_huber",
        "xgb_l1",
        "xgb_pseudohuber",
        "catboost_mae",
        "catboost_huber",
        "extra_trees_l1",
        "hist_gbdt_l1",
    )
    if candidates != expected:
        raise ValueError(f"candidate order differs: {candidates!r}")
    if set(config["resolved_wrapper_defaults"]) != set(candidates):
        raise ValueError("resolved default snapshots differ from candidate set")
    blend = config["blend"]
    if not np.isclose(float(blend["locked_v3_weight"]), 0.9):
        raise ValueError("locked blend weight must remain 0.9")
    if not np.isclose(float(blend["candidate_weight"]), 0.1):
        raise ValueError("candidate blend weight must remain 0.1")
    return config


def _validate_wrapper_defaults(config: Mapping[str, Any]) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for kind in config["candidates_in_fixed_order"]:
        wrapper = TabularRegressor(
            kind=kind,
            params=dict(config["wrapper"]["params_override"]),
            seed=int(config["seed"]),
            n_jobs=int(config["n_jobs"]),
            device="cpu",
            early_stopping_rounds=None,
            fallback_to_cpu=bool(config["wrapper"]["fallback_to_cpu"]),
            validate_time_order=bool(config["wrapper"]["validate_time_order"]),
        )
        wrapper._validate_options()
        backend = wrapper._build_model(
            backend_device="cpu", use_early_stopping=False
        )
        actual = backend.get_params()
        expected = config["resolved_wrapper_defaults"][kind]
        differences: dict[str, Any] = {}
        for key, expected_value in expected.items():
            actual_value = actual.get(key)
            equal = (
                np.isclose(actual_value, expected_value)
                if isinstance(expected_value, (int, float))
                and not isinstance(expected_value, bool)
                else actual_value == expected_value
            )
            if not bool(equal):
                differences[key] = {
                    "expected": expected_value,
                    "actual": actual_value,
                }
        if differences:
            raise ValueError(f"{kind} wrapper defaults drifted: {differences}")
        resolved[kind] = {
            "backend_class": type(backend).__name__,
            "snapshot_fields_verified": sorted(expected),
        }
        del backend, wrapper
    return resolved


def _validate_index(frame: pd.DataFrame, expected: pd.DatetimeIndex, context: str) -> None:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"{context} must have a DatetimeIndex")
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if not frame.index.equals(expected):
        raise ValueError(
            f"{context} index differs: rows={len(frame)}, "
            f"range={frame.index.min()}..{frame.index.max()}"
        )
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{context} index must be unique and sorted")


def _read_pre2024_labels(raw_dir: Path) -> tuple[pd.DataFrame, Path]:
    path = raw_dir / "train/train_labels.csv"
    frame = pd.read_csv(
        path,
        nrows=len(PRE2024_INDEX),
        encoding="utf-8-sig",
        usecols=["kst_dtm", *TARGET_COLS],
    )
    frame.index = pd.DatetimeIndex(
        pd.to_datetime(frame.pop("kst_dtm"), errors="raise"),
        name="forecast_kst_dtm",
    )
    _validate_index(frame, PRE2024_INDEX, "pre-2024 labels")
    return frame.astype(float), path


def _read_pre2024_features(cache_dir: Path, group: str) -> tuple[pd.DataFrame, Path]:
    import pyarrow.parquet as pq

    path = cache_dir / f"{group}_weather_train.parquet"
    batches = pq.ParquetFile(path).iter_batches(batch_size=len(PRE2024_INDEX))
    first = next(batches, None)
    if first is None or first.num_rows != len(PRE2024_INDEX):
        raise ValueError(f"{path} lacks exact bounded pre-2024 batch")
    frame = first.to_pandas()
    _validate_index(frame, PRE2024_INDEX, f"{group} pre-2024 features")
    if not frame.columns.is_unique:
        raise ValueError(f"{group} has duplicate feature columns")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes):
        raise TypeError(f"{group} features must all be numeric")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError(f"{group} pre-2024 features contain non-finite values")
    return frame, path


def _read_full_labels(raw_dir: Path) -> tuple[pd.DataFrame, Path]:
    path = raw_dir / "train/train_labels.csv"
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if tuple(frame.columns) != ("kst_dtm", *TARGET_COLS):
        raise ValueError(f"unexpected label columns: {tuple(frame.columns)!r}")
    frame.index = pd.DatetimeIndex(
        pd.to_datetime(frame.pop("kst_dtm"), errors="raise"),
        name="forecast_kst_dtm",
    )
    _validate_index(frame, EXPECTED_TRAIN_INDEX, "full labels")
    return frame.astype(float), path


def _read_full_features(cache_dir: Path, group: str) -> tuple[pd.DataFrame, Path]:
    path = cache_dir / f"{group}_weather_train.parquet"
    frame = pd.read_parquet(path)
    _validate_index(frame, EXPECTED_TRAIN_INDEX, f"{group} full features")
    if not frame.columns.is_unique or not all(
        pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes
    ):
        raise TypeError(f"{group} full feature schema is invalid")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError(f"{group} full features contain non-finite values")
    return frame, path


def _read_test_features(cache_dir: Path, group: str) -> tuple[pd.DataFrame, Path]:
    path = cache_dir / f"{group}_weather_test.parquet"
    frame = pd.read_parquet(path)
    _validate_index(frame, INDEX_2025, f"{group} test features")
    if not frame.columns.is_unique or not all(
        pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes
    ):
        raise TypeError(f"{group} test feature schema is invalid")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError(f"{group} test features contain non-finite values")
    return frame, path


def _read_prediction(
    path: Path,
    expected_index: pd.DatetimeIndex,
    expected_columns: Sequence[str],
    *,
    allow_extra_columns: bool = False,
) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if allow_extra_columns:
        missing = [column for column in expected_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")
        frame = frame.loc[:, list(expected_columns)]
    elif tuple(frame.columns) != tuple(expected_columns):
        raise ValueError(
            f"{path} columns={tuple(frame.columns)!r}, "
            f"expected={tuple(expected_columns)!r}"
        )
    if isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
        if not frame.index.equals(expected_index):
            raise ValueError(f"{path} DatetimeIndex differs")
    elif len(frame) == len(expected_index) and isinstance(frame.index, pd.RangeIndex):
        frame.index = expected_index
    else:
        raise ValueError(f"{path} has unusable prediction index")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError(f"{path} contains non-finite predictions")
    return frame.astype(float)


def _stage1_baselines(oof_dir: Path) -> tuple[dict[str, pd.Series], list[Path]]:
    dev_path = oof_dir / "dev2023_locked_v3.parquet"
    dev = _read_prediction(dev_path, INDEX_2023, TARGET_COLS[:2])
    g3_path = oof_dir / "g3dev2023h2_candidates.parquet"
    raw = _read_prediction(
        g3_path,
        INDEX_2023_H2,
        ("l1", "q07", "shared_l1", "shared_q07", "top200q07", "ewq06"),
        allow_extra_columns=True,
    )
    weighted = (
        0.20 * raw["q07"]
        + 0.075 * raw["shared_l1"]
        + 0.425 * raw["shared_q07"]
        + 0.025 * raw["top200q07"]
        + 0.275 * raw["ewq06"]
    )
    g3 = pd.Series(
        np.clip(
            1.25 * weighted.to_numpy(dtype=float) - 1200.0,
            0.0,
            1.02 * CAPACITY_KWH["kpx_group_3"],
        ),
        index=INDEX_2023_H2,
        name="kpx_group_3",
    )
    return {
        "kpx_group_1": dev["kpx_group_1"],
        "kpx_group_2": dev["kpx_group_2"],
        "kpx_group_3": g3,
    }, [dev_path, g3_path]


def _fit_model(
    kind: str,
    features: pd.DataFrame,
    actual_kwh: pd.Series,
    capacity: float,
    config: Mapping[str, Any],
) -> tuple[TabularRegressor, dict[str, Any]]:
    eligible = actual_kwh.notna() & (actual_kwh >= 0.10 * capacity)
    x = features.loc[eligible]
    y = actual_kwh.loc[eligible] / capacity
    if not x.index.equals(y.index):
        raise AssertionError("eligible training features/target differ")
    if len(x) < 100:
        raise ValueError(f"too few eligible rows for {kind}: {len(x)}")
    started = time.perf_counter()
    model = TabularRegressor(
        kind=kind,
        params=dict(config["wrapper"]["params_override"]),
        seed=int(config["seed"]),
        n_jobs=int(config["n_jobs"]),
        device="cpu",
        early_stopping_rounds=None,
        fallback_to_cpu=bool(config["wrapper"]["fallback_to_cpu"]),
        validate_time_order=bool(config["wrapper"]["validate_time_order"]),
    )
    model.fit(x, y)
    elapsed = time.perf_counter() - started
    importance = model.feature_importances_
    finite = np.isfinite(importance)
    top_importance: list[dict[str, Any]] = []
    if finite.any():
        order = np.argsort(-importance[finite])[:20]
        finite_positions = np.flatnonzero(finite)[order]
        top_importance = [
            {
                "feature": str(features.columns[position]),
                "importance": float(importance[position]),
            }
            for position in finite_positions
        ]
    metadata = {
        "kind": kind,
        "eligible_train_rows": int(len(x)),
        "train_start": str(x.index.min()),
        "train_end": str(x.index.max()),
        "target_unit": "capacity_factor",
        "fit_seconds": float(elapsed),
        "device_used": model.device_used_,
        "best_iteration": model.best_iteration_,
        "feature_count": int(features.shape[1]),
        "backend_class": type(model.model_).__name__,
        "top_native_feature_importance": top_importance,
    }
    return model, metadata


def _blend(
    locked_kwh: pd.Series,
    candidate_cf: np.ndarray,
    capacity: float,
    config: Mapping[str, Any],
) -> tuple[pd.Series, pd.Series]:
    candidate_kwh = pd.Series(
        np.asarray(candidate_cf, dtype=float) * capacity,
        index=locked_kwh.index,
        name="candidate_kwh",
    )
    blend = config["blend"]
    blended = pd.Series(
        np.clip(
            float(blend["locked_v3_weight"])
            * locked_kwh.to_numpy(dtype=float)
            + float(blend["candidate_weight"])
            * candidate_kwh.to_numpy(dtype=float),
            0.0,
            1.02 * capacity,
        ),
        index=locked_kwh.index,
        name="blended_kwh",
    )
    if not np.isfinite(candidate_kwh.to_numpy()).all() or not np.isfinite(
        blended.to_numpy()
    ).all():
        raise ValueError("candidate/blended prediction contains non-finite values")
    return candidate_kwh, blended


def _metrics(actual: pd.Series, prediction: pd.Series, group: str) -> dict[str, Any]:
    metric = group_metrics(
        actual,
        prediction,
        CAPACITY_KWH[group],
        group_name=group,
    )
    result = metric.as_dict()
    result["score"] = 0.5 * (metric.one_minus_nmae + metric.ficr)
    return result


def _segment_evaluation(
    actual: pd.Series,
    locked: pd.Series,
    blended: pd.Series,
    segments: Mapping[str, pd.DatetimeIndex],
    group: str,
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for name, index in segments.items():
        baseline = _metrics(actual.loc[index], locked.loc[index], group)
        candidate = _metrics(actual.loc[index], blended.loc[index], group)
        records[name] = {
            "locked_v3": baseline,
            "blend": candidate,
            "delta_score": float(candidate["score"] - baseline["score"]),
        }
    return records


def _select_stage1(records: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for group in TARGET_COLS:
        eligible: list[str] = []
        minimum_delta: dict[str, float] = {}
        for candidate, result in records[group].items():
            deltas = [
                float(value["delta_score"])
                for value in result["segments"].values()
            ]
            minimum_delta[candidate] = float(min(deltas))
            if all(delta > 0.0 for delta in deltas):
                eligible.append(candidate)
        # sorted() and max()'s first-maximum behaviour implement the exact
        # lexicographic tie break without a hidden secondary score criterion.
        selected = (
            max(sorted(eligible), key=lambda name: minimum_delta[name])
            if eligible
            else "identity"
        )
        output[group] = {
            "eligible_all_segments_positive": eligible,
            "minimum_segment_delta": minimum_delta,
            "selected_candidate": selected,
        }
    return output


def _model_prediction_frame(
    locked: pd.Series,
    candidate: pd.Series,
    blended: pd.Series,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "locked_v3_kwh": locked,
            "candidate_kwh": candidate,
            "blended_kwh": blended,
        },
        index=locked.index,
    )


def _verify_submission(path: Path, expected: pd.DataFrame) -> dict[str, Any]:
    with path.open("rb") as stream:
        if stream.read(3) != b"\xef\xbb\xbf":
            raise AssertionError("submission lacks UTF-8-SIG BOM")
    observed = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"forecast_id": "string", "forecast_kst_dtm": "string"},
    )
    if tuple(observed.columns) != tuple(expected.columns) or len(observed) != 8760:
        raise AssertionError("submission schema/row count differs")
    for column in ("forecast_id", "forecast_kst_dtm"):
        if not observed[column].equals(expected[column].astype("string")):
            raise AssertionError(f"submission {column} alignment differs")
    maximum_difference = float(
        np.max(
            np.abs(
                observed.loc[:, list(TARGET_COLS)].to_numpy(dtype=float)
                - expected.loc[:, list(TARGET_COLS)].to_numpy(dtype=float)
            )
        )
    )
    if maximum_difference > 5.1e-7:
        raise AssertionError(f"submission rounding mismatch: {maximum_difference}")
    values = observed.loc[:, list(TARGET_COLS)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise AssertionError("submission contains non-finite target values")
    for position, group in enumerate(TARGET_COLS):
        if np.min(values[:, position]) < -1e-9 or np.max(
            values[:, position]
        ) > 1.02 * CAPACITY_KWH[group] + 1e-6:
            raise AssertionError(f"submission {group} violates clip")
    return {
        "rows": int(len(observed)),
        "utf8_sig": True,
        "max_readback_abs_diff": maximum_difference,
        "sha256": sha256_file(path),
    }


def _assert_output_policy(out_dir: Path, overwrite: bool) -> None:
    guarded = (
        out_dir / "stage1_lock.json",
        out_dir / "stage2_lock.json",
        out_dir / "results.json",
        out_dir / "manifest.json",
        out_dir / "model_zoo_diversity_2025.csv",
    )
    existing = [path for path in guarded if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "outputs already exist; use --overwrite for this experiment: "
            + ", ".join(str(path) for path in existing)
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = _read_config(args.config)
    config_hash = sha256_file(args.config)
    default_audit = _validate_wrapper_defaults(config)
    _assert_output_policy(args.out_dir, bool(args.overwrite))

    # Strict boundary: only bounded pre-2024 reads occur above the Stage-1 lock.
    labels_pre2024, labels_path = _read_pre2024_labels(args.raw_dir)
    features_pre2024: dict[str, pd.DataFrame] = {}
    train_cache_paths: list[Path] = []
    for group in TARGET_COLS:
        frame, path = _read_pre2024_features(args.cache_dir, group)
        features_pre2024[group] = frame
        train_cache_paths.append(path)
    schema = features_pre2024[TARGET_COLS[0]].columns
    for group in TARGET_COLS[1:]:
        if not features_pre2024[group].columns.equals(schema):
            raise ValueError("group feature schemas differ before 2024")
    baselines_stage1, baseline_stage1_paths = _stage1_baselines(args.oof_dir)
    print(
        "pre-2024 boundary validated:",
        {
            "rows": len(labels_pre2024),
            "features": len(schema),
            "config_sha256": config_hash,
        },
        flush=True,
    )
    if args.dry_run:
        print("dry-run complete: no fit and no 2024 data read", flush=True)
        return 0

    stage1_specs = {
        "kpx_group_1": {
            "train": INDEX_2022,
            "valid": INDEX_2023,
            "segments": {
                "full": INDEX_2023,
                "h1": INDEX_2023_H1,
                "h2": INDEX_2023_H2,
            },
        },
        "kpx_group_2": {
            "train": INDEX_2022,
            "valid": INDEX_2023,
            "segments": {
                "full": INDEX_2023,
                "h1": INDEX_2023_H1,
                "h2": INDEX_2023_H2,
            },
        },
        "kpx_group_3": {
            "train": INDEX_2023_H1,
            "valid": INDEX_2023_H2,
            "segments": {
                "full": INDEX_2023_H2,
                "q3": INDEX_2023_Q3,
                "q4": INDEX_2023_Q4,
            },
        },
    }
    stage1_records: dict[str, dict[str, Any]] = {
        group: {} for group in TARGET_COLS
    }
    stage1_outputs: list[Path] = []
    progress_path = args.out_dir / "stage1_progress.json"
    for group in TARGET_COLS:
        spec = stage1_specs[group]
        train_index = spec["train"]
        valid_index = spec["valid"]
        if len(train_index.intersection(valid_index)) or train_index.max() >= valid_index.min():
            raise AssertionError(f"{group} Stage1 chronology is invalid")
        locked = baselines_stage1[group]
        if not locked.index.equals(valid_index):
            raise ValueError(f"{group} Stage1 locked baseline index differs")
        for kind in config["candidates_in_fixed_order"]:
            print(
                f"stage1 start group={group} candidate={kind} "
                f"train={len(train_index)} valid={len(valid_index)}",
                flush=True,
            )
            model, metadata = _fit_model(
                kind,
                features_pre2024[group].loc[train_index],
                labels_pre2024.loc[train_index, group],
                CAPACITY_KWH[group],
                config,
            )
            raw_cf = model.predict(features_pre2024[group].loc[valid_index])
            candidate, blended = _blend(
                locked,
                raw_cf,
                CAPACITY_KWH[group],
                config,
            )
            segments = _segment_evaluation(
                labels_pre2024.loc[valid_index, group],
                locked,
                blended,
                spec["segments"],
                group,
            )
            path = args.out_dir / "stage1/predictions" / f"{group}__{kind}.parquet"
            _atomic_parquet(_model_prediction_frame(locked, candidate, blended), path)
            stage1_outputs.append(path)
            stage1_records[group][kind] = {
                "status": "completed",
                "model": metadata,
                "segments": segments,
                "prediction_path": str(path),
                "prediction_sha256": sha256_file(path),
            }
            write_json_atomic(
                progress_path,
                {
                    "warning": "pre-2024 Stage1 only; selection not yet locked",
                    "config_sha256": config_hash,
                    "completed": _json_ready(stage1_records),
                },
                overwrite=True,
            )
            print(
                f"stage1 done group={group} candidate={kind} "
                f"seconds={metadata['fit_seconds']:.2f} deltas="
                f"{ {name: round(value['delta_score'], 9) for name, value in segments.items()} }",
                flush=True,
            )
            del model
            gc.collect()

    selection = _select_stage1(stage1_records)
    stage1_lock = {
        "lock_stage": "completed_before_any_2024_read",
        "config_path": str(args.config),
        "config_sha256": config_hash,
        "selection_rule": config["stage1_selection_rule"],
        "default_audit": default_audit,
        "records": stage1_records,
        "selection": selection,
        "2024_data_read_before_lock": False,
    }
    stage1_lock_path = args.out_dir / "stage1_lock.json"
    write_json_atomic(stage1_lock_path, stage1_lock, overwrite=bool(args.overwrite))
    stage1_lock_hash = sha256_file(stage1_lock_path)
    print(
        "stage1 locked before 2024 read:",
        {
            group: value["selected_candidate"]
            for group, value in selection.items()
        },
        f"sha256={stage1_lock_hash}",
        flush=True,
    )
    if args.stage1_only:
        print("stage1-only complete", flush=True)
        return 0

    # Stage 2 begins only after the immutable Stage-1 lock exists and is hashed.
    labels_full, _ = _read_full_labels(args.raw_dir)
    features_full: dict[str, pd.DataFrame] = {}
    for group in TARGET_COLS:
        frame, _ = _read_full_features(args.cache_dir, group)
        if not frame.columns.equals(schema):
            raise ValueError(f"{group} full feature schema differs from Stage1")
        if not frame.loc[PRE2024_INDEX].equals(features_pre2024[group]):
            raise ValueError(f"{group} pre-2024 feature readback differs")
        features_full[group] = frame
    gate_path = args.oof_dir / "gate2024_locked_v3_cf_fix.parquet"
    gate_baseline = _read_prediction(gate_path, INDEX_2024, TARGET_COLS)
    stage2_records: dict[str, Any] = {}
    stage2_outputs: list[Path] = []
    selected_models: dict[str, str] = {
        group: selection[group]["selected_candidate"] for group in TARGET_COLS
    }
    for group in TARGET_COLS:
        kind = selected_models[group]
        if kind == "identity":
            stage2_records[group] = {
                "selected_stage1": "identity",
                "evaluated": False,
                "promoted": False,
                "reason": "no Stage1 candidate improved all required segments",
            }
            continue
        train_index = PRE2024_INDEX if group != "kpx_group_3" else INDEX_2023
        if train_index.max() >= INDEX_2024.min():
            raise AssertionError(f"{group} Stage2 chronology is invalid")
        print(
            f"stage2 fixed start group={group} candidate={kind}", flush=True
        )
        model, metadata = _fit_model(
            kind,
            features_full[group].loc[train_index],
            labels_full.loc[train_index, group],
            CAPACITY_KWH[group],
            config,
        )
        raw_cf = model.predict(features_full[group].loc[INDEX_2024])
        candidate, blended = _blend(
            gate_baseline[group], raw_cf, CAPACITY_KWH[group], config
        )
        segments = _segment_evaluation(
            labels_full.loc[INDEX_2024, group],
            gate_baseline[group],
            blended,
            {"full": INDEX_2024, "h1": INDEX_2024_H1, "h2": INDEX_2024_H2},
            group,
        )
        promoted = all(
            float(value["delta_score"]) > 0.0 for value in segments.values()
        )
        path = args.out_dir / "stage2/predictions" / f"{group}__{kind}.parquet"
        _atomic_parquet(
            _model_prediction_frame(gate_baseline[group], candidate, blended), path
        )
        stage2_outputs.append(path)
        stage2_records[group] = {
            "selected_stage1": kind,
            "evaluated": True,
            "reselection_performed": False,
            "model": metadata,
            "segments": segments,
            "promoted": promoted,
            "prediction_path": str(path),
            "prediction_sha256": sha256_file(path),
        }
        print(
            f"stage2 fixed done group={group} candidate={kind} "
            f"promoted={promoted} deltas="
            f"{ {name: round(value['delta_score'], 9) for name, value in segments.items()} }",
            flush=True,
        )
        del model
        gc.collect()

    promoted_groups = [
        group for group in TARGET_COLS if stage2_records[group]["promoted"]
    ]
    stage2_lock = {
        "stage1_lock_path": str(stage1_lock_path),
        "stage1_lock_sha256": stage1_lock_hash,
        "fixed_selection": selected_models,
        "records": stage2_records,
        "promoted_groups": promoted_groups,
        "2024_reselection_performed": False,
        "confirmation_rule": config["stage2"]["confirmation_rule"],
        "2024_status": "consumed post-gate confirmation, not independent validation",
    }
    stage2_lock_path = args.out_dir / "stage2_lock.json"
    write_json_atomic(stage2_lock_path, stage2_lock, overwrite=bool(args.overwrite))
    print(f"stage2 locked promoted={promoted_groups}", flush=True)

    final_outputs: list[Path] = []
    final_models: dict[str, Any] = {}
    final_metadata: dict[str, Any] = {}
    submission_check: dict[str, Any] | None = None
    candidate_path: Path | None = None
    final_prediction_path: Path | None = None
    sample_path: Path | None = None
    test_cache_paths: list[Path] = []
    final_base_path: Path | None = None
    if promoted_groups:
        sample_path = args.raw_dir / "sample_submission.csv"
        sample = pd.read_csv(
            sample_path,
            encoding="utf-8-sig",
            dtype={"forecast_id": "string", "forecast_kst_dtm": "string"},
        )
        expected_columns = ("forecast_id", "forecast_kst_dtm", *TARGET_COLS)
        if tuple(sample.columns) != expected_columns or len(sample) != len(INDEX_2025):
            raise ValueError("sample submission schema/rows differ")
        sample_index = pd.DatetimeIndex(
            pd.to_datetime(sample["forecast_kst_dtm"], errors="raise"),
            name="forecast_kst_dtm",
        )
        if not sample_index.equals(INDEX_2025):
            raise ValueError("sample submission time horizon differs")
        final_base_path = args.corrected_dir / "predictions/corrected_v3_test.parquet"
        final_prediction = _read_prediction(final_base_path, INDEX_2025, TARGET_COLS)
        test_features: dict[str, pd.DataFrame] = {}
        for group in promoted_groups:
            frame, path = _read_test_features(args.cache_dir, group)
            if not frame.columns.equals(schema):
                raise ValueError(f"{group} test feature schema differs")
            test_features[group] = frame
            test_cache_paths.append(path)
        for group in promoted_groups:
            kind = selected_models[group]
            full_train_index = (
                EXPECTED_TRAIN_INDEX if group != "kpx_group_3" else EXPECTED_TRAIN_INDEX[EXPECTED_TRAIN_INDEX >= pd.Timestamp("2023-01-01 01:00:00")]
            )
            print(f"final fit group={group} candidate={kind}", flush=True)
            model, metadata = _fit_model(
                kind,
                features_full[group].loc[full_train_index],
                labels_full.loc[full_train_index, group],
                CAPACITY_KWH[group],
                config,
            )
            raw_cf = model.predict(test_features[group])
            _, blended = _blend(
                final_prediction[group], raw_cf, CAPACITY_KWH[group], config
            )
            final_prediction[group] = blended
            final_models[group] = model
            final_metadata[group] = metadata
            model_path = args.out_dir / "models" / f"{group}__{kind}.joblib"
            _atomic_joblib(model, model_path)
            final_outputs.append(model_path)
        final_prediction_path = args.out_dir / "predictions/model_zoo_diversity_2025.parquet"
        _atomic_parquet(final_prediction, final_prediction_path)
        final_outputs.append(final_prediction_path)
        submission = sample.copy()
        for group in TARGET_COLS:
            submission[group] = final_prediction[group].to_numpy(dtype=float)
        candidate_path = args.out_dir / "model_zoo_diversity_2025.csv"
        _atomic_submission(submission, candidate_path)
        submission_check = _verify_submission(candidate_path, submission)
        final_outputs.append(candidate_path)

    results = {
        "warning": (
            "2024 is consumed post-gate confirmation data and is not an untouched "
            "or independent validation set. No leaderboard score is claimed."
        ),
        "config_path": str(args.config),
        "config_sha256": config_hash,
        "wrapper_default_audit": default_audit,
        "strict_boundaries": {
            "stage1_lock_before_2024_read": True,
            "stage1_lock_sha256": stage1_lock_hash,
            "stage2_reselection_performed": False,
            "outer_validation_used_for_early_stopping": False,
            "target_unit": "capacity_factor",
            "train_filter": "actual >= 0.10 * capacity",
        },
        "stage1": stage1_lock,
        "stage2": stage2_lock,
        "final_2025": {
            "promoted_groups": promoted_groups,
            "fit_metadata": final_metadata,
            "prediction_path": str(final_prediction_path) if final_prediction_path else None,
            "submission_path": str(candidate_path) if candidate_path else None,
            "submission_check": submission_check,
            "score_claim": False,
        },
    }
    results_path = args.out_dir / "results.json"
    write_json_atomic(results_path, results, overwrite=bool(args.overwrite))

    inputs: list[Path] = list(
        dict.fromkeys(
            Path(path).resolve()
            for path in (
                args.config,
                labels_path,
                *train_cache_paths,
                *baseline_stage1_paths,
                gate_path,
                *(
                    [sample_path, final_base_path, *test_cache_paths]
                    if promoted_groups
                    else []
                ),
            )
            if path is not None
        )
    )
    outputs = [
        *stage1_outputs,
        progress_path,
        stage1_lock_path,
        *stage2_outputs,
        stage2_lock_path,
        *final_outputs,
        results_path,
    ]
    manifest = make_manifest(
        artifact_type="baram_strict_forward_model_zoo_diversity",
        parameters={
            "config_sha256": config_hash,
            "candidates": config["candidates_in_fixed_order"],
            "wrapper_params_override": config["wrapper"]["params_override"],
            "cpu_only": True,
            "selection_uses_2024": False,
            "leaderboard_score_claim": False,
            "untouched_gate_claim": False,
        },
        input_files=inputs,
        output_files=outputs,
        results={
            "stage1_selection": selected_models,
            "stage1_lock_sha256": stage1_lock_hash,
            "stage2_promoted_groups": promoted_groups,
            "stage2_records": stage2_records,
            "submission_sha256": (
                None if submission_check is None else submission_check["sha256"]
            ),
        },
        project_dir=PROJECT_DIR,
    )
    manifest_path = args.out_dir / "manifest.json"
    write_json_atomic(manifest_path, manifest, overwrite=bool(args.overwrite))
    print(f"complete results={results_path} manifest={manifest_path}", flush=True)
    if submission_check is not None:
        print(
            f"submission={candidate_path} sha256={submission_check['sha256']}",
            flush=True,
        )
    else:
        print("no group promoted; no 2025 submission written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

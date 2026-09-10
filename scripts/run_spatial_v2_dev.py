"""Compare v1 versus v1+turbine-spatial-v2 on locked historical folds.

Validation design
-----------------
* groups 1/2: train 2022, validate 2023;
* group 3: train 2023 H1, validate 2023 H2 (its 2022 labels are absent);
* labels/features at 2024-01-01 01:00 or later are never read;
* fixed-size models, no validation-driven early stopping or tuning.

Example:
    python scripts/run_spatial_v2_dev.py --raw-dir C:/data/open \
        --base-cache-dir artifacts/cache \
        --augmentation-dir artifacts/cache_v2 \
        --out-dir artifacts/experiments/spatial_v2
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lightgbm import LGBMRegressor, log_evaluation


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.manifest import make_manifest, write_json_atomic  # noqa: E402
from src.metric import CAPACITY_KWH, TARGET_COLS, group_metrics  # noqa: E402


DEV_ROWS = 17_520
LAST_ALLOWED_TIMESTAMP = pd.Timestamp("2024-01-01 00:00:00")
FIRST_FORBIDDEN_TIMESTAMP = pd.Timestamp("2024-01-01 01:00:00")
G12_SPLIT = pd.Timestamp("2023-01-01 01:00:00")
G3_START = pd.Timestamp("2023-01-01 01:00:00")
G3_SPLIT = pd.Timestamp("2023-07-01 01:00:00")
FORBIDDEN_FEATURE = re.compile(
    r"(^kpx_group_|scada|target|label|power_kw)", flags=re.IGNORECASE
)


@dataclass(frozen=True)
class BenchmarkConfig:
    n_estimators: int = 400
    learning_rate: float = 0.04
    num_leaves: int = 31
    min_child_samples: int = 30
    subsample: float = 0.80
    subsample_freq: int = 1
    colsample_bytree: float = 0.80
    reg_alpha: float = 0.05
    reg_lambda: float = 2.0
    seed: int = 42
    n_jobs: int = 7
    eligible_fraction: float = 0.10


OBJECTIVES: dict[str, dict[str, Any]] = {
    "l1": {"objective": "regression_l1"},
    "q07": {"objective": "quantile", "alpha": 0.70},
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Leakage-locked small-LightGBM comparison of v1 versus spatial-v2."
        )
    )
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--base-cache-dir", type=Path, required=True)
    parser.add_argument("--augmentation-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--n-jobs", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _read_labels(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "train" / "train_labels.csv"
    labels = pd.read_csv(
        path,
        usecols=["kst_dtm", *TARGET_COLS],
        nrows=DEV_ROWS,
        encoding="utf-8-sig",
        parse_dates=["kst_dtm"],
    ).set_index("kst_dtm")
    labels.index = pd.DatetimeIndex(labels.index, name="forecast_kst_dtm")
    if len(labels) != DEV_ROWS or labels.index.max() != LAST_ALLOWED_TIMESTAMP:
        raise ValueError("label read is not the exact pre-2024 17,520-row window")
    if labels.index.min() != pd.Timestamp("2022-01-01 01:00:00"):
        raise ValueError("unexpected first label timestamp")
    if not labels.index.is_monotonic_increasing or not labels.index.is_unique:
        raise ValueError("labels must have a unique chronological index")
    if (labels.index >= FIRST_FORBIDDEN_TIMESTAMP).any():
        raise AssertionError("2024 forecast labels entered the benchmark")
    return labels


def _read_bounded_parquet(path: Path) -> pd.DataFrame:
    """Decode only the first pre-2024 RecordBatch, never a future batch."""

    batches = pq.ParquetFile(path).iter_batches(batch_size=DEV_ROWS)
    batch = next(batches, None)
    if batch is None or batch.num_rows != DEV_ROWS:
        raise ValueError(f"{path}: expected a first batch of {DEV_ROWS} rows")
    frame = batch.to_pandas()
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if frame.index.max() != LAST_ALLOWED_TIMESTAMP:
        raise ValueError(f"{path}: bounded batch has an unexpected end timestamp")
    if (frame.index >= FIRST_FORBIDDEN_TIMESTAMP).any():
        raise AssertionError(f"{path}: 2024 forecast rows entered memory")
    if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError(f"{path}: feature index is not unique/chronological")
    if frame.isna().any().any() or not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"{path}: feature values are not finite")
    return frame


def _load_features(
    base_dir: Path, augmentation_dir: Path, labels: pd.DataFrame
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    list[Path],
]:
    base: dict[str, pd.DataFrame] = {}
    augmented_core: dict[str, pd.DataFrame] = {}
    augmented_sequence: dict[str, pd.DataFrame] = {}
    inputs: list[Path] = []
    for group in TARGET_COLS:
        base_path = base_dir / f"{group}_weather_train.parquet"
        aug_path = augmentation_dir / f"{group}_spatial_v2_train.parquet"
        for path in (base_path, aug_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        base_frame = _read_bounded_parquet(base_path)
        aug_frame = _read_bounded_parquet(aug_path)
        if not base_frame.index.equals(labels.index) or not aug_frame.index.equals(labels.index):
            raise ValueError(f"{group}: labels/v1/v2 timestamps do not align exactly")
        bad_aug = [column for column in aug_frame if FORBIDDEN_FEATURE.search(column)]
        if bad_aug:
            raise ValueError(f"{group}: target/SCADA-like augmentation: {bad_aug}")
        duplicated = set(base_frame).intersection(aug_frame)
        if duplicated:
            raise ValueError(f"{group}: duplicate v1/v2 columns: {sorted(duplicated)[:5]}")
        sequence_columns = [
            column for column in aug_frame if column.startswith("spv2seq__")
        ]
        if len(sequence_columns) != 60:
            raise ValueError(
                f"{group}: expected 60 explicitly ablatable spv2seq columns, "
                f"found {len(sequence_columns)}"
            )
        core_columns = [column for column in aug_frame if column not in sequence_columns]
        base[group] = base_frame.astype("float32")
        augmented_core[group] = pd.concat(
            [base[group], aug_frame[core_columns]], axis=1
        )
        augmented_sequence[group] = pd.concat([base[group], aug_frame], axis=1)
        inputs.extend([base_path, aug_path])
    return base, augmented_core, augmented_sequence, inputs


def _fold_masks(group: str, index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    if group in ("kpx_group_1", "kpx_group_2"):
        train = index < G12_SPLIT
        valid = index >= G12_SPLIT
        expected = (8_760, 8_760)
    else:
        train = (index >= G3_START) & (index < G3_SPLIT)
        valid = index >= G3_SPLIT
        expected = (4_344, 4_416)
    observed = (int(train.sum()), int(valid.sum()))
    if observed != expected or np.any(train & valid):
        raise AssertionError(f"{group}: fold shape {observed} != {expected}")
    if index[train].max() >= index[valid].min():
        raise AssertionError(f"{group}: training time is not strictly before validation")
    if index[valid].max() != LAST_ALLOWED_TIMESTAMP:
        raise AssertionError(f"{group}: validation ending timestamp changed")
    return np.asarray(train), np.asarray(valid)


def _make_model(config: BenchmarkConfig, objective_name: str) -> LGBMRegressor:
    objective = OBJECTIVES[objective_name]
    return LGBMRegressor(
        **objective,
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        num_leaves=config.num_leaves,
        min_child_samples=config.min_child_samples,
        subsample=config.subsample,
        subsample_freq=config.subsample_freq,
        colsample_bytree=config.colsample_bytree,
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        random_state=config.seed,
        n_jobs=config.n_jobs,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


def _metric_dict(actual: np.ndarray, prediction: np.ndarray, group: str) -> dict[str, Any]:
    values = group_metrics(
        actual,
        prediction,
        CAPACITY_KWH[group],
        group_name=group,
    )
    result = values.as_dict()
    result["score"] = 0.5 * (values.one_minus_nmae + values.ficr)
    return result


def _write_parquet_atomic(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        frame.to_parquet(temporary, compression="zstd", index=True)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _dump_joblib_atomic(model: LGBMRegressor, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        joblib.dump(model, temporary, compress=3)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _feature_importance(model: LGBMRegressor, columns: pd.Index) -> dict[str, Any]:
    gain = model.booster_.feature_importance(importance_type="gain")
    order = np.argsort(gain)[::-1]
    total = float(gain.sum())
    spatial_core = np.asarray([str(column).startswith("spv2__") for column in columns])
    sequence = np.asarray([str(column).startswith("spv2seq__") for column in columns])
    spatial_all = spatial_core | sequence
    return {
        "spatial_v2_gain_share": float(gain[spatial_all].sum() / total) if total else 0.0,
        "spatial_v2_nonzero_count": int(np.count_nonzero(gain[spatial_all])),
        "spatial_core_gain_share": float(gain[spatial_core].sum() / total) if total else 0.0,
        "run_sequence_gain_share": float(gain[sequence].sum() / total) if total else 0.0,
        "run_sequence_nonzero_count": int(np.count_nonzero(gain[sequence])),
        "top20": [
            {"feature": str(columns[position]), "gain": float(gain[position])}
            for position in order[:20]
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    raw_dir = args.raw_dir.expanduser().resolve()
    base_dir = args.base_cache_dir.expanduser().resolve()
    augmentation_dir = args.augmentation_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    config = BenchmarkConfig(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        n_jobs=args.n_jobs,
    )
    results_path = out_dir / "spatial_v2_dev_results.json"
    oof_path = out_dir / "spatial_v2_dev_oof.parquet"
    manifest_path = out_dir / "spatial_v2_dev_manifest.json"
    model_paths = [
        out_dir / "models" / f"{group}__{feature_set}__{objective}.joblib"
        for group in TARGET_COLS
        for feature_set in ("v1", "v1_plus_v2_core", "v1_plus_v2_seq")
        for objective in OBJECTIVES
    ]
    planned = [results_path, oof_path, manifest_path, *model_paths]
    existing = [path for path in planned if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite benchmark outputs: {existing}")

    labels_path = raw_dir / "train" / "train_labels.csv"
    labels = _read_labels(raw_dir)
    base, augmented_core, augmented_sequence, cache_inputs = _load_features(
        base_dir, augmentation_dir, labels
    )
    feature_sets = {
        "v1": base,
        "v1_plus_v2_core": augmented_core,
        "v1_plus_v2_seq": augmented_sequence,
    }

    records: list[dict[str, Any]] = []
    oof_columns: dict[tuple[str, str, str], pd.Series] = {}
    written_models: list[Path] = []
    for group in TARGET_COLS:
        train_fold, valid_fold = _fold_masks(group, labels.index)
        y_all = labels[group].to_numpy(dtype=float)
        capacity = CAPACITY_KWH[group]
        eligible = np.isfinite(y_all) & (y_all >= config.eligible_fraction * capacity)
        fit_rows = train_fold & eligible
        valid_rows = valid_fold
        if not np.any(fit_rows) or not np.any(valid_rows & np.isfinite(y_all)):
            raise ValueError(f"{group}: empty eligible training or validation rows")
        for feature_set_name, frames in feature_sets.items():
            x = frames[group]
            for objective_name in OBJECTIVES:
                print(
                    f"fit {group} {feature_set_name} {objective_name}: "
                    f"{fit_rows.sum()} train / {valid_rows.sum()} valid / {x.shape[1]} features",
                    flush=True,
                )
                model = _make_model(config, objective_name)
                model.fit(
                    x.loc[fit_rows],
                    y_all[fit_rows] / capacity,
                    callbacks=[log_evaluation(period=0)],
                )
                prediction = model.predict(x.loc[valid_rows]) * capacity
                if not np.isfinite(prediction).all():
                    raise ValueError("model produced non-finite validation predictions")
                metrics = _metric_dict(y_all[valid_rows], prediction, group)
                model_path = (
                    out_dir
                    / "models"
                    / f"{group}__{feature_set_name}__{objective_name}.joblib"
                )
                _dump_joblib_atomic(model, model_path)
                written_models.append(model_path)
                records.append(
                    {
                        "group": group,
                        "feature_set": feature_set_name,
                        "objective": objective_name,
                        "train_period": [
                            labels.index[train_fold].min().isoformat(),
                            labels.index[train_fold].max().isoformat(),
                        ],
                        "valid_period": [
                            labels.index[valid_rows].min().isoformat(),
                            labels.index[valid_rows].max().isoformat(),
                        ],
                        "eligible_train_rows": int(fit_rows.sum()),
                        "valid_rows": int(valid_rows.sum()),
                        "feature_count": int(x.shape[1]),
                        "metrics": metrics,
                        "feature_importance": _feature_importance(model, x.columns),
                        "model_path": str(model_path),
                    }
                )
                key = (group, feature_set_name, objective_name)
                oof_columns[key] = pd.Series(
                    prediction,
                    index=labels.index[valid_rows],
                    name="__".join(key),
                )

    gains: dict[str, Any] = {}
    stability: dict[str, list[str]] = {}
    recommendation: dict[str, bool] = {}
    for variant in ("v1_plus_v2_core", "v1_plus_v2_seq"):
        gains[variant] = {}
        stable_groups: list[str] = []
        for group in TARGET_COLS:
            objective_gains: dict[str, float] = {}
            core_lookup: dict[str, float] = {}
            for objective in OBJECTIVES:
                lookup = {
                    row["feature_set"]: row["metrics"]["score"]
                    for row in records
                    if row["group"] == group and row["objective"] == objective
                }
                objective_gains[objective] = float(lookup[variant] - lookup["v1"])
                if variant == "v1_plus_v2_seq":
                    core_lookup[objective] = float(
                        lookup["v1_plus_v2_seq"] - lookup["v1_plus_v2_core"]
                    )
            mean_gain = float(np.mean(list(objective_gains.values())))
            stable = min(objective_gains.values()) >= -0.001 and mean_gain > 0.0
            gains[variant][group] = {
                "score_gain_variant_minus_v1": objective_gains,
                "score_gain_sequence_minus_core": core_lookup,
                "mean_score_gain": mean_gain,
                "stable_by_preregistered_rule": stable,
            }
            if stable:
                stable_groups.append(group)
        worst_group_mean = min(
            values["mean_score_gain"] for values in gains[variant].values()
        )
        stability[variant] = stable_groups
        recommendation[variant] = (
            len(stable_groups) >= 2 and worst_group_mean >= -0.002
        )

    oof = pd.concat(oof_columns.values(), axis=1).sort_index()
    _write_parquet_atomic(oof, oof_path)
    result_payload = {
        "status": "completed",
        "benchmark_config": asdict(config),
        "fold_policy": {
            "groups_1_2": "train 2022; validate 2023",
            "group_3": "train 2023-01-01 01:00 through 2023-07-01 00:00; validate remainder of 2023",
            "first_forbidden_timestamp": FIRST_FORBIDDEN_TIMESTAMP.isoformat(),
            "2024_rows_read": False,
            "validation_used_for_early_stopping": False,
        },
        "records": records,
        "gains": gains,
        "decision_rule": (
            "For core and sequence variants separately, recommend a 2024 OOF "
            "check only when at least two of three groups have both-objective "
            "minimum gain >= -0.001 and positive mean gain, and no group mean "
            "gain is below -0.002."
        ),
        "stable_groups": stability,
        "recommend_2024_oof": recommendation,
        "note": (
            "This fixed-model historical comparison is development evidence, not "
            "an independent leaderboard estimate."
        ),
    }
    write_json_atomic(results_path, result_payload, overwrite=args.overwrite)
    manifest = make_manifest(
        artifact_type="baram_spatial_v2_historical_benchmark",
        parameters={
            "benchmark_config": asdict(config),
            "objectives": OBJECTIVES,
            "strict_pre_2024_bounded_read_rows": DEV_ROWS,
        },
        input_files=[labels_path, *cache_inputs],
        output_files=[results_path, oof_path, *written_models],
        results={
            "gains": gains,
            "stable_groups": stability,
            "recommend_2024_oof": recommendation,
        },
        project_dir=PROJECT_DIR,
    )
    write_json_atomic(manifest_path, manifest, overwrite=args.overwrite)
    print(f"results: {results_path}")
    print(f"stable groups: {stability}; recommend 2024 OOF: {recommendation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

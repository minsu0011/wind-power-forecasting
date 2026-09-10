"""Reproduce the locked 2022 -> 2023 two-group development experiment.

This CLI intentionally cannot train on the 2024 forecast year.  It reads only
the first 17,520 official label rows and applies a Parquet predicate ending at
``2024-01-01 00:00:00``, the closing hour of the 2023 forecast period.

Examples:
    python scripts/train_dev.py --raw-dir C:/data/open --cache-dir artifacts/cache
    python scripts/train_dev.py --raw-dir C:/data/open --cache-dir artifacts/cache \
        --artifact-dir artifacts --candidates lgb_l1 lgb_q07
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.cv import FOLD_2023, assert_aligned  # noqa: E402
from src.manifest import (  # noqa: E402
    append_experiment_jsonl,
    describe_file,
    git_state,
    make_experiment_record,
    package_versions,
    sha256_file,
)
from src.metric import CAPACITY_KWH, score_details  # noqa: E402


GROUPS = ("kpx_group_1", "kpx_group_2")
EXPECTED_DEV_ROWS = 17_520
# Midnight is the interval-closing label for the preceding forecast day.  No
# 2024 forecast horizon (2024-01-01 01:00 onward) is permitted in this CLI.
DEV_LAST_TIMESTAMP = pd.Timestamp("2024-01-01 00:00:00")
DEV_READ_CUTOFF = pd.Timestamp("2024-01-01 00:00:01")
GROUP_ID = {"kpx_group_1": 1.0, "kpx_group_2": 2.0}


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    objective: str
    alpha: float | None
    shared: bool


CANDIDATES: dict[str, CandidateSpec] = {
    "lgb_l1": CandidateSpec("lgb_l1", "regression_l1", None, False),
    "lgb_q06": CandidateSpec("lgb_q06", "quantile", 0.60, False),
    "lgb_q07": CandidateSpec("lgb_q07", "quantile", 0.70, False),
    "shared_l1": CandidateSpec("shared_l1", "regression_l1", None, True),
    "shared_q07": CandidateSpec("shared_q07", "quantile", 0.70, True),
}


@dataclass(frozen=True)
class DevConfig:
    eligible_fraction: float = 0.10
    target_unit: str = "capacity_factor"
    n_estimators: int = 1_500
    learning_rate: float = 0.025
    num_leaves: int = 31
    per_group_min_child_samples: int = 30
    shared_min_child_samples: int = 40
    subsample: float = 0.80
    subsample_freq: int = 1
    colsample_bytree: float = 0.75
    reg_alpha: float = 0.05
    reg_lambda: float = 2.0
    seed: int = 42
    n_jobs: int = 7

    def __post_init__(self) -> None:
        if not 0.0 <= self.eligible_fraction < 1.0:
            raise ValueError("eligible_fraction must be in [0, 1)")
        if self.target_unit != "capacity_factor":
            raise ValueError("this reproducibility CLI requires capacity_factor targets")
        if self.n_estimators <= 0 or self.learning_rate <= 0:
            raise ValueError("n_estimators and learning_rate must be positive")
        if self.n_jobs == 0:
            raise ValueError("n_jobs must not be zero")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train leakage-locked 2022 -> 2023 LightGBM OOF candidates for "
            "KPX groups 1 and 2 only."
        )
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="official data root containing train/train_labels.csv",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="weather-feature cache created by scripts/build_features.py",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=PROJECT_DIR / "artifacts",
        help="root for models/, oof/, and logs/ (default: project artifacts/)",
    )
    parser.add_argument(
        "--candidates",
        nargs="+",
        choices=tuple(CANDIDATES),
        default=list(CANDIDATES),
        help="candidate subset; defaults to the five locked base models",
    )
    parser.add_argument(
        "--eligible-fraction",
        type=float,
        default=0.10,
        help="train only where actual >= this fraction of capacity (default: 0.10)",
    )
    parser.add_argument("--n-estimators", type=int, default=1_500)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--per-group-min-child-samples", type=int, default=30)
    parser.add_argument("--shared-min-child-samples", type=int, default=40)
    parser.add_argument("--subsample", type=float, default=0.80)
    parser.add_argument("--subsample-freq", type=int, default=1)
    parser.add_argument("--colsample-bytree", type=float, default=0.75)
    parser.add_argument("--reg-alpha", type=float, default=0.05)
    parser.add_argument("--reg-lambda", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=7)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace candidate artifacts and the run JSONL after successful writes",
    )
    return parser.parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> DevConfig:
    return DevConfig(
        eligible_fraction=args.eligible_fraction,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        per_group_min_child_samples=args.per_group_min_child_samples,
        shared_min_child_samples=args.shared_min_child_samples,
        subsample=args.subsample,
        subsample_freq=args.subsample_freq,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )


def _artifact_stems(spec: CandidateSpec, n_estimators: int) -> tuple[str, str]:
    """Return (OOF stem, model stem), matching the original inline artifacts."""

    suffix = "" if n_estimators == 1_500 else f"_n{n_estimators}"
    if spec.name == "lgb_l1":
        return (
            f"dev2023_lgb_l1_eligible_n{n_estimators}",
            f"dev2023_lgb_l1_eligible_{n_estimators}",
        )
    return f"dev2023_{spec.name}_eligible{suffix}", f"dev2023_{spec.name}_eligible{suffix}"


def _expected_paths(
    artifact_dir: Path, specs: Sequence[CandidateSpec], config: DevConfig
) -> tuple[dict[str, tuple[Path, Path]], Path]:
    paths: dict[str, tuple[Path, Path]] = {}
    for spec in specs:
        oof_stem, model_stem = _artifact_stems(spec, config.n_estimators)
        paths[spec.name] = (
            artifact_dir / "oof" / f"{oof_stem}.parquet",
            artifact_dir / "models" / f"{model_stem}.joblib",
        )
    log_path = artifact_dir / "logs" / "dev2023_base_models.jsonl"
    return paths, log_path


def _read_labels_before_2024(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "train" / "train_labels.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    labels = pd.read_csv(
        path,
        usecols=["kst_dtm", *GROUPS],
        nrows=EXPECTED_DEV_ROWS,
        encoding="utf-8-sig",
        parse_dates=["kst_dtm"],
    ).set_index("kst_dtm")
    labels.index = pd.DatetimeIndex(labels.index)
    if len(labels) != EXPECTED_DEV_ROWS:
        raise ValueError(
            f"expected {EXPECTED_DEV_ROWS} pre-2024 labels, found {len(labels)}"
        )
    _assert_dev_index(labels.index, context="labels")
    return labels


def _read_features_before_2024(cache_dir: Path, group: str) -> pd.DataFrame:
    path = cache_dir / f"{group}_weather_train.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}; run scripts/build_features.py before train_dev.py"
        )
    # The official cache is chronologically sorted but currently has one large
    # row group. A predicate can therefore require decoding that whole row
    # group. Reading exactly one bounded RecordBatch is stricter: iteration is
    # stopped after the 17,520 pre-2024 rows, before any future batch is asked
    # for or materialised.
    import pyarrow.parquet as pq

    batches = pq.ParquetFile(path).iter_batches(batch_size=EXPECTED_DEV_ROWS)
    first_batch = next(batches, None)
    if first_batch is None:
        raise ValueError(f"empty feature cache: {path}")
    frame = first_batch.to_pandas()
    frame.index = pd.DatetimeIndex(frame.index)
    _assert_dev_index(frame.index, context=f"{group} features")
    if len(frame) != EXPECTED_DEV_ROWS:
        raise ValueError(
            f"{group} cache returned {len(frame)} rows, expected {EXPECTED_DEV_ROWS}"
        )
    if not frame.columns.is_unique:
        raise ValueError(f"{group} cache has duplicate feature columns")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes):
        raise TypeError(f"{group} cache must contain numeric features only")
    values = frame.to_numpy(copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"{group} cache contains NaN or infinite features")
    return frame


def _assert_dev_index(index: pd.DatetimeIndex, *, context: str) -> None:
    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError(f"{context} index must be unique and sorted")
    if len(index) == 0:
        raise ValueError(f"{context} is empty")
    if index[0] != pd.Timestamp("2022-01-01 01:00:00"):
        raise ValueError(f"{context} starts at unexpected timestamp {index[0]}")
    if index[-1] != DEV_LAST_TIMESTAMP:
        raise ValueError(f"{context} ends at unexpected timestamp {index[-1]}")
    if (index >= DEV_READ_CUTOFF).any():
        raise AssertionError(f"{context} contains forbidden 2024 forecast rows")


def _load_dev_data(
    raw_dir: Path, cache_dir: Path
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], np.ndarray, np.ndarray]:
    labels = _read_labels_before_2024(raw_dir)
    features = {group: _read_features_before_2024(cache_dir, group) for group in GROUPS}
    assert_aligned(labels, *features.values())
    reference_columns = features[GROUPS[0]].columns
    for group in GROUPS[1:]:
        if not reference_columns.equals(features[group].columns):
            raise ValueError(f"feature schema differs between {GROUPS[0]} and {group}")
    train_mask, valid_mask = FOLD_2023.masks(labels.index)
    if np.any(train_mask & valid_mask):
        raise AssertionError("outer train and validation overlap")
    if labels.index[train_mask].max() >= labels.index[valid_mask].min():
        raise AssertionError("outer fold is not strictly chronological")
    if int(train_mask.sum()) != 8_760 or int(valid_mask.sum()) != 8_760:
        raise ValueError(
            "locked dev fold must contain exactly 8,760 train and 8,760 validation rows"
        )
    return labels, features, train_mask, valid_mask


def _lgbm_params(
    spec: CandidateSpec, config: DevConfig
) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "objective": spec.objective,
        "n_estimators": config.n_estimators,
        "learning_rate": config.learning_rate,
        "num_leaves": config.num_leaves,
        "min_child_samples": (
            config.shared_min_child_samples
            if spec.shared
            else config.per_group_min_child_samples
        ),
        "subsample": config.subsample,
        "subsample_freq": config.subsample_freq,
        "colsample_bytree": config.colsample_bytree,
        "reg_alpha": config.reg_alpha,
        "reg_lambda": config.reg_lambda,
        "random_state": config.seed,
        "n_jobs": config.n_jobs,
        "verbosity": -1,
    }
    if spec.alpha is not None:
        parameters["alpha"] = spec.alpha
    return parameters


def _eligible_mask(
    labels: pd.Series, train_mask: np.ndarray, capacity: float, fraction: float
) -> np.ndarray:
    values = labels.to_numpy(dtype="float64", copy=False)
    eligible = np.isfinite(values) & (values >= capacity * fraction)
    selected = np.asarray(train_mask) & eligible
    if not selected.any():
        raise ValueError(f"no eligible training rows for {labels.name}")
    return selected


def _fit_candidate(
    spec: CandidateSpec,
    config: DevConfig,
    labels: pd.DataFrame,
    features: Mapping[str, pd.DataFrame],
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[pd.DataFrame, Any, dict[str, int], dict[str, Any]]:
    from lightgbm import LGBMRegressor

    params = _lgbm_params(spec, config)
    valid_index = labels.index[valid_mask]
    predictions = pd.DataFrame(index=valid_index, columns=GROUPS, dtype="float64")
    predictions.index.name = "kst_dtm"
    eligible_counts: dict[str, int] = {}

    if not spec.shared:
        models: dict[str, Any] = {}
        for group in GROUPS:
            capacity = CAPACITY_KWH[group]
            selected = _eligible_mask(
                labels[group], train_mask, capacity, config.eligible_fraction
            )
            target = labels.loc[selected, group] / capacity
            model = LGBMRegressor(**params)
            model.fit(features[group].loc[selected], target)
            forecast_cf = np.asarray(
                model.predict(features[group].loc[valid_mask]), dtype="float64"
            )
            predictions.loc[:, group] = forecast_cf * capacity
            models[group] = model
            eligible_counts[group] = int(selected.sum())
        model_artifact: Any = models
    else:
        train_frames: list[pd.DataFrame] = []
        train_targets: list[pd.Series] = []
        for group in GROUPS:
            capacity = CAPACITY_KWH[group]
            selected = _eligible_mask(
                labels[group], train_mask, capacity, config.eligible_fraction
            )
            group_frame = features[group].loc[selected].copy()
            group_frame["model__group_id"] = np.float32(GROUP_ID[group])
            train_frames.append(group_frame)
            train_targets.append(labels.loc[selected, group] / capacity)
            eligible_counts[group] = int(selected.sum())
        shared_train = pd.concat(train_frames, axis=0, copy=False)
        shared_target = pd.concat(train_targets, axis=0, copy=False)
        model = LGBMRegressor(**params)
        model.fit(shared_train, shared_target)
        for group in GROUPS:
            valid_frame = features[group].loc[valid_mask].copy()
            valid_frame["model__group_id"] = np.float32(GROUP_ID[group])
            forecast_cf = np.asarray(model.predict(valid_frame), dtype="float64")
            predictions.loc[:, group] = forecast_cf * CAPACITY_KWH[group]
        model_artifact = model

    if not np.isfinite(predictions.to_numpy()).all():
        raise RuntimeError(f"{spec.name} produced NaN or infinite OOF predictions")
    metrics = score_details(
        labels.loc[valid_mask, list(GROUPS)],
        predictions,
        target_cols=GROUPS,
        capacities=CAPACITY_KWH,
    ).as_dict()
    return predictions, model_artifact, eligible_counts, {"params": params, "metrics": metrics}


def _write_parquet_atomic(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        frame.to_parquet(temporary, engine="pyarrow", compression="snappy", index=True)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_joblib_atomic(value: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        joblib.dump(value, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _loaded_slice_metadata(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    """Describe and hash only the already-loaded pre-2024 frame.

    Hashing the whole source file would read 2024 bytes solely for provenance,
    violating this CLI's deliberately strict data-access boundary.
    """

    result = describe_file(path, include_hash=False)
    digest = hashlib.sha256()
    schema = [(str(column), str(dtype)) for column, dtype in frame.dtypes.items()]
    digest.update(json.dumps(schema, ensure_ascii=False).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    result.update(
        {
            "loaded_slice_sha256": digest.hexdigest(),
            "loaded_rows": int(len(frame)),
            "loaded_time_start": str(frame.index.min()),
            "loaded_time_end": str(frame.index.max()),
            "whole_file_hash_deliberately_omitted": True,
        }
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    raw_dir = args.raw_dir.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    artifact_dir = args.artifact_dir.expanduser().resolve()
    config = _config_from_args(args)
    specs = [CANDIDATES[name] for name in dict.fromkeys(args.candidates)]
    output_paths, log_path = _expected_paths(artifact_dir, specs, config)

    existing = [
        path
        for pair in output_paths.values()
        for path in pair
        if path.exists()
    ]
    if log_path.exists():
        existing.append(log_path)
    if existing and not args.overwrite:
        rendered = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(
            "development artifacts already exist; use a new --artifact-dir or "
            f"pass --overwrite:\n{rendered}"
        )

    print("locked fold: 2022 train -> 2023 validation (groups 1 and 2)")
    print(f"raw labels: {raw_dir / 'train' / 'train_labels.csv'}")
    print(f"feature cache: {cache_dir}")
    print(f"candidates: {', '.join(spec.name for spec in specs)}")
    print(json.dumps(asdict(config), sort_keys=True))
    labels, features, train_mask, valid_mask = _load_dev_data(raw_dir, cache_dir)
    print(
        f"loaded only {labels.index.min()} .. {labels.index.max()} "
        f"({len(labels):,} rows); 2024 forecast horizons excluded"
    )

    label_path = raw_dir / "train" / "train_labels.csv"
    input_metadata = [_loaded_slice_metadata(label_path, labels)]
    input_metadata.extend(
        _loaded_slice_metadata(
            cache_dir / f"{group}_weather_train.parquet", features[group]
        )
        for group in GROUPS
    )
    runtime = {"packages": package_versions(), "git": git_state(PROJECT_DIR)}
    temporary_log = log_path.with_name(f".{log_path.name}.tmp-{os.getpid()}")
    temporary_log.parent.mkdir(parents=True, exist_ok=True)
    if temporary_log.exists():
        temporary_log.unlink()

    try:
        for spec in specs:
            print(f"training {spec.name} ...", flush=True)
            oof, model, eligible_counts, outcome = _fit_candidate(
                spec, config, labels, features, train_mask, valid_mask
            )
            oof_path, model_path = output_paths[spec.name]
            _write_parquet_atomic(oof, oof_path)
            _write_joblib_atomic(model, model_path)
            metrics = outcome["metrics"]
            record = make_experiment_record(
                experiment_id=oof_path.stem,
                status="completed",
                validation={
                    "fold": FOLD_2023.name,
                    "train_start": str(labels.index[train_mask].min()),
                    "train_end": str(labels.index[train_mask].max()),
                    "valid_start": str(labels.index[valid_mask].min()),
                    "valid_end": str(labels.index[valid_mask].max()),
                    "groups": list(GROUPS),
                    "eligible_fraction_train_only": config.eligible_fraction,
                    "eligible_train_rows": eligible_counts,
                    "capacity_factor_target": True,
                    "forbidden_2024_forecast_rows_loaded": False,
                },
                model={
                    "candidate": spec.name,
                    "shared_across_groups": spec.shared,
                    "lightgbm": outcome["params"],
                    "capacity_kwh": {group: CAPACITY_KWH[group] for group in GROUPS},
                },
                dev_metrics={
                    "score": metrics["total_score"],
                    "one_minus_nmae": metrics["one_minus_nmae"],
                    "ficr": metrics["ficr"],
                    "by_group": metrics["by_group"],
                },
                leaderboard_metrics={},
                artifacts={
                    "oof": str(oof_path),
                    "oof_sha256": sha256_file(oof_path),
                    "model": str(model_path),
                    "model_sha256": sha256_file(model_path),
                },
                notes=(
                    "Locked two-group dev result; group 3 is unavailable in the "
                    "2022 training year and this score is not leaderboard-comparable."
                ),
            )
            record["inputs"] = input_metadata
            record["runtime"] = runtime
            record["dev_config"] = asdict(config)
            append_experiment_jsonl(temporary_log, _json_ready(record))
            print(
                f"{spec.name}: score={metrics['total_score']:.6f}, "
                f"1-NMAE={metrics['one_minus_nmae']:.6f}, "
                f"FICR={metrics['ficr']:.6f}"
            )
        os.replace(temporary_log, log_path)
    finally:
        if temporary_log.exists():
            temporary_log.unlink()

    print(f"experiment log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

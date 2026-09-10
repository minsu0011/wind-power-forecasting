"""Consume the untouched 2024 gate exactly once with a locked recipe.

This command is deliberately separate from development training.  It reads no
development OOF/model artifacts and performs no tuning.  Candidate definitions,
group-3 handling, ensemble weights, affine maps, power bins, and bounds all come
from the supplied locked final-recipe JSON.

Safe schema-only inspection (does not read label values)::

    python scripts/evaluate_gate.py --raw-dir C:/data/open \
        --cache-dir artifacts/cache --out-dir artifacts/gate/v2 \
        --config configs/train_final.v2.locked.json --dry-run

Intentional one-time gate consumption::

    python scripts/evaluate_gate.py --raw-dir C:/data/open \
        --cache-dir artifacts/cache --out-dir artifacts/gate/v2 \
        --config configs/train_final.v2.locked.json \
        --confirm-consume-2024-gate

The real run reserves ``out-dir`` before reading label values.  A completed or
partial run therefore refuses to start again unless ``--overwrite`` is supplied
explicitly.  A partial run is still considered a consumed gate.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.final_training import (  # noqa: E402
    FinalDataBundle,
    TrainedCandidate,
    _assemble_final_predictions,
    read_recipe,
    train_candidate,
)
from src.manifest import (  # noqa: E402
    make_manifest,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from src.metric import CAPACITY_KWH, TARGET_COLS, score_details  # noqa: E402


GATE_SCHEMA_VERSION = 1
TRAIN_END_INCLUSIVE = pd.Timestamp("2024-01-01 00:00:00")
VALID_START = pd.Timestamp("2024-01-01 01:00:00")
VALID_END_INCLUSIVE = pd.Timestamp("2025-01-01 00:00:00")
EXPECTED_VALID_HOURS = 8_784  # 2024 is a leap year.


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        help="weather cache directory produced by scripts/build_features.py",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="dedicated, initially empty directory for this one gate run",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="locked final-recipe JSON, including the group-3 recipe",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "validate config, raw header, and cache time/schema metadata only; "
            "does not read label values, fit, score, reserve, or write"
        ),
    )
    parser.add_argument(
        "--confirm-consume-2024-gate",
        action="store_true",
        help="required acknowledgement before any 2024 label value is read",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow an explicit rerun over a completed/partial gate directory",
    )
    return parser.parse_args(argv)


def _enabled_models(recipe: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(name)
        for name, specification in recipe["models"].items()
        if bool(specification["enabled"])
    )


def _input_paths(raw_dir: Path, cache_dir: Path) -> tuple[Path, dict[str, Path]]:
    label_path = raw_dir / "train" / "train_labels.csv"
    cache_paths = {
        group: cache_dir / f"{group}_weather_train.parquet"
        for group in TARGET_COLS
    }
    missing = [
        path for path in (label_path, *cache_paths.values()) if not path.is_file()
    ]
    if missing:
        rendered = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"gate raw/cache inputs are incomplete:\n{rendered}")
    return label_path, cache_paths


def _expected_hourly_index(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="h", name="forecast_kst_dtm")


def _assert_exact_index(
    index: pd.Index,
    *,
    expected_start: pd.Timestamp,
    expected_end: pd.Timestamp,
    name: str,
) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"{name} must use a DatetimeIndex")
    expected = _expected_hourly_index(expected_start, expected_end)
    if not index.equals(expected):
        raise ValueError(
            f"{name} must be the exact hourly range {expected_start} through "
            f"{expected_end}; got rows={len(index)}, min={index.min()}, max={index.max()}"
        )
    return expected


def _cache_metadata(path: Path) -> dict[str, Any]:
    """Inspect weather-cache metadata/index without loading any target values."""

    parquet_file = pq.ParquetFile(path)
    index_only = pd.read_parquet(path, columns=[])
    if not isinstance(index_only.index, pd.DatetimeIndex):
        raise TypeError(f"{path.name} must store a DatetimeIndex")
    if not index_only.index.is_unique or not index_only.index.is_monotonic_increasing:
        raise ValueError(f"{path.name} index must be unique and sorted")
    if index_only.index.min() > TRAIN_END_INCLUSIVE:
        raise ValueError(f"{path.name} does not cover gate training end")
    _assert_exact_index(
        index_only.loc[VALID_START:VALID_END_INCLUSIVE].index,
        expected_start=VALID_START,
        expected_end=VALID_END_INCLUSIVE,
        name=f"{path.name} gate slice",
    )
    physical_columns = list(parquet_file.schema_arrow.names)
    feature_count = len(physical_columns) - (
        1 if "forecast_kst_dtm" in physical_columns else 0
    )
    return {
        "path": str(path),
        "rows": int(parquet_file.metadata.num_rows),
        "features": int(feature_count),
        "index_min": str(index_only.index.min()),
        "index_max": str(index_only.index.max()),
        "gate_rows": EXPECTED_VALID_HOURS,
    }


def dry_run_summary(
    recipe: Mapping[str, Any],
    *,
    label_path: Path,
    cache_paths: Mapping[str, Path],
    out_dir: Path,
) -> dict[str, Any]:
    """Return a non-consuming plan; only the CSV header is opened."""

    label_columns = tuple(
        pd.read_csv(label_path, encoding="utf-8-sig", nrows=0).columns
    )
    expected_columns = ("kst_dtm", *TARGET_COLS)
    if label_columns != expected_columns:
        raise ValueError(f"train_labels columns must be {expected_columns!r}")
    cache = {group: _cache_metadata(path) for group, path in cache_paths.items()}
    feature_counts = {metadata["features"] for metadata in cache.values()}
    if len(feature_counts) != 1:
        raise ValueError("group cache feature counts differ")
    return {
        "mode": "dry_run_no_label_values_read",
        "recipe_name": recipe["recipe_name"],
        "recipe_locked": bool(recipe["recipe_locked"]),
        "ready_for_gate": bool(recipe["recipe_locked"]),
        "enabled_models": list(_enabled_models(recipe)),
        "groups": list(TARGET_COLS),
        "group3_from_config": "kpx_group_3" in recipe["ensemble"]["weights"]
        if recipe["recipe_locked"]
        else False,
        "train_end_inclusive": str(TRAIN_END_INCLUSIVE),
        "valid_start": str(VALID_START),
        "valid_end_inclusive": str(VALID_END_INCLUSIVE),
        "valid_hours": EXPECTED_VALID_HOURS,
        "label_header_only": list(label_columns),
        "cache": cache,
        "output_directory": str(out_dir),
        "would_refuse_existing_output_without_overwrite": True,
    }


def _read_labels(label_path: Path) -> pd.DataFrame:
    labels = pd.read_csv(label_path, encoding="utf-8-sig")
    expected_columns = ("kst_dtm", *TARGET_COLS)
    if tuple(labels.columns) != expected_columns:
        raise ValueError(f"train_labels columns must be {expected_columns!r}")
    times = pd.to_datetime(labels.pop("kst_dtm"), errors="raise")
    labels.index = pd.DatetimeIndex(times, name="forecast_kst_dtm")
    labels = labels.astype(float)
    if not labels.index.is_unique or not labels.index.is_monotonic_increasing:
        raise ValueError("label timestamps must be unique and sorted")
    if np.isinf(labels.to_numpy()).any():
        raise ValueError("labels contain infinite values")
    return labels


def _label_availability(
    train_labels: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    availability: dict[str, dict[str, Any]] = {}
    for group in TARGET_COLS:
        finite = train_labels[group].notna()
        if not finite.any():
            raise ValueError(f"{group} has no labels before the gate")
        first = train_labels.index[finite][0]
        expected = _expected_hourly_index(first, TRAIN_END_INCLUSIVE)
        observed = train_labels[group].reindex(expected)
        # The official labels contain a small number of unavailable target
        # rows inside an otherwise complete hourly index.  Final training and
        # the published metric both exclude those NaNs; rejecting them here
        # would make the gate stricter than the competition itself.
        finite_observed = observed.notna()
        if not finite_observed.any():
            raise ValueError(f"{group} has no finite labels in its train interval")
        observed_values = observed.loc[finite_observed].to_numpy(dtype=float)
        if not np.isfinite(observed_values).all():
            raise ValueError(f"{group} has infinite labels in its train interval")
        if first != train_labels.index[finite].min():
            raise AssertionError("label availability start calculation changed")
        availability[group] = {
            "start": str(first),
            "end_inclusive": str(TRAIN_END_INCLUSIVE),
            "rows": int(len(expected)),
            "finite_rows": int(finite_observed.sum()),
            "missing_rows": int((~finite_observed).sum()),
        }
    return availability


def _load_gate_bundle(
    label_path: Path,
    cache_paths: Mapping[str, Path],
) -> tuple[FinalDataBundle, pd.DataFrame, dict[str, dict[str, Any]], list[Path]]:
    """Load only labels and train weather caches, then isolate the gate."""

    labels = _read_labels(label_path)
    valid_index = _assert_exact_index(
        labels.loc[VALID_START:VALID_END_INCLUSIVE].index,
        expected_start=VALID_START,
        expected_end=VALID_END_INCLUSIVE,
        name="2024 gate labels",
    )
    valid_answer = labels.loc[valid_index, list(TARGET_COLS)].copy()
    # Missing actuals are explicitly ignored by src.metric, matching DACON's
    # reference implementation.  Infinite actuals remain invalid, and every
    # group must still contain scored rows at or above 10% capacity.
    for group in TARGET_COLS:
        finite = valid_answer[group].notna()
        values = valid_answer.loc[finite, group].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"2024 gate has infinite labels for {group}")
        capacity = float(CAPACITY_KWH[group])
        if not (values >= 0.10 * capacity).any():
            raise ValueError(f"2024 gate has no evaluated rows for {group}")

    common_train = labels.index[labels.index <= TRAIN_END_INCLUSIVE]
    if common_train.empty:
        raise ValueError("gate training interval is empty")
    _assert_exact_index(
        common_train,
        expected_start=labels.index.min(),
        expected_end=TRAIN_END_INCLUSIVE,
        name="common gate training index",
    )
    train_labels = labels.loc[common_train, list(TARGET_COLS)].copy()
    availability = _label_availability(train_labels)
    if common_train.max() >= valid_index.min():
        # Exactly one hour apart is expected, never an overlap.
        raise AssertionError("gate train and validation periods overlap")
    if valid_index.min() - common_train.max() != pd.Timedelta(hours=1):
        raise AssertionError("gate train/validation boundary is not contiguous hourly")

    train_features: dict[str, pd.DataFrame] = {}
    valid_features: dict[str, pd.DataFrame] = {}
    canonical_columns: tuple[str, ...] | None = None
    for group in TARGET_COLS:
        full = pd.read_parquet(cache_paths[group])
        if not isinstance(full.index, pd.DatetimeIndex):
            raise TypeError(f"{group} cache must use a DatetimeIndex")
        if not full.index.is_unique or not full.index.is_monotonic_increasing:
            raise ValueError(f"{group} cache index must be unique and sorted")
        if not full.columns.is_unique:
            raise ValueError(f"{group} cache feature names must be unique")
        if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in full.dtypes):
            raise TypeError(f"{group} cache must contain only numeric features")
        train = full.reindex(common_train)
        valid = full.reindex(valid_index)
        if train.isna().any().any() or valid.isna().any().any():
            raise ValueError(f"{group} cache is incomplete on gate train/valid")
        if not np.isfinite(train.to_numpy()).all() or not np.isfinite(
            valid.to_numpy()
        ).all():
            raise ValueError(f"{group} cache has non-finite gate features")
        if tuple(train.columns) != tuple(valid.columns):
            raise ValueError(f"{group} train/valid feature schemas differ")
        if canonical_columns is None:
            canonical_columns = tuple(train.columns)
        elif tuple(train.columns) != canonical_columns:
            raise ValueError("group cache feature schemas differ")
        train_features[group] = train.astype(np.float32, copy=False)
        valid_features[group] = valid.astype(np.float32, copy=False)

    # train_candidate only needs this frame for its index; no final-submission
    # identifier is involved in a gate evaluation.
    placeholder = pd.DataFrame(
        {
            "forecast_id": pd.Series(
                [f"gate_{position:05d}" for position in range(len(valid_index))],
                dtype="string",
            ),
            "forecast_kst_dtm": valid_index.astype(str),
            **{group: np.nan for group in TARGET_COLS},
        }
    )
    feature_manifest = cache_paths[TARGET_COLS[0]].parent / "feature_manifest.json"
    inputs = [label_path, *[cache_paths[group] for group in TARGET_COLS]]
    if feature_manifest.is_file():
        inputs.append(feature_manifest)
    bundle = FinalDataBundle(
        labels=train_labels,
        sample_submission=placeholder,
        sample_times=valid_index,
        train_features=train_features,
        test_features=valid_features,
        input_files=inputs,
    )
    return bundle, valid_answer, availability, inputs


def _output_paths(
    out_dir: Path,
    recipe: Mapping[str, Any],
) -> dict[str, Path]:
    paths: dict[str, Path] = {
        "state": out_dir / "gate_run_state.json",
        "recipe": out_dir / "gate_recipe_snapshot.json",
        "metrics": out_dir / "gate_metrics.json",
        "oof": out_dir / "gate_oof.parquet",
        "manifest": out_dir / "gate_manifest.json",
    }
    for candidate in _enabled_models(recipe):
        paths[f"model:{candidate}"] = out_dir / "models" / f"{candidate}.joblib"
        paths[f"prediction:{candidate}"] = (
            out_dir / "predictions" / f"{candidate}_gate.parquet"
        )
    return paths


def _reserve_gate(
    out_dir: Path,
    paths: Mapping[str, Path],
    *,
    overwrite: bool,
    config_path: Path,
    recipe: Mapping[str, Any],
) -> None:
    existing_files = []
    if out_dir.exists():
        existing_files = [path for path in out_dir.rglob("*") if path.is_file()]
    if existing_files and not overwrite:
        rendered = "\n".join(f"  - {path}" for path in existing_files[:20])
        raise FileExistsError(
            "gate output directory already contains a completed or partial result; "
            "the gate is considered consumed. Pass --overwrite only for an "
            f"intentional rerun:\n{rendered}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "gate_schema_version": GATE_SCHEMA_VERSION,
        "status": "started",
        "reserved_utc": utc_now(),
        "recipe_name": recipe["recipe_name"],
        "recipe_sha256": sha256_file(config_path),
        "train_end_inclusive": str(TRAIN_END_INCLUSIVE),
        "valid_start": str(VALID_START),
        "valid_end_inclusive": str(VALID_END_INCLUSIVE),
        "warning": "A started/partial state counts as a consumed gate.",
    }
    write_json_atomic(paths["state"], state, overwrite=overwrite)


def _atomic_joblib(payload: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        joblib.dump(payload, temporary, compress=3)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        frame.to_parquet(
            temporary,
            engine="pyarrow",
            compression="zstd",
            index=True,
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _candidate_payload(
    recipe: Mapping[str, Any],
    candidate_name: str,
    specification: Mapping[str, Any],
    candidate: TrainedCandidate,
) -> dict[str, Any]:
    return {
        "artifact_schema_version": 1,
        "artifact_type": "baram_2024_gate_candidate",
        "recipe_name": recipe["recipe_name"],
        "candidate_name": candidate_name,
        "candidate_config": dict(specification),
        "scope": candidate.scope,
        "groups": TARGET_COLS,
        "training_rows": candidate.training_rows,
        "feature_names": candidate.feature_names,
        "train_end_inclusive": str(TRAIN_END_INCLUSIVE),
        "valid_start": str(VALID_START),
        "valid_end_inclusive": str(VALID_END_INCLUSIVE),
        "models": candidate.models,
    }


def evaluate_gate(
    *,
    raw_dir: Path,
    cache_dir: Path,
    out_dir: Path,
    config_path: Path,
    overwrite: bool,
) -> dict[str, Any]:
    """Run the locked gate once and persist models, OOF, metrics, and manifest."""

    recipe = read_recipe(config_path)
    if not recipe["recipe_locked"]:
        raise ValueError("recipe_locked is false; refusing to consume the 2024 gate")
    recipe_sha256 = sha256_file(config_path)
    label_path, cache_paths = _input_paths(raw_dir, cache_dir)
    paths = _output_paths(out_dir, recipe)

    # Reserve before _read_labels is called. A crash from this point onward is a
    # partial/consumed gate and cannot silently be repeated.
    _reserve_gate(
        out_dir,
        paths,
        overwrite=overwrite,
        config_path=config_path,
        recipe=recipe,
    )
    write_json_atomic(paths["recipe"], recipe, overwrite=overwrite)

    run_started = time.perf_counter()
    data, valid_answer, availability, input_files = _load_gate_bundle(
        label_path,
        cache_paths,
    )

    trained: dict[str, TrainedCandidate] = {}
    candidate_seconds: dict[str, float] = {}
    for candidate_name, raw_specification in recipe["models"].items():
        specification = dict(raw_specification)
        if not specification["enabled"]:
            continue
        candidate_started = time.perf_counter()
        print(f"training locked gate candidate {candidate_name} ...", flush=True)
        candidate = train_candidate(
            candidate_name,
            specification,
            recipe=recipe,
            data=data,
            trained=trained,
        )
        trained[candidate_name] = candidate
        candidate_seconds[candidate_name] = time.perf_counter() - candidate_started
        _atomic_joblib(
            _candidate_payload(recipe, candidate_name, specification, candidate),
            paths[f"model:{candidate_name}"],
        )
        _atomic_parquet(
            candidate.predictions,
            paths[f"prediction:{candidate_name}"],
        )

    final_prediction = _assemble_final_predictions(recipe, trained)
    if sha256_file(config_path) != recipe_sha256:
        raise RuntimeError("locked recipe file changed while the gate was running")
    if not final_prediction.index.equals(valid_answer.index):
        raise AssertionError("gate OOF index changed during assembly")
    if tuple(final_prediction.columns) != TARGET_COLS:
        raise AssertionError("gate OOF group columns/order changed")
    if not np.isfinite(final_prediction.to_numpy()).all():
        raise ValueError("gate OOF contains non-finite values")
    details = score_details(valid_answer, final_prediction)
    official_metrics = asdict(details)
    _atomic_parquet(final_prediction, paths["oof"])

    candidate_summary = {
        name: {
            "scope": candidate.scope,
            "training_rows": candidate.training_rows,
            "feature_counts": {
                group: len(columns)
                for group, columns in candidate.feature_names.items()
            },
            "fit_predict_seconds": candidate_seconds[name],
        }
        for name, candidate in trained.items()
    }
    elapsed = time.perf_counter() - run_started
    metrics_payload = {
        "gate_schema_version": GATE_SCHEMA_VERSION,
        "recipe_name": recipe["recipe_name"],
        "recipe_sha256": sha256_file(config_path),
        "evaluation": "official_three_group_2024_untouched_gate",
        "train_end_inclusive": str(TRAIN_END_INCLUSIVE),
        "valid_start": str(VALID_START),
        "valid_end_inclusive": str(VALID_END_INCLUSIVE),
        "valid_rows": int(len(valid_answer)),
        "label_availability": availability,
        "official_metrics": official_metrics,
        "candidates": candidate_summary,
        "total_fit_score_seconds": elapsed,
        "selection_or_tuning_performed": False,
    }
    write_json_atomic(paths["metrics"], metrics_payload, overwrite=overwrite)

    completed_state = {
        "gate_schema_version": GATE_SCHEMA_VERSION,
        "status": "completed",
        "reserved_utc": json.loads(paths["state"].read_text(encoding="utf-8"))[
            "reserved_utc"
        ],
        "completed_utc": utc_now(),
        "recipe_name": recipe["recipe_name"],
        "recipe_sha256": sha256_file(config_path),
        "train_end_inclusive": str(TRAIN_END_INCLUSIVE),
        "valid_start": str(VALID_START),
        "valid_end_inclusive": str(VALID_END_INCLUSIVE),
        "valid_rows": int(len(valid_answer)),
    }
    # This is the same reserved state file, so internal completion always
    # replaces it even on the first run.
    write_json_atomic(paths["state"], completed_state, overwrite=True)

    output_files = [
        path for key, path in paths.items() if key != "manifest"
    ]
    manifest = make_manifest(
        artifact_type="baram_2024_untouched_gate",
        parameters={
            "recipe": recipe,
            "capacities_kwh": CAPACITY_KWH,
            "target_columns": TARGET_COLS,
            "train_end_inclusive": str(TRAIN_END_INCLUSIVE),
            "valid_start": str(VALID_START),
            "valid_end_inclusive": str(VALID_END_INCLUSIVE),
            "one_time_gate_guard": True,
            "selection_or_tuning_performed": False,
        },
        input_files=[config_path, *input_files],
        output_files=output_files,
        results={
            "official_metrics": official_metrics,
            "label_availability": availability,
            "candidates": candidate_summary,
            "oof_sha256": sha256_file(paths["oof"]),
            "valid_rows": int(len(valid_answer)),
            "total_fit_score_seconds": elapsed,
        },
        project_dir=PROJECT_DIR,
    )
    write_json_atomic(paths["manifest"], manifest, overwrite=overwrite)
    return {
        "status": "completed",
        "oof": str(paths["oof"]),
        "metrics": str(paths["metrics"]),
        "manifest": str(paths["manifest"]),
        "official_metrics": official_metrics,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    raw_dir = args.raw_dir.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    recipe = read_recipe(config_path)
    label_path, cache_paths = _input_paths(raw_dir, cache_dir)

    if args.dry_run:
        if args.overwrite or args.confirm_consume_2024_gate:
            raise ValueError(
                "--dry-run cannot be combined with --overwrite or "
                "--confirm-consume-2024-gate"
            )
        summary = dry_run_summary(
            recipe,
            label_path=label_path,
            cache_paths=cache_paths,
            out_dir=out_dir,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if not args.confirm_consume_2024_gate:
        raise ValueError(
            "real evaluation requires --confirm-consume-2024-gate; no label "
            "values have been read"
        )
    result = evaluate_gate(
        raw_dir=raw_dir,
        cache_dir=cache_dir,
        out_dir=out_dir,
        config_path=config_path,
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

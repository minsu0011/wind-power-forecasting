"""Build turbine-resolved spatial augmentation without touching the v1 cache.

Example:
    python scripts/build_spatial_v2.py --raw-dir C:/data/open \
        --cache-dir artifacts/cache_v2

The six outputs contain augmentation columns only.  Consumers should align
their DatetimeIndex exactly and concatenate them to the existing v1 weather
cache.  No label or SCADA file is opened by this program.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.manifest import make_manifest, write_json_atomic  # noqa: E402
from src.metric import CAPACITY_KWH, TARGET_COLS  # noqa: E402
from src.spatial_v2 import (  # noqa: E402
    SpatialV2Config,
    build_spatial_v2_train_test,
    load_turbine_table,
)


RAW_RELATIVE_FILES = (
    Path("train/ldaps_train.csv"),
    Path("train/gfs_train.csv"),
    Path("test/ldaps_test.csv"),
    Path("test/gfs_test.csv"),
    Path("info.xlsx"),
)
FORBIDDEN_NAME_TOKENS = ("scada", "target", "label", "power_kw")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build leakage-safe turbine-level spatial/wake augmentation Parquets."
        )
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="official data root containing train/, test/, and info.xlsx",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="new destination directory (recommended: artifacts/cache_v2)",
    )
    parser.add_argument(
        "--compression",
        choices=("snappy", "zstd", "gzip", "brotli", "none"),
        default="zstd",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace only spatial-v2 files already present in --cache-dir",
    )
    return parser.parse_args(argv)


def _output_paths(cache_dir: Path) -> dict[tuple[str, str], Path]:
    return {
        (split, group): cache_dir / f"{group}_spatial_v2_{split}.parquet"
        for split in ("train", "test")
        for group in TARGET_COLS
    }


def _write_parquet_atomic(
    frame: pd.DataFrame, destination: Path, *, compression: str | None
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        frame.to_parquet(
            temporary, engine="pyarrow", compression=compression, index=True
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_frame(frame: pd.DataFrame, *, group: str, split: str) -> None:
    expected_rows = 26_304 if split == "train" else 8_760
    if len(frame) != expected_rows:
        raise ValueError(f"{split}/{group}: expected {expected_rows} rows, got {len(frame)}")
    if not isinstance(frame.index, pd.DatetimeIndex) or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{split}/{group}: index must be sorted DatetimeIndex")
    if not frame.index.is_unique:
        raise ValueError(f"{split}/{group}: timestamp index is duplicated")
    forbidden = [
        column
        for column in frame.columns
        if any(token in column.lower() for token in FORBIDDEN_NAME_TOKENS)
        or column.startswith("kpx_group_")
    ]
    if forbidden:
        raise ValueError(f"{split}/{group}: forbidden columns: {forbidden}")
    if frame.isna().any().any() or not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"{split}/{group}: non-finite augmentation values")


def _frame_summary(frame: pd.DataFrame) -> dict[str, object]:
    land_columns = [column for column in frame if "land_" in column]
    return {
        "rows": int(len(frame)),
        "features": int(frame.shape[1]),
        "time_start": frame.index.min().isoformat(),
        "time_end": frame.index.max().isoformat(),
        "index_unique": bool(frame.index.is_unique),
        "finite": bool(np.isfinite(frame.to_numpy()).all()),
        "land_feature_ranges": {
            column: [float(frame[column].min()), float(frame[column].max())]
            for column in land_columns
        },
    }


def _turbine_audit(turbines: pd.DataFrame) -> dict[str, object]:
    groups: dict[str, object] = {}
    for group in TARGET_COLS:
        rows = turbines.loc[turbines["group"] == group].copy()
        capacity = rows["capacity_mw"].to_numpy(dtype=float)
        weights = capacity / capacity.sum()
        unweighted = rows[["latitude", "longitude"]].mean().to_numpy(dtype=float)
        weighted = np.average(
            rows[["latitude", "longitude"]].to_numpy(dtype=float),
            axis=0,
            weights=weights,
        )
        groups[group] = {
            "turbine_count": int(len(rows)),
            "turbine_capacity_mw": capacity.tolist(),
            "capacity_sum_kwh": float(capacity.sum() * 1_000.0),
            "expected_capacity_kwh": float(CAPACITY_KWH[group]),
            "unweighted_centroid": unweighted.tolist(),
            "capacity_weighted_centroid": weighted.tolist(),
            "centroid_difference_degrees": (weighted - unweighted).tolist(),
            "coordinates": rows[
                ["manufacturer", "turbine_number", "latitude", "longitude"]
            ].to_dict(orient="records"),
        }
    return {
        "v1_behavior": (
            "src/features.py interpolates once at the arithmetic group centroid; "
            "it does not retain turbine-level nearest/IDW or directional layout."
        ),
        "equal_capacity_consequence": (
            "All turbines within each group have equal rated capacity, so the v1 "
            "arithmetic centroid equals the capacity-weighted centroid here."
        ),
        "v2_behavior": (
            "NWP is interpolated per turbine, then aggregated with info.xlsx "
            "nameplate-capacity weights; wake/layout uses forecast vectors only."
        ),
        "groups": groups,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    raw_dir = args.raw_dir.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    raw_files = [raw_dir / relative for relative in RAW_RELATIVE_FILES]
    missing = [path for path in raw_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing official inputs: {missing}")

    outputs = _output_paths(cache_dir)
    manifest_path = cache_dir / "spatial_v2_manifest.json"
    existing = [path for path in [*outputs.values(), manifest_path] if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "refusing to overwrite spatial-v2 outputs; use a new --cache-dir or "
            f"pass --overwrite: {existing}"
        )
    # These names are intentionally disjoint from the v1 *_weather_*.parquet cache.
    if any("_weather_" in path.name for path in outputs.values()):
        raise AssertionError("spatial-v2 output unexpectedly aliases a v1 cache name")

    config = SpatialV2Config()
    print(f"raw data: {raw_dir}")
    print(f"v2 cache: {cache_dir}")
    print("building turbine-resolved augmentation (no labels/SCADA) ...", flush=True)
    train, test, builder = build_spatial_v2_train_test(raw_dir, config=config)

    summaries: dict[str, dict[str, object]] = {"train": {}, "test": {}}
    written: list[Path] = []
    compression = None if args.compression == "none" else args.compression
    for split, frames in (("train", train), ("test", test)):
        for group in TARGET_COLS:
            frame = frames[group]
            _validate_frame(frame, group=group, split=split)
            destination = outputs[(split, group)]
            print(f"writing {destination.name} {frame.shape} ...", flush=True)
            _write_parquet_atomic(frame, destination, compression=compression)
            readback = pd.read_parquet(destination)
            if not readback.index.equals(frame.index) or tuple(readback.columns) != tuple(frame.columns):
                raise IOError(f"Parquet readback mismatch: {destination}")
            written.append(destination)
            summaries[split][group] = _frame_summary(frame)

    turbines = load_turbine_table(raw_dir / "info.xlsx")
    parameters = {
        "spatial_v2_config": asdict(config),
        "feature_names": list(builder.feature_names_ or ()),
        "parquet_compression": args.compression,
        "leakage_policy": {
            "opened_labels": False,
            "opened_scada": False,
            "same_data_available_run_dynamics_only": True,
            "test_missing_fallback": (
                "same-run same-grid temporal neighbours, then same-time spatial "
                "median, then training-weather median"
            ),
        },
        "turbine_audit": _turbine_audit(turbines),
    }
    manifest = make_manifest(
        artifact_type="baram_turbine_spatial_v2_augmentation_cache",
        parameters=parameters,
        input_files=raw_files,
        output_files=written,
        results=summaries,
        project_dir=PROJECT_DIR,
    )
    write_json_atomic(manifest_path, manifest, overwrite=args.overwrite)
    print(f"complete: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build reproducible BARAM weather-feature Parquet caches.

Example:
    python scripts/build_features.py --raw-dir C:/data/open --cache-dir artifacts/cache
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.features import (  # noqa: E402
    WeatherFeatureConfig,
    build_train_test_weather_features,
)
from src.manifest import make_manifest, write_json_atomic  # noqa: E402


GROUPS = ("kpx_group_1", "kpx_group_2", "kpx_group_3")
RAW_RELATIVE_FILES = (
    Path("train/ldaps_train.csv"),
    Path("train/gfs_train.csv"),
    Path("test/ldaps_test.csv"),
    Path("test/gfs_test.csv"),
    Path("info.xlsx"),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build leakage-safe, group-specific weather features and a hashed manifest."
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
        help="destination directory for six Parquet files and feature_manifest.json",
    )
    parser.add_argument(
        "--compression",
        choices=("snappy", "zstd", "gzip", "brotli", "none"),
        default="zstd",
        help="Parquet compression codec (default: zstd)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="atomically replace an existing complete or partial cache",
    )
    return parser.parse_args(argv)


def _validate_raw_dir(raw_dir: Path) -> list[Path]:
    raw_files = [raw_dir / relative for relative in RAW_RELATIVE_FILES]
    missing = [path for path in raw_files if not path.is_file()]
    if missing:
        rendered = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"raw data directory is incomplete:\n{rendered}")
    return raw_files


def _output_paths(cache_dir: Path) -> dict[tuple[str, str], Path]:
    return {
        (split, group): cache_dir / f"{group}_weather_{split}.parquet"
        for split in ("train", "test")
        for group in GROUPS
    }


def _write_parquet_atomic(
    frame: pd.DataFrame,
    destination: Path,
    *,
    compression: str | None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        frame.to_parquet(
            temporary,
            engine="pyarrow",
            compression=compression,
            index=True,
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _frame_summary(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(frame.shape[0]),
        "features": int(frame.shape[1]),
        "index_name": frame.index.name,
        "time_start": pd.Timestamp(frame.index.min()).isoformat(),
        "time_end": pd.Timestamp(frame.index.max()).isoformat(),
        "dtypes": sorted({str(dtype) for dtype in frame.dtypes}),
        "missing_cells": int(frame.isna().sum().sum()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    raw_dir = args.raw_dir.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    raw_files = _validate_raw_dir(raw_dir)
    outputs = _output_paths(cache_dir)
    manifest_path = cache_dir / "feature_manifest.json"
    existing = [path for path in [*outputs.values(), manifest_path] if path.exists()]
    if existing and not args.overwrite:
        rendered = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(
            "cache outputs already exist; pass --overwrite to replace them atomically:\n"
            f"{rendered}"
        )

    print(f"raw data: {raw_dir}")
    print(f"cache:    {cache_dir}")
    print("building train/test weather features ...", flush=True)
    config = WeatherFeatureConfig()
    train, test, builder = build_train_test_weather_features(
        raw_dir, config=config, info_path=raw_dir / "info.xlsx"
    )

    summaries: dict[str, dict[str, object]] = {"train": {}, "test": {}}
    written: list[Path] = []
    compression = None if args.compression == "none" else args.compression
    for split, frames in (("train", train), ("test", test)):
        for group in GROUPS:
            frame = frames[group]
            destination = outputs[(split, group)]
            print(f"writing {destination.name} {frame.shape} ...", flush=True)
            _write_parquet_atomic(frame, destination, compression=compression)
            written.append(destination)
            summaries[split][group] = _frame_summary(frame)

    parameters = {
        "weather_feature_config": asdict(config),
        "group_sites": {
            name: asdict(site) for name, site in builder.group_sites.items()
        },
        "feature_names": list(builder.feature_names_ or ()),
        "parquet_compression": args.compression,
    }
    print("hashing inputs and outputs for feature_manifest.json ...", flush=True)
    manifest = make_manifest(
        artifact_type="baram_weather_feature_cache",
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

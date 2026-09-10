"""Strict-forward three-seed bagging audit for corrected-v3 shared_q07.

The experiment has one candidate only: replace the strict capacity-factor
seed-42 shared_q07 prediction by the arithmetic mean of seeds 42, 7, and 2026.
Stage 1 is completed and locked without reading 2024 labels/features/candidate
predictions.  Stage 2 may read 2024 only after verifying that lock.  No source
artifact is ever overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import io
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

from src.manifest import (  # noqa: E402
    describe_file,
    git_state,
    package_versions,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from src.features import (  # noqa: E402
    AVAILABLE_COL,
    COORD_COLS,
    GRID_COL,
    SOURCE_VARIABLES,
    TIME_COL,
    WeatherFeatureBuilder,
    load_group_sites,
)
from src.metric import CAPACITY_KWH, TARGET_COLS, group_metrics  # noqa: E402


PREREGISTER_SHA256 = (
    "6a1d78137a1bc6c0a129382564dc18eb4bc8abaf467a19466faf26398784adbe"
)
SEEDS = (42, 7, 2026)
NEW_SEEDS = (7, 2026)
YEAR_2022_START = pd.Timestamp("2022-01-01 01:00:00")
YEAR_2022_END = pd.Timestamp("2023-01-01 00:00:00")
YEAR_2023_START = pd.Timestamp("2023-01-01 01:00:00")
YEAR_2023_END = pd.Timestamp("2024-01-01 00:00:00")
YEAR_2024_START = pd.Timestamp("2024-01-01 01:00:00")
YEAR_2024_END = pd.Timestamp("2025-01-01 00:00:00")
H2_2023_START = pd.Timestamp("2023-07-01 01:00:00")
Q4_2023_START = pd.Timestamp("2023-10-01 01:00:00")
H2_2024_START = pd.Timestamp("2024-07-01 01:00:00")
EXPECTED_ROWS_PRE2024 = 17_520
EXPECTED_ROWS_THROUGH2024 = 26_304
EXPECTED_TEST_ROWS = 8_760
RAW_ROWS_PER_TIMESTAMP = {"ldaps": 16, "gfs": 9}
EXPECTED_STAGE1_RAW_ROWS = {
    source: EXPECTED_ROWS_PRE2024 * rows_per_timestamp
    for source, rows_per_timestamp in RAW_ROWS_PER_TIMESTAMP.items()
}
EXPECTED_STAGE1_NEXT_TIMESTAMP = YEAR_2024_START
OFFICIAL_NWP_PREFIX_SHA256 = {
    "ldaps": "2ae4fd41ea9c0ba06c0aa6873cbf9bd3b810b0de83051d47f9b80efa0a47c262",
    "gfs": "c1a3e8fb072101d9d24161320375c53849cc803a039e17d222ff27ad92f65934",
}
OFFICIAL_STAGE1_PREFIX_BYTES = {
    "labels": 742_551,
    "ldaps": 86_258_374,
    "gfs": 56_101_144,
}
OFFICIAL_LABEL_PREFIX_SHA256 = (
    "ee9707cc81229db760fc035d0bf1d10eb3586d2fef95849bdf463d73fae3aacd"
)
OFFICIAL_NWP_FULL_SHA256 = {
    "ldaps": "61ae944e7ae1fcb17391be6737792a2205c6507bf2446ed5d9d0daf07fdea026",
    "gfs": "cd56b67d357e7bbaff5d0d51d3537d935c9e7a3f012e9f37516bdc4d38c66a5d",
}

COMPONENTS = (
    "lgb_l1",
    "lgb_q07",
    "shared_l1",
    "shared_q07",
    "top200_q07",
    "energy_q06",
)

STAGE1_REQUIRED_SLICES = {
    "kpx_group_1": ("full", "H1", "H2"),
    "kpx_group_2": ("full", "H1", "H2"),
    "kpx_group_3": ("full", "Q3", "Q4"),
}
LOCK_RULE = "all required slice deltas strictly > 0"
CANDIDATE_DESCRIPTION = "mean seeds 42,7,2026 replacing shared_q07 only"

DEV_COMPONENT_FILES = {
    "lgb_l1": "dev2023_lgb_l1_eligible_n1500.parquet",
    "lgb_q07": "dev2023_lgb_q07_eligible.parquet",
    "shared_l1": "dev2023_shared_l1_eligible.parquet",
    "shared_q07": "dev2023_shared_q07_eligible.parquet",
    "top200_q07": "dev2023_lgb_top200_q07_eligible.parquet",
    "energy_q06": "dev2023_lgb_q06_energywt_eligible.parquet",
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
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--recipe",
        type=Path,
        default=PROJECT_DIR / "configs" / "train_final.v3.locked.json",
    )
    parser.add_argument(
        "--preregister",
        type=Path,
        default=PROJECT_DIR
        / "configs"
        / "shared_q07_multiseed_preregister.json",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=PROJECT_DIR / "artifacts",
        help="read-only root containing existing OOF/final artifacts",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_DIR
        / "artifacts"
        / "postgate"
        / "shared_q07_multiseed_strict",
    )
    parser.add_argument(
        "--stage",
        choices=("stage1", "stage2", "final", "all"),
        default="all",
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
        frame.to_parquet(
            temporary, engine="pyarrow", compression="zstd", index=True
        )
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


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        frame.to_csv(
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


def _frame_sha256(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    schema = [(str(column), str(dtype)) for column, dtype in frame.dtypes.items()]
    digest.update(json.dumps(schema, sort_keys=True).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        _json_ready(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _csv_prefix_identity(
    path: Path, *, data_rows: int, byte_limit: int | None = None
) -> tuple[str, int]:
    """Hash and size the header plus exactly ``data_rows`` records."""

    if data_rows < 1:
        raise ValueError("data_rows must be positive")
    digest = hashlib.sha256()
    byte_count = 0
    newline_count = 0
    if byte_limit is not None:
        with path.open("rb", buffering=0) as stream:
            remaining = int(byte_limit)
            while remaining:
                chunk = stream.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise AssertionError(f"{path} ended inside fixed byte prefix")
                digest.update(chunk)
                byte_count += len(chunk)
                newline_count += chunk.count(b"\n")
                remaining -= len(chunk)
        if newline_count != data_rows + 1:
            raise AssertionError(
                f"{path} fixed prefix line count changed: "
                f"{newline_count} != {data_rows + 1}"
            )
        return digest.hexdigest(), byte_count
    with path.open("rb", buffering=0) as stream:
        header = stream.readline()
        if not header:
            raise ValueError(f"empty CSV: {path}")
        digest.update(header)
        byte_count += len(header)
        for row_number in range(1, data_rows + 1):
            line = stream.readline()
            if not line:
                raise AssertionError(
                    f"{path} ended before bounded row {row_number}/{data_rows}"
                )
            digest.update(line)
            byte_count += len(line)
    return digest.hexdigest(), byte_count


def _sha256_csv_prefix(
    path: Path, *, data_rows: int, byte_limit: int | None = None
) -> str:
    return _csv_prefix_identity(
        path, data_rows=data_rows, byte_limit=byte_limit
    )[0]


class _BoundedRawReader(io.RawIOBase):
    """Non-fileno binary stream that exposes at most a fixed byte prefix."""

    def __init__(self, path: Path, *, byte_limit: int) -> None:
        super().__init__()
        self._stream = path.open("rb", buffering=0)
        self._remaining = int(byte_limit)
        self.bytes_returned = 0

    @property
    def underlying_position(self) -> int:
        return int(self._stream.tell())

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: bytearray) -> int:
        if self._remaining <= 0:
            return 0
        requested = min(len(buffer), self._remaining)
        chunk = self._stream.read(requested)
        if not chunk:
            return 0
        size = len(chunk)
        buffer[:size] = chunk
        self._remaining -= size
        self.bytes_returned += size
        return size

    def fileno(self) -> int:
        raise io.UnsupportedOperation("bounded reader deliberately exposes no fileno")

    def close(self) -> None:
        if not self.closed:
            self._stream.close()
        super().close()


def _next_csv_first_field(
    path: Path, *, after_data_rows: int, prefix_bytes: int | None = None
) -> str:
    """Read only the first cell following a CSV prefix.

    This boundary probe deliberately stops at the first comma, so no future
    weather value is read or materialized.
    """

    with path.open("rb", buffering=0) as stream:
        if prefix_bytes is None:
            if not stream.readline():
                raise ValueError(f"empty CSV: {path}")
            for row_number in range(1, after_data_rows + 1):
                if not stream.readline():
                    raise AssertionError(
                        f"{path} ended before boundary row "
                        f"{row_number}/{after_data_rows}"
                    )
        else:
            stream.seek(int(prefix_bytes), os.SEEK_SET)
        first_field = bytearray()
        while True:
            byte = stream.read(1)
            if not byte:
                raise AssertionError(f"{path} has no row after bounded prefix")
            if byte == b",":
                break
            if byte in {b"\r", b"\n"}:
                raise AssertionError(f"{path} boundary row has no comma")
            first_field.extend(byte)
    return first_field.decode("utf-8-sig")


def _snapshot_file(path: Path) -> dict[str, Any]:
    return {"snapshot_kind": "whole_file", **describe_file(path)}


def _snapshot_csv_prefix(
    path: Path, *, data_rows: int, prefix_bytes: int | None = None
) -> dict[str, Any]:
    record = describe_file(path, include_hash=False)
    prefix_sha256, observed_prefix_bytes = _csv_prefix_identity(
        path, data_rows=data_rows, byte_limit=prefix_bytes
    )
    record.update(
        {
            "snapshot_kind": "csv_prefix",
            "prefix_data_rows": int(data_rows),
            "prefix_bytes": int(observed_prefix_bytes),
            "prefix_sha256": prefix_sha256,
            "whole_file_hash_omitted_to_avoid_forbidden_suffix_access": True,
        }
    )
    return record


def _refresh_snapshot_entry(record: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(record["path"]))
    kind = str(record["snapshot_kind"])
    if kind == "whole_file":
        return _snapshot_file(path)
    if kind == "csv_prefix":
        return _snapshot_csv_prefix(
            path,
            data_rows=int(record["prefix_data_rows"]),
            prefix_bytes=int(record["prefix_bytes"]),
        )
    raise AssertionError(f"unknown snapshot kind: {kind}")


def _refresh_snapshot(snapshot: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        str(name): _refresh_snapshot_entry(record)
        for name, record in snapshot.items()
    }


def _assert_snapshot_equal(
    before: Mapping[str, Any], after: Mapping[str, Any], *, name: str
) -> None:
    if _json_ready(before) != _json_ready(after):
        raise AssertionError(f"{name} changed during the locked stage")


def _provenance_paths(
    *, recipe_path: Path, preregister_path: Path
) -> dict[str, Path]:
    return {
        "runner": Path(__file__).resolve(),
        "features": PROJECT_DIR / "src" / "features.py",
        "metric": PROJECT_DIR / "src" / "metric.py",
        "manifest": PROJECT_DIR / "src" / "manifest.py",
        "test": PROJECT_DIR / "tests" / "test_shared_q07_multiseed.py",
        "preregister": preregister_path.resolve(),
        "recipe": recipe_path.resolve(),
    }


def _snapshot_named_files(paths: Mapping[str, Path]) -> dict[str, Any]:
    return {str(name): _snapshot_file(path) for name, path in paths.items()}


def _provenance_sha_map(snapshot: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    return {str(name): str(record["sha256"]) for name, record in snapshot.items()}


def _stage1_historical_paths(artifact_root: Path) -> dict[str, Path]:
    dev_root = artifact_root / "oof"
    paths = {
        f"dev_component_{name}": dev_root / filename
        for name, filename in DEV_COMPONENT_FILES.items()
    }
    paths.update(
        {
            "dev_reference": dev_root / "dev2023_locked_v3.parquet",
            "g3_historical_candidates": dev_root / "g3dev2023h2_candidates.parquet",
            "dev_seed42_model": artifact_root
            / "models"
            / "dev2023_shared_q07_eligible.joblib",
            "g3_seed42_model": artifact_root
            / "models"
            / "g3dev2023h2_candidates.joblib",
        }
    )
    return paths


def _stage1_input_snapshot(raw_dir: Path, artifact_root: Path) -> dict[str, Any]:
    return {
        "labels_prefix": _snapshot_csv_prefix(
            raw_dir / "train" / "train_labels.csv",
            data_rows=EXPECTED_ROWS_PRE2024,
            prefix_bytes=OFFICIAL_STAGE1_PREFIX_BYTES["labels"],
        ),
        "ldaps_prefix": _snapshot_csv_prefix(
            raw_dir / "train" / "ldaps_train.csv",
            data_rows=EXPECTED_STAGE1_RAW_ROWS["ldaps"],
            prefix_bytes=OFFICIAL_STAGE1_PREFIX_BYTES["ldaps"],
        ),
        "gfs_prefix": _snapshot_csv_prefix(
            raw_dir / "train" / "gfs_train.csv",
            data_rows=EXPECTED_STAGE1_RAW_ROWS["gfs"],
            prefix_bytes=OFFICIAL_STAGE1_PREFIX_BYTES["gfs"],
        ),
        "info_workbook": _snapshot_file(raw_dir / "info.xlsx"),
        **_snapshot_named_files(_stage1_historical_paths(artifact_root)),
    }


def _read_bounded_weather_csv(
    path: Path,
    *,
    source: str,
    expected_timestamps: int,
    expected_start: pd.Timestamp,
    expected_end: pd.Timestamp,
    expected_next: pd.Timestamp,
    expected_prefix_bytes: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Materialize an exact chronological raw-NWP prefix.

    Only required columns are parsed, compact locked dtypes are used, and
    ``nrows`` is the exact timestamp count times the official grid count.  The
    next-boundary check reads only its timestamp cell, never its weather values.
    """

    if source not in SOURCE_VARIABLES:
        raise KeyError(f"unknown weather source: {source}")
    rows_per_timestamp = RAW_ROWS_PER_TIMESTAMP[source]
    expected_rows = expected_timestamps * rows_per_timestamp
    required = [
        TIME_COL,
        AVAILABLE_COL,
        GRID_COL,
        *COORD_COLS,
        *SOURCE_VARIABLES[source],
    ]
    with path.open("rb") as stream:
        header_line = stream.readline()
    header = pd.read_csv(
        io.BytesIO(header_line), nrows=0, encoding="utf-8-sig"
    ).columns
    missing = set(required).difference(header)
    if missing:
        raise ValueError(f"{path} missing {source} columns: {sorted(missing)}")
    dtypes: dict[str, str] = {GRID_COL: "int16"}
    dtypes.update(
        {column: "float32" for column in [*COORD_COLS, *SOURCE_VARIABLES[source]]}
    )
    prefix_sha256, prefix_bytes = _csv_prefix_identity(
        path, data_rows=expected_rows, byte_limit=expected_prefix_bytes
    )
    bounded_raw = _BoundedRawReader(path, byte_limit=prefix_bytes)
    try:
        with io.BufferedReader(bounded_raw, buffer_size=1024 * 1024) as bounded:
            frame = pd.read_csv(
                bounded,
                usecols=required,
                dtype=dtypes,
                parse_dates=[TIME_COL, AVAILABLE_COL],
                nrows=expected_rows,
                encoding="utf-8-sig",
                low_memory=False,
                memory_map=False,
            )
            bytes_returned = bounded_raw.bytes_returned
            underlying_position = bounded_raw.underlying_position
    finally:
        bounded_raw.close()
    if bytes_returned != prefix_bytes:
        raise AssertionError(
            f"{source} parser did not consume exact bounded prefix bytes: "
            f"{bytes_returned} != {prefix_bytes}"
        )
    if underlying_position != prefix_bytes:
        raise AssertionError(
            f"{source} underlying file read crossed bounded prefix: "
            f"{underlying_position} != {prefix_bytes}"
        )
    if len(frame) != expected_rows:
        raise AssertionError(
            f"{source} bounded rows changed: {len(frame)} != {expected_rows}"
        )
    times = pd.DatetimeIndex(frame[TIME_COL])
    if not times.is_monotonic_increasing:
        raise AssertionError(f"{source} raw prefix is not chronological")
    counts = frame.groupby(TIME_COL, sort=False)[GRID_COL].size()
    expected_index = pd.date_range(
        expected_start, expected_end, periods=expected_timestamps
    )
    if not counts.index.equals(expected_index):
        raise AssertionError(f"{source} bounded timestamp sequence changed")
    if not counts.eq(rows_per_timestamp).all():
        raise AssertionError(f"{source} grid rows per timestamp changed")
    next_timestamp = pd.Timestamp(
        _next_csv_first_field(
            path, after_data_rows=expected_rows, prefix_bytes=prefix_bytes
        )
    )
    if next_timestamp != expected_next:
        raise AssertionError(
            f"{source} next boundary changed: {next_timestamp} != {expected_next}"
        )
    evidence = {
        "source": source,
        "path": str(path.resolve()),
        "nrows_argument": int(expected_rows),
        "physical_byte_limit": int(prefix_bytes),
        "physical_bytes_returned": int(bytes_returned),
        "underlying_file_position_after_parse": int(underlying_position),
        "physical_prefix_sha256": prefix_sha256,
        "suffix_bytes_exposed_to_parser": 0,
        "rows_per_timestamp": int(rows_per_timestamp),
        "materialized_rows": int(len(frame)),
        "materialized_timestamp_rows": int(len(counts)),
        "materialized_start": counts.index.min().isoformat(),
        "materialized_end": counts.index.max().isoformat(),
        "next_boundary_timestamp_only": next_timestamp.isoformat(),
        "next_boundary_weather_values_read": False,
        "usecols": list(required),
        "dtypes": {name: str(dtype) for name, dtype in frame.dtypes.items()},
        "materialized_frame_sha256": _frame_sha256(frame),
        "memory_map": False,
    }
    return frame, evidence


def _read_stage1_raw_features(
    raw_dir: Path, labels: pd.DataFrame
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    raw_frames: dict[str, pd.DataFrame] = {}
    raw_evidence: dict[str, Any] = {}
    for source in ("ldaps", "gfs"):
        frame, evidence = _read_bounded_weather_csv(
            raw_dir / "train" / f"{source}_train.csv",
            source=source,
            expected_timestamps=EXPECTED_ROWS_PRE2024,
            expected_start=YEAR_2022_START,
            expected_end=YEAR_2023_END,
            expected_next=EXPECTED_STAGE1_NEXT_TIMESTAMP,
            expected_prefix_bytes=OFFICIAL_STAGE1_PREFIX_BYTES[source],
        )
        raw_frames[source] = frame
        raw_evidence[source] = evidence
        if evidence["physical_prefix_sha256"] != OFFICIAL_NWP_PREFIX_SHA256[source]:
            raise AssertionError(f"{source} official bounded-prefix hash changed")
    sites = load_group_sites(raw_dir / "info.xlsx")
    builder = WeatherFeatureBuilder(group_sites=sites)
    features = builder.fit_transform(raw_frames["ldaps"], raw_frames["gfs"])
    canonical: tuple[str, ...] | None = None
    feature_evidence: dict[str, Any] = {}
    for group in TARGET_COLS:
        frame = features[group]
        frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
        if not frame.index.equals(labels.index):
            raise AssertionError(f"raw-built {group} feature index differs from labels")
        if canonical is not None and tuple(frame.columns) != canonical:
            raise AssertionError("raw-built group feature schemas differ")
        canonical = tuple(frame.columns)
        forbidden = [
            column
            for column in frame.columns
            if "scada" in str(column).lower()
            or str(column).lower() in {"target", "actual", *TARGET_COLS}
        ]
        if forbidden:
            raise AssertionError(f"forbidden leakage features found: {forbidden[:5]}")
        values = frame.to_numpy(dtype=np.float32, copy=False)
        if not np.isfinite(values).all():
            raise AssertionError(f"{group} raw-built features contain non-finite values")
        features[group] = frame.astype(np.float32, copy=False)
        feature_evidence[group] = {
            "rows": int(len(frame)),
            "columns": int(frame.shape[1]),
            "start": frame.index.min().isoformat(),
            "end": frame.index.max().isoformat(),
            "frame_sha256": _frame_sha256(features[group]),
        }
    if canonical is None or len(canonical) != 612:
        raise AssertionError(
            f"expected 612 locked v1 features, found {0 if canonical is None else len(canonical)}"
        )
    return features, {
        "raw_prefix": raw_evidence,
        "built_features": feature_evidence,
        "builder": {
            "class": "src.features.WeatherFeatureBuilder",
            "feature_count": len(canonical),
            "group_sites_source": str((raw_dir / "info.xlsx").resolve()),
        },
    }


def _loaded_slice_source(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    record = describe_file(path, include_hash=False)
    record.update(
        {
            "loaded_slice_sha256": _frame_sha256(frame),
            "loaded_rows": int(len(frame)),
            "loaded_start": frame.index.min().isoformat(),
            "loaded_end": frame.index.max().isoformat(),
            "whole_file_hash_deliberately_omitted_prelock": True,
        }
    )
    return record


def _verify_preregister(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256_file(path)
    if observed != PREREGISTER_SHA256:
        raise AssertionError(
            "preregister hash changed; refusing candidate reads or fits: "
            f"{observed} != {PREREGISTER_SHA256}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["single_candidate"]["candidate_count"] != 1:
        raise AssertionError("preregister candidate count is not one")
    if tuple(payload["single_candidate"]["seeds"]) != SEEDS:
        raise AssertionError("preregister seeds changed")
    return {"path": str(path.resolve()), "sha256": observed}


def _read_recipe(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    specification = payload["models"]["shared_q07"]
    required = {
        "scope": "shared",
        "target_scale": "capacity_factor",
        "objective": "quantile",
        "alpha": 0.7,
        "row_filter": "eligible",
        "features": "all",
    }
    for key, expected in required.items():
        if specification.get(key) != expected:
            raise AssertionError(f"shared_q07 {key} changed")
    expected_parameters = {
        "n_estimators": 1500,
        "learning_rate": 0.025,
        "num_leaves": 31,
        "min_child_samples": 40,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.05,
        "reg_lambda": 2.0,
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
    }
    if specification["params"] != expected_parameters:
        raise AssertionError("shared_q07 exact 1500-tree parameters changed")
    if int(payload["n_jobs"]) != 7 or str(payload["device"]) != "cpu":
        raise AssertionError("locked CPU/n_jobs execution contract changed")
    if tuple(payload["ensemble"]["weights"]) != TARGET_COLS:
        raise AssertionError("ensemble group ordering changed")
    return payload


def _read_labels(
    path: Path,
    *,
    nrows: int,
    expected_end: pd.Timestamp,
    prefix_bytes: int | None = None,
    expected_prefix_sha256: str | None = None,
) -> pd.DataFrame:
    if prefix_bytes is None:
        labels = pd.read_csv(path, encoding="utf-8-sig", nrows=nrows)
    else:
        observed_sha256, observed_bytes = _csv_prefix_identity(
            path, data_rows=nrows, byte_limit=prefix_bytes
        )
        if observed_bytes != prefix_bytes:
            raise AssertionError("label bounded byte count changed")
        if (
            expected_prefix_sha256 is not None
            and observed_sha256 != expected_prefix_sha256
        ):
            raise AssertionError("label bounded-prefix hash changed")
        bounded_raw = _BoundedRawReader(path, byte_limit=prefix_bytes)
        try:
            with io.BufferedReader(bounded_raw, buffer_size=1024 * 1024) as bounded:
                labels = pd.read_csv(
                    bounded,
                    encoding="utf-8-sig",
                    nrows=nrows,
                    memory_map=False,
                )
                bytes_returned = bounded_raw.bytes_returned
                underlying_position = bounded_raw.underlying_position
        finally:
            bounded_raw.close()
        if bytes_returned != prefix_bytes or underlying_position != prefix_bytes:
            raise AssertionError("label reader crossed or did not consume exact prefix")
    if tuple(labels.columns) != ("kst_dtm", *TARGET_COLS):
        raise ValueError("train label schema changed")
    times = pd.to_datetime(labels.pop("kst_dtm"), errors="raise")
    labels.index = pd.DatetimeIndex(times, name="forecast_kst_dtm")
    labels = labels.astype(float)
    if len(labels) != nrows:
        raise AssertionError(f"bounded label row count changed: {len(labels)}")
    if labels.index.min() != YEAR_2022_START or labels.index.max() != expected_end:
        raise AssertionError("bounded label time range changed")
    if not labels.index.is_unique or not labels.index.is_monotonic_increasing:
        raise AssertionError("label index must be unique and sorted")
    if np.isinf(labels.to_numpy()).any():
        raise AssertionError("labels contain infinity")
    return labels


def _read_features(
    cache_dir: Path,
    labels: pd.DataFrame,
    *,
    expected_end: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    features: dict[str, pd.DataFrame] = {}
    canonical: tuple[str, ...] | None = None
    for group in TARGET_COLS:
        path = cache_dir / f"{group}_weather_train.parquet"
        frame = pd.read_parquet(
            path,
            engine="pyarrow",
            filters=[("forecast_kst_dtm", "<=", expected_end)],
        )
        frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
        if not frame.index.equals(labels.index):
            raise AssertionError(f"bounded {group} feature index differs from labels")
        if tuple(frame.columns) != canonical and canonical is not None:
            raise AssertionError("group feature schemas differ")
        canonical = tuple(frame.columns)
        forbidden = [
            column
            for column in frame.columns
            if "scada" in str(column).lower()
            or str(column).lower() in {"target", "actual", *TARGET_COLS}
        ]
        if forbidden:
            raise AssertionError(f"forbidden leakage features found: {forbidden[:5]}")
        values = frame.to_numpy(dtype=np.float32, copy=False)
        if not np.isfinite(values).all():
            raise AssertionError(f"{group} bounded features contain non-finite values")
        features[group] = frame.astype(np.float32, copy=False)
    assert canonical is not None
    if len(canonical) != 612:
        raise AssertionError(f"expected 612 locked v1 features, found {len(canonical)}")
    return features


def _read_test_features(
    cache_dir: Path, expected_index: pd.DatetimeIndex
) -> dict[str, pd.DataFrame]:
    features: dict[str, pd.DataFrame] = {}
    canonical: tuple[str, ...] | None = None
    for group in TARGET_COLS:
        path = cache_dir / f"{group}_weather_test.parquet"
        frame = pd.read_parquet(path, engine="pyarrow")
        frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
        if not frame.index.equals(expected_index):
            raise AssertionError(f"{group} test cache differs from sample times")
        if canonical is not None and tuple(frame.columns) != canonical:
            raise AssertionError("test group feature schemas differ")
        canonical = tuple(frame.columns)
        if not np.isfinite(frame.to_numpy(dtype=np.float32, copy=False)).all():
            raise AssertionError(f"{group} test features contain non-finite values")
        features[group] = frame.astype(np.float32, copy=False)
    return features


def _interval(index: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    mask = (index >= start) & (index <= end)
    if not np.any(mask):
        raise AssertionError(f"empty interval {start} .. {end}")
    return np.asarray(mask)


def _fit_shared_q07(
    *,
    seed: int,
    recipe: Mapping[str, Any],
    labels: pd.DataFrame,
    features: Mapping[str, pd.DataFrame],
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    predict_indices: Mapping[str, pd.DatetimeIndex],
) -> tuple[Any, pd.DataFrame, dict[str, int], list[str]]:
    from lightgbm import LGBMRegressor

    specification = recipe["models"]["shared_q07"]
    selected = list(map(str, features[TARGET_COLS[0]].columns))
    train_parts: list[pd.DataFrame] = []
    target_parts: list[pd.Series] = []
    counts: dict[str, int] = {}
    time_mask = _interval(labels.index, train_start, train_end)
    for group_number, group in enumerate(TARGET_COLS, start=1):
        capacity = CAPACITY_KWH[group]
        target = labels[group]
        eligible = (
            time_mask
            & np.isfinite(target.to_numpy(dtype=float, copy=False))
            & (target.to_numpy(dtype=float, copy=False) >= 0.10 * capacity)
        )
        counts[group] = int(eligible.sum())
        if counts[group] == 0:
            continue
        selected_index = labels.index[eligible]
        part = features[group].loc[selected_index, selected].copy()
        part["model__group_id"] = np.float32(group_number)
        part.reset_index(drop=True, inplace=True)
        group_target = (
            target.loc[selected_index].astype(float) / capacity
        ).reset_index(drop=True)
        train_parts.append(part)
        target_parts.append(group_target)
    if not train_parts:
        raise AssertionError("no pooled eligible rows")
    train_x = pd.concat(train_parts, axis=0, ignore_index=True, copy=False)
    train_y = pd.concat(target_parts, axis=0, ignore_index=True)
    if not np.isfinite(train_y.to_numpy()).all():
        raise AssertionError("capacity-factor target contains non-finite values")
    if float(train_y.min()) < 0.1 - 1e-12:
        raise AssertionError("eligible capacity-factor target fell below 0.1")
    params = dict(specification["params"])
    params.update(
        {
            "objective": "quantile",
            "alpha": 0.7,
            "random_state": int(seed),
            "n_jobs": int(recipe["n_jobs"]),
        }
    )
    model = LGBMRegressor(**params)
    model.fit(train_x, train_y)
    prediction = pd.DataFrame(dtype=float)
    for group_number, group in enumerate(TARGET_COLS, start=1):
        index = predict_indices.get(group)
        if index is None:
            continue
        if not index.isin(features[group].index).all():
            raise AssertionError(f"{group} prediction index not in bounded features")
        valid_x = features[group].loc[index, selected].copy()
        valid_x["model__group_id"] = np.float32(group_number)
        values = np.asarray(model.predict(valid_x), dtype=float) * CAPACITY_KWH[group]
        if prediction.empty:
            prediction = pd.DataFrame(index=index)
        if not prediction.index.equals(index):
            raise AssertionError("one fit requested non-common prediction indices")
        prediction[group] = values
    prediction.index.name = "forecast_kst_dtm"
    if not np.isfinite(prediction.to_numpy()).all():
        raise AssertionError("shared_q07 prediction contains non-finite values")
    return model, prediction, counts, [*selected, "model__group_id"]


def _read_prediction(
    path: Path,
    *,
    expected_index: pd.DatetimeIndex,
    required_groups: Sequence[str],
) -> pd.DataFrame:
    frame = pd.read_parquet(path, engine="pyarrow")
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    missing = [group for group in required_groups if group not in frame.columns]
    if missing:
        raise AssertionError(f"{path} missing groups {missing}")
    frame = frame.loc[:, list(required_groups)].astype(float)
    if not frame.index.equals(expected_index):
        raise AssertionError(f"{path} index differs from locked validation interval")
    if not np.isfinite(frame.to_numpy()).all():
        raise AssertionError(f"{path} contains non-finite predictions")
    return frame


def _assemble_group(
    recipe: Mapping[str, Any],
    group: str,
    components: Mapping[str, pd.Series],
) -> pd.Series:
    weights = recipe["ensemble"]["weights"][group]
    if set(weights) != set(COMPONENTS):
        raise AssertionError(f"{group} candidate component set changed")
    index = components[COMPONENTS[0]].index
    raw = np.zeros(len(index), dtype=float)
    for name in COMPONENTS:
        values = components[name]
        if not values.index.equals(index):
            raise AssertionError(f"{group}/{name} index mismatch")
        raw += float(weights[name]) * values.to_numpy(dtype=float, copy=False)
    affine = recipe["ensemble"]["affine"][group]
    clip = recipe["ensemble"]["clip"][group]
    capacity = CAPACITY_KWH[group]
    lower = float(clip["lower_capacity_fraction"]) * capacity
    upper = float(clip["upper_capacity_fraction"]) * capacity
    output = np.clip(
        float(affine["scale"]) * raw + float(affine["bias_kwh"]),
        lower,
        upper,
    )
    bins = recipe["ensemble"]["power_bins"][group]
    if bins is not None:
        edges = np.asarray(bins["edges_cf"], dtype=float)
        deltas = np.asarray(bins["delta_kwh"], dtype=float)
        positions = np.searchsorted(edges[1:-1], output / capacity, side="right")
        output = np.clip(output + deltas[positions], lower, upper)
    return pd.Series(output, index=index, name=group)


def _score_group(
    actual: pd.Series, prediction: pd.Series, group: str
) -> dict[str, float | int]:
    if not actual.index.equals(prediction.index):
        raise AssertionError("actual/prediction index mismatch")
    result = group_metrics(actual, prediction, CAPACITY_KWH[group], group_name=group)
    group_score = 0.5 * result.one_minus_nmae + 0.5 * result.ficr
    return {
        "score": float(group_score),
        "one_minus_nmae": float(result.one_minus_nmae),
        "ficr": float(result.ficr),
        "n_evaluated": int(result.n_evaluated),
    }


def _slice_comparison(
    *,
    labels: pd.Series,
    baseline: pd.Series,
    candidate: pd.Series,
    group: str,
    slices: Mapping[str, tuple[pd.Timestamp, pd.Timestamp]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, (start, end) in slices.items():
        index = baseline.index[(baseline.index >= start) & (baseline.index <= end)]
        before = _score_group(labels.loc[index], baseline.loc[index], group)
        after = _score_group(labels.loc[index], candidate.loc[index], group)
        output[name] = {
            "baseline": before,
            "candidate": after,
            "delta": float(after["score"] - before["score"]),
        }
    return output


def _component_paths(base: Path, mapping: Mapping[str, str]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, relative in mapping.items():
        path = Path(relative)
        paths[name] = path if path.is_absolute() else base / path
    return paths


def _stage1(
    *,
    raw_dir: Path,
    cache_dir: Path,
    artifact_root: Path,
    out_dir: Path,
    recipe: Mapping[str, Any],
    preregister: Mapping[str, Any],
    recipe_path: Path,
) -> dict[str, Any]:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"stage1 requires a new empty output directory: {out_dir}"
        )
    provenance_paths = _provenance_paths(
        recipe_path=recipe_path,
        preregister_path=Path(str(preregister["path"])),
    )
    provenance_before = _snapshot_named_files(provenance_paths)
    input_snapshot_before = _stage1_input_snapshot(raw_dir, artifact_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    prereg_copy = out_dir / "preregister.json"
    shutil.copyfile(preregister["path"], prereg_copy)
    if sha256_file(prereg_copy) != PREREGISTER_SHA256:
        raise AssertionError("copied preregistration hash changed")

    label_path = raw_dir / "train" / "train_labels.csv"
    labels = _read_labels(
        label_path,
        nrows=EXPECTED_ROWS_PRE2024,
        expected_end=YEAR_2023_END,
        prefix_bytes=OFFICIAL_STAGE1_PREFIX_BYTES["labels"],
        expected_prefix_sha256=OFFICIAL_LABEL_PREFIX_SHA256,
    )
    features, raw_feature_contract = _read_stage1_raw_features(raw_dir, labels)
    g12_valid_index = labels.index[
        _interval(labels.index, YEAR_2023_START, YEAR_2023_END)
    ]
    g3_valid_index = labels.index[
        _interval(labels.index, H2_2023_START, YEAR_2023_END)
    ]

    output_files: list[Path] = [prereg_copy]
    training: dict[str, Any] = {}
    g12_new: dict[int, pd.DataFrame] = {}
    for seed in NEW_SEEDS:
        print(f"stage1: fitting pooled 2022 -> 2023 g1/g2 seed={seed}", flush=True)
        model, prediction, counts, feature_names = _fit_shared_q07(
            seed=seed,
            recipe=recipe,
            labels=labels,
            features=features,
            train_start=YEAR_2022_START,
            train_end=YEAR_2022_END,
            predict_indices={
                "kpx_group_1": g12_valid_index,
                "kpx_group_2": g12_valid_index,
            },
        )
        model_path = out_dir / "models" / f"stage1_g12_seed{seed}.joblib"
        pred_path = out_dir / "oof" / f"stage1_g12_seed{seed}.parquet"
        _atomic_joblib(model, model_path)
        _atomic_parquet(prediction, pred_path)
        output_files.extend([model_path, pred_path])
        g12_new[seed] = prediction
        training[f"g12_seed{seed}"] = {
            "train": [YEAR_2022_START, YEAR_2022_END],
            "valid": [YEAR_2023_START, YEAR_2023_END],
            "eligible_rows": counts,
            "features": len(feature_names),
            "model_path": model_path,
            "model_sha256": sha256_file(model_path),
            "prediction_path": pred_path,
            "prediction_sha256": sha256_file(pred_path),
        }

    g3_new: dict[int, pd.DataFrame] = {}
    for seed in NEW_SEEDS:
        print(f"stage1: fitting pooled 2023H1 -> 2023H2 g3 seed={seed}", flush=True)
        model, prediction, counts, feature_names = _fit_shared_q07(
            seed=seed,
            recipe=recipe,
            labels=labels,
            features=features,
            train_start=YEAR_2023_START,
            train_end=H2_2023_START - pd.Timedelta(hours=1),
            predict_indices={"kpx_group_3": g3_valid_index},
        )
        model_path = out_dir / "models" / f"stage1_g3_seed{seed}.joblib"
        pred_path = out_dir / "oof" / f"stage1_g3_seed{seed}.parquet"
        _atomic_joblib(model, model_path)
        _atomic_parquet(prediction, pred_path)
        output_files.extend([model_path, pred_path])
        g3_new[seed] = prediction
        training[f"g3_seed{seed}"] = {
            "train": [YEAR_2023_START, H2_2023_START - pd.Timedelta(hours=1)],
            "valid": [H2_2023_START, YEAR_2023_END],
            "eligible_rows": counts,
            "features": len(feature_names),
            "model_path": model_path,
            "model_sha256": sha256_file(model_path),
            "prediction_path": pred_path,
            "prediction_sha256": sha256_file(pred_path),
        }

    dev_root = artifact_root / "oof"
    dev_paths = {
        name: dev_root / filename for name, filename in DEV_COMPONENT_FILES.items()
    }
    dev_components = {
        name: _read_prediction(
            path,
            expected_index=g12_valid_index,
            required_groups=("kpx_group_1", "kpx_group_2"),
        )
        for name, path in dev_paths.items()
    }
    dev_reference_path = dev_root / "dev2023_locked_v3.parquet"
    dev_reference = _read_prediction(
        dev_reference_path,
        expected_index=g12_valid_index,
        required_groups=("kpx_group_1", "kpx_group_2"),
    )
    baseline = pd.DataFrame(index=g12_valid_index, columns=TARGET_COLS, dtype=float)
    candidate = pd.DataFrame(index=g12_valid_index, columns=TARGET_COLS, dtype=float)
    for group in ("kpx_group_1", "kpx_group_2"):
        base_parts = {name: frame[group] for name, frame in dev_components.items()}
        baseline[group] = _assemble_group(recipe, group, base_parts)
        mean_shared = (
            dev_components["shared_q07"][group]
            + g12_new[7][group]
            + g12_new[2026][group]
        ) / 3.0
        candidate_parts = dict(base_parts)
        candidate_parts["shared_q07"] = mean_shared
        candidate[group] = _assemble_group(recipe, group, candidate_parts)
    reconstruction_difference = float(
        np.max(
            np.abs(
                baseline.loc[:, ["kpx_group_1", "kpx_group_2"]].to_numpy()
                - dev_reference.to_numpy()
            )
        )
    )
    if reconstruction_difference != 0.0:
        raise AssertionError(
            f"locked dev reconstruction differs by {reconstruction_difference}"
        )

    g3_source_path = dev_root / "g3dev2023h2_candidates.parquet"
    dev_seed42_model_path = artifact_root / "models/dev2023_shared_q07_eligible.joblib"
    g3_seed42_model_path = artifact_root / "models/g3dev2023h2_candidates.joblib"
    g3_raw = pd.read_parquet(g3_source_path, engine="pyarrow")
    g3_raw.index = pd.DatetimeIndex(g3_raw.index, name="forecast_kst_dtm")
    if not g3_raw.index.equals(g3_valid_index):
        raise AssertionError("g3 historical candidate index changed")
    required_g3 = {"q07", "shared_l1", "shared_q07", "top200q07", "ewq06"}
    if not required_g3.issubset(g3_raw.columns):
        raise AssertionError("g3 historical candidates missing locked components")
    g3_parts = {
        "lgb_l1": g3_raw["q07"] * 0.0,
        "lgb_q07": g3_raw["q07"],
        "shared_l1": g3_raw["shared_l1"],
        "shared_q07": g3_raw["shared_q07"],
        "top200_q07": g3_raw["top200q07"],
        "energy_q06": g3_raw["ewq06"],
    }
    baseline.loc[g3_valid_index, "kpx_group_3"] = _assemble_group(
        recipe, "kpx_group_3", g3_parts
    )
    g3_candidate_parts = dict(g3_parts)
    g3_candidate_parts["shared_q07"] = (
        g3_raw["shared_q07"]
        + g3_new[7]["kpx_group_3"]
        + g3_new[2026]["kpx_group_3"]
    ) / 3.0
    candidate.loc[g3_valid_index, "kpx_group_3"] = _assemble_group(
        recipe, "kpx_group_3", g3_candidate_parts
    )

    baseline_path = out_dir / "oof" / "stage1_corrected_v3_baseline.parquet"
    candidate_path = out_dir / "oof" / "stage1_multiseed_candidate.parquet"
    _atomic_parquet(baseline, baseline_path)
    _atomic_parquet(candidate, candidate_path)
    output_files.extend([baseline_path, candidate_path])

    comparisons = {
        "kpx_group_1": _slice_comparison(
            labels=labels["kpx_group_1"],
            baseline=baseline["kpx_group_1"],
            candidate=candidate["kpx_group_1"],
            group="kpx_group_1",
            slices={
                "full": (YEAR_2023_START, YEAR_2023_END),
                "H1": (YEAR_2023_START, H2_2023_START - pd.Timedelta(hours=1)),
                "H2": (H2_2023_START, YEAR_2023_END),
            },
        ),
        "kpx_group_2": _slice_comparison(
            labels=labels["kpx_group_2"],
            baseline=baseline["kpx_group_2"],
            candidate=candidate["kpx_group_2"],
            group="kpx_group_2",
            slices={
                "full": (YEAR_2023_START, YEAR_2023_END),
                "H1": (YEAR_2023_START, H2_2023_START - pd.Timedelta(hours=1)),
                "H2": (H2_2023_START, YEAR_2023_END),
            },
        ),
        "kpx_group_3": _slice_comparison(
            labels=labels["kpx_group_3"],
            baseline=baseline.loc[g3_valid_index, "kpx_group_3"],
            candidate=candidate.loc[g3_valid_index, "kpx_group_3"],
            group="kpx_group_3",
            slices={
                "full": (H2_2023_START, YEAR_2023_END),
                "Q3": (H2_2023_START, Q4_2023_START - pd.Timedelta(hours=1)),
                "Q4": (Q4_2023_START, YEAR_2023_END),
            },
        ),
    }
    locked_groups = [
        group
        for group, group_results in comparisons.items()
        if all(float(values["delta"]) > 0.0 for values in group_results.values())
    ]
    provenance_after = _snapshot_named_files(provenance_paths)
    input_snapshot_after = _stage1_input_snapshot(raw_dir, artifact_root)
    _assert_snapshot_equal(
        provenance_before, provenance_after, name="source/config provenance snapshot"
    )
    _assert_snapshot_equal(
        input_snapshot_before,
        input_snapshot_after,
        name="stage1 bounded/raw/historical input snapshot",
    )
    provenance_snapshot_sha256 = _canonical_sha256(provenance_before)
    input_snapshot_sha256 = _canonical_sha256(input_snapshot_before)
    comparisons_sha256 = _canonical_sha256(comparisons)
    sources = [
        _loaded_slice_source(label_path, labels),
        *provenance_before.values(),
        *input_snapshot_before.values(),
    ]
    result = {
        "schema_version": 2,
        "experiment_id": "shared_q07_multiseed_strict_forward_stage1_v2",
        "created_utc": utc_now(),
        "preregister": preregister,
        "candidate_count": 1,
        "candidate": CANDIDATE_DESCRIPTION,
        "forbidden_2024_data_read_occurred": False,
        "stage1_cache_files_read": False,
        "cache_dir_deliberately_excluded_from_stage1": str(cache_dir.resolve()),
        "raw_feature_contract": raw_feature_contract,
        "bounded_label_rows": int(len(labels)),
        "bounded_time": [labels.index.min(), labels.index.max()],
        "training": training,
        "baseline_reconstruction_max_abs_diff": reconstruction_difference,
        "comparisons": comparisons,
        "comparisons_sha256": comparisons_sha256,
        "lock_rule": LOCK_RULE,
        "locked_groups": locked_groups,
        "provenance_before": provenance_before,
        "provenance_after": provenance_after,
        "provenance_snapshot_sha256": provenance_snapshot_sha256,
        "provenance_sha256": _provenance_sha_map(provenance_before),
        "stage1_input_snapshot_before": input_snapshot_before,
        "stage1_input_snapshot_after": input_snapshot_after,
        "stage1_input_snapshot_sha256": input_snapshot_sha256,
        "snapshots_unchanged_during_stage1": True,
        "sources": sources,
        "outputs_before_result": [describe_file(path) for path in output_files],
        "leaderboard_score_claim": False,
    }
    result_path = out_dir / "stage1_results.json"
    _write_json(result_path, result)
    lock = {
        "schema_version": 2,
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "stage1_results_sha256": sha256_file(result_path),
        "locked_groups": locked_groups,
        "comparisons_sha256": comparisons_sha256,
        "selection_rule": LOCK_RULE,
        "provenance_snapshot_sha256": provenance_snapshot_sha256,
        "provenance_sha256": _provenance_sha_map(provenance_before),
        "stage1_input_snapshot_sha256": input_snapshot_sha256,
        "stage2_candidate_frozen": True,
        "no_2024_reselection": True,
    }
    lock_path = out_dir / "stage1_lock.json"
    _write_json(lock_path, lock)
    print("stage1 locked groups:", locked_groups, flush=True)
    return lock


def _locked_groups_from_comparisons(
    comparisons: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> list[str]:
    if set(comparisons) != set(STAGE1_REQUIRED_SLICES):
        raise AssertionError("stage1 comparison group set changed")
    locked: list[str] = []
    for group in TARGET_COLS:
        group_results = comparisons[group]
        required_slices = STAGE1_REQUIRED_SLICES[group]
        if set(group_results) != set(required_slices):
            raise AssertionError(f"{group} comparison slice set changed")
        passes = []
        for slice_name in required_slices:
            values = group_results[slice_name]
            if set(values) != {"baseline", "candidate", "delta"}:
                raise AssertionError(f"{group}/{slice_name} comparison schema changed")
            expected_delta = float(values["candidate"]["score"]) - float(
                values["baseline"]["score"]
            )
            if float(values["delta"]) != expected_delta:
                raise AssertionError(
                    f"{group}/{slice_name} stored delta differs from scores"
                )
            passes.append(expected_delta > 0.0)
        if all(passes):
            locked.append(group)
    return locked


def _load_stage1_lock(out_dir: Path) -> dict[str, Any]:
    result_path = out_dir / "stage1_results.json"
    lock_path = out_dir / "stage1_lock.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["preregister_sha256"] != PREREGISTER_SHA256:
        raise AssertionError("stage1 lock preregister hash differs")
    if lock["stage1_results_sha256"] != sha256_file(result_path):
        raise AssertionError("stage1 result changed after lock")
    if not lock.get("stage2_candidate_frozen") or not lock.get(
        "no_2024_reselection"
    ):
        raise AssertionError("stage1 lock does not freeze stage2")
    if int(result.get("candidate_count", -1)) != 1:
        raise AssertionError("stage1 result candidate count changed")
    if result.get("candidate") != CANDIDATE_DESCRIPTION:
        raise AssertionError("stage1 candidate definition changed")
    if result.get("lock_rule") != LOCK_RULE or lock.get("selection_rule") != LOCK_RULE:
        raise AssertionError("stage1 selection rule changed")
    comparisons = result.get("comparisons")
    if not isinstance(comparisons, Mapping):
        raise AssertionError("stage1 comparisons missing")
    comparisons_sha256 = _canonical_sha256(comparisons)
    if result.get("comparisons_sha256") != comparisons_sha256:
        raise AssertionError("stage1 comparison hash differs")
    if lock.get("comparisons_sha256") != comparisons_sha256:
        raise AssertionError("stage1 lock comparison hash differs")
    recalculated_groups = _locked_groups_from_comparisons(comparisons)
    if list(result.get("locked_groups", [])) != recalculated_groups:
        raise AssertionError("stage1 result locked_groups differs from recalculated rule")
    if list(lock.get("locked_groups", [])) != recalculated_groups:
        raise AssertionError("stage1 lock locked_groups differs from result/rule")

    provenance_before = result.get("provenance_before")
    provenance_after = result.get("provenance_after")
    input_before = result.get("stage1_input_snapshot_before")
    input_after = result.get("stage1_input_snapshot_after")
    if not all(
        isinstance(value, Mapping)
        for value in (provenance_before, provenance_after, input_before, input_after)
    ):
        raise AssertionError("stage1 source/input snapshots missing")
    _assert_snapshot_equal(
        provenance_before, provenance_after, name="recorded provenance snapshot"
    )
    _assert_snapshot_equal(
        input_before, input_after, name="recorded stage1 input snapshot"
    )
    provenance_sha256 = _canonical_sha256(provenance_before)
    input_sha256 = _canonical_sha256(input_before)
    if result.get("provenance_snapshot_sha256") != provenance_sha256:
        raise AssertionError("stage1 result provenance digest differs")
    if lock.get("provenance_snapshot_sha256") != provenance_sha256:
        raise AssertionError("stage1 lock provenance digest differs")
    expected_sha_map = _provenance_sha_map(provenance_before)
    if result.get("provenance_sha256") != expected_sha_map:
        raise AssertionError("stage1 result provenance SHA map differs")
    if lock.get("provenance_sha256") != expected_sha_map:
        raise AssertionError("stage1 lock provenance SHA map differs")
    if result.get("stage1_input_snapshot_sha256") != input_sha256:
        raise AssertionError("stage1 result input digest differs")
    if lock.get("stage1_input_snapshot_sha256") != input_sha256:
        raise AssertionError("stage1 lock input digest differs")

    current_provenance = _refresh_snapshot(provenance_before)
    current_inputs = _refresh_snapshot(input_before)
    _assert_snapshot_equal(
        provenance_before, current_provenance, name="current provenance snapshot"
    )
    _assert_snapshot_equal(input_before, current_inputs, name="current stage1 inputs")
    return lock


def _bit_exact_frame_comparison(
    observed: pd.DataFrame, expected: pd.DataFrame
) -> dict[str, Any]:
    if not observed.index.equals(expected.index):
        raise AssertionError("bit-exact audit index mismatch")
    if tuple(observed.columns) != tuple(expected.columns):
        raise AssertionError("bit-exact audit column mismatch")
    if tuple(map(str, observed.dtypes)) != tuple(map(str, expected.dtypes)):
        raise AssertionError("bit-exact audit dtype mismatch")
    observed_values = np.ascontiguousarray(observed.to_numpy(copy=False))
    expected_values = np.ascontiguousarray(expected.to_numpy(copy=False))
    bit_exact = observed_values.tobytes() == expected_values.tobytes()
    if not bit_exact:
        raise AssertionError("bit-exact audit value bytes differ")
    finite = np.isfinite(observed_values) & np.isfinite(expected_values)
    max_abs = (
        float(np.max(np.abs(observed_values[finite] - expected_values[finite])))
        if np.any(finite)
        else 0.0
    )
    return {
        "rows": int(len(observed)),
        "columns": int(observed.shape[1]),
        "index_exact": True,
        "column_order_exact": True,
        "dtype_exact": True,
        "value_bits_exact": True,
        "max_abs_finite": max_abs,
        "observed_frame_sha256": _frame_sha256(observed),
        "expected_frame_sha256": _frame_sha256(expected),
    }


def _postlock_stage1_audit(
    *, raw_dir: Path, cache_dir: Path, artifact_root: Path, out_dir: Path
) -> dict[str, Any]:
    """Post-selection-only cache and legacy-output bit-exact comparison."""

    lock = _load_stage1_lock(out_dir)
    audit_path = out_dir / "postlock_bit_exact_audit.json"
    if audit_path.exists():
        raise FileExistsError(audit_path)
    result = json.loads((out_dir / "stage1_results.json").read_text(encoding="utf-8"))
    full_raw_hashes: dict[str, Any] = {}
    for source in ("ldaps", "gfs"):
        raw_path = raw_dir / "train" / f"{source}_train.csv"
        observed_full_sha256 = sha256_file(raw_path)
        if observed_full_sha256 != OFFICIAL_NWP_FULL_SHA256[source]:
            raise AssertionError(f"{source} official full-file hash changed")
        full_raw_hashes[source] = {
            **describe_file(raw_path, include_hash=False),
            "sha256": observed_full_sha256,
            "read_only_after_stage1_selection_lock": True,
        }
    cache_checks: dict[str, Any] = {}
    for group in TARGET_COLS:
        cache_path = cache_dir / f"{group}_weather_train.parquet"
        cached = pd.read_parquet(cache_path, engine="pyarrow").iloc[
            :EXPECTED_ROWS_PRE2024
        ]
        cached.index = pd.DatetimeIndex(cached.index, name="forecast_kst_dtm")
        expected_hash = result["raw_feature_contract"]["built_features"][group][
            "frame_sha256"
        ]
        observed_hash = _frame_sha256(cached)
        if observed_hash != expected_hash:
            raise AssertionError(f"{group} raw-prefix/cache feature hash differs")
        cache_checks[group] = {
            "selection_input": False,
            "read_only_after_stage1_lock": True,
            "rows_compared": int(len(cached)),
            "columns_compared": int(cached.shape[1]),
            "frame_sha256": observed_hash,
            "bit_exact": True,
            "cache_file": describe_file(cache_path),
        }

    legacy_dir = artifact_root / "postgate" / "shared_q07_multiseed"
    relative_outputs = (
        Path("oof/stage1_g12_seed7.parquet"),
        Path("oof/stage1_g12_seed2026.parquet"),
        Path("oof/stage1_g3_seed7.parquet"),
        Path("oof/stage1_g3_seed2026.parquet"),
        Path("oof/stage1_corrected_v3_baseline.parquet"),
        Path("oof/stage1_multiseed_candidate.parquet"),
    )
    output_checks: dict[str, Any] = {}
    for relative in relative_outputs:
        observed_path = out_dir / relative
        expected_path = legacy_dir / relative
        observed = pd.read_parquet(observed_path, engine="pyarrow")
        expected = pd.read_parquet(expected_path, engine="pyarrow")
        comparison = _bit_exact_frame_comparison(observed, expected)
        comparison.update(
            {
                "strict_output": describe_file(observed_path),
                "legacy_reference": describe_file(expected_path),
            }
        )
        output_checks[str(relative).replace("\\", "/")] = comparison

    legacy_result_path = legacy_dir / "stage1_results.json"
    legacy_result = json.loads(legacy_result_path.read_text(encoding="utf-8"))
    comparisons_exact = result["comparisons"] == legacy_result["comparisons"]
    locked_groups_exact = result["locked_groups"] == legacy_result["locked_groups"]
    reconstruction_exact = (
        result["baseline_reconstruction_max_abs_diff"]
        == legacy_result["baseline_reconstruction_max_abs_diff"]
    )
    if not comparisons_exact or not locked_groups_exact or not reconstruction_exact:
        raise AssertionError("strict/legacy Stage1 numeric result differs")
    audit = {
        "schema_version": 2,
        "created_utc": utc_now(),
        "stage1_lock_sha256": sha256_file(out_dir / "stage1_lock.json"),
        "stage1_locked_groups": list(lock["locked_groups"]),
        "selection_was_already_immutable_before_this_audit": True,
        "cache_was_not_a_stage1_model_input": True,
        "official_raw_full_file_hashes_postlock_only": full_raw_hashes,
        "cache_feature_prefix_checks": cache_checks,
        "legacy_output_value_checks": output_checks,
        "stage1_numeric_result_exact": {
            "comparisons": comparisons_exact,
            "locked_groups": locked_groups_exact,
            "baseline_reconstruction": reconstruction_exact,
        },
        "all_feature_and_prediction_values_bit_exact": True,
        "legacy_stage1_result": describe_file(legacy_result_path),
    }
    _write_json(audit_path, audit)
    return audit


def _stage2(
    *,
    raw_dir: Path,
    cache_dir: Path,
    artifact_root: Path,
    out_dir: Path,
    recipe: Mapping[str, Any],
) -> dict[str, Any]:
    lock = _load_stage1_lock(out_dir)
    locked_groups = tuple(lock["locked_groups"])
    result_path = out_dir / "stage2_results.json"
    promotion_path = out_dir / "stage2_promotion_lock.json"
    if result_path.exists() or promotion_path.exists():
        raise FileExistsError("stage2 outputs already exist")
    if not locked_groups:
        result = {
            "schema_version": 2,
            "created_utc": utc_now(),
            "stage1_lock_sha256": sha256_file(out_dir / "stage1_lock.json"),
            "provenance_snapshot_sha256": lock["provenance_snapshot_sha256"],
            "provenance_sha256": lock["provenance_sha256"],
            "stage1_input_snapshot_sha256": lock[
                "stage1_input_snapshot_sha256"
            ],
            "locked_groups": [],
            "promoted_groups": [],
            "2024_read": False,
            "reason": "no group passed every stage1 slice",
        }
        _write_json(result_path, result)
        promotion = {
            "schema_version": 2,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "stage2_results_sha256": sha256_file(result_path),
            "provenance_snapshot_sha256": lock["provenance_snapshot_sha256"],
            "provenance_sha256": lock["provenance_sha256"],
            "promoted_groups": [],
        }
        _write_json(promotion_path, promotion)
        return promotion

    # The first 2024 value read occurs only after the immutable stage1 lock above.
    label_path = raw_dir / "train" / "train_labels.csv"
    labels = _read_labels(
        label_path,
        nrows=EXPECTED_ROWS_THROUGH2024,
        expected_end=YEAR_2024_END,
    )
    features = _read_features(cache_dir, labels, expected_end=YEAR_2024_END)
    valid_index = labels.index[_interval(labels.index, YEAR_2024_START, YEAR_2024_END)]

    new_predictions: dict[str, dict[int, pd.Series]] = {
        group: {} for group in locked_groups
    }
    training: dict[str, Any] = {}
    if any(group in locked_groups for group in TARGET_COLS[:2]):
        requested = {
            group: valid_index
            for group in TARGET_COLS[:2]
            if group in locked_groups
        }
        for seed in NEW_SEEDS:
            print(f"stage2: fitting pooled 2022-2023 -> 2024 g1/g2 seed={seed}", flush=True)
            model, prediction, counts, names = _fit_shared_q07(
                seed=seed,
                recipe=recipe,
                labels=labels,
                features=features,
                train_start=YEAR_2022_START,
                train_end=YEAR_2023_END,
                predict_indices=requested,
            )
            model_path = out_dir / "models" / f"stage2_g12_seed{seed}.joblib"
            pred_path = out_dir / "oof" / f"stage2_g12_seed{seed}.parquet"
            _atomic_joblib(model, model_path)
            _atomic_parquet(prediction, pred_path)
            for group in requested:
                new_predictions[group][seed] = prediction[group]
            training[f"g12_seed{seed}"] = {
                "train": [YEAR_2022_START, YEAR_2023_END],
                "valid": [YEAR_2024_START, YEAR_2024_END],
                "eligible_rows": counts,
                "features": len(names),
                "model_path": model_path,
                "model_sha256": sha256_file(model_path),
                "prediction_path": pred_path,
                "prediction_sha256": sha256_file(pred_path),
            }
    if "kpx_group_3" in locked_groups:
        for seed in NEW_SEEDS:
            print(f"stage2: fitting pooled 2023 -> 2024 g3 seed={seed}", flush=True)
            model, prediction, counts, names = _fit_shared_q07(
                seed=seed,
                recipe=recipe,
                labels=labels,
                features=features,
                train_start=YEAR_2023_START,
                train_end=YEAR_2023_END,
                predict_indices={"kpx_group_3": valid_index},
            )
            model_path = out_dir / "models" / f"stage2_g3_seed{seed}.joblib"
            pred_path = out_dir / "oof" / f"stage2_g3_seed{seed}.parquet"
            _atomic_joblib(model, model_path)
            _atomic_parquet(prediction, pred_path)
            new_predictions["kpx_group_3"][seed] = prediction["kpx_group_3"]
            training[f"g3_seed{seed}"] = {
                "train": [YEAR_2023_START, YEAR_2023_END],
                "valid": [YEAR_2024_START, YEAR_2024_END],
                "eligible_rows": counts,
                "features": len(names),
                "model_path": model_path,
                "model_sha256": sha256_file(model_path),
                "prediction_path": pred_path,
                "prediction_sha256": sha256_file(pred_path),
            }

    gate_paths = _component_paths(artifact_root, GATE_COMPONENT_FILES)
    gate_components = {
        name: _read_prediction(
            path,
            expected_index=valid_index,
            required_groups=locked_groups,
        )
        for name, path in gate_paths.items()
    }
    baseline = pd.DataFrame(index=valid_index, columns=list(locked_groups), dtype=float)
    candidate = pd.DataFrame(index=valid_index, columns=list(locked_groups), dtype=float)
    for group in locked_groups:
        parts = {name: frame[group] for name, frame in gate_components.items()}
        baseline[group] = _assemble_group(recipe, group, parts)
        candidate_parts = dict(parts)
        candidate_parts["shared_q07"] = (
            gate_components["shared_q07"][group]
            + new_predictions[group][7]
            + new_predictions[group][2026]
        ) / 3.0
        candidate[group] = _assemble_group(recipe, group, candidate_parts)

    reference_path = artifact_root / "oof" / "gate2024_locked_v3_cf_fix.parquet"
    gate_seed42_model_path = artifact_root / "models/gate2024_shared_cf_fix.joblib"
    reference = _read_prediction(
        reference_path,
        expected_index=valid_index,
        required_groups=locked_groups,
    )
    reconstruction_difference = float(
        np.max(np.abs(baseline.to_numpy() - reference.to_numpy()))
    )
    if reconstruction_difference != 0.0:
        raise AssertionError(
            f"corrected gate reconstruction differs by {reconstruction_difference}"
        )
    baseline_path = out_dir / "oof" / "stage2_corrected_v3_baseline.parquet"
    candidate_path = out_dir / "oof" / "stage2_multiseed_candidate.parquet"
    _atomic_parquet(baseline, baseline_path)
    _atomic_parquet(candidate, candidate_path)
    comparisons = {
        group: _slice_comparison(
            labels=labels[group],
            baseline=baseline[group],
            candidate=candidate[group],
            group=group,
            slices={
                "full": (YEAR_2024_START, YEAR_2024_END),
                "H1": (YEAR_2024_START, H2_2024_START - pd.Timedelta(hours=1)),
                "H2": (H2_2024_START, YEAR_2024_END),
            },
        )
        for group in locked_groups
    }
    promoted_groups = [
        group
        for group, group_results in comparisons.items()
        if all(float(values["delta"]) > 0.0 for values in group_results.values())
    ]
    result = {
        "schema_version": 2,
        "experiment_id": "shared_q07_multiseed_strict_forward_stage2",
        "created_utc": utc_now(),
        "stage1_lock_sha256": sha256_file(out_dir / "stage1_lock.json"),
        "provenance_snapshot_sha256": lock["provenance_snapshot_sha256"],
        "provenance_sha256": lock["provenance_sha256"],
        "stage1_input_snapshot_sha256": lock["stage1_input_snapshot_sha256"],
        "stage1_locked_groups": list(locked_groups),
        "candidate_frozen_before_2024": True,
        "no_2024_reselection_or_retuning": True,
        "bounded_label_rows": int(len(labels)),
        "bounded_time": [labels.index.min(), labels.index.max()],
        "training": training,
        "baseline_reconstruction_max_abs_diff": reconstruction_difference,
        "comparisons": comparisons,
        "promotion_rule": "all 2024 full/H1/H2 deltas strictly > 0",
        "promoted_groups": promoted_groups,
        "sources": [
            describe_file(label_path),
            *[
                describe_file(cache_dir / f"{group}_weather_train.parquet")
                for group in TARGET_COLS
            ],
            *[describe_file(path) for path in gate_paths.values()],
            describe_file(reference_path),
            describe_file(gate_seed42_model_path),
        ],
        "outputs_before_result": [
            describe_file(baseline_path),
            describe_file(candidate_path),
        ],
        "leaderboard_score_claim": False,
    }
    _write_json(result_path, result)
    promotion = {
        "schema_version": 2,
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "stage1_lock_sha256": sha256_file(out_dir / "stage1_lock.json"),
        "stage2_results_sha256": sha256_file(result_path),
        "provenance_snapshot_sha256": lock["provenance_snapshot_sha256"],
        "provenance_sha256": lock["provenance_sha256"],
        "promoted_groups": promoted_groups,
        "candidate_frozen": True,
    }
    _write_json(promotion_path, promotion)
    print("stage2 promoted groups:", promoted_groups, flush=True)
    return promotion


def _load_promotion_lock(out_dir: Path) -> dict[str, Any]:
    path = out_dir / "stage2_promotion_lock.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    stage2_result = json.loads(
        (out_dir / "stage2_results.json").read_text(encoding="utf-8")
    )
    if payload["preregister_sha256"] != PREREGISTER_SHA256:
        raise AssertionError("promotion lock preregister hash differs")
    if payload["stage2_results_sha256"] != sha256_file(
        out_dir / "stage2_results.json"
    ):
        raise AssertionError("stage2 result changed after promotion lock")
    if list(payload.get("promoted_groups", [])) != list(
        stage2_result.get("promoted_groups", [])
    ):
        raise AssertionError("promotion lock groups differ from stage2 result")
    for key in ("provenance_snapshot_sha256", "provenance_sha256"):
        if payload.get(key) != stage2_result.get(key):
            raise AssertionError(f"promotion lock {key} differs from stage2 result")
    return payload


def _verify_submission(
    path: Path,
    sample: pd.DataFrame,
    expected_predictions: pd.DataFrame,
) -> dict[str, Any]:
    with path.open("rb") as stream:
        if stream.read(3) != b"\xef\xbb\xbf":
            raise AssertionError("submission lacks UTF-8 BOM")
    observed = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"forecast_id": "string", "forecast_kst_dtm": "string"},
    )
    if len(observed) != EXPECTED_TEST_ROWS or tuple(observed.columns) != tuple(
        sample.columns
    ):
        raise AssertionError("submission row count/schema differs from sample")
    for column in ("forecast_id", "forecast_kst_dtm"):
        if not np.array_equal(
            observed[column].astype(str).to_numpy(),
            sample[column].astype(str).to_numpy(),
        ):
            raise AssertionError(f"submission {column} differs from sample")
    values = observed.loc[:, list(TARGET_COLS)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise AssertionError("submission contains non-finite predictions")
    maximum_readback_difference = float(
        np.max(np.abs(values - expected_predictions.to_numpy(dtype=float)))
    )
    if maximum_readback_difference > 5.1e-7:
        raise AssertionError("submission six-decimal readback differs")
    ranges: dict[str, Any] = {}
    for group in TARGET_COLS:
        lower, upper = 0.0, CAPACITY_KWH[group] * 1.02
        if (observed[group] < lower - 1e-6).any() or (
            observed[group] > upper + 1e-6
        ).any():
            raise AssertionError(f"submission violates {group} capacity clip")
        ranges[group] = {
            "min": float(observed[group].min()),
            "max": float(observed[group].max()),
            "clip": [lower, upper],
        }
    return {
        "rows": int(len(observed)),
        "sample_schema_id_time_exact": True,
        "utf8_bom": True,
        "finite": True,
        "capacity_clip": True,
        "max_six_decimal_readback_abs_diff": maximum_readback_difference,
        "ranges": ranges,
        "sha256": sha256_file(path),
    }


def _finalize(
    *,
    raw_dir: Path,
    cache_dir: Path,
    artifact_root: Path,
    out_dir: Path,
    recipe: Mapping[str, Any],
    recipe_path: Path,
    preregister_path: Path,
) -> dict[str, Any]:
    promotion = _load_promotion_lock(out_dir)
    promoted_groups = tuple(promotion["promoted_groups"])
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    if not promoted_groups:
        stage1_result = json.loads(
            (out_dir / "stage1_results.json").read_text(encoding="utf-8")
        )
        postlock_audit_path = out_dir / "postlock_bit_exact_audit.json"
        if not postlock_audit_path.is_file():
            raise AssertionError("post-lock bit-exact audit is required before manifest")
        existing_outputs = sorted(
            path
            for path in out_dir.rglob("*")
            if path.is_file() and path != manifest_path
        )
        manifest = {
            "schema_version": 2,
            "artifact_type": "shared_q07_multiseed_strict_forward_v2",
            "created_utc": utc_now(),
            "command": [sys.executable, *sys.argv],
            "runtime": {
                "packages": package_versions(),
                "git": git_state(PROJECT_DIR),
            },
            "preregister_sha256": PREREGISTER_SHA256,
            "candidate_count": 1,
            "seeds": list(SEEDS),
            "provenance_snapshot_sha256": stage1_result[
                "provenance_snapshot_sha256"
            ],
            "provenance_sha256": stage1_result["provenance_sha256"],
            "stage1_input_snapshot_sha256": stage1_result[
                "stage1_input_snapshot_sha256"
            ],
            "stage1_snapshots_unchanged": stage1_result[
                "snapshots_unchanged_during_stage1"
            ],
            "stage1_lock": describe_file(out_dir / "stage1_lock.json"),
            "stage2_promotion_lock": describe_file(
                out_dir / "stage2_promotion_lock.json"
            ),
            "promoted_groups": [],
            "final_fit_performed": False,
            "submission_created": False,
            "inputs": [
                *stage1_result["provenance_before"].values(),
                *stage1_result["stage1_input_snapshot_before"].values(),
                describe_file(postlock_audit_path),
            ],
            "outputs": [describe_file(path) for path in existing_outputs],
            "leaderboard_score_claim": False,
        }
        _write_json(manifest_path, manifest)
        return manifest

    label_path = raw_dir / "train" / "train_labels.csv"
    labels = _read_labels(
        label_path,
        nrows=EXPECTED_ROWS_THROUGH2024,
        expected_end=YEAR_2024_END,
    )
    train_features = _read_features(cache_dir, labels, expected_end=YEAR_2024_END)
    sample_path = raw_dir / "sample_submission.csv"
    sample = pd.read_csv(
        sample_path,
        encoding="utf-8-sig",
        dtype={"forecast_id": "string", "forecast_kst_dtm": "string"},
    )
    if len(sample) != EXPECTED_TEST_ROWS:
        raise AssertionError("sample row count changed")
    sample_times = pd.DatetimeIndex(
        pd.to_datetime(sample["forecast_kst_dtm"], errors="raise"),
        name="forecast_kst_dtm",
    )
    test_features = _read_test_features(cache_dir, sample_times)
    full_features = {
        group: pd.concat(
            [train_features[group], test_features[group]], axis=0, copy=False
        )
        for group in TARGET_COLS
    }
    new_predictions: dict[int, pd.DataFrame] = {}
    final_training: dict[str, Any] = {}
    for seed in NEW_SEEDS:
        print(f"final: fitting pooled actual through 2024 seed={seed}", flush=True)
        model, prediction, counts, names = _fit_shared_q07(
            seed=seed,
            recipe=recipe,
            labels=labels,
            features=full_features,
            train_start=YEAR_2022_START,
            train_end=YEAR_2024_END,
            predict_indices={group: sample_times for group in TARGET_COLS},
        )
        model_path = out_dir / "models" / f"final_shared_q07_cf_seed{seed}.joblib"
        pred_path = out_dir / "predictions" / f"shared_q07_cf_seed{seed}_2025.parquet"
        _atomic_joblib(model, model_path)
        _atomic_parquet(prediction, pred_path)
        new_predictions[seed] = prediction
        final_training[f"seed{seed}"] = {
            "eligible_rows": counts,
            "features": len(names),
            "model_path": model_path,
            "model_sha256": sha256_file(model_path),
            "prediction_path": pred_path,
            "prediction_sha256": sha256_file(pred_path),
        }

    final_paths = _component_paths(artifact_root, FINAL_COMPONENT_FILES)
    final_components = {
        name: _read_prediction(
            path,
            expected_index=sample_times,
            required_groups=TARGET_COLS,
        )
        for name, path in final_paths.items()
    }
    mean_shared = (
        final_components["shared_q07"] + new_predictions[7] + new_predictions[2026]
    ) / 3.0
    mean_path = out_dir / "predictions" / "shared_q07_cf_seed_mean_2025.parquet"
    _atomic_parquet(mean_shared, mean_path)
    baseline = pd.DataFrame(index=sample_times, columns=TARGET_COLS, dtype=float)
    candidate = pd.DataFrame(index=sample_times, columns=TARGET_COLS, dtype=float)
    for group in TARGET_COLS:
        parts = {name: frame[group] for name, frame in final_components.items()}
        baseline[group] = _assemble_group(recipe, group, parts)
        candidate_parts = dict(parts)
        if group in promoted_groups:
            candidate_parts["shared_q07"] = mean_shared[group]
        candidate[group] = _assemble_group(recipe, group, candidate_parts)
    baseline_reference_path = (
        artifact_root / "final_cf_fix/predictions/corrected_v3_test.parquet"
    )
    baseline_reference = _read_prediction(
        baseline_reference_path,
        expected_index=sample_times,
        required_groups=TARGET_COLS,
    )
    reconstruction_difference = float(
        np.max(np.abs(baseline.to_numpy() - baseline_reference.to_numpy()))
    )
    if reconstruction_difference != 0.0:
        raise AssertionError(
            f"final corrected-v3 reconstruction differs by {reconstruction_difference}"
        )
    candidate_path = out_dir / "predictions" / "corrected_v3_multiseed_2025.parquet"
    _atomic_parquet(candidate, candidate_path)
    submission = sample.copy()
    for group in TARGET_COLS:
        submission[group] = candidate[group].to_numpy(dtype=float)
    submission_path = out_dir / "corrected_v3_shared_q07_multiseed.csv"
    _atomic_csv(submission, submission_path)
    verification = _verify_submission(submission_path, sample, candidate)

    output_paths = sorted(
        path
        for path in out_dir.rglob("*")
        if path.is_file() and path != manifest_path
    )
    source_paths = [
        preregister_path,
        recipe_path,
        label_path,
        sample_path,
        *[
            cache_dir / f"{group}_weather_train.parquet" for group in TARGET_COLS
        ],
        *[cache_dir / f"{group}_weather_test.parquet" for group in TARGET_COLS],
        *final_paths.values(),
        baseline_reference_path,
        artifact_root / "final_cf_fix/models/shared_q07_cf_seed42.joblib",
    ]
    stage1_result = json.loads(
        (out_dir / "stage1_results.json").read_text(encoding="utf-8")
    )
    manifest = {
        "schema_version": 2,
        "artifact_type": "shared_q07_multiseed_strict_forward_v2",
        "created_utc": utc_now(),
        "command": [sys.executable, *sys.argv],
        "runtime": {
            "packages": package_versions(),
            "git": git_state(PROJECT_DIR),
        },
        "preregister": describe_file(preregister_path),
        "preregister_sha256_pinned": PREREGISTER_SHA256,
        "provenance_snapshot_sha256": stage1_result[
            "provenance_snapshot_sha256"
        ],
        "provenance_sha256": stage1_result["provenance_sha256"],
        "stage1_input_snapshot_sha256": stage1_result[
            "stage1_input_snapshot_sha256"
        ],
        "candidate_count": 1,
        "seeds": list(SEEDS),
        "target_scale": "capacity_factor",
        "eligible_rule": "finite actual and CF >= 0.1",
        "model_parameters": {
            **recipe["models"]["shared_q07"],
            "n_jobs": int(recipe["n_jobs"]),
            "device": recipe["device"],
        },
        "replacement_formula": (
            "shared_q07 := arithmetic_mean(seed42_cf, seed7_cf, seed2026_cf) "
            "for promoted groups only"
        ),
        "corrected_v3_ensemble_unchanged": recipe["ensemble"],
        "stage1_lock": describe_file(out_dir / "stage1_lock.json"),
        "stage2_promotion_lock": describe_file(
            out_dir / "stage2_promotion_lock.json"
        ),
        "promoted_groups": list(promoted_groups),
        "failed_groups_unchanged_from_corrected_v3": [
            group for group in TARGET_COLS if group not in promoted_groups
        ],
        "final_training": final_training,
        "baseline_reconstruction_max_abs_diff": reconstruction_difference,
        "submission_verification": verification,
        "inputs": [
            *stage1_result["provenance_before"].values(),
            *stage1_result["stage1_input_snapshot_before"].values(),
            *[describe_file(path) for path in source_paths],
        ],
        "outputs": [describe_file(path) for path in output_paths],
        "leaderboard_score_claim": False,
        "independence": {
            "stage1": "strict-forward selection using only pre-2024 data",
            "stage2": "fixed candidate confirmation on 2024; no reselection",
            "2025": "unlabeled test prediction",
        },
    }
    _write_json(manifest_path, manifest)
    print("submission:", submission_path, flush=True)
    print("sha256:", verification["sha256"], flush=True)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    raw_dir = args.raw_dir.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    artifact_root = args.artifact_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    recipe_path = args.recipe.expanduser().resolve()
    preregister_path = args.preregister.expanduser().resolve()
    protected = {
        raw_dir,
        cache_dir,
        recipe_path.parent,
    }
    if any(out_dir == path or path in out_dir.parents for path in protected):
        raise ValueError("output directory overlaps a protected input directory")
    if out_dir == artifact_root:
        raise ValueError("output directory must not equal the artifact input root")
    preregister = _verify_preregister(preregister_path)
    recipe = _read_recipe(recipe_path)
    if args.stage in {"stage1", "all"}:
        _stage1(
            raw_dir=raw_dir,
            cache_dir=cache_dir,
            artifact_root=artifact_root,
            out_dir=out_dir,
            recipe=recipe,
            preregister=preregister,
            recipe_path=recipe_path,
        )
        _postlock_stage1_audit(
            raw_dir=raw_dir,
            cache_dir=cache_dir,
            artifact_root=artifact_root,
            out_dir=out_dir,
        )
    if args.stage in {"stage2", "all"}:
        if not (out_dir / "postlock_bit_exact_audit.json").is_file():
            raise AssertionError("stage2 requires completed post-lock bit-exact audit")
        _stage2(
            raw_dir=raw_dir,
            cache_dir=cache_dir,
            artifact_root=artifact_root,
            out_dir=out_dir,
            recipe=recipe,
        )
    if args.stage in {"final", "all"}:
        _finalize(
            raw_dir=raw_dir,
            cache_dir=cache_dir,
            artifact_root=artifact_root,
            out_dir=out_dir,
            recipe=recipe,
            recipe_path=recipe_path,
            preregister_path=preregister_path,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

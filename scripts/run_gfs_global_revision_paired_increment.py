"""Run the preregistered 2024-forward GFS fixed-lead revision experiment."""

from __future__ import annotations

import argparse
import ast
import atexit
import csv
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence
import uuid

import joblib
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metric import CAPACITY_KWH, TARGET_COLS, group_metrics, score_details  # noqa: E402


CONFIG_SHA = "486252908399408ef17b87bf452e3f145e18ea946dfcc56d54515298b1604b01"
CONFIG_PATH = ROOT / "configs/gfs_global_revision_paired_increment_preregister_v1.json"
SIDECAR_PATH = CONFIG_PATH.with_suffix(".sha256")
DOWNLOADER_PATH = ROOT / "scripts/download_openmeteo_gfs_revision_previous_runs.py"
EXTERNAL_ROOT = ROOT / "artifacts/external/openmeteo_gfs_global_previous_runs_v1"
EXTERNAL_PATH = EXTERNAL_ROOT / "gfs_global_group_centroids_2024.parquet"
SOURCE_MANIFEST_PATH = EXTERNAL_ROOT / "source_manifest.json"
CENSUS_PATH = ROOT / "artifacts/audits/external_previous_runs_distinct_nwp_feasibility_v1.json"
LABEL_PATH = Path(r"data/local/open/train/train_labels.csv")
PRIMARY_PATH = ROOT / "artifacts/oof/gate2024_locked_v3_cf_fix.parquet"
INTERACTION_PATH = ROOT / "artifacts/oof/gate2024_recent_v4_cf_fix_calibration_fit.parquet"
CACHE_PATHS = {
    group: ROOT / f"artifacts/cache/{group}_weather_train.parquet" for group in TARGET_COLS
}
OUTPUT_DIR = ROOT / "artifacts/postgate/gfs_global_revision_paired_increment_2024_forward_v1"
HEAVY_GUARD_PATH = ROOT / "artifacts/locks/heavy_cpu_fit.pid.json"

FIT_START = pd.Timestamp("2024-02-18 00:00:00")
FIT_END = pd.Timestamp("2024-06-30 23:00:00")
APPLY_START = pd.Timestamp("2024-07-01 00:00:00")
MODEL_APPLY_END = pd.Timestamp("2024-12-31 23:00:00")
TERMINAL = pd.Timestamp("2025-01-01 00:00:00")
FIT_INDEX = pd.date_range(FIT_START, FIT_END, freq="h", name="forecast_kst_dtm")
MODEL_APPLY_INDEX = pd.date_range(
    APPLY_START, MODEL_APPLY_END, freq="h", name="forecast_kst_dtm"
)
SCORE_INDEX = pd.date_range(APPLY_START, TERMINAL, freq="h", name="forecast_kst_dtm")
FIT_PREFIX_DATA_ROWS = 21_887
SCORE_DATA_ROWS = 4_417
EXPECTED_LABEL_BYTES = 1_138_967
EXPECTED_LABEL_SHA = "47bb64252195cf4734e67394d6e50485f27a608def3b5a8791fcc7674bbceb03"

HEIGHTS = (10, 80, 100)
CANDIDATES = ("A_day2_w025", "B_day1_hours01_13_else_day2_w025")
BASELINES = {
    "primary_corrected_v3": PRIMARY_PATH,
    "interaction_recent_v4": INTERACTION_PATH,
}
EXTENDED_COLUMNS = tuple(
    name
    for height in HEIGHTS
    for name in (
        f"gfsrev__ws{height}_ms",
        f"gfsrev__u{height}_ms",
        f"gfsrev__v{height}_ms",
        f"gfsrev__ws{height}_minus_cross_hub_ws_mean",
    )
)
TIME_SLICES = {
    "H2": (pd.Timestamp("2024-07-01 00:00"), TERMINAL),
    "Q3": (pd.Timestamp("2024-07-01 00:00"), pd.Timestamp("2024-09-30 23:00")),
    "Q4": (pd.Timestamp("2024-10-01 00:00"), TERMINAL),
    "Jul": (pd.Timestamp("2024-07-01 00:00"), pd.Timestamp("2024-07-31 23:00")),
    "Aug": (pd.Timestamp("2024-08-01 00:00"), pd.Timestamp("2024-08-31 23:00")),
    "Sep": (pd.Timestamp("2024-09-01 00:00"), pd.Timestamp("2024-09-30 23:00")),
    "Oct": (pd.Timestamp("2024-10-01 00:00"), pd.Timestamp("2024-10-31 23:00")),
    "Nov": (pd.Timestamp("2024-11-01 00:00"), pd.Timestamp("2024-11-30 23:00")),
    "Dec": (pd.Timestamp("2024-12-01 00:00"), TERMINAL),
}
REGIMES = ("low", "mid", "high")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args(argv)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        frame.to_parquet(temporary, engine="pyarrow", compression="zstd", index=True)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_joblib(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        joblib.dump(model, temporary, compress=3)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def copy_exclusive(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst)


def verify_file(spec: Mapping[str, Any], *, base: Path = ROOT) -> Path:
    path = Path(str(spec["path"]))
    if not path.is_absolute():
        path = base / path
    size = int(spec.get("bytes", spec.get("size_bytes")))
    if not path.is_file() or path.stat().st_size != size or sha256(path) != str(spec["sha256"]):
        raise AssertionError(f"registered file identity differs: {path}")
    return path.resolve()


def verify_config(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if path != CONFIG_PATH.resolve() or sha256(path) != CONFIG_SHA:
        raise AssertionError("preregistration identity differs")
    if SIDECAR_PATH.read_text(encoding="utf-8") != f"{CONFIG_SHA}  {CONFIG_PATH.name}\n":
        raise AssertionError("preregistration sidecar differs")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["status"] != "frozen_before_any_fit_label_value_read_candidate_fit_prediction_H2_label_read_or_score":
        raise AssertionError("preregistration status differs")
    if tuple(config["forward_design"]["candidate_ids_fixed_order"]) != CANDIDATES:
        raise AssertionError("candidate order differs")
    if tuple(config["paired_model"]["extended_features_after_control"]) != EXTENDED_COLUMNS:
        raise AssertionError("extended feature contract differs")
    if float(config["paired_model"]["transfer_weight"]) != 0.25:
        raise AssertionError("transfer weight differs")
    if tuple(config["registered_time_slices"]) != tuple(TIME_SLICES):
        raise AssertionError("registered time-slice order differs")
    return config


def _module_files(module: str) -> tuple[Path, ...]:
    if not module:
        return ()
    base = ROOT.joinpath(*module.split("."))
    result: list[Path] = []
    parts = module.split(".")
    for length in range(1, len(parts)):
        initializer = ROOT.joinpath(*parts[:length]) / "__init__.py"
        if initializer.is_file():
            result.append(initializer.resolve())
    if base.with_suffix(".py").is_file():
        result.append(base.with_suffix(".py").resolve())
    if base.is_dir() and (base / "__init__.py").is_file():
        result.append((base / "__init__.py").resolve())
    return tuple(dict.fromkeys(result))


def resolve_ast_closure(entry: Path) -> tuple[Path, ...]:
    queue = [entry.resolve()]
    visited: set[Path] = set()
    while queue:
        path = queue.pop()
        if path in visited:
            continue
        if not path.is_file() or ROOT not in path.parents:
            raise AssertionError(f"invalid local closure path: {path}")
        visited.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    queue.extend(_module_files(alias.name))
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    relative = path.relative_to(ROOT).with_suffix("").parts[:-1]
                    keep = max(len(relative) - node.level + 1, 0)
                    base = ".".join((*relative[:keep], *((node.module or "").split("."))))
                else:
                    base = node.module or ""
                resolved = list(_module_files(base.strip(".")))
                for alias in node.names:
                    if alias.name != "*":
                        resolved.extend(_module_files(f"{base}.{alias.name}".strip(".")))
                if not resolved:
                    top = base.strip(".").split(".")[0]
                    if top and (ROOT / top).exists():
                        raise AssertionError(f"unresolved local import: {base}")
                queue.extend(resolved)
    return tuple(sorted(visited))


def source_closure(config: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    closure = resolve_ast_closure(Path(__file__))
    explicit = [
        args.config.resolve(),
        SIDECAR_PATH.resolve(),
        DOWNLOADER_PATH.resolve(),
        (ROOT / "tests/test_gfs_global_revision_paired_increment.py").resolve(),
    ]
    return {
        "resolver": "recursive Python AST local imports plus package initializers",
        "resolved_relative_paths": [path.relative_to(ROOT).as_posix() for path in closure],
        "resolved_files": [file_record(path) for path in closure],
        "explicit_sources": [file_record(path) for path in explicit],
        "unresolved_local_imports": [],
        "fixed_nonlabel_inputs": [
            file_record(EXTERNAL_PATH),
            file_record(SOURCE_MANIFEST_PATH),
            *[file_record(EXTERNAL_ROOT / f"raw_{group}_2024.json") for group in TARGET_COLS],
            file_record(CENSUS_PATH),
            file_record(PRIMARY_PATH),
            file_record(INTERACTION_PATH),
            *[file_record(CACHE_PATHS[group]) for group in TARGET_COLS],
        ],
        "label_metadata_only": {
            "path": str(LABEL_PATH),
            "expected_bytes": int(config["label_access_contract"]["expected_file_bytes"]),
            "observed_bytes": LABEL_PATH.stat().st_size,
            "content_bytes_read_for_this_lock": 0,
            "content_hash_deferred_until_registered_phase_reads_complete": True,
        },
    }


def assert_closure_unchanged(first: Mapping[str, Any], config: Mapping[str, Any], args: argparse.Namespace) -> None:
    second = source_closure(config, args)
    if first != second:
        raise AssertionError("source/input closure changed during run")


def validate_inputs(config: Mapping[str, Any]) -> None:
    source = config["official_source_contract"]
    verify_file(source["availability_manifest"])
    verify_file(source["normalized_data"])
    for record in source["raw_responses"]:
        verify_file(record)
    verify_file(config["nonduplication_census"]["report"])
    for record in config["fixed_baselines"]:
        verify_file(record)
    for record in config["bound_inputs"]["weather_cache"].values():
        verify_file(record)
    verify_file(config["bound_inputs"]["metric"])
    verify_file(config["bound_inputs"]["downloader"])
    if LABEL_PATH.stat().st_size != EXPECTED_LABEL_BYTES:
        raise AssertionError("label file metadata differs before bounded access")


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return False
    ctypes.windll.kernel32.CloseHandle(handle)
    return True


def acquire_heavy_guard(path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    for _ in range(2):
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            current = json.loads(path.read_text(encoding="utf-8"))
            if pid_alive(int(current.get("pid", -1))):
                raise RuntimeError(f"heavy guard held by live PID {current.get('pid')}: {current}")
            stale = path.with_name(
                f"heavy_cpu_fit.stale-pid{current.get('pid','unknown')}-{uuid.uuid4().hex}.json"
            )
            os.replace(path, stale)
            continue
        owner = {
            "schema_version": 1,
            "pid": os.getpid(),
            "token": token,
            "experiment_id": "gfs_global_revision_paired_increment_2024_forward_v1",
            "stage": "GFS_revision_2024_forward",
            "created_utc": utc_now(),
        }
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(owner, stream, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
        return owner
    raise RuntimeError("could not acquire shared heavy guard")


def release_heavy_guard(path: Path, owner: Mapping[str, Any]) -> None:
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        return
    if int(current.get("pid", -1)) == int(owner["pid"]) and current.get("token") == owner.get("token"):
        path.unlink()


def _decode_header(line: bytes) -> list[str]:
    return next(csv.reader([line.decode("utf-8-sig").rstrip("\r\n")]))


def read_fit_labels_only() -> tuple[pd.DataFrame, int, dict[str, Any], bytes, Any]:
    if LABEL_PATH.stat().st_size != EXPECTED_LABEL_BYTES:
        raise AssertionError("label size differs")
    digest = hashlib.sha256()
    fit_times: list[pd.Timestamp] = []
    fit_values: list[list[float]] = []
    first_time: pd.Timestamp | None = None
    last_time: pd.Timestamp | None = None
    with LABEL_PATH.open("rb", buffering=0) as stream:
        header = stream.readline()
        if _decode_header(header) != ["kst_dtm", *TARGET_COLS]:
            raise AssertionError("label header differs")
        digest.update(header)
        for row_number in range(FIT_PREFIX_DATA_ROWS):
            line = stream.readline()
            if not line:
                raise AssertionError(f"label fit prefix ended at row {row_number}")
            digest.update(line)
            timestamp = pd.Timestamp(line.split(b",", 1)[0].decode("ascii"))
            first_time = timestamp if first_time is None else first_time
            last_time = timestamp
            if timestamp >= FIT_START:
                fields = next(csv.reader([line.decode("utf-8").rstrip("\r\n")]))
                if len(fields) != 4:
                    raise AssertionError("fit label row width differs")
                fit_times.append(timestamp)
                fit_values.append([float(value) if value else np.nan for value in fields[1:]])
        score_offset = stream.tell()
    frame = pd.DataFrame(
        fit_values,
        index=pd.DatetimeIndex(fit_times, name="forecast_kst_dtm"),
        columns=TARGET_COLS,
        dtype=np.float64,
    )
    if first_time != pd.Timestamp("2022-01-01 01:00") or last_time != FIT_END:
        raise AssertionError("fit prefix timestamp boundary differs")
    if not frame.index.equals(FIT_INDEX):
        raise AssertionError("fit label materialization window differs")
    record = {
        "physical_data_rows_read": FIT_PREFIX_DATA_ROWS,
        "physical_prefix_bytes": score_offset,
        "physical_prefix_sha256_including_header": digest.copy().hexdigest(),
        "first_timestamp": str(first_time),
        "last_timestamp": str(last_time),
        "materialized_value_window": [str(FIT_START), str(FIT_END)],
        "materialized_value_rows": len(frame),
        "materialized_value_cells": int(frame.size),
        "prehistory_value_cells_materialized": 0,
        "H2_bytes_or_value_cells_read": 0,
        "score_segment_start_byte_offset": score_offset,
    }
    return frame, score_offset, record, header, digest


def read_score_labels(
    score_offset: int, header: bytes, full_digest: Any
) -> tuple[pd.DataFrame, dict[str, Any]]:
    segment_digest = hashlib.sha256()
    times: list[pd.Timestamp] = []
    values: list[list[float]] = []
    with LABEL_PATH.open("rb", buffering=0) as stream:
        stream.seek(score_offset)
        if stream.tell() != score_offset:
            raise AssertionError("score label seek failed")
        for row_number in range(SCORE_DATA_ROWS):
            line = stream.readline()
            if not line:
                raise AssertionError(f"score label segment ended at row {row_number}")
            segment_digest.update(line)
            full_digest.update(line)
            fields = next(csv.reader([line.decode("utf-8").rstrip("\r\n")]))
            if len(fields) != 4:
                raise AssertionError("score label row width differs")
            times.append(pd.Timestamp(fields[0]))
            values.append([float(value) if value else np.nan for value in fields[1:]])
        end_offset = stream.tell()
        if stream.read(1) != b"":
            raise AssertionError("unregistered bytes follow score segment")
    frame = pd.DataFrame(
        values,
        index=pd.DatetimeIndex(times, name="forecast_kst_dtm"),
        columns=TARGET_COLS,
        dtype=np.float64,
    )
    if not frame.index.equals(SCORE_INDEX) or end_offset != EXPECTED_LABEL_BYTES:
        raise AssertionError("score label segment differs")
    if full_digest.hexdigest() != EXPECTED_LABEL_SHA:
        raise AssertionError("full label identity differs after registered phase reads")
    return frame, {
        "header_sha256": hashlib.sha256(header).hexdigest(),
        "seek_start_byte_offset": score_offset,
        "physical_data_rows_read": SCORE_DATA_ROWS,
        "segment_bytes": end_offset - score_offset,
        "segment_sha256": segment_digest.hexdigest(),
        "full_source_sha256_after_registered_reads": full_digest.hexdigest(),
        "first_timestamp": str(frame.index[0]),
        "last_timestamp": str(frame.index[-1]),
        "materialized_value_cells": int(frame.size),
        "bytes_or_rows_after_segment_read": 0,
    }


def select_external_features(frame: pd.DataFrame, candidate: str) -> pd.DataFrame:
    if candidate not in CANDIDATES:
        raise ValueError(f"unknown candidate: {candidate}")
    use_day1 = (
        np.zeros(len(frame), dtype=bool)
        if candidate == CANDIDATES[0]
        else np.isin(frame.index.hour.to_numpy(), np.arange(1, 14))
    )
    result = pd.DataFrame(index=frame.index)
    for height in HEIGHTS:
        speed = np.where(
            use_day1,
            frame[f"wind_speed_{height}m_previous_day1"].to_numpy(np.float64),
            frame[f"wind_speed_{height}m_previous_day2"].to_numpy(np.float64),
        )
        direction = np.where(
            use_day1,
            frame[f"wind_direction_{height}m_previous_day1"].to_numpy(np.float64),
            frame[f"wind_direction_{height}m_previous_day2"].to_numpy(np.float64),
        )
        radians = np.deg2rad(direction)
        result[f"gfsrev__ws{height}_ms"] = speed
        result[f"gfsrev__u{height}_ms"] = -speed * np.sin(radians)
        result[f"gfsrev__v{height}_ms"] = -speed * np.cos(radians)
    return result.astype(np.float32)


def add_disagreements(external: pd.DataFrame, control: pd.DataFrame) -> pd.DataFrame:
    if not external.index.equals(control.index):
        raise AssertionError("external/control indexes differ")
    reference = control["cross__hub_ws_mean"].astype(np.float64)
    result = pd.DataFrame(index=external.index)
    for height in HEIGHTS:
        for component in ("ws", "u", "v"):
            result[f"gfsrev__{component}{height}_ms"] = external[
                f"gfsrev__{component}{height}_ms"
            ]
        result[f"gfsrev__ws{height}_minus_cross_hub_ws_mean"] = (
            external[f"gfsrev__ws{height}_ms"].astype(np.float64) - reference
        ).astype(np.float32)
    if tuple(result.columns) != EXTENDED_COLUMNS or not np.isfinite(result.to_numpy()).all():
        raise AssertionError("registered twelve-feature extension differs")
    return result.astype(np.float32)


def apply_increment(
    baseline: pd.Series, increment: pd.Series, *, capacity: float, weight: float = 0.25
) -> pd.Series:
    if not baseline.index.equals(increment.index):
        raise AssertionError("baseline/increment indexes differ")
    values = np.clip(
        baseline.to_numpy(np.float64) + weight * capacity * increment.to_numpy(np.float64),
        0.0,
        1.02 * capacity,
    )
    result = pd.Series(values, index=baseline.index, name=baseline.name)
    result.loc[TERMINAL] = baseline.loc[TERMINAL]
    return result


def regime_name(values: pd.Series) -> pd.Series:
    result = pd.Series(index=values.index, dtype="string")
    result.loc[values < 4.0] = "low"
    result.loc[(values >= 4.0) & (values < 8.0)] = "mid"
    result.loc[values >= 8.0] = "high"
    if result.isna().any():
        raise AssertionError("regime driver is nonfinite or uncovered")
    return result


def index_sha(index: pd.DatetimeIndex) -> str:
    return hashlib.sha256(np.ascontiguousarray(index.asi8).tobytes()).hexdigest()


def _group_metric(actual: pd.Series, prediction: pd.Series, group: str) -> dict[str, Any]:
    record = group_metrics(actual, prediction, CAPACITY_KWH[group], group_name=group).as_dict()
    record["total_score"] = 0.5 * (record["one_minus_nmae"] + record["ficr"])
    return record


def _record_group(
    labels: pd.DataFrame,
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    group: str,
    index: pd.DatetimeIndex,
) -> dict[str, Any]:
    base = _group_metric(labels.loc[index, group], baseline.loc[index, group], group)
    cand = _group_metric(labels.loc[index, group], candidate.loc[index, group], group)
    return {
        "rows": len(index),
        "index_sha256": index_sha(index),
        "baseline": base,
        "candidate": cand,
        "delta_total_score": cand["total_score"] - base["total_score"],
        "delta_one_minus_nmae": cand["one_minus_nmae"] - base["one_minus_nmae"],
        "delta_ficr": cand["ficr"] - base["ficr"],
    }


def _record_mixed(
    labels: pd.DataFrame,
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    index: pd.DatetimeIndex,
) -> dict[str, Any]:
    base = score_details(labels.loc[index], baseline.loc[index]).as_dict()
    cand = score_details(labels.loc[index], candidate.loc[index]).as_dict()
    return {
        "rows": len(index),
        "index_sha256": index_sha(index),
        "baseline": base,
        "candidate": cand,
        "delta_total_score": cand["total_score"] - base["total_score"],
        "delta_one_minus_nmae": cand["one_minus_nmae"] - base["one_minus_nmae"],
        "delta_ficr": cand["ficr"] - base["ficr"],
    }


def comparison_payload(
    labels: pd.DataFrame,
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    drivers: pd.DataFrame,
) -> dict[str, Any]:
    group_records: dict[str, Any] = {}
    mixed_records: dict[str, Any] = {"time": {}, "regime": {}}
    for group in TARGET_COLS:
        group_records[group] = {"time": {}, "regime": {}}
        group_regime = regime_name(drivers[group])
        for name, (start, end) in TIME_SLICES.items():
            index = SCORE_INDEX[(SCORE_INDEX >= start) & (SCORE_INDEX <= end)]
            group_records[group]["time"][name] = _record_group(
                labels, baseline, candidate, group, index
            )
        for name in REGIMES:
            index = pd.DatetimeIndex(group_regime.index[group_regime == name], name="forecast_kst_dtm")
            if len(index) == 0:
                raise AssertionError(f"empty group regime: {group}/{name}")
            group_records[group]["regime"][name] = _record_group(
                labels, baseline, candidate, group, index
            )
    mixed_driver = drivers.mean(axis=1)
    mixed_regime = regime_name(mixed_driver)
    for name, (start, end) in TIME_SLICES.items():
        index = SCORE_INDEX[(SCORE_INDEX >= start) & (SCORE_INDEX <= end)]
        mixed_records["time"][name] = _record_mixed(labels, baseline, candidate, index)
    for name in REGIMES:
        index = pd.DatetimeIndex(mixed_regime.index[mixed_regime == name], name="forecast_kst_dtm")
        if len(index) == 0:
            raise AssertionError(f"empty mixed regime: {name}")
        mixed_records["regime"][name] = _record_mixed(labels, baseline, candidate, index)
    return {"groups": group_records, "mixed": mixed_records}


def gate_summary(comparisons: Mapping[str, Any]) -> dict[str, Any]:
    baseline_results: dict[str, Any] = {}
    all_group: list[float] = []
    dual = True
    for baseline_name, payload in comparisons.items():
        group_deltas = [
            float(record["delta_total_score"])
            for group in TARGET_COLS
            for family in ("time", "regime")
            for record in payload["groups"][group][family].values()
        ]
        mixed_deltas = [
            float(record["delta_total_score"])
            for family in ("time", "regime")
            for record in payload["mixed"][family].values()
        ]
        h2 = payload["mixed"]["time"]["H2"]
        passed = (
            len(group_deltas) == 36
            and len(mixed_deltas) == 12
            and all(delta > 0.0 for delta in group_deltas)
            and all(delta > 0.0 for delta in mixed_deltas)
            and h2["delta_one_minus_nmae"] >= 0.0
            and h2["delta_ficr"] >= 0.0
        )
        dual = dual and passed
        all_group.extend(group_deltas)
        baseline_results[baseline_name] = {
            "passed": passed,
            "minimum_group_delta": min(group_deltas),
            "minimum_mixed_delta": min(mixed_deltas),
            "H2_mixed_delta_one_minus_nmae": h2["delta_one_minus_nmae"],
            "H2_mixed_delta_ficr": h2["delta_ficr"],
        }
    return {
        "dual_gate_pass": dual,
        "baseline_results": baseline_results,
        "minimum_group_record_delta": min(all_group),
        "mean_group_record_delta": float(np.mean(all_group)),
        "registered_group_records_across_both_baselines": len(all_group),
    }


def read_external() -> pd.DataFrame:
    frame = pd.read_parquet(EXTERNAL_PATH)
    frame["time"] = pd.to_datetime(frame["time"], errors="raise")
    expected_columns = (
        "group",
        "time",
        *tuple(
            f"{kind}_{height}m_previous_day{day}"
            for height in HEIGHTS
            for day in (1, 2)
            for kind in ("wind_speed", "wind_direction")
        ),
    )
    if tuple(frame.columns) != expected_columns or len(frame) != 26_352:
        raise AssertionError("external source schema/rows differ")
    if frame["time"].max() != MODEL_APPLY_END:
        raise AssertionError("external source crossed or missed 2024 boundary")
    return frame


def read_baseline(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    frame = frame.loc[SCORE_INDEX, list(TARGET_COLS)].astype(np.float64)
    if not frame.index.equals(SCORE_INDEX) or not np.isfinite(frame.to_numpy()).all():
        raise AssertionError(f"baseline differs: {path}")
    return frame


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.config = args.config.resolve()
    args.out_dir = args.out_dir.resolve()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    config = verify_config(args.config)
    validate_inputs(config)
    if LABEL_PATH.stat().st_size != EXPECTED_LABEL_BYTES:
        raise AssertionError("label metadata differs")
    args.out_dir.mkdir(parents=True)
    copy_exclusive(args.config, args.out_dir / "preregister.json")
    copy_exclusive(SIDECAR_PATH, args.out_dir / "preregister.sha256")
    closure = source_closure(config, args)
    source_lock_path = args.out_dir / "source_lock_before_any_label_read.json"
    atomic_json(
        {
            "schema_version": 1,
            "status": "locked_before_any_label_read_candidate_fit_prediction_or_score",
            "created_utc": utc_now(),
            "preregister_sha256": CONFIG_SHA,
            "source_closure": closure,
            "candidate_fits_predictions_or_scores": 0,
            "label_content_bytes_read": 0,
            "year_2025_external_requests_response_bytes_or_values": 0,
        },
        source_lock_path,
    )
    owner = acquire_heavy_guard(HEAVY_GUARD_PATH)
    atexit.register(release_heavy_guard, HEAVY_GUARD_PATH, owner)
    print(f"source lock={sha256(source_lock_path)} guard PID={owner['pid']}", flush=True)
    try:
        fit_labels, score_offset, fit_access, header, full_digest = read_fit_labels_only()
        fit_lock_path = args.out_dir / "fit_label_lock_before_candidate_fit.json"
        atomic_json(
            {
                "schema_version": 1,
                "status": "locked_after_H1_only_label_read_before_candidate_fit_prediction_or_score",
                "created_utc": utc_now(),
                "preregister_sha256": CONFIG_SHA,
                "source_lock": file_record(source_lock_path),
                "fit_label_access": fit_access,
                "fit_index_sha256": index_sha(fit_labels.index),
                "fit_values_sha256": hashlib.sha256(
                    np.ascontiguousarray(fit_labels.to_numpy(np.float64)).tobytes()
                ).hexdigest(),
                "H2_label_bytes_or_value_cells_read": 0,
                "candidate_fits_predictions_or_scores": 0,
            },
            fit_lock_path,
        )
        print(f"fit-label lock={sha256(fit_lock_path)} H2 cells=0", flush=True)

        external_all = read_external()
        increments = pd.DataFrame(index=SCORE_INDEX)
        drivers = pd.DataFrame(index=SCORE_INDEX, columns=TARGET_COLS, dtype=np.float64)
        model_records: list[dict[str, Any]] = []
        fit_records: dict[str, Any] = {}
        parameters = dict(config["paired_model"]["parameters"])
        for group in TARGET_COLS:
            control_all = pd.read_parquet(CACHE_PATHS[group])
            control_all.index = pd.DatetimeIndex(control_all.index, name="forecast_kst_dtm")
            if control_all.shape[1] != 612 or not control_all.columns.is_unique:
                raise AssertionError(f"{group}: control schema differs")
            control_fit = control_all.loc[FIT_INDEX].astype(np.float32)
            control_apply = control_all.loc[MODEL_APPLY_INDEX].astype(np.float32)
            drivers[group] = control_all.loc[SCORE_INDEX, "cross__hub_ws_mean"].to_numpy(np.float64)
            if not np.isfinite(control_fit.to_numpy()).all() or not np.isfinite(control_apply.to_numpy()).all():
                raise AssertionError(f"{group}: control features nonfinite")
            target = fit_labels[group]
            eligible = np.isfinite(target.to_numpy(np.float64)) & (
                target.to_numpy(np.float64) >= 0.10 * CAPACITY_KWH[group]
            )
            fit_rows = FIT_INDEX[eligible]
            if len(fit_rows) == 0:
                raise AssertionError(f"{group}: no eligible fit rows")
            y = (target.loc[fit_rows] / CAPACITY_KWH[group]).to_numpy(np.float64)
            control_model = LGBMRegressor(**parameters)
            control_model.fit(control_fit.loc[fit_rows], y)
            control_cf = np.clip(control_model.predict(control_apply), 0.0, 1.02)
            control_path = args.out_dir / f"models/{group}__control.joblib"
            atomic_joblib(control_model, control_path)
            model_records.append(file_record(control_path))
            external = (
                external_all.loc[external_all["group"] == group]
                .drop(columns="group")
                .set_index("time")
            )
            external.index.name = "forecast_kst_dtm"
            fit_records[group] = {
                "eligible_rows": len(fit_rows),
                "eligible_index_sha256": index_sha(fit_rows),
                "target_cf_sha256": hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest(),
                "control_feature_count": 612,
                "extended_feature_count": 624,
            }
            for candidate in CANDIDATES:
                selected = select_external_features(external, candidate)
                extension_fit = add_disagreements(selected.loc[FIT_INDEX], control_fit)
                extension_apply = add_disagreements(
                    selected.loc[MODEL_APPLY_INDEX], control_apply
                )
                extended_fit = pd.concat([control_fit, extension_fit], axis=1)
                extended_apply = pd.concat([control_apply, extension_apply], axis=1)
                if extended_fit.shape[1] != 624 or tuple(extended_fit.columns[-12:]) != EXTENDED_COLUMNS:
                    raise AssertionError("extended feature schema differs")
                model = LGBMRegressor(**parameters)
                model.fit(extended_fit.loc[fit_rows], y)
                extended_cf = np.clip(model.predict(extended_apply), 0.0, 1.02)
                registered = pd.Series(0.0, index=SCORE_INDEX, dtype=np.float64)
                registered.loc[MODEL_APPLY_INDEX] = extended_cf - control_cf
                if registered.loc[TERMINAL] != 0.0:
                    raise AssertionError("terminal increment differs")
                increments[f"{candidate}__{group}"] = registered
                model_path = args.out_dir / f"models/{group}__{candidate}.joblib"
                atomic_joblib(model, model_path)
                model_records.append(file_record(model_path))
            print(f"fit {group}: eligible={len(fit_rows)}", flush=True)

        driver_path = args.out_dir / "predictions/registered_regime_drivers.parquet"
        increment_path = args.out_dir / "predictions/candidate_increments_cf.parquet"
        atomic_parquet(drivers, driver_path)
        atomic_parquet(increments, increment_path)
        baseline_frames = {name: read_baseline(path) for name, path in BASELINES.items()}
        baseline_records: dict[str, Any] = {}
        prediction_frames: dict[str, dict[str, pd.DataFrame]] = {
            candidate: {} for candidate in CANDIDATES
        }
        prediction_records: dict[str, Any] = {candidate: {} for candidate in CANDIDATES}
        for baseline_name, baseline in baseline_frames.items():
            baseline_path = args.out_dir / f"predictions/{baseline_name}__baseline.parquet"
            atomic_parquet(baseline, baseline_path)
            baseline_records[baseline_name] = file_record(baseline_path)
            for candidate in CANDIDATES:
                prediction = baseline.copy()
                for group in TARGET_COLS:
                    prediction[group] = apply_increment(
                        baseline[group],
                        increments[f"{candidate}__{group}"],
                        capacity=CAPACITY_KWH[group],
                    )
                    left = np.ascontiguousarray(
                        prediction.loc[[TERMINAL], group].to_numpy(np.float64)
                    ).view(np.uint64)[0]
                    right = np.ascontiguousarray(
                        baseline.loc[[TERMINAL], group].to_numpy(np.float64)
                    ).view(np.uint64)[0]
                    if left != right:
                        raise AssertionError("terminal baseline bits differ")
                path = args.out_dir / f"predictions/{candidate}__{baseline_name}.parquet"
                atomic_parquet(prediction, path)
                prediction_frames[candidate][baseline_name] = prediction
                prediction_records[candidate][baseline_name] = file_record(path)

        candidate_lock_path = args.out_dir / "candidate_lock_before_H2_labels.json"
        atomic_json(
            {
                "schema_version": 1,
                "status": "all_models_increments_baselines_candidates_and_regimes_locked_before_H2_label_read_or_score",
                "created_utc": utc_now(),
                "preregister_sha256": CONFIG_SHA,
                "fit_label_lock": file_record(fit_lock_path),
                "models": model_records,
                "increments": file_record(increment_path),
                "regime_drivers": file_record(driver_path),
                "baselines": baseline_records,
                "candidate_predictions": prediction_records,
                "fit_records": fit_records,
                "terminal_increment_cf_all_candidates_groups": 0.0,
                "terminal_candidate_baseline_float64_bit_identity": True,
                "H2_label_bytes_or_value_cells_read": 0,
                "metric_calls": 0,
                "retune_grid_group_rescue_or_fallback": False,
            },
            candidate_lock_path,
        )
        print(f"candidate-before-H2-label lock={sha256(candidate_lock_path)}", flush=True)

        score_labels, score_access = read_score_labels(score_offset, header, full_digest)
        score_lock_path = args.out_dir / "score_label_access_after_candidate_lock.json"
        atomic_json(
            {
                "schema_version": 1,
                "status": "bounded_H2_plus_terminal_labels_read_after_candidate_lock_before_metric",
                "created_utc": utc_now(),
                "preregister_sha256": CONFIG_SHA,
                "candidate_lock": file_record(candidate_lock_path),
                "score_label_access": score_access,
                "metric_calls_before_this_lock": 0,
                "year_2025_external_requests_response_bytes_or_values": 0,
            },
            score_lock_path,
        )
        print(f"score-label lock={sha256(score_lock_path)}", flush=True)

        comparisons: dict[str, Any] = {candidate: {} for candidate in CANDIDATES}
        for candidate in CANDIDATES:
            for baseline_name, baseline in baseline_frames.items():
                comparisons[candidate][baseline_name] = comparison_payload(
                    score_labels,
                    baseline,
                    prediction_frames[candidate][baseline_name],
                    drivers,
                )
        gates = {candidate: gate_summary(comparisons[candidate]) for candidate in CANDIDATES}
        passing = [candidate for candidate in CANDIDATES if gates[candidate]["dual_gate_pass"]]
        winner = max(
            passing,
            key=lambda candidate: (
                gates[candidate]["minimum_group_record_delta"],
                gates[candidate]["mean_group_record_delta"],
                -CANDIDATES.index(candidate),
            ),
            default=None,
        )
        results_path = args.out_dir / "stage_results.json"
        results = {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "created_utc": utc_now(),
            "preregister_sha256": CONFIG_SHA,
            "fit_records": fit_records,
            "comparisons": comparisons,
            "gates": gates,
            "winner": winner,
            "status": "promoted_for_separate_prescore_only" if winner else "rejected_on_2024_forward_dual_gate",
            "registered_metric_records_per_candidate_per_baseline": 48,
            "phase_locks": {
                "source": file_record(source_lock_path),
                "fit_labels": file_record(fit_lock_path),
                "candidate_before_H2_labels": file_record(candidate_lock_path),
                "score_labels_before_metric": file_record(score_lock_path),
            },
            "physical_access": {
                "maximum_external_calendar_year_requested_or_parsed": 2024,
                "terminal_target_rows_scored_with_identity": 1,
                "terminal_increment_cf": 0.0,
                "year_2025_external_requests_response_bytes_or_values": 0,
                "submission_CSV_created": False,
                "retune_grid_group_rescue_or_fallback": False,
            },
        }
        atomic_json(results, results_path)
        decision_path = args.out_dir / ("promotion_lock.json" if winner else "rejection.json")
        atomic_json(
            {
                "schema_version": 1,
                "status": results["status"],
                "winner": winner,
                "preregister_sha256": CONFIG_SHA,
                "stage_results": file_record(results_path),
                "year_2025_access_authorized_by_this_process": False,
                "retune_retry_rescue_authorized": False,
            },
            decision_path,
        )
        assert_closure_unchanged(closure, config, args)
        if list(args.out_dir.rglob("*.csv")):
            raise AssertionError("this experiment must not create a CSV")
        files = sorted(
            (
                path
                for path in args.out_dir.rglob("*")
                if path.is_file() and path.name not in ("manifest.json", "manifest.sha256")
            ),
            key=lambda path: path.relative_to(args.out_dir).as_posix(),
        )
        manifest_path = args.out_dir / "manifest.json"
        atomic_json(
            {
                "schema_version": 1,
                "artifact_type": config["experiment_id"],
                "created_utc": utc_now(),
                "status": results["status"],
                "winner": winner,
                "preregister_sha256": CONFIG_SHA,
                "source_closure": closure,
                "outputs_excluding_manifest_and_sidecar": [file_record(path) for path in files],
                "output_count_excluding_manifest_and_sidecar": len(files),
                "phase_locks": results["phase_locks"],
                "physical_access": results["physical_access"],
                "no_CSV": True,
            },
            manifest_path,
        )
        sidecar_path = args.out_dir / "manifest.sha256"
        with sidecar_path.open("x", encoding="ascii", newline="\n") as stream:
            stream.write(f"{sha256(manifest_path)}  manifest.json\n")
        print(f"status={results['status']} winner={winner}", flush=True)
        print(f"manifest={manifest_path} sha256={sha256(manifest_path)}", flush=True)
    finally:
        release_heavy_guard(HEAVY_GUARD_PATH, owner)


if __name__ == "__main__":
    main()

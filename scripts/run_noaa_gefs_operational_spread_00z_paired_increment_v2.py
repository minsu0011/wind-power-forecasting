#!/usr/bin/env python3
"""Run the frozen GEFS v3-resolved Stage1 paired-increment experiment.

This executable has no Stage2/final/2025 branch.  It physically rebuilds the
612 controls from bounded 2022-2023 raw NWP prefixes, reads only the exact
registered fit-label slices, saves and reloads all six models, locks complete
candidate bytes, and only then opens the exact registered score-label slices.
"""

from __future__ import annotations

import argparse
import ast
import ctypes
import hashlib
import io
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_shared_q07_multiseed as shared  # noqa: E402
from src.manifest import describe_file, git_state, package_versions, sha256_file, utc_now, write_json_atomic  # noqa: E402
from src.metric import CAPACITY_KWH, TARGET_COLS, group_metrics, score_details  # noqa: E402


BASE_PREREG = PROJECT_ROOT / "configs/noaa_gefs_operational_spread_00z_paired_increment_preregister_v1.json"
V2_PREREG = PROJECT_ROOT / "configs/noaa_gefs_operational_spread_00z_paired_increment_preregister_v2.json"
V3_PREREG = PROJECT_ROOT / "configs/noaa_gefs_operational_spread_00z_paired_increment_preregister_v3.json"
V4_PREREG = PROJECT_ROOT / "configs/noaa_gefs_operational_spread_00z_paired_increment_preregister_v4.json"
V5_PREREG = PROJECT_ROOT / "configs/noaa_gefs_operational_spread_00z_paired_increment_preregister_v5.json"
EXECUTION_PROTOCOL = PROJECT_ROOT / "configs/noaa_gefs_operational_spread_00z_stage1_execution_protocol_v2.json"
INCIDENT = PROJECT_ROOT / "artifacts/incidents/noaa_gefs_operational_spread_00z_paired_increment_v1_label_mask_contradiction.json"
EXTRACTOR = PROJECT_ROOT / "scripts/download_noaa_gefs_operational_spread_00z_original_v2.py"
FOCUSED_TEST = PROJECT_ROOT / "tests/test_noaa_gefs_operational_spread_00z_paired_increment_v2.py"
POSTRUN_TEST = PROJECT_ROOT / "tests/test_noaa_gefs_operational_spread_00z_paired_increment_v2_postrun.py"
SOURCE_ROOT = PROJECT_ROOT / "artifacts/external/noaa_gefs_operational_spread_00z_original_v2/stage1_source_through_operating_2023"
STAGE2_SOURCE = PROJECT_ROOT / "artifacts/external/noaa_gefs_operational_spread_00z_original_v2/stage2_source_operating_2024"
SOURCE_LAUNCH_LOCK = PROJECT_ROOT / "artifacts/audits/noaa_gefs_operational_spread_00z_original_v2_stage1_launch_lock.json"
SOURCE_LAUNCH_LOCK_SHA = SOURCE_LAUNCH_LOCK.with_suffix(".json.sha256")
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/postgate/noaa_gefs_operational_spread_00z_paired_increment_strict_v2"
HEAVY_GUARD = PROJECT_ROOT / "artifacts/locks/heavy_cpu_fit.pid.json"
MODEL_ATTEMPT_TOMBSTONE = PROJECT_ROOT / "artifacts/audits/noaa_gefs_operational_spread_00z_paired_increment_v5_stage1_single_attempt.json"

BASE_SHA = "ccc2a55b979cbe50a6d79abc76b36e8553020da3f9f8d4461c1b4c2325438d9b"
V2_SHA = "7af72639b806c559d1a417b8254f43582c7e728daeba4f5a42ae9cb42af8b2cd"
V3_SHA = "a6600228fd646f1dc347b682ad2db1008ccc9129707caa241d0842dac09aa893"
V4_SHA = "07e9ec6e9f08d18b7fe09421bcae6308009889a7970c9471a30447fd47e0e9a4"
V5_SHA = "df1514f6a8a1709929bc96e82bd043d11b0215de117f25a06232bab8fa531187"
EXECUTION_PROTOCOL_SHA = "d3785e8557dd5d48a89174dd0a6f2f6a2321d588d17d7a75ec051423e936192d"
INCIDENT_SHA = "360b1369d83568d381bb2b6d9109debef21e9e32c13445bd6ef875fb82541f43"

YEAR_2022 = pd.date_range("2022-01-01 01:00", "2023-01-01 00:00", freq="h", name="forecast_kst_dtm")
YEAR_2023 = pd.date_range("2023-01-01 01:00", "2024-01-01 00:00", freq="h", name="forecast_kst_dtm")
PRE2024 = pd.date_range("2022-01-01 01:00", "2024-01-01 00:00", freq="h", name="forecast_kst_dtm")
G3_H1 = pd.date_range("2023-01-01 01:00", "2023-07-01 00:00", freq="h", name="forecast_kst_dtm")
G3_H2 = pd.date_range("2023-07-01 01:00", "2024-01-01 00:00", freq="h", name="forecast_kst_dtm")
STAGE1_FINAL = pd.Timestamp("2024-01-01 00:00")

TRANSFER_WEIGHT = 0.25
MODEL_PARAMETERS = {
    "objective": "regression_l1",
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
DIRECT_COMPONENT_COLUMNS = (
    "gefs00z__u10_mean_ms",
    "gefs00z__v10_mean_ms",
    "gefs00z__u10_std_ms",
    "gefs00z__v10_std_ms",
    "gefs00z__u850_mean_ms",
    "gefs00z__v850_mean_ms",
    "gefs00z__u850_std_ms",
    "gefs00z__v850_std_ms",
)
EXTENDED_COLUMNS = (
    "gefs00z__u10_mean_ms",
    "gefs00z__v10_mean_ms",
    "gefs00z__ws10_mean_ms",
    "gefs00z__u10_std_ms",
    "gefs00z__v10_std_ms",
    "gefs00z__ws10_mean_minus_cross_hub_ws_mean",
    "gefs00z__u850_mean_ms",
    "gefs00z__v850_mean_ms",
    "gefs00z__ws850_mean_ms",
    "gefs00z__u850_std_ms",
    "gefs00z__v850_std_ms",
    "gefs00z__ws850_mean_minus_control_gfs850_idw_ws",
)
CONTROL_HUB_WS = "cross__hub_ws_mean"
CONTROL_GFS850_U = "gfs__idw__isobaricInhPa_850_u"
CONTROL_GFS850_V = "gfs__idw__isobaricInhPa_850_v"


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    write_json_atomic(path, _json_ready(payload), overwrite=False)


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(_json_ready(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _write_sha_sidecar(path: Path) -> Path:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    payload = f"{sha256_file(path)}  {path.name}\n"
    temporary = sidecar.with_name(f".{sidecar.name}.tmp-{os.getpid()}")
    if sidecar.exists():
        raise FileExistsError(sidecar)
    temporary.write_text(payload, encoding="ascii")
    os.replace(temporary, sidecar)
    return sidecar


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        frame.to_parquet(temporary, engine="pyarrow", index=True, compression="zstd")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_joblib(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        joblib.dump(model, temporary, compress=3)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _copy_exclusive(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst)


def _verify_sha(path: Path, expected: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise AssertionError(f"registered file identity differs: {path}")


def _verify_record(record: Mapping[str, Any]) -> Path:
    path = Path(str(record["path"]))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    expected_bytes = int(record.get("bytes", record.get("size_bytes", -1)))
    if not path.is_file() or path.stat().st_size != expected_bytes or sha256_file(path) != str(record["sha256"]):
        raise AssertionError(f"bound record differs: {path}")
    return path.resolve()


def verify_effective_preregister() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    for path, expected in (
        (BASE_PREREG, BASE_SHA), (V2_PREREG, V2_SHA), (V3_PREREG, V3_SHA), (V4_PREREG, V4_SHA), (V5_PREREG, V5_SHA),
        (EXECUTION_PROTOCOL, EXECUTION_PROTOCOL_SHA), (INCIDENT, INCIDENT_SHA),
    ):
        _verify_sha(path, expected)
    v1 = json.loads(BASE_PREREG.read_text(encoding="utf-8"))
    v2 = json.loads(V2_PREREG.read_text(encoding="utf-8"))
    v3 = json.loads(V3_PREREG.read_text(encoding="utf-8"))
    v4 = json.loads(V4_PREREG.read_text(encoding="utf-8"))
    v5 = json.loads(V5_PREREG.read_text(encoding="utf-8"))
    protocol = json.loads(EXECUTION_PROTOCOL.read_text(encoding="utf-8"))
    if v2["supersedes"]["sha256"] != BASE_SHA or v3["lineage"]["label_and_source_phase_v2"]["sha256"] != V2_SHA:
        raise AssertionError("v3-v2-v1 lineage differs")
    if v3["lineage"]["stage1_execution_protocol"]["sha256"] != EXECUTION_PROTOCOL_SHA:
        raise AssertionError("execution-protocol lineage differs")
    if v4["supersedes"]["sha256"] != V3_SHA or not v4["effective_contract"]["stage1_final_2024_01_01_00_modeled"]:
        raise AssertionError("v4-v3 chronology/effective lineage differs")
    if v5["supersedes"]["sha256"] != V4_SHA or v5["effective_contract"]["inherited_output_directory"] != DEFAULT_OUTPUT.relative_to(PROJECT_ROOT).as_posix():
        raise AssertionError("v5-v4 chronology/output lineage differs")
    if v1["paired_model"]["parameters"] != MODEL_PARAMETERS or float(v1["paired_model"]["transfer_weight"]) != TRANSFER_WEIGHT:
        raise AssertionError("model parameters or transfer weight differ")
    if tuple(v1["extended_features_after_control"]) != EXTENDED_COLUMNS:
        raise AssertionError("twelve-feature order differs")
    if v3["immutable_scientific_inheritance"]["candidate_count"] != 1:
        raise AssertionError("candidate count differs")
    if v3["superseded_v2_ambiguities"]["stage1_terminal_boundary"]["resolution"].startswith("Preserve v1 exactly") is False:
        raise AssertionError("Stage1 final-row resolution differs")
    if v3["physical_bounded_stage1_control_rebuild"]["timestamp_index"]["rows"] != len(PRE2024):
        raise AssertionError("bounded weather index differs")
    if protocol["status"] != "FROZEN_BEFORE_STAGE1_SOURCE_VALUE_GET_LABEL_READ_MODEL_FIT_PREDICTION_OR_SCORE":
        raise AssertionError("execution protocol is not frozen")
    expected_top = {
        "schema_version", "protocol_id", "created_utc", "status", "effective_preregister",
        "stage1_label_source", "actual_stage1_runner", "source_extraction_order", "safety_accounting_at_freeze",
    }
    if set(protocol) != expected_top:
        raise AssertionError("execution protocol gained unregistered top-level fields")
    return v1, v2, v3, v4, v5, protocol


def _module_files(module: str) -> list[Path]:
    if not module:
        return []
    parts = module.split(".")
    found: set[Path] = set()
    file_path = PROJECT_ROOT.joinpath(*parts).with_suffix(".py")
    init_path = PROJECT_ROOT.joinpath(*parts, "__init__.py")
    if file_path.is_file():
        found.add(file_path.resolve())
    if init_path.is_file():
        found.add(init_path.resolve())
    for depth in range(1, len(parts)):
        init = PROJECT_ROOT.joinpath(*parts[:depth], "__init__.py")
        if init.is_file():
            found.add(init.resolve())
    return sorted(found)


def resolve_ast_local_import_closure(entrypoints: Iterable[Path]) -> tuple[Path, ...]:
    queue = [path.resolve() for path in entrypoints]
    visited: set[Path] = set()
    while queue:
        path = queue.pop()
        if path in visited:
            continue
        visited.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    relative = path.relative_to(PROJECT_ROOT).with_suffix("").parts[:-1]
                    keep = len(relative) - node.level + 1
                    prefix = relative[: max(keep, 0)]
                    base = ".".join((*prefix, *((node.module or "").split("."))))
                else:
                    base = node.module or ""
                modules.append(base.strip("."))
                modules.extend(f"{base}.{alias.name}".strip(".") for alias in node.names if alias.name != "*")
            for module in modules:
                resolved = _module_files(module)
                if resolved:
                    queue.extend(resolved)
                else:
                    top = module.split(".")[0] if module else ""
                    namespace_path = PROJECT_ROOT.joinpath(*module.split(".")) if module else PROJECT_ROOT
                    if namespace_path.is_dir():
                        continue
                    if top and (PROJECT_ROOT / top).exists():
                        raise AssertionError(f"unresolved local import: {module}")
    return tuple(sorted(visited))


def _assert_records_current(records: Sequence[Mapping[str, Any]]) -> None:
    for record in records:
        _verify_record(record)


def verify_source_launch_lock() -> dict[str, Any]:
    if not SOURCE_LAUNCH_LOCK.is_file() or not SOURCE_LAUNCH_LOCK_SHA.is_file():
        raise FileNotFoundError("source launch lock/sidecar absent")
    expected = f"{sha256_file(SOURCE_LAUNCH_LOCK)}  {SOURCE_LAUNCH_LOCK.name}\n"
    if SOURCE_LAUNCH_LOCK_SHA.read_text(encoding="ascii") != expected:
        raise AssertionError("source launch-lock sidecar differs")
    lock = json.loads(SOURCE_LAUNCH_LOCK.read_text(encoding="utf-8"))
    if lock["status"] != "FROZEN_BEFORE_FIRST_STAGE1_RANGE_GET" or lock["scope"]["2024_operating_values"] != 0:
        raise AssertionError("source launch-lock status/scope differs")
    _assert_records_current(lock["source_closure"])
    _assert_records_current(lock["source_roots"])
    _assert_records_current(lock["recursive_local_imports"])
    runner_records = [record for record in lock["source_roots"] if Path(record["path"]).name == Path(__file__).name]
    if len(runner_records) != 1 or runner_records[0]["sha256"] != sha256_file(Path(__file__)):
        raise AssertionError("actual Stage1 runner is not the pre-GET-bound runner")
    return lock


def verify_source_canonical() -> dict[str, Any]:
    if STAGE2_SOURCE.exists():
        raise AssertionError("conditional Stage2 source exists before Stage1 promotion")
    manifest_path = SOURCE_ROOT / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Stage1 source canonical absent")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "PASS_STAGE1_SOURCE_EXTRACTION":
        raise AssertionError("Stage1 source did not pass extraction")
    _assert_records_current(manifest["source_closure"])
    _assert_records_current(manifest["artifacts"])
    if manifest["nonmutation"] != {"label_model_metric_csv_writes": 0, "stage2_namespace_created": False}:
        raise AssertionError("source nonmutation record differs")
    return manifest


def _closure_record() -> dict[str, Any]:
    closure = resolve_ast_local_import_closure((Path(__file__), EXTRACTOR))
    return {
        "resolver": "recursive Python AST local imports plus executed package initializers",
        "resolved_relative_paths": [path.relative_to(PROJECT_ROOT).as_posix() for path in closure],
        "resolved_files": [describe_file(path) for path in closure],
        "unresolved_local_imports": [],
        "focused_test": describe_file(FOCUSED_TEST),
        "postrun_test": describe_file(POSTRUN_TEST),
        "prereg_v1": describe_file(BASE_PREREG),
        "prereg_v2": describe_file(V2_PREREG),
        "prereg_v3": describe_file(V3_PREREG),
        "prereg_v4": describe_file(V4_PREREG),
        "prereg_v5": describe_file(V5_PREREG),
        "execution_protocol": describe_file(EXECUTION_PROTOCOL),
        "incident": describe_file(INCIDENT),
        "source_launch_lock": describe_file(SOURCE_LAUNCH_LOCK),
        "source_launch_lock_sidecar": describe_file(SOURCE_LAUNCH_LOCK_SHA),
        "source_manifest": describe_file(SOURCE_ROOT / "manifest.json"),
        "model_single_attempt_tombstone": describe_file(MODEL_ATTEMPT_TOMBSTONE),
    }


def _assert_closure_equal(first: Mapping[str, Any], second: Mapping[str, Any]) -> None:
    if _json_ready(first) != _json_ready(second):
        raise AssertionError("source closure changed")


def _label_contract(protocol: Mapping[str, Any]) -> Mapping[str, Any]:
    contract = protocol["stage1_label_source"]
    if int(contract["header_bytes"]) != 48 or int(contract["expected_bytes_without_whole_file_read"]) != 1_138_967:
        raise AssertionError("label source size/header contract differs")
    return contract


def _decode_label_slice(path: Path, spec: Mapping[str, Any], expected_index: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = tuple(str(column) for column in spec["columns"])
    if not columns or columns[0] != "kst_dtm" or any(column not in ("kst_dtm", *TARGET_COLS) for column in columns):
        raise AssertionError("label columns differ")
    start, end = int(spec["source_start_byte"]), int(spec["source_end_byte"])
    with path.open("rb") as stream:
        header = stream.readline()
        if len(header) != 48:
            raise AssertionError("label header byte count differs")
        stream.seek(start)
        body = stream.read(end - start)
        if stream.tell() != end:
            raise AssertionError("label reader crossed registered byte boundary")
    parser_payload = header + body
    if len(parser_payload) != int(spec["parser_bytes"]) or hashlib.sha256(parser_payload).hexdigest() != str(spec["parser_sha256"]):
        raise AssertionError("label parser slice identity differs")
    frame = pd.read_csv(io.BytesIO(parser_payload), encoding="utf-8-sig", usecols=list(columns), nrows=int(spec["nrows"]), memory_map=False)
    if tuple(frame.columns) != columns or len(frame) != int(spec["nrows"]):
        raise AssertionError("label slice schema/rows differ")
    index = pd.DatetimeIndex(pd.to_datetime(frame.pop("kst_dtm"), errors="raise"), name="forecast_kst_dtm")
    if not index.equals(expected_index):
        raise AssertionError("label slice timestamp mask differs")
    frame.index = index
    frame = frame.astype(np.float64)
    values = np.ascontiguousarray(frame.to_numpy(np.float64))
    return frame, {
        "columns": list(frame.columns), "rows": len(frame), "value_cells": int(frame.size),
        "start": index.min(), "end": index.max(), "parser_bytes": len(parser_payload),
        "parser_sha256": hashlib.sha256(parser_payload).hexdigest(),
        "values_float64_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
    }


def _mask_sha(groups_and_indexes: Sequence[tuple[Sequence[str], pd.DatetimeIndex]]) -> str:
    records = sorted((group, timestamp.strftime("%Y-%m-%d %H:%M:%S")) for groups, index in groups_and_indexes for group in groups for timestamp in index)
    return hashlib.sha256("".join(f"{group}\t{timestamp}\n" for group, timestamp in records).encode("utf-8")).hexdigest()


def read_stage1_fit_labels(raw_dir: Path, out_dir: Path, protocol: Mapping[str, Any]) -> tuple[dict[str, pd.Series], dict[str, Any]]:
    ledger = out_dir / "stage1_fit_label_access.json"
    if ledger.exists() or (out_dir / "stage1_score_label_access.json").exists() or (out_dir / "stage1_candidate_before_score_labels_lock.json").exists():
        raise AssertionError("fit-label phase is not single-use and pre-candidate")
    contract = _label_contract(protocol)
    source = raw_dir / "train/train_labels.csv"
    if source.resolve() != Path(contract["path"]).resolve() or source.stat().st_size != int(contract["expected_bytes_without_whole_file_read"]):
        raise AssertionError("label source path/size differs")
    fit = contract["fit_slices_before_global_candidate_lock"]
    g12, evidence_g12 = _decode_label_slice(source, fit["g12_2022"], YEAR_2022)
    g3, evidence_g3 = _decode_label_slice(source, fit["g3_2023_h1"], G3_H1)
    if tuple(g12.columns) != TARGET_COLS[:2] or tuple(g3.columns) != (TARGET_COLS[2],):
        raise AssertionError("fit-label group isolation differs")
    mask_sha = _mask_sha(((TARGET_COLS[:2], YEAR_2022), ((TARGET_COLS[2],), G3_H1)))
    payload = {
        "schema_version": 1, "phase": "stage1_fit_exact_single_use", "created_utc": utc_now(),
        "config_v3_sha256": V3_SHA, "config_v4_sha256": V4_SHA, "config_v5_sha256": V5_SHA, "g12": evidence_g12, "g3": evidence_g3,
        "exact_value_cells": int(g12.size + g3.size), "exact_cell_mask_sha256": mask_sha,
        "score_label_cells_materialized": 0, "2024_label_cells_materialized": 0,
        "whole_label_file_hashed": False,
    }
    if payload["exact_value_cells"] != 21_864:
        raise AssertionError("fit-label cell count differs")
    _write_json(ledger, payload)
    return {TARGET_COLS[0]: g12[TARGET_COLS[0]], TARGET_COLS[1]: g12[TARGET_COLS[1]], TARGET_COLS[2]: g3[TARGET_COLS[2]]}, payload


def _verify_lock_sidecar(lock_path: Path) -> dict[str, Any]:
    sidecar = lock_path.with_suffix(lock_path.suffix + ".sha256")
    if not lock_path.is_file() or not sidecar.is_file():
        raise FileNotFoundError("global candidate lock/sidecar absent")
    expected = f"{sha256_file(lock_path)}  {lock_path.name}\n"
    if sidecar.read_text(encoding="ascii") != expected:
        raise AssertionError("global candidate lock byte sidecar differs")
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    if payload.get("lock_kind") != "stage1_candidate_before_score_labels" or payload.get("config_v3_sha256") != V3_SHA or payload.get("config_v4_sha256") != V4_SHA or payload.get("config_v5_sha256") != V5_SHA:
        raise AssertionError("global candidate lock identity differs")
    if payload.get("score_label_value_cells_before_lock") != 0 or payload.get("metric_values_before_lock") != 0:
        raise AssertionError("global lock was written after score/metric access")
    if not HEAVY_GUARD.is_file() or json.loads(HEAVY_GUARD.read_text(encoding="utf-8")) != payload.get("continuous_heavy_guard"):
        raise AssertionError("global lock heavy-guard identity is not continuously live before score labels")
    _assert_records_current(payload["bound_outputs"])
    _verify_record(payload["fit_label_access_ledger"])
    _assert_closure_equal(payload["source_closure"], _closure_record())
    return payload


def read_stage1_score_labels(raw_dir: Path, out_dir: Path, protocol: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    ledger = out_dir / "stage1_score_label_access.json"
    if ledger.exists():
        raise AssertionError("score-label phase is single-use")
    lock_path = out_dir / "stage1_candidate_before_score_labels_lock.json"
    lock = _verify_lock_sidecar(lock_path)  # must complete before label path/stat/open
    contract = _label_contract(protocol)
    source = raw_dir / "train/train_labels.csv"
    if source.resolve() != Path(contract["path"]).resolve() or source.stat().st_size != int(contract["expected_bytes_without_whole_file_read"]):
        raise AssertionError("label source path/size differs")
    score_specs = contract["score_slices_only_after_verified_global_candidate_lock"]
    g12, evidence_g12 = _decode_label_slice(source, score_specs["g12_2023"], YEAR_2023)
    g3, evidence_g3 = _decode_label_slice(source, score_specs["g3_2023_h2"], G3_H2)
    if tuple(g12.columns) != TARGET_COLS[:2] or tuple(g3.columns) != (TARGET_COLS[2],):
        raise AssertionError("score-label group isolation differs")
    frame = pd.DataFrame(np.nan, index=YEAR_2023, columns=TARGET_COLS, dtype=np.float64)
    frame.loc[YEAR_2023, list(TARGET_COLS[:2])] = g12.to_numpy(np.float64)
    frame.loc[G3_H2, TARGET_COLS[2]] = g3[TARGET_COLS[2]].to_numpy(np.float64)
    mask_sha = _mask_sha(((TARGET_COLS[:2], YEAR_2023), ((TARGET_COLS[2],), G3_H2)))
    payload = {
        "schema_version": 1, "phase": "stage1_score_exact_single_use_after_global_lock", "created_utc": utc_now(),
        "config_v3_sha256": V3_SHA, "config_v4_sha256": V4_SHA, "config_v5_sha256": V5_SHA, "verified_global_lock_sha256": sha256_file(lock_path),
        "g12": evidence_g12, "g3": evidence_g3, "exact_value_cells": int(g12.size + g3.size),
        "exact_cell_mask_sha256": mask_sha, "2024_label_cells_materialized": 0,
        "whole_label_file_hashed": False,
    }
    if payload["exact_value_cells"] != 21_936:
        raise AssertionError("score-label cell count differs")
    _write_json(ledger, payload)
    return frame, payload


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class HeavyFitGuard:
    def __init__(self, *, defer_success_release: bool = False) -> None:
        self.token = uuid.uuid4().hex
        self.acquired = False
        self.defer_success_release = defer_success_release

    def __enter__(self) -> "HeavyFitGuard":
        HEAVY_GUARD.parent.mkdir(parents=True, exist_ok=True)
        if HEAVY_GUARD.exists():
            record = json.loads(HEAVY_GUARD.read_text(encoding="utf-8"))
            if _pid_alive(int(record.get("pid", -1))):
                raise RuntimeError(f"heavy guard held by live PID {record.get('pid')}")
            stale = HEAVY_GUARD.with_name(f"heavy_cpu_fit.stale-pid{record.get('pid','unknown')}-{uuid.uuid4().hex}.json")
            os.replace(HEAVY_GUARD, stale)
        record = {
            "schema_version": 1, "pid": os.getpid(), "token": self.token,
            "experiment": "noaa_gefs_operational_spread_00z_paired_increment_strict_v5",
            "stage": "stage1", "created_utc": utc_now(),
        }
        descriptor = os.open(HEAVY_GUARD, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
            handle.write("\n")
        self.acquired = True
        return self

    def record(self) -> dict[str, Any]:
        if not self.acquired or not HEAVY_GUARD.is_file():
            raise RuntimeError("heavy guard is not live")
        record = json.loads(HEAVY_GUARD.read_text(encoding="utf-8"))
        if int(record.get("pid", -1)) != os.getpid() or record.get("token") != self.token:
            raise RuntimeError("heavy guard ownership differs")
        return record

    def release(self) -> None:
        if self.acquired and HEAVY_GUARD.exists():
            record = json.loads(HEAVY_GUARD.read_text(encoding="utf-8"))
            if int(record.get("pid", -1)) == os.getpid() and record.get("token") == self.token:
                HEAVY_GUARD.unlink()
            self.acquired = False

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc_type is None and self.defer_success_release:
            return
        self.release()


def _load_external_components() -> pd.DataFrame:
    path = SOURCE_ROOT / "hourly_components.parquet"
    frame = pd.read_parquet(path)
    required = ("operating_date", "forecast_kst_dtm", "source_init_utc", "target_lead_hour", "missing_gefs_00z", *DIRECT_COMPONENT_COLUMNS)
    if tuple(frame.columns) != required or len(frame) != len(PRE2024):
        raise AssertionError("GEFS hourly component schema/rows differ")
    index = pd.DatetimeIndex(pd.to_datetime(frame.pop("forecast_kst_dtm"), errors="raise"), name="forecast_kst_dtm")
    if not index.equals(PRE2024):
        raise AssertionError("GEFS hourly component index differs")
    frame.index = index
    if frame["operating_date"].min() != "2022-01-01" or frame["operating_date"].max() != "2023-12-31":
        raise AssertionError("GEFS operating-day bounds differ")
    if set(frame["missing_gefs_00z"].unique()) != {0}:
        raise AssertionError("Stage1 unexpectedly has unavailable GEFS days")
    numeric = frame.loc[:, DIRECT_COMPONENT_COLUMNS].astype(np.float64)
    if not np.isfinite(numeric.to_numpy()).all() or (numeric[[column for column in numeric if "_std_" in column]] < 0).any().any():
        raise AssertionError("GEFS components are nonfinite or negative spread")
    return numeric


def _build_extended(control: pd.DataFrame, components: pd.DataFrame) -> pd.DataFrame:
    if not control.index.equals(components.index):
        raise AssertionError("control/GEFS indexes differ")
    for column in (CONTROL_HUB_WS, CONTROL_GFS850_U, CONTROL_GFS850_V):
        if column not in control:
            raise AssertionError(f"control missing disagreement column: {column}")
    extra = components.copy().astype(np.float64)
    extra["gefs00z__ws10_mean_ms"] = np.hypot(extra["gefs00z__u10_mean_ms"], extra["gefs00z__v10_mean_ms"])
    extra["gefs00z__ws10_mean_minus_cross_hub_ws_mean"] = extra["gefs00z__ws10_mean_ms"] - control[CONTROL_HUB_WS].to_numpy(np.float64)
    extra["gefs00z__ws850_mean_ms"] = np.hypot(extra["gefs00z__u850_mean_ms"], extra["gefs00z__v850_mean_ms"])
    control_850 = np.hypot(control[CONTROL_GFS850_U].to_numpy(np.float64), control[CONTROL_GFS850_V].to_numpy(np.float64))
    extra["gefs00z__ws850_mean_minus_control_gfs850_idw_ws"] = extra["gefs00z__ws850_mean_ms"] - control_850
    extra = extra.loc[:, EXTENDED_COLUMNS].astype(np.float32)
    result = pd.concat([control.astype(np.float32, copy=False), extra], axis=1)
    if result.shape[1] != 624 or tuple(result.columns[-12:]) != EXTENDED_COLUMNS or not np.isfinite(result.to_numpy(np.float32)).all():
        raise AssertionError("extended feature frame differs")
    return result


def _load_stage1_baseline(v1: Mapping[str, Any]) -> pd.DataFrame:
    g12 = pd.read_parquet(_verify_record(v1["fixed_baselines_and_inputs"]["stage1_g12_baseline"])).astype(np.float64)
    g12.index = pd.DatetimeIndex(g12.index, name="forecast_kst_dtm")
    if not g12.index.equals(YEAR_2023) or tuple(g12.columns) != TARGET_COLS[:2]:
        raise AssertionError("G1/G2 baseline differs")
    raw = pd.read_parquet(_verify_record(v1["fixed_baselines_and_inputs"]["stage1_g3_components"])).astype(np.float64)
    raw.index = pd.DatetimeIndex(raw.index, name="forecast_kst_dtm")
    if not raw.index.equals(G3_H2):
        raise AssertionError("G3 baseline components differ")
    weighted = 0.20 * raw["q07"] + 0.075 * raw["shared_l1"] + 0.425 * raw["shared_q07"] + 0.025 * raw["top200q07"] + 0.275 * raw["ewq06"]
    baseline = pd.DataFrame(np.nan, index=YEAR_2023, columns=TARGET_COLS, dtype=np.float64)
    baseline.loc[:, list(TARGET_COLS[:2])] = g12.to_numpy(np.float64)
    baseline.loc[G3_H2, TARGET_COLS[2]] = np.clip(1.25 * weighted.to_numpy(np.float64) - 1200.0, 0.0, 1.02 * CAPACITY_KWH[TARGET_COLS[2]])
    return baseline


def _fit_model(features: pd.DataFrame, labels: pd.Series, group: str) -> tuple[LGBMRegressor, dict[str, Any]]:
    if not features.index.equals(labels.index):
        raise AssertionError("fit feature/label indexes differ")
    actual = labels.to_numpy(np.float64)
    eligible = np.isfinite(actual) & (actual >= 0.10 * CAPACITY_KWH[group])
    if not eligible.any():
        raise AssertionError("no eligible fit rows")
    target = actual[eligible] / CAPACITY_KWH[group]
    model = LGBMRegressor(**MODEL_PARAMETERS)
    model.fit(features.iloc[np.flatnonzero(eligible)], target)
    return model, {
        "rows": len(features), "eligible_rows": int(eligible.sum()),
        "eligible_mask_sha256": hashlib.sha256(np.ascontiguousarray(eligible.astype(np.uint8)).tobytes()).hexdigest(),
        "features": features.shape[1], "parameters": MODEL_PARAMETERS,
    }


def _predict(model: LGBMRegressor, features: pd.DataFrame) -> np.ndarray:
    return np.clip(np.asarray(model.predict(features), dtype=np.float64), 0.0, 1.02)


def _segments() -> dict[str, pd.DatetimeIndex]:
    make = lambda a, b: pd.date_range(a, b, freq="h", name="forecast_kst_dtm")
    return {
        "full": make("2023-01-01 01:00", "2024-01-01 00:00"),
        "H1": make("2023-01-01 01:00", "2023-07-01 00:00"),
        "H2": make("2023-07-01 01:00", "2024-01-01 00:00"),
        "Q1": make("2023-01-01 01:00", "2023-04-01 00:00"),
        "Q2": make("2023-04-01 01:00", "2023-07-01 00:00"),
        "Q3": make("2023-07-01 01:00", "2023-10-01 00:00"),
        "Q4": make("2023-10-01 01:00", "2024-01-01 00:00"),
    }


def _comparison(actual: pd.Series, baseline: pd.Series, candidate: pd.Series, group: str) -> dict[str, Any]:
    base = group_metrics(actual, baseline, CAPACITY_KWH[group], group_name=group).as_dict()
    cand = group_metrics(actual, candidate, CAPACITY_KWH[group], group_name=group).as_dict()
    base_score = 0.5 * (base["one_minus_nmae"] + base["ficr"])
    cand_score = 0.5 * (cand["one_minus_nmae"] + cand["ficr"])
    return {"baseline": base, "candidate": cand, "baseline_score": base_score, "candidate_score": cand_score, "delta": cand_score - base_score}


def _mixed_comparison(actual: pd.DataFrame, baseline: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, Any]:
    base = score_details(actual, baseline).as_dict()
    cand = score_details(actual, candidate).as_dict()
    return {
        "baseline": base, "candidate": cand,
        "delta_total_score": cand["total_score"] - base["total_score"],
        "delta_one_minus_nmae": cand["one_minus_nmae"] - base["one_minus_nmae"],
        "delta_ficr": cand["ficr"] - base["ficr"],
    }


def _snapshot_inputs(v1: Mapping[str, Any], v3: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "label_slices_registered_without_source_open": protocol["stage1_label_source"],
        "bounded_raw_weather_prefixes_registered_without_source_open": v3["physical_bounded_stage1_control_rebuild"],
        "info_registered_without_source_open": v1["fixed_baselines_and_inputs"]["info_xlsx_expected_without_new_read"],
        "source_manifest": describe_file(SOURCE_ROOT / "manifest.json"),
        "hourly_components_registered": next(item for item in json.loads((SOURCE_ROOT / "manifest.json").read_text(encoding="utf-8"))["artifacts"] if item["path"].endswith("hourly_components.parquet")),
        "stage1_g12_baseline_registered_without_source_open": v1["fixed_baselines_and_inputs"]["stage1_g12_baseline"],
        "stage1_g3_components_registered_without_source_open": v1["fixed_baselines_and_inputs"]["stage1_g3_components"],
        "full_weather_cache_files_opened_or_hashed": 0,
        "label_value_cells_materialized": 0,
        "raw_weather_value_cells_materialized": 0,
        "operating_2024_external_values": 0,
    }


def _bound_outputs(records: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    result = list(records)
    _assert_records_current(result)
    return result


def _write_manifest(out_dir: Path, status: str, closure: Mapping[str, Any]) -> None:
    if list(out_dir.rglob("*.csv")):
        raise AssertionError("GEFS Stage1 must not create CSV")
    files = sorted((path for path in out_dir.rglob("*") if path.is_file() and path.name != "manifest.json"), key=lambda p: p.relative_to(out_dir).as_posix())
    _write_json(out_dir / "manifest.json", {
        "schema_version": 1, "artifact_type": "noaa_gefs_operational_spread_00z_paired_increment_strict_v5",
        "created_utc": utc_now(), "status": status, "config_v1_sha256": BASE_SHA,
        "config_v2_sha256": V2_SHA, "config_v3_sha256": V3_SHA, "config_v4_sha256": V4_SHA, "config_v5_sha256": V5_SHA,
        "source_provenance": closure, "outputs": [describe_file(path) for path in files],
        "output_count_excluding_manifest": len(files), "no_csv": True,
        "operating_2024_external_or_label_values": 0, "2025_requests_or_values": 0,
        "runtime": {"packages": package_versions()}, "git": git_state(PROJECT_ROOT),
    })


def static_audit() -> dict[str, Any]:
    v1, v2, v3, v4, v5, protocol = verify_effective_preregister()
    contract = _label_contract(protocol)
    fit = contract["fit_slices_before_global_candidate_lock"]
    score = contract["score_slices_only_after_verified_global_candidate_lock"]
    if sum(int(item["value_cells"]) for item in fit.values()) != 21_864 or sum(int(item["value_cells"]) for item in score.values()) != 21_936:
        raise AssertionError("exact label-cell accounting differs")
    if tuple(v1["extended_features_after_control"]) != EXTENDED_COLUMNS or len(EXTENDED_COLUMNS) != 12:
        raise AssertionError("feature contract differs")
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"), filename=str(Path(__file__)))
    function_names = {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assigned_names = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name)
    }
    if "run_stage2" in function_names or "YEAR_2024" in assigned_names:
        raise AssertionError("forbidden Stage2 executable branch")
    return {
        "status": "PASS_ACTUAL_STAGE1_RUNNER_STATIC_CONTRACT", "config_v3_sha256": V3_SHA, "config_v4_sha256": V4_SHA, "config_v5_sha256": V5_SHA,
        "fit_label_cells": 21_864, "score_label_cells": 21_936, "models_to_save_reload": 6,
        "features_control": 612, "features_extended": 624, "weight": TRANSFER_WEIGHT,
        "stage1_final_row_modeled": True, "stage2_branch": False,
        "source_launch_lock_exists": SOURCE_LAUNCH_LOCK.exists(), "source_canonical_exists": SOURCE_ROOT.exists(),
        "2024_external_or_label_values": 0, "2025_requests_or_values": 0,
    }


def run_stage1(raw_dir: Path, out_dir: Path) -> None:
    if out_dir.resolve() != DEFAULT_OUTPUT.resolve():
        raise AssertionError("runtime output directory differs from frozen v2 output path")
    v1, _v2, v3, _v4, _v5, protocol = verify_effective_preregister()
    expected_raw_dir = Path(protocol["stage1_label_source"]["path"]).resolve().parents[1]
    if raw_dir.resolve() != expected_raw_dir:
        raise AssertionError("runtime raw directory differs from frozen official input root")
    verify_source_launch_lock()
    verify_source_canonical()
    if STAGE2_SOURCE.exists():
        raise AssertionError("Stage2 source namespace exists")
    if out_dir.exists():
        raise FileExistsError(out_dir)
    matching_quarantine = list((PROJECT_ROOT / "artifacts/quarantine").glob("noaa_gefs_operational_spread_00z_paired_increment_v5_partial_*")) if (PROJECT_ROOT / "artifacts/quarantine").exists() else []
    if MODEL_ATTEMPT_TOMBSTONE.exists() or matching_quarantine:
        raise RuntimeError("Stage1 model single-attempt tombstone/quarantine already exists; retry forbidden")
    attempt_token = hashlib.sha256(f"{os.getpid()}:{uuid.uuid4().hex}".encode("ascii")).hexdigest()
    _write_json_exclusive(MODEL_ATTEMPT_TOMBSTONE, {
        "schema_version": 1,
        "attempt_id": "noaa_gefs_operational_spread_00z_paired_increment_v5_stage1_single_attempt",
        "created_utc": utc_now(), "status": "SINGLE_ATTEMPT_CONSUMED_ON_EXCLUSIVE_CREATE_NO_RETRY",
        "pid": os.getpid(), "token": attempt_token, "config_v4_sha256": V4_SHA, "config_v5_sha256": V5_SHA,
        "source_manifest": describe_file(SOURCE_ROOT / "manifest.json"),
        "label_cells_before_tombstone": 0, "model_fits_predictions_scores": 0,
        "operating_2024_external_or_label_values": 0, "2025_requests_or_values": 0,
    })
    score_guard: HeavyFitGuard | None = None
    try:
        out_dir.mkdir(parents=True)
        for source in (BASE_PREREG, V2_PREREG, V3_PREREG, V4_PREREG, V5_PREREG, EXECUTION_PROTOCOL, INCIDENT, SOURCE_LAUNCH_LOCK, SOURCE_LAUNCH_LOCK_SHA):
            _copy_exclusive(source, out_dir / "provenance" / source.name)
        closure = _closure_record()
        snapshot = _snapshot_inputs(v1, v3, protocol)
        prescore = out_dir / "stage1_prescore_lock.json"
        _write_json(prescore, {
            "schema_version": 1, "created_utc": utc_now(), "config_v3_sha256": V3_SHA, "config_v4_sha256": V4_SHA, "config_v5_sha256": V5_SHA,
            "source_closure": closure, "input_snapshot": snapshot,
            "candidate_fits_predictions_metrics_before_lock": 0, "score_label_cells_before_lock": 0,
            "operating_2024_external_or_label_values": 0, "2025_requests_or_values": 0,
        })
        _write_sha_sidecar(prescore)

        verified_info = describe_file(_verify_record(v1["fixed_baselines_and_inputs"]["info_xlsx_expected_without_new_read"]))
        verified_raw_prefix_files: dict[str, Any] = {}
        for source_name in ("ldaps", "gfs"):
            spec = v3["physical_bounded_stage1_control_rebuild"][f"{source_name}_prefix"]
            path = Path(spec["path"]).resolve()
            if not path.is_file() or path.stat().st_size != int(spec["whole_file_bytes_observed_without_suffix_hash"]):
                raise AssertionError(f"{source_name} raw file path/size differs")
            verified_raw_prefix_files[source_name] = {
                "path": str(path), "whole_file_bytes": path.stat().st_size,
                "physical_prefix_bytes": spec["physical_prefix_bytes"],
                "physical_prefix_sha256": spec["physical_prefix_sha256"],
                "whole_file_hash_computed": False,
            }
        features, raw_contract = shared._read_stage1_raw_features(raw_dir, pd.DataFrame(index=PRE2024))
        if any(frame.shape != (17_520, 612) or not frame.index.equals(PRE2024) for frame in features.values()):
            raise AssertionError("bounded 612-feature control rebuild differs")
        components = _load_external_components()
        fit_labels, fit_ledger = read_stage1_fit_labels(raw_dir, out_dir, protocol)
        baseline = _load_stage1_baseline(v1)
        baseline_path = out_dir / "stage1/baseline_corrected_v3.parquet"
        _atomic_parquet(baseline, baseline_path)

        train_indexes = {TARGET_COLS[0]: YEAR_2022, TARGET_COLS[1]: YEAR_2022, TARGET_COLS[2]: G3_H1}
        apply_indexes = {TARGET_COLS[0]: YEAR_2023, TARGET_COLS[1]: YEAR_2023, TARGET_COLS[2]: G3_H2}
        increments = pd.DataFrame(0.0, index=YEAR_2023, columns=TARGET_COLS, dtype=np.float64)
        candidate = baseline.copy()
        bound_records: list[Mapping[str, Any]] = [describe_file(baseline_path)]
        training: dict[str, Any] = {}
        group_locks: dict[str, Any] = {}
        score_guard = HeavyFitGuard(defer_success_release=True)
        with score_guard:
            for group in TARGET_COLS:
                train_index, apply_index = train_indexes[group], apply_indexes[group]
                control_train = features[group].loc[train_index]
                control_apply = features[group].loc[apply_index]
                extended_train = _build_extended(control_train, components.loc[train_index])
                extended_apply = _build_extended(control_apply, components.loc[apply_index])
                labels = fit_labels[group]
                control_model, control_meta = _fit_model(control_train, labels, group)
                extended_model, extended_meta = _fit_model(extended_train, labels, group)
                control_live = _predict(control_model, control_apply)
                extended_live = _predict(extended_model, extended_apply)
                control_path = out_dir / f"stage1/models/{group}_control.joblib"
                extended_path = out_dir / f"stage1/models/{group}_extended.joblib"
                _atomic_joblib(control_model, control_path)
                _atomic_joblib(extended_model, extended_path)
                control_reload = joblib.load(control_path)
                extended_reload = joblib.load(extended_path)
                control_replay = _predict(control_reload, control_apply)
                extended_replay = _predict(extended_reload, extended_apply)
                if not np.array_equal(control_live, control_replay) or not np.array_equal(extended_live, extended_replay):
                    raise AssertionError(f"saved-model reload prediction bits differ: {group}")
                increment = extended_replay - control_replay
                increments.loc[apply_index, group] = increment
                values = np.clip(baseline.loc[apply_index, group].to_numpy(np.float64) + TRANSFER_WEIGHT * CAPACITY_KWH[group] * increment, 0.0, 1.02 * CAPACITY_KWH[group])
                candidate.loc[apply_index, group] = values
                # v3 explicitly models the Stage1 final row; it is not forced to identity.
                diagnostic = pd.DataFrame({
                    "control_cf_reloaded": control_replay, "extended_cf_reloaded": extended_replay,
                    "increment_cf": increment, "baseline_kwh": baseline.loc[apply_index, group].to_numpy(np.float64),
                    "candidate_kwh": values, "stage1_final_row_modeled": apply_index == STAGE1_FINAL,
                }, index=apply_index)
                diagnostic_path = out_dir / f"stage1/diagnostics/{group}.parquet"
                _atomic_parquet(diagnostic, diagnostic_path)
                model_records = [describe_file(control_path), describe_file(extended_path), describe_file(diagnostic_path)]
                bound_records.extend(model_records)
                group_lock_path = out_dir / f"stage1/group_locks/{group}.json"
                _write_json(group_lock_path, {
                    "schema_version": 1, "lock_kind": "stage1_group_model_and_surface_before_score_labels",
                    "created_utc": utc_now(), "config_v3_sha256": V3_SHA, "config_v4_sha256": V4_SHA, "config_v5_sha256": V5_SHA, "group": group,
                    "fit_label_access_ledger": describe_file(out_dir / "stage1_fit_label_access.json"),
                    "bound_outputs": model_records, "control_training": control_meta, "extended_training": extended_meta,
                    "saved_models_reloaded": 2, "prediction_replay_bit_exact": True,
                    "continuous_heavy_guard": score_guard.record(),
                    "apply_start": apply_index.min(), "apply_end": apply_index.max(),
                    "stage1_final_row_modeled": bool(STAGE1_FINAL in apply_index),
                    "score_label_value_cells_before_lock": 0, "metric_values_before_lock": 0,
                })
                _write_sha_sidecar(group_lock_path)
                group_locks[group] = {"lock": describe_file(group_lock_path), "sidecar": describe_file(group_lock_path.with_suffix(".json.sha256"))}
                bound_records.extend(group_locks[group].values())
                training[group] = {"control": control_meta, "extended": extended_meta, "reload_bit_exact": True}

        increment_path = out_dir / "stage1/paired_increment_cf.parquet"
        candidate_path = out_dir / "stage1/candidate.parquet"
        _atomic_parquet(increments, increment_path)
        _atomic_parquet(candidate, candidate_path)
        bound_records.extend([describe_file(increment_path), describe_file(candidate_path)])
        bound_records = _bound_outputs(bound_records)
        global_lock_path = out_dir / "stage1_candidate_before_score_labels_lock.json"
        _write_json(global_lock_path, {
            "schema_version": 1, "lock_kind": "stage1_candidate_before_score_labels", "created_utc": utc_now(),
            "config_v1_sha256": BASE_SHA, "config_v2_sha256": V2_SHA, "config_v3_sha256": V3_SHA, "config_v4_sha256": V4_SHA, "config_v5_sha256": V5_SHA,
            "source_closure": closure, "input_snapshot": snapshot,
            "verified_info_workbook": verified_info, "verified_raw_prefix_files": verified_raw_prefix_files,
            "fit_label_access_ledger": describe_file(out_dir / "stage1_fit_label_access.json"),
            "group_locks": group_locks, "bound_outputs": bound_records, "training": training,
            "continuous_heavy_guard": score_guard.record(),
            "six_models_saved_and_reloaded": True, "all_prediction_replays_bit_exact": True,
            "stage1_final_2024_01_01_00_modeled": True,
            "score_label_value_cells_before_lock": 0, "metric_values_before_lock": 0,
            "operating_2024_external_or_label_values": 0, "2025_requests_or_values": 0,
        })
        _write_sha_sidecar(global_lock_path)

        # Authoritative candidate bytes are reloaded only after the global lock.
        locked_baseline = pd.read_parquet(baseline_path).astype(np.float64)
        locked_candidate = pd.read_parquet(candidate_path).astype(np.float64)
        locked_increment = pd.read_parquet(increment_path).astype(np.float64)
        for frame in (locked_baseline, locked_candidate, locked_increment):
            frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
        score_labels, score_ledger = read_stage1_score_labels(raw_dir, out_dir, protocol)

        segments = _segments()
        comparisons: dict[str, Any] = {}
        deltas: list[float] = []
        for group in TARGET_COLS[:2]:
            comparisons[group] = {}
            for segment in ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4"):
                index = segments[segment]
                record = _comparison(score_labels.loc[index, group], locked_baseline.loc[index, group], locked_candidate.loc[index, group], group)
                comparisons[group][segment] = record
                deltas.append(float(record["delta"]))
        comparisons[TARGET_COLS[2]] = {}
        for segment, index in (("full_H2", G3_H2), ("Q3", segments["Q3"]), ("Q4", segments["Q4"])):
            record = _comparison(score_labels.loc[index, TARGET_COLS[2]], locked_baseline.loc[index, TARGET_COLS[2]], locked_candidate.loc[index, TARGET_COLS[2]], TARGET_COLS[2])
            comparisons[TARGET_COLS[2]][segment] = record
            deltas.append(float(record["delta"]))
        if len(deltas) != 17:
            raise AssertionError("registered comparison count differs")
        mixed = {
            "full": _mixed_comparison(score_labels.loc[segments["full"]], locked_baseline.loc[segments["full"]], locked_candidate.loc[segments["full"]]),
            "H2": _mixed_comparison(score_labels.loc[segments["H2"]], locked_baseline.loc[segments["H2"]], locked_candidate.loc[segments["H2"]]),
        }
        passed = all(delta > 0.0 for delta in deltas) and all(
            record["delta_one_minus_nmae"] >= 0.0 and record["delta_ficr"] >= 0.0 for record in mixed.values()
        )
        result_path = out_dir / "stage1_results.json"
        _write_json(result_path, {
            "schema_version": 1, "created_utc": utc_now(), "config_v3_sha256": V3_SHA, "config_v4_sha256": V4_SHA, "config_v5_sha256": V5_SHA,
            "candidate_before_score_labels_lock": describe_file(global_lock_path),
            "candidate_before_score_labels_lock_sidecar": describe_file(global_lock_path.with_suffix(".json.sha256")),
            "score_label_evidence_after_lock": score_ledger, "comparisons": comparisons,
            "delta_vector_17": deltas, "mixed_component_gates": mixed,
            "minimum_group_slice_delta": min(deltas), "stage1_passed": passed,
            "stage1_final_2024_01_01_00_modeled": True,
            "operating_2024_external_or_label_values": 0, "2025_requests_or_values": 0,
            "raw_feature_contract": raw_contract,
        })
        selection_path = out_dir / "stage1_selection_lock.json"
        _write_json(selection_path, {
            "schema_version": 1, "created_utc": utc_now(), "config_v3_sha256": V3_SHA, "config_v4_sha256": V4_SHA, "config_v5_sha256": V5_SHA,
            "stage1_results": describe_file(result_path), "global_candidate_lock": describe_file(global_lock_path),
            "continuous_heavy_guard_through_result_seal": score_guard.record(),
            "stage1_passed": passed, "stage2_source_open_allowed": passed,
            "stage2_source_opened_by_this_runner": False, "2024_label_cells": 0, "2025_requests_or_values": 0,
        })
        score_guard.release()
        score_guard = None
        _assert_closure_equal(closure, _closure_record())
        _write_manifest(out_dir, "PASSED_STAGE1_AWAITING_SEPARATE_STAGE2_SOURCE_AUTHORIZATION" if passed else "REJECTED_STAGE1", closure)
        print(json.dumps({"event": "stage1_complete", "passed": passed, "manifest_sha256": sha256_file(out_dir / "manifest.json")}, sort_keys=True), flush=True)
    except BaseException as exc:
        if score_guard is not None:
            score_guard.release()
            score_guard = None
        quarantine = PROJECT_ROOT / "artifacts/quarantine"
        quarantine.mkdir(parents=True, exist_ok=True)
        target = quarantine / f"noaa_gefs_operational_spread_00z_paired_increment_v5_partial_{os.getpid()}_{uuid.uuid4().hex}"
        if out_dir.exists():
            os.replace(out_dir, target)
            (target / "failure.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        failure_path = PROJECT_ROOT / "artifacts/incidents" / f"noaa_gefs_operational_spread_00z_paired_increment_v5_stage1_failure_{attempt_token}.json"
        _write_json_exclusive(failure_path, {
            "schema_version": 1, "incident_id": failure_path.stem, "created_utc": utc_now(),
            "status": "TERMINAL_MODEL_ATTEMPT_FAILURE_NO_RETRY",
            "attempt_tombstone": describe_file(MODEL_ATTEMPT_TOMBSTONE),
            "error": f"{type(exc).__name__}: {exc}",
            "quarantine": str(target.resolve()) if target.exists() else None,
            "refit_retry_retune_or_alternate_source_allowed": False,
        })
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-audit", action="store_true")
    parser.add_argument("--run-stage1", action="store_true")
    parser.add_argument("--raw-dir", type=Path, default=Path(r"data/local/open"))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.static_audit == args.run_stage1:
        raise ValueError("choose exactly one of --static-audit or --run-stage1")
    if args.static_audit:
        print(json.dumps(_json_ready(static_audit()), sort_keys=True))
        return 0
    run_stage1(args.raw_dir.resolve(), args.out_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

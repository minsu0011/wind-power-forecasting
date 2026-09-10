from __future__ import annotations

import argparse
import atexit
import csv
import hashlib
import json
import os
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_ecmwf_ifs025_paired_increment import (
    HEAVY_GUARD_PATH,
    SLICES,
    apply_paired_increment,
    atomic_json,
    atomic_parquet,
    comparison_payload,
    file_record,
    gate_summary,
    pid_is_alive,
    sha256,
)
from src.metric import CAPACITY_KWH, TARGET_COLS


CONFIG_PATH = ROOT / "configs/icon_global_paired_increment_preregister_v3.json"
SIDECAR_PATH = CONFIG_PATH.with_suffix(".sha256")
BASE_CONFIG_PATH = ROOT / "configs/icon_global_paired_increment_preregister_v1.json"
BASE_SIDECAR_PATH = BASE_CONFIG_PATH.with_suffix(".sha256")
PHASE_CONFIG_PATH = ROOT / "configs/icon_global_paired_increment_preregister_v2.json"
PHASE_SIDECAR_PATH = PHASE_CONFIG_PATH.with_suffix(".sha256")
EXTERNAL_PATH = ROOT / "artifacts/external/openmeteo_icon_global_previous_runs_v1/icon_global_group_centroids_2024.parquet"
SOURCE_MANIFEST_PATH = EXTERNAL_PATH.with_name("source_manifest.json")
PRIMARY_PATH = ROOT / "artifacts/oof/gate2024_locked_v3_cf_fix.parquet"
INTERACTION_PATH = ROOT / "artifacts/oof/gate2024_recent_v4_cf_fix_calibration_fit.parquet"
OUTPUT_DIR = ROOT / "artifacts/postgate/icon_global_paired_increment_2024_forward_v3"
LABEL_PATH = Path(r"data/local/open/train/train_labels.csv")
CACHE_PATHS = {group: ROOT / f"artifacts/cache/{group}_weather_train.parquet" for group in TARGET_COLS}
FIT_START = pd.Timestamp("2024-02-18 00:00:00")
FIT_END = pd.Timestamp("2024-06-30 23:00:00")
APPLY_START = pd.Timestamp("2024-07-01 00:00:00")
MODEL_APPLY_END = pd.Timestamp("2024-12-31 23:00:00")
APPLY_END = pd.Timestamp("2025-01-01 00:00:00")
FIT_PREFIX_DATA_ROWS = 21_887
SCORE_DATA_ROWS = 4_417
EXPECTED_LABEL_BYTES = 1_138_967
HEIGHTS = (10, 80, 120)
CANDIDATES = ("A_day2_w025", "B_day1_hours01_13_else_day2_w025")
BASELINES = {"primary_corrected_v3": PRIMARY_PATH, "interaction_recent_v4": INTERACTION_PATH}
EXTENDED_COLUMNS = tuple(
    name
    for height in HEIGHTS
    for name in (
        f"icon__ws{height}_ms",
        f"icon__u{height}_ms",
        f"icon__v{height}_ms",
        f"icon__ws{height}_minus_cross_hub_ws_mean",
    )
)


def config() -> tuple[dict[str, Any], dict[str, Any]]:
    expected = SIDECAR_PATH.read_text(encoding="utf-8").split()[0]
    observed = sha256(CONFIG_PATH)
    if expected != observed:
        raise RuntimeError("preregistration sidecar mismatch")
    base_expected = BASE_SIDECAR_PATH.read_text(encoding="utf-8").split()[0]
    base_observed = sha256(BASE_CONFIG_PATH)
    if base_expected != base_observed:
        raise RuntimeError("base preregistration sidecar mismatch")
    phase_expected = PHASE_SIDECAR_PATH.read_text(encoding="utf-8").split()[0]
    phase_observed = sha256(PHASE_CONFIG_PATH)
    if phase_expected != phase_observed:
        raise RuntimeError("phase preregistration sidecar mismatch")
    amendment = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    phase = json.loads(PHASE_CONFIG_PATH.read_text(encoding="utf-8"))
    base = json.loads(BASE_CONFIG_PATH.read_text(encoding="utf-8"))
    if amendment["status"] != "frozen_before_v3_label_read_candidate_fit_prediction_or_score":
        raise RuntimeError("v3 preregistration is not frozen")
    if amendment["base_preregister"]["sha256"] != phase_observed:
        raise RuntimeError("v3 does not bind the frozen v2 phase contract")
    if phase["base_preregister"]["sha256"] != base_observed:
        raise RuntimeError("v2 does not bind the frozen v1 candidate contract")
    if amendment["reaffirmed"]["candidate_ids_fixed_order"] != list(CANDIDATES):
        raise RuntimeError("v3 candidate contract differs")
    return base, amendment


def acquire_heavy_guard(path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    owner = {
        "schema_version": 1,
        "pid": os.getpid(),
        "token": secrets.token_hex(16),
        "experiment_id": "icon_global_paired_increment_2024_forward_v3",
        "stage": "ICON_stage1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    for _ in range(2):
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                current = json.loads(path.read_text(encoding="utf-8"))
                current_pid = int(current["pid"])
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"heavy CPU guard exists but is unreadable: {path}") from exc
            if pid_is_alive(current_pid):
                raise RuntimeError(f"heavy CPU guard is held by live PID {current_pid}: {current}")
            path.unlink()
            continue
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(owner, stream, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
        return owner
    raise RuntimeError("could not acquire heavy CPU guard")


def release_heavy_guard(path: Path, owner: Mapping[str, Any]) -> None:
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        return
    if int(current.get("pid", -1)) == int(owner["pid"]) and current.get("token") == owner.get("token"):
        path.unlink()


def _decode_header(line: bytes) -> list[str]:
    return next(csv.reader([line.decode("utf-8-sig").rstrip("\r\n")]))


def read_fit_labels_only() -> tuple[pd.DataFrame, int, dict[str, Any], bytes]:
    if LABEL_PATH.stat().st_size != EXPECTED_LABEL_BYTES:
        raise ValueError("label file size differs before bounded H1 read")
    prefix_digest = hashlib.sha256()
    fit_times: list[pd.Timestamp] = []
    fit_values: list[list[float]] = []
    first_time: pd.Timestamp | None = None
    last_time: pd.Timestamp | None = None
    with LABEL_PATH.open("rb", buffering=0) as stream:
        header = stream.readline()
        if not header:
            raise ValueError("label header missing")
        prefix_digest.update(header)
        if _decode_header(header) != ["kst_dtm", *TARGET_COLS]:
            raise ValueError("label header differs")
        for row_number in range(FIT_PREFIX_DATA_ROWS):
            line = stream.readline()
            if not line:
                raise ValueError(f"label file ended in H1 prefix at row {row_number}")
            prefix_digest.update(line)
            timestamp = pd.Timestamp(line.split(b",", 1)[0].decode("ascii"))
            first_time = timestamp if first_time is None else first_time
            last_time = timestamp
            if timestamp >= FIT_START:
                fields = next(csv.reader([line.decode("utf-8").rstrip("\r\n")]))
                if len(fields) != 4:
                    raise ValueError("fit label row width differs")
                fit_times.append(timestamp)
                fit_values.append([float(value) if value else np.nan for value in fields[1:]])
        score_offset = stream.tell()
    expected_index = pd.date_range(FIT_START, FIT_END, freq="h", name="forecast_kst_dtm")
    frame = pd.DataFrame(fit_values, index=pd.DatetimeIndex(fit_times, name="forecast_kst_dtm"), columns=TARGET_COLS)
    if first_time != pd.Timestamp("2022-01-01 01:00") or last_time != FIT_END:
        raise ValueError("H1 prefix timestamp boundary differs")
    if not frame.index.equals(expected_index):
        raise ValueError("materialized fit-label index differs")
    record = {
        "path": str(LABEL_PATH),
        "file_bytes_from_metadata": LABEL_PATH.stat().st_size,
        "physical_data_rows_read": FIT_PREFIX_DATA_ROWS,
        "physical_prefix_bytes": score_offset,
        "physical_prefix_sha256_including_header": prefix_digest.hexdigest(),
        "first_timestamp": str(first_time),
        "last_timestamp": str(last_time),
        "materialized_value_window": [str(frame.index[0]), str(frame.index[-1])],
        "materialized_value_rows": len(frame),
        "materialized_value_cells": int(frame.size),
        "prehistory_value_cells_materialized": 0,
        "H2_bytes_or_value_cells_read": 0,
        "score_segment_start_byte_offset": score_offset,
    }
    return frame, score_offset, record, header


def read_score_labels(score_offset: int, header: bytes) -> tuple[pd.DataFrame, dict[str, Any]]:
    segment_digest = hashlib.sha256()
    times: list[pd.Timestamp] = []
    values: list[list[float]] = []
    with LABEL_PATH.open("rb", buffering=0) as stream:
        stream.seek(score_offset)
        if stream.tell() != score_offset:
            raise ValueError("score label seek failed")
        for row_number in range(SCORE_DATA_ROWS):
            line = stream.readline()
            if not line:
                raise ValueError(f"label file ended in score segment at row {row_number}")
            segment_digest.update(line)
            fields = next(csv.reader([line.decode("utf-8").rstrip("\r\n")]))
            if len(fields) != 4:
                raise ValueError("score label row width differs")
            times.append(pd.Timestamp(fields[0]))
            values.append([float(value) if value else np.nan for value in fields[1:]])
        end_offset = stream.tell()
    expected_index = pd.date_range(APPLY_START, APPLY_END, freq="h", name="forecast_kst_dtm")
    frame = pd.DataFrame(values, index=pd.DatetimeIndex(times, name="forecast_kst_dtm"), columns=TARGET_COLS)
    if not frame.index.equals(expected_index) or end_offset != EXPECTED_LABEL_BYTES:
        raise ValueError("bounded score-label segment differs")
    record = {
        "path": str(LABEL_PATH),
        "header_sha256": hashlib.sha256(header).hexdigest(),
        "seek_start_byte_offset": score_offset,
        "physical_data_rows_read": SCORE_DATA_ROWS,
        "segment_bytes": end_offset - score_offset,
        "segment_sha256": segment_digest.hexdigest(),
        "first_timestamp": str(frame.index[0]),
        "last_timestamp": str(frame.index[-1]),
        "materialized_value_cells": int(frame.size),
        "bytes_or_rows_after_segment_read": 0,
    }
    return frame, record


def select_external_features(frame: pd.DataFrame, candidate: str) -> pd.DataFrame:
    if candidate not in CANDIDATES:
        raise ValueError(f"unknown candidate {candidate!r}")
    if candidate == CANDIDATES[0]:
        use_day1 = np.zeros(len(frame), dtype=bool)
    else:
        use_day1 = frame.index.hour.astype(int).isin(range(1, 14))
    result = pd.DataFrame(index=frame.index)
    for height in HEIGHTS:
        speed = np.where(
            use_day1,
            frame[f"wind_speed_{height}m_previous_day1"].to_numpy(dtype=np.float64),
            frame[f"wind_speed_{height}m_previous_day2"].to_numpy(dtype=np.float64),
        )
        direction = np.where(
            use_day1,
            frame[f"wind_direction_{height}m_previous_day1"].to_numpy(dtype=np.float64),
            frame[f"wind_direction_{height}m_previous_day2"].to_numpy(dtype=np.float64),
        )
        radians = np.deg2rad(direction)
        result[f"icon__ws{height}_ms"] = speed
        result[f"icon__u{height}_ms"] = -speed * np.sin(radians)
        result[f"icon__v{height}_ms"] = -speed * np.cos(radians)
    return result.astype(np.float32)


def add_disagreements(external: pd.DataFrame, control: pd.DataFrame) -> pd.DataFrame:
    if not external.index.equals(control.index):
        raise ValueError("external/control indexes differ")
    result = pd.DataFrame(index=external.index)
    reference = control["cross__hub_ws_mean"].astype(np.float64)
    for height in HEIGHTS:
        for component in ("ws", "u", "v"):
            column = f"icon__{component}{height}_ms"
            result[column] = external[column]
        result[f"icon__ws{height}_minus_cross_hub_ws_mean"] = (
            external[f"icon__ws{height}_ms"].astype(np.float64) - reference
        ).astype(np.float32)
    if tuple(result.columns) != EXTENDED_COLUMNS or not np.isfinite(result.to_numpy()).all():
        raise ValueError("registered ICON features differ or contain non-finite values")
    return result


def validate_fixed_inputs(payload: Mapping[str, Any]) -> None:
    expected = {
        EXTERNAL_PATH: payload["official_source_contract"]["normalized_data"]["sha256"],
        SOURCE_MANIFEST_PATH: payload["official_source_contract"]["availability_manifest"]["sha256"],
        PRIMARY_PATH: payload["fixed_baselines"][0]["sha256"],
        INTERACTION_PATH: payload["fixed_baselines"][1]["sha256"],
        ROOT / "src/metric.py": payload["bound_inputs"]["metric_source_sha256"],
        ROOT / "scripts/download_openmeteo_icon_global_previous_runs.py": payload["bound_inputs"]["download_source_sha256"],
    }
    expected.update({CACHE_PATHS[group]: value for group, value in payload["bound_inputs"]["weather_cache_sha256"].items()})
    mismatches = {str(path): (wanted, sha256(path)) for path, wanted in expected.items() if sha256(path) != wanted}
    if mismatches:
        raise RuntimeError(f"fixed input hash mismatch: {mismatches}")


def read_baseline(path: Path, index: pd.DatetimeIndex) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    frame = frame.loc[index, list(TARGET_COLS)].astype(np.float64)
    if not frame.index.equals(index) or not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"baseline differs: {path}")
    return frame


def source_lock(config_hash: str, out_dir: Path) -> Path:
    sources = [
        CONFIG_PATH, SIDECAR_PATH, PHASE_CONFIG_PATH, PHASE_SIDECAR_PATH,
        BASE_CONFIG_PATH, BASE_SIDECAR_PATH,
        ROOT / "scripts/run_icon_global_paired_increment.py",
        ROOT / "scripts/run_ecmwf_ifs025_paired_increment.py",
        ROOT / "scripts/download_openmeteo_icon_global_previous_runs.py",
        ROOT / "tests/test_icon_global_paired_increment.py",
        ROOT / "src/__init__.py", ROOT / "src/metric.py",
    ]
    inputs = [EXTERNAL_PATH, SOURCE_MANIFEST_PATH, PRIMARY_PATH, INTERACTION_PATH, *CACHE_PATHS.values()]
    payload = {
        "schema_version": 1,
        "status": "locked_before_any_label_read_candidate_fit_prediction_or_score",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "preregister_sha256": config_hash,
        "sources": [file_record(path) for path in sources],
        "inputs": [file_record(path) for path in inputs],
        "bounded_label_source_metadata_only": {
            "path": str(LABEL_PATH),
            "expected_bytes": EXPECTED_LABEL_BYTES,
            "observed_bytes": LABEL_PATH.stat().st_size,
            "content_bytes_read_for_this_lock": 0,
            "content_hash_deferred_to_bounded_phase_locks": True
        },
        "recursive_local_import_resolution": {
            "runner_imports": ["scripts.run_ecmwf_ifs025_paired_increment", "src.metric"],
            "resolved": ["scripts/run_ecmwf_ifs025_paired_increment.py", "src/__init__.py", "src/metric.py"],
            "unresolved": [],
        },
        "physical_access": {
            "maximum_external_calendar_year_requested_or_parsed": 2024,
            "2025_external_forecast_value_cells_read": 0,
            "operating_2024_terminal_target_rows_bound": 1,
            "feedback_files_read": 0,
            "candidate_fits_predictions_scores": 0,
        },
    }
    path = out_dir / "source_lock_before_any_label_read.json"
    atomic_json(payload, path)
    return path


def fit_label_lock(record: Mapping[str, Any], labels: pd.DataFrame, source_lock_path: Path, out_dir: Path) -> Path:
    values = labels.to_numpy(dtype=np.float64)
    payload = {
        "schema_version": 1,
        "status": "locked_after_H1_only_label_read_before_candidate_fit_prediction_or_score",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_lock": file_record(source_lock_path),
        "fit_label_access": dict(record),
        "fit_index_sha256": hashlib.sha256(labels.index.asi8.tobytes()).hexdigest(),
        "fit_values_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "H2_label_bytes_or_value_cells_read": 0,
        "candidate_fits_predictions_or_scores": 0,
    }
    path = out_dir / "fit_label_lock_before_candidate_fit.json"
    atomic_json(payload, path)
    return path


def candidate_lock(
    model_records: list[dict[str, Any]],
    increment_path: Path,
    fit_lock_path: Path,
    increments: pd.DataFrame,
    out_dir: Path,
) -> Path:
    if not (increments.loc[APPLY_END] == 0.0).all():
        raise AssertionError("candidate lock requires exact terminal zero for all candidates/groups")
    payload = {
        "schema_version": 1,
        "status": "candidate_models_and_increments_locked_before_H2_label_read_or_score",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "fit_label_lock": file_record(fit_lock_path),
        "models": model_records,
        "increments": file_record(increment_path),
        "increment_index_sha256": hashlib.sha256(increments.index.asi8.tobytes()).hexdigest(),
        "increment_values_sha256": hashlib.sha256(increments.to_numpy(dtype=np.float64).tobytes()).hexdigest(),
        "terminal_increment_cf_all_candidates_groups": 0.0,
        "H2_label_bytes_or_value_cells_read": 0,
        "metric_calls": 0,
        "retune_retry_or_fallback": False,
    }
    path = out_dir / "candidate_lock_before_h2_score.json"
    atomic_json(payload, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {out_dir}")
    out_dir.mkdir(parents=True)

    base_payload, amendment = config()
    config_hash = sha256(CONFIG_PATH)
    validate_fixed_inputs(base_payload)
    closure_path = source_lock(config_hash, out_dir)
    print(f"source closure locked: {sha256(closure_path)}", flush=True)
    guard_owner = acquire_heavy_guard(HEAVY_GUARD_PATH)
    atexit.register(release_heavy_guard, HEAVY_GUARD_PATH, guard_owner)
    print(f"heavy CPU guard acquired by PID {guard_owner['pid']} token={guard_owner['token']}", flush=True)

    fit_labels, score_offset, fit_label_record, label_header = read_fit_labels_only()
    fit_lock_path = fit_label_lock(fit_label_record, fit_labels, closure_path, out_dir)
    print(f"H1-only label lock: {sha256(fit_lock_path)}; H2 bytes read=0", flush=True)
    fit_index = pd.date_range(FIT_START, FIT_END, freq="h", name="forecast_kst_dtm")
    model_apply_index = pd.date_range(APPLY_START, MODEL_APPLY_END, freq="h", name="forecast_kst_dtm")
    apply_index = pd.date_range(APPLY_START, APPLY_END, freq="h", name="forecast_kst_dtm")
    external_all = pd.read_parquet(EXTERNAL_PATH)
    external_all["time"] = pd.to_datetime(external_all["time"], errors="raise")
    increments = pd.DataFrame(index=apply_index)
    model_records: list[dict[str, Any]] = []
    fit_records: dict[str, Any] = {}

    for group in TARGET_COLS:
        control_all = pd.read_parquet(CACHE_PATHS[group])
        control_all.index = pd.DatetimeIndex(control_all.index, name="forecast_kst_dtm")
        if control_all.shape[1] != 612 or not control_all.columns.is_unique:
            raise ValueError(f"{group}: canonical control schema differs")
        control_fit = control_all.loc[fit_index].astype(np.float32)
        control_apply = control_all.loc[model_apply_index].astype(np.float32)
        if not np.isfinite(control_fit.to_numpy()).all() or not np.isfinite(control_apply.to_numpy()).all():
            raise ValueError(f"{group}: control features contain non-finite values")
        external = external_all.loc[external_all["group"] == group].drop(columns="group").set_index("time")
        external.index.name = "forecast_kst_dtm"
        target = fit_labels.loc[fit_index, group].astype(np.float64)
        eligible = target.notna() & (target >= 0.10 * CAPACITY_KWH[group])
        fit_rows = fit_index[eligible.to_numpy()]
        y = (target.loc[fit_rows] / CAPACITY_KWH[group]).to_numpy(dtype=np.float64)
        params = dict(base_payload["paired_model"]["parameters"])
        control_model = LGBMRegressor(**params)
        control_model.fit(control_fit.loc[fit_rows], y)
        control_cf = np.clip(control_model.predict(control_apply), 0.0, 1.02)
        control_path = out_dir / f"models/{group}__control.joblib"
        control_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(control_model, control_path, compress=3)
        model_records.append(file_record(control_path))
        fit_records[group] = {
            "fit_rows": len(fit_rows),
            "fit_index_sha256": hashlib.sha256(fit_rows.asi8.tobytes()).hexdigest(),
            "target_sha256": hashlib.sha256(y.tobytes()).hexdigest(),
            "control_feature_count": 612,
            "extended_feature_count": 624,
        }
        for candidate in CANDIDATES:
            selected = select_external_features(external, candidate)
            ext_fit = add_disagreements(selected.loc[fit_index], control_fit)
            ext_apply = add_disagreements(selected.loc[model_apply_index], control_apply)
            extended_fit = pd.concat([control_fit, ext_fit], axis=1)
            extended_apply = pd.concat([control_apply, ext_apply], axis=1)
            if extended_fit.shape[1] != 624 or tuple(extended_fit.columns[-12:]) != EXTENDED_COLUMNS:
                raise ValueError("extended model schema differs")
            model = LGBMRegressor(**params)
            model.fit(extended_fit.loc[fit_rows], y)
            delta = np.clip(model.predict(extended_apply), 0.0, 1.02) - control_cf
            registered = pd.Series(0.0, index=apply_index, dtype=np.float64)
            registered.loc[model_apply_index] = delta
            if registered.loc[APPLY_END] != 0.0:
                raise AssertionError("terminal paired increment must be exact zero")
            increments[f"{candidate}__{group}"] = registered
            model_path = out_dir / f"models/{group}__{candidate}.joblib"
            joblib.dump(model, model_path, compress=3)
            model_records.append(file_record(model_path))
        print(f"fit {group}: {len(fit_rows)} eligible rows", flush=True)

    increment_path = out_dir / "predictions/candidate_increments.parquet"
    atomic_parquet(increments, increment_path)
    candidate_lock_path = candidate_lock(model_records, increment_path, fit_lock_path, increments, out_dir)
    print(f"candidate lock before H2 labels: {sha256(candidate_lock_path)}", flush=True)

    labels_apply, score_label_record = read_score_labels(score_offset, label_header)
    score_label_lock_path = out_dir / "score_label_access_after_candidate_lock.json"
    atomic_json({
        "schema_version": 1,
        "status": "bounded_H2_plus_terminal_labels_read_after_candidate_lock_before_metric",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_lock": file_record(candidate_lock_path),
        "score_label_access": score_label_record,
        "2025_external_forecast_value_cells_read": 0,
        "metric_calls_before_this_lock": 0,
    }, score_label_lock_path)
    print(f"bounded score-label access lock: {sha256(score_label_lock_path)}", flush=True)
    baselines = {name: read_baseline(path, apply_index) for name, path in BASELINES.items()}
    comparisons: dict[str, dict[str, Any]] = {candidate: {} for candidate in CANDIDATES}
    prediction_records = [file_record(increment_path)]
    for candidate in CANDIDATES:
        for baseline_name, baseline in baselines.items():
            predicted = baseline.copy()
            for group in TARGET_COLS:
                predicted[group] = apply_paired_increment(
                    baseline[group], increments[f"{candidate}__{group}"],
                    capacity_kwh=CAPACITY_KWH[group], weight=0.25,
                )
                terminal = APPLY_END
                if np.float64(predicted.loc[terminal, group]).tobytes() != np.float64(baseline.loc[terminal, group]).tobytes():
                    raise AssertionError("terminal baseline bit identity differs")
            prediction_path = out_dir / f"predictions/{candidate}__{baseline_name}.parquet"
            atomic_parquet(predicted, prediction_path)
            prediction_records.append(file_record(prediction_path))
            comparisons[candidate][baseline_name] = comparison_payload(labels_apply, baseline, predicted)

    gates = {candidate: gate_summary(comparisons[candidate]) for candidate in CANDIDATES}
    passing = [candidate for candidate in CANDIDATES if gates[candidate]["dual_gate_pass"]]
    winner = max(
        passing,
        key=lambda candidate: (
            gates[candidate]["minimum_group_slice_total_score_delta"],
            gates[candidate]["mean_group_slice_total_score_delta"],
            -CANDIDATES.index(candidate),
        ),
        default=None,
    )
    results = {
        "schema_version": 1,
        "experiment_id": amendment["experiment_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "preregister_sha256": config_hash,
        "fit_window": [str(FIT_START), str(FIT_END)],
        "apply_window": [str(APPLY_START), str(APPLY_END)],
        "fit_records": fit_records,
        "increment_diagnostics": {
            column: {"mean_cf": float(increments[column].mean()), "std_cf": float(increments[column].std()),
                     "min_cf": float(increments[column].min()), "max_cf": float(increments[column].max())}
            for column in increments
        },
        "comparisons": comparisons,
        "gates": gates,
        "winner": winner,
        "status": "promoted_for_separate_prescore_lock_only" if winner else "rejected_on_2024_forward_dual_gate",
        "physical_access": {
            "maximum_external_calendar_year_requested_or_parsed": 2024,
            "source_lock_before_any_label_read_sha256": sha256(closure_path),
            "fit_label_lock_before_candidate_fit_sha256": sha256(fit_lock_path),
            "candidate_lock_before_H2_label_read_sha256": sha256(candidate_lock_path),
            "score_label_access_after_candidate_lock_sha256": sha256(score_label_lock_path),
            "fit_prefix_physical_data_rows": FIT_PREFIX_DATA_ROWS,
            "score_segment_physical_data_rows": SCORE_DATA_ROWS,
            "operating_2024_terminal_target_row_scored": True,
            "terminal_paired_increment_cf": 0.0,
            "2025_external_forecast_value_cells_read": 0,
            "feedback_files_read": 0,
            "submission_csv_created": False,
        },
    }
    results_path = out_dir / "stage_results.json"
    atomic_json(results, results_path)
    decision_path = out_dir / ("promotion_lock.json" if winner else "rejection.json")
    atomic_json({
        "status": results["status"], "winner": winner, "preregister_sha256": config_hash,
        "source_lock_sha256": sha256(closure_path), "stage_results_sha256": sha256(results_path),
        "2025_access_authorized_by_this_process": False,
    }, decision_path)
    outputs = [closure_path, fit_lock_path, candidate_lock_path, score_label_lock_path,
               *[Path(record["path"]) for record in model_records],
               *[Path(record["path"]) for record in prediction_records], results_path, decision_path]
    manifest_path = out_dir / "manifest.json"
    manifest = {
        "schema_version": 1, "experiment_id": amendment["experiment_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "preregister": file_record(CONFIG_PATH), "base_preregister": file_record(BASE_CONFIG_PATH),
        "source_lock": file_record(closure_path),
        "phase_locks": [file_record(path) for path in [fit_lock_path, candidate_lock_path, score_label_lock_path]],
        "inputs": [file_record(path) for path in [EXTERNAL_PATH, SOURCE_MANIFEST_PATH, PRIMARY_PATH, INTERACTION_PATH, *CACHE_PATHS.values()]],
        "outputs_excluding_this_manifest_and_terminal_sidecar": [file_record(path) for path in outputs],
        "unlisted_files_before_manifest": sorted(str(path.relative_to(out_dir)) for path in out_dir.rglob("*") if path.is_file() and path not in outputs),
        "physical_access": results["physical_access"],
    }
    if manifest["unlisted_files_before_manifest"]:
        raise RuntimeError(f"unlisted output files: {manifest['unlisted_files_before_manifest']}")
    atomic_json(manifest, manifest_path)
    sidecar_path = out_dir / "manifest.sha256"
    sidecar_path.write_text(f"{sha256(manifest_path)}  manifest.json\n", encoding="ascii")
    print(f"status={results['status']} winner={winner}", flush=True)
    print(f"results={results_path} sha256={sha256(results_path)}", flush=True)
    print(f"manifest={manifest_path} sha256={sha256(manifest_path)}", flush=True)
    release_heavy_guard(HEAVY_GUARD_PATH, guard_owner)


if __name__ == "__main__":
    main()

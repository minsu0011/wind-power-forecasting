from __future__ import annotations

import argparse
import atexit
import hashlib
import io
import json
import os
import ctypes
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

from src.metric import CAPACITY_KWH, TARGET_COLS, group_metrics, score_details


CONFIG_PATH = ROOT / "configs/ecmwf_ifs025_paired_increment_preregister_v2.json"
SIDECAR_PATH = CONFIG_PATH.with_suffix(".sha256")
EXTERNAL_PATH = ROOT / "artifacts/external/openmeteo_ecmwf_ifs025_previous_runs_v1/ecmwf_ifs025_group_centroids_2024.parquet"
SOURCE_MANIFEST_PATH = EXTERNAL_PATH.with_name("source_manifest.json")
LABEL_PATH = Path(r"data/local/open/train/train_labels.csv")
PRIMARY_PATH = ROOT / "artifacts/oof/gate2024_locked_v3_cf_fix.parquet"
INTERACTION_PATH = ROOT / "artifacts/oof/gate2024_recent_v4_cf_fix_calibration_fit.parquet"
OUTPUT_DIR = ROOT / "artifacts/postgate/ecmwf_ifs025_paired_increment_2024_forward_v2"
CACHE_PATHS = {group: ROOT / f"artifacts/cache/{group}_weather_train.parquet" for group in TARGET_COLS}
FIT_START = pd.Timestamp("2024-03-09 00:00:00")
FIT_END = pd.Timestamp("2024-06-30 23:00:00")
APPLY_START = pd.Timestamp("2024-07-01 00:00:00")
MODEL_APPLY_END = pd.Timestamp("2024-12-31 23:00:00")
APPLY_END = pd.Timestamp("2025-01-01 00:00:00")
LABEL_PREFIX_DATA_ROWS = 26_304
HEAVY_GUARD_PATH = ROOT / "artifacts/locks/heavy_cpu_fit.pid.json"
EXTENDED_COLUMNS = (
    "ecmwf__ws100_ms",
    "ecmwf__u100_ms",
    "ecmwf__v100_ms",
    "ecmwf__ws100_minus_cross_hub_ws_mean",
)
CANDIDATES = (
    "A_day2_w025",
    "B_day1_hours01_13_else_day2_w025",
)
BASELINES = {
    "primary_corrected_v3": PRIMARY_PATH,
    "interaction_recent_v4": INTERACTION_PATH,
}
SLICES = {
    "H2": (pd.Timestamp("2024-07-01 00:00"), pd.Timestamp("2025-01-01 00:00")),
    "Q3": (pd.Timestamp("2024-07-01 00:00"), pd.Timestamp("2024-09-30 23:00")),
    "Q4": (pd.Timestamp("2024-10-01 00:00"), pd.Timestamp("2025-01-01 00:00")),
    "Jul": (pd.Timestamp("2024-07-01 00:00"), pd.Timestamp("2024-07-31 23:00")),
    "Aug": (pd.Timestamp("2024-08-01 00:00"), pd.Timestamp("2024-08-31 23:00")),
    "Sep": (pd.Timestamp("2024-09-01 00:00"), pd.Timestamp("2024-09-30 23:00")),
    "OctNov": (pd.Timestamp("2024-10-01 00:00"), pd.Timestamp("2024-11-30 23:00")),
    "Dec": (pd.Timestamp("2024-12-01 00:00"), pd.Timestamp("2025-01-01 00:00")),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    frame.to_parquet(temporary, engine="pyarrow", compression="zstd", index=True)
    os.replace(temporary, path)


def pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    process_query_limited_information = 0x1000
    handle = ctypes.windll.kernel32.OpenProcess(process_query_limited_information, False, int(pid))
    if not handle:
        return False
    ctypes.windll.kernel32.CloseHandle(handle)
    return True


def acquire_heavy_guard(path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    owner = {
        "pid": os.getpid(),
        "experiment_id": "ecmwf_ifs025_paired_increment_2024_forward_v2",
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
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError):
        return
    if int(current.get("pid", -1)) == int(owner["pid"]):
        path.unlink()


def _config() -> dict[str, Any]:
    expected = SIDECAR_PATH.read_text(encoding="utf-8").split()[0]
    observed = sha256(CONFIG_PATH)
    if expected != observed:
        raise RuntimeError("preregistration sidecar mismatch")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["status"] != "frozen_before_any_label_join_candidate_fit_prediction_or_score":
        raise RuntimeError("preregistration is not frozen")
    return config


def bounded_label_prefix() -> tuple[bytes, dict[str, Any]]:
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    with LABEL_PATH.open("rb", buffering=0) as stream:
        for line_number in range(LABEL_PREFIX_DATA_ROWS + 1):
            line = stream.readline()
            if not line:
                raise ValueError(f"label file ended at physical line {line_number}")
            chunks.append(line)
            digest.update(line)
    payload = b"".join(chunks)
    return payload, {
        "path": str(LABEL_PATH),
        "physical_lines_read": LABEL_PREFIX_DATA_ROWS + 1,
        "data_rows": LABEL_PREFIX_DATA_ROWS,
        "physical_prefix_bytes": len(payload),
        "physical_prefix_sha256": digest.hexdigest(),
        "later_bytes_read_or_parsed": False,
    }


def read_labels(payload: bytes) -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(payload), encoding="utf-8-sig")
    if tuple(frame.columns) != ("kst_dtm", *TARGET_COLS) or len(frame) != LABEL_PREFIX_DATA_ROWS:
        raise ValueError("bounded labels differ in schema or row count")
    index = pd.DatetimeIndex(pd.to_datetime(frame.pop("kst_dtm"), errors="raise"), name="forecast_kst_dtm")
    frame.index = index
    frame = frame.astype(np.float64)
    if frame.index[0] != pd.Timestamp("2022-01-01 01:00") or frame.index[-1] != APPLY_END:
        raise ValueError("bounded labels do not end at the operating-2024 terminal target")
    if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError("label index differs")
    return frame


def model_parameters(config: Mapping[str, Any]) -> dict[str, Any]:
    return dict(config["paired_model"]["parameters"])


def select_external_features(frame: pd.DataFrame, candidate: str) -> pd.DataFrame:
    if candidate not in CANDIDATES:
        raise ValueError(f"unknown candidate {candidate!r}")
    if candidate == CANDIDATES[0]:
        suffix = np.full(len(frame), "day2", dtype=object)
    else:
        suffix = np.where(frame.index.hour.astype(int).isin(range(1, 14)), "day1", "day2")
    day1 = suffix == "day1"
    speed = np.where(
        day1,
        frame["wind_speed_100m_previous_day1"].to_numpy(dtype=np.float64),
        frame["wind_speed_100m_previous_day2"].to_numpy(dtype=np.float64),
    )
    direction = np.where(
        day1,
        frame["wind_direction_100m_previous_day1"].to_numpy(dtype=np.float64),
        frame["wind_direction_100m_previous_day2"].to_numpy(dtype=np.float64),
    )
    radians = np.deg2rad(direction)
    result = pd.DataFrame(index=frame.index)
    result["ecmwf__ws100_ms"] = speed
    result["ecmwf__u100_ms"] = -speed * np.sin(radians)
    result["ecmwf__v100_ms"] = -speed * np.cos(radians)
    return result.astype(np.float32)


def add_disagreement(external: pd.DataFrame, control: pd.DataFrame) -> pd.DataFrame:
    if not external.index.equals(control.index):
        raise ValueError("external/control indexes differ")
    result = external.copy()
    result["ecmwf__ws100_minus_cross_hub_ws_mean"] = (
        result["ecmwf__ws100_ms"].astype(np.float64)
        - control["cross__hub_ws_mean"].astype(np.float64)
    ).astype(np.float32)
    if tuple(result.columns) != EXTENDED_COLUMNS or not np.isfinite(result.to_numpy()).all():
        raise ValueError("extended features differ or contain non-finite values")
    return result


def apply_paired_increment(
    baseline_kwh: pd.Series,
    increment_cf: pd.Series,
    *,
    capacity_kwh: float,
    weight: float = 0.25,
) -> pd.Series:
    if not baseline_kwh.index.equals(increment_cf.index):
        raise ValueError("baseline/increment indexes differ")
    values = np.clip(
        baseline_kwh.to_numpy(dtype=np.float64)
        + float(weight) * float(capacity_kwh) * increment_cf.to_numpy(dtype=np.float64),
        0.0,
        1.02 * float(capacity_kwh),
    )
    return pd.Series(values, index=baseline_kwh.index, name=baseline_kwh.name)


def metric_dict(actual: pd.Series, prediction: pd.Series, group: str) -> dict[str, Any]:
    values = group_metrics(actual, prediction, CAPACITY_KWH[group], group_name=group)
    payload = values.as_dict()
    payload["total_score"] = 0.5 * payload["one_minus_nmae"] + 0.5 * payload["ficr"]
    return payload


def comparison_payload(
    labels: pd.DataFrame,
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
) -> dict[str, Any]:
    result: dict[str, Any] = {"by_group": {}, "mixed": {}}
    for group in TARGET_COLS:
        result["by_group"][group] = {}
        for name, (start, end) in SLICES.items():
            idx = labels.index[(labels.index >= start) & (labels.index <= end)]
            base = metric_dict(labels.loc[idx, group], baseline.loc[idx, group], group)
            cand = metric_dict(labels.loc[idx, group], candidate.loc[idx, group], group)
            result["by_group"][group][name] = {
                "baseline": base,
                "candidate": cand,
                "delta": {
                    "total_score": cand["total_score"] - base["total_score"],
                    "one_minus_nmae": cand["one_minus_nmae"] - base["one_minus_nmae"],
                    "ficr": cand["ficr"] - base["ficr"],
                },
            }
    for name, (start, end) in SLICES.items():
        idx = labels.index[(labels.index >= start) & (labels.index <= end)]
        base = score_details(labels.loc[idx], baseline.loc[idx]).as_dict()
        cand = score_details(labels.loc[idx], candidate.loc[idx]).as_dict()
        result["mixed"][name] = {
            "baseline": base,
            "candidate": cand,
            "delta": {
                "total_score": cand["total_score"] - base["total_score"],
                "one_minus_nmae": cand["one_minus_nmae"] - base["one_minus_nmae"],
                "ficr": cand["ficr"] - base["ficr"],
            },
        }
    return result


def gate_summary(comparisons: Mapping[str, Any]) -> dict[str, Any]:
    group_deltas: list[float] = []
    mixed_deltas: list[float] = []
    baseline_pass: dict[str, bool] = {}
    for baseline, payload in comparisons.items():
        local_group = [
            payload["by_group"][group][slice_name]["delta"]["total_score"]
            for group in TARGET_COLS
            for slice_name in SLICES
        ]
        local_mixed = [payload["mixed"][slice_name]["delta"]["total_score"] for slice_name in SLICES]
        h2 = payload["mixed"]["H2"]["delta"]
        baseline_pass[baseline] = bool(
            all(value > 0.0 for value in local_group)
            and all(value > 0.0 for value in local_mixed)
            and h2["one_minus_nmae"] >= 0.0
            and h2["ficr"] >= 0.0
        )
        group_deltas.extend(local_group)
        mixed_deltas.extend(local_mixed)
    return {
        "pass_by_baseline": baseline_pass,
        "dual_gate_pass": bool(all(baseline_pass.values())),
        "group_comparison_count": len(group_deltas),
        "mixed_comparison_count": len(mixed_deltas),
        "minimum_group_slice_total_score_delta": min(group_deltas),
        "mean_group_slice_total_score_delta": float(np.mean(group_deltas)),
        "minimum_mixed_slice_total_score_delta": min(mixed_deltas),
    }


def source_lock(config_hash: str, label_record: Mapping[str, Any], out_dir: Path) -> Path:
    sources = [
        CONFIG_PATH,
        SIDECAR_PATH,
        ROOT / "scripts/run_ecmwf_ifs025_paired_increment.py",
        ROOT / "scripts/download_openmeteo_ecmwf_previous_runs.py",
        ROOT / "tests/test_ecmwf_ifs025_paired_increment.py",
        ROOT / "src/__init__.py",
        ROOT / "src/metric.py",
    ]
    inputs = [EXTERNAL_PATH, SOURCE_MANIFEST_PATH, PRIMARY_PATH, INTERACTION_PATH, *CACHE_PATHS.values()]
    payload = {
        "schema_version": 1,
        "status": "locked_before_label_parse_join_candidate_fit_prediction_or_score",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "preregister_sha256": config_hash,
        "sources": [file_record(path) for path in sources],
        "inputs": [file_record(path) for path in inputs],
        "bounded_label_source": dict(label_record),
        "recursive_local_import_resolution": {
            "runner_import": "src.metric",
            "resolved": ["src/__init__.py", "src/metric.py"],
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
    path = out_dir / "source_lock_before_fit.json"
    atomic_json(payload, path)
    return path


def validate_fixed_inputs(config: Mapping[str, Any]) -> None:
    expected = {
        EXTERNAL_PATH: config["official_source_contract"]["normalized_data"]["sha256"],
        SOURCE_MANIFEST_PATH: config["official_source_contract"]["availability_manifest"]["sha256"],
        PRIMARY_PATH: config["fixed_baselines"][0]["sha256"],
        INTERACTION_PATH: config["fixed_baselines"][1]["sha256"],
        ROOT / "src/metric.py": config["bound_inputs"]["metric_source_sha256"],
        ROOT / "scripts/download_openmeteo_ecmwf_previous_runs.py": config["bound_inputs"]["download_source_sha256"],
    }
    expected.update({CACHE_PATHS[group]: value for group, value in config["bound_inputs"]["weather_cache_sha256"].items()})
    mismatches = {str(path): (wanted, sha256(path)) for path, wanted in expected.items() if sha256(path) != wanted}
    if mismatches:
        raise RuntimeError(f"fixed input hash mismatch: {mismatches}")


def _read_baseline(path: Path, index: pd.DatetimeIndex) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    frame = frame.loc[index, list(TARGET_COLS)].astype(np.float64)
    if not frame.index.equals(index) or not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"baseline differs: {path}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {out_dir}")
    out_dir.mkdir(parents=True)

    config = _config()
    config_hash = sha256(CONFIG_PATH)
    validate_fixed_inputs(config)
    label_bytes, label_record = bounded_label_prefix()
    closure_path = source_lock(config_hash, label_record, out_dir)
    print(f"source closure locked: {sha256(closure_path)}", flush=True)
    guard_owner = acquire_heavy_guard(HEAVY_GUARD_PATH)
    atexit.register(release_heavy_guard, HEAVY_GUARD_PATH, guard_owner)
    print(f"heavy CPU guard acquired by PID {guard_owner['pid']}", flush=True)

    labels = read_labels(label_bytes)
    apply_index = pd.date_range(APPLY_START, APPLY_END, freq="h", name="forecast_kst_dtm")
    model_apply_index = pd.date_range(APPLY_START, MODEL_APPLY_END, freq="h", name="forecast_kst_dtm")
    fit_index = pd.date_range(FIT_START, FIT_END, freq="h", name="forecast_kst_dtm")
    labels_apply = labels.loc[apply_index, list(TARGET_COLS)]
    external_all = pd.read_parquet(EXTERNAL_PATH)
    external_all["time"] = pd.to_datetime(external_all["time"], errors="raise")
    baselines = {name: _read_baseline(path, apply_index) for name, path in BASELINES.items()}

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
            raise ValueError(f"{group}: control feature non-finite")
        external = external_all.loc[external_all["group"] == group].drop(columns="group").set_index("time")
        external.index.name = "forecast_kst_dtm"
        target = labels.loc[fit_index, group].astype(np.float64)
        eligible = target.notna() & (target >= 0.10 * CAPACITY_KWH[group])
        fit_rows = fit_index[eligible.to_numpy()]
        y = (target.loc[fit_rows] / CAPACITY_KWH[group]).to_numpy(dtype=np.float64)
        params = model_parameters(config)
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
            "control_feature_count": control_fit.shape[1],
        }
        for candidate in CANDIDATES:
            external_features = select_external_features(external, candidate)
            ext_fit = add_disagreement(external_features.loc[fit_index], control_fit)
            ext_apply = add_disagreement(external_features.loc[model_apply_index], control_apply)
            extended_fit = pd.concat([control_fit, ext_fit], axis=1)
            extended_apply = pd.concat([control_apply, ext_apply], axis=1)
            if extended_fit.shape[1] != 616 or tuple(extended_fit.columns[-4:]) != EXTENDED_COLUMNS:
                raise ValueError("extended model schema differs")
            model = LGBMRegressor(**params)
            model.fit(extended_fit.loc[fit_rows], y)
            extended_cf = np.clip(model.predict(extended_apply), 0.0, 1.02)
            delta = extended_cf - control_cf
            registered_delta = pd.Series(0.0, index=apply_index, dtype=np.float64)
            registered_delta.loc[model_apply_index] = delta
            if registered_delta.loc[APPLY_END] != 0.0:
                raise AssertionError("terminal paired increment must be exact zero")
            increments[f"{candidate}__{group}"] = registered_delta
            model_path = out_dir / f"models/{group}__{candidate}.joblib"
            joblib.dump(model, model_path, compress=3)
            model_records.append(file_record(model_path))
        print(f"fit {group}: {len(fit_rows)} eligible rows", flush=True)

    increment_path = out_dir / "predictions/candidate_increments.parquet"
    atomic_parquet(increments, increment_path)
    comparisons: dict[str, dict[str, Any]] = {candidate: {} for candidate in CANDIDATES}
    prediction_records: list[dict[str, Any]] = [file_record(increment_path)]
    for candidate in CANDIDATES:
        for baseline_name, baseline in baselines.items():
            predicted = baseline.copy()
            for group in TARGET_COLS:
                predicted[group] = apply_paired_increment(
                    baseline[group], increments[f"{candidate}__{group}"],
                    capacity_kwh=CAPACITY_KWH[group], weight=0.25,
                )
            prediction_path = out_dir / f"predictions/{candidate}__{baseline_name}.parquet"
            atomic_parquet(predicted, prediction_path)
            prediction_records.append(file_record(prediction_path))
            comparisons[candidate][baseline_name] = comparison_payload(labels_apply, baseline, predicted)

    gates = {candidate: gate_summary(comparisons[candidate]) for candidate in CANDIDATES}
    passing = [candidate for candidate in CANDIDATES if gates[candidate]["dual_gate_pass"]]
    winner = None
    if passing:
        winner = max(
            passing,
            key=lambda candidate: (
                gates[candidate]["minimum_group_slice_total_score_delta"],
                gates[candidate]["mean_group_slice_total_score_delta"],
                -CANDIDATES.index(candidate),
            ),
        )
    results = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "preregister_sha256": config_hash,
        "fit_window": [str(FIT_START), str(FIT_END)],
        "apply_window": [str(APPLY_START), str(APPLY_END)],
        "fit_records": fit_records,
        "increment_diagnostics": {
            column: {
                "mean_cf": float(increments[column].mean()),
                "std_cf": float(increments[column].std()),
                "min_cf": float(increments[column].min()),
                "max_cf": float(increments[column].max()),
            }
            for column in increments
        },
        "comparisons": comparisons,
        "gates": gates,
        "winner": winner,
        "status": "promoted_for_separate_prescore_lock_only" if winner else "rejected_on_2024_forward_dual_gate",
        "physical_access": {
            "maximum_external_calendar_year_requested_or_parsed": 2024,
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
        "status": results["status"],
        "winner": winner,
        "preregister_sha256": config_hash,
        "source_lock_sha256": sha256(closure_path),
        "stage_results_sha256": sha256(results_path),
        "2025_access_authorized_by_this_process": False,
    }, decision_path)

    outputs = [closure_path, *[Path(record["path"]) for record in model_records], *[Path(record["path"]) for record in prediction_records], results_path, decision_path]
    manifest_path = out_dir / "manifest.json"
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "preregister": file_record(CONFIG_PATH),
        "source_lock": file_record(closure_path),
        "inputs": [file_record(path) for path in [EXTERNAL_PATH, SOURCE_MANIFEST_PATH, PRIMARY_PATH, INTERACTION_PATH, *CACHE_PATHS.values()]],
        "outputs_excluding_this_manifest_and_terminal_sidecar": [file_record(path) for path in outputs],
        "unlisted_files_before_manifest": sorted(
            str(path.relative_to(out_dir)) for path in out_dir.rglob("*") if path.is_file() and path not in outputs
        ),
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

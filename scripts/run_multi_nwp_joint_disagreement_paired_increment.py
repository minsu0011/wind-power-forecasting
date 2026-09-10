"""Run one frozen three-source joint NWP paired increment on 2024 H1 -> H2.

This process is intentionally unable to request or read 2025 external values.
It promotes only by writing a lock that authorizes a separate final-stage
process.
"""

from __future__ import annotations

import argparse
import atexit
import ctypes
import hashlib
import io
import json
import os
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
from src.multi_nwp_joint import JOINT_COLUMNS, SOURCE_ORDER, build_joint_features, paired_transfer


EXPERIMENT_ID = "multi_nwp_joint_disagreement_2024_forward_v1"
CONFIG_PATH = ROOT / "configs/multi_nwp_joint_disagreement_paired_increment_preregister_v1.json"
SIDECAR_PATH = CONFIG_PATH.with_suffix(".sha256")
OUTPUT_DIR = ROOT / f"artifacts/postgate/{EXPERIMENT_ID}"
LABEL_PATH = Path(r"data/local/open/train/train_labels.csv")
BASELINE_PATH = ROOT / "artifacts/oof/gate2024_recent_v4_cf_fix_calibration_fit.parquet"
SOURCE_PATHS = {
    "ecmwf": ROOT / "artifacts/external/openmeteo_ecmwf_ifs025_previous_runs_v1/ecmwf_ifs025_group_centroids_2024.parquet",
    "icon": ROOT / "artifacts/external/openmeteo_icon_global_previous_runs_v1/icon_global_group_centroids_2024.parquet",
    "gfs": ROOT / "artifacts/external/openmeteo_gfs_global_previous_runs_v1/gfs_global_group_centroids_2024.parquet",
}
SOURCE_MANIFESTS = {name: path.with_name("source_manifest.json") for name, path in SOURCE_PATHS.items()}
CACHE_PATHS = {group: ROOT / f"artifacts/cache/{group}_weather_train.parquet" for group in TARGET_COLS}
HEAVY_GUARD_PATH = ROOT / "artifacts/locks/heavy_cpu_fit.pid.json"

FIT_START = pd.Timestamp("2024-03-09 00:00:00")
FIT_END = pd.Timestamp("2024-06-30 23:00:00")
APPLY_START = pd.Timestamp("2024-07-01 00:00:00")
MODEL_APPLY_END = pd.Timestamp("2024-12-31 23:00:00")
TERMINAL = pd.Timestamp("2025-01-01 00:00:00")
H1_LABEL_ROWS = 21_887
FULL_LABEL_ROWS = 26_304
TRANSFER_WEIGHT = 0.25
SLICES = {
    "H2": (pd.Timestamp("2024-07-01 00:00"), pd.Timestamp("2025-01-01 00:00")),
    "Q3": (pd.Timestamp("2024-07-01 00:00"), pd.Timestamp("2024-09-30 23:00")),
    "Q4": (pd.Timestamp("2024-10-01 00:00"), pd.Timestamp("2025-01-01 00:00")),
    "Jul": (pd.Timestamp("2024-07-01 00:00"), pd.Timestamp("2024-07-31 23:00")),
    "Aug": (pd.Timestamp("2024-08-01 00:00"), pd.Timestamp("2024-08-31 23:00")),
    "Sep": (pd.Timestamp("2024-09-01 00:00"), pd.Timestamp("2024-09-30 23:00")),
    "Oct": (pd.Timestamp("2024-10-01 00:00"), pd.Timestamp("2024-10-31 23:00")),
    "Nov": (pd.Timestamp("2024-11-01 00:00"), pd.Timestamp("2024-11-30 23:00")),
    "Dec": (pd.Timestamp("2024-12-01 00:00"), pd.Timestamp("2025-01-01 00:00")),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


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
    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return False
    ctypes.windll.kernel32.CloseHandle(handle)
    return True


def acquire_heavy_guard(path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    owner = {"pid": os.getpid(), "experiment_id": EXPERIMENT_ID, "created_utc": datetime.now(timezone.utc).isoformat()}
    for _ in range(2):
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            current = json.loads(path.read_text(encoding="utf-8"))
            if pid_is_alive(int(current["pid"])):
                raise RuntimeError(f"heavy CPU guard held by live process: {current}")
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
    except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError):
        return
    if int(current.get("pid", -1)) == int(owner["pid"]):
        path.unlink()


def read_prefix_bytes(data_rows: int) -> tuple[bytes, dict[str, Any]]:
    chunks: list[bytes] = []
    digest = hashlib.sha256()
    with LABEL_PATH.open("rb", buffering=0) as stream:
        for line_number in range(data_rows + 1):
            line = stream.readline()
            if not line:
                raise ValueError(f"label file ended at physical line {line_number}")
            chunks.append(line)
            digest.update(line)
    payload = b"".join(chunks)
    return payload, {
        "path": str(LABEL_PATH),
        "physical_lines_read": data_rows + 1,
        "data_rows": data_rows,
        "bytes": len(payload),
        "sha256": digest.hexdigest(),
    }


def parse_labels(payload: bytes, expected_rows: int, expected_end: pd.Timestamp) -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(payload), encoding="utf-8-sig")
    if tuple(frame.columns) != ("kst_dtm", *TARGET_COLS) or len(frame) != expected_rows:
        raise ValueError("bounded label schema/rows changed")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("kst_dtm"), errors="raise"), name="forecast_kst_dtm")
    frame = frame.astype(np.float64)
    if frame.index[0] != pd.Timestamp("2022-01-01 01:00") or frame.index[-1] != expected_end:
        raise ValueError("bounded label time range changed")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("bounded label index changed")
    return frame


def load_config() -> dict[str, Any]:
    expected = SIDECAR_PATH.read_text(encoding="ascii").split()[0]
    observed = sha256(CONFIG_PATH)
    if expected != observed:
        raise RuntimeError("preregistration sidecar mismatch")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["status"] != "frozen_before_any_candidate_fit_prediction_or_metric":
        raise RuntimeError("preregistration status is not frozen")
    return config


def validate_hashes(config: Mapping[str, Any]) -> None:
    bound = config["bound_files"]
    paths = [
        CONFIG_PATH.with_name("multi_nwp_joint_disagreement_paired_increment_preregister_v1.json"),
        ROOT / "src/multi_nwp_joint.py",
        ROOT / "src/metric.py",
        ROOT / "scripts/run_multi_nwp_joint_disagreement_paired_increment.py",
        ROOT / "tests/test_multi_nwp_joint.py",
        BASELINE_PATH,
        *SOURCE_PATHS.values(),
        *SOURCE_MANIFESTS.values(),
        *CACHE_PATHS.values(),
    ]
    mismatches = {}
    for path in paths:
        key = str(path.relative_to(ROOT)).replace("\\", "/")
        if key == "configs/multi_nwp_joint_disagreement_paired_increment_preregister_v1.json":
            continue
        observed = sha256(path)
        wanted = bound[key]["sha256"]
        if observed != wanted:
            mismatches[key] = {"expected": wanted, "observed": observed}
    if mismatches:
        raise RuntimeError(f"bound file hash mismatch: {mismatches}")


def load_source_group(path: Path, group: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["time"] = pd.to_datetime(frame["time"], errors="raise")
    frame = frame.loc[frame["group"] == group].drop(columns="group").set_index("time")
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    frame = frame.loc[index].astype(np.float64)
    if not frame.index.equals(index) or not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"{group}: incomplete source {path}")
    return frame


def metric_payload(actual: pd.Series, prediction: pd.Series, group: str) -> dict[str, Any]:
    payload = group_metrics(actual, prediction, CAPACITY_KWH[group], group_name=group).as_dict()
    payload["total_score"] = 0.5 * (payload["one_minus_nmae"] + payload["ficr"])
    return payload


def compare(labels: pd.DataFrame, baseline: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {"groups": {}, "mixed": {}}
    for group in TARGET_COLS:
        result["groups"][group] = {}
        for name, (start, end) in SLICES.items():
            index = labels.index[(labels.index >= start) & (labels.index <= end)]
            base = metric_payload(labels.loc[index, group], baseline.loc[index, group], group)
            cand = metric_payload(labels.loc[index, group], candidate.loc[index, group], group)
            result["groups"][group][name] = {
                "baseline": base,
                "candidate": cand,
                "delta": {key: cand[key] - base[key] for key in ("total_score", "one_minus_nmae", "ficr")},
            }
    for name, (start, end) in SLICES.items():
        index = labels.index[(labels.index >= start) & (labels.index <= end)]
        base = score_details(labels.loc[index], baseline.loc[index]).as_dict()
        cand = score_details(labels.loc[index], candidate.loc[index]).as_dict()
        result["mixed"][name] = {
            "baseline": base,
            "candidate": cand,
            "delta": {key: cand[key] - base[key] for key in ("total_score", "one_minus_nmae", "ficr")},
        }
    return result


def gate(comparison: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    rules = config["promotion_gate"]
    h2 = comparison["mixed"]["H2"]["delta"]
    group_h2 = {group: comparison["groups"][group]["H2"]["delta"]["total_score"] for group in TARGET_COLS}
    q_deltas = {name: comparison["mixed"][name]["delta"]["total_score"] for name in ("Q3", "Q4")}
    mixed_deltas = {name: comparison["mixed"][name]["delta"]["total_score"] for name in SLICES}
    checks = {
        "mixed_h2_total_positive": h2["total_score"] > rules["mixed_h2_min_total_score_delta"],
        "mixed_h2_ficr_positive": h2["ficr"] > rules["mixed_h2_min_ficr_delta"],
        "mixed_h2_nmae_floor": h2["one_minus_nmae"] >= rules["mixed_h2_min_one_minus_nmae_delta"],
        "every_group_h2_floor": min(group_h2.values()) > rules["minimum_group_h2_total_score_delta"],
        "q3_q4_floor": min(q_deltas.values()) > rules["minimum_q3_q4_mixed_total_score_delta"],
        "positive_mixed_slice_count": sum(value > 0.0 for value in mixed_deltas.values()) >= rules["minimum_positive_mixed_slices"],
    }
    return {
        "rules": dict(rules),
        "checks": checks,
        "pass": bool(all(checks.values())),
        "h2_mixed_delta": h2,
        "group_h2_total_score_deltas": group_h2,
        "q3_q4_mixed_total_score_deltas": q_deltas,
        "mixed_slice_total_score_deltas": mixed_deltas,
        "positive_mixed_slices": int(sum(value > 0.0 for value in mixed_deltas.values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {out_dir}")
    out_dir.mkdir(parents=True)

    config = load_config()
    validate_hashes(config)
    h1_bytes, h1_record = read_prefix_bytes(H1_LABEL_ROWS)
    source_lock_path = out_dir / "source_lock_before_h1_label_parse_or_candidate_fit.json"
    atomic_json({
        "schema_version": 1,
        "status": "locked_before_h1_label_parse_or_candidate_fit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "preregister": file_record(CONFIG_PATH),
        "sources": [file_record(path) for path in [ROOT / "src/multi_nwp_joint.py", ROOT / "src/metric.py", Path(__file__).resolve(), ROOT / "tests/test_multi_nwp_joint.py"]],
        "inputs": [file_record(path) for path in [BASELINE_PATH, *SOURCE_PATHS.values(), *SOURCE_MANIFESTS.values(), *CACHE_PATHS.values()]],
        "bounded_h1_label_prefix": h1_record,
        "physical_access": {"maximum_external_year": 2024, "2025_external_value_cells": 0, "h2_label_cells": 0, "candidate_fits_or_predictions": 0},
    }, source_lock_path)

    owner = acquire_heavy_guard(HEAVY_GUARD_PATH)
    atexit.register(release_heavy_guard, HEAVY_GUARD_PATH, owner)
    print(f"source locked; heavy guard acquired pid={owner['pid']}", flush=True)

    labels_h1 = parse_labels(h1_bytes, H1_LABEL_ROWS, FIT_END)
    fit_index = pd.date_range(FIT_START, FIT_END, freq="h", name="forecast_kst_dtm")
    model_apply_index = pd.date_range(APPLY_START, MODEL_APPLY_END, freq="h", name="forecast_kst_dtm")
    score_index = pd.date_range(APPLY_START, TERMINAL, freq="h", name="forecast_kst_dtm")
    complete_index = fit_index.append(model_apply_index)
    baseline_raw = pd.read_parquet(BASELINE_PATH)
    baseline_raw.index = pd.DatetimeIndex(baseline_raw.index, name="forecast_kst_dtm")
    baseline = baseline_raw.loc[score_index, list(TARGET_COLS)].astype(np.float64)
    for group in TARGET_COLS:
        baseline[group] = np.clip(0.97 * baseline[group], 0.0, 1.02 * CAPACITY_KWH[group])
    if not np.isfinite(baseline.to_numpy()).all():
        raise ValueError("scaled recent-v4 baseline is incomplete")

    increments = pd.DataFrame(0.0, index=score_index, columns=TARGET_COLS, dtype=np.float64)
    candidate = baseline.copy()
    fit_records: dict[str, Any] = {}
    model_paths: list[Path] = []
    params = dict(config["paired_model"]["parameters"])
    for group in TARGET_COLS:
        control_all = pd.read_parquet(CACHE_PATHS[group])
        control_all.index = pd.DatetimeIndex(control_all.index, name="forecast_kst_dtm")
        control = control_all.loc[complete_index].astype(np.float32)
        if control.shape[1] != 612 or not np.isfinite(control.to_numpy()).all():
            raise ValueError(f"{group}: canonical control differs")
        sources = {name: load_source_group(SOURCE_PATHS[name], group, complete_index) for name in SOURCE_ORDER}
        joint = build_joint_features(sources, control["cross__hub_ws_mean"])
        extended = pd.concat([control, joint], axis=1)
        if extended.shape[1] != 612 + len(JOINT_COLUMNS):
            raise AssertionError("extended feature count changed")

        target = labels_h1.loc[fit_index, group]
        eligible = target.notna() & (target >= 0.10 * CAPACITY_KWH[group])
        fit_rows = fit_index[eligible.to_numpy()]
        target_cf = (target.loc[fit_rows] / CAPACITY_KWH[group]).to_numpy(dtype=np.float64)
        control_model = LGBMRegressor(**params)
        extended_model = LGBMRegressor(**params)
        control_model.fit(control.loc[fit_rows], target_cf)
        extended_model.fit(extended.loc[fit_rows], target_cf)
        group_model_paths = {
            "control": out_dir / f"models/{group}__control.joblib",
            "extended": out_dir / f"models/{group}__joint_extended.joblib",
        }
        group_model_paths["control"].parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(control_model, group_model_paths["control"], compress=3)
        joblib.dump(extended_model, group_model_paths["extended"], compress=3)
        reloaded_control = joblib.load(group_model_paths["control"])
        reloaded_extended = joblib.load(group_model_paths["extended"])
        control_cf = np.clip(reloaded_control.predict(control.loc[model_apply_index]), 0.0, 1.02)
        extended_cf = np.clip(reloaded_extended.predict(extended.loc[model_apply_index]), 0.0, 1.02)
        delta = extended_cf - control_cf
        increments.loc[model_apply_index, group] = delta
        base_series = baseline.loc[model_apply_index, group]
        candidate.loc[model_apply_index, group] = paired_transfer(
            base_series,
            pd.Series(control_cf, index=model_apply_index),
            pd.Series(extended_cf, index=model_apply_index),
            capacity_kwh=CAPACITY_KWH[group],
            transfer_weight=TRANSFER_WEIGHT,
        )
        if candidate.loc[TERMINAL, group] != baseline.loc[TERMINAL, group] or increments.loc[TERMINAL, group] != 0.0:
            raise AssertionError("terminal prediction must be baseline identity")
        model_paths.extend(group_model_paths.values())
        fit_records[group] = {
            "complete_source_fit_rows": len(fit_index),
            "eligible_fit_rows": len(fit_rows),
            "fit_index_sha256": hashlib.sha256(fit_rows.asi8.tobytes()).hexdigest(),
            "target_cf_sha256": hashlib.sha256(target_cf.tobytes()).hexdigest(),
            "control_features": control.shape[1],
            "extended_features": extended.shape[1],
            "increment_mean_cf": float(np.mean(delta)),
            "increment_std_cf": float(np.std(delta)),
            "increment_min_cf": float(np.min(delta)),
            "increment_max_cf": float(np.max(delta)),
        }
        print(f"fit {group}: eligible={len(fit_rows)} delta_std={np.std(delta):.8f}", flush=True)

    increment_path = out_dir / "predictions/joint_paired_increment_cf.parquet"
    candidate_path = out_dir / "predictions/joint_candidate_scale097.parquet"
    atomic_parquet(increments, increment_path)
    atomic_parquet(candidate, candidate_path)
    candidate_lock_path = out_dir / "candidate_lock_before_h2_label_access.json"
    atomic_json({
        "schema_version": 1,
        "status": "single_candidate_locked_before_h2_label_access_or_metric",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_formula": "clip(clip(0.97*recent_v4,0,1.02C)+0.25*C*(joint_extended_cf-control_cf),0,1.02C)",
        "models": [file_record(path) for path in model_paths],
        "increments": file_record(increment_path),
        "candidate": file_record(candidate_path),
        "fit_records": fit_records,
        "2025_external_value_cells": 0,
    }, candidate_lock_path)
    print(f"candidate locked sha256={sha256(candidate_lock_path)}", flush=True)

    full_bytes, full_record = read_prefix_bytes(FULL_LABEL_ROWS)
    if not full_bytes.startswith(h1_bytes):
        raise AssertionError("full label prefix does not contain locked H1 bytes")
    score_access_path = out_dir / "h2_score_label_access_after_candidate_lock.json"
    atomic_json({
        "schema_version": 1,
        "status": "h2_score_labels_opened_after_single_candidate_hash_lock",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_lock": file_record(candidate_lock_path),
        "bounded_full_label_prefix": full_record,
        "h1_prefix_sha256_reconfirmed": hashlib.sha256(h1_bytes).hexdigest(),
    }, score_access_path)
    labels = parse_labels(full_bytes, FULL_LABEL_ROWS, TERMINAL).loc[score_index, list(TARGET_COLS)]
    comparison = compare(labels, baseline, candidate)
    gate_result = gate(comparison, config)
    status = "promoted_for_separate_2025_final_stage" if gate_result["pass"] else "rejected_on_frozen_2024_h2_gate"
    results_path = out_dir / "stage_results.json"
    atomic_json({
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "preregister_sha256": sha256(CONFIG_PATH),
        "fit_window": [str(FIT_START), str(FIT_END)],
        "apply_window": [str(APPLY_START), str(TERMINAL)],
        "fit_records": fit_records,
        "comparison": comparison,
        "gate": gate_result,
        "physical_access": {"maximum_external_year": 2024, "2025_external_requests_response_bytes_or_values": 0, "submission_csv_created": False},
    }, results_path)
    decision_path = out_dir / ("promotion_lock.json" if gate_result["pass"] else "rejection.json")
    atomic_json({
        "status": status,
        "preregister_sha256": sha256(CONFIG_PATH),
        "candidate_lock_sha256": sha256(candidate_lock_path),
        "stage_results_sha256": sha256(results_path),
        "separate_2025_final_stage_authorized": bool(gate_result["pass"]),
    }, decision_path)

    output_paths = [source_lock_path, *model_paths, increment_path, candidate_path, candidate_lock_path, score_access_path, results_path, decision_path]
    manifest_path = out_dir / "manifest.json"
    atomic_json({
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "preregister": file_record(CONFIG_PATH),
        "outputs_excluding_manifest_and_sidecar": [file_record(path) for path in output_paths],
        "unlisted_files_before_manifest": sorted(str(path.relative_to(out_dir)) for path in out_dir.rglob("*") if path.is_file() and path not in output_paths),
        "physical_access": {"maximum_external_year": 2024, "year_2025_external_value_cells": 0, "submission_csv_created": False},
    }, manifest_path)
    if json.loads(manifest_path.read_text(encoding="utf-8"))["unlisted_files_before_manifest"]:
        raise RuntimeError("unlisted outputs detected")
    sidecar_path = out_dir / "manifest.sha256"
    sidecar_path.write_text(f"{sha256(manifest_path)}  manifest.json\n", encoding="ascii")
    release_heavy_guard(HEAVY_GUARD_PATH, owner)
    print(f"status={status} h2_delta={gate_result['h2_mixed_delta']} positive_slices={gate_result['positive_mixed_slices']}", flush=True)
    print(f"manifest={manifest_path} sha256={sha256(manifest_path)}", flush=True)


if __name__ == "__main__":
    main()

"""Refit the promoted G1/G2 joint model through 2024 and create 2025 CSV."""

from __future__ import annotations

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

from src.metric import CAPACITY_KWH, TARGET_COLS
from src.multi_nwp_joint import JOINT_COLUMNS, SOURCE_ORDER, build_joint_features


EXPERIMENT_ID = "multi_nwp_joint_g12_posthoc_rescue_final_2025_v1"
CONFIG_PATH = ROOT / "configs/multi_nwp_joint_g12_final_preregister_v1.json"
SIDECAR_PATH = CONFIG_PATH.with_suffix(".sha256")
OUTPUT_DIR = ROOT / f"artifacts/postgate/{EXPERIMENT_ID}"
PROMOTION_LOCK = ROOT / "artifacts/postgate/multi_nwp_joint_g12_posthoc_rescue_v1/promotion_lock.json"
SOURCE_2025_DIR = ROOT / "artifacts/external/openmeteo_multi_nwp_g23_rescue_2025_v1"
SOURCE_2025_PATHS = {
    "ecmwf": SOURCE_2025_DIR / "ecmwf_ifs025_group_centroids_2025.parquet",
    "icon": SOURCE_2025_DIR / "icon_global_group_centroids_2025.parquet",
    "gfs": SOURCE_2025_DIR / "gfs_global_group_centroids_2025.parquet",
}
SOURCE_2025_MANIFEST = SOURCE_2025_DIR / "source_manifest.json"
SOURCE_2024_PATHS = {
    "ecmwf": ROOT / "artifacts/external/openmeteo_ecmwf_ifs025_previous_runs_v1/ecmwf_ifs025_group_centroids_2024.parquet",
    "icon": ROOT / "artifacts/external/openmeteo_icon_global_previous_runs_v1/icon_global_group_centroids_2024.parquet",
    "gfs": ROOT / "artifacts/external/openmeteo_gfs_global_previous_runs_v1/gfs_global_group_centroids_2024.parquet",
}
TRAIN_CACHE = {group: ROOT / f"artifacts/cache/{group}_weather_train.parquet" for group in TARGET_COLS}
TEST_CACHE = {group: ROOT / f"artifacts/cache/{group}_weather_test.parquet" for group in TARGET_COLS}
RECENT_V4_TEST = ROOT / "artifacts/final_cf_fix/predictions/corrected_recent_v4_test.parquet"
SAMPLE_PATH = Path(r"data/local/open/sample_submission.csv")
LABEL_PATH = Path(r"data/local/open/train/train_labels.csv")
HEAVY_GUARD_PATH = ROOT / "artifacts/locks/heavy_cpu_fit.pid.json"
FULL_LABEL_ROWS = 26_304
FIT_INDEX = pd.date_range("2024-03-09 00:00", "2024-12-31 23:00", freq="h", name="forecast_kst_dtm")
TEST_INDEX = pd.date_range("2025-01-01 01:00", "2026-01-01 00:00", freq="h", name="forecast_kst_dtm")
MODEL_APPLY_INDEX = pd.date_range("2025-01-01 01:00", "2025-12-31 23:00", freq="h", name="forecast_kst_dtm")
ACTIVE_GROUPS = ("kpx_group_1", "kpx_group_2")
IDENTITY_GROUP = "kpx_group_3"
TRANSFER_WEIGHT = 0.25


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
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


def acquire_guard(path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    owner = {"pid": os.getpid(), "experiment_id": EXPERIMENT_ID, "stage": "through_2024_paired_final_fit", "created_utc": datetime.now(timezone.utc).isoformat()}
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


def release_guard(path: Path, owner: Mapping[str, Any]) -> None:
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError):
        return
    if int(current.get("pid", -1)) == int(owner["pid"]):
        path.unlink()


def load_config() -> dict[str, Any]:
    expected = SIDECAR_PATH.read_text(encoding="ascii").split()[0]
    if sha256(CONFIG_PATH) != expected:
        raise RuntimeError("final preregistration sidecar mismatch")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["status"] != "frozen_after_rescue_promotion_and_compatible_source_reuse_binding_before_label_read_or_final_fit":
        raise RuntimeError("final preregistration status differs")
    for relative, wanted in config["bound_files"].items():
        path = ROOT / relative
        if path.stat().st_size != wanted["bytes"] or sha256(path) != wanted["sha256"]:
            raise RuntimeError(f"bound final file changed: {relative}")
    promotion = json.loads(PROMOTION_LOCK.read_text(encoding="utf-8"))
    if not promotion["separate_2025_stage_authorized"] or promotion["stage_results_sha256"] != config["promotion_evidence"]["stage_results_sha256"]:
        raise RuntimeError("rescue promotion evidence differs")
    return config


def bounded_labels() -> tuple[pd.DataFrame, dict[str, Any]]:
    chunks: list[bytes] = []
    digest = hashlib.sha256()
    with LABEL_PATH.open("rb", buffering=0) as stream:
        for line_number in range(FULL_LABEL_ROWS + 1):
            line = stream.readline()
            if not line:
                raise ValueError(f"label file ended at line {line_number}")
            chunks.append(line)
            digest.update(line)
    payload = b"".join(chunks)
    frame = pd.read_csv(io.BytesIO(payload), encoding="utf-8-sig")
    if tuple(frame.columns) != ("kst_dtm", *TARGET_COLS) or len(frame) != FULL_LABEL_ROWS:
        raise ValueError("bounded label schema changed")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("kst_dtm"), errors="raise"), name="forecast_kst_dtm")
    return frame.astype(np.float64), {"path": str(LABEL_PATH), "data_rows": FULL_LABEL_ROWS, "bytes": len(payload), "sha256": digest.hexdigest()}


def load_source(path: Path, group: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["time"] = pd.to_datetime(frame["time"], errors="raise")
    frame = frame.loc[frame["group"] == group].drop(columns="group").set_index("time")
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    frame = frame.loc[index].astype(np.float64)
    if not frame.index.equals(index) or not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"{group}: source is incomplete: {path}")
    return frame


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(OUTPUT_DIR)
    config = load_config()
    if not SOURCE_2025_MANIFEST.exists() or not all(path.exists() for path in SOURCE_2025_PATHS.values()):
        raise FileNotFoundError("run promoted 2025 source downloader first")
    OUTPUT_DIR.mkdir(parents=True)
    source_lock_path = OUTPUT_DIR / "source_lock_before_label_read_or_final_fit.json"
    atomic_json({
        "schema_version": 1,
        "status": "all_sources_locked_before_label_read_or_final_fit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "preregister": record(CONFIG_PATH),
        "promotion_lock": record(PROMOTION_LOCK),
        "source_2025_manifest": record(SOURCE_2025_MANIFEST),
        "source_2025": [record(path) for path in SOURCE_2025_PATHS.values()],
        "source_2024": [record(path) for path in SOURCE_2024_PATHS.values()],
        "canonical_weather": [record(path) for path in [*TRAIN_CACHE.values(), *TEST_CACHE.values()]],
        "baseline": record(RECENT_V4_TEST),
        "sample": record(SAMPLE_PATH),
        "physical_access": {"maximum_external_valid_timestamp_in_bound_archive": "2026-01-01 00:00:00", "terminal_source_row_used_by_model": False, "latest_model_application_source_time": "2025-12-31 23:00:00", "label_cells_read": 0, "model_fits": 0},
    }, source_lock_path)
    owner = acquire_guard(HEAVY_GUARD_PATH)
    atexit.register(release_guard, HEAVY_GUARD_PATH, owner)
    print(f"source locked; heavy guard acquired pid={owner['pid']}", flush=True)

    labels, label_record = bounded_labels()
    label_lock_path = OUTPUT_DIR / "label_lock_before_final_fit.json"
    atomic_json({"status": "bounded_operating_2024_labels_opened_after_source_lock_and_guard", "source_lock": record(source_lock_path), "labels": label_record}, label_lock_path)

    base_raw = pd.read_parquet(RECENT_V4_TEST)
    base_raw.index = pd.DatetimeIndex(base_raw.index, name="forecast_kst_dtm")
    base = base_raw.loc[TEST_INDEX, list(TARGET_COLS)].astype(np.float64)
    for group in TARGET_COLS:
        base[group] = np.clip(0.97 * base[group], 0.0, 1.02 * CAPACITY_KWH[group])
    if not np.isfinite(base.to_numpy()).all():
        raise ValueError("final scale097 base incomplete")
    increments = pd.DataFrame(0.0, index=TEST_INDEX, columns=TARGET_COLS, dtype=np.float64)
    prediction = base.copy()
    fit_records: dict[str, Any] = {}
    model_paths: list[Path] = []
    params = dict(config["paired_model_parameters"])
    for group in ACTIVE_GROUPS:
        control_train_all = pd.read_parquet(TRAIN_CACHE[group])
        control_train_all.index = pd.DatetimeIndex(control_train_all.index, name="forecast_kst_dtm")
        control_fit = control_train_all.loc[FIT_INDEX].astype(np.float32)
        control_test_all = pd.read_parquet(TEST_CACHE[group])
        control_test_all.index = pd.DatetimeIndex(control_test_all.index, name="forecast_kst_dtm")
        control_apply = control_test_all.loc[MODEL_APPLY_INDEX].astype(np.float32)
        if control_fit.shape[1] != 612 or control_apply.shape[1] != 612 or not np.isfinite(control_fit.to_numpy()).all() or not np.isfinite(control_apply.to_numpy()).all():
            raise ValueError(f"{group}: canonical 612 final schema differs")
        sources_fit = {name: load_source(SOURCE_2024_PATHS[name], group, FIT_INDEX) for name in SOURCE_ORDER}
        sources_apply = {name: load_source(SOURCE_2025_PATHS[name], group, MODEL_APPLY_INDEX) for name in SOURCE_ORDER}
        joint_fit = build_joint_features(sources_fit, control_fit["cross__hub_ws_mean"])
        joint_apply = build_joint_features(sources_apply, control_apply["cross__hub_ws_mean"])
        extended_fit = pd.concat([control_fit, joint_fit], axis=1)
        extended_apply = pd.concat([control_apply, joint_apply], axis=1)
        if extended_fit.shape[1] != 612 + len(JOINT_COLUMNS) or tuple(extended_fit.columns) != tuple(extended_apply.columns):
            raise AssertionError("final extended schema differs")
        target = labels.loc[FIT_INDEX, group]
        eligible = target.notna() & (target >= 0.10 * CAPACITY_KWH[group])
        fit_rows = FIT_INDEX[eligible.to_numpy()]
        target_cf = (target.loc[fit_rows] / CAPACITY_KWH[group]).to_numpy(dtype=np.float64)
        control_model = LGBMRegressor(**params)
        extended_model = LGBMRegressor(**params)
        control_model.fit(control_fit.loc[fit_rows], target_cf)
        extended_model.fit(extended_fit.loc[fit_rows], target_cf)
        control_before = np.clip(control_model.predict(control_apply), 0.0, 1.02)
        extended_before = np.clip(extended_model.predict(extended_apply), 0.0, 1.02)
        control_path = OUTPUT_DIR / f"models/{group}__control.joblib"
        extended_path = OUTPUT_DIR / f"models/{group}__joint_extended.joblib"
        control_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(control_model, control_path, compress=3)
        joblib.dump(extended_model, extended_path, compress=3)
        control_after = np.clip(joblib.load(control_path).predict(control_apply), 0.0, 1.02)
        extended_after = np.clip(joblib.load(extended_path).predict(extended_apply), 0.0, 1.02)
        if not np.array_equal(control_before, control_after) or not np.array_equal(extended_before, extended_after):
            raise AssertionError("model roundtrip changed final predictions")
        delta = extended_after - control_after
        increments.loc[MODEL_APPLY_INDEX, group] = delta
        prediction.loc[MODEL_APPLY_INDEX, group] = np.clip(
            base.loc[MODEL_APPLY_INDEX, group] + TRANSFER_WEIGHT * CAPACITY_KWH[group] * delta,
            0.0, 1.02 * CAPACITY_KWH[group],
        )
        model_paths.extend((control_path, extended_path))
        fit_records[group] = {
            "complete_source_rows": len(FIT_INDEX),
            "eligible_fit_rows": len(fit_rows),
            "fit_index_sha256": hashlib.sha256(fit_rows.asi8.tobytes()).hexdigest(),
            "target_cf_sha256": hashlib.sha256(target_cf.tobytes()).hexdigest(),
            "control_features": 612,
            "extended_features": 612 + len(JOINT_COLUMNS),
            "parameters_identical": control_model.get_params() == extended_model.get_params(),
            "increment_mean_cf": float(np.mean(delta)),
            "increment_std_cf": float(np.std(delta)),
            "increment_min_cf": float(np.min(delta)),
            "increment_max_cf": float(np.max(delta)),
        }
        print(f"final fit {group}: eligible={len(fit_rows)} delta_std={np.std(delta):.8f}", flush=True)

    if not (increments[IDENTITY_GROUP] == 0.0).all() or not (increments.loc[pd.Timestamp("2026-01-01 00:00")] == 0.0).all():
        raise AssertionError("G3 and terminal increments must be exact zero")
    if not np.array_equal(prediction[IDENTITY_GROUP].to_numpy(), base[IDENTITY_GROUP].to_numpy()):
        raise AssertionError("G3 final prediction must be exact base identity")
    for group in TARGET_COLS:
        if (prediction[group] < 0.0).any() or (prediction[group] > 1.02 * CAPACITY_KWH[group]).any():
            raise AssertionError(f"{group}: final prediction outside physical bounds")

    base_path = OUTPUT_DIR / "base_scale097_2025.parquet"
    increment_path = OUTPUT_DIR / "final_joint_increment_cf_2025.parquet"
    prediction_path = OUTPUT_DIR / "final_prediction_kwh_2025.parquet"
    atomic_parquet(base, base_path)
    atomic_parquet(increments, increment_path)
    atomic_parquet(prediction, prediction_path)

    sample = pd.read_csv(SAMPLE_PATH, encoding="utf-8-sig", dtype={"forecast_id": str, "forecast_kst_dtm": str})
    sample_index = pd.DatetimeIndex(pd.to_datetime(sample["forecast_kst_dtm"], errors="raise"), name="forecast_kst_dtm")
    if tuple(sample.columns) != ("forecast_id", "forecast_kst_dtm", *TARGET_COLS) or not sample_index.equals(TEST_INDEX):
        raise ValueError("sample submission schema/order changed")
    output = sample.copy()
    rounded = prediction.round(6)
    for group in TARGET_COLS:
        output[group] = rounded[group].to_numpy(dtype=np.float64)
    csv_path = OUTPUT_DIR / "multi_nwp_joint_g12_posthoc_rescue_recent097_2025.csv"
    output.to_csv(csv_path, index=False, encoding="utf-8-sig", lineterminator="\n", float_format="%.6f")
    raw_prefix = csv_path.read_bytes()[:3]
    reloaded = pd.read_csv(csv_path, encoding="utf-8-sig", dtype={"forecast_id": str, "forecast_kst_dtm": str})
    if raw_prefix != b"\xef\xbb\xbf" or tuple(reloaded.columns) != tuple(sample.columns) or len(reloaded) != len(sample):
        raise AssertionError("CSV BOM/schema/row count differs")
    if not reloaded["forecast_id"].equals(sample["forecast_id"]) or not reloaded["forecast_kst_dtm"].equals(sample["forecast_kst_dtm"]):
        raise AssertionError("CSV identifier/time order differs")
    csv_values = reloaded.loc[:, list(TARGET_COLS)].to_numpy(dtype=np.float64)
    if not np.array_equal(csv_values, rounded.to_numpy(dtype=np.float64)):
        raise AssertionError("CSV numeric replay differs from six-decimal prediction")

    audit_path = OUTPUT_DIR / "final_audit.json"
    atomic_json({
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_status": "posthoc_selection_unsafe",
        "promotion_evidence": record(PROMOTION_LOCK),
        "source_lock": record(source_lock_path),
        "label_lock": record(label_lock_path),
        "fit_records": fit_records,
        "artifacts": {
            "base": record(base_path), "increment": record(increment_path), "prediction": record(prediction_path), "csv": record(csv_path),
            "models": [record(path) for path in model_paths]
        },
        "csv_contract": {
            "utf8_bom_hex": raw_prefix.hex(), "rows": len(reloaded), "columns": list(reloaded.columns),
            "forecast_id_exact_sample_identity": True, "forecast_kst_dtm_exact_sample_identity": True,
            "numeric_exact_six_decimal_replay": True, "non_finite_cells": int((~np.isfinite(csv_values)).sum())
        },
        "identity_contract": {
            "g3_increment_nonzero_cells": int((increments[IDENTITY_GROUP] != 0.0).sum()),
            "g3_prediction_base_max_abs_difference": float(np.max(np.abs(prediction[IDENTITY_GROUP] - base[IDENTITY_GROUP]))),
            "terminal_increment_nonzero_cells": int((increments.loc[pd.Timestamp("2026-01-01 00:00")] != 0.0).sum())
        },
        "prediction_ranges_kwh": {group: {"min": float(prediction[group].min()), "max": float(prediction[group].max()), "capacity": CAPACITY_KWH[group]} for group in TARGET_COLS},
        "physical_access": {"maximum_external_valid_timestamp_in_bound_archive": "2026-01-01 00:00:00", "terminal_source_row_used_by_model": False, "latest_model_application_source_time": "2025-12-31 23:00:00", "public_feedback_files_read": 0},
        "no_retune_rescue_or_fallback_after_promotion": True,
    }, audit_path)

    outputs = [source_lock_path, label_lock_path, *model_paths, base_path, increment_path, prediction_path, csv_path, audit_path]
    manifest_path = OUTPUT_DIR / "manifest.json"
    atomic_json({
        "schema_version": 1, "experiment_id": EXPERIMENT_ID, "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_status": "posthoc_selection_unsafe", "preregister": record(CONFIG_PATH),
        "source_2025_manifest": record(SOURCE_2025_MANIFEST),
        "outputs_excluding_manifest_and_sidecar": [record(path) for path in outputs],
        "unlisted_files_before_manifest": sorted(str(path.relative_to(OUTPUT_DIR)) for path in OUTPUT_DIR.rglob("*") if path.is_file() and path not in outputs),
        "physical_access": {"maximum_external_valid_timestamp_in_bound_archive": "2026-01-01 00:00:00", "terminal_source_row_used_by_model": False, "latest_model_application_source_time": "2025-12-31 23:00:00", "submission_csv_created": True},
    }, manifest_path)
    if json.loads(manifest_path.read_text(encoding="utf-8"))["unlisted_files_before_manifest"]:
        raise RuntimeError("unlisted final outputs")
    (OUTPUT_DIR / "manifest.sha256").write_text(f"{sha256(manifest_path)}  manifest.json\n", encoding="ascii")
    release_guard(HEAVY_GUARD_PATH, owner)
    print(f"csv={record(csv_path)}", flush=True)
    print(f"manifest={record(manifest_path)}", flush=True)


if __name__ == "__main__":
    main()

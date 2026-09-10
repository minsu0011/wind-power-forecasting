"""Fit the promoted G2/G3 paired models and build the final 2025 submission."""

from __future__ import annotations

import atexit
import ctypes
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import joblib
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metric import CAPACITY_KWH, TARGET_COLS


EXPERIMENT_ID = "multi_nwp_consensus_g23_rescue_v1"
CONFIG_PATH = ROOT / "configs/multi_nwp_consensus_g23_rescue_preregister_v1.json"
CONFIG_SIDECAR = CONFIG_PATH.with_suffix(".sha256")
VALIDATION_PATH = ROOT / f"artifacts/postgate/{EXPERIMENT_ID}/validation_results.json"
SOURCE_2025_ROOT = ROOT / "artifacts/external/openmeteo_multi_nwp_g23_rescue_2025_v1"
SOURCE_2025_MANIFEST = SOURCE_2025_ROOT / "source_manifest.json"
BASELINE_PATH = ROOT / "artifacts/final_cf_fix/predictions/corrected_recent_v4_test.parquet"
LABEL_PATH = Path(r"data/local/open/train/train_labels.csv")
SAMPLE_PATH = Path(r"data/local/open/sample_submission.csv")
OUTPUT_DIR = ROOT / f"artifacts/final_{EXPERIMENT_ID}"
HEAVY_GUARD_PATH = ROOT / "artifacts/locks/heavy_cpu_fit.pid.json"
TRAIN_INDEX = pd.date_range(
    "2024-01-01 01:00", "2024-12-31 23:00", freq="h", name="forecast_kst_dtm"
)
TEST_INDEX = pd.date_range(
    "2025-01-01 01:00", "2026-01-01 00:00", freq="h", name="forecast_kst_dtm"
)
SOURCE_COLUMN_ID = "B_day1_hours01_13_else_day2_w025"
GROUPS_TO_FIT = ("kpx_group_2", "kpx_group_3")
BASE_SCALE = 0.97
TRANSFER_WEIGHT = 0.15
PARAMETERS = {
    "objective": "regression_l1",
    "n_estimators": 800,
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
    "n_jobs": 4,
}
SOURCE_SPECS = {
    "ecmwf_ifs025": {
        "heights": (100,),
        "prefix": "ecmwf",
        "train": ROOT
        / "artifacts/external/openmeteo_ecmwf_ifs025_previous_runs_v1/ecmwf_ifs025_group_centroids_2024.parquet",
        "train_manifest": ROOT
        / "artifacts/external/openmeteo_ecmwf_ifs025_previous_runs_v1/source_manifest.json",
        "test": SOURCE_2025_ROOT / "ecmwf_ifs025_group_centroids_2025.parquet",
        "parameter_config": ROOT / "configs/ecmwf_ifs025_paired_increment_preregister_v2.json",
    },
    "icon_global": {
        "heights": (10, 80, 120),
        "prefix": "icon",
        "train": ROOT
        / "artifacts/external/openmeteo_icon_global_previous_runs_v1/icon_global_group_centroids_2024.parquet",
        "train_manifest": ROOT
        / "artifacts/external/openmeteo_icon_global_previous_runs_v1/source_manifest.json",
        "test": SOURCE_2025_ROOT / "icon_global_group_centroids_2025.parquet",
        "parameter_config": ROOT / "configs/icon_global_paired_increment_preregister_v1.json",
    },
    "gfs_global": {
        "heights": (10, 80, 100),
        "prefix": "gfsrev",
        "train": ROOT
        / "artifacts/external/openmeteo_gfs_global_previous_runs_v1/gfs_global_group_centroids_2024.parquet",
        "train_manifest": ROOT
        / "artifacts/external/openmeteo_gfs_global_previous_runs_v1/source_manifest.json",
        "test": SOURCE_2025_ROOT / "gfs_global_group_centroids_2025.parquet",
        "parameter_config": ROOT / "configs/gfs_global_revision_paired_increment_preregister_v1.json",
    },
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
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    frame.to_parquet(temporary, engine="pyarrow", compression="zstd", index=True)
    os.replace(temporary, path)


def atomic_joblib(model: LGBMRegressor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    joblib.dump(model, temporary, compress=3)
    os.replace(temporary, path)


def pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return False
    ctypes.windll.kernel32.CloseHandle(handle)
    return True


def acquire_heavy_guard() -> dict[str, Any]:
    HEAVY_GUARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    owner = {
        "pid": os.getpid(),
        "experiment_id": EXPERIMENT_ID,
        "stage": "full_2024_paired_final_fit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    for _ in range(2):
        try:
            descriptor = os.open(HEAVY_GUARD_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            current = json.loads(HEAVY_GUARD_PATH.read_text(encoding="utf-8"))
            if pid_is_alive(int(current.get("pid", -1))):
                raise RuntimeError(f"heavy CPU guard is held: {current}")
            HEAVY_GUARD_PATH.unlink()
            continue
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(owner, stream, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
        return owner
    raise RuntimeError("could not acquire heavy CPU guard")


def release_heavy_guard(owner: Mapping[str, Any]) -> None:
    try:
        current = json.loads(HEAVY_GUARD_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return
    if int(current.get("pid", -1)) == int(owner["pid"]):
        HEAVY_GUARD_PATH.unlink()


def verify_promotion_and_inputs() -> dict[str, Any]:
    expected_config_hash = CONFIG_SIDECAR.read_text(encoding="utf-8").split()[0].lower()
    if sha256(CONFIG_PATH) != expected_config_hash:
        raise RuntimeError("frozen G23 rescue config hash mismatch")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    validation = json.loads(VALIDATION_PATH.read_text(encoding="utf-8"))
    gate = validation["promotion_gate"]
    if validation["config_sha256"] != expected_config_hash or not gate["promoted"]:
        raise RuntimeError("validation does not authorize final fitting")
    if not (gate["mixed_H2_delta"]["total_score"] > 0 and gate["mixed_H2_delta"]["ficr"] > 0):
        raise RuntimeError("validation promotion components differ")
    final_source_manifest = json.loads(SOURCE_2025_MANIFEST.read_text(encoding="utf-8"))
    if final_source_manifest["promotion_source"]["sha256"] != sha256(VALIDATION_PATH):
        raise RuntimeError("2025 source manifest is not bound to the promoted validation")
    for source_id, spec in SOURCE_SPECS.items():
        test_record = final_source_manifest["sources"][source_id]["normalized"]
        if sha256(Path(spec["test"])) != test_record["sha256"]:
            raise RuntimeError(f"{source_id}: final source hash mismatch")
        parameter_config = json.loads(Path(spec["parameter_config"]).read_text(encoding="utf-8"))
        if parameter_config["paired_model"]["parameters"] != PARAMETERS:
            raise RuntimeError(f"{source_id}: inherited paired-model parameters differ")
        train_manifest = json.loads(Path(spec["train_manifest"]).read_text(encoding="utf-8"))
        train_hash = train_manifest["normalized"]["sha256"]
        if sha256(Path(spec["train"])) != train_hash:
            raise RuntimeError(f"{source_id}: 2024 source hash mismatch")
    if sha256(BASELINE_PATH) != config["deployment_baseline"]["final_input"]["sha256"]:
        raise RuntimeError("recent-v4 final baseline hash mismatch")
    return {
        "config": config,
        "validation": validation,
        "source_2025_manifest": final_source_manifest,
    }


def read_external(path: Path, expected_index: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    frame = pd.read_parquet(path)
    frame["time"] = pd.to_datetime(frame["time"], errors="raise")
    result: dict[str, pd.DataFrame] = {}
    for group in GROUPS_TO_FIT:
        local = frame.loc[frame["group"] == group].drop(columns="group").set_index("time")
        local.index = pd.DatetimeIndex(local.index, name="forecast_kst_dtm")
        if not expected_index.isin(local.index).all():
            raise ValueError(f"{path}/{group}: expected timestamps missing")
        result[group] = local.reindex(expected_index)
    return result


def select_cutoff_features(
    raw: pd.DataFrame, *, prefix: str, heights: tuple[int, ...]
) -> pd.DataFrame:
    use_day1 = raw.index.hour.astype(int).isin(range(1, 14))
    result = pd.DataFrame(index=raw.index)
    for height in heights:
        speed = np.where(
            use_day1,
            raw[f"wind_speed_{height}m_previous_day1"].to_numpy(dtype=np.float64),
            raw[f"wind_speed_{height}m_previous_day2"].to_numpy(dtype=np.float64),
        )
        direction = np.where(
            use_day1,
            raw[f"wind_direction_{height}m_previous_day1"].to_numpy(dtype=np.float64),
            raw[f"wind_direction_{height}m_previous_day2"].to_numpy(dtype=np.float64),
        )
        radians = np.deg2rad(direction)
        result[f"{prefix}__ws{height}_ms"] = speed
        result[f"{prefix}__u{height}_ms"] = -speed * np.sin(radians)
        result[f"{prefix}__v{height}_ms"] = -speed * np.cos(radians)
    return result.astype(np.float32)


def add_disagreements(
    selected: pd.DataFrame,
    control: pd.DataFrame,
    *,
    prefix: str,
    heights: tuple[int, ...],
) -> pd.DataFrame:
    if not selected.index.equals(control.index):
        raise ValueError("selected external and control indexes differ")
    reference = control["cross__hub_ws_mean"].astype(np.float64)
    result = pd.DataFrame(index=selected.index)
    for height in heights:
        for component in ("ws", "u", "v"):
            name = f"{prefix}__{component}{height}_ms"
            result[name] = selected[name]
        result[f"{prefix}__ws{height}_minus_cross_hub_ws_mean"] = (
            selected[f"{prefix}__ws{height}_ms"].astype(np.float64) - reference
        ).astype(np.float32)
    return result.astype(np.float32)


def prediction_twice(path: Path, features: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    first_model = joblib.load(path)
    second_model = joblib.load(path)
    first = np.asarray(first_model.predict(features), dtype=np.float64)
    second = np.asarray(second_model.predict(features), dtype=np.float64)
    if not np.array_equal(first, second):
        raise AssertionError(f"two reload predictions are not exact: {path}")
    return first, {
        "model": file_record(path),
        "two_independent_joblib_loads": True,
        "prediction_float64_bit_exact": True,
        "prediction_sha256": hashlib.sha256(np.ascontiguousarray(first).tobytes()).hexdigest(),
    }


def write_submission(prediction: pd.DataFrame, path: Path) -> dict[str, Any]:
    sample = pd.read_csv(SAMPLE_PATH, encoding="utf-8-sig", dtype="string")
    expected_columns = ("forecast_id", "forecast_kst_dtm", *TARGET_COLS)
    if tuple(sample.columns) != expected_columns or len(sample) != len(TEST_INDEX):
        raise ValueError("sample submission schema differs")
    sample_index = pd.DatetimeIndex(
        pd.to_datetime(sample["forecast_kst_dtm"], errors="raise"), name="forecast_kst_dtm"
    )
    if not sample_index.equals(TEST_INDEX) or not prediction.index.equals(TEST_INDEX):
        raise ValueError("sample/prediction test index differs")
    output = sample.loc[:, ["forecast_id", "forecast_kst_dtm"]].copy()
    for group in TARGET_COLS:
        output[group] = prediction[group].to_numpy(dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    output.to_csv(
        temporary,
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
        lineterminator="\n",
    )
    os.replace(temporary, path)
    rendered = pd.read_csv(path, encoding="utf-8-sig", dtype="string", keep_default_na=False)
    if tuple(rendered.columns) != expected_columns or len(rendered) != len(sample):
        raise AssertionError("CSV readback schema differs")
    if not rendered.loc[:, ["forecast_id", "forecast_kst_dtm"]].equals(
        sample.loc[:, ["forecast_id", "forecast_kst_dtm"]]
    ):
        raise AssertionError("CSV identifiers differ from sample")
    for group in TARGET_COLS:
        expected_text = pd.Series(
            [f"{value:.6f}" for value in prediction[group].to_numpy(dtype=np.float64)],
            dtype="string",
        )
        if not rendered[group].reset_index(drop=True).equals(expected_text):
            raise AssertionError(f"CSV six-decimal rendering differs for {group}")
    raw = path.read_bytes()
    if not raw.startswith(b"\xef\xbb\xbf"):
        raise AssertionError("CSV is missing UTF-8 BOM")
    return {
        **file_record(path),
        "rows": len(rendered),
        "columns": list(rendered.columns),
        "encoding": "utf-8-sig",
        "line_ending": "LF",
        "numeric_decimals": 6,
        "identifier_match_sample": True,
        "readback_text_exact": True,
    }


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"refusing to overwrite final directory: {OUTPUT_DIR}")
    provenance = verify_promotion_and_inputs()
    OUTPUT_DIR.mkdir(parents=True)
    source_lock_path = OUTPUT_DIR / "source_lock_before_fit.json"
    atomic_json(
        {
            "experiment_id": EXPERIMENT_ID,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "selection_safety": provenance["config"]["selection_safety"],
            "promotion": file_record(VALIDATION_PATH),
            "source_2025_manifest": file_record(SOURCE_2025_MANIFEST),
            "baseline": file_record(BASELINE_PATH),
            "labels": file_record(LABEL_PATH),
            "sample": file_record(SAMPLE_PATH),
            "sources_2024": {
                source_id: {
                    "data": file_record(Path(spec["train"])),
                    "manifest": file_record(Path(spec["train_manifest"])),
                }
                for source_id, spec in SOURCE_SPECS.items()
            },
            "sources_2025": {
                source_id: file_record(Path(spec["test"]))
                for source_id, spec in SOURCE_SPECS.items()
            },
            "weather_caches": {
                group: {
                    "train": file_record(ROOT / f"artifacts/cache/{group}_weather_train.parquet"),
                    "test": file_record(ROOT / f"artifacts/cache/{group}_weather_test.parquet"),
                }
                for group in GROUPS_TO_FIT
            },
            "public_arrays_or_subgroup_feedback_used": False,
        },
        source_lock_path,
    )

    owner = acquire_heavy_guard()
    atexit.register(release_heavy_guard, owner)
    print(f"source lock={sha256(source_lock_path)}; heavy guard PID={owner['pid']}", flush=True)
    try:
        labels = pd.read_csv(LABEL_PATH, encoding="utf-8-sig", parse_dates=["kst_dtm"])
        labels = labels.set_index("kst_dtm")
        labels.index = pd.DatetimeIndex(labels.index, name="forecast_kst_dtm")
        labels = labels.loc[TRAIN_INDEX, list(TARGET_COLS)].astype(np.float64)
        external_train = {
            source_id: read_external(Path(spec["train"]), TRAIN_INDEX)
            for source_id, spec in SOURCE_SPECS.items()
        }
        external_test = {
            source_id: read_external(Path(spec["test"]), TEST_INDEX)
            for source_id, spec in SOURCE_SPECS.items()
        }
        increments: dict[str, pd.DataFrame] = {
            source_id: pd.DataFrame(index=TEST_INDEX, columns=GROUPS_TO_FIT, dtype=np.float64)
            for source_id in SOURCE_SPECS
        }
        model_records: dict[str, Any] = {}
        fit_records: dict[str, Any] = {}

        for group in GROUPS_TO_FIT:
            control_train_all = pd.read_parquet(
                ROOT / f"artifacts/cache/{group}_weather_train.parquet"
            )
            control_test = pd.read_parquet(ROOT / f"artifacts/cache/{group}_weather_test.parquet")
            control_train_all.index = pd.DatetimeIndex(
                control_train_all.index, name="forecast_kst_dtm"
            )
            control_test.index = pd.DatetimeIndex(control_test.index, name="forecast_kst_dtm")
            control_train = control_train_all.loc[TRAIN_INDEX].astype(np.float32)
            control_test = control_test.loc[TEST_INDEX].astype(np.float32)
            if control_train.shape[1] != 612 or control_test.shape[1] != 612:
                raise ValueError(f"{group}: canonical control feature count differs")
            target = labels[group]

            for source_id, spec in SOURCE_SPECS.items():
                heights = tuple(spec["heights"])
                prefix = str(spec["prefix"])
                selected_train = select_cutoff_features(
                    external_train[source_id][group], prefix=prefix, heights=heights
                )
                selected_test = select_cutoff_features(
                    external_test[source_id][group], prefix=prefix, heights=heights
                )
                complete = (
                    np.isfinite(selected_train.to_numpy(dtype=np.float64)).all(axis=1)
                    & np.isfinite(control_train.to_numpy(dtype=np.float64)).all(axis=1)
                    & np.isfinite(target.to_numpy(dtype=np.float64))
                    & (target.to_numpy(dtype=np.float64) >= 0.10 * CAPACITY_KWH[group])
                )
                fit_rows = TRAIN_INDEX[complete]
                if len(fit_rows) < 1_000:
                    raise ValueError(f"{source_id}/{group}: too few complete eligible 2024 rows")
                y = (target.loc[fit_rows] / CAPACITY_KWH[group]).to_numpy(dtype=np.float64)
                extension_train = add_disagreements(
                    selected_train, control_train, prefix=prefix, heights=heights
                )
                extension_test = add_disagreements(
                    selected_test, control_test, prefix=prefix, heights=heights
                )
                extended_train = pd.concat([control_train, extension_train], axis=1)
                extended_test = pd.concat([control_test, extension_test], axis=1)
                if not np.isfinite(extension_test.to_numpy(dtype=np.float64)).all():
                    raise ValueError(f"{source_id}/{group}: non-finite final extension")

                control_model = LGBMRegressor(**PARAMETERS)
                extended_model = LGBMRegressor(**PARAMETERS)
                control_model.fit(control_train.loc[fit_rows], y)
                extended_model.fit(extended_train.loc[fit_rows], y)
                control_path = OUTPUT_DIR / f"models/{source_id}/{group}__control.joblib"
                extended_path = OUTPUT_DIR / f"models/{source_id}/{group}__extended_B.joblib"
                atomic_joblib(control_model, control_path)
                atomic_joblib(extended_model, extended_path)

                control_prediction, control_reload = prediction_twice(control_path, control_test)
                extended_prediction, extended_reload = prediction_twice(
                    extended_path, extended_test
                )
                increment = np.clip(extended_prediction, 0.0, 1.02) - np.clip(
                    control_prediction, 0.0, 1.02
                )
                increments[source_id][group] = increment
                key = f"{source_id}/{group}"
                model_records[key] = {
                    "control": control_reload,
                    "extended": extended_reload,
                }
                fit_records[key] = {
                    "paired_row_identity": True,
                    "calendar": "all complete, official-metric-eligible operating-2024 rows",
                    "fit_rows": len(fit_rows),
                    "first_fit_time": str(fit_rows.min()),
                    "last_fit_time": str(fit_rows.max()),
                    "fit_index_sha256": hashlib.sha256(
                        np.ascontiguousarray(fit_rows.asi8).tobytes()
                    ).hexdigest(),
                    "target_cf_sha256": hashlib.sha256(
                        np.ascontiguousarray(y).tobytes()
                    ).hexdigest(),
                    "control_features": 612,
                    "extended_features": extended_train.shape[1],
                    "source_increment": {
                        "mean_cf": float(np.mean(increment)),
                        "std_cf": float(np.std(increment)),
                        "min_cf": float(np.min(increment)),
                        "max_cf": float(np.max(increment)),
                    },
                }
                print(f"fit {key}: rows={len(fit_rows)}", flush=True)

        source_increment_records: dict[str, Any] = {}
        for source_id, frame in increments.items():
            path = OUTPUT_DIR / f"predictions/{source_id}_increment_cf_2025.parquet"
            atomic_parquet(frame, path)
            source_increment_records[source_id] = file_record(path)

        consensus = pd.DataFrame(index=TEST_INDEX, columns=GROUPS_TO_FIT, dtype=np.float64)
        for group in GROUPS_TO_FIT:
            values = np.column_stack(
                [increments[source_id][group].to_numpy(dtype=np.float64) for source_id in SOURCE_SPECS]
            )
            consensus[group] = values.mean(axis=1)
        consensus_path = OUTPUT_DIR / "predictions/source3_average_increment_cf_2025.parquet"
        atomic_parquet(consensus, consensus_path)
        final_consensus = pd.DataFrame(0.0, index=TEST_INDEX, columns=TARGET_COLS, dtype=np.float64)
        for group in GROUPS_TO_FIT:
            final_consensus[group] = consensus[group].to_numpy(dtype=np.float64)
        if not np.array_equal(
            final_consensus["kpx_group_1"].to_numpy(dtype=np.float64),
            np.zeros(len(TEST_INDEX), dtype=np.float64),
        ):
            raise AssertionError("final consensus G1 increment is not exact zero")
        final_consensus_path = (
            OUTPUT_DIR / "predictions/final_consensus_increment_cf_2025.parquet"
        )
        atomic_parquet(final_consensus, final_consensus_path)

        recent = pd.read_parquet(BASELINE_PATH)
        recent.index = pd.DatetimeIndex(recent.index, name="forecast_kst_dtm")
        recent = recent.loc[TEST_INDEX, list(TARGET_COLS)].astype(np.float64)
        baseline = pd.DataFrame(index=TEST_INDEX, columns=TARGET_COLS, dtype=np.float64)
        final = pd.DataFrame(index=TEST_INDEX, columns=TARGET_COLS, dtype=np.float64)
        for group in TARGET_COLS:
            capacity = CAPACITY_KWH[group]
            baseline[group] = np.clip(
                BASE_SCALE * recent[group].to_numpy(dtype=np.float64), 0.0, 1.02 * capacity
            )
            if group == "kpx_group_1":
                final[group] = baseline[group].to_numpy(dtype=np.float64)
            else:
                final[group] = np.clip(
                    baseline[group].to_numpy(dtype=np.float64)
                    + TRANSFER_WEIGHT
                    * capacity
                    * final_consensus[group].to_numpy(dtype=np.float64),
                    0.0,
                    1.02 * capacity,
                )
        if not np.array_equal(final["kpx_group_1"].to_numpy(), baseline["kpx_group_1"].to_numpy()):
            raise AssertionError("final G1 is not exact deployment-baseline identity")
        if not np.isfinite(final.to_numpy()).all():
            raise AssertionError("final predictions contain non-finite values")
        for group in TARGET_COLS:
            if (final[group] < 0).any() or (final[group] > 1.02 * CAPACITY_KWH[group]).any():
                raise AssertionError(f"final predictions are outside bounds: {group}")

        baseline_out = OUTPUT_DIR / "predictions/recent097_baseline_2025.parquet"
        final_out = OUTPUT_DIR / "predictions/multi_nwp_consensus_g23_rescue_v1_2025.parquet"
        atomic_parquet(baseline, baseline_out)
        atomic_parquet(final, final_out)
        csv_path = OUTPUT_DIR / "predictions/multi_nwp_consensus_g23_rescue_v1_2025.csv"
        csv_record = write_submission(final, csv_path)

        final_lock = {
            "experiment_id": EXPERIMENT_ID,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_lock": file_record(source_lock_path),
            "formula": {
                "base": "clip(0.97*recent_v4, 0, 1.02*capacity)",
                "kpx_group_1": "exact base identity",
                "kpx_group_2_and_3": "clip(base + 0.15*capacity*mean(ECMWF,ICON,GFS raw paired CF increments), bounds)",
            },
            "cutoff_feature_id": SOURCE_COLUMN_ID,
            "model_parameters": PARAMETERS,
            "fit_records": fit_records,
            "model_reload_records": model_records,
            "all_twelve_models_two_reload_prediction_exact": all(
                item[kind]["prediction_float64_bit_exact"]
                for item in model_records.values()
                for kind in ("control", "extended")
            ),
            "source_increments": source_increment_records,
            "source_three_way_unweighted_mean": file_record(consensus_path),
            "final_consensus_increment_cf_G1_zero_G23_active": file_record(
                final_consensus_path
            ),
            "baseline": file_record(baseline_out),
            "final_parquet": file_record(final_out),
            "final_csv": csv_record,
            "g1_baseline_float64_bit_identity": True,
            "no_public_subgroup_feedback_or_array_inversion": True,
            "prediction_summary": {
                group: {
                    "min_kwh": float(final[group].min()),
                    "max_kwh": float(final[group].max()),
                    "mean_kwh": float(final[group].mean()),
                    "mean_delta_from_recent097_kwh": float((final[group] - baseline[group]).mean()),
                    "mean_abs_delta_from_recent097_kwh": float(
                        np.abs(final[group] - baseline[group]).mean()
                    ),
                }
                for group in TARGET_COLS
            },
        }
        final_lock_path = OUTPUT_DIR / "final_lock.json"
        atomic_json(final_lock, final_lock_path)
        manifest = {
            "experiment_id": EXPERIMENT_ID,
            "status": "final_csv_complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "selection_safety": provenance["config"]["selection_safety"],
            "files": [
                file_record(CONFIG_PATH),
                file_record(CONFIG_SIDECAR),
                file_record(Path(__file__)),
                file_record(source_lock_path),
                file_record(final_lock_path),
                file_record(csv_path),
                file_record(final_out),
                file_record(final_consensus_path),
                file_record(baseline_out),
            ],
        }
        atomic_json(manifest, OUTPUT_DIR / "manifest.json")
        print(f"FINAL_CSV={csv_path}", flush=True)
        print(f"FINAL_SHA256={sha256(csv_path)}", flush=True)
    finally:
        release_heavy_guard(owner)


if __name__ == "__main__":
    main()

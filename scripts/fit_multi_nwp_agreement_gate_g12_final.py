"""Fit the promoted source voters and build the agreement-gated G12 CSV."""

from __future__ import annotations

import atexit
import ctypes
import hashlib
import json
import os
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metric import CAPACITY_KWH, TARGET_COLS
from src.multi_nwp_agreement_gate import SOURCE_ORDER, agreement_gated_increment


EXPERIMENT_ID = "multi_nwp_agreement_gate_g12_v1"
CONFIG_PATH = ROOT / "configs/multi_nwp_agreement_gate_g12_preregister_v1.json"
CONFIG_SIDECAR = CONFIG_PATH.with_suffix(".sha256")
VALIDATION_ROOT = ROOT / f"artifacts/postgate/{EXPERIMENT_ID}"
PROMOTION_PATH = VALIDATION_ROOT / "promotion_lock.json"
STAGE_RESULTS_PATH = VALIDATION_ROOT / "stage_results.json"
OUTPUT_DIR = VALIDATION_ROOT / "final"
SOURCE_2025_ROOT = ROOT / "artifacts/external/openmeteo_multi_nwp_g23_rescue_2025_v1"
SOURCE_2025_MANIFEST = SOURCE_2025_ROOT / "source_manifest.json"
JOINT_INCREMENT_PATH = ROOT / "artifacts/postgate/multi_nwp_joint_g12_posthoc_rescue_final_2025_v1/final_joint_increment_cf_2025.parquet"
JOINT_FINAL_AUDIT_PATH = ROOT / "artifacts/postgate/multi_nwp_joint_g12_posthoc_rescue_final_2025_v1/final_audit.json"
BASELINE_PATH = ROOT / "artifacts/postgate/multi_nwp_joint_g12_posthoc_rescue_final_2025_v1/base_scale097_2025.parquet"
LABEL_PATH = Path(r"data/local/open/train/train_labels.csv")
SAMPLE_PATH = Path(r"data/local/open/sample_submission.csv")
HEAVY_GUARD_PATH = ROOT / "artifacts/locks/heavy_cpu_fit.pid.json"
TRAIN_INDEX = pd.date_range(
    "2024-01-01 01:00", "2024-12-31 23:00", freq="h", name="forecast_kst_dtm"
)
TEST_INDEX = pd.date_range(
    "2025-01-01 01:00", "2026-01-01 00:00", freq="h", name="forecast_kst_dtm"
)
TERMINAL = pd.Timestamp("2026-01-01 00:00:00")
GROUPS_TO_FIT = ("kpx_group_1", "kpx_group_2")
IDENTITY_GROUP = "kpx_group_3"
TRANSFER_WEIGHT = 0.25
EXPECTED_CONFIG_SHA = "c205704467f520d4547f7aa6ac3e2ffa30cf49aa557fa6d3f7ff30256cdb4067"
EXPECTED_JOINT_INCREMENT_SHA = "204d01db3fdf8919440126423963d89a32afb8c2b46bd451f5027da012c62a30"
EXPECTED_BASELINE_SHA = "a09ab6906dc026ad82b0ea7d8a0e9f6e88e8b4f839ba5d4d2426f37f0b7405cb"
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
    "ecmwf": {
        "manifest_id": "ecmwf_ifs025",
        "heights": (100,),
        "prefix": "ecmwf",
        "train": ROOT / "artifacts/external/openmeteo_ecmwf_ifs025_previous_runs_v1/ecmwf_ifs025_group_centroids_2024.parquet",
        "train_manifest": ROOT / "artifacts/external/openmeteo_ecmwf_ifs025_previous_runs_v1/source_manifest.json",
        "test": SOURCE_2025_ROOT / "ecmwf_ifs025_group_centroids_2025.parquet",
        "parameter_config": ROOT / "configs/ecmwf_ifs025_paired_increment_preregister_v2.json",
    },
    "icon": {
        "manifest_id": "icon_global",
        "heights": (10, 80, 120),
        "prefix": "icon",
        "train": ROOT / "artifacts/external/openmeteo_icon_global_previous_runs_v1/icon_global_group_centroids_2024.parquet",
        "train_manifest": ROOT / "artifacts/external/openmeteo_icon_global_previous_runs_v1/source_manifest.json",
        "test": SOURCE_2025_ROOT / "icon_global_group_centroids_2025.parquet",
        "parameter_config": ROOT / "configs/icon_global_paired_increment_preregister_v1.json",
    },
    "gfs": {
        "manifest_id": "gfs_global",
        "heights": (10, 80, 100),
        "prefix": "gfsrev",
        "train": ROOT / "artifacts/external/openmeteo_gfs_global_previous_runs_v1/gfs_global_group_centroids_2024.parquet",
        "train_manifest": ROOT / "artifacts/external/openmeteo_gfs_global_previous_runs_v1/source_manifest.json",
        "test": SOURCE_2025_ROOT / "gfs_global_group_centroids_2025.parquet",
        "parameter_config": ROOT / "configs/gfs_global_revision_paired_increment_preregister_v1.json",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        "stage": "through_2024_source_voter_fit",
        "created_utc": utc_now(),
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
    sidecar_hash = CONFIG_SIDECAR.read_text(encoding="ascii").split()[0].lower()
    if sidecar_hash != EXPECTED_CONFIG_SHA or sha256(CONFIG_PATH) != EXPECTED_CONFIG_SHA:
        raise RuntimeError("frozen agreement-gate config hash mismatch")
    promotion = json.loads(PROMOTION_PATH.read_text(encoding="utf-8"))
    results = json.loads(STAGE_RESULTS_PATH.read_text(encoding="utf-8"))
    if (
        not promotion["separate_final_stage_authorized"]
        or promotion["preregister_sha256"] != EXPECTED_CONFIG_SHA
        or promotion["stage_results_sha256"] != sha256(STAGE_RESULTS_PATH)
        or not results["gate"]["pass"]
        or not all(results["gate"]["checks"].values())
    ):
        raise RuntimeError("validation promotion lock does not authorize final fitting")
    if sha256(JOINT_INCREMENT_PATH) != EXPECTED_JOINT_INCREMENT_SHA:
        raise RuntimeError("locked through-2024 joint increment differs")
    if sha256(BASELINE_PATH) != EXPECTED_BASELINE_SHA:
        raise RuntimeError("locked scale-0.97 final baseline differs")
    source_manifest = json.loads(SOURCE_2025_MANIFEST.read_text(encoding="utf-8"))
    for source_name, spec in SOURCE_SPECS.items():
        manifest_id = str(spec["manifest_id"])
        expected_test = source_manifest["sources"][manifest_id]["normalized"]["sha256"]
        if sha256(Path(spec["test"])) != expected_test:
            raise RuntimeError(f"{source_name}: 2025 source hash mismatch")
        train_manifest = json.loads(Path(spec["train_manifest"]).read_text(encoding="utf-8"))
        if sha256(Path(spec["train"])) != train_manifest["normalized"]["sha256"]:
            raise RuntimeError(f"{source_name}: 2024 source hash mismatch")
        parameter_config = json.loads(Path(spec["parameter_config"]).read_text(encoding="utf-8"))
        if parameter_config["paired_model"]["parameters"] != PARAMETERS:
            raise RuntimeError(f"{source_name}: paired model parameter contract differs")
    return {"promotion": promotion, "results": results, "source_manifest": source_manifest}


def read_external(
    path: Path, expected_index: pd.DatetimeIndex
) -> dict[str, pd.DataFrame]:
    frame = pd.read_parquet(path)
    frame["time"] = pd.to_datetime(frame["time"], errors="raise")
    output: dict[str, pd.DataFrame] = {}
    for group in GROUPS_TO_FIT:
        local = frame.loc[frame["group"] == group].drop(columns="group").set_index("time")
        local.index = pd.DatetimeIndex(local.index, name="forecast_kst_dtm")
        if not expected_index.isin(local.index).all():
            raise ValueError(f"{path}/{group}: expected timestamps missing")
        output[group] = local.reindex(expected_index)
    return output


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
    first = np.asarray(joblib.load(path).predict(features), dtype=np.float64)
    second = np.asarray(joblib.load(path).predict(features), dtype=np.float64)
    if not np.array_equal(first, second):
        raise AssertionError(f"two reload predictions differ: {path}")
    return first, {
        "model": file_record(path),
        "two_independent_joblib_loads": True,
        "prediction_float64_bit_exact": True,
        "prediction_sha256": hashlib.sha256(np.ascontiguousarray(first).tobytes()).hexdigest(),
    }


def write_submission(prediction: pd.DataFrame, path: Path) -> dict[str, Any]:
    sample = pd.read_csv(SAMPLE_PATH, encoding="utf-8-sig", dtype="string")
    expected_columns = ("forecast_id", "forecast_kst_dtm", *TARGET_COLS)
    sample_index = pd.DatetimeIndex(
        pd.to_datetime(sample["forecast_kst_dtm"], errors="raise"), name="forecast_kst_dtm"
    )
    if (
        tuple(sample.columns) != expected_columns
        or len(sample) != len(TEST_INDEX)
        or not sample_index.equals(TEST_INDEX)
        or not prediction.index.equals(TEST_INDEX)
    ):
        raise ValueError("sample submission contract differs")
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
        raise AssertionError("CSV identifiers differ")
    for group in TARGET_COLS:
        expected_text = pd.Series(
            [f"{value:.6f}" for value in prediction[group].to_numpy(dtype=np.float64)],
            dtype="string",
        )
        if not rendered[group].reset_index(drop=True).equals(expected_text):
            raise AssertionError(f"CSV six-decimal rendering differs: {group}")
    if not path.read_bytes().startswith(b"\xef\xbb\xbf"):
        raise AssertionError("CSV is missing UTF-8 BOM")
    return {
        **file_record(path),
        "rows": len(rendered),
        "columns": list(rendered.columns),
        "encoding": "utf-8-sig",
        "numeric_decimals": 6,
        "identifier_match_sample": True,
        "readback_text_exact": True,
    }


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"refusing to overwrite final directory: {OUTPUT_DIR}")
    provenance = verify_promotion_and_inputs()
    OUTPUT_DIR.mkdir(parents=True)
    source_lock_path = OUTPUT_DIR / "source_lock_before_label_read_or_fit.json"
    atomic_json(
        {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "status": "all_final_sources_locked_before_label_read_or_model_fit",
            "created_utc": utc_now(),
            "config": file_record(CONFIG_PATH),
            "validation_manifest": file_record(VALIDATION_ROOT / "manifest.json"),
            "promotion": file_record(PROMOTION_PATH),
            "stage_results": file_record(STAGE_RESULTS_PATH),
            "final_builder": file_record(Path(__file__).resolve()),
            "gate_code": file_record(ROOT / "src/multi_nwp_agreement_gate.py"),
            "joint_increment": file_record(JOINT_INCREMENT_PATH),
            "joint_final_audit": file_record(JOINT_FINAL_AUDIT_PATH),
            "baseline": file_record(BASELINE_PATH),
            "source_2025_manifest": file_record(SOURCE_2025_MANIFEST),
            "labels": file_record(LABEL_PATH),
            "sample": file_record(SAMPLE_PATH),
            "sources": {
                name: {
                    "train": file_record(Path(spec["train"])),
                    "train_manifest": file_record(Path(spec["train_manifest"])),
                    "test": file_record(Path(spec["test"])),
                    "parameter_config": file_record(Path(spec["parameter_config"])),
                }
                for name, spec in SOURCE_SPECS.items()
            },
            "weather_caches": {
                group: {
                    "train": file_record(ROOT / f"artifacts/cache/{group}_weather_train.parquet"),
                    "test": file_record(ROOT / f"artifacts/cache/{group}_weather_test.parquet"),
                }
                for group in GROUPS_TO_FIT
            },
            "physical_access": {"label_values_parsed": 0, "model_fits": 0},
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
            name: read_external(Path(spec["train"]), TRAIN_INDEX)
            for name, spec in SOURCE_SPECS.items()
        }
        external_test = {
            name: read_external(Path(spec["test"]), TEST_INDEX)
            for name, spec in SOURCE_SPECS.items()
        }
        increments = {
            name: pd.DataFrame(0.0, index=TEST_INDEX, columns=TARGET_COLS, dtype=np.float64)
            for name in SOURCE_ORDER
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

            for source_name, spec in SOURCE_SPECS.items():
                heights = tuple(spec["heights"])
                prefix = str(spec["prefix"])
                selected_train = select_cutoff_features(
                    external_train[source_name][group], prefix=prefix, heights=heights
                )
                selected_test = select_cutoff_features(
                    external_test[source_name][group], prefix=prefix, heights=heights
                )
                complete = (
                    np.isfinite(selected_train.to_numpy(dtype=np.float64)).all(axis=1)
                    & np.isfinite(control_train.to_numpy(dtype=np.float64)).all(axis=1)
                    & np.isfinite(target.to_numpy(dtype=np.float64))
                    & (target.to_numpy(dtype=np.float64) >= 0.10 * CAPACITY_KWH[group])
                )
                fit_rows = TRAIN_INDEX[complete]
                if len(fit_rows) < 1_000:
                    raise ValueError(f"{source_name}/{group}: too few complete eligible rows")
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
                    raise ValueError(f"{source_name}/{group}: non-finite final extension")

                control_model = LGBMRegressor(**PARAMETERS)
                extended_model = LGBMRegressor(**PARAMETERS)
                control_model.fit(control_train.loc[fit_rows], y)
                extended_model.fit(extended_train.loc[fit_rows], y)
                control_path = OUTPUT_DIR / f"models/{source_name}/{group}__control.joblib"
                extended_path = OUTPUT_DIR / f"models/{source_name}/{group}__extended_B.joblib"
                atomic_joblib(control_model, control_path)
                atomic_joblib(extended_model, extended_path)
                control_prediction, control_reload = prediction_twice(control_path, control_test)
                extended_prediction, extended_reload = prediction_twice(
                    extended_path, extended_test
                )
                increment = np.clip(extended_prediction, 0.0, 1.02) - np.clip(
                    control_prediction, 0.0, 1.02
                )
                increments[source_name][group] = increment
                key = f"{source_name}/{group}"
                model_records[key] = {"control": control_reload, "extended": extended_reload}
                fit_records[key] = {
                    "paired_row_identity": True,
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
                    "increment_summary": {
                        "mean_cf": float(np.mean(increment)),
                        "std_cf": float(np.std(increment)),
                        "min_cf": float(np.min(increment)),
                        "max_cf": float(np.max(increment)),
                    },
                }
                print(f"fit {key}: rows={len(fit_rows)}", flush=True)

        for frame in increments.values():
            frame.loc[TERMINAL, :] = 0.0
        source_increment_records: dict[str, Any] = {}
        for source_name, frame in increments.items():
            path = OUTPUT_DIR / f"predictions/{source_name}_increment_cf_2025.parquet"
            atomic_parquet(frame, path)
            source_increment_records[source_name] = file_record(path)

        joint = pd.read_parquet(JOINT_INCREMENT_PATH).loc[TEST_INDEX, list(TARGET_COLS)].astype(np.float64)
        joint.index = pd.DatetimeIndex(joint.index, name="forecast_kst_dtm")
        if np.count_nonzero(joint.loc[TERMINAL].to_numpy(dtype=np.float64)):
            raise AssertionError("through-2024 joint terminal increment differs")
        gated, confidence, agreement_count = agreement_gated_increment(joint, increments)
        gated[IDENTITY_GROUP] = 0.0
        confidence[IDENTITY_GROUP] = 0.0
        agreement_count[IDENTITY_GROUP] = 0

        gated_path = OUTPUT_DIR / "predictions/agreement_gated_joint_increment_cf_2025.parquet"
        confidence_path = OUTPUT_DIR / "diagnostics/agreement_confidence_2025.parquet"
        count_path = OUTPUT_DIR / "diagnostics/source_sign_agreement_count_2025.parquet"
        atomic_parquet(gated, gated_path)
        atomic_parquet(confidence, confidence_path)
        atomic_parquet(agreement_count, count_path)

        baseline = pd.read_parquet(BASELINE_PATH).loc[TEST_INDEX, list(TARGET_COLS)].astype(np.float64)
        baseline.index = pd.DatetimeIndex(baseline.index, name="forecast_kst_dtm")
        final = baseline.copy()
        for group in GROUPS_TO_FIT:
            final[group] = np.clip(
                baseline[group].to_numpy(dtype=np.float64)
                + TRANSFER_WEIGHT
                * CAPACITY_KWH[group]
                * gated[group].to_numpy(dtype=np.float64),
                0.0,
                1.02 * CAPACITY_KWH[group],
            )
        if not np.array_equal(
            final[IDENTITY_GROUP].to_numpy(dtype=np.float64).view(np.uint64),
            baseline[IDENTITY_GROUP].to_numpy(dtype=np.float64).view(np.uint64),
        ):
            raise AssertionError("final G3 is not exact baseline identity")
        if not np.array_equal(
            final.loc[[TERMINAL]].to_numpy(dtype=np.float64).view(np.uint64),
            baseline.loc[[TERMINAL]].to_numpy(dtype=np.float64).view(np.uint64),
        ):
            raise AssertionError("final terminal is not exact baseline identity")
        if not np.isfinite(final.to_numpy()).all():
            raise AssertionError("final predictions contain non-finite values")
        for group in TARGET_COLS:
            if (final[group] < 0).any() or (final[group] > 1.02 * CAPACITY_KWH[group]).any():
                raise AssertionError(f"final predictions out of bounds: {group}")

        final_parquet_path = OUTPUT_DIR / "multi_nwp_agreement_gate_g12_recent097_2025.parquet"
        atomic_parquet(final, final_parquet_path)
        csv_path = OUTPUT_DIR / "multi_nwp_agreement_gate_g12_recent097_2025.csv"
        csv_record = write_submission(final, csv_path)
        diagnostics = {
            group: {
                "agreement_count_histogram": {
                    str(value): int((agreement_count[group] == value).sum()) for value in range(4)
                },
                "mean_confidence": float(confidence[group].mean()),
                "median_confidence": float(confidence[group].median()),
                "nonzero_gated_rows": int(np.count_nonzero(gated[group].to_numpy())),
                "joint_mean_abs_cf": float(joint[group].abs().mean()),
                "gated_mean_abs_cf": float(gated[group].abs().mean()),
                "mean_abs_delta_from_baseline_kwh": float(
                    np.abs(final[group] - baseline[group]).mean()
                ),
            }
            for group in TARGET_COLS
        }
        final_audit_path = OUTPUT_DIR / "final_audit.json"
        atomic_json(
            {
                "schema_version": 1,
                "experiment_id": EXPERIMENT_ID,
                "status": "final_csv_complete",
                "created_utc": utc_now(),
                "source_lock": file_record(source_lock_path),
                "promotion": provenance["promotion"],
                "formula": {
                    "active_groups": list(GROUPS_TO_FIT),
                    "identity_group": IDENTITY_GROUP,
                    "transfer_weight": TRANSFER_WEIGHT,
                    "agreement": "count same-sign source increments; require 2 of 3; vote=count/3",
                    "magnitude_cap": "min(1,median_abs_source/abs_joint)",
                    "terminal_identity": True,
                },
                "model_parameters": PARAMETERS,
                "fit_records": fit_records,
                "model_reload_records": model_records,
                "all_twelve_models_two_reload_prediction_exact": all(
                    record[kind]["prediction_float64_bit_exact"]
                    for record in model_records.values()
                    for kind in ("control", "extended")
                ),
                "source_increments": source_increment_records,
                "joint_increment": file_record(JOINT_INCREMENT_PATH),
                "gated_increment": file_record(gated_path),
                "confidence": file_record(confidence_path),
                "agreement_count": file_record(count_path),
                "baseline": file_record(BASELINE_PATH),
                "final_parquet": file_record(final_parquet_path),
                "final_csv": csv_record,
                "diagnostics": diagnostics,
                "g3_float64_bit_identity": True,
                "terminal_float64_bit_identity": True,
                "no_public_subgroup_time_or_array_inversion": True,
                "no_final_retune_or_fallback": True,
            },
            final_audit_path,
        )
        manifest_path = OUTPUT_DIR / "manifest.json"
        listed = [
            source_lock_path,
            *[
                Path(record[kind]["model"]["path"])
                for record in model_records.values()
                for kind in ("control", "extended")
            ],
            *[Path(record["path"]) for record in source_increment_records.values()],
            gated_path,
            confidence_path,
            count_path,
            final_parquet_path,
            csv_path,
            final_audit_path,
        ]
        atomic_json(
            {
                "schema_version": 1,
                "experiment_id": EXPERIMENT_ID,
                "status": "final_csv_complete",
                "created_utc": utc_now(),
                "config": file_record(CONFIG_PATH),
                "final_builder": file_record(Path(__file__).resolve()),
                "outputs_excluding_manifest_and_sidecar": [file_record(path) for path in listed],
                "unlisted_files_before_manifest": sorted(
                    str(path.relative_to(OUTPUT_DIR))
                    for path in OUTPUT_DIR.rglob("*")
                    if path.is_file() and path not in listed
                ),
            },
            manifest_path,
        )
        if json.loads(manifest_path.read_text(encoding="utf-8"))["unlisted_files_before_manifest"]:
            raise RuntimeError("unlisted final output detected")
        sidecar = OUTPUT_DIR / "manifest.sha256"
        sidecar.write_text(f"{sha256(manifest_path)}  manifest.json\n", encoding="ascii")
        print(f"FINAL_CSV={csv_path}", flush=True)
        print(f"FINAL_SHA256={sha256(csv_path)}", flush=True)
        print(f"FINAL_MANIFEST_SHA256={sha256(manifest_path)}", flush=True)
    finally:
        release_heavy_guard(owner)


if __name__ == "__main__":
    main()

"""Validate and, only on promotion, finalize the multi-NWP FICR router."""

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

from src.metric import CAPACITY_KWH, TARGET_COLS, group_metrics, score_details
from src.multi_nwp_ficr_boundary_router import (
    ACTION_ALPHAS,
    COMMON_PARAMETERS,
    MultiNWPBoundaryRouter,
    build_boundary_context,
)
from src.multi_nwp_joint import SOURCE_ORDER, build_joint_features


EXPERIMENT_ID = "multi_nwp_ficr_boundary_router_g12_v1"
CONFIG_PATH = ROOT / "configs/multi_nwp_ficr_boundary_router_g12_preregister_v1.json"
SIDECAR_PATH = CONFIG_PATH.with_suffix(".sha256")
OUTPUT_DIR = ROOT / f"artifacts/postgate/{EXPERIMENT_ID}"
LABEL_PATH = Path(r"data/local/open/train/train_labels.csv")
SAMPLE_PATH = Path(r"data/local/open/sample_submission.csv")
BASELINE_2024 = ROOT / "artifacts/oof/gate2024_recent_v4_cf_fix_calibration_fit.parquet"
BASELINE_2025 = ROOT / "artifacts/final_cf_fix/predictions/corrected_recent_v4_test.parquet"
ORIGINAL_INCREMENT_2024 = ROOT / "artifacts/postgate/multi_nwp_joint_disagreement_2024_forward_v1/predictions/joint_paired_increment_cf.parquet"
FINAL_INCREMENT_2025 = ROOT / "artifacts/postgate/multi_nwp_joint_g12_posthoc_rescue_final_2025_v1/final_joint_increment_cf_2025.parquet"
PUBLIC_SUBMITTED_PARENT = ROOT / "artifacts/postgate/multi_nwp_joint_g12_posthoc_rescue_final_2025_v1/multi_nwp_joint_g12_posthoc_rescue_recent097_2025.csv"
SOURCE_2024 = {
    "ecmwf": ROOT / "artifacts/external/openmeteo_ecmwf_ifs025_previous_runs_v1/ecmwf_ifs025_group_centroids_2024.parquet",
    "icon": ROOT / "artifacts/external/openmeteo_icon_global_previous_runs_v1/icon_global_group_centroids_2024.parquet",
    "gfs": ROOT / "artifacts/external/openmeteo_gfs_global_previous_runs_v1/gfs_global_group_centroids_2024.parquet",
}
SOURCE_2025_ROOT = ROOT / "artifacts/external/openmeteo_multi_nwp_g23_rescue_2025_v1"
SOURCE_2025 = {
    "ecmwf": SOURCE_2025_ROOT / "ecmwf_ifs025_group_centroids_2025.parquet",
    "icon": SOURCE_2025_ROOT / "icon_global_group_centroids_2025.parquet",
    "gfs": SOURCE_2025_ROOT / "gfs_global_group_centroids_2025.parquet",
}
SOURCE_2025_MANIFEST = SOURCE_2025_ROOT / "source_manifest.json"
TRAIN_CACHE = {group: ROOT / f"artifacts/cache/{group}_weather_train.parquet" for group in TARGET_COLS}
TEST_CACHE = {group: ROOT / f"artifacts/cache/{group}_weather_test.parquet" for group in TARGET_COLS}
HEAVY_GUARD = ROOT / "artifacts/locks/heavy_cpu_fit.pid.json"

PAIR_PARAMETERS = {
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
PREFIT_INDEX = pd.date_range("2024-03-09 00:00", "2024-04-30 23:00", freq="h", name="forecast_kst_dtm")
ROUTER_FIT_INDEX = pd.date_range("2024-05-01 00:00", "2024-06-30 23:00", freq="h", name="forecast_kst_dtm")
VALID_MODEL_INDEX = pd.date_range("2024-07-01 00:00", "2024-12-31 23:00", freq="h", name="forecast_kst_dtm")
VALID_SCORE_INDEX = pd.date_range("2024-07-01 00:00", "2025-01-01 00:00", freq="h", name="forecast_kst_dtm")
FINAL_MODEL_INDEX = pd.date_range("2025-01-01 01:00", "2025-12-31 23:00", freq="h", name="forecast_kst_dtm")
FINAL_SCORE_INDEX = pd.date_range("2025-01-01 01:00", "2026-01-01 00:00", freq="h", name="forecast_kst_dtm")
H1_LABEL_ROWS = 21_887
FULL_LABEL_ROWS = 26_304
ACTIVE_GROUPS = ("kpx_group_1", "kpx_group_2")
SLICES = {
    "H2": ("2024-07-01 00:00", "2025-01-01 00:00"),
    "Q3": ("2024-07-01 00:00", "2024-09-30 23:00"),
    "Q4": ("2024-10-01 00:00", "2025-01-01 00:00"),
    "Jul": ("2024-07-01 00:00", "2024-07-31 23:00"),
    "Aug": ("2024-08-01 00:00", "2024-08-31 23:00"),
    "Sep": ("2024-09-01 00:00", "2024-09-30 23:00"),
    "Oct": ("2024-10-01 00:00", "2024-10-31 23:00"),
    "Nov": ("2024-11-01 00:00", "2024-11-30 23:00"),
    "Dec": ("2024-12-01 00:00", "2025-01-01 00:00"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    frame.to_parquet(temporary, engine="pyarrow", compression="zstd", index=True)
    os.replace(temporary, path)


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return False
    ctypes.windll.kernel32.CloseHandle(handle)
    return True


def acquire_guard() -> dict[str, Any]:
    HEAVY_GUARD.parent.mkdir(parents=True, exist_ok=True)
    owner = {"pid": os.getpid(), "experiment_id": EXPERIMENT_ID, "created_utc": datetime.now(timezone.utc).isoformat()}
    for _ in range(2):
        try:
            descriptor = os.open(HEAVY_GUARD, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            current = json.loads(HEAVY_GUARD.read_text(encoding="utf-8"))
            if pid_alive(int(current["pid"])):
                raise RuntimeError(f"heavy guard held by live process: {current}")
            HEAVY_GUARD.unlink()
            continue
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(owner, stream, sort_keys=True)
            stream.write("\n")
        return owner
    raise RuntimeError("could not acquire heavy guard")


def release_guard(owner: Mapping[str, Any]) -> None:
    try:
        current = json.loads(HEAVY_GUARD.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return
    if int(current.get("pid", -1)) == int(owner["pid"]):
        HEAVY_GUARD.unlink()


def bounded_labels(rows: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    chunks: list[bytes] = []
    digest = hashlib.sha256()
    with LABEL_PATH.open("rb", buffering=0) as stream:
        for line_number in range(rows + 1):
            line = stream.readline()
            if not line:
                raise ValueError(f"label file ended at line {line_number}")
            chunks.append(line)
            digest.update(line)
    payload = b"".join(chunks)
    frame = pd.read_csv(io.BytesIO(payload), encoding="utf-8-sig")
    if tuple(frame.columns) != ("kst_dtm", *TARGET_COLS) or len(frame) != rows:
        raise ValueError("bounded label schema changed")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("kst_dtm"), errors="raise"), name="forecast_kst_dtm")
    return frame.astype(np.float64), {"rows": rows, "bytes": len(payload), "sha256": digest.hexdigest(), "path": str(LABEL_PATH)}


def load_source(path: Path, group: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["time"] = pd.to_datetime(frame["time"], errors="raise")
    frame = frame.loc[frame["group"] == group].drop(columns="group").set_index("time")
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    frame = frame.loc[index].astype(np.float64)
    if not frame.index.equals(index) or not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"incomplete source {path} for {group}")
    return frame


def load_config() -> dict[str, Any]:
    expected = SIDECAR_PATH.read_text(encoding="ascii").split()[0]
    if sha256(CONFIG_PATH) != expected:
        raise RuntimeError("preregister sidecar mismatch")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["status"] != "frozen_before_any_router_fit_prediction_or_new_validation_metric":
        raise RuntimeError("preregister status differs")
    for relative, wanted in config["bound_files"].items():
        path = ROOT / relative
        if path.stat().st_size != wanted["bytes"] or sha256(path) != wanted["sha256"]:
            raise RuntimeError(f"bound file changed: {relative}")
    return config


def metric_payload(actual: pd.Series, prediction: pd.Series, group: str) -> dict[str, Any]:
    value = group_metrics(actual, prediction, CAPACITY_KWH[group], group_name=group).as_dict()
    value["total_score"] = 0.5 * (value["one_minus_nmae"] + value["ficr"])
    return value


def comparisons(labels: pd.DataFrame, baseline: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {"groups": {}, "mixed": {}}
    for group in TARGET_COLS:
        output["groups"][group] = {}
        for name, bounds in SLICES.items():
            index = labels.index[(labels.index >= pd.Timestamp(bounds[0])) & (labels.index <= pd.Timestamp(bounds[1]))]
            base = metric_payload(labels.loc[index, group], baseline.loc[index, group], group)
            cand = metric_payload(labels.loc[index, group], candidate.loc[index, group], group)
            output["groups"][group][name] = {"baseline": base, "candidate": cand, "delta": {key: cand[key] - base[key] for key in ("total_score", "one_minus_nmae", "ficr")}}
    for name, bounds in SLICES.items():
        index = labels.index[(labels.index >= pd.Timestamp(bounds[0])) & (labels.index <= pd.Timestamp(bounds[1]))]
        base = score_details(labels.loc[index], baseline.loc[index]).as_dict()
        cand = score_details(labels.loc[index], candidate.loc[index]).as_dict()
        output["mixed"][name] = {"baseline": base, "candidate": cand, "delta": {key: cand[key] - base[key] for key in ("total_score", "one_minus_nmae", "ficr")}}
    return output


def promotion_gate(result: Mapping[str, Any], routed_fraction: Mapping[str, float], config: Mapping[str, Any]) -> dict[str, Any]:
    rules = config["promotion_gate"]
    h2 = result["mixed"]["H2"]["delta"]
    quarters = {name: result["mixed"][name]["delta"]["total_score"] for name in ("Q3", "Q4")}
    groups = {group: result["groups"][group]["H2"]["delta"]["total_score"] for group in ACTIVE_GROUPS}
    slices = {name: result["mixed"][name]["delta"]["total_score"] for name in SLICES}
    strength = h2["total_score"] >= rules["minimum_h2_total_score_delta"] or h2["ficr"] >= rules["minimum_h2_ficr_delta"]
    checks = {
        "strong_total_or_ficr": strength,
        "h2_nmae_floor": h2["one_minus_nmae"] >= rules["minimum_h2_one_minus_nmae_delta"],
        "q3_q4_floor": min(quarters.values()) >= rules["minimum_q3_q4_total_score_delta"],
        "active_group_floor": min(groups.values()) >= rules["minimum_active_group_h2_total_score_delta"],
        "positive_slice_breadth": sum(value > 0.0 for value in slices.values()) >= rules["minimum_positive_mixed_slices"],
        "materially_distinct_router": min(routed_fraction.values()) >= rules["minimum_non_full_action_fraction_each_active_group"],
    }
    return {"rules": dict(rules), "checks": checks, "pass": bool(all(checks.values())), "h2_delta": h2, "q3_q4_total_score_delta": quarters, "active_group_h2_total_score_delta": groups, "slice_total_score_delta": slices, "routed_fraction": dict(routed_fraction)}


def save_model(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(OUTPUT_DIR)
    config = load_config()
    OUTPUT_DIR.mkdir(parents=True)
    source_lock = OUTPUT_DIR / "source_lock_before_h1_label_parse_or_router_fit.json"
    atomic_json({
        "status": "all_2024_inputs_and_sources_locked_before_router_fit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "preregister": record(CONFIG_PATH),
        "inputs": [record(path) for path in [BASELINE_2024, ORIGINAL_INCREMENT_2024, *SOURCE_2024.values(), *TRAIN_CACHE.values()]],
        "public_feedback_contract": {"aggregate_family_confirmation_only": True, "group_or_time_inversion": False, "feedback_values_are_model_inputs": False},
        "physical_access": {"h2_label_cells": 0, "year_2025_value_cells": 0},
    }, source_lock)
    owner = acquire_guard()
    atexit.register(release_guard, owner)
    labels_h1, label_h1_record = bounded_labels(H1_LABEL_ROWS)
    atomic_json({"status": "h1_prefix_opened_after_source_lock", "source_lock": record(source_lock), "labels": label_h1_record}, OUTPUT_DIR / "h1_label_access.json")

    complete_2024 = PREFIT_INDEX.append(ROUTER_FIT_INDEX).append(VALID_MODEL_INDEX)
    base_raw = pd.read_parquet(BASELINE_2024)
    base_raw.index = pd.DatetimeIndex(base_raw.index, name="forecast_kst_dtm")
    base = base_raw.loc[VALID_SCORE_INDEX, list(TARGET_COLS)].astype(np.float64)
    for group in TARGET_COLS:
        base[group] = np.clip(0.97 * base[group], 0.0, 1.02 * CAPACITY_KWH[group])
    original_increment = pd.read_parquet(ORIGINAL_INCREMENT_2024)
    original_increment.index = pd.DatetimeIndex(original_increment.index, name="forecast_kst_dtm")
    original_increment = original_increment.loc[VALID_SCORE_INDEX, list(TARGET_COLS)].astype(np.float64)
    candidate = base.copy()
    diagnostics: list[pd.DataFrame] = []
    model_paths: list[Path] = []
    fit_records: dict[str, Any] = {}
    routed_fraction: dict[str, float] = {}
    for group in ACTIVE_GROUPS:
        canonical_all = pd.read_parquet(TRAIN_CACHE[group])
        canonical_all.index = pd.DatetimeIndex(canonical_all.index, name="forecast_kst_dtm")
        canonical = canonical_all.loc[complete_2024].astype(np.float32)
        sources = {name: load_source(SOURCE_2024[name], group, complete_2024) for name in SOURCE_ORDER}
        joint = build_joint_features(sources, canonical["cross__hub_ws_mean"])
        extended = pd.concat((canonical, joint), axis=1)
        prefit_target = labels_h1.loc[PREFIT_INDEX, group]
        eligible = prefit_target.notna() & (prefit_target >= 0.10 * CAPACITY_KWH[group])
        fit_rows = PREFIT_INDEX[eligible.to_numpy()]
        y_cf = (prefit_target.loc[fit_rows] / CAPACITY_KWH[group]).to_numpy(dtype=np.float64)
        control_model = LGBMRegressor(**PAIR_PARAMETERS).fit(canonical.loc[fit_rows], y_cf)
        extended_model = LGBMRegressor(**PAIR_PARAMETERS).fit(extended.loc[fit_rows], y_cf)
        early_control = np.clip(control_model.predict(canonical.loc[ROUTER_FIT_INDEX]), 0.0, 1.02)
        early_extended = np.clip(extended_model.predict(extended.loc[ROUTER_FIT_INDEX]), 0.0, 1.02)
        early_increment = pd.Series(early_extended - early_control, index=ROUTER_FIT_INDEX, name=group)
        train_base = np.clip(0.97 * base_raw.loc[ROUTER_FIT_INDEX, group], 0.0, 1.02 * CAPACITY_KWH[group])
        train_context = build_boundary_context(canonical.loc[ROUTER_FIT_INDEX], joint.loc[ROUTER_FIT_INDEX], train_base, early_increment, capacity_kwh=CAPACITY_KWH[group])
        router = MultiNWPBoundaryRouter().fit(train_context, labels_h1.loc[ROUTER_FIT_INDEX, group], capacity_kwh=CAPACITY_KWH[group])
        valid_context = build_boundary_context(canonical.loc[VALID_MODEL_INDEX], joint.loc[VALID_MODEL_INDEX], base.loc[VALID_MODEL_INDEX, group], original_increment.loc[VALID_MODEL_INDEX, group], capacity_kwh=CAPACITY_KWH[group])
        routed, diag = router.predict(valid_context, base.loc[VALID_MODEL_INDEX, group], capacity_kwh=CAPACITY_KWH[group])
        candidate.loc[VALID_MODEL_INDEX, group] = routed
        diag.insert(0, "group", group)
        diagnostics.append(diag)
        routed_fraction[group] = float(np.mean(diag["selected_alpha"].to_numpy() != 1.0))
        paths = [OUTPUT_DIR / f"models/{group}__early_control.joblib", OUTPUT_DIR / f"models/{group}__early_joint.joblib", OUTPUT_DIR / f"models/{group}__boundary_router.joblib"]
        for model, path in zip((control_model, extended_model, router), paths):
            save_model(model, path)
        model_paths.extend(paths)
        fit_records[group] = {"paired_prefit_rows": int(len(fit_rows)), "router": router.metadata(), "early_increment_mean_cf": float(early_increment.mean()), "early_increment_std_cf": float(early_increment.std(ddof=0)), "validation_selected_alpha_counts": {str(alpha): int(np.sum(diag["selected_alpha"].to_numpy() == alpha)) for alpha in ACTION_ALPHAS}}
        print(f"{group}: router fit={router.metadata()['rows_eligible']} routed_fraction={routed_fraction[group]:.4f}", flush=True)
    if not np.array_equal(candidate["kpx_group_3"].to_numpy(), base["kpx_group_3"].to_numpy()):
        raise AssertionError("G3 must remain exact identity")
    if not np.array_equal(candidate.loc[pd.Timestamp("2025-01-01 00:00")].to_numpy(), base.loc[pd.Timestamp("2025-01-01 00:00")].to_numpy()):
        raise AssertionError("terminal row must remain exact identity")
    candidate_path = OUTPUT_DIR / "validation/candidate_2024_h2.parquet"
    diagnostic_path = OUTPUT_DIR / "validation/router_diagnostics_2024_h2.parquet"
    atomic_parquet(candidate, candidate_path)
    atomic_parquet(pd.concat(diagnostics).sort_index(), diagnostic_path)
    candidate_lock = OUTPUT_DIR / "candidate_lock_before_h2_label_access_or_metric.json"
    atomic_json({"status": "single_router_candidate_locked_before_h2_labels", "candidate": record(candidate_path), "diagnostics": record(diagnostic_path), "models": [record(path) for path in model_paths], "fit_records": fit_records, "formula": "G1/G2 exact argmax estimated official utility among alpha=[1,.5,0] times frozen 0.25 joint increment; G3 scale097 identity", "public_feedback_subgroup_or_time_inversion": False, "year_2025_values_read": 0}, candidate_lock)

    labels_full, label_full_record = bounded_labels(FULL_LABEL_ROWS)
    atomic_json({"status": "h2_labels_opened_after_candidate_hash_lock", "candidate_lock": record(candidate_lock), "labels": label_full_record}, OUTPUT_DIR / "h2_label_access.json")
    labels = labels_full.loc[VALID_SCORE_INDEX, list(TARGET_COLS)]
    comparison = comparisons(labels, base, candidate)
    gate = promotion_gate(comparison, routed_fraction, config)
    results_path = OUTPUT_DIR / "validation_results.json"
    atomic_json({"experiment_id": EXPERIMENT_ID, "status": "promoted" if gate["pass"] else "rejected_no_final_csv", "selection_status": "posthoc_selection_unsafe_but_strict_forward_router_validation", "comparison": comparison, "gate": gate}, results_path)
    print(f"gate={gate}", flush=True)
    if not gate["pass"]:
        atomic_json({"status": "rejected_no_final_csv", "validation_results": record(results_path), "year_2025_values_read": 0}, OUTPUT_DIR / "rejection_lock.json")
        release_guard(owner)
        return

    # Conditional final stage: only now bind/read existing cutoff-safe 2025
    # sources and fit the same router on causally generated 2024 H2 actions.
    required_final = [BASELINE_2025, FINAL_INCREMENT_2025, SAMPLE_PATH, SOURCE_2025_MANIFEST, *SOURCE_2025.values(), *TEST_CACHE.values()]
    if not all(path.exists() for path in required_final):
        raise FileNotFoundError("promoted final-stage input missing")
    final_source_lock = OUTPUT_DIR / "final_source_lock_before_final_router_fit.json"
    atomic_json({"status": "promoted_2025_sources_locked_before_final_router_fit", "promotion_results": record(results_path), "inputs": [record(path) for path in required_final], "cutoff_contract": "hours01-13 previous_day1 else previous_day2; source manifests retained", "local_inference_only": True}, final_source_lock)
    base_2025_raw = pd.read_parquet(BASELINE_2025)
    base_2025_raw.index = pd.DatetimeIndex(base_2025_raw.index, name="forecast_kst_dtm")
    base_2025 = base_2025_raw.loc[FINAL_SCORE_INDEX, list(TARGET_COLS)].astype(np.float64)
    for group in TARGET_COLS:
        base_2025[group] = np.clip(0.97 * base_2025[group], 0.0, 1.02 * CAPACITY_KWH[group])
    final_increment = pd.read_parquet(FINAL_INCREMENT_2025)
    final_increment.index = pd.DatetimeIndex(final_increment.index, name="forecast_kst_dtm")
    final_increment = final_increment.loc[FINAL_SCORE_INDEX, list(TARGET_COLS)].astype(np.float64)
    final_prediction = base_2025.copy()
    final_diagnostics: list[pd.DataFrame] = []
    final_models: list[Path] = []
    for group in ACTIVE_GROUPS:
        train_weather_all = pd.read_parquet(TRAIN_CACHE[group])
        train_weather_all.index = pd.DatetimeIndex(train_weather_all.index, name="forecast_kst_dtm")
        train_weather = train_weather_all.loc[VALID_MODEL_INDEX].astype(np.float32)
        train_sources = {name: load_source(SOURCE_2024[name], group, VALID_MODEL_INDEX) for name in SOURCE_ORDER}
        train_joint = build_joint_features(train_sources, train_weather["cross__hub_ws_mean"])
        final_fit_context = build_boundary_context(train_weather, train_joint, base.loc[VALID_MODEL_INDEX, group], original_increment.loc[VALID_MODEL_INDEX, group], capacity_kwh=CAPACITY_KWH[group])
        final_router = MultiNWPBoundaryRouter().fit(final_fit_context, labels.loc[VALID_MODEL_INDEX, group], capacity_kwh=CAPACITY_KWH[group])
        apply_weather_all = pd.read_parquet(TEST_CACHE[group])
        apply_weather_all.index = pd.DatetimeIndex(apply_weather_all.index, name="forecast_kst_dtm")
        apply_weather = apply_weather_all.loc[FINAL_MODEL_INDEX].astype(np.float32)
        apply_sources = {name: load_source(SOURCE_2025[name], group, FINAL_MODEL_INDEX) for name in SOURCE_ORDER}
        apply_joint = build_joint_features(apply_sources, apply_weather["cross__hub_ws_mean"])
        apply_context = build_boundary_context(apply_weather, apply_joint, base_2025.loc[FINAL_MODEL_INDEX, group], final_increment.loc[FINAL_MODEL_INDEX, group], capacity_kwh=CAPACITY_KWH[group])
        routed, diag = final_router.predict(apply_context, base_2025.loc[FINAL_MODEL_INDEX, group], capacity_kwh=CAPACITY_KWH[group])
        final_prediction.loc[FINAL_MODEL_INDEX, group] = routed
        diag.insert(0, "group", group)
        final_diagnostics.append(diag)
        path = OUTPUT_DIR / f"final/models/{group}__boundary_router.joblib"
        save_model(final_router, path)
        final_models.append(path)
        print(f"final {group}: alpha_counts={diag['selected_alpha'].value_counts().to_dict()}", flush=True)
    if not np.array_equal(final_prediction["kpx_group_3"].to_numpy(), base_2025["kpx_group_3"].to_numpy()):
        raise AssertionError("final G3 must remain exact base identity")
    if not np.array_equal(final_prediction.loc[pd.Timestamp("2026-01-01 00:00")].to_numpy(), base_2025.loc[pd.Timestamp("2026-01-01 00:00")].to_numpy()):
        raise AssertionError("final terminal row must remain exact base identity")
    final_prediction_path = OUTPUT_DIR / "final/prediction_kwh_2025.parquet"
    final_diagnostic_path = OUTPUT_DIR / "final/router_diagnostics_2025.parquet"
    atomic_parquet(final_prediction, final_prediction_path)
    atomic_parquet(pd.concat(final_diagnostics).sort_index(), final_diagnostic_path)
    sample = pd.read_csv(SAMPLE_PATH, encoding="utf-8-sig", dtype={"forecast_id": str, "forecast_kst_dtm": str})
    sample_index = pd.DatetimeIndex(pd.to_datetime(sample["forecast_kst_dtm"], errors="raise"), name="forecast_kst_dtm")
    if tuple(sample.columns) != ("forecast_id", "forecast_kst_dtm", *TARGET_COLS) or not sample_index.equals(FINAL_SCORE_INDEX):
        raise ValueError("sample schema/order changed")
    output = sample.copy()
    rounded = final_prediction.round(6)
    for group in TARGET_COLS:
        output[group] = rounded[group].to_numpy(dtype=np.float64)
    csv_path = OUTPUT_DIR / "final/multi_nwp_ficr_boundary_router_g12_recent097_2025.csv"
    output.to_csv(csv_path, index=False, encoding="utf-8-sig", lineterminator="\n", float_format="%.6f")
    reloaded = pd.read_csv(csv_path, encoding="utf-8-sig", dtype={"forecast_id": str, "forecast_kst_dtm": str})
    values = reloaded.loc[:, list(TARGET_COLS)].to_numpy(dtype=np.float64)
    if csv_path.read_bytes()[:3] != b"\xef\xbb\xbf" or len(reloaded) != 8760 or not np.isfinite(values).all():
        raise AssertionError("final CSV contract failed")
    if not reloaded["forecast_id"].equals(sample["forecast_id"]) or not reloaded["forecast_kst_dtm"].equals(sample["forecast_kst_dtm"]):
        raise AssertionError("final CSV identifiers/order differ")
    if not np.array_equal(values, rounded.to_numpy(dtype=np.float64)):
        raise AssertionError("final CSV numeric replay differs")
    parent = pd.read_csv(PUBLIC_SUBMITTED_PARENT, encoding="utf-8-sig")
    parent_values = parent.loc[:, list(TARGET_COLS)].to_numpy(dtype=np.float64)
    distinct_cells = int(np.sum(values != parent_values))
    if distinct_cells < 100:
        raise AssertionError("router final is not materially distinct from submitted parent")
    final_audit = OUTPUT_DIR / "final/audit.json"
    atomic_json({"status": "ready", "validation_results": record(results_path), "final_source_lock": record(final_source_lock), "models": [record(path) for path in final_models], "prediction": record(final_prediction_path), "diagnostics": record(final_diagnostic_path), "csv": record(csv_path), "csv_contract": {"rows": len(reloaded), "utf8_bom": True, "finite": True, "sample_id_time_identity": True, "six_decimal_replay": True}, "distinct_cells_from_submitted_parent": distinct_cells, "public_feedback_used_only_as_family_level_confirmation": True, "public_group_or_time_inversion": False, "local_model_inference_only": True}, final_audit)
    manifest = OUTPUT_DIR / "manifest.json"
    output_files = [source_lock, OUTPUT_DIR / "h1_label_access.json", *model_paths, candidate_path, diagnostic_path, candidate_lock, OUTPUT_DIR / "h2_label_access.json", results_path, final_source_lock, *final_models, final_prediction_path, final_diagnostic_path, csv_path, final_audit]
    atomic_json({"experiment_id": EXPERIMENT_ID, "status": "promoted_and_final_csv_ready", "preregister": record(CONFIG_PATH), "outputs_excluding_manifest_and_sidecar": [record(path) for path in output_files], "csv": record(csv_path)}, manifest)
    (OUTPUT_DIR / "manifest.sha256").write_text(f"{sha256(manifest)}  manifest.json\n", encoding="ascii")
    release_guard(owner)
    print(f"CSV_READY {record(csv_path)} distinct_cells={distinct_cells}", flush=True)


if __name__ == "__main__":
    main()

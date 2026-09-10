"""Run the preregistered Copernicus DEM paired-increment Stage1 diagnostic."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence
import uuid

import joblib
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts import run_shared_q07_multiseed as shared  # noqa: E402
from src.copernicus_dem_exposure import (  # noqa: E402
    EXTENDED_COLUMNS,
    build_dynamic_exposure_features,
    parse_turbines,
)
from src.jma_paired_increment import eligible_rows, fit_direct_model, predict_cf  # noqa: E402
from src.manifest import describe_file, git_state, package_versions, sha256_file, utc_now  # noqa: E402
from src.metric import CAPACITY_KWH, TARGET_COLS, group_metrics, score_details  # noqa: E402


CONFIG_SHA = "21e2a6aaee02d11c00d2b8a266e30917efa4f5a505417ceadb9f618fe7d2443b"
STATIC_TABLE_SHA = "71945cab5a07c328eabc5a8fd01642eaffb68194e23efe921f1ba53317482fae"
STATIC_AUDIT_SHA = "c8272513b6d1fed5883bb1b7e113bebb8cfc5ca3288f22a7c53c5adc62d0ff6d"
EXPERIMENT_ID = "copernicus_dem_directional_exposure_paired_increment_strict_v1"
TRANSFER_WEIGHT = 0.05
HEAVY_GUARD = PROJECT_DIR / "artifacts/locks/heavy_cpu_fit.pid.json"

YEAR_2022 = pd.date_range("2022-01-01 01:00", "2023-01-01 00:00", freq="h", name="forecast_kst_dtm")
YEAR_2023 = pd.date_range("2023-01-01 01:00", "2024-01-01 00:00", freq="h", name="forecast_kst_dtm")
PRE2024 = pd.date_range("2022-01-01 01:00", "2024-01-01 00:00", freq="h", name="forecast_kst_dtm")
G3_H1 = pd.date_range("2023-01-01 01:00", "2023-07-01 00:00", freq="h", name="forecast_kst_dtm")
G3_H2 = pd.date_range("2023-07-01 01:00", "2024-01-01 00:00", freq="h", name="forecast_kst_dtm")

LABEL_SLICES = {
    "g12_fit": {
        "start": 48,
        "end": 340783,
        "parser_bytes": 340783,
        "parser_sha": "734bfbd7b8e509e300e53b4613b4f36b8fc3c303dbf82f6c5636323fee45ee7b",
        "nrows": 8760,
        "columns": ("kst_dtm", "kpx_group_1", "kpx_group_2"),
    },
    "g3_fit": {
        "start": 340783,
        "end": 541848,
        "parser_bytes": 201113,
        "parser_sha": "b1d0db12160f8b691f1b552be01df2e62cc8c7affc44dcc97aa91271fa21ff2b",
        "nrows": 4344,
        "columns": ("kst_dtm", "kpx_group_3"),
    },
    "g12_score": {
        "start": 340783,
        "end": 742551,
        "parser_bytes": 401816,
        "parser_sha": "88684b6d1c2ce290b2bc22a78b6da6ea8aaa412d85d6477b562a1420066b574b",
        "nrows": 8760,
        "columns": ("kst_dtm", "kpx_group_1", "kpx_group_2"),
    },
    "g3_score": {
        "start": 541848,
        "end": 742551,
        "parser_bytes": 200751,
        "parser_sha": "2229a72840ded9d8bec17a63d4f699f69254864f2c089a14e6d0fa9421044bb8",
        "nrows": 4416,
        "columns": ("kst_dtm", "kpx_group_3"),
    },
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path(r"data/local/open"))
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs/copernicus_dem_directional_exposure_paired_increment_preregister_v1.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_DIR / "artifacts/postgate/copernicus_dem_directional_exposure_paired_increment_strict_v1",
    )
    return parser.parse_args(argv)


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
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(_json_ready(payload), stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temp)
    os.replace(temp, path)


def _atomic_joblib(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    joblib.dump(model, temp)
    os.replace(temp, path)


def _verify(path: Path, *, size: int | None = None, sha: str | None = None) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if size is not None and path.stat().st_size != size:
        raise AssertionError(f"size differs: {path}")
    if sha is not None and sha256_file(path) != sha:
        raise AssertionError(f"SHA differs: {path}")
    return path


def verify_config(args: argparse.Namespace) -> dict[str, Any]:
    _verify(args.config, sha=CONFIG_SHA)
    sidecar = args.config.with_suffix(".sha256")
    expected = f"{CONFIG_SHA}  {args.config.name}\n"
    if sidecar.read_text(encoding="utf-8") != expected:
        raise AssertionError("config sidecar differs")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["experiment_id"] != EXPERIMENT_ID:
        raise AssertionError("experiment id differs")
    if config["paired_model"]["candidate_id"] != "copdem_directional_exposure_w005":
        raise AssertionError("candidate differs")
    if float(config["paired_model"]["transfer_weight"]) != TRANSFER_WEIGHT:
        raise AssertionError("weight differs")
    if tuple(config["dynamic_feature_algorithm"]["extended_features_after_612_in_exact_order"]) != EXTENDED_COLUMNS:
        raise AssertionError("terrain features differ")
    return config


def _read_label_slice(path: Path, key: str, expected_index: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = LABEL_SLICES[key]
    with path.open("rb") as stream:
        header = stream.readline()
        if len(header) != 48:
            raise AssertionError("label header differs")
        stream.seek(int(spec["start"]))
        body = stream.read(int(spec["end"]) - int(spec["start"]))
    payload = header + body
    if len(payload) != int(spec["parser_bytes"]) or hashlib.sha256(payload).hexdigest() != spec["parser_sha"]:
        raise AssertionError(f"label bounded slice differs: {key}")
    frame = pd.read_csv(io.BytesIO(payload), encoding="utf-8-sig", usecols=list(spec["columns"]), nrows=int(spec["nrows"]))
    times = pd.to_datetime(frame.pop("kst_dtm"), errors="raise")
    frame.index = pd.DatetimeIndex(times, name="forecast_kst_dtm")
    if tuple(("kst_dtm", *frame.columns)) != spec["columns"] or not frame.index.equals(expected_index):
        raise AssertionError(f"label schema/index differs: {key}")
    frame = frame.astype(np.float64)
    return frame, {
        "slice": key,
        "source_start_byte": spec["start"],
        "source_end_byte": spec["end"],
        "parser_bytes": len(payload),
        "parser_sha256": hashlib.sha256(payload).hexdigest(),
        "label_value_cells_materialized": int(frame.size),
    }


def read_fit_labels(args: argparse.Namespace, config: Mapping[str, Any]) -> tuple[dict[str, pd.Series], dict[str, Any]]:
    spec = config["official_data_and_lineage"]["label_file"]
    path = _verify(Path(spec["path"]), size=int(spec["bytes"]), sha=str(spec["sha256"]))
    g12, e12 = _read_label_slice(path, "g12_fit", YEAR_2022)
    g3, e3 = _read_label_slice(path, "g3_fit", G3_H1)
    return {
        "kpx_group_1": g12["kpx_group_1"],
        "kpx_group_2": g12["kpx_group_2"],
        "kpx_group_3": g3["kpx_group_3"],
    }, {"g12": e12, "g3": e3, "score_label_value_cells_materialized": 0}


def verify_candidate_lock(record: Mapping[str, Any]) -> None:
    path = _verify(Path(record["path"]), size=int(record["size_bytes"]), sha=str(record["sha256"]))
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("lock_kind") != "stage1_candidate_before_score_labels":
        raise AssertionError("candidate lock kind differs")
    if payload.get("config_sha256") != CONFIG_SHA or payload.get("score_label_value_cells_before_lock") != 0:
        raise AssertionError("candidate lock identity/order differs")


def read_score_labels(
    args: argparse.Namespace, config: Mapping[str, Any], *, candidate_lock: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    # This verification must complete before the label file is resolved or opened.
    verify_candidate_lock(candidate_lock)
    spec = config["official_data_and_lineage"]["label_file"]
    path = _verify(Path(spec["path"]), size=int(spec["bytes"]), sha=str(spec["sha256"]))
    g12, e12 = _read_label_slice(path, "g12_score", YEAR_2023)
    g3, e3 = _read_label_slice(path, "g3_score", G3_H2)
    labels = pd.DataFrame(np.nan, index=YEAR_2023, columns=TARGET_COLS, dtype=np.float64)
    labels.loc[YEAR_2023, list(TARGET_COLS[:2])] = g12.to_numpy(np.float64)
    labels.loc[G3_H2, "kpx_group_3"] = g3["kpx_group_3"].to_numpy(np.float64)
    return labels, {
        "candidate_lock_verified_before_source_open": True,
        "g12": e12,
        "g3": e3,
        "label_value_cells_materialized": int(e12["label_value_cells_materialized"] + e3["label_value_cells_materialized"]),
    }


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
    def __init__(self) -> None:
        self.token = uuid.uuid4().hex
        self.acquired = False

    def __enter__(self) -> "HeavyFitGuard":
        HEAVY_GUARD.parent.mkdir(parents=True, exist_ok=True)
        if HEAVY_GUARD.exists():
            record = json.loads(HEAVY_GUARD.read_text(encoding="utf-8"))
            if _pid_alive(int(record.get("pid", -1))):
                raise RuntimeError(f"heavy guard held by live PID {record.get('pid')}: {record}")
            stale = HEAVY_GUARD.with_name(f"heavy_cpu_fit.stale-pid{record.get('pid','unknown')}-{uuid.uuid4().hex}.json")
            os.replace(HEAVY_GUARD, stale)
        payload = {
            "schema_version": 1,
            "pid": os.getpid(),
            "token": self.token,
            "experiment_id": EXPERIMENT_ID,
            "stage": "stage1",
            "created_utc": utc_now(),
        }
        fd = os.open(HEAVY_GUARD, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        self.acquired = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if not self.acquired or not HEAVY_GUARD.exists():
            return
        record = json.loads(HEAVY_GUARD.read_text(encoding="utf-8"))
        if (
            record.get("experiment_id") == EXPERIMENT_ID
            and record.get("token") == self.token
            and int(record.get("pid", -1)) == os.getpid()
        ):
            HEAVY_GUARD.unlink()


def _load_baseline(config: Mapping[str, Any]) -> pd.DataFrame:
    g12_spec = config["official_data_and_lineage"]["stage1_g12_baseline"]
    g12 = pd.read_parquet(_verify(PROJECT_DIR / g12_spec["path"], size=int(g12_spec["bytes"]), sha=str(g12_spec["sha256"]))).astype(np.float64)
    g12.index = pd.DatetimeIndex(g12.index, name="forecast_kst_dtm")
    if not g12.index.equals(YEAR_2023) or tuple(g12.columns) != TARGET_COLS[:2]:
        raise AssertionError("G1/G2 baseline differs")
    g3_spec = config["official_data_and_lineage"]["stage1_g3_components"]
    raw = pd.read_parquet(_verify(PROJECT_DIR / g3_spec["path"], size=int(g3_spec["bytes"]), sha=str(g3_spec["sha256"]))).astype(np.float64)
    raw.index = pd.DatetimeIndex(raw.index, name="forecast_kst_dtm")
    if not raw.index.equals(G3_H2):
        raise AssertionError("G3 baseline components differ")
    weighted = 0.20 * raw["q07"] + 0.075 * raw["shared_l1"] + 0.425 * raw["shared_q07"] + 0.025 * raw["top200q07"] + 0.275 * raw["ewq06"]
    baseline = pd.DataFrame(np.nan, index=YEAR_2023, columns=TARGET_COLS, dtype=np.float64)
    baseline.loc[:, list(TARGET_COLS[:2])] = g12.to_numpy(np.float64)
    baseline.loc[G3_H2, "kpx_group_3"] = np.clip(1.25 * weighted.to_numpy(np.float64) - 1200.0, 0.0, 1.02 * CAPACITY_KWH["kpx_group_3"])
    return baseline


def _segments() -> dict[str, pd.DatetimeIndex]:
    make = lambda a, b: pd.date_range(a, b, freq="h", name="forecast_kst_dtm")
    return {
        "full": YEAR_2023,
        "H1": make("2023-01-01 01:00", "2023-07-01 00:00"),
        "H2": G3_H2,
        "Q1": make("2023-01-01 01:00", "2023-04-01 00:00"),
        "Q2": make("2023-04-01 01:00", "2023-07-01 00:00"),
        "Q3": make("2023-07-01 01:00", "2023-10-01 00:00"),
        "Q4": make("2023-10-01 01:00", "2024-01-01 00:00"),
    }


def _group_score(actual: pd.Series, pred: pd.Series, group: str) -> dict[str, Any]:
    record = group_metrics(actual, pred, CAPACITY_KWH[group], group_name=group).as_dict()
    record["total_score"] = 0.5 * (record["one_minus_nmae"] + record["ficr"])
    return record


def _comparison(actual: pd.Series, base: pd.Series, candidate: pd.Series, group: str) -> dict[str, Any]:
    a = _group_score(actual, base, group)
    b = _group_score(actual, candidate, group)
    return {
        "baseline": a,
        "candidate": b,
        "delta_total_score": b["total_score"] - a["total_score"],
        "delta_one_minus_nmae": b["one_minus_nmae"] - a["one_minus_nmae"],
        "delta_ficr": b["ficr"] - a["ficr"],
    }


def _mixed_comparison(actual: pd.DataFrame, base: pd.DataFrame, candidate: pd.DataFrame, groups: Sequence[str]) -> dict[str, Any]:
    a = score_details(actual, base, target_cols=groups).as_dict()
    b = score_details(actual, candidate, target_cols=groups).as_dict()
    return {
        "groups_with_available_labels": list(groups),
        "baseline": a,
        "candidate": b,
        "delta_total_score": b["total_score"] - a["total_score"],
        "delta_one_minus_nmae": b["one_minus_nmae"] - a["one_minus_nmae"],
        "delta_ficr": b["ficr"] - a["ficr"],
    }


def _source_closure(args: argparse.Namespace, config: Mapping[str, Any]) -> dict[str, Any]:
    paths = [
        args.config,
        args.config.with_suffix(".sha256"),
        Path(__file__),
        PROJECT_DIR / "src/copernicus_dem_exposure.py",
        PROJECT_DIR / "src/jma_paired_increment.py",
        PROJECT_DIR / "src/features.py",
        PROJECT_DIR / "src/metric.py",
        PROJECT_DIR / "src/manifest.py",
        PROJECT_DIR / "scripts/run_shared_q07_multiseed.py",
        PROJECT_DIR / "tests/test_copernicus_dem_directional_exposure.py",
        PROJECT_DIR / config["official_rules_and_source"]["source_feasibility_audit"]["path"],
        PROJECT_DIR / config["official_rules_and_source"]["source_manifest"]["path"],
        PROJECT_DIR / "artifacts/external/copernicus_dem_glo30_2021_terrain_v1/features_v1/static_directional_terrain_17x16.parquet",
        PROJECT_DIR / "artifacts/external/copernicus_dem_glo30_2021_terrain_v1/features_v1/static_feature_audit.json",
    ]
    return {str(path.resolve()): describe_file(_verify(path)) for path in paths}


def _write_manifest(args: argparse.Namespace, status: str, closure: Mapping[str, Any]) -> None:
    if list(args.out_dir.rglob("*.csv")):
        raise AssertionError("Stage1 must not create CSV")
    files = [
        describe_file(path)
        for path in sorted(args.out_dir.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    ]
    _write_json(
        args.out_dir / "manifest.json",
        {
            "schema_version": 1,
            "artifact_type": EXPERIMENT_ID,
            "status": status,
            "created_utc": utc_now(),
            "config_sha256": CONFIG_SHA,
            "source_closure": closure,
            "files": files,
            "runtime": {"packages": package_versions()},
            "git": git_state(PROJECT_DIR),
            "2024_label_value_cells_read": 0,
            "2025_contest_values_read": 0,
            "csv_created": False,
        },
    )


def run_stage1(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    static_path = _verify(
        PROJECT_DIR / "artifacts/external/copernicus_dem_glo30_2021_terrain_v1/features_v1/static_directional_terrain_17x16.parquet",
        sha=STATIC_TABLE_SHA,
    )
    _verify(
        PROJECT_DIR / "artifacts/external/copernicus_dem_glo30_2021_terrain_v1/features_v1/static_feature_audit.json",
        sha=STATIC_AUDIT_SHA,
    )
    args.out_dir.mkdir(parents=True)
    shutil.copyfile(args.config, args.out_dir / "preregister_v1.json")
    shutil.copyfile(args.config.with_suffix(".sha256"), args.out_dir / "preregister_v1.sha256")
    closure = _source_closure(args, config)
    _write_json(
        args.out_dir / "stage1_prescore_lock.json",
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "config_sha256": CONFIG_SHA,
            "source_closure": closure,
            "static_table": describe_file(static_path),
            "candidate_models_fit_before_lock": 0,
            "candidate_prediction_value_cells_before_lock": 0,
            "metric_values_before_lock": 0,
            "score_label_value_cells_before_lock": 0,
            "2024_label_value_cells": 0,
            "2025_contest_value_cells": 0,
        },
    )

    # Target-free, physically bounded NWP features are materialized before fit labels.
    features, raw_contract = shared._read_stage1_raw_features(args.raw_dir, pd.DataFrame(index=PRE2024))
    static_table = pd.read_parquet(static_path)
    turbines = parse_turbines(config["turbines"]["rows"])
    terrain: dict[str, pd.DataFrame] = {}
    terrain_files: dict[str, Any] = {}
    for group in TARGET_COLS:
        terrain[group] = build_dynamic_exposure_features(features[group], static_table, turbines, group=group)
        path = args.out_dir / f"stage1/features/{group}_terrain.parquet"
        _atomic_parquet(terrain[group], path)
        terrain_files[group] = describe_file(path)

    fit_labels, fit_label_evidence = read_fit_labels(args, config)
    baseline = _load_baseline(config)
    baseline_path = args.out_dir / "stage1/baseline_corrected_v3.parquet"
    _atomic_parquet(baseline, baseline_path)
    train_indexes = {TARGET_COLS[0]: YEAR_2022, TARGET_COLS[1]: YEAR_2022, TARGET_COLS[2]: G3_H1}
    apply_indexes = {TARGET_COLS[0]: YEAR_2023, TARGET_COLS[1]: YEAR_2023, TARGET_COLS[2]: G3_H2}
    candidate = baseline.copy()
    training: dict[str, Any] = {}
    with HeavyFitGuard():
        for group in TARGET_COLS:
            train = train_indexes[group]
            apply = apply_indexes[group]
            control_train = features[group].loc[train]
            control_apply = features[group].loc[apply]
            actual = fit_labels[group]
            mask = eligible_rows(actual, CAPACITY_KWH[group])
            control_model, control_meta = fit_direct_model(
                control_train, actual, capacity_kwh=CAPACITY_KWH[group], eligible=mask
            )
            extended_train = pd.concat([control_train, terrain[group].loc[train]], axis=1)
            extended_apply = pd.concat([control_apply, terrain[group].loc[apply]], axis=1)
            if tuple(extended_train.columns[-10:]) != EXTENDED_COLUMNS or extended_train.shape[1] != 622:
                raise AssertionError("extended feature order/count differs")
            extended_model, extended_meta = fit_direct_model(
                extended_train, actual, capacity_kwh=CAPACITY_KWH[group], eligible=mask
            )
            control_cf = predict_cf(control_model, control_apply)
            extended_cf = predict_cf(extended_model, extended_apply)
            increment = extended_cf - control_cf
            values = np.clip(
                baseline.loc[apply, group].to_numpy(np.float64)
                + TRANSFER_WEIGHT * CAPACITY_KWH[group] * increment,
                0.0,
                1.02 * CAPACITY_KWH[group],
            )
            candidate.loc[apply, group] = values
            control_path = args.out_dir / f"stage1/models/{group}_control.joblib"
            extended_path = args.out_dir / f"stage1/models/{group}_extended.joblib"
            _atomic_joblib(control_model, control_path)
            _atomic_joblib(extended_model, extended_path)
            detail_path = args.out_dir / f"stage1/diagnostics/{group}.parquet"
            _atomic_parquet(
                pd.DataFrame(
                    {
                        "control_cf": control_cf,
                        "extended_cf": extended_cf,
                        "increment_cf": increment,
                        "baseline_kwh": baseline.loc[apply, group].to_numpy(np.float64),
                        "candidate_kwh": values,
                    },
                    index=apply,
                ),
                detail_path,
            )
            training[group] = {
                "control": {**control_meta, "model": describe_file(control_path)},
                "extended": {**extended_meta, "model": describe_file(extended_path)},
                "same_fit_rows_order_target_seed_parameters": True,
                "eligible_mask_sha256": hashlib.sha256(np.ascontiguousarray(mask.astype(np.uint8)).tobytes()).hexdigest(),
                "diagnostics": describe_file(detail_path),
            }

    candidate_path = args.out_dir / "stage1/candidate_copdem_directional_exposure_w005.parquet"
    _atomic_parquet(candidate, candidate_path)
    lock_path = args.out_dir / "stage1_candidate_before_score_labels_lock.json"
    _write_json(
        lock_path,
        {
            "schema_version": 1,
            "lock_kind": "stage1_candidate_before_score_labels",
            "created_utc": utc_now(),
            "config_sha256": CONFIG_SHA,
            "source_closure": closure,
            "static_table": describe_file(static_path),
            "terrain_feature_files": terrain_files,
            "baseline": describe_file(baseline_path),
            "candidate": describe_file(candidate_path),
            "training": training,
            "score_label_value_cells_before_lock": 0,
            "candidate_metric_values_before_lock": 0,
            "2024_label_value_cells_before_lock": 0,
            "2025_contest_value_cells_before_lock": 0,
        },
    )
    candidate_lock = describe_file(lock_path)
    score_labels, score_evidence = read_score_labels(args, config, candidate_lock=candidate_lock)

    segments = _segments()
    group_records: dict[str, Any] = {}
    total_deltas: list[float] = []
    full_component_deltas: list[float] = []
    for group in TARGET_COLS[:2]:
        group_records[group] = {}
        for name in ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4"):
            idx = segments[name]
            record = _comparison(score_labels.loc[idx, group], baseline.loc[idx, group], candidate.loc[idx, group], group)
            group_records[group][name] = record
            total_deltas.append(float(record["delta_total_score"]))
            if name == "full":
                full_component_deltas.extend([float(record["delta_one_minus_nmae"]), float(record["delta_ficr"])])
    group_records[TARGET_COLS[2]] = {}
    for name, idx in (("full_H2", G3_H2), ("Q3", segments["Q3"]), ("Q4", segments["Q4"])):
        record = _comparison(score_labels.loc[idx, TARGET_COLS[2]], baseline.loc[idx, TARGET_COLS[2]], candidate.loc[idx, TARGET_COLS[2]], TARGET_COLS[2])
        group_records[TARGET_COLS[2]][name] = record
        total_deltas.append(float(record["delta_total_score"]))
        if name == "full_H2":
            full_component_deltas.extend([float(record["delta_one_minus_nmae"]), float(record["delta_ficr"])])
    if len(total_deltas) != 17:
        raise AssertionError("registered group comparison count differs")

    mixed_records: dict[str, Any] = {}
    for name, idx in segments.items():
        groups = TARGET_COLS if name in ("full", "H2", "Q3", "Q4") else TARGET_COLS[:2]
        record = _mixed_comparison(score_labels.loc[idx], baseline.loc[idx], candidate.loc[idx], groups)
        mixed_records[name] = record
        total_deltas.append(float(record["delta_total_score"]))
        if name == "full":
            full_component_deltas.extend([float(record["delta_one_minus_nmae"]), float(record["delta_ficr"])])
    if len(total_deltas) != 24 or len(full_component_deltas) != 8:
        raise AssertionError("registered gate vector differs")
    passed = all(delta > 0.0 for delta in total_deltas) and all(delta >= 0.0 for delta in full_component_deltas)
    result_path = args.out_dir / "stage1_results.json"
    _write_json(
        result_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "config_sha256": CONFIG_SHA,
            "candidate_before_score_labels_lock": candidate_lock,
            "score_label_evidence_after_candidate_lock": score_evidence,
            "group_comparisons": group_records,
            "mixed_comparisons": mixed_records,
            "registered_total_score_deltas": total_deltas,
            "registered_full_component_deltas": full_component_deltas,
            "stage1_passed": passed,
            "stage2_allowed": passed,
            "raw_feature_contract": raw_contract,
            "2024_label_value_cells_read": 0,
            "2025_contest_values_read": 0,
        },
    )
    _write_json(
        args.out_dir / "stage1_selection_lock.json",
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "config_sha256": CONFIG_SHA,
            "candidate_before_score_labels_lock": candidate_lock,
            "stage1_results": describe_file(result_path),
            "stage1_passed": passed,
            "stage2_allowed": passed,
            "2024_label_value_cells_read": 0,
            "2025_contest_values_read": 0,
        },
    )
    if closure != _source_closure(args, config):
        raise AssertionError("source closure changed during execution")
    _write_manifest(args, "STAGE1_PASS" if passed else "REJECTED_STAGE1", closure)
    print(
        json.dumps(
            {
                "stage1_passed": passed,
                "min_total_delta": min(total_deltas),
                "min_full_component_delta": min(full_component_deltas),
                "manifest": describe_file(args.out_dir / "manifest.json"),
            }
        ),
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.raw_dir = args.raw_dir.resolve()
    args.config = args.config.resolve()
    args.out_dir = args.out_dir.resolve()
    config = verify_config(args)
    run_stage1(args, config)


if __name__ == "__main__":
    main()


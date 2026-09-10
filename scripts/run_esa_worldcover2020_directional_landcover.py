"""Run the preregistered ESA WorldCover paired-increment Stage1 gate."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
import uuid

import joblib
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_DIR))

from scripts import run_copernicus_dem_directional_exposure as protocol  # noqa: E402
from src.copernicus_dem_exposure import parse_turbines  # noqa: E402
from src.esa_worldcover_landcover import (  # noqa: E402
    EXTENDED_COLUMNS,
    build_dynamic_landcover_features,
)
from src.jma_paired_increment import eligible_rows, fit_direct_model, predict_cf  # noqa: E402
from src.manifest import describe_file, git_state, package_versions, sha256_file, utc_now  # noqa: E402
from src.metric import CAPACITY_KWH, TARGET_COLS  # noqa: E402


EXPERIMENT_ID = "esa_worldcover2020_directional_landcover_paired_increment_strict_v2"
CONFIG_SHA = "6d0845fa36983cadfc8fd422b9b1b419d8e934d7e90b0f7e15fdbc9d48cd2c7b"
LINEAGE_SHA = "21e2a6aaee02d11c00d2b8a266e30917efa4f5a505417ceadb9f618fe7d2443b"
STATIC_SHA = "059285f4ef99d11e3adf57f89897ca225d4820cd16917865779750362968363f"
TRANSFER_WEIGHT = 0.25
YEAR_2022 = protocol.YEAR_2022
YEAR_2023 = protocol.YEAR_2023
PRE2024 = protocol.PRE2024
G3_H1 = protocol.G3_H1
G3_H2 = protocol.G3_H2
HEAVY_GUARD = PROJECT_DIR / "artifacts/locks/heavy_cpu_fit.pid.json"


def _args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path(r"data/local/open"))
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR
        / "configs/esa_worldcover2020_directional_landcover_paired_increment_preregister_v2.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_DIR
        / "artifacts/postgate/esa_worldcover2020_directional_landcover_paired_increment_strict_v2",
    )
    return parser.parse_args(argv)


def _verify(path: Path, *, size: int | None = None, sha: str | None = None) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if size is not None and path.stat().st_size != size:
        raise AssertionError(f"size differs: {path}")
    if sha is not None and sha256_file(path) != sha:
        raise AssertionError(f"SHA differs: {path}")
    return path


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    protocol._write_json(path, payload)


def _load_config(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    _verify(args.config, sha=CONFIG_SHA)
    expected = f"{CONFIG_SHA}  {args.config.name}\n"
    if args.config.with_suffix(".sha256").read_text(encoding="utf-8") != expected:
        raise AssertionError("preregister sidecar differs")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["experiment_id"] != EXPERIMENT_ID:
        raise AssertionError("experiment identity differs")
    if config["paired_model"]["candidate_id"] != "worldcover2020_directional_landcover_w025":
        raise AssertionError("candidate identity differs")
    if float(config["paired_model"]["transfer_weight"]) != TRANSFER_WEIGHT:
        raise AssertionError("transfer weight differs")
    lineage_spec = config["lineage_parent"]
    lineage_path = _verify(
        PROJECT_DIR / lineage_spec["path"],
        size=int(lineage_spec["bytes"]),
        sha=LINEAGE_SHA,
    )
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    return config, lineage


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
                raise RuntimeError(f"heavy guard held by live PID: {record}")
            stale = HEAVY_GUARD.with_name(
                f"heavy_cpu_fit.stale-pid{record.get('pid','unknown')}-{uuid.uuid4().hex}.json"
            )
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


def _fit_labels(lineage: Mapping[str, Any]) -> tuple[dict[str, pd.Series], dict[str, Any]]:
    spec = lineage["official_data_and_lineage"]["label_file"]
    path = _verify(Path(spec["path"]), size=int(spec["bytes"]), sha=str(spec["sha256"]))
    g12, e12 = protocol._read_label_slice(path, "g12_fit", YEAR_2022)
    g3, e3 = protocol._read_label_slice(path, "g3_fit", G3_H1)
    return {
        TARGET_COLS[0]: g12[TARGET_COLS[0]],
        TARGET_COLS[1]: g12[TARGET_COLS[1]],
        TARGET_COLS[2]: g3[TARGET_COLS[2]],
    }, {"g12": e12, "g3": e3, "score_label_value_cells_materialized": 0}


def _score_labels(
    lineage: Mapping[str, Any], candidate_lock: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = _verify(
        Path(candidate_lock["path"]),
        size=int(candidate_lock["size_bytes"]),
        sha=str(candidate_lock["sha256"]),
    )
    lock = json.loads(path.read_text(encoding="utf-8"))
    if (
        lock.get("lock_kind") != "stage1_candidate_before_score_labels"
        or lock.get("config_sha256") != CONFIG_SHA
        or lock.get("score_label_value_cells_before_lock") != 0
    ):
        raise AssertionError("candidate lock identity/order differs")
    spec = lineage["official_data_and_lineage"]["label_file"]
    label_path = _verify(
        Path(spec["path"]), size=int(spec["bytes"]), sha=str(spec["sha256"])
    )
    g12, e12 = protocol._read_label_slice(label_path, "g12_score", YEAR_2023)
    g3, e3 = protocol._read_label_slice(label_path, "g3_score", G3_H2)
    labels = pd.DataFrame(np.nan, index=YEAR_2023, columns=TARGET_COLS, dtype=np.float64)
    labels.loc[YEAR_2023, list(TARGET_COLS[:2])] = g12.to_numpy(np.float64)
    labels.loc[G3_H2, TARGET_COLS[2]] = g3[TARGET_COLS[2]].to_numpy(np.float64)
    return labels, {
        "candidate_lock_verified_before_source_open": True,
        "g12": e12,
        "g3": e3,
        "label_value_cells_materialized": int(
            e12["label_value_cells_materialized"] + e3["label_value_cells_materialized"]
        ),
    }


def _closure(args: argparse.Namespace, config: Mapping[str, Any]) -> dict[str, Any]:
    static = config["static_lookup"]
    source_paths = [
        args.config,
        args.config.with_suffix(".sha256"),
        Path(__file__),
        PROJECT_DIR / "src/esa_worldcover_landcover.py",
        PROJECT_DIR / "src/jma_paired_increment.py",
        PROJECT_DIR / "src/features.py",
        PROJECT_DIR / "src/metric.py",
        PROJECT_DIR / "scripts/run_copernicus_dem_directional_exposure.py",
        PROJECT_DIR / config["lineage_parent"]["path"],
        PROJECT_DIR / static["path"],
        PROJECT_DIR / config["supersedes"]["v1_path"],
        PROJECT_DIR / config["supersedes"]["incident_path"],
    ]
    for item in config["source"]["precutoff_objects"]:
        source_paths.append(
            PROJECT_DIR
            / "artifacts/external/esa_worldcover2020_v100_landcover_fasttrack/precutoff_2021_mirror"
            / item["name"]
        )
    return {str(path.resolve()): describe_file(_verify(path)) for path in source_paths}


def run_stage1(args: argparse.Namespace, config: Mapping[str, Any], lineage: Mapping[str, Any]) -> None:
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    static_spec = config["static_lookup"]
    static_path = _verify(
        PROJECT_DIR / static_spec["path"],
        size=int(static_spec["bytes"]),
        sha=STATIC_SHA,
    )
    args.out_dir.mkdir(parents=True)
    shutil.copyfile(args.config, args.out_dir / "preregister_v1.json")
    shutil.copyfile(args.config.with_suffix(".sha256"), args.out_dir / "preregister_v1.sha256")
    closure = _closure(args, config)
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

    features, raw_contract = protocol.shared._read_stage1_raw_features(
        args.raw_dir, pd.DataFrame(index=PRE2024)
    )
    if any(frame.shape[1] != 612 for frame in features.values()):
        raise AssertionError("control feature count differs")
    static_table = pd.read_parquet(static_path)
    turbines = parse_turbines(lineage["turbines"]["rows"])
    landcover: dict[str, pd.DataFrame] = {}
    landcover_files: dict[str, Any] = {}
    for group in TARGET_COLS:
        landcover[group] = build_dynamic_landcover_features(
            features[group], static_table, turbines, group=group
        )
        path = args.out_dir / f"stage1/features/{group}_worldcover.parquet"
        protocol._atomic_parquet(landcover[group], path)
        landcover_files[group] = describe_file(path)

    fit_labels, fit_evidence = _fit_labels(lineage)
    baseline = protocol._load_baseline(lineage)
    baseline_path = args.out_dir / "stage1/baseline_corrected_v3.parquet"
    protocol._atomic_parquet(baseline, baseline_path)
    train_indexes = {
        TARGET_COLS[0]: YEAR_2022,
        TARGET_COLS[1]: YEAR_2022,
        TARGET_COLS[2]: G3_H1,
    }
    apply_indexes = {
        TARGET_COLS[0]: YEAR_2023,
        TARGET_COLS[1]: YEAR_2023,
        TARGET_COLS[2]: G3_H2,
    }
    candidate = baseline.copy()
    training: dict[str, Any] = {}
    with HeavyFitGuard():
        for group in TARGET_COLS:
            train, apply = train_indexes[group], apply_indexes[group]
            control_train, control_apply = features[group].loc[train], features[group].loc[apply]
            actual = fit_labels[group]
            mask = eligible_rows(actual, CAPACITY_KWH[group])
            control_model, control_meta = fit_direct_model(
                control_train, actual, capacity_kwh=CAPACITY_KWH[group], eligible=mask
            )
            extended_train = pd.concat([control_train, landcover[group].loc[train]], axis=1)
            extended_apply = pd.concat([control_apply, landcover[group].loc[apply]], axis=1)
            if (
                tuple(extended_train.columns[-len(EXTENDED_COLUMNS) :]) != EXTENDED_COLUMNS
                or extended_train.shape[1] != 648
            ):
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
            protocol._atomic_joblib(control_model, control_path)
            protocol._atomic_joblib(extended_model, extended_path)
            detail_path = args.out_dir / f"stage1/diagnostics/{group}.parquet"
            protocol._atomic_parquet(
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
                "eligible_mask_sha256": hashlib.sha256(
                    np.ascontiguousarray(mask.astype(np.uint8)).tobytes()
                ).hexdigest(),
                "diagnostics": describe_file(detail_path),
            }

    candidate_path = args.out_dir / "stage1/candidate_worldcover2020_w025.parquet"
    protocol._atomic_parquet(candidate, candidate_path)
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
            "landcover_feature_files": landcover_files,
            "baseline": describe_file(baseline_path),
            "candidate": describe_file(candidate_path),
            "fit_label_evidence": fit_evidence,
            "training": training,
            "score_label_value_cells_before_lock": 0,
            "candidate_metric_values_before_lock": 0,
            "2024_label_value_cells_before_lock": 0,
            "2025_contest_value_cells_before_lock": 0,
        },
    )
    candidate_lock = describe_file(lock_path)
    score_labels, score_evidence = _score_labels(lineage, candidate_lock)

    segments = protocol._segments()
    group_records: dict[str, Any] = {}
    total_deltas: list[float] = []
    full_component_deltas: list[float] = []
    for group in TARGET_COLS[:2]:
        group_records[group] = {}
        for name in ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4"):
            idx = segments[name]
            record = protocol._comparison(
                score_labels.loc[idx, group],
                baseline.loc[idx, group],
                candidate.loc[idx, group],
                group,
            )
            group_records[group][name] = record
            total_deltas.append(float(record["delta_total_score"]))
            if name == "full":
                full_component_deltas.extend(
                    [float(record["delta_one_minus_nmae"]), float(record["delta_ficr"])]
                )
    group_records[TARGET_COLS[2]] = {}
    for name, idx in (("full_H2", G3_H2), ("Q3", segments["Q3"]), ("Q4", segments["Q4"])):
        record = protocol._comparison(
            score_labels.loc[idx, TARGET_COLS[2]],
            baseline.loc[idx, TARGET_COLS[2]],
            candidate.loc[idx, TARGET_COLS[2]],
            TARGET_COLS[2],
        )
        group_records[TARGET_COLS[2]][name] = record
        total_deltas.append(float(record["delta_total_score"]))
        if name == "full_H2":
            full_component_deltas.extend(
                [float(record["delta_one_minus_nmae"]), float(record["delta_ficr"])]
            )

    mixed_records: dict[str, Any] = {}
    for name, idx in segments.items():
        groups = TARGET_COLS if name in ("full", "H2", "Q3", "Q4") else TARGET_COLS[:2]
        record = protocol._mixed_comparison(
            score_labels.loc[idx], baseline.loc[idx], candidate.loc[idx], groups
        )
        mixed_records[name] = record
        total_deltas.append(float(record["delta_total_score"]))
        if name == "full":
            full_component_deltas.extend(
                [float(record["delta_one_minus_nmae"]), float(record["delta_ficr"])]
            )
    if len(total_deltas) != 24 or len(full_component_deltas) != 8:
        raise AssertionError("registered gate vector differs")
    passed = all(delta > 0.0 for delta in total_deltas) and all(
        delta >= 0.0 for delta in full_component_deltas
    )
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
            "positive_total_score_comparisons": int(sum(x > 0.0 for x in total_deltas)),
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
    if closure != _closure(args, config):
        raise AssertionError("source closure changed during execution")
    files = [
        describe_file(path)
        for path in sorted(args.out_dir.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    ]
    if list(args.out_dir.rglob("*.csv")):
        raise AssertionError("Stage1 must not create CSV")
    _write_json(
        args.out_dir / "manifest.json",
        {
            "schema_version": 1,
            "artifact_type": EXPERIMENT_ID,
            "status": "STAGE1_PASS" if passed else "REJECTED_STAGE1",
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
    print(
        json.dumps(
            {
                "stage1_passed": passed,
                "positive_total_comparisons": int(sum(x > 0.0 for x in total_deltas)),
                "min_total_delta": min(total_deltas),
                "min_full_component_delta": min(full_component_deltas),
                "mixed_full": mixed_records["full"],
                "manifest": describe_file(args.out_dir / "manifest.json"),
            }
        ),
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = _args(argv)
    args.raw_dir = args.raw_dir.resolve()
    args.config = args.config.resolve()
    args.out_dir = args.out_dir.resolve()
    config, lineage = _load_config(args)
    run_stage1(args, config, lineage)


if __name__ == "__main__":
    main()

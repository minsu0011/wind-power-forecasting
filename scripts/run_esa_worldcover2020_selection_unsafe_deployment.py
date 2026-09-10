"""One-shot selection-unsafe 2024 deployment-parity audit for WorldCover."""

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

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from scripts import run_copernicus_dem_directional_exposure as metric_protocol  # noqa: E402
from scripts import run_esa_worldcover2020_directional_landcover as strict  # noqa: E402
from scripts import run_jma_gsm_paired_increment as full_protocol  # noqa: E402
from src.copernicus_dem_exposure import parse_turbines  # noqa: E402
from src.esa_worldcover_landcover import EXTENDED_COLUMNS, build_dynamic_landcover_features  # noqa: E402
from src.jma_paired_increment import eligible_rows, fit_direct_model, predict_cf  # noqa: E402
from src.manifest import describe_file, git_state, package_versions, sha256_file, utc_now  # noqa: E402
from src.metric import CAPACITY_KWH, TARGET_COLS  # noqa: E402


EXPERIMENT_ID = "esa_worldcover2020_selection_unsafe_deployment_parity_v1"
CONFIG_SHA = "f37da2e0c38f1734445e439f81c36264dcc44cee52f430d27ecf86d218abc17a"
LABEL_CONFIG_SHA = "cb5ff8ebb613230e67487499acfcc8f4030d55d637d43d357582d76a72b7103e"
STATIC_SHA = "059285f4ef99d11e3adf57f89897ca225d4820cd16917865779750362968363f"
TRANSFER_WEIGHT = 0.25
PRE2024 = full_protocol.PRE2024
YEAR_2024 = full_protocol.YEAR_2024
HEAVY_GUARD = ROOT / "artifacts/locks/heavy_cpu_fit.pid.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path(r"data/local/open"))
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/esa_worldcover2020_selection_unsafe_deployment_parity_preregister_v1.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "artifacts/postgate/esa_worldcover2020_selection_unsafe_deployment_parity_v1",
    )
    return parser.parse_args(argv)


def verify(path: Path, *, size: int | None = None, sha: str | None = None) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if size is not None and path.stat().st_size != size:
        raise AssertionError(f"size differs: {path}")
    if sha is not None and sha256_file(path) != sha:
        raise AssertionError(f"SHA differs: {path}")
    return path


def load_config(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    verify(args.config, sha=CONFIG_SHA)
    expected = f"{CONFIG_SHA}  {args.config.name}\n"
    if args.config.with_suffix(".sha256").read_text(encoding="utf-8") != expected:
        raise AssertionError("config sidecar differs")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["experiment_id"] != EXPERIMENT_ID:
        raise AssertionError("experiment identity differs")
    if float(config["frozen_science"]["transfer_weight"]) != TRANSFER_WEIGHT:
        raise AssertionError("transfer weight differs")
    lineage_spec = config["bound_inputs"]["lineage"]
    lineage_path = verify(
        Path(lineage_spec["path"]), size=int(lineage_spec["bytes"]), sha=str(lineage_spec["sha256"])
    )
    label_spec = config["bound_inputs"]["label_contract"]
    label_path = verify(
        Path(label_spec["path"]), size=int(label_spec["bytes"]), sha=LABEL_CONFIG_SHA
    )
    return (
        config,
        json.loads(lineage_path.read_text(encoding="utf-8")),
        json.loads(label_path.read_text(encoding="utf-8")),
    )


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


class HeavyGuard:
    def __init__(self) -> None:
        self.token = uuid.uuid4().hex
        self.acquired = False

    def __enter__(self) -> "HeavyGuard":
        HEAVY_GUARD.parent.mkdir(parents=True, exist_ok=True)
        if HEAVY_GUARD.exists():
            record = json.loads(HEAVY_GUARD.read_text(encoding="utf-8"))
            if _pid_alive(int(record.get("pid", -1))):
                raise RuntimeError(f"heavy guard held: {record}")
            os.replace(
                HEAVY_GUARD,
                HEAVY_GUARD.with_name(
                    f"heavy_cpu_fit.stale-pid{record.get('pid','unknown')}-{uuid.uuid4().hex}.json"
                ),
            )
        payload = {
            "schema_version": 1,
            "pid": os.getpid(),
            "token": self.token,
            "experiment_id": EXPERIMENT_ID,
            "stage": "selection_unsafe_stage2",
            "created_utc": utc_now(),
        }
        fd = os.open(HEAVY_GUARD, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
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


def source_closure(args: argparse.Namespace, config: Mapping[str, Any]) -> dict[str, Any]:
    paths = [
        args.config,
        args.config.with_suffix(".sha256"),
        Path(__file__),
        ROOT / "src/esa_worldcover_landcover.py",
        ROOT / "src/jma_paired_increment.py",
        ROOT / "src/features.py",
        ROOT / "src/metric.py",
        ROOT / config["failed_strict_parent"]["config_path"],
        ROOT / config["failed_strict_parent"]["results"]["path"],
        ROOT / config["failed_strict_parent"]["selection_lock"]["path"],
        ROOT / config["failed_strict_parent"]["manifest"]["path"],
        ROOT / config["frozen_science"]["static_table"]["path"],
        ROOT / config["bound_inputs"]["lineage"]["path"],
        ROOT / config["bound_inputs"]["label_contract"]["path"],
        Path(config["bound_inputs"]["train_ldaps"]["path"]),
        Path(config["bound_inputs"]["train_gfs"]["path"]),
        ROOT / config["bound_inputs"]["stage2_recent_v4"]["path"],
    ]
    return {str(path.resolve()): describe_file(verify(path)) for path in paths}


def read_fit_labels(
    args: argparse.Namespace, label_config: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    contract = label_config["label_slice_contract"]
    spec = contract["source"]
    source = verify(
        Path(spec["path"]), size=int(spec["bytes"]), sha=str(spec["sha256"])
    )
    return full_protocol._read_label_slice(
        source, contract["stage2_fit"], expected_index=PRE2024
    )


def read_score_labels(
    label_config: Mapping[str, Any], lock_record: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    lock_path = verify(
        Path(lock_record["path"]),
        size=int(lock_record["size_bytes"]),
        sha=str(lock_record["sha256"]),
    )
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if (
        lock.get("lock_kind") != "selection_unsafe_stage2_candidate_before_2024_score_labels"
        or lock.get("config_sha256") != CONFIG_SHA
        or lock.get("score_label_value_cells_before_lock") != 0
    ):
        raise AssertionError("candidate lock differs")
    contract = label_config["label_slice_contract"]
    spec = contract["source"]
    source = verify(
        Path(spec["path"]), size=int(spec["bytes"]), sha=str(spec["sha256"])
    )
    frame, evidence = full_protocol._read_label_slice(
        source, contract["stage2_score"], expected_index=YEAR_2024
    )
    evidence["candidate_lock_verified_before_source_open"] = True
    return frame, evidence


def run(args: argparse.Namespace, config: Mapping[str, Any], lineage: Mapping[str, Any], label_config: Mapping[str, Any]) -> None:
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True)
    shutil.copyfile(args.config, args.out_dir / "preregister_v1.json")
    shutil.copyfile(args.config.with_suffix(".sha256"), args.out_dir / "preregister_v1.sha256")
    closure = source_closure(args, config)
    metric_protocol._write_json(
        args.out_dir / "stage2_prescore_lock.json",
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "config_sha256": CONFIG_SHA,
            "selection_unsafe": True,
            "source_closure": closure,
            "models_fit_before_lock": 0,
            "2024_label_value_cells_before_lock": 0,
            "2025_value_cells_before_lock": 0,
            "metrics_before_lock": 0,
            "csv_before_lock": 0,
        },
    )

    features = full_protocol._read_full_features(args)
    static_spec = config["frozen_science"]["static_table"]
    static_path = verify(
        ROOT / static_spec["path"], size=int(static_spec["bytes"]), sha=STATIC_SHA
    )
    table = pd.read_parquet(static_path)
    turbines = parse_turbines(lineage["turbines"]["rows"])
    landcover: dict[str, pd.DataFrame] = {}
    landcover_files: dict[str, Any] = {}
    for group in TARGET_COLS:
        landcover[group] = build_dynamic_landcover_features(
            features[group], table, turbines, group=group
        )
        path = args.out_dir / f"stage2/features/{group}_worldcover.parquet"
        metric_protocol._atomic_parquet(landcover[group], path)
        landcover_files[group] = describe_file(path)

    fit_labels, fit_evidence = read_fit_labels(args, label_config)
    baseline_spec = config["bound_inputs"]["stage2_recent_v4"]
    baseline_path = verify(
        ROOT / baseline_spec["path"],
        size=int(baseline_spec["bytes"]),
        sha=str(baseline_spec["sha256"]),
    )
    baseline = pd.read_parquet(baseline_path).astype(np.float64)
    baseline.index = pd.DatetimeIndex(baseline.index, name="forecast_kst_dtm")
    if not baseline.index.equals(YEAR_2024) or tuple(baseline.columns) != TARGET_COLS:
        raise AssertionError("recent-v4 2024 baseline differs")
    candidate = baseline.copy()
    increments = pd.DataFrame(0.0, index=YEAR_2024, columns=TARGET_COLS, dtype=np.float64)
    training: dict[str, Any] = {}
    with HeavyGuard():
        for group in TARGET_COLS:
            control_train = features[group].loc[PRE2024]
            control_apply = features[group].loc[YEAR_2024]
            actual = fit_labels[group]
            mask = eligible_rows(actual, CAPACITY_KWH[group])
            control_model, control_meta = fit_direct_model(
                control_train, actual, capacity_kwh=CAPACITY_KWH[group], eligible=mask
            )
            extended_train = pd.concat([control_train, landcover[group].loc[PRE2024]], axis=1)
            extended_apply = pd.concat([control_apply, landcover[group].loc[YEAR_2024]], axis=1)
            if (
                tuple(extended_train.columns[-len(EXTENDED_COLUMNS) :]) != EXTENDED_COLUMNS
                or extended_train.shape[1] != 648
            ):
                raise AssertionError("extended feature identity differs")
            extended_model, extended_meta = fit_direct_model(
                extended_train, actual, capacity_kwh=CAPACITY_KWH[group], eligible=mask
            )
            control_cf = predict_cf(control_model, control_apply)
            extended_cf = predict_cf(extended_model, extended_apply)
            increment = extended_cf - control_cf
            increments[group] = increment
            candidate[group] = np.clip(
                baseline[group].to_numpy(np.float64)
                + TRANSFER_WEIGHT * CAPACITY_KWH[group] * increment,
                0.0,
                1.02 * CAPACITY_KWH[group],
            )
            control_model_path = args.out_dir / f"stage2/models/{group}_control.joblib"
            extended_model_path = args.out_dir / f"stage2/models/{group}_extended.joblib"
            metric_protocol._atomic_joblib(control_model, control_model_path)
            metric_protocol._atomic_joblib(extended_model, extended_model_path)
            diagnostic_path = args.out_dir / f"stage2/diagnostics/{group}.parquet"
            metric_protocol._atomic_parquet(
                pd.DataFrame(
                    {
                        "control_cf": control_cf,
                        "extended_cf": extended_cf,
                        "increment_cf": increment,
                        "baseline_kwh": baseline[group].to_numpy(np.float64),
                        "candidate_kwh": candidate[group].to_numpy(np.float64),
                    },
                    index=YEAR_2024,
                ),
                diagnostic_path,
            )
            training[group] = {
                "control": {**control_meta, "model": describe_file(control_model_path)},
                "extended": {**extended_meta, "model": describe_file(extended_model_path)},
                "eligible_mask_sha256": hashlib.sha256(
                    np.ascontiguousarray(mask.astype(np.uint8)).tobytes()
                ).hexdigest(),
                "same_rows_target_seed_parameters": True,
                "diagnostics": describe_file(diagnostic_path),
            }

    increment_path = args.out_dir / "stage2/paired_increment_cf.parquet"
    baseline_copy_path = args.out_dir / "stage2/recent_v4_baseline_2024.parquet"
    candidate_path = args.out_dir / "stage2/recent_v4_candidate_2024.parquet"
    metric_protocol._atomic_parquet(increments, increment_path)
    metric_protocol._atomic_parquet(baseline, baseline_copy_path)
    metric_protocol._atomic_parquet(candidate, candidate_path)
    lock_path = args.out_dir / "stage2_candidate_before_2024_score_labels_lock.json"
    metric_protocol._write_json(
        lock_path,
        {
            "schema_version": 1,
            "lock_kind": "selection_unsafe_stage2_candidate_before_2024_score_labels",
            "created_utc": utc_now(),
            "config_sha256": CONFIG_SHA,
            "selection_unsafe": True,
            "source_closure": closure,
            "fit_label_evidence": fit_evidence,
            "landcover_features": landcover_files,
            "training": training,
            "increment": describe_file(increment_path),
            "baseline": describe_file(baseline_copy_path),
            "candidate": describe_file(candidate_path),
            "score_label_value_cells_before_lock": 0,
            "candidate_metric_values_before_lock": 0,
            "2025_value_cells_before_lock": 0,
        },
    )
    lock_record = describe_file(lock_path)
    score_labels, score_evidence = read_score_labels(label_config, lock_record)

    segments = full_protocol._segments(2024)
    groups: dict[str, Any] = {}
    for group in TARGET_COLS:
        groups[group] = {}
        for name, index in segments.items():
            groups[group][name] = metric_protocol._comparison(
                score_labels.loc[index, group],
                baseline.loc[index, group],
                candidate.loc[index, group],
                group,
            )
    mixed: dict[str, Any] = {}
    for name, index in segments.items():
        mixed[name] = metric_protocol._mixed_comparison(
            score_labels.loc[index], baseline.loc[index], candidate.loc[index], TARGET_COLS
        )
    full = mixed["full"]
    passed = full["delta_total_score"] > 0.0 and full["delta_ficr"] > 0.0
    result_path = args.out_dir / "stage2_results.json"
    metric_protocol._write_json(
        result_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "config_sha256": CONFIG_SHA,
            "selection_unsafe": True,
            "candidate_before_2024_score_labels_lock": lock_record,
            "score_label_evidence_after_lock": score_evidence,
            "group_comparisons": groups,
            "mixed_comparisons": mixed,
            "promotion_gate": {
                "delta_total_score_strict_positive": full["delta_total_score"] > 0.0,
                "delta_ficr_strict_positive": full["delta_ficr"] > 0.0,
                "delta_one_minus_nmae_report_only": full["delta_one_minus_nmae"],
                "passed": passed,
            },
            "final_allowed": passed,
            "2025_values_read": 0,
            "csv_created": 0,
        },
    )
    promotion_path = args.out_dir / "stage2_promotion_lock.json"
    metric_protocol._write_json(
        promotion_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "config_sha256": CONFIG_SHA,
            "selection_unsafe": True,
            "stage2_candidate_lock": lock_record,
            "stage2_results": describe_file(result_path),
            "passed": passed,
            "final_allowed": passed,
            "requires_separate_final_prescore_lock": True,
            "2025_values_read": 0,
            "csv_created": 0,
        },
    )
    if closure != source_closure(args, config):
        raise AssertionError("source closure changed")
    files = [
        describe_file(path)
        for path in sorted(args.out_dir.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    ]
    metric_protocol._write_json(
        args.out_dir / "manifest.json",
        {
            "schema_version": 1,
            "artifact_type": EXPERIMENT_ID,
            "status": "SELECTION_UNSAFE_STAGE2_PASS" if passed else "REJECTED_SELECTION_UNSAFE_STAGE2",
            "created_utc": utc_now(),
            "config_sha256": CONFIG_SHA,
            "selection_unsafe": True,
            "source_closure": closure,
            "files": files,
            "runtime": {"packages": package_versions()},
            "git": git_state(ROOT),
            "2025_values_read": 0,
            "csv_created": False,
        },
    )
    print(
        json.dumps(
            {
                "passed": passed,
                "mixed_full": full,
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
    config, lineage, label_config = load_config(args)
    run(args, config, lineage, label_config)


if __name__ == "__main__":
    main()

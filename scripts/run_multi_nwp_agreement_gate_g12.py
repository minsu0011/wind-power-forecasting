"""Validate one frozen target-free multi-NWP agreement gate on 2024 H2.

This runner materializes and hash-locks the single candidate before it opens
the H2 label prefix.  It cannot access any 2025 external NWP values and never
fits or searches an agreement threshold.
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
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metric import CAPACITY_KWH, TARGET_COLS, group_metrics, score_details
from src.multi_nwp_agreement_gate import SOURCE_ORDER, agreement_gated_increment


EXPERIMENT_ID = "multi_nwp_agreement_gate_g12_v1"
CONFIG_PATH = ROOT / "configs/multi_nwp_agreement_gate_g12_preregister_v1.json"
SIDECAR_PATH = CONFIG_PATH.with_suffix(".sha256")
OUTPUT_DIR = ROOT / f"artifacts/postgate/{EXPERIMENT_ID}"
LABEL_PATH = Path(r"data/local/open/train/train_labels.csv")
BASELINE_PATH = ROOT / "artifacts/postgate/multi_nwp_consensus_g23_rescue_v1/validation/baseline_recent097_2024_h2.parquet"
JOINT_PATH = ROOT / "artifacts/postgate/multi_nwp_joint_disagreement_2024_forward_v1/predictions/joint_paired_increment_cf.parquet"
SOURCE_INCREMENT_PATHS = {
    "ecmwf": ROOT / "artifacts/postgate/ecmwf_ifs025_paired_increment_2024_forward_v2/predictions/candidate_increments.parquet",
    "icon": ROOT / "artifacts/postgate/icon_global_paired_increment_2024_forward_v3/predictions/candidate_increments.parquet",
    "gfs": ROOT / "artifacts/postgate/gfs_global_revision_paired_increment_2024_forward_v1/predictions/candidate_increments_cf.parquet",
}
SOURCE_MANIFEST_PATHS = {
    "ecmwf": ROOT / "artifacts/postgate/ecmwf_ifs025_paired_increment_2024_forward_v2/manifest.json",
    "icon": ROOT / "artifacts/postgate/icon_global_paired_increment_2024_forward_v3/manifest.json",
    "gfs": ROOT / "artifacts/postgate/gfs_global_revision_paired_increment_2024_forward_v1/manifest.json",
    "joint": ROOT / "artifacts/postgate/multi_nwp_joint_disagreement_2024_forward_v1/manifest.json",
}
HEAVY_GUARD_PATH = ROOT / "artifacts/locks/heavy_cpu_fit.pid.json"
TRANSFER_WEIGHT = 0.25
ACTIVE_GROUPS = ("kpx_group_1", "kpx_group_2")
IDENTITY_GROUP = "kpx_group_3"
TERMINAL = pd.Timestamp("2025-01-01 00:00:00")
LABEL_ROWS = 26_304
SOURCE_COLUMN = "B_day1_hours01_13_else_day2_w025__{group}"
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
    owner = {"pid": os.getpid(), "experiment_id": EXPERIMENT_ID, "created_utc": utc_now()}
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


def load_config() -> dict[str, Any]:
    expected = SIDECAR_PATH.read_text(encoding="ascii").split()[0]
    observed = sha256(CONFIG_PATH)
    if expected != observed:
        raise RuntimeError("preregistration sidecar mismatch")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["status"] != "frozen_before_candidate_materialization_or_2024_metric":
        raise RuntimeError("preregistration status differs")
    return config


def validate_bound_files(config: Mapping[str, Any]) -> None:
    mismatch: dict[str, Any] = {}
    for relative, expected in config["bound_files"].items():
        path = ROOT / relative
        observed = file_record(path)
        if observed["bytes"] != expected["bytes"] or observed["sha256"] != expected["sha256"]:
            mismatch[relative] = {"expected": expected, "observed": observed}
    if mismatch:
        raise RuntimeError(f"bound file mismatch: {mismatch}")


def read_increment(path: Path, index: pd.DatetimeIndex) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    result = pd.DataFrame(index=index, columns=TARGET_COLS, dtype=np.float64)
    for group in TARGET_COLS:
        column = SOURCE_COLUMN.format(group=group)
        result[group] = frame.loc[index, column].astype(np.float64)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError(f"non-finite increment: {path}")
    return result


def bounded_label_prefix() -> tuple[bytes, dict[str, Any]]:
    chunks: list[bytes] = []
    digest = hashlib.sha256()
    with LABEL_PATH.open("rb", buffering=0) as stream:
        for line_number in range(LABEL_ROWS + 1):
            line = stream.readline()
            if not line:
                raise ValueError(f"label file ended at physical line {line_number}")
            chunks.append(line)
            digest.update(line)
    payload = b"".join(chunks)
    return payload, {
        "path": str(LABEL_PATH),
        "physical_lines_read": LABEL_ROWS + 1,
        "data_rows": LABEL_ROWS,
        "bytes": len(payload),
        "sha256": digest.hexdigest(),
    }


def parse_labels(payload: bytes, index: pd.DatetimeIndex) -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(payload), encoding="utf-8-sig")
    if tuple(frame.columns) != ("kst_dtm", *TARGET_COLS) or len(frame) != LABEL_ROWS:
        raise ValueError("bounded label schema or row count differs")
    frame.index = pd.DatetimeIndex(
        pd.to_datetime(frame.pop("kst_dtm"), errors="raise"), name="forecast_kst_dtm"
    )
    if frame.index[0] != pd.Timestamp("2022-01-01 01:00") or frame.index[-1] != TERMINAL:
        raise ValueError("bounded label time range differs")
    return frame.loc[index, list(TARGET_COLS)].astype(np.float64)


def metric_payload(actual: pd.Series, prediction: pd.Series, group: str) -> dict[str, Any]:
    result = group_metrics(actual, prediction, CAPACITY_KWH[group], group_name=group).as_dict()
    result["total_score"] = 0.5 * (result["one_minus_nmae"] + result["ficr"])
    return result


def compare(labels: pd.DataFrame, baseline: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {"groups": {}, "mixed": {}}
    for group in TARGET_COLS:
        output["groups"][group] = {}
        for name, (start, end) in SLICES.items():
            index = labels.index[(labels.index >= start) & (labels.index <= end)]
            base = metric_payload(labels.loc[index, group], baseline.loc[index, group], group)
            cand = metric_payload(labels.loc[index, group], candidate.loc[index, group], group)
            output["groups"][group][name] = {
                "baseline": base,
                "candidate": cand,
                "delta": {
                    key: cand[key] - base[key]
                    for key in ("total_score", "one_minus_nmae", "ficr")
                },
            }
    for name, (start, end) in SLICES.items():
        index = labels.index[(labels.index >= start) & (labels.index <= end)]
        base = score_details(labels.loc[index], baseline.loc[index]).as_dict()
        cand = score_details(labels.loc[index], candidate.loc[index]).as_dict()
        output["mixed"][name] = {
            "baseline": base,
            "candidate": cand,
            "delta": {
                key: cand[key] - base[key]
                for key in ("total_score", "one_minus_nmae", "ficr")
            },
        }
    return output


def promotion_gate(comparison: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    rules = config["promotion_gate"]
    h2 = comparison["mixed"]["H2"]["delta"]
    q = {name: comparison["mixed"][name]["delta"]["total_score"] for name in ("Q3", "Q4")}
    months = {
        name: comparison["mixed"][name]["delta"]["total_score"]
        for name in ("Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
    }
    active_total = {
        group: comparison["groups"][group]["H2"]["delta"]["total_score"]
        for group in ACTIVE_GROUPS
    }
    active_ficr = {
        group: comparison["groups"][group]["H2"]["delta"]["ficr"]
        for group in ACTIVE_GROUPS
    }
    checks = {
        "h2_total_minimum": h2["total_score"] >= rules["mixed_h2_min_total_score_delta"],
        "h2_ficr_minimum": h2["ficr"] >= rules["mixed_h2_min_ficr_delta"],
        "h2_nmae_floor": h2["one_minus_nmae"] >= rules["mixed_h2_min_one_minus_nmae_delta"],
        "both_active_groups_h2_total_positive": min(active_total.values()) > rules["active_group_h2_min_total_score_delta"],
        "at_least_one_active_group_h2_ficr_positive": sum(value > 0.0 for value in active_ficr.values()) >= 1,
        "q3_q4_floor": min(q.values()) >= rules["minimum_q3_q4_total_score_delta"],
        "positive_quarter_count": sum(value > 0.0 for value in q.values()) >= rules["minimum_positive_quarters"],
        "month_floor": min(months.values()) >= rules["minimum_month_total_score_delta"],
        "positive_month_count": sum(value > 0.0 for value in months.values()) >= rules["minimum_positive_months"],
        "identity_group_exact_metric_identity": all(
            comparison["groups"][IDENTITY_GROUP][name]["delta"][key] == 0.0
            for name in SLICES
            for key in ("total_score", "one_minus_nmae", "ficr")
        ),
    }
    return {
        "rules": dict(rules),
        "checks": checks,
        "pass": bool(all(checks.values())),
        "mixed_h2_delta": h2,
        "active_group_h2_total_score_deltas": active_total,
        "active_group_h2_ficr_deltas": active_ficr,
        "quarter_total_score_deltas": q,
        "month_total_score_deltas": months,
        "positive_months": int(sum(value > 0.0 for value in months.values())),
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
    validate_bound_files(config)
    input_paths = [BASELINE_PATH, JOINT_PATH, *SOURCE_INCREMENT_PATHS.values(), *SOURCE_MANIFEST_PATHS.values()]
    source_lock = out_dir / "source_lock_before_candidate_or_label_access.json"
    atomic_json(
        {
            "schema_version": 1,
            "status": "all_inputs_locked_before_candidate_materialization_or_label_access",
            "created_utc": utc_now(),
            "preregister": file_record(CONFIG_PATH),
            "code": [
                file_record(ROOT / "src/multi_nwp_agreement_gate.py"),
                file_record(ROOT / "src/metric.py"),
                file_record(Path(__file__).resolve()),
                file_record(ROOT / "tests/test_multi_nwp_agreement_gate.py"),
            ],
            "inputs": [file_record(path) for path in input_paths],
            "physical_access": {
                "label_cells_read": 0,
                "maximum_external_year": 2024,
                "year_2025_external_value_cells": 0,
                "candidate_cells_materialized": 0,
            },
        },
        source_lock,
    )

    owner = acquire_heavy_guard(HEAVY_GUARD_PATH)
    atexit.register(release_heavy_guard, HEAVY_GUARD_PATH, owner)
    print(f"source locked; heavy guard acquired pid={owner['pid']}", flush=True)

    baseline = pd.read_parquet(BASELINE_PATH).astype(np.float64)
    baseline.index = pd.DatetimeIndex(baseline.index, name="forecast_kst_dtm")
    index = baseline.index
    if (
        tuple(baseline.columns) != TARGET_COLS
        or index[0] != pd.Timestamp("2024-07-01 00:00")
        or index[-1] != TERMINAL
        or not np.isfinite(baseline.to_numpy()).all()
    ):
        raise ValueError("baseline schema, range, or values differ")
    joint = pd.read_parquet(JOINT_PATH).loc[index, list(TARGET_COLS)].astype(np.float64)
    joint.index = pd.DatetimeIndex(joint.index, name="forecast_kst_dtm")
    sources = {
        name: read_increment(SOURCE_INCREMENT_PATHS[name], index) for name in SOURCE_ORDER
    }
    gated, confidence, agreement_count = agreement_gated_increment(joint, sources)
    for group in (IDENTITY_GROUP,):
        gated[group] = 0.0
        confidence[group] = 0.0
        agreement_count[group] = 0
    if np.count_nonzero(gated.loc[TERMINAL].to_numpy(dtype=np.float64)):
        raise AssertionError("terminal gated increment is not exact zero")

    candidate = baseline.copy()
    for group in ACTIVE_GROUPS:
        candidate[group] = np.clip(
            baseline[group].to_numpy(dtype=np.float64)
            + TRANSFER_WEIGHT * CAPACITY_KWH[group] * gated[group].to_numpy(dtype=np.float64),
            0.0,
            1.02 * CAPACITY_KWH[group],
        )
    if not np.array_equal(
        candidate[IDENTITY_GROUP].to_numpy(dtype=np.float64).view(np.uint64),
        baseline[IDENTITY_GROUP].to_numpy(dtype=np.float64).view(np.uint64),
    ):
        raise AssertionError("identity group differs from baseline")
    if not np.array_equal(
        candidate.loc[[TERMINAL]].to_numpy(dtype=np.float64).view(np.uint64),
        baseline.loc[[TERMINAL]].to_numpy(dtype=np.float64).view(np.uint64),
    ):
        raise AssertionError("terminal row differs from baseline")

    gated_path = out_dir / "predictions/agreement_gated_joint_increment_cf.parquet"
    confidence_path = out_dir / "diagnostics/agreement_confidence.parquet"
    count_path = out_dir / "diagnostics/source_sign_agreement_count.parquet"
    candidate_path = out_dir / "predictions/agreement_gated_g12_candidate_scale097.parquet"
    atomic_parquet(gated, gated_path)
    atomic_parquet(confidence, confidence_path)
    atomic_parquet(agreement_count, count_path)
    atomic_parquet(candidate, candidate_path)
    diagnostics = {
        group: {
            "nonzero_gated_rows": int(np.count_nonzero(gated[group].to_numpy())),
            "zero_gated_rows": int((gated[group].to_numpy() == 0.0).sum()),
            "mean_confidence": float(confidence[group].mean()),
            "median_confidence": float(confidence[group].median()),
            "agreement_count_histogram": {
                str(value): int((agreement_count[group] == value).sum()) for value in range(4)
            },
            "joint_mean_abs_cf": float(joint[group].abs().mean()),
            "gated_mean_abs_cf": float(gated[group].abs().mean()),
        }
        for group in TARGET_COLS
    }
    candidate_lock = out_dir / "candidate_lock_before_2024_h2_label_access.json"
    atomic_json(
        {
            "schema_version": 1,
            "status": "single_candidate_hash_locked_before_2024_h2_label_access_or_metric",
            "created_utc": utc_now(),
            "source_lock": file_record(source_lock),
            "formula": "G1/G2=clip(base097+0.25*C*joint*vote2of3*magnitude_cap,0,1.02C); G3=base097",
            "threshold_learning": "none",
            "gated_increment": file_record(gated_path),
            "confidence": file_record(confidence_path),
            "agreement_count": file_record(count_path),
            "candidate": file_record(candidate_path),
            "diagnostics": diagnostics,
            "physical_access": {
                "label_cells_read": 0,
                "maximum_external_year": 2024,
                "year_2025_external_value_cells": 0,
            },
        },
        candidate_lock,
    )
    print(f"candidate locked sha256={sha256(candidate_lock)}", flush=True)

    label_bytes, label_record = bounded_label_prefix()
    label_access = out_dir / "h2_label_access_after_candidate_lock.json"
    atomic_json(
        {
            "schema_version": 1,
            "status": "2024_h2_labels_opened_after_single_candidate_hash_lock",
            "created_utc": utc_now(),
            "candidate_lock": file_record(candidate_lock),
            "bounded_label_prefix": label_record,
        },
        label_access,
    )
    labels = parse_labels(label_bytes, index)
    comparison = compare(labels, baseline, candidate)
    gate = promotion_gate(comparison, config)
    status = "promoted_for_separate_through_2024_final_stage" if gate["pass"] else "rejected_on_frozen_2024_gate"
    results_path = out_dir / "stage_results.json"
    atomic_json(
        {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "created_utc": utc_now(),
            "status": status,
            "preregister_sha256": sha256(CONFIG_PATH),
            "diagnostics": diagnostics,
            "comparison": comparison,
            "gate": gate,
            "physical_access": {
                "maximum_external_year": 2024,
                "year_2025_external_requests_response_bytes_or_values": 0,
                "submission_csv_created": False,
            },
        },
        results_path,
    )
    decision_path = out_dir / ("promotion_lock.json" if gate["pass"] else "rejection.json")
    atomic_json(
        {
            "status": status,
            "preregister_sha256": sha256(CONFIG_PATH),
            "candidate_lock_sha256": sha256(candidate_lock),
            "stage_results_sha256": sha256(results_path),
            "separate_final_stage_authorized": bool(gate["pass"]),
        },
        decision_path,
    )

    outputs = [
        source_lock,
        gated_path,
        confidence_path,
        count_path,
        candidate_path,
        candidate_lock,
        label_access,
        results_path,
        decision_path,
    ]
    manifest_path = out_dir / "manifest.json"
    atomic_json(
        {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "created_utc": utc_now(),
            "preregister": file_record(CONFIG_PATH),
            "outputs_excluding_manifest_and_sidecar": [file_record(path) for path in outputs],
            "unlisted_files_before_manifest": sorted(
                str(path.relative_to(out_dir))
                for path in out_dir.rglob("*")
                if path.is_file() and path not in outputs
            ),
            "physical_access": {
                "maximum_external_year": 2024,
                "year_2025_external_value_cells": 0,
                "submission_csv_created": False,
            },
        },
        manifest_path,
    )
    if json.loads(manifest_path.read_text(encoding="utf-8"))["unlisted_files_before_manifest"]:
        raise RuntimeError("unlisted output detected")
    sidecar = out_dir / "manifest.sha256"
    sidecar.write_text(f"{sha256(manifest_path)}  manifest.json\n", encoding="ascii")
    release_heavy_guard(HEAVY_GUARD_PATH, owner)
    print(f"status={status} h2_delta={gate['mixed_h2_delta']}", flush=True)
    print(f"manifest={manifest_path} sha256={sha256(manifest_path)}", flush=True)


if __name__ == "__main__":
    main()

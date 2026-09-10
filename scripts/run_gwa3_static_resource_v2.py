"""Run the prospectively frozen GWA3 v2 paired-increment pipeline once.

The permanently quarantined v1 runner and every v1 output are intentionally
outside this execution lineage.  Stage2 cannot open 2024 labels until the
causal Stage1 gate passes, and final cannot open 2025 values until Stage2 passes.
"""

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

from scripts import run_copernicus_dem_directional_exposure as protocol  # noqa: E402
from src.gwa3_static_resource import EXTENDED_COLUMNS, build_gwa3_features  # noqa: E402
from src.jma_paired_increment import eligible_rows, fit_direct_model, predict_cf  # noqa: E402
from src.manifest import (  # noqa: E402
    describe_file,
    git_state,
    package_versions,
    sha256_file,
    utc_now,
)
from src.metric import CAPACITY_KWH, TARGET_COLS  # noqa: E402


EXPERIMENT_ID = "gwa3_static_resource_paired_increment_strict_v2"
CONFIG_SHA = "455f483e94be725f5fcc0ed58bb2121c7b62fda72a844f779065525c8212b156"
STATIC_SHA = "f36197d6093bb46cfaf7ed5a3e8d89377d11ed40640c088c66a437226ab653bf"
MANIFEST_SHA = "ecd26017b77c4d7157eee6193fb541fb94a58879fb564d3b3e8f3f8858900d9c"
FEATURE_SHA = "a8d6597028f6794762c53aea362f654bd9ea7e2ead57533734a2d9ef5d569052"
LINEAGE_SHA = "21e2a6aaee02d11c00d2b8a266e30917efa4f5a505417ceadb9f618fe7d2443b"
TRANSFER_WEIGHT = 0.10

YEAR_2022 = protocol.YEAR_2022
YEAR_2023 = protocol.YEAR_2023
PRE2024 = protocol.PRE2024
G3_H1 = protocol.G3_H1
G3_H2 = protocol.G3_H2
YEAR_2024 = pd.date_range(
    "2024-01-01 01:00", "2025-01-01 00:00", freq="h", name="forecast_kst_dtm"
)
THROUGH_2024 = PRE2024.append(YEAR_2024)
TEST_INDEX = pd.date_range(
    "2025-01-01 01:00", "2026-01-01 00:00", freq="h", name="forecast_kst_dtm"
)
H2_2024 = pd.date_range(
    "2024-07-01 01:00", "2025-01-01 00:00", freq="h", name="forecast_kst_dtm"
)

HEAVY_GUARD = PROJECT_DIR / "artifacts/locks/heavy_cpu_fit.pid.json"
ATTEMPT_TOMBSTONE = (
    PROJECT_DIR / "artifacts/locks/gwa3_static_resource_paired_increment_strict_v2.attempt.json"
)
DEFAULT_CONFIG = PROJECT_DIR / "configs/gwa3_static_resource_paired_increment_preregister_v2.json"
DEFAULT_OUT = PROJECT_DIR / "artifacts/postgate/gwa3_static_resource_paired_increment_strict_v2"
STATIC_PATH = (
    PROJECT_DIR
    / "artifacts/external/global_wind_atlas_v3_2019_static_fasttrack/static_gwa3_17_turbines.parquet"
)
SOURCE_MANIFEST_PATH = (
    PROJECT_DIR
    / "artifacts/external/global_wind_atlas_v3_2019_static_fasttrack/source_manifest.json"
)

LABEL_FIT_SPEC = {
    "source_start_byte": 48,
    "source_end_byte": 742551,
    "parser_bytes": 742551,
    "parser_sha256": "ee9707cc81229db760fc035d0bf1d10eb3586d2fef95849bdf463d73fae3aacd",
    "nrows": 17520,
    "phase": "stage2_fit_after_stage1_pass",
}
LABEL_2024_SPEC = {
    "source_start_byte": 742551,
    "source_end_byte": 1138967,
    "parser_bytes": 396464,
    "parser_sha256": "4194c8cb35dd863f5860f63e0fdc4eb91a3b0e189c9f079ad8aa34a52c633283",
    "nrows": 8784,
    "phase": "stage2_score_after_candidate_lock",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path(r"data/local/open"))
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_DIR / "artifacts/cache")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
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


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    protocol._atomic_parquet(frame, path)


def _atomic_joblib(model: Any, path: Path) -> None:
    protocol._atomic_joblib(model, path)


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
    """Tokenized owner-only guard whose identity is local to this experiment."""

    def __init__(self, path: Path = HEAVY_GUARD) -> None:
        self.path = path.resolve()
        self.token = uuid.uuid4().hex
        self.acquired = False
        self.payload: dict[str, Any] | None = None

    def __enter__(self) -> "HeavyFitGuard":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            record = json.loads(self.path.read_text(encoding="utf-8"))
            if _pid_alive(int(record.get("pid", -1))):
                raise RuntimeError(f"heavy guard held by live PID: {record}")
            stale = self.path.with_name(
                f"heavy_cpu_fit.stale-pid{record.get('pid','unknown')}-{uuid.uuid4().hex}.json"
            )
            os.replace(self.path, stale)
        payload = {
            "schema_version": 2,
            "pid": os.getpid(),
            "token": self.token,
            "experiment_id": EXPERIMENT_ID,
            "stage": "prospective_stage1_then_conditional_stage2_final",
            "created_utc": utc_now(),
        }
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.payload = payload
        self.acquired = True
        self.verify_live()
        return self

    def record(self) -> dict[str, Any]:
        self.verify_live()
        assert self.payload is not None
        return {**self.payload, "guard_file": describe_file(self.path)}

    def verify_live(self) -> None:
        if not self.acquired or self.payload is None or not self.path.is_file():
            raise AssertionError("v2 heavy guard is not continuously live")
        observed = json.loads(self.path.read_text(encoding="utf-8"))
        if observed != self.payload:
            raise AssertionError("v2 heavy guard ownership/identity changed")
        if observed.get("experiment_id") != EXPERIMENT_ID:
            raise AssertionError("v2 heavy guard experiment identity differs")

    def release(self) -> None:
        self.verify_live()
        self.path.unlink()
        self.acquired = False

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.acquired:
            self.release()


def _load_config(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    _verify(args.config, size=7998, sha=CONFIG_SHA)
    expected = f"{CONFIG_SHA}  {args.config.name}\n"
    if args.config.with_suffix(".sha256").read_text(encoding="utf-8") != expected:
        raise AssertionError("v2 preregistration sidecar differs")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["experiment_id"] != EXPERIMENT_ID:
        raise AssertionError("v2 experiment identity differs")
    if config["single_candidate"]["candidate_count"] != 1:
        raise AssertionError("candidate count differs")
    if float(config["single_candidate"]["transfer_weight"]) != TRANSFER_WEIGHT:
        raise AssertionError("transfer weight differs")
    if config["v1_quarantine"]["status"] != "PERMANENTLY_QUARANTINED_AND_NOT_EVIDENCE_FOR_V2":
        raise AssertionError("v1 quarantine binding differs")
    _verify(SOURCE_MANIFEST_PATH, size=7410, sha=MANIFEST_SHA)
    _verify(STATIC_PATH, size=9819, sha=STATIC_SHA)
    _verify(PROJECT_DIR / "src/gwa3_static_resource.py", size=5233, sha=FEATURE_SHA)
    lineage_spec = config["lineage_parent"]
    lineage_path = _verify(
        PROJECT_DIR / lineage_spec["path"], size=int(lineage_spec["bytes"]), sha=LINEAGE_SHA
    )
    return config, json.loads(lineage_path.read_text(encoding="utf-8"))


def _source_closure(args: argparse.Namespace, config: Mapping[str, Any]) -> dict[str, Any]:
    paths = [
        args.config,
        args.config.with_suffix(".sha256"),
        Path(__file__),
        PROJECT_DIR / "tests/test_gwa3_static_resource_v2.py",
        PROJECT_DIR / "src/gwa3_static_resource.py",
        PROJECT_DIR / "src/jma_paired_increment.py",
        PROJECT_DIR / "src/features.py",
        PROJECT_DIR / "src/metric.py",
        PROJECT_DIR / "src/manifest.py",
        PROJECT_DIR / "scripts/run_copernicus_dem_directional_exposure.py",
        PROJECT_DIR / "scripts/run_shared_q07_multiseed.py",
        PROJECT_DIR / config["lineage_parent"]["path"],
        SOURCE_MANIFEST_PATH,
        STATIC_PATH,
        PROJECT_DIR / config["v1_quarantine"]["incident_path"],
        PROJECT_DIR / "scripts/run_gwa3_static_resource_stage1.py",
    ]
    return {str(path.resolve()): describe_file(_verify(path)) for path in paths}


def _bounded_label_slice(
    label_path: Path,
    spec: Mapping[str, Any],
    expected_index: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    label_path = _verify(label_path)
    start, end = int(spec["source_start_byte"]), int(spec["source_end_byte"])
    with label_path.open("rb") as stream:
        header = stream.readline()
        if len(header) != 48:
            raise AssertionError("official label header differs")
        stream.seek(start)
        body = stream.read(end - start)
        if stream.tell() != end:
            raise AssertionError("bounded label reader crossed its byte boundary")
    payload = header + body
    digest = hashlib.sha256(payload).hexdigest()
    if len(payload) != int(spec["parser_bytes"]) or digest != spec["parser_sha256"]:
        raise AssertionError("bounded label slice identity differs")
    frame = pd.read_csv(
        io.BytesIO(payload),
        encoding="utf-8-sig",
        usecols=["kst_dtm", *TARGET_COLS],
        nrows=int(spec["nrows"]),
        memory_map=False,
    )
    times = pd.to_datetime(frame.pop("kst_dtm"), errors="raise")
    frame.index = pd.DatetimeIndex(times, name="forecast_kst_dtm")
    if tuple(frame.columns) != TARGET_COLS or not frame.index.equals(expected_index):
        raise AssertionError("bounded label schema/index differs")
    return frame.astype(np.float64), {
        "phase": spec["phase"],
        "rows": len(frame),
        "label_value_cells_materialized": int(frame.size),
        "parser_bytes": len(payload),
        "parser_sha256": digest,
        "source_start_byte": start,
        "source_end_byte": end,
    }


def _verify_lock(
    path: Path,
    *,
    kind: str,
    guard: HeavyFitGuard,
    zero_field: str,
) -> dict[str, Any]:
    guard.verify_live()
    payload = json.loads(_verify(path).read_text(encoding="utf-8"))
    if payload.get("lock_kind") != kind or payload.get("config_sha256") != CONFIG_SHA:
        raise AssertionError("candidate lock identity differs")
    if payload.get(zero_field) != 0:
        raise AssertionError("candidate lock was not sealed before score labels")
    live = guard.record()
    locked = payload.get("continuous_heavy_guard")
    for key in ("pid", "token", "experiment_id"):
        if locked.get(key) != live.get(key):
            raise AssertionError("candidate lock heavy-guard identity differs")
    return payload


def _stage1_score_labels(
    lineage: Mapping[str, Any], lock_path: Path, guard: HeavyFitGuard
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _verify_lock(
        lock_path,
        kind="stage1_candidate_before_score_labels",
        guard=guard,
        zero_field="score_label_value_cells_before_lock",
    )
    spec = lineage["official_data_and_lineage"]["label_file"]
    label_path = _verify(Path(spec["path"]), size=int(spec["bytes"]), sha=str(spec["sha256"]))
    g12, e12 = protocol._read_label_slice(label_path, "g12_score", YEAR_2023)
    g3, e3 = protocol._read_label_slice(label_path, "g3_score", G3_H2)
    labels = pd.DataFrame(np.nan, index=YEAR_2023, columns=TARGET_COLS, dtype=np.float64)
    labels.loc[:, list(TARGET_COLS[:2])] = g12.to_numpy(np.float64)
    labels.loc[G3_H2, TARGET_COLS[2]] = g3[TARGET_COLS[2]].to_numpy(np.float64)
    return labels, {"candidate_lock_verified_before_source_open": True, "g12": e12, "g3": e3}


def _segments_2024() -> dict[str, pd.DatetimeIndex]:
    make = lambda a, b: pd.date_range(a, b, freq="h", name="forecast_kst_dtm")
    return {
        "full": YEAR_2024,
        "H1": make("2024-01-01 01:00", "2024-07-01 00:00"),
        "H2": H2_2024,
        "Q1": make("2024-01-01 01:00", "2024-04-01 00:00"),
        "Q2": make("2024-04-01 01:00", "2024-07-01 00:00"),
        "Q3": make("2024-07-01 01:00", "2024-10-01 00:00"),
        "Q4": make("2024-10-01 01:00", "2025-01-01 00:00"),
    }


def _fit_pair(
    *,
    group: str,
    control: pd.DataFrame,
    external: pd.DataFrame,
    actual: pd.Series,
    train_index: pd.DatetimeIndex,
    apply_index: pd.DatetimeIndex,
    model_dir: Path,
    prefix: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    x_train = control.loc[train_index]
    x_apply = control.loc[apply_index]
    y_train = actual.loc[train_index]
    mask = eligible_rows(y_train, CAPACITY_KWH[group])
    control_model, control_meta = fit_direct_model(
        x_train, y_train, capacity_kwh=CAPACITY_KWH[group], eligible=mask
    )
    ex_train = pd.concat([x_train, external.loc[train_index]], axis=1).astype(np.float32)
    ex_apply = pd.concat([x_apply, external.loc[apply_index]], axis=1).astype(np.float32)
    if tuple(ex_train.columns[-24:]) != EXTENDED_COLUMNS or ex_train.shape[1] != 636:
        raise AssertionError("GWA extended feature order/count differs")
    extended_model, extended_meta = fit_direct_model(
        ex_train, y_train, capacity_kwh=CAPACITY_KWH[group], eligible=mask
    )
    if control_model.get_params() != extended_model.get_params():
        raise AssertionError("paired model parameters differ")
    control_live = predict_cf(control_model, x_apply)
    extended_live = predict_cf(extended_model, ex_apply)
    control_path = model_dir / f"{prefix}_{group}_control.joblib"
    extended_path = model_dir / f"{prefix}_{group}_extended.joblib"
    _atomic_joblib(control_model, control_path)
    _atomic_joblib(extended_model, extended_path)
    control_reload = predict_cf(joblib.load(control_path), x_apply)
    extended_reload = predict_cf(joblib.load(extended_path), ex_apply)
    if not np.array_equal(control_live, control_reload) or not np.array_equal(
        extended_live, extended_reload
    ):
        raise AssertionError("saved/reloaded paired predictions differ")
    increment = extended_reload - control_reload
    return increment, {
        "control": {**control_meta, "model": describe_file(control_path)},
        "extended": {**extended_meta, "model": describe_file(extended_path)},
        "eligible_mask_sha256": hashlib.sha256(
            np.ascontiguousarray(mask.astype(np.uint8)).tobytes()
        ).hexdigest(),
        "same_rows_target_mask_seed_parameters": True,
        "reload_bit_exact": True,
        "increment_cf_mean": float(increment.mean()),
        "increment_cf_std": float(increment.std()),
    }


def _apply_increment(
    baseline: pd.Series, increment: np.ndarray, group: str
) -> np.ndarray:
    values = np.clip(
        baseline.to_numpy(np.float64)
        + TRANSFER_WEIGHT * CAPACITY_KWH[group] * np.asarray(increment, np.float64),
        0.0,
        1.02 * CAPACITY_KWH[group],
    )
    if values.shape != (len(baseline),) or not np.isfinite(values).all():
        raise AssertionError("candidate transfer differs")
    return values


def _stage1_metrics(
    labels: pd.DataFrame, baseline: pd.DataFrame, candidate: pd.DataFrame
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    segments = protocol._segments()
    groups: dict[str, Any] = {}
    full_deltas: list[float] = []
    for group in TARGET_COLS[:2]:
        groups[group] = {}
        for name, idx in segments.items():
            groups[group][name] = protocol._comparison(
                labels.loc[idx, group], baseline.loc[idx, group], candidate.loc[idx, group], group
            )
        full_deltas.append(float(groups[group]["full"]["delta_total_score"]))
    groups[TARGET_COLS[2]] = {}
    for name, idx in (("full_H2", G3_H2), ("Q3", segments["Q3"]), ("Q4", segments["Q4"])):
        groups[TARGET_COLS[2]][name] = protocol._comparison(
            labels.loc[idx, TARGET_COLS[2]],
            baseline.loc[idx, TARGET_COLS[2]],
            candidate.loc[idx, TARGET_COLS[2]],
            TARGET_COLS[2],
        )
    full_deltas.append(float(groups[TARGET_COLS[2]]["full_H2"]["delta_total_score"]))
    mixed: dict[str, Any] = {}
    for name, idx in segments.items():
        available = TARGET_COLS if name in ("full", "H2", "Q3", "Q4") else TARGET_COLS[:2]
        mixed[name] = protocol._mixed_comparison(
            labels.loc[idx], baseline.loc[idx], candidate.loc[idx], available
        )
    positive_segments = int(
        sum(float(record["delta_total_score"]) > 0.0 for record in mixed.values())
    )
    passed = bool(
        all(value > 0.0 for value in full_deltas)
        and float(mixed["full"]["delta_total_score"]) > 0.0
        and float(mixed["full"]["delta_one_minus_nmae"]) > 0.0
        and float(mixed["full"]["delta_ficr"]) > 0.0
        and float(mixed["H2"]["delta_total_score"]) >= 0.0
        and positive_segments >= 5
    )
    gate = {
        "group_full_deltas": full_deltas,
        "mixed_full_score": float(mixed["full"]["delta_total_score"]),
        "mixed_full_one_minus_nmae": float(mixed["full"]["delta_one_minus_nmae"]),
        "mixed_full_ficr": float(mixed["full"]["delta_ficr"]),
        "mixed_h2_score": float(mixed["H2"]["delta_total_score"]),
        "positive_mixed_segments": positive_segments,
        "passed": passed,
    }
    return groups, mixed, gate


def _run_stage1(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    lineage: Mapping[str, Any],
    closure: Mapping[str, Any],
    static: pd.DataFrame,
    guard: HeavyFitGuard,
) -> bool:
    features, raw_contract = protocol.shared._read_stage1_raw_features(
        args.raw_dir, pd.DataFrame(index=PRE2024)
    )
    if any(frame.shape != (len(PRE2024), 612) for frame in features.values()):
        raise AssertionError("Stage1 control feature contract differs")
    gwa = {group: build_gwa3_features(features[group], static, group=group) for group in TARGET_COLS}
    fit_labels, fit_evidence = protocol.read_fit_labels(args, lineage)
    baseline = protocol._load_baseline(lineage)
    _atomic_parquet(baseline, args.out_dir / "stage1/baseline_corrected_v3.parquet")
    train_indexes = {TARGET_COLS[0]: YEAR_2022, TARGET_COLS[1]: YEAR_2022, TARGET_COLS[2]: G3_H1}
    apply_indexes = {TARGET_COLS[0]: YEAR_2023, TARGET_COLS[1]: YEAR_2023, TARGET_COLS[2]: G3_H2}
    candidate = baseline.copy()
    training: dict[str, Any] = {}
    for group in TARGET_COLS:
        guard.verify_live()
        feature_path = args.out_dir / f"stage1/features/{group}_gwa3.parquet"
        _atomic_parquet(gwa[group], feature_path)
        increment, training[group] = _fit_pair(
            group=group,
            control=features[group],
            external=gwa[group],
            actual=fit_labels[group],
            train_index=train_indexes[group],
            apply_index=apply_indexes[group],
            model_dir=args.out_dir / "stage1/models",
            prefix="stage1",
        )
        candidate.loc[apply_indexes[group], group] = _apply_increment(
            baseline.loc[apply_indexes[group], group], increment, group
        )
        training[group]["feature"] = describe_file(feature_path)
    candidate_path = args.out_dir / "stage1/candidate_gwa3_v2_w010.parquet"
    _atomic_parquet(candidate, candidate_path)
    guard.verify_live()
    if closure != _source_closure(args, config):
        raise AssertionError("source closure changed before Stage1 candidate lock")
    lock_path = args.out_dir / "stage1_candidate_before_score_labels_lock.json"
    _write_json(
        lock_path,
        {
            "schema_version": 2,
            "lock_kind": "stage1_candidate_before_score_labels",
            "created_utc": utc_now(),
            "config_sha256": CONFIG_SHA,
            "candidate": describe_file(candidate_path),
            "baseline": describe_file(args.out_dir / "stage1/baseline_corrected_v3.parquet"),
            "training": training,
            "fit_label_evidence": fit_evidence,
            "raw_feature_contract": raw_contract,
            "continuous_heavy_guard": guard.record(),
            "score_label_value_cells_before_lock": 0,
            "candidate_metric_values_before_lock": 0,
            "2024_label_value_cells_before_lock": 0,
            "2025_contest_value_cells_before_lock": 0,
        },
    )
    score_labels, score_evidence = _stage1_score_labels(lineage, lock_path, guard)
    groups, mixed, gate = _stage1_metrics(score_labels, baseline, candidate)
    guard.verify_live()
    result_path = args.out_dir / "stage1_results.json"
    _write_json(
        result_path,
        {
            "schema_version": 2,
            "created_utc": utc_now(),
            "config_sha256": CONFIG_SHA,
            "candidate_before_score_labels_lock": describe_file(lock_path),
            "score_label_evidence_after_candidate_lock": score_evidence,
            "group_comparisons": groups,
            "mixed_comparisons": mixed,
            "strict_gate": gate,
            "stage1_passed": gate["passed"],
            "stage2_allowed": gate["passed"],
            "continuous_heavy_guard": guard.record(),
            "2024_label_value_cells_read": 0,
            "2025_contest_value_cells_read": 0,
        },
    )
    _write_json(
        args.out_dir / "stage1_selection_lock.json",
        {
            "schema_version": 2,
            "created_utc": utc_now(),
            "config_sha256": CONFIG_SHA,
            "stage1_results": describe_file(result_path),
            "stage1_passed": gate["passed"],
            "stage2_allowed": gate["passed"],
            "continuous_heavy_guard_through_result_seal": guard.record(),
            "2024_label_value_cells_read": 0,
            "2025_contest_value_cells_read": 0,
        },
    )
    print(json.dumps({"stage1_gate": gate}, indent=2), flush=True)
    return bool(gate["passed"])


def _load_stage2_baseline(lineage: Mapping[str, Any]) -> pd.DataFrame:
    spec = lineage["official_data_and_lineage"]["stage2_interaction"]
    path = _verify(
        PROJECT_DIR / spec["path"], size=int(spec["bytes"]), sha=str(spec["sha256"])
    )
    frame = pd.read_parquet(path).astype(np.float64)
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if tuple(frame.columns) != TARGET_COLS or not frame.index.equals(YEAR_2024):
        raise AssertionError("deployment-parity Stage2 baseline differs")
    return frame


def _run_stage2(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    lineage: Mapping[str, Any],
    closure: Mapping[str, Any],
    static: pd.DataFrame,
    guard: HeavyFitGuard,
) -> tuple[bool, pd.DataFrame | None, dict[str, pd.DataFrame] | None]:
    stage1_lock = json.loads(
        _verify(args.out_dir / "stage1_selection_lock.json").read_text(encoding="utf-8")
    )
    if not stage1_lock.get("stage1_passed") or not stage1_lock.get("stage2_allowed"):
        raise AssertionError("Stage2 opened without exact Stage1 pass")
    guard.verify_live()
    label_spec = lineage["official_data_and_lineage"]["label_file"]
    label_path = _verify(
        Path(label_spec["path"]), size=int(label_spec["bytes"]), sha=str(label_spec["sha256"])
    )
    fit_labels, fit_evidence = _bounded_label_slice(label_path, LABEL_FIT_SPEC, PRE2024)
    placeholder = pd.DataFrame(index=THROUGH_2024)
    features = protocol.shared._read_features(
        args.cache_dir, placeholder, expected_end=pd.Timestamp("2025-01-01 00:00")
    )
    gwa = {group: build_gwa3_features(features[group], static, group=group) for group in TARGET_COLS}
    baseline = _load_stage2_baseline(lineage)
    _atomic_parquet(baseline, args.out_dir / "stage2/baseline_recent_v4_2024.parquet")
    candidate = baseline.copy()
    training: dict[str, Any] = {}
    for group in TARGET_COLS:
        guard.verify_live()
        increment, training[group] = _fit_pair(
            group=group,
            control=features[group],
            external=gwa[group],
            actual=fit_labels[group],
            train_index=PRE2024,
            apply_index=YEAR_2024,
            model_dir=args.out_dir / "stage2/models",
            prefix="stage2",
        )
        candidate.loc[:, group] = _apply_increment(baseline[group], increment, group)
    candidate_path = args.out_dir / "stage2/candidate_gwa3_v2_w010_2024.parquet"
    _atomic_parquet(candidate, candidate_path)
    guard.verify_live()
    if closure != _source_closure(args, config):
        raise AssertionError("source closure changed before Stage2 candidate lock")
    lock_path = args.out_dir / "stage2_candidate_before_2024_score_labels_lock.json"
    _write_json(
        lock_path,
        {
            "schema_version": 2,
            "lock_kind": "stage2_candidate_before_2024_score_labels",
            "created_utc": utc_now(),
            "config_sha256": CONFIG_SHA,
            "stage1_selection_lock": describe_file(args.out_dir / "stage1_selection_lock.json"),
            "candidate": describe_file(candidate_path),
            "baseline": describe_file(args.out_dir / "stage2/baseline_recent_v4_2024.parquet"),
            "training": training,
            "fit_label_evidence": fit_evidence,
            "continuous_heavy_guard": guard.record(),
            "2024_score_label_value_cells_before_lock": 0,
            "candidate_metric_values_before_lock": 0,
            "2025_contest_value_cells_before_lock": 0,
        },
    )
    _verify_lock(
        lock_path,
        kind="stage2_candidate_before_2024_score_labels",
        guard=guard,
        zero_field="2024_score_label_value_cells_before_lock",
    )
    score_labels, score_evidence = _bounded_label_slice(label_path, LABEL_2024_SPEC, YEAR_2024)
    segments = _segments_2024()
    group_records: dict[str, Any] = {}
    group_full: list[float] = []
    for group in TARGET_COLS:
        group_records[group] = {}
        for name, idx in segments.items():
            group_records[group][name] = protocol._comparison(
                score_labels.loc[idx, group], baseline.loc[idx, group], candidate.loc[idx, group], group
            )
        group_full.append(float(group_records[group]["full"]["delta_total_score"]))
    mixed = {
        name: protocol._mixed_comparison(score_labels.loc[idx], baseline.loc[idx], candidate.loc[idx], TARGET_COLS)
        for name, idx in segments.items()
    }
    passed = bool(
        all(value > 0.0 for value in group_full)
        and float(mixed["full"]["delta_total_score"]) > 0.0
        and float(mixed["full"]["delta_one_minus_nmae"]) >= 0.0
        and float(mixed["full"]["delta_ficr"]) >= 0.0
        and float(mixed["H2"]["delta_total_score"]) >= 0.0
    )
    gate = {
        "group_full_deltas": group_full,
        "mixed_full_score": float(mixed["full"]["delta_total_score"]),
        "mixed_full_one_minus_nmae": float(mixed["full"]["delta_one_minus_nmae"]),
        "mixed_full_ficr": float(mixed["full"]["delta_ficr"]),
        "mixed_h2_score": float(mixed["H2"]["delta_total_score"]),
        "passed": passed,
    }
    result_path = args.out_dir / "stage2_results.json"
    _write_json(
        result_path,
        {
            "schema_version": 2,
            "created_utc": utc_now(),
            "config_sha256": CONFIG_SHA,
            "candidate_before_2024_score_labels_lock": describe_file(lock_path),
            "score_label_evidence_after_candidate_lock": score_evidence,
            "group_comparisons": group_records,
            "mixed_comparisons": mixed,
            "strict_gate": gate,
            "stage2_passed": passed,
            "final_allowed": passed,
            "continuous_heavy_guard": guard.record(),
            "2025_contest_value_cells_read": 0,
        },
    )
    _write_json(
        args.out_dir / "stage2_selection_lock.json",
        {
            "schema_version": 2,
            "created_utc": utc_now(),
            "config_sha256": CONFIG_SHA,
            "stage2_results": describe_file(result_path),
            "stage2_passed": passed,
            "final_allowed": passed,
            "continuous_heavy_guard_through_result_seal": guard.record(),
            "2025_contest_value_cells_read": 0,
        },
    )
    print(json.dumps({"stage2_gate": gate}, indent=2), flush=True)
    return passed, (pd.concat([fit_labels, score_labels]) if passed else None), (features if passed else None)


def _run_final(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    closure: Mapping[str, Any],
    static: pd.DataFrame,
    all_labels: pd.DataFrame,
    train_features: Mapping[str, pd.DataFrame],
    guard: HeavyFitGuard,
) -> Path:
    selection = json.loads(
        _verify(args.out_dir / "stage2_selection_lock.json").read_text(encoding="utf-8")
    )
    if not selection.get("stage2_passed") or not selection.get("final_allowed"):
        raise AssertionError("final opened without exact Stage2 pass")
    guard.verify_live()
    test_features = protocol.shared._read_test_features(args.cache_dir, TEST_INDEX)
    gwa_train = {
        group: build_gwa3_features(train_features[group], static, group=group) for group in TARGET_COLS
    }
    gwa_test = {
        group: build_gwa3_features(test_features[group], static, group=group) for group in TARGET_COLS
    }
    baseline_path = _verify(
        PROJECT_DIR / "artifacts/final_cf_fix/predictions/corrected_recent_v4_test.parquet"
    )
    baseline = pd.read_parquet(baseline_path).astype(np.float64)
    baseline.index = pd.DatetimeIndex(baseline.index, name="forecast_kst_dtm")
    if tuple(baseline.columns) != TARGET_COLS or not baseline.index.equals(TEST_INDEX):
        raise AssertionError("2025 corrected-recent-v4 baseline differs")
    prediction = baseline.copy()
    training: dict[str, Any] = {}
    for group in TARGET_COLS:
        guard.verify_live()
        # The train/apply frames differ in the final stage, so fit the same pair explicitly.
        x_train = train_features[group].loc[THROUGH_2024]
        x_apply = test_features[group].loc[TEST_INDEX]
        y_train = all_labels.loc[THROUGH_2024, group]
        mask = eligible_rows(y_train, CAPACITY_KWH[group])
        control_model, control_meta = fit_direct_model(
            x_train, y_train, capacity_kwh=CAPACITY_KWH[group], eligible=mask
        )
        ex_train = pd.concat([x_train, gwa_train[group].loc[THROUGH_2024]], axis=1).astype(np.float32)
        ex_apply = pd.concat([x_apply, gwa_test[group].loc[TEST_INDEX]], axis=1).astype(np.float32)
        if tuple(ex_train.columns[-24:]) != EXTENDED_COLUMNS or ex_train.shape[1] != 636:
            raise AssertionError("final extended schema differs")
        extended_model, extended_meta = fit_direct_model(
            ex_train, y_train, capacity_kwh=CAPACITY_KWH[group], eligible=mask
        )
        if control_model.get_params() != extended_model.get_params():
            raise AssertionError("final paired parameters differ")
        control_path = args.out_dir / f"final/models/final_{group}_control.joblib"
        extended_path = args.out_dir / f"final/models/final_{group}_extended.joblib"
        _atomic_joblib(control_model, control_path)
        _atomic_joblib(extended_model, extended_path)
        control_cf = predict_cf(joblib.load(control_path), x_apply)
        extended_cf = predict_cf(joblib.load(extended_path), ex_apply)
        delta = extended_cf - control_cf
        prediction[group] = _apply_increment(baseline[group], delta, group)
        training[group] = {
            "control": {**control_meta, "model": describe_file(control_path)},
            "extended": {**extended_meta, "model": describe_file(extended_path)},
            "same_rows_target_mask_seed_parameters": True,
            "increment_cf_mean": float(delta.mean()),
            "increment_cf_std": float(delta.std()),
        }
    if closure != _source_closure(args, config):
        raise AssertionError("source closure changed before final seal")
    prediction_path = args.out_dir / "final/gwa3_static_resource_v2_w010_2025.parquet"
    _atomic_parquet(prediction, prediction_path)
    sample_path = _verify(args.raw_dir / "sample_submission.csv")
    sample = pd.read_csv(
        sample_path,
        encoding="utf-8-sig",
        dtype={"forecast_id": str, "forecast_kst_dtm": str},
    )
    sample_index = pd.DatetimeIndex(
        pd.to_datetime(sample["forecast_kst_dtm"], errors="raise"), name="forecast_kst_dtm"
    )
    if tuple(sample.columns) != ("forecast_id", "forecast_kst_dtm", *TARGET_COLS):
        raise AssertionError("sample submission columns differ")
    if len(sample) != 8760 or not sample_index.equals(TEST_INDEX):
        raise AssertionError("sample submission row/time identity differs")
    output = sample.copy()
    rounded = prediction.round(6)
    for group in TARGET_COLS:
        output[group] = rounded[group].to_numpy(np.float64)
    csv_path = args.out_dir / "final/gwa3_static_resource_v2_w010_2025.csv"
    output.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
        float_format="%.6f",
    )
    reloaded = pd.read_csv(
        csv_path,
        encoding="utf-8-sig",
        dtype={"forecast_id": str, "forecast_kst_dtm": str},
    )
    values = reloaded.loc[:, list(TARGET_COLS)].to_numpy(np.float64)
    if csv_path.read_bytes()[:3] != b"\xef\xbb\xbf":
        raise AssertionError("CSV UTF-8 BOM differs")
    if tuple(reloaded.columns) != tuple(sample.columns) or len(reloaded) != 8760:
        raise AssertionError("CSV schema/rows differ")
    if not reloaded["forecast_id"].equals(sample["forecast_id"]):
        raise AssertionError("CSV forecast_id order differs")
    if not reloaded["forecast_kst_dtm"].equals(sample["forecast_kst_dtm"]):
        raise AssertionError("CSV timestamp order differs")
    if not np.array_equal(values, rounded.to_numpy(np.float64)) or not np.isfinite(values).all():
        raise AssertionError("CSV numeric six-decimal replay differs")
    for position, group in enumerate(TARGET_COLS):
        if (values[:, position] < 0.0).any() or (
            values[:, position] > 1.02 * CAPACITY_KWH[group]
        ).any():
            raise AssertionError(f"CSV physical bounds differ: {group}")
    _write_json(
        args.out_dir / "final_audit.json",
        {
            "schema_version": 2,
            "created_utc": utc_now(),
            "experiment_id": EXPERIMENT_ID,
            "config_sha256": CONFIG_SHA,
            "stage2_selection_lock": describe_file(args.out_dir / "stage2_selection_lock.json"),
            "baseline": describe_file(baseline_path),
            "prediction": describe_file(prediction_path),
            "csv": describe_file(csv_path),
            "training": training,
            "continuous_heavy_guard_through_final_seal": guard.record(),
            "csv_contract": {
                "rows": 8760,
                "utf8_bom": True,
                "exact_sample_id_time_order": True,
                "six_decimal_numeric_replay": True,
                "finite_and_capacity_bounded": True,
            },
            "public_feedback_values_read": 0,
            "v1_outputs_read_or_reused": 0,
        },
    )
    return csv_path


def _terminal_manifest(
    args: argparse.Namespace,
    closure: Mapping[str, Any],
    status: str,
    guard: HeavyFitGuard,
) -> None:
    guard.verify_live()
    files = [
        describe_file(path)
        for path in sorted(args.out_dir.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    ]
    _write_json(
        args.out_dir / "manifest.json",
        {
            "schema_version": 2,
            "experiment_id": EXPERIMENT_ID,
            "status": status,
            "created_utc": utc_now(),
            "config_sha256": CONFIG_SHA,
            "source_closure": closure,
            "attempt_tombstone": describe_file(ATTEMPT_TOMBSTONE),
            "continuous_heavy_guard_through_terminal_manifest": guard.record(),
            "files_excluding_manifest": files,
            "runtime": {"packages": package_versions()},
            "git": git_state(PROJECT_DIR),
            "v1_outputs_read_or_reused": 0,
            "public_feedback_subgroup_inference": False,
        },
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.raw_dir = args.raw_dir.resolve()
    args.cache_dir = args.cache_dir.resolve()
    args.config = args.config.resolve()
    args.out_dir = args.out_dir.resolve()
    config, lineage = _load_config(args)
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    if ATTEMPT_TOMBSTONE.exists():
        raise FileExistsError("v2 single-attempt tombstone already exists")
    static = pd.read_parquet(STATIC_PATH)
    if len(static) != 17 or static.isna().any().any():
        raise AssertionError("frozen GWA static table differs")
    closure = _source_closure(args, config)
    with HeavyFitGuard() as guard:
        args.out_dir.mkdir(parents=True)
        shutil.copyfile(args.config, args.out_dir / "preregister_v2.json")
        shutil.copyfile(args.config.with_suffix(".sha256"), args.out_dir / "preregister_v2.sha256")
        _write_json(
            ATTEMPT_TOMBSTONE,
            {
                "schema_version": 2,
                "created_utc": utc_now(),
                "experiment_id": EXPERIMENT_ID,
                "config_sha256": CONFIG_SHA,
                "runner": describe_file(Path(__file__)),
                "continuous_heavy_guard": guard.record(),
                "attempt_number": 1,
                "second_attempt_forbidden": True,
            },
        )
        _write_json(
            args.out_dir / "v2_prescore_lock.json",
            {
                "schema_version": 2,
                "created_utc": utc_now(),
                "config_sha256": CONFIG_SHA,
                "source_closure": closure,
                "static_table": describe_file(STATIC_PATH),
                "source_manifest": describe_file(SOURCE_MANIFEST_PATH),
                "attempt_tombstone": describe_file(ATTEMPT_TOMBSTONE),
                "continuous_heavy_guard": guard.record(),
                "models_fit": 0,
                "candidate_predictions_materialized": 0,
                "metric_calls": 0,
                "2024_label_values_read": 0,
                "2025_contest_values_read": 0,
            },
        )
        stage1_pass = _run_stage1(args, config, lineage, closure, static, guard)
        if not stage1_pass:
            _terminal_manifest(args, closure, "REJECTED_STAGE1_NO_2024_OR_2025_ACCESS", guard)
            return {"status": "REJECTED_STAGE1", "csv": None}
        stage2_pass, labels, features = _run_stage2(
            args, config, lineage, closure, static, guard
        )
        if not stage2_pass:
            _terminal_manifest(args, closure, "REJECTED_STAGE2_NO_2025_ACCESS", guard)
            return {"status": "REJECTED_STAGE2", "csv": None}
        assert labels is not None and features is not None
        csv_path = _run_final(args, config, closure, static, labels, features, guard)
        _terminal_manifest(args, closure, "STRICT_STAGE1_STAGE2_PASS_FINAL_CREATED", guard)
        return {"status": "FINAL_CREATED", "csv": describe_file(csv_path)}


def main(argv: Sequence[str] | None = None) -> None:
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Strict stored-array Stage1 for the fixed raw/smooth-FICR model average."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_ficr_bayes_decision_strict as bayes
from scripts import run_raw_grid_wind_lgb as raw_protocol
from scripts import run_raw_spatiotemporal_smooth_ficr as smooth_runner
from scripts import run_weather_quantile_bayes as weather_protocol
from src.manifest import describe_file, git_state, package_versions, sha256_file, utc_now
from src.metric import CAPACITY_KWH, TARGET_COLS


CONFIG_PATH = ROOT / "configs/raw_attention_sficr_fixed_average_preregister_v1.json"
CONFIG_SHA256 = "d4a5dc7011c1c6125854b208750aed83f52e766c594966088f885b206e138031"
FEASIBILITY_PATH = ROOT / "artifacts/audits/raw_attention_sficr_fixed_average_feasibility_v1.json"
FEASIBILITY_SHA256 = "063a5de28651329dad883dfe77a170e5df0b927403d4975dbc9fdf154a2e74fb"
RAW_DEPENDENCY = ROOT / "configs/raw_grid_wind_lgb_preregister_v1.json"
RAW_DEPENDENCY_SHA256 = "06b56426347f64d91f1a69e1c56083014f0dd003ee8186fa98ffe97ad3b61d01"
DEFAULT_RAW_DIR = Path(r"data/local/open")
DEFAULT_OUT_DIR = ROOT / "artifacts/postgate/raw_attention_sficr_fixed_average_strict_v1"
CANDIDATE_ID = "raw_sficr_avg_w025"
DELTA_WEIGHT = 0.0125
YEAR_2023 = raw_protocol.YEAR_2023
PRE2024 = raw_protocol.PRE2024
G3_H2 = weather_protocol._year_segments(2023)["H2"]
REQUIRED: Mapping[str, tuple[str, ...]] = {
    TARGET_COLS[0]: ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4"),
    TARGET_COLS[1]: ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4"),
    TARGET_COLS[2]: ("full", "Q3", "Q4"),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("stage1", "manifest", "all"), required=True)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    return parser.parse_args(argv)


def _verify(path: Path, spec: Mapping[str, Any]) -> Path:
    path = path.resolve()
    if (
        not path.is_file()
        or path.stat().st_size != int(spec["bytes"])
        or sha256_file(path) != str(spec["sha256"])
    ):
        raise AssertionError(f"registered input changed: {path}")
    return path


def verify_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = path.resolve()
    if path != CONFIG_PATH.resolve() or path.stat().st_size != 8413 or sha256_file(path) != CONFIG_SHA256:
        raise AssertionError("preregistration changed")
    if path.with_suffix(".sha256").read_text(encoding="utf-8") != f"{CONFIG_SHA256}  {CONFIG_PATH.name}\n":
        raise AssertionError("preregistration sidecar changed")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["experiment_id"] != "raw_attention_sficr_fixed_average_strict_forward_v1":
        raise AssertionError("experiment id changed")
    candidate = config["candidate"]
    if (
        candidate["count"] != 1
        or candidate["id"] != CANDIDATE_ID
        or float(candidate["total_blend_weight"]) != 0.025
        or candidate["constituent_model_average_weights"] != [0.5, 0.5]
    ):
        raise AssertionError("candidate contract changed")
    if int(config["stage1_gate"]["registered_slice_count"]) != 17:
        raise AssertionError("registered slice count changed")
    _verify(FEASIBILITY_PATH, config["feasibility"])
    if sha256_file(RAW_DEPENDENCY) != RAW_DEPENDENCY_SHA256:
        raise AssertionError("physical label contract changed")
    return config, json.loads(RAW_DEPENDENCY.read_text(encoding="utf-8"))


def _all_stage1_specs(config: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    fixed = config["fixed_source_arrays"]
    specs: list[Mapping[str, Any]] = []
    for family in ("raw_attention", "smooth_ficr_attention"):
        for value in fixed[family].values():
            if isinstance(value, dict) and "path" in value:
                specs.append(value)
    for value in fixed["baseline"].values():
        if isinstance(value, dict) and "path" in value:
            specs.append(value)
    return specs


def source_closure(config: Mapping[str, Any], raw_dir: Path) -> dict[str, Any]:
    closure = smooth_runner.resolve_ast_closure(Path(__file__))
    explicit = [
        CONFIG_PATH,
        CONFIG_PATH.with_suffix(".sha256"),
        FEASIBILITY_PATH,
        ROOT / "tests/test_raw_attention_sficr_fixed_average.py",
    ]
    fixed = [_verify(ROOT / spec["path"], spec) for spec in _all_stage1_specs(config)]
    label_path = raw_dir / "train/train_labels.csv"
    return {
        "resolver": "recursive Python AST local imports plus explicit sources",
        "resolved_relative_paths": [path.relative_to(ROOT).as_posix() for path in closure],
        "resolved_files": [describe_file(path) for path in closure],
        "explicit_sources": [describe_file(path.resolve()) for path in explicit],
        "fixed_target_free_stage1_inputs": [describe_file(path) for path in fixed],
        "label_metadata_only": {
            "path": str(label_path.resolve()),
            "bytes": label_path.stat().st_size,
            "content_values_materialized": 0,
        },
        "2024_or_2025_value_inputs": [],
    }


def _segments(group: str) -> dict[str, pd.DatetimeIndex]:
    year = weather_protocol._year_segments(2023)
    if group == TARGET_COLS[2]:
        return {"full": year["H2"], "Q3": year["Q3"], "Q4": year["Q4"]}
    return {name: year[name] for name in REQUIRED[group]}


def fixed_candidate(
    baseline_kwh: pd.Series,
    raw_cf: pd.Series,
    smooth_cf: pd.Series,
    *,
    capacity_kwh: float,
) -> pd.Series:
    if not baseline_kwh.index.equals(raw_cf.index) or not baseline_kwh.index.equals(smooth_cf.index):
        raise ValueError("constituent indexes differ")
    baseline = baseline_kwh.to_numpy(dtype=np.float64)
    raw_delta = np.clip(raw_cf.to_numpy(dtype=np.float64), 0.0, 1.02) * capacity_kwh - baseline
    smooth_delta = np.clip(smooth_cf.to_numpy(dtype=np.float64), 0.0, 1.02) * capacity_kwh - baseline
    values = np.clip(
        baseline + DELTA_WEIGHT * raw_delta + DELTA_WEIGHT * smooth_delta,
        0.0,
        1.02 * capacity_kwh,
    )
    if not np.isfinite(values).all():
        raise ValueError("fixed candidate contains non-finite values")
    return pd.Series(values, index=baseline_kwh.index, name=baseline_kwh.name)


def gate(comparisons: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if set(comparisons) != set(TARGET_COLS):
        raise ValueError("group set changed")
    groups: dict[str, Any] = {}
    passed = True
    for group in TARGET_COLS:
        records = comparisons[group]
        if tuple(records) != REQUIRED[group]:
            raise ValueError(f"slice order changed for {group}")
        deltas = {name: float(records[name]["delta"]) for name in REQUIRED[group]}
        for name in REQUIRED[group]:
            exact = float(records[name]["candidate"]["score"]) - float(
                records[name]["baseline"]["score"]
            )
            if deltas[name] != exact:
                raise AssertionError("metric delta arithmetic changed")
        full = records["full"]
        nmae = float(full["candidate"]["one_minus_nmae"]) - float(
            full["baseline"]["one_minus_nmae"]
        )
        ficr = float(full["candidate"]["ficr"]) - float(full["baseline"]["ficr"])
        row_pass = all(value > 0.0 for value in deltas.values()) and nmae >= 0.0 and ficr >= 0.0
        groups[group] = {
            "deltas": deltas,
            "positive_slices": sum(value > 0.0 for value in deltas.values()),
            "slice_count": len(deltas),
            "minimum": min(deltas.values()),
            "mean": float(np.mean(list(deltas.values()))),
            "full_one_minus_nmae_delta": nmae,
            "full_ficr_delta": ficr,
            "passed": row_pass,
        }
        passed = passed and row_pass
    return {
        "groups": groups,
        "registered_slice_count": sum(row["slice_count"] for row in groups.values()),
        "passed_all_groups": passed,
        "decision": CANDIDATE_ID if passed else "identity",
    }


def _require_candidate_lock(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("candidate-before-label lock is absent")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("preregister_sha256") != CONFIG_SHA256 or payload.get("all_source_and_candidate_arrays_frozen") is not True:
        raise RuntimeError("candidate-before-label lock is invalid")
    return payload


def _read_score_after_lock(
    raw_dir: Path, prefix: Mapping[str, Any], lock_path: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _require_candidate_lock(lock_path)
    return raw_protocol._read_label_prefix(raw_dir, prefix, TARGET_COLS)


def run_stage1(
    *, out_dir: Path, raw_dir: Path, config: Mapping[str, Any], raw_contract: Mapping[str, Any]
) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(out_dir)
    out_dir.mkdir(parents=True)
    for path in (CONFIG_PATH, CONFIG_PATH.with_suffix(".sha256"), FEASIBILITY_PATH):
        bayes._copy_exclusive(path, out_dir / path.name)
    closure = source_closure(config, raw_dir)
    source_lock = out_dir / "source_lock_before_candidate_or_labels.json"
    bayes._write_json(
        source_lock,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": CONFIG_SHA256,
            "source_closure": closure,
            "candidate_arrays_materialized": 0,
            "application_label_cells_materialized": 0,
            "2024_or_2025_values_read": 0,
        },
    )

    fixed = config["fixed_source_arrays"]
    raw_g12 = pd.read_parquet(_verify(ROOT / fixed["raw_attention"]["g12_prediction"]["path"], fixed["raw_attention"]["g12_prediction"]))
    raw_g3 = pd.read_parquet(_verify(ROOT / fixed["raw_attention"]["g3_prediction"]["path"], fixed["raw_attention"]["g3_prediction"]))
    smooth_g12 = pd.read_parquet(_verify(ROOT / fixed["smooth_ficr_attention"]["g12_prediction"]["path"], fixed["smooth_ficr_attention"]["g12_prediction"]))
    smooth_g3 = pd.read_parquet(_verify(ROOT / fixed["smooth_ficr_attention"]["g3_prediction"]["path"], fixed["smooth_ficr_attention"]["g3_prediction"]))
    if not raw_g12.index.equals(smooth_g12.index) or not raw_g3.index.equals(smooth_g3.index):
        raise AssertionError("constituent application indexes changed")

    candidates: dict[str, pd.Series] = {}
    paths: list[Path] = []
    for group in TARGET_COLS:
        fit = "g12" if group in TARGET_COLS[:2] else "g3"
        raw = raw_g12 if fit == "g12" else raw_g3
        smooth = smooth_g12 if fit == "g12" else smooth_g3
        baseline_spec = fixed["baseline"][
            "g12_group_1" if group == TARGET_COLS[0] else
            "g12_group_2" if group == TARGET_COLS[1] else "g3_group_3"
        ]
        baseline = pd.read_parquet(_verify(ROOT / baseline_spec["path"], baseline_spec))[group]
        if not baseline.index.equals(raw.index) or tuple(raw.columns) != tuple(smooth.columns):
            raise AssertionError("stored source/baseline alignment changed")
        candidate = fixed_candidate(
            baseline, raw[group], smooth[group], capacity_kwh=CAPACITY_KWH[group]
        )
        candidates[group] = candidate
        source_path = out_dir / "stage1" / f"{group}__sources.parquet"
        candidate_path = out_dir / "stage1" / f"{group}__candidate.parquet"
        baseline_path = out_dir / "stage1" / f"{group}__baseline.parquet"
        bayes._atomic_parquet(
            pd.DataFrame(
                {
                    "baseline_kwh": baseline,
                    "raw_attention_cf": raw[group],
                    "smooth_ficr_attention_cf": smooth[group],
                }
            ),
            source_path,
        )
        bayes._atomic_parquet(candidate.to_frame(group), candidate_path)
        bayes._atomic_parquet(baseline.to_frame(group), baseline_path)
        if not np.array_equal(pd.read_parquet(candidate_path)[group].to_numpy(), candidate.to_numpy()):
            raise AssertionError("candidate parquet roundtrip changed")
        paths.extend((source_path, candidate_path, baseline_path))

    candidate_record = out_dir / "candidate_record_before_application_labels.json"
    bayes._write_json(
        candidate_record,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": CONFIG_SHA256,
            "source_lock": describe_file(source_lock),
            "candidate_id": CANDIDATE_ID,
            "ordered_formula": config["candidate"]["ordered_formula"],
            "outputs": [describe_file(path) for path in paths],
            "candidate_array_hashes": {
                group: hashlib.sha256(
                    np.ascontiguousarray(candidates[group].to_numpy(np.float64)).tobytes()
                ).hexdigest()
                for group in TARGET_COLS
            },
            "metric_calls": 0,
            "application_label_cells_materialized": 0,
        },
    )
    candidate_lock = out_dir / "candidate_lock_before_application_labels.json"
    bayes._write_json(
        candidate_lock,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": CONFIG_SHA256,
            "candidate_record": describe_file(candidate_record),
            "candidate_artifacts": [describe_file(path) for path in paths],
            "all_source_and_candidate_arrays_frozen": True,
            "metric_calls": 0,
        },
    )

    prefix = raw_contract["physical_stage1_inputs"]["labels_stage1_score_prefix_after_lock"]
    labels, evidence = _read_score_after_lock(raw_dir, prefix, candidate_lock)
    if not labels.index.equals(PRE2024):
        raise AssertionError("score label prefix changed")
    score_access = out_dir / "score_label_access_after_candidate_lock.json"
    bayes._write_json(
        score_access,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "candidate_lock": describe_file(candidate_lock),
            "evidence": evidence,
            "metric_calls_before_access_record": 0,
        },
    )
    comparisons: dict[str, Any] = {}
    for group in TARGET_COLS:
        application = YEAR_2023 if group in TARGET_COLS[:2] else G3_H2
        baseline = pd.read_parquet(out_dir / "stage1" / f"{group}__baseline.parquet")[group]
        comparisons[group] = bayes._comparison(
            labels.loc[application, group],
            baseline,
            candidates[group],
            group,
            _segments(group),
        )
    selection = gate(comparisons)
    if selection["registered_slice_count"] != 17:
        raise AssertionError("registered metric count changed")
    if json.dumps(closure, sort_keys=True, default=str) != json.dumps(
        source_closure(config, raw_dir), sort_keys=True, default=str
    ):
        raise AssertionError("source closure changed during Stage1")
    result = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "preregister_sha256": CONFIG_SHA256,
        "risk_classification": config["risk_classification"],
        "candidate_lock": describe_file(candidate_lock),
        "score_label_access": describe_file(score_access),
        "comparisons": comparisons,
        "gate": selection,
        "stage2_unlocked": bool(selection["passed_all_groups"]),
        "2024_values_read": False,
        "2025_values_read": False,
        "csv_created": False,
    }
    result_path = out_dir / "stage1_results.json"
    bayes._write_json(result_path, result)
    promotion_path = out_dir / "stage1_promotion_lock.json"
    bayes._write_json(
        promotion_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": CONFIG_SHA256,
            "candidate_lock": describe_file(candidate_lock),
            "score_label_access": describe_file(score_access),
            "stage1_results": describe_file(result_path),
            "decision": selection["decision"],
            "stage2_unlocked": bool(selection["passed_all_groups"]),
            "no_group_rescue": True,
            "no_2024_or_2025_read": True,
        },
    )
    if not selection["passed_all_groups"]:
        bayes._write_json(
            out_dir / "stage1_rejection.json",
            {
                "schema_version": 1,
                "created_utc": utc_now(),
                "promotion_lock": describe_file(promotion_path),
                "decision": "REJECT_IDENTITY",
                "selection_unsafe_result_informed": True,
                "no_retune_grid_group_rescue_2024_2025_or_csv": True,
            },
        )
    return result


def write_manifest(*, out_dir: Path, raw_dir: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    path = out_dir / "manifest.json"
    sidecar = out_dir / "manifest.sha256"
    if path.exists() or sidecar.exists():
        raise FileExistsError("manifest exists")
    result = json.loads((out_dir / "stage1_results.json").read_text(encoding="utf-8"))
    outputs = sorted(
        item for item in out_dir.rglob("*") if item.is_file() and item not in (path, sidecar)
    )
    payload = {
        "schema_version": 1,
        "artifact_type": "raw_attention_sficr_fixed_average_strict_forward_v1_stage1",
        "created_utc": utc_now(),
        "command": [sys.executable, *sys.argv],
        "preregister_sha256": CONFIG_SHA256,
        "feasibility_sha256": FEASIBILITY_SHA256,
        "risk": config["risk_classification"],
        "source_closure": source_closure(config, raw_dir),
        "selection": result["gate"],
        "read_flags": {
            "2024": False,
            "2025": False,
            "sample": False,
            "csv": False,
        },
        "tests": {"focused_expected": 8, "focused_status": "pass"},
        "runtime": {"packages": package_versions(), "git": git_state(ROOT)},
        "outputs_excluding_manifest_and_sidecar": [describe_file(item) for item in outputs],
        "output_count_excluding_manifest_and_sidecar": len(outputs),
    }
    bayes._write_json(path, payload)
    with sidecar.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(f"{sha256_file(path)}  manifest.json\n")
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config, raw_contract = verify_config(args.config)
    out_dir = args.out_dir.resolve()
    raw_dir = args.raw_dir.resolve()
    if args.stage in ("stage1", "all"):
        run_stage1(out_dir=out_dir, raw_dir=raw_dir, config=config, raw_contract=raw_contract)
    if args.stage in ("manifest", "all"):
        write_manifest(out_dir=out_dir, raw_dir=raw_dir, config=config)


if __name__ == "__main__":
    main()

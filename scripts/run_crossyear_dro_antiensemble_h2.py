"""Run the preregistered H2 cross-year anti-ensemble DRO experiment.

Stage 1 is intentionally self-contained: it constructs and hashes every 2023
candidate before parsing any label value, then selects at most one shared alpha.
No 2024 or 2025 value is touched unless a non-identity Stage-1 lock exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.manifest import describe_file, sha256_file, write_json_atomic  # noqa: E402
from src.metric import CAPACITY_KWH, TARGET_COLS, group_metrics, score_details  # noqa: E402


PREREG_SHA256 = "42255b097e4d0e38a867f1dd42cd45eb63f549dde353fd84dac1d79c69879f9c"
CENSUS_SHA256 = "9ffd3875f94f58560d9db1ab4989bdabac823710e4cc587d2b6b514ea01c177b"
GROUPS = tuple(TARGET_COLS)
FAMILIES = (
    "lgbm_huber",
    "xgb_l1",
    "xgb_pseudohuber",
    "catboost_mae",
    "catboost_huber",
    "extra_trees_l1",
    "hist_gbdt_l1",
)
ALPHAS = (-0.025, -0.05, -0.075, -0.1)
H2_INDEX = pd.date_range("2023-07-01 01:00:00", "2024-01-01 00:00:00", freq="h")
BLOCKS_2023 = {
    "H2": (pd.Timestamp("2023-07-01 01:00:00"), pd.Timestamp("2024-01-01 00:00:00")),
    "Q3": (pd.Timestamp("2023-07-01 01:00:00"), pd.Timestamp("2023-10-01 00:00:00")),
    "Q4": (pd.Timestamp("2023-10-01 01:00:00"), pd.Timestamp("2024-01-01 00:00:00")),
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _frame_digest(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps([(str(c), str(t)) for c, t in frame.dtypes.items()]).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    write_json_atomic(path, _jsonable(value), overwrite=False)


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        frame.to_parquet(temporary, compression="zstd", index=True)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _prediction_path(root: Path, group: str, family: str) -> Path:
    return root / "postgate/model_zoo_diversity/stage1/predictions" / f"{group}__{family}.parquet"


def _read_stage1_arrays(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    baseline = pd.DataFrame(index=H2_INDEX, columns=GROUPS, dtype="float64")
    family_mean = pd.DataFrame(index=H2_INDEX, columns=GROUPS, dtype="float64")
    paths: list[Path] = []
    for group in GROUPS:
        expected_full = (
            pd.date_range("2023-01-01 01:00:00", "2024-01-01 00:00:00", freq="h")
            if group != "kpx_group_3"
            else H2_INDEX
        )
        bases: list[np.ndarray] = []
        candidates: list[np.ndarray] = []
        for family in FAMILIES:
            path = _prediction_path(root, group, family)
            frame = pd.read_parquet(path)
            paths.append(path)
            if list(frame.columns) != ["locked_v3_kwh", "candidate_kwh", "blended_kwh"]:
                raise ValueError(f"unexpected columns: {path}")
            if len(frame) != len(expected_full):
                raise ValueError(f"unexpected rows: {path}")
            values = frame[["locked_v3_kwh", "candidate_kwh"]].to_numpy(dtype="float64")
            if not np.isfinite(values).all():
                raise ValueError(f"non-finite source: {path}")
            bases.append(values[:, 0])
            candidates.append(values[:, 1])
        base_matrix = np.column_stack(bases)
        if not np.array_equal(base_matrix, np.repeat(base_matrix[:, :1], len(FAMILIES), axis=1)):
            raise AssertionError(f"locked-v3 disagreement across family files: {group}")
        candidate_matrix = np.column_stack(candidates)
        full = pd.DataFrame(
            {
                "baseline": base_matrix[:, 0],
                "family_mean": np.mean(candidate_matrix, axis=1),
            },
            index=expected_full,
        ).loc[H2_INDEX]
        baseline[group] = full["baseline"]
        family_mean[group] = full["family_mean"]
    if not np.isfinite(baseline.to_numpy()).all() or not np.isfinite(family_mean.to_numpy()).all():
        raise AssertionError("assembled arrays contain non-finite values")
    return baseline, family_mean, paths


def _candidate(baseline: pd.DataFrame, family_mean: pd.DataFrame, alpha: float) -> pd.DataFrame:
    output = baseline + float(alpha) * (family_mean - baseline)
    for group in GROUPS:
        output[group] = np.clip(output[group].to_numpy(), 0.0, 1.02 * CAPACITY_KWH[group])
    return output


def _metric_dict(answer: pd.DataFrame, prediction: pd.DataFrame, groups: tuple[str, ...]) -> dict[str, float]:
    if len(groups) == 1:
        group = groups[0]
        values = group_metrics(answer[group], prediction[group], CAPACITY_KWH[group], group_name=group)
        return {
            "score": 0.5 * values.one_minus_nmae + 0.5 * values.ficr,
            "one_minus_nmae": values.one_minus_nmae,
            "ficr": values.ficr,
            "n_evaluated": values.n_evaluated,
        }
    values = score_details(answer[list(groups)], prediction[list(groups)], target_cols=groups)
    return {
        "score": values.total_score,
        "one_minus_nmae": values.one_minus_nmae,
        "ficr": values.ficr,
        "n_evaluated": sum(item.n_evaluated for item in values.by_group.values()),
    }


def _compare(answer: pd.DataFrame, baseline: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {"groups": {}, "mixed": {}}
    for block, (start, end) in BLOCKS_2023.items():
        mask = (answer.index >= start) & (answer.index <= end)
        for group in GROUPS:
            before = _metric_dict(answer.loc[mask], baseline.loc[mask], (group,))
            after = _metric_dict(answer.loc[mask], candidate.loc[mask], (group,))
            result["groups"].setdefault(group, {})[block] = {
                "baseline": before,
                "candidate": after,
                "delta": {key: after[key] - before[key] for key in ("score", "one_minus_nmae", "ficr")},
            }
        before = _metric_dict(answer.loc[mask], baseline.loc[mask], GROUPS)
        after = _metric_dict(answer.loc[mask], candidate.loc[mask], GROUPS)
        result["mixed"][block] = {
            "baseline": before,
            "candidate": after,
            "delta": {key: after[key] - before[key] for key in ("score", "one_minus_nmae", "ficr")},
        }
    return result


def _eligible(comparison: Mapping[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for group in GROUPS:
        h2 = comparison["groups"][group]["H2"]["delta"]
        if h2["score"] < 0.0:
            failures.append(f"{group}/H2 score")
        if h2["ficr"] < 0.0:
            failures.append(f"{group}/H2 ficr")
        if h2["one_minus_nmae"] < -0.0015:
            failures.append(f"{group}/H2 one_minus_nmae")
        for quarter in ("Q3", "Q4"):
            if comparison["groups"][group][quarter]["delta"]["score"] < -0.0005:
                failures.append(f"{group}/{quarter} score")
    mixed = comparison["mixed"]
    if mixed["H2"]["delta"]["score"] < 0.001:
        failures.append("mixed/H2 score")
    if mixed["H2"]["delta"]["ficr"] < 0.002:
        failures.append("mixed/H2 ficr")
    if mixed["H2"]["delta"]["one_minus_nmae"] < -0.001:
        failures.append("mixed/H2 one_minus_nmae")
    for quarter in ("Q3", "Q4"):
        if mixed[quarter]["delta"]["score"] < 0.0:
            failures.append(f"mixed/{quarter} score")
    return not failures, failures


def _objective(comparison: Mapping[str, Any], alpha: float) -> tuple[float, ...]:
    group_quarter_f = [
        comparison["groups"][group][quarter]["delta"]["ficr"]
        for group in GROUPS
        for quarter in ("Q3", "Q4")
    ]
    group_quarter_s = [
        comparison["groups"][group][quarter]["delta"]["score"]
        for group in GROUPS
        for quarter in ("Q3", "Q4")
    ]
    return (
        min(group_quarter_f),
        float(np.mean(group_quarter_f)),
        min(group_quarter_s),
        comparison["mixed"]["H2"]["delta"]["score"],
        -abs(alpha),
    )


def run_stage1(args: argparse.Namespace) -> int:
    if sha256_file(args.preregister) != PREREG_SHA256:
        raise AssertionError("preregistration hash mismatch")
    if sha256_file(args.census) != CENSUS_SHA256:
        raise AssertionError("duplicate census hash mismatch")
    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    shutil.copyfile(args.preregister, args.out_dir / "preregister.json")
    shutil.copyfile(args.census, args.out_dir / "duplicate_census.json")

    baseline, family_mean, source_paths = _read_stage1_arrays(args.artifact_root)
    source_before = [describe_file(path) for path in source_paths]
    candidate_records: dict[str, Any] = {}
    candidate_frames: dict[str, pd.DataFrame] = {}
    for alpha in ALPHAS:
        key = f"alpha_m{int(round(abs(alpha) * 1000)):03d}"
        frame = _candidate(baseline, family_mean, alpha)
        path = args.out_dir / "stage1" / f"candidate_{key}_2023_h2.parquet"
        _write_parquet(frame, path)
        candidate_frames[key] = frame
        candidate_records[key] = {
            "alpha": alpha,
            "path": str(path),
            "sha256": sha256_file(path),
            "frame_sha256": _frame_digest(frame),
        }
    baseline_path = args.out_dir / "stage1" / "baseline_locked_v3_2023_h2.parquet"
    mean_path = args.out_dir / "stage1" / "uniform_family_mean_2023_h2.parquet"
    _write_parquet(baseline, baseline_path)
    _write_parquet(family_mean, mean_path)
    source_after = [describe_file(path) for path in source_paths]
    if source_before != source_after:
        raise AssertionError("source file changed during candidate construction")
    candidate_lock = {
        "schema_version": 1,
        "preregister_sha256": PREREG_SHA256,
        "2024_read": False,
        "2025_read": False,
        "label_values_read_before_lock": False,
        "source_snapshot": source_before,
        "baseline": {"path": str(baseline_path), "sha256": sha256_file(baseline_path), "frame_sha256": _frame_digest(baseline)},
        "uniform_family_mean": {"path": str(mean_path), "sha256": sha256_file(mean_path), "frame_sha256": _frame_digest(family_mean)},
        "candidates": candidate_records,
    }
    lock_path = args.out_dir / "candidate_lock_before_2023_labels.json"
    _write_json(lock_path, candidate_lock)

    labels_full = pd.read_csv(
        args.raw_dir / "train/train_labels.csv",
        nrows=17_520,
        usecols=["kst_dtm", *GROUPS],
        parse_dates=["kst_dtm"],
        encoding="utf-8-sig",
    ).set_index("kst_dtm")
    if labels_full.index.max() != pd.Timestamp("2024-01-01 00:00:00"):
        raise AssertionError("pre-2024 label prefix boundary mismatch")
    answer = labels_full.loc[H2_INDEX, list(GROUPS)]

    scored: dict[str, Any] = {}
    eligible: list[tuple[tuple[float, ...], str]] = []
    for key, frame in candidate_frames.items():
        comparison = _compare(answer, baseline, frame)
        passed, failures = _eligible(comparison)
        objective = _objective(comparison, float(candidate_records[key]["alpha"]))
        scored[key] = {
            **candidate_records[key],
            "comparison": comparison,
            "eligible": passed,
            "eligibility_failures": failures,
            "objective_tuple": list(objective),
        }
        if passed:
            eligible.append((objective, key))
    selected_key = max(eligible, key=lambda item: item[0])[1] if eligible else None
    selected_alpha = float(candidate_records[selected_key]["alpha"]) if selected_key else None
    results = {
        "schema_version": 1,
        "experiment_id": "crossyear_dro_antiensemble_h2_v1",
        "preregister_sha256": PREREG_SHA256,
        "candidate_lock": {"path": str(lock_path), "sha256": sha256_file(lock_path)},
        "stage1_label_access": {
            "rows_parsed": 17_520,
            "end": labels_full.index.max().isoformat(),
            "2024_rows_read": 0,
            "2025_rows_read": 0,
        },
        "candidates": scored,
        "selected_key": selected_key,
        "selected_alpha": selected_alpha,
        "passed": selected_key is not None,
        "no_postscore_rescue": True,
    }
    results_path = args.out_dir / "stage1_results.json"
    _write_json(results_path, results)
    stage1_lock = {
        "schema_version": 1,
        "preregister_sha256": PREREG_SHA256,
        "candidate_lock_sha256": sha256_file(lock_path),
        "stage1_results_sha256": sha256_file(results_path),
        "selected_key": selected_key,
        "selected_alpha": selected_alpha,
        "identity_locked": selected_key is None,
        "2024_read_before_lock": False,
        "2025_read_before_lock": False,
        "stage2_formula_frozen": True,
        "no_2024_reselection": True,
    }
    _write_json(args.out_dir / "stage1_lock.json", stage1_lock)
    print(json.dumps({"passed": selected_key is not None, "selected_key": selected_key, "selected_alpha": selected_alpha}, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("stage1",), default="stage1")
    parser.add_argument("--raw-dir", type=Path, default=Path(r"data/local/open"))
    parser.add_argument("--artifact-root", type=Path, default=PROJECT / "artifacts")
    parser.add_argument("--preregister", type=Path, default=PROJECT / "configs/crossyear_dro_antiensemble_h2_preregister_v1.json")
    parser.add_argument("--census", type=Path, default=PROJECT / "artifacts/audits/crossyear_dro_antiensemble_h2_duplicate_census_v1.json")
    parser.add_argument("--out-dir", type=Path, default=PROJECT / "artifacts/postgate/crossyear_dro_antiensemble_h2_v1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_stage1(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the preregistered strict-forward residual stacking audit.

Stage 1 reads only pre-2024 labels and OOF predictions.  Its immutable lock is
verified before Stage 2 can touch a 2024 label/prediction.  A submission is
written only if the group, aggregate, and final-refit stability gates all pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.manifest import (  # noqa: E402
    describe_file,
    make_manifest,
    sha256_file,
    write_json_atomic,
)
from src.metric import CAPACITY_KWH, TARGET_COLS, score_details  # noqa: E402
from src.rolling_stacking import (  # noqa: E402
    DEFAULT_COMPONENTS,
    SimplexHuberParameter,
    apply_simplex_huber,
    compare_segments,
    fit_simplex_huber,
    group_score_triplet,
    passes_registered_gate,
    residual_blend,
)


V1_PREREGISTER_SHA256 = (
    "54e4b43c7a4b88216e25ff240ccc59feaa278eefb87362af7ebf69a2f302054f"
)
V2_PREREGISTER_SHA256 = (
    "aa6f2378cf9c9e6cb358478e794b5760cbc8fcedcaf3422e0acd16195cb5cffa"
)
GROUPS = tuple(TARGET_COLS)
COMPONENTS = tuple(DEFAULT_COMPONENTS)
REGULARIZATION_GRID = (0.0, 0.0001, 0.001, 0.01)
BLEND_WEIGHTS = (0.05, 0.10, 0.20, 0.35)
YEAR_2023_START = pd.Timestamp("2023-01-01 01:00:00")
YEAR_2023_END = pd.Timestamp("2024-01-01 00:00:00")
YEAR_2024_START = pd.Timestamp("2024-01-01 01:00:00")
YEAR_2024_END = pd.Timestamp("2025-01-01 00:00:00")
YEAR_2025_START = pd.Timestamp("2025-01-01 01:00:00")
YEAR_2025_END = pd.Timestamp("2026-01-01 00:00:00")
EXPECTED_PRE2024_LABEL_ROWS = 17_520
EXPECTED_THROUGH2024_LABEL_ROWS = 26_304
EXPECTED_2025_ROWS = 8_760

DEV_COMPONENT_FILES = {
    "lgb_l1": "oof/dev2023_lgb_l1_eligible_n1500.parquet",
    "lgb_q07": "oof/dev2023_lgb_q07_eligible.parquet",
    "shared_l1": "oof/dev2023_shared_l1_eligible.parquet",
    "shared_q07": "oof/dev2023_shared_q07_eligible.parquet",
    "top200_q07": "oof/dev2023_lgb_top200_q07_eligible.parquet",
    "energy_q06": "oof/dev2023_lgb_q06_energywt_eligible.parquet",
}
G3_DEV_FILE = "oof/g3dev2023h2_candidates.parquet"
G3_COLUMN_MAP = {
    "lgb_l1": "l1",
    "lgb_q07": "q07",
    "shared_l1": "shared_l1",
    "shared_q07": "shared_q07",
    "top200_q07": "top200q07",
    "energy_q06": "ewq06",
}
GATE_COMPONENT_FILES = {
    "lgb_l1": "gate/v3/predictions/lgb_l1_gate.parquet",
    "lgb_q07": "gate/v3/predictions/lgb_q07_gate.parquet",
    "shared_l1": "oof/gate2024_shared_l1_cf.parquet",
    "shared_q07": "oof/gate2024_shared_q07_cf.parquet",
    "top200_q07": "gate/v3/predictions/top200_q07_gate.parquet",
    "energy_q06": "gate/v3/predictions/energy_q06_gate.parquet",
}
FINAL_COMPONENT_FILES = {
    "lgb_l1": "final_v3/predictions/v3_locked_full_2025__lgb_l1_test.parquet",
    "lgb_q07": "final_v3/predictions/v3_locked_full_2025__lgb_q07_test.parquet",
    "shared_l1": "final_cf_fix/predictions/shared_l1_cf_seed42_test.parquet",
    "shared_q07": "final_cf_fix/predictions/shared_q07_cf_seed42_test.parquet",
    "top200_q07": "final_v3/predictions/v3_locked_full_2025__top200_q07_test.parquet",
    "energy_q06": "final_v3/predictions/v3_locked_full_2025__energy_q06_test.parquet",
}
STAGE1_BASELINE_G12 = "oof/dev2023_locked_v3.parquet"
STAGE2_BASELINE = "oof/gate2024_locked_v3_cf_fix.parquet"
FINAL_BASELINE_PREDICTION = "final_cf_fix/predictions/corrected_v3_test.parquet"
FINAL_BASELINE_SUBMISSION = "final_cf_fix/corrected_v3.csv"

STAGE1_SPLITS = {
    "kpx_group_1": {
        "inner_fit": ("2023-01-01 01:00:00", "2023-04-01 00:00:00"),
        "inner_valid": ("2023-04-01 01:00:00", "2023-07-01 00:00:00"),
        "outer_fit": ("2023-01-01 01:00:00", "2023-07-01 00:00:00"),
        "outer_valid": ("2023-07-01 01:00:00", "2024-01-01 00:00:00"),
        "segments": {
            "H2": ("2023-07-01 01:00:00", "2024-01-01 00:00:00"),
            "Q3": ("2023-07-01 01:00:00", "2023-10-01 00:00:00"),
            "Q4": ("2023-10-01 01:00:00", "2024-01-01 00:00:00"),
        },
    },
    "kpx_group_2": {
        "inner_fit": ("2023-01-01 01:00:00", "2023-04-01 00:00:00"),
        "inner_valid": ("2023-04-01 01:00:00", "2023-07-01 00:00:00"),
        "outer_fit": ("2023-01-01 01:00:00", "2023-07-01 00:00:00"),
        "outer_valid": ("2023-07-01 01:00:00", "2024-01-01 00:00:00"),
        "segments": {
            "H2": ("2023-07-01 01:00:00", "2024-01-01 00:00:00"),
            "Q3": ("2023-07-01 01:00:00", "2023-10-01 00:00:00"),
            "Q4": ("2023-10-01 01:00:00", "2024-01-01 00:00:00"),
        },
    },
    "kpx_group_3": {
        "inner_fit": ("2023-07-01 01:00:00", "2023-09-01 00:00:00"),
        "inner_valid": ("2023-09-01 01:00:00", "2023-10-01 00:00:00"),
        "outer_fit": ("2023-07-01 01:00:00", "2023-10-01 00:00:00"),
        "outer_valid": ("2023-10-01 01:00:00", "2024-01-01 00:00:00"),
        "segments": {
            "Q4": ("2023-10-01 01:00:00", "2024-01-01 00:00:00"),
            "OctNov": ("2023-10-01 01:00:00", "2023-12-01 00:00:00"),
            "Dec": ("2023-12-01 01:00:00", "2024-01-01 00:00:00"),
        },
    },
}
STAGE2_SEGMENTS = {
    "full": ("2024-01-01 01:00:00", "2025-01-01 00:00:00"),
    "H1": ("2024-01-01 01:00:00", "2024-07-01 00:00:00"),
    "H2": ("2024-07-01 01:00:00", "2025-01-01 00:00:00"),
    "Q1": ("2024-01-01 01:00:00", "2024-04-01 00:00:00"),
    "Q2": ("2024-04-01 01:00:00", "2024-07-01 00:00:00"),
    "Q3": ("2024-07-01 01:00:00", "2024-10-01 00:00:00"),
    "Q4": ("2024-10-01 01:00:00", "2025-01-01 00:00:00"),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=PROJECT_DIR / "artifacts")
    parser.add_argument(
        "--recipe", type=Path, default=PROJECT_DIR / "configs/train_final.v3.locked.json"
    )
    parser.add_argument(
        "--preregister-v1",
        type=Path,
        default=PROJECT_DIR / "configs/rolling_oof_stacking_preregister.json",
    )
    parser.add_argument(
        "--preregister-v2",
        type=Path,
        default=PROJECT_DIR / "configs/rolling_oof_stacking_preregister_v2.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_DIR / "artifacts/postgate/rolling_oof_stacking_v2",
    )
    parser.add_argument("--stage", choices=("stage1", "stage2", "final", "all"), default="all")
    return parser.parse_args(argv)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        _json_ready(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _frame_sha256(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    schema = [(str(column), str(dtype)) for column, dtype in frame.dtypes.items()]
    digest.update(json.dumps(schema, sort_keys=True).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    write_json_atomic(path, _json_ready(payload), overwrite=False)


def _write_parquet(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        frame.to_parquet(temporary, engine="pyarrow", compression="zstd", index=True)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        frame.to_csv(
            temporary,
            index=False,
            encoding="utf-8-sig",
            float_format="%.6f",
            lineterminator="\n",
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _closed_interval(index: pd.DatetimeIndex, bounds: Sequence[str]) -> np.ndarray:
    start, end = map(pd.Timestamp, bounds)
    result = np.asarray((index >= start) & (index <= end))
    if not result.any():
        raise ValueError(f"empty interval {start} .. {end}")
    return result


def _assert_index(
    frame: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    rows: int,
    context: str,
) -> None:
    index = pd.DatetimeIndex(frame.index)
    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError(f"{context} index must be unique and sorted")
    if len(frame) != rows or index.min() != start or index.max() != end:
        raise ValueError(
            f"{context} coverage mismatch: rows={len(frame)}, "
            f"range={index.min()}..{index.max()}"
        )


def _read_labels(raw_dir: Path, *, through_2024: bool) -> pd.DataFrame:
    rows = EXPECTED_THROUGH2024_LABEL_ROWS if through_2024 else EXPECTED_PRE2024_LABEL_ROWS
    path = raw_dir / "train/train_labels.csv"
    frame = pd.read_csv(
        path,
        nrows=rows,
        usecols=["kst_dtm", *GROUPS],
        encoding="utf-8-sig",
        parse_dates=["kst_dtm"],
    ).set_index("kst_dtm")
    end = YEAR_2024_END if through_2024 else YEAR_2023_END
    _assert_index(
        frame,
        start=pd.Timestamp("2022-01-01 01:00:00"),
        end=end,
        rows=rows,
        context="labels",
    )
    return frame


def _read_prediction(path: Path, *, start: pd.Timestamp, end: pd.Timestamp, rows: int) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame.index = pd.DatetimeIndex(frame.index)
    _assert_index(frame, start=start, end=end, rows=rows, context=str(path))
    if not np.isfinite(frame.to_numpy(dtype="float64")).all():
        raise ValueError(f"{path} contains non-finite predictions")
    return frame


def _read_dev_components(artifact_root: Path, group: str) -> dict[str, pd.Series]:
    if group in {"kpx_group_1", "kpx_group_2"}:
        output: dict[str, pd.Series] = {}
        reference: pd.DatetimeIndex | None = None
        for component, relative in DEV_COMPONENT_FILES.items():
            frame = _read_prediction(
                artifact_root / relative,
                start=YEAR_2023_START,
                end=YEAR_2023_END,
                rows=8_760,
            )
            if group not in frame:
                raise ValueError(f"{relative} is missing {group}")
            if reference is None:
                reference = frame.index
            elif not reference.equals(frame.index):
                raise ValueError("development components are not timestamp-aligned")
            output[component] = frame[group]
        return output
    raw = _read_prediction(
        artifact_root / G3_DEV_FILE,
        start=pd.Timestamp("2023-07-01 01:00:00"),
        end=YEAR_2023_END,
        rows=4_416,
    )
    missing = set(G3_COLUMN_MAP.values()).difference(raw.columns)
    if missing:
        raise ValueError(f"group-3 development OOF is missing {sorted(missing)}")
    return {component: raw[column] for component, column in G3_COLUMN_MAP.items()}


def _read_component_set(
    artifact_root: Path,
    mapping: Mapping[str, str],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    rows: int,
) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    reference: pd.DatetimeIndex | None = None
    for component in COMPONENTS:
        frame = _read_prediction(
            artifact_root / mapping[component], start=start, end=end, rows=rows
        )
        if tuple(frame.columns) != GROUPS:
            raise ValueError(f"{mapping[component]} target columns/order differ")
        if reference is None:
            reference = frame.index
        elif not reference.equals(frame.index):
            raise ValueError("component set is not timestamp-aligned")
        result[component] = frame
    return result


def _matrix_cf(components: Mapping[str, pd.Series], group: str) -> np.ndarray:
    if tuple(components) != COMPONENTS:
        raise ValueError("component order differs from preregistration")
    capacity = CAPACITY_KWH[group]
    return np.column_stack(
        [components[name].to_numpy(dtype="float64") / capacity for name in COMPONENTS]
    )


def _assemble_locked_group3(
    recipe: Mapping[str, Any], components: Mapping[str, pd.Series]
) -> np.ndarray:
    group = "kpx_group_3"
    capacity = CAPACITY_KWH[group]
    weights = recipe["ensemble"]["weights"][group]
    values = sum(
        float(weights[name]) * components[name].to_numpy(dtype="float64")
        for name in COMPONENTS
    )
    affine = recipe["ensemble"]["affine"][group]
    values = float(affine["scale"]) * values + float(affine["bias_kwh"])
    return np.clip(values, 0.0, 1.02 * capacity)


def _select_lambda(
    matrix: np.ndarray,
    actual_cf: np.ndarray,
    index: pd.DatetimeIndex,
    fit_mask: np.ndarray,
    valid_mask: np.ndarray,
    *,
    group: str,
) -> tuple[float, list[dict[str, Any]]]:
    diagnostics: list[dict[str, Any]] = []
    for regularization in REGULARIZATION_GRID:
        parameter = fit_simplex_huber(
            matrix[fit_mask],
            actual_cf[fit_mask],
            index[fit_mask],
            regularization=regularization,
        )
        prediction_cf = apply_simplex_huber(
            parameter, matrix[valid_mask], index[valid_mask]
        )
        metrics = group_score_triplet(
            actual_cf[valid_mask], prediction_cf, 1.0, group_name=group
        )
        diagnostics.append(
            {
                "regularization": regularization,
                "parameter": parameter.as_dict(),
                "inner_validation": metrics,
            }
        )
    selected = max(
        diagnostics,
        key=lambda item: (
            float(item["inner_validation"]["score"]),
            float(item["regularization"]),
        ),
    )
    return float(selected["regularization"]), diagnostics


def _stage1_input_paths(
    artifact_root: Path, recipe_path: Path, preregister_v1: Path, preregister_v2: Path
) -> list[Path]:
    paths = [recipe_path, preregister_v1, preregister_v2]
    paths.extend(artifact_root / relative for relative in DEV_COMPONENT_FILES.values())
    paths.extend([artifact_root / G3_DEV_FILE, artifact_root / STAGE1_BASELINE_G12])
    return paths


def _stage1(
    *,
    raw_dir: Path,
    artifact_root: Path,
    recipe_path: Path,
    preregister_v1: Path,
    preregister_v2: Path,
    out_dir: Path,
) -> dict[str, Any]:
    if sha256_file(preregister_v1) != V1_PREREGISTER_SHA256:
        raise AssertionError("v1 preregistration hash mismatch")
    if sha256_file(preregister_v2) != V2_PREREGISTER_SHA256:
        raise AssertionError("v2 preregistration hash mismatch")
    out_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(preregister_v1, out_dir / "preregister_v1_incompatible.json")
    shutil.copyfile(preregister_v2, out_dir / "preregister.json")
    input_paths = _stage1_input_paths(
        artifact_root, recipe_path, preregister_v1, preregister_v2
    )
    input_before = [describe_file(path) for path in input_paths]
    labels = _read_labels(raw_dir, through_2024=False)
    label_evidence = {
        "path": str((raw_dir / "train/train_labels.csv").resolve()),
        "loaded_rows": len(labels),
        "loaded_start": labels.index.min().isoformat(),
        "loaded_end": labels.index.max().isoformat(),
        "loaded_frame_sha256": _frame_sha256(labels),
        "suffix_rows_read": 0,
        "whole_file_hash_deliberately_omitted": True,
    }
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    baseline_g12 = _read_prediction(
        artifact_root / STAGE1_BASELINE_G12,
        start=YEAR_2023_START,
        end=YEAR_2023_END,
        rows=8_760,
    )

    group_results: dict[str, Any] = {}
    locked_groups: list[str] = []
    output_paths: list[Path] = []
    for group in GROUPS:
        components = _read_dev_components(artifact_root, group)
        index = pd.DatetimeIndex(next(iter(components.values())).index)
        matrix = _matrix_cf(components, group)
        capacity = CAPACITY_KWH[group]
        actual_cf = labels.loc[index, group].to_numpy(dtype="float64") / capacity
        if group == "kpx_group_3":
            baseline_cf = _assemble_locked_group3(recipe, components) / capacity
        else:
            baseline_cf = baseline_g12.loc[index, group].to_numpy(dtype="float64") / capacity
        split = STAGE1_SPLITS[group]
        inner_fit = _closed_interval(index, split["inner_fit"])
        inner_valid = _closed_interval(index, split["inner_valid"])
        outer_fit = _closed_interval(index, split["outer_fit"])
        outer_valid = _closed_interval(index, split["outer_valid"])
        if np.any(inner_fit & inner_valid) or np.any(outer_fit & outer_valid):
            raise AssertionError("fit and validation periods overlap")
        if index[inner_fit].max() >= index[inner_valid].min():
            raise AssertionError("inner split is not strict-forward")
        if index[outer_fit].max() >= index[outer_valid].min():
            raise AssertionError("outer split is not strict-forward")
        regularization, lambda_diagnostics = _select_lambda(
            matrix,
            actual_cf,
            index,
            inner_fit,
            inner_valid,
            group=group,
        )
        parameter = fit_simplex_huber(
            matrix[outer_fit],
            actual_cf[outer_fit],
            index[outer_fit],
            regularization=regularization,
        )
        valid_index = index[outer_valid]
        direct_cf = apply_simplex_huber(
            parameter, matrix[outer_valid], valid_index
        )
        direct_path = out_dir / "oof" / f"stage1__{group}__direct_stack.parquet"
        _write_parquet(
            pd.DataFrame({group: direct_cf * capacity}, index=valid_index), direct_path
        )
        output_paths.append(direct_path)
        recipes: dict[str, Any] = {}
        eligible: list[tuple[float, float, float]] = []
        for blend_weight in BLEND_WEIGHTS:
            candidate_cf = residual_blend(
                baseline_cf[outer_valid], direct_cf, blend_weight
            )
            comparisons = compare_segments(
                valid_index,
                actual_cf[outer_valid] * capacity,
                baseline_cf[outer_valid] * capacity,
                candidate_cf * capacity,
                capacity,
                split["segments"],
                group_name=group,
            )
            full_segment_name = "H2" if group != "kpx_group_3" else "Q4"
            passed = passes_registered_gate(
                comparisons, full_segment_name=full_segment_name
            )
            minimum = min(
                float(values["delta"]["score"]) for values in comparisons.values()
            )
            mean = float(
                np.mean(
                    [float(values["delta"]["score"]) for values in comparisons.values()]
                )
            )
            key = f"w{int(round(100 * blend_weight)):02d}"
            candidate_path = out_dir / "oof" / f"stage1__{group}__{key}.parquet"
            _write_parquet(
                pd.DataFrame({group: candidate_cf * capacity}, index=valid_index),
                candidate_path,
            )
            output_paths.append(candidate_path)
            recipes[key] = {
                "blend_weight": blend_weight,
                "comparisons": comparisons,
                "passed": passed,
                "minimum_segment_score_delta": minimum,
                "mean_segment_score_delta": mean,
                "prediction_sha256": sha256_file(candidate_path),
            }
            if passed:
                eligible.append((minimum, mean, blend_weight))
        selected_weight: float | None = None
        if eligible:
            selected_weight = max(eligible, key=lambda item: (item[0], item[1], -item[2]))[2]
            locked_groups.append(group)
        group_results[group] = {
            "selected_regularization": regularization,
            "inner_lambda_diagnostics": lambda_diagnostics,
            "outer_refit_parameter": parameter.as_dict(),
            "outer_validation_start": valid_index.min().isoformat(),
            "outer_validation_end": valid_index.max().isoformat(),
            "outer_validation_rows": len(valid_index),
            "direct_prediction_sha256": sha256_file(direct_path),
            "blend_recipes": recipes,
            "selected_blend_weight": selected_weight,
            "locked": selected_weight is not None,
        }

    input_after = [describe_file(path) for path in input_paths]
    if input_before != input_after:
        raise AssertionError("a Stage1 source artifact changed during execution")
    results: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": "rolling_oof_low_dof_residual_blend_v2",
        "v1_preregister_sha256": V1_PREREGISTER_SHA256,
        "v2_preregister_sha256": V2_PREREGISTER_SHA256,
        "public_metrics_read": False,
        "2024_read": False,
        "stage1_label_evidence": label_evidence,
        "stage1_input_snapshot_before": input_before,
        "stage1_input_snapshot_after": input_after,
        "stage1_input_snapshot_sha256": _canonical_sha256(input_before),
        "groups": group_results,
        "locked_groups": locked_groups,
        "selection_rule": "all registered outer segments and full component metrics must pass",
    }
    results_path = out_dir / "stage1_results.json"
    _write_json(results_path, results)
    lock = {
        "schema_version": 1,
        "v1_preregister_sha256": V1_PREREGISTER_SHA256,
        "v2_preregister_sha256": V2_PREREGISTER_SHA256,
        "stage1_results_sha256": sha256_file(results_path),
        "stage1_input_snapshot_sha256": results["stage1_input_snapshot_sha256"],
        "locked_groups": locked_groups,
        "locked_parameters": {
            group: {
                "regularization": group_results[group]["selected_regularization"],
                "parameter": group_results[group]["outer_refit_parameter"],
                "blend_weight": group_results[group]["selected_blend_weight"],
            }
            for group in locked_groups
        },
        "2024_read_before_lock": False,
        "stage2_candidate_frozen": True,
        "no_2024_reselection": True,
    }
    _write_json(out_dir / "stage1_lock.json", lock)
    return {"results": results, "lock": lock, "outputs": output_paths}


def _parameter_from_dict(value: Mapping[str, Any]) -> SimplexHuberParameter:
    return SimplexHuberParameter(
        component_names=tuple(value["component_names"]),
        weights=tuple(float(item) for item in value["weights"]),
        intercept_cf=float(value["intercept_cf"]),
        regularization=float(value["regularization"]),
        fit_start=str(value["fit_start"]),
        fit_end=str(value["fit_end"]),
        eligible_rows=int(value["eligible_rows"]),
        optimizer_objective=float(value["optimizer_objective"]),
    )


def _load_stage1_lock(out_dir: Path) -> dict[str, Any]:
    results_path = out_dir / "stage1_results.json"
    lock_path = out_dir / "stage1_lock.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["v2_preregister_sha256"] != V2_PREREGISTER_SHA256:
        raise AssertionError("locked v2 preregistration hash mismatch")
    if lock["stage1_results_sha256"] != sha256_file(results_path):
        raise AssertionError("Stage1 result was modified after locking")
    expected_groups = [
        group for group in GROUPS if bool(results["groups"][group]["locked"])
    ]
    if lock["locked_groups"] != expected_groups:
        raise AssertionError("locked_groups differs from Stage1 results")
    for group in expected_groups:
        result = results["groups"][group]
        parameter = lock["locked_parameters"][group]
        if parameter["blend_weight"] != result["selected_blend_weight"]:
            raise AssertionError("locked blend weight differs from Stage1 selection")
        if parameter["parameter"] != result["outer_refit_parameter"]:
            raise AssertionError("locked stack parameter differs from Stage1 result")
    return lock


def _aggregate_comparisons(
    labels: pd.DataFrame,
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    segments: Mapping[str, tuple[str, str]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, bounds in segments.items():
        selected = _closed_interval(pd.DatetimeIndex(labels.index), bounds)
        base = score_details(labels.loc[selected, list(GROUPS)], baseline.loc[selected, list(GROUPS)]).as_dict()
        cand = score_details(labels.loc[selected, list(GROUPS)], candidate.loc[selected, list(GROUPS)]).as_dict()
        output[name] = {
            "baseline": {
                key: base[key] for key in ("total_score", "one_minus_nmae", "ficr")
            },
            "candidate": {
                key: cand[key] for key in ("total_score", "one_minus_nmae", "ficr")
            },
            "delta": {
                key: float(cand[key] - base[key])
                for key in ("total_score", "one_minus_nmae", "ficr")
            },
        }
    return output


def _aggregate_gate_passes(comparisons: Mapping[str, Any]) -> bool:
    full = comparisons["full"]["delta"]
    return (
        float(full["total_score"]) > 0.0
        and float(full["one_minus_nmae"]) >= 0.0
        and float(full["ficr"]) >= 0.0
        and (
            float(full["one_minus_nmae"]) > 0.0 or float(full["ficr"]) > 0.0
        )
        and float(comparisons["H1"]["delta"]["total_score"]) > 0.0
        and float(comparisons["H2"]["delta"]["total_score"]) > 0.0
    )


def _stage2(
    *, raw_dir: Path, artifact_root: Path, out_dir: Path
) -> dict[str, Any]:
    lock = _load_stage1_lock(out_dir)
    locked_groups = list(lock["locked_groups"])
    if not locked_groups:
        results = {
            "schema_version": 1,
            "2024_read": False,
            "reason": "No group passed the pre-2024 Stage1 gate.",
            "locked_groups": [],
            "promoted_groups": [],
            "aggregate_passed": False,
        }
        results_path = out_dir / "stage2_results.json"
        _write_json(results_path, results)
        promotion = {
            "schema_version": 1,
            "stage1_lock_sha256": sha256_file(out_dir / "stage1_lock.json"),
            "stage2_results_sha256": sha256_file(results_path),
            "2024_read": False,
            "promoted_groups": [],
            "candidate_promoted": False,
        }
        _write_json(out_dir / "stage2_promotion_lock.json", promotion)
        return {"results": results, "lock": promotion, "outputs": []}

    labels_all = _read_labels(raw_dir, through_2024=True)
    baseline = _read_prediction(
        artifact_root / STAGE2_BASELINE,
        start=YEAR_2024_START,
        end=YEAR_2024_END,
        rows=8_784,
    ).loc[:, list(GROUPS)]
    components = _read_component_set(
        artifact_root,
        GATE_COMPONENT_FILES,
        start=YEAR_2024_START,
        end=YEAR_2024_END,
        rows=8_784,
    )
    index = pd.DatetimeIndex(baseline.index)
    labels = labels_all.loc[index, list(GROUPS)]
    mixed = baseline.copy()
    group_results: dict[str, Any] = {}
    promoted_groups: list[str] = []
    output_paths: list[Path] = []
    for group in locked_groups:
        capacity = CAPACITY_KWH[group]
        matrix = _matrix_cf({name: components[name][group] for name in COMPONENTS}, group)
        locked = lock["locked_parameters"][group]
        parameter = _parameter_from_dict(locked["parameter"])
        direct_cf = apply_simplex_huber(parameter, matrix, index)
        candidate_cf = residual_blend(
            baseline[group].to_numpy(dtype="float64") / capacity,
            direct_cf,
            float(locked["blend_weight"]),
        )
        comparisons = compare_segments(
            index,
            labels[group].to_numpy(dtype="float64"),
            baseline[group].to_numpy(dtype="float64"),
            candidate_cf * capacity,
            capacity,
            STAGE2_SEGMENTS,
            group_name=group,
        )
        passed = passes_registered_gate(comparisons, full_segment_name="full")
        candidate_path = out_dir / "oof" / f"stage2__{group}__fixed_candidate.parquet"
        _write_parquet(pd.DataFrame({group: candidate_cf * capacity}, index=index), candidate_path)
        output_paths.append(candidate_path)
        group_results[group] = {
            "comparisons": comparisons,
            "passed": passed,
            "prediction_sha256": sha256_file(candidate_path),
        }
        if passed:
            promoted_groups.append(group)
            mixed[group] = candidate_cf * capacity
    aggregate = _aggregate_comparisons(
        labels,
        baseline,
        mixed,
        {key: STAGE2_SEGMENTS[key] for key in ("full", "H1", "H2")},
    )
    aggregate_passed = bool(promoted_groups) and _aggregate_gate_passes(aggregate)
    if not aggregate_passed:
        promoted_groups = []
    results = {
        "schema_version": 1,
        "2024_read": True,
        "reference_baseline": STAGE2_BASELINE,
        "original_target_scale_mismatch_baseline_used": False,
        "locked_groups": locked_groups,
        "groups": group_results,
        "aggregate": aggregate,
        "aggregate_passed": aggregate_passed,
        "promoted_groups": promoted_groups,
        "no_2024_reselection": True,
    }
    results_path = out_dir / "stage2_results.json"
    _write_json(results_path, results)
    promotion = {
        "schema_version": 1,
        "stage1_lock_sha256": sha256_file(out_dir / "stage1_lock.json"),
        "stage2_results_sha256": sha256_file(results_path),
        "2024_read": True,
        "promoted_groups": promoted_groups,
        "candidate_promoted": bool(promoted_groups) and aggregate_passed,
        "no_reselection": True,
    }
    _write_json(out_dir / "stage2_promotion_lock.json", promotion)
    return {"results": results, "lock": promotion, "outputs": output_paths}


def _load_promotion_lock(out_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    stage1 = _load_stage1_lock(out_dir)
    results_path = out_dir / "stage2_results.json"
    promotion_path = out_dir / "stage2_promotion_lock.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    if promotion["stage1_lock_sha256"] != sha256_file(out_dir / "stage1_lock.json"):
        raise AssertionError("Stage1 lock changed before final stage")
    if promotion["stage2_results_sha256"] != sha256_file(results_path):
        raise AssertionError("Stage2 results changed after promotion lock")
    if promotion["promoted_groups"] != results["promoted_groups"]:
        raise AssertionError("promoted_groups differs from Stage2 results")
    if any(group not in stage1["locked_groups"] for group in promotion["promoted_groups"]):
        raise AssertionError("Stage2 promoted a group that Stage1 did not lock")
    return stage1, promotion


def _equal_period_weights(
    actual_cf: np.ndarray, period_ids: np.ndarray
) -> np.ndarray:
    result = np.zeros(len(actual_cf), dtype="float64")
    eligible = np.isfinite(actual_cf) & (actual_cf >= 0.10)
    unique = np.unique(period_ids)
    for period in unique:
        selected = eligible & (period_ids == period)
        count = int(selected.sum())
        if count == 0:
            raise ValueError(f"period {period!r} has no eligible rows")
        result[period_ids == period] = 1.0 / count
    return result


def _final(
    *, raw_dir: Path, artifact_root: Path, out_dir: Path
) -> dict[str, Any]:
    stage1, promotion = _load_promotion_lock(out_dir)
    promoted_groups = list(promotion["promoted_groups"])
    if not promotion["candidate_promoted"] or not promoted_groups:
        results = {
            "schema_version": 1,
            "candidate_created": False,
            "2025_predictions_read": False,
            "reason": "No candidate passed the complete Stage1 and Stage2 rules.",
            "promoted_groups": [],
            "submission": None,
        }
        _write_json(out_dir / "final_results.json", results)
        return {"results": results, "outputs": []}

    labels = _read_labels(raw_dir, through_2024=True)
    gate_components = _read_component_set(
        artifact_root,
        GATE_COMPONENT_FILES,
        start=YEAR_2024_START,
        end=YEAR_2024_END,
        rows=8_784,
    )
    final_components = _read_component_set(
        artifact_root,
        FINAL_COMPONENT_FILES,
        start=YEAR_2025_START,
        end=YEAR_2025_END,
        rows=EXPECTED_2025_ROWS,
    )
    baseline_final = _read_prediction(
        artifact_root / FINAL_BASELINE_PREDICTION,
        start=YEAR_2025_START,
        end=YEAR_2025_END,
        rows=EXPECTED_2025_ROWS,
    ).loc[:, list(GROUPS)]
    final_index = pd.DatetimeIndex(baseline_final.index)
    final_prediction = baseline_final.copy()
    refit_results: dict[str, Any] = {}
    stability_passed = True
    for group in promoted_groups:
        dev_components = _read_dev_components(artifact_root, group)
        dev_index = pd.DatetimeIndex(next(iter(dev_components.values())).index)
        dev_matrix = _matrix_cf(dev_components, group)
        gate_matrix = _matrix_cf(
            {name: gate_components[name][group] for name in COMPONENTS}, group
        )
        matrix = np.row_stack([dev_matrix, gate_matrix])
        combined_index = dev_index.append(pd.DatetimeIndex(next(iter(gate_components.values())).index))
        capacity = CAPACITY_KWH[group]
        actual_cf = labels.loc[combined_index, group].to_numpy(dtype="float64") / capacity
        period_ids = np.concatenate(
            [np.zeros(len(dev_index), dtype=int), np.ones(len(gate_matrix), dtype=int)]
        )
        sample_weight = _equal_period_weights(actual_cf, period_ids)
        locked = stage1["locked_parameters"][group]
        refit = fit_simplex_huber(
            matrix,
            actual_cf,
            combined_index,
            regularization=float(locked["regularization"]),
            sample_weight=sample_weight,
        )
        original = _parameter_from_dict(locked["parameter"])
        weight_drift = float(
            np.sum(np.abs(np.asarray(refit.weights) - np.asarray(original.weights)))
        )
        intercept_drift = abs(refit.intercept_cf - original.intercept_cf)
        group_stable = weight_drift <= 0.35 and intercept_drift <= 0.02
        stability_passed = stability_passed and group_stable
        refit_results[group] = {
            "parameter": refit.as_dict(),
            "stage1_weight_l1_drift": weight_drift,
            "stage1_intercept_absolute_drift_cf": intercept_drift,
            "stability_passed": group_stable,
        }
        if group_stable:
            final_matrix = _matrix_cf(
                {name: final_components[name][group] for name in COMPONENTS}, group
            )
            direct_cf = apply_simplex_huber(refit, final_matrix, final_index)
            candidate_cf = residual_blend(
                baseline_final[group].to_numpy(dtype="float64") / capacity,
                direct_cf,
                float(locked["blend_weight"]),
            )
            final_prediction[group] = candidate_cf * capacity
    if not stability_passed:
        results = {
            "schema_version": 1,
            "candidate_created": False,
            "2025_predictions_read": True,
            "reason": "At least one promoted group's preregistered refit stability gate failed.",
            "promoted_groups": promoted_groups,
            "refit": refit_results,
            "submission": None,
        }
        _write_json(out_dir / "final_results.json", results)
        return {"results": results, "outputs": []}

    sample = pd.read_csv(raw_dir / "sample_submission.csv", encoding="utf-8-sig")
    if len(sample) != EXPECTED_2025_ROWS or tuple(sample.columns) != (
        "forecast_id",
        "forecast_kst_dtm",
        *GROUPS,
    ):
        raise ValueError("sample submission schema/row count mismatch")
    sample_time = pd.to_datetime(sample["forecast_kst_dtm"])
    if not pd.DatetimeIndex(sample_time).equals(final_index):
        raise ValueError("sample submission timestamps differ from final predictions")
    submission = sample.copy()
    for group in GROUPS:
        values = final_prediction[group].to_numpy(dtype="float64")
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(
            values > 1.02 * CAPACITY_KWH[group] + 1e-9
        ):
            raise ValueError(f"final {group} values violate finite/bounds contract")
        submission[group] = values
    parquet_path = out_dir / "predictions/rolling_oof_stacking_v2_2025.parquet"
    csv_path = out_dir / "rolling_oof_stacking_v2_2025.csv"
    _write_parquet(final_prediction, parquet_path)
    _write_csv(submission, csv_path)
    readback = pd.read_csv(csv_path, encoding="utf-8-sig")
    if not readback[["forecast_id", "forecast_kst_dtm"]].equals(
        sample[["forecast_id", "forecast_kst_dtm"]]
    ):
        raise AssertionError("submission identity columns changed on readback")
    rounded = np.round(final_prediction.to_numpy(dtype="float64"), 6)
    np.testing.assert_allclose(
        readback.loc[:, list(GROUPS)].to_numpy(dtype="float64"),
        rounded,
        rtol=0.0,
        atol=5e-7,
    )
    baseline_csv = pd.read_csv(
        artifact_root / FINAL_BASELINE_SUBMISSION, encoding="utf-8-sig"
    )
    for group in set(GROUPS).difference(promoted_groups):
        if not np.array_equal(readback[group].to_numpy(), baseline_csv[group].to_numpy()):
            raise AssertionError(f"identity fallback changed {group} baseline values")
    results = {
        "schema_version": 1,
        "candidate_created": True,
        "2025_predictions_read": True,
        "promoted_groups": promoted_groups,
        "identity_groups": [group for group in GROUPS if group not in promoted_groups],
        "refit": refit_results,
        "submission": str(csv_path.resolve()),
        "submission_sha256": sha256_file(csv_path),
        "prediction_sha256": sha256_file(parquet_path),
        "leaderboard_score_claim": False,
    }
    _write_json(out_dir / "final_results.json", results)
    return {"results": results, "outputs": [parquet_path, csv_path]}


def _write_manifest(
    *,
    artifact_root: Path,
    recipe_path: Path,
    preregister_v1: Path,
    preregister_v2: Path,
    out_dir: Path,
) -> None:
    if (out_dir / "manifest.json").exists():
        raise FileExistsError(f"refusing to overwrite {out_dir / 'manifest.json'}")
    final_results = json.loads((out_dir / "final_results.json").read_text(encoding="utf-8"))
    input_files = _stage1_input_paths(
        artifact_root, recipe_path, preregister_v1, preregister_v2
    )
    input_files.extend(
        [
            PROJECT_DIR / "src/rolling_stacking.py",
            PROJECT_DIR / "scripts/run_rolling_oof_stacking.py",
            PROJECT_DIR / "tests/test_rolling_stacking.py",
        ]
    )
    if json.loads((out_dir / "stage2_results.json").read_text(encoding="utf-8"))["2024_read"]:
        input_files.extend(artifact_root / relative for relative in GATE_COMPONENT_FILES.values())
        input_files.append(artifact_root / STAGE2_BASELINE)
    if final_results["2025_predictions_read"]:
        input_files.extend(artifact_root / relative for relative in FINAL_COMPONENT_FILES.values())
        input_files.extend(
            [
                artifact_root / FINAL_BASELINE_PREDICTION,
                artifact_root / FINAL_BASELINE_SUBMISSION,
            ]
        )
    output_files = sorted(
        path
        for path in out_dir.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    manifest = make_manifest(
        artifact_type="rolling_oof_low_dof_residual_blend_v2",
        parameters={
            "regularization_grid": REGULARIZATION_GRID,
            "blend_weights": BLEND_WEIGHTS,
            "components": COMPONENTS,
            "reference_scale": "corrected_capacity_factor_shared_models",
            "v1_preregister_sha256": V1_PREREGISTER_SHA256,
            "v2_preregister_sha256": V2_PREREGISTER_SHA256,
        },
        input_files=input_files,
        output_files=output_files,
        results={
            "candidate_created": final_results["candidate_created"],
            "promoted_groups": final_results["promoted_groups"],
            "public_metrics_read": False,
            "untouched_2024_claim": False,
        },
        project_dir=PROJECT_DIR,
    )
    _write_json(out_dir / "manifest.json", manifest)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    raw_dir = args.raw_dir.expanduser().resolve()
    artifact_root = args.artifact_root.expanduser().resolve()
    recipe_path = args.recipe.expanduser().resolve()
    preregister_v1 = args.preregister_v1.expanduser().resolve()
    preregister_v2 = args.preregister_v2.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()

    if args.stage in {"stage1", "all"}:
        stage1 = _stage1(
            raw_dir=raw_dir,
            artifact_root=artifact_root,
            recipe_path=recipe_path,
            preregister_v1=preregister_v1,
            preregister_v2=preregister_v2,
            out_dir=out_dir,
        )
        print(f"Stage1 locked groups: {stage1['lock']['locked_groups']}")
    if args.stage in {"stage2", "all"}:
        stage2 = _stage2(raw_dir=raw_dir, artifact_root=artifact_root, out_dir=out_dir)
        print(
            f"Stage2 2024_read={stage2['results']['2024_read']}, "
            f"promoted={stage2['lock']['promoted_groups']}"
        )
    if args.stage in {"final", "all"}:
        final = _final(raw_dir=raw_dir, artifact_root=artifact_root, out_dir=out_dir)
        print(f"Final candidate_created={final['results']['candidate_created']}")
    if args.stage == "all":
        _write_manifest(
            artifact_root=artifact_root,
            recipe_path=recipe_path,
            preregister_v1=preregister_v1,
            preregister_v2=preregister_v2,
            out_dir=out_dir,
        )
        print(f"Manifest: {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

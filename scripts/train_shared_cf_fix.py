"""Retrain final-v3 shared models with the intended capacity-factor target.

Only ``shared_l1`` and ``shared_q07`` are refit. Existing final-v3 *group*
candidate prediction Parquets are treated as immutable inputs. Three corrected
recipe variants are written to a new output directory; no leaderboard score is
claimed by their names or manifest.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.final_training import (  # noqa: E402
    TrainedCandidate,
    _assemble_final_predictions,
    _atomic_joblib,
    _atomic_parquet,
    _atomic_submission_csv,
    _build_submission,
    _verify_written_submission,
    load_final_data,
    read_recipe,
    train_candidate,
    validate_recipe,
)
from src.manifest import make_manifest, sha256_file, write_json_atomic  # noqa: E402
from src.metric import CAPACITY_KWH, TARGET_COLS  # noqa: E402


SOURCE_PREFIX = "v3_locked_full_2025"
SHARED_CANDIDATES = ("shared_l1", "shared_q07")
GROUP_CANDIDATES = ("lgb_l1", "lgb_q07", "top200_q07", "energy_q06")
EXPECTED_TRAINING_ROWS = {
    "kpx_group_1": 15_915,
    "kpx_group_2": 15_891,
    "kpx_group_3": 9_414,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="official data root containing labels and sample_submission.csv",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="weather feature cache created by scripts/build_features.py",
    )
    parser.add_argument(
        "--final-dir",
        type=Path,
        required=True,
        help="immutable final_v3 source directory",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_DIR / "artifacts" / "final_cf_fix",
        help="new corrected artifact directory (default: artifacts/final_cf_fix)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace only files inside --out-dir; source artifacts remain immutable",
    )
    return parser.parse_args(argv)


def _output_paths(out_dir: Path) -> dict[str, Path]:
    paths = {
        "config": out_dir / "corrected_v3_formula_config.json",
        "manifest": out_dir / "shared_cf_fix_manifest.json",
    }
    for name in SHARED_CANDIDATES:
        paths[f"model:{name}"] = out_dir / "models" / f"{name}_cf_seed42.joblib"
        paths[f"prediction:{name}"] = (
            out_dir / "predictions" / f"{name}_cf_seed42_test.parquet"
        )
    for variant in ("corrected_v3", "corrected_recent_v4", "corrected_hybrid50"):
        paths[f"prediction:{variant}"] = (
            out_dir / "predictions" / f"{variant}_test.parquet"
        )
        paths[f"submission:{variant}"] = out_dir / f"{variant}.csv"
    return paths


def _preflight(
    *,
    raw_dir: Path,
    cache_dir: Path,
    final_dir: Path,
    out_dir: Path,
    paths: Mapping[str, Path],
    overwrite: bool,
) -> None:
    protected = {raw_dir.resolve(), cache_dir.resolve(), final_dir.resolve()}
    if out_dir.resolve() in protected:
        raise ValueError("--out-dir must differ from raw/cache/final source directories")
    for path in paths.values():
        resolved = path.resolve()
        if any(resolved == source or source in resolved.parents for source in protected):
            raise ValueError(f"refusing to write inside protected source directory: {path}")
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        rendered = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(
            "corrected outputs already exist; pass --overwrite to replace only "
            f"these new outputs:\n{rendered}"
        )


def _source_prediction_paths(final_dir: Path) -> dict[str, Path]:
    prediction_dir = final_dir / "predictions"
    return {
        name: prediction_dir / f"{SOURCE_PREFIX}__{name}_test.parquet"
        for name in GROUP_CANDIDATES
    }


def _read_prediction(
    path: Path, expected_index: pd.DatetimeIndex, *, context: str
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path, engine="pyarrow")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f"{context} must use a DatetimeIndex")
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if tuple(frame.columns) != TARGET_COLS:
        raise ValueError(f"{context} columns/order must be {TARGET_COLS}")
    if not frame.index.equals(expected_index):
        raise ValueError(f"{context} index/order differs from sample_submission")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{context} index must be unique and sorted")
    values = frame.to_numpy(dtype="float64", copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"{context} contains NaN or infinite predictions")
    return frame.astype("float64")


def _corrected_recipe(source_recipe: Mapping[str, Any]) -> dict[str, Any]:
    recipe = deepcopy(dict(source_recipe))
    recipe["recipe_name"] = "corrected_v3_shared_cf_seed42"
    recipe["seed"] = 42
    recipe["device"] = "cpu"
    for name in SHARED_CANDIDATES:
        specification = recipe["models"][name]
        if specification["scope"] != "shared":
            raise ValueError(f"source {name} is not a shared candidate")
        if specification["row_filter"] != "eligible":
            raise ValueError(f"source {name} must use eligible rows")
        if specification["features"] != "all":
            raise ValueError(f"source {name} must use all weather features")
        if specification["sample_weight"]["kind"] != "none":
            raise ValueError(f"source {name} must be unweighted")
        specification["target_scale"] = "capacity_factor"
    # The four group candidates are not refit in this partial correction. A
    # missing target_scale in the historical snapshot correctly documents that
    # their existing predictions came from the legacy kWh execution.
    validate_recipe(recipe)
    return recipe


def _candidate_payload(
    candidate: TrainedCandidate,
    specification: Mapping[str, Any],
    recipe_name: str,
) -> dict[str, Any]:
    return {
        "artifact_schema_version": 1,
        "recipe_name": recipe_name,
        "candidate_name": candidate.name,
        "candidate_config": dict(specification),
        "scope": candidate.scope,
        "groups": TARGET_COLS,
        "training_rows": candidate.training_rows,
        "feature_names": candidate.feature_names,
        "target_scale": "capacity_factor",
        "seed": 42,
        "models": candidate.models,
    }


def _clip(group: str, values: Any) -> np.ndarray:
    return np.clip(
        np.asarray(values, dtype="float64"),
        0.0,
        CAPACITY_KWH[group] * 1.02,
    )


def _corrected_recent_v4(
    *,
    corrected_v3: pd.DataFrame,
    group_predictions: Mapping[str, pd.DataFrame],
    shared_predictions: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    output = pd.DataFrame(index=corrected_v3.index, columns=TARGET_COLS, dtype=float)
    output["kpx_group_1"] = _clip(
        "kpx_group_1",
        1.10
        * (
            0.80 * shared_predictions["shared_q07"]["kpx_group_1"]
            + 0.05 * group_predictions["top200_q07"]["kpx_group_1"]
            + 0.15 * group_predictions["energy_q06"]["kpx_group_1"]
        )
        - 700.0,
    )
    output["kpx_group_2"] = _clip(
        "kpx_group_2",
        1.025
        * (
            0.95 * group_predictions["lgb_l1"]["kpx_group_2"]
            + 0.05 * group_predictions["energy_q06"]["kpx_group_2"]
        )
        - 100.0,
    )
    shared_branch_g3 = _clip(
        "kpx_group_3",
        1.075
        * (
            0.70 * shared_predictions["shared_q07"]["kpx_group_3"]
            + 0.10 * shared_predictions["shared_l1"]["kpx_group_3"]
            + 0.20 * group_predictions["lgb_q07"]["kpx_group_3"]
        )
        - 300.0,
    )
    output["kpx_group_3"] = _clip(
        "kpx_group_3",
        0.50 * corrected_v3["kpx_group_3"] + 0.50 * shared_branch_g3,
    )
    return output


def _corrected_hybrid50(
    corrected_v3: pd.DataFrame, corrected_recent: pd.DataFrame
) -> pd.DataFrame:
    output = pd.DataFrame(index=corrected_v3.index, columns=TARGET_COLS, dtype=float)
    for group in ("kpx_group_1", "kpx_group_2"):
        output[group] = _clip(
            group, 0.50 * corrected_v3[group] + 0.50 * corrected_recent[group]
        )
    output["kpx_group_3"] = _clip(
        "kpx_group_3", corrected_recent["kpx_group_3"]
    )
    return output


def _prediction_summary(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        group: {"min": float(frame[group].min()), "max": float(frame[group].max())}
        for group in TARGET_COLS
    }


def _verify_csv(
    path: Path, expected: pd.DataFrame, sample: pd.DataFrame
) -> dict[str, Any]:
    with path.open("rb") as stream:
        if stream.read(3) != b"\xef\xbb\xbf":
            raise AssertionError(f"{path.name} is not UTF-8-SIG")
    observed = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"forecast_id": "string", "forecast_kst_dtm": "string"},
    )
    if tuple(observed.columns) != tuple(sample.columns) or len(observed) != len(sample):
        raise AssertionError(f"{path.name} schema/row count changed")
    for column in ("forecast_id", "forecast_kst_dtm"):
        if not np.array_equal(
            observed[column].astype(str).to_numpy(),
            sample[column].astype(str).to_numpy(),
        ):
            raise AssertionError(f"{path.name} {column} values/order changed")
    observed_values = observed.loc[:, list(TARGET_COLS)].to_numpy(dtype=float)
    expected_values = expected.loc[:, list(TARGET_COLS)].to_numpy(dtype=float)
    if not np.isfinite(observed_values).all():
        raise AssertionError(f"{path.name} contains non-finite values")
    max_difference = float(np.max(np.abs(observed_values - expected_values)))
    if max_difference > 5.1e-7:
        raise AssertionError(f"{path.name} exceeds six-decimal rounding tolerance")
    for group in TARGET_COLS:
        if (observed[group] < -1e-6).any() or (
            observed[group] > CAPACITY_KWH[group] * 1.02 + 1e-6
        ).any():
            raise AssertionError(f"{path.name} violates physical clip for {group}")
    return {
        "sha256": sha256_file(path),
        "rows": int(len(observed)),
        "utf8_sig": True,
        "max_readback_abs_diff": max_difference,
    }


def _formula_config(
    corrected_recipe: Mapping[str, Any],
    source_paths: Mapping[str, Path],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "config_name": "corrected_shared_capacity_factor_seed42",
        "leaderboard_score_claim": False,
        "source_recipe": f"{SOURCE_PREFIX}__recipe.json",
        "training_correction": {
            "candidates": list(SHARED_CANDIDATES),
            "seed": 42,
            "row_filter": "actual >= 0.10 * group capacity",
            "target_scale": "capacity_factor",
            "scope": "pooled kpx_group_1/2/3 with model__group_id",
            "params": {
                name: corrected_recipe["models"][name] for name in SHARED_CANDIDATES
            },
        },
        "immutable_group_prediction_inputs": {
            name: str(path) for name, path in source_paths.items()
        },
        "corrected_v3": {
            "ensemble": corrected_recipe["ensemble"],
            "description": (
                "source v3 weights/affine/clip/power bins; immutable group "
                "predictions plus capacity-factor-corrected shared predictions"
            ),
        },
        "corrected_recent_v4": {
            "kpx_group_1": (
                "clip(1.1*(0.8*shared_q07_cf + 0.05*top200_q07 + "
                "0.15*energy_q06)-700, 0, 1.02*capacity)"
            ),
            "kpx_group_2": (
                "clip(1.025*(0.95*lgb_l1 + 0.05*energy_q06)-100, "
                "0, 1.02*capacity)"
            ),
            "kpx_group_3": (
                "clip(0.5*corrected_v3 + 0.5*clip(1.075*(0.7*shared_q07_cf + "
                "0.1*shared_l1_cf + 0.2*lgb_q07)-300, 0, 1.02*capacity), "
                "0, 1.02*capacity)"
            ),
        },
        "corrected_hybrid50": {
            "kpx_group_1": "clip(0.5*corrected_v3 + 0.5*corrected_recent_v4)",
            "kpx_group_2": "clip(0.5*corrected_v3 + 0.5*corrected_recent_v4)",
            "kpx_group_3": "corrected_recent_v4",
            "upper_capacity_fraction": 1.02,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    raw_dir = args.raw_dir.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    final_dir = args.final_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    paths = _output_paths(out_dir)
    _preflight(
        raw_dir=raw_dir,
        cache_dir=cache_dir,
        final_dir=final_dir,
        out_dir=out_dir,
        paths=paths,
        overwrite=bool(args.overwrite),
    )

    source_recipe_path = final_dir / f"{SOURCE_PREFIX}__recipe.json"
    source_recipe = read_recipe(source_recipe_path)
    corrected_recipe = _corrected_recipe(source_recipe)
    data = load_final_data(raw_dir, cache_dir)
    source_paths = _source_prediction_paths(final_dir)
    group_predictions = {
        name: _read_prediction(path, data.sample_times, context=name)
        for name, path in source_paths.items()
    }

    trained_shared: dict[str, TrainedCandidate] = {}
    for name in SHARED_CANDIDATES:
        specification = corrected_recipe["models"][name]
        print(f"training {name}: full eligible capacity-factor seed42 ...", flush=True)
        candidate = train_candidate(
            name,
            specification,
            recipe=corrected_recipe,
            data=data,
            trained={},
        )
        if candidate.scope != "shared":
            raise AssertionError(f"{name} did not produce a shared model")
        if candidate.training_rows != EXPECTED_TRAINING_ROWS:
            raise AssertionError(
                f"{name} eligible row counts changed: {candidate.training_rows}"
            )
        trained_shared[name] = candidate
        _atomic_joblib(
            _candidate_payload(candidate, specification, corrected_recipe["recipe_name"]),
            paths[f"model:{name}"],
        )
        _atomic_parquet(candidate.predictions, paths[f"prediction:{name}"])

    all_candidates: dict[str, TrainedCandidate] = dict(trained_shared)
    for name, prediction in group_predictions.items():
        all_candidates[name] = TrainedCandidate(
            name=name,
            scope="group",
            models=None,
            feature_names={},
            predictions=prediction,
            training_rows={},
        )
    corrected_v3 = _assemble_final_predictions(corrected_recipe, all_candidates)
    corrected_recent = _corrected_recent_v4(
        corrected_v3=corrected_v3,
        group_predictions=group_predictions,
        shared_predictions={
            name: candidate.predictions for name, candidate in trained_shared.items()
        },
    )
    corrected_hybrid = _corrected_hybrid50(corrected_v3, corrected_recent)
    variants = {
        "corrected_v3": corrected_v3,
        "corrected_recent_v4": corrected_recent,
        "corrected_hybrid50": corrected_hybrid,
    }

    checks: dict[str, Any] = {}
    for name, prediction in variants.items():
        _atomic_parquet(prediction, paths[f"prediction:{name}"])
        submission = _build_submission(data, prediction, corrected_recipe)
        _atomic_submission_csv(submission, paths[f"submission:{name}"])
        _verify_written_submission(
            paths[f"submission:{name}"], submission, corrected_recipe
        )
        checks[name] = {
            "prediction_range": _prediction_summary(prediction),
            "csv": _verify_csv(
                paths[f"submission:{name}"], submission, data.sample_submission
            ),
        }

    formula_config = _formula_config(corrected_recipe, source_paths)
    write_json_atomic(paths["config"], formula_config, overwrite=bool(args.overwrite))
    output_files = [path for key, path in paths.items() if key != "manifest"]
    input_files = [source_recipe_path, *data.input_files, *source_paths.values()]
    # De-duplicate paths while preserving their semantic order in the manifest.
    input_files = list(dict.fromkeys(Path(path).resolve() for path in input_files))
    manifest = make_manifest(
        artifact_type="baram_shared_capacity_factor_correction",
        parameters={
            "formula_config": formula_config,
            "capacities_kwh": CAPACITY_KWH,
            "target_columns": TARGET_COLS,
            "csv_encoding": "utf-8-sig",
            "csv_float_format": "%.6f",
            "source_artifacts_immutable": True,
            "leaderboard_score_claim": False,
        },
        input_files=input_files,
        output_files=output_files,
        results={
            "training_rows": EXPECTED_TRAINING_ROWS,
            "shared_prediction_ranges": {
                name: _prediction_summary(candidate.predictions)
                for name, candidate in trained_shared.items()
            },
            "variants": checks,
            "gate_diagnostics": {
                "corrected_locked_mechanical": {
                    "score": 0.640250,
                    "kind": "offline gate mechanical fixed result",
                    "leaderboard_evidence": False,
                },
                "corrected_recent_v4_calibration_fit": {
                    "score": 0.653108,
                    "kind": "calibration-fit diagnostic",
                    "independent": False,
                    "selection_safe": False,
                    "leaderboard_evidence": False,
                },
            },
        },
        project_dir=PROJECT_DIR,
    )
    write_json_atomic(paths["manifest"], manifest, overwrite=bool(args.overwrite))
    print(f"complete: {paths['manifest']}")
    for name in variants:
        csv_path = paths[f"submission:{name}"]
        print(f"{name}: {csv_path} sha256={sha256_file(csv_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

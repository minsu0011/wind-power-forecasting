"""Mechanically insert the locked spatial g2-Q07 replacement into corrected-v3.

No weight, affine constant, power-bin delta, or clip is searched here.  The
only change is the pre-locked candidate-level replacement

``lgb_q07[g2] = 0.9 * existing_1500 + 0.1 * spatial_v2_1500``.

The corrected-v3 ensemble is reconstructed exactly on 2023 and 2024 before
the replacement is applied.  A 2025 submission and six-candidate final wide
frame are written only if the completed ensemble improves in full/H1/H2 on
both years.  On failure, only diagnostic OOF-wide frames and a rejection
manifest are written; the earlier candidate-only 2025 artifact remains intact.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import os
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
    read_recipe,
)
from src.manifest import make_manifest, sha256_file, write_json_atomic  # noqa: E402
from src.metric import TARGET_COLS, score_details  # noqa: E402


CANDIDATES = (
    "lgb_l1",
    "lgb_q07",
    "shared_l1",
    "shared_q07",
    "top200_q07",
    "energy_q06",
)
G2_REPLACEMENT_COLUMN = "locked_recipe__kpx_group_2__core__q07"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir", type=Path, default=Path(r"data/local/open")
    )
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--recipe", type=Path, default=Path("configs/train_final.v3.locked.json")
    )
    parser.add_argument(
        "--spatial-dir",
        type=Path,
        default=Path("artifacts/experiments/spatial_v2_1500_audit"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/experiments/spatial_v2_corrected_v3_assembly"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _paths(out_dir: Path) -> dict[str, Path]:
    return {
        "results": out_dir / "corrected_v3_spatial_results.json",
        "manifest": out_dir / "corrected_v3_spatial_manifest.json",
        "dev_wide": out_dir / "wide" / "dev2023_g12_six_candidates_spatial.parquet",
        "gate_wide": out_dir / "wide" / "gate2024_six_candidates_spatial.parquet",
        "dev_final": out_dir / "oof" / "dev2023_corrected_v3_spatial.parquet",
        "gate_final": out_dir / "oof" / "gate2024_corrected_v3_spatial.parquet",
        "test_wide": out_dir / "wide" / "final2025_six_candidates_spatial.parquet",
        "test_final": out_dir / "final" / "corrected_v3_spatial_2025.parquet",
        "submission": out_dir / "final" / "corrected_v3_spatial_2025.csv",
    }


def _preflight(paths: Mapping[str, Path], overwrite: bool) -> None:
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"assembly outputs already exist: {existing}")


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        frame.to_parquet(temporary, engine="pyarrow", compression="zstd", index=True)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
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


def _read(path: Path, expected_columns: Sequence[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path).astype(float)
    if tuple(frame.columns) != tuple(expected_columns):
        raise ValueError(f"{path}: unexpected prediction columns")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f"{path}: predictions need a DatetimeIndex")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{path}: prediction index must be unique and sorted")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"{path}: predictions contain non-finite values")
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    return frame


def _dummy_candidate(name: str, frame: pd.DataFrame) -> TrainedCandidate:
    return TrainedCandidate(
        name=name,
        scope="assembly_only",
        models=None,
        feature_names={},
        predictions=frame,
        training_rows={},
    )


def _assemble(recipe: Mapping[str, Any], candidates: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    if tuple(candidates) != CANDIDATES:
        raise ValueError("six candidate names/order changed")
    return _assemble_final_predictions(
        recipe,
        {name: _dummy_candidate(name, frame) for name, frame in candidates.items()},
    )


def _add_dummy_g3(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["kpx_group_3"] = 0.0
    return output.loc[:, list(TARGET_COLS)]


def _dev_sources(artifact_dir: Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, list[Path]]:
    root = artifact_dir / "oof"
    paths = {
        "lgb_l1": root / "dev2023_lgb_l1_eligible_n1500.parquet",
        "lgb_q07": root / "dev2023_lgb_q07_eligible.parquet",
        "shared_l1": root / "dev2023_shared_l1_eligible.parquet",
        "shared_q07": root / "dev2023_shared_q07_eligible.parquet",
        "top200_q07": root / "dev2023_lgb_top200_q07_eligible.parquet",
        "energy_q06": root / "dev2023_lgb_q06_energywt_eligible.parquet",
    }
    candidates = {
        name: _add_dummy_g3(_read(path, TARGET_COLS[:2]))
        for name, path in paths.items()
    }
    baseline_path = root / "dev2023_locked_v3.parquet"
    baseline = _read(baseline_path, TARGET_COLS[:2])
    return candidates, baseline, [*paths.values(), baseline_path]


def _gate_sources(artifact_dir: Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, list[Path]]:
    paths = {
        "lgb_l1": artifact_dir / "gate/v3/predictions/lgb_l1_gate.parquet",
        "lgb_q07": artifact_dir / "gate/v3/predictions/lgb_q07_gate.parquet",
        "shared_l1": artifact_dir / "oof/gate2024_shared_l1_cf.parquet",
        "shared_q07": artifact_dir / "oof/gate2024_shared_q07_cf.parquet",
        "top200_q07": artifact_dir / "gate/v3/predictions/top200_q07_gate.parquet",
        "energy_q06": artifact_dir / "gate/v3/predictions/energy_q06_gate.parquet",
    }
    candidates = {
        name: _read(path, TARGET_COLS) for name, path in paths.items()
    }
    baseline_path = artifact_dir / "oof/gate2024_locked_v3_cf_fix.parquet"
    baseline = _read(baseline_path, TARGET_COLS)
    return candidates, baseline, [*paths.values(), baseline_path]


def _test_sources(
    artifact_dir: Path,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, list[Path]]:
    final_v3 = artifact_dir / "final_v3/predictions"
    corrected = artifact_dir / "final_cf_fix/predictions"
    prefix = "v3_locked_full_2025"
    paths = {
        "lgb_l1": final_v3 / f"{prefix}__lgb_l1_test.parquet",
        "lgb_q07": final_v3 / f"{prefix}__lgb_q07_test.parquet",
        "shared_l1": corrected / "shared_l1_cf_seed42_test.parquet",
        "shared_q07": corrected / "shared_q07_cf_seed42_test.parquet",
        "top200_q07": final_v3 / f"{prefix}__top200_q07_test.parquet",
        "energy_q06": final_v3 / f"{prefix}__energy_q06_test.parquet",
    }
    candidates = {
        name: _read(path, TARGET_COLS) for name, path in paths.items()
    }
    baseline_path = corrected / "corrected_v3_test.parquet"
    baseline = _read(baseline_path, TARGET_COLS)
    return candidates, baseline, [*paths.values(), baseline_path]


def _replacement_series(path: Path) -> pd.Series:
    frame = pd.read_parquet(path)
    if G2_REPLACEMENT_COLUMN not in frame:
        raise ValueError(f"{path}: locked g2 replacement column missing")
    result = frame[G2_REPLACEMENT_COLUMN].dropna().astype(float)
    result.index = pd.DatetimeIndex(result.index, name="forecast_kst_dtm")
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("locked g2 replacement contains non-finite values")
    return result


def _replace_g2_q07(
    candidates: Mapping[str, pd.DataFrame], replacement: pd.Series
) -> dict[str, pd.DataFrame]:
    output = {name: frame.copy() for name, frame in candidates.items()}
    if not output["lgb_q07"].index.equals(replacement.index):
        raise ValueError("g2 replacement index differs from lgb_q07 candidate")
    output["lgb_q07"]["kpx_group_2"] = replacement
    return output


def _wide(candidates: Mapping[str, pd.DataFrame], groups: Sequence[str]) -> pd.DataFrame:
    columns = {
        f"{name}__{group}": candidates[name][group]
        for name in CANDIDATES
        for group in groups
    }
    return pd.DataFrame(columns, index=next(iter(candidates.values())).index)


def _score_slices(
    labels: pd.DataFrame,
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    groups: Sequence[str],
    half: pd.Timestamp,
) -> dict[str, Any]:
    indexes = {
        "full": baseline.index,
        "h1": baseline.index[baseline.index < half],
        "h2": baseline.index[baseline.index >= half],
    }
    output: dict[str, Any] = {}
    for name, index in indexes.items():
        base_score = score_details(
            labels.reindex(index), baseline.reindex(index), target_cols=groups
        )
        candidate_score = score_details(
            labels.reindex(index), candidate.reindex(index), target_cols=groups
        )
        output[name] = {
            "baseline": asdict(base_score),
            "candidate": asdict(candidate_score),
            "score_gain": candidate_score.total_score - base_score.total_score,
        }
    return output


def _prebin_recipe(recipe: Mapping[str, Any]) -> dict[str, Any]:
    output = deepcopy(dict(recipe))
    output["ensemble"] = deepcopy(dict(recipe["ensemble"]))
    output["ensemble"]["power_bins"] = deepcopy(
        dict(recipe["ensemble"]["power_bins"])
    )
    output["ensemble"]["power_bins"]["kpx_group_2"] = None
    return output


def _max_abs(left: pd.DataFrame, right: pd.DataFrame) -> float:
    if not left.index.equals(right.index) or tuple(left.columns) != tuple(right.columns):
        raise ValueError("baseline reconstruction frames are not aligned")
    return float(np.max(np.abs(left.to_numpy(dtype=float) - right.to_numpy(dtype=float))))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.raw_dir = args.raw_dir.expanduser().resolve()
    args.artifact_dir = args.artifact_dir.expanduser().resolve()
    args.recipe = args.recipe.expanduser().resolve()
    args.spatial_dir = args.spatial_dir.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    paths = _paths(args.out_dir)
    _preflight(paths, bool(args.overwrite))
    recipe = read_recipe(args.recipe)

    label_path = args.raw_dir / "train/train_labels.csv"
    labels = pd.read_csv(label_path, encoding="utf-8-sig", parse_dates=["kst_dtm"])
    labels.set_index("kst_dtm", inplace=True)
    labels.index = pd.DatetimeIndex(labels.index, name="forecast_kst_dtm")

    dev_candidates, dev_expected, dev_inputs = _dev_sources(args.artifact_dir)
    gate_candidates, gate_expected, gate_inputs = _gate_sources(args.artifact_dir)
    dev_replacement_path = args.spatial_dir / "oof/spatial_v2_1500_historical.parquet"
    gate_replacement_path = args.spatial_dir / "oof/spatial_v2_1500_gate2024.parquet"
    dev_replacement = _replacement_series(dev_replacement_path)
    gate_replacement = _replacement_series(gate_replacement_path)

    dev_base_full = _assemble(recipe, dev_candidates)
    dev_base = dev_base_full.loc[:, list(TARGET_COLS[:2])]
    dev_reconstruct_diff = _max_abs(dev_base, dev_expected)
    gate_base = _assemble(recipe, gate_candidates)
    gate_reconstruct_diff = _max_abs(gate_base, gate_expected)
    if dev_reconstruct_diff > 1e-8 or gate_reconstruct_diff > 1e-8:
        raise AssertionError("corrected-v3 baseline reconstruction is not exact")

    dev_replaced_candidates = _replace_g2_q07(dev_candidates, dev_replacement)
    gate_replaced_candidates = _replace_g2_q07(gate_candidates, gate_replacement)
    dev_candidate = _assemble(recipe, dev_replaced_candidates).loc[
        :, list(TARGET_COLS[:2])
    ]
    gate_candidate = _assemble(recipe, gate_replaced_candidates)
    dev_scores = _score_slices(
        labels, dev_base, dev_candidate, TARGET_COLS[:2], pd.Timestamp("2023-07-01 01:00:00")
    )
    gate_scores = _score_slices(
        labels, gate_base, gate_candidate, TARGET_COLS, pd.Timestamp("2024-07-01 01:00:00")
    )

    prebin = _prebin_recipe(recipe)
    dev_pre_base = _assemble(prebin, dev_candidates).loc[:, list(TARGET_COLS[:2])]
    dev_pre_candidate = _assemble(prebin, dev_replaced_candidates).loc[
        :, list(TARGET_COLS[:2])
    ]
    gate_pre_base = _assemble(prebin, gate_candidates)
    gate_pre_candidate = _assemble(prebin, gate_replaced_candidates)
    prebin_scores = {
        "dev2023": _score_slices(
            labels,
            dev_pre_base,
            dev_pre_candidate,
            TARGET_COLS[:2],
            pd.Timestamp("2023-07-01 01:00:00"),
        ),
        "gate2024": _score_slices(
            labels,
            gate_pre_base,
            gate_pre_candidate,
            TARGET_COLS,
            pd.Timestamp("2024-07-01 01:00:00"),
        ),
    }

    dev_pass = all(value["score_gain"] > 0.0 for value in dev_scores.values())
    gate_pass = all(value["score_gain"] > 0.0 for value in gate_scores.values())
    promoted = dev_pass and gate_pass
    _atomic_parquet(_wide(dev_replaced_candidates, TARGET_COLS[:2]), paths["dev_wide"])
    _atomic_parquet(_wide(gate_replaced_candidates, TARGET_COLS), paths["gate_wide"])
    _atomic_parquet(dev_candidate, paths["dev_final"])
    _atomic_parquet(gate_candidate, paths["gate_final"])

    final_inputs: list[Path] = []
    final_status: dict[str, Any]
    if promoted:
        test_candidates, test_expected, test_inputs = _test_sources(args.artifact_dir)
        final_replacement_path = args.spatial_dir / "final/lgb_q07_replacement_test.parquet"
        final_replacement_frame = _read(final_replacement_path, TARGET_COLS)
        test_replaced = _replace_g2_q07(
            test_candidates, final_replacement_frame["kpx_group_2"]
        )
        test_base = _assemble(recipe, test_candidates)
        if _max_abs(test_base, test_expected) > 1e-8:
            raise AssertionError("final corrected-v3 baseline reconstruction changed")
        test_prediction = _assemble(recipe, test_replaced)
        _atomic_parquet(_wide(test_replaced, TARGET_COLS), paths["test_wide"])
        _atomic_parquet(test_prediction, paths["test_final"])
        sample_path = args.raw_dir / "sample_submission.csv"
        sample = pd.read_csv(
            sample_path,
            encoding="utf-8-sig",
            dtype={"forecast_id": "string", "forecast_kst_dtm": "string"},
        )
        sample_index = pd.DatetimeIndex(
            pd.to_datetime(sample["forecast_kst_dtm"], errors="raise"),
            name="forecast_kst_dtm",
        )
        if not sample_index.equals(test_prediction.index):
            raise ValueError("sample/final prediction indexes differ")
        for group in TARGET_COLS:
            sample[group] = test_prediction[group].to_numpy(dtype=float)
        _atomic_csv(sample, paths["submission"])
        final_inputs = [*test_inputs, final_replacement_path, sample_path]
        final_status = {
            "status": "created_after_both_years_all_slices_passed",
            "prediction": str(paths["test_final"]),
            "submission": str(paths["submission"]),
            "six_candidate_wide": str(paths["test_wide"]),
        }
    else:
        candidate_only = args.spatial_dir / "final/lgb_q07_replacement_test.parquet"
        final_status = {
            "status": "rejected_final_ensemble_interaction_reversed",
            "submission_created": False,
            "final_2025_six_candidate_wide_created": False,
            "candidate_only_artifact_preserved": str(candidate_only),
            "probabilistic_input_policy": (
                "2023/2024 diagnostic wide frames are available, but no final-2025 wide "
                "frame is authorised because corrected-v3 interaction failed."
            ),
        }

    results = {
        "status": "promoted" if promoted else "rejected",
        "mechanical_change": (
            "replace only lgb_q07[kpx_group_2] with 0.9 existing1500 + "
            "0.1 spatial1500; preserve weights/affine/power bins/clip"
        ),
        "additional_weight_search_performed": False,
        "baseline_reconstruction_max_abs_difference": {
            "dev2023": dev_reconstruct_diff,
            "gate2024": gate_reconstruct_diff,
        },
        "completed_ensemble_scores": {
            "dev2023": dev_scores,
            "gate2024": gate_scores,
        },
        "pre_power_bin_scores": prebin_scores,
        "stability": {
            "dev2023_full_h1_h2_all_positive": dev_pass,
            "gate2024_full_h1_h2_all_positive": gate_pass,
            "both_periods_passed": promoted,
        },
        "wide_oof": {
            "dev2023_g12": str(paths["dev_wide"]),
            "gate2024": str(paths["gate_wide"]),
            "candidate_order": list(CANDIDATES),
            "replacement_reflected": True,
        },
        "final2025": final_status,
    }
    write_json_atomic(paths["results"], results, overwrite=bool(args.overwrite))

    outputs = [
        paths["results"],
        paths["dev_wide"],
        paths["gate_wide"],
        paths["dev_final"],
        paths["gate_final"],
    ]
    if promoted:
        outputs.extend([paths["test_wide"], paths["test_final"], paths["submission"]])
    inputs = [
        label_path,
        args.recipe,
        dev_replacement_path,
        gate_replacement_path,
        *dev_inputs,
        *gate_inputs,
        *final_inputs,
    ]
    manifest = make_manifest(
        artifact_type="baram_corrected_v3_spatial_candidate_mechanical_assembly",
        parameters={
            "candidate_order": CANDIDATES,
            "replacement_candidate_weight": 0.10,
            "existing_candidate_weight": 0.90,
            "ensemble_weights_changed": False,
            "affine_changed": False,
            "power_bins_changed": False,
            "clip_changed": False,
            "full2025_requires_both_years_full_h1_h2_positive": True,
        },
        input_files=list(dict.fromkeys(Path(path).resolve() for path in inputs)),
        output_files=outputs,
        results={
            "status": results["status"],
            "stability": results["stability"],
            "final2025": final_status,
        },
        project_dir=PROJECT_DIR,
    )
    write_json_atomic(paths["manifest"], manifest, overwrite=bool(args.overwrite))
    print(json.dumps(results["stability"], ensure_ascii=False, indent=2))
    print(json.dumps(final_status, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

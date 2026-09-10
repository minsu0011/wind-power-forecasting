"""Resolve the spatial-v2 400-tree versus 1500-tree promotion mismatch.

Only the four candidates named by the already-completed 400-tree promotion are
retrained, with every locked hyperparameter unchanged except
``n_estimators=1500``.  A new replacement/blend recipe is selected from 2023
evidence and written before this program reads either 2024 labels or the old
postgate result JSON.

Historical selection folds
--------------------------
* group 2: train 2022, validate 2023 (full/H1/H2);
* group 3: train 2023 H1, validate 2023 H2 (full/Q3/Q4).

Post-lock confirmation folds
----------------------------
* group 2: train 2022-2023, validate 2024 (full/H1/H2);
* group 3: train 2023, validate 2024 (full/H1/H2 and Q3/Q4).

The comparison baseline is the project's existing matching 1500-tree L1 or
Q07 OOF candidate.  Candidate weights are restricted to 0.10, 0.20, or a
mechanical 1.00 replacement and are never changed after the pre-2024 lock.
Full-2025 models and candidate-assembly inputs are written only for a
pre-2024-primary recipe whose every required 2023 and 2024 segment improves.
The original 400-tree directory is read-only and remains untouched.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import joblib
from lightgbm import LGBMRegressor, log_evaluation
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.run_spatial_v2_postgate import (  # noqa: E402
    _dump_joblib_atomic,
    _make_model,
    _read_feature_pair,
    _score,
    _write_parquet_atomic,
)
from src.manifest import make_manifest, sha256_file, write_json_atomic  # noqa: E402
from src.metric import CAPACITY_KWH  # noqa: E402


PRE2024_ROWS = 17_520
ALL_TRAIN_ROWS = 26_304
TEST_ROWS = 8_760
FIRST_TIMESTAMP = pd.Timestamp("2022-01-01 01:00:00")
PRE2024_END = pd.Timestamp("2024-01-01 00:00:00")
GATE_START = pd.Timestamp("2024-01-01 01:00:00")
GATE_END = pd.Timestamp("2025-01-01 00:00:00")
TEST_START = pd.Timestamp("2025-01-01 01:00:00")
TEST_END = pd.Timestamp("2026-01-01 00:00:00")
G2_HIST_START = pd.Timestamp("2023-01-01 01:00:00")
G2_HIST_HALF = pd.Timestamp("2023-07-01 01:00:00")
G3_START = pd.Timestamp("2023-01-01 01:00:00")
G3_HIST_START = pd.Timestamp("2023-07-01 01:00:00")
G3_HIST_HALF = pd.Timestamp("2023-10-01 01:00:00")
GATE_HALF = pd.Timestamp("2024-07-01 01:00:00")
GATE_Q4 = pd.Timestamp("2024-10-01 01:00:00")
RECIPE_WEIGHTS = (0.10, 0.20, 1.00)


@dataclass(frozen=True)
class Candidate:
    group: str
    variant: str
    objective: str

    @property
    def identifier(self) -> str:
        return f"{self.group}__{self.variant}__{self.objective}"


# This list comes directly from the parent task. It is intentionally not read
# from the 2024-derived postgate result until after the new recipe is locked.
CANDIDATES = (
    Candidate("kpx_group_2", "core", "q07"),
    Candidate("kpx_group_3", "core", "l1"),
    Candidate("kpx_group_3", "sequence", "l1"),
    Candidate("kpx_group_3", "sequence", "q07"),
)
GROUPS = ("kpx_group_2", "kpx_group_3")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir", type=Path, default=Path(r"data/local/open")
    )
    parser.add_argument("--base-cache-dir", type=Path, default=Path("artifacts/cache"))
    parser.add_argument(
        "--augmentation-dir", type=Path, default=Path("artifacts/cache_v2_seq")
    )
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--source-lock",
        type=Path,
        default=Path("configs/spatial_v2_postgate_lock.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/experiments/spatial_v2_1500_audit"),
    )
    parser.add_argument("--n-jobs", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _paths(out_dir: Path) -> dict[str, Path]:
    return {
        "lock": out_dir / "spatial_v2_1500_pre2024_lock.json",
        "results": out_dir / "spatial_v2_1500_results.json",
        "manifest": out_dir / "spatial_v2_1500_manifest.json",
        "hist_oof": out_dir / "oof" / "spatial_v2_1500_historical.parquet",
        "gate_oof": out_dir / "oof" / "spatial_v2_1500_gate2024.parquet",
        "final_predictions": out_dir / "final" / "spatial_v2_1500_2025_predictions.parquet",
        "assembly": out_dir / "final" / "candidate_replacement_assembly.json",
    }


def _preflight(paths: Mapping[str, Path], overwrite: bool) -> None:
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        rendered = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(
            "spatial-v2 1500 audit outputs exist; original 400 artifacts are not "
            f"overwrite targets:\n{rendered}"
        )


def _read_labels(path: Path, nrows: int) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=["kst_dtm", *GROUPS],
        nrows=nrows,
        encoding="utf-8-sig",
        parse_dates=["kst_dtm"],
    ).set_index("kst_dtm")
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    expected_end = PRE2024_END if nrows == PRE2024_ROWS else GATE_END
    if len(frame) != nrows or frame.index.min() != FIRST_TIMESTAMP:
        raise ValueError("official label row/start boundary changed")
    if frame.index.max() != expected_end:
        raise ValueError("official label end boundary changed")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("official labels must be unique and chronological")
    if np.isinf(frame.to_numpy(dtype=float)).any():
        raise ValueError("official labels contain infinite values")
    return frame.astype(float)


def _load_source_lock(source_lock: Path, dev_results: Path) -> dict[str, Any]:
    lock = json.loads(source_lock.read_text(encoding="utf-8"))
    if lock.get("lock_stage") != "before_2024_oof":
        raise ValueError("source spatial-v2 lock is not pre-2024")
    observed = sha256_file(dev_results)
    expected = lock["selection_evidence"]["results_sha256"]
    if observed != expected:
        raise ValueError("source development evidence hash differs from its lock")
    if int(lock["model"]["n_estimators"]) != 400:
        raise ValueError("source mismatch audit expected a 400-tree lock")
    return lock


def _model_1500_config(source_lock: Mapping[str, Any]) -> dict[str, Any]:
    model = dict(source_lock["model"])
    model["n_estimators"] = 1_500
    if model.get("early_stopping") is not False:
        raise ValueError("source recipe unexpectedly enables early stopping")
    return model


def _load_features(
    base_dir: Path, augmentation_dir: Path
) -> tuple[dict[str, dict[str, pd.DataFrame]], list[Path]]:
    features: dict[str, dict[str, pd.DataFrame]] = {}
    inputs: list[Path] = []
    for group in GROUPS:
        frames, paths = _read_feature_pair(base_dir, augmentation_dir, group, "train")
        features[group] = frames
        inputs.extend(paths)
    return features, inputs


def _fit_predict(
    candidate: Candidate,
    labels: pd.DataFrame,
    feature_frames: Mapping[str, pd.DataFrame],
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
    model_config: Mapping[str, Any],
    n_jobs: int,
) -> tuple[LGBMRegressor, pd.Series, int]:
    group = candidate.group
    target = labels[group]
    eligible = target.notna() & (
        target >= float(model_config["eligible_fraction"]) * CAPACITY_KWH[group]
    )
    fit_mask = np.asarray(train_mask) & eligible.to_numpy(dtype=bool)
    if int(fit_mask.sum()) < 1_000:
        raise ValueError(f"{candidate.identifier}: too few eligible training rows")
    feature_frame = feature_frames[candidate.variant].reindex(labels.index)
    if feature_frame.isna().any().any():
        raise ValueError(f"{candidate.identifier}: feature coverage differs from labels")
    model = _make_model(dict(model_config), candidate.objective, n_jobs)
    model.fit(
        feature_frame.loc[fit_mask],
        target.loc[fit_mask] / CAPACITY_KWH[group],
        callbacks=[log_evaluation(period=0)],
    )
    if model.booster_.num_trees() != 1_500:
        raise AssertionError(
            f"{candidate.identifier}: fitted {model.booster_.num_trees()} trees, not 1500"
        )
    prediction = pd.Series(
        model.predict(feature_frame.loc[valid_mask])
        * CAPACITY_KWH[group],
        index=labels.index[valid_mask],
        name=candidate.identifier,
        dtype=float,
    )
    if not np.isfinite(prediction.to_numpy()).all():
        raise ValueError(f"{candidate.identifier}: non-finite model prediction")
    return model, prediction, int(fit_mask.sum())


def _historical_masks(group: str, index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    if group == "kpx_group_2":
        train = index < G2_HIST_START
        valid = index >= G2_HIST_START
        expected = (8_760, 8_760)
    else:
        train = (index >= G3_START) & (index < G3_HIST_START)
        valid = index >= G3_HIST_START
        expected = (4_344, 4_416)
    if (int(train.sum()), int(valid.sum())) != expected:
        raise AssertionError(f"{group}: historical fold sizes changed")
    if index[train].max() >= index[valid].min():
        raise AssertionError(f"{group}: historical train/valid overlap")
    return np.asarray(train), np.asarray(valid)


def _gate_masks(group: str, index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    if group == "kpx_group_2":
        train = index < GATE_START
        expected_train = PRE2024_ROWS
    else:
        train = (index >= G3_START) & (index < GATE_START)
        expected_train = 8_760
    valid = index >= GATE_START
    if int(train.sum()) != expected_train or int(valid.sum()) != 8_784:
        raise AssertionError(f"{group}: gate fold sizes changed")
    if index[train].max() >= index[valid].min():
        raise AssertionError(f"{group}: gate train/valid overlap")
    return np.asarray(train), np.asarray(valid)


def _historical_reference_paths(artifact_dir: Path) -> dict[str, Path]:
    return {
        "g2_q07_oof": artifact_dir / "oof" / "dev2023_lgb_q07_eligible.parquet",
        "g2_q07_model": artifact_dir / "models" / "dev2023_lgb_q07_eligible.joblib",
        "g3_oof": artifact_dir / "oof" / "g3dev2023h2_candidates.parquet",
        "g3_models": artifact_dir / "models" / "g3dev2023h2_candidates.joblib",
    }


def _load_historical_references(
    artifact_dir: Path,
) -> tuple[dict[tuple[str, str], pd.Series], list[Path], dict[str, Any]]:
    paths = _historical_reference_paths(artifact_dir)
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"existing 1500-tree historical references missing: {missing}")
    g2 = pd.read_parquet(paths["g2_q07_oof"])
    g3 = pd.read_parquet(paths["g3_oof"])
    g2_models = joblib.load(paths["g2_q07_model"])
    g3_models = joblib.load(paths["g3_models"])
    if int(g2_models["kpx_group_2"].get_params()["n_estimators"]) != 1_500:
        raise ValueError("existing group2 q07 reference is not 1500-tree")
    for objective in ("l1", "q07"):
        if int(g3_models[objective].get_params()["n_estimators"]) != 1_500:
            raise ValueError(f"existing group3 {objective} reference is not 1500-tree")
    references = {
        ("kpx_group_2", "q07"): g2["kpx_group_2"].astype(float),
        ("kpx_group_3", "l1"): g3["l1"].astype(float),
        ("kpx_group_3", "q07"): g3["q07"].astype(float),
    }
    return references, list(paths.values()), {
        "group2_q07": {
            "trees": 1_500,
            "learning_rate": float(
                g2_models["kpx_group_2"].get_params()["learning_rate"]
            ),
            "path": str(paths["g2_q07_model"]),
        },
        "group3_l1": {
            "trees": 1_500,
            "learning_rate": float(g3_models["l1"].get_params()["learning_rate"]),
            "path": str(paths["g3_models"]),
        },
        "group3_q07": {
            "trees": 1_500,
            "learning_rate": float(g3_models["q07"].get_params()["learning_rate"]),
            "path": str(paths["g3_models"]),
        },
    }


def _gate_reference_paths(artifact_dir: Path) -> dict[str, Path]:
    return {
        "l1_oof": artifact_dir / "gate" / "v3" / "predictions" / "lgb_l1_gate.parquet",
        "q07_oof": artifact_dir / "gate" / "v3" / "predictions" / "lgb_q07_gate.parquet",
        "l1_model": artifact_dir / "gate" / "v3" / "models" / "lgb_l1.joblib",
        "q07_model": artifact_dir / "gate" / "v3" / "models" / "lgb_q07.joblib",
    }


def _load_gate_references(
    artifact_dir: Path,
) -> tuple[dict[tuple[str, str], pd.Series], list[Path], dict[str, Any]]:
    paths = _gate_reference_paths(artifact_dir)
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"existing 1500-tree gate references missing: {missing}")
    frames = {
        objective: pd.read_parquet(paths[f"{objective}_oof"])
        for objective in ("l1", "q07")
    }
    audit: dict[str, Any] = {}
    for objective in ("l1", "q07"):
        payload = joblib.load(paths[f"{objective}_model"])
        trees = int(payload["candidate_config"]["params"]["n_estimators"])
        if trees != 1_500:
            raise ValueError(f"existing gate {objective} reference is not 1500-tree")
        audit[objective] = {
            "trees": trees,
            "learning_rate": float(
                payload["candidate_config"]["params"]["learning_rate"]
            ),
            "path": str(paths[f"{objective}_model"]),
        }
    references = {
        (group, objective): frames[objective][group].astype(float)
        for group in GROUPS
        for objective in ("l1", "q07")
    }
    return references, list(paths.values()), audit


def _segment_masks(
    candidate: Candidate, index: pd.DatetimeIndex, *, gate: bool
) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {
        "full": np.ones(len(index), dtype=bool),
    }
    if gate:
        masks["h1"] = np.asarray(index < GATE_HALF)
        masks["h2"] = np.asarray(index >= GATE_HALF)
        if candidate.group == "kpx_group_3":
            masks["q3"] = np.asarray((index >= GATE_HALF) & (index < GATE_Q4))
            masks["q4"] = np.asarray(index >= GATE_Q4)
    elif candidate.group == "kpx_group_2":
        masks["h1"] = np.asarray(index < G2_HIST_HALF)
        masks["h2"] = np.asarray(index >= G2_HIST_HALF)
    else:
        masks["q3"] = np.asarray(index < G3_HIST_HALF)
        masks["q4"] = np.asarray(index >= G3_HIST_HALF)
    if any(not mask.any() for mask in masks.values()):
        raise ValueError(f"{candidate.identifier}: an evaluation segment is empty")
    return masks


def _comparison(
    candidate: Candidate,
    actual: pd.Series,
    reference: pd.Series,
    prediction: pd.Series,
    *,
    gate: bool,
) -> dict[str, Any]:
    if not actual.index.equals(reference.index) or not actual.index.equals(
        prediction.index
    ):
        raise ValueError(f"{candidate.identifier}: comparison indexes differ")
    result: dict[str, Any] = {}
    for name, mask in _segment_masks(candidate, actual.index, gate=gate).items():
        baseline_score = _score(actual.loc[mask], reference.loc[mask], candidate.group)
        candidate_score = _score(actual.loc[mask], prediction.loc[mask], candidate.group)
        result[name] = {
            "existing_1500_reference": baseline_score,
            "spatial_v2_1500": candidate_score,
            "score_gain": candidate_score["score"] - baseline_score["score"],
        }
    return result


def _blend_prediction(reference: pd.Series, prediction: pd.Series, weight: float) -> pd.Series:
    if not reference.index.equals(prediction.index):
        raise ValueError("blend reference/candidate indexes differ")
    return (1.0 - float(weight)) * reference + float(weight) * prediction


def _select_recipe(
    candidate: Candidate,
    actual: pd.Series,
    reference: pd.Series,
    prediction: pd.Series,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates: dict[str, Any] = {}
    stable: list[tuple[float, float, str]] = []
    for weight in RECIPE_WEIGHTS:
        blended = _blend_prediction(reference, prediction, weight)
        scores = _comparison(
            candidate, actual, reference, blended, gate=False
        )
        gains = {name: float(value["score_gain"]) for name, value in scores.items()}
        key = f"w{int(weight * 100):03d}"
        passed = all(gain > 0.0 for gain in gains.values())
        candidates[key] = {
            "candidate_weight": weight,
            "existing_1500_weight": 1.0 - weight,
            "scores": scores,
            "required_segments_all_positive": passed,
        }
        if passed:
            stable.append((gains["full"], -weight, key))
    if stable:
        stable.sort(reverse=True)
        selected_key = stable[0][2]
        selected = {
            "eligible_for_2024_confirmation": True,
            "selected_key": selected_key,
            **candidates[selected_key],
        }
    else:
        selected = {
            "eligible_for_2024_confirmation": False,
            "selected_key": None,
            "candidate_weight": 0.0,
            "existing_1500_weight": 1.0,
            "reason": "no 10%, 20%, or 100% recipe improved every required 2023 segment",
        }
    return selected, candidates


def _flatten_oof(
    labels: pd.DataFrame,
    predictions: Mapping[str, pd.Series],
    references: Mapping[tuple[str, str], pd.Series],
    recipes: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    for candidate in CANDIDATES:
        identifier = candidate.identifier
        prediction = predictions[identifier]
        reference = references[(candidate.group, candidate.objective)].reindex(
            prediction.index
        )
        columns[f"actual__{candidate.group}"] = labels[candidate.group].reindex(
            prediction.index
        )
        columns[f"reference1500__{candidate.group}__{candidate.objective}"] = reference
        columns[f"spatial1500__{identifier}"] = prediction
        weight = float(recipes[identifier]["candidate_weight"])
        columns[f"locked_recipe__{identifier}"] = _blend_prediction(
            reference, prediction, weight
        )
    return pd.DataFrame(columns).sort_index()


def _audit_old_400(
    source_lock: Mapping[str, Any],
    postgate_results_path: Path,
    postgate_dir: Path,
) -> tuple[dict[str, Any], list[Path]]:
    results = json.loads(postgate_results_path.read_text(encoding="utf-8"))
    promoted = tuple(results.get("promoted_candidates", ()))
    expected = tuple(candidate.identifier for candidate in CANDIDATES)
    if set(promoted) != set(expected):
        raise ValueError(
            f"old 400-tree promoted set differs from delegated set: {promoted}"
        )
    models: dict[str, Any] = {}
    inputs = [postgate_results_path]
    for identifier in expected:
        path = postgate_dir / "models" / f"full_2025__{identifier}.joblib"
        model = joblib.load(path)
        trees = int(model.get_params()["n_estimators"])
        if trees != 400:
            raise ValueError(f"old promoted model {identifier} is not 400-tree")
        models[identifier] = {
            "n_estimators": trees,
            "learning_rate": float(model.get_params()["learning_rate"]),
            "path": str(path),
            "sha256": sha256_file(path),
        }
        inputs.append(path)
    return {
        "source_lock_n_estimators": int(source_lock["model"]["n_estimators"]),
        "old_promoted_candidates": list(promoted),
        "old_full_2025_models": models,
        "mismatch_confirmed": True,
    }, inputs


def _final_reference_path(artifact_dir: Path, objective: str) -> Path:
    return (
        artifact_dir
        / "final_v3"
        / "predictions"
        / f"v3_locked_full_2025__lgb_{objective}_test.parquet"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.raw_dir = args.raw_dir.expanduser().resolve()
    args.base_cache_dir = args.base_cache_dir.expanduser().resolve()
    args.augmentation_dir = args.augmentation_dir.expanduser().resolve()
    args.artifact_dir = args.artifact_dir.expanduser().resolve()
    args.source_lock = args.source_lock.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    if args.n_jobs == 0:
        raise ValueError("--n-jobs must not be zero")
    paths = _paths(args.out_dir)
    _preflight(paths, bool(args.overwrite))
    started = time.perf_counter()

    label_path = args.raw_dir / "train" / "train_labels.csv"
    dev_results_path = (
        args.artifact_dir
        / "experiments"
        / "spatial_v2_seq"
        / "spatial_v2_dev_results.json"
    )
    source_lock = _load_source_lock(args.source_lock, dev_results_path)
    model_config = _model_1500_config(source_lock)

    # Pre-lock phase: exactly 17,520 rows are read. The old postgate result JSON
    # is also deliberately unopened until this phase is complete.
    pre_labels = _read_labels(label_path, PRE2024_ROWS)
    features, feature_paths = _load_features(
        args.base_cache_dir, args.augmentation_dir
    )
    historical_references, historical_reference_paths, reference_audit = (
        _load_historical_references(args.artifact_dir)
    )
    hist_predictions: dict[str, pd.Series] = {}
    hist_models: dict[str, LGBMRegressor] = {}
    hist_metadata: dict[str, Any] = {}
    recipes: dict[str, Any] = {}
    recipe_candidates: dict[str, Any] = {}
    for candidate in CANDIDATES:
        train_mask, valid_mask = _historical_masks(candidate.group, pre_labels.index)
        print(f"historical 1500-tree {candidate.identifier}", flush=True)
        model, prediction, eligible_rows = _fit_predict(
            candidate,
            pre_labels,
            features[candidate.group],
            train_mask,
            valid_mask,
            model_config,
            args.n_jobs,
        )
        reference = historical_references[(candidate.group, candidate.objective)]
        if not reference.index.equals(prediction.index):
            raise ValueError(f"{candidate.identifier}: historical reference index differs")
        actual = pre_labels[candidate.group].reindex(prediction.index)
        recipe, alternatives = _select_recipe(
            candidate, actual, reference, prediction
        )
        recipes[candidate.identifier] = recipe
        recipe_candidates[candidate.identifier] = alternatives
        hist_predictions[candidate.identifier] = prediction
        hist_models[candidate.identifier] = model
        hist_metadata[candidate.identifier] = {
            "eligible_train_rows": eligible_rows,
            "feature_count": int(features[candidate.group][candidate.variant].shape[1]),
            "raw_comparison": _comparison(
                candidate, actual, reference, prediction, gate=False
            ),
        }
        model_path = args.out_dir / "models" / f"historical__{candidate.identifier}.joblib"
        _dump_joblib_atomic(model, model_path)
        hist_metadata[candidate.identifier]["model_path"] = str(model_path)

    # If core and sequence L1 are both eligible, the higher full-period 2023
    # recipe is the sole primary replacement. This choice is never revisited in
    # 2024, though both candidates are still audited there.
    primary_by_group_objective: dict[str, str] = {}
    for group, objective in (
        ("kpx_group_2", "q07"),
        ("kpx_group_3", "l1"),
        ("kpx_group_3", "q07"),
    ):
        eligible = [
            candidate
            for candidate in CANDIDATES
            if candidate.group == group
            and candidate.objective == objective
            and recipes[candidate.identifier]["eligible_for_2024_confirmation"]
        ]
        if eligible:
            eligible.sort(
                key=lambda candidate: recipe_candidates[candidate.identifier][
                    recipes[candidate.identifier]["selected_key"]
                ]["scores"]["full"]["score_gain"],
                reverse=True,
            )
            primary_by_group_objective[f"{group}__{objective}"] = eligible[0].identifier

    prelock = {
        "schema_version": 1,
        "lock_stage": "after_1500_tree_2023_selection_before_any_2024_label_or_postgate_result_read",
        "selection_label_rows": PRE2024_ROWS,
        "selection_max_timestamp": str(pre_labels.index.max()),
        "source_400_lock": str(args.source_lock),
        "source_400_lock_sha256": sha256_file(args.source_lock),
        "source_development_results_sha256": sha256_file(dev_results_path),
        "model": model_config,
        "allowed_recipe_weights": list(RECIPE_WEIGHTS),
        "candidates": [candidate.__dict__ for candidate in CANDIDATES],
        "recipes": recipes,
        "primary_by_group_objective": primary_by_group_objective,
        "selection_rule": (
            "weight must improve every required 2023 segment; maximise full-period "
            "gain; for duplicate group/objective candidates choose the largest "
            "2023 full-period gain"
        ),
    }
    write_json_atomic(paths["lock"], prelock, overwrite=bool(args.overwrite))
    prelock_sha = sha256_file(paths["lock"])
    _write_parquet_atomic(
        _flatten_oof(pre_labels, hist_predictions, historical_references, recipes),
        paths["hist_oof"],
    )
    print(f"pre-2024 1500 recipe locked sha256={prelock_sha}", flush=True)

    # Post-lock audit/read phase.
    postgate_dir = args.artifact_dir / "experiments" / "spatial_v2_postgate"
    postgate_results_path = postgate_dir / "spatial_v2_postgate_results.json"
    old_400_audit, old_400_paths = _audit_old_400(
        source_lock, postgate_results_path, postgate_dir
    )
    labels = _read_labels(label_path, ALL_TRAIN_ROWS)
    if not labels.iloc[:PRE2024_ROWS].equals(pre_labels):
        raise AssertionError("pre-2024 labels changed after recipe lock")
    if sha256_file(paths["lock"]) != prelock_sha:
        raise RuntimeError("1500-tree pre-2024 lock changed before gate confirmation")
    gate_references, gate_reference_paths, gate_reference_audit = (
        _load_gate_references(args.artifact_dir)
    )

    gate_predictions: dict[str, pd.Series] = {}
    gate_metadata: dict[str, Any] = {}
    confirmation: dict[str, Any] = {}
    for candidate in CANDIDATES:
        train_mask, valid_mask = _gate_masks(candidate.group, labels.index)
        print(f"fixed 2024 confirmation 1500-tree {candidate.identifier}", flush=True)
        model, prediction, eligible_rows = _fit_predict(
            candidate,
            labels,
            features[candidate.group],
            train_mask,
            valid_mask,
            model_config,
            args.n_jobs,
        )
        reference = gate_references[(candidate.group, candidate.objective)]
        if not reference.index.equals(prediction.index):
            raise ValueError(f"{candidate.identifier}: gate reference index differs")
        actual = labels[candidate.group].reindex(prediction.index)
        raw_comparison = _comparison(
            candidate, actual, reference, prediction, gate=True
        )
        recipe = recipes[candidate.identifier]
        weight = float(recipe["candidate_weight"])
        locked_prediction = _blend_prediction(reference, prediction, weight)
        locked_comparison = _comparison(
            candidate, actual, reference, locked_prediction, gate=True
        )
        gate_pass = bool(recipe["eligible_for_2024_confirmation"]) and all(
            value["score_gain"] > 0.0 for value in locked_comparison.values()
        )
        primary = (
            primary_by_group_objective.get(
                f"{candidate.group}__{candidate.objective}"
            )
            == candidate.identifier
        )
        confirmation[candidate.identifier] = {
            "recipe_locked_before_2024": recipe,
            "primary_locked_before_2024": primary,
            "raw_comparison": raw_comparison,
            "locked_recipe_comparison": locked_comparison,
            "every_required_2024_segment_positive": gate_pass,
            "eligible_for_full_2025": gate_pass and primary,
        }
        gate_predictions[candidate.identifier] = prediction
        gate_metadata[candidate.identifier] = {
            "eligible_train_rows": eligible_rows,
            "feature_count": int(features[candidate.group][candidate.variant].shape[1]),
        }
        model_path = args.out_dir / "models" / f"gate2024__{candidate.identifier}.joblib"
        _dump_joblib_atomic(model, model_path)
        gate_metadata[candidate.identifier]["model_path"] = str(model_path)

    _write_parquet_atomic(
        _flatten_oof(labels, gate_predictions, gate_references, recipes),
        paths["gate_oof"],
    )

    promoted = [
        candidate
        for candidate in CANDIDATES
        if confirmation[candidate.identifier]["eligible_for_full_2025"]
    ]
    final_model_paths: list[Path] = []
    final_input_paths: list[Path] = []
    final_summary: dict[str, Any]
    if promoted:
        test_features: dict[str, dict[str, pd.DataFrame]] = {}
        test_feature_paths: list[Path] = []
        for group in sorted({candidate.group for candidate in promoted}):
            frames, inputs = _read_feature_pair(
                args.base_cache_dir, args.augmentation_dir, group, "test"
            )
            test_features[group] = frames
            test_feature_paths.extend(inputs)
        test_index = next(iter(test_features.values()))["v1"].index
        if len(test_index) != TEST_ROWS or test_index.min() != TEST_START or test_index.max() != TEST_END:
            raise ValueError("2025 test feature period changed")
        final_columns: dict[str, pd.Series] = {}
        replacements: dict[str, pd.DataFrame] = {}
        assembly_entries: dict[str, Any] = {}
        for candidate in promoted:
            group = candidate.group
            target = labels[group]
            eligible = target.notna() & (
                target
                >= float(model_config["eligible_fraction"]) * CAPACITY_KWH[group]
            )
            model = _make_model(model_config, candidate.objective, args.n_jobs)
            model.fit(
                features[group][candidate.variant].loc[eligible],
                target.loc[eligible] / CAPACITY_KWH[group],
                callbacks=[log_evaluation(period=0)],
            )
            if model.booster_.num_trees() != 1_500:
                raise AssertionError("full-2025 spatial model tree count changed")
            spatial = pd.Series(
                model.predict(test_features[group][candidate.variant])
                * CAPACITY_KWH[group],
                index=test_index,
                dtype=float,
            )
            reference_path = _final_reference_path(
                args.artifact_dir, candidate.objective
            )
            reference_frame = pd.read_parquet(reference_path)
            if not reference_frame.index.equals(test_index):
                raise ValueError("final existing 1500 reference index differs")
            weight = float(recipes[candidate.identifier]["candidate_weight"])
            replacement = _blend_prediction(
                reference_frame[group], spatial, weight
            )
            candidate_name = f"lgb_{candidate.objective}"
            if candidate_name not in replacements:
                replacements[candidate_name] = reference_frame.copy()
            replacements[candidate_name][group] = replacement
            final_columns[f"existing1500__{candidate.identifier}"] = reference_frame[group]
            final_columns[f"spatial1500__{candidate.identifier}"] = spatial
            final_columns[f"replacement__{candidate.identifier}"] = replacement
            model_path = args.out_dir / "models" / f"full2025__{candidate.identifier}.joblib"
            _dump_joblib_atomic(model, model_path)
            final_model_paths.append(model_path)
            final_input_paths.extend([reference_path, *test_feature_paths])
            assembly_entries[candidate.identifier] = {
                "replace_candidate": candidate_name,
                "replace_group": group,
                "candidate_weight": weight,
                "existing_1500_weight": 1.0 - weight,
                "model_path": str(model_path),
            }
        _write_parquet_atomic(
            pd.DataFrame(final_columns, index=test_index), paths["final_predictions"]
        )
        replacement_paths: dict[str, str] = {}
        for candidate_name, frame in replacements.items():
            path = args.out_dir / "final" / f"{candidate_name}_replacement_test.parquet"
            _write_parquet_atomic(frame, path)
            replacement_paths[candidate_name] = str(path)
            final_model_paths.append(path)
        assembly = {
            "schema_version": 1,
            "pre2024_lock_sha256": prelock_sha,
            "eligible_candidates": assembly_entries,
            "compatible_full_candidate_prediction_inputs": replacement_paths,
            "instruction": (
                "Use these full candidate frames in place of the matching lgb_l1/lgb_q07 "
                "frame before the already-locked ensemble/affine/power-bin assembler. "
                "No final ensemble weights are changed here."
            ),
        }
        write_json_atomic(paths["assembly"], assembly, overwrite=bool(args.overwrite))
        final_summary = {
            "status": "assembly_inputs_written",
            "eligible_candidates": [candidate.identifier for candidate in promoted],
            "prediction_path": str(paths["final_predictions"]),
            "assembly_path": str(paths["assembly"]),
            "replacement_paths": replacement_paths,
        }
    else:
        final_summary = {
            "status": "blocked_no_candidate_stable_in_both_2023_and_2024",
            "eligible_candidates": [],
        }

    results = {
        "status": "completed",
        "experiment": "spatial_v2_400_to_1500_consistency_audit",
        "old_400_mismatch_audit": old_400_audit,
        "new_1500_model_config": model_config,
        "existing_1500_reference_audit": {
            "historical": reference_audit,
            "gate": gate_reference_audit,
            "note": (
                "Existing references use the project's original 1500-tree locked-v3 "
                "learning rate; spatial candidates keep the spatial lock's 0.04 rate."
            ),
        },
        "pre2024": {
            "lock_sha256": prelock_sha,
            "selection_or_weight_search_used_2024": False,
            "candidate_metadata": hist_metadata,
            "recipe_candidates": recipe_candidates,
            "selected_recipes": recipes,
            "primary_by_group_objective": primary_by_group_objective,
        },
        "fixed_2024_confirmation": {
            "candidate_metadata": gate_metadata,
            "records": confirmation,
            "candidate_or_weight_reselection_performed": False,
        },
        "full_2025": final_summary,
        "leakage_audit": {
            "pre2024_label_read_rows": PRE2024_ROWS,
            "pre2024_label_max": str(pre_labels.index.max()),
            "old_2024_postgate_result_read_after_new_lock": True,
            "full_label_read_after_new_lock": True,
            "new_lock_sha_reverified_before_gate": True,
            "early_stopping": False,
            "validation_labels_used_for_model_fit": False,
            "2024_weight_search_count": 0,
            "original_400_artifacts_modified": False,
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json_atomic(paths["results"], results, overwrite=bool(args.overwrite))

    output_files = [
        paths["lock"],
        paths["results"],
        paths["hist_oof"],
        paths["gate_oof"],
        *[
            Path(value["model_path"])
            for value in hist_metadata.values()
        ],
        *[
            Path(value["model_path"])
            for value in gate_metadata.values()
        ],
        *final_model_paths,
    ]
    if paths["final_predictions"].is_file():
        output_files.extend([paths["final_predictions"], paths["assembly"]])
    # De-duplicate paths without changing their semantic order.
    output_files = list(dict.fromkeys(path.resolve() for path in output_files))
    input_files = list(
        dict.fromkeys(
            path.resolve()
            for path in (
                label_path,
                args.source_lock,
                dev_results_path,
                *feature_paths,
                *historical_reference_paths,
                *gate_reference_paths,
                *old_400_paths,
                *final_input_paths,
            )
        )
    )
    manifest = make_manifest(
        artifact_type="baram_spatial_v2_1500_consistency_audit",
        parameters={
            "new_pre2024_lock": prelock,
            "n_estimators": 1_500,
            "allowed_recipe_weights": RECIPE_WEIGHTS,
            "2024_reselection_performed": False,
            "original_400_directory_read_only": str(postgate_dir),
        },
        input_files=input_files,
        output_files=output_files,
        results={
            "old_400_mismatch_confirmed": True,
            "primary_by_group_objective": primary_by_group_objective,
            "fixed_2024_eligible": [
                identifier
                for identifier, value in confirmation.items()
                if value["eligible_for_full_2025"]
            ],
            "full_2025": final_summary,
            "runtime_seconds": results["runtime_seconds"],
        },
        project_dir=PROJECT_DIR,
    )
    write_json_atomic(paths["manifest"], manifest, overwrite=bool(args.overwrite))
    print(
        json.dumps(
            {
                "pre2024_primary": primary_by_group_objective,
                "full_2025": final_summary,
                "results": str(paths["results"]),
                "runtime_seconds": results["runtime_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

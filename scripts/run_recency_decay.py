"""Strict group-1/2 recency-weight ablation for BARAM 2026.

The first causal check is 2022 + 2023 H1 train -> 2023 H2 validation (the
request's literal "2023 H1 valid with 2023 H1 train" would leak labels).  The
second is 2022-2023 train -> 2024 validation.  A half-life is eligible for 2025
only when it beats the equal-weight model in both full validation windows and
all four disjoint subperiods (2023 Q3/Q4 and 2024 H1/H2).

2024 is consumed post-gate development data and is never described as a new
independent validation result.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import sys
from typing import Any, Sequence

from lightgbm import LGBMRegressor
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
from src.metric import CAPACITY_KWH, TARGET_COLS, group_metrics  # noqa: E402
from src.recency import exponential_time_weights, select_stable_half_life  # noqa: E402


GROUPS = TARGET_COLS[:2]
HALF_LIVES = (180, 365, 730)
PREFIX = "v3_locked_full_2025"
MODEL_PARAMS = {
    "objective": "regression_l1",
    # Equal-compute screening budget; an accepted decay is refit at 1,500.
    "n_estimators": 600,
    "learning_rate": 0.025,
    "num_leaves": 31,
    "min_child_samples": 30,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.75,
    "reg_alpha": 0.05,
    "reg_lambda": 2.0,
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
    "random_state": 42,
    "n_jobs": 4,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(r"data/local/open"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/cache"))
    parser.add_argument("--final-dir", type=Path, default=Path("artifacts/final_v3"))
    parser.add_argument(
        "--corrected-dir", type=Path, default=Path("artifacts/final_cf_fix")
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/train_final.v3.locked.json")
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/postgate/recency_decay"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _interval(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="h", name="forecast_kst_dtm")


def _read_labels(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "train" / "train_labels.csv"
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame.index = pd.DatetimeIndex(
        pd.to_datetime(frame.pop("kst_dtm"), errors="raise"),
        name="forecast_kst_dtm",
    )
    return frame.loc[:, list(TARGET_COLS)].astype(float)


def _read_features(cache_dir: Path, group: str, kind: str) -> pd.DataFrame:
    number = int(group.rsplit("_", 1)[1])
    path = cache_dir / f"kpx_group_{number}_weather_{kind}.parquet"
    frame = pd.read_parquet(path)
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"{path} must have a DatetimeIndex")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{path} index must be unique and sorted")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError(f"{path} contains non-finite features")
    return frame


def _fit_predict(
    train_x: pd.DataFrame,
    train_y_kwh: pd.Series,
    valid_x: pd.DataFrame,
    *,
    capacity: float,
    half_life: int | None,
    n_estimators: int = 600,
) -> tuple[LGBMRegressor, np.ndarray, dict[str, float]]:
    eligible = train_y_kwh.notna() & (train_y_kwh >= 0.10 * capacity)
    x = train_x.loc[eligible]
    y = train_y_kwh.loc[eligible].to_numpy(dtype=float) / capacity
    if half_life is None:
        weights = np.ones(len(x), dtype=float)
    else:
        weights = exponential_time_weights(
            x.index,
            training_end=x.index.max(),
            half_life_days=float(half_life),
        )
    params = dict(MODEL_PARAMS)
    params["n_estimators"] = int(n_estimators)
    model = LGBMRegressor(**params)
    model.fit(x, y, sample_weight=weights)
    prediction = np.clip(
        model.predict(valid_x) * capacity,
        0.0,
        1.02 * capacity,
    )
    return model, prediction, {
        "rows": int(len(x)),
        "weight_min": float(np.min(weights)),
        "weight_max": float(np.max(weights)),
        "weight_mean": float(np.mean(weights)),
        "effective_sample_size": float(
            np.sum(weights) ** 2 / np.sum(np.square(weights))
        ),
    }


def _summary(actual: pd.Series, prediction: pd.Series, group: str) -> dict[str, Any]:
    metric = group_metrics(actual, prediction, CAPACITY_KWH[group], group_name=group)
    result = metric.as_dict()
    result["score"] = 0.5 * (metric.one_minus_nmae + metric.ficr)
    return result


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        frame.to_parquet(temporary, index=True)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8-sig", float_format="%.6f")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_final_candidates(
    final_dir: Path,
    corrected_dir: Path,
    index: pd.DatetimeIndex,
) -> tuple[dict[str, pd.DataFrame], list[Path]]:
    paths = {
        "lgb_l1": final_dir / "predictions" / f"{PREFIX}__lgb_l1_test.parquet",
        "lgb_q07": final_dir / "predictions" / f"{PREFIX}__lgb_q07_test.parquet",
        "top200_q07": final_dir
        / "predictions"
        / f"{PREFIX}__top200_q07_test.parquet",
        "energy_q06": final_dir
        / "predictions"
        / f"{PREFIX}__energy_q06_test.parquet",
        "shared_l1": corrected_dir
        / "predictions"
        / "shared_l1_cf_seed42_test.parquet",
        "shared_q07": corrected_dir
        / "predictions"
        / "shared_q07_cf_seed42_test.parquet",
    }
    result: dict[str, pd.DataFrame] = {}
    for name, path in paths.items():
        frame = pd.read_parquet(path)
        if len(frame) != len(index) or tuple(frame.columns) != TARGET_COLS:
            raise ValueError(f"invalid final candidate schema: {path}")
        frame.index = index
        result[name] = frame.astype(float)
    return result, list(paths.values())


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    labels = _read_labels(args.raw_dir)
    features = {
        group: _read_features(args.cache_dir, group, "train") for group in GROUPS
    }
    test_features = {
        group: _read_features(args.cache_dir, group, "test") for group in GROUPS
    }
    tracks = {
        "2023_h2": {
            "train": _interval("2022-01-01 01:00:00", "2023-07-01 00:00:00"),
            "valid": _interval("2023-07-01 01:00:00", "2024-01-01 00:00:00"),
            "segments": {
                "2023_h2_full": _interval(
                    "2023-07-01 01:00:00", "2024-01-01 00:00:00"
                ),
                "2023_q3": _interval(
                    "2023-07-01 01:00:00", "2023-10-01 00:00:00"
                ),
                "2023_q4": _interval(
                    "2023-10-01 01:00:00", "2024-01-01 00:00:00"
                ),
            },
        },
        "2024": {
            "train": _interval("2022-01-01 01:00:00", "2024-01-01 00:00:00"),
            "valid": _interval("2024-01-01 01:00:00", "2025-01-01 00:00:00"),
            "segments": {
                "2024_full": _interval(
                    "2024-01-01 01:00:00", "2025-01-01 00:00:00"
                ),
                "2024_h1": _interval(
                    "2024-01-01 01:00:00", "2024-07-01 00:00:00"
                ),
                "2024_h2": _interval(
                    "2024-07-01 01:00:00", "2025-01-01 00:00:00"
                ),
            },
        },
    }
    for group in GROUPS:
        if not features[group].index.equals(labels.index):
            raise ValueError(f"feature/label index mismatch for {group}")
    print(
        "validated causal tracks: "
        + ", ".join(
            f"{name} train={len(spec['train'])} valid={len(spec['valid'])}"
            for name, spec in tracks.items()
        )
    )
    if args.dry_run:
        print("dry-run complete; no models fitted and no outputs written")
        return 0

    output_paths = {
        f"oof:{track}:{group}": args.out_dir
        / "oof"
        / f"{track}__{group}.parquet"
        for track in tracks
        for group in GROUPS
    }
    output_paths.update(
        {
            "selected_lgb_l1": args.out_dir
            / "predictions"
            / "selected_lgb_l1_2025.parquet",
            "selected_base": args.out_dir
            / "predictions"
            / "selected_corrected_v3_2025.parquet",
            "submission": args.out_dir / "recency_selected_corrected_v3_2025.csv",
            "results": args.out_dir / "results.json",
            "manifest": args.out_dir / "manifest.json",
        }
    )
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("outputs exist; pass --overwrite")

    results: dict[str, Any] = {
        "warning": (
            "2024 is consumed post-gate development data, not a new independent "
            "validation. The causal first track is 2022+2023H1 -> 2023H2."
        ),
        "half_lives_days": list(HALF_LIVES),
        "model": {
            "objective": "regression_l1",
            **MODEL_PARAMS,
            "screening_n_estimators": 600,
            "selected_final_n_estimators": 1500,
        },
        "tracks": {},
    }
    all_predictions: dict[tuple[str, str], pd.DataFrame] = {}
    for track, specification in tracks.items():
        train_index = specification["train"]
        valid_index = specification["valid"]
        results["tracks"][track] = {}
        for group in GROUPS:
            print(f"{track} {group} uniform", flush=True)
            prediction_frame = pd.DataFrame(index=valid_index)
            _, uniform_prediction, uniform_weight = _fit_predict(
                features[group].loc[train_index],
                labels.loc[train_index, group],
                features[group].loc[valid_index],
                capacity=CAPACITY_KWH[group],
                half_life=None,
            )
            prediction_frame["uniform"] = uniform_prediction
            weights = {"uniform": uniform_weight}
            for half_life in HALF_LIVES:
                print(f"{track} {group} half_life_{half_life}", flush=True)
                _, prediction, weight_summary = _fit_predict(
                    features[group].loc[train_index],
                    labels.loc[train_index, group],
                    features[group].loc[valid_index],
                    capacity=CAPACITY_KWH[group],
                    half_life=half_life,
                )
                prediction_frame[f"half_life_{half_life}"] = prediction
                weights[str(half_life)] = weight_summary
            segment_metrics: dict[str, Any] = {}
            for segment_name, segment_index in specification["segments"].items():
                baseline = _summary(
                    labels.loc[segment_index, group],
                    prediction_frame.loc[segment_index, "uniform"],
                    group,
                )
                candidates: dict[str, Any] = {}
                for half_life in HALF_LIVES:
                    metric = _summary(
                        labels.loc[segment_index, group],
                        prediction_frame.loc[
                            segment_index, f"half_life_{half_life}"
                        ],
                        group,
                    )
                    metric["delta_score"] = metric["score"] - baseline["score"]
                    metric["delta_one_minus_nmae"] = (
                        metric["one_minus_nmae"] - baseline["one_minus_nmae"]
                    )
                    metric["delta_ficr"] = metric["ficr"] - baseline["ficr"]
                    candidates[str(half_life)] = metric
                segment_metrics[segment_name] = {
                    "uniform": baseline,
                    "decays": candidates,
                }
            results["tracks"][track][group] = {
                "train_start": str(train_index.min()),
                "train_end": str(train_index.max()),
                "valid_start": str(valid_index.min()),
                "valid_end": str(valid_index.max()),
                "weights": weights,
                "segments": segment_metrics,
            }
            all_predictions[(track, group)] = prediction_frame

    required_segments = (
        "2023_h2_full",
        "2023_q3",
        "2023_q4",
        "2024_full",
        "2024_h1",
        "2024_h2",
    )
    selected: dict[str, int | None] = {}
    selection_diagnostics: dict[str, Any] = {}
    for group in GROUPS:
        deltas: dict[str, dict[str, float]] = {
            str(half_life): {} for half_life in HALF_LIVES
        }
        for track in tracks:
            for segment, values in results["tracks"][track][group]["segments"].items():
                for half_life in HALF_LIVES:
                    deltas[str(half_life)][segment] = float(
                        values["decays"][str(half_life)]["delta_score"]
                    )
        choice = select_stable_half_life(
            deltas,
            half_lives=HALF_LIVES,
            required_segments=required_segments,
        )
        selected[group] = choice
        selection_diagnostics[group] = {
            "selected_half_life_days": choice,
            "required_segments": list(required_segments),
            "strict_rule": "delta_score > 0 in every required segment",
            "deltas": deltas,
            "recent_only_2024_forbidden": True,
        }

    # Train a final L1 only for groups that passed the predeclared stability gate.
    test_index = test_features[GROUPS[0]].index
    final_candidates, final_input_paths = _load_final_candidates(
        args.final_dir, args.corrected_dir, test_index
    )
    selected_lgb_l1 = final_candidates["lgb_l1"].copy()
    final_models: dict[str, Any] = {}
    final_training: dict[str, Any] = {}
    for group in GROUPS:
        choice = selected[group]
        if choice is None:
            final_training[group] = {
                "selected": "uniform_existing",
                "reason": "no half-life improved every required segment",
            }
            continue
        print(f"2025 final {group} half_life_{choice}", flush=True)
        model, prediction, weight_summary = _fit_predict(
            features[group],
            labels[group],
            test_features[group],
            capacity=CAPACITY_KWH[group],
            half_life=choice,
            n_estimators=1500,
        )
        selected_lgb_l1[group] = prediction
        final_models[group] = model
        final_training[group] = {
            "selected": f"half_life_{choice}",
            "weight_summary": weight_summary,
        }

    recipe = read_recipe(args.config)
    trained = {
        name: TrainedCandidate(
            name=name,
            scope="group" if name not in {"shared_l1", "shared_q07"} else "shared",
            models=None,
            feature_names={},
            predictions=(selected_lgb_l1 if name == "lgb_l1" else prediction),
            training_rows={},
        )
        for name, prediction in final_candidates.items()
    }
    selected_base = _assemble_final_predictions(recipe, trained)
    sample_path = args.raw_dir / "sample_submission.csv"
    sample = pd.read_csv(sample_path, encoding="utf-8-sig")
    submission = sample.copy()
    for group in TARGET_COLS:
        submission[group] = selected_base[group].to_numpy(dtype=float)
    if len(submission) != 8760 or not np.isfinite(
        submission.loc[:, list(TARGET_COLS)].to_numpy(dtype=float)
    ).all():
        raise AssertionError("invalid recency submission")

    for (track, group), frame in all_predictions.items():
        _atomic_parquet(frame, output_paths[f"oof:{track}:{group}"])
    _atomic_parquet(selected_lgb_l1, output_paths["selected_lgb_l1"])
    _atomic_parquet(selected_base, output_paths["selected_base"])
    _atomic_csv(submission, output_paths["submission"])
    results["selection"] = selection_diagnostics
    results["selected_half_lives"] = selected
    results["final_training"] = final_training
    results["candidate_2025"] = {
        "score_claim": False,
        "path": str(output_paths["submission"]),
        "identical_to_corrected_v3_when_all_rejected": all(
            value is None for value in selected.values()
        ),
    }
    write_json_atomic(output_paths["results"], results, overwrite=bool(args.overwrite))

    input_files = [
        args.raw_dir / "train" / "train_labels.csv",
        sample_path,
        args.config,
        *[
            args.cache_dir / f"kpx_group_{int(group.rsplit('_', 1)[1])}_weather_train.parquet"
            for group in GROUPS
        ],
        *[
            args.cache_dir / f"kpx_group_{int(group.rsplit('_', 1)[1])}_weather_test.parquet"
            for group in GROUPS
        ],
        *final_input_paths,
    ]
    artifact_outputs = [
        path for key, path in output_paths.items() if key != "manifest"
    ]
    manifest = make_manifest(
        artifact_type="baram_recency_decay_strict_ablation",
        parameters={
            "half_lives_days": HALF_LIVES,
            "selection_rule": "positive delta in all six validation segments",
            "leaderboard_score_claim": False,
            "untouched_gate_claim": False,
        },
        input_files=list(dict.fromkeys(Path(path).resolve() for path in input_files)),
        output_files=artifact_outputs,
        results={
            "selected_half_lives": selected,
            "candidate_sha256": sha256_file(output_paths["submission"]),
        },
        project_dir=PROJECT_DIR,
    )
    write_json_atomic(output_paths["manifest"], manifest, overwrite=bool(args.overwrite))
    print(f"selected half-lives: {selected}")
    print(f"complete: {output_paths['results']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

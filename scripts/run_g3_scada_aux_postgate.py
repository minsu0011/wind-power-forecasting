"""Post-gate group-3 experiment with leakage-safe SCADA auxiliary features.

This is explicitly *not* an untouched-gate evaluation: the 2024 gate has
already been consumed.  The experiment treats 2024 as development OOF and asks
whether weather->SCADA wind predictions improve direct group-3 LightGBM models.

Leakage contract
----------------
* Main split: 2023 train -> 2024 validation.
* Raw UNISON SCADA is hard-cut before 2024, prior to hourly aggregation.
* Main-train auxiliary values are calendar-month-block OOF predictions.
* 2024 auxiliary values are produced by an encoder fitted on 2023 only.
* A historical check repeats the process as 2023 H1 train -> 2023 H2 valid;
  its encoder receives H1 SCADA only.
* Observed validation SCADA is never passed to fit or transform.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import joblib
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.final_training import read_recipe  # noqa: E402
from src.manifest import make_manifest, sha256_file, write_json_atomic  # noqa: E402
from src.metric import CAPACITY_KWH, group_metrics  # noqa: E402
from src.scada import (  # noqa: E402
    GROUP_COLUMN,
    SCADAAuxiliaryRegressor,
    TIME_COLUMN,
    aggregate_scada_hourly,
    auxiliary_target_frame,
    cross_fit_scada_auxiliary,
)


GROUP = "kpx_group_3"
GROUP_NUMBER = 3
CAPACITY = CAPACITY_KWH[GROUP]

TRAIN_START = pd.Timestamp("2023-01-01 01:00:00")
TRAIN_END = pd.Timestamp("2024-01-01 00:00:00")
VALID_START = pd.Timestamp("2024-01-01 01:00:00")
VALID_END = pd.Timestamp("2025-01-01 00:00:00")
RAW_SCADA_CUTOFF = pd.Timestamp("2024-01-01 00:00:00")

HIST_TRAIN_START = TRAIN_START
HIST_TRAIN_END = pd.Timestamp("2023-07-01 00:00:00")
HIST_VALID_START = pd.Timestamp("2023-07-01 01:00:00")
HIST_VALID_END = TRAIN_END

AUX_TARGETS = (
    "obs_ws_mean",
    "obs_ws_std",
    "obs_wd_sin",
    "obs_wd_cos",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"data/local/open"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("artifacts/cache"),
    )
    parser.add_argument(
        "--gate-dir",
        type=Path,
        default=Path("artifacts/gate/v3"),
        help="consumed gate artifacts used only as fixed reference predictions",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_final.v3.locked.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/postgate/g3_scada_aux"),
    )
    parser.add_argument("--aux-estimators", type=int, default=260)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _exact_index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="h", name="forecast_kst_dtm")


def _assert_exact_index(
    index: pd.Index,
    start: pd.Timestamp,
    end: pd.Timestamp,
    context: str,
) -> None:
    expected = _exact_index(start, end)
    if not isinstance(index, pd.DatetimeIndex) or not index.equals(expected):
        raise ValueError(
            f"{context} must equal {start}..{end} hourly; "
            f"rows={len(index)}, min={index.min()}, max={index.max()}"
        )


def _read_unison_before(path: Path, cutoff: pd.Timestamp) -> pd.DataFrame:
    retained: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, encoding="utf-8-sig", chunksize=20_000):
        timestamp = pd.to_datetime(chunk["kst_dtm"], errors="raise")
        keep = timestamp < cutoff
        if keep.any():
            part = chunk.loc[keep].copy()
            part["kst_dtm"] = timestamp.loc[keep]
            retained.append(part)
        if (~keep).any():
            break
    if not retained:
        raise ValueError("no pre-cutoff UNISON rows")
    frame = pd.concat(retained, ignore_index=True)
    maximum = pd.to_datetime(frame["kst_dtm"]).max()
    if maximum >= cutoff:
        raise AssertionError("2024 validation SCADA crossed the raw hard cutoff")
    return frame


def _aux_weather_columns(columns: pd.Index) -> list[str]:
    tokens = (
        "__idw__hub_ws",
        "__nearest__hub_ws",
        "cross__hub_ws",
        "__idw__ws10",
        "__nearest__ws10",
        "__idw__ws50",
        "__nearest__ws50",
        "__idw__ws80",
        "__nearest__ws80",
        "__idw__ws100",
        "__nearest__ws100",
        "__idw__wind_range50",
        "__nearest__wind_range50",
        "__idw__heightAboveGround_10_10u",
        "__idw__heightAboveGround_10_10v",
        "__nearest__heightAboveGround_10_10u",
        "__nearest__heightAboveGround_10_10v",
        "__idw__heightAboveGround_50_50MU",
        "__idw__heightAboveGround_50_50MV",
        "__nearest__heightAboveGround_50_50MU",
        "__nearest__heightAboveGround_50_50MV",
        "__idw__heightAboveGround_80_u",
        "__idw__heightAboveGround_80_v",
        "__nearest__heightAboveGround_80_u",
        "__nearest__heightAboveGround_80_v",
        "__idw__heightAboveGround_100_100u",
        "__idw__heightAboveGround_100_100v",
        "__nearest__heightAboveGround_100_100u",
        "__nearest__heightAboveGround_100_100v",
    )
    selected = [
        column
        for column in columns
        if column.startswith("time__") or any(token in column for token in tokens)
    ]
    if not selected:
        raise ValueError("no weather columns selected for SCADA encoder")
    return selected


def _with_aux_keys(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.insert(0, TIME_COLUMN, result.index)
    result.insert(1, GROUP_COLUMN, GROUP_NUMBER)
    return result.reset_index(drop=True)


def _month_block_splits(index: pd.DatetimeIndex) -> list[tuple[np.ndarray, np.ndarray]]:
    periods = index.to_period("M")
    unique = periods.unique().sort_values()
    if len(unique) < 2:
        raise ValueError("month-block OOF needs at least two calendar months")
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for period in unique:
        valid = np.flatnonzero(periods == period)
        train = np.flatnonzero(periods != period)
        if not len(train) or not len(valid) or np.intersect1d(train, valid).size:
            raise AssertionError(f"invalid month holdout {period}")
        splits.append((train, valid))
    return splits


def _aux_estimator(args: argparse.Namespace) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression_l1",
        n_estimators=args.aux_estimators,
        learning_rate=0.035,
        num_leaves=15,
        min_child_samples=64,
        colsample_bytree=0.80,
        subsample=0.85,
        subsample_freq=1,
        reg_alpha=0.10,
        reg_lambda=3.0,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )


def _make_aux_features(
    args: argparse.Namespace,
    *,
    weather_train: pd.DataFrame,
    weather_valid: pd.DataFrame,
    scada_train: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, SCADAAuxiliaryRegressor, dict[str, float]]:
    if scada_train[TIME_COLUMN].max() >= weather_valid.index.min():
        raise AssertionError("validation SCADA entered auxiliary fit inputs")
    columns = _aux_weather_columns(weather_train.columns)
    train_input = _with_aux_keys(weather_train.loc[:, columns])
    valid_input = _with_aux_keys(weather_valid.loc[:, columns])
    kwargs = {
        "feature_columns": columns,
        "target_columns": AUX_TARGETS,
        "group": GROUP_NUMBER,
        "estimator": _aux_estimator(args),
        "min_samples": 168,
        "random_state": args.seed,
    }
    train_oof = cross_fit_scada_auxiliary(
        train_input,
        scada_train,
        _month_block_splits(weather_train.index),
        model_kwargs=kwargs,
    )
    train_oof.index = weather_train.index
    if train_oof.isna().any().any():
        raise AssertionError("SCADA auxiliary train OOF is incomplete")

    full_model = SCADAAuxiliaryRegressor(**kwargs)
    full_model.fit(train_input, scada_train)
    valid_prediction = full_model.predict_auxiliary(valid_input)
    valid_prediction.index = weather_valid.index
    if valid_prediction.isna().any().any():
        raise AssertionError("SCADA auxiliary valid prediction is incomplete")

    actual = (
        scada_train.loc[scada_train[GROUP_COLUMN] == GROUP_NUMBER]
        .set_index(TIME_COLUMN)
        .reindex(weather_train.index)
    )
    quality: dict[str, float] = {}
    for target in AUX_TARGETS:
        prediction_column = f"pred_scada_{target}"
        mask = actual[target].notna()
        observed = actual.loc[mask, target].to_numpy(dtype=float)
        predicted = train_oof.loc[mask, prediction_column].to_numpy(dtype=float)
        quality[f"{target}__mae"] = float(np.mean(np.abs(observed - predicted)))
        quality[f"{target}__correlation"] = float(
            np.corrcoef(observed, predicted)[0, 1]
        )
    return (
        train_oof.astype("float32"),
        valid_prediction.astype("float32"),
        full_model,
        quality,
    )


def _model_params(
    recipe: Mapping[str, Any],
    candidate: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    specification = recipe["models"][candidate]
    params = dict(specification["params"])
    params["objective"] = (
        "regression_l1" if specification["objective"] == "l1" else "quantile"
    )
    if specification["objective"] == "quantile":
        params["alpha"] = float(specification["alpha"])
    params["random_state"] = args.seed
    params["n_jobs"] = args.n_jobs
    return params


def _fit_predict_direct(
    args: argparse.Namespace,
    recipe: Mapping[str, Any],
    *,
    objective_candidate: str,
    weather_train: pd.DataFrame,
    target_train: pd.Series,
    weather_valid: pd.DataFrame,
    aux_train: pd.DataFrame | None,
    aux_valid: pd.DataFrame | None,
) -> tuple[np.ndarray, LGBMRegressor, float]:
    train_x = weather_train.copy()
    valid_x = weather_valid.copy()
    if aux_train is not None and aux_valid is not None:
        train_x = pd.concat([train_x, aux_train], axis=1)
        valid_x = pd.concat([valid_x, aux_valid], axis=1)
    if not train_x.columns.equals(valid_x.columns):
        raise AssertionError("direct model feature schemas differ")
    if not np.isfinite(train_x.to_numpy()).all() or not np.isfinite(
        valid_x.to_numpy()
    ).all():
        raise ValueError("direct model features contain non-finite values")
    eligible = target_train.notna() & (target_train >= 0.10 * CAPACITY)
    started = time.perf_counter()
    model = LGBMRegressor(**_model_params(recipe, objective_candidate, args))
    model.fit(train_x.loc[eligible], target_train.loc[eligible])
    prediction = np.asarray(model.predict(valid_x), dtype=float)
    elapsed = time.perf_counter() - started
    return prediction, model, elapsed


def _metric_dict(
    actual: pd.Series,
    prediction: pd.Series | np.ndarray,
    *,
    name: str,
) -> dict[str, Any]:
    details = group_metrics(actual, prediction, CAPACITY, group_name=GROUP)
    payload = asdict(details)
    payload["group_score"] = 0.5 * (
        details.one_minus_nmae + details.ficr
    )
    payload["slice"] = name
    return payload


def _score_slices(
    actual: pd.Series,
    prediction: pd.Series,
    *,
    split_point: pd.Timestamp,
) -> dict[str, dict[str, Any]]:
    if not actual.index.equals(prediction.index):
        raise ValueError("actual/prediction indexes differ")
    first = actual.index < split_point
    second = actual.index >= split_point
    return {
        "full": _metric_dict(actual, prediction, name="full"),
        "first_half": _metric_dict(
            actual.loc[first], prediction.loc[first], name="first_half"
        ),
        "second_half": _metric_dict(
            actual.loc[second], prediction.loc[second], name="second_half"
        ),
    }


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        frame.to_parquet(temporary, engine="pyarrow", compression="zstd", index=True)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_joblib(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        joblib.dump(payload, temporary, compress=3)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _preflight(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists():
        files = [path for path in out_dir.rglob("*") if path.is_file()]
        if files and not overwrite:
            raise FileExistsError(
                f"post-gate output already exists ({files[0]}); pass --overwrite"
            )
    out_dir.mkdir(parents=True, exist_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.data_root = args.data_root.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.gate_dir = args.gate_dir.expanduser().resolve()
    args.config = args.config.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    _preflight(args.out_dir, bool(args.overwrite))
    started = time.perf_counter()

    recipe = read_recipe(args.config)
    if not recipe["recipe_locked"]:
        raise ValueError("the comparison recipe must be locked")
    labels_path = args.data_root / "train" / "train_labels.csv"
    scada_path = args.data_root / "train" / "scada_unison_train.csv"
    weather_path = args.cache_dir / f"{GROUP}_weather_train.parquet"
    reference_paths = {
        "gate_ensemble": args.gate_dir / "gate_oof.parquet",
        "gate_lgb_l1": args.gate_dir / "predictions" / "lgb_l1_gate.parquet",
        "gate_lgb_q07": args.gate_dir / "predictions" / "lgb_q07_gate.parquet",
        "gate_shared_l1": args.gate_dir / "predictions" / "shared_l1_gate.parquet",
        "gate_shared_q07": args.gate_dir / "predictions" / "shared_q07_gate.parquet",
    }
    required = [
        labels_path,
        scada_path,
        weather_path,
        args.config,
        *reference_paths.values(),
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing experiment inputs: {missing}")

    labels = pd.read_csv(
        labels_path,
        encoding="utf-8-sig",
        parse_dates=["kst_dtm"],
    ).set_index("kst_dtm").sort_index()[GROUP]
    weather = pd.read_parquet(weather_path)
    train_index = _exact_index(TRAIN_START, TRAIN_END)
    valid_index = _exact_index(VALID_START, VALID_END)
    _assert_exact_index(weather.reindex(train_index).index, TRAIN_START, TRAIN_END, "train")
    _assert_exact_index(weather.reindex(valid_index).index, VALID_START, VALID_END, "valid")
    weather_train = weather.reindex(train_index)
    weather_valid = weather.reindex(valid_index)
    target_train = labels.reindex(train_index)
    target_valid = labels.reindex(valid_index)
    if target_train.isna().any():
        raise ValueError("group-3 training labels are incomplete")
    # The official scorer ignores unavailable actuals. Group 3 has six such
    # hours in 2024; retain them as NaN and never impute them.
    if np.isinf(target_valid.to_numpy()).any() or not target_valid.notna().any():
        raise ValueError("group-3 validation labels are invalid")
    if weather_train.isna().any().any() or weather_valid.isna().any().any():
        raise ValueError("group-3 weather cache is incomplete")

    raw_scada = _read_unison_before(scada_path, RAW_SCADA_CUTOFF)
    hourly_scada = aggregate_scada_hourly(raw_scada, "unison")
    hourly_scada = hourly_scada.loc[
        hourly_scada[GROUP_COLUMN].eq(GROUP_NUMBER)
        & hourly_scada[TIME_COLUMN].isin(train_index)
    ].copy()
    if pd.to_datetime(raw_scada["kst_dtm"]).max() >= RAW_SCADA_CUTOFF:
        raise AssertionError("raw SCADA cutoff failed")
    if hourly_scada[TIME_COLUMN].max() >= VALID_START:
        raise AssertionError("2024 SCADA entered hourly auxiliary targets")
    scada_targets = auxiliary_target_frame(hourly_scada, target_columns=AUX_TARGETS)

    aux_started = time.perf_counter()
    aux_train, aux_valid, aux_full_model, aux_quality = _make_aux_features(
        args,
        weather_train=weather_train,
        weather_valid=weather_valid,
        scada_train=scada_targets,
    )
    aux_seconds = time.perf_counter() - aux_started

    full_predictions = pd.DataFrame(index=valid_index)
    full_models: dict[str, LGBMRegressor] = {}
    full_seconds: dict[str, float] = {}
    candidate_specs = {
        "direct_l1": ("lgb_l1", False),
        "direct_q07": ("lgb_q07", False),
        "direct_l1_wind_aux": ("lgb_l1", True),
        "direct_q07_wind_aux": ("lgb_q07", True),
    }
    for name, (objective, use_aux) in candidate_specs.items():
        prediction, model, seconds = _fit_predict_direct(
            args,
            recipe,
            objective_candidate=objective,
            weather_train=weather_train,
            target_train=target_train,
            weather_valid=weather_valid,
            aux_train=aux_train if use_aux else None,
            aux_valid=aux_valid if use_aux else None,
        )
        full_predictions[name] = prediction
        full_models[name] = model
        full_seconds[name] = seconds

    # Historical direction check: the H2 encoder is fitted on H1 SCADA only.
    hist_train_index = _exact_index(HIST_TRAIN_START, HIST_TRAIN_END)
    hist_valid_index = _exact_index(HIST_VALID_START, HIST_VALID_END)
    hist_weather_train = weather.reindex(hist_train_index)
    hist_weather_valid = weather.reindex(hist_valid_index)
    hist_scada = scada_targets.loc[
        scada_targets[TIME_COLUMN].isin(hist_train_index)
    ].copy()
    if hist_scada[TIME_COLUMN].max() >= HIST_VALID_START:
        raise AssertionError("2023 H2 SCADA entered historical auxiliary fit")
    hist_aux_train, hist_aux_valid, hist_aux_model, hist_aux_quality = _make_aux_features(
        args,
        weather_train=hist_weather_train,
        weather_valid=hist_weather_valid,
        scada_train=hist_scada,
    )
    historical_predictions = pd.DataFrame(index=hist_valid_index)
    historical_models: dict[str, LGBMRegressor] = {}
    historical_seconds: dict[str, float] = {}
    for name, (objective, use_aux) in candidate_specs.items():
        prediction, model, seconds = _fit_predict_direct(
            args,
            recipe,
            objective_candidate=objective,
            weather_train=hist_weather_train,
            target_train=labels.reindex(hist_train_index),
            weather_valid=hist_weather_valid,
            aux_train=hist_aux_train if use_aux else None,
            aux_valid=hist_aux_valid if use_aux else None,
        )
        historical_predictions[name] = prediction
        historical_models[name] = model
        historical_seconds[name] = seconds

    # Fixed gate artifacts are comparison references, not fitting inputs.
    reference_predictions: dict[str, pd.Series] = {}
    for name, path in reference_paths.items():
        frame = pd.read_parquet(path)
        if not frame.index.equals(valid_index) or GROUP not in frame:
            raise ValueError(f"invalid gate reference {path}")
        reference_predictions[name] = frame[GROUP]

    scores: dict[str, Any] = {}
    for name in full_predictions:
        scores[name] = _score_slices(
            target_valid,
            full_predictions[name],
            split_point=pd.Timestamp("2024-07-01 01:00:00"),
        )
    for name, prediction in reference_predictions.items():
        scores[name] = _score_slices(
            target_valid,
            prediction,
            split_point=pd.Timestamp("2024-07-01 01:00:00"),
        )
    historical_scores = {
        name: _score_slices(
            labels.reindex(hist_valid_index),
            historical_predictions[name],
            split_point=pd.Timestamp("2023-10-01 01:00:00"),
        )
        for name in historical_predictions
    }

    stability: dict[str, Any] = {}
    for objective in ("l1", "q07"):
        baseline = f"direct_{objective}"
        auxiliary = f"direct_{objective}_wind_aux"
        deltas = {
            "2024_full": (
                scores[auxiliary]["full"]["group_score"]
                - scores[baseline]["full"]["group_score"]
            ),
            "2024_first_half": (
                scores[auxiliary]["first_half"]["group_score"]
                - scores[baseline]["first_half"]["group_score"]
            ),
            "2024_second_half": (
                scores[auxiliary]["second_half"]["group_score"]
                - scores[baseline]["second_half"]["group_score"]
            ),
            "2023_h1_to_h2_full": (
                historical_scores[auxiliary]["full"]["group_score"]
                - historical_scores[baseline]["full"]["group_score"]
            ),
            "2023_h1_to_h2_first_half": (
                historical_scores[auxiliary]["first_half"]["group_score"]
                - historical_scores[baseline]["first_half"]["group_score"]
            ),
            "2023_h1_to_h2_second_half": (
                historical_scores[auxiliary]["second_half"]["group_score"]
                - historical_scores[baseline]["second_half"]["group_score"]
            ),
        }
        stable_positive = all(value > 0.0 for value in deltas.values())
        meaningful = deltas["2024_full"] >= 0.001
        stability[auxiliary] = {
            "deltas_vs_same_objective_baseline": deltas,
            "stable_positive_all_periods": stable_positive,
            "meaningful_full_year_gain_at_least_0.001": meaningful,
            "adoption_recommended": bool(stable_positive and meaningful),
        }

    adoption = [
        name
        for name, result in stability.items()
        if result["adoption_recommended"]
    ]
    report = {
        "experiment": "g3_scada_wind_aux_post_gate_development_oof",
        "status": "post_gate_development_not_untouched_gate",
        "selection_warning": (
            "2024 was already consumed; these results must not be described as a new "
            "untouched gate. Adoption additionally requires both 2024 halves and the "
            "2023 H1->H2 check to improve."
        ),
        "leakage_audit": {
            "raw_scada_cutoff_exclusive": str(RAW_SCADA_CUTOFF),
            "retained_raw_scada_max": str(pd.to_datetime(raw_scada["kst_dtm"]).max()),
            "aggregated_scada_max": str(hourly_scada[TIME_COLUMN].max()),
            "2024_scada_used_for_feature_or_fit": False,
            "main_aux_train": "2023 calendar-month-block OOF",
            "main_aux_valid": "2023 full fit -> 2024 weather-only transform",
            "historical_aux_train": "2023 H1 calendar-month-block OOF",
            "historical_aux_valid": "2023 H1 full fit -> H2 weather-only transform",
        },
        "boundaries": {
            "main_train": [str(TRAIN_START), str(TRAIN_END)],
            "main_valid": [str(VALID_START), str(VALID_END)],
            "historical_train": [str(HIST_TRAIN_START), str(HIST_TRAIN_END)],
            "historical_valid": [str(HIST_VALID_START), str(HIST_VALID_END)],
        },
        "parameters": {
            "aux_targets": list(AUX_TARGETS),
            "aux_weather_features": len(_aux_weather_columns(weather.columns)),
            "aux_estimators": args.aux_estimators,
            "main_recipe": recipe["recipe_name"],
            "seed": args.seed,
        },
        "aux_oof_quality_2023": aux_quality,
        "aux_oof_quality_2023_h1": hist_aux_quality,
        "scores_2024": scores,
        "scores_2023_h1_to_h2": historical_scores,
        "stability": stability,
        "adoption_candidates": adoption,
        "adoption_recommended": bool(adoption),
        "runtime_seconds": {
            "aux_main": aux_seconds,
            "full_candidates": full_seconds,
            "historical_candidates": historical_seconds,
            "total": time.perf_counter() - started,
        },
    }

    paths = {
        "report": args.out_dir / "results.json",
        "manifest": args.out_dir / "manifest.json",
        "scada": args.out_dir / "cache" / "scada_2023_train_only.parquet",
        "aux_train": args.out_dir / "cache" / "aux_2023_month_oof.parquet",
        "aux_valid": args.out_dir / "cache" / "aux_2024_weather_only.parquet",
        "predictions": args.out_dir / "oof" / "g3_2024_candidates.parquet",
        "historical_predictions": args.out_dir
        / "oof"
        / "g3_2023_h1_to_h2_candidates.parquet",
        "models": args.out_dir / "models" / "models.joblib",
    }
    _atomic_parquet(hourly_scada.set_index(TIME_COLUMN), paths["scada"])
    _atomic_parquet(aux_train, paths["aux_train"])
    _atomic_parquet(aux_valid, paths["aux_valid"])
    _atomic_parquet(full_predictions, paths["predictions"])
    _atomic_parquet(historical_predictions, paths["historical_predictions"])
    _atomic_joblib(
        {
            "artifact_type": "g3_scada_aux_post_gate_models",
            "main_models": full_models,
            "main_aux_encoder_2023": aux_full_model,
            "historical_models": historical_models,
            "historical_aux_encoder_2023_h1": hist_aux_model,
            "aux_targets": AUX_TARGETS,
            "recipe_sha256": sha256_file(args.config),
        },
        paths["models"],
    )
    write_json_atomic(paths["report"], report, overwrite=bool(args.overwrite))

    output_files = [path for key, path in paths.items() if key != "manifest"]
    manifest = make_manifest(
        artifact_type="baram_g3_scada_aux_post_gate_development",
        parameters={
            "experiment_status": report["status"],
            "boundaries": report["boundaries"],
            "leakage_audit": report["leakage_audit"],
            "recipe": recipe,
            "aux_targets": list(AUX_TARGETS),
        },
        input_files=required,
        output_files=output_files,
        results={
            "scores_2024": scores,
            "scores_2023_h1_to_h2": historical_scores,
            "stability": stability,
            "adoption_recommended": bool(adoption),
            "prediction_sha256": sha256_file(paths["predictions"]),
        },
        project_dir=PROJECT_DIR,
    )
    write_json_atomic(paths["manifest"], manifest, overwrite=bool(args.overwrite))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report={paths['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Leakage-audited SCADA auxiliary-feature experiment for the 2023 dev fold.

Outer split
-----------
* generation model train: 2022, including the interval ending 2023-01-01 00:00
* untouched validation: 2023-01-01 01:00 through 2024-01-01 00:00
* evaluated groups: KPX 1 and 2 (VESTAS)

Only pre-2023 raw SCADA rows are retained before hourly aggregation.  Auxiliary
features for the 2022 generation-model train rows are quarter-block OOF: the
SCADA encoder fitting each row never sees SCADA targets from that row's quarter.
Validation features are predictions of a fresh encoder fitted on 2022 only;
2023 SCADA is neither aggregated nor passed to any fit/transform call.

Run from the repository root::

    .venv/Scripts/python scripts/run_scada_aux_dev2023.py
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import joblib
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.metric import CAPACITY_KWH, score_details
from src.scada import (
    GROUP_COLUMN,
    SCADAAuxiliaryRegressor,
    TIME_COLUMN,
    aggregate_scada_hourly,
    auxiliary_target_frame,
    cross_fit_scada_auxiliary,
)


GROUPS = (1, 2)
TARGETS = tuple(f"kpx_group_{group}" for group in GROUPS)
TRAIN_START = pd.Timestamp("2022-01-01 01:00:00")
TRAIN_END_EXCLUSIVE = pd.Timestamp("2023-01-01 00:00:01")
VALID_START = pd.Timestamp("2023-01-01 00:00:01")
VALID_END_EXCLUSIVE = pd.Timestamp("2024-01-01 00:00:01")
RAW_SCADA_CUTOFF = pd.Timestamp("2023-01-01 00:00:00")

# Wind mean and spread capture forecast calibration/turbulence.  Circular
# direction avoids 0/360 discontinuity.  Availability is variable in 2022 and
# is tested separately so it cannot silently degrade the wind-only candidate.
AUX_TARGETS = (
    "obs_ws_mean",
    "obs_ws_std",
    "obs_wd_sin",
    "obs_wd_cos",
    "obs_turbine_availability",
)
WIND_AUX_TARGETS = AUX_TARGETS[:-1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"data/local/open"),
    )
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--n-estimators", type=int, default=1500)
    parser.add_argument("--aux-estimators", type=int, default=260)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=7)
    return parser.parse_args()


def _read_vestas_before(path: Path, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Read chronologically and discard a chunk before it can enter aggregation."""

    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, encoding="utf-8-sig", chunksize=20_000):
        timestamp = pd.to_datetime(chunk["kst_dtm"], errors="raise")
        keep = timestamp < cutoff
        if keep.any():
            retained = chunk.loc[keep].copy()
            retained["kst_dtm"] = timestamp.loc[keep]
            chunks.append(retained)
        if (~keep).any():
            break
    if not chunks:
        raise ValueError("No pre-cutoff VESTAS rows were found")
    result = pd.concat(chunks, ignore_index=True)
    if pd.to_datetime(result["kst_dtm"]).max() >= cutoff:
        raise AssertionError("Validation-period SCADA crossed the hard cutoff")
    return result


def _aux_weather_columns(columns: pd.Index) -> list[str]:
    """Small weather-only input set for the auxiliary encoder."""

    direct_tokens = (
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
        if column.startswith("time__")
        or any(token in column for token in direct_tokens)
    ]
    if not selected:
        raise ValueError("No auxiliary weather columns matched the cache schema")
    return selected


def _with_aux_keys(frame: pd.DataFrame, group: int) -> pd.DataFrame:
    result = frame.copy()
    result.insert(0, TIME_COLUMN, result.index)
    result.insert(1, GROUP_COLUMN, group)
    return result.reset_index(drop=True)


def _quarter_splits(index: pd.DatetimeIndex) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Return four non-overlapping, contiguous quarter holdouts."""

    quarter = ((index.month.to_numpy() - 1) // 3).astype(np.int8)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for block in range(4):
        valid = np.flatnonzero(quarter == block)
        train = np.flatnonzero(quarter != block)
        if len(valid) == 0 or len(train) == 0:
            raise AssertionError(f"empty quarter block {block}")
        if np.intersect1d(train, valid).size:
            raise AssertionError("SCADA encoder train/OOF block overlap")
        splits.append((train, valid))
    return splits, quarter


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
    )


def _main_estimator(args: argparse.Namespace) -> LGBMRegressor:
    # Exact parameters of the locked dev2023_lgb_l1_eligible_1500 baseline.
    return LGBMRegressor(
        objective="regression_l1",
        n_estimators=args.n_estimators,
        learning_rate=0.025,
        num_leaves=31,
        min_child_samples=30,
        colsample_bytree=0.75,
        subsample=0.80,
        subsample_freq=1,
        reg_alpha=0.05,
        reg_lambda=2.0,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        verbosity=-1,
    )


def _score(answer: pd.DataFrame, prediction: pd.DataFrame) -> dict[str, Any]:
    details = score_details(
        answer,
        prediction,
        target_cols=TARGETS,
        capacities=CAPACITY_KWH,
    )
    return asdict(details)


def _fit_generation_candidate(
    args: argparse.Namespace,
    weather_train: dict[int, pd.DataFrame],
    weather_valid: dict[int, pd.DataFrame],
    labels: pd.DataFrame,
    aux_train: dict[int, pd.DataFrame] | None,
    aux_valid: dict[int, pd.DataFrame] | None,
) -> tuple[pd.DataFrame, dict[str, LGBMRegressor], float]:
    prediction = pd.DataFrame(index=next(iter(weather_valid.values())).index)
    models: dict[str, LGBMRegressor] = {}
    started = time.perf_counter()
    for group in GROUPS:
        target = f"kpx_group_{group}"
        train_x = weather_train[group].copy()
        valid_x = weather_valid[group].copy()
        if aux_train is not None and aux_valid is not None:
            train_x = pd.concat([train_x, aux_train[group]], axis=1)
            valid_x = pd.concat([valid_x, aux_valid[group]], axis=1)
        if not train_x.columns.equals(valid_x.columns):
            raise AssertionError(f"generation feature mismatch for group {group}")
        if train_x.isna().any().any() or valid_x.isna().any().any():
            raise AssertionError(f"non-finite generation features for group {group}")

        target_y = labels[target].reindex(train_x.index)
        eligible = target_y.notna() & (target_y >= CAPACITY_KWH[target] * 0.10)
        # The locked baseline learns capacity fraction and scales back to kWh.
        # This matters because LightGBM regularisation is not perfectly invariant
        # to target scale; using raw kWh would not be an identical control.
        capacity = CAPACITY_KWH[target]
        model = _main_estimator(args)
        model.fit(train_x.loc[eligible], target_y.loc[eligible] / capacity)
        prediction[target] = model.predict(valid_x) * capacity
        models[target] = model
    elapsed = time.perf_counter() - started
    return prediction, models, elapsed


def main() -> None:
    args = _parse_args()
    started_total = time.perf_counter()
    cache_dir = args.artifact_root / "cache"
    oof_dir = args.artifact_root / "oof"
    model_dir = args.artifact_root / "models"
    log_dir = args.artifact_root / "logs"
    for directory in (cache_dir, oof_dir, model_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(
        args.data_root / "train" / "train_labels.csv",
        encoding="utf-8-sig",
        parse_dates=["kst_dtm"],
    ).set_index("kst_dtm").sort_index()

    full_weather: dict[int, pd.DataFrame] = {}
    weather_train: dict[int, pd.DataFrame] = {}
    weather_valid: dict[int, pd.DataFrame] = {}
    for group in GROUPS:
        frame = pd.read_parquet(cache_dir / f"kpx_group_{group}_weather_train.parquet")
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise TypeError("weather cache must have a DatetimeIndex")
        full_weather[group] = frame
        train_mask = (frame.index >= TRAIN_START) & (frame.index < TRAIN_END_EXCLUSIVE)
        valid_mask = (frame.index >= VALID_START) & (frame.index < VALID_END_EXCLUSIVE)
        weather_train[group] = frame.loc[train_mask].copy()
        weather_valid[group] = frame.loc[valid_mask].copy()
        if weather_train[group].index.max() >= weather_valid[group].index.min():
            raise AssertionError("outer train/validation chronology is not strict")
        if len(weather_train[group]) != 8_760 or len(weather_valid[group]) != 8_760:
            raise AssertionError("expected complete 8760-hour outer train/validation years")

    raw_vestas = _read_vestas_before(
        args.data_root / "train" / "scada_vestas_train.csv",
        RAW_SCADA_CUTOFF,
    )
    hourly_scada = aggregate_scada_hourly(raw_vestas, "vestas")
    allowed_times = weather_train[1].index
    hourly_scada = hourly_scada.loc[
        hourly_scada[TIME_COLUMN].isin(allowed_times)
        & hourly_scada[GROUP_COLUMN].isin(GROUPS)
    ].copy()
    if hourly_scada[TIME_COLUMN].max() >= weather_valid[1].index.min():
        raise AssertionError("Validation SCADA entered the training-only hourly table")
    hourly_scada.to_parquet(cache_dir / "dev2023_scada_hourly_train_only.parquet", index=False)
    scada_targets = auxiliary_target_frame(hourly_scada, target_columns=AUX_TARGETS)

    aux_train: dict[int, pd.DataFrame] = {}
    aux_valid: dict[int, pd.DataFrame] = {}
    aux_models: dict[int, SCADAAuxiliaryRegressor] = {}
    aux_audit_rows: list[pd.DataFrame] = []
    aux_quality: dict[str, dict[str, float]] = {}
    aux_started = time.perf_counter()

    for group in GROUPS:
        train_base = weather_train[group]
        valid_base = weather_valid[group]
        aux_columns = _aux_weather_columns(train_base.columns)
        train_input = _with_aux_keys(train_base.loc[:, aux_columns], group)
        valid_input = _with_aux_keys(valid_base.loc[:, aux_columns], group)
        splits, quarter = _quarter_splits(train_base.index)
        model_kwargs = {
            "feature_columns": aux_columns,
            "target_columns": AUX_TARGETS,
            "group": group,
            "estimator": _aux_estimator(args),
            "min_samples": 168,
            "random_state": args.seed,
        }
        train_prediction = cross_fit_scada_auxiliary(
            train_input,
            scada_targets,
            splits,
            model_kwargs=model_kwargs,
        )
        if train_prediction.isna().any().any():
            raise AssertionError(f"incomplete auxiliary OOF for group {group}")
        train_prediction.index = train_base.index
        train_prediction = train_prediction.astype("float32")

        full_model = SCADAAuxiliaryRegressor(**model_kwargs)
        full_model.fit(train_input, scada_targets)
        valid_prediction = full_model.predict_auxiliary(valid_input)
        valid_prediction.index = valid_base.index
        valid_prediction = valid_prediction.astype("float32")
        if valid_prediction.isna().any().any():
            raise AssertionError(f"incomplete auxiliary validation prediction for group {group}")

        aux_train[group] = train_prediction
        aux_valid[group] = valid_prediction
        aux_models[group] = full_model

        actual = (
            scada_targets.loc[scada_targets[GROUP_COLUMN] == group]
            .set_index(TIME_COLUMN)
            .reindex(train_base.index)
        )
        group_quality: dict[str, float] = {}
        for target in AUX_TARGETS:
            predicted_column = f"pred_scada_{target}"
            valid_target = actual[target].notna()
            group_quality[f"{target}__mae"] = float(
                np.mean(
                    np.abs(
                        actual.loc[valid_target, target].to_numpy()
                        - train_prediction.loc[valid_target, predicted_column].to_numpy()
                    )
                )
            )
            group_quality[f"{target}__correlation"] = float(
                np.corrcoef(
                    actual.loc[valid_target, target].to_numpy(),
                    train_prediction.loc[valid_target, predicted_column].to_numpy(),
                )[0, 1]
            )
        aux_quality[f"kpx_group_{group}"] = group_quality

        audit = train_prediction.reset_index(names=TIME_COLUMN)
        audit.insert(1, GROUP_COLUMN, group)
        audit.insert(2, "aux_oof_quarter", quarter)
        aux_audit_rows.append(audit)
        valid_audit = valid_prediction.reset_index(names=TIME_COLUMN)
        valid_audit.insert(1, GROUP_COLUMN, group)
        valid_audit.to_parquet(
            cache_dir / f"dev2023_scada_aux_valid_predictions_g{group}.parquet",
            index=False,
        )

    aux_elapsed = time.perf_counter() - aux_started
    pd.concat(aux_audit_rows, ignore_index=True).to_parquet(
        cache_dir / "dev2023_scada_aux_train_quarter_oof.parquet", index=False
    )

    valid_answer = labels.loc[weather_valid[1].index, list(TARGETS)]
    candidates: dict[str, dict[str, Any]] = {}
    model_bundle: dict[str, Any] = {"aux_full_2022": aux_models}

    baseline_prediction, baseline_models, baseline_elapsed = _fit_generation_candidate(
        args,
        weather_train,
        weather_valid,
        labels,
        aux_train=None,
        aux_valid=None,
    )
    candidates["baseline"] = {
        "metrics": _score(valid_answer, baseline_prediction),
        "fit_seconds": baseline_elapsed,
        "n_features": int(weather_train[1].shape[1]),
    }
    model_bundle["baseline"] = baseline_models

    subsets = {
        "wind_aux": [f"pred_scada_{target}" for target in WIND_AUX_TARGETS],
        "wind_availability_aux": [f"pred_scada_{target}" for target in AUX_TARGETS],
    }
    predictions = {"baseline": baseline_prediction}
    for candidate, columns in subsets.items():
        candidate_train = {group: aux_train[group].loc[:, columns] for group in GROUPS}
        candidate_valid = {group: aux_valid[group].loc[:, columns] for group in GROUPS}
        prediction, models, elapsed = _fit_generation_candidate(
            args,
            weather_train,
            weather_valid,
            labels,
            aux_train=candidate_train,
            aux_valid=candidate_valid,
        )
        predictions[candidate] = prediction
        metrics = _score(valid_answer, prediction)
        candidates[candidate] = {
            "metrics": metrics,
            "fit_seconds": elapsed,
            "n_features": int(weather_train[1].shape[1] + len(columns)),
            "aux_columns": columns,
            "delta_total_vs_baseline": (
                metrics["total_score"]
                - candidates["baseline"]["metrics"]["total_score"]
            ),
            "delta_1_minus_nmae_vs_baseline": (
                metrics["one_minus_nmae"]
                - candidates["baseline"]["metrics"]["one_minus_nmae"]
            ),
            "delta_ficr_vs_baseline": (
                metrics["ficr"] - candidates["baseline"]["metrics"]["ficr"]
            ),
        }
        model_bundle[candidate] = models

    for candidate, prediction in predictions.items():
        prediction.to_parquet(
            oof_dir / f"dev2023_lgb_l1_{candidate}_n{args.n_estimators}.parquet"
        )
    joblib.dump(
        model_bundle,
        model_dir / f"dev2023_scada_aux_experiment_n{args.n_estimators}.joblib",
        compress=3,
    )

    report = {
        "experiment": "dev2023_scada_aux_quarter_oof",
        "created_unix_time": time.time(),
        "python": platform.python_version(),
        "outer_split": {
            "train_start": str(TRAIN_START),
            "train_end_exclusive": str(TRAIN_END_EXCLUSIVE),
            "valid_start": str(VALID_START),
            "valid_end_exclusive": str(VALID_END_EXCLUSIVE),
            "groups": list(GROUPS),
        },
        "leakage_audit": {
            "raw_scada_hard_cutoff_exclusive": str(RAW_SCADA_CUTOFF),
            "retained_raw_scada_max": str(pd.to_datetime(raw_vestas["kst_dtm"]).max()),
            "aggregated_scada_max": str(hourly_scada[TIME_COLUMN].max()),
            "valid_scada_used": False,
            "aux_train_method": "4 contiguous calendar-quarter holdouts; held quarter excluded",
            "aux_valid_method": "encoder fit on all retained 2022 rows, weather-only transform",
        },
        "parameters": {
            "main_n_estimators": args.n_estimators,
            "aux_n_estimators": args.aux_estimators,
            "seed": args.seed,
            "n_jobs": args.n_jobs,
            "aux_targets": list(AUX_TARGETS),
            "aux_weather_feature_count": len(_aux_weather_columns(weather_train[1].columns)),
        },
        "rows": {
            "generation_train": len(weather_train[1]),
            "generation_valid": len(weather_valid[1]),
            "hourly_scada_train_only": len(hourly_scada),
        },
        "aux_oof_quality_train_only": aux_quality,
        "aux_fit_and_predict_seconds": aux_elapsed,
        "candidates": candidates,
        "total_seconds": time.perf_counter() - started_total,
    }
    log_path = log_dir / f"dev2023_scada_aux_n{args.n_estimators}.json"
    log_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report={log_path}")


if __name__ == "__main__":
    main()

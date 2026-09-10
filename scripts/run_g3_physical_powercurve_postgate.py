"""Strict-forward group-3 turbine power-curve experiment (post-gate).

Unlike the earlier stacking experiment, SCADA predictions are not appended to
another tree model.  Fold-training SCADA fits five turbine-specific monotonic
wind-speed curves.  A weather->SCADA wind mapper supplies validation wind, then
the five 4.2 MW turbine estimates are summed to group kWh.

Both evaluations are forward in time:

* 2023 H1 train -> 2023 H2 validation;
* all 2023 train -> all 2024 validation (already-consumed development gate).

Within each training period, physical OOF predictions use calendar-month
holdouts.  Each holdout refits both the wind mapper and turbine curves without
that month's SCADA.  Current-validation SCADA is never read into a fit/feature.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
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
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import HuberRegressor


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.run_g3_scada_aux_postgate import (  # noqa: E402
    AUX_TARGETS,
    CAPACITY,
    GROUP,
    GROUP_NUMBER,
    HIST_TRAIN_END,
    HIST_TRAIN_START,
    HIST_VALID_END,
    HIST_VALID_START,
    RAW_SCADA_CUTOFF,
    TRAIN_END,
    TRAIN_START,
    VALID_END,
    VALID_START,
    _atomic_joblib,
    _atomic_parquet,
    _aux_estimator,
    _aux_weather_columns,
    _exact_index,
    _month_block_splits,
    _read_unison_before,
    _score_slices,
    _with_aux_keys,
)
from src.final_training import read_recipe  # noqa: E402
from src.manifest import make_manifest, sha256_file, write_json_atomic  # noqa: E402
from src.metric import CAPACITY_KWH  # noqa: E402
from src.scada import (  # noqa: E402
    GROUP_COLUMN,
    SCADAAuxiliaryRegressor,
    TIME_COLUMN,
    aggregate_scada_hourly,
    auxiliary_target_frame,
)


TURBINES = tuple(range(1, 6))
TURBINE_CAPACITY_KW = 4_200.0
RATED_KWH10M = TURBINE_CAPACITY_KW / 6.0
PHYSICAL_POWER_MAX_KWH10M = RATED_KWH10M * 1.05
CUT_IN_MS = 2.5
CUT_OUT_MS = 25.0
REFERENCE_DENSITY = 1.225
BLEND_WEIGHTS = (0.10, 0.20, 0.30)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"data/local/open"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/cache"))
    parser.add_argument("--gate-dir", type=Path, default=Path("artifacts/gate/v3"))
    parser.add_argument(
        "--config", type=Path, default=Path("configs/train_final.v3.locked.json")
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/postgate/g3_physical_powercurve"),
    )
    parser.add_argument("--aux-estimators", type=int, default=260)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _density(weather: pd.DataFrame) -> pd.Series:
    columns = [
        column
        for column in ("ldaps__idw__air_density", "gfs__idw__air_density")
        if column in weather.columns
    ]
    if not columns:
        return pd.Series(REFERENCE_DENSITY, index=weather.index, dtype=float)
    value = weather.loc[:, columns].mean(axis=1).astype(float)
    plausible = value.between(0.7, 1.5)
    fallback = float(value.loc[plausible].median())
    return value.where(plausible, fallback)


def _circular_mean_degrees(values: np.ndarray, axis: int = 1) -> np.ndarray:
    radians = np.deg2rad(np.mod(values, 360.0))
    valid = np.isfinite(radians)
    count = valid.sum(axis=axis)
    sine = np.divide(
        np.where(valid, np.sin(radians), 0.0).sum(axis=axis),
        count,
        out=np.full(count.shape, np.nan, dtype=float),
        where=count > 0,
    )
    cosine = np.divide(
        np.where(valid, np.cos(radians), 0.0).sum(axis=axis),
        count,
        out=np.full(count.shape, np.nan, dtype=float),
        where=count > 0,
    )
    return np.mod(np.rad2deg(np.arctan2(sine, cosine)), 360.0)


def _circular_difference(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return (left - right + 180.0) % 360.0 - 180.0


def _target_time(frame: pd.DataFrame) -> pd.Series:
    source = pd.to_datetime(frame["kst_dtm"], errors="raise")
    return source.dt.floor("h") + pd.Timedelta(hours=1)


@dataclass
class TurbineCurveState:
    curve: IsotonicRegression
    wind_bias_ms: float
    direction_bias_deg: float
    availability_global: float
    availability_cells: dict[tuple[int, int], float]
    curve_samples: int
    availability_samples: int


class TurbinePhysicalFleet:
    """Five UNISON monotonic curves plus train-only availability lookups."""

    def __init__(
        self,
        *,
        use_air_density: bool = True,
        availability_shrinkage: float = 120.0,
        wind_bin_width: float = 0.25,
    ) -> None:
        self.use_air_density = bool(use_air_density)
        self.availability_shrinkage = float(availability_shrinkage)
        self.wind_bin_width = float(wind_bin_width)

    @staticmethod
    def _columns(measurement: str) -> list[str]:
        return [f"unison_wtg{number:02d}_{measurement}" for number in TURBINES]

    def fit(
        self,
        raw_scada: pd.DataFrame,
        weather: pd.DataFrame,
    ) -> "TurbinePhysicalFleet":
        required = [
            "kst_dtm",
            *self._columns("power_kw10m"),
            *self._columns("ws"),
            *self._columns("wd"),
        ]
        missing = [column for column in required if column not in raw_scada]
        if missing:
            raise ValueError(f"UNISON SCADA columns missing: {missing}")
        frame = raw_scada.copy()
        frame[TIME_COLUMN] = _target_time(frame)
        frame = frame.loc[frame[TIME_COLUMN].isin(weather.index)].copy()
        if frame.empty:
            raise ValueError("no SCADA rows overlap physical-curve weather")

        power = frame.loc[:, self._columns("power_kw10m")].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float, copy=True)
        ws = frame.loc[:, self._columns("ws")].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float, copy=True)
        wd = frame.loc[:, self._columns("wd")].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float, copy=True)
        physical_power = np.isfinite(power) & (power >= 0.0) & (
            power <= PHYSICAL_POWER_MAX_KWH10M
        )
        self.rejected_power_count_ = int((np.isfinite(power) & ~physical_power).sum())
        power[~physical_power] = np.nan
        ws[~(np.isfinite(ws) & (ws >= 0.0) & (ws <= 50.0))] = np.nan
        wd[~np.isfinite(wd)] = np.nan
        wd = np.mod(wd, 360.0)

        ws_count = np.isfinite(ws).sum(axis=1)
        group_ws = np.divide(
            np.nansum(ws, axis=1),
            ws_count,
            out=np.full(len(ws), np.nan, dtype=float),
            where=ws_count > 0,
        )
        group_wd = _circular_mean_degrees(wd)
        density = _density(weather).reindex(frame[TIME_COLUMN]).to_numpy(dtype=float)
        density_factor = (
            np.cbrt(np.clip(density, 0.7, 1.5) / REFERENCE_DENSITY)
            if self.use_air_density
            else np.ones(len(frame), dtype=float)
        )
        month = frame[TIME_COLUMN].dt.month.to_numpy(dtype=int)

        self.states_: dict[int, TurbineCurveState] = {}
        for position, turbine in enumerate(TURBINES):
            turbine_ws = ws[:, position]
            turbine_wd = wd[:, position]
            normalized_power = power[:, position] / RATED_KWH10M
            wind_bias = float(np.nanmedian(turbine_ws - group_ws))
            differences = _circular_difference(turbine_wd, group_wd)
            direction_bias = float(np.nanmedian(differences[np.isfinite(differences)]))
            effective_ws = turbine_ws * density_factor

            curve_mask = (
                np.isfinite(effective_ws)
                & np.isfinite(normalized_power)
                & (effective_ws >= CUT_IN_MS)
                & (effective_ws < CUT_OUT_MS)
                & (normalized_power > 0.002)
            )
            if int(curve_mask.sum()) < 500:
                raise ValueError(f"turbine {turbine} has too few curve samples")
            x = effective_ws[curve_mask]
            y = np.clip(normalized_power[curve_mask], 0.0, 1.05)
            bins = np.floor(x / self.wind_bin_width).astype(int)
            binned = pd.DataFrame({"x": x, "y": y, "bin": bins}).groupby(
                "bin", sort=True
            ).agg(x=("x", "median"), y=("y", "median"), count=("y", "size"))
            # Physical anchors are deliberately low-weight relative to data bins.
            anchor_x = np.asarray([0.0, CUT_IN_MS, 12.0, 20.0, CUT_OUT_MS - 0.01])
            anchor_y = np.asarray([0.0, 0.0, 1.0, 1.0, 1.0])
            fit_x = np.concatenate([anchor_x, binned["x"].to_numpy(dtype=float)])
            fit_y = np.concatenate([anchor_y, binned["y"].to_numpy(dtype=float)])
            fit_weight = np.concatenate(
                [np.full(len(anchor_x), 12.0), binned["count"].to_numpy(dtype=float)]
            )
            order = np.argsort(fit_x, kind="stable")
            curve = IsotonicRegression(
                increasing=True,
                y_min=0.0,
                y_max=1.03,
                out_of_bounds="clip",
            ).fit(fit_x[order], fit_y[order], sample_weight=fit_weight[order])

            potential = np.full(len(effective_ws), np.nan, dtype=float)
            finite_effective = np.isfinite(effective_ws)
            potential[finite_effective] = curve.predict(
                np.clip(
                    effective_ws[finite_effective],
                    0.0,
                    CUT_OUT_MS - 0.01,
                )
            )
            potential[
                finite_effective
                & ((effective_ws < CUT_IN_MS) | (effective_ws >= CUT_OUT_MS))
            ] = 0.0
            availability_mask = (
                np.isfinite(normalized_power)
                & np.isfinite(potential)
                & (potential >= 0.05)
                & np.isfinite(turbine_wd)
            )
            ratio = np.clip(
                normalized_power[availability_mask] / potential[availability_mask],
                0.0,
                1.0,
            )
            global_availability = float(np.mean(ratio))
            sector = np.floor(np.mod(turbine_wd[availability_mask], 360.0) / 45.0).astype(int)
            lookup = pd.DataFrame(
                {
                    "month": month[availability_mask],
                    "sector": sector,
                    "ratio": ratio,
                }
            ).groupby(["month", "sector"], sort=True)["ratio"].agg(["mean", "count"])
            cells: dict[tuple[int, int], float] = {}
            for key, row in lookup.iterrows():
                weight = float(row["count"]) / (
                    float(row["count"]) + self.availability_shrinkage
                )
                cells[(int(key[0]), int(key[1]))] = float(
                    weight * float(row["mean"]) + (1.0 - weight) * global_availability
                )
            self.states_[turbine] = TurbineCurveState(
                curve=curve,
                wind_bias_ms=wind_bias,
                direction_bias_deg=direction_bias,
                availability_global=global_availability,
                availability_cells=cells,
                curve_samples=int(curve_mask.sum()),
                availability_samples=int(availability_mask.sum()),
            )
        self.fit_start_ = frame[TIME_COLUMN].min()
        self.fit_end_ = frame[TIME_COLUMN].max()
        return self

    def predict(
        self,
        auxiliary_wind: pd.DataFrame,
        weather: pd.DataFrame,
    ) -> pd.DataFrame:
        if not hasattr(self, "states_"):
            raise RuntimeError("TurbinePhysicalFleet is not fitted")
        required = [f"pred_scada_{target}" for target in AUX_TARGETS]
        missing = [column for column in required if column not in auxiliary_wind]
        if missing:
            raise ValueError(f"auxiliary wind columns missing: {missing}")
        if not auxiliary_wind.index.equals(weather.index):
            raise ValueError("auxiliary wind/weather indexes differ")

        group_ws = auxiliary_wind["pred_scada_obs_ws_mean"].to_numpy(dtype=float)
        group_std = np.clip(
            auxiliary_wind["pred_scada_obs_ws_std"].to_numpy(dtype=float), 0.0, 8.0
        )
        sine = auxiliary_wind["pred_scada_obs_wd_sin"].to_numpy(dtype=float)
        cosine = auxiliary_wind["pred_scada_obs_wd_cos"].to_numpy(dtype=float)
        group_wd = np.mod(np.rad2deg(np.arctan2(sine, cosine)), 360.0)
        density = _density(weather).to_numpy(dtype=float)
        density_factor = (
            np.cbrt(np.clip(density, 0.7, 1.5) / REFERENCE_DENSITY)
            if self.use_air_density
            else np.ones(len(weather), dtype=float)
        )
        months = weather.index.month.to_numpy(dtype=int)

        potential_group = np.zeros(len(weather), dtype=float)
        availability_group = np.zeros(len(weather), dtype=float)
        # Three-point approximation captures power-curve convexity within hour.
        offsets = np.asarray([-1.224744871, 0.0, 1.224744871])
        weights = np.asarray([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0])
        for turbine, state in self.states_.items():
            base_ws = group_ws + state.wind_bias_ms
            speeds = base_ws[:, None] + group_std[:, None] * offsets[None, :]
            effective = np.clip(speeds, 0.0, None) * density_factor[:, None]
            cf_points = state.curve.predict(
                np.clip(effective.reshape(-1), 0.0, CUT_OUT_MS - 0.01)
            ).reshape(effective.shape)
            cf_points[(effective < CUT_IN_MS) | (effective >= CUT_OUT_MS)] = 0.0
            potential_cf = np.sum(cf_points * weights[None, :], axis=1)
            turbine_wd = np.mod(group_wd + state.direction_bias_deg, 360.0)
            sector = np.floor(turbine_wd / 45.0).astype(int)
            availability = np.asarray(
                [
                    state.availability_cells.get(
                        (int(month), int(direction_sector)),
                        state.availability_global,
                    )
                    for month, direction_sector in zip(months, sector)
                ],
                dtype=float,
            )
            potential_group += potential_cf * TURBINE_CAPACITY_KW
            availability_group += potential_cf * availability * TURBINE_CAPACITY_KW
        return pd.DataFrame(
            {
                "physical_potential_raw": np.clip(potential_group, 0.0, CAPACITY * 1.02),
                "physical_availability_raw": np.clip(
                    availability_group, 0.0, CAPACITY * 1.02
                ),
            },
            index=weather.index,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "fit_start": str(self.fit_start_),
            "fit_end": str(self.fit_end_),
            "use_air_density": self.use_air_density,
            "rejected_power_count": self.rejected_power_count_,
            "turbines": {
                str(turbine): {
                    "wind_bias_ms": state.wind_bias_ms,
                    "direction_bias_deg": state.direction_bias_deg,
                    "availability_global": state.availability_global,
                    "availability_cells": len(state.availability_cells),
                    "curve_samples": state.curve_samples,
                    "availability_samples": state.availability_samples,
                }
                for turbine, state in self.states_.items()
            },
        }


def _fit_wind_mapper(
    args: argparse.Namespace,
    weather_train: pd.DataFrame,
    scada_targets: pd.DataFrame,
) -> SCADAAuxiliaryRegressor:
    columns = _aux_weather_columns(weather_train.columns)
    model = SCADAAuxiliaryRegressor(
        feature_columns=columns,
        target_columns=AUX_TARGETS,
        group=GROUP_NUMBER,
        estimator=_aux_estimator(args),
        min_samples=168,
        random_state=args.seed,
    )
    model.fit(_with_aux_keys(weather_train.loc[:, columns]), scada_targets)
    return model


def _predict_wind(
    model: SCADAAuxiliaryRegressor,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    columns = list(model.feature_columns_)
    prediction = model.predict_auxiliary(_with_aux_keys(weather.loc[:, columns]))
    prediction.index = weather.index
    return prediction.astype("float32")


def _raw_for_times(raw_scada: pd.DataFrame, times: pd.DatetimeIndex) -> pd.DataFrame:
    target = _target_time(raw_scada)
    return raw_scada.loc[target.isin(times)].copy()


def _physical_cross_fit(
    args: argparse.Namespace,
    *,
    raw_scada: pd.DataFrame,
    hourly_targets: pd.DataFrame,
    weather_train: pd.DataFrame,
    weather_valid: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    SCADAAuxiliaryRegressor,
    TurbinePhysicalFleet,
    dict[str, Any],
]:
    oof = pd.DataFrame(
        np.nan,
        index=weather_train.index,
        columns=("physical_potential_raw", "physical_availability_raw"),
    )
    fold_audit: list[dict[str, Any]] = []
    for fold_number, (train_positions, holdout_positions) in enumerate(
        _month_block_splits(weather_train.index)
    ):
        fold_weather = weather_train.iloc[train_positions]
        holdout_weather = weather_train.iloc[holdout_positions]
        fold_times = fold_weather.index
        fold_targets = hourly_targets.loc[
            hourly_targets[TIME_COLUMN].isin(fold_times)
        ].copy()
        if fold_targets[TIME_COLUMN].max() in holdout_weather.index:
            raise AssertionError("physical fold SCADA/holdout overlap")
        mapper = _fit_wind_mapper(args, fold_weather, fold_targets)
        holdout_wind = _predict_wind(mapper, holdout_weather)
        fleet = TurbinePhysicalFleet().fit(
            _raw_for_times(raw_scada, fold_times),
            fold_weather,
        )
        prediction = fleet.predict(holdout_wind, holdout_weather)
        oof.iloc[holdout_positions, :] = prediction.to_numpy()
        fold_audit.append(
            {
                "fold": fold_number,
                "holdout_month": str(holdout_weather.index.to_period("M")[0]),
                "train_rows": len(fold_weather),
                "holdout_rows": len(holdout_weather),
                "curve": fleet.summary(),
            }
        )
    if oof.isna().any().any():
        raise AssertionError("physical OOF is incomplete")

    full_mapper = _fit_wind_mapper(args, weather_train, hourly_targets)
    valid_wind = _predict_wind(full_mapper, weather_valid)
    full_fleet = TurbinePhysicalFleet().fit(raw_scada, weather_train)
    valid_prediction = full_fleet.predict(valid_wind, weather_valid)
    return (
        oof.astype("float32"),
        valid_prediction.astype("float32"),
        full_mapper,
        full_fleet,
        {"folds": fold_audit, "full_curve": full_fleet.summary()},
    )


def _calibrate_physical(
    train_prediction: pd.DataFrame,
    target_train: pd.Series,
    valid_prediction: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, HuberRegressor], dict[str, Any]]:
    eligible = target_train.notna() & (target_train >= CAPACITY * 0.10)
    output = pd.DataFrame(index=valid_prediction.index)
    models: dict[str, HuberRegressor] = {}
    audit: dict[str, Any] = {}
    for raw_column, calibrated_column in (
        ("physical_potential_raw", "physical_potential"),
        ("physical_availability_raw", "physical_availability"),
    ):
        model = HuberRegressor(epsilon=1.5, alpha=0.001, max_iter=500)
        x = train_prediction.loc[eligible, raw_column].to_numpy(dtype=float) / CAPACITY
        y = target_train.loc[eligible].to_numpy(dtype=float) / CAPACITY
        model.fit(x.reshape(-1, 1), y)
        predicted = model.predict(
            valid_prediction[raw_column].to_numpy(dtype=float).reshape(-1, 1)
            / CAPACITY
        )
        output[calibrated_column] = np.clip(predicted * CAPACITY, 0.0, CAPACITY * 1.02)
        models[calibrated_column] = model
        audit[calibrated_column] = {
            "scale": float(model.coef_[0]),
            "bias_capacity_fraction": float(model.intercept_),
            "training_rows": int(eligible.sum()),
        }
    return output, models, audit


def _shared_q07_historical(
    args: argparse.Namespace,
    recipe: Mapping[str, Any],
    *,
    labels: pd.DataFrame,
    cache_dir: Path,
    train_end: pd.Timestamp,
    valid_index: pd.DatetimeIndex,
) -> tuple[pd.Series, LGBMRegressor, dict[str, int]]:
    specification = recipe["models"]["shared_q07"]
    params = dict(specification["params"])
    params.update(
        objective="quantile",
        alpha=float(specification["alpha"]),
        random_state=args.seed,
        n_jobs=args.n_jobs,
    )
    parts: list[pd.DataFrame] = []
    targets: list[pd.Series] = []
    rows: dict[str, int] = {}
    canonical: list[str] | None = None
    group3_valid: pd.DataFrame | None = None
    for group_number, group in enumerate(("kpx_group_1", "kpx_group_2", GROUP), start=1):
        weather = pd.read_parquet(cache_dir / f"{group}_weather_train.parquet")
        if canonical is None:
            canonical = list(weather.columns)
        elif list(weather.columns) != canonical:
            raise ValueError("shared historical cache schemas differ")
        target = labels[group].reindex(weather.index)
        mask = (
            (weather.index <= train_end)
            & target.notna()
            & (target >= CAPACITY_KWH[group] * 0.10)
        )
        part = weather.loc[mask, canonical].copy()
        part["model__group_id"] = np.float32(group_number)
        parts.append(part.reset_index(drop=True))
        targets.append(target.loc[mask].reset_index(drop=True))
        rows[group] = int(mask.sum())
        if group == GROUP:
            group3_valid = weather.reindex(valid_index).loc[:, canonical].copy()
            group3_valid["model__group_id"] = np.float32(group_number)
    assert canonical is not None and group3_valid is not None
    train_x = pd.concat(parts, ignore_index=True)
    train_y = pd.concat(targets, ignore_index=True)
    model = LGBMRegressor(**params)
    model.fit(train_x, train_y)
    prediction = pd.Series(
        model.predict(group3_valid), index=valid_index, name="direct_shared_q07"
    )
    return prediction, model, rows


def _residual_diagnostics(
    actual: pd.Series,
    direct: pd.Series,
    physical: pd.Series,
) -> dict[str, float]:
    eligible = actual.notna() & (actual >= CAPACITY * 0.10)
    y = actual.loc[eligible].to_numpy(dtype=float)
    direct_values = direct.loc[eligible].to_numpy(dtype=float)
    physical_values = physical.loc[eligible].to_numpy(dtype=float)
    direct_residual = y - direct_values
    physical_residual = y - physical_values
    return {
        "prediction_correlation": float(np.corrcoef(direct_values, physical_values)[0, 1]),
        "signed_residual_correlation": float(
            np.corrcoef(direct_residual, physical_residual)[0, 1]
        ),
        "absolute_error_correlation": float(
            np.corrcoef(np.abs(direct_residual), np.abs(physical_residual))[0, 1]
        ),
        "direct_residual_std_kwh": float(np.std(direct_residual)),
        "physical_residual_std_kwh": float(np.std(physical_residual)),
    }


def _evaluate_period(
    args: argparse.Namespace,
    recipe: Mapping[str, Any],
    *,
    name: str,
    raw_scada: pd.DataFrame,
    hourly_targets: pd.DataFrame,
    weather_train: pd.DataFrame,
    weather_valid: pd.DataFrame,
    target_train: pd.Series,
    target_valid: pd.Series,
    direct_prediction: pd.Series,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    physical_oof, physical_valid_raw, mapper, fleet, curve_audit = _physical_cross_fit(
        args,
        raw_scada=raw_scada,
        hourly_targets=hourly_targets,
        weather_train=weather_train,
        weather_valid=weather_valid,
    )
    physical_valid, calibrators, calibration_audit = _calibrate_physical(
        physical_oof,
        target_train,
        physical_valid_raw,
    )
    predictions = pd.DataFrame(index=weather_valid.index)
    predictions["direct_shared_q07"] = direct_prediction.reindex(weather_valid.index)
    predictions = pd.concat([predictions, physical_valid], axis=1)
    for physical_name in ("physical_potential", "physical_availability"):
        for weight in BLEND_WEIGHTS:
            predictions[f"blend_{physical_name}_w{int(weight * 100):02d}"] = (
                (1.0 - weight) * predictions["direct_shared_q07"]
                + weight * predictions[physical_name]
            )
    if predictions.isna().any().any():
        raise AssertionError(f"{name} candidate predictions are incomplete")
    midpoint = (
        pd.Timestamp("2024-07-01 01:00:00")
        if weather_valid.index.min().year == 2024
        else pd.Timestamp("2023-10-01 01:00:00")
    )
    scores = {
        column: _score_slices(target_valid, predictions[column], split_point=midpoint)
        for column in predictions
    }
    residual = {
        physical_name: _residual_diagnostics(
            target_valid,
            predictions["direct_shared_q07"],
            predictions[physical_name],
        )
        for physical_name in ("physical_potential", "physical_availability")
    }
    models = {
        "wind_mapper": mapper,
        "fleet": fleet,
        "calibrators": calibrators,
    }
    audit = {
        "scores": scores,
        "residual_diagnostics": residual,
        "calibration": calibration_audit,
        "curves": curve_audit,
        "physical_oof": physical_oof,
        "physical_valid_raw": physical_valid_raw,
        "predictions": predictions,
        "models": models,
    }
    compact = {
        key: value
        for key, value in audit.items()
        if key not in {"physical_oof", "physical_valid_raw", "predictions", "models"}
    }
    return compact, audit, models


def _preflight(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists():
        files = [path for path in out_dir.rglob("*") if path.is_file()]
        if files and not overwrite:
            raise FileExistsError(f"output exists: {files[0]}; pass --overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for attribute in ("data_root", "cache_dir", "gate_dir", "config", "out_dir"):
        setattr(args, attribute, getattr(args, attribute).expanduser().resolve())
    _preflight(args.out_dir, bool(args.overwrite))
    started = time.perf_counter()

    recipe = read_recipe(args.config)
    if not recipe["recipe_locked"]:
        raise ValueError("physical experiment requires a locked direct recipe")
    label_path = args.data_root / "train" / "train_labels.csv"
    scada_path = args.data_root / "train" / "scada_unison_train.csv"
    g3_weather_path = args.cache_dir / f"{GROUP}_weather_train.parquet"
    direct_gate_path = args.gate_dir / "predictions" / "shared_q07_gate.parquet"
    required = [label_path, scada_path, g3_weather_path, direct_gate_path, args.config]
    required.extend(
        args.cache_dir / f"{group}_weather_train.parquet"
        for group in ("kpx_group_1", "kpx_group_2")
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing physical experiment inputs: {missing}")

    labels = pd.read_csv(
        label_path, encoding="utf-8-sig", parse_dates=["kst_dtm"]
    ).set_index("kst_dtm").sort_index()
    weather = pd.read_parquet(g3_weather_path)
    raw_scada = _read_unison_before(scada_path, RAW_SCADA_CUTOFF)
    if pd.to_datetime(raw_scada["kst_dtm"]).max() >= RAW_SCADA_CUTOFF:
        raise AssertionError("main raw SCADA cutoff failed")
    hourly = aggregate_scada_hourly(raw_scada, "unison")
    hourly = hourly.loc[
        hourly[GROUP_COLUMN].eq(GROUP_NUMBER)
        & hourly[TIME_COLUMN].between(TRAIN_START, TRAIN_END, inclusive="both")
    ].copy()
    if hourly[TIME_COLUMN].max() >= VALID_START:
        raise AssertionError("2024 SCADA entered main physical fit")
    hourly_targets = auxiliary_target_frame(hourly, target_columns=AUX_TARGETS)

    main_train_index = _exact_index(TRAIN_START, TRAIN_END)
    main_valid_index = _exact_index(VALID_START, VALID_END)
    direct_gate = pd.read_parquet(direct_gate_path)[GROUP]
    if not direct_gate.index.equals(main_valid_index):
        raise ValueError("gate shared_q07 reference index differs")
    main_started = time.perf_counter()
    main_compact, main_audit, main_models = _evaluate_period(
        args,
        recipe,
        name="train_2023_valid_2024",
        raw_scada=raw_scada,
        hourly_targets=hourly_targets,
        weather_train=weather.reindex(main_train_index),
        weather_valid=weather.reindex(main_valid_index),
        target_train=labels[GROUP].reindex(main_train_index),
        target_valid=labels[GROUP].reindex(main_valid_index),
        direct_prediction=direct_gate,
    )
    main_seconds = time.perf_counter() - main_started

    hist_train_index = _exact_index(HIST_TRAIN_START, HIST_TRAIN_END)
    hist_valid_index = _exact_index(HIST_VALID_START, HIST_VALID_END)
    hist_raw = _raw_for_times(raw_scada, hist_train_index)
    if _target_time(hist_raw).max() >= HIST_VALID_START:
        raise AssertionError("historical H2 SCADA entered physical fit")
    hist_hourly = hourly_targets.loc[
        hourly_targets[TIME_COLUMN].isin(hist_train_index)
    ].copy()
    direct_historical, shared_historical_model, shared_rows = _shared_q07_historical(
        args,
        recipe,
        labels=labels,
        cache_dir=args.cache_dir,
        train_end=HIST_TRAIN_END,
        valid_index=hist_valid_index,
    )
    historical_started = time.perf_counter()
    hist_compact, hist_audit, hist_models = _evaluate_period(
        args,
        recipe,
        name="train_2023_h1_valid_2023_h2",
        raw_scada=hist_raw,
        hourly_targets=hist_hourly,
        weather_train=weather.reindex(hist_train_index),
        weather_valid=weather.reindex(hist_valid_index),
        target_train=labels[GROUP].reindex(hist_train_index),
        target_valid=labels[GROUP].reindex(hist_valid_index),
        direct_prediction=direct_historical,
    )
    historical_seconds = time.perf_counter() - historical_started

    candidate_names = [
        name
        for name in main_compact["scores"]
        if name != "direct_shared_q07"
    ]
    stability: dict[str, Any] = {}
    for candidate in candidate_names:
        main_scores = main_compact["scores"]
        historical_scores = hist_compact["scores"]
        deltas = {
            "2024_full": main_scores[candidate]["full"]["group_score"]
            - main_scores["direct_shared_q07"]["full"]["group_score"],
            "2024_first_half": main_scores[candidate]["first_half"]["group_score"]
            - main_scores["direct_shared_q07"]["first_half"]["group_score"],
            "2024_second_half": main_scores[candidate]["second_half"]["group_score"]
            - main_scores["direct_shared_q07"]["second_half"]["group_score"],
            "2023_h1_to_h2_full": historical_scores[candidate]["full"]["group_score"]
            - historical_scores["direct_shared_q07"]["full"]["group_score"],
            "2023_h1_to_h2_first_half": historical_scores[candidate]["first_half"]["group_score"]
            - historical_scores["direct_shared_q07"]["first_half"]["group_score"],
            "2023_h1_to_h2_second_half": historical_scores[candidate]["second_half"]["group_score"]
            - historical_scores["direct_shared_q07"]["second_half"]["group_score"],
        }
        stable = all(value > 0.0 for value in deltas.values())
        meaningful = deltas["2024_full"] >= 0.001
        stability[candidate] = {
            "deltas_vs_direct_shared_q07": deltas,
            "stable_positive_all_periods": stable,
            "meaningful_2024_gain_at_least_0.001": meaningful,
            "adoption_recommended": bool(stable and meaningful),
        }
    adopted = [
        candidate
        for candidate, result in stability.items()
        if result["adoption_recommended"]
    ]

    report = {
        "experiment": "g3_turbine_monotonic_powercurve_post_gate",
        "status": "post_gate_development_not_untouched_gate",
        "method": {
            "manufacturer": "UNISON",
            "turbines": len(TURBINES),
            "turbine_capacity_kw": TURBINE_CAPACITY_KW,
            "power_curve": "per-turbine binned isotonic increasing",
            "timestamp_alignment": "floor(kst_dtm, 1h) + 1h",
            "density": "mean LDAPS/GFS air density, cube-root wind correction",
            "availability": "train-only turbine month x 45-degree-sector shrinkage",
            "calibration": "Huber affine fit on cross-fitted physical train predictions",
            "fixed_blend_weights": list(BLEND_WEIGHTS),
        },
        "leakage_audit": {
            "main_raw_scada_cutoff_exclusive": str(RAW_SCADA_CUTOFF),
            "retained_raw_scada_max": str(pd.to_datetime(raw_scada["kst_dtm"]).max()),
            "main_hourly_scada_max": str(hourly[TIME_COLUMN].max()),
            "historical_hourly_scada_max": str(hist_hourly[TIME_COLUMN].max()),
            "validation_current_scada_used": False,
            "inner_physical_train_predictions": "calendar-month cross-fit; mapper and curves refit",
            "vestas_consistency": (
                "group3 uses only UNISON; VESTAS +/-5e7 cleaner remains enforced in "
                "src.scada for groups1/2 and no VESTAS rows enter this experiment"
            ),
            "unison_consistency": (
                f"4.2MW / 6 = {RATED_KWH10M:.0f} kWh10m nominal; readings above "
                f"{PHYSICAL_POWER_MAX_KWH10M:.0f} excluded from curve fit"
            ),
        },
        "main_2023_to_2024": main_compact,
        "historical_2023_h1_to_h2": hist_compact,
        "historical_shared_q07_training_rows": shared_rows,
        "stability": stability,
        "adoption_candidates": adopted,
        "adoption_recommended": bool(adopted),
        "runtime_seconds": {
            "main": main_seconds,
            "historical": historical_seconds,
            "total": time.perf_counter() - started,
        },
    }

    paths = {
        "report": args.out_dir / "results.json",
        "manifest": args.out_dir / "manifest.json",
        "main_predictions": args.out_dir / "oof" / "g3_2024_physical_candidates.parquet",
        "historical_predictions": args.out_dir
        / "oof"
        / "g3_2023_h1_h2_physical_candidates.parquet",
        "main_physical_oof": args.out_dir / "cache" / "g3_2023_physical_month_oof.parquet",
        "historical_physical_oof": args.out_dir
        / "cache"
        / "g3_2023_h1_physical_month_oof.parquet",
        "models": args.out_dir / "models" / "physical_models.joblib",
    }
    _atomic_parquet(main_audit["predictions"], paths["main_predictions"])
    _atomic_parquet(hist_audit["predictions"], paths["historical_predictions"])
    _atomic_parquet(main_audit["physical_oof"], paths["main_physical_oof"])
    _atomic_parquet(hist_audit["physical_oof"], paths["historical_physical_oof"])
    _atomic_joblib(
        {
            "artifact_type": "g3_turbine_physical_powercurve_models",
            "main": main_models,
            "historical": hist_models,
            "historical_shared_q07": shared_historical_model,
            "recipe_sha256": sha256_file(args.config),
        },
        paths["models"],
    )
    write_json_atomic(paths["report"], report, overwrite=bool(args.overwrite))
    outputs = [path for key, path in paths.items() if key != "manifest"]
    manifest = make_manifest(
        artifact_type="baram_g3_turbine_physical_powercurve_post_gate",
        parameters={
            "status": report["status"],
            "method": report["method"],
            "leakage_audit": report["leakage_audit"],
            "recipe": recipe,
        },
        input_files=required,
        output_files=outputs,
        results={
            "main_scores": main_compact["scores"],
            "historical_scores": hist_compact["scores"],
            "stability": stability,
            "adoption_recommended": bool(adopted),
            "prediction_sha256": sha256_file(paths["main_predictions"]),
        },
        project_dir=PROJECT_DIR,
    )
    write_json_atomic(paths["manifest"], manifest, overwrite=bool(args.overwrite))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report={paths['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

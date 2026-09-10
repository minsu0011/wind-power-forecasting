"""Leakage-safe helpers for the competition's training-only SCADA data.

SCADA is observed at the wind turbines and is *not* available for the 2025 test
period.  Consequently, columns produced by :func:`aggregate_scada_hourly` are
targets for an auxiliary model, never direct test features.  The safe pattern is
therefore::

    hourly = load_scada_hourly(vestas_path, unison_path)

    # Inside every outer/rolling fold only:
    aux = SCADAAuxiliaryRegressor(feature_columns=weather_columns)
    aux.fit(X_train_fold, hourly)       # joins SCADA only to train-fold dates
    X_valid_fold = aux.transform(X_valid_fold)  # weather -> predicted SCADA

    # After model selection, refit on all permissible training rows:
    aux.fit(X_train, hourly)
    X_test = aux.transform(X_test)      # no test SCADA argument is accepted

For a stacked model, do not call ``fit(...).transform(X_train)`` and train the
final model on those in-sample predictions.  Use
:func:`cross_fit_scada_auxiliary` with the same forward/rolling splits used by
the outer validation.  This prevents both future leakage and overly optimistic
stacking scores.

The 10-minute timestamp alignment is deliberate.  Empirically and from the
label definition, records from ``01:00`` through ``01:50`` form the generation
label whose interval ends at ``02:00``.  Thus the default hourly key is
``floor(kst_dtm, 1h) + 1h``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


TIME_COLUMN = "forecast_kst_dtm"
GROUP_COLUMN = "kpx_group"


@dataclass(frozen=True)
class SCADAGroupSpec:
    """Physical and column-layout information for one KPX group."""

    group: int
    manufacturer: str
    turbine_numbers: tuple[int, ...]
    rated_power_kw_per_turbine: float
    expected_intervals_per_hour: int = 6
    cut_in_wind_speed: float = 3.0
    cut_out_wind_speed: float = 25.0

    @property
    def rated_energy_kwh10m(self) -> float:
        """Nominal 10-minute energy at rated output for one turbine."""

        return self.rated_power_kw_per_turbine / 6.0

    @property
    def expected_observations_per_hour(self) -> int:
        return len(self.turbine_numbers) * self.expected_intervals_per_hour

    @property
    def group_capacity_kw(self) -> float:
        return len(self.turbine_numbers) * self.rated_power_kw_per_turbine

    def columns(self, measurement: str) -> list[str]:
        if measurement not in {"power_kw10m", "ws", "wd"}:
            raise ValueError(f"Unknown SCADA measurement: {measurement!r}")
        return [
            f"{self.manufacturer}_wtg{number:02d}_{measurement}"
            for number in self.turbine_numbers
        ]


GROUP_SPECS: Mapping[int, SCADAGroupSpec] = {
    1: SCADAGroupSpec(1, "vestas", tuple(range(1, 7)), 3_600.0),
    2: SCADAGroupSpec(2, "vestas", tuple(range(7, 13)), 3_600.0),
    3: SCADAGroupSpec(3, "unison", tuple(range(1, 6)), 4_200.0),
}

MANUFACTURER_GROUPS: Mapping[str, tuple[int, ...]] = {
    "vestas": (1, 2),
    "unison": (3,),
}


# Targets which can be learned from forecast-weather features without directly
# reconstructing the label.  The direction is represented by sine/cosine, not a
# discontinuous 0/360-degree scalar.
AUXILIARY_TARGET_COLUMNS: tuple[str, ...] = (
    "obs_ws_mean",
    "obs_ws_std",
    "obs_ws_p10",
    "obs_ws_p90",
    "obs_wd_sin",
    "obs_wd_cos",
    "obs_wd_resultant",
    "obs_record_fraction",
    "obs_power_valid_fraction",
    "obs_wind_valid_fraction",
    "obs_turbine_availability",
)

# More target-like quantities are kept separate so callers must opt in.  They
# can be useful for multi-task stacking, but require especially strict OOF use.
POWER_AUXILIARY_TARGET_COLUMNS: tuple[str, ...] = (
    "obs_power_kwh",
    "obs_power_scaled_kwh",
    "obs_capacity_factor",
)

_FRACTION_TARGETS = {
    "obs_record_fraction",
    "obs_power_valid_fraction",
    "obs_ws_valid_fraction",
    "obs_wd_valid_fraction",
    "obs_wind_valid_fraction",
    "obs_turbine_availability",
    "obs_zero_power_fraction",
    "obs_power_spike_fraction",
    "obs_capacity_factor",
}


def _normalise_manufacturer(manufacturer: str) -> str:
    value = str(manufacturer).strip().lower()
    if value not in MANUFACTURER_GROUPS:
        choices = ", ".join(sorted(MANUFACTURER_GROUPS))
        raise ValueError(f"manufacturer must be one of {{{choices}}}, got {manufacturer!r}")
    return value


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], context: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        preview = ", ".join(missing[:8])
        suffix = " ..." if len(missing) > 8 else ""
        raise ValueError(f"{context} is missing required columns: {preview}{suffix}")


def clean_vestas_power_spikes(
    frame: pd.DataFrame,
    *,
    copy: bool = True,
    lower_kwh10m: float = -30.0,
    upper_kwh10m: float = 630.0,
    add_row_diagnostics: bool = False,
) -> pd.DataFrame:
    """Replace impossible VESTAS power readings with ``NaN``.

    A V126 turbine is rated at 3.6 MW, or 600 kWh over ten minutes.  The valid
    data top out at approximately 602 while the corrupt observations reach
    roughly +/- 5.2e7.  A small physical tolerance ([-30, 630]) removes those
    extreme values without winsorising genuine production.  Missing values are
    retained as missing rather than converted to zero.

    When ``add_row_diagnostics`` is true, ``vestas_power_spike_count`` and
    ``vestas_power_valid_count`` are added for audit purposes.  The hourly
    aggregator always exposes analogous group-level diagnostics.
    """

    result = frame.copy() if copy else frame
    columns = [
        f"vestas_wtg{number:02d}_power_kw10m" for number in range(1, 13)
    ]
    _require_columns(result, columns, "VESTAS SCADA frame")

    numeric = result.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(numeric.to_numpy(dtype=float, copy=False))
    physical = numeric.ge(lower_kwh10m) & numeric.le(upper_kwh10m)
    spikes = pd.DataFrame(
        finite & ~physical.to_numpy(dtype=bool, copy=False),
        index=result.index,
        columns=columns,
    )
    result.loc[:, columns] = numeric.mask(spikes)

    if add_row_diagnostics:
        result["vestas_power_spike_count"] = spikes.sum(axis=1).astype("int16")
        result["vestas_power_valid_count"] = (
            result.loc[:, columns].notna().sum(axis=1).astype("int16")
        )
    return result


def _power_bounds(spec: SCADAGroupSpec) -> tuple[float, float]:
    # VESTAS is the known extreme-spike source.  UNISON has repeated 800 values,
    # so retain those with a wider tolerance while rejecting isolated 900/1000.
    if spec.manufacturer == "vestas":
        return (-30.0, 1.05 * spec.rated_energy_kwh10m)
    return (-35.0, 1.15 * spec.rated_energy_kwh10m)


def _numeric_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    return (
        frame.loc[:, list(columns)]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float, copy=True)
    )


def _safe_fraction(numerator: pd.Series, denominator: pd.Series | float) -> pd.Series:
    result = numerator.astype(float).div(denominator)
    if isinstance(denominator, pd.Series):
        result = result.where(denominator > 0)
    elif denominator <= 0:
        result[:] = np.nan
    return result


def _aggregate_group(
    frame: pd.DataFrame,
    spec: SCADAGroupSpec,
    source_times: pd.Series,
    target_times: pd.Series,
    *,
    clean_power: bool,
    max_wind_speed: float,
    online_power_kwh10m: float,
) -> pd.DataFrame:
    power_columns = spec.columns("power_kw10m")
    ws_columns = spec.columns("ws")
    wd_columns = spec.columns("wd")
    _require_columns(
        frame,
        [*power_columns, *ws_columns, *wd_columns],
        f"{spec.manufacturer.upper()} group {spec.group}",
    )

    power = _numeric_matrix(frame, power_columns)
    wind_speed = _numeric_matrix(frame, ws_columns)
    wind_direction = _numeric_matrix(frame, wd_columns)

    finite_power = np.isfinite(power)
    lower_power, upper_power = _power_bounds(spec)
    power_spike = finite_power & ((power < lower_power) | (power > upper_power))
    if clean_power:
        power[power_spike] = np.nan

    finite_ws = np.isfinite(wind_speed)
    invalid_ws = finite_ws & ((wind_speed < 0.0) | (wind_speed > max_wind_speed))
    wind_speed[invalid_ws] = np.nan

    # Both negative angles and angles above 360 are valid circular encodings.
    # Only non-finite values are missing; modulo maps everything else to [0,360).
    wind_direction[~np.isfinite(wind_direction)] = np.nan
    wind_direction = np.mod(wind_direction, 360.0)

    n_turbines = len(spec.turbine_numbers)
    long = pd.DataFrame(
        {
            TIME_COLUMN: np.repeat(target_times.to_numpy(), n_turbines),
            "source_kst_dtm": np.repeat(source_times.to_numpy(), n_turbines),
            "turbine": np.tile(np.asarray(spec.turbine_numbers), len(frame)),
            "power": power.reshape(-1),
            "ws": wind_speed.reshape(-1),
            "wd": wind_direction.reshape(-1),
            "power_spike": power_spike.reshape(-1),
            "invalid_ws": invalid_ws.reshape(-1),
        }
    )
    long = long.loc[long[TIME_COLUMN].notna()].copy()
    if long.empty:
        return pd.DataFrame(columns=[TIME_COLUMN, GROUP_COLUMN])

    radians = np.deg2rad(long["wd"])
    long["wd_sin_component"] = np.sin(radians)
    long["wd_cos_component"] = np.cos(radians)
    long["wind_pair_valid"] = long["ws"].notna() & long["wd"].notna()
    long["adequate_wind"] = (
        long["power"].notna()
        & long["ws"].between(
            spec.cut_in_wind_speed,
            spec.cut_out_wind_speed,
            inclusive="both",
        )
    )
    long["online"] = long["adequate_wind"] & (
        long["power"] > online_power_kwh10m
    )
    long["zero_power"] = long["adequate_wind"] & (long["power"] <= 0.0)

    grouped = long.groupby(TIME_COLUMN, sort=True, observed=True)
    out = pd.DataFrame(index=grouped.size().index)
    out["obs_power_kwh"] = grouped["power"].sum(min_count=1)
    out["obs_power_mean_kwh10m"] = grouped["power"].mean()
    out["obs_power_std_kwh10m"] = grouped["power"].std(ddof=0)
    out["obs_ws_mean"] = grouped["ws"].mean()
    out["obs_ws_std"] = grouped["ws"].std(ddof=0)
    out["obs_ws_min"] = grouped["ws"].min()
    out["obs_ws_max"] = grouped["ws"].max()
    out["obs_ws_p10"] = grouped["ws"].quantile(0.10)
    out["obs_ws_p90"] = grouped["ws"].quantile(0.90)
    out["obs_wd_sin"] = grouped["wd_sin_component"].mean()
    out["obs_wd_cos"] = grouped["wd_cos_component"].mean()
    out["obs_wd_resultant"] = np.hypot(out["obs_wd_sin"], out["obs_wd_cos"])
    out["obs_wd_mean_deg"] = np.mod(
        np.rad2deg(np.arctan2(out["obs_wd_sin"], out["obs_wd_cos"])),
        360.0,
    )

    out["obs_n_intervals"] = grouped["source_kst_dtm"].nunique().astype("int16")
    out["obs_n_power_valid"] = grouped["power"].count().astype("int16")
    out["obs_n_ws_valid"] = grouped["ws"].count().astype("int16")
    out["obs_n_wd_valid"] = grouped["wd"].count().astype("int16")
    out["obs_n_wind_pair_valid"] = grouped["wind_pair_valid"].sum().astype("int16")
    out["obs_n_adequate_wind"] = grouped["adequate_wind"].sum().astype("int16")
    out["obs_n_online"] = grouped["online"].sum().astype("int16")
    out["obs_power_spike_count"] = grouped["power_spike"].sum().astype("int16")
    out["obs_invalid_ws_count"] = grouped["invalid_ws"].sum().astype("int16")

    expected = float(spec.expected_observations_per_hour)
    expected_intervals = float(spec.expected_intervals_per_hour)
    out["obs_record_fraction"] = (out["obs_n_intervals"] / expected_intervals).clip(0.0, 1.0)
    out["obs_power_valid_fraction"] = (out["obs_n_power_valid"] / expected).clip(0.0, 1.0)
    out["obs_ws_valid_fraction"] = (out["obs_n_ws_valid"] / expected).clip(0.0, 1.0)
    out["obs_wd_valid_fraction"] = (out["obs_n_wd_valid"] / expected).clip(0.0, 1.0)
    out["obs_wind_valid_fraction"] = (
        out["obs_n_wind_pair_valid"] / expected
    ).clip(0.0, 1.0)
    out["obs_power_spike_fraction"] = (out["obs_power_spike_count"] / expected).clip(0.0, 1.0)
    out["obs_turbine_availability"] = _safe_fraction(
        out["obs_n_online"], out["obs_n_adequate_wind"]
    ).clip(0.0, 1.0)
    out["obs_zero_power_fraction"] = _safe_fraction(
        grouped["zero_power"].sum(), out["obs_n_adequate_wind"]
    ).clip(0.0, 1.0)

    # Raw sum preserves true outages; scaled sum is a separate estimate that
    # compensates only for missing/corrupt sensor readings.
    out["obs_power_scaled_kwh"] = out["obs_power_mean_kwh10m"] * expected
    out["obs_capacity_factor"] = (
        out["obs_power_scaled_kwh"] / spec.group_capacity_kw
    ).clip(lower=0.0, upper=1.10)

    out = out.reset_index()
    out.insert(1, GROUP_COLUMN, spec.group)
    out.insert(2, "manufacturer", spec.manufacturer)
    out.insert(3, "group_capacity_kw", spec.group_capacity_kw)
    return out


def aggregate_scada_hourly(
    frame: pd.DataFrame,
    manufacturer: str,
    *,
    timestamp_column: str = "kst_dtm",
    label_shift_hours: int = 1,
    clean_power: bool = True,
    max_wind_speed: float = 75.0,
    online_power_kwh10m: float = 1.0,
) -> pd.DataFrame:
    """Aggregate one manufacturer's 10-minute SCADA into KPX-hour targets.

    Parameters
    ----------
    frame:
        Raw VESTAS or UNISON training frame.
    manufacturer:
        ``"vestas"`` (groups 1 and 2) or ``"unison"`` (group 3).
    label_shift_hours:
        Offset after flooring the source timestamp.  Keep the default ``1`` for
        the supplied data: six readings at HH:00..HH:50 predict the interval
        ending at HH+1:00.
    clean_power:
        Convert readings outside conservative physical bounds to missing.  This
        removes the +/-5e7 VESTAS corruption instead of clipping it into a valid
        but misleading value.

    Returns
    -------
    pandas.DataFrame
        One row per ``(forecast_kst_dtm, kpx_group)``.  Direction summaries use
        circular sine/cosine means.  Availability distinguishes missing sensors
        from turbine online rate under adequate observed wind.
    """

    manufacturer = _normalise_manufacturer(manufacturer)
    _require_columns(frame, [timestamp_column], f"{manufacturer.upper()} SCADA frame")
    if max_wind_speed <= 0:
        raise ValueError("max_wind_speed must be positive")
    if online_power_kwh10m < 0:
        raise ValueError("online_power_kwh10m must be non-negative")

    source_times = pd.to_datetime(frame[timestamp_column], errors="coerce")
    target_times = source_times.dt.floor("h") + pd.to_timedelta(label_shift_hours, unit="h")

    outputs = [
        _aggregate_group(
            frame,
            GROUP_SPECS[group],
            source_times,
            target_times,
            clean_power=clean_power,
            max_wind_speed=max_wind_speed,
            online_power_kwh10m=online_power_kwh10m,
        )
        for group in MANUFACTURER_GROUPS[manufacturer]
    ]
    result = pd.concat(outputs, ignore_index=True, sort=False)
    if result.empty:
        return result
    return result.sort_values([TIME_COLUMN, GROUP_COLUMN], kind="stable").reset_index(drop=True)


def combine_scada_hourly(
    vestas: pd.DataFrame | None = None,
    unison: pd.DataFrame | None = None,
    **aggregate_kwargs: Any,
) -> pd.DataFrame:
    """Aggregate and combine any available training SCADA manufacturers."""

    outputs: list[pd.DataFrame] = []
    if vestas is not None:
        outputs.append(aggregate_scada_hourly(vestas, "vestas", **aggregate_kwargs))
    if unison is not None:
        outputs.append(aggregate_scada_hourly(unison, "unison", **aggregate_kwargs))
    if not outputs:
        raise ValueError("At least one of vestas or unison must be provided")

    result = pd.concat(outputs, ignore_index=True, sort=False)
    duplicated = result.duplicated([TIME_COLUMN, GROUP_COLUMN], keep=False)
    if duplicated.any():
        examples = result.loc[duplicated, [TIME_COLUMN, GROUP_COLUMN]].head().to_dict("records")
        raise ValueError(f"Duplicate hourly SCADA keys after combining: {examples}")
    return result.sort_values([TIME_COLUMN, GROUP_COLUMN], kind="stable").reset_index(drop=True)


def load_scada_hourly(
    vestas_path: str | Path | None = None,
    unison_path: str | Path | None = None,
    *,
    encoding: str = "utf-8-sig",
    **aggregate_kwargs: Any,
) -> pd.DataFrame:
    """Read supplied SCADA CSVs and return :func:`combine_scada_hourly` output."""

    vestas = pd.read_csv(vestas_path, encoding=encoding) if vestas_path is not None else None
    unison = pd.read_csv(unison_path, encoding=encoding) if unison_path is not None else None
    return combine_scada_hourly(vestas, unison, **aggregate_kwargs)


def auxiliary_target_frame(
    hourly_scada: pd.DataFrame,
    *,
    include_power_targets: bool = False,
    target_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Select stable auxiliary targets plus their timestamp/group keys."""

    if target_columns is None:
        selected = list(AUXILIARY_TARGET_COLUMNS)
        if include_power_targets:
            selected.extend(POWER_AUXILIARY_TARGET_COLUMNS)
    else:
        selected = list(target_columns)
    _require_columns(hourly_scada, [TIME_COLUMN, GROUP_COLUMN, *selected], "hourly SCADA")
    return hourly_scada.loc[:, [TIME_COLUMN, GROUP_COLUMN, *selected]].copy()


_UNSAFE_AUTOMATIC_FEATURE = re.compile(
    r"(^kpx_group_[123]$|^obs_|^scada_|label|actual)", re.IGNORECASE
)


class SCADAAuxiliaryRegressor:
    """Forecast training-only SCADA summaries from test-available features.

    ``fit`` intersects ``scada_targets`` with the keys present in ``X``.  Passing
    the complete SCADA table is therefore safe *provided X contains only the
    current fold's training rows*.  ``transform`` accepts only X, which makes it
    impossible for this object to consume test-period SCADA.

    Parameters
    ----------
    feature_columns:
        Explicit weather/calendar/power-curve columns.  Supplying this list is
        recommended.  If omitted, numeric columns are selected while labels and
        names beginning ``obs_``/``scada_`` are rejected.
    target_columns:
        Defaults to observed wind summaries and availability.  Add
        :data:`POWER_AUXILIARY_TARGET_COLUMNS` only with fully cross-fitted
        stacking.
    group:
        Set to 1, 2, or 3 when X is a single-group table without ``group_column``.
    estimator:
        A scikit-learn compatible regressor, cloned per group/target.  The
        default is a conservative L1 ``HistGradientBoostingRegressor``.

    Notes
    -----
    For rolling validation, instantiate and fit this class separately inside
    every fold.  Use :func:`cross_fit_scada_auxiliary` to build train-level OOF
    columns, then refit one new instance on all allowed training dates before
    transforming test weather.  In-sample auxiliary predictions create a
    stacking/overfit illusion even though they do not technically use the future.
    """

    def __init__(
        self,
        *,
        feature_columns: Sequence[str] | None = None,
        target_columns: Sequence[str] = AUXILIARY_TARGET_COLUMNS,
        time_column: str = TIME_COLUMN,
        group_column: str = GROUP_COLUMN,
        group: int | None = None,
        estimator: Any | None = None,
        estimator_factory: Callable[[], Any] | None = None,
        min_samples: int = 168,
        prediction_prefix: str = "pred_scada_",
        random_state: int = 2026,
        strict_feature_safety: bool = True,
    ) -> None:
        if estimator is not None and estimator_factory is not None:
            raise ValueError("Use estimator or estimator_factory, not both")
        self.feature_columns = None if feature_columns is None else tuple(feature_columns)
        self.target_columns = tuple(target_columns)
        self.time_column = time_column
        self.group_column = group_column
        self.group = group
        self.estimator = estimator
        self.estimator_factory = estimator_factory
        self.min_samples = int(min_samples)
        self.prediction_prefix = prediction_prefix
        self.random_state = int(random_state)
        self.strict_feature_safety = bool(strict_feature_safety)

    def _with_keys(self, frame: pd.DataFrame) -> pd.DataFrame:
        _require_columns(frame, [self.time_column], "feature frame")
        result = frame.copy()
        result[self.time_column] = pd.to_datetime(result[self.time_column], errors="coerce")
        if self.group_column not in result.columns:
            if self.group is None:
                raise ValueError(
                    f"X needs {self.group_column!r}, or construct with group=1/2/3"
                )
            result[self.group_column] = self.group
        return result

    def _select_features(self, frame: pd.DataFrame) -> tuple[str, ...]:
        if self.feature_columns is None:
            excluded = {self.time_column, self.group_column, "forecast_id", *self.target_columns}
            columns = tuple(
                column
                for column in frame.columns
                if column not in excluded
                and pd.api.types.is_numeric_dtype(frame[column])
                and not _UNSAFE_AUTOMATIC_FEATURE.search(str(column))
            )
        else:
            columns = self.feature_columns
            _require_columns(frame, columns, "feature frame")

        if not columns:
            raise ValueError("No numeric auxiliary-model features were selected")
        unsafe = [
            column
            for column in columns
            if column in self.target_columns
            or column in {self.time_column, self.group_column}
            or _UNSAFE_AUTOMATIC_FEATURE.search(str(column))
        ]
        if unsafe and self.strict_feature_safety:
            raise ValueError(
                "Potential label/SCADA columns cannot be auxiliary inputs: "
                + ", ".join(map(str, unsafe))
            )
        return tuple(columns)

    def _new_estimator(self) -> Any:
        from sklearn.base import clone
        from sklearn.ensemble import HistGradientBoostingRegressor

        if self.estimator_factory is not None:
            return self.estimator_factory()
        if self.estimator is not None:
            return clone(self.estimator)
        return HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.045,
            max_iter=180,
            max_leaf_nodes=15,
            min_samples_leaf=48,
            l2_regularization=2.0,
            early_stopping=False,
            random_state=self.random_state,
        )

    def fit(self, X: pd.DataFrame, scada_targets: pd.DataFrame) -> "SCADAAuxiliaryRegressor":
        """Fit only on the keys in X; never pass validation/test rows in X."""

        from sklearn.dummy import DummyRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline

        if self.min_samples < 1:
            raise ValueError("min_samples must be at least 1")
        features = self._with_keys(X)
        self.feature_columns_ = self._select_features(features)

        _require_columns(
            scada_targets,
            [self.time_column, self.group_column, *self.target_columns],
            "SCADA target frame",
        )
        targets = scada_targets.loc[
            :, [self.time_column, self.group_column, *self.target_columns]
        ].copy()
        targets[self.time_column] = pd.to_datetime(targets[self.time_column], errors="coerce")
        if targets.duplicated([self.time_column, self.group_column]).any():
            raise ValueError("SCADA targets must be unique by time and group")

        merged = features.loc[
            :, [self.time_column, self.group_column, *self.feature_columns_]
        ].merge(
            targets,
            on=[self.time_column, self.group_column],
            how="left",
            validate="many_to_one",
            sort=False,
        )
        if merged.loc[:, list(self.target_columns)].notna().sum().sum() == 0:
            raise ValueError("No SCADA targets overlap the supplied fold-training feature keys")

        self.models_: dict[tuple[Any, str], Any] = {}
        self.fallbacks_: dict[tuple[Any, str], float] = {}
        self.training_rows_: dict[tuple[Any, str], int] = {}
        self.groups_ = tuple(pd.unique(features[self.group_column].dropna()))

        for group_value in self.groups_:
            group_rows = merged.loc[merged[self.group_column] == group_value]
            for target in self.target_columns:
                valid = group_rows[target].notna()
                n_valid = int(valid.sum())
                if n_valid == 0:
                    raise ValueError(
                        f"No target {target!r} for group {group_value!r} in this fold"
                    )
                train_x = group_rows.loc[valid, list(self.feature_columns_)].replace(
                    [np.inf, -np.inf], np.nan
                )
                train_y = group_rows.loc[valid, target].astype(float)
                fallback = float(train_y.median())
                self.fallbacks_[(group_value, target)] = fallback
                self.training_rows_[(group_value, target)] = n_valid

                regressor = (
                    self._new_estimator()
                    if n_valid >= self.min_samples
                    else DummyRegressor(strategy="median")
                )
                pipeline = make_pipeline(
                    SimpleImputer(strategy="median", keep_empty_features=True),
                    regressor,
                )
                pipeline.fit(train_x, train_y)
                self.models_[(group_value, target)] = pipeline

        valid_times = features[self.time_column].dropna()
        self.fit_min_time_ = valid_times.min() if not valid_times.empty else pd.NaT
        self.fit_max_time_ = valid_times.max() if not valid_times.empty else pd.NaT
        self.is_fitted_ = True
        return self

    def _prediction_columns(self) -> list[str]:
        return [f"{self.prediction_prefix}{target}" for target in self.target_columns]

    @staticmethod
    def _clip_target(target: str, values: np.ndarray) -> np.ndarray:
        if target in _FRACTION_TARGETS:
            return np.clip(values, 0.0, 1.0)
        if target in {"obs_wd_sin", "obs_wd_cos"}:
            return np.clip(values, -1.0, 1.0)
        if target.startswith("obs_ws_") or target.startswith("obs_power_"):
            return np.clip(values, 0.0, None)
        return values

    def predict_auxiliary(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return predicted SCADA features; no observed SCADA is accepted."""

        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("SCADAAuxiliaryRegressor must be fitted before prediction")
        features = self._with_keys(X)
        _require_columns(features, self.feature_columns_, "prediction feature frame")
        result = pd.DataFrame(index=X.index, columns=self._prediction_columns(), dtype=float)

        for group_value in pd.unique(features[self.group_column].dropna()):
            positions = np.flatnonzero(
                features[self.group_column].to_numpy() == group_value
            )
            if len(positions) == 0:
                continue
            group_x = features.iloc[positions].loc[:, list(self.feature_columns_)].replace(
                [np.inf, -np.inf], np.nan
            )
            for target, output_column in zip(
                self.target_columns, self._prediction_columns()
            ):
                model = self.models_.get((group_value, target))
                if model is None:
                    known = [
                        value
                        for (known_group, known_target), value in self.fallbacks_.items()
                        if known_target == target
                    ]
                    if not known:
                        raise ValueError(f"No fitted fallback exists for target {target!r}")
                    prediction = np.full(len(positions), float(np.median(known)))
                else:
                    prediction = np.asarray(model.predict(group_x), dtype=float)
                result.iloc[positions, result.columns.get_loc(output_column)] = self._clip_target(
                    target, prediction
                )
        return result

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Append predicted auxiliary columns to X without requiring test SCADA."""

        predictions = self.predict_auxiliary(X)
        result = X.copy()
        for column in predictions:
            result[column] = predictions[column]
        return result


def cross_fit_scada_auxiliary(
    X: pd.DataFrame,
    scada_targets: pd.DataFrame,
    splits: Iterable[tuple[Sequence[int], Sequence[int]]],
    *,
    model_kwargs: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Create leakage-safe OOF auxiliary predictions from positional splits.

    ``splits`` should be expanding/rolling time splits.  Every validation row may
    appear at most once.  Rows outside all validation windows remain ``NaN`` so
    callers cannot accidentally mistake them for OOF predictions.
    """

    kwargs = dict(model_kwargs or {})
    template = SCADAAuxiliaryRegressor(**kwargs)
    columns = template._prediction_columns()
    oof = pd.DataFrame(np.nan, index=X.index, columns=columns, dtype=float)
    assigned = np.zeros(len(X), dtype=bool)

    for fold_number, (train_positions, valid_positions) in enumerate(splits):
        train_positions = np.asarray(train_positions, dtype=int)
        valid_positions = np.asarray(valid_positions, dtype=int)
        if train_positions.size == 0 or valid_positions.size == 0:
            raise ValueError(f"Fold {fold_number} has an empty train or validation set")
        if assigned[valid_positions].any():
            raise ValueError(f"Fold {fold_number} overlaps an earlier validation fold")

        model = SCADAAuxiliaryRegressor(**kwargs)
        model.fit(X.iloc[train_positions], scada_targets)
        fold_prediction = model.predict_auxiliary(X.iloc[valid_positions])
        oof.iloc[valid_positions, :] = fold_prediction.to_numpy()
        assigned[valid_positions] = True
    return oof


__all__ = [
    "AUXILIARY_TARGET_COLUMNS",
    "GROUP_COLUMN",
    "GROUP_SPECS",
    "MANUFACTURER_GROUPS",
    "POWER_AUXILIARY_TARGET_COLUMNS",
    "SCADAAuxiliaryRegressor",
    "SCADAGroupSpec",
    "TIME_COLUMN",
    "aggregate_scada_hourly",
    "auxiliary_target_frame",
    "clean_vestas_power_spikes",
    "combine_scada_hourly",
    "cross_fit_scada_auxiliary",
    "load_scada_hourly",
]

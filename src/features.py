"""Leakage-safe weather features for the BARAM 2026 wind-power task.

The weather files contain one 24-hour forecast *run* per day and several
spatial grid points per forecast hour.  This module deliberately uses only
information present in those files.  In particular it never reads labels or
SCADA data, and temporal neighbours are formed only inside an identical
``data_available_kst_dtm`` run.  A lead value is therefore available at the
same issuance time as the current value and is not future-observation leakage.

The public entry point is :func:`build_train_test_weather_features`.  It
returns one frame per KPX group because each wind-farm group has a different
spatial footprint.  Frames have a sorted ``forecast_kst_dtm`` index and an
identical, numeric float32 schema for train and test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


TIME_COL = "forecast_kst_dtm"
AVAILABLE_COL = "data_available_kst_dtm"
GRID_COL = "grid_id"
COORD_COLS = ("latitude", "longitude")

LDAPS_VARIABLES: tuple[str, ...] = (
    "heightAboveGround_10_10u",
    "heightAboveGround_10_10v",
    "heightAboveGround_50_50MUmax",
    "heightAboveGround_50_50MUmin",
    "heightAboveGround_50_50MVmax",
    "heightAboveGround_50_50MVmin",
    "heightAboveGround_5_XBLWS",
    "heightAboveGround_5_YBLWS",
    "heightAboveGround_2_t",
    "heightAboveGround_2_dpt",
    "heightAboveGround_2_r",
    "heightAboveGround_2_q",
    "surface_0_sp",
    "meanSea_0_prmsl",
    "etc_0_blh",
    "surface_0_NDNSW",
    "surface_0_NDNLW",
    "heightAboveGround_2_SWDIR",
    "heightAboveGround_2_SWDIF",
    "etc_0_hcc",
    "etc_0_mcc",
    "etc_0_lcc",
    "etc_0_VLCDC",
    "surface_0_avg_lsprate",
    "surface_0_lssrate",
    "surface_0_ncpcp",
    "surface_0_snol",
    "surface_0_SNOM",
    "surface_0_lsm",
    "surface_0_h",
)

GFS_VARIABLES: tuple[str, ...] = (
    "heightAboveGround_10_10u",
    "heightAboveGround_10_10v",
    "heightAboveGround_80_u",
    "heightAboveGround_80_v",
    "heightAboveGround_100_100u",
    "heightAboveGround_100_100v",
    "heightAboveGround_2_2t",
    "heightAboveGround_2_2d",
    "heightAboveGround_2_2r",
    "heightAboveGround_2_2sh",
    "planetaryBoundaryLayer_0_u",
    "planetaryBoundaryLayer_0_v",
    "planetaryBoundaryLayer_0_VRATE",
    "surface_0_dswrf",
    "surface_0_dlwrf",
    "surface_0_prate",
    "surface_0_tp",
    "surface_0_sp",
    "meanSea_0_prmsl",
    "surface_0_gust",
    "lowCloudLayer_0_lcc",
    "middleCloudLayer_0_mcc",
    "highCloudLayer_0_hcc",
    "atmosphere_0_tcc",
    "isobaricInhPa_850_t",
    "isobaricInhPa_850_u",
    "isobaricInhPa_850_v",
    "isobaricInhPa_850_r",
    "isobaricInhPa_700_t",
    "isobaricInhPa_700_u",
    "isobaricInhPa_700_v",
    "isobaricInhPa_500_gh",
    "isobaricInhPa_500_t",
    "isobaricInhPa_500_u",
    "isobaricInhPa_500_v",
)

SOURCE_VARIABLES: dict[str, tuple[str, ...]] = {
    "ldaps": LDAPS_VARIABLES,
    "gfs": GFS_VARIABLES,
}

LDAPS_WIDE_VARIABLES: tuple[str, ...] = LDAPS_VARIABLES[:8]
GFS_WIDE_VARIABLES: tuple[str, ...] = (
    "heightAboveGround_10_10u",
    "heightAboveGround_10_10v",
    "heightAboveGround_80_u",
    "heightAboveGround_80_v",
    "heightAboveGround_100_100u",
    "heightAboveGround_100_100v",
    "planetaryBoundaryLayer_0_u",
    "planetaryBoundaryLayer_0_v",
)
SOURCE_WIDE_VARIABLES = {
    "ldaps": LDAPS_WIDE_VARIABLES,
    "gfs": GFS_WIDE_VARIABLES,
}


@dataclass(frozen=True)
class GroupSite:
    """Physical metadata for one KPX target group."""

    name: str
    latitude: float
    longitude: float
    hub_height_m: float
    rotor_diameter_m: float
    capacity_kwh: float


# Exact centroids computed from every turbine coordinate in info.xlsx.  The
# workbook loader below is preferred; these constants make the module usable
# when only the four CSV files are copied to another machine.
DEFAULT_GROUP_SITES: dict[str, GroupSite] = {
    "kpx_group_1": GroupSite(
        "kpx_group_1", 37.287127, 128.952021, 117.0, 126.0, 21_600.0
    ),
    "kpx_group_2": GroupSite(
        "kpx_group_2", 37.282255, 128.965148, 117.0, 126.0, 21_600.0
    ),
    "kpx_group_3": GroupSite(
        "kpx_group_3", 37.275199, 128.971444, 117.0, 136.0, 21_000.0
    ),
}


@dataclass(frozen=True)
class WeatherFeatureConfig:
    """Controls the bounded-size spatial and within-run feature expansion."""

    nearest_k: int = 4
    wide_nearest_k: int = 4
    idw_power: float = 2.0
    idw_distance_floor_km: float = 0.05
    run_offsets: tuple[int, ...] = (1, 3)
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.nearest_k < 1 or self.wide_nearest_k < 1:
            raise ValueError("nearest_k and wide_nearest_k must be positive")
        if self.idw_power <= 0 or self.idw_distance_floor_km <= 0:
            raise ValueError("IDW parameters must be positive")
        if not self.run_offsets or any(offset <= 0 for offset in self.run_offsets):
            raise ValueError("run_offsets must contain positive integers")


WeatherInput = str | Path | pd.DataFrame


def _parse_dms(value: object) -> tuple[float, float]:
    tokens = re.findall(r"[\d.]+|[NSEW]", str(value).upper())
    if len(tokens) != 8:
        raise ValueError(f"cannot parse turbine coordinate: {value!r}")
    lat_d, lat_m, lat_s, ns, lon_d, lon_m, lon_s, ew = tokens
    latitude = float(lat_d) + float(lat_m) / 60.0 + float(lat_s) / 3600.0
    longitude = float(lon_d) + float(lon_m) / 60.0 + float(lon_s) / 3600.0
    if ns == "S":
        latitude *= -1.0
    if ew == "W":
        longitude *= -1.0
    return latitude, longitude


def load_group_sites(info_path: str | Path) -> dict[str, GroupSite]:
    """Load turbine centroids and capacities without relying on fixed row numbers."""

    raw = pd.read_excel(info_path, sheet_name="info", header=None)
    header_candidates = raw.index[
        raw.apply(lambda row: row.astype(str).eq("KPX그룹").any(), axis=1)
    ]
    if len(header_candidates) != 1:
        raise ValueError("info.xlsx must contain exactly one KPX그룹 header row")
    header_row = int(header_candidates[0])
    table = raw.iloc[header_row + 1 :].copy()
    table.columns = raw.iloc[header_row].tolist()
    required = {
        "좌표(Google)",
        "KPX그룹",
        "Hub Height(m)",
        "Rotor Diameter(m)",
        "설비용량(MW)",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"info.xlsx missing columns: {sorted(missing)}")
    table = table.loc[table["좌표(Google)"].notna()].copy()
    table["KPX그룹"] = pd.to_numeric(table["KPX그룹"], errors="coerce").ffill()
    coordinates = table["좌표(Google)"].map(_parse_dms)
    table["_latitude"] = coordinates.map(lambda pair: pair[0])
    table["_longitude"] = coordinates.map(lambda pair: pair[1])

    sites: dict[str, GroupSite] = {}
    for group_number, group in table.groupby("KPX그룹", sort=True):
        number = int(group_number)
        name = f"kpx_group_{number}"
        sites[name] = GroupSite(
            name=name,
            latitude=float(group["_latitude"].mean()),
            longitude=float(group["_longitude"].mean()),
            hub_height_m=float(
                pd.to_numeric(group["Hub Height(m)"], errors="coerce").mean()
            ),
            rotor_diameter_m=float(
                pd.to_numeric(group["Rotor Diameter(m)"], errors="coerce").mean()
            ),
            capacity_kwh=float(
                pd.to_numeric(group["설비용량(MW)"], errors="coerce").sum() * 1_000.0
            ),
        )
    expected = set(DEFAULT_GROUP_SITES)
    if set(sites) != expected:
        raise ValueError(f"expected groups {sorted(expected)}, got {sorted(sites)}")
    return sites


def read_weather_csv(path: str | Path, source: str) -> pd.DataFrame:
    """Read one weather CSV with compact dtypes and parsed KST timestamps."""

    if source not in SOURCE_VARIABLES:
        raise KeyError(f"unknown weather source: {source}")
    required = [TIME_COL, AVAILABLE_COL, GRID_COL, *COORD_COLS, *SOURCE_VARIABLES[source]]
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns
    missing = set(required).difference(header)
    if missing:
        raise ValueError(f"{path} missing {source} columns: {sorted(missing)}")
    dtypes: dict[str, str] = {GRID_COL: "int16"}
    dtypes.update({column: "float32" for column in [*COORD_COLS, *SOURCE_VARIABLES[source]]})
    return pd.read_csv(
        path,
        usecols=required,
        dtype=dtypes,
        parse_dates=[TIME_COL, AVAILABLE_COL],
        encoding="utf-8-sig",
        memory_map=True,
        low_memory=False,
    )


def _as_weather_frame(data: WeatherInput, source: str) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
        for column in (TIME_COL, AVAILABLE_COL):
            if column in frame and not isinstance(frame[column].dtype, pd.DatetimeTZDtype):
                frame[column] = pd.to_datetime(frame[column], errors="raise")
        return frame
    return read_weather_csv(data, source)


def _haversine_km(
    latitude: np.ndarray, longitude: np.ndarray, site: GroupSite
) -> np.ndarray:
    lat1 = np.radians(site.latitude)
    lat2 = np.radians(latitude.astype("float64", copy=False))
    delta_lat = lat2 - lat1
    delta_lon = np.radians(longitude.astype("float64", copy=False) - site.longitude)
    hav = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2.0) ** 2
    )
    return 6_371.0088 * 2.0 * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))


class WeatherFeatureBuilder:
    """Fit training-only fallbacks, then build group-specific weather features.

    ``fit`` learns only raw-column medians and the fixed weather grid catalogue.
    It never observes the target.  For completely strict rolling validation,
    fit a separate instance on each fold's training weather.  The supplied
    files have no train weather missingness, so the medians do not alter any
    provided training row.
    """

    def __init__(
        self,
        group_sites: Mapping[str, GroupSite] | None = None,
        config: WeatherFeatureConfig | None = None,
    ) -> None:
        self.group_sites = dict(group_sites or DEFAULT_GROUP_SITES)
        self.config = config or WeatherFeatureConfig()
        if set(self.group_sites) != set(DEFAULT_GROUP_SITES):
            raise ValueError("group_sites must contain kpx_group_1, _2, and _3")
        self.fill_values_: dict[str, pd.Series] = {}
        self.grid_catalog_: dict[str, pd.DataFrame] = {}
        self.feature_names_: tuple[str, ...] | None = None

    def fit(self, ldaps: WeatherInput, gfs: WeatherInput) -> "WeatherFeatureBuilder":
        """Learn non-target fallbacks and fixed grid geometry from training weather."""

        for source, data in (("ldaps", ldaps), ("gfs", gfs)):
            frame = _as_weather_frame(data, source)
            self._validate_frame(frame, source)
            variables = list(SOURCE_VARIABLES[source])
            medians = frame[variables].median(axis=0, skipna=True).fillna(0.0)
            self.fill_values_[source] = medians.astype(self.config.dtype)
            self.grid_catalog_[source] = self._grid_catalog(frame)
        self.feature_names_ = None
        return self

    def fit_transform(
        self, ldaps: WeatherInput, gfs: WeatherInput
    ) -> dict[str, pd.DataFrame]:
        self.fit(ldaps, gfs)
        return self.transform(ldaps, gfs)

    def transform(
        self, ldaps: WeatherInput, gfs: WeatherInput
    ) -> dict[str, pd.DataFrame]:
        """Create same-schema float32 frames keyed by ``kpx_group_*``."""

        if set(self.fill_values_) != {"ldaps", "gfs"}:
            raise RuntimeError("WeatherFeatureBuilder.fit must be called before transform")

        source_frames: dict[str, dict[str, pd.DataFrame]] = {}
        available: dict[str, pd.Series] = {}
        for source, data in (("ldaps", ldaps), ("gfs", gfs)):
            raw = _as_weather_frame(data, source)
            self._validate_frame(raw, source, check_catalog=True)
            source_frames[source], available[source] = self._aggregate_source(raw, source)

        ldaps_index = next(iter(source_frames["ldaps"].values())).index
        gfs_index = next(iter(source_frames["gfs"].values())).index
        if not ldaps_index.equals(gfs_index):
            raise ValueError("LDAPS and GFS forecast timestamps are not identically aligned")
        if not available["ldaps"].equals(available["gfs"]):
            raise ValueError("LDAPS and GFS data-availability runs are not aligned")

        result: dict[str, pd.DataFrame] = {}
        for group_name, site in self.group_sites.items():
            frame = pd.concat(
                [source_frames["ldaps"][group_name], source_frames["gfs"][group_name]],
                axis=1,
                copy=False,
            )
            frame = self._add_physics(frame, site)
            frame = self._add_time_features(frame, available["ldaps"])
            frame = self._add_within_run_dynamics(frame, available["ldaps"])
            frame["site__latitude"] = np.float32(site.latitude)
            frame["site__longitude"] = np.float32(site.longitude)
            frame["site__hub_height_m"] = np.float32(site.hub_height_m)
            frame["site__rotor_diameter_m"] = np.float32(site.rotor_diameter_m)
            frame["site__capacity_kwh"] = np.float32(site.capacity_kwh)
            frame = frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            frame = frame.astype(self.config.dtype, copy=False)
            if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
                raise AssertionError("weather feature index must be unique and sorted")
            if self.feature_names_ is None:
                self.feature_names_ = tuple(frame.columns)
            elif set(frame.columns) != set(self.feature_names_):
                missing = set(self.feature_names_).difference(frame.columns)
                extra = set(frame.columns).difference(self.feature_names_)
                raise ValueError(
                    f"weather feature schema changed; missing={sorted(missing)}, "
                    f"extra={sorted(extra)}"
                )
            result[group_name] = frame.reindex(columns=self.feature_names_)
        return result

    def _validate_frame(
        self, frame: pd.DataFrame, source: str, *, check_catalog: bool = False
    ) -> None:
        required = {
            TIME_COL,
            AVAILABLE_COL,
            GRID_COL,
            *COORD_COLS,
            *SOURCE_VARIABLES[source],
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{source} frame missing columns: {sorted(missing)}")
        forbidden = [
            column
            for column in frame.columns
            if re.search(r"(^kpx_group_|scada|target|label|power_kw)", column, re.I)
        ]
        if forbidden:
            raise ValueError(f"target/SCADA-like columns are forbidden: {forbidden}")
        if frame.duplicated([TIME_COL, GRID_COL]).any():
            raise ValueError(f"{source} has duplicate forecast/grid rows")
        if frame[[TIME_COL, AVAILABLE_COL, GRID_COL, *COORD_COLS]].isna().any().any():
            raise ValueError(f"{source} has missing key or coordinate values")
        run_counts = frame.groupby(TIME_COL, sort=False)[AVAILABLE_COL].nunique()
        if not run_counts.eq(1).all():
            raise ValueError(f"{source} has multiple availability runs for one forecast hour")
        lead_hours = (
            pd.to_datetime(frame[TIME_COL]) - pd.to_datetime(frame[AVAILABLE_COL])
        ).dt.total_seconds() / 3_600.0
        if (lead_hours < 0).any():
            raise ValueError(f"{source} contains forecasts issued after their target time")
        if check_catalog:
            expected = self.grid_catalog_[source]
            actual = self._grid_catalog(frame)
            if not expected.index.equals(actual.index):
                raise ValueError(f"{source} grid IDs differ from fitted training grid")
            if not np.allclose(expected.to_numpy(), actual.to_numpy(), atol=1e-5):
                raise ValueError(f"{source} grid coordinates changed after fitting")
            counts = frame.groupby(TIME_COL, sort=False)[GRID_COL].nunique()
            if not counts.eq(len(expected)).all():
                raise ValueError(f"{source} does not contain every fitted grid at every hour")

    @staticmethod
    def _grid_catalog(frame: pd.DataFrame) -> pd.DataFrame:
        coordinate_counts = frame.groupby(GRID_COL)[list(COORD_COLS)].nunique()
        if not coordinate_counts.eq(1).all().all():
            raise ValueError("a grid_id maps to more than one coordinate")
        return (
            frame[[GRID_COL, *COORD_COLS]]
            .drop_duplicates(GRID_COL)
            .sort_values(GRID_COL)
            .set_index(GRID_COL)
            .astype("float64")
        )

    def _fill_within_run(self, frame: pd.DataFrame, source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fill only from the same issuance run, then training-only fallbacks."""

        variables = list(SOURCE_VARIABLES[source])
        missing_mask = frame[variables].isna()
        missing_summary = pd.DataFrame(
            {
                f"{source}__raw_missing_cells": missing_mask.sum(axis=1),
                f"{source}__raw_missing_grids": missing_mask.any(axis=1).astype("int8"),
            },
            index=frame.index,
        ).groupby(frame[TIME_COL]).sum()
        if missing_mask.any().any():
            values = frame[variables]
            grouped = frame.groupby([AVAILABLE_COL, GRID_COL], sort=False)[variables]
            previous = grouped.ffill()
            following = grouped.bfill()
            temporal = previous.combine_first(following)
            both = previous.notna() & following.notna()
            temporal = temporal.mask(both, (previous + following) * 0.5)
            values = values.fillna(temporal)
            # Spatial filling is also safe: every grid here belongs to the same
            # forecast target and the same already-issued weather product.
            remaining = values.isna().any(axis=0)
            if remaining.any():
                columns = remaining.index[remaining].tolist()
                spatial = values[columns].groupby(frame[TIME_COL], sort=False).transform("median")
                values.loc[:, columns] = values[columns].fillna(spatial)
            values = values.fillna(self.fill_values_[source]).fillna(0.0)
            frame.loc[:, variables] = values
        return frame, missing_summary

    def _aggregate_source(
        self, raw: pd.DataFrame, source: str
    ) -> tuple[dict[str, pd.DataFrame], pd.Series]:
        variables = list(SOURCE_VARIABLES[source])
        frame = raw.sort_values([AVAILABLE_COL, GRID_COL, TIME_COL]).reset_index(drop=True)
        frame, missing_summary = self._fill_within_run(frame, source)
        available = (
            frame.groupby(TIME_COL, sort=True)[AVAILABLE_COL]
            .first()
            .rename(AVAILABLE_COL)
        )
        catalog = self.grid_catalog_[source]
        grid_ids = catalog.index.to_list()
        times = pd.DatetimeIndex(sorted(frame[TIME_COL].unique()), name=TIME_COL)
        columns = pd.MultiIndex.from_product([variables, grid_ids])
        wide = (
            frame.set_index([TIME_COL, GRID_COL])[variables]
            .unstack(GRID_COL)
            .reindex(index=times, columns=columns)
        )
        values = np.ascontiguousarray(
            wide.to_numpy(dtype=self.config.dtype, copy=False)
        ).reshape(len(times), len(variables), len(grid_ids))
        if np.isnan(values).any():
            raise ValueError(f"{source} still contains NaN after safe imputation")

        common_blocks: list[np.ndarray] = [
            values.mean(axis=2, dtype=np.float32),
            values.std(axis=2, dtype=np.float32),
        ]
        common_columns = [
            *[f"{source}__global_mean__{column}" for column in variables],
            *[f"{source}__global_std__{column}" for column in variables],
        ]
        wind_variables = list(SOURCE_WIDE_VARIABLES[source])
        wind_indices = [variables.index(column) for column in wind_variables]
        common_blocks.extend(
            [values[:, wind_indices, :].min(axis=2), values[:, wind_indices, :].max(axis=2)]
        )
        common_columns.extend(
            [
                *[f"{source}__global_min__{column}" for column in wind_variables],
                *[f"{source}__global_max__{column}" for column in wind_variables],
            ]
        )

        missing_summary = missing_summary.reindex(times, fill_value=0)
        common_blocks.append(missing_summary.to_numpy(dtype=self.config.dtype))
        common_columns.extend(missing_summary.columns.tolist())

        latitudes = catalog["latitude"].to_numpy()
        longitudes = catalog["longitude"].to_numpy()
        result: dict[str, pd.DataFrame] = {}
        for group_name, site in self.group_sites.items():
            distances = _haversine_km(latitudes, longitudes, site)
            order = np.argsort(distances)
            nearest = order[: min(self.config.nearest_k, len(order))]
            wide_nearest = order[: min(self.config.wide_nearest_k, len(order))]
            weights = 1.0 / np.power(
                distances + self.config.idw_distance_floor_km, self.config.idw_power
            )
            weights /= weights.sum()

            blocks = list(common_blocks)
            feature_columns = list(common_columns)
            blocks.extend(
                [
                    np.einsum("tvg,g->tv", values, weights, optimize=True),
                    values[:, :, order[0]],
                    values[:, :, nearest].std(axis=2, dtype=np.float32),
                ]
            )
            feature_columns.extend(
                [
                    *[f"{source}__idw__{column}" for column in variables],
                    *[f"{source}__nearest__{column}" for column in variables],
                    *[
                        f"{source}__near{len(nearest)}_std__{column}"
                        for column in variables
                    ],
                ]
            )
            for rank, grid_position in enumerate(wide_nearest, start=1):
                blocks.append(values[:, wind_indices, grid_position])
                feature_columns.extend(
                    f"{source}__rank{rank}__{column}" for column in wind_variables
                )
            distance_block = np.broadcast_to(
                distances[wide_nearest].astype(self.config.dtype),
                (len(times), len(wide_nearest)),
            )
            blocks.append(distance_block)
            feature_columns.extend(
                f"{source}__rank{rank}_distance_km"
                for rank in range(1, len(wide_nearest) + 1)
            )
            matrix = np.concatenate(blocks, axis=1).astype(self.config.dtype, copy=False)
            result[group_name] = pd.DataFrame(
                matrix, index=times, columns=feature_columns, copy=False
            )
        return result, available.reindex(times)

    @staticmethod
    def _speed_features(
        frame: pd.DataFrame, output: str, u_column: str, v_column: str
    ) -> pd.Series:
        u = frame[u_column].astype("float64", copy=False)
        v = frame[v_column].astype("float64", copy=False)
        speed = pd.Series(np.hypot(u, v), index=frame.index, name=output)
        safe_speed = speed.clip(lower=1e-4)
        frame[output] = speed
        frame[f"{output}__flow_sin"] = u / safe_speed
        frame[f"{output}__flow_cos"] = v / safe_speed
        frame[f"{output}__cube"] = speed.clip(upper=50.0) ** 3
        return speed

    def _add_physics(self, frame: pd.DataFrame, site: GroupSite) -> pd.DataFrame:
        frame = frame.copy()
        for aggregate in ("idw", "nearest"):
            lp = f"ldaps__{aggregate}"
            self._speed_features(
                frame,
                f"{lp}__ws10",
                f"{lp}__heightAboveGround_10_10u",
                f"{lp}__heightAboveGround_10_10v",
            )
            u50 = 0.5 * (
                frame[f"{lp}__heightAboveGround_50_50MUmax"]
                + frame[f"{lp}__heightAboveGround_50_50MUmin"]
            )
            v50 = 0.5 * (
                frame[f"{lp}__heightAboveGround_50_50MVmax"]
                + frame[f"{lp}__heightAboveGround_50_50MVmin"]
            )
            frame[f"{lp}__u50_mid"] = u50
            frame[f"{lp}__v50_mid"] = v50
            ws50 = self._speed_features(
                frame, f"{lp}__ws50", f"{lp}__u50_mid", f"{lp}__v50_mid"
            )
            component_range = np.hypot(
                frame[f"{lp}__heightAboveGround_50_50MUmax"]
                - frame[f"{lp}__heightAboveGround_50_50MUmin"],
                frame[f"{lp}__heightAboveGround_50_50MVmax"]
                - frame[f"{lp}__heightAboveGround_50_50MVmin"],
            )
            frame[f"{lp}__wind_range50"] = component_range
            ws10 = frame[f"{lp}__ws10"]
            alpha = np.log((ws50 + 0.1) / (ws10 + 0.1)) / np.log(5.0)
            alpha = alpha.clip(-0.5, 0.8)
            frame[f"{lp}__shear_alpha_10_50"] = alpha
            frame[f"{lp}__hub_ws"] = ws50 * (site.hub_height_m / 50.0) ** alpha

            gp = f"gfs__{aggregate}"
            ws10_gfs = self._speed_features(
                frame,
                f"{gp}__ws10",
                f"{gp}__heightAboveGround_10_10u",
                f"{gp}__heightAboveGround_10_10v",
            )
            ws100 = self._speed_features(
                frame,
                f"{gp}__ws100",
                f"{gp}__heightAboveGround_100_100u",
                f"{gp}__heightAboveGround_100_100v",
            )
            if aggregate == "idw":
                self._speed_features(
                    frame,
                    f"{gp}__ws80",
                    f"{gp}__heightAboveGround_80_u",
                    f"{gp}__heightAboveGround_80_v",
                )
                self._speed_features(
                    frame,
                    f"{gp}__ws_pbl",
                    f"{gp}__planetaryBoundaryLayer_0_u",
                    f"{gp}__planetaryBoundaryLayer_0_v",
                )
            alpha_gfs = np.log((ws100 + 0.1) / (ws10_gfs + 0.1)) / np.log(10.0)
            alpha_gfs = alpha_gfs.clip(-0.5, 0.8)
            frame[f"{gp}__shear_alpha_10_100"] = alpha_gfs
            frame[f"{gp}__hub_ws"] = ws100 * (site.hub_height_m / 100.0) ** alpha_gfs

        ldaps_temp = frame["ldaps__idw__heightAboveGround_2_t"].clip(180.0, 340.0)
        ldaps_pressure = frame["ldaps__idw__surface_0_sp"].clip(40_000.0, 120_000.0)
        frame["ldaps__idw__air_density"] = ldaps_pressure / (287.05 * ldaps_temp)
        frame["ldaps__idw__wind_power_density"] = (
            0.5
            * frame["ldaps__idw__air_density"]
            * frame["ldaps__idw__hub_ws"].clip(0.0, 50.0) ** 3
        )
        frame["ldaps__idw__dewpoint_depression"] = (
            frame["ldaps__idw__heightAboveGround_2_t"]
            - frame["ldaps__idw__heightAboveGround_2_dpt"]
        )

        gfs_temp = frame["gfs__idw__heightAboveGround_2_2t"].clip(180.0, 340.0)
        gfs_pressure = frame["gfs__idw__surface_0_sp"].clip(40_000.0, 120_000.0)
        frame["gfs__idw__air_density"] = gfs_pressure / (287.05 * gfs_temp)
        frame["gfs__idw__wind_power_density"] = (
            0.5
            * frame["gfs__idw__air_density"]
            * frame["gfs__idw__hub_ws"].clip(0.0, 50.0) ** 3
        )
        frame["gfs__idw__dewpoint_depression"] = (
            frame["gfs__idw__heightAboveGround_2_2t"]
            - frame["gfs__idw__heightAboveGround_2_2d"]
        )
        frame["gfs__idw__gust_excess"] = (
            frame["gfs__idw__surface_0_gust"] - frame["gfs__idw__ws10"]
        )
        frame["gfs__idw__gust_ratio"] = frame["gfs__idw__surface_0_gust"] / (
            frame["gfs__idw__ws10"] + 0.2
        )

        frame["cross__hub_ws_mean"] = 0.5 * (
            frame["ldaps__idw__hub_ws"] + frame["gfs__idw__hub_ws"]
        )
        frame["cross__hub_ws_difference"] = (
            frame["ldaps__idw__hub_ws"] - frame["gfs__idw__hub_ws"]
        )
        frame["cross__ws10_difference"] = (
            frame["ldaps__idw__ws10"] - frame["gfs__idw__ws10"]
        )
        frame["cross__u10_difference"] = (
            frame["ldaps__idw__heightAboveGround_10_10u"]
            - frame["gfs__idw__heightAboveGround_10_10u"]
        )
        frame["cross__v10_difference"] = (
            frame["ldaps__idw__heightAboveGround_10_10v"]
            - frame["gfs__idw__heightAboveGround_10_10v"]
        )
        frame["cross__hub_ws_product"] = (
            frame["ldaps__idw__hub_ws"] * frame["gfs__idw__hub_ws"]
        )
        return frame

    @staticmethod
    def _add_time_features(frame: pd.DataFrame, available: pd.Series) -> pd.DataFrame:
        index = pd.DatetimeIndex(frame.index)
        hour = index.hour.to_numpy(dtype="float32")
        day = index.dayofyear.to_numpy(dtype="float32") - 1.0 + hour / 24.0
        month = index.month.to_numpy(dtype="float32")
        lead = (index - pd.DatetimeIndex(available.reindex(index))).total_seconds() / 3_600.0
        additions = pd.DataFrame(index=index)
        additions["time__hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        additions["time__hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
        additions["time__doy_sin"] = np.sin(2.0 * np.pi * day / 365.2425)
        additions["time__doy_cos"] = np.cos(2.0 * np.pi * day / 365.2425)
        additions["time__month_sin"] = np.sin(2.0 * np.pi * (month - 1.0) / 12.0)
        additions["time__month_cos"] = np.cos(2.0 * np.pi * (month - 1.0) / 12.0)
        additions["time__lead_hours"] = np.asarray(lead, dtype="float32")
        additions["time__run_position"] = np.asarray(lead, dtype="float32") - 12.0
        additions["time__hour_x_doy_sin"] = (
            additions["time__hour_sin"] * additions["time__doy_sin"]
        )
        additions["time__hour_x_doy_cos"] = (
            additions["time__hour_cos"] * additions["time__doy_cos"]
        )
        return pd.concat([frame, additions], axis=1, copy=False)

    def _add_within_run_dynamics(
        self, frame: pd.DataFrame, available: pd.Series
    ) -> pd.DataFrame:
        dynamic_columns = [
            "ldaps__idw__ws10",
            "ldaps__idw__ws50",
            "ldaps__idw__hub_ws",
            "ldaps__idw__wind_power_density",
            "gfs__idw__ws10",
            "gfs__idw__ws80",
            "gfs__idw__ws100",
            "gfs__idw__hub_ws",
            "gfs__idw__surface_0_gust",
            "gfs__idw__wind_power_density",
            "cross__hub_ws_mean",
            "cross__hub_ws_difference",
        ]
        run_key = available.reindex(frame.index)
        additions: dict[str, pd.Series] = {}
        for column in dynamic_columns:
            current = frame[column]
            grouped = current.groupby(run_key, sort=False)
            for offset in self.config.run_offsets:
                lag = grouped.shift(offset).fillna(current)
                lead = grouped.shift(-offset).fillna(current)
                additions[f"{column}__run_lag_{offset}h"] = lag
                additions[f"{column}__run_lead_{offset}h"] = lead
                additions[f"{column}__run_gradient_{offset}h"] = (
                    lead - lag
                ) / (2.0 * offset)
                additions[f"{column}__run_curvature_{offset}h"] = (
                    lead - 2.0 * current + lag
                ) / float(offset * offset)
        return pd.concat([frame, pd.DataFrame(additions, index=frame.index)], axis=1, copy=False)


def build_train_test_weather_features(
    data_dir: str | Path,
    *,
    config: WeatherFeatureConfig | None = None,
    info_path: str | Path | None = None,
) -> tuple[
    dict[str, pd.DataFrame], dict[str, pd.DataFrame], WeatherFeatureBuilder
]:
    """Build aligned group-wise features from the official directory layout.

    Returns ``(train_features, test_features, fitted_builder)``.  The builder
    is returned so its fitted schema and grid catalogue can be persisted with
    the downstream model.
    """

    root = Path(data_dir)
    workbook = Path(info_path) if info_path is not None else root / "info.xlsx"
    sites = load_group_sites(workbook) if workbook.exists() else DEFAULT_GROUP_SITES
    builder = WeatherFeatureBuilder(sites, config=config)
    train = builder.fit_transform(
        root / "train" / "ldaps_train.csv", root / "train" / "gfs_train.csv"
    )
    test = builder.transform(
        root / "test" / "ldaps_test.csv", root / "test" / "gfs_test.csv"
    )
    for group_name in builder.group_sites:
        if tuple(train[group_name].columns) != tuple(test[group_name].columns):
            raise AssertionError(f"train/test schema mismatch for {group_name}")
    return train, test, builder


def stack_group_features(features: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Optionally stack group frames for a single global model.

    The returned MultiIndex levels are ``group`` and ``forecast_kst_dtm``;
    ``site__*`` columns allow the model to distinguish the physical groups.
    """

    if set(features) != set(DEFAULT_GROUP_SITES):
        raise ValueError("features must contain all three KPX groups")
    return pd.concat(features, names=["group", TIME_COL], copy=False)


__all__: Sequence[str] = (
    "DEFAULT_GROUP_SITES",
    "GFS_VARIABLES",
    "GroupSite",
    "LDAPS_VARIABLES",
    "WeatherFeatureBuilder",
    "WeatherFeatureConfig",
    "build_train_test_weather_features",
    "load_group_sites",
    "read_weather_csv",
    "stack_group_features",
)

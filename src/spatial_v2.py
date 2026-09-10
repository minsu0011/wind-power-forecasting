"""Turbine-resolved, leakage-safe spatial augmentation for BARAM 2026.

Unlike the v1 weather features, which interpolate once at a group centroid,
this module interpolates NWP values at every turbine coordinate and then
aggregates them with the turbine nameplate capacities from ``info.xlsx``.
Directional layout and wake proxies use only forecast wind vectors and static
turbine geometry. No label or SCADA column is accepted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .features import (
    AVAILABLE_COL,
    COORD_COLS,
    GRID_COL,
    SOURCE_VARIABLES,
    TIME_COL,
    read_weather_csv,
)
from .metric import CAPACITY_KWH, TARGET_COLS


LDAPS_FIELDS: tuple[str, ...] = (
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
    "surface_0_sp",
    "surface_0_lsm",
    "surface_0_h",
)

GFS_FIELDS: tuple[str, ...] = (
    "heightAboveGround_10_10u",
    "heightAboveGround_10_10v",
    "heightAboveGround_80_u",
    "heightAboveGround_80_v",
    "heightAboveGround_100_100u",
    "heightAboveGround_100_100v",
    "planetaryBoundaryLayer_0_u",
    "planetaryBoundaryLayer_0_v",
    "heightAboveGround_2_2t",
    "heightAboveGround_2_2d",
    "surface_0_sp",
    "surface_0_gust",
    "surface_0_dswrf",
    "surface_0_dlwrf",
    "atmosphere_0_tcc",
)

SOURCE_FIELDS = {"ldaps": LDAPS_FIELDS, "gfs": GFS_FIELDS}


@dataclass(frozen=True)
class SpatialV2Config:
    idw_power: float = 2.0
    distance_floor_km: float = 0.05
    wake_expansion: float = 0.05
    wake_loss_scale: float = 0.08
    run_offset: int = 1
    sequence_offsets: tuple[int, ...] = (6, 12)
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.idw_power <= 0 or self.distance_floor_km <= 0:
            raise ValueError("IDW parameters must be positive")
        if self.wake_expansion <= 0 or self.wake_loss_scale <= 0:
            raise ValueError("wake parameters must be positive")
        if self.run_offset <= 0:
            raise ValueError("run_offset must be positive")
        if not self.sequence_offsets or any(
            offset <= 0 or offset >= 24 for offset in self.sequence_offsets
        ):
            raise ValueError("sequence_offsets must be unique hours between 1 and 23")
        if len(set(self.sequence_offsets)) != len(self.sequence_offsets):
            raise ValueError("sequence_offsets must be unique")


WeatherInput = str | Path | pd.DataFrame


def _parse_dms(value: object) -> tuple[float, float]:
    tokens = re.findall(r"[\d.]+|[NSEW]", str(value).upper())
    if len(tokens) != 8:
        raise ValueError(f"cannot parse turbine coordinate: {value!r}")
    lat_d, lat_m, lat_s, ns, lon_d, lon_m, lon_s, ew = tokens
    latitude = float(lat_d) + float(lat_m) / 60.0 + float(lat_s) / 3_600.0
    longitude = float(lon_d) + float(lon_m) / 60.0 + float(lon_s) / 3_600.0
    return (-latitude if ns == "S" else latitude), (-longitude if ew == "W" else longitude)


def load_turbine_table(info_path: str | Path) -> pd.DataFrame:
    """Return one validated row per turbine with explicit group membership."""

    raw = pd.read_excel(info_path, sheet_name="info", header=None)
    header_rows = raw.index[
        raw.apply(lambda row: row.astype(str).eq("KPX그룹").any(), axis=1)
    ]
    if len(header_rows) != 1:
        raise ValueError("info.xlsx must contain exactly one KPX그룹 header")
    header_row = int(header_rows[0])
    table = raw.iloc[header_row + 1 :].copy()
    table.columns = raw.iloc[header_row].tolist()
    required = {
        "제작사",
        "호기",
        "좌표(Google)",
        "KPX그룹",
        "Hub Height(m)",
        "Rotor Diameter(m)",
        "설비용량(MW)",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"info.xlsx missing turbine fields: {sorted(missing)}")
    table = table.loc[table["좌표(Google)"].notna()].copy()
    table["KPX그룹"] = pd.to_numeric(table["KPX그룹"], errors="coerce").ffill()
    coordinates = table["좌표(Google)"].map(_parse_dms)
    result = pd.DataFrame(
        {
            "group": table["KPX그룹"].astype(int).map(lambda value: f"kpx_group_{value}"),
            "manufacturer": table["제작사"].astype(str).str.lower(),
            "turbine_number": pd.to_numeric(table["호기"], errors="raise").astype(int),
            "latitude": coordinates.map(lambda pair: pair[0]).astype(float),
            "longitude": coordinates.map(lambda pair: pair[1]).astype(float),
            "hub_height_m": pd.to_numeric(table["Hub Height(m)"], errors="raise"),
            "rotor_diameter_m": pd.to_numeric(
                table["Rotor Diameter(m)"], errors="raise"
            ),
            "capacity_mw": pd.to_numeric(table["설비용량(MW)"], errors="raise"),
        }
    ).reset_index(drop=True)
    if result.isna().any().any() or result.duplicated(
        ["manufacturer", "turbine_number"]
    ).any():
        raise ValueError("turbine metadata is incomplete or duplicated")
    if set(result["group"]) != set(TARGET_COLS):
        raise ValueError("turbine metadata does not cover all target groups")
    observed_capacity = result.groupby("group")["capacity_mw"].sum() * 1_000.0
    for group in TARGET_COLS:
        if not np.isclose(observed_capacity[group], CAPACITY_KWH[group], atol=1e-6):
            raise ValueError(
                f"{group} turbine capacity {observed_capacity[group]} != target capacity"
            )
    return result


def _haversine_matrix_km(
    turbine_lat: np.ndarray,
    turbine_lon: np.ndarray,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
) -> np.ndarray:
    lat1 = np.radians(turbine_lat.astype(float))[:, None]
    lat2 = np.radians(grid_lat.astype(float))[None, :]
    delta_lat = lat2 - lat1
    delta_lon = np.radians(grid_lon[None, :] - turbine_lon[:, None])
    hav = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2.0) ** 2
    )
    return 6_371.0088 * 2.0 * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))


def _grid_spacing_km(catalog: pd.DataFrame) -> float:
    """Median nearest-neighbour grid spacing, excluding each grid itself."""

    distances = _haversine_matrix_km(
        catalog["latitude"].to_numpy(dtype=float),
        catalog["longitude"].to_numpy(dtype=float),
        catalog["latitude"].to_numpy(dtype=float),
        catalog["longitude"].to_numpy(dtype=float),
    )
    np.fill_diagonal(distances, np.inf)
    return float(np.median(distances.min(axis=1)))


def _local_xy_km(turbines: pd.DataFrame, capacity_weights: np.ndarray) -> np.ndarray:
    latitude0 = float(np.dot(turbines["latitude"], capacity_weights))
    longitude0 = float(np.dot(turbines["longitude"], capacity_weights))
    x = (
        (turbines["longitude"].to_numpy(dtype=float) - longitude0)
        * 111.320
        * np.cos(np.radians(latitude0))
    )
    y = (turbines["latitude"].to_numpy(dtype=float) - latitude0) * 110.574
    return np.column_stack([x, y])


def _weighted_statistics(values: np.ndarray, weights: np.ndarray) -> dict[str, np.ndarray]:
    mean = values @ weights
    variance = np.maximum((values * values) @ weights - mean * mean, 0.0)
    return {
        "cw_mean": mean,
        "cw_std": np.sqrt(variance),
        "range": values.max(axis=1) - values.min(axis=1),
    }


def _wake_layout_features(
    u: np.ndarray,
    v: np.ndarray,
    positions_xy: np.ndarray,
    capacity_weights: np.ndarray,
    rotor_diameter_m: float,
    config: SpatialV2Config,
) -> dict[str, np.ndarray]:
    """Directional geometric proxies; no observed power is used."""

    mean_u = u @ capacity_weights
    mean_v = v @ capacity_weights
    mean_speed = np.hypot(mean_u, mean_v)
    unit_x = np.divide(mean_u, mean_speed, out=np.zeros_like(mean_u), where=mean_speed > 1e-5)
    unit_y = np.divide(mean_v, mean_speed, out=np.zeros_like(mean_v), where=mean_speed > 1e-5)
    projection = (
        unit_x[:, None] * positions_xy[None, :, 0]
        + unit_y[:, None] * positions_xy[None, :, 1]
    )
    cross_projection = (
        -unit_y[:, None] * positions_xy[None, :, 0]
        + unit_x[:, None] * positions_xy[None, :, 1]
    )
    speed = np.hypot(u, v)
    upstream_index = projection.argmin(axis=1)
    downstream_index = projection.argmax(axis=1)
    row_index = np.arange(len(speed))
    upstream_speed = speed[row_index, upstream_index]
    downstream_speed = speed[row_index, downstream_index]

    pairs = [(i, j) for i in range(len(positions_xy)) for j in range(len(positions_xy)) if i != j]
    target_exposure = np.zeros_like(speed, dtype="float64")
    pair_exposure: list[np.ndarray] = []
    rotor_km = rotor_diameter_m / 1_000.0
    for upstream, downstream in pairs:
        delta = positions_xy[downstream] - positions_xy[upstream]
        along = unit_x * delta[0] + unit_y * delta[1]
        cross = np.abs(-unit_y * delta[0] + unit_x * delta[1])
        positive = np.maximum(along, 0.0)
        wake_radius = 0.5 * rotor_km + config.wake_expansion * positive
        exposure = np.where(
            along > 0.0,
            np.exp(-np.square(cross / np.maximum(wake_radius, 1e-6)))
            / np.square(1.0 + config.wake_expansion * positive / rotor_km),
            0.0,
        )
        target_exposure[:, downstream] += exposure
        pair_exposure.append(exposure)
    exposure_matrix = np.column_stack(pair_exposure)
    weighted_exposure = target_exposure @ capacity_weights
    effective_speed = np.maximum(
        speed * (1.0 - config.wake_loss_scale * target_exposure), 0.0
    )
    raw_power_proxy = np.maximum(speed, 0.0) ** 3 @ capacity_weights
    effective_power_proxy = effective_speed**3 @ capacity_weights
    power_loss = np.divide(
        raw_power_proxy - effective_power_proxy,
        np.maximum(raw_power_proxy, 1e-6),
    )
    centered_projection = projection - (projection @ capacity_weights)[:, None]
    centered_speed = speed - (speed @ capacity_weights)[:, None]
    covariance = (centered_projection * centered_speed) @ capacity_weights
    denominator = np.sqrt(
        np.maximum((centered_projection**2) @ capacity_weights, 1e-8)
        * np.maximum((centered_speed**2) @ capacity_weights, 1e-8)
    )
    return {
        "flow_speed": mean_speed,
        "downwind_span_km": projection.max(axis=1) - projection.min(axis=1),
        "crosswind_span_km": cross_projection.max(axis=1) - cross_projection.min(axis=1),
        "upstream_edge_ws": upstream_speed,
        "downstream_edge_ws": downstream_speed,
        "downstream_minus_upstream_ws": downstream_speed - upstream_speed,
        "projection_ws_correlation": covariance / denominator,
        "wake_pair_mean": exposure_matrix.mean(axis=1),
        "wake_pair_max": exposure_matrix.max(axis=1),
        "wake_active_fraction": (exposure_matrix > 0.10).mean(axis=1),
        "wake_receiver_cw_mean": weighted_exposure,
        "wake_receiver_max": target_exposure.max(axis=1),
        "wake_effective_ws_cw_mean": effective_speed @ capacity_weights,
        "wake_power_loss_proxy": power_loss,
    }


class SpatialV2Builder:
    """Training-fitted transformer for turbine-level NWP augmentation."""

    def __init__(self, turbines: pd.DataFrame, config: SpatialV2Config | None = None):
        self.turbines = turbines.copy()
        self.config = config or SpatialV2Config()
        self.fill_values_: dict[str, pd.Series] = {}
        self.catalog_: dict[str, pd.DataFrame] = {}
        self.feature_names_: tuple[str, ...] | None = None

    def fit(self, ldaps: WeatherInput, gfs: WeatherInput) -> "SpatialV2Builder":
        for source, data in (("ldaps", ldaps), ("gfs", gfs)):
            frame = self._as_frame(data, source)
            self._validate(frame, source)
            fields = list(SOURCE_FIELDS[source])
            self.fill_values_[source] = frame[fields].median().fillna(0.0)
            self.catalog_[source] = self._catalog(frame)
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
        if set(self.fill_values_) != {"ldaps", "gfs"}:
            raise RuntimeError("fit must be called before transform")
        source_frames: dict[str, dict[str, pd.DataFrame]] = {}
        available: dict[str, pd.Series] = {}
        for source, data in (("ldaps", ldaps), ("gfs", gfs)):
            raw = self._as_frame(data, source)
            self._validate(raw, source, check_catalog=True)
            source_frames[source], available[source] = self._aggregate(raw, source)
        first = source_frames["ldaps"][TARGET_COLS[0]].index
        if not first.equals(source_frames["gfs"][TARGET_COLS[0]].index):
            raise ValueError("LDAPS/GFS timestamps differ")
        if not available["ldaps"].equals(available["gfs"]):
            raise ValueError("LDAPS/GFS availability runs differ")

        result: dict[str, pd.DataFrame] = {}
        for group in TARGET_COLS:
            frame = pd.concat(
                [source_frames["ldaps"][group], source_frames["gfs"][group]], axis=1
            )
            frame["spv2__crossres_nearest_distance_ratio"] = frame[
                "spv2__gfs__nearest_distance_cw_mean_km"
            ] / (frame["spv2__ldaps__nearest_distance_cw_mean_km"] + 1e-3)
            frame["spv2__crossres_ws10_mean_diff"] = (
                frame["spv2__ldaps__idw_cw_mean__ws10"]
                - frame["spv2__gfs__idw_cw_mean__ws10"]
            )
            frame["spv2__crossres_high_ws_mean_diff"] = (
                frame["spv2__ldaps__idw_cw_mean__ws50"]
                - frame["spv2__gfs__idw_cw_mean__ws100"]
            )
            frame["spv2__crossres_wake_exposure_diff"] = (
                frame["spv2__ldaps__wake_receiver_cw_mean"]
                - frame["spv2__gfs__wake_receiver_cw_mean"]
            )
            frame = self._add_run_dynamics(frame, available["ldaps"])
            frame = self._add_run_sequence_features(frame, available["ldaps"])
            frame = frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            frame = frame.astype(self.config.dtype)
            if not np.isfinite(frame.to_numpy()).all():
                raise AssertionError("spatial v2 features contain non-finite values")
            if self.feature_names_ is None:
                self.feature_names_ = tuple(frame.columns)
            elif tuple(frame.columns) != self.feature_names_:
                raise ValueError("spatial v2 feature schema changed")
            result[group] = frame
        return result

    @staticmethod
    def _as_frame(data: WeatherInput, source: str) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            frame = data.copy()
            frame[TIME_COL] = pd.to_datetime(frame[TIME_COL], errors="raise")
            frame[AVAILABLE_COL] = pd.to_datetime(frame[AVAILABLE_COL], errors="raise")
            return frame
        return read_weather_csv(data, source)

    def _validate(self, frame: pd.DataFrame, source: str, *, check_catalog: bool = False) -> None:
        required = {
            TIME_COL,
            AVAILABLE_COL,
            GRID_COL,
            *COORD_COLS,
            *SOURCE_FIELDS[source],
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{source} missing spatial-v2 fields: {sorted(missing)}")
        forbidden = [
            column
            for column in frame.columns
            if re.search(r"(^kpx_group_|scada|target|label|power_kw)", column, re.I)
        ]
        if forbidden:
            raise ValueError(f"target/SCADA-like columns are forbidden: {forbidden}")
        if frame.duplicated([TIME_COL, GRID_COL]).any():
            raise ValueError(f"{source} duplicate forecast/grid rows")
        if frame[[TIME_COL, AVAILABLE_COL, GRID_COL, *COORD_COLS]].isna().any().any():
            raise ValueError(f"{source} missing keys/coordinates")
        if not frame.groupby(TIME_COL)[AVAILABLE_COL].nunique().eq(1).all():
            raise ValueError(f"{source} has multiple runs for one forecast time")
        lead_hours = (
            frame[TIME_COL] - frame[AVAILABLE_COL]
        ).dt.total_seconds().to_numpy(dtype=float) / 3_600.0
        if not np.isfinite(lead_hours).all() or np.any(lead_hours < 0.0):
            raise ValueError(f"{source} contains forecasts available after forecast time")
        if check_catalog:
            actual = self._catalog(frame)
            expected = self.catalog_[source]
            if not actual.index.equals(expected.index) or not np.allclose(
                actual.to_numpy(), expected.to_numpy(), atol=1e-5
            ):
                raise ValueError(f"{source} grid geometry differs from fit")
            counts = frame.groupby(TIME_COL)[GRID_COL].nunique()
            if not counts.eq(len(expected)).all():
                raise ValueError(f"{source} missing fitted grids at some forecast hours")

    @staticmethod
    def _catalog(frame: pd.DataFrame) -> pd.DataFrame:
        counts = frame.groupby(GRID_COL)[list(COORD_COLS)].nunique()
        if not counts.eq(1).all().all():
            raise ValueError("grid_id maps to multiple coordinates")
        return (
            frame[[GRID_COL, *COORD_COLS]]
            .drop_duplicates(GRID_COL)
            .sort_values(GRID_COL)
            .set_index(GRID_COL)
            .astype(float)
        )

    def _safe_fill(self, frame: pd.DataFrame, source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        fields = list(SOURCE_FIELDS[source])
        missing = frame[fields].isna()
        summary = pd.DataFrame(
            {
                f"spv2__{source}__raw_missing_cells": missing.sum(axis=1),
                f"spv2__{source}__raw_missing_grids": missing.any(axis=1).astype(int),
            },
            index=frame.index,
        ).groupby(frame[TIME_COL]).sum()
        if missing.any().any():
            values = frame[fields]
            grouped = frame.groupby([AVAILABLE_COL, GRID_COL], sort=False)[fields]
            previous, following = grouped.ffill(), grouped.bfill()
            temporal = previous.combine_first(following)
            both = previous.notna() & following.notna()
            temporal = temporal.mask(both, 0.5 * (previous + following))
            values = values.fillna(temporal)
            remaining = values.isna().any(axis=0)
            if remaining.any():
                columns = remaining.index[remaining].tolist()
                spatial = values[columns].groupby(frame[TIME_COL]).transform("median")
                values.loc[:, columns] = values[columns].fillna(spatial)
            frame.loc[:, fields] = values.fillna(self.fill_values_[source]).fillna(0.0)
        return frame, summary

    def _aggregate(
        self, raw: pd.DataFrame, source: str
    ) -> tuple[dict[str, pd.DataFrame], pd.Series]:
        fields = list(SOURCE_FIELDS[source])
        frame = raw.sort_values([AVAILABLE_COL, GRID_COL, TIME_COL]).reset_index(drop=True)
        frame, missing_summary = self._safe_fill(frame, source)
        available = frame.groupby(TIME_COL, sort=True)[AVAILABLE_COL].first()
        times = pd.DatetimeIndex(sorted(frame[TIME_COL].unique()), name=TIME_COL)
        catalog = self.catalog_[source]
        grid_ids = catalog.index.to_list()
        columns = pd.MultiIndex.from_product([fields, grid_ids])
        wide = (
            frame.set_index([TIME_COL, GRID_COL])[fields]
            .unstack(GRID_COL)
            .reindex(index=times, columns=columns)
        )
        tensor = np.ascontiguousarray(wide.to_numpy(dtype="float32")).reshape(
            len(times), len(fields), len(grid_ids)
        )
        if np.isnan(tensor).any():
            raise ValueError(f"{source} still contains missing values")

        grid_lat = catalog["latitude"].to_numpy(dtype=float)
        grid_lon = catalog["longitude"].to_numpy(dtype=float)
        grid_spacing = _grid_spacing_km(catalog)
        result: dict[str, pd.DataFrame] = {}
        for group in TARGET_COLS:
            turbines = self.turbines.loc[self.turbines["group"] == group].reset_index(drop=True)
            capacity = turbines["capacity_mw"].to_numpy(dtype=float)
            capacity_weights = capacity / capacity.sum()
            distances = _haversine_matrix_km(
                turbines["latitude"].to_numpy(dtype=float),
                turbines["longitude"].to_numpy(dtype=float),
                grid_lat,
                grid_lon,
            )
            weights = 1.0 / np.power(
                distances + self.config.distance_floor_km, self.config.idw_power
            )
            weights /= weights.sum(axis=1, keepdims=True)
            nearest_index = distances.argmin(axis=1)
            interpolated = np.einsum("tvg,ng->tvn", tensor, weights, optimize=True)
            nearest = tensor[:, :, nearest_index]
            idw_fields = {
                field: interpolated[:, position, :]
                for position, field in enumerate(fields)
            }
            nearest_fields = {
                field: nearest[:, position, :] for position, field in enumerate(fields)
            }
            if source == "ldaps":
                for mapping in (idw_fields, nearest_fields):
                    mapping["u50_mid"] = 0.5 * (
                        mapping["heightAboveGround_50_50MUmax"]
                        + mapping["heightAboveGround_50_50MUmin"]
                    )
                    mapping["v50_mid"] = 0.5 * (
                        mapping["heightAboveGround_50_50MVmax"]
                        + mapping["heightAboveGround_50_50MVmin"]
                    )
                    mapping["ws10"] = np.hypot(
                        mapping["heightAboveGround_10_10u"],
                        mapping["heightAboveGround_10_10v"],
                    )
                    mapping["ws50"] = np.hypot(mapping["u50_mid"], mapping["v50_mid"])
                    mapping["ws_xbl"] = np.hypot(
                        mapping["heightAboveGround_5_XBLWS"],
                        mapping["heightAboveGround_5_YBLWS"],
                    )
                high_u, high_v, high_ws = "u50_mid", "v50_mid", "ws50"
            else:
                pairs = {
                    "ws10": ("heightAboveGround_10_10u", "heightAboveGround_10_10v"),
                    "ws80": ("heightAboveGround_80_u", "heightAboveGround_80_v"),
                    "ws100": (
                        "heightAboveGround_100_100u",
                        "heightAboveGround_100_100v",
                    ),
                    "ws_pbl": (
                        "planetaryBoundaryLayer_0_u",
                        "planetaryBoundaryLayer_0_v",
                    ),
                }
                for mapping in (idw_fields, nearest_fields):
                    for name, (u_name, v_name) in pairs.items():
                        mapping[name] = np.hypot(mapping[u_name], mapping[v_name])
                high_u, high_v, high_ws = (
                    "heightAboveGround_100_100u",
                    "heightAboveGround_100_100v",
                    "ws100",
                )

            hub_height = float(
                np.average(turbines["hub_height_m"], weights=capacity)
            )
            for mapping in (idw_fields, nearest_fields):
                if source == "ldaps":
                    shear = np.log(
                        np.maximum(mapping["ws50"], 0.2)
                        / np.maximum(mapping["ws10"], 0.2)
                    ) / np.log(50.0 / 10.0)
                    shear = np.clip(shear, -0.30, 0.80)
                    mapping["hub_ws"] = mapping["ws50"] * np.power(
                        hub_height / 50.0, shear
                    )
                    temperature = mapping["heightAboveGround_2_t"]
                else:
                    shear = np.log(
                        np.maximum(mapping["ws100"], 0.2)
                        / np.maximum(mapping["ws80"], 0.2)
                    ) / np.log(100.0 / 80.0)
                    shear = np.clip(shear, -0.30, 0.80)
                    mapping["hub_ws"] = mapping["ws100"] * np.power(
                        hub_height / 100.0, shear
                    )
                    temperature = mapping["heightAboveGround_2_2t"]
                mapping["air_density"] = np.clip(
                    mapping["surface_0_sp"]
                    / (287.05 * np.maximum(temperature, 180.0)),
                    0.7,
                    1.5,
                )
                mapping["wpd"] = (
                    0.5 * mapping["air_density"] * np.maximum(mapping["hub_ws"], 0.0) ** 3
                )

            feature_data: dict[str, np.ndarray] = {}
            for field in idw_fields:
                idw_stats = _weighted_statistics(idw_fields[field], capacity_weights)
                nearest_stats = _weighted_statistics(nearest_fields[field], capacity_weights)
                bias_stats = _weighted_statistics(
                    idw_fields[field] - nearest_fields[field], capacity_weights
                )
                for stat, values in idw_stats.items():
                    feature_data[f"spv2__{source}__idw_{stat}__{field}"] = values
                for stat in ("cw_mean", "cw_std"):
                    feature_data[f"spv2__{source}__nearest_{stat}__{field}"] = nearest_stats[stat]
                    feature_data[f"spv2__{source}__idw_minus_nearest_{stat}__{field}"] = bias_stats[stat]
            for speed_name in ("ws10", high_ws):
                feature_data[f"spv2__{source}__idw_cw_mean__{speed_name}_cubed"] = (
                    idw_fields[speed_name] ** 3 @ capacity_weights
                )
            if source == "ldaps":
                land_idw = idw_fields["surface_0_lsm"]
                land_nearest = nearest_fields["surface_0_lsm"]
                terrain_idw = idw_fields["surface_0_h"]
                terrain_nearest = nearest_fields["surface_0_h"]
                feature_data[f"spv2__{source}__land_fraction_idw_cw_mean"] = (
                    land_idw @ capacity_weights
                )
                feature_data[f"spv2__{source}__land_fraction_nearest_cw_mean"] = (
                    land_nearest @ capacity_weights
                )
                feature_data[f"spv2__{source}__land_sea_interp_abs_bias"] = (
                    np.abs(land_idw - land_nearest) @ capacity_weights
                )
                feature_data[f"spv2__{source}__terrain_interp_bias_cw_mean_m"] = (
                    (terrain_idw - terrain_nearest) @ capacity_weights
                )

            positions = _local_xy_km(turbines, capacity_weights)
            wake = _wake_layout_features(
                idw_fields[high_u],
                idw_fields[high_v],
                positions,
                capacity_weights,
                float(np.average(turbines["rotor_diameter_m"], weights=capacity)),
                self.config,
            )
            for name, values in wake.items():
                feature_data[f"spv2__{source}__{name}"] = values
            nearest_distance = distances[np.arange(len(turbines)), nearest_index]
            effective_distance = (weights * distances).sum(axis=1)
            constants = {
                "nearest_distance_cw_mean_km": float(nearest_distance @ capacity_weights),
                "nearest_distance_max_km": float(nearest_distance.max()),
                "nearest_distance_std_km": float(
                    np.sqrt(
                        np.maximum(
                            (nearest_distance**2) @ capacity_weights
                            - (nearest_distance @ capacity_weights) ** 2,
                            0.0,
                        )
                    )
                ),
                "idw_effective_distance_cw_mean_km": float(
                    effective_distance @ capacity_weights
                ),
                "grid_spacing_median_km": grid_spacing,
                "nearest_distance_over_grid_spacing": float(
                    (nearest_distance @ capacity_weights) / max(grid_spacing, 1e-6)
                ),
                "turbine_count": float(len(turbines)),
                "capacity_mw": float(capacity.sum()),
                "layout_extent_km": float(
                    np.max(
                        np.sqrt(
                            np.sum(
                                (positions[:, None, :] - positions[None, :, :]) ** 2,
                                axis=2,
                            )
                        )
                    )
                ),
            }
            for name, value in constants.items():
                feature_data[f"spv2__{source}__{name}"] = np.full(len(times), value)
            missing = missing_summary.reindex(times, fill_value=0)
            for column in missing:
                feature_data[column] = missing[column].to_numpy(dtype=float)
            result[group] = pd.DataFrame(feature_data, index=times)
        return result, available.reindex(times)

    def _add_run_dynamics(self, frame: pd.DataFrame, available: pd.Series) -> pd.DataFrame:
        columns = [
            "spv2__ldaps__idw_cw_mean__ws50",
            "spv2__ldaps__wake_receiver_cw_mean",
            "spv2__ldaps__wake_effective_ws_cw_mean",
            "spv2__gfs__idw_cw_mean__ws100",
            "spv2__gfs__wake_receiver_cw_mean",
            "spv2__gfs__wake_effective_ws_cw_mean",
            "spv2__crossres_high_ws_mean_diff",
        ]
        additions: dict[str, pd.Series] = {}
        offset = self.config.run_offset
        run_key = available.reindex(frame.index)
        self._assert_run_index(frame.index, run_key)
        for column in columns:
            current = frame[column]
            grouped = current.groupby(run_key, sort=False)
            lag = grouped.shift(offset).fillna(current)
            lead = grouped.shift(-offset).fillna(current)
            additions[f"{column}__run_lag_{offset}h"] = lag
            additions[f"{column}__run_lead_{offset}h"] = lead
            additions[f"{column}__run_gradient_{offset}h"] = (
                lead - lag
            ) / (2.0 * offset)
        return pd.concat([frame, pd.DataFrame(additions, index=frame.index)], axis=1)

    @staticmethod
    def _assert_run_index(index: pd.DatetimeIndex, run_key: pd.Series) -> None:
        if run_key.isna().any():
            raise ValueError("data_available run key is missing after time alignment")
        if int(run_key.ne(run_key.shift()).sum()) != int(run_key.nunique()):
            raise ValueError("a data_available run is split into non-contiguous blocks")
        timestamp = pd.Series(index, index=index)
        within_run_delta = timestamp.groupby(run_key, sort=False).diff().dropna()
        if not within_run_delta.eq(pd.Timedelta(hours=1)).all():
            raise ValueError("forecast horizons inside a run must be hourly and ordered")

    def _add_run_sequence_features(
        self, frame: pd.DataFrame, available: pd.Series
    ) -> pd.DataFrame:
        """Add bounded 24-h regime/ramp summaries for a small physical signal set."""

        signals = {
            "ldaps_hub_ws": "spv2__ldaps__idw_cw_mean__hub_ws",
            "ldaps_wpd": "spv2__ldaps__idw_cw_mean__wpd",
            "gfs_hub_ws": "spv2__gfs__idw_cw_mean__hub_ws",
            "gfs_wpd": "spv2__gfs__idw_cw_mean__wpd",
            "gfs_gust": "spv2__gfs__idw_cw_mean__surface_0_gust",
        }
        run_key = available.reindex(frame.index)
        self._assert_run_index(frame.index, run_key)
        timestamps = pd.Series(
            frame.index.view("int64") / 3_600_000_000_000.0,
            index=frame.index,
        )
        additions: dict[str, pd.Series] = {}
        for short_name, column in signals.items():
            current = frame[column]
            grouped = current.groupby(run_key, sort=False)
            run_mean = grouped.transform("mean")
            run_std = grouped.transform("std").fillna(0.0)
            run_min = grouped.transform("min")
            run_max = grouped.transform("max")
            prefix = f"spv2seq__{short_name}"
            additions[f"{prefix}__run_mean"] = run_mean
            additions[f"{prefix}__run_std"] = run_std
            additions[f"{prefix}__run_min"] = run_min
            additions[f"{prefix}__run_max"] = run_max
            additions[f"{prefix}__run_range"] = run_max - run_min
            additions[f"{prefix}__current_minus_run_mean"] = current - run_mean

            time_grouped = timestamps.groupby(run_key, sort=False)
            run_first = grouped.transform("first")
            run_last = grouped.transform("last")
            time_first = time_grouped.transform("first")
            time_last = time_grouped.transform("last")
            for offset in self.config.sequence_offsets:
                lag = grouped.shift(offset).fillna(run_first)
                lead = grouped.shift(-offset).fillna(run_last)
                lag_time = time_grouped.shift(offset).fillna(time_first)
                lead_time = time_grouped.shift(-offset).fillna(time_last)
                elapsed = (lead_time - lag_time).clip(lower=1.0)
                additions[f"{prefix}__run_lag_{offset}h"] = lag
                additions[f"{prefix}__run_lead_{offset}h"] = lead
                additions[f"{prefix}__run_ramp_{offset}h"] = (lead - lag) / elapsed
        return pd.concat([frame, pd.DataFrame(additions, index=frame.index)], axis=1)


def build_spatial_v2_train_test(
    data_dir: str | Path, config: SpatialV2Config | None = None
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], SpatialV2Builder]:
    root = Path(data_dir)
    turbines = load_turbine_table(root / "info.xlsx")
    builder = SpatialV2Builder(turbines, config=config)
    train = builder.fit_transform(
        root / "train" / "ldaps_train.csv", root / "train" / "gfs_train.csv"
    )
    test = builder.transform(
        root / "test" / "ldaps_test.csv", root / "test" / "gfs_test.csv"
    )
    for group in TARGET_COLS:
        if tuple(train[group].columns) != tuple(test[group].columns):
            raise AssertionError(f"spatial-v2 train/test schema mismatch for {group}")
    return train, test, builder


__all__: Sequence[str] = (
    "GFS_FIELDS",
    "LDAPS_FIELDS",
    "SOURCE_FIELDS",
    "SpatialV2Builder",
    "SpatialV2Config",
    "build_spatial_v2_train_test",
    "load_turbine_table",
)

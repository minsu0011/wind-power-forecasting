"""Fixed Copernicus DEM directional-exposure features.

The scientific choices in this module are bound by
``configs/copernicus_dem_directional_exposure_paired_increment_preregister_v1.json``.
No labels or predictions are accepted by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image


EARTH_RADIUS_M = 6_371_008.8
PIXELS_PER_DEGREE = 3600
SECTOR_CENTRES = np.arange(16, dtype=np.float64) * 22.5
RADII_M = (500.0, 1000.0, 2000.0, 5000.0)
MIN_HORIZON_DISTANCE_M = 100.0
SECTOR_HALF_WIDTH_DEG = 11.25

U_MAX = "ldaps__idw__heightAboveGround_50_50MUmax"
U_MIN = "ldaps__idw__heightAboveGround_50_50MUmin"
V_MAX = "ldaps__idw__heightAboveGround_50_50MVmax"
V_MIN = "ldaps__idw__heightAboveGround_50_50MVmin"
WIND_COLUMNS = (U_MAX, U_MIN, V_MAX, V_MIN)

TERRAIN_VALUE_COLUMNS = (
    "signed_slope_alignment",
    "horizon_angle_r0500m",
    "horizon_angle_r1000m",
    "horizon_angle_r2000m",
    "horizon_angle_r5000m",
)
EXTENDED_COLUMNS = tuple(
    f"copdem__{name}_cw_{stat}"
    for name in TERRAIN_VALUE_COLUMNS
    for stat in ("mean", "std")
)


@dataclass(frozen=True)
class Turbine:
    turbine_id: int
    group: str
    capacity_mw: float
    latitude: float
    longitude: float


@dataclass(frozen=True)
class DemTile:
    """One fixed one-degree 3600x3600 north-up Copernicus DEM tile."""

    west: int
    south: int
    values: np.ndarray

    def __post_init__(self) -> None:
        if self.values.shape != (PIXELS_PER_DEGREE, PIXELS_PER_DEGREE):
            raise ValueError(f"unexpected DEM shape: {self.values.shape}")
        if self.values.dtype != np.float64:
            raise TypeError("DEM arrays must be float64")

    @property
    def north(self) -> float:
        return float(self.south + 1)

    def pixel_centres(self, rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lat = self.north - (rows.astype(np.float64) + 0.5) / PIXELS_PER_DEGREE
        lon = float(self.west) + (cols.astype(np.float64) + 0.5) / PIXELS_PER_DEGREE
        return lat, lon

    def bilinear(self, latitude: float, longitude: float) -> float:
        x = (longitude - float(self.west)) * PIXELS_PER_DEGREE - 0.5
        y = (self.north - latitude) * PIXELS_PER_DEGREE - 0.5
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        if x0 < 0 or y0 < 0 or x0 + 1 >= PIXELS_PER_DEGREE or y0 + 1 >= PIXELS_PER_DEGREE:
            raise ValueError("bilinear coordinate is outside tile interior")
        dx, dy = x - x0, y - y0
        z = self.values[y0 : y0 + 2, x0 : x0 + 2]
        if not np.isfinite(z).all():
            raise ValueError("nonfinite DEM cell in turbine interpolation")
        return float(
            (1.0 - dx) * (1.0 - dy) * z[0, 0]
            + dx * (1.0 - dy) * z[0, 1]
            + (1.0 - dx) * dy * z[1, 0]
            + dx * dy * z[1, 1]
        )

    def nearest_gradient(self, latitude: float, longitude: float) -> tuple[float, float]:
        col = int(np.floor((longitude - float(self.west)) * PIXELS_PER_DEGREE))
        row = int(np.floor((self.north - latitude) * PIXELS_PER_DEGREE))
        if col < 1 or row < 1 or col + 1 >= PIXELS_PER_DEGREE or row + 1 >= PIXELS_PER_DEGREE:
            raise ValueError("gradient coordinate is outside tile interior")
        west, east = self.values[row, col - 1], self.values[row, col + 1]
        north, south = self.values[row - 1, col], self.values[row + 1, col]
        if not np.isfinite((west, east, north, south)).all():
            raise ValueError("nonfinite DEM cell in local gradient")
        lat_rad = np.deg2rad(latitude)
        metres_per_lon_degree = np.pi * EARTH_RADIUS_M * np.cos(lat_rad) / 180.0
        metres_per_lat_degree = np.pi * EARTH_RADIUS_M / 180.0
        dx = 2.0 * metres_per_lon_degree / PIXELS_PER_DEGREE
        dy = 2.0 * metres_per_lat_degree / PIXELS_PER_DEGREE
        return float((east - west) / dx), float((north - south) / dy)


def read_dem_tile(path: Path, *, west: int, south: int) -> DemTile:
    """Read exactly one float32 GeoTIFF and promote it to float64."""

    with Image.open(path) as image:
        if image.mode != "F" or image.size != (PIXELS_PER_DEGREE, PIXELS_PER_DEGREE):
            raise ValueError(f"unexpected DEM TIFF identity: mode={image.mode}, size={image.size}")
        values = np.asarray(image, dtype=np.float64).copy()
    return DemTile(west=west, south=south, values=values)


def _circular_distance_deg(values: np.ndarray, centre: float) -> np.ndarray:
    return np.abs((values - centre + 180.0) % 360.0 - 180.0)


def _local_displacement(
    latitude: float, longitude: float, pixel_lat: np.ndarray, pixel_lon: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    north = np.deg2rad(pixel_lat - latitude) * EARTH_RADIUS_M
    east = (
        np.deg2rad(pixel_lon - longitude)
        * EARTH_RADIUS_M
        * np.cos(np.deg2rad(latitude))
    )
    distance = np.hypot(east, north)
    bearing = np.mod(np.degrees(np.arctan2(east, north)), 360.0)
    return east, north, distance, bearing


def _nearby_pixels(
    turbine: Turbine, tiles: Sequence[DemTile], *, radius_m: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat_pad = np.degrees(radius_m / EARTH_RADIUS_M) + 1.0 / PIXELS_PER_DEGREE
    lon_pad = lat_pad / np.cos(np.deg2rad(turbine.latitude)) + 1.0 / PIXELS_PER_DEGREE
    all_lat: list[np.ndarray] = []
    all_lon: list[np.ndarray] = []
    all_z: list[np.ndarray] = []
    for tile in tiles:
        row0 = max(0, int(np.floor((tile.north - (turbine.latitude + lat_pad)) * PIXELS_PER_DEGREE)))
        row1 = min(PIXELS_PER_DEGREE, int(np.ceil((tile.north - (turbine.latitude - lat_pad)) * PIXELS_PER_DEGREE)))
        col0 = max(0, int(np.floor(((turbine.longitude - lon_pad) - tile.west) * PIXELS_PER_DEGREE)))
        col1 = min(PIXELS_PER_DEGREE, int(np.ceil(((turbine.longitude + lon_pad) - tile.west) * PIXELS_PER_DEGREE)))
        if row0 >= row1 or col0 >= col1:
            continue
        rows, cols = np.meshgrid(
            np.arange(row0, row1, dtype=np.int32),
            np.arange(col0, col1, dtype=np.int32),
            indexing="ij",
        )
        lat, lon = tile.pixel_centres(rows, cols)
        all_lat.append(lat.ravel())
        all_lon.append(lon.ravel())
        all_z.append(tile.values[row0:row1, col0:col1].ravel())
    if not all_z:
        raise ValueError(f"no DEM pixels near turbine {turbine.turbine_id}")
    lat = np.concatenate(all_lat)
    lon = np.concatenate(all_lon)
    z = np.concatenate(all_z)
    if not np.isfinite(z).all():
        raise ValueError(f"nonfinite DEM neighborhood for turbine {turbine.turbine_id}")
    return lat, lon, z


def build_static_terrain_table(
    turbines: Sequence[Turbine], tiles: Sequence[DemTile]
) -> pd.DataFrame:
    """Build the frozen 17x16 directional terrain lookup table."""

    if len(turbines) != 17 or len({t.turbine_id for t in turbines}) != 17:
        raise ValueError("the frozen design requires 17 unique turbines")
    rows: list[dict[str, float | int | str]] = []
    for turbine in turbines:
        containing = [
            tile
            for tile in tiles
            if tile.south <= turbine.latitude < tile.south + 1
            and tile.west <= turbine.longitude < tile.west + 1
        ]
        if len(containing) != 1:
            raise ValueError(f"turbine {turbine.turbine_id} must be in exactly one tile")
        base_tile = containing[0]
        z0 = base_tile.bilinear(turbine.latitude, turbine.longitude)
        dz_dx, dz_dy = base_tile.nearest_gradient(turbine.latitude, turbine.longitude)
        pixel_lat, pixel_lon, z = _nearby_pixels(turbine, tiles, radius_m=max(RADII_M))
        _, _, distance, bearing = _local_displacement(
            turbine.latitude, turbine.longitude, pixel_lat, pixel_lon
        )
        for sector_index, theta in enumerate(SECTOR_CENTRES):
            theta_rad = np.deg2rad(theta)
            record: dict[str, float | int | str] = {
                "turbine_id": turbine.turbine_id,
                "group": turbine.group,
                "capacity_mw": turbine.capacity_mw,
                "latitude": turbine.latitude,
                "longitude": turbine.longitude,
                "sector_index": sector_index,
                "sector_centre_degrees": float(theta),
                "turbine_elevation_m": z0,
                "signed_slope_alignment": float(dz_dx * np.sin(theta_rad) + dz_dy * np.cos(theta_rad)),
            }
            wedge = _circular_distance_deg(bearing, theta) <= SECTOR_HALF_WIDTH_DEG
            for radius in RADII_M:
                mask = wedge & (distance >= MIN_HORIZON_DISTANCE_M) & (distance <= radius)
                if not mask.any():
                    raise ValueError(
                        f"empty horizon sector turbine={turbine.turbine_id}, sector={sector_index}, radius={radius}"
                    )
                angle = np.degrees(np.arctan2(z[mask] - z0, distance[mask]))
                record[f"horizon_angle_r{int(radius):04d}m"] = float(np.max(angle))
            rows.append(record)
    table = pd.DataFrame(rows).sort_values(["turbine_id", "sector_index"]).reset_index(drop=True)
    if len(table) != 272 or not np.isfinite(table[list(TERRAIN_VALUE_COLUMNS)].to_numpy(np.float64)).all():
        raise AssertionError("static terrain table shape/finite contract differs")
    return table


def build_dynamic_exposure_features(
    control: pd.DataFrame,
    static_table: pd.DataFrame,
    turbines: Sequence[Turbine],
    *,
    group: str,
) -> pd.DataFrame:
    """Select the frozen sector using existing LDAPS u/v and aggregate turbines."""

    missing = [column for column in WIND_COLUMNS if column not in control.columns]
    if missing:
        raise KeyError(f"missing wind columns: {missing}")
    wind = control.loc[:, list(WIND_COLUMNS)].to_numpy(np.float64)
    if not np.isfinite(wind).all():
        raise ValueError("nonfinite direction input")
    u = 0.5 * (wind[:, 0] + wind[:, 1])
    v = 0.5 * (wind[:, 2] + wind[:, 3])
    bearing = np.mod(np.degrees(np.arctan2(-u, -v)), 360.0)
    sectors = np.floor((bearing + SECTOR_HALF_WIDTH_DEG) / 22.5).astype(np.int64) % 16

    group_turbines = [t for t in turbines if t.group == group]
    if not group_turbines:
        raise ValueError(f"no turbines for {group}")
    weights = np.asarray([t.capacity_mw for t in group_turbines], dtype=np.float64)
    weights /= weights.sum()
    lookup = static_table.set_index(["turbine_id", "sector_index"])
    arrays: dict[str, np.ndarray] = {}
    for value_column in TERRAIN_VALUE_COLUMNS:
        selected = np.column_stack(
            [
                lookup.loc[(t.turbine_id, slice(None)), value_column]
                .sort_index()
                .to_numpy(np.float64)[sectors]
                for t in group_turbines
            ]
        )
        mean = selected @ weights
        variance = ((selected - mean[:, None]) ** 2) @ weights
        arrays[f"copdem__{value_column}_cw_mean"] = mean
        arrays[f"copdem__{value_column}_cw_std"] = np.sqrt(np.maximum(variance, 0.0))
    result = pd.DataFrame(arrays, index=control.index)
    result = result.loc[:, list(EXTENDED_COLUMNS)]
    if not np.isfinite(result.to_numpy(np.float64)).all():
        raise AssertionError("nonfinite dynamic terrain features")
    return result.astype(np.float32)


def parse_turbines(rows: Iterable[Mapping[str, object]]) -> tuple[Turbine, ...]:
    turbines = tuple(
        Turbine(
            turbine_id=int(row["id"]),
            group=str(row["group"]),
            capacity_mw=float(row["capacity_mw"]),
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
        )
        for row in rows
    )
    if len(turbines) != 17:
        raise ValueError("expected 17 turbines")
    return turbines


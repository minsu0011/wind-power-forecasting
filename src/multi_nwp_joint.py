"""Feature primitives for a cutoff-safe three-source NWP paired model.

The feature block is deliberately joint: it describes agreement and
disagreement between ECMWF IFS025, DWD ICON Global and NOAA GFS Global in one
row.  It is therefore not an average of predictions made by three independent
models.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


SOURCE_ORDER = ("ecmwf", "icon", "gfs")
JOINT_COLUMNS = (
    "mnwp__ecmwf_ws100",
    "mnwp__ecmwf_u100",
    "mnwp__ecmwf_v100",
    "mnwp__icon_ws100_vector_interp",
    "mnwp__icon_u100_vector_interp",
    "mnwp__icon_v100_vector_interp",
    "mnwp__gfs_ws100",
    "mnwp__gfs_u100",
    "mnwp__gfs_v100",
    "mnwp__ws100_mean",
    "mnwp__ws100_std",
    "mnwp__ws100_range",
    "mnwp__u100_mean",
    "mnwp__u100_std",
    "mnwp__u100_range",
    "mnwp__v100_mean",
    "mnwp__v100_std",
    "mnwp__v100_range",
    "mnwp__consensus_vector_speed",
    "mnwp__directional_coherence",
    "mnwp__vector_distance_ecmwf_icon",
    "mnwp__vector_distance_ecmwf_gfs",
    "mnwp__vector_distance_icon_gfs",
    "mnwp__ecmwf_minus_canonical_hub_ws",
    "mnwp__icon_minus_canonical_hub_ws",
    "mnwp__gfs_minus_canonical_hub_ws",
    "mnwp__mean_minus_canonical_hub_ws",
    "mnwp__spread_to_mean_ws_ratio",
    "mnwp__vertical_shear_mean",
    "mnwp__vertical_shear_abs_disagreement",
)


def _selected_day(index: pd.DatetimeIndex) -> np.ndarray:
    """Return the frozen cutoff-safe Previous Runs day for each valid hour."""

    return np.where(index.hour.astype(int).isin(range(1, 14)), 1, 2)


def _wind_components(speed: np.ndarray, direction_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radians = np.deg2rad(direction_deg)
    return -speed * np.sin(radians), -speed * np.cos(radians)


def _selected(frame: pd.DataFrame, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    days = _selected_day(pd.DatetimeIndex(frame.index))
    day1 = days == 1
    speed = np.where(
        day1,
        frame[f"wind_speed_{height}m_previous_day1"].to_numpy(dtype=np.float64),
        frame[f"wind_speed_{height}m_previous_day2"].to_numpy(dtype=np.float64),
    )
    direction = np.where(
        day1,
        frame[f"wind_direction_{height}m_previous_day1"].to_numpy(dtype=np.float64),
        frame[f"wind_direction_{height}m_previous_day2"].to_numpy(dtype=np.float64),
    )
    u, v = _wind_components(speed, direction)
    return speed, u, v


def build_joint_features(
    sources: Mapping[str, pd.DataFrame],
    canonical_hub_ws: pd.Series,
) -> pd.DataFrame:
    """Create the frozen raw-plus-cross-source feature block.

    ICON's 100 m vector is a deterministic midpoint of its native 80 and 120 m
    vectors.  This avoids circular averaging of direction degrees.  Population
    standard deviations (``ddof=0``) are used throughout.
    """

    if tuple(sources) != SOURCE_ORDER:
        raise ValueError(f"sources must be ordered exactly as {SOURCE_ORDER}")
    index = canonical_hub_ws.index
    if not all(frame.index.equals(index) for frame in sources.values()):
        raise ValueError("source and canonical indexes differ")

    e_ws, e_u, e_v = _selected(sources["ecmwf"], 100)
    i80_ws, i80_u, i80_v = _selected(sources["icon"], 80)
    i120_ws, i120_u, i120_v = _selected(sources["icon"], 120)
    i_u = 0.5 * (i80_u + i120_u)
    i_v = 0.5 * (i80_v + i120_v)
    i_ws = np.hypot(i_u, i_v)
    g80_ws, _, _ = _selected(sources["gfs"], 80)
    g_ws, g_u, g_v = _selected(sources["gfs"], 100)

    ws = np.column_stack((e_ws, i_ws, g_ws))
    u = np.column_stack((e_u, i_u, g_u))
    v = np.column_stack((e_v, i_v, g_v))
    ws_mean = ws.mean(axis=1)
    ws_std = ws.std(axis=1, ddof=0)
    u_mean = u.mean(axis=1)
    v_mean = v.mean(axis=1)
    vector_speed = np.hypot(u_mean, v_mean)
    hub = canonical_hub_ws.to_numpy(dtype=np.float64)
    icon_shear = i120_ws - i80_ws
    gfs_shear = g_ws - g80_ws

    result = pd.DataFrame(index=index)
    raw_values = {
        "mnwp__ecmwf_ws100": e_ws,
        "mnwp__ecmwf_u100": e_u,
        "mnwp__ecmwf_v100": e_v,
        "mnwp__icon_ws100_vector_interp": i_ws,
        "mnwp__icon_u100_vector_interp": i_u,
        "mnwp__icon_v100_vector_interp": i_v,
        "mnwp__gfs_ws100": g_ws,
        "mnwp__gfs_u100": g_u,
        "mnwp__gfs_v100": g_v,
        "mnwp__ws100_mean": ws_mean,
        "mnwp__ws100_std": ws_std,
        "mnwp__ws100_range": np.ptp(ws, axis=1),
        "mnwp__u100_mean": u_mean,
        "mnwp__u100_std": u.std(axis=1, ddof=0),
        "mnwp__u100_range": np.ptp(u, axis=1),
        "mnwp__v100_mean": v_mean,
        "mnwp__v100_std": v.std(axis=1, ddof=0),
        "mnwp__v100_range": np.ptp(v, axis=1),
        "mnwp__consensus_vector_speed": vector_speed,
        "mnwp__directional_coherence": vector_speed / np.maximum(ws_mean, 0.05),
        "mnwp__vector_distance_ecmwf_icon": np.hypot(e_u - i_u, e_v - i_v),
        "mnwp__vector_distance_ecmwf_gfs": np.hypot(e_u - g_u, e_v - g_v),
        "mnwp__vector_distance_icon_gfs": np.hypot(i_u - g_u, i_v - g_v),
        "mnwp__ecmwf_minus_canonical_hub_ws": e_ws - hub,
        "mnwp__icon_minus_canonical_hub_ws": i_ws - hub,
        "mnwp__gfs_minus_canonical_hub_ws": g_ws - hub,
        "mnwp__mean_minus_canonical_hub_ws": ws_mean - hub,
        "mnwp__spread_to_mean_ws_ratio": ws_std / np.maximum(ws_mean, 0.05),
        "mnwp__vertical_shear_mean": 0.5 * (icon_shear + gfs_shear),
        "mnwp__vertical_shear_abs_disagreement": np.abs(icon_shear - gfs_shear),
    }
    for name in JOINT_COLUMNS:
        result[name] = raw_values[name]
    result = result.astype(np.float32)
    if tuple(result.columns) != JOINT_COLUMNS:
        raise AssertionError("joint feature schema changed")
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("joint feature block contains non-finite values")
    return result


def paired_transfer(
    baseline_kwh: pd.Series,
    control_cf: pd.Series,
    extended_cf: pd.Series,
    *,
    capacity_kwh: float,
    transfer_weight: float,
) -> pd.Series:
    """Transfer one paired prediction difference to a frozen kWh baseline."""

    if not baseline_kwh.index.equals(control_cf.index) or not control_cf.index.equals(extended_cf.index):
        raise ValueError("paired transfer indexes differ")
    prediction = np.clip(
        baseline_kwh.to_numpy(dtype=np.float64)
        + float(transfer_weight)
        * float(capacity_kwh)
        * (extended_cf.to_numpy(dtype=np.float64) - control_cf.to_numpy(dtype=np.float64)),
        0.0,
        1.02 * float(capacity_kwh),
    )
    return pd.Series(prediction, index=baseline_kwh.index, name=baseline_kwh.name)


__all__ = ["JOINT_COLUMNS", "SOURCE_ORDER", "build_joint_features", "paired_transfer"]

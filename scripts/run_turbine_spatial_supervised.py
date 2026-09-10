"""Strict-forward turbine-coordinate NWP / SCADA supervised experiment.

Only the preregistered q0.7 model blended 20% into corrected-v3 can be
promoted.  L1 is fitted as an unblended diagnostic control and can never be
selected.  Stage 1 uses physically byte-capped pre-2024 raw inputs and writes
an immutable O_EXCL lock before any 2024 value may be opened.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import joblib
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.run_shared_q07_multiseed import (  # noqa: E402
    EXPECTED_ROWS_PRE2024,
    EXPECTED_STAGE1_RAW_ROWS,
    OFFICIAL_LABEL_PREFIX_SHA256,
    OFFICIAL_NWP_PREFIX_SHA256,
    OFFICIAL_STAGE1_PREFIX_BYTES,
    YEAR_2022_END,
    YEAR_2022_START,
    YEAR_2023_END,
    YEAR_2023_START,
    YEAR_2024_END,
    YEAR_2024_START,
    _BoundedRawReader,
    _canonical_sha256,
    _csv_prefix_identity,
    _frame_sha256,
    _read_bounded_weather_csv,
    _read_labels,
)
from scripts.run_turbine_scada_power import (  # noqa: E402
    H2_2023_START,
    H2_2024_START,
    Q2_2024_START,
    Q3_2024_START,
    Q4_2023_START,
    Q4_2024_START,
    SCADA_SPECS,
    TEST_START,
    YEAR_2025_END,
    _aggregate_turbine_hourly,
    _atomic_csv,
    _atomic_joblib,
    _atomic_json,
    _atomic_parquet,
    _discover_csv_prefix,
    _load_turbine_metadata,
    _read_full_labels,
    _read_prediction,
    _read_scada_prefix,
    _stage1_baseline,
    _stage1_historical_paths,
    _write_exclusive_json,
)
from src.features import (  # noqa: E402
    AVAILABLE_COL,
    COORD_COLS,
    GRID_COL,
    SOURCE_VARIABLES,
    TIME_COL,
)
from src.manifest import (  # noqa: E402
    describe_file,
    git_state,
    package_versions,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from src.metric import CAPACITY_KWH, TARGET_COLS, group_metrics  # noqa: E402


PREREGISTER_SHA256 = (
    "0ed1f51898e0fc0cc2e5512d4166ed88cbf6fc657147d8d7cea0dd7fdd03d312"
)
PROMOTABLE_CANDIDATE = "q07_w20"
BLEND_WEIGHT = 0.20
OBJECTIVES = ("q07", "l1_diagnostic_only")
AIR_DENSITY = 1.225
SHEAR_EXPONENT = 1.0 / 7.0
HUB_HEIGHT_M = 117.0
RAW_ROWS_PER_TIMESTAMP = {"ldaps": 16, "gfs": 9}
EXPECTED_TRAIN_TIMESTAMPS = 26_304
EXPECTED_TEST_TIMESTAMPS = 8_760
STAGE1_SLICES = {
    "kpx_group_1": {
        "full": (YEAR_2023_START, YEAR_2023_END),
        "H1": (YEAR_2023_START, H2_2023_START - pd.Timedelta(hours=1)),
        "H2": (H2_2023_START, YEAR_2023_END),
        "Q1": (YEAR_2023_START, pd.Timestamp("2023-04-01 00:00:00")),
        "Q2": (pd.Timestamp("2023-04-01 01:00:00"), pd.Timestamp("2023-07-01 00:00:00")),
        "Q3": (H2_2023_START, Q4_2023_START - pd.Timedelta(hours=1)),
        "Q4": (Q4_2023_START, YEAR_2023_END),
    },
    "kpx_group_2": {
        "full": (YEAR_2023_START, YEAR_2023_END),
        "H1": (YEAR_2023_START, H2_2023_START - pd.Timedelta(hours=1)),
        "H2": (H2_2023_START, YEAR_2023_END),
        "Q1": (YEAR_2023_START, pd.Timestamp("2023-04-01 00:00:00")),
        "Q2": (pd.Timestamp("2023-04-01 01:00:00"), pd.Timestamp("2023-07-01 00:00:00")),
        "Q3": (H2_2023_START, Q4_2023_START - pd.Timedelta(hours=1)),
        "Q4": (Q4_2023_START, YEAR_2023_END),
    },
    "kpx_group_3": {
        "full": (H2_2023_START, YEAR_2023_END),
        "Q3": (H2_2023_START, Q4_2023_START - pd.Timedelta(hours=1)),
        "Q4": (Q4_2023_START, YEAR_2023_END),
    },
}
STAGE2_SLICES = {
    "full": (YEAR_2024_START, YEAR_2024_END),
    "H1": (YEAR_2024_START, H2_2024_START - pd.Timedelta(hours=1)),
    "H2": (H2_2024_START, YEAR_2024_END),
    "Q1": (YEAR_2024_START, Q2_2024_START - pd.Timedelta(hours=1)),
    "Q2": (Q2_2024_START, Q3_2024_START - pd.Timedelta(hours=1)),
    "Q3": (Q3_2024_START, Q4_2024_START - pd.Timedelta(hours=1)),
    "Q4": (Q4_2024_START, YEAR_2024_END),
}

FEATURE_COLUMNS = (
    "ldaps_u50_idw",
    "ldaps_v50_idw",
    "ldaps_ws50_idw",
    "ldaps_hub_ws_idw",
    "ldaps_wpd_idw",
    "gfs_u100_idw",
    "gfs_v100_idw",
    "gfs_ws100_idw",
    "gfs_hub_ws_idw",
    "gfs_wpd_idw",
    "mean_hub_ws",
    "hub_ws_difference",
    "mean_u",
    "mean_v",
    "mean_wpd",
    "forecast_hour_sin",
    "forecast_hour_cos",
    "forecast_dayofyear_sin",
    "forecast_dayofyear_cos",
    "forecast_month_sin",
    "forecast_month_cos",
    "ldaps_lead_hours",
    "gfs_lead_hours",
    "ldaps_run_hour_sin",
    "ldaps_run_hour_cos",
    "gfs_run_hour_sin",
    "gfs_run_hour_cos",
    "turbine_id",
    "turbine_index_fraction",
    "turbine_latitude",
    "turbine_longitude",
    "turbine_latitude_offset",
    "turbine_longitude_offset",
    "turbine_capacity_fraction",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir", type=Path, default=Path(r"data/local/open")
    )
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--recipe", type=Path, default=Path("configs/train_final.v3.locked.json")
    )
    parser.add_argument(
        "--preregister",
        type=Path,
        default=Path("configs/turbine_spatial_supervised_preregister.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/postgate/turbine_spatial_supervised"),
    )
    parser.add_argument("--n-jobs", type=int, default=7)
    parser.add_argument(
        "--stage", choices=("stage1", "stage2", "final", "all"), default="all"
    )
    return parser.parse_args(argv)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _verify_preregister(path: Path) -> dict[str, Any]:
    path = path.resolve()
    observed = sha256_file(path)
    if observed != PREREGISTER_SHA256:
        raise AssertionError(f"preregister changed: {observed} != {PREREGISTER_SHA256}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    contract = payload["selection_contract"]
    if contract["single_promotable_candidate"] != PROMOTABLE_CANDIDATE:
        raise AssertionError("promotable candidate changed")
    if bool(contract["candidate_or_coefficient_search"]):
        raise AssertionError("candidate search must remain disabled")
    if int(contract["random_seed"]) != 42:
        raise AssertionError("seed changed")
    return {"path": str(path), "sha256": observed, "payload": payload}


def _snapshot_named(paths: Mapping[str, Path]) -> dict[str, Any]:
    return {name: describe_file(path) for name, path in paths.items()}


def _provenance_paths(preregister: Path, recipe: Path) -> dict[str, Path]:
    return {
        "runner": Path(__file__).resolve(),
        "test": PROJECT_DIR / "tests" / "test_turbine_spatial_supervised.py",
        "preregister": preregister.resolve(),
        "recipe": recipe.resolve(),
        "metric": PROJECT_DIR / "src" / "metric.py",
        "feature_schema": PROJECT_DIR / "src" / "features.py",
        "bounded_nwp_helper": PROJECT_DIR / "scripts" / "run_shared_q07_multiseed.py",
        "scada_helper": PROJECT_DIR / "scripts" / "run_turbine_scada_power.py",
    }


def _haversine_distance_km(
    grid_latitude: np.ndarray,
    grid_longitude: np.ndarray,
    turbine_latitude: np.ndarray,
    turbine_longitude: np.ndarray,
) -> np.ndarray:
    radius = 6371.0088
    grid_lat = np.radians(np.asarray(grid_latitude, dtype=float))[:, None]
    grid_lon = np.radians(np.asarray(grid_longitude, dtype=float))[:, None]
    turbine_lat = np.radians(np.asarray(turbine_latitude, dtype=float))[None, :]
    turbine_lon = np.radians(np.asarray(turbine_longitude, dtype=float))[None, :]
    delta_lat = turbine_lat - grid_lat
    delta_lon = turbine_lon - grid_lon
    value = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(grid_lat) * np.cos(turbine_lat) * np.sin(delta_lon / 2.0) ** 2
    )
    return 2.0 * radius * np.arcsin(np.sqrt(np.clip(value, 0.0, 1.0)))


def _idw_weights(
    grid_latitude: np.ndarray,
    grid_longitude: np.ndarray,
    turbine_latitude: np.ndarray,
    turbine_longitude: np.ndarray,
    *,
    power: float = 2.0,
    distance_floor_km: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    distance = _haversine_distance_km(
        grid_latitude, grid_longitude, turbine_latitude, turbine_longitude
    )
    bounded = np.maximum(distance, float(distance_floor_km))
    raw = 1.0 / np.power(bounded, float(power))
    weights = raw / raw.sum(axis=0, keepdims=True)
    if not np.allclose(weights.sum(axis=0), 1.0, atol=1e-15, rtol=0.0):
        raise AssertionError("IDW weights do not sum to one")
    return weights, distance


def _source_core_arrays(
    frame: pd.DataFrame, source: str
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, dict[str, np.ndarray], pd.Series, dict[str, Any]]:
    if source not in {"ldaps", "gfs"}:
        raise KeyError(source)
    required_variables = (
        (
            "heightAboveGround_50_50MUmax",
            "heightAboveGround_50_50MUmin",
            "heightAboveGround_50_50MVmax",
            "heightAboveGround_50_50MVmin",
        )
        if source == "ldaps"
        else ("heightAboveGround_100_100u", "heightAboveGround_100_100v")
    )
    required = {TIME_COL, AVAILABLE_COL, GRID_COL, *COORD_COLS, *required_variables}
    if not required.issubset(frame.columns):
        raise ValueError(f"{source} raw core schema changed: {sorted(required - set(frame.columns))}")
    counts = frame.groupby(TIME_COL, sort=True)[GRID_COL].size()
    expected_grids = RAW_ROWS_PER_TIMESTAMP[source]
    if not counts.eq(expected_grids).all():
        raise AssertionError(f"{source} rows per timestamp changed")
    times = pd.DatetimeIndex(counts.index, name="forecast_kst_dtm")
    grid_ids = np.sort(frame[GRID_COL].unique())
    if len(grid_ids) != expected_grids:
        raise AssertionError(f"{source} grid count changed")
    grid_meta = frame.groupby(GRID_COL, sort=True)[list(COORD_COLS)].agg(["min", "max"])
    for coordinate in COORD_COLS:
        if not np.allclose(
            grid_meta[(coordinate, "min")],
            grid_meta[(coordinate, "max")],
            atol=1e-7,
            rtol=0.0,
        ):
            raise AssertionError(f"{source} grid {coordinate} moved over time")
    grid_latitude = grid_meta[(COORD_COLS[0], "min")].to_numpy(dtype=float)
    grid_longitude = grid_meta[(COORD_COLS[1], "min")].to_numpy(dtype=float)

    def pivot(column: str) -> np.ndarray:
        table = frame.pivot(index=TIME_COL, columns=GRID_COL, values=column)
        table = table.reindex(index=times, columns=grid_ids)
        values = table.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise AssertionError(f"{source}/{column} contains non-finite values")
        return values

    if source == "ldaps":
        u = 0.5 * (
            pivot("heightAboveGround_50_50MUmax")
            + pivot("heightAboveGround_50_50MUmin")
        )
        v = 0.5 * (
            pivot("heightAboveGround_50_50MVmax")
            + pivot("heightAboveGround_50_50MVmin")
        )
    else:
        u = pivot("heightAboveGround_100_100u")
        v = pivot("heightAboveGround_100_100v")
    available_counts = frame.groupby(TIME_COL, sort=True)[AVAILABLE_COL].nunique(dropna=False)
    if not available_counts.eq(1).all():
        raise AssertionError(f"{source} available time differs across grids")
    available = frame.groupby(TIME_COL, sort=True)[AVAILABLE_COL].first().reindex(times)
    available = pd.Series(pd.to_datetime(available, errors="raise").to_numpy(), index=times)
    return times, grid_latitude, grid_longitude, {"u": u, "v": v}, available, {
        "source": source,
        "timestamps": len(times),
        "grids": len(grid_ids),
        "start": times.min().isoformat(),
        "end": times.max().isoformat(),
        "raw_core_sha256": _frame_sha256(
            pd.DataFrame(
                np.concatenate([u, v], axis=1),
                index=times,
            )
        ),
    }


def _spatial_turbine_features(
    raw_frames: Mapping[str, pd.DataFrame],
    metadata: Mapping[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    source_arrays: dict[str, Any] = {}
    source_audit: dict[str, Any] = {}
    for source in ("ldaps", "gfs"):
        source_arrays[source] = _source_core_arrays(raw_frames[source], source)
        source_audit[source] = source_arrays[source][-1]
    ldaps_times, ldaps_lat, ldaps_lon, ldaps_core, ldaps_available, _ = source_arrays["ldaps"]
    gfs_times, gfs_lat, gfs_lon, gfs_core, gfs_available, _ = source_arrays["gfs"]
    if not ldaps_times.equals(gfs_times):
        raise AssertionError("LDAPS/GFS forecast time sequences differ")
    times = ldaps_times
    lead = {
        "ldaps": (
            times.to_series(index=times) - pd.to_datetime(ldaps_available)
        ).dt.total_seconds().to_numpy(dtype=float)
        / 3600.0,
        "gfs": (
            times.to_series(index=times) - pd.to_datetime(gfs_available)
        ).dt.total_seconds().to_numpy(dtype=float)
        / 3600.0,
    }
    if not np.isfinite(np.concatenate(list(lead.values()))).all():
        raise AssertionError("lead hours contain non-finite values")
    available = {"ldaps": pd.DatetimeIndex(ldaps_available), "gfs": pd.DatetimeIndex(gfs_available)}
    time_values = {
        "forecast_hour_sin": np.sin(2.0 * np.pi * times.hour / 24.0),
        "forecast_hour_cos": np.cos(2.0 * np.pi * times.hour / 24.0),
        "forecast_dayofyear_sin": np.sin(2.0 * np.pi * (times.dayofyear - 1) / 365.2425),
        "forecast_dayofyear_cos": np.cos(2.0 * np.pi * (times.dayofyear - 1) / 365.2425),
        "forecast_month_sin": np.sin(2.0 * np.pi * (times.month - 1) / 12.0),
        "forecast_month_cos": np.cos(2.0 * np.pi * (times.month - 1) / 12.0),
        "ldaps_lead_hours": lead["ldaps"],
        "gfs_lead_hours": lead["gfs"],
        "ldaps_run_hour_sin": np.sin(2.0 * np.pi * available["ldaps"].hour / 24.0),
        "ldaps_run_hour_cos": np.cos(2.0 * np.pi * available["ldaps"].hour / 24.0),
        "gfs_run_hour_sin": np.sin(2.0 * np.pi * available["gfs"].hour / 24.0),
        "gfs_run_hour_cos": np.cos(2.0 * np.pi * available["gfs"].hour / 24.0),
    }
    features: dict[str, pd.DataFrame] = {}
    group_audit: dict[str, Any] = {}
    for group, group_metadata in metadata.items():
        turbine_lat = group_metadata["latitude"].to_numpy(dtype=float)
        turbine_lon = group_metadata["longitude"].to_numpy(dtype=float)
        ldaps_weights, ldaps_distance = _idw_weights(
            ldaps_lat, ldaps_lon, turbine_lat, turbine_lon
        )
        gfs_weights, gfs_distance = _idw_weights(
            gfs_lat, gfs_lon, turbine_lat, turbine_lon
        )
        ldaps_u = ldaps_core["u"] @ ldaps_weights
        ldaps_v = ldaps_core["v"] @ ldaps_weights
        ldaps_ws = np.sqrt(ldaps_u**2 + ldaps_v**2)
        ldaps_hub = ldaps_ws * (HUB_HEIGHT_M / 50.0) ** SHEAR_EXPONENT
        ldaps_wpd = 0.5 * AIR_DENSITY * ldaps_hub**3
        gfs_u = gfs_core["u"] @ gfs_weights
        gfs_v = gfs_core["v"] @ gfs_weights
        gfs_ws = np.sqrt(gfs_u**2 + gfs_v**2)
        gfs_hub = gfs_ws * (HUB_HEIGHT_M / 100.0) ** SHEAR_EXPONENT
        gfs_wpd = 0.5 * AIR_DENSITY * gfs_hub**3
        n_turbines = len(group_metadata)
        index = pd.MultiIndex.from_product(
            [times, group_metadata["turbine"].astype(int).tolist()],
            names=["forecast_kst_dtm", "turbine_id_index"],
        )
        values: dict[str, np.ndarray] = {
            "ldaps_u50_idw": ldaps_u.reshape(-1),
            "ldaps_v50_idw": ldaps_v.reshape(-1),
            "ldaps_ws50_idw": ldaps_ws.reshape(-1),
            "ldaps_hub_ws_idw": ldaps_hub.reshape(-1),
            "ldaps_wpd_idw": ldaps_wpd.reshape(-1),
            "gfs_u100_idw": gfs_u.reshape(-1),
            "gfs_v100_idw": gfs_v.reshape(-1),
            "gfs_ws100_idw": gfs_ws.reshape(-1),
            "gfs_hub_ws_idw": gfs_hub.reshape(-1),
            "gfs_wpd_idw": gfs_wpd.reshape(-1),
            "mean_hub_ws": (0.5 * (ldaps_hub + gfs_hub)).reshape(-1),
            "hub_ws_difference": (ldaps_hub - gfs_hub).reshape(-1),
            "mean_u": (0.5 * (ldaps_u + gfs_u)).reshape(-1),
            "mean_v": (0.5 * (ldaps_v + gfs_v)).reshape(-1),
            "mean_wpd": (0.5 * (ldaps_wpd + gfs_wpd)).reshape(-1),
        }
        for name, time_array in time_values.items():
            values[name] = np.repeat(np.asarray(time_array, dtype=float), n_turbines)
        turbine_ids = group_metadata["turbine"].to_numpy(dtype=np.int16)
        rank_fraction = (
            np.arange(n_turbines, dtype=float) / max(n_turbines - 1, 1)
        )
        geometry = {
            "turbine_id": turbine_ids,
            "turbine_index_fraction": rank_fraction,
            "turbine_latitude": turbine_lat,
            "turbine_longitude": turbine_lon,
            "turbine_latitude_offset": group_metadata["latitude_offset"].to_numpy(dtype=float),
            "turbine_longitude_offset": group_metadata["longitude_offset"].to_numpy(dtype=float),
            "turbine_capacity_fraction": group_metadata["rated_kwh"].to_numpy(dtype=float)
            / CAPACITY_KWH[group],
        }
        for name, turbine_array in geometry.items():
            values[name] = np.tile(turbine_array, len(times))
        frame = pd.DataFrame(values, index=index).loc[:, list(FEATURE_COLUMNS)]
        frame = frame.astype({column: "float32" for column in FEATURE_COLUMNS if column != "turbine_id"})
        frame["turbine_id"] = frame["turbine_id"].astype("int16")
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise AssertionError(f"{group} spatial feature matrix contains non-finite values")
        if tuple(frame.columns) != FEATURE_COLUMNS:
            raise AssertionError(f"{group} spatial feature schema changed")
        features[group] = frame
        group_audit[group] = {
            "rows": len(frame),
            "timestamps": len(times),
            "turbines": n_turbines,
            "columns": len(FEATURE_COLUMNS),
            "start": times.min().isoformat(),
            "end": times.max().isoformat(),
            "frame_sha256": _frame_sha256(frame),
            "ldaps_weight_sha256": hashlib.sha256(ldaps_weights.tobytes()).hexdigest(),
            "gfs_weight_sha256": hashlib.sha256(gfs_weights.tobytes()).hexdigest(),
            "ldaps_distance_range_km": [float(ldaps_distance.min()), float(ldaps_distance.max())],
            "gfs_distance_range_km": [float(gfs_distance.min()), float(gfs_distance.max())],
            "interpolation_uses_targets": False,
        }
    return features, {
        "feature_schema": list(FEATURE_COLUMNS),
        "feature_count": len(FEATURE_COLUMNS),
        "source_core": source_audit,
        "groups": group_audit,
        "idw": {"power": 2.0, "distance_floor_km": 0.05, "target_free": True},
    }


def _model_params(preregister: Mapping[str, Any], objective: str, n_jobs: int) -> dict[str, Any]:
    specification = dict(preregister["models"][objective])
    specification.pop("class")
    specification["n_jobs"] = int(n_jobs)
    specification["verbosity"] = -1
    return specification


def _fit_predict(
    *,
    group: str,
    objective: str,
    preregister: Mapping[str, Any],
    features: pd.DataFrame,
    labels: pd.Series,
    hourly_scada: pd.DataFrame,
    metadata: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    valid_index: pd.DatetimeIndex,
    n_jobs: int,
) -> tuple[LGBMRegressor, pd.Series, dict[str, Any]]:
    feature_times = pd.DatetimeIndex(features.index.get_level_values("forecast_kst_dtm"))
    train_mask = (feature_times >= train_start) & (feature_times <= train_end)
    train_times = feature_times[train_mask]
    if len(train_times) == 0 or train_times.max() != train_end:
        raise AssertionError(f"{group} fit period incomplete")
    if train_times.max() >= valid_index.min():
        raise AssertionError("fit.max must be strictly before valid.min")
    if hourly_scada.dropna(how="all").index.max() > train_end:
        raise AssertionError("validation SCADA entered fit target materialization")
    turbines = metadata["turbine"].astype(int).tolist()
    unique_train_times = pd.DatetimeIndex(train_times.unique())
    target = hourly_scada.reindex(index=unique_train_times, columns=turbines)
    target_cf = target.to_numpy(dtype=float) / metadata["rated_kwh"].to_numpy(dtype=float)
    group_labels = labels.reindex(unique_train_times).to_numpy(dtype=float)
    eligible_hours = np.isfinite(group_labels) & (
        group_labels >= 0.10 * CAPACITY_KWH[group]
    )
    fit_rows = np.repeat(eligible_hours, len(turbines)) & np.isfinite(target_cf.reshape(-1))
    train_frame = features.loc[train_mask, list(FEATURE_COLUMNS)].reset_index(drop=True)
    train_frame["turbine_id"] = pd.Categorical(
        train_frame["turbine_id"].astype(int), categories=turbines
    )
    target_vector = target_cf.reshape(-1)
    if int(fit_rows.sum()) < 1000:
        raise AssertionError(f"{group} insufficient turbine-hour fit rows")
    model = LGBMRegressor(**_model_params(preregister, objective, n_jobs))
    model.fit(
        train_frame.loc[fit_rows],
        target_vector[fit_rows],
        categorical_feature=["turbine_id"],
    )
    valid_mask = feature_times.isin(valid_index)
    valid_frame = features.loc[valid_mask, list(FEATURE_COLUMNS)].reset_index(drop=True)
    valid_frame["turbine_id"] = pd.Categorical(
        valid_frame["turbine_id"].astype(int), categories=turbines
    )
    if len(valid_frame) != len(valid_index) * len(turbines):
        raise AssertionError("validation turbine rows incomplete")
    prediction_cf = np.asarray(model.predict(valid_frame), dtype=float).reshape(
        len(valid_index), len(turbines)
    )
    prediction_cf = np.clip(prediction_cf, 0.0, 1.02)
    prediction = pd.Series(
        prediction_cf @ metadata["rated_kwh"].to_numpy(dtype=float),
        index=valid_index,
        name=group,
    )
    complete_sum = target.sum(axis=1, min_count=len(turbines)).to_numpy(dtype=float)
    correlation_mask = eligible_hours & np.isfinite(complete_sum)
    correlation = float(np.corrcoef(complete_sum[correlation_mask], group_labels[correlation_mask])[0, 1])
    return model, prediction, {
        "group": group,
        "objective": objective,
        "fit_start": train_start,
        "fit_end": train_end,
        "fit_max_strictly_before_valid_min": True,
        "valid_start": valid_index.min(),
        "valid_end": valid_index.max(),
        "feature_count": len(FEATURE_COLUMNS),
        "eligible_hours": int(eligible_hours.sum()),
        "turbine_hour_fit_rows": int(fit_rows.sum()),
        "scada_sum_label_correlation": correlation,
        "validation_scada_used": False,
        "interpolation_target_free": True,
        "prediction_sha256": _frame_sha256(prediction.to_frame()),
    }


def _blend(baseline: pd.Series, spatial: pd.Series, group: str) -> pd.Series:
    if not baseline.index.equals(spatial.index):
        raise AssertionError("blend index mismatch")
    values = np.clip(
        (1.0 - BLEND_WEIGHT) * baseline.to_numpy(dtype=float)
        + BLEND_WEIGHT * spatial.to_numpy(dtype=float),
        0.0,
        1.02 * CAPACITY_KWH[group],
    )
    return pd.Series(values, index=baseline.index, name=group)


def _group_score(actual: pd.Series, prediction: pd.Series, group: str) -> dict[str, Any]:
    result = group_metrics(actual, prediction, CAPACITY_KWH[group], group_name=group)
    return {
        "score": float(0.5 * (result.one_minus_nmae + result.ficr)),
        "one_minus_nmae": float(result.one_minus_nmae),
        "ficr": float(result.ficr),
        "n_evaluated": int(result.n_evaluated),
    }


def _slice_comparison(
    labels: pd.Series,
    baseline: pd.Series,
    candidate: pd.Series,
    group: str,
    slices: Mapping[str, tuple[pd.Timestamp, pd.Timestamp]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, (start, end) in slices.items():
        index = baseline.index[(baseline.index >= start) & (baseline.index <= end)]
        before = _group_score(labels.loc[index], baseline.loc[index], group)
        after = _group_score(labels.loc[index], candidate.loc[index], group)
        output[name] = {
            "baseline": before,
            "candidate": after,
            "delta": float(after["score"] - before["score"]),
        }
    return output


def _locked_groups(comparisons: Mapping[str, Any]) -> list[str]:
    return [
        group
        for group in TARGET_COLS
        if group in comparisons
        and all(float(values["delta"]) > 0.0 for values in comparisons[group].values())
    ]


def _read_stage1_nwp(raw_dir: Path) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    raw_frames: dict[str, pd.DataFrame] = {}
    evidence: dict[str, Any] = {}
    for source in ("ldaps", "gfs"):
        frame, audit = _read_bounded_weather_csv(
            raw_dir / "train" / f"{source}_train.csv",
            source=source,
            expected_timestamps=EXPECTED_ROWS_PRE2024,
            expected_start=YEAR_2022_START,
            expected_end=YEAR_2023_END,
            expected_next=YEAR_2024_START,
            expected_prefix_bytes=OFFICIAL_STAGE1_PREFIX_BYTES[source],
        )
        if audit["physical_prefix_sha256"] != OFFICIAL_NWP_PREFIX_SHA256[source]:
            raise AssertionError(f"{source} official prefix changed")
        raw_frames[source] = frame
        evidence[source] = audit
    return raw_frames, evidence


def _refresh_stage1_nwp(raw_dir: Path, evidence: Mapping[str, Any]) -> None:
    for source, record in evidence.items():
        observed, count = _csv_prefix_identity(
            raw_dir / "train" / f"{source}_train.csv",
            data_rows=EXPECTED_STAGE1_RAW_ROWS[source],
            byte_limit=int(record["physical_byte_limit"]),
        )
        if count != int(record["physical_byte_limit"]) or observed != record["physical_prefix_sha256"]:
            raise AssertionError(f"{source} raw prefix changed during stage1")


def _manifest_payload(
    *,
    stage: str,
    preregister: Mapping[str, Any],
    provenance: Mapping[str, Any],
    inputs: Mapping[str, Any],
    output_files: Sequence[Path],
    results: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "turbine_spatial_supervised_strict_forward",
        "created_utc": utc_now(),
        "stage": stage,
        "runtime": {"packages": package_versions()},
        "git": git_state(PROJECT_DIR),
        "preregister": {
            "path": preregister["path"],
            "sha256": preregister["sha256"],
        },
        "provenance": provenance,
        "safe_inputs": inputs,
        "outputs": [describe_file(path) for path in output_files if path.exists()],
        "results_sha256": _canonical_sha256(results),
        "selection_unsafe": False,
        "leaderboard_score_claim": False,
    }


def _stage1(args: argparse.Namespace, preregister: Mapping[str, Any]) -> dict[str, Any]:
    out_dir = args.out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"stage1 requires empty output: {out_dir}")
    raw_dir = args.raw_dir.resolve()
    artifact_root = args.artifact_root.resolve()
    provenance_paths = _provenance_paths(Path(preregister["path"]), args.recipe.resolve())
    provenance_before = _snapshot_named(provenance_paths)
    historical_paths = _stage1_historical_paths(artifact_root)
    historical_before = _snapshot_named(historical_paths)
    info_before = describe_file(raw_dir / "info.xlsx")
    scada_prefixes = {
        "vestas": _discover_csv_prefix(
            raw_dir / "train" / "scada_vestas_train.csv",
            pd.Timestamp("2023-01-01 00:00:00"),
        ),
        "unison": _discover_csv_prefix(
            raw_dir / "train" / "scada_unison_train.csv",
            pd.Timestamp("2023-07-01 00:00:00"),
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    prereg_copy = out_dir / "preregister.json"
    shutil.copyfile(preregister["path"], prereg_copy)
    if sha256_file(prereg_copy) != PREREGISTER_SHA256:
        raise AssertionError("copied preregister changed")
    labels = _read_labels(
        raw_dir / "train" / "train_labels.csv",
        nrows=EXPECTED_ROWS_PRE2024,
        expected_end=YEAR_2023_END,
        prefix_bytes=OFFICIAL_STAGE1_PREFIX_BYTES["labels"],
        expected_prefix_sha256=OFFICIAL_LABEL_PREFIX_SHA256,
    )
    raw_frames, nwp_io = _read_stage1_nwp(raw_dir)
    metadata = _load_turbine_metadata(raw_dir / "info.xlsx")
    spatial_features, feature_audit = _spatial_turbine_features(raw_frames, metadata)
    del raw_frames
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    valid_2023 = labels.index[(labels.index >= YEAR_2023_START) & (labels.index <= YEAR_2023_END)]
    baseline_valid, baseline_audit, baseline_inputs = _stage1_baseline(
        artifact_root, recipe, valid_2023
    )
    baseline = baseline_valid.reindex(labels.index)
    scada_frames: dict[str, pd.DataFrame] = {}
    scada_io: dict[str, Any] = {}
    for manufacturer, prefix in scada_prefixes.items():
        scada_frames[manufacturer], scada_io[manufacturer] = _read_scada_prefix(prefix)
    hourly: dict[str, pd.DataFrame] = {}
    hourly_audit: dict[str, Any] = {}
    for group in TARGET_COLS:
        manufacturer = str(SCADA_SPECS[group]["manufacturer"])
        hourly[group], hourly_audit[group] = _aggregate_turbine_hourly(
            scada_frames[manufacturer], group
        )
    outputs: list[Path] = [prereg_copy]
    training: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    l1_controls: dict[str, Any] = {}
    prediction_wide = pd.DataFrame(index=labels.index)
    for group in TARGET_COLS:
        if group == "kpx_group_3":
            train_start = YEAR_2023_START
            train_end = H2_2023_START - pd.Timedelta(hours=1)
            valid_index = labels.index[(labels.index >= H2_2023_START) & (labels.index <= YEAR_2023_END)]
        else:
            train_start, train_end = YEAR_2022_START, YEAR_2022_END
            valid_index = valid_2023
        group_models: dict[str, LGBMRegressor] = {}
        group_predictions: dict[str, pd.Series] = {}
        for objective in OBJECTIVES:
            print(f"stage1 fit spatial {group} {objective}", flush=True)
            model, prediction, audit = _fit_predict(
                group=group,
                objective=objective,
                preregister=preregister["payload"],
                features=spatial_features[group],
                labels=labels[group],
                hourly_scada=hourly[group],
                metadata=metadata[group],
                train_start=train_start,
                train_end=train_end,
                valid_index=valid_index,
                n_jobs=args.n_jobs,
            )
            group_models[objective] = model
            group_predictions[objective] = prediction
            model_path = out_dir / "models" / f"stage1__{group}__{objective}.joblib"
            _atomic_joblib({"model": model, "audit": audit, "metadata": metadata[group]}, model_path)
            outputs.append(model_path)
            training[f"{group}__{objective}"] = {
                **audit,
                "model_sha256": sha256_file(model_path),
                "promotable": objective == "q07",
            }
            prediction_wide.loc[valid_index, f"{group}__{objective}__raw"] = prediction
        candidate = _blend(
            baseline.loc[valid_index, group], group_predictions["q07"], group
        )
        prediction_wide.loc[valid_index, f"{group}__q07_w20"] = candidate
        comparisons[group] = _slice_comparison(
            labels[group], baseline.loc[valid_index, group], candidate, group, STAGE1_SLICES[group]
        )
        l1_controls[group] = {
            "role": "raw diagnostic only; no blend or selection",
            "full_raw_score": _group_score(labels.loc[valid_index, group], group_predictions["l1_diagnostic_only"], group),
            "selection_eligible": False,
        }
    baseline_path = out_dir / "oof" / "stage1_corrected_v3_baseline.parquet"
    prediction_path = out_dir / "oof" / "stage1_spatial_turbine_predictions.parquet"
    _atomic_parquet(baseline, baseline_path)
    _atomic_parquet(prediction_wide, prediction_path)
    outputs.extend([baseline_path, prediction_path])
    locked = _locked_groups(comparisons)
    provenance_after = _snapshot_named(provenance_paths)
    historical_after = _snapshot_named(historical_paths)
    info_after = describe_file(raw_dir / "info.xlsx")
    _refresh_stage1_nwp(raw_dir, nwp_io)
    scada_after = {
        name: {
            **record,
            "prefix_sha256": _csv_prefix_identity(
                Path(record["path"]),
                data_rows=int(record["prefix_data_rows"]),
                byte_limit=int(record["prefix_bytes"]),
            )[0],
        }
        for name, record in scada_prefixes.items()
    }
    if provenance_before != provenance_after:
        raise AssertionError("provenance changed during stage1")
    if historical_before != historical_after or info_before != info_after:
        raise AssertionError("historical/info input changed during stage1")
    if scada_after != scada_prefixes:
        raise AssertionError("SCADA prefix changed during stage1")
    results = {
        "schema_version": 1,
        "stage": "stage1_pre2024_selection",
        "preregister_sha256": PREREGISTER_SHA256,
        "single_promotable_candidate": PROMOTABLE_CANDIDATE,
        "candidate_or_coefficient_search": False,
        "forbidden_2024_data_read_occurred": False,
        "stage1_cache_files_read": False,
        "nwp_io_contract": nwp_io,
        "scada_io_contract": scada_io,
        "feature_audit": feature_audit,
        "hourly_scada_audit": hourly_audit,
        "baseline_audit": baseline_audit,
        "baseline_inputs": [str(path.resolve()) for path in baseline_inputs],
        "training": training,
        "comparisons": comparisons,
        "comparisons_sha256": _canonical_sha256(comparisons),
        "l1_controls": l1_controls,
        "locked_groups": locked,
        "lock_rule": "q07_w20 delta strictly > 0 in every preregistered group slice",
        "provenance": provenance_before,
        "provenance_sha256": _canonical_sha256(provenance_before),
        "historical_inputs": historical_before,
        "info_input": info_before,
        "selection_unsafe": False,
        "submission_created": False,
    }
    results_path = out_dir / "stage1_results.json"
    _atomic_json(results_path, _json_ready(results))
    outputs.append(results_path)
    lock_path = out_dir / "stage1_lock.json"
    lock_payload = {
        "schema_version": 1,
        "stage": "stage1_pre2024_lock",
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "runner_sha256": provenance_before["runner"]["sha256"],
        "test_sha256": provenance_before["test"]["sha256"],
        "stage1_results_sha256": sha256_file(results_path),
        "comparisons_sha256": results["comparisons_sha256"],
        "single_candidate": PROMOTABLE_CANDIDATE,
        "locked_groups": locked,
        "forbidden_2024_data_read_occurred": False,
    }
    _write_exclusive_json(lock_path, lock_payload)
    outputs.append(lock_path)
    manifest_name = "manifest.json" if not locked else "stage1_manifest.json"
    manifest = _manifest_payload(
        stage="rejected_stage1" if not locked else "stage1",
        preregister=preregister,
        provenance=provenance_before,
        inputs={
            "nwp_prefixes": nwp_io,
            "scada_prefixes": scada_prefixes,
            "labels_prefix_sha256": OFFICIAL_LABEL_PREFIX_SHA256,
            "historical": historical_before,
            "info": info_before,
        },
        output_files=outputs,
        results=results,
    )
    _atomic_json(out_dir / manifest_name, manifest)
    print(json.dumps({"locked_groups": locked}), flush=True)
    return results


CORE_VARIABLES = {
    "ldaps": (
        "heightAboveGround_50_50MUmax",
        "heightAboveGround_50_50MUmin",
        "heightAboveGround_50_50MVmax",
        "heightAboveGround_50_50MVmin",
    ),
    "gfs": ("heightAboveGround_100_100u", "heightAboveGround_100_100v"),
}


def _read_weather_full_bounded(
    path: Path,
    *,
    source: str,
    expected_timestamps: int,
    expected_start: pd.Timestamp,
    expected_end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read an entire chronological raw-NWP file through a physical EOF cap."""

    required = [TIME_COL, AVAILABLE_COL, GRID_COL, *COORD_COLS, *CORE_VARIABLES[source]]
    with path.open("rb") as stream:
        header_line = stream.readline()
    header = pd.read_csv(io.BytesIO(header_line), nrows=0, encoding="utf-8-sig").columns
    if not set(required).issubset(header):
        raise AssertionError(f"{source} full raw schema changed")
    dtypes: dict[str, str] = {GRID_COL: "int16"}
    dtypes.update(
        {column: "float32" for column in [*COORD_COLS, *CORE_VARIABLES[source]]}
    )
    byte_limit = path.stat().st_size
    bounded_raw = _BoundedRawReader(path, byte_limit=byte_limit)
    try:
        with io.BufferedReader(bounded_raw, buffer_size=1024 * 1024) as bounded:
            frame = pd.read_csv(
                bounded,
                usecols=required,
                dtype=dtypes,
                parse_dates=[TIME_COL, AVAILABLE_COL],
                encoding="utf-8-sig",
                low_memory=False,
                memory_map=False,
            )
            returned = bounded_raw.bytes_returned
            position = bounded_raw.underlying_position
    finally:
        bounded_raw.close()
    if returned != byte_limit or position != byte_limit:
        raise AssertionError(f"{source} full reader did not consume exact physical cap")
    expected_rows = expected_timestamps * RAW_ROWS_PER_TIMESTAMP[source]
    if len(frame) != expected_rows:
        raise AssertionError(f"{source} full row count changed: {len(frame)} != {expected_rows}")
    counts = frame.groupby(TIME_COL, sort=True)[GRID_COL].size()
    expected_index = pd.date_range(
        expected_start, expected_end, freq="h", name="forecast_kst_dtm"
    )
    if not counts.index.equals(expected_index):
        raise AssertionError(f"{source} full time sequence changed")
    if not counts.eq(RAW_ROWS_PER_TIMESTAMP[source]).all():
        raise AssertionError(f"{source} full grid counts changed")
    return frame, {
        "source": source,
        "path": str(path.resolve()),
        "physical_byte_limit": byte_limit,
        "physical_bytes_returned": returned,
        "underlying_file_position_after_parse": position,
        "suffix_bytes_exposed_to_parser": 0,
        "materialized_rows": len(frame),
        "materialized_timestamps": len(counts),
        "materialized_start": counts.index.min().isoformat(),
        "materialized_end": counts.index.max().isoformat(),
        "whole_file_sha256": sha256_file(path),
        "usecols": required,
        "memory_map": False,
    }


def _read_nwp_period(
    raw_dir: Path,
    *,
    split: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    if split == "train":
        count, start, end = EXPECTED_TRAIN_TIMESTAMPS, YEAR_2022_START, YEAR_2024_END
    elif split == "test":
        count, start, end = EXPECTED_TEST_TIMESTAMPS, TEST_START, YEAR_2025_END
    else:
        raise KeyError(split)
    frames: dict[str, pd.DataFrame] = {}
    audit: dict[str, Any] = {}
    for source in ("ldaps", "gfs"):
        frames[source], audit[source] = _read_weather_full_bounded(
            raw_dir / split / f"{source}_{split}.csv",
            source=source,
            expected_timestamps=count,
            expected_start=start,
            expected_end=end,
        )
    return frames, audit


def _verify_stage1_lock(
    args: argparse.Namespace, preregister: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    out_dir = args.out_dir.resolve()
    results_path = out_dir / "stage1_results.json"
    lock_path = out_dir / "stage1_lock.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if sha256_file(results_path) != lock["stage1_results_sha256"]:
        raise AssertionError("stage1 results differ from immutable lock")
    if preregister["sha256"] != lock["preregister_sha256"]:
        raise AssertionError("stage1 preregister identity changed")
    if sha256_file(Path(__file__).resolve()) != lock["runner_sha256"]:
        raise AssertionError("runner changed after stage1")
    if (
        sha256_file(PROJECT_DIR / "tests" / "test_turbine_spatial_supervised.py")
        != lock["test_sha256"]
    ):
        raise AssertionError("test changed after stage1")
    recomputed = _locked_groups(results["comparisons"])
    if recomputed != lock["locked_groups"] or recomputed != results["locked_groups"]:
        raise AssertionError("locked groups differ from exact recomputed rule")
    if results["comparisons_sha256"] != lock["comparisons_sha256"]:
        raise AssertionError("stage1 comparison digest changed")
    if lock["single_candidate"] != PROMOTABLE_CANDIDATE:
        raise AssertionError("stage1 candidate identity changed")
    return results, lock


def _stage2(args: argparse.Namespace, preregister: Mapping[str, Any]) -> dict[str, Any]:
    stage1, lock = _verify_stage1_lock(args, preregister)
    out_dir = args.out_dir.resolve()
    locked = list(lock["locked_groups"])
    if not locked:
        results = {
            "schema_version": 1,
            "stage": "stage2_not_run",
            "reason": "no group passed all stage1 slices",
            "locked_groups": [],
            "promoted_groups": [],
            "forbidden_2024_data_read_occurred": False,
            "submission_created": False,
        }
        results_path = out_dir / "stage2_results.json"
        _atomic_json(results_path, results)
        _write_exclusive_json(
            out_dir / "stage2_promotion_lock.json",
            {
                "schema_version": 1,
                "stage1_lock_sha256": sha256_file(out_dir / "stage1_lock.json"),
                "stage2_results_sha256": sha256_file(results_path),
                "promoted_groups": [],
                "single_candidate": PROMOTABLE_CANDIDATE,
            },
        )
        return results
    raw_dir = args.raw_dir.resolve()
    labels = _read_full_labels(raw_dir / "train" / "train_labels.csv")
    metadata_all = _load_turbine_metadata(raw_dir / "info.xlsx")
    metadata = {group: metadata_all[group] for group in locked}
    raw_frames, nwp_io = _read_nwp_period(raw_dir, split="train")
    spatial_features, feature_audit = _spatial_turbine_features(raw_frames, metadata)
    del raw_frames
    valid_index = labels.loc[YEAR_2024_START:YEAR_2024_END].index
    baseline_path = args.artifact_root.resolve() / "oof" / "gate2024_locked_v3_cf_fix.parquet"
    baseline = _read_prediction(baseline_path, valid_index, TARGET_COLS)
    scada_frames: dict[str, pd.DataFrame] = {}
    scada_io: dict[str, Any] = {}
    for group in locked:
        manufacturer = str(SCADA_SPECS[group]["manufacturer"])
        if manufacturer not in scada_frames:
            prefix = _discover_csv_prefix(
                raw_dir / "train" / f"scada_{manufacturer}_train.csv",
                pd.Timestamp("2024-01-01 00:00:00"),
            )
            scada_frames[manufacturer], scada_io[manufacturer] = _read_scada_prefix(prefix)
    candidate = baseline.copy()
    training: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    outputs: list[Path] = []
    for group in locked:
        manufacturer = str(SCADA_SPECS[group]["manufacturer"])
        hourly, hourly_audit = _aggregate_turbine_hourly(scada_frames[manufacturer], group)
        train_start = YEAR_2023_START if group == "kpx_group_3" else YEAR_2022_START
        model, prediction, audit = _fit_predict(
            group=group,
            objective="q07",
            preregister=preregister["payload"],
            features=spatial_features[group],
            labels=labels[group],
            hourly_scada=hourly,
            metadata=metadata[group],
            train_start=train_start,
            train_end=YEAR_2023_END,
            valid_index=valid_index,
            n_jobs=args.n_jobs,
        )
        candidate[group] = _blend(baseline[group], prediction, group)
        comparisons[group] = _slice_comparison(
            labels[group], baseline[group], candidate[group], group, STAGE2_SLICES
        )
        model_path = out_dir / "models" / f"stage2__{group}__q07.joblib"
        _atomic_joblib(
            {"model": model, "audit": audit, "hourly_audit": hourly_audit}, model_path
        )
        outputs.append(model_path)
        training[group] = {
            **audit,
            "hourly_audit": hourly_audit,
            "model_sha256": sha256_file(model_path),
        }
    prediction_path = out_dir / "oof" / "stage2_spatial_turbine_candidate.parquet"
    _atomic_parquet(candidate, prediction_path)
    outputs.append(prediction_path)
    promoted = _locked_groups(comparisons)
    results = {
        "schema_version": 1,
        "stage": "stage2_2024_fixed_transfer",
        "single_candidate": PROMOTABLE_CANDIDATE,
        "locked_groups": locked,
        "promoted_groups": promoted,
        "comparisons": comparisons,
        "comparisons_sha256": _canonical_sha256(comparisons),
        "training": training,
        "nwp_io_contract": nwp_io,
        "scada_io_contract": scada_io,
        "feature_audit": feature_audit,
        "no_reselection": True,
        "submission_created": False,
    }
    results_path = out_dir / "stage2_results.json"
    _atomic_json(results_path, _json_ready(results))
    outputs.append(results_path)
    promotion_path = out_dir / "stage2_promotion_lock.json"
    _write_exclusive_json(
        promotion_path,
        {
            "schema_version": 1,
            "stage": "stage2_promotion_lock",
            "created_utc": utc_now(),
            "stage1_lock_sha256": sha256_file(out_dir / "stage1_lock.json"),
            "stage2_results_sha256": sha256_file(results_path),
            "comparisons_sha256": results["comparisons_sha256"],
            "promoted_groups": promoted,
            "single_candidate": PROMOTABLE_CANDIDATE,
        },
    )
    outputs.append(promotion_path)
    if not promoted:
        manifest = _manifest_payload(
            stage="rejected_stage2",
            preregister=preregister,
            provenance=stage1["provenance"],
            inputs={
                "stage1_lock": describe_file(out_dir / "stage1_lock.json"),
                "nwp": nwp_io,
                "scada": scada_io,
                "baseline": describe_file(baseline_path),
            },
            output_files=outputs,
            results=results,
        )
        _atomic_json(out_dir / "manifest.json", manifest)
    print(json.dumps({"promoted_groups": promoted}), flush=True)
    return results


def _verify_stage2_lock(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    out_dir = args.out_dir.resolve()
    results_path = out_dir / "stage2_results.json"
    lock_path = out_dir / "stage2_promotion_lock.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if sha256_file(results_path) != lock["stage2_results_sha256"]:
        raise AssertionError("stage2 results differ from immutable lock")
    recomputed = _locked_groups(results.get("comparisons", {}))
    if recomputed != lock["promoted_groups"]:
        raise AssertionError("stage2 promotion differs from recomputed rule")
    if lock["single_candidate"] != PROMOTABLE_CANDIDATE:
        raise AssertionError("stage2 candidate identity changed")
    return results, lock


def _final(args: argparse.Namespace, preregister: Mapping[str, Any]) -> dict[str, Any]:
    stage1, _ = _verify_stage1_lock(args, preregister)
    stage2, promotion = _verify_stage2_lock(args)
    promoted = list(promotion["promoted_groups"])
    if not promoted:
        return {
            "stage": "final_not_run",
            "reason": "no group passed all fixed 2024 slices",
            "submission_created": False,
        }
    raw_dir = args.raw_dir.resolve()
    artifact_root = args.artifact_root.resolve()
    out_dir = args.out_dir.resolve()
    labels = _read_full_labels(raw_dir / "train" / "train_labels.csv")
    metadata_all = _load_turbine_metadata(raw_dir / "info.xlsx")
    metadata = {group: metadata_all[group] for group in promoted}
    raw_train, train_nwp_io = _read_nwp_period(raw_dir, split="train")
    raw_test, test_nwp_io = _read_nwp_period(raw_dir, split="test")
    train_features, train_feature_audit = _spatial_turbine_features(raw_train, metadata)
    test_features, test_feature_audit = _spatial_turbine_features(raw_test, metadata)
    del raw_train, raw_test
    for group in promoted:
        if tuple(train_features[group].columns) != tuple(test_features[group].columns):
            raise AssertionError(f"{group} train/test spatial schema differs")
    sample_path = raw_dir / "sample_submission.csv"
    sample = pd.read_csv(sample_path, encoding="utf-8-sig")
    expected_columns = ("forecast_id", "forecast_kst_dtm", *TARGET_COLS)
    if tuple(sample.columns) != expected_columns or len(sample) != 8760:
        raise AssertionError("sample submission schema changed")
    test_index = pd.DatetimeIndex(
        pd.to_datetime(sample["forecast_kst_dtm"], errors="raise"),
        name="forecast_kst_dtm",
    )
    if test_index.min() != TEST_START or test_index.max() != YEAR_2025_END:
        raise AssertionError("test index changed")
    baseline_path = (
        artifact_root / "final_cf_fix" / "predictions" / "corrected_v3_test.parquet"
    )
    baseline = _read_prediction(baseline_path, test_index, TARGET_COLS)
    scada_frames: dict[str, pd.DataFrame] = {}
    scada_io: dict[str, Any] = {}
    candidate = baseline.copy()
    outputs: list[Path] = []
    training: dict[str, Any] = {}
    for group in promoted:
        manufacturer = str(SCADA_SPECS[group]["manufacturer"])
        if manufacturer not in scada_frames:
            prefix = _discover_csv_prefix(
                raw_dir / "train" / f"scada_{manufacturer}_train.csv",
                pd.Timestamp("2025-01-01 00:00:00"),
            )
            scada_frames[manufacturer], scada_io[manufacturer] = _read_scada_prefix(prefix)
        hourly, hourly_audit = _aggregate_turbine_hourly(scada_frames[manufacturer], group)
        combined_features = pd.concat([train_features[group], test_features[group]], axis=0)
        train_start = YEAR_2023_START if group == "kpx_group_3" else YEAR_2022_START
        model, prediction, audit = _fit_predict(
            group=group,
            objective="q07",
            preregister=preregister["payload"],
            features=combined_features,
            labels=labels[group],
            hourly_scada=hourly,
            metadata=metadata[group],
            train_start=train_start,
            train_end=YEAR_2024_END,
            valid_index=test_index,
            n_jobs=args.n_jobs,
        )
        candidate[group] = _blend(baseline[group], prediction, group)
        model_path = out_dir / "models" / f"final__{group}__q07.joblib"
        _atomic_joblib(
            {"model": model, "audit": audit, "hourly_audit": hourly_audit}, model_path
        )
        outputs.append(model_path)
        training[group] = {
            **audit,
            "hourly_audit": hourly_audit,
            "model_sha256": sha256_file(model_path),
        }
    prediction_path = out_dir / "predictions" / "turbine_spatial_supervised_2025.parquet"
    _atomic_parquet(candidate, prediction_path)
    outputs.append(prediction_path)
    readback = pd.read_parquet(prediction_path, engine="pyarrow")
    if not np.array_equal(readback.to_numpy(dtype=float), candidate.to_numpy(dtype=float)):
        raise AssertionError("final prediction parquet readback changed values")
    submission = sample.copy()
    for group in TARGET_COLS:
        values = candidate[group].to_numpy(dtype=float)
        if not np.isfinite(values).all() or values.min() < 0.0 or values.max() > 1.02 * CAPACITY_KWH[group] + 1e-9:
            raise AssertionError(f"{group} final prediction invalid")
        submission[group] = values
    submission_path = out_dir / "turbine_spatial_supervised_2025.csv"
    _atomic_csv(submission, submission_path)
    outputs.append(submission_path)
    if not submission_path.read_bytes().startswith(b"\xef\xbb\xbf"):
        raise AssertionError("submission BOM missing")
    csv_readback = pd.read_csv(submission_path, encoding="utf-8-sig")
    if not csv_readback["forecast_id"].equals(sample["forecast_id"]):
        raise AssertionError("submission IDs changed")
    if not csv_readback["forecast_kst_dtm"].equals(sample["forecast_kst_dtm"]):
        raise AssertionError("submission timestamps changed")
    results = {
        "schema_version": 1,
        "stage": "final_2025",
        "single_candidate": PROMOTABLE_CANDIDATE,
        "promoted_groups": promoted,
        "training": training,
        "train_nwp_io": train_nwp_io,
        "test_nwp_io": test_nwp_io,
        "scada_io": scada_io,
        "train_feature_audit": train_feature_audit,
        "test_feature_audit": test_feature_audit,
        "prediction_sha256": sha256_file(prediction_path),
        "submission_sha256": sha256_file(submission_path),
        "submission_created": True,
        "leaderboard_score_claim": False,
    }
    results_path = out_dir / "results.json"
    _atomic_json(results_path, _json_ready(results))
    outputs.append(results_path)
    manifest = _manifest_payload(
        stage="final",
        preregister=preregister,
        provenance=stage1["provenance"],
        inputs={
            "stage1_lock": describe_file(out_dir / "stage1_lock.json"),
            "stage2_lock": describe_file(out_dir / "stage2_promotion_lock.json"),
            "train_nwp": train_nwp_io,
            "test_nwp": test_nwp_io,
            "baseline": describe_file(baseline_path),
            "sample": describe_file(sample_path),
        },
        output_files=outputs,
        results=results,
    )
    _atomic_json(out_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {"submission": str(submission_path), "sha256": results["submission_sha256"]}
        ),
        flush=True,
    )
    return results


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for name in ("raw_dir", "artifact_root", "recipe", "preregister", "out_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    preregister = _verify_preregister(args.preregister)
    if args.stage in {"stage1", "all"}:
        _stage1(args, preregister)
    if args.stage in {"stage2", "all"}:
        _stage2(args, preregister)
    if args.stage in {"final", "all"}:
        _final(args, preregister)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

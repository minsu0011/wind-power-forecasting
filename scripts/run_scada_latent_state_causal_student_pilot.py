#!/usr/bin/env python
"""Run the single frozen Track-B SCADA state/student falsification pilot.

This runner is intentionally development-only.  It reads byte-bounded SCADA
and official-label prefixes ending in 2023, never opens test data, never reads
2024 values into a dataframe, and has no submission writer.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metric import CAPACITY_KWH, group_metrics
from src.scada_latent_state_pilot import (
    STATES,
    assign_proxy_states,
    fit_upper_envelope,
    initial_normal_mask,
    jensen_shannon,
    leave_one_out_peer_median,
    state_entropy,
)


PREREG_V1 = ROOT / "configs/scada_latent_state_causal_student_pilot_preregister_v1.json"
PREREG_V2 = ROOT / "configs/scada_latent_state_causal_student_pilot_preregister_v2_amendment.json"
PREREG_V3 = ROOT / "configs/scada_latent_state_causal_student_pilot_preregister_v3_capacity_correction.json"
EXPECTED_PREREG = {
    PREREG_V1: (8514, "243e754b90fc6be805067afe8c074a20d6a007ed0bfe4339c792db4dca666890"),
    PREREG_V2: (2816, "4d1ca1d5ab80b7e4d5a6f60c74fd5f4cfc1cbe17602ad8243dc6dcc06d2cc5d0"),
    PREREG_V3: (1125, "170ec8787e58174ec53aace2665df219d9ee136bcfb88fd8c09007d1e1af0e5e"),
}
DEFAULT_OUT = ROOT / "artifacts/baram2026_ncei_scada_research_20260810_210756/track_b"
FEATURES = (
    "ldaps__idw__hub_ws",
    "gfs__idw__hub_ws",
    "cross__hub_ws_mean",
    "cross__hub_ws_difference",
    "ldaps__idw__wind_power_density",
    "gfs__idw__wind_power_density",
    "gfs__idw__surface_0_gust",
    "time__hour_sin",
    "time__hour_cos",
    "time__doy_sin",
    "time__doy_cos",
    "time__month_sin",
    "time__month_cos",
)
MODEL_PARAMS = {
    "loss": "squared_error",
    "learning_rate": 0.04,
    "max_iter": 200,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 100,
    "l2_regularization": 0.1,
    "random_state": 42,
}
STATE_TO_COLUMN = {
    "NORMAL": "normal_fraction",
    "PARTIAL_DERATE": "partial_derate_fraction",
    "CURTAILMENT_LIKE": "curtailment_like_fraction",
    "OUTAGE_LIKE": "outage_like_fraction",
    "LOW_RESOURCE": "low_resource_fraction",
    "SENSOR_OR_DATA_ANOMALY": "anomaly_fraction",
    "UNKNOWN": "unknown_fraction",
}
GROUPS: dict[str, dict[str, Any]] = {
    "G1": {
        "official": "kpx_group_1", "manufacturer": "vestas",
        "turbines": tuple(range(1, 7)), "rated": 600.0,
        "potential_fit": ("2022-01-01 00:00:00", "2022-07-01 00:00:00"),
        "student_train": ("2022-07-01 01:00:00", "2023-01-01 00:00:00"),
        "validation": ("2023-01-01 01:00:00", "2023-07-01 00:00:00"),
    },
    "G2": {
        "official": "kpx_group_2", "manufacturer": "vestas",
        "turbines": tuple(range(7, 13)), "rated": 600.0,
        "potential_fit": ("2022-01-01 00:00:00", "2022-07-01 00:00:00"),
        "student_train": ("2022-07-01 01:00:00", "2023-01-01 00:00:00"),
        "validation": ("2023-01-01 01:00:00", "2023-07-01 00:00:00"),
    },
    "G3": {
        "official": "kpx_group_3", "manufacturer": "unison",
        "turbines": tuple(range(1, 6)), "rated": 700.0,
        "potential_fit": ("2023-01-01 00:00:00", "2023-04-01 00:00:00"),
        "student_train": ("2023-04-01 01:00:00", "2023-07-01 00:00:00"),
        "validation": ("2023-07-01 01:00:00", "2023-10-01 00:00:00"),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def verify_file(path: Path, size: int, digest: str) -> None:
    if not path.is_file() or path.stat().st_size != size or sha256(path) != digest:
        raise RuntimeError(f"frozen identity mismatch: {path}")


def _timestamp_from_prefix(stream: Any) -> tuple[bytes, pd.Timestamp] | None:
    timestamp = bytearray()
    while True:
        token = stream.read(1)
        if token == b"":
            return None if not timestamp else (_raise_truncated())
        if token == b",":
            break
        if token in (b"\n", b"\r"):
            raise RuntimeError("timestamp field ended before delimiter")
        timestamp.extend(token)
    raw = bytes(timestamp)
    parsed = pd.Timestamp(raw.decode("utf-8-sig"))
    return raw, parsed


def _raise_truncated() -> None:
    raise RuntimeError("truncated timestamp at end of CSV")


def read_bounded_csv(path: Path, cutoff_exclusive: pd.Timestamp, usecols: Sequence[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read complete rows before cutoff; at boundary read timestamp field only."""

    collected = io.BytesIO()
    digest = hashlib.sha256()
    rows = 0
    with path.open("rb") as stream:
        header = stream.readline()
        if not header:
            raise RuntimeError(f"empty CSV: {path}")
        collected.write(header)
        digest.update(header)
        next_boundary: pd.Timestamp | None = None
        while True:
            parsed = _timestamp_from_prefix(stream)
            if parsed is None:
                break
            raw_timestamp, timestamp = parsed
            if timestamp >= cutoff_exclusive:
                next_boundary = timestamp
                break
            remainder = stream.readline()
            if not remainder:
                raise RuntimeError("included CSV row has no line terminator/content")
            row = raw_timestamp + b"," + remainder
            collected.write(row)
            digest.update(row)
            rows += 1
    collected.seek(0)
    frame = pd.read_csv(collected, usecols=list(usecols), low_memory=False)
    if len(frame) != rows:
        raise RuntimeError("bounded row accounting differs from parser")
    frame["kst_dtm"] = pd.to_datetime(frame["kst_dtm"], errors="raise")
    if not frame["kst_dtm"].is_monotonic_increasing or frame["kst_dtm"].duplicated().any():
        raise RuntimeError("SCADA/label prefix timestamp order or uniqueness changed")
    if len(frame) and frame["kst_dtm"].max() >= cutoff_exclusive:
        raise RuntimeError("bounded parser crossed cutoff")
    return frame, {
        "path": str(path), "full_size_bytes": path.stat().st_size,
        "full_sha256": sha256(path), "cutoff_exclusive": cutoff_exclusive.isoformat(),
        "prefix_rows": rows, "prefix_bytes": collected.getbuffer().nbytes,
        "prefix_sha256": digest.hexdigest(),
        "next_boundary_timestamp_only": None if next_boundary is None else next_boundary.isoformat(),
        "next_boundary_measurement_cells_read": 0,
        "suffix_rows_parsed": 0,
    }


def _column(manufacturer: str, turbine: int, measurement: str) -> str:
    return f"{manufacturer}_wtg{turbine:02d}_{measurement}"


def _ratio(power: np.ndarray, potential: np.ndarray, wind: np.ndarray) -> np.ndarray:
    eligible = (
        np.isfinite(power) & np.isfinite(potential) & np.isfinite(wind)
        & (power >= 0.0) & (power <= 1.05) & (wind >= 3.0) & (wind <= 20.0)
        & (potential >= 0.10)
    )
    out = np.full(power.shape, np.nan, dtype=np.float64)
    out[eligible] = power[eligible] / potential[eligible]
    return out


def _normal_from_curve(power: np.ndarray, potential: np.ndarray, wind: np.ndarray) -> np.ndarray:
    ratio = _ratio(power, potential, wind)
    peer = leave_one_out_peer_median(ratio)
    return initial_normal_mask(ratio, peer, potential, wind)


def _window(targets: pd.Series, spec: Mapping[str, Any]) -> np.ndarray:
    result = np.full(len(targets), "OUTSIDE_EVALUATED_WINDOWS", dtype=object)
    for name in ("student_train", "validation"):
        start, end = map(pd.Timestamp, spec[name])
        result[(targets >= start) & (targets <= end)] = name.upper()
    return result


def construct_group_labels(raw: pd.DataFrame, group: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    spec = GROUPS[group]
    manufacturer = str(spec["manufacturer"])
    turbines = tuple(spec["turbines"])
    rated = float(spec["rated"])
    source = raw["kst_dtm"].copy()
    target = source.dt.floor("h") + pd.Timedelta(hours=1)
    power = np.column_stack([
        pd.to_numeric(raw[_column(manufacturer, t, "power_kw10m")], errors="coerce").to_numpy(float) / rated
        for t in turbines
    ])
    wind = np.column_stack([
        pd.to_numeric(raw[_column(manufacturer, t, "ws")], errors="coerce").to_numpy(float)
        for t in turbines
    ])
    fit_start, fit_end = map(pd.Timestamp, spec["potential_fit"])
    fit_time = (source >= fit_start) & (source < fit_end)
    initial_curves = []
    initial_potential = np.full(power.shape, np.nan, dtype=np.float64)
    for position, _ in enumerate(turbines):
        curve = fit_upper_envelope(wind[fit_time, position], power[fit_time, position])
        initial_curves.append(curve)
        initial_potential[:, position] = curve.predict(wind[:, position])
    initial_normal = _normal_from_curve(power, initial_potential, wind)
    refined_curves = []
    refined_potential = np.full(power.shape, np.nan, dtype=np.float64)
    refit_failures: list[int] = []
    for position, turbine in enumerate(turbines):
        refit = fit_time.to_numpy() & initial_normal[:, position]
        try:
            curve = fit_upper_envelope(wind[refit, position], power[refit, position])
        except ValueError:
            curve = initial_curves[position]
            refit_failures.append(int(turbine))
        refined_curves.append(curve)
        refined_potential[:, position] = curve.predict(wind[:, position])
    refined_normal = _normal_from_curve(power, refined_potential, wind)
    states, ratio, peer = assign_proxy_states(power, wind, refined_potential)
    union = initial_normal | refined_normal
    intersection = initial_normal & refined_normal
    jaccard = float(intersection.sum() / union.sum()) if union.any() else 0.0
    grid = np.arange(3.0, 20.0001, 0.1)
    curve_changes = [
        float(np.median(np.abs(before.predict(grid) - after.predict(grid))))
        for before, after in zip(initial_curves, refined_curves)
    ]

    frames: list[pd.DataFrame] = []
    target_window = _window(target, spec)
    for position, turbine in enumerate(turbines):
        frames.append(pd.DataFrame({
            "source_kst_dtm": source,
            "forecast_kst_dtm": target,
            "group": group,
            "turbine": int(turbine),
            "window": target_window,
            "power_cf": power[:, position],
            "wind_ms": wind[:, position],
            "potential_cf": refined_potential[:, position],
            "delivered_to_potential_ratio": ratio[:, position],
            "leave_one_out_peer_ratio": peer[:, position],
            "state": states[:, position],
        }))
    labels = pd.concat(frames, ignore_index=True).sort_values(
        ["forecast_kst_dtm", "turbine", "source_kst_dtm"], kind="stable"
    ).reset_index(drop=True)

    hourly_rows: list[dict[str, Any]] = []
    for timestamp, part in labels.groupby("forecast_kst_dtm", sort=True):
        counts = part["state"].value_counts()
        total = len(part)
        eligible = (
            part["power_cf"].between(0.0, 1.05)
            & part["wind_ms"].between(3.0, 20.0)
            & part["potential_cf"].ge(0.10)
            & part[["power_cf", "wind_ms", "potential_cf"]].notna().all(axis=1)
        )
        denominator = float(part.loc[eligible, "potential_cf"].sum())
        availability = (
            float(np.clip(part.loc[eligible, "power_cf"].sum() / denominator, 0.0, 1.0))
            if denominator > 0 else float("nan")
        )
        expected_cells = 6 * len(turbines)
        delivered_cf = float(part.loc[part["power_cf"].between(0.0, 1.05), "power_cf"].sum() / expected_cells)
        potential_cf = float(part.loc[part["potential_cf"].notna(), "potential_cf"].sum() / expected_cells)
        row: dict[str, Any] = {
            "forecast_kst_dtm": timestamp,
            "group": group,
            "window": part["window"].iloc[0],
            "resource_eligible_cells": int(eligible.sum()),
            "availability_factor": availability,
            "delivered_scada_cf": delivered_cf,
            "potential_scada_cf": potential_cf,
            "state_entropy": state_entropy([int(counts.get(state, 0)) for state in STATES]),
        }
        for state, column in STATE_TO_COLUMN.items():
            row[column] = float(counts.get(state, 0) / total)
        row["unknown_resource_fraction"] = float(
            ((part["state"].eq("UNKNOWN")) & eligible).sum() / max(int(eligible.sum()), 1)
        )
        row["normal_hour"] = bool(
            np.isfinite(availability) and availability >= 0.85 and row["normal_fraction"] >= 0.75
        )
        hourly_rows.append(row)
    hourly = pd.DataFrame(hourly_rows).sort_values("forecast_kst_dtm", kind="stable").reset_index(drop=True)
    curve_rows = []
    for position, turbine in enumerate(turbines):
        curve_rows.append({
            "turbine": int(turbine),
            "initial_populated_bins": initial_curves[position].populated_bins,
            "refined_populated_bins": refined_curves[position].populated_bins,
            "initial_fit_rows": initial_curves[position].fit_rows,
            "refined_fit_rows": refined_curves[position].fit_rows,
            "median_abs_curve_change": curve_changes[position],
            "refit_fallback": int(turbine) in refit_failures,
        })
    audit = {
        "group": group,
        "normal_jaccard": jaccard,
        "median_abs_curve_change": float(np.median(curve_changes)),
        "maximum_abs_curve_change": float(np.max(curve_changes)),
        "refit_failures": refit_failures,
        "curve_rows": curve_rows,
    }
    return labels, hourly, audit


def schema_audit(raw_frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for manufacturer, frame in raw_frames.items():
        for column in frame.columns:
            if column == "kst_dtm":
                rows.append({
                    "manufacturer": manufacturer, "column": column, "role": "timestamp",
                    "rows": len(frame), "finite_rows": len(frame), "missing_rows": 0,
                    "minimum": frame[column].min().isoformat(), "maximum": frame[column].max().isoformat(),
                    "out_of_registered_range": 0,
                })
                continue
            values = pd.to_numeric(frame[column], errors="coerce")
            role = "power_kw10m" if "power_kw10m" in column else ("wind_speed" if column.endswith("_ws") else "wind_direction")
            if role == "power_kw10m":
                upper = 630.0 if manufacturer == "vestas" else 735.0
                invalid = values.notna() & ~values.between(0.0, upper)
            elif role == "wind_speed":
                invalid = values.notna() & ~values.between(0.0, 40.0)
            else:
                # Direction is recorded in both signed and unsigned conventions;
                # it is audited but not used in the state mechanism.
                invalid = pd.Series(False, index=values.index)
            finite = np.isfinite(values.to_numpy(float))
            rows.append({
                "manufacturer": manufacturer, "column": column, "role": role,
                "rows": len(frame), "finite_rows": int(finite.sum()),
                "missing_rows": int((~finite).sum()),
                "minimum": float(np.nanmin(values)) if finite.any() else np.nan,
                "maximum": float(np.nanmax(values)) if finite.any() else np.nan,
                "out_of_registered_range": int(invalid.sum()),
            })
    return pd.DataFrame(rows)


def reliability_audits(hourly: pd.DataFrame, curve_audits: Mapping[str, Mapping[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, bool]]:
    normal_rows: list[dict[str, Any]] = []
    availability_rows: list[dict[str, Any]] = []
    group_pass: dict[str, bool] = {}
    for group in GROUPS:
        part = hourly.loc[hourly["group"].eq(group)]
        distributions: dict[str, np.ndarray] = {}
        window_passes: list[bool] = []
        for window in ("STUDENT_TRAIN", "VALIDATION"):
            selected = part.loc[part["window"].eq(window)].copy()
            state_props = np.asarray([selected[column].mean() for column in STATE_TO_COLUMN.values()], float)
            distributions[window] = state_props
            resource_hours = int(selected["availability_factor"].notna().sum())
            unknown = float(selected["unknown_resource_fraction"].mean())
            normal_rows.append({
                "group": group, "window": window, "hours": len(selected),
                "valid_resource_hours": resource_hours,
                "normal_hours": int(selected["normal_hour"].sum()),
                "normal_hour_fraction": float(selected["normal_hour"].mean()),
                "unknown_resource_fraction": unknown,
                **{column: float(selected[column].mean()) for column in STATE_TO_COLUMN.values()},
            })
            finite_availability = selected["availability_factor"].dropna()
            availability_rows.append({
                "group": group, "window": window, "hours": len(selected),
                "finite_availability_hours": len(finite_availability),
                "availability_mean": float(finite_availability.mean()),
                "availability_std": float(finite_availability.std(ddof=0)),
                "availability_q05": float(finite_availability.quantile(0.05)),
                "availability_q50": float(finite_availability.quantile(0.50)),
                "availability_q95": float(finite_availability.quantile(0.95)),
                "delivered_scada_cf_mean": float(selected["delivered_scada_cf"].mean()),
                "potential_scada_cf_mean": float(selected["potential_scada_cf"].mean()),
                "normal_hour_fraction": float(selected["normal_hour"].mean()),
            })
            window_passes.append(resource_hours >= 500 and unknown <= 0.15)
        js = jensen_shannon(distributions["STUDENT_TRAIN"], distributions["VALIDATION"])
        for row in normal_rows[-2:]:
            row["state_distribution_js_train_vs_validation"] = js
            row["normal_jaccard_fit"] = curve_audits[group]["normal_jaccard"]
            row["median_abs_curve_change"] = curve_audits[group]["median_abs_curve_change"]
        curve_rows = curve_audits[group]["curve_rows"]
        bins_pass = all(min(int(row["initial_populated_bins"]), int(row["refined_populated_bins"])) >= 10 for row in curve_rows)
        group_pass[group] = bool(
            all(window_passes)
            and bins_pass
            and float(curve_audits[group]["normal_jaccard"]) >= 0.60
            and float(curve_audits[group]["median_abs_curve_change"]) <= 0.15
            and js <= 0.15
            and not curve_audits[group]["refit_failures"]
        )
    return pd.DataFrame(availability_rows), pd.DataFrame(normal_rows), group_pass


def read_weather(group: str, path: Path, end: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, Any]]:
    schema = pq.ParquetFile(path).schema_arrow.names
    required = [*FEATURES, "forecast_kst_dtm"]
    missing = sorted(set(required) - set(schema))
    if missing:
        raise RuntimeError(f"weather schema missing frozen columns: {missing}")
    table = pq.read_table(
        path,
        columns=required,
        filters=[("forecast_kst_dtm", "<", end.to_pydatetime())],
    )
    # The cache stores forecast_kst_dtm as a pandas index in schema metadata.
    # Ignore that metadata so the explicitly requested field remains a column.
    frame = table.to_pandas(ignore_metadata=True)
    frame["forecast_kst_dtm"] = pd.to_datetime(frame["forecast_kst_dtm"], errors="raise")
    frame = frame.loc[frame["forecast_kst_dtm"] < end].copy()
    if frame.empty or frame["forecast_kst_dtm"].max() >= end:
        raise RuntimeError("weather filter crossed frozen end or yielded no rows")
    if frame["forecast_kst_dtm"].duplicated().any():
        raise RuntimeError("weather target timestamps are not unique")
    return frame, {
        "group": group, "path": str(path), "materialized_rows": len(frame),
        "materialized_min": frame["forecast_kst_dtm"].min().isoformat(),
        "materialized_max": frame["forecast_kst_dtm"].max().isoformat(),
        "materialized_2024_or_later_rows": int((frame["forecast_kst_dtm"] >= pd.Timestamp("2024-01-01")).sum()),
        "selected_columns": required,
    }


def fit_predict(model_x: pd.DataFrame, model_y: pd.Series, validation_x: pd.DataFrame) -> np.ndarray:
    model = HistGradientBoostingRegressor(**MODEL_PARAMS)
    model.fit(model_x.loc[:, FEATURES].to_numpy(float), model_y.to_numpy(float))
    return model.predict(validation_x.loc[:, FEATURES].to_numpy(float))


def student_predictability(hourly: pd.DataFrame, weather: Mapping[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, bool]]:
    rows: list[dict[str, Any]] = []
    passes: dict[str, bool] = {}
    for group, spec in GROUPS.items():
        merged = weather[group].merge(
            hourly.loc[hourly["group"].eq(group)], on="forecast_kst_dtm", how="inner", validate="one_to_one"
        )
        train = merged.loc[merged["window"].eq("STUDENT_TRAIN")].copy()
        validation = merged.loc[merged["window"].eq("VALIDATION")].copy()
        train_valid = train["availability_factor"].notna()
        validation_valid = validation["availability_factor"].notna()
        train = train.loc[train_valid]
        validation = validation.loc[validation_valid]
        prediction = np.clip(fit_predict(train, train["availability_factor"], validation), 0.0, 1.0)
        actual = validation["availability_factor"].to_numpy(float)
        constant = np.full(len(validation), float(train["availability_factor"].median()))
        month_median = train.assign(month=train["forecast_kst_dtm"].dt.month).groupby("month")["availability_factor"].median()
        monthly = validation["forecast_kst_dtm"].dt.month.map(month_median).fillna(float(train["availability_factor"].median())).to_numpy(float)
        mae_model = float(mean_absolute_error(actual, prediction))
        mae_constant = float(mean_absolute_error(actual, constant))
        mae_month = float(mean_absolute_error(actual, monthly))
        best_baseline = min(mae_constant, mae_month)
        improvement = float((best_baseline - mae_model) / best_baseline) if best_baseline > 0 else float("-inf")
        rho = float(spearmanr(actual, prediction, nan_policy="raise").statistic)
        r2 = float(r2_score(actual, prediction))
        binary = validation["normal_hour"].astype(int).to_numpy()
        auc = float(roc_auc_score(binary, prediction)) if np.unique(binary).size == 2 else float("nan")
        passed = bool(r2 >= 0.02 and rho >= 0.15 and improvement >= 0.02 and (not np.isfinite(auc) or auc >= 0.60))
        passes[group] = passed
        rows.append({
            "group": group, "train_rows": len(train), "validation_rows": len(validation),
            "validation_r2": r2, "validation_spearman": rho,
            "validation_mae_model": mae_model, "validation_mae_constant": mae_constant,
            "validation_mae_month": mae_month, "relative_mae_improvement_best_baseline": improvement,
            "normal_hour_auc": auc, "gate_pass": passed,
        })
    return pd.DataFrame(rows), passes


def q5_direct_comparison(
    hourly: pd.DataFrame,
    weather: Mapping[str, pd.DataFrame],
    official_labels: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group, spec in GROUPS.items():
        target_column = str(spec["official"])
        capacity = float(CAPACITY_KWH[target_column])
        merged = weather[group].merge(
            hourly.loc[hourly["group"].eq(group), ["forecast_kst_dtm", "window", "normal_hour"]],
            on="forecast_kst_dtm", how="inner", validate="one_to_one",
        ).merge(
            official_labels.loc[:, ["kst_dtm", target_column]].rename(columns={"kst_dtm": "forecast_kst_dtm"}),
            on="forecast_kst_dtm", how="inner", validate="one_to_one",
        )
        merged["official_cf"] = pd.to_numeric(merged[target_column], errors="coerce") / capacity
        train = merged.loc[merged["window"].eq("STUDENT_TRAIN") & merged["official_cf"].notna()].copy()
        validation = merged.loc[merged["window"].eq("VALIDATION") & merged["official_cf"].notna()].copy()
        direct_cf = np.clip(fit_predict(train, train["official_cf"], validation), 0.0, 1.10)
        normal_train = train.loc[train["normal_hour"]].copy()
        if len(normal_train) < 200:
            potential_cf = np.full(len(validation), np.nan)
            delta = float("nan")
            direct_metrics = group_metrics(validation[target_column], direct_cf * capacity, capacity, group_name=target_column)
            potential_metrics = None
        else:
            potential_cf = np.clip(fit_predict(normal_train, normal_train["official_cf"], validation), 0.0, 1.10)
            direct_metrics = group_metrics(validation[target_column], direct_cf * capacity, capacity, group_name=target_column)
            potential_metrics = group_metrics(validation[target_column], potential_cf * capacity, capacity, group_name=target_column)
            direct_score = 0.5 * (direct_metrics.one_minus_nmae + direct_metrics.ficr)
            potential_score = 0.5 * (potential_metrics.one_minus_nmae + potential_metrics.ficr)
            delta = float(potential_score - direct_score)
        rows.append({
            "group": group, "train_rows_direct": len(train), "train_rows_normal_only": len(normal_train),
            "validation_rows": len(validation),
            "direct_score": 0.5 * (direct_metrics.one_minus_nmae + direct_metrics.ficr),
            "direct_one_minus_nmae": direct_metrics.one_minus_nmae,
            "direct_ficr": direct_metrics.ficr,
            "normal_only_score": np.nan if potential_metrics is None else 0.5 * (potential_metrics.one_minus_nmae + potential_metrics.ficr),
            "normal_only_one_minus_nmae": np.nan if potential_metrics is None else potential_metrics.one_minus_nmae,
            "normal_only_ficr": np.nan if potential_metrics is None else potential_metrics.ficr,
            "score_delta_normal_only_minus_direct": delta,
        })
    frame = pd.DataFrame(rows)
    deltas = frame["score_delta_normal_only_minus_direct"].to_numpy(float)
    summary = {
        "mean_group_score_delta": float(np.mean(deltas)) if np.all(np.isfinite(deltas)) else None,
        "positive_group_count": int((deltas > 0).sum()) if np.all(np.isfinite(deltas)) else 0,
        "worst_group_score_delta": float(np.min(deltas)) if np.all(np.isfinite(deltas)) else None,
    }
    summary["gate_pass"] = bool(
        summary["mean_group_score_delta"] is not None
        and summary["mean_group_score_delta"] >= 0.003
        and summary["mean_group_score_delta"] > 0.0
        and summary["positive_group_count"] >= 2
        and summary["worst_group_score_delta"] >= -0.003
    )
    return frame, summary


def write_state_definition(path: Path) -> None:
    path.write_text(
        "# Track B SCADA proxy-state definition\n\n"
        "These are deterministic observable-pattern labels, not confirmed operational events. "
        "The raw files expose only per-turbine power, wind speed and wind direction. They contain "
        "no alarm/status, pitch, rotor/generator speed or power-limit channel. Consequently "
        "`CURTAILMENT_LIKE` and `OUTAGE_LIKE` mean only that delivered/potential ratios and "
        "contemporaneous peers match the frozen pattern.\n\n"
        "Potential is a per-turbine 0.90-quantile, 0.5 m/s-binned, PAVA-monotone curve. It is "
        "estimated in the registered early prefix and refined exactly once using initial NORMAL "
        "rows. Validation rows never refit it. State-rule order and thresholds are those in the "
        "v1 preregistration. The availability student uses only the 13 frozen weather/calendar "
        "features.\n\n"
        "The Q5 normal-only comparison is an append-only requested diagnostic. It shares the "
        "existing turbine-SCADA conceptual family and cannot override a failed state/student gate.\n",
        encoding="utf-8",
    )


def write_decision(
    path: Path,
    state_passes: Mapping[str, bool],
    student_passes: Mapping[str, bool],
    q5_summary: Mapping[str, Any],
) -> str:
    if not all(state_passes.values()):
        decision = "TRACK_B_STOP_SCADA_STATE_UNRELIABLE"
    elif not all(student_passes.values()):
        decision = "TRACK_B_STOP_AVAILABILITY_UNPREDICTABLE"
    elif not bool(q5_summary["gate_pass"]):
        decision = "TRACK_B_STOP_NORMAL_ONLY_NO_ROBUST_UPLIFT"
    else:
        decision = "TRACK_B_GO_RESEARCH_ONLY_NO_2024_OR_SUBMISSION_YET"
    path.write_text(
        "# Track B decision\n\n"
        f"Decision: `{decision}`\n\n"
        f"State reliability by group: `{dict(state_passes)}`.\n\n"
        f"Weather-only availability predictability by group: `{dict(student_passes)}`.\n\n"
        f"Frozen Q5 normal-only minus direct summary: `{dict(q5_summary)}`.\n\n"
        "This late, selection-unsafe pilot used no Public score as a selector. It materialized no "
        "2024/test/2025 rows and created no prediction or submission CSV. A STOP has precedence; "
        "no rescue, blend, threshold adjustment, or inference is authorized.\n",
        encoding="utf-8",
    )
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    if out_dir != DEFAULT_OUT.resolve():
        raise RuntimeError(f"only canonical Track-B directory is allowed: {DEFAULT_OUT}")
    for path, (size, digest) in EXPECTED_PREREG.items():
        verify_file(path, size, digest)
    out_dir.mkdir(parents=True, exist_ok=True)
    protected = [
        "SCADA_SCHEMA_AUDIT.csv", "SCADA_STATE_DEFINITION.md", "SCADA_STATE_LABELS.parquet",
        "AVAILABILITY_TARGET_AUDIT.csv", "AVAILABILITY_PREDICTABILITY.csv",
        "NORMAL_SAMPLE_AUDIT.csv", "Q5_NORMAL_ONLY_VS_DIRECT.csv",
        "TRACK_B_DECISION.md", "TRACK_B_RUN_MANIFEST.json",
    ]
    existing = [name for name in protected if (out_dir / name).exists()]
    if existing:
        raise RuntimeError(f"no-overwrite guard: {existing}")

    raw_specs = {
        "vestas": (Path(r"data/local/open/train/scada_vestas_train.csv"), 33458140, "f024f3ca57bbe2a6106f515d32c1457d75f519fb27068308680b20fb6f04dcc1", pd.Timestamp("2023-07-01"), tuple(range(1, 13))),
        "unison": (Path(r"data/local/open/train/scada_unison_train.csv"), 16788646, "5d8ccd7ac6b127865d0b2f18de257ceb23dcbf3b0056c209e8b02f1470c6c9be", pd.Timestamp("2023-10-01"), tuple(range(1, 6))),
    }
    raw_frames: dict[str, pd.DataFrame] = {}
    prefix_records: dict[str, Any] = {}
    for manufacturer, (path, size, digest, cutoff, turbines) in raw_specs.items():
        verify_file(path, size, digest)
        columns = ["kst_dtm"]
        for measurement in ("power_kw10m", "ws", "wd"):
            columns.extend(_column(manufacturer, turbine, measurement) for turbine in turbines)
        raw_frames[manufacturer], prefix_records[manufacturer] = read_bounded_csv(path, cutoff, columns)

    schema = schema_audit(raw_frames)
    schema.to_csv(out_dir / "SCADA_SCHEMA_AUDIT.csv", index=False, encoding="utf-8")
    label_parts: list[pd.DataFrame] = []
    hourly_parts: list[pd.DataFrame] = []
    curve_audits: dict[str, Any] = {}
    for group in GROUPS:
        labels, hourly, audit = construct_group_labels(raw_frames[str(GROUPS[group]["manufacturer"])], group)
        label_parts.append(labels)
        hourly_parts.append(hourly)
        curve_audits[group] = audit
    labels = pd.concat(label_parts, ignore_index=True).sort_values(
        ["forecast_kst_dtm", "group", "turbine", "source_kst_dtm"], kind="stable"
    ).reset_index(drop=True)
    if (labels["forecast_kst_dtm"] >= pd.Timestamp("2023-10-01 01:00:00")).any():
        raise RuntimeError("SCADA label output crossed frozen Track-B horizon")
    labels.to_parquet(out_dir / "SCADA_STATE_LABELS.parquet", engine="pyarrow", compression="zstd", index=False)
    hourly = pd.concat(hourly_parts, ignore_index=True).sort_values(["forecast_kst_dtm", "group"], kind="stable").reset_index(drop=True)
    availability_audit, normal_audit, state_passes = reliability_audits(hourly, curve_audits)
    availability_audit.to_csv(out_dir / "AVAILABILITY_TARGET_AUDIT.csv", index=False)
    normal_audit.to_csv(out_dir / "NORMAL_SAMPLE_AUDIT.csv", index=False)

    weather_paths = {
        "G1": ROOT / "artifacts/cache/kpx_group_1_weather_train.parquet",
        "G2": ROOT / "artifacts/cache/kpx_group_2_weather_train.parquet",
        "G3": ROOT / "artifacts/cache/kpx_group_3_weather_train.parquet",
    }
    weather_expected = {
        "G1": (77942069, "c3526f861184a16fef4a68c20ad8f4defab04b7dc0650b571c6beba05a867579"),
        "G2": (77929519, "0e6fc7334e7094af1a2fffceffeee7628930aaa102c43515fde3315ec806a31e"),
        "G3": (77950424, "eb61868a0fd564a60f31fdd9b1fd6324545a168fda9b0137049754d014b332ce"),
    }
    weather: dict[str, pd.DataFrame] = {}
    weather_records: dict[str, Any] = {}
    for group, path in weather_paths.items():
        verify_file(path, *weather_expected[group])
        end = pd.Timestamp(GROUPS[group]["validation"][1]) + pd.Timedelta(seconds=1)
        weather[group], weather_records[group] = read_weather(group, path, end)
        if weather_records[group]["materialized_2024_or_later_rows"] != 0:
            raise RuntimeError("2024 weather materialized")
    predictability, student_passes = student_predictability(hourly, weather)

    official_path = Path(r"data/local/open/train/train_labels.csv")
    verify_file(official_path, 1138967, "47bb64252195cf4734e67394d6e50485f27a608def3b5a8791fcc7674bbceb03")
    official_labels, official_prefix = read_bounded_csv(
        official_path, pd.Timestamp("2023-10-01 01:00:00"),
        ["kst_dtm", "kpx_group_1", "kpx_group_2", "kpx_group_3"],
    )
    if (official_labels["kst_dtm"] >= pd.Timestamp("2024-01-01")).any():
        raise RuntimeError("2024 official label materialized")
    q5, q5_summary = q5_direct_comparison(hourly, weather, official_labels)
    predictability = predictability.merge(
        q5.loc[:, ["group", "score_delta_normal_only_minus_direct"]], on="group", how="left", validate="one_to_one"
    )
    predictability.to_csv(out_dir / "AVAILABILITY_PREDICTABILITY.csv", index=False)
    q5.to_csv(out_dir / "Q5_NORMAL_ONLY_VS_DIRECT.csv", index=False)
    write_state_definition(out_dir / "SCADA_STATE_DEFINITION.md")
    decision = write_decision(out_dir / "TRACK_B_DECISION.md", state_passes, student_passes, q5_summary)

    output_records = {
        name: file_record(out_dir / name)
        for name in protected
        if name != "TRACK_B_RUN_MANIFEST.json" and (out_dir / name).exists()
    }
    manifest = {
        "schema_version": 1,
        "artifact_type": "track_b_scada_latent_state_causal_student_falsification_run",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "preregistrations": [file_record(path) for path in EXPECTED_PREREG],
        "source_prefixes": {**prefix_records, "official_labels": official_prefix},
        "weather_materialization": weather_records,
        "curve_audits": curve_audits,
        "state_gate_by_group": state_passes,
        "student_gate_by_group": student_passes,
        "q5_summary": q5_summary,
        "outputs": output_records,
        "forbidden_access_counters": {
            "public_feedback_reads": 0,
            "public_selector_uses": 0,
            "materialized_2024_rows": 0,
            "test_rows": 0,
            "materialized_2025_rows": 0,
            "test_inference_calls": 0,
            "prediction_files_created": 0,
            "submission_csv_files_created": 0,
            "post_validation_rescues": 0,
        },
        "semantic_limit": "Proxy states are not confirmed curtailment/outage events.",
        "conceptual_duplicate": "Q5 normal-only comparison overlaps existing turbine_scada_power and is not an independent hypothesis.",
    }
    (out_dir / "TRACK_B_RUN_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"decision": decision, "state": state_passes, "student": student_passes, "q5": q5_summary}, indent=2))


if __name__ == "__main__":
    main()

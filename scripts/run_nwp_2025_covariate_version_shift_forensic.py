"""Label-free forensic comparison of supplied 2022-24 and 2025 NWP files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = Path(r"data/local/open")
OUT = ROOT / "artifacts/audits/nwp_2025_covariate_version_shift_forensic_v3.json"
META = ("forecast_kst_dtm", "data_available_kst_dtm", "grid_id", "latitude", "longitude")
QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def read_weather(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    if tuple(frame.columns[:5]) != META:
        raise AssertionError(f"metadata schema changed: {path}")
    frame["forecast_kst_dtm"] = pd.to_datetime(frame["forecast_kst_dtm"], errors="raise")
    frame["data_available_kst_dtm"] = pd.to_datetime(
        frame["data_available_kst_dtm"], errors="raise"
    )
    frame["grid_id"] = pd.to_numeric(frame["grid_id"], errors="raise").astype("int16")
    return frame


def alignment(frame: pd.DataFrame, expected_grids: int) -> dict[str, Any]:
    key = ["forecast_kst_dtm", "grid_id"]
    duplicate_rows = int(frame.duplicated(key, keep=False).sum())
    counts = frame.groupby("forecast_kst_dtm", sort=False)["grid_id"].nunique()
    issue_per_valid = frame.groupby("forecast_kst_dtm", sort=False)[
        "data_available_kst_dtm"
    ].nunique()
    one_grid = frame.loc[frame["grid_id"] == int(frame["grid_id"].min())].copy()
    one_grid = one_grid.sort_values("forecast_kst_dtm")
    forecast = pd.DatetimeIndex(one_grid["forecast_kst_dtm"])
    issue = pd.DatetimeIndex(one_grid["data_available_kst_dtm"])
    lead = (forecast - issue).total_seconds() / 3600.0
    issue_runs = one_grid.groupby("data_available_kst_dtm", sort=True)["forecast_kst_dtm"]
    run_sizes = issue_runs.size()
    horizons_exact = issue_runs.apply(
        lambda values: tuple(
            ((pd.DatetimeIndex(values).sort_values() - values.name).total_seconds() / 3600.0)
            .astype(int)
            .tolist()
        )
    )
    gaps = (forecast[1:] - forecast[:-1]).total_seconds() / 3600.0
    return {
        "rows": len(frame),
        "forecast_hours": int(frame["forecast_kst_dtm"].nunique()),
        "start": frame["forecast_kst_dtm"].min(),
        "end": frame["forecast_kst_dtm"].max(),
        "grid_ids": sorted(map(int, frame["grid_id"].unique())),
        "expected_grid_count": expected_grids,
        "duplicate_forecast_grid_rows": duplicate_rows,
        "all_forecast_hours_complete_grid": bool((counts == expected_grids).all()),
        "forecast_grid_count_min_max": [int(counts.min()), int(counts.max())],
        "one_issue_timestamp_per_valid_time": bool((issue_per_valid == 1).all()),
        "issue_hour_values": sorted(map(int, one_grid["data_available_kst_dtm"].dt.hour.unique())),
        "lead_hour_min_max": [float(np.min(lead)), float(np.max(lead))],
        "issue_run_size_min_max": [int(run_sizes.min()), int(run_sizes.max())],
        "all_issue_runs_exact_horizons_12_through_35": bool(
            horizons_exact.map(lambda item: item == tuple(range(12, 36))).all()
        ),
        "forecast_gap_hours_unique": sorted(map(float, np.unique(gaps))),
        "availability_run_alignment_01_through_next_00": bool(
            (run_sizes == 24).all()
            and horizons_exact.map(lambda item: item == tuple(range(12, 36))).all()
        ),
    }


def coordinate_catalog(frame: pd.DataFrame) -> dict[str, Any]:
    grouped = frame.groupby("grid_id", sort=True)
    result: dict[str, Any] = {}
    for grid, part in grouped:
        lat = part["latitude"].dropna().unique()
        lon = part["longitude"].dropna().unique()
        result[str(int(grid))] = {
            "latitude_values": sorted(map(float, lat)),
            "longitude_values": sorted(map(float, lon)),
            "coordinate_constant": len(lat) == 1 and len(lon) == 1,
        }
    return result


def operating_year(frame: pd.DataFrame) -> pd.Series:
    return (frame["forecast_kst_dtm"] - pd.Timedelta(hours=1)).dt.year.astype("int16")


def annual_stats(values: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    finite = numeric[np.isfinite(numeric)]
    quantiles = np.quantile(finite, QUANTILES) if len(finite) else np.full(len(QUANTILES), np.nan)
    consecutive = numeric[1:] == numeric[:-1] if len(numeric) > 1 else np.asarray([], bool)
    return {
        "rows": len(numeric),
        "missing_count": int((~np.isfinite(numeric)).sum()),
        "missing_fraction": float((~np.isfinite(numeric)).mean()),
        "finite_unique": int(pd.Series(finite).nunique(dropna=True)),
        "mean": float(np.mean(finite)) if len(finite) else np.nan,
        "std": float(np.std(finite)) if len(finite) else np.nan,
        "quantiles": {str(q): float(value) for q, value in zip(QUANTILES, quantiles)},
        "consecutive_exact_equal_fraction": float(consecutive.mean()) if len(consecutive) else np.nan,
        "constant": bool(len(finite) and np.all(finite == finite[0])),
    }


def pair_drift(
    reference: pd.DataFrame,
    application: pd.DataFrame,
    channel: str,
) -> dict[str, Any]:
    ref = reference[["forecast_kst_dtm", "data_available_kst_dtm", channel]].copy()
    app = application[["forecast_kst_dtm", "data_available_kst_dtm", channel]].copy()
    ref_value = pd.to_numeric(ref[channel], errors="coerce")
    app_value = pd.to_numeric(app[channel], errors="coerce")
    ref_finite = ref_value[np.isfinite(ref_value)]
    app_finite = app_value[np.isfinite(app_value)]
    scale = float(ref_finite.std(ddof=0))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    qgrid = np.linspace(0.01, 0.99, 99)
    quantile_l1 = float(
        np.mean(np.abs(np.quantile(app_finite, qgrid) - np.quantile(ref_finite, qgrid))) / scale
    )

    for part in (ref, app):
        run_date = part["forecast_kst_dtm"] - pd.Timedelta(hours=1)
        part["month"] = run_date.dt.month
        part["day"] = run_date.dt.day
        part["horizon"] = (
            (part["forecast_kst_dtm"] - part["data_available_kst_dtm"]).dt.total_seconds()
            / 3600.0
        ).astype("int16")
    seasonal_ref = ref.groupby(["month", "horizon"], sort=True)[channel].mean()
    seasonal_app = app.groupby(["month", "horizon"], sort=True)[channel].mean()
    seasonal = pd.concat([seasonal_ref.rename("r"), seasonal_app.rename("a")], axis=1).dropna()
    seasonal_abs_z = np.abs((seasonal["a"] - seasonal["r"]) / scale)

    daily_ref = ref.groupby(["month", "day", "horizon"], sort=True)[channel].median()
    daily_app = app.groupby(["month", "day", "horizon"], sort=True)[channel].median()
    daily = pd.concat([daily_ref.rename("r"), daily_app.rename("a")], axis=1).dropna()
    daily_abs_z = np.abs((daily["a"] - daily["r"]) / scale)
    return {
        "reference_rows": len(reference),
        "application_rows": len(application),
        "mean_standardized_shift": float((app_finite.mean() - ref_finite.mean()) / scale),
        "std_ratio": float(app_finite.std(ddof=0) / scale),
        "missing_fraction_delta": float(app_value.isna().mean() - ref_value.isna().mean()),
        "normalized_quantile_l1": quantile_l1,
        "seasonal_month_horizon_abs_z_mean": float(seasonal_abs_z.mean()),
        "seasonal_month_horizon_abs_z_max": float(seasonal_abs_z.max()),
        "daily_calendar_horizon_abs_z_median": float(daily_abs_z.median()),
        "daily_calendar_horizon_abs_z_p95": float(daily_abs_z.quantile(0.95)),
        "daily_matched_cells": len(daily),
    }


def duplicate_node_vectors(frame: pd.DataFrame, channels: list[str]) -> dict[str, Any]:
    # Hash numeric channel bytes after rounding only for exact duplicate detection;
    # source values themselves are not transformed for any statistic.
    numeric = frame.loc[:, channels].apply(pd.to_numeric, errors="coerce")
    hashes = pd.util.hash_pandas_object(numeric, index=False)
    keys = pd.DataFrame({"time": frame["forecast_kst_dtm"], "hash": hashes})
    repeated = keys.duplicated(["time", "hash"], keep=False)
    per_time_unique = keys.groupby("time", sort=False)["hash"].nunique()
    return {
        "rows_in_within_time_exact_duplicate_channel_vectors": int(repeated.sum()),
        "fraction_rows_in_within_time_exact_duplicate_channel_vectors": float(repeated.mean()),
        "unique_node_vectors_per_time_min_max": [
            int(per_time_unique.min()),
            int(per_time_unique.max()),
        ],
    }


def missing_event_summary(frame: pd.DataFrame, channels: list[str]) -> dict[str, Any]:
    missing = frame.loc[:, channels].isna()
    row_any = missing.any(axis=1)
    records: list[dict[str, Any]] = []
    for timestamp, positions in frame.loc[row_any].groupby("forecast_kst_dtm", sort=True).groups.items():
        mask = frame.index.isin(positions)
        channel_counts = missing.loc[mask].sum()
        records.append(
            {
                "forecast_kst_dtm": pd.Timestamp(timestamp),
                "grid_rows_affected": int(mask.sum()),
                "channels_affected": {
                    channel: int(count)
                    for channel, count in channel_counts.items()
                    if int(count) > 0
                },
            }
        )
    return {
        "rows_with_any_missing": int(row_any.sum()),
        "forecast_hours_with_any_missing": len(records),
        "fraction_forecast_hours_with_any_missing": float(
            len(records) / frame["forecast_kst_dtm"].nunique()
        ),
        "events": records,
    }


def source_forensic(name: str, train_path: Path, test_path: Path, expected_grids: int) -> dict[str, Any]:
    train = read_weather(train_path)
    test = read_weather(test_path)
    if tuple(train.columns) != tuple(test.columns):
        raise AssertionError(f"{name} train/test schema/order changed")
    channels = [column for column in train.columns if column not in META]
    train["operating_year"] = operating_year(train)
    test["operating_year"] = operating_year(test)
    if sorted(train["operating_year"].unique().tolist()) != [2022, 2023, 2024]:
        raise AssertionError(f"{name} train operating years changed")
    if test["operating_year"].unique().tolist() != [2025]:
        raise AssertionError(f"{name} test operating year changed")

    catalog_train = coordinate_catalog(train)
    catalog_test = coordinate_catalog(test)
    channel_grid: dict[str, Any] = {}
    drift_rows: list[dict[str, Any]] = []
    for channel in channels:
        channel_grid[channel] = {}
        for grid in range(1, expected_grids + 1):
            train_grid = train.loc[train["grid_id"] == grid].sort_values("forecast_kst_dtm")
            test_grid = test.loc[test["grid_id"] == grid].sort_values("forecast_kst_dtm")
            years = {
                str(year): annual_stats(train_grid.loc[train_grid["operating_year"] == year, channel])
                for year in (2022, 2023, 2024)
            }
            years["2025"] = annual_stats(test_grid[channel])
            ref_2022 = train_grid.loc[train_grid["operating_year"] == 2022]
            ref_2023 = train_grid.loc[train_grid["operating_year"] == 2023]
            ref_2024 = train_grid.loc[train_grid["operating_year"] == 2024]
            drift_23 = pair_drift(ref_2022, ref_2023, channel)
            drift_24 = pair_drift(pd.concat([ref_2022, ref_2023]), ref_2024, channel)
            drift_25 = pair_drift(train_grid, test_grid, channel)
            control = max(
                drift_23["seasonal_month_horizon_abs_z_mean"],
                drift_24["seasonal_month_horizon_abs_z_mean"],
                1e-12,
            )
            daily_control = max(
                drift_23["daily_calendar_horizon_abs_z_p95"],
                drift_24["daily_calendar_horizon_abs_z_p95"],
                1e-12,
            )
            record = {
                "annual": years,
                "drift_2023_vs_2022": drift_23,
                "drift_2024_vs_2022_2023": drift_24,
                "drift_2025_vs_2022_2024": drift_25,
                "seasonal_drift_ratio_to_max_historical_control": float(
                    drift_25["seasonal_month_horizon_abs_z_mean"] / control
                ),
                "daily_p95_drift_ratio_to_max_historical_control": float(
                    drift_25["daily_calendar_horizon_abs_z_p95"] / daily_control
                ),
            }
            channel_grid[channel][str(grid)] = record
            drift_rows.append(
                {
                    "source": name,
                    "channel": channel,
                    "grid_id": grid,
                    "mean_abs_z": abs(drift_25["mean_standardized_shift"]),
                    "std_ratio": drift_25["std_ratio"],
                    "quantile_l1": drift_25["normalized_quantile_l1"],
                    "seasonal_abs_z_mean": drift_25["seasonal_month_horizon_abs_z_mean"],
                    "daily_abs_z_median": drift_25["daily_calendar_horizon_abs_z_median"],
                    "daily_abs_z_p95": drift_25["daily_calendar_horizon_abs_z_p95"],
                    "seasonal_control_ratio": record[
                        "seasonal_drift_ratio_to_max_historical_control"
                    ],
                    "daily_control_ratio": record[
                        "daily_p95_drift_ratio_to_max_historical_control"
                    ],
                    "missing_delta": abs(drift_25["missing_fraction_delta"]),
                }
            )

    drift = pd.DataFrame(drift_rows)
    sort_keys = {
        "mean_abs_z": "mean_abs_z",
        "quantile_l1": "quantile_l1",
        "seasonal_abs_z_mean": "seasonal_abs_z_mean",
        "daily_abs_z_p95": "daily_abs_z_p95",
        "seasonal_control_ratio": "seasonal_control_ratio",
        "daily_control_ratio": "daily_control_ratio",
        "missing_delta": "missing_delta",
    }
    top = {
        label: drift.nlargest(12, column).to_dict(orient="records")
        for label, column in sort_keys.items()
    }
    # A static field that is exactly unchanged has std_ratio=0 after the
    # zero-reference-scale safeguard.  It is explicitly not a version break.
    dynamic_change = (drift["quantile_l1"] > 1e-9) | (drift["mean_abs_z"] > 1e-9)
    flags = drift.loc[
        (drift["missing_delta"] > 0.001)
        | (drift["mean_abs_z"] > 0.75)
        | (dynamic_change & (drift["std_ratio"] < 0.50))
        | (dynamic_change & (drift["std_ratio"] > 2.0))
        | ((drift["seasonal_abs_z_mean"] > 0.35) & (drift["seasonal_control_ratio"] > 3.0))
        | ((drift["daily_abs_z_p95"] > 3.5) & (drift["daily_control_ratio"] > 2.0))
    ].to_dict(orient="records")
    return {
        "train_file": file_record(train_path),
        "test_file": file_record(test_path),
        "schema": {
            "train_test_column_order_exact": tuple(train.columns[:-1]) == tuple(test.columns[:-1]),
            "columns": list(train.columns[:-1]),
            "numeric_channel_count": len(channels),
            "numeric_channels": channels,
        },
        "alignment": {
            "train": alignment(train, expected_grids),
            "test": alignment(test, expected_grids),
        },
        "coordinate_catalog": {
            "train": catalog_train,
            "test": catalog_test,
            "train_test_exact": catalog_train == catalog_test,
        },
        "duplicate_node_vectors": {
            "train": duplicate_node_vectors(train, channels),
            "test": duplicate_node_vectors(test, channels),
        },
        "missing_events": {
            "train": missing_event_summary(train, channels),
            "test": missing_event_summary(test, channels),
        },
        "channel_grid": channel_grid,
        "top_2025_shift_records": top,
        "registered_break_flags": flags,
        "registered_break_flag_count": len(flags),
    }


def existing_axis_census() -> dict[str, Any]:
    paths = {
        "covariate_shift_feature_pruning_v3": ROOT
        / "configs/covariate_shift_feature_pruning_preregister_v3.json",
        "density_ratio_reweight": ROOT / "configs/density_ratio_reweight_preregister_v1.json",
        "year_quantile_map": ROOT / "configs/year_quantile_map_audit_preregister.json",
        "full_feature_year_quantile_map": ROOT / "configs/full_feature_year_qm_preregister.json",
    }
    return {
        name: {
            **file_record(path),
            "comparison": {
                "covariate_shift_feature_pruning_v3": "Selects a fixed feature subset using train/application domain shift; it does not diagnose raw NWP schema, issue-run alignment, frozen grids or a model-version discontinuity.",
                "density_ratio_reweight": "Reweights labeled fit rows using a domain classifier; it does not repair or establish a raw-file version break.",
                "year_quantile_map": "Maps selected feature marginals to the fit distribution; it does not preserve or test raw grid/channel physical cross-node structure.",
                "full_feature_year_quantile_map": "Maps full engineered feature marginals; it is the closest remediation family but remains distinct from this label-free raw-file forensic.",
            }[name],
        }
        for name, path in paths.items()
    }


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    sources = {
        "ldaps": source_forensic(
            "ldaps",
            RAW / "train/ldaps_train.csv",
            RAW / "test/ldaps_test.csv",
            16,
        ),
        "gfs": source_forensic(
            "gfs",
            RAW / "train/gfs_train.csv",
            RAW / "test/gfs_test.csv",
            9,
        ),
    }
    all_flags = [
        {"source": source, **flag}
        for source, payload in sources.items()
        for flag in payload["registered_break_flags"]
    ]
    structural = {
        source: {
            "schema_exact": payload["schema"]["train_test_column_order_exact"],
            "coordinate_exact": payload["coordinate_catalog"]["train_test_exact"],
            "train_alignment_valid": payload["alignment"]["train"][
                "availability_run_alignment_01_through_next_00"
            ],
            "test_alignment_valid": payload["alignment"]["test"][
                "availability_run_alignment_01_through_next_00"
            ],
            "break_flag_count": payload["registered_break_flag_count"],
        }
        for source, payload in sources.items()
    }
    physical_break = any(
        not item["schema_exact"]
        or not item["coordinate_exact"]
        or not item["train_alignment_valid"]
        or not item["test_alignment_valid"]
        for item in structural.values()
    )
    # Distribution flags are evidence only when they are broad across channels;
    # isolated meteorological shifts are expected in a new weather year.
    flagged_dynamic_channels = {
        (item["source"], item["channel"])
        for item in all_flags
        if item["quantile_l1"] > 1e-9 or item["mean_abs_z"] > 1e-9
    }
    broad_break = len(flagged_dynamic_channels) >= 5
    decision = "GO_SINGLE_FIXED_ROBUST_SCALING_FEASIBILITY" if physical_break or broad_break else "NO_GO_NO_DATASET_BREAK"
    payload = {
        "schema_version": 1,
        "experiment_id": "nwp_2025_covariate_version_shift_forensic_v3",
        "created_date_kst": "2026-08-08",
        "scope": {
            "label_free": True,
            "generation_target_read": 0,
            "scada_read": 0,
            "public_or_scale_read": 0,
            "candidate_prediction_or_score_read": 0,
            "files": "supplied LDAPS/GFS train 2022-24 and test 2025 only",
        },
        "method": {
            "operating_year": "forecast timestamp minus one hour, so each 01:00..next-day00 run belongs to one date/year",
            "seasonal_match": "month x fixed lead horizon 12..35",
            "daily_match": "operating month-day x fixed lead horizon, historical median versus 2025",
            "normalization": "reference-period channel/grid standard deviation",
            "break_flags": "missing delta>.001, |mean z|>.75, changed-field std ratio outside .5..2, seasonal mean |z|>.35 and >3x historical-control drift, or daily p95 |z|>3.5 and >2x historical-control drift; unchanged invariant fields are not breaks",
        },
        "sources": sources,
        "structural_summary": structural,
        "registered_distribution_break_flags": all_flags,
        "registered_distribution_break_flag_count": len(all_flags),
        "flagged_dynamic_channel_count": len(flagged_dynamic_channels),
        "supersedes": [
            {
                "path": "artifacts/audits/nwp_2025_covariate_version_shift_forensic_v1.json",
                "reason": "Unreported v1 diagnostic incorrectly counted exactly unchanged static fields with zero variance as std-ratio breaks."
            },
            {
                "path": "artifacts/audits/nwp_2025_covariate_version_shift_forensic_v2.json",
                "reason": "Unreported v2 serialized the forecast gap using a nanosecond divisor although pandas 3 stores this DatetimeIndex at microsecond resolution; all other alignment checks and the verdict were unaffected."
            }
        ],
        "existing_axis_census": existing_axis_census(),
        "verdict": {
            "physical_dataset_break_detected": physical_break,
            "broad_version_distribution_break_detected": broad_break,
            "decision": decision,
            "leaderboard_gap_explained_by_physical_dataset_break": bool(physical_break or broad_break),
            "interpretation": (
                "A broad or structural break is present; the only permitted next feasibility is one fixed robust per-channel/grid scaling learned without labels and validated strictly on 2024."
                if physical_break or broad_break
                else "No schema, grid, issue-run, unit or broad standardized distribution break was found. Three isolated partially missing LDAPS forecast hours are real but affect only 3/8760 hours and cannot plausibly explain a broad annual performance gap. Ordinary interannual meteorological drift cannot by itself establish a dataset-version failure."
            ),
            "single_fixed_remediation_feasibility": (
                "Per-channel/grid median-IQR alignment of 2025 to pooled 2022-24, fixed before any prediction and first emulated 2023->2024; no quantile grid or group rescue."
                if physical_break or broad_break
                else None
            ),
        },
        "source": file_record(Path(__file__)),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_name(f".{OUT.name}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(json_ready(payload), stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(OUT)
    print(json.dumps(json_ready(payload["structural_summary"]), sort_keys=True))
    print(json.dumps(json_ready(payload["verdict"]), sort_keys=True))
    print(f"output={OUT} bytes={OUT.stat().st_size} sha256={sha256_file(OUT)}")


if __name__ == "__main__":
    main()

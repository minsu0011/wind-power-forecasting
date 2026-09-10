"""Strict-forward SCADA physical residual experiment for all BARAM groups.

SCADA is used only inside the causal fit interval to learn (1) a robust
two-feature NWP-to-fleet-wind mapper and (2) per-turbine monotone wind-to-power
curves.  Validation/test inference consumes weather only.  The immutable
pre-registration limits the candidate search to two small blends active only
below 6 m/s or above 12 m/s predicted fleet wind.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.run_shared_q07_multiseed import (  # noqa: E402
    EXPECTED_ROWS_PRE2024,
    OFFICIAL_LABEL_PREFIX_SHA256,
    OFFICIAL_STAGE1_PREFIX_BYTES,
    YEAR_2022_START,
    YEAR_2023_END,
    YEAR_2023_START,
    YEAR_2024_END,
    YEAR_2024_START,
    _canonical_sha256,
    _frame_sha256,
    _read_labels,
    _read_stage1_raw_features,
)
from scripts.run_turbine_scada_power import (  # noqa: E402
    _discover_csv_prefix,
    _read_cached_features,
    _read_full_labels,
    _read_prediction,
    _read_scada_prefix,
    _stage1_baseline,
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
    "53b9094dd66054a387cddf77a65f3a09058c27303a3158d72cd89b89adde8a42"
)
FEATURE_COLUMNS = ("ldaps__idw__hub_ws", "gfs__idw__hub_ws")
WEIGHTS = (0.10, 0.20)
LOW_WIND_MS = 6.0
HIGH_WIND_MS = 12.0
RATIO_BOUNDS = (0.90, 1.10)
SHIFT_STRENGTH = 0.5
H2_2023_START = pd.Timestamp("2023-07-01 01:00:00")
H2_2024_START = pd.Timestamp("2024-07-01 01:00:00")
TEST_START = pd.Timestamp("2025-01-01 01:00:00")
TEST_END = pd.Timestamp("2026-01-01 00:00:00")
GROUP_SPECS: dict[str, dict[str, Any]] = {
    "kpx_group_1": {
        "manufacturer": "vestas",
        "turbines": tuple(range(1, 7)),
        "rated_kwh": 3600.0,
        "interval_capacity": 600.0,
        "power_bounds": (0.0, 630.0),
        "min_wind_per_hour": 29,
    },
    "kpx_group_2": {
        "manufacturer": "vestas",
        "turbines": tuple(range(7, 13)),
        "rated_kwh": 3600.0,
        "interval_capacity": 600.0,
        "power_bounds": (0.0, 630.0),
        "min_wind_per_hour": 29,
    },
    "kpx_group_3": {
        "manufacturer": "unison",
        "turbines": tuple(range(1, 6)),
        "rated_kwh": 4200.0,
        "interval_capacity": 700.0,
        "power_bounds": (0.0, 805.0),
        "min_wind_per_hour": 24,
    },
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir", type=Path, default=Path(r"data/local/open")
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/cache"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--recipe", type=Path, default=Path("configs/train_final.v3.locked.json")
    )
    parser.add_argument(
        "--preregister",
        type=Path,
        default=Path("configs/scada_physical_residual_preregister.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/postgate/scada_physical_residual_strict_v1_run2"),
    )
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


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    write_json_atomic(path, _json_ready(payload), overwrite=False)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        frame.to_parquet(temporary, engine="pyarrow", compression="zstd", index=True)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_joblib(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        joblib.dump(payload, temporary, compress=3)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        frame.to_csv(
            temporary,
            index=False,
            encoding="utf-8-sig",
            float_format="%.6f",
            lineterminator="\n",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(_json_ready(payload), stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _load_preregister(path: Path) -> dict[str, Any]:
    observed = sha256_file(path)
    if observed != PREREGISTER_SHA256:
        raise AssertionError(f"preregister SHA changed: {observed}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("experiment_id") != "scada_physical_residual_strict_forward_v1":
        raise AssertionError("unexpected experiment id")
    fixed = payload["fixed_regime_and_candidates"]
    if tuple(map(float, fixed["weights"])) != WEIGHTS:
        raise AssertionError("candidate weights changed")
    if int(fixed["candidate_count_per_group"]) != 2:
        raise AssertionError("candidate count changed")
    if tuple(payload["weather_to_scada_wind_mapper"]["features"]) != FEATURE_COLUMNS:
        raise AssertionError("weather mapper features changed")
    return payload


def _target_time(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(frame["kst_dtm"], errors="raise").dt.floor("h") + pd.Timedelta(hours=1)


def _clean_numeric(series: pd.Series, lower: float, upper: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    finite = np.isfinite(values.to_numpy(dtype=float, copy=False))
    valid = finite & values.between(lower, upper, inclusive="both").to_numpy(dtype=bool)
    return values.where(valid)


def aggregate_fleet_wind_hourly(
    frame: pd.DataFrame, group: str
) -> tuple[pd.Series, dict[str, Any]]:
    """Aggregate only causal SCADA wind into the label-aligned forecast hour."""

    spec = GROUP_SPECS[group]
    manufacturer = str(spec["manufacturer"])
    target_time = _target_time(frame)
    columns = [f"{manufacturer}_wtg{turbine:02d}_ws" for turbine in spec["turbines"]]
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"SCADA wind columns missing: {missing}")
    clean = pd.DataFrame(
        {column: _clean_numeric(frame[column], 0.0, 40.0) for column in columns}
    )
    sums = clean.sum(axis=1, min_count=1).groupby(target_time, sort=True).sum(min_count=1)
    counts = clean.notna().sum(axis=1).groupby(target_time, sort=True).sum()
    mean = (sums / counts).where(counts >= int(spec["min_wind_per_hour"]))
    mean.index = pd.DatetimeIndex(mean.index, name="forecast_kst_dtm")
    return mean.astype(float), {
        "group": group,
        "alignment": "floor(source_kst_dtm,1h)+1h",
        "minimum_valid_observations": int(spec["min_wind_per_hour"]),
        "hours": len(mean),
        "valid_hours": int(mean.notna().sum()),
        "start": mean.index.min(),
        "end": mean.index.max(),
        "sha256": _frame_sha256(mean.to_frame("fleet_ws")),
    }


def shift_hedged_application_features(
    train: pd.DataFrame, application: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the fixed half-strength, covariate-only annual median hedge."""

    missing = [column for column in FEATURE_COLUMNS if column not in train or column not in application]
    if missing:
        raise ValueError(f"weather mapper columns missing: {missing}")
    output = application.loc[:, FEATURE_COLUMNS].astype(float).copy()
    audit: dict[str, Any] = {}
    for column in FEATURE_COLUMNS:
        reference = float(np.nanmedian(train[column].to_numpy(dtype=float)))
        observed = float(np.nanmedian(application[column].to_numpy(dtype=float)))
        if not np.isfinite(reference) or reference <= 0.0 or not np.isfinite(observed):
            raise ValueError(f"invalid median for {column}")
        raw_ratio = observed / reference
        ratio = float(np.clip(raw_ratio, RATIO_BOUNDS[0], RATIO_BOUNDS[1]))
        divisor = float(ratio**SHIFT_STRENGTH)
        output[column] = output[column] / divisor
        audit[column] = {
            "reference_median": reference,
            "application_median": observed,
            "raw_ratio": raw_ratio,
            "clipped_ratio": ratio,
            "divisor": divisor,
        }
    return output, audit


def fit_wind_mapper(
    train_weather: pd.DataFrame, hourly_wind: pd.Series
) -> tuple[Pipeline, dict[str, Any]]:
    target = hourly_wind.reindex(train_weather.index)
    design = train_weather.loc[:, FEATURE_COLUMNS].astype(float)
    mask = target.notna() & np.isfinite(target.to_numpy(dtype=float))
    mask &= np.isfinite(design.to_numpy(dtype=float)).all(axis=1)
    if int(mask.sum()) < 1000:
        raise AssertionError("too few causal NWP/SCADA wind mapper rows")
    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "huber",
                HuberRegressor(epsilon=1.5, alpha=0.01, max_iter=500, tol=1e-5),
            ),
        ]
    )
    model.fit(design.loc[mask], target.loc[mask])
    fitted = np.asarray(model.predict(design.loc[mask]), dtype=float)
    actual = target.loc[mask].to_numpy(dtype=float)
    return model, {
        "training_hours": int(mask.sum()),
        "fit_start": design.index[mask].min(),
        "fit_end": design.index[mask].max(),
        "mae_ms": float(np.mean(np.abs(actual - fitted))),
        "correlation": float(np.corrcoef(actual, fitted)[0, 1]),
    }


def fit_turbine_curves(
    frame: pd.DataFrame, group: str
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Fit fixed q0.60 binned isotonic curves and turbine wind biases."""

    spec = GROUP_SPECS[group]
    manufacturer = str(spec["manufacturer"])
    turbines = tuple(spec["turbines"])
    wind_columns = [f"{manufacturer}_wtg{turbine:02d}_ws" for turbine in turbines]
    clean_wind = pd.DataFrame(
        {
            turbine: _clean_numeric(frame[column], 0.0, 40.0)
            for turbine, column in zip(turbines, wind_columns)
        }
    )
    fleet_mean = clean_wind.mean(axis=1, skipna=True)
    states: dict[int, dict[str, Any]] = {}
    audit: dict[str, Any] = {}
    lower, upper = map(float, spec["power_bounds"])
    anchors_x = np.asarray([0.0, 3.0, 12.0, 20.0, 24.99], dtype=float)
    anchors_y = np.asarray([0.0, 0.0, 1.0, 1.0, 1.0], dtype=float)
    anchors_w = np.full(5, 20.0, dtype=float)
    for turbine in turbines:
        power_column = f"{manufacturer}_wtg{turbine:02d}_power_kw10m"
        if power_column not in frame:
            raise ValueError(f"SCADA power column missing: {power_column}")
        power = _clean_numeric(frame[power_column], lower, upper)
        wind = clean_wind[turbine]
        mask = wind.notna() & power.notna()
        x = wind.loc[mask].to_numpy(dtype=float)
        y = np.clip(
            power.loc[mask].to_numpy(dtype=float) / float(spec["interval_capacity"]),
            0.0,
            1.02,
        )
        bins = np.floor(x / 0.25).astype(int)
        binned = (
            pd.DataFrame({"x": x, "y": y, "bin": bins})
            .groupby("bin", sort=True)
            .agg(
                x=("x", "median"),
                y=("y", lambda values: float(np.quantile(values, 0.60))),
                count=("y", "size"),
            )
        )
        binned = binned.loc[binned["count"] >= 30]
        if len(binned) < 12:
            raise AssertionError(f"{group} turbine {turbine}: too few stable wind bins")
        fit_x = np.concatenate([anchors_x, binned["x"].to_numpy(dtype=float)])
        fit_y = np.concatenate([anchors_y, binned["y"].to_numpy(dtype=float)])
        fit_weight = np.concatenate([anchors_w, binned["count"].to_numpy(dtype=float)])
        order = np.argsort(fit_x, kind="stable")
        curve = IsotonicRegression(
            increasing=True, y_min=0.0, y_max=1.02, out_of_bounds="clip"
        ).fit(fit_x[order], fit_y[order], sample_weight=fit_weight[order])
        bias_mask = wind.notna() & fleet_mean.notna()
        bias = float(np.nanmedian((wind.loc[bias_mask] - fleet_mean.loc[bias_mask]).to_numpy()))
        states[int(turbine)] = {"curve": curve, "wind_bias_ms": bias}
        audit[str(turbine)] = {
            "curve_samples": int(mask.sum()),
            "retained_bins": len(binned),
            "wind_bias_ms": bias,
            "observed_zero_power_share": float(np.mean(y <= 0.002)),
        }
    return states, audit


def predict_physical_fleet(
    predicted_fleet_wind: pd.Series,
    states: Mapping[int, Mapping[str, Any]],
    group: str,
) -> pd.Series:
    spec = GROUP_SPECS[group]
    values = np.zeros(len(predicted_fleet_wind), dtype=float)
    base = predicted_fleet_wind.to_numpy(dtype=float)
    for turbine in spec["turbines"]:
        state = states[int(turbine)]
        effective = base + float(state["wind_bias_ms"])
        clipped = np.clip(effective, 0.0, 24.99)
        cf = np.asarray(state["curve"].predict(clipped), dtype=float)
        cf[(effective < 3.0) | (effective >= 25.0)] = 0.0
        values += np.clip(cf, 0.0, 1.02) * float(spec["rated_kwh"])
    return pd.Series(
        np.clip(values, 0.0, 1.02 * CAPACITY_KWH[group]),
        index=predicted_fleet_wind.index,
        name=group,
    )


def fit_predict_physical(
    *,
    group: str,
    scada: pd.DataFrame,
    weather: pd.DataFrame,
    train_index: pd.DatetimeIndex,
    application_index: pd.DatetimeIndex,
) -> tuple[dict[str, Any], pd.Series, pd.Series, dict[str, Any]]:
    if train_index.max() >= application_index.min():
        raise AssertionError("physical fit/application order is not strict-forward")
    if not train_index.isin(weather.index).all() or not application_index.isin(weather.index).all():
        raise AssertionError("physical weather index is incomplete")
    source_target = _target_time(scada)
    fit_scada = scada.loc[source_target.isin(train_index)].copy()
    if fit_scada.empty or source_target.loc[fit_scada.index].max() > train_index.max():
        raise AssertionError("SCADA hard cutoff/order failed")
    hourly_wind, wind_audit = aggregate_fleet_wind_hourly(fit_scada, group)
    mapper, mapper_audit = fit_wind_mapper(weather.loc[train_index], hourly_wind)
    adjusted, shift_audit = shift_hedged_application_features(
        weather.loc[train_index], weather.loc[application_index]
    )
    predicted_wind = pd.Series(
        np.clip(np.asarray(mapper.predict(adjusted), dtype=float), 0.0, 40.0),
        index=application_index,
        name="predicted_fleet_scada_wind",
    )
    states, curve_audit = fit_turbine_curves(fit_scada, group)
    physical = predict_physical_fleet(predicted_wind, states, group)
    model = {"mapper": mapper, "turbine_states": states}
    audit = {
        "group": group,
        "train_start": train_index.min(),
        "train_end": train_index.max(),
        "application_start": application_index.min(),
        "application_end": application_index.max(),
        "validation_scada_used": False,
        "group_labels_used_in_fit": False,
        "weather_features": list(FEATURE_COLUMNS),
        "hourly_wind": wind_audit,
        "mapper": mapper_audit,
        "shift_hedge": shift_audit,
        "curves": curve_audit,
        "predicted_wind_sha256": _frame_sha256(predicted_wind.to_frame()),
        "physical_prediction_sha256": _frame_sha256(physical.to_frame()),
        "regime_counts": {
            "low": int((predicted_wind <= LOW_WIND_MS).sum()),
            "mid": int(((predicted_wind > LOW_WIND_MS) & (predicted_wind < HIGH_WIND_MS)).sum()),
            "high": int((predicted_wind >= HIGH_WIND_MS).sum()),
        },
    }
    return model, predicted_wind, physical, audit


def residual_blend(
    baseline: pd.Series,
    physical: pd.Series,
    predicted_wind: pd.Series,
    group: str,
    weight: float,
) -> pd.Series:
    if not baseline.index.equals(physical.index) or not baseline.index.equals(predicted_wind.index):
        raise AssertionError("residual blend index mismatch")
    gate = (
        (predicted_wind.to_numpy(dtype=float) <= LOW_WIND_MS)
        | (predicted_wind.to_numpy(dtype=float) >= HIGH_WIND_MS)
    ).astype(float)
    values = baseline.to_numpy(dtype=float) + float(weight) * gate * (
        physical.to_numpy(dtype=float) - baseline.to_numpy(dtype=float)
    )
    return pd.Series(
        np.clip(values, 0.0, 1.02 * CAPACITY_KWH[group]),
        index=baseline.index,
        name=group,
    )


def _all_slices(index: pd.DatetimeIndex) -> dict[str, pd.DatetimeIndex]:
    start = index.min()
    end = index.max()
    year = int(start.year)
    boundaries = {
        "Q1": (pd.Timestamp(year, 1, 1, 1), pd.Timestamp(year, 4, 1, 0)),
        "Q2": (pd.Timestamp(year, 4, 1, 1), pd.Timestamp(year, 7, 1, 0)),
        "Q3": (pd.Timestamp(year, 7, 1, 1), pd.Timestamp(year, 10, 1, 0)),
        "Q4": (pd.Timestamp(year, 10, 1, 1), pd.Timestamp(year + 1, 1, 1, 0)),
    }
    result = {"full": index}
    h2 = pd.Timestamp(year, 7, 1, 1)
    h1 = index[(index >= start) & (index < h2)]
    h2_index = index[(index >= h2) & (index <= end)]
    if len(h1):
        result["H1"] = h1
    if len(h2_index):
        result["H2"] = h2_index
    for name, (left, right) in boundaries.items():
        selected = index[(index >= left) & (index <= right)]
        if len(selected):
            result[name] = selected
    return result


def _metric(actual: pd.Series, prediction: pd.Series, group: str) -> dict[str, Any]:
    metric = group_metrics(actual, prediction, CAPACITY_KWH[group], group_name=group)
    return {
        "score": float(0.5 * (metric.one_minus_nmae + metric.ficr)),
        "one_minus_nmae": float(metric.one_minus_nmae),
        "ficr": float(metric.ficr),
        "n_evaluated": int(metric.n_evaluated),
    }


def compare_slices(
    actual: pd.Series, baseline: pd.Series, candidate: pd.Series, group: str
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, index in _all_slices(baseline.index).items():
        before = _metric(actual.loc[index], baseline.loc[index], group)
        after = _metric(actual.loc[index], candidate.loc[index], group)
        output[name] = {
            "baseline": before,
            "candidate": after,
            "delta": float(after["score"] - before["score"]),
            "one_minus_nmae_delta": float(
                after["one_minus_nmae"] - before["one_minus_nmae"]
            ),
            "ficr_delta": float(after["ficr"] - before["ficr"]),
        }
    return output


def stage1_candidate_passes(group: str, comparisons: Mapping[str, Any]) -> bool:
    core = ("full", "Q3", "Q4") if group == "kpx_group_3" else ("full", "H1", "H2")
    quarters = ("Q3", "Q4") if group == "kpx_group_3" else ("Q1", "Q2", "Q3", "Q4")
    required_positive = 2 if group == "kpx_group_3" else 3
    return bool(
        all(float(comparisons[name]["delta"]) > 0.0 for name in core)
        and all(float(comparisons[name]["delta"]) >= -0.001 for name in quarters)
        and sum(float(comparisons[name]["delta"]) > 0.0 for name in quarters)
        >= required_positive
        and float(comparisons["full"]["one_minus_nmae_delta"]) >= -0.0005
    )


def select_stage1_candidate(
    group: str, comparisons: Mapping[str, Mapping[str, Any]]
) -> tuple[list[str], str | None]:
    passing = [name for name, values in comparisons.items() if stage1_candidate_passes(group, values)]
    if not passing:
        return [], None
    core = ("full", "Q3", "Q4") if group == "kpx_group_3" else ("full", "H1", "H2")

    def key(name: str) -> tuple[float, float, float]:
        weight = int(name.removeprefix("extreme_w")) / 100.0
        values = comparisons[name]
        return (
            min(float(values[slice_name]["delta"]) for slice_name in core),
            float(values["full"]["delta"]),
            -weight,
        )

    return sorted(passing), max(passing, key=key)


def stage2_candidate_passes(comparisons: Mapping[str, Any]) -> bool:
    quarters = ("Q1", "Q2", "Q3", "Q4")
    return bool(
        all(float(comparisons[name]["delta"]) > 0.0 for name in ("full", "H1", "H2"))
        and all(float(comparisons[name]["delta"]) >= -0.001 for name in quarters)
        and sum(float(comparisons[name]["delta"]) > 0.0 for name in quarters) >= 3
        and float(comparisons["full"]["one_minus_nmae_delta"]) >= -0.0005
    )


def _candidate_name(weight: float) -> str:
    return f"extreme_w{int(round(100 * weight)):02d}"


def _manifest(
    out_dir: Path,
    *,
    stage: str,
    inputs: Mapping[str, Any],
    outputs: Sequence[Path],
) -> None:
    payload = {
        "schema_version": 1,
        "artifact_type": "scada_physical_residual_strict_forward",
        "stage": stage,
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "runtime": {"packages": package_versions()},
        "git": git_state(PROJECT_DIR),
        "inputs": inputs,
        "outputs": [describe_file(path) for path in outputs if path.exists()],
    }
    name = "manifest.json" if stage.startswith("rejected") or stage == "final" else f"{stage}_manifest.json"
    _atomic_json(out_dir / name, payload)


def _stage1(args: argparse.Namespace, prereg: Mapping[str, Any]) -> dict[str, Any]:
    out_dir = args.out_dir
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"stage1 requires empty output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    prereg_copy = out_dir / "preregister.json"
    shutil.copyfile(args.preregister, prereg_copy)
    if sha256_file(prereg_copy) != PREREGISTER_SHA256:
        raise AssertionError("preregister copy changed")

    labels = _read_labels(
        args.raw_dir / "train" / "train_labels.csv",
        nrows=EXPECTED_ROWS_PRE2024,
        expected_end=YEAR_2023_END,
        prefix_bytes=OFFICIAL_STAGE1_PREFIX_BYTES["labels"],
        expected_prefix_sha256=OFFICIAL_LABEL_PREFIX_SHA256,
    )
    features, raw_nwp_contract = _read_stage1_raw_features(args.raw_dir, labels)
    valid_2023 = labels.index[(labels.index >= YEAR_2023_START) & (labels.index <= YEAR_2023_END)]
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    baseline_2023, baseline_audit, baseline_inputs = _stage1_baseline(
        args.artifact_root, recipe, valid_2023
    )

    prefixes = {
        "vestas": _discover_csv_prefix(
            args.raw_dir / "train" / "scada_vestas_train.csv",
            pd.Timestamp("2023-01-01 00:00:00"),
        ),
        "unison": _discover_csv_prefix(
            args.raw_dir / "train" / "scada_unison_train.csv",
            pd.Timestamp("2023-07-01 00:00:00"),
        ),
    }
    scada: dict[str, pd.DataFrame] = {}
    scada_contract: dict[str, Any] = {}
    for manufacturer, prefix in prefixes.items():
        scada[manufacturer], scada_contract[manufacturer] = _read_scada_prefix(prefix)

    wide = pd.DataFrame(index=labels.index)
    comparisons: dict[str, Any] = {}
    selection: dict[str, Any] = {}
    training: dict[str, Any] = {}
    outputs: list[Path] = [prereg_copy]
    for group in TARGET_COLS:
        if group == "kpx_group_3":
            train_index = labels.index[(labels.index >= YEAR_2023_START) & (labels.index < H2_2023_START)]
            valid_index = labels.index[(labels.index >= H2_2023_START) & (labels.index <= YEAR_2023_END)]
        else:
            train_index = labels.index[(labels.index >= YEAR_2022_START) & (labels.index < YEAR_2023_START)]
            valid_index = valid_2023
        manufacturer = str(GROUP_SPECS[group]["manufacturer"])
        print(f"stage1 physical fit {group}", flush=True)
        model, wind, physical, audit = fit_predict_physical(
            group=group,
            scada=scada[manufacturer],
            weather=features[group],
            train_index=train_index,
            application_index=valid_index,
        )
        model_path = out_dir / "models" / f"stage1__{group}.joblib"
        _atomic_joblib({"model": model, "audit": audit}, model_path)
        outputs.append(model_path)
        training[group] = {**audit, "model": describe_file(model_path)}
        wide.loc[valid_index, f"{group}__predicted_wind"] = wind
        wide.loc[valid_index, f"{group}__physical"] = physical
        group_comparisons: dict[str, Any] = {}
        baseline = baseline_2023.loc[valid_index, group]
        if baseline.isna().any():
            raise AssertionError(f"{group} stage1 baseline incomplete")
        for weight in WEIGHTS:
            name = _candidate_name(weight)
            candidate = residual_blend(baseline, physical, wind, group, weight)
            wide.loc[valid_index, f"{group}__{name}"] = candidate
            group_comparisons[name] = compare_slices(
                labels.loc[valid_index, group], baseline, candidate, group
            )
        passing, selected = select_stage1_candidate(group, group_comparisons)
        comparisons[group] = group_comparisons
        selection[group] = {"passing_candidates": passing, "selected": selected}

    prediction_path = out_dir / "oof" / "stage1_candidates.parquet"
    _atomic_parquet(wide, prediction_path)
    outputs.append(prediction_path)
    locked_groups = [group for group in TARGET_COLS if selection[group]["selected"] is not None]
    results = {
        "schema_version": 1,
        "stage": "stage1_pre2024_selection",
        "preregister_sha256": PREREGISTER_SHA256,
        "public_scores_read": False,
        "forbidden_2024_data_read_occurred": False,
        "test_weather_read": False,
        "stage1_cache_files_read": False,
        "raw_nwp_contract": raw_nwp_contract,
        "raw_scada_contract": scada_contract,
        "baseline_audit": baseline_audit,
        "baseline_inputs": [describe_file(path) for path in baseline_inputs],
        "training": training,
        "comparisons": comparisons,
        "comparisons_sha256": _canonical_sha256(comparisons),
        "selection": selection,
        "locked_groups": locked_groups,
        "candidate_count_per_group": 2,
        "prediction": describe_file(prediction_path),
    }
    results_path = out_dir / "stage1_results.json"
    _atomic_json(results_path, results)
    outputs.append(results_path)
    lock = {
        "schema_version": 1,
        "stage": "stage1_pre2024_lock",
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "test_sha256": sha256_file(PROJECT_DIR / "tests" / "test_scada_physical_residual.py"),
        "stage1_results_sha256": sha256_file(results_path),
        "comparisons_sha256": results["comparisons_sha256"],
        "locked_groups": locked_groups,
        "selected": {group: selection[group]["selected"] for group in locked_groups},
        "forbidden_2024_data_read_occurred": False,
    }
    lock_path = out_dir / "stage1_lock.json"
    _write_exclusive_json(lock_path, lock)
    outputs.append(lock_path)
    _manifest(
        out_dir,
        stage="stage1" if locked_groups else "rejected_stage1",
        inputs={"preregister": describe_file(args.preregister), "raw_scada": scada_contract},
        outputs=outputs,
    )
    print(json.dumps({"locked_groups": locked_groups, "selected": lock["selected"]}), flush=True)
    return results


def _verify_stage1(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    results_path = args.out_dir / "stage1_results.json"
    lock_path = args.out_dir / "stage1_lock.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["preregister_sha256"] != PREREGISTER_SHA256:
        raise AssertionError("stage1 preregister hash changed")
    if sha256_file(results_path) != lock["stage1_results_sha256"]:
        raise AssertionError("stage1 results changed")
    if sha256_file(Path(__file__).resolve()) != lock["runner_sha256"]:
        raise AssertionError("runner changed after Stage 1")
    test_path = PROJECT_DIR / "tests" / "test_scada_physical_residual.py"
    if sha256_file(test_path) != lock["test_sha256"]:
        raise AssertionError("test changed after Stage 1")
    recomputed: dict[str, str] = {}
    for group, values in results["comparisons"].items():
        _, selected = select_stage1_candidate(group, values)
        if selected is not None:
            recomputed[group] = selected
    if recomputed != lock["selected"] or sorted(recomputed) != sorted(lock["locked_groups"]):
        raise AssertionError("Stage-1 lock differs from recomputed selection")
    return results, lock


def _stage2(args: argparse.Namespace) -> dict[str, Any]:
    stage1, lock = _verify_stage1(args)
    locked_groups = list(lock["locked_groups"])
    if not locked_groups:
        results = {
            "schema_version": 1,
            "stage": "stage2_not_run",
            "reason": "no group passed Stage 1",
            "locked_groups": [],
            "2024_read": False,
            "promoted_groups": [],
        }
        results_path = args.out_dir / "stage2_results.json"
        _atomic_json(results_path, results)
        _write_exclusive_json(
            args.out_dir / "stage2_promotion_lock.json",
            {
                "schema_version": 1,
                "stage1_lock_sha256": sha256_file(args.out_dir / "stage1_lock.json"),
                "stage2_results_sha256": sha256_file(results_path),
                "promoted_groups": [],
                "selected": {},
            },
        )
        return results

    labels = _read_full_labels(args.raw_dir / "train" / "train_labels.csv")
    features = _read_cached_features(args.cache_dir, locked_groups, labels.index)
    valid_index = labels.index[(labels.index >= YEAR_2024_START) & (labels.index <= YEAR_2024_END)]
    baseline = _read_prediction(
        args.artifact_root / "oof" / "gate2024_locked_v3_cf_fix.parquet",
        valid_index,
        TARGET_COLS,
    )
    manufacturers = sorted({str(GROUP_SPECS[group]["manufacturer"]) for group in locked_groups})
    scada: dict[str, pd.DataFrame] = {}
    contracts: dict[str, Any] = {}
    for manufacturer in manufacturers:
        prefix = _discover_csv_prefix(
            args.raw_dir / "train" / f"scada_{manufacturer}_train.csv",
            pd.Timestamp("2024-01-01 00:00:00"),
        )
        scada[manufacturer], contracts[manufacturer] = _read_scada_prefix(prefix)

    candidate = baseline.copy()
    comparisons: dict[str, Any] = {}
    training: dict[str, Any] = {}
    outputs: list[Path] = []
    for group in locked_groups:
        train_start = YEAR_2023_START if group == "kpx_group_3" else YEAR_2022_START
        train_index = labels.index[(labels.index >= train_start) & (labels.index <= YEAR_2023_END)]
        manufacturer = str(GROUP_SPECS[group]["manufacturer"])
        model, wind, physical, audit = fit_predict_physical(
            group=group,
            scada=scada[manufacturer],
            weather=features[group],
            train_index=train_index,
            application_index=valid_index,
        )
        selected = str(lock["selected"][group])
        weight = int(selected.removeprefix("extreme_w")) / 100.0
        candidate[group] = residual_blend(baseline[group], physical, wind, group, weight)
        comparisons[group] = compare_slices(
            labels.loc[valid_index, group], baseline[group], candidate[group], group
        )
        model_path = args.out_dir / "models" / f"stage2__{group}.joblib"
        _atomic_joblib({"model": model, "audit": audit}, model_path)
        outputs.append(model_path)
        training[group] = {**audit, "selected": selected, "model": describe_file(model_path)}
    prediction_path = args.out_dir / "oof" / "stage2_locked_candidates.parquet"
    _atomic_parquet(candidate, prediction_path)
    outputs.append(prediction_path)
    promoted = [group for group in locked_groups if stage2_candidate_passes(comparisons[group])]
    results = {
        "schema_version": 1,
        "stage": "stage2_2024_fixed_transfer",
        "stage1_lock_sha256": sha256_file(args.out_dir / "stage1_lock.json"),
        "locked_groups": locked_groups,
        "selected": lock["selected"],
        "comparisons": comparisons,
        "comparisons_sha256": _canonical_sha256(comparisons),
        "training": training,
        "raw_scada_contract": contracts,
        "no_reselection": True,
        "promoted_groups": promoted,
        "prediction": describe_file(prediction_path),
        "submission_created": False,
    }
    results_path = args.out_dir / "stage2_results.json"
    _atomic_json(results_path, results)
    outputs.append(results_path)
    promotion_path = args.out_dir / "stage2_promotion_lock.json"
    _write_exclusive_json(
        promotion_path,
        {
            "schema_version": 1,
            "stage": "stage2_promotion_lock",
            "created_utc": utc_now(),
            "stage1_lock_sha256": sha256_file(args.out_dir / "stage1_lock.json"),
            "stage2_results_sha256": sha256_file(results_path),
            "comparisons_sha256": results["comparisons_sha256"],
            "promoted_groups": promoted,
            "selected": {group: lock["selected"][group] for group in promoted},
        },
    )
    outputs.append(promotion_path)
    _manifest(
        args.out_dir,
        stage="stage2" if promoted else "rejected_stage2",
        inputs={"stage1_lock": describe_file(args.out_dir / "stage1_lock.json"), "raw_scada": contracts},
        outputs=outputs,
    )
    print(json.dumps({"promoted_groups": promoted}), flush=True)
    return results


def _verify_stage2(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    results_path = args.out_dir / "stage2_results.json"
    lock_path = args.out_dir / "stage2_promotion_lock.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if sha256_file(results_path) != lock["stage2_results_sha256"]:
        raise AssertionError("Stage-2 results changed")
    recomputed = [
        group
        for group, comparisons in results.get("comparisons", {}).items()
        if stage2_candidate_passes(comparisons)
    ]
    if sorted(recomputed) != sorted(lock["promoted_groups"]):
        raise AssertionError("Stage-2 promotion lock differs from recomputation")
    return results, lock


def _stage_final(args: argparse.Namespace) -> dict[str, Any]:
    _verify_stage1(args)
    _, promotion = _verify_stage2(args)
    promoted = list(promotion["promoted_groups"])
    if not promoted:
        return {"stage": "final_not_run", "reason": "no Stage-2 promotion", "submission_created": False}
    labels = _read_full_labels(args.raw_dir / "train" / "train_labels.csv")
    train_features = _read_cached_features(args.cache_dir, promoted, labels.index)
    sample = pd.read_csv(args.raw_dir / "sample_submission.csv", encoding="utf-8-sig")
    if tuple(sample.columns) != ("ID", "forecast_kst_dtm", *TARGET_COLS):
        raise AssertionError("sample submission schema changed")
    test_index = pd.DatetimeIndex(
        pd.to_datetime(sample["forecast_kst_dtm"], errors="raise"), name="forecast_kst_dtm"
    )
    if len(test_index) != 8760 or test_index.min() != TEST_START or test_index.max() != TEST_END:
        raise AssertionError("test index changed")
    test_features: dict[str, pd.DataFrame] = {}
    for group in promoted:
        frame = pd.read_parquet(args.cache_dir / f"{group}_weather_test.parquet", engine="pyarrow")
        frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
        if not frame.index.equals(test_index):
            raise AssertionError(f"{group} test weather index changed")
        test_features[group] = frame.astype(np.float32, copy=False)
    manufacturers = sorted({str(GROUP_SPECS[group]["manufacturer"]) for group in promoted})
    scada: dict[str, pd.DataFrame] = {}
    contracts: dict[str, Any] = {}
    for manufacturer in manufacturers:
        prefix = _discover_csv_prefix(
            args.raw_dir / "train" / f"scada_{manufacturer}_train.csv",
            pd.Timestamp("2025-01-01 00:00:00"),
        )
        scada[manufacturer], contracts[manufacturer] = _read_scada_prefix(prefix)
    baseline_path = args.artifact_root / "final_cf_fix" / "predictions" / "corrected_v3_test.parquet"
    baseline = _read_prediction(baseline_path, test_index, TARGET_COLS)
    candidate = baseline.copy()
    models: dict[str, Any] = {}
    outputs: list[Path] = []
    for group in promoted:
        train_start = YEAR_2023_START if group == "kpx_group_3" else YEAR_2022_START
        train_index = labels.index[(labels.index >= train_start) & (labels.index <= YEAR_2024_END)]
        all_weather = pd.concat([train_features[group], test_features[group]], axis=0)
        manufacturer = str(GROUP_SPECS[group]["manufacturer"])
        model, wind, physical, audit = fit_predict_physical(
            group=group,
            scada=scada[manufacturer],
            weather=all_weather,
            train_index=train_index,
            application_index=test_index,
        )
        selected = str(promotion["selected"][group])
        weight = int(selected.removeprefix("extreme_w")) / 100.0
        candidate[group] = residual_blend(baseline[group], physical, wind, group, weight)
        model_path = args.out_dir / "models" / f"final__{group}.joblib"
        _atomic_joblib({"model": model, "audit": audit}, model_path)
        outputs.append(model_path)
        models[group] = {**audit, "selected": selected, "model": describe_file(model_path)}
    prediction_path = args.out_dir / "predictions" / "scada_physical_residual_2025.parquet"
    _atomic_parquet(candidate, prediction_path)
    outputs.append(prediction_path)
    submission = sample.copy()
    for group in TARGET_COLS:
        submission[group] = candidate[group].to_numpy(dtype=float)
    csv_path = args.out_dir / "scada_physical_residual_2025.csv"
    _atomic_csv(submission, csv_path)
    outputs.append(csv_path)
    results = {
        "schema_version": 1,
        "stage": "final_2025",
        "promoted_groups": promoted,
        "selected": promotion["selected"],
        "models": models,
        "raw_scada_contract": contracts,
        "failed_groups_bit_identical_to_corrected_v3": [group for group in TARGET_COLS if group not in promoted],
        "prediction": describe_file(prediction_path),
        "submission": describe_file(csv_path),
        "submission_rows": len(submission),
        "leaderboard_score_claim": False,
    }
    results_path = args.out_dir / "results.json"
    _atomic_json(results_path, results)
    outputs.append(results_path)
    _manifest(
        args.out_dir,
        stage="final",
        inputs={"stage2_lock": describe_file(args.out_dir / "stage2_promotion_lock.json"), "baseline": describe_file(baseline_path)},
        outputs=outputs,
    )
    print(json.dumps({"submission": str(csv_path), "sha256": sha256_file(csv_path)}), flush=True)
    return results


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for name in ("raw_dir", "cache_dir", "artifact_root", "recipe", "preregister", "out_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    prereg = _load_preregister(args.preregister)
    if args.stage in {"stage1", "all"}:
        _stage1(args, prereg)
    if args.stage in {"stage2", "all"}:
        _stage2(args)
    if args.stage in {"final", "all"}:
        _stage_final(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

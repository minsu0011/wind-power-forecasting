"""Last-48-hour, official-data-only vertical-profile residual experiment.

The experiment deliberately has a small hypothesis space:

* one fixed rotation-invariant feature map built only from the existing official
  GFS cache (100 m, 850 hPa, and planetary-boundary-layer winds plus stability);
* one standardized Ridge residual model with fixed regularisation;
* three predeclared residual blend weights: 0.05, 0.10, and 0.20; and
* a causal 2023 -> 2024 transfer check plus a separate 2024-H1 -> 2024-H2
  stability check.  Group 3 participates only in the latter because no causal
  2023 baseline prediction exists for it.

No Public-leaderboard value is read.  A 2025 submission is emitted only when a
predeclared weight clears both validation gates.  Otherwise the script writes a
diagnostic JSON report but no candidate CSV.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metric import CAPACITY_KWH, TARGET_COLS, score_details


OPEN_ROOT = Path(r"data/local/open")
OUT_DIR = ROOT / "artifacts" / "final_submission_sprint_20260812"

LABEL_PATH = OPEN_ROOT / "train" / "train_labels.csv"
SAMPLE_PATH = OPEN_ROOT / "sample_submission.csv"
BASE_2023_PATH = ROOT / "artifacts" / "oof" / "dev2023_locked_v3.parquet"
BASE_2024_PATH = (
    ROOT
    / "artifacts"
    / "oof"
    / "gate2024_recent_v4_cf_fix_calibration_fit.parquet"
)
BASE_2025_PATH = (
    ROOT
    / "artifacts"
    / "final_cf_fix"
    / "predictions"
    / "corrected_recent_v4_test.parquet"
)
CACHE_TRAIN_PATTERN = ROOT / "artifacts" / "cache" / "{group}_weather_train.parquet"
CACHE_TEST_PATTERN = ROOT / "artifacts" / "cache" / "{group}_weather_test.parquet"

BASELINE_SCALE = 0.97
RIDGE_ALPHA = 100.0
RESIDUAL_TARGET_CLIP = 0.40
RESIDUAL_PREDICTION_CLIP = 0.25
WEIGHTS = (0.05, 0.10, 0.20)

# These gates are defined independently of any Public score.  Full-period
# components must not regress.  Fine slices have small tolerances because FICR
# is a discontinuous threshold metric and a month is a relatively small sample.
GATE = {
    "transfer_mixed_full_total_min_exclusive": 0.0,
    "transfer_mixed_full_component_min": 0.0,
    "transfer_each_group_full_total_min_exclusive": 0.0,
    "transfer_worst_group_half_total_min": -0.0010,
    "transfer_worst_group_quarter_total_min": -0.0030,
    "transfer_worst_group_month_total_min": -0.0150,
    "transfer_positive_group_month_fraction_min": 0.50,
    "h1h2_mixed_h2_total_min_exclusive": 0.0,
    "h1h2_mixed_h2_component_min": 0.0,
    "h1h2_each_group_h2_total_min_exclusive": 0.0,
    "h1h2_worst_group_quarter_total_min": -0.0020,
    "h1h2_worst_group_month_total_min": -0.0150,
    "h1h2_positive_group_month_fraction_min": 0.50,
    "minimum_residual_correlation_exclusive": 0.0,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def operating_clock(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Map 01:00..next-day-00:00 forecasts to their operating calendar day."""

    return index - pd.Timedelta(hours=1)


def period_mask(index: pd.DatetimeIndex, period: str) -> np.ndarray:
    clock = operating_clock(index)
    if period == "FULL":
        return np.ones(len(index), dtype=bool)
    if period == "H1":
        return np.asarray(clock.month <= 6)
    if period == "H2":
        return np.asarray(clock.month >= 7)
    if period.startswith("Q"):
        quarter = int(period[1:])
        return np.asarray(clock.quarter == quarter)
    if period.startswith("M"):
        month = int(period[1:])
        return np.asarray(clock.month == month)
    raise KeyError(period)


def _col(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name not in frame.columns:
        raise KeyError(f"required official-cache column missing: {name}")
    return frame[name].to_numpy(dtype=np.float64, copy=False)


def vertical_features(weather: pd.DataFrame, baseline_cf: np.ndarray) -> pd.DataFrame:
    """Create the one predeclared rotation-invariant vertical feature map."""

    prefix = "gfs__idw__"
    u100 = _col(weather, prefix + "heightAboveGround_100_100u")
    v100 = _col(weather, prefix + "heightAboveGround_100_100v")
    u850 = _col(weather, prefix + "isobaricInhPa_850_u")
    v850 = _col(weather, prefix + "isobaricInhPa_850_v")
    upbl = _col(weather, prefix + "planetaryBoundaryLayer_0_u")
    vpbl = _col(weather, prefix + "planetaryBoundaryLayer_0_v")

    t2 = _col(weather, prefix + "heightAboveGround_2_2t")
    surface_pressure = _col(weather, prefix + "surface_0_sp")
    t850 = _col(weather, prefix + "isobaricInhPa_850_t")
    rh850 = _col(weather, prefix + "isobaricInhPa_850_r")
    pbl_vrate = _col(weather, prefix + "planetaryBoundaryLayer_0_VRATE")

    ws100 = np.hypot(u100, v100)
    ws850 = np.hypot(u850, v850)
    wspbl = np.hypot(upbl, vpbl)

    def relative_geometry(
        ua: np.ndarray,
        va: np.ndarray,
        ub: np.ndarray,
        vb: np.ndarray,
        wsa: np.ndarray,
        wsb: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        denominator = np.maximum(wsa * wsb, 0.25)
        cosine = np.clip((ua * ub + va * vb) / denominator, -1.0, 1.0)
        sine = np.clip((ua * vb - va * ub) / denominator, -1.0, 1.0)
        shear = np.hypot(ua - ub, va - vb)
        return cosine, sine, shear

    cos_100_850, sin_100_850, shear_100_850 = relative_geometry(
        u100, v100, u850, v850, ws100, ws850
    )
    cos_100_pbl, sin_100_pbl, shear_100_pbl = relative_geometry(
        u100, v100, upbl, vpbl, ws100, wspbl
    )
    cos_850_pbl, sin_850_pbl, shear_850_pbl = relative_geometry(
        u850, v850, upbl, vpbl, ws850, wspbl
    )

    ratio_850_100 = np.clip(ws850 / np.maximum(ws100, 0.5), 0.0, 4.0)
    ratio_pbl_100 = np.clip(wspbl / np.maximum(ws100, 0.5), 0.0, 4.0)
    relative_shear_100_850 = np.clip(
        shear_100_850 / np.maximum(ws100 + ws850, 0.5), 0.0, 2.0
    )
    relative_shear_100_pbl = np.clip(
        shear_100_pbl / np.maximum(ws100 + wspbl, 0.5), 0.0, 2.0
    )

    # Potential-temperature difference is a compact static-stability proxy.
    # Surface pressure is clipped only to prevent corrupt cache values from
    # producing unbounded powers; normal GFS values are unaffected.
    theta_surface = t2 * (100000.0 / np.clip(surface_pressure, 70000.0, 110000.0)) ** 0.286
    theta_850 = t850 * (1000.0 / 850.0) ** 0.286
    stability = np.clip(theta_850 - theta_surface, -40.0, 40.0)

    values = {
        "baseline_cf": np.asarray(baseline_cf, dtype=np.float64),
        "ws100": ws100,
        "ws850": ws850,
        "ws_pbl": wspbl,
        "ratio_850_100": ratio_850_100,
        "ratio_pbl_100": ratio_pbl_100,
        "shear_100_850": shear_100_850,
        "shear_100_pbl": shear_100_pbl,
        "shear_850_pbl": shear_850_pbl,
        "relative_shear_100_850": relative_shear_100_850,
        "relative_shear_100_pbl": relative_shear_100_pbl,
        "relative_cos_100_850": cos_100_850,
        "relative_sin_100_850": sin_100_850,
        "relative_cos_100_pbl": cos_100_pbl,
        "relative_sin_100_pbl": sin_100_pbl,
        "relative_cos_850_pbl": cos_850_pbl,
        "relative_sin_850_pbl": sin_850_pbl,
        "t850": t850,
        "rh850": rh850,
        "pbl_vrate": pbl_vrate,
        "stability_theta_850_minus_surface": stability,
        "stability_x_ws100": stability * ws100,
        "stability_x_shear_100_850": stability * shear_100_850,
        "stability_x_ratio_850_100": stability * ratio_850_100,
        "baseline_x_ratio_850_100": baseline_cf * ratio_850_100,
        "baseline_x_relative_shear_100_850": baseline_cf * relative_shear_100_850,
        "baseline_x_stability": baseline_cf * stability,
    }
    result = pd.DataFrame(values, index=weather.index, dtype=np.float64)
    if len(result.columns) != 27:
        raise AssertionError("vertical feature dimensionality changed")
    return result


def fit_residual(
    features: pd.DataFrame,
    actual_kwh: pd.Series,
    baseline_cf: np.ndarray,
    capacity: float,
    fit_mask: np.ndarray,
) -> tuple[Pipeline, dict[str, Any]]:
    actual = actual_kwh.to_numpy(dtype=np.float64, copy=False)
    eligible = (
        np.asarray(fit_mask, dtype=bool)
        & np.isfinite(actual)
        & (actual >= 0.10 * capacity)
        & np.isfinite(baseline_cf)
    )
    target_raw = actual / capacity - baseline_cf
    target = np.clip(target_raw, -RESIDUAL_TARGET_CLIP, RESIDUAL_TARGET_CLIP)
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("standardizer", StandardScaler()),
            ("ridge", Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)),
        ]
    )
    model.fit(features.iloc[np.flatnonzero(eligible)], target[eligible])
    ridge = model.named_steps["ridge"]
    return model, {
        "fit_rows": int(eligible.sum()),
        "raw_target_mean": float(np.mean(target_raw[eligible])),
        "raw_target_std": float(np.std(target_raw[eligible])),
        "target_clipped_rows": int(
            np.sum(np.abs(target_raw[eligible]) > RESIDUAL_TARGET_CLIP)
        ),
        "ridge_intercept": float(ridge.intercept_),
        "ridge_coefficients": {
            name: float(value) for name, value in zip(features.columns, ridge.coef_)
        },
    }


def predict_residual(model: Pipeline, features: pd.DataFrame) -> tuple[np.ndarray, int]:
    raw = np.asarray(model.predict(features), dtype=np.float64)
    clipped = np.clip(raw, -RESIDUAL_PREDICTION_CLIP, RESIDUAL_PREDICTION_CLIP)
    return clipped, int(np.sum(raw != clipped))


def metrics_dict(
    actual: pd.DataFrame,
    prediction: pd.DataFrame,
    groups: Sequence[str],
    mask: np.ndarray,
) -> dict[str, float]:
    record = score_details(
        actual.loc[mask, list(groups)],
        prediction.loc[mask, list(groups)],
        target_cols=groups,
    )
    return {
        "total_score": float(record.total_score),
        "one_minus_nmae": float(record.one_minus_nmae),
        "ficr": float(record.ficr),
    }


def metric_delta(
    actual: pd.DataFrame,
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    groups: Sequence[str],
    mask: np.ndarray,
) -> dict[str, Any]:
    before = metrics_dict(actual, baseline, groups, mask)
    after = metrics_dict(actual, candidate, groups, mask)
    return {
        "baseline": before,
        "candidate": after,
        "delta": {key: float(after[key] - before[key]) for key in before},
    }


def correlation(
    actual_kwh: pd.Series,
    baseline_cf: np.ndarray,
    predicted_residual: np.ndarray,
    capacity: float,
    mask: np.ndarray,
) -> float | None:
    actual = actual_kwh.to_numpy(dtype=np.float64, copy=False)
    eligible = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(actual)
        & (actual >= 0.10 * capacity)
        & np.isfinite(predicted_residual)
    )
    realized = actual[eligible] / capacity - baseline_cf[eligible]
    predicted = predicted_residual[eligible]
    if len(realized) < 3 or np.std(realized) == 0.0 or np.std(predicted) == 0.0:
        return None
    return float(np.corrcoef(realized, predicted)[0, 1])


def evaluate_stage(
    actual: pd.DataFrame,
    baseline: pd.DataFrame,
    residuals: Mapping[str, np.ndarray],
    groups: Sequence[str],
    weight: float,
    periods: Sequence[str],
    primary_period: str,
) -> dict[str, Any]:
    candidate = baseline.copy()
    for group in groups:
        capacity = CAPACITY_KWH[group]
        candidate[group] = np.clip(
            baseline[group].to_numpy(dtype=np.float64)
            + weight * capacity * residuals[group],
            0.0,
            1.02 * capacity,
        )

    by_group: dict[str, dict[str, Any]] = {}
    for group in groups:
        by_group[group] = {}
        for period in periods:
            mask = period_mask(actual.index, period)
            by_group[group][period] = metric_delta(
                actual, baseline, candidate, (group,), mask
            )

    mixed: dict[str, Any] = {}
    for period in periods:
        mixed[period] = metric_delta(
            actual, baseline, candidate, groups, period_mask(actual.index, period)
        )

    half_values = [
        by_group[group][period]["delta"]["total_score"]
        for group in groups
        for period in ("H1", "H2")
        if period in by_group[group]
    ]
    quarter_values = [
        by_group[group][period]["delta"]["total_score"]
        for group in groups
        for period in ("Q1", "Q2", "Q3", "Q4")
        if period in by_group[group]
    ]
    month_values = [
        by_group[group][period]["delta"]["total_score"]
        for group in groups
        for period in tuple(f"M{month:02d}" for month in range(1, 13))
        if period in by_group[group]
    ]
    return {
        "weight": weight,
        "primary_period": primary_period,
        "by_group": by_group,
        "mixed": mixed,
        "worst": {
            "group_half_total_delta": float(min(half_values)) if half_values else None,
            "group_quarter_total_delta": (
                float(min(quarter_values)) if quarter_values else None
            ),
            "group_month_total_delta": float(min(month_values)) if month_values else None,
            "positive_group_month_fraction": (
                float(np.mean(np.asarray(month_values) > 0.0)) if month_values else None
            ),
        },
    }


def _positive(value: float | None) -> bool:
    return value is not None and np.isfinite(value) and value > 0.0


def transfer_gate(record: Mapping[str, Any], correlations: Mapping[str, float | None]) -> dict[str, Any]:
    full = record["mixed"]["FULL"]["delta"]
    group_full = [
        record["by_group"][group]["FULL"]["delta"]["total_score"]
        for group in ("kpx_group_1", "kpx_group_2")
    ]
    checks = {
        "mixed_full_total_positive": full["total_score"]
        > GATE["transfer_mixed_full_total_min_exclusive"],
        "mixed_full_one_minus_nmae_nonnegative": full["one_minus_nmae"]
        >= GATE["transfer_mixed_full_component_min"],
        "mixed_full_ficr_nonnegative": full["ficr"]
        >= GATE["transfer_mixed_full_component_min"],
        "each_group_full_total_positive": min(group_full)
        > GATE["transfer_each_group_full_total_min_exclusive"],
        "worst_group_half_within_tolerance": record["worst"]["group_half_total_delta"]
        >= GATE["transfer_worst_group_half_total_min"],
        "worst_group_quarter_within_tolerance": record["worst"][
            "group_quarter_total_delta"
        ]
        >= GATE["transfer_worst_group_quarter_total_min"],
        "worst_group_month_within_tolerance": record["worst"][
            "group_month_total_delta"
        ]
        >= GATE["transfer_worst_group_month_total_min"],
        "positive_group_month_fraction": record["worst"][
            "positive_group_month_fraction"
        ]
        >= GATE["transfer_positive_group_month_fraction_min"],
        "each_residual_correlation_positive": all(
            _positive(correlations[group])
            and correlations[group] > GATE["minimum_residual_correlation_exclusive"]
            for group in ("kpx_group_1", "kpx_group_2")
        ),
    }
    return {"passed": bool(all(checks.values())), "checks": checks}


def h1h2_gate(record: Mapping[str, Any], correlations: Mapping[str, float | None]) -> dict[str, Any]:
    h2 = record["mixed"]["H2"]["delta"]
    group_h2 = [
        record["by_group"][group]["H2"]["delta"]["total_score"]
        for group in TARGET_COLS
    ]
    checks = {
        "mixed_h2_total_positive": h2["total_score"]
        > GATE["h1h2_mixed_h2_total_min_exclusive"],
        "mixed_h2_one_minus_nmae_nonnegative": h2["one_minus_nmae"]
        >= GATE["h1h2_mixed_h2_component_min"],
        "mixed_h2_ficr_nonnegative": h2["ficr"]
        >= GATE["h1h2_mixed_h2_component_min"],
        "each_group_h2_total_positive": min(group_h2)
        > GATE["h1h2_each_group_h2_total_min_exclusive"],
        "worst_group_quarter_within_tolerance": record["worst"][
            "group_quarter_total_delta"
        ]
        >= GATE["h1h2_worst_group_quarter_total_min"],
        "worst_group_month_within_tolerance": record["worst"][
            "group_month_total_delta"
        ]
        >= GATE["h1h2_worst_group_month_total_min"],
        "positive_group_month_fraction": record["worst"][
            "positive_group_month_fraction"
        ]
        >= GATE["h1h2_positive_group_month_fraction_min"],
        "each_residual_correlation_positive": all(
            _positive(correlations[group])
            and correlations[group] > GATE["minimum_residual_correlation_exclusive"]
            for group in TARGET_COLS
        ),
    }
    return {"passed": bool(all(checks.values())), "checks": checks}


def load_labels() -> pd.DataFrame:
    labels = pd.read_csv(LABEL_PATH, parse_dates=["kst_dtm"])
    labels = labels.set_index("kst_dtm").sort_index()
    return labels.loc[:, list(TARGET_COLS)].astype(np.float64)


def scaled_baseline(path: Path, index: pd.DatetimeIndex, groups: Sequence[str]) -> pd.DataFrame:
    frame = pd.read_parquet(path).reindex(index)
    if frame[list(groups)].isna().any().any():
        raise ValueError(f"baseline alignment introduced missing values: {path}")
    return frame.loc[:, list(groups)].astype(np.float64) * BASELINE_SCALE


def load_weather(group: str, split: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    pattern = CACHE_TRAIN_PATTERN if split == "train" else CACHE_TEST_PATTERN
    path = Path(str(pattern).format(group=group))
    frame = pd.read_parquet(path).reindex(index)
    if len(frame) != len(index):
        raise AssertionError("weather cache length differs after alignment")
    return frame


def year_mask(index: pd.DatetimeIndex, year: int) -> np.ndarray:
    return np.asarray(operating_clock(index).year == year)


def main() -> None:
    required = [LABEL_PATH, SAMPLE_PATH, BASE_2023_PATH, BASE_2024_PATH, BASE_2025_PATH]
    required.extend(
        Path(str(pattern).format(group=group))
        for group in TARGET_COLS
        for pattern in (CACHE_TRAIN_PATTERN, CACHE_TEST_PATTERN)
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required inputs: {missing}")

    labels = load_labels()
    mask_2023 = year_mask(labels.index, 2023)
    mask_2024 = year_mask(labels.index, 2024)
    labels_2023 = labels.loc[mask_2023, ["kpx_group_1", "kpx_group_2"]]
    labels_2024 = labels.loc[mask_2024, list(TARGET_COLS)]
    if len(labels_2023) != 8760 or len(labels_2024) != 8784:
        raise AssertionError("unexpected operating-year label lengths")

    base_2023 = scaled_baseline(
        BASE_2023_PATH, labels_2023.index, ("kpx_group_1", "kpx_group_2")
    )
    base_2024 = scaled_baseline(BASE_2024_PATH, labels_2024.index, TARGET_COLS)

    periods_all = (
        "FULL",
        "H1",
        "H2",
        "Q1",
        "Q2",
        "Q3",
        "Q4",
        *(f"M{month:02d}" for month in range(1, 13)),
    )
    periods_h2 = ("H2", "Q3", "Q4", *(f"M{month:02d}" for month in range(7, 13)))

    transfer_residuals: dict[str, np.ndarray] = {}
    transfer_fit: dict[str, Any] = {}
    transfer_correlations: dict[str, float | None] = {}
    for group in ("kpx_group_1", "kpx_group_2"):
        capacity = CAPACITY_KWH[group]
        weather_2023 = load_weather(group, "train", labels_2023.index)
        weather_2024 = load_weather(group, "train", labels_2024.index)
        base23_cf = base_2023[group].to_numpy(dtype=np.float64) / capacity
        base24_cf = base_2024[group].to_numpy(dtype=np.float64) / capacity
        x23 = vertical_features(weather_2023, base23_cf)
        x24 = vertical_features(weather_2024, base24_cf)
        model, diagnostics = fit_residual(
            x23,
            labels_2023[group],
            base23_cf,
            capacity,
            np.ones(len(labels_2023), dtype=bool),
        )
        predicted, clipped = predict_residual(model, x24)
        diagnostics["prediction_clipped_rows_2024"] = clipped
        transfer_fit[group] = diagnostics
        transfer_residuals[group] = predicted
        transfer_correlations[group] = correlation(
            labels_2024[group],
            base24_cf,
            predicted,
            capacity,
            np.ones(len(labels_2024), dtype=bool),
        )

    h1h2_residuals: dict[str, np.ndarray] = {}
    h1h2_fit: dict[str, Any] = {}
    h1h2_correlations: dict[str, float | None] = {}
    h1_mask = period_mask(labels_2024.index, "H1")
    h2_mask = period_mask(labels_2024.index, "H2")
    for group in TARGET_COLS:
        capacity = CAPACITY_KWH[group]
        weather_2024 = load_weather(group, "train", labels_2024.index)
        base24_cf = base_2024[group].to_numpy(dtype=np.float64) / capacity
        x24 = vertical_features(weather_2024, base24_cf)
        model, diagnostics = fit_residual(
            x24, labels_2024[group], base24_cf, capacity, h1_mask
        )
        predicted, clipped = predict_residual(model, x24)
        diagnostics["prediction_clipped_rows_2024"] = clipped
        h1h2_fit[group] = diagnostics
        h1h2_residuals[group] = predicted
        h1h2_correlations[group] = correlation(
            labels_2024[group], base24_cf, predicted, capacity, h2_mask
        )

    weight_records: dict[str, Any] = {}
    passing: list[tuple[float, float]] = []
    for weight in WEIGHTS:
        key = f"{weight:.2f}"
        transfer = evaluate_stage(
            labels_2024,
            base_2024,
            transfer_residuals,
            ("kpx_group_1", "kpx_group_2"),
            weight,
            periods_all,
            "FULL",
        )
        transfer["residual_correlation"] = transfer_correlations
        transfer["gate"] = transfer_gate(transfer, transfer_correlations)

        h1h2 = evaluate_stage(
            labels_2024,
            base_2024,
            h1h2_residuals,
            TARGET_COLS,
            weight,
            periods_h2,
            "H2",
        )
        h1h2["residual_correlation"] = h1h2_correlations
        h1h2["gate"] = h1h2_gate(h1h2, h1h2_correlations)

        joint_pass = bool(transfer["gate"]["passed"] and h1h2["gate"]["passed"])
        robustness = float(
            min(
                transfer["mixed"]["FULL"]["delta"]["total_score"],
                h1h2["mixed"]["H2"]["delta"]["total_score"],
            )
        )
        weight_records[key] = {
            "joint_pass": joint_pass,
            "robustness_objective": robustness,
            "transfer_2023_to_2024": transfer,
            "stability_2024_h1_to_h2": h1h2,
        }
        if joint_pass:
            passing.append((robustness, weight))

    # Deterministic validation-only selection: maximize the weaker of the two
    # causal aggregate improvements, then prefer the smaller weight on a tie.
    selected_weight: float | None = None
    if passing:
        selected_weight = sorted(passing, key=lambda item: (-item[0], item[1]))[0][1]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidate_identity: dict[str, Any] | None = None
    candidate_path: Path | None = None
    final_fit: dict[str, Any] = {}
    if selected_weight is not None:
        sample = pd.read_csv(SAMPLE_PATH)
        sample_index = pd.DatetimeIndex(pd.to_datetime(sample["forecast_kst_dtm"]))
        base_2025 = scaled_baseline(BASE_2025_PATH, sample_index, TARGET_COLS)
        candidate = sample.copy()
        for group in TARGET_COLS:
            capacity = CAPACITY_KWH[group]
            weather_2024 = load_weather(group, "train", labels_2024.index)
            base24_cf = base_2024[group].to_numpy(dtype=np.float64) / capacity
            x24 = vertical_features(weather_2024, base24_cf)
            model, diagnostics = fit_residual(
                x24,
                labels_2024[group],
                base24_cf,
                capacity,
                np.ones(len(labels_2024), dtype=bool),
            )

            weather_2025 = load_weather(group, "test", sample_index)
            base25_cf = base_2025[group].to_numpy(dtype=np.float64) / capacity
            x25 = vertical_features(weather_2025, base25_cf)
            residual25, clipped = predict_residual(model, x25)
            diagnostics["prediction_clipped_rows_2025"] = clipped
            diagnostics["prediction_2025_mean"] = float(np.mean(residual25))
            diagnostics["prediction_2025_std"] = float(np.std(residual25))
            final_fit[group] = diagnostics
            candidate[group] = np.clip(
                base_2025[group].to_numpy(dtype=np.float64)
                + selected_weight * capacity * residual25,
                0.0,
                1.02 * capacity,
            )

        expected_columns = ["forecast_id", "forecast_kst_dtm", *TARGET_COLS]
        if candidate.columns.tolist() != expected_columns or len(candidate) != 8760:
            raise AssertionError("submission schema changed")
        values = candidate.loc[:, list(TARGET_COLS)].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise AssertionError("submission contains a non-finite prediction")
        for group in TARGET_COLS:
            values_group = candidate[group].to_numpy(dtype=np.float64)
            if np.min(values_group) < 0 or np.max(values_group) > 1.02 * CAPACITY_KWH[group]:
                raise AssertionError(f"submission outside capacity bounds for {group}")

        candidate_path = OUT_DIR / (
            f"official_vertical_profile_residual_scale097_w{int(100 * selected_weight):02d}_2025.csv"
        )
        if candidate_path.exists():
            raise FileExistsError(f"refusing to overwrite existing candidate: {candidate_path}")
        candidate.to_csv(
            candidate_path,
            index=False,
            encoding="utf-8-sig",
            float_format="%.6f",
        )
        candidate_identity = file_identity(candidate_path)

    report = {
        "experiment": "official_vertical_profile_residual_sprint_v1",
        "status": "GO" if selected_weight is not None else "REJECT",
        "candidate_csv_created": candidate_path is not None,
        "selected_weight": selected_weight,
        "selection_rule": (
            "among joint-passing predeclared weights, maximize min(2023->2024 mixed FULL total delta, "
            "2024-H1->H2 mixed H2 total delta); tie -> smaller weight"
        ),
        "data_contract": {
            "official_cache_only": True,
            "public_leaderboard_values_read": False,
            "external_weather_used": False,
            "baseline_scale": BASELINE_SCALE,
            "groups_2023_to_2024": ["kpx_group_1", "kpx_group_2"],
            "group_3_validation": "2024 H1 fit -> 2024 H2 only",
            "feature_count": 27,
            "feature_geometry": "rotation-invariant relative vector dot/cross, magnitude, ratio, stability interactions",
            "model": "SimpleImputer(median) + StandardScaler + Ridge",
            "ridge_alpha": RIDGE_ALPHA,
            "weights": list(WEIGHTS),
            "residual_target_clip_cf": RESIDUAL_TARGET_CLIP,
            "residual_prediction_clip_cf": RESIDUAL_PREDICTION_CLIP,
        },
        "gate_thresholds": GATE,
        "input_identities": [file_identity(path) for path in required],
        "fit_diagnostics": {
            "transfer_2023_to_2024": transfer_fit,
            "stability_2024_h1_to_h2": h1h2_fit,
            "final_2024_full_to_2025": final_fit,
        },
        "weights": weight_records,
        "candidate": candidate_identity,
    }
    report_path = OUT_DIR / "vertical_profile_residual_report.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {report_path}")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = {
        "status": report["status"],
        "selected_weight": selected_weight,
        "candidate": candidate_identity,
        "report": file_identity(report_path),
        "weight_summary": {
            key: {
                "joint_pass": value["joint_pass"],
                "robustness_objective": value["robustness_objective"],
                "transfer_full_delta": value["transfer_2023_to_2024"]["mixed"][
                    "FULL"
                ]["delta"],
                "h1h2_h2_delta": value["stability_2024_h1_to_h2"]["mixed"][
                    "H2"
                ]["delta"],
            }
            for key, value in weight_records.items()
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

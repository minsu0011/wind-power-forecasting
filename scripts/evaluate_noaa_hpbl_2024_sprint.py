#!/usr/bin/env python
"""Run the frozen NOAA HPBL 2024 H1-fit -> H2 untouched causal gate.

The script intentionally has no 2025 input path and cannot create a submission.
It evaluates exactly one low-degree-of-freedom Ridge recipe and one fixed blend
weight frozen in ``configs/noaa_hpbl_2024_sprint_gate_v2.json``.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.metric import CAPACITY_KWH, TARGET_COLS, score_details


ROOT = REPO / "artifacts" / "final_submission_sprint_20260812" / "hpbl_2024"
CONFIG = REPO / "configs" / "noaa_hpbl_2024_sprint_gate_v2.json"
MANIFEST = ROOT / "MANIFEST.json"
HPBL = ROOT / "hpbl_2024_hourly.parquet"
BASELINE = (
    REPO
    / "artifacts"
    / "postgate"
    / "public_adaptive_scale097_g2_delta_v2"
    / "diagnostic_2024"
    / "A_scale097.parquet"
)
LABELS = Path(r"data/local/open/train/train_labels.csv")
OUT = ROOT / "gate"
EXPECTED_2024_MISSING_TIMESTAMPS = pd.DatetimeIndex(
    [
        "2024-07-11 12:00:00",
        "2024-07-11 13:00:00",
        "2024-07-11 14:00:00",
        "2024-07-30 12:00:00",
        "2024-07-30 13:00:00",
        "2024-07-30 14:00:00",
    ]
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    try:
        display = path.resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError:
        display = str(path.resolve())
    return {"path": display, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def write_parquet_exclusive(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    if path.exists() or partial.exists():
        raise FileExistsError(path)
    frame.to_parquet(partial, index=True, compression="zstd")
    partial.replace(path)


def metric_record(actual: pd.DataFrame, prediction: pd.DataFrame) -> dict[str, Any]:
    return asdict(score_details(actual, prediction))


def delta_record(
    actual: pd.DataFrame,
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    mask: np.ndarray,
) -> dict[str, Any]:
    before = metric_record(actual.loc[mask], baseline.loc[mask])
    after = metric_record(actual.loc[mask], candidate.loc[mask])
    return {
        "baseline": before,
        "candidate": after,
        "delta": {
            key: float(after[key] - before[key])
            for key in ("total_score", "one_minus_nmae", "ficr")
        },
    }


def group_delta(
    actual: pd.DataFrame,
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    group: str,
    mask: np.ndarray,
) -> dict[str, Any]:
    before = asdict(score_details(
        actual.loc[mask, [group]], baseline.loc[mask, [group]], target_cols=(group,)
    ))
    after = asdict(score_details(
        actual.loc[mask, [group]], candidate.loc[mask, [group]], target_cols=(group,)
    ))
    return {
        "baseline": before,
        "candidate": after,
        "delta": {
            key: float(after[key] - before[key])
            for key in ("total_score", "one_minus_nmae", "ficr")
        },
    }


def build_features(
    baseline_cf: np.ndarray, hpbl_m: np.ndarray, forecast_hour: np.ndarray
) -> pd.DataFrame:
    log_hpbl = np.log1p(np.asarray(hpbl_m, dtype=np.float64) / 1000.0)
    lead = np.asarray(forecast_hour, dtype=np.float64) - 27.0
    angle = 2.0 * np.pi * (lead - 1.0) / 24.0
    values = {
        "log1p_hpbl_km": log_hpbl,
        "baseline_cf_x_log1p_hpbl_km": baseline_cf * log_hpbl,
        "lead_sin_24h": np.sin(angle),
        "lead_cos_24h": np.cos(angle),
    }
    return pd.DataFrame(values, dtype=np.float64)


def build_model(config: dict[str, Any]) -> Pipeline:
    specification = config["single_recipe"]
    estimator = Ridge(
        alpha=float(specification["ridge_alpha"]),
        fit_intercept=bool(specification["fit_intercept"]),
    )
    return Pipeline([("scale", StandardScaler()), ("model", estimator)])


def require_exact_year_identity(
    manifest: dict[str, Any], observed: dict[str, Any]
) -> None:
    expected = manifest.get("year_values")
    required_keys = {"path", "size_bytes", "sha256"}
    if (
        not isinstance(expected, dict)
        or set(expected) != required_keys
        or set(observed) != required_keys
        or expected != observed
    ):
        raise RuntimeError(
            f"HPBL year parquet identity differs from acquisition manifest: "
            f"expected={expected!r}, observed={observed!r}"
        )


def validate_label_missingness(
    labels: pd.DataFrame,
    expected_index: pd.DatetimeIndex,
    h1_mask: np.ndarray,
    h2_mask: np.ndarray,
) -> dict[str, Any]:
    if len(labels) != 8784 or not labels.index.equals(expected_index):
        raise RuntimeError("2024 label index alignment failed")
    values = labels.loc[:, list(TARGET_COLS)].to_numpy(dtype=np.float64)
    if np.isinf(values).any():
        raise RuntimeError("2024 labels contain infinite values")
    missing = labels.loc[:, list(TARGET_COLS)].isna()
    reference = missing[TARGET_COLS[0]].to_numpy(dtype=bool)
    if not all(
        np.array_equal(reference, missing[group].to_numpy(dtype=bool))
        for group in TARGET_COLS[1:]
    ):
        raise RuntimeError("2024 label missing masks differ across groups")
    observed_missing = labels.index[reference]
    if not observed_missing.equals(EXPECTED_2024_MISSING_TIMESTAMPS):
        raise RuntimeError(
            f"2024 synchronized missing timestamps changed: {list(observed_missing)}"
        )
    by_group: dict[str, Any] = {}
    for group in TARGET_COLS:
        mask = missing[group].to_numpy(dtype=bool)
        h1_missing = int(np.sum(mask & h1_mask))
        h2_missing = int(np.sum(mask & h2_mask))
        if int(mask.sum()) != 6 or h1_missing != 0 or h2_missing != 6:
            raise RuntimeError(f"2024 label missing counts changed: {group}")
        by_group[group] = {
            "total": int(mask.sum()),
            "H1": h1_missing,
            "H2": h2_missing,
        }
    return {
        "missing_masks_identical_across_groups": True,
        "missing_timestamps": [str(value) for value in observed_missing],
        "by_group": by_group,
        "imputed": False,
        "official_metric_excludes_nonfinite_actual": True,
    }


def residual_fit_mask(
    actual: np.ndarray, capacity: float, h1_mask: np.ndarray
) -> np.ndarray:
    return (
        np.asarray(h1_mask, dtype=bool)
        & np.isfinite(actual)
        & (actual >= 0.10 * capacity)
    )


def correlation(
    actual: np.ndarray,
    baseline_cf: np.ndarray,
    residual_prediction: np.ndarray,
    capacity: float,
    mask: np.ndarray,
) -> float | None:
    eligible = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(actual)
        & (actual >= 0.10 * capacity)
        & np.isfinite(residual_prediction)
    )
    realized = actual[eligible] / capacity - baseline_cf[eligible]
    predicted = residual_prediction[eligible]
    if len(realized) < 3 or np.std(realized) == 0.0 or np.std(predicted) == 0.0:
        return None
    return float(np.corrcoef(realized, predicted)[0, 1])


def main() -> None:
    for path in (CONFIG, MANIFEST, HPBL, BASELINE, LABELS):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE" or manifest.get("hours") != 8784:
        raise RuntimeError("HPBL 2024 acquisition is not complete")
    observed_hpbl_identity = identity(HPBL)
    require_exact_year_identity(manifest, observed_hpbl_identity)
    if config["baseline"]["additional_scaling_forbidden"] is not True:
        raise RuntimeError("baseline scale contract changed")
    recipe = config["single_recipe"]
    if list(recipe["feature_order"]) != [
        "log1p_hpbl_km",
        "baseline_cf_x_log1p_hpbl_km",
        "lead_sin_24h",
        "lead_cos_24h",
    ]:
        raise RuntimeError("feature contract changed")
    if (
        config.get("status")
        != "SUPERSEDES_V1_BEFORE_ANY_2024_LABEL_OR_GATE_EVALUATION"
        or config.get("superseded_v1_execution_forbidden") is not True
        or recipe.get("standalone_baseline_cf_feature_forbidden") is not True
        or float(recipe.get("blend_weight")) != 0.10
    ):
        raise RuntimeError("single-recipe v2 supersession contract changed")

    hpbl = pd.read_parquet(HPBL)
    hpbl["forecast_kst_dtm"] = pd.to_datetime(hpbl["forecast_kst_dtm"], errors="raise")
    hpbl = hpbl.set_index("forecast_kst_dtm").sort_index()
    expected = pd.date_range("2024-01-01 01:00:00", "2025-01-01 00:00:00", freq="h")
    if len(hpbl) != 8784 or not hpbl.index.equals(expected):
        raise RuntimeError("HPBL year index changed")

    operating_clock = expected - pd.Timedelta(hours=1)
    h1 = np.asarray(operating_clock.month <= 6)
    h2 = np.asarray(operating_clock.month >= 7)
    late = h2 & hpbl["forecast_hour"].to_numpy(dtype=int).__ge__(44)
    if int(h1.sum()) != 182 * 24 or int(h2.sum()) != 184 * 24 or int(late.sum()) != 184 * 8:
        raise RuntimeError("H1/H2/late slice sizes changed")

    labels_all = pd.read_csv(LABELS, parse_dates=["kst_dtm"]).set_index("kst_dtm").sort_index()
    operating_clock_all = labels_all.index - pd.Timedelta(hours=1)
    labels = labels_all.loc[np.asarray(operating_clock_all.year == 2024), list(TARGET_COLS)].copy()
    labels = labels.reindex(expected)
    label_missingness = validate_label_missingness(labels, expected, h1, h2)
    baseline = pd.read_parquet(BASELINE).reindex(expected).loc[:, list(TARGET_COLS)].astype(np.float64)
    if baseline.isna().any().any():
        raise RuntimeError("already-scaled baseline alignment failed")

    prediction_columns: dict[str, np.ndarray] = {}
    fit_diagnostics: dict[str, Any] = {}
    residuals: dict[str, np.ndarray] = {}
    for group_position, group in enumerate(TARGET_COLS, start=1):
        capacity = float(CAPACITY_KWH[group])
        actual = labels[group].to_numpy(dtype=np.float64)
        baseline_cf = baseline[group].to_numpy(dtype=np.float64) / capacity
        hpbl_values = hpbl[f"hpbl_group_{group_position}_m"].to_numpy(dtype=np.float64)
        features = build_features(
            baseline_cf,
            hpbl_values,
            hpbl["forecast_hour"].to_numpy(dtype=np.float64),
        )
        if list(features.columns) != list(recipe["feature_order"]):
            raise RuntimeError("constructed feature order differs from frozen v2 config")
        fit_mask = residual_fit_mask(actual, capacity, h1)
        raw_target = actual / capacity - baseline_cf
        target_clip = recipe["residual_target_clip"]
        target = np.clip(raw_target, float(target_clip[0]), float(target_clip[1]))
        model = build_model(config)
        model.fit(features.loc[fit_mask], target[fit_mask])
        raw_prediction = np.asarray(model.predict(features), dtype=np.float64)
        prediction_clip = recipe["residual_prediction_clip"]
        prediction = np.clip(
            raw_prediction, float(prediction_clip[0]), float(prediction_clip[1])
        )
        if not np.isfinite(prediction).all():
            raise RuntimeError(f"non-finite residual prediction: {group}")
        residuals[group] = prediction
        prediction_columns[f"fixed_ridge__{group}__residual_cf"] = prediction
        estimator = model.named_steps["model"]
        fit_diagnostics[group] = {
            "fit_rows": int(fit_mask.sum()),
            "target_mean": float(target[fit_mask].mean()),
            "target_std": float(target[fit_mask].std()),
            "target_clipped_rows": int(np.sum(target[fit_mask] != raw_target[fit_mask])),
            "prediction_clipped_rows": int(np.sum(prediction != raw_prediction)),
            "prediction_h2_mean": float(prediction[h2].mean()),
            "prediction_h2_std": float(prediction[h2].std()),
            "intercept": float(estimator.intercept_),
            "coefficients_standardized": {
                name: float(value)
                for name, value in zip(recipe["feature_order"], estimator.coef_)
            },
        }

    prediction_frame = pd.DataFrame(prediction_columns, index=expected)
    prediction_frame.index.name = "forecast_kst_dtm"
    prediction_path = OUT / "head_residual_predictions_2024.parquet"
    write_parquet_exclusive(prediction_path, prediction_frame)

    gate = config["gate"]
    weight = float(recipe["blend_weight"])
    candidate = baseline.copy()
    correlations: dict[str, float | None] = {}
    for group in TARGET_COLS:
        capacity = float(CAPACITY_KWH[group])
        candidate[group] = np.clip(
            baseline[group].to_numpy(dtype=np.float64)
            + weight * capacity * residuals[group],
            0.0,
            capacity,
        )
        correlations[group] = correlation(
            labels[group].to_numpy(dtype=np.float64),
            baseline[group].to_numpy(dtype=np.float64) / capacity,
            residuals[group],
            capacity,
            h2,
        )
    mixed_h2 = delta_record(labels, baseline, candidate, h2)
    by_group = {
        group: group_delta(labels, baseline, candidate, group, h2)
        for group in TARGET_COLS
    }
    late_record = delta_record(labels, baseline, candidate, late)
    months = {
        f"M{month:02d}": delta_record(
            labels,
            baseline,
            candidate,
            h2 & np.asarray(operating_clock.month == month),
        )
        for month in range(7, 13)
    }
    worst_month = min(record["delta"]["total_score"] for record in months.values())
    checks = {
        "mixed_h2_delta_score": mixed_h2["delta"]["total_score"]
        >= float(gate["mixed_h2_delta_score_min"]),
        "mixed_h2_delta_ficr": mixed_h2["delta"]["ficr"]
        >= float(gate["mixed_h2_delta_ficr_min"]),
        "each_group_h2_delta_score": min(
            record["delta"]["total_score"] for record in by_group.values()
        )
        >= float(gate["each_group_h2_delta_score_min"]),
        "late_f044_f051_mixed_delta_score": late_record["delta"]["total_score"]
        >= float(gate["late_f044_f051_mixed_delta_score_min"]),
        "worst_mixed_operating_month_delta_score": worst_month
        >= float(gate["worst_mixed_operating_month_delta_score_min"]),
        "residual_correlation_each_group": all(
            value is not None
            and np.isfinite(value)
            and value > float(gate["residual_correlation_each_group_min_exclusive"])
            for value in correlations.values()
        ),
    }
    passed = bool(all(checks.values()))
    decision = "GO_DOWNLOAD_2025" if passed else "REJECT_STOP_NO_2025_NO_CSV"
    fixed_candidate = {
        "recipe": "per_group_standardized_ridge_alpha100",
        "blend_weight": weight,
        "passed": passed,
        "checks": checks,
        "mixed_h2": mixed_h2,
        "by_group_h2": by_group,
        "late_f044_f051_h2": late_record,
        "operating_months_h2": months,
        "worst_mixed_operating_month_delta_score": float(worst_month),
        "residual_correlation_h2": correlations,
    }
    report = {
        "schema_version": 2,
        "artifact_type": "NOAA_HPBL_2024_H1_FIT_H2_UNTOUCHED_SINGLE_RECIPE_GATE",
        "decision": decision,
        "fixed_candidate_passed": passed,
        "candidate_count": 1,
        "model_or_weight_selection_performed": False,
        "no_2025_data_read": True,
        "csv_created": False,
        "baseline_already_scaled_097_and_not_scaled_again": True,
        "label_missingness": label_missingness,
        "slice_rows": {"H1": int(h1.sum()), "H2": int(h2.sum()), "late_f044_f051_H2": int(late.sum())},
        "gate_thresholds": gate,
        "selection_rule": config["selection"],
        "fit_diagnostics": fit_diagnostics,
        "fixed_candidate": fixed_candidate,
        "inputs": {
            "config": identity(CONFIG),
            "acquisition_manifest": identity(MANIFEST),
            "hpbl_values": observed_hpbl_identity,
            "already_scaled_baseline": identity(BASELINE),
            "labels": identity(LABELS),
        },
        "outputs": {"head_residual_predictions": identity(prediction_path)},
        "risk_notes": [
            "The A_scale097 baseline is already 2024 calibration-fit/selection-consumed and is not a pristine holdout baseline.",
            "The v1 multi-head/multi-weight design was superseded and never executed; exactly one v2 prediction is frozen before H2 metrics.",
            "A GO authorizes a separate 2025 acquisition; this program has no 2025 input or CSV code path."
        ],
        "script": identity(Path(__file__).resolve()),
    }
    report_path = OUT / "GATE_RESULTS.json"
    write_json_exclusive(report_path, report)
    summary = {
        "decision": decision,
        "fixed_candidate": {
            "passed": passed,
            "delta_score": mixed_h2["delta"]["total_score"],
            "delta_n": mixed_h2["delta"]["one_minus_nmae"],
            "delta_ficr": mixed_h2["delta"]["ficr"],
            "group_delta_score": {
                group: record["delta"]["total_score"] for group, record in by_group.items()
            },
            "late_delta_score": late_record["delta"]["total_score"],
            "worst_month_delta_score": worst_month,
        },
        "report": identity(report_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

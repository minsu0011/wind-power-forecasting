"""Last-48-hour, official-data-only vertical-profile direct-model sprint.

This script deliberately evaluates only two fixed, low-complexity LightGBM
recipes and three fixed deployment choices (standalone, 5% blend, 10% blend).
The validation design is forward-only:

* 2022 fit -> 2023 apply for groups 1 and 2; and
* 2022-2023 fit -> 2024 apply for groups 1, 2 and 3.

Group 3 cannot be evaluated in the first fold because all of its 2022 labels
are unavailable.  The current official-only deployment baseline is represented
by the already-fixed 0.97 scale applied to the matching stored forward OOF
baseline.  No Public score or Public component is read by this program.

All generated artifacts are confined to
``artifacts/final_submission_sprint_20260812/vertical_direct``.  A submission
CSV is emitted only when a candidate passes the fixed stability gate below.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metric import CAPACITY_KWH, group_metrics, score_details


RAW = Path(r"data/local/open")
OUT = ROOT / "artifacts" / "final_submission_sprint_20260812" / "vertical_direct"
LABELS_PATH = RAW / "train" / "train_labels.csv"
SAMPLE_PATH = RAW / "sample_submission.csv"
BASELINE_2023_PATH = ROOT / "artifacts" / "oof" / "dev2023_locked_v3.parquet"
BASELINE_2024_PATH = (
    ROOT
    / "artifacts"
    / "oof"
    / "gate2024_recent_v4_cf_fix_calibration_fit.parquet"
)
FINAL_BASELINE_PATH = (
    ROOT
    / "artifacts"
    / "final_submission_sprint_20260812"
    / "01_OFFICIAL_ONLY_SAFE.csv"
)

GROUPS = ("kpx_group_1", "kpx_group_2", "kpx_group_3")
CURRENT_BASE_SCALE = 0.97
BLEND_WEIGHTS = (1.0, 0.05, 0.10)
EPS = 1.0e-6

# Fixed before labels are loaded.  These are intentionally modest models: the
# sprint is a targeted feature test, not a broad HPO exercise.
RECIPES: dict[str, dict[str, Any]] = {
    "lgb_mae_vertical": {
        "objective": "regression_l1",
        "alpha": None,
        "n_estimators": 500,
        "learning_rate": 0.035,
        "num_leaves": 24,
        "max_depth": 6,
        "min_child_samples": 120,
        "max_bin": 127,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.65,
        "reg_alpha": 0.10,
        "reg_lambda": 1.00,
    },
    "lgb_q060_vertical": {
        "objective": "quantile",
        "alpha": 0.60,
        "n_estimators": 500,
        "learning_rate": 0.035,
        "num_leaves": 24,
        "max_depth": 6,
        "min_child_samples": 120,
        "max_bin": 127,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.65,
        "reg_alpha": 0.10,
        "reg_lambda": 1.00,
    },
}

# A candidate may retain tiny slice noise, but it may not show a systematic or
# material failure in any group/fold.  These thresholds are fixed and do not
# depend on the observed results.
GATE = {
    "aggregate_full_min_total_delta_each_fold": 0.0,
    "aggregate_full_min_component_delta_each_fold": -0.0005,
    "group_full_min_total_delta": 0.0,
    "group_full_min_component_delta": -0.0010,
    "group_half_min_total_delta": -0.0010,
    "group_quarter_min_total_delta": -0.0015,
    "group_month_min_total_delta": -0.0060,
    "group_quarter_min_nonnegative_fraction": 0.60,
    "group_month_min_nonnegative_fraction": 0.50,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(json_ready(payload), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return numerator / np.maximum(denominator, EPS)


def add_vertical_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Append rotation-invariant GFS 100 m/PBL/850-hPa profile geometry."""

    augmented = frame.copy()
    created: list[str] = []
    for method in ("global_mean", "idw", "nearest"):
        prefix = f"gfs__{method}__"
        u100 = augmented[f"{prefix}heightAboveGround_100_100u"].to_numpy(np.float64)
        v100 = augmented[f"{prefix}heightAboveGround_100_100v"].to_numpy(np.float64)
        upbl = augmented[f"{prefix}planetaryBoundaryLayer_0_u"].to_numpy(np.float64)
        vpbl = augmented[f"{prefix}planetaryBoundaryLayer_0_v"].to_numpy(np.float64)
        u850 = augmented[f"{prefix}isobaricInhPa_850_u"].to_numpy(np.float64)
        v850 = augmented[f"{prefix}isobaricInhPa_850_v"].to_numpy(np.float64)

        ws100 = np.hypot(u100, v100)
        wspbl = np.hypot(upbl, vpbl)
        ws850 = np.hypot(u850, v850)

        def append(name: str, values: np.ndarray) -> None:
            column = f"vertical__{method}__{name}"
            augmented[column] = np.asarray(values, dtype=np.float32)
            created.append(column)

        append("ws100", ws100)
        append("ws_pbl", wspbl)
        append("ws850", ws850)
        append("speed_pbl_minus_100", wspbl - ws100)
        append("speed_850_minus_pbl", ws850 - wspbl)
        append("speed_850_minus_100", ws850 - ws100)
        append("ratio_pbl_to_100", safe_ratio(wspbl, ws100))
        append("ratio_850_to_pbl", safe_ratio(ws850, wspbl))
        append("ratio_850_to_100", safe_ratio(ws850, ws100))
        append("vector_shear_100_pbl", np.hypot(upbl - u100, vpbl - v100))
        append("vector_shear_pbl_850", np.hypot(u850 - upbl, v850 - vpbl))
        append("vector_shear_100_850", np.hypot(u850 - u100, v850 - v100))
        cos_100_pbl = safe_ratio(u100 * upbl + v100 * vpbl, ws100 * wspbl)
        cos_pbl_850 = safe_ratio(upbl * u850 + vpbl * v850, wspbl * ws850)
        cos_100_850 = safe_ratio(u100 * u850 + v100 * v850, ws100 * ws850)
        append("cos_100_pbl", np.clip(cos_100_pbl, -1.0, 1.0))
        append("cos_pbl_850", np.clip(cos_pbl_850, -1.0, 1.0))
        append("cos_100_850", np.clip(cos_100_850, -1.0, 1.0))
        speed_stack = np.column_stack([ws100, wspbl, ws850])
        append("profile_speed_mean", np.mean(speed_stack, axis=1))
        append("profile_speed_std", np.std(speed_stack, axis=1))
        append(
            "profile_alignment_mean",
            np.mean(np.column_stack([cos_100_pbl, cos_pbl_850, cos_100_850]), axis=1),
        )

    if len(created) != 54 or augmented.shape[1] != frame.shape[1] + 54:
        raise AssertionError("vertical feature contract changed")
    return augmented, created


def load_feature_pair(group: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    number = int(group.rsplit("_", 1)[1])
    train_path = ROOT / "artifacts" / "cache" / f"kpx_group_{number}_weather_train.parquet"
    test_path = ROOT / "artifacts" / "cache" / f"kpx_group_{number}_weather_test.parquet"
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    if train.shape != (26304, 612) or test.shape != (8760, 612):
        raise AssertionError(f"unexpected cache shape for {group}")
    if not train.columns.equals(test.columns):
        raise AssertionError(f"train/test cache columns differ for {group}")
    train_aug, created = add_vertical_features(train)
    test_aug, created_test = add_vertical_features(test)
    if created != created_test or not train_aug.columns.equals(test_aug.columns):
        raise AssertionError(f"augmented feature parity failed for {group}")
    return train_aug.astype(np.float32), test_aug.astype(np.float32), created


def make_model(recipe_name: str, group: str) -> lgb.LGBMRegressor:
    params = dict(RECIPES[recipe_name])
    if params.get("alpha") is None:
        params.pop("alpha")
    group_number = int(group.rsplit("_", 1)[1])
    return lgb.LGBMRegressor(
        **params,
        random_state=20260812 + group_number,
        bagging_seed=20260842 + group_number,
        feature_fraction_seed=20260872 + group_number,
        deterministic=True,
        force_col_wise=True,
        n_jobs=max(1, min(10, os.cpu_count() or 1)),
        verbosity=-1,
    )


def fit_predict(
    recipe_name: str,
    group: str,
    features: pd.DataFrame,
    labels: pd.Series,
    train_index: pd.DatetimeIndex,
    valid_index: pd.DatetimeIndex,
) -> tuple[np.ndarray, dict[str, Any]]:
    capacity = CAPACITY_KWH[group]
    y = labels.reindex(train_index).to_numpy(np.float64)
    eligible = np.isfinite(y) & (y >= 0.10 * capacity)
    if int(eligible.sum()) == 0:
        raise ValueError(f"no eligible training labels for {group}")
    x_train = features.reindex(train_index).iloc[np.flatnonzero(eligible)]
    y_cf = y[eligible] / capacity
    x_valid = features.reindex(valid_index)
    model = make_model(recipe_name, group)
    model.fit(x_train, y_cf)
    prediction_cf = np.asarray(model.predict(x_valid), dtype=np.float64)
    prediction = np.clip(prediction_cf * capacity, 0.0, capacity)
    diagnostics = {
        "eligible_training_rows": int(eligible.sum()),
        "training_start": str(train_index.min()),
        "training_end": str(train_index.max()),
        "validation_start": str(valid_index.min()),
        "validation_end": str(valid_index.max()),
        "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
        "prediction_min_kwh": float(np.min(prediction)),
        "prediction_max_kwh": float(np.max(prediction)),
    }
    return prediction, diagnostics


def slice_masks(index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    month = index.month.to_numpy()
    masks: dict[str, np.ndarray] = {"full": np.ones(len(index), dtype=bool)}
    masks["H1"] = month <= 6
    masks["H2"] = month >= 7
    for quarter in range(1, 5):
        masks[f"Q{quarter}"] = ((month - 1) // 3 + 1) == quarter
    for number in range(1, 13):
        masks[f"M{number:02d}"] = month == number
    return masks


def slice_type(name: str) -> str:
    if name == "full":
        return "full"
    if name.startswith("H"):
        return "half"
    if name.startswith("Q"):
        return "quarter"
    return "month"


def metrics_tuple(actual: pd.DataFrame, prediction: pd.DataFrame, groups: Iterable[str]) -> tuple[float, float, float]:
    selected = tuple(groups)
    result = score_details(actual, prediction, target_cols=selected, capacities=CAPACITY_KWH)
    return result.total_score, result.one_minus_nmae, result.ficr


def evaluate_variant(
    *,
    recipe: str,
    variant: str,
    fold: str,
    actual: pd.DataFrame,
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    groups: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    masks = slice_masks(actual.index)
    for name, mask in masks.items():
        ix = actual.index[mask]
        base_values = metrics_tuple(actual.loc[ix], baseline.loc[ix], groups)
        cand_values = metrics_tuple(actual.loc[ix], candidate.loc[ix], groups)
        rows.append(
            {
                "recipe": recipe,
                "variant": variant,
                "fold": fold,
                "level": "aggregate",
                "group": "ALL",
                "slice": name,
                "slice_type": slice_type(name),
                "n_rows": len(ix),
                "base_score": base_values[0],
                "candidate_score": cand_values[0],
                "delta_score": cand_values[0] - base_values[0],
                "base_one_minus_nmae": base_values[1],
                "candidate_one_minus_nmae": cand_values[1],
                "delta_one_minus_nmae": cand_values[1] - base_values[1],
                "base_ficr": base_values[2],
                "candidate_ficr": cand_values[2],
                "delta_ficr": cand_values[2] - base_values[2],
            }
        )
        for group in groups:
            capacity = CAPACITY_KWH[group]
            base_group = group_metrics(actual.loc[ix, group], baseline.loc[ix, group], capacity, group_name=group)
            cand_group = group_metrics(actual.loc[ix, group], candidate.loc[ix, group], capacity, group_name=group)
            base_score = 0.5 * base_group.one_minus_nmae + 0.5 * base_group.ficr
            cand_score = 0.5 * cand_group.one_minus_nmae + 0.5 * cand_group.ficr
            rows.append(
                {
                    "recipe": recipe,
                    "variant": variant,
                    "fold": fold,
                    "level": "group",
                    "group": group,
                    "slice": name,
                    "slice_type": slice_type(name),
                    "n_rows": len(ix),
                    "base_score": base_score,
                    "candidate_score": cand_score,
                    "delta_score": cand_score - base_score,
                    "base_one_minus_nmae": base_group.one_minus_nmae,
                    "candidate_one_minus_nmae": cand_group.one_minus_nmae,
                    "delta_one_minus_nmae": cand_group.one_minus_nmae - base_group.one_minus_nmae,
                    "base_ficr": base_group.ficr,
                    "candidate_ficr": cand_group.ficr,
                    "delta_ficr": cand_group.ficr - base_group.ficr,
                }
            )
    return rows


def summarize_gate(table: pd.DataFrame, recipe: str, variant: str) -> dict[str, Any]:
    subset = table[(table["recipe"] == recipe) & (table["variant"] == variant)]
    agg_full = subset[(subset["level"] == "aggregate") & (subset["slice_type"] == "full")]
    group_full = subset[(subset["level"] == "group") & (subset["slice_type"] == "full")]
    group_half = subset[(subset["level"] == "group") & (subset["slice_type"] == "half")]
    group_quarter = subset[(subset["level"] == "group") & (subset["slice_type"] == "quarter")]
    group_month = subset[(subset["level"] == "group") & (subset["slice_type"] == "month")]

    checks = {
        "aggregate_full_total_each_fold": bool(
            (agg_full["delta_score"] >= GATE["aggregate_full_min_total_delta_each_fold"]).all()
        ),
        "aggregate_full_components_each_fold": bool(
            (
                agg_full[["delta_one_minus_nmae", "delta_ficr"]]
                >= GATE["aggregate_full_min_component_delta_each_fold"]
            ).all().all()
        ),
        "group_full_total": bool(
            (group_full["delta_score"] >= GATE["group_full_min_total_delta"]).all()
        ),
        "group_full_components": bool(
            (
                group_full[["delta_one_minus_nmae", "delta_ficr"]]
                >= GATE["group_full_min_component_delta"]
            ).all().all()
        ),
        "group_half_total": bool(
            group_half["delta_score"].min() >= GATE["group_half_min_total_delta"]
        ),
        "group_quarter_total": bool(
            group_quarter["delta_score"].min() >= GATE["group_quarter_min_total_delta"]
        ),
        "group_month_total": bool(
            group_month["delta_score"].min() >= GATE["group_month_min_total_delta"]
        ),
        "group_quarter_sign_fraction": bool(
            (group_quarter["delta_score"] >= 0.0).mean()
            >= GATE["group_quarter_min_nonnegative_fraction"]
        ),
        "group_month_sign_fraction": bool(
            (group_month["delta_score"] >= 0.0).mean()
            >= GATE["group_month_min_nonnegative_fraction"]
        ),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "aggregate_full_mean_delta_score": float(agg_full["delta_score"].mean()),
        "aggregate_full_worst_delta_score": float(agg_full["delta_score"].min()),
        "aggregate_full_worst_delta_one_minus_nmae": float(agg_full["delta_one_minus_nmae"].min()),
        "aggregate_full_worst_delta_ficr": float(agg_full["delta_ficr"].min()),
        "group_full_worst_delta_score": float(group_full["delta_score"].min()),
        "group_half_worst_delta_score": float(group_half["delta_score"].min()),
        "group_quarter_worst_delta_score": float(group_quarter["delta_score"].min()),
        "group_month_worst_delta_score": float(group_month["delta_score"].min()),
        "group_quarter_nonnegative_fraction": float((group_quarter["delta_score"] >= 0.0).mean()),
        "group_month_nonnegative_fraction": float((group_month["delta_score"] >= 0.0).mean()),
    }


def variant_name(weight: float) -> str:
    if weight == 1.0:
        return "standalone"
    return f"blend_{int(round(weight * 100)):02d}pct"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    validation_dir = OUT / "validation_predictions"
    validation_dir.mkdir(exist_ok=True)

    labels = pd.read_csv(LABELS_PATH, encoding="utf-8-sig", parse_dates=["kst_dtm"]).set_index("kst_dtm")
    labels = labels.loc[:, list(GROUPS)].astype(np.float64)
    baseline_2023_raw = pd.read_parquet(BASELINE_2023_PATH).astype(np.float64)
    baseline_2024_raw = pd.read_parquet(BASELINE_2024_PATH).astype(np.float64)
    baseline_2023 = (CURRENT_BASE_SCALE * baseline_2023_raw).clip(lower=0.0)
    baseline_2024 = (CURRENT_BASE_SCALE * baseline_2024_raw).clip(lower=0.0)
    for group in baseline_2023:
        baseline_2023[group] = baseline_2023[group].clip(upper=CAPACITY_KWH[group])
    for group in baseline_2024:
        baseline_2024[group] = baseline_2024[group].clip(upper=CAPACITY_KWH[group])

    folds = {
        "fit2022_apply2023": {
            "train_index": labels.index[labels.index.year == 2022],
            "valid_index": baseline_2023.index,
            "groups": ("kpx_group_1", "kpx_group_2"),
            "baseline": baseline_2023,
        },
        "fit2022_2023_apply2024": {
            "train_index": labels.index[labels.index.year <= 2023],
            "valid_index": baseline_2024.index,
            "groups": GROUPS,
            "baseline": baseline_2024,
        },
    }

    features_by_group: dict[str, pd.DataFrame] = {}
    test_features_by_group: dict[str, pd.DataFrame] = {}
    feature_names: list[str] | None = None
    input_records: dict[str, Any] = {
        "labels": file_record(LABELS_PATH),
        "sample_submission": file_record(SAMPLE_PATH),
        "baseline_2023": file_record(BASELINE_2023_PATH),
        "baseline_2024": file_record(BASELINE_2024_PATH),
        "final_baseline": file_record(FINAL_BASELINE_PATH),
        "weather_cache": {},
    }
    for group in GROUPS:
        train, test, created = load_feature_pair(group)
        features_by_group[group] = train
        test_features_by_group[group] = test
        if feature_names is None:
            feature_names = created
        elif feature_names != created:
            raise AssertionError("created features differ across groups")
        number = int(group.rsplit("_", 1)[1])
        input_records["weather_cache"][group] = {
            "train": file_record(ROOT / "artifacts" / "cache" / f"kpx_group_{number}_weather_train.parquet"),
            "test": file_record(ROOT / "artifacts" / "cache" / f"kpx_group_{number}_weather_test.parquet"),
        }

    all_metric_rows: list[dict[str, Any]] = []
    fit_diagnostics: dict[str, Any] = {}
    raw_predictions: dict[tuple[str, str], pd.DataFrame] = {}
    for recipe in RECIPES:
        fit_diagnostics[recipe] = {}
        for fold, contract in folds.items():
            valid_index = contract["valid_index"]
            groups = contract["groups"]
            direct = pd.DataFrame(index=valid_index, columns=list(groups), dtype=np.float64)
            fit_diagnostics[recipe][fold] = {}
            for group in groups:
                prediction, diagnostics = fit_predict(
                    recipe,
                    group,
                    features_by_group[group],
                    labels[group],
                    contract["train_index"],
                    valid_index,
                )
                direct[group] = prediction
                fit_diagnostics[recipe][fold][group] = diagnostics
            direct_path = validation_dir / f"{recipe}__{fold}__standalone.parquet"
            direct.to_parquet(direct_path)
            raw_predictions[(recipe, fold)] = direct
            baseline = contract["baseline"].loc[valid_index, list(groups)]
            actual = labels.loc[valid_index, list(groups)]
            for weight in BLEND_WEIGHTS:
                variant = variant_name(weight)
                if weight == 1.0:
                    proposed = direct.copy()
                else:
                    proposed = (1.0 - weight) * baseline + weight * direct
                for group in groups:
                    proposed[group] = proposed[group].clip(0.0, CAPACITY_KWH[group])
                all_metric_rows.extend(
                    evaluate_variant(
                        recipe=recipe,
                        variant=variant,
                        fold=fold,
                        actual=actual,
                        baseline=baseline,
                        candidate=proposed,
                        groups=groups,
                    )
                )

    metric_table = pd.DataFrame(all_metric_rows)
    metric_path = OUT / "slice_metrics.csv"
    metric_table.to_csv(metric_path, index=False, encoding="utf-8-sig", float_format="%.12f")

    gates: dict[str, Any] = {}
    passing: list[tuple[str, str, dict[str, Any]]] = []
    for recipe in RECIPES:
        gates[recipe] = {}
        for weight in BLEND_WEIGHTS:
            variant = variant_name(weight)
            summary = summarize_gate(metric_table, recipe, variant)
            gates[recipe][variant] = summary
            if summary["passed"]:
                passing.append((recipe, variant, summary))

    passing.sort(
        key=lambda item: (
            item[2]["aggregate_full_mean_delta_score"],
            -({"standalone": 1.0, "blend_05pct": 0.05, "blend_10pct": 0.10}[item[1]]),
        ),
        reverse=True,
    )

    final_artifacts: dict[str, Any] = {"submission_generated": False}
    if passing:
        selected_recipe, selected_variant, selected_summary = passing[0]
        selected_weight = {"standalone": 1.0, "blend_05pct": 0.05, "blend_10pct": 0.10}[selected_variant]
        full_train_index = labels.index
        final_direct = pd.DataFrame(index=test_features_by_group[GROUPS[0]].index, columns=list(GROUPS), dtype=np.float64)
        final_fit_diagnostics: dict[str, Any] = {}
        for group in GROUPS:
            capacity = CAPACITY_KWH[group]
            y = labels[group].to_numpy(np.float64)
            eligible = np.isfinite(y) & (y >= 0.10 * capacity)
            model = make_model(selected_recipe, group)
            model.fit(features_by_group[group].iloc[np.flatnonzero(eligible)], y[eligible] / capacity)
            prediction = np.clip(
                np.asarray(model.predict(test_features_by_group[group]), dtype=np.float64) * capacity,
                0.0,
                capacity,
            )
            final_direct[group] = prediction
            final_fit_diagnostics[group] = {
                "final_refit_eligible_rows": int(eligible.sum()),
                "training_start": str(full_train_index.min()),
                "training_end": str(full_train_index.max()),
                "test_start": str(test_features_by_group[group].index.min()),
                "test_end": str(test_features_by_group[group].index.max()),
                "prediction_min_kwh": float(np.min(prediction)),
                "prediction_max_kwh": float(np.max(prediction)),
            }

        sample = pd.read_csv(SAMPLE_PATH, encoding="utf-8-sig")
        sample_index = pd.to_datetime(sample["forecast_kst_dtm"], errors="raise")
        if not pd.DatetimeIndex(sample_index).equals(final_direct.index):
            raise AssertionError("sample and test feature timestamps differ")
        base_csv = pd.read_csv(FINAL_BASELINE_PATH, encoding="utf-8-sig")
        if not sample[["forecast_id", "forecast_kst_dtm"]].equals(
            base_csv[["forecast_id", "forecast_kst_dtm"]]
        ):
            raise AssertionError("final baseline identifiers differ from sample")
        base_values = base_csv.loc[:, list(GROUPS)].to_numpy(np.float64)
        direct_values = final_direct.loc[:, list(GROUPS)].to_numpy(np.float64)
        final_values = (
            direct_values
            if selected_weight == 1.0
            else (1.0 - selected_weight) * base_values + selected_weight * direct_values
        )
        for column_number, group in enumerate(GROUPS):
            final_values[:, column_number] = np.clip(final_values[:, column_number], 0.0, CAPACITY_KWH[group])
        submission = sample.copy()
        submission.loc[:, list(GROUPS)] = final_values
        csv_path = OUT / f"02_VERTICAL_DIRECT_{selected_recipe}_{selected_variant}.csv"
        submission.to_csv(csv_path, index=False, encoding="utf-8-sig", float_format="%.6f")
        parquet_path = OUT / f"02_VERTICAL_DIRECT_{selected_recipe}_{selected_variant}.parquet"
        final_direct.to_parquet(parquet_path)
        reread = pd.read_csv(csv_path, encoding="utf-8-sig")
        checks = {
            "rows": len(reread) == 8760,
            "columns_exact": reread.columns.tolist() == sample.columns.tolist(),
            "forecast_id_exact": reread["forecast_id"].equals(sample["forecast_id"]),
            "forecast_kst_dtm_exact": reread["forecast_kst_dtm"].equals(sample["forecast_kst_dtm"]),
            "utf8_bom": csv_path.read_bytes()[:3] == b"\xef\xbb\xbf",
            "finite": bool(np.isfinite(reread.loc[:, list(GROUPS)].to_numpy(np.float64)).all()),
            "within_capacity": bool(
                all(
                    reread[group].between(0.0, CAPACITY_KWH[group], inclusive="both").all()
                    for group in GROUPS
                )
            ),
        }
        if not all(checks.values()):
            raise AssertionError(f"submission validation failed: {checks}")
        final_artifacts = {
            "submission_generated": True,
            "selected_recipe": selected_recipe,
            "selected_variant": selected_variant,
            "selected_weight": selected_weight,
            "selected_gate_summary": selected_summary,
            "submission": file_record(csv_path),
            "direct_test_prediction": file_record(parquet_path),
            "validation_checks": checks,
            "final_fit_diagnostics": final_fit_diagnostics,
            "observed_ranges_kwh": {
                group: [float(reread[group].min()), float(reread[group].max())]
                for group in GROUPS
            },
        }
    else:
        marker = OUT / "NO_CSV_GENERATED.txt"
        marker.write_text(
            "No vertical direct candidate passed the pre-fixed forward stability gate.\n"
            "Creating a submission CSV was therefore forbidden by the sprint contract.\n",
            encoding="utf-8",
        )
        final_artifacts["reason"] = "no candidate passed all stability checks"
        final_artifacts["marker"] = file_record(marker)

    compact_full = (
        metric_table[
            (metric_table["level"] == "aggregate")
            & (metric_table["slice"] == "full")
        ][
            [
                "recipe",
                "variant",
                "fold",
                "base_score",
                "candidate_score",
                "delta_score",
                "delta_one_minus_nmae",
                "delta_ficr",
            ]
        ]
        .sort_values(["recipe", "variant", "fold"])
        .to_dict(orient="records")
    )
    results = {
        "experiment": "official_only_vertical_profile_direct_sprint_v1",
        "public_feedback_used": False,
        "external_data_used": False,
        "target": "eligible actual_kwh / group_capacity_kwh",
        "baseline": {
            "description": "fixed 0.97 scale of matching current forward OOF baseline",
            "scale": CURRENT_BASE_SCALE,
        },
        "forward_folds": {
            "fit2022_apply2023": {
                "groups": ["kpx_group_1", "kpx_group_2"],
                "group_3_exclusion": "all 2022 group-3 labels are unavailable",
            },
            "fit2022_2023_apply2024": {"groups": list(GROUPS)},
        },
        "base_feature_count": 612,
        "created_vertical_feature_count": len(feature_names or []),
        "total_feature_count": 612 + len(feature_names or []),
        "created_vertical_features": feature_names,
        "recipes": RECIPES,
        "blend_weights": list(BLEND_WEIGHTS),
        "gate_contract": GATE,
        "gate_results": gates,
        "aggregate_full_results": compact_full,
        "fit_diagnostics": fit_diagnostics,
        "inputs": input_records,
        "metric_table": file_record(metric_path),
        "final": final_artifacts,
    }
    results_path = OUT / "RESULTS.json"
    write_json(results_path, results)
    print(json.dumps({"results": str(results_path), "final": final_artifacts}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

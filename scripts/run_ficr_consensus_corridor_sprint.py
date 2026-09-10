"""Evaluate the preregistered official-only FICR consensus corridor recipes."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metric import CAPACITY_KWH, TARGET_COLS, group_metrics, score_details


EXPERIMENT_ID = "ficr_consensus_corridor_sprint_v1"
PREREG = ROOT / "configs/ficr_consensus_corridor_sprint_preregister_v1.json"
PREREG_SHA = "af0f9d83ea2276712ff07eb32d5a589f948f76ac4cdaee0fec75d8632f1768e5"
LABELS = Path(r"data/local/open/train/train_labels.csv")
OUT = ROOT / "artifacts/final_submission_sprint_20260812/ficr_consensus_corridor"
ACTIVE = TARGET_COLS[:2]
COMPONENTS = (
    "lgb_l1",
    "lgb_q07",
    "shared_l1",
    "shared_q07",
    "top200_q07",
    "energy_q06",
)
PATHS = {
    2023: {
        "lgb_l1": ROOT / "artifacts/oof/dev2023_lgb_l1_eligible_n1500.parquet",
        "lgb_q07": ROOT / "artifacts/oof/dev2023_lgb_q07_eligible.parquet",
        "shared_l1": ROOT / "artifacts/oof/dev2023_shared_l1_eligible.parquet",
        "shared_q07": ROOT / "artifacts/oof/dev2023_shared_q07_eligible.parquet",
        "top200_q07": ROOT / "artifacts/oof/dev2023_lgb_top200_q07_eligible.parquet",
        "energy_q06": ROOT / "artifacts/oof/dev2023_lgb_q06_energywt_eligible.parquet",
        "baseline": ROOT / "artifacts/oof/dev2023_locked_v3.parquet",
    },
    2024: {
        "lgb_l1": ROOT / "artifacts/gate/v3/predictions/lgb_l1_gate.parquet",
        "lgb_q07": ROOT / "artifacts/gate/v3/predictions/lgb_q07_gate.parquet",
        "shared_l1": ROOT / "artifacts/oof/gate2024_shared_l1_cf.parquet",
        "shared_q07": ROOT / "artifacts/oof/gate2024_shared_q07_cf.parquet",
        "top200_q07": ROOT / "artifacts/gate/v3/predictions/top200_q07_gate.parquet",
        "energy_q06": ROOT / "artifacts/gate/v3/predictions/energy_q06_gate.parquet",
        "baseline": ROOT / "artifacts/oof/gate2024_locked_v3_cf_fix.parquet",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def describe(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def atomic_json(payload: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def year_index(year: int) -> pd.DatetimeIndex:
    return pd.date_range(
        f"{year}-01-01 01:00:00",
        f"{year + 1}-01-01 00:00:00",
        freq="h",
        name="forecast_kst_dtm",
    )


def load_prediction(path: Path, index: pd.DatetimeIndex, columns: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.read_parquet(path, engine="pyarrow")
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if not frame.index.equals(index) or not set(columns).issubset(frame.columns):
        raise ValueError(f"prediction contract mismatch: {path}")
    frame = frame.loc[:, list(columns)].astype(np.float64)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"non-finite prediction: {path}")
    return frame


def exact_utility_action(matrix: np.ndarray, capacity: float) -> np.ndarray:
    """Equal-scenario exact corridor action, independently for every row."""
    offsets = capacity * np.asarray((-0.08, -0.06, 0.0, 0.06, 0.08), dtype=np.float64)
    actions = np.clip(matrix[:, :, None] + offsets[None, None, :], 0.0, capacity).reshape(len(matrix), -1)
    scenario = matrix[:, None, :]
    errors = np.abs(actions[:, :, None] - scenario) / capacity
    payment = np.where(errors <= 0.06, 4.0, np.where(errors <= 0.08, 3.0, 0.0))
    utility = np.mean(-errors + 0.25 * (scenario / capacity) * payment, axis=2)
    best = np.max(utility, axis=1, keepdims=True)
    eligible = np.isclose(utility, best, rtol=0.0, atol=1e-14)
    median = np.median(matrix, axis=1)
    distance = np.where(eligible, np.abs(actions - median[:, None]), np.inf)
    best_distance = np.min(distance, axis=1, keepdims=True)
    eligible &= np.isclose(distance, best_distance, rtol=0.0, atol=1e-14)
    return np.min(np.where(eligible, actions, np.inf), axis=1)


def candidate_predictions(year: int) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, dict[str, float]]]:
    index = year_index(year)
    baseline_columns = TARGET_COLS if year == 2024 else ACTIVE
    baseline = load_prediction(PATHS[year]["baseline"], index, baseline_columns)
    sources = {name: load_prediction(PATHS[year][name], index, ACTIVE) for name in COMPONENTS}
    recipes = {
        "equal_scenario_exact_utility_w25": baseline.copy(),
        "interquartile_corridor_center_w25": baseline.copy(),
    }
    diagnostics: dict[str, dict[str, float]] = {}
    for group in ACTIVE:
        capacity = CAPACITY_KWH[group]
        matrix = np.column_stack([sources[name][group].to_numpy() for name in COMPONENTS])
        exact = exact_utility_action(matrix, capacity)
        q25, q75 = np.quantile(matrix, (0.25, 0.75), axis=1, method="linear")
        center = 0.5 * (q25 + q75)
        base = baseline[group].to_numpy()
        recipes["equal_scenario_exact_utility_w25"][group] = np.clip(0.75 * base + 0.25 * exact, 0.0, capacity)
        recipes["interquartile_corridor_center_w25"][group] = np.clip(0.75 * base + 0.25 * center, 0.0, capacity)
        diagnostics[group] = {
            "mean_component_spread_cf": float(np.mean(np.ptp(matrix, axis=1) / capacity)),
            "exact_mean_abs_move_cf": float(np.mean(np.abs(exact - base) / capacity)),
            "iqr_center_mean_abs_move_cf": float(np.mean(np.abs(center - base) / capacity)),
        }
    return baseline, recipes, diagnostics


def metric_dict(actual: pd.DataFrame, prediction: pd.DataFrame, groups: tuple[str, ...]) -> dict[str, Any]:
    return score_details(actual, prediction, target_cols=groups).as_dict()


def delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    return {key: float(candidate[key] - baseline[key]) for key in ("total_score", "one_minus_nmae", "ficr")}


def main() -> int:
    if sha256(PREREG) != PREREG_SHA:
        raise RuntimeError("preregister changed")
    config = json.loads(PREREG.read_text(encoding="utf-8"))
    if config["status"] != "frozen_before_loading_any_train_label_for_this_experiment":
        raise RuntimeError("preregister status changed")

    # Construct every application prediction before opening labels.
    predicted = {year: candidate_predictions(year) for year in (2023, 2024)}
    labels = pd.read_csv(LABELS, encoding="utf-8-sig", parse_dates=["kst_dtm"]).set_index("kst_dtm")
    labels.index = pd.DatetimeIndex(labels.index, name="forecast_kst_dtm")

    results: dict[str, Any] = {}
    for recipe in ("equal_scenario_exact_utility_w25", "interquartile_corridor_center_w25"):
        folds: dict[str, Any] = {}
        for year, fold_name in ((2023, "fit2022_apply2023"), (2024, "fit2022_2023_apply2024")):
            index = year_index(year)
            baseline, recipes, diagnostics = predicted[year]
            groups = ACTIVE if year == 2023 else TARGET_COLS
            actual = labels.loc[index, list(groups)].astype(np.float64)
            base_metric = metric_dict(actual, baseline, groups)
            candidate_metric = metric_dict(actual, recipes[recipe], groups)
            group_deltas: dict[str, Any] = {}
            for group in ACTIVE:
                base_group = group_metrics(actual[group], baseline[group], CAPACITY_KWH[group], group_name=group).as_dict()
                cand_group = group_metrics(actual[group], recipes[recipe][group], CAPACITY_KWH[group], group_name=group).as_dict()
                group_deltas[group] = {
                    key: float(cand_group[key] - base_group[key])
                    for key in ("one_minus_nmae", "ficr")
                }
                group_deltas[group]["total_score"] = 0.5 * (
                    group_deltas[group]["one_minus_nmae"] + group_deltas[group]["ficr"]
                )
            folds[fold_name] = {
                "year": year,
                "groups_scored": list(groups),
                "baseline": base_metric,
                "candidate": candidate_metric,
                "delta": delta(candidate_metric, base_metric),
                "active_group_deltas": group_deltas,
                "prediction_diagnostics": diagnostics,
            }
        mean_score = float(np.mean([fold["delta"]["total_score"] for fold in folds.values()]))
        mean_ficr = float(np.mean([fold["delta"]["ficr"] for fold in folds.values()]))
        rules = config["promotion_rules"]
        checks = {
            "mean_score": mean_score >= float(rules["minimum_mean_full_fold_total_score_delta"]),
            "mean_ficr": mean_ficr >= float(rules["minimum_mean_full_fold_ficr_delta"]),
            "each_fold_score": all(fold["delta"]["total_score"] >= float(rules["minimum_each_fold_total_score_delta"]) for fold in folds.values()),
            "each_fold_ficr": all(fold["delta"]["ficr"] >= float(rules["minimum_each_fold_ficr_delta"]) for fold in folds.values()),
            "each_active_group_score": all(group["total_score"] >= float(rules["minimum_each_active_group_full_total_score_delta"]) for fold in folds.values() for group in fold["active_group_deltas"].values()),
        }
        results[recipe] = {
            "folds": folds,
            "mean_delta": {"total_score": mean_score, "ficr": mean_ficr},
            "checks": checks,
            "pass": all(checks.values()),
        }

    passing = [name for name, result in results.items() if result["pass"]]
    selected = max(passing, key=lambda name: results[name]["mean_delta"]["total_score"]) if passing else None
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "decision": "PASS" if selected else "NO-GO",
        "selected_recipe": selected,
        "csv_created": False,
        "reason_if_no_go": None if selected else "No preregistered recipe passed every mean, fold-sign, and active-group stability threshold.",
        "preregister": describe(PREREG),
        "labels": describe(LABELS),
        "inputs": {str(year): {name: describe(path) for name, path in paths.items()} for year, paths in PATHS.items()},
        "results": results,
        "forbidden_inputs": {"public_feedback": False, "2025_actual": False, "public_inverse": False, "2025_predictions_read": False},
    }
    atomic_json(manifest, OUT / "MANIFEST.json")
    print(json.dumps({"decision": manifest["decision"], "selected_recipe": selected, "means": {name: value["mean_delta"] for name, value in results.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

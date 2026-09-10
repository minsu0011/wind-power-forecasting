"""Run the frozen KMA D-1 11 KST WSD two-recipe forward model gate.

This runner deliberately separates candidate production from scoring.  It first
verifies all bound identities, fits both recipes, and writes every model,
baseline, and candidate prediction create-exclusively.  A fold freeze is written
before labels from that fold's application year may later serve as training data
in a subsequent fold.  Only after a global freeze exists are metrics computed.
It never reads KMA 2025 data or creates a submission CSV.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metric import CAPACITY_KWH, group_metrics  # noqa: E402
from src.weather_quantile import (  # noqa: E402
    BayesActionConfig,
    TrainOnlyMedianTransform,
    WeatherQuantileSurface,
)


GROUPS = ("kpx_group_1", "kpx_group_2", "kpx_group_3")
FOLDS: dict[str, dict[str, Any]] = {
    "fit_operating_2022_apply_operating_2023": {
        "train_years": (2022,), "apply_year": 2023, "groups": GROUPS[:2]
    },
    "fit_operating_2022_2023_apply_operating_2024": {
        "train_years": (2022, 2023), "apply_year": 2024, "groups": GROUPS
    },
}
RECIPES = ("R1_KMA_DIRECT_QUANTILE_BAYES", "R2_KMA_DISAGREEMENT_RESIDUAL_Q60")
MATERIALIZED = ROOT / "artifacts/final_submission_sprint_20260812/kma_d1_1100/wsd_typ01/materialized/KMA_D1_1100_WSD_HOURLY_2022_2024.parquet"
MATERIALIZED_MANIFEST = MATERIALIZED.parent / "MANIFEST.json"
DEFAULT_OUTPUT = ROOT / "artifacts/final_submission_sprint_20260812/kma_d1_1100/model_gate"
LABELS = Path(r"data/local/open/train/train_labels.csv")
PREREGS = (
    (ROOT / "configs/kma_d1_1100_wsd_sprint_preregister_v1.json", 4312, "2353e2b39ef6d64645393af749fbf7ea13e3e89f1b6e4f35698b02b59efe389f"),
    (ROOT / "configs/kma_d1_1100_wsd_sprint_preregister_v2_amendment.json", 21034, "4e7d99b1d7ea67fa724d0956300a131d3b00a7b34cb777d6ce68eef57a63c99d"),
    (ROOT / "configs/kma_d1_1100_wsd_sprint_preregister_v3_decoder_errata.json", 3606, "747cca0d49d28c572c35ad66c8a397e9e5467af653d3df2f61189659eed80ea7"),
    (ROOT / "configs/kma_d1_1100_wsd_sprint_preregister_v4_anchor_amendment.json", 8943, "0fe5f310c727e31d2fbbd786ef55ebb02990b4903ba18d5d91dbfe0c31b7fa4d"),
    (ROOT / "configs/kma_d1_1100_wsd_sprint_preregister_v5_ordering_errata.json", 1391, "140d78ace4fe675d597fb2c2a8feca6b70fb0ca681b317a21d4f0c62912b0b60"),
)
IDENTITIES: dict[str, tuple[Path, int, str]] = {
    "labels": (LABELS, 1138967, "47bb64252195cf4734e67394d6e50485f27a608def3b5a8791fcc7674bbceb03"),
    "features": (ROOT / "src/features.py", 34098, "34c7bf46444c9a2a88eabdc23f41254f66a9ba04eda1adfa53e19b1263a81a5e"),
    "weather_quantile": (ROOT / "src/weather_quantile.py", 13206, "005fcc8069bc707b5d6c270bb640897696e2eb04e050a5ad81a251433041075d"),
    "metric": (ROOT / "src/metric.py", 7248, "555950e6892a808d9091b4e6749128b9f96888ce0a83270b6959a4e927dc5f1d"),
    "baseline_2022": (ROOT / "artifacts/postgate/common_cf_forward/oof/dev2022_selected_inner_oof.parquet", 172215, "ce781f49f07146a982891ecc156b389aae3c7c4b38e4c0a35031649e55bd81d2"),
    "baseline_2023": (ROOT / "artifacts/postgate/weather_quantile_bayes/oof/stage1_baseline_2023.parquet", 292085, "040d4fe663cecde856c4c2229d3d98b472954cc2c2c541c787cfbc3af85aa9b6"),
    "baseline_2024": (ROOT / "artifacts/postgate/public_adaptive_scale097_g2_delta_v2/diagnostic_2024/A_scale097.parquet", 335807, "93d8e2b7d19971ba46a29d548e77ef4f85553bbbdaad2b5983a5b6f1d84e406c"),
}
WEATHER: dict[str, tuple[Path, int, str]] = {
    "kpx_group_1": (ROOT / "artifacts/cache/kpx_group_1_weather_train.parquet", 77942069, "c3526f861184a16fef4a68c20ad8f4defab04b7dc0650b571c6beba05a867579"),
    "kpx_group_2": (ROOT / "artifacts/cache/kpx_group_2_weather_train.parquet", 77929519, "0e6fc7334e7094af1a2fffceffeee7628930aaa102c43515fde3315ec806a31e"),
    "kpx_group_3": (ROOT / "artifacts/cache/kpx_group_3_weather_train.parquet", 77950424, "eb61868a0fd564a60f31fdd9b1fd6324545a168fda9b0137049754d014b332ce"),
}
R1_FEATURES = (
    "wsd_cell_94_121", "wsd_cell_94_122", "wsd_group_weighted",
    "wsd_cell_122_minus_121", "lead_sin_24h", "lead_cos_24h",
)
R2_FEATURES = R1_FEATURES[:4] + (
    "wsd_group_minus_official_ws10", "wsd_group_to_official_ws10_ratio_clipped",
    "lead_sin_24h", "lead_cos_24h", "baseline_cf",
)
R1_PARAMS = {
    "n_estimators": 700, "learning_rate": 0.03, "num_leaves": 24,
    "max_depth": 6, "min_child_samples": 120, "subsample": 0.85,
    "subsample_freq": 1, "colsample_bytree": 0.8, "reg_alpha": 0.0,
    "reg_lambda": 2.0, "verbosity": -1, "deterministic": True,
    "force_col_wise": True, "bagging_seed": 20260812,
    "feature_fraction_seed": 20260812, "data_random_seed": 20260812,
}
R2_PARAMS = {
    "objective": "quantile", "alpha": 0.6, "n_estimators": 500,
    "learning_rate": 0.035, "num_leaves": 20, "max_depth": 5,
    "min_child_samples": 150, "subsample": 0.85, "subsample_freq": 1,
    "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 2.0,
    "verbosity": -1, "deterministic": True, "force_col_wise": True,
    "random_state": 20260813, "bagging_seed": 20260813,
    "feature_fraction_seed": 20260813, "data_random_seed": 20260813,
    "n_jobs": 10, "device_type": "cpu",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    return {"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def exclusive_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    if path.read_bytes() != payload:
        raise RuntimeError(f"write verification failed: {path.name}")


def exclusive_json(path: Path, payload: Any) -> None:
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n").encode("utf-8")
    exclusive_bytes(path, raw)


def exclusive_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    frame.to_parquet(temporary, engine="pyarrow", compression="zstd", index=True)
    with temporary.open("r+b") as handle:
        os.fsync(handle.fileno())
    os.link(temporary, path)
    temporary.unlink()
    reopened = pd.read_parquet(path, engine="pyarrow")
    if not reopened.index.equals(frame.index) or list(reopened.columns) != list(frame.columns):
        raise RuntimeError(f"parquet readback schema mismatch: {path.name}")
    if not np.array_equal(reopened.to_numpy(), frame.to_numpy(), equal_nan=True):
        raise RuntimeError(f"parquet readback value mismatch: {path.name}")


def verify_bound(path: Path, size: int, digest: str, name: str) -> dict[str, Any]:
    observed = identity(path)
    if observed["bytes"] != size or observed["sha256"] != digest:
        raise RuntimeError(f"bound identity drift: {name}")
    return observed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--materialized", type=Path, default=MATERIALIZED)
    parser.add_argument("--materialized-manifest", type=Path, default=MATERIALIZED_MANIFEST)
    parser.add_argument("--synthetic-smoke", action="store_true")
    return parser.parse_args()


def expected_year_index(year: int) -> pd.DatetimeIndex:
    return pd.date_range(f"{year}-01-01 01:00", f"{year + 1}-01-01 00:00", freq="h", name="forecast_kst_dtm")


def load_materialized(path: Path, manifest_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    file_id = identity(path)
    serialized = json.dumps(manifest, sort_keys=True)
    if file_id["sha256"] not in serialized or str(file_id["bytes"]) not in serialized:
        raise RuntimeError("materializer manifest does not bind hourly parquet bytes/SHA")
    frame = pd.read_parquet(path, engine="pyarrow")
    if "forecast_kst_dtm" in frame.columns:
        frame = frame.set_index("forecast_kst_dtm", verify_integrity=True)
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    required = {
        "operating_year", "operating_day", "lead_hour", "issue_base_kst",
        "wsd_cell_94_121", "wsd_cell_94_122",
        "wsd_kpx_group_1", "wsd_kpx_group_2", "wsd_kpx_group_3",
    }
    if not required.issubset(frame.columns) or not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise RuntimeError("materialized KMA schema/index mismatch")
    expected = expected_year_index(2022).append(expected_year_index(2023)).append(expected_year_index(2024))
    if not frame.index.equals(expected):
        raise RuntimeError("materialized KMA does not have exact 2022-2024 hourly identities")
    clock = frame.index - pd.Timedelta(hours=1)
    if not np.array_equal(frame["operating_year"].to_numpy(dtype=int), clock.year):
        raise RuntimeError("operating_year mismatch")
    leads = clock.hour.to_numpy() + 1
    if not np.array_equal(frame["lead_hour"].to_numpy(dtype=int), leads):
        raise RuntimeError("lead_hour mismatch")
    issue = pd.to_datetime(frame["issue_base_kst"], errors="raise")
    expected_issue = pd.DatetimeIndex(clock.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=11))
    if not pd.DatetimeIndex(issue).equals(expected_issue):
        raise RuntimeError("issue_base_kst mismatch")
    a = frame["wsd_cell_94_121"].to_numpy(dtype=float)
    b = frame["wsd_cell_94_122"].to_numpy(dtype=float)
    for year in (2022, 2023, 2024):
        ym = clock.year == year
        for values in (a, b):
            if np.isfinite(values[ym]).mean() < 0.995:
                raise RuntimeError(f"annual WSD coverage below 99.5%: {year}")
            for lead in range(1, 25):
                mask = ym & (leads == lead)
                if np.isfinite(values[mask]).mean() < 0.995:
                    raise RuntimeError(f"lead WSD coverage below 99.5%: {year}/{lead}")
    expected_groups = {GROUPS[0]: 0.5 * a + 0.5 * b, GROUPS[1]: (5.0 / 6.0) * a + (1.0 / 6.0) * b, GROUPS[2]: a}
    for group, values in expected_groups.items():
        observed = frame[f"wsd_{group}"].to_numpy(dtype=float)
        if not np.allclose(observed, values, rtol=0.0, atol=1e-12, equal_nan=True):
            raise RuntimeError(f"materialized group WSD formula mismatch: {group}")
    return frame, {"hourly": file_id, "manifest": identity(manifest_path)}


def load_baselines() -> dict[int, pd.DataFrame]:
    raw22 = pd.read_parquet(IDENTITIES["baseline_2022"][0], engine="pyarrow")
    raw23 = pd.read_parquet(IDENTITIES["baseline_2023"][0], engine="pyarrow")
    raw24 = pd.read_parquet(IDENTITIES["baseline_2024"][0], engine="pyarrow")
    raw22 = raw22.astype(float).mul(0.97).clip(
        lower=0.0,
        upper=pd.Series({g: 1.02 * CAPACITY_KWH[g] for g in raw22.columns}),
        axis="columns",
    )
    raw23 = raw23.astype(float).mul(0.97).clip(
        lower=0.0,
        upper=pd.Series({g: 1.02 * CAPACITY_KWH[g] for g in raw23.columns}),
        axis="columns",
    )
    for year, frame in ((2023, raw23), (2024, raw24)):
        if not frame.index.equals(expected_year_index(year)) or tuple(frame.columns) != GROUPS:
            raise RuntimeError(f"baseline {year} identity/schema mismatch")
    if not raw22.index.equals(pd.date_range("2022-04-01 01:00", "2023-01-01 00:00", freq="h", name="forecast_kst_dtm")):
        raise RuntimeError("baseline 2022 index mismatch")
    return {2022: raw22, 2023: raw23, 2024: raw24.astype(float)}


def load_official_ws10() -> dict[str, pd.Series]:
    result: dict[str, pd.Series] = {}
    for group, (path, _, _) in WEATHER.items():
        frame = pd.read_parquet(path, columns=["ldaps__idw__ws10"], engine="pyarrow")
        if not frame.index.equals(expected_year_index(2022).append(expected_year_index(2023)).append(expected_year_index(2024))):
            raise RuntimeError(f"official WS10 index mismatch: {group}")
        values = frame["ldaps__idw__ws10"].astype(float)
        if not np.isfinite(values).all():
            raise RuntimeError(f"official WS10 contains missing values: {group}")
        result[group] = values
    return result


def feature_frame(kma: pd.DataFrame, group: str, ws10: pd.Series, baseline: pd.Series | None) -> pd.DataFrame:
    a = kma["wsd_cell_94_121"].astype(float)
    b = kma["wsd_cell_94_122"].astype(float)
    weighted = kma[f"wsd_{group}"].astype(float)
    lead = kma["lead_hour"].to_numpy(dtype=float)
    frame = pd.DataFrame(index=kma.index)
    frame["wsd_cell_94_121"] = a
    frame["wsd_cell_94_122"] = b
    frame["wsd_group_weighted"] = weighted
    frame["wsd_cell_122_minus_121"] = b - a
    frame["wsd_group_minus_official_ws10"] = weighted - ws10
    frame["wsd_group_to_official_ws10_ratio_clipped"] = np.clip(weighted / np.maximum(ws10, 0.5), 0.0, 4.0)
    frame["lead_sin_24h"] = np.sin(2.0 * np.pi * (lead - 1.0) / 24.0)
    frame["lead_cos_24h"] = np.cos(2.0 * np.pi * (lead - 1.0) / 24.0)
    frame["baseline_cf"] = np.nan if baseline is None else baseline / CAPACITY_KWH[group]
    return frame


def serialize_model(path: Path, payload: Any) -> dict[str, Any]:
    exclusive_bytes(path, pickle.dumps(payload, protocol=5))
    return identity(path)


def fit_all_without_validation_labels(kma: pd.DataFrame, out: Path) -> dict[str, Any]:
    # Deliberately read only training-year label prefixes during each fit.  No
    # application-year label value is opened until all prediction files lock.
    ws10 = load_official_ws10()
    baselines = load_baselines()
    prediction_ids: dict[str, Any] = {}
    baseline_prediction_ids: dict[str, Any] = {}
    model_ids: dict[str, Any] = {}
    fold_freeze_ids: dict[str, Any] = {}
    for fold_name, fold in FOLDS.items():
        train_indexes = [expected_year_index(y) for y in fold["train_years"]]
        train_index = train_indexes[0]
        for extra in train_indexes[1:]:
            train_index = train_index.append(extra)
        apply_index = expected_year_index(fold["apply_year"])
        max_fit_row = len(expected_year_index(2022)) + (len(expected_year_index(2023)) if 2023 in fold["train_years"] else 0)
        # nrows restricts decoding to this fold's registered training years.
        labels = pd.read_csv(LABELS, usecols=["kst_dtm", *fold["groups"]], nrows=max_fit_row, parse_dates=["kst_dtm"]).set_index("kst_dtm")
        labels.index.name = "forecast_kst_dtm"
        if not labels.index.equals(train_index):
            raise RuntimeError(f"fit label prefix mismatch: {fold_name}")
        candidates = {recipe: pd.DataFrame(index=apply_index, columns=fold["groups"], dtype=float) for recipe in RECIPES}
        fold_baseline = baselines[fold["apply_year"]].loc[apply_index, list(fold["groups"])].astype(float)
        baseline_path = out / "predictions" / f"{fold_name}__BASELINE.parquet"
        exclusive_parquet(baseline_path, fold_baseline)
        baseline_prediction_ids[fold_name] = identity(baseline_path)
        fold_models: dict[str, dict[str, Any]] = {recipe: {} for recipe in RECIPES}
        for group in fold["groups"]:
            capacity = CAPACITY_KWH[group]
            apply_base = baselines[fold["apply_year"]][group]
            if not np.isfinite(apply_base).all():
                raise RuntimeError(f"application baseline missing: {fold_name}/{group}")
            r1_all = feature_frame(kma, group, ws10[group], None)
            r1_x = r1_all.loc[train_index, R1_FEATURES]
            r1_surface = WeatherQuantileSurface(
                quantile_levels=(0.1, 0.25, 0.5, 0.75, 0.9), model_parameters=R1_PARAMS,
                random_state=20260812, n_jobs=10, minimum_actual_cf=0.10,
                action_config=BayesActionConfig(
                    quantile_levels=(0.1, 0.25, 0.5, 0.75, 0.9), interpolation_count=33,
                    outcome_lower_cf=0.1, outcome_upper_cf=1.2,
                    candidate_lower_cf=0.0, candidate_upper_cf=1.02, candidate_step_cf=0.01,
                ),
            ).fit(r1_x, labels[group], capacity_kwh=capacity)
            action, quantiles = r1_surface.predict_action(r1_all.loc[apply_index, R1_FEATURES], apply_base)
            candidates[RECIPES[0]][group] = np.clip(0.75 * apply_base + 0.25 * action, 0.0, 1.02 * capacity)
            fold_models[RECIPES[0]][group] = {"surface": r1_surface, "metadata": r1_surface.metadata(), "quantiles": quantiles}

            baseline_parts = [baselines[y][group] for y in fold["train_years"] if group in baselines[y].columns]
            train_base = pd.concat(baseline_parts).reindex(train_index)
            r2_all = feature_frame(kma, group, ws10[group], pd.concat([train_base, apply_base]))
            actual = labels[group].reindex(train_index).astype(float)
            kma_train_finite = np.isfinite(r2_all.loc[train_index, R2_FEATURES[:-1]].to_numpy(dtype=float)).all(axis=1)
            eligible = np.isfinite(actual) & (actual >= 0.10 * capacity) & np.isfinite(train_base) & kma_train_finite
            r2_train = r2_all.loc[train_index[eligible], R2_FEATURES]
            if len(r2_train) < 300:
                raise RuntimeError(f"too few R2 training rows: {fold_name}/{group}")
            transform = TrainOnlyMedianTransform().fit(r2_train)
            from lightgbm import LGBMRegressor
            r2_model = LGBMRegressor(**R2_PARAMS)
            target = actual.loc[r2_train.index].to_numpy(dtype=float) / capacity - train_base.loc[r2_train.index].to_numpy(dtype=float) / capacity
            r2_model.fit(transform.transform(r2_train), target)
            raw = r2_model.predict(transform.transform(r2_all.loc[apply_index, R2_FEATURES]))
            correction = 0.10 * np.clip(np.asarray(raw, dtype=float), -0.20, 0.20) * capacity
            candidates[RECIPES[1]][group] = np.clip(apply_base.to_numpy(dtype=float) + correction, 0.0, 1.02 * capacity)
            fold_models[RECIPES[1]][group] = {"model": r2_model, "transform": transform, "fit_rows": len(r2_train)}
        for recipe in RECIPES:
            if not np.isfinite(candidates[recipe].to_numpy(dtype=float)).all():
                raise RuntimeError(f"candidate contains non-finite values: {fold_name}/{recipe}")
            pred_path = out / "predictions" / f"{fold_name}__{recipe}.parquet"
            exclusive_parquet(pred_path, candidates[recipe])
            prediction_ids[f"{fold_name}/{recipe}"] = identity(pred_path)
            model_path = out / "models" / f"{fold_name}__{recipe}.pkl"
            model_ids[f"{fold_name}/{recipe}"] = serialize_model(model_path, fold_models[recipe])
        fold_freeze = {
            "schema_version": 1,
            "fold": fold_name,
            "training_operating_years_decoded": list(fold["train_years"]),
            "application_operating_year": fold["apply_year"],
            "baseline": baseline_prediction_ids[fold_name],
            "predictions": {
                recipe: prediction_ids[f"{fold_name}/{recipe}"] for recipe in RECIPES
            },
            "models": {
                recipe: model_ids[f"{fold_name}/{recipe}"] for recipe in RECIPES
            },
            "metrics_computed": False,
        }
        fold_freeze_path = out / "fold_locks" / f"{fold_name}__FREEZE.json"
        exclusive_json(fold_freeze_path, fold_freeze)
        fold_freeze_ids[fold_name] = identity(fold_freeze_path)
    lock = {
        "schema_version": 1,
        "status": "ALL_RECIPE_FOLD_GROUP_PREDICTIONS_AND_BASELINES_FROZEN_BEFORE_ANY_METRIC_COMPUTATION",
        "predictions": prediction_ids,
        "baseline_predictions": baseline_prediction_ids,
        "models": model_ids,
        "fold_freezes": fold_freeze_ids,
        "label_access_before_lock": {
            "training_year_prefixes_only_per_fold": True,
            "fold_1_training_operating_years": [2022],
            "fold_2_training_operating_years": [2022, 2023],
            "application_year_metrics_computed_before_lock": False,
            "application_labels_may_be_reused_as_later_fold_training_only_after_their_fold_freeze": True,
            "full_2022_2024_label_frame_opened_before_global_lock": False
        },
        "public_scores_read_or_used": False,
        "kma_2025_read": False,
        "csv_created": False,
    }
    exclusive_json(out / "PREDICTION_FREEZE_BEFORE_METRICS.json", lock)
    return lock


def aggregate_metric(labels: pd.DataFrame, predictions: pd.DataFrame, groups: Iterable[str], mask: np.ndarray) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for group in groups:
        metric = group_metrics(labels.loc[mask, group], predictions.loc[mask, group], CAPACITY_KWH[group], group_name=group)
        records[group] = asdict(metric)
    one_minus = float(np.mean([v["one_minus_nmae"] for v in records.values()]))
    ficr = float(np.mean([v["ficr"] for v in records.values()]))
    return {"total_score": 0.5 * (one_minus + ficr), "one_minus_nmae": one_minus, "ficr": ficr, "groups": records}


def score_slice(labels: pd.DataFrame, candidate: pd.DataFrame, baseline: pd.DataFrame, groups: tuple[str, ...], mask: np.ndarray) -> dict[str, Any]:
    base = aggregate_metric(labels, baseline, groups, mask)
    cand = aggregate_metric(labels, candidate, groups, mask)
    return {"baseline": base, "candidate": cand, "delta_score": cand["total_score"] - base["total_score"], "delta_ficr": cand["ficr"] - base["ficr"], "delta_one_minus_nmae": cand["one_minus_nmae"] - base["one_minus_nmae"]}


def evaluate_recipe(recipe: str, out: Path, labels: pd.DataFrame, lock: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"recipe": recipe, "folds": {}}
    for fold_name, fold in FOLDS.items():
        candidate_path = out / "predictions" / f"{fold_name}__{recipe}.parquet"
        baseline_path = out / "predictions" / f"{fold_name}__BASELINE.parquet"
        if identity(candidate_path) != lock["predictions"][f"{fold_name}/{recipe}"]:
            raise RuntimeError("frozen candidate identity drift")
        if identity(baseline_path) != lock["baseline_predictions"][fold_name]:
            raise RuntimeError("frozen baseline identity drift")
        candidate = pd.read_parquet(candidate_path, engine="pyarrow")
        baseline = pd.read_parquet(baseline_path, engine="pyarrow")
        index = expected_year_index(fold["apply_year"])
        if not candidate.index.equals(index):
            raise RuntimeError("frozen prediction index drift")
        groups = fold["groups"]
        actual = labels.loc[index, list(groups)]
        if not baseline.index.equals(index) or tuple(baseline.columns) != tuple(groups):
            raise RuntimeError("frozen baseline schema/index drift")
        clock = index - pd.Timedelta(hours=1)
        masks: dict[str, np.ndarray] = {"full": np.ones(len(index), dtype=bool)}
        masks.update({"H1": clock.month <= 6, "H2": clock.month >= 7})
        masks.update({f"Q{q}": ((clock.month - 1) // 3 + 1) == q for q in range(1, 5)})
        masks.update({f"M{month:02d}": clock.month == month for month in range(1, 13)})
        masks["late_leads_13_24"] = (clock.hour + 1) >= 13
        slices = {name: score_slice(actual, candidate, baseline, groups, np.asarray(mask)) for name, mask in masks.items()}
        group_full: dict[str, Any] = {}
        for group in groups:
            gm = score_slice(actual[[group]], candidate[[group]], baseline[[group]], (group,), np.ones(len(index), dtype=bool))
            group_full[group] = gm
        result["folds"][fold_name] = {"apply_operating_year": fold["apply_year"], "groups": list(groups), "slices": slices, "group_full": group_full}
    folds = list(result["folds"].values())
    full = [f["slices"]["full"] for f in folds]
    quarters = [f["slices"][f"Q{q}"] for f in folds for q in range(1, 5)]
    months = [f["slices"][f"M{m:02d}"] for f in folds for m in range(1, 13)]
    halves = [f["slices"][half] for f in folds for half in ("H1", "H2")]
    groups_full = [record for f in folds for record in f["group_full"].values()]
    late = [f["slices"]["late_leads_13_24"] for f in folds]
    checks = {
        "each_fold_delta_score_positive": all(x["delta_score"] > 0.0 for x in full),
        "each_fold_delta_ficr_positive": all(x["delta_ficr"] > 0.0 for x in full),
        "mean_fold_delta_score_min_0_002": float(np.mean([x["delta_score"] for x in full])) >= 0.002,
        "mean_fold_delta_ficr_min_0_003": float(np.mean([x["delta_ficr"] for x in full])) >= 0.003,
        "each_group_delta_score_min_neg0_0005": all(x["delta_score"] >= -0.0005 for x in groups_full),
        "each_group_delta_ficr_min_neg0_0005": all(x["delta_ficr"] >= -0.0005 for x in groups_full),
        "each_half_delta_score_nonnegative": all(x["delta_score"] >= 0.0 for x in halves),
        "worst_quarter_delta_score_min_neg0_0015": min(x["delta_score"] for x in quarters) >= -0.0015,
        "worst_month_delta_score_min_neg0_003": min(x["delta_score"] for x in months) >= -0.003,
        "quarter_nonnegative_fraction_min_0_75": np.mean([x["delta_score"] >= 0.0 for x in quarters]) >= 0.75,
        "month_nonnegative_fraction_min_0_67": np.mean([x["delta_score"] >= 0.0 for x in months]) >= 0.67,
        "late_each_fold_delta_score_nonnegative": all(x["delta_score"] >= 0.0 for x in late),
        "late_each_fold_delta_ficr_nonnegative": all(x["delta_ficr"] >= 0.0 for x in late),
    }
    result["gate_summary"] = {
        "checks": checks, "all_gates_pass": all(checks.values()),
        "minimum_fold_delta_score": min(x["delta_score"] for x in full),
        "mean_fold_delta_score": float(np.mean([x["delta_score"] for x in full])),
        "mean_fold_delta_ficr": float(np.mean([x["delta_ficr"] for x in full])),
        "worst_quarter_delta_score": min(x["delta_score"] for x in quarters),
        "worst_month_delta_score": min(x["delta_score"] for x in months),
        "quarter_nonnegative_fraction": float(np.mean([x["delta_score"] >= 0.0 for x in quarters])),
        "month_nonnegative_fraction": float(np.mean([x["delta_score"] >= 0.0 for x in months])),
    }
    return result


def synthetic_smoke() -> None:
    index = expected_year_index(2023)[:48]
    labels = pd.DataFrame({g: np.linspace(3000.0, 12000.0, len(index)) for g in GROUPS[:2]}, index=index)
    baseline = labels * 0.98
    candidate = labels * 0.99
    mask = ((index - pd.Timedelta(hours=1)).hour + 1) >= 13
    record = score_slice(labels, candidate, baseline, GROUPS[:2], mask)
    if not record["delta_score"] > 0.0 or not math.isfinite(record["delta_ficr"]):
        raise RuntimeError("synthetic metric smoke failed")
    print("synthetic_smoke=PASS")


def main() -> None:
    args = parse_args()
    if args.synthetic_smoke:
        synthetic_smoke()
        return
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"create-exclusive output already exists: {out}")
    out.mkdir(parents=True, exist_ok=False)
    bound: dict[str, Any] = {}
    for i, (path, size, digest) in enumerate(PREREGS, start=1):
        bound[f"preregister_v{i}"] = verify_bound(path, size, digest, f"preregister_v{i}")
    for name, (path, size, digest) in IDENTITIES.items():
        bound[name] = verify_bound(path, size, digest, name)
    for group, (path, size, digest) in WEATHER.items():
        bound[f"weather_{group}"] = verify_bound(path, size, digest, f"weather_{group}")
    kma, materialized_ids = load_materialized(args.materialized.resolve(), args.materialized_manifest.resolve())
    bound["materialized"] = materialized_ids
    exclusive_json(out / "INPUT_LOCK_BEFORE_MODEL_FIT.json", {"bound": bound, "public_scores_used": False, "kma_2025_read": False})
    lock = fit_all_without_validation_labels(kma, out)
    # From this line onward scoring labels may be opened, but only after lock identity.
    if verify_bound(*IDENTITIES["labels"], "labels") != bound["labels"]:
        raise RuntimeError("label identity drift after prediction freeze")
    labels = pd.read_csv(LABELS, parse_dates=["kst_dtm"]).set_index("kst_dtm")
    labels.index.name = "forecast_kst_dtm"
    expected = expected_year_index(2022).append(expected_year_index(2023)).append(expected_year_index(2024))
    if not labels.index.equals(expected) or tuple(labels.columns) != GROUPS:
        raise RuntimeError("official label schema/index mismatch")
    evaluations = {recipe: evaluate_recipe(recipe, out, labels, lock) for recipe in RECIPES}
    passing = [recipe for recipe in RECIPES if evaluations[recipe]["gate_summary"]["all_gates_pass"]]
    selected: str | None = None
    if passing:
        selected = max(passing, key=lambda recipe: (evaluations[recipe]["gate_summary"]["minimum_fold_delta_score"], evaluations[recipe]["gate_summary"]["mean_fold_delta_score"], recipe == RECIPES[0]))
    report = {
        "schema_version": 1, "prediction_lock": identity(out / "PREDICTION_FREEZE_BEFORE_METRICS.json"),
        "evaluations": evaluations, "passing_recipes": passing, "selected_recipe": selected,
        "all_gates_pass": selected is not None, "public_scores_used": False,
        "kma_2025_read": False, "csv_created": False,
    }
    exclusive_json(out / "MODEL_GATE_RESULTS.json", report)
    if selected is not None:
        gate = {"all_gates_pass": True, "selected_recipe": selected, "results": identity(out / "MODEL_GATE_RESULTS.json")}
        exclusive_json(args.materialized.resolve().parents[1] / "MODEL_GATE_PASS.json", gate)
    else:
        exclusive_json(out / "NO_PROMOTION.json", {"all_gates_pass": False, "reason": "no frozen recipe passed every preregistered gate", "kma_2025_access_authorized": False, "csv_authorized": False})
    print(json.dumps({"all_gates_pass": selected is not None, "selected_recipe": selected}, sort_keys=True))


if __name__ == "__main__":
    main()

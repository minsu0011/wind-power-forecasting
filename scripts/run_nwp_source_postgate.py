"""Strict NWP-source ablation, pre-2024 lock, and fixed confirmation.

The program has two deliberately ordered stages.  Stage 1 reads only the first
17,520 rows (through the closing hour of 2023), compares LDAPS-only, GFS-only,
and all-weather 400-tree models, evaluates three predeclared source ensembles,
and atomically writes the selected recipe lock.  Only then does Stage 2 open
2024 features/labels.  A recipe stable in both 2023 and 2024 halves is promoted
to a fixed 1,500-tree full fit and a 2025 prediction Parquet.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lightgbm import LGBMRegressor, log_evaluation


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.manifest import make_manifest, sha256_file, write_json_atomic  # noqa: E402
from src.metric import CAPACITY_KWH, TARGET_COLS, group_metrics  # noqa: E402


DEV_ROWS = 17_520
TRAIN_ROWS = 26_304
TEST_ROWS = 8_760
DEV_END = pd.Timestamp("2024-01-01 00:00:00")
OOF_START = pd.Timestamp("2024-01-01 01:00:00")
OOF_END = pd.Timestamp("2025-01-01 00:00:00")
G12_DEV_START = pd.Timestamp("2023-01-01 01:00:00")
G12_DEV_HALF = pd.Timestamp("2023-07-01 01:00:00")
G3_START = pd.Timestamp("2023-01-01 01:00:00")
G3_DEV_START = pd.Timestamp("2023-07-01 01:00:00")
G3_DEV_HALF = pd.Timestamp("2023-10-01 01:00:00")
OOF_HALF = pd.Timestamp("2024-07-01 01:00:00")
OBJECTIVES = ("l1", "q07")
SOURCES = ("ldaps", "gfs", "all")
ENSEMBLES: dict[str, dict[str, float]] = {
    "ldaps_gfs_equal": {"ldaps": 0.50, "gfs": 0.50, "all": 0.00},
    "tri_balanced": {"ldaps": 0.25, "gfs": 0.25, "all": 0.50},
    "tri_conservative": {"ldaps": 0.10, "gfs": 0.10, "all": 0.80},
}
SPATIAL_PROMOTED = {
    ("kpx_group_2", "q07"): "core",
    ("kpx_group_3", "l1"): "sequence",
    ("kpx_group_3", "q07"): "sequence",
}
FORBIDDEN = re.compile(r"(^kpx_group_|scada|target|label|power_kw)", re.I)
STABILITY_FLOOR = -0.001


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run strict LDAPS/GFS/all source ensembles and locked postgate."
    )
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--base-cache-dir", type=Path, required=True)
    parser.add_argument("--augmentation-dir", type=Path, required=True)
    parser.add_argument(
        "--spatial-postgate-results",
        type=Path,
        default=(
            PROJECT_DIR
            / "artifacts"
            / "experiments"
            / "spatial_v2_postgate"
            / "spatial_v2_postgate_results.json"
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _model_config(n_estimators: int) -> dict[str, Any]:
    return {
        "n_estimators": n_estimators,
        "learning_rate": 0.04 if n_estimators == 400 else 0.025,
        "num_leaves": 31,
        "min_child_samples": 30,
        "subsample": 0.80,
        "subsample_freq": 1,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.05,
        "reg_lambda": 2.0,
        "seed": 42,
        "eligible_fraction": 0.10,
        "target_unit": "capacity_factor",
        "early_stopping": False,
    }


def _make_model(config: dict[str, Any], objective: str, n_jobs: int) -> LGBMRegressor:
    objective_args = (
        {"objective": "regression_l1"}
        if objective == "l1"
        else {"objective": "quantile", "alpha": 0.70}
    )
    return LGBMRegressor(
        **objective_args,
        n_estimators=config["n_estimators"],
        learning_rate=config["learning_rate"],
        num_leaves=config["num_leaves"],
        min_child_samples=config["min_child_samples"],
        subsample=config["subsample"],
        subsample_freq=config["subsample_freq"],
        colsample_bytree=config["colsample_bytree"],
        reg_alpha=config["reg_alpha"],
        reg_lambda=config["reg_lambda"],
        random_state=config["seed"],
        n_jobs=n_jobs,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


def _read_dev_labels(raw_dir: Path) -> pd.DataFrame:
    labels = pd.read_csv(
        raw_dir / "train" / "train_labels.csv",
        usecols=["kst_dtm", *TARGET_COLS],
        nrows=DEV_ROWS,
        encoding="utf-8-sig",
        parse_dates=["kst_dtm"],
    ).set_index("kst_dtm")
    labels.index = pd.DatetimeIndex(labels.index, name="forecast_kst_dtm")
    if len(labels) != DEV_ROWS or labels.index.max() != DEV_END:
        raise ValueError("Stage 1 label read is not the exact pre-2024 window")
    if (labels.index >= OOF_START).any():
        raise AssertionError("Stage 1 read a 2024 label")
    return labels


def _read_full_labels(raw_dir: Path) -> pd.DataFrame:
    labels = pd.read_csv(
        raw_dir / "train" / "train_labels.csv",
        usecols=["kst_dtm", *TARGET_COLS],
        nrows=TRAIN_ROWS,
        encoding="utf-8-sig",
        parse_dates=["kst_dtm"],
    ).set_index("kst_dtm")
    labels.index = pd.DatetimeIndex(labels.index, name="forecast_kst_dtm")
    if len(labels) != TRAIN_ROWS or labels.index.max() != OOF_END:
        raise ValueError("unexpected full training label window")
    return labels


def _read_parquet(path: Path, rows: int | None = None) -> pd.DataFrame:
    if rows is None:
        frame = pd.read_parquet(path)
    else:
        batch = next(pq.ParquetFile(path).iter_batches(batch_size=rows), None)
        if batch is None or batch.num_rows != rows:
            raise ValueError(f"{path}: bounded row count mismatch")
        frame = batch.to_pandas()
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{path}: invalid timestamp index")
    if frame.isna().any().any() or not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"{path}: non-finite features")
    return frame


def _feature_paths(
    base_dir: Path, augmentation_dir: Path, group: str, split: str
) -> tuple[Path, Path]:
    return (
        base_dir / f"{group}_weather_{split}.parquet",
        augmentation_dir / f"{group}_spatial_v2_{split}.parquet",
    )


def _source_sets(
    base: pd.DataFrame,
    augmentation: pd.DataFrame | None,
    spatial_variant: str | None,
) -> dict[str, pd.DataFrame]:
    common = [column for column in base if column.startswith(("time__", "site__"))]
    ldaps = common + [column for column in base if column.startswith("ldaps__")]
    gfs = common + [column for column in base if column.startswith("gfs__")]
    all_columns = list(base.columns)
    if augmentation is not None and spatial_variant is not None:
        bad = [column for column in augmentation if FORBIDDEN.search(column)]
        if bad:
            raise ValueError(f"target/SCADA-like spatial columns: {bad}")
        sequence = [column for column in augmentation if column.startswith("spv2seq__")]
        if len(sequence) != 60:
            raise ValueError("spatial sequence schema changed")
        selected = (
            [column for column in augmentation if column not in sequence]
            if spatial_variant == "core"
            else list(augmentation.columns)
        )
        ldaps += [
            column
            for column in selected
            if column.startswith("spv2__ldaps__")
            or column.startswith("spv2seq__ldaps_")
        ]
        gfs += [
            column
            for column in selected
            if column.startswith("spv2__gfs__")
            or column.startswith("spv2seq__gfs_")
        ]
        all_columns += selected
        combined = pd.concat([base, augmentation[selected]], axis=1)
    else:
        combined = base
    result = {
        "ldaps": combined.loc[:, ldaps].astype("float32"),
        "gfs": combined.loc[:, gfs].astype("float32"),
        "all": combined.loc[:, all_columns].astype("float32"),
    }
    if any(frame.columns.duplicated().any() for frame in result.values()):
        raise ValueError("source feature set contains duplicated columns")
    return result


def _fold(group: str, index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    if group in ("kpx_group_1", "kpx_group_2"):
        train = index < G12_DEV_START
        valid = index >= G12_DEV_START
        expected = (8_760, 8_760)
    else:
        train = (index >= G3_START) & (index < G3_DEV_START)
        valid = index >= G3_DEV_START
        expected = (4_344, 4_416)
    if (int(train.sum()), int(valid.sum())) != expected:
        raise AssertionError(f"{group}: Stage 1 fold changed")
    if index[train].max() >= index[valid].min():
        raise AssertionError("future leakage in Stage 1 fold")
    return np.asarray(train), np.asarray(valid)


def _score(actual: pd.Series, prediction: pd.Series, group: str) -> dict[str, Any]:
    if not actual.index.equals(prediction.index):
        raise ValueError("metric inputs are time-misaligned")
    metric = group_metrics(actual, prediction, CAPACITY_KWH[group], group_name=group)
    result = metric.as_dict()
    result["score"] = 0.5 * (metric.one_minus_nmae + metric.ficr)
    return result


def _segments(index: pd.DatetimeIndex, boundary: pd.Timestamp) -> dict[str, np.ndarray]:
    return {
        "full": np.ones(len(index), dtype=bool),
        "first_half": np.asarray(index < boundary),
        "second_half": np.asarray(index >= boundary),
    }


def _evaluate(
    actual: pd.Series,
    baseline: pd.Series,
    candidate: pd.Series,
    group: str,
    boundary: pd.Timestamp,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for segment, mask in _segments(actual.index, boundary).items():
        base_metric = _score(actual.loc[mask], baseline.loc[mask], group)
        candidate_metric = _score(actual.loc[mask], candidate.loc[mask], group)
        result[segment] = {
            "baseline": base_metric,
            "candidate": candidate_metric,
            "score_gain": float(candidate_metric["score"] - base_metric["score"]),
        }
    return result


def _stable(evaluation: dict[str, Any]) -> bool:
    full = evaluation["full"]["score_gain"]
    halves = [
        evaluation["first_half"]["score_gain"],
        evaluation["second_half"]["score_gain"],
    ]
    return bool(
        full > 0.0
        and min(halves) >= STABILITY_FLOOR
        and np.mean(halves) > 0.0
    )


def _fit_predict(
    x: pd.DataFrame,
    y: pd.Series,
    train_mask: np.ndarray,
    predict_mask: np.ndarray,
    group: str,
    objective: str,
    config: dict[str, Any],
    n_jobs: int,
) -> tuple[LGBMRegressor, pd.Series]:
    eligible = y.notna() & (
        y >= float(config["eligible_fraction"]) * CAPACITY_KWH[group]
    )
    fit_mask = train_mask & eligible.to_numpy()
    model = _make_model(config, objective, n_jobs)
    model.fit(
        x.loc[fit_mask],
        y.loc[fit_mask] / CAPACITY_KWH[group],
        callbacks=[log_evaluation(period=0)],
    )
    prediction = pd.Series(
        model.predict(x.loc[predict_mask]) * CAPACITY_KWH[group],
        index=x.index[predict_mask],
    )
    return model, prediction


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        frame.to_parquet(temporary, compression="zstd", index=True)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_model(model: LGBMRegressor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        joblib.dump(model, temporary, compress=3)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _verify_spatial_promotions(path: Path) -> None:
    promoted = set(json.loads(path.read_text(encoding="utf-8"))["promoted_candidates"])
    required = {
        "kpx_group_2__core__q07",
        "kpx_group_3__sequence__l1",
        "kpx_group_3__sequence__q07",
    }
    if not required.issubset(promoted):
        raise ValueError(f"required spatial inputs were not postgate-promoted: {required-promoted}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.n_jobs == 0:
        raise ValueError("--n-jobs must not be zero")
    raw_dir = args.raw_dir.expanduser().resolve()
    base_dir = args.base_cache_dir.expanduser().resolve()
    augmentation_dir = args.augmentation_dir.expanduser().resolve()
    spatial_results = args.spatial_postgate_results.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty output: {out_dir}")
    _verify_spatial_promotions(spatial_results)

    # Stage 1: no 2024 label or feature batch is requested before lock write.
    dev_labels = _read_dev_labels(raw_dir)
    config400 = _model_config(400)
    dev_predictions: dict[tuple[str, str, str], pd.Series] = {}
    dev_records: list[dict[str, Any]] = []
    cache_inputs: list[Path] = []
    for group in TARGET_COLS:
        base_path, aug_path = _feature_paths(base_dir, augmentation_dir, group, "train")
        base = _read_parquet(base_path, DEV_ROWS)
        if not base.index.equals(dev_labels.index):
            raise ValueError(f"{group}: bounded dev cache/label mismatch")
        cache_inputs.append(base_path)
        augmentation = None
        if group != "kpx_group_1":
            augmentation = _read_parquet(aug_path, DEV_ROWS)
            if not augmentation.index.equals(dev_labels.index):
                raise ValueError(f"{group}: bounded spatial cache/label mismatch")
            cache_inputs.append(aug_path)
        train_mask, valid_mask = _fold(group, dev_labels.index)
        for objective in OBJECTIVES:
            variant = SPATIAL_PROMOTED.get((group, objective))
            sets = _source_sets(base, augmentation, variant)
            for source in SOURCES:
                print(f"Stage1 {group} {objective} {source} {sets[source].shape[1]}f", flush=True)
                _, prediction = _fit_predict(
                    sets[source],
                    dev_labels[group],
                    train_mask,
                    valid_mask,
                    group,
                    objective,
                    config400,
                    args.n_jobs,
                )
                dev_predictions[(group, objective, source)] = prediction
                metric = _score(dev_labels.loc[prediction.index, group], prediction, group)
                dev_records.append(
                    {
                        "kind": "component",
                        "group": group,
                        "objective": objective,
                        "source": source,
                        "spatial_variant": variant,
                        "feature_count": int(sets[source].shape[1]),
                        "metrics": metric,
                    }
                )

    selections: list[dict[str, Any]] = []
    for group in TARGET_COLS:
        actual_index = dev_predictions[(group, "l1", "all")].index
        actual = dev_labels.loc[actual_index, group]
        boundary = G12_DEV_HALF if group != "kpx_group_3" else G3_DEV_HALF
        eligible_recipes: list[dict[str, Any]] = []
        for objective in OBJECTIVES:
            baseline = dev_predictions[(group, objective, "all")]
            for recipe_name, weights in ENSEMBLES.items():
                candidate = sum(
                    weight * dev_predictions[(group, objective, source)]
                    for source, weight in weights.items()
                )
                evaluation = _evaluate(actual, baseline, candidate, group, boundary)
                record = {
                    "kind": "ensemble",
                    "group": group,
                    "objective": objective,
                    "recipe": recipe_name,
                    "weights": weights,
                    "spatial_variant": SPATIAL_PROMOTED.get((group, objective)),
                    "evaluation": evaluation,
                    "stable": _stable(evaluation),
                }
                dev_records.append(record)
                if record["stable"]:
                    eligible_recipes.append(record)
        if eligible_recipes:
            chosen = max(
                eligible_recipes,
                key=lambda row: (
                    row["evaluation"]["full"]["candidate"]["score"],
                    row["recipe"],
                ),
            )
            selections.append(
                {
                    "group": group,
                    "objective": chosen["objective"],
                    "recipe": chosen["recipe"],
                    "weights": chosen["weights"],
                    "spatial_variant": chosen["spatial_variant"],
                    "stage1_evaluation": chosen["evaluation"],
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    dev_results_path = out_dir / "nwp_source_dev_results.json"
    lock_path = out_dir / "nwp_source_pre2024_lock.json"
    dev_results = {
        "stage": "pre_2024_only",
        "last_materialized_timestamp": DEV_END.isoformat(),
        "model_config": config400,
        "ensemble_candidates": ENSEMBLES,
        "stability_rule": {
            "full_gain_exclusive": 0.0,
            "each_half_gain_inclusive": STABILITY_FLOOR,
            "mean_half_gain_exclusive": 0.0,
        },
        "records": dev_records,
        "selections": selections,
    }
    write_json_atomic(dev_results_path, dev_results, overwrite=args.overwrite)
    lock = {
        "schema_version": 1,
        "lock_stage": "written_before_full_2024_read",
        "dev_results_sha256": sha256_file(dev_results_path),
        "confirmation_model_config": config400,
        "final_model_config": _model_config(1_500),
        "selections": selections,
        "confirmation_rule": dev_results["stability_rule"],
        "selection_changes_after_2024": False,
    }
    write_json_atomic(lock_path, lock, overwrite=args.overwrite)
    print(f"LOCK WRITTEN BEFORE 2024 READ: {lock_path}", flush=True)

    dev_oof_frame = pd.concat(
        [series.rename("__".join(key)) for key, series in dev_predictions.items()],
        axis=1,
    ).sort_index()
    dev_oof_path = out_dir / "nwp_source_dev_oof.parquet"
    _write_parquet(dev_oof_frame, dev_oof_path)

    # Stage 2: the selection lock now exists and is immutable for this run.
    full_labels = _read_full_labels(raw_dir)
    confirmation_records: list[dict[str, Any]] = []
    confirmation_oof: dict[str, pd.Series] = {}
    promoted: list[dict[str, Any]] = []
    model_outputs: list[Path] = []
    test_inputs: list[Path] = []
    final_predictions = pd.DataFrame()
    individual_predictions: list[Path] = []
    for selection in selections:
        group = selection["group"]
        objective = selection["objective"]
        base_path, aug_path = _feature_paths(base_dir, augmentation_dir, group, "train")
        base = _read_parquet(base_path)
        augmentation = (
            _read_parquet(aug_path) if selection["spatial_variant"] is not None else None
        )
        if not base.index.equals(full_labels.index):
            raise ValueError(f"{group}: full feature/label mismatch")
        sets = _source_sets(base, augmentation, selection["spatial_variant"])
        train_mask = (
            np.asarray(base.index < OOF_START)
            if group != "kpx_group_3"
            else np.asarray((base.index >= G3_START) & (base.index < OOF_START))
        )
        valid_mask = np.asarray(base.index >= OOF_START)
        component_predictions: dict[str, pd.Series] = {}
        needed = {source for source, weight in selection["weights"].items() if weight > 0}
        needed.add("all")
        for source in sorted(needed):
            print(f"Stage2 {group} {objective} {source}", flush=True)
            model, prediction = _fit_predict(
                sets[source],
                full_labels[group],
                train_mask,
                valid_mask,
                group,
                objective,
                config400,
                args.n_jobs,
            )
            component_predictions[source] = prediction
            model_path = out_dir / "models" / f"oof__{group}__{objective}__{source}.joblib"
            _write_model(model, model_path)
            model_outputs.append(model_path)
        candidate = sum(
            weight * component_predictions[source]
            for source, weight in selection["weights"].items()
            if weight > 0
        )
        baseline = component_predictions["all"]
        actual = full_labels.loc[candidate.index, group]
        evaluation = _evaluate(actual, baseline, candidate, group, OOF_HALF)
        passed = _stable(evaluation)
        identifier = f"{group}__{objective}__{selection['recipe']}"
        confirmation_records.append(
            {
                **selection,
                "candidate_id": identifier,
                "stage2_evaluation": evaluation,
                "promoted_to_1500_tree_final": passed,
            }
        )
        confirmation_oof[f"baseline__{group}__{objective}"] = baseline
        confirmation_oof[f"candidate__{identifier}"] = candidate
        if not passed:
            continue
        promoted.append(selection)

        test_base_path, test_aug_path = _feature_paths(
            base_dir, augmentation_dir, group, "test"
        )
        test_base = _read_parquet(test_base_path)
        test_augmentation = (
            _read_parquet(test_aug_path)
            if selection["spatial_variant"] is not None
            else None
        )
        test_sets = _source_sets(
            test_base, test_augmentation, selection["spatial_variant"]
        )
        test_inputs.extend(
            [test_base_path]
            + ([test_aug_path] if selection["spatial_variant"] is not None else [])
        )
        full_component_predictions: dict[str, np.ndarray] = {}
        config1500 = lock["final_model_config"]
        for source, weight in selection["weights"].items():
            if weight <= 0:
                continue
            print(f"Final1500 {group} {objective} {source}", flush=True)
            eligible = full_labels[group].notna() & (
                full_labels[group]
                >= config1500["eligible_fraction"] * CAPACITY_KWH[group]
            )
            model = _make_model(config1500, objective, args.n_jobs)
            model.fit(
                sets[source].loc[eligible],
                full_labels.loc[eligible, group] / CAPACITY_KWH[group],
                callbacks=[log_evaluation(period=0)],
            )
            pred = model.predict(test_sets[source]) * CAPACITY_KWH[group]
            if not np.isfinite(pred).all():
                raise ValueError(f"{identifier}/{source}: non-finite final prediction")
            full_component_predictions[source] = pred
            model_path = out_dir / "models" / f"final1500__{identifier}__{source}.joblib"
            _write_model(model, model_path)
            model_outputs.append(model_path)
        final_prediction = sum(
            weight * full_component_predictions[source]
            for source, weight in selection["weights"].items()
            if weight > 0
        )
        if final_predictions.empty:
            final_predictions = pd.DataFrame(index=test_base.index)
        final_predictions[identifier] = final_prediction
        individual = pd.DataFrame({group: final_prediction}, index=test_base.index)
        path = out_dir / "predictions" / f"{identifier}__test.parquet"
        _write_parquet(individual, path)
        individual_predictions.append(path)

    confirmation_oof_path = out_dir / "nwp_source_2024_oof.parquet"
    _write_parquet(pd.concat(confirmation_oof, axis=1), confirmation_oof_path)
    final_path = out_dir / "nwp_source_promoted_2025_predictions.parquet"
    _write_parquet(final_predictions, final_path)
    results_path = out_dir / "nwp_source_postgate_results.json"
    results = {
        "status": "completed",
        "pre2024_lock_sha256": sha256_file(lock_path),
        "confirmation_records": confirmation_records,
        "promoted_candidate_ids": [
            f"{item['group']}__{item['objective']}__{item['recipe']}"
            for item in promoted
        ],
        "note": (
            "Selection occurred before Stage 2 opened 2024. Only locked recipes "
            "stable in full and both halves of both periods were promoted."
        ),
    }
    write_json_atomic(results_path, results, overwrite=args.overwrite)
    manifest_path = out_dir / "nwp_source_postgate_manifest.json"
    outputs = [
        dev_results_path,
        lock_path,
        dev_oof_path,
        confirmation_oof_path,
        final_path,
        results_path,
        *model_outputs,
        *individual_predictions,
    ]
    manifest = make_manifest(
        artifact_type="baram_locked_nwp_source_ablation_and_postgate",
        parameters={
            "ensemble_candidates": ENSEMBLES,
            "spatial_promoted_map": {
                "|".join(key): value for key, value in SPATIAL_PROMOTED.items()
            },
            "stability_floor": STABILITY_FLOOR,
            "lock_written_before_2024_read": True,
            "n_jobs_runtime_only": args.n_jobs,
        },
        input_files=[
            raw_dir / "train" / "train_labels.csv",
            spatial_results,
            *dict.fromkeys([*cache_inputs, *test_inputs]),
        ],
        output_files=outputs,
        results={
            "stage1_selections": selections,
            "stage2_confirmation": confirmation_records,
            "promoted_candidate_ids": results["promoted_candidate_ids"],
        },
        project_dir=PROJECT_DIR,
    )
    write_json_atomic(manifest_path, manifest, overwrite=args.overwrite)
    print(f"promoted: {results['promoted_candidate_ids']}")
    print(f"complete: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

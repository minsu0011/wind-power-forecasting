"""Run the pre-locked spatial-v2 2024 postgate confirmation.

This program does not select candidates or thresholds from 2024.  Both are
loaded from ``configs/spatial_v2_postgate_lock.json``, whose source 2023 result
hash is verified before fitting.  Only candidates satisfying the locked full,
half-year, and (for group 3) prior H2-subhalf criteria are retrained on all
available labels and used to produce 2025 test prediction Parquets.
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
from lightgbm import LGBMRegressor, log_evaluation


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.manifest import make_manifest, sha256_file, write_json_atomic  # noqa: E402
from src.metric import CAPACITY_KWH, group_metrics  # noqa: E402


GROUPS = ("kpx_group_2", "kpx_group_3")
TRAIN_ROWS = 26_304
TEST_ROWS = 8_760
OOF_START = pd.Timestamp("2024-01-01 01:00:00")
OOF_END = pd.Timestamp("2025-01-01 00:00:00")
OOF_HALF = pd.Timestamp("2024-07-01 01:00:00")
G3_TRAIN_START = pd.Timestamp("2023-01-01 01:00:00")
G3_PRIOR_START = pd.Timestamp("2023-07-01 01:00:00")
G3_PRIOR_HALF = pd.Timestamp("2023-10-01 01:00:00")
G3_PRIOR_END = pd.Timestamp("2024-01-01 00:00:00")
FORBIDDEN_FEATURE = re.compile(
    r"(^kpx_group_|scada|target|label|power_kw)", re.IGNORECASE
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run locked 2024 OOF confirmation and promoted full-2025 fits."
    )
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--base-cache-dir", type=Path, required=True)
    parser.add_argument("--augmentation-dir", type=Path, required=True)
    parser.add_argument(
        "--lock-config",
        type=Path,
        default=PROJECT_DIR / "configs" / "spatial_v2_postgate_lock.json",
    )
    parser.add_argument(
        "--dev-results",
        type=Path,
        default=(
            PROJECT_DIR
            / "artifacts"
            / "experiments"
            / "spatial_v2_seq"
            / "spatial_v2_dev_results.json"
        ),
    )
    parser.add_argument(
        "--dev-oof",
        type=Path,
        default=(
            PROJECT_DIR
            / "artifacts"
            / "experiments"
            / "spatial_v2_seq"
            / "spatial_v2_dev_oof.parquet"
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _read_labels(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "train" / "train_labels.csv"
    labels = pd.read_csv(
        path,
        usecols=["kst_dtm", *GROUPS],
        nrows=TRAIN_ROWS,
        encoding="utf-8-sig",
        parse_dates=["kst_dtm"],
    ).set_index("kst_dtm")
    labels.index = pd.DatetimeIndex(labels.index, name="forecast_kst_dtm")
    if len(labels) != TRAIN_ROWS:
        raise ValueError(f"expected {TRAIN_ROWS} official training rows")
    if labels.index.min() != pd.Timestamp("2022-01-01 01:00:00"):
        raise ValueError("unexpected first training timestamp")
    if labels.index.max() != OOF_END or not labels.index.is_unique:
        raise ValueError("unexpected or duplicated training timestamp index")
    if not labels.index.is_monotonic_increasing:
        raise ValueError("training labels are not chronological")
    return labels


def _read_feature_pair(
    base_dir: Path, augmentation_dir: Path, group: str, split: str
) -> tuple[dict[str, pd.DataFrame], list[Path]]:
    base_path = base_dir / f"{group}_weather_{split}.parquet"
    aug_path = augmentation_dir / f"{group}_spatial_v2_{split}.parquet"
    for path in (base_path, aug_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    base = pd.read_parquet(base_path)
    aug = pd.read_parquet(aug_path)
    base.index = pd.DatetimeIndex(base.index, name="forecast_kst_dtm")
    aug.index = pd.DatetimeIndex(aug.index, name="forecast_kst_dtm")
    expected_rows = TRAIN_ROWS if split == "train" else TEST_ROWS
    if len(base) != expected_rows or len(aug) != expected_rows:
        raise ValueError(f"{group}/{split}: unexpected cache row count")
    if not base.index.equals(aug.index) or not base.index.is_unique:
        raise ValueError(f"{group}/{split}: v1/v2 index mismatch")
    if not base.index.is_monotonic_increasing:
        raise ValueError(f"{group}/{split}: cache index is not chronological")
    forbidden = [column for column in aug if FORBIDDEN_FEATURE.search(column)]
    if forbidden:
        raise ValueError(f"{group}/{split}: target/SCADA-like features: {forbidden}")
    if base.isna().any().any() or aug.isna().any().any():
        raise ValueError(f"{group}/{split}: missing feature values")
    if not np.isfinite(base.to_numpy()).all() or not np.isfinite(aug.to_numpy()).all():
        raise ValueError(f"{group}/{split}: non-finite feature values")
    sequence = [column for column in aug if column.startswith("spv2seq__")]
    if len(sequence) != 60:
        raise ValueError(f"{group}/{split}: expected 60 sequence features")
    core = [column for column in aug if column not in sequence]
    frames = {
        "v1": base.astype("float32"),
        "core": pd.concat([base, aug[core]], axis=1).astype("float32"),
        "sequence": pd.concat([base, aug], axis=1).astype("float32"),
    }
    return frames, [base_path, aug_path]


def _load_lock(path: Path, dev_results: Path) -> dict[str, Any]:
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("lock_stage") != "before_2024_oof":
        raise ValueError("lock config is not marked before_2024_oof")
    expected_hash = lock["selection_evidence"]["results_sha256"]
    observed_hash = sha256_file(dev_results)
    if observed_hash != expected_hash:
        raise ValueError(
            "2023 selection evidence hash changed after candidate locking: "
            f"{observed_hash} != {expected_hash}"
        )
    recipe = lock["prediction_recipe"]
    if recipe != {
        "augmented_model_weight": 1.0,
        "baseline_model_weight": 0.0,
        "calibration": None,
        "clipping": None,
    }:
        raise ValueError("this runner only supports the pre-locked raw augmented recipe")
    return lock


def _make_model(
    model_config: dict[str, Any], objective: str, n_jobs: int
) -> LGBMRegressor:
    objective_args: dict[str, Any]
    if objective == "l1":
        objective_args = {"objective": "regression_l1"}
    elif objective == "q07":
        objective_args = {"objective": "quantile", "alpha": 0.70}
    else:
        raise ValueError(f"unsupported locked objective: {objective}")
    return LGBMRegressor(
        **objective_args,
        n_estimators=int(model_config["n_estimators"]),
        learning_rate=float(model_config["learning_rate"]),
        num_leaves=int(model_config["num_leaves"]),
        min_child_samples=int(model_config["min_child_samples"]),
        subsample=float(model_config["subsample"]),
        subsample_freq=int(model_config["subsample_freq"]),
        colsample_bytree=float(model_config["colsample_bytree"]),
        reg_alpha=float(model_config["reg_alpha"]),
        reg_lambda=float(model_config["reg_lambda"]),
        random_state=int(model_config["seed"]),
        n_jobs=n_jobs,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


def _score(actual: pd.Series, prediction: pd.Series, group: str) -> dict[str, Any]:
    if not actual.index.equals(prediction.index):
        raise ValueError("actual/prediction timestamps differ")
    metric = group_metrics(
        actual.to_numpy(dtype=float),
        prediction.to_numpy(dtype=float),
        CAPACITY_KWH[group],
        group_name=group,
    )
    result = metric.as_dict()
    result["score"] = 0.5 * (metric.one_minus_nmae + metric.ficr)
    return result


def _segment_scores(
    actual: pd.Series,
    baseline: pd.Series,
    candidate: pd.Series,
    group: str,
    segments: dict[str, np.ndarray],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, mask in segments.items():
        baseline_metrics = _score(actual.loc[mask], baseline.loc[mask], group)
        candidate_metrics = _score(actual.loc[mask], candidate.loc[mask], group)
        result[name] = {
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "score_gain": float(candidate_metrics["score"] - baseline_metrics["score"]),
        }
    return result


def _dump_joblib_atomic(model: LGBMRegressor, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        joblib.dump(model, temporary, compress=3)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_parquet_atomic(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        frame.to_parquet(temporary, compression="zstd", index=True)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _candidate_id(candidate: dict[str, str]) -> str:
    return f"{candidate['group']}__{candidate['variant']}__{candidate['objective']}"


def _promoted(
    candidate: dict[str, str],
    scores_2024: dict[str, Any],
    prior_scores: dict[str, Any] | None,
    thresholds: dict[str, Any],
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    full_gain = scores_2024["full"]["score_gain"]
    half_gains = [scores_2024["h1"]["score_gain"], scores_2024["h2"]["score_gain"]]
    if not full_gain > float(thresholds["2024_full_min_gain_exclusive"]):
        failures.append("2024 full gain is not positive")
    half_floor = float(thresholds["2024_each_half_min_gain_inclusive"])
    if any(gain < half_floor for gain in half_gains):
        failures.append("a 2024 half gain is below the locked floor")
    if not np.mean(half_gains) > float(thresholds["2024_half_mean_min_gain_exclusive"]):
        failures.append("mean 2024 half gain is not positive")
    if candidate["group"] == "kpx_group_3":
        if prior_scores is None:
            raise ValueError("group 3 prior subhalf evidence is missing")
        prior_gains = [
            prior_scores["q3"]["score_gain"],
            prior_scores["q4"]["score_gain"],
        ]
        prior_floor = float(
            thresholds["g3_2023_h2_each_subhalf_min_gain_inclusive"]
        )
        if any(gain < prior_floor for gain in prior_gains):
            failures.append("a g3 2023-H2 subhalf gain is below the locked floor")
        if not np.mean(prior_gains) > float(
            thresholds["g3_2023_h2_subhalf_mean_min_gain_exclusive"]
        ):
            failures.append("mean g3 2023-H2 subhalf gain is not positive")
    return not failures, failures


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.n_jobs == 0:
        raise ValueError("--n-jobs must not be zero")
    raw_dir = args.raw_dir.expanduser().resolve()
    base_dir = args.base_cache_dir.expanduser().resolve()
    augmentation_dir = args.augmentation_dir.expanduser().resolve()
    lock_path = args.lock_config.expanduser().resolve()
    dev_results = args.dev_results.expanduser().resolve()
    dev_oof_path = args.dev_oof.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    results_path = out_dir / "spatial_v2_postgate_results.json"
    oof_path = out_dir / "spatial_v2_2024_oof.parquet"
    promoted_path = out_dir / "spatial_v2_promoted_2025_predictions.parquet"
    manifest_path = out_dir / "spatial_v2_postgate_manifest.json"
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty postgate directory: {out_dir}")

    lock = _load_lock(lock_path, dev_results)
    candidates = lock["candidates"]
    if len({_candidate_id(candidate) for candidate in candidates}) != len(candidates):
        raise ValueError("locked candidate identifiers are duplicated")
    labels = _read_labels(raw_dir)
    train_features: dict[str, dict[str, pd.DataFrame]] = {}
    test_features: dict[str, dict[str, pd.DataFrame]] = {}
    cache_inputs: list[Path] = []
    for group in GROUPS:
        train_features[group], train_inputs = _read_feature_pair(
            base_dir, augmentation_dir, group, "train"
        )
        test_features[group], test_inputs = _read_feature_pair(
            base_dir, augmentation_dir, group, "test"
        )
        if not train_features[group]["v1"].index.equals(labels.index):
            raise ValueError(f"{group}: labels/train-feature alignment failed")
        cache_inputs.extend([*train_inputs, *test_inputs])
    test_index = test_features[GROUPS[0]]["v1"].index
    if not test_index.equals(test_features[GROUPS[1]]["v1"].index):
        raise ValueError("group test indices differ")
    if test_index.min() != pd.Timestamp("2025-01-01 01:00:00") or test_index.max() != pd.Timestamp("2026-01-01 00:00:00"):
        raise ValueError("unexpected 2025 test period")

    dev_oof = pd.read_parquet(dev_oof_path)
    dev_oof.index = pd.DatetimeIndex(dev_oof.index, name="forecast_kst_dtm")
    model_config = lock["model"]
    eligible_fraction = float(model_config["eligible_fraction"])
    baseline_predictions: dict[tuple[str, str], pd.Series] = {}
    candidate_predictions: dict[str, pd.Series] = {}
    oof_models: list[Path] = []

    for group in GROUPS:
        if group == "kpx_group_2":
            train_mask = labels.index < OOF_START
        else:
            train_mask = (labels.index >= G3_TRAIN_START) & (labels.index < OOF_START)
        valid_mask = labels.index >= OOF_START
        expected_train = 17_520 if group == "kpx_group_2" else 8_760
        if int(train_mask.sum()) != expected_train or int(valid_mask.sum()) != 8_784:
            raise AssertionError(f"{group}: 2024 OOF fold sizes changed")
        if labels.index[train_mask].max() >= labels.index[valid_mask].min():
            raise AssertionError(f"{group}: future row entered OOF training")
        y = labels[group]
        eligible = y.notna() & (y >= eligible_fraction * CAPACITY_KWH[group])
        fit_mask = train_mask & eligible
        group_objectives = sorted(
            {candidate["objective"] for candidate in candidates if candidate["group"] == group}
        )
        for objective in group_objectives:
            print(f"2024 OOF baseline {group} {objective}", flush=True)
            model = _make_model(model_config, objective, args.n_jobs)
            model.fit(
                train_features[group]["v1"].loc[fit_mask],
                y.loc[fit_mask] / CAPACITY_KWH[group],
                callbacks=[log_evaluation(period=0)],
            )
            prediction = pd.Series(
                model.predict(train_features[group]["v1"].loc[valid_mask])
                * CAPACITY_KWH[group],
                index=labels.index[valid_mask],
            )
            baseline_predictions[(group, objective)] = prediction
            path = out_dir / "models" / f"oof__{group}__v1__{objective}.joblib"
            _dump_joblib_atomic(model, path)
            oof_models.append(path)

        for candidate in [item for item in candidates if item["group"] == group]:
            identifier = _candidate_id(candidate)
            print(f"2024 OOF candidate {identifier}", flush=True)
            model = _make_model(model_config, candidate["objective"], args.n_jobs)
            model.fit(
                train_features[group][candidate["variant"]].loc[fit_mask],
                y.loc[fit_mask] / CAPACITY_KWH[group],
                callbacks=[log_evaluation(period=0)],
            )
            prediction = pd.Series(
                model.predict(
                    train_features[group][candidate["variant"]].loc[valid_mask]
                )
                * CAPACITY_KWH[group],
                index=labels.index[valid_mask],
            )
            candidate_predictions[identifier] = prediction
            path = out_dir / "models" / f"oof__{identifier}.joblib"
            _dump_joblib_atomic(model, path)
            oof_models.append(path)

    records: list[dict[str, Any]] = []
    promoted_candidates: list[dict[str, str]] = []
    for candidate in candidates:
        identifier = _candidate_id(candidate)
        group = candidate["group"]
        actual_2024 = labels.loc[OOF_START:OOF_END, group]
        baseline_2024 = baseline_predictions[(group, candidate["objective"])]
        prediction_2024 = candidate_predictions[identifier]
        segments_2024 = {
            "full": np.ones(len(actual_2024), dtype=bool),
            "h1": np.asarray(actual_2024.index < OOF_HALF),
            "h2": np.asarray(actual_2024.index >= OOF_HALF),
        }
        scores_2024 = _segment_scores(
            actual_2024,
            baseline_2024,
            prediction_2024,
            group,
            segments_2024,
        )

        prior_scores: dict[str, Any] | None = None
        if group == "kpx_group_3":
            feature_set = (
                "v1_plus_v2_core" if candidate["variant"] == "core" else "v1_plus_v2_seq"
            )
            prior_index = labels.loc[G3_PRIOR_START:G3_PRIOR_END].index
            baseline_column = f"{group}__v1__{candidate['objective']}"
            candidate_column = f"{group}__{feature_set}__{candidate['objective']}"
            actual_prior = labels.loc[prior_index, group]
            baseline_prior = dev_oof.loc[prior_index, baseline_column]
            candidate_prior = dev_oof.loc[prior_index, candidate_column]
            if baseline_prior.isna().any() or candidate_prior.isna().any():
                raise ValueError(f"{identifier}: missing prior H2 predictions")
            prior_scores = _segment_scores(
                actual_prior,
                baseline_prior,
                candidate_prior,
                group,
                {
                    "q3": np.asarray(prior_index < G3_PRIOR_HALF),
                    "q4": np.asarray(prior_index >= G3_PRIOR_HALF),
                },
            )
        promoted, failures = _promoted(
            candidate,
            scores_2024,
            prior_scores,
            lock["promotion_rule"],
        )
        records.append(
            {
                "candidate_id": identifier,
                **candidate,
                "feature_count": int(train_features[group][candidate["variant"]].shape[1]),
                "scores_2024": scores_2024,
                "scores_g3_2023_h2_subhalves": prior_scores,
                "promoted_to_full_2025": promoted,
                "promotion_failures": failures,
            }
        )
        if promoted:
            promoted_candidates.append(candidate)

    oof_frame = pd.DataFrame(index=pd.date_range(OOF_START, OOF_END, freq="h"))
    oof_frame.index.name = "forecast_kst_dtm"
    for (group, objective), prediction in baseline_predictions.items():
        oof_frame[f"baseline__{group}__{objective}"] = prediction
    for identifier, prediction in candidate_predictions.items():
        oof_frame[f"candidate__{identifier}"] = prediction
    _write_parquet_atomic(oof_frame, oof_path)

    promoted_predictions = pd.DataFrame(index=test_index)
    full_models: list[Path] = []
    individual_predictions: list[Path] = []
    for candidate in promoted_candidates:
        identifier = _candidate_id(candidate)
        group = candidate["group"]
        y = labels[group]
        eligible = y.notna() & (y >= eligible_fraction * CAPACITY_KWH[group])
        print(f"full 2025 promoted candidate {identifier}", flush=True)
        model = _make_model(model_config, candidate["objective"], args.n_jobs)
        model.fit(
            train_features[group][candidate["variant"]].loc[eligible],
            y.loc[eligible] / CAPACITY_KWH[group],
            callbacks=[log_evaluation(period=0)],
        )
        prediction = model.predict(test_features[group][candidate["variant"]]) * CAPACITY_KWH[group]
        if not np.isfinite(prediction).all():
            raise ValueError(f"{identifier}: non-finite test prediction")
        promoted_predictions[identifier] = prediction
        model_path = out_dir / "models" / f"full_2025__{identifier}.joblib"
        _dump_joblib_atomic(model, model_path)
        full_models.append(model_path)
        individual = pd.DataFrame({group: prediction}, index=test_index)
        individual_path = out_dir / "predictions" / f"{identifier}__test.parquet"
        _write_parquet_atomic(individual, individual_path)
        individual_predictions.append(individual_path)
    _write_parquet_atomic(promoted_predictions, promoted_path)

    result_payload = {
        "status": "completed",
        "postgate_only": True,
        "lock_config_sha256": sha256_file(lock_path),
        "selection_evidence_sha256_verified": lock["selection_evidence"]["results_sha256"],
        "records": records,
        "promoted_candidates": [_candidate_id(candidate) for candidate in promoted_candidates],
        "note": (
            "2024 was used only for the pre-locked confirmation rule. No model, "
            "feature variant, objective, blend, clipping, or threshold was reselected."
        ),
    }
    write_json_atomic(results_path, result_payload, overwrite=args.overwrite)
    output_files = [
        results_path,
        oof_path,
        promoted_path,
        *oof_models,
        *full_models,
        *individual_predictions,
    ]
    manifest = make_manifest(
        artifact_type="baram_spatial_v2_locked_2024_postgate_confirmation",
        parameters={
            "lock_config": lock,
            "n_jobs_runtime_only": args.n_jobs,
            "train_test_alignment_asserted": True,
            "target_scada_feature_asserted_absent": True,
        },
        input_files=[
            lock_path,
            dev_results,
            dev_oof_path,
            raw_dir / "train" / "train_labels.csv",
            *cache_inputs,
        ],
        output_files=output_files,
        results={
            "records": records,
            "promoted_candidates": result_payload["promoted_candidates"],
        },
        project_dir=PROJECT_DIR,
    )
    write_json_atomic(manifest_path, manifest, overwrite=args.overwrite)
    print(f"promoted: {result_payload['promoted_candidates']}")
    print(f"complete: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Preregistered label-free annual quantile-map audit.

This script deliberately has no submission writer and never opens test weather,
SCADA, prior predictions, or leaderboard feedback.  It compares raw and
application-year-marginal-mapped features with one identical forward-fitted
low-dimensional model on the fixed 2023 and 2024 folds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.metric import CAPACITY_KWH, group_metrics


EXPECTED_PREREGISTER_SHA256 = (
    "da2116e741d5b0920601c350289ab8856ed6853a26f59b493b28a2dd2639e4db"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _quantile_map(
    reference: pd.Series,
    application: pd.Series,
    *,
    grid_size: int,
) -> tuple[pd.Series, dict[str, float | int]]:
    """Map the application marginal to the reference marginal without labels."""

    reference_values = reference.to_numpy(dtype=np.float64, copy=False)
    application_values = application.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(reference_values).all() or not np.isfinite(application_values).all():
        raise ValueError("quantile-map inputs must be finite")
    probabilities = np.linspace(0.0, 1.0, grid_size, dtype=np.float64)
    reference_quantiles = np.quantile(reference_values, probabilities, method="linear")
    application_quantiles = np.quantile(
        application_values, probabilities, method="linear"
    )

    unique_values, inverse = np.unique(application_quantiles, return_inverse=True)
    probability_sums = np.bincount(inverse, weights=probabilities)
    probability_counts = np.bincount(inverse)
    mean_probabilities = probability_sums / probability_counts
    mapped_knots = np.interp(
        mean_probabilities, probabilities, reference_quantiles
    )
    mapped_values = np.interp(
        application_values,
        unique_values,
        mapped_knots,
        left=reference_quantiles[0],
        right=reference_quantiles[-1],
    )
    if not np.isfinite(mapped_values).all():
        raise AssertionError("quantile mapping produced non-finite values")
    audit = {
        "reference_mean": float(reference_values.mean()),
        "application_mean": float(application_values.mean()),
        "mapped_mean": float(mapped_values.mean()),
        "reference_median": float(np.median(reference_values)),
        "application_median": float(np.median(application_values)),
        "mapped_median": float(np.median(mapped_values)),
        "application_unique_quantile_knots": int(len(unique_values)),
    }
    return pd.Series(mapped_values, index=application.index), audit


def _period_masks(index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    return {
        "full": np.ones(len(index), dtype=bool),
        "H1": index.month <= 6,
        "H2": index.month >= 7,
    }


def _score(
    truth: pd.DataFrame,
    prediction: pd.DataFrame,
    groups: list[str],
) -> dict[str, Any]:
    group_payload: dict[str, Any] = {}
    one_minus_nmae: list[float] = []
    ficr: list[float] = []
    for group in groups:
        metric = group_metrics(truth[group], prediction[group], CAPACITY_KWH[group])
        payload = metric.as_dict()
        group_payload[group] = payload
        one_minus_nmae.append(metric.one_minus_nmae)
        ficr.append(metric.ficr)
    aggregate_one_minus_nmae = float(np.mean(one_minus_nmae))
    aggregate_ficr = float(np.mean(ficr))
    return {
        "total_score": 0.5 * (aggregate_one_minus_nmae + aggregate_ficr),
        "one_minus_nmae": aggregate_one_minus_nmae,
        "ficr": aggregate_ficr,
        "by_group": group_payload,
    }


def _delta(candidate: dict[str, Any], identity: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        key: float(candidate[key] - identity[key])
        for key in ("total_score", "one_minus_nmae", "ficr")
    }
    payload["by_group_total_score"] = {}
    for group, candidate_group in candidate["by_group"].items():
        identity_group = identity["by_group"][group]
        candidate_total = 0.5 * (
            candidate_group["one_minus_nmae"] + candidate_group["ficr"]
        )
        identity_total = 0.5 * (
            identity_group["one_minus_nmae"] + identity_group["ficr"]
        )
        payload["by_group_total_score"][group] = float(
            candidate_total - identity_total
        )
    return payload


def _run_fold(
    fold: dict[str, Any],
    config: dict[str, Any],
    labels: pd.DataFrame,
    cache_dir: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], Path]:
    fit_start = pd.Timestamp(fold["fit_start"])
    fit_end = pd.Timestamp(fold["fit_end"])
    application_start = pd.Timestamp(fold["application_start"])
    application_end = pd.Timestamp(fold["application_end"])
    groups = list(fold["groups"])
    features = list(config["model_features"])
    mapped_features = list(config["mapped_features"])
    grid_size = int(config["adaptation"]["quantile_grid_size"])
    estimator_parameters = dict(config["estimator"])
    estimator_parameters.pop("library")
    estimator_parameters.pop("target")

    identity = pd.DataFrame(index=pd.date_range(application_start, application_end, freq="h"))
    candidate = pd.DataFrame(index=identity.index)
    mapping_audit: dict[str, Any] = {}
    training_rows: dict[str, int] = {}

    for group in groups:
        cache_path = cache_dir / f"{group}_weather_train.parquet"
        frame = pd.read_parquet(cache_path, columns=features)
        expected_full_index = labels.index
        if not frame.index.equals(expected_full_index):
            raise ValueError(f"{group} feature cache and labels are not exactly aligned")
        missing_columns = set(mapped_features).difference(frame.columns)
        if missing_columns:
            raise ValueError(f"{group} missing mapped columns: {sorted(missing_columns)}")

        fit_mask = frame.index.to_series().between(fit_start, fit_end).to_numpy()
        application_mask = frame.index.to_series().between(
            application_start, application_end
        ).to_numpy()
        fit_frame = frame.loc[fit_mask, features]
        application_frame = frame.loc[application_mask, features]
        if not application_frame.index.equals(identity.index):
            raise ValueError(f"{fold['name']} application index is incomplete for {group}")
        if not np.isfinite(fit_frame.to_numpy(dtype=float)).all():
            raise ValueError("fit features contain non-finite values")
        if not np.isfinite(application_frame.to_numpy(dtype=float)).all():
            raise ValueError("application features contain non-finite values")

        adapted = application_frame.astype(np.float64, copy=True)
        group_mapping_audit: dict[str, Any] = {}
        for feature in mapped_features:
            adapted_feature, feature_audit = _quantile_map(
                fit_frame[feature], application_frame[feature], grid_size=grid_size
            )
            adapted.loc[:, feature] = adapted_feature
            group_mapping_audit[feature] = feature_audit
        mapping_audit[group] = group_mapping_audit

        capacity = CAPACITY_KWH[group]
        target = labels.loc[fit_frame.index, group] / capacity
        eligible = target.notna() & target.ge(0.10)
        if not eligible.any():
            raise ValueError(f"no eligible training rows for {group}")
        training_rows[group] = int(eligible.sum())
        model = LGBMRegressor(**estimator_parameters)
        model.fit(fit_frame.loc[eligible], target.loc[eligible])
        identity[group] = np.clip(
            model.predict(application_frame) * capacity, 0.0, 1.02 * capacity
        )
        candidate[group] = np.clip(
            model.predict(adapted) * capacity, 0.0, 1.02 * capacity
        )

    truth = labels.loc[identity.index, groups]
    period_results: dict[str, Any] = {}
    for period, mask in _period_masks(identity.index).items():
        identity_score = _score(truth.iloc[mask], identity.iloc[mask], groups)
        candidate_score = _score(truth.iloc[mask], candidate.iloc[mask], groups)
        period_results[period] = {
            "identity": identity_score,
            "candidate": candidate_score,
            "delta": _delta(candidate_score, identity_score),
        }

    prediction_path = output_dir / "oof" / f"{fold['name']}__predictions.parquet"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(
        {"identity": identity, "quantile_mapped": candidate}, axis=1
    )
    combined.index.name = "forecast_kst_dtm"
    combined.to_parquet(prediction_path, index=True)
    fold_result = {
        "fold": fold,
        "training_rows": training_rows,
        "mapping_audit": mapping_audit,
        "periods": period_results,
        "prediction_path": str(prediction_path.resolve()),
        "prediction_sha256": _sha256(prediction_path),
    }
    return fold_result, prediction_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preregister",
        type=Path,
        default=Path("configs/year_quantile_map_audit_preregister.json"),
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=Path(r"data/local/open")
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/cache"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/postgate/year_quantile_map_audit"),
    )
    args = parser.parse_args()

    preregister_sha256 = _sha256(args.preregister)
    if preregister_sha256 != EXPECTED_PREREGISTER_SHA256:
        raise ValueError(
            "preregister hash mismatch; refusing to score an altered experiment: "
            f"{preregister_sha256}"
        )
    config = json.loads(args.preregister.read_text(encoding="utf-8"))
    labels_path = args.raw_dir / "train" / "train_labels.csv"
    labels = pd.read_csv(
        labels_path, encoding="utf-8-sig", parse_dates=["kst_dtm"]
    ).set_index("kst_dtm")
    if not labels.index.is_unique or not labels.index.is_monotonic_increasing:
        raise ValueError("label index must be unique and increasing")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_results: dict[str, Any] = {}
    output_paths: list[Path] = []
    for fold in config["forward_folds"]:
        result, output_path = _run_fold(
            fold, config, labels, args.cache_dir, args.output_dir
        )
        fold_results[fold["name"]] = result
        output_paths.append(output_path)

    all_period_nonnegative = all(
        result["periods"][period]["delta"]["total_score"] >= -1e-12
        for result in fold_results.values()
        for period in ("full", "H1", "H2")
    )
    full_nmae_safe = all(
        result["periods"]["full"]["delta"]["one_minus_nmae"] >= -0.001
        for result in fold_results.values()
    )
    group_safe = all(
        delta >= -0.001
        for result in fold_results.values()
        for delta in result["periods"]["full"]["delta"][
            "by_group_total_score"
        ].values()
    )
    promoted = bool(all_period_nonnegative and full_nmae_safe and group_safe)
    results = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "preregister_sha256": preregister_sha256,
        "public_scores_read": False,
        "test_weather_read": False,
        "submission_csv_written": False,
        "fold_results": fold_results,
        "gate": {
            "all_period_total_deltas_nonnegative": all_period_nonnegative,
            "full_one_minus_nmae_safe": full_nmae_safe,
            "all_group_full_total_deltas_at_least_minus_0.001": group_safe,
            "promoted": promoted,
            "decision": "promote" if promoted else "reject",
        },
    }
    results_path = args.output_dir / "results.json"
    _write_json(results_path, results)
    output_paths.append(results_path)

    input_paths = [args.preregister, labels_path]
    for group in ("kpx_group_1", "kpx_group_2", "kpx_group_3"):
        input_paths.append(args.cache_dir / f"{group}_weather_train.parquet")
    manifest = {
        "schema_version": 1,
        "artifact_type": "baram_year_quantile_map_audit",
        "inputs": [
            {
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in input_paths
        ],
        "outputs": [
            {
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in output_paths
        ],
        "gate": results["gate"],
    }
    manifest_path = args.output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    print(json.dumps(results["gate"], indent=2, sort_keys=True))
    for fold_name, fold_result in fold_results.items():
        print(fold_name)
        for period, payload in fold_result["periods"].items():
            print(period, payload["delta"])


if __name__ == "__main__":
    main()

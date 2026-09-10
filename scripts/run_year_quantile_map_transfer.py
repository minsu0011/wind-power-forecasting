"""Build the fixed 2025 year-quantile-map candidates after strict gates.

No weight, delta scale, group switch, or Public-score parameter exists here.
The exact low-dimensional model and mapping are inherited from the already
locked year-quantile-map audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.metric import CAPACITY_KWH
from scripts.run_year_quantile_map_audit import (
    _delta,
    _period_masks,
    _quantile_map,
    _score,
)


EXPECTED_TRANSFER_PREREGISTER_SHA256 = (
    "ca30a24c8a6c60c521fb7d9e9e4f9ac5f2845f37a9afeca58bde5de763c27a5c"
)
EXPECTED_PARENT_PREREGISTER_SHA256 = (
    "da2116e741d5b0920601c350289ab8856ed6853a26f59b493b28a2dd2639e4db"
)
GROUPS = ("kpx_group_1", "kpx_group_2", "kpx_group_3")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_label_window_bounded(
    path: Path,
    *,
    expected_index: pd.DatetimeIndex,
    groups: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Materialize exactly one fixed label window and stop before later rows.

    This streaming loader is used for staged gates.  Unlike ``pd.read_csv`` on
    the whole label file, it breaks as soon as the first timestamp after the
    requested window is encountered and never stores a later label row.
    """

    if len(expected_index) == 0:
        raise ValueError("expected_index must not be empty")
    if not expected_index.is_unique or not expected_index.is_monotonic_increasing:
        raise ValueError("expected_index must be unique and increasing")
    start = pd.Timestamp(expected_index.min())
    end = pd.Timestamp(expected_index.max())
    timestamps: list[pd.Timestamp] = []
    values: dict[str, list[float]] = {group: [] for group in groups}
    rows_scanned_before_window = 0
    prefix_digest = hashlib.sha256()
    prefix_bytes_read = 0
    stopped_at_requested_end = False
    # Unbuffered binary reads plus an immediate break on the requested end row
    # make the audited prefix boundary explicit.  No later CSV row is asked for.
    with path.open("rb", buffering=0) as stream:
        header_bytes = stream.readline()
        if not header_bytes:
            raise ValueError("label CSV is empty")
        prefix_digest.update(header_bytes)
        prefix_bytes_read += len(header_bytes)
        header_text = header_bytes.decode("utf-8-sig").rstrip("\r\n")
        fieldnames = next(csv.reader([header_text]))
        required = {"kst_dtm", *groups}
        if not required.issubset(fieldnames):
            raise ValueError("label CSV is missing staged-loader columns")
        for line_bytes in iter(stream.readline, b""):
            prefix_digest.update(line_bytes)
            prefix_bytes_read += len(line_bytes)
            fields = next(csv.reader([line_bytes.decode("utf-8").rstrip("\r\n")]))
            if len(fields) != len(fieldnames):
                raise ValueError("malformed row in bounded label prefix")
            raw = dict(zip(fieldnames, fields))
            timestamp = pd.Timestamp(raw["kst_dtm"])
            if timestamp < start:
                rows_scanned_before_window += 1
                continue
            if timestamp > end:
                raise ValueError("bounded loader advanced past requested_end")
            timestamps.append(timestamp)
            for group in groups:
                token = raw[group].strip()
                values[group].append(float(token) if token else np.nan)
            if timestamp == end:
                stopped_at_requested_end = True
                break
    index = pd.DatetimeIndex(timestamps, name="kst_dtm")
    if not index.equals(expected_index.rename("kst_dtm")):
        raise ValueError("bounded label window does not match the expected fold index")
    frame = pd.DataFrame(values, index=index)
    audit = {
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "rows_materialized": len(frame),
        "rows_scanned_before_window": rows_scanned_before_window,
        "max_materialized_timestamp": frame.index.max().isoformat(),
        "stopped_at_requested_end": stopped_at_requested_end,
        "later_label_values_materialized": 0,
        "groups_materialized": groups,
        "source_prefix_bytes_read": prefix_bytes_read,
        "source_prefix_sha256": prefix_digest.hexdigest(),
        "source_size_bytes": path.stat().st_size,
        "source_mtime_ns": path.stat().st_mtime_ns,
    }
    return frame, audit


def _gate(periods: dict[str, Any]) -> dict[str, Any]:
    all_period_nonnegative = all(
        periods[period]["delta"]["total_score"] >= -1e-12
        for period in ("full", "H1", "H2")
    )
    full_nmae_safe = periods["full"]["delta"]["one_minus_nmae"] >= -0.001
    group_safe = all(
        value >= -0.001
        for value in periods["full"]["delta"]["by_group_total_score"].values()
    )
    passed = bool(all_period_nonnegative and full_nmae_safe and group_safe)
    return {
        "all_period_total_deltas_nonnegative": all_period_nonnegative,
        "full_one_minus_nmae_delta_at_least_minus_0.001": full_nmae_safe,
        "all_group_full_total_deltas_at_least_minus_0.001": group_safe,
        "passed": passed,
    }


def _transfer_fold(
    *,
    labels: pd.DataFrame,
    baseline_path: Path,
    lowdim_path: Path,
    groups: list[str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    baseline = pd.read_parquet(baseline_path).loc[:, groups]
    lowdim = pd.read_parquet(lowdim_path)
    if not isinstance(lowdim.columns, pd.MultiIndex):
        raise ValueError("low-dimensional audit predictions need MultiIndex columns")
    identity = lowdim["identity"].loc[:, groups]
    mapped = lowdim["quantile_mapped"].loc[:, groups]
    if not baseline.index.equals(identity.index) or not identity.index.equals(mapped.index):
        raise ValueError("baseline and low-dimensional OOF indices differ")
    transfer = baseline.copy()
    for group in groups:
        capacity = CAPACITY_KWH[group]
        transfer[group] = np.clip(
            baseline[group] + mapped[group] - identity[group],
            0.0,
            1.02 * capacity,
        )
    truth = labels.loc[baseline.index, groups]
    periods: dict[str, Any] = {}
    for period, mask in _period_masks(baseline.index).items():
        baseline_score = _score(truth.iloc[mask], baseline.iloc[mask], groups)
        transfer_score = _score(truth.iloc[mask], transfer.iloc[mask], groups)
        periods[period] = {
            "baseline": baseline_score,
            "transfer": transfer_score,
            "delta": _delta(transfer_score, baseline_score),
        }
    components = pd.concat(
        {
            "baseline": baseline,
            "lowdim_raw": identity,
            "lowdim_mapped": mapped,
            "delta": mapped - identity,
            "transfer": transfer,
        },
        axis=1,
    )
    return {"periods": periods, "gate": _gate(periods)}, components


def _fit_final_lowdim(
    *,
    labels: pd.DataFrame,
    sample_times: pd.DatetimeIndex,
    parent_config: dict[str, Any],
    cache_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    features = list(parent_config["model_features"])
    mapped_features = list(parent_config["mapped_features"])
    grid_size = int(parent_config["adaptation"]["quantile_grid_size"])
    estimator_parameters = dict(parent_config["estimator"])
    estimator_parameters.pop("library")
    estimator_parameters.pop("target")

    raw_prediction = pd.DataFrame(index=sample_times, columns=GROUPS, dtype=float)
    mapped_prediction = pd.DataFrame(index=sample_times, columns=GROUPS, dtype=float)
    models: dict[str, Any] = {}
    audit: dict[str, Any] = {}
    for group in GROUPS:
        train_path = cache_dir / f"{group}_weather_train.parquet"
        test_path = cache_dir / f"{group}_weather_test.parquet"
        train = pd.read_parquet(train_path, columns=features)
        test = pd.read_parquet(test_path, columns=features)
        if not train.index.equals(labels.index):
            raise ValueError(f"{group} train cache is not aligned to labels")
        if not test.index.equals(sample_times):
            raise ValueError(f"{group} test cache is not aligned to sample")
        if tuple(train.columns) != tuple(test.columns):
            raise ValueError(f"{group} train/test feature schema mismatch")
        if not np.isfinite(train.to_numpy(dtype=float)).all():
            raise ValueError(f"{group} train features contain non-finite values")
        if not np.isfinite(test.to_numpy(dtype=float)).all():
            raise ValueError(f"{group} test features contain non-finite values")

        mapped_test = test.astype(np.float64).copy()
        mapping_audit: dict[str, Any] = {}
        for feature in mapped_features:
            values, feature_audit = _quantile_map(
                train[feature], test[feature], grid_size=grid_size
            )
            mapped_test.loc[:, feature] = values
            mapping_audit[feature] = feature_audit

        capacity = CAPACITY_KWH[group]
        target = labels[group] / capacity
        eligible = target.notna() & target.ge(0.10)
        model = LGBMRegressor(**estimator_parameters)
        model.fit(train.loc[eligible], target.loc[eligible])
        raw_prediction[group] = np.clip(
            model.predict(test) * capacity, 0.0, 1.02 * capacity
        )
        mapped_prediction[group] = np.clip(
            model.predict(mapped_test) * capacity, 0.0, 1.02 * capacity
        )
        models[group] = model
        audit[group] = {
            "training_rows": int(eligible.sum()),
            "fit_start": train.index.min().isoformat(),
            "fit_end": train.index.max().isoformat(),
            "application_start": test.index.min().isoformat(),
            "application_end": test.index.max().isoformat(),
            "mapping": mapping_audit,
        }
    return raw_prediction, mapped_prediction, models, audit


def _verify_submission(
    path: Path,
    expected: pd.DataFrame,
    sample: pd.DataFrame,
) -> dict[str, Any]:
    raw_prefix = path.read_bytes()[:3]
    if raw_prefix != b"\xef\xbb\xbf":
        raise ValueError(f"{path} is missing UTF-8 BOM")
    readback = pd.read_csv(path, encoding="utf-8-sig")
    if list(readback.columns) != list(sample.columns):
        raise ValueError(f"{path} schema differs from sample")
    if len(readback) != 8760:
        raise ValueError(f"{path} must contain 8760 rows")
    if not readback["forecast_id"].equals(sample["forecast_id"]):
        raise ValueError(f"{path} forecast_id differs from sample")
    if not readback["forecast_kst_dtm"].equals(sample["forecast_kst_dtm"]):
        raise ValueError(f"{path} forecast time differs from sample")
    values = readback.loc[:, GROUPS].to_numpy(dtype=float)
    expected_values = expected.loc[:, GROUPS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite values")
    for position, group in enumerate(GROUPS):
        if values[:, position].min() < 0.0:
            raise ValueError(f"{path} contains negative {group}")
        if values[:, position].max() > 1.02 * CAPACITY_KWH[group] + 1e-8:
            raise ValueError(f"{path} exceeds the clip for {group}")
    maximum_difference = float(np.max(np.abs(values - expected_values)))
    if maximum_difference > 0.500001e-6:
        raise ValueError(f"{path} readback differs by {maximum_difference}")
    return {
        "rows": len(readback),
        "utf8_sig": True,
        "sha256": _sha256(path),
        "max_readback_abs_diff": maximum_difference,
    }


def _write_submission(
    path: Path,
    sample: pd.DataFrame,
    prediction: pd.DataFrame,
) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    output = sample.copy()
    output.loc[:, list(GROUPS)] = prediction.loc[:, GROUPS].to_numpy(dtype=float)
    temporary = path.with_suffix(path.suffix + ".tmp")
    output.to_csv(
        temporary,
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )
    os.replace(temporary, path)
    return _verify_submission(path, prediction, sample)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preregister",
        type=Path,
        default=Path("configs/year_quantile_map_transfer_preregister.json"),
    )
    parser.add_argument(
        "--parent-preregister",
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
        default=Path("artifacts/postgate/year_quantile_map_transfer"),
    )
    args = parser.parse_args()

    preregister_sha256 = _sha256(args.preregister)
    parent_sha256 = _sha256(args.parent_preregister)
    if preregister_sha256 != EXPECTED_TRANSFER_PREREGISTER_SHA256:
        raise ValueError("transfer preregister hash mismatch")
    if parent_sha256 != EXPECTED_PARENT_PREREGISTER_SHA256:
        raise ValueError("parent preregister hash mismatch")
    transfer_config = json.loads(args.preregister.read_text(encoding="utf-8"))
    parent_config = json.loads(args.parent_preregister.read_text(encoding="utf-8"))

    labels_path = args.raw_dir / "train" / "train_labels.csv"
    audit_dir = Path("artifacts/postgate/year_quantile_map_audit")
    selection_baseline = Path(transfer_config["selection"]["baseline"])
    selection_lowdim = audit_dir / "oof" / "forward_2023__predictions.parquet"
    selection_index = pd.DatetimeIndex(pd.read_parquet(selection_lowdim).index)
    selection_labels, selection_label_io = _load_label_window_bounded(
        labels_path,
        expected_index=selection_index,
        groups=list(transfer_config["selection"]["groups"]),
    )
    selection_result, selection_components = _transfer_fold(
        labels=selection_labels,
        baseline_path=selection_baseline,
        lowdim_path=selection_lowdim,
        groups=list(transfer_config["selection"]["groups"]),
    )
    if not selection_result["gate"]["passed"]:
        rejection_path = args.output_dir / "selection_rejection.json"
        rejection_manifest_path = args.output_dir / "selection_rejection_manifest.json"
        rejection = {
            "schema_version": 1,
            "experiment": transfer_config["experiment"],
            "preregister_sha256": preregister_sha256,
            "parent_preregister_sha256": parent_sha256,
            "selection": selection_result,
            "decision": "rejected_at_2023_selection",
            "confirmation_2024_read": False,
            "test_weather_read_by_this_script": False,
            "test_predictions_generated_by_this_script": False,
            "sample_submission_read_by_this_script": False,
            "submission_csv_written": False,
            "physical_io_audit": {
                "selection_labels": selection_label_io,
                "confirmation_2024_label_rows_materialized": 0,
                "confirmation_2024_prediction_artifacts_opened": 0,
                "sample_submission_rows_materialized": 0,
                "test_weather_files_opened": [],
            },
        }
        _write_json(rejection_path, rejection)
        rejection_manifest = {
            "schema_version": 1,
            "artifact_type": "baram_year_quantile_map_delta_transfer_rejection",
            "inputs": [
                {
                    "path": str(path.resolve()),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in (
                    args.preregister,
                    args.parent_preregister,
                    selection_baseline,
                    selection_lowdim,
                )
            ]
            + [
                {
                    "path": str(labels_path.resolve()),
                    "size_bytes": labels_path.stat().st_size,
                    "mtime_ns": labels_path.stat().st_mtime_ns,
                    "whole_file_sha256_computed_in_rejected_stage": False,
                    "bounded_prefix_bytes_read": selection_label_io[
                        "source_prefix_bytes_read"
                    ],
                    "bounded_prefix_sha256": selection_label_io[
                        "source_prefix_sha256"
                    ],
                    "bounded_materialization": selection_label_io,
                }
            ],
            "outputs": [
                {
                    "path": str(rejection_path.resolve()),
                    "size_bytes": rejection_path.stat().st_size,
                    "sha256": _sha256(rejection_path),
                }
            ],
            "selection_gate": selection_result["gate"],
        }
        _write_json(rejection_manifest_path, rejection_manifest)
        print("selection", json.dumps(selection_result["gate"], sort_keys=True))
        for period, payload in selection_result["periods"].items():
            print("selection", period, payload["delta"])
        print("rejected before confirmation/test construction")
        return

    confirmation_baseline = Path(transfer_config["confirmation"]["baseline"])
    confirmation_lowdim = audit_dir / "oof" / "forward_2024__predictions.parquet"
    confirmation_index = pd.DatetimeIndex(pd.read_parquet(confirmation_lowdim).index)
    confirmation_labels, confirmation_label_io = _load_label_window_bounded(
        labels_path,
        expected_index=confirmation_index,
        groups=list(transfer_config["confirmation"]["groups"]),
    )
    confirmation_result, confirmation_components = _transfer_fold(
        labels=confirmation_labels,
        baseline_path=confirmation_baseline,
        lowdim_path=confirmation_lowdim,
        groups=list(transfer_config["confirmation"]["groups"]),
    )
    if not confirmation_result["gate"]["passed"]:
        raise RuntimeError(
            "fixed 2024 confirmation failed; 2025 construction is forbidden"
        )

    labels = pd.read_csv(
        labels_path, encoding="utf-8-sig", parse_dates=["kst_dtm"]
    ).set_index("kst_dtm")
    sample_path = args.raw_dir / "sample_submission.csv"
    sample = pd.read_csv(sample_path, encoding="utf-8-sig", dtype={"forecast_id": str})
    sample_times = pd.DatetimeIndex(
        pd.to_datetime(sample["forecast_kst_dtm"], errors="raise"),
        name="forecast_kst_dtm",
    )
    if len(sample) != 8760 or not sample_times.is_unique or not sample_times.is_monotonic_increasing:
        raise ValueError("invalid sample timeline")

    raw_prediction, mapped_prediction, models, final_audit = _fit_final_lowdim(
        labels=labels,
        sample_times=sample_times,
        parent_config=parent_config,
        cache_dir=args.cache_dir,
    )
    delta = mapped_prediction - raw_prediction
    v4_path = Path("artifacts/final_cf_fix/corrected_recent_v4.csv")
    v4 = pd.read_csv(v4_path, encoding="utf-8-sig", dtype={"forecast_id": str})
    if not v4["forecast_id"].equals(sample["forecast_id"]):
        raise ValueError("v4 forecast IDs differ from sample")
    if not v4["forecast_kst_dtm"].equals(sample["forecast_kst_dtm"]):
        raise ValueError("v4 times differ from sample")
    v4_prediction = pd.DataFrame(
        v4.loc[:, GROUPS].to_numpy(dtype=float), index=sample_times, columns=GROUPS
    )
    v4_transfer = pd.DataFrame(index=sample_times, columns=GROUPS, dtype=float)
    for group in GROUPS:
        capacity = CAPACITY_KWH[group]
        v4_transfer[group] = np.clip(
            v4_prediction[group] + delta[group], 0.0, 1.02 * capacity
        )

    output_paths = {
        "selection_components": args.output_dir / "oof" / "selection_2023_components.parquet",
        "confirmation_components": args.output_dir / "oof" / "confirmation_2024_components.parquet",
        "final_components": args.output_dir / "predictions" / "final_2025_components.parquet",
        "models": args.output_dir / "models" / "lowdim_l1_models.joblib",
        "direct_csv": args.output_dir / "direct_year_qm_l1_2025.csv",
        "v4_transfer_csv": args.output_dir / "v4_year_qm_delta_transfer_2025.csv",
        "results": args.output_dir / "results.json",
        "manifest": args.output_dir / "manifest.json",
    }
    for key, path in output_paths.items():
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {key}: {path}")
    for key in ("selection_components", "confirmation_components", "final_components", "models"):
        output_paths[key].parent.mkdir(parents=True, exist_ok=True)

    selection_components.to_parquet(output_paths["selection_components"], index=True)
    confirmation_components.to_parquet(output_paths["confirmation_components"], index=True)
    final_components = pd.concat(
        {
            "lowdim_raw": raw_prediction,
            "lowdim_mapped": mapped_prediction,
            "delta": delta,
            "v4_baseline": v4_prediction,
            "v4_delta_transfer": v4_transfer,
        },
        axis=1,
    )
    final_components.to_parquet(output_paths["final_components"], index=True)
    joblib.dump(models, output_paths["models"])
    direct_csv_audit = _write_submission(
        output_paths["direct_csv"], sample, mapped_prediction
    )
    transfer_csv_audit = _write_submission(
        output_paths["v4_transfer_csv"], sample, v4_transfer
    )

    delta_stats = {
        group: {
            "mean_kwh": float(delta[group].mean()),
            "median_kwh": float(delta[group].median()),
            "min_kwh": float(delta[group].min()),
            "max_kwh": float(delta[group].max()),
            "mean_capacity_fraction": float(delta[group].mean() / CAPACITY_KWH[group]),
            "changed_rows": int(delta[group].ne(0.0).sum()),
        }
        for group in GROUPS
    }
    results = {
        "schema_version": 1,
        "experiment": transfer_config["experiment"],
        "preregister_sha256": preregister_sha256,
        "parent_preregister_sha256": parent_sha256,
        "public_scores_used": False,
        "test_labels_seen": False,
        "test_covariates_previously_aggregated_for_shift_audit": True,
        "preexisting_v4_test_prediction_used_only_as_fixed_baseline": True,
        "selection": selection_result,
        "confirmation": confirmation_result,
        "both_strict_gates_passed": True,
        "final_fit_audit": final_audit,
        "final_delta_stats": delta_stats,
        "outputs": {
            "direct_mapped": {
                "path": str(output_paths["direct_csv"].resolve()),
                "audit": direct_csv_audit,
                "status": "deployable_lowdim_control",
            },
            "v4_delta_transfer": {
                "path": str(output_paths["v4_transfer_csv"].resolve()),
                "audit": transfer_csv_audit,
                "status": "deployable_cross_baseline_transfer",
                "strict_adoption": False,
                "risk": transfer_config["final_outputs"]["v4_transfer_risk"],
            },
        },
    }
    _write_json(output_paths["results"], results)

    input_paths = [
        args.preregister,
        args.parent_preregister,
        labels_path,
        sample_path,
        selection_baseline,
        selection_lowdim,
        confirmation_baseline,
        confirmation_lowdim,
        v4_path,
    ]
    for group in GROUPS:
        input_paths.extend(
            [
                args.cache_dir / f"{group}_weather_train.parquet",
                args.cache_dir / f"{group}_weather_test.parquet",
            ]
        )
    manifest_outputs = [
        output_paths[key]
        for key in (
            "selection_components",
            "confirmation_components",
            "final_components",
            "models",
            "direct_csv",
            "v4_transfer_csv",
            "results",
        )
    ]
    manifest = {
        "schema_version": 1,
        "artifact_type": "baram_year_quantile_map_delta_transfer",
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
            for path in manifest_outputs
        ],
        "selection_gate": selection_result["gate"],
        "confirmation_gate": confirmation_result["gate"],
        "v4_cross_baseline_transfer_strict_adoption": False,
    }
    _write_json(output_paths["manifest"], manifest)

    print("selection", json.dumps(selection_result["gate"], sort_keys=True))
    for period, payload in selection_result["periods"].items():
        print("selection", period, payload["delta"])
    print("confirmation", json.dumps(confirmation_result["gate"], sort_keys=True))
    for period, payload in confirmation_result["periods"].items():
        print("confirmation", period, payload["delta"])
    print("delta_stats", json.dumps(delta_stats, indent=2, sort_keys=True))
    print("direct_csv", direct_csv_audit)
    print("v4_transfer_csv", transfer_csv_audit)


if __name__ == "__main__":
    main()

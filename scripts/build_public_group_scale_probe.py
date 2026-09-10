"""Build three non-submitting, one-group-only Public-adaptive scale probes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metric import CAPACITY_KWH, TARGET_COLS, group_metrics, score_details  # noqa: E402
from src.public_group_scale_probe import (  # noqa: E402
    macro_separability_residuals,
    scale_one_group,
)
from src.public_scale_probe import scale_predictions, validate_prediction_frame  # noqa: E402


CONFIG_SHA256 = "db6011f63ab18461c69df5de79e956bfe137001b7bb58b048f3652d33723b65e"
EXPECTED_ROWS = 8760


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/public_group_scale_probe_20260808.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/postgate/public_group_scale_probe"),
    )
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, engine="pyarrow", compression="zstd", index=True)
    temporary.replace(path)


def load_submission(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    submission = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"forecast_id": "string", "forecast_kst_dtm": "string"},
    )
    expected = ("forecast_id", "forecast_kst_dtm", *TARGET_COLS)
    if tuple(submission.columns) != expected or len(submission) != EXPECTED_ROWS:
        raise AssertionError(f"submission contract changed: {path}")
    index = pd.DatetimeIndex(
        pd.to_datetime(submission["forecast_kst_dtm"], errors="raise"),
        name="forecast_kst_dtm",
    )
    prediction = submission.loc[:, TARGET_COLS].astype(np.float64)
    prediction.index = index
    validate_prediction_frame(prediction, name=str(path))
    return submission, prediction


def write_submission(
    path: Path,
    identifiers: pd.DataFrame,
    prediction: pd.DataFrame,
    base_prediction: pd.DataFrame,
    *,
    scaled_group: str,
) -> dict[str, Any]:
    output = identifiers.loc[:, ["forecast_id", "forecast_kst_dtm"]].copy()
    for group in TARGET_COLS:
        output[group] = prediction[group].to_numpy(dtype=np.float64)
    temporary = path.with_suffix(path.suffix + ".tmp")
    output.to_csv(
        temporary,
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
        lineterminator="\n",
    )
    temporary.replace(path)
    raw = path.read_bytes()
    if raw[:3] != b"\xef\xbb\xbf":
        raise AssertionError("UTF-8-SIG BOM missing")
    observed = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"forecast_id": "string", "forecast_kst_dtm": "string"},
    )
    if not observed.loc[:, ["forecast_id", "forecast_kst_dtm"]].equals(
        identifiers.loc[:, ["forecast_id", "forecast_kst_dtm"]]
    ):
        raise AssertionError("CSV identifiers or timestamps changed")
    observed_values = observed.loc[:, TARGET_COLS].to_numpy(dtype=np.float64)
    expected_values = prediction.loc[:, TARGET_COLS].to_numpy(dtype=np.float64)
    if not np.isfinite(observed_values).all():
        raise AssertionError("CSV contains non-finite values")
    maximum_difference = float(np.max(np.abs(observed_values - expected_values)))
    if maximum_difference > 5.1e-7:
        raise AssertionError("CSV six-decimal roundtrip tolerance exceeded")
    identity: dict[str, bool] = {}
    for position, group in enumerate(TARGET_COLS):
        capacity = CAPACITY_KWH[group]
        if np.any(observed_values[:, position] < 0.0) or np.any(
            observed_values[:, position] > 1.02 * capacity + 1e-9
        ):
            raise AssertionError(f"{group} violates physical bounds")
        if group != scaled_group:
            identity[group] = bool(
                np.array_equal(
                    observed_values[:, position],
                    base_prediction[group].to_numpy(dtype=np.float64),
                )
            )
            if not identity[group]:
                raise AssertionError(f"{group} changed in CSV identity column")
    return {
        "rows": int(len(observed)),
        "utf8_sig_bom": True,
        "ids_exact": True,
        "timestamps_exact": True,
        "finite": True,
        "physical_bounds": True,
        "max_abs_csv_roundtrip_difference": maximum_difference,
        "non_target_groups_equal_base_after_roundtrip": identity,
    }


def group_delta(
    labels: pd.Series,
    prediction: pd.Series,
    *,
    group: str,
    factor: float,
) -> dict[str, float]:
    capacity = CAPACITY_KWH[group]
    baseline = group_metrics(labels, prediction, capacity, group_name=group)
    candidate_values = np.clip(prediction.to_numpy(dtype=float) * factor, 0.0, 1.02 * capacity)
    candidate = group_metrics(labels, candidate_values, capacity, group_name=group)
    return {
        "total": float(
            0.5
            * (
                candidate.one_minus_nmae
                - baseline.one_minus_nmae
                + candidate.ficr
                - baseline.ficr
            )
        ),
        "one_minus_nmae": float(candidate.one_minus_nmae - baseline.one_minus_nmae),
        "ficr": float(candidate.ficr - baseline.ficr),
    }


def historical_order_evidence(config: Mapping[str, Any]) -> dict[str, Any]:
    evidence = config["prepublic_group_order_evidence"]
    label_path = resolve_path(evidence["labels"])
    prediction_2023_path = resolve_path(evidence["prediction_2023"])
    prediction_2024_path = resolve_path(evidence["prediction_2024"])
    expected_hashes = {
        label_path: str(evidence["labels_sha256"]),
        prediction_2023_path: str(evidence["prediction_2023_sha256"]),
        prediction_2024_path: str(evidence["prediction_2024_sha256"]),
    }
    for path, expected in expected_hashes.items():
        if sha256_file(path) != expected:
            raise AssertionError(f"historical-order input hash changed: {path}")
    labels = pd.read_csv(label_path, encoding="utf-8-sig", parse_dates=["kst_dtm"]).set_index("kst_dtm")
    labels.index = pd.DatetimeIndex(labels.index, name="forecast_kst_dtm")
    predictions = {
        2023: pd.read_parquet(prediction_2023_path, engine="pyarrow"),
        2024: pd.read_parquet(prediction_2024_path, engine="pyarrow"),
    }
    factor = float(config["factor"])
    records: dict[str, Any] = {}
    ranking_rows: list[tuple[float, float, str]] = []
    for group in TARGET_COLS:
        segments: dict[str, Any] = {}
        deltas: list[float] = []
        for year, frame in predictions.items():
            frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
            series = frame[group].dropna().astype(float)
            middle = pd.Timestamp(year=year, month=7, day=1, hour=0)
            masks = (
                {"H2": np.asarray(series.index > middle)}
                if group == "kpx_group_3" and year == 2023
                else {
                    "full": np.ones(len(series), dtype=bool),
                    "H1": np.asarray(series.index <= middle),
                    "H2": np.asarray(series.index > middle),
                }
            )
            for segment, mask in masks.items():
                delta = group_delta(
                    labels.loc[series.index[mask], group],
                    series.iloc[np.flatnonzero(mask)],
                    group=group,
                    factor=factor,
                )
                segments[f"{year}_{segment}"] = delta
                deltas.append(delta["total"])
        minimum = float(min(deltas))
        mean = float(np.mean(deltas))
        frozen = evidence["frozen_values"][group]
        if not np.isclose(minimum, float(frozen["minimum_segment_delta"]), rtol=0.0, atol=1e-15):
            raise AssertionError(f"{group} frozen minimum changed")
        if not np.isclose(mean, float(frozen["mean_segment_delta"]), rtol=0.0, atol=1e-15):
            raise AssertionError(f"{group} frozen mean changed")
        records[group] = {
            "segments": segments,
            "minimum_segment_delta": minimum,
            "mean_segment_delta": mean,
        }
        ranking_rows.append((minimum, mean, group))
    rank = [group for _, _, group in sorted(ranking_rows, reverse=True)]
    if rank != list(evidence["frozen_rank"]):
        raise AssertionError("historical group order changed")
    return {
        "public_results_used": False,
        "factor": factor,
        "ranking_rule": evidence["ranking_rule"],
        "rank": rank,
        "first_group_only_probe_if_activation_gate_passes": rank[0],
        "records": records,
        "inputs": [record(path) for path in expected_hashes],
        "warning": evidence["warning"],
    }


def preflight(out_dir: Path, output_names: Mapping[str, str]) -> None:
    expected = {"preregister.json", "results.json", "manifest.json"}
    for stem in output_names.values():
        expected.add(f"{stem}.csv")
        expected.add(f"{stem}.parquet")
    if out_dir.exists():
        existing = list(out_dir.iterdir())
        if existing:
            raise FileExistsError(
                "refusing to overwrite group-scale probe directory: "
                + ", ".join(str(path) for path in existing)
            )
    unexpected_existing = [out_dir / name for name in expected if (out_dir / name).exists()]
    if unexpected_existing:
        raise FileExistsError(f"refusing to overwrite: {unexpected_existing}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = resolve_path(args.config)
    out_dir = args.out_dir.resolve()
    if sha256_file(config_path) != CONFIG_SHA256:
        raise AssertionError("group-scale config hash changed")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not config["public_adaptive"] or not config["selection_unsafe"]:
        raise AssertionError("unsafe/adaptive flags must remain true")
    if config["private_champion"] is not False:
        raise AssertionError("private_champion must remain false")
    if float(config["factor"]) != 0.92:
        raise AssertionError("factor must remain exactly 0.92")
    output_names = config["output_names"]
    if tuple(output_names) != TARGET_COLS:
        raise AssertionError("output group order changed")
    preflight(out_dir, output_names)
    out_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config_path, out_dir / "preregister.json")

    base_path = resolve_path(config["base_submission"])
    sample_path = resolve_path(config["sample_submission"])
    if sha256_file(base_path) != config["base_submission_sha256"]:
        raise AssertionError("byte-pinned base submission changed")
    if sha256_file(sample_path) != config["sample_submission_sha256"]:
        raise AssertionError("sample submission changed")
    base_submission, base_prediction = load_submission(base_path)
    sample, _ = load_submission(sample_path)
    if not base_submission.loc[:, ["forecast_id", "forecast_kst_dtm"]].equals(
        sample.loc[:, ["forecast_id", "forecast_kst_dtm"]]
    ):
        raise AssertionError("base IDs/times differ from official sample")

    evidence = historical_order_evidence(config)
    factor = float(config["factor"])
    probes: dict[str, Any] = {}
    output_paths: list[Path] = []
    predictions: dict[str, pd.DataFrame] = {}
    for group in TARGET_COLS:
        prediction = scale_one_group(
            base_prediction,
            group,
            factor=factor,
            upper_capacity_fraction=float(config["upper_capacity_fraction"]),
        )
        predictions[group] = prediction
        stem = str(output_names[group])
        parquet_path = out_dir / f"{stem}.parquet"
        csv_path = out_dir / f"{stem}.csv"
        atomic_parquet(parquet_path, prediction)
        reopened = pd.read_parquet(parquet_path, engine="pyarrow")
        reopened.index = pd.DatetimeIndex(reopened.index, name="forecast_kst_dtm")
        parquet_identity = {
            identity_group: bool(
                np.array_equal(
                    reopened[identity_group].to_numpy(dtype=np.float64),
                    base_prediction[identity_group].to_numpy(dtype=np.float64),
                )
            )
            for identity_group in TARGET_COLS
            if identity_group != group
        }
        if not all(parquet_identity.values()):
            raise AssertionError("Parquet non-target identity changed")
        csv_audit = write_submission(
            csv_path,
            sample,
            prediction,
            base_prediction,
            scaled_group=group,
        )
        output_paths.extend((parquet_path, csv_path))
        probes[group] = {
            "scaled_group": group,
            "factor": factor,
            "csv": record(csv_path),
            "parquet": record(parquet_path),
            "parquet_non_target_value_bits_exact": parquet_identity,
            "csv_audit": csv_audit,
            "delta_vs_base": {
                target: {
                    "mean_kwh": float((prediction[target] - base_prediction[target]).mean()),
                    "max_abs_kwh": float((prediction[target] - base_prediction[target]).abs().max()),
                }
                for target in TARGET_COLS
            },
        }

    # Algebraic diagnostic on one fully labelled historical application year.
    label_path = resolve_path(config["prepublic_group_order_evidence"]["labels"])
    labels = pd.read_csv(label_path, encoding="utf-8-sig", parse_dates=["kst_dtm"]).set_index("kst_dtm")
    labels.index = pd.DatetimeIndex(labels.index, name="forecast_kst_dtm")
    prediction_2024 = pd.read_parquet(
        resolve_path(config["prepublic_group_order_evidence"]["prediction_2024"]),
        engine="pyarrow",
    ).astype(float)
    prediction_2024.index = pd.DatetimeIndex(prediction_2024.index, name="forecast_kst_dtm")
    labels_2024 = labels.reindex(prediction_2024.index)
    base_metrics = score_details(labels_2024, prediction_2024)
    global_metrics = score_details(labels_2024, scale_predictions(prediction_2024, factor))
    group_metrics_map = {
        group: score_details(labels_2024, scale_one_group(prediction_2024, group, factor=factor))
        for group in TARGET_COLS
    }
    separability = macro_separability_residuals(
        base={
            "score": base_metrics.total_score,
            "one_minus_nmae": base_metrics.one_minus_nmae,
            "ficr": base_metrics.ficr,
        },
        global_scaled={
            "score": global_metrics.total_score,
            "one_minus_nmae": global_metrics.one_minus_nmae,
            "ficr": global_metrics.ficr,
        },
        group_only={
            group: {
                "score": metric.total_score,
                "one_minus_nmae": metric.one_minus_nmae,
                "ficr": metric.ficr,
            }
            for group, metric in group_metrics_map.items()
        },
    )
    if max(abs(item["residual"]) for item in separability.values()) > 5e-16:
        raise AssertionError("macro separability numerical audit failed")

    protocol = dict(config["sequential_protocol"])
    if protocol["global_092_result_available_at_build_time"] is not False:
        raise AssertionError("global result availability changed")
    if protocol["automatic_next_file_when_global_result_missing"] is not None:
        raise AssertionError("automatic recommendation must remain null")
    results = {
        "schema_version": 1,
        "artifact_type": "baram_public_adaptive_group_scale_probe_pack",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "public_adaptive": True,
        "selection_unsafe": True,
        "private_champion": False,
        "submission_performed": False,
        "base_submission": record(base_path),
        "sample_submission": record(sample_path),
        "factor": factor,
        "formula": config["formula"],
        "sequential_protocol": protocol,
        "current_automatic_recommendation": None,
        "reason_no_recommendation": "Global 0.92 Public result has not been supplied; activation gate is unevaluated.",
        "historical_group_order_evidence": evidence,
        "macro_separability": {
            "identity": "delta_global = delta_g1_only + delta_g2_only + delta_g3_only",
            "components": ["score", "one_minus_nmae", "ficr"],
            "historical_2024_numeric_audit": separability,
            "maximum_absolute_residual": float(max(abs(item["residual"]) for item in separability.values())),
            "inference": "Submit two activated group-only probes; infer the third group delta by subtraction from the observed global delta.",
        },
        "probes": probes,
    }
    results_path = out_dir / "results.json"
    atomic_json(results_path, results)
    output_paths.append(results_path)

    source_paths = [
        Path(__file__).resolve(),
        PROJECT_ROOT / "src/public_group_scale_probe.py",
        PROJECT_ROOT / "src/public_scale_probe.py",
        PROJECT_ROOT / "src/metric.py",
        PROJECT_ROOT / "tests/test_public_group_scale_probe.py",
    ]
    input_paths = [
        config_path,
        base_path,
        sample_path,
        label_path,
        resolve_path(config["prepublic_group_order_evidence"]["prediction_2023"]),
        resolve_path(config["prepublic_group_order_evidence"]["prediction_2024"]),
        resolve_path(protocol["global_probe"]),
    ]
    manifest = {
        "schema_version": 1,
        "artifact_type": "baram_public_adaptive_group_scale_probe_manifest",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "public_adaptive": True,
        "selection_unsafe": True,
        "private_champion": False,
        "submission_performed": False,
        "overwrite_guard": "new non-empty output directories are refused; no overwrite flag exists",
        "config_sha256": CONFIG_SHA256,
        "inputs": [record(path) for path in input_paths],
        "sources": [record(path) for path in source_paths],
        "outputs": [record(path) for path in [out_dir / "preregister.json", *output_paths]],
        "current_automatic_recommendation": None,
    }
    manifest_path = out_dir / "manifest.json"
    atomic_json(manifest_path, manifest)
    print(json.dumps({"out_dir": str(out_dir), "probe_count": 3, "automatic_recommendation": None}, ensure_ascii=False))
    print(f"manifest_sha256={sha256_file(manifest_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

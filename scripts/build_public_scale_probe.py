"""Build two controlled scale probes from the best observed Public submission.

This runner intentionally consumes the immutable user-reported Public log.  It
does not overwrite that log or any existing candidate.  Its outputs are marked
Public-adaptive and selection-unsafe because the scales were chosen after
leaderboard feedback; they are not eligible to become a Private champion based
on this diagnostic alone.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metric import CAPACITY_KWH, TARGET_COLS, group_metrics, score_details
from src.public_scale_probe import (
    fit_oof_scale_to_public,
    scale_predictions,
    validate_prediction_frame,
    validate_public_triplet,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/public_scale_probe_20260808.json"),
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=Path(r"data/local/open")
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/postgate/public_scale_probe"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"{path}: JSON root must be an object")
    return value


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_parquet(path: Path, prediction: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    prediction.to_parquet(temporary, compression="zstd", index=True)
    temporary.replace(path)


def _read_oof(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path).astype(float)
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    validate_prediction_frame(frame, name=str(path))
    return frame


def _read_submission(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    submission = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"forecast_id": "string", "forecast_kst_dtm": "string"},
    )
    expected = ("forecast_id", "forecast_kst_dtm", *TARGET_COLS)
    if tuple(submission.columns) != expected:
        raise ValueError(f"{path}: submission schema differs from {expected}")
    if len(submission) != 8_760:
        raise ValueError(f"{path}: expected 8,760 rows")
    times = pd.DatetimeIndex(
        pd.to_datetime(submission["forecast_kst_dtm"], errors="raise"),
        name="forecast_kst_dtm",
    )
    prediction = submission.loc[:, TARGET_COLS].astype(float)
    prediction.index = times
    validate_prediction_frame(prediction, name=str(path))
    return submission, prediction


def _atomic_submission(
    path: Path,
    identifiers: pd.DataFrame,
    prediction: pd.DataFrame,
) -> dict[str, Any]:
    output = identifiers.loc[:, ["forecast_id", "forecast_kst_dtm"]].copy()
    for group in TARGET_COLS:
        output[group] = prediction[group].to_numpy(dtype=float)
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
        raise AssertionError(f"{path}: UTF-8-SIG BOM missing")
    observed = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"forecast_id": "string", "forecast_kst_dtm": "string"},
    )
    if not observed.loc[:, ["forecast_id", "forecast_kst_dtm"]].equals(
        identifiers.loc[:, ["forecast_id", "forecast_kst_dtm"]]
    ):
        raise AssertionError(f"{path}: IDs or timestamps changed")
    observed_values = observed.loc[:, TARGET_COLS].to_numpy(dtype=float)
    expected_values = prediction.loc[:, TARGET_COLS].to_numpy(dtype=float)
    maximum_difference = float(np.max(np.abs(observed_values - expected_values)))
    if maximum_difference > 5.1e-7:
        raise AssertionError(f"{path}: six-decimal roundtrip tolerance exceeded")
    if not np.isfinite(observed_values).all():
        raise AssertionError(f"{path}: non-finite values after CSV readback")
    for position, group in enumerate(TARGET_COLS):
        if np.any(observed_values[:, position] < 0.0) or np.any(
            observed_values[:, position]
            > CAPACITY_KWH[group] * 1.02 + 1e-9
        ):
            raise AssertionError(f"{path}: {group} violates physical bounds")
    return {
        "rows": int(len(observed)),
        "utf8_sig_bom": True,
        "ids_exact": True,
        "timestamps_exact": True,
        "finite": True,
        "physical_bounds": True,
        "max_abs_csv_roundtrip_difference": maximum_difference,
    }


def _preflight(out_dir: Path, scales: list[float], overwrite: bool) -> None:
    names = {"results.json", "manifest.json"}
    for scale in scales:
        tag = f"{int(round(scale * 100)):03d}"
        names.add(f"corrected_recent_v4_scale_{tag}_2025.csv")
        names.add(f"corrected_recent_v4_scale_{tag}_2025.parquet")
    existing = [out_dir / name for name in sorted(names) if (out_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite existing public probe artifacts: "
            + ", ".join(map(str, existing))
        )


def _public_records_by_file(public_log: dict[str, Any]) -> dict[str, dict[str, Any]]:
    submissions = public_log.get("submissions")
    if not isinstance(submissions, list) or len(submissions) != 3:
        raise ValueError("Public log must contain exactly the three supplied submissions")
    records: dict[str, dict[str, Any]] = {}
    for raw_record in submissions:
        if not isinstance(raw_record, dict):
            raise TypeError("Public submission record must be an object")
        record = dict(raw_record)
        file_name = str(record["file"])
        validate_public_triplet(
            score=float(record["score"]),
            one_minus_nmae=float(record["one_minus_nmae"]),
            ficr=float(record["ficr"]),
        )
        source = _resolve_project_path(file_name)
        if not source.is_file():
            raise FileNotFoundError(source)
        observed_hash = _sha256(source)
        if observed_hash != str(record["sha256"]):
            raise AssertionError(f"Public log hash mismatch for {source}")
        records[file_name] = record
    return records


def _group_components(labels: pd.DataFrame, prediction: pd.DataFrame, group: str) -> dict[str, float]:
    metrics = group_metrics(
        labels[group], prediction[group], CAPACITY_KWH[group], group_name=group
    )
    return {
        "one_minus_nmae": float(metrics.one_minus_nmae),
        "ficr": float(metrics.ficr),
        "score": float(0.5 * (metrics.one_minus_nmae + metrics.ficr)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = _resolve_project_path(args.config)
    raw_dir = args.raw_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    config = _read_json(config_path)
    if int(config.get("schema_version", -1)) != 1:
        raise ValueError("unsupported config schema_version")
    if not config.get("public_adaptive") or not config.get("selection_unsafe"):
        raise ValueError("config must explicitly mark this experiment unsafe/adaptive")
    if config.get("private_champion") is not False:
        raise ValueError("public scale probes cannot be marked as a Private champion")
    scales = [float(value) for value in config["probe_scales"]]
    if scales != [0.92, 0.95]:
        raise ValueError("canonical probe scales must remain exactly [0.92, 0.95]")
    _preflight(out_dir, scales, bool(args.overwrite))
    out_dir.mkdir(parents=True, exist_ok=True)

    base_path = _resolve_project_path(config["base_submission"])
    public_log_path = _resolve_project_path(config["public_log"])
    label_path = raw_dir / "train/train_labels.csv"
    sample_path = raw_dir / "sample_submission.csv"
    required = [base_path, public_log_path, label_path, sample_path, config_path]
    oof_paths = {
        file_name: _resolve_project_path(path)
        for file_name, path in config["oof_analogs"].items()
    }
    required.extend(oof_paths.values())
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing scale-probe inputs: {missing}")
    if _sha256(base_path) != str(config["base_submission_sha256"]):
        raise AssertionError("base submission hash differs from locked config")

    public_log = _read_json(public_log_path)
    public_records = _public_records_by_file(public_log)
    if set(public_records) != set(oof_paths):
        raise ValueError("OOF analog mapping must cover exactly the three Public files")

    labels = pd.read_csv(
        label_path, encoding="utf-8-sig", parse_dates=["kst_dtm"]
    ).set_index("kst_dtm")
    labels.index = pd.DatetimeIndex(labels.index, name="forecast_kst_dtm")
    sample = pd.read_csv(
        sample_path,
        encoding="utf-8-sig",
        dtype={"forecast_id": "string", "forecast_kst_dtm": "string"},
    )
    base_submission, base_prediction = _read_submission(base_path)
    if not base_submission.loc[:, ["forecast_id", "forecast_kst_dtm"]].equals(
        sample.loc[:, ["forecast_id", "forecast_kst_dtm"]]
    ):
        raise AssertionError("base submission IDs/timestamps differ from official sample")

    grid = config["scale_fit_grid"]
    joint_fits: dict[str, Any] = {}
    loaded_oof: dict[str, pd.DataFrame] = {}
    for file_name, oof_path in oof_paths.items():
        oof = _read_oof(oof_path)
        oof_labels = labels.reindex(oof.index)
        if not oof_labels.index.equals(oof.index):
            raise AssertionError(f"label reindex failed for {oof_path}")
        loaded_oof[file_name] = oof
        joint_fits[file_name] = fit_oof_scale_to_public(
            oof_labels,
            oof,
            public_records[file_name],
            scale_min=float(grid["minimum"]),
            scale_max=float(grid["maximum"]),
            scale_step=float(grid["step"]),
        )

    ficr_file = "artifacts/postgate/ficr_bayes_decision/ficr_bayes_decision_2025.csv"
    prob_file = "artifacts/postgate/probabilistic_ficr/prob_stable_g1g3_2025.csv"
    ficr_oof = loaded_oof[ficr_file]
    prob_oof = loaded_oof[prob_file]
    common_index = ficr_oof.index
    if not common_index.equals(prob_oof.index):
        raise ValueError("probabilistic and FICR OOF indices differ")
    common_labels = labels.reindex(common_index)
    public_g3_delta = {
        "one_minus_nmae": float(
            3.0
            * (
                public_records[prob_file]["one_minus_nmae"]
                - public_records[ficr_file]["one_minus_nmae"]
            )
        ),
        "ficr": float(
            3.0
            * (public_records[prob_file]["ficr"] - public_records[ficr_file]["ficr"])
        ),
        "score": float(
            3.0
            * (public_records[prob_file]["score"] - public_records[ficr_file]["score"])
        ),
    }
    oof_g3_prob = _group_components(common_labels, prob_oof, "kpx_group_3")
    oof_g3_ficr = _group_components(common_labels, ficr_oof, "kpx_group_3")
    oof_g3_delta = {
        key: float(oof_g3_prob[key] - oof_g3_ficr[key])
        for key in ("one_minus_nmae", "ficr", "score")
    }

    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "baram_public_adaptive_scale_probe",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "public_adaptive": True,
        "selection_unsafe": True,
        "private_champion": False,
        "untouched_validation_claim": False,
        "base_submission": _record(base_path),
        "public_feedback_snapshot": {
            "reported_at_kst": public_log.get("reported_at_kst"),
            "source": public_log.get("source"),
            "current_best_public": public_log.get("current_best_public"),
            "leader": public_log.get("leader"),
            "submissions": public_log["submissions"],
            "submissions_used_today": public_log.get("submissions_used_today"),
            "submissions_remaining_today": public_log.get("submissions_remaining_today"),
        },
        "joint_fit_diagnostics": joint_fits,
        "confounding_audit": {
            "isolated_change": "prob_stable minus ficr_bayes; only kpx_group_3 differs",
            "public_group3_delta": public_g3_delta,
            "oof_2024_unscaled_group3_delta": oof_g3_delta,
            "warning": (
                "The isolated group-3 NMAE delta reverses sign between consumed "
                "2024 OOF and Public feedback, so the scalar analogy is not a "
                "causal or group-identifiable calibration model."
            ),
        },
        "probe_formula": "clip(scale * corrected_recent_v4, 0, 1.02*group_capacity)",
        "probe_scales": scales,
        "probes": {},
        "warning": config["selection_note"],
    }

    outputs: list[Path] = []
    v4_file = "artifacts/final_cf_fix/corrected_recent_v4.csv"
    v4_oof = loaded_oof[v4_file]
    v4_labels = labels.reindex(v4_oof.index)
    inferred_public_scale = float(
        joint_fits[v4_file]["joint_fit"]["scale"]
    )
    for scale in scales:
        tag = f"{int(round(scale * 100)):03d}"
        prediction = scale_predictions(
            base_prediction,
            scale,
            upper_capacity_fraction=float(config["upper_capacity_fraction"]),
        )
        parquet_path = out_dir / f"corrected_recent_v4_scale_{tag}_2025.parquet"
        csv_path = out_dir / f"corrected_recent_v4_scale_{tag}_2025.csv"
        _atomic_parquet(parquet_path, prediction)
        csv_audit = _atomic_submission(csv_path, sample, prediction)
        outputs.extend([parquet_path, csv_path])
        delta = prediction - base_prediction
        effective_scale = inferred_public_scale * scale
        analogy = score_details(v4_labels, v4_oof * effective_scale)
        result["probes"][f"scale_{tag}"] = {
            "scale": scale,
            "csv": _record(csv_path),
            "parquet": _record(parquet_path),
            "csv_audit": csv_audit,
            "delta_vs_base": {
                group: {
                    "mean_kwh": float(delta[group].mean()),
                    "mean_abs_kwh": float(delta[group].abs().mean()),
                    "max_abs_kwh": float(delta[group].abs().max()),
                }
                for group in TARGET_COLS
            },
            "public_bias_analogy_only": {
                "effective_2024_oof_scale": effective_scale,
                "score": float(analogy.total_score),
                "one_minus_nmae": float(analogy.one_minus_nmae),
                "ficr": float(analogy.ficr),
                "not_a_leaderboard_prediction": True,
            },
        }

    result_path = out_dir / "results.json"
    _atomic_json(result_path, result)
    outputs.append(result_path)
    input_paths = list(dict.fromkeys(required + list(oof_paths.values())))
    input_paths.extend(
        [
            Path(__file__).resolve(),
            PROJECT_ROOT / "src/public_scale_probe.py",
            PROJECT_ROOT / "src/metric.py",
        ]
    )
    input_paths = list(dict.fromkeys(path.resolve() for path in input_paths))
    manifest = {
        "schema_version": 1,
        "artifact_type": "baram_public_adaptive_scale_probe_manifest",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "public_adaptive": True,
        "selection_unsafe": True,
        "private_champion": False,
        "inputs": [_record(path) for path in input_paths],
        "outputs": [_record(path) for path in outputs],
    }
    manifest_path = out_dir / "manifest.json"
    _atomic_json(manifest_path, manifest)

    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"manifest={manifest_path}")
    print(f"manifest_sha256={_sha256(manifest_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

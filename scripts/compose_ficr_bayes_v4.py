"""Compose the strict group-1 FICR delta with existing v4 submissions.

This is intentionally a *post-gate diagnostic* composition.  The upstream
group-1 correction was selected by the strict rolling-forward protocol in
``run_ficr_bayes_decision_strict.py``.  No coefficient is fitted here: the
exact delta between its adjusted and reference predictions is added to the
existing v4 target.  Because the decision to keep this composition was made
after inspecting 2024 diagnostics, its score must not be presented as an
independent validation result.
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

from src.metric import score_details
from src.submission_composition import (
    CAPACITY_KWH,
    TARGET_COLUMNS,
    transfer_prediction_delta,
)


GROUP = "kpx_group_1"
H2_2024_START = pd.Timestamp("2024-07-01 01:00:00")
VARIANTS: dict[str, str] = {
    "recent_v4": "corrected_recent_v4.csv",
    "recent_v4_smoothed": "corrected_recent_v4_smoothed.csv",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/postgate/ficr_bayes_v4_composition"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _read_prediction_parquet(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if tuple(frame.columns) != TARGET_COLUMNS:
        raise ValueError(f"{path}: unexpected prediction columns")
    return frame.astype(float)


def _read_submission_prediction(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    submission = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["forecast_kst_dtm"])
    expected = ("forecast_id", "forecast_kst_dtm", *TARGET_COLUMNS)
    if tuple(submission.columns) != expected:
        raise ValueError(f"{path}: unexpected submission columns")
    prediction = submission.set_index("forecast_kst_dtm").loc[:, TARGET_COLUMNS]
    prediction.index = pd.DatetimeIndex(prediction.index, name="forecast_kst_dtm")
    return submission, prediction.astype(float)


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, compression="zstd")
    temporary.replace(path)


def _atomic_submission(
    path: Path, sample: pd.DataFrame, prediction: pd.DataFrame
) -> dict[str, float]:
    output = sample.loc[:, ["forecast_id", "forecast_kst_dtm"]].copy()
    for group in TARGET_COLUMNS:
        output[group] = prediction[group].to_numpy(dtype=float)
    temporary = path.with_suffix(path.suffix + ".tmp")
    output.to_csv(
        temporary,
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
        date_format="%Y-%m-%d %H:%M:%S",
    )
    temporary.replace(path)
    raw = path.read_bytes()
    if raw[:3] != b"\xef\xbb\xbf":
        raise AssertionError(f"{path}: UTF-8-SIG BOM missing")
    readback = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["forecast_kst_dtm"])
    if not readback.loc[:, ["forecast_id", "forecast_kst_dtm"]].equals(
        sample.loc[:, ["forecast_id", "forecast_kst_dtm"]]
    ):
        raise AssertionError(f"{path}: sample identifiers changed")
    values = readback.loc[:, TARGET_COLUMNS].to_numpy(dtype=float)
    expected = prediction.to_numpy(dtype=float)
    maximum_difference = float(np.max(np.abs(values - expected)))
    if maximum_difference > 5.1e-7:
        raise AssertionError(f"{path}: CSV readback changed predictions")
    if not np.isfinite(values).all():
        raise AssertionError(f"{path}: non-finite CSV prediction")
    for position, group in enumerate(TARGET_COLUMNS):
        if np.any(values[:, position] < 0.0) or np.any(
            values[:, position] > 1.02 * CAPACITY_KWH[group] + 1e-9
        ):
            raise AssertionError(f"{path}: {group} exceeds clip bounds")
    return {"max_abs_csv_readback_difference": maximum_difference}


def _metric_slices(
    labels: pd.DataFrame, baseline: pd.DataFrame, candidate: pd.DataFrame
) -> dict[str, Any]:
    segments = {
        "full": baseline.index,
        "H1": baseline.index[baseline.index < H2_2024_START],
        "H2": baseline.index[baseline.index >= H2_2024_START],
    }
    output: dict[str, Any] = {}
    for name, index in segments.items():
        base_metrics = score_details(labels.reindex(index), baseline.reindex(index))
        candidate_metrics = score_details(labels.reindex(index), candidate.reindex(index))
        output[name] = {
            "baseline": base_metrics.as_dict(),
            "candidate": candidate_metrics.as_dict(),
            "score_delta": candidate_metrics.total_score - base_metrics.total_score,
        }
    return output


def _preflight(out_dir: Path, overwrite: bool) -> None:
    names = {
        "results.json",
        "manifest.json",
        *(f"ficr_bayes_{name}_2025.parquet" for name in VARIANTS),
        *(f"ficr_bayes_{name}_2025.csv" for name in VARIANTS),
    }
    existing = [out_dir / name for name in names if (out_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing outputs: {[str(path) for path in existing]}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    raw_dir = args.raw_dir.expanduser().resolve()
    artifact_dir = args.artifact_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    _preflight(out_dir, bool(args.overwrite))
    out_dir.mkdir(parents=True, exist_ok=True)

    strict_dir = artifact_dir / "postgate/ficr_bayes_decision"
    final_dir = artifact_dir / "final_cf_fix"
    oof_dir = artifact_dir / "oof"
    input_paths = [
        strict_dir / "oof/stage2_baseline_2024.parquet",
        strict_dir / "oof/stage2_fixed_candidate_2024.parquet",
        strict_dir / "predictions/ficr_bayes_decision_2025.parquet",
        strict_dir / "stage1_recipe_lock.json",
        strict_dir / "stage2_promotion_lock.json",
        final_dir / "predictions/corrected_v3_test.parquet",
        final_dir / "corrected_recent_v4.csv",
        final_dir / "corrected_recent_v4_smoothed.csv",
        oof_dir / "gate2024_recent_v4_cf_fix_calibration_fit.parquet",
        oof_dir / "gate2024_recent_v4_cf_fix_smoothed_calibration_fit.parquet",
        raw_dir / "train/train_labels.csv",
        raw_dir / "sample_submission.csv",
        PROJECT_ROOT / "src/submission_composition.py",
        Path(__file__).resolve(),
    ]
    missing = [str(path) for path in input_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing composition inputs: {missing}")

    stage2_reference = _read_prediction_parquet(input_paths[0])
    stage2_adjusted = _read_prediction_parquet(input_paths[1])
    final_adjusted = _read_prediction_parquet(input_paths[2])
    final_reference = _read_prediction_parquet(input_paths[5])
    if not stage2_reference.index.equals(stage2_adjusted.index):
        raise ValueError("stage2 prediction indices differ")
    if not final_reference.index.equals(final_adjusted.index):
        raise ValueError("2025 prediction indices differ")

    labels = pd.read_csv(
        input_paths[10], encoding="utf-8-sig", parse_dates=["kst_dtm"]
    ).set_index("kst_dtm")
    labels.index = pd.DatetimeIndex(labels.index, name="forecast_kst_dtm")
    sample = pd.read_csv(
        input_paths[11], encoding="utf-8-sig", parse_dates=["forecast_kst_dtm"]
    )
    if len(sample) != 8_760:
        raise ValueError("sample submission must contain 8,760 rows")
    sample_index = pd.DatetimeIndex(sample["forecast_kst_dtm"], name="forecast_kst_dtm")
    if not sample_index.equals(final_reference.index):
        raise ValueError("sample and final prediction indices differ")

    oof_paths = {
        "recent_v4": input_paths[8],
        "recent_v4_smoothed": input_paths[9],
    }
    final_csv_paths = {
        "recent_v4": input_paths[6],
        "recent_v4_smoothed": input_paths[7],
    }
    result: dict[str, Any] = {
        "artifact_type": "baram_postgate_ficr_bayes_v4_delta_composition",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selected_group": GROUP,
        "formula": "target + (strict_adjusted_reference - strict_reference), then clip",
        "fit_or_tuning_performed": False,
        "postgate_selection_unsafe": True,
        "untouched_gate_claim": False,
        "warning": (
            "The strict upstream delta is rolling-forward, but retaining these v4 "
            "compositions was decided after inspecting already-consumed 2024 labels."
        ),
        "variants": {},
    }
    outputs: list[Path] = []
    for name in VARIANTS:
        stage2_target = _read_prediction_parquet(oof_paths[name])
        stage2_composed = transfer_prediction_delta(
            stage2_target,
            stage2_reference,
            stage2_adjusted,
            groups=[GROUP],
        )
        _, final_target = _read_submission_prediction(final_csv_paths[name])
        final_composed = transfer_prediction_delta(
            final_target,
            final_reference,
            final_adjusted,
            groups=[GROUP],
        )
        parquet_path = out_dir / f"ficr_bayes_{name}_2025.parquet"
        csv_path = out_dir / f"ficr_bayes_{name}_2025.csv"
        _atomic_parquet(parquet_path, final_composed)
        csv_audit = _atomic_submission(csv_path, sample, final_composed)
        outputs.extend([parquet_path, csv_path])
        result["variants"][name] = {
            "diagnostic_2024": _metric_slices(
                labels, stage2_target, stage2_composed
            ),
            "csv": _record(csv_path),
            "parquet": _record(parquet_path),
            **csv_audit,
        }

    result_path = out_dir / "results.json"
    _atomic_json(result_path, result)
    outputs.append(result_path)
    manifest = {
        "artifact_type": "baram_postgate_ficr_bayes_v4_delta_composition_manifest",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "postgate_selection_unsafe": True,
        "inputs": [_record(path) for path in input_paths],
        "outputs": [_record(path) for path in outputs],
    }
    manifest_path = out_dir / "manifest.json"
    _atomic_json(manifest_path, manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"manifest_sha256={_sha256(manifest_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

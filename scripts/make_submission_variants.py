"""Create three auditable submission variants from locked final-v3 predictions.

The names identify recipes only. They intentionally make no leaderboard-score
claim. No fitting, label access, or parameter search is performed here.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.manifest import make_manifest, sha256_file, write_json_atomic  # noqa: E402
from src.metric import CAPACITY_KWH, TARGET_COLS  # noqa: E402


PREFIX = "v3_locked_full_2025"
SAMPLE_COLUMNS = ("forecast_id", "forecast_kst_dtm", *TARGET_COLS)
EXPECTED_ROWS = 8_760
CSV_ATOL = 5.1e-7

BASE_FILES: dict[str, str] = {
    "lgb_l1": f"{PREFIX}__lgb_l1_test.parquet",
    "lgb_q07": f"{PREFIX}__lgb_q07_test.parquet",
    "shared_l1": f"{PREFIX}__shared_l1_test.parquet",
    "shared_q07": f"{PREFIX}__shared_q07_test.parquet",
    "top200_q07": f"{PREFIX}__top200_q07_test.parquet",
    "energy_q06": f"{PREFIX}__energy_q06_test.parquet",
}
FINAL_FILE = f"{PREFIX}__final_test.parquet"
EXISTING_SUBMISSION_FILE = f"{PREFIX}__submission.csv"

OUTPUT_FILES = {
    "v3_existing": "v3_existing_checked.csv",
    "recent_v4": "recent_v4.csv",
    "hybrid50": "hybrid50.csv",
}
MANIFEST_FILE = "submission_variants_manifest.json"

FORMULAS = {
    "v3_existing": "byte-exact copy of the checked final-v3 submission",
    "recent_v4": {
        "kpx_group_1": (
            "clip(1.1 * (0.8*shared_q07 + 0.05*top200_q07 + "
            "0.15*energy_q06) - 700, 0, 1.02*capacity)"
        ),
        "kpx_group_2": (
            "clip(1.025 * (0.95*lgb_l1 + 0.05*energy_q06) - 100, "
            "0, 1.02*capacity)"
        ),
        "kpx_group_3": (
            "clip(0.5*v3 + 0.5*clip(1.075*(0.7*shared_q07 + "
            "0.1*shared_l1 + 0.2*lgb_q07) - 300, 0, 1.02*capacity), "
            "0, 1.02*capacity)"
        ),
    },
    "hybrid50": {
        "kpx_group_1": "clip(0.5*v3 + 0.5*recent_v4, 0, 1.02*capacity)",
        "kpx_group_2": "clip(0.5*v3 + 0.5*recent_v4, 0, 1.02*capacity)",
        "kpx_group_3": "recent_v4 group 3",
    },
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="official data root containing sample_submission.csv",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        required=True,
        help="final_v3 directory containing predictions/ and the existing submission",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="destination for three CSV files and their hash manifest",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="atomically replace existing variant outputs",
    )
    return parser.parse_args(argv)


def _read_sample(raw_dir: Path) -> tuple[pd.DataFrame, pd.DatetimeIndex, Path]:
    path = raw_dir / "sample_submission.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    sample = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"forecast_id": "string", "forecast_kst_dtm": "string"},
    )
    if tuple(sample.columns) != SAMPLE_COLUMNS:
        raise ValueError(
            f"sample submission columns must be {SAMPLE_COLUMNS}, got {tuple(sample.columns)}"
        )
    if len(sample) != EXPECTED_ROWS:
        raise ValueError(f"sample submission must contain {EXPECTED_ROWS} rows")
    if sample["forecast_id"].isna().any() or sample["forecast_id"].duplicated().any():
        raise ValueError("sample forecast_id must be complete and unique")
    if sample["forecast_kst_dtm"].isna().any():
        raise ValueError("sample forecast timestamps must be complete")
    times = pd.DatetimeIndex(
        pd.to_datetime(sample["forecast_kst_dtm"], errors="raise"),
        name="forecast_kst_dtm",
    )
    if not times.is_unique or not times.is_monotonic_increasing:
        raise ValueError("sample forecast timestamps must be unique and sorted")
    if times.min() != pd.Timestamp("2025-01-01 01:00:00") or times.max() != pd.Timestamp(
        "2026-01-01 00:00:00"
    ):
        raise ValueError("sample forecast time range differs from the official 2025 horizon")
    return sample, times, path


def _read_prediction(path: Path, expected_times: pd.DatetimeIndex) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path, engine="pyarrow")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f"{path.name} must use a DatetimeIndex")
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if tuple(frame.columns) != TARGET_COLS:
        raise ValueError(
            f"{path.name} groups/order must be {TARGET_COLS}, got {tuple(frame.columns)}"
        )
    if len(frame) != EXPECTED_ROWS:
        raise ValueError(f"{path.name} must contain {EXPECTED_ROWS} rows")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{path.name} index must be unique and sorted")
    if not frame.index.equals(expected_times):
        raise ValueError(f"{path.name} index/order differs from sample_submission")
    values = frame.to_numpy(dtype="float64", copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"{path.name} contains NaN or infinite predictions")
    return frame.astype("float64")


def _read_submission(path: Path, sample: pd.DataFrame) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as stream:
        if stream.read(3) != b"\xef\xbb\xbf":
            raise ValueError(f"{path.name} is not UTF-8-SIG")
    frame = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"forecast_id": "string", "forecast_kst_dtm": "string"},
    )
    _validate_submission_frame(frame, sample, context=path.name)
    return frame


def _validate_submission_frame(
    frame: pd.DataFrame, sample: pd.DataFrame, *, context: str
) -> None:
    if tuple(frame.columns) != SAMPLE_COLUMNS:
        raise ValueError(f"{context} columns/order differ from sample_submission")
    if len(frame) != len(sample):
        raise ValueError(f"{context} row count differs from sample_submission")
    for column in ("forecast_id", "forecast_kst_dtm"):
        observed = frame[column].astype(str).to_numpy()
        expected = sample[column].astype(str).to_numpy()
        if not np.array_equal(observed, expected):
            raise ValueError(f"{context} {column} values/order changed")
    numeric = frame.loc[:, list(TARGET_COLS)].to_numpy(dtype="float64")
    if not np.isfinite(numeric).all():
        raise ValueError(f"{context} contains NaN or infinite predictions")
    for group in TARGET_COLS:
        upper = CAPACITY_KWH[group] * 1.02
        values = frame[group].to_numpy(dtype="float64")
        if (values < -1e-6).any() or (values > upper + 1e-6).any():
            raise ValueError(f"{context} violates [0, 1.02*capacity] for {group}")


def _check_existing_v3(
    existing: pd.DataFrame, final_prediction: pd.DataFrame
) -> None:
    observed = existing.loc[:, list(TARGET_COLS)].to_numpy(dtype="float64")
    expected = final_prediction.to_numpy(dtype="float64", copy=False)
    difference = np.abs(observed - expected)
    if not np.all(difference <= CSV_ATOL):
        raise ValueError(
            "existing v3 submission differs from final_test parquet beyond "
            f"six-decimal CSV rounding; max_abs_diff={difference.max():.12g}"
        )


def _clip(group: str, values: np.ndarray | pd.Series) -> np.ndarray:
    return np.clip(
        np.asarray(values, dtype="float64"),
        0.0,
        CAPACITY_KWH[group] * 1.02,
    )


def _build_recent_v4(
    base: Mapping[str, pd.DataFrame], v3: pd.DataFrame
) -> pd.DataFrame:
    output = pd.DataFrame(index=v3.index, columns=TARGET_COLS, dtype="float64")
    g1 = 1.10 * (
        0.80 * base["shared_q07"]["kpx_group_1"]
        + 0.05 * base["top200_q07"]["kpx_group_1"]
        + 0.15 * base["energy_q06"]["kpx_group_1"]
    ) - 700.0
    output["kpx_group_1"] = _clip("kpx_group_1", g1)

    g2 = 1.025 * (
        0.95 * base["lgb_l1"]["kpx_group_2"]
        + 0.05 * base["energy_q06"]["kpx_group_2"]
    ) - 100.0
    output["kpx_group_2"] = _clip("kpx_group_2", g2)

    g3_recent_component = 1.075 * (
        0.70 * base["shared_q07"]["kpx_group_3"]
        + 0.10 * base["shared_l1"]["kpx_group_3"]
        + 0.20 * base["lgb_q07"]["kpx_group_3"]
    ) - 300.0
    g3_recent_component = _clip("kpx_group_3", g3_recent_component)
    output["kpx_group_3"] = _clip(
        "kpx_group_3",
        0.50 * v3["kpx_group_3"] + 0.50 * g3_recent_component,
    )
    return output


def _build_hybrid50(v3: pd.DataFrame, recent: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=v3.index, columns=TARGET_COLS, dtype="float64")
    for group in ("kpx_group_1", "kpx_group_2"):
        output[group] = _clip(group, 0.50 * v3[group] + 0.50 * recent[group])
    output["kpx_group_3"] = _clip("kpx_group_3", recent["kpx_group_3"])
    return output


def _assemble_submission(sample: pd.DataFrame, prediction: pd.DataFrame) -> pd.DataFrame:
    submission = sample.copy()
    for group in TARGET_COLS:
        submission[group] = prediction[group].to_numpy(dtype="float64")
    _validate_submission_frame(submission, sample, context="assembled submission")
    return submission


def _write_csv_atomic(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        frame.to_csv(
            temporary,
            index=False,
            encoding="utf-8-sig",
            float_format="%.6f",
            lineterminator="\n",
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _verify_readback(
    path: Path,
    expected: pd.DataFrame,
    sample: pd.DataFrame,
    *,
    exact_source: Path | None = None,
) -> dict[str, object]:
    observed = _read_submission(path, sample)
    expected_values = expected.loc[:, list(TARGET_COLS)].to_numpy(dtype="float64")
    observed_values = observed.loc[:, list(TARGET_COLS)].to_numpy(dtype="float64")
    max_abs_diff = float(np.max(np.abs(observed_values - expected_values)))
    if max_abs_diff > CSV_ATOL:
        raise AssertionError(
            f"{path.name} readback differs after six-decimal write: {max_abs_diff}"
        )
    output_hash = sha256_file(path)
    if exact_source is not None:
        source_hash = sha256_file(exact_source)
        if output_hash != source_hash:
            raise AssertionError("v3_existing output is not a byte-exact copy")
    else:
        source_hash = None
    return {
        "rows": int(len(observed)),
        "sha256": output_hash,
        "source_sha256": source_hash,
        "max_readback_abs_diff": max_abs_diff,
        "utf8_sig": True,
        "minimum_by_group": {
            group: float(observed[group].min()) for group in TARGET_COLS
        },
        "maximum_by_group": {
            group: float(observed[group].max()) for group in TARGET_COLS
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    raw_dir = args.raw_dir.expanduser().resolve()
    base_dir = args.base_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    prediction_dir = base_dir / "predictions"

    output_paths = {
        name: out_dir / filename for name, filename in OUTPUT_FILES.items()
    }
    manifest_path = out_dir / MANIFEST_FILE
    existing_outputs = [
        path for path in [*output_paths.values(), manifest_path] if path.exists()
    ]
    if existing_outputs and not args.overwrite:
        rendered = "\n".join(f"  - {path}" for path in existing_outputs)
        raise FileExistsError(
            "variant outputs already exist; pass --overwrite to replace them:\n"
            f"{rendered}"
        )

    sample, sample_times, sample_path = _read_sample(raw_dir)
    input_paths = [sample_path]
    base_predictions: dict[str, pd.DataFrame] = {}
    for name, filename in BASE_FILES.items():
        path = prediction_dir / filename
        base_predictions[name] = _read_prediction(path, sample_times)
        input_paths.append(path)
    final_path = prediction_dir / FINAL_FILE
    final_prediction = _read_prediction(final_path, sample_times)
    input_paths.append(final_path)
    existing_path = base_dir / EXISTING_SUBMISSION_FILE
    existing_submission = _read_submission(existing_path, sample)
    _check_existing_v3(existing_submission, final_prediction)
    input_paths.append(existing_path)

    recent_prediction = _build_recent_v4(base_predictions, final_prediction)
    hybrid_prediction = _build_hybrid50(final_prediction, recent_prediction)
    recent_submission = _assemble_submission(sample, recent_prediction)
    hybrid_submission = _assemble_submission(sample, hybrid_prediction)

    print("writing v3_existing_checked.csv (byte-exact checked copy) ...", flush=True)
    _copy_atomic(existing_path, output_paths["v3_existing"])
    print("writing recent_v4.csv ...", flush=True)
    _write_csv_atomic(recent_submission, output_paths["recent_v4"])
    print("writing hybrid50.csv ...", flush=True)
    _write_csv_atomic(hybrid_submission, output_paths["hybrid50"])

    checks = {
        "v3_existing": _verify_readback(
            output_paths["v3_existing"],
            existing_submission,
            sample,
            exact_source=existing_path,
        ),
        "recent_v4": _verify_readback(
            output_paths["recent_v4"], recent_submission, sample
        ),
        "hybrid50": _verify_readback(
            output_paths["hybrid50"], hybrid_submission, sample
        ),
    }
    manifest = make_manifest(
        artifact_type="baram_submission_variants",
        parameters={
            "source_recipe_prefix": PREFIX,
            "capacities_kwh": {group: CAPACITY_KWH[group] for group in TARGET_COLS},
            "upper_clip_capacity_fraction": 1.02,
            "csv_encoding": "utf-8-sig",
            "csv_float_format": "%.6f",
            "formulas": FORMULAS,
            "names_are_recipe_labels_not_leaderboard_scores": True,
        },
        input_files=input_paths,
        output_files=list(output_paths.values()),
        results={"variants": checks},
        project_dir=PROJECT_DIR,
    )
    write_json_atomic(manifest_path, manifest, overwrite=args.overwrite)
    print(f"complete: {manifest_path}")
    for name, path in output_paths.items():
        print(f"{name}: {path} sha256={checks[name]['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Add the full-width consensus artifact and close the promoted final manifest."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "artifacts/final_multi_nwp_consensus_g23_rescue_v1"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
SOURCE_MEAN_PATH = PREDICTION_DIR / "source3_average_increment_cf_2025.parquet"
CONSENSUS_PATH = PREDICTION_DIR / "final_consensus_increment_cf_2025.parquet"
BASELINE_PATH = PREDICTION_DIR / "recent097_baseline_2025.parquet"
FINAL_PATH = PREDICTION_DIR / "multi_nwp_consensus_g23_rescue_v1_2025.parquet"
FINAL_LOCK_PATH = OUTPUT_DIR / "final_lock.json"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
TARGET_COLS = ("kpx_group_1", "kpx_group_2", "kpx_group_3")
CAPACITIES = {"kpx_group_1": 21_600.0, "kpx_group_2": 21_600.0, "kpx_group_3": 21_000.0}
TEST_INDEX = pd.date_range(
    "2025-01-01 01:00", "2026-01-01 00:00", freq="h", name="forecast_kst_dtm"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def atomic_json(payload: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    frame.to_parquet(temporary, engine="pyarrow", compression="zstd", index=True)
    os.replace(temporary, path)


def checked(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if not frame.index.equals(TEST_INDEX) or tuple(frame.columns) != columns:
        raise AssertionError(f"index/columns differ: {path}")
    if not np.isfinite(frame.to_numpy(dtype=np.float64)).all():
        raise AssertionError(f"non-finite values: {path}")
    return frame.astype(np.float64)


def main() -> None:
    source_mean = checked(SOURCE_MEAN_PATH, ("kpx_group_2", "kpx_group_3"))
    baseline = checked(BASELINE_PATH, TARGET_COLS)
    final = checked(FINAL_PATH, TARGET_COLS)
    consensus = pd.DataFrame(0.0, index=TEST_INDEX, columns=TARGET_COLS, dtype=np.float64)
    consensus.loc[:, ["kpx_group_2", "kpx_group_3"]] = source_mean.to_numpy(dtype=np.float64)
    if not np.array_equal(
        consensus["kpx_group_1"].to_numpy(dtype=np.float64),
        np.zeros(len(TEST_INDEX), dtype=np.float64),
    ):
        raise AssertionError("G1 consensus increment is not exact float64 zero")

    replay = baseline.copy()
    for group in ("kpx_group_2", "kpx_group_3"):
        replay[group] = np.clip(
            baseline[group].to_numpy(dtype=np.float64)
            + 0.15 * CAPACITIES[group] * consensus[group].to_numpy(dtype=np.float64),
            0.0,
            1.02 * CAPACITIES[group],
        )
    if not np.array_equal(replay.to_numpy(dtype=np.float64), final.to_numpy(dtype=np.float64)):
        raise AssertionError("base + frozen consensus formula does not exactly replay final prediction")
    atomic_parquet(consensus, CONSENSUS_PATH)
    reloaded = checked(CONSENSUS_PATH, TARGET_COLS)
    if not np.array_equal(reloaded.to_numpy(dtype=np.float64), consensus.to_numpy(dtype=np.float64)):
        raise AssertionError("full-width consensus Parquet reload differs")

    final_lock = json.loads(FINAL_LOCK_PATH.read_text(encoding="utf-8"))
    final_lock["final_consensus_increment_cf_G1_zero_G23_active"] = file_record(CONSENSUS_PATH)
    final_lock["scale097_base_exact_index_columns"] = file_record(BASELINE_PATH)
    final_lock["consensus_formula_final_prediction_float64_exact_replay"] = True
    final_lock["artifact_finalizer"] = file_record(Path(__file__))
    atomic_json(final_lock, FINAL_LOCK_PATH)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    replacements = {
        str(FINAL_LOCK_PATH): file_record(FINAL_LOCK_PATH),
        str(CONSENSUS_PATH): file_record(CONSENSUS_PATH),
        str(BASELINE_PATH): file_record(BASELINE_PATH),
        str(Path(__file__)): file_record(Path(__file__)),
    }
    records = {str(record["path"]): record for record in manifest["files"]}
    records.update(replacements)
    manifest["files"] = list(records.values())
    manifest["full_width_consensus_artifact_complete"] = True
    manifest["scale097_base_artifact_complete"] = True
    manifest["consensus_formula_exact_replay"] = True
    atomic_json(manifest, MANIFEST_PATH)
    print(f"CONSENSUS={file_record(CONSENSUS_PATH)}")
    print(f"BASELINE={file_record(BASELINE_PATH)}")
    print(f"FINAL_LOCK={file_record(FINAL_LOCK_PATH)}")
    print(f"MANIFEST={file_record(MANIFEST_PATH)}")


if __name__ == "__main__":
    main()

"""Build the frozen score-aware map from causal INNER OOF predictions.

This script is intentionally downstream of ``run_virtual_feature_exit.py
--stage inner``.  It does not fit a model, inspect 2024 labels, or read 2025
data.  It joins the locked INNER control prediction to the physically bounded
label prefix and the <=2023 weather rows, then delegates all statistics to
``src.score_aware_correlation_map``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.manifest import describe_file, sha256_file, utc_now, write_json_atomic
from src.metric import CAPACITY_KWH, TARGET_COLS
from src.score_aware_correlation_map import (
    CorrelationMapConfig,
    build_correlation_map,
    write_correlation_map,
)


PREREGISTER = ROOT / "configs/feature_exit_virtual_score_preregister_v1.json"
PREREGISTER_SHA256 = (
    "ccb857578b812e61e05fa63f43a45382f147f7abf3464bdb5c6a597bde42990c"
)
PREREGISTER_BYTES = 11_259
LABEL_PREFIX_BYTES = 742_551
LABEL_PREFIX_SHA256 = (
    "ee9707cc81229db760fc035d0bf1d10eb3586d2fef95849bdf463d73fae3aacd"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir", type=Path, default=Path("data/local/open")
    )
    parser.add_argument(
        "--artifact-root", type=Path, default=ROOT / "artifacts"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "artifacts/feature_exit_virtual_v1",
    )
    return parser.parse_args()


def _bounded_labels(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    with path.open("rb", buffering=0) as stream:
        payload = stream.read(LABEL_PREFIX_BYTES)
        position = stream.tell()
    digest = hashlib.sha256(payload).hexdigest()
    if position != LABEL_PREFIX_BYTES or digest != LABEL_PREFIX_SHA256:
        raise AssertionError("physical label prefix changed")
    frame = pd.read_csv(io.BytesIO(payload), encoding="utf-8-sig")
    if len(frame) != 17_520 or tuple(frame.columns) != ("kst_dtm", *TARGET_COLS):
        raise AssertionError("label prefix schema changed")
    frame.index = pd.DatetimeIndex(
        pd.to_datetime(frame.pop("kst_dtm"), errors="raise"),
        name="forecast_kst_dtm",
    )
    return frame.astype(np.float64), {
        "path": str(path.resolve()),
        "bytes_returned_to_parser": LABEL_PREFIX_BYTES,
        "sha256": digest,
        "last_timestamp": frame.index.max().isoformat(),
        "full_file_hash_computed": False,
        "2024_label_value_cells_materialized": 0,
    }


def _weather_prefix(
    artifact_root: Path, contract: dict[str, Any]
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    frames: dict[str, pd.DataFrame] = {}
    evidence: dict[str, Any] = {}
    names: tuple[str, ...] | None = None
    for group in TARGET_COLS:
        spec = contract["data_contract"]["weather_cache"][group]
        path = ROOT / spec["path"]
        if not path.is_file():
            path = artifact_root / "cache" / f"{group}_weather_train.parquet"
        if path.stat().st_size != int(spec["bytes"]) or sha256_file(path) != spec["sha256"]:
            raise AssertionError(f"{group} weather identity changed")
        frame = pd.read_parquet(
            path,
            engine="pyarrow",
            filters=[("forecast_kst_dtm", "<=", pd.Timestamp("2024-01-01 00:00:00"))],
        )
        frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
        current = tuple(map(str, frame.columns))
        if names is not None and current != names:
            raise AssertionError("weather feature schemas differ")
        names = current
        if len(frame) != 17_520 or len(current) != 612:
            raise AssertionError("weather prefix shape changed")
        if not np.isfinite(frame.to_numpy(dtype=np.float32, copy=False)).all():
            raise AssertionError("weather prefix contains non-finite values")
        frames[group] = frame.astype(np.float32, copy=False)
        evidence[group] = {
            "input": describe_file(path),
            "materialized_rows": len(frame),
            "materialized_end": frame.index.max().isoformat(),
            "physical_row_group_isolation_claimed": False,
        }
    return frames, evidence


def main() -> int:
    args = parse_args()
    artifact_root = args.artifact_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    preregister = PREREGISTER.resolve()
    if preregister.stat().st_size != PREREGISTER_BYTES:
        raise AssertionError("preregister byte count changed")
    if sha256_file(preregister) != PREREGISTER_SHA256:
        raise AssertionError("preregister hash changed")
    contract = json.loads(preregister.read_text(encoding="utf-8"))

    inner_manifest = out_dir / "inner/manifest.json"
    inner_oof_path = out_dir / "inner/inner_oof_predictions.parquet"
    inner_scores = out_dir / "inner/inner_direct_scores.json"
    for required in (inner_manifest, inner_oof_path, inner_scores):
        if not required.is_file():
            raise FileNotFoundError(required)
    manifest = json.loads(inner_manifest.read_text(encoding="utf-8"))
    if manifest.get("preregister_sha256") != PREREGISTER_SHA256:
        raise AssertionError("INNER manifest preregister mismatch")
    locked_outputs = {
        str(Path(record["path"]).resolve()): record
        for record in manifest.get("outputs", [])
        if isinstance(record, dict) and "path" in record
    }
    for locked_path in (inner_oof_path, inner_scores):
        record = locked_outputs.get(str(locked_path.resolve()))
        if record is None:
            raise AssertionError(f"INNER manifest does not bind {locked_path}")
        observed = describe_file(locked_path)
        if (
            int(record["size_bytes"]) != int(observed["size_bytes"])
            or record["sha256"] != observed["sha256"]
        ):
            raise AssertionError(f"INNER output identity changed: {locked_path}")
    map_dir = out_dir / "correlation_map"
    map_manifest = out_dir / "correlation_map_manifest.json"
    if map_manifest.exists() or (map_dir.exists() and any(map_dir.iterdir())):
        raise FileExistsError("correlation-map output already exists; refusing overwrite")

    labels, label_evidence = _bounded_labels(
        args.raw_dir.expanduser().resolve() / "train/train_labels.csv"
    )
    weather, weather_evidence = _weather_prefix(artifact_root, contract)
    oof = pd.read_parquet(inner_oof_path, engine="pyarrow")
    required_columns = {
        "forecast_kst_dtm",
        "group",
        "fold",
        "control_all",
    }
    if not required_columns.issubset(oof.columns):
        raise AssertionError("INNER OOF schema changed")
    oof["forecast_kst_dtm"] = pd.to_datetime(
        oof["forecast_kst_dtm"], errors="raise"
    )

    feature_parts: list[pd.DataFrame] = []
    row_parts: list[pd.DataFrame] = []
    for group in TARGET_COLS:
        part = oof.loc[oof["group"] == group].copy()
        times = pd.DatetimeIndex(part["forecast_kst_dtm"])
        if times.has_duplicates or not times.is_monotonic_increasing:
            raise AssertionError(f"{group} OOF time index changed")
        feature_parts.append(weather[group].loc[times].reset_index(drop=True))
        capacity = CAPACITY_KWH[group]
        row_parts.append(
            pd.DataFrame(
                {
                    "group": group,
                    "fold": part["fold"].astype(str).to_numpy(),
                    "forecast_kst_dtm": times.to_numpy(),
                    "actual_kwh": labels.loc[times, group].to_numpy(dtype=np.float64),
                    "baseline_kwh": part["control_all"].to_numpy(dtype=np.float64)
                    * capacity,
                    "capacity_kwh": capacity,
                }
            )
        )
    oof_features = pd.concat(feature_parts, axis=0, ignore_index=True)
    oof_rows = pd.concat(row_parts, axis=0, ignore_index=True)
    if not oof_features.index.equals(oof_rows.index):
        raise AssertionError("map rows are not aligned")

    train_parts = [
        weather["kpx_group_1"].loc[
            "2022-01-01 01:00:00":"2022-10-01 00:00:00"
        ],
        weather["kpx_group_2"].loc[
            "2022-01-01 01:00:00":"2022-10-01 00:00:00"
        ],
        weather["kpx_group_3"].loc[
            "2023-01-01 01:00:00":"2023-06-01 00:00:00"
        ],
    ]
    causal_train = pd.concat(train_parts, axis=0, ignore_index=True)
    result = build_correlation_map(
        oof_features=oof_features,
        oof_rows=oof_rows,
        redundancy_sources={
            "causal_train": causal_train,
            "causal_inner_apply": oof_features,
        },
        config=CorrelationMapConfig(n_jobs=7),
    )
    paths = write_correlation_map(result, map_dir)
    write_json_atomic(
        out_dir / "correlation_map_manifest.json",
        {
            "schema_version": 1,
            "artifact_type": "feature_exit_score_aware_correlation_map",
            "created_utc": utc_now(),
            "preregister": describe_file(preregister),
            "inner_manifest": describe_file(inner_manifest),
            "inner_oof": describe_file(inner_oof_path),
            "inner_scores": describe_file(inner_scores),
            "label_prefix": label_evidence,
            "weather": weather_evidence,
            "map_outputs": {
                name: describe_file(path) for name, path in sorted(paths.items())
            },
            "oof_rows": len(oof_rows),
            "features": oof_features.shape[1],
            "2024_labels_read": False,
            "2025_data_read": False,
            "leaderboard_score_claimed": False,
        },
        overwrite=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

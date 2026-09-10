"""Strict-forward conditional Bayes decision experiment for the BARAM metric.

Stage 1 reads only a physically bounded pre-2024 label prefix and 2023 OOF
predictions.  It writes an immutable single-recipe lock.  Stage 2 validates
that lock before it may read 2024 and performs only the fixed 2023 -> 2024
confirmation.  Final 2025 inference is allowed only after the complete locked
recipe passes every preregistered Stage-2 segment.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.manifest import (  # noqa: E402
    describe_file,
    git_state,
    package_versions,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from src.metric import CAPACITY_KWH, TARGET_COLS, group_metrics  # noqa: E402
from src.probabilistic import (  # noqa: E402
    ProbabilisticDecisionConfig,
    ProbabilisticFICRDecision,
)


PREREGISTER_SHA256 = "65f13d400ad88d08bcca9efdc596a02a1396a67bb590a304eb7bdbeba349ae6e"
METHODS = (
    "global_conformal",
    "binned_kde",
    "quantile_stack",
    "threshold_meta",
)
COMMON_CANDIDATES = (
    "lgb_l1",
    "lgb_q07",
    "shared_l1",
    "shared_q07",
    "top200_q07",
    "energy_q06",
)
STAGE1_REQUIRED_SEGMENTS: dict[str, tuple[str, ...]] = {
    "kpx_group_1": ("full", "Q2", "Q3", "Q4"),
    "kpx_group_2": ("full", "Q2", "Q3", "Q4"),
    "kpx_group_3": (
        "full",
        "August",
        "September",
        "October",
        "November",
        "December",
    ),
}
STAGE2_REQUIRED_SEGMENTS = ("full", "H1", "H2")

YEAR_2022_START = pd.Timestamp("2022-01-01 01:00:00")
YEAR_2023_START = pd.Timestamp("2023-01-01 01:00:00")
Q2_2023_START = pd.Timestamp("2023-04-01 01:00:00")
H2_2023_START = pd.Timestamp("2023-07-01 01:00:00")
Q4_2023_START = pd.Timestamp("2023-10-01 01:00:00")
YEAR_2023_END = pd.Timestamp("2024-01-01 00:00:00")
YEAR_2024_START = pd.Timestamp("2024-01-01 01:00:00")
H2_2024_START = pd.Timestamp("2024-07-01 01:00:00")
YEAR_2024_END = pd.Timestamp("2025-01-01 00:00:00")
YEAR_2025_START = pd.Timestamp("2025-01-01 01:00:00")
YEAR_2025_END = pd.Timestamp("2026-01-01 00:00:00")

EXPECTED_STAGE1_LABEL_ROWS = 17_520
OFFICIAL_STAGE1_LABEL_PREFIX_BYTES = 742_551
OFFICIAL_STAGE1_LABEL_PREFIX_SHA256 = (
    "ee9707cc81229db760fc035d0bf1d10eb3586d2fef95849bdf463d73fae3aacd"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("stage1", "stage2", "final"), required=True
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=Path(r"data/local/open")
    )
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/postgate/ficr_bayes_decision"),
    )
    parser.add_argument(
        "--preregister",
        type=Path,
        default=Path("configs/ficr_bayes_decision_preregister.json"),
    )
    return parser.parse_args(argv)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        _json_ready(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    write_json_atomic(path, _json_ready(payload), overwrite=False)


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        frame.to_parquet(temporary, index=True)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_joblib(payload: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        joblib.dump(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
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


def _copy_exclusive(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_bytes(source.read_bytes())


def _year_index(year: int) -> pd.DatetimeIndex:
    return pd.date_range(
        f"{year}-01-01 01:00:00",
        f"{year + 1}-01-01 00:00:00",
        freq="h",
        name="forecast_kst_dtm",
    )


def _interval(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="h", name="forecast_kst_dtm")


def _stage1_rolling_contract() -> dict[str, Any]:
    q1 = _interval(YEAR_2023_START, Q2_2023_START - pd.Timedelta(hours=1))
    q2 = _interval(Q2_2023_START, H2_2023_START - pd.Timedelta(hours=1))
    h1 = _interval(YEAR_2023_START, H2_2023_START - pd.Timedelta(hours=1))
    q3 = _interval(H2_2023_START, Q4_2023_START - pd.Timedelta(hours=1))
    q1_q3 = _interval(YEAR_2023_START, Q4_2023_START - pd.Timedelta(hours=1))
    q4 = _interval(Q4_2023_START, YEAR_2023_END)
    g12_directions = (
        ("Q1_fit_Q2_apply", q1, q2),
        ("H1_fit_Q3_apply", h1, q3),
        ("Q1_Q3_fit_Q4_apply", q1_q3, q4),
    )
    g3_directions = (
        (
            "August",
            _interval(H2_2023_START, pd.Timestamp("2023-08-01 00:00:00")),
            _interval(
                pd.Timestamp("2023-08-01 01:00:00"),
                pd.Timestamp("2023-09-01 00:00:00"),
            ),
        ),
        (
            "September",
            _interval(H2_2023_START, pd.Timestamp("2023-09-01 00:00:00")),
            _interval(
                pd.Timestamp("2023-09-01 01:00:00"),
                pd.Timestamp("2023-10-01 00:00:00"),
            ),
        ),
        (
            "October",
            _interval(H2_2023_START, pd.Timestamp("2023-10-01 00:00:00")),
            _interval(
                pd.Timestamp("2023-10-01 01:00:00"),
                pd.Timestamp("2023-11-01 00:00:00"),
            ),
        ),
        (
            "November",
            _interval(H2_2023_START, pd.Timestamp("2023-11-01 00:00:00")),
            _interval(
                pd.Timestamp("2023-11-01 01:00:00"),
                pd.Timestamp("2023-12-01 00:00:00"),
            ),
        ),
        (
            "December",
            _interval(H2_2023_START, pd.Timestamp("2023-12-01 00:00:00")),
            _interval(pd.Timestamp("2023-12-01 01:00:00"), YEAR_2023_END),
        ),
    )
    return {
        "g12_directions": g12_directions,
        "g12_application": _interval(Q2_2023_START, YEAR_2023_END),
        "g3_h2": _interval(H2_2023_START, YEAR_2023_END),
        "g3_directions": g3_directions,
        "g3_application": _interval(
            pd.Timestamp("2023-08-01 01:00:00"), YEAR_2023_END
        ),
        "g12_segments": {"Q2": q2, "Q3": q3, "Q4": q4},
    }


def _prefix_identity(path: Path, byte_limit: int, data_rows: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    remaining = int(byte_limit)
    newline_count = 0
    returned = 0
    with path.open("rb", buffering=0) as stream:
        while remaining:
            chunk = stream.read(min(1024 * 1024, remaining))
            if not chunk:
                raise AssertionError(f"{path} ended inside the locked prefix")
            digest.update(chunk)
            newline_count += chunk.count(b"\n")
            returned += len(chunk)
            remaining -= len(chunk)
        position = stream.tell()
    if returned != byte_limit or position != byte_limit:
        raise AssertionError("physical prefix reader crossed its byte cap")
    if newline_count != data_rows + 1:
        raise AssertionError(
            f"locked prefix line count changed: {newline_count} != {data_rows + 1}"
        )
    return digest.hexdigest(), returned


def _snapshot_file(path: Path) -> dict[str, Any]:
    return {"snapshot_kind": "whole_file", **describe_file(path)}


def _snapshot_label_prefix(path: Path) -> dict[str, Any]:
    sha256, returned = _prefix_identity(
        path, OFFICIAL_STAGE1_LABEL_PREFIX_BYTES, EXPECTED_STAGE1_LABEL_ROWS
    )
    record = describe_file(path, include_hash=False)
    record.update(
        {
            "snapshot_kind": "csv_prefix",
            "prefix_data_rows": EXPECTED_STAGE1_LABEL_ROWS,
            "prefix_bytes": returned,
            "prefix_sha256": sha256,
            "whole_file_hash_omitted_prelock": True,
        }
    )
    return record


def _refresh_snapshot(snapshot: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    refreshed: dict[str, Any] = {}
    for name, record in snapshot.items():
        path = Path(str(record["path"]))
        if record["snapshot_kind"] == "whole_file":
            refreshed[name] = _snapshot_file(path)
        elif record["snapshot_kind"] == "csv_prefix":
            refreshed[name] = _snapshot_label_prefix(path)
        else:
            raise AssertionError(f"unknown snapshot kind: {record['snapshot_kind']}")
    return refreshed


def _assert_snapshot_equal(
    before: Mapping[str, Any], after: Mapping[str, Any], *, name: str
) -> None:
    if _json_ready(before) != _json_ready(after):
        raise AssertionError(f"{name} changed")


def _verify_preregister(path: Path) -> tuple[dict[str, Any], ProbabilisticDecisionConfig]:
    observed = sha256_file(path)
    if observed != PREREGISTER_SHA256:
        raise AssertionError(
            f"preregister hash changed: {observed} != {PREREGISTER_SHA256}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    family = payload["candidate_family"]
    if int(family["candidate_count"]) != 4:
        raise AssertionError("candidate_count must remain four")
    if tuple(family["methods_in_complexity_order"]) != METHODS:
        raise AssertionError("candidate method order changed")
    raw_config = dict(family["config"])
    for name in (
        "candidate_offsets_cf",
        "bin_edges_cf",
        "quantile_levels",
    ):
        raw_config[name] = tuple(float(value) for value in raw_config[name])
    config = ProbabilisticDecisionConfig(**raw_config)
    if _json_ready(asdict(config)) != _json_ready(family["config"]):
        raise AssertionError("preregistered probabilistic config is not exact")
    return payload, config


def _read_bounded_labels(
    path: Path,
    *,
    data_rows: int = EXPECTED_STAGE1_LABEL_ROWS,
    byte_limit: int = OFFICIAL_STAGE1_LABEL_PREFIX_BYTES,
    expected_prefix_sha256: str = OFFICIAL_STAGE1_LABEL_PREFIX_SHA256,
    expected_index: pd.DatetimeIndex | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prefix_sha256, prefix_bytes = _prefix_identity(path, byte_limit, data_rows)
    if prefix_sha256 != expected_prefix_sha256:
        raise AssertionError("official pre-2024 label prefix hash changed")
    with path.open("rb", buffering=0) as stream:
        payload = stream.read(prefix_bytes)
        underlying_position = stream.tell()
    if len(payload) != prefix_bytes or underlying_position != prefix_bytes:
        raise AssertionError("bounded label read crossed its physical byte cap")
    labels = pd.read_csv(io.BytesIO(payload), encoding="utf-8-sig")
    expected_columns = ("kst_dtm", *TARGET_COLS)
    if tuple(labels.columns) != expected_columns or len(labels) != data_rows:
        raise AssertionError("bounded label schema/rows changed")
    labels.index = pd.DatetimeIndex(
        pd.to_datetime(labels.pop("kst_dtm"), errors="raise"),
        name="forecast_kst_dtm",
    )
    if expected_index is None:
        expected_index = pd.date_range(
            YEAR_2022_START, YEAR_2023_END, freq="h", name="forecast_kst_dtm"
        )
    else:
        expected_index = pd.DatetimeIndex(
            expected_index, name="forecast_kst_dtm"
        )
    if not labels.index.equals(expected_index):
        raise AssertionError("bounded label timestamp sequence changed")
    return labels.astype(float), {
        "path": str(path.resolve()),
        "physical_byte_limit": prefix_bytes,
        "physical_bytes_returned": len(payload),
        "underlying_file_position_after_read": underlying_position,
        "prefix_sha256": prefix_sha256,
        "materialized_rows": len(labels),
        "materialized_start": labels.index.min(),
        "materialized_end": labels.index.max(),
        "suffix_bytes_exposed_to_parser": 0,
        "2024_label_rows_materialized": 0,
    }


def _read_full_labels(path: Path) -> pd.DataFrame:
    labels = pd.read_csv(path, encoding="utf-8-sig")
    if tuple(labels.columns) != ("kst_dtm", *TARGET_COLS):
        raise AssertionError("full label schema changed")
    labels.index = pd.DatetimeIndex(
        pd.to_datetime(labels.pop("kst_dtm"), errors="raise"),
        name="forecast_kst_dtm",
    )
    expected = pd.date_range(
        YEAR_2022_START, YEAR_2024_END, freq="h", name="forecast_kst_dtm"
    )
    if not labels.index.equals(expected):
        raise AssertionError("full training label timestamp sequence changed")
    return labels.astype(float)


def _read_prediction(
    path: Path,
    expected_index: pd.DatetimeIndex,
    *,
    required_columns: Sequence[str],
    exact_columns: bool = True,
) -> pd.DataFrame:
    frame = pd.read_parquet(path, engine="pyarrow")
    observed_index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if not observed_index.equals(expected_index):
        raise AssertionError(f"{path} timestamp index changed")
    if exact_columns:
        if tuple(frame.columns) != tuple(required_columns):
            raise AssertionError(f"{path} columns changed")
    elif not set(required_columns).issubset(frame.columns):
        raise AssertionError(f"{path} is missing required columns")
    frame = frame.copy()
    frame.index = expected_index
    if not np.isfinite(frame.loc[:, list(required_columns)].to_numpy(dtype=float)).all():
        raise AssertionError(f"{path} contains non-finite predictions")
    return frame.astype(float)


def _component_paths(artifact_root: Path, period: str) -> dict[str, Path]:
    if period == "2023":
        oof = artifact_root / "oof"
        return {
            "lgb_l1": oof / "dev2023_lgb_l1_eligible_n1500.parquet",
            "lgb_q07": oof / "dev2023_lgb_q07_eligible.parquet",
            "shared_l1": oof / "dev2023_shared_l1_eligible.parquet",
            "shared_q07": oof / "dev2023_shared_q07_eligible.parquet",
            "top200_q07": oof / "dev2023_lgb_top200_q07_eligible.parquet",
            "energy_q06": oof / "dev2023_lgb_q06_energywt_eligible.parquet",
        }
    if period == "2024":
        gate = artifact_root / "gate" / "v3" / "predictions"
        oof = artifact_root / "oof"
        return {
            "lgb_l1": gate / "lgb_l1_gate.parquet",
            "lgb_q07": gate / "lgb_q07_gate.parquet",
            "shared_l1": oof / "gate2024_shared_l1_cf.parquet",
            "shared_q07": oof / "gate2024_shared_q07_cf.parquet",
            "top200_q07": gate / "top200_q07_gate.parquet",
            "energy_q06": gate / "energy_q06_gate.parquet",
        }
    if period == "2025":
        final = artifact_root / "final_v3" / "predictions"
        corrected = artifact_root / "final_cf_fix" / "predictions"
        prefix = "v3_locked_full_2025"
        return {
            "lgb_l1": final / f"{prefix}__lgb_l1_test.parquet",
            "lgb_q07": final / f"{prefix}__lgb_q07_test.parquet",
            "shared_l1": corrected / "shared_l1_cf_seed42_test.parquet",
            "shared_q07": corrected / "shared_q07_cf_seed42_test.parquet",
            "top200_q07": final / f"{prefix}__top200_q07_test.parquet",
            "energy_q06": final / f"{prefix}__energy_q06_test.parquet",
        }
    raise ValueError(period)


def _load_2023_sources(
    artifact_root: Path,
) -> tuple[
    dict[str, pd.DataFrame],
    pd.DataFrame,
    dict[str, pd.DataFrame],
    pd.DataFrame,
    dict[str, Path],
]:
    index = _year_index(2023)
    paths = _component_paths(artifact_root, "2023")
    candidates = {
        name: _read_prediction(path, index, required_columns=TARGET_COLS[:2])
        for name, path in paths.items()
    }
    base_path = artifact_root / "oof" / "dev2023_locked_v3.parquet"
    base = _read_prediction(base_path, index, required_columns=TARGET_COLS[:2])

    g3_index = _interval(H2_2023_START, YEAR_2023_END)
    g3_path = artifact_root / "oof" / "g3dev2023h2_candidates.parquet"
    raw = _read_prediction(
        g3_path,
        g3_index,
        required_columns=("l1", "q07", "shared_l1", "shared_q07", "top200q07", "ewq06"),
        exact_columns=False,
    )
    rename = {
        "l1": "lgb_l1",
        "q07": "lgb_q07",
        "shared_l1": "shared_l1",
        "shared_q07": "shared_q07",
        "top200q07": "top200_q07",
        "ewq06": "energy_q06",
    }
    g3_candidates = {
        target: raw[[source]].rename(columns={source: "kpx_group_3"})
        for source, target in rename.items()
    }
    weighted = (
        0.20 * g3_candidates["lgb_q07"]["kpx_group_3"]
        + 0.075 * g3_candidates["shared_l1"]["kpx_group_3"]
        + 0.425 * g3_candidates["shared_q07"]["kpx_group_3"]
        + 0.025 * g3_candidates["top200_q07"]["kpx_group_3"]
        + 0.275 * g3_candidates["energy_q06"]["kpx_group_3"]
    )
    g3_base = pd.DataFrame(
        {
            "kpx_group_3": np.clip(
                1.25 * weighted.to_numpy(dtype=float) - 1200.0,
                0.0,
                1.02 * CAPACITY_KWH["kpx_group_3"],
            )
        },
        index=g3_index,
    )
    input_paths = {**{f"component_{name}": path for name, path in paths.items()}}
    input_paths.update({"base_2023": base_path, "g3_candidates_2023h2": g3_path})
    return candidates, base, g3_candidates, g3_base, input_paths


def _load_period_sources(
    artifact_root: Path, period: str, expected_index: pd.DatetimeIndex
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, Path]]:
    paths = _component_paths(artifact_root, period)
    candidates = {
        name: _read_prediction(path, expected_index, required_columns=TARGET_COLS)
        for name, path in paths.items()
    }
    if period == "2024":
        base_path = artifact_root / "oof" / "gate2024_locked_v3_cf_fix.parquet"
    elif period == "2025":
        base_path = (
            artifact_root / "final_cf_fix" / "predictions" / "corrected_v3_test.parquet"
        )
    else:
        raise ValueError(period)
    base = _read_prediction(base_path, expected_index, required_columns=TARGET_COLS)
    input_paths = {**{f"component_{name}_{period}": path for name, path in paths.items()}}
    input_paths[f"base_{period}"] = base_path
    return candidates, base, input_paths


def _meta_features(
    candidates: Mapping[str, pd.DataFrame], base: pd.DataFrame, group: str
) -> pd.DataFrame:
    if tuple(candidates) != COMMON_CANDIDATES:
        raise AssertionError("candidate names/order changed")
    capacity = CAPACITY_KWH[group]
    index = base.index
    values: dict[str, np.ndarray] = {
        "base_cf": base[group].to_numpy(dtype=float) / capacity
    }
    stacked: list[np.ndarray] = []
    for name in COMMON_CANDIDATES:
        frame = candidates[name]
        if not frame.index.equals(index):
            raise AssertionError(f"{group}/{name} candidate index changed")
        array = frame[group].to_numpy(dtype=float) / capacity
        values[f"pred__{name}_cf"] = array
        stacked.append(array)
    matrix = np.column_stack(stacked)
    values["candidate_mean_cf"] = np.mean(matrix, axis=1)
    values["candidate_std_cf"] = np.std(matrix, axis=1)
    values["candidate_min_cf"] = np.min(matrix, axis=1)
    values["candidate_max_cf"] = np.max(matrix, axis=1)
    values["q07_minus_l1_cf"] = values["pred__lgb_q07_cf"] - values["pred__lgb_l1_cf"]
    values["shared_q07_minus_l1_cf"] = (
        values["pred__shared_q07_cf"] - values["pred__shared_l1_cf"]
    )
    hour_angle = 2.0 * np.pi * index.hour.to_numpy() / 24.0
    day_angle = 2.0 * np.pi * (index.dayofyear.to_numpy() - 1.0) / 365.25
    values["hour_sin"] = np.sin(hour_angle)
    values["hour_cos"] = np.cos(hour_angle)
    values["day_sin"] = np.sin(day_angle)
    values["day_cos"] = np.cos(day_angle)
    result = pd.DataFrame(values, index=index)
    if tuple(result.columns) != (
        "base_cf",
        "pred__lgb_l1_cf",
        "pred__lgb_q07_cf",
        "pred__shared_l1_cf",
        "pred__shared_q07_cf",
        "pred__top200_q07_cf",
        "pred__energy_q06_cf",
        "candidate_mean_cf",
        "candidate_std_cf",
        "candidate_min_cf",
        "candidate_max_cf",
        "q07_minus_l1_cf",
        "shared_q07_minus_l1_cf",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
    ):
        raise AssertionError("meta feature schema changed")
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise AssertionError("meta features contain non-finite values")
    return result


def _group_summary(
    actual: pd.Series, prediction: pd.Series, group: str
) -> dict[str, Any]:
    details = group_metrics(
        actual,
        prediction,
        CAPACITY_KWH[group],
        group_name=group,
    )
    result = details.as_dict()
    result["score"] = 0.5 * (details.one_minus_nmae + details.ficr)
    return result


def _comparison(
    actual: pd.Series,
    baseline: pd.Series,
    candidate: pd.Series,
    group: str,
    segments: Mapping[str, pd.DatetimeIndex],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, index in segments.items():
        base_metrics = _group_summary(actual.loc[index], baseline.loc[index], group)
        candidate_metrics = _group_summary(
            actual.loc[index], candidate.loc[index], group
        )
        result[name] = {
            "baseline": base_metrics,
            "candidate": candidate_metrics,
            "delta": candidate_metrics["score"] - base_metrics["score"],
        }
    return result


def _fit_apply(
    method: str,
    config: ProbabilisticDecisionConfig,
    features: pd.DataFrame,
    actual: pd.Series,
    base: pd.Series,
    fit_index: pd.DatetimeIndex,
    application_index: pd.DatetimeIndex,
    group: str,
) -> tuple[pd.Series, dict[str, Any], ProbabilisticFICRDecision]:
    if len(fit_index.intersection(application_index)):
        raise AssertionError("fit/application indexes overlap")
    decision = ProbabilisticFICRDecision(method, config=config).fit(
        features.loc[fit_index],
        actual.loc[fit_index],
        base.loc[fit_index],
        capacity_kwh=CAPACITY_KWH[group],
    )
    prediction = decision.predict(
        features.loc[application_index], base.loc[application_index]
    )
    return prediction, decision.metadata(), decision


def _validate_comparisons(
    comparisons: Mapping[str, Any], required: Mapping[str, Sequence[str]]
) -> None:
    if set(comparisons) != set(TARGET_COLS):
        raise AssertionError("comparison group set changed")
    for group in TARGET_COLS:
        if set(comparisons[group]) != set(METHODS):
            raise AssertionError(f"{group} method set changed")
        for method in METHODS:
            values = comparisons[group][method]
            if set(values) != set(required[group]):
                raise AssertionError(f"{group}/{method} segment set changed")
            for segment in required[group]:
                record = values[segment]
                expected = float(record["candidate"]["score"]) - float(
                    record["baseline"]["score"]
                )
                if float(record["delta"]) != expected:
                    raise AssertionError(
                        f"{group}/{method}/{segment} stored delta changed"
                    )


def _select_recipe(comparisons: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    _validate_comparisons(comparisons, STAGE1_REQUIRED_SEGMENTS)
    order = {method: position for position, method in enumerate(METHODS)}
    recipe: dict[str, str] = {}
    diagnostics: dict[str, Any] = {}
    for group in TARGET_COLS:
        stats: dict[str, Any] = {}
        eligible: list[str] = []
        for method in METHODS:
            deltas = [
                float(comparisons[group][method][segment]["delta"])
                for segment in STAGE1_REQUIRED_SEGMENTS[group]
            ]
            stats[method] = {
                "deltas": dict(zip(STAGE1_REQUIRED_SEGMENTS[group], deltas)),
                "minimum": min(deltas),
                "mean": float(np.mean(deltas)),
                "all_strictly_positive": all(delta > 0.0 for delta in deltas),
            }
            if stats[method]["all_strictly_positive"]:
                eligible.append(method)
        if eligible:
            selected = max(
                eligible,
                key=lambda method: (
                    stats[method]["minimum"],
                    stats[method]["mean"],
                    -order[method],
                ),
            )
        else:
            selected = "identity"
        recipe[group] = selected
        diagnostics[group] = {
            "selected": selected,
            "eligible_methods": eligible,
            "method_statistics": stats,
            "selection_uses_2024": False,
        }
    return recipe, diagnostics


def _stage1_input_paths(raw_dir: Path, artifact_root: Path) -> dict[str, Any]:
    component_paths = _component_paths(artifact_root, "2023")
    return {
        "labels_prefix": _snapshot_label_prefix(raw_dir / "train" / "train_labels.csv"),
        **{
            f"component_{name}": _snapshot_file(path)
            for name, path in component_paths.items()
        },
        "base_2023": _snapshot_file(
            artifact_root / "oof" / "dev2023_locked_v3.parquet"
        ),
        "g3_candidates_2023h2": _snapshot_file(
            artifact_root / "oof" / "g3dev2023h2_candidates.parquet"
        ),
    }


def _provenance_paths(preregister_path: Path) -> dict[str, Path]:
    return {
        "runner": Path(__file__).resolve(),
        "probabilistic": PROJECT_DIR / "src" / "probabilistic.py",
        "metric": PROJECT_DIR / "src" / "metric.py",
        "manifest": PROJECT_DIR / "src" / "manifest.py",
        "test": PROJECT_DIR / "tests" / "test_ficr_bayes_decision_strict.py",
        "preregister": preregister_path.resolve(),
    }


def _snapshot_named_files(paths: Mapping[str, Path]) -> dict[str, Any]:
    return {name: _snapshot_file(path) for name, path in paths.items()}


def _stage1(
    *,
    raw_dir: Path,
    artifact_root: Path,
    out_dir: Path,
    preregister_path: Path,
    preregister: Mapping[str, Any],
    config: ProbabilisticDecisionConfig,
) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"Stage1 requires a new output directory: {out_dir}")
    out_dir.mkdir(parents=True)
    _copy_exclusive(preregister_path, out_dir / "preregister.json")

    provenance_before = _snapshot_named_files(_provenance_paths(preregister_path))
    input_before = _stage1_input_paths(raw_dir, artifact_root)
    labels, label_evidence = _read_bounded_labels(
        raw_dir / "train" / "train_labels.csv"
    )
    labels_2023 = labels.loc[_year_index(2023)]
    candidates, base, g3_candidates, g3_base, _ = _load_2023_sources(artifact_root)
    features = {
        group: _meta_features(candidates, base, group) for group in TARGET_COLS[:2]
    }
    features["kpx_group_3"] = _meta_features(
        g3_candidates, g3_base, "kpx_group_3"
    )

    rolling = _stage1_rolling_contract()
    g12_application = rolling["g12_application"]
    g3_h2 = rolling["g3_h2"]
    g3_month_applications = rolling["g3_directions"]
    g3_application = rolling["g3_application"]
    method_outputs = {
        method: pd.DataFrame(np.nan, index=_year_index(2023), columns=TARGET_COLS)
        for method in METHODS
    }
    fit_records: list[dict[str, Any]] = []
    for method in METHODS:
        for group in TARGET_COLS[:2]:
            for direction, fit_index, application_index in rolling[
                "g12_directions"
            ]:
                print(f"stage1 {group} {method} {direction}", flush=True)
                prediction, metadata, _ = _fit_apply(
                    method,
                    config,
                    features[group],
                    labels_2023[group],
                    base[group],
                    fit_index,
                    application_index,
                    group,
                )
                method_outputs[method].loc[application_index, group] = prediction
                fit_records.append(
                    {
                        "group": group,
                        "method": method,
                        "direction": direction,
                        "fit_start": fit_index.min(),
                        "fit_end": fit_index.max(),
                        "application_start": application_index.min(),
                        "application_end": application_index.max(),
                        "fit_application_disjoint": True,
                        "strict_temporal_direction": True,
                        "fit_end_before_application_start": bool(
                            fit_index.max() < application_index.min()
                        ),
                        "fit_score_calculated": False,
                        "metadata": metadata,
                    }
                )
        for month_name, fit_index, application_index in g3_month_applications:
            group = "kpx_group_3"
            direction = f"expanding_fit_{month_name}_apply"
            print(f"stage1 {group} {method} {direction}", flush=True)
            prediction, metadata, _ = _fit_apply(
                method,
                config,
                features[group],
                labels_2023.loc[g3_h2, group],
                g3_base[group],
                fit_index,
                application_index,
                group,
            )
            method_outputs[method].loc[application_index, group] = prediction
            fit_records.append(
                {
                    "group": group,
                    "method": method,
                    "direction": direction,
                    "fit_start": fit_index.min(),
                    "fit_end": fit_index.max(),
                    "application_start": application_index.min(),
                    "application_end": application_index.max(),
                    "fit_application_disjoint": True,
                    "strict_temporal_direction": True,
                    "fit_end_before_application_start": bool(
                        fit_index.max() < application_index.min()
                    ),
                    "fit_score_calculated": False,
                    "metadata": metadata,
                }
            )

    baseline = pd.DataFrame(np.nan, index=_year_index(2023), columns=TARGET_COLS)
    baseline.loc[:, list(TARGET_COLS[:2])] = base
    baseline.loc[g3_h2, "kpx_group_3"] = g3_base["kpx_group_3"]
    comparisons: dict[str, Any] = {}
    for group in TARGET_COLS:
        if group in TARGET_COLS[:2]:
            actual = labels_2023.loc[g12_application, group]
            base_series = baseline.loc[g12_application, group]
            segments = {"full": g12_application, **rolling["g12_segments"]}
        else:
            actual = labels_2023.loc[g3_application, group]
            base_series = baseline.loc[g3_application, group]
            segments = {
                "full": g3_application,
                **{
                    month_name: application_index
                    for month_name, _, application_index in g3_month_applications
                },
            }
        comparisons[group] = {
            method: _comparison(
                actual,
                base_series,
                method_outputs[method].loc[actual.index, group],
                group,
                segments,
            )
            for method in METHODS
        }
    recipe, selection = _select_recipe(comparisons)

    baseline_path = out_dir / "oof" / "stage1_baseline_2023.parquet"
    _atomic_parquet(baseline, baseline_path)
    output_paths = [baseline_path]
    for method in METHODS:
        path = out_dir / "oof" / f"stage1_crossfit_2023__{method}.parquet"
        _atomic_parquet(method_outputs[method], path)
        output_paths.append(path)

    provenance_after = _snapshot_named_files(_provenance_paths(preregister_path))
    input_after = _stage1_input_paths(raw_dir, artifact_root)
    _assert_snapshot_equal(
        provenance_before, provenance_after, name="Stage1 provenance"
    )
    _assert_snapshot_equal(input_before, input_after, name="Stage1 inputs")
    result = {
        "schema_version": 1,
        "experiment_id": preregister["experiment_id"],
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "candidate_methods": METHODS,
        "candidate_count": len(METHODS),
        "config": asdict(config),
        "2024_read": False,
        "label_prefix_evidence": label_evidence,
        "fit_application_overlap_count": 0,
        "same_row_fit_score_reported": False,
        "fit_records": fit_records,
        "comparisons": comparisons,
        "comparisons_sha256": _canonical_sha256(comparisons),
        "selection": selection,
        "locked_recipe": recipe,
        "locked_recipe_sha256": _canonical_sha256(recipe),
        "nonidentity_groups": [group for group, method in recipe.items() if method != "identity"],
        "selection_rule": preregister["stage1_pre2024_selection"][
            "per_group_selection_rule"
        ],
        "provenance_before": provenance_before,
        "provenance_after": provenance_after,
        "provenance_snapshot_sha256": _canonical_sha256(provenance_before),
        "stage1_input_snapshot_before": input_before,
        "stage1_input_snapshot_after": input_after,
        "stage1_input_snapshot_sha256": _canonical_sha256(input_before),
        "snapshots_unchanged": True,
        "outputs_before_result": [describe_file(path) for path in output_paths],
        "leaderboard_score_claim": False,
        "untouched_gate_claim": False,
    }
    result_path = out_dir / "stage1_results.json"
    _write_json(result_path, result)
    lock = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "stage1_results_sha256": sha256_file(result_path),
        "comparisons_sha256": result["comparisons_sha256"],
        "locked_recipe": recipe,
        "locked_recipe_sha256": result["locked_recipe_sha256"],
        "provenance_snapshot_sha256": result["provenance_snapshot_sha256"],
        "stage1_input_snapshot_sha256": result["stage1_input_snapshot_sha256"],
        "candidate_and_hyperparameters_frozen": True,
        "no_2024_reselection_or_retuning": True,
    }
    _write_json(out_dir / "stage1_recipe_lock.json", lock)
    print(f"Stage1 locked recipe: {recipe}", flush=True)
    return lock


def _load_stage1_lock(out_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    result_path = out_dir / "stage1_results.json"
    lock_path = out_dir / "stage1_recipe_lock.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["preregister_sha256"] != PREREGISTER_SHA256:
        raise AssertionError("Stage1 lock preregister hash changed")
    if lock["stage1_results_sha256"] != sha256_file(result_path):
        raise AssertionError("Stage1 result changed after recipe lock")
    if not lock.get("candidate_and_hyperparameters_frozen") or not lock.get(
        "no_2024_reselection_or_retuning"
    ):
        raise AssertionError("Stage1 lock is not immutable")
    if tuple(result.get("candidate_methods", ())) != METHODS or int(
        result.get("candidate_count", -1)
    ) != len(METHODS):
        raise AssertionError("Stage1 candidate family changed")
    comparisons = result["comparisons"]
    if _canonical_sha256(comparisons) != result["comparisons_sha256"]:
        raise AssertionError("Stage1 comparison digest changed")
    if lock["comparisons_sha256"] != result["comparisons_sha256"]:
        raise AssertionError("Stage1 lock comparison digest changed")
    recipe, selection = _select_recipe(comparisons)
    if recipe != result["locked_recipe"] or recipe != lock["locked_recipe"]:
        raise AssertionError("Stage1 recipe differs from independently recomputed rule")
    if selection != result["selection"]:
        raise AssertionError("Stage1 selection diagnostics changed")
    if _canonical_sha256(recipe) != lock["locked_recipe_sha256"]:
        raise AssertionError("Stage1 recipe digest changed")
    for key, snapshot_key in (
        ("provenance_before", "provenance_snapshot_sha256"),
        ("stage1_input_snapshot_before", "stage1_input_snapshot_sha256"),
    ):
        snapshot = result[key]
        if _canonical_sha256(snapshot) != result[snapshot_key]:
            raise AssertionError(f"Stage1 {key} digest changed")
        if result[snapshot_key] != lock[snapshot_key]:
            raise AssertionError(f"Stage1 lock {snapshot_key} changed")
        _assert_snapshot_equal(
            snapshot, _refresh_snapshot(snapshot), name=f"current {key}"
        )
    return lock, result


def _stage2_input_snapshot(raw_dir: Path, artifact_root: Path) -> dict[str, Any]:
    paths = _component_paths(artifact_root, "2024")
    return {
        "labels_full": _snapshot_file(raw_dir / "train" / "train_labels.csv"),
        **{
            f"component_{name}_2024": _snapshot_file(path)
            for name, path in paths.items()
        },
        "base_2024": _snapshot_file(
            artifact_root / "oof" / "gate2024_locked_v3_cf_fix.parquet"
        ),
    }


def _stage2_promoted(
    comparisons: Mapping[str, Any], recipe: Mapping[str, str]
) -> tuple[bool, dict[str, Any]]:
    selected = [group for group, method in recipe.items() if method != "identity"]
    audit: dict[str, Any] = {}
    if set(comparisons) != set(selected):
        raise AssertionError("Stage2 comparison group set differs from locked recipe")
    for group in selected:
        values = comparisons[group]
        if set(values) != set(STAGE2_REQUIRED_SEGMENTS):
            raise AssertionError(f"Stage2 {group} segment set changed")
        deltas: dict[str, float] = {}
        for segment in STAGE2_REQUIRED_SEGMENTS:
            record = values[segment]
            expected = float(record["candidate"]["score"]) - float(
                record["baseline"]["score"]
            )
            if float(record["delta"]) != expected:
                raise AssertionError(f"Stage2 {group}/{segment} delta changed")
            deltas[segment] = expected
        audit[group] = {
            "method": recipe[group],
            "deltas": deltas,
            "all_strictly_positive": all(value > 0.0 for value in deltas.values()),
        }
    promoted = bool(selected) and all(
        audit[group]["all_strictly_positive"] for group in selected
    )
    return promoted, audit


def _write_stage2_result_and_lock(
    out_dir: Path, result: Mapping[str, Any]
) -> dict[str, Any]:
    result_path = out_dir / "stage2_results.json"
    _write_json(result_path, result)
    lock = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "stage1_recipe_lock_sha256": sha256_file(
            out_dir / "stage1_recipe_lock.json"
        ),
        "stage2_results_sha256": sha256_file(result_path),
        "locked_recipe": result["locked_recipe"],
        "recipe_promoted": bool(result["recipe_promoted"]),
        "promotion_audit_sha256": _canonical_sha256(result["promotion_audit"]),
        "no_2024_reselection_or_retuning": True,
        "csv_allowed": bool(result["recipe_promoted"]),
    }
    _write_json(out_dir / "stage2_promotion_lock.json", lock)
    return lock


def _stage2(
    *,
    raw_dir: Path,
    artifact_root: Path,
    out_dir: Path,
    config: ProbabilisticDecisionConfig,
) -> dict[str, Any]:
    lock, stage1_result = _load_stage1_lock(out_dir)
    if (out_dir / "stage2_results.json").exists() or (
        out_dir / "stage2_promotion_lock.json"
    ).exists():
        raise FileExistsError("Stage2 outputs already exist")
    recipe = dict(lock["locked_recipe"])
    selected = [group for group, method in recipe.items() if method != "identity"]
    if not selected:
        result = {
            "schema_version": 1,
            "created_utc": utc_now(),
            "stage1_recipe_lock_sha256": sha256_file(
                out_dir / "stage1_recipe_lock.json"
            ),
            "locked_recipe": recipe,
            "2024_read": False,
            "comparisons": {},
            "promotion_audit": {},
            "recipe_promoted": False,
            "reason": "Stage1 selected identity for every group",
            "no_2024_reselection_or_retuning": True,
        }
        return _write_stage2_result_and_lock(out_dir, result)

    input_before = _stage2_input_snapshot(raw_dir, artifact_root)
    labels = _read_full_labels(raw_dir / "train" / "train_labels.csv")
    labels_2023 = labels.loc[_year_index(2023)]
    labels_2024 = labels.loc[_year_index(2024)]
    candidates_2023, base_2023, g3_candidates, g3_base, _ = _load_2023_sources(
        artifact_root
    )
    candidates_2024, base_2024, _ = _load_period_sources(
        artifact_root, "2024", _year_index(2024)
    )
    features_2023 = {
        group: _meta_features(candidates_2023, base_2023, group)
        for group in TARGET_COLS[:2]
    }
    features_2023["kpx_group_3"] = _meta_features(
        g3_candidates, g3_base, "kpx_group_3"
    )
    features_2024 = {
        group: _meta_features(candidates_2024, base_2024, group)
        for group in TARGET_COLS
    }
    candidate_2024 = base_2024.copy()
    models: dict[str, Any] = {}
    fit_metadata: dict[str, Any] = {}
    for group in selected:
        method = recipe[group]
        if group in TARGET_COLS[:2]:
            fit_features = features_2023[group]
            fit_actual = labels_2023[group]
            fit_base = base_2023[group]
        else:
            fit_features = features_2023[group]
            fit_actual = labels_2023.loc[fit_features.index, group]
            fit_base = g3_base[group]
        if len(fit_features.index.intersection(features_2024[group].index)):
            raise AssertionError("Stage2 transfer fit/application overlap")
        print(f"stage2 fixed transfer {group} {method}", flush=True)
        decision = ProbabilisticFICRDecision(method, config=config).fit(
            fit_features,
            fit_actual,
            fit_base,
            capacity_kwh=CAPACITY_KWH[group],
        )
        candidate_2024[group] = decision.predict(
            features_2024[group], base_2024[group]
        )
        models[group] = decision
        fit_metadata[group] = decision.metadata()

    baseline_path = out_dir / "oof" / "stage2_baseline_2024.parquet"
    candidate_path = out_dir / "oof" / "stage2_fixed_candidate_2024.parquet"
    model_path = out_dir / "models" / "stage2_transfer_models.joblib"
    _atomic_parquet(base_2024, baseline_path)
    _atomic_parquet(candidate_2024, candidate_path)
    _atomic_joblib(models, model_path)
    h1 = _interval(YEAR_2024_START, H2_2024_START - pd.Timedelta(hours=1))
    h2 = _interval(H2_2024_START, YEAR_2024_END)
    segments = {"full": _year_index(2024), "H1": h1, "H2": h2}
    comparisons = {
        group: _comparison(
            labels_2024[group],
            base_2024[group],
            candidate_2024[group],
            group,
            segments,
        )
        for group in selected
    }
    promoted, promotion_audit = _stage2_promoted(comparisons, recipe)
    input_after = _stage2_input_snapshot(raw_dir, artifact_root)
    _assert_snapshot_equal(input_before, input_after, name="Stage2 inputs")
    result = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "stage1_recipe_lock_sha256": sha256_file(
            out_dir / "stage1_recipe_lock.json"
        ),
        "stage1_results_sha256": lock["stage1_results_sha256"],
        "locked_recipe": recipe,
        "2024_read": True,
        "fit_period": {
            "kpx_group_1": "2023",
            "kpx_group_2": "2023",
            "kpx_group_3": "2023H2",
        },
        "application_period": "2024",
        "fit_application_overlap_count": 0,
        "same_2024_fit_score_reported": False,
        "fit_metadata": fit_metadata,
        "comparisons": comparisons,
        "comparisons_sha256": _canonical_sha256(comparisons),
        "promotion_audit": promotion_audit,
        "recipe_promoted": promoted,
        "no_2024_reselection_or_retuning": True,
        "stage2_input_snapshot_before": input_before,
        "stage2_input_snapshot_after": input_after,
        "stage2_input_snapshot_sha256": _canonical_sha256(input_before),
        "outputs_before_result": [
            describe_file(baseline_path),
            describe_file(candidate_path),
            describe_file(model_path),
        ],
        "leaderboard_score_claim": False,
        "untouched_gate_claim": False,
        "stage1_selection": stage1_result["selection"],
    }
    promotion_lock = _write_stage2_result_and_lock(out_dir, result)
    print(f"Stage2 recipe promoted: {promoted}", flush=True)
    return promotion_lock


def _load_stage2_lock(out_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    stage1_lock, _ = _load_stage1_lock(out_dir)
    result_path = out_dir / "stage2_results.json"
    lock_path = out_dir / "stage2_promotion_lock.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["preregister_sha256"] != PREREGISTER_SHA256:
        raise AssertionError("Stage2 lock preregister hash changed")
    if lock["stage1_recipe_lock_sha256"] != sha256_file(
        out_dir / "stage1_recipe_lock.json"
    ):
        raise AssertionError("Stage1 recipe lock changed before final")
    if lock["stage2_results_sha256"] != sha256_file(result_path):
        raise AssertionError("Stage2 result changed after promotion lock")
    if lock["locked_recipe"] != stage1_lock["locked_recipe"] or result[
        "locked_recipe"
    ] != stage1_lock["locked_recipe"]:
        raise AssertionError("Stage2 recipe differs from Stage1 lock")
    promoted, audit = _stage2_promoted(
        result.get("comparisons", {}), result["locked_recipe"]
    )
    if promoted != bool(result["recipe_promoted"]) or audit != result[
        "promotion_audit"
    ]:
        raise AssertionError("Stage2 promotion differs from recomputed rule")
    if promoted != bool(lock["recipe_promoted"]) or promoted != bool(
        lock["csv_allowed"]
    ):
        raise AssertionError("Stage2 promotion lock changed")
    if _canonical_sha256(audit) != lock["promotion_audit_sha256"]:
        raise AssertionError("Stage2 promotion audit digest changed")
    if not lock.get("no_2024_reselection_or_retuning"):
        raise AssertionError("Stage2 no-reselection guard changed")
    if result.get("2024_read"):
        before = result["stage2_input_snapshot_before"]
        after = result["stage2_input_snapshot_after"]
        _assert_snapshot_equal(before, after, name="recorded Stage2 inputs")
        if _canonical_sha256(before) != result["stage2_input_snapshot_sha256"]:
            raise AssertionError("Stage2 input digest changed")
        _assert_snapshot_equal(
            before, _refresh_snapshot(before), name="current Stage2 inputs"
        )
    return lock, result


def _load_2025_sources(
    raw_dir: Path, artifact_root: Path
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    sample_path = raw_dir / "sample_submission.csv"
    sample = pd.read_csv(
        sample_path,
        encoding="utf-8-sig",
        dtype={"forecast_id": "string", "forecast_kst_dtm": "string"},
    )
    if tuple(sample.columns) != ("forecast_id", "forecast_kst_dtm", *TARGET_COLS):
        raise AssertionError("sample submission schema changed")
    index = pd.DatetimeIndex(
        pd.to_datetime(sample["forecast_kst_dtm"], errors="raise"),
        name="forecast_kst_dtm",
    )
    if not index.equals(_year_index(2025)) or len(sample) != 8760:
        raise AssertionError("sample submission horizon changed")
    candidates, base, paths = _load_period_sources(
        artifact_root, "2025", index
    )
    paths["sample_submission"] = sample_path
    return candidates, base, sample, paths


def _verify_submission(
    path: Path, sample: pd.DataFrame, expected: pd.DataFrame
) -> dict[str, Any]:
    with path.open("rb") as stream:
        if stream.read(3) != b"\xef\xbb\xbf":
            raise AssertionError("submission lacks UTF-8-SIG BOM")
    observed = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"forecast_id": "string", "forecast_kst_dtm": "string"},
    )
    if tuple(observed.columns) != tuple(sample.columns) or len(observed) != 8760:
        raise AssertionError("submission schema/row count changed")
    for column in ("forecast_id", "forecast_kst_dtm"):
        if not np.array_equal(
            observed[column].astype(str).to_numpy(),
            sample[column].astype(str).to_numpy(),
        ):
            raise AssertionError(f"submission {column} changed")
    values = observed.loc[:, list(TARGET_COLS)].to_numpy(dtype=float)
    expected_values = expected.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise AssertionError("submission contains non-finite predictions")
    for position, group in enumerate(TARGET_COLS):
        if np.any(values[:, position] < 0.0) or np.any(
            values[:, position] > 1.02 * CAPACITY_KWH[group] + 5.1e-7
        ):
            raise AssertionError(f"submission violates {group} clip")
    max_difference = float(np.max(np.abs(values - expected_values)))
    if max_difference > 5.1e-7:
        raise AssertionError("submission six-decimal readback changed")
    return {
        "rows": len(observed),
        "sample_schema_id_time_exact": True,
        "utf8_sig": True,
        "finite": True,
        "capacity_clip": True,
        "max_readback_abs_diff": max_difference,
        "sha256": sha256_file(path),
    }


def _deduplicate_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        key = (str(record["path"]), str(record.get("snapshot_kind", "whole_file")))
        if key not in seen:
            result.append(dict(record))
            seen.add(key)
    return result


def _finalize(
    *,
    raw_dir: Path,
    artifact_root: Path,
    out_dir: Path,
    config: ProbabilisticDecisionConfig,
) -> dict[str, Any]:
    stage2_lock, stage2_result = _load_stage2_lock(out_dir)
    manifest_path = out_dir / "manifest.json"
    final_result_path = out_dir / "final_results.json"
    if manifest_path.exists() or final_result_path.exists():
        raise FileExistsError("final outputs already exist")
    recipe = dict(stage2_lock["locked_recipe"])
    final_inputs: dict[str, Any] = {}
    output_details: dict[str, Any] = {}
    if not stage2_lock["recipe_promoted"]:
        final_result = {
            "schema_version": 1,
            "created_utc": utc_now(),
            "recipe_promoted": False,
            "locked_recipe": recipe,
            "2025_read": False,
            "final_fit_performed": False,
            "submission_created": False,
            "reason": "The complete locked recipe failed preregistered promotion conditions",
            "leaderboard_score_claim": False,
        }
        _write_json(final_result_path, final_result)
    else:
        labels = _read_full_labels(raw_dir / "train" / "train_labels.csv")
        labels_2024 = labels.loc[_year_index(2024)]
        candidates_2024, base_2024, paths_2024 = _load_period_sources(
            artifact_root, "2024", _year_index(2024)
        )
        candidates_2025, base_2025, sample, paths_2025 = _load_2025_sources(
            raw_dir, artifact_root
        )
        final_inputs = {
            "labels_full": _snapshot_file(raw_dir / "train" / "train_labels.csv"),
            **{name: _snapshot_file(path) for name, path in paths_2024.items()},
            **{name: _snapshot_file(path) for name, path in paths_2025.items()},
        }
        features_2024 = {
            group: _meta_features(candidates_2024, base_2024, group)
            for group in TARGET_COLS
        }
        features_2025 = {
            group: _meta_features(candidates_2025, base_2025, group)
            for group in TARGET_COLS
        }
        candidate_2025 = base_2025.copy()
        models: dict[str, Any] = {}
        metadata: dict[str, Any] = {}
        for group, method in recipe.items():
            if method == "identity":
                metadata[group] = {"method": "identity", "fit_score_calculated": False}
                continue
            print(f"final 2024 OOF -> 2025 {group} {method}", flush=True)
            decision = ProbabilisticFICRDecision(method, config=config).fit(
                features_2024[group],
                labels_2024[group],
                base_2024[group],
                capacity_kwh=CAPACITY_KWH[group],
            )
            candidate_2025[group] = decision.predict(
                features_2025[group], base_2025[group]
            )
            models[group] = decision
            metadata[group] = decision.metadata()
        for group in TARGET_COLS:
            candidate_2025[group] = np.clip(
                candidate_2025[group].to_numpy(dtype=float),
                0.0,
                1.02 * CAPACITY_KWH[group],
            )
        prediction_path = out_dir / "predictions" / "ficr_bayes_decision_2025.parquet"
        model_path = out_dir / "models" / "final_2024_oof_calibrators.joblib"
        csv_path = out_dir / "ficr_bayes_decision_2025.csv"
        _atomic_parquet(candidate_2025, prediction_path)
        _atomic_joblib(models, model_path)
        submission = sample.copy()
        for group in TARGET_COLS:
            submission[group] = candidate_2025[group].to_numpy(dtype=float)
        _atomic_csv(submission, csv_path)
        verification = _verify_submission(csv_path, sample, candidate_2025)
        output_details = {
            "prediction": describe_file(prediction_path),
            "models": describe_file(model_path),
            "submission": describe_file(csv_path),
            "submission_verification": verification,
        }
        final_result = {
            "schema_version": 1,
            "created_utc": utc_now(),
            "recipe_promoted": True,
            "locked_recipe": recipe,
            "2025_read": True,
            "final_fit_performed": True,
            "fit_period": "2024 corrected-v3 OOF",
            "application_period": "unlabeled 2025 full-model predictions",
            "fit_application_overlap_count": 0,
            "same_2024_fit_score_reported": False,
            "fit_metadata": metadata,
            "submission_created": True,
            "outputs": output_details,
            "leaderboard_score_claim": False,
        }
        _write_json(final_result_path, final_result)

    stage1_result = json.loads(
        (out_dir / "stage1_results.json").read_text(encoding="utf-8")
    )
    input_records = [
        *stage1_result["provenance_before"].values(),
        *stage1_result["stage1_input_snapshot_before"].values(),
    ]
    if stage2_result.get("2024_read"):
        input_records.extend(stage2_result["stage2_input_snapshot_before"].values())
    input_records.extend(final_inputs.values())
    outputs = sorted(
        path
        for path in out_dir.rglob("*")
        if path.is_file() and path != manifest_path
    )
    manifest = {
        "schema_version": 1,
        "artifact_type": "ficr_bayes_decision_strict_forward_v1",
        "created_utc": utc_now(),
        "command": [sys.executable, *sys.argv],
        "runtime": {
            "packages": package_versions(),
            "git": git_state(PROJECT_DIR),
        },
        "preregister_sha256": PREREGISTER_SHA256,
        "candidate_methods": METHODS,
        "candidate_count": len(METHODS),
        "locked_recipe": recipe,
        "recipe_promoted": bool(stage2_lock["recipe_promoted"]),
        "stage1_recipe_lock": describe_file(out_dir / "stage1_recipe_lock.json"),
        "stage2_promotion_lock": describe_file(out_dir / "stage2_promotion_lock.json"),
        "inputs": _deduplicate_records(input_records),
        "outputs": [describe_file(path) for path in outputs],
        "results": {
            "2024_read": bool(stage2_result.get("2024_read")),
            "2025_read": bool(final_result["2025_read"]),
            "final_fit_performed": bool(final_result["final_fit_performed"]),
            "submission_created": bool(final_result["submission_created"]),
            "output_details": output_details,
        },
        "contracts": {
            "2024_candidate_or_score_used_for_selection": False,
            "same_row_fit_score_reported": False,
            "no_2024_reselection_or_retuning": True,
            "csv_requires_complete_recipe_promotion": True,
            "untouched_gate_claim": False,
            "leaderboard_score_claim": False,
        },
    }
    _write_json(manifest_path, manifest)
    print(
        f"Finalized promoted={stage2_lock['recipe_promoted']} manifest={manifest_path}",
        flush=True,
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    raw_dir = args.raw_dir.expanduser().resolve()
    artifact_root = args.artifact_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    preregister_path = args.preregister.expanduser().resolve()
    preregister, config = _verify_preregister(preregister_path)
    if out_dir == artifact_root or artifact_root not in out_dir.parents:
        raise ValueError("out-dir must be a distinct child of artifact-root")
    if args.stage == "stage1":
        _stage1(
            raw_dir=raw_dir,
            artifact_root=artifact_root,
            out_dir=out_dir,
            preregister_path=preregister_path,
            preregister=preregister,
            config=config,
        )
    elif args.stage == "stage2":
        _stage2(
            raw_dir=raw_dir,
            artifact_root=artifact_root,
            out_dir=out_dir,
            config=config,
        )
    else:
        _finalize(
            raw_dir=raw_dir,
            artifact_root=artifact_root,
            out_dir=out_dir,
            config=config,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

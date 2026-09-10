"""Strict-forward turbine-level SCADA power decomposition experiment.

Stage 1 is the only selection stage.  It reads bounded raw NWP/labels/SCADA
prefixes, evaluates the preregistered six blends per group, and creates an
immutable O_EXCL lock.  Stage 2 may open 2024 data only after independently
recomputing that lock.  A 2025 CSV is emitted only when the exact locked
candidate improves every preregistered 2024 slice.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Mapping, Sequence

import joblib
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.run_shared_q07_multiseed import (  # noqa: E402
    COMPONENTS,
    DEV_COMPONENT_FILES,
    EXPECTED_ROWS_PRE2024,
    OFFICIAL_LABEL_PREFIX_SHA256,
    OFFICIAL_STAGE1_PREFIX_BYTES,
    YEAR_2022_END,
    YEAR_2022_START,
    YEAR_2023_END,
    YEAR_2023_START,
    YEAR_2024_END,
    YEAR_2024_START,
    _BoundedRawReader,
    _assemble_group,
    _canonical_sha256,
    _csv_prefix_identity,
    _frame_sha256,
    _read_labels,
    _read_stage1_raw_features,
)
from src.manifest import (  # noqa: E402
    describe_file,
    git_state,
    package_versions,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from src.metric import CAPACITY_KWH, TARGET_COLS, group_metrics  # noqa: E402


PREREGISTER_SHA256 = (
    "d631cf7ef897e33bb54e87ea0df89343c96fc25120a91843a56ec8df2e11847f"
)
H2_2023_START = pd.Timestamp("2023-07-01 01:00:00")
Q4_2023_START = pd.Timestamp("2023-10-01 01:00:00")
H2_2024_START = pd.Timestamp("2024-07-01 01:00:00")
Q2_2024_START = pd.Timestamp("2024-04-01 01:00:00")
Q3_2024_START = H2_2024_START
Q4_2024_START = pd.Timestamp("2024-10-01 01:00:00")
YEAR_2025_END = pd.Timestamp("2026-01-01 00:00:00")
TEST_START = pd.Timestamp("2025-01-01 01:00:00")
SCADA_SPECS = {
    "kpx_group_1": {
        "manufacturer": "vestas",
        "turbines": tuple(range(1, 7)),
        "rated_kwh": 3600.0,
        "bounds": (-30.0, 630.0),
    },
    "kpx_group_2": {
        "manufacturer": "vestas",
        "turbines": tuple(range(7, 13)),
        "rated_kwh": 3600.0,
        "bounds": (-30.0, 630.0),
    },
    "kpx_group_3": {
        "manufacturer": "unison",
        "turbines": tuple(range(1, 6)),
        "rated_kwh": 4200.0,
        "bounds": (-35.0, 805.0),
    },
}
OBJECTIVES = ("l1", "q07")
BLEND_WEIGHTS = (0.05, 0.10, 0.20)
STAGE1_SLICES = {
    "kpx_group_1": {
        "full": (YEAR_2023_START, YEAR_2023_END),
        "H1": (YEAR_2023_START, H2_2023_START - pd.Timedelta(hours=1)),
        "H2": (H2_2023_START, YEAR_2023_END),
    },
    "kpx_group_2": {
        "full": (YEAR_2023_START, YEAR_2023_END),
        "H1": (YEAR_2023_START, H2_2023_START - pd.Timedelta(hours=1)),
        "H2": (H2_2023_START, YEAR_2023_END),
    },
    "kpx_group_3": {
        "full": (H2_2023_START, YEAR_2023_END),
        "Q3": (H2_2023_START, Q4_2023_START - pd.Timedelta(hours=1)),
        "Q4": (Q4_2023_START, YEAR_2023_END),
    },
}
STAGE2_SLICES = {
    "full": (YEAR_2024_START, YEAR_2024_END),
    "H1": (YEAR_2024_START, H2_2024_START - pd.Timedelta(hours=1)),
    "H2": (H2_2024_START, YEAR_2024_END),
    "Q1": (YEAR_2024_START, Q2_2024_START - pd.Timedelta(hours=1)),
    "Q2": (Q2_2024_START, Q3_2024_START - pd.Timedelta(hours=1)),
    "Q3": (Q3_2024_START, Q4_2024_START - pd.Timedelta(hours=1)),
    "Q4": (Q4_2024_START, YEAR_2024_END),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir", type=Path, default=Path(r"data/local/open")
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/cache"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--recipe", type=Path, default=Path("configs/train_final.v3.locked.json")
    )
    parser.add_argument(
        "--preregister",
        type=Path,
        default=Path("configs/turbine_scada_power_preregister.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/postgate/turbine_scada_power"),
    )
    parser.add_argument("--n-jobs", type=int, default=7)
    parser.add_argument(
        "--stage", choices=("stage1", "stage2", "final", "all"), default="all"
    )
    return parser.parse_args(argv)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    write_json_atomic(path, _json_ready(payload), overwrite=False)


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        frame.to_parquet(temporary, engine="pyarrow", compression="zstd", index=True)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_joblib(payload: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        joblib.dump(payload, temporary, compress=3)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
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


def _write_exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(_json_ready(payload), stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _verify_preregister(path: Path) -> dict[str, Any]:
    path = path.resolve()
    observed = sha256_file(path)
    if observed != PREREGISTER_SHA256:
        raise AssertionError(
            f"preregister changed: {observed} != {PREREGISTER_SHA256}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    grid = payload["candidate_grid"]
    if tuple(grid["objectives"]) != OBJECTIVES:
        raise AssertionError("objective grid changed")
    if tuple(map(float, grid["fixed_blend_weights_on_scada_model"])) != BLEND_WEIGHTS:
        raise AssertionError("blend grid changed")
    if int(grid["total_candidates_per_group"]) != 6:
        raise AssertionError("candidate count changed")
    return {"path": str(path), "sha256": observed, "payload": payload}


def _snapshot_named(paths: Mapping[str, Path]) -> dict[str, Any]:
    return {name: describe_file(path) for name, path in paths.items()}


def _provenance_paths(preregister: Path, recipe: Path) -> dict[str, Path]:
    return {
        "runner": Path(__file__).resolve(),
        "test": PROJECT_DIR / "tests" / "test_turbine_scada_power.py",
        "preregister": preregister.resolve(),
        "recipe": recipe.resolve(),
        "features": PROJECT_DIR / "src" / "features.py",
        "metric": PROJECT_DIR / "src" / "metric.py",
        "manifest": PROJECT_DIR / "src" / "manifest.py",
        "strict_raw_helper": PROJECT_DIR / "scripts" / "run_shared_q07_multiseed.py",
    }


def _discover_csv_prefix(path: Path, cutoff: pd.Timestamp) -> dict[str, Any]:
    """Discover a safe CSV prefix while exposing only the next timestamp cell."""

    digest = hashlib.sha256()
    rows = 0
    with path.open("rb", buffering=0) as stream:
        header = stream.readline()
        if not header:
            raise ValueError(f"empty CSV: {path}")
        digest.update(header)
        prefix_bytes = len(header)
        next_timestamp: pd.Timestamp | None = None
        while True:
            row_start = stream.tell()
            first = bytearray()
            while True:
                byte = stream.read(1)
                if not byte:
                    break
                if byte == b",":
                    break
                if byte in {b"\r", b"\n"}:
                    raise AssertionError(f"malformed CSV row in {path}")
                first.extend(byte)
            if not first and not byte:
                prefix_bytes = row_start
                break
            timestamp = pd.Timestamp(first.decode("utf-8-sig"))
            if timestamp >= cutoff:
                next_timestamp = timestamp
                prefix_bytes = row_start
                break
            rest = stream.readline()
            line = bytes(first) + b"," + rest
            digest.update(line)
            rows += 1
            prefix_bytes = stream.tell()
    observed, observed_bytes = _csv_prefix_identity(
        path, data_rows=rows, byte_limit=prefix_bytes
    )
    if observed != digest.hexdigest() or observed_bytes != prefix_bytes:
        raise AssertionError("SCADA prefix identity mismatch")
    return {
        "path": str(path.resolve()),
        "cutoff_exclusive": cutoff.isoformat(),
        "prefix_data_rows": rows,
        "prefix_bytes": prefix_bytes,
        "prefix_sha256": observed,
        "next_boundary_timestamp_only": (
            None if next_timestamp is None else next_timestamp.isoformat()
        ),
        "next_boundary_measurements_read": False,
        "suffix_bytes_exposed_to_parser": 0,
        "whole_file_hash_omitted_before_lock": True,
        "size_bytes": path.stat().st_size,
    }


def _refresh_prefix(record: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(record["path"]))
    observed, observed_bytes = _csv_prefix_identity(
        path,
        data_rows=int(record["prefix_data_rows"]),
        byte_limit=int(record["prefix_bytes"]),
    )
    if observed_bytes != int(record["prefix_bytes"]):
        raise AssertionError("SCADA prefix byte count changed")
    refreshed = dict(record)
    refreshed["prefix_sha256"] = observed
    refreshed["size_bytes"] = path.stat().st_size
    return refreshed


def _read_scada_prefix(record: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = Path(str(record["path"]))
    byte_limit = int(record["prefix_bytes"])
    bounded_raw = _BoundedRawReader(path, byte_limit=byte_limit)
    try:
        with io.BufferedReader(bounded_raw, buffer_size=1024 * 1024) as bounded:
            frame = pd.read_csv(
                bounded,
                encoding="utf-8-sig",
                nrows=int(record["prefix_data_rows"]),
                memory_map=False,
                low_memory=False,
            )
            returned = bounded_raw.bytes_returned
            position = bounded_raw.underlying_position
    finally:
        bounded_raw.close()
    if returned != byte_limit or position != byte_limit:
        raise AssertionError("SCADA parser crossed or under-read bounded prefix")
    if len(frame) != int(record["prefix_data_rows"]):
        raise AssertionError("SCADA bounded row count changed")
    frame["kst_dtm"] = pd.to_datetime(frame["kst_dtm"], errors="raise")
    cutoff = pd.Timestamp(str(record["cutoff_exclusive"]))
    if not (frame["kst_dtm"] < cutoff).all():
        raise AssertionError("SCADA validation row entered fit prefix")
    evidence = dict(record)
    evidence.update(
        {
            "physical_byte_limit": byte_limit,
            "physical_bytes_returned": returned,
            "underlying_file_position_after_parse": position,
            "materialized_rows": len(frame),
            "materialized_start": frame["kst_dtm"].min().isoformat(),
            "materialized_end": frame["kst_dtm"].max().isoformat(),
            "materialized_frame_sha256": _frame_sha256(frame),
            "memory_map": False,
        }
    )
    return frame, evidence


def _aggregate_turbine_hourly(
    frame: pd.DataFrame, group: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = SCADA_SPECS[group]
    manufacturer = str(spec["manufacturer"])
    turbines = tuple(spec["turbines"])
    lower, upper = map(float, spec["bounds"])
    target_times = frame["kst_dtm"].dt.floor("h") + pd.Timedelta(hours=1)
    output: dict[int, pd.Series] = {}
    diagnostics: dict[str, Any] = {}
    total_spikes = 0
    for turbine in turbines:
        column = f"{manufacturer}_wtg{turbine:02d}_power_kw10m"
        if column not in frame:
            raise ValueError(f"SCADA missing {column}")
        numeric = pd.to_numeric(frame[column], errors="coerce")
        finite = np.isfinite(numeric.to_numpy(dtype=float, copy=False))
        physical = numeric.between(lower, upper, inclusive="both")
        spikes = finite & ~physical.to_numpy(dtype=bool, copy=False)
        total_spikes += int(spikes.sum())
        cleaned = numeric.mask(spikes)
        grouped = pd.DataFrame({"time": target_times, "value": cleaned}).groupby(
            "time", sort=True
        )["value"]
        counts = grouped.count()
        energy = grouped.sum(min_count=6).where(counts == 6)
        output[int(turbine)] = energy
        diagnostics[str(turbine)] = {
            "valid_hours": int(energy.notna().sum()),
            "missing_hours": int(energy.isna().sum()),
            "spikes_removed": int(spikes.sum()),
        }
    hourly = pd.DataFrame(output)
    hourly.index = pd.DatetimeIndex(hourly.index, name="forecast_kst_dtm")
    return hourly, {
        "group": group,
        "alignment": "floor(source_kst_dtm,1h)+1h",
        "minimum_records": 6,
        "bounds": [lower, upper],
        "total_spikes_removed": total_spikes,
        "by_turbine": diagnostics,
        "rows": len(hourly),
        "start": hourly.index.min().isoformat(),
        "end": hourly.index.max().isoformat(),
        "frame_sha256": _frame_sha256(hourly),
    }


def _parse_dms(value: object) -> tuple[float, float]:
    tokens = re.findall(r"[\d.]+|[NSEW]", str(value).upper())
    if len(tokens) != 8:
        raise ValueError(f"cannot parse coordinate {value!r}")
    lat_d, lat_m, lat_s, ns, lon_d, lon_m, lon_s, ew = tokens
    lat = float(lat_d) + float(lat_m) / 60.0 + float(lat_s) / 3600.0
    lon = float(lon_d) + float(lon_m) / 60.0 + float(lon_s) / 3600.0
    return (-lat if ns == "S" else lat, -lon if ew == "W" else lon)


def _load_turbine_metadata(path: Path) -> dict[str, pd.DataFrame]:
    raw = pd.read_excel(path, sheet_name="info", header=3)
    required = {"제작사", "호기", "좌표(Google)", "KPX그룹", "설비용량(MW)"}
    if not required.issubset(raw.columns):
        raise ValueError(f"info.xlsx schema changed: {sorted(required - set(raw.columns))}")
    raw = raw.loc[raw["좌표(Google)"].notna()].copy()
    raw["KPX그룹"] = pd.to_numeric(raw["KPX그룹"], errors="coerce").ffill()
    coordinates = raw["좌표(Google)"].map(_parse_dms)
    raw["latitude"] = coordinates.map(lambda pair: pair[0])
    raw["longitude"] = coordinates.map(lambda pair: pair[1])
    result: dict[str, pd.DataFrame] = {}
    for number in (1, 2, 3):
        group = f"kpx_group_{number}"
        part = raw.loc[raw["KPX그룹"].eq(number)].copy()
        part["turbine"] = pd.to_numeric(part["호기"], errors="raise").astype(int)
        part["rated_kwh"] = (
            pd.to_numeric(part["설비용량(MW)"], errors="raise") * 1000.0
        )
        part = part.sort_values("turbine").reset_index(drop=True)
        expected = list(SCADA_SPECS[group]["turbines"])
        if part["turbine"].tolist() != expected:
            raise AssertionError(f"{group} turbine metadata changed")
        if abs(float(part["rated_kwh"].sum()) - CAPACITY_KWH[group]) > 1e-9:
            raise AssertionError(f"{group} metadata capacity changed")
        part["latitude_offset"] = part["latitude"] - part["latitude"].mean()
        part["longitude_offset"] = part["longitude"] - part["longitude"].mean()
        result[group] = part.loc[
            :, ["turbine", "latitude", "longitude", "latitude_offset", "longitude_offset", "rated_kwh"]
        ]
    return result


def _weather_columns(frame: pd.DataFrame, prereg: Mapping[str, Any]) -> list[str]:
    tokens = tuple(str(token) for token in prereg["features"]["wind_tokens"])
    selected = [
        str(column)
        for column in frame.columns
        if str(column).startswith("time__") or any(token in str(column) for token in tokens)
    ]
    if not selected:
        raise AssertionError("no preregistered weather columns selected")
    forbidden = [
        column
        for column in selected
        if "scada" in column.lower()
        or "target" in column.lower()
        or "actual" in column.lower()
        or column in TARGET_COLS
    ]
    if forbidden:
        raise AssertionError(f"forbidden weather features: {forbidden[:5]}")
    return selected


def _design_matrix(
    weather: pd.DataFrame,
    metadata: pd.DataFrame,
    selected: Sequence[str],
    group: str,
) -> pd.DataFrame:
    n_turbines = len(metadata)
    values = np.repeat(
        weather.loc[:, list(selected)].to_numpy(dtype=np.float32, copy=False),
        n_turbines,
        axis=0,
    )
    design = pd.DataFrame(values, columns=list(selected), dtype=np.float32)
    turbine = np.tile(metadata["turbine"].to_numpy(dtype=np.int16), len(weather))
    categories = list(metadata["turbine"].astype(int))
    design["turbine_id"] = pd.Categorical(turbine, categories=categories)
    fraction = (metadata["turbine"].rank(method="dense") - 1.0) / max(n_turbines - 1, 1)
    extras = {
        "turbine_index_fraction": fraction.to_numpy(dtype=np.float32),
        "turbine_latitude": metadata["latitude"].to_numpy(dtype=np.float32),
        "turbine_longitude": metadata["longitude"].to_numpy(dtype=np.float32),
        "turbine_latitude_offset": metadata["latitude_offset"].to_numpy(dtype=np.float32),
        "turbine_longitude_offset": metadata["longitude_offset"].to_numpy(dtype=np.float32),
        "turbine_capacity_fraction": (
            metadata["rated_kwh"].to_numpy(dtype=np.float32) / CAPACITY_KWH[group]
        ),
    }
    for name, row_values in extras.items():
        design[name] = np.tile(row_values, len(weather))
    return design


def _model_params(prereg: Mapping[str, Any], objective: str, n_jobs: int) -> dict[str, Any]:
    specification = dict(prereg["models"][objective])
    specification.pop("class")
    specification["n_jobs"] = int(n_jobs)
    specification["verbosity"] = -1
    return specification


def _fit_predict_group(
    *,
    group: str,
    objective: str,
    prereg: Mapping[str, Any],
    weather: pd.DataFrame,
    labels: pd.Series,
    hourly: pd.DataFrame,
    metadata: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    predict_index: pd.DatetimeIndex,
    n_jobs: int,
) -> tuple[LGBMRegressor, pd.Series, dict[str, Any]]:
    selected = _weather_columns(weather, prereg)
    train_index = weather.index[(weather.index >= train_start) & (weather.index <= train_end)]
    if not train_index.equals(labels.loc[train_index].index):
        raise AssertionError("training label index mismatch")
    if not predict_index.isin(weather.index).all():
        raise AssertionError("prediction index outside weather materialization")
    target = hourly.reindex(index=train_index, columns=metadata["turbine"].tolist())
    target_cf = target.to_numpy(dtype=float) / metadata["rated_kwh"].to_numpy(dtype=float)
    eligible_hour = (
        np.isfinite(labels.loc[train_index].to_numpy(dtype=float))
        & (labels.loc[train_index].to_numpy(dtype=float) >= 0.10 * CAPACITY_KWH[group])
    )
    mask = np.repeat(eligible_hour, len(metadata)) & np.isfinite(target_cf.reshape(-1))
    train_design = _design_matrix(weather.loc[train_index], metadata, selected, group)
    train_target = target_cf.reshape(-1)
    if int(mask.sum()) < 1000:
        raise AssertionError(f"{group} insufficient turbine-hour training rows")
    model = LGBMRegressor(**_model_params(prereg, objective, n_jobs))
    model.fit(
        train_design.loc[mask],
        train_target[mask],
        categorical_feature=["turbine_id"],
    )
    valid_design = _design_matrix(weather.loc[predict_index], metadata, selected, group)
    prediction_cf = np.asarray(model.predict(valid_design), dtype=float).reshape(
        len(predict_index), len(metadata)
    )
    prediction_cf = np.clip(prediction_cf, 0.0, 1.02)
    group_prediction = prediction_cf @ metadata["rated_kwh"].to_numpy(dtype=float)
    prediction = pd.Series(group_prediction, index=predict_index, name=group)
    if not np.isfinite(prediction.to_numpy()).all():
        raise AssertionError("non-finite turbine group prediction")
    fitted_target_sum = target.sum(axis=1, min_count=len(metadata))
    overlap = eligible_hour & np.isfinite(fitted_target_sum.to_numpy(dtype=float))
    corr = float(
        np.corrcoef(
            fitted_target_sum.to_numpy(dtype=float)[overlap],
            labels.loc[train_index].to_numpy(dtype=float)[overlap],
        )[0, 1]
    )
    audit = {
        "group": group,
        "objective": objective,
        "train_start": train_start,
        "train_end": train_end,
        "predict_start": predict_index.min(),
        "predict_end": predict_index.max(),
        "weather_features": len(selected),
        "model_features": train_design.shape[1],
        "eligible_hours": int(eligible_hour.sum()),
        "turbine_hour_rows": int(mask.sum()),
        "scada_group_sum_label_correlation_on_complete_eligible_fit_hours": corr,
        "validation_scada_used": False,
        "target_group_label_rescaling_used": False,
        "prediction_sha256": _frame_sha256(prediction.to_frame()),
    }
    return model, prediction, audit


def _read_prediction(path: Path, index: pd.DatetimeIndex, groups: Sequence[str]) -> pd.DataFrame:
    frame = pd.read_parquet(path, engine="pyarrow")
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if not frame.index.equals(index):
        raise AssertionError(f"{path} index changed")
    if not set(groups).issubset(frame.columns):
        raise AssertionError(f"{path} groups changed")
    output = frame.loc[:, list(groups)].astype(float)
    if not np.isfinite(output.to_numpy()).all():
        raise AssertionError(f"{path} contains non-finite predictions")
    return output


def _stage1_baseline(
    artifact_root: Path, recipe: Mapping[str, Any], index: pd.DatetimeIndex
) -> tuple[pd.DataFrame, dict[str, Any], list[Path]]:
    baseline = pd.DataFrame(index=index, columns=TARGET_COLS, dtype=float)
    input_paths: list[Path] = []
    dev_components: dict[str, pd.DataFrame] = {}
    for name, filename in DEV_COMPONENT_FILES.items():
        path = artifact_root / "oof" / filename
        input_paths.append(path)
        dev_components[name] = _read_prediction(path, index, ("kpx_group_1", "kpx_group_2"))
    for group in ("kpx_group_1", "kpx_group_2"):
        baseline[group] = _assemble_group(
            recipe, group, {name: dev_components[name][group] for name in COMPONENTS}
        )
    reference_path = artifact_root / "oof" / "dev2023_locked_v3.parquet"
    input_paths.append(reference_path)
    reference = _read_prediction(reference_path, index, ("kpx_group_1", "kpx_group_2"))
    reconstruction = float(
        np.max(
            np.abs(
                baseline.loc[:, ["kpx_group_1", "kpx_group_2"]].to_numpy()
                - reference.to_numpy()
            )
        )
    )
    if reconstruction != 0.0:
        raise AssertionError(f"g1/g2 baseline reconstruction changed: {reconstruction}")
    g3_path = artifact_root / "oof" / "g3dev2023h2_candidates.parquet"
    input_paths.append(g3_path)
    g3_raw = pd.read_parquet(g3_path, engine="pyarrow")
    g3_raw.index = pd.DatetimeIndex(g3_raw.index, name="forecast_kst_dtm")
    g3_index = index[(index >= H2_2023_START) & (index <= YEAR_2023_END)]
    if not g3_raw.index.equals(g3_index):
        raise AssertionError("g3 historical baseline index changed")
    required = {"q07", "shared_l1", "shared_q07", "top200q07", "ewq06"}
    if not required.issubset(g3_raw.columns):
        raise AssertionError("g3 historical components changed")
    parts = {
        "lgb_l1": g3_raw["q07"] * 0.0,
        "lgb_q07": g3_raw["q07"],
        "shared_l1": g3_raw["shared_l1"],
        "shared_q07": g3_raw["shared_q07"],
        "top200_q07": g3_raw["top200q07"],
        "energy_q06": g3_raw["ewq06"],
    }
    baseline.loc[g3_index, "kpx_group_3"] = _assemble_group(recipe, "kpx_group_3", parts)
    return baseline, {"g12_max_abs_reconstruction_difference": reconstruction}, input_paths


def _group_score(actual: pd.Series, prediction: pd.Series, group: str) -> dict[str, Any]:
    result = group_metrics(actual, prediction, CAPACITY_KWH[group], group_name=group)
    score = 0.5 * result.one_minus_nmae + 0.5 * result.ficr
    return {
        "score": float(score),
        "one_minus_nmae": float(result.one_minus_nmae),
        "ficr": float(result.ficr),
        "n_evaluated": int(result.n_evaluated),
    }


def _slice_comparison(
    labels: pd.Series,
    baseline: pd.Series,
    candidate: pd.Series,
    group: str,
    slices: Mapping[str, tuple[pd.Timestamp, pd.Timestamp]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, (start, end) in slices.items():
        index = baseline.index[(baseline.index >= start) & (baseline.index <= end)]
        before = _group_score(labels.loc[index], baseline.loc[index], group)
        after = _group_score(labels.loc[index], candidate.loc[index], group)
        output[name] = {
            "baseline": before,
            "candidate": after,
            "delta": float(after["score"] - before["score"]),
        }
    return output


def _candidate_key(objective: str, weight: float) -> str:
    return f"{objective}__w{int(round(weight * 100)):02d}"


def _passing_and_selection(comparisons: Mapping[str, Any]) -> tuple[list[str], str | None]:
    passing = [
        name
        for name, slices in comparisons.items()
        if all(float(values["delta"]) > 0.0 for values in slices.values())
    ]
    if not passing:
        return [], None

    def rank(name: str) -> tuple[float, float, float, int]:
        slices = comparisons[name]
        objective, weight_token = name.split("__w")
        weight = int(weight_token) / 100.0
        deltas = [float(values["delta"]) for values in slices.values()]
        return (
            min(deltas),
            float(slices["full"]["delta"]),
            -weight,
            1 if objective == "l1" else 0,
        )

    return sorted(passing), max(passing, key=rank)


def _blend(baseline: pd.Series, scada: pd.Series, group: str, weight: float) -> pd.Series:
    if not baseline.index.equals(scada.index):
        raise AssertionError("blend index mismatch")
    values = np.clip(
        (1.0 - weight) * baseline.to_numpy(dtype=float)
        + weight * scada.to_numpy(dtype=float),
        0.0,
        1.02 * CAPACITY_KWH[group],
    )
    return pd.Series(values, index=baseline.index, name=group)


def _stage1_historical_paths(artifact_root: Path) -> dict[str, Path]:
    paths = {
        f"dev_{name}": artifact_root / "oof" / filename
        for name, filename in DEV_COMPONENT_FILES.items()
    }
    paths.update(
        {
            "dev_reference": artifact_root / "oof" / "dev2023_locked_v3.parquet",
            "g3_historical": artifact_root / "oof" / "g3dev2023h2_candidates.parquet",
        }
    )
    return paths


def _output_manifest(
    *,
    out_dir: Path,
    stage: str,
    preregister: Mapping[str, Any],
    provenance: Mapping[str, Any],
    safe_inputs: Mapping[str, Any],
    output_files: Sequence[Path],
    results: Mapping[str, Any],
) -> None:
    payload = {
        "schema_version": 1,
        "artifact_type": "turbine_scada_power_strict_forward",
        "created_utc": utc_now(),
        "stage": stage,
        "runtime": {"packages": package_versions()},
        "git": git_state(PROJECT_DIR),
        "preregister": {
            "path": preregister["path"],
            "sha256": preregister["sha256"],
        },
        "provenance": provenance,
        "safe_inputs": safe_inputs,
        "outputs": [describe_file(path) for path in output_files if path.exists()],
        "results_digest": _canonical_sha256(results),
        "submission_created": any(path.suffix.lower() == ".csv" for path in output_files),
    }
    name = "manifest.json" if stage in {"rejected_stage1", "rejected_stage2", "final"} else f"{stage}_manifest.json"
    _atomic_json(out_dir / name, payload)


def _stage1(args: argparse.Namespace, preregister: Mapping[str, Any]) -> dict[str, Any]:
    out_dir = args.out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"stage1 requires empty output directory: {out_dir}")
    raw_dir = args.raw_dir.resolve()
    artifact_root = args.artifact_root.resolve()
    recipe_path = args.recipe.resolve()
    provenance_paths = _provenance_paths(Path(preregister["path"]), recipe_path)
    provenance_before = _snapshot_named(provenance_paths)
    historical_paths = _stage1_historical_paths(artifact_root)
    historical_before = _snapshot_named(historical_paths)
    scada_prefixes = {
        "vestas": _discover_csv_prefix(
            raw_dir / "train" / "scada_vestas_train.csv",
            pd.Timestamp("2023-01-01 00:00:00"),
        ),
        "unison": _discover_csv_prefix(
            raw_dir / "train" / "scada_unison_train.csv",
            pd.Timestamp("2023-07-01 00:00:00"),
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    prereg_copy = out_dir / "preregister.json"
    shutil.copyfile(preregister["path"], prereg_copy)
    if sha256_file(prereg_copy) != PREREGISTER_SHA256:
        raise AssertionError("preregister copy changed")
    labels = _read_labels(
        raw_dir / "train" / "train_labels.csv",
        nrows=EXPECTED_ROWS_PRE2024,
        expected_end=YEAR_2023_END,
        prefix_bytes=OFFICIAL_STAGE1_PREFIX_BYTES["labels"],
        expected_prefix_sha256=OFFICIAL_LABEL_PREFIX_SHA256,
    )
    features, raw_nwp_contract = _read_stage1_raw_features(raw_dir, labels)
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    stage1_valid_index = labels.index[
        (labels.index >= YEAR_2023_START) & (labels.index <= YEAR_2023_END)
    ]
    baseline_valid, baseline_audit, baseline_inputs = _stage1_baseline(
        artifact_root, recipe, stage1_valid_index
    )
    baseline = baseline_valid.reindex(labels.index)
    metadata = _load_turbine_metadata(raw_dir / "info.xlsx")
    scada_frames: dict[str, pd.DataFrame] = {}
    scada_io: dict[str, Any] = {}
    for manufacturer, prefix in scada_prefixes.items():
        scada_frames[manufacturer], scada_io[manufacturer] = _read_scada_prefix(prefix)
    hourly: dict[str, pd.DataFrame] = {}
    hourly_audit: dict[str, Any] = {}
    for group in TARGET_COLS:
        manufacturer = str(SCADA_SPECS[group]["manufacturer"])
        hourly[group], hourly_audit[group] = _aggregate_turbine_hourly(
            scada_frames[manufacturer], group
        )
    models: dict[str, dict[str, LGBMRegressor]] = {group: {} for group in TARGET_COLS}
    raw_predictions: dict[str, dict[str, pd.Series]] = {group: {} for group in TARGET_COLS}
    training: dict[str, Any] = {}
    candidate_wide = pd.DataFrame(index=labels.index)
    comparisons: dict[str, dict[str, Any]] = {}
    selected: dict[str, Any] = {}
    outputs: list[Path] = [prereg_copy]
    for group in TARGET_COLS:
        if group == "kpx_group_3":
            train_start, train_end = YEAR_2023_START, H2_2023_START - pd.Timedelta(hours=1)
            valid_index = labels.index[(labels.index >= H2_2023_START) & (labels.index <= YEAR_2023_END)]
        else:
            train_start, train_end = YEAR_2022_START, YEAR_2022_END
            valid_index = labels.index[(labels.index >= YEAR_2023_START) & (labels.index <= YEAR_2023_END)]
        group_comparisons: dict[str, Any] = {}
        for objective in OBJECTIVES:
            print(f"stage1 fit {group} {objective}", flush=True)
            model, prediction, audit = _fit_predict_group(
                group=group,
                objective=objective,
                prereg=preregister["payload"],
                weather=features[group],
                labels=labels[group],
                hourly=hourly[group],
                metadata=metadata[group],
                train_start=train_start,
                train_end=train_end,
                predict_index=valid_index,
                n_jobs=args.n_jobs,
            )
            models[group][objective] = model
            raw_predictions[group][objective] = prediction
            model_path = out_dir / "models" / f"stage1__{group}__{objective}.joblib"
            _atomic_joblib(
                {"model": model, "audit": audit, "metadata": metadata[group]}, model_path
            )
            outputs.append(model_path)
            training[f"{group}__{objective}"] = {
                **audit,
                "model_path": str(model_path.resolve()),
                "model_sha256": sha256_file(model_path),
            }
            candidate_wide.loc[valid_index, f"{group}__{objective}__raw"] = prediction
            for weight in BLEND_WEIGHTS:
                name = _candidate_key(objective, weight)
                candidate = _blend(
                    baseline.loc[valid_index, group], prediction, group, weight
                )
                candidate_wide.loc[valid_index, f"{group}__{name}"] = candidate
                group_comparisons[name] = _slice_comparison(
                    labels[group],
                    baseline.loc[valid_index, group],
                    candidate,
                    group,
                    STAGE1_SLICES[group],
                )
        passing, choice = _passing_and_selection(group_comparisons)
        comparisons[group] = group_comparisons
        selected[group] = {"passing_candidates": passing, "selected": choice}
    baseline_path = out_dir / "oof" / "stage1_corrected_v3_baseline.parquet"
    candidate_path = out_dir / "oof" / "stage1_turbine_candidates.parquet"
    _atomic_parquet(baseline, baseline_path)
    _atomic_parquet(candidate_wide, candidate_path)
    outputs.extend([baseline_path, candidate_path])
    locked_groups = [group for group in TARGET_COLS if selected[group]["selected"] is not None]
    provenance_after = _snapshot_named(provenance_paths)
    historical_after = _snapshot_named(historical_paths)
    scada_after = {name: _refresh_prefix(record) for name, record in scada_prefixes.items()}
    if provenance_before != provenance_after:
        raise AssertionError("source/config provenance changed during stage1")
    if historical_before != historical_after:
        raise AssertionError("historical baseline inputs changed during stage1")
    for name in scada_prefixes:
        if scada_prefixes[name] != scada_after[name]:
            raise AssertionError(f"{name} bounded prefix changed during stage1")
    results = {
        "schema_version": 1,
        "stage": "stage1_pre2024_selection",
        "preregister_sha256": PREREGISTER_SHA256,
        "forbidden_2024_data_read_occurred": False,
        "stage1_cache_files_read": False,
        "raw_nwp_contract": raw_nwp_contract,
        "raw_scada_contract": scada_io,
        "turbine_hourly_audit": hourly_audit,
        "baseline_audit": baseline_audit,
        "training": training,
        "comparisons": comparisons,
        "comparisons_sha256": _canonical_sha256(comparisons),
        "selection": selected,
        "locked_groups": locked_groups,
        "lock_rule": "all required stage1 slice deltas strictly > 0",
        "candidate_count_per_group": 6,
        "provenance": provenance_before,
        "provenance_sha256": _canonical_sha256(provenance_before),
        "historical_inputs": historical_before,
        "baseline_input_paths": [str(path.resolve()) for path in baseline_inputs],
        "output_prediction_sha256": sha256_file(candidate_path),
    }
    results_path = out_dir / "stage1_results.json"
    _atomic_json(results_path, results)
    outputs.append(results_path)
    lock_payload = {
        "schema_version": 1,
        "stage": "stage1_pre2024_lock",
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "runner_sha256": provenance_before["runner"]["sha256"],
        "test_sha256": provenance_before["test"]["sha256"],
        "stage1_results_sha256": sha256_file(results_path),
        "comparisons_sha256": results["comparisons_sha256"],
        "locked_groups": locked_groups,
        "selected": {group: selected[group]["selected"] for group in locked_groups},
        "forbidden_2024_data_read_occurred": False,
    }
    lock_path = out_dir / "stage1_lock.json"
    _write_exclusive_json(lock_path, lock_payload)
    outputs.append(lock_path)
    _output_manifest(
        out_dir=out_dir,
        stage="rejected_stage1" if not locked_groups else "stage1",
        preregister=preregister,
        provenance=provenance_before,
        safe_inputs={
            "raw_scada_prefixes": scada_prefixes,
            "historical": historical_before,
            "raw_nwp": raw_nwp_contract,
        },
        output_files=outputs,
        results=results,
    )
    print(json.dumps({"locked_groups": locked_groups, "selected": lock_payload["selected"]}), flush=True)
    return results


def _verify_stage1_lock(args: argparse.Namespace, preregister: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    out_dir = args.out_dir.resolve()
    results_path = out_dir / "stage1_results.json"
    lock_path = out_dir / "stage1_lock.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if sha256_file(results_path) != lock["stage1_results_sha256"]:
        raise AssertionError("stage1 results hash differs from immutable lock")
    if lock["preregister_sha256"] != preregister["sha256"]:
        raise AssertionError("stage1 lock preregister hash changed")
    current_runner = sha256_file(Path(__file__).resolve())
    current_test = sha256_file(PROJECT_DIR / "tests" / "test_turbine_scada_power.py")
    if lock["runner_sha256"] != current_runner or lock["test_sha256"] != current_test:
        raise AssertionError("runner/test changed after stage1 selection")
    recomputed: dict[str, str] = {}
    for group, comparisons in results["comparisons"].items():
        _, choice = _passing_and_selection(comparisons)
        if choice is not None:
            recomputed[group] = choice
    if sorted(recomputed) != sorted(lock["locked_groups"]):
        raise AssertionError("locked_groups do not equal recomputed promotion rule")
    if recomputed != lock["selected"]:
        raise AssertionError("locked selected candidates do not equal recomputation")
    if results["comparisons_sha256"] != lock["comparisons_sha256"]:
        raise AssertionError("comparison digest changed")
    return results, lock


def _read_full_labels(path: Path) -> pd.DataFrame:
    labels = pd.read_csv(path, encoding="utf-8-sig")
    if tuple(labels.columns) != ("kst_dtm", *TARGET_COLS):
        raise AssertionError("full label schema changed")
    labels.index = pd.DatetimeIndex(
        pd.to_datetime(labels.pop("kst_dtm"), errors="raise"), name="forecast_kst_dtm"
    )
    if labels.index.min() != YEAR_2022_START or labels.index.max() != YEAR_2024_END:
        raise AssertionError("full label period changed")
    return labels.astype(float)


def _read_cached_features(
    cache_dir: Path, groups: Sequence[str], expected_index: pd.DatetimeIndex
) -> dict[str, pd.DataFrame]:
    output: dict[str, pd.DataFrame] = {}
    for group in groups:
        frame = pd.read_parquet(cache_dir / f"{group}_weather_train.parquet", engine="pyarrow")
        frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
        if not frame.index.equals(expected_index):
            raise AssertionError(f"{group} full train feature index changed")
        output[group] = frame.astype(np.float32, copy=False)
    return output


def _stage2(args: argparse.Namespace, preregister: Mapping[str, Any]) -> dict[str, Any]:
    stage1, lock = _verify_stage1_lock(args, preregister)
    out_dir = args.out_dir.resolve()
    locked_groups = list(lock["locked_groups"])
    if not locked_groups:
        results = {
            "schema_version": 1,
            "stage": "stage2_not_run",
            "reason": "no group passed all preregistered stage1 slices",
            "locked_groups": [],
            "forbidden_2024_data_read_occurred": False,
            "promoted_groups": [],
            "submission_created": False,
        }
        path = out_dir / "stage2_results.json"
        _atomic_json(path, results)
        promotion_path = out_dir / "stage2_promotion_lock.json"
        _write_exclusive_json(
            promotion_path,
            {
                "schema_version": 1,
                "stage2_results_sha256": sha256_file(path),
                "stage1_lock_sha256": sha256_file(out_dir / "stage1_lock.json"),
                "promoted_groups": [],
                "selected": {},
            },
        )
        return results
    raw_dir = args.raw_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    labels = _read_full_labels(raw_dir / "train" / "train_labels.csv")
    features = _read_cached_features(cache_dir, locked_groups, labels.index)
    metadata = _load_turbine_metadata(raw_dir / "info.xlsx")
    scada_by_manufacturer: dict[str, pd.DataFrame] = {}
    scada_contract: dict[str, Any] = {}
    for group in locked_groups:
        manufacturer = str(SCADA_SPECS[group]["manufacturer"])
        if manufacturer not in scada_by_manufacturer:
            prefix = _discover_csv_prefix(
                raw_dir / "train" / f"scada_{manufacturer}_train.csv",
                pd.Timestamp("2024-01-01 00:00:00"),
            )
            scada_by_manufacturer[manufacturer], scada_contract[manufacturer] = _read_scada_prefix(prefix)
    baseline = _read_prediction(
        args.artifact_root.resolve() / "oof" / "gate2024_locked_v3_cf_fix.parquet",
        labels.loc[YEAR_2024_START:YEAR_2024_END].index,
        TARGET_COLS,
    )
    comparisons: dict[str, Any] = {}
    model_audit: dict[str, Any] = {}
    outputs: list[Path] = []
    candidate = baseline.copy()
    for group in locked_groups:
        selected = str(lock["selected"][group])
        objective, weight_token = selected.split("__w")
        weight = int(weight_token) / 100.0
        manufacturer = str(SCADA_SPECS[group]["manufacturer"])
        hourly, hourly_audit = _aggregate_turbine_hourly(scada_by_manufacturer[manufacturer], group)
        train_start = YEAR_2023_START if group == "kpx_group_3" else YEAR_2022_START
        train_end = YEAR_2023_END
        model, prediction, audit = _fit_predict_group(
            group=group,
            objective=objective,
            prereg=preregister["payload"],
            weather=features[group],
            labels=labels[group],
            hourly=hourly,
            metadata=metadata[group],
            train_start=train_start,
            train_end=train_end,
            predict_index=baseline.index,
            n_jobs=args.n_jobs,
        )
        candidate[group] = _blend(baseline[group], prediction, group, weight)
        comparisons[group] = _slice_comparison(
            labels[group], baseline[group], candidate[group], group, STAGE2_SLICES
        )
        model_path = out_dir / "models" / f"stage2__{group}__{objective}.joblib"
        _atomic_joblib(
            {"model": model, "audit": audit, "hourly_audit": hourly_audit}, model_path
        )
        outputs.append(model_path)
        model_audit[group] = {
            "selected": selected,
            "fit": audit,
            "hourly": hourly_audit,
            "model_sha256": sha256_file(model_path),
        }
    candidate_path = out_dir / "oof" / "stage2_locked_candidates.parquet"
    _atomic_parquet(candidate, candidate_path)
    outputs.append(candidate_path)
    promoted_groups = [
        group
        for group in locked_groups
        if all(float(values["delta"]) > 0.0 for values in comparisons[group].values())
    ]
    results = {
        "schema_version": 1,
        "stage": "stage2_2024_fixed_transfer",
        "stage1_lock_sha256": sha256_file(out_dir / "stage1_lock.json"),
        "locked_groups": locked_groups,
        "selected": lock["selected"],
        "comparisons": comparisons,
        "comparisons_sha256": _canonical_sha256(comparisons),
        "promoted_groups": promoted_groups,
        "promotion_rule": "exact locked candidate delta strictly > 0 in full/H1/H2/Q1/Q2/Q3/Q4",
        "models": model_audit,
        "scada_io_contract": scada_contract,
        "no_reselection": True,
        "submission_created": False,
    }
    results_path = out_dir / "stage2_results.json"
    _atomic_json(results_path, results)
    outputs.append(results_path)
    promotion_path = out_dir / "stage2_promotion_lock.json"
    _write_exclusive_json(
        promotion_path,
        {
            "schema_version": 1,
            "stage": "stage2_promotion_lock",
            "created_utc": utc_now(),
            "stage1_lock_sha256": sha256_file(out_dir / "stage1_lock.json"),
            "stage2_results_sha256": sha256_file(results_path),
            "comparisons_sha256": results["comparisons_sha256"],
            "promoted_groups": promoted_groups,
            "selected": {group: lock["selected"][group] for group in promoted_groups},
        },
    )
    outputs.append(promotion_path)
    if not promoted_groups:
        _output_manifest(
            out_dir=out_dir,
            stage="rejected_stage2",
            preregister=preregister,
            provenance=stage1["provenance"],
            safe_inputs={"stage1_lock": describe_file(out_dir / "stage1_lock.json")},
            output_files=outputs,
            results=results,
        )
    print(json.dumps({"promoted_groups": promoted_groups}), flush=True)
    return results


def _verify_stage2_lock(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    out_dir = args.out_dir.resolve()
    results_path = out_dir / "stage2_results.json"
    lock_path = out_dir / "stage2_promotion_lock.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if sha256_file(results_path) != lock["stage2_results_sha256"]:
        raise AssertionError("stage2 results differ from lock")
    recomputed = []
    for group, comparisons in results.get("comparisons", {}).items():
        if all(float(values["delta"]) > 0.0 for values in comparisons.values()):
            recomputed.append(group)
    if sorted(recomputed) != sorted(lock["promoted_groups"]):
        raise AssertionError("stage2 promoted groups differ from recomputation")
    return results, lock


def _stage_final(args: argparse.Namespace, preregister: Mapping[str, Any]) -> dict[str, Any]:
    stage1, stage1_lock = _verify_stage1_lock(args, preregister)
    stage2, promotion = _verify_stage2_lock(args)
    promoted_groups = list(promotion["promoted_groups"])
    if not promoted_groups:
        return {"stage": "final_not_run", "reason": "no stage2 promotions", "submission_created": False}
    raw_dir = args.raw_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    artifact_root = args.artifact_root.resolve()
    out_dir = args.out_dir.resolve()
    labels = _read_full_labels(raw_dir / "train" / "train_labels.csv")
    train_features = _read_cached_features(cache_dir, promoted_groups, labels.index)
    sample = pd.read_csv(raw_dir / "sample_submission.csv", encoding="utf-8-sig")
    if tuple(sample.columns) != ("ID", "forecast_kst_dtm", *TARGET_COLS):
        raise AssertionError("sample submission schema changed")
    test_index = pd.DatetimeIndex(
        pd.to_datetime(sample["forecast_kst_dtm"], errors="raise"), name="forecast_kst_dtm"
    )
    if test_index.min() != TEST_START or test_index.max() != YEAR_2025_END or len(test_index) != 8760:
        raise AssertionError("test index changed")
    test_features: dict[str, pd.DataFrame] = {}
    for group in promoted_groups:
        frame = pd.read_parquet(cache_dir / f"{group}_weather_test.parquet", engine="pyarrow")
        frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
        if not frame.index.equals(test_index):
            raise AssertionError(f"{group} test weather index changed")
        test_features[group] = frame.astype(np.float32, copy=False)
    metadata = _load_turbine_metadata(raw_dir / "info.xlsx")
    scada_frames: dict[str, pd.DataFrame] = {}
    scada_contract: dict[str, Any] = {}
    for group in promoted_groups:
        manufacturer = str(SCADA_SPECS[group]["manufacturer"])
        if manufacturer not in scada_frames:
            prefix = _discover_csv_prefix(
                raw_dir / "train" / f"scada_{manufacturer}_train.csv",
                pd.Timestamp("2025-01-01 00:00:00"),
            )
            scada_frames[manufacturer], scada_contract[manufacturer] = _read_scada_prefix(prefix)
    baseline_path = artifact_root / "final_cf_fix" / "predictions" / "corrected_v3_test.parquet"
    baseline = _read_prediction(baseline_path, test_index, TARGET_COLS)
    candidate = baseline.copy()
    models: dict[str, Any] = {}
    outputs: list[Path] = []
    for group in promoted_groups:
        selected = str(promotion["selected"][group])
        objective, weight_token = selected.split("__w")
        weight = int(weight_token) / 100.0
        manufacturer = str(SCADA_SPECS[group]["manufacturer"])
        hourly, hourly_audit = _aggregate_turbine_hourly(scada_frames[manufacturer], group)
        all_weather = pd.concat([train_features[group], test_features[group]], axis=0)
        train_start = YEAR_2023_START if group == "kpx_group_3" else YEAR_2022_START
        model, prediction, audit = _fit_predict_group(
            group=group,
            objective=objective,
            prereg=preregister["payload"],
            weather=all_weather,
            labels=labels[group].reindex(all_weather.index),
            hourly=hourly,
            metadata=metadata[group],
            train_start=train_start,
            train_end=YEAR_2024_END,
            predict_index=test_index,
            n_jobs=args.n_jobs,
        )
        candidate[group] = _blend(baseline[group], prediction, group, weight)
        model_path = out_dir / "models" / f"final__{group}__{objective}.joblib"
        _atomic_joblib(
            {"model": model, "audit": audit, "hourly_audit": hourly_audit}, model_path
        )
        outputs.append(model_path)
        models[group] = {
            "selected": selected,
            "audit": audit,
            "hourly": hourly_audit,
            "model_sha256": sha256_file(model_path),
        }
    prediction_path = out_dir / "predictions" / "turbine_scada_power_2025.parquet"
    _atomic_parquet(candidate, prediction_path)
    outputs.append(prediction_path)
    submission = sample.copy()
    for group in TARGET_COLS:
        submission[group] = candidate[group].to_numpy(dtype=float)
    submission_path = out_dir / "turbine_scada_power_2025.csv"
    _atomic_csv(submission, submission_path)
    outputs.append(submission_path)
    results = {
        "schema_version": 1,
        "stage": "final_2025",
        "promoted_groups": promoted_groups,
        "selected": promotion["selected"],
        "models": models,
        "scada_io_contract": scada_contract,
        "failed_groups_bit_identical_to_corrected_v3": [
            group for group in TARGET_COLS if group not in promoted_groups
        ],
        "prediction_sha256": sha256_file(prediction_path),
        "submission_sha256": sha256_file(submission_path),
        "submission_rows": len(submission),
        "submission_created": True,
        "leaderboard_score_claim": False,
    }
    results_path = out_dir / "results.json"
    _atomic_json(results_path, results)
    outputs.append(results_path)
    _output_manifest(
        out_dir=out_dir,
        stage="final",
        preregister=preregister,
        provenance=stage1["provenance"],
        safe_inputs={
            "stage1_lock": describe_file(out_dir / "stage1_lock.json"),
            "stage2_lock": describe_file(out_dir / "stage2_promotion_lock.json"),
            "corrected_v3_baseline": describe_file(baseline_path),
        },
        output_files=outputs,
        results=results,
    )
    print(json.dumps({"submission": str(submission_path), "sha256": results["submission_sha256"]}), flush=True)
    return results


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.raw_dir = args.raw_dir.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.artifact_root = args.artifact_root.expanduser().resolve()
    args.recipe = args.recipe.expanduser().resolve()
    args.preregister = args.preregister.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    preregister = _verify_preregister(args.preregister)
    if args.stage in {"stage1", "all"}:
        _stage1(args, preregister)
    if args.stage in {"stage2", "all"}:
        _stage2(args, preregister)
    if args.stage in {"final", "all"}:
        _stage_final(args, preregister)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Stream and decode NOAA NODD GFS HPBL for the 2024 causal gate.

The competition operating day ``D`` is mapped to the GFS run initialized at
``D-2 12Z`` and forecast hours 028..051.  Each HTTP byte range is decoded in
memory and discarded; durable outputs contain only compact site/group values
and enough response/index evidence to reproduce and verify every range.

This program never opens competition labels or prediction files.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import math
import os
import ssl
import sys
import threading
import time
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable

import eccodes
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "artifacts" / "final_submission_sprint_20260812" / "hpbl_2024"
COORDINATE_LOCK = (
    REPO
    / "artifacts"
    / "baram2026_ncei_scada_longrun_20260810_v2"
    / "prereg"
    / "authoritative_turbine_coordinate_lock_v1.json"
)
HOST = "noaa-gfs-bdp-pds.s3.amazonaws.com"
USER_AGENT = "baram2026-noaa-hpbl-2024-sprint/1.0"
GROUPS = ("kpx_group_1", "kpx_group_2", "kpx_group_3")
EXPECTED_HOURS = 366 * 24
HPBL_MATCH = ":HPBL:surface:"
CORNER_COORDINATES = (
    (37.50, 128.75),
    (37.50, 129.00),
    (37.25, 128.75),
    (37.25, 129.00),
)
TLS_CONTEXT = ssl.create_default_context()
THREAD_LOCAL = threading.local()
# The Windows ecCodes wheel reads its MEMFS definition catalogue through a
# process-global parser.  Concurrent first/use calls can corrupt that parser,
# so HTTP stays parallel while GRIB decode is deliberately serialized.
ECCODES_LOCK = threading.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(REPO.resolve()).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def write_parquet_exclusive(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    if path.exists() or partial.exists():
        raise FileExistsError(path)
    pq.write_table(
        pa.Table.from_pandas(frame, preserve_index=False),
        partial,
        version="2.6",
        compression="zstd",
        use_dictionary=False,
        write_statistics=True,
        row_group_size=len(frame),
    )
    os.replace(partial, path)


def connection() -> http.client.HTTPSConnection:
    active = getattr(THREAD_LOCAL, "connection", None)
    if active is None:
        active = http.client.HTTPSConnection(HOST, timeout=90, context=TLS_CONTEXT)
        THREAD_LOCAL.connection = active
    return active


def reset_connection() -> None:
    active = getattr(THREAD_LOCAL, "connection", None)
    if active is not None:
        try:
            active.close()
        except Exception:
            pass
    THREAD_LOCAL.connection = None


def get_with_retry(
    path: str,
    *,
    headers: dict[str, str],
    validator: Callable[[int, dict[str, str], bytes], None],
    attempts: int = 4,
) -> tuple[bytes, dict[str, str], int, float]:
    last_error: Exception | None = None
    started = time.perf_counter()
    for attempt in range(1, attempts + 1):
        try:
            client = connection()
            client.request("GET", path, headers=headers)
            response = client.getresponse()
            response_headers = {
                str(key).lower(): str(value).strip()
                for key, value in response.getheaders()
            }
            payload = response.read()
            validator(int(response.status), response_headers, payload)
            return payload, response_headers, attempt, time.perf_counter() - started
        except Exception as exc:
            last_error = exc
            reset_connection()
            if attempt < attempts:
                time.sleep(float(2 ** (attempt - 1)))
    raise RuntimeError(f"GET failed after {attempts} attempts: {path}: {last_error}") from last_error


def object_paths(run_day: date, forecast_hour: int) -> tuple[str, str]:
    stem = (
        f"/gfs.{run_day:%Y%m%d}/12/atmos/"
        f"gfs.t12z.pgrb2.0p25.f{forecast_hour:03d}"
    )
    return stem, stem + ".idx"


def parse_hpbl_range(index_payload: bytes) -> tuple[int, int, str, str]:
    text = index_payload.decode("utf-8", errors="strict")
    lines = text.splitlines()
    positions = [i for i, line in enumerate(lines) if HPBL_MATCH in line]
    if len(positions) != 1:
        raise RuntimeError(f"expected one exact HPBL surface index row, got {len(positions)}")
    position = positions[0]
    if position + 1 >= len(lines):
        raise RuntimeError("HPBL is the final index row; exact range end is unavailable")
    line = lines[position]
    next_line = lines[position + 1]
    fields = line.split(":")
    next_fields = next_line.split(":")
    if len(fields) < 6 or fields[3] != "HPBL" or fields[4] != "surface":
        raise RuntimeError(f"HPBL index semantics changed: {line}")
    start = int(fields[1])
    next_start = int(next_fields[1])
    if start < 0 or next_start <= start:
        raise RuntimeError("invalid HPBL byte offsets")
    return start, next_start - 1, line, next_line


def spatial_contract() -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, dict[str, Any]]:
    lock = json.loads(COORDINATE_LOCK.read_text(encoding="utf-8"))
    sites = sorted(lock["sites"], key=lambda row: int(row["site_id"]))
    if [int(site["site_id"]) for site in sites] != list(range(1, 18)):
        raise RuntimeError("authoritative site IDs changed")
    site_weights = np.zeros((17, 4), dtype=np.float64)
    site_audit: list[dict[str, Any]] = []
    for i, site in enumerate(sites):
        latitude = float(site["latitude"])
        longitude = float(site["longitude"])
        if not (37.25 <= latitude <= 37.50 and 128.75 <= longitude <= 129.00):
            raise RuntimeError(f"site outside common bilinear cell: {site['site_id']}")
        wx = (longitude - 128.75) / 0.25
        wy = (37.50 - latitude) / 0.25
        weights = np.asarray(
            ((1 - wx) * (1 - wy), wx * (1 - wy), (1 - wx) * wy, wx * wy),
            dtype=np.float64,
        )
        if (
            not np.isfinite(weights).all()
            or np.min(weights) < 0
            or np.max(weights) > 1
            or not math.isclose(float(weights.sum()), 1.0, abs_tol=1e-12)
        ):
            raise RuntimeError(f"invalid site weights: {site['site_id']}")
        site_weights[i] = weights
        site_audit.append(
            {
                "site_id": int(site["site_id"]),
                "group": str(site["group"]),
                "latitude": latitude,
                "longitude": longitude,
                "capacity_mw": float(site["capacity_mw"]),
                "corner_weights": [float(value) for value in weights],
            }
        )
    group_weights = np.zeros((3, 17), dtype=np.float64)
    group_audit: list[dict[str, Any]] = []
    for group_position, group in enumerate(GROUPS):
        members = [i for i, site in enumerate(sites) if site["group"] == group]
        capacities = np.asarray([float(sites[i]["capacity_mw"]) for i in members])
        total = float(capacities.sum())
        expected_count = (6, 6, 5)[group_position]
        expected_capacity = (21.6, 21.6, 21.0)[group_position]
        if len(members) != expected_count or not math.isclose(total, expected_capacity, abs_tol=1e-12):
            raise RuntimeError(f"group capacity contract changed: {group}")
        group_weights[group_position, members] = capacities / total
        group_audit.append(
            {
                "group": group,
                "site_ids": [int(sites[i]["site_id"]) for i in members],
                "capacity_mw": total,
                "site_capacity_weights": [float(value) for value in capacities / total],
            }
        )
    audit = {
        "coordinate_lock": file_identity(COORDINATE_LOCK),
        "corner_coordinates": [list(row) for row in CORNER_COORDINATES],
        "sites": site_audit,
        "groups": group_audit,
        "method": "regular_ll bilinear at each authoritative site, then capacity-weighted group mean",
    }
    return sites, site_weights, group_weights, audit


def expected_grid_indexes(keys: dict[str, Any]) -> list[int]:
    if (
        keys["gridType"] != "regular_ll"
        or int(keys["Ni"]) != 1440
        or int(keys["Nj"]) != 721
        or int(keys["iScansNegatively"]) != 0
        or int(keys["jScansPositively"]) != 0
        or int(keys["jPointsAreConsecutive"]) != 0
        or int(keys["alternativeRowScanning"]) != 0
        or float(keys["latitudeOfFirstGridPointInDegrees"]) != 90.0
        or float(keys["longitudeOfFirstGridPointInDegrees"]) != 0.0
        or float(keys["iDirectionIncrementInDegrees"]) != 0.25
        or float(keys["jDirectionIncrementInDegrees"]) != 0.25
    ):
        raise RuntimeError(f"unexpected regular_ll grid: {keys}")
    indexes: list[int] = []
    for latitude, longitude in CORNER_COORDINATES:
        i = int(round(longitude / 0.25)) % 1440
        j = int(round((90.0 - latitude) / 0.25))
        indexes.append(j * 1440 + i)
    return indexes


def decode_hpbl(
    payload: bytes,
    *,
    run_utc: datetime,
    valid_utc: datetime,
    forecast_hour: int,
    site_weights: np.ndarray,
    group_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not payload.startswith(b"GRIB") or not payload.endswith(b"7777"):
        raise RuntimeError("range does not contain a complete GRIB message")
    with ECCODES_LOCK:
        handle = eccodes.codes_new_from_message(payload)
        if handle is None:
            raise RuntimeError("ecCodes could not open in-memory HPBL message")
        try:
            key_names = (
            "discipline",
            "parameterCategory",
            "parameterNumber",
            "shortName",
            "typeOfLevel",
            "level",
            "dataDate",
            "dataTime",
            "forecastTime",
            "stepUnits",
            "indicatorOfUnitOfTimeRange",
            "validityDate",
            "validityTime",
            "gridType",
            "Ni",
            "Nj",
            "latitudeOfFirstGridPointInDegrees",
            "longitudeOfFirstGridPointInDegrees",
            "iDirectionIncrementInDegrees",
            "jDirectionIncrementInDegrees",
            "iScansNegatively",
            "jScansPositively",
            "jPointsAreConsecutive",
            "alternativeRowScanning",
            "totalLength",
        )
            keys = {name: eccodes.codes_get(handle, name) for name in key_names}
            if (
                str(keys["shortName"]) != "unknown"
                or int(keys["discipline"]) != 0
                or int(keys["parameterCategory"]) != 3
                or int(keys["parameterNumber"]) != 196
                or str(keys["typeOfLevel"]) != "surface"
                or int(keys["level"]) != 0
                or int(keys["forecastTime"]) != forecast_hour
                or int(keys["stepUnits"]) != 1
                or int(keys["indicatorOfUnitOfTimeRange"]) != 1
                or int(keys["dataDate"]) != int(run_utc.strftime("%Y%m%d"))
                or int(keys["dataTime"]) != 1200
                or int(keys["validityDate"]) != int(valid_utc.strftime("%Y%m%d"))
                or int(keys["validityTime"]) != int(valid_utc.strftime("%H%M"))
                or int(keys["totalLength"]) != len(payload)
            ):
                raise RuntimeError(f"HPBL GRIB semantic identity mismatch: {keys}")
            corner_values = np.asarray(
                eccodes.codes_get_elements(handle, "values", expected_grid_indexes(keys)),
                dtype=np.float64,
            )
        finally:
            eccodes.codes_release(handle)
    if (
        corner_values.shape != (4,)
        or not np.isfinite(corner_values).all()
        or np.min(corner_values) < 0
        or np.max(corner_values) > 20_000
    ):
        raise RuntimeError(f"implausible HPBL corner values: {corner_values}")
    site_values = site_weights @ corner_values
    group_values = group_weights @ site_values
    if not np.isfinite(site_values).all() or not np.isfinite(group_values).all():
        raise RuntimeError("non-finite interpolated HPBL values")
    return site_values, group_values, {
        "corner_values_m": [float(value) for value in corner_values],
        "grib_identity": {
            key: (int(value) if isinstance(value, (np.integer, int)) else value)
            for key, value in keys.items()
        },
    }


def fetch_hour(
    operating_day: date,
    forecast_hour: int,
    site_weights: np.ndarray,
    group_weights: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_day = operating_day - timedelta(days=2)
    run_utc = datetime.combine(run_day, datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=12)
    valid_utc = run_utc + timedelta(hours=forecast_hour)
    expected_kst_naive = datetime.combine(operating_day, datetime.min.time()) + timedelta(
        hours=forecast_hour - 27
    )
    observed_kst_naive = (valid_utc + timedelta(hours=9)).replace(tzinfo=None)
    if observed_kst_naive != expected_kst_naive:
        raise RuntimeError("D-2 12Z/f028..f051 time mapping changed")
    cutoff_utc = datetime.combine(
        operating_day - timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc
    ) + timedelta(hours=5)
    object_path, index_path = object_paths(run_day, forecast_hour)

    def validate_index(status: int, headers: dict[str, str], payload: bytes) -> None:
        if status != 200 or not payload:
            raise RuntimeError(f"index HTTP response invalid: status={status}, bytes={len(payload)}")
        if "content-length" in headers and int(headers["content-length"]) != len(payload):
            raise RuntimeError("index Content-Length mismatch")

    index_payload, index_headers, index_attempts, index_seconds = get_with_retry(
        index_path,
        headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"},
        validator=validate_index,
    )
    range_start, range_end, hpbl_index_line, next_index_line = parse_hpbl_range(index_payload)
    expected_bytes = range_end - range_start + 1

    def validate_range(status: int, headers: dict[str, str], payload: bytes) -> None:
        if status != 206:
            raise RuntimeError(f"range request was not honored: HTTP {status}")
        if len(payload) != expected_bytes:
            raise RuntimeError(f"range payload bytes changed: {len(payload)} != {expected_bytes}")
        if int(headers.get("content-length", "-1")) != expected_bytes:
            raise RuntimeError("range Content-Length mismatch")
        content_range = headers.get("content-range", "")
        if not content_range.startswith(f"bytes {range_start}-{range_end}/"):
            raise RuntimeError(f"range Content-Range mismatch: {content_range}")
        if "etag" not in headers or "last-modified" not in headers:
            raise RuntimeError("range response lacks ETag/Last-Modified")

    range_payload, range_headers, range_attempts, range_seconds = get_with_retry(
        object_path,
        headers={
            "User-Agent": USER_AGENT,
            "Accept-Encoding": "identity",
            "Range": f"bytes={range_start}-{range_end}",
        },
        validator=validate_range,
    )
    last_modified = parsedate_to_datetime(range_headers["last-modified"]).astimezone(timezone.utc)
    if last_modified > cutoff_utc:
        raise RuntimeError(
            f"object was not finalized by D-1 14:00 KST: {last_modified} > {cutoff_utc}"
        )
    site_values, group_values, decode_audit = decode_hpbl(
        range_payload,
        run_utc=run_utc,
        valid_utc=valid_utc,
        forecast_hour=forecast_hour,
        site_weights=site_weights,
        group_weights=group_weights,
    )
    row: dict[str, Any] = {
        "operating_day_kst": operating_day.isoformat(),
        "forecast_kst_dtm": expected_kst_naive,
        "valid_time_utc": valid_utc,
        "run_init_utc": run_utc,
        "forecast_hour": np.int16(forecast_hour),
        "cutoff_utc": cutoff_utc,
    }
    for i, value in enumerate(site_values, start=1):
        row[f"hpbl_site_{i:02d}_m"] = np.float32(value)
    for i, value in enumerate(group_values, start=1):
        row[f"hpbl_group_{i}_m"] = np.float32(value)
    evidence = {
        "operating_day_kst": operating_day.isoformat(),
        "forecast_kst_dtm": expected_kst_naive.isoformat(sep=" "),
        "valid_time_utc": valid_utc.isoformat().replace("+00:00", "Z"),
        "run_init_utc": run_utc.isoformat().replace("+00:00", "Z"),
        "forecast_hour": forecast_hour,
        "cutoff_utc": cutoff_utc.isoformat().replace("+00:00", "Z"),
        "publication_last_modified_utc": last_modified.isoformat().replace("+00:00", "Z"),
        "cutoff_margin_seconds": int((cutoff_utc - last_modified).total_seconds()),
        "index": {
            "url": f"https://{HOST}{index_path}",
            "http_status": 200,
            "response_bytes": len(index_payload),
            "response_sha256": sha256_bytes(index_payload),
            "etag": index_headers.get("etag"),
            "last_modified": index_headers.get("last-modified"),
            "hpbl_line": hpbl_index_line,
            "next_line": next_index_line,
            "attempts": index_attempts,
            "elapsed_seconds": index_seconds,
        },
        "range": {
            "url": f"https://{HOST}{object_path}",
            "request_range_start": range_start,
            "request_range_end": range_end,
            "http_status": 206,
            "content_range": range_headers["content-range"],
            "content_length": int(range_headers["content-length"]),
            "response_bytes": len(range_payload),
            "response_sha256": sha256_bytes(range_payload),
            "etag": range_headers["etag"],
            "last_modified": range_headers["last-modified"],
            "x_amz_version_id": range_headers.get("x-amz-version-id"),
            "x_amz_request_id": range_headers.get("x-amz-request-id"),
            "attempts": range_attempts,
            "elapsed_seconds": range_seconds,
            "retrieved_at_utc": utc_now(),
        },
        "decode": decode_audit,
    }
    return row, evidence


def validate_day_frame(frame: pd.DataFrame, operating_day: date) -> None:
    expected = pd.date_range(
        datetime.combine(operating_day, datetime.min.time()) + timedelta(hours=1),
        periods=24,
        freq="h",
    )
    if (
        len(frame) != 24
        or not pd.DatetimeIndex(frame["forecast_kst_dtm"]).equals(expected)
        or list(frame["forecast_hour"].astype(int)) != list(range(28, 52))
        or frame["forecast_kst_dtm"].duplicated().any()
    ):
        raise RuntimeError(f"day index validation failed: {operating_day}")
    value_columns = [column for column in frame if column.startswith("hpbl_")]
    values = frame[value_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or np.min(values) < 0 or np.max(values) > 20_000:
        raise RuntimeError(f"day HPBL value validation failed: {operating_day}")


def day_paths(root: Path, operating_day: date) -> tuple[Path, Path]:
    stem = operating_day.isoformat()
    return root / "daily_values" / f"{stem}.parquet", root / "daily_provenance" / f"{stem}.json"


def load_completed_day(root: Path, operating_day: date) -> pd.DataFrame | None:
    parquet_path, provenance_path = day_paths(root, operating_day)
    if not parquet_path.exists() and not provenance_path.exists():
        return None
    if not parquet_path.is_file() or not provenance_path.is_file():
        raise RuntimeError(f"incomplete daily checkpoint: {operating_day}")
    frame = pd.read_parquet(parquet_path)
    validate_day_frame(frame, operating_day)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if (
        provenance.get("operating_day_kst") != operating_day.isoformat()
        or provenance.get("hours") != 24
        or provenance.get("values_parquet", {}).get("sha256") != sha256_file(parquet_path)
        or len(provenance.get("responses", [])) != 24
    ):
        raise RuntimeError(f"daily provenance validation failed: {operating_day}")
    return frame


def acquire_day(
    root: Path,
    operating_day: date,
    executor: concurrent.futures.ThreadPoolExecutor,
    site_weights: np.ndarray,
    group_weights: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any], bool]:
    completed = load_completed_day(root, operating_day)
    if completed is not None:
        provenance_path = day_paths(root, operating_day)[1]
        return completed, json.loads(provenance_path.read_text(encoding="utf-8")), False
    started = time.perf_counter()
    futures = {
        executor.submit(fetch_hour, operating_day, fh, site_weights, group_weights): fh
        for fh in range(28, 52)
    }
    rows: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    try:
        for future in concurrent.futures.as_completed(futures):
            row, record = future.result()
            rows.append(row)
            evidence.append(record)
    except Exception:
        for future in futures:
            future.cancel()
        raise
    rows.sort(key=lambda row: int(row["forecast_hour"]))
    evidence.sort(key=lambda row: int(row["forecast_hour"]))
    frame = pd.DataFrame.from_records(rows)
    validate_day_frame(frame, operating_day)
    parquet_path, provenance_path = day_paths(root, operating_day)
    write_parquet_exclusive(parquet_path, frame)
    response_bytes = int(sum(record["range"]["response_bytes"] for record in evidence))
    payload = {
        "schema_version": 1,
        "artifact_type": "NOAA_NODD_GFS_HPBL_DAILY_STREAM_DECODE_PROVENANCE",
        "created_utc": utc_now(),
        "operating_day_kst": operating_day.isoformat(),
        "hours": 24,
        "range_response_bytes": response_bytes,
        "elapsed_seconds": time.perf_counter() - started,
        "values_parquet": file_identity(parquet_path),
        "responses": evidence,
    }
    write_json_exclusive(provenance_path, payload)
    return frame, payload, True


def run_preflight(workers: int) -> dict[str, Any]:
    root = ROOT / "preflight"
    root.mkdir(parents=True, exist_ok=True)
    _, site_weights, group_weights, spatial_audit = spatial_contract()
    day = date(2024, 1, 1)
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        frame, provenance, downloaded = acquire_day(
            root, day, executor, site_weights, group_weights
        )
    elapsed = time.perf_counter() - started
    response_bytes = int(provenance["range_response_bytes"])
    retry_attempts = int(
        sum(
            record[part]["attempts"] - 1
            for record in provenance["responses"]
            for part in ("index", "range")
        )
    )
    estimated_seconds = elapsed * EXPECTED_HOURS / 24
    report = {
        "schema_version": 1,
        "artifact_type": "NOAA_NODD_GFS_HPBL_2024_ONE_DAY_PREFLIGHT",
        "created_utc": utc_now(),
        "operating_day_kst": day.isoformat(),
        "downloaded_now": downloaded,
        "workers": workers,
        "hours": len(frame),
        "range_response_bytes": response_bytes,
        "elapsed_seconds": elapsed,
        "range_throughput_mib_per_second": response_bytes / max(elapsed, 1e-9) / (1 << 20),
        "retry_attempts": retry_attempts,
        "logical_request_error_rate": retry_attempts / 48,
        "estimated_full_2024_seconds_linear": estimated_seconds,
        "estimated_full_2024_hours_linear": estimated_seconds / 3600,
        "eta_gate_le_12_hours": estimated_seconds <= 12 * 3600,
        "index_exact": True,
        "time_mapping_exact": True,
        "finite_values": True,
        "value_min_m": float(frame.filter(like="hpbl_").min().min()),
        "value_max_m": float(frame.filter(like="hpbl_").max().max()),
        "spatial_contract": spatial_audit,
        "daily_values": file_identity(day_paths(root, day)[0]),
        "daily_provenance": file_identity(day_paths(root, day)[1]),
    }
    report_path = root / "PREFLIGHT_REPORT.json"
    if report_path.exists():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("daily_values", {}).get("sha256") != report["daily_values"]["sha256"]:
            raise RuntimeError("existing preflight report binds different values")
    else:
        write_json_exclusive(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return report


def validate_year(frame: pd.DataFrame) -> None:
    expected = pd.date_range("2024-01-01 01:00:00", "2025-01-01 00:00:00", freq="h")
    if (
        len(frame) != EXPECTED_HOURS
        or not pd.DatetimeIndex(frame["forecast_kst_dtm"]).equals(expected)
        or frame["forecast_kst_dtm"].duplicated().any()
    ):
        raise RuntimeError("full-year 2024 time index is not exact")
    values = frame.filter(like="hpbl_").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or np.min(values) < 0 or np.max(values) > 20_000:
        raise RuntimeError("full-year HPBL values are invalid")


def run_full(workers: int) -> dict[str, Any]:
    preflight_report = ROOT / "preflight" / "PREFLIGHT_REPORT.json"
    if not preflight_report.is_file():
        raise RuntimeError("one-day preflight must be completed first")
    preflight = json.loads(preflight_report.read_text(encoding="utf-8"))
    if not preflight.get("eta_gate_le_12_hours"):
        raise RuntimeError("preflight ETA exceeds the fixed 12-hour launch gate")
    ROOT.mkdir(parents=True, exist_ok=True)
    _, site_weights, group_weights, spatial_audit = spatial_contract()
    days = [date(2024, 1, 1) + timedelta(days=offset) for offset in range(366)]
    started = time.perf_counter()
    frames: list[pd.DataFrame] = []
    downloaded_days = 0
    total_range_bytes = 0
    total_retries = 0
    last_progress = started
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for position, day in enumerate(days, start=1):
            try:
                frame, provenance, downloaded = acquire_day(
                    ROOT, day, executor, site_weights, group_weights
                )
            except Exception as exc:
                failure_path = ROOT / "failures" / f"{day.isoformat()}__{int(time.time())}.json"
                write_json_exclusive(
                    failure_path,
                    {
                        "created_utc": utc_now(),
                        "operating_day_kst": day.isoformat(),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                raise
            frames.append(frame)
            downloaded_days += int(downloaded)
            total_range_bytes += int(provenance["range_response_bytes"])
            total_retries += int(
                sum(
                    record[part]["attempts"] - 1
                    for record in provenance["responses"]
                    for part in ("index", "range")
                )
            )
            now = time.perf_counter()
            if position == 1 or position == len(days) or now - last_progress >= 600:
                elapsed = now - started
                remaining_seconds = elapsed / position * (len(days) - position)
                progress = {
                    "created_utc": utc_now(),
                    "completed_days": position,
                    "total_days": len(days),
                    "completed_hours": position * 24,
                    "total_hours": EXPECTED_HOURS,
                    "range_bytes": total_range_bytes,
                    "retry_attempts": total_retries,
                    "elapsed_seconds": elapsed,
                    "eta_seconds": remaining_seconds,
                    "eta_utc": (
                        datetime.now(timezone.utc) + timedelta(seconds=remaining_seconds)
                    ).isoformat().replace("+00:00", "Z"),
                }
                print(json.dumps(progress, ensure_ascii=False, sort_keys=True), flush=True)
                progress_path = ROOT / "progress" / f"day_{position:03d}.json"
                if not progress_path.exists():
                    write_json_exclusive(progress_path, progress)
                last_progress = now
    year = pd.concat(frames, ignore_index=True)
    year = year.sort_values("forecast_kst_dtm").reset_index(drop=True)
    validate_year(year)
    year_path = ROOT / "hpbl_2024_hourly.parquet"
    if year_path.exists():
        persisted = pd.read_parquet(year_path)
        validate_year(persisted)
        if not persisted.equals(year):
            raise RuntimeError("existing full-year parquet differs from daily checkpoints")
    else:
        write_parquet_exclusive(year_path, year)
    elapsed = time.perf_counter() - started
    manifest = {
        "schema_version": 1,
        "artifact_type": "NOAA_NODD_GFS_HPBL_2024_STREAM_DECODE_MANIFEST",
        "created_utc": utc_now(),
        "status": "COMPLETE",
        "causal_mapping": "operating D -> D-2 12Z, f028..f051",
        "publication_cutoff": "D-1 14:00 KST",
        "source": f"https://{HOST}/gfs.YYYYMMDD/12/atmos/gfs.t12z.pgrb2.0p25.fFFF",
        "range_selector": "exactly one :HPBL:surface: index line through byte before next index offset",
        "operating_days": len(days),
        "hours": len(year),
        "workers": workers,
        "downloaded_days_this_invocation": downloaded_days,
        "range_response_bytes": total_range_bytes,
        "retry_attempts": total_retries,
        "elapsed_seconds_this_invocation": elapsed,
        "raw_grib_payloads_persisted": 0,
        "daily_provenance_files": len(days),
        "daily_value_files": len(days),
        "spatial_contract": spatial_audit,
        "preflight": file_identity(preflight_report),
        "year_values": file_identity(year_path),
        "value_columns": [column for column in year if column.startswith("hpbl_")],
        "value_min_m": float(year.filter(like="hpbl_").min().min()),
        "value_max_m": float(year.filter(like="hpbl_").max().max()),
        "script": file_identity(Path(__file__).resolve()),
        "runtime": {
            "python": sys.version,
            "eccodes": eccodes.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
        },
    }
    manifest_path = ROOT / "MANIFEST.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("year_values", {}).get("sha256") != manifest["year_values"]["sha256"]:
            raise RuntimeError("existing year manifest binds different values")
    else:
        write_json_exclusive(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "full"))
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if not 1 <= args.workers <= 24:
        raise SystemExit("--workers must be in [1, 24]")
    if args.mode == "preflight":
        run_preflight(args.workers)
    else:
        run_full(args.workers)


if __name__ == "__main__":
    main()

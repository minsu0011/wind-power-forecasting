#!/usr/bin/env python3
"""Read-only 2022-2024 NOAA operational GEFS object/index census.

This program intentionally HEADs GRIB objects and GETs only their text `.idx`
sidecars.  It never GETs or decodes meteorological GRIB values and has no
label, model, metric, prediction, 2025-initialization, or CSV-submission reader.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gzip
import hashlib
import http.client
import io
import json
import os
import re
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_ROOT / "configs" / "noaa_gefs_operational_spread_upperair_census_protocol_v1.json"
OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "external" / "noaa_gefs_operational_spread_upperair_census_v1"
HOST = "noaa-gefs-pds.s3.amazonaws.com"
USER_AGENT = "baram2026-noaa-gefs-readonly-census-v1/1.0"
FIRST_OPERATING_DAY = date(2022, 1, 1)
LAST_OPERATING_DAY = date(2024, 12, 31)
LEADS = tuple(range(27, 55, 3))
EXPECTED_DAY_COUNT = 1096
EXPECTED_OBJECT_COUNT = EXPECTED_DAY_COUNT * len(LEADS) * 4
MAX_RESPONSE_BYTES = 256_000
TRANSIENT_STATUS = {408, 425, 429, 500, 502, 503, 504}
INDEX_LINE_RE = re.compile(r"^(?P<record>\d+):(?P<offset>\d+):(?P<body>.*)$")


@dataclass(frozen=True)
class Product:
    product_id: str
    directory: str
    stem: str
    level: str
    statistic: str


PRODUCTS = (
    Product("surface_0p25_ens_mean", "pgrb2sp25", "geavg.t12z.pgrb2s.0p25", "10 m above ground", "ens mean"),
    Product("surface_0p25_ens_spread", "pgrb2sp25", "gespr.t12z.pgrb2s.0p25", "10 m above ground", "ens std dev"),
    Product("pressure_0p50_ens_mean", "pgrb2ap5", "geavg.t12z.pgrb2a.0p50", "850 mb", "ens mean"),
    Product("pressure_0p50_ens_spread", "pgrb2ap5", "gespr.t12z.pgrb2a.0p50", "850 mb", "ens std dev"),
)


OFFICIAL_SOURCES = (
    (
        "ncei_gefs",
        "https://www.ncei.noaa.gov/products/weather-climate-models/global-ensemble-forecast",
        ("not officially archived", "1/1/2017"),
    ),
    (
        "ncei_nodd",
        "https://www.ncei.noaa.gov/products/ncei-data-noaa-open-dissemination-program",
        ("Global Ensemble Forecast System",),
    ),
    (
        "emc_gefs",
        "https://www.emc.ncep.noaa.gov/emc/pages/numerical_forecast_systems/gefs.php",
        ("31 members", "September 23, 2020"),
    ),
    (
        "aws_registry_gefs",
        "https://registry.opendata.aws/noaa-gefs/",
        ("open to the public", "used as desired", "noaa-gefs-pds"),
    ),
    (
        "nco_geavg_0p50_inventory",
        "https://www.nco.ncep.noaa.gov/pmb/products/gens/geavg.t00z.pgrb2a.0p50.f003.shtml",
        ("850 mb", "UGRD", "10 m above ground", "ens mean"),
    ),
    (
        "nco_gespr_0p50_inventory",
        "https://www.nco.ncep.noaa.gov/pmb/products/gens/gespr.t00z.pgrb2a.0p50.f003.shtml",
        ("850 mb", "UGRD", "10 m above ground", "ens std dev"),
    ),
    (
        "nco_gespr_0p25_inventory",
        "https://www.nco.ncep.noaa.gov/pmb/products/gens/gespr.t00z.pgrb2s.0p25.f003.shtml",
        ("10 m above ground", "UGRD", "ens std dev"),
    ),
    (
        "dacon_rules",
        "https://dacon.io/competitions/official/236727/overview/rules",
        ("14:00",),
    ),
    (
        "dacon_evaluation",
        "https://dacon.io/competitions/official/236727/overview/evaluation",
        ("Public Score", "40%"),
    ),
)


_THREAD_LOCAL = threading.local()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iso_utc(value: datetime | None) -> str:
    if value is None:
        return ""
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_http_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = parsedate_to_datetime(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def operating_days() -> Iterable[date]:
    current = FIRST_OPERATING_DAY
    while current <= LAST_OPERATING_DAY:
        yield current
        current += timedelta(days=1)


def data_key(source_day: date, product: Product, lead: int) -> str:
    return (
        f"/gefs.{source_day:%Y%m%d}/12/atmos/{product.directory}/"
        f"{product.stem}.f{lead:03d}"
    )


def get_connection(reset: bool = False) -> http.client.HTTPSConnection:
    connection = getattr(_THREAD_LOCAL, "connection", None)
    if reset and connection is not None:
        try:
            connection.close()
        except Exception:
            pass
        connection = None
    if connection is None:
        context = ssl.create_default_context()
        connection = http.client.HTTPSConnection(HOST, timeout=35, context=context)
        _THREAD_LOCAL.connection = connection
    return connection


def s3_request(method: str, key: str, retries: int) -> dict[str, Any]:
    last_error = ""
    for attempt in range(1, retries + 1):
        connection = get_connection(reset=(attempt > 1))
        try:
            connection.request(
                method,
                key,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "*/*",
                    "Accept-Encoding": "identity",
                    "Connection": "keep-alive",
                },
            )
            response = connection.getresponse()
            body = response.read(MAX_RESPONSE_BYTES + 1) if method == "GET" else response.read()
            headers = {name.lower(): value for name, value in response.getheaders()}
            status = int(response.status)
            if len(body) > MAX_RESPONSE_BYTES:
                raise RuntimeError(f"response exceeds {MAX_RESPONSE_BYTES} bytes")
            if status in TRANSIENT_STATUS and attempt < retries:
                time.sleep(min(8.0, 0.25 * (2 ** (attempt - 1))))
                continue
            return {
                "status": status,
                "headers": headers,
                "body": body,
                "attempts": attempt,
                "error": "",
            }
        except Exception as exc:  # network census must record and fail closed
            last_error = f"{type(exc).__name__}: {exc}"
            get_connection(reset=True)
            if attempt < retries:
                time.sleep(min(8.0, 0.25 * (2 ** (attempt - 1))))
    return {"status": 0, "headers": {}, "body": b"", "attempts": retries, "error": last_error}


def parse_index(
    payload: bytes,
    source_day: date,
    product: Product,
    lead: int,
    data_length: int,
) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        return {"pass": False, "error": f"UnicodeDecodeError: {exc}"}

    records: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        match = INDEX_LINE_RE.match(raw_line)
        if not match:
            continue
        body_parts = match.group("body").split(":")
        records.append(
            {
                "record": int(match.group("record")),
                "offset": int(match.group("offset")),
                "body_parts": body_parts,
                "line": raw_line,
            }
        )

    offsets = [record["offset"] for record in records]
    offsets_strict = bool(offsets) and all(right > left for left, right in zip(offsets, offsets[1:]))
    offsets_in_bounds = bool(offsets) and offsets[0] >= 0 and offsets[-1] < data_length
    expected_init = f"d={source_day:%Y%m%d}12"
    expected_fcst = f"{lead} hour fcst"
    selected: dict[str, dict[str, Any]] = {}
    duplicate_or_missing = False

    for variable in ("UGRD", "VGRD"):
        matches: list[tuple[int, dict[str, Any]]] = []
        for index, record in enumerate(records):
            parts = record["body_parts"]
            if len(parts) < 5:
                continue
            if (
                parts[0] == expected_init
                and parts[1] == variable
                and parts[2] == product.level
                and parts[3] == expected_fcst
                and parts[4] == product.statistic
            ):
                matches.append((index, record))
        if len(matches) != 1:
            duplicate_or_missing = True
            continue
        index, record = matches[0]
        start = int(record["offset"])
        end = int(records[index + 1]["offset"] - 1) if index + 1 < len(records) else data_length - 1
        selected[variable] = {
            "line": record["line"],
            "record": int(record["record"]),
            "byte_start": start,
            "byte_end": end,
            "byte_count": end - start + 1,
            "range_valid": 0 <= start <= end < data_length,
        }

    selected_range_valid = len(selected) == 2 and all(item["range_valid"] for item in selected.values())
    passed = offsets_strict and offsets_in_bounds and not duplicate_or_missing and selected_range_valid
    return {
        "pass": passed,
        "error": "" if passed else "index semantic or byte-range validation failed",
        "record_count": len(records),
        "offsets_strict": offsets_strict,
        "offsets_in_bounds": offsets_in_bounds,
        "selected": selected,
    }


def inspect_job(job: tuple[date, Product, int], retries: int) -> dict[str, Any]:
    operating_day, product, lead = job
    source_day = operating_day - timedelta(days=2)
    cutoff = datetime.combine(operating_day - timedelta(days=1), dt_time(5, 0), tzinfo=timezone.utc)
    key = data_key(source_day, product, lead)
    idx_key = key + ".idx"
    assert source_day.year <= 2024
    assert "/gefs.2025" not in key

    head = s3_request("HEAD", key, retries)
    index = s3_request("GET", idx_key, retries)
    head_headers = head["headers"]
    idx_headers = index["headers"]
    data_length_text = head_headers.get("content-length", "0")
    try:
        data_length = int(data_length_text)
    except ValueError:
        data_length = 0
    data_last_modified = parse_http_datetime(head_headers.get("last-modified"))
    idx_last_modified = parse_http_datetime(idx_headers.get("last-modified"))
    data_cutoff_pass = data_last_modified is not None and data_last_modified < cutoff
    idx_cutoff_pass = idx_last_modified is not None and idx_last_modified < cutoff
    idx_length_header = idx_headers.get("content-length", "")
    try:
        idx_length_match = int(idx_length_header) == len(index["body"])
    except ValueError:
        idx_length_match = False

    parsed = (
        parse_index(index["body"], source_day, product, lead, data_length)
        if head["status"] == 200 and index["status"] == 200 and data_length > 0
        else {"pass": False, "error": "data HEAD or index GET did not return 200", "selected": {}}
    )
    u = parsed.get("selected", {}).get("UGRD", {})
    v = parsed.get("selected", {}).get("VGRD", {})
    overall_pass = bool(
        head["status"] == 200
        and index["status"] == 200
        and data_length > 0
        and bool(head_headers.get("etag"))
        and bool(idx_headers.get("etag"))
        and idx_length_match
        and data_cutoff_pass
        and idx_cutoff_pass
        and parsed.get("pass", False)
    )

    return {
        "operating_date": operating_day.isoformat(),
        "source_date": source_day.isoformat(),
        "cutoff_utc": iso_utc(cutoff),
        "product_id": product.product_id,
        "lead": lead,
        "data_key": key.lstrip("/"),
        "idx_key": idx_key.lstrip("/"),
        "data_url": f"https://{HOST}{key}",
        "idx_url": f"https://{HOST}{idx_key}",
        "data_status": head["status"],
        "data_bytes": data_length,
        "data_etag": head_headers.get("etag", "").strip('"'),
        "data_last_modified_utc": iso_utc(data_last_modified),
        "data_version_id": head_headers.get("x-amz-version-id", ""),
        "data_checksum_sha256_header": head_headers.get("x-amz-checksum-sha256", ""),
        "idx_status": index["status"],
        "idx_bytes": len(index["body"]),
        "idx_etag": idx_headers.get("etag", "").strip('"'),
        "idx_last_modified_utc": iso_utc(idx_last_modified),
        "idx_sha256": sha256_bytes(index["body"]),
        "idx_content_length_match": idx_length_match,
        "index_record_count": parsed.get("record_count", 0),
        "index_offsets_strict": parsed.get("offsets_strict", False),
        "index_offsets_in_bounds": parsed.get("offsets_in_bounds", False),
        "u_line": u.get("line", ""),
        "u_byte_start": u.get("byte_start", -1),
        "u_byte_end": u.get("byte_end", -1),
        "u_byte_count": u.get("byte_count", 0),
        "v_line": v.get("line", ""),
        "v_byte_start": v.get("byte_start", -1),
        "v_byte_end": v.get("byte_end", -1),
        "v_byte_count": v.get("byte_count", 0),
        "semantic_and_range_pass": bool(parsed.get("pass", False)),
        "data_cutoff_pass": data_cutoff_pass,
        "idx_cutoff_pass": idx_cutoff_pass,
        "overall_pass": overall_pass,
        "head_attempts": head["attempts"],
        "idx_attempts": index["attempts"],
        "error": " | ".join(item for item in (head["error"], index["error"], parsed.get("error", "")) if item),
        "idx_payload": index["body"],
    }


def retrieve_official_source(source_id: str, url: str, markers: tuple[str, ...], retries: int) -> dict[str, Any]:
    error = ""
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(
            url,
            headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity", "Accept": "text/html,*/*"},
        )
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                payload = response.read(8_000_001)
                if len(payload) > 8_000_000:
                    raise RuntimeError("official source snapshot exceeds 8 MB")
                status = int(response.status)
                headers = {name.lower(): value for name, value in response.headers.items()}
                final_url = response.geturl()
            decoded = payload.decode("utf-8", errors="replace")
            marker_pass = {marker: marker.lower() in decoded.lower() for marker in markers}
            return {
                "source_id": source_id,
                "url": url,
                "final_url": final_url,
                "status": status,
                "headers": headers,
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
                "markers": marker_pass,
                "all_markers_pass": all(marker_pass.values()),
                "attempts": attempt,
                "error": "",
                "payload": payload,
            }
        except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(min(8.0, 0.5 * (2 ** (attempt - 1))))
    return {
        "source_id": source_id,
        "url": url,
        "final_url": "",
        "status": 0,
        "headers": {},
        "bytes": 0,
        "sha256": sha256_bytes(b""),
        "markers": {marker: False for marker in markers},
        "all_markers_pass": False,
        "attempts": retries,
        "error": error,
        "payload": b"",
    }


def write_deterministic_zip(path: Path, entries: list[tuple[str, bytes]]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, payload in sorted(entries, key=lambda item: item[0]):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            info.create_system = 3
            archive.writestr(info, payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def write_deterministic_csv_gz(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, compresslevel=9, mtime=0) as gzip_handle:
            with io.TextIOWrapper(gzip_handle, encoding="utf-8", newline="") as text_handle:
                writer = csv.DictWriter(text_handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)


def file_identity(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--retries", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.workers <= 96:
        raise ValueError("workers must be in [1,96]")
    if not 1 <= args.retries <= 8:
        raise ValueError("retries must be in [1,8]")
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"refusing to overwrite canonical output: {OUTPUT_ROOT}")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol["status"] != "FROZEN_BEFORE_FULL_CENSUS_NETWORK_REQUESTS":
        raise RuntimeError("unexpected protocol status")
    if sha256_file(PROTOCOL_PATH) != "8bc6f637bd081dd05699d89f9829144e4d3527aae47091d6462403276142d1c0":
        raise RuntimeError("protocol SHA-256 mismatch")

    days = list(operating_days())
    if len(days) != EXPECTED_DAY_COUNT:
        raise RuntimeError(f"unexpected day count: {len(days)}")
    jobs = [(day, product, lead) for day in days for product in PRODUCTS for lead in LEADS]
    if len(jobs) != EXPECTED_OBJECT_COUNT:
        raise RuntimeError(f"unexpected job count: {len(jobs)}")
    all_keys = [data_key(day - timedelta(days=2), product, lead) for day, product, lead in jobs]
    if any("/gefs.2025" in key for key in all_keys):
        raise RuntimeError("2025 initialization URL generated")
    if min(day - timedelta(days=2) for day in days) != date(2021, 12, 30):
        raise RuntimeError("unexpected first source date")
    if max(day - timedelta(days=2) for day in days) != date(2024, 12, 29):
        raise RuntimeError("unexpected last source date")

    started = time.monotonic()
    print(
        json.dumps(
            {
                "event": "census_start",
                "operating_days": len(days),
                "objects": len(jobs),
                "head_requests": len(jobs),
                "index_get_requests": len(jobs),
                "workers": args.workers,
                "no_2025_initializations": True,
                "grib_gets": 0,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="gefs-census") as executor:
        futures = {executor.submit(inspect_job, job, args.retries): job for job in jobs}
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            try:
                result = future.result()
            except Exception as exc:
                day, product, lead = futures[future]
                result = {
                    "operating_date": day.isoformat(),
                    "source_date": (day - timedelta(days=2)).isoformat(),
                    "product_id": product.product_id,
                    "lead": lead,
                    "overall_pass": False,
                    "error": f"unhandled {type(exc).__name__}: {exc}",
                    "idx_payload": b"",
                }
            results.append(result)
            if completed % 500 == 0 or completed == len(futures):
                elapsed = time.monotonic() - started
                rate = completed / elapsed if elapsed else 0.0
                remaining = (len(futures) - completed) / rate if rate else 0.0
                failures = sum(not bool(item.get("overall_pass")) for item in results)
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "completed": completed,
                            "total": len(futures),
                            "failures_so_far": failures,
                            "rate_objects_per_second": round(rate, 2),
                            "eta_seconds": round(remaining, 1),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    results.sort(key=lambda row: (row["operating_date"], row["product_id"], int(row["lead"])))
    official_results = [retrieve_official_source(source_id, url, markers, args.retries) for source_id, url, markers in OFFICIAL_SOURCES]
    retrieved_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)

    index_entries = [(row["idx_key"], row.pop("idx_payload")) for row in results if row.get("idx_payload")]
    index_zip = OUTPUT_ROOT / "required_index_bytes.zip"
    write_deterministic_zip(index_zip, index_entries)

    snapshot_entries = [(f"{item['source_id']}.html", item.pop("payload")) for item in official_results if item.get("payload")]
    snapshot_zip = OUTPUT_ROOT / "official_source_snapshots.zip"
    write_deterministic_zip(snapshot_zip, snapshot_entries)

    fieldnames = [
        "operating_date", "source_date", "cutoff_utc", "product_id", "lead", "data_key", "idx_key",
        "data_url", "idx_url", "data_status", "data_bytes", "data_etag", "data_last_modified_utc",
        "data_version_id", "data_checksum_sha256_header", "idx_status", "idx_bytes", "idx_etag",
        "idx_last_modified_utc", "idx_sha256", "idx_content_length_match", "index_record_count",
        "index_offsets_strict", "index_offsets_in_bounds", "u_line", "u_byte_start", "u_byte_end",
        "u_byte_count", "v_line", "v_byte_start", "v_byte_end", "v_byte_count",
        "semantic_and_range_pass", "data_cutoff_pass", "idx_cutoff_pass", "overall_pass",
        "head_attempts", "idx_attempts", "error",
    ]
    metadata_path = OUTPUT_ROOT / "object_metadata.csv.gz"
    write_deterministic_csv_gz(metadata_path, results, fieldnames)

    total_required_range_bytes = sum(int(row.get("u_byte_count", 0)) + int(row.get("v_byte_count", 0)) for row in results)
    failures = [row for row in results if not bool(row.get("overall_pass"))]
    semantic_failures = sum(not bool(row.get("semantic_and_range_pass")) for row in results)
    cutoff_failures = sum(not (bool(row.get("data_cutoff_pass")) and bool(row.get("idx_cutoff_pass"))) for row in results)
    status_failures = sum(row.get("data_status") != 200 or row.get("idx_status") != 200 for row in results)
    retry_objects = sum(int(row.get("head_attempts", 1)) > 1 or int(row.get("idx_attempts", 1)) > 1 for row in results)
    product_counts: dict[str, dict[str, int]] = {}
    for product in PRODUCTS:
        subset = [row for row in results if row["product_id"] == product.product_id]
        product_counts[product.product_id] = {
            "expected": EXPECTED_DAY_COUNT * len(LEADS),
            "observed": len(subset),
            "passed": sum(bool(row.get("overall_pass")) for row in subset),
            "failed": sum(not bool(row.get("overall_pass")) for row in subset),
        }

    official_by_id = {item["source_id"]: item for item in official_results}
    required_provenance_ids = {
        "ncei_gefs",
        "ncei_nodd",
        "emc_gefs",
        "aws_registry_gefs",
        "nco_geavg_0p50_inventory",
        "nco_gespr_0p50_inventory",
        "nco_gespr_0p25_inventory",
        "dacon_rules",
        "dacon_evaluation",
    }
    provenance_pass = all(
        official_by_id[source_id]["status"] == 200 and official_by_id[source_id]["all_markers_pass"]
        for source_id in required_provenance_ids
    )
    coverage_pass = len(results) == EXPECTED_OBJECT_COUNT and not failures
    global_pass = coverage_pass and provenance_pass
    summary = {
        "schema_version": 1,
        "census_id": "noaa_gefs_operational_spread_upperair_census_v1",
        "created_utc": retrieved_utc,
        "status": "PASS_COVERAGE_CUTOFF_VARIABLES_AND_PROVENANCE" if global_pass else "NO_GO_CENSUS_FAILED",
        "protocol": file_identity(PROTOCOL_PATH),
        "scope": {
            "first_operating_day": FIRST_OPERATING_DAY.isoformat(),
            "last_operating_day": LAST_OPERATING_DAY.isoformat(),
            "operating_days": len(days),
            "first_source_date": (FIRST_OPERATING_DAY - timedelta(days=2)).isoformat(),
            "last_source_date": (LAST_OPERATING_DAY - timedelta(days=2)).isoformat(),
            "leads": list(LEADS),
            "products": [product.product_id for product in PRODUCTS],
            "expected_objects": EXPECTED_OBJECT_COUNT,
        },
        "request_accounting": {
            "grib_head_requests": len(results),
            "grib_get_requests": 0,
            "index_get_requests": len(results),
            "official_document_get_requests": len(official_results),
            "2025_initialization_requests": 0,
            "meteorological_value_bytes_read": 0,
            "label_reads": 0,
            "fits": 0,
            "predictions": 0,
            "scores": 0,
            "csvs": 0,
            "submissions": 0,
        },
        "coverage": {
            "coverage_pass": coverage_pass,
            "observed_objects": len(results),
            "passed_objects": len(results) - len(failures),
            "failed_objects": len(failures),
            "status_failures": status_failures,
            "semantic_or_range_failures": semantic_failures,
            "cutoff_failures": cutoff_failures,
            "objects_requiring_retry": retry_objects,
            "product_counts": product_counts,
            "failure_examples": [{key: row.get(key) for key in ("operating_date", "source_date", "product_id", "lead", "data_status", "idx_status", "error")} for row in failures[:100]],
        },
        "byte_preservation": {
            "exact_index_payloads_preserved": len(index_entries),
            "expected_index_payloads": EXPECTED_OBJECT_COUNT,
            "estimated_exact_required_grib_range_bytes_for_later_download": total_required_range_bytes,
            "grib_values_downloaded_now": False,
            "etag_is_treated_as_metadata_not_cryptographic_hash": True,
            "later_requirement": "Before any fit, range-download and locally retain every used U/V GRIB record with SHA-256 plus its bound index and response metadata."
        },
        "official_provenance": {
            "pass": provenance_pass,
            "source_snapshots": official_results,
            "nodd_is_official_archive": False,
            "description": "NOAA operational GEFS forecast products disseminated through the NOAA-managed NODD bucket; NCEI explicitly states NODD is not officially archived.",
            "license": "Open to the public and usable as desired; NOAA attribution requested; no NOAA endorsement implication; modified data must not be represented as original unaltered NOAA data.",
            "member_documentation_caveat": "NCEI/AWS landing prose contains a stale legacy 21-member description; EMC operational GEFSv12 and NCO raw-product semantics govern 2022-2024. This candidate consumes only geavg/gespr aggregate products and does not reconstruct members."
        },
        "decision": {
            "global_pass": global_pass,
            "model_preregister_allowed": global_pass,
            "fit_or_score_allowed_by_this_census": False,
            "if_fail": "NO-GO with no fallback, reanalysis, alternate cycle, alternate lead, or missing-value fill.",
            "if_pass": "Freeze one separate paired-increment preregistration and report its SHA before any label read, fit, prediction, or score."
        },
    }
    summary_path = OUTPUT_ROOT / "census_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "manifest_id": "noaa_gefs_operational_spread_upperair_census_v1_manifest",
        "created_utc": retrieved_utc,
        "canonical_root": OUTPUT_ROOT.relative_to(PROJECT_ROOT).as_posix(),
        "status": summary["status"],
        "source_closure": [file_identity(Path(__file__).resolve()), file_identity(PROTOCOL_PATH)],
        "artifacts": [
            file_identity(metadata_path),
            file_identity(index_zip),
            file_identity(snapshot_zip),
            file_identity(summary_path),
        ],
        "nonmutation": {
            "canonical_preexisted": False,
            "candidate_csvs_modified": 0,
            "labels_models_predictions_metrics_modified_or_created": 0,
        },
    }
    manifest_path = OUTPUT_ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "census_complete",
                "status": summary["status"],
                "passed_objects": len(results) - len(failures),
                "failed_objects": len(failures),
                "provenance_pass": provenance_pass,
                "summary_sha256": sha256_file(summary_path),
                "manifest_sha256": sha256_file(manifest_path),
                "elapsed_seconds": round(time.monotonic() - started, 1),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if global_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())

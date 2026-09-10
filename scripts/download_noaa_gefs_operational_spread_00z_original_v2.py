#!/usr/bin/env python3
"""Download the frozen Stage1-only NOAA GEFS original byte ranges.

The executable contract is deliberately narrower than the source census:
only operating days 2022-01-01..2023-12-31 are reachable.  It downloads two
merged GRIB2 ranges (10 m U/V and 850-hPa U/V) from each frozen NOAA pgrb2a
mean/spread object, preserves the exact response bytes in deterministic packs,
and derives one fixed native point plus hourly component interpolation.

There is no label/model/metric/2024-operating-day/2025 reader in this module.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import csv
import gzip
import hashlib
import http.client
import io
import json
import os
import re
import shutil
import ssl
import sys
import threading
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_PREREG = PROJECT_ROOT / "configs" / "noaa_gefs_operational_spread_00z_paired_increment_preregister_v1.json"
PREREG = PROJECT_ROOT / "configs" / "noaa_gefs_operational_spread_00z_paired_increment_preregister_v2.json"
PREREG_V3 = PROJECT_ROOT / "configs" / "noaa_gefs_operational_spread_00z_paired_increment_preregister_v3.json"
PREREG_V4 = PROJECT_ROOT / "configs" / "noaa_gefs_operational_spread_00z_paired_increment_preregister_v4.json"
PREREG_V5 = PROJECT_ROOT / "configs" / "noaa_gefs_operational_spread_00z_paired_increment_preregister_v5.json"
EXECUTION_PROTOCOL = PROJECT_ROOT / "configs" / "noaa_gefs_operational_spread_00z_stage1_execution_protocol_v2.json"
INCIDENT = PROJECT_ROOT / "artifacts" / "incidents" / "noaa_gefs_operational_spread_00z_paired_increment_v1_label_mask_contradiction.json"
CENSUS_ROOT = PROJECT_ROOT / "artifacts" / "external" / "noaa_gefs_operational_spread_00z_census_v2"
METADATA = CENSUS_ROOT / "object_metadata.csv.gz"
INDEX_ZIP = CENSUS_ROOT / "required_index_bytes.zip"
AVAILABILITY = CENSUS_ROOT / "operating_day_availability.json"
OUTPUT_PARENT = PROJECT_ROOT / "artifacts" / "external" / "noaa_gefs_operational_spread_00z_original_v2"
OUTPUT_ROOT = OUTPUT_PARENT / "stage1_source_through_operating_2023"
STAGE2_ROOT = OUTPUT_PARENT / "stage2_source_operating_2024"
LAUNCH_LOCK = PROJECT_ROOT / "artifacts" / "audits" / "noaa_gefs_operational_spread_00z_original_v2_stage1_launch_lock.json"
LAUNCH_LOCK_SHA = LAUNCH_LOCK.with_suffix(".json.sha256")
RUNNER = PROJECT_ROOT / "scripts" / "run_noaa_gefs_operational_spread_00z_paired_increment_v2.py"
FOCUSED_TEST = PROJECT_ROOT / "tests" / "test_noaa_gefs_operational_spread_00z_paired_increment_v2.py"
POSTRUN_TEST = PROJECT_ROOT / "tests" / "test_noaa_gefs_operational_spread_00z_paired_increment_v2_postrun.py"
INDEPENDENT_REVIEW = PROJECT_ROOT / "artifacts" / "audits" / "noaa_gefs_operational_spread_00z_paired_increment_v3_prelaunch_independent_review_v1.json"
SOURCE_ATTEMPT_TOMBSTONE = PROJECT_ROOT / "artifacts" / "audits" / "noaa_gefs_operational_spread_00z_original_v2_stage1_single_attempt.json"

HOST = "noaa-gefs-pds.s3.amazonaws.com"
USER_AGENT = "baram2026-noaa-gefs-stage1-original-range-v2/2.0"
FIRST_DAY = date(2022, 1, 1)
LAST_DAY = date(2023, 12, 31)
EXPECTED_DAYS = 730
LEADS = tuple(range(39, 64, 3))
PRODUCTS = ("pressure_0p50_ens_mean", "pressure_0p50_ens_spread")
PRODUCT_STATISTIC = {
    "pressure_0p50_ens_mean": "ens mean",
    "pressure_0p50_ens_spread": "ens std dev",
}
LEVELS = {"10m": "10 m above ground", "850": "850 mb"}
EXPECTED_OBJECTS = 13_140
EXPECTED_RANGES = 26_280
EXPECTED_BYTES = 11_524_284_739
EXPECTED_10M_BYTES = 5_810_625_029
EXPECTED_850_BYTES = 5_713_659_710
EXPECTED_BASE_SHA = "ccc2a55b979cbe50a6d79abc76b36e8553020da3f9f8d4461c1b4c2325438d9b"
EXPECTED_PREREG_SHA = "7af72639b806c559d1a417b8254f43582c7e728daeba4f5a42ae9cb42af8b2cd"
EXPECTED_PREREG_V3_SHA = "a6600228fd646f1dc347b682ad2db1008ccc9129707caa241d0842dac09aa893"
EXPECTED_PREREG_V4_SHA = "07e9ec6e9f08d18b7fe09421bcae6308009889a7970c9471a30447fd47e0e9a4"
EXPECTED_PREREG_V5_SHA = "df1514f6a8a1709929bc96e82bd043d11b0215de117f25a06232bab8fa531187"
EXPECTED_INCIDENT_SHA = "360b1369d83568d381bb2b6d9109debef21e9e32c13445bd6ef875fb82541f43"
EXPECTED_EXECUTION_PROTOCOL_SHA = "d3785e8557dd5d48a89174dd0a6f2f6a2321d588d17d7a75ec051423e936192d"
EXPECTED_METADATA_SHA = "44d8e5702599f57d1115c7b4dd5a137a55c03eae31178cfb6214265bf59f675a"
EXPECTED_INDEX_ZIP_SHA = "aa2fd9c880f04294bac75a62b73597317f56c450442c97ab0fea96ef5427346f"
EXPECTED_AVAILABILITY_SHA = "422c70c3404d5529e802e8955b10039fef18e7b27f2db64091b07b2bd7fecd07"
TRANSIENT_STATUS = {408, 425, 429, 500, 502, 503, 504}
INDEX_RE = re.compile(r"^(?P<record>\d+):(?P<offset>\d+):(?P<body>.*)$")
DATA_KEY_RE = re.compile(
    r"^gefs\.(?P<date>\d{8})/00/atmos/pgrb2ap5/"
    r"(?P<stem>geavg|gespr)\.t00z\.pgrb2a\.0p50\.f(?P<lead>039|042|045|048|051|054|057|060|063)$"
)
_THREAD_LOCAL = threading.local()


@dataclass(frozen=True)
class RangePlan:
    operating_date: str
    source_date: str
    cutoff_utc: str
    product_id: str
    statistic: str
    lead: int
    level_id: str
    level: str
    data_key: str
    data_url: str
    data_bytes: int
    data_etag: str
    data_last_modified_utc: str
    idx_key: str
    idx_sha256: str
    u_line: str
    v_line: str
    byte_start: int
    byte_end: int
    byte_count: int
    pack_path: str
    pack_offset: int


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def identity(path: Path) -> dict[str, Any]:
    return {"path": relative(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def write_text_exclusive(path: Path, payload: str, *, encoding: str = "ascii") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        with os.fdopen(descriptor, "w", encoding=encoding, newline="\n") as handle:
            handle.write(payload)
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _module_files(module: str) -> list[Path]:
    if not module:
        return []
    parts = tuple(part for part in module.split(".") if part)
    found: set[Path] = set()
    file_path = PROJECT_ROOT.joinpath(*parts).with_suffix(".py")
    init_path = PROJECT_ROOT.joinpath(*parts, "__init__.py")
    if file_path.is_file():
        found.add(file_path.resolve())
    if init_path.is_file():
        found.add(init_path.resolve())
    for depth in range(1, len(parts)):
        package_init = PROJECT_ROOT.joinpath(*parts[:depth], "__init__.py")
        if package_init.is_file():
            found.add(package_init.resolve())
    return sorted(found)


def recursive_local_imports(source_roots: Iterable[Path]) -> list[Path]:
    """Resolve executed local imports, including packages and relative imports."""
    visited: set[Path] = set()
    pending = [path.resolve() for path in source_roots]
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        if not path.is_file() or path.suffix != ".py":
            raise RuntimeError(f"source-closure root is not a Python file: {path}")
        visited.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    relative_parts = path.relative_to(PROJECT_ROOT).with_suffix("").parts[:-1]
                    keep = len(relative_parts) - node.level + 1
                    prefix = relative_parts[: max(keep, 0)]
                    base = ".".join((*prefix, *((node.module or "").split("."))))
                else:
                    base = node.module or ""
                modules.append(base.strip("."))
                modules.extend(
                    f"{base}.{alias.name}".strip(".")
                    for alias in node.names
                    if alias.name != "*"
                )
            for module in modules:
                resolved = _module_files(module)
                if resolved:
                    pending.extend(resolved)
                else:
                    top = module.split(".")[0] if module else ""
                    namespace_path = PROJECT_ROOT.joinpath(*module.split(".")) if module else PROJECT_ROOT
                    if namespace_path.is_dir():
                        continue
                    if top and (PROJECT_ROOT / top).exists():
                        raise RuntimeError(f"unresolved local import in closure: {module}")
    return sorted(visited)


def require_file(path: Path, expected_sha: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected_sha:
        raise RuntimeError(f"SHA mismatch for {path}: {actual} != {expected_sha}")


def parse_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"invalid frozen bool: {value!r}")


def parse_index(payload: bytes, row: dict[str, str], level: str) -> tuple[dict[str, Any], dict[str, Any]]:
    text = payload.decode("utf-8")
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        match = INDEX_RE.match(line)
        if match:
            records.append(
                {
                    "record": int(match.group("record")),
                    "offset": int(match.group("offset")),
                    "parts": match.group("body").split(":"),
                    "line": line,
                }
            )
    if not records or any(b["offset"] <= a["offset"] for a, b in zip(records, records[1:])):
        raise RuntimeError("index offsets are absent or not strictly increasing")
    expected = {
        "init": f"d={row['source_date'].replace('-', '')}00",
        "lead": f"{int(row['lead'])} hour fcst",
        "stat": PRODUCT_STATISTIC[row["product_id"]],
    }
    chosen: list[tuple[int, dict[str, Any]]] = []
    for variable in ("UGRD", "VGRD"):
        matches = [
            (i, item)
            for i, item in enumerate(records)
            if len(item["parts"]) >= 5
            and item["parts"][:5] == [expected["init"], variable, level, expected["lead"], expected["stat"]]
        ]
        if len(matches) != 1:
            raise RuntimeError(f"expected one {variable}/{level} record, found {len(matches)}")
        chosen.append(matches[0])
    (u_index, u), (v_index, v) = chosen
    if v_index != u_index + 1 or v["record"] != u["record"] + 1:
        raise RuntimeError("U/V records are not consecutive")
    if v_index + 1 < len(records):
        v_end = records[v_index + 1]["offset"] - 1
    else:
        v_end = int(row["data_bytes"]) - 1
    u_end = v["offset"] - 1
    if u_end < u["offset"] or v_end < v["offset"] or v_end >= int(row["data_bytes"]):
        raise RuntimeError("invalid merged U/V range")
    return (
        {"line": u["line"], "start": u["offset"], "end": u_end},
        {"line": v["line"], "start": v["offset"], "end": v_end},
    )


def build_stage1_plan() -> tuple[list[RangePlan], dict[str, int]]:
    require_file(BASE_PREREG, EXPECTED_BASE_SHA)
    require_file(PREREG, EXPECTED_PREREG_SHA)
    require_file(PREREG_V3, EXPECTED_PREREG_V3_SHA)
    require_file(PREREG_V4, EXPECTED_PREREG_V4_SHA)
    require_file(PREREG_V5, EXPECTED_PREREG_V5_SHA)
    require_file(INCIDENT, EXPECTED_INCIDENT_SHA)
    require_file(EXECUTION_PROTOCOL, EXPECTED_EXECUTION_PROTOCOL_SHA)
    require_file(METADATA, EXPECTED_METADATA_SHA)
    require_file(INDEX_ZIP, EXPECTED_INDEX_ZIP_SHA)
    require_file(AVAILABILITY, EXPECTED_AVAILABILITY_SHA)
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if prereg["status"] != "FROZEN_EXECUTABLE_SUPERSESSION_BEFORE_FIRST_METEOROLOGICAL_GRIB_VALUE_GET_OR_LABEL_READ_OR_FIT_OR_SCORE":
        raise RuntimeError("v2 prereg is not frozen")
    prereg_v3 = json.loads(PREREG_V3.read_text(encoding="utf-8"))
    if prereg_v3["status"] != "FROZEN_SOURCE_ONLY_EXECUTABLE_SUPERSESSION_BEFORE_FIRST_METEOROLOGICAL_GRIB_VALUE_GET_LABEL_READ_FIT_PREDICTION_OR_SCORE":
        raise RuntimeError("v3 prereg is not frozen")
    prereg_v4 = json.loads(PREREG_V4.read_text(encoding="utf-8"))
    if prereg_v4["status"] != "FROZEN_CHRONOLOGY_CORRECTION_BEFORE_FIRST_METEOROLOGICAL_GRIB_VALUE_GET_LABEL_READ_FIT_PREDICTION_OR_SCORE":
        raise RuntimeError("v4 prereg is not frozen")
    prereg_v5 = json.loads(PREREG_V5.read_text(encoding="utf-8"))
    if prereg_v5["status"] != "FROZEN_COMPLETE_CHRONOLOGY_CORRECTION_BEFORE_FIRST_METEOROLOGICAL_GRIB_VALUE_GET_LABEL_READ_FIT_PREDICTION_OR_SCORE":
        raise RuntimeError("v5 prereg is not frozen")
    phase = prereg["physical_source_phase_override"]
    if phase["stage1_only_operating_days"] != ["2022-01-01", "2023-12-31"]:
        raise RuntimeError("unexpected Stage1 date bounds")
    if STAGE2_ROOT.exists():
        raise RuntimeError("conditional Stage2 source namespace exists before Stage1 promotion")

    availability = json.loads(AVAILABILITY.read_text(encoding="utf-8"))
    stage1_days = [item for item in availability["days"] if FIRST_DAY <= date.fromisoformat(item["operating_date"]) <= LAST_DAY]
    if len(stage1_days) != EXPECTED_DAYS or not all(item["available"] for item in stage1_days):
        raise RuntimeError("Stage1 availability is not exactly 730/730")

    with gzip.open(METADATA, "rt", encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["product_id"] in PRODUCTS and FIRST_DAY <= date.fromisoformat(row["operating_date"]) <= LAST_DAY
        ]
    rows.sort(key=lambda r: (r["operating_date"], r["product_id"], int(r["lead"])))
    if len(rows) != EXPECTED_OBJECTS:
        raise RuntimeError(f"unexpected Stage1 object count: {len(rows)}")
    if any(not parse_bool(row["overall_pass"]) for row in rows):
        raise RuntimeError("a Stage1 source object failed the frozen census")
    if any(date.fromisoformat(row["operating_date"]).year not in (2022, 2023) for row in rows):
        raise RuntimeError("non-Stage1 operating day entered plan")
    if any(date.fromisoformat(row["source_date"]) > date(2023, 12, 29) for row in rows):
        raise RuntimeError("post-Stage1 source initialization entered plan")
    if any("gefs.2024" in row["data_key"] or "gefs.2025" in row["data_key"] for row in rows):
        raise RuntimeError("2024/2025 source key entered Stage1 plan")

    provisional: list[dict[str, Any]] = []
    with zipfile.ZipFile(INDEX_ZIP) as archive:
        infos = archive.infolist()
        if len(infos) != len({item.filename for item in infos}):
            raise RuntimeError("frozen index ZIP contains duplicate member names")
        if archive.testzip() is not None:
            raise RuntimeError("frozen index ZIP CRC validation failed")
        info_by_name = {item.filename: item for item in infos}
        names = set(info_by_name)
        for row in rows:
            if row["idx_key"] not in names:
                raise RuntimeError(f"missing frozen index member: {row['idx_key']}")
            if info_by_name[row["idx_key"]].file_size != int(row["idx_bytes"]):
                raise RuntimeError(f"index ZIP member length mismatch: {row['idx_key']}")
            payload = archive.read(row["idx_key"])
            if sha256_bytes(payload) != row["idx_sha256"]:
                raise RuntimeError(f"index SHA mismatch: {row['idx_key']}")
            for level_id, level in LEVELS.items():
                u, v = parse_index(payload, row, level)
                if level_id == "850":
                    frozen_850 = {
                        "u_line": row["u_line"],
                        "u_start": int(row["u_byte_start"]),
                        "u_end": int(row["u_byte_end"]),
                        "v_line": row["v_line"],
                        "v_start": int(row["v_byte_start"]),
                        "v_end": int(row["v_byte_end"]),
                    }
                    derived_850 = {
                        "u_line": u["line"], "u_start": u["start"], "u_end": u["end"],
                        "v_line": v["line"], "v_start": v["start"], "v_end": v["end"],
                    }
                    if derived_850 != frozen_850:
                        raise RuntimeError(f"850 range differs from frozen census row: {row['data_key']}")
                start, end = u["start"], v["end"]
                provisional.append(
                    {
                        "operating_date": row["operating_date"],
                        "source_date": row["source_date"],
                        "cutoff_utc": row["cutoff_utc"],
                        "product_id": row["product_id"],
                        "statistic": PRODUCT_STATISTIC[row["product_id"]],
                        "lead": int(row["lead"]),
                        "level_id": level_id,
                        "level": level,
                        "data_key": row["data_key"],
                        "data_url": row["data_url"],
                        "data_bytes": int(row["data_bytes"]),
                        "data_etag": row["data_etag"],
                        "data_last_modified_utc": row["data_last_modified_utc"],
                        "idx_key": row["idx_key"],
                        "idx_sha256": row["idx_sha256"],
                        "u_line": u["line"],
                        "v_line": v["line"],
                        "byte_start": start,
                        "byte_end": end,
                        "byte_count": end - start + 1,
                    }
                )
    provisional.sort(key=lambda x: (x["operating_date"], x["product_id"], x["lead"], x["level_id"]))
    if len(provisional) != EXPECTED_RANGES:
        raise RuntimeError(f"unexpected range count: {len(provisional)}")
    totals = {
        "10m": sum(x["byte_count"] for x in provisional if x["level_id"] == "10m"),
        "850": sum(x["byte_count"] for x in provisional if x["level_id"] == "850"),
    }
    if totals != {"10m": EXPECTED_10M_BYTES, "850": EXPECTED_850_BYTES}:
        raise RuntimeError(f"range byte total mismatch: {totals}")

    pack_offsets: dict[str, int] = {}
    plans: list[RangePlan] = []
    for item in provisional:
        source_year = date.fromisoformat(item["source_date"]).year
        pack_path = f"raw_packs/source_{source_year}_{item['product_id']}_{item['level_id']}.grib2pack"
        offset = pack_offsets.get(pack_path, 0)
        plans.append(RangePlan(**item, pack_path=pack_path, pack_offset=offset))
        pack_offsets[pack_path] = offset + item["byte_count"]
    if len(pack_offsets) != 12 or sum(pack_offsets.values()) != EXPECTED_BYTES:
        raise RuntimeError("pack plan closure mismatch")
    source_year_totals = {
        year: sum(plan.byte_count for plan in plans if plan.source_date.startswith(str(year)))
        for year in (2021, 2022, 2023)
    }
    if source_year_totals != {2021: 31_670_989, 2022: 5_767_366_204, 2023: 5_725_247_546}:
        raise RuntimeError(f"source-year byte closure mismatch: {source_year_totals}")
    return plans, dict(sorted(pack_offsets.items()))


def get_connection(reset: bool = False) -> http.client.HTTPSConnection:
    connection = getattr(_THREAD_LOCAL, "connection", None)
    if reset and connection is not None:
        try:
            connection.close()
        except Exception:
            pass
        connection = None
    if connection is None:
        connection = http.client.HTTPSConnection(HOST, timeout=60, context=ssl.create_default_context())
        _THREAD_LOCAL.connection = connection
    return connection


def parse_http_time(value: str) -> str:
    parsed = parsedate_to_datetime(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def range_get(
    plan: RangePlan,
    retries: int,
    cancel_event: threading.Event | None = None,
) -> tuple[bytes, dict[str, str], int]:
    operating_day = date.fromisoformat(plan.operating_date)
    source_day = date.fromisoformat(plan.source_date)
    if not FIRST_DAY <= operating_day <= LAST_DAY:
        raise RuntimeError("HTTP request outside Stage1 operating-date boundary")
    if source_day != operating_day - timedelta(days=2) or source_day > date(2023, 12, 29):
        raise RuntimeError("HTTP request has invalid or post-Stage1 initialization")
    if "gefs.2024" in plan.data_key or "gefs.2025" in plan.data_key:
        raise RuntimeError("forbidden 2024/2025 initialization key")
    if not plan.data_url.startswith(f"https://{HOST}/gefs.") or plan.data_url != f"https://{HOST}/{plan.data_key}":
        raise RuntimeError("source host/key whitelist failure")
    key_match = DATA_KEY_RE.fullmatch(plan.data_key)
    expected_stem = "geavg" if plan.product_id.endswith("ens_mean") else "gespr"
    if (
        key_match is None
        or key_match.group("date") != plan.source_date.replace("-", "")
        or key_match.group("stem") != expected_stem
        or int(key_match.group("lead")) != plan.lead
    ):
        raise RuntimeError("source path family/cycle/lead whitelist failure")
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/octet-stream",
        "Accept-Encoding": "identity",
        "Range": f"bytes={plan.byte_start}-{plan.byte_end}",
        "If-Match": f'"{plan.data_etag}"',
        "Connection": "keep-alive",
    }
    last_error = ""
    for attempt in range(1, retries + 1):
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("cancelled before HTTP request")
        try:
            connection = get_connection(reset=attempt > 1)
            connection.request("GET", "/" + plan.data_key, headers=headers)
            response = connection.getresponse()
            payload = response.read(plan.byte_count + 1)
            got_headers = {k.lower(): v for k, v in response.getheaders()}
            status = int(response.status)
            if status in TRANSIENT_STATUS and attempt < retries:
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("cancelled before transient retry")
                time.sleep(min(8.0, 0.5 * (2 ** (attempt - 1))))
                continue
            if status != 206:
                raise RuntimeError(f"HTTP {status}, expected 206")
            if len(payload) != plan.byte_count:
                raise RuntimeError(f"response length {len(payload)} != {plan.byte_count}")
            expected_content_range = f"bytes {plan.byte_start}-{plan.byte_end}/{plan.data_bytes}"
            if got_headers.get("content-range") != expected_content_range:
                raise RuntimeError(f"Content-Range mismatch: {got_headers.get('content-range')!r}")
            content_length = got_headers.get("content-length", "")
            if not content_length.isdecimal() or int(content_length) != plan.byte_count:
                raise RuntimeError(f"Content-Length mismatch: {content_length!r}")
            response_etag = got_headers.get("etag", "")
            if not response_etag or response_etag.startswith("W/") or len(response_etag) < 3 or response_etag[0] != '"' or response_etag[-1] != '"':
                raise RuntimeError(f"missing, weak, or malformed strong ETag: {response_etag!r}")
            if response_etag[1:-1] != plan.data_etag:
                raise RuntimeError("response ETag differs from census")
            response_last_modified = parse_http_time(got_headers["last-modified"])
            if response_last_modified != plan.data_last_modified_utc:
                raise RuntimeError("response Last-Modified differs from census")
            if datetime.fromisoformat(response_last_modified.replace("Z", "+00:00")) >= datetime.fromisoformat(plan.cutoff_utc.replace("Z", "+00:00")):
                raise RuntimeError("response object is not cutoff-safe")
            return payload, got_headers, attempt
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            get_connection(reset=True)
            if attempt < retries:
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("cancelled before exception retry")
                time.sleep(min(8.0, 0.5 * (2 ** (attempt - 1))))
    raise RuntimeError(f"range GET failed after {retries} attempts: {plan.data_key}: {last_error}")


def split_grib2_messages(payload: bytes) -> list[bytes]:
    messages: list[bytes] = []
    position = 0
    while position < len(payload):
        if len(payload) - position < 16 or payload[position : position + 4] != b"GRIB":
            raise RuntimeError("response is not a contiguous GRIB stream")
        if payload[position + 7] != 2:
            raise RuntimeError("response contains a non-GRIB2 message")
        length = int.from_bytes(payload[position + 8 : position + 16], "big")
        if length < 20 or position + length > len(payload):
            raise RuntimeError("invalid GRIB Section-0 totalLength")
        message = payload[position : position + length]
        if message[-4:] != b"7777":
            raise RuntimeError("GRIB message lacks terminal 7777")
        messages.append(message)
        position += length
    if position != len(payload) or len(messages) != 2:
        raise RuntimeError(f"expected exactly two contiguous GRIB2 messages, found {len(messages)}")
    return messages


def decode_two_messages(payload: bytes, plan: RangePlan) -> tuple[list[dict[str, Any]], list[float]]:
    import eccodes

    structural_messages = split_grib2_messages(payload)
    handle = io.BytesIO(payload)
    metadata: list[dict[str, Any]] = []
    values: list[float] = []
    expected_short = ("10u", "10v") if plan.level_id == "10m" else ("u", "v")
    expected_level_type = "heightAboveGround" if plan.level_id == "10m" else "isobaricInhPa"
    expected_level = 10 if plan.level_id == "10m" else 850
    for expected_name in expected_short:
        gid = eccodes.codes_grib_new_from_file(handle)
        if gid is None:
            raise RuntimeError("merged response ended before two GRIB messages")
        try:
            def get(key: str, default: Any = None) -> Any:
                try:
                    return eccodes.codes_get(gid, key)
                except Exception:
                    return default

            item = {
                "edition": int(get("edition", -1)),
                "shortName": str(get("shortName", "")),
                "typeOfLevel": str(get("typeOfLevel", "")),
                "level": int(get("level", -1)),
                "dataDate": int(get("dataDate", -1)),
                "dataTime": int(get("dataTime", -1)),
                "forecastTime": int(get("forecastTime", -1)),
                "stepType": str(get("stepType", "")),
                "derivedForecast": int(get("derivedForecast", -1)),
                "units": str(get("units", "")),
                "gridType": str(get("gridType", "")),
                "Ni": int(get("Ni", -1)),
                "Nj": int(get("Nj", -1)),
                "iDirectionIncrementInDegrees": float(get("iDirectionIncrementInDegrees", -1.0)),
                "jDirectionIncrementInDegrees": float(get("jDirectionIncrementInDegrees", -1.0)),
                "latitudeOfFirstGridPointInDegrees": float(get("latitudeOfFirstGridPointInDegrees", -999.0)),
                "latitudeOfLastGridPointInDegrees": float(get("latitudeOfLastGridPointInDegrees", -999.0)),
                "longitudeOfFirstGridPointInDegrees": float(get("longitudeOfFirstGridPointInDegrees", -999.0)),
                "longitudeOfLastGridPointInDegrees": float(get("longitudeOfLastGridPointInDegrees", -999.0)),
                "productDefinitionTemplateNumber": int(get("productDefinitionTemplateNumber", -1)),
            }
            if item["edition"] != 2 or item["shortName"] != expected_name:
                raise RuntimeError(f"unexpected GRIB variable metadata: {item}")
            if item["typeOfLevel"] != expected_level_type or item["level"] != expected_level:
                raise RuntimeError(f"unexpected GRIB level metadata: {item}")
            if item["dataDate"] != int(plan.source_date.replace("-", "")) or item["dataTime"] != 0:
                raise RuntimeError(f"unexpected GRIB initialization: {item}")
            if item["forecastTime"] != plan.lead:
                raise RuntimeError(f"unexpected GRIB lead: {item}")
            if item["stepType"] != "instant" or item["units"] not in {"m s**-1", "m/s"}:
                raise RuntimeError(f"unexpected GRIB step/units: {item}")
            if item["gridType"] != "regular_ll" or item["Ni"] != 720 or item["Nj"] != 361:
                raise RuntimeError(f"unexpected 0.5-degree grid: {item}")
            if (
                abs(item["iDirectionIncrementInDegrees"] - 0.5) > 1e-12
                or abs(item["jDirectionIncrementInDegrees"] - 0.5) > 1e-12
                or abs(item["latitudeOfFirstGridPointInDegrees"] - 90.0) > 1e-12
                or abs(item["latitudeOfLastGridPointInDegrees"] + 90.0) > 1e-12
                or abs(item["longitudeOfFirstGridPointInDegrees"] - 0.0) > 1e-12
                or abs(item["longitudeOfLastGridPointInDegrees"] - 359.5) > 1e-12
            ):
                raise RuntimeError(f"unexpected exact native-grid geometry: {item}")
            if item["productDefinitionTemplateNumber"] != 2:
                raise RuntimeError(f"unexpected ensemble-derived PDT: {item}")
            expected_derived = 0 if plan.product_id.endswith("ens_mean") else 4
            if item["derivedForecast"] != expected_derived:
                raise RuntimeError(f"unexpected derivedForecast: {item}")
            nearest = eccodes.codes_grib_find_nearest(gid, 37.5, 129.0, npoints=1)
            if len(nearest) != 1:
                raise RuntimeError("nearest-point lookup did not return exactly one point")
            nearest_item = nearest[0]
            if abs(float(nearest_item["lat"]) - 37.5) > 1e-12 or abs(float(nearest_item["lon"]) - 129.0) > 1e-12:
                raise RuntimeError(f"nearest point mismatch: {nearest_item}")
            value = float(nearest_item["value"])
            if not (-250.0 < value < 250.0):
                raise RuntimeError(f"nonfinite or implausible point value: {value}")
            if plan.product_id.endswith("ens_spread") and value < 0.0:
                raise RuntimeError(f"negative component standard deviation: {value}")
            metadata.append(item)
            values.append(value)
        finally:
            eccodes.codes_release(gid)
    third = eccodes.codes_grib_new_from_file(handle)
    if third is not None:
        eccodes.codes_release(third)
        raise RuntimeError("merged response contains more than two GRIB messages")
    if handle.tell() != len(payload):
        raise RuntimeError("GRIB decoder did not consume exact merged response")
    if [len(message) for message in structural_messages] != [
        int(plan.v_line.split(":", 2)[1]) - int(plan.u_line.split(":", 2)[1]),
        plan.byte_end - int(plan.v_line.split(":", 2)[1]) + 1,
    ]:
        raise RuntimeError("GRIB structural message lengths differ from frozen index ranges")
    return metadata, values


def download_one(plan: RangePlan, retries: int, cancel_event: threading.Event) -> tuple[RangePlan, bytes, dict[str, Any]]:
    if cancel_event.is_set():
        raise RuntimeError("cancelled before identical range GET")
    payload, headers, attempts = range_get(plan, retries, cancel_event)
    if cancel_event.is_set():
        raise RuntimeError("cancelled after range GET before decode/write")
    decoded, values = decode_two_messages(payload, plan)
    return plan, payload, {
        "response_sha256": sha256_bytes(payload),
        "response_content_range": headers["content-range"],
        "response_content_length": int(headers["content-length"]),
        "response_etag": headers["etag"].strip('"'),
        "response_last_modified_utc": parse_http_time(headers["last-modified"]),
        "attempts": attempts,
        "u_value": values[0],
        "v_value": values[1],
        "u_metadata": json.dumps(decoded[0], sort_keys=True, separators=(",", ":")),
        "v_metadata": json.dumps(decoded[1], sort_keys=True, separators=(",", ":")),
    }


def deterministic_csv_gz(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(text, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)


def expected_launch_semantics(plans: list[RangePlan], pack_sizes: dict[str, int]) -> dict[str, Any]:
    if not INDEPENDENT_REVIEW.is_file():
        raise FileNotFoundError("independent prelaunch review must exist before launch-lock freeze")
    review = json.loads(INDEPENDENT_REVIEW.read_text(encoding="utf-8"))
    if review.get("verdict") != "PASS" or review.get("network_value_label_fit_score_access") != 0:
        raise RuntimeError("independent prelaunch review did not record a zero-access PASS")
    source_roots = [Path(__file__), RUNNER, FOCUSED_TEST, POSTRUN_TEST]
    recursive_sources = recursive_local_imports(source_roots)
    return {
        "scope": {
            "operating_days": [FIRST_DAY.isoformat(), LAST_DAY.isoformat()],
            "days": EXPECTED_DAYS,
            "objects": EXPECTED_OBJECTS,
            "ranges": EXPECTED_RANGES,
            "bytes": EXPECTED_BYTES,
            "pack_sizes": pack_sizes,
            "2024_operating_values": 0,
            "2025_requests_or_values": 0,
            "labels_models_predictions_scores": 0,
        },
        "source_closure": [
            identity(BASE_PREREG), identity(PREREG), identity(PREREG_V3), identity(PREREG_V4), identity(PREREG_V5), identity(INCIDENT), identity(EXECUTION_PROTOCOL), identity(INDEPENDENT_REVIEW),
            identity(METADATA), identity(INDEX_ZIP), identity(AVAILABILITY),
        ],
        "source_roots": [identity(path) for path in source_roots],
        "recursive_local_imports": [identity(path) for path in recursive_sources if path not in {item.resolve() for item in source_roots}],
        "plan_sha256": sha256_bytes(
            ("\n".join(json.dumps(asdict(plan), sort_keys=True, separators=(",", ":")) for plan in plans) + "\n").encode("utf-8")
        ),
        "decoder": {
            "python": sys.version.split()[0],
            "eccodes_python": __import__("eccodes").__version__,
            "eccodes_api": __import__("eccodes").codes_get_api_version(),
        },
    }


def create_launch_lock(plans: list[RangePlan], pack_sizes: dict[str, int]) -> dict[str, Any]:
    if LAUNCH_LOCK.exists() or LAUNCH_LOCK_SHA.exists():
        raise FileExistsError("refusing to overwrite existing launch lock or SHA sidecar")
    matching_quarantine = list((PROJECT_ROOT / "artifacts/quarantine").glob("noaa_gefs_operational_spread_00z_original_v2_stage1_partial_*")) if (PROJECT_ROOT / "artifacts/quarantine").exists() else []
    if SOURCE_ATTEMPT_TOMBSTONE.exists() or matching_quarantine:
        raise RuntimeError("single-attempt tombstone/quarantine exists before launch-lock freeze")
    LAUNCH_LOCK.parent.mkdir(parents=True, exist_ok=True)
    semantics = expected_launch_semantics(plans, pack_sizes)
    payload = {
        "schema_version": 1,
        "lock_id": "noaa_gefs_operational_spread_00z_original_v2_stage1_launch_lock",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "FROZEN_BEFORE_FIRST_STAGE1_RANGE_GET",
        "candidate_or_performance_information": 0,
        "single_attempt_contract": {
            "tombstone_path": relative(SOURCE_ATTEMPT_TOMBSTONE),
            "must_be_absent_at_launch_lock_freeze": True,
            "exclusive_create_required_before_first_range_get": True,
            "any_existing_tombstone_or_matching_quarantine_forbids_launch": True,
        },
    }
    payload.update(semantics)
    write_json_exclusive(LAUNCH_LOCK, payload)
    lock_hash = sha256_file(LAUNCH_LOCK)
    write_text_exclusive(LAUNCH_LOCK_SHA, f"{lock_hash}  {LAUNCH_LOCK.name}\n")
    return payload


def verify_launch_lock(plans: list[RangePlan], pack_sizes: dict[str, int]) -> dict[str, Any]:
    if not LAUNCH_LOCK.is_file() or not LAUNCH_LOCK_SHA.is_file():
        raise FileNotFoundError("immutable launch lock and SHA sidecar must preexist before --execute")
    sidecar = LAUNCH_LOCK_SHA.read_text(encoding="ascii")
    expected_sidecar = f"{sha256_file(LAUNCH_LOCK)}  {LAUNCH_LOCK.name}\n"
    if sidecar != expected_sidecar:
        raise RuntimeError("launch-lock byte SHA sidecar mismatch")
    existing = json.loads(LAUNCH_LOCK.read_text(encoding="utf-8"))
    if existing.get("status") != "FROZEN_BEFORE_FIRST_STAGE1_RANGE_GET":
        raise RuntimeError("existing launch lock has unexpected status")
    current = expected_launch_semantics(plans, pack_sizes)
    for key, value in current.items():
        if existing.get(key) != value:
            raise RuntimeError(f"launch-lock current closure mismatch: {key}")
    if existing.get("candidate_or_performance_information") != 0:
        raise RuntimeError("launch lock contains candidate/performance information")
    return existing


def build_hourly(native: pd.DataFrame) -> pd.DataFrame:
    keys = ["operating_date", "lead"]
    wide = native.pivot(index=keys, columns=["product_id", "level_id"], values=["u_value", "v_value"])
    wide.columns = [f"{value}__{product}__{level}" for value, product, level in wide.columns]
    wide = wide.reset_index().sort_values(keys).reset_index(drop=True)
    expected_native_rows = EXPECTED_DAYS * len(LEADS)
    if len(wide) != expected_native_rows:
        raise RuntimeError(f"native wide row count mismatch: {len(wide)}")
    records: list[dict[str, Any]] = []
    mean_product = "pressure_0p50_ens_mean"
    spread_product = "pressure_0p50_ens_spread"
    for operating_date, group in wide.groupby("operating_date", sort=True):
        by_lead = group.set_index("lead")
        if tuple(int(x) for x in by_lead.index) != LEADS:
            raise RuntimeError(f"native lead closure mismatch: {operating_date}")
        day = pd.Timestamp(operating_date)
        for target_lead in range(40, 64):
            lower = 39 + 3 * ((target_lead - 39) // 3)
            upper = lower if target_lead == lower else lower + 3
            weight = 0.0 if lower == upper else (target_lead - lower) / (upper - lower)
            timestamp = day + pd.Timedelta(hours=target_lead - 39)
            row: dict[str, Any] = {
                "operating_date": operating_date,
                "forecast_kst_dtm": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "source_init_utc": (day - pd.Timedelta(days=2)).strftime("%Y-%m-%d 00:00:00"),
                "target_lead_hour": target_lead,
                "missing_gefs_00z": 0,
            }
            for level_id in ("10m", "850"):
                for component in ("u", "v"):
                    for product, statistic in ((mean_product, "mean"), (spread_product, "std")):
                        column = f"{component}_value__{product}__{level_id}"
                        left = float(by_lead.loc[lower, column])
                        right = float(by_lead.loc[upper, column])
                        feature_level = "10" if level_id == "10m" else "850"
                        row[f"gefs00z__{component}{feature_level}_{statistic}_ms"] = left + weight * (right - left)
            records.append(row)
    hourly = pd.DataFrame.from_records(records)
    hourly = hourly.loc[:, [
        "operating_date", "forecast_kst_dtm", "source_init_utc", "target_lead_hour", "missing_gefs_00z",
        "gefs00z__u10_mean_ms", "gefs00z__v10_mean_ms", "gefs00z__u10_std_ms", "gefs00z__v10_std_ms",
        "gefs00z__u850_mean_ms", "gefs00z__v850_mean_ms", "gefs00z__u850_std_ms", "gefs00z__v850_std_ms",
    ]]
    if len(hourly) != EXPECTED_DAYS * 24:
        raise RuntimeError("hourly row count mismatch")
    if hourly["forecast_kst_dtm"].min() != "2022-01-01 01:00:00" or hourly["forecast_kst_dtm"].max() != "2024-01-01 00:00:00":
        raise RuntimeError("hourly Stage1 boundary mismatch")
    numeric = [column for column in hourly.columns if column.startswith("gefs00z__")]
    if not all(pd.to_numeric(hourly[column], errors="coerce").notna().all() for column in numeric):
        raise RuntimeError("nonfinite hourly GEFS component")
    return hourly


def quarantine_partial(partial: Path, reason: str) -> Path | None:
    if not partial.exists():
        return None
    quarantine = PROJECT_ROOT / "artifacts" / "quarantine"
    quarantine.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = quarantine / f"noaa_gefs_operational_spread_00z_original_v2_stage1_partial_{stamp}_{os.getpid()}"
    os.replace(partial, target)
    (target / "failure.txt").write_text(reason + "\n", encoding="utf-8")
    return target


def bounded_download_results(
    plans: Iterable[RangePlan],
    *,
    workers: int,
    retries: int,
) -> Iterable[tuple[RangePlan, bytes, dict[str, Any]]]:
    """Yield results with at most 2*workers futures and fail-closed cancellation."""
    iterator = iter(plans)
    cancel_event = threading.Event()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gefs-stage1")
    pending: set[concurrent.futures.Future[tuple[RangePlan, bytes, dict[str, Any]]]] = set()
    try:
        for _ in range(2 * workers):
            try:
                plan = next(iterator)
            except StopIteration:
                break
            pending.add(executor.submit(download_one, plan, retries, cancel_event))
        while pending:
            done, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
            failed = next(
                (
                    future
                    for future in done
                    if future.cancelled() or future.exception() is not None
                ),
                None,
            )
            if failed is not None:
                cancel_event.set()
                for queued in pending:
                    queued.cancel()
                executor.shutdown(wait=True, cancel_futures=True)
                if failed.cancelled():
                    raise RuntimeError("bounded range future cancelled unexpectedly")
                failed.result()
            for future in done:
                result = future.result()
                yield result
                if not cancel_event.is_set():
                    try:
                        plan = next(iterator)
                    except StopIteration:
                        pass
                    else:
                        pending.add(executor.submit(download_one, plan, retries, cancel_event))
    finally:
        cancel_event.set()
        for queued in pending:
            queued.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def execute(plans: list[RangePlan], pack_sizes: dict[str, int], workers: int, retries: int) -> None:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"refusing to overwrite canonical: {OUTPUT_ROOT}")
    if STAGE2_ROOT.exists():
        raise RuntimeError("Stage2 source namespace exists before Stage1 promotion")
    existing_quarantine = list((PROJECT_ROOT / "artifacts/quarantine").glob("noaa_gefs_operational_spread_00z_original_v2_stage1_partial_*")) if (PROJECT_ROOT / "artifacts/quarantine").exists() else []
    if SOURCE_ATTEMPT_TOMBSTONE.exists() or existing_quarantine:
        raise RuntimeError("Stage1 source single-attempt tombstone/quarantine already exists; retry forbidden")
    partial = OUTPUT_PARENT / f"_stage1_partial_{os.getpid()}"
    if partial.exists():
        raise FileExistsError(partial)
    attempt_token = hashlib.sha256(f"{os.getpid()}:{time.time_ns()}".encode("ascii")).hexdigest()
    write_json_exclusive(SOURCE_ATTEMPT_TOMBSTONE, {
        "schema_version": 1,
        "attempt_id": "noaa_gefs_operational_spread_00z_original_v2_stage1_single_attempt",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "SINGLE_ATTEMPT_CONSUMED_ON_EXCLUSIVE_CREATE_NO_RETRY",
        "pid": os.getpid(), "token": attempt_token,
        "launch_lock": identity(LAUNCH_LOCK), "launch_lock_sidecar": identity(LAUNCH_LOCK_SHA),
        "planned_ranges": EXPECTED_RANGES, "planned_bytes": EXPECTED_BYTES,
        "meteorological_value_bytes_before_tombstone": 0,
        "labels_models_predictions_scores": 0, "2024_operating_values": 0, "2025_requests_or_values": 0,
    })
    try:
        partial.mkdir(parents=True)
        for relpath, size in pack_sizes.items():
            path = partial / relpath
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as handle:
                handle.truncate(size)
        pack_handles = {name: (partial / name).open("r+b", buffering=0) for name in pack_sizes}
        rows: list[dict[str, Any]] = []
        started = time.monotonic()
        try:
            for completed, (plan, payload, result) in enumerate(
                bounded_download_results(plans, workers=workers, retries=retries), 1
            ):
                handle = pack_handles[plan.pack_path]
                handle.seek(plan.pack_offset)
                written = handle.write(payload)
                if written != len(payload):
                    raise RuntimeError("short pack write")
                row = asdict(plan)
                row.update(result)
                rows.append(row)
                if completed % 250 == 0 or completed == len(plans):
                    elapsed = time.monotonic() - started
                    rate = completed / elapsed if elapsed else 0.0
                    eta = (len(plans) - completed) / rate if rate else 0.0
                    print(json.dumps({
                        "event": "stage1_range_progress", "completed": completed, "total": len(plans),
                        "bytes": sum(int(item["byte_count"]) for item in rows),
                        "rate_ranges_s": round(rate, 2), "eta_seconds": round(eta, 1),
                    }, sort_keys=True), flush=True)
        finally:
            for handle in pack_handles.values():
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()
        rows.sort(key=lambda row: (row["operating_date"], row["product_id"], int(row["lead"]), row["level_id"]))
        if len(rows) != EXPECTED_RANGES or sum(int(row["byte_count"]) for row in rows) != EXPECTED_BYTES:
            raise RuntimeError("completed response closure mismatch")

        manifest_fields = list(asdict(plans[0]).keys()) + [
            "response_sha256", "response_content_range", "response_content_length", "response_etag",
            "response_last_modified_utc", "attempts", "u_value", "v_value", "u_metadata", "v_metadata",
        ]
        response_manifest = partial / "range_response_manifest.csv.gz"
        deterministic_csv_gz(response_manifest, rows, manifest_fields)
        native_columns = [
            "operating_date", "source_date", "product_id", "statistic", "lead", "level_id",
            "u_value", "v_value", "response_sha256", "pack_path", "pack_offset", "byte_count",
        ]
        native = pd.DataFrame(rows)[native_columns].sort_values(
            ["operating_date", "product_id", "lead", "level_id"]
        ).reset_index(drop=True)
        native_path = partial / "native_point_values.parquet"
        native.to_parquet(native_path, index=False, compression="zstd")
        hourly = build_hourly(native)
        hourly_path = partial / "hourly_components.parquet"
        hourly.to_parquet(hourly_path, index=False, compression="zstd")

        artifacts = [identity(path) for path in sorted((partial / "raw_packs").glob("*.grib2pack"))]
        artifacts.extend([identity(response_manifest), identity(native_path), identity(hourly_path)])
        # identities above are relative to project even while in the partial namespace; after rename,
        # replace only the deterministic namespace prefix in the stored path.
        partial_prefix = relative(partial)
        canonical_prefix = relative(OUTPUT_ROOT)
        for item in artifacts:
            item["path"] = item["path"].replace(partial_prefix, canonical_prefix, 1)
        summary = {
            "schema_version": 1,
            "extraction_id": "noaa_gefs_operational_spread_00z_original_v2_stage1",
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "status": "PASS_STAGE1_SOURCE_EXTRACTION",
            "scope": {
                "operating_days": [FIRST_DAY.isoformat(), LAST_DAY.isoformat()], "days": EXPECTED_DAYS,
                "objects": EXPECTED_OBJECTS, "ranges": EXPECTED_RANGES, "raw_bytes": EXPECTED_BYTES,
                "native_rows": len(native), "hourly_rows": len(hourly),
                "final_target_boundary": "2024-01-01 00:00:00 Asia/Seoul",
            },
            "request_accounting": {
                "stage1_grib_range_gets": EXPECTED_RANGES, "meteorological_grib_value_bytes": EXPECTED_BYTES,
                "operating_2024_grib_range_gets_or_values": 0, "2025_requests_or_values": 0,
                "label_reads": 0, "fits": 0, "predictions": 0, "scores": 0, "csvs": 0,
            },
            "source_closure": [identity(BASE_PREREG), identity(PREREG), identity(PREREG_V3), identity(PREREG_V4), identity(PREREG_V5), identity(INCIDENT), identity(EXECUTION_PROTOCOL), identity(INDEPENDENT_REVIEW), identity(Path(__file__)), identity(LAUNCH_LOCK), identity(LAUNCH_LOCK_SHA), identity(SOURCE_ATTEMPT_TOMBSTONE), identity(METADATA), identity(INDEX_ZIP), identity(AVAILABILITY)],
            "artifacts": artifacts,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        summary_path = partial / "extraction_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary_identity = identity(summary_path)
        summary_identity["path"] = summary_identity["path"].replace(partial_prefix, canonical_prefix, 1)
        manifest = {
            "schema_version": 1,
            "manifest_id": "noaa_gefs_operational_spread_00z_original_v2_stage1_manifest",
            "created_utc": summary["created_utc"],
            "status": summary["status"],
            "canonical_root": canonical_prefix,
            "source_closure": summary["source_closure"],
            "artifacts": artifacts + [summary_identity],
            "nonmutation": {"stage2_namespace_created": False, "label_model_metric_csv_writes": 0},
        }
        manifest_path = partial / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        OUTPUT_PARENT.mkdir(parents=True, exist_ok=True)
        os.replace(partial, OUTPUT_ROOT)
        print(json.dumps({
            "event": "stage1_source_complete", "canonical": relative(OUTPUT_ROOT),
            "manifest_sha256": sha256_file(OUTPUT_ROOT / "manifest.json"),
            "summary_sha256": sha256_file(OUTPUT_ROOT / "extraction_summary.json"),
        }, sort_keys=True), flush=True)
    except BaseException as exc:
        target = quarantine_partial(partial, f"{type(exc).__name__}: {exc}")
        failure_path = PROJECT_ROOT / "artifacts/incidents" / f"noaa_gefs_operational_spread_00z_original_v2_stage1_failure_{attempt_token}.json"
        write_json_exclusive(failure_path, {
            "schema_version": 1, "incident_id": failure_path.stem,
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "status": "TERMINAL_SOURCE_ATTEMPT_FAILURE_NO_RETRY",
            "attempt_tombstone": identity(SOURCE_ATTEMPT_TOMBSTONE),
            "error": f"{type(exc).__name__}: {exc}",
            "quarantine": relative(target) if target else None,
            "alternate_source_cycle_lead_or_retry_allowed": False,
        })
        print(json.dumps({"event": "stage1_source_failed", "error": f"{type(exc).__name__}: {exc}", "quarantine": str(target) if target else None}, sort_keys=True), file=sys.stderr, flush=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-launch-lock", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--retries", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.freeze_launch_lock == args.execute:
        raise ValueError("choose exactly one of --freeze-launch-lock or --execute")
    if not 1 <= args.workers <= 32 or not 1 <= args.retries <= 8:
        raise ValueError("workers must be 1..32 and retries 1..8")
    plans, pack_sizes = build_stage1_plan()
    if args.freeze_launch_lock:
        lock = create_launch_lock(plans, pack_sizes)
        print(json.dumps({
            "event": "launch_lock_frozen", "path": relative(LAUNCH_LOCK),
            "bytes": LAUNCH_LOCK.stat().st_size, "sha256": sha256_file(LAUNCH_LOCK),
            "plan_sha256": lock["plan_sha256"], "ranges": len(plans), "raw_bytes": sum(pack_sizes.values()),
            "operating_2024_values": 0, "labels": 0,
        }, sort_keys=True))
        return 0
    lock = verify_launch_lock(plans, pack_sizes)
    execute(plans, pack_sizes, args.workers, args.retries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

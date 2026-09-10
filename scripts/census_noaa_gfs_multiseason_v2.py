#!/usr/bin/env python
"""Exact target-free NOAA GFS index/publication census for frozen v2 dates."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.noaa_gfs_provenance import (  # noqa: E402
    S3_HTTPS_ROOT,
    conservative_run_init_utc,
    cutoff_utc,
    find_unique_record,
    forecast_hour,
    object_key,
    object_url,
    operating_valid_times_utc,
    parse_grib_index,
    record_byte_range,
)


ROOT_DEFAULT = REPO / "artifacts" / "baram2026_ncei_scada_longrun_20260810_v2"
PLAN_SHA256 = "a3b483b08c1a22cd63fb08b98f83303e6a1466eec5ff8176be13d39d4db0a1a6"
AMENDMENT_SHA256 = "abbc6668b9be4bf3c0db79b7a0e648657851372ec14de8bb377bd01a938b7a2c"
PREREG_MANIFEST_V2_SHA256 = "4123fca18ad4ae28022373cd59ebec00bda7395a7292ae092d496dc50dadc12c"
USER_AGENT = "baram2026-noaa-gfs-multiseason-census/2.0"
S3_NAMESPACE = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path, root: Path | None = None) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix() if root else str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def write_text_exclusive(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def write_csv_exclusive(
    path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_parquet_exclusive(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(path)
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False, engine="pyarrow")


def write_progress_checkpoint(root: Path, name: str, payload: dict[str, Any]) -> None:
    """Persist a monotonic audit checkpoint without replacing an older one."""

    path = root / "census" / "progress" / f"{name}.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("checkpoint_name") != name:
            raise RuntimeError(f"invalid existing checkpoint: {path}")
        return
    write_json_exclusive(
        path,
        {
            "checkpoint_name": name,
            "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            **payload,
        },
    )


def fetch(
    url: str,
    *,
    range_start: int | None = None,
    max_bytes: int,
    attempts: int = 4,
) -> tuple[bytes, dict[str, str], int, str]:
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    if range_start is not None:
        headers["Range"] = f"bytes={range_start}-"
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                response_headers = {
                    str(key).lower(): str(value).strip()
                    for key, value in response.headers.items()
                }
                content_length = response_headers.get("content-length")
                if content_length is not None and int(content_length) > max_bytes:
                    raise RuntimeError(
                        f"response exceeds pre-read cap: {content_length} > {max_bytes}"
                    )
                payload = response.read(max_bytes + 1)
                if len(payload) > max_bytes:
                    raise RuntimeError(f"response exceeds read cap: {len(payload)} > {max_bytes}")
                retrieved_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                return payload, response_headers, int(response.status), retrieved_at
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
    raise RuntimeError(f"failed GET {url}: {last_error}") from last_error


def parse_list_xml(payload: bytes) -> list[dict[str, Any]]:
    root = ET.fromstring(payload)
    if root.findtext("s3:IsTruncated", "false", S3_NAMESPACE).lower() != "false":
        raise RuntimeError("ListObjectsV2 response was truncated")
    rows = []
    for item in root.findall("s3:Contents", S3_NAMESPACE):
        rows.append(
            {
                "key": urllib.parse.unquote(item.findtext("s3:Key", "", S3_NAMESPACE)),
                "last_modified_utc": item.findtext("s3:LastModified", "", S3_NAMESPACE),
                "etag": item.findtext("s3:ETag", "", S3_NAMESPACE).strip('"'),
                "size_bytes": int(item.findtext("s3:Size", "0", S3_NAMESPACE)),
            }
        )
    return rows


def list_url_for_run(run: datetime) -> str:
    prefix = f"gfs.{run:%Y%m%d}/{run:%H}/atmos/gfs.t{run:%H}z.pgrb2.0p25.f"
    query = urllib.parse.urlencode(
        {"list-type": "2", "prefix": prefix, "max-keys": "1000", "encoding-type": "url"}
    )
    return f"{S3_HTTPS_ROOT}/?{query}"


def compact_initialization(run_init_utc: str) -> str:
    parsed = datetime.fromisoformat(run_init_utc.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("run initialization is not timezone-aware")
    return parsed.astimezone(timezone.utc).strftime("%Y%m%d%H")


def get_or_reuse_list(root: Path, operating_day: date) -> dict[str, Any]:
    run = conservative_run_init_utc(operating_day)
    stem = f"operating_{operating_day.isoformat()}__run_{run:%Y%m%d%H}"
    xml_path = root / "census" / "list_xml" / f"{stem}.xml"
    meta_path = root / "census" / "list_xml" / f"{stem}.meta.json"
    url = list_url_for_run(run)
    if xml_path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            meta["url"] != url
            or meta["payload"]["size_bytes"] != xml_path.stat().st_size
            or meta["payload"]["sha256"] != sha256_file(xml_path)
        ):
            raise RuntimeError(f"cached ListObjectsV2 evidence mismatch: {xml_path}")
        return {"meta": meta, "rows": parse_list_xml(xml_path.read_bytes()), "network": False}
    if xml_path.exists() or meta_path.exists():
        raise RuntimeError(f"incomplete cached ListObjectsV2 evidence: {stem}")
    payload, headers, status, retrieved_at = fetch(url, max_bytes=1_000_000)
    if status != 200:
        raise RuntimeError(f"ListObjectsV2 HTTP status {status}")
    rows = parse_list_xml(payload)
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    with xml_path.open("xb") as stream:
        stream.write(payload)
    meta = {
        "url": url,
        "http_status": status,
        "retrieved_at": retrieved_at,
        "response_headers": headers,
        "payload": identity(xml_path, root),
    }
    write_json_exclusive(meta_path, meta)
    return {"meta": meta, "rows": rows, "network": True}


def get_or_resume_idx(
    root: Path, record: dict[str, Any]
) -> dict[str, Any]:
    key = record["idx_key"]
    run_date = record["run_init_utc"][:10].replace("-", "")
    fh = int(record["forecast_hour"])
    path = root / "census" / "index" / f"gfs.{run_date}" / "12" / f"f{fh:03d}.idx"
    sidecar = path.with_suffix(".idx.meta.json")
    expected_size = int(record["idx_size_bytes"])
    expected_etag = str(record["idx_etag"])
    url = object_url(key)
    if path.is_file() and sidecar.is_file():
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        if (
            path.stat().st_size != expected_size
            or meta["sha256"] != sha256_file(path)
            or meta["size_bytes"] != expected_size
            or meta["etag"] != expected_etag
            or meta["key"] != key
        ):
            raise RuntimeError(f"cached index mismatch: {path}")
        return {"path": path, "meta": meta, "network": False, "network_bytes": 0}
    if path.exists() or sidecar.exists():
        raise RuntimeError(f"incomplete finalized index state: {path}")
    part = path.with_suffix(".idx.part")
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = part.stat().st_size if part.exists() else 0
    if existing > expected_size:
        raise RuntimeError(f"partial index exceeds expected size: {part}")
    if existing == expected_size and existing > 0:
        network_bytes = 0
        retrieved_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        headers: dict[str, str] = {}
        status = 0
    else:
        remaining = expected_size - existing
        payload, headers, status, retrieved_at = fetch(
            url,
            range_start=existing if existing else None,
            max_bytes=remaining,
        )
        expected_status = 206 if existing else 200
        if status != expected_status or len(payload) != remaining:
            raise RuntimeError(
                f"index resume mismatch for {key}: status {status}, bytes {len(payload)}, expected {expected_status}/{remaining}"
            )
        with part.open("ab" if existing else "xb") as stream:
            stream.write(payload)
        network_bytes = len(payload)
    if part.stat().st_size != expected_size:
        raise RuntimeError(f"index final size mismatch: {part}")
    if len(expected_etag) == 32 and "-" not in expected_etag:
        if md5_file(part) != expected_etag:
            raise RuntimeError(f"index ETag/MD5 mismatch: {key}")
    sha = sha256_file(part)
    os.replace(part, path)
    meta = {
        "key": key,
        "url": url,
        "retrieved_at": retrieved_at,
        "http_status": status,
        "response_headers": headers,
        "size_bytes": expected_size,
        "etag": expected_etag,
        "sha256": sha,
        "resumed_from_bytes": existing,
    }
    write_json_exclusive(sidecar, meta)
    return {"path": path, "meta": meta, "network": network_bytes > 0, "network_bytes": network_bytes}


def main_records(plan: dict[str, Any], list_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_day = {
        result["operating_day"]: {row["key"]: row for row in result["rows"]}
        for result in list_results
    }
    records = []
    for day_text in plan["operating_days_kst"]:
        day = date.fromisoformat(day_text)
        run = conservative_run_init_utc(day)
        cutoff = cutoff_utc(day)
        valid_times = operating_valid_times_utc(day)
        metadata = by_day[day_text]
        for valid in valid_times:
            fh = forecast_hour(run, valid)
            key = object_key(run, fh)
            idx_key = f"{key}.idx"
            full_hits = [metadata[k] for k in metadata if k == key]
            idx_hits = [metadata[k] for k in metadata if k == idx_key]
            if len(full_hits) != 1 or len(idx_hits) != 1:
                raise RuntimeError(
                    f"exact ListObjectsV2 key multiplicity failed for {key}: {len(full_hits)}/{len(idx_hits)}"
                )
            full = full_hits[0]
            idx = idx_hits[0]
            publication = datetime.fromisoformat(full["last_modified_utc"].replace("Z", "+00:00"))
            margin = int((cutoff - publication).total_seconds())
            records.append(
                {
                    "target_operating_day_kst": day_text,
                    "run_init_utc": run.isoformat().replace("+00:00", "Z"),
                    "forecast_hour": fh,
                    "valid_time_utc": valid.isoformat().replace("+00:00", "Z"),
                    "cutoff_utc": cutoff.isoformat().replace("+00:00", "Z"),
                    "object_key": key,
                    "object_size_bytes": full["size_bytes"],
                    "object_etag": full["etag"],
                    "publication_last_modified_utc": full["last_modified_utc"],
                    "cutoff_margin_seconds": margin,
                    "idx_key": idx_key,
                    "idx_size_bytes": idx["size_bytes"],
                    "idx_etag": idx["etag"],
                    "idx_last_modified_utc": idx["last_modified_utc"],
                    "status": "PUBLICATION_VERIFIED" if margin >= 0 else "AFTER_CUTOFF",
                }
            )
    return records


def run(root: Path) -> dict[str, Any]:
    root = root.resolve()
    plan_path = root / "prereg" / "target_free_multiseason_sampling_plan_v1.json"
    amendment_path = root / "prereg" / "target_free_multiseason_amendment_v2.json"
    prereg_manifest = root / "manifest_preregister_v2.json"
    if sha256_file(plan_path) != PLAN_SHA256 or sha256_file(amendment_path) != AMENDMENT_SHA256:
        raise RuntimeError("frozen preregistration identity changed")
    if sha256_file(prereg_manifest) != PREREG_MANIFEST_V2_SHA256:
        raise RuntimeError("frozen preregistration manifest identity changed")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    outputs = {
        "objects": root / "census" / "object_census.parquet",
        "ranges": root / "census" / "field_range_census.parquet",
        "field_summary": root / "census" / "FIELD_CENSUS_SUMMARY.csv",
        "day_summary": root / "census" / "DAY_CENSUS_SUMMARY.csv",
        "access": root / "census" / "CENSUS_ACCESS_LEDGER.json",
        "plan": root / "census" / "RAW_DOWNLOAD_PLAN_EXACT.json",
        "summary": root / "census" / "CENSUS_SUMMARY.md",
        "manifest": root / "manifest_census_v1.json",
    }
    conflicts = [str(path) for path in outputs.values() if path.exists()]
    if conflicts:
        raise FileExistsError(f"no-overwrite census output preflight failed: {conflicts}")

    list_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(get_or_reuse_list, root, date.fromisoformat(day)): day
            for day in plan["operating_days_kst"]
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            result["operating_day"] = futures[future]
            list_results.append(result)
    list_results.sort(key=lambda row: row["operating_day"])
    write_progress_checkpoint(
        root,
        "list_runs_000048",
        {
            "phase": "LISTOBJECTSV2",
            "completed_runs": 48,
            "network_requests_this_invocation": sum(int(row["network"]) for row in list_results),
            "raw_range_requests": 0,
        },
    )
    records = main_records(plan, list_results)
    if len(records) != 1_152 or any(row["status"] != "PUBLICATION_VERIFIED" for row in records):
        raise RuntimeError("selected object publication census failed")
    expected_idx_bytes = sum(int(row["idx_size_bytes"]) for row in records)
    max_idx = int(plan["download_contract"]["max_index_census_transfer_bytes"])
    if expected_idx_bytes > max_idx:
        raise RuntimeError(f"index census cap exceeded before index GET: {expected_idx_bytes} > {max_idx}")

    idx_results: dict[str, dict[str, Any]] = {}
    completed_idx_keys: list[str] = []
    checkpoint_network_requests = 0
    checkpoint_network_bytes = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(get_or_resume_idx, root, record): record["object_key"] for record in records}
        for future in concurrent.futures.as_completed(futures):
            key = futures[future]
            result = future.result()
            idx_results[key] = result
            completed_idx_keys.append(key)
            checkpoint_network_requests += int(result["network"])
            checkpoint_network_bytes += int(result["network_bytes"])
            completed = len(completed_idx_keys)
            if completed % 100 == 0 or completed == 1_152:
                ordered = sorted(completed_idx_keys)
                key_digest = hashlib.sha256(
                    ("\n".join(ordered) + "\n").encode("utf-8")
                ).hexdigest()
                write_progress_checkpoint(
                    root,
                    f"idx_files_{completed:06d}",
                    {
                        "phase": "INDEX_DOWNLOAD",
                        "completed_files": completed,
                        "completed_key_set_sha256": key_digest,
                        "network_requests_this_invocation_so_far": checkpoint_network_requests,
                        "network_bytes_this_invocation_so_far": checkpoint_network_bytes,
                        "raw_range_requests": 0,
                    },
                )
    if len(idx_results) != 1_152:
        raise RuntimeError("index result count mismatch")

    selectors = [
        (row["variable"], row["level"], family["family"])
        for family in plan["candidate_families"]
        for row in family["selectors"]
    ]
    range_rows: list[dict[str, Any]] = []
    for record in records:
        result = idx_results[record["object_key"]]
        index_path: Path = result["path"]
        parsed = parse_grib_index(index_path.read_text(encoding="utf-8"))
        expected_init = compact_initialization(record["run_init_utc"])
        for variable, level, family in selectors:
            index = find_unique_record(parsed, variable, level)
            selected = parsed[index]
            if selected.initialization != expected_init:
                raise RuntimeError(f"index initialization mismatch: {record['object_key']}")
            if f"{record['forecast_hour']} hour fcst" not in selected.forecast_descriptor:
                raise RuntimeError(f"index forecast descriptor mismatch: {selected}")
            message_range = record_byte_range(parsed, index, int(record["object_size_bytes"]))
            range_rows.append(
                {
                    **record,
                    "source_archive": "NOAA_NODD_S3",
                    "archive_product": "gfs.t12z.pgrb2.0p25.fFFF",
                    "retrieval_url_or_request_id": object_url(record["object_key"]),
                    "official_metadata": "ListObjectsV2 exact Key/LastModified/ETag/Size",
                    "publication_evidence_type": "S3_LISTOBJECTSV2_EXACT_KEY_LASTMODIFIED_ETAG_SIZE",
                    "publication_evidence_reference": f"census/list_xml/operating_{record['target_operating_day_kst']}__run_{expected_init}.xml",
                    "family": family,
                    "variable": variable,
                    "level": level,
                    "grib_record_number": selected.record_number,
                    "forecast_descriptor": selected.forecast_descriptor,
                    "range_start": message_range.start,
                    "range_end": message_range.end,
                    "range_bytes": message_range.length,
                    "idx_relative_path": index_path.relative_to(root).as_posix(),
                    "idx_sha256": result["meta"]["sha256"],
                    "status": "CENSUS_VERIFIED",
                }
            )
    if len(range_rows) != 1_152 * len(selectors):
        raise RuntimeError("field range census row count mismatch")

    free_disk = shutil.disk_usage(root).free
    raw_bytes = sum(int(row["range_bytes"]) for row in range_rows)
    raw_requests = len(range_rows)
    list_network_requests = sum(int(row["network"]) for row in list_results)
    idx_network_requests = sum(int(row["network"]) for row in idx_results.values())
    idx_network_bytes = sum(int(row["network_bytes"]) for row in idx_results.values())
    total_requests_if_launched = 48 + 1_152 + raw_requests
    contract = plan["download_contract"]
    caps = {
        "raw_bytes_pass": raw_bytes <= int(contract["max_raw_payload_bytes"]),
        "requests_pass": total_requests_if_launched <= int(contract["max_total_http_requests"]),
        "free_disk_reserve_pass": free_disk - raw_bytes
        >= int(contract["minimum_free_disk_after_reserved_raw_bytes"]),
    }
    authorization = all(caps.values()) and all(
        row["status"] == "CENSUS_VERIFIED" for row in range_rows
    )

    write_parquet_exclusive(outputs["objects"], records)
    write_parquet_exclusive(outputs["ranges"], range_rows)
    field_summary = []
    for variable, level, family in selectors:
        subset = [row for row in range_rows if row["variable"] == variable and row["level"] == level]
        field_summary.append(
            {
                "family": family,
                "variable": variable,
                "level": level,
                "expected_occurrences": 1_152,
                "observed_unique_occurrences": len(subset),
                "presence_fraction": len(subset) / 1_152,
                "range_bytes_total": sum(int(row["range_bytes"]) for row in subset),
                "range_bytes_min": min(int(row["range_bytes"]) for row in subset),
                "range_bytes_max": max(int(row["range_bytes"]) for row in subset),
                "status": "PASS" if len(subset) == 1_152 else "FAIL",
            }
        )
    write_csv_exclusive(outputs["field_summary"], field_summary, list(field_summary[0]))
    day_summary = []
    for day in plan["operating_days_kst"]:
        obj = [row for row in records if row["target_operating_day_kst"] == day]
        rng = [row for row in range_rows if row["target_operating_day_kst"] == day]
        day_summary.append(
            {
                "target_operating_day_kst": day,
                "object_count": len(obj),
                "field_range_count": len(rng),
                "raw_range_bytes": sum(int(row["range_bytes"]) for row in rng),
                "minimum_cutoff_margin_seconds": min(int(row["cutoff_margin_seconds"]) for row in obj),
                "all_publication_verified": all(row["status"] == "PUBLICATION_VERIFIED" for row in obj),
                "status": "PASS" if len(obj) == 24 and len(rng) == 216 else "FAIL",
            }
        )
    write_csv_exclusive(outputs["day_summary"], day_summary, list(day_summary[0]))
    access = {
        "listobjectsv2": {
            "selected_runs": 48,
            "network_requests_this_run": list_network_requests,
            "cached_reuses": 48 - list_network_requests,
        },
        "index": {
            "selected_files": 1_152,
            "expected_bytes": expected_idx_bytes,
            "network_requests_this_run": idx_network_requests,
            "network_bytes_this_run": idx_network_bytes,
            "cached_reuses": 1_152 - idx_network_requests,
        },
        "raw_range": {"network_requests": 0, "downloaded_bytes": 0},
        "labels_read": False,
        "2024_arrays_read": False,
        "2025_arrays_read": False,
        "models_fit": 0,
        "submission_csv_created": False,
    }
    write_json_exclusive(outputs["access"], access)
    exact_plan = {
        "artifact_type": "EXACT_RAW_RANGE_DOWNLOAD_PLAN",
        "parent_preregistration": identity(prereg_manifest, root),
        "object_rows": 1_152,
        "field_range_rows": len(range_rows),
        "candidate_families": sorted({row["family"] for row in range_rows}),
        "exact_index_bytes": expected_idx_bytes,
        "exact_raw_range_bytes": raw_bytes,
        "raw_range_requests": raw_requests,
        "census_requests": 48 + 1_152,
        "total_requests_census_plus_raw": total_requests_if_launched,
        "free_disk_at_lock_bytes": free_disk,
        "estimated_free_disk_after_raw_bytes": free_disk - raw_bytes,
        "caps": caps,
        "all_1_152_objects_publication_verified": all(
            row["status"] == "PUBLICATION_VERIFIED" for row in records
        ),
        "all_10_368_field_ranges_census_verified": all(
            row["status"] == "CENSUS_VERIFIED" for row in range_rows
        ),
        "raw_launch_protocol_eligible": authorization,
        "raw_launch_executed": False,
        "requires_separate_durable_runner_lock_and_report_before_launch": True,
        "labels_read": False,
    }
    write_json_exclusive(outputs["plan"], exact_plan)
    write_text_exclusive(
        outputs["summary"],
        f"""# Exact NOAA GFS v2 field census

All 1,152 frozen D-2 12Z objects have exact S3 ListObjectsV2 metadata and publication Last-Modified before their cutoff. Every one of the nine locked selectors occurs exactly once in every index, producing 10,368 verified message ranges.

- exact index bytes: {expected_idx_bytes:,}
- exact raw range bytes: {raw_bytes:,}
- raw range requests: {raw_requests:,}
- total census + raw requests: {total_requests_if_launched:,}
- free disk at lock: {free_disk:,}
- projected free disk after raw payload: {free_disk - raw_bytes:,}
- protocol eligibility before runner lock: **{authorization}**

No raw GRIB range, external value, label, 2024/2025 array, model or submission CSV was read in this census. Raw launch still requires a separate runner-code/hash lock and a prelaunch report.
""",
    )
    source = Path(__file__).resolve()
    test = REPO / "tests" / "test_noaa_gfs_multiseason_census_v2.py"
    manifest = {
        "artifact_type": "NOAA_GFS_MULTISEASON_CENSUS_MANIFEST",
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "parent_preregister_manifest": identity(prereg_manifest, root),
        "object_census": identity(outputs["objects"], root),
        "field_range_census": identity(outputs["ranges"], root),
        "field_summary": identity(outputs["field_summary"], root),
        "day_summary": identity(outputs["day_summary"], root),
        "access_ledger": identity(outputs["access"], root),
        "raw_download_plan": identity(outputs["plan"], root),
        "summary": identity(outputs["summary"], root),
        "progress_checkpoints": [
            identity(path, root)
            for path in sorted((root / "census" / "progress").glob("*.json"))
        ],
        "reproduction_code": identity(source),
        "test_code": identity(test),
        "raw_network_requests": 0,
        "raw_downloaded_bytes": 0,
        "labels_read": False,
        "models_fit": 0,
        "submission_csv_created": False,
    }
    write_json_exclusive(outputs["manifest"], manifest)
    return {
        "manifest": identity(outputs["manifest"], root),
        "objects": len(records),
        "field_ranges": len(range_rows),
        "exact_idx_bytes": expected_idx_bytes,
        "exact_raw_bytes": raw_bytes,
        "total_requests": total_requests_if_launched,
        "raw_launch_protocol_eligible": authorization,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(run(args.root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

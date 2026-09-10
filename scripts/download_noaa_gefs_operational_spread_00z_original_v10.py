#!/usr/bin/env python3
"""Fresh v10 NOAA GEFS Stage1 extractor with a cold-safe message decoder.

The v7 full-source attempt remains terminal.  This separately authorized
source-only experiment preserves the exact frozen plan and changes only the
process-wide cold initialization of ``codes_new_from_message``: exactly one
ordinary first worker fully decodes its own payload before waiting workers may
decode concurrently.  No sample or separate-request warm-up is permitted.

There is no label, model, metric, operating-2024, or 2025 reader in this module.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import download_noaa_gefs_operational_spread_00z_original_v2 as legacy  # noqa: E402


RangePlan = legacy.RangePlan

V1_PREREG = legacy.BASE_PREREG
V2_PREREG = legacy.PREREG
V3_PREREG = legacy.PREREG_V3
V4_PREREG = legacy.PREREG_V4
V5_PREREG = legacy.PREREG_V5
V6_PREREG = PROJECT_ROOT / "configs/noaa_gefs_operational_spread_00z_paired_increment_preregister_v6.json"
V7_PREREG = PROJECT_ROOT / "configs/noaa_gefs_operational_spread_00z_paired_increment_preregister_v7.json"
V10_PREREG = PROJECT_ROOT / "configs/noaa_gefs_operational_spread_00z_paired_increment_preregister_v10.json"
V8_PREREG = PROJECT_ROOT / "configs/noaa_gefs_operational_spread_00z_paired_increment_preregister_v8.json"
V9_PREREG = PROJECT_ROOT / "configs/noaa_gefs_operational_spread_00z_paired_increment_preregister_v9.json"
EXECUTION_PROTOCOL = legacy.EXECUTION_PROTOCOL
LABEL_INCIDENT = legacy.INCIDENT
V2_FAILURE_SEAL = PROJECT_ROOT / "artifacts/incidents/noaa_gefs_operational_spread_00z_original_v2_stage1_protocol_failure_seal_v1.json"
V2_FAILURE_INCIDENT = PROJECT_ROOT / "artifacts/incidents/noaa_gefs_operational_spread_00z_original_v2_stage1_failure_694c6fdfa6f95e363b7f4001f34e3862b83f16ee42db8fcaa7384db52963086f.json"
V2_TOMBSTONE = PROJECT_ROOT / "artifacts/audits/noaa_gefs_operational_spread_00z_original_v2_stage1_single_attempt.json"
V2_LAUNCH_LOCK = PROJECT_ROOT / "artifacts/audits/noaa_gefs_operational_spread_00z_original_v2_stage1_launch_lock.json"
V7_SYNTHETIC_INCIDENT = PROJECT_ROOT / "artifacts/incidents/noaa_gefs_operational_spread_00z_v7_synthetic_reference_cleanup_v1.json"
V7_FAILURE_SEAL = PROJECT_ROOT / "artifacts/incidents/noaa_gefs_operational_spread_00z_original_v7_stage1_protocol_failure_seal_v1.json"
V10_AUTHORIZATION = PROJECT_ROOT / "artifacts/incidents/noaa_gefs_operational_spread_00z_original_v10_user_authorized_source_boundary_v1.json"

OUTPUT_PARENT = PROJECT_ROOT / "artifacts/external/noaa_gefs_operational_spread_00z_original_v10"
PREFLIGHT_OUTPUT = PROJECT_ROOT / "artifacts/external/noaa_gefs_operational_spread_00z_original_v7/preflight_one_range"
OUTPUT_ROOT = OUTPUT_PARENT / "stage1_source_through_operating_2023"
STAGE2_ROOT = OUTPUT_PARENT / "stage2_source_operating_2024"
PREFLIGHT_LOCK = PROJECT_ROOT / "artifacts/audits/noaa_gefs_operational_spread_00z_original_v7_preflight_lock.json"
PREFLIGHT_LOCK_SHA = PREFLIGHT_LOCK.with_suffix(".json.sha256")
PREFLIGHT_TOMBSTONE = PROJECT_ROOT / "artifacts/audits/noaa_gefs_operational_spread_00z_original_v7_preflight_single_attempt.json"
LAUNCH_LOCK = PROJECT_ROOT / "artifacts/audits/noaa_gefs_operational_spread_00z_original_v10_stage1_launch_lock.json"
LAUNCH_LOCK_SHA = LAUNCH_LOCK.with_suffix(".json.sha256")
SOURCE_TOMBSTONE = PROJECT_ROOT / "artifacts/audits/noaa_gefs_operational_spread_00z_original_v10_stage1_single_attempt.json"
INDEPENDENT_REVIEW = PROJECT_ROOT / "artifacts/audits/noaa_gefs_operational_spread_00z_original_v10_prelaunch_independent_review_v1.json"
POST_PREFLIGHT_REVIEW = PROJECT_ROOT / "artifacts/audits/noaa_gefs_operational_spread_00z_original_v7_preflight_postrun_independent_review_v1.json"
RUNNER = PROJECT_ROOT / "scripts/run_noaa_gefs_operational_spread_00z_paired_increment_v10.py"
LEGACY_MODEL_RUNNER = PROJECT_ROOT / "scripts/run_noaa_gefs_operational_spread_00z_paired_increment_v2.py"
FOCUSED_TEST = PROJECT_ROOT / "tests/test_noaa_gefs_operational_spread_00z_paired_increment_v10.py"
POSTRUN_TEST = PROJECT_ROOT / "tests/test_noaa_gefs_operational_spread_00z_paired_increment_v10_postrun.py"
REGRESSION_SCRIPT = PROJECT_ROOT / "scripts/audit_noaa_gefs_operational_spread_00z_v10_decoder_gate.py"
REGRESSION_REPORT = PROJECT_ROOT / "artifacts/audits/noaa_gefs_operational_spread_00z_v10_decoder_gate_regression_v1.json"

EXPECTED_V6_SHA = "243b3c14b9bfdfa5e46eccd767727559aadab14b4aff1d9be9df4f3a66c1255c"
EXPECTED_V10_PREREG_SHA = "b154793821dcebd9e859605006cbecf139c3753f4bab79693958bba90092e70d"
EXPECTED_V7_FAILURE_SEAL_SHA = "797eb8188f4f8726f1941efdfd194877a77c0018c1bd5a8ee22f958f60bdc7ba"
EXPECTED_V10_AUTHORIZATION_SHA = "0ef28e4d641b6a11291d974cbcd329a7d25ba3be0a47c46737e7a855b0aa0d68"
EXPECTED_V2_FAILURE_SEAL_SHA = "89d114289de2a7d3c6793c243955330cdb6285ea028ded9275ab258086592a88"
EXPECTED_V2_FAILURE_INCIDENT_SHA = "617fbc8b04891080f93e3e182a3ac11b8ce0844459f93622e609d9b3acddf676"
EXPECTED_V2_TOMBSTONE_SHA = "7522ee27037f19d4570ab6e06fef4cc4445285aac4fe882bade12460aea6d8ef"
EXPECTED_V2_LAUNCH_LOCK_SHA = "4d5796f11df09f30d1af34787a44988b16ea8d9285062bcb54899d2f2177f98b"
EXPECTED_PLAN_SHA = "501ccfeb34c1cd5e057b76a195315f6da3ffd49632b368e4ab578d51afdf8ff1"

# ecCodes' embedded-definition parser is not cold-thread-safe in this Windows
# runtime.  The event is published only after one ordinary worker has fully
# decoded and released both messages from its own freshly downloaded payload.
_DECODER_COLD_LOCK = threading.Lock()
_DECODER_READY = threading.Event()
_DECODER_FAILURE: str | None = None
_DECODER_FIRST_SUCCESS_COUNT = 0
PREFLIGHT_FIELDS = {
    "operating_date": "2022-01-01",
    "source_date": "2021-12-30",
    "product_id": "pressure_0p50_ens_mean",
    "lead": 39,
    "level_id": "10m",
    "data_key": "gefs.20211230/00/atmos/pgrb2ap5/geavg.t00z.pgrb2a.0p50.f039",
    "byte_start": 10_686_111,
    "byte_end": 11_151_080,
    "byte_count": 464_970,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    return legacy.sha256_file(path)


def identity(path: Path) -> dict[str, Any]:
    return legacy.identity(path)


def relative(path: Path) -> str:
    return legacy.relative(path)


def _require(path: Path, sha256: str) -> None:
    legacy.require_file(path, sha256)


def _lineage_files() -> tuple[Path, ...]:
    return (
        V1_PREREG,
        V2_PREREG,
        V3_PREREG,
        V4_PREREG,
        V5_PREREG,
        V6_PREREG,
        V7_PREREG,
        V8_PREREG,
        V9_PREREG,
        V10_PREREG,
        EXECUTION_PROTOCOL,
        LABEL_INCIDENT,
        V2_FAILURE_SEAL,
        V2_FAILURE_INCIDENT,
        V2_TOMBSTONE,
        V2_LAUNCH_LOCK,
        V7_SYNTHETIC_INCIDENT,
        V7_FAILURE_SEAL,
        V10_AUTHORIZATION,
    )


def verify_v10_lineage() -> Mapping[str, Any]:
    _require(V6_PREREG, EXPECTED_V6_SHA)
    _require(V10_PREREG, EXPECTED_V10_PREREG_SHA)
    _require(V7_FAILURE_SEAL, EXPECTED_V7_FAILURE_SEAL_SHA)
    _require(V10_AUTHORIZATION, EXPECTED_V10_AUTHORIZATION_SHA)
    _require(V2_FAILURE_SEAL, EXPECTED_V2_FAILURE_SEAL_SHA)
    _require(V2_FAILURE_INCIDENT, EXPECTED_V2_FAILURE_INCIDENT_SHA)
    _require(V2_TOMBSTONE, EXPECTED_V2_TOMBSTONE_SHA)
    _require(V2_LAUNCH_LOCK, EXPECTED_V2_LAUNCH_LOCK_SHA)
    if not V7_PREREG.is_file():
        raise FileNotFoundError("v7 executable head is absent")
    if not V8_PREREG.is_file():
        raise FileNotFoundError("v8 source-only test amendment is absent")
    if not V9_PREREG.is_file():
        raise FileNotFoundError("v9 source-closure amendment is absent")
    v6 = json.loads(V6_PREREG.read_text(encoding="utf-8"))
    v7 = json.loads(V7_PREREG.read_text(encoding="utf-8"))
    v10 = json.loads(V10_PREREG.read_text(encoding="utf-8"))
    v8 = json.loads(V8_PREREG.read_text(encoding="utf-8"))
    v9 = json.loads(V9_PREREG.read_text(encoding="utf-8"))
    if v6.get("status") != "FROZEN_USER_AUTHORIZED_FRESH_ENGINEERING_EXPERIMENT_BEFORE_ANY_V7_NETWORK_VALUE_LABEL_FIT_PREDICTION_OR_SCORE":
        raise RuntimeError("v6 authorization head is not frozen")
    if v7.get("status") != "FROZEN_FRESH_V7_EXECUTABLE_SOURCE_ONLY_BEFORE_INDEPENDENT_REVIEW_OR_ANY_V7_NETWORK_VALUE_LABEL_FIT_PREDICTION_OR_SCORE":
        raise RuntimeError("v7 executable head is not frozen")
    if v8.get("status") != "FROZEN_V8_REFERENCE_CLEANUP_AMENDMENT_BEFORE_INDEPENDENT_REVIEW_OR_ANY_V7_NETWORK_VALUE_LABEL_FIT_PREDICTION_OR_SCORE":
        raise RuntimeError("v8 source-only amendment is not frozen")
    if v8.get("supersedes", {}).get("sha256") != legacy.sha256_file(V7_PREREG):
        raise RuntimeError("v8 does not bind exact v7")
    if v9.get("status") != "FROZEN_V9_RECURSIVE_SOURCE_CLOSURE_AMENDMENT_BEFORE_INDEPENDENT_REVIEW_OR_ANY_V7_NETWORK_VALUE_LABEL_FIT_PREDICTION_OR_SCORE":
        raise RuntimeError("v9 source-closure amendment is not frozen")
    if v9.get("supersedes", {}).get("sha256") != legacy.sha256_file(V8_PREREG):
        raise RuntimeError("v9 does not bind exact v8")
    if v10.get("status") != "FROZEN_V10_USER_AUTHORIZED_SOURCE_ONLY_COLD_DECODER_GATE_BEFORE_SOURCE_CHANGE_OR_ANY_V10_NETWORK_VALUE_LABEL_FIT_PREDICTION_SCORE":
        raise RuntimeError("v10 source-only head is not frozen")
    if v10.get("supersedes", {}).get("sha256") != legacy.sha256_file(V9_PREREG):
        raise RuntimeError("v10 does not bind exact v9")
    if v10.get("authorization_lineage", {}).get("sha256") != EXPECTED_V10_AUTHORIZATION_SHA:
        raise RuntimeError("v10 does not bind exact authorization lineage")
    if v10.get("immutable_scientific_inheritance", {}).get("blend_weight") != 0.25:
        raise RuntimeError("v10 scientific inheritance differs")
    return v10


def build_stage1_plan() -> tuple[list[RangePlan], dict[str, int]]:
    verify_v10_lineage()
    plans, pack_sizes = legacy.build_stage1_plan()
    plan_sha = legacy.sha256_bytes(
        ("\n".join(json.dumps(asdict(plan), sort_keys=True, separators=(",", ":")) for plan in plans) + "\n").encode("utf-8")
    )
    if plan_sha != EXPECTED_PLAN_SHA:
        raise RuntimeError("v10 Stage1 plan differs from frozen v2 plan")
    if STAGE2_ROOT.exists():
        raise RuntimeError("fresh conditional Stage2 namespace exists before Stage1 promotion")
    return plans, pack_sizes


def fixed_preflight_plan(plans: Sequence[RangePlan]) -> RangePlan:
    matches = [
        plan
        for plan in plans
        if all(getattr(plan, key) == value for key, value in PREFLIGHT_FIELDS.items())
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one fixed preflight plan, found {len(matches)}")
    return matches[0]


def _decode_handle(gid: Any, plan: RangePlan, expected_name: str) -> tuple[dict[str, Any], float]:
    import eccodes

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
    expected_level_type = "heightAboveGround" if plan.level_id == "10m" else "isobaricInhPa"
    expected_level = 10 if plan.level_id == "10m" else 850
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
    return item, value


def _validate_structural_lengths(messages: Sequence[bytes], plan: RangePlan) -> None:
    if [len(message) for message in messages] != [
        int(plan.v_line.split(":", 2)[1]) - int(plan.u_line.split(":", 2)[1]),
        plan.byte_end - int(plan.v_line.split(":", 2)[1]) + 1,
    ]:
        raise RuntimeError("GRIB structural message lengths differ from frozen index ranges")


def _decode_two_messages_unlocked(payload: bytes, plan: RangePlan) -> tuple[list[dict[str, Any]], list[float]]:
    """Decode one range after the caller has satisfied cold-init ordering."""
    import eccodes

    messages = legacy.split_grib2_messages(payload)
    _validate_structural_lengths(messages, plan)
    expected_names = ("10u", "10v") if plan.level_id == "10m" else ("u", "v")
    metadata: list[dict[str, Any]] = []
    values: list[float] = []
    for message, expected_name in zip(messages, expected_names, strict=True):
        gid = eccodes.codes_new_from_message(message)
        if gid is None:
            raise RuntimeError("codes_new_from_message returned no handle")
        try:
            item, value = _decode_handle(gid, plan, expected_name)
            metadata.append(item)
            values.append(value)
        finally:
            eccodes.codes_release(gid)
    return metadata, values


def decode_two_messages(payload: bytes, plan: RangePlan) -> tuple[list[dict[str, Any]], list[float]]:
    """Decode with exactly one process-global first-worker initialization."""
    global _DECODER_FAILURE, _DECODER_FIRST_SUCCESS_COUNT

    if not _DECODER_READY.is_set():
        with _DECODER_COLD_LOCK:
            if not _DECODER_READY.is_set():
                try:
                    decoded = _decode_two_messages_unlocked(payload, plan)
                except BaseException as exc:
                    _DECODER_FAILURE = f"{type(exc).__name__}: {exc}"
                    _DECODER_READY.set()
                    raise
                _DECODER_FIRST_SUCCESS_COUNT += 1
                if _DECODER_FIRST_SUCCESS_COUNT != 1:
                    _DECODER_FAILURE = "decoder first-success count differs from one"
                    _DECODER_READY.set()
                    raise RuntimeError(_DECODER_FAILURE)
                _DECODER_READY.set()
                return decoded
    if _DECODER_FAILURE is not None:
        raise RuntimeError(f"process-global decoder initialization failed: {_DECODER_FAILURE}")
    return _decode_two_messages_unlocked(payload, plan)


def decoder_gate_state() -> dict[str, Any]:
    """Return non-meteorological synchronization state for tests and manifests."""
    return {
        "ready": _DECODER_READY.is_set(),
        "failure": _DECODER_FAILURE,
        "first_success_count": _DECODER_FIRST_SUCCESS_COUNT,
    }


def _decode_two_messages_real_file_once(path: Path, plan: RangePlan) -> tuple[list[dict[str, Any]], list[float]]:
    """Decode a real file in a disposable process (Windows FILE* is process-bound)."""
    import eccodes

    payload = path.read_bytes()
    messages = legacy.split_grib2_messages(payload)
    _validate_structural_lengths(messages, plan)
    expected_names = ("10u", "10v") if plan.level_id == "10m" else ("u", "v")
    metadata: list[dict[str, Any]] = []
    values: list[float] = []
    with path.open("rb") as handle:
        for expected_name in expected_names:
            gid = eccodes.codes_grib_new_from_file(handle)
            if gid is None:
                raise RuntimeError("real-file reference ended before two GRIB messages")
            try:
                item, value = _decode_handle(gid, plan, expected_name)
                metadata.append(item)
                values.append(value)
            finally:
                eccodes.codes_release(gid)
        third = eccodes.codes_grib_new_from_file(handle)
        if third is not None:
            eccodes.codes_release(third)
            raise RuntimeError("real-file reference contains more than two GRIB messages")
        if handle.tell() != len(payload):
            raise RuntimeError("real-file reference did not consume exact response")
    return metadata, values


def decode_two_messages_real_file_reference(payload: bytes, plan: RangePlan) -> tuple[list[dict[str, Any]], list[float]]:
    """Audit-only real-file decoder isolated in a short-lived child process."""
    with tempfile.TemporaryDirectory(prefix="gefs-v10-reference-") as directory:
        root = Path(directory)
        payload_path = root / "payload.grib2"
        plan_path = root / "plan.json"
        output_path = root / "decoded.json"
        payload_path.write_bytes(payload)
        plan_path.write_text(json.dumps(asdict(plan), sort_keys=True), encoding="utf-8")
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                str(Path(__file__).resolve()),
                "--reference-decode-file",
                str(payload_path),
                "--reference-plan-json",
                str(plan_path),
                "--reference-output-json",
                str(output_path),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"real-file reference child failed: {completed.stderr.strip()}")
        decoded = json.loads(output_path.read_text(encoding="utf-8"))
        return decoded["metadata"], [float(value) for value in decoded["values"]]


def execute_reference_decode_file(payload_path: Path, plan_path: Path, output_path: Path) -> None:
    plan = RangePlan(**json.loads(plan_path.read_text(encoding="utf-8")))
    metadata, values = _decode_two_messages_real_file_once(payload_path, plan)
    output_path.write_text(json.dumps({"metadata": metadata, "values": values}, sort_keys=True), encoding="utf-8")


def download_one(plan: RangePlan, retries: int, cancel_event: threading.Event) -> tuple[RangePlan, bytes, dict[str, Any]]:
    if cancel_event.is_set():
        raise RuntimeError("cancelled before identical range GET")
    payload, headers, attempts = legacy.range_get(plan, retries, cancel_event)
    if cancel_event.is_set():
        raise RuntimeError("cancelled after range GET before decode/write")
    decoded, values = decode_two_messages(payload, plan)
    return plan, payload, {
        "response_sha256": legacy.sha256_bytes(payload),
        "response_content_range": headers["content-range"],
        "response_content_length": int(headers["content-length"]),
        "response_etag": headers["etag"].strip('"'),
        "response_last_modified_utc": legacy.parse_http_time(headers["last-modified"]),
        "attempts": attempts,
        "u_value": values[0],
        "v_value": values[1],
        "u_metadata": json.dumps(decoded[0], sort_keys=True, separators=(",", ":")),
        "v_metadata": json.dumps(decoded[1], sort_keys=True, separators=(",", ":")),
    }


def bounded_download_results(
    plans: Iterable[RangePlan], *, workers: int, retries: int
) -> Iterable[tuple[RangePlan, bytes, dict[str, Any]]]:
    iterator = iter(plans)
    cancel_event = threading.Event()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gefs-v10-stage1")
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
            failed = next((future for future in done if future.cancelled() or future.exception() is not None), None)
            if failed is not None:
                cancel_event.set()
                for queued in pending:
                    queued.cancel()
                executor.shutdown(wait=True, cancel_futures=True)
                if failed.cancelled():
                    raise RuntimeError("bounded range future cancelled unexpectedly")
                failed.result()
            for future in done:
                yield future.result()
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


def _review_identity() -> dict[str, Any]:
    if not INDEPENDENT_REVIEW.is_file():
        raise FileNotFoundError("independent v10 pre-network review is absent")
    review = json.loads(INDEPENDENT_REVIEW.read_text(encoding="utf-8"))
    if review.get("verdict") != "PASS" or review.get("v10_full_network_value_label_fit_score_access") != 0:
        raise RuntimeError("independent v10 review is not an exact zero-access PASS")
    return identity(INDEPENDENT_REVIEW)


def _post_preflight_review_identity(preflight: Mapping[str, Any]) -> dict[str, Any]:
    if not POST_PREFLIGHT_REVIEW.is_file():
        raise FileNotFoundError("independent v10 post-preflight review is absent")
    review = json.loads(POST_PREFLIGHT_REVIEW.read_text(encoding="utf-8"))
    if review.get("verdict") != "PASS" or review.get("preflight_network_requests") != 1:
        raise RuntimeError("independent post-preflight review did not record exact one-request PASS")
    if review.get("full_stage1_network_value_label_fit_score_access") != 0:
        raise RuntimeError("independent post-preflight review is not zero-access before full launch")
    if review.get("bound_preflight") != preflight:
        raise RuntimeError("independent post-preflight review binds different preflight bytes")
    return identity(POST_PREFLIGHT_REVIEW)


def _source_roots_and_imports() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    roots = [Path(__file__), RUNNER, LEGACY_MODEL_RUNNER, FOCUSED_TEST, POSTRUN_TEST, REGRESSION_SCRIPT]
    recursive = legacy.recursive_local_imports(roots)
    root_set = {path.resolve() for path in roots}
    return [identity(path) for path in roots], [identity(path) for path in recursive if path not in root_set]


def expected_preflight_semantics(plans: Sequence[RangePlan]) -> dict[str, Any]:
    plan = fixed_preflight_plan(plans)
    roots, imports = _source_roots_and_imports()
    return {
        "scope": {"requests": 1, "concurrency": 1, "retries": 1, "response_bytes": plan.byte_count},
        "fixed_range": asdict(plan),
        "decoder": {
            "authoritative_api": "eccodes.codes_new_from_message",
            "messages": 2,
            "real_file_reference_equality_required": True,
        },
        "lineage": [identity(path) for path in _lineage_files()],
        "independent_review": _review_identity(),
        "source_roots": roots,
        "recursive_local_imports": imports,
        "old_v2_partial_files_opened_hashed_copied_or_reused": 0,
        "label_fit_prediction_score_2024_2025_access": 0,
    }


def _write_sidecar(path: Path) -> None:
    legacy.write_text_exclusive(path.with_suffix(".json.sha256"), f"{sha256_file(path)}  {path.name}\n")


def _verify_sidecar(path: Path) -> None:
    sidecar = path.with_suffix(".json.sha256")
    expected = f"{sha256_file(path)}  {path.name}\n"
    if not sidecar.is_file() or sidecar.read_text(encoding="ascii") != expected:
        raise RuntimeError(f"SHA sidecar mismatch: {path}")


def freeze_preflight_lock(plans: Sequence[RangePlan]) -> dict[str, Any]:
    if PREFLIGHT_LOCK.exists() or PREFLIGHT_LOCK_SHA.exists():
        raise FileExistsError("preflight lock already exists")
    if PREFLIGHT_TOMBSTONE.exists() or PREFLIGHT_OUTPUT.exists():
        raise RuntimeError("preflight attempt/output already exists")
    payload = {
        "schema_version": 1,
        "lock_id": "noaa_gefs_operational_spread_00z_original_v10_preflight_lock",
        "created_utc": utc_now(),
        "status": "FROZEN_AFTER_INDEPENDENT_REVIEW_BEFORE_ONE_LIVE_RANGE_GET",
        "candidate_or_performance_information": 0,
        **expected_preflight_semantics(plans),
    }
    legacy.write_json_exclusive(PREFLIGHT_LOCK, payload)
    _write_sidecar(PREFLIGHT_LOCK)
    return payload


def verify_preflight_lock(plans: Sequence[RangePlan]) -> Mapping[str, Any]:
    if not PREFLIGHT_LOCK.is_file():
        raise FileNotFoundError("preflight lock absent")
    _verify_sidecar(PREFLIGHT_LOCK)
    existing = json.loads(PREFLIGHT_LOCK.read_text(encoding="utf-8"))
    if existing.get("status") != "FROZEN_AFTER_INDEPENDENT_REVIEW_BEFORE_ONE_LIVE_RANGE_GET":
        raise RuntimeError("preflight lock status differs")
    for key, value in expected_preflight_semantics(plans).items():
        if existing.get(key) != value:
            raise RuntimeError(f"preflight lock closure differs: {key}")
    return existing


def _quarantine(partial: Path, prefix: str, reason: str) -> Path | None:
    if not partial.exists():
        return None
    quarantine = PROJECT_ROOT / "artifacts/quarantine"
    quarantine.mkdir(parents=True, exist_ok=True)
    target = quarantine / f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{os.getpid()}_{uuid.uuid4().hex}"
    os.replace(partial, target)
    (target / "failure.txt").write_text(reason + "\n", encoding="utf-8")
    return target


def execute_preflight(plans: Sequence[RangePlan]) -> None:
    verify_preflight_lock(plans)
    if PREFLIGHT_OUTPUT.exists() or PREFLIGHT_TOMBSTONE.exists():
        raise RuntimeError("v10 preflight attempt already consumed")
    matching = list((PROJECT_ROOT / "artifacts/quarantine").glob("noaa_gefs_operational_spread_00z_original_v10_preflight_partial_*")) if (PROJECT_ROOT / "artifacts/quarantine").exists() else []
    if matching:
        raise RuntimeError("v10 preflight quarantine exists")
    token = hashlib.sha256(f"{os.getpid()}:{time.time_ns()}".encode("ascii")).hexdigest()
    legacy.write_json_exclusive(PREFLIGHT_TOMBSTONE, {
        "schema_version": 1,
        "attempt_id": "noaa_gefs_operational_spread_00z_original_v10_preflight_single_attempt",
        "created_utc": utc_now(),
        "status": "SINGLE_PREFLIGHT_ATTEMPT_CONSUMED",
        "pid": os.getpid(),
        "token": token,
        "planned_requests": 1,
        "planned_bytes": PREFLIGHT_FIELDS["byte_count"],
        "network_requests_before_tombstone": 0,
        "label_fit_prediction_score_2024_2025_access": 0,
    })
    partial = OUTPUT_PARENT / f"_preflight_partial_{os.getpid()}_{token[:12]}"
    try:
        partial.mkdir(parents=True)
        plan = fixed_preflight_plan(plans)
        cancel = threading.Event()
        payload, headers, attempts = legacy.range_get(plan, retries=1, cancel_event=cancel)
        if attempts != 1:
            raise RuntimeError("preflight issued other than exactly one request")
        memory_meta, memory_values = decode_two_messages(payload, plan)
        file_meta, file_values = decode_two_messages_real_file_reference(payload, plan)
        if memory_meta != file_meta or memory_values != file_values:
            raise RuntimeError("new message decoder differs from real-file reference")
        response_path = partial / "response.grib2"
        response_path.write_bytes(payload)
        result = {
            "schema_version": 1,
            "preflight_id": "noaa_gefs_operational_spread_00z_original_v10_preflight_one_range",
            "created_utc": utc_now(),
            "status": "PASS_ONE_RANGE_NEW_DECODER_EQUALS_REAL_FILE_REFERENCE",
            "request_accounting": {"requests": 1, "response_bytes": len(payload), "retries": 0},
            "fixed_range": asdict(plan),
            "response": {
                "sha256": legacy.sha256_bytes(payload),
                "content_range": headers["content-range"],
                "content_length": int(headers["content-length"]),
                "etag": headers["etag"].strip('"'),
                "last_modified_utc": legacy.parse_http_time(headers["last-modified"]),
            },
            "message_decoder": {"api": "eccodes.codes_new_from_message", "metadata": memory_meta, "values": memory_values},
            "real_file_reference": {"api": "eccodes.codes_grib_new_from_file", "metadata": file_meta, "values": file_values},
            "exact_metadata_and_value_equality": True,
            "full_extraction_payload_reuse_allowed": False,
            "label_fit_prediction_score_2024_2025_access": 0,
        }
        result_path = partial / "preflight_result.json"
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "manifest_id": "noaa_gefs_operational_spread_00z_original_v10_preflight_manifest",
            "created_utc": result["created_utc"],
            "status": result["status"],
            "source_closure": [identity(path) for path in _lineage_files()] + [identity(INDEPENDENT_REVIEW), identity(PREFLIGHT_LOCK), identity(PREFLIGHT_LOCK_SHA), identity(PREFLIGHT_TOMBSTONE), identity(Path(__file__))],
            "artifacts": [
                {"path": "response.grib2", "bytes": response_path.stat().st_size, "sha256": sha256_file(response_path)},
                {"path": "preflight_result.json", "bytes": result_path.stat().st_size, "sha256": sha256_file(result_path)},
            ],
            "payload_reuse_by_full_extraction": 0,
        }
        (partial / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        OUTPUT_PARENT.mkdir(parents=True, exist_ok=True)
        os.replace(partial, PREFLIGHT_OUTPUT)
        print(json.dumps({"event": "v10_preflight_complete", "manifest_sha256": sha256_file(PREFLIGHT_OUTPUT / "manifest.json"), "response_sha256": sha256_file(PREFLIGHT_OUTPUT / "response.grib2")}, sort_keys=True), flush=True)
    except BaseException as exc:
        target = _quarantine(partial, "noaa_gefs_operational_spread_00z_original_v10_preflight_partial", f"{type(exc).__name__}: {exc}")
        incident = PROJECT_ROOT / "artifacts/incidents" / f"noaa_gefs_operational_spread_00z_original_v10_preflight_failure_{token}.json"
        legacy.write_json_exclusive(incident, {
            "schema_version": 1,
            "incident_id": incident.stem,
            "created_utc": utc_now(),
            "status": "TERMINAL_V10_PREFLIGHT_FAILURE_NO_FULL_LAUNCH",
            "tombstone": identity(PREFLIGHT_TOMBSTONE),
            "error": f"{type(exc).__name__}: {exc}",
            "quarantine": relative(target) if target else None,
            "retry_or_alternate_range_decoder_source_allowed": False,
        })
        raise


def verify_preflight_canonical() -> dict[str, Any]:
    manifest_path = PREFLIGHT_OUTPUT / "manifest.json"
    result_path = PREFLIGHT_OUTPUT / "preflight_result.json"
    response_path = PREFLIGHT_OUTPUT / "response.grib2"
    if not all(path.is_file() for path in (manifest_path, result_path, response_path)):
        raise FileNotFoundError("sealed v7 preflight fixture is absent")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS_ONE_RANGE_NEW_DECODER_EQUALS_REAL_FILE_REFERENCE" or result.get("status") != manifest.get("status"):
        raise RuntimeError("sealed v7 preflight fixture did not pass")
    if result.get("exact_metadata_and_value_equality") is not True or result.get("request_accounting") != {"requests": 1, "response_bytes": PREFLIGHT_FIELDS["byte_count"], "retries": 0}:
        raise RuntimeError("sealed v7 preflight equality/accounting differs")
    if response_path.stat().st_size != PREFLIGHT_FIELDS["byte_count"] or sha256_file(response_path) != result["response"]["sha256"]:
        raise RuntimeError("sealed v7 preflight response identity differs")
    return {"manifest": identity(manifest_path), "result": identity(result_path), "response": identity(response_path)}


def expected_launch_semantics(plans: Sequence[RangePlan], pack_sizes: Mapping[str, int]) -> dict[str, Any]:
    roots, imports = _source_roots_and_imports()
    preflight = verify_preflight_canonical()
    return {
        "scope": {
            "operating_days": [legacy.FIRST_DAY.isoformat(), legacy.LAST_DAY.isoformat()],
            "days": legacy.EXPECTED_DAYS,
            "objects": legacy.EXPECTED_OBJECTS,
            "ranges": legacy.EXPECTED_RANGES,
            "bytes": legacy.EXPECTED_BYTES,
            "pack_sizes": dict(pack_sizes),
            "operating_2024_values": 0,
            "2025_requests_or_values": 0,
            "labels_models_predictions_scores": 0,
            "workers": 24,
            "retries": 5,
        },
        "plan_sha256": EXPECTED_PLAN_SHA,
        "preflight": preflight,
        "post_preflight_independent_review": _post_preflight_review_identity(preflight),
        "preflight_payload_reuse": 0,
        "offline_decoder_gate_regression": identity(REGRESSION_REPORT),
        "lineage": [identity(path) for path in _lineage_files()],
        "independent_review": _review_identity(),
        "source_roots": roots,
        "recursive_local_imports": imports,
        "decoder": {"api": "eccodes.codes_new_from_message", "messages_per_range": 2, "cold_gate": "one process-global Lock plus ready/failure state; first ordinary worker fully decodes its own payload", "sample_warmup": 0, "python": sys.version.split()[0], "eccodes_python": __import__("eccodes").__version__, "eccodes_api": __import__("eccodes").codes_get_api_version()},
        "old_v2_partial_files_opened_hashed_copied_or_reused": 0,
        "terminal_v7_partial_files_opened_hashed_copied_or_reused": 0,
    }


def freeze_launch_lock(plans: Sequence[RangePlan], pack_sizes: Mapping[str, int]) -> dict[str, Any]:
    if LAUNCH_LOCK.exists() or LAUNCH_LOCK_SHA.exists():
        raise FileExistsError("v10 full launch lock already exists")
    if SOURCE_TOMBSTONE.exists() or OUTPUT_ROOT.exists():
        raise RuntimeError("v10 full source attempt/output already exists")
    matching = list((PROJECT_ROOT / "artifacts/quarantine").glob("noaa_gefs_operational_spread_00z_original_v10_stage1_partial_*")) if (PROJECT_ROOT / "artifacts/quarantine").exists() else []
    if matching:
        raise RuntimeError("v10 full source quarantine exists")
    payload = {
        "schema_version": 1,
        "lock_id": "noaa_gefs_operational_spread_00z_original_v10_stage1_launch_lock",
        "created_utc": utc_now(),
        "status": "FROZEN_V10_AFTER_OFFLINE_REGRESSION_AND_INDEPENDENT_REVIEW_BEFORE_FULL_STAGE1_RANGE_GET",
        "candidate_or_performance_information": 0,
        **expected_launch_semantics(plans, pack_sizes),
    }
    legacy.write_json_exclusive(LAUNCH_LOCK, payload)
    _write_sidecar(LAUNCH_LOCK)
    return payload


def verify_launch_lock(plans: Sequence[RangePlan], pack_sizes: Mapping[str, int]) -> Mapping[str, Any]:
    if not LAUNCH_LOCK.is_file():
        raise FileNotFoundError("v10 full launch lock absent")
    _verify_sidecar(LAUNCH_LOCK)
    existing = json.loads(LAUNCH_LOCK.read_text(encoding="utf-8"))
    if existing.get("status") != "FROZEN_V10_AFTER_OFFLINE_REGRESSION_AND_INDEPENDENT_REVIEW_BEFORE_FULL_STAGE1_RANGE_GET":
        raise RuntimeError("v10 full launch lock status differs")
    for key, value in expected_launch_semantics(plans, pack_sizes).items():
        if existing.get(key) != value:
            raise RuntimeError(f"v10 full launch closure differs: {key}")
    return existing


def execute(plans: list[RangePlan], pack_sizes: dict[str, int], workers: int, retries: int) -> None:
    verify_launch_lock(plans, pack_sizes)
    if (workers, retries) != (24, 5):
        raise RuntimeError("v10 full source execution requires frozen workers=24/retries=5")
    if decoder_gate_state() != {"ready": False, "failure": None, "first_success_count": 0}:
        raise RuntimeError("v10 full source process decoder gate is not cold before the first ordinary worker")
    if OUTPUT_ROOT.exists() or STAGE2_ROOT.exists():
        raise RuntimeError("v10 source canonical/Stage2 namespace already exists")
    matching = list((PROJECT_ROOT / "artifacts/quarantine").glob("noaa_gefs_operational_spread_00z_original_v10_stage1_partial_*")) if (PROJECT_ROOT / "artifacts/quarantine").exists() else []
    if SOURCE_TOMBSTONE.exists() or matching:
        raise RuntimeError("v10 source single attempt already consumed")
    partial = OUTPUT_PARENT / f"_stage1_partial_{os.getpid()}_{uuid.uuid4().hex}"
    token = hashlib.sha256(f"{os.getpid()}:{time.time_ns()}".encode("ascii")).hexdigest()
    legacy.write_json_exclusive(SOURCE_TOMBSTONE, {
        "schema_version": 1,
        "attempt_id": "noaa_gefs_operational_spread_00z_original_v10_stage1_single_attempt",
        "created_utc": utc_now(),
        "status": "SINGLE_FULL_SOURCE_ATTEMPT_CONSUMED_NO_RETRY",
        "pid": os.getpid(),
        "token": token,
        "launch_lock": identity(LAUNCH_LOCK),
        "launch_lock_sidecar": identity(LAUNCH_LOCK_SHA),
        "preflight_manifest": identity(PREFLIGHT_OUTPUT / "manifest.json"),
        "preflight_payload_reuse": 0,
        "planned_ranges": legacy.EXPECTED_RANGES,
        "planned_bytes": legacy.EXPECTED_BYTES,
        "labels_models_predictions_scores_2024_2025": 0,
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
            for completed, (plan, payload, result) in enumerate(bounded_download_results(plans, workers=workers, retries=retries), 1):
                handle = pack_handles[plan.pack_path]
                handle.seek(plan.pack_offset)
                if handle.write(payload) != len(payload):
                    raise RuntimeError("short pack write")
                row = asdict(plan)
                row.update(result)
                rows.append(row)
                if completed % 250 == 0 or completed == len(plans):
                    elapsed = time.monotonic() - started
                    rate = completed / elapsed if elapsed else 0.0
                    eta = (len(plans) - completed) / rate if rate else 0.0
                    print(json.dumps({"event": "v10_stage1_range_progress", "completed": completed, "total": len(plans), "bytes": sum(int(item["byte_count"]) for item in rows), "rate_ranges_s": round(rate, 2), "eta_seconds": round(eta, 1)}, sort_keys=True), flush=True)
        finally:
            for handle in pack_handles.values():
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()
        rows.sort(key=lambda row: (row["operating_date"], row["product_id"], int(row["lead"]), row["level_id"]))
        if len(rows) != legacy.EXPECTED_RANGES or sum(int(row["byte_count"]) for row in rows) != legacy.EXPECTED_BYTES:
            raise RuntimeError("v10 completed response closure mismatch")
        if decoder_gate_state() != {"ready": True, "failure": None, "first_success_count": 1}:
            raise RuntimeError("v10 decoder gate completion state differs")
        manifest_fields = list(asdict(plans[0]).keys()) + ["response_sha256", "response_content_range", "response_content_length", "response_etag", "response_last_modified_utc", "attempts", "u_value", "v_value", "u_metadata", "v_metadata"]
        response_manifest = partial / "range_response_manifest.csv.gz"
        legacy.deterministic_csv_gz(response_manifest, rows, manifest_fields)
        native_columns = ["operating_date", "source_date", "product_id", "statistic", "lead", "level_id", "u_value", "v_value", "response_sha256", "pack_path", "pack_offset", "byte_count"]
        native = pd.DataFrame(rows)[native_columns].sort_values(["operating_date", "product_id", "lead", "level_id"]).reset_index(drop=True)
        native_path = partial / "native_point_values.parquet"
        native.to_parquet(native_path, index=False, compression="zstd")
        hourly = legacy.build_hourly(native)
        hourly_path = partial / "hourly_components.parquet"
        hourly.to_parquet(hourly_path, index=False, compression="zstd")
        artifacts = [identity(path) for path in sorted((partial / "raw_packs").glob("*.grib2pack"))]
        artifacts.extend([identity(response_manifest), identity(native_path), identity(hourly_path)])
        partial_prefix = relative(partial)
        canonical_prefix = relative(OUTPUT_ROOT)
        for item in artifacts:
            item["path"] = item["path"].replace(partial_prefix, canonical_prefix, 1)
        summary = {
            "schema_version": 1,
            "extraction_id": "noaa_gefs_operational_spread_00z_original_v10_stage1",
            "created_utc": utc_now(),
            "status": "PASS_V10_STAGE1_SOURCE_EXTRACTION",
            "scope": {"operating_days": [legacy.FIRST_DAY.isoformat(), legacy.LAST_DAY.isoformat()], "days": legacy.EXPECTED_DAYS, "objects": legacy.EXPECTED_OBJECTS, "ranges": legacy.EXPECTED_RANGES, "raw_bytes": legacy.EXPECTED_BYTES, "native_rows": len(native), "hourly_rows": len(hourly), "final_target_boundary": "2024-01-01 00:00:00 Asia/Seoul"},
            "request_accounting": {"stage1_grib_range_gets": legacy.EXPECTED_RANGES, "meteorological_grib_value_bytes": legacy.EXPECTED_BYTES, "preflight_payload_reused": 0, "terminal_v7_partial_reused": 0, "operating_2024_grib_range_gets_or_values": 0, "2025_requests_or_values": 0, "label_reads": 0, "fits": 0, "predictions": 0, "scores": 0, "csvs": 0},
            "decoder_gate": decoder_gate_state(),
            "source_closure": [identity(path) for path in _lineage_files()] + [identity(INDEPENDENT_REVIEW), identity(POST_PREFLIGHT_REVIEW), identity(Path(__file__)), identity(PREFLIGHT_OUTPUT / "manifest.json"), identity(LAUNCH_LOCK), identity(LAUNCH_LOCK_SHA), identity(SOURCE_TOMBSTONE), identity(legacy.METADATA), identity(legacy.INDEX_ZIP), identity(legacy.AVAILABILITY)],
            "artifacts": artifacts,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        summary_path = partial / "extraction_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary_record = identity(summary_path)
        summary_record["path"] = summary_record["path"].replace(partial_prefix, canonical_prefix, 1)
        manifest = {"schema_version": 1, "manifest_id": "noaa_gefs_operational_spread_00z_original_v10_stage1_manifest", "created_utc": summary["created_utc"], "status": summary["status"], "canonical_root": canonical_prefix, "source_closure": summary["source_closure"], "artifacts": artifacts + [summary_record], "nonmutation": {"stage2_namespace_created": False, "label_model_metric_csv_writes": 0, "old_v2_partial_reused": 0, "terminal_v7_partial_reused": 0}}
        (partial / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        OUTPUT_PARENT.mkdir(parents=True, exist_ok=True)
        os.replace(partial, OUTPUT_ROOT)
        print(json.dumps({"event": "v10_stage1_source_complete", "canonical": relative(OUTPUT_ROOT), "manifest_sha256": sha256_file(OUTPUT_ROOT / "manifest.json"), "summary_sha256": sha256_file(OUTPUT_ROOT / "extraction_summary.json")}, sort_keys=True), flush=True)
    except BaseException as exc:
        target = _quarantine(partial, "noaa_gefs_operational_spread_00z_original_v10_stage1_partial", f"{type(exc).__name__}: {exc}")
        incident = PROJECT_ROOT / "artifacts/incidents" / f"noaa_gefs_operational_spread_00z_original_v10_stage1_failure_{token}.json"
        legacy.write_json_exclusive(incident, {"schema_version": 1, "incident_id": incident.stem, "created_utc": utc_now(), "status": "TERMINAL_V10_SOURCE_ATTEMPT_FAILURE_NO_RETRY", "attempt_tombstone": identity(SOURCE_TOMBSTONE), "error": f"{type(exc).__name__}: {exc}", "quarantine": relative(target) if target else None, "alternate_source_cycle_lead_decoder_or_retry_allowed": False})
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-audit", action="store_true")
    parser.add_argument("--freeze-launch-lock", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--reference-decode-file", type=Path)
    parser.add_argument("--reference-plan-json", type=Path)
    parser.add_argument("--reference-output-json", type=Path)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--retries", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    reference_args = (args.reference_decode_file, args.reference_plan_json, args.reference_output_json)
    if any(value is not None for value in reference_args):
        if not all(value is not None for value in reference_args):
            raise ValueError("all reference-decoder paths are required")
        if any((args.static_audit, args.freeze_launch_lock, args.execute)):
            raise ValueError("reference-decoder helper is exclusive")
        execute_reference_decode_file(*reference_args)
        return 0
    modes = [args.static_audit, args.freeze_launch_lock, args.execute]
    if sum(bool(mode) for mode in modes) != 1:
        raise ValueError("choose exactly one v10 mode")
    if (args.workers, args.retries) != (24, 5):
        raise ValueError("v10 freezes workers=24 and retries=5")
    plans, pack_sizes = build_stage1_plan()
    if args.static_audit:
        print(json.dumps({"status": "PASS_V10_STATIC_SOURCE_CONTRACT", "plan_sha256": EXPECTED_PLAN_SHA, "ranges": len(plans), "bytes": sum(pack_sizes.values()), "decoder": "eccodes.codes_new_from_message", "decoder_gate": "first-worker-own-plan-full-decode-lock-ready", "sample_warmup": 0, "live_preflight_requests": 0, "offline_fixture_only": True, "v7_partial_reuse": 0, "network_value_label_fit_score_access": 0}, sort_keys=True))
        return 0
    if args.freeze_launch_lock:
        lock = freeze_launch_lock(plans, pack_sizes)
        print(json.dumps({"event": "v10_launch_lock_frozen", "path": relative(LAUNCH_LOCK), "sha256": sha256_file(LAUNCH_LOCK), "ranges": lock["scope"]["ranges"], "bytes": lock["scope"]["bytes"]}, sort_keys=True))
        return 0
    execute(plans, pack_sizes, args.workers, args.retries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

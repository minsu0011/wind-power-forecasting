"""Credential-safe KMA D-1 11 KST WSD typ01 archive acquisition.

The API key is read from the Windows user environment and is never serialized.
Forecast timestamps are interval ends: operating_clock = timestamp - 1 hour.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import winreg
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
ENDPOINT = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-dfs_shrt_grd"
OUTPUT = PROJECT / "artifacts" / "final_submission_sprint_20260812" / "kma_d1_1100" / "wsd_typ01"
EXECUTION_CONFIG = PROJECT / "configs" / "kma_d1_1100_wsd_sprint_execution_v1.json"
NX = 149
NY = 253
TOKEN_COUNT = NX * NY
CELLS = ((94, 121), (94, 122))
ANCHOR_HOURS = (1, 4, 7, 10, 13, 16, 19, 22, 24)
PRE_GATE_YEARS = (2022, 2023, 2024)
PRE_GATE_CALLS = 9_864
POST_GATE_CALLS = 3_285
GRAND_CALLS = 13_149
BYTE_STOP = 4_700_000_000


@dataclass(frozen=True)
class RequestSpec:
    sequence: int
    operating_year: int
    forecast_kst: datetime
    issue_kst: datetime

    @property
    def tmef(self) -> str:
        return self.forecast_kst.strftime("%Y%m%d%H")

    @property
    def tmfc(self) -> str:
        return self.issue_kst.strftime("%Y%m%d%H")

    @property
    def stem(self) -> str:
        return f"{self.sequence:05d}__tmfc_{self.tmfc}__tmef_{self.tmef}__WSD"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def user_key() -> str:
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as handle:
        value, _ = winreg.QueryValueEx(handle, "KMA_APIHUB_AUTH_KEY")
    key = str(value).strip()
    if not key:
        raise RuntimeError("KMA APIHub user-scope credential is missing")
    return key


def canonical_json_bytes(value: Any, *, pretty: bool = True) -> bytes:
    if pretty:
        text = json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, indent=2) + "\n"
    else:
        text = json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return text.encode("ascii")


def write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    if path.read_bytes() != payload:
        raise RuntimeError(f"exclusive write verification failed for {path.name}")


def specs_for_year(year: int, *, sequence_start: int) -> list[RequestSpec]:
    result: list[RequestSpec] = []
    operating_day = datetime(year, 1, 1)
    stop = datetime(year + 1, 1, 1)
    while operating_day < stop:
        issue = (operating_day - timedelta(days=1)).replace(hour=11)
        for lead in ANCHOR_HOURS:
            target = operating_day + timedelta(hours=lead)
            result.append(RequestSpec(sequence_start + len(result), year, target, issue))
        operating_day += timedelta(days=1)
    expected = (366 if year % 4 == 0 else 365) * len(ANCHOR_HOURS)
    if len(result) != expected:
        raise AssertionError(f"unexpected operating-year hours for {year}: {len(result)}")
    return result


def all_specs() -> list[RequestSpec]:
    result: list[RequestSpec] = []
    for year in (*PRE_GATE_YEARS, 2025):
        result.extend(specs_for_year(year, sequence_start=len(result) + 1))
    if len(result) != GRAND_CALLS or len([x for x in result if x.operating_year in PRE_GATE_YEARS]) != PRE_GATE_CALLS:
        raise AssertionError("anchor request count mismatch")
    return result


def parse_grid(payload: bytes) -> tuple[np.ndarray, dict[str, Any]]:
    text = payload.decode("utf-8", errors="strict").strip(" \t\r\n")
    tokens = text.split(",")
    trailing_empty = bool(tokens and tokens[-1] == "")
    if trailing_empty:
        tokens.pop()
    if any(token.strip() == "" for token in tokens):
        raise ValueError("nonterminal empty WSD grid token")
    if len(tokens) != TOKEN_COUNT:
        raise ValueError(f"WSD grid token count {len(tokens)} != {TOKEN_COUNT}")
    normalized = ",".join(token.strip() for token in tokens)
    values = np.fromstring(normalized, dtype=np.float64, sep=",")
    if values.shape != (TOKEN_COUNT,) or not np.isfinite(values).all():
        raise ValueError("WSD grid float64 parse or finiteness failure")
    grid = values.reshape((NY, NX), order="C")
    selected = np.asarray([grid[y - 1, x - 1] for x, y in CELLS], dtype=np.float64)
    if not np.isfinite(selected).all() or np.any(selected < 0.0) or np.any(selected > 75.0):
        raise ValueError("selected KMA WSD cells lie outside [0,75] m/s")
    facts = {
        "numeric_token_count": int(values.size),
        "one_terminal_comma_present": trailing_empty,
        "sentinel_le_minus_99_count": int(np.sum(values <= -99.0)),
        "wsd_cell_94_121": float(selected[0]),
        "wsd_cell_94_122": float(selected[1]),
    }
    return selected, facts


def paths_for(spec: RequestSpec) -> tuple[Path, Path]:
    folder = OUTPUT / "raw_gzip" / str(spec.operating_year)
    return folder / f"{spec.stem}.txt.gz", folder / f"{spec.stem}.json"


def read_existing(spec: RequestSpec) -> dict[str, Any] | None:
    raw_path, meta_path = paths_for(spec)
    if not raw_path.exists() and not meta_path.exists():
        return None
    if meta_path.exists() and not raw_path.is_file():
        raise RuntimeError(f"partial pre-existing request record {spec.sequence}")
    if not raw_path.is_file():
        raise RuntimeError(f"invalid raw request record {spec.sequence}")
    compressed = raw_path.read_bytes()
    payload = gzip.decompress(compressed)
    _, facts = parse_grid(payload)
    if not meta_path.exists():
        metadata = make_metadata(
            spec,
            payload,
            compressed,
            facts,
            elapsed_seconds=None,
            retrieved_utc=None,
            recovered_from_raw_only=True,
        )
        write_exclusive(meta_path, canonical_json_bytes(metadata))
    else:
        metadata = json.loads(meta_path.read_text(encoding="ascii"))
    expected = {
        "sequence": spec.sequence,
        "operating_year": spec.operating_year,
        "forecast_kst": spec.forecast_kst.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
        "tmfc_kst": spec.tmfc,
        "tmef_kst": spec.tmef,
        "variable": "WSD",
        "endpoint_id": "KMA_APIHUB_TYP01_NPH_DFS_SHRT_GRD",
        "response_bytes": len(payload),
        "response_sha256": sha256_bytes(payload),
        "gzip_bytes": len(compressed),
        "gzip_sha256": sha256_bytes(compressed),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"existing request binding mismatch {spec.sequence}: {key}")
    for key in ("numeric_token_count", "sentinel_le_minus_99_count", "wsd_cell_94_121", "wsd_cell_94_122"):
        if facts[key] != metadata[key]:
            raise RuntimeError(f"existing decoded fact mismatch {spec.sequence}: {key}")
    return metadata


def make_metadata(
    spec: RequestSpec,
    payload: bytes,
    compressed: bytes,
    facts: dict[str, Any],
    *,
    elapsed_seconds: float | None,
    retrieved_utc: str | None,
    recovered_from_raw_only: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "sequence": spec.sequence,
        "operating_year": spec.operating_year,
        "forecast_kst": spec.forecast_kst.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
        "tmfc_kst": spec.tmfc,
        "tmef_kst": spec.tmef,
        "variable": "WSD",
        "endpoint_id": "KMA_APIHUB_TYP01_NPH_DFS_SHRT_GRD",
        "response_http_status": 200,
        "response_content_type": "text/plain",
        "response_bytes": len(payload),
        "response_sha256": sha256_bytes(payload),
        "gzip_bytes": len(compressed),
        "gzip_sha256": sha256_bytes(compressed),
        "elapsed_seconds": elapsed_seconds,
        "retrieved_utc": retrieved_utc,
        "recovered_from_raw_only": recovered_from_raw_only,
        "reused_existing": False,
        **facts,
    }


def acquire_one(spec: RequestSpec, key: str) -> dict[str, Any]:
    existing = read_existing(spec)
    if existing is not None:
        return {**existing, "reused_existing": True}
    started = time.perf_counter()
    try:
        query = urllib.parse.urlencode({"tmfc": spec.tmfc, "tmef": spec.tmef, "vars": "WSD", "authKey": key})
        request = urllib.request.Request(
            ENDPOINT + "?" + query,
            headers={"User-Agent": "baram2026-kma-wsd-sprint/2", "Accept-Encoding": "identity"},
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            status = int(response.status)
            content_type = response.headers.get_content_type().lower()
            final = urllib.parse.urlsplit(response.geturl())
            if final.scheme != "https" or final.hostname != "apihub.kma.go.kr":
                raise RuntimeError("sanitized KMA redirect target rejected")
            payload = response.read()
    except Exception:
        raise RuntimeError(f"sanitized KMA transport failure at sequence {spec.sequence}") from None
    elapsed = time.perf_counter() - started
    if status != 200:
        raise RuntimeError(f"sanitized KMA HTTP status {status} at sequence {spec.sequence}")
    if content_type != "text/plain":
        raise RuntimeError(f"sanitized KMA content type {content_type!r} at sequence {spec.sequence}")
    _, facts = parse_grid(payload)
    compressed = gzip.compress(payload, compresslevel=9, mtime=0)
    raw_path, meta_path = paths_for(spec)
    metadata = make_metadata(
        spec,
        payload,
        compressed,
        facts,
        elapsed_seconds=float(elapsed),
        retrieved_utc=datetime.utcnow().isoformat(timespec="microseconds") + "Z",
        recovered_from_raw_only=False,
    )
    write_exclusive(raw_path, compressed)
    write_exclusive(meta_path, canonical_json_bytes(metadata))
    if gzip.decompress(raw_path.read_bytes()) != payload:
        raise RuntimeError(f"gzip round-trip mismatch {spec.sequence}")
    return metadata


def verify_execution_config() -> dict[str, Any]:
    payload = json.loads(EXECUTION_CONFIG.read_text(encoding="ascii"))
    script = Path(__file__).resolve()
    if payload["script_identity"] != {
        "path": "scripts/download_kma_d1_1100_wsd_grid_sprint.py",
        "bytes": script.stat().st_size,
        "sha256": sha256_file(script),
    }:
        raise RuntimeError("execution config does not bind current downloader")
    if payload["first_100_projection_formula"] != "max(response_bytes_first_100)*13149":
        raise RuntimeError("execution config projection formula mismatch")
    return payload


def run_batch(specs: list[RequestSpec], *, workers: int) -> list[dict[str, Any]]:
    key = user_key()
    results: list[dict[str, Any]] = []
    for start in range(0, len(specs), workers):
        chunk = specs[start : start + workers]
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {pool.submit(acquire_one, spec, key): spec for spec in chunk}
            for future in concurrent.futures.as_completed(future_map):
                spec = future_map[future]
                result = future.result()
                results.append(result)
                if len(results) % 10 == 0 or len(results) == len(specs):
                    print(f"progress={len(results)}/{len(specs)} latest_sequence={spec.sequence}", flush=True)
        cumulative = sum(int(item["response_bytes"]) for item in results)
        if cumulative > BYTE_STOP:
            raise RuntimeError("cumulative 4.7GB response-byte stop reached")
    return sorted(results, key=lambda item: int(item["sequence"]))


def summarize(results: list[dict[str, Any]], *, mode: str) -> dict[str, Any]:
    sizes = [int(item["response_bytes"]) for item in results]
    elapsed = [float(item["elapsed_seconds"]) for item in results if not item.get("reused_existing")]
    return {
        "schema_version": 1,
        "mode": mode,
        "records": len(results),
        "sequence_min": min(int(item["sequence"]) for item in results),
        "sequence_max": max(int(item["sequence"]) for item in results),
        "response_bytes_sum": sum(sizes),
        "response_bytes_max": max(sizes),
        "projected_grand_bytes_by_max": max(sizes) * GRAND_CALLS,
        "projected_grand_calls": GRAND_CALLS,
        "byte_gate_limit": BYTE_STOP,
        "byte_gate_pass": max(sizes) * GRAND_CALLS <= BYTE_STOP,
        "mean_request_elapsed_seconds_new_only": float(np.mean(elapsed)) if elapsed else 0.0,
        "sentinel_count_min": min(int(item["sentinel_le_minus_99_count"]) for item in results),
        "sentinel_count_max": max(int(item["sentinel_le_minus_99_count"]) for item in results),
        "secret_persisted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("first100", "pregate", "year2025"), required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if not 1 <= args.workers <= 24:
        raise ValueError("workers must be in [1,24]")
    verify_execution_config()
    requests_all = all_specs()
    pregate_specs = [spec for spec in requests_all if spec.operating_year in PRE_GATE_YEARS]
    if args.mode == "first100":
        specs = pregate_specs[:100]
        summary_path = OUTPUT / "FIRST100_SUMMARY.json"
    elif args.mode == "pregate":
        first_path = OUTPUT / "FIRST100_SUMMARY.json"
        first = json.loads(first_path.read_text(encoding="ascii"))
        if first["records"] != 100 or not first["byte_gate_pass"]:
            raise RuntimeError("first100 gate is absent or failed")
        specs = pregate_specs
        summary_path = OUTPUT / "PREGATE_2022_2024_SUMMARY.json"
    else:
        gate_path = OUTPUT / "MODEL_GATE_PASS.json"
        gate = json.loads(gate_path.read_text(encoding="ascii"))
        if gate.get("all_gates_pass") is not True:
            raise RuntimeError("2025 KMA access remains blocked by model gates")
        specs = [spec for spec in requests_all if spec.operating_year == 2025]
        summary_path = OUTPUT / "YEAR2025_SUMMARY.json"
    results = run_batch(specs, workers=args.workers)
    summary = summarize(results, mode=args.mode)
    if args.mode == "first100" and not summary["byte_gate_pass"]:
        write_exclusive(summary_path, canonical_json_bytes(summary))
        raise RuntimeError("first100 projected byte gate failed")
    write_exclusive(summary_path, canonical_json_bytes(summary))
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

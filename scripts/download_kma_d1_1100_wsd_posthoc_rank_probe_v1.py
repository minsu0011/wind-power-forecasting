"""Acquire only the preregistered 2025 KMA WSD anchors for posthoc probe V1.

The 2022--2024 V8 raw and materialized artifacts are immutable inputs.  This
wrapper writes the 3,285 2025 responses only under a distinct posthoc root.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "scripts/download_kma_d1_1100_wsd_grid_sprint_recovery.py"
PREREG = PROJECT / "configs/kma_d1_1100_wsd_posthoc_rank_probe_v1_preregister.json"
EXECUTION_CONFIG = PROJECT / "configs/kma_d1_1100_wsd_posthoc_rank_probe_v1_acquisition_execution.json"
ROOT = PROJECT / "artifacts/final_submission_sprint_20260812/kma_d1_1100_posthoc_probe_v1"
WSD_ROOT = ROOT / "wsd_typ01"
PREGATE_SUMMARY = PROJECT / "artifacts/final_submission_sprint_20260812/kma_d1_1100_v8/wsd_typ01/PREGATE_2022_2024_SUMMARY_V8.json"
SUMMARY = WSD_ROOT / "YEAR2025_SUMMARY_POSTHOC_V1.json"
EXPECTED_RECORDS = 3_285
EXPECTED_SEQUENCE_MIN = 9_865
EXPECTED_SEQUENCE_MAX = 13_149
PRIOR_SUCCESSFUL_CALLS = 9_864
PRIOR_EXTRA_PHYSICAL_ATTEMPTS = 3
PRIOR_EXTRA_BYTES = 1_023_891
MAXIMUM_PHYSICAL_ATTEMPTS = 13_152
BYTE_STOP = 4_700_000_000


def _load_base():
    spec = importlib.util.spec_from_file_location("_kma_posthoc_probe_v1_acquisition_base", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen KMA acquisition source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = _load_base()
base.OUTPUT = WSD_ROOT


def identity(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(PROJECT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": base.sha256_file(path),
    }


def verify_execution_config() -> dict[str, Any]:
    payload = json.loads(EXECUTION_CONFIG.read_text(encoding="ascii"))
    if payload.get("schema_version") != 1 or payload.get("status") != "FROZEN_BEFORE_POSTHOC_2025_NETWORK_EXECUTION":
        raise RuntimeError("posthoc acquisition execution binding status/schema mismatch")
    if payload.get("script_identity") != identity(Path(__file__).resolve()):
        raise RuntimeError("posthoc acquisition script identity drift")
    for declared in payload.get("bound_inputs", []):
        path = PROJECT / str(declared.get("path"))
        if identity(path) != declared:
            raise RuntimeError(f"posthoc acquisition input drift: {path.name}")
    if payload.get("actual_argv") != [
        ".venv/Scripts/python.exe", "-B",
        "scripts/download_kma_d1_1100_wsd_posthoc_rank_probe_v1.py",
        "--mode", "year2025", "--workers", "1",
    ]:
        raise RuntimeError("posthoc acquisition argv binding mismatch")
    if payload.get("output_root") != WSD_ROOT.relative_to(PROJECT).as_posix():
        raise RuntimeError("posthoc acquisition output root mismatch")
    accounting = payload.get("physical_attempt_accounting", {})
    if accounting != {
        "prior_successful_calls": PRIOR_SUCCESSFUL_CALLS,
        "prior_extra_attempts": PRIOR_EXTRA_PHYSICAL_ATTEMPTS,
        "new_calls": EXPECTED_RECORDS,
        "maximum_physical_attempts": MAXIMUM_PHYSICAL_ATTEMPTS,
        "byte_stop": BYTE_STOP,
    }:
        raise RuntimeError("posthoc acquisition physical accounting mismatch")
    return payload


def prior_accounting() -> tuple[int, int]:
    summary = json.loads(PREGATE_SUMMARY.read_text(encoding="ascii"))
    if (
        summary.get("records") != PRIOR_SUCCESSFUL_CALLS
        or summary.get("sequence_min") != 1
        or summary.get("sequence_max") != PRIOR_SUCCESSFUL_CALLS
        or summary.get("byte_gate_pass") is not True
        or summary.get("prior_physical_attempts") != PRIOR_EXTRA_PHYSICAL_ATTEMPTS
        or summary.get("prior_physical_bytes_conservative") != PRIOR_EXTRA_BYTES
    ):
        raise RuntimeError("immutable pregate physical accounting mismatch")
    return int(summary["response_bytes_sum"]), int(summary["response_bytes_max"])


def run_batch(specs, *, workers: int):
    if workers != 1:
        raise ValueError("posthoc exact physical ordering requires --workers 1")
    prior_sum, _ = prior_accounting()
    key = base.user_key()
    results: list[dict[str, Any]] = []
    for spec in specs:
        result = base.acquire_one(spec, key)
        results.append(result)
        accounted = prior_sum + PRIOR_EXTRA_BYTES + sum(int(item["response_bytes"]) for item in results)
        if accounted > BYTE_STOP:
            raise RuntimeError("cumulative physical 4.7GB response-byte stop reached")
        if len(results) % 10 == 0 or len(results) == len(specs):
            print(f"progress={len(results)}/{len(specs)} latest_sequence={spec.sequence}", flush=True)
    return sorted(results, key=lambda item: int(item["sequence"]))


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    prior_sum, prior_max = prior_accounting()
    sizes = [int(item["response_bytes"]) for item in results]
    elapsed = [float(item["elapsed_seconds"]) for item in results if not item.get("reused_existing")]
    invalid = [int(item["sequence"]) for item in results if not bool(item.get("selected_cells_all_valid_0_75", True))]
    observed_max = max([prior_max, *sizes])
    projected = observed_max * MAXIMUM_PHYSICAL_ATTEMPTS
    return {
        "schema_version": 1,
        "experiment": "KMA_D1_1100_WSD_POSTHOC_HIGH_RISK_RANK_PROBE_V1",
        "mode": "year2025",
        "records": len(results),
        "sequence_min": min(int(item["sequence"]) for item in results),
        "sequence_max": max(int(item["sequence"]) for item in results),
        "response_bytes_sum": sum(sizes),
        "response_bytes_max": max(sizes),
        "prior_successful_calls": PRIOR_SUCCESSFUL_CALLS,
        "prior_successful_response_bytes": prior_sum,
        "prior_extra_physical_attempts": PRIOR_EXTRA_PHYSICAL_ATTEMPTS,
        "prior_extra_physical_bytes_conservative": PRIOR_EXTRA_BYTES,
        "maximum_physical_attempts": MAXIMUM_PHYSICAL_ATTEMPTS,
        "total_response_bytes_accounted": prior_sum + PRIOR_EXTRA_BYTES + sum(sizes),
        "projected_physical_bytes_by_observed_max": projected,
        "byte_gate_limit": BYTE_STOP,
        "byte_gate_pass": projected <= BYTE_STOP and prior_sum + PRIOR_EXTRA_BYTES + sum(sizes) <= BYTE_STOP,
        "mean_request_elapsed_seconds_new_only": float(base.np.mean(elapsed)) if elapsed else 0.0,
        "invalid_selected_anchor_count": len(invalid),
        "invalid_selected_anchor_sequences": invalid,
        "posthoc_high_risk": True,
        "independent_confirmation": False,
        "secret_persisted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("year2025",), required=False, default="year2025")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    verify_execution_config()
    prior_accounting()
    requests = [item for item in base.all_specs() if item.operating_year == 2025]
    if len(requests) != EXPECTED_RECORDS or requests[0].sequence != EXPECTED_SEQUENCE_MIN or requests[-1].sequence != EXPECTED_SEQUENCE_MAX:
        raise RuntimeError("posthoc 2025 request census mismatch")
    if args.verify_only:
        print(json.dumps({"verify_only": "PASS", "records": len(requests), "sequence_range": [requests[0].sequence, requests[-1].sequence]}, sort_keys=True))
        return
    if args.workers != 1:
        raise ValueError("posthoc exact physical ordering requires --workers 1")
    results = run_batch(requests, workers=1)
    summary = summarize(results)
    if summary["records"] != EXPECTED_RECORDS or summary["sequence_min"] != EXPECTED_SEQUENCE_MIN or summary["sequence_max"] != EXPECTED_SEQUENCE_MAX:
        raise RuntimeError("posthoc completed request census mismatch")
    base.write_exclusive(SUMMARY, base.canonical_json_bytes(summary))
    if not summary["byte_gate_pass"]:
        raise RuntimeError("posthoc physical byte gate failed")
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

"""V8/V9 availability-tolerant KMA acquisition with immutable raw reuse.

Sequences 1..5418 are read-only inputs from the terminal V1-V7 namespace.
Only sequences 5419 onward may be written, under the distinct V8 namespace.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
SOURCE_SCRIPT = PROJECT / "scripts" / "download_kma_d1_1100_wsd_grid_sprint_recovery.py"
OLD_ROOT = PROJECT / "artifacts/final_submission_sprint_20260812/kma_d1_1100/wsd_typ01"
V8_ROOT = PROJECT / "artifacts/final_submission_sprint_20260812/kma_d1_1100_v8/wsd_typ01"
EXECUTION_CONFIG = PROJECT / "configs/kma_d1_1100_wsd_sprint_execution_v8.json"
REUSE_LAST_SEQUENCE = 5418
PRIOR_ATTEMPTS = 3
PRIOR_BYTES = 1_023_891
PHYSICAL_CALLS_MAX = 13_152


def _load_base():
    spec = importlib.util.spec_from_file_location("_kma_recovery_v8_base", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen recovery downloader")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = _load_base()
base.OUTPUT = V8_ROOT
base.EXECUTION_CONFIG = EXECUTION_CONFIG
base.RECOVERY_UNPERSISTED_TRANSFER_COUNT = PRIOR_ATTEMPTS
base.RECOVERY_UNPERSISTED_RESPONSE_BYTES_KNOWN = PRIOR_BYTES


def identity(path: Path) -> dict[str, Any]:
    return {"path": path.relative_to(PROJECT).as_posix(), "bytes": path.stat().st_size, "sha256": base.sha256_file(path)}


def paths_for(spec):
    root = OLD_ROOT if spec.sequence <= REUSE_LAST_SEQUENCE else V8_ROOT
    folder = root / "raw_gzip" / str(spec.operating_year)
    return folder / f"{spec.stem}.txt.gz", folder / f"{spec.stem}.json"


base.paths_for = paths_for


def acquire_one(spec, key: str):
    existing = base.read_existing(spec)
    if existing is not None:
        return {**existing, "reused_existing": True, "source_namespace": "V1_V7_READ_ONLY" if spec.sequence <= REUSE_LAST_SEQUENCE else "V8"}
    if spec.sequence <= REUSE_LAST_SEQUENCE:
        raise RuntimeError(f"immutable reused record missing at sequence {spec.sequence}; refetch forbidden")
    result = _base_acquire_one(spec, key)
    return {**result, "source_namespace": "V8"}


_base_acquire_one = base.acquire_one
base.acquire_one = acquire_one


def verify_execution_config() -> dict[str, Any]:
    payload = json.loads(EXECUTION_CONFIG.read_text(encoding="ascii"))
    if payload.get("schema_version") != 8:
        raise RuntimeError("V8 acquisition execution config schema mismatch")
    expected = identity(Path(__file__).resolve())
    if payload.get("script_identity") != expected:
        raise RuntimeError("V8 execution config does not bind current downloader")
    for declared in payload.get("bound_inputs", []):
        path = PROJECT / str(declared.get("path"))
        if identity(path) != declared:
            raise RuntimeError(f"V8 bound identity mismatch: {path.name}")
    accounting = payload.get("physical_attempt_accounting", {})
    if accounting != {"prior_attempts": PRIOR_ATTEMPTS, "prior_bytes": PRIOR_BYTES, "maximum_physical_attempts": PHYSICAL_CALLS_MAX, "projected_bytes_at_observed_max": 4_488_738_144}:
        raise RuntimeError("V8 physical-attempt accounting mismatch")
    return payload


base.verify_execution_config = verify_execution_config


_base_summarize = base.summarize


def summarize(results, *, mode: str) -> dict[str, Any]:
    summary = _base_summarize(results, mode=mode)
    summary.update({
        "schema_version": 8,
        "experiment": "KMA_D1_1100_WSD_AVAILABILITY_TOLERANT_V8_V9",
        "immutable_reuse_sequence_range": [1, REUSE_LAST_SEQUENCE],
        "v8_new_sequence_min": REUSE_LAST_SEQUENCE + 1,
        "prior_physical_attempts": PRIOR_ATTEMPTS,
        "prior_physical_bytes_conservative": PRIOR_BYTES,
        "maximum_physical_attempts": PHYSICAL_CALLS_MAX,
        "projected_physical_bytes_by_observed_max": max(int(x["response_bytes"]) for x in results) * PHYSICAL_CALLS_MAX,
    })
    if summary["projected_physical_bytes_by_observed_max"] > base.BYTE_STOP:
        summary["byte_gate_pass"] = False
    return summary


base.summarize = summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("first100", "pregate", "year2025"), required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers != 1:
        raise ValueError("V8 exact physical ordering requires --workers 1")
    verify_execution_config()
    all_requests = base.all_specs()
    pregate = [x for x in all_requests if x.operating_year in base.PRE_GATE_YEARS]
    if args.mode == "first100":
        requests = pregate[:100]
        summary_path = V8_ROOT / "FIRST100_SUMMARY_V8.json"
    elif args.mode == "pregate":
        first = json.loads((V8_ROOT / "FIRST100_SUMMARY_V8.json").read_text(encoding="ascii"))
        if first.get("records") != 100 or first.get("byte_gate_pass") is not True:
            raise RuntimeError("V8 first100 gate absent or failed")
        requests = pregate
        summary_path = V8_ROOT / "PREGATE_2022_2024_SUMMARY_V8.json"
    else:
        gate = json.loads((V8_ROOT / "MODEL_GATE_PASS_V8.json").read_text(encoding="ascii"))
        if gate.get("all_gates_pass") is not True:
            raise RuntimeError("V8 2025 access remains model-gate blocked")
        requests = [x for x in all_requests if x.operating_year == 2025]
        summary_path = V8_ROOT / "YEAR2025_SUMMARY_V8.json"
    results = base.run_batch(requests, workers=1)
    summary = summarize(results, mode=args.mode)
    base.write_exclusive(summary_path, base.canonical_json_bytes(summary))
    if not summary["byte_gate_pass"]:
        raise RuntimeError("V8 physical byte projection gate failed")
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

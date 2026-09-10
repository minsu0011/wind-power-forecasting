"""Append-only recovery after posthoc acquisition attempt 1 was killed."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "scripts/download_kma_d1_1100_wsd_grid_sprint_recovery.py"
ROOT = PROJECT / "artifacts/final_submission_sprint_20260812/kma_d1_1100_posthoc_probe_v1/wsd_typ01"
PREREG = PROJECT / "configs/kma_d1_1100_wsd_posthoc_rank_probe_v1_preregister.json"
ATTEMPT1_CONFIG = PROJECT / "configs/kma_d1_1100_wsd_posthoc_rank_probe_v1_acquisition_execution.json"
INCIDENT = PROJECT / "configs/kma_d1_1100_wsd_posthoc_rank_probe_v1_acquisition_attempt1_incident.json"
EXECUTION_CONFIG = PROJECT / "configs/kma_d1_1100_wsd_posthoc_rank_probe_v1_acquisition_recovery_execution.json"
PREGATE_SUMMARY = PROJECT / "artifacts/final_submission_sprint_20260812/kma_d1_1100_v8/wsd_typ01/PREGATE_2022_2024_SUMMARY_V8.json"
SUMMARY = ROOT / "YEAR2025_SUMMARY_POSTHOC_V1_RECOVERY.json"
EXPECTED_RECORDS = 3285
SEQUENCE_MIN = 9865
SEQUENCE_MAX = 13149
REUSED_PERSISTED_MIN = 9865
REUSED_PERSISTED_MAX = 9874
FIRST_NETWORK_SEQUENCE = 9875
PRIOR_SUCCESSFUL_CALLS = 9864
PRIOR_EXTRA_ATTEMPTS = 4
PRIOR_EXTRA_BYTES = 1_365_188
MAXIMUM_PHYSICAL_ATTEMPTS = 13_153
BYTE_STOP = 4_700_000_000


def _load_base():
    spec = importlib.util.spec_from_file_location("_kma_posthoc_probe_v1_recovery_base", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen acquisition source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = _load_base()
base.OUTPUT = ROOT


def identity(path: Path) -> dict[str, Any]:
    return {"path": path.relative_to(PROJECT).as_posix(), "bytes": path.stat().st_size, "sha256": base.sha256_file(path)}


def verify_execution_config() -> dict[str, Any]:
    payload = json.loads(EXECUTION_CONFIG.read_text(encoding="ascii"))
    if payload.get("schema_version") != 1 or payload.get("status") != "FROZEN_BEFORE_SINGLE_RECOVERY_NETWORK_EXECUTION":
        raise RuntimeError("posthoc recovery binding status/schema mismatch")
    if payload.get("script_identity") != identity(Path(__file__).resolve()):
        raise RuntimeError("posthoc recovery script identity drift")
    for declared in payload.get("bound_inputs", []):
        path = PROJECT / declared["path"]
        if identity(path) != declared:
            raise RuntimeError(f"posthoc recovery input drift: {path.name}")
    if payload.get("actual_argv") != [
        ".venv/Scripts/python.exe", "-B", "scripts/download_kma_d1_1100_wsd_posthoc_rank_probe_v1_recovery.py", "--workers", "1"
    ]:
        raise RuntimeError("posthoc recovery argv mismatch")
    return payload


def prior_accounting() -> tuple[int, int]:
    summary = json.loads(PREGATE_SUMMARY.read_text(encoding="ascii"))
    if summary.get("records") != PRIOR_SUCCESSFUL_CALLS or summary.get("sequence_max") != PRIOR_SUCCESSFUL_CALLS:
        raise RuntimeError("pregate accounting drift")
    return int(summary["response_bytes_sum"]), int(summary["response_bytes_max"])


def validate_persisted_prefix(specs) -> None:
    prefix = [spec for spec in specs if REUSED_PERSISTED_MIN <= spec.sequence <= REUSED_PERSISTED_MAX]
    if len(prefix) != 10:
        raise RuntimeError("recovery prefix internal census mismatch")
    for spec in prefix:
        record = base.read_existing(spec)
        if record is None:
            raise RuntimeError(f"persisted prefix missing: {spec.sequence}")
    first_network = next(spec for spec in specs if spec.sequence == FIRST_NETWORK_SEQUENCE)
    raw, meta = base.paths_for(first_network)
    if raw.exists() or meta.exists():
        raise RuntimeError("sequence 9875 unexpectedly persisted; recovery ledger must be amended")


def run(specs):
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
    return results


def summarize(results):
    prior_sum, prior_max = prior_accounting()
    sizes = [int(item["response_bytes"]) for item in results]
    invalid = [int(item["sequence"]) for item in results if not bool(item.get("selected_cells_all_valid_0_75", True))]
    elapsed = [float(item["elapsed_seconds"]) for item in results if not item.get("reused_existing")]
    observed_max = max([prior_max, *sizes])
    projected = observed_max * MAXIMUM_PHYSICAL_ATTEMPTS
    return {
        "schema_version": 1,
        "experiment": "KMA_D1_1100_WSD_POSTHOC_HIGH_RISK_RANK_PROBE_V1_RECOVERY",
        "mode": "year2025",
        "records": len(results),
        "sequence_min": min(int(item["sequence"]) for item in results),
        "sequence_max": max(int(item["sequence"]) for item in results),
        "persisted_prefix_reused_count": 10,
        "persisted_prefix_reused_sequence_range": [9865, 9874],
        "first_recovery_network_sequence": 9875,
        "response_bytes_sum": sum(sizes),
        "response_bytes_max": max(sizes),
        "prior_successful_calls": PRIOR_SUCCESSFUL_CALLS,
        "prior_successful_response_bytes": prior_sum,
        "prior_extra_physical_attempts": PRIOR_EXTRA_ATTEMPTS,
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
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.workers != 1:
        raise ValueError("recovery ordering requires --workers 1")
    verify_execution_config()
    prior_accounting()
    specs = [item for item in base.all_specs() if item.operating_year == 2025]
    if len(specs) != EXPECTED_RECORDS or specs[0].sequence != SEQUENCE_MIN or specs[-1].sequence != SEQUENCE_MAX:
        raise RuntimeError("2025 request census mismatch")
    validate_persisted_prefix(specs)
    if args.verify_only:
        print(json.dumps({"verify_only": "PASS", "records": len(specs), "first_network_sequence": FIRST_NETWORK_SEQUENCE}, sort_keys=True))
        return
    results = run(specs)
    summary = summarize(results)
    if summary["records"] != EXPECTED_RECORDS or summary["sequence_min"] != SEQUENCE_MIN or summary["sequence_max"] != SEQUENCE_MAX:
        raise RuntimeError("recovery completion census mismatch")
    base.write_exclusive(SUMMARY, base.canonical_json_bytes(summary))
    if not summary["byte_gate_pass"]:
        raise RuntimeError("posthoc recovery byte gate failed")
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

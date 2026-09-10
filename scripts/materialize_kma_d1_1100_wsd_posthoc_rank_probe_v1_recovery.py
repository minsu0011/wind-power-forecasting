"""Offline recovery-bound 2025 materializer for posthoc rank probe V1."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SOURCE_WRAPPER = PROJECT / "scripts/materialize_kma_d1_1100_wsd_posthoc_rank_probe_v1.py"
ROOT = PROJECT / "artifacts/final_submission_sprint_20260812/kma_d1_1100_posthoc_probe_v1/wsd_typ01"
RAW_ROOT = ROOT / "raw_gzip"
OUTPUT_ROOT = ROOT / "materialized_recovery"
PREREG = PROJECT / "configs/kma_d1_1100_wsd_posthoc_rank_probe_v1_preregister.json"
INCIDENT = PROJECT / "configs/kma_d1_1100_wsd_posthoc_rank_probe_v1_acquisition_attempt1_incident.json"
RECOVERY_ACQUISITION_CONFIG = PROJECT / "configs/kma_d1_1100_wsd_posthoc_rank_probe_v1_acquisition_recovery_execution.json"
RECOVERY_DOWNLOADER = PROJECT / "scripts/download_kma_d1_1100_wsd_posthoc_rank_probe_v1_recovery.py"
RECOVERY_MATERIALIZER_CONFIG = PROJECT / "configs/kma_d1_1100_wsd_posthoc_rank_probe_v1_materializer_recovery_execution.json"
COVERAGE_MINIMUM = 0.98
SUMMARY_NAME = "YEAR2025_SUMMARY_POSTHOC_V1_RECOVERY.json"


def _load_wrapper():
    spec = importlib.util.spec_from_file_location("_kma_posthoc_probe_v1_materializer_recovery_base", SOURCE_WRAPPER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen posthoc materializer wrapper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


wrapper = _load_wrapper()
base = wrapper.base
base.__file__ = str(Path(__file__).resolve())
base.ROOT = ROOT
base.RAW_ROOT = RAW_ROOT
base.OUTPUT_ROOT = OUTPUT_ROOT
base.CONFIG = RECOVERY_ACQUISITION_CONFIG
base.MATERIALIZER_CONFIG = RECOVERY_MATERIALIZER_CONFIG
base.DOWNLOADER = RECOVERY_DOWNLOADER
base.MODE["2025"].update(
    summary=SUMMARY_NAME,
    output="KMA_D1_1100_WSD_HOURLY_2025_POSTHOC_V1_RECOVERY.parquet",
    manifest="MANIFEST_2025_POSTHOC_V1_RECOVERY.json",
)


def validate_chain():
    config = base.read_json(RECOVERY_MATERIALIZER_CONFIG)
    if config.get("schema_version") != 1 or config.get("status") != "FROZEN_BEFORE_POSTHOC_RECOVERY_2025_MATERIALIZATION":
        raise RuntimeError("posthoc recovery materializer binding status/schema mismatch")
    if config.get("materializer") != base.identity(Path(__file__).resolve()):
        raise RuntimeError("posthoc recovery materializer identity drift")
    verified = [base.identity(RECOVERY_MATERIALIZER_CONFIG)]
    for declared in config.get("bound_inputs", []):
        path = PROJECT / str(declared.get("path"))
        actual = base.identity(path)
        if actual != declared:
            raise RuntimeError(f"posthoc recovery materializer input drift: {path.name}")
        verified.append(actual)
    return verified


base.validate_chain = validate_chain


def validate_summaries(mode, spec):
    if mode != "2025":
        raise RuntimeError("posthoc recovery materializer authorizes 2025 only")
    summary_path = ROOT / SUMMARY_NAME
    summary = base.read_json(summary_path)
    expected = {
        "mode": "year2025",
        "records": 3285,
        "sequence_min": 9865,
        "sequence_max": 13149,
        "persisted_prefix_reused_count": 10,
        "persisted_prefix_reused_sequence_range": [9865, 9874],
        "first_recovery_network_sequence": 9875,
        "prior_successful_calls": 9864,
        "prior_extra_physical_attempts": 4,
        "prior_extra_physical_bytes_conservative": 1365188,
        "maximum_physical_attempts": 13153,
    }
    if any(summary.get(key) != value for key, value in expected.items()):
        raise RuntimeError("posthoc recovery acquisition summary census/accounting mismatch")
    if summary.get("byte_gate_pass") is not True:
        raise RuntimeError("posthoc recovery acquisition byte gate failed")
    if int(summary.get("projected_physical_bytes_by_observed_max", -1)) > 4_700_000_000:
        raise RuntimeError("posthoc recovery projected physical bytes exceed limit")
    if int(summary.get("total_response_bytes_accounted", -1)) > 4_700_000_000:
        raise RuntimeError("posthoc recovery accounted physical bytes exceed limit")
    invalid = summary.get("invalid_selected_anchor_sequences")
    if not isinstance(invalid, list) or int(summary.get("invalid_selected_anchor_count", -1)) != len(invalid):
        raise RuntimeError("posthoc recovery invalid-anchor census mismatch")
    required = [summary_path, RECOVERY_ACQUISITION_CONFIG, INCIDENT, PREREG]
    return [base.identity(path) for path in required], summary


base.validate_summaries = validate_summaries


def materialize(mode):
    table, evidence = wrapper.materialize(mode)
    evidence["recovery_binding"] = {
        "incident": base.identity(INCIDENT),
        "acquisition_recovery_execution": base.identity(RECOVERY_ACQUISITION_CONFIG),
        "downloader_recovery": base.identity(RECOVERY_DOWNLOADER),
        "summary": base.identity(ROOT / SUMMARY_NAME),
        "physical_attempts_maximum": 13153,
        "prior_extra_attempts": 4,
        "prior_extra_bytes": 1365188,
    }
    evidence["posthoc_high_risk"] = True
    evidence["independent_confirmation"] = False
    return table, evidence


base.materialize = materialize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", choices=("2025",), default="2025")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    validate_chain()
    if args.verify_only:
        print(base.json.dumps({"verify_only": "PASS", "mode": "2025", "coverage_minimum": COVERAGE_MINIMUM}, sort_keys=True))
        return
    table, evidence = materialize("2025")
    out, manifest = base.publish("2025", table, evidence)
    print(base.json.dumps({
        "mode": "2025",
        "rows": table.num_rows,
        "parquet": out.relative_to(PROJECT).as_posix(),
        "manifest": manifest.relative_to(PROJECT).as_posix(),
        "coverage_minimum": COVERAGE_MINIMUM,
        "network_access": False,
        "label_access": False,
        "recovery_bound": True,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

"""Offline materializer bound to the second acquisition recovery ledger."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
PARENT = PROJECT / "scripts/materialize_kma_d1_1100_wsd_posthoc_rank_probe_v1_recovery_v2.py"
BASE_SOURCE = PROJECT / "scripts/materialize_kma_d1_1100_wsd_anchors_recovery.py"
ROOT = PROJECT / "artifacts/final_submission_sprint_20260812/kma_d1_1100_posthoc_probe_v1/wsd_typ01"
OUTPUT_ROOT = ROOT / "materialized_recovery2_v3"
SUMMARY = ROOT / "YEAR2025_SUMMARY_POSTHOC_V1_RECOVERY2.json"
PREGATE_SUMMARY = PROJECT / "artifacts/final_submission_sprint_20260812/kma_d1_1100_v8/wsd_typ01/PREGATE_2022_2024_SUMMARY_V8.json"
PREREG = PROJECT / "configs/kma_d1_1100_wsd_posthoc_rank_probe_v1_preregister.json"
INCIDENT1 = PROJECT / "configs/kma_d1_1100_wsd_posthoc_rank_probe_v1_acquisition_attempt1_incident.json"
RECOVERY1_CONFIG = PROJECT / "configs/kma_d1_1100_wsd_posthoc_rank_probe_v1_acquisition_recovery_execution.json"
INCIDENT2 = PROJECT / "configs/kma_d1_1100_wsd_posthoc_rank_probe_v1_acquisition_recovery_attempt2_incident.json"
RECOVERY2_DOWNLOADER = PROJECT / "scripts/download_kma_d1_1100_wsd_posthoc_rank_probe_v1_recovery2.py"
RECOVERY2_CONFIG = PROJECT / "configs/kma_d1_1100_wsd_posthoc_rank_probe_v1_acquisition_recovery2_execution.json"
RECOVERY2_LOG_ERRATUM = PROJECT / "configs/kma_d1_1100_wsd_posthoc_rank_probe_v1_acquisition_recovery2_log_erratum.json"
CONFIG = PROJECT / "configs/kma_d1_1100_wsd_posthoc_rank_probe_v1_materializer_recovery2_execution_v3.json"


def _load_parent():
    spec = importlib.util.spec_from_file_location("_kma_posthoc_materializer_recovery2_v3_parent", PARENT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen recovery V2 materializer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


parent = _load_parent()
base = parent.base
base.__file__ = str(Path(__file__).resolve())
base.OUTPUT_ROOT = OUTPUT_ROOT
base.MATERIALIZER_CONFIG = CONFIG
base.MODE["2025"].update(
    summary=SUMMARY.name,
    output="KMA_D1_1100_WSD_HOURLY_2025_POSTHOC_V1_RECOVERY2_V3.parquet",
    manifest="MANIFEST_2025_POSTHOC_V1_RECOVERY2_V3.json",
)


def validate_chain():
    config = base.read_json(CONFIG)
    if config.get("schema_version") != 3 or config.get("status") != "FROZEN_BEFORE_POSTHOC_RECOVERY2_V3_MATERIALIZATION":
        raise RuntimeError("recovery2 V3 materializer binding status/schema mismatch")
    if config.get("materializer") != base.identity(Path(__file__).resolve()):
        raise RuntimeError("recovery2 V3 materializer identity drift")
    verified = [base.identity(CONFIG)]
    for declared in config.get("bound_inputs", []):
        path = PROJECT / declared["path"]
        actual = base.identity(path)
        if actual != declared:
            raise RuntimeError(f"recovery2 V3 materializer input drift: {path.name}")
        verified.append(actual)
    required = {BASE_SOURCE, PARENT, PREREG, INCIDENT1, RECOVERY1_CONFIG, INCIDENT2, RECOVERY2_DOWNLOADER, RECOVERY2_CONFIG, RECOVERY2_LOG_ERRATUM, PREGATE_SUMMARY}
    declared = {PROJECT / item["path"] for item in config.get("bound_inputs", [])}
    if not required.issubset(declared):
        raise RuntimeError("recovery2 V3 transitive provenance incomplete")
    return verified


base.validate_chain = validate_chain


def validate_summaries(mode, spec):
    if mode != "2025":
        raise RuntimeError("recovery2 V3 authorizes 2025 only")
    summary = base.read_json(SUMMARY)
    expected = {
        "mode": "year2025", "records": 3285, "sequence_min": 9865, "sequence_max": 13149,
        "persisted_prefix_reused_count": 2970, "persisted_prefix_reused_sequence_range": [9865, 12834],
        "first_recovery_network_sequence": 12835, "remaining_network_calls": 315,
        "prior_successful_calls": 9864, "prior_extra_physical_attempts": 5,
        "prior_extra_physical_bytes_conservative": 1706485, "maximum_physical_attempts": 13154,
    }
    if any(summary.get(key) != value for key, value in expected.items()) or summary.get("byte_gate_pass") is not True:
        raise RuntimeError("recovery2 V3 acquisition summary contract mismatch")
    response_max = int(summary.get("response_bytes_max", -1))
    projected = response_max * 13154
    if int(summary.get("projected_physical_bytes_by_observed_max", -1)) != projected or projected > 4_700_000_000:
        raise RuntimeError("recovery2 V3 projected-byte equation mismatch")
    pregate = base.read_json(PREGATE_SUMMARY)
    total = int(pregate.get("response_bytes_sum", -1)) + 1_706_485 + int(summary.get("response_bytes_sum", -1))
    if int(summary.get("total_response_bytes_accounted", -1)) != total or total > 4_700_000_000:
        raise RuntimeError("recovery2 V3 total-byte equation mismatch")
    invalid = summary.get("invalid_selected_anchor_sequences")
    if not isinstance(invalid, list) or int(summary.get("invalid_selected_anchor_count", -1)) != len(invalid):
        raise RuntimeError("recovery2 V3 invalid-anchor census mismatch")
    return [base.identity(path) for path in (SUMMARY, PREGATE_SUMMARY, RECOVERY2_CONFIG, RECOVERY2_LOG_ERRATUM, INCIDENT2, RECOVERY1_CONFIG, INCIDENT1, PREREG)], summary


base.validate_summaries = validate_summaries


_parent_materialize = parent.materialize


def materialize(mode):
    table, evidence = _parent_materialize(mode)
    evidence["recovery2_v3_binding"] = {
        "base_decoder_materializer": base.identity(BASE_SOURCE),
        "recovery2_downloader": base.identity(RECOVERY2_DOWNLOADER),
        "recovery2_execution": base.identity(RECOVERY2_CONFIG),
        "recovery2_log_erratum": base.identity(RECOVERY2_LOG_ERRATUM),
        "attempt2_incident": base.identity(INCIDENT2),
        "summary": base.identity(SUMMARY),
        "maximum_physical_attempts": 13154,
        "prior_extra_attempts": 5,
        "prior_extra_bytes": 1706485,
    }
    return table, evidence


base.materialize = materialize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", choices=("2025",), default="2025")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    validate_chain()
    if args.verify_only:
        print(base.json.dumps({"verify_only": "PASS", "mode": "2025", "coverage_minimum": 0.98, "recovery2_v3": True}, sort_keys=True)); return
    table, evidence = materialize("2025")
    out, manifest = base.publish("2025", table, evidence)
    print(base.json.dumps({"mode":"2025","rows":table.num_rows,"parquet":out.relative_to(PROJECT).as_posix(),"manifest":manifest.relative_to(PROJECT).as_posix(),"coverage_minimum":0.98,"network_access":False,"label_access":False,"recovery2_v3":True}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

"""Provenance-complete offline recovery materializer for posthoc probe V1."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
PARENT = PROJECT / "scripts/materialize_kma_d1_1100_wsd_posthoc_rank_probe_v1_recovery.py"
BASE_SOURCE = PROJECT / "scripts/materialize_kma_d1_1100_wsd_anchors_recovery.py"
V1_WRAPPER = PROJECT / "scripts/materialize_kma_d1_1100_wsd_posthoc_rank_probe_v1.py"
V1_CONFIG = PROJECT / "configs/kma_d1_1100_wsd_posthoc_rank_probe_v1_materializer_execution.json"
ROOT = PROJECT / "artifacts/final_submission_sprint_20260812/kma_d1_1100_posthoc_probe_v1/wsd_typ01"
OUTPUT_ROOT = ROOT / "materialized_recovery_v2"
PREREG = PROJECT / "configs/kma_d1_1100_wsd_posthoc_rank_probe_v1_preregister.json"
INCIDENT = PROJECT / "configs/kma_d1_1100_wsd_posthoc_rank_probe_v1_acquisition_attempt1_incident.json"
RECOVERY_ACQUISITION_CONFIG = PROJECT / "configs/kma_d1_1100_wsd_posthoc_rank_probe_v1_acquisition_recovery_execution.json"
RECOVERY_DOWNLOADER = PROJECT / "scripts/download_kma_d1_1100_wsd_posthoc_rank_probe_v1_recovery.py"
V1_RECOVERY_CONFIG = PROJECT / "configs/kma_d1_1100_wsd_posthoc_rank_probe_v1_materializer_recovery_execution.json"
CONFIG = PROJECT / "configs/kma_d1_1100_wsd_posthoc_rank_probe_v1_materializer_recovery_execution_v2.json"
PREGATE_SUMMARY = PROJECT / "artifacts/final_submission_sprint_20260812/kma_d1_1100_v8/wsd_typ01/PREGATE_2022_2024_SUMMARY_V8.json"
SUMMARY = ROOT / "YEAR2025_SUMMARY_POSTHOC_V1_RECOVERY.json"


def _load_parent():
    spec = importlib.util.spec_from_file_location("_kma_posthoc_probe_v1_materializer_recovery_v2_parent", PARENT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen recovery materializer parent")
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
    output="KMA_D1_1100_WSD_HOURLY_2025_POSTHOC_V1_RECOVERY_V2.parquet",
    manifest="MANIFEST_2025_POSTHOC_V1_RECOVERY_V2.json",
)


def validate_chain():
    config = base.read_json(CONFIG)
    if config.get("schema_version") != 2 or config.get("status") != "FROZEN_BEFORE_POSTHOC_RECOVERY_V2_2025_MATERIALIZATION":
        raise RuntimeError("posthoc recovery v2 materializer binding status/schema mismatch")
    if config.get("materializer") != base.identity(Path(__file__).resolve()):
        raise RuntimeError("posthoc recovery v2 materializer identity drift")
    verified = [base.identity(CONFIG)]
    for declared in config.get("bound_inputs", []):
        path = PROJECT / str(declared.get("path"))
        actual = base.identity(path)
        if actual != declared:
            raise RuntimeError(f"posthoc recovery v2 materializer input drift: {path.name}")
        verified.append(actual)
    required_paths = {BASE_SOURCE, V1_WRAPPER, V1_CONFIG, PARENT, V1_RECOVERY_CONFIG, RECOVERY_DOWNLOADER, RECOVERY_ACQUISITION_CONFIG, INCIDENT, PREREG, PREGATE_SUMMARY}
    declared_paths = {PROJECT / item["path"] for item in config.get("bound_inputs", [])}
    if declared_paths != required_paths:
        raise RuntimeError("posthoc recovery v2 complete provenance path census mismatch")
    return verified


base.validate_chain = validate_chain


def validate_summaries(mode, spec):
    if mode != "2025":
        raise RuntimeError("posthoc recovery v2 materializer authorizes 2025 only")
    summary = base.read_json(SUMMARY)
    expected = {
        "mode": "year2025", "records": 3285, "sequence_min": 9865, "sequence_max": 13149,
        "persisted_prefix_reused_count": 10, "persisted_prefix_reused_sequence_range": [9865, 9874],
        "first_recovery_network_sequence": 9875, "prior_successful_calls": 9864,
        "prior_extra_physical_attempts": 4, "prior_extra_physical_bytes_conservative": 1365188,
        "maximum_physical_attempts": 13153,
    }
    if any(summary.get(key) != value for key, value in expected.items()):
        raise RuntimeError("posthoc recovery v2 acquisition census/accounting mismatch")
    if summary.get("byte_gate_pass") is not True:
        raise RuntimeError("posthoc recovery v2 acquisition byte gate failed")
    response_max = int(summary.get("response_bytes_max", -1))
    projected = response_max * 13153
    if int(summary.get("projected_physical_bytes_by_observed_max", -1)) != projected or projected > 4_700_000_000:
        raise RuntimeError("posthoc recovery v2 projected-byte equation mismatch")
    pregate = base.read_json(PREGATE_SUMMARY)
    if pregate.get("records") != 9864 or pregate.get("sequence_min") != 1 or pregate.get("sequence_max") != 9864:
        raise RuntimeError("posthoc recovery v2 pregate summary census mismatch")
    total = int(pregate.get("response_bytes_sum", -1)) + 1_365_188 + int(summary.get("response_bytes_sum", -1))
    if int(summary.get("total_response_bytes_accounted", -1)) != total or total > 4_700_000_000:
        raise RuntimeError("posthoc recovery v2 total-byte equation mismatch")
    invalid = summary.get("invalid_selected_anchor_sequences")
    if not isinstance(invalid, list) or int(summary.get("invalid_selected_anchor_count", -1)) != len(invalid):
        raise RuntimeError("posthoc recovery v2 invalid-anchor census mismatch")
    required = [SUMMARY, PREGATE_SUMMARY, RECOVERY_ACQUISITION_CONFIG, INCIDENT, PREREG]
    return [base.identity(path) for path in required], summary


base.validate_summaries = validate_summaries


_parent_materialize = parent.materialize


def materialize(mode):
    table, evidence = _parent_materialize(mode)
    evidence["provenance_v2"] = {
        "base_decoder_materializer": base.identity(BASE_SOURCE),
        "v1_wrapper": base.identity(V1_WRAPPER),
        "v1_execution_config": base.identity(V1_CONFIG),
        "recovery_parent": base.identity(PARENT),
        "recovery_v1_execution_config": base.identity(V1_RECOVERY_CONFIG),
        "pregate_summary": base.identity(PREGATE_SUMMARY),
        "revised_maximum_physical_attempts": 13153,
        "revised_prior_extra_attempts": 4,
        "revised_prior_extra_bytes": 1365188,
    }
    return table, evidence


base.materialize = materialize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", choices=("2025",), default="2025")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    validate_chain()
    if args.verify_only:
        print(base.json.dumps({"verify_only": "PASS", "mode": "2025", "coverage_minimum": 0.98, "provenance_v2": True}, sort_keys=True))
        return
    table, evidence = materialize("2025")
    out, manifest = base.publish("2025", table, evidence)
    print(base.json.dumps({
        "mode": "2025", "rows": table.num_rows,
        "parquet": out.relative_to(PROJECT).as_posix(),
        "manifest": manifest.relative_to(PROJECT).as_posix(),
        "coverage_minimum": 0.98, "network_access": False, "label_access": False,
        "recovery_bound": True, "provenance_v2": True,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

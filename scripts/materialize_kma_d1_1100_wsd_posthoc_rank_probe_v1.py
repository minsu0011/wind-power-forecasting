"""Offline 0.98-coverage materializer for posthoc probe V1 2025 WSD raw."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "scripts/materialize_kma_d1_1100_wsd_anchors_recovery.py"
ROOT = PROJECT / "artifacts/final_submission_sprint_20260812/kma_d1_1100_posthoc_probe_v1/wsd_typ01"
RAW_ROOT = ROOT / "raw_gzip"
OUTPUT_ROOT = ROOT / "materialized"
ACQUISITION_CONFIG = PROJECT / "configs/kma_d1_1100_wsd_posthoc_rank_probe_v1_acquisition_execution.json"
MATERIALIZER_CONFIG = PROJECT / "configs/kma_d1_1100_wsd_posthoc_rank_probe_v1_materializer_execution.json"
DOWNLOADER = PROJECT / "scripts/download_kma_d1_1100_wsd_posthoc_rank_probe_v1.py"
PREREG = PROJECT / "configs/kma_d1_1100_wsd_posthoc_rank_probe_v1_preregister.json"
COVERAGE_MINIMUM = 0.98


def _load_base():
    spec = importlib.util.spec_from_file_location("_kma_posthoc_probe_v1_materializer_base", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen KMA materializer source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    source_text = SOURCE.read_text(encoding="utf-8")
    if source_text.count("0.995") != 2:
        raise RuntimeError("frozen materializer coverage sites drifted")
    exec(compile(source_text.replace("0.995", "0.98"), str(SOURCE) + "::POSTHOC_V1_COVERAGE_0_98", "exec"), module.__dict__)
    return module


base = _load_base()
base.__file__ = str(Path(__file__).resolve())
base.CONFIG = ACQUISITION_CONFIG
base.MATERIALIZER_CONFIG = MATERIALIZER_CONFIG
base.DOWNLOADER = DOWNLOADER
base.ROOT = ROOT
base.RAW_ROOT = RAW_ROOT
base.OUTPUT_ROOT = OUTPUT_ROOT
base.MODE["2025"].update(
    summary="YEAR2025_SUMMARY_POSTHOC_V1.json",
    output="KMA_D1_1100_WSD_HOURLY_2025_POSTHOC_V1.parquet",
    manifest="MANIFEST_2025_POSTHOC_V1.json",
)


def validate_chain():
    config = base.read_json(MATERIALIZER_CONFIG)
    if config.get("schema_version") != 1 or config.get("status") != "FROZEN_BEFORE_POSTHOC_2025_MATERIALIZATION":
        raise RuntimeError("posthoc materializer execution binding status/schema mismatch")
    if config.get("materializer") != base.identity(Path(__file__).resolve()):
        raise RuntimeError("posthoc materializer identity drift")
    verified = [base.identity(MATERIALIZER_CONFIG)]
    for declared in config.get("bound_inputs", []):
        path = PROJECT / str(declared.get("path"))
        actual = base.identity(path)
        if actual != declared:
            raise RuntimeError(f"posthoc materializer input drift: {path.name}")
        verified.append(actual)
    return verified


base.validate_chain = validate_chain


def validate_summaries(mode, spec):
    if mode != "2025":
        raise RuntimeError("posthoc materializer authorizes 2025 only")
    summary_path = ROOT / str(spec["summary"])
    summary = base.read_json(summary_path)
    expected = {
        "mode": "year2025",
        "records": 3285,
        "sequence_min": 9865,
        "sequence_max": 13149,
        "maximum_physical_attempts": 13152,
        "prior_successful_calls": 9864,
        "prior_extra_physical_attempts": 3,
    }
    if any(summary.get(key) != value for key, value in expected.items()):
        raise RuntimeError("posthoc acquisition summary census/accounting mismatch")
    if summary.get("byte_gate_pass") is not True:
        raise RuntimeError("posthoc acquisition byte gate failed")
    if int(summary.get("projected_physical_bytes_by_observed_max", -1)) > 4_700_000_000:
        raise RuntimeError("posthoc projected physical bytes exceed limit")
    if int(summary.get("total_response_bytes_accounted", -1)) > 4_700_000_000:
        raise RuntimeError("posthoc accounted physical bytes exceed limit")
    invalid = summary.get("invalid_selected_anchor_sequences")
    if not isinstance(invalid, list) or int(summary.get("invalid_selected_anchor_count", -1)) != len(invalid):
        raise RuntimeError("posthoc acquisition invalid-anchor census mismatch")
    return [base.identity(summary_path), base.identity(ACQUISITION_CONFIG), base.identity(PREREG)], summary


base.validate_summaries = validate_summaries


_base_materialize = base.materialize


def materialize(mode):
    table, evidence = _base_materialize(mode)
    evidence["posthoc_high_risk"] = True
    evidence["independent_confirmation"] = False
    evidence["coverage_minimum"] = COVERAGE_MINIMUM
    evidence["raw_source"] = {
        "root": RAW_ROOT.relative_to(PROJECT).as_posix(),
        "sequence_range": [9865, 13149],
        "distinct_from_v8": True,
    }
    for year, report in evidence["coverage"].items():
        values = [
            report["cell_94_121_annual_finite_fraction"],
            report["cell_94_122_annual_finite_fraction"],
            *report["cell_94_121_per_lead_finite_fraction"].values(),
            *report["cell_94_122_per_lead_finite_fraction"].values(),
        ]
        if min(values) < COVERAGE_MINIMUM:
            raise RuntimeError(f"posthoc annual/per-lead/cell coverage below 0.98: {year}")
    return table, evidence


base.materialize = materialize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", choices=("2025",), required=False, default="2025")
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
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

"""Offline V8/V9 KMA materializer over immutable old plus distinct V8 raw roots."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "scripts/materialize_kma_d1_1100_wsd_anchors_recovery.py"
OLD_RAW = PROJECT / "artifacts/final_submission_sprint_20260812/kma_d1_1100/wsd_typ01/raw_gzip"
V8_ROOT = PROJECT / "artifacts/final_submission_sprint_20260812/kma_d1_1100_v8/wsd_typ01"
V8_RAW = V8_ROOT / "raw_gzip"
OUTPUT_ROOT = V8_ROOT / "materialized"
EXECUTION_CONFIG = PROJECT / "configs/kma_d1_1100_wsd_materializer_execution_v8.json"
REUSE_LAST_SEQUENCE = 5418
COVERAGE_MINIMUM = 0.98


def _load():
    spec = importlib.util.spec_from_file_location("_kma_materializer_v8_base", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen recovery materializer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    # V9 authorizes exactly one scientific-code delta: the target-free availability
    # floor. Execute an otherwise byte-derived copy with all three literal threshold
    # uses replaced before the module can run; the strict source stays immutable.
    source_text = SOURCE.read_text(encoding="utf-8")
    if source_text.count("0.995") != 2:
        raise RuntimeError("frozen recovery materializer threshold sites drifted")
    v8_text = source_text.replace("0.995", "0.98")
    exec(compile(v8_text, str(SOURCE) + "::V8_THRESHOLD_0_98", "exec"), module.__dict__)
    return module


base = _load()
# The inherited publisher/chain validator must identify this executed V8 wrapper,
# not the immutable source text from which the implementation was derived.
base.__file__ = str(Path(__file__).resolve())
base.ROOT = V8_ROOT
base.RAW_ROOT = V8_RAW
base.OUTPUT_ROOT = OUTPUT_ROOT
base.CONFIG = PROJECT / "configs/kma_d1_1100_wsd_sprint_execution_v8.json"
base.MATERIALIZER_CONFIG = EXECUTION_CONFIG
base.DOWNLOADER = PROJECT / "scripts/download_kma_d1_1100_wsd_grid_sprint_v8.py"
base.MODE["pregate"].update(summary="PREGATE_2022_2024_SUMMARY_V8.json", output="KMA_D1_1100_WSD_HOURLY_2022_2024_V8.parquet", manifest="MANIFEST_V8.json")
base.MODE["2025"].update(summary="YEAR2025_SUMMARY_V8.json", output="KMA_D1_1100_WSD_HOURLY_2025_V8.parquet", manifest="MANIFEST_2025_V8.json")


def raw_paths(item):
    root = OLD_RAW if int(item["sequence"]) <= REUSE_LAST_SEQUENCE else V8_RAW
    folder = root / str(item["operating_year"])
    return folder / f"{item['stem']}.txt.gz", folder / f"{item['stem']}.json"


_base_validate_record = base.validate_record
def validate_record(item):
    original = base.RAW_ROOT
    base.RAW_ROOT = OLD_RAW if int(item["sequence"]) <= REUSE_LAST_SEQUENCE else V8_RAW
    try:
        return _base_validate_record(item)
    finally:
        base.RAW_ROOT = original
base.validate_record = validate_record


def validate_census(years, expected):
    expected_by_root = {}
    for item in expected:
        root = OLD_RAW if int(item["sequence"]) <= REUSE_LAST_SEQUENCE else V8_RAW
        key = (root, int(item["operating_year"]))
        expected_by_root.setdefault(key, set()).update({f"{item['stem']}.txt.gz", f"{item['stem']}.json"})
    for (root, year), names in expected_by_root.items():
        folder = root / str(year)
        if not folder.is_dir():
            raise RuntimeError(f"V8 union raw folder missing: {folder}")
        actual = {p.name for p in folder.iterdir() if p.is_file()}
        # Old namespace may contain later records; require every exact reused input, never write it.
        missing = names - actual
        if missing:
            raise RuntimeError(f"V8 union raw census missing={len(missing)} for {year}")
        if root == V8_RAW and actual != names:
            raise RuntimeError(f"V8 distinct raw census extra={len(actual - names)} for {year}")
base.validate_census = validate_census


def validate_chain():
    config = base.read_json(EXECUTION_CONFIG)
    if config.get("schema_version") != 8 or config.get("materializer") != base.identity(Path(__file__).resolve()):
        raise RuntimeError("V8 materializer execution binding mismatch")
    verified = []
    for declared in config.get("bound_inputs", []):
        path = PROJECT / declared["path"]
        actual = base.identity(path)
        if actual != declared:
            raise RuntimeError(f"V8 materializer input drift: {path.name}")
        verified.append(actual)
    return [base.identity(EXECUTION_CONFIG), *verified]
base.validate_chain = validate_chain


def validate_summaries(mode, spec):
    required = [V8_ROOT / "FIRST100_SUMMARY_V8.json", V8_ROOT / spec["summary"]]
    if mode == "2025":
        required.insert(1, V8_ROOT / "PREGATE_2022_2024_SUMMARY_V8.json")
        required.append(V8_ROOT / "MODEL_GATE_PASS_V8.json")
        if base.read_json(required[-1]).get("all_gates_pass") is not True:
            raise RuntimeError("V8 2025 materialization remains blocked")
    first, summary = base.read_json(required[0]), base.read_json(V8_ROOT / spec["summary"])
    expected = {"mode": spec["summary_mode"], "records": spec["records"], "sequence_min": spec["sequence_min"], "sequence_max": spec["sequence_max"]}
    if first.get("records") != 100 or first.get("byte_gate_pass") is not True:
        raise RuntimeError("V8 first100 summary invalid")
    if any(summary.get(k) != v for k, v in expected.items()) or summary.get("byte_gate_pass") is not True:
        raise RuntimeError("V8 acquisition summary invalid")
    if summary.get("maximum_physical_attempts") != 13152 or summary.get("prior_physical_bytes_conservative") != 1023891:
        raise RuntimeError("V8 physical attempt accounting invalid")
    projected = int(summary.get("response_bytes_max", -1)) * 13152
    if int(summary.get("projected_physical_bytes_by_observed_max", -1)) != projected or projected > 4_700_000_000:
        raise RuntimeError("V8 projected physical bytes invalid")
    if int(summary.get("quota_response_bytes_accounted_current_mode", -1)) != int(summary.get("response_bytes_sum", -1)) + 1023891:
        raise RuntimeError("V8 quota byte census invalid")
    invalid = summary.get("invalid_selected_anchor_sequences")
    if not isinstance(invalid, list) or int(summary.get("invalid_selected_anchor_count", -1)) != len(invalid):
        raise RuntimeError("V8 invalid-anchor summary census invalid")
    return [base.identity(path) for path in required], summary
base.validate_summaries = validate_summaries


_base_materialize = base.materialize
def materialize(mode):
    table, evidence = _base_materialize(mode)
    evidence["union_raw_sources"] = {
        "immutable_read_only_source": OLD_RAW.relative_to(PROJECT).as_posix(),
        "immutable_sequences": [1, REUSE_LAST_SEQUENCE],
        "distinct_v8_source": V8_RAW.relative_to(PROJECT).as_posix(),
        "distinct_v8_sequences_begin": REUSE_LAST_SEQUENCE + 1,
    }
    for year, report in evidence["coverage"].items():
        report["coverage_gate_minimum"] = COVERAGE_MINIMUM
        values = [report["cell_94_121_annual_finite_fraction"], report["cell_94_122_annual_finite_fraction"], *report["cell_94_121_per_lead_finite_fraction"].values(), *report["cell_94_122_per_lead_finite_fraction"].values()]
        report["coverage_gate_pass"] = min(values) >= COVERAGE_MINIMUM
        if not report["coverage_gate_pass"]:
            raise RuntimeError(f"V8 annual/per-lead/cell coverage below 0.98: {year}")
    return table, evidence
base.materialize = materialize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", choices=("pregate", "2025"), required=True)
    args = parser.parse_args()
    table, evidence = materialize(args.years)
    out, manifest = base.publish(args.years, table, evidence)
    print(base.json.dumps({"mode": args.years, "rows": table.num_rows, "parquet": out.relative_to(PROJECT).as_posix(), "manifest": manifest.relative_to(PROJECT).as_posix(), "coverage_minimum": COVERAGE_MINIMUM}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

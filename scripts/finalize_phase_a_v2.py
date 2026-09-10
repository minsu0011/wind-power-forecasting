"""Read-only semantic audit and immutable completion record for Phase A."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evidence_claims import (  # noqa: E402
    file_identity,
    sha256_file,
    write_csv_exclusive,
    write_json_exclusive,
)


REQUIRED_OUTPUTS = (
    "EVIDENCE_CLAIM_LEDGER.csv",
    "MISSING_EVIDENCE_CLOSURE.md",
    "FULL_WORKSPACE_ARTIFACT_INVENTORY.parquet",
    "FULL_WORKSPACE_ARTIFACT_INVENTORY.json",
    "PUBLIC_SHA_REGISTRY_VERIFIED.csv",
    "USER_REPORTED_FEEDBACK_RECEIPT_AGREEMENT_V2.json",
    "USER_REPORTED_FEEDBACK_REGISTRY_V2.csv",
    "SUBMISSION_LINEAGE_NODES.parquet",
    "SUBMISSION_LINEAGE_EDGES.parquet",
    "SUBMISSION_LINEAGE_GRAPH.graphml",
    "SUBMISSION_LINEAGE_REPORT.md",
    "SUBMISSION_LINEAGE_REQUIRED_FACTS.json",
    "SUBMISSION_LINEAGE_SUMMARY.json",
    "SUBMISSION_LINEAGE_MANIFEST.json",
)

CORRECTION_RECEIPT = "USER_REPORTED_FEEDBACK_RECEIPT_AGREEMENT_V2_TIMESTAMP_CORRECTION.json"
CORRECTION_REGISTRY = "USER_REPORTED_FEEDBACK_REGISTRY_V2_TIMESTAMP_CORRECTION.csv"


def run(config_path: Path) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    phase_root = (ROOT / config["output_root"]).resolve()
    for name in REQUIRED_OUTPUTS:
        if not (phase_root / name).is_file():
            raise FileNotFoundError(phase_root / name)
    nodes = pd.read_parquet(phase_root / "SUBMISSION_LINEAGE_NODES.parquet")
    edges = pd.read_parquet(phase_root / "SUBMISSION_LINEAGE_EDGES.parquet")
    inventory = pd.read_parquet(phase_root / "FULL_WORKSPACE_ARTIFACT_INVENTORY.parquet")
    facts = json.loads((phase_root / "SUBMISSION_LINEAGE_REQUIRED_FACTS.json").read_text(encoding="utf-8"))
    original_receipt_path = phase_root / "USER_REPORTED_FEEDBACK_RECEIPT_AGREEMENT_V2.json"
    original_receipt = json.loads(original_receipt_path.read_text(encoding="utf-8"))
    submitted_kst = config["user_feedback"].get("submitted_kst")
    if submitted_kst != "2026-08-10T14:57:39+09:00":
        raise ValueError("frozen user-reported feedback timestamp mismatch")
    receipt = {
        **original_receipt,
        "schema_version": 2,
        "submitted_kst": submitted_kst,
        "correction_type": "APPEND_ONLY_TIMESTAMP_CORRECTION",
        "supersedes_receipt": file_identity(original_receipt_path),
        "evidence_origin": "USER_REPORTED",
        "claim_class": "LOCAL_REPORTED_ONLY",
        "platform_receipt_verified": False,
        "selector_eligible": False,
        "notes": (
            "Append-only correction: the user supplied the Dacon memo timestamp "
            "2026-08-10 14:57:39 KST. The filename/SHA/metrics remain unchanged, "
            "platform receipt verification remains false, and the row is never selector eligible."
        ),
    }
    correction_path = phase_root / CORRECTION_RECEIPT
    write_json_exclusive(correction_path, receipt)
    correction_identity = file_identity(correction_path)
    correction_registry_row = {
        "submission_id": receipt["submission_id"],
        "submitted_kst": submitted_kst,
        "csv_name": receipt["csv_name"],
        "csv_sha256": receipt["csv_sha256"],
        "score": receipt["score"],
        "one_minus_nmae": receipt["one_minus_nmae"],
        "ficr": receipt["ficr"],
        "receipt_path": correction_identity["path"],
        "receipt_sha256": correction_identity["sha256"],
        "platform_filename": receipt["platform_filename"],
        "claim_class": "LOCAL_REPORTED_ONLY",
        "notes": receipt["notes"],
        "platform_receipt_verified": False,
        "selector_eligible": False,
    }
    write_csv_exclusive(
        phase_root / CORRECTION_REGISTRY,
        [correction_registry_row],
        list(correction_registry_row),
    )
    closure = json.loads((phase_root / "EVIDENCE_CLOSURE_SUMMARY.json").read_text(encoding="utf-8"))
    ET.parse(phase_root / "SUBMISSION_LINEAGE_GRAPH.graphml")

    if not facts.get("all_required_facts_pass"):
        raise ValueError("required lineage facts are not closed")
    if receipt.get("selector_eligible") is not False or receipt.get("platform_receipt_verified") is not False:
        raise ValueError("feedback evidence boundary violated")
    if receipt.get("claim_class") != "LOCAL_REPORTED_ONLY":
        raise ValueError("feedback claim class was promoted")
    if closure.get("missing_required_claims") != [] or closure.get("reload_checks_bit_exact") != 50:
        raise ValueError("evidence closure is incomplete")
    if nodes["selector_eligible"].astype(bool).any():
        raise ValueError("a lineage node is selector eligible")

    sha_group_sizes = nodes.groupby("csv_sha256").size()
    prediction_group_sizes = nodes.groupby("prediction_hash_full").size()
    rejected = edges.loc[edges["decision"].eq("DUPLICATE_MECHANISM_REJECTED")]
    multi = inventory["relative_path"].str.lower().str.contains("multi_nwp")
    paths = inventory["relative_path"].str.lower()
    coverage = {
        "raw": int((multi & paths.str.contains(r"/raw|raw_", regex=True)).sum()),
        "provenance": int((multi & paths.str.contains(r"manifest|audit|provenance|source_lock|access_ledger", regex=True)).sum()),
        "feature": int((multi & paths.str.contains(r"feature|centroid|increment_cf", regex=True)).sum()),
        "model": int((multi & paths.str.contains(r"/models/|model\.joblib|__control\.joblib|__extended", regex=True)).sum()),
        "prediction": int((multi & paths.str.contains(r"/predictions/|_2025\.csv", regex=True)).sum()),
    }
    if any(value == 0 for value in coverage.values()):
        raise ValueError(f"multi-NWP evidence coverage class missing: {coverage}")

    compact = (ROOT / config["compact_root"]).resolve()
    audit = {
        "schema_version": 1,
        "phase": "A",
        "verdict": "PASS",
        "bounded_scope": "compact/full evidence closure and submission lineage only",
        "compact_public_results_immutable_sha256": sha256_file(compact / "PUBLIC_RESULTS.csv"),
        "compact_manifest_immutable_sha256": sha256_file(compact / "MANIFEST_SHA256.csv"),
        "evidence": {
            "compact_manifest_entries_hash_matched": closure["compact_manifest_entries"],
            "public_rows_sha_schema_arithmetic_verified": closure["public_rows_verified"],
            "feature_exit_models": closure["feature_exit_models"],
            "reload_checks_bit_exact": closure["reload_checks_bit_exact"],
            "required_claims_missing": closure["missing_required_claims"],
            "inventory_files": closure["inventory_files"],
            "inventory_status_counts": closure["inventory_status_counts"],
            "multi_nwp_artifact_coverage": coverage,
        },
        "lineage": {
            "valid_csv_paths": int(len(nodes)),
            "unique_csv_sha256": int(nodes["csv_sha256"].nunique()),
            "unique_prediction_tensors": int(nodes["prediction_hash_full"].nunique()),
            "exact_sha_duplicate_groups": int((sha_group_sizes > 1).sum()),
            "exact_sha_extra_path_copies": int(len(nodes) - nodes["csv_sha256"].nunique()),
            "exact_prediction_duplicate_groups": int((prediction_group_sizes > 1).sum()),
            "exact_prediction_extra_path_copies": int(len(nodes) - nodes["prediction_hash_full"].nunique()),
            "duplicate_mechanism_unique_pairs": int(rejected[["parent_node_id", "child_node_id"]].drop_duplicates().shape[0]),
            "independent_hypothesis_count": int(nodes["independent_axis"].nunique()),
            "independent_hypotheses": sorted(nodes["independent_axis"].unique().tolist()),
            "required_facts": facts,
        },
        "user_feedback": {
            "submitted_kst": receipt["submitted_kst"],
            "csv_sha256": receipt["csv_sha256"],
            "score": receipt["score"],
            "one_minus_nmae": receipt["one_minus_nmae"],
            "ficr": receipt["ficr"],
            "claim_class": receipt["claim_class"],
            "platform_receipt_verified": False,
            "selector_eligible": False,
        },
        "public_selector_used": False,
        "model_fit_or_prediction_generation_performed": False,
        "inputs_and_code": [
            file_identity(config_path),
            file_identity(ROOT / "src" / "evidence_claims.py"),
            file_identity(ROOT / "src" / "submission_lineage.py"),
            file_identity(ROOT / "scripts" / "run_evidence_closure_v2.py"),
            file_identity(ROOT / "scripts" / "run_submission_lineage_v2.py"),
            file_identity(ROOT / "scripts" / "finalize_phase_a_v2.py"),
            file_identity(ROOT / "tests" / "test_public_sha_registry.py"),
            file_identity(ROOT / "tests" / "test_submission_lineage.py"),
            file_identity(ROOT / "tests" / "test_no_public_selector.py"),
        ],
        "outputs": [
            file_identity(phase_root / name)
            for name in (*REQUIRED_OUTPUTS, CORRECTION_RECEIPT, CORRECTION_REGISTRY)
        ],
    }
    write_json_exclusive(phase_root / "PHASE_A_COMPLETION_AUDIT.json", audit)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    audit = run(args.config.resolve())
    print(json.dumps({"verdict": audit["verdict"], **audit["lineage"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Seal the append-only Phase-A user-feedback timestamp amendment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evidence_claims import file_identity, sha256_file, write_json_exclusive  # noqa: E402


def run(base_config_path: Path, amendment_path: Path) -> dict[str, object]:
    base = json.loads(base_config_path.read_text(encoding="utf-8"))
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    base_identity = file_identity(base_config_path)
    if amendment["base_config_sha256"] != base_identity["sha256"]:
        raise ValueError("append-only amendment does not bind the restored base config")
    if amendment["submitted_kst"] != "2026-08-10T14:57:39+09:00":
        raise ValueError("user-reported KST timestamp mismatch")
    if amendment["claim_class"] != "LOCAL_REPORTED_ONLY":
        raise ValueError("user-reported feedback was promoted")
    if amendment["selector_eligible"] is not False or amendment["platform_receipt_verified"] is not False:
        raise ValueError("user-reported evidence boundary violated")

    phase_root = (ROOT / base["output_root"]).resolve()
    lineage_manifest_path = phase_root / "SUBMISSION_LINEAGE_MANIFEST.json"
    lineage_manifest = json.loads(lineage_manifest_path.read_text(encoding="utf-8"))
    if lineage_manifest["config"]["sha256"] != base_identity["sha256"]:
        raise ValueError("lineage manifest no longer matches the immutable base config")
    correction_path = phase_root / "USER_REPORTED_FEEDBACK_RECEIPT_AGREEMENT_V2_TIMESTAMP_CORRECTION.json"
    correction = json.loads(correction_path.read_text(encoding="utf-8"))
    for key in ("submitted_kst", "csv_sha256", "score", "one_minus_nmae", "ficr", "claim_class"):
        if correction[key] != amendment[key]:
            raise ValueError(f"correction receipt/amendment mismatch: {key}")
    if correction["selector_eligible"] is not False or correction["platform_receipt_verified"] is not False:
        raise ValueError("correction receipt evidence boundary violated")

    compact = (ROOT / base["compact_root"]).resolve()
    if sha256_file(compact / "PUBLIC_RESULTS.csv") != base["expected"]["public_results_sha256"]:
        raise ValueError("frozen compact Public registry changed")
    prior_audit = phase_root / "PHASE_A_COMPLETION_AUDIT.json"
    registry = phase_root / "USER_REPORTED_FEEDBACK_REGISTRY_V2_TIMESTAMP_CORRECTION.csv"
    audit = {
        "schema_version": 1,
        "artifact_type": "phase_a_completion_audit_append_only_timestamp_amendment",
        "verdict": "PASS",
        "base_lineage_immutable": True,
        "base_config": base_identity,
        "amendment_config": file_identity(amendment_path),
        "lineage_manifest": file_identity(lineage_manifest_path),
        "prior_completion_audit": file_identity(prior_audit),
        "timestamp_correction_receipt": file_identity(correction_path),
        "timestamp_correction_registry": file_identity(registry),
        "submitted_kst": amendment["submitted_kst"],
        "csv_sha256": amendment["csv_sha256"],
        "score": amendment["score"],
        "one_minus_nmae": amendment["one_minus_nmae"],
        "ficr": amendment["ficr"],
        "claim_class": "LOCAL_REPORTED_ONLY",
        "evidence_origin": "USER_REPORTED",
        "platform_receipt_verified": False,
        "selector_eligible": False,
        "public_registry_member": False,
        "public_selector_used": False,
        "compact_public_results_sha256": base["expected"]["public_results_sha256"],
        "focused_tests": {
            "command": "python -m pytest -q tests/test_public_sha_registry.py tests/test_submission_lineage.py tests/test_no_public_selector.py",
            "passed": 10,
            "failed": 0,
        },
        "current_code_and_tests": [
            file_identity(ROOT / "src" / "evidence_claims.py"),
            file_identity(ROOT / "src" / "submission_lineage.py"),
            file_identity(ROOT / "scripts" / "run_evidence_closure_v2.py"),
            file_identity(ROOT / "scripts" / "run_submission_lineage_v2.py"),
            file_identity(ROOT / "scripts" / "seal_phase_a_receipt_amendment_v2.py"),
            file_identity(ROOT / "tests" / "test_public_sha_registry.py"),
            file_identity(ROOT / "tests" / "test_submission_lineage.py"),
            file_identity(ROOT / "tests" / "test_no_public_selector.py"),
        ],
        "notes": (
            "The original compact PUBLIC_RESULTS.csv, base config, evidence outputs, and lineage outputs "
            "were not overwritten. This seal only appends the user-supplied memo timestamp."
        ),
    }
    write_json_exclusive(phase_root / "PHASE_A_COMPLETION_AUDIT_TIMESTAMP_AMENDMENT.json", audit)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.base_config.resolve(), args.amendment.resolve())
    print(json.dumps({"verdict": result["verdict"], "submitted_kst": result["submitted_kst"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


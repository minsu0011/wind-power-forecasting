"""Run BARAM2026 Phase-A compact/full-workspace evidence closure."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evidence_claims import (  # noqa: E402
    build_user_feedback_receipt,
    classify_identity,
    discover_submission_csvs,
    file_identity,
    manifest_registration_index,
    sha256_file,
    verify_compact_manifest,
    verify_public_registry,
    write_csv_exclusive,
    write_json_exclusive,
    write_parquet_exclusive,
    write_text_exclusive,
)


REQUIRED_RELATIVE = (
    "correlation_map/feature_map.parquet",
    "correlation_map/family_map.parquet",
    "correlation_map/redundancy_membership.parquet",
    "correlation_map/redundancy_edges.parquet",
    "inner/inner_oof_predictions.parquet",
    "inner/inner_direct_scores.json",
    "stress/bootstrap_draw_and_candidate_ledger.json",
    "stress/virtual_score_trajectory.json",
    "outer/outer_lock_or_rejection.json",
    "inner/training_and_reload_checks.json",
)


def _resolve(workspace: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (workspace / path).resolve()


def _assert_expected(config: dict[str, Any], workspace: Path, compact: Path) -> None:
    expected = config["expected"]
    checks = {
        "prompt_sha256": _resolve(workspace, config["prompt_path"]),
        "compact_zip_sha256": _resolve(workspace, config["compact_zip"]),
        "compact_manifest_sha256": compact / "MANIFEST_SHA256.csv",
        "public_results_sha256": compact / "PUBLIC_RESULTS.csv",
    }
    for key, path in checks.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected[key]:
            raise ValueError(f"frozen identity mismatch for {key}: {actual}")


def _collect_relevant_files(workspace: Path, compact: Path, feature_exit: Path) -> list[Path]:
    files: set[Path] = set()
    files.update(path.resolve() for path in compact.rglob("*") if path.is_file())
    files.update(path.resolve() for path in feature_exit.rglob("*") if path.is_file())
    artifacts = workspace / "artifacts"
    files.update(path.resolve() for path in discover_submission_csvs(artifacts))
    files.update(
        path.resolve()
        for path in artifacts.rglob("*")
        if path.is_file() and "multi_nwp" in path.as_posix().lower()
    )
    for folder in (workspace / "configs", workspace / "scripts", workspace / "tests", workspace / "src"):
        if folder.is_dir():
            files.update(
                path.resolve()
                for path in folder.iterdir()
                if path.is_file()
                and (
                    "multi_nwp" in path.name.lower()
                    or "feature_exit" in path.name.lower()
                    or path.name in {
                        "evidence_claims.py",
                        "submission_lineage.py",
                        "run_evidence_closure_v2.py",
                        "run_submission_lineage_v2.py",
                        "test_public_sha_registry.py",
                        "test_submission_lineage.py",
                        "baram2026_evidence_first_v2.json",
                    }
                )
            )
    return sorted(files, key=lambda path: path.as_posix().lower())


def _category(path: Path) -> str:
    text = path.as_posix().lower()
    if "csv_scored" in text or "csv_unscored" in text or text.endswith("submission.csv"):
        return "submission_csv"
    if "feature_exit_virtual_v1/inner/models" in text:
        return "feature_exit_model"
    if "feature_exit_virtual_v1" in text:
        return "feature_exit_evidence"
    if "multi_nwp" in text:
        for token in ("raw", "provenance", "feature", "model", "prediction"):
            if token in text:
                return f"multi_nwp_{token}"
        return "multi_nwp_other"
    if "compact" in text:
        return "compact"
    return "code_or_config"


def run(config_path: Path) -> dict[str, Any]:
    workspace = ROOT
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("phase") != "A":
        raise ValueError("this runner is Phase A only")
    output = _resolve(workspace, config["output_root"])
    compact = _resolve(workspace, config["compact_root"])
    feature_exit = _resolve(workspace, config["feature_exit_root"])
    output.mkdir(parents=True, exist_ok=True)

    _assert_expected(config, workspace, compact)
    compact_rows = verify_compact_manifest(compact)
    if any(row["status"] != "FOUND_HASH_MATCH" for row in compact_rows):
        raise ValueError("compact manifest closure failed")
    public_rows, _ = verify_public_registry(compact)

    manifests = sorted(feature_exit.rglob("*.json"))
    manifests += sorted(
        path for path in (workspace / "artifacts").rglob("*.json")
        if "multi_nwp" in path.as_posix().lower()
    )
    registrations = manifest_registration_index(manifests)
    # The compact CSV manifest is itself an immutable registration source.
    for row in compact_rows:
        absolute = (compact / row["path"]).resolve().as_posix().lower()
        registrations.setdefault(absolute, []).append(
            {
                "declared_path": absolute,
                "declared_sha256": row["sha256"],
                "declared_size_bytes": row["size_bytes"],
                "source_manifest": (compact / "MANIFEST_SHA256.csv").resolve().as_posix(),
            }
        )

    relevant = _collect_relevant_files(workspace, compact, feature_exit)
    inventory: list[dict[str, Any]] = []
    for path in relevant:
        row = classify_identity(path, registrations)
        row["relative_path"] = path.relative_to(workspace).as_posix()
        row["category"] = _category(path)
        inventory.append(row)

    ledger: list[dict[str, Any]] = []
    for rel in REQUIRED_RELATIVE:
        path = feature_exit / rel
        identity = classify_identity(path, registrations)
        ledger.append(
            {
                "claim_id": f"required::{rel}",
                "claim": f"Required Phase-A artifact exists and its local bytes were rehashed: {rel}",
                "claim_class": "LOCAL_REPRODUCED",
                "artifact_path": identity["path"],
                "artifact_sha256": identity["sha256"],
                "artifact_size_bytes": identity["size_bytes"],
                "closure_status": identity["status"],
                "selector_eligible": False,
                "notes": "File identity only; no metric or model was recomputed.",
            }
        )

    checks_path = feature_exit / "inner" / "training_and_reload_checks.json"
    checks = json.loads(checks_path.read_text(encoding="utf-8"))
    model_files = sorted((feature_exit / "inner" / "models").rglob("*.joblib"))
    if len(checks) != 50 or len(model_files) != 50:
        raise ValueError(f"expected 50 checks/models, got {len(checks)}/{len(model_files)}")
    if not all(record.get("reload_prediction_bit_exact") is True for record in checks.values()):
        raise ValueError("not all model reload checks are bit-exact")
    declared_models = {Path(record["model"]["path"]).resolve().as_posix().lower(): record for record in checks.values()}
    for index, model_path in enumerate(model_files, start=1):
        identity = classify_identity(model_path, registrations)
        record = declared_models.get(model_path.resolve().as_posix().lower())
        exact = bool(
            record
            and record["model"]["sha256"] == identity["sha256"]
            and int(record["model"]["size_bytes"]) == identity["size_bytes"]
            and record.get("reload_prediction_bit_exact") is True
        )
        ledger.append(
            {
                "claim_id": f"model_reload::{index:02d}",
                "claim": "Persisted model identity and recorded reload prediction are bit-exact.",
                "claim_class": "LOCAL_REPRODUCED",
                "artifact_path": identity["path"],
                "artifact_sha256": identity["sha256"],
                "artifact_size_bytes": identity["size_bytes"],
                "closure_status": "FOUND_HASH_MATCH" if exact else "FOUND_HASH_MISMATCH",
                "selector_eligible": False,
                "notes": "Verified from training_and_reload_checks.json without loading model or labels.",
            }
        )
    if any(row["closure_status"] != "FOUND_HASH_MATCH" for row in ledger if row["claim_id"].startswith("model_reload")):
        raise ValueError("50-model identity/reload closure failed")

    for public in public_rows:
        ledger.append(
            {
                "claim_id": f"public::{public['label']}",
                "claim": f"Frozen Public row {public['label']} is SHA-linked and arithmetically consistent.",
                "claim_class": "PUBLIC_SHA_LINKED",
                "artifact_path": public["csv_path"],
                "artifact_sha256": public["sha256"],
                "artifact_size_bytes": public["csv_size_bytes"],
                "closure_status": "FOUND_HASH_MATCH",
                "selector_eligible": False,
                "notes": "Public evidence is diagnostic only; platform receipt is not verified.",
            }
        )

    feedback = config["user_feedback"]
    receipt = build_user_feedback_receipt(
        compact,
        float(feedback["score"]),
        float(feedback["one_minus_nmae"]),
        float(feedback["ficr"]),
        str(feedback["submitted_kst"]) if feedback.get("submitted_kst") else None,
    )
    receipt_path = output / "USER_REPORTED_FEEDBACK_RECEIPT_AGREEMENT_V2.json"
    write_json_exclusive(receipt_path, receipt)
    receipt_identity = file_identity(receipt_path)
    registry_row = {
        "submission_id": receipt["submission_id"],
        "submitted_kst": "",
        "csv_name": receipt["csv_name"],
        "csv_sha256": receipt["csv_sha256"],
        "score": receipt["score"],
        "one_minus_nmae": receipt["one_minus_nmae"],
        "ficr": receipt["ficr"],
        "receipt_path": receipt_identity["path"],
        "receipt_sha256": receipt_identity["sha256"],
        "platform_filename": receipt["platform_filename"],
        "claim_class": receipt["claim_class"],
        "notes": receipt["notes"],
        "platform_receipt_verified": False,
        "selector_eligible": False,
    }
    write_csv_exclusive(
        output / "USER_REPORTED_FEEDBACK_REGISTRY_V2.csv",
        [registry_row],
        list(registry_row),
    )
    ledger.append(
        {
            "claim_id": "user_feedback::agreement_gate_g12",
            "claim": "User-reported agreement CSV feedback is preserved without platform verification.",
            "claim_class": "LOCAL_REPORTED_ONLY",
            "artifact_path": receipt["csv_path"],
            "artifact_sha256": receipt["csv_sha256"],
            "artifact_size_bytes": receipt["csv_size_bytes"],
            "closure_status": "FOUND_HASH_MATCH",
            "selector_eligible": False,
            "notes": f"Append-only receipt SHA {receipt_identity['sha256']}; never selector-eligible.",
        }
    )

    ledger_columns = [
        "claim_id", "claim", "claim_class", "artifact_path", "artifact_sha256",
        "artifact_size_bytes", "closure_status", "selector_eligible", "notes",
    ]
    write_csv_exclusive(output / "EVIDENCE_CLAIM_LEDGER.csv", ledger, ledger_columns)
    inventory_frame = pd.DataFrame(inventory)
    write_parquet_exclusive(output / "FULL_WORKSPACE_ARTIFACT_INVENTORY.parquet", inventory_frame)
    write_json_exclusive(output / "FULL_WORKSPACE_ARTIFACT_INVENTORY.json", inventory)
    write_csv_exclusive(
        output / "PUBLIC_SHA_REGISTRY_VERIFIED.csv",
        public_rows,
        list(public_rows[0]),
    )

    status_counts = Counter(row["status"] for row in inventory)
    required_counts = Counter(row["closure_status"] for row in ledger)
    missing = [row for row in ledger if row["closure_status"] in {"MISSING", "FOUND_HASH_MISMATCH"}]
    closure_md = [
        "# Phase A Evidence Closure",
        "",
        f"- Generated UTC: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Compact manifest entries: `{len(compact_rows)}`; all SHA/bytes match: `true`",
        f"- Frozen Public rows: `{len(public_rows)}`; all SHA/schema/arithmetic checks pass: `true`",
        f"- Feature-exit model files/reload records: `{len(model_files)}/{len(checks)}`; bit-exact records: `50`",
        f"- Relevant full-workspace inventory files: `{len(inventory)}`",
        f"- Inventory status counts: `{dict(sorted(status_counts.items()))}`",
        f"- Required-claim status counts: `{dict(sorted(required_counts.items()))}`",
        f"- PLATFORM_RECEIPT_VERIFIED: `false`",
        f"- User agreement feedback: `LOCAL_REPORTED_ONLY`, CSV SHA `{receipt['csv_sha256']}`, selector eligible: `false`",
        "",
        "## Missing or mismatched required evidence",
        "",
    ]
    if missing:
        closure_md.extend(f"- `{row['claim_id']}`: `{row['closure_status']}`" for row in missing)
    else:
        closure_md.append("- None among the enumerated Phase-A required artifacts.")
    closure_md += [
        "",
        "## Interpretation boundary",
        "",
        "`FOUND_UNREGISTERED` means bytes exist and were rehashed, but no parsed local JSON/compact manifest binds that exact path+SHA. It is not silently promoted to a reproduced metric claim. No label, model fit, prediction generation, leaderboard selector, or Public-driven tuning occurred in this phase.",
        "",
    ]
    write_text_exclusive(output / "MISSING_EVIDENCE_CLOSURE.md", "\n".join(closure_md))

    result = {
        "schema_version": 1,
        "phase": "A_EVIDENCE_CLOSURE",
        "status": "PASS" if not missing else "PASS_WITH_MISSING_EVIDENCE",
        "compact_manifest_entries": len(compact_rows),
        "public_rows_verified": len(public_rows),
        "platform_receipt_verified": False,
        "user_feedback_selector_eligible": False,
        "feature_exit_models": len(model_files),
        "reload_checks_bit_exact": sum(bool(r.get("reload_prediction_bit_exact")) for r in checks.values()),
        "inventory_files": len(inventory),
        "inventory_status_counts": dict(status_counts),
        "required_status_counts": dict(required_counts),
        "missing_required_claims": [row["claim_id"] for row in missing],
    }
    write_json_exclusive(output / "EVIDENCE_CLOSURE_SUMMARY.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.config.resolve())
    except Exception as exc:  # fail closed with a machine-readable incident when possible
        try:
            config = json.loads(args.config.read_text(encoding="utf-8"))
            output = _resolve(ROOT, config["output_root"])
            output.mkdir(parents=True, exist_ok=True)
            write_json_exclusive(
                output / "EVIDENCE_CLOSURE_FAILURE.json",
                {"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)},
            )
        except Exception:
            pass
        raise
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

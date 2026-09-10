"""Freeze the postrun-safe source-only supersession for ordinal Bayes Stage2."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_ficr_bayes_decision_strict as strict  # noqa: E402
from scripts import run_full_weather_ordinal_bayes as stage1  # noqa: E402
from scripts import run_full_weather_ordinal_bayes_stage2_final as producer  # noqa: E402
from scripts import run_shared_q07_multiseed as shared  # noqa: E402
from src.manifest import describe_file, sha256_file, utc_now  # noqa: E402


CANONICAL = ROOT / "artifacts/postgate/full_weather_ordinal_bayes_strict_v1"
REPORT = ROOT / "artifacts/audits/full_weather_ordinal_bayes_g3_stage2_postrun_v2.json"
STAGE1_MANIFEST_SHA = producer.STAGE1_MANIFEST_SHA256
V1_APPEND_MANIFEST_SHA = "a443c2ab714053d0a2f23b6f7de5d21e1498ae6161c61341df93835b7c2325e1"
LOCAL_NUMERIC_AUDIT_SHA = "ea9272bd615b39fac7340256dca8821746a5c5e6dddc2eaa164607511e617175"


def _tree_snapshot() -> dict[str, dict]:
    return {
        path.relative_to(CANONICAL).as_posix(): describe_file(path)
        for path in sorted(CANONICAL.rglob("*"))
        if path.is_file()
    }


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(REPORT)
    before = _tree_snapshot()
    stage1_manifest = CANONICAL / "manifest.json"
    append_manifest = CANONICAL / "manifest_stage2_final_g3.json"
    if sha256_file(stage1_manifest) != STAGE1_MANIFEST_SHA:
        raise AssertionError("Stage1 manifest changed")
    if sha256_file(append_manifest) != V1_APPEND_MANIFEST_SHA:
        raise AssertionError("v1 append manifest changed")
    append = json.loads(append_manifest.read_text(encoding="utf-8"))
    for record in append["outputs"]:
        path = Path(record["path"])
        if path.stat().st_size != record["size_bytes"] or sha256_file(path) != record["sha256"]:
            raise AssertionError(f"numeric/output artifact changed: {path}")
    local_audit = ROOT / "artifacts/audits/full_weather_ordinal_bayes_g3_stage2_v1_local.json"
    if sha256_file(local_audit) != LOCAL_NUMERIC_AUDIT_SHA:
        raise AssertionError("local numeric audit changed")

    command = [
        str(ROOT / ".venv/Scripts/python.exe"),
        "-m",
        "pytest",
        "tests/test_full_weather_ordinal_bayes.py",
        "tests/test_full_weather_ordinal_bayes_runner.py",
        "tests/test_full_weather_ordinal_bayes_stage2_final.py",
        "-q",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or "23 passed" not in completed.stdout:
        raise AssertionError(
            f"postrun focused tests failed: {completed.stdout}\n{completed.stderr}"
        )
    closure = stage1.protocol._static_repo_import_closure((Path(producer.__file__).resolve(),))
    current_source = {
        f"repo_source::{path.relative_to(ROOT).as_posix()}": shared._snapshot_file(path)
        for path in closure
    }
    for name, path in {
        "stage2_test": ROOT / "tests/test_full_weather_ordinal_bayes_stage2_final.py",
        "audit_script": Path(__file__).resolve(),
        "preregister": ROOT / "configs/full_weather_ordinal_bayes_preregister_v1.json",
    }.items():
        current_source[name] = shared._snapshot_file(path)

    after = _tree_snapshot()
    if before != after:
        raise AssertionError("canonical tree changed during postrun audit")
    result = json.loads((CANONICAL / "stage2_results_g3.json").read_text(encoding="utf-8"))
    report = {
        "schema_version": 2,
        "audit_id": "full_weather_ordinal_bayes_g3_stage2_postrun_v2",
        "created_utc": utc_now(),
        "status": "AUTHORITATIVE_SOURCE_SUPERSESSION_NUMERIC_REJECT_UNCHANGED",
        "canonical": str(CANONICAL.resolve()),
        "preregister_sha256": producer.PREREGISTER_SHA256,
        "frozen_stage1_manifest": describe_file(stage1_manifest),
        "v1_append_manifest": describe_file(append_manifest),
        "v1_append_manifest_numeric_and_output_records_valid": True,
        "v1_append_manifest_source_snapshot_superseded": True,
        "local_numeric_audit": describe_file(local_audit),
        "current_source_and_test_snapshot": current_source,
        "canonical_tree_snapshot": before,
        "canonical_tree_nonmutation": True,
        "postrun_focused_tests": {
            "command": command,
            "returncode": completed.returncode,
            "passed": 23,
            "stdout": completed.stdout.strip(),
        },
        "incident": {
            "kind": "postrun_append_closure_verifier_rejected_registered_stage2_outputs",
            "initial_postrun_test_result": "7 passed, 1 failed",
            "failure_scope": "Stage1 closure verifier treated the already registered append-only Stage2 output set as unexpected extras.",
            "model_fit_rerun": False,
            "probability_or_candidate_rerun": False,
            "metric_rerun_for_selection": False,
            "formula_weight_bin_class_calibration_or_group_change": False,
            "fix": "Source-only verifier now admits exactly the files and hashes registered by the immutable v1 append manifest while still rejecting every unregistered extra or missing Stage1 file.",
            "postrun_test_result": "23 passed",
        },
        "numeric_outcome": {
            "stage2_candidate_promoted": result["candidate_promoted"],
            "primary_v3_pass": result["dual_baseline_comparisons"]["primary_v3"]["gate"]["passed"],
            "recent_v4_pass": result["dual_baseline_comparisons"]["recent_v4"]["gate"]["passed"],
            "final_or_csv_created": False,
            "no_retune_or_rescue": True,
        },
        "authoritative_for": "current producer source/test closure and postrun-safe contract only; all frozen model, probability, candidate, locks, metrics, and reject outcome remain those in v1 append manifest",
    }
    strict._write_json(REPORT, report)
    if _tree_snapshot() != before:
        raise AssertionError("canonical tree changed while writing external report")
    print(f"postrun v2 PASS {sha256_file(REPORT)}", flush=True)


if __name__ == "__main__":
    main()

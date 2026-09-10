"""Freeze an append-only authoritative manifest after the post-run test fix."""

from __future__ import annotations

import json
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts import run_public_adaptive_scale097_g2_delta as producer  # noqa: E402
from src.manifest import describe_file, git_state, package_versions, sha256_file, utc_now, write_json_atomic  # noqa: E402


ROOT = PROJECT_DIR / "artifacts/postgate/public_adaptive_scale097_g2_delta_v2"
ADDENDUM = PROJECT_DIR / "configs/public_adaptive_scale097_g2_delta_postrun_test_addendum_20260808.json"
ADDENDUM_SHA = "cfde6cfd71ceab0f35d32239e0c17dcab04ebbbbf0bdbdc05d75547fb127e6d8"


def main() -> None:
    if sha256_file(ADDENDUM) != ADDENDUM_SHA:
        raise AssertionError("postrun addendum hash differs")
    addendum = json.loads(ADDENDUM.read_text(encoding="utf-8"))
    historical = ROOT / "manifest.json"
    if sha256_file(historical) != addendum["historical_manifest"]["sha256"]:
        raise AssertionError("historical manifest differs")
    test = PROJECT_DIR / addendum["fix"]["current_test"]["path"]
    if test.stat().st_size != int(addendum["fix"]["current_test"]["bytes"]) or sha256_file(test) != addendum["fix"]["current_test"]["sha256"]:
        raise AssertionError("current postrun test differs")
    v2_config = PROJECT_DIR / "configs/public_adaptive_scale097_g2_delta_preregister_v2.json"
    v1_config = PROJECT_DIR / "configs/public_adaptive_scale097_g2_delta_preregister_v1.json"
    args = producer.parse_args([
        "--config", str(v2_config), "--v1-config", str(v1_config), "--out-dir", str(ROOT)
    ])
    current_producer_closure = producer._closure(args)
    original = json.loads(historical.read_text(encoding="utf-8"))
    generation_closure = original["source_provenance"]["recursive_AST_closure"]
    if current_producer_closure["resolved_relative_paths"] != generation_closure["resolved_relative_paths"]:
        raise AssertionError("producer runtime import closure set changed")
    for key in ("A_csv", "B_csv"):
        csv_name = (
            "corrected_recent_v4_scale_097_public_adaptive_2025.csv"
            if key == "A_csv" else
            "corrected_recent_v4_scale_097_plus_g2_rescue_delta_2025.csv"
        )
        if sha256_file(ROOT / csv_name) != addendum["immutable_output_hashes"][key]:
            raise AssertionError(f"immutable output changed: {key}")
    destination = ROOT / "manifest_frozen_postrun_v2.json"
    if destination.exists():
        raise FileExistsError(destination)
    files = sorted(
        (path for path in ROOT.rglob("*") if path.is_file() and path != destination),
        key=lambda path: path.relative_to(ROOT).as_posix(),
    )
    payload = {
        "schema_version": 1,
        "artifact_type": "public_adaptive_scale097_plain_g2_delta_postrun_test_supersession",
        "created_utc": utc_now(),
        "authoritative_manifest": True,
        "supersedes_historical_manifest_for_current_test_provenance": describe_file(historical),
        "postrun_test_addendum": describe_file(ADDENDUM),
        "postrun_test_addendum_sidecar": describe_file(ADDENDUM.with_suffix(".sha256")),
        "candidate_and_CSV_recomputed": False,
        "candidate_and_CSV_bytes_unchanged": addendum["immutable_output_hashes"],
        "generation_source_closure_historical": generation_closure,
        "current_producer_source_config_test_closure": current_producer_closure,
        "runtime_source_path_set_unchanged": True,
        "test_change_only": addendum["fix"],
        "focused_postrun_tests": {"passed": 7, "failed": 0, "command": ".venv/Scripts/python.exe -m pytest -q tests/test_public_adaptive_scale097_g2_delta.py"},
        "diagnostic_recommendation": "A_over_B",
        "risk": original["risk"],
        "strict_or_private_claim": False,
        "historical_independent_audit": addendum["historical_local_audit"],
        "outputs_and_historical_records": [describe_file(path) for path in files],
        "output_count_excluding_this_manifest": len(files),
        "runtime": {"packages": package_versions()},
        "git": git_state(PROJECT_DIR),
    }
    write_json_atomic(destination, payload, overwrite=False)
    print(f"frozen_postrun_manifest_sha256={sha256_file(destination)}")


if __name__ == "__main__":
    main()

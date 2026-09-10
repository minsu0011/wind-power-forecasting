#!/usr/bin/env python
"""Seal Track-B v2 code identity and STOP before any numeric experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy
import pandas
import pyarrow
import scipy
import sklearn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


OUT_DIR = ROOT / "artifacts/baram2026_ncei_scada_longrun_20260810_v2/track_b"
PREREG = OUT_DIR / "TRACK_B_SOFT_CONFIDENCE_PREREGISTER_V2.json"
ROOTCAUSE = OUT_DIR / "TRACK_B_V2_ROOTCAUSE_DECOMPOSITION.json"
DUPLICATE = OUT_DIR / "TRACK_B_V2_DUPLICATE_AUDIT.json"
PREVIOUS_DECISION = ROOT / "artifacts/baram2026_ncei_scada_research_20260810_210756/track_b/TRACK_B_DECISION.json"
WEIGHT_MODULE = ROOT / "src/scada_soft_confidence_v2.py"
METRIC_MODULE = ROOT / "src/metric.py"
FOCUSED_TEST = ROOT / "tests/test_scada_soft_confidence_v2.py"
ATTEMPT1_INCIDENT = OUT_DIR / "incidents/attempt1_code_only_bool_nameerror/INCIDENT.json"
ATTEMPT2_INCIDENT = OUT_DIR / "incidents/attempt2_crlf_identity_mismatch/INCIDENT.json"
ATTEMPT2_PARTIAL = OUT_DIR / "incidents/attempt2_crlf_identity_mismatch/TRACK_B_V2_DECISION.partial.json"
DECISION_OUTPUT = OUT_DIR / "TRACK_B_V2_DECISION.json"
MANIFEST_OUTPUT = OUT_DIR / "TRACK_B_V2_CODE_IDENTITY_MANIFEST.json"

FROZEN_IDENTITIES: dict[Path, tuple[int, str]] = {
    PREREG: (6278, "c95ccccb5419e59edd480f8d10a7d26db3dd77506287a0ff409fe22750d08d8c"),
    ROOTCAUSE: (4184, "865450f6d7daa7c870669a4b8872d4673f0423d43e79ab5b42df367a0be4bb4a"),
    DUPLICATE: (3121, "14736fd7cd9c80cfd7489e14c82501015e3ca4ef7212570113c1fa05c9132254"),
    PREVIOUS_DECISION: (4074, "6cc68e2ae86b9f7b9fd9066096c8964d3b3e0b0e62eb436c0994c890a60c15f8"),
    WEIGHT_MODULE: (3710, "f2e9665db052365c49b910e5aa7fc9cd4275b2add5cd23c0045a4e00a26ace61"),
    METRIC_MODULE: (7248, "555950e6892a808d9091b4e6749128b9f96888ce0a83270b6959a4e927dc5f1d"),
    FOCUSED_TEST: (4497, "f10970aed6e42434c8d5dbdd46732bb01a7d6ec05403c7ce58ae2a091a8aa3da"),
    ATTEMPT1_INCIDENT: (989, "5ff7887a4b198f1e95f1e68ba0a4fe681f20abae4dead2dc64f2e4daba45f5b9"),
    ATTEMPT2_INCIDENT: (1335, "64a2a17d7abb19c17bdacd4884cbfdd7f3ed3f20c5451c5aaa98ab4efa622b41"),
    ATTEMPT2_PARTIAL: (1087, "84b102ac65135ac84b9623d50e17488ba5f821dd58f03ce6be3b524d6cfa5ca5"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def content_record(path: Path, role: str, content: str) -> dict[str, Any]:
    encoded = content.encode("utf-8")
    return {
        "role": role,
        "path": str(path.resolve()),
        "size_bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def write_new_utf8(path: Path, content: str) -> None:
    """Write exact UTF-8 bytes with exclusive creation and no newline translation."""

    encoded = content.encode("utf-8")
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()


def verify_frozen() -> None:
    for path, (expected_size, expected_sha) in FROZEN_IDENTITIES.items():
        if not path.is_file():
            raise RuntimeError(f"missing frozen input: {path}")
        if path.stat().st_size != expected_size or sha256(path) != expected_sha:
            raise RuntimeError(f"frozen identity mismatch: {path}")


def validate_identity_closure(records: list[dict[str, Any]]) -> None:
    roles = [str(record["role"]) for record in records]
    paths = [str(record["path"]) for record in records]
    if len(roles) != len(set(roles)):
        raise RuntimeError("duplicate implementation identity role")
    if len(paths) != len(set(paths)):
        raise RuntimeError("duplicate implementation identity path")
    if not all(len(str(record["sha256"])) == 64 for record in records):
        raise RuntimeError("invalid implementation identity digest")


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--code-only",
        action="store_true",
        help="write identity/STOP artifacts without any label, model, prediction or metric access",
    )
    mode.add_argument(
        "--runtime-smoke",
        action="store_true",
        help="execute frozen verification and terminal-payload construction without writing outputs",
    )
    args = parser.parse_args()
    if OUT_DIR.resolve() != (ROOT / "artifacts/baram2026_ncei_scada_longrun_20260810_v2/track_b").resolve():
        raise RuntimeError("canonical output root mismatch")
    if args.code_only and (DECISION_OUTPUT.exists() or MANIFEST_OUTPUT.exists()):
        raise RuntimeError("no-overwrite guard: v2 code-only output already exists")

    verify_frozen()

    # transitive metric dependency have been verified.
    from src.scada_soft_confidence_v2 import scientific_prerequisite_decision

    previous = json.loads(PREVIOUS_DECISION.read_text(encoding="utf-8"))
    state_gate = {
        group: bool(previous["state_reliability"][group]["pass"])
        for group in ("G1", "G2", "G3")
    }
    decision = scientific_prerequisite_decision(state_gate)
    if decision != "TRACK_B_V2_STOP_INVALID_PROXY_PREREQUISITE":
        raise RuntimeError("current immutable state unexpectedly passed; independent amendment required")

    decision_payload = {
        "schema_version": 1,
        "artifact_type": "track_b_v2_code_only_terminal_decision",
        "decision": decision,
        "state_reliability_prerequisite": state_gate,
        "scientific_reason": "G3 proxy semantics failed the frozen reliability gate; a soft transform cannot validate that proxy and G3 omission is forbidden.",
        "hard_filter_retired": True,
        "only_dormant_candidate": "fixed all-row continuous soft confidence weight from the frozen preregistration",
        "unregistered_minimum_normal_rows_retired": True,
        "minimum_normal_rows": None,
        "all_eligible_rows_must_be_retained": True,
        "numeric_execution_authorized": False,
        "new_numeric_effects": None,
        "counters": {
            "v2_scada_or_proxy_rows_read": 0,
            "v2_official_label_rows_read": 0,
            "model_fit_calls": 0,
            "prediction_calls": 0,
            "metric_calls": 0,
            "materialized_2024_rows": 0,
            "test_rows": 0,
            "materialized_2025_rows": 0,
            "public_feedback_reads": 0,
            "submission_csv_files": 0,
        },
    }
    decision_text = json.dumps(decision_payload, ensure_ascii=False, indent=2) + "\n"

    implementation = [
        file_record(Path(__file__), "runner_self"),
        file_record(WEIGHT_MODULE, "local_import_scada_soft_confidence_v2"),
        file_record(METRIC_MODULE, "transitive_local_import_metric_capacity"),
    ]
    validate_identity_closure(implementation)
    manifest = {
        "schema_version": 1,
        "artifact_type": "track_b_v2_code_identity_manifest",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "terminal_decision": decision,
        "implementation_identity": implementation,
        "science_contract": file_record(PREREG, "science_preregister"),
        "evidence_identity": [
            file_record(ROOTCAUSE, "readonly_rootcause"),
            file_record(DUPLICATE, "duplicate_audit"),
            file_record(PREVIOUS_DECISION, "immutable_previous_decision"),
            file_record(ATTEMPT1_INCIDENT, "prefit_code_only_attempt1_incident"),
            file_record(ATTEMPT2_INCIDENT, "code_only_attempt2_incident"),
            file_record(ATTEMPT2_PARTIAL, "code_only_attempt2_preserved_partial"),
        ],
        "verification_identity": [file_record(FOCUSED_TEST, "focused_test_suite")],
        "terminal_output": content_record(
            DECISION_OUTPUT, "terminal_decision_output", decision_text
        ),
        "runtime_identity": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "platform": platform.platform(),
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "pyarrow": pyarrow.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
        "identity_contract": {
            "roles_unique": True,
            "paths_unique": True,
            "all_sizes_and_sha256_verified_before_output": True,
            "runner_self_bound": True,
            "all_local_imports_bound": True,
            "package_versions_bound": True,
        },
        "support_contract": {
            "hidden_minimum_row_thresholds": 0,
            "hard_filter": False,
            "all_eligible_rows_retained_if_future_execution_authorized": True,
            "positive_weight_floor_registered": 0.5,
        },
        "access_counters": decision_payload["counters"],
        "generated_model_files": [],
        "generated_prediction_files": [],
        "generated_csv_files": [],
    }
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    if args.runtime_smoke:
        print(json.dumps({
            "status": "RUNTIME_SMOKE_PASS",
            "decision": decision,
            "decision_record": manifest["terminal_output"],
            "manifest_bytes": len(manifest_text.encode("utf-8")),
        }, indent=2))
        return
    write_new_utf8(DECISION_OUTPUT, decision_text)
    if file_record(DECISION_OUTPUT, "terminal_decision_output") != manifest["terminal_output"]:
        raise RuntimeError("terminal decision write differs from precomputed identity")
    write_new_utf8(MANIFEST_OUTPUT, manifest_text)
    if file_record(MANIFEST_OUTPUT, "code_identity_manifest")["size_bytes"] != len(
        manifest_text.encode("utf-8")
    ):
        raise RuntimeError("manifest exact-byte write differs")
    print(json.dumps({"decision": decision, "manifest": file_record(MANIFEST_OUTPUT, "code_identity_manifest")}, indent=2))


if __name__ == "__main__":
    main()

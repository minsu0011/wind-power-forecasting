from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "artifacts" / "baram2026_ncei_scada_longrun_20260810_v2"


def assert_record(record: dict) -> None:
    path = Path(record["path"])
    target = path if path.is_absolute() else ROOT / path
    payload = target.read_bytes()
    assert len(payload) == record["size_bytes"]
    assert hashlib.sha256(payload).hexdigest() == record["sha256"]


def test_v5_binds_exact_independent_windows_results_append_only() -> None:
    path = ROOT / "prelaunch" / "RAW_RUNNER_STATIC_PREFLIGHT_AMENDMENT_V5.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["independent_windows_test_result"] == "25 passed"
    assert payload["py_compile_result"] == "PASS"
    assert payload["independent_windows_test"]["network_requests"] == 0
    assert payload["v4_artifacts_immutable"] is True
    assert payload["authorization_created"] is False
    assert payload["independent_go_present"] is False
    for key in ("parent_manifest", "runner", "runner_test"):
        assert_record(payload[key])


def test_v5_draft_amendment_requires_future_audit_and_go_binding() -> None:
    path = ROOT / "prelaunch" / "RAW_LAUNCH_AUTHORIZATION_DRAFT_AMENDMENT_V2.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    requirements = payload["final_executable_authorization_requirements"]
    assert requirements["must_not_copy_v4_proposed_payload_verbatim"] is True
    assert requirements["must_bind_effective_preflight_manifest_v5"] is True
    assert requirements["must_bind_forthcoming_independent_prelaunch_audit_identity"] is True
    assert requirements["independent_prelaunch_audit_status_must_be_pass"] is True
    assert requirements["final_authorization_sha_must_be_bound_by_independent_go"] is True
    assert requirements["independent_windows_test_result"] == "25 passed"
    assert requirements["py_compile_result"] == "PASS"
    assert payload["actual_authorization_created"] is False
    assert payload["independent_go_created"] is False
    for key in ("parent_authorization_draft", "effective_test_result_amendment"):
        assert_record(payload[key])


def test_v5_manifest_rehashes_chain_and_keeps_zero_state() -> None:
    path = ROOT / "manifest_raw_preflight_v5.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["independent_windows_test_result"] == "25 passed"
    assert payload["py_compile_result"] == "PASS"
    for key in (
        "parent_manifest",
        "amendment",
        "authorization_draft_amendment",
        "runner",
        "runner_test",
        "seal_code",
        "seal_test",
    ):
        assert_record(payload[key])
    assert payload["authorization_created"] is False
    assert payload["independent_go_present"] is False
    assert not (ROOT / "prereg" / "raw_launch_authorization_v1.json").exists()
    assert not (
        ROOT / "independent_redteam" / "TRACK_A_RAW_LAUNCH_GO.json"
    ).exists()
    assert not (ROOT / "raw").exists()
    assert not (ROOT / "decoded").exists()

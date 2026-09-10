from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


ROOT = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "baram2026_ncei_scada_longrun_20260810_v2"
)


def test_raw_preflight_v2_is_append_only_and_not_authorized() -> None:
    path = ROOT / "manifest_raw_preflight_v2.json"
    if not path.is_file():
        pytest.skip("raw preflight v2 not sealed")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["parent_manifest"]["sha256"] == (
        "b2fc180579f239730e0ca01582fc6d3dfbb62113447a0a24eefca476326f38ad"
    )
    assert manifest["authorization_created"] is False
    assert manifest["raw_network_requests"] == 0
    for key in ("parent_manifest", "amendment"):
        record = manifest[key]
        target = ROOT / record["path"] if not Path(record["path"]).is_absolute() else Path(record["path"])
        payload = target.read_bytes()
        assert len(payload) == record["size_bytes"]
        assert hashlib.sha256(payload).hexdigest() == record["sha256"]
    # v2 pointed at mutable working source paths. The reviewed v4/v5 append-only
    # chain explicitly supersedes those stale code identities while preserving
    # the immutable v2 artifact and parent hash.
    v4_path = ROOT / "prelaunch" / "RAW_RUNNER_STATIC_PREFLIGHT_AMENDMENT_V4.json"
    v5_path = ROOT / "manifest_raw_preflight_v5.json"
    assert v4_path.is_file() and v5_path.is_file()
    v4 = json.loads(v4_path.read_text(encoding="utf-8"))
    assert v4["parent_manifest"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert "never created v3 artifacts" in v4["version_note"]
    assert v4["runner"]["sha256"] != manifest["corrected_runner"]["sha256"]
    assert v4["runner_test"]["sha256"] != manifest["corrected_runner_test"]["sha256"]


def test_completed_part_recovery_contract_is_exact_and_fail_closed() -> None:
    path = ROOT / "prelaunch" / "RAW_RUNNER_STATIC_PREFLIGHT_AMENDMENT_V2.json"
    if not path.is_file():
        pytest.skip("raw preflight v2 not sealed")
    amendment = json.loads(path.read_text(encoding="utf-8"))
    correction = amendment["correction"]
    assert correction["synthetic_exact_crash_recovery_test"] == "PASS"
    assert correction["synthetic_tampered_payload_test"] == "PASS_FAIL_CLOSED"
    assert correction["ambiguous_missing_or_tampered_event"] == "STOP_FAIL_CLOSED"
    assert amendment["authorization_created"] is False
    assert amendment["independent_go_present"] is False

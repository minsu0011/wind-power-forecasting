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


def load(relative: str) -> dict:
    path = ROOT / relative
    if not path.is_file():
        pytest.skip("raw static preflight is not sealed")
    return json.loads(path.read_text(encoding="utf-8"))


def test_cumulative_access_includes_initial_population_and_raw_zero() -> None:
    audit = load("audit/CENSUS_CUMULATIVE_ACCESS_AMENDMENT.json")
    phase = audit["phase_total"]
    assert phase["successful_http_requests"] == 1_200
    assert phase["successful_payload_bytes"] == 52_516_501
    assert phase["raw_range_requests"] == 0
    assert phase["raw_range_bytes"] == 0
    assert phase["transport_attempt_total_exact"] is False
    assert phase["content_and_successful_request_total_exact"] is True


def test_preflight_binds_both_independent_gates_and_hardening() -> None:
    preflight = load("prelaunch/RAW_RUNNER_STATIC_PREFLIGHT.json")
    assert preflight["status"] == "READY_FOR_INDEPENDENT_RUNNER_REVIEW_NOT_AUTHORIZED"
    assert preflight["bound_inputs"]["independent_prereg"]["sha256"] == (
        "3605f88682060fe73faedad57852ee357bc6243489eabc00022533b4b81e5630"
    )
    assert preflight["bound_inputs"]["independent_census_audit"]["sha256"] == (
        "7fc717406e6e7587530cc922a16bf785b4471f99f0488c7483b2abc72890af68"
    )
    assert preflight["raw_actual_http_attempt_budget"] == 18_800
    assert all(preflight["hardening"].values())
    assert preflight["all_expected_outputs_absent"] is True
    assert preflight["network_requests_by_this_seal"] == 0


def test_preflight_manifest_hash_closes_all_new_artifacts() -> None:
    manifest = load("manifest_raw_preflight_v1.json")
    assert manifest["raw_authorization_created"] is False
    assert manifest["raw_network_requests"] == 0
    for key in ("census_manifest", "cumulative_access_amendment", "incident", "static_preflight"):
        record = manifest[key]
        payload = (ROOT / record["path"]).read_bytes()
        assert len(payload) == record["size_bytes"]
        assert hashlib.sha256(payload).hexdigest() == record["sha256"]


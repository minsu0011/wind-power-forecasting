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


def test_global_attempt_budget_is_mathematically_bounded_append_only() -> None:
    path = ROOT / "prelaunch" / "RAW_RUNNER_STATIC_PREFLIGHT_AMENDMENT_V3.json"
    if not path.is_file():
        pytest.skip("raw preflight v3 not sealed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["parent_manifest"]["sha256"] == (
        "47ddc9df6e4346df20ed259de093e96a88dc2202424ff3f36ccceaff025d59a3"
    )
    assert payload["census"]["actual_attempt_upper_bound"] == 4_800
    assert payload["raw"]["actual_attempt_budget"] == 15_200
    assert payload["raw"]["retry_headroom_after_one_success_per_range"] == 4_832
    assert (
        payload["census"]["actual_attempt_upper_bound"]
        + payload["raw"]["actual_attempt_budget"]
        == payload["global_actual_http_attempt_cap"]
    )
    assert payload["authorization_created"] is False
    assert payload["raw_network_requests"] == 0


def test_preflight_v3_manifest_hash_closes_parent_and_budget_amendment() -> None:
    path = ROOT / "manifest_raw_preflight_v3.json"
    if not path.is_file():
        pytest.skip("raw preflight v3 not sealed")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["effective_raw_actual_http_attempt_budget"] == 15_200
    assert manifest["authorization_created"] is False
    for key in ("parent_manifest", "amendment", "runner", "runner_test"):
        record = manifest[key]
        target = ROOT / record["path"] if not Path(record["path"]).is_absolute() else Path(record["path"])
        data = target.read_bytes()
        assert len(data) == record["size_bytes"]
        assert hashlib.sha256(data).hexdigest() == record["sha256"]


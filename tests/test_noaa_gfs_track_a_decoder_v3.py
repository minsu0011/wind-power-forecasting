from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


ROOT = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "baram2026_ncei_scada_research_20260810_210756"
    / "track_a"
)


def test_append_only_decoder_correction_is_directly_evidenced() -> None:
    path = ROOT / "audit" / "HPBL_ECCODES_DECODER_AUDIT.json"
    if not path.is_file():
        pytest.skip("decoder amendment is not present")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["previous_operational_decoder_available"] is False
    assert report["corrected_operational_decoder_available"] is True
    assert report["direct_decode"]["decode_success"] is True
    assert report["direct_decode"]["decoder"] == "python-eccodes-direct"
    assert report["direct_decode"]["value_count"] == 1_038_240
    assert report["direct_decode"]["metadata"]["forecastTime"] == 39
    assert report["direct_decode"]["metadata"]["validityDate"] == 20230703
    assert report["three_hour_verdict_corrected"] == "STOP_3H_HONEST_MODEL"
    assert report["network_requests"] == 0
    assert report["downloaded_bytes"] == 0
    assert report["labels_read"] is False
    assert report["models_fit"] == 0


def test_manifest_v3_binds_v2_and_correction_without_mutation() -> None:
    path = ROOT / "manifest_v3.json"
    if not path.is_file():
        pytest.skip("decoder amendment is not present")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["parent_manifest"]["sha256"] == (
        "7d2d9478b349fd44f4f60874c3a16bb93a9d94c2f5969ba7ee2a2f3eccd05296"
    )
    assert manifest["existing_files_modified"] == []
    assert manifest["network_requests"] == 0
    for key in ("parent_manifest", "previous_feasibility", "correction", "correction_markdown"):
        record = manifest[key]
        payload = (ROOT / record["path"]).read_bytes()
        assert len(payload) == record["size_bytes"]
        assert hashlib.sha256(payload).hexdigest() == record["sha256"]


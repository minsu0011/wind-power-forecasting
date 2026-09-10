from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/download_noaa_gefs_operational_spread_00z_original_v12.py"


def load_module():
    name = "gefs_v12_extractor_postrun_under_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_preflight_postrun_or_clean_zero_state():
    module = load_module()
    if not module.PREFLIGHT_OUTPUT.exists():
        assert not module.PREFLIGHT_TOMBSTONE.exists()
        assert not module.PREFLIGHT_LOCK.exists()
        pytest.skip("v12 semantic preflight not authorized/executed")
    closure = module.verify_preflight_canonical()
    assert closure["manifest"]["sha256"]
    assert closure["result"]["sha256"]
    assert len(closure["responses"]) == 4
    result = json.loads((module.PREFLIGHT_OUTPUT / "preflight_result.json").read_text(encoding="utf-8"))
    assert result["request_accounting"]["requests"] == 4
    assert result["request_accounting"]["response_bytes"] == 1_765_921
    assert result["expected_derivedForecast_by_range"] == [0, 0, 2, 2]
    assert result["automatic_full_launch"] is False
    assert result["full_extraction_payload_reuse_allowed"] is False


def test_full_source_postrun_or_not_authorized():
    module = load_module()
    if not module.OUTPUT_ROOT.exists():
        assert not module.SOURCE_TOMBSTONE.exists()
        return
    summary = json.loads((module.OUTPUT_ROOT / "extraction_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "PASS_V12_STAGE1_SOURCE_EXTRACTION"
    assert summary["scope"]["ranges"] == 26_280
    assert summary["scope"]["raw_bytes"] == 11_524_284_739
    assert summary["request_accounting"]["preflight_payload_reused"] == 0
    assert summary["request_accounting"]["operating_2024_grib_range_gets_or_values"] == 0
    assert summary["request_accounting"]["2025_requests_or_values"] == 0
    assert summary["request_accounting"]["label_reads"] == 0
    assert summary["request_accounting"]["fits"] == 0
    assert summary["request_accounting"]["scores"] == 0


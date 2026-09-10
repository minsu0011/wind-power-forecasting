from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import threading
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/download_noaa_gefs_operational_spread_00z_original_v12.py"


def load_module():
    name = "gefs_v12_extractor_under_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def module():
    return load_module()


def test_frozen_lineage_plan_and_four_cell_contract(module):
    prereg = module.verify_v12_lineage()
    assert prereg["schema_version"] == 12
    plans, pack_sizes = module.build_stage1_plan()
    assert len(plans) == 26_280
    assert sum(pack_sizes.values()) == 11_524_284_739
    fixed = module.fixed_preflight_plans(plans)
    assert [item[0] for item in fixed] == ["mean_10m", "mean_850", "stddev_10m", "stddev_850"]
    assert [item[2] for item in fixed] == [0, 0, 2, 2]
    assert sum(item[1].byte_count for item in fixed) == 1_765_921
    assert [item[1].level_id for item in fixed] == ["10m", "850", "10m", "850"]
    assert [item[1].lead for item in fixed] == [39, 39, 39, 39]
    assert [item[1].statistic for item in fixed] == ["ens mean", "ens mean", "ens std dev", "ens std dev"]


def test_static_cli_is_real_nonempty_pass_and_unknown_arg_fails():
    good = subprocess.run(
        [sys.executable, "-B", str(SCRIPT), "--static-audit"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert good.returncode == 0
    assert good.stderr == ""
    lines = [line for line in good.stdout.splitlines() if line.strip()]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["status"] == "PASS_V12_STATIC_SOURCE_CONTRACT"
    assert payload["semantic_preflight_expected_derivedForecast"] == [0, 0, 2, 2]
    assert payload["semantic_preflight_requests"] == 4
    assert payload["semantic_preflight_concurrency"] == 1
    assert payload["semantic_preflight_maximum_attempts_per_range"] == 1
    assert payload["v12_network_value_label_fit_score_access"] == 0

    bad = subprocess.run(
        [sys.executable, "-B", str(SCRIPT), "--not-a-real-option"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert bad.returncode != 0


def fake_metadata(plan, name: str, derived: int) -> dict:
    return {
        "edition": 2,
        "shortName": name,
        "typeOfLevel": "heightAboveGround" if plan.level_id == "10m" else "isobaricInhPa",
        "level": 10 if plan.level_id == "10m" else 850,
        "dataDate": int(plan.source_date.replace("-", "")),
        "dataTime": 0,
        "forecastTime": plan.lead,
        "stepType": "instant",
        "derivedForecast": derived,
        "units": "m/s",
        "gridType": "regular_ll",
        "Ni": 720,
        "Nj": 361,
        "iDirectionIncrementInDegrees": 0.5,
        "jDirectionIncrementInDegrees": 0.5,
        "latitudeOfFirstGridPointInDegrees": 90.0,
        "latitudeOfLastGridPointInDegrees": -90.0,
        "longitudeOfFirstGridPointInDegrees": 0.0,
        "longitudeOfLastGridPointInDegrees": 359.5,
        "productDefinitionTemplateNumber": 2,
    }


def test_official_mean_zero_stddev_two_and_code_four_fails_plan_bound(module, monkeypatch):
    plans, _ = module.build_stage1_plan()
    fixed = module.fixed_preflight_plans(plans)
    import eccodes

    state: dict[str, object] = {}
    monkeypatch.setattr(eccodes, "codes_get", lambda _gid, key: state[key])
    monkeypatch.setattr(eccodes, "codes_grib_find_nearest", lambda *_args, **_kwargs: [{"lat": 37.5, "lon": 129.0, "value": 1.25}])

    for _cell, plan, expected in fixed:
        name = "10u" if plan.level_id == "10m" else "u"
        state.clear()
        state.update(fake_metadata(plan, name, expected))
        metadata, value = module._decode_handle(object(), plan, name)
        assert metadata["derivedForecast"] == expected
        assert value == 1.25

    _cell, spread_plan, _expected = fixed[2]
    state.clear()
    state.update(fake_metadata(spread_plan, "10u", 4))
    with pytest.raises(RuntimeError) as caught:
        module._decode_handle(object(), spread_plan, "10u")
    message = str(caught.value)
    assert "GEFS_PLAN_VALIDATION_ERROR" in message
    assert '"expected_derivedForecast":2' in message
    assert '"observed_derivedForecast":4' in message
    assert json.dumps(asdict(spread_plan), sort_keys=True, separators=(",", ":")) in message


def test_pre_network_review_is_required_before_lock(module, tmp_path, monkeypatch):
    plans, _ = module.build_stage1_plan()
    monkeypatch.setattr(module, "INDEPENDENT_REVIEW", tmp_path / "absent_review.json")
    monkeypatch.setattr(module, "PREFLIGHT_LOCK", tmp_path / "lock.json")
    monkeypatch.setattr(module, "PREFLIGHT_LOCK_SHA", tmp_path / "lock.json.sha256")
    monkeypatch.setattr(module, "PREFLIGHT_TOMBSTONE", tmp_path / "attempt.json")
    monkeypatch.setattr(module, "PREFLIGHT_OUTPUT", tmp_path / "output")
    with pytest.raises(FileNotFoundError):
        module.freeze_preflight_lock(plans)
    assert not (tmp_path / "lock.json").exists()


def test_exact_four_serial_single_attempt_synthetic_e2e_no_full_launch(module, tmp_path, monkeypatch):
    plans, _ = module.build_stage1_plan()
    fixed = module.fixed_preflight_plans(plans)
    parent = tmp_path / "external_v12"
    preflight_output = parent / "semantic_preflight_four_cells"
    lock = tmp_path / "preflight_lock.json"
    lock_sha = tmp_path / "preflight_lock.json.sha256"
    tombstone = tmp_path / "preflight_attempt.json"
    full_lock = tmp_path / "full_launch_lock.json"
    full_tombstone = tmp_path / "full_attempt.json"
    calls: list[tuple[str, str, int, str, int, bool]] = []

    monkeypatch.setattr(module, "OUTPUT_PARENT", parent)
    monkeypatch.setattr(module, "PREFLIGHT_OUTPUT", preflight_output)
    monkeypatch.setattr(module, "PREFLIGHT_LOCK", lock)
    monkeypatch.setattr(module, "PREFLIGHT_LOCK_SHA", lock_sha)
    monkeypatch.setattr(module, "PREFLIGHT_TOMBSTONE", tombstone)
    monkeypatch.setattr(module, "LAUNCH_LOCK", full_lock)
    monkeypatch.setattr(module, "SOURCE_TOMBSTONE", full_tombstone)
    lock.write_text('{"status":"synthetic-pass"}\n', encoding="utf-8")
    lock_sha.write_text("synthetic-sidecar\n", encoding="ascii")
    monkeypatch.setattr(
        module,
        "identity",
        lambda path: {
            "path": str(path),
            "bytes": path.stat().st_size if path.exists() else 0,
            "sha256": module.sha256_file(path) if path.is_file() else "absent",
        },
    )
    monkeypatch.setattr(module, "verify_preflight_lock", lambda _plans: {"status": "synthetic-pass"})
    monkeypatch.setattr(module, "decoder_gate_state", lambda: {"ready": False, "failure": None, "first_success_count": 0})

    def fake_get(plan, retries, cancel_event: threading.Event):
        calls.append((plan.product_id, plan.level_id, plan.lead, plan.data_key, retries, cancel_event.is_set()))
        payload = bytes([len(calls)]) * plan.byte_count
        headers = {
            "content-range": f"bytes {plan.byte_start}-{plan.byte_end}/{plan.data_bytes}",
            "content-length": str(plan.byte_count),
            "etag": f'"{plan.data_etag}"',
            "last-modified": "Thu, 30 Dec 2021 03:58:50 GMT",
        }
        return payload, headers, 1

    def fake_decode(_payload, plan):
        expected = 0 if plan.product_id.endswith("ens_mean") else 2
        names = ("10u", "10v") if plan.level_id == "10m" else ("u", "v")
        metadata = [fake_metadata(plan, name, expected) for name in names]
        values = [1.0, 2.0] if expected == 0 else [0.5, 0.75]
        return metadata, values

    monkeypatch.setattr(module.legacy, "range_get", fake_get)
    monkeypatch.setattr(module, "decode_two_messages", fake_decode)
    monkeypatch.setattr(module, "decode_two_messages_real_file_reference", fake_decode)

    module.execute_preflight(plans)

    assert [(item[0], item[1]) for item in calls] == [
        ("pressure_0p50_ens_mean", "10m"),
        ("pressure_0p50_ens_mean", "850"),
        ("pressure_0p50_ens_spread", "10m"),
        ("pressure_0p50_ens_spread", "850"),
    ]
    assert all(item[2] == 39 and item[4] == 1 and item[5] is False for item in calls)
    assert len(calls) == 4
    result = json.loads((preflight_output / "preflight_result.json").read_text(encoding="utf-8"))
    assert result["request_accounting"] == {"requests": 4, "response_bytes": 1_765_921, "attempts": 4, "automatic_retries": 0, "concurrency": 1}
    assert [item["expected_derivedForecast"] for item in result["records"]] == [0, 0, 2, 2]
    assert sum((preflight_output / item["response"]["path"]).stat().st_size for item in result["records"]) == 1_765_921
    assert result["full_extraction_payload_reuse_allowed"] is False
    assert result["automatic_full_launch"] is False
    assert tombstone.is_file()
    assert not full_lock.exists()
    assert not full_tombstone.exists()


def test_no_preflight_fifth_request_or_automatic_full_launch_literals(module):
    source = SCRIPT.read_text(encoding="utf-8")
    assert "PREFLIGHT_REQUESTS = 4" in source
    assert 'maximum_attempts_per_range": 1' in source
    assert '"automatic_full_launch": False' in source
    assert "expected_derived = 0 if plan.product_id.endswith(\"ens_mean\") else 2" in source
    assert "else 4" not in source
    assert "codes_grib_new_from_samples" not in source
    assert "if __name__ == \"__main__\":" in source
    assert "raise SystemExit(main())" in source


def test_full_launch_closure_never_opens_audit_only_result_or_response_values(module, tmp_path, monkeypatch):
    root = tmp_path / "semantic_preflight_four_cells"
    root.mkdir()
    manifest = {
        "status": "PASS_EXACT_FOUR_CELL_OFFICIAL_MEAN_STDDEV_SEMANTICS",
        "payload_reuse_by_full_extraction": 0,
        "automatic_full_launch": 0,
        "artifacts": [
            {"path": "responses/00_mean_10m.grib2", "bytes": 464970, "sha256": "0" * 64},
            {"path": "responses/01_mean_850.grib2", "bytes": 459291, "sha256": "1" * 64},
            {"path": "responses/02_stddev_10m.grib2", "bytes": 413765, "sha256": "2" * 64},
            {"path": "responses/03_stddev_850.grib2", "bytes": 427895, "sha256": "3" * 64},
            {"path": "preflight_result.json", "bytes": 1234, "sha256": "4" * 64},
        ],
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(module, "PREFLIGHT_OUTPUT", root)
    monkeypatch.setattr(
        module,
        "identity",
        lambda path: {"path": str(path), "bytes": path.stat().st_size, "sha256": module.sha256_file(path)},
    )
    closure = module.preflight_manifest_closure_for_full_launch()
    assert closure["declared_artifacts"] == manifest["artifacts"]
    assert not (root / "preflight_result.json").exists()
    assert not (root / "responses").exists()


def test_full_gate_parses_only_value_free_postreview_summary(module, tmp_path, monkeypatch):
    summary_path = tmp_path / "postreview_summary.json"
    detailed_path = tmp_path / "postreview_detail.json"
    bound_preflight = {"manifest": {"sha256": "a" * 64}, "declared_artifacts": []}
    summary = {
        "schema_version": 1,
        "audit_id": "noaa_gefs_operational_spread_00z_original_v12_semantic_preflight_postrun_independent_review_v1",
        "created_utc": "2026-08-08T00:00:00Z",
        "status": "PASS_VALUE_FREE_POSTRUN_AUTHORIZATION_SUMMARY",
        "verdict": "PASS",
        "authorization_scope": "FULL_STAGE1_SOURCE_LOCK_ELIGIBILITY_ONLY_ROOT_APPROVAL_STILL_REQUIRED",
        "preflight_requests": 4,
        "preflight_response_bytes": 1_765_921,
        "verified_message_count": 8,
        "verified_decoded_component_cells": 8,
        "verified_expected_derivedForecast": [0, 0, 2, 2],
        "metadata_value_exact_equality_verified": True,
        "full_source_label_fit_score_access": 0,
        "inline_decoded_metadata_or_values": 0,
        "bound_preflight": bound_preflight,
        "detailed_audit": {"path": str(detailed_path), "bytes": 999, "sha256": "b" * 64},
        "automatic_full_launch": 0,
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    monkeypatch.setattr(module, "POST_PREFLIGHT_REVIEW", summary_path)
    monkeypatch.setattr(module, "POST_PREFLIGHT_DETAILED_REVIEW", detailed_path)
    monkeypatch.setattr(module, "relative", lambda path: str(path))
    monkeypatch.setattr(
        module,
        "identity",
        lambda path: {"path": str(path), "bytes": path.stat().st_size, "sha256": module.sha256_file(path)},
    )
    identity = module._post_preflight_review_identity(bound_preflight)
    assert identity["sha256"] == module.sha256_file(summary_path)
    assert not detailed_path.exists()

    summary["detailed_audit"]["values"] = [1.0]
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(RuntimeError, match="forbidden decoded fields"):
        module._post_preflight_review_identity(bound_preflight)

    summary["detailed_audit"].pop("values")
    summary["payload"] = [3.41, 1.31]
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(RuntimeError, match="top-level schema differs"):
        module._post_preflight_review_identity(bound_preflight)


def test_fresh_v12_zero_state_before_independent_review(module):
    assert not module.OUTPUT_PARENT.exists()
    assert not module.PREFLIGHT_LOCK.exists()
    assert not module.PREFLIGHT_LOCK_SHA.exists()
    assert not module.PREFLIGHT_TOMBSTONE.exists()
    assert not module.LAUNCH_LOCK.exists()
    assert not module.LAUNCH_LOCK_SHA.exists()
    assert not module.SOURCE_TOMBSTONE.exists()
    assert not module.STAGE2_ROOT.exists()

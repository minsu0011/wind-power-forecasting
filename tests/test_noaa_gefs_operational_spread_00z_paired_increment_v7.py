from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
EXTRACTOR = ROOT / "scripts/download_noaa_gefs_operational_spread_00z_original_v7.py"
RUNNER = ROOT / "scripts/run_noaa_gefs_operational_spread_00z_paired_increment_v7.py"
V6 = ROOT / "configs/noaa_gefs_operational_spread_00z_paired_increment_preregister_v6.json"
V7 = ROOT / "configs/noaa_gefs_operational_spread_00z_paired_increment_preregister_v7.json"
V8 = ROOT / "configs/noaa_gefs_operational_spread_00z_paired_increment_preregister_v8.json"
V9 = ROOT / "configs/noaa_gefs_operational_spread_00z_paired_increment_preregister_v9.json"
OLD_SEAL = ROOT / "artifacts/incidents/noaa_gefs_operational_spread_00z_original_v2_stage1_protocol_failure_seal_v1.json"


def load_extractor():
    name = "_test_noaa_gefs_v7_extractor"
    spec = importlib.util.spec_from_file_location(name, EXTRACTOR)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def synthetic_message(short_name: str, value: float) -> bytes:
    import eccodes

    handle = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
    try:
        settings = (
            ("productDefinitionTemplateNumber", 2),
            ("derivedForecast", 0),
            ("typeOfLevel", "heightAboveGround"),
            ("level", 10),
            ("shortName", short_name),
            ("dataDate", 20211230),
            ("dataTime", 0),
            ("forecastTime", 39),
            ("Ni", 720),
            ("Nj", 361),
            ("numberOfDataPoints", 720 * 361),
            ("numberOfValues", 720 * 361),
            ("latitudeOfFirstGridPointInDegrees", 90.0),
            ("latitudeOfLastGridPointInDegrees", -90.0),
            ("longitudeOfFirstGridPointInDegrees", 0.0),
            ("longitudeOfLastGridPointInDegrees", 359.5),
            ("iDirectionIncrementInDegrees", 0.5),
            ("jDirectionIncrementInDegrees", 0.5),
        )
        for key, item in settings:
            eccodes.codes_set(handle, key, item)
        eccodes.codes_set_values(handle, np.full(720 * 361, value, dtype=np.float64))
        return bytes(eccodes.codes_get_message(handle))
    finally:
        eccodes.codes_release(handle)


def synthetic_plan(module, first_length: int, total_length: int):
    return module.RangePlan(
        operating_date="2022-01-01",
        source_date="2021-12-30",
        cutoff_utc="2021-12-31T05:00:00Z",
        product_id="pressure_0p50_ens_mean",
        statistic="ens mean",
        lead=39,
        level_id="10m",
        level="10 m above ground",
        data_key="gefs.20211230/00/atmos/pgrb2ap5/geavg.t00z.pgrb2a.0p50.f039",
        data_url="https://noaa-gefs-pds.s3.amazonaws.com/gefs.20211230/00/atmos/pgrb2ap5/geavg.t00z.pgrb2a.0p50.f039",
        data_bytes=total_length,
        data_etag="synthetic",
        data_last_modified_utc="2021-12-30T08:00:00Z",
        idx_key="synthetic.idx",
        idx_sha256="0" * 64,
        u_line="1:0:d=2021123000:UGRD:10 m above ground:39 hour fcst:ens mean",
        v_line=f"2:{first_length}:d=2021123000:VGRD:10 m above ground:39 hour fcst:ens mean",
        byte_start=0,
        byte_end=total_length - 1,
        byte_count=total_length,
        pack_path="synthetic.grib2pack",
        pack_offset=0,
    )


def test_v6_preserves_terminal_incident_and_science() -> None:
    payload = json.loads(V6.read_text(encoding="utf-8"))
    assert payload["status"].startswith("FROZEN_USER_AUTHORIZED_FRESH_ENGINEERING_EXPERIMENT")
    assert payload["superseded_terminal_lineage_preserved"]["v2_protocol_failure_seal"]["sha256"] == "89d114289de2a7d3c6793c243955330cdb6285ea028ded9275ab258086592a88"
    assert payload["superseded_terminal_lineage_preserved"]["v2_quarantine"]["v7_must_not_open_hash_copy_move_or_reuse_any_file"] is True
    science = payload["immutable_scientific_inheritance"]
    assert science["candidate_id"] == "gefs00z_0p50_10m_850_mean_spread_w025"
    assert science["transfer_weight"] == 0.25
    assert science["stage1_folds_slices_components_or_gate_changed"] is False


def test_v7_decoder_uses_only_generic_message_api() -> None:
    tree = ast.parse(EXTRACTOR.read_text(encoding="utf-8"))
    decoder = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "decode_two_messages")
    attributes = {node.attr for node in ast.walk(decoder) if isinstance(node, ast.Attribute)}
    assert "codes_new_from_message" in attributes
    assert "codes_grib_new_from_file" not in attributes
    assert "codes_grib_new_from_message" not in EXTRACTOR.read_text(encoding="utf-8")


def test_synthetic_new_decoder_equals_real_file_reference_exactly() -> None:
    module = load_extractor()
    first = synthetic_message("10u", 3.25)
    second = synthetic_message("10v", -1.75)
    payload = first + second
    plan = synthetic_plan(module, len(first), len(payload))
    memory_metadata, memory_values = module.decode_two_messages(payload, plan)
    file_metadata, file_values = module.decode_two_messages_real_file_reference(payload, plan)
    assert memory_metadata == file_metadata
    assert memory_values == file_values == [3.25, -1.75]
    assert [item["shortName"] for item in memory_metadata] == ["10u", "10v"]


def test_synthetic_incomplete_or_third_message_fails() -> None:
    module = load_extractor()
    first = synthetic_message("10u", 1.0)
    second = synthetic_message("10v", 2.0)
    plan = synthetic_plan(module, len(first), len(first) + len(second))
    with pytest.raises(RuntimeError, match="exactly two"):
        module.decode_two_messages(first, plan)
    with pytest.raises(RuntimeError, match="exactly two"):
        module.decode_two_messages(first + second + second, plan)


def test_fixed_preflight_is_one_exact_range_and_no_retry() -> None:
    payload = json.loads(V6.read_text(encoding="utf-8"))["one_bounded_live_preflight"]
    assert payload["requests"] == payload["concurrency"] == payload["retries"] == 1
    assert payload["range_bytes"] == 464970
    assert payload["full_extraction_payload_reuse"] is False
    source = EXTRACTOR.read_text(encoding="utf-8")
    assert "legacy.range_get(plan, retries=1, cancel_event=cancel)" in source
    assert '"preflight_payload_reuse": 0' in source


def test_preflight_requires_independent_zero_access_pass_before_lock(monkeypatch, tmp_path: Path) -> None:
    module = load_extractor()
    missing = tmp_path / "missing-review.json"
    monkeypatch.setattr(module, "INDEPENDENT_REVIEW", missing)
    with pytest.raises(FileNotFoundError, match="independent"):
        module._review_identity()
    missing.write_text(json.dumps({"verdict": "PASS", "v7_network_value_label_fit_score_access": 1}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="zero-access"):
        module._review_identity()


def test_full_launch_requires_separate_post_preflight_review(monkeypatch, tmp_path: Path) -> None:
    module = load_extractor()
    missing = tmp_path / "missing-post-preflight-review.json"
    monkeypatch.setattr(module, "POST_PREFLIGHT_REVIEW", missing)
    with pytest.raises(FileNotFoundError, match="post-preflight"):
        module._post_preflight_review_identity({"manifest": {"sha256": "x"}})


def test_fresh_namespaces_do_not_alias_old_outputs() -> None:
    module = load_extractor()
    assert "original_v7" in module.OUTPUT_ROOT.as_posix()
    assert "original_v2" not in module.OUTPUT_ROOT.as_posix()
    assert "original_v7" in module.PREFLIGHT_OUTPUT.as_posix()
    assert module.PREFLIGHT_TOMBSTONE != module.legacy.SOURCE_ATTEMPT_TOMBSTONE
    assert module.SOURCE_TOMBSTONE != module.legacy.SOURCE_ATTEMPT_TOMBSTONE
    assert module.V2_FAILURE_SEAL.is_file()
    assert OLD_SEAL.read_bytes() == module.V2_FAILURE_SEAL.read_bytes()


def test_old_quarantine_path_is_not_opened_by_v7_code() -> None:
    source = EXTRACTOR.read_text(encoding="utf-8")
    old_partial = "noaa_gefs_operational_spread_00z_original_v2_stage1_partial_20260808T183448Z_5988"
    assert old_partial not in source
    assert "old_v2_partial_files_opened_hashed_copied_or_reused\": 0" in source


def test_runner_transform_is_identifier_only_and_pins_legacy_bytes() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "LEGACY_RUNNER_SHA" in source
    assert '"scientific_code_changes": 0' in source
    assert "FRESH_MODEL_OUTPUT" in source
    assert "FRESH_MODEL_TOMBSTONE" in source
    assert "LEGACY_DEFAULT_OUTPUT" in source


def test_recursive_source_roots_include_legacy_scientific_runner() -> None:
    module = load_extractor()
    roots, recursive = module._source_roots_and_imports()
    root_names = {Path(record["path"]).name for record in roots}
    assert "run_noaa_gefs_operational_spread_00z_paired_increment_v2.py" in root_names
    recursive_names = {record["path"] for record in recursive}
    assert any(name.endswith("src/metric.py") for name in recursive_names)


def test_v9_head_exists_before_static_execution() -> None:
    assert V7.is_file(), "v7 executable head must be frozen before tests/review"
    assert V8.is_file(), "v8 source-only cleanup amendment must be frozen before tests/review"
    assert V9.is_file(), "v9 recursive closure amendment must be frozen before tests/review"
    v7 = json.loads(V7.read_text(encoding="utf-8"))
    v8 = json.loads(V8.read_text(encoding="utf-8"))
    v9 = json.loads(V9.read_text(encoding="utf-8"))
    assert v7["status"] == "FROZEN_FRESH_V7_EXECUTABLE_SOURCE_ONLY_BEFORE_INDEPENDENT_REVIEW_OR_ANY_V7_NETWORK_VALUE_LABEL_FIT_PREDICTION_OR_SCORE"
    assert v8["status"] == "FROZEN_V8_REFERENCE_CLEANUP_AMENDMENT_BEFORE_INDEPENDENT_REVIEW_OR_ANY_V7_NETWORK_VALUE_LABEL_FIT_PREDICTION_OR_SCORE"
    assert v9["status"] == "FROZEN_V9_RECURSIVE_SOURCE_CLOSURE_AMENDMENT_BEFORE_INDEPENDENT_REVIEW_OR_ANY_V7_NETWORK_VALUE_LABEL_FIT_PREDICTION_OR_SCORE"
    assert v9["safety_accounting_at_v9_freeze"]["network_requests"] == 0

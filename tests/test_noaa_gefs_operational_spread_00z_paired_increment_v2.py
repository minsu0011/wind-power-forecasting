from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import threading
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from email.utils import format_datetime
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


extractor = load("gefs_stage1_extractor_v2", "scripts/download_noaa_gefs_operational_spread_00z_original_v2.py")
runner = load("gefs_stage1_runner_v2", "scripts/run_noaa_gefs_operational_spread_00z_paired_increment_v2.py")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def plan():
    return extractor.build_stage1_plan()


def test_v1_v2_v3_v4_v5_resolution_and_zero_state_are_exact():
    assert sha(extractor.BASE_PREREG) == extractor.EXPECTED_BASE_SHA
    assert sha(extractor.PREREG) == extractor.EXPECTED_PREREG_SHA
    assert sha(extractor.PREREG_V3) == extractor.EXPECTED_PREREG_V3_SHA
    assert sha(extractor.PREREG_V4) == extractor.EXPECTED_PREREG_V4_SHA
    assert sha(extractor.PREREG_V5) == extractor.EXPECTED_PREREG_V5_SHA
    assert sha(extractor.INCIDENT) == extractor.EXPECTED_INCIDENT_SHA
    v2 = json.loads(extractor.PREREG.read_text(encoding="utf-8"))
    v3 = json.loads(extractor.PREREG_V3.read_text(encoding="utf-8"))
    v4 = json.loads(extractor.PREREG_V4.read_text(encoding="utf-8"))
    v5 = json.loads(extractor.PREREG_V5.read_text(encoding="utf-8"))
    incident = json.loads(extractor.INCIDENT.read_text(encoding="utf-8"))
    assert v2["supersedes"]["sha256"] == extractor.EXPECTED_BASE_SHA
    assert v3["lineage"]["label_and_source_phase_v2"]["sha256"] == extractor.EXPECTED_PREREG_SHA
    assert v4["supersedes"]["sha256"] == extractor.EXPECTED_PREREG_V3_SHA
    assert v5["supersedes"]["sha256"] == extractor.EXPECTED_PREREG_V4_SHA
    assert v5["chronology_correction"]["stage1_execution_protocol"]["recorded_created_utc_is_a_transcription_error"] is True
    assert v5["effective_contract"]["inherited_output_directory"].endswith("strict_v2")
    assert v4["chronology"]["v3_created_utc_was_a_transcription_error"] is True
    assert v4["effective_contract"]["stage1_final_2024_01_01_00_modeled"] is True
    assert v4["effective_contract"]["exact_zero_identity_only_conditional_2025_01_01_00"] is True
    assert v3["physical_bounded_stage1_control_rebuild"]["forbidden_cache_opens"] == 0
    assert incident["physical_state_at_discovery"]["meteorological_grib_range_gets"] == 0
    assert incident["physical_state_at_discovery"]["label_cells_materialized"] == 0
    assert incident["physical_state_at_discovery"]["2025_requests_or_values"] == 0


def test_stage1_plan_exact_counts_dates_products_leads_and_bytes(plan):
    plans, pack_sizes = plan
    assert len(plans) == 26_280
    objects = {(x.operating_date, x.product_id, x.lead, x.data_key) for x in plans}
    assert len(objects) == 13_140
    assert {x.operating_date for x in plans} == set(pd.date_range("2022-01-01", "2023-12-31", freq="D").strftime("%Y-%m-%d"))
    assert min(x.source_date for x in plans) == "2021-12-30"
    assert max(x.source_date for x in plans) == "2023-12-29"
    assert {x.product_id for x in plans} == set(extractor.PRODUCTS)
    assert {x.lead for x in plans} == set(extractor.LEADS)
    assert {x.level_id for x in plans} == {"10m", "850"}
    assert sum(x.byte_count for x in plans if x.level_id == "10m") == 5_810_625_029
    assert sum(x.byte_count for x in plans if x.level_id == "850") == 5_713_659_710
    assert sum(pack_sizes.values()) == 11_524_284_739
    assert len(pack_sizes) == 12
    source_year_bytes = {year: sum(x.byte_count for x in plans if x.source_date.startswith(str(year))) for year in (2021, 2022, 2023)}
    assert source_year_bytes == {2021: 31_670_989, 2022: 5_767_366_204, 2023: 5_725_247_546}
    assert all(Path(x.pack_path).name.startswith(f"source_{x.source_date[:4]}_") for x in plans)
    assert all("gefs.2024" not in x.data_key and "gefs.2025" not in x.data_key for x in plans)


def test_pack_offsets_gapless_ranges_adjacent_and_exact_url(plan):
    plans, pack_sizes = plan
    by_pack = defaultdict(list)
    for item in plans:
        by_pack[item.pack_path].append(item)
        assert item.byte_count == item.byte_end - item.byte_start + 1
        assert int(item.v_line.split(":", 2)[0]) == int(item.u_line.split(":", 2)[0]) + 1
        assert int(item.v_line.split(":", 2)[1]) > int(item.u_line.split(":", 2)[1])
        assert extractor.DATA_KEY_RE.fullmatch(item.data_key)
        assert item.data_url == f"https://{extractor.HOST}/{item.data_key}"
    for path, items in by_pack.items():
        by_offset = sorted(items, key=lambda x: x.pack_offset)
        assert by_offset[0].pack_offset == 0
        assert all(left.pack_offset + left.byte_count == right.pack_offset for left, right in zip(by_offset, by_offset[1:]))
        assert by_offset[-1].pack_offset + by_offset[-1].byte_count == pack_sizes[path]


def fake_grib2(length: int = 20) -> bytes:
    assert length >= 20
    return b"GRIB" + b"\x00\x00\x00" + b"\x02" + length.to_bytes(8, "big") + b"\x00" * (length - 20) + b"7777"


def test_structural_grib2_split_literal_empty_and_trailing_regressions():
    first, second = fake_grib2(20), fake_grib2(24)
    assert extractor.split_grib2_messages(first + second) == [first, second]
    for bad in (b"", first, first + second + b"x"):
        with pytest.raises(RuntimeError):
            extractor.split_grib2_messages(bad)
    altered = bytearray(first + second)
    altered[7] = 1
    with pytest.raises(RuntimeError, match="non-GRIB2"):
        extractor.split_grib2_messages(bytes(altered))


class FakeResponse:
    def __init__(self, status: int, headers: dict[str, str], body: bytes):
        self.status = status
        self._headers = headers
        self._body = body

    def read(self, _size: int | None = None) -> bytes:
        return self._body

    def getheaders(self):
        return list(self._headers.items())


class FakeConnection:
    def __init__(self, response: FakeResponse):
        self.response = response

    def request(self, *args, **kwargs):
        return None

    def getresponse(self):
        return self.response

    def close(self):
        return None


def response_headers(item, *, content_length: int | None = None, etag: str | None = None):
    modified = datetime.fromisoformat(item.data_last_modified_utc.replace("Z", "+00:00"))
    return {
        "content-range": f"bytes {item.byte_start}-{item.byte_end}/{item.data_bytes}",
        "content-length": str(item.byte_count if content_length is None else content_length),
        "etag": f'"{item.data_etag}"' if etag is None else etag,
        "last-modified": format_datetime(modified.astimezone(timezone.utc), usegmt=True),
    }


def test_http_guard_rejects_operating_2024_before_connection(plan, monkeypatch):
    plans, _ = plan
    forbidden = replace(plans[0], operating_date="2024-01-01", source_date="2023-12-30", data_key=plans[0].data_key.replace("gefs.20211230", "gefs.20231230"), data_url=plans[0].data_url.replace("gefs.20211230", "gefs.20231230"))
    monkeypatch.setattr(extractor, "get_connection", lambda *args, **kwargs: pytest.fail("network opened"))
    with pytest.raises(RuntimeError, match="outside Stage1"):
        extractor.range_get(forbidden, 1)


@pytest.mark.parametrize("header_change,match", [({"content_length": -1}, "Content-Length"), ({"etag": 'W/"weak"'}, "weak")])
def test_http_rejects_content_length_and_weak_etag(plan, monkeypatch, header_change, match):
    item = plan[0][0]
    body = b"x" * item.byte_count
    headers = response_headers(item, **header_change)
    monkeypatch.setattr(extractor, "get_connection", lambda *args, **kwargs: FakeConnection(FakeResponse(206, headers, body)))
    with pytest.raises(RuntimeError, match=match):
        extractor.range_get(item, 1)


def test_bounded_pool_cancels_without_submitting_all(plan, monkeypatch):
    calls: list[int] = []
    lock = threading.Lock()

    def fake_download(item, retries, cancel_event):
        with lock:
            calls.append(item.lead)
            ordinal = len(calls)
        if ordinal == 1:
            raise RuntimeError("first failure")
        if cancel_event.is_set():
            raise RuntimeError("cancelled")
        return item, b"", {}

    monkeypatch.setattr(extractor, "download_one", fake_download)
    with pytest.raises(RuntimeError, match="first failure"):
        list(extractor.bounded_download_results(plan[0][:100], workers=2, retries=1))
    assert len(calls) <= 4


def test_stale_launch_lock_semantics_fail_even_with_updated_sidecar(plan, monkeypatch, tmp_path):
    lock_path = tmp_path / "lock.json"
    sidecar = tmp_path / "lock.json.sha256"
    stale = {"status": "FROZEN_BEFORE_FIRST_STAGE1_RANGE_GET", "scope": {"bytes": 1}, "candidate_or_performance_information": 0}
    lock_path.write_text(json.dumps(stale) + "\n", encoding="utf-8")
    sidecar.write_text(f"{sha(lock_path)}  {lock_path.name}\n", encoding="ascii")
    monkeypatch.setattr(extractor, "LAUNCH_LOCK", lock_path)
    monkeypatch.setattr(extractor, "LAUNCH_LOCK_SHA", sidecar)
    monkeypatch.setattr(extractor, "expected_launch_semantics", lambda plans, sizes: {"scope": {"bytes": 2}})
    with pytest.raises(RuntimeError, match="current closure mismatch"):
        extractor.verify_launch_lock(*plan)


def test_recursive_closure_includes_package_initializer_and_local_dependencies():
    closure = {path.relative_to(ROOT).as_posix() for path in runner.resolve_ast_local_import_closure((Path(runner.__file__), Path(extractor.__file__)))}
    assert "src/__init__.py" in closure
    assert "src/metric.py" in closure
    assert "src/manifest.py" in closure
    assert "src/features.py" in closure
    assert "scripts/run_shared_q07_multiseed.py" in closure


def test_exact_label_protocol_internal_lock_order_and_no_subset_api(tmp_path):
    _v1, _v2, _v3, _v4, _v5, protocol = runner.verify_effective_preregister()
    contract = protocol["stage1_label_source"]
    assert sum(int(item["value_cells"]) for item in contract["fit_slices_before_global_candidate_lock"].values()) == 21_864
    assert sum(int(item["value_cells"]) for item in contract["score_slices_only_after_verified_global_candidate_lock"].values()) == 21_936
    assert not hasattr(runner, "assert_label_request")
    with pytest.raises(FileNotFoundError, match="global candidate lock"):
        runner.read_stage1_score_labels(tmp_path / "raw_never_opened", tmp_path / "out", protocol)


def test_actual_runner_constants_reload_boundary_and_static_audit():
    audit = runner.static_audit()
    assert audit["status"] == "PASS_ACTUAL_STAGE1_RUNNER_STATIC_CONTRACT"
    assert audit["models_to_save_reload"] == 6
    assert audit["fit_label_cells"] == 21_864 and audit["score_label_cells"] == 21_936
    assert audit["features_control"] == 612 and audit["features_extended"] == 624
    assert audit["stage1_final_row_modeled"] is True and audit["stage2_branch"] is False
    assert runner.MODEL_PARAMETERS["objective"] == "regression_l1"
    assert runner.MODEL_PARAMETERS["random_state"] == 42
    assert runner.DEFAULT_OUTPUT.relative_to(ROOT).as_posix().endswith("strict_v2")
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert source.count("joblib.load(") == 2
    assert "run_stage2" not in {node.name for node in __import__("ast").walk(__import__("ast").parse(source)) if isinstance(node, __import__("ast").FunctionDef)}


def test_attempt_tombstones_are_absent_and_exclusive_writers_refuse_reuse(tmp_path):
    assert not extractor.SOURCE_ATTEMPT_TOMBSTONE.exists()
    assert not runner.MODEL_ATTEMPT_TOMBSTONE.exists()
    first = tmp_path / "source_attempt.json"
    extractor.write_json_exclusive(first, {"token": 1})
    with pytest.raises(FileExistsError):
        extractor.write_json_exclusive(first, {"token": 2})
    second = tmp_path / "model_attempt.json"
    runner._write_json_exclusive(second, {"token": 1})
    with pytest.raises(FileExistsError):
        runner._write_json_exclusive(second, {"token": 2})


def test_guard_and_tombstone_order_are_fail_closed_in_actual_sources():
    runner_source = Path(runner.__file__).read_text(encoding="utf-8")
    source_source = Path(extractor.__file__).read_text(encoding="utf-8")
    assert runner_source.index("_write_json_exclusive(MODEL_ATTEMPT_TOMBSTONE") < runner_source.index("fit_labels, fit_ledger = read_stage1_fit_labels")
    score_acquire = runner_source.index("score_guard = HeavyFitGuard(defer_success_release=True)")
    score_read = runner_source.index("score_labels, score_ledger = read_stage1_score_labels")
    selection_write = runner_source.index("_write_json(selection_path")
    score_release = runner_source.index("score_guard.release()")
    assert score_acquire < score_read < selection_write < score_release
    assert runner_source.count("score_guard = HeavyFitGuard(") == 1
    assert "continuous_heavy_guard_through_result_seal" in runner_source
    assert source_source.index("write_json_exclusive(SOURCE_ATTEMPT_TOMBSTONE") < source_source.index("bounded_download_results(plans")
    assert "os.replace(temporary, LAUNCH_LOCK)" not in source_source


def test_extractor_has_no_label_model_metric_or_2025_value_reader():
    source = Path(extractor.__file__).read_text(encoding="utf-8")
    for forbidden in ("train_labels.csv", "src.metric", "LGBMRegressor", "gefs.2025/"):
        assert forbidden not in source
    assert "--execute" in source and "--freeze-launch-lock" in source
    assert "verify_launch_lock(plans, pack_sizes)" in source
    assert "INDEPENDENT_REVIEW" in source

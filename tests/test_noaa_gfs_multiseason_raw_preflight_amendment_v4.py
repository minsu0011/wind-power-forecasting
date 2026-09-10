from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "artifacts" / "baram2026_ncei_scada_longrun_20260810_v2"
RUNNER_SHA = "51e6c4b68d47916f56975b16b9ceaae2e345f9f68963dba197501448d655934c"
TEST_SHA = "068ea40e2afab6b12421741229c14e73d71b7dda3e81df3ee9d0874f8bba2468"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_record(record: dict) -> Path:
    path = Path(record["path"])
    return path if path.is_absolute() else ROOT / path


def assert_record(record: dict) -> None:
    path = resolve_record(record)
    payload = path.read_bytes()
    assert len(payload) == record["size_bytes"]
    assert hashlib.sha256(payload).hexdigest() == record["sha256"]


def load_runner() -> object:
    path = REPO / "scripts" / "run_noaa_gfs_multiseason_raw_v2.py"
    spec = importlib.util.spec_from_file_location("sealed_runner_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v4_preflight_closes_reviewed_head_caps_runtime_and_source_chain() -> None:
    amendment_path = ROOT / "prelaunch" / "RAW_RUNNER_STATIC_PREFLIGHT_AMENDMENT_V4.json"
    assert amendment_path.is_file()
    payload = json.loads(amendment_path.read_text(encoding="utf-8"))
    assert payload["status"] == "INDEPENDENT_STATIC_REVIEW_PASS_AWAITING_GO_NOT_AUTHORIZED"
    assert payload["runner"]["sha256"] == RUNNER_SHA
    assert payload["runner_test"]["sha256"] == TEST_SHA
    caps = payload["attempt_caps"]
    assert caps["census_successful_logical_requests"] == 1_200
    assert caps["census_attempts_per_logical_request_hard_max"] == 4
    assert caps["census_worst_case_http_attempts"] == 4_800
    assert caps["raw_actual_http_attempt_budget"] == 15_200
    assert caps["raw_retry_headroom"] == 4_832
    assert caps["census_plus_raw_max_http_attempts"] == 20_000
    for record in payload["source_and_provenance_chain"].values():
        assert_record(record)
    runner = load_runner()
    assert payload["runtime_identity"] == runner.runtime_identity()
    assert payload["authorization_created"] is False
    assert payload["independent_go_present"] is False
    assert payload["raw_network_requests"] == 0


def test_auth_draft_is_non_executable_and_has_exact_runner_input_caps() -> None:
    path = ROOT / "prelaunch" / "RAW_LAUNCH_AUTHORIZATION_DRAFT_V1.json"
    assert path.is_file()
    draft = json.loads(path.read_text(encoding="utf-8"))
    assert draft["status"] == "DRAFT_ONLY_NON_EXECUTABLE_AWAITING_INDEPENDENT_GO"
    assert draft["actual_authorization_created"] is False
    assert draft["independent_go_created"] is False
    proposed = draft["proposed_authorization_payload"]
    assert proposed["runner"]["sha256"] == RUNNER_SHA
    assert proposed["runner_test"]["sha256"] == TEST_SHA
    assert proposed["census_cumulative_access"]["successful_http_requests"] == 1_200
    assert proposed["census_cumulative_access"]["worst_case_http_attempts"] == 4_800
    assert proposed["raw_actual_http_attempt_budget"] == 15_200
    assert proposed["census_worst_case_http_attempts"] == 4_800
    assert proposed["census_plus_raw_max_http_attempts"] == 20_000
    for key in (
        "field_range_census",
        "census_manifest",
        "coordinate_lock",
        "effective_preflight_amendment",
        "independent_target_free_preregister",
        "independent_census_audit",
        "bounded_predictor_manifest",
    ):
        assert_record(proposed[key])
    assert not (ROOT / draft["proposed_authorization_target"]).exists()


def test_v4_manifest_rehashes_every_sealed_artifact_and_zero_outputs() -> None:
    path = ROOT / "manifest_raw_preflight_v4.json"
    assert path.is_file()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for key in (
        "parent_manifest",
        "amendment",
        "authorization_draft",
        "runner",
        "runner_test",
        "seal_code",
        "seal_test",
    ):
        assert_record(manifest[key])
    assert manifest["runner"]["sha256"] == RUNNER_SHA
    assert manifest["runner_test"]["sha256"] == TEST_SHA
    assert manifest["expected_zero_output_inventory_all_absent"] is True
    assert manifest["authorization_created"] is False
    assert manifest["independent_go_present"] is False
    assert manifest["raw_network_requests"] == 0
    assert not (ROOT / "raw").exists()
    assert not (ROOT / "decoded").exists()
    assert not (ROOT / "manifest_raw_v1.json").exists()
    assert not (ROOT / "prereg" / "raw_launch_authorization_v1.json").exists()
    assert not (
        ROOT / "independent_redteam" / "TRACK_A_RAW_LAUNCH_GO.json"
    ).exists()

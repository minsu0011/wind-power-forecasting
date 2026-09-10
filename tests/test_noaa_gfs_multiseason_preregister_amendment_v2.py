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


def _payload(name: str) -> dict:
    path = ROOT / name
    if not path.is_file():
        pytest.skip("append-only prereg amendment not generated")
    return json.loads(path.read_text(encoding="utf-8"))


def test_authoritative_coordinates_match_all_17_sites() -> None:
    lock = _payload("prereg/authoritative_turbine_coordinate_lock_v1.json")
    assert lock["authoritative_source"]["sha256"] == (
        "89e83a52e0eb2ce367a3573a96d6795ed4b4d4ac624965cb3530beec0cbd2bd6"
    )
    assert lock["audit"]["row_count"] == 17
    assert lock["audit"]["all_config_rows_match"] is True
    assert lock["audit"]["maximum_coordinate_abs_diff_degrees"] <= 1e-12
    assert lock["audit"]["group_capacity_mw"] == {
        "kpx_group_1": 21.6,
        "kpx_group_2": 21.6,
        "kpx_group_3": 21.0,
    }


def test_family_salvage_is_independent_and_frozen_before_values() -> None:
    amendment = _payload("prereg/target_free_multiseason_amendment_v2.json")
    rules = amendment["family_missingness_and_salvage"]
    assert rules["PBL_HEIGHT"]["on_failure"] == "REJECT_PBL_HEIGHT_ONLY"
    assert rules["LOW_LEVEL_ISOBARIC_WIND_PROFILE"]["on_failure"] == (
        "REJECT_LOW_LEVEL_ISOBARIC_WIND_PROFILE_ONLY"
    )
    assert rules["LOW_LEVEL_ISOBARIC_WIND_PROFILE"]["posthoc_level_dropping_forbidden"] is True
    assert rules["rule_frozen_before_values"] is True


def test_only_target_free_duplicate_estimators_are_allowed() -> None:
    amendment = _payload("prereg/target_free_multiseason_amendment_v2.json")
    scope = amendment["model_fit_scope_clarification"]
    assert scope["competition_generation_target_models"] == "FORBIDDEN"
    assert scope["competition_labels"] == "FORBIDDEN"
    assert scope["target_free_duplicate_estimator"] == (
        "ALLOWED_ONLY_AFTER_DECODED_MATRIX_SHA_LOCK"
    )
    assert [row["name"] for row in scope["allowed_estimators"]] == [
        "RIDGE_AFFINE",
        "EXTRA_TREES_NONLINEAR",
    ]
    assert amendment["novelty_thresholds"]["no_threshold_change_after_decode"] is True


def test_manifest_v2_hash_closes_parent_amendment_and_coordinate_lock() -> None:
    manifest = _payload("manifest_preregister_v2.json")
    assert manifest["parent_manifest"]["sha256"] == (
        "ac9ab919b6eba7ba4c6f36170b7035c7e0f21e1bf16dac8e19b4aa6a993b0bf2"
    )
    assert manifest["existing_files_modified"] == []
    assert manifest["index_files_read"] == 0
    assert manifest["external_values_decoded"] == 0
    for key in ("parent_manifest", "amendment", "coordinate_lock"):
        record = manifest[key]
        payload = (ROOT / record["path"]).read_bytes()
        assert len(payload) == record["size_bytes"]
        assert hashlib.sha256(payload).hexdigest() == record["sha256"]


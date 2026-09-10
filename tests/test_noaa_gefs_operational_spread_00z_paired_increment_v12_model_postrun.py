from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "artifacts/postgate/noaa_gefs_operational_spread_00z_paired_increment_strict_v12"
STAGE2 = ROOT / "artifacts/external/noaa_gefs_operational_spread_00z_original_v12/stage2_source_operating_2024"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_record(record):
    path = Path(record["path"])
    if not path.is_absolute():
        path = ROOT / path
    expected_bytes = int(record.get("bytes", record.get("size_bytes")))
    assert path.is_file() and path.stat().st_size == expected_bytes
    assert sha(path) == record["sha256"]
    return path


@pytest.fixture(scope="module")
def manifest():
    if not MODEL.exists():
        pytest.skip("v12 Stage1 model has not been authorized or executed")
    return json.loads((MODEL / "manifest.json").read_text(encoding="utf-8"))


def test_manifest_output_and_no_future_closure(manifest):
    assert manifest["status"] in {"REJECTED_STAGE1", "PASSED_STAGE1_AWAITING_SEPARATE_STAGE2_SOURCE_AUTHORIZATION"}
    assert manifest["artifact_type"] == "noaa_gefs_operational_spread_00z_paired_increment_strict_v12"
    assert manifest["no_csv"] is True
    assert manifest["operating_2024_external_or_label_values"] == 0
    assert manifest["2025_requests_or_values"] == 0
    assert not STAGE2.exists()
    assert not list(MODEL.rglob("*.csv"))
    for record in manifest["outputs"]:
        verify_record(record)


def test_candidate_lock_models_and_exact_label_ledgers(manifest):
    lock_path = MODEL / "stage1_candidate_before_score_labels_lock.json"
    sidecar = lock_path.with_suffix(".json.sha256")
    assert sidecar.read_text(encoding="ascii") == f"{sha(lock_path)}  {lock_path.name}\n"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock["lock_kind"] == "stage1_candidate_before_score_labels"
    assert lock["score_label_value_cells_before_lock"] == lock["metric_values_before_lock"] == 0
    assert lock["six_models_saved_and_reloaded"] is True
    assert lock["all_prediction_replays_bit_exact"] is True
    assert lock["stage1_final_2024_01_01_00_modeled"] is True
    assert len(list((MODEL / "stage1/models").glob("*.joblib"))) == 6
    for record in lock["bound_outputs"]:
        verify_record(record)
    fit = json.loads((MODEL / "stage1_fit_label_access.json").read_text(encoding="utf-8"))
    score = json.loads((MODEL / "stage1_score_label_access.json").read_text(encoding="utf-8"))
    assert fit["exact_value_cells"] == 21_864 and fit["score_label_cells_materialized"] == 0
    assert score["exact_value_cells"] == 21_936 and score["2024_label_cells_materialized"] == 0
    assert fit["whole_label_file_hashed"] is False and score["whole_label_file_hashed"] is False


def test_metric_and_selection_gate_formula(manifest):
    result = json.loads((MODEL / "stage1_results.json").read_text(encoding="utf-8"))
    assert len(result["delta_vector_17"]) == 17
    assert set(result["mixed_component_gates"]) == {"full", "H2"}
    expected = all(float(value) > 0.0 for value in result["delta_vector_17"]) and all(
        float(record["delta_one_minus_nmae"]) >= 0.0 and float(record["delta_ficr"]) >= 0.0
        for record in result["mixed_component_gates"].values()
    )
    assert result["stage1_passed"] is expected
    selection = json.loads((MODEL / "stage1_selection_lock.json").read_text(encoding="utf-8"))
    assert selection["stage1_passed"] is expected
    assert selection["stage2_source_open_allowed"] is expected
    assert selection["stage2_source_opened_by_this_runner"] is False
    assert selection["2024_label_cells"] == 0 and selection["2025_requests_or_values"] == 0


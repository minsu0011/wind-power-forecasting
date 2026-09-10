from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "artifacts" / "external" / "noaa_gefs_operational_spread_00z_original_v2" / "stage1_source_through_operating_2023"
STAGE2 = ROOT / "artifacts" / "external" / "noaa_gefs_operational_spread_00z_original_v2" / "stage2_source_operating_2024"
MODEL_CANONICAL = ROOT / "artifacts" / "postgate" / "noaa_gefs_operational_spread_00z_paired_increment_strict_v2"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture(scope="module")
def manifest():
    if not CANONICAL.exists():
        pytest.skip("Stage1 source canonical has not been created yet")
    return json.loads((CANONICAL / "manifest.json").read_text(encoding="utf-8"))


def test_postrun_manifest_artifact_and_nonmutation_closure(manifest):
    assert manifest["status"] == "PASS_STAGE1_SOURCE_EXTRACTION"
    assert manifest["canonical_root"].endswith("stage1_source_through_operating_2023")
    assert not STAGE2.exists()
    assert manifest["nonmutation"] == {"label_model_metric_csv_writes": 0, "stage2_namespace_created": False}
    for item in manifest["artifacts"]:
        path = ROOT / item["path"]
        assert path.is_file()
        assert path.stat().st_size == item["bytes"]
        assert sha(path) == item["sha256"]


def test_postrun_response_manifest_and_raw_range_replay(manifest):
    path = CANONICAL / "range_response_manifest.csv.gz"
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 26_280
    assert sum(int(row["byte_count"]) for row in rows) == 11_524_284_739
    assert {row["operating_date"][:4] for row in rows} == {"2022", "2023"}
    assert all("gefs.2024" not in row["data_key"] and "gefs.2025" not in row["data_key"] for row in rows)
    assert all(row["response_etag"] == row["data_etag"] for row in rows)
    for row in (rows[0], rows[len(rows) // 2], rows[-1]):
        pack = CANONICAL / row["pack_path"]
        with pack.open("rb") as handle:
            handle.seek(int(row["pack_offset"]))
            payload = handle.read(int(row["byte_count"]))
        assert hashlib.sha256(payload).hexdigest() == row["response_sha256"]


def test_postrun_native_and_hourly_cache_boundaries(manifest):
    native = pd.read_parquet(CANONICAL / "native_point_values.parquet")
    hourly = pd.read_parquet(CANONICAL / "hourly_components.parquet")
    assert len(native) == 26_280
    assert len(hourly) == 17_520
    assert native["operating_date"].min() == "2022-01-01"
    assert native["operating_date"].max() == "2023-12-31"
    assert hourly["forecast_kst_dtm"].min() == "2022-01-01 01:00:00"
    assert hourly["forecast_kst_dtm"].max() == "2024-01-01 00:00:00"
    assert set(hourly["missing_gefs_00z"].unique()) == {0}
    numeric = [column for column in hourly if column.startswith("gefs00z__")]
    assert len(numeric) == 8
    assert hourly[numeric].notna().all().all()
    spread = [column for column in numeric if "_std_" in column]
    assert (hourly[spread] >= 0.0).all().all()


def test_postrun_summary_has_zero_future_label_model_metric_access(manifest):
    summary = json.loads((CANONICAL / "extraction_summary.json").read_text(encoding="utf-8"))
    accounting = summary["request_accounting"]
    assert summary["status"] == "PASS_STAGE1_SOURCE_EXTRACTION"
    assert accounting["operating_2024_grib_range_gets_or_values"] == 0
    assert accounting["2025_requests_or_values"] == 0
    assert accounting["label_reads"] == 0
    assert accounting["fits"] == accounting["predictions"] == accounting["scores"] == 0
    assert accounting["csvs"] == 0


@pytest.fixture(scope="module")
def model_manifest():
    if not MODEL_CANONICAL.exists():
        pytest.skip("Stage1 paired-model canonical has not been created yet")
    return json.loads((MODEL_CANONICAL / "manifest.json").read_text(encoding="utf-8"))


def verify_record(record):
    path = Path(record["path"])
    if not path.is_absolute():
        path = ROOT / path
    expected_bytes = int(record.get("bytes", record.get("size_bytes")))
    assert path.is_file() and path.stat().st_size == expected_bytes
    assert sha(path) == record["sha256"]
    return path


def test_model_postrun_manifest_lock_and_no_future_closure(model_manifest):
    assert model_manifest["status"] in {
        "REJECTED_STAGE1",
        "PASSED_STAGE1_AWAITING_SEPARATE_STAGE2_SOURCE_AUTHORIZATION",
    }
    assert model_manifest["artifact_type"].endswith("strict_v5")
    assert model_manifest["no_csv"] is True
    assert model_manifest["operating_2024_external_or_label_values"] == 0
    assert model_manifest["2025_requests_or_values"] == 0
    assert not list(MODEL_CANONICAL.rglob("*.csv"))
    for record in model_manifest["outputs"]:
        verify_record(record)


def test_model_postrun_candidate_before_score_lock_and_exact_ledgers(model_manifest):
    lock_path = MODEL_CANONICAL / "stage1_candidate_before_score_labels_lock.json"
    sidecar = lock_path.with_suffix(".json.sha256")
    assert sidecar.read_text(encoding="ascii") == f"{sha(lock_path)}  {lock_path.name}\n"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock["lock_kind"] == "stage1_candidate_before_score_labels"
    assert lock["score_label_value_cells_before_lock"] == 0
    assert lock["metric_values_before_lock"] == 0
    assert lock["six_models_saved_and_reloaded"] is True
    assert lock["all_prediction_replays_bit_exact"] is True
    assert lock["stage1_final_2024_01_01_00_modeled"] is True
    assert len(list((MODEL_CANONICAL / "stage1/models").glob("*.joblib"))) == 6
    for record in lock["bound_outputs"]:
        verify_record(record)
    fit = json.loads((MODEL_CANONICAL / "stage1_fit_label_access.json").read_text(encoding="utf-8"))
    score = json.loads((MODEL_CANONICAL / "stage1_score_label_access.json").read_text(encoding="utf-8"))
    assert fit["exact_value_cells"] == 21_864 and fit["score_label_cells_materialized"] == 0
    assert score["exact_value_cells"] == 21_936 and score["2024_label_cells_materialized"] == 0
    assert fit["whole_label_file_hashed"] is False and score["whole_label_file_hashed"] is False


def test_model_postrun_registered_metrics_and_gate_formula(model_manifest):
    result = json.loads((MODEL_CANONICAL / "stage1_results.json").read_text(encoding="utf-8"))
    assert len(result["delta_vector_17"]) == 17
    assert set(result["mixed_component_gates"]) == {"full", "H2"}
    expected_pass = all(float(value) > 0.0 for value in result["delta_vector_17"]) and all(
        float(record["delta_one_minus_nmae"]) >= 0.0 and float(record["delta_ficr"]) >= 0.0
        for record in result["mixed_component_gates"].values()
    )
    assert result["stage1_passed"] is expected_pass
    selection = json.loads((MODEL_CANONICAL / "stage1_selection_lock.json").read_text(encoding="utf-8"))
    assert selection["stage1_passed"] is expected_pass
    assert selection["stage2_source_open_allowed"] is expected_pass
    assert selection["stage2_source_opened_by_this_runner"] is False
    assert selection["2024_label_cells"] == 0 and selection["2025_requests_or_values"] == 0

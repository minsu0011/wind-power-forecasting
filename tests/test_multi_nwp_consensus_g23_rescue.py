from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.metric import CAPACITY_KWH, TARGET_COLS


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/multi_nwp_consensus_g23_rescue_preregister_v1.json"
VALIDATION = ROOT / "artifacts/postgate/multi_nwp_consensus_g23_rescue_v1/validation_results.json"
FINAL_ROOT = ROOT / "artifacts/final_multi_nwp_consensus_g23_rescue_v1"
PREDICTIONS = FINAL_ROOT / "predictions"
TEST_INDEX = pd.date_range(
    "2025-01-01 01:00", "2026-01-01 00:00", freq="h", name="forecast_kst_dtm"
)


def test_frozen_candidate_and_2024_gate() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["experiment_id"] == "multi_nwp_consensus_g23_rescue_v1"
    assert config["selection_safety"]["classification"] == "posthoc_selection_unsafe"
    assert config["selection_safety"]["public_subgroup_or_time_feedback_used"] is False
    assert config["candidate"]["transfer_weight"] == 0.15
    assert config["candidate"]["g1_identity_relative_to_deployment_baseline"] is True
    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))
    gate = validation["promotion_gate"]
    assert gate["promoted"] is True
    assert gate["mixed_H2_delta"]["total_score"] > 0
    assert gate["mixed_H2_delta"]["ficr"] > 0
    for component in ("total_score", "one_minus_nmae", "ficr"):
        assert validation["comparisons"]["by_group"]["kpx_group_1"]["H2"]["delta"][component] == 0


def test_final_consensus_base_formula_and_reload_proof() -> None:
    consensus = pd.read_parquet(PREDICTIONS / "final_consensus_increment_cf_2025.parquet")
    baseline = pd.read_parquet(PREDICTIONS / "recent097_baseline_2025.parquet")
    final = pd.read_parquet(PREDICTIONS / "multi_nwp_consensus_g23_rescue_v1_2025.parquet")
    for frame in (consensus, baseline, final):
        assert frame.index.equals(TEST_INDEX)
        assert tuple(frame.columns) == TARGET_COLS
        assert np.isfinite(frame.to_numpy(dtype=np.float64)).all()
    assert np.array_equal(consensus["kpx_group_1"].to_numpy(), np.zeros(len(TEST_INDEX)))
    replay = baseline.copy()
    for group in ("kpx_group_2", "kpx_group_3"):
        replay[group] = np.clip(
            baseline[group].to_numpy(dtype=np.float64)
            + 0.15 * CAPACITY_KWH[group] * consensus[group].to_numpy(dtype=np.float64),
            0,
            1.02 * CAPACITY_KWH[group],
        )
    assert np.array_equal(replay.to_numpy(dtype=np.float64), final.to_numpy(dtype=np.float64))
    for group in TARGET_COLS:
        assert final[group].between(0, 1.02 * CAPACITY_KWH[group], inclusive="both").all()

    lock = json.loads((FINAL_ROOT / "final_lock.json").read_text(encoding="utf-8"))
    assert lock["all_twelve_models_two_reload_prediction_exact"] is True
    assert len(lock["model_reload_records"]) == 6
    assert sum(len(pair) for pair in lock["model_reload_records"].values()) == 12
    assert all(
        record["prediction_float64_bit_exact"]
        for pair in lock["model_reload_records"].values()
        for record in pair.values()
    )
    assert lock["consensus_formula_final_prediction_float64_exact_replay"] is True


def test_submission_text_and_cutoff_manifest() -> None:
    csv_path = PREDICTIONS / "multi_nwp_consensus_g23_rescue_v1_2025.csv"
    raw = csv_path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" not in raw
    frame = pd.read_csv(csv_path, encoding="utf-8-sig", dtype="string", keep_default_na=False)
    sample = pd.read_csv(
        Path(r"data/local/open/sample_submission.csv"),
        encoding="utf-8-sig",
        dtype="string",
        keep_default_na=False,
    )
    assert tuple(frame.columns) == ("forecast_id", "forecast_kst_dtm", *TARGET_COLS)
    assert len(frame) == 8_760
    assert frame[["forecast_id", "forecast_kst_dtm"]].equals(
        sample[["forecast_id", "forecast_kst_dtm"]]
    )
    for group in TARGET_COLS:
        assert frame[group].str.fullmatch(r"\d+\.\d{6}").all()

    source = json.loads(
        (
            ROOT
            / "artifacts/external/openmeteo_multi_nwp_g23_rescue_2025_v1/source_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert source["cutoff_semantics"]["post_cutoff_forecast_or_observation_used"] is False
    assert source["cutoff_semantics"]["fixed_feature_id"] == "B_day1_hours01_13_else_day2_w025"
    assert source["labels_or_public_feedback_read"] == 0
    assert set(source["sources"]) == {"ecmwf_ifs025", "icon_global", "gfs_global"}

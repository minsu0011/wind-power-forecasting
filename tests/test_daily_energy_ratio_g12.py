from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.run_daily_energy_ratio_g12 as runner


def test_preregister_chain_and_single_candidate_are_frozen() -> None:
    root=Path(__file__).resolve().parents[1]
    v2=root/"configs/daily_energy_ratio_g12_identity_g3_preregister_v2.json"
    v3=root/"configs/daily_energy_ratio_g12_identity_g3_preregister_v3.json"
    assert hashlib.sha256(v2.read_bytes()).hexdigest()==v2.with_suffix(".sha256").read_text().split()[0]
    assert hashlib.sha256(v3.read_bytes()).hexdigest()==v3.with_suffix(".sha256").read_text().split()[0]
    payload=json.loads(v3.read_text(encoding="utf-8")); base=json.loads(v2.read_text(encoding="utf-8"))
    assert payload["base_v2"]["sha256"]==hashlib.sha256(v2.read_bytes()).hexdigest()
    assert payload["unchanged"]["retune_or_result_based_change"] is False
    assert base["permanent_physical_access_ledger"]["prepreregister_2024_weather_cache_read_only"] is True


def test_run_key_is_exact_01_through_00() -> None:
    index=pd.date_range("2023-04-01 01:00","2023-04-02 00:00",freq="h")
    key=runner.run_key(index)
    assert len(set(key))==1 and key[0]==pd.Timestamp("2023-04-01")


def test_daily_features_exact_schema_and_finite() -> None:
    index=pd.date_range("2023-04-01 01:00","2023-04-02 00:00",freq="h",name="forecast_kst_dtm")
    baseline=pd.Series(np.linspace(100,2400,24),index=index)
    weather=pd.DataFrame({"ldaps__idw__hub_ws":np.arange(24.),"gfs__idw__hub_ws":np.arange(24.)+1,"cross__hub_ws_mean":np.arange(24.)+.5},index=index)
    features=runner.daily_features(baseline,weather,21600.)
    assert tuple(features.columns)==runner.FEATURE_COLUMNS
    assert features.shape==(1,20) and np.isfinite(features.to_numpy()).all()


def test_factor_clip_shrink_and_exact_shape_preservation() -> None:
    class Model:
        def predict(self, features): return np.array([0.5,2.0])
    features=pd.DataFrame(np.zeros((2,1)),index=pd.to_datetime(["2023-01-01","2023-01-02"]))
    factor=runner.factor_from_model(Model(),features)
    np.testing.assert_array_equal(factor.to_numpy(),[.975,1.0125])
    index=pd.date_range("2023-01-01 01:00","2023-01-02 00:00",freq="h")
    base=pd.Series(np.arange(1,25,dtype=float),index=index)
    cand=runner.apply_factor(base,factor.iloc[:1])
    np.testing.assert_allclose(cand/base,.975)


def test_main_locks_each_candidate_before_next_labels_and_stage2() -> None:
    source=inspect.getsource(runner.main)
    ordered=["source_lock(OUT)","acquire_guard", "read_initial_q1()", "block_fit_predict", "read_segment(block)", "stage1_results.json", "if passed:", "stage2_candidate_lock_before_2024_labels.json", "cursor.read_2024()"]
    positions=[source.index(marker) for marker in ordered]
    assert positions==sorted(positions)


def test_empty_cell_parses_as_nan_and_incomplete_day_is_dropped(tmp_path: Path) -> None:
    path=tmp_path/"labels.csv"
    rows=["kst_dtm,kpx_group_1,kpx_group_2,kpx_group_3\n"]
    for i,timestamp in enumerate(runner.S1["Q2"]):
        g1="" if i==0 else "1"
        rows.append(f"{timestamp:%Y-%m-%d %H:%M:%S},{g1},2,3\n")
    path.write_text("".join(rows),encoding="utf-8")
    cursor=runner.LabelCursor(path); frame,_=cursor.read_segment("Q2"); cursor.close()
    assert np.isnan(frame.iloc[0,0])
    baseline=pd.Series(10.0,index=frame.index)
    target,audit=runner.daily_target(frame["kpx_group_1"],baseline)
    assert audit["all_run_count"]==91 and audit["eligible_run_count"]==90
    assert audit["dropped_run_keys"]==["2023-04-01 00:00:00"]
    assert len(target)==90 and len(audit["eligible_mask_sha256"])==64

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.run_long_lead_scale095 as runner


def test_preregister_literal_candidate_is_exact() -> None:
    root=Path(__file__).resolve().parents[1]; path=root/"configs/long_lead_scale095_preregister_v1.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest()==path.with_suffix(".sha256").read_text().split()[0]
    p=json.loads(path.read_text(encoding="utf-8")); c=p["single_candidate"]
    assert c["identity_leads"]==[12,23] and c["scaled_leads"]==[24,35]
    assert c["factor"]==.95 and c["factor_candidates"]==c["cutoff_candidates"]==1
    assert p["selection_risk"]["selection_unsafe"] is True


def test_exact_leads_mask_and_twelve_scaled_rows() -> None:
    idx=pd.date_range("2023-01-01 01:00","2023-01-02 00:00",freq="h",name="forecast_kst_dtm")
    np.testing.assert_array_equal(runner.lead_values(idx),np.arange(12,36))
    mask=runner.long_lead_mask(idx); assert mask.sum()==12
    assert not mask[:12].any() and mask[12:].all()


def test_candidate_preserves_early_bits_and_scales_long_leads() -> None:
    idx=pd.date_range("2023-01-01 01:00","2023-01-02 00:00",freq="h",name="forecast_kst_dtm")
    baseline=pd.DataFrame({g:np.arange(1,25,dtype=np.float64) for g in runner.TARGET_COLS},index=idx)
    candidate,audit=runner.candidate_from_baseline(baseline)
    for g in runner.TARGET_COLS:
        assert np.array_equal(candidate[g].iloc[:12].to_numpy(),baseline[g].iloc[:12].to_numpy())
        np.testing.assert_array_equal(candidate[g].iloc[12:].to_numpy(),.95*baseline[g].iloc[12:].to_numpy())
    assert audit["factor"].tolist()==[1.]*12+[.95]*12


def test_empty_label_values_parse_as_nan(tmp_path: Path) -> None:
    path=tmp_path/"labels.csv"; lines=["kst_dtm,kpx_group_1,kpx_group_2,kpx_group_3\n"]
    pre=pd.date_range("2022-01-01 01:00","2023-01-01 00:00",freq="h")
    app=runner.Y23
    lines.extend(f"{t:%Y-%m-%d %H:%M:%S},1,2,3\n" for t in pre)
    for i,t in enumerate(app): lines.append(f"{t:%Y-%m-%d %H:%M:%S},{'' if i==0 else '1'},2,3\n")
    path.write_text("".join(lines),encoding="utf-8")
    original=runner.LABEL; runner.LABEL=path
    try:
        cursor=runner.Labels(); frame,_=cursor.read23(); cursor.close()
    finally: runner.LABEL=original
    assert np.isnan(frame.iloc[0,0]) and frame.shape==(8760,3)


def test_stage2_hashes_are_deferred_until_after_stage1_pass() -> None:
    verify_source=inspect.getsource(runner.verify)
    main_source=inspect.getsource(runner.main)
    assert "BASE24" not in verify_source
    assert main_source.index("if passed:") < main_source.index("verify_stage2_inputs(cfg)")

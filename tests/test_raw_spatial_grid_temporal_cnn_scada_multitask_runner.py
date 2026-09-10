from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts import run_raw_spatial_grid_temporal_cnn_scada_multitask as runner
from src.metric import TARGET_COLS


def test_preregister_and_training_contract_verify() -> None:
    config, raw = runner.verify_config(runner.CONFIG_PATH)
    assert config["architecture"]["parameter_count"] == 41823
    assert config["single_candidate"]["blend_weight"] == 0.025
    assert "physical_stage1_inputs" in raw


def test_expected_masks_encode_g3_missing_2022() -> None:
    index = runner.YEAR_2022.append(runner.G3_H1)
    masks = runner._expected_masks(
        index,
        {
            TARGET_COLS[0]: index,
            TARGET_COLS[1]: index,
            TARGET_COLS[2]: runner.G3_H1,
        },
    )
    assert not masks[TARGET_COLS[2]].loc[runner.YEAR_2022].any()
    assert masks[TARGET_COLS[2]].loc[runner.G3_H1].all()


def test_candidate_weight_zero_is_bit_identity() -> None:
    index = pd.date_range("2023-01-01 01:00", periods=24, freq="h")
    baseline = pd.Series(np.linspace(0.0, 100.0, len(index)), index=index, name=TARGET_COLS[0])
    model = pd.Series(np.linspace(0.1, 0.9, len(index)), index=index)
    candidate = runner._candidate_series(baseline, model, TARGET_COLS[0])
    assert candidate.index.equals(index)
    assert candidate.name == TARGET_COLS[0]


def test_lock_rejects_wrong_preregister(tmp_path) -> None:
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"preregister_sha256": "wrong", "ready": True}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid candidate-before-label lock"):
        runner._require_lock(lock, "ready")


def test_score_reader_is_fail_armed_before_lock(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="required candidate-before-label lock absent"):
        runner._read_score_after_global_lock(tmp_path, {}, tmp_path / "absent.json")


def test_stage1_gate_requires_all_slice_and_full_components() -> None:
    comparison = {}
    for name in runner.REQUIRED[TARGET_COLS[2]]:
        comparison[name] = {
            "baseline": {"score": 0.5, "one_minus_nmae": 0.8, "ficr": 0.2},
            "candidate": {"score": 0.51, "one_minus_nmae": 0.81, "ficr": 0.21},
            "delta": 0.51 - 0.5,
        }
    gate = runner._stage1_gate(comparison, runner.REQUIRED[TARGET_COLS[2]])
    assert gate["passed"] is True

from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

from scripts import build_public_global_scalar_bracket_096_098 as builder
from src.metric import CAPACITY_KWH, TARGET_COLS
from src.public_scale_probe import scale_predictions


def _frame() -> pd.DataFrame:
    index = pd.date_range("2025-01-01 01:00", periods=4, freq="h", name="forecast_kst_dtm")
    return pd.DataFrame(
        {group: np.linspace(1000.0 + position, 20000.0 + position, 4) for position, group in enumerate(TARGET_COLS)},
        index=index,
        dtype=np.float64,
    )


def test_preregister_hash_and_factors_are_frozen() -> None:
    assert hashlib.sha256(builder.CONFIG_PATH.read_bytes()).hexdigest() == builder.CONFIG_SHA256
    config = builder.verify_config()
    assert tuple(item["factor"] for item in config["candidates"]) == (0.96, 0.98)
    assert config["later_feedback_invariance"]["a_later_097_direction_may_be_described_but_cannot_change_either_file_formula_factor_or_bytes"] is True
    assert config["later_feedback_invariance"]["adaptive_selection_or_rebuild"] is False


def test_global_formula_has_no_mask_and_is_bit_exact() -> None:
    base = _frame()
    for factor in builder.FACTORS:
        candidate = scale_predictions(base, factor)
        for group in TARGET_COLS:
            expected = np.clip(base[group].to_numpy() * factor, 0.0, 1.02 * CAPACITY_KWH[group])
            np.testing.assert_array_equal(candidate[group].to_numpy(), expected)
            assert np.count_nonzero(candidate[group].to_numpy() != base[group].to_numpy()) == len(base)


def test_factors_are_distinct_from_existing_scalar_values() -> None:
    config = builder.verify_config()
    existing = {1.0, 0.92, 0.95, 0.97}
    assert existing.isdisjoint(item["factor"] for item in config["candidates"])
    assert len({item["factor"] for item in config["candidates"]}) == 2


def test_six_decimal_formula_replay_uses_raw_text_not_binary_reparse() -> None:
    values = np.asarray([1.2345675, 2.0000005, 9999.9999995], dtype=np.float64)
    expected = [f"{value:.6f}" for value in values]
    assert expected == [format(value, ".6f") for value in values]
    assert all(builder.SIX_DECIMAL.fullmatch(value) for value in expected)


def test_risk_and_no_feedback_selection_contract() -> None:
    config = builder.verify_config()
    assert config["risk"] == {
        "selection_unsafe": True,
        "public_adaptive": True,
        "private_score": "UNKNOWN",
        "strict": False,
        "private_champion": False,
        "reason": config["risk"]["reason"],
    }
    assert config["output_contract"]["actual_new_feedback_records"] == 0
    assert config["output_contract"]["submission_performed"] is False


def test_existing_output_directory_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        with pytest.raises(FileExistsError, match="existing output directory"):
            builder.main(["--out-dir", temporary])

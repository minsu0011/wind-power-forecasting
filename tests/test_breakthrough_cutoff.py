from __future__ import annotations

import pandas as pd
import pytest

from src.baram_breakthrough.cutoff import assert_available_before_cutoff


def test_cutoff_accepts_d_minus_1_13_kst() -> None:
    frame = pd.DataFrame({"forecast_kst_dtm": ["2025-01-02 01:00"], "data_available_kst_dtm": ["2024-12-31 13:00"]})
    assert_available_before_cutoff(frame)


def test_cutoff_rejects_post_cutoff() -> None:
    frame = pd.DataFrame({"forecast_kst_dtm": ["2025-01-02 01:00"], "data_available_kst_dtm": ["2025-01-01 15:00"]})
    with pytest.raises(AssertionError):
        assert_available_before_cutoff(frame)

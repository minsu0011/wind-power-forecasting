from __future__ import annotations

import numpy as np
import pandas as pd

from src.baram_breakthrough.metric import TARGETS, parity_report


def test_metric_parity_at_thresholds() -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="h")
    actual = pd.DataFrame({target: [3000.0, 5000.0, 10000.0, 15000.0, 20000.0, np.nan] for target in TARGETS}, index=index)
    prediction = actual.fillna(0.0).copy()
    for target, capacity in zip(TARGETS, (21600.0, 21600.0, 21000.0)):
        prediction[target] = prediction[target] + np.array([0.0, 0.06, 0.08, 0.08001, 0.12, 0.0]) * capacity
    report = parity_report(actual, prediction)
    assert max(abs(value) for value in report["differences"].values()) <= 2e-15

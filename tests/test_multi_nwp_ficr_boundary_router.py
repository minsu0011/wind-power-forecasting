from __future__ import annotations

import numpy as np
import pandas as pd

from src.direct_interval_probability import CONTEXT_COLUMNS
from src.multi_nwp_ficr_boundary_router import (
    ACTION_ALPHAS,
    BASE_FEATURE_COLUMNS,
    PAIR_TRANSFER_WEIGHT,
    build_boundary_context,
    exact_conditional_utility,
    expanded_action_design,
    repair_probabilities,
)
from src.multi_nwp_joint import JOINT_COLUMNS


def _inputs(rows: int = 4):
    index = pd.date_range("2024-05-01", periods=rows, freq="h", name="forecast_kst_dtm")
    canonical = pd.DataFrame(1.0, index=index, columns=CONTEXT_COLUMNS)
    joint = pd.DataFrame(2.0, index=index, columns=JOINT_COLUMNS)
    baseline = pd.Series(np.linspace(2000.0, 5000.0, rows), index=index)
    increment = pd.Series(np.asarray([0.04, -0.02, 0.01, -0.03])[:rows], index=index)
    return canonical, joint, baseline, increment


def test_context_and_action_grid_are_exact() -> None:
    canonical, joint, baseline, increment = _inputs()
    context = build_boundary_context(
        canonical, joint, baseline, increment, capacity_kwh=20_000.0
    )
    assert tuple(context.columns) == BASE_FEATURE_COLUMNS
    design, action = expanded_action_design(context)
    assert design.shape == (len(context) * 3, len(BASE_FEATURE_COLUMNS) + 4)
    expected = np.clip(
        baseline.to_numpy()[:, None] / 20_000.0
        + PAIR_TRANSFER_WEIGHT * increment.to_numpy()[:, None] * ACTION_ALPHAS[None, :],
        0.0,
        1.02,
    )
    np.testing.assert_allclose(action, expected, rtol=0.0, atol=1e-7)


def test_probability_repair_and_exact_utility_formula() -> None:
    p6, p8, repairs = repair_probabilities(
        np.asarray([[0.8, 0.2, 0.3]]), np.asarray([[0.6, 0.4, 0.3]])
    )
    assert repairs == 1
    assert np.all(p6 <= p8)
    eae = np.asarray([[0.1, 0.2, 0.3]])
    observed = exact_conditional_utility(p6, p8, eae)
    expected = -0.5 * eae + 0.5 * (0.25 * p6 + 0.75 * p8)
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=0.0)

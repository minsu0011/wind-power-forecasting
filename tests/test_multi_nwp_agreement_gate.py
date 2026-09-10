from __future__ import annotations

import numpy as np
import pandas as pd

from src.multi_nwp_agreement_gate import SOURCE_ORDER, agreement_gated_increment


def _frame(values: list[float]) -> pd.DataFrame:
    index = pd.date_range("2023-01-01", periods=len(values), freq="h")
    return pd.DataFrame({"kpx_group_1": values}, index=index)


def test_majority_vote_and_magnitude_cap_are_exact() -> None:
    joint = _frame([0.30, 0.30, -0.30, 0.0])
    sources = {
        "ecmwf": _frame([0.10, 0.10, -0.60, 0.10]),
        "icon": _frame([0.20, -0.20, -0.20, -0.10]),
        "gfs": _frame([0.40, 0.40, 0.50, 0.20]),
    }
    gated, confidence, count = agreement_gated_increment(joint, sources)
    np.testing.assert_array_equal(count.iloc[:, 0].to_numpy(), [3, 2, 2, 0])
    np.testing.assert_allclose(
        confidence.iloc[:, 0].to_numpy(),
        [2.0 / 3.0, 4.0 / 9.0, 2.0 / 3.0, 0.0],
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        gated.iloc[:, 0].to_numpy(),
        [0.20, 2.0 / 15.0, -0.20, 0.0],
        rtol=0.0,
        atol=1e-15,
    )


def test_no_majority_means_exact_zero_and_gate_never_enlarges() -> None:
    joint = _frame([0.2, -0.2])
    sources = {
        "ecmwf": _frame([0.1, 0.1]),
        "icon": _frame([-0.1, -0.1]),
        "gfs": _frame([-0.2, 0.1]),
    }
    gated, confidence, count = agreement_gated_increment(joint, sources)
    np.testing.assert_array_equal(count.iloc[:, 0].to_numpy(), [1, 1])
    np.testing.assert_array_equal(confidence.iloc[:, 0].to_numpy(), [0.0, 0.0])
    np.testing.assert_array_equal(gated.iloc[:, 0].to_numpy(), [0.0, 0.0])


def test_source_order_and_schema_are_enforced() -> None:
    joint = _frame([0.1])
    frame = _frame([0.1])
    wrong_order = {"icon": frame, "ecmwf": frame, "gfs": frame}
    try:
        agreement_gated_increment(joint, wrong_order)
    except ValueError as error:
        assert str(SOURCE_ORDER) in str(error)
    else:
        raise AssertionError("wrong source order was accepted")

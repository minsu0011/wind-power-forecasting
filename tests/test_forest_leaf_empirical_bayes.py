from __future__ import annotations

import numpy as np
import pandas as pd

from src.forest_leaf_empirical_bayes import (
    FOREST_PARAMETERS,
    ExtraTreesLeafEmpiricalBayes,
    empirical_expected_utility_actions,
    qrf_weight_matrix_from_leaf_ids,
)
from src.residual_histogram_bayes import ACTION_DELTAS_CF


def test_qrf_weights_match_explicit_leaf_formula() -> None:
    fit = np.asarray([[1, 3], [1, 4], [2, 3], [2, 4]], dtype=np.int64)
    query = np.asarray([[1, 3], [2, 4]], dtype=np.int64)
    weights = qrf_weight_matrix_from_leaf_ids(fit, query)
    expected = np.asarray(
        [[0.50, 0.25, 0.25, 0.00], [0.00, 0.25, 0.25, 0.50]],
        dtype=np.float64,
    )
    assert np.array_equal(weights, expected)
    assert np.array_equal(weights.sum(axis=1), np.ones(2))


def _brute_action(
    weight: np.ndarray, residual: np.ndarray, base: float, mean: float
) -> tuple[float, float]:
    outcomes = np.clip(base + residual, 0.10, 1.20)
    actions = np.clip(base + ACTION_DELTAS_CF, 0.0, 1.02)
    error = np.abs(actions[:, None] - outcomes[None, :])
    settlement = np.select(
        [error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0
    )
    utility = np.sum(
        weight[None, :]
        * (-error + outcomes[None, :] * settlement / (4.0 * mean)),
        axis=1,
    )
    maximum = float(np.max(utility))
    tied = np.flatnonzero(utility >= maximum - 1e-12)
    deltas = ACTION_DELTAS_CF[tied]
    minimum_abs = float(np.min(np.abs(deltas)))
    tied = tied[np.abs(deltas) <= minimum_abs + 1e-15]
    chosen = int(tied[0])
    return float(ACTION_DELTAS_CF[chosen]), float(utility[chosen])


def test_empirical_action_matches_dense_bruteforce() -> None:
    weights = np.asarray(
        [[0.10, 0.20, 0.30, 0.40], [0.40, 0.30, 0.20, 0.10]],
        dtype=np.float64,
    )
    residual = np.asarray([-0.14, -0.03, 0.04, 0.17], dtype=np.float64)
    base = np.asarray([0.31, 0.83], dtype=np.float64)
    selected, utility, diagnostic = empirical_expected_utility_actions(
        weights, residual, base, mean_actual_cf=0.44
    )
    brute = [_brute_action(weights[row], residual, base[row], 0.44) for row in range(2)]
    assert np.array_equal(selected, np.asarray([value[0] for value in brute]))
    assert np.allclose(
        utility, np.asarray([value[1] for value in brute]), atol=1e-14, rtol=0.0
    )
    assert np.array_equal(diagnostic["nonzero_support_count"], np.asarray([4, 4]))


def test_model_contract_and_reload_inputs() -> None:
    rng = np.random.default_rng(1234)
    fit_index = pd.date_range("2023-01-01 01:00", periods=160, freq="h")
    apply_index = pd.date_range("2024-01-01 01:00", periods=12, freq="h")
    columns = [f"feature_{value:03d}" for value in range(615)]
    fit = pd.DataFrame(
        rng.normal(size=(len(fit_index), len(columns))).astype(np.float32),
        index=fit_index,
        columns=columns,
    )
    application = pd.DataFrame(
        rng.normal(size=(len(apply_index), len(columns))).astype(np.float32),
        index=apply_index,
        columns=columns,
    )
    baseline = pd.Series(9000.0, index=fit_index)
    actual = pd.Series(
        np.clip(9000.0 + 1000.0 * rng.normal(size=len(fit_index)), 2500.0, 17000.0),
        index=fit_index,
    )
    actual.iloc[:3] = np.nan
    model = ExtraTreesLeafEmpiricalBayes().fit(
        fit, actual, baseline, capacity_kwh=21600.0
    )
    action, leaf, diagnostic = model.predict_action(
        application, pd.Series(9000.0, index=apply_index)
    )
    assert len(action) == len(application)
    assert leaf.shape == (len(application), FOREST_PARAMETERS["n_estimators"])
    assert np.allclose(diagnostic["qrf_weight_sum"], 1.0, atol=1e-12, rtol=0.0)
    assert model.metadata()["leaf_size_min"] >= FOREST_PARAMETERS["min_samples_leaf"]


def test_fit_label_nan_is_excluded_but_infinity_is_rejected() -> None:
    rng = np.random.default_rng(44)
    index = pd.date_range("2023-01-01 01:00", periods=132, freq="h")
    columns = [f"feature_{value:03d}" for value in range(615)]
    features = pd.DataFrame(
        rng.normal(size=(len(index), len(columns))).astype(np.float32),
        index=index,
        columns=columns,
    )
    baseline = pd.Series(9000.0, index=index)
    actual = pd.Series(9500.0, index=index)
    actual.iloc[:4] = np.nan
    model = ExtraTreesLeafEmpiricalBayes().fit(
        features, actual, baseline, capacity_kwh=21600.0
    )
    assert model.metadata()["fit_rows_eligible"] == 128
    bad_actual = actual.copy()
    bad_actual.iloc[0] = np.inf
    try:
        ExtraTreesLeafEmpiricalBayes().fit(
            features, bad_actual, baseline, capacity_kwh=21600.0
        )
    except ValueError as exc:
        assert "infinity" in str(exc)
    else:
        raise AssertionError("infinite fit label was accepted")
    bad_baseline = baseline.copy()
    bad_baseline.iloc[0] = np.nan
    try:
        ExtraTreesLeafEmpiricalBayes().fit(
            features, actual, bad_baseline, capacity_kwh=21600.0
        )
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("NaN fit baseline was accepted")

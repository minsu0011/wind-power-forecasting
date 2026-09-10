from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.full_weather_ordinal_bayes import (
    ACTION_DELTAS_CF,
    BIN_CENTRES_CF,
    BIN_COUNT,
    BIN_WIDTH_CF,
    FullWeatherOrdinalBayes,
    MODEL_PARAMETERS,
    TRANSFER_WEIGHT,
    expected_official_utility,
    fixed_probability_matrix,
    group_stage1_pass,
    ordinal_bayes_action_cf,
    target_classes,
    transfer_action_kwh,
)


def test_frozen_bin_model_action_contract() -> None:
    assert BIN_COUNT == 42
    assert BIN_WIDTH_CF == 0.025
    np.testing.assert_allclose(
        BIN_CENTRES_CF,
        0.1125 + 0.025 * np.arange(42, dtype=np.float64),
        rtol=0,
        atol=3e-16,
    )
    np.testing.assert_allclose(
        ACTION_DELTAS_CF,
        np.arange(-0.150, 0.150 + 0.0025, 0.005, dtype=np.float64),
        rtol=0,
        atol=3e-16,
    )
    assert len(ACTION_DELTAS_CF) == 61
    assert TRANSFER_WEIGHT == 0.10
    assert MODEL_PARAMETERS == {
        "boosting_type": "gbdt",
        "objective": "multiclass",
        "n_estimators": 120,
        "learning_rate": 0.035,
        "num_leaves": 7,
        "max_depth": 3,
        "min_child_samples": 100,
        "min_split_gain": 0.0,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.35,
        "reg_alpha": 0.5,
        "reg_lambda": 10.0,
        "max_bin": 63,
        "random_state": 42,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
        "n_jobs": 7,
    }


def test_target_classes_fixed_edges_and_absorbing_upper_tail() -> None:
    values = np.array(
        [0.10, np.nextafter(0.125, 0.0), 0.125, 1.124999, 1.125, 1.15, 2.0]
    )
    np.testing.assert_array_equal(target_classes(values), [0, 0, 1, 40, 41, 41, 41])
    with pytest.raises(ValueError, match="eligible"):
        target_classes(np.array([0.099]))


def test_fixed_probability_mapping_smoke_is_exact_42_columns() -> None:
    raw = np.array([[0.2, 0.3, 0.5], [0.0, 1.0, 0.0]], dtype=np.float64)
    mapped = fixed_probability_matrix(raw, np.array([0, 2, 41]))
    assert mapped.shape == (2, 42)
    np.testing.assert_array_equal(mapped.sum(axis=1), [1.0, 1.0])
    np.testing.assert_array_equal(np.flatnonzero(mapped[0]), [0, 2, 41])
    np.testing.assert_array_equal(mapped[0, [0, 2, 41]], [0.2, 0.3, 0.5])
    remaining = [value for value in range(42) if value not in (0, 2, 41)]
    assert np.all(mapped[:, remaining] == 0.0)
    with pytest.raises(ValueError, match="sum"):
        fixed_probability_matrix(raw * 0.9, np.array([0, 2, 41]))


def test_expected_utility_and_action_match_independent_bruteforce() -> None:
    probabilities = np.zeros((2, 42), dtype=np.float64)
    probabilities[0, 8] = 1.0
    probabilities[1, [4, 20]] = [0.4, 0.6]
    baseline = np.array([0.36, 0.58], dtype=np.float64)
    actions = np.clip(baseline[:, None] + ACTION_DELTAS_CF, 0.0, 1.02)
    utility = expected_official_utility(
        actions, probabilities, mean_train_actual_cf=0.45
    )
    selected, diagnostics = ordinal_bayes_action_cf(
        probabilities, baseline, mean_train_actual_cf=0.45
    )
    expected = []
    for row in range(2):
        maximum = utility[row].max()
        tied = np.flatnonzero(utility[row] >= maximum - 1e-12)
        order = np.lexsort(
            (actions[row, tied], np.abs(actions[row, tied] - baseline[row]))
        )
        expected.append(actions[row, tied[int(order[0])]])
    np.testing.assert_array_equal(selected, expected)
    np.testing.assert_array_equal(diagnostics["raw_action_cf"], selected)
    np.testing.assert_allclose(
        diagnostics["raw_action_delta_cf"], selected - baseline, rtol=0, atol=0
    )


def test_transfer_is_exact_fixed_point_one_and_bounded() -> None:
    index = pd.date_range("2023-01-01 01:00", periods=3, freq="h")
    baseline = pd.Series([2160.0, 10800.0, 21600.0], index=index, name="g")
    action = pd.Series([0.25, 0.35, 1.02], index=index, name="g")
    result = transfer_action_kwh(baseline, action, capacity_kwh=21600.0)
    baseline_cf = baseline.to_numpy() / 21600.0
    expected_cf = np.clip(0.9 * baseline_cf + 0.1 * action.to_numpy(), 0.0, 1.02)
    np.testing.assert_array_equal(result.to_numpy(), expected_cf * 21600.0)
    assert np.max(
        np.abs(result.to_numpy() / 21600.0 - baseline.to_numpy() / 21600.0)
    ) <= 0.015 + 1e-15


def _record(delta: float, nmae: float = 0.001, ficr: float = 0.001) -> dict:
    baseline = {"score": 0.5, "one_minus_nmae": 0.8, "ficr": 0.2}
    candidate = {
        "score": 0.5 + delta,
        "one_minus_nmae": 0.8 + nmae,
        "ficr": 0.2 + ficr,
    }
    return {"baseline": baseline, "candidate": candidate, "delta": delta}


def test_group_stage1_pass_requires_all_slices_and_full_components() -> None:
    names = ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4")
    comparisons = {name: _record(0.001) for name in names}
    passed, diagnostics = group_stage1_pass(comparisons, "kpx_group_1")
    assert passed and diagnostics["passed"]
    comparisons["Q2"] = _record(0.0)
    assert not group_stage1_pass(comparisons, "kpx_group_1")[0]
    comparisons["Q2"] = _record(0.001)
    comparisons["full"] = _record(0.001, nmae=-0.0001, ficr=0.0021)
    assert not group_stage1_pass(comparisons, "kpx_group_1")[0]


def test_actual_lightgbm_mapping_predicts_fixed_42_columns() -> None:
    rng = np.random.default_rng(4201)
    fit_index = pd.date_range("2022-01-01 01:00", periods=500, freq="h")
    apply_index = pd.date_range("2023-01-01 01:00", periods=12, freq="h")
    fit_features = pd.DataFrame(
        rng.normal(size=(500, 6)).astype(np.float32),
        index=fit_index,
        columns=[f"f{value}" for value in range(6)],
    )
    apply_features = pd.DataFrame(
        rng.normal(size=(12, 6)).astype(np.float32),
        index=apply_index,
        columns=fit_features.columns,
    )
    target_cf = 0.105 + 0.099 * (np.arange(500) % 5) / 4.0
    actual = pd.Series(target_cf * 21600.0, index=fit_index)
    model = FullWeatherOrdinalBayes().fit(
        fit_features, actual, capacity_kwh=21600.0
    )
    probability = model.predict_probability(apply_features)
    assert probability.shape == (12, 42)
    np.testing.assert_allclose(probability.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    observed = set(model.observed_classes_)
    unseen = [value for value in range(42) if value not in observed]
    assert np.all(probability[[f"class_{value:02d}" for value in unseen]] == 0.0)
    assert model.metadata()["fit_score_calculated"] is False
    with pytest.raises(ValueError, match="same-row"):
        model.predict_probability(fit_features.iloc[:5])

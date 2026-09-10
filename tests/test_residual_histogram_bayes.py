from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.residual_histogram_bayes import (
    ACTION_DELTAS_CF,
    COMPONENT_NAMES,
    META_FEATURE_COLUMNS,
    RESIDUAL_CENTRES_CF,
    ResidualHistogramBayesClassifier,
    WEATHER_COLUMNS,
    apply_action_shrink,
    build_meta_features,
    choose_group_shrinks,
    expected_utility_actions,
    residual_class_index,
)


def fixture(rows: int = 720, start: str = "2023-01-01 01:00:00"):
    index = pd.date_range(start, periods=rows, freq="h", name="forecast_kst_dtm")
    hour = np.arange(rows, dtype=float)
    base = pd.Series(9000.0 + 1500.0 * np.sin(hour / 24.0), index=index)
    components = {
        name: pd.Series(base.to_numpy() + 50.0 * position, index=index)
        for position, name in enumerate(COMPONENT_NAMES)
    }
    weather = pd.DataFrame(
        {
            name: 4.0 + position + 0.01 * hour
            for position, name in enumerate(WEATHER_COLUMNS)
        },
        index=index,
    )
    features = build_meta_features(
        base, components, weather, capacity_kwh=21600.0
    )
    residual = np.choose(hour.astype(int) % 3, [-0.025, 0.0, 0.025])
    actual = pd.Series(
        (base.to_numpy() / 21600.0 + residual) * 21600.0,
        index=index,
    )
    return features, actual, base, components, weather


class ResidualHistogramBayesTests(unittest.TestCase):
    def test_frozen_grids_are_exact(self) -> None:
        self.assertEqual(len(RESIDUAL_CENTRES_CF), 43)
        self.assertEqual(len(ACTION_DELTAS_CF), 61)
        self.assertAlmostEqual(float(RESIDUAL_CENTRES_CF[0]), -0.525)
        self.assertAlmostEqual(float(RESIDUAL_CENTRES_CF[-1]), 0.525)
        self.assertAlmostEqual(float(ACTION_DELTAS_CF[0]), -0.15)
        self.assertAlmostEqual(float(ACTION_DELTAS_CF[-1]), 0.15)
        np.testing.assert_allclose(np.diff(RESIDUAL_CENTRES_CF), 0.025)
        np.testing.assert_allclose(np.diff(ACTION_DELTAS_CF), 0.005)

    def test_residual_class_tails_and_nearest_centres(self) -> None:
        residual = np.array([-9.0, -0.525, -0.50, 0.0, 0.50, 0.525, 9.0])
        np.testing.assert_array_equal(
            residual_class_index(residual), [0, 0, 1, 21, 41, 42, 42]
        )
        self.assertEqual(int(residual_class_index(np.array([-0.5125]))[0]), 0)
        self.assertEqual(int(residual_class_index(np.array([0.5125001]))[0]), 42)

    def test_meta_feature_schema_and_values(self) -> None:
        features, _, base, components, weather = fixture(48)
        self.assertEqual(tuple(features.columns), META_FEATURE_COLUMNS)
        self.assertEqual(features.shape, (48, 25))
        self.assertAlmostEqual(features.iloc[0]["base_cf"], base.iloc[0] / 21600.0)
        values = np.array([components[name].iloc[0] for name in COMPONENT_NAMES]) / 21600.0
        self.assertAlmostEqual(features.iloc[0]["component_mean_cf"], values.mean())
        self.assertAlmostEqual(features.iloc[0]["component_std_cf"], values.std(ddof=0))
        self.assertAlmostEqual(features.iloc[0]["lead_fraction"], 0.0)
        self.assertAlmostEqual(features.iloc[23]["lead_fraction"], 1.0)
        bad = weather.rename(columns={WEATHER_COLUMNS[0]: "scada_actual"})
        with self.assertRaises(KeyError):
            build_meta_features(base, components, bad, capacity_kwh=21600.0)

    def test_one_hot_outcome_selects_exact_residual_action(self) -> None:
        probability = np.zeros((1, 43))
        probability[0, 25] = 1.0  # residual centre +0.10
        action, _ = expected_utility_actions(
            probability, np.array([0.50]), mean_actual_cf=0.50
        )
        self.assertAlmostEqual(float(action[0]), 0.10, places=12)

    def test_expected_utility_matches_independent_bruteforce(self) -> None:
        probability = np.zeros((1, 43))
        probability[0, [18, 21, 25]] = [0.2, 0.5, 0.3]
        base = 0.97
        mean = 0.45
        action, utility = expected_utility_actions(
            probability, np.array([base]), mean_actual_cf=mean
        )
        outcomes = np.clip(base + RESIDUAL_CENTRES_CF, 0.10, 1.20)
        expected = []
        for delta in ACTION_DELTAS_CF:
            candidate = np.clip(base + delta, 0.0, 1.02)
            error = np.abs(candidate - outcomes)
            unit = np.where(error <= 0.06, 4.0, np.where(error <= 0.08, 3.0, 0.0))
            value = -error + outcomes * unit / (4.0 * mean)
            expected.append(float(np.dot(probability[0], value)))
        maximum = max(expected)
        tied = [i for i, value in enumerate(expected) if value >= maximum - 1e-12]
        chosen = min(tied, key=lambda i: (abs(ACTION_DELTAS_CF[i]), ACTION_DELTAS_CF[i]))
        self.assertAlmostEqual(float(action[0]), float(ACTION_DELTAS_CF[chosen]), places=12)
        self.assertAlmostEqual(float(utility[0]), expected[chosen], places=12)

    def test_classifier_is_deterministic_and_rejects_overlap(self) -> None:
        fit_features, actual, base, _, _ = fixture(720)
        apply_features, _, apply_base, _, _ = fixture(72, "2023-02-01 01:00:00")
        params = {
            "boosting_type": "gbdt",
            "objective": "multiclass",
            "n_estimators": 12,
            "learning_rate": 0.05,
            "num_leaves": 5,
            "max_depth": 3,
            "min_child_samples": 20,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.5,
            "reg_lambda": 10.0,
            "max_bin": 63,
            "random_state": 42,
            "deterministic": True,
            "force_col_wise": True,
            "verbosity": -1,
            "n_jobs": 1,
        }
        first = ResidualHistogramBayesClassifier(
            model_parameters=params, minimum_eligible_fit_rows=300
        ).fit(fit_features, actual, base, capacity_kwh=21600.0)
        second = ResidualHistogramBayesClassifier(
            model_parameters=params, minimum_eligible_fit_rows=300
        ).fit(fit_features, actual, base, capacity_kwh=21600.0)
        first_action, first_utility, first_prob = first.predict_raw_action(
            apply_features, apply_base
        )
        second_action, second_utility, second_prob = second.predict_raw_action(
            apply_features, apply_base
        )
        np.testing.assert_array_equal(first_action.to_numpy(), second_action.to_numpy())
        np.testing.assert_array_equal(first_utility.to_numpy(), second_utility.to_numpy())
        np.testing.assert_array_equal(first_prob, second_prob)
        np.testing.assert_allclose(first_prob.sum(axis=1), 1.0, atol=1e-12)
        with self.assertRaises(ValueError):
            first.predict_raw_action(fit_features.iloc[:24], base.iloc[:24])

    def test_action_shrink_and_group_selection(self) -> None:
        index = pd.date_range("2023-01-01", periods=3, freq="h")
        base = pd.Series([10000.0, 10000.0, 10000.0], index=index)
        action = pd.Series([0.1, -0.1, 0.1], index=index)
        candidate = apply_action_shrink(
            base, action, capacity_kwh=20000.0, shrink=0.5
        )
        np.testing.assert_allclose(candidate, [11000.0, 9000.0, 11000.0])
        comparisons = {
            "0.25": {
                "g1": {"full": {"delta": 0.02}, "Q4": {"delta": 0.01}},
                "g2": {"full": {"delta": -0.01}, "Q4": {"delta": 0.01}},
            },
            "0.5": {
                "g1": {"full": {"delta": 0.03}, "Q4": {"delta": 0.005}},
                "g2": {"full": {"delta": -0.02}, "Q4": {"delta": -0.01}},
            },
            "1.0": {
                "g1": {"full": {"delta": -0.01}, "Q4": {"delta": 0.02}},
                "g2": {"full": {"delta": -0.03}, "Q4": {"delta": 0.02}},
            },
        }
        self.assertEqual(
            choose_group_shrinks(comparisons, ["g1", "g2"]),
            {"g1": 0.25, "g2": None},
        )


if __name__ == "__main__":
    unittest.main()

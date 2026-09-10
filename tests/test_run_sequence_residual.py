from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from src.run_sequence_residual import (
    apply_residual_model,
    assert_strict_fit_predict_order,
    build_run_sequence_features,
    choose_single_candidate,
    validate_complete_runs,
)


WEATHER_COLUMNS = (
    "cross__hub_ws_mean",
    "ldaps__idw__hub_ws",
)


def _two_runs() -> tuple[pd.DataFrame, pd.Series]:
    index = pd.date_range("2023-01-01 01:00:00", periods=48, freq="h")
    weather = pd.DataFrame(
        {
            "cross__hub_ws_mean": np.r_[np.arange(24), 100 + np.arange(24)],
            "ldaps__idw__hub_ws": np.r_[2 * np.arange(24), 200 + np.arange(24)],
        },
        index=index,
        dtype=float,
    )
    baseline = pd.Series(np.r_[1000 + 10 * np.arange(24), 3000 + np.arange(24)], index=index)
    return weather, baseline


class RunSequenceFeatureTests(unittest.TestCase):
    def test_future_horizon_inside_run_is_allowed_but_next_run_is_isolated(self) -> None:
        weather, baseline = _two_runs()
        original = build_run_sequence_features(
            weather,
            baseline,
            capacity_kwh=10_000.0,
            weather_columns=WEATHER_COLUMNS,
        )
        next_run_changed = weather.copy()
        next_run_changed.iloc[24:, :] += 10_000.0
        isolated = build_run_sequence_features(
            next_run_changed,
            baseline,
            capacity_kwh=10_000.0,
            weather_columns=WEATHER_COLUMNS,
        )
        np.testing.assert_array_equal(
            original.iloc[:24].to_numpy(), isolated.iloc[:24].to_numpy()
        )

        same_run_changed = weather.copy()
        same_run_changed.iloc[23, 0] += 1000.0
        shared = build_run_sequence_features(
            same_run_changed,
            baseline,
            capacity_kwh=10_000.0,
            weather_columns=WEATHER_COLUMNS,
        )
        self.assertNotEqual(
            original.iloc[0]["cross__hub_ws_mean__run_mean"],
            shared.iloc[0]["cross__hub_ws_mean__run_mean"],
        )
        np.testing.assert_array_equal(
            original.iloc[24:].to_numpy(), shared.iloc[24:].to_numpy()
        )

    def test_incomplete_or_cross_run_index_is_rejected(self) -> None:
        weather, baseline = _two_runs()
        with self.assertRaises(ValueError):
            validate_complete_runs(weather.index[:-1])
        broken = weather.drop(weather.index[12])
        with self.assertRaises(ValueError):
            build_run_sequence_features(
                broken,
                baseline.loc[broken.index],
                capacity_kwh=10_000.0,
                weather_columns=WEATHER_COLUMNS,
            )

    def test_target_scada_or_group_columns_are_rejected(self) -> None:
        weather, baseline = _two_runs()
        for forbidden in ("actual", "target_cf", "scada_wind", "kpx_group_1"):
            contaminated = weather.assign(**{forbidden: 0.0})
            with self.assertRaises(ValueError):
                build_run_sequence_features(
                    contaminated,
                    baseline,
                    capacity_kwh=10_000.0,
                    weather_columns=WEATHER_COLUMNS,
                )

    def test_feature_count_is_compact_and_finite(self) -> None:
        weather, baseline = _two_runs()
        features = build_run_sequence_features(
            weather,
            baseline,
            capacity_kwh=10_000.0,
            weather_columns=WEATHER_COLUMNS,
        )
        self.assertEqual(features.shape, (48, 85))
        self.assertTrue(np.isfinite(features.to_numpy()).all())


class RunSequenceProtocolTests(unittest.TestCase):
    def test_fit_must_be_strictly_before_prediction(self) -> None:
        fit = pd.date_range("2023-01-01", periods=24, freq="h")
        valid = pd.date_range("2023-01-02", periods=24, freq="h")
        assert_strict_fit_predict_order(fit, valid)
        with self.assertRaises(ValueError):
            assert_strict_fit_predict_order(fit, fit)
        with self.assertRaises(ValueError):
            assert_strict_fit_predict_order(valid, fit)

    def test_selection_requires_every_group_and_slice_positive(self) -> None:
        comparisons = {
            "all_positive": {
                "g1": {"full": {"delta": 0.01}, "H2": {"delta": 0.001}},
                "g2": {"full": {"delta": 0.002}},
            },
            "one_negative": {
                "g1": {"full": {"delta": 0.02}, "H2": {"delta": -1e-6}},
                "g2": {"full": {"delta": 0.03}},
            },
        }
        self.assertEqual(choose_single_candidate(comparisons), "all_positive")
        self.assertIsNone(choose_single_candidate({"bad": comparisons["one_negative"]}))

    def test_residual_application_is_bounded(self) -> None:
        index = pd.date_range("2023-01-01 01:00:00", periods=24, freq="h")
        features = pd.DataFrame({"x": np.ones(24)}, index=index)
        baseline = pd.Series(np.full(24, 10_000.0), index=index, name="g")
        model = DummyRegressor(strategy="constant", constant=1.0).fit(
            features, np.ones(24)
        )
        specification = {
            "max_abs_residual_cf": 0.1,
            "correction_shrinkage": 0.25,
        }
        prediction, correction = apply_residual_model(
            model,
            specification,
            features,
            baseline,
            capacity_kwh=10_000.0,
        )
        np.testing.assert_allclose(correction, 0.025)
        np.testing.assert_allclose(prediction, 10_200.0)


if __name__ == "__main__":
    unittest.main()

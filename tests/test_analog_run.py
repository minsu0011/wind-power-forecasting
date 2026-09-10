from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.analog_run import AnalogRunRetriever, blend_analog_with_baseline


CHANNELS = ("cross__hub_ws_mean", "ldaps__idw__hub_ws")


def _weather(start: str, runs: int, *, offset: float = 0.0) -> pd.DataFrame:
    index = pd.date_range(start, periods=runs * 24, freq="h")
    horizon = np.tile(np.arange(24, dtype=float), runs)
    run = np.repeat(np.arange(runs, dtype=float), 24)
    return pd.DataFrame(
        {
            CHANNELS[0]: offset + 3.0 + 0.1 * run + np.sin(horizon / 24 * 2 * np.pi),
            CHANNELS[1]: offset + 4.0 + 0.05 * run + np.cos(horizon / 24 * 2 * np.pi),
        },
        index=index,
    )


def _actual(index: pd.DatetimeIndex) -> pd.Series:
    horizon = np.tile(np.arange(24, dtype=float), len(index) // 24)
    return pd.Series(0.3 + 0.2 * np.sin(horizon / 24 * 2 * np.pi), index=index)


class AnalogRunTests(unittest.TestCase):
    def test_neighbors_are_strictly_earlier_and_prediction_is_bounded(self) -> None:
        train = _weather("2022-01-01 01:00:00", 10)
        apply = _weather("2023-01-01 01:00:00", 3, offset=0.2)
        model = AnalogRunRetriever(CHANNELS, "compact_summary", k=3).fit(
            train, _actual(train.index)
        )
        prediction, audit = model.predict(apply)
        self.assertEqual(len(prediction), 72)
        self.assertTrue(prediction.between(0.0, 1.02).all())
        self.assertTrue(audit["neighbor_strictly_before_apply"])
        self.assertEqual(audit["neighbor_timestamp_overlap_count"], 0)
        self.assertGreater(audit["minimum_neighbor_gap_hours"], 0.0)

    def test_overlap_or_future_neighbor_library_is_rejected(self) -> None:
        train = _weather("2023-01-01 01:00:00", 5)
        model = AnalogRunRetriever(CHANNELS, "compact_summary", k=2).fit(
            train, _actual(train.index)
        )
        with self.assertRaises(ValueError):
            model.predict(train)

    def test_incomplete_actual_run_is_excluded_from_neighbors(self) -> None:
        train = _weather("2022-01-01 01:00:00", 6)
        actual = _actual(train.index)
        actual.iloc[30] = np.nan
        model = AnalogRunRetriever(CHANNELS, "compact_summary", k=3).fit(
            train, actual
        )
        self.assertEqual(model.fit_audit_["weather_runs"], 6)
        self.assertEqual(model.fit_audit_["complete_actual_neighbor_runs"], 5)
        self.assertEqual(model.fit_audit_["excluded_incomplete_actual_runs"], 1)

    def test_forbidden_actual_target_scada_weather_is_rejected(self) -> None:
        train = _weather("2022-01-01 01:00:00", 6)
        actual = _actual(train.index)
        for name in ("actual", "target_cf", "scada_wind", "kpx_group_1"):
            contaminated = train.assign(**{name: 0.0})
            with self.assertRaises(ValueError):
                AnalogRunRetriever(CHANNELS, "compact_summary", k=3).fit(
                    contaminated, actual
                )

    def test_pca_is_fit_on_train_and_has_registered_dimension(self) -> None:
        train = _weather("2022-01-01 01:00:00", 20)
        apply = _weather("2023-01-01 01:00:00", 2, offset=0.5)
        params = {
            "n_components": 12,
            "whiten": True,
            "svd_solver": "full",
            "random_state": 42,
        }
        model = AnalogRunRetriever(
            CHANNELS, "pca12_core_trajectory", k=3, pca_params=params
        ).fit(train, _actual(train.index))
        scaler_mean = model.scaler_.mean_.copy()
        prediction, _ = model.predict(apply)
        self.assertEqual(model.train_embedding_.shape, (20, 12))
        np.testing.assert_array_equal(model.scaler_.mean_, scaler_mean)
        self.assertTrue(np.isfinite(prediction).all())

    def test_prediction_is_deterministic(self) -> None:
        train = _weather("2022-01-01 01:00:00", 10)
        apply = _weather("2023-01-01 01:00:00", 2)
        model = AnalogRunRetriever(CHANNELS, "compact_summary", k=3).fit(
            train, _actual(train.index)
        )
        first, first_audit = model.predict(apply)
        second, second_audit = model.predict(apply)
        np.testing.assert_array_equal(first.to_numpy(), second.to_numpy())
        self.assertEqual(first_audit["chosen_position_sha256"], second_audit["chosen_position_sha256"])

    def test_blend_weights_and_clip_are_exact(self) -> None:
        index = pd.date_range("2023-01-01 01:00:00", periods=24, freq="h")
        analog = pd.Series(np.full(24, 1.02), index=index)
        baseline = pd.Series(np.full(24, 5_000.0), index=index, name="g")
        output = blend_analog_with_baseline(
            analog,
            baseline,
            capacity_kwh=10_000.0,
            analog_weight=0.2,
        )
        np.testing.assert_allclose(output, 0.8 * 5_000.0 + 0.2 * 10_200.0)


if __name__ == "__main__":
    unittest.main()

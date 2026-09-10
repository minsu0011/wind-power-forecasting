from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.temporal_mlp import (
    RunDCTTransformer,
    blend_joint_prediction,
    build_joint_cf_target,
    build_run_channel_tensor,
    choose_group_candidates,
    fit_seed_ensemble,
    orthonormal_dct2_basis,
    predict_seed_ensemble,
)


WEATHER_COLUMNS = (
    "cross__hub_ws_mean",
    "ldaps__idw__hub_ws",
    "gfs__idw__hub_ws",
    "ldaps__idw__wind_power_density",
    "gfs__idw__wind_power_density",
)


def fixtures(runs: int = 4) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    index = pd.date_range("2023-01-01 01:00:00", periods=24 * runs, freq="h")
    hour = np.tile(np.arange(24, dtype=float), runs)
    day = np.repeat(np.arange(runs, dtype=float), 24)
    weather = pd.DataFrame(
        {
            name: 5.0 + position + 0.1 * hour + 0.2 * day
            for position, name in enumerate(WEATHER_COLUMNS)
        },
        index=index,
    )
    baseline = pd.Series(6000.0 + 100.0 * hour + 30.0 * day, index=index)
    actual = pd.Series(6200.0 + 90.0 * hour + 25.0 * day, index=index)
    return weather, baseline, actual


class TemporalMLPTests(unittest.TestCase):
    def test_dct_basis_is_orthonormal(self) -> None:
        basis = orthonormal_dct2_basis(24, 8)
        self.assertEqual(basis.shape, (8, 24))
        np.testing.assert_allclose(basis @ basis.T, np.eye(8), atol=1e-12, rtol=0)

    def test_run_tensor_has_frozen_channel_order(self) -> None:
        weather, baseline, _ = fixtures(2)
        tensor, keys = build_run_channel_tensor(
            weather,
            baseline,
            capacity_kwh=21600.0,
            weather_columns=WEATHER_COLUMNS,
        )
        self.assertEqual(tensor.shape, (2, 6, 24))
        np.testing.assert_allclose(tensor[0, 0], baseline.iloc[:24] / 21600.0)
        np.testing.assert_allclose(tensor[0, 1], weather.iloc[:24, 0])
        self.assertEqual(len(keys), 2)

    def test_fold_scaler_is_not_refit_on_transform(self) -> None:
        weather, baseline, _ = fixtures(4)
        transformer = RunDCTTransformer(WEATHER_COLUMNS, coefficients=8)
        fit_x, _ = transformer.fit_transform(
            weather.iloc[:48], baseline.iloc[:48], capacity_kwh=21600.0
        )
        mean = transformer.channel_mean_.copy()
        scale = transformer.channel_scale_.copy()
        valid_x, _ = transformer.transform(
            weather.iloc[48:], baseline.iloc[48:], capacity_kwh=21600.0
        )
        self.assertEqual(fit_x.shape, (2, 48))
        self.assertEqual(valid_x.shape, (2, 48))
        np.testing.assert_array_equal(transformer.channel_mean_, mean)
        np.testing.assert_array_equal(transformer.channel_scale_, scale)

    def test_joint_target_and_blend_shapes(self) -> None:
        _, baseline, actual = fixtures(2)
        target, keys = build_joint_cf_target(actual, capacity_kwh=21600.0)
        self.assertEqual(target.shape, (2, 24))
        self.assertEqual(len(keys), 2)
        candidate = blend_joint_prediction(
            baseline,
            target,
            capacity_kwh=21600.0,
            blend_weight=0.1,
        )
        expected = 0.9 * baseline.to_numpy() + 0.1 * actual.to_numpy()
        np.testing.assert_allclose(candidate.to_numpy(), expected)

    def test_seed_ensemble_is_deterministic_and_joint(self) -> None:
        x = np.arange(8 * 6, dtype=float).reshape(8, 6) / 20.0
        y = np.tile(np.linspace(0.1, 0.8, 24), (8, 1)) + x[:, :1] * 0.01
        spec = {
            "hidden_layer_sizes": [4],
            "activation": "tanh",
            "solver": "lbfgs",
            "alpha": 0.1,
            "seeds": [42, 2026],
            "max_iter": 100,
            "max_fun": 500,
            "tol": 1e-6,
            "shuffle": False,
            "early_stopping": False,
        }
        first = predict_seed_ensemble(fit_seed_ensemble(spec, x, y), x)
        second = predict_seed_ensemble(fit_seed_ensemble(spec, x, y), x)
        self.assertEqual(first.shape, (8, 24))
        np.testing.assert_array_equal(first, second)

    def test_group_selection_is_independent_and_strict(self) -> None:
        comparisons = {
            "a": {
                "g1": {"full": {"delta": 0.2}, "H2": {"delta": 0.1}},
                "g2": {"full": {"delta": -0.1}, "H2": {"delta": 0.2}},
            },
            "b": {
                "g1": {"full": {"delta": 0.3}, "H2": {"delta": 0.01}},
                "g2": {"full": {"delta": 0.05}, "H2": {"delta": 0.04}},
            },
        }
        self.assertEqual(
            choose_group_candidates(comparisons, ["g1", "g2"]),
            {"g1": "a", "g2": "b"},
        )

    def test_rejects_incomplete_run(self) -> None:
        weather, baseline, _ = fixtures(2)
        with self.assertRaises(ValueError):
            build_run_channel_tensor(
                weather.iloc[:-1],
                baseline.iloc[:-1],
                capacity_kwh=21600.0,
                weather_columns=WEATHER_COLUMNS,
            )


if __name__ == "__main__":
    unittest.main()

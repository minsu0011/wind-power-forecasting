import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.final_training import FinalDataBundle, train_candidate
from src.metric import CAPACITY_KWH, TARGET_COLS


class _RecordingRegressor:
    fits = []

    def __init__(self, *args, **kwargs):
        pass

    def fit(self, x, y, sample_weight=None):
        self.__class__.fits.append(
            (
                np.asarray(y, dtype=float).copy(),
                None
                if sample_weight is None
                else np.asarray(sample_weight, dtype=float).copy(),
            )
        )
        return self

    def predict(self, x):
        return np.full(len(x), 0.5, dtype=float)


class CapacityFactorTrainingTests(unittest.TestCase):
    def setUp(self):
        _RecordingRegressor.fits.clear()
        train_index = pd.date_range("2023-01-01 01:00", periods=120, freq="h")
        test_index = pd.date_range("2024-01-01 01:00", periods=3, freq="h")
        labels = pd.DataFrame(
            {
                group: np.full(len(train_index), CAPACITY_KWH[group] * 0.5)
                for group in TARGET_COLS
            },
            index=train_index,
        )
        train_features = {
            group: pd.DataFrame(
                {"weather": np.linspace(0.0, 1.0, len(train_index))},
                index=train_index,
            )
            for group in TARGET_COLS
        }
        test_features = {
            group: pd.DataFrame(
                {"weather": np.linspace(0.1, 0.3, len(test_index))},
                index=test_index,
            )
            for group in TARGET_COLS
        }
        self.bundle = FinalDataBundle(
            labels=labels,
            sample_submission=pd.DataFrame(index=np.arange(len(test_index))),
            sample_times=test_index,
            train_features=train_features,
            test_features=test_features,
            input_files=[],
        )

    def test_shared_capacity_factor_is_scaled_back_per_group(self):
        specification = {
            "scope": "shared",
            "target_scale": "capacity_factor",
            "objective": "quantile",
            "alpha": 0.7,
            "row_filter": "eligible",
            "features": "all",
            "sample_weight": {"kind": "energy_linear", "base": 0.5, "strength": 1.0},
            "params": {"n_estimators": 2},
        }
        recipe = {"seed": 42, "n_jobs": 1, "device": "cpu"}
        with patch("src.final_training.TabularRegressor", _RecordingRegressor):
            result = train_candidate(
                "shared_test",
                specification,
                recipe=recipe,
                data=self.bundle,
                trained={},
            )

        self.assertEqual(len(_RecordingRegressor.fits), 1)
        fitted_y, fitted_weight = _RecordingRegressor.fits[0]
        np.testing.assert_allclose(fitted_y, 0.5)
        np.testing.assert_allclose(fitted_weight, 1.0)
        for group in TARGET_COLS:
            np.testing.assert_allclose(
                result.predictions[group], CAPACITY_KWH[group] * 0.5
            )


if __name__ == "__main__":
    unittest.main()

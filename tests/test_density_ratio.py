from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.density_ratio import (
    assemble_locked_group,
    estimate_density_ratio_weights,
    half_delta_candidate,
    run_fold_ids,
)


FEATURES = [f"x{i}" for i in range(15)]


class DensityRatioTests(unittest.TestCase):
    def test_run_folds_keep_each_24_hour_run_intact(self) -> None:
        index = pd.date_range("2022-01-01 01:00", periods=24 * 12, freq="h")
        folds = run_fold_ids(index)
        keys = (index - pd.Timedelta(hours=1)).normalize()
        for key in keys.unique():
            self.assertEqual(len(np.unique(folds[keys == key])), 1)

    def test_density_weights_are_deterministic_normalized_and_finite(self) -> None:
        source_index = pd.date_range("2022-01-01 01:00", periods=24 * 20, freq="h")
        application_index = pd.date_range(
            "2023-01-01 01:00", periods=24 * 20, freq="h"
        )
        rng = np.random.default_rng(7)
        source = pd.DataFrame(rng.normal(size=(len(source_index), 15)), index=source_index, columns=FEATURES)
        application = pd.DataFrame(
            rng.normal(loc=0.2, size=(len(application_index), 15)),
            index=application_index,
            columns=FEATURES,
        )
        eligible = pd.Series(np.arange(len(source)) % 3 != 0, index=source.index)
        first = estimate_density_ratio_weights(
            source, application, eligible_source=eligible, feature_names=FEATURES
        )
        second = estimate_density_ratio_weights(
            source, application, eligible_source=eligible, feature_names=FEATURES
        )
        np.testing.assert_array_equal(first.source_weight, second.source_weight)
        self.assertAlmostEqual(float(first.source_weight[eligible].mean()), 1.0)
        self.assertTrue(np.isfinite(first.source_weight).all())
        self.assertEqual(first.audit["run_fold_overlap"], 0)
        self.assertGreater(first.audit["eligible_weight_ess"], 0.0)

    def test_locked_assembly_order_and_identity_delta(self) -> None:
        ensemble = {
            "weights": {"g": {"a": 0.25, "b": 0.75}},
            "affine": {"g": {"scale": 1.2, "bias_kwh": -5.0}},
            "clip": {
                "g": {
                    "lower_capacity_fraction": 0.0,
                    "upper_capacity_fraction": 1.02,
                }
            },
            "power_bins": {
                "g": {
                    "edges_cf": [0.0, 0.5, 1.02],
                    "delta_kwh": [-1.0, 2.0],
                }
            },
        }
        assembled = assemble_locked_group(
            {"a": np.array([10.0, 80.0]), "b": np.array([30.0, 60.0])},
            group="g",
            capacity_kwh=100.0,
            ensemble=ensemble,
        )
        expected_pre_bin = np.clip(1.2 * np.array([25.0, 65.0]) - 5.0, 0.0, 102.0)
        expected = np.array([expected_pre_bin[0] - 1.0, expected_pre_bin[1] + 2.0])
        np.testing.assert_array_equal(assembled, expected)
        np.testing.assert_array_equal(
            half_delta_candidate(assembled, assembled, capacity_kwh=100.0), assembled
        )


if __name__ == "__main__":
    unittest.main()

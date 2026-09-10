from __future__ import annotations

import pickle
import unittest

import numpy as np
import pandas as pd

from src.seasonal_wind_ridge import (
    FEATURE_COLUMNS,
    STAGE1_REQUIRED,
    STAGE2_REQUIRED,
    SeasonalWindRidge,
    assert_strict_forward,
    blend_direct_kwh,
    build_seasonal_wind_features,
    select_stage1_weight,
    stage2_promoted,
)


class SeasonalWindFeatureTests(unittest.TestCase):
    def test_exact_feature_schema_and_formulas(self) -> None:
        index = pd.date_range("2022-01-01 01:00:00", periods=48, freq="h")
        wind = np.linspace(-2.0, 32.0, len(index))
        features = build_seasonal_wind_features(index, wind)
        self.assertEqual(tuple(features.columns), FEATURE_COLUMNS)
        self.assertEqual(features.shape, (48, 23))
        self.assertTrue(np.isfinite(features.to_numpy()).all())
        np.testing.assert_array_equal(
            features["wind_z"].to_numpy(), np.clip(wind, 0.0, 30.0) / 15.0
        )
        np.testing.assert_allclose(
            features["wind_hinge2_09"].to_numpy(),
            np.square(np.maximum(features["wind_z"].to_numpy() - 9.0 / 15.0, 0.0)),
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            features["wind_x_doy_sin1"].to_numpy(),
            features["wind_z"].to_numpy() * features["doy_sin1"].to_numpy(),
            rtol=0.0,
            atol=0.0,
        )

    def test_feature_builder_rejects_nonfinite_or_changed_knots(self) -> None:
        index = pd.date_range("2022-01-01 01:00:00", periods=2, freq="h")
        with self.assertRaises(ValueError):
            build_seasonal_wind_features(index, [1.0, np.nan])
        with self.assertRaises(ValueError):
            build_seasonal_wind_features(index, [1.0, 2.0], knots_ms=(3.0, 6.0))


class SeasonalWindModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = pd.date_range("2022-01-01 01:00:00", periods=240, freq="h")
        self.wind = 7.0 + 3.0 * np.sin(np.arange(240) / 12.0)
        self.features = build_seasonal_wind_features(self.index, self.wind)
        cf = np.clip(0.12 + 0.04 * self.wind, 0.1, 0.9)
        self.actual = pd.Series(cf * 21_600.0, index=self.index)

    def test_fit_predict_and_pickle_reload_are_bit_exact(self) -> None:
        model = SeasonalWindRidge().fit(
            self.features, self.actual, capacity_kwh=21_600.0
        )
        before = model.predict_cf(self.features)
        after = pickle.loads(pickle.dumps(model)).predict_cf(self.features)
        np.testing.assert_array_equal(before.to_numpy(), after.to_numpy())
        self.assertEqual(model.fit_metadata_["feature_count"], 23)
        self.assertEqual(model.fit_metadata_["target_kind"], "direct_capacity_factor")

    def test_model_rejects_schema_and_parameter_changes(self) -> None:
        with self.assertRaises(ValueError):
            SeasonalWindRidge(alpha=1.0)
        model = SeasonalWindRidge()
        with self.assertRaises(ValueError):
            model.fit(
                self.features.rename(columns={"wind_z": "actual"}),
                self.actual,
                capacity_kwh=21_600.0,
            )

    def test_strict_forward_and_blend_identity(self) -> None:
        fit = self.index[:120]
        apply = self.index[120:]
        assert_strict_forward(fit, apply)
        with self.assertRaises(ValueError):
            assert_strict_forward(self.index[:121], self.index[120:])
        baseline = pd.Series(np.linspace(1000.0, 20_000.0, len(apply)), index=apply)
        direct = pd.Series(np.linspace(0.1, 0.9, len(apply)), index=apply)
        zero = blend_direct_kwh(
            baseline, direct, weight=0.0, capacity_kwh=21_600.0
        )
        self.assertEqual(baseline.to_numpy().tobytes(), zero.to_numpy().tobytes())
        with self.assertRaises(ValueError):
            blend_direct_kwh(
                baseline, direct, weight=0.10, capacity_kwh=21_600.0
            )


def _group_record(delta: float) -> dict[str, object]:
    candidate = 0.5 + delta
    return {
        "baseline": {"score": 0.5},
        "candidate": {"score": candidate},
        "delta": candidate - 0.5,
    }


def _mixed_record(delta: float) -> dict[str, object]:
    candidate = 0.5 + delta
    return {
        "baseline": {"total_score": 0.5},
        "candidate": {"total_score": candidate},
        "delta_total_score": candidate - 0.5,
    }


class SeasonalWindGateTests(unittest.TestCase):
    def test_stage1_requires_all_17_and_tie_prefers_smaller_weight(self) -> None:
        weights = {"w025": 0.025, "w05": 0.05}
        comparisons = {
            key: {
                group: {segment: _group_record(0.001) for segment in segments}
                for group, segments in STAGE1_REQUIRED.items()
            }
            for key in weights
        }
        selected, audit = select_stage1_weight(comparisons, weights=weights)
        self.assertEqual(selected, 0.025)
        self.assertEqual(audit["selected"], "w025")
        comparisons["w025"]["kpx_group_3"]["Q4"] = _group_record(-1e-8)
        comparisons["w05"]["kpx_group_2"]["Q2"] = _group_record(0.0)
        selected, audit = select_stage1_weight(comparisons, weights=weights)
        self.assertIsNone(selected)
        self.assertEqual(audit["selected"], "identity")

    def test_stage2_requires_21_group_and_7_mixed_positive(self) -> None:
        groups = {
            group: {segment: _group_record(0.001) for segment in STAGE2_REQUIRED}
            for group in ("kpx_group_1", "kpx_group_2", "kpx_group_3")
        }
        mixed = {segment: _mixed_record(0.001) for segment in STAGE2_REQUIRED}
        promoted, audit = stage2_promoted(groups, mixed)
        self.assertTrue(promoted)
        self.assertEqual(audit["group_slice_count"], 21)
        self.assertEqual(audit["mixed_slice_count"], 7)
        mixed["Q4"] = _mixed_record(0.0)
        promoted, audit = stage2_promoted(groups, mixed)
        self.assertFalse(promoted)
        self.assertFalse(audit["all_7_mixed_strictly_positive"])


if __name__ == "__main__":
    unittest.main()

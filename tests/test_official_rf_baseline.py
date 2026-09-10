from __future__ import annotations

import pickle
import unittest

import numpy as np
import pandas as pd

from src.features import AVAILABLE_COL, COORD_COLS, GRID_COL, SOURCE_VARIABLES, TIME_COL
from src.official_rf_baseline import (
    CALENDAR_COLUMNS,
    EXPLICIT_RF_PARAMETERS,
    FEATURE_COLUMNS,
    RELEVANT_DEFAULT_RF_PARAMETERS,
    STAGE1_REQUIRED,
    STAGE2_REQUIRED,
    OfficialRandomForestBaseline,
    assert_strict_forward,
    blend_direct_kwh,
    build_official_rf_features,
    official_calendar_features,
    select_stage1_weight,
    stage2_promoted,
)


def _raw_weather(
    source: str, index: pd.DatetimeIndex, rows_per_time: int = 3
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for time_number, timestamp in enumerate(index):
        for grid in range(rows_per_time):
            record: dict[str, object] = {
                TIME_COL: timestamp,
                AVAILABLE_COL: timestamp - pd.Timedelta(hours=12),
                GRID_COL: grid + 1,
                COORD_COLS[0]: 37.0 + grid / 10.0,
                COORD_COLS[1]: 128.0 + grid / 10.0,
            }
            for variable_number, variable in enumerate(SOURCE_VARIABLES[source]):
                record[variable] = (
                    100.0 * time_number + 10.0 * grid + variable_number / 100.0
                )
            records.append(record)
    return pd.DataFrame.from_records(records)


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


class OfficialRFFeatureTests(unittest.TestCase):
    def test_calendar_formulas_match_notebook_including_month_phase(self) -> None:
        index = pd.DatetimeIndex(
            ["2023-01-07 00:00:00", "2023-12-11 06:00:00"],
            name="forecast_kst_dtm",
        )
        frame = official_calendar_features(index)
        self.assertEqual(tuple(frame.columns), CALENDAR_COLUMNS)
        self.assertEqual(frame.loc[index[0], "is_weekend"], 1.0)
        self.assertEqual(frame.loc[index[1], "is_weekend"], 0.0)
        np.testing.assert_allclose(
            frame["month_sin"].to_numpy(),
            np.sin(2.0 * np.pi * np.array([1.0, 12.0]) / 12.0),
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            frame["hour_cos"].to_numpy(),
            np.cos(2.0 * np.pi * np.array([0.0, 6.0]) / 24.0),
            rtol=0.0,
            atol=0.0,
        )

    def test_exact_74_columns_and_all_grid_arithmetic_means(self) -> None:
        index = pd.date_range("2022-01-01 01:00:00", periods=4, freq="h")
        ldaps = _raw_weather("ldaps", index)
        gfs = _raw_weather("gfs", index)
        features = build_official_rf_features(ldaps, gfs)
        self.assertEqual(tuple(features.columns), FEATURE_COLUMNS)
        self.assertEqual(features.shape, (4, 74))
        first = SOURCE_VARIABLES["ldaps"][0]
        expected = ldaps.groupby(TIME_COL)[first].mean().to_numpy()
        np.testing.assert_array_equal(
            features[f"ldaps_{first}_mean"].to_numpy(), expected
        )
        self.assertTrue(np.isfinite(features.to_numpy()).all())

    def test_builder_allows_nan_for_training_median_but_rejects_inf(self) -> None:
        index = pd.date_range("2022-01-01 01:00:00", periods=3, freq="h")
        ldaps = _raw_weather("ldaps", index)
        gfs = _raw_weather("gfs", index)
        ldaps.loc[0, SOURCE_VARIABLES["ldaps"][0]] = np.nan
        features = build_official_rf_features(ldaps, gfs)
        self.assertTrue(np.isfinite(features.to_numpy()).all())
        ldaps.loc[0, SOURCE_VARIABLES["ldaps"][0]] = np.inf
        with self.assertRaises(ValueError):
            build_official_rf_features(ldaps, gfs)


class OfficialRFModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = pd.date_range("2022-01-01 01:00:00", periods=160, freq="h")
        ldaps = _raw_weather("ldaps", self.index, rows_per_time=2)
        gfs = _raw_weather("gfs", self.index, rows_per_time=2)
        self.features = build_official_rf_features(ldaps, gfs)
        self.features.iloc[0, 10] = np.nan
        phase = np.arange(len(self.index), dtype=np.float64)
        cf = 0.25 + 0.12 * np.sin(phase / 11.0)
        cf[:7] = 0.05
        self.actual = pd.Series(cf * 21_600.0, index=self.index)

    def test_fit_imputer_parameters_and_pickle_reload_are_exact(self) -> None:
        model = OfficialRandomForestBaseline().fit(
            self.features, self.actual, capacity_kwh=21_600.0
        )
        before = model.predict_cf(self.features)
        after = pickle.loads(pickle.dumps(model)).predict_cf(self.features)
        np.testing.assert_array_equal(before.to_numpy(), after.to_numpy())
        self.assertEqual(model.fit_metadata_["feature_count"], 74)
        self.assertEqual(model.fit_metadata_["imputer_fit_rows"], 160)
        self.assertEqual(model.fit_metadata_["explicit_parameters"], EXPLICIT_RF_PARAMETERS)
        self.assertEqual(
            model.fit_metadata_["relevant_default_parameters"],
            RELEVANT_DEFAULT_RF_PARAMETERS,
        )
        self.assertEqual(model.estimator_.n_jobs, -1)
        self.assertGreaterEqual(float(before.min()), 0.0)
        self.assertLessEqual(float(before.max()), 1.0)

    def test_rejects_parameter_schema_alignment_and_all_missing_changes(self) -> None:
        with self.assertRaises(ValueError):
            OfficialRandomForestBaseline(n_estimators=121)
        with self.assertRaises(ValueError):
            OfficialRandomForestBaseline(max_features=1.0)
        model = OfficialRandomForestBaseline()
        with self.assertRaises(ValueError):
            model.fit(
                self.features.loc[:, list(reversed(FEATURE_COLUMNS))],
                self.actual,
                capacity_kwh=21_600.0,
            )
        all_missing = self.features.copy()
        all_missing.iloc[:, 20] = np.nan
        with self.assertRaises(ValueError):
            model.fit(all_missing, self.actual, capacity_kwh=21_600.0)

    def test_strict_forward_and_only_registered_blends(self) -> None:
        fit = self.index[:80]
        application = self.index[80:]
        assert_strict_forward(fit, application)
        with self.assertRaises(ValueError):
            assert_strict_forward(self.index[:81], self.index[80:])
        baseline = pd.Series(np.linspace(1000.0, 20_000.0, len(application)), index=application)
        direct = pd.Series(np.linspace(0.1, 0.9, len(application)), index=application)
        zero = blend_direct_kwh(baseline, direct, weight=0.0, capacity_kwh=21_600.0)
        self.assertEqual(zero.to_numpy().tobytes(), baseline.to_numpy().tobytes())
        for weight in (0.05, 0.10, 1.0):
            result = blend_direct_kwh(
                baseline, direct, weight=weight, capacity_kwh=21_600.0
            )
            self.assertTrue(np.isfinite(result.to_numpy()).all())
        with self.assertRaises(ValueError):
            blend_direct_kwh(baseline, direct, weight=0.075, capacity_kwh=21_600.0)


class OfficialRFGateTests(unittest.TestCase):
    def test_stage1_requires_all_17_and_smaller_weight_wins_exact_tie(self) -> None:
        weights = {"w05": 0.05, "w10": 0.10}
        comparisons = {
            key: {
                group: {segment: _group_record(0.001) for segment in segments}
                for group, segments in STAGE1_REQUIRED.items()
            }
            for key in weights
        }
        selected, audit = select_stage1_weight(comparisons, weights=weights)
        self.assertEqual(selected, 0.05)
        self.assertEqual(audit["selected"], "w05")
        comparisons["w05"]["kpx_group_1"]["Q2"] = _group_record(0.0)
        comparisons["w10"]["kpx_group_3"]["Q4"] = _group_record(-1e-8)
        selected, audit = select_stage1_weight(comparisons, weights=weights)
        self.assertIsNone(selected)
        self.assertEqual(audit["selected"], "identity")

    def test_stage2_requires_all_21_group_and_7_mixed(self) -> None:
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

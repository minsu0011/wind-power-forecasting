from __future__ import annotations

import json
from pathlib import Path
import pickle
import unittest

import numpy as np
import pandas as pd

from scripts import run_compact_physics_extratrees as runner
from src.compact_physics_extratrees import (
    FEATURE_COLUMNS,
    STAGE1_REQUIRED,
    STAGE2_REQUIRED,
    CompactPhysicsExtraTrees,
    assert_strict_forward,
    blend_direct_kwh,
    build_compact_physics_features,
    select_stage1_weight,
    stage2_promoted,
)
from src.manifest import sha256_file


PROJECT_DIR = Path(__file__).resolve().parents[1]
PREREGISTER = (
    PROJECT_DIR / "configs/compact_physics_extratrees_mse_preregister_v2.json"
)


def _weather(index: pd.DatetimeIndex) -> pd.DataFrame:
    rows = np.arange(len(index), dtype=np.float64)[:, None]
    columns = np.arange(len(FEATURE_COLUMNS), dtype=np.float64)[None, :]
    values = 3.0 + 0.01 * rows + 0.001 * columns
    return pd.DataFrame(values, index=index, columns=FEATURE_COLUMNS)


class CompactPhysicsFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = pd.date_range("2022-01-01 01:00:00", periods=96, freq="h")

    def test_exact_immutable_feature_schema(self) -> None:
        weather = _weather(self.index)
        weather["unused_weather_channel"] = 1.0
        features = build_compact_physics_features(weather)
        self.assertEqual(tuple(features.columns), FEATURE_COLUMNS)
        self.assertEqual(features.shape, (96, 36))
        self.assertTrue(np.isfinite(features.to_numpy()).all())
        np.testing.assert_array_equal(
            features.to_numpy(), weather.loc[:, list(FEATURE_COLUMNS)].to_numpy()
        )

    def test_rejects_missing_nonfinite_or_bad_index(self) -> None:
        weather = _weather(self.index)
        with self.assertRaises(KeyError):
            build_compact_physics_features(weather.drop(columns=FEATURE_COLUMNS[0]))
        damaged = weather.copy()
        damaged.iloc[0, 0] = np.nan
        with self.assertRaises(ValueError):
            build_compact_physics_features(damaged)
        with self.assertRaises(TypeError):
            build_compact_physics_features(weather.reset_index(drop=True))
        with self.assertRaises(ValueError):
            build_compact_physics_features(weather.iloc[::-1])


class CompactPhysicsModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = pd.date_range("2022-01-01 01:00:00", periods=120, freq="h")
        self.features = build_compact_physics_features(_weather(self.index))
        phase = np.arange(len(self.index), dtype=np.float64)
        cf = 0.25 + 0.18 * np.sin(phase / 13.0)
        cf[:9] = 0.05
        self.actual = pd.Series(cf * 21_600.0, index=self.index)

    def test_fixed_fit_predict_pickle_and_eligibility(self) -> None:
        model = CompactPhysicsExtraTrees().fit(
            self.features, self.actual, capacity_kwh=21_600.0
        )
        before = model.predict_cf(self.features)
        after = pickle.loads(pickle.dumps(model)).predict_cf(self.features)
        np.testing.assert_array_equal(before.to_numpy(), after.to_numpy())
        expected_eligible = int(
            np.sum(self.actual.to_numpy(dtype=np.float64) / 21_600.0 >= 0.10)
        )
        self.assertEqual(model.fit_metadata_["rows_eligible"], expected_eligible)
        self.assertEqual(model.fit_metadata_["feature_count"], 36)
        self.assertEqual(
            model.fit_metadata_["resolved_parameters"], runner.EXPECTED_MODEL_PARAMETERS
        )
        self.assertGreaterEqual(float(before.min()), 0.0)
        self.assertLessEqual(float(before.max()), 1.0)

    def test_model_rejects_any_parameter_schema_or_alignment_change(self) -> None:
        with self.assertRaises(ValueError):
            CompactPhysicsExtraTrees(n_estimators=399)
        with self.assertRaises(ValueError):
            CompactPhysicsExtraTrees(criterion="absolute_error")
        model = CompactPhysicsExtraTrees()
        with self.assertRaises(ValueError):
            model.fit(
                self.features.loc[:, list(reversed(FEATURE_COLUMNS))],
                self.actual,
                capacity_kwh=21_600.0,
            )
        with self.assertRaises(ValueError):
            model.fit(
                self.features,
                self.actual.rename(index={self.index[0]: self.index[-1] + pd.Timedelta("1h")}),
                capacity_kwh=21_600.0,
            )

    def test_strict_forward_and_registered_blends(self) -> None:
        fit_index = self.index[:60]
        application_index = self.index[60:]
        assert_strict_forward(fit_index, application_index)
        with self.assertRaises(ValueError):
            assert_strict_forward(self.index[:61], self.index[60:])
        baseline = pd.Series(
            np.linspace(1000.0, 20_000.0, len(application_index)),
            index=application_index,
        )
        direct = pd.Series(
            np.linspace(0.1, 0.9, len(application_index)), index=application_index
        )
        zero = blend_direct_kwh(
            baseline, direct, weight=0.0, capacity_kwh=21_600.0
        )
        self.assertEqual(baseline.to_numpy().tobytes(), zero.to_numpy().tobytes())
        for weight in (0.05, 0.10, 1.0):
            observed = blend_direct_kwh(
                baseline, direct, weight=weight, capacity_kwh=21_600.0
            )
            expected = np.clip(
                (1.0 - weight) * baseline.to_numpy()
                + weight * direct.to_numpy() * 21_600.0,
                0.0,
                21_600.0,
            )
            np.testing.assert_array_equal(observed.to_numpy(), expected)
        with self.assertRaises(ValueError):
            blend_direct_kwh(
                baseline, direct, weight=0.075, capacity_kwh=21_600.0
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


class CompactPhysicsGateAndPreregisterTests(unittest.TestCase):
    def test_stage1_is_exactly_17_slices_and_tie_prefers_005(self) -> None:
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
        comparisons["w05"]["kpx_group_3"]["Q4"] = _group_record(-1e-8)
        comparisons["w10"]["kpx_group_2"]["Q2"] = _group_record(0.0)
        selected, audit = select_stage1_weight(comparisons, weights=weights)
        self.assertIsNone(selected)
        self.assertEqual(audit["selected"], "identity")

    def test_stage2_requires_all_21_group_and_7_mixed_slices(self) -> None:
        groups = {
            group: {segment: _group_record(0.001) for segment in STAGE2_REQUIRED}
            for group in ("kpx_group_1", "kpx_group_2", "kpx_group_3")
        }
        mixed = {segment: _mixed_record(0.001) for segment in STAGE2_REQUIRED}
        promoted, audit = stage2_promoted(groups, mixed)
        self.assertTrue(promoted)
        self.assertEqual(audit["group_slice_count"], 21)
        self.assertEqual(audit["mixed_slice_count"], 7)
        groups["kpx_group_1"]["H2"] = _group_record(0.0)
        promoted, audit = stage2_promoted(groups, mixed)
        self.assertFalse(promoted)
        self.assertFalse(audit["all_21_group_strictly_positive"])

    def test_preregister_hash_model_features_and_diagnostic_role_are_exact(self) -> None:
        self.assertEqual(sha256_file(PREREGISTER), runner.PREREGISTER_SHA256)
        payload = runner._verify_preregister(PREREGISTER)
        self.assertEqual(tuple(payload["feature_contract"]["columns_in_exact_order"]), FEATURE_COLUMNS)
        self.assertEqual(payload["model"]["parameters"], runner.EXPECTED_MODEL_PARAMETERS)
        self.assertEqual(
            payload["candidate_family"]["fixed_global_blend_weights"], [0.05, 0.10]
        )
        self.assertIn("never eligible", payload["candidate_family"]["standalone_role"])
        self.assertFalse(payload["public_contract"]["public_metric_triplets_used"])

    def test_runner_defaults_are_frozen_without_running_protocol(self) -> None:
        args = runner._with_defaults(["--stage", "stage1"])
        self.assertEqual(args.count("--out-dir"), 1)
        self.assertEqual(args.count("--preregister"), 1)
        self.assertEqual(runner.WEIGHTS, {"w05": 0.05, "w10": 0.10})
        self.assertEqual(len(FEATURE_COLUMNS), 36)
        payload = runner._verify_preregister(PREREGISTER)
        self.assertEqual(payload["stage1"]["registered_slice_count"], 17)


if __name__ == "__main__":
    unittest.main()

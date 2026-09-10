from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd

from src.raw_grid_wind import TIME_FEATURES
from src.raw_temporal_kernel import (
    CANDIDATE_KEYS,
    ESTIMATOR_ID,
    RawRunPCAKernelTransformer,
    RawTemporalKernelRegressor,
    assert_fit_before_apply,
    blend_kernel_prediction,
    candidate_frame,
    parse_candidate_key,
    registered_raw_node_columns,
    select_group_candidate,
)


def _features(start: str, runs: int, seed: int = 42) -> pd.DataFrame:
    index = pd.date_range(start, periods=runs * 24, freq="h", name="forecast_kst_dtm")
    ldaps = [f"ldaps__grid_{grid:02d}__c{channel:02d}" for channel in range(8) for grid in range(1, 17)]
    gfs = [f"gfs__grid_{grid:02d}__c{channel:02d}" for channel in range(8) for grid in range(1, 10)]
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(len(index), 200)).astype(np.float32)
    clock = rng.normal(size=(len(index), len(TIME_FEATURES))).astype(np.float32)
    return pd.DataFrame(
        np.column_stack((raw, clock)), index=index, columns=[*ldaps, *gfs, *TIME_FEATURES]
    )


def _comparison(delta: float) -> dict[str, object]:
    candidate = 0.5 + delta
    return {
        "baseline": {"score": 0.5},
        "candidate": {"score": candidate},
        "delta": candidate - 0.5,
    }


class RawTemporalKernelTests(unittest.TestCase):
    def test_registered_columns_are_raw_200_only(self) -> None:
        features = _features("2022-01-01 01:00:00", 1)
        raw = registered_raw_node_columns(features)
        self.assertEqual(len(raw), 200)
        self.assertFalse(set(raw).intersection(TIME_FEATURES))
        broken = features.copy()
        columns = list(broken.columns)
        columns[-1], columns[-2] = columns[-2], columns[-1]
        broken.columns = columns
        with self.assertRaises(ValueError):
            registered_raw_node_columns(broken)

    def test_flatten_order_is_channel_then_horizon(self) -> None:
        values = np.arange(24 * 200, dtype=float).reshape(24, 200)
        flat = RawRunPCAKernelTransformer._flatten(values)
        self.assertEqual(flat.shape, (1, 4800))
        np.testing.assert_array_equal(flat[0, :24], values[:, 0])
        np.testing.assert_array_equal(flat[0, 24:48], values[:, 1])

    def test_transformer_is_fit_only_and_outputs_pca32(self) -> None:
        fit = _features("2022-01-01 01:00:00", 40)
        raw = registered_raw_node_columns(fit)
        fit.iloc[0, 0] = np.nan
        transformer = RawRunPCAKernelTransformer(raw).fit(fit)
        medians = transformer.medians_.copy()
        components = transformer.pca_.components_.copy()
        app = _features("2022-02-10 01:00:00", 2, seed=9)
        app.iloc[:, 0] = 1e8
        matrix, keys = transformer.transform(app)
        self.assertEqual(matrix.shape, (2, 32))
        self.assertEqual(len(keys), 2)
        np.testing.assert_array_equal(transformer.medians_, medians)
        np.testing.assert_array_equal(transformer.pca_.components_, components)

    def test_joint_kernel_model_parameters_and_prediction_shape(self) -> None:
        fit = _features("2022-01-01 01:00:00", 40)
        app = _features("2022-02-10 01:00:00", 2, seed=7)
        horizon = np.tile(np.arange(24), 40)
        target = pd.Series(
            0.4 + 0.1 * np.sin(2 * np.pi * horizon / 24), index=fit.index
        )
        model = RawTemporalKernelRegressor().fit(fit, target)
        prediction = model.predict(app)
        self.assertEqual(prediction.shape, (48,))
        self.assertTrue(np.isfinite(prediction).all())
        self.assertEqual(model.estimator_.alpha, 10.0)
        self.assertEqual(model.estimator_.gamma, 1.0 / 32.0)
        self.assertEqual(model.metadata()["transformer"]["flat_dimension"], 4800)
        self.assertFalse(
            model.metadata()["transformer"][
                "baseline_clock_issuance_target_scada_public_scale_features_used"
            ]
        )

    def test_incomplete_targets_drop_whole_run_with_keys(self) -> None:
        fit = _features("2022-01-01 01:00:00", 40)
        target = pd.Series(0.4, index=fit.index)
        target.iloc[-1] = np.nan
        model = RawTemporalKernelRegressor().fit(fit, target)
        metadata = model.metadata()
        self.assertEqual(metadata["requested_fit_runs"], 40)
        self.assertEqual(metadata["fit_runs"], 39)
        self.assertEqual(metadata["dropped_incomplete_target_runs"], 1)
        self.assertEqual(
            metadata["dropped_incomplete_target_run_keys"],
            [pd.Timestamp("2022-02-09").isoformat()],
        )

    def test_all_incomplete_target_runs_are_rejected(self) -> None:
        fit = _features("2022-01-01 01:00:00", 40)
        target = pd.Series(np.nan, index=fit.index)
        with self.assertRaises(ValueError):
            RawTemporalKernelRegressor().fit(fit, target)

    def test_incomplete_application_run_is_rejected(self) -> None:
        fit = _features("2022-01-01 01:00:00", 40)
        target = pd.Series(0.4, index=fit.index)
        model = RawTemporalKernelRegressor().fit(fit, target)
        incomplete = _features("2022-02-10 01:00:00", 1).iloc[:-1]
        with self.assertRaises(ValueError):
            model.predict(incomplete)

    def test_strict_fit_before_apply(self) -> None:
        fit = _features("2022-01-01 01:00:00", 2).index
        app = _features("2022-01-03 01:00:00", 1).index
        self.assertTrue(assert_fit_before_apply(fit, app)["fit_runs_complete"])
        with self.assertRaises(ValueError):
            assert_fit_before_apply(fit, fit[-24:])

    def test_blend_zero_identity_and_candidate_order(self) -> None:
        index = _features("2022-01-01 01:00:00", 1).index
        baseline = pd.Series(5000.0, index=index)
        model = pd.Series(np.linspace(-1, 2, 24), index=index)
        identity = blend_kernel_prediction(
            baseline, model, capacity_kwh=21600.0, weight=0.0
        )
        self.assertEqual(identity.to_numpy().tobytes(), baseline.to_numpy().tobytes())
        candidates = candidate_frame(baseline, model, capacity_kwh=21600.0)
        self.assertEqual(tuple(candidates.columns), CANDIDATE_KEYS)
        self.assertTrue(np.isfinite(candidates.to_numpy()).all())

    def test_candidate_parser_is_frozen(self) -> None:
        self.assertEqual(parse_candidate_key(f"{ESTIMATOR_ID}_w05"), 0.05)
        self.assertEqual(parse_candidate_key(f"{ESTIMATOR_ID}_w10"), 0.10)
        with self.assertRaises(ValueError):
            parse_candidate_key(f"{ESTIMATOR_ID}_w20")

    def test_selection_requires_all_slices_and_survives_sorted_json(self) -> None:
        slices = ("full", "H1")
        comparisons = {
            key: {
                name: _comparison(0.01 if key.endswith("w05") else -0.01)
                for name in slices
            }
            for key in CANDIDATE_KEYS
        }
        sorted_copy = json.loads(json.dumps(comparisons, sort_keys=True))
        selected, audit = select_group_candidate(sorted_copy, slices)
        self.assertEqual(selected, f"{ESTIMATOR_ID}_w05")
        self.assertEqual(tuple(audit["candidates"]), CANDIDATE_KEYS)
        comparisons[CANDIDATE_KEYS[0]]["H1"] = _comparison(-0.001)
        selected, _ = select_group_candidate(comparisons, slices)
        self.assertIsNone(selected)


if __name__ == "__main__":
    unittest.main()

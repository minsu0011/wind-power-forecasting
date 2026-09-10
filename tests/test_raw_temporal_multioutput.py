from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd

from src.raw_grid_wind import TIME_FEATURES
from src.raw_temporal_multioutput import (
    CANDIDATE_KEYS,
    ESTIMATOR_IDS,
    RawNodeDCTTransformer,
    RawTemporalMultiOutputRegressor,
    assert_fit_before_apply,
    blend_direct_prediction,
    candidate_frame,
    orthonormal_dct2_basis,
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


class RawTemporalMultiOutputTests(unittest.TestCase):
    def test_dct_basis_is_orthonormal(self) -> None:
        basis = orthonormal_dct2_basis()
        self.assertEqual(basis.shape, (4, 24))
        np.testing.assert_allclose(basis @ basis.T, np.eye(4), atol=1e-12, rtol=0)

    def test_registered_columns_are_exactly_raw_200(self) -> None:
        features = _features("2022-01-01 01:00:00", 1)
        columns = registered_raw_node_columns(features)
        self.assertEqual(len(columns), 200)
        self.assertFalse(set(columns).intersection(TIME_FEATURES))
        broken = features.copy()
        changed = list(broken.columns)
        changed[-1], changed[-2] = changed[-2], changed[-1]
        broken.columns = changed
        with self.assertRaises(ValueError):
            registered_raw_node_columns(broken)

    def test_transformer_has_run_by_800_shape_and_fit_only_statistics(self) -> None:
        fit = _features("2022-01-01 01:00:00", 3)
        raw = registered_raw_node_columns(fit)
        fit.iloc[0, 0] = np.nan
        transformer = RawNodeDCTTransformer(raw).fit(fit)
        before = transformer.medians_.copy()
        application = _features("2022-01-04 01:00:00", 2, seed=9)
        application.iloc[:, 0] = 1e9
        matrix, keys = transformer.transform(application)
        self.assertEqual(matrix.shape, (2, 800))
        self.assertEqual(len(keys), 2)
        np.testing.assert_array_equal(transformer.medians_, before)

    def test_incomplete_run_is_rejected(self) -> None:
        features = _features("2022-01-01 01:00:00", 2).iloc[:-1]
        raw = tuple(features.columns[:200])
        with self.assertRaises(ValueError):
            RawNodeDCTTransformer(raw).fit(features)

    def test_both_estimators_emit_joint_24_hour_predictions(self) -> None:
        fit = _features("2022-01-01 01:00:00", 16)
        app = _features("2022-01-17 01:00:00", 2, seed=7)
        hour = np.tile(np.arange(24), 16)
        target = pd.Series(
            np.clip(0.35 + 0.15 * np.sin(2 * np.pi * hour / 24), 0, 1.02),
            index=fit.index,
        )
        for estimator_id in ESTIMATOR_IDS:
            model = RawTemporalMultiOutputRegressor(estimator_id).fit(fit, target)
            prediction = model.predict(app)
            self.assertEqual(prediction.shape, (48,))
            self.assertTrue(np.isfinite(prediction).all())
            self.assertTrue(((prediction >= 0) & (prediction <= 1.02)).all())
            self.assertFalse(model.metadata()["transformer"]["clock_issuance_features_used"])

    def test_only_complete_finite_target_runs_are_retained_with_evidence(self) -> None:
        fit = _features("2022-01-01 01:00:00", 2)
        target = pd.Series(0.3, index=fit.index)
        target.iloc[-1] = np.nan
        model = RawTemporalMultiOutputRegressor("ridge_a100").fit(fit, target)
        metadata = model.metadata()
        self.assertEqual(metadata["requested_fit_runs"], 2)
        self.assertEqual(metadata["fit_runs"], 1)
        self.assertEqual(metadata["dropped_incomplete_target_runs"], 1)
        self.assertEqual(
            metadata["dropped_incomplete_target_run_keys"],
            [pd.Timestamp("2022-01-02").isoformat()],
        )
        target[:] = np.nan
        with self.assertRaises(ValueError):
            RawTemporalMultiOutputRegressor("ridge_a100").fit(fit, target)

    def test_model_predict_rejects_incomplete_application_run(self) -> None:
        fit = _features("2022-01-01 01:00:00", 2)
        target = pd.Series(0.3, index=fit.index)
        model = RawTemporalMultiOutputRegressor("ridge_a100").fit(fit, target)
        incomplete = _features("2022-01-03 01:00:00", 1).iloc[:-1]
        with self.assertRaises(ValueError):
            model.predict(incomplete)

    def test_strict_fit_before_apply_and_overlap_guard(self) -> None:
        fit = _features("2022-01-01 01:00:00", 2).index
        apply = _features("2022-01-03 01:00:00", 1).index
        audit = assert_fit_before_apply(fit, apply)
        self.assertTrue(audit["fit_end_before_application_start"])
        with self.assertRaises(ValueError):
            assert_fit_before_apply(fit, fit[-24:])

    def test_blend_zero_is_bit_exact_and_bounds_apply(self) -> None:
        index = _features("2022-01-01 01:00:00", 1).index
        baseline = pd.Series(np.arange(24, dtype=float), index=index)
        model = pd.Series(np.linspace(-2, 2, 24), index=index)
        identity = blend_direct_prediction(
            baseline, model, capacity_kwh=21600.0, weight=0.0
        )
        self.assertEqual(identity.to_numpy().tobytes(), baseline.to_numpy().tobytes())
        blended = blend_direct_prediction(
            baseline, model, capacity_kwh=21600.0, weight=0.10
        )
        self.assertTrue(((blended >= 0) & (blended <= 1.02 * 21600)).all())

    def test_candidate_frame_and_parser_use_frozen_order(self) -> None:
        index = _features("2022-01-01 01:00:00", 1).index
        baseline = pd.Series(5000.0, index=index)
        predictions = {
            name: pd.Series(0.4, index=index) for name in ESTIMATOR_IDS
        }
        frame = candidate_frame(baseline, predictions, capacity_kwh=21600.0)
        self.assertEqual(tuple(frame.columns), CANDIDATE_KEYS)
        self.assertEqual(parse_candidate_key("ridge_a100_w05"), ("ridge_a100", 0.05))
        with self.assertRaises(ValueError):
            parse_candidate_key("ridge_a100_w20")

    def test_selection_requires_every_slice_positive(self) -> None:
        slices = ("full", "H1")
        comparisons = {
            key: {name: _comparison(0.001) for name in slices}
            for key in CANDIDATE_KEYS
        }
        comparisons["ridge_a100_w05"]["H1"] = _comparison(0.002)
        selected, audit = select_group_candidate(comparisons, slices)
        self.assertEqual(selected, "ridge_a100_w05")
        self.assertEqual(audit["selected"], selected)
        for key in CANDIDATE_KEYS:
            comparisons[key]["H1"] = _comparison(-0.001)
        selected, _ = select_group_candidate(comparisons, slices)
        self.assertIsNone(selected)

    def test_json_sorted_selection_reconstructs_frozen_order(self) -> None:
        slices = ("full", "H1", "H2")
        comparisons = {
            key: {
                name: _comparison(0.01 if key == "ridge_a100_w05" else -0.01)
                for name in slices
            }
            for key in CANDIDATE_KEYS
        }
        roundtripped = json.loads(json.dumps(comparisons, sort_keys=True))
        self.assertNotEqual(tuple(roundtripped), CANDIDATE_KEYS)
        selected, audit = select_group_candidate(roundtripped, slices)
        self.assertEqual(selected, "ridge_a100_w05")
        self.assertEqual(tuple(audit["candidates"]), CANDIDATE_KEYS)


if __name__ == "__main__":
    unittest.main()

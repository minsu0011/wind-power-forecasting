from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from src.target_transform import (
    TargetTransformRegressor,
    assert_strict_fit_apply_order,
    blend_with_corrected_v3,
    forward_target_transform,
    inverse_target_transform,
)


PREREGISTER_SHA256 = (
    "2fb9420c2e43bee2e71f910874aa87525bb6afe8c1f4601bc5d9ee79369c201d"
)


def _features(start: str = "2022-01-01 01:00:00", rows: int = 96) -> pd.DataFrame:
    index = pd.date_range(start, periods=rows, freq="h")
    x = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "ldaps__idw__hub_ws": 6.0 + np.sin(x / 10.0),
            "gfs__idw__hub_ws": 7.0 + np.cos(x / 12.0),
        },
        index=index,
    )


def _params() -> dict[str, object]:
    return {
        "objective": "regression_l1",
        "n_estimators": 8,
        "learning_rate": 0.05,
        "num_leaves": 7,
        "min_child_samples": 5,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.05,
        "reg_lambda": 2.0,
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
        "random_state": 42,
        "n_jobs": 7,
    }


class TargetTransformTests(unittest.TestCase):
    def test_preregister_hash_candidate_family_and_blends_are_fixed(self) -> None:
        path = Path(__file__).resolve().parents[1] / "configs" / "target_transform_preregister.json"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), PREREGISTER_SHA256)
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            [item["id"] for item in payload["transforms"]],
            ["logit_eps02", "arcsin_sqrt", "cube_root"],
        )
        self.assertEqual(payload["blend_weights"], [0.05, 0.1, 0.2])
        self.assertEqual(payload["candidate_count"], 9)

    def test_forward_inverse_is_finite_bounded_and_matches_fixed_clips(self) -> None:
        values = np.array([0.0, 0.02, 0.1, 0.5, 0.98, 1.0, 1.02])
        identity = inverse_target_transform(
            forward_target_transform(values, "identity"), "identity"
        )
        cube = inverse_target_transform(
            forward_target_transform(values, "cube_root"), "cube_root"
        )
        np.testing.assert_allclose(identity, values)
        np.testing.assert_allclose(cube, values, rtol=0.0, atol=1e-14)
        logit = inverse_target_transform(
            forward_target_transform(values, "logit_eps02"), "logit_eps02"
        )
        np.testing.assert_allclose(logit, np.clip(values, 0.02, 0.98), atol=1e-15)
        arcsin = inverse_target_transform(
            forward_target_transform(values, "arcsin_sqrt"), "arcsin_sqrt"
        )
        np.testing.assert_allclose(arcsin, np.clip(values, 0.0, 1.0), atol=1e-14)
        for transform in ("identity", "logit_eps02", "arcsin_sqrt", "cube_root"):
            inverse = inverse_target_transform(np.array([-1e6, 0.0, 1e6]), transform)
            self.assertTrue(np.isfinite(inverse).all())
            self.assertTrue(np.logical_and(inverse >= 0.0, inverse <= 1.02).all())

    def test_fit_apply_order_rejects_overlap_or_future_fit(self) -> None:
        fit = pd.date_range("2022-01-01", periods=24, freq="h")
        apply = pd.date_range("2023-01-01", periods=24, freq="h")
        assert_strict_fit_apply_order(fit, apply)
        with self.assertRaises(ValueError):
            assert_strict_fit_apply_order(apply, fit)
        with self.assertRaises(ValueError):
            assert_strict_fit_apply_order(fit, fit)

    def test_training_uses_only_eligible_targets_and_prediction_is_deterministic(self) -> None:
        features = _features()
        target = pd.Series(
            0.4 + 0.2 * np.sin(np.arange(len(features)) / 9.0),
            index=features.index,
        )
        target.iloc[0] = np.nan
        target.iloc[1] = 0.09
        first = TargetTransformRegressor("arcsin_sqrt", _params()).fit(features, target)
        second = TargetTransformRegressor("arcsin_sqrt", _params()).fit(features, target)
        first_prediction, audit = first.predict_cf(features)
        second_prediction, _ = second.predict_cf(features)
        self.assertEqual(first.fit_audit_["fit_rows_eligible"], len(features) - 2)
        np.testing.assert_array_equal(first_prediction.to_numpy(), second_prediction.to_numpy())
        self.assertTrue(first_prediction.between(0.0, 1.02).all())
        self.assertTrue(audit["inverse_finite"])
        self.assertFalse(audit["query_actual_values_accessed"])

    def test_target_scada_or_group_weather_columns_are_rejected(self) -> None:
        base = _features()
        target = pd.Series(np.full(len(base), 0.4), index=base.index)
        for column in ("actual", "target_cf", "scada_power", "kpx_group_1"):
            with self.assertRaises(ValueError):
                TargetTransformRegressor("cube_root", _params()).fit(
                    base.assign(**{column: 0.0}), target
                )

    def test_blend_formula_and_registered_weights_are_exact(self) -> None:
        index = pd.date_range("2023-01-01 01:00:00", periods=24, freq="h")
        transformed = pd.Series(np.full(24, 0.8), index=index)
        baseline = pd.Series(np.full(24, 5_000.0), index=index, name="g")
        output = blend_with_corrected_v3(
            transformed,
            baseline,
            capacity_kwh=10_000.0,
            transformed_weight=0.10,
        )
        np.testing.assert_allclose(output, 0.9 * 5_000.0 + 0.1 * 8_000.0)
        with self.assertRaises(ValueError):
            blend_with_corrected_v3(
                transformed,
                baseline,
                capacity_kwh=10_000.0,
                transformed_weight=0.15,
            )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.probabilistic import (
    ProbabilisticDecisionConfig,
    ProbabilisticFICRDecision,
)


class ProbabilisticFICRDecisionTests(unittest.TestCase):
    @staticmethod
    def _data(start: str, rows: int, residual: float = 0.04):
        index = pd.date_range(start, periods=rows, freq="h")
        phase = np.linspace(0.0, 5.0 * np.pi, rows)
        base_cf = 0.45 + 0.15 * np.sin(phase)
        actual_cf = base_cf + residual + 0.004 * np.cos(phase * 1.7)
        features = pd.DataFrame(
            {
                "base_cf": base_cf,
                "q07_cf": base_cf + 0.035,
                "spread_cf": np.full(rows, 0.035),
            },
            index=index,
        )
        actual = pd.Series(actual_cf * 1000.0, index=index, name="actual")
        base = pd.Series(base_cf * 1000.0, index=index, name="base")
        return features, actual, base

    @staticmethod
    def _config() -> ProbabilisticDecisionConfig:
        return ProbabilisticDecisionConfig(
            minimum_fit_samples=50,
            residual_sample_count=21,
            minimum_bin_samples=30,
            bin_shrinkage_samples=40.0,
            quantile_max_iter=20,
            quantile_min_samples_leaf=20,
            meta_max_iter=20,
            meta_min_samples_leaf=30,
        )

    def test_global_conformal_moves_toward_persistent_residual(self):
        fit_x, fit_y, fit_base = self._data("2022-01-01", 240)
        app_x, app_y, app_base = self._data("2023-01-01", 96)
        decision = ProbabilisticFICRDecision(
            "global_conformal", config=self._config()
        ).fit(fit_x, fit_y, fit_base, capacity_kwh=1000.0)
        prediction = decision.predict(app_x, app_base)
        self.assertTrue((prediction > app_base).all())
        base_mae = np.mean(np.abs(app_base - app_y))
        decision_mae = np.mean(np.abs(prediction - app_y))
        self.assertLess(decision_mae, base_mae)
        self.assertFalse(decision.metadata()["fit_score_calculated"])

    def test_application_overlap_is_rejected(self):
        fit_x, fit_y, fit_base = self._data("2022-01-01", 120)
        decision = ProbabilisticFICRDecision(
            "binned_kde", config=self._config()
        ).fit(fit_x, fit_y, fit_base, capacity_kwh=1000.0)
        with self.assertRaisesRegex(ValueError, "overlap"):
            decision.predict(fit_x, fit_base)

    def test_all_methods_are_deterministic_finite_and_bounded(self):
        fit_x, fit_y, fit_base = self._data("2022-01-01", 240)
        app_x, _, app_base = self._data("2023-01-01", 72)
        for method in ProbabilisticFICRDecision.METHODS:
            with self.subTest(method=method):
                first = ProbabilisticFICRDecision(
                    method, config=self._config()
                ).fit(fit_x, fit_y, fit_base, capacity_kwh=1000.0)
                second = ProbabilisticFICRDecision(
                    method, config=self._config()
                ).fit(fit_x, fit_y, fit_base, capacity_kwh=1000.0)
                p1 = first.predict(app_x, app_base)
                p2 = second.predict(app_x, app_base)
                np.testing.assert_allclose(p1, p2, rtol=0.0, atol=1e-12)
                self.assertTrue(np.isfinite(p1).all())
                self.assertTrue((p1 >= 0.0).all())
                self.assertTrue((p1 <= 1020.0).all())


if __name__ == "__main__":
    unittest.main()

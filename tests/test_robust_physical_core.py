from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_robust_physical_core import (
    PROJECT_DIR,
    _blend,
    _load_preregister,
    _select,
    _transfer_delta,
)


class RobustPhysicalCoreTests(unittest.TestCase):
    def test_composed_v2_preregister_is_locked(self) -> None:
        payload = _load_preregister(
            PROJECT_DIR / "configs/robust_physical_core_preregister_v2.json"
        )
        self.assertEqual(payload["experiment_id"], "robust_physical_core_strict_forward_v2")
        self.assertEqual(len(payload["feature_columns"]), 36)
        self.assertEqual(len(payload["objectives"]) * len(payload["blend_weights"]), 12)

    def test_blend_and_delta_transfer_are_exact(self) -> None:
        index = pd.date_range("2025-01-01 01:00:00", periods=3, freq="h")
        baseline = pd.Series([1000.0, 2000.0, 3000.0], index=index)
        robust = pd.Series([2000.0, 4000.0, 6000.0], index=index)
        direct = _blend(baseline, robust, 0.25, "kpx_group_1")
        np.testing.assert_allclose(direct, [1250.0, 2500.0, 3750.0])
        v4 = pd.Series([900.0, 1800.0, 2700.0], index=index)
        transferred = _transfer_delta(v4, baseline, direct, "kpx_group_1")
        np.testing.assert_allclose(transferred, [1150.0, 2300.0, 3450.0])

    def test_selection_requires_every_registered_slice_positive(self) -> None:
        candidates = {
            "mono_l1__w025": {
                "objective_id": "mono_l1",
                "weight": 0.25,
                "comparisons": {
                    "full": {"delta": 0.02},
                    "H1": {"delta": 0.01},
                    "H2": {"delta": -1e-6},
                },
            },
            "mono_q060__w025": {
                "objective_id": "mono_q060",
                "weight": 0.25,
                "comparisons": {
                    "full": {"delta": 0.01},
                    "H1": {"delta": 0.004},
                    "H2": {"delta": 0.003},
                },
            },
        }
        self.assertEqual(
            _select(candidates, ("full", "H1", "H2")), "mono_q060__w025"
        )

    def test_selection_returns_identity_when_no_candidate_is_stable(self) -> None:
        candidates = {
            "mono_l1__w025": {
                "objective_id": "mono_l1",
                "weight": 0.25,
                "comparisons": {
                    "full": {"delta": 0.01},
                    "H1": {"delta": -0.01},
                    "H2": {"delta": 0.01},
                },
            }
        }
        self.assertIsNone(_select(candidates, ("full", "H1", "H2")))


if __name__ == "__main__":
    unittest.main()

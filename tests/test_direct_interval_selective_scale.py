from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from src.direct_interval_selective_scale import (
    ACTION_GRID_CF,
    PROMOTED_GROUPS,
    SCALE_FACTOR,
    SEGMENTS,
    UTILITY_MARGIN,
    interpolate_row_utility,
    selective_scale_candidate,
    stage1_group_promotions,
    stage2_promoted,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
V2_SHA = "217ec04c2943bb02fc53a6d8c02ee5ab9c715d9e0d55c0b8b5dae1cbf1391f03"
V3_SHA = "917a197037d3ad1e625805efafa5f5141340546e38a96a18e61da4245383a3fe"


class DirectIntervalSelectiveScaleTests(unittest.TestCase):
    def test_both_configs_frozen_before_2024(self) -> None:
        for filename, expected in (
            ("direct_interval_selective_scale_preregister_v2.json", V2_SHA),
            ("direct_interval_public_adaptive_preregister_v3.json", V3_SHA),
        ):
            path = PROJECT_DIR / "configs" / filename
            observed = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(observed, expected)
            self.assertEqual(
                path.with_suffix(".sha256").read_text(encoding="utf-8"),
                f"{observed}  {filename}\n",
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "frozen_before_any_2024_prediction_or_label_value_read_for_v2" if "v2" in filename else "frozen_before_any_2024_prediction_or_label_value_read_for_v2_or_v3")

    def test_linear_interpolation_exact_including_upper_endpoint(self) -> None:
        surface = np.vstack((ACTION_GRID_CF, 2.0 * ACTION_GRID_CF))
        observed = interpolate_row_utility(surface, np.array([0.125, 1.02]))
        np.testing.assert_allclose(observed, [0.125, 2.04], atol=0.0, rtol=0.0)

    def test_v2_selective_gate_is_strict_and_identity_elsewhere(self) -> None:
        index = pd.date_range("2024-01-01", periods=2, freq="h")
        baseline = pd.Series([0.5 * 21600, 0.5 * 21600], index=index)
        surface = np.zeros((2, 103))
        # At base 0.50 utility is zero.  At scaled 0.49, set one row exactly at
        # the margin and one just above it.  Strict > must select only row two.
        surface[0, 49] = UTILITY_MARGIN
        surface[1, 49] = UTILITY_MARGIN + 1e-6
        candidate, diagnostics = selective_scale_candidate(
            baseline, surface, capacity_kwh=21600.0
        )
        self.assertEqual(SCALE_FACTOR, 0.98)
        self.assertEqual(diagnostics["gate"].tolist(), [False, True])
        self.assertEqual(candidate.iloc[0], baseline.iloc[0])
        self.assertEqual(candidate.iloc[1], baseline.iloc[1] * 0.98)

    @staticmethod
    def _record(delta: float) -> dict[str, object]:
        return {
            "baseline": {"score": 0.5},
            "candidate": {"score": 0.5 + delta},
            "delta": (0.5 + delta) - 0.5,
        }

    def test_group_specific_stage1_promotion(self) -> None:
        required = {
            "kpx_group_1": SEGMENTS,
            "kpx_group_2": SEGMENTS,
            "kpx_group_3": ("full", "Q3", "Q4"),
        }
        comparisons = {
            group: {name: self._record(0.001) for name in names}
            for group, names in required.items()
        }
        comparisons["kpx_group_3"]["Q3"] = self._record(-0.001)
        passed, failed, audit = stage1_group_promotions(comparisons, required)
        self.assertEqual(passed, PROMOTED_GROUPS)
        self.assertEqual(failed, ("kpx_group_3",))
        self.assertFalse(audit["kpx_group_3"]["all_strictly_positive"])

    def test_stage2_requires_group_mixed_and_full_components(self) -> None:
        groups = {
            group: {name: self._record(0.001) for name in SEGMENTS}
            for group in PROMOTED_GROUPS
        }
        mixed = {
            name: {
                "baseline": {"total_score": 0.5, "one_minus_nmae": 0.7, "ficr": 0.3},
                "candidate": {"total_score": 0.501, "one_minus_nmae": 0.701, "ficr": 0.301},
                "delta_total_score": 0.0010000000000000009,
                "delta_one_minus_nmae": 0.0010000000000000009,
                "delta_ficr": 0.0010000000000000009,
            }
            for name in SEGMENTS
        }
        promoted, audit = stage2_promoted(groups, mixed)
        self.assertTrue(promoted)
        self.assertTrue(audit["full_components_nonnegative"])
        mixed["full"]["candidate"]["ficr"] = 0.299
        mixed["full"]["delta_ficr"] = -0.0010000000000000009
        promoted, audit = stage2_promoted(groups, mixed)
        self.assertFalse(promoted)
        self.assertFalse(audit["full_components_nonnegative"])


if __name__ == "__main__":
    unittest.main()

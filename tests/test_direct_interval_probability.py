from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from src.direct_interval_probability import (
    ACTION_GRID_CF,
    BLEND_WEIGHTS,
    COMMON_PARAMETERS,
    CONTEXT_COLUMNS,
    DirectIntervalProbabilityModel,
    assert_strict_forward,
    blend_action_kwh,
    compact_context,
    expanded_action_design,
    official_action_utility,
    repair_interval_probabilities,
    select_actions,
    select_stage1_weight,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
PREREGISTER_SHA256 = "1796cc8628e3ad46039a6f5352c4c51b1a851afc7e6e6f055d9c85fcd1d699ff"


def _context(rows: int = 140) -> pd.DataFrame:
    index = pd.date_range("2022-01-01 01:00", periods=rows, freq="h", name="forecast_kst_dtm")
    x = np.linspace(0.0, 1.0, rows)
    data = {
        column: (position + 1.0) * 0.01 + x * (0.2 + position * 0.001)
        for position, column in enumerate(CONTEXT_COLUMNS)
    }
    return pd.DataFrame(data, index=index, columns=CONTEXT_COLUMNS)


class DirectIntervalProbabilityTests(unittest.TestCase):
    def test_preregister_hash_and_exact_contract(self) -> None:
        path = PROJECT_DIR / "configs/direct_interval_probability_preregister_v1.json"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(digest, PREREGISTER_SHA256)
        sidecar = path.with_suffix(".sha256").read_text(encoding="utf-8")
        self.assertEqual(sidecar, f"{digest}  {path.name}\n")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            tuple(payload["feature_contract"]["context_columns_in_exact_order"]),
            CONTEXT_COLUMNS,
        )
        self.assertEqual(payload["feature_contract"]["expanded_feature_count"], 27)
        self.assertEqual(payload["action_expansion_and_targets"]["action_count"], 103)
        self.assertEqual(
            tuple(payload["candidate_family"]["fixed_global_blend_weights"]),
            tuple(BLEND_WEIGHTS.values()),
        )
        self.assertEqual(payload["models"]["common_parameters"], COMMON_PARAMETERS)
        self.assertTrue(payload["public_contract"]["public_scale_artifacts_used"] is False)

    def test_compact_schema_and_row_major_action_expansion(self) -> None:
        source = _context(2)
        source["unused"] = 7.0
        compact = compact_context(source)
        self.assertEqual(tuple(compact.columns), CONTEXT_COLUMNS)
        expanded = expanded_action_design(compact)
        self.assertEqual(expanded.shape, (206, 27))
        np.testing.assert_array_equal(expanded[:103, -1], ACTION_GRID_CF.astype(np.float32))
        np.testing.assert_array_equal(expanded[103:, -1], ACTION_GRID_CF.astype(np.float32))
        np.testing.assert_array_equal(expanded[0, :-1], expanded[102, :-1])
        self.assertFalse(np.array_equal(expanded[102, :-1], expanded[103, :-1]))

    def test_probability_repair_is_pairwise_euclidean_projection(self) -> None:
        p6 = np.array([[0.8, 0.2], [1.2, -0.1]])
        p8 = np.array([[0.4, 0.5], [0.9, 0.1]])
        repaired6, repaired8, count = repair_interval_probabilities(p6, p8)
        np.testing.assert_allclose(repaired6, [[0.6, 0.2], [0.95, 0.0]])
        np.testing.assert_allclose(repaired8, [[0.6, 0.5], [0.95, 0.1]])
        self.assertEqual(count, 2)
        self.assertTrue(np.all(repaired6 <= repaired8))

    def test_official_utility_formula_and_action_tie_break(self) -> None:
        p6 = np.array([[0.2, 0.4], [0.5, 0.5]])
        p8 = np.array([[0.6, 0.8], [0.8, 0.8]])
        eae = np.array([[0.1, 0.2], [0.1, 0.1]])
        utility = official_action_utility(p6, p8, eae)
        expected = -0.5 * eae + 0.5 * (0.25 * p6 + 0.75 * p8)
        np.testing.assert_allclose(utility, expected)

        surface = np.zeros((2, 103))
        surface[0, [40, 60]] = 1.0
        surface[1, [40, 60]] = 1.0
        action, value, index = select_actions(surface, np.array([0.57, 0.50]))
        np.testing.assert_array_equal(index, [60, 40])
        np.testing.assert_allclose(action, [0.60, 0.40])
        np.testing.assert_allclose(value, [1.0, 1.0])

    def test_strict_forward_and_zero_blend_identity(self) -> None:
        fit = pd.date_range("2022-01-01", periods=2, freq="h")
        apply = pd.date_range("2022-01-02", periods=2, freq="h")
        assert_strict_forward(fit, apply)
        with self.assertRaises(ValueError):
            assert_strict_forward(apply, fit)
        baseline = pd.Series([1000.0, 2000.0], index=apply, name="g")
        action = pd.Series([0.9, 0.1], index=apply)
        observed = blend_action_kwh(baseline, action, weight=0.0, capacity_kwh=21600)
        self.assertEqual(
            np.ascontiguousarray(observed.to_numpy()).tobytes(),
            np.ascontiguousarray(baseline.to_numpy()).tobytes(),
        )

    def test_model_fits_three_heads_without_application_label_use(self) -> None:
        context = _context(140)
        actual_cf = 0.15 + 0.65 * np.square(np.sin(np.linspace(0, 3.0, len(context))))
        actual = pd.Series(actual_cf * 21600.0, index=context.index)
        model = DirectIntervalProbabilityModel().fit(context, actual, capacity_kwh=21600.0)
        baseline = pd.Series(actual.to_numpy() * 0.95, index=context.index)
        action, diagnostics, surface = model.predict_action(
            context.iloc[-8:], baseline.iloc[-8:], capacity_kwh=21600.0
        )
        self.assertEqual(len(action), 8)
        self.assertTrue(action.between(0.0, 1.02).all())
        self.assertTrue((diagnostics["selected_p6"] <= diagnostics["selected_p8"]).all())
        self.assertTrue(np.all(np.asarray(surface["p6"]) <= np.asarray(surface["p8"])))
        metadata = model.metadata()
        self.assertEqual(metadata["expanded_rows"], metadata["rows_eligible"] * 103)
        self.assertFalse(metadata["fit_score_calculated"])
        self.assertFalse(metadata["application_labels_used"])

    def test_stage1_selection_requires_all_17_strictly_positive(self) -> None:
        groups = {
            "kpx_group_1": ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4"),
            "kpx_group_2": ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4"),
            "kpx_group_3": ("full", "Q3", "Q4"),
        }
        comparisons = {}
        for key, weight in BLEND_WEIGHTS.items():
            comparisons[key] = {}
            for group, segments in groups.items():
                comparisons[key][group] = {
                    segment: {
                        "baseline": {"score": 0.5},
                        "candidate": {"score": 0.5 + weight},
                        "delta": (0.5 + weight) - 0.5,
                    }
                    for segment in segments
                }
        selected, audit = select_stage1_weight(comparisons)
        self.assertEqual(selected, 0.10)
        self.assertEqual(audit["selected"], "w10")
        comparisons["w10"]["kpx_group_3"]["Q4"] = {
            "baseline": {"score": 0.5},
            "candidate": {"score": 0.49},
            "delta": -0.010000000000000009,
        }
        selected, audit = select_stage1_weight(comparisons)
        self.assertEqual(selected, 0.05)
        self.assertEqual(audit["selected"], "w05")


if __name__ == "__main__":
    unittest.main()

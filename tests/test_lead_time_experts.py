from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.lead_time_experts import (
    choose_registered_candidate,
    fit_global_and_lead_experts,
    lead_bin_masks,
    specialization_candidate,
    validate_lead_partition,
)


BINS = [
    {"id": "lead12_17", "lower_inclusive": 12, "upper_exclusive": 18},
    {"id": "lead18_23", "lower_inclusive": 18, "upper_exclusive": 24},
    {"id": "lead24_29", "lower_inclusive": 24, "upper_exclusive": 30},
    {"id": "lead30_35", "lower_inclusive": 30, "upper_exclusive": 36},
]


def _features(runs: int) -> pd.DataFrame:
    index = pd.date_range("2022-01-01 01:00", periods=24 * runs, freq="h")
    lead = np.tile(np.arange(12, 36, dtype=np.float32), runs)
    return pd.DataFrame(
        {
            "time__lead_hours": lead,
            "weather_a": np.sin(np.arange(len(index)) / 17.0).astype(np.float32),
            "weather_b": np.cos(np.arange(len(index)) / 31.0).astype(np.float32),
        },
        index=index,
    )


class LeadTimeExpertsTest(unittest.TestCase):
    def test_partition_has_four_disjoint_six_hour_bins(self) -> None:
        features = _features(3)
        masks, audit = validate_lead_partition(features, BINS)
        self.assertEqual(audit["runs"], 3)
        self.assertTrue(np.all(np.sum(list(masks.values()), axis=0) == 1))
        self.assertEqual([int(mask.sum()) for mask in masks.values()], [18] * 4)

    def test_partition_rejects_gap_or_incomplete_run(self) -> None:
        features = _features(2)
        broken = [dict(item) for item in BINS]
        broken[1]["lower_inclusive"] = 19
        with self.assertRaises(ValueError):
            lead_bin_masks(features["time__lead_hours"], broken)
        with self.assertRaises(ValueError):
            validate_lead_partition(features.iloc[:-1], BINS)

    def test_partition_rejects_target_like_feature(self) -> None:
        features = _features(1)
        features["scada_wind"] = 1.0
        with self.assertRaises(ValueError):
            validate_lead_partition(features, BINS)

    def test_specialization_delta_and_clip(self) -> None:
        index = pd.date_range("2023-01-01", periods=3, freq="h")
        baseline = pd.Series([10.0, 90.0, 20.0], index=index)
        expert = pd.Series([30.0, 150.0, -50.0], index=index)
        global_control = pd.Series([10.0, 50.0, 30.0], index=index)
        output = specialization_candidate(
            baseline,
            expert,
            global_control,
            weight=0.5,
            capacity_kwh=100.0,
        )
        np.testing.assert_allclose(output, [20.0, 102.0, 0.0])

    def test_selection_requires_every_delta_positive_and_tie_breaks_weight(self) -> None:
        comparisons = {
            "l1__w10": {"g": {"full": {"delta": 0.01}}},
            "l1__w05": {"g": {"full": {"delta": 0.01}}},
            "q07__w05": {"g": {"full": {"delta": -1e-8}}},
        }
        specifications = {
            "l1__w10": {"weight": 0.10, "recipe_id": "expert_l1"},
            "l1__w05": {"weight": 0.05, "recipe_id": "expert_l1"},
            "q07__w05": {"weight": 0.05, "recipe_id": "expert_q07"},
        }
        self.assertEqual(
            choose_registered_candidate(comparisons, specifications), "l1__w05"
        )

    def test_fit_stitches_only_matching_lead_rows_and_is_forward(self) -> None:
        features = _features(40)
        validate_lead_partition(features, BINS)
        actual = pd.Series(
            np.clip(
                0.45
                + 0.15 * features["weather_a"].to_numpy()
                + 0.002 * features["time__lead_hours"].to_numpy(),
                0.1,
                1.0,
            ),
            index=features.index,
        )
        fit_index = features.index[: 20 * 24]
        apply_index = features.index[20 * 24 :]
        contract = {
            "common_params": {
                "n_estimators": 5,
                "learning_rate": 0.1,
                "num_leaves": 7,
                "min_child_samples": 5,
                "subsample": 1.0,
                "subsample_freq": 0,
                "colsample_bytree": 1.0,
                "reg_alpha": 0.0,
                "reg_lambda": 0.0,
                "verbosity": -1,
                "deterministic": True,
                "force_col_wise": True,
            },
            "random_state": 42,
            "n_jobs": 1,
        }
        recipe = {"id": "expert_l1", "objective": "regression_l1", "alpha": None}
        _, experts, global_prediction, stitched, audit = fit_global_and_lead_experts(
            features=features,
            actual_cf=actual,
            fit_index=fit_index,
            apply_index=apply_index,
            bins=BINS,
            recipe=recipe,
            model_contract=contract,
        )
        self.assertEqual(set(experts), {item["id"] for item in BINS})
        self.assertEqual(len(global_prediction), len(apply_index))
        self.assertTrue(np.isfinite(stitched).all())
        self.assertTrue(audit["fit_max_strictly_before_apply_min"])
        self.assertEqual(sum(item["apply_rows"] for item in audit["bins"].values()), 480)


if __name__ == "__main__":
    unittest.main()

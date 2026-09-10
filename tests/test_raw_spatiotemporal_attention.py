from __future__ import annotations

import json
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch

from src.metric import TARGET_COLS
from src.raw_grid_wind import SOURCE_CHANNELS, SOURCE_GRID_COUNTS, TIME_FEATURES
from src import raw_spatiotemporal_attention as module
from src.raw_spatiotemporal_attention import (
    BLEND_WEIGHTS,
    CANDIDATE_KEYS,
    MODEL_ID,
    RawSpatiotemporalAttentionRegressor,
    RawSpatiotemporalTransformer,
    SharedNodeGeoAttentionConvNet,
    assert_fit_before_apply,
    blend_attention_prediction,
    candidate_frame,
    parse_candidate_key,
    select_group_candidate,
)


def _catalogs() -> dict[str, list[dict[str, float | int]]]:
    return {
        source: [
            {
                "grid_id": grid,
                "latitude": 37.0 + 0.01 * grid,
                "longitude": 128.0 + 0.02 * grid,
            }
            for grid in range(1, SOURCE_GRID_COUNTS[source] + 1)
        ]
        for source in ("ldaps", "gfs")
    }


def _features(start: str, runs: int, seed: int = 42) -> pd.DataFrame:
    index = pd.date_range(start, periods=runs * 24, freq="h", name="forecast_kst_dtm")
    raw_columns = [
        f"{source}__grid_{grid:02d}__{channel}"
        for source in ("ldaps", "gfs")
        for channel in SOURCE_CHANNELS[source]
        for grid in range(1, SOURCE_GRID_COUNTS[source] + 1)
    ]
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(len(index), len(raw_columns) + len(TIME_FEATURES))).astype(
        np.float32
    )
    return pd.DataFrame(values, index=index, columns=[*raw_columns, *TIME_FEATURES])


def _comparison(delta: float) -> dict[str, object]:
    candidate = 0.5 + delta
    return {
        "baseline": {"score": 0.5},
        "candidate": {"score": candidate},
        "delta": candidate - 0.5,
    }


class RawSpatiotemporalAttentionTests(unittest.TestCase):
    def test_transformer_outputs_source_run_tensors_and_is_fit_only(self) -> None:
        fit = _features("2022-01-01 01:00:00", 3)
        fit.iloc[0, 0] = np.nan
        transformer = RawSpatiotemporalTransformer(_catalogs()).fit(fit)
        means = {
            source: np.asarray(transformer.statistics_[source]["mean"]).copy()
            for source in ("ldaps", "gfs")
        }
        app = _features("2022-01-04 01:00:00", 2, seed=7)
        tensors, keys = transformer.transform(app)
        self.assertEqual(tensors["ldaps"].shape, (2, 24, 16, 8))
        self.assertEqual(tensors["gfs"].shape, (2, 24, 9, 8))
        self.assertEqual(len(keys), 2)
        for source in ("ldaps", "gfs"):
            np.testing.assert_array_equal(transformer.statistics_[source]["mean"], means[source])
        self.assertFalse(
            transformer.metadata()[
                "baseline_calendar_issuance_target_scada_public_scale_features_used"
            ]
        )

    def test_network_has_shared_encoder_source_attention_group_heads_and_shape(self) -> None:
        model = SharedNodeGeoAttentionConvNet().eval()
        self.assertEqual(set(model.source_attention), {"ldaps", "gfs"})
        self.assertEqual(set(model.group_heads), set(TARGET_COLS))
        ldaps = torch.zeros((2, 24, 16, 8))
        gfs = torch.zeros((2, 24, 9, 8))
        coords_l = torch.zeros((16, 2))
        coords_g = torch.zeros((9, 2))
        with torch.no_grad():
            output = model(ldaps, gfs, coords_l, coords_g)
        self.assertEqual(tuple(output.shape), (2, 3, 24))
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(((output >= 0.0) & (output <= 1.02)).all())

    def test_one_epoch_multitask_fit_handles_missing_group_runs(self) -> None:
        fit = _features("2022-01-01 01:00:00", 3)
        app = _features("2022-01-04 01:00:00", 1, seed=9)
        targets = pd.DataFrame(0.4, index=fit.index, columns=TARGET_COLS)
        targets.loc[fit.index[:24], TARGET_COLS[2]] = np.nan
        with mock.patch.object(module, "EPOCHS", 1), mock.patch.object(module, "SEEDS", (42,)):
            model = RawSpatiotemporalAttentionRegressor(_catalogs()).fit(fit, targets)
            prediction = model.predict(app)
        self.assertEqual(prediction.shape, (24, 3))
        self.assertTrue(np.isfinite(prediction.to_numpy()).all())
        self.assertEqual(model.metadata()["eligible_runs_by_group"][TARGET_COLS[2]], 2)

    def test_strict_boundary_blends_parser_and_gate(self) -> None:
        fit = _features("2022-01-01 01:00:00", 2).index
        app = _features("2022-01-03 01:00:00", 1).index
        self.assertTrue(assert_fit_before_apply(fit, app)["fit_runs_complete"])
        with self.assertRaises(ValueError):
            assert_fit_before_apply(fit, fit[-24:])
        baseline = pd.Series(5000.0, index=app)
        direct = pd.Series(np.linspace(0.0, 1.0, len(app)), index=app)
        identity = blend_attention_prediction(
            baseline, direct, capacity_kwh=21600.0, weight=0.0
        )
        self.assertEqual(identity.to_numpy().tobytes(), baseline.to_numpy().tobytes())
        candidates = candidate_frame(baseline, direct, capacity_kwh=21600.0)
        self.assertEqual(tuple(candidates.columns), CANDIDATE_KEYS)
        self.assertEqual(parse_candidate_key(f"{MODEL_ID}_w025"), 0.025)
        with self.assertRaises(ValueError):
            parse_candidate_key(f"{MODEL_ID}_w100")
        slices = ("full", "H1")
        comparisons = {
            key: {name: _comparison(0.01 if key.endswith("w025") else -0.01) for name in slices}
            for key in CANDIDATE_KEYS
        }
        sorted_copy = json.loads(json.dumps(comparisons, sort_keys=True))
        selected, _ = select_group_candidate(sorted_copy, slices)
        self.assertEqual(selected, f"{MODEL_ID}_w025")
        self.assertEqual(BLEND_WEIGHTS, (0.025, 0.05))


if __name__ == "__main__":
    unittest.main()

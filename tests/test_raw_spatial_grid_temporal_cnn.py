from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import torch

from src.metric import CAPACITY_KWH, TARGET_COLS
from src import raw_spatial_grid_temporal_cnn as module


ROOT = Path(__file__).resolve().parents[1]


class RawSpatialGridTemporalCNNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_contract = json.loads(
            (ROOT / "configs/raw_grid_wind_lgb_preregister_v1.json").read_text(encoding="utf-8")
        )

    def test_preregistered_constants_are_exact(self) -> None:
        self.assertEqual(module.SEEDS, (42, 2026))
        self.assertEqual((module.EPOCHS, module.BATCH_SIZE), (240, 32))
        self.assertEqual((module.LEARNING_RATE, module.WEIGHT_DECAY), (0.002, 0.001))
        self.assertEqual((module.SMOOTH_L1_BETA, module.LOW_CF_LOSS_WEIGHT), (0.05, 0.25))
        self.assertEqual(module.BLEND_WEIGHT, 0.025)

    def test_registered_layouts_are_lossless_and_geographic(self) -> None:
        observed = module.validate_raster_catalogs(self.raw_contract["grid_catalog"])
        self.assertEqual(observed["ldaps"]["shape"], [4, 5])
        self.assertEqual(observed["ldaps"]["missing_cells"], 4)
        self.assertEqual(observed["gfs"]["shape"], [3, 3])
        self.assertTrue(observed["ldaps"]["lossless_node_permutation"])

    def test_ldaps_rasterization_is_value_bit_lossless(self) -> None:
        values = np.arange(2 * 24 * 16 * 8, dtype=np.float32).reshape(2, 24, 16, 8)
        coords = np.linspace(-1, 1, 32, dtype=np.float32).reshape(16, 2)
        raster = module.rasterize_source(values, coords, "ldaps")
        self.assertEqual(raster.shape, (2, 24, 11, 4, 5))
        for row in range(4):
            for column in range(5):
                grid_id = int(module.LDAPS_LAYOUT[row, column])
                if grid_id:
                    self.assertTrue(np.array_equal(raster[:, :, :8, row, column], values[:, :, grid_id - 1]))

    def test_gfs_rasterization_is_value_bit_lossless(self) -> None:
        rng = np.random.default_rng(42)
        values = rng.normal(size=(3, 24, 9, 8)).astype(np.float32)
        coords = rng.normal(size=(9, 2)).astype(np.float32)
        raster = module.rasterize_source(values, coords, "gfs")
        self.assertEqual(raster.shape, (3, 24, 11, 3, 3))
        self.assertTrue((raster[:, :, 8].sum(axis=(-2, -1)) == 9).all())

    def test_ldaps_missing_corners_are_exact_zero(self) -> None:
        values = np.ones((1, 24, 16, 8), dtype=np.float32)
        coords = np.ones((16, 2), dtype=np.float32)
        raster = module.rasterize_source(values, coords, "ldaps")
        for row, column in ((0, 0), (0, 4), (3, 0), (3, 4)):
            self.assertEqual(raster[:, :, :, row, column].tobytes(), np.zeros((1, 24, 11), np.float32).tobytes())

    def test_network_parameter_count_output_and_cuda_determinism(self) -> None:
        self.assertTrue(torch.cuda.is_available())
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.use_deterministic_algorithms(True)
        model = module.SpatialGridTemporalConvNet().cuda().eval()
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 41235)
        ldaps = torch.zeros(2, 24, 11, 4, 5, device="cuda")
        gfs = torch.zeros(2, 24, 11, 3, 3, device="cuda")
        ldaps[:, :, 8] = torch.as_tensor(module.LDAPS_LAYOUT > 0, device="cuda")
        gfs[:, :, 8] = 1.0
        with torch.no_grad():
            left = model(ldaps, gfs)
            right = model(ldaps, gfs)
        self.assertEqual(tuple(left.shape), (2, 3, 24))
        self.assertTrue(torch.equal(left, right))
        self.assertTrue(torch.isfinite(left).all())

    def test_spatial_encoder_is_distinct_and_source_specific(self) -> None:
        model = module.SpatialGridTemporalConvNet()
        self.assertIsNot(model.ldaps_spatial, model.gfs_spatial)
        self.assertFalse(any("attention" in name.lower() for name, _ in model.named_modules()))
        self.assertEqual(tuple(model.ldaps_spatial.conv1.kernel_size), (3, 3))

    def test_single_candidate_formula_and_identity(self) -> None:
        index = pd.date_range("2023-01-01 01:00", periods=3, freq="h")
        baseline = pd.Series([1000.0, 2000.0, 3000.0], index=index, name=TARGET_COLS[0])
        cf = pd.Series([0.2, 0.4, 0.6], index=index)
        observed = module.candidate_series(baseline, cf, capacity_kwh=CAPACITY_KWH[TARGET_COLS[0]])
        expected = np.clip(0.975 * baseline.to_numpy() + 0.025 * cf.to_numpy() * 21600.0, 0, 1.02 * 21600.0)
        self.assertTrue(np.array_equal(observed.to_numpy(), expected))
        identity = module.blend_prediction(baseline, cf, capacity_kwh=21600.0, weight=0.0)
        self.assertEqual(identity.to_numpy().tobytes(), baseline.to_numpy().tobytes())

    @staticmethod
    def _comparison(component: float = 0.01, bad_slice: str | None = None) -> dict[str, dict]:
        result: dict[str, dict] = {}
        for name in ("full", "H1", "H2"):
            baseline_score = 0.5
            candidate_score = 0.5 + (0.0 if name == bad_slice else 0.01)
            result[name] = {
                "baseline": {"score": baseline_score, "one_minus_nmae": 0.7, "ficr": 0.3},
                "candidate": {
                    "score": candidate_score,
                    "one_minus_nmae": 0.7 + (component if name == "full" else 0.0),
                    "ficr": 0.3 + (component if name == "full" else 0.0),
                },
                "delta": candidate_score - baseline_score,
            }
        return result

    def test_stage1_gate_requires_slices_and_full_components(self) -> None:
        self.assertTrue(module.stage1_group_gate(self._comparison(), ("full", "H1", "H2"))["passed"])
        self.assertFalse(module.stage1_group_gate(self._comparison(bad_slice="H2"), ("full", "H1", "H2"))["passed"])
        self.assertFalse(module.stage1_group_gate(self._comparison(component=-0.01), ("full", "H1", "H2"))["passed"])

    def test_only_availability_run_alignment_is_accepted(self) -> None:
        availability = pd.date_range("2023-01-01 01:00", periods=24, freq="h", name="forecast_kst_dtm")
        keys = module._run_keys(availability)
        self.assertEqual(len(keys), 1)
        natural = pd.date_range("2023-01-01 00:00", periods=24, freq="h", name="forecast_kst_dtm")
        with self.assertRaises(ValueError):
            module._run_keys(natural)


if __name__ == "__main__":
    unittest.main()

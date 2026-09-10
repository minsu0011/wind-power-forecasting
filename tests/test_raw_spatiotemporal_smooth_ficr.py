from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch

from src.raw_spatiotemporal_attention import (
    BATCH_SIZE,
    BLEND_WEIGHTS,
    EPOCHS,
    MODEL_ID as BASE_MODEL_ID,
    SharedNodeGeoAttentionConvNet,
)
from src.raw_spatiotemporal_smooth_ficr import (
    CANDIDATE_KEYS,
    MAE_COEFFICIENT,
    MODEL_ID,
    OFFICIAL_ELIGIBILITY_CF,
    OFFICIAL_THRESHOLDS_CF,
    RawSpatiotemporalSmoothFICRRegressor,
    SETTLEMENT_WEIGHTS,
    SIGMOID_CLIP,
    SMOOTH_ABSOLUTE_EPSILON_CF,
    TEMPERATURE_CF,
    candidate_frame,
    select_group_candidate,
    torch_smooth_ficr_elementwise,
)
from src.smooth_ficr_objective import smooth_ficr_loss_grad_hess


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/raw_spatiotemporal_smooth_ficr_preregister_v1.json"
CONFIG_SHA = "0dc17775151963f9324186c922deabdc2bb1f869f051b156091255eb284dde08"


def _metric(score: float, one_minus_nmae: float, ficr: float) -> dict[str, float]:
    return {
        "score": score,
        "one_minus_nmae": one_minus_nmae,
        "ficr": ficr,
    }


def _comparisons(
    slices: tuple[str, ...],
    *,
    first_delta: float = 0.002,
    second_delta: float = 0.001,
    nmae_delta: float = 0.001,
    ficr_delta: float = 0.001,
) -> dict[str, dict[str, dict[str, object]]]:
    result: dict[str, dict[str, dict[str, object]]] = {}
    for key, delta in zip(CANDIDATE_KEYS, (first_delta, second_delta)):
        records: dict[str, dict[str, object]] = {}
        for name in slices:
            baseline = _metric(0.6, 0.85, 0.35)
            candidate = _metric(0.6 + delta, 0.85, 0.35)
            if name == "full":
                candidate["one_minus_nmae"] += nmae_delta
                candidate["ficr"] += ficr_delta
            records[name] = {
                "baseline": baseline,
                "candidate": candidate,
                "delta": candidate["score"] - baseline["score"],
            }
        result[key] = records
    return result


class RawSpatiotemporalSmoothFICRTests(unittest.TestCase):
    def test_01_preregister_hash_and_sidecar_are_frozen(self) -> None:
        digest = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
        self.assertEqual(digest, CONFIG_SHA)
        self.assertEqual(
            CONFIG.with_suffix(".sha256").read_text(encoding="utf-8"),
            f"{CONFIG_SHA}  {CONFIG.name}\n",
        )
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(payload["candidate_family"]["fixed_blend_weights"], [0.025, 0.05])

    def test_02_architecture_and_training_identifiers_are_unchanged(self) -> None:
        self.assertTrue(issubclass(RawSpatiotemporalSmoothFICRRegressor, object))
        self.assertEqual(BASE_MODEL_ID, "node32_geoattn_conv48_e240_s2")
        self.assertEqual(MODEL_ID, f"{BASE_MODEL_ID}_sficr")
        self.assertEqual(sum(p.numel() for p in SharedNodeGeoAttentionConvNet().parameters()), 34373)
        self.assertEqual((EPOCHS, BATCH_SIZE, BLEND_WEIGHTS), (240, 32, (0.025, 0.05)))

    def test_03_loss_constants_are_exact(self) -> None:
        self.assertEqual(OFFICIAL_ELIGIBILITY_CF, 0.10)
        self.assertEqual(OFFICIAL_THRESHOLDS_CF, (0.06, 0.08))
        self.assertEqual(SETTLEMENT_WEIGHTS, (1.0, 3.0))
        self.assertEqual(MAE_COEFFICIENT, 0.5)
        self.assertEqual(TEMPERATURE_CF, 0.005)
        self.assertEqual(SMOOTH_ABSOLUTE_EPSILON_CF, 0.0025)
        self.assertEqual(SIGMOID_CLIP, 40.0)

    def test_04_torch_loss_and_autograd_match_registered_numpy(self) -> None:
        errors = np.asarray(
            (-0.081, -0.080, -0.079, -0.061, -0.060, -0.059, 0.0,
             0.059, 0.060, 0.061, 0.079, 0.080, 0.081),
            dtype=np.float64,
        )
        actual = np.resize(np.asarray((0.12, 0.30, 0.55, 0.90)), len(errors))
        prediction = torch.tensor(
            (actual + errors).reshape(1, 1, -1), dtype=torch.float64, requires_grad=True
        )
        truth = torch.tensor(actual.reshape(1, 1, -1), dtype=torch.float64)
        mean = torch.tensor([[[float(actual.mean())]]], dtype=torch.float64)
        loss = torch_smooth_ficr_elementwise(prediction, truth, mean)
        loss.sum().backward()
        np_loss, np_gradient, _ = smooth_ficr_loss_grad_hess(
            actual,
            actual + errors,
            mean_train_actual_cf=float(actual.mean()),
        )
        np.testing.assert_allclose(loss.detach().numpy().ravel(), np_loss, atol=1e-14, rtol=0)
        np.testing.assert_allclose(
            prediction.grad.detach().numpy().ravel(), np_gradient, atol=2e-13, rtol=0
        )

    def test_05_candidate_formula_order_and_identity(self) -> None:
        index = pd.date_range("2023-01-01 01:00", periods=3, freq="h")
        baseline = pd.Series([1000.0, 5000.0, 10000.0], index=index, name="kpx_group_1")
        model_cf = pd.Series([0.2, 0.4, 0.6], index=index)
        frame = candidate_frame(baseline, model_cf, capacity_kwh=21600.0)
        self.assertEqual(tuple(frame.columns), CANDIDATE_KEYS)
        for key, weight in zip(CANDIDATE_KEYS, BLEND_WEIGHTS):
            expected = np.clip(
                (1.0 - weight) * baseline.to_numpy() + weight * model_cf.to_numpy() * 21600.0,
                0.0,
                1.02 * 21600.0,
            )
            np.testing.assert_array_equal(frame[key].to_numpy(), expected)

    def test_06_selector_requires_all_slices_and_chooses_maximin(self) -> None:
        slices = ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4")
        selected, audit = select_group_candidate(_comparisons(slices), slices)
        self.assertEqual(selected, CANDIDATE_KEYS[0])
        self.assertTrue(audit["candidates"][selected]["passed"])

    def test_07_selector_rejects_negative_full_component(self) -> None:
        slices = ("full", "Q3", "Q4")
        selected, audit = select_group_candidate(
            _comparisons(slices, nmae_delta=0.001, ficr_delta=-0.0001), slices
        )
        self.assertIsNone(selected)
        self.assertEqual(audit["selected"], "identity")
        self.assertFalse(audit["candidates"][CANDIDATE_KEYS[0]]["full_components_nonnegative"])

    def test_08_cuda_fallback_is_fail_armed(self) -> None:
        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "fallback is forbidden"):
                RawSpatiotemporalSmoothFICRRegressor._device()


if __name__ == "__main__":
    unittest.main()

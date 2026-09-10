from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from src.metric import TARGET_COLS
from src.raw_spatial_grid_temporal_cnn_scada_multitask import (
    AUX_CHANNELS,
    EXPECTED_PARAMETER_COUNT,
    RawSpatialGridTemporalCNNScadaMultitaskRegressor,
    SpatialGridTemporalConvNetSCADAMultitask,
    multitask_loss,
    prepare_auxiliary_targets,
)


def test_registered_parameter_count() -> None:
    model = SpatialGridTemporalConvNetSCADAMultitask()
    assert sum(parameter.numel() for parameter in model.parameters()) == EXPECTED_PARAMETER_COUNT


def test_forward_direct_and_auxiliary_ranges() -> None:
    model = SpatialGridTemporalConvNetSCADAMultitask().eval()
    ldaps = torch.zeros(2, 24, 11, 4, 5)
    gfs = torch.zeros(2, 24, 11, 3, 3)
    ldaps[:, :, 8, 1:3, :] = 1.0
    gfs[:, :, 8] = 1.0
    with torch.no_grad():
        direct, auxiliary = model(ldaps, gfs, return_auxiliary=True)
    assert direct.shape == (2, 3, 24)
    assert auxiliary.shape == (2, 3, 4, 24)
    assert torch.all((direct >= 0.0) & (direct <= 1.02))
    assert torch.all((auxiliary[:, :, 0] >= 0.0) & (auxiliary[:, :, 0] <= 1.0))
    assert torch.all(torch.abs(auxiliary[:, :, 1:3]) <= 1.0)
    assert torch.all((auxiliary[:, :, 3] >= 0.0) & (auxiliary[:, :, 3] <= 1.0))


def test_predict_signature_has_no_scada_argument() -> None:
    signature = inspect.signature(RawSpatialGridTemporalCNNScadaMultitaskRegressor.predict)
    assert tuple(signature.parameters) == ("self", "features")


def test_multitask_loss_exact_frozen_formula() -> None:
    direct_prediction = torch.tensor([[[0.2]], [[0.4]]])
    direct_truth = torch.tensor([[[0.1]], [[0.0]]])
    direct_mask = torch.tensor([[True], [True]])
    auxiliary_prediction = torch.tensor(
        [
            [[[0.5], [0.2], [-0.1], [0.8]]],
            [[[0.1], [0.0], [0.3], [0.4]]],
        ]
    )
    auxiliary_truth = torch.zeros_like(auxiliary_prediction)
    auxiliary_mask = torch.ones_like(auxiliary_prediction, dtype=torch.bool)
    total, components = multitask_loss(
        direct_prediction,
        auxiliary_prediction,
        direct_truth,
        direct_mask,
        auxiliary_truth,
        auxiliary_mask,
    )
    element = F.smooth_l1_loss(direct_prediction, direct_truth, reduction="none", beta=0.05)
    weights = torch.tensor([[[1.0]], [[0.25]]])
    direct = torch.sum(element * weights) / torch.sum(weights)
    aux = F.smooth_l1_loss(auxiliary_prediction, auxiliary_truth, reduction="none", beta=0.05)
    ws = aux[:, :, 0].mean()
    direction = 0.5 * (aux[:, :, 1].mean() + aux[:, :, 2].mean())
    availability = aux[:, :, 3].mean()
    expected = direct + 0.10 * ws + 0.05 * direction + 0.10 * availability
    assert torch.equal(total, expected)
    assert torch.equal(components["direct"], direct)


def _targets(index: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    outputs = {}
    for group in TARGET_COLS:
        outputs[group] = pd.DataFrame(
            {
                "ws_norm": np.full(len(index), 20.0),
                "wd_sin": np.zeros(len(index)),
                "wd_cos": np.ones(len(index)),
                "availability": np.full(len(index), 0.75),
            },
            index=index,
        )
    return outputs


def test_prepare_auxiliary_transform_and_g3_mask_zero() -> None:
    index = pd.date_range("2022-01-01 01:00", periods=48, freq="h")
    targets = _targets(index)
    targets[TARGET_COLS[2]].loc[:, :] = np.nan
    expected = {
        TARGET_COLS[0]: np.ones(len(index), dtype=bool),
        TARGET_COLS[1]: np.ones(len(index), dtype=bool),
        TARGET_COLS[2]: np.zeros(len(index), dtype=bool),
    }
    values, mask, coverage = prepare_auxiliary_targets(index, targets, expected)
    assert values.shape == (48, 3, 4)
    assert np.all(values[:, 0, 0] == 0.5)
    assert not mask[:, 2].any()
    assert coverage[TARGET_COLS[2]]["expected_cells"] == 0


def test_prepare_auxiliary_rejects_target_outside_causal_mask() -> None:
    index = pd.date_range("2022-01-01 01:00", periods=24, freq="h")
    targets = _targets(index)
    expected = {group: np.ones(len(index), dtype=bool) for group in TARGET_COLS}
    expected[TARGET_COLS[2]][0] = False
    with pytest.raises(ValueError, match="outside causal expected mask"):
        prepare_auxiliary_targets(index, targets, expected)


def test_prepare_auxiliary_enforces_registered_coverage() -> None:
    index = pd.date_range("2022-01-01 01:00", periods=100, freq="h")
    targets = _targets(index)
    targets[TARGET_COLS[0]].iloc[:2, 0] = np.nan
    expected = {group: np.ones(len(index), dtype=bool) for group in TARGET_COLS}
    with pytest.raises(ValueError, match="coverage below 99%"):
        prepare_auxiliary_targets(index, targets, expected)


def test_auxiliary_channel_order_is_frozen() -> None:
    assert AUX_CHANNELS == ("ws_norm", "wd_sin", "wd_cos", "availability")

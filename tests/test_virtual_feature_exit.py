from __future__ import annotations

import numpy as np

from src.metric import group_metrics
from src.virtual_feature_exit import (
    correlation_targets,
    exact_group_row_utility,
    exit_membership,
    fixed_paired_increment,
    kept_features,
    risk_adjusted_virtual_score,
)


def test_row_utility_mean_matches_official_group_score() -> None:
    capacity = 100.0
    actual = np.array([5.0, 20.0, 40.0, 80.0])
    prediction = np.array([0.0, 25.0, 47.0, 90.0])
    utility = exact_group_row_utility(actual, prediction, capacity)
    official = group_metrics(actual, prediction, capacity)
    expected = 0.5 * official.one_minus_nmae + 0.5 * official.ficr
    assert np.isnan(utility[0])
    assert np.isclose(np.nanmean(utility), expected, atol=1e-15)


def test_declared_exit_membership_and_order() -> None:
    names = (
        "gfs__idw__ws100__run_gradient_3h",
        "gfs__global_std__heightAboveGround_100_100u",
        "ldaps__idw__surface_0_avg_lsprate",
        "cross__hub_ws_difference__run_lag_1h",
        "time__hour_sin",
    )
    assert exit_membership(names[0])["run_dynamics"]
    assert exit_membership(names[1])["spatial_redundancy"]
    assert exit_membership(names[2])["nonwind_atmospheric"]
    assert exit_membership(names[3])["cross_source_disagreement"]
    assert kept_features(names, "spatial_redundancy") == (
        names[0],
        names[2],
        names[3],
        names[4],
    )


def test_virtual_score_frozen_formula() -> None:
    result = risk_adjusted_virtual_score(
        baseline_score=0.60,
        raw_delta=0.01,
        simultaneous_lcb=0.008,
        cell_deltas=[0.02, 0.01, -0.005],
        delta_ficr=-0.002,
        delta_one_minus_nmae=-0.0003,
    )
    expected_penalty = 0.005 + 0.5 * 0.002 + 0.5 * 0.0001
    expected_increment = 0.008 - expected_penalty
    assert np.isclose(result.instability_penalty, expected_penalty)
    assert np.isclose(result.risk_adjusted_increment, expected_increment)
    assert np.isclose(result.local_virtual_score, 0.60 + expected_increment)


def test_fixed_paired_increment_and_correlation_targets() -> None:
    base = np.array([30.0, 40.0])
    exited = np.array([0.5, 0.6])
    control = np.array([0.4, 0.7])
    candidate = fixed_paired_increment(base, exited, control, 100.0, weight=0.25)
    assert np.allclose(candidate, [32.5, 37.5])
    targets = correlation_targets(
        np.array([20.0, 80.0]), np.array([25.0, 70.0]), 100.0
    )
    assert set(targets) >= {
        "signed_residual_cf",
        "within_6",
        "within_8",
        "plus_vs_minus_utility",
    }
    assert all(values.shape == (2,) for values in targets.values())

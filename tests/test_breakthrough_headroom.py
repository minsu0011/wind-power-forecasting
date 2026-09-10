from __future__ import annotations

import itertools

import numpy as np

from src.baram_breakthrough.headroom import boundary_energy_census, rowwise_oracle
from src.metric import group_metrics


def _score(actual: np.ndarray, pred: np.ndarray, capacity: float) -> float:
    result = group_metrics(actual, pred, capacity)
    return 0.5 * result.one_minus_nmae + 0.5 * result.ficr


def test_rowwise_oracle_matches_exhaustive_combinations() -> None:
    capacity = 100.0
    actual = np.array([20.0, 40.0, 80.0, np.nan])
    candidates = {
        "baseline": np.array([11.0, 47.0, 73.0, 55.0]),
        "alternative": np.array([19.0, 34.0, 90.0, 50.0]),
    }
    result = rowwise_oracle(
        actual, candidates, baseline_name="baseline", capacity_kwh=capacity
    )
    exhaustive = []
    for choice in itertools.product((0, 1), repeat=3):
        pred = candidates["baseline"].copy()
        for row, position in enumerate(choice):
            pred[row] = candidates[("baseline", "alternative")[position]][row]
        exhaustive.append(_score(actual, pred, capacity))
    assert np.isclose(
        _score(actual, result.prediction, capacity), max(exhaustive), atol=1e-15
    )


def test_oracle_cannot_be_worse_than_any_member() -> None:
    actual = np.array([15.0, 30.0, 55.0, 90.0])
    candidates = {
        "baseline": np.array([12.0, 20.0, 60.0, 70.0]),
        "a": np.array([18.0, 34.0, 40.0, 88.0]),
        "b": np.array([8.0, 29.0, 54.0, 95.0]),
    }
    result = rowwise_oracle(
        actual, candidates, baseline_name="baseline", capacity_kwh=100.0
    )
    oracle_score = _score(actual, result.prediction, 100.0)
    assert all(
        oracle_score + 1e-15 >= _score(actual, pred, 100.0)
        for pred in candidates.values()
    )


def test_boundary_energy_shares_sum_to_one() -> None:
    frame = boundary_energy_census(
        np.array([20.0, 30.0, 60.0, 90.0]),
        np.array([20.0, 23.0, 70.0, 70.0]),
        capacity_kwh=100.0,
        group="g",
    )
    assert frame["count"].sum() == 4
    assert np.isclose(frame["actual_energy_share"].sum(), 1.0, atol=1e-15)


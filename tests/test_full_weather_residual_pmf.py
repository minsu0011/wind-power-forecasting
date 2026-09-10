from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.full_weather_residual_pmf import (
    MODEL_PARAMETERS,
    STATE_FEATURE_COLUMNS,
    TOTAL_FEATURE_COUNT,
    TRANSFER_WEIGHT,
    FullWeatherResidualPMF,
    build_full_weather_state_features,
    fixed_residual_probability_matrix,
    transfer_residual_action_kwh,
)
from src.residual_histogram_bayes import (
    ACTION_DELTAS_CF,
    COMPONENT_NAMES,
    RESIDUAL_CENTRES_CF,
)


def _index(start: str, rows: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=rows, freq="h", name="forecast_kst_dtm")


def _weather(index: pd.DatetimeIndex) -> pd.DataFrame:
    values = np.linspace(0.0, 1.0, len(index) * 612, dtype=np.float32).reshape(
        len(index), 612
    )
    return pd.DataFrame(
        values,
        index=index,
        columns=[f"weather_{position:03d}" for position in range(612)],
    )


def _components(
    index: pd.DatetimeIndex, capacity: float
) -> dict[str, pd.Series]:
    return {
        name: pd.Series(
            capacity * (0.2 + 0.01 * position), index=index, name=name
        )
        for position, name in enumerate(COMPONENT_NAMES)
    }


def test_frozen_constants_exact() -> None:
    assert len(RESIDUAL_CENTRES_CF) == 43
    assert np.array_equal(
        RESIDUAL_CENTRES_CF,
        np.asarray([-0.525 + 0.025 * index for index in range(43)]),
    )
    assert len(ACTION_DELTAS_CF) == 61
    assert np.array_equal(
        ACTION_DELTAS_CF,
        np.asarray([-0.150 + 0.005 * index for index in range(61)]),
    )
    assert TRANSFER_WEIGHT == 0.10
    assert TOTAL_FEATURE_COUNT == 615
    assert MODEL_PARAMETERS["random_state"] == 42
    assert MODEL_PARAMETERS["n_estimators"] == 120
    assert MODEL_PARAMETERS["class_weight"] if "class_weight" in MODEL_PARAMETERS else None is None


def test_feature_builder_has_exact_612_plus_state3_contract() -> None:
    index = _index("2023-01-01 01:00", 5)
    capacity = 21600.0
    baseline = pd.Series(capacity * 0.3, index=index)
    frame = build_full_weather_state_features(
        _weather(index), baseline, _components(index, capacity), capacity_kwh=capacity
    )
    assert frame.shape == (5, 615)
    assert tuple(frame.columns[-3:]) == STATE_FEATURE_COLUMNS
    assert np.allclose(frame.iloc[:, -3].to_numpy(), 0.3)
    component_cf = np.asarray([0.2 + 0.01 * i for i in range(6)])
    assert np.allclose(frame.iloc[:, -2].to_numpy(), component_cf.mean())
    assert np.allclose(frame.iloc[:, -1].to_numpy(), component_cf.std(ddof=0))


def test_fixed_probability_mapping_preserves_zero_unobserved_classes() -> None:
    compact = np.asarray([[0.25, 0.75], [0.9, 0.1]], dtype=np.float64)
    fixed = fixed_residual_probability_matrix(compact, np.asarray([2, 41]))
    assert fixed.shape == (2, 43)
    assert np.array_equal(fixed[:, 2], compact[:, 0])
    assert np.array_equal(fixed[:, 41], compact[:, 1])
    assert np.count_nonzero(fixed[:, [i for i in range(43) if i not in (2, 41)]]) == 0
    assert np.array_equal(fixed.sum(axis=1), np.ones(2))


def test_transfer_is_exact_point_one_action_and_clipped() -> None:
    index = _index("2024-01-01 01:00", 3)
    baseline = pd.Series(np.asarray([0.0, 0.5, 1.02]) * 21600.0, index=index)
    action = pd.Series([-0.15, 0.15, 0.15], index=index)
    candidate = transfer_residual_action_kwh(
        baseline, action, capacity_kwh=21600.0
    )
    assert np.array_equal(
        candidate.to_numpy() / 21600.0, np.asarray([0.0, 0.515, 1.02])
    )
    assert np.max(np.abs(candidate.to_numpy() - baseline.to_numpy()) / 21600.0) <= 0.015


def test_model_smoke_fixed_probability_and_reload_ready() -> None:
    fit_index = _index("2023-01-01 01:00", 360)
    apply_index = _index("2024-01-01 01:00", 24)
    capacity = 21600.0
    fit_baseline = pd.Series(capacity * 0.45, index=fit_index)
    apply_baseline = pd.Series(capacity * 0.45, index=apply_index)
    fit_features = build_full_weather_state_features(
        _weather(fit_index),
        fit_baseline,
        _components(fit_index, capacity),
        capacity_kwh=capacity,
    )
    apply_features = build_full_weather_state_features(
        _weather(apply_index),
        apply_baseline,
        _components(apply_index, capacity),
        capacity_kwh=capacity,
    )
    residual = np.resize(np.asarray([-0.05, 0.0, 0.05]), len(fit_index))
    actual = pd.Series(capacity * (0.45 + residual), index=fit_index)
    model = FullWeatherResidualPMF()
    model.model_parameters.update(
        {"n_estimators": 2, "n_jobs": 1, "min_child_samples": 10}
    )
    model.fit(fit_features, actual, fit_baseline, capacity_kwh=capacity)
    action, probability, diagnostics = model.predict_action(
        apply_features, apply_baseline
    )
    assert probability.shape == (24, 43)
    assert np.allclose(probability.sum(axis=1), 1.0, atol=1e-12, rtol=0.0)
    assert set(np.unique(action.to_numpy())).issubset(set(ACTION_DELTAS_CF))
    assert tuple(diagnostics.columns) == (
        "primary_baseline_cf",
        "raw_action_delta_cf",
        "raw_action_cf",
        "expected_utility",
    )
    with pytest.raises(ValueError, match="overlap"):
        model.predict_action(fit_features.iloc[:4], fit_baseline.iloc[:4])


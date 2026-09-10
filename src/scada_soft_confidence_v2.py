"""Code-only primitives for the frozen Track-B v2 soft-confidence contract.

The current scientific prerequisite is false, so this module exposes only the
fixed weighting arithmetic and prerequisite decision.  It contains no reader,
estimator fit, prediction, metric, 2024, test, or submission path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from src.metric import CAPACITY_KWH


RAW_WEIGHT_FLOOR = 0.5
CONFIDENCE_COMPONENTS = (
    "availability_factor",
    "normal_fraction",
    "unknown_resource_fraction",
    "state_entropy",
)
OFFICIAL_CAPACITIES = dict(CAPACITY_KWH)


@dataclass(frozen=True)
class WeightResult:
    confidence: np.ndarray
    raw_weight: np.ndarray
    normalized_weight: np.ndarray


def continuous_soft_confidence_weights(
    availability_factor: np.ndarray,
    normal_fraction: np.ndarray,
    unknown_resource_fraction: np.ndarray,
    state_entropy: np.ndarray,
) -> WeightResult:
    """Return the single frozen all-row weight formula.

    Any non-finite proxy component receives confidence zero and therefore the
    positive floor.  No input row is filtered or conditionally omitted.
    """

    availability = np.asarray(availability_factor, dtype=np.float64)
    normal = np.asarray(normal_fraction, dtype=np.float64)
    unknown = np.asarray(unknown_resource_fraction, dtype=np.float64)
    entropy = np.asarray(state_entropy, dtype=np.float64)
    if not (availability.shape == normal.shape == unknown.shape == entropy.shape):
        raise ValueError("all confidence components must have identical shape")
    if availability.ndim != 1 or availability.size == 0:
        raise ValueError("confidence components must be non-empty one-dimensional arrays")

    finite = (
        np.isfinite(availability) & np.isfinite(normal)
        & np.isfinite(unknown) & np.isfinite(entropy)
    )
    confidence = np.zeros(availability.shape, dtype=np.float64)
    if finite.any():
        a = np.clip(availability[finite], 0.0, 1.0)
        n = np.clip(normal[finite], 0.0, 1.0)
        u = np.clip(unknown[finite], 0.0, 1.0)
        e = np.clip(entropy[finite] / np.log(7.0), 0.0, 1.0)
        confidence[finite] = a * n * (1.0 - u) * (1.0 - e)
    raw = RAW_WEIGHT_FLOOR + (1.0 - RAW_WEIGHT_FLOOR) * confidence
    mean = float(np.mean(raw, dtype=np.float64))
    if not np.isfinite(mean) or mean <= 0:
        raise AssertionError("positive finite raw-weight mean required")
    normalized = raw / mean
    if normalized.shape != availability.shape:
        raise AssertionError("weighting changed eligible row count")
    if not np.all(np.isfinite(normalized)) or not np.all(normalized > 0):
        raise AssertionError("every eligible row must retain positive finite weight")
    return WeightResult(confidence=confidence, raw_weight=raw, normalized_weight=normalized)


def scientific_prerequisite_decision(state_gate_by_group: Mapping[str, bool]) -> str:
    """Require an independently valid proxy for all three groups."""

    expected = {"G1", "G2", "G3"}
    if set(state_gate_by_group) != expected:
        raise ValueError("state gate must contain exactly G1, G2 and G3")
    if not all(type(state_gate_by_group[group]) is bool for group in sorted(expected)):
        raise TypeError("state gate values must be booleans")
    if not all(state_gate_by_group.values()):
        return "TRACK_B_V2_STOP_INVALID_PROXY_PREREQUISITE"
    return "TRACK_B_V2_AWAIT_INDEPENDENT_PRELAUNCH_GO"


__all__ = [
    "CONFIDENCE_COMPONENTS",
    "OFFICIAL_CAPACITIES",
    "RAW_WEIGHT_FLOOR",
    "WeightResult",
    "continuous_soft_confidence_weights",
    "scientific_prerequisite_decision",
]

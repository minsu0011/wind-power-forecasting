"""Frozen primitives for the Track-B SCADA latent-state falsification pilot.

The states produced here are pattern labels, not operator-confirmed events.  In
particular, the supplied SCADA has no alarm, status, pitch, rotor-speed or
power-limit fields.  This module deliberately contains no file reader, model
selection loop, 2024 path or submission writer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


STATES: tuple[str, ...] = (
    "NORMAL",
    "PARTIAL_DERATE",
    "CURTAILMENT_LIKE",
    "OUTAGE_LIKE",
    "LOW_RESOURCE",
    "SENSOR_OR_DATA_ANOMALY",
    "UNKNOWN",
)


@dataclass(frozen=True)
class UpperEnvelopeCurve:
    centers: np.ndarray
    values: np.ndarray
    populated_bins: int
    fit_rows: int

    def predict(self, wind_ms: Iterable[float]) -> np.ndarray:
        wind = np.asarray(wind_ms, dtype=np.float64)
        out = np.full(wind.shape, np.nan, dtype=np.float64)
        valid = np.isfinite(wind)
        out[valid] = np.interp(
            wind[valid], self.centers, self.values,
            left=float(self.values[0]), right=float(self.values[-1]),
        )
        return np.clip(out, 0.0, 1.05)


def pava_nondecreasing(values: Iterable[float]) -> np.ndarray:
    """Equal-weight pool-adjacent-violators fit with deterministic ties."""

    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 1 or y.size == 0 or not np.all(np.isfinite(y)):
        raise ValueError("values must be a non-empty finite one-dimensional array")
    means: list[float] = []
    weights: list[int] = []
    starts: list[int] = []
    for index, value in enumerate(y):
        means.append(float(value))
        weights.append(1)
        starts.append(index)
        while len(means) >= 2 and means[-2] > means[-1]:
            weight = weights[-2] + weights[-1]
            mean = (means[-2] * weights[-2] + means[-1] * weights[-1]) / weight
            means[-2:] = [mean]
            weights[-2:] = [weight]
            starts.pop()
    fitted = np.empty_like(y)
    for block, start in enumerate(starts):
        end = starts[block + 1] if block + 1 < len(starts) else y.size
        fitted[start:end] = means[block]
    return fitted


def fit_upper_envelope(
    wind_ms: Iterable[float],
    power_cf: Iterable[float],
    *,
    bin_width: float = 0.5,
    minimum_bin_rows: int = 30,
    quantile: float = 0.90,
) -> UpperEnvelopeCurve:
    """Fit the preregistered binned-q90 monotone turbine potential curve."""

    wind = np.asarray(wind_ms, dtype=np.float64)
    power = np.asarray(power_cf, dtype=np.float64)
    if wind.shape != power.shape or wind.ndim != 1:
        raise ValueError("wind and power must be one-dimensional with equal shape")
    if bin_width <= 0 or minimum_bin_rows <= 0 or not 0 < quantile < 1:
        raise ValueError("invalid curve parameters")
    valid = (
        np.isfinite(wind) & np.isfinite(power)
        & (wind >= 0.0) & (wind <= 40.0)
        & (power >= 0.0) & (power <= 1.05)
    )
    wind = wind[valid]
    power = power[valid]
    bin_id = np.floor(wind / bin_width).astype(np.int32)
    centers: list[float] = []
    values: list[float] = []
    for identifier in np.unique(bin_id):
        selected = power[bin_id == identifier]
        if selected.size < minimum_bin_rows:
            continue
        centers.append((float(identifier) + 0.5) * bin_width)
        values.append(float(np.quantile(selected, quantile, method="linear")))
    if len(centers) < 2:
        raise ValueError("too few populated bins for a potential curve")
    fitted = np.clip(pava_nondecreasing(values), 0.0, 1.05)
    return UpperEnvelopeCurve(
        centers=np.asarray(centers, dtype=np.float64),
        values=fitted,
        populated_bins=len(centers),
        fit_rows=int(valid.sum()),
    )


def leave_one_out_peer_median(values: np.ndarray) -> np.ndarray:
    """Return contemporaneous turbine peer medians, excluding each turbine."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError("values must have shape (timestamps, at least two turbines)")
    out = np.full(array.shape, np.nan, dtype=np.float64)
    for turbine in range(array.shape[1]):
        peers = np.delete(array, turbine, axis=1)
        with np.errstate(all="ignore"):
            out[:, turbine] = np.nanmedian(peers, axis=1)
    return out


def initial_normal_mask(
    ratio: np.ndarray,
    peer_median: np.ndarray,
    potential_cf: np.ndarray,
    wind_ms: np.ndarray,
) -> np.ndarray:
    return (
        np.isfinite(ratio) & np.isfinite(peer_median) & np.isfinite(potential_cf)
        & np.isfinite(wind_ms) & (ratio >= 0.80) & (peer_median >= 0.75)
        & (potential_cf >= 0.10) & (wind_ms >= 3.0) & (wind_ms <= 20.0)
    )


def assign_proxy_states(
    power_cf: np.ndarray,
    wind_ms: np.ndarray,
    potential_cf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the frozen ordered state rules to timestamp-by-turbine matrices."""

    power = np.asarray(power_cf, dtype=np.float64)
    wind = np.asarray(wind_ms, dtype=np.float64)
    potential = np.asarray(potential_cf, dtype=np.float64)
    if power.shape != wind.shape or power.shape != potential.shape or power.ndim != 2:
        raise ValueError("power, wind and potential must share a two-dimensional shape")
    valid = (
        np.isfinite(power) & np.isfinite(wind) & np.isfinite(potential)
        & (power >= 0.0) & (power <= 1.05) & (wind >= 0.0) & (wind <= 40.0)
    )
    eligible = valid & (wind >= 3.0) & (wind <= 20.0) & (potential >= 0.10)
    ratio = np.full(power.shape, np.nan, dtype=np.float64)
    ratio[eligible] = power[eligible] / potential[eligible]
    peer = leave_one_out_peer_median(ratio)
    with np.errstate(all="ignore"):
        group_median = np.nanmedian(ratio, axis=1)
    eligible_count = eligible.sum(axis=1)
    low_fraction = np.divide(
        ((ratio < 0.70) & eligible).sum(axis=1),
        eligible_count,
        out=np.zeros(power.shape[0], dtype=np.float64),
        where=eligible_count > 0,
    )
    states = np.full(power.shape, "UNKNOWN", dtype=object)
    anomaly = ~valid
    states[anomaly] = "SENSOR_OR_DATA_ANOMALY"
    low_resource = valid & ((wind < 3.0) | (potential < 0.10))
    states[low_resource] = "LOW_RESOURCE"
    supported = eligible & (eligible_count[:, None] >= 3)
    outage = supported & (ratio <= 0.10) & (peer >= 0.75)
    states[outage] = "OUTAGE_LIKE"
    common_low = (
        supported & (group_median[:, None] <= 0.60)
        & (low_fraction[:, None] >= (2.0 / 3.0)) & (ratio < 0.75)
        & ~outage
    )
    states[common_low] = "CURTAILMENT_LIKE"
    normal = supported & (ratio >= 0.80) & (peer >= 0.75) & ~outage & ~common_low
    states[normal] = "NORMAL"
    partial = (
        supported & (ratio > 0.10) & (ratio < 0.80)
        & ~outage & ~common_low & ~normal
    )
    states[partial] = "PARTIAL_DERATE"
    return states, ratio, peer


def jensen_shannon(proportions_a: Iterable[float], proportions_b: Iterable[float]) -> float:
    """Natural-log Jensen-Shannon divergence with deterministic zero handling."""

    a = np.asarray(proportions_a, dtype=np.float64)
    b = np.asarray(proportions_b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or np.any(a < 0) or np.any(b < 0):
        raise ValueError("proportions must be non-negative one-dimensional equal shapes")
    if a.sum() <= 0 or b.sum() <= 0:
        raise ValueError("each distribution must have positive mass")
    a = a / a.sum()
    b = b / b.sum()
    midpoint = 0.5 * (a + b)

    def kl(left: np.ndarray, right: np.ndarray) -> float:
        mask = left > 0
        return float(np.sum(left[mask] * np.log(left[mask] / right[mask])))

    return 0.5 * kl(a, midpoint) + 0.5 * kl(b, midpoint)


def state_entropy(counts: Iterable[int]) -> float:
    array = np.asarray(counts, dtype=np.float64)
    if array.ndim != 1 or np.any(array < 0):
        raise ValueError("counts must be a non-negative vector")
    if array.sum() == 0:
        return float("nan")
    proportions = array / array.sum()
    positive = proportions > 0
    return float(-np.sum(proportions[positive] * np.log(proportions[positive])))


__all__ = [
    "STATES",
    "UpperEnvelopeCurve",
    "assign_proxy_states",
    "fit_upper_envelope",
    "initial_normal_mask",
    "jensen_shannon",
    "leave_one_out_peer_median",
    "pava_nondecreasing",
    "state_entropy",
]

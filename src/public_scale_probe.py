"""Auditable helpers for Public-adaptive multiplicative scale probes.

These helpers are deliberately small.  They do not fit a forecasting model and
they do not turn leaderboard feedback into an independent validation result.
Their only purposes are to (1) apply one fixed scalar to an existing prediction
frame and (2) reproduce a diagnostic comparison with the official metric.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .metric import CAPACITY_KWH, TARGET_COLS, score_details


def validate_public_triplet(
    *, score: float, one_minus_nmae: float, ficr: float, atol: float = 5e-10
) -> None:
    """Reject malformed leaderboard triplets before using them diagnostically."""

    values = np.asarray([score, one_minus_nmae, ficr], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Public metric triplet must be finite")
    expected = 0.5 * (float(one_minus_nmae) + float(ficr))
    if not np.isclose(float(score), expected, rtol=0.0, atol=float(atol)):
        raise ValueError(
            "Public score is inconsistent with 0.5*(1-NMAE+FICR): "
            f"{score} != {expected}"
        )


def validate_prediction_frame(frame: pd.DataFrame, *, name: str) -> None:
    """Validate the numeric prediction contract shared by probe artifacts."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if tuple(frame.columns) != TARGET_COLS:
        raise ValueError(
            f"{name} columns must be exactly {TARGET_COLS}; got {tuple(frame.columns)}"
        )
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f"{name} must use a DatetimeIndex")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be unique and increasing")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError(f"{name} contains NaN or infinite values")


def scale_predictions(
    prediction: pd.DataFrame,
    scale: float,
    *,
    upper_capacity_fraction: float = 1.02,
) -> pd.DataFrame:
    """Multiply every group by one fixed scale and apply physical bounds."""

    validate_prediction_frame(prediction, name="prediction")
    multiplier = float(scale)
    upper = float(upper_capacity_fraction)
    if not np.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("scale must be finite and positive")
    if not np.isfinite(upper) or upper <= 0.0:
        raise ValueError("upper_capacity_fraction must be finite and positive")

    output = prediction.astype(float).copy(deep=True)
    for group in TARGET_COLS:
        output[group] = np.clip(
            prediction[group].to_numpy(dtype=float) * multiplier,
            0.0,
            upper * CAPACITY_KWH[group],
        )
    validate_prediction_frame(output, name="scaled prediction")
    return output


def fit_oof_scale_to_public(
    labels: pd.DataFrame,
    oof_prediction: pd.DataFrame,
    public_metrics: Mapping[str, float],
    *,
    scale_min: float,
    scale_max: float,
    scale_step: float,
) -> dict[str, Any]:
    """Find a deterministic grid scale matching Public NMAE/FICR components.

    The result is a diagnostic analogy, not an estimate with causal or Private
    leaderboard validity.  The objective is the Euclidean distance between the
    two official component metrics; total score is redundant.
    """

    validate_prediction_frame(oof_prediction, name="oof_prediction")
    if not labels.index.equals(oof_prediction.index):
        raise ValueError("labels and OOF prediction indices differ")
    validate_public_triplet(
        score=float(public_metrics["score"]),
        one_minus_nmae=float(public_metrics["one_minus_nmae"]),
        ficr=float(public_metrics["ficr"]),
    )
    lower = float(scale_min)
    upper = float(scale_max)
    step = float(scale_step)
    if not all(np.isfinite([lower, upper, step])) or step <= 0.0 or upper < lower:
        raise ValueError("invalid scale grid")
    intervals = int(round((upper - lower) / step))
    if not np.isclose(lower + intervals * step, upper, rtol=0.0, atol=1e-12):
        raise ValueError("scale grid endpoints are not divisible by scale_step")
    scales = lower + step * np.arange(intervals + 1, dtype=float)

    target = np.asarray(
        [public_metrics["one_minus_nmae"], public_metrics["ficr"]], dtype=float
    )
    rows: list[dict[str, float]] = []
    for scale in scales:
        metrics = score_details(labels, oof_prediction * scale)
        residual = np.asarray(
            [metrics.one_minus_nmae, metrics.ficr], dtype=float
        ) - target
        rows.append(
            {
                "scale": float(scale),
                "score": float(metrics.total_score),
                "one_minus_nmae": float(metrics.one_minus_nmae),
                "ficr": float(metrics.ficr),
                "one_minus_nmae_residual": float(residual[0]),
                "ficr_residual": float(residual[1]),
                "component_l2": float(np.linalg.norm(residual)),
            }
        )
    joint = min(rows, key=lambda row: (row["component_l2"], row["scale"]))
    nmae_match = min(
        rows,
        key=lambda row: (abs(row["one_minus_nmae_residual"]), row["scale"]),
    )
    ficr_match = min(
        rows, key=lambda row: (abs(row["ficr_residual"]), row["scale"])
    )
    unscaled = score_details(labels, oof_prediction)
    return {
        "objective": "L2([OOF_1-NMAE, OOF_FICR] - [Public_1-NMAE, Public_FICR])",
        "grid": {
            "minimum": lower,
            "maximum": upper,
            "step": step,
            "count": int(len(scales)),
        },
        "public": {
            "score": float(public_metrics["score"]),
            "one_minus_nmae": float(public_metrics["one_minus_nmae"]),
            "ficr": float(public_metrics["ficr"]),
        },
        "oof_unscaled": {
            "score": float(unscaled.total_score),
            "one_minus_nmae": float(unscaled.one_minus_nmae),
            "ficr": float(unscaled.ficr),
        },
        "joint_fit": joint,
        "one_minus_nmae_nearest": nmae_match,
        "ficr_nearest": ficr_match,
        "joint_fit_reciprocal": float(1.0 / joint["scale"]),
        "interpretation_warning": (
            "This is a Public-adaptive curve analogy on consumed 2024 OOF, not "
            "evidence that a reciprocal scale will transfer to Private labels."
        ),
    }


__all__ = [
    "fit_oof_scale_to_public",
    "scale_predictions",
    "validate_prediction_frame",
    "validate_public_triplet",
]

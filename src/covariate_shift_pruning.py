"""Label-free covariate-shift feature pruning primitives.

The selector intentionally changes only the ordered set of columns.  Values and
row weights passed to the downstream estimator remain untouched.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


FORBIDDEN_PREFIXES = ("time__", "site__")
FORBIDDEN_SUBSTRINGS = (
    "target",
    "label",
    "actual",
    "scada",
    "prediction",
    "pred__",
    "public",
    "leaderboard",
    "forecast_id",
    "kst_dtm",
    "year",
)
QUANTILE_PROBABILITIES = np.linspace(0.0, 1.0, 257, dtype=np.float64)
MIN_POOLED_STD = 1.0e-12


def eligible_feature_names(columns: Iterable[str]) -> tuple[str, ...]:
    """Return cached weather names allowed by the frozen name-only rule."""

    result: list[str] = []
    seen: set[str] = set()
    for value in columns:
        if not isinstance(value, str):
            raise TypeError("all cached feature names must be strings")
        if value in seen:
            raise ValueError(f"duplicate cached feature name: {value}")
        seen.add(value)
        lowered = value.lower()
        if lowered.startswith(FORBIDDEN_PREFIXES):
            continue
        if any(token in lowered for token in FORBIDDEN_SUBSTRINGS):
            continue
        result.append(value)
    return tuple(result)


def standardized_quantile_wasserstein(
    source: pd.DataFrame,
    application: pd.DataFrame,
    *,
    eligible_names: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Compute the frozen covariate-only shift statistic for every eligible column.

    The returned table is in original cached-column order.  Degenerate features
    remain in the audit table but have ``eligible_for_ranking=False``.
    """

    if source.empty or application.empty:
        raise ValueError("source and application covariates must both be non-empty")
    if tuple(source.columns) != tuple(application.columns):
        raise ValueError("source/application cached feature schemas differ")
    if source.index.intersection(application.index).size:
        raise ValueError("source and application covariate rows overlap")
    if source.index.max() >= application.index.min():
        raise ValueError("strict-forward order violated")

    allowed = (
        eligible_feature_names(source.columns)
        if eligible_names is None
        else tuple(eligible_names)
    )
    if len(set(allowed)) != len(allowed):
        raise ValueError("eligible feature names are not unique")
    missing = [name for name in allowed if name not in source.columns]
    if missing:
        raise KeyError(f"eligible features missing from cached schema: {missing[:3]}")

    positions = {name: position for position, name in enumerate(source.columns)}
    records: list[dict[str, object]] = []
    for name in allowed:
        source_values = source[name].to_numpy(dtype=np.float64, copy=False)
        application_values = application[name].to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(source_values).all() or not np.isfinite(application_values).all():
            raise ValueError(f"non-finite covariate values for feature {name}")
        pooled = np.concatenate((source_values, application_values))
        pooled_std = float(np.std(pooled, ddof=0))
        nondegenerate = bool(np.isfinite(pooled_std) and pooled_std > MIN_POOLED_STD)
        if nondegenerate:
            source_quantiles = np.quantile(
                source_values, QUANTILE_PROBABILITIES, method="linear"
            )
            application_quantiles = np.quantile(
                application_values, QUANTILE_PROBABILITIES, method="linear"
            )
            numerator = float(
                np.trapezoid(
                    np.abs(source_quantiles - application_quantiles),
                    QUANTILE_PROBABILITIES,
                )
            )
            score = float(numerator / pooled_std)
        else:
            numerator = 0.0
            score = float("nan")
        records.append(
            {
                "feature": name,
                "original_position": int(positions[name]),
                "source_rows": int(len(source_values)),
                "application_rows": int(len(application_values)),
                "pooled_std": pooled_std,
                "wasserstein_numerator": numerator,
                "shift_score": score,
                "eligible_for_ranking": nondegenerate,
            }
        )
    return pd.DataFrame.from_records(records)


def select_stable_features(
    shift_table: pd.DataFrame,
    keep_fraction: float,
) -> tuple[str, ...]:
    """Select the least shifted fraction, returning cached-column order."""

    if not (0.0 < float(keep_fraction) <= 1.0):
        raise ValueError("keep_fraction must be in (0,1]")
    required = {
        "feature",
        "original_position",
        "shift_score",
        "eligible_for_ranking",
    }
    if not required.issubset(shift_table.columns):
        raise KeyError("shift table schema changed")
    eligible = shift_table.loc[shift_table["eligible_for_ranking"].astype(bool)].copy()
    if eligible.empty:
        raise ValueError("no nondegenerate eligible weather features")
    if not np.isfinite(eligible["shift_score"].to_numpy(dtype=float)).all():
        raise ValueError("eligible shift score is non-finite")
    keep_count = max(1, math.floor(float(keep_fraction) * len(eligible)))
    ranked = eligible.sort_values(
        ["shift_score", "original_position"], kind="stable"
    ).iloc[:keep_count]
    retained = ranked.sort_values("original_position", kind="stable")["feature"]
    result = tuple(retained.astype(str))
    if len(result) != keep_count or len(set(result)) != keep_count:
        raise AssertionError("selected feature set is not unique and complete")
    return result


def blend_candidate_kwh(
    baseline_kwh: np.ndarray | pd.Series,
    raw_model_cf: np.ndarray | pd.Series,
    *,
    weight: float,
    capacity_kwh: float,
) -> np.ndarray:
    """Apply the sole registered candidate formula."""

    if not (0.0 <= float(weight) <= 1.0):
        raise ValueError("weight must be in [0,1]")
    baseline = np.asarray(baseline_kwh, dtype=np.float64)
    model_cf = np.asarray(raw_model_cf, dtype=np.float64)
    if baseline.shape != model_cf.shape:
        raise ValueError("baseline and model predictions are misaligned")
    if not np.isfinite(baseline).all() or not np.isfinite(model_cf).all():
        raise ValueError("candidate inputs must be finite")
    raw_kwh = np.clip(model_cf, 0.0, 1.02) * float(capacity_kwh)
    return np.clip(
        (1.0 - float(weight)) * baseline + float(weight) * raw_kwh,
        0.0,
        1.02 * float(capacity_kwh),
    )


__all__ = [
    "FORBIDDEN_PREFIXES",
    "FORBIDDEN_SUBSTRINGS",
    "MIN_POOLED_STD",
    "QUANTILE_PROBABILITIES",
    "blend_candidate_kwh",
    "eligible_feature_names",
    "select_stable_features",
    "standardized_quantile_wasserstein",
]

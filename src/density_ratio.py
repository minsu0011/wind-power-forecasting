"""Strict, label-free density-ratio weighting helpers.

The domain classifier sees weather covariates only.  Its source probabilities
are out-of-fold at the 24-hour issuance-run level, so an operating run can
never be split between the classifier's fit and held-out partitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


@dataclass(frozen=True)
class DensityRatioResult:
    source_weight: pd.Series
    source_probability: pd.Series
    application_probability: pd.Series
    audit: dict[str, Any]


def issuance_run_key(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return the operating-day key for hourly runs spanning 01:00..00:00."""

    values = pd.DatetimeIndex(index)
    if not values.is_unique or not values.is_monotonic_increasing:
        raise ValueError("weather index must be unique and increasing")
    return (values - pd.Timedelta(hours=1)).normalize()


def run_fold_ids(index: pd.DatetimeIndex, *, fold_count: int = 5) -> np.ndarray:
    if fold_count < 2:
        raise ValueError("fold_count must be at least two")
    keys = issuance_run_key(index)
    epoch = pd.Timestamp("1970-01-01")
    days = ((keys - epoch) / pd.Timedelta(days=1)).astype(np.int64)
    folds = np.asarray(days % int(fold_count), dtype=np.int8)
    for key in keys.unique():
        observed = np.unique(folds[keys == key])
        if len(observed) != 1:
            raise AssertionError("one issuance run crossed domain folds")
    return folds


def _pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("standard", StandardScaler()),
            (
                "quadratic",
                PolynomialFeatures(degree=2, include_bias=False),
            ),
            (
                "logistic",
                LogisticRegression(
                    C=0.1,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def _quantiles(values: np.ndarray) -> dict[str, float]:
    probs = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
    observed = np.quantile(values, probs)
    return {f"q{int(prob * 100):02d}": float(value) for prob, value in zip(probs, observed)}


def estimate_density_ratio_weights(
    source: pd.DataFrame,
    application: pd.DataFrame,
    *,
    eligible_source: pd.Series,
    feature_names: Sequence[str],
    fold_count: int = 5,
) -> DensityRatioResult:
    """Estimate tempered source weights from run-level OOF domain probabilities."""

    ordered = list(map(str, feature_names))
    if len(ordered) != 15 or len(set(ordered)) != len(ordered):
        raise ValueError("exactly 15 unique ordered domain features are required")
    for name, frame in (("source", source), ("application", application)):
        missing = [column for column in ordered if column not in frame.columns]
        if missing:
            raise ValueError(f"{name} lacks domain features: {missing}")
        values = frame.loc[:, ordered].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{name} domain features must be finite")
    eligible = eligible_source.reindex(source.index)
    if eligible.isna().any():
        raise ValueError("eligible_source index differs from source")
    eligible_values = eligible.to_numpy(dtype=bool)
    if eligible_values.sum() < 100:
        raise ValueError("too few eligible source rows")

    source_x = source.loc[:, ordered].to_numpy(dtype=np.float64)
    application_x = application.loc[:, ordered].to_numpy(dtype=np.float64)
    source_folds = run_fold_ids(source.index, fold_count=fold_count)
    application_folds = run_fold_ids(application.index, fold_count=fold_count)
    source_probability = np.full(len(source), np.nan, dtype=np.float64)
    application_probability = np.full(len(application), np.nan, dtype=np.float64)
    fold_records: list[dict[str, Any]] = []

    source_keys = issuance_run_key(source.index)
    application_keys = issuance_run_key(application.index)
    for fold in range(fold_count):
        source_hold = source_folds == fold
        application_hold = application_folds == fold
        train_x = np.vstack((source_x[~source_hold], application_x[~application_hold]))
        train_y = np.concatenate(
            (
                np.zeros((~source_hold).sum(), dtype=np.int8),
                np.ones((~application_hold).sum(), dtype=np.int8),
            )
        )
        model = _pipeline()
        model.fit(train_x, train_y)
        source_probability[source_hold] = model.predict_proba(source_x[source_hold])[:, 1]
        application_probability[application_hold] = model.predict_proba(
            application_x[application_hold]
        )[:, 1]

        held_keys = set(source_keys[source_hold]) | set(application_keys[application_hold])
        fit_keys = set(source_keys[~source_hold]) | set(application_keys[~application_hold])
        overlap = held_keys.intersection(fit_keys)
        if overlap:
            raise AssertionError("domain fit and holdout issuance runs overlap")
        fold_records.append(
            {
                "fold": fold,
                "source_fit_rows": int((~source_hold).sum()),
                "source_holdout_rows": int(source_hold.sum()),
                "application_fit_rows": int((~application_hold).sum()),
                "application_holdout_rows": int(application_hold.sum()),
                "fit_holdout_run_overlap": 0,
            }
        )

    if not np.isfinite(source_probability).all() or not np.isfinite(
        application_probability
    ).all():
        raise AssertionError("domain OOF probabilities are incomplete")
    clipped_probability = np.clip(source_probability, 0.05, 0.95)
    ratio = clipped_probability / (1.0 - clipped_probability)
    tempered = np.sqrt(np.clip(ratio, 0.25, 4.0))
    normalization = float(tempered[eligible_values].mean())
    if not np.isfinite(normalization) or normalization <= 0:
        raise AssertionError("invalid density-weight normalization")
    weights = tempered / normalization
    eligible_weights = weights[eligible_values]
    ess = float(eligible_weights.sum() ** 2 / np.square(eligible_weights).sum())
    domain_truth = np.concatenate(
        (np.zeros(len(source), dtype=np.int8), np.ones(len(application), dtype=np.int8))
    )
    domain_probability = np.concatenate((source_probability, application_probability))
    audit = {
        "feature_count": len(ordered),
        "fold_count": fold_count,
        "run_fold_overlap": 0,
        "folds": fold_records,
        "domain_oof_auc": float(roc_auc_score(domain_truth, domain_probability)),
        "source_probability_quantiles": _quantiles(source_probability),
        "application_probability_quantiles": _quantiles(application_probability),
        "eligible_weight_count": int(len(eligible_weights)),
        "eligible_weight_mean": float(eligible_weights.mean()),
        "eligible_weight_min": float(eligible_weights.min()),
        "eligible_weight_max": float(eligible_weights.max()),
        "eligible_weight_quantiles": _quantiles(eligible_weights),
        "eligible_weight_ess": ess,
        "eligible_weight_ess_fraction": float(ess / len(eligible_weights)),
    }
    return DensityRatioResult(
        source_weight=pd.Series(weights, index=source.index, name="density_weight"),
        source_probability=pd.Series(
            source_probability, index=source.index, name="domain_probability"
        ),
        application_probability=pd.Series(
            application_probability,
            index=application.index,
            name="domain_probability",
        ),
        audit=audit,
    )


def assemble_locked_group(
    components: Mapping[str, pd.Series | np.ndarray],
    *,
    group: str,
    capacity_kwh: float,
    ensemble: Mapping[str, Any],
) -> np.ndarray:
    """Apply locked ensemble -> affine -> clip -> power-bin in exact order."""

    weights = ensemble["weights"][group]
    expected = tuple(weights)
    if set(components) != set(expected):
        raise ValueError("component names differ from locked ensemble")
    arrays = {name: np.asarray(components[name], dtype=np.float64) for name in expected}
    lengths = {len(values) for values in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("component lengths differ")
    prediction = sum(float(weights[name]) * arrays[name] for name in expected)
    affine = ensemble["affine"][group]
    prediction = (
        float(affine["scale"]) * prediction + float(affine["bias_kwh"])
    )
    clip_spec = ensemble["clip"][group]
    lower = float(clip_spec["lower_capacity_fraction"]) * capacity_kwh
    upper = float(clip_spec["upper_capacity_fraction"]) * capacity_kwh
    prediction = np.clip(prediction, lower, upper)
    bin_spec = ensemble["power_bins"][group]
    if bin_spec is not None:
        edges = np.asarray(bin_spec["edges_cf"], dtype=np.float64)
        deltas = np.asarray(bin_spec["delta_kwh"], dtype=np.float64)
        positions = np.searchsorted(
            edges[1:-1], prediction / capacity_kwh, side="right"
        )
        prediction = np.clip(prediction + deltas[positions], lower, upper)
    return prediction


def half_delta_candidate(
    baseline_kwh: Sequence[float],
    weighted_recipe_kwh: Sequence[float],
    *,
    capacity_kwh: float,
) -> np.ndarray:
    baseline = np.asarray(baseline_kwh, dtype=np.float64)
    weighted = np.asarray(weighted_recipe_kwh, dtype=np.float64)
    if baseline.shape != weighted.shape:
        raise ValueError("baseline and weighted recipe shapes differ")
    return np.clip(0.5 * baseline + 0.5 * weighted, 0.0, 1.02 * capacity_kwh)


__all__ = [
    "DensityRatioResult",
    "assemble_locked_group",
    "estimate_density_ratio_weights",
    "half_delta_candidate",
    "issuance_run_key",
    "run_fold_ids",
]

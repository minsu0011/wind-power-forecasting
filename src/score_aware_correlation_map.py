"""Deterministic score-aware correlation maps for causal OOF feature rows.

This module is intentionally downstream of model fitting.  It never reads a
label file, trains a model, or estimates a leaderboard score.  Instead it
accepts already-created causal out-of-fold (OOF) rows and describes how each
weather feature relates to the frozen baseline's residual and settlement
geometry.

The map has three complementary parts:

* continuous Spearman and Gaussian-rank Pearson correlations;
* group-balanced, energy-weighted AUC at the official 6% and 8% boundaries;
* unsupervised redundancy components at ``abs(Spearman) >= 0.92`` on causal
  train/apply feature sources.

All target construction delegates to :func:`correlation_targets` and the exact
row-utility helper in :mod:`src.virtual_feature_exit`.  Group/fold sign
stability is reported so a large pooled correlation cannot hide a reversal in
one causal validation block.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from scipy.special import ndtri
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from src.virtual_feature_exit import (
    PHYSICAL_FAMILIES,
    build_feature_taxonomy,
    correlation_targets,
)


CONTINUOUS_TARGETS: tuple[str, ...] = (
    "actual_cf",
    "signed_residual_cf",
    "absolute_error_cf",
    "margin_6",
    "margin_8",
)

BINARY_TARGETS: tuple[str, ...] = (
    "eligible",
    "within_6",
    "within_8",
)

UTILITY_TARGETS: tuple[str, ...] = (
    "plus_vs_minus_utility",
    "best_one_percent_utility_gain",
)

TARGET_ORDER: tuple[str, ...] = (
    "eligible",
    "actual_cf",
    "signed_residual_cf",
    "absolute_error_cf",
    "within_6",
    "within_8",
    "margin_6",
    "margin_8",
    "plus_vs_minus_utility",
    "best_one_percent_utility_gain",
)

TARGET_KIND: dict[str, str] = {
    **{name: "continuous" for name in CONTINUOUS_TARGETS},
    **{name: "binary_boundary" for name in BINARY_TARGETS},
    **{name: "exact_utility" for name in UTILITY_TARGETS},
}


@dataclass(frozen=True)
class CorrelationMapConfig:
    """Frozen resource and statistical settings for a correlation-map run."""

    group_col: str = "group"
    fold_col: str = "fold"
    actual_col: str = "actual_kwh"
    prediction_col: str = "baseline_kwh"
    capacity_col: str = "capacity_kwh"
    action_fraction: float = 0.01
    redundancy_threshold: float = 0.92
    min_pair_rows: int = 30
    min_slice_rows: int = 30
    max_redundancy_rows_per_source: int = 50_000
    n_jobs: int = 7
    stability_epsilon: float = 1e-12
    lead_bin_edges: tuple[float, ...] = (6.0, 12.0, 18.0, 24.0)
    control_cf_bin_edges: tuple[float, ...] = (0.25, 0.50, 0.75)

    def __post_init__(self) -> None:
        if not 0.0 < float(self.action_fraction) <= 0.10:
            raise ValueError("action_fraction must be in (0, 0.10]")
        if not 0.0 < float(self.redundancy_threshold) <= 1.0:
            raise ValueError("redundancy_threshold must be in (0, 1]")
        if int(self.min_pair_rows) < 3:
            raise ValueError("min_pair_rows must be at least 3")
        if int(self.min_slice_rows) < 3:
            raise ValueError("min_slice_rows must be at least 3")
        if int(self.max_redundancy_rows_per_source) < 3:
            raise ValueError(
                "max_redundancy_rows_per_source must be at least 3"
            )
        if not 1 <= int(self.n_jobs) <= 7:
            raise ValueError("n_jobs must be between 1 and 7")
        if float(self.stability_epsilon) < 0.0:
            raise ValueError("stability_epsilon must be non-negative")
        if tuple(sorted(self.lead_bin_edges)) != self.lead_bin_edges:
            raise ValueError("lead_bin_edges must be sorted")
        if tuple(sorted(self.control_cf_bin_edges)) != self.control_cf_bin_edges:
            raise ValueError("control_cf_bin_edges must be sorted")


@dataclass
class CorrelationMapResult:
    """In-memory score-aware map and deterministic run metadata."""

    feature_map: pd.DataFrame
    family_map: pd.DataFrame
    redundancy_membership: pd.DataFrame
    redundancy_edges: pd.DataFrame
    metadata: dict[str, Any]


def _require_numeric_features(frame: pd.DataFrame, *, name: str) -> tuple[str, ...]:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if frame.empty:
        raise ValueError(f"{name} must contain at least one row")
    if frame.columns.has_duplicates:
        raise ValueError(f"{name} feature columns must be unique")
    columns = tuple(str(value) for value in frame.columns)
    if len(set(columns)) != len(columns):
        raise ValueError(f"{name} feature names collide after string conversion")
    non_numeric = [
        str(column)
        for column in frame.columns
        if not pd.api.types.is_numeric_dtype(frame[column].dtype)
    ]
    if non_numeric:
        preview = ", ".join(repr(value) for value in non_numeric[:5])
        raise ValueError(f"{name} has non-numeric feature columns: {preview}")
    return columns


def _validate_oof_inputs(
    oof_features: pd.DataFrame,
    oof_rows: pd.DataFrame,
    config: CorrelationMapConfig,
) -> tuple[str, ...]:
    feature_names = _require_numeric_features(oof_features, name="oof_features")
    if not isinstance(oof_rows, pd.DataFrame):
        raise TypeError("oof_rows must be a pandas DataFrame")
    if len(oof_features) != len(oof_rows):
        raise ValueError("oof_features and oof_rows must have the same row count")
    if not oof_features.index.equals(oof_rows.index):
        raise ValueError(
            "oof_features and oof_rows indices must be exactly aligned"
        )
    required = (
        config.group_col,
        config.fold_col,
        config.actual_col,
        config.prediction_col,
        config.capacity_col,
    )
    missing = [column for column in required if column not in oof_rows.columns]
    if missing:
        raise ValueError(f"oof_rows is missing required columns: {missing}")
    if oof_rows[config.group_col].isna().any():
        raise ValueError("oof_rows group values must be non-null")
    if oof_rows[config.fold_col].isna().any():
        raise ValueError("oof_rows fold values must be non-null")
    for column in (
        config.actual_col,
        config.prediction_col,
        config.capacity_col,
    ):
        if not pd.api.types.is_numeric_dtype(oof_rows[column].dtype):
            raise ValueError(f"oof_rows column {column!r} must be numeric")
    capacities = oof_rows[config.capacity_col].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(capacities) & (capacities > 0.0)):
        raise ValueError("all capacity values must be positive and finite")
    return feature_names


def _build_targets_and_weights(
    oof_rows: pd.DataFrame,
    config: CorrelationMapConfig,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    """Construct pooled targets and official-group-balanced energy weights."""

    n_rows = len(oof_rows)
    targets = {
        name: np.full(n_rows, np.nan, dtype=np.float64)
        for name in TARGET_ORDER
    }
    eligible = np.zeros(n_rows, dtype=bool)
    eligible_energy_weights = np.zeros(n_rows, dtype=np.float64)
    all_energy_weights = np.zeros(n_rows, dtype=np.float64)

    group_values = oof_rows[config.group_col].astype(str).to_numpy()
    actual_all = oof_rows[config.actual_col].to_numpy(dtype=np.float64)
    prediction_all = oof_rows[config.prediction_col].to_numpy(dtype=np.float64)
    capacity_all = oof_rows[config.capacity_col].to_numpy(dtype=np.float64)

    for group in sorted(np.unique(group_values).tolist()):
        positions = np.flatnonzero(group_values == group)
        group_capacities = np.unique(capacity_all[positions])
        if len(group_capacities) != 1:
            raise ValueError(
                f"group {group!r} must have one immutable capacity; "
                f"found {group_capacities.tolist()}"
            )
        capacity = float(group_capacities[0])
        actual = actual_all[positions]
        prediction = prediction_all[positions]
        group_targets = correlation_targets(
            actual,
            prediction,
            capacity,
            action_fraction=config.action_fraction,
        )
        group_eligible = group_targets["eligible"].astype(bool)
        eligible[positions] = group_eligible
        for target in TARGET_ORDER:
            targets[target][positions] = group_targets[target]

        # Each group has total weight one, mirroring the official equal-group
        # aggregation; within a group weights remain proportional to generation.
        group_actual = actual[group_eligible]
        energy_sum = float(np.sum(group_actual))
        if not math.isfinite(energy_sum) or energy_sum <= 0.0:
            raise ValueError(f"group {group!r} has invalid eligible energy")
        eligible_positions = positions[group_eligible]
        eligible_energy_weights[eligible_positions] = group_actual / energy_sum

        all_energy_valid = np.isfinite(actual) & (actual > 0.0)
        all_energy_sum = float(np.sum(actual[all_energy_valid]))
        if math.isfinite(all_energy_sum) and all_energy_sum > 0.0:
            all_energy_weights[positions[all_energy_valid]] = (
                actual[all_energy_valid] / all_energy_sum
            )

    if not np.any(eligible):
        raise ValueError("OOF rows contain no score-eligible observations")
    target_weights = {
        name: (
            all_energy_weights if name == "eligible" else eligible_energy_weights
        )
        for name in TARGET_ORDER
    }
    return targets, eligible, target_weights


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = x - float(np.mean(x))
    y_centered = y - float(np.mean(y))
    denominator = math.sqrt(
        float(np.dot(x_centered, x_centered))
        * float(np.dot(y_centered, y_centered))
    )
    if denominator <= 0.0 or not math.isfinite(denominator):
        return math.nan
    return float(np.dot(x_centered, y_centered) / denominator)


def _weighted_pearson(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> float:
    weight_sum = float(np.sum(weights))
    if not math.isfinite(weight_sum) or weight_sum <= 0.0:
        return math.nan
    normalized = weights / weight_sum
    x_centered = x - float(np.sum(normalized * x))
    y_centered = y - float(np.sum(normalized * y))
    denominator = math.sqrt(
        float(np.sum(normalized * x_centered * x_centered))
        * float(np.sum(normalized * y_centered * y_centered))
    )
    if denominator <= 0.0 or not math.isfinite(denominator):
        return math.nan
    return float(np.sum(normalized * x_centered * y_centered) / denominator)


def _continuous_pair_metrics(
    feature: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    *,
    min_rows: int,
) -> tuple[int, float, float, float]:
    valid = (
        np.isfinite(feature)
        & np.isfinite(target)
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    count = int(np.sum(valid))
    if count < min_rows:
        return count, math.nan, math.nan, math.nan
    x = feature[valid]
    y = target[valid]
    weight = weights[valid]
    if float(np.max(x)) == float(np.min(x)):
        return count, math.nan, math.nan, math.nan
    if float(np.max(y)) == float(np.min(y)):
        return count, math.nan, math.nan, math.nan
    x_rank = rankdata(x, method="average")
    y_rank = rankdata(y, method="average")
    spearman = _pearson(x_rank, y_rank)
    weighted_spearman = _weighted_pearson(x_rank, y_rank, weight)
    denominator = float(count)
    x_normal = ndtri((x_rank - 0.5) / denominator)
    y_normal = ndtri((y_rank - 0.5) / denominator)
    rank_normal_pearson = _pearson(x_normal, y_normal)
    return count, spearman, weighted_spearman, rank_normal_pearson


def _weighted_auc(
    feature: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    *,
    min_rows: int,
) -> tuple[int, float]:
    valid = (
        np.isfinite(feature)
        & np.isfinite(target)
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    count = int(np.sum(valid))
    if count < min_rows:
        return count, math.nan
    x = feature[valid]
    y = target[valid]
    weight = weights[valid]
    if float(np.max(x)) == float(np.min(x)):
        return count, 0.5
    classes = np.unique(y)
    if len(classes) != 2 or not np.array_equal(classes, [0.0, 1.0]):
        return count, math.nan
    auc = roc_auc_score(y, x, sample_weight=weight)
    return count, float(auc)


def _finite_summary(values: Sequence[float]) -> tuple[int, float, float, float]:
    array = np.asarray(tuple(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    if not len(finite):
        return 0, math.nan, math.nan, math.nan
    return (
        int(len(finite)),
        float(np.mean(finite)),
        float(np.min(finite)),
        float(np.max(finite)),
    )


def _sign_stability(
    overall: float,
    slice_values: Sequence[float],
    *,
    neutral: float,
    epsilon: float,
) -> tuple[int, float]:
    overall_edge = float(overall) - float(neutral)
    if not math.isfinite(overall_edge) or abs(overall_edge) <= epsilon:
        return 0, math.nan
    expected = math.copysign(1.0, overall_edge)
    informative: list[float] = []
    for value in slice_values:
        edge = float(value) - float(neutral)
        if math.isfinite(edge) and abs(edge) > epsilon:
            informative.append(edge)
    if not informative:
        return 0, math.nan
    matches = sum(math.copysign(1.0, value) == expected for value in informative)
    return len(informative), float(matches / len(informative))


def _stability_by_stratum(
    overall: float,
    slice_masks: Sequence[tuple[str, np.ndarray]],
    slice_values: Sequence[float],
    *,
    neutral: float,
    epsilon: float,
) -> dict[str, dict[str, float | int | None]]:
    grouped: dict[str, list[float]] = {}
    for (slice_name, _), value in zip(slice_masks, slice_values, strict=True):
        stratum = slice_name.split("=", 1)[0]
        grouped.setdefault(stratum, []).append(float(value))
    output: dict[str, dict[str, float | int | None]] = {}
    for stratum, values in grouped.items():
        count, stability = _sign_stability(
            overall,
            values,
            neutral=neutral,
            epsilon=epsilon,
        )
        output[stratum] = {
            "informative_slice_count": count,
            "sign_stability": stability if math.isfinite(stability) else None,
        }
    return output


def _feature_records(
    *,
    feature_position: int,
    feature_name: str,
    feature_values: np.ndarray,
    taxonomy: Mapping[str, Any],
    targets: Mapping[str, np.ndarray],
    target_weights: Mapping[str, np.ndarray],
    slice_masks: Sequence[tuple[str, np.ndarray]],
    config: CorrelationMapConfig,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    taxonomy_columns = {
        key: value
        for key, value in taxonomy.items()
        if key not in {"position", "feature"}
    }
    for target_name in TARGET_ORDER:
        target_values = targets[target_name]
        weights = target_weights[target_name]
        target_kind = TARGET_KIND[target_name]
        base: dict[str, Any] = {
            "position": int(feature_position),
            "feature": feature_name,
            **taxonomy_columns,
            "target": target_name,
            "target_kind": target_kind,
            "n_valid": 0,
            "spearman": math.nan,
            "weighted_spearman": math.nan,
            "rank_normalized_pearson": math.nan,
            "energy_weighted_auc": math.nan,
            "weighted_auc_signed_edge": math.nan,
            "rank_biserial": math.nan,
            "weighted_auc_skill": math.nan,
            "registered_slice_count": int(len(slice_masks)),
            "valid_spearman_slice_count": 0,
            "mean_slice_spearman": math.nan,
            "minimum_slice_spearman": math.nan,
            "maximum_slice_spearman": math.nan,
            "spearman_sign_slice_count": 0,
            "spearman_sign_stability": math.nan,
            "valid_weighted_spearman_slice_count": 0,
            "mean_slice_weighted_spearman": math.nan,
            "minimum_slice_weighted_spearman": math.nan,
            "maximum_slice_weighted_spearman": math.nan,
            "weighted_spearman_sign_slice_count": 0,
            "weighted_spearman_sign_stability": math.nan,
            "valid_rank_normal_slice_count": 0,
            "mean_slice_rank_normalized_pearson": math.nan,
            "minimum_slice_rank_normalized_pearson": math.nan,
            "maximum_slice_rank_normalized_pearson": math.nan,
            "rank_normal_sign_slice_count": 0,
            "rank_normal_sign_stability": math.nan,
            "valid_auc_slice_count": 0,
            "mean_slice_energy_weighted_auc": math.nan,
            "minimum_slice_energy_weighted_auc": math.nan,
            "maximum_slice_energy_weighted_auc": math.nan,
            "auc_sign_slice_count": 0,
            "auc_sign_stability": math.nan,
            "group_sign_stability": math.nan,
            "fold_sign_stability": math.nan,
            "sign_stability_by_stratum_json": None,
        }

        if target_kind == "binary_boundary":
            count, overall_auc = _weighted_auc(
                feature_values,
                target_values,
                weights,
                min_rows=config.min_pair_rows,
            )
            slice_auc: list[float] = []
            for _, mask in slice_masks:
                _, value = _weighted_auc(
                    feature_values[mask],
                    target_values[mask],
                    weights[mask],
                    min_rows=config.min_slice_rows,
                )
                slice_auc.append(value)
            valid_count, mean_value, min_value, max_value = _finite_summary(
                slice_auc
            )
            sign_count, stability = _sign_stability(
                overall_auc,
                slice_auc,
                neutral=0.5,
                epsilon=config.stability_epsilon,
            )
            signed_edge = (
                2.0 * (overall_auc - 0.5)
                if math.isfinite(overall_auc)
                else math.nan
            )
            by_stratum = _stability_by_stratum(
                overall_auc,
                slice_masks,
                slice_auc,
                neutral=0.5,
                epsilon=config.stability_epsilon,
            )
            base.update(
                {
                    "n_valid": count,
                    "energy_weighted_auc": overall_auc,
                    "weighted_auc_signed_edge": signed_edge,
                    "rank_biserial": signed_edge,
                    "weighted_auc_skill": (
                        abs(signed_edge) if math.isfinite(signed_edge) else math.nan
                    ),
                    "valid_auc_slice_count": valid_count,
                    "mean_slice_energy_weighted_auc": mean_value,
                    "minimum_slice_energy_weighted_auc": min_value,
                    "maximum_slice_energy_weighted_auc": max_value,
                    "auc_sign_slice_count": sign_count,
                    "auc_sign_stability": stability,
                    "group_sign_stability": by_stratum.get("group", {}).get(
                        "sign_stability", math.nan
                    ),
                    "fold_sign_stability": by_stratum.get("fold", {}).get(
                        "sign_stability", math.nan
                    ),
                    "sign_stability_by_stratum_json": json.dumps(
                        by_stratum,
                        ensure_ascii=False,
                        sort_keys=True,
                        allow_nan=False,
                    ),
                }
            )
        else:
            count, spearman, weighted_spearman, rank_normal = _continuous_pair_metrics(
                feature_values,
                target_values,
                weights,
                min_rows=config.min_pair_rows,
            )
            slice_spearman: list[float] = []
            slice_weighted_spearman: list[float] = []
            slice_rank_normal: list[float] = []
            for _, mask in slice_masks:
                _, slice_rho, slice_weighted_rho, slice_gaussian = _continuous_pair_metrics(
                    feature_values[mask],
                    target_values[mask],
                    weights[mask],
                    min_rows=config.min_slice_rows,
                )
                slice_spearman.append(slice_rho)
                slice_weighted_spearman.append(slice_weighted_rho)
                slice_rank_normal.append(slice_gaussian)
            sp_count, sp_mean, sp_min, sp_max = _finite_summary(slice_spearman)
            wsp_count, wsp_mean, wsp_min, wsp_max = _finite_summary(
                slice_weighted_spearman
            )
            rn_count, rn_mean, rn_min, rn_max = _finite_summary(slice_rank_normal)
            sp_sign_count, sp_stability = _sign_stability(
                spearman,
                slice_spearman,
                neutral=0.0,
                epsilon=config.stability_epsilon,
            )
            wsp_sign_count, wsp_stability = _sign_stability(
                weighted_spearman,
                slice_weighted_spearman,
                neutral=0.0,
                epsilon=config.stability_epsilon,
            )
            rn_sign_count, rn_stability = _sign_stability(
                rank_normal,
                slice_rank_normal,
                neutral=0.0,
                epsilon=config.stability_epsilon,
            )
            by_stratum = _stability_by_stratum(
                weighted_spearman,
                slice_masks,
                slice_weighted_spearman,
                neutral=0.0,
                epsilon=config.stability_epsilon,
            )
            base.update(
                {
                    "n_valid": count,
                    "spearman": spearman,
                    "weighted_spearman": weighted_spearman,
                    "rank_normalized_pearson": rank_normal,
                    "valid_spearman_slice_count": sp_count,
                    "mean_slice_spearman": sp_mean,
                    "minimum_slice_spearman": sp_min,
                    "maximum_slice_spearman": sp_max,
                    "spearman_sign_slice_count": sp_sign_count,
                    "spearman_sign_stability": sp_stability,
                    "valid_weighted_spearman_slice_count": wsp_count,
                    "mean_slice_weighted_spearman": wsp_mean,
                    "minimum_slice_weighted_spearman": wsp_min,
                    "maximum_slice_weighted_spearman": wsp_max,
                    "weighted_spearman_sign_slice_count": wsp_sign_count,
                    "weighted_spearman_sign_stability": wsp_stability,
                    "valid_rank_normal_slice_count": rn_count,
                    "mean_slice_rank_normalized_pearson": rn_mean,
                    "minimum_slice_rank_normalized_pearson": rn_min,
                    "maximum_slice_rank_normalized_pearson": rn_max,
                    "rank_normal_sign_slice_count": rn_sign_count,
                    "rank_normal_sign_stability": rn_stability,
                    "group_sign_stability": by_stratum.get("group", {}).get(
                        "sign_stability", math.nan
                    ),
                    "fold_sign_stability": by_stratum.get("fold", {}).get(
                        "sign_stability", math.nan
                    ),
                    "sign_stability_by_stratum_json": json.dumps(
                        by_stratum,
                        ensure_ascii=False,
                        sort_keys=True,
                        allow_nan=False,
                    ),
                }
            )
        records.append(base)
    return records


def _deterministic_row_sample(
    frame: pd.DataFrame,
    *,
    max_rows: int,
) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame
    positions = np.linspace(0, len(frame) - 1, num=max_rows, dtype=np.int64)
    return frame.iloc[positions]


def _fast_spearman_matrix(
    frame: pd.DataFrame,
    *,
    min_rows: int,
    n_jobs: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a bounded-memory Spearman matrix and missing-data provenance.

    Complete rows are preferred because a single rank matrix can then be
    multiplied efficiently.  If fewer than ``min_rows`` complete cases exist,
    each feature is deterministically median-imputed before ranking.  The
    strategy is always disclosed in the returned metadata.
    """

    values = frame.to_numpy(dtype=np.float64, copy=False)
    complete_mask = np.all(np.isfinite(values), axis=1)
    complete_count = int(np.sum(complete_mask))
    minimum_complete_cases = max(min_rows, int(math.ceil(0.80 * len(frame))))
    if complete_count >= minimum_complete_cases:
        working = np.asarray(values[complete_mask], dtype=np.float64)
        strategy = "complete_case"
    else:
        working = np.asarray(values, dtype=np.float64).copy()
        for column in range(working.shape[1]):
            finite = np.isfinite(working[:, column])
            if int(np.sum(finite)) < min_rows:
                working[:, column] = 0.0
                continue
            median = float(np.median(working[finite, column]))
            working[~finite, column] = median
        strategy = "feature_median_imputation"

    if len(working) < min_rows:
        raise ValueError("redundancy source has too few usable rows")
    # pandas' vectorised average ranks preserve ties and are materially faster
    # than O(p^2) pairwise scipy calls for the 612-column cache.
    ranked = pd.DataFrame(working, copy=False).rank(
        axis=0,
        method="average",
        na_option="keep",
    ).to_numpy(dtype=np.float32, copy=True)
    del working
    ranked -= np.mean(ranked, axis=0, dtype=np.float64).astype(np.float32)
    norms = np.sqrt(np.sum(ranked * ranked, axis=0, dtype=np.float64))
    constant = (~np.isfinite(norms)) | (norms <= 0.0)
    safe_norms = norms.copy()
    safe_norms[constant] = 1.0
    ranked /= safe_norms.astype(np.float32)
    with threadpool_limits(limits=n_jobs):
        correlations = ranked.T @ ranked
    correlations = np.asarray(correlations, dtype=np.float64)
    correlations = np.clip(correlations, -1.0, 1.0)
    correlations[constant, :] = np.nan
    correlations[:, constant] = np.nan
    np.fill_diagonal(correlations, np.where(constant, np.nan, 1.0))
    return correlations, {
        "input_rows": int(len(frame)),
        "used_rows": int(len(ranked)),
        "complete_rows": complete_count,
        "minimum_complete_cases_for_complete_case_strategy": (
            minimum_complete_cases
        ),
        "missing_strategy": strategy,
        "constant_feature_count": int(np.sum(constant)),
    }


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def build_redundancy_clusters(
    *,
    feature_names: Sequence[str],
    sources: Mapping[str, pd.DataFrame],
    config: CorrelationMapConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build ``abs(Spearman)`` redundancy components from causal sources.

    Sources are sampled deterministically and with the same per-source cap, so
    a much larger training block cannot drown out the causal apply block.  The
    threshold is applied to the pooled, source-balanced Spearman matrix.  The
    edge table also records source-level correlations and sign stability.
    """

    cfg = config or CorrelationMapConfig()
    names = tuple(str(value) for value in feature_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("feature_names must be non-empty and unique")
    if not isinstance(sources, Mapping) or not sources:
        raise ValueError("sources must be a non-empty mapping of causal frames")

    raw_source_names = list(sources)
    if any(not isinstance(value, str) or not value for value in raw_source_names):
        raise ValueError("redundancy source names must be non-empty strings")
    prepared: dict[str, pd.DataFrame] = {}
    for source_name in sorted(raw_source_names):
        frame = sources[source_name]
        _require_numeric_features(frame, name=f"redundancy source {source_name!r}")
        missing = [name for name in names if name not in frame.columns]
        if missing:
            raise ValueError(
                f"redundancy source {source_name!r} is missing features: "
                f"{missing[:5]}"
            )
        prepared[source_name] = frame.loc[:, list(names)]
    balanced_row_count = min(
        cfg.max_redundancy_rows_per_source,
        *(len(frame) for frame in prepared.values()),
    )
    if balanced_row_count < cfg.min_pair_rows:
        raise ValueError(
            "every redundancy source must supply at least min_pair_rows rows"
        )

    sampled: dict[str, pd.DataFrame] = {}
    source_metadata: dict[str, Any] = {}
    source_correlations: dict[str, np.ndarray] = {}
    for source_name in sorted(prepared):
        frame = prepared[source_name]
        selected = frame
        selected = _deterministic_row_sample(
            selected,
            max_rows=balanced_row_count,
        )
        sampled[source_name] = selected
        matrix, details = _fast_spearman_matrix(
            selected,
            min_rows=cfg.min_pair_rows,
            n_jobs=cfg.n_jobs,
        )
        source_correlations[source_name] = matrix
        source_metadata[source_name] = {
            "original_rows": int(len(frame)),
            **details,
        }

    pooled = pd.concat(
        [sampled[source] for source in sorted(sampled)],
        axis=0,
        ignore_index=True,
    )
    pooled_matrix, pooled_metadata = _fast_spearman_matrix(
        pooled,
        min_rows=cfg.min_pair_rows,
        n_jobs=cfg.n_jobs,
    )
    threshold = float(cfg.redundancy_threshold)
    upper = np.triu(np.ones(pooled_matrix.shape, dtype=bool), k=1)
    edge_left, edge_right = np.where(
        upper & np.isfinite(pooled_matrix) & (np.abs(pooled_matrix) >= threshold)
    )

    union_find = _UnionFind(len(names))
    edge_records: list[dict[str, Any]] = []
    for left, right in zip(edge_left.tolist(), edge_right.tolist(), strict=True):
        union_find.union(left, right)
        correlations: dict[str, float | None] = {
            source: (
                float(source_correlations[source][left, right])
                if math.isfinite(source_correlations[source][left, right])
                else None
            )
            for source in sorted(source_correlations)
        }
        finite_source_values = np.asarray(
            [value for value in correlations.values() if value is not None],
            dtype=np.float64,
        )
        pooled_value = float(pooled_matrix[left, right])
        if len(finite_source_values):
            signs = np.sign(finite_source_values)
            pooled_sign = np.sign(pooled_value)
            sign_stability = float(np.mean(signs == pooled_sign))
            min_source_abs = float(np.min(np.abs(finite_source_values)))
        else:
            sign_stability = math.nan
            min_source_abs = math.nan
        edge_records.append(
            {
                "feature_a_position": int(left),
                "feature_b_position": int(right),
                "feature_a": names[left],
                "feature_b": names[right],
                "pooled_spearman": pooled_value,
                "pooled_abs_spearman": abs(pooled_value),
                "spearman_distance": 1.0 - abs(pooled_value),
                "minimum_source_abs_spearman": min_source_abs,
                "source_sign_stability": sign_stability,
                "source_correlations_json": json.dumps(
                    correlations,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                ),
            }
        )

    components: dict[int, list[int]] = {}
    for position in range(len(names)):
        components.setdefault(union_find.find(position), []).append(position)
    ordered_components = sorted(components.values(), key=lambda values: min(values))
    membership_records: list[dict[str, Any]] = []
    for component_number, positions in enumerate(ordered_components, start=1):
        component_id = f"R{component_number:04d}"
        members = [names[position] for position in positions]
        for position in positions:
            membership_records.append(
                {
                    "position": int(position),
                    "feature": names[position],
                    "redundancy_cluster_id": component_id,
                    "redundancy_cluster_size": int(len(positions)),
                    "is_redundant": bool(len(positions) > 1),
                    "cluster_members_json": json.dumps(
                        members,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                }
            )
    membership = pd.DataFrame(membership_records).sort_values(
        "position", kind="stable"
    ).reset_index(drop=True)
    edge_columns = (
        "feature_a_position",
        "feature_b_position",
        "feature_a",
        "feature_b",
        "pooled_spearman",
        "pooled_abs_spearman",
        "spearman_distance",
        "minimum_source_abs_spearman",
        "source_sign_stability",
        "source_correlations_json",
    )
    edges = pd.DataFrame(edge_records, columns=edge_columns)
    metadata = {
        "threshold": threshold,
        "criterion": "abs(source_balanced_pooled_spearman) >= threshold",
        "source_count": int(len(sampled)),
        "balanced_rows_per_source": int(balanced_row_count),
        "sources": source_metadata,
        "pooled": pooled_metadata,
        "cluster_count": int(len(ordered_components)),
        "non_singleton_cluster_count": int(
            sum(len(values) > 1 for values in ordered_components)
        ),
        "edge_count": int(len(edges)),
    }
    return membership, edges, metadata


def _finite_values(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    return values[np.isfinite(values)]


def _quantile_or_nan(values: np.ndarray, probability: float) -> float:
    if not len(values):
        return math.nan
    return float(np.quantile(values, probability))


def _family_map(
    feature_map: pd.DataFrame,
    taxonomy: pd.DataFrame,
    redundancy: pd.DataFrame,
) -> pd.DataFrame:
    family_order = {name: position for position, name in enumerate(PHYSICAL_FAMILIES)}
    family_counts = taxonomy.groupby("physical_family", sort=False).agg(
        feature_count=("feature", "nunique"),
        run_dynamics_feature_count=("run_dynamics", "sum"),
        spatial_redundancy_exit_feature_count=("spatial_redundancy", "sum"),
        nonwind_atmospheric_feature_count=("nonwind_atmospheric", "sum"),
        cross_source_disagreement_feature_count=(
            "cross_source_disagreement",
            "sum",
        ),
    )
    taxonomy_redundancy = taxonomy[["feature", "physical_family"]].merge(
        redundancy[["feature", "is_redundant", "redundancy_cluster_id"]],
        on="feature",
        how="left",
        validate="one_to_one",
    )
    redundancy_counts = taxonomy_redundancy.groupby(
        "physical_family", sort=False
    ).agg(
        redundant_feature_count=("is_redundant", "sum"),
        redundancy_cluster_count=("redundancy_cluster_id", "nunique"),
    )

    records: list[dict[str, Any]] = []
    grouped = feature_map.groupby(["physical_family", "target"], sort=False)
    for (family, target), block in grouped:
        target_kind = str(block["target_kind"].iloc[0])
        base = {
            "physical_family": str(family),
            "target": str(target),
            "target_kind": target_kind,
            **{
                key: int(value)
                for key, value in family_counts.loc[family].to_dict().items()
            },
            **{
                key: int(value)
                for key, value in redundancy_counts.loc[family].to_dict().items()
            },
        }
        if target_kind == "binary_boundary":
            auc = _finite_values(block["energy_weighted_auc"])
            skill = _finite_values(block["weighted_auc_skill"])
            stability = _finite_values(block["auc_sign_stability"])
            ranked = block.assign(
                _sort=pd.to_numeric(
                    block["weighted_auc_skill"], errors="coerce"
                ).fillna(-np.inf)
            ).sort_values(["_sort", "feature"], ascending=[False, True])
            top = ranked.iloc[0] if len(ranked) and ranked["_sort"].iloc[0] >= 0 else None
            base.update(
                {
                    "finite_feature_count": int(len(auc)),
                    "median_energy_weighted_auc": (
                        float(np.median(auc)) if len(auc) else math.nan
                    ),
                    "median_weighted_auc_skill": (
                        float(np.median(skill)) if len(skill) else math.nan
                    ),
                    "q90_weighted_auc_skill": _quantile_or_nan(skill, 0.90),
                    "maximum_weighted_auc_skill": (
                        float(np.max(skill)) if len(skill) else math.nan
                    ),
                    "mean_sign_stability": (
                        float(np.mean(stability)) if len(stability) else math.nan
                    ),
                    "stable_feature_fraction": (
                        float(np.mean(stability >= 0.75))
                        if len(stability)
                        else math.nan
                    ),
                    "top_feature": str(top["feature"]) if top is not None else None,
                    "top_signed_value": (
                        float(top["weighted_auc_signed_edge"])
                        if top is not None
                        else math.nan
                    ),
                    "top_absolute_value": (
                        float(top["weighted_auc_skill"])
                        if top is not None
                        else math.nan
                    ),
                }
            )
        else:
            spearman = _finite_values(block["spearman"])
            weighted_spearman = _finite_values(block["weighted_spearman"])
            rank_normal = _finite_values(block["rank_normalized_pearson"])
            abs_spearman = np.abs(spearman)
            abs_rank_normal = np.abs(rank_normal)
            stability = _finite_values(
                block["weighted_spearman_sign_stability"]
            )
            ranked = block.assign(
                _sort=pd.to_numeric(block["weighted_spearman"], errors="coerce")
                .abs()
                .fillna(-np.inf)
            ).sort_values(["_sort", "feature"], ascending=[False, True])
            top = ranked.iloc[0] if len(ranked) and ranked["_sort"].iloc[0] >= 0 else None
            base.update(
                {
                    "finite_feature_count": int(len(spearman)),
                    "median_abs_weighted_spearman": (
                        float(np.median(np.abs(weighted_spearman)))
                        if len(weighted_spearman)
                        else math.nan
                    ),
                    "q90_abs_weighted_spearman": _quantile_or_nan(
                        np.abs(weighted_spearman), 0.90
                    ),
                    "maximum_abs_weighted_spearman": (
                        float(np.max(np.abs(weighted_spearman)))
                        if len(weighted_spearman)
                        else math.nan
                    ),
                    "median_abs_spearman": (
                        float(np.median(abs_spearman))
                        if len(abs_spearman)
                        else math.nan
                    ),
                    "q90_abs_spearman": _quantile_or_nan(abs_spearman, 0.90),
                    "maximum_abs_spearman": (
                        float(np.max(abs_spearman))
                        if len(abs_spearman)
                        else math.nan
                    ),
                    "median_abs_rank_normalized_pearson": (
                        float(np.median(abs_rank_normal))
                        if len(abs_rank_normal)
                        else math.nan
                    ),
                    "q90_abs_rank_normalized_pearson": _quantile_or_nan(
                        abs_rank_normal, 0.90
                    ),
                    "maximum_abs_rank_normalized_pearson": (
                        float(np.max(abs_rank_normal))
                        if len(abs_rank_normal)
                        else math.nan
                    ),
                    "mean_sign_stability": (
                        float(np.mean(stability)) if len(stability) else math.nan
                    ),
                    "stable_feature_fraction": (
                        float(np.mean(stability >= 0.75))
                        if len(stability)
                        else math.nan
                    ),
                    "top_feature": str(top["feature"]) if top is not None else None,
                    "top_signed_value": (
                        float(top["weighted_spearman"])
                        if top is not None
                        else math.nan
                    ),
                    "top_absolute_value": (
                        abs(float(top["weighted_spearman"]))
                        if top is not None
                        else math.nan
                    ),
                }
            )
        records.append(base)

    result = pd.DataFrame(records)
    result["_family_order"] = result["physical_family"].map(family_order)
    target_order = {name: position for position, name in enumerate(TARGET_ORDER)}
    result["_target_order"] = result["target"].map(target_order)
    return result.sort_values(
        ["_family_order", "_target_order"], kind="stable"
    ).drop(columns=["_family_order", "_target_order"]).reset_index(drop=True)


def _bin_labels(
    values: np.ndarray,
    edges: Sequence[float],
    *,
    prefix: str,
) -> np.ndarray:
    numeric = np.asarray(values, dtype=np.float64)
    edge_array = np.asarray(tuple(edges), dtype=np.float64)
    positions = np.searchsorted(edge_array, numeric, side="right")
    labels = np.asarray(
        [f"{prefix}{position}" for position in positions],
        dtype=object,
    )
    labels[~np.isfinite(numeric)] = f"{prefix}missing"
    return labels.astype(str)


def _season_from_month(month: np.ndarray) -> np.ndarray:
    values = np.asarray(month, dtype=np.int64)
    output = np.full(len(values), "unknown", dtype=object)
    output[np.isin(values, [12, 1, 2])] = "DJF"
    output[np.isin(values, [3, 4, 5])] = "MAM"
    output[np.isin(values, [6, 7, 8])] = "JJA"
    output[np.isin(values, [9, 10, 11])] = "SON"
    return output.astype(str)


def _registered_strata(
    oof_features: pd.DataFrame,
    oof_rows: pd.DataFrame,
    config: CorrelationMapConfig,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Materialise the five strata frozen in the preregistration."""

    strata: dict[str, np.ndarray] = {
        "group": oof_rows[config.group_col].astype(str).to_numpy(),
        "fold": oof_rows[config.fold_col].astype(str).to_numpy(),
    }
    provenance = {
        "group": f"oof_rows.{config.group_col}",
        "fold": f"oof_rows.{config.fold_col}",
    }

    if "lead_bin" in oof_rows.columns:
        strata["lead_bin"] = oof_rows["lead_bin"].astype(str).to_numpy()
        provenance["lead_bin"] = "oof_rows.lead_bin"
    elif "time__lead_hours" in oof_features.columns:
        strata["lead_bin"] = _bin_labels(
            oof_features["time__lead_hours"].to_numpy(dtype=np.float64),
            config.lead_bin_edges,
            prefix="lead_bin_",
        )
        provenance["lead_bin"] = (
            "derived from time__lead_hours with right-open searchsorted edges "
            f"{list(config.lead_bin_edges)}"
        )
    else:
        strata["lead_bin"] = np.repeat("unavailable", len(oof_rows))
        provenance["lead_bin"] = "unavailable: no lead_bin or time__lead_hours"

    if "season" in oof_rows.columns:
        strata["season"] = oof_rows["season"].astype(str).to_numpy()
        provenance["season"] = "oof_rows.season"
    else:
        timestamps: pd.DatetimeIndex | None = None
        if "forecast_kst_dtm" in oof_rows.columns:
            timestamps = pd.DatetimeIndex(
                pd.to_datetime(oof_rows["forecast_kst_dtm"], errors="coerce")
            )
            provenance["season"] = "derived from oof_rows.forecast_kst_dtm"
        elif isinstance(oof_rows.index, pd.DatetimeIndex):
            timestamps = oof_rows.index
            provenance["season"] = "derived from OOF DatetimeIndex"
        if timestamps is None or timestamps.isna().any():
            strata["season"] = np.repeat("unavailable", len(oof_rows))
            provenance["season"] = "unavailable: no complete forecast timestamp"
        else:
            strata["season"] = _season_from_month(timestamps.month.to_numpy())

    if "causal_control_cf_bin" in oof_rows.columns:
        strata["causal_control_cf_bin"] = oof_rows[
            "causal_control_cf_bin"
        ].astype(str).to_numpy()
        provenance["causal_control_cf_bin"] = (
            "oof_rows.causal_control_cf_bin"
        )
    else:
        control_cf = (
            oof_rows[config.prediction_col].to_numpy(dtype=np.float64)
            / oof_rows[config.capacity_col].to_numpy(dtype=np.float64)
        )
        strata["causal_control_cf_bin"] = _bin_labels(
            control_cf,
            config.control_cf_bin_edges,
            prefix="control_cf_bin_",
        )
        provenance["causal_control_cf_bin"] = (
            "derived from frozen baseline_kwh/capacity_kwh with right-open "
            f"searchsorted edges {list(config.control_cf_bin_edges)}"
        )
    return strata, provenance


def build_correlation_map(
    *,
    oof_features: pd.DataFrame,
    oof_rows: pd.DataFrame,
    redundancy_sources: Mapping[str, pd.DataFrame] | None = None,
    config: CorrelationMapConfig | None = None,
) -> CorrelationMapResult:
    """Build a score-aware map from already-causal OOF rows.

    ``oof_features`` and ``oof_rows`` must have exactly equal indices.  The
    latter supplies group, fold, actual, frozen baseline prediction, and fixed
    capacity columns.  Passing causal training and application feature frames
    through ``redundancy_sources`` is recommended; if omitted, OOF application
    rows alone are used for the unsupervised redundancy graph.
    """

    cfg = config or CorrelationMapConfig()
    feature_names = _validate_oof_inputs(oof_features, oof_rows, cfg)
    taxonomy_records = build_feature_taxonomy(feature_names)
    taxonomy = pd.DataFrame(taxonomy_records)
    targets, eligible, target_weights = _build_targets_and_weights(oof_rows, cfg)

    strata, strata_provenance = _registered_strata(
        oof_features,
        oof_rows,
        cfg,
    )
    slice_masks = [
        (f"{stratum}={level}", values == level)
        for stratum, values in strata.items()
        for level in sorted(np.unique(values).tolist())
    ]
    group = strata["group"]

    sources = (
        redundancy_sources
        if redundancy_sources is not None
        else {"causal_oof_apply": oof_features}
    )
    redundancy, redundancy_edges, redundancy_metadata = build_redundancy_clusters(
        feature_names=feature_names,
        sources=sources,
        config=cfg,
    )

    feature_values = oof_features.to_numpy(copy=False)
    with threadpool_limits(limits=1):
        nested = Parallel(n_jobs=cfg.n_jobs, prefer="threads")(
            delayed(_feature_records)(
                feature_position=position,
                feature_name=feature_name,
                feature_values=np.asarray(
                    feature_values[:, position], dtype=np.float64
                ),
                taxonomy=taxonomy_records[position],
                targets=targets,
                target_weights=target_weights,
                slice_masks=slice_masks,
                config=cfg,
            )
            for position, feature_name in enumerate(feature_names)
        )
    records = [record for feature_records in nested for record in feature_records]
    feature_map = pd.DataFrame(records)
    feature_map = feature_map.merge(
        redundancy.drop(columns="position"),
        on="feature",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    feature_map = feature_map.sort_values(
        ["position", "target"],
        key=lambda values: (
            values.map({name: index for index, name in enumerate(TARGET_ORDER)})
            if values.name == "target"
            else values
        ),
        kind="stable",
    ).reset_index(drop=True)
    within_group_constant: dict[str, bool] = {}
    for position, feature_name in enumerate(feature_names):
        if taxonomy_records[position]["physical_family"] != "static_site":
            within_group_constant[feature_name] = False
            continue
        values = np.asarray(feature_values[:, position], dtype=np.float64)
        within_group_constant[feature_name] = all(
            len(np.unique(values[(group == level) & np.isfinite(values)])) <= 1
            for level in np.unique(group)
        )
    feature_map["site_constant_within_group"] = feature_map["feature"].map(
        within_group_constant
    )
    family_map = _family_map(feature_map, taxonomy, redundancy)

    metadata: dict[str, Any] = {
        "schema_version": "score_aware_correlation_map_v1",
        "interpretation": (
            "Causal OOF diagnostic only; correlations and AUC are not a Public "
            "or Private leaderboard-score estimate."
        ),
        "target_source": "src.virtual_feature_exit.correlation_targets",
        "rank_normalization": "ndtri((average_rank - 0.5) / n)",
        "continuous_primary": (
            "energy-weighted Pearson correlation of average ranks "
            "(weighted_spearman)"
        ),
        "auc_weighting": (
            "actual energy normalized to total one within each group; score "
            "targets exclude ineligible rows, while eligible uses all positive-"
            "energy labeled rows"
        ),
        "sign_stability": (
            "fraction of informative registered-stratum effects matching the "
            "pooled effect sign"
        ),
        "site_constant_rule": (
            "static_site features constant inside every group retain pooled "
            "diagnostics but have undefined within-group correlations"
        ),
        "config": asdict(cfg),
        "oof_row_count": int(len(oof_rows)),
        "eligible_row_count": int(np.sum(eligible)),
        "feature_count": int(len(feature_names)),
        "feature_target_row_count": int(len(feature_map)),
        "family_target_row_count": int(len(family_map)),
        "groups": sorted(np.unique(group).tolist()),
        "registered_strata": {
            stratum: {
                "levels": sorted(np.unique(values).tolist()),
                "provenance": strata_provenance[stratum],
            }
            for stratum, values in strata.items()
        },
        "registered_slices": [name for name, _ in slice_masks],
        "targets": [
            {"name": name, "kind": TARGET_KIND[name]} for name in TARGET_ORDER
        ],
        "redundancy": redundancy_metadata,
    }
    return CorrelationMapResult(
        feature_map=feature_map,
        family_map=family_map,
        redundancy_membership=redundancy,
        redundancy_edges=redundancy_edges,
        metadata=metadata,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp.parquet")
    frame.to_parquet(
        temporary,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    temporary.replace(path)


def _write_records_json_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp.json")
    frame.to_json(
        temporary,
        orient="records",
        indent=2,
        force_ascii=False,
        double_precision=15,
    )
    temporary.replace(path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_correlation_map(
    result: CorrelationMapResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write feature, family, and redundancy maps as Parquet plus JSON."""

    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"no-overwrite correlation-map directory is not empty: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    tables = {
        "feature_map": result.feature_map,
        "family_map": result.family_map,
        "redundancy_membership": result.redundancy_membership,
        "redundancy_edges": result.redundancy_edges,
    }
    paths: dict[str, Path] = {}
    for name, frame in tables.items():
        parquet_path = destination / f"{name}.parquet"
        json_path = destination / f"{name}.json"
        _write_parquet_atomic(frame, parquet_path)
        _write_records_json_atomic(frame, json_path)
        paths[f"{name}_parquet"] = parquet_path
        paths[f"{name}_json"] = json_path

    output_hashes = {
        name: _sha256(path)
        for name, path in sorted(paths.items())
    }
    metadata = {
        **result.metadata,
        "output_sha256": output_hashes,
    }
    metadata_path = destination / "correlation_map_metadata.json"
    temporary = metadata_path.with_name(f".{metadata_path.name}.tmp.json")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            _json_safe(metadata),
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
    temporary.replace(metadata_path)
    paths["metadata_json"] = metadata_path
    return paths


__all__ = [
    "BINARY_TARGETS",
    "CONTINUOUS_TARGETS",
    "CorrelationMapConfig",
    "CorrelationMapResult",
    "TARGET_KIND",
    "TARGET_ORDER",
    "UTILITY_TARGETS",
    "build_correlation_map",
    "build_redundancy_clusters",
    "write_correlation_map",
]

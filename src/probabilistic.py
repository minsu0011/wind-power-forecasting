"""Leakage-safe probabilistic decisions for the discontinuous BARAM metric.

The official group score is the average of one-minus-NMAE and FICR.  For an
evaluated row, and after dropping constants that do not depend on the forecast,
the conditional decision utility in capacity-factor units is

``-0.5 * abs(p-y) + y * settlement(p, y) / (8 * mean(y))``.

``settlement`` is 4 / 3 / 0 at the official 6% / 8% error boundaries.  The
classes below estimate the conditional distribution or the conditional utility
on labelled *fit* rows, then choose a bounded forecast on disjoint application
rows.  They deliberately refuse overlapping indexes so an in-sample fit score
cannot accidentally be reported as validation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


DecisionMethod = Literal[
    "global_conformal",
    "binned_kde",
    "quantile_stack",
    "threshold_meta",
]


@dataclass(frozen=True)
class ProbabilisticDecisionConfig:
    """Fixed regularisation controls shared by all decision estimators."""

    candidate_offsets_cf: tuple[float, ...] = (
        -0.12,
        -0.10,
        -0.08,
        -0.06,
        -0.04,
        -0.03,
        -0.02,
        -0.01,
        0.0,
        0.01,
        0.02,
        0.03,
        0.04,
        0.06,
        0.08,
        0.10,
        0.12,
    )
    lower_cf: float = 0.0
    upper_cf: float = 1.02
    minimum_actual_cf: float = 0.10
    decision_shrinkage: float = 0.65
    minimum_fit_samples: int = 300
    residual_sample_count: int = 41
    bin_edges_cf: tuple[float, ...] = (
        0.10,
        0.20,
        0.35,
        0.50,
        0.65,
        0.80,
        0.95,
    )
    minimum_bin_samples: int = 250
    bin_shrinkage_samples: float = 500.0
    kde_bandwidth_floor: float = 0.003
    kde_bandwidth_ceiling: float = 0.030
    quantile_levels: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)
    quantile_sample_count: int = 19
    quantile_max_iter: int = 90
    quantile_min_samples_leaf: int = 180
    meta_max_iter: int = 80
    meta_min_samples_leaf: int = 300
    random_state: int = 42


def _frame(values: Any, *, name: str) -> pd.DataFrame:
    if not isinstance(values, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if values.empty:
        raise ValueError(f"{name} must not be empty")
    if not values.index.is_unique:
        raise ValueError(f"{name} index must be unique")
    if not values.index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be monotonically increasing")
    if not values.columns.is_unique:
        raise ValueError(f"{name} columns must be unique")
    result = values.astype(float)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _series(values: Any, *, name: str, allow_nan: bool) -> pd.Series:
    if not isinstance(values, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    if values.empty:
        raise ValueError(f"{name} must not be empty")
    if not values.index.is_unique or not values.index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be unique and monotonically increasing")
    result = values.astype(float)
    array = result.to_numpy()
    if np.isinf(array).any() or (not allow_nan and np.isnan(array).any()):
        raise ValueError(f"{name} contains invalid values")
    return result


def _require_aligned(reference: pd.Index, other: pd.Index, *, name: str) -> None:
    if not reference.equals(other):
        raise ValueError(f"{name} index is not aligned")


def _thin_quantiles(values: np.ndarray, count: int) -> np.ndarray:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("cannot sample an empty residual distribution")
    if count < 5:
        raise ValueError("residual_sample_count must be at least 5")
    probabilities = np.linspace(0.5 / count, 1.0 - 0.5 / count, count)
    return np.quantile(finite, probabilities)


def _bandwidth(values: np.ndarray, config: ProbabilisticDecisionConfig) -> float:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return float(config.kde_bandwidth_floor)
    standard = float(np.std(values, ddof=1))
    q25, q75 = np.quantile(values, [0.25, 0.75])
    robust = float((q75 - q25) / 1.349)
    scale = min(standard, robust) if robust > 0 else standard
    estimate = 0.9 * max(scale, 1e-12) * values.size ** (-0.2)
    return float(
        np.clip(
            estimate,
            config.kde_bandwidth_floor,
            config.kde_bandwidth_ceiling,
        )
    )


def _kde_samples(
    residuals: np.ndarray,
    config: ProbabilisticDecisionConfig,
) -> np.ndarray:
    centres = _thin_quantiles(residuals, config.residual_sample_count)
    bandwidth = _bandwidth(residuals, config)
    # Repeating the centre gives deterministic kernel weights 1/4, 1/2, 1/4.
    return np.concatenate(
        [centres - bandwidth, centres, centres, centres + bandwidth]
    )


def _utility_matrix(
    candidates_cf: np.ndarray,
    outcome_samples_cf: np.ndarray,
    mean_actual_cf: float,
) -> np.ndarray:
    candidates = np.asarray(candidates_cf, dtype=float)
    outcomes = np.asarray(outcome_samples_cf, dtype=float)
    error = np.abs(candidates[:, None] - outcomes[None, :])
    settlement = np.select(
        [error <= 0.06, error <= 0.08],
        [4.0, 3.0],
        default=0.0,
    )
    utility = -0.5 * error + outcomes[None, :] * settlement / (
        8.0 * mean_actual_cf
    )
    return utility


class ProbabilisticFICRDecision:
    """Choose forecasts by a regularised conditional expected-score estimate.

    Four fixed methods are supported:

    * ``global_conformal``: global empirical residual distribution;
    * ``binned_kde``: capacity-factor-bin residual KDE shrunk to the global KDE;
    * ``quantile_stack``: shallow quantile HistGB models over base predictions;
    * ``threshold_meta``: shallow direct utility meta-regression over candidate
      forecasts, trained with the exact 6% / 8% realised reward.
    """

    METHODS: tuple[str, ...] = (
        "global_conformal",
        "binned_kde",
        "quantile_stack",
        "threshold_meta",
    )

    def __init__(
        self,
        method: DecisionMethod,
        *,
        config: ProbabilisticDecisionConfig | None = None,
    ) -> None:
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS!r}")
        self.method = method
        self.config = config or ProbabilisticDecisionConfig()

    def _validate_config(self) -> None:
        config = self.config
        offsets = np.asarray(config.candidate_offsets_cf, dtype=float)
        if offsets.ndim != 1 or offsets.size < 3 or not np.isfinite(offsets).all():
            raise ValueError("candidate_offsets_cf must be a finite 1-D grid")
        if not np.any(np.isclose(offsets, 0.0)):
            raise ValueError("candidate_offsets_cf must include zero")
        if not config.lower_cf <= 0.0 < config.upper_cf:
            raise ValueError("prediction bounds must include zero")
        if not 0.0 < config.decision_shrinkage <= 1.0:
            raise ValueError("decision_shrinkage must be in (0, 1]")
        if config.minimum_fit_samples < 20:
            raise ValueError("minimum_fit_samples must be at least 20")
        levels = np.asarray(config.quantile_levels, dtype=float)
        if levels.size < 3 or np.any(np.diff(levels) <= 0):
            raise ValueError("quantile_levels must be strictly increasing")
        if levels[0] <= 0 or levels[-1] >= 1:
            raise ValueError("quantile_levels must lie strictly inside (0, 1)")

    def fit(
        self,
        features: pd.DataFrame,
        actual_kwh: pd.Series,
        base_prediction_kwh: pd.Series,
        *,
        capacity_kwh: float,
    ) -> "ProbabilisticFICRDecision":
        """Fit only on the supplied rows; no fit-row score is calculated."""

        self._validate_config()
        feature_frame = _frame(features, name="fit_features")
        actual = _series(actual_kwh, name="fit_actual_kwh", allow_nan=True)
        base = _series(
            base_prediction_kwh,
            name="fit_base_prediction_kwh",
            allow_nan=False,
        )
        _require_aligned(feature_frame.index, actual.index, name="fit_actual_kwh")
        _require_aligned(
            feature_frame.index,
            base.index,
            name="fit_base_prediction_kwh",
        )
        capacity = float(capacity_kwh)
        if not np.isfinite(capacity) or capacity <= 0:
            raise ValueError("capacity_kwh must be positive and finite")

        actual_cf = actual.to_numpy(dtype=float) / capacity
        base_cf = base.to_numpy(dtype=float) / capacity
        eligible = np.isfinite(actual_cf) & (
            actual_cf >= self.config.minimum_actual_cf
        )
        if int(eligible.sum()) < self.config.minimum_fit_samples:
            raise ValueError(
                "too few eligible fit rows: "
                f"{int(eligible.sum())} < {self.config.minimum_fit_samples}"
            )

        self.capacity_kwh_ = capacity
        self.feature_columns_ = tuple(feature_frame.columns)
        self.fit_index_ = feature_frame.index.copy()
        self.fit_rows_ = len(feature_frame)
        self.eligible_fit_rows_ = int(eligible.sum())
        self.mean_actual_cf_ = float(np.mean(actual_cf[eligible]))
        self.global_residuals_ = actual_cf[eligible] - base_cf[eligible]

        x = feature_frame.to_numpy(dtype=float)[eligible]
        y = actual_cf[eligible]
        b = base_cf[eligible]
        if self.method == "global_conformal":
            self.global_samples_ = _thin_quantiles(
                self.global_residuals_, self.config.residual_sample_count
            )
        elif self.method == "binned_kde":
            self.global_samples_ = _kde_samples(self.global_residuals_, self.config)
            fit_bins = np.digitize(b, self.config.bin_edges_cf)
            self.bin_samples_: dict[int, np.ndarray] = {}
            self.bin_counts_: dict[int, int] = {}
            for bin_id in range(len(self.config.bin_edges_cf) + 1):
                values = self.global_residuals_[fit_bins == bin_id]
                self.bin_counts_[bin_id] = int(values.size)
                if values.size >= self.config.minimum_bin_samples:
                    self.bin_samples_[bin_id] = _kde_samples(values, self.config)
        elif self.method == "quantile_stack":
            leaf = min(
                self.config.quantile_min_samples_leaf,
                max(20, self.eligible_fit_rows_ // 8),
            )
            self.quantile_models_: dict[float, HistGradientBoostingRegressor] = {}
            for level in self.config.quantile_levels:
                model = HistGradientBoostingRegressor(
                    loss="quantile",
                    quantile=float(level),
                    learning_rate=0.05,
                    max_iter=self.config.quantile_max_iter,
                    max_leaf_nodes=7,
                    min_samples_leaf=leaf,
                    l2_regularization=10.0,
                    random_state=self.config.random_state,
                )
                model.fit(x, y)
                self.quantile_models_[float(level)] = model
        elif self.method == "threshold_meta":
            candidates = self._candidate_matrix(b)
            repeated_x = np.repeat(x, candidates.shape[1], axis=0)
            repeated_base = np.repeat(b, candidates.shape[1])
            flat_candidates = candidates.reshape(-1)
            design = np.column_stack(
                [
                    repeated_x,
                    flat_candidates,
                    flat_candidates - repeated_base,
                ]
            )
            repeated_y = np.repeat(y, candidates.shape[1])
            error = np.abs(flat_candidates - repeated_y)
            settlement = np.select(
                [error <= 0.06, error <= 0.08],
                [4.0, 3.0],
                default=0.0,
            )
            realised_utility = -0.5 * error + repeated_y * settlement / (
                8.0 * self.mean_actual_cf_
            )
            leaf = min(
                self.config.meta_min_samples_leaf,
                max(30, len(realised_utility) // 80),
            )
            self.meta_model_ = HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.05,
                max_iter=self.config.meta_max_iter,
                max_leaf_nodes=7,
                min_samples_leaf=leaf,
                l2_regularization=20.0,
                random_state=self.config.random_state,
            )
            self.meta_model_.fit(design, realised_utility)
        return self

    def _candidate_matrix(
        self,
        base_cf: np.ndarray,
        *,
        extra: np.ndarray | None = None,
    ) -> np.ndarray:
        offsets = np.asarray(self.config.candidate_offsets_cf, dtype=float)
        candidates = base_cf[:, None] + offsets[None, :]
        if extra is not None:
            candidates = np.column_stack([candidates, extra])
        return np.clip(candidates, self.config.lower_cf, self.config.upper_cf)

    @staticmethod
    def _argmax_near_base(utilities: np.ndarray, candidates: np.ndarray, base: float) -> int:
        maximum = float(np.max(utilities))
        tied = np.flatnonzero(utilities >= maximum - 1e-12)
        if tied.size == 1:
            return int(tied[0])
        distance = np.abs(candidates[tied] - base)
        return int(tied[int(np.argmin(distance))])

    def _sample_decisions(
        self,
        base_cf: np.ndarray,
        residual_samples: Sequence[np.ndarray],
        *,
        local_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        candidates = self._candidate_matrix(base_cf)
        decisions = np.empty(len(base_cf), dtype=float)
        for row in range(len(base_cf)):
            outcomes = np.clip(
                base_cf[row] + residual_samples[row],
                self.config.minimum_actual_cf,
                1.20,
            )
            utility = _utility_matrix(
                candidates[row], outcomes, self.mean_actual_cf_
            ).mean(axis=1)
            if local_weights is not None and local_weights[row] < 1.0:
                global_outcomes = np.clip(
                    base_cf[row] + self.global_samples_,
                    self.config.minimum_actual_cf,
                    1.20,
                )
                global_utility = _utility_matrix(
                    candidates[row], global_outcomes, self.mean_actual_cf_
                ).mean(axis=1)
                alpha = float(local_weights[row])
                utility = alpha * utility + (1.0 - alpha) * global_utility
            selected = self._argmax_near_base(
                utility, candidates[row], base_cf[row]
            )
            decisions[row] = candidates[row, selected]
        return decisions

    def _quantile_samples(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        levels = np.asarray(self.config.quantile_levels, dtype=float)
        predictions = np.column_stack(
            [self.quantile_models_[float(level)].predict(x) for level in levels]
        )
        predictions = np.sort(predictions, axis=1)
        probabilities = np.linspace(
            levels[0], levels[-1], self.config.quantile_sample_count
        )
        samples = np.empty((len(x), len(probabilities)), dtype=float)
        for row in range(len(x)):
            samples[row] = np.interp(probabilities, levels, predictions[row])
        return np.clip(samples, self.config.minimum_actual_cf, 1.20), predictions

    def predict(
        self,
        features: pd.DataFrame,
        base_prediction_kwh: pd.Series,
    ) -> pd.Series:
        """Apply to disjoint rows and return bounded kWh forecasts."""

        if not hasattr(self, "fit_index_"):
            raise RuntimeError("fit must be called before predict")
        feature_frame = _frame(features, name="application_features")
        if tuple(feature_frame.columns) != self.feature_columns_:
            raise ValueError(
                "application feature columns/order differ from fit columns"
            )
        base = _series(
            base_prediction_kwh,
            name="application_base_prediction_kwh",
            allow_nan=False,
        )
        _require_aligned(
            feature_frame.index,
            base.index,
            name="application_base_prediction_kwh",
        )
        overlap = feature_frame.index.intersection(self.fit_index_)
        if len(overlap):
            raise ValueError(
                "application rows overlap fit rows; same-row fit-score is forbidden "
                f"(first overlap: {overlap[0]!r})"
            )

        x = feature_frame.to_numpy(dtype=float)
        base_cf = base.to_numpy(dtype=float) / self.capacity_kwh_
        if self.method == "global_conformal":
            residuals = [self.global_samples_] * len(base_cf)
            raw = self._sample_decisions(base_cf, residuals)
        elif self.method == "binned_kde":
            bins = np.digitize(base_cf, self.config.bin_edges_cf)
            residuals: list[np.ndarray] = []
            weights = np.zeros(len(base_cf), dtype=float)
            for row, bin_id in enumerate(bins):
                count = self.bin_counts_.get(int(bin_id), 0)
                if int(bin_id) in self.bin_samples_:
                    residuals.append(self.bin_samples_[int(bin_id)])
                    weights[row] = count / (
                        count + self.config.bin_shrinkage_samples
                    )
                else:
                    residuals.append(self.global_samples_)
                    weights[row] = 0.0
            raw = self._sample_decisions(
                base_cf, residuals, local_weights=weights
            )
        elif self.method == "quantile_stack":
            samples, predicted_quantiles = self._quantile_samples(x)
            candidates = self._candidate_matrix(base_cf, extra=predicted_quantiles)
            raw = np.empty(len(base_cf), dtype=float)
            for row in range(len(base_cf)):
                utility = _utility_matrix(
                    candidates[row], samples[row], self.mean_actual_cf_
                ).mean(axis=1)
                selected = self._argmax_near_base(
                    utility, candidates[row], base_cf[row]
                )
                raw[row] = candidates[row, selected]
        else:
            candidates = self._candidate_matrix(base_cf)
            repeated_x = np.repeat(x, candidates.shape[1], axis=0)
            repeated_base = np.repeat(base_cf, candidates.shape[1])
            flat_candidates = candidates.reshape(-1)
            design = np.column_stack(
                [
                    repeated_x,
                    flat_candidates,
                    flat_candidates - repeated_base,
                ]
            )
            utility = self.meta_model_.predict(design).reshape(candidates.shape)
            raw = np.empty(len(base_cf), dtype=float)
            for row in range(len(base_cf)):
                selected = self._argmax_near_base(
                    utility[row], candidates[row], base_cf[row]
                )
                raw[row] = candidates[row, selected]

        shrunk = base_cf + self.config.decision_shrinkage * (raw - base_cf)
        bounded = np.clip(shrunk, self.config.lower_cf, self.config.upper_cf)
        return pd.Series(
            bounded * self.capacity_kwh_,
            index=feature_frame.index,
            name=base.name,
        )

    def metadata(self) -> dict[str, Any]:
        """Return JSON-safe fit metadata without any in-sample score."""

        if not hasattr(self, "fit_index_"):
            raise RuntimeError("fit must be called before metadata")
        result: dict[str, Any] = {
            "method": self.method,
            "config": asdict(self.config),
            "fit_rows": self.fit_rows_,
            "eligible_fit_rows": self.eligible_fit_rows_,
            "mean_actual_cf": self.mean_actual_cf_,
            "feature_columns": list(self.feature_columns_),
            "fit_index_min": str(self.fit_index_.min()),
            "fit_index_max": str(self.fit_index_.max()),
            "fit_score_calculated": False,
        }
        if self.method == "binned_kde":
            result["bin_counts"] = {
                str(key): value for key, value in self.bin_counts_.items()
            }
        return result


__all__ = (
    "DecisionMethod",
    "ProbabilisticDecisionConfig",
    "ProbabilisticFICRDecision",
)

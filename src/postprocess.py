"""Leakage-safe calibration, ensembling, and bounded post-processing.

Every fitted component has separate fit and application methods. By default,
application rows may not overlap the index used to choose parameters. This is
intentional: reporting a score on the same OOF labels used to tune an affine
map, ensemble weights, or FICR thresholds would be an optimistic score.

All searches use the official group objective: one half one-minus-NMAE and one
half energy-weighted FICR. Group-level optimization is valid because the final
competition score averages groups equally.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .metric import CAPACITY_KWH, TARGET_COLS, group_metrics


@dataclass(frozen=True)
class AffineParameter:
    """One group's affine map: output = scale * prediction + bias_kwh."""

    scale: float
    bias_kwh: float


@dataclass(frozen=True)
class BoundedParameter:
    """One group's bounded FICR-aware map."""

    scale: float
    bias_fraction: float
    lower_capacity_fraction: float
    upper_capacity_fraction: float


@dataclass(frozen=True)
class SearchDiagnostic:
    """Before/after official group scores for a fitted transform."""

    base_score: float
    fitted_score: float
    base_one_minus_nmae: float
    fitted_one_minus_nmae: float
    base_ficr: float
    fitted_ficr: float


def _resolve_groups_and_capacities(
    groups: Sequence[str],
    capacities: Mapping[str, float] | None,
) -> tuple[tuple[str, ...], dict[str, float]]:
    result = tuple(groups)
    if not result:
        raise ValueError("groups must contain at least one target")
    if len(set(result)) != len(result):
        raise ValueError("groups contains duplicate target names")
    supplied = CAPACITY_KWH if capacities is None else capacities
    resolved: dict[str, float] = {}
    for group in result:
        if group not in supplied:
            raise ValueError(f"capacities is missing group {group!r}")
        capacity = float(supplied[group])
        if not np.isfinite(capacity) or capacity <= 0:
            raise ValueError(f"capacity for {group!r} must be positive and finite")
        resolved[group] = capacity
    return result, resolved


def _validate_frame(
    frame: Any,
    *,
    name: str,
    groups: Sequence[str],
    require_finite: bool,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if frame.empty:
        raise ValueError(f"{name} must not be empty")
    if not frame.index.is_unique:
        raise ValueError(f"{name} index must be unique")
    if not frame.index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be monotonically increasing")
    if not frame.columns.is_unique:
        raise ValueError(f"{name} has duplicate columns")
    expected = tuple(groups)
    missing = [group for group in expected if group not in frame.columns]
    unexpected = [column for column in frame.columns if column not in expected]
    if missing or unexpected:
        raise ValueError(
            f"{name} groups differ from configured groups; "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )
    result = frame.loc[:, list(expected)].astype(float)
    if require_finite and not np.isfinite(result.to_numpy()).all():
        raise ValueError(f"{name} contains NaN or infinite predictions")
    return result


def _validate_fit_pair(
    actual: Any,
    prediction: Any,
    *,
    groups: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    actual_frame = _validate_frame(
        actual,
        name="calibration_actual",
        groups=groups,
        require_finite=False,
    )
    prediction_frame = _validate_frame(
        prediction,
        name="calibration_prediction",
        groups=groups,
        require_finite=True,
    )
    if not actual_frame.index.equals(prediction_frame.index):
        raise ValueError(
            "calibration_actual and calibration_prediction indexes differ"
        )
    actual_values = actual_frame.to_numpy()
    if np.isinf(actual_values).any():
        raise ValueError("calibration_actual contains infinite labels")
    return actual_frame, prediction_frame


def _validate_prediction_map(
    predictions: Any,
    *,
    groups: Sequence[str],
    expected_models: Sequence[str] | None = None,
    name: str,
) -> tuple[tuple[str, ...], dict[str, pd.DataFrame], pd.Index]:
    if not isinstance(predictions, Mapping) or not predictions:
        raise TypeError(f"{name} must be a non-empty mapping of model to DataFrame")
    model_names = tuple(str(model) for model in predictions)
    if len(set(model_names)) != len(model_names):
        raise ValueError(f"{name} model names are not unique after string conversion")
    if expected_models is not None and model_names != tuple(expected_models):
        raise ValueError(
            f"{name} models/order differ from locked models; "
            f"expected={tuple(expected_models)!r}, received={model_names!r}"
        )
    frames: dict[str, pd.DataFrame] = {}
    reference_index: pd.Index | None = None
    for original_name, model_name in zip(predictions, model_names):
        frame = _validate_frame(
            predictions[original_name],
            name=f"{name}[{model_name!r}]",
            groups=groups,
            require_finite=True,
        )
        if reference_index is None:
            reference_index = frame.index
        elif not reference_index.equals(frame.index):
            raise ValueError(f"{name}[{model_name!r}] index is not aligned")
        frames[model_name] = frame
    assert reference_index is not None
    return model_names, frames, reference_index


def _official_group_tuple(
    actual: Any,
    prediction: Any,
    capacity: float,
    group: str,
) -> tuple[float, float, float]:
    details = group_metrics(
        actual,
        prediction,
        capacity,
        group_name=group,
    )
    score = 0.5 * details.one_minus_nmae + 0.5 * details.ficr
    return score, details.one_minus_nmae, details.ficr


def _is_better(
    candidate: tuple[float, float, float],
    incumbent: tuple[float, float, float],
    *,
    candidate_distance: float,
    incumbent_distance: float,
) -> bool:
    tolerance = 1e-14
    if candidate[0] > incumbent[0] + tolerance:
        return True
    if abs(candidate[0] - incumbent[0]) <= tolerance:
        if candidate[1] > incumbent[1] + tolerance:
            return True
        if (
            abs(candidate[1] - incumbent[1]) <= tolerance
            and candidate_distance < incumbent_distance - tolerance
        ):
            return True
    return False


def _application_index_guard(
    application_index: pd.Index,
    fit_index: pd.Index | None,
    *,
    allow_fit_index: bool,
) -> None:
    if fit_index is None or allow_fit_index:
        return
    overlap = application_index.intersection(fit_index)
    if len(overlap):
        example = overlap[0]
        raise ValueError(
            "application rows overlap parameter-fit rows "
            f"(first overlap: {example!r}); use a disjoint application frame. "
            "allow_fit_index=True is only for explicitly labelled diagnostics"
        )


class GroupAffineCalibrator:
    """Fit a bounded group-specific affine map on calibration OOF only."""

    def __init__(
        self,
        *,
        groups: Sequence[str] = TARGET_COLS,
        capacities: Mapping[str, float] | None = None,
        scale_bounds: tuple[float, float] = (0.85, 1.25),
        bias_fraction_bounds: tuple[float, float] = (-0.10, 0.10),
        grid_size: int = 31,
        refinements: int = 2,
    ) -> None:
        self.groups = tuple(groups)
        self.capacities = capacities
        self.scale_bounds = scale_bounds
        self.bias_fraction_bounds = bias_fraction_bounds
        self.grid_size = grid_size
        self.refinements = refinements

    def _validate_options(self) -> None:
        self.groups_, self.capacities_ = _resolve_groups_and_capacities(
            self.groups, self.capacities
        )
        scale_low, scale_high = map(float, self.scale_bounds)
        bias_low, bias_high = map(float, self.bias_fraction_bounds)
        if not 0 < scale_low <= 1.0 <= scale_high:
            raise ValueError("scale_bounds must be positive and include 1.0")
        if not bias_low <= 0.0 <= bias_high:
            raise ValueError("bias_fraction_bounds must include 0.0")
        if int(self.grid_size) < 3:
            raise ValueError("grid_size must be at least 3")
        if int(self.refinements) < 0:
            raise ValueError("refinements must be non-negative")

    def fit(
        self,
        calibration_actual: pd.DataFrame,
        calibration_prediction: pd.DataFrame,
    ) -> "GroupAffineCalibrator":
        """Choose affine parameters using only the named calibration rows."""

        self._validate_options()
        actual, prediction = _validate_fit_pair(
            calibration_actual,
            calibration_prediction,
            groups=self.groups_,
        )
        self.fit_index_ = actual.index.copy()
        self.parameters_: dict[str, AffineParameter] = {}
        self.diagnostics_: dict[str, SearchDiagnostic] = {}

        for group in self.groups_:
            capacity = self.capacities_[group]
            y = actual[group].to_numpy(dtype=float)
            p = prediction[group].to_numpy(dtype=float)
            base = _official_group_tuple(y, p, capacity, group)
            best = base
            best_scale = 1.0
            best_bias = 0.0
            best_distance = 0.0
            scale_low, scale_high = map(float, self.scale_bounds)
            bias_low = float(self.bias_fraction_bounds[0]) * capacity
            bias_high = float(self.bias_fraction_bounds[1]) * capacity
            scales = np.unique(
                np.append(np.linspace(scale_low, scale_high, int(self.grid_size)), 1.0)
            )
            biases = np.unique(
                np.append(np.linspace(bias_low, bias_high, int(self.grid_size)), 0.0)
            )

            for refinement in range(int(self.refinements) + 1):
                for scale, bias in product(scales, biases):
                    candidate = _official_group_tuple(
                        y, scale * p + bias, capacity, group
                    )
                    distance = abs(scale - 1.0) + abs(bias) / capacity
                    if _is_better(
                        candidate,
                        best,
                        candidate_distance=distance,
                        incumbent_distance=best_distance,
                    ):
                        best = candidate
                        best_scale = float(scale)
                        best_bias = float(bias)
                        best_distance = distance
                if refinement == int(self.refinements):
                    break
                scale_step = (scales[-1] - scales[0]) / max(len(scales) - 1, 1)
                bias_step = (biases[-1] - biases[0]) / max(len(biases) - 1, 1)
                scales = np.linspace(
                    max(scale_low, best_scale - scale_step),
                    min(scale_high, best_scale + scale_step),
                    11,
                )
                biases = np.linspace(
                    max(bias_low, best_bias - bias_step),
                    min(bias_high, best_bias + bias_step),
                    11,
                )

            self.parameters_[group] = AffineParameter(best_scale, best_bias)
            self.diagnostics_[group] = SearchDiagnostic(
                base_score=base[0],
                fitted_score=best[0],
                base_one_minus_nmae=base[1],
                fitted_one_minus_nmae=best[1],
                base_ficr=base[2],
                fitted_ficr=best[2],
            )
        return self

    @classmethod
    def from_parameters(
        cls,
        parameters: Mapping[str, AffineParameter | Sequence[float]],
        *,
        capacities: Mapping[str, float] | None = None,
    ) -> "GroupAffineCalibrator":
        """Create an immutable application-only calibrator from locked values."""

        groups = tuple(parameters)
        result = cls(groups=groups, capacities=capacities)
        result.groups_, result.capacities_ = _resolve_groups_and_capacities(
            groups, capacities
        )
        result.parameters_ = {}
        for group in groups:
            value = parameters[group]
            if isinstance(value, AffineParameter):
                parameter = value
            else:
                if len(value) != 2:
                    raise ValueError(
                        f"affine parameter for {group!r} must have scale and bias"
                    )
                parameter = AffineParameter(float(value[0]), float(value[1]))
            if not np.isfinite([parameter.scale, parameter.bias_kwh]).all():
                raise ValueError(f"affine parameter for {group!r} is not finite")
            if parameter.scale <= 0:
                raise ValueError(f"affine scale for {group!r} must be positive")
            result.parameters_[group] = parameter
        result.fit_index_ = None
        result.diagnostics_ = {}
        return result

    def transform(
        self,
        application_prediction: pd.DataFrame,
        *,
        allow_fit_index: bool = False,
    ) -> pd.DataFrame:
        """Apply locked parameters to a separate prediction frame."""

        if not hasattr(self, "parameters_"):
            raise RuntimeError("calibrator is not fitted")
        prediction = _validate_frame(
            application_prediction,
            name="application_prediction",
            groups=self.groups_,
            require_finite=True,
        )
        _application_index_guard(
            prediction.index,
            self.fit_index_,
            allow_fit_index=allow_fit_index,
        )
        output = prediction.copy()
        for group in self.groups_:
            parameter = self.parameters_[group]
            output[group] = (
                parameter.scale * prediction[group] + parameter.bias_kwh
            )
        return output

    def parameters_as_dict(self) -> dict[str, dict[str, float]]:
        """Return JSON-friendly locked parameters."""

        if not hasattr(self, "parameters_"):
            raise RuntimeError("calibrator is not fitted")
        return {group: asdict(value) for group, value in self.parameters_.items()}


class SimplexEnsembler:
    """Search non-negative, sum-to-one ensemble weights on calibration OOF."""

    def __init__(
        self,
        *,
        groups: Sequence[str] = TARGET_COLS,
        capacities: Mapping[str, float] | None = None,
        model_names: Sequence[str] | None = None,
        per_group: bool = True,
        n_random: int = 1500,
        refinement_steps: Sequence[float] = (0.10, 0.05, 0.02, 0.01, 0.005),
        seed: int = 42,
    ) -> None:
        self.groups = tuple(groups)
        self.capacities = capacities
        self.model_names = None if model_names is None else tuple(model_names)
        self.per_group = per_group
        self.n_random = n_random
        self.refinement_steps = tuple(refinement_steps)
        self.seed = seed

    def _validate_options(self) -> None:
        self.groups_, self.capacities_ = _resolve_groups_and_capacities(
            self.groups, self.capacities
        )
        if int(self.n_random) < 0:
            raise ValueError("n_random must be non-negative")
        if any(not 0 < float(step) <= 1 for step in self.refinement_steps):
            raise ValueError("refinement_steps must be in (0, 1]")

    def _score_weights(
        self,
        weights: np.ndarray,
        matrices: Mapping[str, np.ndarray],
        actual: pd.DataFrame,
        groups: Sequence[str],
    ) -> tuple[float, float, float]:
        metrics = [
            _official_group_tuple(
                actual[group].to_numpy(dtype=float),
                matrices[group] @ weights,
                self.capacities_[group],
                group,
            )
            for group in groups
        ]
        return tuple(float(np.mean([value[i] for value in metrics])) for i in range(3))

    def _search(
        self,
        matrices: Mapping[str, np.ndarray],
        actual: pd.DataFrame,
        groups: Sequence[str],
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, tuple[float, float, float]]:
        n_models = len(self.model_names_)
        equal = np.full(n_models, 1.0 / n_models)
        candidates = [equal]
        candidates.extend(np.eye(n_models))
        if int(self.n_random):
            candidates.extend(
                rng.dirichlet(np.ones(n_models), size=int(self.n_random))
            )
        best_weights = equal
        best = self._score_weights(equal, matrices, actual, groups)
        best_distance = 0.0
        for candidate in candidates:
            score = self._score_weights(candidate, matrices, actual, groups)
            distance = float(np.sum(np.abs(candidate - equal)))
            if _is_better(
                score,
                best,
                candidate_distance=distance,
                incumbent_distance=best_distance,
            ):
                best = score
                best_weights = np.asarray(candidate, dtype=float).copy()
                best_distance = distance

        for step in map(float, self.refinement_steps):
            improved = True
            while improved:
                improved = False
                for donor in range(n_models):
                    if best_weights[donor] + 1e-15 < step:
                        continue
                    for receiver in range(n_models):
                        if receiver == donor:
                            continue
                        candidate = best_weights.copy()
                        candidate[donor] -= step
                        candidate[receiver] += step
                        score = self._score_weights(
                            candidate, matrices, actual, groups
                        )
                        distance = float(np.sum(np.abs(candidate - equal)))
                        if _is_better(
                            score,
                            best,
                            candidate_distance=distance,
                            incumbent_distance=best_distance,
                        ):
                            best = score
                            best_weights = candidate
                            best_distance = distance
                            improved = True
                            break
                    if improved:
                        break
        return best_weights, best

    def fit(
        self,
        calibration_actual: pd.DataFrame,
        calibration_predictions: Mapping[str, pd.DataFrame],
    ) -> "SimplexEnsembler":
        """Choose simplex weights from explicitly named calibration OOF."""

        self._validate_options()
        actual = _validate_frame(
            calibration_actual,
            name="calibration_actual",
            groups=self.groups_,
            require_finite=False,
        )
        names, frames, index = _validate_prediction_map(
            calibration_predictions,
            groups=self.groups_,
            expected_models=self.model_names,
            name="calibration_predictions",
        )
        if not actual.index.equals(index):
            raise ValueError(
                "calibration_actual and calibration_predictions indexes differ"
            )
        if np.isinf(actual.to_numpy()).any():
            raise ValueError("calibration_actual contains infinite labels")
        self.model_names_ = names
        self.fit_index_ = actual.index.copy()
        matrices = {
            group: np.column_stack(
                [frames[model][group].to_numpy(dtype=float) for model in names]
            )
            for group in self.groups_
        }
        rng = np.random.default_rng(int(self.seed))
        weight_rows: dict[str, np.ndarray] = {}
        self.search_scores_: dict[str, tuple[float, float, float]] = {}
        if self.per_group:
            for group in self.groups_:
                weights, score = self._search(
                    matrices, actual, (group,), rng
                )
                weight_rows[group] = weights
                self.search_scores_[group] = score
        else:
            weights, score = self._search(
                matrices, actual, self.groups_, rng
            )
            for group in self.groups_:
                weight_rows[group] = weights.copy()
                self.search_scores_[group] = score
        self.weights_ = pd.DataFrame.from_dict(
            weight_rows,
            orient="index",
            columns=self.model_names_,
            dtype=float,
        ).loc[list(self.groups_), list(self.model_names_)]
        return self

    @classmethod
    def from_weights(
        cls,
        weights: Mapping[str, Mapping[str, float]],
        *,
        capacities: Mapping[str, float] | None = None,
    ) -> "SimplexEnsembler":
        """Create an application-only ensemble from locked simplex weights."""

        if not weights:
            raise ValueError("weights must not be empty")
        groups = tuple(weights)
        first_group = groups[0]
        model_names = tuple(weights[first_group])
        if not model_names:
            raise ValueError("weights must contain at least one model")
        result = cls(
            groups=groups,
            capacities=capacities,
            model_names=model_names,
        )
        result.groups_, result.capacities_ = _resolve_groups_and_capacities(
            groups, capacities
        )
        rows: dict[str, list[float]] = {}
        for group in groups:
            if tuple(weights[group]) != model_names:
                raise ValueError(
                    f"weight model names/order for {group!r} differ from "
                    f"{first_group!r}"
                )
            row = np.asarray([weights[group][name] for name in model_names], dtype=float)
            if not np.isfinite(row).all() or np.any(row < 0):
                raise ValueError(f"weights for {group!r} must be finite and non-negative")
            if not np.isclose(row.sum(), 1.0, rtol=0, atol=1e-10):
                raise ValueError(f"weights for {group!r} must sum to one")
            rows[group] = row.tolist()
        result.model_names_ = model_names
        result.weights_ = pd.DataFrame.from_dict(
            rows, orient="index", columns=model_names, dtype=float
        )
        result.fit_index_ = None
        result.search_scores_ = {}
        return result

    def predict(
        self,
        application_predictions: Mapping[str, pd.DataFrame],
        *,
        allow_fit_index: bool = False,
    ) -> pd.DataFrame:
        """Blend a separate application prediction mapping."""

        if not hasattr(self, "weights_"):
            raise RuntimeError("ensembler is not fitted")
        _, frames, index = _validate_prediction_map(
            application_predictions,
            groups=self.groups_,
            expected_models=self.model_names_,
            name="application_predictions",
        )
        _application_index_guard(
            index, self.fit_index_, allow_fit_index=allow_fit_index
        )
        output = pd.DataFrame(index=index, columns=self.groups_, dtype=float)
        for group in self.groups_:
            matrix = np.column_stack(
                [
                    frames[model][group].to_numpy(dtype=float)
                    for model in self.model_names_
                ]
            )
            output[group] = matrix @ self.weights_.loc[group].to_numpy(dtype=float)
        return output

    def weights_as_dict(self) -> dict[str, dict[str, float]]:
        """Return JSON-friendly locked weights."""

        if not hasattr(self, "weights_"):
            raise RuntimeError("ensembler is not fitted")
        return {
            group: {
                model: float(self.weights_.loc[group, model])
                for model in self.model_names_
            }
            for group in self.groups_
        }


class FICRBoundedPostprocessor:
    """Tune a small bounded map against the official FICR-aware group score."""

    def __init__(
        self,
        *,
        groups: Sequence[str] = TARGET_COLS,
        capacities: Mapping[str, float] | None = None,
        scales: Sequence[float] = (0.98, 0.99, 1.0, 1.01, 1.02),
        bias_fractions: Sequence[float] = (
            -0.02,
            -0.01,
            -0.005,
            0.0,
            0.005,
            0.01,
            0.02,
        ),
        lower_capacity_fractions: Sequence[float] = (0.0,),
        upper_capacity_fractions: Sequence[float] = (1.0, 1.02, 1.05),
    ) -> None:
        self.groups = tuple(groups)
        self.capacities = capacities
        self.scales = tuple(scales)
        self.bias_fractions = tuple(bias_fractions)
        self.lower_capacity_fractions = tuple(lower_capacity_fractions)
        self.upper_capacity_fractions = tuple(upper_capacity_fractions)

    def _validate_options(self) -> None:
        self.groups_, self.capacities_ = _resolve_groups_and_capacities(
            self.groups, self.capacities
        )
        if not self.scales or not self.bias_fractions:
            raise ValueError("scales and bias_fractions must not be empty")
        if not self.lower_capacity_fractions or not self.upper_capacity_fractions:
            raise ValueError("bound grids must not be empty")
        if any(not np.isfinite(value) or value <= 0 for value in self.scales):
            raise ValueError("scales must be positive and finite")
        if any(not np.isfinite(value) for value in self.bias_fractions):
            raise ValueError("bias_fractions must be finite")
        for lower, upper in product(
            self.lower_capacity_fractions, self.upper_capacity_fractions
        ):
            if not np.isfinite([lower, upper]).all() or lower < 0 or upper <= lower:
                raise ValueError(
                    "capacity bounds must be finite and satisfy 0 <= lower < upper"
                )

    def fit(
        self,
        calibration_actual: pd.DataFrame,
        calibration_prediction: pd.DataFrame,
    ) -> "FICRBoundedPostprocessor":
        """Choose the bounded map using only calibration OOF labels."""

        self._validate_options()
        actual, prediction = _validate_fit_pair(
            calibration_actual,
            calibration_prediction,
            groups=self.groups_,
        )
        self.fit_index_ = actual.index.copy()
        self.parameters_: dict[str, BoundedParameter] = {}
        self.diagnostics_: dict[str, SearchDiagnostic] = {}
        for group in self.groups_:
            capacity = self.capacities_[group]
            y = actual[group].to_numpy(dtype=float)
            p = prediction[group].to_numpy(dtype=float)
            base = _official_group_tuple(y, p, capacity, group)
            best: tuple[float, float, float] | None = None
            best_parameter = BoundedParameter(1.0, 0.0, 0.0, 1.02)
            best_distance = float("inf")
            for scale, bias, lower, upper in product(
                self.scales,
                self.bias_fractions,
                self.lower_capacity_fractions,
                self.upper_capacity_fractions,
            ):
                transformed = np.clip(
                    float(scale) * p + float(bias) * capacity,
                    float(lower) * capacity,
                    float(upper) * capacity,
                )
                candidate = _official_group_tuple(
                    y, transformed, capacity, group
                )
                distance = (
                    abs(float(scale) - 1.0)
                    + abs(float(bias))
                    + abs(float(lower))
                    + abs(float(upper) - 1.02)
                )
                if best is None or _is_better(
                    candidate,
                    best,
                    candidate_distance=distance,
                    incumbent_distance=best_distance,
                ):
                    best = candidate
                    best_parameter = BoundedParameter(
                        float(scale),
                        float(bias),
                        float(lower),
                        float(upper),
                    )
                    best_distance = distance
            assert best is not None
            self.parameters_[group] = best_parameter
            self.diagnostics_[group] = SearchDiagnostic(
                base_score=base[0],
                fitted_score=best[0],
                base_one_minus_nmae=base[1],
                fitted_one_minus_nmae=best[1],
                base_ficr=base[2],
                fitted_ficr=best[2],
            )
        return self

    @classmethod
    def from_parameters(
        cls,
        parameters: Mapping[str, BoundedParameter | Sequence[float]],
        *,
        capacities: Mapping[str, float] | None = None,
    ) -> "FICRBoundedPostprocessor":
        """Create an application-only processor from locked values."""

        groups = tuple(parameters)
        result = cls(groups=groups, capacities=capacities)
        result.groups_, result.capacities_ = _resolve_groups_and_capacities(
            groups, capacities
        )
        result.parameters_ = {}
        for group in groups:
            value = parameters[group]
            if isinstance(value, BoundedParameter):
                parameter = value
            else:
                if len(value) != 4:
                    raise ValueError(
                        f"bounded parameter for {group!r} needs four values"
                    )
                parameter = BoundedParameter(*map(float, value))
            values = np.asarray(
                [
                    parameter.scale,
                    parameter.bias_fraction,
                    parameter.lower_capacity_fraction,
                    parameter.upper_capacity_fraction,
                ]
            )
            if not np.isfinite(values).all():
                raise ValueError(f"bounded parameter for {group!r} is not finite")
            if (
                parameter.scale <= 0
                or parameter.lower_capacity_fraction < 0
                or parameter.upper_capacity_fraction
                <= parameter.lower_capacity_fraction
            ):
                raise ValueError(f"bounded parameter for {group!r} is invalid")
            result.parameters_[group] = parameter
        result.fit_index_ = None
        result.diagnostics_ = {}
        return result

    def transform(
        self,
        application_prediction: pd.DataFrame,
        *,
        allow_fit_index: bool = False,
    ) -> pd.DataFrame:
        """Apply the locked bounded map to separate application rows."""

        if not hasattr(self, "parameters_"):
            raise RuntimeError("postprocessor is not fitted")
        prediction = _validate_frame(
            application_prediction,
            name="application_prediction",
            groups=self.groups_,
            require_finite=True,
        )
        _application_index_guard(
            prediction.index,
            self.fit_index_,
            allow_fit_index=allow_fit_index,
        )
        output = prediction.copy()
        for group in self.groups_:
            parameter = self.parameters_[group]
            capacity = self.capacities_[group]
            output[group] = np.clip(
                parameter.scale * prediction[group]
                + parameter.bias_fraction * capacity,
                parameter.lower_capacity_fraction * capacity,
                parameter.upper_capacity_fraction * capacity,
            )
        return output

    def parameters_as_dict(self) -> dict[str, dict[str, float]]:
        """Return JSON-friendly locked parameters."""

        if not hasattr(self, "parameters_"):
            raise RuntimeError("postprocessor is not fitted")
        return {group: asdict(value) for group, value in self.parameters_.items()}


__all__ = [
    "AffineParameter",
    "BoundedParameter",
    "FICRBoundedPostprocessor",
    "GroupAffineCalibrator",
    "SearchDiagnostic",
    "SimplexEnsembler",
]

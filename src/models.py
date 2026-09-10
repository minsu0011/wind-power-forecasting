"""Leakage-conscious wrappers for tabular regression models.

TabularRegressor presents one fit/predict interface across LightGBM, XGBoost,
CatBoost, ExtraTrees, and HistGradientBoosting. Early stopping is enabled only
when fit receives an explicit inner_eval_set. That set should be a later slice
drawn solely from the outer training window, never the outer score fold.

Optional boosting-library imports are lazy, so sklearn models remain usable
when one of those packages is unavailable.
"""

from __future__ import annotations

from collections.abc import Mapping
import os
from typing import Any
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.inspection import permutation_importance
from sklearn.utils.validation import check_is_fitted


SUPPORTED_MODELS: tuple[str, ...] = (
    "lgbm_l1",
    "lgbm_huber",
    "lgbm_tweedie",
    "lgbm_regression",
    "xgb_l1",
    "xgb_pseudohuber",
    "catboost_mae",
    "catboost_huber",
    "catboost_rmse",
    "extra_trees",
    "extra_trees_l1",
    "hist_gbdt",
    "hist_gbdt_l1",
)

_ALIASES: dict[str, str] = {
    "lightgbm_l1": "lgbm_l1",
    "lightgbm_mae": "lgbm_l1",
    "lightgbm_huber": "lgbm_huber",
    "lightgbm_tweedie": "lgbm_tweedie",
    "lightgbm_regression": "lgbm_regression",
    "lgbm_mae": "lgbm_l1",
    "xgboost_l1": "xgb_l1",
    "xgboost_mae": "xgb_l1",
    "xgboost_absolute": "xgb_l1",
    "xgboost_pseudohuber": "xgb_pseudohuber",
    "xgb_mae": "xgb_l1",
    "cat_mae": "catboost_mae",
    "cat_huber": "catboost_huber",
    "cat_rmse": "catboost_rmse",
    "extratrees": "extra_trees",
    "extratrees_l1": "extra_trees_l1",
    "hist_gradient_boosting": "hist_gbdt",
    "hist_gradient_boosting_l1": "hist_gbdt_l1",
    "histgb": "hist_gbdt",
    "histgb_l1": "hist_gbdt_l1",
}


class DeviceFallbackWarning(UserWarning):
    """A requested GPU fit was retried safely on the CPU."""


def available_models() -> tuple[str, ...]:
    """Return canonical model names accepted by TabularRegressor."""

    return SUPPORTED_MODELS


def _canonical_kind(kind: str) -> str:
    value = str(kind).strip().lower().replace("-", "_")
    value = _ALIASES.get(value, value)
    if value not in SUPPORTED_MODELS:
        choices = ", ".join(SUPPORTED_MODELS)
        raise ValueError(f"unknown model kind {kind!r}; choose one of: {choices}")
    return value


def _require_frame(X: Any, *, name: str = "X") -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if X.empty:
        raise ValueError(f"{name} must contain at least one row and one column")
    if not X.columns.is_unique:
        raise ValueError(f"{name} contains duplicate column names")
    if not X.index.is_unique:
        raise ValueError(f"{name} index must be unique")
    return X


def _as_1d_target(values: Any, *, name: str, index: pd.Index) -> np.ndarray:
    if isinstance(values, pd.Series) and not values.index.equals(index):
        raise ValueError(f"{name} index is not exactly aligned with X")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got {array.shape}")
    if len(array) != len(index):
        raise ValueError(
            f"{name} and X have different row counts: {len(array)} != {len(index)}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def _as_weight(values: Any, *, name: str, index: pd.Index) -> np.ndarray | None:
    if values is None:
        return None
    if isinstance(values, pd.Series) and not values.index.equals(index):
        raise ValueError(f"{name} index is not exactly aligned with X")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) != len(index):
        raise ValueError(f"{name} must be one-dimensional and have len(X) values")
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError(f"{name} must contain finite, non-negative values")
    if not np.any(array > 0):
        raise ValueError(f"{name} must contain at least one positive value")
    return array


class TabularRegressor(RegressorMixin, BaseEstimator):
    """Common fit/predict API for all regression candidates.

    device may be cpu, gpu, or auto. CPU is the reproducible default. GPU and
    auto attempt the requested backend and optionally rebuild on CPU after an
    error. sklearn candidates are CPU-only. early_stopping_rounds is used only
    with an explicitly supplied inner_eval_set. DatetimeIndex inputs enforce a
    disjoint, strictly later inner set by default.
    """

    def __init__(
        self,
        kind: str = "lgbm_l1",
        *,
        params: Mapping[str, Any] | None = None,
        seed: int = 42,
        n_jobs: int | None = None,
        device: str = "cpu",
        early_stopping_rounds: int | None = 100,
        fallback_to_cpu: bool = True,
        validate_time_order: bool = True,
    ) -> None:
        self.kind = kind
        self.params = params
        self.seed = seed
        self.n_jobs = n_jobs
        self.device = device
        self.early_stopping_rounds = early_stopping_rounds
        self.fallback_to_cpu = fallback_to_cpu
        self.validate_time_order = validate_time_order

    def _resolved_n_jobs(self) -> int:
        if self.n_jobs is None:
            return max(1, min(8, os.cpu_count() or 1))
        value = int(self.n_jobs)
        if value == 0:
            raise ValueError("n_jobs must not be zero")
        return value

    def _validate_options(self) -> None:
        self.kind_ = _canonical_kind(self.kind)
        self.device_ = str(self.device).strip().lower()
        if self.device_ not in {"cpu", "gpu", "auto"}:
            raise ValueError("device must be cpu, gpu, or auto")
        self.n_jobs_ = self._resolved_n_jobs()
        if self.early_stopping_rounds is not None:
            if int(self.early_stopping_rounds) <= 0:
                raise ValueError("early_stopping_rounds must be positive or None")
        if self.params is not None and not isinstance(self.params, Mapping):
            raise TypeError("params must be a mapping or None")

    def _fit_encoder(self, X: pd.DataFrame) -> pd.DataFrame:
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        self._internal_columns_ = [
            f"f{position:04d}" for position in range(X.shape[1])
        ]
        self._feature_kinds_: list[str] = []
        self._categories_: list[pd.Index | None] = []
        encoded: dict[str, np.ndarray] = {}

        for position, (_, values) in enumerate(X.items()):
            output_name = self._internal_columns_[position]
            if pd.api.types.is_datetime64_any_dtype(values.dtype):
                raw = values.astype("int64").to_numpy(dtype=np.float64, copy=True)
                raw[values.isna().to_numpy()] = np.nan
                encoded[output_name] = raw / 3_600_000_000_000.0
                self._feature_kinds_.append("datetime")
                self._categories_.append(None)
            elif pd.api.types.is_numeric_dtype(values.dtype):
                encoded[output_name] = pd.to_numeric(
                    values, errors="coerce"
                ).to_numpy(dtype=np.float64, copy=True)
                self._feature_kinds_.append("numeric")
                self._categories_.append(None)
            else:
                non_missing = values.loc[~values.isna()]
                categories = pd.Index(pd.unique(non_missing), dtype=object)
                codes = pd.Categorical(values, categories=categories).codes.astype(
                    np.float64
                )
                codes[codes < 0] = np.nan
                encoded[output_name] = codes
                self._feature_kinds_.append("categorical")
                self._categories_.append(categories)

        frame = pd.DataFrame(encoded, index=X.index, dtype=np.float64)
        frame.replace([np.inf, -np.inf], np.nan, inplace=True)
        self._imputation_values_ = (
            frame.median(axis=0, skipna=True).fillna(0.0).astype(np.float64)
        )
        return frame.astype(np.float32)

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = _require_frame(X)
        expected = list(self.feature_names_in_)
        missing = [column for column in expected if column not in X.columns]
        unexpected = [column for column in X.columns if column not in expected]
        if missing or unexpected:
            raise ValueError(
                "prediction columns differ from fit columns; "
                f"missing={missing[:5]!r}, unexpected={unexpected[:5]!r}"
            )
        X = X.loc[:, expected]
        encoded: dict[str, np.ndarray] = {}

        for position, (_, values) in enumerate(X.items()):
            output_name = self._internal_columns_[position]
            feature_kind = self._feature_kinds_[position]
            if feature_kind == "datetime":
                converted = pd.to_datetime(values, errors="coerce")
                raw = converted.astype("int64").to_numpy(
                    dtype=np.float64, copy=True
                )
                raw[converted.isna().to_numpy()] = np.nan
                encoded[output_name] = raw / 3_600_000_000_000.0
            elif feature_kind == "numeric":
                encoded[output_name] = pd.to_numeric(
                    values, errors="coerce"
                ).to_numpy(dtype=np.float64, copy=True)
            else:
                categories = self._categories_[position]
                codes = pd.Categorical(values, categories=categories).codes.astype(
                    np.float64
                )
                codes[codes < 0] = np.nan
                encoded[output_name] = codes

        frame = pd.DataFrame(encoded, index=X.index, dtype=np.float64)
        frame.replace([np.inf, -np.inf], np.nan, inplace=True)
        return frame.astype(np.float32)

    def _prepare_backend_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.kind_.startswith("extra_trees"):
            return X.fillna(self._imputation_values_).astype(np.float32)
        return X

    def _validate_inner_time_order(
        self, train_index: pd.Index, inner_index: pd.Index
    ) -> None:
        if not self.validate_time_order:
            return
        if isinstance(train_index, pd.DatetimeIndex) and isinstance(
            inner_index, pd.DatetimeIndex
        ):
            if train_index.intersection(inner_index).size:
                raise ValueError("fit and inner_eval_set timestamps overlap")
            if train_index.max() >= inner_index.min():
                raise ValueError(
                    "inner_eval_set must be strictly later than X for time-series "
                    "early stopping"
                )

    def _build_model(self, *, backend_device: str, use_early_stopping: bool) -> Any:
        params = dict(self.params or {})
        seed = int(self.seed)
        jobs = self.n_jobs_
        kind = self.kind_

        if kind.startswith("lgbm_"):
            try:
                from lightgbm import LGBMRegressor
            except ImportError as exc:  # pragma: no cover
                raise ImportError("lgbm models require lightgbm 4.6+") from exc
            objectives = {
                "lgbm_l1": "regression_l1",
                "lgbm_huber": "huber",
                "lgbm_tweedie": "tweedie",
                "lgbm_regression": "regression",
            }
            defaults: dict[str, Any] = {
                "objective": objectives[kind],
                "n_estimators": 1500,
                "learning_rate": 0.03,
                "num_leaves": 31,
                "min_child_samples": 40,
                "subsample": 0.85,
                "subsample_freq": 1,
                "colsample_bytree": 0.80,
                "reg_alpha": 0.05,
                "reg_lambda": 0.20,
                "random_state": seed,
                "n_jobs": jobs,
                "verbosity": -1,
                "importance_type": "gain",
            }
            if kind == "lgbm_tweedie":
                defaults["tweedie_variance_power"] = 1.3
            defaults.update(params)
            defaults["device_type"] = (
                "gpu" if backend_device == "gpu" else "cpu"
            )
            return LGBMRegressor(**defaults)

        if kind.startswith("xgb_"):
            try:
                from xgboost import XGBRegressor
            except ImportError as exc:  # pragma: no cover
                raise ImportError("xgb models require xgboost 3.0+") from exc
            objectives = {
                "xgb_l1": "reg:absoluteerror",
                "xgb_pseudohuber": "reg:pseudohubererror",
            }
            defaults = {
                "objective": objectives[kind],
                "n_estimators": 1500,
                "learning_rate": 0.03,
                "max_depth": 7,
                "min_child_weight": 5.0,
                "subsample": 0.85,
                "colsample_bytree": 0.80,
                "reg_alpha": 0.05,
                "reg_lambda": 1.0,
                "tree_method": "hist",
                "random_state": seed,
                "n_jobs": jobs,
                "verbosity": 0,
                "importance_type": "gain",
            }
            defaults.update(params)
            defaults["device"] = "cuda" if backend_device == "gpu" else "cpu"
            if use_early_stopping:
                defaults["early_stopping_rounds"] = int(
                    self.early_stopping_rounds  # type: ignore[arg-type]
                )
            else:
                defaults.pop("early_stopping_rounds", None)
            return XGBRegressor(**defaults)

        if kind.startswith("catboost_"):
            try:
                from catboost import CatBoostRegressor
            except ImportError as exc:  # pragma: no cover
                raise ImportError("catboost models require catboost 1.2.8+") from exc
            losses = {
                "catboost_mae": "MAE",
                "catboost_huber": "Huber:delta=1.0",
                "catboost_rmse": "RMSE",
            }
            defaults = {
                "loss_function": losses[kind],
                "iterations": 1500,
                "learning_rate": 0.03,
                "depth": 8,
                "l2_leaf_reg": 3.0,
                "random_strength": 0.5,
                "random_seed": seed,
                "thread_count": jobs,
                "verbose": False,
                "allow_writing_files": False,
            }
            defaults.update(params)
            defaults["task_type"] = (
                "GPU" if backend_device == "gpu" else "CPU"
            )
            return CatBoostRegressor(**defaults)

        if kind.startswith("extra_trees"):
            from sklearn.ensemble import ExtraTreesRegressor

            defaults = {
                "n_estimators": 600,
                "criterion": (
                    "absolute_error"
                    if kind == "extra_trees_l1"
                    else "squared_error"
                ),
                "max_features": 0.8,
                "min_samples_leaf": 2,
                "bootstrap": False,
                "random_state": seed,
                "n_jobs": jobs,
            }
            defaults.update(params)
            return ExtraTreesRegressor(**defaults)

        from sklearn.ensemble import HistGradientBoostingRegressor

        defaults = {
            "loss": (
                "absolute_error"
                if kind == "hist_gbdt_l1"
                else "squared_error"
            ),
            "learning_rate": 0.05,
            "max_iter": 500,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 30,
            "l2_regularization": 0.2,
            "early_stopping": False,
            "random_state": seed,
        }
        defaults.update(params)
        if defaults.get("early_stopping") not in {False, None}:
            raise ValueError(
                "hist_gbdt early_stopping must remain False because sklearn's "
                "internal random split is unsuitable for this time series"
            )
        return HistGradientBoostingRegressor(**defaults)

    def _fit_one(
        self,
        model: Any,
        X: pd.DataFrame,
        y: np.ndarray,
        sample_weight: np.ndarray | None,
        inner: tuple[pd.DataFrame, np.ndarray] | None,
        inner_weight: np.ndarray | None,
        use_early_stopping: bool,
    ) -> Any:
        kind = self.kind_
        common: dict[str, Any] = {}
        if sample_weight is not None:
            common["sample_weight"] = sample_weight

        if kind.startswith("lgbm_"):
            if use_early_stopping and inner is not None:
                from lightgbm import early_stopping, log_evaluation

                X_inner, y_inner = inner
                common["eval_set"] = [(X_inner, y_inner)]
                if inner_weight is not None:
                    common["eval_sample_weight"] = [inner_weight]
                common["eval_metric"] = "l1"
                common["callbacks"] = [
                    early_stopping(
                        int(self.early_stopping_rounds),  # type: ignore[arg-type]
                        verbose=False,
                    ),
                    log_evaluation(period=0),
                ]
            return model.fit(X, y, **common)

        if kind.startswith("xgb_"):
            if use_early_stopping and inner is not None:
                X_inner, y_inner = inner
                common["eval_set"] = [(X_inner, y_inner)]
                if inner_weight is not None:
                    common["sample_weight_eval_set"] = [inner_weight]
                common["verbose"] = False
            return model.fit(X, y, **common)

        if kind.startswith("catboost_"):
            from catboost import Pool

            train_pool = Pool(X, y, weight=sample_weight)
            fit_params: dict[str, Any] = {"verbose": False}
            if use_early_stopping and inner is not None:
                X_inner, y_inner = inner
                valid_pool = Pool(X_inner, y_inner, weight=inner_weight)
                fit_params.update(
                    eval_set=valid_pool,
                    early_stopping_rounds=int(
                        self.early_stopping_rounds  # type: ignore[arg-type]
                    ),
                    use_best_model=True,
                )
            return model.fit(train_pool, **fit_params)

        if inner is not None:
            warnings.warn(
                f"{kind} has no native explicit-eval early stopping; "
                "inner_eval_set was validated but is not passed to the estimator",
                UserWarning,
                stacklevel=3,
            )
        return model.fit(X, y, **common)

    def fit(
        self,
        X: pd.DataFrame,
        y: Any,
        sample_weight: Any = None,
        *,
        inner_eval_set: tuple[pd.DataFrame, Any] | None = None,
        inner_eval_sample_weight: Any = None,
    ) -> "TabularRegressor":
        """Fit a model with an optional explicit inner validation slice."""

        self._validate_options()
        X = _require_frame(X)
        y_array = _as_1d_target(y, name="y", index=X.index)
        weight_array = _as_weight(
            sample_weight, name="sample_weight", index=X.index
        )
        if self.kind_ == "lgbm_tweedie" and np.any(y_array < 0):
            raise ValueError("lgbm_tweedie requires non-negative target values")

        transformed = self._prepare_backend_frame(self._fit_encoder(X))
        inner: tuple[pd.DataFrame, np.ndarray] | None = None
        inner_weight: np.ndarray | None = None
        if inner_eval_set is not None:
            if not isinstance(inner_eval_set, tuple) or len(inner_eval_set) != 2:
                raise TypeError(
                    "inner_eval_set must be one (X_inner, y_inner) tuple"
                )
            X_inner_raw = _require_frame(inner_eval_set[0], name="X_inner")
            self._validate_inner_time_order(X.index, X_inner_raw.index)
            y_inner = _as_1d_target(
                inner_eval_set[1], name="y_inner", index=X_inner_raw.index
            )
            if self.kind_ == "lgbm_tweedie" and np.any(y_inner < 0):
                raise ValueError("lgbm_tweedie requires non-negative targets")
            inner_weight = _as_weight(
                inner_eval_sample_weight,
                name="inner_eval_sample_weight",
                index=X_inner_raw.index,
            )
            X_inner = self._prepare_backend_frame(self._transform(X_inner_raw))
            inner = (X_inner, y_inner)
        elif inner_eval_sample_weight is not None:
            raise ValueError(
                "inner_eval_sample_weight requires an explicit inner_eval_set"
            )

        use_early_stopping = (
            inner is not None
            and self.early_stopping_rounds is not None
            and self.kind_.startswith(("lgbm_", "xgb_", "catboost_"))
        )
        if inner is not None and self.early_stopping_rounds is None:
            warnings.warn(
                "inner_eval_set was supplied but early_stopping_rounds is None; "
                "validation data will not affect fitting",
                UserWarning,
                stacklevel=2,
            )

        supports_gpu = self.kind_.startswith(("lgbm_", "xgb_", "catboost_"))
        if supports_gpu and self.device_ in {"gpu", "auto"}:
            attempts = ["gpu", "cpu"] if self.fallback_to_cpu else ["gpu"]
        else:
            attempts = ["cpu"]

        first_error: Exception | None = None
        for position, backend_device in enumerate(attempts):
            try:
                model = self._build_model(
                    backend_device=backend_device,
                    use_early_stopping=use_early_stopping,
                )
                self.model_ = self._fit_one(
                    model,
                    transformed,
                    y_array,
                    weight_array,
                    inner,
                    inner_weight,
                    use_early_stopping,
                )
                self.device_used_ = backend_device
                break
            except Exception as exc:
                if position + 1 >= len(attempts):
                    if first_error is not None:
                        raise RuntimeError(
                            f"GPU fit failed ({first_error}); "
                            "CPU fallback also failed"
                        ) from exc
                    raise
                first_error = exc
                warnings.warn(
                    f"{self.kind_} GPU fit failed ({exc!s}); retrying on CPU",
                    DeviceFallbackWarning,
                    stacklevel=2,
                )

        self.best_iteration_ = self._read_best_iteration()
        return self

    def _read_best_iteration(self) -> int | None:
        model = self.model_
        if self.kind_.startswith("lgbm_"):
            value = getattr(model, "best_iteration_", None)
            if value is None or int(value) <= 0:
                value = getattr(model, "n_estimators_", None)
            return None if value is None else int(value)
        if self.kind_.startswith("xgb_"):
            try:
                return int(model.best_iteration) + 1
            except (AttributeError, TypeError):
                return int(model.get_params().get("n_estimators", 0)) or None
        if self.kind_.startswith("catboost_"):
            raw_value = model.get_best_iteration()
            if raw_value is None:
                return int(model.tree_count_)
            value = int(raw_value)
            return value + 1 if value >= 0 else int(model.tree_count_)
        if self.kind_.startswith("extra_trees"):
            return len(model.estimators_)
        value = getattr(model, "n_iter_", None)
        return None if value is None else int(value)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict after strict feature alignment."""

        check_is_fitted(self, "model_")
        transformed = self._prepare_backend_frame(self._transform(X))
        prediction = np.asarray(
            self.model_.predict(transformed), dtype=np.float64
        )
        if prediction.ndim != 1 or len(prediction) != len(X):
            raise RuntimeError(
                f"backend returned invalid prediction shape {prediction.shape}"
            )
        if not np.all(np.isfinite(prediction)):
            raise RuntimeError("backend returned NaN or infinite predictions")
        return prediction

    def _native_importance(self) -> np.ndarray | None:
        model = self.model_
        values: Any = None
        if self.kind_.startswith("lgbm_") and hasattr(model, "booster_"):
            values = model.booster_.feature_importance(importance_type="gain")
        elif self.kind_.startswith("catboost_"):
            values = model.get_feature_importance(type="FeatureImportance")
        elif hasattr(model, "feature_importances_"):
            values = model.feature_importances_
        if values is None:
            return None
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (self.n_features_in_,):
            raise RuntimeError(
                "backend importance length does not match fitted features"
            )
        return array

    @property
    def feature_importances_(self) -> np.ndarray:
        """Native importance, or NaNs when permutation importance is needed."""

        check_is_fitted(self, "model_")
        values = self._native_importance()
        if values is None:
            return np.full(self.n_features_in_, np.nan, dtype=np.float64)
        return values.copy()

    def get_feature_importance(
        self,
        X: pd.DataFrame | None = None,
        y: Any = None,
        *,
        sample_weight: Any = None,
        scoring: str = "neg_mean_absolute_error",
        n_repeats: int = 3,
        max_samples: int | None = 5000,
        normalize: bool = True,
        sort: bool = True,
    ) -> pd.Series:
        """Return named native or held-out permutation importance.

        Supplying X and y forces permutation importance and is required for the
        HistGradientBoosting candidates. Held-out data is recommended.
        """

        check_is_fitted(self, "model_")
        if (X is None) != (y is None):
            raise ValueError("X and y must both be supplied or both omitted")
        if X is None:
            values = self._native_importance()
            if values is None:
                raise ValueError(
                    f"{self.kind_} has no native importance; supply held-out "
                    "X and y for permutation importance"
                )
        else:
            X = _require_frame(X)
            y_array = _as_1d_target(y, name="y", index=X.index)
            weights = _as_weight(
                sample_weight, name="sample_weight", index=X.index
            )
            transformed = self._prepare_backend_frame(self._transform(X))
            if max_samples is not None and len(transformed) > int(max_samples):
                if int(max_samples) <= 0:
                    raise ValueError("max_samples must be positive or None")
                rng = np.random.default_rng(int(self.seed))
                positions = np.sort(
                    rng.choice(
                        len(transformed), size=int(max_samples), replace=False
                    )
                )
                transformed = transformed.iloc[positions]
                y_array = y_array[positions]
                if weights is not None:
                    weights = weights[positions]
            result = permutation_importance(
                self.model_,
                transformed,
                y_array,
                scoring=scoring,
                n_repeats=int(n_repeats),
                random_state=int(self.seed),
                n_jobs=self.n_jobs_,
                sample_weight=weights,
            )
            values = np.asarray(result.importances_mean, dtype=np.float64)

        if normalize:
            denominator = float(np.sum(np.abs(values)))
            if denominator > 0:
                values = values / denominator
        series = pd.Series(
            values,
            index=pd.Index(self.feature_names_in_, name="feature"),
            name="importance",
            dtype=np.float64,
        )
        return series.sort_values(ascending=False) if sort else series


def make_regressor(kind: str, **kwargs: Any) -> TabularRegressor:
    """Create a TabularRegressor."""

    return TabularRegressor(kind=kind, **kwargs)


ModelWrapper = TabularRegressor


__all__ = [
    "DeviceFallbackWarning",
    "ModelWrapper",
    "SUPPORTED_MODELS",
    "TabularRegressor",
    "available_models",
    "make_regressor",
]

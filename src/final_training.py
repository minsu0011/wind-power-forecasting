"""Reproducible full-data training and submission assembly.

This module intentionally consumes only official raw labels/submission schema
and deterministic weather-feature caches. Development OOF files and previously
fitted model artifacts are never inputs. A JSON recipe fixes every candidate,
feature-selection rule, ensemble weight, affine map, and physical clip.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .manifest import make_manifest, sha256_file, write_json_atomic
from .metric import CAPACITY_KWH, TARGET_COLS
from .models import TabularRegressor
from .postprocess import (
    FICRBoundedPostprocessor,
    GroupAffineCalibrator,
    SimplexEnsembler,
)


FINAL_RECIPE_SCHEMA_VERSION = 1
REQUIRED_BASE_CANDIDATES = (
    "lgb_l1",
    "lgb_q07",
    "shared_l1",
    "shared_q07",
)
OPTIONAL_V2_CANDIDATES = ("top200_q07", "energy_q06")
SAFE_RECIPE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass
class FinalDataBundle:
    """Validated labels, submission template, and group feature frames."""

    labels: pd.DataFrame
    sample_submission: pd.DataFrame
    sample_times: pd.DatetimeIndex
    train_features: dict[str, pd.DataFrame]
    test_features: dict[str, pd.DataFrame]
    input_files: list[Path]


@dataclass
class TrainedCandidate:
    """Models, selected columns, predictions, and row counts for one candidate."""

    name: str
    scope: str
    models: Any
    feature_names: dict[str, list[str]]
    predictions: pd.DataFrame
    training_rows: dict[str, int]


def read_recipe(path: str | Path) -> dict[str, Any]:
    """Read and structurally validate one final-training JSON recipe."""

    recipe_path = Path(path)
    with recipe_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise TypeError("recipe root must be a JSON object")
    validate_recipe(payload)
    return payload


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _require_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _enabled_models(recipe: Mapping[str, Any]) -> tuple[str, ...]:
    models = _require_mapping(recipe["models"], name="models")
    return tuple(
        name
        for name, specification in models.items()
        if bool(_require_mapping(specification, name=f"models.{name}").get("enabled"))
    )


def validate_recipe(recipe: Mapping[str, Any]) -> None:
    """Validate candidate definitions and, when locked, final blend constants."""

    if int(recipe.get("schema_version", -1)) != FINAL_RECIPE_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {FINAL_RECIPE_SCHEMA_VERSION}"
        )
    name = str(recipe.get("recipe_name", ""))
    if not SAFE_RECIPE_NAME.fullmatch(name):
        raise ValueError(
            "recipe_name must contain only letters, digits, dot, underscore, or dash"
        )
    if tuple(recipe.get("groups", ())) != TARGET_COLS:
        raise ValueError(f"groups must be exactly {TARGET_COLS!r}")
    _require_number(recipe.get("seed"), name="seed")
    n_jobs = int(recipe.get("n_jobs", 0))
    if n_jobs == 0:
        raise ValueError("n_jobs must not be zero")
    if str(recipe.get("device", "")).lower() not in {"cpu", "gpu", "auto"}:
        raise ValueError("device must be cpu, gpu, or auto")
    if not isinstance(recipe.get("recipe_locked"), bool):
        raise TypeError("recipe_locked must be boolean")
    for forbidden in ("claimed_score", "expected_score", "leaderboard_score"):
        if forbidden in recipe:
            raise ValueError(
                f"{forbidden} is not part of the final recipe schema"
            )

    models = _require_mapping(recipe.get("models"), name="models")
    if not models:
        raise ValueError("models must not be empty")
    enabled = _enabled_models(recipe)
    missing_required = [
        candidate for candidate in REQUIRED_BASE_CANDIDATES if candidate not in enabled
    ]
    if missing_required:
        raise ValueError(
            f"required base candidates are disabled/missing: {missing_required}"
        )
    if not any(candidate in enabled for candidate in OPTIONAL_V2_CANDIDATES):
        raise ValueError(
            "enable at least one of top200_q07 or energy_q06 for the v2 recipe"
        )

    enabled_seen: set[str] = set()
    for candidate_name, raw_specification in models.items():
        if not SAFE_RECIPE_NAME.fullmatch(str(candidate_name)):
            raise ValueError(f"unsafe candidate name: {candidate_name!r}")
        specification = _require_mapping(
            raw_specification, name=f"models.{candidate_name}"
        )
        allowed_keys = {
            "enabled",
            "scope",
            "objective",
            "alpha",
            "target_scale",
            "row_filter",
            "features",
            "sample_weight",
            "params",
        }
        extra = set(specification).difference(allowed_keys)
        if extra:
            raise ValueError(
                f"models.{candidate_name} has unknown keys: {sorted(extra)}"
            )
        if not isinstance(specification.get("enabled"), bool):
            raise TypeError(f"models.{candidate_name}.enabled must be boolean")
        if not specification["enabled"]:
            continue
        scope = str(specification.get("scope"))
        if scope not in {"group", "shared"}:
            raise ValueError(
                f"models.{candidate_name}.scope must be group or shared"
            )
        target_scale = str(specification.get("target_scale", "kwh"))
        if target_scale not in {"kwh", "capacity_factor"}:
            raise ValueError(
                f"models.{candidate_name}.target_scale must be kwh or "
                "capacity_factor"
            )
        objective = str(specification.get("objective"))
        if objective not in {"l1", "quantile"}:
            raise ValueError(
                f"models.{candidate_name}.objective must be l1 or quantile"
            )
        alpha = specification.get("alpha")
        if objective == "quantile":
            alpha_value = _require_number(
                alpha, name=f"models.{candidate_name}.alpha"
            )
            if not 0 < alpha_value < 1:
                raise ValueError(
                    f"models.{candidate_name}.alpha must be in (0, 1)"
                )
        elif alpha is not None:
            raise ValueError(
                f"models.{candidate_name}.alpha must be null for l1"
            )
        contracts = {
            "lgb_l1": ("group", "l1", None),
            "lgb_q07": ("group", "quantile", 0.7),
            "shared_l1": ("shared", "l1", None),
            "shared_q07": ("shared", "quantile", 0.7),
            "top200_q07": ("group", "quantile", 0.7),
            "energy_q06": ("group", "quantile", 0.6),
        }
        if candidate_name in contracts:
            expected_scope, expected_objective, expected_alpha = contracts[candidate_name]
            if scope != expected_scope or objective != expected_objective:
                raise ValueError(
                    f"models.{candidate_name} must use scope={expected_scope} "
                    f"and objective={expected_objective}"
                )
            if expected_alpha is not None and not np.isclose(
                float(alpha), expected_alpha, rtol=0, atol=1e-12
            ):
                raise ValueError(
                    f"models.{candidate_name}.alpha must be {expected_alpha}"
                )
        if specification.get("row_filter") not in {"all", "eligible"}:
            raise ValueError(
                f"models.{candidate_name}.row_filter must be all or eligible"
            )
        native_params = _require_mapping(
            specification.get("params"), name=f"models.{candidate_name}.params"
        )
        conflicting = {"objective", "alpha"}.intersection(native_params)
        if conflicting:
            raise ValueError(
                f"models.{candidate_name}.params duplicates top-level "
                f"objective settings: {sorted(conflicting)}"
            )
        features = specification.get("features")
        if features == "all":
            pass
        elif isinstance(features, list):
            if not features or len(set(map(str, features))) != len(features):
                raise ValueError(
                    f"models.{candidate_name}.features list must be non-empty/unique"
                )
        elif isinstance(features, Mapping):
            feature_kind = features.get("kind")
            if feature_kind not in {"top_gain", "explicit_by_group"}:
                raise ValueError(
                    f"models.{candidate_name}.features.kind must be top_gain "
                    "or explicit_by_group"
                )
            if scope != "group":
                raise ValueError("mapped feature selection requires group scope")
            if feature_kind == "top_gain":
                source = str(features.get("source"))
                if source not in enabled_seen:
                    raise ValueError(
                        f"models.{candidate_name} top_gain source {source!r} must be "
                        "an earlier enabled candidate"
                    )
                top_k = int(features.get("top_k", 0))
                if top_k <= 0:
                    raise ValueError("top_gain top_k must be positive")
            else:
                columns_by_group = _require_mapping(
                    features.get("columns"),
                    name=f"models.{candidate_name}.features.columns",
                )
                if tuple(columns_by_group) != TARGET_COLS:
                    raise ValueError(
                        "explicit_by_group columns must list all target groups in order"
                    )
                for group in TARGET_COLS:
                    columns = columns_by_group[group]
                    if (
                        not isinstance(columns, list)
                        or not columns
                        or len(set(map(str, columns))) != len(columns)
                    ):
                        raise ValueError(
                            f"explicit_by_group columns for {group} must be "
                            "non-empty and unique"
                        )
        else:
            raise TypeError(
                f"models.{candidate_name}.features must be all, list, or object"
            )
        weight = _require_mapping(
            specification.get("sample_weight"),
            name=f"models.{candidate_name}.sample_weight",
        )
        weight_kind = weight.get("kind")
        if weight_kind not in {"none", "energy_linear"}:
            raise ValueError(
                f"models.{candidate_name}.sample_weight.kind is invalid"
            )
        if weight_kind == "energy_linear":
            base = _require_number(
                weight.get("base"), name=f"models.{candidate_name}.sample_weight.base"
            )
            strength = _require_number(
                weight.get("strength"),
                name=f"models.{candidate_name}.sample_weight.strength",
            )
            if base <= 0 or strength < 0:
                raise ValueError("energy_linear requires base > 0 and strength >= 0")
        if candidate_name == "energy_q06" and weight_kind != "energy_linear":
            raise ValueError("models.energy_q06 must use energy_linear weights")
        enabled_seen.add(str(candidate_name))

    if not recipe["recipe_locked"]:
        return
    ensemble = _require_mapping(recipe.get("ensemble"), name="ensemble")
    weights = _require_mapping(ensemble.get("weights"), name="ensemble.weights")
    affine = _require_mapping(ensemble.get("affine"), name="ensemble.affine")
    clip = _require_mapping(ensemble.get("clip"), name="ensemble.clip")
    power_bins = _require_mapping(
        ensemble.get("power_bins"), name="ensemble.power_bins"
    )
    if tuple(power_bins) != TARGET_COLS:
        raise ValueError("ensemble.power_bins must list all target groups in order")
    for group in TARGET_COLS:
        group_weights = _require_mapping(
            weights.get(group), name=f"ensemble.weights.{group}"
        )
        if tuple(group_weights) != enabled:
            raise ValueError(
                f"ensemble.weights.{group} models/order must equal enabled models "
                f"{enabled!r}"
            )
        values = np.asarray(
            [
                _require_number(
                    group_weights[model],
                    name=f"ensemble.weights.{group}.{model}",
                )
                for model in enabled
            ]
        )
        if np.any(values < 0) or not np.isclose(
            values.sum(), 1.0, rtol=0, atol=1e-10
        ):
            raise ValueError(
                f"ensemble.weights.{group} must be non-negative and sum to one"
            )
        group_affine = _require_mapping(
            affine.get(group), name=f"ensemble.affine.{group}"
        )
        scale = _require_number(
            group_affine.get("scale"), name=f"ensemble.affine.{group}.scale"
        )
        _require_number(
            group_affine.get("bias_kwh"),
            name=f"ensemble.affine.{group}.bias_kwh",
        )
        if scale <= 0:
            raise ValueError(f"ensemble.affine.{group}.scale must be positive")
        group_clip = _require_mapping(
            clip.get(group), name=f"ensemble.clip.{group}"
        )
        lower = _require_number(
            group_clip.get("lower_capacity_fraction"),
            name=f"ensemble.clip.{group}.lower_capacity_fraction",
        )
        upper = _require_number(
            group_clip.get("upper_capacity_fraction"),
            name=f"ensemble.clip.{group}.upper_capacity_fraction",
        )
        if lower < 0 or upper <= lower:
            raise ValueError(
                f"ensemble.clip.{group} must satisfy 0 <= lower < upper"
            )
        bin_specification = power_bins[group]
        if bin_specification is not None:
            bin_mapping = _require_mapping(
                bin_specification, name=f"ensemble.power_bins.{group}"
            )
            edges = np.asarray(
                [
                    _require_number(
                        value, name=f"ensemble.power_bins.{group}.edges_cf"
                    )
                    for value in bin_mapping.get("edges_cf", ())
                ],
                dtype=float,
            )
            deltas = np.asarray(
                [
                    _require_number(
                        value, name=f"ensemble.power_bins.{group}.delta_kwh"
                    )
                    for value in bin_mapping.get("delta_kwh", ())
                ],
                dtype=float,
            )
            if (
                len(edges) < 2
                or len(deltas) != len(edges) - 1
                or not np.all(np.diff(edges) > 0)
            ):
                raise ValueError(
                    f"ensemble.power_bins.{group} needs increasing edges and "
                    "one fewer deltas"
                )
            if edges[0] > lower or edges[-1] < upper:
                raise ValueError(
                    f"ensemble.power_bins.{group} edges must cover clip bounds"
                )


def _cache_paths(cache_dir: Path) -> dict[tuple[str, str], Path]:
    return {
        (split, group): cache_dir / f"{group}_weather_{split}.parquet"
        for split in ("train", "test")
        for group in TARGET_COLS
    }


def load_final_data(raw_dir: str | Path, cache_dir: str | Path) -> FinalDataBundle:
    """Load official labels/template and validate all six weather caches."""

    raw_root = Path(raw_dir).expanduser().resolve()
    cache_root = Path(cache_dir).expanduser().resolve()
    label_path = raw_root / "train" / "train_labels.csv"
    sample_path = raw_root / "sample_submission.csv"
    cache_paths = _cache_paths(cache_root)
    required = [label_path, sample_path, *cache_paths.values()]
    missing = [path for path in required if not path.is_file()]
    if missing:
        rendered = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"raw/cache inputs are incomplete:\n{rendered}")

    labels = pd.read_csv(label_path, encoding="utf-8-sig")
    expected_label_columns = ("kst_dtm", *TARGET_COLS)
    if tuple(labels.columns) != expected_label_columns:
        raise ValueError(
            f"train_labels columns must be {expected_label_columns!r}"
        )
    label_times = pd.to_datetime(labels.pop("kst_dtm"), errors="raise")
    labels.index = pd.DatetimeIndex(label_times, name="forecast_kst_dtm")
    labels = labels.astype(float)
    if not labels.index.is_unique or not labels.index.is_monotonic_increasing:
        raise ValueError("train label timestamps must be unique and sorted")
    if np.isinf(labels.to_numpy()).any():
        raise ValueError("train labels contain infinite values")
    finite_labels = labels.to_numpy()[np.isfinite(labels.to_numpy())]
    if finite_labels.size == 0 or np.any(finite_labels < 0):
        raise ValueError("train labels must contain non-negative finite values")

    sample = pd.read_csv(
        sample_path,
        encoding="utf-8-sig",
        dtype={"forecast_id": "string", "forecast_kst_dtm": "string"},
    )
    expected_submission_columns = (
        "forecast_id",
        "forecast_kst_dtm",
        *TARGET_COLS,
    )
    if tuple(sample.columns) != expected_submission_columns:
        raise ValueError(
            f"sample_submission columns must be {expected_submission_columns!r}"
        )
    if sample["forecast_id"].isna().any() or sample["forecast_id"].duplicated().any():
        raise ValueError("forecast_id must be complete and unique")
    sample_times = pd.DatetimeIndex(
        pd.to_datetime(sample["forecast_kst_dtm"], errors="raise"),
        name="forecast_kst_dtm",
    )
    if not sample_times.is_unique or not sample_times.is_monotonic_increasing:
        raise ValueError("sample submission timestamps must be unique and sorted")
    if labels.index.max() >= sample_times.min():
        raise ValueError("train label and test forecast periods overlap")

    train_features: dict[str, pd.DataFrame] = {}
    test_features: dict[str, pd.DataFrame] = {}
    canonical_columns: tuple[str, ...] | None = None
    for group in TARGET_COLS:
        train = pd.read_parquet(cache_paths[("train", group)])
        test = pd.read_parquet(cache_paths[("test", group)])
        for split, frame in (("train", train), ("test", test)):
            if not isinstance(frame.index, pd.DatetimeIndex):
                raise TypeError(f"{group} {split} cache must use a DatetimeIndex")
            if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
                raise ValueError(f"{group} {split} cache index must be unique/sorted")
            if not frame.columns.is_unique:
                raise ValueError(f"{group} {split} cache has duplicate features")
            if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes):
                raise TypeError(f"{group} {split} cache features must be numeric")
            if not np.isfinite(frame.to_numpy()).all():
                raise ValueError(f"{group} {split} cache contains non-finite values")
        if not train.index.equals(labels.index):
            raise ValueError(f"{group} train cache is not aligned to train labels")
        if not test.index.equals(sample_times):
            raise ValueError(
                f"{group} test cache is not aligned to sample_submission order"
            )
        if tuple(train.columns) != tuple(test.columns):
            raise ValueError(f"{group} train/test feature schemas differ")
        if canonical_columns is None:
            canonical_columns = tuple(train.columns)
        elif tuple(train.columns) != canonical_columns:
            raise ValueError("group weather feature schemas must be identical")
        train_features[group] = train.astype(np.float32, copy=False)
        test_features[group] = test.astype(np.float32, copy=False)

    feature_manifest = cache_root / "feature_manifest.json"
    inputs = required.copy()
    if feature_manifest.is_file():
        inputs.append(feature_manifest)
    return FinalDataBundle(
        labels=labels,
        sample_submission=sample,
        sample_times=sample_times,
        train_features=train_features,
        test_features=test_features,
        input_files=inputs,
    )


def schema_summary(
    recipe: Mapping[str, Any],
    data: FinalDataBundle,
) -> dict[str, Any]:
    """Return a no-fit plan for a safe CLI dry run."""

    enabled = _enabled_models(recipe)
    groups: dict[str, Any] = {}
    for group in TARGET_COLS:
        target = data.labels[group]
        finite = target.notna()
        eligible = finite & (target >= CAPACITY_KWH[group] * 0.10)
        groups[group] = {
            "train_rows": int(len(target)),
            "finite_label_rows": int(finite.sum()),
            "eligible_label_rows": int(eligible.sum()),
            "test_rows": int(len(data.test_features[group])),
            "features": int(data.train_features[group].shape[1]),
            "train_start": data.train_features[group].index.min().isoformat(),
            "train_end": data.train_features[group].index.max().isoformat(),
            "test_start": data.test_features[group].index.min().isoformat(),
            "test_end": data.test_features[group].index.max().isoformat(),
        }
    return {
        "schema_status": "valid",
        "recipe_name": recipe["recipe_name"],
        "recipe_locked": bool(recipe["recipe_locked"]),
        "ready_for_training": bool(recipe["recipe_locked"]),
        "enabled_candidates": list(enabled),
        "groups": groups,
        "submission_columns": list(data.sample_submission.columns),
        "writes_performed": False,
    }


def _training_mask(
    target: pd.Series,
    *,
    row_filter: str,
    capacity: float,
) -> pd.Series:
    mask = target.notna() & np.isfinite(target.to_numpy())
    if row_filter == "eligible":
        mask &= target >= capacity * 0.10
    return mask


def _sample_weights(
    target: pd.Series,
    *,
    capacity: float,
    specification: Mapping[str, Any],
) -> pd.Series | None:
    kind = specification["kind"]
    if kind == "none":
        return None
    base = float(specification["base"])
    strength = float(specification["strength"])
    normalized = np.clip(target.to_numpy(dtype=float) / capacity, 0.0, 1.5)
    return pd.Series(
        base + strength * normalized,
        index=target.index,
        dtype=float,
        name="sample_weight",
    )


def _native_model_params(specification: Mapping[str, Any]) -> dict[str, Any]:
    params = dict(specification["params"])
    objective = specification["objective"]
    if objective == "l1":
        params["objective"] = "regression_l1"
    else:
        params["objective"] = "quantile"
        params["alpha"] = float(specification["alpha"])
    return params


def _rank_gain_features(
    source: TrainedCandidate,
    group: str,
    available_columns: Sequence[str],
    top_k: int,
) -> list[str]:
    if source.scope != "group":
        raise ValueError("top_gain source must be a group-specific candidate")
    source_model = source.models[group]
    importance = source_model.get_feature_importance(
        normalize=False, sort=False
    )
    positions = {str(column): position for position, column in enumerate(available_columns)}
    rows: list[tuple[float, int, str]] = []
    for feature, value in importance.items():
        feature_name = str(feature)
        if feature_name in positions:
            rows.append((-float(value), positions[feature_name], feature_name))
    if len(rows) != len(available_columns):
        raise ValueError(
            f"gain source feature schema differs for {group}: "
            f"{len(rows)} != {len(available_columns)}"
        )
    rows.sort()
    if top_k > len(rows):
        raise ValueError(
            f"top_k={top_k} exceeds available features={len(rows)}"
        )
    return [feature for _, _, feature in rows[:top_k]]


def _resolve_feature_names(
    specification: Mapping[str, Any],
    *,
    group: str,
    columns: Sequence[str],
    trained: Mapping[str, TrainedCandidate],
) -> list[str]:
    feature_specification = specification["features"]
    available = list(map(str, columns))
    if feature_specification == "all":
        return available
    if isinstance(feature_specification, list):
        selected = list(map(str, feature_specification))
        missing = [feature for feature in selected if feature not in available]
        if missing:
            raise ValueError(f"explicit features missing for {group}: {missing[:10]}")
        return selected
    if feature_specification["kind"] == "explicit_by_group":
        selected = list(map(str, feature_specification["columns"][group]))
        missing = [feature for feature in selected if feature not in available]
        if missing:
            raise ValueError(
                f"explicit_by_group features missing for {group}: {missing[:10]}"
            )
        return selected
    source_name = str(feature_specification["source"])
    if source_name not in trained:
        raise ValueError(f"top_gain source {source_name!r} has not been trained")
    return _rank_gain_features(
        trained[source_name],
        group,
        available,
        int(feature_specification["top_k"]),
    )


def train_candidate(
    name: str,
    specification: Mapping[str, Any],
    *,
    recipe: Mapping[str, Any],
    data: FinalDataBundle,
    trained: Mapping[str, TrainedCandidate],
) -> TrainedCandidate:
    """Fit one fully specified candidate on every available training year."""

    scope = str(specification["scope"])
    target_scale = str(specification.get("target_scale", "kwh"))
    seed = int(recipe["seed"])
    n_jobs = int(recipe["n_jobs"])
    device = str(recipe["device"])
    params = _native_model_params(specification)
    predictions = pd.DataFrame(
        index=data.sample_times,
        columns=TARGET_COLS,
        dtype=float,
    )
    training_rows: dict[str, int] = {}
    feature_names: dict[str, list[str]] = {}

    if scope == "group":
        models: dict[str, TabularRegressor] = {}
        for group in TARGET_COLS:
            capacity = CAPACITY_KWH[group]
            selected = _resolve_feature_names(
                specification,
                group=group,
                columns=data.train_features[group].columns,
                trained=trained,
            )
            target = data.labels[group]
            mask = _training_mask(
                target,
                row_filter=str(specification["row_filter"]),
                capacity=capacity,
            )
            train_x = data.train_features[group].loc[mask, selected]
            raw_train_y = target.loc[mask].astype(float)
            weights = _sample_weights(
                raw_train_y,
                capacity=capacity,
                specification=specification["sample_weight"],
            )
            train_y = (
                raw_train_y / capacity
                if target_scale == "capacity_factor"
                else raw_train_y
            )
            if len(train_y) < 100:
                raise ValueError(f"{name}/{group} has too few training labels")
            model = TabularRegressor(
                "lgbm_l1",
                params=params,
                seed=seed,
                n_jobs=n_jobs,
                device=device,
                early_stopping_rounds=None,
                fallback_to_cpu=True,
            )
            model.fit(train_x, train_y, sample_weight=weights)
            prediction = model.predict(data.test_features[group].loc[:, selected])
            if target_scale == "capacity_factor":
                prediction = prediction * capacity
            models[group] = model
            predictions[group] = prediction
            training_rows[group] = int(len(train_y))
            feature_names[group] = selected
        return TrainedCandidate(
            name=name,
            scope=scope,
            models=models,
            feature_names=feature_names,
            predictions=predictions,
            training_rows=training_rows,
        )

    if specification["features"] != "all" and not isinstance(
        specification["features"], list
    ):
        raise ValueError("shared candidates support all or an explicit feature list")
    common_columns = list(map(str, data.train_features[TARGET_COLS[0]].columns))
    if specification["features"] == "all":
        selected = common_columns
    else:
        selected = list(map(str, specification["features"]))
        missing = [feature for feature in selected if feature not in common_columns]
        if missing:
            raise ValueError(f"shared features are missing: {missing[:10]}")
    shared_feature_names = [*selected, "model__group_id"]
    train_parts: list[pd.DataFrame] = []
    target_parts: list[pd.Series] = []
    weight_parts: list[pd.Series] = []
    use_weights = specification["sample_weight"]["kind"] != "none"
    for group_number, group in enumerate(TARGET_COLS, start=1):
        capacity = CAPACITY_KWH[group]
        target = data.labels[group]
        mask = _training_mask(
            target,
            row_filter=str(specification["row_filter"]),
            capacity=capacity,
        )
        part = data.train_features[group].loc[mask, selected].copy()
        part["model__group_id"] = np.float32(group_number)
        part.reset_index(drop=True, inplace=True)
        raw_group_target = target.loc[mask].astype(float).reset_index(drop=True)
        group_target = (
            raw_group_target / capacity
            if target_scale == "capacity_factor"
            else raw_group_target
        )
        train_parts.append(part)
        target_parts.append(group_target)
        training_rows[group] = int(len(group_target))
        feature_names[group] = shared_feature_names
        if use_weights:
            group_weights = _sample_weights(
                raw_group_target,
                capacity=capacity,
                specification=specification["sample_weight"],
            )
            assert group_weights is not None
            weight_parts.append(group_weights.reset_index(drop=True))
    train_x = pd.concat(train_parts, axis=0, ignore_index=True)
    train_y = pd.concat(target_parts, axis=0, ignore_index=True)
    weights = (
        pd.concat(weight_parts, axis=0, ignore_index=True)
        if use_weights
        else None
    )
    model = TabularRegressor(
        "lgbm_l1",
        params=params,
        seed=seed,
        n_jobs=n_jobs,
        device=device,
        early_stopping_rounds=None,
        fallback_to_cpu=True,
    )
    model.fit(train_x, train_y, sample_weight=weights)
    for group_number, group in enumerate(TARGET_COLS, start=1):
        test_x = data.test_features[group].loc[:, selected].copy()
        test_x["model__group_id"] = np.float32(group_number)
        prediction = model.predict(test_x)
        if target_scale == "capacity_factor":
            prediction = prediction * CAPACITY_KWH[group]
        predictions[group] = prediction
    return TrainedCandidate(
        name=name,
        scope=scope,
        models=model,
        feature_names=feature_names,
        predictions=predictions,
        training_rows=training_rows,
    )


def _atomic_joblib(payload: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        joblib.dump(payload, temporary, compress=3)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        frame.to_parquet(temporary, engine="pyarrow", compression="zstd", index=True)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_submission_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        frame.to_csv(
            temporary,
            index=False,
            encoding="utf-8-sig",
            float_format="%.6f",
            lineterminator="\n",
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _output_paths(
    out_dir: Path,
    recipe: Mapping[str, Any],
) -> dict[str, Path]:
    recipe_name = str(recipe["recipe_name"])
    result: dict[str, Path] = {
        "submission": out_dir / f"{recipe_name}__submission.csv",
        "final_predictions": out_dir
        / "predictions"
        / f"{recipe_name}__final_test.parquet",
        "recipe_snapshot": out_dir / f"{recipe_name}__recipe.json",
        "manifest": out_dir / f"{recipe_name}__manifest.json",
    }
    for candidate in _enabled_models(recipe):
        result[f"model:{candidate}"] = (
            out_dir / "models" / f"{recipe_name}__{candidate}.joblib"
        )
        result[f"prediction:{candidate}"] = (
            out_dir / "predictions" / f"{recipe_name}__{candidate}_test.parquet"
        )
    return result


def _preflight_outputs(paths: Mapping[str, Path], *, overwrite: bool) -> None:
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        rendered = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(
            "final outputs already exist; pass --overwrite to replace them:\n"
            f"{rendered}"
        )


def _assemble_final_predictions(
    recipe: Mapping[str, Any],
    candidates: Mapping[str, TrainedCandidate],
) -> pd.DataFrame:
    ensemble = recipe["ensemble"]
    enabled = _enabled_models(recipe)
    weights = ensemble["weights"]
    ensembler = SimplexEnsembler.from_weights(weights, capacities=CAPACITY_KWH)
    ordered_predictions = {
        model: candidates[model].predictions for model in enabled
    }
    blended = ensembler.predict(ordered_predictions)
    affine_parameters = {
        group: (
            float(ensemble["affine"][group]["scale"]),
            float(ensemble["affine"][group]["bias_kwh"]),
        )
        for group in TARGET_COLS
    }
    affine = GroupAffineCalibrator.from_parameters(
        affine_parameters, capacities=CAPACITY_KWH
    )
    calibrated = affine.transform(blended)
    bounded_parameters = {
        group: (
            1.0,
            0.0,
            float(ensemble["clip"][group]["lower_capacity_fraction"]),
            float(ensemble["clip"][group]["upper_capacity_fraction"]),
        )
        for group in TARGET_COLS
    }
    bounded = FICRBoundedPostprocessor.from_parameters(
        bounded_parameters, capacities=CAPACITY_KWH
    )
    output = bounded.transform(calibrated)
    for group in TARGET_COLS:
        bin_specification = ensemble["power_bins"][group]
        if bin_specification is None:
            continue
        edges = np.asarray(bin_specification["edges_cf"], dtype=float)
        deltas = np.asarray(bin_specification["delta_kwh"], dtype=float)
        capacity = CAPACITY_KWH[group]
        capacity_fraction = output[group].to_numpy(dtype=float) / capacity
        positions = np.searchsorted(
            edges[1:-1], capacity_fraction, side="right"
        )
        if np.any(positions < 0) or np.any(positions >= len(deltas)):
            raise ValueError(f"power-bin edges do not cover predictions for {group}")
        adjusted = output[group].to_numpy(dtype=float) + deltas[positions]
        lower = (
            float(ensemble["clip"][group]["lower_capacity_fraction"])
            * capacity
        )
        upper = (
            float(ensemble["clip"][group]["upper_capacity_fraction"])
            * capacity
        )
        output[group] = np.clip(adjusted, lower, upper)
    return output


def _build_submission(
    data: FinalDataBundle,
    predictions: pd.DataFrame,
    recipe: Mapping[str, Any],
) -> pd.DataFrame:
    if not predictions.index.equals(data.sample_times):
        raise ValueError("final prediction index differs from sample submission")
    if tuple(predictions.columns) != TARGET_COLS:
        raise ValueError("final prediction groups/order changed")
    values = predictions.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("final predictions contain non-finite values")
    for group in TARGET_COLS:
        lower = (
            float(recipe["ensemble"]["clip"][group]["lower_capacity_fraction"])
            * CAPACITY_KWH[group]
        )
        upper = (
            float(recipe["ensemble"]["clip"][group]["upper_capacity_fraction"])
            * CAPACITY_KWH[group]
        )
        if (predictions[group] < lower - 1e-8).any() or (
            predictions[group] > upper + 1e-8
        ).any():
            raise ValueError(f"final predictions violate configured clip for {group}")
    submission = data.sample_submission.copy()
    for group in TARGET_COLS:
        submission[group] = predictions[group].to_numpy(dtype=float)
    return submission


def _verify_written_submission(
    path: Path,
    expected: pd.DataFrame,
    recipe: Mapping[str, Any],
) -> None:
    with path.open("rb") as stream:
        if stream.read(3) != b"\xef\xbb\xbf":
            raise AssertionError("submission CSV is not UTF-8-SIG")
    observed = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"forecast_id": "string", "forecast_kst_dtm": "string"},
    )
    if tuple(observed.columns) != tuple(expected.columns):
        raise AssertionError("written submission columns/order changed")
    if len(observed) != len(expected):
        raise AssertionError("written submission row count changed")
    if not np.array_equal(
        observed["forecast_id"].astype(str).to_numpy(),
        expected["forecast_id"].astype(str).to_numpy(),
    ):
        raise AssertionError("written forecast_id values/order changed")
    if not np.array_equal(
        observed["forecast_kst_dtm"].astype(str).to_numpy(),
        expected["forecast_kst_dtm"].astype(str).to_numpy(),
    ):
        raise AssertionError("written forecast timestamps/order changed")
    numeric = observed.loc[:, list(TARGET_COLS)].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise AssertionError("written submission has non-finite predictions")
    for group in TARGET_COLS:
        lower = (
            float(recipe["ensemble"]["clip"][group]["lower_capacity_fraction"])
            * CAPACITY_KWH[group]
        )
        upper = (
            float(recipe["ensemble"]["clip"][group]["upper_capacity_fraction"])
            * CAPACITY_KWH[group]
        )
        if (observed[group] < lower - 1e-6).any() or (
            observed[group] > upper + 1e-6
        ).any():
            raise AssertionError(f"written submission violates {group} clip")


def train_final(
    *,
    raw_dir: str | Path,
    cache_dir: str | Path,
    out_dir: str | Path,
    recipe_path: str | Path,
    overwrite: bool = False,
    project_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Train the locked recipe from cache/raw and write hashed final artifacts."""

    recipe_file = Path(recipe_path).expanduser().resolve()
    recipe = read_recipe(recipe_file)
    if not recipe["recipe_locked"]:
        raise ValueError(
            "recipe_locked is false; refusing expensive training/submission. "
            "Fill the validated v2 constants and set recipe_locked=true"
        )
    data = load_final_data(raw_dir, cache_dir)
    output_root = Path(out_dir).expanduser().resolve()
    paths = _output_paths(output_root, recipe)
    _preflight_outputs(paths, overwrite=overwrite)

    trained: dict[str, TrainedCandidate] = {}
    for candidate_name, raw_specification in recipe["models"].items():
        specification = dict(raw_specification)
        if not specification["enabled"]:
            continue
        print(f"training {candidate_name} ({specification['scope']}) ...", flush=True)
        candidate = train_candidate(
            candidate_name,
            specification,
            recipe=recipe,
            data=data,
            trained=trained,
        )
        trained[candidate_name] = candidate
        artifact_payload = {
            "artifact_schema_version": 1,
            "recipe_name": recipe["recipe_name"],
            "candidate_name": candidate_name,
            "candidate_config": specification,
            "scope": candidate.scope,
            "groups": TARGET_COLS,
            "training_rows": candidate.training_rows,
            "feature_names": candidate.feature_names,
            "models": candidate.models,
        }
        _atomic_joblib(artifact_payload, paths[f"model:{candidate_name}"])
        _atomic_parquet(
            candidate.predictions,
            paths[f"prediction:{candidate_name}"],
        )

    final_predictions = _assemble_final_predictions(recipe, trained)
    submission = _build_submission(data, final_predictions, recipe)
    _atomic_parquet(final_predictions, paths["final_predictions"])
    _atomic_submission_csv(submission, paths["submission"])
    _verify_written_submission(paths["submission"], submission, recipe)
    write_json_atomic(
        paths["recipe_snapshot"],
        recipe,
        overwrite=overwrite,
    )

    output_files = [
        path
        for key, path in paths.items()
        if key != "manifest"
    ]
    input_files = [recipe_file, *data.input_files]
    summaries = {
        candidate: {
            "scope": value.scope,
            "training_rows": value.training_rows,
            "feature_counts": {
                group: len(columns)
                for group, columns in value.feature_names.items()
            },
        }
        for candidate, value in trained.items()
    }
    manifest = make_manifest(
        artifact_type="baram_final_submission",
        parameters={
            "recipe": recipe,
            "capacities_kwh": CAPACITY_KWH,
            "target_columns": TARGET_COLS,
            "csv_encoding": "utf-8-sig",
            "csv_float_format": "%.6f",
        },
        input_files=input_files,
        output_files=output_files,
        results={
            "rows": int(len(submission)),
            "submission_sha256": sha256_file(paths["submission"]),
            "candidates": summaries,
            "prediction_ranges": {
                group: {
                    "min": float(final_predictions[group].min()),
                    "max": float(final_predictions[group].max()),
                }
                for group in TARGET_COLS
            },
        },
        project_dir=project_dir,
    )
    write_json_atomic(paths["manifest"], manifest, overwrite=overwrite)
    return {
        "submission": str(paths["submission"]),
        "manifest": str(paths["manifest"]),
        "submission_sha256": manifest["results"]["submission_sha256"],
        "rows": int(len(submission)),
    }


__all__ = [
    "FINAL_RECIPE_SCHEMA_VERSION",
    "FinalDataBundle",
    "OPTIONAL_V2_CANDIDATES",
    "REQUIRED_BASE_CANDIDATES",
    "TrainedCandidate",
    "load_final_data",
    "read_recipe",
    "schema_summary",
    "train_candidate",
    "train_final",
    "validate_recipe",
]

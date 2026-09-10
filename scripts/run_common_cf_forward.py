"""Strict-forward experiment for a timestamp-level common capacity factor.

The common target is the mean or median of the *available* official group
capacity factors at a timestamp (at least two groups are required).  One model
uses the element-wise mean of the three exogenous group weather caches.  Its
single capacity-factor forecast is converted back to each group's kWh and a
group-specific correction is fitted on expanding, train-only OOF predictions.

Selection is deliberately completed before 2024 labels are read:

* groups 1/2: 2022 train -> 2023 validation;
* group 3: 2023 H1 train -> 2023 H2 validation.

The selected target/objective/correction/blend weight is written to
``locked_pre2024_recipe.json``.  Only then is the same recipe trained through
2023 and checked on the already-consumed 2024 gate.  No 2024 alternative is
searched.  A 2025 CSV is possible only with ``--build-final-if-stable`` and
only if the pre-2024 and 2024 stability gates both pass.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import joblib
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.manifest import make_manifest, sha256_file, write_json_atomic  # noqa: E402
from src.metric import CAPACITY_KWH, TARGET_COLS, score_details  # noqa: E402
from src.postprocess import (  # noqa: E402
    FICRBoundedPostprocessor,
    GroupAffineCalibrator,
)


DEV_GROUPS = ("kpx_group_1", "kpx_group_2")
G3_GROUPS = ("kpx_group_3",)

DEV_TRAIN_START = pd.Timestamp("2022-01-01 01:00:00")
DEV_TRAIN_END = pd.Timestamp("2023-01-01 00:00:00")
DEV_VALID_START = pd.Timestamp("2023-01-01 01:00:00")
DEV_VALID_END = pd.Timestamp("2024-01-01 00:00:00")
DEV_HALF_SPLIT = pd.Timestamp("2023-07-01 01:00:00")

G3_TRAIN_START = pd.Timestamp("2023-01-01 01:00:00")
G3_TRAIN_END = pd.Timestamp("2023-07-01 00:00:00")
G3_VALID_START = pd.Timestamp("2023-07-01 01:00:00")
G3_VALID_END = pd.Timestamp("2024-01-01 00:00:00")
G3_HALF_SPLIT = pd.Timestamp("2023-10-01 01:00:00")

GATE_TRAIN_START = DEV_TRAIN_START
GATE_TRAIN_END = DEV_VALID_END
GATE_VALID_START = pd.Timestamp("2024-01-01 01:00:00")
GATE_VALID_END = pd.Timestamp("2025-01-01 00:00:00")
GATE_HALF_SPLIT = pd.Timestamp("2024-07-01 01:00:00")

FINAL_TRAIN_END = GATE_VALID_END
PRE2024_ROWS = 17_520
ALL_TRAIN_ROWS = 26_304
BLEND_WEIGHTS = (0.05, 0.10, 0.15, 0.20)
WIND_FEATURE_TOKENS = (
    "10u",
    "10v",
    "UGRD",
    "VGRD",
    "MUmax",
    "MUmin",
    "MVmax",
    "MVmin",
    "XBLWS",
    "YBLWS",
    "gust",
    "ws10",
    "ws80",
    "ws100",
    "hub_ws",
    "wind_power_density",
    "air_density",
    "wind_shear",
    "wind_dir",
    "wd_sin",
    "wd_cos",
)


@dataclass(frozen=True)
class CommonCandidate:
    name: str
    target_aggregate: str
    objective: str
    alpha: float | None


CANDIDATES = (
    CommonCandidate("mean_l1", "mean", "regression_l1", None),
    CommonCandidate("median_l1", "median", "regression_l1", None),
    CommonCandidate("mean_q07", "mean", "quantile", 0.70),
    CommonCandidate("median_q07", "median", "quantile", 0.70),
)
CORRECTIONS = ("raw", "affine", "ficr_bounded")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir", type=Path, default=Path(r"data/local/open")
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/cache"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/postgate/common_cf_forward"),
    )
    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--build-final-if-stable",
        action="store_true",
        help="write a 2025 CSV only when every strict stability condition passes",
    )
    return parser.parse_args(argv)


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        frame.to_parquet(temporary, engine="pyarrow", compression="zstd", index=True)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_joblib(payload: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        joblib.dump(payload, temporary, compress=3)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8-sig", float_format="%.6f")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _output_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "lock": out_dir / "locked_pre2024_recipe.json",
        "results": out_dir / "results.json",
        "manifest": out_dir / "manifest.json",
        "models": out_dir / "models" / "common_cf_models.joblib",
        "dev": out_dir / "oof" / "dev2023_common_cf_candidates.parquet",
        "g3": out_dir / "oof" / "g3_2023h2_common_cf_candidates.parquet",
        "gate": out_dir / "oof" / "gate2024_locked_common_cf.parquet",
        "calibration_dev": out_dir / "oof" / "dev2022_selected_inner_oof.parquet",
        "calibration_g3": out_dir / "oof" / "g3_2023h1_selected_inner_oof.parquet",
        "calibration_gate": out_dir / "oof" / "pre2024_selected_inner_oof.parquet",
        "final_prediction": out_dir / "final" / "common_cf_2025.parquet",
        "final_csv": out_dir / "final" / "common_cf_2025.csv",
    }


def _preflight(paths: Mapping[str, Path], overwrite: bool) -> None:
    required = [
        paths["lock"],
        paths["results"],
        paths["manifest"],
        paths["models"],
        paths["dev"],
        paths["g3"],
        paths["gate"],
    ]
    existing = [path for path in required if path.exists()]
    if existing and not overwrite:
        rendered = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(
            "common-CF outputs already exist; use --overwrite only for an "
            f"intentional reproducibility rerun:\n{rendered}"
        )


def _read_labels(path: Path, *, nrows: int | None) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(
        path,
        encoding="utf-8-sig",
        nrows=nrows,
        parse_dates=["kst_dtm"],
    ).set_index("kst_dtm")
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if tuple(frame.columns) != TARGET_COLS:
        raise ValueError(f"label columns must be {TARGET_COLS}")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("labels must use a unique sorted hourly index")
    if np.isinf(frame.to_numpy(dtype=float)).any():
        raise ValueError("labels contain infinite values")
    return frame.astype(float)


def _expected_index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="h", name="forecast_kst_dtm")


def _exact_slice(
    frame: pd.DataFrame | pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    context: str,
) -> pd.DataFrame | pd.Series:
    expected = _expected_index(start, end)
    result = frame.reindex(expected)
    if not result.index.equals(expected):
        raise AssertionError(f"{context} index construction changed")
    return result


def _load_regional_features(cache_dir: Path) -> tuple[pd.DataFrame, list[Path]]:
    paths = [cache_dir / f"{group}_weather_train.parquet" for group in TARGET_COLS]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing weather caches: {missing}")
    regional: pd.DataFrame | None = None
    canonical_index: pd.DatetimeIndex | None = None
    canonical_columns: pd.Index | None = None
    for path in paths:
        frame = pd.read_parquet(path).astype(np.float32)
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise TypeError(f"{path.name} must use a DatetimeIndex")
        if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
            raise ValueError(f"{path.name} index must be unique and sorted")
        if not np.isfinite(frame.to_numpy(copy=False)).all():
            raise ValueError(f"{path.name} contains non-finite weather values")
        if regional is None:
            regional = frame.copy()
            canonical_index = frame.index
            canonical_columns = frame.columns
        else:
            assert canonical_index is not None and canonical_columns is not None
            if not frame.index.equals(canonical_index):
                raise ValueError("group weather indexes differ")
            if not frame.columns.equals(canonical_columns):
                raise ValueError("group weather schemas differ")
            regional += frame
    assert regional is not None
    regional /= np.float32(len(TARGET_COLS))
    selected = [
        column
        for column in regional.columns
        if column.startswith("time__")
        or any(token.lower() in column.lower() for token in WIND_FEATURE_TOKENS)
    ]
    if len(selected) < 100:
        raise ValueError("physical wind feature filter unexpectedly selected too few columns")
    regional = regional.loc[:, selected]
    regional.index = pd.DatetimeIndex(regional.index, name="forecast_kst_dtm")
    return regional.astype(np.float32), paths


def _load_regional_test_features(cache_dir: Path) -> tuple[pd.DataFrame, list[Path]]:
    paths = [cache_dir / f"{group}_weather_test.parquet" for group in TARGET_COLS]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing test weather caches: {missing}")
    regional: pd.DataFrame | None = None
    for path in paths:
        frame = pd.read_parquet(path).astype(np.float32)
        if regional is None:
            regional = frame.copy()
        else:
            if not frame.index.equals(regional.index) or not frame.columns.equals(
                regional.columns
            ):
                raise ValueError("test weather caches are not aligned")
            regional += frame
    assert regional is not None
    regional /= np.float32(len(TARGET_COLS))
    selected = [
        column
        for column in regional.columns
        if column.startswith("time__")
        or any(token.lower() in column.lower() for token in WIND_FEATURE_TOKENS)
    ]
    regional = regional.loc[:, selected]
    if not np.isfinite(regional.to_numpy(copy=False)).all():
        raise ValueError("regional test weather contains non-finite values")
    regional.index = pd.DatetimeIndex(regional.index, name="forecast_kst_dtm")
    return regional.astype(np.float32), paths


def _common_target(labels: pd.DataFrame, aggregate: str) -> pd.Series:
    capacity = pd.Series(CAPACITY_KWH, dtype=float)
    cf = labels.loc[:, list(TARGET_COLS)].div(capacity, axis="columns")
    available = cf.notna().sum(axis=1)
    if aggregate == "mean":
        target = cf.mean(axis=1, skipna=True)
    elif aggregate == "median":
        target = cf.median(axis=1, skipna=True)
    else:
        raise ValueError(f"unsupported target aggregate {aggregate!r}")
    target = target.where(available >= 2)
    finite = target.notna()
    if finite.any() and not target.loc[finite].between(0.0, 1.10).all():
        raise ValueError("common capacity-factor target is physically implausible")
    return target.astype(float).rename(f"common_cf_{aggregate}")


def _model_params(
    candidate: CommonCandidate,
    *,
    n_estimators: int,
    learning_rate: float,
    seed: int,
    n_jobs: int,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "objective": candidate.objective,
        "n_estimators": int(n_estimators),
        "learning_rate": float(learning_rate),
        "num_leaves": 31,
        "min_child_samples": 40,
        "subsample": 0.80,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.05,
        "reg_lambda": 2.0,
        "random_state": int(seed),
        "n_jobs": int(n_jobs),
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
    }
    if candidate.alpha is not None:
        params["alpha"] = float(candidate.alpha)
    return params


def _fit_model(
    candidate: CommonCandidate,
    labels: pd.DataFrame,
    features: pd.DataFrame,
    train_index: pd.DatetimeIndex,
    *,
    params: Mapping[str, Any],
) -> LGBMRegressor:
    target = _common_target(labels, candidate.target_aggregate).reindex(train_index)
    selected = target.notna()
    if int(selected.sum()) < 1_000:
        raise ValueError(f"{candidate.name} has too few common-target rows")
    train_x = features.reindex(train_index).loc[selected]
    if train_x.isna().any().any():
        raise ValueError("weather feature coverage is incomplete in model fit")
    model = LGBMRegressor(**dict(params))
    model.fit(train_x, target.loc[selected])
    return model


def _predict_groups(
    model: LGBMRegressor,
    features: pd.DataFrame,
    index: pd.DatetimeIndex,
    groups: Sequence[str],
) -> pd.DataFrame:
    x = features.reindex(index)
    if x.isna().any().any():
        raise ValueError("weather feature coverage is incomplete in prediction")
    common_cf = np.asarray(model.predict(x), dtype=float)
    if not np.isfinite(common_cf).all():
        raise ValueError("common model produced non-finite predictions")
    output = pd.DataFrame(index=index, columns=groups, dtype=float)
    for group in groups:
        output[group] = np.clip(common_cf, 0.0, 1.02) * CAPACITY_KWH[group]
    return output


def _expanding_oof(
    candidate: CommonCandidate,
    labels: pd.DataFrame,
    features: pd.DataFrame,
    *,
    outer_start: pd.Timestamp,
    outer_end: pd.Timestamp,
    validation_starts: Sequence[pd.Timestamp],
    groups: Sequence[str],
    params: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    starts = tuple(pd.Timestamp(value) for value in validation_starts)
    if not starts or starts != tuple(sorted(starts)):
        raise ValueError("expanding validation starts must be sorted and non-empty")
    parts: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    for position, valid_start in enumerate(starts):
        valid_end = (
            starts[position + 1] - pd.Timedelta(hours=1)
            if position + 1 < len(starts)
            else outer_end
        )
        train_end = valid_start - pd.Timedelta(hours=1)
        train_index = _expected_index(outer_start, train_end)
        valid_index = _expected_index(valid_start, valid_end)
        if train_index.max() >= valid_index.min():
            raise AssertionError("inner expanding train/validation overlap")
        model = _fit_model(
            candidate, labels, features, train_index, params=params
        )
        prediction = _predict_groups(model, features, valid_index, groups)
        parts.append(prediction)
        folds.append(
            {
                "train_start": str(train_index.min()),
                "train_end": str(train_index.max()),
                "valid_start": str(valid_index.min()),
                "valid_end": str(valid_index.max()),
                "train_common_target_rows": int(
                    _common_target(labels, candidate.target_aggregate)
                    .reindex(train_index)
                    .notna()
                    .sum()
                ),
                "valid_rows": int(len(valid_index)),
            }
        )
    output = pd.concat(parts, axis=0)
    if not output.index.is_unique or not output.index.is_monotonic_increasing:
        raise AssertionError("inner OOF is not unique and chronological")
    return output, folds


def _clip_groups(frame: pd.DataFrame, groups: Sequence[str]) -> pd.DataFrame:
    output = frame.loc[:, list(groups)].copy()
    for group in groups:
        output[group] = output[group].clip(0.0, CAPACITY_KWH[group] * 1.02)
    return output


def _fit_corrections(
    labels: pd.DataFrame,
    calibration_raw: pd.DataFrame,
    application_raw: pd.DataFrame,
    groups: Sequence[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    actual = labels.reindex(calibration_raw.index).loc[:, list(groups)]
    predictions: dict[str, pd.DataFrame] = {
        "raw": _clip_groups(application_raw, groups)
    }
    metadata: dict[str, Any] = {}

    affine = GroupAffineCalibrator(
        groups=groups,
        scale_bounds=(0.80, 1.30),
        bias_fraction_bounds=(-0.12, 0.12),
        grid_size=25,
        refinements=1,
    ).fit(actual, calibration_raw)
    predictions["affine"] = _clip_groups(affine.transform(application_raw), groups)
    metadata["affine"] = {
        "parameters": affine.parameters_as_dict(),
        "fit_diagnostics": {
            group: asdict(value) for group, value in affine.diagnostics_.items()
        },
    }

    bounded = FICRBoundedPostprocessor(
        groups=groups,
        scales=(0.94, 0.97, 1.0, 1.03, 1.06),
        bias_fractions=(-0.04, -0.02, -0.01, 0.0, 0.01, 0.02, 0.04),
        lower_capacity_fractions=(0.0,),
        upper_capacity_fractions=(1.0, 1.02),
    ).fit(actual, calibration_raw)
    predictions["ficr_bounded"] = bounded.transform(application_raw)
    metadata["ficr_bounded"] = {
        "parameters": bounded.parameters_as_dict(),
        "fit_diagnostics": {
            group: asdict(value) for group, value in bounded.diagnostics_.items()
        },
    }
    return predictions, {"objects": {"affine": affine, "ficr_bounded": bounded}, **metadata}


def _run_outer_candidates(
    labels: pd.DataFrame,
    features: pd.DataFrame,
    *,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    valid_start: pd.Timestamp,
    valid_end: pd.Timestamp,
    inner_validation_starts: Sequence[pd.Timestamp],
    groups: Sequence[str],
    base_params: Mapping[str, Any],
) -> tuple[dict[str, dict[str, pd.DataFrame]], dict[str, Any]]:
    train_index = _expected_index(train_start, train_end)
    valid_index = _expected_index(valid_start, valid_end)
    if train_index.max() >= valid_index.min():
        raise AssertionError("outer train and validation overlap")
    outputs: dict[str, dict[str, pd.DataFrame]] = {}
    fitted: dict[str, Any] = {}
    for candidate in CANDIDATES:
        started = time.perf_counter()
        params = dict(base_params)
        params["objective"] = candidate.objective
        if candidate.alpha is None:
            params.pop("alpha", None)
        else:
            params["alpha"] = candidate.alpha
        calibration_raw, folds = _expanding_oof(
            candidate,
            labels,
            features,
            outer_start=train_start,
            outer_end=train_end,
            validation_starts=inner_validation_starts,
            groups=groups,
            params=params,
        )
        model = _fit_model(candidate, labels, features, train_index, params=params)
        application_raw = _predict_groups(model, features, valid_index, groups)
        corrected, correction_meta = _fit_corrections(
            labels, calibration_raw, application_raw, groups
        )
        outputs[candidate.name] = corrected
        fitted[candidate.name] = {
            "candidate": asdict(candidate),
            "model": model,
            "correction_objects": correction_meta.pop("objects"),
            "correction_metadata": correction_meta,
            "inner_folds": folds,
            "inner_oof": calibration_raw,
            "runtime_seconds": time.perf_counter() - started,
        }
        print(
            f"completed {candidate.name} {train_start.date()}->{valid_start.date()} "
            f"in {fitted[candidate.name]['runtime_seconds']:.1f}s",
            flush=True,
        )
    return outputs, fitted


def _run_locked_outer(
    candidate: CommonCandidate,
    correction: str,
    labels: pd.DataFrame,
    features: pd.DataFrame,
    *,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    valid_start: pd.Timestamp,
    valid_end: pd.Timestamp,
    inner_validation_starts: Sequence[pd.Timestamp],
    groups: Sequence[str],
    params: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train_index = _expected_index(train_start, train_end)
    valid_index = _expected_index(valid_start, valid_end)
    calibration_raw, folds = _expanding_oof(
        candidate,
        labels,
        features,
        outer_start=train_start,
        outer_end=train_end,
        validation_starts=inner_validation_starts,
        groups=groups,
        params=params,
    )
    model = _fit_model(candidate, labels, features, train_index, params=params)
    application_raw = _predict_groups(model, features, valid_index, groups)
    corrected, correction_meta = _fit_corrections(
        labels, calibration_raw, application_raw, groups
    )
    return corrected[correction], {
        "candidate": asdict(candidate),
        "model": model,
        "correction": correction,
        "correction_objects": correction_meta.pop("objects"),
        "correction_metadata": correction_meta,
        "inner_folds": folds,
        "inner_oof": calibration_raw,
    }


def _score_slices(
    actual: pd.DataFrame,
    prediction: pd.DataFrame,
    groups: Sequence[str],
    half_split: pd.Timestamp,
) -> dict[str, Any]:
    slices = {
        "full": prediction.index,
        "first_half": prediction.index[prediction.index < half_split],
        "second_half": prediction.index[prediction.index >= half_split],
    }
    output: dict[str, Any] = {}
    for name, index in slices.items():
        metrics = score_details(
            actual.reindex(index),
            prediction.reindex(index),
            target_cols=groups,
        )
        output[name] = asdict(metrics)
    return output


def _blend(
    baseline: pd.DataFrame,
    common: pd.DataFrame,
    weight: float,
    groups: Sequence[str],
) -> pd.DataFrame:
    if not baseline.index.equals(common.index):
        raise ValueError("baseline/common blend indexes differ")
    output = (1.0 - float(weight)) * baseline.loc[:, list(groups)] + float(
        weight
    ) * common.loc[:, list(groups)]
    return _clip_groups(output, groups)


def _load_pre2024_baselines(artifact_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    dev_path = artifact_dir / "oof" / "dev2023_locked_v3.parquet"
    g3_path = artifact_dir / "oof" / "g3dev2023h2_candidates.parquet"
    if not dev_path.is_file() or not g3_path.is_file():
        raise FileNotFoundError("locked historical baseline OOF artifacts are missing")
    dev = pd.read_parquet(dev_path).loc[:, list(DEV_GROUPS)]
    raw = pd.read_parquet(g3_path)
    required = {"q07", "shared_l1", "shared_q07", "top200q07", "ewq06"}
    if not required.issubset(raw.columns):
        raise ValueError("group3 historical components do not match v3 recipe")
    weighted = (
        0.20 * raw["q07"]
        + 0.075 * raw["shared_l1"]
        + 0.425 * raw["shared_q07"]
        + 0.025 * raw["top200q07"]
        + 0.275 * raw["ewq06"]
    )
    g3 = pd.DataFrame(
        {
            "kpx_group_3": np.clip(
                1.25 * weighted - 1_200.0,
                0.0,
                CAPACITY_KWH["kpx_group_3"] * 1.02,
            )
        },
        index=raw.index,
    )
    expected_dev = _expected_index(DEV_VALID_START, DEV_VALID_END)
    expected_g3 = _expected_index(G3_VALID_START, G3_VALID_END)
    if not dev.index.equals(expected_dev) or not g3.index.equals(expected_g3):
        raise ValueError("historical baseline indexes differ from locked outer folds")
    return dev, g3, [dev_path, g3_path]


def _load_gate_baselines(artifact_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    v3_path = artifact_dir / "oof" / "gate2024_locked_v3_cf_fix.parquet"
    v4_path = artifact_dir / "oof" / "gate2024_recent_v4_cf_fix_calibration_fit.parquet"
    if not v3_path.is_file() or not v4_path.is_file():
        raise FileNotFoundError("2024 v3/v4 comparison artifacts are missing")
    v3 = pd.read_parquet(v3_path).loc[:, list(TARGET_COLS)]
    v4 = pd.read_parquet(v4_path).loc[:, list(TARGET_COLS)]
    expected = _expected_index(GATE_VALID_START, GATE_VALID_END)
    if not v3.index.equals(expected) or not v4.index.equals(expected):
        raise ValueError("2024 baseline indexes differ from fixed gate interval")
    return v3, v4, [v3_path, v4_path]


def _flatten_predictions(
    baseline: pd.DataFrame,
    candidates: Mapping[str, Mapping[str, pd.DataFrame]],
    groups: Sequence[str],
) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    for group in groups:
        columns[f"baseline_v3__{group}"] = baseline[group]
    for candidate_name, correction_map in candidates.items():
        for correction, prediction in correction_map.items():
            for group in groups:
                columns[f"common__{candidate_name}__{correction}__{group}"] = prediction[
                    group
                ]
            for weight in BLEND_WEIGHTS:
                blended = _blend(baseline, prediction, weight, groups)
                key = f"blend__{candidate_name}__{correction}__w{int(weight * 100):02d}"
                for group in groups:
                    columns[f"{key}__{group}"] = blended[group]
    return pd.DataFrame(columns, index=baseline.index)


def _pre2024_candidate_scores(
    labels: pd.DataFrame,
    dev_baseline: pd.DataFrame,
    g3_baseline: pd.DataFrame,
    dev_candidates: Mapping[str, Mapping[str, pd.DataFrame]],
    g3_candidates: Mapping[str, Mapping[str, pd.DataFrame]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    dev_actual = labels.reindex(dev_baseline.index)
    g3_actual = labels.reindex(g3_baseline.index)
    baseline_scores = {
        "dev_2022_to_2023": _score_slices(
            dev_actual, dev_baseline, DEV_GROUPS, DEV_HALF_SPLIT
        ),
        "g3_2023_h1_to_h2": _score_slices(
            g3_actual, g3_baseline, G3_GROUPS, G3_HALF_SPLIT
        ),
    }
    candidate_scores: dict[str, Any] = {}
    ranking_rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        for correction in CORRECTIONS:
            common_dev = dev_candidates[candidate.name][correction]
            common_g3 = g3_candidates[candidate.name][correction]
            common_key = f"{candidate.name}__{correction}"
            candidate_scores[f"common__{common_key}"] = {
                "dev_2022_to_2023": _score_slices(
                    dev_actual, common_dev, DEV_GROUPS, DEV_HALF_SPLIT
                ),
                "g3_2023_h1_to_h2": _score_slices(
                    g3_actual, common_g3, G3_GROUPS, G3_HALF_SPLIT
                ),
            }
            for weight in BLEND_WEIGHTS:
                dev_blend = _blend(dev_baseline, common_dev, weight, DEV_GROUPS)
                g3_blend = _blend(g3_baseline, common_g3, weight, G3_GROUPS)
                key = f"{common_key}__w{int(weight * 100):02d}"
                score = {
                    "dev_2022_to_2023": _score_slices(
                        dev_actual, dev_blend, DEV_GROUPS, DEV_HALF_SPLIT
                    ),
                    "g3_2023_h1_to_h2": _score_slices(
                        g3_actual, g3_blend, G3_GROUPS, G3_HALF_SPLIT
                    ),
                }
                candidate_scores[key] = score
                dev_delta = (
                    score["dev_2022_to_2023"]["full"]["total_score"]
                    - baseline_scores["dev_2022_to_2023"]["full"]["total_score"]
                )
                g3_delta = (
                    score["g3_2023_h1_to_h2"]["full"]["total_score"]
                    - baseline_scores["g3_2023_h1_to_h2"]["full"]["total_score"]
                )
                ranking_rows.append(
                    {
                        "key": key,
                        "candidate": candidate.name,
                        "correction": correction,
                        "blend_weight": weight,
                        "dev_full_delta": dev_delta,
                        "g3_full_delta": g3_delta,
                        "minimum_full_delta": min(dev_delta, g3_delta),
                        "mean_full_delta": 0.5 * (dev_delta + g3_delta),
                    }
                )
    # Robust predeclared selection: maximise the weaker of the two full-period
    # forward deltas, then their mean. Smaller blends break exact ties.
    ranking_rows.sort(
        key=lambda row: (
            row["minimum_full_delta"],
            row["mean_full_delta"],
            -row["blend_weight"],
            row["key"],
        ),
        reverse=True,
    )
    return baseline_scores, candidate_scores, {
        "selected": ranking_rows[0],
        "ranking": ranking_rows,
    }


def _all_slice_group_deltas_positive(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any]
) -> tuple[bool, dict[str, float]]:
    deltas: dict[str, float] = {}
    for slice_name in ("full", "first_half", "second_half"):
        candidate_slice = candidate[slice_name]
        baseline_slice = baseline[slice_name]
        deltas[f"aggregate__{slice_name}"] = (
            candidate_slice["total_score"] - baseline_slice["total_score"]
        )
        for group in candidate_slice["by_group"]:
            c = candidate_slice["by_group"][group]
            b = baseline_slice["by_group"][group]
            c_score = 0.5 * c["one_minus_nmae"] + 0.5 * c["ficr"]
            b_score = 0.5 * b["one_minus_nmae"] + 0.5 * b["ficr"]
            deltas[f"{group}__{slice_name}"] = c_score - b_score
    return all(value > 0.0 for value in deltas.values()), deltas


def _candidate_from_name(name: str) -> CommonCandidate:
    matches = [candidate for candidate in CANDIDATES if candidate.name == name]
    if len(matches) != 1:
        raise ValueError(f"unknown locked candidate {name!r}")
    return matches[0]


def _final_if_stable(
    *,
    args: argparse.Namespace,
    paths: Mapping[str, Path],
    labels: pd.DataFrame,
    regional_train: pd.DataFrame,
    candidate: CommonCandidate,
    correction: str,
    blend_weight: float,
    params: Mapping[str, Any],
    use_v4: bool,
) -> tuple[dict[str, Any], list[Path], Any | None]:
    if not args.build_final_if_stable:
        return {"status": "not_requested"}, [], None
    regional_test, test_paths = _load_regional_test_features(args.cache_dir)
    sample_path = args.raw_dir / "sample_submission.csv"
    sample = pd.read_csv(
        sample_path,
        encoding="utf-8-sig",
        dtype={"forecast_id": "string", "forecast_kst_dtm": "string"},
    )
    sample_index = pd.DatetimeIndex(
        pd.to_datetime(sample["forecast_kst_dtm"], errors="raise"),
        name="forecast_kst_dtm",
    )
    if not regional_test.index.equals(sample_index):
        raise ValueError("test feature index differs from sample submission")
    # Every calibration prediction is made by a model trained strictly before
    # its block.  No 2025 target exists or is read.
    combined_features = pd.concat([regional_train, regional_test], axis=0)
    if not combined_features.index.is_unique or not combined_features.index.is_monotonic_increasing:
        raise ValueError("combined train/test regional weather index is not unique and sorted")
    common_prediction, fitted = _run_locked_outer(
        candidate,
        correction,
        labels,
        combined_features,
        train_start=DEV_TRAIN_START,
        train_end=FINAL_TRAIN_END,
        valid_start=sample_index.min(),
        valid_end=sample_index.max(),
        inner_validation_starts=(
            pd.Timestamp("2023-01-01 01:00:00"),
            pd.Timestamp("2023-07-01 01:00:00"),
            pd.Timestamp("2024-01-01 01:00:00"),
            pd.Timestamp("2024-07-01 01:00:00"),
        ),
        groups=TARGET_COLS,
        params=params,
    )
    baseline_name = "corrected_recent_v4" if use_v4 else "corrected_v3"
    baseline_path = (
        args.artifact_dir
        / "final_cf_fix"
        / "predictions"
        / f"{baseline_name}_test.parquet"
    )
    baseline = pd.read_parquet(baseline_path).loc[:, list(TARGET_COLS)]
    final_prediction = _blend(
        baseline, common_prediction, blend_weight, TARGET_COLS
    )
    _atomic_parquet(final_prediction, paths["final_prediction"])
    output = sample.copy()
    for group in TARGET_COLS:
        output[group] = final_prediction[group].to_numpy(dtype=float)
    _atomic_csv(output, paths["final_csv"])
    return {
        "status": "written_after_stability_pass",
        "baseline": baseline_name,
        "prediction": str(paths["final_prediction"]),
        "csv": str(paths["final_csv"]),
        "csv_sha256": sha256_file(paths["final_csv"]),
    }, [*test_paths, sample_path, baseline_path], fitted


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.raw_dir = args.raw_dir.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.artifact_dir = args.artifact_dir.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    paths = _output_paths(args.out_dir)
    _preflight(paths, bool(args.overwrite))
    started = time.perf_counter()

    label_path = args.raw_dir / "train" / "train_labels.csv"
    # This first read physically stops before 2024. Candidate/weight selection
    # cannot inspect a 2024 target value.
    pre_labels = _read_labels(label_path, nrows=PRE2024_ROWS)
    if len(pre_labels) != PRE2024_ROWS or pre_labels.index.max() != DEV_VALID_END:
        raise ValueError("pre-2024 label read boundary changed")
    regional, cache_paths = _load_regional_features(args.cache_dir)
    if regional.index.min() != DEV_TRAIN_START or regional.index.max() != FINAL_TRAIN_END:
        raise ValueError("train weather cache time coverage changed")
    dev_baseline, g3_baseline, historical_baseline_paths = _load_pre2024_baselines(
        args.artifact_dir
    )

    base_params = _model_params(
        CANDIDATES[0],
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )
    print("pre-2024 phase: fitting 2022 -> 2023 candidates", flush=True)
    dev_candidates, dev_fitted = _run_outer_candidates(
        pre_labels,
        regional,
        train_start=DEV_TRAIN_START,
        train_end=DEV_TRAIN_END,
        valid_start=DEV_VALID_START,
        valid_end=DEV_VALID_END,
        inner_validation_starts=(
            pd.Timestamp("2022-04-01 01:00:00"),
            pd.Timestamp("2022-07-01 01:00:00"),
            pd.Timestamp("2022-10-01 01:00:00"),
        ),
        groups=DEV_GROUPS,
        base_params=base_params,
    )
    print("pre-2024 phase: fitting 2023 H1 -> H2 candidates", flush=True)
    g3_candidates, g3_fitted = _run_outer_candidates(
        pre_labels,
        regional,
        train_start=G3_TRAIN_START,
        train_end=G3_TRAIN_END,
        valid_start=G3_VALID_START,
        valid_end=G3_VALID_END,
        inner_validation_starts=(
            pd.Timestamp("2023-03-01 01:00:00"),
            pd.Timestamp("2023-04-01 01:00:00"),
            pd.Timestamp("2023-05-01 01:00:00"),
            pd.Timestamp("2023-06-01 01:00:00"),
        ),
        groups=G3_GROUPS,
        base_params=base_params,
    )
    baseline_scores, candidate_scores, selection = _pre2024_candidate_scores(
        pre_labels,
        dev_baseline,
        g3_baseline,
        dev_candidates,
        g3_candidates,
    )
    selected = selection["selected"]
    selected_key = selected["key"]
    selected_score = candidate_scores[selected_key]
    dev_stable, dev_deltas = _all_slice_group_deltas_positive(
        selected_score["dev_2022_to_2023"], baseline_scores["dev_2022_to_2023"]
    )
    g3_stable, g3_deltas = _all_slice_group_deltas_positive(
        selected_score["g3_2023_h1_to_h2"], baseline_scores["g3_2023_h1_to_h2"]
    )
    pre2024_stable = dev_stable and g3_stable

    locked_candidate = _candidate_from_name(selected["candidate"])
    locked_params = _model_params(
        locked_candidate,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )
    locked_recipe = {
        "schema_version": 1,
        "selection_data_max_timestamp": str(pre_labels.index.max()),
        "selection_used_2024_labels": False,
        "feature_mode": "elementwise_mean_of_three_group_weather_caches_then_physical_wind_filter",
        "feature_count": int(regional.shape[1]),
        "feature_filter_tokens": list(WIND_FEATURE_TOKENS),
        "common_target": {
            "aggregate": locked_candidate.target_aggregate,
            "minimum_available_groups": 2,
            "unit": "capacity_factor",
        },
        "candidate": asdict(locked_candidate),
        "correction": selected["correction"],
        "blend_baseline": "locked_v3_capacity_factor_corrected",
        "blend_weight_common": selected["blend_weight"],
        "selection_rule": "maximise minimum full-period delta over two strict-forward folds; then mean delta; then smaller weight",
        "lgbm_params": locked_params,
        "pre2024_stability_passed": pre2024_stable,
    }
    write_json_atomic(paths["lock"], locked_recipe, overwrite=bool(args.overwrite))
    lock_sha256 = sha256_file(paths["lock"])
    print(
        f"locked before 2024 read: {selected_key} sha256={lock_sha256}", flush=True
    )

    _atomic_parquet(
        _flatten_predictions(dev_baseline, dev_candidates, DEV_GROUPS), paths["dev"]
    )
    _atomic_parquet(
        _flatten_predictions(g3_baseline, g3_candidates, G3_GROUPS), paths["g3"]
    )
    _atomic_parquet(
        dev_fitted[locked_candidate.name]["inner_oof"], paths["calibration_dev"]
    )
    _atomic_parquet(
        g3_fitted[locked_candidate.name]["inner_oof"], paths["calibration_g3"]
    )

    # Post-lock only: the gate is already consumed by the project, so this is a
    # fixed post-gate confirmation rather than a new untouched validation.
    all_labels = _read_labels(label_path, nrows=None)
    if len(all_labels) != ALL_TRAIN_ROWS or all_labels.index.max() != FINAL_TRAIN_END:
        raise ValueError("full official label boundary changed")
    if not all_labels.iloc[:PRE2024_ROWS].equals(pre_labels):
        raise AssertionError("pre-2024 labels changed between selection and gate read")
    if sha256_file(paths["lock"]) != lock_sha256:
        raise RuntimeError("locked common-CF recipe changed before 2024 evaluation")

    print("post-lock phase: fixed recipe train-through-2023 -> 2024", flush=True)
    gate_common, gate_fitted = _run_locked_outer(
        locked_candidate,
        selected["correction"],
        all_labels,
        regional,
        train_start=GATE_TRAIN_START,
        train_end=GATE_TRAIN_END,
        valid_start=GATE_VALID_START,
        valid_end=GATE_VALID_END,
        inner_validation_starts=(
            pd.Timestamp("2022-07-01 01:00:00"),
            pd.Timestamp("2023-01-01 01:00:00"),
            pd.Timestamp("2023-07-01 01:00:00"),
        ),
        groups=TARGET_COLS,
        params=locked_params,
    )
    gate_v3, gate_v4, gate_baseline_paths = _load_gate_baselines(args.artifact_dir)
    locked_weight = float(selected["blend_weight"])
    gate_v3_blend = _blend(gate_v3, gate_common, locked_weight, TARGET_COLS)
    gate_v4_blend = _blend(gate_v4, gate_common, locked_weight, TARGET_COLS)
    gate_actual = all_labels.reindex(gate_common.index)
    gate_scores = {
        "v3_baseline": _score_slices(
            gate_actual, gate_v3, TARGET_COLS, GATE_HALF_SPLIT
        ),
        "v4_baseline_calibration_fit_not_selection_safe": _score_slices(
            gate_actual, gate_v4, TARGET_COLS, GATE_HALF_SPLIT
        ),
        "locked_common": _score_slices(
            gate_actual, gate_common, TARGET_COLS, GATE_HALF_SPLIT
        ),
        "locked_v3_blend": _score_slices(
            gate_actual, gate_v3_blend, TARGET_COLS, GATE_HALF_SPLIT
        ),
        "locked_v4_blend_diagnostic": _score_slices(
            gate_actual, gate_v4_blend, TARGET_COLS, GATE_HALF_SPLIT
        ),
    }
    gate_v3_stable, gate_v3_deltas = _all_slice_group_deltas_positive(
        gate_scores["locked_v3_blend"], gate_scores["v3_baseline"]
    )
    gate_v4_stable, gate_v4_deltas = _all_slice_group_deltas_positive(
        gate_scores["locked_v4_blend_diagnostic"],
        gate_scores["v4_baseline_calibration_fit_not_selection_safe"],
    )
    adoption_recommended = pre2024_stable and gate_v3_stable

    gate_output = pd.DataFrame(index=gate_common.index)
    for name, frame in (
        ("v3", gate_v3),
        ("v4", gate_v4),
        ("common", gate_common),
        ("v3_blend", gate_v3_blend),
        ("v4_blend", gate_v4_blend),
    ):
        for group in TARGET_COLS:
            gate_output[f"{name}__{group}"] = frame[group]
    _atomic_parquet(gate_output, paths["gate"])
    _atomic_parquet(gate_fitted["inner_oof"], paths["calibration_gate"])

    final_result: dict[str, Any] = {
        "status": "blocked_by_stability_gate"
        if not adoption_recommended
        else "eligible_but_not_requested"
    }
    final_inputs: list[Path] = []
    final_fitted: Any | None = None
    if adoption_recommended:
        final_result, final_inputs, final_fitted = _final_if_stable(
            args=args,
            paths=paths,
            labels=all_labels,
            regional_train=regional,
            candidate=locked_candidate,
            correction=selected["correction"],
            blend_weight=locked_weight,
            params=locked_params,
            # v4 remains comparison-only because its existing formula was
            # calibration-fitted on 2024; it is never selected for this strict
            # chain even if its diagnostic delta happens to be positive.
            use_v4=False,
        )

    # Avoid serialising all searched inner models. The selected outer models,
    # OOF-derived correction objects, and fold metadata are sufficient to audit
    # and reproduce the locked path.
    model_payload = {
        "schema_version": 1,
        "locked_recipe_sha256": lock_sha256,
        "development_2022_to_2023": {
            key: value
            for key, value in dev_fitted[locked_candidate.name].items()
            if key != "inner_oof"
        },
        "historical_g3_2023_h1_to_h2": {
            key: value
            for key, value in g3_fitted[locked_candidate.name].items()
            if key != "inner_oof"
        },
        "fixed_2024_confirmation": {
            key: value for key, value in gate_fitted.items() if key != "inner_oof"
        },
        "final_2025": final_fitted,
    }
    _atomic_joblib(model_payload, paths["models"])

    results = {
        "experiment": "strict_forward_timestamp_common_capacity_factor",
        "status": "adopt" if adoption_recommended else "reject",
        "adoption_recommended": adoption_recommended,
        "pre2024": {
            "baseline_scores": baseline_scores,
            "candidate_scores": candidate_scores,
            "selection": selection,
            "selected_key": selected_key,
            "selected_scores": selected_score,
            "stability": {
                "passed": pre2024_stable,
                "dev_passed": dev_stable,
                "g3_passed": g3_stable,
                "dev_deltas": dev_deltas,
                "g3_deltas": g3_deltas,
            },
        },
        "fixed_2024_confirmation": {
            "warning": "2024 gate was already consumed; recipe was locked before these labels were read and was not retuned",
            "selection_or_tuning_performed": False,
            "locked_recipe_sha256": lock_sha256,
            "scores": gate_scores,
            "v3_stability": {"passed": gate_v3_stable, "deltas": gate_v3_deltas},
            "v4_incremental_diagnostic": {
                "passed": gate_v4_stable,
                "deltas": gate_v4_deltas,
                "selection_safe": False,
                "reason": "the existing v4 baseline was calibration-fitted on 2024",
            },
        },
        "final_2025": final_result,
        "leakage_audit": {
            "selection_label_read_nrows": PRE2024_ROWS,
            "selection_max_timestamp": str(pre_labels.index.max()),
            "lock_written_before_full_label_read": True,
            "lock_sha256_rechecked_before_gate": True,
            "inner_calibration": "strict expanding OOF; every calibration prediction trained only on earlier timestamps",
            "validation_weather_only": True,
            "validation_current_scada_used": False,
            "2024_candidate_search_count": 0,
            "2024_blend_weight_search_count": 0,
            "v4_baseline_selection_safe": False,
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json_atomic(paths["results"], results, overwrite=bool(args.overwrite))

    output_files = [
        paths["lock"],
        paths["results"],
        paths["models"],
        paths["dev"],
        paths["g3"],
        paths["gate"],
        paths["calibration_dev"],
        paths["calibration_g3"],
        paths["calibration_gate"],
    ]
    if paths["final_prediction"].is_file():
        output_files.extend([paths["final_prediction"], paths["final_csv"]])
    manifest = make_manifest(
        artifact_type="baram_strict_forward_common_capacity_factor",
        parameters={
            "locked_recipe": locked_recipe,
            "blend_weights_searched_pre2024": BLEND_WEIGHTS,
            "candidate_names": [candidate.name for candidate in CANDIDATES],
            "corrections": CORRECTIONS,
            "2024_selection_or_tuning_performed": False,
            "final_requires_stability_gate": True,
        },
        input_files=[
            label_path,
            *cache_paths,
            *historical_baseline_paths,
            *gate_baseline_paths,
            *final_inputs,
        ],
        output_files=output_files,
        results={
            "adoption_recommended": adoption_recommended,
            "selected_key": selected_key,
            "pre2024_stability_passed": pre2024_stable,
            "fixed_2024_v3_stability_passed": gate_v3_stable,
            "final_2025": final_result,
            "runtime_seconds": results["runtime_seconds"],
        },
        project_dir=PROJECT_DIR,
    )
    write_json_atomic(paths["manifest"], manifest, overwrite=bool(args.overwrite))

    print(
        json.dumps(
            {
                "selected": selected,
                "pre2024_stable": pre2024_stable,
                "gate_v3_stable": gate_v3_stable,
                "gate_v4_incremental_stable": gate_v4_stable,
                "adoption_recommended": adoption_recommended,
                "results": str(paths["results"]),
                "runtime_seconds": results["runtime_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

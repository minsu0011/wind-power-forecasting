"""Leakage-safe cross-farm pseudo-label transfer for KPX group 3.

Only two operations may inspect group-1/group-2 actual generation:

1. fit a mapper on timestamps explicitly declared as fold training; and
2. apply that mapper to explicitly declared 2022 timestamps to create group-3
   pseudo labels.

The downstream weather model receives group-3 weather columns and a one-
dimensional group-3 capacity-factor target only.  In particular, validation or
test group-1/group-2 actuals can never become downstream features.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

from .metric import CAPACITY_KWH


GROUP1 = "kpx_group_1"
GROUP2 = "kpx_group_2"
GROUP3 = "kpx_group_3"
TRANSFER_COLUMNS = (GROUP1, GROUP2, GROUP3)
FORBIDDEN_WEATHER = re.compile(
    r"(^kpx_group_|scada|target|label|power_kw)", re.IGNORECASE
)


@dataclass(frozen=True)
class MapperConfig:
    loss: str = "absolute_error"
    learning_rate: float = 0.05
    max_iter: int = 200
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 80
    l2_regularization: float = 5.0
    random_state: int = 42
    pseudo_clip_lower: float = 0.0
    pseudo_clip_upper: float = 1.02


@dataclass(frozen=True)
class WeatherModelConfig:
    n_estimators: int = 400
    learning_rate: float = 0.04
    num_leaves: int = 31
    min_child_samples: int = 30
    subsample: float = 0.80
    subsample_freq: int = 1
    colsample_bytree: float = 0.80
    reg_alpha: float = 0.05
    reg_lambda: float = 2.0
    random_state: int = 42
    n_jobs: int = 7
    eligible_fraction: float = 0.10


@dataclass
class PseudoLabelBundle:
    pseudo_cf: pd.Series
    mapper: HistGradientBoostingRegressor
    audit: dict[str, Any]


@dataclass
class WeatherTrainingBundle:
    features: pd.DataFrame
    target_cf: np.ndarray
    sample_weight: np.ndarray
    ordered_index: pd.DatetimeIndex
    actual_index: pd.DatetimeIndex
    pseudo_index: pd.DatetimeIndex


@dataclass
class ActualAccessLedger:
    """Record every materialized actual-label frame used by strict transfer.

    The ledger checks timestamp overlap *before* callers inspect frame values.
    It is deliberately passed through both the bounded CSV reader and the
    strict mapper API so the artifact can distinguish values that were merely
    available in the source CSV from values that were actually materialized.
    """

    events: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        frame: pd.DataFrame,
        *,
        purpose: str,
        forbidden_g12_index: Sequence[Any] | pd.Index = (),
    ) -> None:
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise TypeError("actual access frames must use a DatetimeIndex")
        forbidden = _as_datetime_index(
            forbidden_g12_index, name=f"{purpose} forbidden g1/g2"
        )
        g12_columns = tuple(
            column for column in (GROUP1, GROUP2) if column in frame.columns
        )
        overlap = (
            frame.index.intersection(forbidden)
            if g12_columns and len(forbidden)
            else pd.DatetimeIndex([], name="forecast_kst_dtm")
        )
        if len(overlap):
            raise ValueError(
                f"{purpose}: forbidden validation/test g1/g2 actual rows "
                f"would be materialized: {overlap[:3].tolist()}"
            )
        self.events.append(
            {
                "purpose": purpose,
                "columns": [str(column) for column in frame.columns],
                "rows": int(len(frame)),
                "first_timestamp": (
                    frame.index.min().isoformat() if len(frame) else None
                ),
                "last_timestamp": (
                    frame.index.max().isoformat() if len(frame) else None
                ),
                "g1g2_columns": list(g12_columns),
                "forbidden_g1g2_rows_materialized": 0,
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "events": list(self.events),
            "event_count": int(len(self.events)),
            "forbidden_g1g2_rows_materialized": int(
                sum(
                    event["forbidden_g1g2_rows_materialized"]
                    for event in self.events
                )
            ),
        }


def _as_datetime_index(values: Sequence[Any] | pd.Index, *, name: str) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(values, name="forecast_kst_dtm")
    if not index.is_unique:
        raise ValueError(f"{name} timestamps are duplicated")
    return index


def _validate_labels(labels: pd.DataFrame) -> None:
    missing = set(TRANSFER_COLUMNS).difference(labels.columns)
    if missing:
        raise ValueError(f"transfer labels missing columns: {sorted(missing)}")
    if not isinstance(labels.index, pd.DatetimeIndex):
        raise TypeError("transfer labels must use a DatetimeIndex")
    if not labels.index.is_unique or not labels.index.is_monotonic_increasing:
        raise ValueError("transfer label timestamps must be unique and sorted")


def read_bounded_actuals(
    path: str | Path,
    *,
    columns: Sequence[str],
    expected_index: Sequence[Any] | pd.Index,
    ledger: ActualAccessLedger,
    purpose: str,
    forbidden_g12_index: Sequence[Any] | pd.Index = (),
    encoding: str = "utf-8-sig",
    skip_data_rows: int = 0,
) -> pd.DataFrame:
    """Materialize exactly one bounded label slice and requested columns.

    ``nrows`` is derived from ``expected_index``; ``skip_data_rows`` is an
    explicit data-row offset (the header is always retained), and ``usecols``
    excludes every unrequested actual column at the CSV parser boundary.  This
    lets validation targets be loaded as a separate group-3-only slice while
    proving that later group-1/2 values never entered a pandas object.
    """

    requested_columns = tuple(str(column) for column in columns)
    if not requested_columns or len(set(requested_columns)) != len(
        requested_columns
    ):
        raise ValueError("bounded actual columns must be unique and non-empty")
    unknown = set(requested_columns).difference(TRANSFER_COLUMNS)
    if unknown:
        raise ValueError(f"unknown bounded actual columns: {sorted(unknown)}")
    expected = _as_datetime_index(expected_index, name=f"{purpose} expected")
    if not expected.is_monotonic_increasing:
        raise ValueError(f"{purpose}: expected timestamps must be sorted")
    if not isinstance(skip_data_rows, int) or skip_data_rows < 0:
        raise ValueError("skip_data_rows must be a non-negative integer")
    forbidden = _as_datetime_index(
        forbidden_g12_index, name=f"{purpose} forbidden preflight"
    )
    if set(requested_columns).intersection((GROUP1, GROUP2)) and len(
        expected.intersection(forbidden)
    ):
        raise ValueError(
            f"{purpose}: bounded CSV request overlaps forbidden g1/g2 timestamps"
        )
    frame = pd.read_csv(
        Path(path),
        usecols=["kst_dtm", *requested_columns],
        nrows=len(expected),
        skiprows=(range(1, skip_data_rows + 1) if skip_data_rows else None),
        encoding=encoding,
        parse_dates=["kst_dtm"],
    ).set_index("kst_dtm")
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if not frame.index.equals(expected):
        raise ValueError(
            f"{purpose}: bounded label index differs from the declared prefix"
        )
    if tuple(frame.columns) != requested_columns:
        raise ValueError(
            f"{purpose}: bounded columns {tuple(frame.columns)!r} differ from "
            f"{requested_columns!r}"
        )
    ledger.record(
        frame,
        purpose=purpose,
        forbidden_g12_index=forbidden_g12_index,
    )
    ledger.events[-1]["csv_reader_contract"] = {
        "usecols": ["kst_dtm", *requested_columns],
        "nrows": int(len(expected)),
        "skip_data_rows": int(skip_data_rows),
    }
    values = frame.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError(f"{purpose}: bounded actuals contain infinite values")
    return frame


def _validate_strict_actual_frame(
    frame: pd.DataFrame,
    *,
    expected_columns: Sequence[str],
    name: str,
) -> None:
    if tuple(frame.columns) != tuple(expected_columns):
        raise ValueError(
            f"{name} columns={tuple(frame.columns)!r}, expected="
            f"{tuple(expected_columns)!r}"
        )
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f"{name} must use a DatetimeIndex")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{name} timestamps must be unique and sorted")


def make_pseudo_labels_strict(
    mapper_training_kwh: pd.DataFrame,
    pseudo_inputs_kwh: pd.DataFrame,
    *,
    forbidden_validation_index: Sequence[Any] | pd.Index = (),
    config: MapperConfig | None = None,
    ledger: ActualAccessLedger | None = None,
    audit_prefix: str = "strict_mapper",
) -> PseudoLabelBundle:
    """Fit transfer using only physically bounded allowed actual frames.

    Unlike :func:`make_pseudo_labels`, this API cannot receive a full label
    table.  Mapper training must contain exactly group 1/2/3 at its allowed
    timestamps and pseudo application must contain exactly group 1/2 at its
    allowed historical timestamps.  Any forbidden overlap is rejected before
    the values are converted to NumPy.
    """

    cfg = config or MapperConfig()
    access = ledger or ActualAccessLedger()
    _validate_strict_actual_frame(
        mapper_training_kwh,
        expected_columns=TRANSFER_COLUMNS,
        name="strict mapper training",
    )
    _validate_strict_actual_frame(
        pseudo_inputs_kwh,
        expected_columns=(GROUP1, GROUP2),
        name="strict pseudo inputs",
    )
    if len(mapper_training_kwh.index.intersection(pseudo_inputs_kwh.index)):
        raise ValueError("mapper training and pseudo application periods overlap")
    access.record(
        mapper_training_kwh,
        purpose=f"{audit_prefix}:mapper_fit",
        forbidden_g12_index=forbidden_validation_index,
    )
    access.record(
        pseudo_inputs_kwh,
        purpose=f"{audit_prefix}:pseudo_application",
        forbidden_g12_index=forbidden_validation_index,
    )

    mapper_frame = pd.DataFrame(
        {
            GROUP1: mapper_training_kwh[GROUP1] / CAPACITY_KWH[GROUP1],
            GROUP2: mapper_training_kwh[GROUP2] / CAPACITY_KWH[GROUP2],
            GROUP3: mapper_training_kwh[GROUP3] / CAPACITY_KWH[GROUP3],
        },
        index=mapper_training_kwh.index,
    ).dropna()
    if np.isinf(mapper_frame.to_numpy(dtype=float)).any():
        raise ValueError("strict mapper training contains infinite values")
    if len(mapper_frame) < cfg.min_samples_leaf * 2:
        raise ValueError("too few simultaneous finite actuals for transfer mapper")
    mapper = HistGradientBoostingRegressor(
        loss=cfg.loss,
        learning_rate=cfg.learning_rate,
        max_iter=cfg.max_iter,
        max_leaf_nodes=cfg.max_leaf_nodes,
        min_samples_leaf=cfg.min_samples_leaf,
        l2_regularization=cfg.l2_regularization,
        random_state=cfg.random_state,
    )
    mapper.fit(mapper_frame[[GROUP1, GROUP2]], mapper_frame[GROUP3])

    application = pd.DataFrame(
        {
            GROUP1: pseudo_inputs_kwh[GROUP1] / CAPACITY_KWH[GROUP1],
            GROUP2: pseudo_inputs_kwh[GROUP2] / CAPACITY_KWH[GROUP2],
        },
        index=pseudo_inputs_kwh.index,
    )
    finite_application = application.notna().all(axis=1)
    if np.isinf(application.to_numpy(dtype=float)).any():
        raise ValueError("strict pseudo inputs contain infinite values")
    pseudo = pd.Series(
        np.nan,
        index=application.index,
        dtype=float,
        name="g3_pseudo_cf",
    )
    pseudo.loc[finite_application] = np.clip(
        mapper.predict(application.loc[finite_application]),
        cfg.pseudo_clip_lower,
        cfg.pseudo_clip_upper,
    )
    forbidden = _as_datetime_index(
        forbidden_validation_index, name="strict forbidden validation"
    )
    audit = {
        "mapper_config": asdict(cfg),
        "mapper_train_period": [
            mapper_training_kwh.index.min().isoformat(),
            mapper_training_kwh.index.max().isoformat(),
        ],
        "mapper_requested_rows": int(len(mapper_training_kwh)),
        "mapper_simultaneous_finite_rows": int(len(mapper_frame)),
        "pseudo_application_period": [
            pseudo_inputs_kwh.index.min().isoformat(),
            pseudo_inputs_kwh.index.max().isoformat(),
        ],
        "pseudo_requested_rows": int(len(pseudo_inputs_kwh)),
        "pseudo_finite_input_rows": int(finite_application.sum()),
        "pseudo_finite_output_rows": int(pseudo.notna().sum()),
        "forbidden_validation_rows": int(len(forbidden)),
        "validation_or_test_g1g2_values_accessed": False,
        "allowed_g1g2_actual_uses": [
            "mapper fit at mapper_train_period",
            "mapper application at pseudo_application_period",
        ],
        "access_ledger": access.as_dict(),
    }
    return PseudoLabelBundle(pseudo_cf=pseudo, mapper=mapper, audit=audit)


def make_pseudo_labels(
    labels_kwh: pd.DataFrame,
    *,
    mapper_train_index: Sequence[Any] | pd.Index,
    pseudo_index: Sequence[Any] | pd.Index,
    forbidden_validation_index: Sequence[Any] | pd.Index = (),
    config: MapperConfig | None = None,
) -> PseudoLabelBundle:
    """Fit the fixed CF mapper and predict pseudo CF only at allowed timestamps."""

    _validate_labels(labels_kwh)
    train_index = _as_datetime_index(mapper_train_index, name="mapper train")
    application_index = _as_datetime_index(pseudo_index, name="pseudo application")
    forbidden_index = _as_datetime_index(
        forbidden_validation_index, name="forbidden validation"
    )
    full_index = labels_kwh.index
    for name, index in (("mapper train", train_index), ("pseudo application", application_index)):
        missing = index.difference(full_index)
        if len(missing):
            raise ValueError(f"{name} timestamps are absent from labels: {missing[:3].tolist()}")
    if len(train_index.intersection(application_index)):
        raise ValueError("mapper training and pseudo application periods overlap")
    if len(train_index.intersection(forbidden_index)) or len(
        application_index.intersection(forbidden_index)
    ):
        raise ValueError("validation/test actual timestamps entered pseudo generation")

    # Slice before numeric validation. The legacy convenience API may receive a
    # larger table, but forbidden rows are never converted, divided, or scanned.
    mapper_training = labels_kwh.loc[
        train_index, list(TRANSFER_COLUMNS)
    ].copy()
    pseudo_inputs = labels_kwh.loc[
        application_index, [GROUP1, GROUP2]
    ].copy()
    return make_pseudo_labels_strict(
        mapper_training,
        pseudo_inputs,
        forbidden_validation_index=forbidden_index,
        config=config,
        ledger=ActualAccessLedger(),
        audit_prefix="legacy_bounded",
    )


def build_weather_training(
    weather: pd.DataFrame,
    actual_g3_kwh: pd.Series,
    pseudo_cf: pd.Series,
    *,
    actual_index: Sequence[Any] | pd.Index,
    pseudo_index: Sequence[Any] | pd.Index,
    eligible_fraction: float = 0.10,
) -> WeatherTrainingBundle:
    """Return exact actual-then-pseudo rows without sorting or resetting them."""

    if not isinstance(weather.index, pd.DatetimeIndex) or not weather.index.is_unique:
        raise ValueError("weather must use a unique DatetimeIndex")
    bad = [column for column in weather if FORBIDDEN_WEATHER.search(str(column))]
    if bad:
        raise ValueError(f"target/SCADA-like weather features are forbidden: {bad}")
    if not np.isfinite(weather.to_numpy()).all():
        raise ValueError("weather contains non-finite values")
    actual_requested = _as_datetime_index(actual_index, name="weather actual")
    pseudo_requested = _as_datetime_index(pseudo_index, name="weather pseudo")
    if len(actual_requested.intersection(pseudo_requested)):
        raise ValueError("actual and pseudo weather periods overlap")
    for name, index in (("actual", actual_requested), ("pseudo", pseudo_requested)):
        missing = index.difference(weather.index)
        if len(missing):
            raise ValueError(f"{name} timestamps are absent from weather")
    if not actual_g3_kwh.index.equals(weather.index):
        raise ValueError("actual group-3 labels and weather are not exactly aligned")
    missing_pseudo = pseudo_requested.difference(pseudo_cf.index)
    if len(missing_pseudo):
        raise ValueError("pseudo labels do not cover the requested pseudo period")

    actual_cf = actual_g3_kwh / CAPACITY_KWH[GROUP3]
    actual_values = actual_cf.loc[actual_requested]
    pseudo_values = pseudo_cf.loc[pseudo_requested]
    actual_keep = actual_values.notna() & np.isfinite(actual_values) & (
        actual_values >= eligible_fraction
    )
    pseudo_keep = pseudo_values.notna() & np.isfinite(pseudo_values) & (
        pseudo_values >= eligible_fraction
    )
    retained_actual = actual_requested[actual_keep.to_numpy()]
    retained_pseudo = pseudo_requested[pseudo_keep.to_numpy()]
    # This intentionally crosses the year boundary backwards. LightGBM row
    # subsampling is order-sensitive, so sorting would change the locked screen.
    ordered_index = retained_actual.append(retained_pseudo)
    target = np.r_[
        actual_values.loc[retained_actual].to_numpy(dtype=float),
        pseudo_values.loc[retained_pseudo].to_numpy(dtype=float),
    ]
    features = weather.loc[ordered_index]
    if len(features) != len(target) or not features.index.equals(ordered_index):
        raise AssertionError("actual/pseudo feature-target ordering changed")
    return WeatherTrainingBundle(
        features=features,
        target_cf=target,
        sample_weight=np.ones(len(target), dtype=float),
        ordered_index=ordered_index,
        actual_index=retained_actual,
        pseudo_index=retained_pseudo,
    )


def build_weather_training_strict(
    weather: pd.DataFrame,
    actual_g3_kwh: pd.Series,
    pseudo_cf: pd.Series,
    *,
    actual_index: Sequence[Any] | pd.Index,
    pseudo_index: Sequence[Any] | pd.Index,
    eligible_fraction: float = 0.10,
) -> WeatherTrainingBundle:
    """Build weather training without accepting any non-training g3 actuals.

    ``actual_g3_kwh`` must be indexed by exactly ``actual_index``.  Validation
    g3 actuals therefore remain in a separate scoring object and cannot be
    inspected while the weather model is fitted.  Feature row order remains
    the historical locked order: eligible actual rows followed by eligible
    pseudo rows, with no chronological sort across the boundary.
    """

    if not isinstance(weather.index, pd.DatetimeIndex) or not weather.index.is_unique:
        raise ValueError("weather must use a unique DatetimeIndex")
    bad = [column for column in weather if FORBIDDEN_WEATHER.search(str(column))]
    if bad:
        raise ValueError(f"target/SCADA-like weather features are forbidden: {bad}")
    if not np.isfinite(weather.to_numpy()).all():
        raise ValueError("weather contains non-finite values")
    actual_requested = _as_datetime_index(actual_index, name="strict weather actual")
    pseudo_requested = _as_datetime_index(pseudo_index, name="strict weather pseudo")
    if len(actual_requested.intersection(pseudo_requested)):
        raise ValueError("actual and pseudo weather periods overlap")
    if not isinstance(actual_g3_kwh.index, pd.DatetimeIndex):
        raise TypeError("strict actual group-3 labels must use a DatetimeIndex")
    if not actual_g3_kwh.index.equals(actual_requested):
        raise ValueError(
            "strict actual group-3 labels must contain exactly training timestamps"
        )
    if not isinstance(pseudo_cf.index, pd.DatetimeIndex):
        raise TypeError("strict pseudo labels must use a DatetimeIndex")
    if not pseudo_cf.index.equals(pseudo_requested):
        raise ValueError("strict pseudo labels must contain exactly pseudo timestamps")
    for name, index in (("actual", actual_requested), ("pseudo", pseudo_requested)):
        missing = index.difference(weather.index)
        if len(missing):
            raise ValueError(f"{name} timestamps are absent from weather")

    actual_values = actual_g3_kwh.astype(float) / CAPACITY_KWH[GROUP3]
    pseudo_values = pseudo_cf.astype(float)
    actual_keep = actual_values.notna() & np.isfinite(actual_values) & (
        actual_values >= eligible_fraction
    )
    pseudo_keep = pseudo_values.notna() & np.isfinite(pseudo_values) & (
        pseudo_values >= eligible_fraction
    )
    retained_actual = actual_requested[actual_keep.to_numpy()]
    retained_pseudo = pseudo_requested[pseudo_keep.to_numpy()]
    ordered_index = retained_actual.append(retained_pseudo)
    target = np.r_[
        actual_values.loc[retained_actual].to_numpy(dtype=float),
        pseudo_values.loc[retained_pseudo].to_numpy(dtype=float),
    ]
    features = weather.loc[ordered_index]
    if len(features) != len(target) or not features.index.equals(ordered_index):
        raise AssertionError("actual/pseudo feature-target ordering changed")
    return WeatherTrainingBundle(
        features=features,
        target_cf=target,
        sample_weight=np.ones(len(target), dtype=float),
        ordered_index=ordered_index,
        actual_index=retained_actual,
        pseudo_index=retained_pseudo,
    )


def make_weather_model(
    objective: str,
    config: WeatherModelConfig | None = None,
) -> LGBMRegressor:
    cfg = config or WeatherModelConfig()
    if objective == "l1":
        objective_args: dict[str, Any] = {"objective": "regression_l1"}
    elif objective == "q07":
        objective_args = {"objective": "quantile", "alpha": 0.70}
    else:
        raise ValueError("objective must be 'l1' or 'q07'")
    return LGBMRegressor(
        **objective_args,
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        min_child_samples=cfg.min_child_samples,
        subsample=cfg.subsample,
        subsample_freq=cfg.subsample_freq,
        colsample_bytree=cfg.colsample_bytree,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


__all__ = (
    "ActualAccessLedger",
    "GROUP1",
    "GROUP2",
    "GROUP3",
    "MapperConfig",
    "PseudoLabelBundle",
    "WeatherModelConfig",
    "WeatherTrainingBundle",
    "build_weather_training",
    "build_weather_training_strict",
    "make_pseudo_labels",
    "make_pseudo_labels_strict",
    "make_weather_model",
    "read_bounded_actuals",
)

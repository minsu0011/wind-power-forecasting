"""Leakage-safe probabilistic FICR decisions on existing BARAM OOF predictions.

This is a post-gate development experiment, not a new untouched gate.  It keeps
four validation tracks separate:

* 2023 H1 -> H2 and H2 -> H1 for groups 1/2 (base OOF was trained on 2022);
* 2023 Q3 -> Q4 and Q4 -> Q3 for group 3, because no 2022 group-3 labels exist;
* 2024 H1 -> H2 and H2 -> H1 for all groups; and
* a recipe fitted on 2023 OOF and transferred unchanged to 2024 OOF.

No method is scored on the rows used to fit its conditional residual/utility
model.  The final 2025 candidate uses a method selected only by the 2023
bidirectional stability checks, refitted on 2024 OOF, and applied to 2025 base
predictions without test labels.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.manifest import make_manifest, sha256_file, write_json_atomic  # noqa: E402
from src.metric import (  # noqa: E402
    CAPACITY_KWH,
    TARGET_COLS,
    group_metrics,
    score_details,
)
from src.probabilistic import (  # noqa: E402
    ProbabilisticDecisionConfig,
    ProbabilisticFICRDecision,
)
from src.temporal import smooth_within_runs  # noqa: E402


METHODS = ProbabilisticFICRDecision.METHODS
COMMON_CANDIDATES = (
    "lgb_l1",
    "lgb_q07",
    "shared_l1",
    "shared_q07",
    "top200_q07",
    "energy_q06",
)
SMOOTHING_STRENGTHS = {
    "kpx_group_1": 0.15,
    "kpx_group_2": 0.05,
    "kpx_group_3": 0.0,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(r"data/local/open"),
    )
    parser.add_argument("--oof-dir", type=Path, default=Path("artifacts/oof"))
    parser.add_argument("--gate-dir", type=Path, default=Path("artifacts/gate/v3"))
    parser.add_argument("--final-dir", type=Path, default=Path("artifacts/final_v3"))
    parser.add_argument(
        "--corrected-dir", type=Path, default=Path("artifacts/final_cf_fix")
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/postgate/probabilistic_ficr"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate source schemas and print split sizes without fitting",
    )
    return parser.parse_args(argv)


def _year_index(year: int) -> pd.DatetimeIndex:
    return pd.date_range(
        f"{year}-01-01 01:00:00",
        f"{year + 1}-01-01 00:00:00",
        freq="h",
        name="forecast_kst_dtm",
    )


def _interval(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="h", name="forecast_kst_dtm")


def _read_labels(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "train" / "train_labels.csv"
    labels = pd.read_csv(path, encoding="utf-8-sig")
    if tuple(labels.columns) != ("kst_dtm", *TARGET_COLS):
        raise ValueError(f"unexpected label columns: {tuple(labels.columns)!r}")
    labels.index = pd.DatetimeIndex(
        pd.to_datetime(labels.pop("kst_dtm"), errors="raise"),
        name="forecast_kst_dtm",
    )
    if not labels.index.is_unique or not labels.index.is_monotonic_increasing:
        raise ValueError("label timestamps must be unique and sorted")
    return labels.astype(float)


def _read_prediction(
    path: Path,
    index: pd.DatetimeIndex,
    *,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if len(frame) != len(index):
        raise ValueError(f"{path} rows={len(frame)}, expected={len(index)}")
    if columns is not None and tuple(frame.columns) != tuple(columns):
        raise ValueError(
            f"{path} columns={tuple(frame.columns)!r}, expected={tuple(columns)!r}"
        )
    frame = frame.copy()
    frame.index = index
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite predictions")
    return frame.astype(float)


def _dev_2023_sources(
    oof_dir: Path,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, list[Path]]:
    index = _year_index(2023)
    paths = {
        "lgb_l1": oof_dir / "dev2023_lgb_l1_eligible_n1500.parquet",
        "lgb_q07": oof_dir / "dev2023_lgb_q07_eligible.parquet",
        "shared_l1": oof_dir / "dev2023_shared_l1_eligible.parquet",
        "shared_q07": oof_dir / "dev2023_shared_q07_eligible.parquet",
        "top200_q07": oof_dir / "dev2023_lgb_top200_q07_eligible.parquet",
        "energy_q06": oof_dir / "dev2023_lgb_q06_energywt_eligible.parquet",
    }
    candidates = {
        name: _read_prediction(path, index, columns=TARGET_COLS[:2])
        for name, path in paths.items()
    }
    base_path = oof_dir / "dev2023_locked_v3.parquet"
    base = _read_prediction(base_path, index, columns=TARGET_COLS[:2])
    return candidates, base, [*paths.values(), base_path]


def _g3_2023_sources(
    oof_dir: Path,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, list[Path]]:
    index = _interval("2023-07-01 01:00:00", "2024-01-01 00:00:00")
    path = oof_dir / "g3dev2023h2_candidates.parquet"
    raw = _read_prediction(path, index)
    rename = {
        "l1": "lgb_l1",
        "q07": "lgb_q07",
        "shared_l1": "shared_l1",
        "shared_q07": "shared_q07",
        "top200q07": "top200_q07",
        "ewq06": "energy_q06",
    }
    candidates: dict[str, pd.DataFrame] = {}
    for source, target in rename.items():
        candidates[target] = raw[[source]].rename(columns={source: "kpx_group_3"})
    weighted = (
        0.20 * candidates["lgb_q07"]["kpx_group_3"]
        + 0.075 * candidates["shared_l1"]["kpx_group_3"]
        + 0.425 * candidates["shared_q07"]["kpx_group_3"]
        + 0.025 * candidates["top200_q07"]["kpx_group_3"]
        + 0.275 * candidates["energy_q06"]["kpx_group_3"]
    )
    base = pd.DataFrame(
        {
            "kpx_group_3": np.clip(
                1.25 * weighted.to_numpy(dtype=float) - 1200.0,
                0.0,
                1.02 * CAPACITY_KWH["kpx_group_3"],
            )
        },
        index=index,
    )
    return candidates, base, [path]


def _gate_2024_sources(
    gate_dir: Path,
    oof_dir: Path,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, list[Path]]:
    index = _year_index(2024)
    paths = {
        "lgb_l1": gate_dir / "predictions" / "lgb_l1_gate.parquet",
        "lgb_q07": gate_dir / "predictions" / "lgb_q07_gate.parquet",
        "top200_q07": gate_dir / "predictions" / "top200_q07_gate.parquet",
        "energy_q06": gate_dir / "predictions" / "energy_q06_gate.parquet",
        "shared_l1": oof_dir / "gate2024_shared_l1_cf.parquet",
        "shared_q07": oof_dir / "gate2024_shared_q07_cf.parquet",
    }
    candidates = {
        name: _read_prediction(path, index, columns=TARGET_COLS)
        for name, path in paths.items()
    }
    base_path = oof_dir / "gate2024_locked_v3_cf_fix.parquet"
    base = _read_prediction(base_path, index, columns=TARGET_COLS)
    return candidates, base, [*paths.values(), base_path]


def _test_2025_sources(
    raw_dir: Path,
    final_dir: Path,
    corrected_dir: Path,
) -> tuple[
    dict[str, pd.DataFrame],
    pd.DataFrame,
    pd.DataFrame,
    list[Path],
]:
    sample_path = raw_dir / "sample_submission.csv"
    sample = pd.read_csv(
        sample_path,
        encoding="utf-8-sig",
        dtype={"forecast_id": "string", "forecast_kst_dtm": "string"},
    )
    expected_columns = ("forecast_id", "forecast_kst_dtm", *TARGET_COLS)
    if tuple(sample.columns) != expected_columns:
        raise ValueError(f"unexpected sample columns: {tuple(sample.columns)!r}")
    index = pd.DatetimeIndex(
        pd.to_datetime(sample["forecast_kst_dtm"], errors="raise"),
        name="forecast_kst_dtm",
    )
    expected_index = _year_index(2025)
    if not index.equals(expected_index):
        raise ValueError("sample timestamps are not the exact 2025 hourly horizon")

    prefix = "v3_locked_full_2025"
    paths = {
        "lgb_l1": final_dir / "predictions" / f"{prefix}__lgb_l1_test.parquet",
        "lgb_q07": final_dir / "predictions" / f"{prefix}__lgb_q07_test.parquet",
        "top200_q07": final_dir
        / "predictions"
        / f"{prefix}__top200_q07_test.parquet",
        "energy_q06": final_dir
        / "predictions"
        / f"{prefix}__energy_q06_test.parquet",
        "shared_l1": corrected_dir
        / "predictions"
        / "shared_l1_cf_seed42_test.parquet",
        "shared_q07": corrected_dir
        / "predictions"
        / "shared_q07_cf_seed42_test.parquet",
    }
    candidates = {
        name: _read_prediction(path, index, columns=TARGET_COLS)
        for name, path in paths.items()
    }
    base_path = corrected_dir / "predictions" / "corrected_v3_test.parquet"
    base = _read_prediction(base_path, index, columns=TARGET_COLS)
    return candidates, base, sample, [sample_path, *paths.values(), base_path]


def _meta_features(
    candidates: Mapping[str, pd.DataFrame],
    base: pd.DataFrame,
    group: str,
) -> pd.DataFrame:
    if tuple(candidates) != COMMON_CANDIDATES:
        raise ValueError(
            f"candidate names/order differ: {tuple(candidates)!r}"
        )
    capacity = CAPACITY_KWH[group]
    index = base.index
    values: dict[str, np.ndarray] = {
        "base_cf": base[group].to_numpy(dtype=float) / capacity
    }
    stacked: list[np.ndarray] = []
    for name in COMMON_CANDIDATES:
        frame = candidates[name]
        if not frame.index.equals(index):
            raise ValueError(f"candidate {name!r} index differs for {group}")
        array = frame[group].to_numpy(dtype=float) / capacity
        values[f"pred__{name}_cf"] = array
        stacked.append(array)
    matrix = np.column_stack(stacked)
    values["candidate_mean_cf"] = np.mean(matrix, axis=1)
    values["candidate_std_cf"] = np.std(matrix, axis=1)
    values["candidate_min_cf"] = np.min(matrix, axis=1)
    values["candidate_max_cf"] = np.max(matrix, axis=1)
    values["q07_minus_l1_cf"] = (
        values["pred__lgb_q07_cf"] - values["pred__lgb_l1_cf"]
    )
    values["shared_q07_minus_l1_cf"] = (
        values["pred__shared_q07_cf"] - values["pred__shared_l1_cf"]
    )
    hour_angle = 2.0 * np.pi * index.hour.to_numpy() / 24.0
    day_angle = 2.0 * np.pi * (index.dayofyear.to_numpy() - 1.0) / 365.25
    values["hour_sin"] = np.sin(hour_angle)
    values["hour_cos"] = np.cos(hour_angle)
    values["day_sin"] = np.sin(day_angle)
    values["day_cos"] = np.cos(day_angle)
    result = pd.DataFrame(values, index=index)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError(f"non-finite meta features for {group}")
    return result


def _group_summary(actual: pd.Series, prediction: pd.Series, group: str) -> dict[str, Any]:
    details = group_metrics(
        actual,
        prediction,
        CAPACITY_KWH[group],
        group_name=group,
    )
    result = details.as_dict()
    result["score"] = 0.5 * (details.one_minus_nmae + details.ficr)
    return result


def _metric_summary(
    actual: pd.DataFrame,
    prediction: pd.DataFrame,
    groups: Sequence[str],
) -> dict[str, Any]:
    result = score_details(
        actual.loc[:, list(groups)],
        prediction.loc[:, list(groups)],
        target_cols=groups,
        capacities=CAPACITY_KWH,
    )
    return result.as_dict()


def _fit_apply(
    method: str,
    config: ProbabilisticDecisionConfig,
    features: pd.DataFrame,
    actual: pd.Series,
    base: pd.Series,
    fit_index: pd.DatetimeIndex,
    application_index: pd.DatetimeIndex,
    group: str,
) -> tuple[pd.Series, dict[str, Any]]:
    if len(fit_index.intersection(application_index)):
        raise AssertionError("fit/application indexes overlap")
    decision = ProbabilisticFICRDecision(method, config=config)
    decision.fit(
        features.loc[fit_index],
        actual.loc[fit_index],
        base.loc[fit_index],
        capacity_kwh=CAPACITY_KWH[group],
    )
    prediction = decision.predict(
        features.loc[application_index], base.loc[application_index]
    )
    return prediction, decision.metadata()


def _direction_record(
    *,
    period: str,
    direction: str,
    group: str,
    method: str,
    fit_index: pd.DatetimeIndex,
    application_index: pd.DatetimeIndex,
    actual: pd.Series,
    base: pd.Series,
    prediction: pd.Series,
    metadata: Mapping[str, Any],
    temporal_direction: bool,
) -> dict[str, Any]:
    baseline = _group_summary(
        actual.loc[application_index], base.loc[application_index], group
    )
    candidate = _group_summary(
        actual.loc[application_index], prediction.loc[application_index], group
    )
    return {
        "period": period,
        "direction": direction,
        "group": group,
        "method": method,
        "fit_start": str(fit_index.min()),
        "fit_end": str(fit_index.max()),
        "fit_rows": len(fit_index),
        "application_start": str(application_index.min()),
        "application_end": str(application_index.max()),
        "application_rows": len(application_index),
        "fit_application_disjoint": True,
        "temporal_direction": temporal_direction,
        "same_data_fit_score": False,
        "baseline": baseline,
        "decision": candidate,
        "delta_score": candidate["score"] - baseline["score"],
        "delta_one_minus_nmae": (
            candidate["one_minus_nmae"] - baseline["one_minus_nmae"]
        ),
        "delta_ficr": candidate["ficr"] - baseline["ficr"],
        "fit_metadata": dict(metadata),
    }


def _run_crossfit(
    *,
    period: str,
    groups: Sequence[str],
    features: Mapping[str, pd.DataFrame],
    labels: pd.DataFrame,
    base: pd.DataFrame,
    directions: Sequence[
        tuple[str, pd.DatetimeIndex, pd.DatetimeIndex, bool]
    ],
    config: ProbabilisticDecisionConfig,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    outputs = {
        method: pd.DataFrame(np.nan, index=base.index, columns=list(groups))
        for method in METHODS
    }
    records: list[dict[str, Any]] = []
    for direction, fit_index, application_index, temporal in directions:
        for group in groups:
            for method in METHODS:
                print(f"{period} {direction} {group} {method}", flush=True)
                prediction, metadata = _fit_apply(
                    method,
                    config,
                    features[group],
                    labels[group],
                    base[group],
                    fit_index,
                    application_index,
                    group,
                )
                outputs[method].loc[application_index, group] = prediction
                records.append(
                    _direction_record(
                        period=period,
                        direction=direction,
                        group=group,
                        method=method,
                        fit_index=fit_index,
                        application_index=application_index,
                        actual=labels[group],
                        base=base[group],
                        prediction=prediction,
                        metadata=metadata,
                        temporal_direction=temporal,
                    )
                )
    return outputs, records


def _selection_from_2023(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], dict[str, Any]]:
    selected: dict[str, str] = {}
    diagnostics: dict[str, Any] = {}
    complexity = {name: position for position, name in enumerate(METHODS)}
    for group in TARGET_COLS:
        stats: dict[str, Any] = {
            "identity": {"deltas": [0.0, 0.0], "minimum": 0.0, "mean": 0.0}
        }
        for method in METHODS:
            deltas = [
                float(record["delta_score"])
                for record in records
                if record["group"] == group and record["method"] == method
            ]
            if len(deltas) != 2:
                raise ValueError(
                    f"2023 selection for {group}/{method} needs two directions"
                )
            stats[method] = {
                "deltas": deltas,
                "minimum": min(deltas),
                "mean": float(np.mean(deltas)),
            }
        stable = [
            method
            for method in METHODS
            if stats[method]["minimum"] > 0.0 and stats[method]["mean"] > 0.0
        ]
        if stable:
            choice = max(
                stable,
                key=lambda method: (
                    stats[method]["minimum"],
                    stats[method]["mean"],
                    -complexity[method],
                ),
            )
            rule = "positive in both 2023 disjoint directions; maximise worst delta"
        else:
            choice = "identity"
            rule = "no method improved both 2023 directions; retain base"
        selected[group] = choice
        diagnostics[group] = {
            "selected": choice,
            "rule": rule,
            "method_statistics": stats,
            "selection_uses_2024": False,
        }
    return selected, diagnostics


def _run_transfer(
    *,
    methods: Sequence[str],
    groups: Sequence[str],
    fit_features: Mapping[str, pd.DataFrame],
    fit_labels: pd.DataFrame,
    fit_base: pd.DataFrame,
    application_features: Mapping[str, pd.DataFrame],
    application_labels: pd.DataFrame,
    application_base: pd.DataFrame,
    config: ProbabilisticDecisionConfig,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    outputs = {
        method: pd.DataFrame(index=application_base.index, columns=list(groups), dtype=float)
        for method in methods
    }
    records: list[dict[str, Any]] = []
    for group in groups:
        fit_index = fit_features[group].index
        application_index = application_features[group].index
        for method in methods:
            print(f"2023_to_2024 {group} {method}", flush=True)
            if len(fit_index.intersection(application_index)):
                raise AssertionError("transfer fit/application indexes overlap")
            decision = ProbabilisticFICRDecision(method, config=config).fit(
                fit_features[group],
                fit_labels[group].loc[fit_index],
                fit_base[group].loc[fit_index],
                capacity_kwh=CAPACITY_KWH[group],
            )
            prediction = decision.predict(
                application_features[group], application_base[group]
            )
            metadata = decision.metadata()
            outputs[method][group] = prediction
            records.append(
                _direction_record(
                    period="2023_to_2024_transfer",
                    direction="past_to_future",
                    group=group,
                    method=method,
                    fit_index=fit_index,
                    application_index=application_index,
                    actual=application_labels[group],
                    base=application_base[group],
                    prediction=prediction,
                    metadata=metadata,
                    temporal_direction=True,
                )
            )
    return outputs, records


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        frame.to_parquet(temporary, index=True)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_joblib(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        joblib.dump(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_submission(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8-sig", float_format="%.6f")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _verify_submission(path: Path, expected: pd.DataFrame) -> dict[str, Any]:
    if path.read_bytes()[:3] != b"\xef\xbb\xbf":
        raise AssertionError(f"{path} is not UTF-8-SIG")
    observed = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"forecast_id": "string", "forecast_kst_dtm": "string"},
    )
    if tuple(observed.columns) != tuple(expected.columns) or len(observed) != 8760:
        raise AssertionError(f"{path} schema/rows changed on readback")
    id_columns = ["forecast_id", "forecast_kst_dtm"]
    if not observed.loc[:, id_columns].equals(
        expected.loc[:, id_columns].astype(
            {"forecast_id": "string", "forecast_kst_dtm": "string"}
        )
    ):
        raise AssertionError(f"{path} identifiers changed on readback")
    numeric = observed.loc[:, list(TARGET_COLS)].to_numpy(dtype=float)
    expected_numeric = expected.loc[:, list(TARGET_COLS)].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise AssertionError(f"{path} contains non-finite predictions")
    for position, group in enumerate(TARGET_COLS):
        if np.any(numeric[:, position] < 0.0) or np.any(
            numeric[:, position] > 1.02 * CAPACITY_KWH[group] + 5.1e-7
        ):
            raise AssertionError(f"{path} predictions exceed clip bounds for {group}")
    maximum_difference = float(np.max(np.abs(numeric - expected_numeric)))
    if maximum_difference > 5.1e-7:
        raise AssertionError(f"{path} CSV rounding mismatch: {maximum_difference}")
    return {
        "rows": len(observed),
        "utf8_sig": True,
        "max_readback_abs_diff": maximum_difference,
        "sha256": sha256_file(path),
    }


def _assert_outputs_available(paths: Sequence[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "outputs already exist; pass --overwrite to replace this exact experiment: "
            + ", ".join(str(path) for path in existing[:3])
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    labels = _read_labels(args.raw_dir)
    labels_2023 = labels.loc[_year_index(2023)]
    labels_2024 = labels.loc[_year_index(2024)]
    candidates_2023, base_2023, inputs_2023 = _dev_2023_sources(args.oof_dir)
    candidates_g3, base_g3, inputs_g3 = _g3_2023_sources(args.oof_dir)
    candidates_2024, base_2024, inputs_2024 = _gate_2024_sources(
        args.gate_dir, args.oof_dir
    )
    candidates_2025, base_2025, sample, inputs_2025 = _test_2025_sources(
        args.raw_dir, args.final_dir, args.corrected_dir
    )

    # Common candidate order is part of the transfer contract.
    candidates_2023 = {name: candidates_2023[name] for name in COMMON_CANDIDATES}
    candidates_g3 = {name: candidates_g3[name] for name in COMMON_CANDIDATES}
    candidates_2024 = {name: candidates_2024[name] for name in COMMON_CANDIDATES}
    candidates_2025 = {name: candidates_2025[name] for name in COMMON_CANDIDATES}

    features_2023 = {
        group: _meta_features(candidates_2023, base_2023, group)
        for group in TARGET_COLS[:2]
    }
    features_2023["kpx_group_3"] = _meta_features(
        candidates_g3, base_g3, "kpx_group_3"
    )
    transfer_base_2023 = base_2023.copy()
    transfer_base_2023["kpx_group_3"] = np.nan
    transfer_base_2023 = {
        "kpx_group_1": base_2023[["kpx_group_1"]],
        "kpx_group_2": base_2023[["kpx_group_2"]],
        "kpx_group_3": base_g3[["kpx_group_3"]],
    }
    features_2024 = {
        group: _meta_features(candidates_2024, base_2024, group)
        for group in TARGET_COLS
    }
    features_2025 = {
        group: _meta_features(candidates_2025, base_2025, group)
        for group in TARGET_COLS
    }

    h1_2023 = _interval("2023-01-01 01:00:00", "2023-07-01 00:00:00")
    h2_2023 = _interval("2023-07-01 01:00:00", "2024-01-01 00:00:00")
    q3_2023 = _interval("2023-07-01 01:00:00", "2023-10-01 00:00:00")
    q4_2023 = _interval("2023-10-01 01:00:00", "2024-01-01 00:00:00")
    h1_2024 = _interval("2024-01-01 01:00:00", "2024-07-01 00:00:00")
    h2_2024 = _interval("2024-07-01 01:00:00", "2025-01-01 00:00:00")
    split_sizes = {
        "2023_h1": len(h1_2023),
        "2023_h2": len(h2_2023),
        "2023_g3_q3": len(q3_2023),
        "2023_g3_q4": len(q4_2023),
        "2024_h1": len(h1_2024),
        "2024_h2": len(h2_2024),
    }
    print(f"validated split sizes: {split_sizes}")
    if args.dry_run:
        print("dry-run complete: no labels were scored and no outputs were written")
        return 0

    config = ProbabilisticDecisionConfig(random_state=args.seed)
    output_paths: dict[str, Path] = {}
    for method in METHODS:
        output_paths[f"2023:{method}"] = (
            args.out_dir / "oof" / f"crossfit_2023__{method}.parquet"
        )
        output_paths[f"2024:{method}"] = (
            args.out_dir / "oof" / f"crossfit_2024__{method}.parquet"
        )
        output_paths[f"transfer:{method}"] = (
            args.out_dir / "oof" / f"transfer_2023_to_2024__{method}.parquet"
        )
    output_paths["selected_transfer"] = (
        args.out_dir / "oof" / "selected_transfer_2023_to_2024.parquet"
    )
    output_paths["stable_transfer"] = (
        args.out_dir / "oof" / "stable_g1g3_transfer_2023_to_2024.parquet"
    )
    output_paths["candidate_prediction"] = (
        args.out_dir / "predictions" / "selected_2025_test.parquet"
    )
    output_paths["candidate_csv"] = args.out_dir / "probabilistic_ficr_2025.csv"
    output_paths["stable_candidate_prediction"] = (
        args.out_dir / "predictions" / "prob_stable_g1g3_2025_test.parquet"
    )
    output_paths["stable_candidate_csv"] = (
        args.out_dir / "prob_stable_g1g3_2025.csv"
    )
    output_paths["smoothed_candidate_prediction"] = (
        args.out_dir / "predictions" / "prob_all_selected_smoothed_2025_test.parquet"
    )
    output_paths["smoothed_candidate_csv"] = (
        args.out_dir / "probabilistic_ficr_2025_smoothed.csv"
    )
    output_paths["models"] = args.out_dir / "models" / "selected_2024_fit.joblib"
    output_paths["results"] = args.out_dir / "results.json"
    output_paths["manifest"] = args.out_dir / "manifest.json"
    _assert_outputs_available(output_paths.values(), bool(args.overwrite))

    crossfit_2023, records_2023 = _run_crossfit(
        period="2023_h1_h2",
        groups=TARGET_COLS[:2],
        features=features_2023,
        labels=labels_2023,
        base=base_2023,
        directions=(
            ("h1_fit_h2_apply", h1_2023, h2_2023, True),
            ("h2_fit_h1_apply", h2_2023, h1_2023, False),
        ),
        config=config,
    )
    crossfit_g3, records_g3 = _run_crossfit(
        period="2023_g3_q3_q4_substitute",
        groups=("kpx_group_3",),
        features={"kpx_group_3": features_2023["kpx_group_3"]},
        labels=labels_2023.loc[h2_2023],
        base=base_g3,
        directions=(
            ("q3_fit_q4_apply", q3_2023, q4_2023, True),
            ("q4_fit_q3_apply", q4_2023, q3_2023, False),
        ),
        config=config,
    )
    for method in METHODS:
        crossfit_2023[method]["kpx_group_3"] = crossfit_g3[method][
            "kpx_group_3"
        ].reindex(crossfit_2023[method].index)
    all_2023_records = [*records_2023, *records_g3]
    selected_methods, selection = _selection_from_2023(all_2023_records)

    crossfit_2024, records_2024 = _run_crossfit(
        period="2024_h1_h2_postgate_development",
        groups=TARGET_COLS,
        features=features_2024,
        labels=labels_2024,
        base=base_2024,
        directions=(
            ("h1_fit_h2_apply", h1_2024, h2_2024, True),
            ("h2_fit_h1_apply", h2_2024, h1_2024, False),
        ),
        config=config,
    )

    fit_labels_by_group = {
        "kpx_group_1": labels_2023,
        "kpx_group_2": labels_2023,
        "kpx_group_3": labels_2023.loc[h2_2023],
    }
    transfer_outputs, transfer_records = _run_transfer(
        methods=METHODS,
        groups=TARGET_COLS,
        fit_features=features_2023,
        fit_labels=pd.concat(
            [
                labels_2023[["kpx_group_1", "kpx_group_2"]],
                labels_2023[["kpx_group_3"]],
            ],
            axis=1,
        ),
        fit_base={
            group: transfer_base_2023[group][group]
            for group in TARGET_COLS
        },
        application_features=features_2024,
        application_labels=labels_2024,
        application_base=base_2024,
        config=config,
    )

    selected_transfer = base_2024.copy()
    for group, method in selected_methods.items():
        if method != "identity":
            selected_transfer[group] = transfer_outputs[method][group]

    # A second, more conservative identity gate uses only disjoint 2024 half
    # cross-fit directions.  It does not refit method hyperparameters.  A group
    # is retained only when the 2023-selected method improves both 2024 halves.
    stable_methods = dict(selected_methods)
    stable_audit: dict[str, Any] = {}
    for group, method in selected_methods.items():
        deltas = [
            float(record["delta_score"])
            for record in records_2024
            if record["group"] == group and record["method"] == method
        ]
        keep = method != "identity" and len(deltas) == 2 and all(
            delta > 0.0 for delta in deltas
        )
        if not keep:
            stable_methods[group] = "identity"
        stable_audit[group] = {
            "method_selected_on_2023": method,
            "2024_half_deltas": deltas,
            "retained_in_stable_candidate": keep,
            "stable_method": stable_methods[group],
        }
    stable_transfer = base_2024.copy()
    for group, method in stable_methods.items():
        if method != "identity":
            stable_transfer[group] = transfer_outputs[method][group]

    # Fixed 2023-selected method is refitted on 2024 OOF for 2025 inference.
    selected_2025 = base_2025.copy()
    final_models: dict[str, Any] = {}
    final_fit_metadata: dict[str, Any] = {}
    for group, method in selected_methods.items():
        if method == "identity":
            final_fit_metadata[group] = {
                "method": "identity",
                "fit_score_calculated": False,
            }
            continue
        print(f"2024_to_2025 {group} {method}", flush=True)
        decision = ProbabilisticFICRDecision(method, config=config).fit(
            features_2024[group],
            labels_2024[group],
            base_2024[group],
            capacity_kwh=CAPACITY_KWH[group],
        )
        selected_2025[group] = decision.predict(
            features_2025[group], base_2025[group]
        )
        final_models[group] = decision
        final_fit_metadata[group] = decision.metadata()

    stable_2025 = base_2025.copy()
    for group, method in stable_methods.items():
        if method != "identity":
            stable_2025[group] = selected_2025[group]
    smoothed_2025 = smooth_within_runs(selected_2025, SMOOTHING_STRENGTHS)
    for group in TARGET_COLS:
        smoothed_2025[group] = np.clip(
            smoothed_2025[group].to_numpy(dtype=float),
            0.0,
            1.02 * CAPACITY_KWH[group],
        )

    submission = sample.copy()
    for group in TARGET_COLS:
        submission[group] = selected_2025[group].to_numpy(dtype=float)
    if len(submission) != 8760 or tuple(submission.columns) != (
        "forecast_id",
        "forecast_kst_dtm",
        *TARGET_COLS,
    ):
        raise AssertionError("final submission schema changed")
    if not np.isfinite(submission.loc[:, TARGET_COLS].to_numpy()).all():
        raise AssertionError("final submission contains non-finite predictions")
    stable_submission = sample.copy()
    for group in TARGET_COLS:
        stable_submission[group] = stable_2025[group].to_numpy(dtype=float)
    if tuple(stable_submission.columns) != tuple(submission.columns) or len(
        stable_submission
    ) != 8760:
        raise AssertionError("stable submission schema changed")
    if not np.isfinite(
        stable_submission.loc[:, list(TARGET_COLS)].to_numpy(dtype=float)
    ).all():
        raise AssertionError("stable submission contains non-finite predictions")
    smoothed_submission = sample.copy()
    for group in TARGET_COLS:
        smoothed_submission[group] = smoothed_2025[group].to_numpy(dtype=float)
    if tuple(smoothed_submission.columns) != tuple(submission.columns) or len(
        smoothed_submission
    ) != 8760:
        raise AssertionError("smoothed submission schema changed")

    summary_2023 = {
        method: {
            "groups_1_2": _metric_summary(
                labels_2023,
                crossfit_2023[method],
                TARGET_COLS[:2],
            ),
            "group_3_h2": _group_summary(
                labels_2023.loc[h2_2023, "kpx_group_3"],
                crossfit_2023[method].loc[h2_2023, "kpx_group_3"],
                "kpx_group_3",
            ),
        }
        for method in METHODS
    }
    summary_2024 = {
        method: _metric_summary(labels_2024, crossfit_2024[method], TARGET_COLS)
        for method in METHODS
    }
    summary_transfer = {
        method: _metric_summary(labels_2024, transfer_outputs[method], TARGET_COLS)
        for method in METHODS
    }
    baseline_metrics = {
        "2023_groups_1_2": _metric_summary(
            labels_2023, base_2023, TARGET_COLS[:2]
        ),
        "2023_group_3_h2": _group_summary(
            labels_2023.loc[h2_2023, "kpx_group_3"],
            base_g3["kpx_group_3"],
            "kpx_group_3",
        ),
        "2024_corrected_locked": _metric_summary(
            labels_2024, base_2024, TARGET_COLS
        ),
    }
    selected_transfer_metrics = _metric_summary(
        labels_2024, selected_transfer, TARGET_COLS
    )
    stable_transfer_metrics = _metric_summary(
        labels_2024, stable_transfer, TARGET_COLS
    )
    selected_crossfit_2024 = base_2024.copy()
    stable_crossfit_2024 = base_2024.copy()
    for group, method in selected_methods.items():
        if method != "identity":
            selected_crossfit_2024[group] = crossfit_2024[method][group]
    for group, method in stable_methods.items():
        if method != "identity":
            stable_crossfit_2024[group] = crossfit_2024[method][group]
    selected_crossfit_2024_metrics = _metric_summary(
        labels_2024, selected_crossfit_2024, TARGET_COLS
    )
    stable_crossfit_2024_metrics = _metric_summary(
        labels_2024, stable_crossfit_2024, TARGET_COLS
    )
    smoothed_transfer = smooth_within_runs(
        selected_transfer, SMOOTHING_STRENGTHS
    )
    smoothed_crossfit_2024 = smooth_within_runs(
        selected_crossfit_2024, SMOOTHING_STRENGTHS
    )

    def smoothing_period_audit(
        baseline: pd.DataFrame,
        smoothed: pd.DataFrame,
        index: pd.DatetimeIndex,
    ) -> dict[str, Any]:
        before = _metric_summary(labels_2024.loc[index], baseline.loc[index], TARGET_COLS)
        after = _metric_summary(labels_2024.loc[index], smoothed.loc[index], TARGET_COLS)
        return {
            "before": before,
            "after": after,
            "delta_score": after["total_score"] - before["total_score"],
            "group_delta_score": {
                group: 0.5
                * (
                    after["by_group"][group]["one_minus_nmae"]
                    + after["by_group"][group]["ficr"]
                    - before["by_group"][group]["one_minus_nmae"]
                    - before["by_group"][group]["ficr"]
                )
                for group in TARGET_COLS
            },
        }

    smoothing_audit = {
        "strengths": SMOOTHING_STRENGTHS,
        "transfer_2023_to_2024": {
            "full": smoothing_period_audit(
                selected_transfer, smoothed_transfer, labels_2024.index
            ),
            "h1": smoothing_period_audit(
                selected_transfer, smoothed_transfer, h1_2024
            ),
            "h2": smoothing_period_audit(
                selected_transfer, smoothed_transfer, h2_2024
            ),
        },
        "selected_2024_half_crossfit": {
            "full": smoothing_period_audit(
                selected_crossfit_2024,
                smoothed_crossfit_2024,
                labels_2024.index,
            ),
            "h1": smoothing_period_audit(
                selected_crossfit_2024, smoothed_crossfit_2024, h1_2024
            ),
            "h2": smoothing_period_audit(
                selected_crossfit_2024, smoothed_crossfit_2024, h2_2024
            ),
        },
        "conservative_adopted": False,
        "decision": (
            "attack-only: selected 2024 half-crossfit reverses on H1/group1 "
            "despite positive full transfer diagnostics"
        ),
    }

    for method, frame in crossfit_2023.items():
        _atomic_parquet(frame, output_paths[f"2023:{method}"])
    for method, frame in crossfit_2024.items():
        _atomic_parquet(frame, output_paths[f"2024:{method}"])
    for method, frame in transfer_outputs.items():
        _atomic_parquet(frame, output_paths[f"transfer:{method}"])
    _atomic_parquet(selected_transfer, output_paths["selected_transfer"])
    _atomic_parquet(stable_transfer, output_paths["stable_transfer"])
    _atomic_parquet(selected_2025, output_paths["candidate_prediction"])
    _atomic_submission(submission, output_paths["candidate_csv"])
    _atomic_parquet(stable_2025, output_paths["stable_candidate_prediction"])
    _atomic_submission(stable_submission, output_paths["stable_candidate_csv"])
    _atomic_parquet(smoothed_2025, output_paths["smoothed_candidate_prediction"])
    _atomic_submission(smoothed_submission, output_paths["smoothed_candidate_csv"])
    submission_checks = {
        "conservative_stable_g1g3": _verify_submission(
            output_paths["stable_candidate_csv"], stable_submission
        ),
        "aggressive_all_selected": _verify_submission(
            output_paths["candidate_csv"], submission
        ),
        "attack_only_smoothed": _verify_submission(
            output_paths["smoothed_candidate_csv"], smoothed_submission
        ),
    }
    _atomic_joblib(final_models, output_paths["models"])

    results = {
        "warning": (
            "2024 is consumed post-gate development data. No 2024 result in this "
            "file is an untouched or independent validation score. Same-row fit "
            "scores are never calculated."
        ),
        "methods": list(METHODS),
        "config": asdict(config),
        "split_sizes": split_sizes,
        "contracts": {
            "all_fit_application_indexes_disjoint": True,
            "same_data_fit_score_reported": False,
            "2023_group_3_limitation": (
                "No 2022 group-3 labels exist, so a leakage-safe full 2023 "
                "H1<->H2 base OOF is unavailable. Q3<->Q4 within the existing "
                "H1-trained H2 OOF is used only for stability selection."
            ),
            "reverse_half_direction": (
                "H2-fit -> H1-apply is a disjoint reverse stability diagnostic, "
                "not a causal temporal validation."
            ),
            "2024_status": "consumed gate used as development OOF",
            "method_selection_uses": "2023 bidirectional deltas only",
            "2025_fit_uses": "full 2024 OOF after method identity fixed on 2023",
        },
        "baseline_metrics": baseline_metrics,
        "crossfit_2023": {
            "direction_records": all_2023_records,
            "combined_metrics": summary_2023,
        },
        "crossfit_2024_postgate": {
            "direction_records": records_2024,
            "combined_metrics": summary_2024,
            "all_2023_selected_methods_metrics": selected_crossfit_2024_metrics,
            "stable_g1g3_metrics": stable_crossfit_2024_metrics,
            "stable_audit": stable_audit,
            "temporal_smoothing_audit": smoothing_audit[
                "selected_2024_half_crossfit"
            ],
        },
        "transfer_2023_to_2024": {
            "direction_records": transfer_records,
            "combined_metrics": summary_transfer,
            "selected_methods_from_2023": selected_methods,
            "stable_methods_after_2024_half_audit": stable_methods,
            "selection_diagnostics": selection,
            "selected_transfer_metrics": selected_transfer_metrics,
            "stable_transfer_metrics": stable_transfer_metrics,
            "temporal_smoothing_audit": smoothing_audit[
                "transfer_2023_to_2024"
            ],
        },
        "candidate_2025": {
            "score_claim": False,
            "aggressive_selected_methods": selected_methods,
            "conservative_stable_methods": stable_methods,
            "fit_metadata": final_fit_metadata,
            "conservative_priority": str(output_paths["stable_candidate_csv"]),
            "aggressive_all_selected": str(output_paths["candidate_csv"]),
            "attack_only_smoothed": str(output_paths["smoothed_candidate_csv"]),
            "temporal_smoothing": smoothing_audit,
            "csv_checks": submission_checks,
        },
    }
    write_json_atomic(
        output_paths["results"], results, overwrite=bool(args.overwrite)
    )

    input_files = list(
        dict.fromkeys(
            Path(path).resolve()
            for path in [
                args.raw_dir / "train" / "train_labels.csv",
                *inputs_2023,
                *inputs_g3,
                *inputs_2024,
                *inputs_2025,
            ]
        )
    )
    artifact_outputs = [
        path for key, path in output_paths.items() if key != "manifest"
    ]
    manifest = make_manifest(
        artifact_type="baram_probabilistic_ficr_crossfit",
        parameters={
            "methods": METHODS,
            "config": asdict(config),
            "leaderboard_score_claim": False,
            "untouched_gate_claim": False,
            "same_data_fit_score_reported": False,
        },
        input_files=input_files,
        output_files=artifact_outputs,
        results={
            "selected_methods": selected_methods,
            "stable_methods": stable_methods,
            "baseline_2024": baseline_metrics["2024_corrected_locked"],
            "selected_transfer_2024": selected_transfer_metrics,
            "stable_transfer_2024": stable_transfer_metrics,
            "candidate_2025_sha256": sha256_file(output_paths["candidate_csv"]),
            "stable_candidate_2025_sha256": sha256_file(
                output_paths["stable_candidate_csv"]
            ),
            "smoothed_candidate_2025_sha256": sha256_file(
                output_paths["smoothed_candidate_csv"]
            ),
        },
        project_dir=PROJECT_DIR,
    )
    write_json_atomic(
        output_paths["manifest"], manifest, overwrite=bool(args.overwrite)
    )
    print(f"complete: {output_paths['results']}")
    print(f"manifest: {output_paths['manifest']}")
    print(f"selected methods: {selected_methods}")
    print(f"stable methods: {stable_methods}")
    print(
        "2024 transfer score: "
        f"{selected_transfer_metrics['total_score']:.9f} "
        f"(baseline {baseline_metrics['2024_corrected_locked']['total_score']:.9f})"
    )
    print(
        f"2025 candidate: {output_paths['candidate_csv']} "
        f"sha256={sha256_file(output_paths['candidate_csv'])}"
    )
    print(
        f"2025 conservative candidate: {output_paths['stable_candidate_csv']} "
        f"sha256={sha256_file(output_paths['stable_candidate_csv'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Nested causal LightGBM HPO with strict 2023/2024 promotion gates.

All search decisions are made on inner folds contained inside each outer fit
period.  The selected recipes, fitted outer models, reload predictions, and
fixed-weight candidates are content-addressed before any outer application
label is materialized.  Public/scale artifacts are never inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
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

from scripts import run_shared_q07_multiseed as shared  # noqa: E402
from src.features import (  # noqa: E402
    WeatherFeatureBuilder,
    load_group_sites,
    read_weather_csv,
)
from src.manifest import (  # noqa: E402
    describe_file,
    git_state,
    package_versions,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from src.metric import CAPACITY_KWH, TARGET_COLS, group_metrics  # noqa: E402
from src.nested_timeseries_hpo import (  # noqa: E402
    InnerFold,
    N_TRIALS,
    blend_with_baseline,
    fit_direct_cf,
    predict_direct_cf,
    run_nested_search,
)


PREREGISTER_SHA256 = "9b469b303c792cf71f43f09f229f692d47c57f2c61a52f607841081233252ead"
YEAR_2022 = pd.date_range(
    "2022-01-01 01:00:00", "2023-01-01 00:00:00", freq="h", name="forecast_kst_dtm"
)
YEAR_2023 = pd.date_range(
    "2023-01-01 01:00:00", "2024-01-01 00:00:00", freq="h", name="forecast_kst_dtm"
)
YEAR_2023_H1 = pd.date_range(
    "2023-01-01 01:00:00", "2023-07-01 00:00:00", freq="h", name="forecast_kst_dtm"
)
YEAR_2023_H2 = pd.date_range(
    "2023-07-01 01:00:00", "2024-01-01 00:00:00", freq="h", name="forecast_kst_dtm"
)
YEAR_2024 = pd.date_range(
    "2024-01-01 01:00:00", "2025-01-01 00:00:00", freq="h", name="forecast_kst_dtm"
)
YEAR_2025 = pd.date_range(
    "2025-01-01 01:00:00", "2026-01-01 00:00:00", freq="h", name="forecast_kst_dtm"
)
PRE2024 = YEAR_2022.append(YEAR_2023)
TRAIN_ALL = PRE2024.append(YEAR_2024)
WEIGHTS: tuple[float, ...] = (0.025, 0.05, 0.10, 0.20, 1.0)
WEIGHT_KEYS: dict[float, str] = {
    0.025: "w0025",
    0.05: "w0050",
    0.10: "w0100",
    0.20: "w0200",
    1.0: "w1000",
}
STAGE1_REQUIRED: dict[str, tuple[str, ...]] = {
    "kpx_group_1": ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4"),
    "kpx_group_2": ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4"),
    "kpx_group_3": ("full", "Q3", "Q4"),
}
STAGE2_REQUIRED: tuple[str, ...] = (
    "full",
    "H1",
    "H2",
    "Q1",
    "Q2",
    "Q3",
    "Q4",
)
FULL_INPUTS: dict[str, dict[str, Any]] = {
    "train_labels": {
        "rows": 26304,
        "bytes": 1138967,
        "sha256": "47bb64252195cf4734e67394d6e50485f27a608def3b5a8791fcc7674bbceb03",
    },
    "ldaps_train": {
        "rows": 420864,
        "timestamp_rows": 26304,
        "bytes": 129687357,
        "sha256": "61ae944e7ae1fcb17391be6737792a2205c6507bf2446ed5d9d0daf07fdea026",
    },
    "gfs_train": {
        "rows": 236736,
        "timestamp_rows": 26304,
        "bytes": 84315594,
        "sha256": "cd56b67d357e7bbaff5d0d51d3537d935c9e7a3f012e9f37516bdc4d38c66a5d",
    },
    "ldaps_test": {
        "rows": 140160,
        "timestamp_rows": 8760,
        "bytes": 43122637,
        "sha256": "60e94f7cc80384eee335e90dc896b6cf4d36b35cde8d37bc03bd5c08a788b0fa",
    },
    "gfs_test": {
        "rows": 78840,
        "timestamp_rows": 8760,
        "bytes": 28037722,
        "sha256": "aa33febb24ecd46b82be34880a06910e16a3382319287548e6ce2af721b4f848",
    },
    "sample_submission": {
        "bytes": 359229,
        "sha256": "c925d2066a834f937f8091ed55acfe50ff86c8be4745b52c3adc95b056c5aaaa",
    },
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("stage1", "stage2", "final", "all"), required=True)
    parser.add_argument(
        "--raw-dir", type=Path, default=Path(r"data/local/open")
    )
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/postgate/nested_timeseries_hpo_lgb_strict_v1"),
    )
    parser.add_argument(
        "--preregister",
        type=Path,
        default=Path("configs/nested_timeseries_hpo_preregister_v1.json"),
    )
    return parser.parse_args(argv)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    write_json_atomic(path, _json_ready(payload), overwrite=False)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        frame.to_parquet(temporary, engine="pyarrow", compression="zstd", index=True)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_joblib(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        joblib.dump(payload, temporary, compress=3)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        frame.to_csv(
            temporary,
            index=False,
            encoding="utf-8-sig",
            float_format="%.6f",
            lineterminator="\n",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        _json_ready(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verify_preregister(path: Path) -> dict[str, Any]:
    if sha256_file(path) != PREREGISTER_SHA256:
        raise AssertionError("nested HPO preregistration hash changed")
    sidecar = path.with_suffix(".sha256")
    expected = f"{PREREGISTER_SHA256}  {path.name}\n"
    if sidecar.read_text(encoding="utf-8") != expected:
        raise AssertionError("nested HPO preregistration sidecar changed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["experiment_id"] != "nested_timeseries_hpo_lgb_direct_cf_strict_v1":
        raise AssertionError("experiment id changed")
    hpo = payload["inner_hpo"]
    if int(hpo["n_trials_per_group"]) != N_TRIALS:
        raise AssertionError("trial count changed")
    if tuple(map(float, payload["stage1_outer"]["candidate_weights_in_fixed_order"])) != WEIGHTS:
        raise AssertionError("blend weights changed")
    if payload["pre_score_contract"]["public_or_scale_artifacts_may_be_read"] is not False:
        raise AssertionError("forbidden artifact contract changed")
    return payload


def _assert_identity(path: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    record = describe_file(path)
    if record["size_bytes"] != int(spec["bytes"]) or record["sha256"] != spec["sha256"]:
        raise AssertionError(f"file identity changed: {path}")
    return record


def _read_label_prefix(
    path: Path, spec: Mapping[str, Any], groups: Sequence[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = int(spec["data_rows"])
    byte_limit = int(spec["bytes"])
    digest, observed_bytes = shared._csv_prefix_identity(
        path, data_rows=rows, byte_limit=byte_limit
    )
    if observed_bytes != byte_limit or digest != spec["sha256"]:
        raise AssertionError("label physical prefix identity changed")
    bounded_raw = shared._BoundedRawReader(path, byte_limit=byte_limit)
    try:
        with io.BufferedReader(bounded_raw, buffer_size=1024 * 1024) as bounded:
            frame = pd.read_csv(
                bounded,
                usecols=["kst_dtm", *groups],
                encoding="utf-8-sig",
                nrows=rows,
                memory_map=False,
            )
            bytes_returned = bounded_raw.bytes_returned
            underlying_position = bounded_raw.underlying_position
    finally:
        bounded_raw.close()
    if bytes_returned != byte_limit or underlying_position != byte_limit:
        raise AssertionError("label parser crossed or failed to consume its byte cap")
    if tuple(frame.columns) != ("kst_dtm", *groups) or len(frame) != rows:
        raise AssertionError("bounded label schema or row count changed")
    times = pd.to_datetime(frame.pop("kst_dtm"), errors="raise")
    frame.index = pd.DatetimeIndex(times, name="forecast_kst_dtm")
    frame = frame.astype(np.float64)
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise AssertionError("bounded label index changed")
    if frame.index.max() != pd.Timestamp(spec["end"]):
        raise AssertionError("bounded label end changed")
    next_timestamp = shared._next_csv_first_field(
        path, after_data_rows=rows, prefix_bytes=byte_limit
    )
    if next_timestamp != spec["next_row_first_field_only"]:
        raise AssertionError("next label boundary changed")
    evidence = {
        "path": str(path.resolve()),
        "groups_materialized": list(groups),
        "data_rows": rows,
        "physical_byte_limit": byte_limit,
        "physical_bytes_returned": bytes_returned,
        "underlying_position_after_parse": underlying_position,
        "physical_prefix_sha256": digest,
        "suffix_bytes_exposed_to_parser": 0,
        "start": frame.index.min(),
        "end": frame.index.max(),
        "next_row_timestamp_only": next_timestamp,
        "next_row_target_values_read": False,
    }
    return frame, evidence


def _inner_folds(group: str) -> tuple[InnerFold, ...]:
    if group in TARGET_COLS[:2]:
        return (
            InnerFold(pd.Timestamp("2022-01-01 01:00"), pd.Timestamp("2022-04-01 00:00"), pd.Timestamp("2022-04-01 01:00"), pd.Timestamp("2022-07-01 00:00")),
            InnerFold(pd.Timestamp("2022-01-01 01:00"), pd.Timestamp("2022-07-01 00:00"), pd.Timestamp("2022-07-01 01:00"), pd.Timestamp("2022-10-01 00:00")),
            InnerFold(pd.Timestamp("2022-01-01 01:00"), pd.Timestamp("2022-10-01 00:00"), pd.Timestamp("2022-10-01 01:00"), pd.Timestamp("2023-01-01 00:00")),
        )
    if group == "kpx_group_3":
        return (
            InnerFold(pd.Timestamp("2023-01-01 01:00"), pd.Timestamp("2023-03-01 00:00"), pd.Timestamp("2023-03-01 01:00"), pd.Timestamp("2023-04-01 00:00")),
            InnerFold(pd.Timestamp("2023-01-01 01:00"), pd.Timestamp("2023-04-01 00:00"), pd.Timestamp("2023-04-01 01:00"), pd.Timestamp("2023-05-01 00:00")),
            InnerFold(pd.Timestamp("2023-01-01 01:00"), pd.Timestamp("2023-05-01 00:00"), pd.Timestamp("2023-05-01 01:00"), pd.Timestamp("2023-06-01 00:00")),
            InnerFold(pd.Timestamp("2023-01-01 01:00"), pd.Timestamp("2023-06-01 00:00"), pd.Timestamp("2023-06-01 01:00"), pd.Timestamp("2023-07-01 00:00")),
        )
    raise KeyError(group)


def _outer_indexes(group: str) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    return (YEAR_2022, YEAR_2023) if group in TARGET_COLS[:2] else (YEAR_2023_H1, YEAR_2023_H2)


def _segments(index: pd.DatetimeIndex, year: int) -> dict[str, pd.DatetimeIndex]:
    boundaries = {
        "full": (f"{year}-01-01 01:00", f"{year + 1}-01-01 00:00"),
        "H1": (f"{year}-01-01 01:00", f"{year}-07-01 00:00"),
        "H2": (f"{year}-07-01 01:00", f"{year + 1}-01-01 00:00"),
        "Q1": (f"{year}-01-01 01:00", f"{year}-04-01 00:00"),
        "Q2": (f"{year}-04-01 01:00", f"{year}-07-01 00:00"),
        "Q3": (f"{year}-07-01 01:00", f"{year}-10-01 00:00"),
        "Q4": (f"{year}-10-01 01:00", f"{year + 1}-01-01 00:00"),
    }
    output: dict[str, pd.DatetimeIndex] = {}
    for name, (start, end) in boundaries.items():
        selected = index[(index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))]
        if len(selected):
            output[name] = selected
    return output


def _load_stage1_baseline(
    artifact_root: Path, preregister: Mapping[str, Any]
) -> tuple[dict[str, pd.Series], dict[str, Any]]:
    contract = preregister["baseline_contract"]
    g12_path = artifact_root / "oof/dev2023_locked_v3.parquet"
    g3_path = artifact_root / "oof/g3dev2023h2_candidates.parquet"
    evidence = {
        "g12": _assert_identity(g12_path, contract["stage1_group_1_and_2"]),
        "g3": _assert_identity(g3_path, contract["stage1_group_3"]),
    }
    g12 = pd.read_parquet(g12_path, engine="pyarrow")
    g12.index = pd.DatetimeIndex(g12.index, name="forecast_kst_dtm")
    if not g12.index.equals(YEAR_2023) or tuple(g12.columns) != TARGET_COLS[:2]:
        raise AssertionError("G1/G2 baseline schema changed")
    raw = pd.read_parquet(g3_path, engine="pyarrow")
    raw.index = pd.DatetimeIndex(raw.index, name="forecast_kst_dtm")
    required = ("q07", "shared_l1", "shared_q07", "top200q07", "ewq06")
    if not raw.index.equals(YEAR_2023_H2) or any(column not in raw for column in required):
        raise AssertionError("G3 baseline components changed")
    weighted = (
        0.20 * raw["q07"]
        + 0.075 * raw["shared_l1"]
        + 0.425 * raw["shared_q07"]
        + 0.025 * raw["top200q07"]
        + 0.275 * raw["ewq06"]
    )
    g3 = pd.Series(
        np.clip(1.25 * weighted.to_numpy(dtype=np.float64) - 1200.0, 0.0, 1.02 * 21000.0),
        index=YEAR_2023_H2,
        name="kpx_group_3",
    )
    baseline = {
        "kpx_group_1": g12["kpx_group_1"].astype(np.float64),
        "kpx_group_2": g12["kpx_group_2"].astype(np.float64),
        "kpx_group_3": g3,
    }
    if any(not np.isfinite(series.to_numpy()).all() for series in baseline.values()):
        raise AssertionError("baseline contains non-finite values")
    return baseline, evidence


def _score_summary(actual: pd.Series, prediction: pd.Series, group: str) -> dict[str, Any]:
    details = group_metrics(actual, prediction, CAPACITY_KWH[group], group_name=group)
    output = details.as_dict()
    output["score"] = 0.5 * (details.one_minus_nmae + details.ficr)
    return output


def _compare_candidates(
    actual: pd.Series,
    baseline: pd.Series,
    candidates: pd.DataFrame,
    group: str,
    segment_names: Sequence[str],
    year: int,
) -> dict[str, Any]:
    segments = _segments(actual.index, year)
    output: dict[str, Any] = {}
    for key in candidates.columns:
        output[key] = {}
        for segment in segment_names:
            idx = segments[segment]
            base_metrics = _score_summary(actual.loc[idx], baseline.loc[idx], group)
            candidate_metrics = _score_summary(actual.loc[idx], candidates.loc[idx, key], group)
            output[key][segment] = {
                "baseline": base_metrics,
                "candidate": candidate_metrics,
                "delta": float(candidate_metrics["score"] - base_metrics["score"]),
            }
    return output


def _select_weight(
    comparisons: Mapping[str, Any], required: Sequence[str]
) -> tuple[str, dict[str, Any]]:
    order = {WEIGHT_KEYS[weight]: position for position, weight in enumerate(WEIGHTS)}
    diagnostics: dict[str, Any] = {}
    passed: list[str] = []
    for key in order:
        deltas = [float(comparisons[key][segment]["delta"]) for segment in required]
        diagnostics[key] = {
            "minimum_delta": min(deltas),
            "mean_delta": float(np.mean(deltas)),
            "all_strictly_positive": all(value > 0.0 for value in deltas),
            "deltas": dict(zip(required, deltas)),
        }
        if diagnostics[key]["all_strictly_positive"]:
            passed.append(key)
    selected = (
        max(
            passed,
            key=lambda key: (
                diagnostics[key]["minimum_delta"],
                diagnostics[key]["mean_delta"],
                -order[key],
            ),
        )
        if passed
        else "identity"
    )
    return selected, {"selected": selected, "passing_candidates": passed, "statistics": diagnostics}


def _provenance_paths(preregister_path: Path) -> dict[str, Path]:
    paths = {
        "runner": Path(__file__).resolve(),
        "model": PROJECT_DIR / "src/nested_timeseries_hpo.py",
        "features": PROJECT_DIR / "src/features.py",
        "metric": PROJECT_DIR / "src/metric.py",
        "manifest": PROJECT_DIR / "src/manifest.py",
        "bounded_reader": PROJECT_DIR / "scripts/run_shared_q07_multiseed.py",
        "focused_test": PROJECT_DIR / "tests/test_nested_timeseries_hpo.py",
        "preregister": preregister_path.resolve(),
        "preregister_sidecar": preregister_path.with_suffix(".sha256").resolve(),
    }
    forbidden = ("public", "scale")
    for name, path in paths.items():
        rendered = path.as_posix().lower()
        if any(token in rendered for token in forbidden):
            raise AssertionError(f"forbidden provenance path: {name}")
    return paths


def _snapshot(paths: Mapping[str, Path]) -> dict[str, Any]:
    return {name: describe_file(path) for name, path in paths.items()}


def _fit_outer_group(
    *,
    group: str,
    features: pd.DataFrame,
    actual_fit: pd.Series,
    sampled: Mapping[str, Any],
    fit_index: pd.DatetimeIndex,
    apply_index: pd.DatetimeIndex,
    baseline: pd.Series,
) -> tuple[Any, pd.Series, pd.DataFrame, dict[str, Any]]:
    if fit_index.max() >= apply_index.min() or len(fit_index.intersection(apply_index)):
        raise AssertionError("outer fit/application causality changed")
    model, training = fit_direct_cf(
        features.loc[fit_index],
        actual_fit.loc[fit_index] / CAPACITY_KWH[group],
        sampled,
    )
    direct = predict_direct_cf(model, features.loc[apply_index])
    candidates = pd.DataFrame(index=apply_index)
    for weight in WEIGHTS:
        key = WEIGHT_KEYS[weight]
        candidates[key] = blend_with_baseline(
            baseline.loc[apply_index],
            direct,
            weight=weight,
            capacity_kwh=CAPACITY_KWH[group],
        )
    metadata = {
        "fit_start": fit_index.min(),
        "fit_end": fit_index.max(),
        "apply_start": apply_index.min(),
        "apply_end": apply_index.max(),
        "fit_before_apply": True,
        "training": training,
        "candidate_weights": list(WEIGHTS),
    }
    return model, direct, candidates, metadata


def _stage1(
    *,
    raw_dir: Path,
    artifact_root: Path,
    out_dir: Path,
    preregister_path: Path,
    preregister: Mapping[str, Any],
) -> dict[str, Any]:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    provenance_paths = _provenance_paths(preregister_path)
    provenance_before = _snapshot(provenance_paths)
    baseline, baseline_evidence = _load_stage1_baseline(artifact_root, preregister)
    dummy = pd.DataFrame(index=PRE2024)
    features, weather_evidence = shared._read_stage1_raw_features(raw_dir, dummy)
    if any(not features[group].index.equals(PRE2024) for group in TARGET_COLS):
        raise AssertionError("Stage1 raw feature index changed")

    label_path = raw_dir / "train/train_labels.csv"
    physical = preregister["physical_stage1_inputs"]
    labels_g12, g12_label_evidence = _read_label_prefix(
        label_path, physical["labels_g12_fit_prefix"], TARGET_COLS[:2]
    )
    if not labels_g12.index.equals(YEAR_2022):
        raise AssertionError("G1/G2 fit label prefix changed")

    best_by_group: dict[str, Any] = {}
    histories: dict[str, Any] = {}
    models: dict[str, Any] = {}
    direct_predictions: dict[str, pd.Series] = {}
    candidates_by_group: dict[str, pd.DataFrame] = {}
    training: dict[str, Any] = {}
    artifacts: list[Path] = []
    for group in TARGET_COLS[:2]:
        print(f"nested HPO {group}: {N_TRIALS} trials x 3 causal folds", flush=True)
        best, history, _ = run_nested_search(
            features=features[group].loc[YEAR_2022],
            actual_kwh=labels_g12[group],
            capacity_kwh=CAPACITY_KWH[group],
            group=group,
            folds=_inner_folds(group),
        )
        fit_index, apply_index = _outer_indexes(group)
        model, direct, candidates, metadata = _fit_outer_group(
            group=group,
            features=features[group],
            actual_fit=labels_g12[group],
            sampled=best["sampled_parameters"],
            fit_index=fit_index,
            apply_index=apply_index,
            baseline=baseline[group],
        )
        best_by_group[group] = best
        histories[group] = history
        models[group] = model
        direct_predictions[group] = direct
        candidates_by_group[group] = candidates
        training[group] = metadata
        study_path = out_dir / f"hpo/{group}_study.json"
        _write_json(study_path, {"group": group, "best": best, "trials": history})
        artifacts.append(study_path)

    g12_model_path = out_dir / "models/stage1_g12_outer_models.joblib"
    _atomic_joblib({group: models[group] for group in TARGET_COLS[:2]}, g12_model_path)
    g12_direct_path = out_dir / "predictions/stage1_g12_direct_cf.parquet"
    _atomic_parquet(
        pd.concat([direct_predictions[group].rename(group) for group in TARGET_COLS[:2]], axis=1),
        g12_direct_path,
    )
    g12_candidate_paths: list[Path] = []
    for group in TARGET_COLS[:2]:
        path = out_dir / f"predictions/stage1_{group}_candidates.parquet"
        _atomic_parquet(candidates_by_group[group], path)
        g12_candidate_paths.append(path)
    reloaded_g12 = joblib.load(g12_model_path)
    g12_reload: dict[str, Any] = {}
    for group in TARGET_COLS[:2]:
        observed = predict_direct_cf(reloaded_g12[group], features[group].loc[YEAR_2023])
        if not np.array_equal(observed.to_numpy(), direct_predictions[group].to_numpy()):
            raise AssertionError(f"{group} reloaded Stage1 prediction differs")
        g12_reload[group] = {"prediction_bit_exact": True, "rows": len(observed)}
    g12_lock_path = out_dir / "stage1_g12_prescore_lock.json"
    _write_json(
        g12_lock_path,
        {
            "preregister_sha256": PREREGISTER_SHA256,
            "fit_label_evidence": g12_label_evidence,
            "hpo_best": {group: best_by_group[group] for group in TARGET_COLS[:2]},
            "training": {group: training[group] for group in TARGET_COLS[:2]},
            "outputs": [describe_file(path) for path in [g12_model_path, g12_direct_path, *g12_candidate_paths]],
            "reload_audit": g12_reload,
            "outer_application_target_values_materialized": 0,
            "created_before_g3_fit_labels_and_all_outer_application_labels": True,
        },
    )
    artifacts.extend([g12_model_path, g12_direct_path, *g12_candidate_paths, g12_lock_path])

    labels_g3, g3_label_evidence = _read_label_prefix(
        label_path, physical["labels_g3_fit_prefix"], ("kpx_group_3",)
    )
    if not labels_g3.index.equals(YEAR_2022.append(YEAR_2023_H1)):
        raise AssertionError("G3 fit label prefix changed")
    group = "kpx_group_3"
    print(f"nested HPO {group}: {N_TRIALS} trials x 4 causal folds", flush=True)
    best, history, _ = run_nested_search(
        features=features[group].loc[YEAR_2023_H1],
        actual_kwh=labels_g3.loc[YEAR_2023_H1, group],
        capacity_kwh=CAPACITY_KWH[group],
        group=group,
        folds=_inner_folds(group),
    )
    fit_index, apply_index = _outer_indexes(group)
    model, direct, candidates, metadata = _fit_outer_group(
        group=group,
        features=features[group],
        actual_fit=labels_g3[group],
        sampled=best["sampled_parameters"],
        fit_index=fit_index,
        apply_index=apply_index,
        baseline=baseline[group],
    )
    best_by_group[group] = best
    histories[group] = history
    models[group] = model
    direct_predictions[group] = direct
    candidates_by_group[group] = candidates
    training[group] = metadata
    study_path = out_dir / f"hpo/{group}_study.json"
    _write_json(study_path, {"group": group, "best": best, "trials": history})
    g3_model_path = out_dir / "models/stage1_g3_outer_model.joblib"
    _atomic_joblib(model, g3_model_path)
    g3_direct_path = out_dir / "predictions/stage1_g3_direct_cf.parquet"
    _atomic_parquet(direct.rename(group).to_frame(), g3_direct_path)
    g3_candidate_path = out_dir / f"predictions/stage1_{group}_candidates.parquet"
    _atomic_parquet(candidates, g3_candidate_path)
    reloaded = joblib.load(g3_model_path)
    observed = predict_direct_cf(reloaded, features[group].loc[YEAR_2023_H2])
    if not np.array_equal(observed.to_numpy(), direct.to_numpy()):
        raise AssertionError("G3 reloaded Stage1 prediction differs")
    g3_lock_path = out_dir / "stage1_g3_prescore_lock.json"
    _write_json(
        g3_lock_path,
        {
            "preregister_sha256": PREREGISTER_SHA256,
            "fit_label_evidence": g3_label_evidence,
            "hpo_best": best,
            "training": metadata,
            "outputs": [describe_file(path) for path in (g3_model_path, g3_direct_path, g3_candidate_path)],
            "reload_audit": {"prediction_bit_exact": True, "rows": len(observed)},
            "outer_application_target_values_materialized": 0,
            "created_before_all_outer_application_labels": True,
        },
    )
    artifacts.extend([study_path, g3_model_path, g3_direct_path, g3_candidate_path, g3_lock_path])

    recipe_path = out_dir / "locked_inner_hpo_recipes.json"
    _write_json(
        recipe_path,
        {
            "preregister_sha256": PREREGISTER_SHA256,
            "recipes": {group: best_by_group[group] for group in TARGET_COLS},
            "recipe_sha256": _canonical_sha256({group: best_by_group[group] for group in TARGET_COLS}),
            "outer_application_target_values_materialized": 0,
            "2024_read": False,
            "2025_read": False,
        },
    )
    artifacts.append(recipe_path)
    provenance_after = _snapshot(provenance_paths)
    if provenance_before != provenance_after:
        raise AssertionError("source provenance changed during Stage1")
    global_lock_path = out_dir / "stage1_global_prescore_lock.json"
    _write_json(
        global_lock_path,
        {
            "preregister_sha256": PREREGISTER_SHA256,
            "recipe_lock": describe_file(recipe_path),
            "g12_lock": describe_file(g12_lock_path),
            "g3_lock": describe_file(g3_lock_path),
            "baseline_inputs": baseline_evidence,
            "weather_evidence": weather_evidence,
            "provenance_before": provenance_before,
            "provenance_after": provenance_after,
            "all_model_and_candidate_outputs": [describe_file(path) for path in artifacts],
            "all_three_hpo_recipes_and_outer_predictions_frozen": True,
            "outer_application_target_values_materialized": 0,
            "2024_read": False,
            "2025_read": False,
        },
    )

    score_labels, score_evidence = _read_label_prefix(
        label_path, physical["labels_stage1_score_prefix_after_lock"], TARGET_COLS
    )
    group_results: dict[str, Any] = {}
    locked_candidates: dict[str, str] = {}
    for group in TARGET_COLS:
        _, apply_index = _outer_indexes(group)
        comparisons = _compare_candidates(
            score_labels.loc[apply_index, group],
            baseline[group],
            candidates_by_group[group],
            group,
            STAGE1_REQUIRED[group],
            2023,
        )
        selected, selection = _select_weight(comparisons, STAGE1_REQUIRED[group])
        locked_candidates[group] = selected
        group_results[group] = {
            "comparisons": comparisons,
            "comparisons_sha256": _canonical_sha256(comparisons),
            "selection": selection,
            "selected_candidate": selected,
        }
    passed_groups = [group for group in TARGET_COLS if locked_candidates[group] != "identity"]
    result_path = out_dir / "stage1_results.json"
    result = {
        "experiment_id": preregister["experiment_id"],
        "preregister_sha256": PREREGISTER_SHA256,
        "global_prescore_lock": describe_file(global_lock_path),
        "score_labels_read_after_prescore_lock": score_evidence,
        "group_results": group_results,
        "locked_candidates": locked_candidates,
        "passed_groups": passed_groups,
        "2024_read": False,
        "2025_read": False,
        "leaderboard_score_claim": False,
    }
    _write_json(result_path, result)
    lock = {
        "preregister_sha256": PREREGISTER_SHA256,
        "global_prescore_lock": describe_file(global_lock_path),
        "recipe_lock": describe_file(recipe_path),
        "stage1_results": describe_file(result_path),
        "locked_candidates": locked_candidates,
        "passed_groups": passed_groups,
        "hpo_recipes": {group: best_by_group[group]["sampled_parameters"] for group in TARGET_COLS},
        "only_passing_groups_may_open_2024": True,
        "no_2024_reselection_or_retuning": True,
        "2024_read": False,
        "2025_read": False,
    }
    _write_json(out_dir / "stage1_promotion_lock.json", lock)
    print(f"Stage1 nested-HPO passed groups: {passed_groups}", flush=True)
    return lock


_FULL_TRAIN_FEATURE_CACHE: tuple[dict[str, pd.DataFrame], dict[str, Any]] | None = None
_TEST_FEATURE_CACHE: tuple[dict[str, pd.DataFrame], dict[str, Any]] | None = None


def _build_raw_features(
    raw_dir: Path, *, period: str
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    global _FULL_TRAIN_FEATURE_CACHE, _TEST_FEATURE_CACHE
    if period == "train" and _FULL_TRAIN_FEATURE_CACHE is not None:
        return _FULL_TRAIN_FEATURE_CACHE
    if period == "test" and _TEST_FEATURE_CACHE is not None:
        return _TEST_FEATURE_CACHE
    if period == "train":
        paths = {
            "ldaps": raw_dir / "train/ldaps_train.csv",
            "gfs": raw_dir / "train/gfs_train.csv",
        }
        specs = {"ldaps": FULL_INPUTS["ldaps_train"], "gfs": FULL_INPUTS["gfs_train"]}
        expected_index = TRAIN_ALL
    elif period == "test":
        paths = {
            "ldaps": raw_dir / "test/ldaps_test.csv",
            "gfs": raw_dir / "test/gfs_test.csv",
        }
        specs = {"ldaps": FULL_INPUTS["ldaps_test"], "gfs": FULL_INPUTS["gfs_test"]}
        expected_index = YEAR_2025
    else:
        raise ValueError(period)
    identities = {source: _assert_identity(paths[source], specs[source]) for source in paths}
    raw = {source: read_weather_csv(paths[source], source) for source in paths}
    sites = load_group_sites(raw_dir / "info.xlsx")
    builder = WeatherFeatureBuilder(group_sites=sites)
    features = builder.fit_transform(raw["ldaps"], raw["gfs"])
    schema: tuple[str, ...] | None = None
    feature_evidence: dict[str, Any] = {}
    for group in TARGET_COLS:
        frame = features[group]
        frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
        if not frame.index.equals(expected_index):
            raise AssertionError(f"{period}/{group} raw feature index changed")
        if schema is not None and tuple(frame.columns) != schema:
            raise AssertionError("group raw feature schemas differ")
        schema = tuple(map(str, frame.columns))
        if frame.shape[1] != 612 or not np.isfinite(frame.to_numpy(dtype=np.float32)).all():
            raise AssertionError("raw feature shape or finiteness changed")
        features[group] = frame.astype(np.float32, copy=False)
        feature_evidence[group] = {
            "rows": len(frame),
            "columns": frame.shape[1],
            "start": frame.index.min(),
            "end": frame.index.max(),
            "frame_sha256": shared._frame_sha256(features[group]),
        }
    contract = {
        "period": period,
        "raw_identities": identities,
        "feature_schema_sha256": hashlib.sha256(
            "\n".join(schema or ()).encode("utf-8")
        ).hexdigest(),
        "features": feature_evidence,
    }
    result = (features, contract)
    if period == "train":
        _FULL_TRAIN_FEATURE_CACHE = result
    else:
        _TEST_FEATURE_CACHE = result
    return result


def _read_full_labels(raw_dir: Path, groups: Sequence[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = raw_dir / "train/train_labels.csv"
    identity = _assert_identity(path, FULL_INPUTS["train_labels"])
    frame = pd.read_csv(path, usecols=["kst_dtm", *groups], encoding="utf-8-sig")
    if tuple(frame.columns) != ("kst_dtm", *groups) or len(frame) != len(TRAIN_ALL):
        raise AssertionError("full label schema changed")
    index = pd.to_datetime(frame.pop("kst_dtm"), errors="raise")
    frame.index = pd.DatetimeIndex(index, name="forecast_kst_dtm")
    frame = frame.astype(np.float64)
    if not frame.index.equals(TRAIN_ALL) or np.isinf(frame.to_numpy()).any():
        raise AssertionError("full label index or values changed")
    return frame, {"file": identity, "groups_materialized": list(groups), "rows": len(frame)}


def _load_stage1_lock(out_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    lock_path = out_dir / "stage1_promotion_lock.json"
    result_path = out_dir / "stage1_results.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if lock["preregister_sha256"] != PREREGISTER_SHA256:
        raise AssertionError("Stage1 lock preregistration changed")
    if lock["stage1_results"]["sha256"] != sha256_file(result_path):
        raise AssertionError("Stage1 result identity changed")
    recomputed: dict[str, str] = {}
    for group in TARGET_COLS:
        selected, selection = _select_weight(
            result["group_results"][group]["comparisons"], STAGE1_REQUIRED[group]
        )
        if selection != result["group_results"][group]["selection"]:
            raise AssertionError("Stage1 selection cannot be reproduced")
        recomputed[group] = selected
    if recomputed != lock["locked_candidates"]:
        raise AssertionError("Stage1 locked candidates changed")
    passed = [group for group in TARGET_COLS if recomputed[group] != "identity"]
    if passed != lock["passed_groups"]:
        raise AssertionError("Stage1 passed group list changed")
    return lock, result


def _selected_weight(key: str) -> float:
    reverse = {value: key_weight for key_weight, value in WEIGHT_KEYS.items()}
    if key not in reverse:
        raise ValueError(f"unregistered candidate key: {key}")
    return reverse[key]


def _stage2(
    *,
    raw_dir: Path,
    artifact_root: Path,
    out_dir: Path,
    preregister: Mapping[str, Any],
) -> dict[str, Any]:
    stage1_lock, _ = _load_stage1_lock(out_dir)
    passed_groups = list(stage1_lock["passed_groups"])
    result_path = out_dir / "stage2_results.json"
    lock_path = out_dir / "stage2_promotion_lock.json"
    if result_path.exists() or lock_path.exists():
        raise FileExistsError("Stage2 output already exists")
    if not passed_groups:
        result = {
            "preregister_sha256": PREREGISTER_SHA256,
            "performed": False,
            "reason": "no Stage1 group passed every registered slice",
            "stage1_passed_groups": [],
            "stage2_passed_groups": [],
            "2024_weather_read": False,
            "2024_label_read": False,
            "2025_read": False,
        }
        _write_json(result_path, result)
        lock = {
            "preregister_sha256": PREREGISTER_SHA256,
            "stage1_promotion_lock": describe_file(out_dir / "stage1_promotion_lock.json"),
            "stage2_results": describe_file(result_path),
            "stage2_passed_groups": [],
            "2024_read": False,
            "2025_read": False,
        }
        _write_json(lock_path, lock)
        return lock

    features, feature_evidence = _build_raw_features(raw_dir, period="train")
    bounded_features, _ = shared._read_stage1_raw_features(raw_dir, pd.DataFrame(index=PRE2024))
    prefix_audit: dict[str, Any] = {}
    for group in TARGET_COLS:
        full_prefix = features[group].loc[PRE2024]
        exact = np.array_equal(
            full_prefix.to_numpy(), bounded_features[group].to_numpy(), equal_nan=True
        )
        if not exact or tuple(full_prefix.columns) != tuple(bounded_features[group].columns):
            raise AssertionError("bounded/full raw feature prefix differs")
        prefix_audit[group] = {"rows": len(full_prefix), "columns": 612, "bit_exact": True}

    physical = preregister["physical_stage1_inputs"]
    fit_labels, fit_label_evidence = _read_label_prefix(
        raw_dir / "train/train_labels.csv",
        physical["labels_stage1_score_prefix_after_lock"],
        passed_groups,
    )
    baseline_path = artifact_root / "oof/gate2024_locked_v3_cf_fix.parquet"
    baseline_identity = _assert_identity(
        baseline_path, preregister["baseline_contract"]["stage2_after_group_promotion_only"]
    )
    baseline = pd.read_parquet(baseline_path, engine="pyarrow")
    baseline.index = pd.DatetimeIndex(baseline.index, name="forecast_kst_dtm")
    if not baseline.index.equals(YEAR_2024) or tuple(baseline.columns) != TARGET_COLS:
        raise AssertionError("Stage2 baseline changed")

    models: dict[str, Any] = {}
    direct: dict[str, pd.Series] = {}
    candidates: dict[str, pd.Series] = {}
    training: dict[str, Any] = {}
    for group in passed_groups:
        fit_index = PRE2024 if group in TARGET_COLS[:2] else YEAR_2023
        sampled = stage1_lock["hpo_recipes"][group]
        model, metadata = fit_direct_cf(
            features[group].loc[fit_index],
            fit_labels.loc[fit_index, group] / CAPACITY_KWH[group],
            sampled,
        )
        raw_prediction = predict_direct_cf(model, features[group].loc[YEAR_2024])
        candidate = blend_with_baseline(
            baseline[group],
            raw_prediction,
            weight=_selected_weight(stage1_lock["locked_candidates"][group]),
            capacity_kwh=CAPACITY_KWH[group],
        )
        models[group] = model
        direct[group] = raw_prediction
        candidates[group] = candidate
        training[group] = {
            "fit_start": fit_index.min(),
            "fit_end": fit_index.max(),
            "apply_start": YEAR_2024.min(),
            "apply_end": YEAR_2024.max(),
            "fit_before_apply": True,
            "fixed_hpo_recipe": sampled,
            "fixed_candidate": stage1_lock["locked_candidates"][group],
            "model": metadata,
        }
    model_path = out_dir / "models/stage2_fixed_models.joblib"
    direct_path = out_dir / "predictions/stage2_direct_cf.parquet"
    candidate_path = out_dir / "predictions/stage2_candidates.parquet"
    _atomic_joblib(models, model_path)
    _atomic_parquet(pd.concat([direct[group].rename(group) for group in passed_groups], axis=1), direct_path)
    _atomic_parquet(pd.concat([candidates[group].rename(group) for group in passed_groups], axis=1), candidate_path)
    reloaded = joblib.load(model_path)
    reload_audit: dict[str, Any] = {}
    for group in passed_groups:
        observed = predict_direct_cf(reloaded[group], features[group].loc[YEAR_2024])
        if not np.array_equal(observed.to_numpy(), direct[group].to_numpy()):
            raise AssertionError("Stage2 model reload prediction differs")
        reload_audit[group] = {"prediction_bit_exact": True, "rows": len(observed)}
    prescore_path = out_dir / "stage2_prescore_lock.json"
    _write_json(
        prescore_path,
        {
            "preregister_sha256": PREREGISTER_SHA256,
            "stage1_promotion_lock": describe_file(out_dir / "stage1_promotion_lock.json"),
            "passed_groups_only": passed_groups,
            "feature_evidence": feature_evidence,
            "bounded_vs_full_prefix_audit": prefix_audit,
            "fit_label_evidence": fit_label_evidence,
            "baseline": baseline_identity,
            "training": training,
            "reload_audit": reload_audit,
            "outputs": [describe_file(path) for path in (model_path, direct_path, candidate_path)],
            "2024_application_target_values_materialized": 0,
            "created_before_2024_application_labels": True,
            "2025_read": False,
        },
    )

    score_labels, score_label_evidence = _read_full_labels(raw_dir, passed_groups)
    group_results: dict[str, Any] = {}
    promoted: list[str] = []
    segments = _segments(YEAR_2024, 2024)
    for group in passed_groups:
        comparisons: dict[str, Any] = {}
        for segment in STAGE2_REQUIRED:
            idx = segments[segment]
            base_metrics = _score_summary(score_labels.loc[idx, group], baseline.loc[idx, group], group)
            candidate_metrics = _score_summary(score_labels.loc[idx, group], candidates[group].loc[idx], group)
            comparisons[segment] = {
                "baseline": base_metrics,
                "candidate": candidate_metrics,
                "delta": float(candidate_metrics["score"] - base_metrics["score"]),
            }
        deltas = [float(comparisons[name]["delta"]) for name in STAGE2_REQUIRED]
        passed = all(delta > 0.0 for delta in deltas)
        if passed:
            promoted.append(group)
        group_results[group] = {
            "fixed_candidate": stage1_lock["locked_candidates"][group],
            "comparisons": comparisons,
            "minimum_delta": min(deltas),
            "mean_delta": float(np.mean(deltas)),
            "all_strictly_positive": passed,
        }
    result = {
        "preregister_sha256": PREREGISTER_SHA256,
        "performed": True,
        "stage1_passed_groups": passed_groups,
        "stage2_passed_groups": promoted,
        "prescore_lock": describe_file(prescore_path),
        "score_labels_read_after_prescore_lock": score_label_evidence,
        "group_results": group_results,
        "no_2024_reselection_or_retuning": True,
        "2025_read": False,
    }
    _write_json(result_path, result)
    lock = {
        "preregister_sha256": PREREGISTER_SHA256,
        "stage1_promotion_lock": describe_file(out_dir / "stage1_promotion_lock.json"),
        "stage2_results": describe_file(result_path),
        "stage2_passed_groups": promoted,
        "locked_candidates": {
            group: stage1_lock["locked_candidates"][group] for group in promoted
        },
        "hpo_recipes": {group: stage1_lock["hpo_recipes"][group] for group in promoted},
        "2024_read": True,
        "2025_read": False,
        "only_promoted_groups_may_open_2025": True,
    }
    _write_json(lock_path, lock)
    print(f"Stage2 nested-HPO promoted groups: {promoted}", flush=True)
    return lock


def _load_stage2_lock(out_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    lock_path = out_dir / "stage2_promotion_lock.json"
    result_path = out_dir / "stage2_results.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if lock["preregister_sha256"] != PREREGISTER_SHA256:
        raise AssertionError("Stage2 lock preregistration changed")
    if lock["stage2_results"]["sha256"] != sha256_file(result_path):
        raise AssertionError("Stage2 result identity changed")
    expected = [
        group
        for group in result.get("stage1_passed_groups", [])
        if result["group_results"][group]["all_strictly_positive"]
    ] if result.get("performed") else []
    if expected != lock["stage2_passed_groups"]:
        raise AssertionError("Stage2 promotion cannot be reproduced")
    return lock, result


def _stage_final(
    *,
    raw_dir: Path,
    artifact_root: Path,
    out_dir: Path,
    preregister: Mapping[str, Any],
) -> dict[str, Any]:
    stage2_lock, _ = _load_stage2_lock(out_dir)
    promoted = list(stage2_lock["stage2_passed_groups"])
    result_path = out_dir / "final_results.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    if not promoted:
        result = {
            "preregister_sha256": PREREGISTER_SHA256,
            "performed": False,
            "reason": "no group passed the fixed 2024 confirmation",
            "promoted_groups": [],
            "2025_weather_read": False,
            "2025_baseline_read": False,
            "sample_read": False,
            "csv_written": False,
        }
        _write_json(result_path, result)
        return result

    train_features, train_evidence = _build_raw_features(raw_dir, period="train")
    labels, label_evidence = _read_full_labels(raw_dir, promoted)
    test_features, test_evidence = _build_raw_features(raw_dir, period="test")
    baseline_path = artifact_root / "final_cf_fix/predictions/corrected_v3_test.parquet"
    baseline_identity = _assert_identity(
        baseline_path, preregister["baseline_contract"]["final_after_stage2_promotion_only"]
    )
    baseline = pd.read_parquet(baseline_path, engine="pyarrow")
    baseline.index = pd.DatetimeIndex(baseline.index, name="forecast_kst_dtm")
    if not baseline.index.equals(YEAR_2025) or tuple(baseline.columns) != TARGET_COLS:
        raise AssertionError("final corrected-v3 baseline changed")
    final_prediction = baseline.astype(np.float64).copy()
    models: dict[str, Any] = {}
    direct: dict[str, pd.Series] = {}
    training: dict[str, Any] = {}
    for group in promoted:
        sampled = stage2_lock["hpo_recipes"][group]
        model, metadata = fit_direct_cf(
            train_features[group],
            labels[group] / CAPACITY_KWH[group],
            sampled,
        )
        raw_prediction = predict_direct_cf(model, test_features[group])
        final_prediction[group] = blend_with_baseline(
            baseline[group],
            raw_prediction,
            weight=_selected_weight(stage2_lock["locked_candidates"][group]),
            capacity_kwh=CAPACITY_KWH[group],
        )
        models[group] = model
        direct[group] = raw_prediction
        training[group] = {
            "fixed_hpo_recipe": sampled,
            "fixed_candidate": stage2_lock["locked_candidates"][group],
            "model": metadata,
        }
    for group in TARGET_COLS:
        values = final_prediction[group].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.02 * CAPACITY_KWH[group]):
            raise AssertionError("final prediction violates physical bounds")
        if group not in promoted and not np.array_equal(values, baseline[group].to_numpy()):
            raise AssertionError("failed group differs from corrected-v3 baseline")
    model_path = out_dir / "models/final_promoted_models.joblib"
    direct_path = out_dir / "final/final_direct_cf.parquet"
    prediction_path = out_dir / "final/nested_timeseries_hpo_lgb_2025.parquet"
    _atomic_joblib(models, model_path)
    _atomic_parquet(pd.concat([direct[group].rename(group) for group in promoted], axis=1), direct_path)
    _atomic_parquet(final_prediction, prediction_path)
    reloaded = joblib.load(model_path)
    for group in promoted:
        observed = predict_direct_cf(reloaded[group], test_features[group])
        if not np.array_equal(observed.to_numpy(), direct[group].to_numpy()):
            raise AssertionError("final reload prediction differs")

    sample_path = raw_dir / "sample_submission.csv"
    sample_identity = _assert_identity(sample_path, FULL_INPUTS["sample_submission"])
    sample = pd.read_csv(sample_path, encoding="utf-8-sig")
    expected_columns = ("forecast_id", "forecast_kst_dtm", *TARGET_COLS)
    if tuple(sample.columns) != expected_columns or len(sample) != len(YEAR_2025):
        raise AssertionError("sample submission schema changed")
    sample_times = pd.to_datetime(sample["forecast_kst_dtm"], errors="raise")
    if not pd.DatetimeIndex(sample_times).equals(YEAR_2025.rename("forecast_kst_dtm")):
        raise AssertionError("sample timestamp sequence changed")
    submission = sample.copy()
    for group in TARGET_COLS:
        submission[group] = final_prediction[group].to_numpy(dtype=np.float64)
    csv_path = out_dir / "final/nested_timeseries_hpo_lgb_2025.csv"
    _atomic_csv(submission, csv_path)
    if csv_path.read_bytes()[:3] != b"\xef\xbb\xbf":
        raise AssertionError("submission is missing UTF-8 BOM")
    readback = pd.read_csv(csv_path, encoding="utf-8-sig")
    if tuple(readback.columns) != expected_columns or len(readback) != len(submission):
        raise AssertionError("submission round-trip schema changed")
    for group in TARGET_COLS:
        expected = np.round(submission[group].to_numpy(dtype=np.float64), 6)
        if not np.array_equal(readback[group].to_numpy(dtype=np.float64), expected):
            raise AssertionError("submission round-trip values changed")
    result = {
        "preregister_sha256": PREREGISTER_SHA256,
        "performed": True,
        "promoted_groups": promoted,
        "train_features": train_evidence,
        "train_labels": label_evidence,
        "test_features": test_evidence,
        "baseline": baseline_identity,
        "sample": sample_identity,
        "training": training,
        "models": describe_file(model_path),
        "direct_prediction": describe_file(direct_path),
        "final_prediction": describe_file(prediction_path),
        "csv": describe_file(csv_path),
        "csv_validated": True,
        "leaderboard_score_claim": False,
    }
    _write_json(result_path, result)
    return result


def _write_manifest(out_dir: Path, preregister_path: Path) -> Path:
    path = out_dir / "manifest.json"
    if path.exists():
        raise FileExistsError(path)
    output_paths = sorted(
        (candidate for candidate in out_dir.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(out_dir).as_posix(),
    )
    forbidden = ("public", "scale")
    if any(
        token in candidate.relative_to(out_dir).as_posix().lower()
        for candidate in output_paths
        for token in forbidden
    ):
        raise AssertionError("forbidden artifact entered output manifest")
    stage1 = json.loads((out_dir / "stage1_results.json").read_text(encoding="utf-8"))
    stage2 = json.loads((out_dir / "stage2_results.json").read_text(encoding="utf-8"))
    final = json.loads((out_dir / "final_results.json").read_text(encoding="utf-8"))
    manifest = {
        "schema_version": 1,
        "experiment_id": "nested_timeseries_hpo_lgb_direct_cf_strict_v1",
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "runtime": {
            "packages": package_versions(("numpy", "pandas", "lightgbm", "optuna", "pyarrow", "joblib")),
            "git": git_state(PROJECT_DIR),
        },
        "provenance": _snapshot(_provenance_paths(preregister_path)),
        "outputs": [describe_file(candidate) for candidate in output_paths],
        "results": {
            "stage1_passed_groups": stage1["passed_groups"],
            "stage2_passed_groups": stage2["stage2_passed_groups"],
            "final_performed": final["performed"],
            "csv_written": bool(final.get("csv_written", final.get("performed", False))),
            "leaderboard_score_claim": False,
        },
        "forbidden_input_audit": {
            "public_metric_triplets_read": False,
            "public_artifacts_read": False,
            "scale_artifacts_read": False,
            "forbidden_paths_in_input_or_provenance": 0,
        },
    }
    _write_json(path, manifest)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    preregister_path = args.preregister.resolve()
    preregister = _verify_preregister(preregister_path)
    raw_dir = args.raw_dir.resolve()
    artifact_root = args.artifact_root.resolve()
    out_dir = args.out_dir.resolve()
    if args.stage in ("stage1", "all"):
        _stage1(
            raw_dir=raw_dir,
            artifact_root=artifact_root,
            out_dir=out_dir,
            preregister_path=preregister_path,
            preregister=preregister,
        )
    if args.stage in ("stage2", "all"):
        _stage2(
            raw_dir=raw_dir,
            artifact_root=artifact_root,
            out_dir=out_dir,
            preregister=preregister,
        )
    if args.stage in ("final", "all"):
        _stage_final(
            raw_dir=raw_dir,
            artifact_root=artifact_root,
            out_dir=out_dir,
            preregister=preregister,
        )
    if args.stage == "all":
        manifest = _write_manifest(out_dir, preregister_path)
        print(f"manifest: {manifest} ({sha256_file(manifest)})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

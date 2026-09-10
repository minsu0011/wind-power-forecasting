"""Strict-v2 locked, leakage-safe group-3 pseudo-label transfer pipeline.

Stage 1 materializes only labels/features through 2023, reproduces the fixed
screen, evaluates a predeclared direct-blend grid against the reconstructed v3
group-3 OOF, and writes the pre-2024 lock. Stage 2 opens 2024 only after that
lock exists. L1 and Q0.7 standalone pseudo models are always refit on all
available actual group-3 labels plus 2022 pseudo labels and exported for meta
model diversity, even when direct submission blending is rejected.

Validation group-1/group-2 values are excluded at the CSV ``usecols``/``nrows``
boundary, the direct-blend preregistration is exact-schema validated, and the
pre-2024 lock is created exclusively and hash-checked throughout Stage 2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lightgbm import log_evaluation


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.manifest import make_manifest, sha256_file, write_json_atomic  # noqa: E402
from src.metric import CAPACITY_KWH, group_metrics  # noqa: E402
from src.pseudo_transfer import (  # noqa: E402
    ActualAccessLedger,
    GROUP1,
    GROUP2,
    GROUP3,
    MapperConfig,
    WeatherModelConfig,
    build_weather_training_strict,
    make_pseudo_labels_strict,
    make_weather_model,
    read_bounded_actuals,
)


DEV_ROWS = 17_520
TRAIN_ROWS = 26_304
TEST_ROWS = 8_760
YEAR_2022_START = pd.Timestamp("2022-01-01 01:00:00")
YEAR_2023_START = pd.Timestamp("2023-01-01 01:00:00")
DEV_VALID_START = pd.Timestamp("2023-07-01 01:00:00")
DEV_VALID_HALF = pd.Timestamp("2023-10-01 01:00:00")
DEV_END = pd.Timestamp("2024-01-01 00:00:00")
GATE_START = pd.Timestamp("2024-01-01 01:00:00")
# The supplied 8,784-hour leap-year gate is split into two equal 4,392-hour
# blocks.  This boundary reproduces the preregistered screen diagnostics.
GATE_HALF = pd.Timestamp("2024-07-02 01:00:00")
GATE_END = pd.Timestamp("2025-01-01 00:00:00")
TEST_START = pd.Timestamp("2025-01-01 01:00:00")
TEST_END = pd.Timestamp("2026-01-01 00:00:00")
OBJECTIVES = ("l1", "q07")
ALPHAS = (0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
V3_COMPONENT_WEIGHTS = {
    "q07": 0.20,
    "shared_l1": 0.075,
    "shared_q07": 0.425,
    "top200q07": 0.025,
    "ewq06": 0.275,
}
V3_AFFINE_SCALE = 1.25
V3_AFFINE_BIAS_KWH = -1_200.0
G3_CLIP = (0.0, CAPACITY_KWH[GROUP3] * 1.02)
EXPECTED_DEV_L1 = {
    "baseline_score": 0.5357196695829786,
    "candidate_score": 0.5625832011205407,
}
DIRECT_BLEND_PREREGISTER = {
    "schema_version": 2,
    "experiment": "g3_pseudo_direct_blend",
    "lock_stage": "before_2024_labels_weather_or_predictions_are_read",
    "base_prediction": "locked_v3_group3",
    "candidate": "pseudo_l1",
    "alpha_grid": list(ALPHAS),
    "variants": {
        "raw": {"transform": "identity"},
        "affine": {
            "transform": "affine_clip",
            "scale": V3_AFFINE_SCALE,
            "bias_kwh": V3_AFFINE_BIAS_KWH,
            "clip_capacity_fraction": [0.0, 1.02],
        },
    },
    "selection_period": {
        "start": DEV_VALID_START.isoformat(),
        "end": DEV_END.isoformat(),
        "half_boundary": DEV_VALID_HALF.isoformat(),
    },
    "selection_rule": {
        "eligibility": "strictly_positive_group_score_gain_on_full_and_both_halves",
        "primary": "maximum_full_candidate_group_score",
        "tie_break": ["lower_alpha", "lexicographic_variant"],
    },
    "confirmation_period": {
        "start": GATE_START.isoformat(),
        "end": GATE_END.isoformat(),
        "half_boundary": GATE_HALF.isoformat(),
    },
    "confirmation_rule": "promote_only_if_preselected_blend_is_positive_on_full_and_both_halves",
    "selection_changes_after_2024": False,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run locked group-3 pseudo transfer, confirmation, and full fit."
    )
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--dev-v3-candidates",
        type=Path,
        default=PROJECT_DIR / "artifacts" / "oof" / "g3dev2023h2_candidates.parquet",
    )
    parser.add_argument(
        "--gate-v3",
        type=Path,
        default=PROJECT_DIR / "artifacts" / "gate" / "v3" / "gate_oof.parquet",
    )
    parser.add_argument(
        "--final-v3",
        type=Path,
        default=(
            PROJECT_DIR
            / "artifacts"
            / "final_v3"
            / "predictions"
            / "v3_locked_full_2025__final_test.parquet"
        ),
    )
    parser.add_argument(
        "--ensemble-preregister",
        type=Path,
        default=(
            PROJECT_DIR
            / "configs"
            / "g3_pseudo_direct_blend_preregister.v2.json"
        ),
        help="exact-schema pre-2024 direct-blend contract",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=PROJECT_DIR / "artifacts" / "postgate" / "g3_pseudo_transfer",
        help="legacy artifact used only after all strict outputs exist for bit checks",
    )
    parser.add_argument("--n-jobs", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _expected_index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="h", name="forecast_kst_dtm")


def _assert_index(
    index: pd.Index, start: pd.Timestamp, end: pd.Timestamp, *, name: str
) -> pd.DatetimeIndex:
    observed = pd.DatetimeIndex(index, name="forecast_kst_dtm")
    expected = _expected_index(start, end)
    if not observed.equals(expected):
        raise ValueError(
            f"{name}: expected {len(expected)} hours {start}..{end}, got "
            f"{len(observed)} {observed.min()}..{observed.max()}"
        )
    return observed


def _read_actual_slice(
    path: Path,
    *,
    columns: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    skip_data_rows: int,
    ledger: ActualAccessLedger,
    purpose: str,
    forbidden_g12_index: Sequence[Any] | pd.Index = (),
) -> pd.DataFrame:
    """Read one declared actual slice without materializing other columns/rows."""

    expected = _expected_index(start, end)
    return read_bounded_actuals(
        path,
        columns=columns,
        expected_index=expected,
        ledger=ledger,
        purpose=purpose,
        forbidden_g12_index=forbidden_g12_index,
        skip_data_rows=skip_data_rows,
    )


def _read_bounded_weather(path: Path, rows: int) -> pd.DataFrame:
    batch = next(pq.ParquetFile(path).iter_batches(batch_size=rows), None)
    if batch is None or batch.num_rows != rows:
        raise ValueError(f"{path}: expected bounded batch of {rows} rows")
    frame = batch.to_pandas()
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if frame.index.max() != DEV_END:
        raise ValueError("bounded Stage-1 weather unexpectedly reaches beyond 2023")
    if frame.isna().any().any() or not np.isfinite(frame.to_numpy()).all():
        raise ValueError("weather cache contains non-finite values")
    return frame.astype("float32")


def _read_weather(path: Path, expected_rows: int) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if len(frame) != expected_rows or not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{path}: unexpected weather cache index")
    if frame.isna().any().any() or not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"{path}: non-finite weather")
    return frame.astype("float32")


def _fit_weather_pair(
    weather: pd.DataFrame,
    actual_g3_kwh: pd.Series,
    pseudo_cf: pd.Series,
    *,
    actual_index: pd.DatetimeIndex,
    pseudo_index: pd.DatetimeIndex,
    predict_index: pd.DatetimeIndex,
    config: WeatherModelConfig,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    empty_pseudo = pd.Series(np.nan, index=pseudo_index, dtype=float)
    baseline_training = build_weather_training_strict(
        weather,
        actual_g3_kwh,
        empty_pseudo,
        actual_index=actual_index,
        pseudo_index=pseudo_index,
        eligible_fraction=config.eligible_fraction,
    )
    candidate_training = build_weather_training_strict(
        weather,
        actual_g3_kwh,
        pseudo_cf,
        actual_index=actual_index,
        pseudo_index=pseudo_index,
        eligible_fraction=config.eligible_fraction,
    )
    baseline_models: dict[str, Any] = {}
    candidate_models: dict[str, Any] = {}
    baseline_predictions = pd.DataFrame(index=predict_index)
    candidate_predictions = pd.DataFrame(index=predict_index)
    for objective in OBJECTIVES:
        baseline = make_weather_model(objective, config)
        baseline.fit(
            baseline_training.features,
            baseline_training.target_cf,
            sample_weight=baseline_training.sample_weight,
            callbacks=[log_evaluation(period=0)],
        )
        candidate = make_weather_model(objective, config)
        candidate.fit(
            candidate_training.features,
            candidate_training.target_cf,
            sample_weight=candidate_training.sample_weight,
            callbacks=[log_evaluation(period=0)],
        )
        baseline_predictions[f"baseline_{objective}"] = (
            baseline.predict(weather.loc[predict_index]) * CAPACITY_KWH[GROUP3]
        )
        candidate_predictions[f"pseudo_{objective}"] = (
            candidate.predict(weather.loc[predict_index]) * CAPACITY_KWH[GROUP3]
        )
        baseline_models[objective] = baseline
        candidate_models[objective] = candidate
    training_audit = {
        "actual_eligible_rows": int(len(candidate_training.actual_index)),
        "pseudo_eligible_rows": int(len(candidate_training.pseudo_index)),
        "combined_rows": int(len(candidate_training.ordered_index)),
        "combined_first_timestamp": candidate_training.ordered_index[0].isoformat(),
        "combined_last_timestamp": candidate_training.ordered_index[-1].isoformat(),
        "actual_then_pseudo_nonchronological_order": bool(
            candidate_training.ordered_index[0] > candidate_training.ordered_index[-1]
        ),
        "sample_weight_unique": np.unique(candidate_training.sample_weight).tolist(),
        "downstream_feature_columns": int(candidate_training.features.shape[1]),
        "downstream_contains_g1g2_actual": False,
    }
    return (
        baseline_models,
        candidate_models,
        baseline_predictions,
        candidate_predictions,
        training_audit,
    )


def _metric(actual: pd.Series, prediction: pd.Series) -> dict[str, Any]:
    if not actual.index.equals(prediction.index):
        raise ValueError("metric input timestamps differ")
    values = group_metrics(
        actual.to_numpy(dtype=float),
        prediction.to_numpy(dtype=float),
        CAPACITY_KWH[GROUP3],
        group_name=GROUP3,
    )
    result = values.as_dict()
    result["score"] = 0.5 * (values.one_minus_nmae + values.ficr)
    return result


def _evaluation(
    actual: pd.Series,
    baseline: pd.Series,
    candidate: pd.Series,
    half: pd.Timestamp,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    masks = {
        "full": np.ones(len(actual), dtype=bool),
        "first_half": np.asarray(actual.index < half),
        "second_half": np.asarray(actual.index >= half),
    }
    for name, mask in masks.items():
        base_metric = _metric(actual.loc[mask], baseline.loc[mask])
        candidate_metric = _metric(actual.loc[mask], candidate.loc[mask])
        result[name] = {
            "baseline": base_metric,
            "candidate": candidate_metric,
            "score_gain": float(candidate_metric["score"] - base_metric["score"]),
        }
    return result


def _strict_positive(evaluation: dict[str, Any]) -> bool:
    return bool(
        evaluation["full"]["score_gain"] > 0.0
        and evaluation["first_half"]["score_gain"] > 0.0
        and evaluation["second_half"]["score_gain"] > 0.0
    )


def _v3_dev_predictions(candidates_path: Path) -> tuple[pd.Series, pd.Series]:
    candidates = pd.read_parquet(candidates_path)
    candidates.index = _assert_index(
        candidates.index, DEV_VALID_START, DEV_END, name="v3 dev candidates"
    )
    missing = set(V3_COMPONENT_WEIGHTS).difference(candidates.columns)
    if missing:
        raise ValueError(f"v3 candidate frame missing columns: {sorted(missing)}")
    raw = sum(
        weight * candidates[column]
        for column, weight in V3_COMPONENT_WEIGHTS.items()
    ).rename("v3_raw")
    v3 = np.clip(
        V3_AFFINE_SCALE * raw + V3_AFFINE_BIAS_KWH,
        *G3_CLIP,
    ).rename("v3")
    return raw, v3


def _blend_screen(
    actual: pd.Series,
    v3: pd.Series,
    pseudo_l1: pd.Series,
    preregister: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    variants: dict[str, pd.Series] = {}
    for name, specification in preregister["variants"].items():
        if specification["transform"] == "identity":
            variants[name] = pseudo_l1
        elif specification["transform"] == "affine_clip":
            lower, upper = specification["clip_capacity_fraction"]
            variants[name] = pd.Series(
                np.clip(
                    float(specification["scale"]) * pseudo_l1
                    + float(specification["bias_kwh"]),
                    lower * CAPACITY_KWH[GROUP3],
                    upper * CAPACITY_KWH[GROUP3],
                ),
                index=pseudo_l1.index,
            )
        else:  # Exact-schema validation should make this unreachable.
            raise ValueError(f"unsupported preregistered transform: {specification}")
    records: list[dict[str, Any]] = []
    for variant, pseudo in variants.items():
        for alpha in preregister["alpha_grid"]:
            blended = pd.Series(
                np.clip((1.0 - alpha) * v3 + alpha * pseudo, *G3_CLIP),
                index=v3.index,
            )
            evaluation = _evaluation(actual, v3, blended, DEV_VALID_HALF)
            records.append(
                {
                    "variant": variant,
                    "alpha": alpha,
                    "evaluation": evaluation,
                    "stable_full_and_both_halves": _strict_positive(evaluation),
                }
            )
    stable = [record for record in records if record["stable_full_and_both_halves"]]
    selected = (
        max(
            stable,
            key=lambda record: (
                record["evaluation"]["full"]["candidate"]["score"],
                -record["alpha"],
                record["variant"],
            ),
        )
        if stable
        else None
    )
    return records, selected


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        frame.to_parquet(temporary, compression="zstd", index=True)
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


def _write_submission(
    sample_path: Path,
    final_v3_path: Path,
    pseudo_prediction: pd.Series,
    selected: dict[str, Any],
    preregister: dict[str, Any],
    destination: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    final_v3 = pd.read_parquet(final_v3_path)
    final_v3.index = _assert_index(final_v3.index, TEST_START, TEST_END, name="final v3")
    pseudo = pseudo_prediction
    variant = preregister["variants"][selected["variant"]]
    if variant["transform"] == "affine_clip":
        lower, upper = variant["clip_capacity_fraction"]
        pseudo = pd.Series(
            np.clip(
                float(variant["scale"]) * pseudo + float(variant["bias_kwh"]),
                lower * CAPACITY_KWH[GROUP3],
                upper * CAPACITY_KWH[GROUP3],
            ),
            index=pseudo.index,
        )
    alpha = float(selected["alpha"])
    combined = final_v3.copy()
    combined[GROUP3] = np.clip(
        (1.0 - alpha) * final_v3[GROUP3] + alpha * pseudo,
        *G3_CLIP,
    )
    sample = pd.read_csv(sample_path, encoding="utf-8-sig")
    sample_times = pd.DatetimeIndex(
        pd.to_datetime(sample["forecast_kst_dtm"], errors="raise"),
        name="forecast_kst_dtm",
    )
    if not sample_times.equals(combined.index):
        raise ValueError("sample submission/final prediction timestamps differ")
    output = sample.copy()
    for group in (GROUP1, GROUP2, GROUP3):
        output[group] = combined[group].to_numpy(dtype=float)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        output.to_csv(
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
    readback = pd.read_csv(destination, encoding="utf-8-sig")
    if tuple(readback.columns) != tuple(sample.columns) or len(readback) != len(sample):
        raise IOError("submission readback schema changed")
    return combined, output


def _validate_direct_preregister(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload != DIRECT_BLEND_PREREGISTER:
        raise ValueError(
            "direct-blend preregistration does not match the exact v2 schema, "
            "variants, boundaries, grid, and selection rules"
        )
    return payload


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _create_or_verify_lock(path: Path, payload: dict[str, Any]) -> str:
    """Create a lock exclusively; an existing lock is verified, never replaced."""

    path.parent.mkdir(parents=True, exist_ok=True)
    expected = _canonical_json_bytes(payload)
    expected_sha = hashlib.sha256(expected).hexdigest()
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        observed_sha = sha256_file(path)
        if observed_sha != expected_sha:
            raise FileExistsError(
                f"immutable pre-2024 lock differs: expected {expected_sha}, "
                f"observed {observed_sha}"
            )
        return observed_sha
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(expected)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    observed_sha = sha256_file(path)
    if observed_sha != expected_sha:
        raise IOError("exclusive lock readback hash differs")
    return observed_sha


def _assert_lock_snapshot(path: Path, expected_sha: str, *, phase: str) -> None:
    observed = sha256_file(path)
    if observed != expected_sha:
        raise RuntimeError(
            f"immutable lock changed during {phase}: {observed} != {expected_sha}"
        )


def _assert_file_snapshot(snapshot: dict[str, str], *, phase: str) -> None:
    changed: dict[str, dict[str, str]] = {}
    for path, expected in snapshot.items():
        observed = sha256_file(path)
        if observed != expected:
            changed[path] = {"expected": expected, "observed": observed}
    if changed:
        raise RuntimeError(f"locked inputs changed during {phase}: {changed}")


def _strict_mapper_frame(
    g12: pd.DataFrame,
    g3: pd.Series,
    index: pd.DatetimeIndex,
) -> pd.DataFrame:
    if not g12.index.is_unique or not g3.index.is_unique:
        raise ValueError("strict mapper sources must be unique")
    frame = pd.DataFrame(
        {
            GROUP1: g12.loc[index, GROUP1].to_numpy(dtype=float),
            GROUP2: g12.loc[index, GROUP2].to_numpy(dtype=float),
            GROUP3: g3.loc[index].to_numpy(dtype=float),
        },
        index=index,
    )
    if tuple(frame.columns) != (GROUP1, GROUP2, GROUP3) or not frame.index.equals(
        index
    ):
        raise AssertionError("strict mapper frame assembly changed")
    return frame


def _assert_zero_forbidden_g12(
    ledger: ActualAccessLedger, *, phase: str
) -> None:
    audit = ledger.as_dict()
    if audit["forbidden_g1g2_rows_materialized"] != 0:
        raise AssertionError(f"{phase}: forbidden g1/g2 actual access was recorded")


def _bit_exact_reference_check(
    paths: dict[str, Path], reference_dir: Path
) -> dict[str, Any]:
    references = {
        "dev_oof": reference_dir / "oof" / "g3_pseudo_2023h2.parquet",
        "dev_baseline": reference_dir
        / "oof"
        / "g3_pseudo_baseline_2023h2.parquet",
        "gate_oof": reference_dir / "oof" / "g3_pseudo_2024.parquet",
        "gate_baseline": reference_dir
        / "oof"
        / "g3_pseudo_baseline_2024.parquet",
        "final": reference_dir / "predictions" / "g3_pseudo_2025.parquet",
        "dev_pseudo": reference_dir / "pseudo" / "g3_pseudo_cf_2022_dev.parquet",
        "gate_pseudo": reference_dir / "pseudo" / "g3_pseudo_cf_2022_gate.parquet",
        "full_pseudo": reference_dir / "pseudo" / "g3_pseudo_cf_2022_full.parquet",
    }
    result: dict[str, Any] = {}
    for name, reference_path in references.items():
        if not reference_path.is_file():
            raise FileNotFoundError(f"bit-exact reference missing: {reference_path}")
        candidate = pd.read_parquet(paths[name])
        reference = pd.read_parquet(reference_path)
        schema_exact = tuple(candidate.columns) == tuple(reference.columns)
        index_exact = candidate.index.equals(reference.index)
        left = candidate.to_numpy(dtype=np.float64)
        right = reference.to_numpy(dtype=np.float64)
        shape_exact = left.shape == right.shape
        value_bit_exact = bool(
            shape_exact
            and np.array_equal(left.view(np.uint64), right.view(np.uint64))
        )
        record = {
            "reference": str(reference_path),
            "rows": int(len(candidate)),
            "schema_exact": schema_exact,
            "index_exact": index_exact,
            "value_bit_exact": value_bit_exact,
            "parquet_sha256_equal": sha256_file(paths[name])
            == sha256_file(reference_path),
            "candidate_sha256": sha256_file(paths[name]),
            "reference_sha256": sha256_file(reference_path),
        }
        result[name] = record
        if not (schema_exact and index_exact and value_bit_exact):
            raise AssertionError(f"strict output is not bit-exact for {name}: {record}")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    raw_dir = args.raw_dir.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    reference_dir = args.reference_dir.expanduser().resolve()
    dev_v3_path = args.dev_v3_candidates.expanduser().resolve()
    gate_v3_path = args.gate_v3.expanduser().resolve()
    final_v3_path = args.final_v3.expanduser().resolve()
    ensemble_preregister_path = args.ensemble_preregister.expanduser().resolve()
    label_path = raw_dir / "train" / "train_labels.csv"
    train_weather_path = cache_dir / f"{GROUP3}_weather_train.parquet"
    test_weather_path = cache_dir / f"{GROUP3}_weather_test.parquet"
    sample_path = raw_dir / "sample_submission.csv"
    runner_path = Path(__file__).resolve()
    module_path = PROJECT_DIR / "src" / "pseudo_transfer.py"
    metric_path = PROJECT_DIR / "src" / "metric.py"
    manifest_source_path = PROJECT_DIR / "src" / "manifest.py"
    source_files = [
        runner_path,
        module_path,
        metric_path,
        manifest_source_path,
        ensemble_preregister_path,
    ]
    stage1_required = [
        label_path,
        train_weather_path,
        dev_v3_path,
        *source_files,
    ]
    missing = [path for path in stage1_required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"strict Stage-1 inputs missing: {missing}")
    ensemble_preregister = _validate_direct_preregister(ensemble_preregister_path)
    source_snapshot = {str(path): sha256_file(path) for path in source_files}
    stage1_input_snapshot = {
        str(path): sha256_file(path)
        for path in (label_path, train_weather_path, dev_v3_path)
    }
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty pseudo output: {out_dir}")

    mapper_config = MapperConfig()
    weather_config = WeatherModelConfig(n_jobs=args.n_jobs)
    paths = {
        "dev_oof": out_dir / "oof" / "g3_pseudo_2023h2.parquet",
        "dev_baseline": out_dir / "oof" / "g3_pseudo_baseline_2023h2.parquet",
        "gate_oof": out_dir / "oof" / "g3_pseudo_2024.parquet",
        "gate_baseline": out_dir / "oof" / "g3_pseudo_baseline_2024.parquet",
        "final": out_dir / "predictions" / "g3_pseudo_2025.parquet",
        "dev_pseudo": out_dir / "pseudo" / "g3_pseudo_cf_2022_dev.parquet",
        "gate_pseudo": out_dir / "pseudo" / "g3_pseudo_cf_2022_gate.parquet",
        "full_pseudo": out_dir / "pseudo" / "g3_pseudo_cf_2022_full.parquet",
        "stage1": out_dir / "stage1_pre2024_results.json",
        "lock": out_dir / "pre2024_lock.json",
        "results": out_dir / "results.json",
        "manifest": out_dir / "manifest.json",
        "submission": out_dir / "submission" / "g3_pseudo_v3_blend.csv",
        "submission_prediction": out_dir / "predictions" / "g3_pseudo_v3_blend_2025.parquet",
    }

    # Stage 1: group-3 is bounded to exactly 17,520 rows. Group-1/2 is
    # physically bounded even earlier, so 2023-H2 forbidden values never enter
    # a pandas object.
    index_2022 = _expected_index(YEAR_2022_START, pd.Timestamp("2023-01-01 00:00:00"))
    actual_h1 = _expected_index(YEAR_2023_START, pd.Timestamp("2023-07-01 00:00:00"))
    valid_h2 = _expected_index(DEV_VALID_START, DEV_END)
    stage1_full_index = _expected_index(YEAR_2022_START, DEV_END)
    stage1_access = ActualAccessLedger()
    stage1_g3_bounded = _read_actual_slice(
        label_path,
        columns=(GROUP3,),
        start=YEAR_2022_START,
        end=DEV_END,
        skip_data_rows=0,
        ledger=stage1_access,
        purpose="stage1:g3_bounded_nrows_17520",
    )
    if len(stage1_g3_bounded) != DEV_ROWS:
        raise AssertionError("Stage-1 group-3 reader did not stop at 17,520 rows")
    stage1_g12_allowed = _read_actual_slice(
        label_path,
        columns=(GROUP1, GROUP2),
        start=YEAR_2022_START,
        end=actual_h1[-1],
        skip_data_rows=0,
        ledger=stage1_access,
        purpose="stage1:g1g2_2022_through_train_only",
        forbidden_g12_index=valid_h2,
    )
    actual_h1_g3 = stage1_g3_bounded.loc[actual_h1, GROUP3].copy()
    actual_dev = stage1_g3_bounded.loc[valid_h2, GROUP3].copy()
    mapper_dev_frame = _strict_mapper_frame(
        stage1_g12_allowed, actual_h1_g3, actual_h1
    )
    pseudo_dev_inputs = stage1_g12_allowed.loc[
        index_2022, [GROUP1, GROUP2]
    ].copy()
    pseudo_dev = make_pseudo_labels_strict(
        mapper_dev_frame,
        pseudo_dev_inputs,
        forbidden_validation_index=valid_h2,
        config=mapper_config,
        ledger=stage1_access,
        audit_prefix="stage1",
    )
    _assert_zero_forbidden_g12(stage1_access, phase="Stage-1")
    weather_dev = _read_bounded_weather(train_weather_path, DEV_ROWS)
    if not weather_dev.index.equals(stage1_full_index):
        raise ValueError("Stage1 bounded weather index changed")
    (
        dev_baseline_models,
        dev_candidate_models,
        dev_baseline,
        dev_candidate,
        dev_training_audit,
    ) = _fit_weather_pair(
        weather_dev,
        actual_h1_g3,
        pseudo_dev.pseudo_cf,
        actual_index=actual_h1,
        pseudo_index=index_2022,
        predict_index=valid_h2,
        config=weather_config,
    )
    dev_objective_results: dict[str, Any] = {}
    for objective in OBJECTIVES:
        dev_objective_results[objective] = _evaluation(
            actual_dev,
            dev_baseline[f"baseline_{objective}"],
            dev_candidate[f"pseudo_{objective}"],
            DEV_VALID_HALF,
        )
    if not np.isclose(
        dev_objective_results["l1"]["full"]["baseline"]["score"],
        EXPECTED_DEV_L1["baseline_score"],
        atol=1e-12,
    ) or not np.isclose(
        dev_objective_results["l1"]["full"]["candidate"]["score"],
        EXPECTED_DEV_L1["candidate_score"],
        atol=1e-12,
    ):
        raise AssertionError("fixed L1 screen no longer reproduces exactly")

    v3_raw, v3 = _v3_dev_predictions(dev_v3_path)
    blend_records, selected_blend = _blend_screen(
        actual_dev,
        v3,
        dev_candidate["pseudo_l1"],
        ensemble_preregister,
    )
    best_raw = max(
        [record for record in blend_records if record["variant"] == "raw"],
        key=lambda record: record["evaluation"]["full"]["score_gain"],
    )
    if best_raw["alpha"] != 0.025 or best_raw["evaluation"]["first_half"]["score_gain"] >= 0.0:
        raise AssertionError("locked raw direct-blend rejection screen changed")
    if selected_blend is not None:
        raise AssertionError("direct blend unexpectedly passed the pre-2024 half rule")

    _atomic_parquet(dev_candidate, paths["dev_oof"])
    _atomic_parquet(dev_baseline, paths["dev_baseline"])
    _atomic_parquet(pseudo_dev.pseudo_cf.to_frame(), paths["dev_pseudo"])
    for objective in OBJECTIVES:
        _atomic_joblib(
            dev_candidate_models[objective],
            out_dir / "models" / f"dev__g3_pseudo_{objective}.joblib",
        )
    _atomic_joblib(pseudo_dev.mapper, out_dir / "models" / "dev__mapper.joblib")

    stage1 = {
        "stage": "pre_2024_only",
        "last_label_and_weather_timestamp_materialized": DEV_END.isoformat(),
        "g3_label_rows_bounded": DEV_ROWS,
        "actual_access_ledger": stage1_access.as_dict(),
        "mapper_audit": pseudo_dev.audit,
        "weather_training_audit": dev_training_audit,
        "objective_results": dev_objective_results,
        "v3_reconstruction": {
            "candidate_path": str(dev_v3_path),
            "candidate_sha256": sha256_file(dev_v3_path),
            "raw_weights": V3_COMPONENT_WEIGHTS,
            "affine_scale": V3_AFFINE_SCALE,
            "affine_bias_kwh": V3_AFFINE_BIAS_KWH,
            "clip_kwh": list(G3_CLIP),
            "raw_summary": {
                "mean": float(v3_raw.mean()),
                "min": float(v3_raw.min()),
                "max": float(v3_raw.max()),
            },
        },
        "direct_blend_records": blend_records,
        "selected_direct_blend": selected_blend,
        "direct_blend_adoption": selected_blend is not None,
    }
    write_json_atomic(paths["stage1"], stage1, overwrite=args.overwrite)
    lock = {
        "schema_version": 2,
        "lock_stage": "written_before_2024_labels_weather_or_predictions_are_read",
        "stage1_results_sha256": sha256_file(paths["stage1"]),
        "stage1_input_sha256": stage1_input_snapshot,
        "source_and_config_sha256": source_snapshot,
        "mapper_config": asdict(mapper_config),
        "weather_model_config": asdict(weather_config),
        "standalone_candidates": ["pseudo_l1", "pseudo_q07"],
        "standalone_confirmation_rule": "full, first_half, second_half gains must each be > 0",
        "standalone_confirmation_half_boundary": GATE_HALF.isoformat(),
        "direct_blend_preregister": ensemble_preregister,
        "selected_direct_blend": selected_blend,
        "direct_blend_adoption": selected_blend is not None,
        "ensemble_preregister_path": str(ensemble_preregister_path),
        "ensemble_preregister_sha256": sha256_file(ensemble_preregister_path),
        "selection_changes_after_2024": False,
        "always_export_standalone_for_meta": True,
    }
    lock_snapshot_sha = _create_or_verify_lock(paths["lock"], lock)
    lock = json.loads(paths["lock"].read_text(encoding="utf-8"))
    _assert_lock_snapshot(paths["lock"], lock_snapshot_sha, phase="Stage-1 handoff")
    _assert_file_snapshot(source_snapshot, phase="Stage-1 source handoff")
    _assert_file_snapshot(stage1_input_snapshot, phase="Stage-1 data handoff")
    print(
        f"PRE-2024 IMMUTABLE LOCK SNAPSHOT: {lock_snapshot_sha} {paths['lock']}",
        flush=True,
    )

    # Stage 2 inputs are checked only after the lock snapshot exists. Validation
    # group-1/2 values are never requested: gate mapper inputs end at DEV_END,
    # while group-3 validation actuals are loaded in a separate score-only slice.
    stage2_required = [gate_v3_path, final_v3_path, test_weather_path, sample_path]
    missing = [path for path in stage2_required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"strict Stage-2 inputs missing: {missing}")
    weather_full = _read_weather(train_weather_path, TRAIN_ROWS)
    _assert_index(weather_full.index, YEAR_2022_START, GATE_END, name="train weather")
    actual_2023 = _expected_index(YEAR_2023_START, DEV_END)
    valid_2024 = _expected_index(GATE_START, GATE_END)
    gate_access = ActualAccessLedger()
    gate_g12_allowed = _read_actual_slice(
        label_path,
        columns=(GROUP1, GROUP2),
        start=YEAR_2022_START,
        end=DEV_END,
        skip_data_rows=0,
        ledger=gate_access,
        purpose="stage2_gate:g1g2_2022_through_2023_train_only",
        forbidden_g12_index=valid_2024,
    )
    gate_g3_train_frame = _read_actual_slice(
        label_path,
        columns=(GROUP3,),
        start=YEAR_2023_START,
        end=DEV_END,
        skip_data_rows=TEST_ROWS,
        ledger=gate_access,
        purpose="stage2_gate:g3_train_only",
    )
    gate_g3_score_frame = _read_actual_slice(
        label_path,
        columns=(GROUP3,),
        start=GATE_START,
        end=GATE_END,
        skip_data_rows=DEV_ROWS,
        ledger=gate_access,
        purpose="stage2_gate:g3_validation_score_only",
    )
    gate_g3_train = gate_g3_train_frame[GROUP3].copy()
    actual_gate = gate_g3_score_frame[GROUP3].copy()
    mapper_gate_frame = _strict_mapper_frame(
        gate_g12_allowed, gate_g3_train, actual_2023
    )
    pseudo_gate_inputs = gate_g12_allowed.loc[
        index_2022, [GROUP1, GROUP2]
    ].copy()
    pseudo_gate = make_pseudo_labels_strict(
        mapper_gate_frame,
        pseudo_gate_inputs,
        forbidden_validation_index=valid_2024,
        config=mapper_config,
        ledger=gate_access,
        audit_prefix="stage2_gate",
    )
    _assert_zero_forbidden_g12(gate_access, phase="Stage-2 gate")
    (
        gate_baseline_models,
        gate_candidate_models,
        gate_baseline,
        gate_candidate,
        gate_training_audit,
    ) = _fit_weather_pair(
        weather_full,
        gate_g3_train,
        pseudo_gate.pseudo_cf,
        actual_index=actual_2023,
        pseudo_index=index_2022,
        predict_index=valid_2024,
        config=weather_config,
    )
    gate_objective_results: dict[str, Any] = {}
    standalone_pass: dict[str, bool] = {}
    for objective in OBJECTIVES:
        evaluation = _evaluation(
            actual_gate,
            gate_baseline[f"baseline_{objective}"],
            gate_candidate[f"pseudo_{objective}"],
            GATE_HALF,
        )
        gate_objective_results[objective] = evaluation
        standalone_pass[objective] = _strict_positive(evaluation)
    _atomic_parquet(gate_candidate, paths["gate_oof"])
    _atomic_parquet(gate_baseline, paths["gate_baseline"])
    _atomic_parquet(pseudo_gate.pseudo_cf.to_frame(), paths["gate_pseudo"])
    for objective in OBJECTIVES:
        _atomic_joblib(
            gate_candidate_models[objective],
            out_dir / "models" / f"gate__g3_pseudo_{objective}.joblib",
        )
    _atomic_joblib(pseudo_gate.mapper, out_dir / "models" / "gate__mapper.joblib")

    direct_blend_confirmation: dict[str, Any] | None = None
    direct_blend_pass = False
    if lock["selected_direct_blend"] is not None:
        selected = lock["selected_direct_blend"]
        gate_v3 = pd.read_parquet(gate_v3_path)[GROUP3]
        gate_v3.index = _assert_index(
            gate_v3.index, GATE_START, GATE_END, name="gate v3"
        )
        pseudo = gate_candidate["pseudo_l1"]
        variant = ensemble_preregister["variants"][selected["variant"]]
        if variant["transform"] == "affine_clip":
            lower, upper = variant["clip_capacity_fraction"]
            pseudo = pd.Series(
                np.clip(
                    float(variant["scale"]) * pseudo + float(variant["bias_kwh"]),
                    lower * CAPACITY_KWH[GROUP3],
                    upper * CAPACITY_KWH[GROUP3],
                ),
                index=pseudo.index,
            )
        alpha = float(selected["alpha"])
        blended = pd.Series(
            np.clip((1.0 - alpha) * gate_v3 + alpha * pseudo, *G3_CLIP),
            index=gate_v3.index,
        )
        direct_blend_confirmation = _evaluation(
            actual_gate, gate_v3, blended, GATE_HALF
        )
        direct_blend_pass = _strict_positive(direct_blend_confirmation)
    _assert_lock_snapshot(paths["lock"], lock_snapshot_sha, phase="Stage-2 gate")
    _assert_file_snapshot(source_snapshot, phase="Stage-2 gate")
    _assert_file_snapshot(stage1_input_snapshot, phase="Stage-2 gate")

    # Final-fit transition occurs only after the locked 2024 confirmation is
    # complete. Group-1/2 2024 actuals now become historical training inputs for
    # the 2025 mapper; they were absent from the gate phase above.
    actual_full = _expected_index(YEAR_2023_START, GATE_END)
    final_access = ActualAccessLedger()
    final_g12_allowed = _read_actual_slice(
        label_path,
        columns=(GROUP1, GROUP2),
        start=YEAR_2022_START,
        end=GATE_END,
        skip_data_rows=0,
        ledger=final_access,
        purpose="final_fit:g1g2_2022_through_2024_historical",
    )
    final_g3_train = pd.concat([gate_g3_train, actual_gate])
    final_g3_train.index = pd.DatetimeIndex(
        final_g3_train.index, name="forecast_kst_dtm"
    )
    if not final_g3_train.index.equals(actual_full):
        raise AssertionError("final group-3 training target order changed")
    mapper_full_frame = _strict_mapper_frame(
        final_g12_allowed, final_g3_train, actual_full
    )
    pseudo_full_inputs = final_g12_allowed.loc[
        index_2022, [GROUP1, GROUP2]
    ].copy()
    pseudo_full = make_pseudo_labels_strict(
        mapper_full_frame,
        pseudo_full_inputs,
        forbidden_validation_index=(),
        config=mapper_config,
        ledger=final_access,
        audit_prefix="final_fit",
    )
    full_training = build_weather_training_strict(
        weather_full,
        final_g3_train,
        pseudo_full.pseudo_cf,
        actual_index=actual_full,
        pseudo_index=index_2022,
        eligible_fraction=weather_config.eligible_fraction,
    )
    test_weather = _read_weather(test_weather_path, TEST_ROWS)
    test_index = _assert_index(test_weather.index, TEST_START, TEST_END, name="test weather")
    full_predictions = pd.DataFrame(index=test_index)
    for objective in OBJECTIVES:
        model = make_weather_model(objective, weather_config)
        model.fit(
            full_training.features,
            full_training.target_cf,
            sample_weight=full_training.sample_weight,
            callbacks=[log_evaluation(period=0)],
        )
        prediction = model.predict(test_weather) * CAPACITY_KWH[GROUP3]
        if not np.isfinite(prediction).all():
            raise ValueError(f"full pseudo {objective} prediction is non-finite")
        full_predictions[f"pseudo_{objective}"] = prediction
        _atomic_joblib(
            model, out_dir / "models" / f"full__g3_pseudo_{objective}.joblib"
        )
    _atomic_joblib(pseudo_full.mapper, out_dir / "models" / "full__mapper.joblib")
    _atomic_parquet(full_predictions, paths["final"])
    _atomic_parquet(pseudo_full.pseudo_cf.to_frame(), paths["full_pseudo"])

    submission_created = False
    if lock["selected_direct_blend"] is not None and direct_blend_pass:
        combined, _ = _write_submission(
            sample_path,
            final_v3_path,
            full_predictions["pseudo_l1"],
            lock["selected_direct_blend"],
            ensemble_preregister,
            paths["submission"],
        )
        _atomic_parquet(combined, paths["submission_prediction"])
        submission_created = True

    # Readback contract used by downstream threshold-meta experiments.
    expected_frames = (
        (paths["dev_oof"], 4_416, DEV_VALID_START, DEV_END),
        (paths["gate_oof"], 8_784, GATE_START, GATE_END),
        (paths["final"], 8_760, TEST_START, TEST_END),
    )
    for path, rows, start, end in expected_frames:
        frame = pd.read_parquet(path)
        if tuple(frame.columns) != ("pseudo_l1", "pseudo_q07"):
            raise IOError(f"{path}: downstream pseudo column contract changed")
        _assert_index(frame.index, start, end, name=path.name)
        if len(frame) != rows or not np.isfinite(frame.to_numpy()).all():
            raise IOError(f"{path}: downstream pseudo readback failed")
    _assert_lock_snapshot(paths["lock"], lock_snapshot_sha, phase="final fit")
    _assert_file_snapshot(source_snapshot, phase="final fit")
    _assert_file_snapshot(stage1_input_snapshot, phase="final fit")
    bit_exact = _bit_exact_reference_check(paths, reference_dir)

    results = {
        "status": "completed",
        "strict_schema_version": 2,
        "pre2024_lock_sha256": lock_snapshot_sha,
        "lock_unchanged_through_final_fit": True,
        "source_and_config_sha256": source_snapshot,
        "stage1_input_sha256": stage1_input_snapshot,
        "stage1_l1_exact_reproduction": EXPECTED_DEV_L1,
        "stage1_objective_results": dev_objective_results,
        "stage1_direct_blend_adoption": lock["direct_blend_adoption"],
        "stage1_actual_access": stage1_access.as_dict(),
        "stage2_mapper_audit": pseudo_gate.audit,
        "stage2_actual_access": gate_access.as_dict(),
        "stage2_weather_training_audit": gate_training_audit,
        "stage2_objective_results": gate_objective_results,
        "standalone_pass_full_and_both_halves": standalone_pass,
        "direct_blend_confirmation": direct_blend_confirmation,
        "direct_blend_pass": direct_blend_pass,
        "submission_created": submission_created,
        "full_mapper_audit": pseudo_full.audit,
        "final_fit_actual_access": final_access.as_dict(),
        "full_weather_training": {
            "actual_eligible_rows": int(len(full_training.actual_index)),
            "pseudo_eligible_rows": int(len(full_training.pseudo_index)),
            "combined_rows": int(len(full_training.ordered_index)),
            "downstream_contains_g1g2_actual": False,
        },
        "downstream_contract": {
            "unit": "kWh",
            "columns": ["pseudo_l1", "pseudo_q07"],
            "dev_2023h2": str(paths["dev_oof"]),
            "gate_2024": str(paths["gate_oof"]),
            "full_2025": str(paths["final"]),
        },
        "legacy_prediction_bit_exact": bit_exact,
        "note": (
            "Standalone pseudo predictions are always exported for meta-model "
            "diversity. A direct submission CSV is created only after the "
            "prelocked v3 blend passes both 2023 and 2024 half checks."
        ),
    }
    write_json_atomic(paths["results"], results, overwrite=args.overwrite)

    model_files = sorted((out_dir / "models").glob("*.joblib"))
    output_files = [
        paths["dev_oof"],
        paths["dev_baseline"],
        paths["gate_oof"],
        paths["gate_baseline"],
        paths["final"],
        paths["dev_pseudo"],
        paths["gate_pseudo"],
        paths["full_pseudo"],
        paths["stage1"],
        paths["lock"],
        paths["results"],
        *model_files,
    ]
    if submission_created:
        output_files.extend([paths["submission"], paths["submission_prediction"]])
    reference_inputs = [Path(record["reference"]) for record in bit_exact.values()]
    manifest_inputs = [
        label_path,
        train_weather_path,
        test_weather_path,
        sample_path,
        dev_v3_path,
        gate_v3_path,
        final_v3_path,
        *source_files,
        *reference_inputs,
    ]
    manifest = make_manifest(
        artifact_type="baram_g3_strict_v2_pseudo_transfer",
        parameters={
            "mapper_config": asdict(mapper_config),
            "weather_model_config": asdict(weather_config),
            "direct_blend_preregister": ensemble_preregister,
            "v3_component_weights": V3_COMPONENT_WEIGHTS,
            "valid_test_g1g2_actual_as_downstream_features": False,
            "validation_g1g2_actual_rows_materialized_before_gate_score": 0,
            "pre2024_lock_exclusive_and_unchanged": True,
            "source_and_config_sha256_bound": True,
        },
        input_files=manifest_inputs,
        output_files=output_files,
        results=results,
        project_dir=PROJECT_DIR,
    )
    write_json_atomic(paths["manifest"], manifest, overwrite=args.overwrite)
    print(
        "standalone pass:", standalone_pass,
        "direct blend adoption:", lock["direct_blend_adoption"],
        "submission:", submission_created,
    )
    print(f"complete: {paths['manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Append-only G3 Stage2/final for the frozen full-weather ordinal Bayes v1."""

from __future__ import annotations

import argparse
import atexit
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts import run_catboost_multiquantile_bayes as bounded  # noqa: E402
from scripts import run_ficr_bayes_decision_strict as strict  # noqa: E402
from scripts import run_full_weather_ordinal_bayes as stage1  # noqa: E402
from scripts import run_shared_q07_multiseed as shared  # noqa: E402
from src.full_weather_ordinal_bayes import (  # noqa: E402
    BIN_COUNT,
    FullWeatherOrdinalBayes,
    TRANSFER_WEIGHT,
    transfer_action_kwh,
)
from src.manifest import (  # noqa: E402
    describe_file,
    git_state,
    package_versions,
    sha256_file,
    utc_now,
)
from src.metric import CAPACITY_KWH, TARGET_COLS, score_details  # noqa: E402


STAGE1_MANIFEST_SHA256 = "29786f6826f9a4873bec1d2781653aba2edad2a5b037e8ed023b38f88e594efe"
PREREGISTER_SHA256 = stage1.PREREGISTER_SHA256
ARTIFACT_TYPE = "full_weather_ordinal_bayes_g3_stage2_final_v1"
REQUIRED_SEGMENTS = ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4")
G3 = "kpx_group_3"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir", type=Path, default=Path(r"data/local/open")
    )
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/cache"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/postgate/full_weather_ordinal_bayes_strict_v1"),
    )
    parser.add_argument(
        "--preregister",
        type=Path,
        default=Path("configs/full_weather_ordinal_bayes_preregister_v1.json"),
    )
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def _verify_record(record: Mapping[str, Any]) -> Path:
    path = Path(record["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(record["size_bytes"]):
        raise AssertionError(f"size changed: {path}")
    if sha256_file(path) != record["sha256"]:
        raise AssertionError(f"hash changed: {path}")
    return path


def _verify_stage1(out_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = out_dir / "manifest.json"
    if sha256_file(manifest_path) != STAGE1_MANIFEST_SHA256:
        raise AssertionError("frozen Stage1 manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "stage1_group_promotion_pending_stage2":
        raise AssertionError("Stage1 status changed")
    for record in manifest["source_and_config_snapshot"].values():
        _verify_record(record)
    for record in manifest["census_snapshot"].values():
        _verify_record(record)
    for record in manifest["outputs"]:
        _verify_record(record)
    expected = {
        Path(record["path"]).resolve() for record in manifest["outputs"]
    } | {manifest_path.resolve()}
    observed = {path.resolve() for path in out_dir.rglob("*") if path.is_file()}
    extra_observed = observed - expected
    if extra_observed:
        append_path = out_dir / "manifest_stage2_final_g3.json"
        if not append_path.is_file():
            extra = sorted(str(path) for path in extra_observed)
            raise AssertionError(f"unregistered Stage1 closure extras: {extra}")
        append = json.loads(append_path.read_text(encoding="utf-8"))
        if append["frozen_stage1_manifest_sha256"] != STAGE1_MANIFEST_SHA256:
            raise AssertionError("append manifest Stage1 lineage changed")
        registered_append = {append_path.resolve()}
        for record in append["outputs"]:
            registered_append.add(_verify_record(record).resolve())
        if extra_observed != registered_append:
            extra = sorted(str(path) for path in extra_observed - registered_append)
            missing_append = sorted(str(path) for path in registered_append - extra_observed)
            raise AssertionError(
                f"append closure changed extra={extra} missing={missing_append}"
            )
    if expected - observed:
        missing = sorted(str(path) for path in expected - observed)
        raise AssertionError(f"Stage1 output closure missing={missing}")
    result_path = out_dir / "stage1_results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["passed_groups"] != [G3]:
        raise AssertionError("Stage1 promoted group set changed")
    lock = json.loads((out_dir / "stage1_promotion_lock.json").read_text(encoding="utf-8"))
    if lock["passed_groups"] != [G3] or lock["stage1_results"]["sha256"] != sha256_file(result_path):
        raise AssertionError("Stage1 promotion lock changed")
    return manifest, result


def _source_snapshot(preregister: Path) -> dict[str, Any]:
    closure = stage1.protocol._static_repo_import_closure((Path(__file__).resolve(),))
    paths = {
        f"repo_source::{path.relative_to(PROJECT_DIR).as_posix()}": path
        for path in closure
    }
    paths.update(
        {
            "focused_test": PROJECT_DIR
            / "tests/test_full_weather_ordinal_bayes_stage2_final.py",
            "preregister": preregister.resolve(),
            "preregister_sidecar": preregister.with_suffix(".sha256").resolve(),
        }
    )
    return {name: shared._snapshot_file(path) for name, path in paths.items()}


def _read_feature_cache(path: Path, expected_index: pd.DatetimeIndex) -> pd.DataFrame:
    frame = pd.read_parquet(path, engine="pyarrow")
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise AssertionError("feature cache index is not unique/increasing")
    if not expected_index.isin(frame.index).all():
        raise AssertionError("feature cache misses required timestamps")
    frame = frame.loc[expected_index]
    if frame.shape != (len(expected_index), 612):
        raise AssertionError(f"feature cache shape changed: {frame.shape}")
    values = frame.to_numpy(dtype=np.float32, copy=False)
    if not np.isfinite(values).all():
        raise AssertionError("feature cache contains non-finite values")
    return frame.astype(np.float32, copy=False)


def _same_delta_candidate(
    primary_baseline: pd.DataFrame,
    recent_baseline: pd.DataFrame,
    raw_action_cf: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    if not primary_baseline.index.equals(recent_baseline.index):
        raise AssertionError("dual baseline indexes differ")
    primary = primary_baseline.copy()
    recent = recent_baseline.copy()
    primary_g3 = transfer_action_kwh(
        primary_baseline[G3], raw_action_cf, capacity_kwh=CAPACITY_KWH[G3]
    )
    delta_cf = (
        primary_g3.to_numpy(dtype=np.float64)
        - primary_baseline[G3].to_numpy(dtype=np.float64)
    ) / CAPACITY_KWH[G3]
    recent_g3_cf = np.clip(
        recent_baseline[G3].to_numpy(dtype=np.float64) / CAPACITY_KWH[G3]
        + delta_cf,
        0.0,
        1.02,
    )
    primary[G3] = primary_g3
    recent[G3] = recent_g3_cf * CAPACITY_KWH[G3]
    for group in TARGET_COLS[:2]:
        if not np.array_equal(
            primary[group].to_numpy(), primary_baseline[group].to_numpy()
        ) or not np.array_equal(
            recent[group].to_numpy(), recent_baseline[group].to_numpy()
        ):
            raise AssertionError(f"{group} identity changed")
    return primary, recent, pd.Series(delta_cf, index=primary.index, name="g3_primary_delta_cf")


def _fit_predict_locked(
    *,
    fit_features: pd.DataFrame,
    fit_actual: pd.Series,
    application_features: pd.DataFrame,
    primary_baseline_g3: pd.Series,
    prefix: Path,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, dict[str, Any], list[Path]]:
    if not fit_features.index.equals(fit_actual.index):
        raise AssertionError("fit feature/label indexes differ")
    if not fit_features.index.max() < application_features.index.min():
        raise AssertionError("fit end is not before application start")
    model = FullWeatherOrdinalBayes().fit(
        fit_features, fit_actual, capacity_kwh=CAPACITY_KWH[G3]
    )
    action, probability, diagnostics = model.predict_action(
        application_features, primary_baseline_g3
    )
    model_path = prefix / "model_g3.joblib"
    probability_path = prefix / "probability42_g3.parquet"
    action_path = prefix / "raw_action_cf_g3.parquet"
    diagnostics_path = prefix / "action_diagnostics_g3.parquet"
    strict._atomic_joblib(model, model_path)
    strict._atomic_parquet(probability, probability_path)
    strict._atomic_parquet(action.to_frame(), action_path)
    strict._atomic_parquet(diagnostics, diagnostics_path)
    reloaded: FullWeatherOrdinalBayes = joblib.load(model_path)
    action2, probability2, diagnostics2 = reloaded.predict_action(
        application_features, primary_baseline_g3
    )
    checks = {
        "action": np.array_equal(action.to_numpy(), action2.to_numpy()),
        "probability": np.array_equal(
            probability.to_numpy(), probability2.to_numpy()
        ),
        "diagnostics": np.array_equal(
            diagnostics.to_numpy(), diagnostics2.to_numpy()
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"model reload differs: {checks}")
    metadata = model.metadata()
    metadata.update(
        {
            "fit_start": fit_features.index.min(),
            "fit_end": fit_features.index.max(),
            "application_start": application_features.index.min(),
            "application_end": application_features.index.max(),
            "fit_end_before_application_start": True,
            "fit_application_overlap_count": 0,
            "model_reload_checks": checks,
            "model_reload_all_outputs_bit_exact": True,
            "probability_shape": list(probability.shape),
            "probability_row_sum_max_abs_error": float(
                np.max(np.abs(probability.sum(axis=1).to_numpy() - 1.0))
            ),
        }
    )
    return action, probability, diagnostics, metadata, [
        model_path,
        probability_path,
        action_path,
        diagnostics_path,
    ]


def _mixed_comparison(
    labels: pd.DataFrame,
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    segments: Mapping[str, pd.DatetimeIndex],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name in REQUIRED_SEGMENTS:
        index = segments[name]
        before = score_details(labels.loc[index, list(TARGET_COLS)], baseline.loc[index, list(TARGET_COLS)]).as_dict()
        after = score_details(labels.loc[index, list(TARGET_COLS)], candidate.loc[index, list(TARGET_COLS)]).as_dict()
        output[name] = {
            "baseline": before,
            "candidate": after,
            "delta": float(after["total_score"] - before["total_score"]),
        }
    return output


def _dual_gate(
    labels: pd.DataFrame,
    baselines: Mapping[str, pd.DataFrame],
    candidates: Mapping[str, pd.DataFrame],
    year: int,
) -> tuple[bool, dict[str, Any]]:
    segments = bounded._year_segments(year)
    comparisons: dict[str, Any] = {}
    passed_all = True
    for name in ("primary_v3", "recent_v4"):
        group = strict._comparison(
            labels.loc[segments["full"], G3],
            baselines[name].loc[segments["full"], G3],
            candidates[name].loc[segments["full"], G3],
            G3,
            {segment: segments[segment] for segment in REQUIRED_SEGMENTS},
        )
        mixed = _mixed_comparison(
            labels, baselines[name], candidates[name], segments
        )
        group_full = group["full"]
        mixed_full = mixed["full"]
        group_n = float(group_full["candidate"]["one_minus_nmae"] - group_full["baseline"]["one_minus_nmae"])
        group_f = float(group_full["candidate"]["ficr"] - group_full["baseline"]["ficr"])
        mixed_n = float(mixed_full["candidate"]["one_minus_nmae"] - mixed_full["baseline"]["one_minus_nmae"])
        mixed_f = float(mixed_full["candidate"]["ficr"] - mixed_full["baseline"]["ficr"])
        gate = {
            "group_all_7_total_score_deltas_strictly_positive": all(
                float(group[segment]["delta"]) > 0.0 for segment in REQUIRED_SEGMENTS
            ),
            "mixed_all_7_total_score_deltas_strictly_positive": all(
                float(mixed[segment]["delta"]) > 0.0 for segment in REQUIRED_SEGMENTS
            ),
            "group_full_delta_one_minus_nmae": group_n,
            "group_full_delta_ficr": group_f,
            "mixed_full_delta_one_minus_nmae": mixed_n,
            "mixed_full_delta_ficr": mixed_f,
            "group_full_components_nonnegative": group_n >= 0.0 and group_f >= 0.0,
            "mixed_full_components_nonnegative": mixed_n >= 0.0 and mixed_f >= 0.0,
        }
        gate["passed"] = bool(all(gate[key] for key in (
            "group_all_7_total_score_deltas_strictly_positive",
            "mixed_all_7_total_score_deltas_strictly_positive",
            "group_full_components_nonnegative",
            "mixed_full_components_nonnegative",
        )))
        passed_all = passed_all and gate["passed"]
        comparisons[name] = {"group_g3": group, "mixed": mixed, "gate": gate}
    return bool(passed_all), comparisons


def _stage2_input_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "labels": args.raw_dir / "train/train_labels.csv",
        "g3_train_cache": args.cache_dir / f"{G3}_weather_train.parquet",
        "stage2_primary": args.artifact_root / "oof/gate2024_locked_v3_cf_fix.parquet",
        "stage2_recent": args.artifact_root / "oof/gate2024_recent_v4_cf_fix_calibration_fit.parquet",
    }
    return {name: shared._snapshot_file(path) for name, path in paths.items()}


def _final_input_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "labels": args.raw_dir / "train/train_labels.csv",
        "sample": args.raw_dir / "sample_submission.csv",
        "g3_train_cache": args.cache_dir / f"{G3}_weather_train.parquet",
        "g3_test_cache": args.cache_dir / f"{G3}_weather_test.parquet",
        "final_primary": args.artifact_root / "final_cf_fix/predictions/corrected_v3_test.parquet",
        "final_recent": args.artifact_root / "final_cf_fix/predictions/corrected_recent_v4_test.parquet",
    }
    return {name: shared._snapshot_file(path) for name, path in paths.items()}


def _stage2(args: argparse.Namespace, source_before: Mapping[str, Any], input_before: Mapping[str, Any]) -> tuple[bool, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    index_2023 = strict._year_index(2023)
    index_2024 = strict._year_index(2024)
    preregister = json.loads(args.preregister.read_text(encoding="utf-8"))
    label_path = args.raw_dir / "train/train_labels.csv"
    fit_labels, fit_evidence = bounded._bounded_label_prefix(
        label_path,
        stage1._label_spec(
            preregister, "labels_stage1_application_after_prescore_lock"
        ),
    )
    train_all_index = pd.date_range(
        "2022-01-01 01:00", "2025-01-01 00:00", freq="h", name="forecast_kst_dtm"
    )
    g3_all_features = _read_feature_cache(
        args.cache_dir / f"{G3}_weather_train.parquet", train_all_index
    )
    primary_baseline = strict._read_prediction(
        args.artifact_root / "oof/gate2024_locked_v3_cf_fix.parquet",
        index_2024,
        required_columns=TARGET_COLS,
    )
    recent_baseline = strict._read_prediction(
        args.artifact_root / "oof/gate2024_recent_v4_cf_fix_calibration_fit.parquet",
        index_2024,
        required_columns=TARGET_COLS,
    )
    action, probability, diagnostics, training, model_outputs = _fit_predict_locked(
        fit_features=g3_all_features.loc[index_2023],
        fit_actual=fit_labels.loc[index_2023, G3],
        application_features=g3_all_features.loc[index_2024],
        primary_baseline_g3=primary_baseline[G3],
        prefix=args.out_dir / "stage2",
    )
    primary_candidate, recent_candidate, delta = _same_delta_candidate(
        primary_baseline, recent_baseline, action
    )
    paths = {
        "primary_baseline": args.out_dir / "stage2/primary_baseline_2024.parquet",
        "recent_baseline": args.out_dir / "stage2/recent_baseline_2024.parquet",
        "primary_candidate": args.out_dir / "stage2/primary_candidate_2024.parquet",
        "recent_candidate": args.out_dir / "stage2/recent_same_delta_candidate_2024.parquet",
        "primary_delta": args.out_dir / "stage2/primary_delta_cf_2024.parquet",
    }
    for key, frame in (
        ("primary_baseline", primary_baseline),
        ("recent_baseline", recent_baseline),
        ("primary_candidate", primary_candidate),
        ("recent_candidate", recent_candidate),
        ("primary_delta", delta.to_frame()),
    ):
        strict._atomic_parquet(frame, paths[key])
    source_after = _source_snapshot(args.preregister)
    input_after = _stage2_input_snapshot(args)
    shared._assert_snapshot_equal(source_before, source_after, name="Stage2 source")
    shared._assert_snapshot_equal(input_before, input_after, name="Stage2 inputs")
    outputs = [*model_outputs, *paths.values()]
    prescore_record = args.out_dir / "stage2_prescore_record.json"
    strict._write_json(
        prescore_record,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "stage1_manifest_sha256": STAGE1_MANIFEST_SHA256,
            "stage1_passed_groups": [G3],
            "fit_label_evidence": fit_evidence,
            "training": training,
            "outputs": [describe_file(path) for path in outputs],
            "primary_delta_sha256": stage1.protocol._array_sha256(delta.to_numpy()),
            "same_primary_delta_transferred_to_recent": True,
            "recent_independent_action_or_gate": False,
            "g1_g2_primary_and_recent_bit_identity": True,
            "application_2024_label_value_cells_materialized": 0,
            "year_2025_or_sample_value_bytes_read": 0,
            "public_or_scale_artifact_bytes_read": 0,
            "source_before": source_before,
            "source_after": source_after,
            "input_before": input_before,
            "input_after": input_after,
        },
    )
    prescore_lock = args.out_dir / "stage2_2024_dual_prescore_lock.json"
    strict._write_json(
        prescore_lock,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "stage1_manifest_sha256": STAGE1_MANIFEST_SHA256,
            "prescore_record": describe_file(prescore_record),
            "candidate_outputs": [describe_file(path) for path in outputs],
            "model_probability_action_primary_delta_and_dual_candidates_frozen": True,
            "created_before_any_2024_application_label_value": True,
            "recent_uses_exact_primary_delta": True,
            "g1_g2_bit_identity": True,
        },
    )
    full_labels = strict._read_full_labels(label_path)
    passed, comparisons = _dual_gate(
        full_labels,
        {"primary_v3": primary_baseline, "recent_v4": recent_baseline},
        {"primary_v3": primary_candidate, "recent_v4": recent_candidate},
        2024,
    )
    result = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "stage2_prescore_lock": describe_file(prescore_lock),
        "executed": True,
        "passed_group": G3,
        "identity_groups": list(TARGET_COLS[:2]),
        "dual_baseline_comparisons": comparisons,
        "comparisons_sha256": strict._canonical_sha256(comparisons),
        "candidate_promoted": passed,
        "no_reselection_retuning_or_rescue": True,
        "year_2025_or_sample_read": False,
        "csv_created": False,
    }
    result_path = args.out_dir / "stage2_results_g3.json"
    strict._write_json(result_path, result)
    lock_path = args.out_dir / "stage2_promotion_lock_g3.json"
    strict._write_json(
        lock_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "stage1_manifest_sha256": STAGE1_MANIFEST_SHA256,
            "stage2_prescore_lock": describe_file(prescore_lock),
            "stage2_results": describe_file(result_path),
            "comparisons_sha256": result["comparisons_sha256"],
            "candidate_promoted": passed,
            "csv_allowed": passed,
            "no_reselection_retuning_or_rescue": True,
        },
    )
    return passed, result, primary_candidate, recent_candidate


def _final(args: argparse.Namespace, source_before: Mapping[str, Any], input_before: Mapping[str, Any]) -> dict[str, Any]:
    label_path = args.raw_dir / "train/train_labels.csv"
    labels = strict._read_full_labels(label_path)
    fit_index = pd.date_range(
        "2023-01-01 01:00", "2025-01-01 00:00", freq="h", name="forecast_kst_dtm"
    )
    train_features = _read_feature_cache(
        args.cache_dir / f"{G3}_weather_train.parquet", labels.index
    ).loc[fit_index]
    sample = pd.read_csv(
        args.raw_dir / "sample_submission.csv",
        encoding="utf-8-sig",
        dtype={"forecast_id": "string", "forecast_kst_dtm": "string"},
    )
    if tuple(sample.columns) != ("forecast_id", "forecast_kst_dtm", *TARGET_COLS):
        raise AssertionError("sample schema changed")
    test_index = pd.DatetimeIndex(
        pd.to_datetime(sample["forecast_kst_dtm"], errors="raise"),
        name="forecast_kst_dtm",
    )
    if len(test_index) != 8760 or not test_index.equals(strict._year_index(2025)):
        raise AssertionError("sample time/index changed")
    test_features = _read_feature_cache(
        args.cache_dir / f"{G3}_weather_test.parquet", test_index
    )
    primary_baseline = strict._read_prediction(
        args.artifact_root / "final_cf_fix/predictions/corrected_v3_test.parquet",
        test_index,
        required_columns=TARGET_COLS,
    )
    recent_baseline = strict._read_prediction(
        args.artifact_root / "final_cf_fix/predictions/corrected_recent_v4_test.parquet",
        test_index,
        required_columns=TARGET_COLS,
    )
    action, probability, diagnostics, training, model_outputs = _fit_predict_locked(
        fit_features=train_features,
        fit_actual=labels.loc[fit_index, G3],
        application_features=test_features,
        primary_baseline_g3=primary_baseline[G3],
        prefix=args.out_dir / "final",
    )
    primary_candidate, recent_candidate, delta = _same_delta_candidate(
        primary_baseline, recent_baseline, action
    )
    paths = {
        "primary_baseline": args.out_dir / "final/primary_baseline_2025.parquet",
        "recent_baseline": args.out_dir / "final/recent_baseline_2025.parquet",
        "primary_candidate": args.out_dir / "final/primary_candidate_2025.parquet",
        "recent_candidate": args.out_dir / "final/recent_same_delta_candidate_2025.parquet",
        "primary_delta": args.out_dir / "final/primary_delta_cf_2025.parquet",
    }
    for key, frame in (
        ("primary_baseline", primary_baseline),
        ("recent_baseline", recent_baseline),
        ("primary_candidate", primary_candidate),
        ("recent_candidate", recent_candidate),
        ("primary_delta", delta.to_frame()),
    ):
        strict._atomic_parquet(frame, paths[key])
    prescore = args.out_dir / "final_2025_prescore_lock.json"
    strict._write_json(
        prescore,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "stage2_promotion_lock": describe_file(
                args.out_dir / "stage2_promotion_lock_g3.json"
            ),
            "training": training,
            "model_outputs": [describe_file(path) for path in model_outputs],
            "candidate_outputs": [describe_file(path) for path in paths.values()],
            "primary_delta_sha256": stage1.protocol._array_sha256(delta.to_numpy()),
            "primary_action_and_delta_derived_once": True,
            "recent_independent_action_or_gate": False,
            "g1_g2_recent_bit_identity": True,
            "candidate_locked_before_csv": True,
        },
    )
    submission = sample.copy()
    for group in TARGET_COLS:
        submission[group] = recent_candidate[group].to_numpy(dtype=np.float64)
    csv_path = args.out_dir / "final/corrected_recent_v4_full_weather_ordinal_g3_2025.csv"
    strict._atomic_csv(submission, csv_path)
    raw = csv_path.read_bytes()
    if raw[:3] != b"\xef\xbb\xbf":
        raise AssertionError("submission lacks UTF-8-SIG BOM")
    readback = pd.read_csv(
        csv_path,
        encoding="utf-8-sig",
        dtype={"forecast_id": "string", "forecast_kst_dtm": "string"},
    )
    if tuple(readback.columns) != tuple(sample.columns) or len(readback) != 8760:
        raise AssertionError("submission schema/rows changed")
    if not readback[["forecast_id", "forecast_kst_dtm"]].equals(
        sample[["forecast_id", "forecast_kst_dtm"]]
    ):
        raise AssertionError("submission id/time text changed")
    expected_round = np.round(
        recent_candidate.loc[:, list(TARGET_COLS)].to_numpy(dtype=np.float64), 6
    )
    if not np.array_equal(
        readback.loc[:, list(TARGET_COLS)].to_numpy(dtype=np.float64), expected_round
    ):
        raise AssertionError("submission six-decimal roundtrip changed")
    if not np.isfinite(expected_round).all():
        raise AssertionError("submission contains non-finite prediction")
    for group in TARGET_COLS:
        values = expected_round[:, TARGET_COLS.index(group)]
        if np.any(values < 0.0) or np.any(values > 1.02 * CAPACITY_KWH[group] + 5e-7):
            raise AssertionError(f"submission bounds changed: {group}")
    source_after = _source_snapshot(args.preregister)
    input_after = _final_input_snapshot(args)
    shared._assert_snapshot_equal(source_before, source_after, name="Final source")
    shared._assert_snapshot_equal(input_before, input_after, name="Final inputs")
    result = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "stage2_promotion_lock": describe_file(
            args.out_dir / "stage2_promotion_lock_g3.json"
        ),
        "final_prescore_lock": describe_file(prescore),
        "executed": True,
        "prediction_parquet": describe_file(paths["recent_candidate"]),
        "submission_csv": describe_file(csv_path),
        "rows": 8760,
        "sample_schema_exact": True,
        "sample_id_time_text_exact": True,
        "utf8_sig_bom": True,
        "finite": True,
        "bounds": True,
        "six_decimal_roundtrip_exact": True,
        "g1_g2_recent_parquet_bit_identity": True,
        "same_primary_delta_transferred_to_recent": True,
        "leaderboard_score_claim": False,
        "private_champion_claim": False,
    }
    result_path = args.out_dir / "final_results_g3.json"
    strict._write_json(result_path, result)
    return result


def _write_manifest(args: argparse.Namespace, source_before: Mapping[str, Any], input_before: Mapping[str, Any], promoted: bool) -> Path:
    stage2_and_final_files = sorted(
        [
            path
            for path in args.out_dir.rglob("*")
            if path.is_file()
            and path.name != "manifest.json"
            and (
                path.parent.name in {"stage2", "final"}
                or path.name.startswith("stage2_")
                or path.name.startswith("final_")
            )
        ],
        key=lambda path: path.relative_to(args.out_dir).as_posix(),
    )
    manifest = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "created_utc": utc_now(),
        "command": [sys.executable, *sys.argv],
        "preregister": describe_file(args.preregister),
        "preregister_sha256": PREREGISTER_SHA256,
        "frozen_stage1_manifest": describe_file(args.out_dir / "manifest.json"),
        "frozen_stage1_manifest_sha256": STAGE1_MANIFEST_SHA256,
        "status": "stage2_pass_final_created" if promoted else "stage2_reject_no_final_csv",
        "source_and_config_snapshot": source_before,
        "input_snapshot": input_before,
        "outputs": [describe_file(path) for path in stage2_and_final_files],
        "output_count": len(stage2_and_final_files),
        "runtime": {"packages": package_versions()},
        "git": git_state(PROJECT_DIR),
        "audit_contracts": {
            "stage1_manifest_unchanged": True,
            "only_g3_refit": True,
            "g1_g2_bit_identity": True,
            "primary_action_delta_derived_once": True,
            "recent_uses_exact_primary_delta": True,
            "dual_group_and_mixed_all_7_strict_gate": True,
            "dual_full_components_nonnegative_gate": True,
            "no_retuning_or_rescue": True,
            "public_or_scale_used": False,
            "final_created_only_if_stage2_pass": True,
        },
    }
    path = args.out_dir / "manifest_stage2_final_g3.json"
    strict._write_json(path, manifest)
    return path


def run(args: argparse.Namespace) -> dict[str, Any]:
    stage1._verify_preregister(
        args.preregister,
        PROJECT_DIR / "artifacts/audits/full_weather_ordinal_bayes_feasibility_v1.json",
    )
    _verify_stage1(args.out_dir)
    for name in (
        "stage2_results_g3.json",
        "stage2_promotion_lock_g3.json",
        "manifest_stage2_final_g3.json",
    ):
        if (args.out_dir / name).exists():
            raise FileExistsError(args.out_dir / name)
    source_before = _source_snapshot(args.preregister)
    stage2_input_before = _stage2_input_snapshot(args)
    promoted, stage2_result, _, _ = _stage2(
        args, source_before, stage2_input_before
    )
    final_result = None
    final_input_before: dict[str, Any] = {}
    if promoted:
        final_input_before = _final_input_snapshot(args)
        final_result = _final(args, source_before, final_input_before)
    manifest_path = _write_manifest(
        args,
        source_before,
        {
            "stage2": stage2_input_before,
            "final_after_stage2_promotion": final_input_before,
        },
        promoted,
    )
    print(f"Stage2 G3 promoted={promoted}", flush=True)
    print(f"manifest sha256={sha256_file(manifest_path)}", flush=True)
    if final_result is not None:
        print(f"CSV={final_result['submission_csv']['path']}", flush=True)
    return {"stage2": stage2_result, "final": final_result}


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    stage1._verify_preregister(
        args.preregister,
        PROJECT_DIR / "artifacts/audits/full_weather_ordinal_bayes_feasibility_v1.json",
    )
    _verify_stage1(args.out_dir)
    if args.preflight_only:
        if stage1.HEAVY_GUARD_PATH.exists():
            current = json.loads(stage1.HEAVY_GUARD_PATH.read_text(encoding="utf-8"))
            if stage1._pid_is_alive(int(current["pid"])):
                raise RuntimeError(f"heavy guard occupied: {current}")
        print("Stage2/final preflight PASS 2024_labels=0 2025=0", flush=True)
        return
    owner = stage1.acquire_heavy_guard(stage1.HEAVY_GUARD_PATH)
    atexit.register(stage1.release_heavy_guard, stage1.HEAVY_GUARD_PATH, owner)
    print(
        f"heavy guard acquired pid={owner['pid']} prereg={PREREGISTER_SHA256} ",
        f"stage1_manifest={STAGE1_MANIFEST_SHA256}",
        flush=True,
    )
    try:
        run(args)
    finally:
        stage1.release_heavy_guard(stage1.HEAVY_GUARD_PATH, owner)


if __name__ == "__main__":
    main()

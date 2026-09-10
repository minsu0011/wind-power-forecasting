"""Run the frozen full-weather baseline-conditioned residual-PMF candidate."""

from __future__ import annotations

import argparse
import atexit
import hashlib
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

from scripts import run_catboost_multiquantile_bayes as bounded  # noqa: E402
from scripts import run_catboost_residual_multiquantile_bayes as residual_q  # noqa: E402
from scripts import run_direct_interval_probability as protocol  # noqa: E402
from scripts import run_ficr_bayes_decision_strict as strict  # noqa: E402
from scripts import run_full_weather_ordinal_bayes as guardmod  # noqa: E402
from scripts import run_shared_q07_multiseed as shared  # noqa: E402
from src.full_weather_residual_pmf import (  # noqa: E402
    MODEL_PARAMETERS,
    STATE_FEATURE_COLUMNS,
    TOTAL_FEATURE_COUNT,
    TRANSFER_WEIGHT,
    FullWeatherResidualPMF,
    build_full_weather_state_features,
    transfer_residual_action_kwh,
)
from src.manifest import (  # noqa: E402
    describe_file,
    git_state,
    package_versions,
    sha256_file,
    utc_now,
)
from src.metric import CAPACITY_KWH, TARGET_COLS, score_details  # noqa: E402
from src.residual_histogram_bayes import (  # noqa: E402
    ACTION_DELTAS_CF,
    COMPONENT_NAMES,
    RESIDUAL_CENTRES_CF,
)


PREREGISTER_SHA256 = "5c091c8dfdc0c9c8b854132eab9e2ba1162813d4157be65506fb0546e93cdbea"
FEASIBILITY_SHA256 = "54515620692cb26825a77bbf72cec24470a9d2a3a826670069e7b7ecdca63f92"
ORDINAL_PREREGISTER_SHA256 = "0052188e1ee6ed1a61cc458e620f357451d2ec6bdb9faf304156f1e185c82f44"
RESIDUAL_SOURCE_PREREGISTER_SHA256 = "60be50a27a6c4ffe46c4e4f3e1e5d61fead040b50aa59d867040031cd5209f7b"
COMPONENT_PREREGISTER_SHA256 = "b92eff04b33fac9f049afda6f8dd62cbd26439f3d2c9b67b4df84a4bd4144564"
ARTIFACT_TYPE = "full_weather_baseline_conditioned_residual_pmf_v1"
REQUIRED_SEGMENTS = ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4")
G3_MAPPING = {
    "lgb_l1": "l1",
    "lgb_q07": "q07",
    "shared_l1": "shared_l1",
    "shared_q07": "shared_q07",
    "top200_q07": "top200q07",
    "energy_q06": "ewq06",
}


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
        default=Path("artifacts/postgate/full_weather_residual_pmf_v1"),
    )
    parser.add_argument(
        "--preregister",
        type=Path,
        default=Path("configs/full_weather_residual_pmf_preregister_v1.json"),
    )
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def _project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_DIR / path


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _verify_fixed_file(path: Path, expected_sha256: str, name: str) -> None:
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise AssertionError(f"{name} changed: {path}")


def _verify_preregister(path: Path) -> dict[str, Any]:
    path = _project_path(path)
    feasibility = PROJECT_DIR / "artifacts/audits/full_weather_residual_pmf_feasibility_v1.json"
    ordinal = PROJECT_DIR / "configs/full_weather_ordinal_bayes_preregister_v1.json"
    residual = PROJECT_DIR / "configs/catboost_residual_multiquantile_bayes_preregister_v3.json"
    components = PROJECT_DIR / "configs/residual_histogram_bayes_preregister.json"
    _verify_fixed_file(path, PREREGISTER_SHA256, "residual-PMF preregister")
    _verify_fixed_file(feasibility, FEASIBILITY_SHA256, "residual-PMF feasibility")
    _verify_fixed_file(ordinal, ORDINAL_PREREGISTER_SHA256, "ordinal I/O preregister")
    _verify_fixed_file(
        residual, RESIDUAL_SOURCE_PREREGISTER_SHA256, "residual source preregister"
    )
    _verify_fixed_file(
        components, COMPONENT_PREREGISTER_SHA256, "component source preregister"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["experiment_id"] != "full_weather_baseline_conditioned_residual_pmf_v1":
        raise AssertionError("experiment id changed")
    if payload["classifier"]["parameters"] != MODEL_PARAMETERS:
        raise AssertionError("model parameters differ from frozen preregister")
    if payload["classifier"]["class_weight"] is not None:
        raise AssertionError("class weight must remain null")
    if payload["target_and_bins"]["class_count"] != len(RESIDUAL_CENTRES_CF):
        raise AssertionError("residual-bin count changed")
    if payload["official_utility_action"]["raw_action_count"] != len(
        ACTION_DELTAS_CF
    ):
        raise AssertionError("action-grid count changed")
    if payload["official_utility_action"]["fixed_transfer_weight"] != TRANSFER_WEIGHT:
        raise AssertionError("transfer changed")
    if payload["feature_contract"]["total_feature_count"] != TOTAL_FEATURE_COUNT:
        raise AssertionError("feature count changed")
    if tuple(payload["feature_contract"]["minimal_state_features_in_fixed_order"]) != STATE_FEATURE_COLUMNS:
        raise AssertionError("state feature contract changed")
    return payload


def _source_snapshot(preregister: Path) -> dict[str, Any]:
    preregister = _project_path(preregister).resolve()
    closure = protocol._static_repo_import_closure((Path(__file__).resolve(),))
    paths = {
        f"repo_source::{path.relative_to(PROJECT_DIR).as_posix()}": path
        for path in closure
    }
    paths.update(
        {
            "focused_core_test": PROJECT_DIR / "tests/test_full_weather_residual_pmf.py",
            "focused_runner_test": PROJECT_DIR / "tests/test_full_weather_residual_pmf_runner.py",
            "preregister": preregister,
            "preregister_sidecar": preregister.with_suffix(".sha256"),
            "feasibility": PROJECT_DIR / "artifacts/audits/full_weather_residual_pmf_feasibility_v1.json",
            "feasibility_sidecar": PROJECT_DIR / "artifacts/audits/full_weather_residual_pmf_feasibility_v1.sha256",
            "ordinal_io_preregister": PROJECT_DIR / "configs/full_weather_ordinal_bayes_preregister_v1.json",
            "residual_source_preregister": PROJECT_DIR / "configs/catboost_residual_multiquantile_bayes_preregister_v3.json",
            "component_source_preregister": PROJECT_DIR / "configs/residual_histogram_bayes_preregister.json",
        }
    )
    return {name: shared._snapshot_file(path) for name, path in sorted(paths.items())}


def _read_feature_cache(path: Path, expected_index: pd.DatetimeIndex) -> pd.DataFrame:
    frame = pd.read_parquet(path, engine="pyarrow")
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise AssertionError(f"weather cache index changed: {path}")
    if not expected_index.isin(frame.index).all():
        raise AssertionError(f"weather cache misses timestamps: {path}")
    frame = frame.loc[expected_index]
    if frame.shape != (len(expected_index), 612):
        raise AssertionError(f"weather cache shape changed: {path} {frame.shape}")
    return frame.astype(np.float32, copy=False)


def _read_stage1_sources(
    artifact_root: Path,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    residual_preregister = json.loads(
        (PROJECT_DIR / "configs/catboost_residual_multiquantile_bayes_preregister_v3.json").read_text(
            encoding="utf-8"
        )
    )
    baseline, components, evidence = residual_q._stage1_sources(
        artifact_root, residual_preregister
    )
    reference_path = (
        artifact_root
        / "postgate/shared_q07_multiseed_strict/oof/stage1_corrected_v3_baseline.parquet"
    )
    reference = pd.read_parquet(reference_path, engine="pyarrow")
    reference.index = pd.DatetimeIndex(reference.index, name="forecast_kst_dtm")
    g3_index = pd.date_range(
        "2023-07-01 01:00", "2024-01-01 00:00", freq="h", name="forecast_kst_dtm"
    )
    if not g3_index.isin(reference.index).all():
        raise AssertionError("G3 corrected-v3 reference coverage changed")
    if not np.array_equal(
        baseline.loc[g3_index, "kpx_group_3"].to_numpy(dtype=np.float64),
        reference.loc[g3_index, "kpx_group_3"].to_numpy(dtype=np.float64),
    ):
        raise AssertionError("G3 corrected-v3 reconstruction differs from exact reference")
    evidence["g3_corrected_v3_reference"] = describe_file(reference_path)
    return baseline, components, evidence


def _read_period_sources(
    artifact_root: Path, key: str, index: pd.DatetimeIndex
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    residual_preregister = json.loads(
        (PROJECT_DIR / "configs/catboost_residual_multiquantile_bayes_preregister_v3.json").read_text(
            encoding="utf-8"
        )
    )
    specs = residual_preregister[key]
    return residual_q._period_sources(specs, index)


def _component_series(
    components: Mapping[str, pd.DataFrame], group: str, index: pd.DatetimeIndex
) -> dict[str, pd.Series]:
    if tuple(components) != COMPONENT_NAMES:
        raise AssertionError("component order changed")
    return {name: components[name].loc[index, group] for name in COMPONENT_NAMES}


def _stage2_input_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    residual_preregister = json.loads(
        (PROJECT_DIR / "configs/catboost_residual_multiquantile_bayes_preregister_v3.json").read_text(
            encoding="utf-8"
        )
    )
    paths: dict[str, Path] = {
        "stage1_baseline_g12": args.artifact_root / "oof/dev2023_locked_v3.parquet",
        "stage1_g3_components": args.artifact_root / "oof/g3dev2023h2_candidates.parquet",
        "stage1_g3_reference": args.artifact_root
        / "postgate/shared_q07_multiseed_strict/oof/stage1_corrected_v3_baseline.parquet",
        "stage2_primary": args.artifact_root / "oof/gate2024_locked_v3_cf_fix.parquet",
        "stage2_recent": args.artifact_root
        / "oof/gate2024_recent_v4_cf_fix_calibration_fit.parquet",
    }
    for key, spec in residual_preregister["stage1_expected_inputs"].items():
        paths[f"stage1::{key}"] = PROJECT_DIR / spec[0]
    for key, spec in residual_preregister[
        "stage2_expected_inputs_after_promotion_only"
    ].items():
        paths[f"stage2::{key}"] = PROJECT_DIR / spec[0]
    for group in TARGET_COLS:
        paths[f"weather_train::{group}"] = args.cache_dir / f"{group}_weather_train.parquet"
    snapshot = {name: shared._snapshot_file(path) for name, path in sorted(paths.items())}
    ordinal_preregister = json.loads(
        (PROJECT_DIR / "configs/full_weather_ordinal_bayes_preregister_v1.json").read_text(
            encoding="utf-8"
        )
    )
    label_spec = guardmod._label_spec(
        ordinal_preregister, "labels_stage1_application_after_prescore_lock"
    )
    snapshot["labels_pre2024_prefix"] = shared._snapshot_csv_prefix(
        args.raw_dir / "train/train_labels.csv",
        data_rows=int(label_spec["data_rows"]),
        prefix_bytes=int(label_spec["bytes"]),
    )
    return snapshot


def _final_input_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    residual_preregister = json.loads(
        (PROJECT_DIR / "configs/catboost_residual_multiquantile_bayes_preregister_v3.json").read_text(
            encoding="utf-8"
        )
    )
    paths: dict[str, Path] = {
        "labels": args.raw_dir / "train/train_labels.csv",
        "sample": args.raw_dir / "sample_submission.csv",
        "final_recent": args.artifact_root
        / "final_cf_fix/predictions/corrected_recent_v4_test.parquet",
    }
    for key, spec in residual_preregister[
        "stage2_expected_inputs_after_promotion_only"
    ].items():
        paths[f"fit2024::{key}"] = PROJECT_DIR / spec[0]
    for key, spec in residual_preregister[
        "final_expected_inputs_after_stage2_only"
    ].items():
        paths[f"apply2025::{key}"] = PROJECT_DIR / spec[0]
    for group in TARGET_COLS:
        paths[f"weather_train::{group}"] = args.cache_dir / f"{group}_weather_train.parquet"
        paths[f"weather_test::{group}"] = args.cache_dir / f"{group}_weather_test.parquet"
    return {name: shared._snapshot_file(path) for name, path in sorted(paths.items())}


def _fit_predict_locked(
    *,
    group: str,
    fit_features: pd.DataFrame,
    fit_actual: pd.Series,
    fit_baseline: pd.Series,
    application_features: pd.DataFrame,
    application_baseline: pd.Series,
    output_dir: Path,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, dict[str, Any], list[Path]]:
    if len(fit_features.index.intersection(application_features.index)):
        raise AssertionError("fit/application overlap")
    if not fit_features.index.max() < application_features.index.min():
        raise AssertionError("fit is not strictly before application")
    print(
        f"fit residual-PMF {group}: {fit_features.index.min()}..{fit_features.index.max()}"
        f" -> {application_features.index.min()}..{application_features.index.max()}",
        flush=True,
    )
    model = FullWeatherResidualPMF().fit(
        fit_features,
        fit_actual,
        fit_baseline,
        capacity_kwh=CAPACITY_KWH[group],
    )
    action, probability, diagnostics = model.predict_action(
        application_features, application_baseline
    )
    model_path = output_dir / f"model_{group}.joblib"
    probability_path = output_dir / f"probability43_{group}.parquet"
    action_path = output_dir / f"raw_action_delta_cf_{group}.parquet"
    diagnostics_path = output_dir / f"action_diagnostics_{group}.parquet"
    strict._atomic_joblib(model, model_path)
    strict._atomic_parquet(probability, probability_path)
    strict._atomic_parquet(action.to_frame(), action_path)
    strict._atomic_parquet(diagnostics, diagnostics_path)
    reloaded: FullWeatherResidualPMF = joblib.load(model_path)
    action2, probability2, diagnostics2 = reloaded.predict_action(
        application_features, application_baseline
    )
    reload_checks = {
        "action_bit_exact": np.array_equal(action.to_numpy(), action2.to_numpy()),
        "probability_bit_exact": np.array_equal(
            probability.to_numpy(), probability2.to_numpy()
        ),
        "diagnostics_bit_exact": np.array_equal(
            diagnostics.to_numpy(), diagnostics2.to_numpy()
        ),
    }
    if not all(reload_checks.values()):
        raise AssertionError(f"model reload changed outputs for {group}: {reload_checks}")
    metadata = model.metadata()
    metadata.update(
        {
            "group": group,
            "fit_start": fit_features.index.min(),
            "fit_end": fit_features.index.max(),
            "application_start": application_features.index.min(),
            "application_end": application_features.index.max(),
            "fit_application_overlap_count": 0,
            "model_reload_checks": reload_checks,
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


def _same_primary_delta_candidates(
    primary_baseline: pd.DataFrame,
    recent_baseline: pd.DataFrame,
    actions: Mapping[str, pd.Series],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not primary_baseline.index.equals(recent_baseline.index):
        raise AssertionError("dual baseline indexes differ")
    primary = primary_baseline.copy()
    recent = recent_baseline.copy()
    delta_cf = pd.DataFrame(index=primary_baseline.index, columns=TARGET_COLS, dtype=float)
    for group in TARGET_COLS:
        candidate = transfer_residual_action_kwh(
            primary_baseline[group], actions[group], capacity_kwh=CAPACITY_KWH[group]
        )
        local_delta = (
            candidate.to_numpy(dtype=np.float64)
            - primary_baseline[group].to_numpy(dtype=np.float64)
        ) / CAPACITY_KWH[group]
        primary[group] = candidate
        recent[group] = np.clip(
            recent_baseline[group].to_numpy(dtype=np.float64) / CAPACITY_KWH[group]
            + local_delta,
            0.0,
            1.02,
        ) * CAPACITY_KWH[group]
        delta_cf[group] = local_delta
        reconstructed = np.clip(
            recent_baseline[group].to_numpy(dtype=np.float64) / CAPACITY_KWH[group]
            + delta_cf[group].to_numpy(dtype=np.float64),
            0.0,
            1.02,
        ) * CAPACITY_KWH[group]
        if not np.array_equal(recent[group].to_numpy(dtype=np.float64), reconstructed):
            raise AssertionError(f"recent exact-primary-delta transfer changed: {group}")
    return primary, recent, delta_cf


def _mixed_comparison(
    labels: pd.DataFrame,
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    segments: Mapping[str, pd.DatetimeIndex],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name in REQUIRED_SEGMENTS:
        index = segments[name]
        before = score_details(
            labels.loc[index, list(TARGET_COLS)],
            baseline.loc[index, list(TARGET_COLS)],
        ).as_dict()
        after = score_details(
            labels.loc[index, list(TARGET_COLS)],
            candidate.loc[index, list(TARGET_COLS)],
        ).as_dict()
        output[name] = {
            "baseline": before,
            "candidate": after,
            "delta": float(after["total_score"] - before["total_score"]),
        }
    return output


def _component_deltas(record: Mapping[str, Any]) -> tuple[float, float]:
    return (
        float(
            record["candidate"]["one_minus_nmae"]
            - record["baseline"]["one_minus_nmae"]
        ),
        float(record["candidate"]["ficr"] - record["baseline"]["ficr"]),
    )


def _dual_gate(
    labels: pd.DataFrame,
    baselines: Mapping[str, pd.DataFrame],
    candidates: Mapping[str, pd.DataFrame],
) -> tuple[bool, dict[str, Any]]:
    segments = bounded._year_segments(2024)
    registered = {name: segments[name] for name in REQUIRED_SEGMENTS}
    comparisons: dict[str, Any] = {}
    dual_pass = True
    for variant in ("primary_v3", "recent_v4"):
        group_comparisons: dict[str, Any] = {}
        group_gates: dict[str, Any] = {}
        for group in TARGET_COLS:
            comparison = strict._comparison(
                labels.loc[segments["full"], group],
                baselines[variant].loc[segments["full"], group],
                candidates[variant].loc[segments["full"], group],
                group,
                registered,
            )
            delta_n, delta_f = _component_deltas(comparison["full"])
            gate = {
                "all_7_total_score_deltas_strictly_positive": all(
                    float(comparison[name]["delta"]) > 0.0
                    for name in REQUIRED_SEGMENTS
                ),
                "full_delta_one_minus_nmae": delta_n,
                "full_delta_ficr": delta_f,
                "full_components_nonnegative": delta_n >= 0.0 and delta_f >= 0.0,
            }
            gate["passed"] = bool(
                gate["all_7_total_score_deltas_strictly_positive"]
                and gate["full_components_nonnegative"]
            )
            group_comparisons[group] = comparison
            group_gates[group] = gate
        mixed = _mixed_comparison(
            labels, baselines[variant], candidates[variant], registered
        )
        mixed_n, mixed_f = _component_deltas(mixed["full"])
        mixed_gate = {
            "all_7_total_score_deltas_strictly_positive": all(
                float(mixed[name]["delta"]) > 0.0 for name in REQUIRED_SEGMENTS
            ),
            "full_delta_one_minus_nmae": mixed_n,
            "full_delta_ficr": mixed_f,
            "full_components_nonnegative": mixed_n >= 0.0 and mixed_f >= 0.0,
        }
        mixed_gate["passed"] = bool(
            mixed_gate["all_7_total_score_deltas_strictly_positive"]
            and mixed_gate["full_components_nonnegative"]
        )
        variant_pass = bool(
            all(group_gates[group]["passed"] for group in TARGET_COLS)
            and mixed_gate["passed"]
        )
        dual_pass = dual_pass and variant_pass
        comparisons[variant] = {
            "groups": group_comparisons,
            "group_gates": group_gates,
            "mixed": mixed,
            "mixed_gate": mixed_gate,
            "passed": variant_pass,
        }
    return bool(dual_pass), comparisons


def _write_stage2_candidates(
    args: argparse.Namespace,
    source_before: Mapping[str, Any],
    input_before: Mapping[str, Any],
) -> tuple[Path, list[Path], dict[str, Any]]:
    index_2023 = strict._year_index(2023)
    index_g3 = pd.date_range(
        "2023-07-01 01:00", "2024-01-01 00:00", freq="h", name="forecast_kst_dtm"
    )
    index_2024 = strict._year_index(2024)
    ordinal_preregister = json.loads(
        (PROJECT_DIR / "configs/full_weather_ordinal_bayes_preregister_v1.json").read_text(
            encoding="utf-8"
        )
    )
    fit_labels, fit_evidence = bounded._bounded_label_prefix(
        args.raw_dir / "train/train_labels.csv",
        guardmod._label_spec(
            ordinal_preregister, "labels_stage1_application_after_prescore_lock"
        ),
    )
    if fit_labels.index.max() != pd.Timestamp("2024-01-01 00:00:00"):
        raise AssertionError("bounded fit labels expose unexpected period")
    primary_fit, components_fit, stage1_source_evidence = _read_stage1_sources(
        args.artifact_root
    )
    primary_2024, components_2024, stage2_source_evidence = _read_period_sources(
        args.artifact_root,
        "stage2_expected_inputs_after_promotion_only",
        index_2024,
    )
    recent_2024 = strict._read_prediction(
        args.artifact_root / "oof/gate2024_recent_v4_cf_fix_calibration_fit.parquet",
        index_2024,
        required_columns=TARGET_COLS,
    )
    actions: dict[str, pd.Series] = {}
    training: dict[str, Any] = {}
    model_outputs: list[Path] = []
    for group in TARGET_COLS:
        fit_index = index_g3 if group == "kpx_group_3" else index_2023
        weather = _read_feature_cache(
            args.cache_dir / f"{group}_weather_train.parquet",
            fit_index.append(index_2024),
        )
        fit_features = build_full_weather_state_features(
            weather.loc[fit_index],
            primary_fit.loc[fit_index, group],
            _component_series(components_fit, group, fit_index),
            capacity_kwh=CAPACITY_KWH[group],
        )
        application_features = build_full_weather_state_features(
            weather.loc[index_2024],
            primary_2024[group],
            _component_series(components_2024, group, index_2024),
            capacity_kwh=CAPACITY_KWH[group],
        )
        action, _probability, _diagnostics, metadata, outputs = _fit_predict_locked(
            group=group,
            fit_features=fit_features,
            fit_actual=fit_labels.loc[fit_index, group],
            fit_baseline=primary_fit.loc[fit_index, group],
            application_features=application_features,
            application_baseline=primary_2024[group],
            output_dir=args.out_dir / "stage2",
        )
        actions[group] = action
        training[group] = metadata
        model_outputs.extend(outputs)
    primary_candidate, recent_candidate, delta_cf = _same_primary_delta_candidates(
        primary_2024, recent_2024, actions
    )
    candidate_paths = {
        "primary_baseline": args.out_dir / "stage2/primary_baseline_2024.parquet",
        "recent_baseline": args.out_dir / "stage2/recent_baseline_2024.parquet",
        "primary_candidate": args.out_dir / "stage2/primary_candidate_2024.parquet",
        "recent_candidate": args.out_dir
        / "stage2/recent_same_primary_delta_candidate_2024.parquet",
        "primary_delta_cf": args.out_dir / "stage2/primary_delta_cf_2024.parquet",
    }
    for name, frame in (
        ("primary_baseline", primary_2024),
        ("recent_baseline", recent_2024),
        ("primary_candidate", primary_candidate),
        ("recent_candidate", recent_candidate),
        ("primary_delta_cf", delta_cf),
    ):
        strict._atomic_parquet(frame, candidate_paths[name])
    outputs = [*model_outputs, *candidate_paths.values()]
    source_after = _source_snapshot(args.preregister)
    input_after = _stage2_input_snapshot(args)
    shared._assert_snapshot_equal(source_before, source_after, name="Stage2 source")
    shared._assert_snapshot_equal(input_before, input_after, name="Stage2 input")
    prescore_record = args.out_dir / "stage2_prescore_record.json"
    strict._write_json(
        prescore_record,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "fit_label_prefix_evidence": fit_evidence,
            "stage1_source_evidence": stage1_source_evidence,
            "stage2_source_evidence": stage2_source_evidence,
            "training": training,
            "outputs": [describe_file(path) for path in outputs],
            "primary_delta_array_sha256": _array_sha256(delta_cf.to_numpy()),
            "primary_pmf_action_and_delta_generation_count": 1,
            "recent_v4_independent_pmf_action_or_gate_count": 0,
            "recent_uses_exact_locked_primary_delta": True,
            "application_2024_label_value_cells_materialized": 0,
            "year_2025_feature_sample_or_prediction_values_read": 0,
            "public_feedback_or_scale_artifact_bytes_read": 0,
            "source_snapshot_before": source_before,
            "source_snapshot_after": source_after,
            "input_snapshot_before": input_before,
            "input_snapshot_after": input_after,
        },
    )
    lock_path = args.out_dir / "stage2_candidate_before_2024_label_lock.json"
    strict._write_json(
        lock_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "prescore_record": describe_file(prescore_record),
            "candidate_outputs": [describe_file(path) for path in outputs],
            "model_probability_reload_action_primary_and_recent_candidates_locked": True,
            "created_before_any_2024_application_label_value": True,
            "same_primary_delta_transferred_to_recent": True,
            "recent_independent_gate": False,
            "all_three_groups_active_no_subset": True,
        },
    )
    print(
        f"candidate-before-2024-label lock sha256={sha256_file(lock_path)} labels2024=0",
        flush=True,
    )
    return lock_path, [*outputs, prescore_record, lock_path], {
        "primary_baseline": primary_2024,
        "recent_baseline": recent_2024,
        "primary_candidate": primary_candidate,
        "recent_candidate": recent_candidate,
    }


def _run_stage2(
    args: argparse.Namespace,
    source_before: Mapping[str, Any],
    input_before: Mapping[str, Any],
) -> tuple[bool, dict[str, Any], list[Path]]:
    lock_path, outputs, payload = _write_stage2_candidates(
        args, source_before, input_before
    )
    labels = strict._read_full_labels(args.raw_dir / "train/train_labels.csv")
    passed, comparisons = _dual_gate(
        labels,
        {
            "primary_v3": payload["primary_baseline"],
            "recent_v4": payload["recent_baseline"],
        },
        {
            "primary_v3": payload["primary_candidate"],
            "recent_v4": payload["recent_candidate"],
        },
    )
    result = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "candidate_before_2024_label_lock": describe_file(lock_path),
        "executed": True,
        "dual_baseline_comparisons": comparisons,
        "comparisons_sha256": strict._canonical_sha256(comparisons),
        "candidate_promoted": passed,
        "whole_candidate_required_all_three_groups": True,
        "no_group_subset_identity_rescue_retuning_or_support_rescue": True,
        "year_2025_or_sample_read": False,
        "csv_created": False,
    }
    result_path = args.out_dir / "stage2_results.json"
    strict._write_json(result_path, result)
    promotion_lock = args.out_dir / "stage2_promotion_lock.json"
    strict._write_json(
        promotion_lock,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "candidate_before_2024_label_lock": describe_file(lock_path),
            "stage2_results": describe_file(result_path),
            "comparisons_sha256": result["comparisons_sha256"],
            "candidate_promoted": passed,
            "csv_allowed": passed,
            "no_retuning_support_rescue_or_group_subset": True,
        },
    )
    return passed, result, [*outputs, result_path, promotion_lock]


def _sample(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    sample = pd.read_csv(
        args.raw_dir / "sample_submission.csv",
        encoding="utf-8-sig",
        dtype={"forecast_id": "string", "forecast_kst_dtm": "string"},
    )
    if tuple(sample.columns) != ("forecast_id", "forecast_kst_dtm", *TARGET_COLS):
        raise AssertionError("sample schema changed")
    index = pd.DatetimeIndex(
        pd.to_datetime(sample["forecast_kst_dtm"], errors="raise"),
        name="forecast_kst_dtm",
    )
    if len(index) != 8760 or not index.equals(strict._year_index(2025)):
        raise AssertionError("sample index changed")
    return sample, index


def _run_final(
    args: argparse.Namespace,
    source_before: Mapping[str, Any],
    input_before: Mapping[str, Any],
) -> tuple[dict[str, Any], list[Path]]:
    index_2024 = strict._year_index(2024)
    labels = strict._read_full_labels(args.raw_dir / "train/train_labels.csv")
    fit_baseline, fit_components, fit_source_evidence = _read_period_sources(
        args.artifact_root,
        "stage2_expected_inputs_after_promotion_only",
        index_2024,
    )
    sample, index_2025 = _sample(args)
    application_baseline, application_components, application_source_evidence = (
        _read_period_sources(
            args.artifact_root,
            "final_expected_inputs_after_stage2_only",
            index_2025,
        )
    )
    recent_2025 = strict._read_prediction(
        args.artifact_root / "final_cf_fix/predictions/corrected_recent_v4_test.parquet",
        index_2025,
        required_columns=TARGET_COLS,
    )
    actions: dict[str, pd.Series] = {}
    training: dict[str, Any] = {}
    model_outputs: list[Path] = []
    for group in TARGET_COLS:
        fit_weather = _read_feature_cache(
            args.cache_dir / f"{group}_weather_train.parquet", index_2024
        )
        application_weather = _read_feature_cache(
            args.cache_dir / f"{group}_weather_test.parquet", index_2025
        )
        fit_features = build_full_weather_state_features(
            fit_weather,
            fit_baseline[group],
            _component_series(fit_components, group, index_2024),
            capacity_kwh=CAPACITY_KWH[group],
        )
        application_features = build_full_weather_state_features(
            application_weather,
            application_baseline[group],
            _component_series(application_components, group, index_2025),
            capacity_kwh=CAPACITY_KWH[group],
        )
        action, _probability, _diagnostics, metadata, outputs = _fit_predict_locked(
            group=group,
            fit_features=fit_features,
            fit_actual=labels.loc[index_2024, group],
            fit_baseline=fit_baseline[group],
            application_features=application_features,
            application_baseline=application_baseline[group],
            output_dir=args.out_dir / "final",
        )
        actions[group] = action
        training[group] = metadata
        model_outputs.extend(outputs)
    primary_candidate, recent_candidate, delta_cf = _same_primary_delta_candidates(
        application_baseline, recent_2025, actions
    )
    candidate_paths = {
        "primary_baseline": args.out_dir / "final/primary_baseline_2025.parquet",
        "recent_baseline": args.out_dir / "final/recent_baseline_2025.parquet",
        "primary_candidate": args.out_dir / "final/primary_candidate_2025.parquet",
        "recent_candidate": args.out_dir
        / "final/recent_same_primary_delta_candidate_2025.parquet",
        "primary_delta_cf": args.out_dir / "final/primary_delta_cf_2025.parquet",
    }
    for name, frame in (
        ("primary_baseline", application_baseline),
        ("recent_baseline", recent_2025),
        ("primary_candidate", primary_candidate),
        ("recent_candidate", recent_candidate),
        ("primary_delta_cf", delta_cf),
    ):
        strict._atomic_parquet(frame, candidate_paths[name])
    prescore_lock = args.out_dir / "final_2025_prescore_lock.json"
    strict._write_json(
        prescore_lock,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "stage2_promotion_lock": describe_file(
                args.out_dir / "stage2_promotion_lock.json"
            ),
            "fit_source_evidence": fit_source_evidence,
            "application_source_evidence": application_source_evidence,
            "training": training,
            "model_outputs": [describe_file(path) for path in model_outputs],
            "candidate_outputs": [
                describe_file(path) for path in candidate_paths.values()
            ],
            "primary_delta_array_sha256": _array_sha256(delta_cf.to_numpy()),
            "primary_pmf_action_and_delta_generation_count": 1,
            "recent_independent_pmf_action_or_gate_count": 0,
            "candidate_locked_before_csv": True,
        },
    )
    submission = sample.copy()
    for group in TARGET_COLS:
        submission[group] = recent_candidate[group].to_numpy(dtype=np.float64)
    csv_path = (
        args.out_dir
        / "final/corrected_recent_v4_full_weather_residual_pmf_2025.csv"
    )
    strict._atomic_csv(submission, csv_path)
    raw = csv_path.read_bytes()
    readback = pd.read_csv(
        csv_path,
        encoding="utf-8-sig",
        dtype={"forecast_id": "string", "forecast_kst_dtm": "string"},
    )
    expected_round = np.round(
        recent_candidate.loc[:, list(TARGET_COLS)].to_numpy(dtype=np.float64), 6
    )
    if raw[:3] != b"\xef\xbb\xbf":
        raise AssertionError("CSV lacks UTF-8-SIG BOM")
    if tuple(readback.columns) != tuple(sample.columns) or len(readback) != 8760:
        raise AssertionError("CSV schema/row count changed")
    if not readback[["forecast_id", "forecast_kst_dtm"]].equals(
        sample[["forecast_id", "forecast_kst_dtm"]]
    ):
        raise AssertionError("CSV id/time text changed")
    if not np.array_equal(
        readback.loc[:, list(TARGET_COLS)].to_numpy(dtype=np.float64), expected_round
    ):
        raise AssertionError("CSV six-decimal roundtrip changed")
    if not np.isfinite(expected_round).all():
        raise AssertionError("CSV predictions are nonfinite")
    for position, group in enumerate(TARGET_COLS):
        values = expected_round[:, position]
        if np.any(values < 0.0) or np.any(
            values > 1.02 * CAPACITY_KWH[group] + 5e-7
        ):
            raise AssertionError(f"CSV bounds changed: {group}")
    source_after = _source_snapshot(args.preregister)
    input_after = _final_input_snapshot(args)
    shared._assert_snapshot_equal(source_before, source_after, name="Final source")
    shared._assert_snapshot_equal(input_before, input_after, name="Final input")
    result = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "stage2_promotion_lock": describe_file(
            args.out_dir / "stage2_promotion_lock.json"
        ),
        "final_prescore_lock": describe_file(prescore_lock),
        "executed": True,
        "prediction_parquet": describe_file(candidate_paths["recent_candidate"]),
        "submission_csv": describe_file(csv_path),
        "rows": 8760,
        "sample_schema_exact": True,
        "sample_id_time_text_exact": True,
        "utf8_sig_bom": True,
        "finite": True,
        "bounds": True,
        "six_decimal_roundtrip_exact": True,
        "same_primary_delta_transferred_to_recent": True,
        "leaderboard_score_claim": False,
        "private_champion_claim": False,
    }
    result_path = args.out_dir / "final_results.json"
    strict._write_json(result_path, result)
    return result, [
        *model_outputs,
        *candidate_paths.values(),
        prescore_lock,
        csv_path,
        result_path,
    ]


def _write_manifest(
    args: argparse.Namespace,
    source_before: Mapping[str, Any],
    input_snapshot: Mapping[str, Any],
    promoted: bool,
) -> Path:
    files = sorted(
        (
            path
            for path in args.out_dir.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        ),
        key=lambda path: path.relative_to(args.out_dir).as_posix(),
    )
    manifest = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "created_utc": utc_now(),
        "command": [sys.executable, *sys.argv],
        "preregister": describe_file(_project_path(args.preregister)),
        "preregister_sha256": PREREGISTER_SHA256,
        "feasibility": describe_file(
            PROJECT_DIR / "artifacts/audits/full_weather_residual_pmf_feasibility_v1.json"
        ),
        "status": "dual_stage2_pass_final_created" if promoted else "dual_stage2_reject_no_2025_read_no_csv",
        "source_and_config_snapshot": source_before,
        "input_snapshot": input_snapshot,
        "outputs": [describe_file(path) for path in files],
        "output_count": len(files),
        "runtime": {"packages": package_versions()},
        "git": git_state(PROJECT_DIR),
        "audit_contracts": {
            "candidate_count": 1,
            "seed": 42,
            "residual_bins": 43,
            "action_count": 61,
            "transfer_weight": 0.10,
            "state_feature_count": 3,
            "candidate_before_2024_label_lock": True,
            "primary_action_delta_derived_once": True,
            "recent_uses_exact_primary_delta": True,
            "dual_all_three_group_and_mixed_all_7_strict_gate": True,
            "dual_group_and_mixed_full_components_nonnegative_gate": True,
            "no_retuning_support_rescue_or_group_subset": True,
            "public_feedback_or_scale_used": False,
            "final_created_only_if_complete_dual_pass": True,
        },
    }
    path = args.out_dir / "manifest.json"
    strict._write_json(path, manifest)
    return path


def run(args: argparse.Namespace) -> dict[str, Any]:
    _verify_preregister(args.preregister)
    if args.out_dir.exists():
        raise FileExistsError(f"fresh output namespace required: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    source_before = _source_snapshot(args.preregister)
    stage2_input_before = _stage2_input_snapshot(args)
    promoted, stage2_result, _stage2_outputs = _run_stage2(
        args, source_before, stage2_input_before
    )
    final_result: dict[str, Any] | None = None
    final_input_before: dict[str, Any] = {}
    if promoted:
        final_input_before = _final_input_snapshot(args)
        final_result, _final_outputs = _run_final(
            args, source_before, final_input_before
        )
    manifest_path = _write_manifest(
        args,
        source_before,
        {
            "stage2_before_any_2024_label_value": stage2_input_before,
            "final_only_after_complete_dual_promotion": final_input_before,
        },
        promoted,
    )
    print(f"dual Stage2 promoted={promoted}", flush=True)
    print(f"final manifest={manifest_path} sha256={sha256_file(manifest_path)}", flush=True)
    return {"stage2": stage2_result, "final": final_result}


def _guard_state() -> dict[str, Any] | None:
    if not guardmod.HEAVY_GUARD_PATH.exists():
        return None
    current = json.loads(guardmod.HEAVY_GUARD_PATH.read_text(encoding="utf-8"))
    if guardmod._pid_is_alive(int(current["pid"])):
        return current
    return None


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _verify_preregister(args.preregister)
    if args.preflight_only:
        occupied = _guard_state()
        if occupied is not None:
            raise RuntimeError(f"heavy guard occupied: {occupied}")
        output_state = "absent" if not args.out_dir.exists() else "existing_postrun"
        print(
            "residual-PMF preflight PASS fit=0 score=0 labels2024=0 "
            f"output={output_state} prereg={PREREGISTER_SHA256}",
            flush=True,
        )
        return
    owner = guardmod.acquire_heavy_guard(guardmod.HEAVY_GUARD_PATH)
    atexit.register(guardmod.release_heavy_guard, guardmod.HEAVY_GUARD_PATH, owner)
    print(
        f"heavy guard acquired pid={owner['pid']} prereg={PREREGISTER_SHA256}",
        flush=True,
    )
    try:
        run(args)
    finally:
        guardmod.release_heavy_guard(guardmod.HEAVY_GUARD_PATH, owner)


if __name__ == "__main__":
    main()

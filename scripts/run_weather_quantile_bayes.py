"""Strict-forward direct-weather quantile Bayes experiment for BARAM."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts import run_ficr_bayes_decision_strict as bayes  # noqa: E402
from scripts import run_shared_q07_multiseed as shared_strict  # noqa: E402
from src.manifest import (  # noqa: E402
    describe_file,
    git_state,
    package_versions,
    sha256_file,
    utc_now,
)
from src.metric import CAPACITY_KWH, TARGET_COLS, score_details  # noqa: E402
from src.weather_quantile import (  # noqa: E402
    BayesActionConfig,
    WeatherQuantileSurface,
    blend_with_baseline_kwh,
)


PREREGISTER_SHA256 = "c674d7b30f6edee5dbb29abf2232e665400330bd6b7f0b2be9bdccef064983c8"
QUANTILE_LEVELS = (0.10, 0.25, 0.50, 0.75, 0.90)
BLEND_WEIGHTS = (0.05, 0.10, 0.20)
WEIGHT_KEYS = ("w05", "w10", "w20")
WEIGHT_BY_KEY = dict(zip(WEIGHT_KEYS, BLEND_WEIGHTS))
STAGE1_REQUIRED: dict[str, tuple[str, ...]] = {
    "kpx_group_1": ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4"),
    "kpx_group_2": ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4"),
    "kpx_group_3": ("full", "Q3", "Q4"),
}
STAGE2_REQUIRED = ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("stage1", "stage2", "final"), required=True
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=Path(r"data/local/open")
    )
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/cache"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/postgate/weather_quantile_bayes"),
    )
    parser.add_argument(
        "--preregister",
        type=Path,
        default=Path("configs/weather_quantile_bayes_preregister.json"),
    )
    return parser.parse_args(argv)


def _verify_preregister(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], BayesActionConfig]:
    observed = sha256_file(path)
    if observed != PREREGISTER_SHA256:
        raise AssertionError(
            f"preregister hash changed: {observed} != {PREREGISTER_SHA256}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    surface = payload["quantile_surface"]
    if tuple(map(float, surface["quantile_levels"])) != QUANTILE_LEVELS:
        raise AssertionError("quantile levels changed")
    candidates = payload["candidate_family"]
    if int(candidates["candidate_count"]) != 3 or tuple(
        map(float, candidates["fixed_blend_weights"])
    ) != BLEND_WEIGHTS:
        raise AssertionError("fixed blend candidate family changed")
    action = payload["bayes_action"]
    grid = action["candidate_grid"]
    action_config = BayesActionConfig(
        quantile_levels=QUANTILE_LEVELS,
        interpolation_count=33,
        outcome_lower_cf=float(action["outcome_sample_clip_cf"][0]),
        outcome_upper_cf=float(action["outcome_sample_clip_cf"][1]),
        candidate_lower_cf=float(grid["absolute_cf_start"]),
        candidate_upper_cf=float(grid["absolute_cf_end"]),
        candidate_step_cf=float(grid["absolute_cf_step"]),
    )
    model_spec = {
        "quantile_levels": QUANTILE_LEVELS,
        "model_parameters": dict(surface["parameters"]),
        "random_state": int(surface["seed"]),
        "n_jobs": int(surface["n_jobs"]),
        "minimum_actual_cf": 0.10,
    }
    return payload, model_spec, action_config


def _surface(
    model_spec: Mapping[str, Any], action_config: BayesActionConfig
) -> WeatherQuantileSurface:
    return WeatherQuantileSurface(
        quantile_levels=model_spec["quantile_levels"],
        model_parameters=model_spec["model_parameters"],
        random_state=model_spec["random_state"],
        n_jobs=model_spec["n_jobs"],
        minimum_actual_cf=model_spec["minimum_actual_cf"],
        action_config=action_config,
    )


def _year_segments(year: int) -> dict[str, pd.DatetimeIndex]:
    start = pd.Timestamp(f"{year}-01-01 01:00:00")
    q2 = pd.Timestamp(f"{year}-04-01 01:00:00")
    h2 = pd.Timestamp(f"{year}-07-01 01:00:00")
    q4 = pd.Timestamp(f"{year}-10-01 01:00:00")
    end = pd.Timestamp(f"{year + 1}-01-01 00:00:00")
    hour = pd.Timedelta(hours=1)
    return {
        "full": bayes._interval(start, end),
        "H1": bayes._interval(start, h2 - hour),
        "H2": bayes._interval(h2, end),
        "Q1": bayes._interval(start, q2 - hour),
        "Q2": bayes._interval(q2, h2 - hour),
        "Q3": bayes._interval(h2, q4 - hour),
        "Q4": bayes._interval(q4, end),
    }


def _stage1_input_snapshot(raw_dir: Path, artifact_root: Path) -> dict[str, Any]:
    label_path = raw_dir / "train" / "train_labels.csv"
    return {
        "labels_prefix": shared_strict._snapshot_csv_prefix(
            label_path,
            data_rows=shared_strict.EXPECTED_ROWS_PRE2024,
            prefix_bytes=shared_strict.OFFICIAL_STAGE1_PREFIX_BYTES["labels"],
        ),
        "ldaps_prefix": shared_strict._snapshot_csv_prefix(
            raw_dir / "train" / "ldaps_train.csv",
            data_rows=shared_strict.EXPECTED_STAGE1_RAW_ROWS["ldaps"],
            prefix_bytes=shared_strict.OFFICIAL_STAGE1_PREFIX_BYTES["ldaps"],
        ),
        "gfs_prefix": shared_strict._snapshot_csv_prefix(
            raw_dir / "train" / "gfs_train.csv",
            data_rows=shared_strict.EXPECTED_STAGE1_RAW_ROWS["gfs"],
            prefix_bytes=shared_strict.OFFICIAL_STAGE1_PREFIX_BYTES["gfs"],
        ),
        "info_workbook": shared_strict._snapshot_file(raw_dir / "info.xlsx"),
        "baseline_2023_g12": shared_strict._snapshot_file(
            artifact_root / "oof" / "dev2023_locked_v3.parquet"
        ),
        "baseline_2023_g3_source": shared_strict._snapshot_file(
            artifact_root / "oof" / "g3dev2023h2_candidates.parquet"
        ),
    }


def _provenance_paths(preregister_path: Path) -> dict[str, Path]:
    return {
        "runner": Path(__file__).resolve(),
        "weather_quantile": PROJECT_DIR / "src" / "weather_quantile.py",
        "features": PROJECT_DIR / "src" / "features.py",
        "metric": PROJECT_DIR / "src" / "metric.py",
        "manifest": PROJECT_DIR / "src" / "manifest.py",
        "bounded_weather_helper": PROJECT_DIR / "scripts" / "run_shared_q07_multiseed.py",
        "strict_artifact_helper": PROJECT_DIR
        / "scripts"
        / "run_ficr_bayes_decision_strict.py",
        "test": PROJECT_DIR / "tests" / "test_weather_quantile_bayes.py",
        "preregister": preregister_path.resolve(),
    }


def _snapshot_named(paths: Mapping[str, Path]) -> dict[str, Any]:
    return {name: shared_strict._snapshot_file(path) for name, path in paths.items()}


def _load_stage1_baseline(
    artifact_root: Path,
) -> tuple[pd.DataFrame, dict[str, Path]]:
    index = bayes._year_index(2023)
    g12_path = artifact_root / "oof" / "dev2023_locked_v3.parquet"
    g12 = bayes._read_prediction(
        g12_path, index, required_columns=TARGET_COLS[:2]
    )
    h2 = _year_segments(2023)["H2"]
    g3_path = artifact_root / "oof" / "g3dev2023h2_candidates.parquet"
    raw = bayes._read_prediction(
        g3_path,
        h2,
        required_columns=(
            "q07",
            "shared_l1",
            "shared_q07",
            "top200q07",
            "ewq06",
        ),
        exact_columns=False,
    )
    weighted = (
        0.20 * raw["q07"]
        + 0.075 * raw["shared_l1"]
        + 0.425 * raw["shared_q07"]
        + 0.025 * raw["top200q07"]
        + 0.275 * raw["ewq06"]
    )
    baseline = pd.DataFrame(np.nan, index=index, columns=TARGET_COLS)
    baseline.loc[:, list(TARGET_COLS[:2])] = g12
    baseline.loc[h2, "kpx_group_3"] = np.clip(
        1.25 * weighted.to_numpy(dtype=float) - 1200.0,
        0.0,
        1.02 * CAPACITY_KWH["kpx_group_3"],
    )
    return baseline, {"baseline_2023_g12": g12_path, "baseline_2023_g3": g3_path}


def _fit_actions(
    *,
    features: Mapping[str, pd.DataFrame],
    labels: pd.DataFrame,
    baseline: pd.DataFrame,
    train_indexes: Mapping[str, pd.DatetimeIndex],
    application_indexes: Mapping[str, pd.DatetimeIndex],
    model_spec: Mapping[str, Any],
    action_config: BayesActionConfig,
) -> tuple[
    dict[str, WeatherQuantileSurface],
    pd.DataFrame,
    dict[str, pd.DataFrame],
    dict[str, Any],
]:
    actions = pd.DataFrame(np.nan, index=baseline.index, columns=TARGET_COLS)
    quantiles: dict[str, pd.DataFrame] = {}
    models: dict[str, WeatherQuantileSurface] = {}
    metadata: dict[str, Any] = {}
    for group in TARGET_COLS:
        train_index = train_indexes[group]
        application_index = application_indexes[group]
        if len(train_index.intersection(application_index)) or not (
            train_index.max() < application_index.min()
        ):
            raise AssertionError(f"{group} fit/application is not strict-forward")
        print(
            f"fit direct weather quantiles {group}: {train_index.min()} -> {application_index.min()}",
            flush=True,
        )
        model = _surface(model_spec, action_config).fit(
            features[group].loc[train_index],
            labels.loc[train_index, group],
            capacity_kwh=CAPACITY_KWH[group],
        )
        action, group_quantiles = model.predict_action(
            features[group].loc[application_index],
            baseline.loc[application_index, group],
        )
        actions.loc[application_index, group] = action
        quantiles[group] = group_quantiles
        models[group] = model
        metadata[group] = model.metadata()
        metadata[group].update(
            {
                "train_start": train_index.min(),
                "train_end": train_index.max(),
                "application_start": application_index.min(),
                "application_end": application_index.max(),
                "fit_end_before_application_start": True,
                "fit_application_overlap_count": 0,
            }
        )
    return models, actions, quantiles, metadata


def _blend_frame(
    baseline: pd.DataFrame,
    action: pd.DataFrame,
    application_indexes: Mapping[str, pd.DatetimeIndex],
    weight: float,
) -> pd.DataFrame:
    result = baseline.copy()
    for group in TARGET_COLS:
        index = application_indexes[group]
        result.loc[index, group] = blend_with_baseline_kwh(
            baseline.loc[index, group],
            action.loc[index, group],
            weight=weight,
            capacity_kwh=CAPACITY_KWH[group],
        )
    return result


def _validate_stage1_comparisons(comparisons: Mapping[str, Any]) -> None:
    if tuple(comparisons) != WEIGHT_KEYS:
        raise AssertionError("Stage1 blend candidate keys/order changed")
    for key in WEIGHT_KEYS:
        if set(comparisons[key]) != set(TARGET_COLS):
            raise AssertionError(f"{key} group set changed")
        for group in TARGET_COLS:
            records = comparisons[key][group]
            if set(records) != set(STAGE1_REQUIRED[group]):
                raise AssertionError(f"{key}/{group} segment set changed")
            for segment in STAGE1_REQUIRED[group]:
                record = records[segment]
                expected = float(record["candidate"]["score"]) - float(
                    record["baseline"]["score"]
                )
                if float(record["delta"]) != expected:
                    raise AssertionError(f"{key}/{group}/{segment} delta changed")


def _select_weight(
    comparisons: Mapping[str, Any]
) -> tuple[float | None, dict[str, Any]]:
    _validate_stage1_comparisons(comparisons)
    diagnostics: dict[str, Any] = {}
    eligible: list[str] = []
    for key in WEIGHT_KEYS:
        deltas = {
            f"{group}/{segment}": float(comparisons[key][group][segment]["delta"])
            for group in TARGET_COLS
            for segment in STAGE1_REQUIRED[group]
        }
        diagnostics[key] = {
            "weight": WEIGHT_BY_KEY[key],
            "deltas": deltas,
            "minimum": min(deltas.values()),
            "mean": float(np.mean(list(deltas.values()))),
            "all_strictly_positive": all(value > 0.0 for value in deltas.values()),
        }
        if diagnostics[key]["all_strictly_positive"]:
            eligible.append(key)
    if not eligible:
        return None, {"candidates": diagnostics, "selected": "identity"}
    selected_key = max(
        eligible,
        key=lambda key: (
            diagnostics[key]["minimum"],
            diagnostics[key]["mean"],
            -diagnostics[key]["weight"],
        ),
    )
    return WEIGHT_BY_KEY[selected_key], {
        "candidates": diagnostics,
        "selected": selected_key,
    }


def _stage1(
    *,
    raw_dir: Path,
    artifact_root: Path,
    cache_dir: Path,
    out_dir: Path,
    preregister_path: Path,
    preregister: Mapping[str, Any],
    model_spec: Mapping[str, Any],
    action_config: BayesActionConfig,
) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"Stage1 requires a new output directory: {out_dir}")
    out_dir.mkdir(parents=True)
    bayes._copy_exclusive(preregister_path, out_dir / "preregister.json")
    provenance_before = _snapshot_named(_provenance_paths(preregister_path))
    input_before = _stage1_input_snapshot(raw_dir, artifact_root)

    labels, label_evidence = bayes._read_bounded_labels(
        raw_dir / "train" / "train_labels.csv"
    )
    features, raw_feature_contract = shared_strict._read_stage1_raw_features(
        raw_dir, labels
    )
    baseline, _ = _load_stage1_baseline(artifact_root)
    year_2022 = bayes._year_index(2022)
    segments_2023 = _year_segments(2023)
    h1_2023 = segments_2023["H1"]
    h2_2023 = segments_2023["H2"]
    train_indexes = {
        "kpx_group_1": year_2022,
        "kpx_group_2": year_2022,
        "kpx_group_3": h1_2023,
    }
    application_indexes = {
        "kpx_group_1": segments_2023["full"],
        "kpx_group_2": segments_2023["full"],
        "kpx_group_3": h2_2023,
    }
    models, actions, quantiles, training = _fit_actions(
        features=features,
        labels=labels,
        baseline=baseline,
        train_indexes=train_indexes,
        application_indexes=application_indexes,
        model_spec=model_spec,
        action_config=action_config,
    )
    zero_blend = _blend_frame(baseline, actions, application_indexes, 0.0)
    zero_weight_exact = all(
        np.ascontiguousarray(zero_blend[group].to_numpy()).tobytes()
        == np.ascontiguousarray(baseline[group].to_numpy()).tobytes()
        for group in TARGET_COLS
    )
    if not zero_weight_exact:
        raise AssertionError("zero-weight baseline reconstruction is not bit exact")

    blends = {
        key: _blend_frame(baseline, actions, application_indexes, weight)
        for key, weight in WEIGHT_BY_KEY.items()
    }
    comparisons: dict[str, Any] = {}
    for key in WEIGHT_KEYS:
        comparisons[key] = {}
        for group in TARGET_COLS:
            app_index = application_indexes[group]
            if group in TARGET_COLS[:2]:
                segment_indexes = {
                    name: segments_2023[name] for name in STAGE1_REQUIRED[group]
                }
            else:
                segment_indexes = {
                    "full": h2_2023,
                    "Q3": segments_2023["Q3"],
                    "Q4": segments_2023["Q4"],
                }
            comparisons[key][group] = bayes._comparison(
                labels.loc[app_index, group],
                baseline.loc[app_index, group],
                blends[key].loc[app_index, group],
                group,
                segment_indexes,
            )
    locked_weight, selection = _select_weight(comparisons)

    output_paths: list[Path] = []
    baseline_path = out_dir / "oof" / "stage1_baseline_2023.parquet"
    action_path = out_dir / "oof" / "stage1_weather_bayes_action_2023.parquet"
    model_path = out_dir / "models" / "stage1_weather_quantile_models.joblib"
    bayes._atomic_parquet(baseline, baseline_path)
    bayes._atomic_parquet(actions, action_path)
    bayes._atomic_joblib(models, model_path)
    output_paths.extend([baseline_path, action_path, model_path])
    for group, frame in quantiles.items():
        path = out_dir / "oof" / f"stage1_{group}_quantiles.parquet"
        bayes._atomic_parquet(frame, path)
        output_paths.append(path)
    for key, frame in blends.items():
        path = out_dir / "oof" / f"stage1_blend_{key}_2023.parquet"
        bayes._atomic_parquet(frame, path)
        output_paths.append(path)

    provenance_after = _snapshot_named(_provenance_paths(preregister_path))
    input_after = _stage1_input_snapshot(raw_dir, artifact_root)
    shared_strict._assert_snapshot_equal(
        provenance_before, provenance_after, name="Stage1 provenance"
    )
    shared_strict._assert_snapshot_equal(
        input_before, input_after, name="Stage1 bounded inputs"
    )
    result = {
        "schema_version": 1,
        "experiment_id": preregister["experiment_id"],
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "candidate_count": len(BLEND_WEIGHTS),
        "candidate_weights": BLEND_WEIGHTS,
        "model_spec": model_spec,
        "action_config": asdict(action_config),
        "2024_read": False,
        "multi_year_cache_values_read_prelock": False,
        "cache_dir_deliberately_excluded_prelock": str(cache_dir.resolve()),
        "label_prefix_evidence": label_evidence,
        "raw_feature_contract": raw_feature_contract,
        "training": training,
        "fit_application_overlap_count": 0,
        "same_row_fit_score_reported": False,
        "zero_weight_baseline_value_bits_exact": zero_weight_exact,
        "comparisons": comparisons,
        "comparisons_sha256": bayes._canonical_sha256(comparisons),
        "selection": selection,
        "locked_weight": locked_weight,
        "locked_candidate": "identity" if locked_weight is None else f"blend_{locked_weight:.2f}",
        "selection_rule": preregister["stage1_pre2024_selection"]["selection_rule"],
        "provenance_before": provenance_before,
        "provenance_after": provenance_after,
        "provenance_snapshot_sha256": bayes._canonical_sha256(provenance_before),
        "stage1_input_snapshot_before": input_before,
        "stage1_input_snapshot_after": input_after,
        "stage1_input_snapshot_sha256": bayes._canonical_sha256(input_before),
        "snapshots_unchanged": True,
        "outputs_before_result": [describe_file(path) for path in output_paths],
        "leaderboard_score_claim": False,
        "untouched_gate_claim": False,
    }
    result_path = out_dir / "stage1_results.json"
    bayes._write_json(result_path, result)
    lock = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "stage1_results_sha256": sha256_file(result_path),
        "comparisons_sha256": result["comparisons_sha256"],
        "locked_weight": locked_weight,
        "locked_candidate": result["locked_candidate"],
        "selection_sha256": bayes._canonical_sha256(selection),
        "provenance_snapshot_sha256": result["provenance_snapshot_sha256"],
        "stage1_input_snapshot_sha256": result["stage1_input_snapshot_sha256"],
        "candidate_model_action_and_weight_frozen": True,
        "no_2024_reselection_or_retuning": True,
    }
    bayes._write_json(out_dir / "stage1_candidate_lock.json", lock)
    print(f"Stage1 locked candidate: {result['locked_candidate']}", flush=True)
    return lock


def _load_stage1_lock(out_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    result_path = out_dir / "stage1_results.json"
    lock_path = out_dir / "stage1_candidate_lock.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["preregister_sha256"] != PREREGISTER_SHA256:
        raise AssertionError("Stage1 lock preregister hash changed")
    if lock["stage1_results_sha256"] != sha256_file(result_path):
        raise AssertionError("Stage1 results changed after candidate lock")
    if not lock.get("candidate_model_action_and_weight_frozen") or not lock.get(
        "no_2024_reselection_or_retuning"
    ):
        raise AssertionError("Stage1 candidate is not frozen")
    if tuple(map(float, result["candidate_weights"])) != BLEND_WEIGHTS or int(
        result["candidate_count"]
    ) != len(BLEND_WEIGHTS):
        raise AssertionError("Stage1 candidate family changed")
    comparisons = result["comparisons"]
    if bayes._canonical_sha256(comparisons) != result["comparisons_sha256"]:
        raise AssertionError("Stage1 comparison digest changed")
    if result["comparisons_sha256"] != lock["comparisons_sha256"]:
        raise AssertionError("Stage1 lock comparison digest changed")
    selected, selection = _select_weight(comparisons)
    if selected != result["locked_weight"] or selected != lock["locked_weight"]:
        raise AssertionError("locked weight differs from independently recomputed rule")
    if selection != result["selection"] or bayes._canonical_sha256(selection) != lock[
        "selection_sha256"
    ]:
        raise AssertionError("Stage1 selection diagnostics changed")
    for key, digest_key in (
        ("provenance_before", "provenance_snapshot_sha256"),
        ("stage1_input_snapshot_before", "stage1_input_snapshot_sha256"),
    ):
        snapshot = result[key]
        if bayes._canonical_sha256(snapshot) != result[digest_key] or result[
            digest_key
        ] != lock[digest_key]:
            raise AssertionError(f"Stage1 {key} digest changed")
        shared_strict._assert_snapshot_equal(
            snapshot,
            shared_strict._refresh_snapshot(snapshot),
            name=f"current {key}",
        )
    return lock, result


def _postlock_cache_audit(
    *, raw_dir: Path, cache_dir: Path, out_dir: Path
) -> dict[str, Any]:
    lock, result = _load_stage1_lock(out_dir)
    path = out_dir / "postlock_cache_audit.json"
    if path.exists():
        raise FileExistsError(path)
    if lock["locked_weight"] is None:
        audit = {
            "schema_version": 1,
            "created_utc": utc_now(),
            "stage1_candidate_lock_sha256": sha256_file(
                out_dir / "stage1_candidate_lock.json"
            ),
            "locked_weight": None,
            "performed": False,
            "2024_cache_values_read": False,
            "reason": (
                "identity was locked; preregistered no-2024-read rule takes "
                "precedence and the multi-year cache remains unopened"
            ),
            "cache_was_not_a_stage1_model_or_selection_input": True,
            "checks": {},
        }
        bayes._write_json(path, audit)
        return audit
    labels, _ = bayes._read_bounded_labels(raw_dir / "train" / "train_labels.csv")
    rebuilt, _ = shared_strict._read_stage1_raw_features(raw_dir, labels)
    checks: dict[str, Any] = {}
    for group in TARGET_COLS:
        cache_path = cache_dir / f"{group}_weather_train.parquet"
        cached = pd.read_parquet(cache_path, engine="pyarrow").iloc[
            : shared_strict.EXPECTED_ROWS_PRE2024
        ]
        cached.index = pd.DatetimeIndex(cached.index, name="forecast_kst_dtm")
        observed = rebuilt[group]
        index_exact = observed.index.equals(cached.index)
        columns_exact = tuple(observed.columns) == tuple(cached.columns)
        dtypes_exact = tuple(map(str, observed.dtypes)) == tuple(map(str, cached.dtypes))
        value_bits_exact = (
            np.ascontiguousarray(observed.to_numpy(copy=False)).tobytes()
            == np.ascontiguousarray(cached.to_numpy(copy=False)).tobytes()
        )
        if not all((index_exact, columns_exact, dtypes_exact, value_bits_exact)):
            raise AssertionError(f"{group} bounded raw/cache prefix differs")
        checks[group] = {
            "selection_input": False,
            "read_only_after_candidate_lock": True,
            "rows": len(observed),
            "columns": observed.shape[1],
            "index_exact": index_exact,
            "column_order_exact": columns_exact,
            "dtype_exact": dtypes_exact,
            "value_bits_exact": value_bits_exact,
            "frame_sha256": shared_strict._frame_sha256(observed),
            "cache_file": describe_file(cache_path),
        }
    audit = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "stage1_candidate_lock_sha256": sha256_file(
            out_dir / "stage1_candidate_lock.json"
        ),
        "locked_weight": lock["locked_weight"],
        "performed": True,
        "2024_cache_values_read": True,
        "selection_was_immutable_before_cache_read": True,
        "cache_was_not_a_stage1_model_or_selection_input": True,
        "checks": checks,
        "all_group_feature_prefixes_value_bit_exact": True,
        "raw_feature_hashes": result["raw_feature_contract"]["built_features"],
    }
    bayes._write_json(path, audit)
    return audit


def _stage2_input_snapshot(
    raw_dir: Path, cache_dir: Path, artifact_root: Path
) -> dict[str, Any]:
    return {
        "labels_full": shared_strict._snapshot_file(
            raw_dir / "train" / "train_labels.csv"
        ),
        **{
            f"cache_train_{group}": shared_strict._snapshot_file(
                cache_dir / f"{group}_weather_train.parquet"
            )
            for group in TARGET_COLS
        },
        "baseline_2024": shared_strict._snapshot_file(
            artifact_root / "oof" / "gate2024_locked_v3_cf_fix.parquet"
        ),
    }


def _stage2_promotion(
    comparisons: Mapping[str, Any]
) -> tuple[bool, dict[str, Any]]:
    if set(comparisons) != set(TARGET_COLS):
        raise AssertionError("Stage2 group set changed")
    audit: dict[str, Any] = {}
    for group in TARGET_COLS:
        records = comparisons[group]
        if set(records) != set(STAGE2_REQUIRED):
            raise AssertionError(f"Stage2 {group} segment set changed")
        deltas: dict[str, float] = {}
        for segment in STAGE2_REQUIRED:
            record = records[segment]
            expected = float(record["candidate"]["score"]) - float(
                record["baseline"]["score"]
            )
            if float(record["delta"]) != expected:
                raise AssertionError(f"Stage2 {group}/{segment} delta changed")
            deltas[segment] = expected
        audit[group] = {
            "deltas": deltas,
            "all_strictly_positive": all(value > 0.0 for value in deltas.values()),
        }
    promoted = all(record["all_strictly_positive"] for record in audit.values())
    return promoted, audit


def _write_stage2_lock(out_dir: Path, result: Mapping[str, Any]) -> dict[str, Any]:
    result_path = out_dir / "stage2_results.json"
    bayes._write_json(result_path, result)
    lock = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "stage1_candidate_lock_sha256": sha256_file(
            out_dir / "stage1_candidate_lock.json"
        ),
        "stage2_results_sha256": sha256_file(result_path),
        "locked_weight": result["locked_weight"],
        "candidate_promoted": bool(result["candidate_promoted"]),
        "promotion_audit_sha256": bayes._canonical_sha256(
            result["promotion_audit"]
        ),
        "no_2024_reselection_or_retuning": True,
        "csv_allowed": bool(result["candidate_promoted"]),
    }
    bayes._write_json(out_dir / "stage2_promotion_lock.json", lock)
    return lock


def _stage2(
    *,
    raw_dir: Path,
    artifact_root: Path,
    cache_dir: Path,
    out_dir: Path,
    model_spec: Mapping[str, Any],
    action_config: BayesActionConfig,
) -> dict[str, Any]:
    lock, _ = _load_stage1_lock(out_dir)
    if not (out_dir / "postlock_cache_audit.json").is_file():
        raise AssertionError("post-lock cache audit is required before Stage2")
    if (out_dir / "stage2_results.json").exists() or (
        out_dir / "stage2_promotion_lock.json"
    ).exists():
        raise FileExistsError("Stage2 outputs already exist")
    weight = lock["locked_weight"]
    if weight is None:
        result = {
            "schema_version": 1,
            "created_utc": utc_now(),
            "stage1_candidate_lock_sha256": sha256_file(
                out_dir / "stage1_candidate_lock.json"
            ),
            "locked_weight": None,
            "2024_read": False,
            "comparisons": {},
            "promotion_audit": {},
            "candidate_promoted": False,
            "reason": "No global blend passed every Stage1 group/segment",
            "no_2024_reselection_or_retuning": True,
        }
        return _write_stage2_lock(out_dir, result)

    input_before = _stage2_input_snapshot(raw_dir, cache_dir, artifact_root)
    labels = bayes._read_full_labels(raw_dir / "train" / "train_labels.csv")
    features = shared_strict._read_features(
        cache_dir, labels, expected_end=bayes.YEAR_2024_END
    )
    index_2024 = bayes._year_index(2024)
    baseline_path = artifact_root / "oof" / "gate2024_locked_v3_cf_fix.parquet"
    baseline = bayes._read_prediction(
        baseline_path, index_2024, required_columns=TARGET_COLS
    )
    train_indexes = {
        "kpx_group_1": pd.date_range(
            bayes.YEAR_2022_START,
            bayes.YEAR_2023_END,
            freq="h",
            name="forecast_kst_dtm",
        ),
        "kpx_group_2": pd.date_range(
            bayes.YEAR_2022_START,
            bayes.YEAR_2023_END,
            freq="h",
            name="forecast_kst_dtm",
        ),
        "kpx_group_3": bayes._year_index(2023),
    }
    application_indexes = {group: index_2024 for group in TARGET_COLS}
    models, actions, quantiles, training = _fit_actions(
        features=features,
        labels=labels,
        baseline=baseline,
        train_indexes=train_indexes,
        application_indexes=application_indexes,
        model_spec=model_spec,
        action_config=action_config,
    )
    candidate = _blend_frame(baseline, actions, application_indexes, float(weight))
    segments = _year_segments(2024)
    comparisons = {
        group: bayes._comparison(
            labels.loc[index_2024, group],
            baseline[group],
            candidate[group],
            group,
            {name: segments[name] for name in STAGE2_REQUIRED},
        )
        for group in TARGET_COLS
    }
    promoted, promotion_audit = _stage2_promotion(comparisons)
    baseline_metrics = score_details(
        labels.loc[index_2024], baseline, target_cols=TARGET_COLS, capacities=CAPACITY_KWH
    ).as_dict()
    candidate_metrics = score_details(
        labels.loc[index_2024], candidate, target_cols=TARGET_COLS, capacities=CAPACITY_KWH
    ).as_dict()

    output_paths: list[Path] = []
    for name, frame in (
        ("stage2_baseline_2024", baseline),
        ("stage2_weather_bayes_action_2024", actions),
        ("stage2_fixed_blend_2024", candidate),
    ):
        path = out_dir / "oof" / f"{name}.parquet"
        bayes._atomic_parquet(frame, path)
        output_paths.append(path)
    for group, frame in quantiles.items():
        path = out_dir / "oof" / f"stage2_{group}_quantiles.parquet"
        bayes._atomic_parquet(frame, path)
        output_paths.append(path)
    model_path = out_dir / "models" / "stage2_weather_quantile_models.joblib"
    bayes._atomic_joblib(models, model_path)
    output_paths.append(model_path)
    input_after = _stage2_input_snapshot(raw_dir, cache_dir, artifact_root)
    shared_strict._assert_snapshot_equal(
        input_before, input_after, name="Stage2 inputs"
    )
    result = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "stage1_candidate_lock_sha256": sha256_file(
            out_dir / "stage1_candidate_lock.json"
        ),
        "locked_weight": weight,
        "2024_read": True,
        "training": training,
        "fit_application_overlap_count": 0,
        "same_2024_fit_score_reported": False,
        "comparisons": comparisons,
        "comparisons_sha256": bayes._canonical_sha256(comparisons),
        "promotion_audit": promotion_audit,
        "candidate_promoted": promoted,
        "official_full_metrics": {
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "delta_total_score": candidate_metrics["total_score"]
            - baseline_metrics["total_score"],
        },
        "no_2024_reselection_or_retuning": True,
        "stage2_input_snapshot_before": input_before,
        "stage2_input_snapshot_after": input_after,
        "stage2_input_snapshot_sha256": bayes._canonical_sha256(input_before),
        "outputs_before_result": [describe_file(path) for path in output_paths],
        "leaderboard_score_claim": False,
        "untouched_gate_claim": False,
    }
    promotion_lock = _write_stage2_lock(out_dir, result)
    print(f"Stage2 candidate promoted: {promoted}", flush=True)
    return promotion_lock


def _load_stage2_lock(out_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    stage1_lock, _ = _load_stage1_lock(out_dir)
    result_path = out_dir / "stage2_results.json"
    lock_path = out_dir / "stage2_promotion_lock.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["preregister_sha256"] != PREREGISTER_SHA256:
        raise AssertionError("Stage2 preregister hash changed")
    if lock["stage1_candidate_lock_sha256"] != sha256_file(
        out_dir / "stage1_candidate_lock.json"
    ):
        raise AssertionError("Stage1 candidate lock changed")
    if lock["stage2_results_sha256"] != sha256_file(result_path):
        raise AssertionError("Stage2 result changed after promotion lock")
    if result["locked_weight"] != stage1_lock["locked_weight"] or lock[
        "locked_weight"
    ] != stage1_lock["locked_weight"]:
        raise AssertionError("Stage2 weight differs from Stage1 lock")
    if result.get("2024_read"):
        promoted, audit = _stage2_promotion(result["comparisons"])
        before = result["stage2_input_snapshot_before"]
        shared_strict._assert_snapshot_equal(
            before, result["stage2_input_snapshot_after"], name="recorded Stage2 inputs"
        )
        shared_strict._assert_snapshot_equal(
            before,
            shared_strict._refresh_snapshot(before),
            name="current Stage2 inputs",
        )
    else:
        promoted, audit = False, {}
    if promoted != bool(result["candidate_promoted"]) or audit != result[
        "promotion_audit"
    ]:
        raise AssertionError("Stage2 promotion differs from recomputed rule")
    if promoted != bool(lock["candidate_promoted"]) or promoted != bool(
        lock["csv_allowed"]
    ):
        raise AssertionError("Stage2 promotion lock changed")
    if bayes._canonical_sha256(audit) != lock["promotion_audit_sha256"]:
        raise AssertionError("Stage2 promotion audit digest changed")
    return lock, result


def _final_input_snapshot(
    raw_dir: Path, cache_dir: Path, artifact_root: Path
) -> dict[str, Any]:
    return {
        "labels_full": shared_strict._snapshot_file(
            raw_dir / "train" / "train_labels.csv"
        ),
        "sample_submission": shared_strict._snapshot_file(
            raw_dir / "sample_submission.csv"
        ),
        **{
            f"cache_train_{group}": shared_strict._snapshot_file(
                cache_dir / f"{group}_weather_train.parquet"
            )
            for group in TARGET_COLS
        },
        **{
            f"cache_test_{group}": shared_strict._snapshot_file(
                cache_dir / f"{group}_weather_test.parquet"
            )
            for group in TARGET_COLS
        },
        "baseline_2025": shared_strict._snapshot_file(
            artifact_root
            / "final_cf_fix"
            / "predictions"
            / "corrected_v3_test.parquet"
        ),
    }


def _finalize(
    *,
    raw_dir: Path,
    artifact_root: Path,
    cache_dir: Path,
    out_dir: Path,
    model_spec: Mapping[str, Any],
    action_config: BayesActionConfig,
) -> dict[str, Any]:
    lock, stage2_result = _load_stage2_lock(out_dir)
    final_result_path = out_dir / "final_results.json"
    manifest_path = out_dir / "manifest.json"
    if final_result_path.exists() or manifest_path.exists():
        raise FileExistsError("final outputs already exist")
    final_inputs: dict[str, Any] = {}
    output_details: dict[str, Any] = {}
    if not lock["candidate_promoted"]:
        final_result = {
            "schema_version": 1,
            "created_utc": utc_now(),
            "locked_weight": lock["locked_weight"],
            "candidate_promoted": False,
            "2025_read": False,
            "final_fit_performed": False,
            "submission_created": False,
            "reason": "The locked global candidate failed a preregistered gate",
            "leaderboard_score_claim": False,
        }
        bayes._write_json(final_result_path, final_result)
    else:
        final_inputs = _final_input_snapshot(raw_dir, cache_dir, artifact_root)
        labels = bayes._read_full_labels(raw_dir / "train" / "train_labels.csv")
        train_features = shared_strict._read_features(
            cache_dir, labels, expected_end=bayes.YEAR_2024_END
        )
        sample = pd.read_csv(
            raw_dir / "sample_submission.csv",
            encoding="utf-8-sig",
            dtype={"forecast_id": "string", "forecast_kst_dtm": "string"},
        )
        if tuple(sample.columns) != ("forecast_id", "forecast_kst_dtm", *TARGET_COLS):
            raise AssertionError("sample schema changed")
        test_index = pd.DatetimeIndex(
            pd.to_datetime(sample["forecast_kst_dtm"], errors="raise"),
            name="forecast_kst_dtm",
        )
        if not test_index.equals(bayes._year_index(2025)):
            raise AssertionError("sample 2025 index changed")
        test_features = shared_strict._read_test_features(cache_dir, test_index)
        full_features = {
            group: pd.concat(
                [train_features[group], test_features[group]], axis=0, copy=False
            )
            for group in TARGET_COLS
        }
        baseline_path = (
            artifact_root
            / "final_cf_fix"
            / "predictions"
            / "corrected_v3_test.parquet"
        )
        baseline = bayes._read_prediction(
            baseline_path, test_index, required_columns=TARGET_COLS
        )
        train_index = labels.index
        train_indexes = {group: train_index for group in TARGET_COLS}
        application_indexes = {group: test_index for group in TARGET_COLS}
        models, actions, quantiles, training = _fit_actions(
            features=full_features,
            labels=labels,
            baseline=baseline,
            train_indexes=train_indexes,
            application_indexes=application_indexes,
            model_spec=model_spec,
            action_config=action_config,
        )
        candidate = _blend_frame(
            baseline, actions, application_indexes, float(lock["locked_weight"])
        )
        prediction_path = (
            out_dir / "predictions" / "weather_quantile_bayes_2025.parquet"
        )
        action_path = (
            out_dir / "predictions" / "weather_quantile_bayes_action_2025.parquet"
        )
        model_path = out_dir / "models" / "final_weather_quantile_models.joblib"
        csv_path = out_dir / "weather_quantile_bayes_2025.csv"
        bayes._atomic_parquet(candidate, prediction_path)
        bayes._atomic_parquet(actions, action_path)
        bayes._atomic_joblib(models, model_path)
        quantile_paths: list[Path] = []
        for group, frame in quantiles.items():
            path = out_dir / "predictions" / f"{group}_quantiles_2025.parquet"
            bayes._atomic_parquet(frame, path)
            quantile_paths.append(path)
        submission = sample.copy()
        for group in TARGET_COLS:
            submission[group] = candidate[group].to_numpy(dtype=float)
        bayes._atomic_csv(submission, csv_path)
        verification = bayes._verify_submission(csv_path, sample, candidate)
        output_details = {
            "prediction": describe_file(prediction_path),
            "action": describe_file(action_path),
            "models": describe_file(model_path),
            "quantiles": [describe_file(path) for path in quantile_paths],
            "submission": describe_file(csv_path),
            "submission_verification": verification,
        }
        final_result = {
            "schema_version": 1,
            "created_utc": utc_now(),
            "locked_weight": lock["locked_weight"],
            "candidate_promoted": True,
            "2025_read": True,
            "final_fit_performed": True,
            "training": training,
            "fit_application_overlap_count": 0,
            "same_fit_score_reported": False,
            "submission_created": True,
            "outputs": output_details,
            "leaderboard_score_claim": False,
        }
        bayes._write_json(final_result_path, final_result)

    stage1_result = json.loads(
        (out_dir / "stage1_results.json").read_text(encoding="utf-8")
    )
    input_records = [
        *stage1_result["provenance_before"].values(),
        *stage1_result["stage1_input_snapshot_before"].values(),
        describe_file(out_dir / "postlock_cache_audit.json"),
    ]
    if stage2_result.get("2024_read"):
        input_records.extend(stage2_result["stage2_input_snapshot_before"].values())
    input_records.extend(final_inputs.values())
    outputs = sorted(
        path
        for path in out_dir.rglob("*")
        if path.is_file() and path != manifest_path
    )
    manifest = {
        "schema_version": 1,
        "artifact_type": "weather_quantile_bayes_strict_forward_v1",
        "created_utc": utc_now(),
        "command": [sys.executable, *sys.argv],
        "runtime": {"packages": package_versions(), "git": git_state(PROJECT_DIR)},
        "preregister_sha256": PREREGISTER_SHA256,
        "candidate_weights": BLEND_WEIGHTS,
        "locked_weight": lock["locked_weight"],
        "candidate_promoted": bool(lock["candidate_promoted"]),
        "stage1_candidate_lock": describe_file(
            out_dir / "stage1_candidate_lock.json"
        ),
        "stage2_promotion_lock": describe_file(
            out_dir / "stage2_promotion_lock.json"
        ),
        "inputs": bayes._deduplicate_records(input_records),
        "outputs": [describe_file(path) for path in outputs],
        "results": {
            "stage2_official_full_metrics": stage2_result.get(
                "official_full_metrics"
            ),
            "2024_read": bool(stage2_result.get("2024_read")),
            "2025_read": bool(final_result["2025_read"]),
            "submission_created": bool(final_result["submission_created"]),
            "output_details": output_details,
        },
        "contracts": {
            "stage1_multi_year_cache_values_read_prelock": False,
            "stage1_raw_nwp_physically_byte_capped": True,
            "fit_transforms_train_only": True,
            "quantile_crossing_repaired": True,
            "2024_used_for_candidate_selection": False,
            "no_2024_reselection_or_retuning": True,
            "csv_requires_complete_promotion": True,
            "2024_historically_consumed": True,
            "untouched_gate_claim": False,
            "leaderboard_score_claim": False,
        },
    }
    bayes._write_json(manifest_path, manifest)
    print(
        f"Finalized promoted={lock['candidate_promoted']} manifest={manifest_path}",
        flush=True,
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    raw_dir = args.raw_dir.expanduser().resolve()
    artifact_root = args.artifact_root.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    preregister_path = args.preregister.expanduser().resolve()
    preregister, model_spec, action_config = _verify_preregister(preregister_path)
    if out_dir == artifact_root or artifact_root not in out_dir.parents:
        raise ValueError("out-dir must be a distinct child of artifact-root")
    if args.stage == "stage1":
        _stage1(
            raw_dir=raw_dir,
            artifact_root=artifact_root,
            cache_dir=cache_dir,
            out_dir=out_dir,
            preregister_path=preregister_path,
            preregister=preregister,
            model_spec=model_spec,
            action_config=action_config,
        )
        _postlock_cache_audit(
            raw_dir=raw_dir, cache_dir=cache_dir, out_dir=out_dir
        )
    elif args.stage == "stage2":
        _stage2(
            raw_dir=raw_dir,
            artifact_root=artifact_root,
            cache_dir=cache_dir,
            out_dir=out_dir,
            model_spec=model_spec,
            action_config=action_config,
        )
    else:
        _finalize(
            raw_dir=raw_dir,
            artifact_root=artifact_root,
            cache_dir=cache_dir,
            out_dir=out_dir,
            model_spec=model_spec,
            action_config=action_config,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

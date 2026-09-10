"""Strict-forward analog/KNN retrieval of 24-hour generation trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.run_run_sequence_residual import (  # noqa: E402
    FINAL_END,
    FINAL_START,
    STAGE2_END,
    STAGE2_START,
    _atomic_joblib,
    _atomic_parquet,
    _comparisons,
    _component_paths,
    _core_weather,
    _frame_sha256,
    _interval,
    _json_ready,
    _mapping_sha256,
    _read_stage1_baseline,
    _reconstruct_baseline,
    _write_json,
    _write_submission,
)
from scripts.run_shared_q07_multiseed import (  # noqa: E402
    EXPECTED_ROWS_PRE2024,
    EXPECTED_ROWS_THROUGH2024,
    EXPECTED_TEST_ROWS,
    FINAL_COMPONENT_FILES,
    GATE_COMPONENT_FILES,
    OFFICIAL_LABEL_PREFIX_SHA256,
    OFFICIAL_STAGE1_PREFIX_BYTES,
    YEAR_2023_END,
    YEAR_2024_END,
    _read_features,
    _read_labels,
    _read_stage1_raw_features,
    _read_test_features,
)
from src.analog_run import AnalogRunRetriever, blend_analog_with_baseline  # noqa: E402
from src.manifest import (  # noqa: E402
    describe_file,
    git_state,
    package_versions,
    sha256_file,
    utc_now,
)
from src.metric import CAPACITY_KWH, TARGET_COLS  # noqa: E402
from src.run_sequence_residual import choose_single_candidate  # noqa: E402


PREREGISTER_SHA256 = (
    "93ca748901fe6f62881db371981f770e4e72bd62288c05e827ec8526964ebd56"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument(
        "--cache-dir", type=Path, default=PROJECT_DIR / "artifacts" / "cache"
    )
    parser.add_argument(
        "--artifact-root", type=Path, default=PROJECT_DIR / "artifacts"
    )
    parser.add_argument(
        "--recipe",
        type=Path,
        default=PROJECT_DIR / "configs" / "train_final.v3.locked.json",
    )
    parser.add_argument(
        "--preregister",
        type=Path,
        default=PROJECT_DIR / "configs" / "analog_run_retrieval_preregister.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_DIR / "artifacts" / "postgate" / "analog_run_retrieval",
    )
    parser.add_argument(
        "--stage", choices=("stage1", "stage2", "all"), default="all"
    )
    return parser.parse_args(argv)


def _load_preregister(path: Path) -> dict[str, Any]:
    observed = sha256_file(path)
    if observed != PREREGISTER_SHA256:
        raise AssertionError(f"analog preregister SHA changed: {observed}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("experiment_id") != "analog_run_retrieval_strict_forward_v1":
        raise AssertionError("analog preregister experiment id changed")
    analog_ids = [item["id"] for item in payload["analog_candidates"]]
    if analog_ids != ["summary_k10", "summary_k30", "pca12_k60"]:
        raise AssertionError("analog base candidate set changed")
    if int(payload["candidate_count"]) != 12:
        raise AssertionError("analog variant count changed")
    if payload["stage1"]["selection_rule"]["locked_recipe_count_max"] != 1:
        raise AssertionError("analog lock no longer limits selection to one recipe")
    return payload


def _source_snapshot(preregister_path: Path) -> dict[str, Any]:
    paths = {
        "runner": Path(__file__).resolve(),
        "analog_module": PROJECT_DIR / "src" / "analog_run.py",
        "analog_test": PROJECT_DIR / "tests" / "test_analog_run.py",
        "strict_raw_helper": PROJECT_DIR / "scripts" / "run_shared_q07_multiseed.py",
        "baseline_helper": PROJECT_DIR / "scripts" / "run_run_sequence_residual.py",
        "run_contract": PROJECT_DIR / "src" / "run_sequence_residual.py",
        "features": PROJECT_DIR / "src" / "features.py",
        "temporal": PROJECT_DIR / "src" / "temporal.py",
        "metric": PROJECT_DIR / "src" / "metric.py",
        "preregister": preregister_path.resolve(),
    }
    return {name: describe_file(path) for name, path in paths.items()}


def _stage1_input_snapshot(
    raw_dir: Path, artifact_root: Path, preregister_path: Path
) -> dict[str, Any]:
    paths = {
        "info": raw_dir / "info.xlsx",
        "stage1_g12": artifact_root / "oof" / "dev2023_locked_v3.parquet",
        "stage1_g3_components": artifact_root
        / "oof"
        / "g3dev2023h2_candidates.parquet",
        "stage1_exact_reference": artifact_root
        / "postgate"
        / "shared_q07_multiseed_strict"
        / "oof"
        / "stage1_corrected_v3_baseline.parquet",
        "preregister": preregister_path.resolve(),
    }
    return {name: describe_file(path) for name, path in paths.items()}


def _assert_unchanged(before: Mapping[str, Any], after: Mapping[str, Any], name: str) -> None:
    if before != after:
        raise AssertionError(f"{name} changed during pre-2024 selection")


def _make_retriever(
    preregister: Mapping[str, Any], analog_spec: Mapping[str, Any]
) -> AnalogRunRetriever:
    embedding = str(analog_spec["embedding"])
    pca_params = None
    if embedding == "pca12_core_trajectory":
        registered = preregister["embedding_definitions"][embedding]["pca"]
        # ``fit_on`` is preregistration/audit metadata rather than an sklearn
        # constructor argument.  Keep the registered estimator parameters
        # exact while excluding that descriptive field.
        pca_params = {
            name: registered[name]
            for name in ("n_components", "whiten", "svd_solver", "random_state")
        }
    return AnalogRunRetriever(
        channels=tuple(preregister["weather_channels"]),
        embedding_kind=embedding,
        k=int(analog_spec["k"]),
        pca_params=pca_params,
        expected_hours=int(preregister["run_contract"]["hours"]),
    )


def _recipe_map(preregister: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for analog in preregister["analog_candidates"]:
        for variant in preregister["variants"]:
            candidate_id = f"{analog['id']}__{variant['id_suffix']}"
            output[candidate_id] = {
                "id": candidate_id,
                "analog": dict(analog),
                "variant": dict(variant),
            }
    if len(output) != int(preregister["candidate_count"]):
        raise AssertionError("constructed analog recipe count differs from preregister")
    return output


def _fit_predict_analog(
    *,
    preregister: Mapping[str, Any],
    analog_spec: Mapping[str, Any],
    group: str,
    weather: pd.DataFrame,
    actual: pd.Series,
    fit_bounds: Sequence[str],
    apply_bounds: Sequence[str],
) -> tuple[AnalogRunRetriever, pd.Series, dict[str, Any]]:
    fit_index = _interval(weather.index, fit_bounds)
    apply_index = _interval(weather.index, apply_bounds)
    if fit_index.max() >= apply_index.min() or fit_index.intersection(apply_index).size:
        raise AssertionError("analog fit runs overlap or follow apply runs")
    retriever = _make_retriever(preregister, analog_spec)
    retriever.fit(
        weather.loc[fit_index], actual.loc[fit_index] / CAPACITY_KWH[group]
    )
    prediction, neighbor_audit = retriever.predict(weather.loc[apply_index])
    audit = {
        "group": group,
        "fit_start": fit_index.min().isoformat(),
        "fit_end": fit_index.max().isoformat(),
        "apply_start": apply_index.min().isoformat(),
        "apply_end": apply_index.max().isoformat(),
        "fit_apply_overlap_rows": 0,
        "fit_strictly_before_apply": True,
        "fit": retriever.fit_audit_,
        "neighbors": neighbor_audit,
    }
    return retriever, prediction, audit


def _stage1(
    *,
    raw_dir: Path,
    cache_dir: Path,
    artifact_root: Path,
    out_dir: Path,
    preregister_path: Path,
    preregister: Mapping[str, Any],
) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"Stage 1 requires a new output directory: {out_dir}")
    out_dir.mkdir(parents=True)
    shutil.copyfile(preregister_path, out_dir / "preregister.json")
    sources_before = _source_snapshot(preregister_path)
    inputs_before = _stage1_input_snapshot(raw_dir, artifact_root, preregister_path)

    labels = _read_labels(
        raw_dir / "train" / "train_labels.csv",
        nrows=EXPECTED_ROWS_PRE2024,
        expected_end=YEAR_2023_END,
        prefix_bytes=OFFICIAL_STAGE1_PREFIX_BYTES["labels"],
        expected_prefix_sha256=OFFICIAL_LABEL_PREFIX_SHA256,
    )
    print("Stage 1: building byte-bounded pre-2024 NWP features", flush=True)
    raw_features, raw_contract = _read_stage1_raw_features(raw_dir, labels)
    weather = _core_weather(raw_features, preregister["weather_channels"])
    del raw_features
    baseline, baseline_audit = _read_stage1_baseline(artifact_root)

    recipes = _recipe_map(preregister)
    analog_predictions: dict[str, dict[str, pd.Series]] = {
        spec["id"]: {} for spec in preregister["analog_candidates"]
    }
    analog_audit: dict[str, Any] = {}
    for analog_spec in preregister["analog_candidates"]:
        analog_id = str(analog_spec["id"])
        print(f"Stage 1: fitting analog base {analog_id}", flush=True)
        analog_audit[analog_id] = {}
        for group in TARGET_COLS:
            fold = preregister["stage1"]["folds"][group]
            model, prediction, audit = _fit_predict_analog(
                preregister=preregister,
                analog_spec=analog_spec,
                group=group,
                weather=weather[group],
                actual=labels[group],
                fit_bounds=fold["fit"],
                apply_bounds=fold["apply"],
            )
            analog_predictions[analog_id][group] = prediction
            model_path = out_dir / "models" / f"stage1__{analog_id}__{group}.joblib"
            _atomic_joblib(
                {
                    "model": model,
                    "analog_spec": analog_spec,
                    "weather_channels": list(preregister["weather_channels"]),
                    "group": group,
                },
                model_path,
            )
            audit["model"] = describe_file(model_path)
            analog_audit[analog_id][group] = audit

    comparisons: dict[str, Any] = {}
    for candidate_id, recipe in recipes.items():
        frame = pd.DataFrame(index=baseline.index, columns=list(TARGET_COLS), dtype=float)
        by_group: dict[str, Any] = {}
        analog_id = recipe["analog"]["id"]
        analog_weight = float(recipe["variant"]["analog_weight"])
        for group in TARGET_COLS:
            analog_cf = analog_predictions[analog_id][group]
            prediction = blend_analog_with_baseline(
                analog_cf,
                baseline.loc[analog_cf.index, group],
                capacity_kwh=CAPACITY_KWH[group],
                analog_weight=analog_weight,
            )
            frame.loc[prediction.index, group] = prediction
            by_group[group] = _comparisons(
                actual=labels.loc[prediction.index, group],
                baseline=baseline.loc[prediction.index, group],
                candidate=prediction,
                group=group,
                slices=preregister["stage1"]["required_slices"][group],
            )
        _atomic_parquet(
            frame, out_dir / "oof" / f"stage1__{candidate_id}.parquet"
        )
        comparisons[candidate_id] = by_group
    _atomic_parquet(
        baseline, out_dir / "oof" / "stage1_corrected_v3_baseline.parquet"
    )
    selected = choose_single_candidate(comparisons)
    sources_after = _source_snapshot(preregister_path)
    inputs_after = _stage1_input_snapshot(raw_dir, artifact_root, preregister_path)
    _assert_unchanged(sources_before, sources_after, "source snapshot")
    _assert_unchanged(inputs_before, inputs_after, "input snapshot")
    result = {
        "schema_version": 1,
        "experiment_id": "analog_run_retrieval_stage1",
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "candidate_count": len(recipes),
        "recipe_map": recipes,
        "selected_candidate": selected,
        "selection_changes_after_2024": False,
        "2024_labels_weather_or_predictions_read": False,
        "bounded_label_rows": int(len(labels)),
        "raw_feature_contract": raw_contract,
        "cache_dir_deliberately_not_read": str(cache_dir.resolve()),
        "baseline_exact_reconstruction": baseline_audit,
        "analog_audit": analog_audit,
        "comparisons": comparisons,
        "source_snapshot_before": sources_before,
        "source_snapshot_after": sources_after,
        "source_snapshot_sha256": _mapping_sha256(sources_before),
        "input_snapshot_before": inputs_before,
        "input_snapshot_after": inputs_after,
        "input_snapshot_sha256": _mapping_sha256(inputs_before),
        "snapshots_unchanged": True,
        "leaderboard_score_claim": False,
    }
    result_path = out_dir / "stage1_results.json"
    _write_json(result_path, result)
    _write_json(
        out_dir / "pre2024_lock.json",
        {
            "schema_version": 1,
            "experiment_id": "analog_run_retrieval_pre2024_lock",
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "stage1_results_sha256": sha256_file(result_path),
            "source_snapshot_sha256": result["source_snapshot_sha256"],
            "input_snapshot_sha256": result["input_snapshot_sha256"],
            "selected_candidate": selected,
            "selection_changes_after_2024": False,
            "2024_read_before_lock": False,
        },
    )
    print("Stage 1 locked candidate:", selected, flush=True)
    return result


def _load_lock(
    out_dir: Path,
    preregister_path: Path,
    raw_dir: Path,
    artifact_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result_path = out_dir / "stage1_results.json"
    lock_path = out_dir / "pre2024_lock.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["preregister_sha256"] != PREREGISTER_SHA256:
        raise AssertionError("analog lock preregister SHA changed")
    if lock["stage1_results_sha256"] != sha256_file(result_path):
        raise AssertionError("analog Stage 1 result changed after lock")
    if choose_single_candidate(result["comparisons"]) != lock["selected_candidate"]:
        raise AssertionError("analog locked recipe violates selection rule")
    if _mapping_sha256(_source_snapshot(preregister_path)) != lock["source_snapshot_sha256"]:
        raise AssertionError("analog source snapshot changed after lock")
    if _mapping_sha256(
        _stage1_input_snapshot(raw_dir, artifact_root, preregister_path)
    ) != lock["input_snapshot_sha256"]:
        raise AssertionError("analog input snapshot changed after lock")
    return lock, result


def _stage2_no_candidate(out_dir: Path) -> dict[str, Any]:
    result = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "locked_candidate": None,
        "2024_read": False,
        "promoted": False,
        "reason": "no single analog recipe improved every Stage 1 group/slice",
        "selection_changes_after_2024": False,
        "final_fit_performed": False,
        "submission_created": False,
        "leaderboard_score_claim": False,
    }
    _write_json(out_dir / "stage2_results.json", result)
    _write_json(
        out_dir / "promotion_lock.json",
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "locked_candidate": None,
            "promoted": False,
            "2024_read": False,
            "stage2_results_sha256": sha256_file(out_dir / "stage2_results.json"),
            "pre2024_lock_sha256": sha256_file(out_dir / "pre2024_lock.json"),
        },
    )
    return result


def _stage2(
    *,
    raw_dir: Path,
    cache_dir: Path,
    artifact_root: Path,
    recipe_path: Path,
    out_dir: Path,
    preregister_path: Path,
    preregister: Mapping[str, Any],
) -> dict[str, Any]:
    if (out_dir / "stage2_results.json").exists():
        raise FileExistsError("analog Stage 2 already exists")
    lock, stage1_result = _load_lock(
        out_dir, preregister_path, raw_dir, artifact_root
    )
    selected = lock["selected_candidate"]
    if selected is None:
        return _stage2_no_candidate(out_dir)

    locked_recipe = stage1_result["recipe_map"][selected]
    analog_spec = locked_recipe["analog"]
    analog_weight = float(locked_recipe["variant"]["analog_weight"])
    labels = _read_labels(
        raw_dir / "train" / "train_labels.csv",
        nrows=EXPECTED_ROWS_THROUGH2024,
        expected_end=YEAR_2024_END,
    )
    full_weather = _read_features(cache_dir, labels, expected_end=YEAR_2024_END)
    weather = _core_weather(full_weather, preregister["weather_channels"])
    gate_index = pd.date_range(
        STAGE2_START, STAGE2_END, freq="h", name="forecast_kst_dtm"
    )
    gate_baseline, gate_baseline_audit = _reconstruct_baseline(
        artifact_root=artifact_root,
        recipe_path=recipe_path,
        component_files=GATE_COMPONENT_FILES,
        reference_path=artifact_root / "oof" / "gate2024_locked_v3_cf_fix.parquet",
        expected_index=gate_index,
    )
    candidate = pd.DataFrame(index=gate_index, columns=list(TARGET_COLS), dtype=float)
    training: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    for group in TARGET_COLS:
        model, analog_cf, audit = _fit_predict_analog(
            preregister=preregister,
            analog_spec=analog_spec,
            group=group,
            weather=weather[group],
            actual=labels[group],
            fit_bounds=preregister["stage2"]["fit"][group],
            apply_bounds=preregister["stage2"]["apply"],
        )
        prediction = blend_analog_with_baseline(
            analog_cf,
            gate_baseline[group],
            capacity_kwh=CAPACITY_KWH[group],
            analog_weight=analog_weight,
        )
        candidate[group] = prediction
        model_path = out_dir / "models" / f"stage2__{selected}__{group}.joblib"
        _atomic_joblib(
            {
                "model": model,
                "locked_recipe": locked_recipe,
                "weather_channels": list(preregister["weather_channels"]),
                "group": group,
            },
            model_path,
        )
        audit["model"] = describe_file(model_path)
        training[group] = audit
        comparisons[group] = _comparisons(
            actual=labels.loc[gate_index, group],
            baseline=gate_baseline[group],
            candidate=prediction,
            group=group,
            slices=preregister["stage2"]["required_slices"],
        )
    promoted = all(
        float(item["delta"]) > 0.0
        for slices in comparisons.values()
        for item in slices.values()
    )
    _atomic_parquet(
        gate_baseline, out_dir / "oof" / "stage2_corrected_v3_baseline.parquet"
    )
    _atomic_parquet(candidate, out_dir / "oof" / f"stage2__{selected}.parquet")
    result = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "locked_candidate": selected,
        "2024_read": True,
        "selection_changes_after_2024": False,
        "comparisons": comparisons,
        "training": training,
        "baseline_exact_reconstruction": gate_baseline_audit,
        "promoted": bool(promoted),
        "final_fit_performed": False,
        "submission_created": False,
        "leaderboard_score_claim": False,
    }
    result_path = out_dir / "stage2_results.json"
    _write_json(result_path, result)
    _write_json(
        out_dir / "promotion_lock.json",
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "locked_candidate": selected,
            "promoted": bool(promoted),
            "2024_read": True,
            "stage2_results_sha256": sha256_file(result_path),
            "pre2024_lock_sha256": sha256_file(out_dir / "pre2024_lock.json"),
        },
    )
    if promoted:
        _final(
            raw_dir=raw_dir,
            cache_dir=cache_dir,
            artifact_root=artifact_root,
            recipe_path=recipe_path,
            out_dir=out_dir,
            preregister=preregister,
            locked_recipe=locked_recipe,
            labels=labels,
            weather=weather,
        )
        result["final_fit_performed"] = True
        result["submission_created"] = True
    return result


def _final(
    *,
    raw_dir: Path,
    cache_dir: Path,
    artifact_root: Path,
    recipe_path: Path,
    out_dir: Path,
    preregister: Mapping[str, Any],
    locked_recipe: Mapping[str, Any],
    labels: pd.DataFrame,
    weather: Mapping[str, pd.DataFrame],
) -> None:
    promotion = json.loads((out_dir / "promotion_lock.json").read_text(encoding="utf-8"))
    if promotion.get("promoted") is not True:
        raise AssertionError("analog final requires positive promotion lock")
    sample_path = raw_dir / "sample_submission.csv"
    sample = pd.read_csv(sample_path, encoding="utf-8-sig")
    test_index = pd.DatetimeIndex(
        pd.to_datetime(sample["forecast_kst_dtm"], errors="raise"),
        name="forecast_kst_dtm",
    )
    if len(test_index) != EXPECTED_TEST_ROWS or test_index.min() != FINAL_START or test_index.max() != FINAL_END:
        raise AssertionError("analog sample interval changed")
    test_weather_full = _read_test_features(cache_dir, test_index)
    test_weather = _core_weather(test_weather_full, preregister["weather_channels"])
    final_baseline, baseline_audit = _reconstruct_baseline(
        artifact_root=artifact_root,
        recipe_path=recipe_path,
        component_files=FINAL_COMPONENT_FILES,
        reference_path=artifact_root
        / "final_cf_fix"
        / "predictions"
        / "corrected_v3_test.parquet",
        expected_index=test_index,
    )
    predictions = pd.DataFrame(index=test_index, columns=list(TARGET_COLS), dtype=float)
    training: dict[str, Any] = {}
    analog_spec = locked_recipe["analog"]
    weight = float(locked_recipe["variant"]["analog_weight"])
    for group in TARGET_COLS:
        fit_start = "2022-01-01T01:00:00" if group != "kpx_group_3" else "2023-01-01T01:00:00"
        model, analog_cf, audit = _fit_predict_analog(
            preregister=preregister,
            analog_spec=analog_spec,
            group=group,
            weather=pd.concat([weather[group], test_weather[group]], axis=0),
            actual=pd.concat(
                [labels[group], pd.Series(np.nan, index=test_index, name=group)], axis=0
            ),
            fit_bounds=[fit_start, "2025-01-01T00:00:00"],
            apply_bounds=preregister["final"]["apply"],
        )
        prediction = blend_analog_with_baseline(
            analog_cf,
            final_baseline[group],
            capacity_kwh=CAPACITY_KWH[group],
            analog_weight=weight,
        )
        predictions[group] = prediction
        model_path = out_dir / "models" / f"final__{locked_recipe['id']}__{group}.joblib"
        _atomic_joblib(
            {
                "model": model,
                "locked_recipe": locked_recipe,
                "weather_channels": list(preregister["weather_channels"]),
                "group": group,
            },
            model_path,
        )
        audit["model"] = describe_file(model_path)
        training[group] = audit
    prediction_path = out_dir / "predictions" / "analog_run_retrieval_2025.parquet"
    _atomic_parquet(predictions, prediction_path)
    submission_path = out_dir / preregister["final"]["submission_name"]
    submission = _write_submission(sample_path, predictions, submission_path)
    _write_json(
        out_dir / "final_results.json",
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "locked_recipe": locked_recipe,
            "training": training,
            "baseline_exact_reconstruction": baseline_audit,
            "prediction": describe_file(prediction_path),
            "submission": submission,
            "leaderboard_score_claim": False,
        },
    )


def _write_manifest(
    *,
    args: argparse.Namespace,
    preregister: Mapping[str, Any],
    stage2_result: Mapping[str, Any] | None,
    name: str,
) -> None:
    destination = args.out_dir / name
    outputs = sorted(
        path for path in args.out_dir.rglob("*") if path.is_file() and path != destination
    )
    safe_inputs = [
        args.preregister,
        args.recipe,
        args.raw_dir / "info.xlsx",
        args.artifact_root / "oof" / "dev2023_locked_v3.parquet",
        args.artifact_root / "oof" / "g3dev2023h2_candidates.parquet",
        args.artifact_root
        / "postgate"
        / "shared_q07_multiseed_strict"
        / "oof"
        / "stage1_corrected_v3_baseline.parquet",
        Path(__file__).resolve(),
        PROJECT_DIR / "src" / "analog_run.py",
        PROJECT_DIR / "tests" / "test_analog_run.py",
    ]
    stage1 = json.loads(
        (args.out_dir / "stage1_results.json").read_text(encoding="utf-8")
    )
    _write_json(
        destination,
        {
            "schema_version": 1,
            "artifact_type": "analog_run_retrieval_strict_forward",
            "created_utc": utc_now(),
            "command": [sys.executable, *sys.argv],
            "runtime": {"packages": package_versions(), "git": git_state(PROJECT_DIR)},
            "preregister_sha256": PREREGISTER_SHA256,
            "candidate_count": int(preregister["candidate_count"]),
            "inputs": [describe_file(path) for path in safe_inputs],
            "raw_prefix_inputs": stage1["raw_feature_contract"],
            "outputs": [describe_file(path) for path in outputs],
            "stage2": dict(stage2_result or {}),
            "leaderboard_score_claim": False,
        },
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    for name in ("raw_dir", "cache_dir", "artifact_root", "recipe", "preregister", "out_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    preregister = _load_preregister(args.preregister)
    if args.stage in {"stage1", "all"}:
        _stage1(
            raw_dir=args.raw_dir,
            cache_dir=args.cache_dir,
            artifact_root=args.artifact_root,
            out_dir=args.out_dir,
            preregister_path=args.preregister,
            preregister=preregister,
        )
        if args.stage == "stage1":
            _write_manifest(
                args=args, preregister=preregister, stage2_result=None, name="stage1_manifest.json"
            )
            return
    if args.stage in {"stage2", "all"}:
        result = _stage2(
            raw_dir=args.raw_dir,
            cache_dir=args.cache_dir,
            artifact_root=args.artifact_root,
            recipe_path=args.recipe,
            out_dir=args.out_dir,
            preregister_path=args.preregister,
            preregister=preregister,
        )
        _write_manifest(
            args=args, preregister=preregister, stage2_result=result, name="manifest.json"
        )


if __name__ == "__main__":
    main()

"""Strict-forward action-conditional interval-probability Stage1 experiment."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
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
from scripts import run_shared_q07_multiseed as shared  # noqa: E402
from src.direct_interval_probability import (  # noqa: E402
    ACTION_GRID_CF,
    BLEND_WEIGHTS,
    COMMON_PARAMETERS,
    CONTEXT_COLUMNS,
    DirectIntervalProbabilityModel,
    STAGE1_REQUIRED,
    assert_strict_forward,
    blend_action_kwh,
    compact_context,
    official_action_utility,
    select_stage1_weight,
)
from src.manifest import (  # noqa: E402
    describe_file,
    git_state,
    package_versions,
    sha256_file,
    utc_now,
)
from src.metric import CAPACITY_KWH, TARGET_COLS  # noqa: E402


PREREGISTER_SHA256 = "1796cc8628e3ad46039a6f5352c4c51b1a851afc7e6e6f055d9c85fcd1d699ff"
ARTIFACT_TYPE = "direct_interval_probability_strict_forward_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("stage1",), default="stage1")
    parser.add_argument(
        "--raw-dir", type=Path, default=Path(r"data/local/open")
    )
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/postgate/direct_interval_probability_strict_v1"),
    )
    parser.add_argument(
        "--preregister",
        type=Path,
        default=Path("configs/direct_interval_probability_preregister_v1.json"),
    )
    return parser.parse_args(argv)


def _verify_preregister(path: Path) -> dict[str, Any]:
    observed = sha256_file(path)
    if observed != PREREGISTER_SHA256:
        raise AssertionError(f"preregister hash changed: {observed}")
    expected_sidecar = f"{PREREGISTER_SHA256}  {path.name}\n"
    if path.with_suffix(".sha256").read_text(encoding="utf-8") != expected_sidecar:
        raise AssertionError("preregister sidecar changed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["experiment_id"] != "direct_interval_probability_strict_forward_v1":
        raise AssertionError("experiment id changed")
    if tuple(payload["feature_contract"]["context_columns_in_exact_order"]) != CONTEXT_COLUMNS:
        raise AssertionError("compact feature order changed")
    if int(payload["feature_contract"]["expanded_feature_count"]) != len(CONTEXT_COLUMNS) + 1:
        raise AssertionError("expanded feature count changed")
    if int(payload["action_expansion_and_targets"]["action_count"]) != len(ACTION_GRID_CF):
        raise AssertionError("action count changed")
    if tuple(map(float, payload["candidate_family"]["fixed_global_blend_weights"])) != tuple(BLEND_WEIGHTS.values()):
        raise AssertionError("blend weights changed")
    if payload["models"]["common_parameters"] != COMMON_PARAMETERS:
        raise AssertionError("common model parameters changed")
    if payload["models"]["p6"] != {
        "class": "lightgbm.LGBMClassifier",
        "objective": "binary",
        "random_state": 4206,
    }:
        raise AssertionError("p6 model changed")
    if payload["models"]["p8"] != {
        "class": "lightgbm.LGBMClassifier",
        "objective": "binary",
        "random_state": 4208,
    }:
        raise AssertionError("p8 model changed")
    if payload["models"]["eae"] != {
        "class": "lightgbm.LGBMRegressor",
        "objective": "regression_l2",
        "random_state": 4210,
    }:
        raise AssertionError("EAE model changed")
    if int(payload["stage1"]["registered_slice_count"]) != 17:
        raise AssertionError("Stage1 slice count changed")
    public = payload["public_contract"]
    for key in (
        "public_metric_triplets_used",
        "public_scale_artifacts_used",
        "public_submission_artifacts_used",
    ):
        if public[key] is not False:
            raise AssertionError(f"forbidden Public contract changed: {key}")
    return payload


def _resolve_local_module(module: str) -> Path | None:
    if not module or module.split(".", 1)[0] not in {"scripts", "src"}:
        return None
    candidate = PROJECT_DIR.joinpath(*module.split(".")).with_suffix(".py")
    return candidate.resolve() if candidate.is_file() else None


def _direct_local_imports(path: Path) -> set[Path]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[Path] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                resolved = _resolve_local_module(alias.name)
                if resolved is not None:
                    imports.add(resolved)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            module = node.module or ""
            resolved = _resolve_local_module(module)
            if resolved is not None:
                imports.add(resolved)
            if module in {"scripts", "src"}:
                for alias in node.names:
                    resolved = _resolve_local_module(f"{module}.{alias.name}")
                    if resolved is not None:
                        imports.add(resolved)
    return imports


def _static_repo_import_closure(roots: Sequence[Path]) -> tuple[Path, ...]:
    pending = [path.resolve() for path in roots]
    observed: set[Path] = set()
    while pending:
        current = pending.pop()
        if current in observed:
            continue
        if not current.is_file() or current.suffix != ".py":
            raise FileNotFoundError(current)
        observed.add(current)
        pending.extend(_direct_local_imports(current).difference(observed))
    return tuple(sorted(observed, key=lambda item: item.relative_to(PROJECT_DIR).as_posix()))


def _provenance_paths(preregister_path: Path) -> dict[str, Path]:
    closure = _static_repo_import_closure((Path(__file__).resolve(),))
    paths = {
        f"repo_source::{path.relative_to(PROJECT_DIR).as_posix()}": path
        for path in closure
    }
    paths.update(
        {
            "focused_test": PROJECT_DIR / "tests/test_direct_interval_probability.py",
            "runner_test": PROJECT_DIR / "tests/test_direct_interval_probability_runner.py",
            "preregister": preregister_path.resolve(),
            "preregister_sidecar": preregister_path.with_suffix(".sha256").resolve(),
        }
    )
    return paths


def _snapshot_named(paths: Mapping[str, Path]) -> dict[str, Any]:
    return {name: shared._snapshot_file(path) for name, path in paths.items()}


def _census_snapshot(preregister: Mapping[str, Any]) -> dict[str, Any]:
    paths: list[str] = []
    for record in preregister["census_and_novelty_contract"]["censused_implementations"]:
        paths.extend(record["paths"])
    if len(paths) != len(set(paths)):
        raise AssertionError("census paths are duplicated")
    output: dict[str, Any] = {}
    for relative in paths:
        path = PROJECT_DIR / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        output[relative] = shared._snapshot_file(path)
    return output


def _copy_exclusive(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    with source.open("rb") as input_stream, destination.open("xb") as output_stream:
        shutil.copyfileobj(input_stream, output_stream)


def _array_sha256(array: Any) -> str:
    values = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(json.dumps(list(values.shape), separators=(",", ":")).encode("ascii"))
    digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def _atomic_npz(destination: Path, arrays: Mapping[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as stream:
            np.savez(stream, **arrays)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _surface_arrays(surface: Mapping[str, Any]) -> dict[str, np.ndarray]:
    utility = official_action_utility(surface["p6"], surface["p8"], surface["eae"])
    return {
        "p6_raw": np.asarray(surface["p6_raw"], dtype=np.float64),
        "p8_raw": np.asarray(surface["p8_raw"], dtype=np.float64),
        "p6": np.asarray(surface["p6"], dtype=np.float64),
        "p8": np.asarray(surface["p8"], dtype=np.float64),
        "eae": np.asarray(surface["eae"], dtype=np.float64),
        "utility": np.asarray(utility, dtype=np.float64),
        "repair_count": np.asarray([surface["repair_count"]], dtype=np.int64),
    }


def _fit_predict_group(
    *,
    group: str,
    context: pd.DataFrame,
    actual: pd.Series,
    baseline: pd.Series,
    fit_index: pd.DatetimeIndex,
    application_index: pd.DatetimeIndex,
    out_dir: Path,
) -> tuple[pd.Series, dict[str, Any], list[Path]]:
    assert_strict_forward(fit_index, application_index)
    print(
        f"fit direct interval heads {group}: {fit_index.min()}..{fit_index.max()} "
        f"-> {application_index.min()}..{application_index.max()}",
        flush=True,
    )
    model = DirectIntervalProbabilityModel().fit(
        context.loc[fit_index], actual.loc[fit_index], capacity_kwh=CAPACITY_KWH[group]
    )
    action, diagnostics, surface = model.predict_action(
        context.loc[application_index],
        baseline.loc[application_index],
        capacity_kwh=CAPACITY_KWH[group],
    )
    arrays = _surface_arrays(surface)
    model_path = out_dir / f"models/stage1_{group}.joblib"
    action_path = out_dir / f"oof/stage1_{group}_action_cf.parquet"
    diagnostics_path = out_dir / f"oof/stage1_{group}_action_diagnostics.parquet"
    surface_path = out_dir / f"oof/stage1_{group}_surfaces.npz"
    strict._atomic_joblib(model, model_path)
    strict._atomic_parquet(action.to_frame(), action_path)
    strict._atomic_parquet(diagnostics, diagnostics_path)
    _atomic_npz(surface_path, arrays)

    reloaded: DirectIntervalProbabilityModel = joblib.load(model_path)
    reloaded_action, reloaded_diagnostics, reloaded_surface = reloaded.predict_action(
        context.loc[application_index],
        baseline.loc[application_index],
        capacity_kwh=CAPACITY_KWH[group],
    )
    if not np.array_equal(action.to_numpy(), reloaded_action.to_numpy()):
        raise AssertionError(f"{group} reloaded action differs")
    if not np.array_equal(diagnostics.to_numpy(), reloaded_diagnostics.to_numpy()):
        raise AssertionError(f"{group} reloaded diagnostics differ")
    reloaded_arrays = _surface_arrays(reloaded_surface)
    for name, expected in arrays.items():
        if not np.array_equal(expected, reloaded_arrays[name]):
            raise AssertionError(f"{group} reloaded surface differs: {name}")
    metadata = model.metadata()
    metadata.update(
        {
            "group": group,
            "fit_start": fit_index.min(),
            "fit_end": fit_index.max(),
            "application_start": application_index.min(),
            "application_end": application_index.max(),
            "fit_end_before_application_start": True,
            "fit_application_overlap_count": 0,
            "baseline_model_feature_count": 0,
            "application_target_feature_count": 0,
            "probability_repair_count_all_actions": int(surface["repair_count"]),
            "selected_action_min_cf": float(action.min()),
            "selected_action_max_cf": float(action.max()),
            "selected_action_mean_cf": float(action.mean()),
            "action_sha256": _array_sha256(action.to_numpy()),
            "diagnostics_sha256": _array_sha256(diagnostics.to_numpy()),
            "surface_array_sha256": {name: _array_sha256(values) for name, values in arrays.items()},
            "reloaded_all_outputs_bit_exact": True,
        }
    )
    return action, metadata, [model_path, action_path, diagnostics_path, surface_path]


def _blend_frame(
    baseline: pd.DataFrame,
    actions: pd.DataFrame,
    application_indexes: Mapping[str, pd.DatetimeIndex],
    weight: float,
) -> pd.DataFrame:
    result = baseline.copy()
    for group in TARGET_COLS:
        index = application_indexes[group]
        result.loc[index, group] = blend_action_kwh(
            baseline.loc[index, group],
            actions.loc[index, group],
            weight=weight,
            capacity_kwh=CAPACITY_KWH[group],
        )
    return result


def _all_output_files(out_dir: Path) -> list[Path]:
    return sorted(
        [path for path in out_dir.rglob("*") if path.is_file() and path.name != "manifest.json"],
        key=lambda path: path.relative_to(out_dir).as_posix(),
    )


def run_stage1(args: argparse.Namespace, preregister: Mapping[str, Any]) -> dict[str, Any]:
    out_dir = args.out_dir
    if out_dir.exists():
        raise FileExistsError(f"Stage1 requires a new output directory: {out_dir}")
    out_dir.mkdir(parents=True)
    _copy_exclusive(args.preregister, out_dir / "preregister.json")
    _copy_exclusive(args.preregister.with_suffix(".sha256"), out_dir / "preregister.sha256")

    provenance_paths = _provenance_paths(args.preregister)
    provenance_before = _snapshot_named(provenance_paths)
    census = _census_snapshot(preregister)
    census_path = out_dir / "census_evidence.json"
    strict._write_json(
        census_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "searched_terms": preregister["census_and_novelty_contract"]["searched_terms"],
            "snapshots": census,
            "candidate_fit_started_before_census_lock": False,
            "candidate_score_computed_before_census_lock": False,
        },
    )
    input_before = bounded._stage1_input_snapshot(args.raw_dir, args.artifact_root, preregister)
    bounded._assert_preregistered_stage1_files(input_before, preregister)

    full_index = pd.date_range(
        "2022-01-01 01:00:00",
        "2024-01-01 00:00:00",
        freq="h",
        name="forecast_kst_dtm",
    )
    raw_features, raw_contract = shared._read_stage1_raw_features(
        args.raw_dir, pd.DataFrame(index=full_index)
    )
    context: dict[str, pd.DataFrame] = {}
    context_evidence: dict[str, Any] = {}
    for group in TARGET_COLS:
        frame = compact_context(raw_features[group])
        context[group] = frame
        context_evidence[group] = {
            "rows": len(frame),
            "columns": frame.shape[1],
            "start": frame.index.min(),
            "end": frame.index.max(),
            "columns_exact": list(frame.columns),
            "frame_sha256": shared._frame_sha256(frame),
            "actual_target_scada_baseline_residual_public_columns": 0,
        }
    baseline = bounded._load_stage1_baseline(args.artifact_root)
    segments = bounded._year_segments(2023)
    fit_indexes = {
        "kpx_group_1": strict._year_index(2022),
        "kpx_group_2": strict._year_index(2022),
        "kpx_group_3": segments["H1"],
    }
    application_indexes = {
        "kpx_group_1": segments["full"],
        "kpx_group_2": segments["full"],
        "kpx_group_3": segments["H2"],
    }
    specs = preregister["physical_stage1_inputs"]
    label_path = args.raw_dir / "train/train_labels.csv"

    actions_by_group: dict[str, pd.Series] = {}
    training: dict[str, Any] = {}
    group_artifacts: list[Path] = []
    labels_g12, labels_g12_evidence = bounded._bounded_label_prefix(
        label_path, specs["labels_g12_fit_prefix"]
    )
    for group in TARGET_COLS[:2]:
        action, metadata, paths = _fit_predict_group(
            group=group,
            context=context[group],
            actual=labels_g12[group],
            baseline=baseline[group],
            fit_index=fit_indexes[group],
            application_index=application_indexes[group],
            out_dir=out_dir,
        )
        actions_by_group[group] = action
        training[group] = metadata
        group_artifacts.extend(paths)
    g12_lock_path = out_dir / "stage1_g12_prescore_lock.json"
    strict._write_json(
        g12_lock_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "fit_label_evidence": labels_g12_evidence,
            "application_target_value_cells_materialized": 0,
            "groups": list(TARGET_COLS[:2]),
            "outputs": [describe_file(path) for path in group_artifacts],
            "created_before_g3_fit_prefix_and_all_application_score_labels": True,
        },
    )

    labels_g3, labels_g3_evidence = bounded._bounded_label_prefix(
        label_path, specs["labels_g3_fit_prefix"]
    )
    group = "kpx_group_3"
    action, metadata, g3_paths = _fit_predict_group(
        group=group,
        context=context[group],
        actual=labels_g3[group],
        baseline=baseline[group],
        fit_index=fit_indexes[group],
        application_index=application_indexes[group],
        out_dir=out_dir,
    )
    actions_by_group[group] = action
    training[group] = metadata
    group_artifacts.extend(g3_paths)
    g3_lock_path = out_dir / "stage1_g3_prescore_lock.json"
    strict._write_json(
        g3_lock_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "fit_label_evidence": labels_g3_evidence,
            "g12_omitted_target_value_cells_materialized": 0,
            "application_target_value_cells_materialized": 0,
            "group": group,
            "outputs": [describe_file(path) for path in g3_paths],
            "created_before_all_application_score_labels": True,
        },
    )

    actions = pd.DataFrame(np.nan, index=baseline.index, columns=TARGET_COLS)
    for group in TARGET_COLS:
        actions.loc[application_indexes[group], group] = actions_by_group[group]
    zero = _blend_frame(baseline, actions, application_indexes, 0.0)
    zero_exact = all(
        np.ascontiguousarray(zero[group].to_numpy()).tobytes()
        == np.ascontiguousarray(baseline[group].to_numpy()).tobytes()
        for group in TARGET_COLS
    )
    if not zero_exact:
        raise AssertionError("zero blend is not bit-exact identity")
    blends = {
        key: _blend_frame(baseline, actions, application_indexes, weight)
        for key, weight in BLEND_WEIGHTS.items()
    }
    global_paths: list[Path] = []
    for name, frame in (
        ("stage1_baseline_2023", baseline),
        ("stage1_selected_actions_2023", actions),
    ):
        path = out_dir / f"oof/{name}.parquet"
        strict._atomic_parquet(frame, path)
        global_paths.append(path)
    for key, frame in blends.items():
        path = out_dir / f"oof/stage1_blend_{key}_2023.parquet"
        strict._atomic_parquet(frame, path)
        global_paths.append(path)

    provenance_after = _snapshot_named(provenance_paths)
    input_after = bounded._stage1_input_snapshot(args.raw_dir, args.artifact_root, preregister)
    shared._assert_snapshot_equal(provenance_before, provenance_after, name="Stage1 provenance")
    shared._assert_snapshot_equal(input_before, input_after, name="Stage1 inputs")
    prescore_path = out_dir / "stage1_prescore_record.json"
    strict._write_json(
        prescore_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "census_lock": describe_file(census_path),
            "training": training,
            "raw_physical_prefix_contract": raw_contract,
            "compact_context_contract": context_evidence,
            "group_outputs": [describe_file(path) for path in group_artifacts],
            "global_candidate_outputs": [describe_file(path) for path in global_paths],
            "weight_zero_identity_value_bits_exact": True,
            "all_models_actions_surfaces_and_candidates_hashed": True,
            "application_target_value_cells_materialized": 0,
            "year_2024_value_bytes_read": 0,
            "year_2025_value_bytes_read": 0,
            "sample_value_bytes_read": 0,
            "public_or_scale_artifact_bytes_read": 0,
            "csv_files_created": 0,
            "provenance_before": provenance_before,
            "provenance_after": provenance_after,
            "input_before": input_before,
            "input_after": input_after,
        },
    )
    candidate_paths = [*group_artifacts, *global_paths]
    global_lock_path = out_dir / "stage1_global_prescore_lock.json"
    strict._write_json(
        global_lock_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "prescore_record": describe_file(prescore_path),
            "g12_lock": describe_file(g12_lock_path),
            "g3_lock": describe_file(g3_lock_path),
            "candidate_artifacts": [describe_file(path) for path in candidate_paths],
            "candidate_artifact_count": len(candidate_paths),
            "created_before_any_stage1_application_label_value": True,
            "all_action_and_blend_hashes_frozen": True,
            "year_2024_value_bytes_read": 0,
            "year_2025_value_bytes_read": 0,
            "public_or_scale_artifact_bytes_read": 0,
        },
    )

    score_labels, score_label_evidence = bounded._bounded_label_prefix(
        label_path, specs["labels_stage1_score_prefix_after_prescore_lock"]
    )
    comparisons: dict[str, Any] = {}
    for key in BLEND_WEIGHTS:
        comparisons[key] = {}
        for group in TARGET_COLS:
            app_index = application_indexes[group]
            slice_indexes = (
                {name: segments[name] for name in STAGE1_REQUIRED[group]}
                if group in TARGET_COLS[:2]
                else {"full": segments["H2"], "Q3": segments["Q3"], "Q4": segments["Q4"]}
            )
            comparisons[key][group] = strict._comparison(
                score_labels.loc[app_index, group],
                baseline.loc[app_index, group],
                blends[key].loc[app_index, group],
                group,
                slice_indexes,
            )
    locked_weight, selection = select_stage1_weight(comparisons)
    selected_key = selection["selected"]
    group_passed = {
        group: bool(
            selected_key != "identity"
            and all(
                comparisons[selected_key][group][segment]["delta"] > 0.0
                for segment in STAGE1_REQUIRED[group]
            )
        )
        for group in TARGET_COLS
    }
    result = {
        "schema_version": 1,
        "experiment_id": preregister["experiment_id"],
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "global_prescore_lock": describe_file(global_lock_path),
        "score_label_evidence_after_candidate_hash_lock": score_label_evidence,
        "registered_slice_count": 17,
        "comparisons": comparisons,
        "comparisons_sha256": strict._canonical_sha256(comparisons),
        "selection": selection,
        "selection_sha256": strict._canonical_sha256(selection),
        "locked_weight": locked_weight,
        "locked_candidate": "identity" if locked_weight is None else selected_key,
        "group_passed_stage1": group_passed,
        "stage2_opened": False,
        "year_2024_value_bytes_read": 0,
        "year_2025_value_bytes_read": 0,
        "sample_value_bytes_read": 0,
        "public_or_scale_artifact_bytes_read": 0,
        "csv_files_created": 0,
        "leaderboard_score_claim": False,
    }
    result_path = out_dir / "stage1_results.json"
    strict._write_json(result_path, result)
    promotion_lock_path = out_dir / "stage1_promotion_lock.json"
    strict._write_json(
        promotion_lock_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "stage1_results": describe_file(result_path),
            "comparisons_sha256": result["comparisons_sha256"],
            "selection_sha256": result["selection_sha256"],
            "locked_weight": locked_weight,
            "locked_candidate": result["locked_candidate"],
            "group_passed_stage1": group_passed,
            "nonpassing_groups_forbidden_from_2024": True,
            "all_groups_forbidden_from_2025_sample_and_csv_before_stage2_promotion": True,
            "year_2024_value_bytes_read": 0,
            "year_2025_value_bytes_read": 0,
            "csv_files_created": 0,
        },
    )

    output_files = _all_output_files(out_dir)
    manifest = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "created_utc": utc_now(),
        "command": [sys.executable, *sys.argv],
        "preregister": describe_file(args.preregister),
        "preregister_sha256": PREREGISTER_SHA256,
        "status": "stage1_identity_rejected" if locked_weight is None else "stage1_pass_pending_stage2",
        "result": describe_file(result_path),
        "promotion_lock": describe_file(promotion_lock_path),
        "source_import_closure_count": sum(
            key.startswith("repo_source::") for key in provenance_before
        ),
        "source_and_config_snapshot": provenance_before,
        "census_snapshot": census,
        "input_snapshot": input_before,
        "outputs": [describe_file(path) for path in output_files],
        "output_count_excluding_manifest": len(output_files),
        "runtime": {"packages": package_versions()},
        "git": git_state(PROJECT_DIR),
        "audit_contracts": {
            "preregister_verified_before_fit": True,
            "census_locked_before_fit": True,
            "physical_raw_prefix_only": True,
            "g12_2022_to_2023": True,
            "g3_h1_to_h2": True,
            "all_application_labels_after_candidate_hash_lock": True,
            "all_preregistered_slices_nonempty": True,
            "fitted_model_reload_all_surfaces_bit_exact": True,
            "p6_less_than_or_equal_to_p8_repaired": True,
            "public_and_scale_used": False,
            "year_2024_opened": False,
            "year_2025_or_sample_opened": False,
            "csv_created": False,
            "leaderboard_score_claim": False,
        },
    }
    manifest_path = out_dir / "manifest.json"
    strict._write_json(manifest_path, manifest)
    print(f"Stage1 locked candidate: {result['locked_candidate']}", flush=True)
    print(f"manifest sha256: {sha256_file(manifest_path)}", flush=True)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    preregister = _verify_preregister(args.preregister)
    run_stage1(args, preregister)


if __name__ == "__main__":
    main()

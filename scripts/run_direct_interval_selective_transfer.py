"""Dual locked v2/v3 selective-scale transfer with one shared 2024 read."""

from __future__ import annotations

import argparse
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
    CONTEXT_COLUMNS,
    DirectIntervalProbabilityModel,
    official_action_utility,
)
from src.direct_interval_selective_scale import (  # noqa: E402
    PROMOTED_GROUPS,
    SEGMENTS,
    interpolate_row_utility,
    stage1_group_promotions,
    stage2_promoted,
)
from src.manifest import (  # noqa: E402
    describe_file,
    git_state,
    package_versions,
    sha256_file,
    utc_now,
)
from src.metric import CAPACITY_KWH, TARGET_COLS, score_details  # noqa: E402


V2_SHA = "217ec04c2943bb02fc53a6d8c02ee5ab9c715d9e0d55c0b8b5dae1cbf1391f03"
V3_SHA = "917a197037d3ad1e625805efafa5f5141340546e38a96a18e61da4245383a3fe"
ADDENDUM_SHA = "132e4193adf5ce3d4ea4f7f7d4cef83a1f50fe20f7625fd0485a8ba3161e7dee"
V1_MANIFEST_SHA = "0ac3e2b01820d3ca47f449a7efa79c2a7670c35fcf9c61a7ac23c605088a2b0e"

VARIANT_SPECS: dict[str, dict[str, tuple[float, float] | None]] = {
    "v2": {
        "kpx_group_1": (0.98, 0.01),
        "kpx_group_2": (0.98, 0.01),
        "kpx_group_3": None,
    },
    "v3": {
        "kpx_group_1": (0.98, 0.0125),
        "kpx_group_2": (0.95, 0.0025),
        "kpx_group_3": None,
    },
}
BASELINE_NAMES = ("primary_v3", "interaction_v4")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prescore", "score_final"), required=True)
    parser.add_argument(
        "--raw-dir", type=Path, default=Path(r"data/local/open")
    )
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/cache"))
    parser.add_argument(
        "--v1-dir",
        type=Path,
        default=Path("artifacts/postgate/direct_interval_probability_strict_v1"),
    )
    parser.add_argument(
        "--v2-config", type=Path,
        default=Path("configs/direct_interval_selective_scale_preregister_v2.json"),
    )
    parser.add_argument(
        "--v3-config", type=Path,
        default=Path("configs/direct_interval_public_adaptive_preregister_v3.json"),
    )
    parser.add_argument(
        "--addendum", type=Path,
        default=Path("configs/direct_interval_stage2_interaction_addendum_20260808.json"),
    )
    parser.add_argument(
        "--v2-out", type=Path,
        default=Path("artifacts/postgate/direct_interval_selective_scale_transfer_v2"),
    )
    parser.add_argument(
        "--v3-out", type=Path,
        default=Path("artifacts/postgate/direct_interval_public_adaptive_transfer_v3"),
    )
    return parser.parse_args(argv)


def _verify_hash_sidecar(path: Path, expected: str) -> dict[str, Any]:
    if sha256_file(path) != expected:
        raise AssertionError(f"config hash differs: {path}")
    if path.with_suffix(".sha256").read_text(encoding="utf-8") != f"{expected}  {path.name}\n":
        raise AssertionError(f"config sidecar differs: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_configs(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    v2 = _verify_hash_sidecar(args.v2_config, V2_SHA)
    v3 = _verify_hash_sidecar(args.v3_config, V3_SHA)
    addendum = _verify_hash_sidecar(args.addendum, ADDENDUM_SHA)
    if v2["stage1_posthoc_reconstruction_frozen"]["locked_promoted_groups"] != list(PROMOTED_GROUPS):
        raise AssertionError("v2 promoted groups changed")
    for group, expected in VARIANT_SPECS["v2"].items():
        if expected is None:
            continue
        factor, margin = expected
        candidate = v2["immutable_single_candidate"]
        if float(candidate["scale_factor"]) != factor or float(candidate["utility_advantage_margin"]) != margin:
            raise AssertionError("v2 candidate changed")
    for group in PROMOTED_GROUPS:
        expected = VARIANT_SPECS["v3"][group]
        assert expected is not None
        actual = v3["immutable_group_candidates"][group]
        if (float(actual["factor"]), float(actual["margin"])) != expected:
            raise AssertionError(f"v3 {group} candidate changed")
    for variant in ("v2", "v3"):
        for group in TARGET_COLS:
            registered = addendum["immutable_interaction_rule"][variant][group]
            expected = VARIANT_SPECS[variant][group]
            if expected is None:
                if registered.get("identity") is not True:
                    raise AssertionError("interaction identity changed")
            elif (float(registered["factor"]), float(registered["margin"])) != expected:
                raise AssertionError("interaction candidate changed")
    if sha256_file(args.v1_dir / "manifest.json") != V1_MANIFEST_SHA:
        raise AssertionError("v1 manifest changed")
    return {"v2": v2, "v3": v3, "addendum": addendum}


def _outputs(args: argparse.Namespace) -> dict[str, Path]:
    return {"v2": args.v2_out, "v3": args.v3_out}


def _copy_exclusive(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    with source.open("rb") as source_stream, destination.open("xb") as destination_stream:
        shutil.copyfileobj(source_stream, destination_stream)


def _atomic_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as stream:
            np.savez(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _source_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "runner": Path(__file__).resolve(),
        "v1_model": PROJECT_DIR / "src/direct_interval_probability.py",
        "selective_model": PROJECT_DIR / "src/direct_interval_selective_scale.py",
        "features": PROJECT_DIR / "src/features.py",
        "metric": PROJECT_DIR / "src/metric.py",
        "manifest": PROJECT_DIR / "src/manifest.py",
        "bounded_helper": PROJECT_DIR / "scripts/run_catboost_multiquantile_bayes.py",
        "raw_helper": PROJECT_DIR / "scripts/run_shared_q07_multiseed.py",
        "strict_helper": PROJECT_DIR / "scripts/run_ficr_bayes_decision_strict.py",
        "focused_test": PROJECT_DIR / "tests/test_direct_interval_selective_scale.py",
        "runner_test": PROJECT_DIR / "tests/test_direct_interval_selective_transfer_runner.py",
        "v2_config": args.v2_config.resolve(),
        "v2_sidecar": args.v2_config.with_suffix(".sha256").resolve(),
        "v3_config": args.v3_config.resolve(),
        "v3_sidecar": args.v3_config.with_suffix(".sha256").resolve(),
        "addendum": args.addendum.resolve(),
        "addendum_sidecar": args.addendum.with_suffix(".sha256").resolve(),
    }
    return {name: shared._snapshot_file(path) for name, path in paths.items()}


def _apply_spec(
    baseline: pd.Series,
    utility: np.ndarray,
    *,
    group: str,
    spec: tuple[float, float] | None,
) -> tuple[pd.Series, pd.DataFrame]:
    capacity = CAPACITY_KWH[group]
    values = baseline.to_numpy(dtype=np.float64, copy=False)
    baseline_cf = values / capacity
    if spec is None:
        candidate = baseline.copy()
        diagnostics = pd.DataFrame(
            {
                "baseline_cf": baseline_cf,
                "scaled_cf": baseline_cf,
                "base_utility": np.nan,
                "scaled_utility": np.nan,
                "utility_advantage": np.nan,
                "gate": False,
                "candidate_cf": baseline_cf,
            },
            index=baseline.index,
        )
        return candidate, diagnostics
    factor, margin = spec
    base_utility = interpolate_row_utility(utility, baseline_cf)
    scaled_cf = factor * baseline_cf
    scaled_utility = interpolate_row_utility(utility, scaled_cf)
    advantage = scaled_utility - base_utility
    gate = advantage > margin
    candidate_values = np.where(gate, factor * values, values)
    candidate = pd.Series(
        np.clip(candidate_values, 0.0, 1.02 * capacity),
        index=baseline.index,
        name=baseline.name,
    )
    diagnostics = pd.DataFrame(
        {
            "baseline_cf": baseline_cf,
            "scaled_cf": scaled_cf,
            "base_utility": base_utility,
            "scaled_utility": scaled_utility,
            "utility_advantage": advantage,
            "gate": gate,
            "candidate_cf": candidate.to_numpy() / capacity,
        },
        index=baseline.index,
    )
    return candidate, diagnostics


def _segments(year: int) -> dict[str, pd.DatetimeIndex]:
    return bounded._year_segments(year)


def _mixed_comparisons(
    labels: pd.DataFrame,
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    segments: Mapping[str, pd.DatetimeIndex],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in SEGMENTS:
        index = segments[name]
        base = score_details(labels.loc[index], baseline.loc[index]).as_dict()
        cand = score_details(labels.loc[index], candidate.loc[index]).as_dict()
        result[name] = {
            "baseline": base,
            "candidate": cand,
            "delta_total_score": cand["total_score"] - base["total_score"],
            "delta_one_minus_nmae": cand["one_minus_nmae"] - base["one_minus_nmae"],
            "delta_ficr": cand["ficr"] - base["ficr"],
        }
    return result


def _read_context_cache(path: Path, *, expected_sha: str, expected_bytes: int) -> pd.DataFrame:
    if path.stat().st_size != expected_bytes or sha256_file(path) != expected_sha:
        raise AssertionError(f"cache identity changed: {path}")
    frame = pd.read_parquet(path, columns=list(CONTEXT_COLUMNS))
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    expected_index = pd.date_range(
        "2022-01-01 01:00", "2025-01-01 00:00", freq="h", name="forecast_kst_dtm"
    )
    if not frame.index.equals(expected_index) or tuple(frame.columns) != CONTEXT_COLUMNS:
        raise AssertionError(f"train cache schema/index changed: {path}")
    frame = frame.astype(np.float32)
    if not np.isfinite(frame.to_numpy()).all():
        raise AssertionError("cache context contains non-finite values")
    return frame


def _read_test_context(path: Path, *, expected_sha: str, expected_bytes: int) -> pd.DataFrame:
    if path.stat().st_size != expected_bytes or sha256_file(path) != expected_sha:
        raise AssertionError(f"test cache identity changed: {path}")
    frame = pd.read_parquet(path, columns=list(CONTEXT_COLUMNS))
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    expected_index = strict._year_index(2025)
    if not frame.index.equals(expected_index) or tuple(frame.columns) != CONTEXT_COLUMNS:
        raise AssertionError("test cache schema/index changed")
    frame = frame.astype(np.float32)
    if not np.isfinite(frame.to_numpy()).all():
        raise AssertionError("test context contains non-finite values")
    return frame


def _surface_arrays(model: DirectIntervalProbabilityModel, context: pd.DataFrame) -> dict[str, np.ndarray]:
    surface = model.predict_surfaces(context)
    arrays = {
        "p6_raw": np.asarray(surface["p6_raw"], dtype=np.float64),
        "p8_raw": np.asarray(surface["p8_raw"], dtype=np.float64),
        "p6": np.asarray(surface["p6"], dtype=np.float64),
        "p8": np.asarray(surface["p8"], dtype=np.float64),
        "eae": np.asarray(surface["eae"], dtype=np.float64),
    }
    arrays["utility"] = official_action_utility(arrays["p6"], arrays["p8"], arrays["eae"])
    arrays["repair_count"] = np.asarray([surface["repair_count"]], dtype=np.int64)
    return arrays


def _assert_arrays_equal(first: Mapping[str, np.ndarray], second: Mapping[str, np.ndarray]) -> None:
    if set(first) != set(second):
        raise AssertionError("surface array names differ")
    for name in first:
        if not np.array_equal(first[name], second[name]):
            raise AssertionError(f"surface array differs: {name}")


def _prepare_stage1(
    args: argparse.Namespace,
    configs: Mapping[str, Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    outputs = _outputs(args)
    for variant, out_dir in outputs.items():
        if out_dir.exists():
            raise FileExistsError(out_dir)
        out_dir.mkdir(parents=True)
        config_path = args.v2_config if variant == "v2" else args.v3_config
        _copy_exclusive(config_path, out_dir / "preregister.json")
        _copy_exclusive(config_path.with_suffix(".sha256"), out_dir / "preregister.sha256")
        _copy_exclusive(args.addendum, out_dir / "interaction_addendum.json")
        _copy_exclusive(args.addendum.with_suffix(".sha256"), out_dir / "interaction_addendum.sha256")

    baseline = pd.read_parquet(args.v1_dir / "oof/stage1_baseline_2023.parquet")
    segments = _segments(2023)
    application = {
        "kpx_group_1": segments["full"],
        "kpx_group_2": segments["full"],
        "kpx_group_3": segments["H2"],
    }
    utility = {
        group: np.load(args.v1_dir / f"oof/stage1_{group}_surfaces.npz")["utility"]
        for group in TARGET_COLS
    }
    candidate_frames: dict[str, pd.DataFrame] = {}
    diagnostics: dict[str, dict[str, pd.DataFrame]] = {}
    for variant, out_dir in outputs.items():
        candidate = baseline.copy()
        diagnostics[variant] = {}
        candidate_paths: list[Path] = []
        for group in TARGET_COLS:
            index = application[group]
            # v2 evaluates G3 to establish its failed group lock; v3 fixes G3 identity.
            spec = (
                (0.98, 0.01)
                if variant == "v2" and group == "kpx_group_3"
                else VARIANT_SPECS[variant][group]
            )
            selected, detail = _apply_spec(
                baseline.loc[index, group], utility[group], group=group, spec=spec
            )
            candidate.loc[index, group] = selected
            diagnostics[variant][group] = detail
            path = out_dir / f"stage1/{group}_diagnostics.parquet"
            strict._atomic_parquet(detail, path)
            candidate_paths.append(path)
        baseline_path = out_dir / "stage1/baseline_2023.parquet"
        candidate_path = out_dir / "stage1/candidate_2023.parquet"
        strict._atomic_parquet(baseline, baseline_path)
        strict._atomic_parquet(candidate, candidate_path)
        candidate_paths.extend((baseline_path, candidate_path))
        candidate_frames[variant] = candidate
        strict._write_json(
            out_dir / "stage1_candidate_hash_lock.json",
            {
                "schema_version": 1,
                "created_utc": utc_now(),
                "variant": variant,
                "config_sha256": V2_SHA if variant == "v2" else V3_SHA,
                "addendum_sha256": ADDENDUM_SHA,
                "candidate_outputs": [describe_file(path) for path in candidate_paths],
                "created_before_any_2024_value_read": True,
                "created_before_stage1_metric_reconstruction_in_this_runner": True,
            },
        )

    v1_config = json.loads((PROJECT_DIR / "configs/direct_interval_probability_preregister_v1.json").read_text(encoding="utf-8"))
    labels, label_evidence = bounded._bounded_label_prefix(
        args.raw_dir / "train/train_labels.csv",
        v1_config["physical_stage1_inputs"]["labels_stage1_score_prefix_after_prescore_lock"],
    )
    stage1_locks: dict[str, Any] = {}
    required = {
        "kpx_group_1": SEGMENTS,
        "kpx_group_2": SEGMENTS,
        "kpx_group_3": ("full", "Q3", "Q4"),
    }
    for variant, out_dir in outputs.items():
        comparisons: dict[str, Any] = {}
        for group in TARGET_COLS:
            index = application[group]
            names = required[group]
            slices = {
                name: (segments["H2"] if group == "kpx_group_3" and name == "full" else segments[name])
                for name in names
            }
            comparisons[group] = strict._comparison(
                labels.loc[index, group],
                baseline.loc[index, group],
                candidate_frames[variant].loc[index, group],
                group,
                slices,
            )
        if variant == "v2":
            passed, failed, audit = stage1_group_promotions(comparisons, required)
            if passed != PROMOTED_GROUPS or failed != ("kpx_group_3",):
                raise AssertionError("v2 reconstructed group lock differs")
            expected_deltas = configs["v2"]["stage1_posthoc_reconstruction_frozen"]["group_deltas_in_exact_segment_order"]
            expected_counts = configs["v2"]["stage1_posthoc_reconstruction_frozen"]["selected_row_counts"]
            for group in TARGET_COLS:
                if int(diagnostics[variant][group]["gate"].sum()) != int(expected_counts[group]):
                    raise AssertionError("v2 selected row count differs")
                for name, expected in expected_deltas[group].items():
                    if not np.isclose(comparisons[group][name]["delta"], expected, atol=2e-15, rtol=0.0):
                        raise AssertionError(f"v2 Stage1 delta differs: {group}/{name}")
        else:
            passed = PROMOTED_GROUPS
            failed = ("kpx_group_3",)
            audit = {}
            expected = configs["v3"]["stage1_posthoc_results_frozen"]
            for group in PROMOTED_GROUPS:
                if int(diagnostics[variant][group]["gate"].sum()) != int(
                    configs["v3"]["immutable_group_candidates"][group]["stage1_selected_rows"]
                ):
                    raise AssertionError("v3 selected row count differs")
                for name, expected_delta in expected[group]["deltas"].items():
                    if not np.isclose(comparisons[group][name]["delta"], expected_delta, atol=2e-15, rtol=0.0):
                        raise AssertionError(f"v3 Stage1 delta differs: {group}/{name}")
            if not np.array_equal(
                candidate_frames[variant].loc[application["kpx_group_3"], "kpx_group_3"].to_numpy(),
                baseline.loc[application["kpx_group_3"], "kpx_group_3"].to_numpy(),
            ):
                raise AssertionError("v3 G3 identity differs")
        result_path = out_dir / "stage1_results.json"
        strict._write_json(
            result_path,
            {
                "schema_version": 1,
                "created_utc": utc_now(),
                "variant": variant,
                "comparisons": comparisons,
                "comparisons_sha256": strict._canonical_sha256(comparisons),
                "promotion_audit": audit,
                "locked_promoted_groups": list(passed),
                "locked_identity_groups": list(failed),
                "posthoc_2023_selected": True,
                "selection_unsafe": True,
                "public_adaptive": variant == "v3",
                "score_label_evidence": label_evidence,
                "year_2024_value_bytes_read": 0,
                "year_2024_label_value_cells_parsed": 0,
            },
        )
        lock_path = out_dir / "stage1_promotion_lock.json"
        strict._write_json(
            lock_path,
            {
                "schema_version": 1,
                "created_utc": utc_now(),
                "variant": variant,
                "config_sha256": V2_SHA if variant == "v2" else V3_SHA,
                "addendum_sha256": ADDENDUM_SHA,
                "stage1_results": describe_file(result_path),
                "locked_promoted_groups": list(passed),
                "locked_identity_groups": list(failed),
                "no_2024_reselection": True,
                "year_2024_label_value_cells_parsed": 0,
            },
        )
        stage1_locks[variant] = describe_file(lock_path)
    for variant, out_dir in outputs.items():
        strict._write_json(
            out_dir / "dual_stage1_before_2024_lock.json",
            {
                "schema_version": 1,
                "created_utc": utc_now(),
                "v2_config_sha256": V2_SHA,
                "v3_config_sha256": V3_SHA,
                "addendum_sha256": ADDENDUM_SHA,
                "both_stage1_locks": stage1_locks,
                "both_candidates_frozen_before_2024_weather_prediction_or_label_value_decode": True,
                "source_snapshot": provenance,
            },
        )
    return {"baseline": baseline, "stage1_locks": stage1_locks}


def _prepare_stage2(
    args: argparse.Namespace,
    configs: Mapping[str, Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> None:
    outputs = _outputs(args)
    source_spec = json.loads((PROJECT_DIR / "configs/direct_interval_probability_preregister_v1.json").read_text(encoding="utf-8"))["physical_stage1_inputs"]["labels_stage1_score_prefix_after_prescore_lock"]
    source_labels, source_evidence = bounded._bounded_label_prefix(
        args.raw_dir / "train/train_labels.csv", source_spec
    )
    cache_specs = configs["v2"]["stage2_known_input_identities_not_read_by_v2_before_freeze"]["train_cache"]
    contexts = {
        group: _read_context_cache(
            args.cache_dir / f"{group}_weather_train.parquet",
            expected_sha=cache_specs[group]["sha256"],
            expected_bytes=int(cache_specs[group]["bytes"]),
        )
        for group in PROMOTED_GROUPS
    }
    index_2024 = strict._year_index(2024)
    baseline_paths = {
        "primary_v3": args.artifact_root / "oof/gate2024_locked_v3_cf_fix.parquet",
        "interaction_v4": args.artifact_root / "oof/gate2024_recent_v4_cf_fix_calibration_fit.parquet",
    }
    expected_baseline = {
        "primary_v3": (334535, "9ff992c2d3fa3ce0725600fda415f98bd9a9b114cf6cb259df6f0b796005b76a"),
        "interaction_v4": (335710, "47d09dfca7119bf8d1298a609ee47fc8280bd9c0acfac5966f98d4d3e52b04e9"),
    }
    baselines: dict[str, pd.DataFrame] = {}
    for name, path in baseline_paths.items():
        size, digest = expected_baseline[name]
        if path.stat().st_size != size or sha256_file(path) != digest:
            raise AssertionError(f"Stage2 baseline identity changed: {name}")
        frame = pd.read_parquet(path)
        frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
        if not frame.index.equals(index_2024) or tuple(frame.columns) != TARGET_COLS:
            raise AssertionError(f"Stage2 baseline schema/index changed: {name}")
        baselines[name] = frame.astype(float)

    shared_arrays: dict[str, dict[str, np.ndarray]] = {}
    model_metadata: dict[str, Any] = {}
    for group in PROMOTED_GROUPS:
        fit_index = source_labels.index
        model = DirectIntervalProbabilityModel().fit(
            contexts[group].loc[fit_index],
            source_labels[group],
            capacity_kwh=CAPACITY_KWH[group],
        )
        arrays = _surface_arrays(model, contexts[group].loc[index_2024])
        shared_arrays[group] = arrays
        model_metadata[group] = model.metadata()
        v2_model = args.v2_out / f"stage2/models/{group}.joblib"
        v2_surface = args.v2_out / f"stage2/surfaces/{group}.npz"
        strict._atomic_joblib(model, v2_model)
        _atomic_npz(v2_surface, arrays)
        v3_model = args.v3_out / f"stage2/models/{group}.joblib"
        v3_surface = args.v3_out / f"stage2/surfaces/{group}.npz"
        _copy_exclusive(v2_model, v3_model)
        _copy_exclusive(v2_surface, v3_surface)
        reloaded: DirectIntervalProbabilityModel = joblib.load(v2_model)
        rebuilt = _surface_arrays(reloaded, contexts[group].loc[index_2024])
        _assert_arrays_equal(arrays, rebuilt)
        if sha256_file(v2_model) != sha256_file(v3_model) or sha256_file(v2_surface) != sha256_file(v3_surface):
            raise AssertionError("shared model/surface copies differ")

    prescore_locks: dict[str, Any] = {}
    for variant, out_dir in outputs.items():
        all_paths: list[Path] = []
        for baseline_name, baseline in baselines.items():
            candidate = baseline.copy()
            for group in PROMOTED_GROUPS:
                selected, diagnostics = _apply_spec(
                    baseline[group],
                    shared_arrays[group]["utility"],
                    group=group,
                    spec=VARIANT_SPECS[variant][group],
                )
                candidate[group] = selected
                path = out_dir / f"stage2/{baseline_name}/{group}_diagnostics.parquet"
                strict._atomic_parquet(diagnostics, path)
                all_paths.append(path)
            if not np.array_equal(candidate["kpx_group_3"].to_numpy(), baseline["kpx_group_3"].to_numpy()):
                raise AssertionError("G3 identity differs")
            baseline_path = out_dir / f"stage2/{baseline_name}/baseline_2024.parquet"
            candidate_path = out_dir / f"stage2/{baseline_name}/candidate_2024.parquet"
            strict._atomic_parquet(baseline, baseline_path)
            strict._atomic_parquet(candidate, candidate_path)
            all_paths.extend((baseline_path, candidate_path))
        for group in PROMOTED_GROUPS:
            all_paths.extend(
                (
                    out_dir / f"stage2/models/{group}.joblib",
                    out_dir / f"stage2/surfaces/{group}.npz",
                )
            )
        record_path = out_dir / "stage2_prescore_record.json"
        strict._write_json(
            record_path,
            {
                "schema_version": 1,
                "created_utc": utc_now(),
                "variant": variant,
                "config_sha256": V2_SHA if variant == "v2" else V3_SHA,
                "addendum_sha256": ADDENDUM_SHA,
                "stage1_lock": describe_file(out_dir / "stage1_promotion_lock.json"),
                "source_label_evidence_pre2024_only": source_evidence,
                "model_metadata": model_metadata,
                "outputs": [describe_file(path) for path in all_paths],
                "same_fitted_models_and_surfaces_used_for_both_baselines": True,
                "model_reload_surfaces_bit_exact": True,
                "G3_identity_both_baselines": True,
                "year_2024_label_value_cells_parsed": 0,
                "full_label_file_hash_deferred_until_after_joint_lock": True,
                "source_snapshot": provenance,
            },
        )
        lock_path = out_dir / "stage2_dual_baseline_prescore_lock.json"
        strict._write_json(
            lock_path,
            {
                "schema_version": 1,
                "created_utc": utc_now(),
                "variant": variant,
                "config_sha256": V2_SHA if variant == "v2" else V3_SHA,
                "addendum_sha256": ADDENDUM_SHA,
                "prescore_record": describe_file(record_path),
                "candidate_artifacts": [describe_file(path) for path in all_paths],
                "both_authoritative_baselines_frozen": list(BASELINE_NAMES),
                "created_before_any_2024_label_byte_or_value_read": True,
                "factor_margin_lookup_groups_locked": True,
            },
        )
        prescore_locks[variant] = describe_file(lock_path)
    for variant, out_dir in outputs.items():
        strict._write_json(
            out_dir / "joint_before_2024_label_lock.json",
            {
                "schema_version": 1,
                "created_utc": utc_now(),
                "v2_config_sha256": V2_SHA,
                "v3_config_sha256": V3_SHA,
                "addendum_sha256": ADDENDUM_SHA,
                "both_variant_prescore_locks": prescore_locks,
                "single_shared_2024_label_read_allowed_after_this_lock": True,
                "year_2024_label_bytes_read_before_lock": 0,
                "year_2024_label_value_cells_parsed_before_lock": 0,
            },
        )


def run_prescore(args: argparse.Namespace, configs: Mapping[str, Mapping[str, Any]]) -> None:
    provenance = _source_snapshot(args)
    _prepare_stage1(args, configs, provenance)
    _prepare_stage2(args, configs, provenance)
    for variant, out_dir in _outputs(args).items():
        path = out_dir / "joint_before_2024_label_lock.json"
        print(f"{variant} joint prescore lock sha256: {sha256_file(path)}", flush=True)
    print("2024 labels remain unread", flush=True)


def _load_full_labels_after_locks(args: argparse.Namespace, configs: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    for out_dir in _outputs(args).values():
        if not (out_dir / "joint_before_2024_label_lock.json").is_file():
            raise AssertionError("joint prescore lock missing")
    spec = configs["v2"]["stage2_known_input_identities_not_read_by_v2_before_freeze"]["full_train_labels_after_prescore_lock"]
    path = args.raw_dir / "train/train_labels.csv"
    if path.stat().st_size != int(spec["bytes"]) or sha256_file(path) != spec["sha256"]:
        raise AssertionError("full label identity changed")
    labels = strict._read_full_labels(path)
    if len(labels) != 26304 or labels.index.max() != pd.Timestamp("2025-01-01 00:00:00"):
        raise AssertionError("full label rows/end changed")
    return labels


def _score_variant_stage2(
    *,
    out_dir: Path,
    labels: pd.DataFrame,
    variant: str,
) -> dict[str, Any]:
    segments = _segments(2024)
    index = segments["full"]
    baseline_results: dict[str, Any] = {}
    both_pass = True
    for baseline_name in BASELINE_NAMES:
        baseline = pd.read_parquet(out_dir / f"stage2/{baseline_name}/baseline_2024.parquet")
        candidate = pd.read_parquet(out_dir / f"stage2/{baseline_name}/candidate_2024.parquet")
        group_comparisons = {
            group: strict._comparison(
                labels.loc[index, group],
                baseline[group],
                candidate[group],
                group,
                {name: segments[name] for name in SEGMENTS},
            )
            for group in PROMOTED_GROUPS
        }
        mixed = _mixed_comparisons(labels.loc[index], baseline, candidate, segments)
        promoted, audit = stage2_promoted(group_comparisons, mixed)
        baseline_results[baseline_name] = {
            "group_comparisons": group_comparisons,
            "mixed_comparisons": mixed,
            "promotion_audit": audit,
            "passed": promoted,
        }
        both_pass = both_pass and promoted
    return {
        "schema_version": 1,
        "created_utc": utc_now(),
        "variant": variant,
        "stage2_label_read_after_joint_lock": True,
        "baseline_results": baseline_results,
        "primary_passed": baseline_results["primary_v3"]["passed"],
        "interaction_passed": baseline_results["interaction_v4"]["passed"],
        "dual_baseline_promoted": both_pass,
        "no_retune_retry_or_candidate_switch": True,
        "posthoc_2023_selected": True,
        "selection_unsafe": True,
        "public_adaptive": variant == "v3",
    }


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    strict._atomic_csv(frame, path)


def _finalize_passing(
    args: argparse.Namespace,
    configs: Mapping[str, Mapping[str, Any]],
    labels: pd.DataFrame,
    passing: Sequence[str],
) -> None:
    if not passing:
        return
    cache_specs = configs["v2"]["stage2_known_input_identities_not_read_by_v2_before_freeze"]["train_cache"]
    train_context = {
        group: _read_context_cache(
            args.cache_dir / f"{group}_weather_train.parquet",
            expected_sha=cache_specs[group]["sha256"],
            expected_bytes=int(cache_specs[group]["bytes"]),
        )
        for group in PROMOTED_GROUPS
    }
    final_spec = configs["v2"]["final_if_and_only_if_stage2_passes"]
    test_context = {
        group: _read_test_context(
            args.cache_dir / f"{group}_weather_test.parquet",
            expected_sha=final_spec["test_cache"][group]["sha256"],
            expected_bytes=int(final_spec["test_cache"][group]["bytes"]),
        )
        for group in PROMOTED_GROUPS
    }
    baseline_spec = final_spec["baseline"]
    baseline_path = PROJECT_DIR / baseline_spec["path"]
    if baseline_path.stat().st_size != int(baseline_spec["bytes"]) or sha256_file(baseline_path) != baseline_spec["sha256"]:
        raise AssertionError("final baseline identity changed")
    baseline = pd.read_parquet(baseline_path)
    baseline.index = pd.DatetimeIndex(baseline.index, name="forecast_kst_dtm")
    test_index = strict._year_index(2025)
    if not baseline.index.equals(test_index) or tuple(baseline.columns) != TARGET_COLS:
        raise AssertionError("final baseline schema/index changed")
    sample_spec = final_spec["sample"]
    sample_path = Path(sample_spec["path"])
    if sample_path.stat().st_size != int(sample_spec["bytes"]) or sha256_file(sample_path) != sample_spec["sha256"]:
        raise AssertionError("sample identity changed")
    sample = pd.read_csv(
        sample_path,
        encoding="utf-8-sig",
        dtype={"forecast_id": "string", "forecast_kst_dtm": "string"},
    )
    if tuple(sample.columns) != ("forecast_id", "forecast_kst_dtm", *TARGET_COLS):
        raise AssertionError("sample columns changed")
    if not pd.DatetimeIndex(pd.to_datetime(sample["forecast_kst_dtm"]), name="forecast_kst_dtm").equals(test_index):
        raise AssertionError("sample time index changed")

    final_arrays: dict[str, dict[str, np.ndarray]] = {}
    final_models: dict[str, DirectIntervalProbabilityModel] = {}
    for group in PROMOTED_GROUPS:
        model = DirectIntervalProbabilityModel().fit(
            train_context[group], labels[group], capacity_kwh=CAPACITY_KWH[group]
        )
        final_models[group] = model
        final_arrays[group] = _surface_arrays(model, test_context[group])
    for variant in passing:
        out_dir = _outputs(args)[variant]
        candidate = baseline.copy()
        paths: list[Path] = []
        for group in PROMOTED_GROUPS:
            model_path = out_dir / f"final/models/{group}.joblib"
            surface_path = out_dir / f"final/surfaces/{group}.npz"
            strict._atomic_joblib(final_models[group], model_path)
            _atomic_npz(surface_path, final_arrays[group])
            selected, diagnostics = _apply_spec(
                baseline[group],
                final_arrays[group]["utility"],
                group=group,
                spec=VARIANT_SPECS[variant][group],
            )
            candidate[group] = selected
            diagnostics_path = out_dir / f"final/{group}_diagnostics.parquet"
            strict._atomic_parquet(diagnostics, diagnostics_path)
            paths.extend((model_path, surface_path, diagnostics_path))
        if not np.array_equal(candidate["kpx_group_3"].to_numpy(), baseline["kpx_group_3"].to_numpy()):
            raise AssertionError("final G3 identity differs")
        prediction_path = out_dir / "final/predictions.parquet"
        strict._atomic_parquet(candidate, prediction_path)
        paths.append(prediction_path)
        submission = sample.copy()
        for group in TARGET_COLS:
            submission[group] = candidate[group].to_numpy(dtype=np.float64)
        csv_name = (
            "direct_interval_selective_scale_v4_2025.csv"
            if variant == "v2"
            else "direct_interval_public_adaptive_v4_2025.csv"
        )
        csv_path = out_dir / csv_name
        _atomic_csv(submission, csv_path)
        readback = pd.read_csv(
            csv_path,
            encoding="utf-8-sig",
            dtype={"forecast_id": "string", "forecast_kst_dtm": "string"},
        )
        if not readback[["forecast_id", "forecast_kst_dtm"]].equals(sample[["forecast_id", "forecast_kst_dtm"]]):
            raise AssertionError("submission id/time readback differs")
        values = readback.loc[:, list(TARGET_COLS)].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise AssertionError("submission contains non-finite values")
        for position, group in enumerate(TARGET_COLS):
            if np.any(values[:, position] < 0.0) or np.any(values[:, position] > 1.02 * CAPACITY_KWH[group] + 5e-7):
                raise AssertionError("submission outside capacity bounds")
        paths.append(csv_path)
        strict._write_json(
            out_dir / "final_results.json",
            {
                "schema_version": 1,
                "created_utc": utc_now(),
                "executed": True,
                "variant": variant,
                "dual_baseline_promoted": True,
                "baseline": describe_file(baseline_path),
                "sample": describe_file(sample_path),
                "outputs": [describe_file(path) for path in paths],
                "submission": describe_file(csv_path),
                "G3_identity_value_bits_exact": True,
                "prepreregister_2025_identity_hash_bytes_read_only_incident": True,
                "2025_values_parsed_only_after_dual_stage2_promotion": True,
                "leaderboard_score_claim": False,
            },
        )


def _write_manifest(
    args: argparse.Namespace,
    configs: Mapping[str, Mapping[str, Any]],
    variant: str,
    result: Mapping[str, Any],
) -> None:
    out_dir = _outputs(args)[variant]
    files = sorted(
        [path for path in out_dir.rglob("*") if path.is_file() and path.name != "manifest.json"],
        key=lambda path: path.relative_to(out_dir).as_posix(),
    )
    source_now = _source_snapshot(args)
    prescore_source = json.loads((out_dir / "stage2_prescore_record.json").read_text(encoding="utf-8"))["source_snapshot"]
    shared._assert_snapshot_equal(prescore_source, source_now, name=f"{variant} source/config")
    manifest = {
        "schema_version": 1,
        "artifact_type": "direct_interval_selective_transfer_v2" if variant == "v2" else "direct_interval_public_adaptive_transfer_v3",
        "created_utc": utc_now(),
        "variant": variant,
        "config_sha256": V2_SHA if variant == "v2" else V3_SHA,
        "addendum_sha256": ADDENDUM_SHA,
        "risk": {
            "posthoc_2023_selected": True,
            "selection_unsafe": True,
            "public_adaptive": variant == "v3",
            "strict_final_isolation": False,
            "private_champion": False,
            "leaderboard_score_claim": False,
        },
        "incidents": {
            "misaligned_g3_exploration_discarded": True,
            "prepreregister_2025_identity_hash_bytes_read_only": True,
            "preaddendum_2024_prediction_identity_hash_bytes_read_only": True,
            "2024_prediction_values_decoded_only_after_addendum_and_stage1_locks": True,
            "2024_labels_read_only_after_both_variant_dual_baseline_prescore_locks": True,
        },
        "stage2_result": describe_file(out_dir / "stage2_results.json"),
        "dual_baseline_promoted": bool(result["dual_baseline_promoted"]),
        "source_config_snapshot": source_now,
        "outputs": [describe_file(path) for path in files],
        "output_count_excluding_manifest": len(files),
        "runtime": {"packages": package_versions()},
        "git": git_state(PROJECT_DIR),
    }
    strict._write_json(out_dir / "manifest.json", manifest)


def run_score_final(args: argparse.Namespace, configs: Mapping[str, Mapping[str, Any]]) -> None:
    labels = _load_full_labels_after_locks(args, configs)
    results: dict[str, Any] = {}
    passing: list[str] = []
    for variant, out_dir in _outputs(args).items():
        result = _score_variant_stage2(
            out_dir=out_dir, labels=labels, variant=variant
        )
        result_path = out_dir / "stage2_results.json"
        strict._write_json(result_path, result)
        strict._write_json(
            out_dir / "stage2_promotion_lock.json",
            {
                "schema_version": 1,
                "created_utc": utc_now(),
                "variant": variant,
                "config_sha256": V2_SHA if variant == "v2" else V3_SHA,
                "addendum_sha256": ADDENDUM_SHA,
                "joint_prescore_lock": describe_file(out_dir / "joint_before_2024_label_lock.json"),
                "stage2_results": describe_file(result_path),
                "primary_passed": result["primary_passed"],
                "interaction_passed": result["interaction_passed"],
                "dual_baseline_promoted": result["dual_baseline_promoted"],
                "no_retune_retry_or_candidate_switch": True,
                "csv_allowed": result["dual_baseline_promoted"],
            },
        )
        results[variant] = result
        if result["dual_baseline_promoted"]:
            passing.append(variant)
        else:
            strict._write_json(
                out_dir / "final_results.json",
                {
                    "schema_version": 1,
                    "created_utc": utc_now(),
                    "executed": False,
                    "variant": variant,
                    "dual_baseline_promoted": False,
                    "test_weather_value_cells_parsed_after_promotion": 0,
                    "sample_value_bytes_read_after_promotion": 0,
                    "csv_created": False,
                    "reason": "one or both immutable Stage2 baseline gates failed",
                },
            )
    _finalize_passing(args, configs, labels, passing)
    for variant, result in results.items():
        _write_manifest(args, configs, variant, result)
        manifest_path = _outputs(args)[variant] / "manifest.json"
        print(
            f"{variant}: primary={result['primary_passed']} interaction={result['interaction_passed']} "
            f"promoted={result['dual_baseline_promoted']} manifest={sha256_file(manifest_path)}",
            flush=True,
        )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    configs = _verify_configs(args)
    if args.stage == "prescore":
        run_prescore(args, configs)
    else:
        run_score_final(args, configs)


if __name__ == "__main__":
    main()

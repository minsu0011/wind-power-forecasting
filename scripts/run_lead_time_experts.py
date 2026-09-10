"""Strict-forward four-bin NWP forecast lead-time expert experiment."""

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
    G3_STAGE1_START,
    STAGE1_END,
    STAGE1_START,
    STAGE2_END,
    STAGE2_START,
    _atomic_joblib,
    _atomic_parquet,
    _comparisons,
    _frame_sha256,
    _interval,
    _mapping_sha256,
    _read_stage1_baseline,
    _reconstruct_baseline,
    _write_submission,
)
from scripts.run_shared_q07_multiseed import (  # noqa: E402
    EXPECTED_ROWS_PRE2024,
    EXPECTED_ROWS_THROUGH2024,
    EXPECTED_STAGE1_NEXT_TIMESTAMP,
    EXPECTED_TEST_ROWS,
    FINAL_COMPONENT_FILES,
    GATE_COMPONENT_FILES,
    OFFICIAL_LABEL_PREFIX_SHA256,
    OFFICIAL_NWP_PREFIX_SHA256,
    OFFICIAL_STAGE1_PREFIX_BYTES,
    YEAR_2022_START,
    YEAR_2023_END,
    YEAR_2024_END,
    _read_features,
    _read_bounded_weather_csv,
    _read_labels,
    _read_prediction,
    _read_stage1_raw_features,
    _read_test_features,
    _stage1_input_snapshot as _strict_stage1_input_snapshot,
)
from src.lead_time_experts import (  # noqa: E402
    choose_registered_candidate,
    fit_global_and_lead_experts,
    specialization_candidate,
    validate_lead_partition,
)
from src.features import AVAILABLE_COL, TIME_COL  # noqa: E402
from src.manifest import (  # noqa: E402
    describe_file,
    git_state,
    package_versions,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from src.metric import CAPACITY_KWH, TARGET_COLS  # noqa: E402


PREREGISTER_SHA256 = "d469bbc0f20910f40ec4dc5f5e8a2d62b0174067fb0af9f5c4197162be5a7130"
SUPERSEDED_V1_SHA256 = "42b972d907b9b58a78c32fab673f1679e51ecc6b19c82678aaa52c5647c6bc91"
RECIPE_IDS = ("expert_l1", "expert_q07")
WEIGHTS = (0.05, 0.10, 0.20)
STAGE2_SLICES = {
    "full": ("2024-01-01T01:00:00", "2025-01-01T00:00:00"),
    "H1": ("2024-01-01T01:00:00", "2024-07-01T00:00:00"),
    "H2": ("2024-07-01T01:00:00", "2025-01-01T00:00:00"),
    "Q1": ("2024-01-01T01:00:00", "2024-04-01T00:00:00"),
    "Q2": ("2024-04-01T01:00:00", "2024-07-01T00:00:00"),
    "Q3": ("2024-07-01T01:00:00", "2024-10-01T00:00:00"),
    "Q4": ("2024-10-01T01:00:00", "2025-01-01T00:00:00"),
}


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
        default=PROJECT_DIR / "configs" / "lead_time_experts_preregister_v2.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_DIR / "artifacts" / "postgate" / "lead_time_experts",
    )
    parser.add_argument("--stage", choices=("stage1", "stage2", "all"), default="all")
    return parser.parse_args(argv)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    write_json_atomic(path, _json_ready(payload), overwrite=False)


def _load_preregister(path: Path) -> dict[str, Any]:
    observed = sha256_file(path)
    if observed != PREREGISTER_SHA256:
        raise AssertionError(f"lead-time preregister SHA changed: {observed}")
    overlay = json.loads(path.read_text(encoding="utf-8"))
    if overlay.get("experiment_id") != "lead_time_experts_strict_forward_v2":
        raise AssertionError("lead-time experiment id changed")
    inherited = overlay["inherited_contract"]
    base_path = PROJECT_DIR / inherited["path"]
    if inherited["sha256"] != SUPERSEDED_V1_SHA256:
        raise AssertionError("v2 inherited v1 hash declaration changed")
    if sha256_file(base_path) != SUPERSEDED_V1_SHA256:
        raise AssertionError("superseded v1 preregister changed")
    if overlay["supersedes"]["status"] != "superseded_before_any_candidate_fit_or_score":
        raise AssertionError("v1 supersession status changed")
    payload = json.loads(base_path.read_text(encoding="utf-8"))
    payload["schema_version"] = 2
    payload["experiment_id"] = overlay["experiment_id"]
    payload["status"] = overlay["status"]
    payload["interpretation"].update(overlay["interpretation"])
    payload["target_scale_provenance"] = overlay["target_scale_provenance"]
    payload["global_control_override"] = overlay["global_control_override"]
    payload["access_override"] = overlay["access_override"]
    payload["preexecution_compatibility_audit"] = overlay[
        "preexecution_compatibility_audit"
    ]
    payload["supersedes"] = overlay["supersedes"]
    payload["_v2_overlay"] = overlay
    recipes = tuple(item["id"] for item in payload["models"]["recipes"])
    if recipes != RECIPE_IDS:
        raise AssertionError("registered lead-time recipes changed")
    if tuple(map(float, payload["candidate_family"]["blend_weights"])) != WEIGHTS:
        raise AssertionError("registered weights changed")
    if int(payload["candidate_family"]["candidate_count"]) != 6:
        raise AssertionError("registered candidate count changed")
    bins = payload["feature_contract"]["lead_bins"]
    observed_bins = tuple(
        (int(item["lower_inclusive"]), int(item["upper_exclusive"])) for item in bins
    )
    if observed_bins != ((12, 18), (18, 24), (24, 30), (30, 36)):
        raise AssertionError("registered lead bins changed")
    return payload


def _verify_locked_recipe(path: Path, preregister: Mapping[str, Any]) -> dict[str, Any]:
    locked = json.loads(path.read_text(encoding="utf-8"))
    if int(locked["seed"]) != 42 or int(locked["n_jobs"]) != 7:
        raise AssertionError("locked seed or n_jobs changed")
    common = preregister["models"]["common_params"]
    for component, objective, alpha in (
        ("lgb_l1", "l1", None),
        ("lgb_q07", "quantile", 0.7),
    ):
        specification = locked["models"][component]
        if specification["scope"] != "group":
            raise AssertionError(f"{component} is no longer group-scoped")
        if specification["target_scale"] != "capacity_factor":
            raise AssertionError(f"{component} target scale changed")
        if specification["objective"] != objective or specification["alpha"] != alpha:
            raise AssertionError(f"{component} objective changed")
        if specification["row_filter"] != "eligible" or specification["features"] != "all":
            raise AssertionError(f"{component} row or feature contract changed")
        if specification["params"] != common:
            raise AssertionError(f"{component} parameters differ from preregister")
    return locked


def _candidate_specifications(
    preregister: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    recipes = {item["id"]: item for item in preregister["models"]["recipes"]}
    for recipe_id in RECIPE_IDS:
        for weight in WEIGHTS:
            candidate_id = f"{recipe_id}__w{int(round(weight * 100)):02d}"
            output[candidate_id] = {
                "id": candidate_id,
                "recipe_id": recipe_id,
                "matching_global_component": recipes[recipe_id][
                    "matching_global_component"
                ],
                "weight": float(weight),
            }
    if len(output) != 6:
        raise AssertionError("candidate construction changed")
    return output


def _source_snapshot(
    preregister_path: Path, recipe_path: Path
) -> dict[str, Any]:
    paths = {
        "runner": Path(__file__).resolve(),
        "module": PROJECT_DIR / "src" / "lead_time_experts.py",
        "test": PROJECT_DIR / "tests" / "test_lead_time_experts.py",
        "strict_raw_helper": PROJECT_DIR / "scripts" / "run_shared_q07_multiseed.py",
        "baseline_helper": PROJECT_DIR / "scripts" / "run_run_sequence_residual.py",
        "features": PROJECT_DIR / "src" / "features.py",
        "temporal": PROJECT_DIR / "src" / "temporal.py",
        "metric": PROJECT_DIR / "src" / "metric.py",
        "manifest": PROJECT_DIR / "src" / "manifest.py",
        "preregister": preregister_path.resolve(),
        "preregister_sha": preregister_path.with_suffix(".sha256").resolve(),
        "superseded_v1_preregister": PROJECT_DIR
        / "configs"
        / "lead_time_experts_preregister.json",
        "superseded_v1_sha": PROJECT_DIR
        / "configs"
        / "lead_time_experts_preregister.sha256",
        "locked_recipe": recipe_path.resolve(),
    }
    return {name: describe_file(path) for name, path in paths.items()}


def _stage1_input_snapshot(
    raw_dir: Path,
    artifact_root: Path,
    preregister_path: Path,
    recipe_path: Path,
) -> dict[str, Any]:
    snapshot = _strict_stage1_input_snapshot(raw_dir, artifact_root)
    extra = {
        "stage1_exact_baseline": artifact_root
        / "postgate"
        / "shared_q07_multiseed_strict"
        / "oof"
        / "stage1_corrected_v3_baseline.parquet",
        "preregister": preregister_path,
        "preregister_sha": preregister_path.with_suffix(".sha256"),
        "locked_recipe": recipe_path,
        "superseded_v1_preregister": PROJECT_DIR
        / "configs"
        / "lead_time_experts_preregister.json",
        "superseded_v1_sha": PROJECT_DIR
        / "configs"
        / "lead_time_experts_preregister.sha256",
    }
    snapshot.update({name: describe_file(path) for name, path in extra.items()})
    return snapshot


def _assert_snapshot_equal(
    before: Mapping[str, Any], after: Mapping[str, Any], *, name: str
) -> None:
    if _json_ready(before) != _json_ready(after):
        raise AssertionError(f"{name} changed during locked execution")


def _normalise_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, engine="pyarrow")
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise AssertionError(f"prediction index invalid: {path}")
    return frame


def _global_reference(
    *,
    preregister: Mapping[str, Any],
    artifact_root: Path,
    stage: str,
    recipe_id: str,
    group: str,
    expected_index: pd.DatetimeIndex,
) -> tuple[pd.Series, dict[str, Any]]:
    component = {
        "expert_l1": "lgb_l1",
        "expert_q07": "lgb_q07",
    }[recipe_id]
    contract = preregister["global_control_contract"][stage]
    if stage == "stage1":
        relative, column = contract[group][component]
        path = artifact_root / relative.removeprefix("artifacts/")
        frame = _normalise_frame(path)
        if column not in frame:
            raise AssertionError(f"{path} missing {column}")
        series = frame[column].astype(float)
    else:
        relative = contract[component]
        path = artifact_root / relative.removeprefix("artifacts/")
        frame = _read_prediction(
            path, expected_index=expected_index, required_groups=TARGET_COLS
        )
        series = frame[group].astype(float)
    series.index = pd.DatetimeIndex(series.index, name="forecast_kst_dtm")
    if not series.index.equals(expected_index):
        raise AssertionError(f"{stage}/{recipe_id}/{group} control index changed")
    if not np.isfinite(series.to_numpy()).all():
        raise AssertionError("global control contains non-finite values")
    return series, {"source": describe_file(path), "column": str(series.name)}


def _stage1_slices(
    preregister: Mapping[str, Any], group: str
) -> dict[str, Sequence[str]]:
    fold = preregister["stage1_pre2024_selection"]["folds"][group]
    bounds = preregister["stage1_pre2024_selection"]["slice_boundaries"]
    output: dict[str, Sequence[str]] = {}
    for name in fold["required_slices"]:
        key = "2023_H2" if group == "kpx_group_3" and name == "full" else f"2023_{name}"
        output[name] = bounds[key]
    return output


def _assert_stage1_comparison_contract(
    comparisons: Mapping[str, Any],
    specifications: Mapping[str, Mapping[str, Any]],
    preregister: Mapping[str, Any],
) -> None:
    if set(comparisons) != set(specifications) or len(comparisons) != 6:
        raise AssertionError("Stage 1 must score exactly the six registered candidates")
    for candidate_id, by_group in comparisons.items():
        if tuple(by_group) != TARGET_COLS:
            raise AssertionError(f"{candidate_id} group comparison contract changed")
        for group in TARGET_COLS:
            expected = tuple(
                preregister["stage1_pre2024_selection"]["folds"][group][
                    "required_slices"
                ]
            )
            if tuple(by_group[group]) != expected:
                raise AssertionError(
                    f"{candidate_id}/{group} required slices changed"
                )


def _direct_raw_lead_audit(
    raw_dir: Path, features: Mapping[str, pd.DataFrame]
) -> dict[str, Any]:
    """Re-read exact byte-bounded prefixes and verify lead from issuance time."""

    by_source: dict[str, Any] = {}
    availability: dict[str, pd.Series] = {}
    expected_sequence = np.tile(np.arange(12, 36, dtype=np.float64), 730)
    for source in ("ldaps", "gfs"):
        raw, evidence = _read_bounded_weather_csv(
            raw_dir / "train" / f"{source}_train.csv",
            source=source,
            expected_timestamps=EXPECTED_ROWS_PRE2024,
            expected_start=YEAR_2022_START,
            expected_end=YEAR_2023_END,
            expected_next=EXPECTED_STAGE1_NEXT_TIMESTAMP,
            expected_prefix_bytes=OFFICIAL_STAGE1_PREFIX_BYTES[source],
        )
        if evidence["physical_prefix_sha256"] != OFFICIAL_NWP_PREFIX_SHA256[source]:
            raise AssertionError(f"{source} bounded raw prefix hash changed")
        counts = raw.groupby(TIME_COL, sort=False)[AVAILABLE_COL].nunique()
        if not counts.eq(1).all():
            raise AssertionError(f"{source} has multiple issuance times per forecast")
        pairs = raw[[TIME_COL, AVAILABLE_COL]].drop_duplicates(TIME_COL)
        pairs = pairs.set_index(TIME_COL)[AVAILABLE_COL]
        pairs.index = pd.DatetimeIndex(pairs.index, name="forecast_kst_dtm")
        lead = (
            pairs.index - pd.DatetimeIndex(pairs.to_numpy())
        ).total_seconds() / 3600.0
        if not np.array_equal(np.asarray(lead, dtype=float), expected_sequence):
            raise AssertionError(
                f"{source} raw forecast minus data-available lead is not ordered 12..35"
            )
        feature_lead = features[TARGET_COLS[0]]["time__lead_hours"].to_numpy(
            dtype=float, copy=False
        )
        if not pairs.index.equals(features[TARGET_COLS[0]].index):
            raise AssertionError(f"{source} raw lead index differs from built features")
        if not np.array_equal(feature_lead, np.asarray(lead, dtype=float)):
            raise AssertionError(f"{source} raw lead differs from feature lead")
        availability[source] = pairs
        by_source[source] = {
            "rows": int(len(pairs)),
            "lead_min": float(np.min(lead)),
            "lead_max": float(np.max(lead)),
            "ordered_12_through_35_every_run": True,
            "feature_lead_bit_exact": True,
            "suffix_bytes_exposed_to_parser": evidence[
                "suffix_bytes_exposed_to_parser"
            ],
            "physical_prefix_sha256": evidence["physical_prefix_sha256"],
        }
        del raw
    if not availability["ldaps"].equals(availability["gfs"]):
        raise AssertionError("LDAPS and GFS issuance runs differ")
    return {
        "definition": "forecast_kst_dtm - data_available_kst_dtm",
        "ldaps_gfs_availability_aligned": True,
        "sources": by_source,
    }


def _fit_apply_bounds(
    preregister: Mapping[str, Any], stage: str, group: str
) -> tuple[Sequence[str], Sequence[str]]:
    if stage == "stage1":
        fold = preregister["stage1_pre2024_selection"]["folds"][group]
        return fold["fit"], fold["apply"]
    if stage == "stage2":
        fit_start = YEAR_2022_START
        if group == "kpx_group_3":
            fit_start = STAGE1_START
        return (fit_start.isoformat(), STAGE1_END.isoformat()), (
            STAGE2_START.isoformat(),
            STAGE2_END.isoformat(),
        )
    if stage == "final":
        fit_start = YEAR_2022_START
        if group == "kpx_group_3":
            fit_start = STAGE1_START
        return (fit_start.isoformat(), STAGE2_END.isoformat()), (
            FINAL_START.isoformat(),
            FINAL_END.isoformat(),
        )
    raise KeyError(stage)


def _fit_stage_models(
    *,
    stage: str,
    preregister: Mapping[str, Any],
    artifact_root: Path,
    out_dir: Path,
    features: Mapping[str, pd.DataFrame],
    labels: pd.DataFrame,
    recipe_ids: Sequence[str] = RECIPE_IDS,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, Any]]:
    recipes = {item["id"]: item for item in preregister["models"]["recipes"]}
    expert_frames: dict[str, pd.DataFrame] = {}
    global_frames: dict[str, pd.DataFrame] = {}
    training: dict[str, Any] = {}
    for recipe_id in recipe_ids:
        if recipe_id not in RECIPE_IDS:
            raise AssertionError(f"unregistered recipe requested: {recipe_id}")
        expert_frame = pd.DataFrame(columns=list(TARGET_COLS), dtype=float)
        global_frame = pd.DataFrame(columns=list(TARGET_COLS), dtype=float)
        training[recipe_id] = {}
        for group in TARGET_COLS:
            frame = features[group]
            fit_bounds, apply_bounds = _fit_apply_bounds(preregister, stage, group)
            fit_index = _interval(frame.index, fit_bounds)
            apply_index = _interval(frame.index, apply_bounds)
            actual_cf = labels[group].reindex(frame.index).astype(float) / CAPACITY_KWH[group]
            print(f"{stage}: fitting {recipe_id}/{group} global + four experts", flush=True)
            global_model, experts, global_cf, expert_cf, audit = (
                fit_global_and_lead_experts(
                    features=frame,
                    actual_cf=actual_cf,
                    fit_index=fit_index,
                    apply_index=apply_index,
                    bins=preregister["feature_contract"]["lead_bins"],
                    recipe=recipes[recipe_id],
                    model_contract=preregister["models"],
                )
            )
            global_kwh = global_cf * CAPACITY_KWH[group]
            expert_kwh = expert_cf * CAPACITY_KWH[group]
            if stage == "stage1":
                reference, reference_audit = _global_reference(
                    preregister=preregister,
                    artifact_root=artifact_root,
                    stage=stage,
                    recipe_id=recipe_id,
                    group=group,
                    expected_index=apply_index,
                )
                difference = np.abs(
                    global_kwh.to_numpy(dtype=float) - reference.to_numpy(dtype=float)
                )
                maximum = float(difference.max(initial=0.0))
                unequal = int(np.count_nonzero(difference != 0.0))
                if maximum != 0.0 or unequal:
                    raise AssertionError(
                        f"{stage}/{recipe_id}/{group} global OOF is not exact: "
                        f"max={maximum}, unequal={unequal}"
                    )
                subtraction_control = reference
                exact_reproduction: bool | None = True
            else:
                subtraction_control = global_kwh
                reference_audit = {
                    "source": "new same-code-path capacity-factor global refit",
                    "historical_persisted_kwh_global_read": False,
                    "historical_persisted_kwh_global_used_in_formula": False,
                }
                maximum = 0.0
                unequal = 0
                exact_reproduction = None
            if expert_frame.empty:
                expert_frame = pd.DataFrame(index=apply_index, columns=list(TARGET_COLS), dtype=float)
                global_frame = pd.DataFrame(index=apply_index, columns=list(TARGET_COLS), dtype=float)
            else:
                union = expert_frame.index.union(apply_index).sort_values()
                expert_frame = expert_frame.reindex(union)
                global_frame = global_frame.reindex(union)
            expert_frame.loc[apply_index, group] = expert_kwh.to_numpy()
            global_frame.loc[apply_index, group] = subtraction_control.to_numpy()
            model_path = out_dir / "models" / f"{stage}__{recipe_id}__{group}.joblib"
            _atomic_joblib(
                {
                    "stage": stage,
                    "recipe": recipes[recipe_id],
                    "group": group,
                    "feature_columns": list(frame.columns),
                    "global_model": global_model,
                    "lead_experts": experts,
                },
                model_path,
            )
            audit.update(
                {
                    "global_control_exact": exact_reproduction,
                    "global_control_max_abs_difference_kwh": maximum,
                    "global_control_unequal_values": unequal,
                    "global_control": reference_audit,
                    "model": describe_file(model_path),
                }
            )
            training[recipe_id][group] = audit
        expert_frame.index = pd.DatetimeIndex(expert_frame.index, name="forecast_kst_dtm")
        global_frame.index = pd.DatetimeIndex(global_frame.index, name="forecast_kst_dtm")
        expert_frames[recipe_id] = expert_frame
        global_frames[recipe_id] = global_frame
        _atomic_parquet(
            expert_frame,
            out_dir / "predictions" / f"{stage}__{recipe_id}__stitched.parquet",
        )
        _atomic_parquet(
            global_frame,
            out_dir / "predictions" / f"{stage}__{recipe_id}__global_exact.parquet",
        )
    return expert_frames, global_frames, training


def _candidate_frames(
    *,
    baseline: pd.DataFrame,
    expert_frames: Mapping[str, pd.DataFrame],
    global_frames: Mapping[str, pd.DataFrame],
    specifications: Mapping[str, Mapping[str, Any]],
) -> dict[str, pd.DataFrame]:
    output: dict[str, pd.DataFrame] = {}
    for candidate_id, specification in specifications.items():
        recipe_id = str(specification["recipe_id"])
        frame = pd.DataFrame(index=baseline.index, columns=list(TARGET_COLS), dtype=float)
        for group in TARGET_COLS:
            available = expert_frames[recipe_id][group].dropna().index
            frame.loc[available, group] = specialization_candidate(
                baseline.loc[available, group],
                expert_frames[recipe_id].loc[available, group],
                global_frames[recipe_id].loc[available, group],
                weight=float(specification["weight"]),
                capacity_kwh=CAPACITY_KWH[group],
            ).to_numpy()
        output[candidate_id] = frame
    return output


def _manifest(
    *,
    out_dir: Path,
    preregister: Mapping[str, Any],
    source_snapshot: Mapping[str, Any],
    input_snapshot: Mapping[str, Any],
    stage1_result: Mapping[str, Any],
    stage2_result: Mapping[str, Any] | None,
    final_result: Mapping[str, Any] | None,
) -> dict[str, Any]:
    outputs = []
    for path in sorted(out_dir.rglob("*")):
        if path.is_file() and path.name not in {"manifest.json"}:
            outputs.append(describe_file(path))
    return {
        "schema_version": 1,
        "artifact_type": "lead_time_experts_strict_forward",
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "candidate_formula": preregister["global_control_override"][
            "formula_all_stages"
        ],
        "superseded_v1": preregister["supersedes"],
        "target_scale_provenance": preregister["target_scale_provenance"],
        "stage1_selected_candidate": stage1_result["selected_candidate"],
        "stage1_passed": stage1_result["selected_candidate"] is not None,
        "2024_read": stage2_result is not None,
        "stage2_promoted": bool(stage2_result and stage2_result["promoted"]),
        "final_csv_created": bool(final_result and final_result.get("submission")),
        "leaderboard_score_claim": False,
        "untouched_gate_claim": False,
        "source_snapshot": source_snapshot,
        "source_snapshot_sha256": _mapping_sha256(source_snapshot),
        "input_snapshot": input_snapshot,
        "input_snapshot_sha256": _mapping_sha256(input_snapshot),
        "outputs": outputs,
        "runtime": {"packages": package_versions(), "git": git_state(PROJECT_DIR)},
    }


def _stage1(
    *,
    raw_dir: Path,
    cache_dir: Path,
    artifact_root: Path,
    recipe_path: Path,
    preregister_path: Path,
    out_dir: Path,
    preregister: Mapping[str, Any],
    specifications: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"Stage 1 requires a new output directory: {out_dir}")
    out_dir.mkdir(parents=True)
    shutil.copyfile(preregister_path, out_dir / "preregister.json")
    shutil.copyfile(preregister_path.with_suffix(".sha256"), out_dir / "preregister.sha256")
    source_before = _source_snapshot(preregister_path, recipe_path)
    inputs_before = _stage1_input_snapshot(
        raw_dir, artifact_root, preregister_path, recipe_path
    )

    labels = _read_labels(
        raw_dir / "train" / "train_labels.csv",
        nrows=EXPECTED_ROWS_PRE2024,
        expected_end=YEAR_2023_END,
        prefix_bytes=OFFICIAL_STAGE1_PREFIX_BYTES["labels"],
        expected_prefix_sha256=OFFICIAL_LABEL_PREFIX_SHA256,
    )
    print("Stage 1: building physically bounded pre-2024 weather features", flush=True)
    features, raw_contract = _read_stage1_raw_features(raw_dir, labels)
    direct_raw_lead = _direct_raw_lead_audit(raw_dir, features)
    lead_audit: dict[str, Any] = {}
    canonical: tuple[str, ...] | None = None
    for group in TARGET_COLS:
        frame = features[group]
        if canonical is not None and tuple(frame.columns) != canonical:
            raise AssertionError("Stage 1 group feature schemas differ")
        canonical = tuple(frame.columns)
        _, lead_audit[group] = validate_lead_partition(
            frame,
            preregister["feature_contract"]["lead_bins"],
            expected_leads=preregister["feature_contract"]["lead_integer_values"],
        )
    if canonical is None or len(canonical) != 612:
        raise AssertionError("Stage 1 feature schema is not locked 612 columns")
    baseline, baseline_audit = _read_stage1_baseline(artifact_root)
    _atomic_parquet(baseline, out_dir / "predictions" / "stage1_corrected_v3.parquet")

    experts, globals_, training = _fit_stage_models(
        stage="stage1",
        preregister=preregister,
        artifact_root=artifact_root,
        out_dir=out_dir,
        features=features,
        labels=labels,
    )
    candidates = _candidate_frames(
        baseline=baseline,
        expert_frames=experts,
        global_frames=globals_,
        specifications=specifications,
    )
    comparisons: dict[str, Any] = {}
    specialization_diagnostics: dict[str, Any] = {}
    for recipe_id in RECIPE_IDS:
        specialization_diagnostics[recipe_id] = {}
        for group in TARGET_COLS:
            available = experts[recipe_id][group].dropna().index
            specialization_diagnostics[recipe_id][group] = _comparisons(
                actual=labels.loc[available, group],
                baseline=globals_[recipe_id].loc[available, group],
                candidate=experts[recipe_id].loc[available, group],
                group=group,
                slices=_stage1_slices(preregister, group),
            )
    for candidate_id, frame in candidates.items():
        comparisons[candidate_id] = {}
        for group in TARGET_COLS:
            available = frame[group].dropna().index
            comparisons[candidate_id][group] = _comparisons(
                actual=labels.loc[available, group],
                baseline=baseline.loc[available, group],
                candidate=frame.loc[available, group],
                group=group,
                slices=_stage1_slices(preregister, group),
            )
        _atomic_parquet(
            frame, out_dir / "predictions" / f"stage1__{candidate_id}.parquet"
        )
    _assert_stage1_comparison_contract(comparisons, specifications, preregister)
    selected = choose_registered_candidate(comparisons, specifications)

    source_after = _source_snapshot(preregister_path, recipe_path)
    inputs_after = _stage1_input_snapshot(
        raw_dir, artifact_root, preregister_path, recipe_path
    )
    _assert_snapshot_equal(source_before, source_after, name="Stage 1 source snapshot")
    _assert_snapshot_equal(inputs_before, inputs_after, name="Stage 1 input snapshot")
    result = {
        "schema_version": 1,
        "experiment_id": "lead_time_experts_stage1_pre2024",
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "candidate_formula": preregister["candidate_family"]["formula"],
        "candidate_count": len(specifications),
        "selected_candidate": selected,
        "selected_specification": None if selected is None else specifications[selected],
        "selection_changes_after_2024": False,
        "2024_read": False,
        "2024_label_rows_materialized": 0,
        "2024_weather_rows_materialized": 0,
        "2024_prediction_rows_materialized": 0,
        "cache_dir_deliberately_not_read": str(cache_dir.resolve()),
        "bounded_label_rows": int(len(labels)),
        "raw_feature_contract": raw_contract,
        "direct_raw_lead_contract": direct_raw_lead,
        "baseline_exact_reconstruction": baseline_audit,
        "feature_contract": {
            "columns": len(canonical),
            "column_names": list(canonical),
            "lead_partition": lead_audit,
            "target_scada_or_other_group_actual_features": 0,
        },
        "global_control_exact_reproduction": True,
        "training": training,
        "specialization_diagnostics_expert_vs_global": specialization_diagnostics,
        "comparisons_vs_corrected_v3": comparisons,
        "source_snapshot_before": source_before,
        "source_snapshot_after": source_after,
        "source_snapshot_sha256": _mapping_sha256(source_before),
        "input_snapshot_before": inputs_before,
        "input_snapshot_after": inputs_after,
        "input_snapshot_sha256": _mapping_sha256(inputs_before),
        "snapshots_unchanged": True,
        "leaderboard_score_claim": False,
        "untouched_gate_claim": False,
    }
    result_path = out_dir / "stage1_results.json"
    _write_json(result_path, result)
    lock = {
        "schema_version": 1,
        "experiment_id": "lead_time_experts_pre2024_lock",
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "stage1_results_sha256": sha256_file(result_path),
        "source_snapshot_sha256": result["source_snapshot_sha256"],
        "input_snapshot_sha256": result["input_snapshot_sha256"],
        "selected_candidate": selected,
        "selected_specification": None if selected is None else specifications[selected],
        "2024_read_before_lock": False,
        "selection_changes_after_2024": False,
    }
    _write_json(out_dir / "stage1_candidate_lock.json", lock)
    stage1_manifest = {
        "schema_version": 1,
        "artifact_type": "lead_time_experts_stage1",
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "selected_candidate": selected,
        "2024_read": False,
        "source_snapshot": source_before,
        "input_snapshot": inputs_before,
        "outputs": [
            describe_file(path)
            for path in sorted(out_dir.rglob("*"))
            if path.is_file() and path.name != "stage1_manifest.json"
        ],
        "leaderboard_score_claim": False,
    }
    _write_json(out_dir / "stage1_manifest.json", stage1_manifest)
    return result


def _verify_stage1_lock(
    out_dir: Path,
    preregister: Mapping[str, Any],
    specifications: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    result_path = out_dir / "stage1_results.json"
    lock_path = out_dir / "stage1_candidate_lock.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["preregister_sha256"] != PREREGISTER_SHA256:
        raise AssertionError("Stage 1 lock preregister hash changed")
    if lock["stage1_results_sha256"] != sha256_file(result_path):
        raise AssertionError("Stage 1 results changed after lock")
    recomputed = choose_registered_candidate(
        result["comparisons_vs_corrected_v3"], specifications
    )
    if recomputed != lock["selected_candidate"] or recomputed != result["selected_candidate"]:
        raise AssertionError("Stage 1 selected candidate does not reproduce")
    if lock["selected_specification"] != (
        None if recomputed is None else specifications[recomputed]
    ):
        raise AssertionError("Stage 1 locked specification changed")
    if result["2024_read"] or not result["global_control_exact_reproduction"]:
        raise AssertionError("Stage 1 access or global-control contract failed")
    return result, lock


def _postlock_input_snapshot(
    *,
    raw_dir: Path,
    cache_dir: Path,
    artifact_root: Path,
    recipe_path: Path,
    out_dir: Path,
    include_final: bool,
) -> dict[str, Any]:
    paths: dict[str, Path] = {
        "labels": raw_dir / "train" / "train_labels.csv",
        "locked_recipe": recipe_path,
        "stage1_results": out_dir / "stage1_results.json",
        "stage1_lock": out_dir / "stage1_candidate_lock.json",
        "gate_baseline": artifact_root / "oof" / "gate2024_locked_v3_cf_fix.parquet",
    }
    for group in TARGET_COLS:
        paths[f"train_features_{group}"] = cache_dir / f"{group}_weather_train.parquet"
    for name, relative in GATE_COMPONENT_FILES.items():
        paths[f"corrected_v3_gate_component_{name}"] = artifact_root / relative
    if include_final:
        paths.update(
            {
                "sample_submission": raw_dir / "sample_submission.csv",
                "final_baseline": artifact_root
                / "final_cf_fix"
                / "predictions"
                / "corrected_v3_test.parquet",
            }
        )
        for group in TARGET_COLS:
            paths[f"test_features_{group}"] = cache_dir / f"{group}_weather_test.parquet"
        for name, relative in FINAL_COMPONENT_FILES.items():
            paths[f"corrected_v3_final_component_{name}"] = artifact_root / relative
    return {name: describe_file(path) for name, path in paths.items()}


def _stage2(
    *,
    raw_dir: Path,
    cache_dir: Path,
    artifact_root: Path,
    recipe_path: Path,
    preregister_path: Path,
    out_dir: Path,
    preregister: Mapping[str, Any],
    specifications: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    stage1_result, lock = _verify_stage1_lock(out_dir, preregister, specifications)
    selected = lock["selected_candidate"]
    if selected is None:
        print("Stage 1 rejected every candidate; 2024 remains unread", flush=True)
        return {
            "schema_version": 1,
            "experiment_id": "lead_time_experts_stage2_skipped",
            "created_utc": utc_now(),
            "selected_candidate": None,
            "2024_read": False,
            "promoted": False,
            "reason": "no Stage 1 candidate passed every registered slice",
            "leaderboard_score_claim": False,
        }, None
    if (out_dir / "stage2_results.json").exists():
        raise FileExistsError("Stage 2 artifacts already exist")
    source_before = _source_snapshot(preregister_path, recipe_path)
    input_before = _postlock_input_snapshot(
        raw_dir=raw_dir,
        cache_dir=cache_dir,
        artifact_root=artifact_root,
        recipe_path=recipe_path,
        out_dir=out_dir,
        include_final=False,
    )
    labels = _read_labels(
        raw_dir / "train" / "train_labels.csv",
        nrows=EXPECTED_ROWS_THROUGH2024,
        expected_end=YEAR_2024_END,
    )
    features = _read_features(cache_dir, labels, expected_end=YEAR_2024_END)
    lead_audit = {
        group: validate_lead_partition(
            features[group],
            preregister["feature_contract"]["lead_bins"],
            expected_leads=preregister["feature_contract"]["lead_integer_values"],
        )[1]
        for group in TARGET_COLS
    }
    gate_index = pd.date_range(
        STAGE2_START, STAGE2_END, freq="h", name="forecast_kst_dtm"
    )
    baseline, baseline_audit = _reconstruct_baseline(
        artifact_root=artifact_root,
        recipe_path=recipe_path,
        component_files=GATE_COMPONENT_FILES,
        reference_path=artifact_root / "oof" / "gate2024_locked_v3_cf_fix.parquet",
        expected_index=gate_index,
    )
    _atomic_parquet(baseline, out_dir / "predictions" / "stage2_corrected_v3.parquet")
    experts, globals_, training = _fit_stage_models(
        stage="stage2",
        preregister=preregister,
        artifact_root=artifact_root,
        out_dir=out_dir,
        features=features,
        labels=labels,
        recipe_ids=(str(specifications[selected]["recipe_id"]),),
    )
    selected_spec = {selected: specifications[selected]}
    candidate = _candidate_frames(
        baseline=baseline,
        expert_frames=experts,
        global_frames=globals_,
        specifications=selected_spec,
    )[selected]
    comparisons = {
        group: _comparisons(
            actual=labels.loc[gate_index, group],
            baseline=baseline[group],
            candidate=candidate[group],
            group=group,
            slices=STAGE2_SLICES,
        )
        for group in TARGET_COLS
    }
    if tuple(comparisons) != TARGET_COLS:
        raise AssertionError("Stage 2 group comparison contract changed")
    for group in TARGET_COLS:
        if tuple(comparisons[group]) != tuple(STAGE2_SLICES):
            raise AssertionError(f"Stage 2 {group} slice contract changed")
    deltas = [
        float(item["delta"])
        for slices in comparisons.values()
        for item in slices.values()
    ]
    promoted = bool(deltas and all(delta > 0.0 for delta in deltas))
    _atomic_parquet(
        candidate, out_dir / "predictions" / f"stage2__{selected}.parquet"
    )
    source_after = _source_snapshot(preregister_path, recipe_path)
    input_after = _postlock_input_snapshot(
        raw_dir=raw_dir,
        cache_dir=cache_dir,
        artifact_root=artifact_root,
        recipe_path=recipe_path,
        out_dir=out_dir,
        include_final=False,
    )
    _assert_snapshot_equal(source_before, source_after, name="Stage 2 source snapshot")
    _assert_snapshot_equal(input_before, input_after, name="Stage 2 input snapshot")
    result = {
        "schema_version": 1,
        "experiment_id": "lead_time_experts_stage2_fixed_2024",
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "selected_candidate": selected,
        "selected_specification": specifications[selected],
        "2024_read": True,
        "selection_changes_after_2024": False,
        "global_control_contract": {
            "target_scale": "capacity_factor",
            "same_code_path_as_experts": True,
            "historical_persisted_kwh_global_read": False,
            "historical_persisted_kwh_global_used_in_formula": False,
            "exact_persisted_reproduction": "not applicable after Stage1",
        },
        "baseline_exact_reconstruction": baseline_audit,
        "lead_partition": lead_audit,
        "training": training,
        "comparisons": comparisons,
        "minimum_delta": float(min(deltas)),
        "mean_delta": float(np.mean(deltas)),
        "promoted": promoted,
        "source_snapshot": source_before,
        "input_snapshot": input_before,
        "snapshots_unchanged": True,
        "leaderboard_score_claim": False,
        "untouched_gate_claim": False,
    }
    result_path = out_dir / "stage2_results.json"
    _write_json(result_path, result)
    promotion_lock = {
        "schema_version": 1,
        "experiment_id": "lead_time_experts_fixed_2024_lock",
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "stage1_results_sha256": sha256_file(out_dir / "stage1_results.json"),
        "stage1_lock_sha256": sha256_file(out_dir / "stage1_candidate_lock.json"),
        "stage2_results_sha256": sha256_file(result_path),
        "selected_candidate": selected,
        "promoted": promoted,
        "selection_changes_after_2024": False,
    }
    _write_json(out_dir / "promotion_lock.json", promotion_lock)
    final_result = None
    if promoted:
        final_result = _final(
            raw_dir=raw_dir,
            cache_dir=cache_dir,
            artifact_root=artifact_root,
            recipe_path=recipe_path,
            preregister_path=preregister_path,
            out_dir=out_dir,
            preregister=preregister,
            specification=specifications[selected],
            labels=labels,
            train_features=features,
        )
    return result, final_result


def _final(
    *,
    raw_dir: Path,
    cache_dir: Path,
    artifact_root: Path,
    recipe_path: Path,
    preregister_path: Path,
    out_dir: Path,
    preregister: Mapping[str, Any],
    specification: Mapping[str, Any],
    labels: pd.DataFrame,
    train_features: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    promotion_path = out_dir / "promotion_lock.json"
    stage2_path = out_dir / "stage2_results.json"
    promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    if promotion.get("promoted") is not True:
        raise AssertionError("final fit requires complete fixed-2024 promotion")
    if promotion["stage2_results_sha256"] != sha256_file(stage2_path):
        raise AssertionError("Stage 2 results changed after promotion lock")
    source_before = _source_snapshot(
        preregister_path,
        recipe_path,
    )
    input_before = _postlock_input_snapshot(
        raw_dir=raw_dir,
        cache_dir=cache_dir,
        artifact_root=artifact_root,
        recipe_path=recipe_path,
        out_dir=out_dir,
        include_final=True,
    )
    sample = pd.read_csv(raw_dir / "sample_submission.csv", encoding="utf-8-sig")
    test_index = pd.DatetimeIndex(
        pd.to_datetime(sample["forecast_kst_dtm"], errors="raise"),
        name="forecast_kst_dtm",
    )
    if (
        len(test_index) != EXPECTED_TEST_ROWS
        or test_index.min() != FINAL_START
        or test_index.max() != FINAL_END
    ):
        raise AssertionError("sample submission time range changed")
    test_features = _read_test_features(cache_dir, test_index)
    combined: dict[str, pd.DataFrame] = {}
    lead_audit: dict[str, Any] = {}
    for group in TARGET_COLS:
        if tuple(train_features[group].columns) != tuple(test_features[group].columns):
            raise AssertionError(f"{group} train/test feature schema differs")
        combined[group] = pd.concat(
            [train_features[group], test_features[group]], axis=0, copy=False
        )
        combined[group].index = pd.DatetimeIndex(
            combined[group].index, name="forecast_kst_dtm"
        )
        _, lead_audit[group] = validate_lead_partition(
            combined[group],
            preregister["feature_contract"]["lead_bins"],
            expected_leads=preregister["feature_contract"]["lead_integer_values"],
        )
    baseline, baseline_audit = _reconstruct_baseline(
        artifact_root=artifact_root,
        recipe_path=recipe_path,
        component_files=FINAL_COMPONENT_FILES,
        reference_path=artifact_root
        / "final_cf_fix"
        / "predictions"
        / "corrected_v3_test.parquet",
        expected_index=test_index,
    )
    _atomic_parquet(baseline, out_dir / "predictions" / "final_corrected_v3.parquet")
    experts, globals_, training = _fit_stage_models(
        stage="final",
        preregister=preregister,
        artifact_root=artifact_root,
        out_dir=out_dir,
        features=combined,
        labels=labels,
        recipe_ids=(str(specification["recipe_id"]),),
    )
    candidate_id = str(specification["id"])
    prediction = _candidate_frames(
        baseline=baseline,
        expert_frames=experts,
        global_frames=globals_,
        specifications={candidate_id: specification},
    )[candidate_id]
    if not prediction.index.equals(test_index):
        raise AssertionError("final lead-time prediction index changed")
    values = prediction.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise AssertionError("final lead-time prediction contains non-finite values")
    for group in TARGET_COLS:
        if not prediction[group].between(0.0, 1.02 * CAPACITY_KWH[group]).all():
            raise AssertionError(f"final {group} prediction exceeds clip")
    prediction_path = out_dir / "predictions" / "lead_time_experts_2025.parquet"
    _atomic_parquet(prediction, prediction_path)
    submission = _write_submission(
        raw_dir / "sample_submission.csv",
        prediction,
        out_dir / "lead_time_experts_2025.csv",
    )
    source_after = _source_snapshot(
        preregister_path,
        recipe_path,
    )
    input_after = _postlock_input_snapshot(
        raw_dir=raw_dir,
        cache_dir=cache_dir,
        artifact_root=artifact_root,
        recipe_path=recipe_path,
        out_dir=out_dir,
        include_final=True,
    )
    _assert_snapshot_equal(source_before, source_after, name="final source snapshot")
    _assert_snapshot_equal(input_before, input_after, name="final input snapshot")
    result = {
        "schema_version": 1,
        "experiment_id": "lead_time_experts_final_2025",
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "selected_candidate": candidate_id,
        "selected_specification": specification,
        "global_control_contract": {
            "target_scale": "capacity_factor",
            "same_code_path_as_experts": True,
            "historical_persisted_kwh_global_read": False,
            "historical_persisted_kwh_global_used_in_formula": False,
        },
        "baseline_exact_reconstruction": baseline_audit,
        "lead_partition": lead_audit,
        "training": training,
        "prediction": describe_file(prediction_path),
        "prediction_frame_sha256": _frame_sha256(prediction),
        "submission": submission,
        "source_snapshot": source_before,
        "input_snapshot": input_before,
        "snapshots_unchanged": True,
        "leaderboard_score_claim": False,
    }
    _write_json(out_dir / "final_results.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    preregister = _load_preregister(args.preregister)
    _verify_locked_recipe(args.recipe, preregister)
    specifications = _candidate_specifications(preregister)
    stage1_result: dict[str, Any]
    if args.stage in {"stage1", "all"}:
        stage1_result = _stage1(
            raw_dir=args.raw_dir,
            cache_dir=args.cache_dir,
            artifact_root=args.artifact_root,
            recipe_path=args.recipe,
            preregister_path=args.preregister,
            out_dir=args.out_dir,
            preregister=preregister,
            specifications=specifications,
        )
    else:
        stage1_result, _ = _verify_stage1_lock(
            args.out_dir, preregister, specifications
        )
    if args.stage == "stage1":
        print(
            json.dumps(
                {
                    "stage": "stage1",
                    "selected_candidate": stage1_result["selected_candidate"],
                    "2024_read": False,
                },
                sort_keys=True,
            )
        )
        return

    if stage1_result["selected_candidate"] is None:
        stage2_result = {
            "schema_version": 1,
            "experiment_id": "lead_time_experts_stage2_skipped",
            "created_utc": utc_now(),
            "selected_candidate": None,
            "2024_read": False,
            "promoted": False,
            "reason": "Stage 1 all-slice gate rejected every candidate",
            "leaderboard_score_claim": False,
        }
        final_result = None
    else:
        stage2_result, final_result = _stage2(
            raw_dir=args.raw_dir,
            cache_dir=args.cache_dir,
            artifact_root=args.artifact_root,
            recipe_path=args.recipe,
            preregister_path=args.preregister,
            out_dir=args.out_dir,
            preregister=preregister,
            specifications=specifications,
        )
    source = stage1_result["source_snapshot_before"]
    inputs = stage1_result["input_snapshot_before"]
    manifest = _manifest(
        out_dir=args.out_dir,
        preregister=preregister,
        source_snapshot=source,
        input_snapshot=inputs,
        stage1_result=stage1_result,
        stage2_result=stage2_result if stage2_result.get("2024_read") else None,
        final_result=final_result,
    )
    _write_json(args.out_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "selected_candidate": stage1_result["selected_candidate"],
                "2024_read": bool(stage2_result.get("2024_read")),
                "promoted": bool(stage2_result.get("promoted")),
                "submission_created": bool(final_result and final_result.get("submission")),
                "manifest": str((args.out_dir / "manifest.json").resolve()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

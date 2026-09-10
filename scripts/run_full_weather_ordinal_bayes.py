"""Strict-forward 42-bin full-weather ordinal Bayes Stage1 experiment."""

from __future__ import annotations

import argparse
import atexit
import ctypes
import json
import os
from pathlib import Path
import secrets
import sys
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts import run_catboost_multiquantile_bayes as bounded  # noqa: E402
from scripts import run_direct_interval_probability as protocol  # noqa: E402
from scripts import run_ficr_bayes_decision_strict as strict  # noqa: E402
from scripts import run_shared_q07_multiseed as shared  # noqa: E402
from src.full_weather_ordinal_bayes import (  # noqa: E402
    ACTION_DELTAS_CF,
    BIN_COUNT,
    BIN_START_CF,
    BIN_WIDTH_CF,
    FullWeatherOrdinalBayes,
    MODEL_PARAMETERS,
    STAGE1_REQUIRED,
    TRANSFER_WEIGHT,
    group_stage1_pass,
    transfer_action_kwh,
)
from src.manifest import (  # noqa: E402
    describe_file,
    git_state,
    package_versions,
    sha256_file,
    utc_now,
)
from src.metric import CAPACITY_KWH, TARGET_COLS  # noqa: E402


PREREGISTER_SHA256 = "0052188e1ee6ed1a61cc458e620f357451d2ec6bdb9faf304156f1e185c82f44"
FEASIBILITY_SHA256 = "674298028a50397ca258afd0c45686587ccac44b949dbd0ea98bb087251c0b33"
ARTIFACT_TYPE = "full_weather_ordinal_bayes_strict_forward_v1"
HEAVY_GUARD_PATH = PROJECT_DIR / "artifacts/locks/heavy_cpu_fit.pid.json"
BASELINE_EXPECTED = {
    "g12": {
        "bytes": 248873,
        "sha256": "9f9d4a45a0dad3c19a80905c021a960f2cf1c813a1709cb575d17c5c78dccb76",
    },
    "g3": {
        "bytes": 465202,
        "sha256": "03ed9530a80b10a87d8741c85ddc9b8da8901c1188bfcd007f60da186642197a",
    },
}


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
        default=Path("artifacts/postgate/full_weather_ordinal_bayes_strict_v1"),
    )
    parser.add_argument(
        "--preregister",
        type=Path,
        default=Path("configs/full_weather_ordinal_bayes_preregister_v1.json"),
    )
    parser.add_argument(
        "--feasibility-audit",
        type=Path,
        default=Path("artifacts/audits/full_weather_ordinal_bayes_feasibility_v1.json"),
    )
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def _verify_preregister(path: Path, feasibility_path: Path) -> dict[str, Any]:
    if sha256_file(path) != PREREGISTER_SHA256:
        raise AssertionError("preregister hash changed")
    expected_sidecar = f"{PREREGISTER_SHA256}  {path.name}\n"
    if path.with_suffix(".sha256").read_text(encoding="utf-8") != expected_sidecar:
        raise AssertionError("preregister sidecar changed")
    if sha256_file(feasibility_path) != FEASIBILITY_SHA256:
        raise AssertionError("feasibility audit hash changed")
    feasibility_sidecar = feasibility_path.with_suffix(".sha256")
    expected_feasibility_sidecar = f"{FEASIBILITY_SHA256}  {feasibility_path.name}\n"
    if feasibility_sidecar.read_text(encoding="utf-8") != expected_feasibility_sidecar:
        raise AssertionError("feasibility sidecar changed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["experiment_id"] != "full_weather_ordinal_bayes_strict_forward_v1":
        raise AssertionError("experiment id changed")
    if payload["feature_contract"]["ordered_feature_count"] != 612:
        raise AssertionError("feature count changed")
    bins = payload["target_and_bins"]
    if (
        int(bins["nominal_bin_count"]) != BIN_COUNT
        or float(bins["nominal_bin_width_cf"]) != BIN_WIDTH_CF
        or bins["nominal_edges"] != "0.10 + 0.025*i for integer i=0..42"
    ):
        raise AssertionError("fixed bin contract changed")
    classifier = payload["classifier"]
    if classifier["parameters"] != MODEL_PARAMETERS or classifier["class_weight"] is not None:
        raise AssertionError("classifier contract changed")
    action = payload["official_utility_action"]
    if (
        float(action["fixed_transfer_weight"]) != TRANSFER_WEIGHT
        or int(action["raw_action_count_before_clipped_deduplication"])
        != len(ACTION_DELTAS_CF)
        or int(action["candidate_count"]) != 1
    ):
        raise AssertionError("fixed action/transfer contract changed")
    if int(payload["stage1"]["registered_slice_count"]) != 17:
        raise AssertionError("Stage1 slice count changed")
    exclusion = payload["public_exclusion"]
    for key in (
        "public_feedback_used",
        "public_scale_artifact_used",
        "submission_csv_used",
    ):
        if exclusion[key] is not False:
            raise AssertionError(f"forbidden public input changed: {key}")
    return payload


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return False
    ctypes.windll.kernel32.CloseHandle(handle)
    return True


def acquire_heavy_guard_for_experiment(
    path: Path, *, experiment_id: str
) -> dict[str, Any]:
    """Acquire the shared guard with an explicit caller-owned experiment ID."""

    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise ValueError("experiment_id must be a nonempty string")

    path.parent.mkdir(parents=True, exist_ok=True)
    owner = {
        "pid": os.getpid(),
        "token": secrets.token_hex(16),
        "experiment_id": experiment_id,
        "created_utc": utc_now(),
    }
    for _ in range(2):
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                current = json.loads(path.read_text(encoding="utf-8"))
                current_pid = int(current["pid"])
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"heavy guard exists but is unreadable: {path}") from exc
            if _pid_is_alive(current_pid):
                raise RuntimeError(
                    f"heavy guard is held by live PID {current_pid}: {current}"
                )
            path.unlink()
            continue
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(owner, stream, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
        return owner
    raise RuntimeError("could not acquire shared heavy guard")


def acquire_heavy_guard(path: Path) -> dict[str, Any]:
    """Legacy module-owned acquisition retained for existing ordinal callers."""

    return acquire_heavy_guard_for_experiment(path, experiment_id=ARTIFACT_TYPE)


def release_heavy_guard(path: Path, owner: Mapping[str, Any]) -> None:
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError):
        return
    if (
        int(current.get("pid", -1)) == int(owner["pid"])
        and current.get("token") == owner.get("token")
    ):
        path.unlink()


def _label_spec(
    preregister: Mapping[str, Any], key: str
) -> dict[str, Any]:
    base = dict(preregister["physical_prefixes"][key])
    if key == "labels_g12_fit":
        base.update(
            {
                "usecols": ["kst_dtm", "kpx_group_1", "kpx_group_2"],
                "next_row_first_field_only": "2023-01-01 01:00:00",
            }
        )
    elif key == "labels_g3_fit":
        base.update(
            {
                "usecols": ["kst_dtm", "kpx_group_3"],
                "next_row_first_field_only": "2023-07-01 01:00:00",
            }
        )
    elif key == "labels_stage1_application_after_prescore_lock":
        base.update(
            {
                "usecols": ["kst_dtm", *TARGET_COLS],
                "next_row_first_field_only": "2024-01-01 01:00:00",
            }
        )
    else:
        raise KeyError(key)
    return base


def _prefix_snapshot(path: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    return bounded._prefix_snapshot(path, spec)


def _stage1_input_snapshot(
    raw_dir: Path,
    artifact_root: Path,
    preregister: Mapping[str, Any],
) -> dict[str, Any]:
    physical = preregister["physical_prefixes"]
    label_path = raw_dir / "train/train_labels.csv"
    output = {
        "labels_g12_fit": _prefix_snapshot(
            label_path, _label_spec(preregister, "labels_g12_fit")
        ),
        "labels_g3_fit": _prefix_snapshot(
            label_path, _label_spec(preregister, "labels_g3_fit")
        ),
        "labels_stage1_application_hash_only": _prefix_snapshot(
            label_path,
            _label_spec(preregister, "labels_stage1_application_after_prescore_lock"),
        ),
        "ldaps_stage1_prefix": _prefix_snapshot(
            raw_dir / "train/ldaps_train.csv", physical["ldaps_stage1_prefix"]
        ),
        "gfs_stage1_prefix": _prefix_snapshot(
            raw_dir / "train/gfs_train.csv", physical["gfs_stage1_prefix"]
        ),
        "info_workbook": shared._snapshot_file(raw_dir / "info.xlsx"),
        "baseline_g12": shared._snapshot_file(
            artifact_root / "oof/dev2023_locked_v3.parquet"
        ),
        "baseline_g3": shared._snapshot_file(
            artifact_root / "oof/g3dev2023h2_candidates.parquet"
        ),
    }
    if (
        output["info_workbook"]["size_bytes"] != physical["info_workbook"]["bytes"]
        or output["info_workbook"]["sha256"] != physical["info_workbook"]["sha256"]
    ):
        raise AssertionError("info workbook differs from preregistration")
    for key, expected in BASELINE_EXPECTED.items():
        observed = output["baseline_g12" if key == "g12" else "baseline_g3"]
        if observed["size_bytes"] != expected["bytes"] or observed["sha256"] != expected["sha256"]:
            raise AssertionError(f"baseline {key} identity changed")
    return output


def _provenance_paths(
    preregister_path: Path, feasibility_path: Path
) -> dict[str, Path]:
    closure = protocol._static_repo_import_closure((Path(__file__).resolve(),))
    paths = {
        f"repo_source::{path.relative_to(PROJECT_DIR).as_posix()}": path
        for path in closure
    }
    paths.update(
        {
            "focused_test": PROJECT_DIR / "tests/test_full_weather_ordinal_bayes.py",
            "runner_test": PROJECT_DIR
            / "tests/test_full_weather_ordinal_bayes_runner.py",
            "preregister": preregister_path.resolve(),
            "preregister_sidecar": preregister_path.with_suffix(".sha256").resolve(),
            "feasibility_audit": feasibility_path.resolve(),
            "feasibility_sidecar": feasibility_path.with_suffix(".sha256").resolve(),
        }
    )
    return paths


def _snapshot_named(paths: Mapping[str, Path]) -> dict[str, Any]:
    return {name: shared._snapshot_file(path) for name, path in paths.items()}


def _census_snapshot(preregister: Mapping[str, Any]) -> dict[str, Any]:
    paths: list[str] = []
    for record in preregister["novelty_contract"]["nearest_existing"]:
        paths.extend(record["paths"])
    if len(paths) != len(set(paths)):
        raise AssertionError("duplicate census paths")
    return {
        relative: shared._snapshot_file(PROJECT_DIR / relative)
        for relative in paths
    }


def _fit_predict_group(
    *,
    group: str,
    features: pd.DataFrame,
    actual_fit_prefix: pd.Series,
    baseline: pd.Series,
    fit_index: pd.DatetimeIndex,
    application_index: pd.DatetimeIndex,
    out_dir: Path,
) -> tuple[pd.Series, pd.Series, dict[str, Any], list[Path]]:
    if len(fit_index.intersection(application_index)) or not (
        fit_index.max() < application_index.min()
    ):
        raise AssertionError(f"{group} fit/application is not strict-forward")
    print(
        f"fit full-weather ordinal PMF {group}: {fit_index.min()}..{fit_index.max()} "
        f"-> {application_index.min()}..{application_index.max()}",
        flush=True,
    )
    model = FullWeatherOrdinalBayes().fit(
        features.loc[fit_index],
        actual_fit_prefix.loc[fit_index],
        capacity_kwh=CAPACITY_KWH[group],
    )
    action_cf, probability, diagnostics = model.predict_action(
        features.loc[application_index], baseline.loc[application_index]
    )
    candidate = transfer_action_kwh(
        baseline.loc[application_index],
        action_cf,
        capacity_kwh=CAPACITY_KWH[group],
    )
    model_path = out_dir / f"models/stage1_{group}.joblib"
    probability_path = out_dir / f"oof/stage1_{group}_probability42.parquet"
    action_path = out_dir / f"oof/stage1_{group}_raw_action_cf.parquet"
    diagnostics_path = out_dir / f"oof/stage1_{group}_action_diagnostics.parquet"
    candidate_path = out_dir / f"oof/stage1_{group}_candidate_kwh.parquet"
    strict._atomic_joblib(model, model_path)
    strict._atomic_parquet(probability, probability_path)
    strict._atomic_parquet(action_cf.to_frame(), action_path)
    strict._atomic_parquet(diagnostics, diagnostics_path)
    strict._atomic_parquet(candidate.to_frame(), candidate_path)

    reloaded: FullWeatherOrdinalBayes = joblib.load(model_path)
    reload_action, reload_probability, reload_diagnostics = reloaded.predict_action(
        features.loc[application_index], baseline.loc[application_index]
    )
    reload_candidate = transfer_action_kwh(
        baseline.loc[application_index],
        reload_action,
        capacity_kwh=CAPACITY_KWH[group],
    )
    checks = {
        "probability": np.array_equal(
            probability.to_numpy(), reload_probability.to_numpy()
        ),
        "action": np.array_equal(action_cf.to_numpy(), reload_action.to_numpy()),
        "diagnostics": np.array_equal(
            diagnostics.to_numpy(), reload_diagnostics.to_numpy()
        ),
        "candidate": np.array_equal(candidate.to_numpy(), reload_candidate.to_numpy()),
    }
    if not all(checks.values()):
        raise AssertionError(f"{group} fitted-model reload output differs: {checks}")
    delta_cf = (
        candidate.to_numpy(dtype=np.float64)
        - baseline.loc[application_index].to_numpy(dtype=np.float64)
    ) / CAPACITY_KWH[group]
    if float(np.max(np.abs(delta_cf))) > 0.015 + 2e-15:
        raise AssertionError(f"{group} transferred candidate exceeds +/-0.015 CF")
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
            "application_target_value_cells_materialized": 0,
            "probability_shape": list(probability.shape),
            "probability_row_sum_max_abs_error": float(
                np.max(np.abs(probability.sum(axis=1).to_numpy() - 1.0))
            ),
            "unobserved_fixed_probability_columns_exact_zero": bool(
                np.all(
                    probability.drop(
                        columns=[f"class_{value:02d}" for value in model.observed_classes_]
                    ).to_numpy()
                    == 0.0
                )
            ),
            "candidate_delta_cf_min": float(np.min(delta_cf)),
            "candidate_delta_cf_max": float(np.max(delta_cf)),
            "model_reload_checks": checks,
            "model_reload_all_outputs_bit_exact": True,
            "probability_sha256": protocol._array_sha256(probability.to_numpy()),
            "action_sha256": protocol._array_sha256(action_cf.to_numpy()),
            "diagnostics_sha256": protocol._array_sha256(diagnostics.to_numpy()),
            "candidate_sha256": protocol._array_sha256(candidate.to_numpy()),
        }
    )
    return action_cf, candidate, metadata, [
        model_path,
        probability_path,
        action_path,
        diagnostics_path,
        candidate_path,
    ]


def _all_output_files(out_dir: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in out_dir.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        ],
        key=lambda path: path.relative_to(out_dir).as_posix(),
    )


def run_stage1(
    args: argparse.Namespace,
    preregister: Mapping[str, Any],
    guard_owner: Mapping[str, Any],
) -> dict[str, Any]:
    out_dir = args.out_dir
    if out_dir.exists():
        raise FileExistsError(f"Stage1 requires a new output directory: {out_dir}")
    out_dir.mkdir(parents=True)
    protocol._copy_exclusive(args.preregister, out_dir / "preregister.json")
    protocol._copy_exclusive(
        args.preregister.with_suffix(".sha256"), out_dir / "preregister.sha256"
    )
    protocol._copy_exclusive(
        args.feasibility_audit, out_dir / "feasibility_audit.json"
    )
    protocol._copy_exclusive(
        args.feasibility_audit.with_suffix(".sha256"),
        out_dir / "feasibility_audit.sha256",
    )

    provenance_paths = _provenance_paths(args.preregister, args.feasibility_audit)
    provenance_before = _snapshot_named(provenance_paths)
    census = _census_snapshot(preregister)
    census_path = out_dir / "census_evidence.json"
    strict._write_json(
        census_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "snapshots": census,
            "exact_duplicate_found": False,
            "candidate_fit_started_before_census_lock": False,
            "candidate_score_computed_before_census_lock": False,
            "public_or_scale_bytes_read": 0,
        },
    )
    input_before = _stage1_input_snapshot(
        args.raw_dir, args.artifact_root, preregister
    )

    full_index = pd.date_range(
        "2022-01-01 01:00:00",
        "2024-01-01 00:00:00",
        freq="h",
        name="forecast_kst_dtm",
    )
    raw_features, raw_contract = shared._read_stage1_raw_features(
        args.raw_dir, pd.DataFrame(index=full_index)
    )
    for group in TARGET_COLS:
        frame = raw_features[group]
        if frame.shape != (17520, 612) or not frame.index.equals(full_index):
            raise AssertionError(f"{group} raw feature schema/index changed")
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
    label_path = args.raw_dir / "train/train_labels.csv"

    actions = pd.DataFrame(np.nan, index=baseline.index, columns=TARGET_COLS)
    candidate = baseline.copy()
    training: dict[str, Any] = {}
    group_outputs: list[Path] = []
    labels_g12, labels_g12_evidence = bounded._bounded_label_prefix(
        label_path, _label_spec(preregister, "labels_g12_fit")
    )
    g12_paths: list[Path] = []
    for group in TARGET_COLS[:2]:
        action, group_candidate, metadata, paths = _fit_predict_group(
            group=group,
            features=raw_features[group],
            actual_fit_prefix=labels_g12[group],
            baseline=baseline[group],
            fit_index=fit_indexes[group],
            application_index=application_indexes[group],
            out_dir=out_dir,
        )
        actions.loc[application_indexes[group], group] = action
        candidate.loc[application_indexes[group], group] = group_candidate
        training[group] = metadata
        g12_paths.extend(paths)
        group_outputs.extend(paths)
    g12_lock = out_dir / "stage1_g12_prescore_lock.json"
    strict._write_json(
        g12_lock,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "fit_label_evidence": labels_g12_evidence,
            "groups": list(TARGET_COLS[:2]),
            "outputs": [describe_file(path) for path in g12_paths],
            "application_target_value_cells_materialized": 0,
            "created_before_g3_fit_prefix_and_all_application_labels": True,
        },
    )

    labels_g3, labels_g3_evidence = bounded._bounded_label_prefix(
        label_path, _label_spec(preregister, "labels_g3_fit")
    )
    group = "kpx_group_3"
    action, group_candidate, metadata, paths = _fit_predict_group(
        group=group,
        features=raw_features[group],
        actual_fit_prefix=labels_g3[group],
        baseline=baseline[group],
        fit_index=fit_indexes[group],
        application_index=application_indexes[group],
        out_dir=out_dir,
    )
    actions.loc[application_indexes[group], group] = action
    candidate.loc[application_indexes[group], group] = group_candidate
    training[group] = metadata
    group_outputs.extend(paths)
    g3_lock = out_dir / "stage1_g3_prescore_lock.json"
    strict._write_json(
        g3_lock,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "fit_label_evidence": labels_g3_evidence,
            "group": group,
            "outputs": [describe_file(path) for path in paths],
            "omitted_g12_target_value_cells_materialized": 0,
            "application_target_value_cells_materialized": 0,
            "created_before_all_application_labels": True,
        },
    )

    baseline_path = out_dir / "oof/stage1_primary_baseline_2023.parquet"
    actions_path = out_dir / "oof/stage1_raw_actions_cf_2023.parquet"
    candidate_path = out_dir / "oof/stage1_candidate_w010_2023.parquet"
    strict._atomic_parquet(baseline, baseline_path)
    strict._atomic_parquet(actions, actions_path)
    strict._atomic_parquet(candidate, candidate_path)
    global_outputs = [baseline_path, actions_path, candidate_path]

    provenance_after = _snapshot_named(provenance_paths)
    input_after = _stage1_input_snapshot(
        args.raw_dir, args.artifact_root, preregister
    )
    shared._assert_snapshot_equal(
        provenance_before, provenance_after, name="Stage1 provenance"
    )
    shared._assert_snapshot_equal(
        input_before, input_after, name="Stage1 bounded inputs"
    )
    prescore_path = out_dir / "stage1_prescore_record.json"
    strict._write_json(
        prescore_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "feasibility_sha256": FEASIBILITY_SHA256,
            "heavy_guard_owner": dict(guard_owner),
            "census_lock": describe_file(census_path),
            "training": training,
            "raw_physical_prefix_contract": raw_contract,
            "group_outputs": [describe_file(path) for path in group_outputs],
            "global_candidate_outputs": [
                describe_file(path) for path in global_outputs
            ],
            "all_models_probabilities_actions_and_candidate_hashes_frozen": True,
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
    candidate_artifacts = [*group_outputs, *global_outputs]
    global_lock = out_dir / "stage1_global_prescore_lock.json"
    strict._write_json(
        global_lock,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "prescore_record": describe_file(prescore_path),
            "g12_lock": describe_file(g12_lock),
            "g3_lock": describe_file(g3_lock),
            "candidate_artifacts": [
                describe_file(path) for path in candidate_artifacts
            ],
            "candidate_artifact_count": len(candidate_artifacts),
            "candidate_count": 1,
            "fixed_transfer_weight": TRANSFER_WEIGHT,
            "created_before_any_stage1_application_label_value": True,
            "all_model_probability_action_candidate_hashes_frozen": True,
            "year_2024_value_bytes_read": 0,
            "year_2025_value_bytes_read": 0,
            "public_or_scale_artifact_bytes_read": 0,
        },
    )

    score_labels, score_evidence = bounded._bounded_label_prefix(
        label_path,
        _label_spec(preregister, "labels_stage1_application_after_prescore_lock"),
    )
    comparisons: dict[str, Any] = {}
    promotion: dict[str, Any] = {}
    for group in TARGET_COLS:
        app_index = application_indexes[group]
        slice_indexes = (
            {name: segments[name] for name in STAGE1_REQUIRED[group]}
            if group in TARGET_COLS[:2]
            else {
                "full": segments["H2"],
                "Q3": segments["Q3"],
                "Q4": segments["Q4"],
            }
        )
        if any(len(index) == 0 for index in slice_indexes.values()):
            raise AssertionError(f"{group} has an empty preregistered slice")
        comparisons[group] = strict._comparison(
            score_labels.loc[app_index, group],
            baseline.loc[app_index, group],
            candidate.loc[app_index, group],
            group,
            slice_indexes,
        )
        passed, diagnostics = group_stage1_pass(comparisons[group], group)
        promotion[group] = diagnostics
        if bool(passed) != bool(diagnostics["passed"]):
            raise AssertionError("group promotion computation changed")
    passed_groups = [group for group in TARGET_COLS if promotion[group]["passed"]]
    result = {
        "schema_version": 1,
        "experiment_id": preregister["experiment_id"],
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "feasibility_sha256": FEASIBILITY_SHA256,
        "global_prescore_lock": describe_file(global_lock),
        "score_label_evidence_after_candidate_hash_lock": score_evidence,
        "candidate_count": 1,
        "fixed_transfer_weight": TRANSFER_WEIGHT,
        "registered_slice_count": 17,
        "comparisons": comparisons,
        "comparisons_sha256": strict._canonical_sha256(comparisons),
        "group_promotion": promotion,
        "group_promotion_sha256": strict._canonical_sha256(promotion),
        "passed_groups": passed_groups,
        "locked_candidate": "identity" if not passed_groups else "w010_group_selective",
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
    promotion_lock = out_dir / "stage1_promotion_lock.json"
    strict._write_json(
        promotion_lock,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "stage1_results": describe_file(result_path),
            "comparisons_sha256": result["comparisons_sha256"],
            "group_promotion_sha256": result["group_promotion_sha256"],
            "passed_groups": passed_groups,
            "locked_candidate": result["locked_candidate"],
            "failed_groups_bit_identity_required": True,
            "nonpassing_groups_forbidden_from_2024": True,
            "all_groups_forbidden_from_2025_sample_csv_before_stage2": True,
            "no_weight_bin_class_calibration_or_group_rescue": True,
            "year_2024_value_bytes_read": 0,
            "year_2025_value_bytes_read": 0,
            "csv_files_created": 0,
        },
    )

    outputs = _all_output_files(out_dir)
    manifest = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "created_utc": utc_now(),
        "command": [sys.executable, *sys.argv],
        "preregister": describe_file(args.preregister),
        "preregister_sha256": PREREGISTER_SHA256,
        "feasibility_audit": describe_file(args.feasibility_audit),
        "feasibility_sha256": FEASIBILITY_SHA256,
        "status": "stage1_rejected_identity" if not passed_groups else "stage1_group_promotion_pending_stage2",
        "result": describe_file(result_path),
        "promotion_lock": describe_file(promotion_lock),
        "source_import_closure_count": sum(
            name.startswith("repo_source::") for name in provenance_before
        ),
        "source_and_config_snapshot": provenance_before,
        "census_snapshot": census,
        "input_snapshot": input_before,
        "outputs": [describe_file(path) for path in outputs],
        "output_count_excluding_manifest": len(outputs),
        "runtime": {"packages": package_versions()},
        "git": git_state(PROJECT_DIR),
        "audit_contracts": {
            "preregister_verified_before_fit": True,
            "feasibility_verified_before_fit": True,
            "physical_raw_prefix_only": True,
            "g12_2022_to_2023": True,
            "g3_h1_to_h2": True,
            "all_application_labels_after_candidate_hash_lock": True,
            "all_preregistered_slices_nonempty": True,
            "fitted_model_reload_all_outputs_bit_exact": True,
            "fixed_42_probability_mapping": True,
            "candidate_count_exactly_one": True,
            "public_and_scale_used": False,
            "year_2024_opened": False,
            "year_2025_or_sample_opened": False,
            "csv_created": False,
            "leaderboard_score_claim": False,
        },
    }
    manifest_path = out_dir / "manifest.json"
    strict._write_json(manifest_path, manifest)
    print(f"Stage1 passed groups: {passed_groups or ['identity']}", flush=True)
    print(f"manifest sha256: {sha256_file(manifest_path)}", flush=True)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    preregister = _verify_preregister(args.preregister, args.feasibility_audit)
    if args.preflight_only:
        if args.out_dir.exists():
            raise FileExistsError(f"output directory already exists: {args.out_dir}")
        if HEAVY_GUARD_PATH.exists():
            current = json.loads(HEAVY_GUARD_PATH.read_text(encoding="utf-8"))
            if _pid_is_alive(int(current["pid"])):
                raise RuntimeError(f"heavy guard is occupied: {current}")
        print(
            f"preflight PASS prereg={PREREGISTER_SHA256} fit=0 score=0",
            flush=True,
        )
        return
    owner = acquire_heavy_guard(HEAVY_GUARD_PATH)
    atexit.register(release_heavy_guard, HEAVY_GUARD_PATH, owner)
    print(
        f"heavy guard acquired pid={owner['pid']} token={owner['token']} ",
        f"prereg={PREREGISTER_SHA256}",
        flush=True,
    )
    try:
        run_stage1(args, preregister, owner)
    finally:
        release_heavy_guard(HEAVY_GUARD_PATH, owner)


if __name__ == "__main__":
    main()

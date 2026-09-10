"""Strict Stage1 for CUDA raw attention trained by fixed smooth-FICR loss."""

from __future__ import annotations

import argparse
import ast
import atexit
import ctypes
from importlib import metadata
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
import uuid

import joblib
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_ficr_bayes_decision_strict as bayes
from scripts import run_raw_grid_wind_lgb as raw_protocol
from scripts import run_shared_q07_multiseed as shared
from scripts import run_weather_quantile_bayes as weather_protocol
from src.manifest import describe_file, git_state, package_versions, sha256_file, utc_now
from src.metric import CAPACITY_KWH, TARGET_COLS
from src.raw_spatiotemporal_attention import (
    BATCH_SIZE,
    BLEND_WEIGHTS,
    EPOCHS,
    assert_fit_before_apply,
)
from src.raw_spatiotemporal_smooth_ficr import (
    CANDIDATE_KEYS,
    MODEL_ID,
    RawSpatiotemporalSmoothFICRRegressor,
    candidate_frame,
    select_group_candidate,
)


CONFIG_PATH = ROOT / "configs/raw_spatiotemporal_smooth_ficr_preregister_v2.json"
CONFIG_SIDECAR = CONFIG_PATH.with_suffix(".sha256")
CONFIG_SHA256 = "d7e0ae09efd195d59b4c977b55f8031f5b254f61f9e8b16770a4b78e5fd045df"
V1_CONFIG_PATH = ROOT / "configs/raw_spatiotemporal_smooth_ficr_preregister_v1.json"
V1_CONFIG_SHA256 = "0dc17775151963f9324186c922deabdc2bb1f869f051b156091255eb284dde08"
INCIDENT_PATH = ROOT / "artifacts/postgate/raw_spatiotemporal_smooth_ficr_protocol_failed_launcher_timeout_v1/protocol_failure_incident.json"
INCIDENT_SHA256 = "7fe4781bf7a3203abc383b8697b3e588c259f829f57128a47847d68cfb86527e"
LAUNCHER_PATH = ROOT / "scripts/launch_raw_spatiotemporal_smooth_ficr_v2.ps1"
FEASIBILITY_PATH = ROOT / "artifacts/audits/raw_spatiotemporal_smooth_ficr_feasibility_v1.json"
FEASIBILITY_SHA256 = "a0dc54b82f513fb1af5de79c36182ebda079cbc0885d2caa942ae5af8495b243"
RAW_DEPENDENCY_PATH = ROOT / "configs/raw_grid_wind_lgb_preregister_v1.json"
RAW_DEPENDENCY_SHA256 = "06b56426347f64d91f1a69e1c56083014f0dd003ee8186fa98ffe97ad3b61d01"
RAW_ATTENTION_CONFIG_PATH = ROOT / "configs/raw_spatiotemporal_attention_preregister_v1.json"
RAW_ATTENTION_CONFIG_SHA256 = "93f6b3f0aa19407c3d5bdc14c5529fa29e7f32b03b23dc58f87050d50f5a142d"
RAW_ATTENTION_SOURCE_PATH = ROOT / "src/raw_spatiotemporal_attention.py"
RAW_ATTENTION_SOURCE_SHA256 = "51fa4993887fbee1b5b21aa9bf81a27662c289f4764d621ca28c7a6667d94384"
SMOOTH_OBJECTIVE_PATH = ROOT / "src/smooth_ficr_objective.py"
SMOOTH_OBJECTIVE_SHA256 = "9a85699ca089f166aad78d27e1dd49c9ecbf59880309c29b3a45d63f132a8eaf"
HEAVY_GUARD_PATH = ROOT / "artifacts/locks/heavy_cpu_fit.pid.json"
DEFAULT_RAW_DIR = Path(r"data/local/open")
DEFAULT_OUT_DIR = ROOT / "artifacts/postgate/raw_spatiotemporal_smooth_ficr_strict_v2"
TIME_COL = "forecast_kst_dtm"
YEAR_2022 = raw_protocol.YEAR_2022
YEAR_2023 = raw_protocol.YEAR_2023
PRE2024 = raw_protocol.PRE2024
G3_H1 = weather_protocol._year_segments(2023)["H1"]
G3_H2 = weather_protocol._year_segments(2023)["H2"]
STAGE1_REQUIRED: Mapping[str, tuple[str, ...]] = {
    TARGET_COLS[0]: ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4"),
    TARGET_COLS[1]: ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4"),
    TARGET_COLS[2]: ("full", "Q3", "Q4"),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("stage1", "manifest", "all"), required=True)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    return parser.parse_args(argv)


def _verify_exact(path: Path, expected_sha256: str, expected_bytes: int | None = None) -> None:
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise AssertionError(f"registered file hash changed: {path}")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise AssertionError(f"registered file size changed: {path}")


def verify_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = path.resolve()
    if path != CONFIG_PATH.resolve():
        raise AssertionError("only the canonical preregistration path is allowed")
    _verify_exact(path, CONFIG_SHA256, 5351)
    expected_sidecar = f"{CONFIG_SHA256}  {CONFIG_PATH.name}\n"
    if CONFIG_SIDECAR.read_text(encoding="utf-8") != expected_sidecar:
        raise AssertionError("preregistration sidecar changed")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["experiment_id"] != "raw_spatiotemporal_smooth_ficr_strict_forward_v2":
        raise AssertionError("experiment identity changed")
    _verify_exact(V1_CONFIG_PATH, V1_CONFIG_SHA256, 11947)
    expected_v1_sidecar = f"{V1_CONFIG_SHA256}  {V1_CONFIG_PATH.name}\n"
    if V1_CONFIG_PATH.with_suffix(".sha256").read_text(encoding="utf-8") != expected_v1_sidecar:
        raise AssertionError("v1 scientific preregistration sidecar changed")
    v1 = json.loads(V1_CONFIG_PATH.read_text(encoding="utf-8"))
    if tuple(v1["candidate_family"]["fixed_blend_weights"]) != BLEND_WEIGHTS:
        raise AssertionError("blend weights changed")
    if tuple(v1["candidate_family"]["candidate_order"]) != CANDIDATE_KEYS:
        raise AssertionError("candidate order changed")
    if tuple(config["complete_scientific_contract_inheritance"]["candidate_order_unchanged"]) != CANDIDATE_KEYS:
        raise AssertionError("v2 inherited candidate order changed")
    if tuple(config["complete_scientific_contract_inheritance"]["blend_weights_unchanged"]) != BLEND_WEIGHTS:
        raise AssertionError("v2 inherited blend weights changed")
    if int(config["execution_protocol_changes"]["scientific_change_count"]) != 0:
        raise AssertionError("v2 introduced a scientific change")
    if int(config["execution_protocol_changes"]["protocol_change_count"]) != 1:
        raise AssertionError("v2 launcher change count changed")
    training = v1["training_frozen_exactly_from_v1"]
    if (
        int(training["epochs"]) != EPOCHS
        or int(training["batch_size_runs"]) != BATCH_SIZE
        or tuple(training["seeds"]) != (42, 2026)
    ):
        raise AssertionError("training identity changed")
    loss = v1["fixed_official_smooth_ficr_loss"]
    expected = {
        "official_thresholds_cf": [0.06, 0.08],
        "settlement_decomposition_weights": [1.0, 3.0],
        "mae_coefficient": 0.5,
        "temperature_cf": 0.005,
        "smooth_absolute_epsilon_cf": 0.0025,
        "sigmoid_argument_clip": [-40.0, 40.0],
    }
    for key, value in expected.items():
        if loss[key] != value:
            raise AssertionError(f"loss registration changed: {key}")
    _verify_exact(FEASIBILITY_PATH, FEASIBILITY_SHA256, 8798)
    _verify_exact(INCIDENT_PATH, INCIDENT_SHA256, 4635)
    _verify_exact(RAW_DEPENDENCY_PATH, RAW_DEPENDENCY_SHA256)
    _verify_exact(RAW_ATTENTION_CONFIG_PATH, RAW_ATTENTION_CONFIG_SHA256, 8823)
    _verify_exact(RAW_ATTENTION_SOURCE_PATH, RAW_ATTENTION_SOURCE_SHA256, 28092)
    _verify_exact(SMOOTH_OBJECTIVE_PATH, SMOOTH_OBJECTIVE_SHA256, 13130)
    raw_contract = json.loads(RAW_DEPENDENCY_PATH.read_text(encoding="utf-8"))
    return config, raw_contract


def verify_cuda_runtime_before_label_or_fit() -> dict[str, Any]:
    registered_python = (ROOT / ".venv/Scripts/python.exe").resolve()
    if Path(sys.executable).resolve() != registered_python:
        raise RuntimeError(f"registered venv Python required: {registered_python}")
    if torch.__version__ != "2.11.0+cu128" or torch.version.cuda != "12.8":
        raise RuntimeError("registered torch/CUDA runtime changed")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; CPU fallback is forbidden")
    if torch.cuda.device_count() < 1 or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 5060 Ti":
        raise RuntimeError("registered GPU identity changed")
    return {
        "python": str(registered_python),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": True,
        "cuda_device_count": torch.cuda.device_count(),
        "device": torch.cuda.get_device_name(0),
        "device_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "fallback_used": False,
    }


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return False
    ctypes.windll.kernel32.CloseHandle(handle)
    return True


def acquire_heavy_guard(path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    for _ in range(2):
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            current = json.loads(path.read_text(encoding="utf-8"))
            if _pid_alive(int(current.get("pid", -1))):
                raise RuntimeError(f"heavy guard held by live PID {current.get('pid')}: {current}")
            stale = path.with_name(
                f"heavy_cpu_fit.stale-pid{current.get('pid','unknown')}-{uuid.uuid4().hex}.json"
            )
            os.replace(path, stale)
            continue
        owner = {
            "schema_version": 1,
            "pid": os.getpid(),
            "token": token,
            "experiment_id": "raw_spatiotemporal_smooth_ficr_strict_forward_v2",
            "stage": "Stage1_GPU",
            "created_utc": utc_now(),
        }
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(owner, stream, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
        return owner
    raise RuntimeError("could not acquire shared heavy guard")


def release_heavy_guard(path: Path, owner: Mapping[str, Any]) -> None:
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        return
    if int(current.get("pid", -1)) == int(owner["pid"]) and current.get("token") == owner.get("token"):
        path.unlink()


def _module_files(module: str) -> tuple[Path, ...]:
    if not module:
        return ()
    base = ROOT.joinpath(*module.split("."))
    result: list[Path] = []
    parts = module.split(".")
    for length in range(1, len(parts)):
        initializer = ROOT.joinpath(*parts[:length]) / "__init__.py"
        if initializer.is_file():
            result.append(initializer.resolve())
    if base.with_suffix(".py").is_file():
        result.append(base.with_suffix(".py").resolve())
    if base.is_dir() and (base / "__init__.py").is_file():
        result.append((base / "__init__.py").resolve())
    return tuple(dict.fromkeys(result))


def resolve_ast_closure(entry: Path) -> tuple[Path, ...]:
    queue = [entry.resolve()]
    visited: set[Path] = set()
    while queue:
        path = queue.pop()
        if path in visited:
            continue
        if not path.is_file() or ROOT not in path.parents:
            raise AssertionError(f"invalid local closure path: {path}")
        visited.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    queue.extend(_module_files(alias.name))
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    relative = path.relative_to(ROOT).with_suffix("").parts[:-1]
                    keep = max(len(relative) - node.level + 1, 0)
                    base = ".".join((*relative[:keep], *((node.module or "").split("."))))
                else:
                    base = node.module or ""
                resolved = list(_module_files(base.strip(".")))
                for alias in node.names:
                    if alias.name != "*":
                        resolved.extend(_module_files(f"{base}.{alias.name}".strip(".")))
                if not resolved:
                    top = base.strip(".").split(".")[0]
                    if top and (ROOT / top).exists():
                        raise AssertionError(f"unresolved local import: {base}")
                queue.extend(resolved)
    return tuple(sorted(visited))


def _file_record(path: Path) -> dict[str, Any]:
    record = describe_file(path)
    record["relative_path"] = path.resolve().relative_to(ROOT).as_posix()
    return record


def _fixed_nonlabel_inputs(config: Mapping[str, Any]) -> list[Path]:
    paths = [
        CONFIG_PATH,
        CONFIG_SIDECAR,
        V1_CONFIG_PATH,
        V1_CONFIG_PATH.with_suffix(".sha256"),
        INCIDENT_PATH,
        LAUNCHER_PATH,
        FEASIBILITY_PATH,
        RAW_DEPENDENCY_PATH,
        RAW_DEPENDENCY_PATH.with_suffix(".sha256"),
        RAW_ATTENTION_CONFIG_PATH,
        RAW_ATTENTION_CONFIG_PATH.with_suffix(".sha256"),
        RAW_ATTENTION_SOURCE_PATH,
        SMOOTH_OBJECTIVE_PATH,
        ROOT / "artifacts/oof/dev2023_locked_v3.parquet",
        ROOT / "artifacts/oof/g3dev2023h2_candidates.parquet",
    ]
    return [path.resolve() for path in paths]


def source_closure(config: Mapping[str, Any], raw_dir: Path) -> dict[str, Any]:
    closure = resolve_ast_closure(Path(__file__))
    explicit = [
        LAUNCHER_PATH,
        ROOT / "tests/test_raw_spatiotemporal_smooth_ficr.py",
        ROOT / "tests/test_raw_spatiotemporal_smooth_ficr_runner.py",
    ]
    label_path = raw_dir / "train/train_labels.csv"
    return {
        "resolver": "recursive Python AST local imports plus package initializers",
        "resolved_relative_paths": [path.relative_to(ROOT).as_posix() for path in closure],
        "resolved_files": [_file_record(path) for path in closure],
        "explicit_sources": [_file_record(path.resolve()) for path in explicit],
        "unresolved_local_imports": [],
        "fixed_nonlabel_inputs": [_file_record(path) for path in _fixed_nonlabel_inputs(config)],
        "raw_label_metadata_only": {
            "path": str(label_path.resolve()),
            "bytes": label_path.stat().st_size,
            "content_values_materialized": 0,
        },
    }


def _assert_equal(left: Mapping[str, Any], right: Mapping[str, Any], message: str) -> None:
    if json.dumps(left, sort_keys=True, default=str) != json.dumps(right, sort_keys=True, default=str):
        raise AssertionError(message)


def _segments(group: str) -> dict[str, pd.DatetimeIndex]:
    year = weather_protocol._year_segments(2023)
    if group == TARGET_COLS[2]:
        return {"full": year["H2"], "Q3": year["Q3"], "Q4": year["Q4"]}
    return {name: year[name] for name in STAGE1_REQUIRED[group]}


def _capacity_factors(labels: pd.DataFrame) -> pd.DataFrame:
    result = labels.copy()
    for group in result.columns:
        result[group] = result[group] / CAPACITY_KWH[group]
    return result


def _require_candidate_lock(path: Path, required_flag: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required candidate-before-label lock is absent: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("preregister_sha256") != CONFIG_SHA256 or payload.get(required_flag) is not True:
        raise RuntimeError(f"candidate-before-label lock is not valid: {path}")
    return payload


def _read_g3_fit_labels_after_g12_lock(
    raw_dir: Path,
    prefix_spec: Mapping[str, Any],
    g12_lock: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _require_candidate_lock(
        g12_lock, "model_preprocessing_raw_predictions_and_candidates_frozen"
    )
    return raw_protocol._read_label_prefix(raw_dir, prefix_spec, TARGET_COLS)


def _read_score_labels_after_global_lock(
    raw_dir: Path,
    prefix_spec: Mapping[str, Any],
    global_lock: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _require_candidate_lock(
        global_lock, "all_models_preprocessing_predictions_candidates_frozen"
    )
    return raw_protocol._read_label_prefix(raw_dir, prefix_spec, TARGET_COLS)


def _fit_joint(
    *,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    catalogs: Mapping[str, Sequence[Mapping[str, Any]]],
    fit_index: pd.DatetimeIndex,
    application_index: pd.DatetimeIndex,
    application_groups: Sequence[str],
    baseline: pd.DataFrame,
) -> tuple[
    RawSpatiotemporalSmoothFICRRegressor,
    pd.DataFrame,
    dict[str, pd.DataFrame],
    dict[str, Any],
]:
    split = assert_fit_before_apply(fit_index, application_index)
    print(
        f"fit {MODEL_ID}: {fit_index.min()}..{fit_index.max()} -> "
        f"{application_index.min()} for {list(application_groups)}",
        flush=True,
    )
    model = RawSpatiotemporalSmoothFICRRegressor(catalogs).fit(
        features.loc[fit_index], _capacity_factors(labels.loc[fit_index])
    )
    raw_prediction = model.predict(features.loc[application_index])
    candidates = {
        group: candidate_frame(
            baseline.loc[application_index, group],
            raw_prediction[group],
            capacity_kwh=CAPACITY_KWH[group],
        )
        for group in application_groups
    }
    return model, raw_prediction, candidates, {"split": split, "model": model.metadata()}


def _save_reload_joint(
    *,
    out_dir: Path,
    fit_id: str,
    model: RawSpatiotemporalSmoothFICRRegressor,
    prediction: pd.DataFrame,
    application_features: pd.DataFrame,
    candidates: Mapping[str, pd.DataFrame],
    baselines: Mapping[str, pd.Series],
) -> tuple[list[Path], dict[str, Any]]:
    model_path = out_dir / "models" / f"stage1__{fit_id}.joblib"
    prediction_path = out_dir / "predictions" / f"stage1__{fit_id}__raw_cf.parquet"
    bayes._atomic_joblib(model, model_path)
    bayes._atomic_parquet(prediction, prediction_path)
    paths = [model_path, prediction_path]
    for group, frame in candidates.items():
        candidate_path = out_dir / "predictions" / f"stage1__{fit_id}__{group}__candidates.parquet"
        baseline_path = out_dir / "predictions" / f"stage1__{fit_id}__{group}__baseline.parquet"
        bayes._atomic_parquet(frame, candidate_path)
        bayes._atomic_parquet(baselines[group].rename(group).to_frame(), baseline_path)
        paths.extend((candidate_path, baseline_path))
    loaded: RawSpatiotemporalSmoothFICRRegressor = joblib.load(model_path)
    repeated = loaded.predict(application_features)
    if not np.array_equal(repeated.to_numpy(), prediction.to_numpy()):
        raise AssertionError(f"{fit_id} loaded neural prediction changed")
    stored_prediction = pd.read_parquet(prediction_path)
    if not np.array_equal(stored_prediction.to_numpy(), prediction.to_numpy()):
        raise AssertionError(f"{fit_id} raw prediction parquet changed")
    for group, frame in candidates.items():
        stored = pd.read_parquet(
            out_dir / "predictions" / f"stage1__{fit_id}__{group}__candidates.parquet"
        )
        if not np.array_equal(stored.to_numpy(), frame.to_numpy()):
            raise AssertionError(f"{fit_id}/{group} candidate parquet changed")
    return paths, {
        "prediction_reload_bit_exact": True,
        "prediction_and_candidate_parquet_roundtrip_bit_exact": True,
        "metadata": loaded.metadata(),
    }


def run_stage1(
    *,
    raw_dir: Path,
    artifact_root: Path,
    out_dir: Path,
    config: Mapping[str, Any],
    raw_contract: Mapping[str, Any],
    runtime: Mapping[str, Any],
    owner: Mapping[str, Any],
) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"Stage1 requires a new output directory: {out_dir}")
    out_dir.mkdir(parents=True)
    for path in (
        CONFIG_PATH,
        CONFIG_SIDECAR,
        V1_CONFIG_PATH,
        V1_CONFIG_PATH.with_suffix(".sha256"),
        INCIDENT_PATH,
        FEASIBILITY_PATH,
    ):
        bayes._copy_exclusive(path, out_dir / path.name)

    closure_before = source_closure(config, raw_dir)
    prefix = raw_contract["physical_stage1_inputs"]
    prefixes_before = {
        "labels_g12_fit": raw_protocol._prefix_snapshot(
            raw_dir / "train/train_labels.csv", prefix["labels_g12_fit_prefix"]
        ),
        "labels_g3_fit": raw_protocol._prefix_snapshot(
            raw_dir / "train/train_labels.csv", prefix["labels_g3_fit_prefix"]
        ),
        "labels_score": raw_protocol._prefix_snapshot(
            raw_dir / "train/train_labels.csv", prefix["labels_stage1_score_prefix_after_lock"]
        ),
        "ldaps": raw_protocol._prefix_snapshot(
            raw_dir / "train/ldaps_train.csv", prefix["ldaps_weather_prefix"]
        ),
        "gfs": raw_protocol._prefix_snapshot(
            raw_dir / "train/gfs_train.csv", prefix["gfs_weather_prefix"]
        ),
    }
    source_lock_path = out_dir / "source_lock_before_fit_labels.json"
    bayes._write_json(
        source_lock_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "status": "source_and_physical_prefix_hashes_locked_before_fit_label_materialization_or_candidate_fit",
            "preregister_sha256": CONFIG_SHA256,
            "runtime": dict(runtime),
            "heavy_guard_owner": dict(owner),
            "source_closure": closure_before,
            "physical_prefix_snapshots": prefixes_before,
            "fit_target_value_cells_materialized": 0,
            "application_target_value_cells_materialized": 0,
            "2024_or_2025_values_materialized": 0,
        },
    )
    print(f"source lock={sha256_file(source_lock_path)} guard PID={owner['pid']}", flush=True)

    features, weather_evidence = raw_protocol._read_weather_features(
        raw_dir, raw_contract, period="stage1"
    )
    baseline = raw_protocol._stage1_baseline(artifact_root, raw_contract)
    catalogs = raw_contract["grid_catalog"]

    labels_g12, labels_g12_evidence = raw_protocol._read_label_prefix(
        raw_dir, prefix["labels_g12_fit_prefix"], TARGET_COLS[:2]
    )
    if not labels_g12.index.equals(YEAR_2022):
        raise AssertionError("G1/G2 physical fit prefix changed")
    g12_fit_lock_path = out_dir / "g12_fit_labels_before_candidate_fit.json"
    bayes._write_json(
        g12_fit_lock_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "source_lock": describe_file(source_lock_path),
            "fit_label_evidence": labels_g12_evidence,
            "materialized_columns": list(TARGET_COLS[:2]),
            "materialized_index_start": YEAR_2022.min(),
            "materialized_index_end": YEAR_2022.max(),
            "G12_2023_application_target_cells_materialized": 0,
            "G3_H2_application_target_cells_materialized": 0,
        },
    )
    model_g12, prediction_g12, candidates_g12, training_g12 = _fit_joint(
        features=features,
        labels=labels_g12,
        catalogs=catalogs,
        fit_index=YEAR_2022,
        application_index=YEAR_2023,
        application_groups=TARGET_COLS[:2],
        baseline=baseline,
    )
    paths_g12, replay_g12 = _save_reload_joint(
        out_dir=out_dir,
        fit_id="g12_2022",
        model=model_g12,
        prediction=prediction_g12,
        application_features=features.loc[YEAR_2023],
        candidates=candidates_g12,
        baselines={group: baseline.loc[YEAR_2023, group] for group in TARGET_COLS[:2]},
    )
    record_g12 = out_dir / "stage1_g12_candidate_record_before_application_labels.json"
    bayes._write_json(
        record_g12,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": CONFIG_SHA256,
            "fit_label_lock": describe_file(g12_fit_lock_path),
            "training": training_g12,
            "reload": replay_g12,
            "outputs": [describe_file(path) for path in paths_g12],
            "G12_2023_application_target_cells_materialized_before_this_record": 0,
        },
    )
    lock_g12 = out_dir / "stage1_g12_candidate_lock_before_application_labels.json"
    bayes._write_json(
        lock_g12,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": CONFIG_SHA256,
            "record": describe_file(record_g12),
            "candidate_artifacts": [describe_file(path) for path in paths_g12],
            "model_preprocessing_raw_predictions_and_candidates_frozen": True,
            "G12_2023_application_target_cells_materialized": 0,
        },
    )
    print(f"G12 candidate-before-label lock={sha256_file(lock_g12)}", flush=True)

    labels_g3, labels_g3_evidence = _read_g3_fit_labels_after_g12_lock(
        raw_dir, prefix["labels_g3_fit_prefix"], lock_g12
    )
    expected_g3_prefix = YEAR_2022.append(G3_H1)
    if not labels_g3.index.equals(expected_g3_prefix):
        raise AssertionError("G3 physical fit prefix changed")
    g3_fit_lock_path = out_dir / "g3_fit_labels_after_g12_lock_before_candidate_fit.json"
    bayes._write_json(
        g3_fit_lock_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "g12_candidate_lock": describe_file(lock_g12),
            "fit_label_evidence": labels_g3_evidence,
            "materialized_columns": list(TARGET_COLS),
            "materialized_index_start": expected_g3_prefix.min(),
            "materialized_index_end": expected_g3_prefix.max(),
            "G12_candidate_frozen_before_G12_2023H1_auxiliary_cells": True,
            "G3_H2_application_target_cells_materialized": 0,
        },
    )
    model_g3, prediction_g3, candidates_g3, training_g3 = _fit_joint(
        features=features,
        labels=labels_g3,
        catalogs=catalogs,
        fit_index=expected_g3_prefix,
        application_index=G3_H2,
        application_groups=(TARGET_COLS[2],),
        baseline=baseline,
    )
    paths_g3, replay_g3 = _save_reload_joint(
        out_dir=out_dir,
        fit_id="g3_through_2023h1",
        model=model_g3,
        prediction=prediction_g3,
        application_features=features.loc[G3_H2],
        candidates=candidates_g3,
        baselines={TARGET_COLS[2]: baseline.loc[G3_H2, TARGET_COLS[2]]},
    )
    record_g3 = out_dir / "stage1_g3_candidate_record_before_application_labels.json"
    bayes._write_json(
        record_g3,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": CONFIG_SHA256,
            "g12_candidate_lock": describe_file(lock_g12),
            "fit_label_lock": describe_file(g3_fit_lock_path),
            "training": training_g3,
            "reload": replay_g3,
            "outputs": [describe_file(path) for path in paths_g3],
            "G3_H2_application_target_cells_materialized_before_this_record": 0,
        },
    )
    lock_g3 = out_dir / "stage1_g3_candidate_lock_before_application_labels.json"
    bayes._write_json(
        lock_g3,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": CONFIG_SHA256,
            "record": describe_file(record_g3),
            "g12_candidate_lock": describe_file(lock_g12),
            "candidate_artifacts": [describe_file(path) for path in paths_g3],
            "model_preprocessing_raw_predictions_and_candidates_frozen": True,
            "G3_H2_application_target_cells_materialized": 0,
        },
    )
    print(f"G3 candidate-before-label lock={sha256_file(lock_g3)}", flush=True)

    all_paths = [*paths_g12, *paths_g3]
    global_record = out_dir / "stage1_global_candidate_record_before_score_labels.json"
    bayes._write_json(
        global_record,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": CONFIG_SHA256,
            "g12_candidate_lock": describe_file(lock_g12),
            "g3_candidate_lock": describe_file(lock_g3),
            "weather_evidence": weather_evidence,
            "outputs": [describe_file(path) for path in all_paths],
            "validation_target_columns_materialized_for_metric": [],
            "all_group_candidate_arrays_frozen_before_score_reader": True,
            "2024_weather_read": False,
            "2024_label_read": False,
            "2025_read": False,
        },
    )
    global_lock = out_dir / "stage1_global_candidate_lock_before_score_labels.json"
    bayes._write_json(
        global_lock,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": CONFIG_SHA256,
            "record": describe_file(global_record),
            "candidate_artifacts": [describe_file(path) for path in all_paths],
            "all_models_preprocessing_predictions_candidates_frozen": True,
            "metric_calls_before_lock": 0,
        },
    )
    print(f"global candidate-before-score lock={sha256_file(global_lock)}", flush=True)

    score_labels, score_label_evidence = _read_score_labels_after_global_lock(
        raw_dir, prefix["labels_stage1_score_prefix_after_lock"], global_lock
    )
    if not score_labels.index.equals(PRE2024):
        raise AssertionError("Stage1 score prefix changed")
    score_access_path = out_dir / "stage1_score_label_access_after_candidate_lock.json"
    bayes._write_json(
        score_access_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "global_candidate_lock": describe_file(global_lock),
            "score_label_evidence": score_label_evidence,
            "metric_calls_before_this_access_record": 0,
            "materialized_columns": list(TARGET_COLS),
            "materialized_index_start": PRE2024.min(),
            "materialized_index_end": PRE2024.max(),
        },
    )

    group_results: dict[str, Any] = {}
    locked: dict[str, str | None] = {}
    passed: list[str] = []
    comparison_count = 0
    for group in TARGET_COLS:
        application = YEAR_2023 if group in TARGET_COLS[:2] else G3_H2
        group_candidates = candidates_g12[group] if group in TARGET_COLS[:2] else candidates_g3[group]
        comparisons = {
            key: bayes._comparison(
                score_labels.loc[application, group],
                baseline.loc[application, group],
                group_candidates[key],
                group,
                _segments(group),
            )
            for key in CANDIDATE_KEYS
        }
        comparison_count += sum(len(records) for records in comparisons.values())
        selected, audit = select_group_candidate(comparisons, STAGE1_REQUIRED[group])
        locked[group] = selected
        if selected is not None:
            passed.append(group)
        group_results[group] = {
            "comparisons": comparisons,
            "selection": audit,
            "locked_candidate": selected if selected is not None else "identity",
            "passed": selected is not None,
        }
    if comparison_count != 34:
        raise AssertionError("Stage1 comparison record count changed")

    _assert_equal(
        closure_before,
        source_closure(config, raw_dir),
        "source/input closure changed during Stage1",
    )
    prefixes_after = {
        "labels_g12_fit": raw_protocol._prefix_snapshot(
            raw_dir / "train/train_labels.csv", prefix["labels_g12_fit_prefix"]
        ),
        "labels_g3_fit": raw_protocol._prefix_snapshot(
            raw_dir / "train/train_labels.csv", prefix["labels_g3_fit_prefix"]
        ),
        "labels_score": raw_protocol._prefix_snapshot(
            raw_dir / "train/train_labels.csv", prefix["labels_stage1_score_prefix_after_lock"]
        ),
        "ldaps": raw_protocol._prefix_snapshot(
            raw_dir / "train/ldaps_train.csv", prefix["ldaps_weather_prefix"]
        ),
        "gfs": raw_protocol._prefix_snapshot(
            raw_dir / "train/gfs_train.csv", prefix["gfs_weather_prefix"]
        ),
    }
    _assert_equal(prefixes_before, prefixes_after, "physical input prefix changed during Stage1")

    result = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "preregister_sha256": CONFIG_SHA256,
        "global_candidate_lock": describe_file(global_lock),
        "score_label_access_after_candidate_lock": describe_file(score_access_path),
        "score_label_evidence": score_label_evidence,
        "training": {"g12": training_g12, "g3": training_g3},
        "group_results": group_results,
        "registered_slice_count": 17,
        "candidate_comparison_record_count": comparison_count,
        "locked_candidates": locked,
        "passed_groups": passed,
        "stage2_unlocked": bool(passed),
        "full_component_gate_required": True,
        "2024_weather_read": False,
        "2024_label_read": False,
        "2025_read": False,
        "leaderboard_score_claim": False,
    }
    result_path = out_dir / "stage1_results.json"
    bayes._write_json(result_path, result)
    promotion = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "preregister_sha256": CONFIG_SHA256,
        "global_candidate_lock": describe_file(global_lock),
        "score_label_access": describe_file(score_access_path),
        "stage1_results": describe_file(result_path),
        "locked_candidates": locked,
        "passed_groups": passed,
        "stage2_unlocked": bool(passed),
        "all_17_group_slices_strict_and_full_components_nonnegative_required": True,
        "no_2024_or_2025_read": True,
    }
    promotion_path = out_dir / "stage1_promotion_lock.json"
    bayes._write_json(promotion_path, promotion)
    if not passed:
        bayes._write_json(
            out_dir / "stage1_rejection.json",
            {
                "schema_version": 1,
                "created_utc": utc_now(),
                "preregister_sha256": CONFIG_SHA256,
                "promotion_lock": describe_file(promotion_path),
                "decision": "REJECT_IDENTITY",
                "reason": "no group passed every registered slice and both full component gates",
                "no_retune_retry_rescue_2024_2025_or_csv": True,
            },
        )
    print(f"Stage1 passed groups: {passed}", flush=True)
    return promotion


def write_manifest(
    *,
    out_dir: Path,
    raw_dir: Path,
    config: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    path = out_dir / "manifest.json"
    sidecar = out_dir / "manifest.sha256"
    if path.exists() or sidecar.exists():
        raise FileExistsError("manifest already exists")
    result = json.loads((out_dir / "stage1_results.json").read_text(encoding="utf-8"))
    outputs = sorted(
        item
        for item in out_dir.rglob("*")
        if item.is_file() and item not in (path, sidecar) and ".tmp-" not in item.name
    )
    payload = {
        "schema_version": 1,
        "artifact_type": "raw_spatiotemporal_smooth_ficr_strict_forward_v2_stage1",
        "created_utc": utc_now(),
        "command": [sys.executable, *sys.argv],
        "runtime": {
            **dict(runtime),
            "packages": {**package_versions(), "torch": metadata.version("torch")},
            "git": git_state(ROOT),
        },
        "preregister_sha256": CONFIG_SHA256,
        "feasibility_sha256": FEASIBILITY_SHA256,
        "source_closure": source_closure(config, raw_dir),
        "contracts": {
            "single_changed_factor_loss_only": True,
            "architecture_id": "node32_geoattn_conv48_e240_s2",
            "model_id": MODEL_ID,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "blend_weights": list(BLEND_WEIGHTS),
            "registered_group_slice_count": 17,
            "full_NMAE_and_FICR_nonnegative_required": True,
            "cuda_only_no_fallback": True,
            "candidate_locked_before_application_score_labels": True,
        },
        "selection": {
            "locked_candidates": result["locked_candidates"],
            "passed_groups": result["passed_groups"],
            "stage2_unlocked": result["stage2_unlocked"],
        },
        "read_flags": {
            "2024_weather_read": False,
            "2024_label_read": False,
            "2025_weather_read": False,
            "sample_submission_read": False,
            "csv_created": False,
        },
        "tests": {
            "focused_command": ".venv/Scripts/python.exe -B -m unittest tests.test_raw_spatiotemporal_smooth_ficr tests.test_raw_spatiotemporal_smooth_ficr_runner -v",
            "focused_expected": 15,
            "focused_status": "pass",
            "full_repository_status_before_fit": "pass",
        },
        "outputs_excluding_manifest_and_sidecar": [describe_file(item) for item in outputs],
        "output_count_excluding_manifest_and_sidecar": len(outputs),
    }
    bayes._write_json(path, payload)
    with sidecar.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(f"{sha256_file(path)}  manifest.json\n")
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config, raw_contract = verify_config(args.config)
    runtime = verify_cuda_runtime_before_label_or_fit()
    raw_dir = args.raw_dir.resolve()
    artifact_root = args.artifact_root.resolve()
    out_dir = args.out_dir.resolve()
    owner = acquire_heavy_guard(HEAVY_GUARD_PATH)
    atexit.register(release_heavy_guard, HEAVY_GUARD_PATH, owner)
    try:
        if args.stage in ("stage1", "all"):
            run_stage1(
                raw_dir=raw_dir,
                artifact_root=artifact_root,
                out_dir=out_dir,
                config=config,
                raw_contract=raw_contract,
                runtime=runtime,
                owner=owner,
            )
        if args.stage in ("manifest", "all"):
            write_manifest(
                out_dir=out_dir,
                raw_dir=raw_dir,
                config=config,
                runtime=runtime,
            )
    finally:
        release_heavy_guard(HEAVY_GUARD_PATH, owner)


if __name__ == "__main__":
    main()

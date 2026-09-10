"""Run the preregistered cutoff-safe JMA paired incremental-delta experiment."""

from __future__ import annotations

import argparse
import ast
import ctypes
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence
import uuid

import joblib
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts import run_shared_q07_multiseed as shared  # noqa: E402
from src.features import WeatherFeatureBuilder, load_group_sites, read_weather_csv  # noqa: E402
from src.jma_paired_increment import (  # noqa: E402
    CANDIDATES,
    CONTROL_DISAGREEMENT_COLUMN,
    JMA_COLUMNS,
    MODEL_PARAMETERS,
    TRANSFER_WEIGHT,
    build_jma_features,
    eligible_rows,
    extended_features,
    fit_direct_model,
    paired_increment,
    predict_cf,
    select_stage1_candidate,
    transfer_increment,
)
from src.manifest import (  # noqa: E402
    describe_file,
    git_state,
    package_versions,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from src.metric import CAPACITY_KWH, TARGET_COLS, group_metrics, score_details  # noqa: E402


V1_SHA = "b3f03c8d022ee212bdbbdfc9f707b6f54b6b8965ad9c338bf15985a5e1597aa2"
V2_SHA = "99356308ebddfd5acd4b8d6c700c04e701a94c59bc32eb788ab5c5a65e46909e"
V3_SHA = "cb5ff8ebb613230e67487499acfcc8f4030d55d637d43d357582d76a72b7103e"
V4_SHA = "f1568dd08d1b9ad8e6d5667486e90b2608b994954aa6dd3cb2aa70864d640f80"
V5_SHA = "c0a3dc25366b518fcc3fdf3ece4e7105ddefe3a3bfe7bb97b717da4838a2791a"
YEAR_2022 = pd.date_range("2022-01-01 01:00", "2023-01-01 00:00", freq="h", name="forecast_kst_dtm")
YEAR_2023 = pd.date_range("2023-01-01 01:00", "2024-01-01 00:00", freq="h", name="forecast_kst_dtm")
YEAR_2024 = pd.date_range("2024-01-01 01:00", "2025-01-01 00:00", freq="h", name="forecast_kst_dtm")
PRE2024 = pd.date_range("2022-01-01 01:00", "2024-01-01 00:00", freq="h", name="forecast_kst_dtm")
THROUGH2024 = pd.date_range("2022-01-01 01:00", "2025-01-01 00:00", freq="h", name="forecast_kst_dtm")
G3_H1 = pd.date_range("2023-01-01 01:00", "2023-07-01 00:00", freq="h", name="forecast_kst_dtm")
G3_H2 = pd.date_range("2023-07-01 01:00", "2024-01-01 00:00", freq="h", name="forecast_kst_dtm")
STAGE1_BOUNDARY = pd.Timestamp("2024-01-01 00:00")
BOUNDARY = pd.Timestamp("2025-01-01 00:00")
JMA_SHA = "23a424bbaa54fddae3cb14440bff5ed222983359b45e3a81e6854b8a0c9509ff"
JMA_BYTES = 350241
FULL_RAW_SHA = {
    "ldaps": "61ae944e7ae1fcb17391be6737792a2205c6507bf2446ed5d9d0daf07fdea026",
    "gfs": "cd56b67d357e7bbaff5d0d51d3537d935c9e7a3f012e9f37516bdc4d38c66a5d",
}
HEAVY_GUARD = PROJECT_DIR / "artifacts/locks/heavy_cpu_fit.pid.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("stage1", "stage2"), default="stage1")
    parser.add_argument("--raw-dir", type=Path, default=Path(r"data/local/open"))
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs/jma_gsm_paired_increment_preregister_v5.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_DIR / "artifacts/postgate/jma_gsm_paired_increment_strict_v5",
    )
    return parser.parse_args(argv)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool = False) -> None:
    write_json_atomic(path, _json_ready(payload), overwrite=overwrite)


def _copy_exclusive(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        frame.to_parquet(temporary, engine="pyarrow", index=True)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_joblib(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        joblib.dump(model, temporary, compress=3)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _verify_file(spec: Mapping[str, Any], *, base: Path = PROJECT_DIR) -> Path:
    path = Path(str(spec["path"]))
    if not path.is_absolute():
        path = base / path
    expected_bytes = int(spec.get("bytes", spec.get("size_bytes")))
    if not path.is_file() or path.stat().st_size != expected_bytes or sha256_file(path) != str(spec["sha256"]):
        raise AssertionError(f"registered file identity differs: {path}")
    return path.resolve()


def _tree_digest(root: Path) -> tuple[int, str]:
    records = []
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.relative_to(root).as_posix()):
        records.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return len(records), hashlib.sha256(payload).hexdigest()


def verify_config(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if sha256_file(path) != V5_SHA:
        raise AssertionError("v5 preregistration hash differs")
    sidecar = f"{V5_SHA}  {path.name}\n"
    if path.with_suffix(".sha256").read_text(encoding="utf-8") != sidecar:
        raise AssertionError("v5 sidecar differs")
    v5 = json.loads(path.read_text(encoding="utf-8"))
    v4_path = _verify_file(v5["base_v4_preregister"])
    if sha256_file(v4_path) != V4_SHA:
        raise AssertionError("v4 preregistration hash differs")
    if v4_path.with_suffix(".sha256").read_text(encoding="utf-8") != f"{V4_SHA}  {v4_path.name}\n":
        raise AssertionError("v4 sidecar differs")
    v4 = json.loads(v4_path.read_text(encoding="utf-8"))
    v3_path = _verify_file(v4["base_v3_preregister"])
    if sha256_file(v3_path) != V3_SHA:
        raise AssertionError("v3 preregistration hash differs")
    if v3_path.with_suffix(".sha256").read_text(encoding="utf-8") != f"{V3_SHA}  {v3_path.name}\n":
        raise AssertionError("v3 sidecar differs")
    v3 = json.loads(v3_path.read_text(encoding="utf-8"))
    v2_path = _verify_file(v3["base_preregister_v2"])
    if sha256_file(v2_path) != V2_SHA:
        raise AssertionError("v2 preregistration hash differs")
    if v2_path.with_suffix(".sha256").read_text(encoding="utf-8") != f"{V2_SHA}  {v2_path.name}\n":
        raise AssertionError("v2 sidecar differs")
    v2 = json.loads(v2_path.read_text(encoding="utf-8"))
    v1_path = _verify_file(v2["base_preregister"])
    if sha256_file(v1_path) != V1_SHA:
        raise AssertionError("v1 preregistration hash differs")
    if v1_path.with_suffix(".sha256").read_text(encoding="utf-8") != f"{V1_SHA}  {v1_path.name}\n":
        raise AssertionError("v1 sidecar differs")
    v1 = json.loads(v1_path.read_text(encoding="utf-8"))
    if tuple(v1["stage1"]["candidate_ids_in_fixed_order"]) != CANDIDATES:
        raise AssertionError("candidate order differs")
    if float(v1["paired_model"]["transfer_weight"]) != TRANSFER_WEIGHT:
        raise AssertionError("transfer weight differs")
    if v1["paired_model"]["parameters"] != MODEL_PARAMETERS:
        raise AssertionError("model parameters differ")
    if v2["effective_contract"]["boundary_row"]["paired_increment_cf"] != 0.0:
        raise AssertionError("boundary patch differs")
    if v3["immutable_inheritance"]["new_candidates_or_weights"] != 0:
        raise AssertionError("v3 changed the candidate family")
    for record in v3["v2_protocol_failure"]["quarantined_files"]:
        _verify_file(record)
    _verify_file(v4["supersession"]["superseded_access_addendum"])
    physical = v4["physical_stage1_JMA_source"]
    for record in [physical["normalized"], physical["source_manifest"], physical["downloader"], *physical["raw_responses"]]:
        _verify_file(record)
    if v4["immutable_inheritance"]["new_candidate_or_weight_count"] != 0:
        raise AssertionError("v4 changed the candidate family")
    quarantine = PROJECT_DIR / v5["v4_failure_and_quarantine"]["quarantine_root"]
    count, digest = _tree_digest(quarantine)
    expected_tree = v5["v4_failure_and_quarantine"]["quarantined_tree_digest"]
    if count != int(v5["v4_failure_and_quarantine"]["quarantined_file_count"]) or digest != expected_tree["sha256"]:
        raise AssertionError("v4 quarantine tree differs")
    if v5["immutable_inheritance"]["new_candidate_or_weight_count"] != 0:
        raise AssertionError("v5 changed the candidate family")
    return v1, v2, v3, v4, v5


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_label_slice(
    path: Path,
    spec: Mapping[str, Any],
    *,
    expected_index: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Decode only registered target columns from one exact byte-bounded row slice."""

    columns = tuple(str(column) for column in spec["columns"])
    if not columns or columns[0] != "kst_dtm" or any(column not in ("kst_dtm", *TARGET_COLS) for column in columns):
        raise AssertionError("label slice columns differ")
    header_bytes = 48
    start = int(spec["source_start_byte"])
    end = int(spec["source_end_byte"])
    if not (header_bytes <= start < end <= path.stat().st_size):
        raise AssertionError("label slice byte bounds differ")
    with path.open("rb") as stream:
        header = stream.readline()
        if len(header) != header_bytes:
            raise AssertionError("label header byte count differs")
        stream.seek(start)
        body = stream.read(end - start)
        if stream.tell() != end:
            raise AssertionError("label slice reader crossed byte boundary")
    parser_payload = header + body
    if len(parser_payload) != int(spec["parser_bytes"]):
        raise AssertionError("label parser byte count differs")
    if _sha256_bytes(parser_payload) != str(spec["parser_sha256"]):
        raise AssertionError("label parser slice SHA differs")
    frame = pd.read_csv(
        io.BytesIO(parser_payload),
        encoding="utf-8-sig",
        usecols=list(columns),
        nrows=int(spec["nrows"]),
        memory_map=False,
    )
    if tuple(frame.columns) != columns or len(frame) != int(spec["nrows"]):
        raise AssertionError("label slice schema/row count differs")
    times = pd.to_datetime(frame.pop("kst_dtm"), errors="raise")
    frame.index = pd.DatetimeIndex(times, name="forecast_kst_dtm")
    if not frame.index.equals(expected_index):
        raise AssertionError("label slice timestamp sequence differs")
    frame = frame.astype(np.float64)
    return frame, {
        "phase": spec["phase"],
        "columns_materialized": list(frame.columns),
        "label_value_cells_materialized": int(frame.shape[0] * frame.shape[1]),
        "rows": int(len(frame)),
        "start": frame.index.min(),
        "end": frame.index.max(),
        "source_start_byte": start,
        "source_end_byte": end,
        "parser_bytes": len(parser_payload),
        "parser_sha256": _sha256_bytes(parser_payload),
        "other_target_columns_materialized": [],
    }


def _read_stage1_fit_labels(
    args: argparse.Namespace, v3: Mapping[str, Any]
) -> tuple[dict[str, pd.Series], dict[str, Any]]:
    source = _verify_file(v3["label_slice_contract"]["source"], base=args.raw_dir)
    g12, evidence_g12 = _read_label_slice(
        source, v3["label_slice_contract"]["stage1_g12_fit"], expected_index=YEAR_2022
    )
    g3, evidence_g3 = _read_label_slice(
        source, v3["label_slice_contract"]["stage1_g3_fit"], expected_index=G3_H1
    )
    if tuple(g12.columns) != TARGET_COLS[:2] or tuple(g3.columns) != (TARGET_COLS[2],):
        raise AssertionError("Stage1 fit-label group isolation differs")
    return {
        TARGET_COLS[0]: g12[TARGET_COLS[0]],
        TARGET_COLS[1]: g12[TARGET_COLS[1]],
        TARGET_COLS[2]: g3[TARGET_COLS[2]],
    }, {"g12": evidence_g12, "g3": evidence_g3, "score_label_value_cells_materialized": 0}


def _verify_candidate_lock(
    record: Mapping[str, Any], *, expected_kind: str, expected_config_sha: str
) -> Path:
    lock_path = _verify_file(record)
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    if payload.get("lock_kind") != expected_kind or payload.get("config_v5_sha256") != expected_config_sha:
        raise AssertionError("candidate-before-score lock identity differs")
    if payload.get("score_label_value_cells_before_lock") != 0:
        raise AssertionError("candidate lock was written after score-label access")
    return lock_path


def _read_stage1_score_labels(
    args: argparse.Namespace,
    v3: Mapping[str, Any],
    *,
    candidate_lock: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    # Fail closed before opening the label source or decoding a single target cell.
    _verify_candidate_lock(
        candidate_lock,
        expected_kind="stage1_candidate_before_score_labels",
        expected_config_sha=V5_SHA,
    )
    source = _verify_file(v3["label_slice_contract"]["source"], base=args.raw_dir)
    g12, evidence_g12 = _read_label_slice(
        source, v3["label_slice_contract"]["stage1_g12_score"], expected_index=YEAR_2023
    )
    g3, evidence_g3 = _read_label_slice(
        source, v3["label_slice_contract"]["stage1_g3_score"], expected_index=G3_H2
    )
    score = pd.DataFrame(np.nan, index=YEAR_2023, columns=TARGET_COLS, dtype=np.float64)
    score.loc[YEAR_2023, TARGET_COLS[:2]] = g12.loc[YEAR_2023, list(TARGET_COLS[:2])].to_numpy(np.float64)
    score.loc[G3_H2, TARGET_COLS[2]] = g3.loc[G3_H2, TARGET_COLS[2]].to_numpy(np.float64)
    return score, {
        "candidate_lock_verified_before_source_open": True,
        "g12": evidence_g12,
        "g3": evidence_g3,
        "label_value_cells_materialized": int(evidence_g12["label_value_cells_materialized"] + evidence_g3["label_value_cells_materialized"]),
    }


def _uint8_mask_sha256(mask: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(mask.astype(np.uint8)).tobytes()).hexdigest()


def _verify_stage1_score_masks(
    score: pd.DataFrame, v5: Mapping[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    contract = v5["stage1_score_actual_mask_contract"]
    indexes = {TARGET_COLS[0]: YEAR_2023, TARGET_COLS[1]: YEAR_2023, TARGET_COLS[2]: G3_H2}
    eligible_masks: dict[str, np.ndarray] = {}
    evidence: dict[str, Any] = {}
    for group, index in indexes.items():
        actual = score.loc[index, group].to_numpy(np.float64)
        nonfinite = ~np.isfinite(actual)
        eligible = np.isfinite(actual) & (actual >= 0.10 * CAPACITY_KWH[group])
        expected = contract[group]
        observed = {
            "rows": len(actual),
            "nonfinite_count": int(nonfinite.sum()),
            "nonfinite_mask_uint8_sha256": _uint8_mask_sha256(nonfinite),
            "eligible_count": int(eligible.sum()),
            "eligible_mask_uint8_sha256": _uint8_mask_sha256(eligible),
        }
        if observed != expected or not eligible.any():
            raise AssertionError(f"{group} Stage1 official actual-mask contract differs")
        eligible_masks[group] = eligible
        evidence[group] = observed
    return eligible_masks, evidence


def _verify_stage1_forecasts_on_eligible_rows(
    score: pd.DataFrame,
    baseline: pd.DataFrame,
    predictions: Mapping[str, pd.DataFrame],
    eligible_masks: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    indexes = {TARGET_COLS[0]: YEAR_2023, TARGET_COLS[1]: YEAR_2023, TARGET_COLS[2]: G3_H2}
    evidence: dict[str, Any] = {}
    for group, index in indexes.items():
        eligible = eligible_masks[group]
        if not eligible.any():
            raise AssertionError(f"{group} has no eligible Stage1 rows")
        if not np.isfinite(baseline.loc[index, group].to_numpy(np.float64)[eligible]).all():
            raise AssertionError(f"{group} baseline is nonfinite on eligible rows")
        for candidate in CANDIDATES:
            values = predictions[candidate].loc[index, group].to_numpy(np.float64)
            if not np.isfinite(values[eligible]).all():
                raise AssertionError(f"{group}/{candidate} is nonfinite on eligible rows")
        evidence[group] = {
            "eligible_rows": int(eligible.sum()),
            "baseline_finite": True,
            "candidate_finite": {candidate: True for candidate in CANDIDATES},
        }
    return evidence


def _read_stage2_fit_labels(
    args: argparse.Namespace, v3: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = _verify_file(v3["label_slice_contract"]["source"], base=args.raw_dir)
    return _read_label_slice(
        source, v3["label_slice_contract"]["stage2_fit"], expected_index=PRE2024
    )


def _read_stage2_score_labels(
    args: argparse.Namespace,
    v3: Mapping[str, Any],
    *,
    candidate_lock: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _verify_candidate_lock(
        candidate_lock,
        expected_kind="stage2_candidate_before_2024_score_labels",
        expected_config_sha=V5_SHA,
    )
    source = _verify_file(v3["label_slice_contract"]["source"], base=args.raw_dir)
    frame, evidence = _read_label_slice(
        source, v3["label_slice_contract"]["stage2_score"], expected_index=YEAR_2024
    )
    if tuple(frame.columns) != TARGET_COLS:
        raise AssertionError("Stage2 score-label schema differs")
    evidence["candidate_lock_verified_before_source_open"] = True
    return frame, evidence


def _module_files(module: str) -> list[Path]:
    if not module:
        return []
    parts = module.split(".")
    found: set[Path] = set()
    file_path = PROJECT_DIR.joinpath(*parts).with_suffix(".py")
    init_path = PROJECT_DIR.joinpath(*parts, "__init__.py")
    if file_path.is_file():
        found.add(file_path.resolve())
    if init_path.is_file():
        found.add(init_path.resolve())
    for depth in range(1, len(parts)):
        init = PROJECT_DIR.joinpath(*parts[:depth], "__init__.py")
        if init.is_file():
            found.add(init.resolve())
    return sorted(found)


def resolve_ast_local_import_closure(entrypoint: Path) -> tuple[Path, ...]:
    queue = [entrypoint.resolve()]
    visited: set[Path] = set()
    while queue:
        path = queue.pop()
        if path in visited:
            continue
        visited.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    resolved = _module_files(alias.name)
                    if resolved:
                        queue.extend(resolved)
                    elif (PROJECT_DIR / alias.name.split(".")[0]).exists():
                        raise AssertionError(f"unresolved local import: {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    relative = path.relative_to(PROJECT_DIR).with_suffix("").parts[:-1]
                    keep = len(relative) - node.level + 1
                    prefix = relative[: max(keep, 0)]
                    base = ".".join((*prefix, *(node.module or "").split(".")))
                else:
                    base = node.module or ""
                resolved = _module_files(base.strip("."))
                alias_resolved: list[Path] = []
                for alias in node.names:
                    if alias.name != "*":
                        alias_resolved.extend(
                            _module_files(f"{base}.{alias.name}".strip("."))
                        )
                if resolved or alias_resolved:
                    queue.extend(resolved)
                    queue.extend(alias_resolved)
                else:
                    top = base.strip(".").split(".")[0]
                    if top and (PROJECT_DIR / top).exists():
                        raise AssertionError(f"unresolved local import: {base}")
                for alias in node.names:
                    if alias.name != "*":
                        queue.extend(_module_files(f"{base}.{alias.name}".strip(".")))
    return tuple(sorted(visited))


def _closure_record(args: argparse.Namespace) -> dict[str, Any]:
    closure = resolve_ast_local_import_closure(Path(__file__))
    test = PROJECT_DIR / "tests/test_jma_gsm_paired_increment.py"
    v1_path = PROJECT_DIR / "configs/jma_gsm_paired_increment_preregister_v1.json"
    v2_path = PROJECT_DIR / "configs/jma_gsm_paired_increment_preregister_v2.json"
    v3_path = PROJECT_DIR / "configs/jma_gsm_paired_increment_preregister_v3.json"
    v4_path = PROJECT_DIR / "configs/jma_gsm_paired_increment_preregister_v4.json"
    v4 = json.loads(v4_path.read_text(encoding="utf-8"))
    physical = v4["physical_stage1_JMA_source"]
    return {
        "resolver": "recursive Python AST local imports plus package initializers",
        "entrypoint": Path(__file__).resolve().relative_to(PROJECT_DIR).as_posix(),
        "resolved_relative_paths": [path.relative_to(PROJECT_DIR).as_posix() for path in closure],
        "resolved_file_count": len(closure),
        "resolved_files": [describe_file(path) for path in closure],
        "unresolved_local_imports": [],
        "test": describe_file(test),
        "v1_config": describe_file(v1_path),
        "v1_sidecar": describe_file(v1_path.with_suffix(".sha256")),
        "v2_config": describe_file(v2_path),
        "v2_sidecar": describe_file(v2_path.with_suffix(".sha256")),
        "v3_config": describe_file(v3_path),
        "v3_sidecar": describe_file(v3_path.with_suffix(".sha256")),
        "v4_config": describe_file(v4_path),
        "v4_sidecar": describe_file(v4_path.with_suffix(".sha256")),
        "v5_config": describe_file(args.config.resolve()),
        "v5_sidecar": describe_file(args.config.with_suffix(".sha256").resolve()),
        "physical_stage1_JMA_source": {
            "normalized": describe_file(_verify_file(physical["normalized"])),
            "source_manifest": describe_file(_verify_file(physical["source_manifest"])),
            "raw_responses": [describe_file(_verify_file(record)) for record in physical["raw_responses"]],
            "downloader": describe_file(_verify_file(physical["downloader"])),
        },
    }


def _assert_closure_equal(first: Mapping[str, Any], second: Mapping[str, Any]) -> None:
    if _json_ready(first) != _json_ready(second):
        raise AssertionError("source closure changed")


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(process_query_limited_information, False, int(pid))
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class HeavyFitGuard:
    def __init__(self, *, stage: str) -> None:
        self.stage = stage
        self.token = uuid.uuid4().hex
        self.acquired = False

    def __enter__(self) -> "HeavyFitGuard":
        HEAVY_GUARD.parent.mkdir(parents=True, exist_ok=True)
        if HEAVY_GUARD.exists():
            record = json.loads(HEAVY_GUARD.read_text(encoding="utf-8"))
            if _pid_alive(int(record.get("pid", -1))):
                raise RuntimeError(f"heavy fit guard held by live PID {record.get('pid')}: {record}")
            stale = HEAVY_GUARD.with_name(
                f"heavy_cpu_fit.stale-pid{record.get('pid','unknown')}-{uuid.uuid4().hex}.json"
            )
            os.replace(HEAVY_GUARD, stale)
        payload = {
            "schema_version": 1,
            "pid": os.getpid(),
            "token": self.token,
            "experiment": "jma_gsm_paired_increment_strict_v5",
            "stage": self.stage,
            "created_utc": utc_now(),
        }
        fd = os.open(HEAVY_GUARD, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        self.acquired = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.acquired and HEAVY_GUARD.exists():
            record = json.loads(HEAVY_GUARD.read_text(encoding="utf-8"))
            if record.get("token") == self.token and int(record.get("pid", -1)) == os.getpid():
                HEAVY_GUARD.unlink()


def _segments(year: int) -> dict[str, pd.DatetimeIndex]:
    make = lambda a, b: pd.date_range(a, b, freq="h", name="forecast_kst_dtm")
    return {
        "full": make(f"{year}-01-01 01:00", f"{year + 1}-01-01 00:00"),
        "H1": make(f"{year}-01-01 01:00", f"{year}-07-01 00:00"),
        "H2": make(f"{year}-07-01 01:00", f"{year + 1}-01-01 00:00"),
        "Q1": make(f"{year}-01-01 01:00", f"{year}-04-01 00:00"),
        "Q2": make(f"{year}-04-01 01:00", f"{year}-07-01 00:00"),
        "Q3": make(f"{year}-07-01 01:00", f"{year}-10-01 00:00"),
        "Q4": make(f"{year}-10-01 01:00", f"{year + 1}-01-01 00:00"),
    }


def _score(actual: pd.Series, prediction: pd.Series, group: str) -> dict[str, Any]:
    record = group_metrics(actual, prediction, CAPACITY_KWH[group], group_name=group).as_dict()
    record["score"] = 0.5 * (record["one_minus_nmae"] + record["ficr"])
    return record


def _comparison(actual: pd.Series, baseline: pd.Series, candidate: pd.Series, group: str) -> dict[str, Any]:
    base = _score(actual, baseline, group)
    cand = _score(actual, candidate, group)
    return {"baseline": base, "candidate": cand, "delta": cand["score"] - base["score"]}


def _mixed_comparison(actual: pd.DataFrame, baseline: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, Any]:
    base = score_details(actual, baseline).as_dict()
    cand = score_details(actual, candidate).as_dict()
    return {
        "baseline": base,
        "candidate": cand,
        "delta_total_score": cand["total_score"] - base["total_score"],
        "delta_one_minus_nmae": cand["one_minus_nmae"] - base["one_minus_nmae"],
        "delta_ficr": cand["ficr"] - base["ficr"],
    }


def _read_jma(
    v1: Mapping[str, Any], v4: Mapping[str, Any], *, period: str
) -> pd.DataFrame:
    if period == "stage1":
        spec = v4["physical_stage1_JMA_source"]["normalized"]
        expected_start = pd.Timestamp("2022-01-01 00:00")
        expected_end = pd.Timestamp("2023-12-31 23:00")
        expected_per_group = 17520
        expected_total = 52560
        full_path = Path(str(v1["external_source"]["normalized_2022_2024"]["path"])).resolve()
        path = _verify_file(spec)
        if path == full_path:
            raise AssertionError("Stage1 attempted to open the full 2022-2024 JMA artifact")
    elif period == "stage2":
        spec = v1["external_source"]["normalized_2022_2024"]
        path = _verify_file(spec)
        if path.stat().st_size != JMA_BYTES or sha256_file(path) != JMA_SHA:
            raise AssertionError("full Stage2 JMA normalized identity differs")
        expected_start = pd.Timestamp("2022-01-01 00:00")
        expected_end = pd.Timestamp("2024-12-31 23:00")
        expected_per_group = 26304
        expected_total = 78912
    else:
        raise ValueError(f"unknown JMA period: {period}")
    frame = pd.read_parquet(path)
    expected_columns = tuple(v1["external_source"]["normalized_2022_2024"]["columns"])
    if tuple(frame.columns) != expected_columns or len(frame) != expected_total or frame.isna().any().any():
        raise AssertionError("JMA schema/rows/missing differs")
    for group in TARGET_COLS:
        rows = frame.loc[frame["group"] == group]
        times = pd.DatetimeIndex(rows["time"])
        expected = pd.date_range(expected_start, expected_end, freq="h")
        if len(rows) != expected_per_group or not times.equals(expected):
            raise AssertionError(f"JMA calendar coverage differs: {group}")
    return frame


def _load_stage1_baseline(v1: Mapping[str, Any]) -> pd.DataFrame:
    g12 = pd.read_parquet(_verify_file(v1["bound_lineage"]["stage1_g12_baseline"])).astype(np.float64)
    g12.index = pd.DatetimeIndex(g12.index, name="forecast_kst_dtm")
    if not g12.index.equals(YEAR_2023) or tuple(g12.columns) != TARGET_COLS[:2]:
        raise AssertionError("stage1 G1/G2 baseline differs")
    raw = pd.read_parquet(_verify_file(v1["bound_lineage"]["stage1_g3_components"])).astype(np.float64)
    raw.index = pd.DatetimeIndex(raw.index, name="forecast_kst_dtm")
    if not raw.index.equals(G3_H2):
        raise AssertionError("stage1 G3 component index differs")
    weighted = (
        0.20 * raw["q07"]
        + 0.075 * raw["shared_l1"]
        + 0.425 * raw["shared_q07"]
        + 0.025 * raw["top200q07"]
        + 0.275 * raw["ewq06"]
    )
    baseline = pd.DataFrame(np.nan, index=YEAR_2023, columns=TARGET_COLS)
    baseline.loc[:, list(TARGET_COLS[:2])] = g12.to_numpy(np.float64)
    baseline.loc[G3_H2, "kpx_group_3"] = np.clip(
        1.25 * weighted.to_numpy(np.float64) - 1200.0,
        0.0,
        1.02 * CAPACITY_KWH["kpx_group_3"],
    )
    return baseline


def _stage1_snapshot(
    args: argparse.Namespace, v1: Mapping[str, Any], v4: Mapping[str, Any]
) -> dict[str, Any]:
    raw_dir = args.raw_dir.resolve()
    return {
        "labels_prefix": shared._snapshot_csv_prefix(
            raw_dir / "train/train_labels.csv", data_rows=17520, prefix_bytes=742551
        ),
        "ldaps_prefix": shared._snapshot_csv_prefix(
            raw_dir / "train/ldaps_train.csv", data_rows=17520 * 16, prefix_bytes=86258374
        ),
        "gfs_prefix": shared._snapshot_csv_prefix(
            raw_dir / "train/gfs_train.csv", data_rows=17520 * 9, prefix_bytes=56101144
        ),
        "info": describe_file(raw_dir / "info.xlsx"),
        "jma_stage1_physical": describe_file(
            _verify_file(v4["physical_stage1_JMA_source"]["normalized"])
        ),
        "jma_full_2022_2024_opened_or_hashed": False,
        "recipe": describe_file(_verify_file(v1["bound_lineage"]["locked_recipe"])),
        "stage1_g12_baseline": describe_file(_verify_file(v1["bound_lineage"]["stage1_g12_baseline"])),
        "stage1_g3_components": describe_file(_verify_file(v1["bound_lineage"]["stage1_g3_components"])),
    }


def _fit_pair(
    *,
    group: str,
    candidate: str,
    control_model: LGBMRegressor,
    train_features: pd.DataFrame,
    apply_features: pd.DataFrame,
    train_labels: pd.Series,
    jma_source: pd.DataFrame,
    eligible: np.ndarray,
) -> tuple[LGBMRegressor, np.ndarray, np.ndarray, dict[str, Any]]:
    train_jma = build_jma_features(
        jma_source,
        group=group,
        index=train_features.index,
        candidate=candidate,
        cross_hub_ws_mean=train_features[CONTROL_DISAGREEMENT_COLUMN],
    )
    apply_jma = build_jma_features(
        jma_source,
        group=group,
        index=apply_features.index,
        candidate=candidate,
        cross_hub_ws_mean=apply_features[CONTROL_DISAGREEMENT_COLUMN],
    )
    extended_train = extended_features(train_features, train_jma)
    extended_apply = extended_features(apply_features, apply_jma)
    model, metadata = fit_direct_model(
        extended_train,
        train_labels,
        capacity_kwh=CAPACITY_KWH[group],
        eligible=eligible,
    )
    control_cf = predict_cf(control_model, apply_features)
    extended_cf = predict_cf(model, extended_apply)
    return model, control_cf, extended_cf, metadata


def _write_manifest(
    args: argparse.Namespace,
    *,
    status: str,
    closure: Mapping[str, Any],
    risk: Mapping[str, Any],
) -> None:
    files = sorted(
        (path for path in args.out_dir.rglob("*") if path.is_file() and path.name != "manifest.json"),
        key=lambda path: path.relative_to(args.out_dir).as_posix(),
    )
    if list(args.out_dir.rglob("*.csv")):
        raise AssertionError("JMA Stage1/Stage2 branch must not create a CSV")
    _write_json(
        args.out_dir / "manifest.json",
        {
            "schema_version": 1,
            "artifact_type": "jma_gsm_paired_increment_strict_v5",
            "created_utc": utc_now(),
            "status": status,
            "config_v1_sha256": V1_SHA,
            "config_v2_sha256": V2_SHA,
            "config_v3_sha256": V3_SHA,
            "config_v4_sha256": V4_SHA,
            "config_v5_sha256": V5_SHA,
            "risk": dict(risk),
            "source_provenance": dict(closure),
            "outputs": [describe_file(path) for path in files],
            "output_count_excluding_manifest": len(files),
            "no_csv": True,
            "JMA_2025_requested_or_parsed": False,
            "runtime": {"packages": package_versions()},
            "git": git_state(PROJECT_DIR),
        },
    )


def run_stage1(
    args: argparse.Namespace,
    v1: Mapping[str, Any],
    v2: Mapping[str, Any],
    v3: Mapping[str, Any],
    v4: Mapping[str, Any],
    v5: Mapping[str, Any],
) -> None:
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True)
    v4_path = PROJECT_DIR / v5["base_v4_preregister"]["path"]
    v3_path = PROJECT_DIR / v4["base_v3_preregister"]["path"]
    v2_path = PROJECT_DIR / v3["base_preregister_v2"]["path"]
    v1_path = PROJECT_DIR / v2["base_preregister"]["path"]
    for source, destination in (
        (v1_path, args.out_dir / "preregister_v1.json"),
        (v1_path.with_suffix(".sha256"), args.out_dir / "preregister_v1.sha256"),
        (v2_path, args.out_dir / "preregister_v2.json"),
        (v2_path.with_suffix(".sha256"), args.out_dir / "preregister_v2.sha256"),
        (v3_path, args.out_dir / "preregister_v3.json"),
        (v3_path.with_suffix(".sha256"), args.out_dir / "preregister_v3.sha256"),
        (v4_path, args.out_dir / "preregister_v4.json"),
        (v4_path.with_suffix(".sha256"), args.out_dir / "preregister_v4.sha256"),
        (args.config, args.out_dir / "preregister_v5.json"),
        (args.config.with_suffix(".sha256"), args.out_dir / "preregister_v5.sha256"),
    ):
        _copy_exclusive(source, destination)
    closure = _closure_record(args)
    snapshot = _stage1_snapshot(args, v1, v4)
    prescore_path = args.out_dir / "stage1_prescore_lock.json"
    _write_json(
        prescore_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "config_v1_sha256": V1_SHA,
            "config_v2_sha256": V2_SHA,
            "config_v3_sha256": V3_SHA,
            "config_v4_sha256": V4_SHA,
            "config_v5_sha256": V5_SHA,
            "source_closure": closure,
            "input_snapshot": snapshot,
            "candidate_model_fits_before_lock": 0,
            "candidate_prediction_value_cells_before_lock": 0,
            "candidate_metric_values_before_lock": 0,
            "2024_label_value_cells_parsed_before_lock": 0,
            "2024_raw_NWP_value_cells_parsed_before_lock": 0,
            "JMA_2025_response_bytes_or_value_cells": 0,
        },
    )

    # Raw NWP is target-free.  Pass an index-only frame so no validation target is
    # needed by the feature builder before candidate predictions are locked.
    features, raw_contract = shared._read_stage1_raw_features(
        args.raw_dir, pd.DataFrame(index=PRE2024)
    )
    fit_labels, fit_label_evidence = _read_stage1_fit_labels(args, v3)
    jma_source = _read_jma(v1, v4, period="stage1")
    baseline = _load_stage1_baseline(v1)
    baseline_path = args.out_dir / "stage1/baseline_corrected_v3.parquet"
    _atomic_parquet(baseline, baseline_path)

    train_index = {
        "kpx_group_1": YEAR_2022,
        "kpx_group_2": YEAR_2022,
        "kpx_group_3": G3_H1,
    }
    apply_index = {
        "kpx_group_1": YEAR_2023,
        "kpx_group_2": YEAR_2023,
        "kpx_group_3": G3_H2,
    }
    predictions = {
        candidate: pd.DataFrame(np.nan, index=YEAR_2023, columns=TARGET_COLS)
        for candidate in CANDIDATES
    }
    training: dict[str, Any] = {}
    group_locks: dict[str, Mapping[str, Any]] = {}
    with HeavyFitGuard(stage="JMA_stage1"):
        for group in TARGET_COLS:
            train = train_index[group]
            apply = apply_index[group]
            model_apply = apply[apply != STAGE1_BOUNDARY]
            if len(apply) - len(model_apply) != 1 or apply[-1] != STAGE1_BOUNDARY:
                raise AssertionError(f"{group} Stage1 boundary isolation differs")
            control_train = features[group].loc[train]
            control_apply = features[group].loc[model_apply]
            actual_train = fit_labels[group]
            if not actual_train.index.equals(train):
                raise AssertionError(f"{group} isolated fit-label index differs")
            eligible = eligible_rows(actual_train, CAPACITY_KWH[group])
            control_model, control_metadata = fit_direct_model(
                control_train,
                actual_train,
                capacity_kwh=CAPACITY_KWH[group],
                eligible=eligible,
            )
            control_path = args.out_dir / f"stage1/models/{group}_control.joblib"
            _atomic_joblib(control_model, control_path)
            training[f"{group}/control"] = {
                **control_metadata,
                "train_start": train.min(),
                "train_end": train.max(),
                "apply_start": apply.min(),
                "apply_end": apply.max(),
                "model": describe_file(control_path),
            }
            for candidate in CANDIDATES:
                print(f"Stage1 fit {group} {candidate}", flush=True)
                model, control_cf, extended_cf, metadata = _fit_pair(
                    group=group,
                    candidate=candidate,
                    control_model=control_model,
                    train_features=control_train,
                    apply_features=control_apply,
                    train_labels=actual_train,
                    jma_source=jma_source,
                    eligible=eligible,
                )
                increment = paired_increment(control_cf, extended_cf)
                selected = transfer_increment(
                    baseline.loc[model_apply, group],
                    increment,
                    capacity_kwh=CAPACITY_KWH[group],
                )
                predictions[candidate].loc[model_apply, group] = selected.to_numpy(np.float64)
                predictions[candidate].loc[STAGE1_BOUNDARY, group] = baseline.loc[STAGE1_BOUNDARY, group]
                candidate_bit = np.ascontiguousarray(
                    predictions[candidate].loc[[STAGE1_BOUNDARY], group].to_numpy(np.float64)
                ).view(np.uint64)[0]
                baseline_bit = np.ascontiguousarray(
                    baseline.loc[[STAGE1_BOUNDARY], group].to_numpy(np.float64)
                ).view(np.uint64)[0]
                if candidate_bit != baseline_bit:
                    raise AssertionError(f"{group}/{candidate} Stage1 boundary bits differ")
                model_path = args.out_dir / f"stage1/models/{group}_{candidate}_extended.joblib"
                detail_path = args.out_dir / f"stage1/diagnostics/{group}_{candidate}.parquet"
                _atomic_joblib(model, model_path)
                detail = pd.DataFrame(
                    {
                        "control_cf": np.r_[control_cf, np.nan],
                        "extended_cf": np.r_[extended_cf, np.nan],
                        "increment_cf": np.r_[increment, 0.0],
                        "baseline_kwh": baseline.loc[apply, group].to_numpy(np.float64),
                        "candidate_kwh": predictions[candidate].loc[apply, group].to_numpy(np.float64),
                        "boundary_identity": apply == STAGE1_BOUNDARY,
                    },
                    index=apply,
                )
                _atomic_parquet(detail, detail_path)
                training[f"{group}/{candidate}"] = {
                    **metadata,
                    "same_eligible_rows_as_control": True,
                    "control_prediction_reused": True,
                    "valid_application_rows": len(model_apply),
                    "boundary_increment_zero_rows": 1,
                    "boundary_candidate_baseline_bit_identity": True,
                    "model": describe_file(model_path),
                    "diagnostics": describe_file(detail_path),
                }

            group_surface = pd.DataFrame(
                {
                    candidate: predictions[candidate].loc[apply, group].to_numpy(np.float64)
                    for candidate in CANDIDATES
                },
                index=apply,
            )
            group_surface_path = args.out_dir / f"stage1/group_candidates/{group}.parquet"
            _atomic_parquet(group_surface, group_surface_path)
            group_lock_path = args.out_dir / f"stage1/group_candidate_locks/{group}.json"
            _write_json(
                group_lock_path,
                {
                    "schema_version": 1,
                    "lock_kind": "stage1_group_candidate_before_score_labels",
                    "created_utc": utc_now(),
                    "config_v1_sha256": V1_SHA,
                    "config_v2_sha256": V2_SHA,
                    "config_v3_sha256": V3_SHA,
                    "config_v4_sha256": V4_SHA,
                    "config_v5_sha256": V5_SHA,
                    "group": group,
                    "fit_label_evidence": fit_label_evidence["g12"] if group in TARGET_COLS[:2] else fit_label_evidence["g3"],
                    "control": training[f"{group}/control"],
                    "candidates": {candidate: training[f"{group}/{candidate}"] for candidate in CANDIDATES},
                    "candidate_surface": describe_file(group_surface_path),
                    "candidate_order": list(CANDIDATES),
                    "score_label_value_cells_before_lock": 0,
                    "candidate_metric_values_before_lock": 0,
                    "source_closure": closure,
                },
            )
            group_locks[group] = describe_file(group_lock_path)

    candidate_files: dict[str, Mapping[str, Any]] = {}
    for candidate in CANDIDATES:
        candidate_path = args.out_dir / f"stage1/candidate_{candidate}.parquet"
        _atomic_parquet(predictions[candidate], candidate_path)
        candidate_files[candidate] = describe_file(candidate_path)

    global_candidate_lock_path = args.out_dir / "stage1_candidate_before_score_labels_lock.json"
    _write_json(
        global_candidate_lock_path,
        {
            "schema_version": 1,
            "lock_kind": "stage1_candidate_before_score_labels",
            "created_utc": utc_now(),
            "config_v1_sha256": V1_SHA,
            "config_v2_sha256": V2_SHA,
            "config_v3_sha256": V3_SHA,
            "config_v4_sha256": V4_SHA,
            "config_v5_sha256": V5_SHA,
            "source_closure": closure,
            "input_snapshot": snapshot,
            "fit_label_evidence": fit_label_evidence,
            "baseline": describe_file(baseline_path),
            "group_candidate_locks": group_locks,
            "candidate_predictions": candidate_files,
            "training": training,
            "score_label_value_cells_before_lock": 0,
            "candidate_metric_values_before_lock": 0,
            "2024_label_value_cells_before_lock": 0,
            "JMA_2025_response_bytes_or_value_cells": 0,
        },
    )
    global_candidate_lock = describe_file(global_candidate_lock_path)
    score_labels, score_label_evidence = _read_stage1_score_labels(
        args, v3, candidate_lock=global_candidate_lock
    )
    eligible_masks, actual_mask_evidence = _verify_stage1_score_masks(score_labels, v5)
    forecast_finite_evidence = _verify_stage1_forecasts_on_eligible_rows(
        score_labels, baseline, predictions, eligible_masks
    )
    score_label_evidence["official_actual_masks"] = actual_mask_evidence
    score_label_evidence["forecast_finite_on_eligible"] = forecast_finite_evidence

    comparisons: dict[str, Any] = {}
    delta_vectors: dict[str, list[float]] = {}
    segments = _segments(2023)
    for candidate in CANDIDATES:
        candidate_records: dict[str, Any] = {}
        deltas: list[float] = []
        for group in TARGET_COLS[:2]:
            candidate_records[group] = {}
            for segment in ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4"):
                index = segments[segment]
                record = _comparison(
                    score_labels.loc[index, group], baseline.loc[index, group], predictions[candidate].loc[index, group], group
                )
                candidate_records[group][segment] = record
                deltas.append(float(record["delta"]))
        candidate_records["kpx_group_3"] = {}
        for segment, index in (("full_H2", G3_H2), ("Q3", segments["Q3"]), ("Q4", segments["Q4"])):
            record = _comparison(
                score_labels.loc[index, "kpx_group_3"],
                baseline.loc[index, "kpx_group_3"],
                predictions[candidate].loc[index, "kpx_group_3"],
                "kpx_group_3",
            )
            candidate_records["kpx_group_3"][segment] = record
            deltas.append(float(record["delta"]))
        if len(deltas) != 17:
            raise AssertionError("Stage1 delta count differs")
        comparisons[candidate] = candidate_records
        delta_vectors[candidate] = deltas
    winner, selection = select_stage1_candidate(delta_vectors)
    result_path = args.out_dir / "stage1_results.json"
    _write_json(
        result_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "config_v1_sha256": V1_SHA,
            "config_v2_sha256": V2_SHA,
            "config_v3_sha256": V3_SHA,
            "config_v4_sha256": V4_SHA,
            "config_v5_sha256": V5_SHA,
            "candidate_before_score_labels_lock": global_candidate_lock,
            "score_label_evidence_after_candidate_lock": score_label_evidence,
            "candidate_order": list(CANDIDATES),
            "comparisons": comparisons,
            "delta_vectors": delta_vectors,
            "selection": selection,
            "winner": winner,
            "stage1_passed": winner is not None,
            "registered_comparisons_per_candidate": 17,
            "raw_feature_contract": raw_contract,
            "training": training,
            "JMA_2025_requested_or_parsed": False,
        },
    )
    _assert_closure_equal(closure, _closure_record(args))
    if _json_ready(snapshot) != _json_ready(_stage1_snapshot(args, v1, v4)):
        raise AssertionError("Stage1 inputs changed")
    lock_path = args.out_dir / "stage1_selection_lock.json"
    _write_json(
        lock_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "config_v1_sha256": V1_SHA,
            "config_v2_sha256": V2_SHA,
            "config_v3_sha256": V3_SHA,
            "config_v4_sha256": V4_SHA,
            "config_v5_sha256": V5_SHA,
            "stage1_prescore_lock": describe_file(prescore_path),
            "stage1_candidate_before_score_labels_lock": global_candidate_lock,
            "stage1_results": describe_file(result_path),
            "source_closure": closure,
            "winner": winner,
            "stage1_passed": winner is not None,
            "stage2_allowed": winner is not None,
            "2024_label_value_cells_parsed_before_this_lock": 0,
            "2024_raw_NWP_value_cells_parsed_before_this_lock": 0,
            "stage1_score_label_value_cells_parsed_after_candidate_lock": int(score_label_evidence["label_value_cells_materialized"]),
            "JMA_2025_response_bytes_or_value_cells": 0,
        },
    )
    if winner is None:
        _write_manifest(
            args,
            status="REJECTED_STAGE1",
            closure=closure,
            risk=v1["risk_and_scope"],
        )
        print(f"Stage1 REJECTED; manifest sha256={sha256_file(args.out_dir / 'manifest.json')}", flush=True)
    else:
        print(f"Stage1 PASS winner={winner}; lock sha256={sha256_file(lock_path)}", flush=True)


def _read_full_features(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    for source in ("ldaps", "gfs"):
        path = args.raw_dir / f"train/{source}_train.csv"
        if sha256_file(path) != FULL_RAW_SHA[source]:
            raise AssertionError(f"full {source} hash differs")
    ldaps = read_weather_csv(args.raw_dir / "train/ldaps_train.csv", "ldaps")
    gfs = read_weather_csv(args.raw_dir / "train/gfs_train.csv", "gfs")
    builder = WeatherFeatureBuilder(group_sites=load_group_sites(args.raw_dir / "info.xlsx"))
    features = builder.fit_transform(ldaps, gfs)
    canonical: tuple[str, ...] | None = None
    for group in TARGET_COLS:
        frame = features[group]
        frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
        if not frame.index.equals(THROUGH2024):
            raise AssertionError(f"full feature index differs: {group}")
        if canonical is not None and tuple(frame.columns) != canonical:
            raise AssertionError("full feature schemas differ")
        canonical = tuple(frame.columns)
        if len(frame.columns) != 612 or not np.isfinite(frame.to_numpy(np.float32)).all():
            raise AssertionError("full feature count/finite differs")
        forbidden = [
            column
            for column in frame.columns
            if any(token in str(column).lower() for token in ("actual", "target", "label", "scada", "power_kw", "public"))
        ]
        if forbidden:
            raise AssertionError(f"forbidden feature columns: {forbidden[:5]}")
        features[group] = frame.astype(np.float32)
    return features


def _read_2024_baseline(spec: Mapping[str, Any]) -> pd.DataFrame:
    frame = pd.read_parquet(_verify_file(spec)).astype(np.float64)
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if not frame.index.equals(YEAR_2024) or tuple(frame.columns) != TARGET_COLS:
        raise AssertionError("Stage2 baseline schema/index differs")
    return frame


def run_stage2(
    args: argparse.Namespace,
    v1: Mapping[str, Any],
    v2: Mapping[str, Any],
    v3: Mapping[str, Any],
    v4: Mapping[str, Any],
    v5: Mapping[str, Any],
) -> None:
    lock_path = args.out_dir / "stage1_selection_lock.json"
    if not lock_path.is_file() or (args.out_dir / "manifest.json").exists():
        raise AssertionError("Stage2 requires a passed nonterminal Stage1 artifact")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if (
        lock["config_v1_sha256"] != V1_SHA
        or lock["config_v2_sha256"] != V2_SHA
        or lock["config_v3_sha256"] != V3_SHA
        or lock["config_v4_sha256"] != V4_SHA
        or lock["config_v5_sha256"] != V5_SHA
    ):
        raise AssertionError("Stage1 config hashes differ")
    if not lock["stage1_passed"] or not lock["stage2_allowed"] or lock["winner"] not in CANDIDATES:
        raise AssertionError("Stage1 did not promote a fixed candidate")
    closure = _closure_record(args)
    _assert_closure_equal(lock["source_closure"], closure)
    winner = str(lock["winner"])
    fit_labels, stage2_fit_label_evidence = _read_stage2_fit_labels(args, v3)
    features = _read_full_features(args)
    jma_source = _read_jma(v1, v4, period="stage2")
    valid_apply = YEAR_2024[YEAR_2024 != BOUNDARY]
    if len(valid_apply) != 8783 or valid_apply.max() != pd.Timestamp("2024-12-31 23:00"):
        raise AssertionError("Stage2 JMA-valid application index differs")
    increments = pd.DataFrame(0.0, index=YEAR_2024, columns=TARGET_COLS, dtype=np.float64)
    training: dict[str, Any] = {}
    with HeavyFitGuard(stage="JMA_stage2"):
        for group in TARGET_COLS:
            control_train = features[group].loc[PRE2024]
            control_apply = features[group].loc[valid_apply]
            actual_train = fit_labels.loc[PRE2024, group]
            eligible = eligible_rows(actual_train, CAPACITY_KWH[group])
            control_model, control_meta = fit_direct_model(
                control_train, actual_train, capacity_kwh=CAPACITY_KWH[group], eligible=eligible
            )
            model, control_cf, extended_cf, extended_meta = _fit_pair(
                group=group,
                candidate=winner,
                control_model=control_model,
                train_features=control_train,
                apply_features=control_apply,
                train_labels=actual_train,
                jma_source=jma_source,
                eligible=eligible,
            )
            increment = paired_increment(control_cf, extended_cf)
            increments.loc[valid_apply, group] = increment
            if increments.loc[BOUNDARY, group] != 0.0:
                raise AssertionError("boundary increment is not exact zero")
            control_path = args.out_dir / f"stage2/models/{group}_control.joblib"
            extended_path = args.out_dir / f"stage2/models/{group}_{winner}_extended.joblib"
            detail_path = args.out_dir / f"stage2/diagnostics/{group}.parquet"
            _atomic_joblib(control_model, control_path)
            _atomic_joblib(model, extended_path)
            detail = pd.DataFrame(
                {
                    "control_cf": np.r_[control_cf, np.nan],
                    "extended_cf": np.r_[extended_cf, np.nan],
                    "increment_cf": increments[group].to_numpy(np.float64),
                    "boundary_identity": YEAR_2024 == BOUNDARY,
                },
                index=YEAR_2024,
            )
            _atomic_parquet(detail, detail_path)
            training[group] = {
                "control": control_meta,
                "extended": extended_meta,
                "same_eligible_rows": True,
                "valid_application_rows": 8783,
                "boundary_increment_zero_rows": 1,
                "control_model": describe_file(control_path),
                "extended_model": describe_file(extended_path),
                "diagnostics": describe_file(detail_path),
            }
    increment_path = args.out_dir / "stage2/paired_increment_cf.parquet"
    _atomic_parquet(increments, increment_path)

    baseline_specs = {
        "primary_locked_v3": v1["bound_lineage"]["stage2_primary_baseline"],
        "interaction_recent_v4": v1["bound_lineage"]["stage2_interaction_baseline"],
    }
    baseline_candidates: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    baseline_candidate_files: dict[str, Any] = {}
    for name, spec in baseline_specs.items():
        baseline = _read_2024_baseline(spec)
        candidate = baseline.copy()
        for group in TARGET_COLS:
            candidate[group] = transfer_increment(
                baseline[group], increments[group].to_numpy(np.float64), capacity_kwh=CAPACITY_KWH[group]
            )
            if np.ascontiguousarray(candidate.loc[[BOUNDARY], group].to_numpy(np.float64)).view(np.uint64)[0] != np.ascontiguousarray(baseline.loc[[BOUNDARY], group].to_numpy(np.float64)).view(np.uint64)[0]:
                raise AssertionError(f"boundary identity bits differ: {name}/{group}")
        baseline_path = args.out_dir / f"stage2/{name}/baseline.parquet"
        candidate_path = args.out_dir / f"stage2/{name}/candidate.parquet"
        _atomic_parquet(baseline, baseline_path)
        _atomic_parquet(candidate, candidate_path)
        baseline_candidates[name] = (baseline, candidate)
        baseline_candidate_files[name] = {
            "baseline": describe_file(baseline_path),
            "candidate": describe_file(candidate_path),
        }

    stage2_candidate_lock_path = args.out_dir / "stage2_candidate_before_2024_score_labels_lock.json"
    _write_json(
        stage2_candidate_lock_path,
        {
            "schema_version": 1,
            "lock_kind": "stage2_candidate_before_2024_score_labels",
            "created_utc": utc_now(),
            "config_v1_sha256": V1_SHA,
            "config_v2_sha256": V2_SHA,
            "config_v3_sha256": V3_SHA,
            "config_v4_sha256": V4_SHA,
            "config_v5_sha256": V5_SHA,
            "stage1_selection_lock": describe_file(lock_path),
            "source_closure": closure,
            "winner": winner,
            "fit_label_evidence": stage2_fit_label_evidence,
            "training": training,
            "paired_increment": describe_file(increment_path),
            "baseline_candidates": baseline_candidate_files,
            "boundary_identity_exact": True,
            "score_label_value_cells_before_lock": 0,
            "candidate_metric_values_before_lock": 0,
            "JMA_2025_response_bytes_or_value_cells": 0,
        },
    )
    stage2_candidate_lock = describe_file(stage2_candidate_lock_path)
    score_labels, stage2_score_label_evidence = _read_stage2_score_labels(
        args, v3, candidate_lock=stage2_candidate_lock
    )

    baseline_results: dict[str, Any] = {}
    segments = _segments(2024)
    dual_pass = True
    for name, (baseline, candidate) in baseline_candidates.items():
        group_records: dict[str, Any] = {}
        group_deltas: list[float] = []
        mixed_records: dict[str, Any] = {}
        mixed_deltas: list[float] = []
        for group in TARGET_COLS:
            group_records[group] = {}
            for segment, index in segments.items():
                record = _comparison(score_labels.loc[index, group], baseline.loc[index, group], candidate.loc[index, group], group)
                group_records[group][segment] = record
                group_deltas.append(float(record["delta"]))
        for segment, index in segments.items():
            record = _mixed_comparison(score_labels.loc[index], baseline.loc[index], candidate.loc[index])
            mixed_records[segment] = record
            mixed_deltas.append(float(record["delta_total_score"]))
        full = mixed_records["full"]
        passed = (
            all(value > 0.0 for value in group_deltas)
            and all(value > 0.0 for value in mixed_deltas)
            and full["delta_one_minus_nmae"] >= 0.0
            and full["delta_ficr"] >= 0.0
        )
        dual_pass = dual_pass and passed
        baseline_results[name] = {
            "group_comparisons": group_records,
            "mixed_comparisons": mixed_records,
            "minimum_group_delta": min(group_deltas),
            "minimum_mixed_delta": min(mixed_deltas),
            "full_delta_one_minus_nmae": full["delta_one_minus_nmae"],
            "full_delta_ficr": full["delta_ficr"],
            "passed": passed,
            "outputs": list(baseline_candidate_files[name].values()),
        }
    result_path = args.out_dir / "stage2_results.json"
    _write_json(
        result_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "winner": winner,
            "config_v3_sha256": V3_SHA,
            "config_v4_sha256": V4_SHA,
            "config_v5_sha256": V5_SHA,
            "candidate_before_2024_score_labels_lock": stage2_candidate_lock,
            "score_label_evidence_after_candidate_lock": stage2_score_label_evidence,
            "boundary_timestamp": BOUNDARY,
            "boundary_increment_exact_zero_all_groups": True,
            "baseline_results": baseline_results,
            "dual_baseline_passed": dual_pass,
            "training": training,
            "increment": describe_file(increment_path),
            "JMA_2025_requested_or_parsed": False,
        },
    )
    _assert_closure_equal(closure, _closure_record(args))
    promotion_path = args.out_dir / "stage2_promotion_lock.json"
    _write_json(
        promotion_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "config_v1_sha256": V1_SHA,
            "config_v2_sha256": V2_SHA,
            "config_v3_sha256": V3_SHA,
            "config_v4_sha256": V4_SHA,
            "config_v5_sha256": V5_SHA,
            "stage1_selection_lock": describe_file(lock_path),
            "stage2_candidate_before_2024_score_labels_lock": stage2_candidate_lock,
            "stage2_results": describe_file(result_path),
            "winner": winner,
            "dual_baseline_passed": dual_pass,
            "final_prescore_lock_required_before_any_2025_JMA_request": True,
            "JMA_2025_response_bytes_or_value_cells": 0,
            "CSV_allowed_in_this_process": False,
        },
    )
    _write_manifest(
        args,
        status="PROMOTED_STAGE2_AWAITING_SEPARATE_FINAL_PRESCORE" if dual_pass else "REJECTED_STAGE2",
        closure=closure,
        risk=v1["risk_and_scope"],
    )
    print(
        f"Stage2 {'PASS' if dual_pass else 'REJECTED'}; manifest sha256={sha256_file(args.out_dir / 'manifest.json')}",
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.config = args.config.resolve()
    args.raw_dir = args.raw_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    v1, v2, v3, v4, v5 = verify_config(args.config)
    if args.stage == "stage1":
        run_stage1(args, v1, v2, v3, v4, v5)
    else:
        run_stage2(args, v1, v2, v3, v4, v5)


if __name__ == "__main__":
    main()

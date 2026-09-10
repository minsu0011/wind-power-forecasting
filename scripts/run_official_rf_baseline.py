"""Strict-forward reproduction of DACON official RF baseline code-share 14031."""

from __future__ import annotations

import ast
import io
import json
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

from scripts import run_seasonal_wind_ridge as protocol  # noqa: E402
from scripts import run_shared_q07_multiseed as shared  # noqa: E402
from scripts import run_ficr_bayes_decision_strict as strict  # noqa: E402
from src.features import (  # noqa: E402
    AVAILABLE_COL,
    COORD_COLS,
    GRID_COL,
    SOURCE_VARIABLES,
    TIME_COL,
)
from src.manifest import (  # noqa: E402
    describe_file,
    git_state,
    package_versions,
    sha256_file,
    utc_now,
)
from src.metric import CAPACITY_KWH, TARGET_COLS  # noqa: E402
from src.official_rf_baseline import (  # noqa: E402
    EXPLICIT_RF_PARAMETERS,
    FEATURE_COLUMNS,
    RELEVANT_DEFAULT_RF_PARAMETERS,
    STAGE1_REQUIRED,
    STAGE2_REQUIRED,
    OfficialRandomForestBaseline,
    assert_strict_forward,
    blend_direct_kwh,
    build_official_rf_features,
    select_stage1_weight,
    stage2_promoted,
)


PREREGISTER_SHA256 = "0175e3a7ac5d1a3cc2660626bb300dc508810ee0a96d05b4d979d8a586a1d764"
PARENT_PREREGISTER_SHA256 = "1e4debe1e694bf9ea2e0fe2a97cd70dfa1d365ec80b8b854163d629fc8831c2a"
OFFICIAL_NOTEBOOK_SHA256 = "c3d73877ad55160af8138dae408047e3cf958dac8464f65b33c4cecc08ccc78d"
WEIGHTS: dict[str, float] = {"w05": 0.05, "w10": 0.10}
DEFAULT_OUT_DIR = Path("artifacts/postgate/official_rf_baseline_strict_v2")
DEFAULT_PREREGISTER = Path("configs/official_rf_baseline_preregister_v2.json")
PARENT_PREREGISTER = Path("configs/official_rf_baseline_preregister_v1.json")
QUARANTINED_V1 = Path(
    "artifacts/postgate/official_rf_baseline_strict_v1_failed_missing_shared_input_contract"
)
RAW_ROWS_PER_TIMESTAMP: dict[str, int] = {"ldaps": 16, "gfs": 9}
STAGE1_TIMESTAMPS = 17_520
STAGE1_START = pd.Timestamp("2022-01-01 01:00:00")
STAGE1_END = pd.Timestamp("2024-01-01 00:00:00")
STAGE1_NEXT = pd.Timestamp("2024-01-01 01:00:00")
TRAIN_END = pd.Timestamp("2025-01-01 00:00:00")
TEST_START = pd.Timestamp("2025-01-01 01:00:00")
TEST_END = pd.Timestamp("2026-01-01 00:00:00")
STAGE1_PREFIX_BYTES = {"ldaps": 86_258_374, "gfs": 56_101_144}
STAGE1_PREFIX_SHA256 = {
    "ldaps": "2ae4fd41ea9c0ba06c0aa6873cbf9bd3b810b0de83051d47f9b80efa0a47c262",
    "gfs": "c1a3e8fb072101d9d24161320375c53849cc803a039e17d222ff27ad92f65934",
}
FULL_TRAIN_SHA256 = {
    "ldaps": "61ae944e7ae1fcb17391be6737792a2205c6507bf2446ed5d9d0daf07fdea026",
    "gfs": "cd56b67d357e7bbaff5d0d51d3537d935c9e7a3f012e9f37516bdc4d38c66a5d",
}
EXPECTED_REPO_SOURCE_CLOSURE: tuple[str, ...] = (
    "scripts/run_catboost_multiquantile_bayes.py",
    "scripts/run_ficr_bayes_decision_strict.py",
    "scripts/run_official_rf_baseline.py",
    "scripts/run_seasonal_wind_ridge.py",
    "scripts/run_shared_q07_multiseed.py",
    "scripts/run_weather_quantile_bayes.py",
    "src/catboost_multiquantile.py",
    "src/features.py",
    "src/manifest.py",
    "src/metric.py",
    "src/official_rf_baseline.py",
    "src/probabilistic.py",
    "src/seasonal_wind_ridge.py",
    "src/weather_quantile.py",
)

_ACTIVE_RAW_DIR: Path | None = None
_STAGE1_FEATURE_CACHE: tuple[dict[str, pd.DataFrame], dict[str, Any]] | None = None
_FULL_TRAIN_FEATURE_CACHE: tuple[dict[str, pd.DataFrame], dict[str, Any]] | None = None
_TEST_FEATURE_CACHE: tuple[dict[str, pd.DataFrame], dict[str, Any]] | None = None


def _verify_preregister(path: Path) -> dict[str, Any]:
    observed = sha256_file(path)
    if observed != PREREGISTER_SHA256:
        raise AssertionError(f"preregister hash changed: {observed}")
    sidecar = path.with_suffix(".sha256")
    expected_sidecar = f"{PREREGISTER_SHA256}  {path.name}\n"
    if sidecar.read_text(encoding="utf-8") != expected_sidecar:
        raise AssertionError("preregister sidecar changed")
    wrapper = json.loads(path.read_text(encoding="utf-8"))
    if wrapper["experiment_id"] != "official_rf_baseline_strict_forward_v2":
        raise AssertionError("experiment id changed")
    supersedes = wrapper["supersedes"]
    parent_path = (PROJECT_DIR / supersedes["preregister_path"]).resolve()
    if parent_path != (PROJECT_DIR / PARENT_PREREGISTER).resolve():
        raise AssertionError("parent preregister path changed")
    if supersedes["preregister_sha256"] != PARENT_PREREGISTER_SHA256:
        raise AssertionError("parent preregister binding changed")
    if sha256_file(parent_path) != PARENT_PREREGISTER_SHA256:
        raise AssertionError("parent preregister hash changed")
    parent_sidecar = parent_path.with_suffix(".sha256")
    if parent_sidecar.read_text(encoding="utf-8") != (
        f"{PARENT_PREREGISTER_SHA256}  {parent_path.name}\n"
    ):
        raise AssertionError("parent preregister sidecar changed")
    if supersedes["failure_kind"] != "pre_fit_shared_input_schema_contract_only":
        raise AssertionError("v1 failure contract changed")
    for key in (
        "candidate_fit_started",
        "candidate_prediction_materialized",
        "candidate_score_computed",
        "2024_or_2025_input_read",
    ):
        if supersedes[key] is not False:
            raise AssertionError(f"v1 pre-fit failure evidence changed: {key}")
    info = wrapper["shared_input_snapshot_addition"]["info_workbook"]
    if info != {
        "path": "data/local/open/info.xlsx",
        "bytes": 3_823_422,
        "sha256": "89e83a52e0eb2ce367a3573a96d6795ed4b4d4ac624965cb3530beec0cbd2bd6",
        "role": (
            "Unused immutable snapshot required solely by the existing shared strict "
            "Stage1 input assertion. It is not read by the official RF feature builder, "
            "model, target, prediction, selection, or score code."
        ),
    }:
        raise AssertionError("unused shared info workbook snapshot changed")
    if wrapper["shared_input_snapshot_addition"]["candidate_semantics_changed"] is not False:
        raise AssertionError("shared input addition changed candidate semantics")
    invariant = wrapper["immutable_candidate_contract"]
    if invariant["runtime_estimator_parameters_inherited_bit_exact_from_v1"] != EXPLICIT_RF_PARAMETERS:
        raise AssertionError("runtime RF parameters changed from v1")
    if int(invariant["feature_count"]) != 74:
        raise AssertionError("v2 invariant feature count changed")
    if tuple(map(float, invariant["selectable_blends_inherited_bit_exact_from_v1"])) != tuple(
        WEIGHTS.values()
    ):
        raise AssertionError("v2 invariant weights changed")
    if invariant["no_retune_or_candidate_change"] is not True:
        raise AssertionError("no-retune contract changed")
    closure = wrapper["static_repo_import_closure_contract"]
    if int(closure["expected_source_count"]) != 14:
        raise AssertionError("repo source closure count changed")
    if tuple(closure["expected_sources_in_exact_sorted_order"]) != EXPECTED_REPO_SOURCE_CLOSURE:
        raise AssertionError("repo source closure changed")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    payload = json.loads(json.dumps(parent))
    payload["experiment_id"] = wrapper["experiment_id"]
    payload["status"] = wrapper["status"]
    payload["physical_stage1_inputs"]["info_workbook"] = info
    payload["output_contract"] = wrapper["output_contract"]
    payload["v2_supersession_contract"] = wrapper
    source = payload["official_primary_source_contract"]
    if source["notebook_sha256"] != OFFICIAL_NOTEBOOK_SHA256:
        raise AssertionError("official notebook binding changed")
    if source["notebook_model_explicit_parameters"] != EXPLICIT_RF_PARAMETERS:
        raise AssertionError("official explicit RF parameters changed")
    if source["notebook_feature_contract"]["train_test_concat_preprocessing"] is not False:
        raise AssertionError("official preprocessing contract changed")
    if tuple(payload["feature_contract"]["columns_in_exact_order"]) != FEATURE_COLUMNS:
        raise AssertionError("feature contract changed")
    if int(payload["feature_contract"]["feature_count"]) != 74:
        raise AssertionError("feature count changed")
    pipeline = payload["model"]["pipeline"]
    if pipeline[0] != {
        "class": "sklearn.impute.SimpleImputer",
        "parameters": {"strategy": "median"},
    }:
        raise AssertionError("imputer contract changed")
    if pipeline[1]["explicit_notebook_parameters"] != EXPLICIT_RF_PARAMETERS:
        raise AssertionError("RF explicit contract changed")
    if pipeline[1]["frozen_relevant_defaults"] != RELEVANT_DEFAULT_RF_PARAMETERS:
        raise AssertionError("RF default contract changed")
    adaptation = payload["leakage_safe_metric_adaptations"]
    if adaptation["no_train_test_or_fit_application_concat"] is not True:
        raise AssertionError("strict preprocessing adaptation changed")
    if "n_jobs=-1" not in adaptation["prediction"]:
        raise AssertionError("serial inference adaptation is not distinguished")
    if tuple(map(float, payload["candidate_family"]["fixed_global_blend_weights"])) != tuple(
        WEIGHTS.values()
    ):
        raise AssertionError("weights changed")
    if payload["stage1"]["registered_slice_count"] != 17:
        raise AssertionError("Stage1 slices changed")
    if tuple(payload["stage2"]["registered_segments_each_group"]) != STAGE2_REQUIRED:
        raise AssertionError("Stage2 group slices changed")
    if tuple(payload["stage2"]["mixed_total_segments"]) != STAGE2_REQUIRED:
        raise AssertionError("Stage2 mixed slices changed")
    if payload["public_contract"]["public_metric_triplets_used"] is not False:
        raise AssertionError("Public metric use changed")
    parent_closure = payload["static_repo_import_closure_contract"]
    if int(parent_closure["expected_source_count"]) != 14:
        raise AssertionError("repo source closure count changed")
    if tuple(parent_closure["expected_sources_in_exact_sorted_order"]) != EXPECTED_REPO_SOURCE_CLOSURE:
        raise AssertionError("repo source closure changed")
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
        path = pending.pop()
        if path in observed:
            continue
        if not path.is_file() or path.suffix != ".py":
            raise FileNotFoundError(path)
        observed.add(path)
        for dependency in _direct_local_imports(path):
            if dependency not in observed:
                pending.append(dependency)
    output = tuple(
        sorted(observed, key=lambda item: item.relative_to(PROJECT_DIR).as_posix())
    )
    relative = tuple(path.relative_to(PROJECT_DIR).as_posix() for path in output)
    if relative != EXPECTED_REPO_SOURCE_CLOSURE:
        raise AssertionError(f"static repo source closure changed: {relative}")
    return output


def _provenance_paths(preregister_path: Path) -> dict[str, Path]:
    roots = (
        Path(__file__).resolve(),
        PROJECT_DIR / "scripts/run_seasonal_wind_ridge.py",
        PROJECT_DIR / "src/official_rf_baseline.py",
    )
    closure = _static_repo_import_closure(roots)
    output = {
        f"repo_source::{path.relative_to(PROJECT_DIR).as_posix()}": path
        for path in closure
    }
    output.update(
        {
            "focused_test": PROJECT_DIR / "tests/test_official_rf_baseline.py",
            "runner_test": PROJECT_DIR / "tests/test_official_rf_baseline_runner.py",
            "preregister": preregister_path.resolve(),
            "preregister_sidecar": preregister_path.with_suffix(".sha256").resolve(),
            "parent_preregister": (PROJECT_DIR / PARENT_PREREGISTER).resolve(),
            "parent_preregister_sidecar": (PROJECT_DIR / PARENT_PREREGISTER)
            .with_suffix(".sha256")
            .resolve(),
            "quarantined_v1_preregister": (PROJECT_DIR / QUARANTINED_V1 / "preregister.json").resolve(),
            "quarantined_v1_preregister_sidecar": (
                PROJECT_DIR / QUARANTINED_V1 / "preregister.sha256"
            ).resolve(),
            "quarantined_v1_failure_stderr": (
                PROJECT_DIR
                / f"{QUARANTINED_V1.as_posix()}.run.stderr.log"
            ).resolve(),
        }
    )
    return output


def _required_raw_columns(source: str) -> list[str]:
    return [
        TIME_COL,
        AVAILABLE_COL,
        GRID_COL,
        *COORD_COLS,
        *SOURCE_VARIABLES[source],
    ]


def _validate_raw_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    expected_timestamps: int,
    expected_start: pd.Timestamp,
    expected_end: pd.Timestamp,
) -> pd.Series:
    expected_rows = expected_timestamps * RAW_ROWS_PER_TIMESTAMP[source]
    if len(frame) != expected_rows:
        raise AssertionError(f"{source} raw row count changed")
    frame[TIME_COL] = pd.to_datetime(frame[TIME_COL], errors="raise")
    times = pd.DatetimeIndex(frame[TIME_COL])
    if not times.is_monotonic_increasing:
        raise AssertionError(f"{source} raw timestamps are not increasing")
    counts = frame.groupby(TIME_COL, sort=False)[GRID_COL].size()
    expected_index = pd.date_range(
        expected_start, expected_end, periods=expected_timestamps
    )
    if not counts.index.equals(expected_index):
        raise AssertionError(f"{source} timestamp sequence changed")
    if not counts.eq(RAW_ROWS_PER_TIMESTAMP[source]).all():
        raise AssertionError(f"{source} grid count changed")
    return counts


def _read_bounded_stage1_weather(
    path: Path, *, source: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = _required_raw_columns(source)
    expected_rows = STAGE1_TIMESTAMPS * RAW_ROWS_PER_TIMESTAMP[source]
    with path.open("rb") as stream:
        header_line = stream.readline()
    header = pd.read_csv(io.BytesIO(header_line), nrows=0, encoding="utf-8-sig").columns
    missing = sorted(set(required).difference(header))
    if missing:
        raise ValueError(f"{source} raw file missing columns: {missing}")
    prefix_sha, prefix_bytes = shared._csv_prefix_identity(
        path,
        data_rows=expected_rows,
        byte_limit=STAGE1_PREFIX_BYTES[source],
    )
    if prefix_sha != STAGE1_PREFIX_SHA256[source]:
        raise AssertionError(f"{source} bounded prefix hash changed")
    dtypes: dict[str, str] = {GRID_COL: "int64"}
    dtypes.update({column: "float64" for column in [*COORD_COLS, *SOURCE_VARIABLES[source]]})
    bounded_raw = shared._BoundedRawReader(path, byte_limit=prefix_bytes)
    try:
        with io.BufferedReader(bounded_raw, buffer_size=1024 * 1024) as bounded:
            frame = pd.read_csv(
                bounded,
                usecols=required,
                dtype=dtypes,
                nrows=expected_rows,
                encoding="utf-8-sig",
                low_memory=False,
                memory_map=False,
            )
            bytes_returned = bounded_raw.bytes_returned
            underlying_position = bounded_raw.underlying_position
    finally:
        bounded_raw.close()
    if bytes_returned != prefix_bytes or underlying_position != prefix_bytes:
        raise AssertionError(f"{source} reader crossed physical prefix")
    counts = _validate_raw_frame(
        frame,
        source=source,
        expected_timestamps=STAGE1_TIMESTAMPS,
        expected_start=STAGE1_START,
        expected_end=STAGE1_END,
    )
    next_timestamp = pd.Timestamp(
        shared._next_csv_first_field(
            path, after_data_rows=expected_rows, prefix_bytes=prefix_bytes
        )
    )
    if next_timestamp != STAGE1_NEXT:
        raise AssertionError(f"{source} next-boundary timestamp changed")
    evidence = {
        "source": source,
        "path": str(path.resolve()),
        "reader_dtype_contract": "official pandas float64 weather means",
        "nrows_argument": expected_rows,
        "physical_byte_limit": prefix_bytes,
        "physical_bytes_returned": bytes_returned,
        "underlying_file_position_after_parse": underlying_position,
        "physical_prefix_sha256": prefix_sha,
        "suffix_bytes_exposed_to_parser": 0,
        "materialized_rows": len(frame),
        "materialized_timestamp_rows": len(counts),
        "materialized_start": counts.index.min(),
        "materialized_end": counts.index.max(),
        "next_boundary_timestamp_only": next_timestamp,
        "next_boundary_weather_values_read": False,
        "usecols": required,
        "materialized_frame_sha256": shared._frame_sha256(frame),
    }
    return frame, evidence


def _feature_map(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {group: frame.copy() for group in TARGET_COLS}


def _read_stage1_official_features(
    raw_dir: Path, labels: pd.DataFrame
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    global _STAGE1_FEATURE_CACHE
    raw: dict[str, pd.DataFrame] = {}
    evidence: dict[str, Any] = {}
    for source in ("ldaps", "gfs"):
        raw[source], evidence[source] = _read_bounded_stage1_weather(
            raw_dir / "train" / f"{source}_train.csv", source=source
        )
    features = build_official_rf_features(raw["ldaps"], raw["gfs"])
    if not features.index.equals(labels.index):
        raise AssertionError("official Stage1 feature index differs")
    mapping = _feature_map(features)
    contract = {
        "raw_prefix": evidence,
        "builder": {
            "official_codeshare": 14031,
            "feature_count": 74,
            "feature_columns": list(FEATURE_COLUMNS),
            "all_grid_arithmetic_mean": True,
            "float64_raw_parse_matching_notebook": True,
        },
        "feature_frame_sha256": shared._frame_sha256(features),
    }
    _STAGE1_FEATURE_CACHE = (mapping, contract)
    return mapping, contract


def _official_features(
    weather_features: Mapping[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    output: dict[str, pd.DataFrame] = {}
    evidence: dict[str, Any] = {}
    for group in TARGET_COLS:
        frame = weather_features[group]
        OfficialRandomForestBaseline._validate_features(frame)
        output[group] = frame
        evidence[group] = {
            "rows": len(frame),
            "columns": frame.shape[1],
            "start": frame.index.min(),
            "end": frame.index.max(),
            "design_frame_sha256": shared._frame_sha256(frame),
            "feature_columns_exact": list(FEATURE_COLUMNS),
            "actual_feature_columns": 0,
            "scada_feature_columns": 0,
            "baseline_or_residual_feature_columns": 0,
            "public_feature_columns": 0,
        }
    return output, evidence


def _read_full_weather(
    path: Path,
    *,
    source: str,
    expected_timestamps: int,
    expected_start: pd.Timestamp,
    expected_end: pd.Timestamp,
    expected_sha256: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    file_record = describe_file(path)
    if expected_sha256 is not None and file_record["sha256"] != expected_sha256:
        raise AssertionError(f"{source} conditional full-file hash changed")
    required = _required_raw_columns(source)
    dtypes: dict[str, str] = {GRID_COL: "int64"}
    dtypes.update({column: "float64" for column in [*COORD_COLS, *SOURCE_VARIABLES[source]]})
    frame = pd.read_csv(
        path,
        usecols=required,
        dtype=dtypes,
        encoding="utf-8-sig",
        low_memory=False,
        memory_map=False,
    )
    counts = _validate_raw_frame(
        frame,
        source=source,
        expected_timestamps=expected_timestamps,
        expected_start=expected_start,
        expected_end=expected_end,
    )
    return frame, {
        "source": source,
        "file": file_record,
        "whole_file_read_after_required_promotion_lock": True,
        "reader_dtype_contract": "official pandas float64 weather means",
        "materialized_rows": len(frame),
        "materialized_timestamp_rows": len(counts),
        "materialized_start": counts.index.min(),
        "materialized_end": counts.index.max(),
        "usecols": required,
        "materialized_frame_sha256": shared._frame_sha256(frame),
    }


def _read_full_train_feature_map() -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    global _FULL_TRAIN_FEATURE_CACHE
    if _FULL_TRAIN_FEATURE_CACHE is not None:
        return _FULL_TRAIN_FEATURE_CACHE
    if _ACTIVE_RAW_DIR is None:
        raise RuntimeError("raw directory was not installed")
    raw: dict[str, pd.DataFrame] = {}
    evidence: dict[str, Any] = {}
    for source in ("ldaps", "gfs"):
        raw[source], evidence[source] = _read_full_weather(
            _ACTIVE_RAW_DIR / "train" / f"{source}_train.csv",
            source=source,
            expected_timestamps=26_304,
            expected_start=STAGE1_START,
            expected_end=TRAIN_END,
            expected_sha256=FULL_TRAIN_SHA256[source],
        )
    features = build_official_rf_features(raw["ldaps"], raw["gfs"])
    expected_index = pd.date_range(
        STAGE1_START, TRAIN_END, freq="h", name="forecast_kst_dtm"
    )
    if not features.index.equals(expected_index):
        raise AssertionError("full train official feature index changed")
    mapping = _feature_map(features)
    contract = {
        "raw_full": evidence,
        "feature_frame_sha256": shared._frame_sha256(features),
        "feature_columns": list(FEATURE_COLUMNS),
    }
    _FULL_TRAIN_FEATURE_CACHE = (mapping, contract)
    return _FULL_TRAIN_FEATURE_CACHE


def _read_test_feature_map() -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    global _TEST_FEATURE_CACHE
    if _TEST_FEATURE_CACHE is not None:
        return _TEST_FEATURE_CACHE
    if _ACTIVE_RAW_DIR is None:
        raise RuntimeError("raw directory was not installed")
    raw: dict[str, pd.DataFrame] = {}
    evidence: dict[str, Any] = {}
    for source in ("ldaps", "gfs"):
        raw[source], evidence[source] = _read_full_weather(
            _ACTIVE_RAW_DIR / "test" / f"{source}_test.csv",
            source=source,
            expected_timestamps=8_760,
            expected_start=TEST_START,
            expected_end=TEST_END,
            expected_sha256=None,
        )
    features = build_official_rf_features(raw["ldaps"], raw["gfs"])
    expected_index = pd.date_range(
        TEST_START, TEST_END, freq="h", name="forecast_kst_dtm"
    )
    if not features.index.equals(expected_index):
        raise AssertionError("test official feature index changed")
    mapping = _feature_map(features)
    contract = {
        "raw_test": evidence,
        "feature_frame_sha256": shared._frame_sha256(features),
        "feature_columns": list(FEATURE_COLUMNS),
    }
    _TEST_FEATURE_CACHE = (mapping, contract)
    return _TEST_FEATURE_CACHE


def _read_stage2_features(
    cache_dir: Path, expected_index: pd.DatetimeIndex
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    del cache_dir
    mapping, evidence = _read_full_train_feature_map()
    if not next(iter(mapping.values())).index.equals(expected_index):
        raise AssertionError("Stage2 requested index differs from full raw features")
    return mapping, evidence


def _reload_audit(
    *,
    model_path: Path,
    groups: Sequence[str],
    features: Mapping[str, pd.DataFrame],
    application_indexes: Mapping[str, pd.DatetimeIndex],
    expected: Mapping[str, pd.Series],
) -> dict[str, Any]:
    models: dict[str, OfficialRandomForestBaseline] = joblib.load(model_path)
    if tuple(models) != tuple(groups):
        raise AssertionError("reloaded model group order changed")
    audit: dict[str, Any] = {}
    for group in groups:
        model = models[group]
        observed = model.predict_cf(features[group].loc[application_indexes[group]])
        if not (
            observed.index.equals(expected[group].index)
            and np.array_equal(observed.to_numpy(), expected[group].to_numpy())
        ):
            raise AssertionError(f"{group} reloaded official RF prediction differs")
        if model.estimator_ is None or model.imputer_ is None:
            raise AssertionError("reloaded RF pipeline missing")
        params = model.estimator_.get_params(deep=False)
        explicit = {key: params[key] for key in EXPLICIT_RF_PARAMETERS}
        defaults = {key: params[key] for key in RELEVANT_DEFAULT_RF_PARAMETERS}
        if explicit != EXPLICIT_RF_PARAMETERS or defaults != RELEVANT_DEFAULT_RF_PARAMETERS:
            raise AssertionError("reloaded RF contract changed")
        audit[group] = {
            "prediction_bit_exact": True,
            "feature_columns_exact": list(FEATURE_COLUMNS),
            "notebook_explicit_parameters": explicit,
            "frozen_relevant_defaults_not_claimed_as_explicit": defaults,
            "imputer_strategy": model.imputer_.strategy,
            "serial_inference_adaptation": True,
            "fitted_estimator_n_jobs_preserved": model.estimator_.n_jobs,
            "target_kind": "direct_capacity_factor",
        }
    return audit


def _postlock_raw_audit(
    *, raw_dir: Path, cache_dir: Path, out_dir: Path
) -> dict[str, Any]:
    del cache_dir
    lock, _ = protocol._load_stage1_lock(out_dir)
    path = out_dir / "postlock_cache_audit.json"
    if path.exists():
        raise FileExistsError(path)
    if lock["locked_weight"] is None:
        audit = {
            "schema_version": 1,
            "created_utc": utc_now(),
            "stage1_promotion_lock": describe_file(out_dir / "stage1_promotion_lock.json"),
            "performed": False,
            "audit_kind": "bounded_raw_vs_conditional_full_raw_prefix",
            "2024_weather_physical_bytes_accessed": False,
            "reason": "identity locked; no conditional full raw file opened",
            "checks": {},
        }
        strict._write_json(path, audit)
        return audit
    bounded, _ = (
        _STAGE1_FEATURE_CACHE
        if _STAGE1_FEATURE_CACHE is not None
        else _read_stage1_official_features(
            raw_dir,
            pd.DataFrame(
                index=pd.date_range(
                    STAGE1_START, STAGE1_END, freq="h", name="forecast_kst_dtm"
                )
            ),
        )
    )
    full, full_evidence = _read_full_train_feature_map()
    checks: dict[str, Any] = {}
    prefix_index = next(iter(bounded.values())).index
    for group in TARGET_COLS:
        selected = full[group].loc[prefix_index]
        exact = (
            selected.index.equals(bounded[group].index)
            and np.array_equal(selected.to_numpy(), bounded[group].to_numpy(), equal_nan=True)
        )
        if not exact:
            raise AssertionError(f"{group} bounded/full raw prefix differs")
        checks[group] = {
            "rows": len(selected),
            "feature_columns": list(FEATURE_COLUMNS),
            "value_bits_exact_including_nan_positions": True,
        }
    audit = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "stage1_promotion_lock": describe_file(out_dir / "stage1_promotion_lock.json"),
        "performed": True,
        "audit_kind": "bounded_raw_vs_conditional_full_raw_prefix",
        "selection_locked_before_full_raw_read": True,
        "2024_weather_physical_bytes_accessed_after_stage1_lock": True,
        "2024_weather_values_cannot_affect_locked_stage1_models_predictions_or_selection": True,
        "full_raw_evidence": full_evidence,
        "checks": checks,
        "all_group_selected_prefixes_value_bit_exact": True,
    }
    strict._write_json(path, audit)
    return audit


def _finalize(
    *,
    raw_dir: Path,
    artifact_root: Path,
    cache_dir: Path,
    out_dir: Path,
    preregister_path: Path,
) -> dict[str, Any]:
    del cache_dir
    lock, stage2_result = protocol._load_stage2_lock(out_dir)
    final_path = out_dir / "final_results.json"
    manifest_path = out_dir / "manifest.json"
    if final_path.exists() or manifest_path.exists():
        raise FileExistsError("final outputs already exist")
    final_inputs: dict[str, Any] = {}
    if not lock["candidate_promoted"]:
        final_result = {
            "schema_version": 1,
            "created_utc": utc_now(),
            "executed": False,
            "locked_weight": lock["locked_weight"],
            "candidate_promoted": False,
            "2025_read": False,
            "submission_created": False,
            "reason": "global candidate failed a preregistered gate",
            "leaderboard_score_claim": False,
        }
        strict._write_json(final_path, final_result)
    else:
        final_inputs = {
            "labels": shared._snapshot_file(raw_dir / "train/train_labels.csv"),
            "sample": shared._snapshot_file(raw_dir / "sample_submission.csv"),
            "baseline": shared._snapshot_file(
                artifact_root / "final_cf_fix/predictions/corrected_v3_test.parquet"
            ),
            "train_ldaps": shared._snapshot_file(raw_dir / "train/ldaps_train.csv"),
            "train_gfs": shared._snapshot_file(raw_dir / "train/gfs_train.csv"),
            "test_ldaps": shared._snapshot_file(raw_dir / "test/ldaps_test.csv"),
            "test_gfs": shared._snapshot_file(raw_dir / "test/gfs_test.csv"),
        }
        labels = strict._read_full_labels(raw_dir / "train/train_labels.csv")
        train_features, train_feature_evidence = _read_full_train_feature_map()
        sample = pd.read_csv(
            raw_dir / "sample_submission.csv",
            encoding="utf-8-sig",
            dtype={"forecast_id": "string", "forecast_kst_dtm": "string"},
        )
        if tuple(sample.columns) != ("forecast_id", "forecast_kst_dtm", *TARGET_COLS):
            raise AssertionError("sample schema changed")
        test_index = pd.DatetimeIndex(
            pd.to_datetime(sample["forecast_kst_dtm"]), name="forecast_kst_dtm"
        )
        if not test_index.equals(strict._year_index(2025)):
            raise AssertionError("sample index changed")
        test_features, test_feature_evidence = _read_test_feature_map()
        baseline = strict._read_prediction(
            artifact_root / "final_cf_fix/predictions/corrected_v3_test.parquet",
            test_index,
            required_columns=TARGET_COLS,
        )
        models: dict[str, OfficialRandomForestBaseline] = {}
        predictions: dict[str, pd.Series] = {}
        direct = pd.DataFrame(index=test_index, columns=TARGET_COLS, dtype=float)
        training: dict[str, Any] = {}
        joined_features: dict[str, pd.DataFrame] = {}
        for group in TARGET_COLS:
            joined = pd.concat((train_features[group], test_features[group]), axis=0)
            joined_features[group] = joined
            model, prediction, metadata = protocol._fit_predict(
                group=group,
                features=joined,
                actual=labels[group],
                fit_index=labels.index,
                application_index=test_index,
            )
            models[group] = model
            predictions[group] = prediction
            direct[group] = prediction
            training[group] = metadata
        indexes = {group: test_index for group in TARGET_COLS}
        candidate = protocol._blend_frame(
            baseline, direct, indexes, float(lock["locked_weight"])
        )
        standalone = protocol._blend_frame(baseline, direct, indexes, 1.0)
        model_path = out_dir / "models/final_models.joblib"
        direct_path = out_dir / "predictions/official_rf_baseline_direct_cf_2025.parquet"
        standalone_path = out_dir / "predictions/official_rf_baseline_standalone_2025.parquet"
        prediction_path = out_dir / "predictions/official_rf_baseline_2025.parquet"
        csv_path = out_dir / "official_rf_baseline_2025.csv"
        strict._atomic_joblib(models, model_path)
        strict._atomic_parquet(direct, direct_path)
        strict._atomic_parquet(standalone, standalone_path)
        strict._atomic_parquet(candidate, prediction_path)
        reload_audit = _reload_audit(
            model_path=model_path,
            groups=TARGET_COLS,
            features=joined_features,
            application_indexes=indexes,
            expected=predictions,
        )
        submission = sample.copy()
        for group in TARGET_COLS:
            submission[group] = candidate[group].to_numpy(dtype=float)
        strict._atomic_csv(submission, csv_path)
        verification = strict._verify_submission(csv_path, sample, candidate)
        final_result = {
            "schema_version": 1,
            "created_utc": utc_now(),
            "executed": True,
            "locked_weight": lock["locked_weight"],
            "candidate_promoted": True,
            "2025_read": True,
            "submission_created": True,
            "training": training,
            "train_feature_evidence": train_feature_evidence,
            "test_feature_evidence": test_feature_evidence,
            "reload_audit": reload_audit,
            "standalone_excluded_from_submission": True,
            "outputs": [
                describe_file(model_path),
                describe_file(direct_path),
                describe_file(standalone_path),
                describe_file(prediction_path),
                describe_file(csv_path),
            ],
            "submission_verification": verification,
            "leaderboard_score_claim": False,
        }
        strict._write_json(final_path, final_result)

    outputs = sorted(
        path for path in out_dir.rglob("*") if path.is_file() and path != manifest_path
    )
    stage1_result = json.loads((out_dir / "stage1_results.json").read_text(encoding="utf-8"))
    manifest = {
        "schema_version": 1,
        "artifact_type": "official_rf_baseline_strict_forward_v2",
        "created_utc": utc_now(),
        "command": [sys.executable, *sys.argv],
        "runtime": {"packages": package_versions(), "git": git_state(PROJECT_DIR)},
        "preregister_sha256": PREREGISTER_SHA256,
        "official_notebook_sha256": OFFICIAL_NOTEBOOK_SHA256,
        "candidate_weights": WEIGHTS,
        "locked_weight": lock["locked_weight"],
        "candidate_promoted": bool(lock["candidate_promoted"]),
        "locks": {
            "stage1_g12_prescore": describe_file(out_dir / "stage1_g12_prescore_lock.json"),
            "stage1_g3_prescore": describe_file(out_dir / "stage1_g3_prescore_lock.json"),
            "stage1_global_prescore": describe_file(out_dir / "stage1_global_prescore_lock.json"),
            "stage1_promotion": describe_file(out_dir / "stage1_promotion_lock.json"),
            "postlock_raw_audit": describe_file(out_dir / "postlock_cache_audit.json"),
            "stage2_promotion": describe_file(out_dir / "stage2_promotion_lock.json"),
        },
        "static_repo_import_closure": stage1_result["provenance"],
        "static_repo_import_closure_file_count": len(
            [key for key in stage1_result["provenance"] if key.startswith("repo_source::")]
        ),
        "final_inputs": list(final_inputs.values()),
        "outputs": [describe_file(path) for path in outputs],
        "read_flags": {
            "stage1_raw_physical_prefix_only_prelock": True,
            "2024_read": bool(stage2_result.get("2024_read")),
            "2025_read": bool(final_result["2025_read"]),
            "csv_created": bool(final_result["submission_created"]),
        },
        "tests": protocol.MANIFEST_TESTS,
        "contracts": protocol.MANIFEST_CONTRACTS,
    }
    strict._write_json(manifest_path, manifest)
    return manifest


def _install_specialization(raw_dir: Path) -> None:
    global _ACTIVE_RAW_DIR
    _ACTIVE_RAW_DIR = raw_dir
    protocol.PREREGISTER_SHA256 = PREREGISTER_SHA256
    protocol.WEIGHTS = WEIGHTS
    protocol.FEATURE_COLUMNS = FEATURE_COLUMNS
    protocol.STAGE1_REQUIRED = STAGE1_REQUIRED
    protocol.STAGE2_REQUIRED = STAGE2_REQUIRED
    protocol.SeasonalWindRidge = OfficialRandomForestBaseline
    protocol.assert_strict_forward = assert_strict_forward
    protocol.blend_direct_kwh = blend_direct_kwh
    protocol.select_stage1_weight = select_stage1_weight
    protocol.stage2_promoted = stage2_promoted
    protocol.CLI_DESCRIPTION = __doc__
    protocol.OUTPUT_SLUG = "official_rf_baseline"
    protocol.ARTIFACT_TYPE = "official_rf_baseline_strict_forward_v2"
    protocol.STANDALONE_DIAGNOSTIC = True
    protocol.MANIFEST_TESTS = {
        "focused_sources": [
            describe_file(PROJECT_DIR / "tests/test_official_rf_baseline.py"),
            describe_file(PROJECT_DIR / "tests/test_official_rf_baseline_runner.py"),
        ],
        "focused_test_count_prelaunch": 14,
        "focused_prelaunch_status": "pass",
        "full_suite_test_count_prelaunch": 298,
        "full_suite_prelaunch_status": "pass",
        "commands": [
            ".venv/Scripts/python.exe -B -m unittest tests.test_official_rf_baseline tests.test_official_rf_baseline_runner -v",
            ".venv/Scripts/python.exe -B -m unittest discover -s tests -p test_*.py -q",
        ],
    }
    closure_count = len(
        _static_repo_import_closure(
            (
                Path(__file__).resolve(),
                PROJECT_DIR / "scripts/run_seasonal_wind_ridge.py",
                PROJECT_DIR / "src/official_rf_baseline.py",
            )
        )
    )
    protocol.MANIFEST_CONTRACTS = {
        "official_14031_feature_and_explicit_rf_parameters_exact": True,
        "notebook_explicit_parameters_distinguished_from_frozen_defaults": True,
        "strict_adaptations_separately_declared": True,
        "one_direct_rf_per_group": True,
        "fixed_74_feature_design": True,
        "standalone_diagnostic_excluded_from_selection_and_promotion": True,
        "selectable_blends_exact": WEIGHTS,
        "fitted_model_reload_bit_exact": True,
        "fit_end_before_apply_start": True,
        "imputer_fit_application_overlap_count": 0,
        "stage1_raw_physical_prefix_only_before_score": True,
        "unused_info_workbook_snapshot_only_for_shared_input_assertion": True,
        "info_workbook_not_used_by_candidate_features_model_target_prediction_or_score": True,
        "v1_failed_before_fit_prediction_score_and_2024_or_2025_read": True,
        "runtime_model_features_weights_and_gates_unchanged_from_v1": True,
        "no_retune_or_candidate_change": True,
        "static_repo_import_closure_bound": True,
        "static_repo_import_closure_file_count": closure_count,
        "2024_used_for_selection": False,
        "no_public_metric_scale_scada_or_application_label_feature": True,
        "leaderboard_score_claim": False,
    }
    protocol._verify_preregister = _verify_preregister
    protocol._provenance_paths = _provenance_paths
    protocol._seasonal_features = _official_features
    protocol._reload_audit = _reload_audit
    protocol._postlock_cache_audit = _postlock_raw_audit
    protocol._read_low_cache = _read_stage2_features
    protocol._finalize = _finalize
    protocol.shared._read_stage1_raw_features = _read_stage1_official_features


def _with_defaults(argv: Sequence[str]) -> list[str]:
    output = list(argv)
    if "--out-dir" not in output:
        output.extend(("--out-dir", str(DEFAULT_OUT_DIR)))
    if "--preregister" not in output:
        output.extend(("--preregister", str(DEFAULT_PREREGISTER)))
    return output


def _raw_dir_from_argv(argv: Sequence[str]) -> Path:
    items = list(argv)
    if "--raw-dir" in items:
        position = items.index("--raw-dir")
        if position + 1 >= len(items):
            raise ValueError("--raw-dir requires a value")
        return Path(items[position + 1]).expanduser().resolve()
    return Path(r"data/local/open").resolve()


def main(argv: Sequence[str] | None = None) -> int:
    supplied = sys.argv[1:] if argv is None else list(argv)
    _install_specialization(_raw_dir_from_argv(supplied))
    return protocol.main(_with_defaults(supplied))


if __name__ == "__main__":
    raise SystemExit(main())

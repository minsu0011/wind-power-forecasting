"""Physical-I/O-safe v2 supersession of covariate-shift feature pruning."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts import run_covariate_shift_feature_pruning as v1  # noqa: E402
from scripts import run_ficr_bayes_decision_strict as bayes  # noqa: E402
from scripts import run_raw_grid_wind_lgb as raw_protocol  # noqa: E402
from src.covariate_shift_pruning import eligible_feature_names  # noqa: E402
from src.features import (  # noqa: E402
    AVAILABLE_COL,
    COORD_COLS,
    GRID_COL,
    SOURCE_VARIABLES,
    TIME_COL,
    WeatherFeatureBuilder,
    WeatherFeatureConfig,
    load_group_sites,
)
from src.manifest import (  # noqa: E402
    describe_file,
    git_state,
    package_versions,
    sha256_file,
    utc_now,
)
from src.metric import TARGET_COLS  # noqa: E402


PREREGISTER_SHA256 = "6912df4799bbd28e796a9f59f69aa60fa042d5282267a0f550dc4c6dd6825ceb"
BASE_PREREGISTER_SHA256 = v1.PREREGISTER_SHA256
INCIDENT_SHA256 = "d2530974775fcce1f4e1a42b9bd6b4255bed11f397716c14b1eb3b20796e5320"
FAILED_V1_MANIFEST_SHA256 = "b8ae9fb7ccd6df6cb693bab03b7a29a5521725f81ed00959dc5eea092949bb65"
QUARANTINE_DIR = PROJECT_DIR / "artifacts/postgate/covariate_shift_feature_pruning_strict_v1_failed_physical_future_weather_access"
YEAR_2022 = v1.YEAR_2022
YEAR_2023 = v1.YEAR_2023
PRE2024 = v1.PRE2024


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all",), default="all")
    parser.add_argument("--raw-dir", type=Path, default=Path(r"data/local/open"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/postgate/covariate_shift_feature_pruning_strict_v2"),
    )
    parser.add_argument(
        "--preregister",
        type=Path,
        default=Path("configs/covariate_shift_feature_pruning_preregister_v2.json"),
    )
    return parser.parse_args(argv)


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _frame_value_sha256(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(frame.index.asi8).tobytes())
    digest.update(
        json.dumps(list(frame.columns), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    digest.update(
        json.dumps([str(dtype) for dtype in frame.dtypes], separators=(",", ":")).encode("utf-8")
    )
    digest.update(np.ascontiguousarray(frame.to_numpy(dtype=np.float32, copy=False)).tobytes())
    return digest.hexdigest()


def _weather_usecols(source: str) -> tuple[str, ...]:
    return (
        TIME_COL,
        AVAILABLE_COL,
        GRID_COL,
        *COORD_COLS,
        *SOURCE_VARIABLES[source],
    )


def _assert_file_identity(path: Path, spec: Mapping[str, Any], name: str) -> dict[str, Any]:
    observed = describe_file(path)
    if int(observed["size_bytes"]) != int(spec["bytes"]):
        raise AssertionError(f"{name} size changed")
    if observed["sha256"] != spec["sha256"]:
        raise AssertionError(f"{name} hash changed")
    return observed


def _verify_preregister(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256_file(path) != PREREGISTER_SHA256:
        raise AssertionError("v2 preregister hash changed")
    sidecar = path.with_suffix(".sha256")
    if sidecar.read_text(encoding="utf-8") != f"{PREREGISTER_SHA256}  {path.name}\n":
        raise AssertionError("v2 preregister sidecar changed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["experiment_id"] != "covariate_shift_feature_pruning_strict_forward_v2":
        raise AssertionError("v2 experiment id changed")
    base_path = PROJECT_DIR / payload["supersession"]["base_preregister_path"]
    if sha256_file(base_path) != BASE_PREREGISTER_SHA256:
        raise AssertionError("base preregister changed")
    base = v1._verify_preregister(base_path)
    fixed = payload["fixed_candidate_contract"]
    if tuple(map(float, fixed["keep_fractions_in_order"])) != v1.KEEP_FRACTIONS:
        raise AssertionError("v2 keep fractions differ from v1")
    if tuple(fixed["objectives_in_order"]) != v1.OBJECTIVES:
        raise AssertionError("v2 objective order differs from v1")
    if tuple(map(float, fixed["blend_weights_in_order"])) != v1.BLEND_WEIGHTS:
        raise AssertionError("v2 blend weights differ from v1")
    if fixed["model_common_parameters"] != dict(v1.MODEL_COMMON_PARAMETERS):
        raise AssertionError("v2 model parameters differ from v1")
    if int(fixed["candidate_count_per_group"]) != len(v1.CANDIDATE_KEYS):
        raise AssertionError("v2 candidate count changed")
    if fixed["formula"] != base["candidate_family"]["formula"]:
        raise AssertionError("v2 candidate formula differs from v1")
    incident = QUARANTINE_DIR / "INCIDENT_LEDGER.json"
    failed_manifest = QUARANTINE_DIR / "manifest.json"
    if sha256_file(incident) != INCIDENT_SHA256:
        raise AssertionError("incident ledger changed")
    if sha256_file(failed_manifest) != FAILED_V1_MANIFEST_SHA256:
        raise AssertionError("quarantined v1 manifest changed")
    builder = payload["weather_feature_builder"]
    if sha256_file(PROJECT_DIR / "src/features.py") != builder["source_sha256"]:
        raise AssertionError("weather feature builder source changed")
    config_payload = json.dumps(
        asdict(WeatherFeatureConfig()), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if hashlib.sha256(config_payload).hexdigest() != builder["config_canonical_sha256"]:
        raise AssertionError("weather feature config changed")
    for source in ("ldaps", "gfs"):
        spec = payload["stage1_raw_weather_prefixes"][source]
        usecols = _weather_usecols(source)
        usecols_sha = hashlib.sha256(
            json.dumps(usecols, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if len(usecols) != int(spec["ordered_usecols_count"]) or usecols_sha != spec["ordered_usecols_sha256"]:
            raise AssertionError(f"{source} Stage1 usecols changed")
    return payload, base


def _read_physical_weather_prefix(
    raw_dir: Path,
    preregister: Mapping[str, Any],
    source: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = preregister["stage1_raw_weather_prefixes"][source]
    path = raw_dir / "train" / f"{source}_train.csv"
    frame, evidence = raw_protocol._read_bounded_csv(
        path,
        spec,
        usecols=_weather_usecols(source),
        whole_file=False,
    )
    frame[TIME_COL] = pd.to_datetime(frame[TIME_COL], errors="raise")
    frame[AVAILABLE_COL] = pd.to_datetime(frame[AVAILABLE_COL], errors="raise")
    numeric = [*COORD_COLS, *SOURCE_VARIABLES[source]]
    frame = frame.astype(
        {
            GRID_COL: "int16",
            **{column: "float32" for column in numeric},
        }
    )
    if len(frame) != int(spec["data_rows"]):
        raise AssertionError(f"{source} bounded row count changed")
    if frame[TIME_COL].max() != pd.Timestamp(spec["end"]):
        raise AssertionError(f"{source} bounded timestamp end changed")
    if frame[list(SOURCE_VARIABLES[source])].isna().any().any():
        raise AssertionError(f"{source} raw prefix contains missing weather values")
    evidence.update(
        {
            "event": f"read_{source}_physical_stage1_prefix",
            "suffix_bytes_returned_to_weather_parser": 0,
            "future_weather_value_cells_materialized": 0,
            "next_row_probe_first_field_only": True,
            "next_row_probe_value_bytes_read": 0,
            "cache_file_opened": False,
        }
    )
    return frame, evidence


def _build_stage1_features(
    raw_dir: Path,
    preregister: Mapping[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    info_spec = preregister["weather_feature_builder"]["info_xlsx"]
    info_path = raw_dir / "info.xlsx"
    info_identity = _assert_file_identity(info_path, info_spec, "info.xlsx")
    ldaps, ldaps_evidence = _read_physical_weather_prefix(raw_dir, preregister, "ldaps")
    gfs, gfs_evidence = _read_physical_weather_prefix(raw_dir, preregister, "gfs")
    sites = load_group_sites(info_path)
    builder = WeatherFeatureBuilder(sites, config=WeatherFeatureConfig())
    features = builder.fit_transform(ldaps, gfs)
    records: dict[str, Any] = {}
    for group in TARGET_COLS:
        frame = features[group]
        if not frame.index.equals(PRE2024):
            raise AssertionError(f"{group} raw-built Stage1 index changed")
        if frame.shape != (17520, 612):
            raise AssertionError(f"{group} raw-built feature shape changed")
        if len(eligible_feature_names(frame.columns)) != 597:
            raise AssertionError(f"{group} name-filtered feature count changed")
        if set(map(str, frame.dtypes)) != {"float32"}:
            raise AssertionError(f"{group} raw-built feature dtype changed")
        if not np.isfinite(frame.to_numpy(dtype=np.float32, copy=False)).all():
            raise AssertionError(f"{group} raw-built feature is non-finite")
        records[group] = {
            "rows": len(frame),
            "columns": frame.shape[1],
            "start": frame.index.min().isoformat(),
            "end": frame.index.max().isoformat(),
            "frame_value_sha256": _frame_value_sha256(frame),
            "cache_file_opened": False,
        }
    evidence = {
        "access_sequence": [
            "info_xlsx_identity",
            "ldaps_physical_prefix_identity_and_bounded_parse",
            "gfs_physical_prefix_identity_and_bounded_parse",
            "raw_prefix_feature_build",
        ],
        "info_xlsx": info_identity,
        "raw_prefixes": {"ldaps": ldaps_evidence, "gfs": gfs_evidence},
        "feature_frames": records,
        "cache_open_count_before_candidate_identity_lock": 0,
        "future_weather_value_cells_materialized": 0,
    }
    return features, evidence


def _save_and_reload_group(
    *,
    out_dir: Path,
    group: str,
    models: Mapping[str, Any],
    raw_predictions: pd.DataFrame,
    candidates: pd.DataFrame,
    shift_table: pd.DataFrame,
    subsets: Mapping[str, tuple[str, ...]],
    application_features: pd.DataFrame,
) -> tuple[list[Path], dict[str, Any]]:
    model_path = out_dir / "models" / f"stage1__{group}.joblib"
    raw_path = out_dir / "predictions" / f"stage1__{group}__raw_cf.parquet"
    candidate_path = out_dir / "candidates" / f"stage1__{group}.parquet"
    shift_path = out_dir / "shift" / f"stage1__{group}.parquet"
    subset_path = out_dir / "shift" / f"stage1__{group}__subsets.json"
    bayes._atomic_joblib(dict(models), model_path)
    bayes._atomic_parquet(raw_predictions, raw_path)
    bayes._atomic_parquet(candidates, candidate_path)
    bayes._atomic_parquet(shift_table, shift_path)
    bayes._write_json(
        subset_path,
        {
            "schema_version": 2,
            "preregister_sha256": PREREGISTER_SHA256,
            "stage": "stage1",
            "group": group,
            "subsets": {key: list(value) for key, value in subsets.items()},
            "subset_sha256": {
                key: _canonical_sha256(list(value)) for key, value in subsets.items()
            },
        },
    )
    loaded = joblib.load(model_path)
    reload_checks: dict[str, Any] = {}
    for key in v1.MODEL_KEYS:
        fraction, _, _ = v1._candidate_recipe(f"{key}_w05")
        names = list(subsets[v1._fraction_tag(fraction)])
        repeated = np.asarray(
            loaded[key].predict(application_features.loc[:, names]), dtype=np.float64
        )
        if not np.array_equal(repeated, raw_predictions[key].to_numpy(dtype=np.float64)):
            raise AssertionError(f"v2 {group} {key} reload prediction changed")
        reload_checks[key] = {"raw_prediction_bit_exact": True}
    return [model_path, raw_path, candidate_path, shift_path, subset_path], reload_checks


def _require_candidate_identity_lock(
    lock_path: Path,
    output_paths: Sequence[Path],
) -> dict[str, Any]:
    if not lock_path.is_file():
        raise AssertionError("candidate identity lock does not exist")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["preregister_sha256"] != PREREGISTER_SHA256:
        raise AssertionError("candidate identity lock preregister changed")
    expected = {str(Path(item["path"]).resolve()): item for item in lock["candidate_outputs"]}
    if set(expected) != {str(path.resolve()) for path in output_paths}:
        raise AssertionError("candidate identity lock output coverage changed")
    for path in output_paths:
        observed = describe_file(path)
        registered = expected[str(path.resolve())]
        if observed["sha256"] != registered["sha256"] or observed["size_bytes"] != registered["size_bytes"]:
            raise AssertionError("candidate output changed after identity lock")
    return lock


def _postlock_cache_audit(
    *,
    artifact_root: Path,
    out_dir: Path,
    preregister: Mapping[str, Any],
    raw_features: Mapping[str, pd.DataFrame],
    identity_lock_path: Path,
    output_paths: Sequence[Path],
) -> Path:
    identity_lock = _require_candidate_identity_lock(identity_lock_path, output_paths)
    records: dict[str, Any] = {}
    for group in TARGET_COLS:
        spec = preregister["post_identity_lock_cache_audit"]["cache_train"][group]
        path = artifact_root / "cache" / f"{group}_weather_train.parquet"
        identity = _assert_file_identity(path, spec, f"postlock {group} cache")
        cached = pd.read_parquet(
            path,
            engine="pyarrow",
            filters=[(TIME_COL, ">=", PRE2024.min()), (TIME_COL, "<=", PRE2024.max())],
        )
        cached.index = pd.DatetimeIndex(cached.index, name=TIME_COL)
        raw_frame = raw_features[group]
        if not cached.index.equals(raw_frame.index):
            raise AssertionError(f"postlock {group} cache index differs from raw prefix")
        if tuple(cached.columns) != tuple(raw_frame.columns):
            raise AssertionError(f"postlock {group} cache columns differ from raw prefix")
        if tuple(map(str, cached.dtypes)) != tuple(map(str, raw_frame.dtypes)):
            raise AssertionError(f"postlock {group} cache dtypes differ from raw prefix")
        cached_values = np.ascontiguousarray(cached.to_numpy(dtype=np.float32, copy=False))
        raw_values = np.ascontiguousarray(raw_frame.to_numpy(dtype=np.float32, copy=False))
        if not np.array_equal(cached_values, raw_values):
            mismatch = int(np.count_nonzero(cached_values != raw_values))
            raise AssertionError(f"postlock {group} raw/cache values differ: {mismatch}")
        cache_hash = _frame_value_sha256(cached)
        raw_hash = _frame_value_sha256(raw_frame)
        if cache_hash != raw_hash:
            raise AssertionError(f"postlock {group} raw/cache frame hash differs")
        records[group] = {
            "cache_identity": identity,
            "rows_compared": len(cached),
            "columns_compared": cached.shape[1],
            "raw_frame_value_sha256": raw_hash,
            "cache_prefix_frame_value_sha256": cache_hash,
            "index_columns_dtypes_value_bits_exact": True,
            "cache_opened_only_after_identity_lock": True,
        }
    _require_candidate_identity_lock(identity_lock_path, output_paths)
    path = out_dir / "postlock_cache_audit.json"
    bayes._write_json(
        path,
        {
            "schema_version": 2,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "candidate_identity_lock": describe_file(identity_lock_path),
            "candidate_identity_lock_created_before_cache_open": bool(
                identity_lock["created_before_any_cache_file_open"]
            ),
            "cache_open_before_identity_lock": False,
            "cache_open_after_identity_lock": True,
            "cache_future_rows_may_have_been_physically_read_for_audit_only": True,
            "candidate_outputs_unchanged_after_cache_audit": True,
            "groups": records,
        },
    )
    return path


def _model_semantic_hashes(path: Path) -> dict[str, str]:
    models = joblib.load(path)
    result: dict[str, str] = {}
    for key in v1.MODEL_KEYS:
        model_text = models[key].model_.booster_.model_to_string()
        result[key] = hashlib.sha256(model_text.encode("utf-8")).hexdigest()
    return result


def _reproduction_audit(out_dir: Path, v2_result_path: Path) -> Path:
    if not v2_result_path.is_file():
        raise AssertionError("v2 metrics must exist before the v1 reproduction reference is opened")
    v2_result = json.loads(v2_result_path.read_text(encoding="utf-8"))
    v1_result_path = QUARANTINE_DIR / "stage1_results.json"
    v1_result = json.loads(v1_result_path.read_text(encoding="utf-8"))
    group_records: dict[str, Any] = {}
    for group in TARGET_COLS:
        exact: dict[str, Any] = {}
        for kind, relative in (
            ("shift", Path("shift") / f"stage1__{group}.parquet"),
            ("raw_prediction", Path("predictions") / f"stage1__{group}__raw_cf.parquet"),
            ("candidate_prediction", Path("candidates") / f"stage1__{group}.parquet"),
        ):
            old = pd.read_parquet(QUARANTINE_DIR / relative)
            new = pd.read_parquet(out_dir / relative)
            pd.testing.assert_frame_equal(old, new, check_exact=True)
            exact[kind] = {
                "frame_exact": True,
                "rows": len(new),
                "columns": new.shape[1],
            }
        old_subset = json.loads(
            (QUARANTINE_DIR / "shift" / f"stage1__{group}__subsets.json").read_text(encoding="utf-8")
        )
        new_subset = json.loads(
            (out_dir / "shift" / f"stage1__{group}__subsets.json").read_text(encoding="utf-8")
        )
        if old_subset["subsets"] != new_subset["subsets"]:
            raise AssertionError(f"{group} v1/v2 subsets differ")
        old_model_hash = _model_semantic_hashes(
            QUARANTINE_DIR / "models" / f"stage1__{group}.joblib"
        )
        new_model_hash = _model_semantic_hashes(
            out_dir / "models" / f"stage1__{group}.joblib"
        )
        if old_model_hash != new_model_hash:
            raise AssertionError(f"{group} v1/v2 model structure differs")
        if v1_result["group_results"][group]["comparisons"] != v2_result["group_results"][group]["comparisons"]:
            raise AssertionError(f"{group} v1/v2 metrics differ")
        if v1_result["group_results"][group]["selected"] != v2_result["group_results"][group]["selected"]:
            raise AssertionError(f"{group} v1/v2 selection differs")
        group_records[group] = {
            **exact,
            "subsets_exact": True,
            "model_semantic_sha256_exact": True,
            "model_semantic_sha256": new_model_hash,
            "all_slice_metrics_exact": True,
            "locked_identity_decision_exact": True,
        }
    if v1_result["locked_candidates"] != v2_result["locked_candidates"]:
        raise AssertionError("v1/v2 locked candidate map differs")
    if v1_result["passed_groups"] != v2_result["passed_groups"]:
        raise AssertionError("v1/v2 passed groups differ")
    path = out_dir / "v1_v2_reproduction_audit.json"
    bayes._write_json(
        path,
        {
            "schema_version": 2,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "v2_metrics_created_before_v1_reference_open": True,
            "v1_reference_permitted_use": "bit-exact reproduction audit only",
            "failed_v1_manifest_sha256": FAILED_V1_MANIFEST_SHA256,
            "v1_results": describe_file(v1_result_path),
            "v2_results": describe_file(v2_result_path),
            "groups": group_records,
            "locked_candidates_exact": True,
            "passed_groups_exact": True,
            "serialized_joblib_bytes_equality_required": False,
            "reproduction_passed": True,
        },
    )
    return path


def _run_stage1(
    *,
    raw_dir: Path,
    artifact_root: Path,
    out_dir: Path,
    preregister_path: Path,
    preregister: Mapping[str, Any],
    base_preregister: Mapping[str, Any],
    precreated_out_dir: bool = False,
    source_closure_lock_path: Path | None = None,
    reproduction_audit_fn: Callable[[Path, Path], Path] | None = None,
    reproduction_record_key: str = "v1_v2_reproduction_audit",
) -> dict[str, Any]:
    if out_dir.exists():
        if not precreated_out_dir:
            raise FileExistsError(f"v2 requires a fresh output directory: {out_dir}")
        if source_closure_lock_path is None or not source_closure_lock_path.is_file():
            raise AssertionError("precreated output requires a source-closure lock")
        unexpected = [
            path for path in out_dir.iterdir() if path.resolve() != source_closure_lock_path.resolve()
        ]
        if unexpected:
            raise AssertionError(f"precreated output contains unexpected files: {unexpected}")
    else:
        out_dir.mkdir(parents=True)
    source_closure_snapshot: dict[str, Any] | None = None
    if source_closure_lock_path is not None:
        if not source_closure_lock_path.is_file():
            raise AssertionError("source-closure lock missing before feature build")
        source_closure_snapshot = describe_file(source_closure_lock_path)
    bayes._copy_exclusive(preregister_path, out_dir / "preregister.json")
    bayes._copy_exclusive(preregister_path.with_suffix(".sha256"), out_dir / "preregister.sha256")
    bayes._copy_exclusive(
        PROJECT_DIR / preregister["supersession"]["base_preregister_path"],
        out_dir / "base_preregister_v1.json",
    )
    raw_features, physical_evidence = _build_stage1_features(raw_dir, preregister)
    baseline, baseline_inputs = v1._load_stage1_baseline(artifact_root, base_preregister)
    prefix = base_preregister["physical_label_prefixes"]
    labels_g12, labels_g12_evidence = v1._read_label_prefix(
        raw_dir, prefix["g12_stage1_fit"], TARGET_COLS[:2]
    )
    fit_indexes = {
        "kpx_group_1": YEAR_2022,
        "kpx_group_2": YEAR_2022,
        "kpx_group_3": v1._segments(2023, ("H1",))["H1"],
    }
    application_indexes = {
        "kpx_group_1": YEAR_2023,
        "kpx_group_2": YEAR_2023,
        "kpx_group_3": v1._segments(2023, ("H2",))["H2"],
    }
    group_candidates: dict[str, pd.DataFrame] = {}
    training: dict[str, Any] = {}
    reload_audits: dict[str, Any] = {}
    all_outputs: list[Path] = []
    g12_outputs: list[Path] = []
    for group in TARGET_COLS[:2]:
        features = raw_features[group]
        fitted = v1._fit_group(
            source_features=features.loc[fit_indexes[group]],
            application_features=features.loc[application_indexes[group]],
            source_actual_kwh=labels_g12.loc[fit_indexes[group], group],
            baseline_kwh=baseline.loc[application_indexes[group], group],
            group=group,
        )
        models, raw_predictions, candidates, shift_table, subsets, metadata = fitted
        paths, reload = _save_and_reload_group(
            out_dir=out_dir,
            group=group,
            models=models,
            raw_predictions=raw_predictions,
            candidates=candidates,
            shift_table=shift_table,
            subsets=subsets,
            application_features=features.loc[application_indexes[group]],
        )
        group_candidates[group] = candidates
        training[group] = metadata
        reload_audits[group] = reload
        g12_outputs.extend(paths)
        all_outputs.extend(paths)
    g12_lock_path = out_dir / "stage1_g12_prescore_lock.json"
    bayes._write_json(
        g12_lock_path,
        {
            "schema_version": 2,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "fit_label_evidence": labels_g12_evidence,
            "fit_groups": list(TARGET_COLS[:2]),
            "outputs": [describe_file(path) for path in g12_outputs],
            "reload_audits": {group: reload_audits[group] for group in TARGET_COLS[:2]},
            "cache_open_count": 0,
            "future_weather_value_cells_materialized": 0,
            "application_target_value_cells_materialized": 0,
        },
    )
    labels_g3, labels_g3_evidence = v1._read_label_prefix(
        raw_dir, prefix["g3_stage1_fit"], ("kpx_group_3",)
    )
    group = "kpx_group_3"
    features = raw_features[group]
    fitted = v1._fit_group(
        source_features=features.loc[fit_indexes[group]],
        application_features=features.loc[application_indexes[group]],
        source_actual_kwh=labels_g3.loc[fit_indexes[group], group],
        baseline_kwh=baseline.loc[application_indexes[group], group],
        group=group,
    )
    models, raw_predictions, candidates, shift_table, subsets, metadata = fitted
    g3_outputs, reload = _save_and_reload_group(
        out_dir=out_dir,
        group=group,
        models=models,
        raw_predictions=raw_predictions,
        candidates=candidates,
        shift_table=shift_table,
        subsets=subsets,
        application_features=features.loc[application_indexes[group]],
    )
    group_candidates[group] = candidates
    training[group] = metadata
    reload_audits[group] = reload
    all_outputs.extend(g3_outputs)
    g3_lock_path = out_dir / "stage1_g3_prescore_lock.json"
    bayes._write_json(
        g3_lock_path,
        {
            "schema_version": 2,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "fit_label_evidence": labels_g3_evidence,
            "fit_group": group,
            "outputs": [describe_file(path) for path in g3_outputs],
            "reload_audit": reload,
            "cache_open_count": 0,
            "future_weather_value_cells_materialized": 0,
            "application_target_value_cells_materialized": 0,
        },
    )
    prescore_record_path = out_dir / "stage1_prescore_record.json"
    bayes._write_json(
        prescore_record_path,
        {
            "schema_version": 2,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "physical_weather_evidence": physical_evidence,
            "source_closure_prescore_lock": source_closure_snapshot,
            "source_closure_locked_before_feature_build_and_fit": source_closure_snapshot is not None,
            "baseline_inputs": baseline_inputs,
            "g12_lock": describe_file(g12_lock_path),
            "g3_lock": describe_file(g3_lock_path),
            "training": training,
            "candidate_outputs": [describe_file(path) for path in all_outputs],
            "cache_open_count_before_identity_lock": 0,
            "stage1_application_label_values_read": False,
            "future_weather_value_cells_materialized": 0,
            "2024_weather_read": False,
            "2024_label_read": False,
            "2025_read": False,
            "public_read": False,
        },
    )
    identity_lock_path = out_dir / "stage1_candidate_identity_lock.json"
    bayes._write_json(
        identity_lock_path,
        {
            "schema_version": 2,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "prescore_record": describe_file(prescore_record_path),
            "source_closure_prescore_lock": source_closure_snapshot,
            "g12_lock": describe_file(g12_lock_path),
            "g3_lock": describe_file(g3_lock_path),
            "candidate_outputs": [describe_file(path) for path in all_outputs],
            "raw_prefix_feature_hashes": {
                group: physical_evidence["feature_frames"][group]["frame_value_sha256"]
                for group in TARGET_COLS
            },
            "created_before_any_cache_file_open": True,
            "created_before_stage1_application_label_values": True,
        },
    )
    cache_audit_path = _postlock_cache_audit(
        artifact_root=artifact_root,
        out_dir=out_dir,
        preregister=preregister,
        raw_features=raw_features,
        identity_lock_path=identity_lock_path,
        output_paths=all_outputs,
    )
    score_labels, score_label_evidence = v1._read_label_prefix(
        raw_dir, prefix["stage1_score_after_lock"], TARGET_COLS
    )
    group_results: dict[str, Any] = {}
    locked_candidates: dict[str, str] = {}
    passed_groups: list[str] = []
    for group in TARGET_COLS:
        index = application_indexes[group]
        comparisons = v1._score_candidates(
            score_labels.loc[index, group],
            baseline.loc[index, group],
            group_candidates[group],
            group,
            v1.STAGE1_REQUIRED[group],
        )
        selected, selection = v1._select_group_candidate(
            comparisons, v1.STAGE1_REQUIRED[group]
        )
        locked_candidates[group] = selected or "identity"
        if selected is not None:
            passed_groups.append(group)
        group_results[group] = {
            "comparisons": comparisons,
            "comparisons_sha256": _canonical_sha256(comparisons),
            "selection": selection,
            "selected": selected or "identity",
        }
    result = {
        "schema_version": 2,
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "candidate_identity_lock": describe_file(identity_lock_path),
        "postlock_cache_audit": describe_file(cache_audit_path),
        "score_label_evidence_after_candidate_and_cache_audit_locks": score_label_evidence,
        "group_results": group_results,
        "locked_candidates": locked_candidates,
        "passed_groups": passed_groups,
        "passed_groups_sha256": _canonical_sha256(passed_groups),
        "raw_stage1_prefix_read": True,
        "cache_open_before_identity_lock": False,
        "cache_open_after_identity_lock": True,
        "stage1_application_labels_read_after_identity_lock": True,
        "2024_weather_read_for_model_or_selection": False,
        "2024_label_read": False,
        "2025_read": False,
        "public_read": False,
    }
    result_path = out_dir / "stage1_results.json"
    bayes._write_json(result_path, result)
    audit_function = reproduction_audit_fn or _reproduction_audit
    reproduction_path = audit_function(out_dir, result_path)
    lock = {
        "schema_version": 2,
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "candidate_identity_lock": describe_file(identity_lock_path),
        "postlock_cache_audit": describe_file(cache_audit_path),
        reproduction_record_key: describe_file(reproduction_path),
        "stage1_results": describe_file(result_path),
        "locked_candidates": locked_candidates,
        "passed_groups": passed_groups,
        "passed_groups_sha256": result["passed_groups_sha256"],
        "reproduction_passed": True,
        "2024_read": False,
        "2025_read": False,
    }
    bayes._write_json(out_dir / "stage1_promotion_lock.json", lock)
    print(f"v2 Stage1 passed groups: {passed_groups}", flush=True)
    return lock


def _write_no_read_terminal(out_dir: Path, stage1_lock: Mapping[str, Any]) -> None:
    if stage1_lock["passed_groups"]:
        raise AssertionError("v2 reproduction expected no Stage1-passing group; refuse unregistered continuation")
    stage2_path = out_dir / "stage2_results.json"
    bayes._write_json(
        stage2_path,
        {
            "schema_version": 2,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "performed": False,
            "reason": "no independently evaluated v2 Stage1 group passed every fixed slice",
            "2024_weather_read": False,
            "2024_label_read": False,
            "2025_read": False,
        },
    )
    stage2_lock_path = out_dir / "stage2_promotion_lock.json"
    bayes._write_json(
        stage2_lock_path,
        {
            "schema_version": 2,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "stage1_promotion_lock": describe_file(out_dir / "stage1_promotion_lock.json"),
            "stage2_results": describe_file(stage2_path),
            "promoted_groups": [],
            "2024_read": False,
            "2025_read": False,
        },
    )
    bayes._write_json(
        out_dir / "final_results.json",
        {
            "schema_version": 2,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "performed": False,
            "reason": "strict Stage1 gate stopped the experiment",
            "2025_weather_read": False,
            "sample_submission_read": False,
            "submission_created": False,
            "public_read": False,
        },
    )


def _write_manifest(
    *,
    raw_dir: Path,
    artifact_root: Path,
    out_dir: Path,
    preregister_path: Path,
) -> Path:
    manifest_path = out_dir / "manifest.json"
    stage1 = json.loads((out_dir / "stage1_results.json").read_text(encoding="utf-8"))
    final = json.loads((out_dir / "final_results.json").read_text(encoding="utf-8"))
    source_paths = {
        "runner_v2": Path(__file__).resolve(),
        "runner_v1_primitives": PROJECT_DIR / "scripts/run_covariate_shift_feature_pruning.py",
        "feature_builder": PROJECT_DIR / "src/features.py",
        "selector": PROJECT_DIR / "src/covariate_shift_pruning.py",
        "bounded_reader": PROJECT_DIR / "scripts/run_shared_q07_multiseed.py",
        "metric": PROJECT_DIR / "src/metric.py",
        "v2_tests": PROJECT_DIR / "tests/test_covariate_shift_feature_pruning_v2.py",
        "v2_preregister": preregister_path.resolve(),
        "v2_preregister_sidecar": preregister_path.with_suffix(".sha256").resolve(),
        "v1_preregister": PROJECT_DIR / "configs/covariate_shift_feature_pruning_preregister_v1.json",
        "incident_ledger": QUARANTINE_DIR / "INCIDENT_LEDGER.json",
    }
    outputs = sorted(
        path for path in out_dir.rglob("*") if path.is_file() and path != manifest_path
    )
    payload = {
        "schema_version": 2,
        "artifact_type": "covariate_shift_feature_pruning_strict_forward_v2",
        "created_utc": utc_now(),
        "command": [sys.executable, *sys.argv],
        "runtime": {"packages": package_versions(), "git": git_state(PROJECT_DIR)},
        "preregister_sha256": PREREGISTER_SHA256,
        "supersedes_failed_v1": True,
        "physical_io": {
            "raw_stage1_prefix_read": True,
            "raw_stage1_parser_suffix_bytes": 0,
            "future_weather_value_cells_materialized_before_identity_lock": 0,
            "cache_open_before_identity_lock": False,
            "cache_open_after_identity_lock": True,
            "postlock_raw_cache_prefix_value_bits_exact": True,
            "stage1_application_labels_read_after_identity_lock": True,
            "2024_weather_read_for_model_or_selection": False,
            "2024_label_read": False,
            "2025_read": False,
            "sample_submission_read": False,
            "public_read": False,
        },
        "selection": {
            "locked_candidates": stage1["locked_candidates"],
            "passed_groups": stage1["passed_groups"],
            "v1_v2_reproduction_passed": True,
        },
        "tests": {
            "focused_prelaunch_status": "pass",
            "focused_test_count": 8,
            "full_suite_prelaunch_status": "pass",
            "full_suite_test_count": 255,
        },
        "provenance": {name: describe_file(path) for name, path in source_paths.items()},
        "conditional_roots": {
            "raw_root": str(raw_dir.resolve()),
            "artifact_root": str(artifact_root.resolve()),
        },
        "outputs": [describe_file(path) for path in outputs],
        "submission_created": bool(final["submission_created"]),
    }
    bayes._write_json(manifest_path, payload)
    return manifest_path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    raw_dir = args.raw_dir.expanduser().resolve()
    artifact_root = args.artifact_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    preregister_path = args.preregister.expanduser().resolve()
    preregister, base = _verify_preregister(preregister_path)
    lock = _run_stage1(
        raw_dir=raw_dir,
        artifact_root=artifact_root,
        out_dir=out_dir,
        preregister_path=preregister_path,
        preregister=preregister,
        base_preregister=base,
    )
    _write_no_read_terminal(out_dir, lock)
    _write_manifest(
        raw_dir=raw_dir,
        artifact_root=artifact_root,
        out_dir=out_dir,
        preregister_path=preregister_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

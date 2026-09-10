"""Run the frozen causal feature-exit / virtual-score experiment.

The three stages are intentionally separate:

``inner`` fits the five predeclared q0.7 variants on expanding causal folds;
``stress`` performs the frozen 10,000-draw seven-day block/max-T analysis; and
``outer`` opens the one-shot 2023 challenge only for an INNER-promoted family.

The reported virtual score is a local causal validation proxy.  It is not a
leaderboard-score prediction.  This CLI never reads 2024 labels or 2025 data.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import io
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

from src.density_ratio import assemble_locked_group  # noqa: E402
from src.manifest import describe_file, git_state, package_versions, sha256_file, utc_now  # noqa: E402
from src.metric import CAPACITY_KWH, TARGET_COLS, group_metrics  # noqa: E402
from src.virtual_feature_exit import (  # noqa: E402
    EXIT_FAMILIES,
    build_feature_taxonomy,
    kept_features,
    risk_adjusted_virtual_score,
)


PREREGISTER_SHA256 = "ccb857578b812e61e05fa63f43a45382f147f7abf3464bdb5c6a597bde42990c"
PREREGISTER_BYTES = 11_259
DEFAULT_PREREGISTER = PROJECT_DIR / "configs/feature_exit_virtual_score_preregister_v1.json"
DEFAULT_OUTPUT = PROJECT_DIR / "artifacts/feature_exit_virtual_v1"
DEFAULT_RAW_DIR = Path(r"data/local/open")
VARIANTS = ("control_all", *EXIT_FAMILIES)
EXIT_VARIANTS = tuple(EXIT_FAMILIES)
PREFIX_ROWS = 17_520
PREFIX_BYTES = 742_551
PREFIX_SHA256 = "ee9707cc81229db760fc035d0bf1d10eb3586d2fef95849bdf463d73fae3aacd"
PREFIX_START = pd.Timestamp("2022-01-01 01:00:00")
PREFIX_END = pd.Timestamp("2024-01-01 00:00:00")
EXPECTED_PREFIX_INDEX = pd.date_range(
    PREFIX_START, PREFIX_END, freq="h", name="forecast_kst_dtm"
)
EXPECTED_FEATURE_COUNT = 612
BLOCK_HOURS = 168
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_810
MODEL_SEED = 42
MODEL_JOBS = 7
MAX_T_QUANTILE = 0.90
COMPONENT_NAMES = (
    "lgb_l1",
    "lgb_q07",
    "shared_l1",
    "shared_q07",
    "top200_q07",
    "energy_q06",
)
G12_COMPONENT_PATHS: dict[str, str] = {
    "lgb_l1": "artifacts/oof/dev2023_lgb_l1_eligible_n1500.parquet",
    "lgb_q07": "artifacts/oof/dev2023_lgb_q07_eligible.parquet",
    "shared_l1": "artifacts/oof/dev2023_shared_l1_eligible.parquet",
    "shared_q07": "artifacts/oof/dev2023_shared_q07_eligible.parquet",
    "top200_q07": "artifacts/oof/dev2023_lgb_top200_q07_eligible.parquet",
    "energy_q06": "artifacts/oof/dev2023_lgb_q06_energywt_eligible.parquet",
}
G3_COMPONENT_PATH = "artifacts/oof/g3dev2023h2_candidates.parquet"
G3_COMPONENT_COLUMNS: dict[str, str] = {
    "lgb_l1": "l1",
    "lgb_q07": "q07",
    "shared_l1": "shared_l1",
    "shared_q07": "shared_q07",
    "top200_q07": "top200q07",
    "energy_q06": "ewq06",
}


@dataclass(frozen=True)
class CausalFold:
    name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp

    def masks(self, index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
        train = (index >= self.train_start) & (index <= self.train_end)
        valid = (index >= self.valid_start) & (index <= self.valid_end)
        if not np.any(train) or not np.any(valid):
            raise AssertionError(f"empty causal fold: {self.name}")
        if index[train].max() >= index[valid].min() or np.any(train & valid):
            raise AssertionError(f"non-causal fold: {self.name}")
        return np.asarray(train), np.asarray(valid)

    def as_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "valid_start": self.valid_start.isoformat(),
            "valid_end": self.valid_end.isoformat(),
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("inner", "stress", "outer"), required=True)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--artifact-root", type=Path, default=PROJECT_DIR / "artifacts")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preregister", type=Path, default=DEFAULT_PREREGISTER)
    return parser.parse_args(argv)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(_json_ready(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _write_parquet(frame: pd.DataFrame, path: Path, *, index: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        frame.to_parquet(temporary, engine="pyarrow", compression="zstd", index=index)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _write_joblib(value: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        joblib.dump(value, temporary, compress=3)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _write_npz(path: Path, **arrays: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _canonical_sha256_lines(values: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(map(str, values)) + "\n").encode("utf-8")).hexdigest()


def _verify_identity(path: Path, spec: Mapping[str, Any], context: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = describe_file(path)
    expected_bytes = int(spec.get("bytes", spec.get("size_bytes", -1)))
    if observed["size_bytes"] != expected_bytes or observed["sha256"] != str(spec["sha256"]):
        raise AssertionError(f"{context} identity changed: {observed}")
    return observed


def _verify_contract(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if path.stat().st_size != PREREGISTER_BYTES or sha256_file(path) != PREREGISTER_SHA256:
        raise AssertionError("frozen feature-exit preregistration changed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["experiment_id"] != "feature_exit_virtual_score_v1":
        raise AssertionError("unexpected experiment id")
    ledger_ids = tuple(row["id"] for row in payload["candidate_ledger_fixed_before_labels"])
    if ledger_ids != VARIANTS:
        raise AssertionError("candidate ledger differs from runner")
    protocol = payload["virtual_score_protocol"]
    if (
        int(protocol["bootstrap_replicates"]) != BOOTSTRAP_REPLICATES
        or int(protocol["seed"]) != BOOTSTRAP_SEED
    ):
        raise AssertionError("bootstrap contract differs from runner")
    return payload


def _copy_preregister(path: Path, out_dir: Path) -> tuple[Path, Path]:
    destination = out_dir / "preregister.json"
    sidecar = out_dir / "preregister.sha256"
    if destination.exists() or sidecar.exists():
        raise FileExistsError("preregister copy already exists")
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    shutil.copyfile(path, temporary)
    os.replace(temporary, destination)
    _write_json(
        sidecar,
        {
            "filename": destination.name,
            "sha256": PREREGISTER_SHA256,
            "size_bytes": PREREGISTER_BYTES,
        },
    )
    return destination, sidecar


def _folds(contract: Mapping[str, Any]) -> dict[str, tuple[CausalFold, ...]]:
    result: dict[str, tuple[CausalFold, ...]] = {}
    for group in TARGET_COLS:
        key = "g1_g2" if group in TARGET_COLS[:2] else "g3"
        values: list[CausalFold] = []
        for position, row in enumerate(contract["inner_folds"][key], start=1):
            values.append(
                CausalFold(
                    name=f"{key}_fold{position}",
                    train_start=pd.Timestamp(row[0]),
                    train_end=pd.Timestamp(row[1]),
                    valid_start=pd.Timestamp(row[2]),
                    valid_end=pd.Timestamp(row[3]),
                )
            )
        result[group] = tuple(values)
    if sum(map(len, result.values())) != 10:
        raise AssertionError("registered inner cell count changed")
    return result


def _read_bounded_labels(path: Path, contract: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = contract["data_contract"]["train_label_prefix"]
    if int(spec["data_rows"]) != PREFIX_ROWS or int(spec["bytes"]) != PREFIX_BYTES:
        raise AssertionError("label prefix contract differs from runner")
    with path.open("rb", buffering=0) as stream:
        prefix = stream.read(PREFIX_BYTES)
        physical_position = stream.tell()
    if len(prefix) != PREFIX_BYTES or physical_position != PREFIX_BYTES:
        raise AssertionError("bounded label reader did not stop at exact byte cutoff")
    observed_sha = hashlib.sha256(prefix).hexdigest()
    if observed_sha != PREFIX_SHA256 or observed_sha != spec["sha256"]:
        raise AssertionError("bounded label prefix identity changed")
    frame = pd.read_csv(io.BytesIO(prefix), encoding="utf-8-sig", memory_map=False)
    if len(frame) != PREFIX_ROWS or tuple(frame.columns) != ("kst_dtm", *TARGET_COLS):
        raise AssertionError("bounded label prefix schema/row count changed")
    frame.index = pd.DatetimeIndex(
        pd.to_datetime(frame.pop("kst_dtm"), errors="raise"), name="forecast_kst_dtm"
    )
    frame = frame.astype(np.float64)
    if not frame.index.equals(EXPECTED_PREFIX_INDEX) or np.isinf(frame.to_numpy()).any():
        raise AssertionError("bounded label prefix index/values changed")
    return frame, {
        "path": str(path.resolve()),
        "physical_byte_limit": PREFIX_BYTES,
        "bytes_returned_to_parser": PREFIX_BYTES,
        "physical_prefix_sha256": observed_sha,
        "data_rows": PREFIX_ROWS,
        "last_timestamp": frame.index.max().isoformat(),
        "future_label_bytes_read": 0,
        "future_label_value_cells_materialized": 0,
        "full_file_hash_computed": False,
    }


def _read_weather_prefix(
    artifact_root: Path, contract: Mapping[str, Any]
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], tuple[str, ...]]:
    frames: dict[str, pd.DataFrame] = {}
    evidence: dict[str, Any] = {}
    canonical: tuple[str, ...] | None = None
    cache = contract["data_contract"]["weather_cache"]
    for group in TARGET_COLS:
        spec = cache[group]
        path = PROJECT_DIR / str(spec["path"])
        if not path.is_file():
            path = artifact_root / "cache" / f"{group}_weather_train.parquet"
        identity = _verify_identity(path, spec, f"{group} weather cache")
        frame = pd.read_parquet(
            path,
            engine="pyarrow",
            filters=[("forecast_kst_dtm", "<=", PREFIX_END)],
        )
        frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
        if not frame.index.equals(EXPECTED_PREFIX_INDEX):
            raise AssertionError(f"{group} weather prefix index changed")
        if frame.shape[1] != EXPECTED_FEATURE_COUNT or not frame.columns.is_unique:
            raise AssertionError(f"{group} feature schema changed")
        if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes):
            raise TypeError(f"{group} features are not all numeric")
        values = frame.to_numpy(dtype=np.float32, copy=False)
        if not np.isfinite(values).all():
            raise AssertionError(f"{group} feature prefix contains non-finite values")
        names = tuple(map(str, frame.columns))
        if canonical is not None and names != canonical:
            raise AssertionError("group feature schemas differ")
        canonical = names
        frames[group] = frame.astype(np.float32, copy=False)
        evidence[group] = {
            "identity": identity,
            "materialized_rows": len(frame),
            "materialized_end": frame.index.max().isoformat(),
            "physical_row_group_isolation_claimed": False,
        }
    assert canonical is not None
    if _canonical_sha256_lines(canonical) != cache["feature_name_order_sha256"]:
        raise AssertionError("canonical feature-name digest changed")
    return frames, evidence, canonical


def _taxonomy_lock(
    names: Sequence[str], contract: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, tuple[str, ...]], dict[str, Any]]:
    taxonomy = pd.DataFrame(build_feature_taxonomy(names))
    if len(taxonomy) != EXPECTED_FEATURE_COUNT or taxonomy["feature"].tolist() != list(names):
        raise AssertionError("feature taxonomy coverage/order changed")
    expected_family_counts = contract["feature_taxonomy"]["physical_family_counts"]
    observed_family_counts = taxonomy["physical_family"].value_counts().to_dict()
    if observed_family_counts != expected_family_counts:
        raise AssertionError(f"physical family counts changed: {observed_family_counts}")
    subsets: dict[str, tuple[str, ...]] = {}
    audit: dict[str, Any] = {}
    ledger = {row["id"]: row for row in contract["candidate_ledger_fixed_before_labels"]}
    for variant in VARIANTS:
        kept = kept_features(names, variant)
        removed = tuple(name for name in names if name not in set(kept))
        spec = ledger[variant]
        if len(kept) != int(spec["keep_count"]) or len(removed) != int(spec["remove_count"]):
            raise AssertionError(f"{variant} feature counts changed")
        if variant != "control_all" and _canonical_sha256_lines(removed) != spec["removed_names_sha256"]:
            raise AssertionError(f"{variant} removed-name digest changed")
        subsets[variant] = kept
        audit[variant] = {
            "keep_count": len(kept),
            "remove_count": len(removed),
            "kept_names_sha256": _canonical_sha256_lines(kept),
            "removed_names_sha256": _canonical_sha256_lines(removed),
        }
    return taxonomy, subsets, audit


def _model_parameters(contract: Mapping[str, Any]) -> dict[str, Any]:
    registered = dict(contract["model_contract"]["parameters"])
    if (
        registered.get("objective") != "quantile"
        or float(registered.get("alpha")) != 0.7
        or int(registered.get("n_estimators")) != 1500
        or int(registered.get("random_state")) != MODEL_SEED
        or int(registered.get("n_jobs")) != MODEL_JOBS
    ):
        raise AssertionError("registered LightGBM q07 recipe changed")
    return registered


def _fit_one(
    features: pd.DataFrame,
    actual: pd.Series,
    fold: CausalFold,
    columns: Sequence[str],
    capacity: float,
    params: Mapping[str, Any],
) -> tuple[Any, np.ndarray, dict[str, Any], pd.DatetimeIndex]:
    from lightgbm import LGBMRegressor

    train_mask, valid_mask = fold.masks(features.index)
    actual_values = actual.to_numpy(dtype=np.float64, copy=False)
    eligible = train_mask & np.isfinite(actual_values) & (actual_values >= 0.10 * capacity)
    if int(eligible.sum()) < 100:
        raise AssertionError(f"too few eligible rows in {fold.name}")
    selected = list(columns)
    model = LGBMRegressor(**dict(params))
    model.fit(features.loc[eligible, selected], actual.loc[eligible].astype(float) / capacity)
    valid_index = features.index[valid_mask]
    prediction = np.asarray(model.predict(features.loc[valid_index, selected]), dtype=np.float64)
    if not np.isfinite(prediction).all():
        raise AssertionError("non-finite q07 prediction")
    metadata = {
        **fold.as_dict(),
        "eligible_fit_rows": int(eligible.sum()),
        "validation_rows": len(valid_index),
        "feature_count": len(selected),
        "feature_names_sha256": _canonical_sha256_lines(selected),
        "fit_end_before_validation_start": True,
        "target_unit": "capacity_factor",
    }
    return model, prediction, metadata, valid_index


def _score_inner(oof: pd.DataFrame, labels: pd.DataFrame) -> dict[str, Any]:
    slice_rows: dict[str, Any] = {}
    group_full: dict[str, dict[str, Any]] = {variant: {} for variant in VARIANTS}
    cell_deltas: dict[str, list[float]] = {variant: [] for variant in EXIT_VARIANTS}
    cell_component_deltas: dict[str, list[dict[str, float]]] = {
        variant: [] for variant in EXIT_VARIANTS
    }
    for group in TARGET_COLS:
        group_rows = oof[oof["group"] == group].sort_values("forecast_kst_dtm")
        times = pd.DatetimeIndex(group_rows["forecast_kst_dtm"])
        actual = labels.loc[times, group].to_numpy(dtype=np.float64)
        capacity = CAPACITY_KWH[group]
        for variant in VARIANTS:
            metric = group_metrics(
                actual,
                group_rows[variant].to_numpy(dtype=np.float64) * capacity,
                capacity,
                group_name=group,
            )
            group_full[variant][group] = metric.as_dict()
        for fold_name, fold_rows in group_rows.groupby("fold", sort=False):
            fold_times = pd.DatetimeIndex(fold_rows["forecast_kst_dtm"])
            fold_actual = labels.loc[fold_times, group].to_numpy(dtype=np.float64)
            control = group_metrics(
                fold_actual,
                fold_rows["control_all"].to_numpy(dtype=np.float64) * capacity,
                capacity,
                group_name=group,
            )
            candidates: dict[str, Any] = {}
            for variant in EXIT_VARIANTS:
                candidate = group_metrics(
                    fold_actual,
                    fold_rows[variant].to_numpy(dtype=np.float64) * capacity,
                    capacity,
                    group_name=group,
                )
                delta_total = 0.5 * (
                    (candidate.one_minus_nmae - control.one_minus_nmae)
                    + (candidate.ficr - control.ficr)
                )
                component = {
                    "total": float(delta_total),
                    "one_minus_nmae": float(candidate.one_minus_nmae - control.one_minus_nmae),
                    "ficr": float(candidate.ficr - control.ficr),
                }
                cell_deltas[variant].append(float(delta_total))
                cell_component_deltas[variant].append(component)
                candidates[variant] = {"metrics": candidate.as_dict(), "delta": component}
            slice_rows[f"{group}::{fold_name}"] = {
                "group": group,
                "fold": fold_name,
                "control": control.as_dict(),
                "candidates": candidates,
            }
    macro: dict[str, Any] = {}
    for variant in VARIANTS:
        metrics = [group_full[variant][group] for group in TARGET_COLS]
        one_minus_nmae = float(np.mean([row["one_minus_nmae"] for row in metrics]))
        ficr = float(np.mean([row["ficr"] for row in metrics]))
        macro[variant] = {
            "total_score": 0.5 * (one_minus_nmae + ficr),
            "one_minus_nmae": one_minus_nmae,
            "ficr": ficr,
            "by_group": group_full[variant],
        }
    control = macro["control_all"]
    variants: dict[str, Any] = {}
    for variant in EXIT_VARIANTS:
        candidate = macro[variant]
        delta_n = float(candidate["one_minus_nmae"] - control["one_minus_nmae"])
        delta_f = float(candidate["ficr"] - control["ficr"])
        variants[variant] = {
            "direct_macro": candidate,
            "delta_score": float(candidate["total_score"] - control["total_score"]),
            "delta_N": delta_n,
            "delta_F": delta_f,
            "cell_deltas": cell_deltas[variant],
            "worst_cell": float(min(cell_deltas[variant])),
            "positive_cell_count": int(np.sum(np.asarray(cell_deltas[variant]) > 0.0)),
            "cell_component_deltas": cell_component_deltas[variant],
        }
    return {
        "schema_version": 1,
        "score_kind": "causal INNER official-direct diagnostics; not leaderboard score",
        "control_inner_official_score": control["total_score"],
        "control_direct_macro": control,
        "variants": variants,
        "registered_cells": slice_rows,
    }


def _run_inner(
    *, raw_dir: Path, artifact_root: Path, out_dir: Path, preregister_path: Path
) -> None:
    contract = _verify_contract(preregister_path)
    if out_dir.exists():
        raise FileExistsError(f"INNER output root must be absent: {out_dir}")
    out_dir.mkdir(parents=True)
    prereg_copy, prereg_sidecar = _copy_preregister(preregister_path, out_dir)
    recipe_spec = contract["data_contract"]["locked_recipe"]
    recipe_path = PROJECT_DIR / recipe_spec["path"]
    recipe_identity = _verify_identity(recipe_path, recipe_spec, "locked recipe")
    features, weather_evidence, feature_names = _read_weather_prefix(artifact_root, contract)
    taxonomy, subsets, subset_audit = _taxonomy_lock(feature_names, contract)
    taxonomy_path = _write_parquet(taxonomy, out_dir / "taxonomy/feature_taxonomy.parquet", index=False)
    folds = _folds(contract)
    source_lock = _write_json(
        out_dir / "source_and_input_lock.json",
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister": describe_file(prereg_copy),
            "preregister_sidecar": describe_file(prereg_sidecar),
            "runner_source": describe_file(Path(__file__).resolve()),
            "helper_source": describe_file(PROJECT_DIR / "src/virtual_feature_exit.py"),
            "metric_source": describe_file(PROJECT_DIR / "src/metric.py"),
            "recipe": recipe_identity,
            "weather": weather_evidence,
            "taxonomy": describe_file(taxonomy_path),
            "subsets": subset_audit,
            "folds": {group: [fold.as_dict() for fold in values] for group, values in folds.items()},
            "created_before_label_prefix_parse_or_model_fit": True,
            "2024_label_bytes_read": 0,
        },
    )
    labels, label_evidence = _read_bounded_labels(
        raw_dir / "train/train_labels.csv", contract
    )
    params = _model_parameters(contract)
    rows: list[pd.DataFrame] = []
    training: dict[str, Any] = {}
    model_paths: list[Path] = []
    for group in TARGET_COLS:
        for fold in folds[group]:
            fold_predictions: dict[str, np.ndarray] = {}
            fold_index: pd.DatetimeIndex | None = None
            for variant in VARIANTS:
                print(f"INNER fit {group} {fold.name} {variant}", flush=True)
                model, prediction, metadata, valid_index = _fit_one(
                    features[group],
                    labels[group],
                    fold,
                    subsets[variant],
                    CAPACITY_KWH[group],
                    params,
                )
                if fold_index is not None and not fold_index.equals(valid_index):
                    raise AssertionError("variant validation indexes differ")
                fold_index = valid_index
                model_path = out_dir / "inner/models" / group / fold.name / f"{variant}.joblib"
                _write_joblib(model, model_path)
                reloaded = joblib.load(model_path)
                repeated = np.asarray(
                    reloaded.predict(features[group].loc[valid_index, list(subsets[variant])]),
                    dtype=np.float64,
                )
                if not np.array_equal(repeated, prediction):
                    raise AssertionError("saved/reloaded LightGBM prediction changed")
                metadata["model"] = describe_file(model_path)
                metadata["reload_prediction_bit_exact"] = True
                training[f"{group}::{fold.name}::{variant}"] = metadata
                fold_predictions[variant] = prediction
                model_paths.append(model_path)
                del reloaded, model
            assert fold_index is not None
            part = pd.DataFrame(fold_predictions, index=fold_index)
            part.insert(0, "fold", fold.name)
            part.insert(0, "group", group)
            part.index.name = "forecast_kst_dtm"
            rows.append(part.reset_index())
    oof = pd.concat(rows, axis=0, ignore_index=True)
    if len(oof) != sum(
        len(pd.date_range(fold.valid_start, fold.valid_end, freq="h"))
        for values in folds.values() for fold in values
    ):
        raise AssertionError("INNER OOF row count changed")
    oof_path = _write_parquet(oof, out_dir / "inner/inner_oof_predictions.parquet", index=False)
    training_path = _write_json(out_dir / "inner/training_and_reload_checks.json", training)
    prescore_path = _write_json(
        out_dir / "inner/inner_prescore_lock.json",
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "source_and_input_lock": describe_file(source_lock),
            "label_prefix_physical_evidence": label_evidence,
            "oof_predictions": describe_file(oof_path),
            "training_and_reload_checks": describe_file(training_path),
            "model_count": len(model_paths),
            "models": [describe_file(path) for path in model_paths],
            "created_before_any_inner_metric": True,
        },
    )
    results = _score_inner(oof, labels)
    results["prescore_lock"] = describe_file(prescore_path)
    results_path = _write_json(out_dir / "inner/inner_direct_scores.json", results)
    _write_json(
        out_dir / "inner/manifest.json",
        {
            "schema_version": 1,
            "artifact_type": "causal_feature_exit_inner",
            "created_utc": utc_now(),
            "runtime": {"packages": package_versions(), "git": git_state(PROJECT_DIR)},
            "preregister_sha256": PREREGISTER_SHA256,
            "label_physical_scope": label_evidence,
            "outputs": [
                describe_file(path)
                for path in (source_lock, taxonomy_path, oof_path, training_path, prescore_path, results_path)
            ],
            "model_count": len(model_paths),
            "leaderboard_score_claimed": False,
            "2024_labels_read": False,
            "2025_data_read": False,
        },
    )


def _assert_snapshot(snapshot: Mapping[str, Any]) -> Path:
    path = Path(str(snapshot["path"]))
    observed = describe_file(path)
    if (
        observed["size_bytes"] != int(snapshot["size_bytes"])
        or observed["sha256"] != str(snapshot["sha256"])
    ):
        raise AssertionError(f"locked artifact changed: {path}")
    return path


def _load_locked_inner(out_dir: Path) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    prereg_copy = out_dir / "preregister.json"
    if sha256_file(prereg_copy) != PREREGISTER_SHA256:
        raise AssertionError("copied preregistration changed")
    prescore_path = out_dir / "inner/inner_prescore_lock.json"
    score_path = out_dir / "inner/inner_direct_scores.json"
    manifest_path = out_dir / "inner/manifest.json"
    prescore = json.loads(prescore_path.read_text(encoding="utf-8"))
    scores = json.loads(score_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    score_snapshots = [
        row
        for row in manifest["outputs"]
        if Path(str(row["path"])).name == score_path.name
    ]
    if len(score_snapshots) != 1:
        raise AssertionError("INNER manifest does not uniquely bind direct scores")
    _assert_snapshot(score_snapshots[0])
    oof_path = _assert_snapshot(prescore["oof_predictions"])
    if scores["prescore_lock"]["sha256"] != sha256_file(prescore_path):
        raise AssertionError("INNER score does not reference the current prescore lock")
    oof = pd.read_parquet(oof_path, engine="pyarrow")
    required = ("forecast_kst_dtm", "group", "fold", *VARIANTS)
    if tuple(oof.columns) != required:
        raise AssertionError(f"INNER OOF columns changed: {tuple(oof.columns)}")
    oof["forecast_kst_dtm"] = pd.to_datetime(oof["forecast_kst_dtm"], errors="raise")
    if oof.duplicated(["group", "fold", "forecast_kst_dtm"]).any():
        raise AssertionError("INNER OOF keys are duplicated")
    if not np.isfinite(oof.loc[:, list(VARIANTS)].to_numpy(dtype=np.float64)).all():
        raise AssertionError("INNER OOF contains non-finite predictions")
    return oof, scores, prescore


def _bootstrap_inner_deltas(
    oof: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    batch_size: int = 100,
) -> np.ndarray:
    """Return exact official macro deltas under registered common block draws."""

    if replicates <= 0 or batch_size <= 0:
        raise ValueError("replicates and batch_size must be positive")
    variant_count = len(VARIANTS)
    prepared: dict[tuple[str, str], dict[str, Any]] = {}
    draw_lengths: dict[str, int] = {}
    for (group, fold), rows in oof.groupby(["group", "fold"], sort=False):
        rows = rows.sort_values("forecast_kst_dtm")
        times = pd.DatetimeIndex(rows["forecast_kst_dtm"])
        if len(rows) % 24:
            raise AssertionError(f"{group}/{fold} is not whole operating days")
        capacity = float(CAPACITY_KWH[str(group)])
        actual = labels.loc[times, str(group)].to_numpy(dtype=np.float64)
        prediction = rows.loc[:, list(VARIANTS)].to_numpy(dtype=np.float64) * capacity
        valid = np.isfinite(actual) & (actual >= 0.10 * capacity)
        error = np.zeros((len(rows), variant_count), dtype=np.float64)
        settlement = np.zeros_like(error)
        error[valid] = np.abs(prediction[valid] - actual[valid, None]) / capacity
        unit = np.select(
            [error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0
        )
        settlement[valid] = actual[valid, None] * unit[valid]
        prepared[(str(group), str(fold))] = {
            "valid": valid.astype(np.float64),
            "actual": np.where(valid, actual, 0.0),
            "error": error,
            "settlement": settlement,
            "n": len(rows),
        }
        if str(fold) in draw_lengths and draw_lengths[str(fold)] != len(rows):
            raise AssertionError("common-draw fold lengths differ across groups")
        draw_lengths[str(fold)] = len(rows)

    rng = np.random.default_rng(seed)
    candidate_count = len(EXIT_VARIANTS)
    output = np.empty((replicates, candidate_count), dtype=np.float64)
    offsets = np.arange(BLOCK_HOURS, dtype=np.int64)
    for batch_start in range(0, replicates, batch_size):
        count = min(batch_size, replicates - batch_start)
        common_indices: dict[str, np.ndarray] = {}
        for fold, n_rows in draw_lengths.items():
            day_count = n_rows // 24
            block_count = int(np.ceil(n_rows / BLOCK_HOURS))
            starts = rng.integers(0, day_count, size=(count, block_count), endpoint=False) * 24
            sampled = (starts[:, :, None] + offsets[None, None, :]) % n_rows
            common_indices[fold] = sampled.reshape(count, -1)[:, :n_rows]
        group_accumulator: dict[str, dict[str, np.ndarray]] = {}
        for (group, fold), values in prepared.items():
            sampled = common_indices[fold]
            accumulator = group_accumulator.setdefault(
                group,
                {
                    "count": np.zeros(count, dtype=np.float64),
                    "actual": np.zeros(count, dtype=np.float64),
                    "error": np.zeros((count, variant_count), dtype=np.float64),
                    "settlement": np.zeros((count, variant_count), dtype=np.float64),
                },
            )
            accumulator["count"] += values["valid"][sampled].sum(axis=1)
            accumulator["actual"] += values["actual"][sampled].sum(axis=1)
            accumulator["error"] += values["error"][sampled].sum(axis=1)
            accumulator["settlement"] += values["settlement"][sampled].sum(axis=1)
        macro_n = np.zeros((count, variant_count), dtype=np.float64)
        macro_f = np.zeros_like(macro_n)
        for group in TARGET_COLS:
            accumulator = group_accumulator[group]
            if np.any(accumulator["count"] <= 0) or np.any(accumulator["actual"] <= 0):
                raise AssertionError("bootstrap draw has no eligible rows")
            macro_n += 1.0 - accumulator["error"] / accumulator["count"][:, None]
            macro_f += accumulator["settlement"] / (4.0 * accumulator["actual"][:, None])
        macro_n /= len(TARGET_COLS)
        macro_f /= len(TARGET_COLS)
        macro_total = 0.5 * (macro_n + macro_f)
        output[batch_start : batch_start + count] = (
            macro_total[:, 1:] - macro_total[:, [0]]
        )
    return output


def _quantile(values: np.ndarray, probability: float) -> float:
    # The frozen protocol states the ordinary empirical NumPy quantile.
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def _stress_ledger(
    direct_scores: Mapping[str, Any], draws: np.ndarray, contract: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    observed = np.asarray(
        [direct_scores["variants"][variant]["delta_score"] for variant in EXIT_VARIANTS],
        dtype=np.float64,
    )
    if draws.shape != (BOOTSTRAP_REPLICATES, len(EXIT_VARIANTS)):
        raise AssertionError(f"bootstrap shape changed: {draws.shape}")
    maximum_centered = np.max(draws - observed[None, :], axis=1)
    c90 = _quantile(maximum_centered, MAX_T_QUANTILE)
    lcb = observed - c90
    candidate_ledger = contract["candidate_ledger_fixed_before_labels"]
    keep_counts = {row["id"]: int(row["keep_count"]) for row in candidate_ledger}
    records: dict[str, Any] = {}
    passing: list[str] = []
    trajectory: list[dict[str, Any]] = [
        {
            "step": 0,
            "candidate": "control_all",
            "local_virtual_score": float(direct_scores["control_inner_official_score"]),
            "risk_adjusted_increment": 0.0,
            "promoted": False,
        }
    ]
    for order, variant in enumerate(EXIT_VARIANTS, start=1):
        direct = direct_scores["variants"][variant]
        score = risk_adjusted_virtual_score(
            baseline_score=float(direct_scores["control_inner_official_score"]),
            raw_delta=float(direct["delta_score"]),
            simultaneous_lcb=float(lcb[order - 1]),
            cell_deltas=direct["cell_deltas"],
            delta_ficr=float(direct["delta_F"]),
            delta_one_minus_nmae=float(direct["delta_N"]),
        )
        gates = {
            "R_ge_0.003": score.risk_adjusted_increment >= 0.003,
            "delta_F_ge_0.005": score.delta_ficr >= 0.005,
            "delta_N_ge_minus_0.0002": score.delta_one_minus_nmae >= -0.0002,
            "at_least_8_of_10_cells_positive": int(direct["positive_cell_count"]) >= 8,
            "worst_cell_ge_minus_0.001": score.minimum_cell_delta >= -0.001,
        }
        promoted = all(gates.values())
        if promoted:
            passing.append(variant)
        record = {
            **score.as_dict(),
            "bootstrap_mean_delta": float(np.mean(draws[:, order - 1])),
            "bootstrap_std_delta": float(np.std(draws[:, order - 1], ddof=1)),
            "bootstrap_probability_delta_positive": float(np.mean(draws[:, order - 1] > 0.0)),
            "bootstrap_percentile_05": _quantile(draws[:, order - 1], 0.05),
            "bootstrap_percentile_95": _quantile(draws[:, order - 1], 0.95),
            "positive_cell_count": int(direct["positive_cell_count"]),
            "keep_count": keep_counts[variant],
            "gates": gates,
            "promoted": promoted,
            "ledger_order": order,
        }
        records[variant] = record
        trajectory.append(
            {
                "step": order,
                "candidate": variant,
                "local_virtual_score": score.local_virtual_score,
                "risk_adjusted_increment": score.risk_adjusted_increment,
                "promoted": promoted,
            }
        )
    selected: str | None = None
    if passing:
        order_map = {variant: position for position, variant in enumerate(EXIT_VARIANTS)}
        selected = max(
            passing,
            key=lambda variant: (
                records[variant]["risk_adjusted_increment"],
                records[variant]["minimum_cell_delta"],
                -records[variant]["keep_count"],
                -order_map[variant],
            ),
        )
    # The Public anchor is appended for display only after every gate and the
    # selector have finished.  It is absent from every selection expression.
    report_only_anchor = float(contract["score_claim"]["report_only_public_anchor"])
    for row in trajectory[1:]:
        row["public_anchor_display_only"] = (
            report_only_anchor + float(row["risk_adjusted_increment"])
        )
    ledger = {
        "schema_version": 1,
        "preregister_sha256": PREREGISTER_SHA256,
        "method": "frozen unstudentized one-sided simultaneous max-T",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "block_hours": BLOCK_HOURS,
        "common_draws_across_candidates_and_overlapping_g1_g2_folds": True,
        "observed_delta_order": list(EXIT_VARIANTS),
        "observed_deltas": observed.tolist(),
        "max_centered_quantile_probability": MAX_T_QUANTILE,
        "max_centered_c90": c90,
        "candidates": records,
        "passing_candidates": passing,
        "selected_for_outer": selected,
        "leaderboard_score_claimed": False,
    }
    return ledger, {"schema_version": 1, "trajectory": trajectory, "selected_for_outer": selected}


def _run_stress(
    *, raw_dir: Path, out_dir: Path, preregister_path: Path
) -> None:
    contract = _verify_contract(preregister_path)
    stress_dir = out_dir / "stress"
    if stress_dir.exists():
        raise FileExistsError(f"STRESS output directory must be absent: {stress_dir}")
    oof, direct_scores, prescore = _load_locked_inner(out_dir)
    labels, label_evidence = _read_bounded_labels(
        raw_dir / "train/train_labels.csv", contract
    )
    recomputed = _score_inner(oof, labels)
    for key in (
        "control_inner_official_score",
        "control_direct_macro",
        "variants",
        "registered_cells",
    ):
        if recomputed[key] != direct_scores[key]:
            raise AssertionError(f"locked INNER score changed: {key}")
    stress_dir.mkdir(parents=True)
    draws = _bootstrap_inner_deltas(oof, labels)
    draws_path = _write_npz(
        stress_dir / "bootstrap_draws.npz",
        delta_score=draws,
        candidate_order=np.asarray(EXIT_VARIANTS),
    )
    # Selection consumes only metrics freshly recomputed from the hash-locked
    # OOF and bounded label prefix.  The stored score JSON is audit-only here.
    ledger, trajectory = _stress_ledger(recomputed, draws, contract)
    ledger["draws"] = describe_file(draws_path)
    ledger["inner_prescore_lock"] = describe_file(out_dir / "inner/inner_prescore_lock.json")
    ledger["label_prefix_physical_evidence"] = label_evidence
    ledger_path = _write_json(stress_dir / "bootstrap_draw_and_candidate_ledger.json", ledger)
    trajectory_path = _write_json(stress_dir / "virtual_score_trajectory.json", trajectory)
    promotion_path = _write_json(
        stress_dir / "inner_promotion_lock.json",
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "inner_prescore_lock": describe_file(out_dir / "inner/inner_prescore_lock.json"),
            "inner_direct_scores": describe_file(out_dir / "inner/inner_direct_scores.json"),
            "bootstrap_ledger": describe_file(ledger_path),
            "selected_for_outer": ledger["selected_for_outer"],
            "selection_uses_public_anchor": False,
            "2024_labels_read": False,
            "selection_is_immutable": True,
        },
    )
    _write_json(
        stress_dir / "manifest.json",
        {
            "schema_version": 1,
            "artifact_type": "causal_feature_exit_inner_max_t_stress",
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "inputs": {
                "inner_prescore": prescore,
                "inner_scores": describe_file(out_dir / "inner/inner_direct_scores.json"),
            },
            "outputs": [describe_file(path) for path in (draws_path, ledger_path, trajectory_path, promotion_path)],
            "selected_for_outer": ledger["selected_for_outer"],
            "leaderboard_score_claimed": False,
            "2024_labels_read": False,
        },
    )


def _outer_folds() -> dict[str, CausalFold]:
    return {
        "kpx_group_1": CausalFold(
            "outer_2022_to_2023",
            pd.Timestamp("2022-01-01 01:00:00"),
            pd.Timestamp("2023-01-01 00:00:00"),
            pd.Timestamp("2023-01-01 01:00:00"),
            pd.Timestamp("2024-01-01 00:00:00"),
        ),
        "kpx_group_2": CausalFold(
            "outer_2022_to_2023",
            pd.Timestamp("2022-01-01 01:00:00"),
            pd.Timestamp("2023-01-01 00:00:00"),
            pd.Timestamp("2023-01-01 01:00:00"),
            pd.Timestamp("2024-01-01 00:00:00"),
        ),
        "kpx_group_3": CausalFold(
            "outer_2023h1_to_2023h2",
            pd.Timestamp("2023-01-01 01:00:00"),
            pd.Timestamp("2023-07-01 00:00:00"),
            pd.Timestamp("2023-07-01 01:00:00"),
            pd.Timestamp("2024-01-01 00:00:00"),
        ),
    }


def _read_prediction_component(
    path: Path, index: pd.DatetimeIndex, column: str
) -> pd.Series:
    frame = pd.read_parquet(path, engine="pyarrow")
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if not frame.index.equals(index) or column not in frame.columns:
        raise AssertionError(f"component index/column changed: {path}/{column}")
    values = frame[column].astype(np.float64)
    if not np.isfinite(values.to_numpy()).all():
        raise AssertionError(f"component contains non-finite values: {path}/{column}")
    return values


def _load_locked_components(
    artifact_root: Path,
    group: str,
    index: pd.DatetimeIndex,
    contract: Mapping[str, Any] | None = None,
) -> tuple[dict[str, pd.Series], dict[str, Any]]:
    components: dict[str, pd.Series] = {}
    inputs: dict[str, Any] = {}
    if group in TARGET_COLS[:2]:
        for name in COMPONENT_NAMES:
            path = PROJECT_DIR / G12_COMPONENT_PATHS[name]
            if not path.is_file():
                path = artifact_root / Path(G12_COMPONENT_PATHS[name]).relative_to("artifacts")
            components[name] = _read_prediction_component(path, index, group)
            inputs[name] = describe_file(path)
    else:
        path = PROJECT_DIR / G3_COMPONENT_PATH
        if not path.is_file():
            path = artifact_root / Path(G3_COMPONENT_PATH).relative_to("artifacts")
        if contract is not None:
            _verify_identity(
                path,
                contract["data_contract"]["baseline_2023_g3_components"],
                "pinned G3 six-component frame",
            )
        frame = pd.read_parquet(path, engine="pyarrow")
        frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
        if not frame.index.equals(index):
            raise AssertionError("G3 component index changed")
        for name in COMPONENT_NAMES:
            column = G3_COMPONENT_COLUMNS[name]
            if column not in frame.columns:
                raise AssertionError(f"G3 component column missing: {column}")
            components[name] = frame[column].astype(np.float64)
        if not all(np.isfinite(series.to_numpy()).all() for series in components.values()):
            raise AssertionError("G3 component contains non-finite values")
        inputs["six_component_frame"] = describe_file(path)
    return components, inputs


def _assert_bit_exact(actual: np.ndarray, expected: np.ndarray, context: str) -> None:
    left = np.asarray(actual, dtype=np.float64)
    right = np.asarray(expected, dtype=np.float64)
    if not np.array_equal(left, right):
        maximum = float(np.max(np.abs(left - right))) if left.shape == right.shape else None
        raise AssertionError(f"{context} is not bit exact; max_abs={maximum}")


def _verify_outer_anchors(
    contract: Mapping[str, Any], artifact_root: Path
) -> dict[str, Any]:
    """Verify every preregistered OUTER anchor before labels or fits."""

    g12_spec = contract["data_contract"]["baseline_2023_g12"]
    g12_path = PROJECT_DIR / str(g12_spec["path"])
    if not g12_path.is_file():
        g12_path = artifact_root / Path(str(g12_spec["path"])).relative_to("artifacts")
    g3_spec = contract["data_contract"]["baseline_2023_g3_components"]
    g3_path = PROJECT_DIR / str(g3_spec["path"])
    if not g3_path.is_file():
        g3_path = artifact_root / Path(str(g3_spec["path"])).relative_to("artifacts")
    recipe_spec = contract["data_contract"]["locked_recipe"]
    recipe_path = PROJECT_DIR / str(recipe_spec["path"])
    return {
        "locked_recipe": _verify_identity(recipe_path, recipe_spec, "locked-v3 recipe"),
        "baseline_2023_g12": _verify_identity(g12_path, g12_spec, "pinned G12 baseline"),
        "baseline_2023_g3_components": _verify_identity(
            g3_path, g3_spec, "pinned G3 component frame"
        ),
        "verified_before_outer_label_prefix_parse_or_model_fit": True,
    }


def _load_locked_recipe(contract: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = contract["data_contract"]["locked_recipe"]
    path = PROJECT_DIR / str(spec["path"])
    identity = _verify_identity(path, spec, "locked-v3 recipe")
    recipe = json.loads(path.read_text(encoding="utf-8"))
    registered = _model_parameters(contract)
    q07 = recipe["models"]["lgb_q07"]
    expected = dict(q07["params"])
    expected.update(
        objective="quantile",
        alpha=float(q07["alpha"]),
        random_state=int(recipe["seed"]),
        n_jobs=int(recipe["n_jobs"]),
    )
    if expected != registered:
        raise AssertionError("registered q07 params differ from locked-v3 recipe")
    return recipe, identity


def _year_segments_2023() -> dict[str, pd.DatetimeIndex]:
    boundaries = {
        "Q1": ("2023-01-01 01:00:00", "2023-04-01 00:00:00"),
        "Q2": ("2023-04-01 01:00:00", "2023-07-01 00:00:00"),
        "Q3": ("2023-07-01 01:00:00", "2023-10-01 00:00:00"),
        "Q4": ("2023-10-01 01:00:00", "2024-01-01 00:00:00"),
        "H1": ("2023-01-01 01:00:00", "2023-07-01 00:00:00"),
        "H2": ("2023-07-01 01:00:00", "2024-01-01 00:00:00"),
        "FULL": ("2023-01-01 01:00:00", "2024-01-01 00:00:00"),
    }
    return {
        name: pd.date_range(start, end, freq="h", name="forecast_kst_dtm")
        for name, (start, end) in boundaries.items()
    }


def _macro_metrics(
    labels: pd.DataFrame,
    prediction: Mapping[str, pd.Series],
    *,
    groups: Sequence[str],
    index: pd.DatetimeIndex,
) -> dict[str, Any]:
    by_group: dict[str, Any] = {}
    for group in groups:
        series = prediction[group].reindex(index)
        if series.isna().any():
            raise AssertionError(f"{group} prediction does not cover metric index")
        metric = group_metrics(
            labels.loc[index, group], series, CAPACITY_KWH[group], group_name=group
        )
        by_group[group] = metric.as_dict()
    one_minus_nmae = float(np.mean([row["one_minus_nmae"] for row in by_group.values()]))
    ficr = float(np.mean([row["ficr"] for row in by_group.values()]))
    return {
        "total_score": 0.5 * (one_minus_nmae + ficr),
        "one_minus_nmae": one_minus_nmae,
        "ficr": ficr,
        "by_group": by_group,
    }


def _metric_comparison(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "baseline": dict(baseline),
        "candidate": dict(candidate),
        "delta": {
            "total_score": float(candidate["total_score"] - baseline["total_score"]),
            "delta_N": float(candidate["one_minus_nmae"] - baseline["one_minus_nmae"]),
            "delta_F": float(candidate["ficr"] - baseline["ficr"]),
        },
    }


def _outer_block_bootstrap(
    labels: pd.DataFrame,
    baseline: Mapping[str, pd.Series],
    candidate: Mapping[str, pd.Series],
    index: pd.DatetimeIndex,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    batch_size: int = 100,
) -> np.ndarray:
    if len(index) % 24:
        raise AssertionError("OUTER H2 is not whole operating days")
    prepared: dict[str, dict[str, np.ndarray]] = {}
    for group in TARGET_COLS:
        capacity = float(CAPACITY_KWH[group])
        actual = labels.loc[index, group].to_numpy(dtype=np.float64)
        forecasts = np.column_stack(
            (baseline[group].loc[index].to_numpy(), candidate[group].loc[index].to_numpy())
        )
        valid = np.isfinite(actual) & (actual >= 0.10 * capacity)
        error = np.zeros((len(index), 2), dtype=np.float64)
        settlement = np.zeros_like(error)
        error[valid] = np.abs(forecasts[valid] - actual[valid, None]) / capacity
        unit = np.select([error <= 0.06, error <= 0.08], [4.0, 3.0], default=0.0)
        settlement[valid] = actual[valid, None] * unit[valid]
        prepared[group] = {
            "valid": valid.astype(np.float64),
            "actual": np.where(valid, actual, 0.0),
            "error": error,
            "settlement": settlement,
        }
    rng = np.random.default_rng(seed)
    result = np.empty(replicates, dtype=np.float64)
    day_count = len(index) // 24
    block_count = int(np.ceil(len(index) / BLOCK_HOURS))
    offsets = np.arange(BLOCK_HOURS, dtype=np.int64)
    for start in range(0, replicates, batch_size):
        count = min(batch_size, replicates - start)
        block_starts = rng.integers(0, day_count, size=(count, block_count), endpoint=False) * 24
        sampled = (block_starts[:, :, None] + offsets[None, None, :]) % len(index)
        sampled = sampled.reshape(count, -1)[:, : len(index)]
        macro_n = np.zeros((count, 2), dtype=np.float64)
        macro_f = np.zeros_like(macro_n)
        for group in TARGET_COLS:
            values = prepared[group]
            eligible_count = values["valid"][sampled].sum(axis=1)
            actual_sum = values["actual"][sampled].sum(axis=1)
            error_sum = values["error"][sampled].sum(axis=1)
            settlement_sum = values["settlement"][sampled].sum(axis=1)
            macro_n += 1.0 - error_sum / eligible_count[:, None]
            macro_f += settlement_sum / (4.0 * actual_sum[:, None])
        total = 0.5 * ((macro_n / 3.0) + (macro_f / 3.0))
        result[start : start + count] = total[:, 1] - total[:, 0]
    return result


def _score_outer(
    labels: pd.DataFrame,
    control_direct: Mapping[str, pd.Series],
    exit_direct: Mapping[str, pd.Series],
    baseline: Mapping[str, pd.Series],
    replacement: Mapping[str, pd.Series],
    bootstrap_draws: np.ndarray,
) -> dict[str, Any]:
    segments = _year_segments_2023()
    direct: dict[str, Any] = {}
    for group in TARGET_COLS:
        names = ("FULL", "H1", "H2", "Q1", "Q2", "Q3", "Q4") if group in TARGET_COLS[:2] else ("H2", "Q3", "Q4")
        direct[group] = {}
        for name in names:
            base_metric = group_metrics(
                labels.loc[segments[name], group],
                control_direct[group].loc[segments[name]],
                CAPACITY_KWH[group],
                group_name=group,
            ).as_dict()
            exit_metric = group_metrics(
                labels.loc[segments[name], group],
                exit_direct[group].loc[segments[name]],
                CAPACITY_KWH[group],
                group_name=group,
            ).as_dict()
            base_macro = {
                "total_score": 0.5 * (base_metric["one_minus_nmae"] + base_metric["ficr"]),
                "one_minus_nmae": base_metric["one_minus_nmae"],
                "ficr": base_metric["ficr"],
                "by_group": {group: base_metric},
            }
            exit_macro = {
                "total_score": 0.5 * (exit_metric["one_minus_nmae"] + exit_metric["ficr"]),
                "one_minus_nmae": exit_metric["one_minus_nmae"],
                "ficr": exit_metric["ficr"],
                "by_group": {group: exit_metric},
            }
            direct[group][name] = _metric_comparison(base_macro, exit_macro)

    component: dict[str, Any] = {"group_quarters": {}, "macro_segments": {}}
    group_quarter_deltas: list[float] = []
    for group in TARGET_COLS:
        quarter_names = ("Q1", "Q2", "Q3", "Q4") if group in TARGET_COLS[:2] else ("Q3", "Q4")
        for quarter in quarter_names:
            base_metric = _macro_metrics(labels, baseline, groups=(group,), index=segments[quarter])
            cand_metric = _macro_metrics(labels, replacement, groups=(group,), index=segments[quarter])
            comparison = _metric_comparison(base_metric, cand_metric)
            component["group_quarters"][f"{group}::{quarter}"] = comparison
            group_quarter_deltas.append(comparison["delta"]["total_score"])
    for name, groups in (
        ("Q1_G12", TARGET_COLS[:2]),
        ("Q2_G12", TARGET_COLS[:2]),
        ("Q3_G123", TARGET_COLS),
        ("Q4_G123", TARGET_COLS),
        ("H2_G123", TARGET_COLS),
    ):
        segment = name.split("_")[0]
        base_metric = _macro_metrics(labels, baseline, groups=groups, index=segments[segment])
        cand_metric = _macro_metrics(labels, replacement, groups=groups, index=segments[segment])
        component["macro_segments"][name] = _metric_comparison(base_metric, cand_metric)
    for group in TARGET_COLS[:2]:
        for half in ("H1", "H2"):
            key = f"{group}::{half}"
            base_metric = _macro_metrics(labels, baseline, groups=(group,), index=segments[half])
            cand_metric = _macro_metrics(labels, replacement, groups=(group,), index=segments[half])
            component["macro_segments"][key] = _metric_comparison(base_metric, cand_metric)

    h2 = component["macro_segments"]["H2_G123"]["delta"]
    gates = {
        "H2_macro_delta_score_ge_0.005": h2["total_score"] >= 0.005,
        "H2_macro_delta_F_ge_0.008": h2["delta_F"] >= 0.008,
        "H2_macro_delta_N_ge_minus_0.0002": h2["delta_N"] >= -0.0002,
        "G1_and_G2_H1_and_H2_positive": all(
            component["macro_segments"][f"{group}::{half}"]["delta"]["total_score"] > 0.0
            for group in TARGET_COLS[:2] for half in ("H1", "H2")
        ),
        "Q1_Q2_G12_macro_positive": all(
            component["macro_segments"][f"{quarter}_G12"]["delta"]["total_score"] > 0.0
            for quarter in ("Q1", "Q2")
        ),
        "Q3_Q4_G123_macro_positive": all(
            component["macro_segments"][f"{quarter}_G123"]["delta"]["total_score"] > 0.0
            for quarter in ("Q3", "Q4")
        ),
        "worst_group_quarter_ge_minus_0.0005": min(group_quarter_deltas) >= -0.0005,
        "block_bootstrap_probability_delta_positive_ge_0.90": float(np.mean(bootstrap_draws > 0.0)) >= 0.90,
    }
    return {
        "schema_version": 1,
        "direct_control_vs_exit": direct,
        "locked_component_replacement": component,
        "worst_group_quarter": float(min(group_quarter_deltas)),
        "H2_block_bootstrap": {
            "replicates": len(bootstrap_draws),
            "seed": BOOTSTRAP_SEED,
            "block_hours": BLOCK_HOURS,
            "probability_delta_positive": float(np.mean(bootstrap_draws > 0.0)),
            "mean_delta": float(np.mean(bootstrap_draws)),
            "percentile_05": _quantile(bootstrap_draws, 0.05),
            "percentile_95": _quantile(bootstrap_draws, 0.95),
        },
        "gates": gates,
        "outer_passed": all(gates.values()),
        "no_reselection": True,
    }


def _load_promotion_lock(out_dir: Path) -> tuple[dict[str, Any], str | None]:
    path = out_dir / "stress/inner_promotion_lock.json"
    manifest_path = out_dir / "stress/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["preregister_sha256"] != PREREGISTER_SHA256:
        raise AssertionError("STRESS manifest preregistration changed")
    promotion_snapshots = [
        row for row in manifest["outputs"] if Path(str(row["path"])).name == path.name
    ]
    if len(promotion_snapshots) != 1:
        raise AssertionError("STRESS manifest does not uniquely bind promotion lock")
    _assert_snapshot(promotion_snapshots[0])
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock["preregister_sha256"] != PREREGISTER_SHA256:
        raise AssertionError("promotion lock preregistration changed")
    _assert_snapshot(lock["inner_prescore_lock"])
    _assert_snapshot(lock["inner_direct_scores"])
    ledger_path = _assert_snapshot(lock["bootstrap_ledger"])
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if ledger.get("preregister_sha256") != PREREGISTER_SHA256:
        raise AssertionError("bootstrap ledger preregistration changed")
    if tuple(ledger.get("observed_delta_order", ())) != EXIT_VARIANTS:
        raise AssertionError("bootstrap candidate order changed")
    if (
        int(ledger.get("replicates", -1)) != BOOTSTRAP_REPLICATES
        or int(ledger.get("seed", -1)) != BOOTSTRAP_SEED
    ):
        raise AssertionError("bootstrap protocol changed")
    if ledger["selected_for_outer"] != lock["selected_for_outer"]:
        raise AssertionError("promotion lock selection differs from bootstrap ledger")
    selected = lock["selected_for_outer"]
    if selected is not None and selected not in EXIT_VARIANTS:
        raise AssertionError("promotion lock selected an unregistered family")
    if selected is not None:
        record = ledger["candidates"].get(selected)
        if (
            not isinstance(record, Mapping)
            or not bool(record.get("promoted"))
            or selected not in ledger["passing_candidates"]
        ):
            raise AssertionError("selected family is not a promoted max-T ledger entry")
    return lock, selected


def _run_outer(
    *,
    raw_dir: Path,
    artifact_root: Path,
    out_dir: Path,
    preregister_path: Path,
) -> None:
    contract = _verify_contract(preregister_path)
    promotion_lock, selected = _load_promotion_lock(out_dir)
    outer_dir = out_dir / "outer"
    if outer_dir.exists():
        raise FileExistsError(f"OUTER output directory must be absent: {outer_dir}")
    outer_dir.mkdir(parents=True)
    if selected is None:
        decision_path = _write_json(
            outer_dir / "outer_lock_or_rejection.json",
            {
                "schema_version": 1,
                "status": "rejected_before_outer",
                "reason": "No feature-exit family passed every frozen INNER promotion gate.",
                "promotion_lock": describe_file(out_dir / "stress/inner_promotion_lock.json"),
                "label_bytes_read_by_outer": 0,
                "model_fits_by_outer": 0,
                "2024_labels_read": False,
            },
        )
        _write_json(
            outer_dir / "manifest.json",
            {
                "schema_version": 1,
                "artifact_type": "feature_exit_outer_terminal_no_candidate",
                "created_utc": utc_now(),
                "preregister_sha256": PREREGISTER_SHA256,
                "promotion_lock": describe_file(
                    out_dir / "stress/inner_promotion_lock.json"
                ),
                "outputs": [describe_file(decision_path)],
                "terminal": True,
                "selected_family": None,
                "label_bytes_read_by_outer": 0,
                "component_files_read_by_outer": 0,
                "model_fits_by_outer": 0,
                "2024_labels_read": False,
                "2025_data_read": False,
                "submission_created": False,
            },
        )
        return

    outer_anchor_lock = _write_json(
        outer_dir / "outer_anchor_lock.json",
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "promotion_lock": describe_file(out_dir / "stress/inner_promotion_lock.json"),
            "selected_family": selected,
            "anchors": _verify_outer_anchors(contract, artifact_root),
            "created_before_outer_label_prefix_parse_weather_read_or_model_fit": True,
        },
    )
    labels, label_evidence = _read_bounded_labels(
        raw_dir / "train/train_labels.csv", contract
    )
    features, weather_evidence, names = _read_weather_prefix(artifact_root, contract)
    _, subsets, subset_audit = _taxonomy_lock(names, contract)
    recipe, recipe_identity = _load_locked_recipe(contract)
    params = _model_parameters(contract)
    folds = _outer_folds()
    control_direct: dict[str, pd.Series] = {}
    exit_direct: dict[str, pd.Series] = {}
    baseline: dict[str, pd.Series] = {}
    replacement: dict[str, pd.Series] = {}
    prediction_parts: list[pd.DataFrame] = []
    model_paths: list[Path] = []
    audits: dict[str, Any] = {}
    component_inputs: dict[str, Any] = {}
    for group in TARGET_COLS:
        outputs: dict[str, np.ndarray] = {}
        valid_index: pd.DatetimeIndex | None = None
        training: dict[str, Any] = {}
        for variant in ("control_all", selected):
            print(f"OUTER fit {group} {variant}", flush=True)
            model, prediction_cf, metadata, index = _fit_one(
                features[group], labels[group], folds[group], subsets[variant], CAPACITY_KWH[group], params
            )
            if valid_index is not None and not valid_index.equals(index):
                raise AssertionError("OUTER variant validation indexes differ")
            valid_index = index
            model_path = outer_dir / "models" / group / f"{variant}.joblib"
            _write_joblib(model, model_path)
            reloaded = joblib.load(model_path)
            repeated = np.asarray(
                reloaded.predict(features[group].loc[index, list(subsets[variant])]), dtype=np.float64
            )
            if not np.array_equal(repeated, prediction_cf):
                raise AssertionError("OUTER saved/reloaded prediction changed")
            metadata["model"] = describe_file(model_path)
            metadata["reload_prediction_bit_exact"] = True
            training[variant] = metadata
            outputs[variant] = prediction_cf * CAPACITY_KWH[group]
            model_paths.append(model_path)
            del model, reloaded
        assert valid_index is not None
        components, inputs = _load_locked_components(
            artifact_root, group, valid_index, contract
        )
        component_inputs[group] = inputs
        stored_q07 = components["lgb_q07"].to_numpy(dtype=np.float64)
        control_diff = np.abs(outputs["control_all"] - stored_q07)
        max_control_diff = float(np.max(control_diff))
        control_bit_exact = bool(np.array_equal(outputs["control_all"], stored_q07))
        _assert_bit_exact(
            outputs["control_all"], stored_q07, f"{group} refit locked q07 component"
        )
        reconstructed = assemble_locked_group(
            components,
            group=group,
            capacity_kwh=CAPACITY_KWH[group],
            ensemble=recipe["ensemble"],
        )
        if group in TARGET_COLS[:2]:
            base_path = PROJECT_DIR / "artifacts/oof/dev2023_locked_v3.parquet"
            if not base_path.is_file():
                base_path = artifact_root / "oof/dev2023_locked_v3.parquet"
            expected_baseline = _read_prediction_component(base_path, valid_index, group)
            _verify_identity(
                base_path,
                contract["data_contract"]["baseline_2023_g12"],
                "pinned G12 baseline",
            )
            reconstruction_diff = float(
                np.max(np.abs(reconstructed - expected_baseline.to_numpy(dtype=np.float64)))
            )
            _assert_bit_exact(
                reconstructed,
                expected_baseline.to_numpy(dtype=np.float64),
                f"{group} locked baseline reconstruction",
            )
            component_inputs[group]["locked_baseline"] = describe_file(base_path)
            reconstruction_audit = {
                "locked_baseline_reconstruction_bit_exact": True,
                "locked_baseline_reconstruction_max_abs_kwh": reconstruction_diff,
            }
        else:
            reconstruction_audit = {
                "pinned_g3_six_component_frame_verified": True,
                "locked_recipe_assembly_from_pinned_components": True,
                "external_baseline_bit_exact_comparison_applicable": False,
            }
        replaced_components = dict(components)
        replaced_components["lgb_q07"] = outputs[selected]
        replaced = assemble_locked_group(
            replaced_components,
            group=group,
            capacity_kwh=CAPACITY_KWH[group],
            ensemble=recipe["ensemble"],
        )
        control_direct[group] = pd.Series(outputs["control_all"], index=valid_index, name=group)
        exit_direct[group] = pd.Series(outputs[selected], index=valid_index, name=group)
        baseline[group] = pd.Series(reconstructed, index=valid_index, name=group)
        replacement[group] = pd.Series(replaced, index=valid_index, name=group)
        audits[group] = {
            "training": training,
            "control_component_reproduction_bit_exact": control_bit_exact,
            "control_component_reproduction_max_abs_kwh": max_control_diff,
            **reconstruction_audit,
            "replacement": "lgb_q07 component only; 1:1; no blend",
        }
        part = pd.DataFrame(
            {
                "control_q07_kwh": outputs["control_all"],
                "exit_q07_kwh": outputs[selected],
                "locked_v3_kwh": reconstructed,
                "q07_replacement_v3_kwh": replaced,
            },
            index=valid_index,
        )
        part.insert(0, "group", group)
        part.index.name = "forecast_kst_dtm"
        prediction_parts.append(part.reset_index())

    predictions = pd.concat(prediction_parts, ignore_index=True)
    prediction_path = _write_parquet(
        predictions, outer_dir / "outer_predictions.parquet", index=False
    )
    audit_path = _write_json(outer_dir / "model_and_component_audit.json", audits)
    prescore_path = _write_json(
        outer_dir / "outer_prescore_lock.json",
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "promotion_lock": describe_file(out_dir / "stress/inner_promotion_lock.json"),
            "outer_anchor_lock": describe_file(outer_anchor_lock),
            "selected_family": selected,
            "recipe": recipe_identity,
            "weather": weather_evidence,
            "subset": subset_audit[selected],
            "label_prefix_physical_evidence": label_evidence,
            "component_inputs": component_inputs,
            "models": [describe_file(path) for path in model_paths],
            "predictions": describe_file(prediction_path),
            "audit": describe_file(audit_path),
            "created_before_any_outer_metric": True,
            "2024_labels_read": False,
        },
    )
    h2 = _year_segments_2023()["H2"]
    bootstrap_draws = _outer_block_bootstrap(labels, baseline, replacement, h2)
    bootstrap_path = _write_npz(
        outer_dir / "outer_h2_block_bootstrap_draws.npz", delta_score=bootstrap_draws
    )
    results = _score_outer(
        labels, control_direct, exit_direct, baseline, replacement, bootstrap_draws
    )
    results["selected_family"] = selected
    results["outer_prescore_lock"] = describe_file(prescore_path)
    results["bootstrap_draws"] = describe_file(bootstrap_path)
    results_path = _write_json(outer_dir / "outer_results.json", results)
    status = "outer_passed" if results["outer_passed"] else "outer_rejected"
    decision_path = _write_json(
        outer_dir / "outer_lock_or_rejection.json",
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "status": status,
            "selected_family": selected,
            "results": describe_file(results_path),
            "outer_passed": results["outer_passed"],
            "no_rescue_or_reselection": True,
            "2024_labels_read": False,
            "2025_data_or_submission_read": False,
        },
    )
    _write_json(
        outer_dir / "manifest.json",
        {
            "schema_version": 1,
            "artifact_type": "locked_v3_q07_exact_component_replacement_outer",
            "created_utc": utc_now(),
            "preregister_sha256": PREREGISTER_SHA256,
            "promotion_lock": promotion_lock,
            "outputs": [
                describe_file(path)
                for path in (
                    outer_anchor_lock,
                    prediction_path,
                    audit_path,
                    prescore_path,
                    bootstrap_path,
                    results_path,
                    decision_path,
                )
            ],
            "outer_passed": results["outer_passed"],
            "leaderboard_score_claimed": False,
            "2024_labels_read": False,
            "2025_data_read": False,
            "submission_created": False,
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    raw_dir = args.raw_dir.expanduser().resolve()
    artifact_root = args.artifact_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    preregister = args.preregister.expanduser().resolve()
    if args.stage == "inner":
        _run_inner(
            raw_dir=raw_dir,
            artifact_root=artifact_root,
            out_dir=out_dir,
            preregister_path=preregister,
        )
        return 0
    if args.stage == "stress":
        _run_stress(raw_dir=raw_dir, out_dir=out_dir, preregister_path=preregister)
        return 0
    if args.stage == "outer":
        _run_outer(
            raw_dir=raw_dir,
            artifact_root=artifact_root,
            out_dir=out_dir,
            preregister_path=preregister,
        )
        return 0
    raise AssertionError(f"unreachable stage: {args.stage!r}")


if __name__ == "__main__":
    raise SystemExit(main())

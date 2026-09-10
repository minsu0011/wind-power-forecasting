"""Strict-forward low-dimensional monotone weather-to-power experiment.

Stage 1 is bounded to labels/features ending in 2023 and creates an immutable
recipe lock.  Stage 2 verifies that lock before reading 2024.  Final prediction
is possible only for groups that pass both the direct corrected-v3 gate and the
same-delta-on-corrected-recent-v4 gate on all registered 2024 slices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.run_run_sequence_residual import (  # noqa: E402
    _comparisons,
    _read_stage1_baseline,
    _write_submission,
)
from scripts.run_shared_q07_multiseed import (  # noqa: E402
    EXPECTED_ROWS_PRE2024,
    EXPECTED_ROWS_THROUGH2024,
    EXPECTED_TEST_ROWS,
    OFFICIAL_LABEL_PREFIX_SHA256,
    OFFICIAL_STAGE1_PREFIX_BYTES,
    YEAR_2023_END,
    YEAR_2024_END,
    _read_features,
    _read_labels,
    _read_test_features,
)
from src.manifest import (  # noqa: E402
    describe_file,
    git_state,
    make_manifest,
    package_versions,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from src.metric import CAPACITY_KWH, TARGET_COLS  # noqa: E402


PREREGISTER_SHA256 = (
    "b3296f9fab275670770dd3affb029c10240e39ad7a94965b367e370f38de516b"
)
BASE_PREREGISTER_SHA256 = (
    "0532e6843f811d38b2e1579e0085912f5b97ca017aa57e01556261857c034643"
)
STAGE1_START = pd.Timestamp("2023-01-01 01:00:00")
STAGE1_END = pd.Timestamp("2024-01-01 00:00:00")
G3_STAGE1_START = pd.Timestamp("2023-07-01 01:00:00")
STAGE2_START = pd.Timestamp("2024-01-01 01:00:00")
STAGE2_END = pd.Timestamp("2025-01-01 00:00:00")
FINAL_START = pd.Timestamp("2025-01-01 01:00:00")
FINAL_END = pd.Timestamp("2026-01-01 00:00:00")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir", type=Path, default=Path(r"data/local/open")
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/cache"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--preregister",
        type=Path,
        default=Path("configs/robust_physical_core_preregister_v2.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/postgate/robust_physical_core"),
    )
    parser.add_argument(
        "--stage", choices=("stage1", "stage2", "final", "audit"), required=True
    )
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
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    write_json_atomic(path, _json_ready(payload), overwrite=False)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        frame.to_parquet(temporary, engine="pyarrow", compression="zstd", index=True)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_joblib(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        joblib.dump(payload, temporary, compress=3)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _frame_sha256(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update("\n".join(map(str, frame.columns)).encode("utf-8"))
    digest.update("\n".join(map(str, frame.dtypes)).encode("utf-8"))
    digest.update(np.asarray(frame.index.view("i8"), dtype="<i8").tobytes())
    digest.update(
        np.asarray(frame.to_numpy(dtype=np.float64), dtype="<f8", order="C").tobytes()
    )
    return digest.hexdigest()


def _load_preregister(path: Path) -> dict[str, Any]:
    observed = sha256_file(path)
    if observed != PREREGISTER_SHA256:
        raise AssertionError(f"preregister SHA changed: {observed}")
    override = json.loads(path.read_text(encoding="utf-8"))
    if override.get("experiment_id") != "robust_physical_core_strict_forward_v2":
        raise AssertionError("unexpected experiment id")
    base_path = PROJECT_DIR / override["supersedes"]["path"]
    if sha256_file(base_path) != BASE_PREREGISTER_SHA256:
        raise AssertionError("immutable v1 base preregistration changed")
    payload = json.loads(base_path.read_text(encoding="utf-8"))
    payload["experiment_id"] = override["experiment_id"]
    payload["model_family"] = override["model_family_override"]
    payload["fixed_parameters"] = override["fixed_parameters_override"]
    payload["objectives"] = override["objectives_override"]
    payload["v2_override"] = override
    if len(payload["feature_columns"]) != 36:
        raise AssertionError("physical-core feature count changed")
    if [item["id"] for item in payload["objectives"]] != [
        "mono_l1",
        "mono_q060",
        "mono_q065",
    ]:
        raise AssertionError("objective declarations changed")
    if payload["blend_weights"] != [0.25, 0.5, 0.75, 1.0]:
        raise AssertionError("blend weights changed")
    return payload


def _index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="h", name="forecast_kst_dtm")


def _slice(index: pd.DatetimeIndex, start: str, end: str) -> pd.DatetimeIndex:
    selected = index[(index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))]
    if selected.empty:
        raise AssertionError(f"empty interval {start} .. {end}")
    return selected


def _stage1_slices(group: str) -> dict[str, tuple[str, str]]:
    if group != "kpx_group_3":
        return {
            "full": (str(STAGE1_START), str(STAGE1_END)),
            "H1": (str(STAGE1_START), "2023-07-01 00:00:00"),
            "H2": (str(G3_STAGE1_START), str(STAGE1_END)),
        }
    return {
        "full": (str(G3_STAGE1_START), str(STAGE1_END)),
        "Q3": (str(G3_STAGE1_START), "2023-10-01 00:00:00"),
        "Q4": ("2023-10-01 01:00:00", str(STAGE1_END)),
    }


def _stage2_slices() -> dict[str, tuple[str, str]]:
    return {
        "full": (str(STAGE2_START), str(STAGE2_END)),
        "H1": (str(STAGE2_START), "2024-07-01 00:00:00"),
        "H2": ("2024-07-01 01:00:00", str(STAGE2_END)),
    }


def _candidate_id(objective_id: str, weight: float) -> str:
    return f"{objective_id}__w{int(round(weight * 100)):03d}"


def _model_params(preregister: Mapping[str, Any], objective: Mapping[str, Any]) -> dict[str, Any]:
    params = dict(preregister["fixed_parameters"])
    params["loss"] = str(objective["loss"])
    if objective.get("quantile") is not None:
        params["quantile"] = float(objective["quantile"])
    positive = set(map(str, preregister["monotone_positive_columns"]))
    params["monotonic_cst"] = [
        1 if column in positive else 0 for column in preregister["feature_columns"]
    ]
    return params


def _fit_predict(
    *,
    preregister: Mapping[str, Any],
    objective: Mapping[str, Any],
    group: str,
    features: pd.DataFrame,
    actual: pd.Series,
    fit_index: pd.DatetimeIndex,
    apply_index: pd.DatetimeIndex,
) -> tuple[HistGradientBoostingRegressor, pd.Series, dict[str, Any]]:
    if fit_index.max() >= apply_index.min():
        raise AssertionError("fit/apply order is not strict-forward")
    columns = list(map(str, preregister["feature_columns"]))
    missing = [column for column in columns if column not in features.columns]
    if missing:
        raise AssertionError(f"registered features missing: {missing}")
    if any("scada" in column.lower() or "actual" in column.lower() for column in columns):
        raise AssertionError("forbidden supervised feature token")
    capacity = CAPACITY_KWH[group]
    target_cf = actual.loc[fit_index].astype(float) / capacity
    eligible = target_cf.notna() & np.isfinite(target_cf) & (target_cf >= 0.10)
    selected = target_cf.index[eligible]
    model = HistGradientBoostingRegressor(**_model_params(preregister, objective))
    model.fit(features.loc[selected, columns], target_cf.loc[selected])
    raw_cf = np.asarray(model.predict(features.loc[apply_index, columns]), dtype=float)
    prediction = pd.Series(
        np.clip(raw_cf, 0.0, 1.02) * capacity,
        index=apply_index,
        name=group,
    )
    if not np.isfinite(prediction.to_numpy()).all():
        raise AssertionError("model produced non-finite predictions")
    return model, prediction, {
        "fit_rows_total": int(len(fit_index)),
        "fit_rows_eligible": int(len(selected)),
        "fit_start": fit_index.min().isoformat(),
        "fit_end": fit_index.max().isoformat(),
        "apply_start": apply_index.min().isoformat(),
        "apply_end": apply_index.max().isoformat(),
        "fit_max_strictly_before_apply_min": True,
        "feature_count": len(columns),
        "actual_feature_columns": 0,
        "scada_feature_columns": 0,
        "prediction_min_kwh": float(prediction.min()),
        "prediction_max_kwh": float(prediction.max()),
    }


def _blend(baseline: pd.Series, robust: pd.Series, weight: float, group: str) -> pd.Series:
    if not baseline.index.equals(robust.index):
        raise AssertionError("blend index mismatch")
    values = np.clip(
        (1.0 - weight) * baseline.to_numpy(dtype=float)
        + weight * robust.to_numpy(dtype=float),
        0.0,
        1.02 * CAPACITY_KWH[group],
    )
    return pd.Series(values, index=baseline.index, name=group)


def _transfer_delta(
    v4: pd.Series, v3: pd.Series, direct_candidate: pd.Series, group: str
) -> pd.Series:
    if not v4.index.equals(v3.index) or not v4.index.equals(direct_candidate.index):
        raise AssertionError("delta-transfer index mismatch")
    values = np.clip(
        v4.to_numpy(dtype=float)
        + direct_candidate.to_numpy(dtype=float)
        - v3.to_numpy(dtype=float),
        0.0,
        1.02 * CAPACITY_KWH[group],
    )
    return pd.Series(values, index=v4.index, name=group)


def _select(candidates: Mapping[str, Mapping[str, Any]], required: Sequence[str]) -> str | None:
    eligible: list[tuple[float, float, float, int, str]] = []
    objective_order = {"mono_l1": 0, "mono_q060": 1, "mono_q065": 2}
    for candidate_id, values in candidates.items():
        deltas = [float(values["comparisons"][name]["delta"]) for name in required]
        if all(delta > 0.0 for delta in deltas):
            eligible.append(
                (
                    min(deltas),
                    float(values["comparisons"]["full"]["delta"]),
                    -float(values["weight"]),
                    -objective_order[str(values["objective_id"])],
                    candidate_id,
                )
            )
    if not eligible:
        return None
    eligible.sort(reverse=True)
    return eligible[0][-1]


def _input_snapshot(raw_dir: Path, artifact_root: Path, preregister_path: Path) -> dict[str, Any]:
    paths = {
        "labels": raw_dir / "train" / "train_labels.csv",
        "preregister": preregister_path,
        "stage1_baseline_g12": artifact_root / "oof" / "dev2023_locked_v3.parquet",
        "stage1_baseline_g3": artifact_root / "oof" / "g3dev2023h2_candidates.parquet",
        "stage1_baseline_exact": artifact_root
        / "postgate/shared_q07_multiseed_strict/oof/stage1_corrected_v3_baseline.parquet",
    }
    return {name: describe_file(path) for name, path in paths.items()}


def run_stage1(args: argparse.Namespace, preregister: Mapping[str, Any]) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("stage1_results.json", "stage1_lock.json", "stage1_predictions.parquet"):
        if (args.out_dir / name).exists():
            raise FileExistsError(f"refusing to overwrite {args.out_dir / name}")
    snapshot_before = _input_snapshot(args.raw_dir, args.artifact_root, args.preregister)
    labels = _read_labels(
        args.raw_dir / "train/train_labels.csv",
        nrows=EXPECTED_ROWS_PRE2024,
        expected_end=YEAR_2023_END,
        prefix_bytes=OFFICIAL_STAGE1_PREFIX_BYTES["labels"],
        expected_prefix_sha256=OFFICIAL_LABEL_PREFIX_SHA256,
    )
    weather = _read_features(args.cache_dir, labels, expected_end=YEAR_2023_END)
    baseline, baseline_audit = _read_stage1_baseline(args.artifact_root)
    predictions = pd.DataFrame(index=baseline.index)
    models: dict[str, Any] = {}
    results: dict[str, Any] = {}
    selected: dict[str, str | None] = {}
    for group in TARGET_COLS:
        bounds = preregister["stage1"]["selection_data"][group]
        fit_index = _slice(labels.index, *bounds["fit"])
        apply_index = _slice(labels.index, *bounds["apply"])
        group_candidates: dict[str, Any] = {}
        for objective in preregister["objectives"]:
            model, robust, fit_audit = _fit_predict(
                preregister=preregister,
                objective=objective,
                group=group,
                features=weather[group],
                actual=labels[group],
                fit_index=fit_index,
                apply_index=apply_index,
            )
            models[f"{group}__{objective['id']}"] = model
            for weight in preregister["blend_weights"]:
                candidate_id = _candidate_id(str(objective["id"]), float(weight))
                candidate = _blend(
                    baseline.loc[apply_index, group], robust, float(weight), group
                )
                column = f"{group}__{candidate_id}"
                predictions.loc[apply_index, column] = candidate.to_numpy(dtype=float)
                comparisons = _comparisons(
                    actual=labels.loc[apply_index, group],
                    baseline=baseline.loc[apply_index, group],
                    candidate=candidate,
                    group=group,
                    slices=_stage1_slices(group),
                )
                group_candidates[candidate_id] = {
                    "objective_id": str(objective["id"]),
                    "weight": float(weight),
                    "fit_audit": fit_audit,
                    "comparisons": comparisons,
                }
        required = preregister["stage1"]["required_slices"][group]
        selected[group] = _select(group_candidates, required)
        results[group] = {
            "required_slices": required,
            "selected_candidate": selected[group],
            "candidates": group_candidates,
        }
    snapshot_after = _input_snapshot(args.raw_dir, args.artifact_root, args.preregister)
    if snapshot_before != snapshot_after:
        raise AssertionError("Stage-1 immutable inputs changed")
    lock = {
        "experiment_id": preregister["experiment_id"],
        "preregister_sha256": PREREGISTER_SHA256,
        "created_utc": utc_now(),
        "selected_by_group": selected,
        "2024_labels_read": False,
        "selection_complete": True,
        "alternatives_after_lock": 0,
    }
    _atomic_parquet(predictions, args.out_dir / "stage1_predictions.parquet")
    _atomic_joblib(models, args.out_dir / "stage1_models.joblib")
    _write_json(
        args.out_dir / "stage1_results.json",
        {
            "protocol": "bounded pre-2024 selection",
            "baseline_audit": baseline_audit,
            "input_snapshot": snapshot_before,
            "results": results,
            "prediction_frame_sha256": _frame_sha256(predictions),
        },
    )
    _write_json(args.out_dir / "stage1_lock.json", lock)
    print(json.dumps(_json_ready(lock), ensure_ascii=False, indent=2))


def _load_lock(args: argparse.Namespace, preregister: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    path = args.out_dir / "stage1_lock.json"
    lock_sha = sha256_file(path)
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("preregister_sha256") != PREREGISTER_SHA256:
        raise AssertionError("lock belongs to a different preregistration")
    if lock.get("2024_labels_read") is not False or lock.get("selection_complete") is not True:
        raise AssertionError("invalid Stage-1 lock")
    for group, candidate_id in lock["selected_by_group"].items():
        if candidate_id is None:
            continue
        objective_id, weight_code = str(candidate_id).split("__")
        if objective_id not in [item["id"] for item in preregister["objectives"]]:
            raise AssertionError("locked objective is not registered")
        if int(weight_code[1:]) / 100.0 not in preregister["blend_weights"]:
            raise AssertionError("locked weight is not registered")
    return lock, lock_sha


def _read_full_labels(raw_dir: Path) -> pd.DataFrame:
    return _read_labels(
        raw_dir / "train/train_labels.csv",
        nrows=EXPECTED_ROWS_THROUGH2024,
        expected_end=YEAR_2024_END,
    )


def _read_prediction(path: Path, expected_index: pd.DatetimeIndex) -> pd.DataFrame:
    frame = pd.read_parquet(path, engine="pyarrow")
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if not frame.index.equals(expected_index):
        raise AssertionError(f"prediction index changed: {path}")
    if not set(TARGET_COLS).issubset(frame.columns):
        raise AssertionError(f"prediction schema changed: {path}")
    frame = frame.loc[:, list(TARGET_COLS)].astype(float)
    if not np.isfinite(frame.to_numpy()).all():
        raise AssertionError(f"non-finite prediction: {path}")
    return frame


def _objective_by_id(preregister: Mapping[str, Any], objective_id: str) -> Mapping[str, Any]:
    return next(item for item in preregister["objectives"] if item["id"] == objective_id)


def run_stage2(args: argparse.Namespace, preregister: Mapping[str, Any]) -> None:
    lock, lock_sha = _load_lock(args, preregister)
    for name in ("stage2_results.json", "promotion_lock.json", "stage2_predictions.parquet"):
        if (args.out_dir / name).exists():
            raise FileExistsError(f"refusing to overwrite {args.out_dir / name}")
    labels = _read_full_labels(args.raw_dir)
    weather = _read_features(args.cache_dir, labels, expected_end=YEAR_2024_END)
    gate_index = _index(STAGE2_START, STAGE2_END)
    v3 = _read_prediction(
        args.artifact_root / "oof/gate2024_locked_v3_cf_fix.parquet", gate_index
    )
    v4 = _read_prediction(
        args.artifact_root / "oof/gate2024_recent_v4_cf_fix_calibration_fit.parquet",
        gate_index,
    )
    output = pd.DataFrame(index=gate_index)
    models: dict[str, Any] = {}
    results: dict[str, Any] = {}
    confirmed: list[str] = []
    fit_index = _index(pd.Timestamp("2022-01-01 01:00:00"), YEAR_2023_END)
    for group in TARGET_COLS:
        candidate_id = lock["selected_by_group"].get(group)
        if candidate_id is None:
            results[group] = {"selected_candidate": None, "confirmed": False, "reason": "Stage-1 identity"}
            output[group] = v4[group]
            continue
        objective_id, weight_code = str(candidate_id).split("__")
        weight = int(weight_code[1:]) / 100.0
        model, robust, fit_audit = _fit_predict(
            preregister=preregister,
            objective=_objective_by_id(preregister, objective_id),
            group=group,
            features=weather[group],
            actual=labels[group],
            fit_index=fit_index,
            apply_index=gate_index,
        )
        models[group] = model
        direct = _blend(v3[group], robust, weight, group)
        transferred = _transfer_delta(v4[group], v3[group], direct, group)
        direct_comparisons = _comparisons(
            actual=labels.loc[gate_index, group], baseline=v3[group], candidate=direct,
            group=group, slices=_stage2_slices(),
        )
        transfer_comparisons = _comparisons(
            actual=labels.loc[gate_index, group], baseline=v4[group], candidate=transferred,
            group=group, slices=_stage2_slices(),
        )
        passes = all(
            direct_comparisons[name]["delta"] > 0.0
            and transfer_comparisons[name]["delta"] > 0.0
            for name in ("full", "H1", "H2")
        )
        if passes:
            confirmed.append(group)
            output[group] = transferred
        else:
            output[group] = v4[group]
        results[group] = {
            "selected_candidate": candidate_id,
            "fit_audit": fit_audit,
            "direct_vs_v3": direct_comparisons,
            "transferred_vs_v4": transfer_comparisons,
            "confirmed": passes,
        }
    _atomic_parquet(output, args.out_dir / "stage2_predictions.parquet")
    _atomic_joblib(models, args.out_dir / "stage2_models.joblib")
    _write_json(
        args.out_dir / "stage2_results.json",
        {
            "stage1_lock_sha256": lock_sha,
            "alternatives_after_lock": 0,
            "results": results,
            "confirmed_groups": confirmed,
            "prediction_frame_sha256": _frame_sha256(output),
        },
    )
    promotion = {
        "experiment_id": preregister["experiment_id"],
        "preregister_sha256": PREREGISTER_SHA256,
        "stage1_lock_sha256": lock_sha,
        "stage2_results_sha256": sha256_file(args.out_dir / "stage2_results.json"),
        "confirmed_groups": confirmed,
        "final_allowed": bool(confirmed),
        "2025_read": False,
    }
    _write_json(args.out_dir / "promotion_lock.json", promotion)
    print(json.dumps(_json_ready(promotion), ensure_ascii=False, indent=2))


def run_final(args: argparse.Namespace, preregister: Mapping[str, Any]) -> None:
    lock, lock_sha = _load_lock(args, preregister)
    promotion_path = args.out_dir / "promotion_lock.json"
    promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    if promotion.get("stage1_lock_sha256") != lock_sha:
        raise AssertionError("promotion lock does not match Stage-1 lock")
    confirmed = list(map(str, promotion.get("confirmed_groups", [])))
    if not confirmed or promotion.get("final_allowed") is not True:
        raise RuntimeError("no group passed both fixed confirmation gates; final CSV forbidden")
    csv_path = args.out_dir / "robust_physical_core_delta_2025.csv"
    parquet_path = args.out_dir / "robust_physical_core_delta_2025.parquet"
    if csv_path.exists() or parquet_path.exists():
        raise FileExistsError("refusing to overwrite final candidate")
    labels = _read_full_labels(args.raw_dir)
    weather = _read_features(args.cache_dir, labels, expected_end=YEAR_2024_END)
    sample = pd.read_csv(args.raw_dir / "sample_submission.csv", encoding="utf-8-sig")
    test_index = pd.DatetimeIndex(
        pd.to_datetime(sample["forecast_kst_dtm"], errors="raise"),
        name="forecast_kst_dtm",
    )
    if len(test_index) != EXPECTED_TEST_ROWS or not test_index.equals(_index(FINAL_START, FINAL_END)):
        raise AssertionError("sample timestamp contract changed")
    test_weather = _read_test_features(args.cache_dir, test_index)
    v3 = _read_prediction(
        args.artifact_root / "final_cf_fix/predictions/corrected_v3_test.parquet",
        test_index,
    )
    v4 = _read_prediction(
        args.artifact_root / "final_cf_fix/predictions/corrected_recent_v4_test.parquet",
        test_index,
    )
    output = v4.copy()
    models: dict[str, Any] = {}
    audits: dict[str, Any] = {}
    fit_index = labels.index
    for group in confirmed:
        candidate_id = lock["selected_by_group"][group]
        objective_id, weight_code = str(candidate_id).split("__")
        weight = int(weight_code[1:]) / 100.0
        model, robust, audit = _fit_predict(
            preregister=preregister,
            objective=_objective_by_id(preregister, objective_id),
            group=group,
            features=pd.concat([weather[group], test_weather[group]]),
            actual=labels[group].reindex(pd.concat([weather[group], test_weather[group]]).index),
            fit_index=fit_index,
            apply_index=test_index,
        )
        direct = _blend(v3[group], robust, weight, group)
        output[group] = _transfer_delta(v4[group], v3[group], direct, group)
        models[group] = model
        audits[group] = audit
    for group in set(TARGET_COLS).difference(confirmed):
        if not np.array_equal(output[group].to_numpy(), v4[group].to_numpy()):
            raise AssertionError(f"unconfirmed {group} changed from v4")
    _atomic_parquet(output, parquet_path)
    csv_audit = _write_submission(args.raw_dir / "sample_submission.csv", output, csv_path)
    _atomic_joblib(models, args.out_dir / "final_models.joblib")
    manifest = {
        "artifact_type": "strict_forward_robust_physical_core_delta",
        "created_utc": utc_now(),
        "preregister": describe_file(args.preregister),
        "stage1_lock": describe_file(args.out_dir / "stage1_lock.json"),
        "promotion_lock": describe_file(promotion_path),
        "confirmed_groups": confirmed,
        "unchanged_groups_bit_exact_to_v4": sorted(set(TARGET_COLS).difference(confirmed)),
        "fit_audits": audits,
        "csv": csv_audit,
        "parquet": describe_file(parquet_path),
        "leaderboard_score_claim": False,
        "git": git_state(PROJECT_DIR),
        "packages": package_versions(),
    }
    _write_json(args.out_dir / "manifest.json", manifest)
    print(json.dumps(_json_ready({"confirmed_groups": confirmed, "csv": csv_audit}), ensure_ascii=False, indent=2))


def run_audit(args: argparse.Namespace, preregister: Mapping[str, Any]) -> None:
    manifest_path = args.out_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite {manifest_path}")
    lock_path = args.out_dir / "stage1_lock.json"
    results_path = args.out_dir / "stage1_results.json"
    predictions_path = args.out_dir / "stage1_predictions.parquet"
    models_path = args.out_dir / "stage1_models.joblib"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))
    selected = lock["selected_by_group"]
    all_rejected = all(value is None for value in selected.values())
    if not all_rejected:
        raise RuntimeError("audit mode is reserved for the all-rejected Stage-1 result")
    if lock.get("2024_labels_read") is not False:
        raise AssertionError("rejected Stage-1 lock claims 2024 access")
    predictions = pd.read_parquet(predictions_path, engine="pyarrow")
    predictions.index = pd.DatetimeIndex(predictions.index, name="forecast_kst_dtm")
    if _frame_sha256(predictions) != results["prediction_frame_sha256"]:
        raise AssertionError("Stage-1 prediction frame hash changed")
    manifest = make_manifest(
        artifact_type="strict_forward_robust_physical_core_rejected_stage1",
        parameters={
            "experiment_id": preregister["experiment_id"],
            "preregister_sha256": PREREGISTER_SHA256,
            "v1_incompatible_preregister_sha256": BASE_PREREGISTER_SHA256,
            "v1_failure": "LightGBM rejected L1 plus monotone constraints before predictions or metrics; v1 preserved and estimator-only v2 was preregistered.",
            "selected_by_group": selected,
            "2024_labels_read": False,
            "2025_weather_read": False,
            "2025_csv_created": False,
        },
        input_files=[
            args.preregister,
            PROJECT_DIR / "configs/robust_physical_core_preregister_v1_incompatible.json",
            args.raw_dir / "train/train_labels.csv",
            *(args.cache_dir / f"{group}_weather_train.parquet" for group in TARGET_COLS),
            args.artifact_root / "oof/dev2023_locked_v3.parquet",
            args.artifact_root / "oof/g3dev2023h2_candidates.parquet",
            args.artifact_root
            / "postgate/shared_q07_multiseed_strict/oof/stage1_corrected_v3_baseline.parquet",
        ],
        output_files=[lock_path, results_path, predictions_path, models_path],
        results={
            "status": "rejected_stage1",
            "all_groups_identity": True,
            "stage2_skipped": True,
            "reason": "all 36 group-candidates had at least one non-positive required pre-2024 slice",
        },
        project_dir=PROJECT_DIR,
    )
    _write_json(manifest_path, manifest)
    print(json.dumps(_json_ready(manifest["results"]), ensure_ascii=False, indent=2))


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.raw_dir = args.raw_dir.resolve()
    args.cache_dir = args.cache_dir.resolve()
    args.artifact_root = args.artifact_root.resolve()
    args.preregister = args.preregister.resolve()
    args.out_dir = args.out_dir.resolve()
    preregister = _load_preregister(args.preregister)
    if args.stage == "stage1":
        run_stage1(args, preregister)
    elif args.stage == "stage2":
        run_stage2(args, preregister)
    elif args.stage == "final":
        run_final(args, preregister)
    else:
        run_audit(args, preregister)


if __name__ == "__main__":
    main()

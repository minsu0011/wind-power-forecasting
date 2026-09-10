"""Independent reconstruction audit for direct interval-probability Stage1."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts import run_shared_q07_multiseed as physical  # noqa: E402
from src.manifest import describe_file, sha256_file, utc_now, write_json_atomic  # noqa: E402
from src.metric import CAPACITY_KWH, TARGET_COLS  # noqa: E402


PREREGISTER_SHA256 = "1796cc8628e3ad46039a6f5352c4c51b1a851afc7e6e6f055d9c85fcd1d699ff"
EXPECTED_MANIFEST_SHA256 = "0ac3e2b01820d3ca47f449a7efa79c2a7670c35fcf9c61a7ac23c605088a2b0e"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir", type=Path, default=Path(r"data/local/open")
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/postgate/direct_interval_probability_strict_v1"),
    )
    parser.add_argument(
        "--preregister",
        type=Path,
        default=Path("configs/direct_interval_probability_preregister_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/audits/direct_interval_probability_strict_v1_independent.json"),
    )
    return parser.parse_args(argv)


def _canonical_snapshot(root: Path) -> dict[str, dict[str, Any]]:
    return {
        path.relative_to(root).as_posix(): describe_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _exact_prefix(path: Path, *, byte_limit: int, expected_sha256: str) -> bytes:
    with path.open("rb", buffering=0) as stream:
        payload = stream.read(byte_limit)
        position = stream.tell()
    if len(payload) != byte_limit or position != byte_limit:
        raise AssertionError("physical prefix read did not stop at exact byte limit")
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise AssertionError(f"physical prefix hash differs: {path}")
    return payload


def _load_bounded_labels(path: Path, spec: Mapping[str, Any]) -> pd.DataFrame:
    payload = _exact_prefix(
        path, byte_limit=int(spec["bytes"]), expected_sha256=str(spec["sha256"])
    )
    frame = pd.read_csv(io.BytesIO(payload), encoding="utf-8-sig")
    if len(frame) != int(spec["data_rows"]):
        raise AssertionError("bounded label row count differs")
    frame.index = pd.DatetimeIndex(
        pd.to_datetime(frame.pop("kst_dtm"), errors="raise"), name="forecast_kst_dtm"
    )
    if frame.index.max() != pd.Timestamp(spec["end"]):
        raise AssertionError("bounded label end differs")
    return frame.astype(float)


def _manual_group_score(
    actual: pd.Series, prediction: pd.Series, capacity: float
) -> dict[str, float | int]:
    y = actual.to_numpy(dtype=np.float64)
    p = prediction.to_numpy(dtype=np.float64)
    eligible = np.isfinite(y) & (y >= 0.10 * capacity)
    if not np.any(eligible):
        raise AssertionError("registered slice has zero eligible rows")
    yy = y[eligible]
    pp = p[eligible]
    error = np.abs(pp - yy) / capacity
    payment = np.where(error <= 0.06, 4.0, np.where(error <= 0.08, 3.0, 0.0))
    nmae = float(np.mean(error))
    ficr = float(np.sum(yy * payment) / (4.0 * np.sum(yy)))
    one_minus = 1.0 - nmae
    return {
        "n_evaluated": int(eligible.sum()),
        "nmae": nmae,
        "one_minus_nmae": one_minus,
        "ficr": ficr,
        "score": 0.5 * (one_minus + ficr),
    }


def _assert_close(observed: float, expected: float, *, name: str) -> None:
    if not np.isclose(observed, expected, atol=2e-15, rtol=0.0):
        raise AssertionError(f"{name} differs: {observed} != {expected}")


def _manual_surfaces(model: Any, context: pd.DataFrame, actions: np.ndarray) -> dict[str, np.ndarray]:
    source = context.to_numpy(dtype=np.float32, copy=False)
    design = np.empty((len(context) * len(actions), context.shape[1] + 1), dtype=np.float32)
    design[:, :-1] = np.repeat(source, len(actions), axis=0)
    design[:, -1] = np.tile(actions.astype(np.float32), len(context))
    shape = (len(context), len(actions))
    p6_raw = np.asarray(model.p6_model_.predict_proba(design)[:, 1], dtype=np.float64).reshape(shape)
    p8_raw = np.asarray(model.p8_model_.predict_proba(design)[:, 1], dtype=np.float64).reshape(shape)
    eae = np.clip(np.asarray(model.eae_model_.predict(design), dtype=np.float64).reshape(shape), 0.0, 1.20)
    violating = p6_raw > p8_raw
    midpoint = 0.5 * (p6_raw + p8_raw)
    p6 = np.where(violating, midpoint, p6_raw)
    p8 = np.where(violating, midpoint, p8_raw)
    utility = -0.5 * eae + 0.5 * (0.25 * p6 + 0.75 * p8)
    return {
        "p6_raw": p6_raw,
        "p8_raw": p8_raw,
        "p6": p6,
        "p8": p8,
        "eae": eae,
        "utility": utility,
        "repair_count": np.asarray([int(violating.sum())], dtype=np.int64),
    }


def _manual_select(
    utility: np.ndarray, baseline_cf: np.ndarray, actions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    maximum = utility.max(axis=1)
    tied = utility >= maximum[:, None] - 1e-12
    distance = np.abs(actions[None, :] - baseline_cf[:, None])
    nearest_distance = np.where(tied, distance, np.inf).min(axis=1)
    nearest = tied & (distance <= nearest_distance[:, None] + 1e-15)
    selected_index = np.argmax(nearest, axis=1).astype(np.int16)
    return actions[selected_index], selected_index


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(args.output)
    before = _canonical_snapshot(args.artifact_dir)
    if sha256_file(args.preregister) != PREREGISTER_SHA256:
        raise AssertionError("preregister hash differs")
    manifest_path = args.artifact_dir / "manifest.json"
    if sha256_file(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise AssertionError("canonical manifest hash differs")
    preregister = json.loads(args.preregister.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = json.loads((args.artifact_dir / "stage1_results.json").read_text(encoding="utf-8"))

    actual_outputs = {
        path.relative_to(args.artifact_dir).as_posix(): describe_file(path)
        for path in args.artifact_dir.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    recorded_outputs = {
        Path(record["path"]).relative_to(args.artifact_dir.resolve()).as_posix(): record
        for record in manifest["outputs"]
    }
    if set(actual_outputs) != set(recorded_outputs):
        raise AssertionError("manifest output set differs")
    for relative, actual in actual_outputs.items():
        recorded = recorded_outputs[relative]
        if actual["size_bytes"] != recorded["size_bytes"] or actual["sha256"] != recorded["sha256"]:
            raise AssertionError(f"manifest output identity differs: {relative}")

    for name, record in manifest["source_and_config_snapshot"].items():
        path = Path(record["path"])
        actual = describe_file(path)
        if actual["size_bytes"] != record["size_bytes"] or actual["sha256"] != record["sha256"]:
            raise AssertionError(f"source/config changed: {name}")

    specs = preregister["physical_stage1_inputs"]
    for source, key in (("ldaps", "ldaps_weather_prefix"), ("gfs", "gfs_weather_prefix")):
        spec = specs[key]
        _exact_prefix(
            args.raw_dir / "train" / f"{source}_train.csv",
            byte_limit=int(spec["bytes"]),
            expected_sha256=str(spec["sha256"]),
        )
    labels = _load_bounded_labels(
        args.raw_dir / "train/train_labels.csv",
        specs["labels_stage1_score_prefix_after_prescore_lock"],
    )
    full_index = pd.date_range(
        "2022-01-01 01:00:00", "2024-01-01 00:00:00", freq="h", name="forecast_kst_dtm"
    )
    raw_features, raw_evidence = physical._read_stage1_raw_features(
        args.raw_dir, pd.DataFrame(index=full_index)
    )
    feature_names = tuple(preregister["feature_contract"]["context_columns_in_exact_order"])
    if len(feature_names) != 26:
        raise AssertionError("compact feature count differs")
    contexts = {
        group: raw_features[group].loc[:, list(feature_names)].astype(np.float32)
        for group in TARGET_COLS
    }
    baseline = pd.read_parquet(args.artifact_dir / "oof/stage1_baseline_2023.parquet")
    stored_actions = pd.read_parquet(args.artifact_dir / "oof/stage1_selected_actions_2023.parquet")
    action_grid = np.arange(103, dtype=np.float64) / 100.0
    segments = {
        "full": pd.date_range("2023-01-01 01:00", "2024-01-01 00:00", freq="h"),
        "H1": pd.date_range("2023-01-01 01:00", "2023-07-01 00:00", freq="h"),
        "H2": pd.date_range("2023-07-01 01:00", "2024-01-01 00:00", freq="h"),
        "Q1": pd.date_range("2023-01-01 01:00", "2023-04-01 00:00", freq="h"),
        "Q2": pd.date_range("2023-04-01 01:00", "2023-07-01 00:00", freq="h"),
        "Q3": pd.date_range("2023-07-01 01:00", "2023-10-01 00:00", freq="h"),
        "Q4": pd.date_range("2023-10-01 01:00", "2024-01-01 00:00", freq="h"),
    }
    application = {
        "kpx_group_1": segments["full"],
        "kpx_group_2": segments["full"],
        "kpx_group_3": segments["H2"],
    }
    group_audit: dict[str, Any] = {}
    for group in TARGET_COLS:
        index = application[group]
        model = joblib.load(args.artifact_dir / f"models/stage1_{group}.joblib")
        surfaces = _manual_surfaces(model, contexts[group].loc[index], action_grid)
        with np.load(args.artifact_dir / f"oof/stage1_{group}_surfaces.npz") as stored:
            for name, expected in surfaces.items():
                if not np.array_equal(stored[name], expected):
                    raise AssertionError(f"{group}/{name} surface reconstruction differs")
        baseline_cf = baseline.loc[index, group].to_numpy(dtype=np.float64) / CAPACITY_KWH[group]
        action, action_index = _manual_select(surfaces["utility"], baseline_cf, action_grid)
        if not np.array_equal(action, stored_actions.loc[index, group].to_numpy(dtype=np.float64)):
            raise AssertionError(f"{group} manual action differs")
        diagnostics = pd.read_parquet(
            args.artifact_dir / f"oof/stage1_{group}_action_diagnostics.parquet"
        )
        if not np.array_equal(action_index, diagnostics["selected_action_index"].to_numpy()):
            raise AssertionError(f"{group} manual action index differs")
        if np.any(surfaces["p6"] > surfaces["p8"]):
            raise AssertionError(f"{group} monotonic probability repair failed")
        group_audit[group] = {
            "rows": len(index),
            "model_three_heads_present": all(
                getattr(model, name, None) is not None
                for name in ("p6_model_", "p8_model_", "eae_model_")
            ),
            "surface_names_bit_exact": list(surfaces),
            "action_and_tie_break_bit_exact": True,
            "repair_count": int(surfaces["repair_count"][0]),
        }

    weights = {"w025": 0.025, "w05": 0.05, "w10": 0.10}
    independent_deltas: dict[str, Any] = {}
    registered_nonempty = 0
    for key, weight in weights.items():
        stored_candidate = pd.read_parquet(args.artifact_dir / f"oof/stage1_blend_{key}_2023.parquet")
        independent_deltas[key] = {}
        for group in TARGET_COLS:
            index = application[group]
            expected = baseline.loc[index, group].to_numpy(dtype=np.float64).copy()
            expected = np.clip(
                (1.0 - weight) * expected
                + weight * stored_actions.loc[index, group].to_numpy(dtype=np.float64) * CAPACITY_KWH[group],
                0.0,
                1.02 * CAPACITY_KWH[group],
            )
            if not np.array_equal(expected, stored_candidate.loc[index, group].to_numpy(dtype=np.float64)):
                raise AssertionError(f"{key}/{group} candidate formula differs")
            names = ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4") if group != "kpx_group_3" else ("full", "Q3", "Q4")
            independent_deltas[key][group] = {}
            for name in names:
                slice_index = segments["H2"] if group == "kpx_group_3" and name == "full" else segments[name]
                base_score = _manual_group_score(
                    labels.loc[slice_index, group], baseline.loc[slice_index, group], CAPACITY_KWH[group]
                )
                candidate_score = _manual_group_score(
                    labels.loc[slice_index, group], stored_candidate.loc[slice_index, group], CAPACITY_KWH[group]
                )
                recorded = result["comparisons"][key][group][name]
                for metric_name in ("score", "one_minus_nmae", "ficr"):
                    _assert_close(
                        float(recorded["baseline"][metric_name]),
                        float(base_score[metric_name]),
                        name=f"{key}/{group}/{name}/baseline/{metric_name}",
                    )
                    _assert_close(
                        float(recorded["candidate"][metric_name]),
                        float(candidate_score[metric_name]),
                        name=f"{key}/{group}/{name}/candidate/{metric_name}",
                    )
                delta = float(candidate_score["score"]) - float(base_score["score"])
                _assert_close(float(recorded["delta"]), delta, name=f"{key}/{group}/{name}/delta")
                independent_deltas[key][group][name] = delta
                registered_nonempty += 1
    if registered_nonempty != 51:
        raise AssertionError(f"candidate/slice metric count differs: {registered_nonempty}")
    if result["locked_candidate"] != "identity" or any(result["group_passed_stage1"].values()):
        raise AssertionError("independent gate result differs from identity reject")
    if list(args.artifact_dir.rglob("*.csv")):
        raise AssertionError("canonical directory unexpectedly contains CSV")

    after = _canonical_snapshot(args.artifact_dir)
    if before != after:
        raise AssertionError("canonical artifact mutated during audit")
    report = {
        "schema_version": 1,
        "audit_type": "direct_interval_probability_independent_reconstruction_v1",
        "created_utc": utc_now(),
        "status": "PASS",
        "auditor_source": describe_file(Path(__file__)),
        "preregister_sha256": PREREGISTER_SHA256,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "manifest_output_count": len(actual_outputs),
        "source_config_snapshot_count": len(manifest["source_and_config_snapshot"]),
        "raw_physical_prefix_rebuilt": raw_evidence,
        "group_reconstruction": group_audit,
        "candidate_slice_metric_recomputation_count": registered_nonempty,
        "independent_deltas": independent_deltas,
        "gate_recomputed_identity_reject": True,
        "year_2024_value_bytes_read": 0,
        "year_2025_value_bytes_read": 0,
        "sample_value_bytes_read": 0,
        "public_or_scale_artifact_bytes_read": 0,
        "csv_files_in_canonical": 0,
        "canonical_nonmutation": True,
    }
    write_json_atomic(args.output, report, overwrite=False)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    report = run_audit(parse_args(argv))
    print(f"independent audit: {report['status']}", flush=True)
    print(f"audit sha256: {sha256_file(parse_args(argv).output)}", flush=True)


if __name__ == "__main__":
    main()

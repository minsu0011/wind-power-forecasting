"""Build the explicitly unsafe attack submission from the locked SCADA candidate.

This script performs no candidate selection.  It accepts only the stage-1
locked group-1 q0.7 / 20% blend, verifies that it failed the strict stage-2
quarter gate, refits on all permissible 2022-2024 data, and emits a separately
labelled attack-only submission.
"""

from __future__ import annotations

import argparse
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

from scripts.run_turbine_scada_power import (  # noqa: E402
    PREREGISTER_SHA256,
    TEST_START,
    YEAR_2022_START,
    YEAR_2024_END,
    YEAR_2025_END,
    _aggregate_turbine_hourly,
    _atomic_csv,
    _atomic_joblib,
    _atomic_json,
    _atomic_parquet,
    _blend,
    _design_matrix,
    _discover_csv_prefix,
    _fit_predict_group,
    _load_turbine_metadata,
    _read_cached_features,
    _read_full_labels,
    _read_prediction,
    _read_scada_prefix,
    _verify_preregister,
    _verify_stage1_lock,
    _verify_stage2_lock,
    _weather_columns,
)
from src.manifest import make_manifest, sha256_file, write_json_atomic  # noqa: E402
from src.metric import CAPACITY_KWH, TARGET_COLS  # noqa: E402


GROUP = "kpx_group_1"
OBJECTIVE = "q07"
BLEND_WEIGHT = 0.20
LOCKED_CANDIDATE = "q07__w20"
CANONICAL_RUNNER_SHA256 = (
    "25ebfce33160c363f7c510f2de1e5a9507b9c63ce9a65cd99b5fffb4eec08259"
)
CANONICAL_TEST_SHA256 = (
    "1f7e032b6eb39d5644b9b8fa6af7e96d67ae522bbc90a626f4deba9e6c1f856a"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir", type=Path, default=Path(r"data/local/open")
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/cache"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--canonical-dir",
        type=Path,
        default=Path("artifacts/postgate/turbine_scada_power"),
    )
    parser.add_argument(
        "--preregister",
        type=Path,
        default=Path("configs/turbine_scada_power_preregister.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/postgate/turbine_scada_power_attack"),
    )
    parser.add_argument("--n-jobs", type=int, default=7)
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


def _canonical_contract(
    args: argparse.Namespace, preregister: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    canonical_dir = args.canonical_dir.resolve()
    canonical_args = argparse.Namespace(
        out_dir=canonical_dir,
        preregister=args.preregister.resolve(),
        recipe=(PROJECT_DIR / "configs" / "train_final.v3.locked.json").resolve(),
    )
    stage1, stage1_lock = _verify_stage1_lock(canonical_args, preregister)
    stage2, stage2_lock = _verify_stage2_lock(canonical_args)
    if sha256_file(PROJECT_DIR / "scripts" / "run_turbine_scada_power.py") != CANONICAL_RUNNER_SHA256:
        raise AssertionError("canonical runner changed after strict lock")
    if sha256_file(PROJECT_DIR / "tests" / "test_turbine_scada_power.py") != CANONICAL_TEST_SHA256:
        raise AssertionError("canonical test changed after strict lock")
    expected_stage1 = {GROUP: LOCKED_CANDIDATE}
    if stage1_lock["locked_groups"] != [GROUP] or stage1_lock["selected"] != expected_stage1:
        raise AssertionError("attack source is not the one locked stage-1 candidate")
    if stage2["locked_groups"] != [GROUP] or stage2["selected"] != expected_stage1:
        raise AssertionError("stage-2 candidate identity changed")
    if stage2["promoted_groups"] or stage2_lock["promoted_groups"]:
        raise AssertionError("strict stage unexpectedly promoted the attack candidate")
    deltas = {
        name: float(values["delta"])
        for name, values in stage2["comparisons"][GROUP].items()
    }
    if not all(deltas[name] > 0.0 for name in ("full", "H1", "H2")):
        raise AssertionError("attack candidate no longer has positive full/half transfer")
    if not (deltas["Q2"] < 0.0 and deltas["Q3"] < 0.0):
        raise AssertionError("expected strict Q2/Q3 reversal is absent")
    return stage1, stage1_lock, stage2, stage2_lock


def _reload_prediction(
    *,
    model_path: Path,
    test_weather: pd.DataFrame,
    metadata: pd.DataFrame,
    preregister: Mapping[str, Any],
) -> np.ndarray:
    payload = joblib.load(model_path)
    model = payload["model"]
    selected = _weather_columns(test_weather, preregister)
    design = _design_matrix(test_weather, metadata, selected, GROUP)
    capacity_factors = np.asarray(model.predict(design), dtype=float).reshape(
        len(test_weather), len(metadata)
    )
    capacity_factors = np.clip(capacity_factors, 0.0, 1.02)
    return capacity_factors @ metadata["rated_kwh"].to_numpy(dtype=float)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for name in (
        "raw_dir",
        "cache_dir",
        "artifact_root",
        "canonical_dir",
        "preregister",
        "out_dir",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"attack output must be new and empty: {args.out_dir}")
    preregister = _verify_preregister(args.preregister)
    if preregister["sha256"] != PREREGISTER_SHA256:
        raise AssertionError("preregister identity changed")
    stage1, stage1_lock, stage2, stage2_lock = _canonical_contract(args, preregister)

    labels = _read_full_labels(args.raw_dir / "train" / "train_labels.csv")
    train_weather = _read_cached_features(args.cache_dir, [GROUP], labels.index)[GROUP]
    sample_path = args.raw_dir / "sample_submission.csv"
    sample = pd.read_csv(sample_path, encoding="utf-8-sig")
    expected_columns = ("forecast_id", "forecast_kst_dtm", *TARGET_COLS)
    if tuple(sample.columns) != expected_columns:
        raise AssertionError("sample submission schema changed")
    test_index = pd.DatetimeIndex(
        pd.to_datetime(sample["forecast_kst_dtm"], errors="raise"),
        name="forecast_kst_dtm",
    )
    if (
        len(test_index) != 8760
        or test_index.min() != TEST_START
        or test_index.max() != YEAR_2025_END
        or not test_index.is_unique
        or not test_index.is_monotonic_increasing
    ):
        raise AssertionError("test time contract changed")
    test_weather_path = args.cache_dir / f"{GROUP}_weather_test.parquet"
    test_weather = pd.read_parquet(test_weather_path, engine="pyarrow")
    test_weather.index = pd.DatetimeIndex(test_weather.index, name="forecast_kst_dtm")
    if not test_weather.index.equals(test_index):
        raise AssertionError("test weather index changed")
    test_weather = test_weather.astype(np.float32, copy=False)
    metadata = _load_turbine_metadata(args.raw_dir / "info.xlsx")[GROUP]

    scada_path = args.raw_dir / "train" / "scada_vestas_train.csv"
    scada_prefix = _discover_csv_prefix(scada_path, pd.Timestamp("2025-01-01 00:00:00"))
    scada_frame, scada_contract = _read_scada_prefix(scada_prefix)
    hourly, hourly_audit = _aggregate_turbine_hourly(scada_frame, GROUP)
    all_weather = pd.concat([train_weather, test_weather], axis=0)
    print("full-fit fixed kpx_group_1 q07 w20 attack candidate", flush=True)
    model, raw_prediction, fit_audit = _fit_predict_group(
        group=GROUP,
        objective=OBJECTIVE,
        prereg=preregister["payload"],
        weather=all_weather,
        labels=labels[GROUP].reindex(all_weather.index),
        hourly=hourly,
        metadata=metadata,
        train_start=YEAR_2022_START,
        train_end=YEAR_2024_END,
        predict_index=test_index,
        n_jobs=args.n_jobs,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "models" / "g1_q07_w20_full_2022_2024.joblib"
    _atomic_joblib(
        {
            "model": model,
            "selection": LOCKED_CANDIDATE,
            "fit_audit": fit_audit,
            "hourly_audit": hourly_audit,
            "metadata": metadata,
        },
        model_path,
    )
    reloaded = _reload_prediction(
        model_path=model_path,
        test_weather=test_weather,
        metadata=metadata,
        preregister=preregister["payload"],
    )
    original = raw_prediction.to_numpy(dtype=float)
    reload_bit_exact = bool(np.array_equal(original, reloaded))
    reload_max_abs = float(np.max(np.abs(original - reloaded)))
    if not reload_bit_exact or reload_max_abs != 0.0:
        raise AssertionError("serialized full model did not re-predict bit-exactly")

    raw_prediction_path = args.out_dir / "predictions" / "g1_turbine_scada_q07_2025.parquet"
    _atomic_parquet(raw_prediction.to_frame(), raw_prediction_path)
    raw_readback = pd.read_parquet(raw_prediction_path, engine="pyarrow")
    raw_readback.index = pd.DatetimeIndex(raw_readback.index, name="forecast_kst_dtm")
    if not raw_readback.index.equals(test_index) or not np.array_equal(
        raw_readback[GROUP].to_numpy(dtype=float), original
    ):
        raise AssertionError("raw prediction parquet readback changed values")

    baseline_path = (
        args.artifact_root
        / "final_cf_fix"
        / "predictions"
        / "corrected_v3_test.parquet"
    )
    baseline = _read_prediction(baseline_path, test_index, TARGET_COLS)
    candidate = baseline.copy()
    candidate[GROUP] = _blend(baseline[GROUP], raw_prediction, GROUP, BLEND_WEIGHT)
    for group in TARGET_COLS:
        values = candidate[group].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise AssertionError(f"{group} candidate contains non-finite values")
        if float(values.min()) < 0.0 or float(values.max()) > 1.02 * CAPACITY_KWH[group] + 1e-9:
            raise AssertionError(f"{group} candidate violates capacity clipping")
        if group != GROUP and not np.array_equal(values, baseline[group].to_numpy(dtype=float)):
            raise AssertionError(f"{group} must remain bit-identical to corrected-v3")
    candidate_path = args.out_dir / "predictions" / "corrected_v3_g1_scada_attack_2025.parquet"
    _atomic_parquet(candidate, candidate_path)
    candidate_readback = pd.read_parquet(candidate_path, engine="pyarrow")
    candidate_readback.index = pd.DatetimeIndex(
        candidate_readback.index, name="forecast_kst_dtm"
    )
    if not candidate_readback.index.equals(test_index) or not np.array_equal(
        candidate_readback.to_numpy(dtype=float), candidate.to_numpy(dtype=float)
    ):
        raise AssertionError("candidate parquet readback changed values")

    submission = sample.copy()
    for group in TARGET_COLS:
        submission[group] = candidate[group].to_numpy(dtype=float)
    submission_path = args.out_dir / "turbine_scada_power_attack_2025.csv"
    _atomic_csv(submission, submission_path)
    raw_bytes = submission_path.read_bytes()
    if not raw_bytes.startswith(b"\xef\xbb\xbf"):
        raise AssertionError("submission is missing UTF-8 BOM")
    csv_readback = pd.read_csv(submission_path, encoding="utf-8-sig")
    if tuple(csv_readback.columns) != expected_columns or len(csv_readback) != 8760:
        raise AssertionError("submission readback schema/row count changed")
    if not csv_readback["forecast_id"].equals(sample["forecast_id"]):
        raise AssertionError("submission IDs changed")
    if not csv_readback["forecast_kst_dtm"].equals(sample["forecast_kst_dtm"]):
        raise AssertionError("submission timestamps changed")
    csv_max_abs = 0.0
    for group in TARGET_COLS:
        expected = candidate[group].to_numpy(dtype=float)
        observed = csv_readback[group].to_numpy(dtype=float)
        difference = float(np.max(np.abs(observed - expected)))
        csv_max_abs = max(csv_max_abs, difference)
        if difference > 5.01e-7 or not np.isfinite(observed).all():
            raise AssertionError(f"{group} CSV precision/readback failed: {difference}")

    quarter_deltas = {
        name: float(values["delta"])
        for name, values in stage2["comparisons"][GROUP].items()
    }
    results = {
        "schema_version": 1,
        "artifact_type": "turbine_scada_power_attack_only",
        "selection_source": "immutable strict stage1 lock",
        "selection": {"group": GROUP, "objective": OBJECTIVE, "blend_weight": BLEND_WEIGHT},
        "additional_model_or_weight_selection": False,
        "selection_unsafe": True,
        "strict_adoption": False,
        "attack_submission_only": True,
        "strict_rejection_reason": "Q2 and Q3 2024 score deltas are negative despite positive full/H1/H2 transfer",
        "stage2_deltas": quarter_deltas,
        "canonical": {
            "preregister_sha256": PREREGISTER_SHA256,
            "stage1_lock_sha256": sha256_file(args.canonical_dir / "stage1_lock.json"),
            "stage1_results_sha256": sha256_file(args.canonical_dir / "stage1_results.json"),
            "stage2_lock_sha256": sha256_file(args.canonical_dir / "stage2_promotion_lock.json"),
            "stage2_results_sha256": sha256_file(args.canonical_dir / "stage2_results.json"),
            "runner_sha256": CANONICAL_RUNNER_SHA256,
            "test_sha256": CANONICAL_TEST_SHA256,
            "strict_promoted_groups": stage2["promoted_groups"],
        },
        "full_fit": fit_audit,
        "hourly_scada": hourly_audit,
        "scada_io_contract": scada_contract,
        "verification": {
            "serialized_model_reload_repredict_bit_exact": reload_bit_exact,
            "serialized_model_reload_repredict_max_abs": reload_max_abs,
            "raw_prediction_parquet_readback_bit_exact": True,
            "candidate_parquet_readback_bit_exact": True,
            "sample_schema_exact": True,
            "sample_id_exact": True,
            "sample_timestamp_exact": True,
            "utf8_bom": True,
            "finite": True,
            "capacity_clip": True,
            "csv_max_abs_roundtrip_difference": csv_max_abs,
            "unchanged_groups_bit_exact_to_corrected_v3": ["kpx_group_2", "kpx_group_3"],
        },
        "artifacts": {
            "model": str(model_path.resolve()),
            "model_sha256": sha256_file(model_path),
            "raw_prediction": str(raw_prediction_path.resolve()),
            "raw_prediction_sha256": sha256_file(raw_prediction_path),
            "candidate_prediction": str(candidate_path.resolve()),
            "candidate_prediction_sha256": sha256_file(candidate_path),
            "submission": str(submission_path.resolve()),
            "submission_sha256": sha256_file(submission_path),
        },
        "leaderboard_score_claim": False,
    }
    results_path = args.out_dir / "results.json"
    _atomic_json(results_path, _json_ready(results))

    input_files = [
        Path(__file__).resolve(),
        args.preregister,
        PROJECT_DIR / "scripts" / "run_turbine_scada_power.py",
        PROJECT_DIR / "tests" / "test_turbine_scada_power.py",
        args.canonical_dir / "stage1_lock.json",
        args.canonical_dir / "stage1_results.json",
        args.canonical_dir / "stage2_promotion_lock.json",
        args.canonical_dir / "stage2_results.json",
        args.raw_dir / "train" / "train_labels.csv",
        scada_path,
        args.raw_dir / "info.xlsx",
        sample_path,
        args.cache_dir / f"{GROUP}_weather_train.parquet",
        test_weather_path,
        baseline_path,
    ]
    output_files = [
        model_path,
        raw_prediction_path,
        candidate_path,
        submission_path,
        results_path,
    ]
    manifest = make_manifest(
        artifact_type="turbine_scada_power_attack_only",
        parameters={
            "selection": LOCKED_CANDIDATE,
            "selection_unsafe": True,
            "strict_adoption": False,
            "additional_selection": False,
        },
        input_files=input_files,
        output_files=output_files,
        results={
            "submission_sha256": results["artifacts"]["submission_sha256"],
            "stage2_deltas": quarter_deltas,
            "reload_repredict_bit_exact": reload_bit_exact,
            "strict_adoption": False,
            "leaderboard_score_claim": False,
        },
        project_dir=PROJECT_DIR,
    )
    manifest_path = args.out_dir / "manifest.json"
    write_json_atomic(manifest_path, manifest, overwrite=False)
    print(
        json.dumps(
            {
                "submission": str(submission_path),
                "sha256": results["artifacts"]["submission_sha256"],
                "selection_unsafe": True,
                "strict_adoption": False,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

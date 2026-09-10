"""Build the frozen G2-only 2024-posthoc direct-interval rescue."""

from __future__ import annotations

import argparse
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

from scripts import run_direct_interval_selective_transfer as transfer  # noqa: E402
from scripts import run_ficr_bayes_decision_strict as strict  # noqa: E402
from scripts import run_shared_q07_multiseed as shared  # noqa: E402
from src.direct_interval_probability import DirectIntervalProbabilityModel  # noqa: E402
from src.manifest import describe_file, git_state, package_versions, sha256_file, utc_now  # noqa: E402
from src.metric import CAPACITY_KWH, TARGET_COLS  # noqa: E402


CONFIG_SHA = "5cf54b2aad90ca8d2edc70dbc891eeafe8dc0a70804925354d33ebca64e1b45f"
GROUP = "kpx_group_2"
IDENTITY_GROUPS = ("kpx_group_1", "kpx_group_3")
SPEC = (0.98, 0.01)
BASELINES = ("primary_v3", "interaction_v4")
SEGMENTS = ("full", "H1", "H2", "Q1", "Q2", "Q3", "Q4")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prescore", "final"), required=True)
    parser.add_argument("--raw-dir", type=Path, default=Path(r"data/local/open"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/cache"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/direct_interval_g2_posthoc_rescue_preregister_v4.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/postgate/direct_interval_g2_posthoc_rescue_v4"),
    )
    return parser.parse_args(argv)


def _verify_file(spec: Mapping[str, Any], *, base: Path = PROJECT_DIR) -> Path:
    path = Path(str(spec["path"]))
    if not path.is_absolute():
        path = base / path
    if not path.is_file():
        raise FileNotFoundError(path)
    if "bytes" in spec and path.stat().st_size != int(spec["bytes"]):
        raise AssertionError(f"file size changed: {path}")
    if sha256_file(path) != str(spec["sha256"]):
        raise AssertionError(f"file hash changed: {path}")
    return path


def _verify_config(path: Path) -> dict[str, Any]:
    if sha256_file(path) != CONFIG_SHA:
        raise AssertionError("rescue preregistration hash changed")
    sidecar = path.with_suffix(".sha256")
    expected = f"{CONFIG_SHA}  {path.name}\n"
    if sidecar.read_text(encoding="utf-8") != expected:
        raise AssertionError("rescue preregistration sidecar changed")
    config = json.loads(path.read_text(encoding="utf-8"))
    candidate = config["immutable_rescue_candidate"]
    registered = candidate[GROUP]
    if (float(registered["scale_factor"]), float(registered["utility_advantage_margin"])) != SPEC:
        raise AssertionError("G2 factor/margin changed")
    if registered["lookup"].split()[0] != "row-wise":
        raise AssertionError("G2 lookup changed")
    if not all(candidate[group].get("identity") is True for group in IDENTITY_GROUPS):
        raise AssertionError("identity group contract changed")
    if config["risk_classification"]["posthoc_2024_rescue"] is not True:
        raise AssertionError("posthoc rescue disclosure changed")
    for spec in config["lineage"].values():
        _verify_file(spec)
    for key, value in config["locked_2024_artifacts"].items():
        if key in BASELINES:
            for item in value.values():
                _verify_file(item)
        else:
            _verify_file(value)
    return config


def _copy_exclusive(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst)


def _snapshot(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "runner": Path(__file__).resolve(),
        "model": PROJECT_DIR / "src/direct_interval_probability.py",
        "selective_rule": PROJECT_DIR / "src/direct_interval_selective_scale.py",
        "transfer_helper": PROJECT_DIR / "scripts/run_direct_interval_selective_transfer.py",
        "strict_helper": PROJECT_DIR / "scripts/run_ficr_bayes_decision_strict.py",
        "shared_helper": PROJECT_DIR / "scripts/run_shared_q07_multiseed.py",
        "metric": PROJECT_DIR / "src/metric.py",
        "manifest": PROJECT_DIR / "src/manifest.py",
        "config": args.config.resolve(),
        "config_sidecar": args.config.with_suffix(".sha256").resolve(),
        "runner_test": PROJECT_DIR / "tests/test_direct_interval_g2_posthoc_rescue.py",
    }
    return {name: shared._snapshot_file(path) for name, path in paths.items()}


def _bits(values: pd.Series) -> np.ndarray:
    array = values.to_numpy(dtype=np.float64, copy=False)
    return np.ascontiguousarray(array).view(np.uint64)


def assert_identity_bits(candidate: pd.DataFrame, baseline: pd.DataFrame) -> None:
    if not candidate.index.equals(baseline.index):
        raise AssertionError("candidate/baseline index differs")
    for group in IDENTITY_GROUPS:
        if not np.array_equal(_bits(candidate[group]), _bits(baseline[group])):
            raise AssertionError(f"{group} is not float64 bit-exact identity")


def _close_exact(actual: float, expected: float, name: str) -> None:
    if not np.isclose(float(actual), float(expected), rtol=0.0, atol=2e-15):
        raise AssertionError(f"registered value differs: {name}: {actual} != {expected}")


def _rebuild_2024_evidence(
    config: Mapping[str, Any], out_dir: Path, labels: pd.DataFrame
) -> dict[str, Any]:
    surfaces = config["locked_2024_artifacts"]["shared_g2_surface"]
    surface_path = _verify_file(surfaces)
    with np.load(surface_path) as payload:
        utility = np.asarray(payload["utility"], dtype=np.float64)
    expected = config["frozen_2024_posthoc_rescue_evidence"]
    segments = transfer._segments(2024)
    baseline_results: dict[str, Any] = {}
    paths: list[Path] = []
    for baseline_name in BASELINES:
        registered = config["locked_2024_artifacts"][baseline_name]
        baseline = pd.read_parquet(_verify_file(registered["baseline"]))
        original_v2 = pd.read_parquet(_verify_file(registered["v2_candidate"]))
        stored_detail = pd.read_parquet(_verify_file(registered["g2_diagnostics"]))
        for frame in (baseline, original_v2, stored_detail):
            frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
        selected, rebuilt_detail = transfer._apply_spec(
            baseline[GROUP], utility, group=GROUP, spec=SPEC
        )
        if not np.array_equal(_bits(selected), _bits(original_v2[GROUP])):
            raise AssertionError(f"stored v2 G2 candidate formula differs: {baseline_name}")
        for column in rebuilt_detail.columns:
            left = rebuilt_detail[column].to_numpy()
            right = stored_detail[column].to_numpy()
            if left.dtype.kind == "b":
                equal = np.array_equal(left, right)
            else:
                equal = np.array_equal(
                    np.ascontiguousarray(left, dtype=np.float64).view(np.uint64),
                    np.ascontiguousarray(right, dtype=np.float64).view(np.uint64),
                )
            if not equal:
                raise AssertionError(f"stored diagnostics differ: {baseline_name}/{column}")
        candidate = baseline.copy()
        candidate[GROUP] = selected
        assert_identity_bits(candidate, baseline)
        base_path = out_dir / f"rescue_2024/{baseline_name}/baseline.parquet"
        candidate_path = out_dir / f"rescue_2024/{baseline_name}/candidate.parquet"
        detail_path = out_dir / f"rescue_2024/{baseline_name}/g2_diagnostics.parquet"
        strict._atomic_parquet(baseline, base_path)
        strict._atomic_parquet(candidate, candidate_path)
        strict._atomic_parquet(rebuilt_detail, detail_path)
        assert_identity_bits(pd.read_parquet(candidate_path), pd.read_parquet(base_path))
        paths.extend((base_path, candidate_path, detail_path))

        group = strict._comparison(
            labels.loc[segments["full"], GROUP],
            baseline[GROUP],
            candidate[GROUP],
            GROUP,
            {name: segments[name] for name in SEGMENTS},
        )
        mixed = transfer._mixed_comparisons(
            labels.loc[segments["full"]], baseline, candidate, segments
        )
        for name in SEGMENTS:
            _close_exact(
                group[name]["delta"],
                expected["g2_group_delta_total_score"][baseline_name][name],
                f"{baseline_name}/G2/{name}",
            )
            _close_exact(
                mixed[name]["delta_total_score"],
                expected["mixed_delta_total_score"][baseline_name][name],
                f"{baseline_name}/mixed/{name}",
            )
            if not group[name]["delta"] > 0 or not mixed[name]["delta_total_score"] > 0:
                raise AssertionError(f"rescue strict-positive gate failed: {baseline_name}/{name}")
        components = expected["mixed_full_component_deltas"][baseline_name]
        _close_exact(
            mixed["full"]["delta_one_minus_nmae"], components["one_minus_nmae"],
            f"{baseline_name}/full/NMAE",
        )
        _close_exact(
            mixed["full"]["delta_ficr"], components["ficr"],
            f"{baseline_name}/full/FICR",
        )
        if mixed["full"]["delta_one_minus_nmae"] < 0 or mixed["full"]["delta_ficr"] < 0:
            raise AssertionError("full component gate failed")
        baseline_results[baseline_name] = {
            "g2_group_comparisons": group,
            "mixed_comparisons": mixed,
            "g2_selected_rows": int(rebuilt_detail["gate"].sum()),
            "g1_g3_float64_identity_bits": True,
            "all_registered_gates_passed": True,
        }
    return {"baseline_results": baseline_results, "outputs": [describe_file(p) for p in paths]}


def _read_labels(config: Mapping[str, Any]) -> pd.DataFrame:
    spec = config["final_2025_contract"]["labels"]
    path = _verify_file(spec)
    labels = strict._read_full_labels(path)
    if len(labels) != int(spec["rows"]):
        raise AssertionError("label row count changed")
    return labels


def run_prescore(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True)
    _copy_exclusive(args.config, args.out_dir / "preregister.json")
    _copy_exclusive(args.config.with_suffix(".sha256"), args.out_dir / "preregister.sha256")
    source_snapshot = _snapshot(args)
    labels = _read_labels(config)
    evidence = _rebuild_2024_evidence(config, args.out_dir, labels)
    evidence_path = args.out_dir / "rescue_2024_results.json"
    strict._write_json(
        evidence_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "posthoc_2024_rescue": True,
            "selection_unsafe": True,
            "independent_validation_claim": False,
            **evidence,
        },
    )
    train_spec = config["final_2025_contract"]["fit_weather"]
    context = transfer._read_context_cache(
        _verify_file(train_spec),
        expected_sha=str(train_spec["sha256"]),
        expected_bytes=int(train_spec["bytes"]),
    )
    if not context.index.equals(labels.index):
        raise AssertionError("full G2 context/label index differs")
    model = DirectIntervalProbabilityModel().fit(
        context, labels[GROUP], capacity_kwh=CAPACITY_KWH[GROUP]
    )
    model_path = args.out_dir / "final/model/kpx_group_2.joblib"
    strict._atomic_joblib(model, model_path)
    reloaded: DirectIntervalProbabilityModel = joblib.load(model_path)
    if reloaded.metadata() != model.metadata():
        raise AssertionError("reloaded model metadata differs")
    lock_path = args.out_dir / "final_prescore_lock.json"
    strict._write_json(
        lock_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "config_sha256": CONFIG_SHA,
            "rescue_2024_results": describe_file(evidence_path),
            "final_g2_model": describe_file(model_path),
            "model_metadata": model.metadata(),
            "source_config_snapshot": source_snapshot,
            "immutable_final_rule": {"group": GROUP, "factor": SPEC[0], "margin": SPEC[1], "lookup": "linear", "identity_groups": list(IDENTITY_GROUPS)},
            "registered_2025_input_identities": {
                key: config["final_2025_contract"][key]
                for key in ("application_weather", "baseline", "sample")
            },
            "test_weather_value_cells_parsed_before_lock": 0,
            "final_baseline_prediction_value_cells_parsed_before_lock": 0,
            "sample_value_bytes_read_before_lock": 0,
            "posthoc_2024_rescue": True,
            "selection_unsafe": True,
        },
    )
    print(f"final prescore lock sha256: {sha256_file(lock_path)}", flush=True)
    print("2025 test-weather, baseline and sample values remain unparsed", flush=True)


def _validate_csv(
    csv_path: Path, sample: pd.DataFrame, candidate: pd.DataFrame
) -> dict[str, Any]:
    raw = csv_path.read_bytes()
    if raw[:3] != b"\xef\xbb\xbf":
        raise AssertionError("CSV lacks UTF-8 BOM")
    text = pd.read_csv(csv_path, encoding="utf-8-sig", dtype="string")
    if tuple(text.columns) != tuple(sample.columns) or len(text) != 8760:
        raise AssertionError("CSV schema/row count differs")
    if not text[["forecast_id", "forecast_kst_dtm"]].equals(
        sample[["forecast_id", "forecast_kst_dtm"]].astype("string")
    ):
        raise AssertionError("CSV identifiers/timestamps differ")
    for group in TARGET_COLS:
        expected = candidate[group].map(lambda value: f"{float(value):.6f}").astype("string")
        if not text[group].equals(expected.reset_index(drop=True)):
            raise AssertionError(f"six-decimal text roundtrip differs: {group}")
    numeric = text.loc[:, list(TARGET_COLS)].astype(np.float64).to_numpy()
    if not np.isfinite(numeric).all():
        raise AssertionError("CSV contains non-finite values")
    for position, group in enumerate(TARGET_COLS):
        if np.any(numeric[:, position] < 0.0) or np.any(
            numeric[:, position] > 1.02 * CAPACITY_KWH[group] + 5e-7
        ):
            raise AssertionError(f"CSV capacity bounds failed: {group}")
    return {
        "utf8_bom": True,
        "rows": len(text),
        "columns_exact": True,
        "sample_ids_timestamps_exact": True,
        "six_decimal_text_roundtrip": True,
        "finite": True,
        "capacity_bounds": True,
        "sha256": sha256_file(csv_path),
    }


def _write_manifest(
    args: argparse.Namespace, config: Mapping[str, Any], validations: Mapping[str, Any]
) -> None:
    files = sorted(
        (path for path in args.out_dir.rglob("*") if path.is_file() and path.name != "manifest.json"),
        key=lambda path: path.relative_to(args.out_dir).as_posix(),
    )
    manifest = {
        "schema_version": 1,
        "artifact_type": "direct_interval_g2_posthoc_rescue_v4",
        "created_utc": utc_now(),
        "config_sha256": CONFIG_SHA,
        "risk": {
            "posthoc_2023_selected": True,
            "posthoc_2024_rescue": True,
            "selection_unsafe": True,
            "strict_final_isolation": False,
            "private_champion": False,
            "leaderboard_score_claim": False,
        },
        "immutability": {
            "G2_factor": SPEC[0],
            "G2_margin": SPEC[1],
            "lookup": "linear",
            "G1_G3_float64_identity_bits": True,
            "no_new_tuning_grid": True,
        },
        "incidents": config["incident_ledger"],
        "final_prescore_lock": describe_file(args.out_dir / "final_prescore_lock.json"),
        "rescue_2024_results": describe_file(args.out_dir / "rescue_2024_results.json"),
        "final_results": describe_file(args.out_dir / "final_results.json"),
        "validations": dict(validations),
        "source_config_snapshot": _snapshot(args),
        "outputs": [describe_file(path) for path in files],
        "output_count_excluding_manifest": len(files),
        "runtime": {"packages": package_versions()},
        "git": git_state(PROJECT_DIR),
    }
    strict._write_json(args.out_dir / "manifest.json", manifest)


def run_final(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    lock_path = args.out_dir / "final_prescore_lock.json"
    if not lock_path.is_file():
        raise AssertionError("final prescore lock missing")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    shared._assert_snapshot_equal(lock["source_config_snapshot"], _snapshot(args), name="rescue source/config")
    if lock["config_sha256"] != CONFIG_SHA:
        raise AssertionError("prescore config hash differs")

    final = config["final_2025_contract"]
    test_spec = final["application_weather"]
    context = transfer._read_test_context(
        _verify_file(test_spec),
        expected_sha=str(test_spec["sha256"]),
        expected_bytes=int(test_spec["bytes"]),
    )
    baseline_path = _verify_file(final["baseline"])
    baseline = pd.read_parquet(baseline_path)
    baseline.index = pd.DatetimeIndex(baseline.index, name="forecast_kst_dtm")
    if not baseline.index.equals(context.index) or tuple(baseline.columns) != TARGET_COLS:
        raise AssertionError("final baseline schema/index changed")
    baseline = baseline.astype(np.float64)
    if not np.isfinite(baseline.to_numpy()).all():
        raise AssertionError("final baseline contains non-finite values")
    sample_path = _verify_file(final["sample"], base=Path("/"))
    sample = pd.read_csv(
        sample_path,
        encoding="utf-8-sig",
        dtype={"forecast_id": "string", "forecast_kst_dtm": "string"},
    )
    if tuple(sample.columns) != ("forecast_id", "forecast_kst_dtm", *TARGET_COLS) or len(sample) != 8760:
        raise AssertionError("sample schema/rows changed")
    sample_time = pd.DatetimeIndex(pd.to_datetime(sample["forecast_kst_dtm"]), name="forecast_kst_dtm")
    if not sample_time.equals(context.index):
        raise AssertionError("sample time index changed")

    model_path = args.out_dir / "final/model/kpx_group_2.joblib"
    model: DirectIntervalProbabilityModel = joblib.load(model_path)
    arrays = transfer._surface_arrays(model, context)
    surface_path = args.out_dir / "final/surface/kpx_group_2.npz"
    transfer._atomic_npz(surface_path, arrays)
    with np.load(surface_path) as saved:
        transfer._assert_arrays_equal(arrays, {name: saved[name] for name in saved.files})
    reloaded: DirectIntervalProbabilityModel = joblib.load(model_path)
    transfer._assert_arrays_equal(arrays, transfer._surface_arrays(reloaded, context))
    selected, diagnostics = transfer._apply_spec(
        baseline[GROUP], arrays["utility"], group=GROUP, spec=SPEC
    )
    candidate = baseline.copy()
    candidate[GROUP] = selected
    assert_identity_bits(candidate, baseline)
    if not np.array_equal(
        diagnostics["gate"].to_numpy(),
        diagnostics["utility_advantage"].to_numpy() > SPEC[1],
    ):
        raise AssertionError("G2 strict gate formula differs")
    expected_g2 = np.where(
        diagnostics["gate"].to_numpy(),
        SPEC[0] * baseline[GROUP].to_numpy(),
        baseline[GROUP].to_numpy(),
    )
    expected_g2 = np.clip(expected_g2, 0.0, 1.02 * CAPACITY_KWH[GROUP])
    if not np.array_equal(
        np.ascontiguousarray(expected_g2).view(np.uint64), _bits(candidate[GROUP])
    ):
        raise AssertionError("G2 candidate formula differs")

    diagnostics_path = args.out_dir / "final/g2_diagnostics.parquet"
    predictions_path = args.out_dir / "final/predictions.parquet"
    strict._atomic_parquet(diagnostics, diagnostics_path)
    strict._atomic_parquet(candidate, predictions_path)
    readback_prediction = pd.read_parquet(predictions_path).astype(np.float64)
    if not readback_prediction.index.equals(candidate.index):
        raise AssertionError("prediction parquet index differs")
    assert_identity_bits(readback_prediction, baseline)
    for group in TARGET_COLS:
        if not np.array_equal(_bits(readback_prediction[group]), _bits(candidate[group])):
            raise AssertionError(f"prediction parquet differs: {group}")

    submission = sample.copy()
    for group in TARGET_COLS:
        submission[group] = candidate[group].to_numpy(dtype=np.float64)
    csv_path = args.out_dir / str(final["candidate_csv"])
    strict._atomic_csv(submission, csv_path)
    csv_validation = _validate_csv(csv_path, sample, candidate.reset_index(drop=True))
    validations = {
        "config_hash_verified_before_2025_value_parse": True,
        "final_prescore_lock_verified_before_2025_value_parse": True,
        "model_reload_surface_bit_exact": True,
        "surface_npz_bit_exact": True,
        "G2_formula_exact": True,
        "G2_strict_gate_exact": True,
        "G2_selected_rows": int(diagnostics["gate"].sum()),
        "G1_G3_in_memory_float64_identity_bits": True,
        "G1_G3_parquet_float64_identity_bits": True,
        "prediction_rows": len(candidate),
        "prediction_time_index_exact": candidate.index.equals(context.index),
        "csv": csv_validation,
    }
    strict._write_json(
        args.out_dir / "final_results.json",
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "executed": True,
            "posthoc_2024_rescue": True,
            "selection_unsafe": True,
            "strict_final_isolation": False,
            "leaderboard_score_claim": False,
            "baseline": describe_file(baseline_path),
            "sample": describe_file(sample_path),
            "model": describe_file(model_path),
            "surface": describe_file(surface_path),
            "diagnostics": describe_file(diagnostics_path),
            "predictions": describe_file(predictions_path),
            "submission": describe_file(csv_path),
            "validations": validations,
        },
    )
    _write_manifest(args, config, validations)
    print(f"CSV: {csv_path} sha256={sha256_file(csv_path)}", flush=True)
    print(f"manifest sha256: {sha256_file(args.out_dir / 'manifest.json')}", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = _verify_config(args.config)
    if args.stage == "prescore":
        run_prescore(args, config)
    else:
        run_final(args, config)


if __name__ == "__main__":
    main()

"""Build the single preregistered KMA D-1 11 KST WSD final submission.

Production execution is fail closed.  The frozen model-gate result is validated
before any 2025 artifact is opened.  Exactly its selected R1/R2 recipe is refit
per group on the registered 2022--2024 rows and applied once to 2025.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_kma_d1_1100_wsd_model_gate as gate_runner  # noqa: E402
from src.metric import CAPACITY_KWH, TARGET_COLS  # noqa: E402
from src.weather_quantile import (  # noqa: E402
    BayesActionConfig,
    TrainOnlyMedianTransform,
    WeatherQuantileSurface,
)


GROUPS = tuple(TARGET_COLS)
RECIPES = gate_runner.RECIPES
TRAIN_INDEX = gate_runner.expected_year_index(2022).append(
    gate_runner.expected_year_index(2023)
).append(gate_runner.expected_year_index(2024))
TEST_INDEX = gate_runner.expected_year_index(2025)

WSD_ROOT = ROOT / "artifacts/final_submission_sprint_20260812/kma_d1_1100/wsd_typ01"
PREGATE_KMA = WSD_ROOT / "materialized/KMA_D1_1100_WSD_HOURLY_2022_2024.parquet"
PREGATE_MANIFEST = WSD_ROOT / "materialized/MANIFEST.json"
KMA_2025 = WSD_ROOT / "materialized/KMA_D1_1100_WSD_HOURLY_2025.parquet"
KMA_2025_MANIFEST = WSD_ROOT / "materialized/MANIFEST_2025.json"
GATE_PASS = WSD_ROOT / "MODEL_GATE_PASS.json"
GATE_RESULTS = WSD_ROOT / "model_gate/MODEL_GATE_RESULTS.json"

LABELS = Path(r"data/local/open/train/train_labels.csv")
SAMPLE = Path(r"data/local/open/sample_submission.csv")
BASELINE_2025 = ROOT / "artifacts/postgate/public_adaptive_scale097_g2_delta_v2/final/A_plain_scale097.parquet"
FINAL_ROOT = ROOT / "artifacts/final_submission_sprint_20260812"
FINAL_CSV = FINAL_ROOT / "04_OFFICIAL_KMA_D1_WSD_FINAL.csv"
FINAL_MANIFEST = FINAL_ROOT / "04_OFFICIAL_KMA_D1_WSD_FINAL.manifest.json"
FINAL_MODEL_ROOT = WSD_ROOT / "final_model"
FINAL_EXECUTION_CONFIG = ROOT / "configs/kma_d1_1100_wsd_finalizer_execution_v1.json"

PREREGS = (
    ("configs/kma_d1_1100_wsd_sprint_preregister_v1.json", 4312, "2353e2b39ef6d64645393af749fbf7ea13e3e89f1b6e4f35698b02b59efe389f"),
    ("configs/kma_d1_1100_wsd_sprint_preregister_v2_amendment.json", 21034, "4e7d99b1d7ea67fa724d0956300a131d3b00a7b34cb777d6ce68eef57a63c99d"),
    ("configs/kma_d1_1100_wsd_sprint_preregister_v3_decoder_errata.json", 3606, "747cca0d49d28c572c35ad66c8a397e9e5467af653d3df2f61189659eed80ea7"),
    ("configs/kma_d1_1100_wsd_sprint_preregister_v4_anchor_amendment.json", 8943, "0fe5f310c727e31d2fbbd786ef55ebb02990b4903ba18d5d91dbfe0c31b7fa4d"),
    ("configs/kma_d1_1100_wsd_sprint_preregister_v5_ordering_errata.json", 1391, "140d78ace4fe675d597fb2c2a8feca6b70fb0ca681b317a21d4f0c62912b0b60"),
    ("configs/kma_d1_1100_wsd_sprint_preregister_v6_selection_cardinality_errata.json", 3688, "40ec42aa5c17a29e6154ffe55e2eeaa0b94820cff3c9c534017a6fdb201e9089"),
)

BOUND = {
    "gate_runner": ("scripts/run_kma_d1_1100_wsd_model_gate.py", 30530, "ac147d60c354505070cee3df32c94a072fdddc53a7097427e0bd43bb686169f9"),
    "gate_execution": ("configs/kma_d1_1100_wsd_model_gate_execution_v1.json", 1136, "4dd466193c31343f921ee1a49dbbfc09b3ec14183540686d853f5a503f64c80a"),
    "materializer": ("scripts/materialize_kma_d1_1100_wsd_anchors.py", 20705, "5a82407fd5137151fcd7fdff8c8669a65ce221227a9dce1073d606af68614ef3"),
    "labels": (LABELS, 1138967, "47bb64252195cf4734e67394d6e50485f27a608def3b5a8791fcc7674bbceb03"),
    "sample": (SAMPLE, 359229, "c925d2066a834f937f8091ed55acfe50ff86c8be4745b52c3adc95b056c5aaaa"),
    "weather_quantile": ("src/weather_quantile.py", 13206, "005fcc8069bc707b5d6c270bb640897696e2eb04e050a5ad81a251433041075d"),
    "metric": ("src/metric.py", 7248, "555950e6892a808d9091b4e6749128b9f96888ce0a83270b6959a4e927dc5f1d"),
    "baseline_2022": ("artifacts/postgate/common_cf_forward/oof/dev2022_selected_inner_oof.parquet", 172215, "ce781f49f07146a982891ecc156b389aae3c7c4b38e4c0a35031649e55bd81d2"),
    "baseline_2023": ("artifacts/postgate/weather_quantile_bayes/oof/stage1_baseline_2023.parquet", 292085, "040d4fe663cecde856c4c2229d3d98b472954cc2c2c541c787cfbc3af85aa9b6"),
    "baseline_2024": ("artifacts/postgate/public_adaptive_scale097_g2_delta_v2/diagnostic_2024/A_scale097.parquet", 335807, "93d8e2b7d19971ba46a29d548e77ef4f85553bbbdaad2b5983a5b6f1d84e406c"),
    "baseline_2025": (BASELINE_2025, 334789, "85ad8c63eaf7bff4b0390fb8c42dbb31c22acd5e7a64fbec350d9c3878818621"),
    "baseline_01_csv": ("artifacts/final_submission_sprint_20260812/01_OFFICIAL_ONLY_SAFE.csv", 624176, "e8f848b9a08c44b2552367f8e83ae4c9e132c8cf416ea8c437fd7a7e26314f58"),
}

WEATHER_TRAIN = gate_runner.WEATHER
WEATHER_TEST = {
    "kpx_group_1": (ROOT / "artifacts/cache/kpx_group_1_weather_test.parquet", 26037151, "8de6fe0ba96ee5c4d0bae16c1b57bc761863152064fda490caf048a5b6e87771"),
    "kpx_group_2": (ROOT / "artifacts/cache/kpx_group_2_weather_test.parquet", 26031426, "3450c23bd1e6a8270f416956d65f7df0a0265d3d67ea30c1b69dd4f214b43e99"),
    "kpx_group_3": (ROOT / "artifacts/cache/kpx_group_3_weather_test.parquet", 26038173, "54ee1011e1c9f328347500fe1f6113919f542bf7697126201cedc16bfdec2ed6"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def absolute_identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def relative_identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def resolve_bound(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def verify_bound(value: tuple[str | Path, int, str], name: str) -> dict[str, Any]:
    raw_path, size, digest = value
    path = resolve_bound(raw_path)
    observed = absolute_identity(path)
    if observed["bytes"] != size or observed["sha256"] != digest:
        raise RuntimeError(f"bound identity drift: {name}")
    return observed


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, indent=2, default=str) + "\n").encode("ascii")


def exclusive_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}.{time.time_ns()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if temporary.read_bytes() != payload:
            raise RuntimeError(f"temporary write verification failed: {path.name}")
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    if path.read_bytes() != payload:
        raise RuntimeError(f"write verification failed: {path.name}")


def verify_execution_config() -> dict[str, Any]:
    config = json.loads(FINAL_EXECUTION_CONFIG.read_text(encoding="ascii"))
    if config.get("schema_version") != 1 or config.get("status") != "FROZEN_BEFORE_FINALIZER_PRODUCTION_EXECUTION":
        raise RuntimeError("finalizer execution config status/schema mismatch")
    if config.get("script_identity") != relative_identity(Path(__file__)):
        raise RuntimeError("finalizer execution config does not bind current script")
    expected_chain = [relative_identity(resolve_bound(spec[0])) for spec in PREREGS]
    if config.get("preregistration_chain") != expected_chain:
        raise RuntimeError("finalizer execution config preregistration chain mismatch")
    expected_bound = {
        "gate_runner": relative_identity(resolve_bound(BOUND["gate_runner"][0])),
        "gate_execution": relative_identity(resolve_bound(BOUND["gate_execution"][0])),
        "materializer": relative_identity(resolve_bound(BOUND["materializer"][0])),
    }
    if config.get("implementation_bindings") != expected_bound:
        raise RuntimeError("finalizer implementation binding mismatch")
    if config.get("actual_argv") != [".venv/Scripts/python.exe", "-B", "scripts/build_kma_d1_1100_wsd_final_submission.py"]:
        raise RuntimeError("finalizer argv binding mismatch")
    if config.get("output_csv") != FINAL_CSV.relative_to(ROOT).as_posix() or config.get("output_manifest") != FINAL_MANIFEST.relative_to(ROOT).as_posix():
        raise RuntimeError("finalizer output binding mismatch")
    if config.get("production_attempts") != 1 or config.get("public_score_access") is not False:
        raise RuntimeError("finalizer execution-attempt/provenance binding mismatch")
    return relative_identity(FINAL_EXECUTION_CONFIG)


def validate_gate_before_any_2025_read() -> tuple[str, dict[str, Any]]:
    """This must be the first production operation; every path opened is pre-2025."""
    bound: dict[str, Any] = {}
    for index, spec in enumerate(PREREGS, start=1):
        bound[f"preregister_v{index}"] = verify_bound(spec, f"preregister_v{index}")
    bound["finalizer_execution"] = verify_execution_config()
    bound["gate_runner"] = verify_bound(BOUND["gate_runner"], "gate_runner")
    bound["gate_execution"] = verify_bound(BOUND["gate_execution"], "gate_execution")
    execution = json.loads(resolve_bound(BOUND["gate_execution"][0]).read_text(encoding="utf-8"))
    if execution.get("script_identity") != relative_identity(resolve_bound(BOUND["gate_runner"][0])):
        raise RuntimeError("gate execution does not bind the frozen runner")
    gate = json.loads(GATE_PASS.read_text(encoding="utf-8"))
    if set(gate) != {"all_gates_pass", "selected_recipe", "results"} or gate.get("all_gates_pass") is not True:
        raise RuntimeError("exact MODEL_GATE_PASS is absent")
    if gate.get("results") != absolute_identity(GATE_RESULTS):
        raise RuntimeError("MODEL_GATE_PASS result identity mismatch")
    result = json.loads(GATE_RESULTS.read_text(encoding="utf-8"))
    if result.get("public_scores_used") is not False or result.get("kma_2025_read") is not False or result.get("csv_created") is not False:
        raise RuntimeError("model gate provenance is not blind/pre-2025")
    evaluations = result.get("evaluations")
    if not isinstance(evaluations, dict) or set(evaluations) != set(RECIPES):
        raise RuntimeError("model gate recipe census mismatch")
    passing = [recipe for recipe in RECIPES if evaluations[recipe]["gate_summary"].get("all_gates_pass") is True]
    if not passing or result.get("passing_recipes") != passing:
        raise RuntimeError("zero passing recipes or passing-recipe record mismatch")
    expected = max(
        passing,
        key=lambda recipe: (
            evaluations[recipe]["gate_summary"]["minimum_fold_delta_score"],
            evaluations[recipe]["gate_summary"]["mean_fold_delta_score"],
            recipe == RECIPES[0],
        ),
    )
    if result.get("all_gates_pass") is not True or result.get("selected_recipe") != expected or gate.get("selected_recipe") != expected:
        raise RuntimeError("selected deployment recipe differs from frozen selector")
    bound["model_gate_results"] = absolute_identity(GATE_RESULTS)
    bound["model_gate_pass"] = absolute_identity(GATE_PASS)
    return expected, bound


def load_kma_2025_after_gate() -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = json.loads(KMA_2025_MANIFEST.read_text(encoding="ascii"))
    if manifest.get("mode") != "2025" or manifest.get("rows") != 8760:
        raise RuntimeError("2025 KMA materializer manifest contract mismatch")
    if manifest.get("parquet") != relative_identity(KMA_2025):
        raise RuntimeError("2025 KMA parquet is not exactly bound by MANIFEST_2025")
    gate_id = relative_identity(GATE_PASS)
    summary_ids = manifest.get("source_evidence", {}).get("summary_identities", [])
    if gate_id not in summary_ids:
        raise RuntimeError("MANIFEST_2025 does not bind the validated MODEL_GATE_PASS")
    frame = pd.read_parquet(KMA_2025, engine="pyarrow")
    if "forecast_kst_dtm" in frame.columns:
        frame = frame.set_index("forecast_kst_dtm", verify_integrity=True)
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    required = {
        "operating_year", "operating_day", "lead_hour", "issue_base_kst",
        "wsd_cell_94_121", "wsd_cell_94_122", *[f"wsd_{g}" for g in GROUPS],
    }
    if not required.issubset(frame.columns) or not frame.index.equals(TEST_INDEX):
        raise RuntimeError("2025 KMA schema/index mismatch")
    clock = frame.index - pd.Timedelta(hours=1)
    if not np.array_equal(frame["operating_year"].to_numpy(dtype=int), clock.year):
        raise RuntimeError("2025 KMA operating-year mismatch")
    if not np.array_equal(frame["lead_hour"].to_numpy(dtype=int), clock.hour.to_numpy() + 1):
        raise RuntimeError("2025 KMA lead mismatch")
    issue = pd.DatetimeIndex(pd.to_datetime(frame["issue_base_kst"], errors="raise"))
    expected_issue = pd.DatetimeIndex(clock.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=11))
    if not issue.equals(expected_issue):
        raise RuntimeError("2025 KMA issue mapping mismatch")
    a = frame["wsd_cell_94_121"].to_numpy(dtype=float)
    b = frame["wsd_cell_94_122"].to_numpy(dtype=float)
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(a < 0) or np.any(a > 75) or np.any(b < 0) or np.any(b > 75):
        raise RuntimeError("2025 KMA selected-cell coverage/plausibility failure")
    expected_groups = {GROUPS[0]: 0.5 * a + 0.5 * b, GROUPS[1]: (5 / 6) * a + (1 / 6) * b, GROUPS[2]: a}
    for group, values in expected_groups.items():
        if not np.allclose(frame[f"wsd_{group}"], values, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"2025 KMA group formula mismatch: {group}")
    return frame, {"parquet": absolute_identity(KMA_2025), "manifest": absolute_identity(KMA_2025_MANIFEST)}


def load_exact_series(path: Path, column: str, index: pd.DatetimeIndex) -> pd.Series:
    frame = pd.read_parquet(path, columns=[column], engine="pyarrow")
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if not frame.index.equals(index):
        raise RuntimeError(f"index mismatch: {path.name}")
    values = frame[column].astype(float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"non-finite feature: {path.name}/{column}")
    return values


def load_ws10() -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    train: dict[str, pd.Series] = {}
    test: dict[str, pd.Series] = {}
    for group in GROUPS:
        train[group] = load_exact_series(WEATHER_TRAIN[group][0], "ldaps__idw__ws10", TRAIN_INDEX)
        test[group] = load_exact_series(WEATHER_TEST[group][0], "ldaps__idw__ws10", TEST_INDEX)
    return train, test


def load_training_baselines() -> dict[int, pd.DataFrame]:
    raw22 = pd.read_parquet(resolve_bound(BOUND["baseline_2022"][0]), engine="pyarrow").astype(float)
    raw23 = pd.read_parquet(resolve_bound(BOUND["baseline_2023"][0]), engine="pyarrow").astype(float)
    raw24 = pd.read_parquet(resolve_bound(BOUND["baseline_2024"][0]), engine="pyarrow").astype(float)
    raw22 = raw22.mul(0.97).clip(lower=0.0, upper=pd.Series({g: 1.02 * CAPACITY_KWH[g] for g in raw22.columns}), axis="columns")
    raw23 = raw23.mul(0.97).clip(lower=0.0, upper=pd.Series({g: 1.02 * CAPACITY_KWH[g] for g in raw23.columns}), axis="columns")
    expected22 = pd.date_range("2022-04-01 01:00", "2023-01-01 00:00", freq="h", name="forecast_kst_dtm")
    if not raw22.index.equals(expected22) or not raw23.index.equals(gate_runner.expected_year_index(2023)) or not raw24.index.equals(gate_runner.expected_year_index(2024)):
        raise RuntimeError("registered R2 training baseline index mismatch")
    return {2022: raw22, 2023: raw23, 2024: raw24}


def load_deployment_baseline() -> pd.DataFrame:
    frame = pd.read_parquet(BASELINE_2025, engine="pyarrow").astype(float)
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if not frame.index.equals(TEST_INDEX) or tuple(frame.columns) != GROUPS or not np.isfinite(frame.to_numpy()).all():
        raise RuntimeError("full-precision deployment baseline schema/index/finiteness mismatch")
    for group in GROUPS:
        if np.any(frame[group] < 0.0) or np.any(frame[group] > 1.02 * CAPACITY_KWH[group]):
            raise RuntimeError(f"deployment baseline outside capacity bounds: {group}")
    reference = pd.read_csv(resolve_bound(BOUND["baseline_01_csv"][0]), encoding="utf-8-sig", dtype="string", keep_default_na=False)
    sample = pd.read_csv(SAMPLE, encoding="utf-8-sig", dtype="string", keep_default_na=False)
    expected_columns = ("forecast_id", "forecast_kst_dtm", *GROUPS)
    if tuple(reference.columns) != expected_columns or not reference[["forecast_id", "forecast_kst_dtm"]].equals(sample[["forecast_id", "forecast_kst_dtm"]]):
        raise RuntimeError("registered six-decimal baseline 01 identity/time mismatch")
    for group in GROUPS:
        expected = pd.Series([f"{value:.6f}" for value in frame[group].to_numpy(dtype=float)], dtype="string")
        if not reference[group].reset_index(drop=True).equals(expected):
            raise RuntimeError(f"full-precision baseline does not render exactly as baseline 01: {group}")
    return frame


def refit_selected(
    recipe: str,
    kma_train: pd.DataFrame,
    kma_test: pd.DataFrame,
    labels: pd.DataFrame,
    baseline_test: pd.DataFrame,
    ws10_train: dict[str, pd.Series],
    ws10_test: dict[str, pd.Series],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prediction = pd.DataFrame(index=TEST_INDEX, columns=GROUPS, dtype=float)
    models: dict[str, Any] = {}
    training_baselines = load_training_baselines() if recipe == RECIPES[1] else None
    for group in GROUPS:
        capacity = CAPACITY_KWH[group]
        if recipe == RECIPES[0]:
            train_x = gate_runner.feature_frame(kma_train, group, ws10_train[group], None).loc[:, gate_runner.R1_FEATURES]
            test_x = gate_runner.feature_frame(kma_test, group, ws10_test[group], None).loc[:, gate_runner.R1_FEATURES]
            surface = WeatherQuantileSurface(
                quantile_levels=(0.1, 0.25, 0.5, 0.75, 0.9),
                model_parameters=gate_runner.R1_PARAMS,
                random_state=20260812,
                n_jobs=10,
                minimum_actual_cf=0.10,
                action_config=BayesActionConfig(
                    quantile_levels=(0.1, 0.25, 0.5, 0.75, 0.9), interpolation_count=33,
                    outcome_lower_cf=0.1, outcome_upper_cf=1.2,
                    candidate_lower_cf=0.0, candidate_upper_cf=1.02, candidate_step_cf=0.01,
                ),
            ).fit(train_x, labels[group], capacity_kwh=capacity)
            action_kwh, _ = surface.predict_action(test_x, baseline_test[group])
            prediction[group] = np.clip(0.75 * baseline_test[group].to_numpy(dtype=float) + 0.25 * action_kwh.to_numpy(dtype=float), 0.0, 1.02 * capacity)
            models[group] = surface
        else:
            assert training_baselines is not None
            train_base = pd.concat([training_baselines[y][group] for y in (2022, 2023, 2024) if group in training_baselines[y].columns]).reindex(TRAIN_INDEX)
            train_all = gate_runner.feature_frame(kma_train, group, ws10_train[group], train_base)
            test_all = gate_runner.feature_frame(kma_test, group, ws10_test[group], baseline_test[group])
            actual = labels[group].astype(float)
            kma_finite = np.isfinite(train_all.loc[:, gate_runner.R2_FEATURES[:-1]].to_numpy(dtype=float)).all(axis=1)
            eligible = np.isfinite(actual) & (actual >= 0.10 * capacity) & np.isfinite(train_base) & kma_finite
            train_x = train_all.loc[eligible, gate_runner.R2_FEATURES]
            transform = TrainOnlyMedianTransform().fit(train_x)
            from lightgbm import LGBMRegressor
            model = LGBMRegressor(**gate_runner.R2_PARAMS)
            target = actual.loc[train_x.index].to_numpy(dtype=float) / capacity - train_base.loc[train_x.index].to_numpy(dtype=float) / capacity
            model.fit(transform.transform(train_x), target)
            raw = np.asarray(model.predict(transform.transform(test_all.loc[:, gate_runner.R2_FEATURES])), dtype=float)
            prediction[group] = np.clip(baseline_test[group].to_numpy(dtype=float) + 0.10 * np.clip(raw, -0.20, 0.20) * capacity, 0.0, 1.02 * capacity)
            models[group] = {"model": model, "transform": transform, "fit_rows": len(train_x)}
    if not np.isfinite(prediction.to_numpy(dtype=float)).all():
        raise RuntimeError("final candidate contains non-finite values")
    return prediction, models


def submission_payload(prediction: pd.DataFrame, sample_path: Path = SAMPLE) -> tuple[bytes, dict[str, Any]]:
    sample = pd.read_csv(sample_path, encoding="utf-8-sig", dtype="string", keep_default_na=False)
    expected_columns = ("forecast_id", "forecast_kst_dtm", *GROUPS)
    sample_index = pd.DatetimeIndex(pd.to_datetime(sample["forecast_kst_dtm"], errors="raise"), name="forecast_kst_dtm")
    if tuple(sample.columns) != expected_columns or len(sample) != len(prediction) or not sample_index.equals(prediction.index):
        raise RuntimeError("sample identity/time/column contract mismatch")
    output = sample.loc[:, ["forecast_id", "forecast_kst_dtm"]].copy()
    for group in GROUPS:
        values = prediction[group].to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.02 * CAPACITY_KWH[group]):
            raise RuntimeError(f"final capacity/finiteness contract failed: {group}")
        output[group] = values
    text = output.to_csv(index=False, float_format="%.6f", lineterminator="\n")
    payload = b"\xef\xbb\xbf" + text.encode("utf-8")
    with tempfile.TemporaryDirectory() as folder:
        replay_path = Path(folder) / "candidate.csv"
        replay_path.write_bytes(payload)
        replay = pd.read_csv(replay_path, encoding="utf-8-sig", dtype="string", keep_default_na=False)
    if tuple(replay.columns) != expected_columns or len(replay) != len(sample):
        raise RuntimeError("submission readback schema mismatch")
    if not replay[["forecast_id", "forecast_kst_dtm"]].equals(sample[["forecast_id", "forecast_kst_dtm"]]):
        raise RuntimeError("submission identifiers/timestamps differ from sample")
    decimal = re.compile(r"^-?\d+\.\d{6}$")
    for group in GROUPS:
        expected_text = pd.Series([f"{x:.6f}" for x in prediction[group].to_numpy(dtype=float)], dtype="string")
        if not replay[group].reset_index(drop=True).equals(expected_text) or not replay[group].map(lambda x: bool(decimal.fullmatch(x))).all():
            raise RuntimeError(f"six-decimal readback mismatch: {group}")
    return payload, {"rows": len(replay), "columns": list(replay.columns), "encoding": "utf-8-sig", "numeric_decimals": 6, "sample_identity_and_time_exact": True}


def synthetic_smoke() -> None:
    index = pd.date_range("2025-01-01 01:00", periods=24, freq="h", name="forecast_kst_dtm")
    sample = pd.DataFrame({"forecast_id": [f"id-{x:02d}" for x in range(24)], "forecast_kst_dtm": index.strftime("%Y-%m-%d %H:%M:%S")})
    pred = pd.DataFrame({g: np.linspace(0.0, 0.9 * CAPACITY_KWH[g], 24) for g in GROUPS}, index=index)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "sample.csv"
        sample.assign(**{g: "0" for g in GROUPS}).to_csv(path, index=False, encoding="utf-8-sig")
        payload, record = submission_payload(pred, path)
    if not payload.startswith(b"\xef\xbb\xbf") or record["rows"] != 24:
        raise RuntimeError("synthetic smoke failed")
    print("synthetic_smoke=PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-smoke", action="store_true")
    args = parser.parse_args()
    if args.synthetic_smoke:
        synthetic_smoke()
        return

    # Strict ordering: no 2025 path is opened above this validated gate.
    selected, sources = validate_gate_before_any_2025_read()
    if FINAL_CSV.exists() or FINAL_MANIFEST.exists() or FINAL_MODEL_ROOT.exists():
        raise FileExistsError("create-exclusive final output already exists")

    for name, spec in BOUND.items():
        sources[name] = verify_bound(spec, name)
    for group, spec in WEATHER_TRAIN.items():
        sources[f"weather_train_{group}"] = verify_bound(spec, f"weather_train_{group}")
    for group, spec in WEATHER_TEST.items():
        sources[f"weather_test_{group}"] = verify_bound(spec, f"weather_test_{group}")
    kma_test, sources["kma_2025"] = load_kma_2025_after_gate()
    kma_train, sources["kma_2022_2024"] = gate_runner.load_materialized(PREGATE_KMA, PREGATE_MANIFEST)
    sources["sample"] = verify_bound(BOUND["sample"], "sample")
    sources["finalizer"] = absolute_identity(Path(__file__))

    FINAL_MODEL_ROOT.mkdir(parents=True, exist_ok=False)
    source_lock = FINAL_MODEL_ROOT / "SOURCE_LOCK_BEFORE_LABEL_VALUE_DECODE_OR_FINAL_FIT.json"
    exclusive_bytes(source_lock, canonical_json({
        "schema_version": 1,
        "status": "GATE_AND_ALL_FINAL_SOURCES_LOCKED_BEFORE_LABEL_VALUE_DECODE_OR_MODEL_FIT",
        "selected_recipe": selected,
        "sources": sources,
        "public_scores_read_or_used": False,
        "candidate_count": 1,
    }))

    labels = pd.read_csv(LABELS, encoding="utf-8-sig", parse_dates=["kst_dtm"]).set_index("kst_dtm")
    labels.index = pd.DatetimeIndex(labels.index, name="forecast_kst_dtm")
    if not labels.index.equals(TRAIN_INDEX) or tuple(labels.columns) != GROUPS:
        raise RuntimeError("label schema/index mismatch")
    baseline_test = load_deployment_baseline()
    ws10_train, ws10_test = load_ws10()
    prediction, models = refit_selected(selected, kma_train, kma_test, labels, baseline_test, ws10_train, ws10_test)

    model_ids: dict[str, Any] = {}
    for group, model in models.items():
        path = FINAL_MODEL_ROOT / f"{selected}__{group}.pkl"
        exclusive_bytes(path, pickle.dumps(model, protocol=5))
        model_ids[group] = absolute_identity(path)
    prediction_path = FINAL_MODEL_ROOT / "FINAL_PREDICTION_FULL_PRECISION.parquet"
    gate_runner.exclusive_parquet(prediction_path, prediction)
    payload, csv_contract = submission_payload(prediction)
    exclusive_bytes(FINAL_CSV, payload)
    csv_record = absolute_identity(FINAL_CSV)
    replay_payload, _ = submission_payload(prediction)
    if replay_payload != FINAL_CSV.read_bytes():
        raise RuntimeError("final CSV byte readback differs")
    manifest = {
        "schema_version": 1,
        "artifact_type": "KMA_D1_1100_WSD_SINGLE_GATED_FINAL_SUBMISSION",
        "selected_recipe": selected,
        "selected_recipe_count": 1,
        "training_operating_years": [2022, 2023, 2024],
        "application_operating_year": 2025,
        "deployment_parent": "registered full-precision 01 scale097 baseline",
        "public_scores_read_or_used": False,
        "models": model_ids,
        "full_precision_prediction": absolute_identity(prediction_path),
        "source_lock": absolute_identity(source_lock),
        "csv": {**csv_record, **csv_contract},
    }
    exclusive_bytes(FINAL_MANIFEST, canonical_json(manifest))
    print(json.dumps({"selected_recipe": selected, "csv": csv_record, "manifest": absolute_identity(FINAL_MANIFEST)}, sort_keys=True))


if __name__ == "__main__":
    main()

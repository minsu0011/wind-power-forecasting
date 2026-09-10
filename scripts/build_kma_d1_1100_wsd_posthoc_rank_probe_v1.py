"""Refit the frozen R2 recipe and build the posthoc/high-risk 2025 rank probe."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import scripts.run_kma_d1_1100_wsd_model_gate_v8 as gate_wrapper  # noqa: E402
from src.metric import CAPACITY_KWH, TARGET_COLS  # noqa: E402
from src.weather_quantile import TrainOnlyMedianTransform  # noqa: E402


gate = gate_wrapper.base
GROUPS = tuple(TARGET_COLS)
TRAIN_INDEX = gate.expected_year_index(2022).append(gate.expected_year_index(2023)).append(gate.expected_year_index(2024))
TEST_INDEX = gate.expected_year_index(2025)
ROOT = PROJECT / "artifacts/final_submission_sprint_20260812/kma_d1_1100_posthoc_probe_v1"
WSD_ROOT = ROOT / "wsd_typ01"
PREREG = PROJECT / "configs/kma_d1_1100_wsd_posthoc_rank_probe_v1_preregister.json"
ACQUISITION_CONFIG = PROJECT / "configs/kma_d1_1100_wsd_posthoc_rank_probe_v1_acquisition_execution.json"
INCIDENT = PROJECT / "configs/kma_d1_1100_wsd_posthoc_rank_probe_v1_acquisition_attempt1_incident.json"
RECOVERY_ACQUISITION_CONFIG = PROJECT / "configs/kma_d1_1100_wsd_posthoc_rank_probe_v1_acquisition_recovery_execution.json"
RECOVERY_DOWNLOADER = PROJECT / "scripts/download_kma_d1_1100_wsd_posthoc_rank_probe_v1_recovery.py"
RECOVERY2_INCIDENT = PROJECT / "configs/kma_d1_1100_wsd_posthoc_rank_probe_v1_acquisition_recovery_attempt2_incident.json"
RECOVERY2_ACQUISITION_CONFIG = PROJECT / "configs/kma_d1_1100_wsd_posthoc_rank_probe_v1_acquisition_recovery2_execution.json"
RECOVERY2_LOG_ERRATUM = PROJECT / "configs/kma_d1_1100_wsd_posthoc_rank_probe_v1_acquisition_recovery2_log_erratum.json"
RECOVERY2_DOWNLOADER = PROJECT / "scripts/download_kma_d1_1100_wsd_posthoc_rank_probe_v1_recovery2.py"
MATERIALIZER_CONFIG = PROJECT / "configs/kma_d1_1100_wsd_posthoc_rank_probe_v1_materializer_recovery2_execution_v3.json"
MATERIALIZER = PROJECT / "scripts/materialize_kma_d1_1100_wsd_posthoc_rank_probe_v1_recovery2_v3.py"
FINALIZER_CONFIG = PROJECT / "configs/kma_d1_1100_wsd_posthoc_rank_probe_v1_finalizer_recovery2_execution_v3.json"
KMA_TRAIN = PROJECT / "artifacts/final_submission_sprint_20260812/kma_d1_1100_v8/wsd_typ01/materialized/KMA_D1_1100_WSD_HOURLY_2022_2024_V8.parquet"
KMA_TRAIN_MANIFEST = PROJECT / "artifacts/final_submission_sprint_20260812/kma_d1_1100_v8/wsd_typ01/materialized/MANIFEST_V8.json"
KMA_TEST = WSD_ROOT / "materialized_recovery2_v3/KMA_D1_1100_WSD_HOURLY_2025_POSTHOC_V1_RECOVERY2_V3.parquet"
KMA_TEST_MANIFEST = WSD_ROOT / "materialized_recovery2_v3/MANIFEST_2025_POSTHOC_V1_RECOVERY2_V3.json"
SUMMARY_2025 = WSD_ROOT / "YEAR2025_SUMMARY_POSTHOC_V1_RECOVERY2.json"
PREGATE_SUMMARY = PROJECT / "artifacts/final_submission_sprint_20260812/kma_d1_1100_v8/wsd_typ01/PREGATE_2022_2024_SUMMARY_V8.json"
LABELS = Path(r"data/local/open/train/train_labels.csv")
SAMPLE = Path(r"data/local/open/sample_submission.csv")
BASELINE = PROJECT / "artifacts/postgate/public_adaptive_scale097_g2_delta_v2/final/A_plain_scale097.parquet"
BASELINE_01 = PROJECT / "artifacts/final_submission_sprint_20260812/01_OFFICIAL_ONLY_SAFE.csv"
VERTICAL_DIRECT_G2 = PROJECT / "artifacts/final_submission_sprint_20260812/vertical_direct/g2_probe_direct_2025.npy"
EXISTING_03 = PROJECT / "artifacts/final_submission_sprint_20260812/03_OFFICIAL_VERTICAL_G2_PROBE.csv"
MODEL_ROOT = ROOT / "final_model"
FINAL_CSV = ROOT / "04_HIGH_RISK_POSTHOC_KMA_R2_VERTICAL_RANK_PROBE_V1.csv"
FINAL_MANIFEST = ROOT / "04_HIGH_RISK_POSTHOC_KMA_R2_VERTICAL_RANK_PROBE_V1.manifest.json"
R2_INNER_BLEND = 0.10
R2_OUTER_MULTIPLIER = 1.5
VERTICAL_BLEND = 0.10
EXPECTED_ELIGIBLE_ROWS = {"kpx_group_1": 14309, "kpx_group_2": 14292}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path, *, relative: bool = False) -> dict[str, Any]:
    p = path.resolve()
    return {
        "path": p.relative_to(PROJECT).as_posix() if relative else p.as_posix(),
        "bytes": p.stat().st_size,
        "sha256": sha256_file(p),
    }


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
            raise RuntimeError("temporary write verification failed")
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    if path.read_bytes() != payload:
        raise RuntimeError("create-exclusive write verification failed")


def verify_declared(record: dict[str, Any]) -> dict[str, Any]:
    declared_path = Path(record["path"])
    path = declared_path if declared_path.is_absolute() else PROJECT / declared_path
    observed = identity(path, relative=not declared_path.is_absolute())
    if observed != record:
        raise RuntimeError(f"bound identity drift: {path.name}")
    return observed


def verify_execution() -> dict[str, Any]:
    config = json.loads(FINALIZER_CONFIG.read_text(encoding="ascii"))
    if config.get("schema_version") != 1 or config.get("status") != "FROZEN_BEFORE_POSTHOC_FINAL_REFIT_OR_CSV":
        raise RuntimeError("posthoc finalizer execution status/schema mismatch")
    if config.get("script_identity") != identity(Path(__file__).resolve(), relative=True):
        raise RuntimeError("posthoc finalizer script identity drift")
    for declared in config.get("bound_pre_2025_inputs", []):
        verify_declared(declared)
    if config.get("actual_argv") != [".venv/Scripts/python.exe", "-B", "scripts/build_kma_d1_1100_wsd_posthoc_rank_probe_v1.py"]:
        raise RuntimeError("posthoc finalizer argv mismatch")
    if config.get("output_csv") != FINAL_CSV.relative_to(PROJECT).as_posix() or config.get("output_manifest") != FINAL_MANIFEST.relative_to(PROJECT).as_posix():
        raise RuntimeError("posthoc finalizer output mismatch")
    return identity(FINALIZER_CONFIG)


def verify_post_2025_sources() -> dict[str, Any]:
    materializer_config = json.loads(MATERIALIZER_CONFIG.read_text(encoding="ascii"))
    if (
        materializer_config.get("schema_version") != 3
        or materializer_config.get("status") != "FROZEN_BEFORE_POSTHOC_RECOVERY2_V3_MATERIALIZATION"
        or materializer_config.get("materializer") != identity(MATERIALIZER, relative=True)
    ):
        raise RuntimeError("posthoc materializer binding drift")
    summary = json.loads(SUMMARY_2025.read_text(encoding="ascii"))
    if not (
        summary.get("mode") == "year2025"
        and summary.get("records") == 3285
        and summary.get("sequence_min") == 9865
        and summary.get("sequence_max") == 13149
        and summary.get("byte_gate_pass") is True
        and summary.get("persisted_prefix_reused_count") == 2970
        and summary.get("persisted_prefix_reused_sequence_range") == [9865, 12834]
        and summary.get("first_recovery_network_sequence") == 12835
        and summary.get("remaining_network_calls") == 315
        and summary.get("prior_extra_physical_attempts") == 5
        and summary.get("prior_extra_physical_bytes_conservative") == 1706485
        and summary.get("maximum_physical_attempts") == 13154
    ):
        raise RuntimeError("posthoc 2025 acquisition summary contract failed")
    response_max = int(summary.get("response_bytes_max", -1))
    if int(summary.get("projected_physical_bytes_by_observed_max", -1)) != response_max * 13154:
        raise RuntimeError("posthoc recovery projected-byte equation mismatch")
    pregate_summary = json.loads(PREGATE_SUMMARY.read_text(encoding="ascii"))
    accounted = int(pregate_summary.get("response_bytes_sum", -1)) + 1_706_485 + int(summary.get("response_bytes_sum", -1))
    if int(summary.get("total_response_bytes_accounted", -1)) != accounted or accounted > 4_700_000_000:
        raise RuntimeError("posthoc recovery total-byte equation mismatch")
    manifest = json.loads(KMA_TEST_MANIFEST.read_text(encoding="ascii"))
    if (
        manifest.get("mode") != "2025"
        or manifest.get("rows") != 8760
        or manifest.get("parquet") != identity(KMA_TEST, relative=True)
        or manifest.get("materializer") != identity(MATERIALIZER, relative=True)
        or manifest.get("secret_persisted") is not False
    ):
        raise RuntimeError("posthoc 2025 materialization manifest contract failed")
    source_serialized = json.dumps(manifest.get("source_evidence", {}), sort_keys=True)
    for required in (
        sha256_file(SUMMARY_2025),
        sha256_file(PREREG),
        sha256_file(INCIDENT),
        sha256_file(RECOVERY_ACQUISITION_CONFIG),
        sha256_file(RECOVERY_DOWNLOADER),
        sha256_file(RECOVERY2_INCIDENT),
        sha256_file(RECOVERY2_ACQUISITION_CONFIG),
        sha256_file(RECOVERY2_LOG_ERRATUM),
        sha256_file(RECOVERY2_DOWNLOADER),
        sha256_file(PREGATE_SUMMARY),
    ):
        if required not in source_serialized:
            raise RuntimeError("posthoc materialization source lineage incomplete")
    return {
        "summary": identity(SUMMARY_2025),
        "pregate_summary": identity(PREGATE_SUMMARY),
        "materializer_config": identity(MATERIALIZER_CONFIG),
        "kma_2025": identity(KMA_TEST),
        "kma_2025_manifest": identity(KMA_TEST_MANIFEST),
    }


def normalize_kma(path: Path, years: tuple[int, ...]) -> pd.DataFrame:
    frame = pd.read_parquet(path, engine="pyarrow")
    if "forecast_kst_dtm" in frame.columns:
        frame = frame.set_index("forecast_kst_dtm", verify_integrity=True)
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    expected = gate.expected_year_index(years[0])
    for year in years[1:]:
        expected = expected.append(gate.expected_year_index(year))
    if not frame.index.equals(expected):
        raise RuntimeError("KMA hourly index mismatch")
    required = {
        "operating_year", "operating_day", "issue_base_kst", "lead_hour",
        "wsd_cell_94_121", "wsd_cell_94_122", *[f"wsd_{g}" for g in GROUPS],
    }
    if not required.issubset(frame.columns):
        raise RuntimeError("KMA hourly schema mismatch")
    operating_clock = frame.index - pd.Timedelta(hours=1)
    expected_leads = operating_clock.hour.to_numpy(dtype=int) + 1
    if not np.array_equal(frame["lead_hour"].to_numpy(dtype=int), expected_leads):
        raise RuntimeError("KMA interval-end lead mapping mismatch")
    if not np.array_equal(frame["operating_year"].to_numpy(dtype=int), operating_clock.year):
        raise RuntimeError("KMA operating-year mapping mismatch")
    expected_days = operating_clock.normalize()
    observed_days = pd.DatetimeIndex(pd.to_datetime(frame["operating_day"], errors="raise"))
    if not observed_days.equals(expected_days):
        raise RuntimeError("KMA operating-day mapping mismatch")
    observed_issue = pd.DatetimeIndex(pd.to_datetime(frame["issue_base_kst"], errors="raise"))
    expected_issue = expected_days - pd.Timedelta(days=1) + pd.Timedelta(hours=11)
    if not observed_issue.equals(expected_issue):
        raise RuntimeError("KMA issue timestamp is not exact D-1 11 KST")
    a = frame["wsd_cell_94_121"].to_numpy(dtype=float)
    b = frame["wsd_cell_94_122"].to_numpy(dtype=float)
    for values in (a, b):
        finite = np.isfinite(values)
        if np.any(values[finite] < 0.0) or np.any(values[finite] > 75.0):
            raise RuntimeError("KMA selected-cell plausibility mismatch")
    expected_groups = {
        "kpx_group_1": 0.5 * a + 0.5 * b,
        "kpx_group_2": (5.0 / 6.0) * a + (1.0 / 6.0) * b,
        "kpx_group_3": a,
    }
    for group, values in expected_groups.items():
        if not np.allclose(frame[f"wsd_{group}"].to_numpy(dtype=float), values, rtol=0.0, atol=1e-12, equal_nan=True):
            raise RuntimeError(f"KMA group formula mismatch: {group}")
    day_keys = pd.Series(expected_days, index=frame.index)
    for col in ("wsd_cell_94_121", "wsd_cell_94_122"):
        finite_by_day = pd.Series(np.isfinite(frame[col].to_numpy(dtype=float)), index=frame.index).groupby(day_keys).sum()
        if not finite_by_day.isin([0, 24]).all():
            raise RuntimeError(f"KMA invalid-day pattern is not all24-finite/all24-NaN: {col}")
    for year in years:
        mask = operating_clock.year == year
        for col in ("wsd_cell_94_121", "wsd_cell_94_122"):
            values = frame.loc[mask, col].to_numpy(dtype=float)
            if np.isfinite(values).mean() < 0.98:
                raise RuntimeError("KMA annual coverage below 0.98")
            leads = frame.loc[mask, "lead_hour"].to_numpy(dtype=int)
            for lead in range(1, 25):
                if np.isfinite(values[leads == lead]).mean() < 0.98:
                    raise RuntimeError("KMA per-lead coverage below 0.98")
    return frame


def load_baseline() -> pd.DataFrame:
    baseline = pd.read_parquet(BASELINE, engine="pyarrow").astype(float)
    baseline.index = pd.DatetimeIndex(baseline.index, name="forecast_kst_dtm")
    if not baseline.index.equals(TEST_INDEX) or tuple(baseline.columns) != GROUPS or not np.isfinite(baseline.to_numpy()).all():
        raise RuntimeError("full-precision deployment baseline mismatch")
    reference = pd.read_csv(BASELINE_01, encoding="utf-8-sig", dtype="string", keep_default_na=False)
    sample = pd.read_csv(SAMPLE, encoding="utf-8-sig", dtype="string", keep_default_na=False)
    if not reference[["forecast_id", "forecast_kst_dtm"]].equals(sample[["forecast_id", "forecast_kst_dtm"]]):
        raise RuntimeError("baseline/sample identities differ")
    for group in GROUPS:
        expected = pd.Series([f"{value:.6f}" for value in baseline[group]], dtype="string")
        if not reference[group].reset_index(drop=True).equals(expected):
            raise RuntimeError(f"baseline full precision lineage mismatch: {group}")
    return baseline


def load_training_baselines() -> dict[int, pd.DataFrame]:
    result = gate.load_baselines()
    return result


def load_ws10(train: bool) -> dict[str, pd.Series]:
    result: dict[str, pd.Series] = {}
    identities = gate.WEATHER if train else {
        "kpx_group_1": (PROJECT / "artifacts/cache/kpx_group_1_weather_test.parquet", 26037151, "8de6fe0ba96ee5c4d0bae16c1b57bc761863152064fda490caf048a5b6e87771"),
        "kpx_group_2": (PROJECT / "artifacts/cache/kpx_group_2_weather_test.parquet", 26031426, "3450c23bd1e6a8270f416956d65f7df0a0265d3d67ea30c1b69dd4f214b43e99"),
        "kpx_group_3": (PROJECT / "artifacts/cache/kpx_group_3_weather_test.parquet", 26038173, "54ee1011e1c9f328347500fe1f6113919f542bf7697126201cedc16bfdec2ed6"),
    }
    expected_index = TRAIN_INDEX if train else TEST_INDEX
    for group, spec in identities.items():
        path, size, digest = spec
        if path.stat().st_size != size or sha256_file(path) != digest:
            raise RuntimeError(f"weather identity drift: {group}")
        frame = pd.read_parquet(path, columns=["ldaps__idw__ws10"], engine="pyarrow")
        frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
        if not frame.index.equals(expected_index):
            raise RuntimeError(f"weather index drift: {group}")
        result[group] = frame["ldaps__idw__ws10"].astype(float)
    return result


def fit_r2(kma_train: pd.DataFrame, kma_test: pd.DataFrame, labels: pd.DataFrame, baseline: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    train_baselines = load_training_baselines()
    ws10_train, ws10_test = load_ws10(True), load_ws10(False)
    inner = pd.DataFrame(index=TEST_INDEX, columns=("kpx_group_1", "kpx_group_2"), dtype=float)
    models: dict[str, Any] = {}
    from lightgbm import LGBMRegressor
    for group in inner.columns:
        capacity = CAPACITY_KWH[group]
        train_base = pd.concat([train_baselines[year][group] for year in (2022, 2023, 2024) if group in train_baselines[year].columns]).reindex(TRAIN_INDEX)
        train_all = gate.feature_frame(kma_train, group, ws10_train[group], train_base)
        test_all = gate.feature_frame(kma_test, group, ws10_test[group], baseline[group])
        actual = labels[group].astype(float)
        kma_finite = np.isfinite(train_all.loc[:, gate.R2_FEATURES[:-1]].to_numpy(dtype=float)).all(axis=1)
        eligible = np.isfinite(actual) & (actual >= 0.10 * capacity) & np.isfinite(train_base) & kma_finite
        train_x = train_all.loc[eligible, gate.R2_FEATURES]
        if len(train_x) != EXPECTED_ELIGIBLE_ROWS[group]:
            raise RuntimeError(f"R2 full-fit eligible-row invariant drift: {group}/{len(train_x)}")
        transform = TrainOnlyMedianTransform().fit(train_x)
        model = LGBMRegressor(**gate.R2_PARAMS)
        target = actual.loc[train_x.index].to_numpy(dtype=float) / capacity - train_base.loc[train_x.index].to_numpy(dtype=float) / capacity
        model.fit(transform.transform(train_x), target)
        raw = np.asarray(model.predict(transform.transform(test_all.loc[:, gate.R2_FEATURES])), dtype=float)
        inner[group] = np.clip(baseline[group].to_numpy(dtype=float) + R2_INNER_BLEND * np.clip(raw, -0.20, 0.20) * capacity, 0.0, 1.02 * capacity)
        models[group] = {"model": model, "transform": transform, "fit_rows": int(len(train_x))}
    if not np.isfinite(inner.to_numpy()).all():
        raise RuntimeError("R2 inner prediction non-finite")
    return inner, models


def compose(baseline: pd.DataFrame, inner: pd.DataFrame) -> pd.DataFrame:
    vertical = np.load(VERTICAL_DIRECT_G2, allow_pickle=False).astype(float)
    if vertical.shape != (8760,) or not np.isfinite(vertical).all():
        raise RuntimeError("stored vertical direct prediction mismatch")
    output = baseline.copy()
    d1 = R2_OUTER_MULTIPLIER * (inner["kpx_group_1"].to_numpy(dtype=float) - baseline["kpx_group_1"].to_numpy(dtype=float))
    d2 = R2_OUTER_MULTIPLIER * (inner["kpx_group_2"].to_numpy(dtype=float) - baseline["kpx_group_2"].to_numpy(dtype=float))
    v03 = np.clip((1.0 - VERTICAL_BLEND) * baseline["kpx_group_2"].to_numpy(dtype=float) + VERTICAL_BLEND * vertical, 0.0, CAPACITY_KWH["kpx_group_2"])
    output["kpx_group_1"] = np.clip(baseline["kpx_group_1"].to_numpy(dtype=float) + d1, 0.0, 1.02 * CAPACITY_KWH["kpx_group_1"])
    output["kpx_group_2"] = np.clip(v03 + d2, 0.0, 1.02 * CAPACITY_KWH["kpx_group_2"])
    output["kpx_group_3"] = baseline["kpx_group_3"].to_numpy(dtype=float)
    return output


def csv_payload(prediction: pd.DataFrame) -> tuple[bytes, dict[str, Any]]:
    sample = pd.read_csv(SAMPLE, encoding="utf-8-sig", dtype="string", keep_default_na=False)
    sample_index = pd.DatetimeIndex(pd.to_datetime(sample["forecast_kst_dtm"], errors="raise"), name="forecast_kst_dtm")
    expected_columns = ("forecast_id", "forecast_kst_dtm", *GROUPS)
    if tuple(sample.columns) != expected_columns or not sample_index.equals(prediction.index):
        raise RuntimeError("sample schema/index mismatch")
    output = sample[["forecast_id", "forecast_kst_dtm"]].copy()
    for group in GROUPS:
        values = prediction[group].to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1.02 * CAPACITY_KWH[group]):
            raise RuntimeError(f"prediction bound mismatch: {group}")
        output[group] = values
    payload = b"\xef\xbb\xbf" + output.to_csv(index=False, float_format="%.6f", lineterminator="\n").encode("utf-8")
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "probe.csv"
        path.write_bytes(payload)
        replay = pd.read_csv(path, encoding="utf-8-sig", dtype="string", keep_default_na=False)
    decimal = re.compile(r"^-?\d+\.\d{6}$")
    if tuple(replay.columns) != expected_columns or len(replay) != 8760 or not replay[["forecast_id", "forecast_kst_dtm"]].equals(sample[["forecast_id", "forecast_kst_dtm"]]):
        raise RuntimeError("CSV schema/readback mismatch")
    for group in GROUPS:
        expected = pd.Series([f"{x:.6f}" for x in prediction[group].to_numpy(dtype=float)], dtype="string")
        if not replay[group].reset_index(drop=True).equals(expected) or not replay[group].map(lambda x: bool(decimal.fullmatch(x))).all():
            raise RuntimeError(f"CSV six-decimal mismatch: {group}")
    baseline_text = pd.read_csv(BASELINE_01, encoding="utf-8-sig", dtype="string", keep_default_na=False)
    if not replay["kpx_group_3"].equals(baseline_text["kpx_group_3"]):
        raise RuntimeError("G3 is not exact baseline text identity")
    return payload, {"rows": 8760, "columns": list(expected_columns), "encoding": "utf-8-sig", "numeric_decimals": 6, "sample_identity_and_time_exact": True, "g3_baseline_text_identity": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    pre = verify_execution()
    if args.verify_only:
        print(json.dumps({"verify_only": "PASS", "execution": pre}, sort_keys=True))
        return
    if FINAL_CSV.exists() or FINAL_MANIFEST.exists() or MODEL_ROOT.exists():
        raise FileExistsError("refusing to overwrite posthoc final artifacts")
    post = verify_post_2025_sources()
    kma_train = normalize_kma(KMA_TRAIN, (2022, 2023, 2024))
    kma_test = normalize_kma(KMA_TEST, (2025,))
    baseline = load_baseline()
    labels = pd.read_csv(LABELS, encoding="utf-8-sig", parse_dates=["kst_dtm"]).set_index("kst_dtm")
    labels.index = pd.DatetimeIndex(labels.index, name="forecast_kst_dtm")
    labels = labels.loc[:, list(GROUPS)].astype(float)
    if not labels.index.equals(TRAIN_INDEX):
        raise RuntimeError("label index mismatch")
    MODEL_ROOT.mkdir(parents=True, exist_ok=False)
    source_lock = MODEL_ROOT / "SOURCE_LOCK_BEFORE_FINAL_FIT.json"
    exclusive_bytes(source_lock, canonical_json({
        "status": "ALL_POSTHOC_SOURCES_LOCKED_BEFORE_FINAL_FIT",
        "posthoc_high_risk": True,
        "independent_confirmation": False,
        "pre_2025_execution": pre,
        "post_2025_sources": post,
    }))
    inner, models = fit_r2(kma_train, kma_test, labels, baseline)
    model_ids = {}
    for group, model in models.items():
        path = MODEL_ROOT / f"R2_POSTHOC_V1__{group}.pkl"
        exclusive_bytes(path, pickle.dumps(model, protocol=5))
        model_ids[group] = identity(path)
    inner_path = MODEL_ROOT / "P_R2_010_INNER_FULL_PRECISION.parquet"
    gate.exclusive_parquet(inner_path, inner)
    prediction = compose(baseline, inner)
    prediction_path = MODEL_ROOT / "FINAL_PREDICTION_FULL_PRECISION.parquet"
    gate.exclusive_parquet(prediction_path, prediction)
    payload, csv_contract = csv_payload(prediction)
    exclusive_bytes(FINAL_CSV, payload)
    if FINAL_CSV.read_bytes() != payload:
        raise RuntimeError("CSV byte readback mismatch")
    manifest = {
        "schema_version": 1,
        "artifact_type": "KMA_D1_1100_WSD_POSTHOC_HIGH_RISK_RANK_PROBE_V1",
        "status": "CSV_GENERATED_POSTHOC_HIGH_RISK",
        "posthoc_high_risk": True,
        "independent_confirmation": False,
        "warning": "Exploratory rank probe; no guarantee of Public or Private improvement.",
        "training_operating_years": [2022, 2023, 2024],
        "application_operating_year": 2025,
        "formula": {
            "inner": "P_R2_010=clip(B+0.10*clip(raw,-0.20,0.20)*capacity,0,1.02*capacity)",
            "G1": "clip(B1+1.5*(P_R2_010_G1-B1),0,1.02*capacity_G1)",
            "G2": "clip(clip(0.90*B2+0.10*vertical_direct_G2,0,capacity_G2)+1.5*(P_R2_010_G2-B2),0,1.02*capacity_G2)",
            "G3": "B3 exact",
        },
        "oof_evidence_vs_B": {
            "2023_delta_score": 0.0022882758619056087,
            "2024_delta_score": 0.0018219713052041175,
            "mean_delta_score": 0.002055123583554863,
            "mean_delta_ficr": 0.004246630600844736,
        },
        "oof_delta_score_vs_03": {"2023": 0.001459041143731299, "2024": 0.0012197568422537275},
        "source_lock": identity(source_lock),
        "models": model_ids,
        "inner_prediction": identity(inner_path),
        "full_precision_prediction": identity(prediction_path),
        "csv": {**identity(FINAL_CSV), **csv_contract},
    }
    exclusive_bytes(FINAL_MANIFEST, canonical_json(manifest))
    print(json.dumps({"csv": identity(FINAL_CSV), "manifest": identity(FINAL_MANIFEST), "posthoc_high_risk": True}, sort_keys=True))


if __name__ == "__main__":
    main()

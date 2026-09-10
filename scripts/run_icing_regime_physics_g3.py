"""Run the single frozen G3 glaze-icing physics confirmation.

The experiment has two irreversible stages:

1. reproduce the pre-2024 G3-H2 development gate without decoding a 2024
   label value; and
2. materialize/hash the one 2024 candidate before decoding 2024 labels, then
   create a 2025 CSV only if every registered confirmation gate passes.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metric import CAPACITY_KWH, TARGET_COLS, group_metrics, score_details  # noqa: E402


CONFIG = ROOT / "configs/icing_regime_physics_g3_preregister_v1.json"
CONFIG_SIDECAR = CONFIG.with_suffix(".sha256")
CONFIG_SHA256 = "a1c33c4b2d765f629296616babad312a74153a7ce853c839a18a493235709f50"
OUTPUT = ROOT / "artifacts/postgate/icing_regime_physics_g3_strict_v1"
LABELS = Path(r"data/local/open/train/train_labels.csv")
SAMPLE = Path(r"data/local/open/sample_submission.csv")
WEATHER_TRAIN = ROOT / "artifacts/cache/kpx_group_3_weather_train.parquet"
WEATHER_TEST = ROOT / "artifacts/cache/kpx_group_3_weather_test.parquet"
DEV_G3 = ROOT / "artifacts/oof/g3dev2023h2_candidates.parquet"
BASE_2024 = ROOT / "artifacts/oof/gate2024_recent_v4_cf_fix_calibration_fit.parquet"
PARENT_2025 = (
    ROOT
    / "artifacts/postgate/multi_nwp_joint_g12_posthoc_rescue_final_2025_v1/"
    "final_prediction_kwh_2025.parquet"
)
PARENT_CSV = (
    ROOT
    / "artifacts/postgate/multi_nwp_joint_g12_posthoc_rescue_final_2025_v1/"
    "multi_nwp_joint_g12_posthoc_rescue_recent097_2025.csv"
)

G3 = "kpx_group_3"
CAP = CAPACITY_KWH[G3]
DERATE = 0.84
PRE2024_LABEL_ROWS = 17_520

WEATHER_COLUMNS = [
    "ldaps__idw__heightAboveGround_2_t",
    "ldaps__idw__heightAboveGround_2_dpt",
    "ldaps__idw__surface_0_avg_lsprate",
    "ldaps__idw__surface_0_lssrate",
    "ldaps__idw__surface_0_ncpcp",
    "ldaps__idw__etc_0_lcc",
    "ldaps__idw__etc_0_VLCDC",
    "ldaps__idw__hub_ws",
    "gfs__idw__heightAboveGround_2_2t",
    "gfs__idw__heightAboveGround_2_2d",
    "gfs__idw__surface_0_prate",
    "gfs__idw__lowCloudLayer_0_lcc",
    "gfs__idw__hub_ws",
    "site__hub_height_m",
]

SLICES_2024: dict[str, tuple[str, str]] = {
    "full": ("2024-01-01 01:00:00", "2025-01-01 00:00:00"),
    "H1": ("2024-01-01 01:00:00", "2024-07-01 00:00:00"),
    "H2": ("2024-07-01 01:00:00", "2025-01-01 00:00:00"),
    "Q1": ("2024-01-01 01:00:00", "2024-04-01 00:00:00"),
    "Q2": ("2024-04-01 01:00:00", "2024-07-01 00:00:00"),
    "Q3": ("2024-07-01 01:00:00", "2024-10-01 00:00:00"),
    "Q4": ("2024-10-01 01:00:00", "2025-01-01 00:00:00"),
}
for month in range(1, 13):
    start = pd.Timestamp(2024, month, 1, 1)
    if month == 12:
        end = pd.Timestamp(2025, 1, 1, 0)
    else:
        end = pd.Timestamp(2024, month + 1, 1, 0)
    SLICES_2024[f"2024-{month:02d}"] = (str(start), str(end))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("stage1", "confirm", "all"), default="all")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": resolved.as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, engine="pyarrow", index=True)
    temporary.replace(path)


def load_config() -> dict[str, Any]:
    if sha256_file(CONFIG) != CONFIG_SHA256:
        raise AssertionError("frozen icing config hash changed")
    sidecar = CONFIG_SIDECAR.read_text(encoding="utf-8").strip().split()
    if not sidecar or sidecar[0] != CONFIG_SHA256:
        raise AssertionError("icing config sidecar changed")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config["single_candidate"]["candidate_count"] != 1:
        raise AssertionError("candidate count is not one")
    if config["single_candidate"]["id"] != "g3_glaze_consensus_wet_cloud_derate16":
        raise AssertionError("candidate identity changed")
    return config


def verify_locked_inputs(config: dict[str, Any]) -> None:
    specs = [
        config["duplicate_census"],
        config["pre2024_stage1"]["application_prediction"],
        config["single_2024_confirmation"]["baseline"],
        config["final_if_and_only_if_pass"]["parent_prediction"],
        config["final_if_and_only_if_pass"]["parent_csv"],
        config["final_if_and_only_if_pass"]["weather"],
        config["final_if_and_only_if_pass"]["sample"],
        *config["immutable_inputs"].values(),
    ]
    for spec in specs:
        path = Path(spec["path"])
        if not path.is_absolute():
            path = ROOT / path
        if path.stat().st_size != int(spec["bytes"]):
            raise AssertionError(f"locked byte count changed: {path}")
        if sha256_file(path) != str(spec["sha256"]):
            raise AssertionError(f"locked sha256 changed: {path}")


def read_weather_window(path: Path, start: str, end: str) -> pd.DataFrame:
    frame = pd.read_parquet(
        path,
        columns=WEATHER_COLUMNS,
        filters=[
            ("forecast_kst_dtm", ">=", pd.Timestamp(start).to_pydatetime()),
            ("forecast_kst_dtm", "<=", pd.Timestamp(end).to_pydatetime()),
        ],
        engine="pyarrow",
    ).astype(np.float64)
    frame.index = pd.DatetimeIndex(frame.index, name="kst_dtm")
    expected = pd.date_range(start, end, freq="h", name="kst_dtm")
    if not frame.index.equals(expected):
        raise AssertionError(f"weather window/index mismatch: {start} .. {end}")
    if tuple(frame.columns) != tuple(WEATHER_COLUMNS):
        raise AssertionError("weather column contract changed")
    if not np.isfinite(frame.to_numpy()).all():
        raise AssertionError("weather contains a non-finite selected value")
    return frame


def icing_state(weather: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    ldaps_hub_c = (
        weather["ldaps__idw__heightAboveGround_2_t"]
        - 273.15
        - 0.0065 * weather["site__hub_height_m"]
    )
    gfs_hub_c = (
        weather["gfs__idw__heightAboveGround_2_2t"]
        - 273.15
        - 0.0065 * weather["site__hub_height_m"]
    )
    mean_hub_c = 0.5 * (ldaps_hub_c + gfs_hub_c)
    ldaps_dd = (
        weather["ldaps__idw__heightAboveGround_2_t"]
        - weather["ldaps__idw__heightAboveGround_2_dpt"]
    )
    gfs_dd = (
        weather["gfs__idw__heightAboveGround_2_2t"]
        - weather["gfs__idw__heightAboveGround_2_2d"]
    )
    precipitation = (
        (weather["ldaps__idw__surface_0_avg_lsprate"] > 0.0)
        | (weather["ldaps__idw__surface_0_lssrate"] > 0.0)
        | (weather["ldaps__idw__surface_0_ncpcp"] > 0.0)
        | (weather["gfs__idw__surface_0_prate"] > 0.0)
    )
    low_cloud = (
        (weather["ldaps__idw__etc_0_lcc"] >= 0.8)
        | (weather["ldaps__idw__etc_0_VLCDC"] >= 0.5)
        | (weather["gfs__idw__lowCloudLayer_0_lcc"] >= 80.0)
    )
    state = (
        (ldaps_hub_c <= 0.5)
        & (gfs_hub_c <= 0.5)
        & (mean_hub_c >= -5.0)
        & (ldaps_dd <= 3.0)
        & (gfs_dd <= 3.0)
        & (weather["ldaps__idw__hub_ws"] >= 3.0)
        & (weather["gfs__idw__hub_ws"] >= 1.5)
        & precipitation
        & low_cloud
    ).astype(bool)
    state.name = "icing_state"
    diagnostics = pd.DataFrame(
        {
            "ldaps_hub_c": ldaps_hub_c,
            "gfs_hub_c": gfs_hub_c,
            "mean_hub_c": mean_hub_c,
            "ldaps_dewpoint_depression_k": ldaps_dd,
            "gfs_dewpoint_depression_k": gfs_dd,
            "precipitation_proxy": precipitation.astype(np.int8),
            "low_cloud_proxy": low_cloud.astype(np.int8),
            "icing_state": state.astype(np.int8),
        },
        index=weather.index,
    )
    return state, diagnostics


def reconstruct_dev_g3_base(frame: pd.DataFrame) -> pd.Series:
    expected = {
        "l1",
        "q06",
        "q07",
        "ewq06",
        "top200q07",
        "xgb",
        "cat",
        "extra",
        "shared_l1",
        "shared_q07",
    }
    if set(frame.columns) != expected:
        raise AssertionError("G3 development candidate schema changed")
    corrected_mix = (
        0.20 * frame["q07"]
        + 0.075 * frame["shared_l1"]
        + 0.425 * frame["shared_q07"]
        + 0.025 * frame["top200q07"]
        + 0.275 * frame["ewq06"]
    )
    corrected_v3 = np.clip(1.25 * corrected_mix - 1200.0, 0.0, 1.02 * CAP)
    recent_branch = np.clip(
        1.075
        * (
            0.70 * frame["shared_q07"]
            + 0.10 * frame["shared_l1"]
            + 0.20 * frame["q07"]
        )
        - 300.0,
        0.0,
        1.02 * CAP,
    )
    base = 0.97 * (0.50 * corrected_v3 + 0.50 * recent_branch)
    return pd.Series(base, index=frame.index, name=G3, dtype=np.float64)


def comparison(actual: pd.Series, base: pd.Series, candidate: pd.Series) -> dict[str, Any]:
    baseline_metric = group_metrics(actual, base, CAP, group_name=G3)
    candidate_metric = group_metrics(actual, candidate, CAP, group_name=G3)
    delta_n = candidate_metric.one_minus_nmae - baseline_metric.one_minus_nmae
    delta_f = candidate_metric.ficr - baseline_metric.ficr
    return {
        "baseline": baseline_metric.as_dict(),
        "candidate": candidate_metric.as_dict(),
        "delta_total": float(0.5 * (delta_n + delta_f)),
        "delta_one_minus_nmae": float(delta_n),
        "delta_ficr": float(delta_f),
    }


def run_stage1(out_dir: Path, config: dict[str, Any]) -> bool:
    destination = out_dir / "stage1_results.json"
    lock_path = out_dir / "stage1_promotion_lock.json"
    if destination.exists() or lock_path.exists():
        raise FileExistsError("stage1 output already exists; refusing to overwrite")
    raw = pd.read_parquet(DEV_G3, engine="pyarrow").astype(np.float64)
    raw.index = pd.DatetimeIndex(raw.index, name="kst_dtm")
    expected = pd.date_range(
        "2023-07-01 01:00:00", "2024-01-01 00:00:00", freq="h", name="kst_dtm"
    )
    if not raw.index.equals(expected):
        raise AssertionError("G3 development prediction index changed")
    base = reconstruct_dev_g3_base(raw)
    weather = read_weather_window(
        WEATHER_TRAIN, "2023-07-01 01:00:00", "2024-01-01 00:00:00"
    )
    state, diagnostics = icing_state(weather)
    candidate = base.copy()
    candidate.loc[state] = DERATE * base.loc[state]
    candidate_frame = pd.DataFrame({"baseline": base, "candidate": candidate, "icing": state})
    atomic_parquet(candidate_frame, out_dir / "stage1_candidate.parquet")
    atomic_parquet(diagnostics, out_dir / "stage1_physics_diagnostics.parquet")
    # Read only the exact CSV prefix ending at 2024-01-01 00:00.  No 2024
    # label data row is decoded in this stage.
    labels = pd.read_csv(
        LABELS,
        encoding="utf-8-sig",
        nrows=PRE2024_LABEL_ROWS,
        parse_dates=["kst_dtm"],
    ).set_index("kst_dtm")
    if labels.index.max() != pd.Timestamp("2024-01-01 00:00:00"):
        raise AssertionError("pre-2024 label prefix boundary changed")
    labels = labels.reindex(expected)
    windows = {
        "H2": ("2023-07-01 01:00:00", "2024-01-01 00:00:00"),
        "Q3": ("2023-07-01 01:00:00", "2023-10-01 00:00:00"),
        "Q4": ("2023-10-01 01:00:00", "2024-01-01 00:00:00"),
        "November": ("2023-11-01 01:00:00", "2023-12-01 00:00:00"),
        "December": ("2023-12-01 01:00:00", "2024-01-01 00:00:00"),
    }
    metrics: dict[str, Any] = {}
    passes: dict[str, bool] = {}
    for name, (start, end) in windows.items():
        index = pd.date_range(start, end, freq="h")
        active = int(state.reindex(index).sum())
        result = comparison(labels.loc[index, G3], base.loc[index], candidate.loc[index])
        result["active_rows"] = active
        metrics[name] = result
        if name == "Q3":
            identical = np.array_equal(
                base.loc[index].to_numpy(), candidate.loc[index].to_numpy()
            )
            passes[name] = bool(
                active == 0
                and identical
                and result["delta_total"] == 0.0
                and result["delta_one_minus_nmae"] == 0.0
                and result["delta_ficr"] == 0.0
            )
        else:
            passes[name] = bool(
                active > 0
                and result["delta_total"] > 0.0
                and result["delta_one_minus_nmae"] > 0.0
                and result["delta_ficr"] > 0.0
            )
    passed = bool(all(passes.values()) and int(state.sum()) == 69)
    payload = {
        "artifact_type": "icing_regime_physics_g3_pre2024_stage1",
        "created_utc": utc_now(),
        "config_sha256": CONFIG_SHA256,
        "candidate_count": 1,
        "active_rows": int(state.sum()),
        "metrics": metrics,
        "slice_passes": passes,
        "passed": passed,
        "label_access": {
            "read_kind": "csv_nrows_prefix",
            "decoded_data_rows": PRE2024_LABEL_ROWS,
            "decoded_max_timestamp": str(labels.index.max()),
            "2024_label_data_rows_decoded": 0,
        },
        "outputs": [
            record(out_dir / "stage1_candidate.parquet"),
            record(out_dir / "stage1_physics_diagnostics.parquet"),
        ],
    }
    atomic_json(payload, destination)
    atomic_json(
        {
            "status": "PROMOTE_SINGLE_CANDIDATE" if passed else "REJECT_STAGE1",
            "passed": passed,
            "config_sha256": CONFIG_SHA256,
            "candidate": config["single_candidate"],
            "stage1_results": record(destination),
            "no_reselection_or_retuning": True,
        },
        lock_path,
    )
    return passed


def read_prediction(path: Path, expected: pd.DatetimeIndex) -> pd.DataFrame:
    frame = pd.read_parquet(path, engine="pyarrow").astype(np.float64)
    frame.index = pd.DatetimeIndex(frame.index, name="kst_dtm")
    if tuple(frame.columns) != TARGET_COLS or not frame.index.equals(expected):
        raise AssertionError(f"prediction contract changed: {path}")
    if not np.isfinite(frame.to_numpy()).all():
        raise AssertionError(f"non-finite prediction: {path}")
    return frame


def slice_comparison(
    actual: pd.DataFrame,
    base: pd.DataFrame,
    candidate: pd.DataFrame,
    index: pd.DatetimeIndex,
) -> dict[str, Any]:
    g3 = comparison(actual.loc[index, G3], base.loc[index, G3], candidate.loc[index, G3])
    baseline_mixed = score_details(actual.loc[index], base.loc[index])
    candidate_mixed = score_details(actual.loc[index], candidate.loc[index])
    mixed = {
        "baseline": baseline_mixed.as_dict(),
        "candidate": candidate_mixed.as_dict(),
        "delta_total": float(candidate_mixed.total_score - baseline_mixed.total_score),
        "delta_one_minus_nmae": float(
            candidate_mixed.one_minus_nmae - baseline_mixed.one_minus_nmae
        ),
        "delta_ficr": float(candidate_mixed.ficr - baseline_mixed.ficr),
    }
    for component in ("delta_total", "delta_one_minus_nmae", "delta_ficr"):
        if not np.isclose(mixed[component], g3[component] / 3.0, rtol=0.0, atol=1e-12):
            raise AssertionError(f"mixed/G3 algebra changed for {component}")
    return {"g3": g3, "mixed": mixed}


def final_csv(
    out_dir: Path,
    config: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    sample = pd.read_csv(
        SAMPLE,
        encoding="utf-8-sig",
        dtype={"forecast_id": "string", "forecast_kst_dtm": "string"},
    )
    index = pd.DatetimeIndex(pd.to_datetime(sample["forecast_kst_dtm"]), name="kst_dtm")
    expected = pd.date_range(
        "2025-01-01 01:00:00", "2026-01-01 00:00:00", freq="h", name="kst_dtm"
    )
    if not index.equals(expected):
        raise AssertionError("sample time contract changed")
    parent = read_prediction(PARENT_2025, expected)
    parent_csv = pd.read_csv(
        PARENT_CSV,
        encoding="utf-8-sig",
        dtype={"forecast_id": "string", "forecast_kst_dtm": "string"},
    )
    if not parent_csv[["forecast_id", "forecast_kst_dtm"]].equals(
        sample[["forecast_id", "forecast_kst_dtm"]]
    ):
        raise AssertionError("parent CSV identifiers differ from sample")
    for group in TARGET_COLS:
        observed = parent_csv[group].to_numpy(dtype=np.float64)
        if not np.array_equal(observed, np.round(parent[group].to_numpy(), 6)):
            raise AssertionError(f"parent CSV/parquet round-trip mismatch for {group}")
    weather = read_weather_window(
        WEATHER_TEST, "2025-01-01 01:00:00", "2026-01-01 00:00:00"
    )
    state, diagnostics = icing_state(weather)
    final = parent.copy()
    final.loc[state, G3] = DERATE * parent.loc[state, G3]
    final[G3] = np.clip(final[G3], 0.0, 1.02 * CAP)
    if not np.array_equal(
        final.loc[~state, G3].to_numpy(), parent.loc[~state, G3].to_numpy()
    ):
        raise AssertionError("off-risk G3 identity changed")
    if not np.array_equal(
        final[["kpx_group_1", "kpx_group_2"]].to_numpy(),
        parent[["kpx_group_1", "kpx_group_2"]].to_numpy(),
    ):
        raise AssertionError("G1/G2 parent identity changed")
    atomic_parquet(final, out_dir / "final_prediction_kwh_2025.parquet")
    atomic_parquet(diagnostics, out_dir / "final_physics_diagnostics_2025.parquet")
    submission = sample.copy()
    for group in TARGET_COLS:
        submission[group] = final[group].to_numpy(dtype=np.float64)
    csv_path = out_dir / Path(config["final_if_and_only_if_pass"]["output"]).name
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_suffix(csv_path.suffix + ".tmp")
    submission.to_csv(temporary, index=False, encoding="utf-8-sig", float_format="%.6f")
    temporary.replace(csv_path)
    with csv_path.open("rb") as stream:
        if stream.read(3) != b"\xef\xbb\xbf":
            raise AssertionError("final CSV is not UTF-8-SIG")
    observed = pd.read_csv(
        csv_path,
        encoding="utf-8-sig",
        dtype={"forecast_id": "string", "forecast_kst_dtm": "string"},
    )
    if tuple(observed.columns) != tuple(sample.columns) or len(observed) != 8760:
        raise AssertionError("final CSV schema/row count changed")
    for group in TARGET_COLS:
        if not np.array_equal(
            observed[group].to_numpy(dtype=np.float64),
            np.round(final[group].to_numpy(dtype=np.float64), 6),
        ):
            raise AssertionError(f"final CSV round-trip mismatch for {group}")
    final_audit = {
        "active_rows_2025": int(state.sum()),
        "active_by_month": {
            str(month): int(state[state.index.month == month].sum())
            for month in range(1, 13)
        },
        "mean_parent_g3_active_kwh": float(parent.loc[state, G3].mean())
        if state.any()
        else None,
        "mean_final_g3_active_kwh": float(final.loc[state, G3].mean())
        if state.any()
        else None,
        "g1_g2_exact_parent_identity": True,
        "g3_offrisk_exact_parent_identity": True,
        "csv": record(csv_path),
        "prediction": record(out_dir / "final_prediction_kwh_2025.parquet"),
        "physics_diagnostics": record(out_dir / "final_physics_diagnostics_2025.parquet"),
    }
    return csv_path, final_audit


def run_confirmation(out_dir: Path, config: dict[str, Any]) -> bool:
    stage1_lock_path = out_dir / "stage1_promotion_lock.json"
    if not stage1_lock_path.exists():
        raise FileNotFoundError("stage1 promotion lock is missing")
    stage1_lock = json.loads(stage1_lock_path.read_text(encoding="utf-8"))
    if not stage1_lock.get("passed"):
        raise RuntimeError("stage1 did not pass")
    results_path = out_dir / "confirmation_results.json"
    candidate_lock_path = out_dir / "confirmation_candidate_before_labels.json"
    if results_path.exists() or candidate_lock_path.exists():
        raise FileExistsError("confirmation output already exists; refusing to overwrite")
    source_lock_path = out_dir / "source_lock_before_2024_candidate.json"
    atomic_json(
        {
            "created_utc": utc_now(),
            "config": record(CONFIG),
            "sidecar": record(CONFIG_SIDECAR),
            "runner": record(Path(__file__)),
            "stage1_lock": record(stage1_lock_path),
            "locked_inputs_verified": True,
            "new_2024_candidate_values_read_before_this_lock": 0,
            "new_2024_candidate_metrics_before_this_lock": 0,
            "new_2025_values_decoded": 0,
        },
        source_lock_path,
    )
    expected = pd.date_range(
        "2024-01-01 01:00:00", "2025-01-01 00:00:00", freq="h", name="kst_dtm"
    )
    recent = read_prediction(BASE_2024, expected)
    base = recent.copy()
    for group in TARGET_COLS:
        base[group] = np.clip(0.97 * base[group], 0.0, 1.02 * CAPACITY_KWH[group])
    weather = read_weather_window(
        WEATHER_TRAIN, "2024-01-01 01:00:00", "2025-01-01 00:00:00"
    )
    state, diagnostics = icing_state(weather)
    candidate = base.copy()
    candidate.loc[state, G3] = DERATE * base.loc[state, G3]
    if not np.array_equal(
        candidate[["kpx_group_1", "kpx_group_2"]].to_numpy(),
        base[["kpx_group_1", "kpx_group_2"]].to_numpy(),
    ):
        raise AssertionError("confirmation G1/G2 identity changed")
    atomic_parquet(base, out_dir / "confirmation_base_scale097_2024.parquet")
    atomic_parquet(candidate, out_dir / "confirmation_candidate_2024.parquet")
    atomic_parquet(diagnostics, out_dir / "confirmation_physics_diagnostics_2024.parquet")
    atomic_json(
        {
            "created_utc": utc_now(),
            "status": "SINGLE_2024_CANDIDATE_HASHED_BEFORE_LABEL_DECODE",
            "config_sha256": CONFIG_SHA256,
            "candidate_count": 1,
            "active_rows": int(state.sum()),
            "base": record(out_dir / "confirmation_base_scale097_2024.parquet"),
            "candidate": record(out_dir / "confirmation_candidate_2024.parquet"),
            "physics": record(out_dir / "confirmation_physics_diagnostics_2024.parquet"),
            "2024_label_values_decoded_before_candidate_hash_lock": 0,
            "no_reselection_or_retuning": True,
        },
        candidate_lock_path,
    )
    # Only after the immutable candidate lock is written do we decode the full
    # label file and materialize 2024 actual values.
    labels = pd.read_csv(
        LABELS, encoding="utf-8-sig", parse_dates=["kst_dtm"]
    ).set_index("kst_dtm")
    actual = labels.reindex(expected).loc[:, list(TARGET_COLS)].astype(np.float64)
    metrics: dict[str, Any] = {}
    slice_passes: dict[str, bool] = {}
    for name, (start, end) in SLICES_2024.items():
        index = pd.date_range(start, end, freq="h")
        active = int(state.reindex(index).sum())
        result = slice_comparison(actual, base, candidate, index)
        result["active_rows"] = active
        if active == 0:
            identical = np.array_equal(
                base.loc[index].to_numpy(), candidate.loc[index].to_numpy()
            )
            passed = bool(
                identical
                and all(
                    result[scope][component] == 0.0
                    for scope in ("g3", "mixed")
                    for component in ("delta_total", "delta_one_minus_nmae", "delta_ficr")
                )
            )
        else:
            passed = bool(
                result["g3"]["delta_total"] > 0.0
                and result["g3"]["delta_one_minus_nmae"] >= 0.0
                and result["g3"]["delta_ficr"] >= 0.0
                and result["mixed"]["delta_total"] > 0.0
                and result["mixed"]["delta_one_minus_nmae"] >= 0.0
                and result["mixed"]["delta_ficr"] >= 0.0
            )
        metrics[name] = result
        slice_passes[name] = passed
    mandatory = ("full", "H2", "Q4")
    mandatory_strict = all(
        metrics[name]["active_rows"] > 0
        and all(
            metrics[name][scope][component] > 0.0
            for scope in ("g3", "mixed")
            for component in ("delta_total", "delta_one_minus_nmae", "delta_ficr")
        )
        for name in mandatory
    )
    passed = bool(all(slice_passes.values()) and mandatory_strict)
    payload: dict[str, Any] = {
        "artifact_type": "icing_regime_physics_g3_single_2024_confirmation",
        "created_utc": utc_now(),
        "config_sha256": CONFIG_SHA256,
        "candidate_lock": record(candidate_lock_path),
        "candidate_count": 1,
        "active_rows_2024": int(state.sum()),
        "active_by_month": {
            str(month): int(state[state.index.month == month].sum())
            for month in range(1, 13)
        },
        "metrics": metrics,
        "slice_passes": slice_passes,
        "mandatory_full_h2_q4_all_components_strict": mandatory_strict,
        "passed": passed,
        "no_reselection_or_retuning": True,
        "public_feedback_subgroup_or_time_inversion_used": False,
    }
    if passed:
        csv_path, audit = final_csv(out_dir, config)
        payload["status"] = "PROMOTED_AND_FINAL_CSV_CREATED"
        payload["final"] = audit
        payload["csv_path"] = csv_path.resolve().as_posix()
    else:
        payload["status"] = "REJECTED_NO_2025_READ_NO_CSV"
        payload["failure_contract"] = config["single_2024_confirmation"]["failure"]
    atomic_json(payload, results_path)
    files = [
        source_lock_path,
        candidate_lock_path,
        out_dir / "confirmation_base_scale097_2024.parquet",
        out_dir / "confirmation_candidate_2024.parquet",
        out_dir / "confirmation_physics_diagnostics_2024.parquet",
        results_path,
    ]
    if passed:
        files.extend(
            [
                out_dir / "final_prediction_kwh_2025.parquet",
                out_dir / "final_physics_diagnostics_2025.parquet",
                out_dir / Path(config["final_if_and_only_if_pass"]["output"]).name,
            ]
        )
    manifest_path = out_dir / "manifest.json"
    atomic_json(
        {
            "artifact_type": "icing_regime_physics_g3_strict_v1",
            "created_utc": utc_now(),
            "config": record(CONFIG),
            "runner": record(Path(__file__)),
            "stage1": record(out_dir / "stage1_results.json"),
            "confirmation_passed": passed,
            "csv_created": passed,
            "outputs_excluding_manifest": [record(path) for path in files],
        },
        manifest_path,
    )
    (out_dir / "manifest.sha256").write_text(
        f"{sha256_file(manifest_path)}  manifest.json\n", encoding="ascii"
    )
    return passed


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    if out_dir == ROOT.resolve() or ROOT.resolve() not in out_dir.parents:
        raise ValueError("out-dir must be a distinct directory inside the project")
    config = load_config()
    verify_locked_inputs(config)
    if args.stage in ("stage1", "all"):
        passed = run_stage1(out_dir, config)
        print(json.dumps({"stage": "stage1", "passed": passed}, ensure_ascii=False))
        if not passed:
            return
    if args.stage in ("confirm", "all"):
        passed = run_confirmation(out_dir, config)
        print(json.dumps({"stage": "confirmation", "passed": passed}, ensure_ascii=False))


if __name__ == "__main__":
    main()

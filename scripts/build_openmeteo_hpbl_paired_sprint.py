"""Last-48-hour Open-Meteo GFS boundary-layer-height paired sprint.

The experiment is deliberately narrow:

* source: ``boundary_layer_height_previous_day2`` from Open-Meteo's NCEP GFS
  Previous Runs archive at the 17 locked turbine coordinates;
* fit: one fixed pooled LightGBM-L1 residual recipe on 2024 H1 only;
* gate: untouched 2024 H2 only, with one fixed 10% blend into the matching
  official-data-only baseline; and
* deployment: only after every pre-fixed gate passes, refit on full 2024 and
  create ``04_EXTERNAL_HPBL_HIGH_UPSIDE_PROBE.csv`` for 2025.

No Public score or component is read.  Raw response bytes, exact URLs, HTTP
headers, response hashes, normalized coverage and the external-source caveat
are retained below ``artifacts/final_submission_sprint_20260812/hpbl_sprint``.
If source coverage or the H2 gate fails, the script writes evidence and a
``NO_CSV_GENERATED.txt`` marker but no submission CSV.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metric import CAPACITY_KWH, group_metrics, score_details


RAW = Path(r"data/local/open")
OUT = ROOT / "artifacts" / "final_submission_sprint_20260812" / "hpbl_sprint"
RAW_OUT = OUT / "raw"
LABELS_PATH = RAW / "train" / "train_labels.csv"
SAMPLE_PATH = RAW / "sample_submission.csv"
COORDINATE_LOCK_PATH = (
    ROOT
    / "artifacts"
    / "baram2026_ncei_scada_longrun_20260810_v2"
    / "prereg"
    / "authoritative_turbine_coordinate_lock_v1.json"
)
BASELINE_2024_PATH = (
    ROOT
    / "artifacts"
    / "oof"
    / "gate2024_recent_v4_cf_fix_calibration_fit.parquet"
)
BASELINE_2025_PATH = (
    ROOT
    / "artifacts"
    / "final_submission_sprint_20260812"
    / "01_OFFICIAL_ONLY_SAFE.csv"
)
SUBMISSION_PATH = OUT / "04_EXTERNAL_HPBL_HIGH_UPSIDE_PROBE.csv"

ENDPOINT = "https://previous-runs-api.open-meteo.com/v1/forecast"
DOCS = {
    "previous_runs": "https://open-meteo.com/en/docs/previous-runs-api",
    "gfs": "https://open-meteo.com/en/docs/gfs-api",
}
MODEL = "gfs_global"
VARIABLE = "boundary_layer_height_previous_day2"
TIMEZONE = "Asia/Seoul"
GROUPS = ("kpx_group_1", "kpx_group_2", "kpx_group_3")

# The main request is exactly the requested two calendar years.  The one-day
# edge request supplies the sample's final interval-ending timestamp,
# 2026-01-01 00:00 KST, without changing the requested 2024-2025 archive span.
REQUEST_WINDOWS = (
    ("calendar_2024_2025", "2024-01-01", "2025-12-31"),
    ("sample_edge_2026_01_01", "2026-01-01", "2026-01-01"),
)

CURRENT_BASE_SCALE = 0.97
BLEND_WEIGHT = 0.10
MAX_MISSING_FRACTION = 0.01
MAX_ABS_DIRECT_RESIDUAL_CF = 0.35
PUBLIC_FEEDBACK_USED = False

# Fixed before labels are read.  This is one small recipe, not an HPO grid.
MODEL_RECIPE: dict[str, Any] = {
    "objective": "regression_l1",
    "n_estimators": 140,
    "learning_rate": 0.03,
    "num_leaves": 7,
    "max_depth": 3,
    "min_child_samples": 240,
    "max_bin": 63,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_alpha": 1.0,
    "reg_lambda": 10.0,
}
MODEL_FEATURES = (
    "hpbl_m",
    "hpbl_x_base_cf",
    "lead_hour",
    "hour_sin",
    "hour_cos",
    "group_id",
)

GATE_CONTRACT = {
    "h2_aggregate_min_delta_score": 0.003,
    "h2_aggregate_min_delta_ficr": 0.004,
    "h2_each_group_min_delta_score": 0.0,
    "h2_late_lead_17_24_min_delta_score": 0.0,
    "h2_worst_calendar_month_min_delta_score": -0.003,
}

SOURCE_CAVEAT = (
    "Open-Meteo is a third-party transformed archive of NCEP GFS. "
    "Its documented _previous_day2 series is fixed at 48 hours before each "
    "valid time, but the response is not an immutable NCEP run object and "
    "does not independently prove the exact D-2 12Z f028-f051 run/lead "
    "sequence. Model-version, ingestion-time and upstream-object identity "
    "cannot be reconstructed from this response alone."
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def write_no_csv(reason: str) -> dict[str, Any]:
    marker = OUT / "NO_CSV_GENERATED.txt"
    marker.write_text(
        "No HPBL submission CSV was generated.\n"
        f"Reason: {reason}\n"
        "The sprint contract permits a CSV only after coverage and every H2 "
        "gate pass.\n",
        encoding="utf-8",
    )
    return file_record(marker)


def exact_url(latitude: float, longitude: float, start_date: str, end_date: str) -> str:
    query = urllib.parse.urlencode(
        {
            "latitude": f"{latitude:.12f}",
            "longitude": f"{longitude:.12f}",
            "start_date": start_date,
            "end_date": end_date,
            "timezone": TIMEZONE,
            "models": MODEL,
            "cell_selection": "nearest",
            "hourly": VARIABLE,
        }
    )
    return f"{ENDPOINT}?{query}"


def fetch_one(
    *, site: dict[str, Any], window_name: str, start_date: str, end_date: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fetch one raw response, preserving bytes, URL, headers and hashes."""

    site_id = int(site["site_id"])
    stem = f"site_{site_id:02d}__{window_name}"
    body_path = RAW_OUT / f"{stem}.response.json"
    headers_path = RAW_OUT / f"{stem}.headers.json"
    request_path = RAW_OUT / f"{stem}.request.json"
    url = exact_url(float(site["latitude"]), float(site["longitude"]), start_date, end_date)

    if body_path.exists() and headers_path.exists() and request_path.exists():
        request_record = json.loads(request_path.read_text(encoding="utf-8"))
        if request_record.get("url") != url:
            raise RuntimeError(f"cached URL mismatch for site {site_id} {window_name}")
        raw = body_path.read_bytes()
        if request_record.get("response_sha256") != sha256_bytes(raw):
            raise RuntimeError(f"cached response hash mismatch for site {site_id} {window_name}")
        payload = json.loads(raw.decode("utf-8"))
        request_record["cache_reused"] = True
        return payload, request_record

    requested_utc = utc_now()
    headers = {
        "Accept": "application/json",
        "User-Agent": "baram2026-hpbl-paired-sprint/1.0",
    }
    last_error: BaseException | None = None
    for attempt, delay in enumerate((0, 2, 5, 10), start=1):
        if delay:
            time.sleep(delay)
        request = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                raw = response.read()
                status = int(getattr(response, "status", response.getcode()))
                response_headers = dict(response.headers.items())
            body_path.write_bytes(raw)
            write_json(headers_path, response_headers)
            record = {
                "site_id": site_id,
                "group": site["group"],
                "requested_coordinate": {
                    "latitude": float(site["latitude"]),
                    "longitude": float(site["longitude"]),
                },
                "window": window_name,
                "start_date": start_date,
                "end_date": end_date,
                "method": "GET",
                "url": url,
                "requested_utc": requested_utc,
                "completed_utc": utc_now(),
                "attempt": attempt,
                "http_status": status,
                "response_sha256": sha256_bytes(raw),
                "response_bytes": len(raw),
                "response_path": str(body_path.resolve()),
                "response_headers_path": str(headers_path.resolve()),
                "cache_reused": False,
            }
            write_json(request_path, record)
            if status != 200:
                raise RuntimeError(f"Open-Meteo returned HTTP {status} for {url}")
            return json.loads(raw.decode("utf-8")), record
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            body_path.write_bytes(raw)
            response_headers = dict(exc.headers.items()) if exc.headers else {}
            write_json(headers_path, response_headers)
            record = {
                "site_id": site_id,
                "group": site["group"],
                "requested_coordinate": {
                    "latitude": float(site["latitude"]),
                    "longitude": float(site["longitude"]),
                },
                "window": window_name,
                "start_date": start_date,
                "end_date": end_date,
                "method": "GET",
                "url": url,
                "requested_utc": requested_utc,
                "completed_utc": utc_now(),
                "attempt": attempt,
                "http_status": int(exc.code),
                "response_sha256": sha256_bytes(raw),
                "response_bytes": len(raw),
                "response_path": str(body_path.resolve()),
                "response_headers_path": str(headers_path.resolve()),
                "error": str(exc),
                "cache_reused": False,
            }
            write_json(request_path, record)
            last_error = exc
            if 400 <= exc.code < 500 and exc.code != 429:
                break
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            write_json(
                request_path,
                {
                    "site_id": site_id,
                    "group": site["group"],
                    "window": window_name,
                    "start_date": start_date,
                    "end_date": end_date,
                    "method": "GET",
                    "url": url,
                    "requested_utc": requested_utc,
                    "failed_utc": utc_now(),
                    "attempt": attempt,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
    raise RuntimeError(
        f"Open-Meteo request failed for site {site_id} {window_name}: {last_error}"
    )


def parse_response(payload: dict[str, Any], record: dict[str, Any]) -> pd.Series:
    if payload.get("error"):
        raise RuntimeError(f"Open-Meteo API error: {payload}")
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict) or "time" not in hourly or VARIABLE not in hourly:
        raise RuntimeError(
            f"response lacks hourly time/{VARIABLE} for site {record['site_id']}"
        )
    timestamps = pd.to_datetime(hourly["time"], errors="raise")
    values = pd.to_numeric(pd.Series(hourly[VARIABLE]), errors="coerce").to_numpy(np.float64)
    if len(timestamps) != len(values):
        raise RuntimeError("Open-Meteo time/value length mismatch")
    if pd.DatetimeIndex(timestamps).has_duplicates:
        raise RuntimeError("Open-Meteo response contains duplicate timestamps")
    record["returned_coordinate"] = {
        "latitude": payload.get("latitude"),
        "longitude": payload.get("longitude"),
        "elevation": payload.get("elevation"),
    }
    record["timezone"] = payload.get("timezone")
    record["utc_offset_seconds"] = payload.get("utc_offset_seconds")
    record["unit"] = (payload.get("hourly_units") or {}).get(VARIABLE)
    record["rows"] = int(len(values))
    record["non_null_rows"] = int(np.isfinite(values).sum())
    record["first_timestamp"] = timestamps.min()
    record["last_timestamp"] = timestamps.max()
    finite_index = pd.DatetimeIndex(timestamps)[np.isfinite(values)]
    record["first_non_null"] = finite_index.min() if len(finite_index) else None
    record["last_non_null"] = finite_index.max() if len(finite_index) else None
    return pd.Series(values, index=pd.DatetimeIndex(timestamps), dtype=np.float64)


def load_sites() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lock = json.loads(COORDINATE_LOCK_PATH.read_text(encoding="utf-8"))
    sites = lock.get("sites")
    if not isinstance(sites, list) or len(sites) != 17:
        raise RuntimeError("coordinate lock does not contain exactly 17 sites")
    if {site.get("group") for site in sites} != set(GROUPS):
        raise RuntimeError("coordinate lock group closure mismatch")
    if len({int(site["site_id"]) for site in sites}) != 17:
        raise RuntimeError("coordinate lock site IDs are not unique")
    return sorted(sites, key=lambda item: int(item["site_id"])), lock


def aggregate_groups(site_frame: pd.DataFrame, sites: list[dict[str, Any]]) -> pd.DataFrame:
    result = pd.DataFrame(index=site_frame.index, columns=list(GROUPS), dtype=np.float64)
    for group in GROUPS:
        selected = [site for site in sites if site["group"] == group]
        columns = [f"site_{int(site['site_id']):02d}" for site in selected]
        weights = pd.Series(
            {f"site_{int(site['site_id']):02d}": float(site["capacity_mw"]) for site in selected}
        )
        values = site_frame.loc[:, columns]
        numerator = values.mul(weights, axis=1).sum(axis=1, min_count=1)
        denominator = values.notna().mul(weights, axis=1).sum(axis=1)
        result[group] = numerator / denominator.replace(0.0, np.nan)
    return result


def coverage_one(frame: pd.DataFrame, index: pd.DatetimeIndex) -> dict[str, Any]:
    selected = frame.reindex(index)
    output: dict[str, Any] = {}
    for column in selected:
        series = selected[column]
        finite = np.isfinite(series.to_numpy(np.float64))
        finite_index = index[finite]
        output[column] = {
            "expected_rows": int(len(index)),
            "non_null_rows": int(finite.sum()),
            "missing_rows": int((~finite).sum()),
            "missing_fraction": float((~finite).mean()),
            "first_non_null": finite_index.min() if len(finite_index) else None,
            "last_non_null": finite_index.max() if len(finite_index) else None,
        }
    return output


def lead_hour(index: pd.DatetimeIndex) -> np.ndarray:
    hours = index.hour.to_numpy(dtype=np.int16)
    return np.where(hours == 0, 24, hours).astype(np.int16)


def interval_start(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index - pd.Timedelta(hours=1)


def load_2024_baseline(expected_index: pd.DatetimeIndex) -> pd.DataFrame:
    raw = pd.read_parquet(BASELINE_2024_PATH).loc[:, list(GROUPS)].astype(np.float64)
    raw.index = pd.DatetimeIndex(pd.to_datetime(raw.index, errors="raise"))
    if not raw.index.equals(expected_index):
        raise RuntimeError("2024 matching baseline index differs from label contract")
    baseline = CURRENT_BASE_SCALE * raw
    for group in GROUPS:
        baseline[group] = baseline[group].clip(0.0, CAPACITY_KWH[group])
    return baseline


def make_long_features(
    *,
    index: pd.DatetimeIndex,
    baseline: pd.DataFrame,
    hpbl: pd.DataFrame,
    labels: pd.DataFrame | None,
) -> pd.DataFrame:
    lead = lead_hour(index).astype(np.float64)
    radians = 2.0 * np.pi * lead / 24.0
    parts: list[pd.DataFrame] = []
    for group_number, group in enumerate(GROUPS, start=1):
        base_cf = baseline[group].reindex(index).to_numpy(np.float64) / CAPACITY_KWH[group]
        hpbl_values = hpbl[group].reindex(index).to_numpy(np.float64)
        part = pd.DataFrame(
            {
                "timestamp": index,
                "group": group,
                "group_id": float(group_number),
                "hpbl_m": hpbl_values,
                "hpbl_x_base_cf": hpbl_values * base_cf,
                "lead_hour": lead,
                "hour_sin": np.sin(radians),
                "hour_cos": np.cos(radians),
                "base_cf": base_cf,
            }
        )
        if labels is not None:
            part["actual_cf"] = (
                labels[group].reindex(index).to_numpy(np.float64) / CAPACITY_KWH[group]
            )
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def make_model() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        **MODEL_RECIPE,
        random_state=20260812,
        bagging_seed=20260842,
        feature_fraction_seed=20260872,
        deterministic=True,
        force_col_wise=True,
        n_jobs=max(1, min(10, os.cpu_count() or 1)),
        verbosity=-1,
    )


def fit_residual_model(long_frame: pd.DataFrame) -> tuple[lgb.LGBMRegressor, dict[str, Any]]:
    eligible = (
        np.isfinite(long_frame["actual_cf"].to_numpy(np.float64))
        & (long_frame["actual_cf"].to_numpy(np.float64) >= 0.10)
        & np.isfinite(long_frame["base_cf"].to_numpy(np.float64))
    )
    if int(eligible.sum()) == 0:
        raise RuntimeError("no metric-eligible H1 residual training rows")
    train = long_frame.loc[eligible]
    target = train["actual_cf"].to_numpy(np.float64) - train["base_cf"].to_numpy(np.float64)
    model = make_model()
    model.fit(train.loc[:, list(MODEL_FEATURES)], target)
    return model, {
        "eligible_rows": int(len(train)),
        "total_rows": int(len(long_frame)),
        "target_residual_cf_min": float(np.min(target)),
        "target_residual_cf_max": float(np.max(target)),
        "target_residual_cf_mean": float(np.mean(target)),
        "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
    }


def predict_wide(
    model: lgb.LGBMRegressor, long_frame: pd.DataFrame, index: pd.DatetimeIndex
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predicted_residual = np.asarray(
        model.predict(long_frame.loc[:, list(MODEL_FEATURES)]), dtype=np.float64
    )
    predicted_residual = np.clip(
        predicted_residual, -MAX_ABS_DIRECT_RESIDUAL_CF, MAX_ABS_DIRECT_RESIDUAL_CF
    )
    direct_cf = np.clip(
        long_frame["base_cf"].to_numpy(np.float64) + predicted_residual, 0.0, 1.0
    )
    enriched = long_frame.loc[:, ["timestamp", "group", "base_cf"]].copy()
    enriched["predicted_residual_cf"] = predicted_residual
    enriched["direct_cf"] = direct_cf
    direct = enriched.pivot(index="timestamp", columns="group", values="direct_cf")
    direct = direct.reindex(index=index, columns=list(GROUPS))
    direct_kwh = direct.copy()
    for group in GROUPS:
        direct_kwh[group] = direct_kwh[group] * CAPACITY_KWH[group]
    return direct_kwh, enriched


def blend_candidate(baseline: pd.DataFrame, direct: pd.DataFrame) -> pd.DataFrame:
    candidate = (1.0 - BLEND_WEIGHT) * baseline + BLEND_WEIGHT * direct
    for group in GROUPS:
        candidate[group] = candidate[group].clip(0.0, CAPACITY_KWH[group])
    return candidate


def metric_dict(actual: pd.DataFrame, prediction: pd.DataFrame) -> dict[str, Any]:
    detail = score_details(actual, prediction, target_cols=GROUPS, capacities=CAPACITY_KWH)
    return detail.as_dict()


def delta_summary(actual: pd.DataFrame, baseline: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, Any]:
    base = score_details(actual, baseline, target_cols=GROUPS, capacities=CAPACITY_KWH)
    cand = score_details(actual, candidate, target_cols=GROUPS, capacities=CAPACITY_KWH)
    return {
        "baseline": base.as_dict(),
        "candidate": cand.as_dict(),
        "delta_score": cand.total_score - base.total_score,
        "delta_one_minus_nmae": cand.one_minus_nmae - base.one_minus_nmae,
        "delta_ficr": cand.ficr - base.ficr,
    }


def group_delta_rows(
    actual: pd.DataFrame, baseline: pd.DataFrame, candidate: pd.DataFrame
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in GROUPS:
        base = group_metrics(actual[group], baseline[group], CAPACITY_KWH[group], group_name=group)
        cand = group_metrics(actual[group], candidate[group], CAPACITY_KWH[group], group_name=group)
        base_score = 0.5 * base.one_minus_nmae + 0.5 * base.ficr
        cand_score = 0.5 * cand.one_minus_nmae + 0.5 * cand.ficr
        rows.append(
            {
                "group": group,
                "base_score": base_score,
                "candidate_score": cand_score,
                "delta_score": cand_score - base_score,
                "delta_one_minus_nmae": cand.one_minus_nmae - base.one_minus_nmae,
                "delta_ficr": cand.ficr - base.ficr,
            }
        )
    return rows


def evaluate_h2_gate(
    *, actual: pd.DataFrame, baseline: pd.DataFrame, candidate: pd.DataFrame
) -> dict[str, Any]:
    aggregate = delta_summary(actual, baseline, candidate)
    groups = group_delta_rows(actual, baseline, candidate)
    lead = lead_hour(actual.index)
    late_index = actual.index[lead >= 17]
    late = delta_summary(
        actual.loc[late_index], baseline.loc[late_index], candidate.loc[late_index]
    )
    start = interval_start(actual.index)
    monthly: list[dict[str, Any]] = []
    for month in sorted(set(start.month)):
        month_index = actual.index[start.month == month]
        summary = delta_summary(
            actual.loc[month_index], baseline.loc[month_index], candidate.loc[month_index]
        )
        monthly.append(
            {
                "month": int(month),
                "rows": int(len(month_index)),
                "delta_score": summary["delta_score"],
                "delta_one_minus_nmae": summary["delta_one_minus_nmae"],
                "delta_ficr": summary["delta_ficr"],
            }
        )
    checks = {
        "aggregate_delta_score": bool(
            aggregate["delta_score"] >= GATE_CONTRACT["h2_aggregate_min_delta_score"]
        ),
        "aggregate_delta_ficr": bool(
            aggregate["delta_ficr"] >= GATE_CONTRACT["h2_aggregate_min_delta_ficr"]
        ),
        "each_group_delta_score": bool(
            min(row["delta_score"] for row in groups)
            >= GATE_CONTRACT["h2_each_group_min_delta_score"]
        ),
        "late_lead_17_24_delta_score": bool(
            late["delta_score"] >= GATE_CONTRACT["h2_late_lead_17_24_min_delta_score"]
        ),
        "worst_month_delta_score": bool(
            min(row["delta_score"] for row in monthly)
            >= GATE_CONTRACT["h2_worst_calendar_month_min_delta_score"]
        ),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "aggregate": aggregate,
        "groups": groups,
        "late_lead_17_24": late,
        "months": monthly,
        "worst_group_delta_score": min(row["delta_score"] for row in groups),
        "worst_month_delta_score": min(row["delta_score"] for row in monthly),
    }


def validate_submission(path: Path, sample: pd.DataFrame) -> dict[str, Any]:
    reread = pd.read_csv(path, encoding="utf-8-sig")
    values = reread.loc[:, list(GROUPS)].to_numpy(np.float64)
    checks = {
        "rows_exact_8760": len(reread) == 8760,
        "columns_exact": reread.columns.tolist() == sample.columns.tolist(),
        "forecast_id_exact": reread["forecast_id"].equals(sample["forecast_id"]),
        "forecast_kst_dtm_exact": reread["forecast_kst_dtm"].equals(
            sample["forecast_kst_dtm"]
        ),
        "utf8_bom": path.read_bytes()[:3] == b"\xef\xbb\xbf",
        "finite": bool(np.isfinite(values).all()),
        "within_capacity": bool(
            all(
                reread[group].between(0.0, CAPACITY_KWH[group], inclusive="both").all()
                for group in GROUPS
            )
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"submission validation failed: {checks}")
    return checks


def source_manifest_base(
    *, sites: list[dict[str, Any]], source_records: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "artifact_type": "OPENMETEO_GFS_HPBL_PREVIOUS_DAY2_PAIRED_SPRINT",
        "created_utc": utc_now(),
        "external_data_used": True,
        "public_feedback_used": PUBLIC_FEEDBACK_USED,
        "provider": "Open-Meteo Previous Model Runs API",
        "upstream_model_requested": "NCEP GFS Global",
        "endpoint": ENDPOINT,
        "model_parameter": MODEL,
        "hourly_variable": VARIABLE,
        "availability_semantics": (
            "Open-Meteo documents _previous_day2 as the value predicted exactly "
            "48 hours before valid time."
        ),
        "official_documentation": DOCS,
        "request_windows": [
            {"name": name, "start_date": start, "end_date": end}
            for name, start, end in REQUEST_WINDOWS
        ],
        "timezone": TIMEZONE,
        "cell_selection": "nearest",
        "coordinate_lock": file_record(COORDINATE_LOCK_PATH),
        "site_count": len(sites),
        "sites": [
            {
                "site_id": int(site["site_id"]),
                "group": site["group"],
                "latitude": float(site["latitude"]),
                "longitude": float(site["longitude"]),
                "capacity_mw": float(site["capacity_mw"]),
            }
            for site in sites
        ],
        "requests": source_records,
        "verification_caveat": SOURCE_CAVEAT,
    }


def run() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    RAW_OUT.mkdir(parents=True, exist_ok=True)
    if SUBMISSION_PATH.exists():
        raise RuntimeError(
            f"refusing to overwrite existing submission without review: {SUBMISSION_PATH}"
        )

    sites, _ = load_sites()
    source_records: list[dict[str, Any]] = []
    site_series: dict[int, list[pd.Series]] = {int(site["site_id"]): [] for site in sites}
    for site in sites:
        for window_name, start_date, end_date in REQUEST_WINDOWS:
            payload, record = fetch_one(
                site=site,
                window_name=window_name,
                start_date=start_date,
                end_date=end_date,
            )
            series = parse_response(payload, record)
            write_json(
                Path(record["response_path"]).with_suffix(".parsed_metadata.json"), record
            )
            source_records.append(record)
            site_series[int(site["site_id"])].append(series)
        print(
            json.dumps(
                {
                    "status": "downloaded",
                    "site_id": int(site["site_id"]),
                    "completed_sites": len([key for key, value in site_series.items() if value]),
                    "total_sites": len(sites),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    combined: dict[str, pd.Series] = {}
    for site in sites:
        site_id = int(site["site_id"])
        series = pd.concat(site_series[site_id]).sort_index()
        series = series[~series.index.duplicated(keep="last")]
        combined[f"site_{site_id:02d}"] = series
    site_frame = pd.DataFrame(combined).sort_index()
    group_frame = aggregate_groups(site_frame, sites)
    site_path = OUT / "hpbl_site_valid_time_kst.parquet"
    group_path = OUT / "hpbl_group_capacity_weighted_valid_time_kst.parquet"
    site_frame.to_parquet(site_path)
    group_frame.to_parquet(group_path)

    train_2024_index = pd.date_range(
        "2024-01-01 01:00:00", "2025-01-01 00:00:00", freq="h"
    )
    sample_header = pd.read_csv(SAMPLE_PATH, encoding="utf-8-sig")
    test_2025_index = pd.DatetimeIndex(
        pd.to_datetime(sample_header["forecast_kst_dtm"], errors="raise")
    )
    if len(train_2024_index) != 8784 or len(test_2025_index) != 8760:
        raise RuntimeError("expected hourly year contracts changed")
    starts_2024 = interval_start(train_2024_index)
    h1_index = train_2024_index[starts_2024.month <= 6]
    h2_index = train_2024_index[starts_2024.month >= 7]

    coverage = {
        "threshold_max_missing_fraction": MAX_MISSING_FRACTION,
        "split_definition": (
            "generation labels are interval-ending; H1/H2 use timestamp minus one "
            "hour so 2024-07-01 00:00 belongs to June/H1"
        ),
        "H1_2024": {
            "start": h1_index.min(),
            "end": h1_index.max(),
            "rows": len(h1_index),
            "sites": coverage_one(site_frame, h1_index),
            "groups": coverage_one(group_frame, h1_index),
        },
        "H2_2024": {
            "start": h2_index.min(),
            "end": h2_index.max(),
            "rows": len(h2_index),
            "groups": coverage_one(group_frame, h2_index),
        },
        "deployment_2025": {
            "start": test_2025_index.min(),
            "end": test_2025_index.max(),
            "rows": len(test_2025_index),
            "groups": coverage_one(group_frame, test_2025_index),
        },
    }
    h1_site_worst = max(
        row["missing_fraction"] for row in coverage["H1_2024"]["sites"].values()
    )
    h1_group_worst = max(
        row["missing_fraction"] for row in coverage["H1_2024"]["groups"].values()
    )
    h2_group_worst = max(
        row["missing_fraction"] for row in coverage["H2_2024"]["groups"].values()
    )
    deployment_group_worst = max(
        row["missing_fraction"]
        for row in coverage["deployment_2025"]["groups"].values()
    )
    coverage_checks = {
        "H1_each_site_missing_le_1pct": h1_site_worst <= MAX_MISSING_FRACTION,
        "H1_each_group_missing_le_1pct": h1_group_worst <= MAX_MISSING_FRACTION,
        "H2_each_group_missing_le_1pct": h2_group_worst <= MAX_MISSING_FRACTION,
        "deployment_each_group_missing_le_1pct": (
            deployment_group_worst <= MAX_MISSING_FRACTION
        ),
    }
    coverage["worst_missing_fraction"] = {
        "H1_site": h1_site_worst,
        "H1_group": h1_group_worst,
        "H2_group": h2_group_worst,
        "deployment_group": deployment_group_worst,
    }
    coverage["checks"] = coverage_checks
    coverage["passed"] = bool(all(coverage_checks.values()))
    coverage_path = OUT / "COVERAGE.json"
    write_json(coverage_path, coverage)

    manifest = source_manifest_base(sites=sites, source_records=source_records)
    manifest["normalized"] = {
        "site": file_record(site_path),
        "group_capacity_weighted": file_record(group_path),
        "coverage": file_record(coverage_path),
    }
    manifest["coverage_gate"] = coverage

    if not coverage["passed"]:
        if not coverage_checks["H1_each_site_missing_le_1pct"] or not coverage_checks[
            "H1_each_group_missing_le_1pct"
        ]:
            stop_code = "STOP_H1_ARCHIVE_COVERAGE"
        else:
            stop_code = "STOP_H2_OR_DEPLOYMENT_ARCHIVE_COVERAGE"
        marker = write_no_csv(stop_code)
        result = {
            "status": stop_code,
            "submission_generated": False,
            "labels_read": False,
            "model_fit": False,
            "public_feedback_used": PUBLIC_FEEDBACK_USED,
            "coverage": coverage,
            "marker": marker,
        }
        results_path = OUT / "RESULTS.json"
        write_json(results_path, result)
        manifest["result"] = result
        manifest["results"] = file_record(results_path)
        write_json(OUT / "MANIFEST.json", manifest)
        return result

    # Coverage passed.  Only now may competition labels enter the experiment.
    labels = pd.read_csv(
        LABELS_PATH, encoding="utf-8-sig", parse_dates=["kst_dtm"]
    ).set_index("kst_dtm")
    labels = labels.loc[:, list(GROUPS)].astype(np.float64)
    if not train_2024_index.isin(labels.index).all():
        raise RuntimeError("2024 label coverage differs from fixed contract")
    baseline_2024 = load_2024_baseline(train_2024_index)

    h1_long = make_long_features(
        index=h1_index,
        baseline=baseline_2024.loc[h1_index],
        hpbl=group_frame,
        labels=labels,
    )
    h1_model, h1_fit = fit_residual_model(h1_long)
    h2_long = make_long_features(
        index=h2_index,
        baseline=baseline_2024.loc[h2_index],
        hpbl=group_frame,
        labels=None,
    )
    h2_direct, h2_long_predictions = predict_wide(h1_model, h2_long, h2_index)
    h2_baseline = baseline_2024.loc[h2_index, list(GROUPS)]
    h2_candidate = blend_candidate(h2_baseline, h2_direct)
    h2_actual = labels.loc[h2_index, list(GROUPS)]
    h2_gate = evaluate_h2_gate(
        actual=h2_actual, baseline=h2_baseline, candidate=h2_candidate
    )

    h2_predictions_path = OUT / "H2_UNTOUCHED_PREDICTIONS.parquet"
    pd.concat(
        {
            "baseline": h2_baseline,
            "direct": h2_direct,
            "blend_10pct": h2_candidate,
        },
        axis=1,
    ).to_parquet(h2_predictions_path)
    h2_long_path = OUT / "H2_RESIDUAL_HEAD_LONG.parquet"
    h2_long_predictions.to_parquet(h2_long_path, index=False)
    h2_gate_path = OUT / "H2_GATE.json"
    write_json(h2_gate_path, h2_gate)
    h1_model_path = OUT / "H1_FIXED_RECIPE_MODEL.txt"
    h1_model.booster_.save_model(str(h1_model_path))

    validation = {
        "design": "2024 H1 fit; 2024 H2 untouched gate only",
        "public_feedback_used": PUBLIC_FEEDBACK_USED,
        "recipe_count": 1,
        "blend_weights_evaluated": [BLEND_WEIGHT],
        "model_features": list(MODEL_FEATURES),
        "model_recipe": MODEL_RECIPE,
        "fit": h1_fit,
        "gate_contract": GATE_CONTRACT,
        "gate": h2_gate,
        "artifacts": {
            "h2_predictions": file_record(h2_predictions_path),
            "h2_long": file_record(h2_long_path),
            "h2_gate": file_record(h2_gate_path),
            "h1_model": file_record(h1_model_path),
        },
    }
    manifest["validation"] = validation

    if not h2_gate["passed"]:
        marker = write_no_csv("STOP_H2_METRIC_GATE")
        result = {
            "status": "STOP_H2_METRIC_GATE",
            "submission_generated": False,
            "labels_read": True,
            "model_fit": True,
            "public_feedback_used": PUBLIC_FEEDBACK_USED,
            "coverage_passed": True,
            "h2_gate": h2_gate,
            "marker": marker,
        }
        results_path = OUT / "RESULTS.json"
        write_json(results_path, result)
        manifest["result"] = result
        manifest["results"] = file_record(results_path)
        write_json(OUT / "MANIFEST.json", manifest)
        return result

    # PASS only: refit the same fixed recipe on every eligible 2024 row.
    full_long = make_long_features(
        index=train_2024_index,
        baseline=baseline_2024,
        hpbl=group_frame,
        labels=labels,
    )
    final_model, final_fit = fit_residual_model(full_long)

    sample = sample_header.copy()
    if sample.columns.tolist() != ["forecast_id", "forecast_kst_dtm", *GROUPS]:
        raise RuntimeError("sample submission schema changed")
    baseline_csv = pd.read_csv(BASELINE_2025_PATH, encoding="utf-8-sig")
    if not sample[["forecast_id", "forecast_kst_dtm"]].equals(
        baseline_csv[["forecast_id", "forecast_kst_dtm"]]
    ):
        raise RuntimeError("2025 baseline identifiers differ from sample")
    baseline_2025 = baseline_csv.set_index(test_2025_index).loc[:, list(GROUPS)].astype(
        np.float64
    )
    final_long = make_long_features(
        index=test_2025_index,
        baseline=baseline_2025,
        hpbl=group_frame,
        labels=None,
    )
    final_direct, final_long_predictions = predict_wide(
        final_model, final_long, test_2025_index
    )
    final_candidate = blend_candidate(baseline_2025, final_direct)
    submission = sample.copy()
    submission.loc[:, list(GROUPS)] = final_candidate.loc[
        test_2025_index, list(GROUPS)
    ].to_numpy(np.float64)
    submission.to_csv(
        SUBMISSION_PATH, index=False, encoding="utf-8-sig", float_format="%.6f"
    )
    checks = validate_submission(SUBMISSION_PATH, sample)
    final_direct_path = OUT / "FULL2024_FIT_2025_DIRECT.parquet"
    final_direct.to_parquet(final_direct_path)
    final_long_path = OUT / "FULL2024_FIT_2025_LONG.parquet"
    final_long_predictions.to_parquet(final_long_path, index=False)
    final_model_path = OUT / "FULL2024_FIXED_RECIPE_MODEL.txt"
    final_model.booster_.save_model(str(final_model_path))

    result = {
        "status": "PASS_H2_GATE_CSV_GENERATED",
        "submission_generated": True,
        "labels_read": True,
        "model_fit": True,
        "public_feedback_used": PUBLIC_FEEDBACK_USED,
        "coverage_passed": True,
        "h2_gate": h2_gate,
        "final_fit": final_fit,
        "blend_weight": BLEND_WEIGHT,
        "submission": file_record(SUBMISSION_PATH),
        "submission_checks": checks,
        "final_direct": file_record(final_direct_path),
        "final_long": file_record(final_long_path),
        "final_model": file_record(final_model_path),
    }
    results_path = OUT / "RESULTS.json"
    write_json(results_path, result)
    manifest["result"] = result
    manifest["results"] = file_record(results_path)
    write_json(OUT / "MANIFEST.json", manifest)
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        result = run()
    except Exception as exc:
        failure = {
            "status": "STOP_EXCEPTION",
            "submission_generated": False,
            "public_feedback_used": PUBLIC_FEEDBACK_USED,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "failed_utc": utc_now(),
        }
        if not SUBMISSION_PATH.exists():
            failure["marker"] = write_no_csv("STOP_EXCEPTION")
        write_json(OUT / "FAILURE.json", failure)
        print(json.dumps(json_ready(failure), indent=2, ensure_ascii=False), flush=True)
        raise
    print(json.dumps(json_ready(result), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

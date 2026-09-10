from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://www.ncei.noaa.gov/data/global-hourly/access"
HISTORY_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
YEARS = (2022, 2023, 2024)
STATIONS = {
    "kotar_range": "47122399999",
    "daegwallyeong": "47100099999",
    "gangneung": "47105099999",
}
KEEP_COLUMNS = (
    "STATION", "DATE", "SOURCE", "LATITUDE", "LONGITUDE", "ELEVATION",
    "NAME", "REPORT_TYPE", "QUALITY_CONTROL", "WND", "TMP", "SLP", "MA1",
)
ACCEPTED_QC = frozenset(("0", "1"))
USER_AGENT = "baram2026-causal-noaa-isd-feasibility/1.0"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url: str) -> tuple[bytes, dict[str, str]]:
    if "/2025/" in url or "/2026/" in url:
        raise AssertionError("physical observation cap: calendar 2025+ is forbidden")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = response.read()
        headers = {key.lower(): value for key, value in response.headers.items()}
    return payload, headers


def atomic_bytes(payload: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def split_field(values: pd.Series, count: int) -> list[pd.Series]:
    parts = values.fillna("").astype(str).str.split(",", expand=True)
    return [parts[index] if index in parts else pd.Series("", index=values.index) for index in range(count)]


def numeric_with_qc(value: pd.Series, qc: pd.Series, *, missing: int, scale: float) -> pd.Series:
    numeric = pd.to_numeric(value, errors="coerce")
    valid = qc.isin(ACCEPTED_QC) & numeric.notna() & numeric.ne(missing)
    return (numeric.where(valid) / scale).astype(np.float64)


def normalize_synop(raw: pd.DataFrame, station_id: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    if tuple(raw.columns.intersection(KEEP_COLUMNS)) != KEEP_COLUMNS:
        raise ValueError(f"{station_id}: source schema differs")
    if not raw["STATION"].astype(str).eq(station_id).all():
        raise ValueError(f"{station_id}: mixed station IDs")
    synop = raw.loc[raw["REPORT_TYPE"].eq("FM-12"), list(KEEP_COLUMNS)].copy()
    synop["time_utc"] = pd.to_datetime(synop["DATE"], utc=True, errors="raise")
    synop["time_kst"] = synop["time_utc"].dt.tz_convert("Asia/Seoul").dt.tz_localize(None)

    direction, direction_qc, wind_type, speed, speed_qc = split_field(synop["WND"], 5)
    synop["wind_speed_ms"] = numeric_with_qc(speed, speed_qc, missing=9999, scale=10.0)
    direction_value = numeric_with_qc(direction, direction_qc, missing=999, scale=1.0)
    calm = synop["wind_speed_ms"].eq(0.0)
    direction_value = direction_value.mask(calm & direction_value.isna(), 0.0)
    direction_value = direction_value.where(direction_value.between(0.0, 360.0))
    synop["wind_direction_deg"] = direction_value
    radians = np.deg2rad(direction_value)
    synop["wind_u_ms"] = -synop["wind_speed_ms"] * np.sin(radians)
    synop["wind_v_ms"] = -synop["wind_speed_ms"] * np.cos(radians)

    temperature, temperature_qc = split_field(synop["TMP"], 2)
    synop["temperature_c"] = numeric_with_qc(temperature, temperature_qc, missing=9999, scale=10.0)
    pressure, pressure_qc = split_field(synop["SLP"], 2)
    synop["sea_level_pressure_hpa"] = numeric_with_qc(pressure, pressure_qc, missing=99999, scale=10.0)
    station_altimeter, station_altimeter_qc, station_pressure, station_pressure_qc = split_field(synop["MA1"], 4)
    synop["station_pressure_hpa"] = numeric_with_qc(
        station_pressure, station_pressure_qc, missing=99999, scale=10.0
    )
    synop["wind_type"] = wind_type
    normalized_columns = [
        "time_utc", "time_kst", "SOURCE", "QUALITY_CONTROL", "wind_type",
        "wind_speed_ms", "wind_direction_deg", "wind_u_ms", "wind_v_ms",
        "temperature_c", "sea_level_pressure_hpa", "station_pressure_hpa",
    ]
    normalized = synop[normalized_columns].sort_values(["time_utc", "SOURCE"], kind="stable")
    duplicate_rows = int(normalized.duplicated("time_utc", keep=False).sum())
    conflicts: dict[str, int] = {}
    value_columns = [
        "wind_speed_ms", "wind_direction_deg", "temperature_c",
        "sea_level_pressure_hpa", "station_pressure_hpa",
    ]
    for column in value_columns:
        counts = normalized.groupby("time_utc", sort=False)[column].nunique(dropna=True)
        conflicts[column] = int(counts.gt(1).sum())
    # Stable first report is the preregistrable one-per-station/time rule.
    normalized = normalized.drop_duplicates("time_utc", keep="first").reset_index(drop=True)
    diagnostics = {
        "raw_rows": len(raw),
        "report_type_counts": raw["REPORT_TYPE"].value_counts(dropna=False).to_dict(),
        "fm12_rows_before_dedup": len(synop),
        "duplicate_rows": duplicate_rows,
        "conflicting_timestamps_by_variable": conflicts,
        "fm12_rows_after_dedup": len(normalized),
        "first_time_utc": str(normalized["time_utc"].min()),
        "last_time_utc": str(normalized["time_utc"].max()),
        "missing_after_strict_qc": {column: int(normalized[column].isna().sum()) for column in value_columns},
        "source_counts": normalized["SOURCE"].value_counts(dropna=False).to_dict(),
        "quality_control_counts": normalized["QUALITY_CONTROL"].value_counts(dropna=False).to_dict(),
        "wind_type_counts": normalized["wind_type"].value_counts(dropna=False).to_dict(),
    }
    return normalized, diagnostics


def cutoff_summary(frame: pd.DataFrame, station_name: str) -> pd.DataFrame:
    decision_dates = pd.date_range("2022-01-03", "2024-12-31", freq="D")
    rows: list[dict[str, Any]] = []
    variables = {
        "ws": "wind_speed_ms",
        "u": "wind_u_ms",
        "v": "wind_v_ms",
        "ta": "temperature_c",
        "slp": "sea_level_pressure_hpa",
        "pa": "station_pressure_hpa",
    }
    indexed = frame.set_index("time_kst").sort_index()
    for decision_date in decision_dates:
        cutoff = decision_date - pd.Timedelta(days=1) + pd.Timedelta(hours=14)
        start = cutoff - pd.Timedelta(hours=24)
        window = indexed.loc[(indexed.index >= start) & (indexed.index < cutoff)]
        record: dict[str, Any] = {
            "decision_date": decision_date,
            f"{station_name}__report_count_24h": len(window),
            f"{station_name}__unique_hour_count_24h": int(window.index.floor("h").nunique()),
            f"{station_name}__latest_age_h": (
                float((cutoff - window.index.max()) / pd.Timedelta(hours=1)) if len(window) else np.nan
            ),
        }
        for prefix, column in variables.items():
            values = window[column].dropna()
            record[f"{station_name}__{prefix}_n_24h"] = len(values)
            record[f"{station_name}__{prefix}_mean_24h"] = float(values.mean()) if len(values) else np.nan
            record[f"{station_name}__{prefix}_std_24h"] = float(values.std(ddof=0)) if len(values) else np.nan
            record[f"{station_name}__{prefix}_min_24h"] = float(values.min()) if len(values) else np.nan
            record[f"{station_name}__{prefix}_max_24h"] = float(values.max()) if len(values) else np.nan
            record[f"{station_name}__{prefix}_latest"] = float(values.iloc[-1]) if len(values) else np.nan
            record[f"{station_name}__{prefix}_delta"] = float(values.iloc[-1] - values.iloc[0]) if len(values) else np.nan
        rows.append(record)
    return pd.DataFrame(rows)


def main() -> None:
    out_dir = ROOT / "artifacts/external/noaa_isd_synop_lags_v1"
    manifest_path = out_dir / "source_manifest.json"
    normalized_path = out_dir / "noaa_isd_synop_2022_2024.parquet"
    summary_path = out_dir / "noaa_isd_cutoff_summaries_2022_2024.parquet"
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {out_dir}")
    out_dir.mkdir(parents=True)

    history_payload, history_headers = fetch(HISTORY_URL)
    history = pd.read_csv(pd.io.common.BytesIO(history_payload))
    requested_ids = set(STATIONS.values())
    station_rows = history.loc[
        history["USAF"].astype(str).str.zfill(6).add(history["WBAN"].astype(str).str.zfill(5)).isin(requested_ids)
    ].copy()
    if len(station_rows) != len(STATIONS):
        raise ValueError("station history does not uniquely resolve the three fixed stations")

    normalized_frames: list[pd.DataFrame] = []
    requests: list[dict[str, Any]] = []
    census: dict[str, Any] = {}
    for station_name, station_id in STATIONS.items():
        station_frames: list[pd.DataFrame] = []
        census[station_name] = {}
        for year in YEARS:
            url = f"{BASE_URL}/{year}/{station_id}.csv"
            payload, headers = fetch(url)
            raw_path = out_dir / f"raw/{year}/{station_id}.csv"
            atomic_bytes(payload, raw_path)
            raw = pd.read_csv(pd.io.common.BytesIO(payload), low_memory=False)
            normalized, diagnostics = normalize_synop(raw, station_id)
            normalized.insert(0, "year", year)
            normalized.insert(0, "station_id", station_id)
            normalized.insert(0, "station_name", station_name)
            station_frames.append(normalized)
            census[station_name][str(year)] = diagnostics
            requests.append({
                "station_name": station_name,
                "station_id": station_id,
                "year": year,
                "url": url,
                "response_bytes": len(payload),
                "response_sha256": sha256_bytes(payload),
                "last_modified": headers.get("last-modified"),
                "raw_path": str(raw_path),
                "raw_path_sha256": sha256(raw_path),
            })
            print(f"downloaded {station_name}/{year}: {len(raw)} raw rows, {len(normalized)} FM-12", flush=True)
        station_combined = pd.concat(station_frames, ignore_index=True)
        if station_combined.duplicated(["station_id", "time_utc"]).any():
            raise ValueError(f"{station_name}: cross-year duplicate timestamps")
        normalized_frames.append(station_combined)

    all_normalized = pd.concat(normalized_frames, ignore_index=True)
    all_normalized.to_parquet(normalized_path, engine="pyarrow", compression="zstd", index=False)
    summaries = [
        cutoff_summary(all_normalized.loc[all_normalized["station_name"].eq(name)], name)
        for name in STATIONS
    ]
    combined_summary = summaries[0]
    for other in summaries[1:]:
        combined_summary = combined_summary.merge(other, on="decision_date", how="inner", validate="one_to_one")
    combined_summary.to_parquet(summary_path, engine="pyarrow", compression="zstd", index=False)

    summary_missing = {
        column: int(combined_summary[column].isna().sum())
        for column in combined_summary.columns if column != "decision_date"
    }
    manifest = {
        "schema_version": 1,
        "status": "feasibility_only_no_label_join_candidate_fit_prediction_or_score",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "official_source": {
            "provider": "NOAA National Centers for Environmental Information",
            "product": "Global Hourly Integrated Surface Database",
            "product_page": "https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database",
            "format_documentation": "https://www.ncei.noaa.gov/pub/data/noaa/isd-format-document.pdf",
            "annual_csv_base": BASE_URL,
            "station_history_url": HISTORY_URL,
            "station_history_response_bytes": len(history_payload),
            "station_history_response_sha256": sha256_bytes(history_payload),
            "station_history_last_modified": history_headers.get("last-modified"),
            "station_history_subset": station_rows.to_dict(orient="records"),
        },
        "source_comparison": {
            "kma_endpoint_censused_not_downloaded": "https://data.kma.go.kr/data/grnd/selectAsosRltmList.do?pgmNo=36",
            "kma_station": "216 Taebaek",
            "kma_rejection": "Official portal states previous-day records are available after 10:00 the next day and exposes later QC/finalized historical values without an as-issued version; this cannot prove strict D-1 14:00 availability or revision safety.",
            "noaa_scope": "Only timestamped FM-12 fixed-land SYNOP reports and mandatory fields with QC code 0 or 1 are retained; daily summaries and non-SYNOP reports are excluded.",
            "remaining_limitation": "NCEI applies archive QC; timestamp and report type make causality clearer than KMA finalized daily/hourly portal data, but annual files are not a versioned as-issued archive."
        },
        "causal_contract": {
            "decision_cutoff": "D-1 14:00 Asia/Seoul, strict less-than",
            "window": "[D-2 14:00, D-1 14:00) Asia/Seoul",
            "maximum_observation_time": "D-1 13:59:59.999999 Asia/Seoul",
            "timestamp_source": "DATE is the report observation timestamp in UTC; convert to Asia/Seoul before the strict filter",
            "allowed_report_type": "FM-12 only",
            "allowed_quality_codes": ["0", "1"],
            "duplicate_rule": "stable first FM-12 record per station and UTC observation timestamp",
            "daily_or_later_summary_fields": "forbidden",
            "missing_policy_for_future_model": "fixed sentinel plus missing/count/recency indicators only; no future/backward fill",
        },
        "requests": requests,
        "census": census,
        "normalized": {"path": str(normalized_path), "bytes": normalized_path.stat().st_size, "sha256": sha256(normalized_path), "rows": len(all_normalized)},
        "cutoff_summaries": {
            "path": str(summary_path),
            "bytes": summary_path.stat().st_size,
            "sha256": sha256(summary_path),
            "rows": len(combined_summary),
            "start_decision_date": str(combined_summary["decision_date"].min()),
            "end_decision_date": str(combined_summary["decision_date"].max()),
            "missing_counts": summary_missing,
        },
        "physical_access": {
            "maximum_observation_calendar_year_requested_or_parsed": 2024,
            "2025_observation_files_requested_or_parsed": False,
            "label_value_cells_parsed": 0,
            "candidate_fits_predictions_scores": 0,
            "feedback_files_read": 0,
        },
    }
    atomic_json(manifest, manifest_path)
    print(f"normalized={normalized_path} sha256={sha256(normalized_path)}")
    print(f"summaries={summary_path} sha256={sha256(summary_path)}")
    print(f"manifest={manifest_path} sha256={sha256(manifest_path)}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INFO_PATH = Path(r"data/local/open/info.xlsx")
ENDPOINT = "https://previous-runs-api.open-meteo.com/v1/forecast"
MODEL = "jma_gsm"
YEARS = (2022, 2023, 2024)
GROUP_SLICES = {
    "kpx_group_1": slice(0, 6),
    "kpx_group_2": slice(6, 12),
    "kpx_group_3": slice(12, 17),
}
VARIABLES = (
    "wind_speed_10m_previous_day1",
    "wind_direction_10m_previous_day1",
    "wind_speed_10m_previous_day2",
    "wind_direction_10m_previous_day2",
)
USER_AGENT = "baram2026-causal-jma-feasibility/1.0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dms_coordinates() -> list[tuple[float, float]]:
    values = pd.read_excel(INFO_PATH, header=None).iloc[:, 6].dropna().astype(str).tolist()[1:]
    result: list[tuple[float, float]] = []
    for value in values:
        parts = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", value)]
        if len(parts) != 6:
            raise ValueError(f"unexpected coordinate in info.xlsx: {value!r}")
        latitude = parts[0] + parts[1] / 60.0 + parts[2] / 3600.0
        longitude = parts[3] + parts[4] / 60.0 + parts[5] / 3600.0
        result.append((latitude, longitude))
    if len(result) != 17:
        raise ValueError("expected 17 turbine coordinates")
    return result


def group_centroids() -> dict[str, tuple[float, float]]:
    coordinates = dms_coordinates()
    result: dict[str, tuple[float, float]] = {}
    for group, subset in GROUP_SLICES.items():
        values = coordinates[subset]
        result[group] = (
            sum(latitude for latitude, _ in values) / len(values),
            sum(longitude for _, longitude in values) / len(values),
        )
    return result


def query_url(latitude: float, longitude: float, year: int) -> str:
    params = {
        "latitude": f"{latitude:.8f}",
        "longitude": f"{longitude:.8f}",
        "start_date": date(year, 1, 1).isoformat(),
        "end_date": date(year, 12, 31).isoformat(),
        "timezone": "Asia/Seoul",
        "wind_speed_unit": "ms",
        "models": MODEL,
        "cell_selection": "nearest",
        "hourly": ",".join(VARIABLES),
    }
    return ENDPOINT + "?" + urllib.parse.urlencode(params)


def fetch(url: str) -> tuple[dict[str, Any], str, int]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = response.read()
    return json.loads(payload), hashlib.sha256(payload).hexdigest(), len(payload)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    frame.to_parquet(temporary, engine="pyarrow", compression="zstd", index=False)
    os.replace(temporary, path)


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    out_dir = ROOT / "artifacts/external/openmeteo_jma_gsm_previous_runs_v1"
    data_path = out_dir / "jma_gsm_group_centroids_2022_2024.parquet"
    manifest_path = out_dir / "source_manifest.json"
    if data_path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite: {out_dir}")
    if max(YEARS) > 2024:
        raise AssertionError("physical pre-promotion cap must exclude 2025")

    centroids = group_centroids()
    frames: list[pd.DataFrame] = []
    requests: list[dict[str, Any]] = []
    for group, (latitude, longitude) in centroids.items():
        for year in YEARS:
            url = query_url(latitude, longitude, year)
            response, response_sha, response_bytes = fetch(url)
            hourly = response["hourly"]
            expected = 8784 if year == 2024 else 8760
            if len(hourly["time"]) != expected:
                raise ValueError(f"{group}/{year}: unexpected hourly length")
            frame = pd.DataFrame(hourly)
            if tuple(frame.columns) != ("time", *VARIABLES):
                raise ValueError(f"{group}/{year}: response variables differ")
            if frame.loc[:, list(VARIABLES)].isna().any().any():
                raise ValueError(f"{group}/{year}: JMA GSM previous-run values are incomplete")
            frame.insert(0, "group", group)
            frame["time"] = pd.to_datetime(frame["time"], format="%Y-%m-%dT%H:%M")
            if not frame["time"].equals(
                pd.Series(pd.date_range(f"{year}-01-01 00:00", f"{year}-12-31 23:00", freq="h"))
            ):
                raise ValueError(f"{group}/{year}: local hourly index differs")
            frames.append(frame)
            requests.append(
                {
                    "group": group,
                    "year": year,
                    "url": url,
                    "response_sha256": response_sha,
                    "response_bytes": response_bytes,
                    "returned_grid": {
                        "latitude": response["latitude"],
                        "longitude": response["longitude"],
                        "elevation": response["elevation"],
                    },
                    "hourly_units": response["hourly_units"],
                    "rows": len(frame),
                    "missing_value_cells": int(frame.loc[:, list(VARIABLES)].isna().sum().sum()),
                }
            )
            print(f"downloaded {group}/{year}: {len(frame)} rows", flush=True)

    combined = pd.concat(frames, ignore_index=True)
    if combined.duplicated(["group", "time"]).any() or len(combined) != 3 * (8760 + 8760 + 8784):
        raise ValueError("combined JMA key coverage differs")
    atomic_parquet(combined, data_path)
    manifest = {
        "schema_version": 1,
        "status": "availability_feasibility_only_no_label_or_candidate_score",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "Open-Meteo Previous Model Runs API",
        "upstream_model": "Japan Meteorological Agency GSM",
        "endpoint": ENDPOINT,
        "model_parameter": MODEL,
        "timezone": "Asia/Seoul",
        "wind_speed_unit": "m/s",
        "cell_selection": "nearest",
        "variables": list(VARIABLES),
        "cutoff_note": {
            "previous_day1": "24 hours before each valid timestamp; not safe for the entire operating day",
            "candidate_A": "use previous_day2 at every target hour",
            "candidate_B": "use previous_day1 only for operating-day hours 01..13 and previous_day2 for hours 14..24",
            "day0_forbidden": True,
            "historical_forecast_stitched_day0_forbidden": True,
        },
        "unsupported_or_rejected": {
            "wind_80m_and_100m": "JMA GSM sample returned units=undefined and all null",
            "jma_msm_2022": "sample returned all null, so it cannot support the 2022-to-2023 fold",
        },
        "centroids_from_info_xlsx": {
            group: {"latitude": latitude, "longitude": longitude}
            for group, (latitude, longitude) in centroids.items()
        },
        "info_xlsx": {"path": str(INFO_PATH), "bytes": INFO_PATH.stat().st_size, "sha256": sha256(INFO_PATH)},
        "requests": requests,
        "normalized": {
            "path": str(data_path),
            "bytes": data_path.stat().st_size,
            "sha256": sha256(data_path),
            "rows": len(combined),
            "columns": list(combined.columns),
            "missing_value_cells": int(combined.loc[:, list(VARIABLES)].isna().sum().sum()),
        },
        "physical_access": {
            "2025_observations_or_forecasts_requested_or_parsed": False,
            "annual_kpx_generation_used": False,
            "label_value_cells_parsed": 0,
            "candidate_score_computed": False,
        },
    }
    atomic_json(manifest, manifest_path)
    print(f"data={data_path} sha256={sha256(data_path)}")
    print(f"manifest={manifest_path} sha256={sha256(manifest_path)}")


if __name__ == "__main__":
    main()

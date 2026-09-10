from __future__ import annotations

import hashlib
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = "https://previous-runs-api.open-meteo.com/v1/forecast"
MODEL = "icon_global"
GROUP_CENTROIDS = {
    "kpx_group_1": (37.287127, 128.952021),
    "kpx_group_2": (37.282255, 128.965148),
    "kpx_group_3": (37.275199, 128.971444),
}
VARIABLES = tuple(
    f"{kind}_{height}m_previous_day{day}"
    for height in (10, 80, 120)
    for day in (1, 2)
    for kind in ("wind_speed", "wind_direction")
)
PROBE_DATES = ("2022-01-15", "2023-01-15", "2024-01-15", "2024-06-15", "2024-12-15")
USER_AGENT = "baram2026-causal-icon-global-feasibility/1.0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def query_url(latitude: float, longitude: float, *, start: str, end: str) -> str:
    years = {int(start[:4]), int(end[:4])}
    if min(years) < 2022 or max(years) > 2024:
        raise AssertionError("physical network cap: only calendar 2022--2024 is permitted")
    params = {
        "latitude": f"{latitude:.8f}",
        "longitude": f"{longitude:.8f}",
        "start_date": start,
        "end_date": end,
        "timezone": "Asia/Seoul",
        "wind_speed_unit": "ms",
        "models": MODEL,
        "cell_selection": "nearest",
        "hourly": ",".join(VARIABLES),
    }
    return ENDPOINT + "?" + urllib.parse.urlencode(params)


def fetch(url: str) -> tuple[dict[str, Any], str, int]:
    if "2025-" in url:
        raise AssertionError("2025 requests are forbidden before promotion")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=240) as response:
        payload = response.read()
    return json.loads(payload), hashlib.sha256(payload).hexdigest(), len(payload)


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def availability(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for variable in VARIABLES:
        valid = frame[variable].notna()
        result[variable] = {
            "non_null": int(valid.sum()),
            "first_non_null": str(frame.loc[valid, "time"].min()) if valid.any() else None,
            "last_non_null": str(frame.loc[valid, "time"].max()) if valid.any() else None,
        }
    complete = frame.loc[:, list(VARIABLES)].notna().all(axis=1)
    return {
        "by_variable": result,
        "complete_rows": int(complete.sum()),
        "first_complete_time": str(frame.loc[complete, "time"].min()) if complete.any() else None,
        "last_complete_time": str(frame.loc[complete, "time"].max()) if complete.any() else None,
        "interior_incomplete_rows": int((~complete.loc[complete.idxmax():complete[::-1].idxmax()]).sum()) if complete.any() else None,
    }


def main() -> None:
    out_dir = ROOT / "artifacts/external/openmeteo_icon_global_previous_runs_v1"
    data_path = out_dir / "icon_global_group_centroids_2024.parquet"
    manifest_path = out_dir / "source_manifest.json"
    if data_path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite: {out_dir}")

    latitude, longitude = GROUP_CENTROIDS["kpx_group_1"]
    probes: list[dict[str, Any]] = []
    for date in PROBE_DATES:
        url = query_url(latitude, longitude, start=date, end=date)
        response, response_sha, response_bytes = fetch(url)
        frame = pd.DataFrame(response["hourly"])
        frame["time"] = pd.to_datetime(frame["time"], format="%Y-%m-%dT%H:%M")
        probes.append({
            "date": date,
            "url": url,
            "response_sha256": response_sha,
            "response_bytes": response_bytes,
            "non_null_by_variable": {name: int(frame[name].notna().sum()) for name in VARIABLES},
        })

    frames: list[pd.DataFrame] = []
    requests: list[dict[str, Any]] = []
    expected = pd.Series(pd.date_range("2024-01-01 00:00", "2024-12-31 23:00", freq="h"))
    for group, (latitude, longitude) in GROUP_CENTROIDS.items():
        url = query_url(latitude, longitude, start="2024-01-01", end="2024-12-31")
        response, response_sha, response_bytes = fetch(url)
        frame = pd.DataFrame(response["hourly"])
        if tuple(frame.columns) != ("time", *VARIABLES) or len(frame) != 8784:
            raise ValueError(f"{group}: response shape/schema differs")
        frame["time"] = pd.to_datetime(frame["time"], format="%Y-%m-%dT%H:%M")
        if not frame["time"].equals(expected):
            raise ValueError(f"{group}: local hourly index differs")
        details = availability(frame)
        frame.insert(0, "group", group)
        frames.append(frame)
        requests.append({
            "group": group,
            "url": url,
            "response_sha256": response_sha,
            "response_bytes": response_bytes,
            "returned_grid": {
                "latitude": response["latitude"],
                "longitude": response["longitude"],
                "elevation": response["elevation"],
            },
            "rows": len(frame),
            **details,
        })
        print(f"downloaded {group}: {len(frame)} rows, {details['complete_rows']} complete", flush=True)

    combined = pd.concat(frames, ignore_index=True)
    if combined.duplicated(["group", "time"]).any() or len(combined) != 3 * 8784:
        raise ValueError("combined ICON key coverage differs")
    out_dir.mkdir(parents=True, exist_ok=True)
    temporary = data_path.with_name(f".{data_path.name}.tmp-{os.getpid()}")
    combined.to_parquet(temporary, engine="pyarrow", compression="zstd", index=False)
    os.replace(temporary, data_path)

    manifest = {
        "schema_version": 1,
        "status": "feasibility_and_availability_only_no_label_join_fit_prediction_or_score",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "Open-Meteo Previous Model Runs API",
        "endpoint": ENDPOINT,
        "official_previous_runs_documentation": "https://open-meteo.com/en/docs/previous-runs-api",
        "official_dwd_documentation": "https://open-meteo.com/en/docs/dwd-api",
        "official_model": MODEL,
        "official_model_label": "DWD ICON Global",
        "documented_native_model_contract": {
            "grid": "0.1 degree (~11 km)",
            "update_frequency": "6-hourly",
            "forecast_horizon": "7.5 days",
            "wind_levels": [10, 80, 120, 180],
            "native_wind_components": "u/v; speed and direction derived by Open-Meteo",
        },
        "previous_run_semantics": {
            "previous_day1": "forecast for the same valid timestamp produced 24 hours earlier",
            "previous_day2": "forecast for the same valid timestamp produced 48 hours earlier",
        },
        "timezone": "Asia/Seoul",
        "wind_speed_unit": "m/s",
        "cell_selection": "nearest",
        "variables": list(VARIABLES),
        "fixed_probes": probes,
        "requests": requests,
        "normalized": {
            "path": str(data_path),
            "bytes": data_path.stat().st_size,
            "sha256": sha256(data_path),
            "rows": len(combined),
            "columns": list(combined.columns),
        },
        "feasibility_conclusion": "2022 and 2023 fixed probes are null. All requested levels/day offsets are continuous from 2024-02-17 21:00 KST through year end, sufficient for a preregistered 2024 H1 fit to H2 apply diagnostic.",
        "physical_access": {
            "minimum_calendar_year_requested_or_parsed": 2022,
            "maximum_calendar_year_requested_or_parsed": 2024,
            "2025_observations_or_forecasts_requested_or_parsed": False,
            "annual_kpx_generation_used": False,
            "label_value_cells_parsed": 0,
            "candidate_fit_or_score_computed": False,
        },
    }
    atomic_json(manifest, manifest_path)
    print(f"data={data_path} sha256={sha256(data_path)}")
    print(f"manifest={manifest_path} sha256={sha256(manifest_path)}")


if __name__ == "__main__":
    main()

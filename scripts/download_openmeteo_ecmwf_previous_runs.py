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
MODEL = "ecmwf_ifs025"
HRES_MODEL = "ecmwf_ifs"
YEAR = 2024
GROUP_CENTROIDS = {
    "kpx_group_1": (37.287127, 128.952021),
    "kpx_group_2": (37.282255, 128.965148),
    "kpx_group_3": (37.275199, 128.971444),
}
VARIABLES = (
    "wind_speed_100m_previous_day1",
    "wind_direction_100m_previous_day1",
    "wind_speed_100m_previous_day2",
    "wind_direction_100m_previous_day2",
)
USER_AGENT = "baram2026-causal-ecmwf-feasibility/1.0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def query_url(latitude: float, longitude: float, *, model: str, start: str, end: str) -> str:
    if not start.startswith("2024-") or not end.startswith("2024-"):
        raise AssertionError("physical network cap: only calendar 2024 is permitted")
    params = {
        "latitude": f"{latitude:.8f}",
        "longitude": f"{longitude:.8f}",
        "start_date": start,
        "end_date": end,
        "timezone": "Asia/Seoul",
        "wind_speed_unit": "ms",
        "models": model,
        "cell_selection": "nearest",
        "hourly": ",".join(VARIABLES),
    }
    return ENDPOINT + "?" + urllib.parse.urlencode(params)


def fetch(url: str) -> tuple[dict[str, Any], str, int]:
    if "2025-" in url:
        raise AssertionError("2025 requests are forbidden before promotion")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = response.read()
    return json.loads(payload), hashlib.sha256(payload).hexdigest(), len(payload)


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    out_dir = ROOT / "artifacts/external/openmeteo_ecmwf_ifs025_previous_runs_v1"
    data_path = out_dir / "ecmwf_ifs025_group_centroids_2024.parquet"
    manifest_path = out_dir / "source_manifest.json"
    if data_path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite: {out_dir}")

    frames: list[pd.DataFrame] = []
    requests: list[dict[str, Any]] = []
    for group, (latitude, longitude) in GROUP_CENTROIDS.items():
        url = query_url(latitude, longitude, model=MODEL, start="2024-01-01", end="2024-12-31")
        response, response_sha, response_bytes = fetch(url)
        frame = pd.DataFrame(response["hourly"])
        if tuple(frame.columns) != ("time", *VARIABLES) or len(frame) != 8784:
            raise ValueError(f"{group}: response shape/schema differs")
        frame.insert(0, "group", group)
        frame["time"] = pd.to_datetime(frame["time"], format="%Y-%m-%dT%H:%M")
        expected = pd.Series(pd.date_range("2024-01-01 00:00", "2024-12-31 23:00", freq="h"))
        if not frame["time"].equals(expected):
            raise ValueError(f"{group}: local hourly index differs")
        frames.append(frame)
        valid = frame.loc[:, list(VARIABLES)].notna().all(axis=1)
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
            "hourly_units": response["hourly_units"],
            "rows": len(frame),
            "complete_rows": int(valid.sum()),
            "first_complete_time": str(frame.loc[valid, "time"].min()),
            "last_complete_time": str(frame.loc[valid, "time"].max()),
        })
        print(f"downloaded {group}: {len(frame)} rows, {int(valid.sum())} complete", flush=True)

    combined = pd.concat(frames, ignore_index=True)
    if combined.duplicated(["group", "time"]).any() or len(combined) != 3 * 8784:
        raise ValueError("combined ECMWF key coverage differs")
    out_dir.mkdir(parents=True, exist_ok=True)
    temporary = data_path.with_name(f".{data_path.name}.tmp-{os.getpid()}")
    combined.to_parquet(temporary, engine="pyarrow", compression="zstd", index=False)
    os.replace(temporary, data_path)

    # HRES is documented but its 2024 Previous Runs availability is tested, not assumed.
    hres_probes: list[dict[str, Any]] = []
    latitude, longitude = GROUP_CENTROIDS["kpx_group_1"]
    for start, end in (("2024-03-20", "2024-03-22"), ("2024-06-01", "2024-06-03"), ("2024-12-01", "2024-12-03")):
        url = query_url(latitude, longitude, model=HRES_MODEL, start=start, end=end)
        response, response_sha, response_bytes = fetch(url)
        probe = pd.DataFrame(response["hourly"])
        hres_probes.append({
            "start": start,
            "end": end,
            "url": url,
            "response_sha256": response_sha,
            "response_bytes": response_bytes,
            "non_null_by_variable": {name: int(probe[name].notna().sum()) for name in VARIABLES},
        })

    complete = combined.loc[:, list(VARIABLES)].notna().all(axis=1)
    manifest = {
        "schema_version": 1,
        "status": "feasibility_and_availability_only_no_label_join_fit_prediction_or_score",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "Open-Meteo Previous Model Runs API",
        "endpoint": ENDPOINT,
        "official_previous_runs_documentation": "https://open-meteo.com/en/docs/previous-runs-api",
        "official_ecmwf_documentation": "https://open-meteo.com/en/docs/ecmwf-api",
        "selected_model": MODEL,
        "selected_model_label": "ECMWF IFS 0.25 degree",
        "hres_model": HRES_MODEL,
        "hres_2024_probes": hres_probes,
        "hres_conclusion": "No 100 m previous_day1/day2 values in sampled 2024 periods; excluded before preregistration.",
        "timezone": "Asia/Seoul",
        "wind_speed_unit": "m/s",
        "cell_selection": "nearest",
        "variables": list(VARIABLES),
        "requests": requests,
        "normalized": {
            "path": str(data_path),
            "bytes": data_path.stat().st_size,
            "sha256": sha256(data_path),
            "rows": len(combined),
            "complete_rows": int(complete.sum()),
            "first_complete_time": str(combined.loc[complete, "time"].min()),
            "last_complete_time": str(combined.loc[complete, "time"].max()),
            "columns": list(combined.columns),
        },
        "physical_access": {
            "maximum_calendar_year_requested_or_parsed": YEAR,
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

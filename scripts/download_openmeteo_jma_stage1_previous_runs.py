"""Download a physically isolated 2022-2023 JMA GSM Stage1 source."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.download_openmeteo_jma_previous_runs import (
    INFO_PATH,
    MODEL,
    VARIABLES,
    group_centroids,
)

ENDPOINT = "https://previous-runs-api.open-meteo.com/v1/forecast"
START_DATE = "2022-01-01"
END_DATE = "2023-12-31"
USER_AGENT = "baram2026-causal-jma-stage1-physical-prefix/1.0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def query_url(latitude: float, longitude: float) -> str:
    return ENDPOINT + "?" + urllib.parse.urlencode(
        {
            "latitude": f"{latitude:.8f}",
            "longitude": f"{longitude:.8f}",
            "start_date": START_DATE,
            "end_date": END_DATE,
            "timezone": "Asia/Seoul",
            "wind_speed_unit": "ms",
            "models": MODEL,
            "cell_selection": "nearest",
            "hourly": ",".join(VARIABLES),
        }
    )


def atomic_bytes(payload: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        frame.to_parquet(temporary, engine="pyarrow", compression="zstd", index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(payload: Any, path: Path) -> None:
    data = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    atomic_bytes(data, path)


def main() -> None:
    out_dir = ROOT / "artifacts/external/openmeteo_jma_gsm_stage1_2022_2023_v1"
    if out_dir.exists():
        raise FileExistsError(out_dir)
    expected_index = pd.date_range("2022-01-01 00:00", "2023-12-31 23:00", freq="h")
    centroids = group_centroids()
    frames: list[pd.DataFrame] = []
    requests: list[dict[str, Any]] = []
    for group, (latitude, longitude) in centroids.items():
        url = query_url(latitude, longitude)
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=180) as response:
            raw = response.read()
        response_path = out_dir / "responses" / f"{group}.json"
        atomic_bytes(raw, response_path)
        payload = json.loads(raw)
        hourly = payload["hourly"]
        frame = pd.DataFrame(hourly)
        if tuple(frame.columns) != ("time", *VARIABLES):
            raise AssertionError(f"{group}: response variables differ")
        frame["time"] = pd.to_datetime(frame["time"], format="%Y-%m-%dT%H:%M")
        if len(frame) != 17520 or not pd.DatetimeIndex(frame["time"]).equals(expected_index):
            raise AssertionError(f"{group}: Stage1 time coverage differs")
        if frame.loc[:, list(VARIABLES)].isna().any().any():
            raise AssertionError(f"{group}: missing JMA values")
        frame.insert(0, "group", group)
        frames.append(frame)
        requests.append(
            {
                "group": group,
                "url": url,
                "response": {
                    "path": str(response_path.relative_to(ROOT).as_posix()),
                    "bytes": response_path.stat().st_size,
                    "sha256": sha256(response_path),
                },
                "returned_grid": {
                    "latitude": payload["latitude"],
                    "longitude": payload["longitude"],
                    "elevation": payload["elevation"],
                },
                "hourly_units": payload["hourly_units"],
                "rows": len(frame),
                "start": frame["time"].min().isoformat(),
                "end": frame["time"].max().isoformat(),
                "missing_value_cells": 0,
            }
        )
        print(f"downloaded {group}: {len(frame)} rows sha256={sha256(response_path)}", flush=True)

    combined = pd.concat(frames, ignore_index=True)
    if len(combined) != 52560 or combined.duplicated(["group", "time"]).any():
        raise AssertionError("combined Stage1 JMA coverage differs")
    data_path = out_dir / "jma_gsm_group_centroids_2022_2023.parquet"
    atomic_parquet(combined, data_path)
    manifest_path = out_dir / "source_manifest.json"
    manifest = {
        "schema_version": 1,
        "status": "physical_Stage1_source_no_label_or_candidate_score",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "Open-Meteo Previous Model Runs API",
        "official_documentation": "https://open-meteo.com/en/docs/previous-runs-api",
        "endpoint": ENDPOINT,
        "upstream_model": "Japan Meteorological Agency GSM",
        "model_parameter": MODEL,
        "request_start_date": START_DATE,
        "request_end_date": END_DATE,
        "timezone": "Asia/Seoul",
        "wind_speed_unit": "m/s",
        "cell_selection": "nearest",
        "variables": list(VARIABLES),
        "centroids_from_info_xlsx": {
            group: {"latitude": latitude, "longitude": longitude}
            for group, (latitude, longitude) in centroids.items()
        },
        "info_xlsx": {"path": str(INFO_PATH), "bytes": INFO_PATH.stat().st_size, "sha256": sha256(INFO_PATH)},
        "requests": requests,
        "normalized": {
            "path": str(data_path.relative_to(ROOT).as_posix()),
            "bytes": data_path.stat().st_size,
            "sha256": sha256(data_path),
            "rows": len(combined),
            "columns": list(combined.columns),
            "missing_value_cells": 0,
            "start": combined["time"].min().isoformat(),
            "end": combined["time"].max().isoformat(),
        },
        "boundary_patch": {
            "timestamp": "2024-01-01 00:00:00",
            "JMA_values_requested_or_materialized": 0,
            "paired_increment_cf": 0.0,
            "candidate_equals_baseline_float64_bits": True,
        },
        "physical_access": {
            "2024_JMA_value_rows_requested_or_materialized": 0,
            "2025_observations_or_forecasts_requested_or_parsed": False,
            "label_value_cells_parsed": 0,
            "candidate_prediction_or_score_values_computed": 0,
        },
    }
    atomic_json(manifest, manifest_path)
    print(f"data={data_path} sha256={sha256(data_path)}", flush=True)
    print(f"manifest={manifest_path} sha256={sha256(manifest_path)}", flush=True)


if __name__ == "__main__":
    main()

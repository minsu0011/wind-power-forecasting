"""Download the physically bounded 2024 GFS fixed-lead revision source."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import urllib.parse
import urllib.request
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = "https://previous-runs-api.open-meteo.com/v1/forecast"
MODEL = "gfs_global"
OUTPUT_DIR = ROOT / "artifacts/external/openmeteo_gfs_global_previous_runs_v1"
NORMALIZED_PATH = OUTPUT_DIR / "gfs_global_group_centroids_2024.parquet"
MANIFEST_PATH = OUTPUT_DIR / "source_manifest.json"
GROUP_CENTROIDS = {
    "kpx_group_1": (37.287127, 128.952021),
    "kpx_group_2": (37.282255, 128.965148),
    "kpx_group_3": (37.275199, 128.971444),
}
HEIGHTS = (10, 80, 100)
VARIABLES = tuple(
    f"{kind}_{height}m_previous_day{day}"
    for height in HEIGHTS
    for day in (1, 2)
    for kind in ("wind_speed", "wind_direction")
)
USER_AGENT = "baram2026-gfs-fixed-lead-revision-physical-2024/1.0"
EXPECTED_INDEX = pd.date_range("2024-01-01 00:00", "2024-12-31 23:00", freq="h")
FIT_INDEX = pd.date_range("2024-02-18 00:00", "2024-06-30 23:00", freq="h")
APPLY_INDEX = pd.date_range("2024-07-01 00:00", "2024-12-31 23:00", freq="h")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def query_url(latitude: float, longitude: float) -> str:
    params = {
        "latitude": f"{latitude:.8f}",
        "longitude": f"{longitude:.8f}",
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "timezone": "Asia/Seoul",
        "wind_speed_unit": "ms",
        "models": MODEL,
        "cell_selection": "nearest",
        "hourly": ",".join(VARIABLES),
    }
    url = ENDPOINT + "?" + urllib.parse.urlencode(params)
    if "2025-" in url or "2026-" in url:
        raise AssertionError("physical network cap permits calendar 2024 only")
    return url


def fetch(url: str) -> bytes:
    if "start_date=2024-01-01" not in url or "end_date=2024-12-31" not in url:
        raise AssertionError("request is not the exact registered 2024 interval")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read()


def write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_response(payload: bytes, group: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    response = json.loads(payload)
    hourly = response.get("hourly", {})
    if tuple(key for key in hourly if key != "time") != VARIABLES:
        raise AssertionError(f"{group}: response variable order differs")
    frame = pd.DataFrame({"time": hourly["time"], **{name: hourly[name] for name in VARIABLES}})
    frame["time"] = pd.to_datetime(frame["time"], errors="raise")
    if len(frame) != len(EXPECTED_INDEX) or not pd.DatetimeIndex(frame["time"]).equals(EXPECTED_INDEX):
        raise AssertionError(f"{group}: 2024 hourly calendar differs")
    for name in VARIABLES:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    if not frame.loc[frame["time"].isin(FIT_INDEX), list(VARIABLES)].notna().all().all():
        raise AssertionError(f"{group}: registered H1 fit window is incomplete")
    if not frame.loc[frame["time"].isin(APPLY_INDEX), list(VARIABLES)].notna().all().all():
        raise AssertionError(f"{group}: registered H2 application window is incomplete")
    evidence: dict[str, Any] = {
        "returned_latitude": response.get("latitude"),
        "returned_longitude": response.get("longitude"),
        "returned_elevation": response.get("elevation"),
        "timezone": response.get("timezone"),
        "rows": len(frame),
        "start": frame["time"].min().isoformat(),
        "end": frame["time"].max().isoformat(),
        "variables": {},
    }
    for name in VARIABLES:
        valid = frame[name].notna()
        times = frame.loc[valid, "time"]
        evidence["variables"][name] = {
            "non_null": int(valid.sum()),
            "first": times.min().isoformat() if len(times) else None,
            "last": times.max().isoformat() if len(times) else None,
            "fit_non_null": int(frame.loc[frame["time"].isin(FIT_INDEX), name].notna().sum()),
            "apply_non_null": int(frame.loc[frame["time"].isin(APPLY_INDEX), name].notna().sum()),
        }
    frame.insert(0, "group", group)
    return frame, evidence


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)
    frames: list[pd.DataFrame] = []
    requests: list[dict[str, Any]] = []
    for group, (latitude, longitude) in GROUP_CENTROIDS.items():
        url = query_url(latitude, longitude)
        raw = fetch(url)
        raw_path = OUTPUT_DIR / f"raw_{group}_2024.json"
        write_exclusive(raw_path, raw)
        frame, evidence = parse_response(raw, group)
        frames.append(frame)
        requests.append(
            {
                "group": group,
                "requested_coordinate": {"latitude": latitude, "longitude": longitude},
                "url": url,
                "response": file_record(raw_path),
                "response_sha256_recomputed": sha256_bytes(raw),
                "coverage": evidence,
            }
        )
    normalized = pd.concat(frames, ignore_index=True)
    expected_columns = ("group", "time", *VARIABLES)
    if tuple(normalized.columns) != expected_columns or len(normalized) != 3 * len(EXPECTED_INDEX):
        raise AssertionError("normalized source schema/rows differ")
    if normalized["time"].max() >= pd.Timestamp("2025-01-01"):
        raise AssertionError("normalized source crossed the physical 2024 boundary")
    normalized.to_parquet(NORMALIZED_PATH, engine="pyarrow", compression="zstd", index=False)
    manifest = {
        "schema_version": 1,
        "artifact_type": "openmeteo_gfs_global_fixed_lead_revision_2024_source",
        "provider": "Open-Meteo Previous Model Runs API",
        "endpoint": ENDPOINT,
        "official_documentation": "https://open-meteo.com/en/docs/previous-runs-api",
        "model": MODEL,
        "timezone": "Asia/Seoul",
        "wind_speed_unit": "m/s",
        "cell_selection": "nearest",
        "calendar_years_requested_or_parsed": [2024],
        "year_2025_requests_response_bytes_or_values": 0,
        "label_values_or_candidate_scores_accessed": 0,
        "variables": list(VARIABLES),
        "fit_window": [FIT_INDEX.min().isoformat(), FIT_INDEX.max().isoformat()],
        "apply_window": [APPLY_INDEX.min().isoformat(), APPLY_INDEX.max().isoformat()],
        "requests": requests,
        "normalized": file_record(NORMALIZED_PATH),
        "downloader": file_record(Path(__file__).resolve()),
    }
    atomic_json(MANIFEST_PATH, manifest)
    print(f"normalized={file_record(NORMALIZED_PATH)}", flush=True)
    print(f"manifest={file_record(MANIFEST_PATH)}", flush=True)


if __name__ == "__main__":
    main()

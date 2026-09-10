"""Download promoted, cutoff-safe 2025/terminal NWP previous-run features."""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = "https://previous-runs-api.open-meteo.com/v1/forecast"
VALIDATION_PATH = (
    ROOT / "artifacts/postgate/multi_nwp_consensus_g23_rescue_v1/validation_results.json"
)
OUTPUT_DIR = ROOT / "artifacts/external/openmeteo_multi_nwp_g23_rescue_2025_v1"
GROUP_CENTROIDS = {
    "kpx_group_1": (37.287127, 128.952021),
    "kpx_group_2": (37.282255, 128.965148),
    "kpx_group_3": (37.275199, 128.971444),
}
SOURCE_SPECS = {
    "ecmwf_ifs025": {
        "model": "ecmwf_ifs025",
        "heights": (100,),
        "file": "ecmwf_ifs025_group_centroids_2025.parquet",
    },
    "icon_global": {
        "model": "icon_global",
        "heights": (10, 80, 120),
        "file": "icon_global_group_centroids_2025.parquet",
    },
    "gfs_global": {
        "model": "gfs_global",
        "heights": (10, 80, 100),
        "file": "gfs_global_group_centroids_2025.parquet",
    },
}
REQUEST_INDEX = pd.date_range("2025-01-01 00:00", "2026-01-01 23:00", freq="h")
TARGET_INDEX = pd.date_range(
    "2025-01-01 01:00", "2026-01-01 00:00", freq="h", name="forecast_kst_dtm"
)
USER_AGENT = "baram2026-promoted-g23-consensus-final/1.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_json(payload: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def variables(heights: tuple[int, ...]) -> tuple[str, ...]:
    return tuple(
        f"{kind}_{height}m_previous_day{day}"
        for height in heights
        for day in (1, 2)
        for kind in ("wind_speed", "wind_direction")
    )


def query_url(latitude: float, longitude: float, model: str, names: tuple[str, ...]) -> str:
    params = {
        "latitude": f"{latitude:.8f}",
        "longitude": f"{longitude:.8f}",
        "start_date": "2025-01-01",
        "end_date": "2026-01-01",
        "timezone": "Asia/Seoul",
        "wind_speed_unit": "ms",
        "models": model,
        "cell_selection": "nearest",
        "hourly": ",".join(names),
    }
    return ENDPOINT + "?" + urllib.parse.urlencode(params)


def fetch(url: str) -> bytes:
    if "start_date=2025-01-01" not in url or "end_date=2026-01-01" not in url:
        raise AssertionError("request interval differs from promoted final target coverage")
    error: Exception | None = None
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=300) as response:
                return response.read()
        except Exception as exc:  # network retry is deterministic with respect to response content
            error = exc
            if attempt < 3:
                time.sleep(2**attempt)
    raise RuntimeError(f"Open-Meteo request failed after retries: {error}")


def promoted_validation() -> dict[str, Any]:
    payload = json.loads(VALIDATION_PATH.read_text(encoding="utf-8"))
    gate = payload["promotion_gate"]
    if not gate["promoted"] or not gate["final_stage_authorized"]:
        raise RuntimeError("2024-only validation did not authorize 2025 source access")
    if not (
        gate["mixed_H2_delta"]["total_score"] > 0.0
        and gate["mixed_H2_delta"]["ficr"] > 0.0
    ):
        raise RuntimeError("recorded H2 gate components are not positive")
    return payload


def main() -> None:
    validation = promoted_validation()
    manifest_path = OUTPUT_DIR / "source_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite completed source directory: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source_records: dict[str, Any] = {}
    for source_id, spec in SOURCE_SPECS.items():
        names = variables(tuple(spec["heights"]))
        frames: list[pd.DataFrame] = []
        requests: list[dict[str, Any]] = []
        for group, (latitude, longitude) in GROUP_CENTROIDS.items():
            url = query_url(latitude, longitude, str(spec["model"]), names)
            raw_path = OUTPUT_DIR / f"raw_{source_id}_{group}_2025_terminal.json"
            if raw_path.exists():
                raw = raw_path.read_bytes()
            else:
                raw = fetch(url)
                raw_path.write_bytes(raw)
            response = json.loads(raw)
            frame = pd.DataFrame(response["hourly"])
            if tuple(frame.columns) != ("time", *names):
                raise ValueError(f"{source_id}/{group}: response schema differs")
            frame["time"] = pd.to_datetime(frame["time"], format="%Y-%m-%dT%H:%M")
            if not pd.DatetimeIndex(frame["time"]).equals(REQUEST_INDEX):
                raise ValueError(f"{source_id}/{group}: request calendar differs")
            frame = frame.loc[frame["time"].isin(TARGET_INDEX)].copy()
            if not pd.DatetimeIndex(frame["time"]).equals(TARGET_INDEX.rename("time")):
                raise ValueError(f"{source_id}/{group}: final target calendar differs")
            if len(frame) != 8_760 or not frame.loc[:, list(names)].notna().all().all():
                missing = frame.loc[:, list(names)].isna().sum().to_dict()
                raise ValueError(f"{source_id}/{group}: incomplete final fields: {missing}")
            frame.insert(0, "group", group)
            frames.append(frame)
            requests.append(
                {
                    "group": group,
                    "requested_coordinate": {"latitude": latitude, "longitude": longitude},
                    "url": url,
                    "raw_response": file_record(raw_path),
                    "returned_grid": {
                        "latitude": response.get("latitude"),
                        "longitude": response.get("longitude"),
                        "elevation": response.get("elevation"),
                    },
                    "request_rows": len(REQUEST_INDEX),
                    "retained_target_rows": len(frame),
                }
            )
            print(f"downloaded {source_id}/{group}: {len(frame)} retained rows", flush=True)
        combined = pd.concat(frames, ignore_index=True)
        if len(combined) != 3 * 8_760 or combined.duplicated(["group", "time"]).any():
            raise ValueError(f"{source_id}: normalized key coverage differs")
        path = OUTPUT_DIR / str(spec["file"])
        if path.exists():
            existing = pd.read_parquet(path)
            pd.testing.assert_frame_equal(existing, combined, check_exact=True)
        else:
            combined.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
        source_records[source_id] = {
            "model": spec["model"],
            "heights_m": list(spec["heights"]),
            "variables": list(names),
            "requests": requests,
            "normalized": file_record(path),
        }

    manifest = {
        "schema_version": 1,
        "artifact_type": "promoted_multi_nwp_previous_runs_2025_plus_terminal",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "Open-Meteo Previous Model Runs API",
        "endpoint": ENDPOINT,
        "official_documentation": "https://open-meteo.com/en/docs/previous-runs-api",
        "promotion_source": file_record(VALIDATION_PATH),
        "promotion_gate": validation["promotion_gate"],
        "timezone": "Asia/Seoul",
        "wind_speed_unit": "m/s",
        "cell_selection": "nearest",
        "requested_calendar": [str(REQUEST_INDEX[0]), str(REQUEST_INDEX[-1])],
        "retained_competition_index": [str(TARGET_INDEX[0]), str(TARGET_INDEX[-1])],
        "cutoff_semantics": {
            "fixed_feature_id": "B_day1_hours01_13_else_day2_w025",
            "hours_01_through_13": "previous_day1; forecast was generated no later than D-1 13:00 KST",
            "hours_14_through_23_and_00": "previous_day2; forecast was generated before the D-1 14:00 KST cutoff",
            "post_cutoff_forecast_or_observation_used": False,
        },
        "sources": source_records,
        "labels_or_public_feedback_read": 0,
        "downloader": file_record(Path(__file__)),
    }
    atomic_json(manifest, manifest_path)
    print(f"manifest={file_record(manifest_path)}", flush=True)


if __name__ == "__main__":
    main()

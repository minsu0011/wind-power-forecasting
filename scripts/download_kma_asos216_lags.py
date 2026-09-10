from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ENDPOINT = "https://data.kma.go.kr/data/grnd/selectAsosRltmList.do?pgmNo=36"
STATION_ID = 216
STATION_FORM_ID = "162_216"
MAX_STAGE1_DATE = date(2024, 12, 31)
USER_AGENT = "baram2026-causal-kma-feasibility/1.0"

HOURLY_ELEMENTS = (
    "SFC01003001",  # wind speed
    "SFC01003002",  # wind direction
    "SFC01001001",  # temperature
    "SFC01004001",  # relative humidity
    "SFC01005001",  # local pressure
)
HOURLY_GROUPS = ("92", "339", "93", "94")
DAILY_ELEMENTS = (
    "SFC01013001",  # average temperature
    "SFC01015007",  # average wind speed
    "SFC01015009",  # prevailing wind direction
    "SFC01016004",  # average relative humidity
    "SFC01017001",  # average local pressure
)
DAILY_GROUPS = ("102", "104", "105", "106")

MAP_PATTERN = re.compile(r"var egovMapList1\s*=\s*'(.+?)';", re.S)


@dataclass(frozen=True)
class Query:
    kind: str
    start: date
    end: date


def _dates(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _chunks(start: date, end: date, size: int) -> Iterable[tuple[date, date]]:
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=size - 1), end)
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


def _base_form() -> dict[str, str]:
    return {
        "pageIndex": "1",
        "stnIds": STATION_FORM_ID,
        "serviceSe": "F00102",
        "dwldSetupPd": "0",
        "firstLoading": "N",
        "pgmNo": "36",
        "txtStnNm": "태백",
        "lrgClssCd": "SFC",
        "mddlClssCd": "SFC01",
    }


def _form(query: Query) -> dict[str, str]:
    result = _base_form()
    result.update(
        {
            "startDt": query.start.strftime("%Y%m%d"),
            "endDt": query.end.strftime("%Y%m%d"),
        }
    )
    if query.kind == "hour13":
        result.update(
            {
                "dataFormCd": "F00502",
                "elementCds": ",".join(HOURLY_ELEMENTS),
                "elementGroupSns": ",".join(HOURLY_GROUPS),
                "txtElementNm": "풍속,풍향,기온,습도,현지기압",
                "startHh": "13",
                "endHh": "13",
                "pageRowCount": "24",
            }
        )
    elif query.kind == "daily":
        result.update(
            {
                "dataFormCd": "F00501",
                "elementCds": ",".join(DAILY_ELEMENTS),
                "elementGroupSns": ",".join(DAILY_GROUPS),
                "txtElementNm": "평균기온,평균 풍속,최다풍향,평균 상대습도,평균 현지기압",
                "startHh": "",
                "endHh": "",
                "pageRowCount": "31",
            }
        )
    else:
        raise ValueError(query.kind)
    return result


def _request(query: Query, retries: int = 5) -> tuple[Query, list[dict[str, Any]], str]:
    encoded = urllib.parse.urlencode(_form(query)).encode("utf-8")
    request = urllib.request.Request(
        ENDPOINT,
        data=encoded,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": USER_AGENT,
        },
    )
    error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
            text = payload.decode("utf-8")
            match = MAP_PATTERN.search(text)
            if match is None:
                raise ValueError("KMA response did not contain egovMapList1")
            rows = json.loads(match.group(1)) if match.group(1) else []
            return query, rows, hashlib.sha256(payload).hexdigest()
        except Exception as exc:  # network retries must preserve the final cause
            error = exc
            if attempt + 1 < retries:
                time.sleep(float(2**attempt))
    assert error is not None
    raise RuntimeError(f"KMA query failed after {retries} attempts: {query}") from error


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        frame.to_parquet(temporary, engine="pyarrow", compression="zstd", index=False)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(payload: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=date.fromisoformat, default=date(2021, 12, 29))
    parser.add_argument("--end", type=date.fromisoformat, default=MAX_STAGE1_DATE)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/external/kma_asos216_feasibility_v1"),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--forms", choices=("daily", "both"), default="both")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start > args.end:
        raise ValueError("start must not exceed end")
    if args.end > MAX_STAGE1_DATE:
        raise ValueError(
            "Stage1 downloader is physically capped at 2024-12-31; 2025 KMA "
            "observations require a separate post-promotion contract"
        )
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must be in [1, 4]")
    out_dir = args.out_dir.resolve()
    data_path = out_dir / (
        "asos216_daily.parquet" if args.forms == "daily" else "asos216_daily_and_13.parquet"
    )
    manifest_path = out_dir / "source_manifest.json"
    if data_path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite KMA feasibility output: {out_dir}")

    queries: list[Query] = []
    if args.forms == "both":
        queries.extend(Query("hour13", day, day) for day in _dates(args.start, args.end))
    queries.extend(Query("daily", start, end) for start, end in _chunks(args.start, args.end, 10))
    results: list[tuple[Query, list[dict[str, Any]], str]] = []
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_request, query) for query in queries]
        for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            results.append(future.result())
            if done % 100 == 0 or done == len(futures):
                print(f"downloaded {done}/{len(futures)} KMA page queries", flush=True)

    hourly_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    request_ledger: list[dict[str, Any]] = []
    for query, rows, response_sha in sorted(
        results, key=lambda value: (value[0].kind, value[0].start, value[0].end)
    ):
        expected = (
            1 if query.kind == "hour13" else (query.end - query.start).days + 1
        )
        if len(rows) != expected:
            raise ValueError(f"unexpected row count for {query}: {len(rows)} != {expected}")
        request_ledger.append(
            {
                "kind": query.kind,
                "start": query.start.isoformat(),
                "end": query.end.isoformat(),
                "rows": len(rows),
                "response_sha256": response_sha,
            }
        )
        if query.kind == "hour13":
            hourly_rows.extend(rows)
        else:
            daily_rows.extend(rows)

    daily = pd.DataFrame(daily_rows).rename(
        columns={
            "TM": "date",
            "AVG_WS": "avg_ws",
            "MAX_WD": "prevailing_wd",
            "AVG_TA": "avg_ta",
            "AVG_RHM": "avg_hm",
            "AVG_PA": "avg_pa",
        }
    )
    daily["date"] = pd.to_datetime(daily["date"], format="%Y-%m-%d").dt.date
    daily = daily[["date", "avg_ws", "prevailing_wd", "avg_ta", "avg_hm", "avg_pa"]]
    if args.forms == "both":
        hourly = pd.DataFrame(hourly_rows).rename(
            columns={"TM": "timestamp", "WS": "ws13", "WD": "wd13", "TA": "ta13", "HM": "hm13", "PA": "pa13"}
        )
        hourly["date"] = pd.to_datetime(hourly.pop("timestamp"), format="%Y-%m-%d %H:%M").dt.date
        hourly = hourly[["date", "ws13", "wd13", "ta13", "hm13", "pa13"]]
        combined = daily.merge(hourly, on="date", how="outer", validate="one_to_one").sort_values("date")
    else:
        combined = daily.sort_values("date")
    expected_dates = list(_dates(args.start, args.end))
    if combined["date"].tolist() != expected_dates:
        raise ValueError("normalized KMA dates are incomplete, duplicated, or out of order")
    combined.insert(0, "station_id", STATION_ID)
    _atomic_parquet(combined, data_path)

    missing = {
        column: int(combined[column].isna().sum())
        for column in combined.columns
        if column not in {"station_id", "date"}
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "availability_feasibility_only_no_candidate_score",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "official_source": {
            "provider": "Korea Meteorological Administration Weather Data Service",
            "endpoint": ENDPOINT,
            "station": {"id": STATION_ID, "name": "Taebaek", "form_id": STATION_FORM_ID},
            "data_forms": {"hourly": "F00502", "daily": "F00501"},
            "hourly_element_codes": list(HOURLY_ELEMENTS),
            "daily_element_codes": list(DAILY_ELEMENTS),
            "response_parser": "server-rendered egovMapList1 JSON",
        },
        "causal_scope": {
            "start": args.start.isoformat(),
            "end": args.end.isoformat(),
            "hourly_observation_hour_kst": 13,
            "forms_requested": args.forms,
            "2025_observation_requested_or_parsed": False,
            "annual_kpx_generation_used": False,
            "candidate_or_label_score_computed": False,
        },
        "normalized": {
            "rows": len(combined),
            "columns": list(combined.columns),
            "missing_counts": missing,
            "path": str(data_path),
            "sha256": _sha256(data_path),
        },
        "requests": request_ledger,
    }
    _atomic_json(manifest, manifest_path)
    print(json.dumps({"data": str(data_path), "manifest": str(manifest_path), "missing": missing}, ensure_ascii=False))


if __name__ == "__main__":
    main()

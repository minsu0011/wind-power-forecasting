"""Small, cutoff-safe NOAA operational GFS provenance utilities.

This module deliberately does not fit a model or read competition labels.  It
only turns an operating-day policy into exact run/object identities and audits
the publication evidence attached to those objects.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo


KST = ZoneInfo("Asia/Seoul")
UTC = timezone.utc
S3_HTTPS_ROOT = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
GFS_BUCKET = "noaa-gfs-bdp-pds"


@dataclass(frozen=True)
class IndexRecord:
    record_number: int
    offset: int
    initialization: str
    variable: str
    level: str
    forecast_descriptor: str


@dataclass(frozen=True)
class ByteRange:
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start + 1


def cutoff_utc(target_operating_day: date) -> datetime:
    """Return the frozen D-1 14:00 KST submission cutoff in UTC."""

    local = datetime.combine(
        target_operating_day - timedelta(days=1), time(14, 0), tzinfo=KST
    )
    return local.astimezone(UTC)


def conservative_run_init_utc(target_operating_day: date) -> datetime:
    """Return the frozen D-2 12Z operational GFS initialization."""

    return datetime.combine(
        target_operating_day - timedelta(days=2), time(12, 0), tzinfo=UTC
    )


def operating_valid_times_utc(target_operating_day: date) -> tuple[datetime, ...]:
    """D 01:00 through D+1 00:00 KST, converted to 24 UTC instants."""

    local_start = datetime.combine(target_operating_day, time(1, 0), tzinfo=KST)
    return tuple((local_start + timedelta(hours=i)).astimezone(UTC) for i in range(24))


def forecast_hour(run_init: datetime, valid_time: datetime) -> int:
    if run_init.tzinfo is None or valid_time.tzinfo is None:
        raise ValueError("run_init and valid_time must be timezone-aware")
    seconds = (valid_time.astimezone(UTC) - run_init.astimezone(UTC)).total_seconds()
    if seconds < 0 or seconds % 3600:
        raise ValueError("valid_time must be an integral hour at or after run_init")
    return int(seconds // 3600)


def object_key(run_init: datetime, forecast_hour_value: int) -> str:
    run = run_init.astimezone(UTC)
    if run.minute or run.second or run.microsecond or run.hour not in (0, 6, 12, 18):
        raise ValueError("GFS run must be one of 00/06/12/18 UTC")
    if not 0 <= forecast_hour_value <= 384:
        raise ValueError("forecast hour is outside the official 000..384 inventory")
    return (
        f"gfs.{run:%Y%m%d}/{run:%H}/atmos/"
        f"gfs.t{run:%H}z.pgrb2.0p25.f{forecast_hour_value:03d}"
    )


def object_url(key: str) -> str:
    if key.startswith("/") or ".." in key:
        raise ValueError("unsafe object key")
    return f"{S3_HTTPS_ROOT}/{key}"


_INDEX_LINE = re.compile(r"^(\d+):(\d+):d=(\d{10}):(.*)$")


def parse_grib_index(text: str) -> tuple[IndexRecord, ...]:
    records: list[IndexRecord] = []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        raw = raw.strip()
        if not raw:
            continue
        match = _INDEX_LINE.match(raw)
        if match is None:
            raise ValueError(f"malformed GRIB index line {line_number}: {raw!r}")
        tail = match.group(4).split(":")
        if len(tail) < 3:
            raise ValueError(f"incomplete GRIB index line {line_number}: {raw!r}")
        records.append(
            IndexRecord(
                record_number=int(match.group(1)),
                offset=int(match.group(2)),
                initialization=match.group(3),
                variable=tail[0],
                level=tail[1],
                forecast_descriptor=":".join(tail[2:]).rstrip(":"),
            )
        )
    if not records:
        raise ValueError("empty GRIB index")
    if [r.record_number for r in records] != list(range(1, len(records) + 1)):
        raise ValueError("GRIB index records are not contiguous from one")
    offsets = [r.offset for r in records]
    if offsets != sorted(set(offsets)):
        raise ValueError("GRIB index offsets are not strictly increasing")
    return tuple(records)


def record_byte_range(
    records: Sequence[IndexRecord], record_index: int, object_size: int
) -> ByteRange:
    if not 0 <= record_index < len(records):
        raise IndexError(record_index)
    if object_size <= 0:
        raise ValueError("object size must be positive")
    start = records[record_index].offset
    end = (
        records[record_index + 1].offset - 1
        if record_index + 1 < len(records)
        else object_size - 1
    )
    if start < 0 or end < start or end >= object_size:
        raise ValueError("invalid message byte range")
    return ByteRange(start=start, end=end)


def find_unique_record(
    records: Sequence[IndexRecord], variable: str, level: str
) -> int:
    hits = [i for i, row in enumerate(records) if row.variable == variable and row.level == level]
    if len(hits) != 1:
        raise ValueError(
            f"expected one {variable}:{level} record, observed {len(hits)}"
        )
    return hits[0]


def parse_http_datetime(value: str) -> datetime:
    parsed = parsedate_to_datetime(value)
    if parsed.tzinfo is None:
        raise ValueError("HTTP datetime lacks timezone")
    return parsed.astimezone(UTC)


def publication_status(
    *,
    exact_object_key_observed: bool,
    last_modified: datetime | None,
    cutoff: datetime,
    raw_range_sha256: str | None,
    raw_range_bytes: int | None,
    range_matches_index: bool,
) -> str:
    """Fail-closed publication classification used by the pilot ledger."""

    if not exact_object_key_observed:
        return "MISSING_RUN_ID"
    if last_modified is None:
        return "MISSING_PUBLICATION_EVIDENCE"
    if last_modified.astimezone(UTC) > cutoff.astimezone(UTC):
        return "AFTER_CUTOFF"
    if not raw_range_sha256 or not re.fullmatch(r"[0-9a-f]{64}", raw_range_sha256):
        return "MISSING_RAW"
    if not raw_range_bytes or raw_range_bytes <= 0 or not range_matches_index:
        return "FAIL"
    return "VERIFIED"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def normalized_headers(headers: Mapping[str, str] | Iterable[tuple[str, str]]) -> dict[str, str]:
    items = headers.items() if hasattr(headers, "items") else headers
    return {str(key).lower(): str(value).strip() for key, value in items}


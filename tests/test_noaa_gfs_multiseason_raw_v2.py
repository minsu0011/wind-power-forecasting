from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import types
from pathlib import Path

import numpy as np
import pytest

from scripts import run_noaa_gfs_multiseason_raw_v2 as raw


def _write_complete_event(
    root: Path,
    row: dict,
    payload: bytes,
    *,
    start: int,
    stem: str,
) -> None:
    events = root / "raw" / "request_events"
    events.mkdir(parents=True, exist_ok=True)
    end = start + len(payload) - 1
    attempt = int(stem.rsplit("_", 1)[-1])
    event_id = raw.range_request_event_id(row, start, end)
    stem = f"{event_id}__attempt_{attempt:03d}"
    (events / f"{stem}_start.json").write_text(
        json.dumps(
            {
                "event": "RANGE_REQUEST_START",
                "event_id": event_id,
                "attempt": attempt,
                "global_raw_attempt_number": attempt,
                "url": row["retrieval_url_or_request_id"],
                "range_start": start,
                "range_end": end,
            }
        ),
        encoding="utf-8",
    )
    (events / f"{stem}_complete.json").write_text(
        json.dumps(
            {
                "event": "RANGE_REQUEST_COMPLETE",
                "event_id": event_id,
                "attempt": attempt,
                "global_raw_attempt_number": attempt,
                "http_status": 206,
                "range_start": start,
                "range_end": end,
                "content_range": f"bytes {start}-{end}/{row['object_size_bytes']}",
                "bytes": len(payload),
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
                "etag": row["object_etag"],
                "last_modified_utc": row["publication_last_modified_utc"],
                "retrieved_at": "2026-08-10T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )


def _sample_row(payload: bytes) -> dict:
    return {
        "run_init_utc": "2022-01-03T12:00:00Z",
        "forecast_hour": 28,
        "variable": "HPBL",
        "level": "surface",
        "range_start": 100,
        "range_end": 100 + len(payload) - 1,
        "range_bytes": len(payload),
        "retrieval_url_or_request_id": "https://example.invalid/object",
        "source_archive": "NOAA_NODD_S3",
        "archive_product": "gfs",
        "target_operating_day_kst": "2022-01-05",
        "valid_time_utc": "2022-01-04T16:00:00Z",
        "official_metadata": "metadata",
        "publication_evidence_type": "LIST",
        "publication_evidence_reference": "list.xml",
        "cutoff_utc": "2022-01-04T05:00:00Z",
        "cutoff_margin_seconds": 47000,
        "object_key": "gfs.20220103/12/atmos/gfs.t12z.pgrb2.0p25.f028",
        "object_etag": "etag",
        "object_size_bytes": 500,
        "publication_last_modified_utc": "2022-01-03T15:30:00Z",
        "family": "PBL_HEIGHT",
        "idx_sha256": "a" * 64,
    }
def test_independent_go_is_fail_closed_when_absent(tmp_path: Path) -> None:
    authorization = tmp_path / "authorization.json"
    runner = tmp_path / "runner.py"
    census = tmp_path / "census.json"
    coordinate = tmp_path / "coordinate.json"
    for path in (authorization, runner, census, coordinate):
        path.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="GO is absent"):
        raw.validate_independent_go(
            tmp_path / "missing_go.json",
            authorization,
            runner,
            census,
            coordinate,
            {"runtime": "test"},
        )


def test_independent_go_must_bind_every_exact_identity(tmp_path: Path) -> None:
    authorization = tmp_path / "authorization.json"
    runner = tmp_path / "runner.py"
    census = tmp_path / "census.json"
    coordinate = tmp_path / "coordinate.json"
    for path in (authorization, runner, census, coordinate):
        path.write_text(path.name, encoding="utf-8")
    runtime = {"runtime": "test"}
    go = {
        "status": "GO_RAW_RANGE_LAUNCH",
        "bound_authorization_sha256": raw.sha256_file(authorization),
        "bound_runner_sha256": raw.sha256_file(runner),
        "bound_census_manifest_sha256": raw.sha256_file(census),
        "bound_coordinate_lock_sha256": raw.sha256_file(coordinate),
        "bound_runtime_identity_sha256": raw.canonical_payload_sha256(runtime),
        "expected_range_rows": raw.EXPECTED_RANGE_ROWS,
        "expected_range_bytes": raw.EXPECTED_RANGE_BYTES,
        "max_actual_http_attempts": 15_200,
        "census_worst_case_http_attempts": 4_800,
        "census_plus_raw_max_http_attempts": 20_000,
    }
    go_path = tmp_path / "go.json"
    go_path.write_text(json.dumps(go), encoding="utf-8")
    assert (
        raw.validate_independent_go(
            go_path, authorization, runner, census, coordinate, runtime
        )
        == go
    )
    go["expected_range_bytes"] -= 1
    go_path.write_text(json.dumps(go), encoding="utf-8")
    with pytest.raises(RuntimeError, match="expected_range_bytes"):
        raw.validate_independent_go(
            go_path, authorization, runner, census, coordinate, runtime
        )


def test_census_access_binding_and_final_disk_gate_fail_early(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    access_path = tmp_path / "audit" / "CENSUS_CUMULATIVE_ACCESS_AMENDMENT.json"
    access_path.parent.mkdir(parents=True)
    payload = {
        "artifact_type": "CENSUS_CUMULATIVE_ACCESS_APPEND_ONLY_AMENDMENT",
        "initial_population_invocation": {"successful_http_requests": 1_200},
        "phase_total": {
            "successful_http_requests": 1_200,
            "raw_range_requests": 0,
            "raw_range_bytes": 0,
        },
        "raw_payload_read": False,
    }
    access_path.write_text(json.dumps(payload), encoding="utf-8")
    authorization = {
        "census_cumulative_access": {
            **raw.identity(access_path, tmp_path),
            "successful_http_requests": 1_200,
            "attempts_per_logical_request_hard_max": 4,
            "worst_case_http_attempts": 4_800,
        }
    }
    assert raw.validate_census_access_binding(
        authorization, access_path, tmp_path
    ) == payload
    authorization["census_cumulative_access"]["successful_http_requests"] = 1_199
    with pytest.raises(RuntimeError, match="successful_http_requests"):
        raw.validate_census_access_binding(authorization, access_path, tmp_path)

    monkeypatch.setattr(
        raw.shutil,
        "disk_usage",
        lambda path: types.SimpleNamespace(free=199_999_999_999),
    )
    with pytest.raises(RuntimeError, match="final 200 GB"):
        raw.require_final_disk_reserve(tmp_path)


def test_raw_partial_range_resumes_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"GRIBpayload7777"
    row = {
        "run_init_utc": "2022-01-03T12:00:00Z",
        "forecast_hour": 28,
        "variable": "HPBL",
        "level": "surface",
        "range_start": 100,
        "range_end": 100 + len(payload) - 1,
        "range_bytes": len(payload),
        "retrieval_url_or_request_id": "https://example.invalid/object",
        "source_archive": "NOAA_NODD_S3",
        "archive_product": "gfs",
        "target_operating_day_kst": "2022-01-05",
        "valid_time_utc": "2022-01-04T16:00:00Z",
        "official_metadata": "metadata",
        "publication_evidence_type": "LIST",
        "publication_evidence_reference": "list.xml",
        "cutoff_utc": "2022-01-04T05:00:00Z",
        "cutoff_margin_seconds": 47000,
        "object_key": "gfs.20220103/12/atmos/gfs.t12z.pgrb2.0p25.f028",
        "object_etag": "etag",
        "object_size_bytes": 500,
        "publication_last_modified_utc": "2022-01-03T15:30:00Z",
        "family": "PBL_HEIGHT",
        "idx_sha256": "a" * 64,
    }
    destination = raw.raw_path_for(tmp_path, row)
    part = destination.with_suffix(".grib2.part")
    part.parent.mkdir(parents=True)
    part.write_bytes(payload[:4])
    _write_complete_event(
        tmp_path,
        row,
        payload[:4],
        start=row["range_start"],
        stem="prefix__attempt_001",
    )
    calls = []

    def fake_fetch(
        url: str,
        start: int,
        end: int,
        *,
        attempts: int = 4,
        event_root: Path | None = None,
        event_id: str | None = None,
        request_budget: raw.RequestBudget | None = None,
        expected_object_size: int | None = None,
        expected_etag: str | None = None,
        expected_last_modified_utc: str | None = None,
    ):
        calls.append((url, start, end))
        assert start == 104
        assert end == row["range_end"]
        _write_complete_event(
            tmp_path,
            row,
            payload[4:],
            start=start,
            stem="suffix__attempt_001",
        )
        return payload[4:], {"content-range": f"bytes {start}-{end}/500"}, 206, "2026-08-10T00:00:00Z"

    monkeypatch.setattr(raw, "fetch_exact_range", fake_fetch)
    first = raw.download_or_resume_range(tmp_path, row)
    assert first["path"].read_bytes() == payload
    assert first["meta"]["resumed_from_bytes"] == 4
    second = raw.download_or_resume_range(tmp_path, row)
    assert second["network"] is False
    assert len(calls) == 1


def test_orphan_final_recovers_meta_part_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"GRIBpayload7777"
    row = {
        "run_init_utc": "2022-01-03T12:00:00Z",
        "forecast_hour": 28,
        "variable": "HPBL",
        "level": "surface",
        "range_start": 100,
        "range_end": 100 + len(payload) - 1,
        "range_bytes": len(payload),
        "retrieval_url_or_request_id": "https://example.invalid/object",
        "source_archive": "NOAA_NODD_S3",
        "archive_product": "gfs",
        "target_operating_day_kst": "2022-01-05",
        "valid_time_utc": "2022-01-04T16:00:00Z",
        "official_metadata": "metadata",
        "publication_evidence_type": "LIST",
        "publication_evidence_reference": "list.xml",
        "cutoff_utc": "2022-01-04T05:00:00Z",
        "cutoff_margin_seconds": 47000,
        "object_key": "gfs.20220103/12/atmos/gfs.t12z.pgrb2.0p25.f028",
        "object_etag": "etag",
        "object_size_bytes": 500,
        "publication_last_modified_utc": "2022-01-03T15:30:00Z",
        "family": "PBL_HEIGHT",
        "idx_sha256": "a" * 64,
    }
    destination = raw.raw_path_for(tmp_path, row)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(payload)
    meta = {
        "source_archive": row["source_archive"],
        "archive_product": row["archive_product"],
        "target_operating_day_kst": row["target_operating_day_kst"],
        "run_init_utc": row["run_init_utc"],
        "forecast_hour": 28,
        "valid_time_utc": row["valid_time_utc"],
        "retrieval_url_or_request_id": row["retrieval_url_or_request_id"],
        "retrieved_at": "2026-08-10T00:00:00Z",
        "raw_filename": destination.relative_to(tmp_path).as_posix(),
        "raw_sha256": hashlib.sha256(payload).hexdigest(),
        "raw_size_bytes": len(payload),
        "official_metadata": "metadata",
        "publication_evidence_type": "LIST",
        "publication_evidence_reference": "list.xml",
        "cutoff_utc": row["cutoff_utc"],
        "cutoff_margin": 47000,
        "object_key": row["object_key"],
        "object_etag": "etag",
        "object_size_bytes": 500,
        "publication_last_modified_utc": row["publication_last_modified_utc"],
        "range_start": 100,
        "range_end": row["range_end"],
        "range_bytes": len(payload),
        "family": "PBL_HEIGHT",
        "variable": "HPBL",
        "level": "surface",
        "idx_sha256": "a" * 64,
        "http_status": 206,
        "content_range": f"bytes 100-{row['range_end']}/500",
        "request_range_start": 100,
        "resumed_from_bytes": 0,
        "status": "VERIFIED",
    }
    meta_part = destination.with_suffix(".grib2.meta.part.json")
    meta_part.write_text(json.dumps(meta), encoding="utf-8")

    def forbidden_fetch(*args, **kwargs):
        raise AssertionError("orphan recovery must not use network")

    monkeypatch.setattr(raw, "fetch_exact_range", forbidden_fetch)
    result = raw.download_or_resume_range(tmp_path, row)
    assert result["network"] is False
    assert result["recovered_orphan_final"] is True
    assert destination.with_suffix(".grib2.meta.json").is_file()
    assert not meta_part.exists()


@pytest.mark.parametrize("tamper", [False, True])
def test_completed_part_recovers_only_from_exact_durable_request_event(
    tmp_path: Path, tamper: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = b"GRIBpayload7777"
    payload = b"GRIBpayloae7777" if tamper else original
    row = {
        "run_init_utc": "2022-01-03T12:00:00Z",
        "forecast_hour": 28,
        "variable": "HPBL",
        "level": "surface",
        "range_start": 100,
        "range_end": 100 + len(original) - 1,
        "range_bytes": len(original),
        "retrieval_url_or_request_id": "https://example.invalid/object",
        "source_archive": "NOAA_NODD_S3",
        "archive_product": "gfs",
        "target_operating_day_kst": "2022-01-05",
        "valid_time_utc": "2022-01-04T16:00:00Z",
        "official_metadata": "metadata",
        "publication_evidence_type": "LIST",
        "publication_evidence_reference": "list.xml",
        "cutoff_utc": "2022-01-04T05:00:00Z",
        "cutoff_margin_seconds": 47000,
        "object_key": "gfs.20220103/12/atmos/gfs.t12z.pgrb2.0p25.f028",
        "object_etag": "etag",
        "object_size_bytes": 500,
        "publication_last_modified_utc": "2022-01-03T15:30:00Z",
        "family": "PBL_HEIGHT",
        "idx_sha256": "a" * 64,
    }
    destination = raw.raw_path_for(tmp_path, row)
    part = destination.with_suffix(".grib2.part")
    part.parent.mkdir(parents=True)
    part.write_bytes(payload)
    events = tmp_path / "raw" / "request_events"
    events.mkdir(parents=True)
    event_id = raw.range_request_event_id(
        row, int(row["range_start"]), int(row["range_end"])
    )
    stem = f"{event_id}__attempt_001"
    (events / f"{stem}_start.json").write_text(
        json.dumps(
            {
                "event": "RANGE_REQUEST_START",
                "event_id": event_id,
                "attempt": 1,
                "global_raw_attempt_number": 1,
                "url": row["retrieval_url_or_request_id"],
                "range_start": row["range_start"],
                "range_end": row["range_end"],
            }
        ),
        encoding="utf-8",
    )
    (events / f"{stem}_complete.json").write_text(
        json.dumps(
            {
                "event": "RANGE_REQUEST_COMPLETE",
                "event_id": event_id,
                "attempt": 1,
                "global_raw_attempt_number": 1,
                "http_status": 206,
                "range_start": row["range_start"],
                "range_end": row["range_end"],
                "content_range": f"bytes {row['range_start']}-{row['range_end']}/500",
                "bytes": len(original),
                "payload_sha256": hashlib.sha256(original).hexdigest(),
                "etag": "etag",
                "last_modified_utc": row["publication_last_modified_utc"],
                "retrieved_at": "2026-08-10T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    def forbidden_fetch(*args, **kwargs):
        raise AssertionError("completed-part recovery must not use network")

    monkeypatch.setattr(raw, "fetch_exact_range", forbidden_fetch)
    if tamper:
        with pytest.raises(RuntimeError, match="0 exact durable completion events"):
            raw.download_or_resume_range(tmp_path, row)
    else:
        result = raw.download_or_resume_range(tmp_path, row)
        assert result["network"] is False
        assert result["recovered_completed_part_from_request_event"] is True
        assert destination.read_bytes() == original
        assert destination.with_suffix(".grib2.meta.json").is_file()


@pytest.mark.parametrize("tamper_prefix", [False, True])
def test_resumed_suffix_complete_part_recovers_exact_two_segment_sequence(
    tmp_path: Path,
    tamper_prefix: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"GRIBpayload7777"
    row = _sample_row(payload)
    destination = raw.raw_path_for(tmp_path, row)
    part = destination.with_suffix(".grib2.part")
    prefix_path = destination.with_suffix(".grib2.prefix.json")
    part.parent.mkdir(parents=True)
    part.write_bytes(payload[:4])
    _write_complete_event(
        tmp_path,
        row,
        payload[:4],
        start=row["range_start"],
        stem="prefix__attempt_001",
    )
    raw.ensure_resume_prefix(tmp_path, part, prefix_path, row, 4)
    part.write_bytes((b"XRIB" if tamper_prefix else payload[:4]) + payload[4:])
    _write_complete_event(
        tmp_path,
        row,
        payload[4:],
        start=row["range_start"] + 4,
        stem="suffix__attempt_001",
    )
    # A duplicate semantically identical completion is valid and deterministic.
    _write_complete_event(
        tmp_path,
        row,
        payload[4:],
        start=row["range_start"] + 4,
        stem="suffix_dup__attempt_002",
    )

    def forbidden_fetch(*args, **kwargs):
        raise AssertionError("completed two-segment recovery must use zero network")

    monkeypatch.setattr(raw, "fetch_exact_range", forbidden_fetch)
    if tamper_prefix:
        with pytest.raises(RuntimeError, match="prefix record mismatch"):
            raw.download_or_resume_range(tmp_path, row)
    else:
        result = raw.download_or_resume_range(tmp_path, row)
        assert result["network"] is False
        assert result["meta"]["resumed_from_bytes"] == 4
        assert (
            result["meta"]["request_completion_evidence"]
            ["semantically_identical_completion_event_count"]
            == 2
        )


def test_event_lookup_never_opens_unrelated_complete_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"GRIBpayload7777"
    row = _sample_row(payload)
    _write_complete_event(
        tmp_path,
        row,
        payload,
        start=row["range_start"],
        stem="exact__attempt_001",
    )
    events = tmp_path / "raw" / "request_events"
    for index in range(1_000):
        (events / f"unrelated_{index:04d}__attempt_001_complete.json").write_text(
            "malformed", encoding="utf-8"
        )
    original_read_text = Path.read_text
    original_glob = Path.glob

    def guarded_read_text(path: Path, *args, **kwargs):
        if path.name.startswith("unrelated_"):
            raise AssertionError("unrelated completion event was opened")
        return original_read_text(path, *args, **kwargs)

    def forbidden_event_glob(path: Path, pattern: str):
        if path.resolve() == events.resolve():
            raise AssertionError("per-request lookup used directory glob/scandir")
        return original_glob(path, pattern)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    monkeypatch.setattr(Path, "glob", forbidden_event_glob)
    matches = raw._matching_complete_events(
        tmp_path,
        row,
        request_start=row["range_start"],
        request_end=row["range_end"],
        payload_bytes=len(payload),
        payload_sha256=hashlib.sha256(payload).hexdigest(),
    )
    assert len(matches) == 1


@pytest.mark.parametrize(
    ("target", "field", "value"),
    [
        ("start", "event_id", "wrong"),
        ("complete", "attempt", 2),
        ("complete", "global_raw_attempt_number", 2),
        ("complete", "event", "RANGE_REQUEST_ERROR"),
    ],
)
def test_request_event_payload_path_identity_tamper_fails_closed(
    tmp_path: Path, target: str, field: str, value: object
) -> None:
    payload = b"GRIBpayload7777"
    row = _sample_row(payload)
    _write_complete_event(
        tmp_path,
        row,
        payload,
        start=row["range_start"],
        stem="exact__attempt_001",
    )
    event_id = raw.range_request_event_id(
        row, row["range_start"], row["range_end"]
    )
    path = (
        tmp_path
        / "raw"
        / "request_events"
        / f"{event_id}__attempt_001_{target}.json"
    )
    event = json.loads(path.read_text(encoding="utf-8"))
    event[field] = value
    path.write_text(json.dumps(event), encoding="utf-8")
    with pytest.raises(RuntimeError, match="payload/path identity mismatch"):
        raw._matching_complete_events(
            tmp_path,
            row,
            request_start=row["range_start"],
            request_end=row["range_end"],
            payload_bytes=len(payload),
            payload_sha256=hashlib.sha256(payload).hexdigest(),
        )


def test_request_event_complete_without_start_fails_no_gap_pairing(
    tmp_path: Path,
) -> None:
    event_root = tmp_path / "events"
    event_root.mkdir()
    event_id = "a" * 32
    (event_root / f"{event_id}__attempt_001_complete.json").write_text(
        "{}", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="lacks paired start"):
        raw._request_attempt_records(event_root, event_id)


def test_attempt_15200_complete_is_readable_without_attempt_15201(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"GRIBpayload7777"
    row = _sample_row(payload)
    event_id = raw.range_request_event_id(
        row, row["range_start"], row["range_end"]
    )

    def parse(path: Path) -> tuple[int, str]:
        tail = path.name.split("__attempt_", 1)[1]
        attempt = int(tail.split("_", 1)[0])
        kind = tail.rsplit("_", 1)[1].split(".", 1)[0]
        return attempt, kind

    def fake_is_file(path: Path) -> bool:
        if "__attempt_" not in path.name:
            return False
        attempt, kind = parse(path)
        return 1 <= attempt <= 15_200 and kind in {"start", "complete"}

    def fake_read_text(path: Path, *args, **kwargs) -> str:
        attempt, kind = parse(path)
        if kind == "start":
            value = {
                "event": "RANGE_REQUEST_START",
                "event_id": event_id,
                "attempt": attempt,
                "global_raw_attempt_number": attempt,
                "url": row["retrieval_url_or_request_id"],
                "range_start": row["range_start"],
                "range_end": row["range_end"],
            }
        else:
            value = {
                "event": "RANGE_REQUEST_COMPLETE",
                "event_id": event_id,
                "attempt": attempt,
                "global_raw_attempt_number": attempt,
                "http_status": 206,
                "range_start": row["range_start"],
                "range_end": row["range_end"],
                "content_range": (
                    f"bytes {row['range_start']}-{row['range_end']}/"
                    f"{row['object_size_bytes']}"
                ),
                "bytes": len(payload),
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
                "etag": row["object_etag"],
                "last_modified_utc": row["publication_last_modified_utc"],
                "retrieved_at": "2026-08-10T00:00:00Z",
            }
        return json.dumps(value)

    monkeypatch.setattr(Path, "is_file", fake_is_file)
    monkeypatch.setattr(Path, "read_text", fake_read_text)
    matches = raw._matching_complete_events(
        tmp_path,
        row,
        request_start=row["range_start"],
        request_end=row["range_end"],
        payload_bytes=len(payload),
        payload_sha256=hashlib.sha256(payload).hexdigest(),
    )
    assert len(matches) == 15_200
    assert matches[-1][3]["attempt"] == 15_200


def test_mid_append_crash_truncates_to_durable_prefix_then_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"GRIBpayload7777"
    row = _sample_row(payload)
    destination = raw.raw_path_for(tmp_path, row)
    part = destination.with_suffix(".grib2.part")
    prefix_path = destination.with_suffix(".grib2.prefix.json")
    part.parent.mkdir(parents=True)
    part.write_bytes(payload[:4])
    _write_complete_event(
        tmp_path,
        row,
        payload[:4],
        start=row["range_start"],
        stem="prefix__attempt_001",
    )
    raw.ensure_resume_prefix(tmp_path, part, prefix_path, row, 4)
    # Crash while appending the suffix: these bytes have no committed boundary.
    part.write_bytes(payload[:7])
    _write_complete_event(
        tmp_path,
        row,
        payload[4:],
        start=row["range_start"] + 4,
        stem="suffix_old__attempt_001",
    )
    calls: list[tuple[int, int]] = []

    def fake_fetch(url: str, start: int, end: int, **kwargs):
        calls.append((start, end))
        _write_complete_event(
            tmp_path,
            row,
            payload[4:],
            start=start,
            stem="suffix_new__attempt_002",
        )
        return (
            payload[4:],
            {"content-range": f"bytes {start}-{end}/{row['object_size_bytes']}"},
            206,
            "2026-08-10T00:00:01Z",
        )

    monkeypatch.setattr(raw, "fetch_exact_range", fake_fetch)
    result = raw.download_or_resume_range(tmp_path, row)
    assert calls == [(row["range_start"] + 4, row["range_end"])]
    assert result["path"].read_bytes() == payload
    recovery_events = list(
        (tmp_path / "raw" / "resume_recovery_events").glob("*.json")
    )
    assert len(recovery_events) == 1
    recovery = json.loads(recovery_events[0].read_text(encoding="utf-8"))
    assert recovery["old_part_size_bytes"] == 7
    assert recovery["retained_durable_prefix_bytes"] == 4


def test_bounded_parallel_map_never_submits_after_same_batch_failure() -> None:
    barrier = threading.Barrier(4)
    calls: list[int] = []
    lock = threading.Lock()

    def worker(item: int) -> int:
        with lock:
            calls.append(item)
        barrier.wait(timeout=5)
        if item == 0:
            raise RuntimeError("first batch failure")
        return item

    with pytest.raises(RuntimeError, match="first batch failure"):
        list(raw.bounded_parallel_map(range(100), worker, max_workers=4))
    assert sorted(calls) == [0, 1, 2, 3]


def test_output_transaction_recovers_partial_commit_and_ignores_unplanned_staging(
    tmp_path: Path,
) -> None:
    destinations = {
        "alpha": tmp_path / "decoded" / "alpha.json",
        "beta": tmp_path / "raw" / "beta.csv",
    }
    abandoned = raw.staged_output_paths(tmp_path, "decode_failed", destinations)
    abandoned["alpha"].parent.mkdir(parents=True, exist_ok=True)
    abandoned["alpha"].write_text("decode failed before audit", encoding="utf-8")
    assert raw.recover_or_validate_output_transaction(tmp_path, destinations) is None
    assert all(not path.exists() for path in destinations.values())

    staged = raw.staged_output_paths(tmp_path, "successful", destinations)
    for name, path in staged.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{name}\n", encoding="utf-8")
    plan = raw.write_output_transaction_plan(
        tmp_path, "successful", staged, destinations
    )
    # Simulate a process death after the first canonical rename.
    destinations["alpha"].parent.mkdir(parents=True, exist_ok=True)
    os.replace(staged["alpha"], destinations["alpha"])
    committed = raw.recover_or_validate_output_transaction(tmp_path, destinations)
    assert committed is not None and committed.is_file()
    assert destinations["alpha"].read_text(encoding="utf-8") == "alpha\n"
    assert destinations["beta"].read_text(encoding="utf-8") == "beta\n"
    assert raw.commit_output_transaction(tmp_path, plan, destinations) == committed


def test_request_budget_is_thread_safe_and_hard_capped() -> None:
    budget = raw.RequestBudget(maximum_attempts=3, initial_attempts=1)
    assert budget.reserve() == 2
    assert budget.reserve() == 3
    with pytest.raises(RuntimeError, match="budget exhausted"):
        budget.reserve()
    assert budget.attempts == 3


def test_transport_start_and_complete_events_use_fsync_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"GRIBpayload7777"

    class Response:
        status = 206
        headers = {
            "Content-Length": str(len(payload)),
            "Content-Range": f"bytes 100-{100 + len(payload) - 1}/500",
            "ETag": '"etag"',
            "Last-Modified": "Mon, 03 Jan 2022 15:30:00 GMT",
        }

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self, amount: int) -> bytes:
            return payload

    monkeypatch.setattr(raw.urllib.request, "urlopen", lambda request, timeout: Response())
    fsynced: list[Path] = []
    original = raw.write_json_exclusive_fsync

    def recording_writer(path: Path, value: object) -> None:
        fsynced.append(path)
        original(path, value)

    monkeypatch.setattr(raw, "write_json_exclusive_fsync", recording_writer)
    raw.fetch_exact_range(
        "https://example.invalid/object",
        100,
        100 + len(payload) - 1,
        attempts=1,
        event_root=tmp_path / "events",
        event_id="transport",
        request_budget=raw.RequestBudget(1),
        expected_object_size=500,
        expected_etag="etag",
        expected_last_modified_utc="2022-01-03T15:30:00Z",
    )
    assert [path.name for path in fsynced] == [
        "transport__attempt_001_start.json",
        "transport__attempt_001_complete.json",
    ]


def test_exclusive_launch_claim_rejects_live_pid_and_archives_completion(
    tmp_path: Path,
) -> None:
    first = raw.LaunchClaim.acquire(tmp_path)
    with pytest.raises(RuntimeError, match="another raw launch is active"):
        raw.LaunchClaim.acquire(tmp_path)
    first.close("complete")
    assert not (tmp_path / "raw" / "RAW_LAUNCH_ACTIVE.lock").exists()
    history = list((tmp_path / "raw" / "launch_history").glob("*__complete.lock"))
    assert len(history) == 1


def test_bilinear_regular_ll_is_deterministic() -> None:
    grid = np.asarray([[0.0, 10.0], [20.0, 30.0]])
    observed = raw.bilinear_regular_ll(
        grid.ravel(),
        latitude=0.5,
        longitude=0.5,
        ni=2,
        nj=2,
        first_latitude=1.0,
        first_longitude=0.0,
        di=1.0,
        dj=1.0,
        missing_value=9999.0,
    )
    assert observed == 15.0
    missing_grid = grid.copy()
    missing_grid[1, 1] = 9999.0
    assert np.isnan(
        raw.bilinear_regular_ll(
            missing_grid.ravel(),
            latitude=0.5,
            longitude=0.5,
            ni=2,
            nj=2,
            first_latitude=1.0,
            first_longitude=0.0,
            di=1.0,
            dj=1.0,
            missing_value=9999.0,
        )
    )


def _mock_eccodes(*, extra_message: bool, first_latitude: float) -> types.SimpleNamespace:
    keys = {
        "discipline": 0,
        "parameterCategory": 3,
        "parameterNumber": 196,
        "shortName": "unknown",
        "typeOfLevel": "surface",
        "level": 0,
        "dataDate": 20220103,
        "dataTime": 1200,
        "forecastTime": 28,
        "stepUnits": 1,
        "indicatorOfUnitOfTimeRange": 1,
        "validityDate": 20220104,
        "validityTime": 1600,
        "gridType": "regular_ll",
        "Ni": 1440,
        "Nj": 721,
        "latitudeOfFirstGridPointInDegrees": first_latitude,
        "longitudeOfFirstGridPointInDegrees": 0.0,
        "iDirectionIncrementInDegrees": 0.25,
        "jDirectionIncrementInDegrees": 0.25,
        "iScansNegatively": 0,
        "jScansPositively": 0,
        "jPointsAreConsecutive": 0,
        "alternativeRowScanning": 0,
        "missingValue": 9999.0,
    }
    handles = iter([keys, keys if extra_message else None])
    return types.SimpleNamespace(
        codes_grib_new_from_file=lambda stream: next(handles),
        codes_get=lambda handle, key: handle[key],
        codes_get_array=lambda handle, key: np.zeros(1, dtype=float),
        codes_release=lambda handle: None,
    )


def test_decode_rejects_wrong_grid_origin_and_extra_grib_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "message.grib2"
    path.write_bytes(b"GRIBpayload7777")
    row = {
        "forecast_hour": 28,
        "variable": "HPBL",
        "level": "surface",
        "run_init_utc": "2022-01-03T12:00:00Z",
        "valid_time_utc": "2022-01-04T16:00:00Z",
    }
    sites = [{"latitude": 37.0, "longitude": 129.0}]
    monkeypatch.setitem(
        sys.modules,
        "eccodes",
        _mock_eccodes(extra_message=False, first_latitude=89.75),
    )
    with pytest.raises(RuntimeError, match="metadata mismatch"):
        raw.decode_message({"path": path}, row, sites)
    monkeypatch.setitem(
        sys.modules,
        "eccodes",
        _mock_eccodes(extra_message=True, first_latitude=90.0),
    )
    with pytest.raises(RuntimeError, match="more than one GRIB message"):
        raw.decode_message({"path": path}, row, sites)


def test_direct_eccodes_decode_smoke_on_prior_sha_bound_hpbl() -> None:
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "artifacts"
        / "baram2026_ncei_scada_research_20260810_210756"
        / "track_a"
        / "provenance"
        / "raw"
        / "f039"
        / "gfs.t12z.pgrb2.0p25.f039.HPBL_surface.grib2"
    )
    if not path.is_file():
        pytest.skip("prior SHA-bound pilot message is absent")
    result = {"path": path}
    row = {
        "forecast_hour": 39,
        "variable": "HPBL",
        "level": "surface",
        "run_init_utc": "2023-07-01T12:00:00Z",
        "valid_time_utc": "2023-07-03T03:00:00Z",
    }
    sites = [{"latitude": 37.28211388888889, "longitude": 128.95058333333333}]
    decoded = raw.decode_message(result, row, sites)
    assert decoded["feature"] == "HPBL_surface"
    assert len(decoded["site_values"]) == 1
    assert np.isfinite(decoded["site_values"][0])
    assert 0.0 <= decoded["site_values"][0] <= 10_000.0
    with pytest.raises(RuntimeError, match="timestamp mismatch"):
        raw.decode_message(
            result,
            {**row, "valid_time_utc": "2023-07-03T04:00:00Z"},
            sites,
        )
    with pytest.raises(RuntimeError, match="variable identity mismatch"):
        raw.decode_message(result, {**row, "variable": "UGRD"}, sites)

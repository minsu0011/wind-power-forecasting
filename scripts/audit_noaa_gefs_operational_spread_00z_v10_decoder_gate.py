#!/usr/bin/env python3
"""Offline-only fresh-process regression for the v10 ecCodes cold gate."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import download_noaa_gefs_operational_spread_00z_original_v10 as extractor  # noqa: E402


FIXTURE = PROJECT_ROOT / "artifacts/external/noaa_gefs_operational_spread_00z_original_v7/preflight_one_range/response.grib2"
FIXTURE_RESULT = PROJECT_ROOT / "artifacts/external/noaa_gefs_operational_spread_00z_original_v7/preflight_one_range/preflight_result.json"
FIXTURE_SHA256 = "e95bb425e5d743f3d653a026dfb24a507c3a9c5bb5772d5af0817befbc430244"
FIXTURE_RESULT_SHA256 = "bd70d49e156eac7f8e04f5babfe6cbbf2c22ff2a919a7c72b13946953f11241d"
THREADS = 24
TASKS = 240


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(metadata: list[dict[str, Any]], values: list[float]) -> str:
    raw = json.dumps(
        {"metadata": metadata, "values": values},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_fixture() -> tuple[bytes, extractor.RangePlan, list[dict[str, Any]], list[float]]:
    if sha256_file(FIXTURE) != FIXTURE_SHA256:
        raise RuntimeError("offline fixture bytes differ")
    if sha256_file(FIXTURE_RESULT) != FIXTURE_RESULT_SHA256:
        raise RuntimeError("offline fixture result differs")
    result = json.loads(FIXTURE_RESULT.read_text(encoding="utf-8"))
    plans, _ = extractor.build_stage1_plan()
    plan = extractor.fixed_preflight_plan(plans)
    if result["fixed_range"] != extractor.asdict(plan):
        raise RuntimeError("offline fixture plan differs")
    if result.get("request_accounting") != {"requests": 1, "response_bytes": 464970, "retries": 0}:
        raise RuntimeError("offline fixture request accounting differs")
    metadata = result["message_decoder"]["metadata"]
    values = [float(value) for value in result["message_decoder"]["values"]]
    if values != [3.41, 1.31] or len(metadata) != 2:
        raise RuntimeError("offline fixture reference values/metadata differ")
    return FIXTURE.read_bytes(), plan, metadata, values


def run_tasks(
    decoder: Callable[[bytes, extractor.RangePlan], tuple[list[dict[str, Any]], list[float]]],
) -> dict[str, Any]:
    payload, plan, expected_metadata, expected_values = load_fixture()
    reference_digest = canonical_digest(expected_metadata, expected_values)
    barrier = threading.Barrier(THREADS)
    range_get_calls = 0

    def forbidden_range_get(*args: Any, **kwargs: Any) -> Any:
        nonlocal range_get_calls
        range_get_calls += 1
        raise AssertionError("offline decoder regression attempted a network range GET")

    extractor.legacy.range_get = forbidden_range_get

    def one(index: int) -> str:
        if index < THREADS:
            barrier.wait(timeout=60)
        metadata, values = decoder(payload, plan)
        if metadata != expected_metadata or values != expected_values:
            raise AssertionError("decoder metadata/value output differs from sealed reference")
        return canonical_digest(metadata, values)

    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS, thread_name_prefix="gefs-v10-offline") as pool:
        digests = list(pool.map(one, range(TASKS)))
    if range_get_calls != 0:
        raise AssertionError("offline regression made a range GET")
    if len(digests) != TASKS or set(digests) != {reference_digest}:
        raise AssertionError("offline regression digest closure differs")
    return {
        "tasks": len(digests),
        "threads": THREADS,
        "errors": 0,
        "reference_digest": reference_digest,
        "unique_result_digests": len(set(digests)),
        "exact_metadata_equality_tasks": len(digests),
        "exact_value_equality_tasks": len(digests),
        "values": expected_values,
        "metadata_records_per_task": len(expected_metadata),
        "range_get_calls": range_get_calls,
        "new_external_value_bytes": 0,
        "fixture_future_pack_or_model_reuse": 0,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("cold-unlocked", "locked"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps({"event": "v10_offline_decoder_regression_start", "mode": args.mode, "threads": THREADS, "tasks": TASKS, "network_requests": 0}, sort_keys=True), flush=True)
    if args.mode == "cold-unlocked":
        result = run_tasks(extractor._decode_two_messages_unlocked)
        print(json.dumps({"event": "UNEXPECTED_COLD_UNLOCKED_COMPLETION", **result}, sort_keys=True), flush=True)
        return 3
    result = run_tasks(extractor.decode_two_messages)
    gate = extractor.decoder_gate_state()
    if gate != {"ready": True, "failure": None, "first_success_count": 1}:
        raise AssertionError(f"decoder gate final state differs: {gate}")
    print(json.dumps({"event": "PASS_V10_LOCKED_DECODER_GATE", **result, "decoder_gate": gate, "labels_fits_predictions_scores_2024_2025": 0}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

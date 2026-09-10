#!/usr/bin/env python
"""Append-only correction of the Track-A direct ecCodes decoder audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
ROOT_DEFAULT = (
    REPO
    / "artifacts"
    / "baram2026_ncei_scada_research_20260810_210756"
    / "track_a"
)
PARENT_V2_SHA256 = "7d2d9478b349fd44f4f60874c3a16bb93a9d94c2f5969ba7ee2a2f3eccd05296"
PREVIOUS_FEASIBILITY_SHA256 = "4458a2d562576c79e208188b55606d29756c515c0c2faf62ea68be453844c20b"
RAW_F039_SHA256 = "575fe7876ab93e5470c43dc09c78895385925f5662cca75bd0d8c7f91e677407"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path, root: Path | None = None) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix() if root else str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_json_exclusive(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def write_text_exclusive(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def decode_direct_eccodes(raw_path: Path) -> dict[str, Any]:
    import eccodes

    started = time.perf_counter()
    with raw_path.open("rb") as stream:
        handle = eccodes.codes_grib_new_from_file(stream)
        if handle is None:
            raise RuntimeError("ecCodes returned no GRIB handle")
        try:
            metadata = {
                key: eccodes.codes_get(handle, key)
                for key in (
                    "discipline",
                    "parameterCategory",
                    "parameterNumber",
                    "typeOfLevel",
                    "level",
                    "dataDate",
                    "dataTime",
                    "forecastTime",
                    "stepRange",
                    "validityDate",
                    "validityTime",
                    "gridType",
                    "Ni",
                    "Nj",
                    "numberOfDataPoints",
                )
            }
            values = eccodes.codes_get_array(handle, "values")
        finally:
            eccodes.codes_release(handle)
    elapsed = time.perf_counter() - started
    if (
        metadata["typeOfLevel"] != "surface"
        or int(metadata["forecastTime"]) != 39
        or int(metadata["dataDate"]) != 20230701
        or int(metadata["dataTime"]) != 1200
        or int(metadata["validityDate"]) != 20230703
        or int(metadata["validityTime"]) != 300
        or int(metadata["Ni"]) != 1440
        or int(metadata["Nj"]) != 721
        or len(values) != 1_038_240
    ):
        raise RuntimeError(f"unexpected decoded GRIB identity: {metadata}")
    minimum = float(values.min())
    maximum = float(values.max())
    mean = float(values.mean())
    if not all(math.isfinite(x) for x in (minimum, maximum, mean)):
        raise RuntimeError("decoded values are not finite")
    return {
        "decoder": "python-eccodes-direct",
        "eccodes_version": getattr(eccodes, "__version__", "unknown"),
        "decode_success": True,
        "metadata": metadata,
        "value_count": len(values),
        "value_min": minimum,
        "value_max": maximum,
        "value_mean": mean,
        "elapsed_seconds": elapsed,
        "single_message_local_benchmark_only": True,
    }


def run(root: Path) -> dict[str, Any]:
    root = root.resolve()
    parent = root / "manifest_v2.json"
    previous = root / "audit" / "HPBL_CAUSAL_FEASIBILITY.json"
    raw = (
        root
        / "provenance"
        / "raw"
        / "f039"
        / "gfs.t12z.pgrb2.0p25.f039.HPBL_surface.grib2"
    )
    report_path = root / "audit" / "HPBL_ECCODES_DECODER_AUDIT.json"
    md_path = root / "audit" / "HPBL_ECCODES_DECODER_AUDIT.md"
    manifest_path = root / "manifest_v3.json"
    conflicts = [str(path) for path in (report_path, md_path, manifest_path) if path.exists()]
    if conflicts:
        raise FileExistsError(f"append-only v3 preflight failed: {conflicts}")
    if sha256_file(parent) != PARENT_V2_SHA256:
        raise RuntimeError("manifest_v2 identity changed")
    if sha256_file(previous) != PREVIOUS_FEASIBILITY_SHA256:
        raise RuntimeError("previous feasibility identity changed")
    if sha256_file(raw) != RAW_F039_SHA256:
        raise RuntimeError("f039 raw identity changed")

    decoded = decode_direct_eccodes(raw)
    previous_payload = json.loads(previous.read_text(encoding="utf-8"))
    report = {
        "artifact_type": "TRACK_A_DECODER_AUDIT_APPEND_ONLY_CORRECTION",
        "supersedes_only": [
            "audit/HPBL_CAUSAL_FEASIBILITY.json::decoder.operational_grib_decoder_available",
            "audit/HPBL_CAUSAL_FEASIBILITY.json::blockers[0]",
        ],
        "correction_reason": "The v2 audit required cfgrib+eccodes and failed to recognize that the installed eccodes Python bindings decode GRIB directly.",
        "previous_operational_decoder_available": False,
        "corrected_operational_decoder_available": True,
        "direct_decode": decoded,
        "raw_identity": identity(raw, root),
        "three_hour_verdict_previous": previous_payload["three_hour_verdict"],
        "three_hour_verdict_corrected": "STOP_3H_HONEST_MODEL",
        "verdict_unchanged_reason": [
            "Only three pilot messages have cutoff-verified provenance; archive-wide coverage is not proven.",
            "The observed range throughput implies about 47,692 seconds for HPBL payload alone before index/metadata/decode/model work.",
            "At least 78,840 requests are required for the three specified years.",
            "Incremental information remains NOT_RUN/UNDETERMINED.",
        ],
        "network_requests": 0,
        "downloaded_bytes": 0,
        "labels_read": False,
        "models_fit": 0,
        "submission_csv_created": False,
    }
    write_json_exclusive(report_path, report)
    write_text_exclusive(
        md_path,
        f"""# Direct ecCodes decoder correction

`eccodes` is installed and directly decoded the SHA-bound f039 HPBL GRIB message: {decoded['value_count']:,} regular latitude/longitude values ({decoded['metadata']['Ni']} x {decoded['metadata']['Nj']}) in {decoded['elapsed_seconds']:.3f} seconds. Therefore decoder availability is **TRUE**; `cfgrib` is not required for a direct ecCodes path.

The 3-hour verdict remains **`STOP_3H_HONEST_MODEL`**. The bottlenecks are provenance coverage, an observed ~47,692-second HPBL-only payload transfer extrapolation, at least 78,840 requests, and an unrun incremental-information test. This amendment made zero network requests, read no labels, fit no model and created no CSV submission.
""",
    )
    code = Path(__file__).resolve()
    test = REPO / "tests" / "test_noaa_gfs_track_a_decoder_v3.py"
    manifest = {
        "artifact_type": "TRACK_A_APPEND_ONLY_AMENDMENT",
        "schema_version": 3,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "parent_manifest": identity(parent, root),
        "previous_feasibility": identity(previous, root),
        "correction": identity(report_path, root),
        "correction_markdown": identity(md_path, root),
        "reproduction_code": identity(code),
        "test_code": identity(test),
        "existing_files_modified": [],
        "network_requests": 0,
        "downloaded_bytes": 0,
        "labels_read": False,
        "models_fit": 0,
        "submission_csv_created": False,
    }
    write_json_exclusive(manifest_path, manifest)
    return {
        "manifest_v3": identity(manifest_path, root),
        "decoder_correction": identity(report_path, root),
        "decoder_available": True,
        "three_hour_verdict": "STOP_3H_HONEST_MODEL",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(run(args.root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


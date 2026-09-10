#!/usr/bin/env python3
"""Fresh-namespace v7 wrapper around the frozen GEFS Stage1 model runner.

The scientific runner is byte-pinned and loaded with deterministic, audited
identifier-only substitutions.  No model, feature, label, fold, candidate,
weight, metric, or gate code is changed.  The substitutions isolate the newly
authorized v7 source/model/tombstone/quarantine namespaces.
"""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_RUNNER = PROJECT_ROOT / "scripts/run_noaa_gefs_operational_spread_00z_paired_increment_v2.py"
LEGACY_RUNNER_SHA = "8ec2f400a3f9e4e2b727f543d5582078ae89b648cd6a55df5085f600e4779aaa"
V6_PREREG = PROJECT_ROOT / "configs/noaa_gefs_operational_spread_00z_paired_increment_preregister_v6.json"
V7_PREREG = PROJECT_ROOT / "configs/noaa_gefs_operational_spread_00z_paired_increment_preregister_v7.json"
V8_PREREG = PROJECT_ROOT / "configs/noaa_gefs_operational_spread_00z_paired_increment_preregister_v8.json"
V9_PREREG = PROJECT_ROOT / "configs/noaa_gefs_operational_spread_00z_paired_increment_preregister_v9.json"
V6_SHA = "243b3c14b9bfdfa5e46eccd767727559aadab14b4aff1d9be9df4f3a66c1255c"
V2_FAILURE_SEAL = PROJECT_ROOT / "artifacts/incidents/noaa_gefs_operational_spread_00z_original_v2_stage1_protocol_failure_seal_v1.json"
V2_FAILURE_SEAL_SHA = "89d114289de2a7d3c6793c243955330cdb6285ea028ded9275ab258086592a88"
V7_SYNTHETIC_INCIDENT = PROJECT_ROOT / "artifacts/incidents/noaa_gefs_operational_spread_00z_v7_synthetic_reference_cleanup_v1.json"
FRESH_SOURCE = PROJECT_ROOT / "artifacts/external/noaa_gefs_operational_spread_00z_original_v7/stage1_source_through_operating_2023"
FRESH_STAGE2_SOURCE = PROJECT_ROOT / "artifacts/external/noaa_gefs_operational_spread_00z_original_v7/stage2_source_operating_2024"
FRESH_SOURCE_LOCK = PROJECT_ROOT / "artifacts/audits/noaa_gefs_operational_spread_00z_original_v7_stage1_launch_lock.json"
FRESH_MODEL_OUTPUT = PROJECT_ROOT / "artifacts/postgate/noaa_gefs_operational_spread_00z_paired_increment_strict_v7"
FRESH_MODEL_TOMBSTONE = PROJECT_ROOT / "artifacts/audits/noaa_gefs_operational_spread_00z_paired_increment_v7_stage1_single_attempt.json"
FRESH_EXTRACTOR = PROJECT_ROOT / "scripts/download_noaa_gefs_operational_spread_00z_original_v7.py"
FRESH_FOCUSED_TEST = PROJECT_ROOT / "tests/test_noaa_gefs_operational_spread_00z_paired_increment_v7.py"
FRESH_POSTRUN_TEST = PROJECT_ROOT / "tests/test_noaa_gefs_operational_spread_00z_paired_increment_v7_postrun.py"
LEGACY_DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/postgate/noaa_gefs_operational_spread_00z_paired_increment_strict_v2"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_frozen_core() -> types.ModuleType:
    if _sha256_file(LEGACY_RUNNER) != LEGACY_RUNNER_SHA:
        raise RuntimeError("legacy scientific runner bytes changed")
    source = LEGACY_RUNNER.read_text(encoding="utf-8")
    substitutions = (
        (
            '"noaa_gefs_operational_spread_00z_paired_increment_strict_v5"',
            '"noaa_gefs_operational_spread_00z_paired_increment_strict_v7"',
            2,
        ),
        (
            '"noaa_gefs_operational_spread_00z_paired_increment_v5_stage1_single_attempt"',
            '"noaa_gefs_operational_spread_00z_paired_increment_v7_stage1_single_attempt"',
            1,
        ),
        (
            '"noaa_gefs_operational_spread_00z_paired_increment_v5_partial_*"',
            '"noaa_gefs_operational_spread_00z_paired_increment_v7_partial_*"',
            1,
        ),
        (
            'f"noaa_gefs_operational_spread_00z_paired_increment_v5_partial_{os.getpid()}_{uuid.uuid4().hex}"',
            'f"noaa_gefs_operational_spread_00z_paired_increment_v7_partial_{os.getpid()}_{uuid.uuid4().hex}"',
            1,
        ),
        (
            'f"noaa_gefs_operational_spread_00z_paired_increment_v5_stage1_failure_{attempt_token}.json"',
            'f"noaa_gefs_operational_spread_00z_paired_increment_v7_stage1_failure_{attempt_token}.json"',
            1,
        ),
        (
            "for source in (BASE_PREREG, V2_PREREG, V3_PREREG, V4_PREREG, V5_PREREG, EXECUTION_PROTOCOL, INCIDENT, SOURCE_LAUNCH_LOCK, SOURCE_LAUNCH_LOCK_SHA):",
            "for source in (BASE_PREREG, V2_PREREG, V3_PREREG, V4_PREREG, V5_PREREG, V6_PREREG, V7_PREREG, V8_PREREG, V9_PREREG, V2_FAILURE_SEAL, V7_SYNTHETIC_INCIDENT, EXECUTION_PROTOCOL, INCIDENT, SOURCE_LAUNCH_LOCK, SOURCE_LAUNCH_LOCK_SHA):",
            1,
        ),
    )
    for old, new, expected_count in substitutions:
        count = source.count(old)
        if count != expected_count:
            raise RuntimeError(f"runner namespace substitution count differs: {old!r}: {count} != {expected_count}")
        source = source.replace(old, new)
    module = types.ModuleType("_noaa_gefs_v7_frozen_core")
    module.__file__ = str(Path(__file__).resolve())
    module.__package__ = "scripts"
    module.__dict__.update({
        "V6_PREREG": V6_PREREG,
        "V7_PREREG": V7_PREREG,
        "V8_PREREG": V8_PREREG,
        "V9_PREREG": V9_PREREG,
        "V2_FAILURE_SEAL": V2_FAILURE_SEAL,
        "V7_SYNTHETIC_INCIDENT": V7_SYNTHETIC_INCIDENT,
    })
    sys.modules[module.__name__] = module
    exec(compile(source, str(Path(__file__).resolve()), "exec"), module.__dict__)
    return module


core = _load_frozen_core()
core.SOURCE_ROOT = FRESH_SOURCE
core.STAGE2_SOURCE = FRESH_STAGE2_SOURCE
core.SOURCE_LAUNCH_LOCK = FRESH_SOURCE_LOCK
core.SOURCE_LAUNCH_LOCK_SHA = FRESH_SOURCE_LOCK.with_suffix(".json.sha256")
core.DEFAULT_OUTPUT = FRESH_MODEL_OUTPUT
core.MODEL_ATTEMPT_TOMBSTONE = FRESH_MODEL_TOMBSTONE
core.EXTRACTOR = FRESH_EXTRACTOR
core.FOCUSED_TEST = FRESH_FOCUSED_TEST
core.POSTRUN_TEST = FRESH_POSTRUN_TEST
core.V6_PREREG = V6_PREREG
core.V7_PREREG = V7_PREREG
core.V8_PREREG = V8_PREREG
core.V9_PREREG = V9_PREREG
core.V2_FAILURE_SEAL = V2_FAILURE_SEAL
core.V7_SYNTHETIC_INCIDENT = V7_SYNTHETIC_INCIDENT

_legacy_verify_effective = core.verify_effective_preregister
_legacy_closure_record = core._closure_record


def verify_effective_preregister() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    current_output = core.DEFAULT_OUTPUT
    core.DEFAULT_OUTPUT = LEGACY_DEFAULT_OUTPUT
    try:
        result = _legacy_verify_effective()
    finally:
        core.DEFAULT_OUTPUT = current_output
    if _sha256_file(V6_PREREG) != V6_SHA or _sha256_file(V2_FAILURE_SEAL) != V2_FAILURE_SEAL_SHA:
        raise AssertionError("v7 authorization/failure lineage differs")
    v6 = json.loads(V6_PREREG.read_text(encoding="utf-8"))
    v7 = json.loads(V7_PREREG.read_text(encoding="utf-8"))
    v8 = json.loads(V8_PREREG.read_text(encoding="utf-8"))
    v9 = json.loads(V9_PREREG.read_text(encoding="utf-8"))
    if v7.get("supersedes", {}).get("sha256") != V6_SHA:
        raise AssertionError("v7-v6 lineage differs")
    if v8.get("supersedes", {}).get("sha256") != _sha256_file(V7_PREREG):
        raise AssertionError("v8-v7 lineage differs")
    if v9.get("supersedes", {}).get("sha256") != _sha256_file(V8_PREREG):
        raise AssertionError("v9-v8 lineage differs")
    science = v7.get("immutable_scientific_inheritance", {})
    if science.get("candidate_id") != "gefs00z_0p50_10m_850_mean_spread_w025" or science.get("transfer_weight") != 0.25:
        raise AssertionError("v7 scientific candidate differs")
    if v7.get("fresh_namespaces", {}).get("model_output") != FRESH_MODEL_OUTPUT.relative_to(PROJECT_ROOT).as_posix():
        raise AssertionError("v7 model output path differs")
    return result


core.verify_effective_preregister = verify_effective_preregister


def verify_source_launch_lock() -> dict[str, Any]:
    path = core.SOURCE_LAUNCH_LOCK
    sidecar = core.SOURCE_LAUNCH_LOCK_SHA
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError("fresh v7 source launch lock/sidecar absent")
    expected = f"{core.sha256_file(path)}  {path.name}\n"
    if sidecar.read_text(encoding="ascii") != expected:
        raise AssertionError("fresh v7 source launch-lock sidecar differs")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("status") != "FROZEN_AFTER_V7_PREFLIGHT_BEFORE_FULL_STAGE1_RANGE_GET":
        raise AssertionError("fresh v7 source launch-lock status differs")
    if lock.get("scope", {}).get("operating_2024_values") != 0 or lock.get("preflight_payload_reuse") != 0:
        raise AssertionError("fresh v7 source scope/reuse differs")
    core._assert_records_current(lock["lineage"])
    core._assert_records_current(lock["source_roots"])
    core._assert_records_current(lock["recursive_local_imports"])
    runner_records = [record for record in lock["source_roots"] if Path(record["path"]).name == Path(__file__).name]
    if len(runner_records) != 1 or runner_records[0]["sha256"] != core.sha256_file(Path(__file__)):
        raise AssertionError("fresh v7 model runner was not pre-network bound")
    return lock


core.verify_source_launch_lock = verify_source_launch_lock


def verify_source_canonical() -> dict[str, Any]:
    if core.STAGE2_SOURCE.exists():
        raise AssertionError("fresh v7 conditional Stage2 source exists")
    manifest_path = core.SOURCE_ROOT / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("fresh v7 Stage1 source canonical absent")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS_V7_STAGE1_SOURCE_EXTRACTION":
        raise AssertionError("fresh v7 source did not pass")
    core._assert_records_current(manifest["source_closure"])
    core._assert_records_current(manifest["artifacts"])
    expected_nonmutation = {"stage2_namespace_created": False, "label_model_metric_csv_writes": 0, "old_v2_partial_reused": 0}
    if manifest.get("nonmutation") != expected_nonmutation:
        raise AssertionError("fresh v7 source nonmutation differs")
    return manifest


core.verify_source_canonical = verify_source_canonical


def closure_record() -> dict[str, Any]:
    record = _legacy_closure_record()
    closure = core.resolve_ast_local_import_closure((Path(__file__), FRESH_EXTRACTOR, LEGACY_RUNNER))
    record["resolved_relative_paths"] = [path.relative_to(PROJECT_ROOT).as_posix() for path in closure]
    record["resolved_files"] = [core.describe_file(path) for path in closure]
    record.update({
        "prereg_v6": core.describe_file(V6_PREREG),
        "prereg_v7": core.describe_file(V7_PREREG),
        "prereg_v8": core.describe_file(V8_PREREG),
        "prereg_v9": core.describe_file(V9_PREREG),
        "v2_protocol_failure_seal": core.describe_file(V2_FAILURE_SEAL),
        "v7_synthetic_reference_incident": core.describe_file(V7_SYNTHETIC_INCIDENT),
        "frozen_legacy_scientific_runner": core.describe_file(LEGACY_RUNNER),
        "namespace_transform": {
            "kind": "identifier-only deterministic source transform",
            "legacy_runner_sha256": LEGACY_RUNNER_SHA,
            "scientific_code_changes": 0,
        },
    })
    return record


core._closure_record = closure_record


def static_audit() -> dict[str, Any]:
    result = core.static_audit()
    result.update({
        "status": "PASS_ACTUAL_STAGE1_RUNNER_STATIC_CONTRACT_V7_FRESH_NAMESPACE",
        "config_v6_sha256": _sha256_file(V6_PREREG),
        "config_v7_sha256": _sha256_file(V7_PREREG),
        "config_v8_sha256": _sha256_file(V8_PREREG),
        "config_v9_sha256": _sha256_file(V9_PREREG),
        "legacy_scientific_runner_sha256": LEGACY_RUNNER_SHA,
        "scientific_code_changes": 0,
        "fresh_source": FRESH_SOURCE.relative_to(PROJECT_ROOT).as_posix(),
        "fresh_output": FRESH_MODEL_OUTPUT.relative_to(PROJECT_ROOT).as_posix(),
        "old_v2_partial_reuse": 0,
    })
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = core.parse_args(argv)
    if args.static_audit == args.run_stage1:
        raise ValueError("choose exactly one of --static-audit or --run-stage1")
    if args.static_audit:
        print(json.dumps(static_audit(), sort_keys=True))
        return 0
    if args.raw_dir.resolve() != Path(r"data/local/open").resolve():
        raise AssertionError("fresh v7 raw input root differs")
    if args.out_dir.resolve() != FRESH_MODEL_OUTPUT.resolve():
        raise AssertionError("fresh v7 model output differs")
    core.run_stage1(args.raw_dir, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

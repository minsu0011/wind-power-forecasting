#!/usr/bin/env python3
"""Fresh v12 Stage1 model wrapper around the frozen GEFS scientific runner.

Only experiment/path identifiers and source-acceptance lineage differ.  The
frozen feature, model, fold, label-mask, candidate, metric, and gate code is
loaded byte-for-byte and transformed with audited exact-count substitutions.
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
LEGACY_DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/postgate/noaa_gefs_operational_spread_00z_paired_increment_strict_v2"

V6_PREREG = PROJECT_ROOT / "configs/noaa_gefs_operational_spread_00z_paired_increment_preregister_v6.json"
V7_PREREG = PROJECT_ROOT / "configs/noaa_gefs_operational_spread_00z_paired_increment_preregister_v7.json"
V8_PREREG = PROJECT_ROOT / "configs/noaa_gefs_operational_spread_00z_paired_increment_preregister_v8.json"
V9_PREREG = PROJECT_ROOT / "configs/noaa_gefs_operational_spread_00z_paired_increment_preregister_v9.json"
V10_PREREG = PROJECT_ROOT / "configs/noaa_gefs_operational_spread_00z_paired_increment_preregister_v10.json"
V11_PREREG = PROJECT_ROOT / "configs/noaa_gefs_operational_spread_00z_paired_increment_preregister_v11.json"
V12_PREREG = PROJECT_ROOT / "configs/noaa_gefs_operational_spread_00z_paired_increment_preregister_v12.json"
MODEL_EXEC_V1 = PROJECT_ROOT / "configs/noaa_gefs_operational_spread_00z_stage1_model_execution_preregister_v1.json"
MODEL_EXEC_V2 = PROJECT_ROOT / "configs/noaa_gefs_operational_spread_00z_stage1_model_execution_preregister_v2.json"

V6_SHA = "243b3c14b9bfdfa5e46eccd767727559aadab14b4aff1d9be9df4f3a66c1255c"
V10_SHA = "b154793821dcebd9e859605006cbecf139c3753f4bab79693958bba90092e70d"
V11_SHA = "270c2157fe32f869036e4f7f910dc9897b327931a80d914b056487cfca596146"
V12_SHA = "25e0e69dc0abec844e4668ec4d88a9810af93360a925d80f240e79ea8f0b3901"
MODEL_EXEC_V1_SHA = "692db2e44d6df4a5ecbed210c7f08870bd997c51c9c58451742da17e6a86dfbc"
MODEL_EXEC_V2_SHA = "5b3f4cc1dcfb1e3ae1cc0c86e8c40e407f47971ef1a5aa05c16bf554b74cb082"

V2_FAILURE_SEAL = PROJECT_ROOT / "artifacts/incidents/noaa_gefs_operational_spread_00z_original_v2_stage1_protocol_failure_seal_v1.json"
V7_SYNTHETIC_INCIDENT = PROJECT_ROOT / "artifacts/incidents/noaa_gefs_operational_spread_00z_v7_synthetic_reference_cleanup_v1.json"
V7_FAILURE_SEAL = PROJECT_ROOT / "artifacts/incidents/noaa_gefs_operational_spread_00z_original_v7_stage1_protocol_failure_seal_v1.json"
V10_AUTHORIZATION = PROJECT_ROOT / "artifacts/incidents/noaa_gefs_operational_spread_00z_original_v10_user_authorized_source_boundary_v1.json"
V10_FAILURE_SEAL = PROJECT_ROOT / "artifacts/incidents/noaa_gefs_operational_spread_00z_original_v10_stage1_protocol_failure_seal_v1.json"
V12_AUTHORIZATION = PROJECT_ROOT / "artifacts/incidents/noaa_gefs_operational_spread_00z_original_v12_user_authorized_semantic_boundary_v1.json"

FRESH_SOURCE = PROJECT_ROOT / "artifacts/external/noaa_gefs_operational_spread_00z_original_v12/stage1_source_through_operating_2023"
FRESH_STAGE2_SOURCE = PROJECT_ROOT / "artifacts/external/noaa_gefs_operational_spread_00z_original_v12/stage2_source_operating_2024"
FRESH_SOURCE_LOCK = PROJECT_ROOT / "artifacts/audits/noaa_gefs_operational_spread_00z_original_v12_stage1_launch_lock.json"
SOURCE_POSTRUN_REVIEW = PROJECT_ROOT / "artifacts/audits/noaa_gefs_operational_spread_00z_original_v12_stage1_source_postrun_independent_review_v1.json"
SOURCE_POSTRUN_REVIEW_SHA = "eb06daea02998138b8eb347ebdeccbbbd8ca7e8193952e5b8e425fd7a97faeab"
SOURCE_MANIFEST_SHA = "24174dfa9658d72551c31efb13ab981ca926d52cd3bd1f5c3209746555c7fe77"
MODEL_PRELAUNCH_READINESS = PROJECT_ROOT / "artifacts/audits/noaa_gefs_operational_spread_00z_paired_increment_v12_stage1_model_prelaunch_readiness_v1.json"
MODEL_PRELAUNCH_REVIEW = PROJECT_ROOT / "artifacts/audits/noaa_gefs_operational_spread_00z_paired_increment_v12_stage1_model_prelaunch_independent_review_v1.json"
FRESH_MODEL_OUTPUT = PROJECT_ROOT / "artifacts/postgate/noaa_gefs_operational_spread_00z_paired_increment_strict_v12"
FRESH_MODEL_TOMBSTONE = PROJECT_ROOT / "artifacts/audits/noaa_gefs_operational_spread_00z_paired_increment_v12_stage1_single_attempt.json"
FRESH_EXTRACTOR = PROJECT_ROOT / "scripts/download_noaa_gefs_operational_spread_00z_original_v12.py"
FRESH_FOCUSED_TEST = PROJECT_ROOT / "tests/test_noaa_gefs_operational_spread_00z_paired_increment_v12_model.py"
FRESH_POSTRUN_TEST = PROJECT_ROOT / "tests/test_noaa_gefs_operational_spread_00z_paired_increment_v12_model_postrun.py"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_frozen_core() -> types.ModuleType:
    if LEGACY_RUNNER.stat().st_size != 55_799 or _sha256_file(LEGACY_RUNNER) != LEGACY_RUNNER_SHA:
        raise RuntimeError("legacy scientific runner identity changed")
    source = LEGACY_RUNNER.read_text(encoding="utf-8")
    substitutions = (
        (
            '"noaa_gefs_operational_spread_00z_paired_increment_strict_v5"',
            '"noaa_gefs_operational_spread_00z_paired_increment_strict_v12"',
            2,
        ),
        (
            '"noaa_gefs_operational_spread_00z_paired_increment_v5_stage1_single_attempt"',
            '"noaa_gefs_operational_spread_00z_paired_increment_v12_stage1_single_attempt"',
            1,
        ),
        (
            '"noaa_gefs_operational_spread_00z_paired_increment_v5_partial_*"',
            '"noaa_gefs_operational_spread_00z_paired_increment_v12_partial_*"',
            1,
        ),
        (
            'f"noaa_gefs_operational_spread_00z_paired_increment_v5_partial_{os.getpid()}_{uuid.uuid4().hex}"',
            'f"noaa_gefs_operational_spread_00z_paired_increment_v12_partial_{os.getpid()}_{uuid.uuid4().hex}"',
            1,
        ),
        (
            'f"noaa_gefs_operational_spread_00z_paired_increment_v5_stage1_failure_{attempt_token}.json"',
            'f"noaa_gefs_operational_spread_00z_paired_increment_v12_stage1_failure_{attempt_token}.json"',
            1,
        ),
        (
            "for source in (BASE_PREREG, V2_PREREG, V3_PREREG, V4_PREREG, V5_PREREG, EXECUTION_PROTOCOL, INCIDENT, SOURCE_LAUNCH_LOCK, SOURCE_LAUNCH_LOCK_SHA):",
            "for source in (BASE_PREREG, V2_PREREG, V3_PREREG, V4_PREREG, V5_PREREG, V6_PREREG, V7_PREREG, V8_PREREG, V9_PREREG, V10_PREREG, V11_PREREG, V12_PREREG, MODEL_EXEC_V1, MODEL_EXEC_V2, V2_FAILURE_SEAL, V7_SYNTHETIC_INCIDENT, V7_FAILURE_SEAL, V10_AUTHORIZATION, V10_FAILURE_SEAL, V12_AUTHORIZATION, SOURCE_POSTRUN_REVIEW, MODEL_PRELAUNCH_READINESS, MODEL_PRELAUNCH_REVIEW, EXECUTION_PROTOCOL, INCIDENT, SOURCE_LAUNCH_LOCK, SOURCE_LAUNCH_LOCK_SHA):",
            1,
        ),
    )
    for old, new, expected_count in substitutions:
        count = source.count(old)
        if count != expected_count:
            raise RuntimeError(f"runner substitution count differs: {old!r}: {count} != {expected_count}")
        source = source.replace(old, new)
    module = types.ModuleType("_noaa_gefs_v12_frozen_scientific_core")
    module.__file__ = str(Path(__file__).resolve())
    module.__package__ = "scripts"
    module.__dict__.update(
        {
            "V6_PREREG": V6_PREREG,
            "V7_PREREG": V7_PREREG,
            "V8_PREREG": V8_PREREG,
            "V9_PREREG": V9_PREREG,
            "V10_PREREG": V10_PREREG,
            "V11_PREREG": V11_PREREG,
            "V12_PREREG": V12_PREREG,
            "MODEL_EXEC_V1": MODEL_EXEC_V1,
            "MODEL_EXEC_V2": MODEL_EXEC_V2,
            "V2_FAILURE_SEAL": V2_FAILURE_SEAL,
            "V7_SYNTHETIC_INCIDENT": V7_SYNTHETIC_INCIDENT,
            "V7_FAILURE_SEAL": V7_FAILURE_SEAL,
            "V10_AUTHORIZATION": V10_AUTHORIZATION,
            "V10_FAILURE_SEAL": V10_FAILURE_SEAL,
            "V12_AUTHORIZATION": V12_AUTHORIZATION,
            "SOURCE_POSTRUN_REVIEW": SOURCE_POSTRUN_REVIEW,
            "MODEL_PRELAUNCH_READINESS": MODEL_PRELAUNCH_READINESS,
            "MODEL_PRELAUNCH_REVIEW": MODEL_PRELAUNCH_REVIEW,
        }
    )
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
for name, value in {
    "V6_PREREG": V6_PREREG,
    "V7_PREREG": V7_PREREG,
    "V8_PREREG": V8_PREREG,
    "V9_PREREG": V9_PREREG,
    "V10_PREREG": V10_PREREG,
    "V11_PREREG": V11_PREREG,
    "V12_PREREG": V12_PREREG,
    "MODEL_EXEC_V1": MODEL_EXEC_V1,
    "MODEL_EXEC_V2": MODEL_EXEC_V2,
    "V2_FAILURE_SEAL": V2_FAILURE_SEAL,
    "V7_SYNTHETIC_INCIDENT": V7_SYNTHETIC_INCIDENT,
    "V7_FAILURE_SEAL": V7_FAILURE_SEAL,
    "V10_AUTHORIZATION": V10_AUTHORIZATION,
    "V10_FAILURE_SEAL": V10_FAILURE_SEAL,
    "V12_AUTHORIZATION": V12_AUTHORIZATION,
    "SOURCE_POSTRUN_REVIEW": SOURCE_POSTRUN_REVIEW,
    "MODEL_PRELAUNCH_READINESS": MODEL_PRELAUNCH_READINESS,
    "MODEL_PRELAUNCH_REVIEW": MODEL_PRELAUNCH_REVIEW,
}.items():
    setattr(core, name, value)

_legacy_verify_effective = core.verify_effective_preregister
_legacy_closure_record = core._closure_record


def verify_effective_preregister() -> tuple[dict[str, Any], ...]:
    current_output = core.DEFAULT_OUTPUT
    core.DEFAULT_OUTPUT = LEGACY_DEFAULT_OUTPUT
    try:
        result = _legacy_verify_effective()
    finally:
        core.DEFAULT_OUTPUT = current_output
    files = (V6_PREREG, V7_PREREG, V8_PREREG, V9_PREREG, V10_PREREG, V11_PREREG, V12_PREREG)
    docs = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    if _sha256_file(V6_PREREG) != V6_SHA or _sha256_file(V10_PREREG) != V10_SHA:
        raise AssertionError("v6/v10 prereg identity changed")
    if _sha256_file(V11_PREREG) != V11_SHA or _sha256_file(V12_PREREG) != V12_SHA:
        raise AssertionError("v11/v12 prereg identity changed")
    for previous, current in zip(files, files[1:]):
        if json.loads(current.read_text(encoding="utf-8"))["supersedes"]["sha256"] != _sha256_file(previous):
            raise AssertionError(f"scientific lineage differs at {current.name}")
    if _sha256_file(MODEL_EXEC_V1) != MODEL_EXEC_V1_SHA or _sha256_file(MODEL_EXEC_V2) != MODEL_EXEC_V2_SHA:
        raise AssertionError("model execution prereg identity changed")
    execution_v1 = json.loads(MODEL_EXEC_V1.read_text(encoding="utf-8"))
    execution_v2 = json.loads(MODEL_EXEC_V2.read_text(encoding="utf-8"))
    if execution_v2["supersedes"]["sha256"] != MODEL_EXEC_V1_SHA:
        raise AssertionError("model execution v2-v1 lineage differs")
    science = docs[-1]["immutable_scientific_inheritance"]
    if science["candidate_id"] != "gefs00z_0p50_10m_850_mean_spread_w025" or science["blend_weight"] != 0.25:
        raise AssertionError("v12 scientific candidate differs")
    expected_output = execution_v1["fresh_model_namespace"]["output"]
    if expected_output != FRESH_MODEL_OUTPUT.relative_to(PROJECT_ROOT).as_posix():
        raise AssertionError("v12 model output path differs")
    return result


core.verify_effective_preregister = verify_effective_preregister


def verify_source_launch_lock() -> dict[str, Any]:
    path = core.SOURCE_LAUNCH_LOCK
    sidecar = core.SOURCE_LAUNCH_LOCK_SHA
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError("v12 source launch lock or sidecar absent")
    expected = f"{core.sha256_file(path)}  {path.name}\n"
    if sidecar.read_text(encoding="ascii") != expected:
        raise AssertionError("v12 source launch-lock sidecar differs")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("status") != "FROZEN_V12_AFTER_FOUR_CELL_POSTRUN_INDEPENDENT_REVIEW_BEFORE_FULL_STAGE1_RANGE_GET":
        raise AssertionError("v12 source launch-lock status differs")
    if lock.get("plan_sha256") != "501ccfeb34c1cd5e057b76a195315f6da3ffd49632b368e4ab578d51afdf8ff1":
        raise AssertionError("v12 source plan differs")
    scope = lock.get("scope", {})
    if scope.get("operating_2024_values") != 0 or scope.get("2025_requests_or_values") != 0:
        raise AssertionError("v12 source future scope differs")
    if lock.get("preflight_payload_reuse") != 0 or lock.get("terminal_v7_v10_partial_files_opened_hashed_copied_or_reused") != 0:
        raise AssertionError("v12 source reuse contract differs")
    core._assert_records_current(lock["lineage"])
    core._assert_records_current(lock["source_roots"])
    core._assert_records_current(lock["recursive_local_imports"])
    return lock


core.verify_source_launch_lock = verify_source_launch_lock


def verify_source_canonical() -> dict[str, Any]:
    if core.STAGE2_SOURCE.exists():
        raise AssertionError("v12 conditional Stage2 source exists")
    manifest_path = core.SOURCE_ROOT / "manifest.json"
    if not manifest_path.is_file() or _sha256_file(manifest_path) != SOURCE_MANIFEST_SHA:
        raise FileNotFoundError("accepted v12 Stage1 source manifest identity differs")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS_V12_STAGE1_SOURCE_EXTRACTION":
        raise AssertionError("v12 source did not pass")
    core._assert_records_current(manifest["source_closure"])
    core._assert_records_current(manifest["artifacts"])
    expected_nonmutation = {
        "label_model_metric_csv_writes": 0,
        "old_v2_v7_v10_partial_reused": 0,
        "semantic_preflight_payload_reused": 0,
        "stage2_namespace_created": False,
    }
    if manifest.get("nonmutation") != expected_nonmutation:
        raise AssertionError("v12 source nonmutation differs")
    if SOURCE_POSTRUN_REVIEW.stat().st_size != 10_728 or _sha256_file(SOURCE_POSTRUN_REVIEW) != SOURCE_POSTRUN_REVIEW_SHA:
        raise AssertionError("v12 source independent review identity differs")
    review = json.loads(SOURCE_POSTRUN_REVIEW.read_text(encoding="utf-8"))
    if review.get("verdict") != "PASS" or review.get("canonical", {}).get("manifest", {}).get("sha256") != SOURCE_MANIFEST_SHA:
        raise AssertionError("v12 source independent review verdict/binding differs")
    return manifest


core.verify_source_canonical = verify_source_canonical


def verify_model_prelaunch_review() -> dict[str, Any]:
    if not MODEL_PRELAUNCH_READINESS.is_file() or not MODEL_PRELAUNCH_REVIEW.is_file():
        raise FileNotFoundError("independent v12 model prelaunch readiness/review absent")
    review = json.loads(MODEL_PRELAUNCH_REVIEW.read_text(encoding="utf-8"))
    if review.get("verdict") != "PASS" or review.get("label_fit_prediction_score_2024_2025_access") != 0:
        raise AssertionError("independent v12 model prelaunch verdict/access differs")
    expected = {
        "bound_readiness": core.describe_file(MODEL_PRELAUNCH_READINESS),
        "bound_runner": core.describe_file(Path(__file__)),
        "bound_focused_test": core.describe_file(FRESH_FOCUSED_TEST),
        "bound_postrun_test": core.describe_file(FRESH_POSTRUN_TEST),
    }
    for key, record in expected.items():
        if review.get(key) != record:
            raise AssertionError(f"independent v12 model prelaunch binding differs: {key}")
    if review.get("source_manifest_sha256") != SOURCE_MANIFEST_SHA or review.get("source_postrun_review_sha256") != SOURCE_POSTRUN_REVIEW_SHA:
        raise AssertionError("independent v12 model prelaunch source binding differs")
    return review


def closure_record() -> dict[str, Any]:
    record = _legacy_closure_record()
    closure = core.resolve_ast_local_import_closure((Path(__file__), FRESH_EXTRACTOR, LEGACY_RUNNER))
    record["resolved_relative_paths"] = [path.relative_to(PROJECT_ROOT).as_posix() for path in closure]
    record["resolved_files"] = [core.describe_file(path) for path in closure]
    record.update(
        {
            "prereg_v6": core.describe_file(V6_PREREG),
            "prereg_v7": core.describe_file(V7_PREREG),
            "prereg_v8": core.describe_file(V8_PREREG),
            "prereg_v9": core.describe_file(V9_PREREG),
            "prereg_v10": core.describe_file(V10_PREREG),
            "prereg_v11": core.describe_file(V11_PREREG),
            "prereg_v12": core.describe_file(V12_PREREG),
            "model_execution_prereg_v1": core.describe_file(MODEL_EXEC_V1),
            "model_execution_prereg_v2": core.describe_file(MODEL_EXEC_V2),
            "v2_protocol_failure_seal": core.describe_file(V2_FAILURE_SEAL),
            "v7_synthetic_reference_incident": core.describe_file(V7_SYNTHETIC_INCIDENT),
            "v7_terminal_source_failure_seal": core.describe_file(V7_FAILURE_SEAL),
            "v10_user_authorization": core.describe_file(V10_AUTHORIZATION),
            "v10_terminal_source_failure_seal": core.describe_file(V10_FAILURE_SEAL),
            "v12_user_authorization": core.describe_file(V12_AUTHORIZATION),
            "v12_source_postrun_independent_review": core.describe_file(SOURCE_POSTRUN_REVIEW),
            "model_prelaunch_readiness": core.describe_file(MODEL_PRELAUNCH_READINESS),
            "model_prelaunch_independent_review": core.describe_file(MODEL_PRELAUNCH_REVIEW),
            "frozen_legacy_scientific_runner": core.describe_file(LEGACY_RUNNER),
            "namespace_transform": {
                "kind": "identifier-only deterministic source transform",
                "legacy_runner_sha256": LEGACY_RUNNER_SHA,
                "scientific_code_changes": 0,
            },
        }
    )
    return record


core._closure_record = closure_record


def static_audit() -> dict[str, Any]:
    result = core.static_audit()
    verify_source_launch_lock()
    verify_source_canonical()
    quarantine = PROJECT_ROOT / "artifacts/quarantine"
    matching_quarantine = list(quarantine.glob("noaa_gefs_operational_spread_00z_paired_increment_v12_partial_*")) if quarantine.exists() else []
    if FRESH_MODEL_OUTPUT.exists() or FRESH_MODEL_TOMBSTONE.exists() or matching_quarantine:
        raise AssertionError("v12 model zero-state differs")
    result.update(
        {
            "status": "PASS_ACTUAL_STAGE1_RUNNER_STATIC_CONTRACT_V12_ACCEPTED_SOURCE",
            "config_v11_sha256": _sha256_file(V11_PREREG),
            "config_v12_sha256": _sha256_file(V12_PREREG),
            "model_execution_prereg_v1_sha256": _sha256_file(MODEL_EXEC_V1),
            "model_execution_prereg_v2_sha256": _sha256_file(MODEL_EXEC_V2),
            "legacy_scientific_runner_sha256": LEGACY_RUNNER_SHA,
            "scientific_code_changes": 0,
            "fresh_source_manifest_sha256": SOURCE_MANIFEST_SHA,
            "source_postrun_independent_review_sha256": SOURCE_POSTRUN_REVIEW_SHA,
            "fresh_source": FRESH_SOURCE.relative_to(PROJECT_ROOT).as_posix(),
            "fresh_output": FRESH_MODEL_OUTPUT.relative_to(PROJECT_ROOT).as_posix(),
            "model_output_exists": False,
            "model_tombstone_exists": False,
            "matching_model_quarantine_count": 0,
            "model_prelaunch_review_exists": MODEL_PRELAUNCH_REVIEW.exists(),
            "label_fit_prediction_score_2024_2025_access": 0,
        }
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = core.parse_args(argv)
    if args.static_audit == args.run_stage1:
        raise ValueError("choose exactly one of --static-audit or --run-stage1")
    if args.static_audit:
        print(json.dumps(static_audit(), sort_keys=True))
        return 0
    if args.raw_dir.resolve() != Path(r"data/local/open").resolve():
        raise AssertionError("v12 raw input root differs")
    if args.out_dir.resolve() != FRESH_MODEL_OUTPUT.resolve():
        raise AssertionError("v12 model output differs")
    verify_model_prelaunch_review()
    core.run_stage1(args.raw_dir, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
